"""
Pokémon Yellow Gym-style environment wrapper for RL training.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any
import time
import os

# Try to import pyboy
try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("PyBoy not available. Please install with: pip install pyboy")

class PokemonYellowEnv(gym.Env):
    """
    Gym-style environment for Pokémon Yellow RL training using PyBoy emulator.
    """
    
    def __init__(self, phase: int = 1, reward_shaper=None):
        super(PokemonYellowEnv, self).__init__()
        
        # Define action space (discrete buttons)
        self.action_space = spaces.Discrete(12)  # 12 possible actions
        self._held_buttons = set()
        self._action_handler = None  # init lazily

        # Define observation space (WRAM slice used by Pokémon Yellow: 0xD000..0xDFFF)
        # WRAM window here is 0x1000 bytes (4KB) and memory_map offsets are relative to 0xD000
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(43,), dtype=np.float32)
        
        # Emulator and state management
        self.emulator_path = os.getenv("EMULATOR_PATH") or "pokemon_yellow.gb"
        self.save_state_path = os.getenv("SAVE_STATE_PATH") or "pokemon_yellow.gbc.state"
        self.emulator = None
        self.current_state = None
        self.window = os.getenv("WINDOW") or "null"
        self.shouldRender = self.window != "null"
        # Debug toggles (use NUM_ENVS=1 when enabling to avoid log spam)
        self.debug_pos = os.getenv("DEBUG_POS", "false").lower() == "true"
        try:
            self.debug_pos_every = int(os.getenv("DEBUG_POS_EVERY", "250"))
        except Exception:
            self.debug_pos_every = 250
        self._debug_last_printed = None
        # Training parameters
        self.frame_skip = 4
        self.action_repeat = 4
        self.sticky_buttons = True
        
        # Episode tracking
        self.episode_steps = 0
        self.total_tiles_visited = 0
        self.tiles_visited = set()
        self.last_position = None

        # Battle transition tracking (for run-away detection)
        self._prev_in_battle_env = False

        # Action tracking for overlay
        self.action_counts = {}
        self.total_actions = 0
        
        # Reward shaping (phase-based)
        if reward_shaper is None:
            from rewards import PhaseRewardShaper
            self.reward_shaper = PhaseRewardShaper(phase=phase)
        else:
            self.reward_shaper = reward_shaper
            
        # Initialize emulator
        self._init_emulator()
        
    def _init_emulator(self):
        """Initialize the PyBoy emulator."""
        if not PYBOY_AVAILABLE:
            raise ImportError("PyBoy is not available. Please install with: pip install pyboy")
            
        # Check if ROM file exists
        if not os.path.exists(self.emulator_path):
            raise FileNotFoundError(f"ROM file not found: {self.emulator_path}")
            
        # Initialize PyBoy emulator
        self.emulator = PyBoy(
            self.emulator_path,
            cgb=True,
            window=self.window,  # Use null window mode for training (replacing deprecated "headless")
            sound_emulated=os.getenv("SOUND_EMULATED", "false").lower() == "true",
        )
        
        # Disable the boot screen to speed up initialization
        self.emulator.set_emulation_speed(1)
        
        # Load save state if provided
        if self.save_state_path and os.path.exists(self.save_state_path):
            self._load_state(self.save_state_path)
        else:
            # Load default save state if available
            default_state = "states/pallet_town.state"
            if os.path.exists(default_state):
                self._load_state(default_state)
        
    def reset(self, seed=None) -> Tuple[np.ndarray, dict]:
        """Reset environment and return initial observation."""
        # Clear any sticky-held inputs from a prior episode/session
        try:
            self._release_all_buttons()
        except Exception:
            self._held_buttons = set()
        # Load save state
        if self.save_state_path and os.path.exists(self.save_state_path):
            self._load_state(self.save_state_path)
        else:
            # Load default save state
            default_state = "states/pallet_town.state"
            if os.path.exists(default_state):
                self._load_state(default_state)
            else:
                # If no save state, reset emulator
                self.emulator.tick()
                
        # Get initial observation
        obs = self._get_observation()
        self.episode_steps = 0
        self.tiles_visited = set()
        self.last_position = None
        self._prev_in_battle_env = False
        
        try:
            self.reward_shaper.reset(seed=seed)
        except Exception:
            pass

        # Return observation and info dictionary (required by Gymnasium)
        return obs, {}
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return (observation, reward, terminated, truncated, info)."""
        # Press buttons
        self._press_buttons(action)
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        # Use full WRAM (0xC000..0xDFFF) so reward code can read Cxxx flags (e.g., wBattleResult)
        ram = self._get_wram_c000_dfff()
        # ---- DEBUG: compare multiple candidate position decodes (Yellow offset drift) ----
        pos_candidates = None
        if self.debug_pos:
            try:
                from memory_map import decode_player_position
                pos_candidates = decode_player_position(ram)
            except Exception:
                pos_candidates = None
        info_for_reward = {'episode_steps': self.episode_steps}

        # Add battle transition metadata for Option B (env computes it, reward uses it)
        try:
            from memory_map import decode_in_battle, decode_battle_result, decode_escaped_from_battle
            curr_in_battle = bool(decode_in_battle(ram))
            battle_ended = bool(self._prev_in_battle_env and (not curr_in_battle))

            battle_result = -1
            battle_result_known = False
            if battle_ended:
                br = int(decode_battle_result(ram))
                if br in (0, 1, 2):
                    battle_result = br
                    battle_result_known = True

            escaped_from_battle = bool(decode_escaped_from_battle(ram))

            # Only treat as ran_away if we *know* it was draw (2) OR escaped flag set
            ran_away = bool(
                battle_ended and (
                    escaped_from_battle or (battle_result_known and battle_result == 2)
                )
            )

            info_for_reward.update({
                'in_battle': curr_in_battle,
                'battle_ended': battle_ended,
                'battle_result': battle_result,
                'battle_result_known': battle_result_known,
                'escaped_from_battle': escaped_from_battle,
                'ran_away': ran_away,
            })

            self._prev_in_battle_env = curr_in_battle

        except Exception:
            pass

        try:
            reward = float(self.reward_shaper.compute_reward(ram, action, info_for_reward))
        except Exception:
            reward = -0.001
        
        # Check if episode is done
        done = self._is_done()
        
        # Update episode tracking
        self.episode_steps += 1
        
        # Update action counts for overlay
        if hasattr(self, 'action_counts'):
            try:
                from actions import ACTIONS
                action_name = ACTIONS.get(int(action), "UNKNOWN")
                self.action_counts[action_name] = self.action_counts.get(action_name, 0) + 1
                self.total_actions += 1
            except Exception:
                pass
        
        # Info dictionary
        try:
            from memory_map import decode_basic_game_info
            basic = decode_basic_game_info(ram)
            map_id = basic.get("map_id", 0)
            x = basic.get("x", 0)
            y = basic.get("y", 0)
            in_battle = bool(basic.get("in_battle", False))
            badges = int(basic.get("badges", 0))
            money = int(basic.get("money", 0))
            # If debug candidates exist, prefer them for display so you can see drift
            if isinstance(pos_candidates, dict):
                cur = pos_candidates.get('current', {})
                p1 = pos_candidates.get('plus1', {})
                m1 = pos_candidates.get('minus1', {})
        except Exception:
            map_id, x, y = 0, 0, 0
            in_battle = False
            badges = 0
            money = 0
            cur = {}
            p1 = {}
            m1 = {}

        info = {
            'episode_steps': self.episode_steps,
            'tiles_visited': len(getattr(self.reward_shaper, 'tiles_visited', set())),
            'total_tiles_visited': int(getattr(self.reward_shaper, 'total_tiles_visited', 0)),
            'battles_won': int(getattr(self.reward_shaper, 'previous_battle_wins', 0)),
            'battles_lost': int(getattr(self.reward_shaper, 'previous_battle_losses', 0)),
            'badges': badges,
            'money': money,
            'in_battle': in_battle,
            'map_id': int(map_id),
            'x': int(x),
            'y': int(y),
            'reward': float(reward),
            # Debug: candidate positions (helps diagnose off-by-one RAM mapping)
            'pos_cur_map_id': int(cur.get('map_id', 0)) if isinstance(pos_candidates, dict) else int(map_id),
            'pos_cur_x': int(cur.get('x', 0)) if isinstance(pos_candidates, dict) else int(x),
            'pos_cur_y': int(cur.get('y', 0)) if isinstance(pos_candidates, dict) else int(y),

            'pos_plus1_map_id': int(p1.get('map_id', 0)) if isinstance(pos_candidates, dict) else 0,
            'pos_plus1_x': int(p1.get('x', 0)) if isinstance(pos_candidates, dict) else 0,
            'pos_plus1_y': int(p1.get('y', 0)) if isinstance(pos_candidates, dict) else 0,

            'pos_minus1_map_id': int(m1.get('map_id', 0)) if isinstance(pos_candidates, dict) else 0,
            'pos_minus1_x': int(m1.get('x', 0)) if isinstance(pos_candidates, dict) else 0,
            'pos_minus1_y': int(m1.get('y', 0)) if isinstance(pos_candidates, dict) else 0,
        }
        
        # In Gymnasium, step returns (observation, reward, terminated, truncated, info)
        # For this environment, we'll set truncated=False and terminated=done
        if self.debug_pos and (self.episode_steps % self.debug_pos_every == 0):
            try:
                from actions import ACTIONS
                an = ACTIONS.get(int(action), str(action))
            except Exception:
                an = str(action)

            if isinstance(pos_candidates, dict):
                cur = pos_candidates.get('current', {})
                p1 = pos_candidates.get('plus1', {})
                m1 = pos_candidates.get('minus1', {})
                msg = (
                    f"[DEBUG_POS step={self.episode_steps}] action={an} "
                    f"basic=({int(map_id)},{int(x)},{int(y)}) "
                    f"cur=({cur.get('map_id')},{cur.get('x')},{cur.get('y')}) raw={cur.get('raw')} "
                    f"+1=({p1.get('map_id')},{p1.get('x')},{p1.get('y')}) raw={p1.get('raw')} "
                    f"-1=({m1.get('map_id')},{m1.get('x')},{m1.get('y')}) raw={m1.get('raw')}"
                )
            else:
                msg = f"[DEBUG_POS step={self.episode_steps}] action={an} basic=({int(map_id)},{int(x)},{int(y)})"

            # Avoid printing identical lines forever
            if msg != self._debug_last_printed:
                print(msg)
                self._debug_last_printed = msg
        return obs, reward, done, False, info
        
    def _press_buttons(self, action: int):
        """Apply one agent action with press/hold/release timing."""
        if self.emulator is None:
            return

        from actions import ActionHandler
        if self._action_handler is None:
            self._action_handler = ActionHandler()

        button_sequence = self._action_handler.get_button_sequence(action)

        button_map = {
            'UP': (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
            'DOWN': (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
            'LEFT': (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
            'RIGHT': (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
            'A': (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
            'B': (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
            'SELECT': (WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT),
            'START': (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
        }

        desired = set([b for b in button_sequence if b in button_map])

        if self.sticky_buttons:
            # Release buttons that are held but not desired now
            for b in list(self._held_buttons - desired):
                _, rel = button_map[b]
                self.emulator.send_input(rel)
                self._held_buttons.remove(b)

            # Press buttons that are desired but not held
            for b in list(desired - self._held_buttons):
                press, _ = button_map[b]
                self.emulator.send_input(press)
                self._held_buttons.add(b)

            # Hold for action_repeat * frame_skip frames
            hold_frames = max(1, self.frame_skip * self.action_repeat)
            self.emulator.tick(hold_frames, self.shouldRender)

        else:
            # Tap behavior: press → hold → release each step
            for b in desired:
                press, _ = button_map[b]
                self.emulator.send_input(press)

            hold_frames = max(1, self.frame_skip * self.action_repeat)
            self.emulator.tick(hold_frames, self.shouldRender)

            for b in desired:
                _, rel = button_map[b]
                self.emulator.send_input(rel)
                    
    def _tick(self):
        """Advance emulator by one frame."""
        if self.emulator is None:
            return
            
        # Tick the emulator
        self.emulator.tick()
        
    def _get_observation(self) -> np.ndarray:
        """Return normalized observation vector."""
        if self.emulator is None:
            return np.zeros(43, dtype=np.float32)

        from memory_map import get_observation_vector
        ram = self._get_wram_c000_dfff()
        vec = get_observation_vector(ram)
        return np.array(vec, dtype=np.float32)
        
    def _is_done(self) -> bool:
        """Check if episode is done.

        Ends when a terminal condition is detected such as the party being
        entirely fainted. Also respects step limit.
        """
        # Step limit
        if self.episode_steps > 10000:
            return True

        try:
            ram_data = self._get_wram_d000()
            if len(ram_data) >= 0x1000:
                from memory_map import decode_party_stats, get_party_count
                party_count = int(get_party_count(ram_data))

                # Brand-new games can have an empty party (count == 0). That should NOT be terminal.
                if party_count > 0:
                    party_hp = decode_party_stats(ram_data)[:party_count]
                    # Consider episode done only if all existing party members have 0 HP
                    if all(hp <= 0.0 for hp in party_hp):
                        return True
        except Exception:
            pass

        return False
        
    def _load_state(self, state_path: str):
        """Load emulator save state."""
        if self.emulator is None:
            return
            
        # Check if save state file exists
        if not os.path.exists(state_path):
            print(f"Save state file not found: {state_path}")
            return
            
        try:
            # Load the save state using PyBoy
            with open(state_path, "rb") as f:
                self.emulator.load_state(f)
        except Exception as e:
            print(f"Error loading save state {state_path}: {e}")
            
    def render(self, mode='human'):
        """Render the environment (headless mode, so no visual output)."""
        # In headless mode, no rendering needed
        pass
        
    def _release_all_buttons(self) -> None:
        """Release any currently held buttons (sticky input cleanup)."""
        if self.emulator is None:
            self._held_buttons = set()
            return

        try:
            from pyboy.utils import WindowEvent
        except Exception:
            self._held_buttons = set()
            return

        button_release_map = {
            'UP': WindowEvent.RELEASE_ARROW_UP,
            'DOWN': WindowEvent.RELEASE_ARROW_DOWN,
            'LEFT': WindowEvent.RELEASE_ARROW_LEFT,
            'RIGHT': WindowEvent.RELEASE_ARROW_RIGHT,
            'A': WindowEvent.RELEASE_BUTTON_A,
            'B': WindowEvent.RELEASE_BUTTON_B,
            'SELECT': WindowEvent.RELEASE_BUTTON_SELECT,
            'START': WindowEvent.RELEASE_BUTTON_START,
        }

        for b in list(getattr(self, '_held_buttons', set())):
            rel = button_release_map.get(b)
            if rel is not None:
                try:
                    self.emulator.send_input(rel)
                except Exception:
                    pass

        self._held_buttons = set()

    def close(self):
        """Close the emulator."""
        if self.emulator:
            try:
                self._release_all_buttons()
            except Exception:
                pass
            self.emulator.stop()
            self.emulator = None

    def _get_wram_d000(self) -> bytes:
        """Return a 0x1000-byte slice of WRAM (0xD000..0xDFFF) as bytes."""
        if self.emulator is None:
            return b"\x00" * 0x1000

        mem = self.emulator.memory
        base = 0xD000
        buf = bytearray(0x1000)
        for i in range(0x1000):
            buf[i] = int(mem[base + i]) & 0xFF
        return bytes(buf)

    def _get_wram_c000_dfff(self) -> bytes:
        """Return a 0x2000-byte slice of WRAM (0xC000..0xDFFF) as bytes.

        This includes Cxxx battle result flags like `wBattleResult` (0xCF0B).
        """
        if self.emulator is None:
            return b"\x00" * 0x2000

        mem = self.emulator.memory
        base = 0xC000
        buf = bytearray(0x2000)
        for i in range(0x2000):
            buf[i] = int(mem[base + i]) & 0xFF
        return bytes(buf)

    # ---- Hybrid (Option C) tuning hooks for SB3 callbacks ----
    def set_phase(self, phase: int) -> None:
        """Manually set the reward phase for this env instance."""
        try:
            rs = getattr(self, 'reward_shaper', None)
            if rs is not None and hasattr(rs, 'set_phase'):
                rs.set_phase(int(phase))
        except Exception:
            pass

    def get_phase(self) -> int:
        """Return current reward phase."""
        try:
            rs = getattr(self, 'reward_shaper', None)
            if rs is not None and hasattr(rs, 'get_phase'):
                return int(rs.get_phase())
        except Exception:
            pass
        return 1

    def set_reward_multipliers(self, multipliers: dict) -> None:
        """Update per-component reward multipliers (exploration, stuck, etc.)."""
        try:
            rs = getattr(self, 'reward_shaper', None)
            if rs is not None and hasattr(rs, 'set_dynamic_multipliers'):
                rs.set_dynamic_multipliers(multipliers)
        except Exception:
            pass