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
    
    def __init__(self):
        super(PokemonYellowEnv, self).__init__()
        
        # Define action space (discrete buttons)
        self.action_space = spaces.Discrete(12)  # 12 possible actions
        self._held_buttons = set()
        self._action_handler = None  # init lazily

        # Define observation space (WRAM slice used by Pokémon Yellow: 0xD000..0xDFFF)
        # WRAM window here is 0x1000 bytes (4KB) and memory_map offsets are relative to 0xD000
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(0x1000,), dtype=np.uint8
        )
        
        # Emulator and state management
        self.emulator_path = os.getenv("EMULATOR_PATH") or "pokemon_yellow.gb"
        self.save_state_path = os.getenv("SAVE_STATE_PATH") or "pokemon_yellow.gbc.state"
        self.emulator = None
        self.current_state = None
        self.window = os.getenv("WINDOW") or "null"
        self.shouldRender = self.window != "null"
        # Training parameters
        self.frame_skip = 4
        self.action_repeat = 4
        self.sticky_buttons = True
        
        # Episode tracking
        self.episode_steps = 0
        self.total_tiles_visited = 0
        self.tiles_visited = set()
        self.last_position = None

        # Previous state for change detection (used for reward shaping and outcomes)
        self._prev_in_battle = False
        self._prev_badges = 0
        self._prev_money = 0
        self._prev_party_hp = [0.0] * 6
        self._last_battle_result = None  # 'win' | 'loss' | None
        # Track previous opponent roster count to detect end-of-battle via roster clearing
        self._prev_opponent_roster_count = 0
        # Previous battle type (set lazily)
        self._prev_battle_type = None
        
        # Action tracking for overlay
        self.action_counts = {}
        self.total_actions = 0
        
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
            # Optionally enable sound if needed
            # sound=False,
            # sound_volume=0,
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

        # Initialize previous-state values used for reward shaping
        try:
            from memory_map import decode_badge_count, decode_money, decode_party_stats, decode_in_battle, get_byte, RAM_ADDRESSES
            self._prev_badges = decode_badge_count(obs)
            self._prev_money = decode_money(obs)
            self._prev_party_hp = decode_party_stats(obs)
            self._prev_in_battle = bool(decode_in_battle(obs))
            # Initialize opponent roster count (used for more accurate end-of-battle detection)
            try:
                self._prev_opponent_roster_count = get_byte(obs, RAM_ADDRESSES['OPPONENT_ROSTER_COUNT'])
            except Exception:
                self._prev_opponent_roster_count = 0
            self._last_battle_result = None
        except Exception:
            # Keep defaults if decoders are not available or fail
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
        reward = self._compute_reward(action)
        
        # Check if episode is done
        done = self._is_done()
        
        # Update episode tracking
        self.episode_steps += 1
        
        # Update action counts for overlay
        if hasattr(self, 'action_counts'):
            from actions import ACTIONS
            action_name = ACTIONS[action]
            self.action_counts[action_name] = self.action_counts.get(action_name, 0) + 1
            self.total_actions += 1
        
        # Info dictionary
        info = {
            'episode_steps': self.episode_steps,
            'tiles_visited': len(self.tiles_visited),
            'reward': reward
        }
        
        # In Gymnasium, step returns (observation, reward, terminated, truncated, info)
        # For this environment, we'll set truncated=False and terminated=done
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
        """Get current WRAM observation from PyBoy emulator.

        Pokémon Yellow relevant RAM is in the WRAM window at 0xD000..0xDFFF
        (0x1000 bytes). `memory_map` helper functions use offsets relative to
        0xD000, so we return a 4KB slice mapped such that index 0 corresponds
        to address 0xD000.
        """
        if self.emulator is None:
            return np.zeros(0x1000, dtype=np.uint8)
            
        # Get WRAM slice from emulator using PyBoy memory API
        try:
            ram = []
            # Read the WRAM page used by Pokémon (0xD000..0xDFFF)
            for addr in range(0xD000, 0xE000):
                # Access memory byte by address
                try:
                    byte_val = self.emulator.memory[addr]
                    masked_value = byte_val & 0xFF
                    ram.append(masked_value)
                except Exception as e:
                    # If we can't read a specific address, use 0
                    print(f"Error reading memory address {hex(addr)}: {e}")
                    ram.append(0)
            return np.array(ram, dtype=np.uint8)
        except Exception as e:
            # Fallback to zero array if memory access fails
            print(f"Error reading WRAM: {e}")
            return np.zeros(0x1000, dtype=np.uint8)
        
    def _compute_reward(self, action: int) -> float:
        """Compute reward based on current state and action.

        Uses deltas on badges, money, and detects battle outcome transitions
        (end-of-battle) by comparing previous and current battle/party state.
        """
        reward = -0.001  # Small penalty per step

        # Exploration reward
        if self._is_new_tile():
            reward += 1.0

        # Read current RAM-derived values
        try:
            ram_data = self._get_observation()
            from memory_map import (
                decode_badge_count,
                decode_money,
                decode_party_stats,
                decode_in_battle,
            )

            curr_badges = decode_badge_count(ram_data)
            curr_money = decode_money(ram_data)
            curr_party_hp = decode_party_stats(ram_data)
            curr_in_battle = bool(decode_in_battle(ram_data))
            # Opponent roster count helps detect a clean end-of-battle
            from memory_map import get_byte, RAM_ADDRESSES
            try:
                curr_opponent_roster = get_byte(ram_data, RAM_ADDRESSES['OPPONENT_ROSTER_COUNT'])
            except Exception:
                curr_opponent_roster = 0

            # Badge gains are meaningful (large reward)
            badge_delta = curr_badges - self._prev_badges
            if badge_delta > 0:
                reward += 50.0 * badge_delta

            # If we detect a badge gain while previously in a battle, treat it as an immediate win
            if self._prev_in_battle and badge_delta > 0:
                reward += 10.0
                self.battles_won = getattr(self, 'battles_won', 0) + 1
                self._last_battle_result = 'win'

            # Money gains - small reward scaled
            money_delta = curr_money - self._prev_money
            if money_delta > 0:
                reward += 0.001 * money_delta

            # Similarly, a money gain during a prior-battle can be treated as strong evidence of win
            if self._prev_in_battle and money_delta > 0:
                reward += 1.0
                self.battles_won = getattr(self, 'battles_won', 0) + 1
                self._last_battle_result = 'win'
            # Battle end detection: previous in-battle -> now not in battle
            battle_just_ended = False
            if self._prev_in_battle and not curr_in_battle:
                battle_just_ended = True
            # Also consider explicit roster clearing as a strong signal
            if getattr(self, '_prev_opponent_roster_count', 0) > 0 and curr_opponent_roster == 0:
                battle_just_ended = True

            if battle_just_ended:
                # proceed to outcome heuristics

                # Use battle type, badge/money/HP deltas, and roster info to decide outcome
                from memory_map import decode_battle_type, decode_text_state

                prev_sum = sum(self._prev_party_hp)
                curr_sum = sum(curr_party_hp)

                # Battle type may distinguish special battles; treat non-zero as valid battle
                prev_battle_type = getattr(self, '_prev_battle_type', None)
                if prev_battle_type is None:
                    # Initialize from previous observation if not set
                    try:
                        self._prev_battle_type = decode_battle_type(self._get_observation())
                    except Exception:
                        self._prev_battle_type = 0

                # Heuristics for win/loss with roster confirmation
                badge_delta_during = curr_badges - self._prev_badges
                money_delta_during = curr_money - self._prev_money

                # Text state can sometimes indicate end-of-battle dialog; include as soft evidence
                text_state_now = decode_text_state(self._get_observation())

                # Prefer strong evidence: badge/money gains
                if badge_delta_during > 0 or money_delta_during > 0:
                    reward += 10.0
                    self.battles_won = getattr(self, 'battles_won', 0) + 1
                    self._last_battle_result = 'win'
                else:
                    # If HP dropped by more than 30% -> loss (strong signal)
                    if prev_sum > 0 and (prev_sum - curr_sum) / max(prev_sum, 1) > 0.3:
                        reward -= 5.0
                        self.battles_lost = getattr(self, 'battles_lost', 0) + 1
                        self._last_battle_result = 'loss'
                    # If all party members fainted -> loss
                    elif all(hp == 0.0 for hp in curr_party_hp):
                        reward -= 5.0
                        self.battles_lost = getattr(self, 'battles_lost', 0) + 1
                        self._last_battle_result = 'loss'
                    # Roster cleared and we still have enough HP -> likely a win
                    # Use an absolute threshold on normalized HP sum to avoid
                    # treating near-zero HP as a win when everyone nearly fainted.
                    elif curr_opponent_roster == 0 and curr_sum > 0.2:
                        reward += 10.0
                        self.battles_won = getattr(self, 'battles_won', 0) + 1
                        self._last_battle_result = 'win'
                    else:
                        # Otherwise treat as a minor win (uncertain)
                        reward += 2.0
                        self._last_battle_result = 'win'

                # Update previous battle type with current observation
                try:
                    self._prev_battle_type = decode_battle_type(self._get_observation())
                except Exception:
                    self._prev_battle_type = 0

            # Update previous-state trackers
            self._prev_badges = curr_badges
            self._prev_money = curr_money
            self._prev_party_hp = curr_party_hp
            self._prev_in_battle = curr_in_battle
            # Update opponent roster tracker
            try:
                self._prev_opponent_roster_count = curr_opponent_roster
            except Exception:
                self._prev_opponent_roster_count = 0

        except Exception:
            # If RAM decoding fails, fall back to simple reward
            pass

        return reward
        
    def _is_new_tile(self) -> bool:
        """Check if agent has visited a new tile by decoding position from RAM."""
        try:
            ram_data = self._get_observation()
            if len(ram_data) >= 0x1000:
                # Decode position from WRAM slice using memory map (offsets are relative to 0xD000)
                from memory_map import decode_player_position
                position = decode_player_position(ram_data)
                map_id, x, y = position
                
                # Create tile identifier
                tile_id = (map_id, x, y)
                
                # Reward for new tiles
                if tile_id not in self.tiles_visited:
                    self.tiles_visited.add(tile_id)
                    return True
                    
            return False
        except Exception:
            return False
        
    def _is_battle_win(self) -> bool:
        """Return True if the most-recent resolved battle was a win.

        Primary check: consult the recorded `_last_battle_result` set during
        reward computation. If unset, use a conservative fallback heuristic
        based on RAM deltas (badges, money, party HP) when we've just
        transitioned out of a battle.
        """
        # Primary recorded result
        if self._last_battle_result == 'win':
            return True
        if self._last_battle_result == 'loss':
            return False

        # Fallback heuristic: only consider outcomes immediately after
        # transitioning out of battle (previously in battle, now not) or roster clearing
        try:
            ram_data = self._get_observation()
            from memory_map import decode_in_battle, decode_badge_count, decode_money, decode_party_stats, get_byte, RAM_ADDRESSES

            curr_in_battle = bool(decode_in_battle(ram_data))
            try:
                curr_roster = get_byte(ram_data, RAM_ADDRESSES['OPPONENT_ROSTER_COUNT'])
            except Exception:
                curr_roster = 0

            if (self._prev_in_battle and not curr_in_battle) or (self._prev_opponent_roster_count > 0 and curr_roster == 0):
                curr_badges = decode_badge_count(ram_data)
                curr_money = decode_money(ram_data)
                curr_party_hp = decode_party_stats(ram_data)

                prev_sum = sum(self._prev_party_hp)
                curr_sum = sum(curr_party_hp)

                # Any badge or money gain -> likely a win
                if curr_badges > self._prev_badges or curr_money > self._prev_money:
                    return True
                # If roster was explicitly cleared and party retains some HP -> treat as win
                if getattr(self, '_prev_opponent_roster_count', 0) > 0 and curr_roster == 0 and curr_sum > 0.2:
                    return True
                # If HP dropped by significant fraction -> likely a loss
                if prev_sum > 0 and (prev_sum - curr_sum) / max(prev_sum, 1) > 0.3:
                    return False
                # If all party HP zero -> loss
                if all(hp == 0.0 for hp in curr_party_hp):
                    return False
                # Roster cleared & party survived -> win (fallback)
                if curr_roster == 0 and curr_sum > 0.2:
                    return True
        except Exception:
            # Be conservative on failure
            pass
        return False
        
    def _is_battle_loss(self) -> bool:
        """Return True if the most-recent resolved battle was a loss.

        Uses `_last_battle_result` when available. If unset, falls back to a
        conservative heuristic similar to `_is_battle_win` (HP drop >30%
        indicates a likely loss; badge/money gain or HP improvement indicates
        a win/avoid treating as loss).
        """
        if self._last_battle_result == 'loss':
            return True
        if self._last_battle_result == 'win':
            return False

        try:
            ram_data = self._get_observation()
            from memory_map import decode_in_battle, decode_badge_count, decode_money, decode_party_stats

            curr_in_battle = bool(decode_in_battle(ram_data))
            if self._prev_in_battle and not curr_in_battle:
                curr_badges = decode_badge_count(ram_data)
                curr_money = decode_money(ram_data)
                curr_party_hp = decode_party_stats(ram_data)

                prev_sum = sum(self._prev_party_hp)
                curr_sum = sum(curr_party_hp)

                # If badges or money increased -> not a loss (strong evidence of win)
                if curr_badges > self._prev_badges or curr_money > self._prev_money:
                    return False
                # If roster was explicitly cleared and party retains some HP -> treat as win (not a loss)
                try:
                    from memory_map import get_byte, RAM_ADDRESSES
                    curr_roster = get_byte(ram_data, RAM_ADDRESSES['OPPONENT_ROSTER_COUNT'])
                except Exception:
                    curr_roster = 0
                if getattr(self, '_prev_opponent_roster_count', 0) > 0 and curr_roster == 0 and curr_sum > 0.2:
                    return False
                # If HP improved/stayed same -> not a loss
                if curr_sum >= prev_sum:
                    return False
                # If HP dropped by significant fraction -> likely a loss
                if prev_sum > 0 and (prev_sum - curr_sum) / max(prev_sum, 1) > 0.3:
                    return True
        except Exception:
            pass
        return False
        
    def _is_done(self) -> bool:
        """Check if episode is done.

        Ends when a terminal condition is detected such as the party being
        entirely fainted. Also respects step limit.
        """
        # Step limit
        if self.episode_steps > 10000:
            return True

        try:
            ram_data = self._get_observation()
            if len(ram_data) >= 0x1000:
                from memory_map import decode_party_stats
                party_hp = decode_party_stats(ram_data)
                # Consider episode done if all party members have 0 HP
                if all(hp == 0.0 for hp in party_hp):
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
        
    def close(self):
        """Close the emulator."""
        if self.emulator:
            self.emulator.stop()
            self.emulator = None
