"""
Live showcase runner for Pokémon Yellow RL agent.
Runs the game in a visible window with real-time model loading.
"""
import os
import sys
import time
from dotenv import load_dotenv
import pygame
import requests
import numpy as np
from PIL import Image
import threading
from queue import Queue

# Add project root directory to Python path to find pokemon_yellow_env
project_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(project_root)

from pokemon_yellow_env import PokemonYellowEnv
from memory_map import (
    decode_basic_game_info, decode_money,
    pretty_print_battle, type_id_to_name, move_id_to_name
)
from actions import ActionHandler

# Map id -> name (overlay location)
try:
    from memory.map_names import map_name as _map_name
except Exception:
    def _map_name(map_id):
        return "Unknown"
    
class ShowcaseRunner:
    """
    Live showcase runner for Pokémon Yellow RL agent.
    """
    
    def __init__(self, 
                 model_path: str = "runs/checkpoints/latest.zip",
                 save_state_path: str = "states/pallet_town.state",
                 window_size: tuple = (1200, 1000),
                 fps: int = 60):
        load_dotenv()
        self.model_path = model_path
        self.save_state_path = save_state_path
        self.window_size = window_size
        self.fps = fps
        
        # Initialize components
        self.env = PokemonYellowEnv()
        
        # Emulator state
        self.running = False
        self.current_model = None
        self.model = None
        self.last_checkpoint_time = 0
        
        # Display state
        self.screen = None
        self.font = None
        self.clock = None
        
        # Game stats
        self.episode_steps = 0
        self.tiles_visited = 0
        self.battles_won = 0
        self.battles_lost = 0
        self.badges_earned = 0
        self.total_reward = 0
        self.action_counts = {}
        self.total_actions = 0

        # Overlay/streaming updates
        self.overlay_url = os.environ.get('OVERLAY_URL', 'http://127.0.0.1:8080/update')
        self.post_interval = float(os.environ.get('OVERLAY_POST_INTERVAL', '0.5'))
        self._last_overlay_post = 0.0
        self.send_action_data = os.environ.get('SEND_ACTION_DATA', 'true').lower() == 'true'

        # Model loading
        self.checkpoint_queue = Queue()
        self.model_lock = threading.Lock()

    def _send_overlay_update_async(self, payload: dict):
        """Send overlay update in a background thread to avoid blocking the display."""
        def _worker():
            try:
                requests.post(self.overlay_url, json=payload, timeout=1)
            except Exception:
                pass
        threading.Thread(target=_worker, daemon=True).start()
        
    def initialize_display(self):
        """Initialize pygame display."""
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Pokémon Yellow RL Showcase")
        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()
        
    def load_model(self, model_path: str):
        """Load a trained model."""
        try:
            # Import here to avoid issues with pyboy
            from stable_baselines3 import PPO
            
            print(f"Loading model from {model_path}")
            self.model = PPO.load(model_path)
            self.current_model = model_path
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
            
    def get_latest_checkpoint(self):
        """Get the latest checkpoint from the checkpoint directory."""
        checkpoint_dir = "runs/checkpoints"
        if not os.path.exists(checkpoint_dir):
            return None
            
        # Get all checkpoint files
        checkpoints = [f for f in os.listdir(checkpoint_dir) 
                      if f.startswith("ppo_") and f.endswith(".zip")]
        
        if not checkpoints:
            return None
            
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
            return latest_checkpoint
            
        return None
        
    def hot_swap_model(self):
        """Check for and load new checkpoints."""
        latest_checkpoint = self.get_latest_checkpoint()
        
        if latest_checkpoint and latest_checkpoint != self.current_model:
            print(f"Hot-swapping model to {latest_checkpoint}")
            if self.load_model(latest_checkpoint):
                self.last_checkpoint_time = time.time()
                return True
        return False
        
    def run_episode(self):
        """Run a single episode."""
        # Reset environment
        obs, info = self.env.reset()
        # Reset per-episode counters
        self.episode_steps = 0
        self.total_reward = 0.0
        done = False
        total_reward = 0

        print("Starting showcase episode...")

        while not done and self.running:
            # Hot-swap model if needed
            self.hot_swap_model()

            # Get action from model
            action = self.get_action(obs)

            # Execute action
            obs, reward, done, _, info = self.env.step(action)
            total_reward += reward

            # Update stats from env info
            self.episode_steps += 1
            self.total_reward = float(total_reward)
            if 'tiles_visited' in info:
                self.tiles_visited = int(info['tiles_visited'])
            # Pull battle stats from the reward shaper if present
            try:
                rs = getattr(self.env, 'reward_shaper', None)
                if rs is not None:
                    self.battles_won = int(getattr(rs, 'previous_battle_wins', 0))
                    self.battles_lost = int(getattr(rs, 'previous_battle_losses', 0))
            except Exception:
                pass
            # Badges can be read from info if the env provides it
            if 'badges' in info:
                self.badges_earned = int(info['badges'])

            # Render display
            self.render_display()

            # Control frame rate
            self.clock.tick(self.fps)

            # Check for quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    done = True

        print(f"Episode finished. Total reward: {total_reward}")
        
    def get_action(self, obs):
        """Get action from current model."""
        # Use the loaded model to predict action
        if self.model is not None:
            action, _states = self.model.predict(obs)
            # Extract scalar from numpy array if needed
            if isinstance(action, np.ndarray):
                action = action.item()
            return action
        else:
            # Fallback to random action if no model loaded
            return np.random.randint(0, 12)
        
    def render_display(self):
        """Render the current game state and stats."""
        self.screen.fill((0, 0, 0))  # Black background

        # Draw stats
        stats_text = [
            f"Steps: {self.episode_steps}",
            f"Tiles Visited: {self.tiles_visited}",
            f"Battles Won: {self.battles_won}",
            f"Battles Lost: {self.battles_lost}",
            f"Badges: {self.badges_earned}",
            f"Total Reward: {self.total_reward:.2f}",
            f"Model: {os.path.basename(self.current_model) if self.current_model else 'None'}",
        ]

        for i, text in enumerate(stats_text):
            text_surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10 + i * 40))

        # Read WRAM bytes for decoders
        try:
            if hasattr(self.env, '_get_wram_d000'):
                ram = self.env._get_wram_d000()
            else:
                ram = None
        except Exception:
            ram = None

       # Draw position + basic state (single source of truth)
        map_id = 0
        x = 0
        y = 0
        in_battle = False
        money = 0
        try:
            if ram is not None:
                basic = decode_basic_game_info(ram)
                map_id = int(basic.get("map_id", 0))
                x = int(basic.get("x", 0))
                y = int(basic.get("y", 0))
                in_battle = bool(basic.get("in_battle", False))
                self.badges_earned = int(basic.get("badges", 0))
                money = int(basic.get("money", 0))
        except Exception:
            pass

        pos_text = f"Position: ({x}, {y}) on {_map_name(map_id)} (0x{map_id:02X})"
        status_text = "Status: In Battle" if in_battle else "Status: Exploring"
        self.screen.blit(self.font.render(pos_text, True, (255, 255, 255)), (10, 240))
        self.screen.blit(self.font.render(status_text, True, (255, 255, 255)), (10, 280))
        self.screen.blit(self.font.render(f"Money: {money}", True, (255, 255, 255)), (10, 320))

        # Battle inspector overlay (shows a compact summary when in battle)
        try:
            if ram is not None:
                batt = pretty_print_battle(ram)
                if batt.get('in_battle'):
                    y0 = 380
                    header = f"In Battle (type {batt.get('battle_type')})"
                    self.screen.blit(self.font.render(header, True, (255, 200, 0)), (10, y0))
                    y0 += 36

                    # Show party HP summary
                    for p in batt.get('party', []):
                        p_line = f"P[{p['index']}] HP: {p['current_hp']}/{p['max_hp']}"
                        self.screen.blit(self.font.render(p_line, True, (200, 200, 200)), (10, y0))
                        y0 += 24

                    # Show up to 4 opponents
                    for opp in batt.get('opponents', [])[:4]:
                        typ = type_id_to_name(opp.get('type1')) if opp.get('type1') is not None else ''
                        moves = ','.join(move_id_to_name(m) for m in opp.get('moves', []) if m)
                        line = f"Opp #{opp.get('id')} L{opp.get('level')} HP:{opp.get('current_hp')}/{opp.get('max_hp')} Type:{typ} Moves:{moves}"
                        self.screen.blit(self.font.render(line, True, (255, 255, 255)), (10, y0))
                        y0 += 24
        except Exception:
            pass

        # Send periodic overlay updates (non-blocking)
        now = time.time()
        if now - self._last_overlay_post >= self.post_interval:
            self._last_overlay_post = now

            # Build payload with safe defaults
            payload = {
                'stats': {
                    'tiles_visited': int(self.tiles_visited),
                    'episode_steps': int(self.episode_steps),
                    'battles_won': int(self.battles_won),
                    'battles_lost': int(self.battles_lost),
                    'badges': int(self.badges_earned),
                    'money': int(money),
                    'total_reward': float(self.total_reward),
                },
                'commentary': 'Showcase running' + (' (battle)' if in_battle else ''),
                'game_state': {
                    'map_id': int(map_id),
                    'location': _map_name(int(map_id)),
                    'x': int(x),
                    'y': int(y),
                    'in_battle': bool(in_battle),
                    'in_menu': False,
                },
            }

            # Action confidence (optional)
            if self.send_action_data:
                try:
                    # Derive confidence from env action counts if available
                    ac = {'up':0.0,'down':0.0,'left':0.0,'right':0.0,'a':0.0,'b':0.0}
                    total = int(getattr(self.env, 'total_actions', 0) or 0)
                    counts = getattr(self.env, 'action_counts', {}) or {}
                    if total > 0:
                        for k in ['UP','DOWN','LEFT','RIGHT','A','B']:
                            kk = k.lower()
                            if kk in ac:
                                ac[kk] = float(counts.get(k, 0)) / float(total)
                    payload['action_confidence'] = ac
                except Exception:
                    payload['action_confidence'] = {'up':0.0,'down':0.0,'left':0.0,'right':0.0,'a':0.0,'b':0.0}

            self._send_overlay_update_async(payload)

        pygame.display.flip()