"""
Live showcase runner for Pokémon Yellow RL agent.
Runs the game in a visible window with real-time model loading.
"""
import os
import sys
import time
import pygame
import numpy as np
from PIL import Image
import threading
from queue import Queue
import json
import requests
from dotenv import load_dotenv

# Add project root directory to Python path to find pokemon_yellow_env
project_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(project_root)

from pokemon_yellow_env import PokemonYellowEnv
from memory_map import (
    decode_player_position, decode_badge_count, decode_party_stats, get_observation_vector,
    pretty_print_battle, type_id_to_name, move_id_to_name, parse_player_name
)
from actions import ActionHandler
from rewards import PhaseRewardShaper

class ShowcaseRunner:
    """
    Live showcase runner for Pokémon Yellow RL agent.
    """
    
    def __init__(self, 
                 model_path: str = "runs/checkpoints/latest.zip",
                 save_state_path: str = "states/pallet_town.state",
                 window_size: tuple = (1200, 1000),
                 fps: int = 60):
        
        self.model_path = model_path
        self.save_state_path = save_state_path
        self.window_size = window_size
        self.fps = fps
        
        # Initialize components
        self.env = PokemonYellowEnv()
        self.reward_shaper = PhaseRewardShaper()
        
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
            
            # Update stats
            self.episode_steps += 1
            if 'tiles_visited' in info:
                self.tiles_visited = info['tiles_visited']
            if 'reward' in info:
                self.total_reward += info['reward']
                
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
            f"Total Reward: {self.total_reward:.2f}",
            f"Model: {os.path.basename(self.current_model) if self.current_model else 'None'}"
        ]
        
        for i, text in enumerate(stats_text):
            text_surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10 + i * 40))
            
        # Draw position
        try:
            # Decode position from RAM
            ram_data = self.env._get_observation()
            position = decode_player_position(ram_data)
            pos_text = f"Position: ({position[1]}, {position[2]}) on map {position[0]}"
            pos_surface = self.font.render(pos_text, True, (255, 255, 255))
            self.screen.blit(pos_surface, (10, 200))
        except Exception as e:
            pos_text = f"Position: Error reading ({str(e)})"
            pos_surface = self.font.render(pos_text, True, (255, 255, 255))
            self.screen.blit(pos_surface, (10, 200))

            # Battle inspector overlay (shows a compact summary when in battle)
            try:
                batt = pretty_print_battle(ram_data)
                if batt.get('in_battle'):
                    y0 = 260
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
                # Non-fatal: inspector shouldn't crash the showcase
                pass

        # Send periodic overlay updates (non-blocking)
        now = time.time()
        try:
            # Build a small payload; tolerate missing data
            payload = {
                'stats': {
                    'tiles_visited': self.tiles_visited,
                    'episode_steps': self.episode_steps,
                    'battles_won': self.battles_won,
                    'battles_lost': self.battles_lost,
                    'badges': self.badges_earned,
                    'money': 0,  # We'll update this with actual money from RAM
                    'total_reward': self.total_reward
                },
                'commentary': '',
                'action_confidence': {
                    'up': 0.0,
                    'down': 0.0,
                    'left': 0.0,
                    'right': 0.0,
                    'a': 0.0,
                    'b': 0.0
                }
            }

            # Try to include position, battle info, and money if available
            try:
                ram_data = self.env._get_observation()
                position = decode_player_position(ram_data)
                payload['game_state'] = {
                    'location': 'unknown', 
                    'x': position[1], 
                    'y': position[2],
                    'in_battle': False,
                    'in_menu': False
                }
                batt = pretty_print_battle(ram_data)
                payload['game_state']['in_battle'] = bool(batt.get('in_battle')) if batt else False
                
                # Get money from RAM
                from memory_map import decode_money
                money = decode_money(ram_data)
                payload['stats']['money'] = money
                
            except Exception:
                pass

            if now - self._last_overlay_post >= self.post_interval:
                self._send_overlay_update_async(payload)
                self._last_overlay_post = now
        except Exception:
            pass

        pygame.display.flip()
        
    def start(self):
        """Start the showcase runner."""
        print("Starting showcase runner...")
        
        # Initialize display
        self.initialize_display()
        
        # Load initial model
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
        else:
            # Try to find latest checkpoint
            latest_checkpoint = self.get_latest_checkpoint()
            if latest_checkpoint:
                self.load_model(latest_checkpoint)
            else:
                print("No model found, using default behavior")
                # Create a simple test environment
                print("Starting with random actions since no model found")
                
        self.running = True
        
        try:
            # Run episodes
            while self.running:
                self.run_episode()
                
                # Reset for next episode
                self.episode_steps = 0
                self.tiles_visited = 0
                self.total_reward = 0
                
        except KeyboardInterrupt:
            print("Showcase interrupted by user")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        pygame.quit()
        print("Showcase runner stopped")

def main():
    """Main showcase function."""
    load_dotenv()  # Load environment variables from .env file if present
    # Create showcase runner
    runner = ShowcaseRunner(
        model_path="runs/checkpoints/latest.zip",
        save_state_path="states/start_game.state",
        window_size=(800, 600),
        fps=60
    )
    
    # Start showcase
    runner.start()

if __name__ == "__main__":
    main()
