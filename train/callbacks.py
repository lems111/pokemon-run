"""
Custom callbacks for Pokémon Yellow RL training.
"""
from stable_baselines3.common.callbacks import BaseCallback
import os
import time
from datetime import datetime
import numpy as np
import json

class CheckpointCallback(BaseCallback):
    """
    Custom callback to save model checkpoints during training.
    """
    
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "model", verbose: int = 0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
    def _init_callback(self) -> None:
        """Initialize callback."""
        pass
        
    def _on_training_start(self) -> None:
        """Initialize training callback."""
        pass
        
    def _on_training_end(self) -> None:
        """Clean up at the end of training."""
        pass
        
    def _on_step(self) -> bool:
        """
        This method is called by the model after each step in the environment.
        
        :return: (bool) If the model should continue training
        """
        if self.n_calls % self.save_freq == 0:
            # Save model checkpoint
            model_path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.n_calls}.zip"
            )
            
            self.model.save(model_path)
            
            if self.verbose > 0:
                print(f"Saved checkpoint to {model_path}")
                
        return True

class TensorboardCallback(BaseCallback):
    """
    Custom callback to log additional metrics to TensorBoard.
    """
    
    def __init__(self, verbose: int = 0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.tiles_visited = []
        self.battles_won = []
        self.battles_lost = []
        self.badges_earned = []
        
    def _init_callback(self) -> None:
        """Initialize callback."""
        pass
        
    def _on_training_start(self) -> None:
        """Initialize training callback."""
        pass
        
    def _on_training_end(self) -> None:
        """Clean up at the end of training."""
        pass
        
    def _on_step(self) -> bool:
        """
        This method is called by the model after each step in the environment.
        
        :return: (bool) If the model should continue training
        """
        # Log additional metrics to TensorBoard
        # This would be implemented based on the environment's info
        # For now, we'll just log the step count
        
        if self.verbose > 0 and self.n_calls % 1000 == 0:
            print(f"Training step {self.n_calls}")
            
        return True

class OverlayCallback(BaseCallback):
    """
    Callback to POST basic training stats to the overlay server so the overlay can show live training info.
    """
    def __init__(self, url: str = "http://127.0.0.1:8080/update", update_freq: int = 1000, verbose: int = 0):
        super(OverlayCallback, self).__init__(verbose)
        self.url = url
        self.update_freq = update_freq
        self.action_counts = {}
        self.total_actions = 0
        self.last_action = None
        self.env = None

    def _on_step(self) -> bool:
        # For now, we send minimal data since we don't have access to environment stats
        # This can be extended to track stats if needed
        if self.n_calls % self.update_freq == 0:
            # Initialize action confidence with zeros
            action_confidence = {
                'up': 0.0,
                'down': 0.0,
                'left': 0.0,
                'right': 0.0,
                'a': 0.0,
                'b': 0.0
            }
            
            # If we have action data, update confidence
            if self.total_actions > 0:
                for action_name, count in self.action_counts.items():
                    # Convert action name to lowercase for matching
                    if action_name.lower() in action_confidence:
                        action_confidence[action_name.lower()] = count / self.total_actions

            payload = {
                'stats': {
                    'episode_steps': int(self.n_calls)
                },
                'commentary': f'Training step {self.n_calls}',
                'action_confidence': action_confidence,
                'game_state': {
                    'location': 'unknown',
                    'x': 0,
                    'y': 0,
                    'in_battle': False,
                    'in_menu': False
                }
            }
            try:
                import requests
                requests.post(self.url, json=payload, timeout=1)
            except Exception as e:
                if self.verbose > 0:
                    print(f"Overlay update failed: {e}")
        return True

    def _on_training_start(self) -> None:
        """Initialize training callback."""
        # This would be called at training start
        self.action_counts = {}
        self.total_actions = 0
        self.last_action = None

    def _on_training_end(self) -> None:
        """Clean up at the end of training."""
        # This would be called at training end
        pass
        
    def update_action_counts(self, action: int):
        """Update action counts for overlay display."""
        from actions import ACTIONS
        action_name = ACTIONS.get(action, 'UNKNOWN').lower()
        if action_name != 'unknown':
            # Map action names to the ones used in overlay
            if action_name == 'up':
                overlay_action = 'up'
            elif action_name == 'down':
                overlay_action = 'down'
            elif action_name == 'left':
                overlay_action = 'left'
            elif action_name == 'right':
                overlay_action = 'right'
            elif action_name == 'a':
                overlay_action = 'a'
            elif action_name == 'b':
                overlay_action = 'b'
            else:
                overlay_action = action_name  # fallback for other actions
                
            self.action_counts[overlay_action] = self.action_counts.get(overlay_action, 0) + 1
            self.total_actions += 1
            
    def get_action_confidence(self):
        """Get current action confidence values for overlay display."""
        action_confidence = {
            'up': 0.0,
            'down': 0.0,
            'left': 0.0,
            'right': 0.0,
            'a': 0.0,
            'b': 0.0
        }
        
        if self.total_actions > 0:
            for action_name, count in self.action_counts.items():
                if action_name.lower() in action_confidence:
                    action_confidence[action_name.lower()] = count / self.total_actions
                    
        return action_confidence
        
    def set_env(self, env):
        """Set the environment to access its action tracking."""
        self.env = env

class ProgressCallback(BaseCallback):
    """
    Callback to track and log training progress.
    """
    
    def __init__(self, verbose: int = 0):
        super(ProgressCallback, self).__init__(verbose)
        self.start_time = time.time()
        self.episode_count = 0
        self.total_steps = 0
        self.recent_rewards = []
        self.tiles_visited = 0
        
    def _init_callback(self) -> None:
        """Initialize callback."""
        pass
        
    def _on_training_start(self) -> None:
        """Initialize training callback."""
        self.start_time = time.time()
        print("Training started...")
        
    def _on_training_end(self) -> None:
        """Clean up at the end of training."""
        total_time = time.time() - self.start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
    def _on_step(self) -> bool:
        """
        This method is called by the model after each step in the environment.
        """
        self.total_steps += 1
        
        # Log progress every 1000 steps
        if self.n_calls % 1000 == 0:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            steps_per_second = self.total_steps / elapsed_time
            
            print(f"Steps: {self.total_steps}, "
                  f"Time: {elapsed_time:.2f}s, "
                  f"Steps/sec: {steps_per_second:.2f}")
                  
        return True

class CurriculumCallback(BaseCallback):
    """
    Callback to handle curriculum learning transitions.
    """
    
    def __init__(self, verbose: int = 0):
        super(CurriculumCallback, self).__init__(verbose)
        self.current_stage = 1
        self.stage_thresholds = {
            1: 1000,   # Movement sandbox
            2: 5000,   # Early route
            3: 10000,  # First battles
            4: 20000,  # Badge pursuit
        }
        self.stage_start_steps = {}
        self.stage_start_time = {}
        
    def _init_callback(self) -> None:
        """Initialize callback."""
        self.stage_start_steps = {stage: 0 for stage in self.stage_thresholds}
        self.stage_start_time = {stage: time.time() for stage in self.stage_thresholds}
        
    def _on_training_start(self) -> None:
        """Initialize training callback."""
        print("Curriculum learning initialized")
        
    def _on_training_end(self) -> None:
        """Clean up at the end of training."""
        print("Curriculum learning completed")
        
    def _on_step(self) -> bool:
        """
        Check if we should transition to a new stage.
        """
        # This would be called with environment info
        # For now, just log the step count
        if self.verbose > 0 and self.n_calls % 5000 == 0:
            print(f"Stage {self.current_stage} - Step {self.n_calls}")
            
        return True
        
    def update_stage(self, tiles_visited: int):
        """
        Update training stage based on progress.
        """
        for stage, threshold in sorted(self.stage_thresholds.items()):
            if tiles_visited >= threshold and stage > self.current_stage:
                self.current_stage = stage
                self.stage_start_steps[stage] = self.n_calls
                self.stage_start_time[stage] = time.time()
                
                print(f"Transitioned to stage {stage} at step {self.n_calls}")
                break

class ModelSaverCallback(BaseCallback):
    """
    Callback to save the best model based on a metric.
    """
    
    def __init__(self, save_freq: int, save_path: str, metric_name: str = "tiles_visited", verbose: int = 0):
        super(ModelSaverCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.metric_name = metric_name
        self.best_metric = float('-inf')
        self.best_model_path = None
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
    def _init_callback(self) -> None:
        """Initialize callback."""
        pass
        
    def _on_training_start(self) -> None:
        """Initialize training callback."""
        print(f"Model saver initialized with metric: {self.metric_name}")
        
    def _on_training_end(self) -> None:
        """Clean up at the end of training."""
        if self.best_model_path:
            print(f"Best model saved to {self.best_model_path}")
            
    def _on_step(self) -> bool:
        """
        Save best model based on metric.
        """
        if self.n_calls % self.save_freq == 0:
            # This would be implemented with actual metric values from environment
            # For now, just save a checkpoint
            model_path = os.path.join(
                self.save_path,
                f"best_model_{self.n_calls}.zip"
            )
            
            self.model.save(model_path)
            
            if self.verbose > 0:
                print(f"Saved best model checkpoint to {model_path}")
                
        return True
