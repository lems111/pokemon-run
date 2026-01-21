"""
Custom callbacks for Pokémon Yellow RL training.
"""
from stable_baselines3.common.callbacks import BaseCallback
import os
import time
from datetime import datetime
import numpy as np
import json
from typing import Optional

# Map id -> name (overlay location)
try:
    from memory.map_names import map_name as _map_name
except Exception:
    def _map_name(map_id):
        return "Unknown"
    
class HybridTuningCallback(BaseCallback):
    """Option C: manual phase gating, automatic tuning within a phase.

    This callback does NOT change TRAINING_PHASE.
    It only anneals/shifts reward component multipliers based on live progress signals.

    Signals used: tiles_visited (from env info), and global timesteps (SB3 num_timesteps).
    """

    def __init__(
        self,
        update_freq: int = 2000,
        target_tiles: int = 400,
        target_steps: int = 5000,
        min_exploration_mult: float = 0.30,
        max_exploration_mult: float = 1.00,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.update_freq = int(update_freq)
        self.target_tiles = int(target_tiles)
        self.target_steps = int(target_steps)
        self.min_exploration_mult = float(min_exploration_mult)
        self.max_exploration_mult = float(max_exploration_mult)
        self._last_sent = None
        # Weighted blend for progress signals (tiles vs steps)
        # Defaults are tuned for Pokémon exploration so tiles dominate early.
        try:
            self.blend_tiles_weight = float(os.getenv('TUNING_BLEND_TILES_WEIGHT', '0.7'))
        except Exception:
            self.blend_tiles_weight = 0.7
        try:
            self.blend_steps_weight = float(os.getenv('TUNING_BLEND_STEPS_WEIGHT', '0.3'))
        except Exception:
            self.blend_steps_weight = 0.3
        # Track global steps since the current phase began (phase is manual in Option C)
        self._phase_start_timesteps = 0
        self._last_phase = None

    def _extract_info0(self) -> dict:
        # VecEnv -> list of dicts
        try:
            infos = self.locals.get('infos', None)
            if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
                return infos[0]
            if isinstance(infos, dict):
                return infos
        except Exception:
            pass
        return {}

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq != 0:
            return True

        info0 = self._extract_info0()

        # Read progress
        try:
            tiles = int(info0.get('tiles_visited', 0) or 0)
        except Exception:
            tiles = 0

        # Secondary progress signal: global timesteps (more stable than per-episode steps).
        # We use "timesteps since phase start" so the annealing restarts when you manually bump phases.
        phase = None
        try:
            ph_list = self.training_env.env_method('get_phase')
            if isinstance(ph_list, (list, tuple)) and len(ph_list) > 0:
                phase = int(ph_list[0])
        except Exception:
            phase = None

        # If phase changes (manual gating), reset the phase baseline for timesteps
        if phase is not None and phase != self._last_phase:
            self._last_phase = phase
            self._phase_start_timesteps = int(getattr(self, 'num_timesteps', 0) or 0)

        # Steps since the current phase began
        ep_steps = int(getattr(self, 'num_timesteps', 0) or 0) - int(getattr(self, '_phase_start_timesteps', 0) or 0)
        if ep_steps < 0:
            ep_steps = 0

        # Anneal exploration multiplier down as progress increases.
        # Primary signal: tiles_visited
        # Secondary signal: global timesteps since phase start (so we still anneal even if tiles stall)

        frac_tiles = 0.0
        if self.target_tiles > 0:
            frac_tiles = float(min(max(tiles / float(self.target_tiles), 0.0), 1.0))

        frac_steps = 0.0
        if self.target_steps > 0:
            frac_steps = float(min(max(ep_steps / float(self.target_steps), 0.0), 1.0))

        # Combine using a weighted blend (prevents steps from dominating too early)
        tw = float(getattr(self, 'blend_tiles_weight', 0.7) or 0.7)
        sw = float(getattr(self, 'blend_steps_weight', 0.3) or 0.3)
        # Normalize defensively
        if tw < 0.0:
            tw = 0.0
        if sw < 0.0:
            sw = 0.0
        denom = (tw + sw)
        if denom <= 0.0:
            tw, sw, denom = 0.7, 0.3, 1.0
        tw /= denom
        sw /= denom

        frac = (tw * frac_tiles) + (sw * frac_steps)
        frac = float(min(max(frac, 0.0), 1.0))

        exploration_mult = (self.max_exploration_mult * (1.0 - frac)) + (self.min_exploration_mult * frac)

        # Light anti-stuck boost when exploration is low (keeps movement pressure)
        stuck_mult = 1.0
        if exploration_mult <= (self.min_exploration_mult + 0.05):
            stuck_mult = 1.25

        multipliers = {
            'exploration': float(exploration_mult),
            'stuck': float(stuck_mult),
        }

        # Avoid spam if unchanged
        sig = (round(exploration_mult, 3), round(stuck_mult, 3))
        if self._last_sent == sig:
            return True
        self._last_sent = sig

        # Push updates into all envs
        try:
            self.training_env.env_method('set_reward_multipliers', multipliers)
        except Exception:
            # Fallback for some wrappers
            try:
                self.training_env.get_attr('set_reward_multipliers')
                self.training_env.env_method('set_reward_multipliers', multipliers)
            except Exception:
                pass

        if self.verbose > 0:
            gsteps = int(getattr(self, 'num_timesteps', 0) or 0)
            tw = float(getattr(self, 'blend_tiles_weight', 0.7) or 0.7)
            sw = float(getattr(self, 'blend_steps_weight', 0.3) or 0.3)
            print(
                f"[HybridTuning] tiles={tiles} phase={phase} phase_steps={ep_steps} global_steps={gsteps} "
                f"blend=({tw:.2f},{sw:.2f}) frac_tiles={frac_tiles:.3f} frac_steps={frac_steps:.3f} "
                f"exploration_mult={exploration_mult:.3f} stuck_mult={stuck_mult:.3f}"
            )

        return True

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
    Callback to POST training stats to the overlay server so the overlay can show live training info.
    """
    def __init__(self, url: str = "http://127.0.0.1:8080/update", update_freq: int = 1000, verbose: int = 0):
        super(OverlayCallback, self).__init__(verbose)
        self.url = url
        self.update_freq = int(update_freq)
        self.action_counts = {}
        self.total_actions = 0
        # Global/phase step tracking for overlays
        self._phase_start_timesteps = 0
        self._last_phase = None

    def _on_training_start(self) -> None:
        self.action_counts = {}
        self.total_actions = 0

    def _update_action_counts_from_action(self, action: int):
        """Update action counts using ACTIONS mapping; splits combo actions like UP_A."""
        try:
            from actions import ACTIONS
            name = ACTIONS.get(int(action), 'UNKNOWN')
        except Exception:
            name = 'UNKNOWN'

        if name == 'UNKNOWN':
            return

        # Split combo actions like UP_A into UP + A
        parts = name.split('_') if '_' in name else [name]

        for p in parts:
            key = p.lower()
            if key in ['up', 'down', 'left', 'right', 'a', 'b']:
                self.action_counts[key] = self.action_counts.get(key, 0) + 1
                self.total_actions += 1

    def _on_step(self) -> bool:
        # Update action counts every step if actions are available
        try:
            actions = self.locals.get('actions', None)
            if actions is not None:
                # VecEnv returns array-like actions
                if isinstance(actions, (list, tuple, np.ndarray)):
                    for a in np.array(actions).flatten().tolist():
                        self._update_action_counts_from_action(int(a))
                else:
                    self._update_action_counts_from_action(int(actions))
        except Exception:
            pass

        # Only POST every update_freq callback steps
        if self.n_calls % self.update_freq != 0:
            return True

        # Build action confidence (defaults to zero)
        action_confidence = {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0, 'a': 0.0, 'b': 0.0}
        if self.total_actions > 0:
            for k in action_confidence.keys():
                action_confidence[k] = float(self.action_counts.get(k, 0)) / float(self.total_actions)

        # Pull most recent env info if available (VecEnv -> list of dicts)
        info0 = {}
        try:
            infos = self.locals.get('infos', None)
            if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
                info0 = infos[0]
            elif isinstance(infos, dict):
                info0 = infos
        except Exception:
            info0 = {}

        def _iget(key, default):
            try:
                v = info0.get(key, default)
                return default if v is None else v
            except Exception:
                return default

        # --- Compute global steps and phase/phase_steps ---
        # Global step counter (SB3)
        global_steps = int(getattr(self, 'num_timesteps', 0) or 0)

        # Optional: phase + steps since phase start (restarts when you manually bump TRAINING_PHASE)
        phase = None
        try:
            ph_list = self.training_env.env_method('get_phase')
            if isinstance(ph_list, (list, tuple)) and len(ph_list) > 0:
                phase = int(ph_list[0])
        except Exception:
            phase = None

        if phase is not None and phase != self._last_phase:
            self._last_phase = phase
            self._phase_start_timesteps = global_steps

        phase_steps = global_steps - int(getattr(self, '_phase_start_timesteps', 0) or 0)
        if phase_steps < 0:
            phase_steps = 0
        # --- end compute global/phase steps ---

        map_id = _iget('map_id', None)
        try:
            map_id_int = int(map_id) if map_id is not None else None
        except Exception:
            map_id_int = None

        payload = {
            'stats': {
                'episode_steps': int(_iget('episode_steps', 0)),
                'global_steps': int(global_steps),
                'phase': int(phase) if phase is not None else None,
                'phase_steps': int(phase_steps),
                'tiles_visited': int(_iget('tiles_visited', 0)),
                'battles_won': int(_iget('battles_won', 0)),
                'battles_lost': int(_iget('battles_lost', 0)),
                'badges': int(_iget('badges', 0)),
                'money': int(_iget('money', 0)),
            },
            'commentary': f'Training global_steps {global_steps}',
            'action_confidence': action_confidence,
            'game_state': {
                'map_id': map_id_int,
                'location': _map_name(map_id_int),
                'x': int(_iget('x', 0)),
                'y': int(_iget('y', 0)),
                'in_battle': bool(_iget('in_battle', False)),
                'in_menu': bool(_iget('in_menu', False)),
            }
        }

        try:
            import requests
            requests.post(self.url, json=payload, timeout=1)
        except Exception as e:
            if self.verbose > 0:
                print(f"Overlay update failed: {e}")

        return True

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
