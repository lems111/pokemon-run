"""
PPO training loop for Pokémon Yellow RL agent.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from typing import Optional
from dotenv import load_dotenv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common import logger

# Add current directory to Python path to find pokemon_yellow_env
project_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(project_root)

from pokemon_yellow_env import PokemonYellowEnv
from train.callbacks import CheckpointCallback, TensorboardCallback, OverlayCallback, HybridTuningCallback

class PPOTrainer:
    """
    PPO trainer for Pokémon Yellow RL agent.
    """
    
    def __init__(self, 
                 env_id: str = "PokemonYellow-v0",
                 model_save_path: str = "runs/checkpoints",
                 log_path: str = "runs/tensorboard",
                 num_envs: int = 4,
                 total_timesteps: int = 1000000):
        
        self.env_id = env_id
        self.model_save_path = model_save_path
        self.log_path = log_path
        self.num_envs = num_envs
        self.total_timesteps = total_timesteps
        
        # Create directories
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.env = None
        self.vec_env = None
        
        # Training stats
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.tiles_visited = deque(maxlen=100)
        
    def create_env(self, n_envs: Optional[int] = None, vec_normalize: Optional[bool] = None):
        """Create vectorized environment.

        Args:
            n_envs: Override number of parallel envs (defaults to self.num_envs).
            vec_normalize: Override VecNormalize usage (defaults to env var VEC_NORMALIZE).
        """
        def make_env():
            phase = int(os.getenv('TRAINING_PHASE', '1'))
            return PokemonYellowEnv(phase=phase)

        env_count = int(n_envs) if n_envs is not None else int(self.num_envs)
        self.vec_env = make_vec_env(make_env, n_envs=env_count)

        do_norm = vec_normalize
        if do_norm is None:
            do_norm = os.getenv('VEC_NORMALIZE', 'true').lower() == 'true'

        if do_norm:
            self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        return self.vec_env
    
    def create_model(self):
        """Create PPO model."""
        # For now, we'll use stable-baselines3 PPO
        # In a real implementation, this would be more customized
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            verbose=1,
            tensorboard_log=self.log_path,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            normalize_advantage=True,
            target_kl=None,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
        
        return self.model
    
    def train(self):
        """Run training loop."""
        print("Starting training...")
        
        # Create environment and model
        env = self.create_env()
        model = self.create_model()
        
        # Create callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=self.model_save_path,
            name_prefix="ppo"
        )
        
        tensorboard_callback = TensorboardCallback()
        callbacks = [checkpoint_callback, tensorboard_callback]
        if os.getenv("OVERLAY_ENABLE", "false").lower() == "true":
            overlay_callback = OverlayCallback(url=os.environ.get('OVERLAY_URL', 'http://127.0.0.1:8080/update'),
                                            update_freq=int(os.environ.get('OVERLAY_UPDATE_FREQ', 1000)),
                                            verbose=0)
            
            # Set the environment in overlay callback for action tracking
            callbacks.append(overlay_callback)

        if os.getenv('HYBRID_TUNING', 'true').lower() == 'true':
            hybrid_cb = HybridTuningCallback(
                update_freq=int(os.getenv('TUNING_UPDATE_FREQ', '2000')),
                target_tiles=int(os.getenv('TUNING_TARGET_TILES', '400')),
                target_steps=int(os.getenv('TUNING_TARGET_STEPS', '5000')),
                min_exploration_mult=float(os.getenv('TUNING_MIN_EXPLORATION_MULT', '0.30')),
                max_exploration_mult=float(os.getenv('TUNING_MAX_EXPLORATION_MULT', '1.00')),
                verbose=int(os.getenv('TUNING_VERBOSE', '0')),
            )
            callbacks.append(hybrid_cb)

        # Train the model
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
            tb_log_name="ppo_training"
        )
        
        print("Training completed!")
        
        # Save final model
        final_model_path = os.path.join(self.model_save_path, "final_model.zip")
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Save VecNormalize statistics (needed for correct evaluation/inference)
        try:
            if isinstance(self.vec_env, VecNormalize):
                vecnorm_path = os.path.join(self.model_save_path, "vecnormalize.pkl")
                self.vec_env.save(vecnorm_path)
                print(f"VecNormalize stats saved to {vecnorm_path}")
        except Exception:
            pass

        try:
            if self.vec_env is not None:
                self.vec_env.close()
        except Exception:
            pass

        return model
    
    def evaluate_model(self, model_path: str, n_eval_steps: int = 20000):
        """Evaluate a trained model in a vectorized env (multi-env).

        Handles SB3 VecEnv outputs where rewards/dones are arrays of shape (n_envs,).
        If VecNormalize was used during training, loads the saved stats file.
        """
        print(f"Evaluating model from {model_path}")

        model = PPO.load(model_path)

        # Create a non-normalized VecEnv first; then load VecNormalize stats if available.
        env = self.create_env(
            n_envs=int(os.getenv("NUM_ENVS_EVAL", str(self.num_envs))),
            vec_normalize=False
        )

        vecnorm_path = os.path.join(self.model_save_path, "vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            try:
                env = VecNormalize.load(vecnorm_path, env)
            except Exception:
                pass

        if isinstance(env, VecNormalize):
            env.training = False
            env.norm_reward = False

        obs = env.reset()

        n_envs = int(getattr(env, "num_envs", self.num_envs))
        ep_rewards = np.zeros(n_envs, dtype=np.float64)
        ep_lengths = np.zeros(n_envs, dtype=np.int64)

        completed_rewards = []
        completed_lengths = []

        try:
            for _ in range(int(n_eval_steps)):
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)

                rewards = np.asarray(rewards, dtype=np.float64)
                dones = np.asarray(dones, dtype=bool)

                ep_rewards += rewards
                ep_lengths += 1

                # When a sub-env is done, SB3 auto-resets that sub-env internally.
                for i in range(n_envs):
                    if bool(dones[i]):
                        r = float(ep_rewards[i])
                        l = int(ep_lengths[i])
                        completed_rewards.append(r)
                        completed_lengths.append(l)

                        print(f"[eval] env={i} ep={len(completed_rewards)} reward={r:.3f} length={l}")

                        ep_rewards[i] = 0.0
                        ep_lengths[i] = 0

        except KeyboardInterrupt:
            print("Evaluation interrupted")
        finally:
            try:
                env.close()
            except Exception:
                pass

        if completed_rewards:
            mean_r = float(np.mean(completed_rewards))
            std_r = float(np.std(completed_rewards))
            mean_l = float(np.mean(completed_lengths))
            std_l = float(np.std(completed_lengths))

            print(f"Completed episodes: {len(completed_rewards)}")
            print(f"Mean episode reward: {mean_r:.3f} +/- {std_r:.3f}")
            print(f"Mean episode length: {mean_l:.1f} +/- {std_l:.1f}")
        else:
            print("No complete episodes finished during evaluation window.")
            print(f"Partial mean reward per env: {float(np.mean(ep_rewards)):.3f}")

def main():
    """Main training function."""
    load_dotenv()  # Load environment variables from .env file if present
    # Set random seeds for reproducibility
    set_random_seed(42)
    
    # Initialize trainer
    trainer = PPOTrainer(
        model_save_path="runs/checkpoints",
        log_path="runs/tensorboard",
        num_envs=int(os.environ.get('NUM_ENVS', 4)),
        total_timesteps=1000000
    )
    
    # Start training
    model = trainer.train()
    
    # Evaluate the model
    trainer.evaluate_model(
        "runs/checkpoints/final_model.zip",
        n_eval_steps=int(os.getenv("EVAL_STEPS", "20000")),
    )

if __name__ == "__main__":
    main()
