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
from train.callbacks import CheckpointCallback, TensorboardCallback, OverlayCallback
from memory_map import get_raw_ram_slice, get_observation_vector
from rewards import PhaseRewardShaper
from actions import ActionHandler

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
        self.reward_shaper = PhaseRewardShaper()
        self.model = None
        self.env = None
        self.vec_env = None
        
        # Training stats
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.tiles_visited = deque(maxlen=100)
        
    def create_env(self):
        """Create vectorized environment."""
        # Create a function that returns a new instance of the environment
        def make_env():
            return PokemonYellowEnv()
        
        # For now, use a simple vectorized environment without multiprocessing
        # This avoids serialization issues with complex PyBoy environments
        self.vec_env = make_vec_env(make_env, n_envs=self.num_envs)
        
        # Normalize observations
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
        overlay_callback = OverlayCallback(url=os.environ.get('OVERLAY_URL', 'http://127.0.0.1:8080/update'),
                                           update_freq=int(os.environ.get('OVERLAY_UPDATE_FREQ', 1000)),
                                           verbose=0)
        
        # Set the environment in overlay callback for action tracking
        overlay_callback.set_env(self.vec_env)

        # Train the model
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[checkpoint_callback, tensorboard_callback, overlay_callback],
            tb_log_name="ppo_training"
        )
        
        
        # Note: The OverlayCallback currently doesn't update stats during training
        # This would require a more complex implementation to track stats from environment
        # For now, we're providing a complete structure that can be extended
        
        print("Training completed!")
        
        # Save final model
        final_model_path = os.path.join(self.model_save_path, "final_model.zip")
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        return model
    
    def evaluate_model(self, model_path: str):
        """Evaluate a trained model."""
        print(f"Evaluating model from {model_path}")
        
        # Load the model
        model = PPO.load(model_path)
        
        # Create environment
        env = self.create_env()
        
        # Run evaluation
        obs = env.reset()
        total_reward = 0
        episode_count = 0
        
        try:
            for i in range(1000):  # Run for 1000 steps
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                if done:
                    episode_count += 1
                    obs = env.reset()
                    print(f"Episode {episode_count} reward: {total_reward}")
                    total_reward = 0
                    
        except KeyboardInterrupt:
            print("Evaluation interrupted")
            
        print(f"Average reward: {total_reward / max(episode_count, 1)}")

def main():
    """Main training function."""
    load_dotenv()  # Load environment variables from .env file if present
    # Set random seeds for reproducibility
    set_random_seed(42)
    
    # Initialize trainer
    trainer = PPOTrainer(
        model_save_path="runs/checkpoints",
        log_path="runs/tensorboard",
        num_envs=4,
        total_timesteps=1000000
    )
    
    # Start training
    model = trainer.train()
    
    # Evaluate the model
    trainer.evaluate_model("runs/checkpoints/final_model.zip")

if __name__ == "__main__":
    main()
