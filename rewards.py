"""
Reward shaping functions for Pokémon Yellow RL training.
"""
import numpy as np
from typing import Tuple, List
from memory_map import decode_player_position, decode_in_battle, decode_badge_count, decode_party_stats

class RewardShaper:
    """
    Handles reward computation and shaping for Pokémon Yellow training.
    """
    
    def __init__(self):
        self.previous_position = None
        self.steps_at_position = 0
        self.tiles_visited = set()
        self.previous_badge_count = 0
        self.previous_battle_wins = 0
        self.previous_battle_losses = 0
        self.total_tiles_visited = 0
        
    def compute_reward(self, ram_data: bytes, action: int, info: dict) -> float:
        """
        Compute reward based on current RAM state and action.
        
        Args:
            ram_data: Current RAM state
            action: Action taken
            info: Episode information
            
        Returns:
            float: Computed reward
        """
        reward = 0.0
        
        # Base penalty for inaction
        reward -= 0.001
        
        # Exploration reward
        exploration_reward = self._compute_exploration_reward(ram_data)
        reward += exploration_reward
        
        # Battle rewards
        battle_reward = self._compute_battle_reward(ram_data)
        reward += battle_reward
        
        # Progress rewards
        progress_reward = self._compute_progress_reward(ram_data)
        reward += progress_reward
        
        # Anti-stuck reward
        anti_stuck_reward = self._compute_anti_stuck_reward(ram_data)
        reward += anti_stuck_reward
        
        return reward
    
    def _compute_exploration_reward(self, ram_data: bytes) -> float:
        """Reward for visiting new tiles."""
        position = decode_player_position(ram_data)
        map_id, x, y = position
        
        # Create tile identifier
        tile_id = (map_id, x, y)
        
        # Reward for new tiles
        if tile_id not in self.tiles_visited:
            self.tiles_visited.add(tile_id)
            self.total_tiles_visited += 1
            return 1.0
            
        return 0.0
    
    def _compute_battle_reward(self, ram_data: bytes) -> float:
        """Reward for battles won/lost."""
        # This would need to track battle state changes
        # For now, return a placeholder
        return 0.0
    
    def _compute_progress_reward(self, ram_data: bytes) -> float:
        """Reward for game progress like badges."""
        current_badge_count = decode_badge_count(ram_data)
        reward = 0.0
        
        # Reward for earning badges
        if current_badge_count > self.previous_badge_count:
            reward += 20.0
            self.previous_badge_count = current_badge_count
            
        return reward
    
    def _compute_anti_stuck_reward(self, ram_data: bytes) -> float:
        """Penalize staying in same position too long."""
        position = decode_player_position(ram_data)
        map_id, x, y = position
        
        # Check if position is the same as previous
        if self.previous_position == (map_id, x, y):
            self.steps_at_position += 1
            # Penalize being stuck
            if self.steps_at_position > 10:
                return -0.1
        else:
            # Reset counter when position changes
            self.steps_at_position = 0
            self.previous_position = (map_id, x, y)
            
        return 0.0
    
    def reset(self, seed=None):
        """Reset reward tracking state."""
        self.previous_position = None
        self.steps_at_position = 0
        self.tiles_visited = set()
        self.previous_badge_count = 0
        self.previous_battle_wins = 0
        self.previous_battle_losses = 0
        self.total_tiles_visited = 0
        # Return empty info dict to match Gymnasium reset signature
        return {}

# Phase-based reward shaping
class PhaseRewardShaper(RewardShaper):
    """
    Reward shaper that adapts based on training phase.
    """
    
    def __init__(self, phase: int = 1):
        super().__init__()
        self.phase = phase
        
        # Phase-specific reward weights
        self.weights = {
            1: {  # Movement sandbox
                'exploration': 1.0,
                'battle_win': 0.0,
                'battle_loss': 0.0,
                'badge': 0.0,
                'stuck': 0.1,
            },
            2: {  # Early route
                'exploration': 1.0,
                'battle_win': 0.5,
                'battle_loss': -1.0,
                'badge': 0.0,
                'stuck': 0.1,
            },
            3: {  # First battles
                'exploration': 0.5,
                'battle_win': 2.0,
                'battle_loss': -2.0,
                'badge': 0.0,
                'stuck': 0.1,
            },
            4: {  # Badge pursuit
                'exploration': 0.3,
                'battle_win': 1.0,
                'battle_loss': -1.0,
                'badge': 5.0,
                'stuck': 0.1,
            }
        }
        
    def compute_reward(self, ram_data: bytes, action: int, info: dict) -> float:
        """Compute reward with phase-specific weights."""
        reward = 0.0
        
        # Base penalty for inaction
        reward -= 0.001
        
        # Exploration reward
        exploration_reward = self._compute_exploration_reward(ram_data)
        reward += exploration_reward * self.weights[self.phase]['exploration']
        
        # Battle rewards
        battle_reward = self._compute_battle_reward(ram_data)
        reward += battle_reward * self.weights[self.phase]['battle_win']
        
        # Progress rewards
        progress_reward = self._compute_progress_reward(ram_data)
        reward += progress_reward * self.weights[self.phase]['badge']
        
        # Anti-stuck reward
        anti_stuck_reward = self._compute_anti_stuck_reward(ram_data)
        reward += anti_stuck_reward * self.weights[self.phase]['stuck']
        
        return reward

# Curriculum-based reward shaping
class CurriculumRewardShaper(RewardShaper):
    """
    Reward shaper that implements curriculum learning.
    """
    
    def __init__(self):
        super().__init__()
        self.current_stage = 1
        self.stage_thresholds = {
            1: 100,  # Movement sandbox
            2: 500,  # Early route
            3: 1000, # First battles
            4: 2000, # Badge pursuit
        }
        
    def update_stage(self, tiles_visited: int):
        """Update training stage based on progress."""
        for stage, threshold in self.stage_thresholds.items():
            if tiles_visited >= threshold:
                self.current_stage = stage
            else:
                break
                
    def compute_reward(self, ram_data: bytes, action: int, info: dict) -> float:
        """Compute reward with curriculum-based weights."""
        # This would be implemented based on current stage
        return super().compute_reward(ram_data, action, info)
