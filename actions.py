"""
Action mappings and handling for Pokémon Yellow RL training.
"""
from typing import List, Tuple
import numpy as np

# Action mappings (buttons)
ACTIONS = {
    0: 'UP',
    1: 'DOWN', 
    2: 'LEFT',
    3: 'RIGHT',
    4: 'A',
    5: 'B',
    6: 'SELECT',
    7: 'START',
    8: 'UP_A',
    9: 'DOWN_A',
    10: 'LEFT_A',
    11: 'RIGHT_A'
}

# Action combinations for more complex movements
ACTION_COMBINATIONS = {
    'UP': ['UP'],
    'DOWN': ['DOWN'],
    'LEFT': ['LEFT'],
    'RIGHT': ['RIGHT'],
    'UP_A': ['UP', 'A'],
    'DOWN_A': ['DOWN', 'A'],
    'LEFT_A': ['LEFT', 'A'],
    'RIGHT_A': ['RIGHT', 'A'],
    'A': ['A'],
    'B': ['B'],
    'SELECT': ['SELECT'],
    'START': ['START'],
    'UP_DOWN': ['UP', 'DOWN'],  # This would be invalid, but shown for completeness
    'LEFT_RIGHT': ['LEFT', 'RIGHT'],  # This would be invalid, but shown for completeness
}

class ActionHandler:
    """
    Handles action processing for Pokémon Yellow.
    """
    
    def __init__(self, action_repeat: int = 4, frame_skip: int = 4, sticky_buttons: bool = True):
        self.action_repeat = action_repeat
        self.frame_skip = frame_skip
        self.sticky_buttons = sticky_buttons
        self.previous_action = None
        self.sticky_counter = 0
        
    def get_action_buttons(self, action_index: int) -> List[str]:
        """
        Get button presses for a given action index.
        
        Args:
            action_index: Index of action (0-11)
            
        Returns:
            List of button names to press
        """
        if action_index < 0 or action_index >= len(ACTIONS):
            raise ValueError(f"Invalid action index: {action_index}")
            
        action_name = ACTIONS[action_index]
        return ACTION_COMBINATIONS.get(action_name, [action_name])
        
    def get_button_sequence(self, action_index: int) -> List[str]:
        action_name = ACTIONS[action_index]

        if action_name in ['UP_A', 'DOWN_A', 'LEFT_A', 'RIGHT_A']:
            direction = action_name.split('_', 1)[0]   # "UP_A" -> "UP"
            return [direction, 'A']
        else:
            return [action_name]
    
    def process_action(self, action_index: int) -> List[str]:
        """
        Process action with repetition and sticky button handling.
        
        Args:
            action_index: Index of action (0-11)
            
        Returns:
            List of button presses to execute
        """
        buttons = self.get_action_buttons(action_index)
        
        # Apply action repeat
        repeated_buttons = []
        for _ in range(self.action_repeat):
            repeated_buttons.extend(buttons)
            
        return repeated_buttons
    
    def get_sticky_buttons(self, action_index: int) -> List[str]:
        """
        Get buttons with sticky handling.
        
        Args:
            action_index: Index of action (0-11)
            
        Returns:
            List of button presses with sticky handling
        """
        if not self.sticky_buttons:
            return self.process_action(action_index)
            
        # If same action as previous, continue holding
        if self.previous_action == action_index:
            self.sticky_counter += 1
            if self.sticky_counter > 2:  # Hold for at least 2 frames
                # Continue holding the same action
                return self.process_action(action_index)
        else:
            # New action, reset counter
            self.previous_action = action_index
            self.sticky_counter = 0
            
        return self.process_action(action_index)
    
    def get_action_vector(self, action_index: int) -> np.ndarray:
        """
        Convert action index to a one-hot vector.
        
        Args:
            action_index: Index of action (0-11)
            
        Returns:
            One-hot encoded action vector
        """
        vector = np.zeros(len(ACTIONS), dtype=np.float32)
        vector[action_index] = 1.0
        return vector
    
    def get_action_distribution(self, action_probs: np.ndarray) -> int:
        """
        Sample action from probability distribution.
        
        Args:
            action_probs: Probability distribution over actions
            
        Returns:
            Selected action index
        """
        return np.random.choice(len(ACTIONS), p=action_probs)
    
    def is_movement_action(self, action_index: int) -> bool:
        """
        Check if action is a movement action.
        
        Args:
            action_index: Index of action (0-11)
            
        Returns:
            True if action is a movement action
        """
        movement_actions = [0, 1, 2, 3, 8, 9, 10, 11]  # UP, DOWN, LEFT, RIGHT, and their combinations
        return action_index in movement_actions

# Action sequence generator for curriculum learning
class ActionSequenceGenerator:
    """
    Generates action sequences for different training stages.
    """
    
    def __init__(self):
        self.sequences = {
            1: self._movement_sandbox_sequence,
            2: self._early_route_sequence,
            3: self._first_battles_sequence,
            4: self._badge_pursuit_sequence,
        }
    
    def generate_sequence(self, stage: int, steps: int) -> List[int]:
        """
        Generate a sequence of actions for a given training stage.
        
        Args:
            stage: Training stage (1-4)
            steps: Number of steps to generate
            
        Returns:
            List of action indices
        """
        if stage not in self.sequences:
            raise ValueError(f"Unknown stage: {stage}")
            
        return self.sequences[stage](steps)
    
    def _movement_sandbox_sequence(self, steps: int) -> List[int]:
        """Generate sequence for movement sandbox stage."""
        # Simple random movement for exploration
        return [np.random.randint(0, 4) for _ in range(steps)]
    
    def _early_route_sequence(self, steps: int) -> List[int]:
        """Generate sequence for early route stage."""
        # Mix of movement and interaction
        actions = []
        for i in range(steps):
            if i % 10 == 0:  # Every 10 steps, try to interact
                actions.append(np.random.choice([4, 5, 6, 7]))  # A, B, SELECT, START
            else:
                actions.append(np.random.randint(0, 4))  # Movement
        return actions
    
    def _first_battles_sequence(self, steps: int) -> List[int]:
        """Generate sequence for first battles stage."""
        # Mix of movement and battle actions
        actions = []
        for i in range(steps):
            if i % 15 == 0:  # Every 15 steps, try to battle
                actions.append(np.random.choice([4, 5]))  # A or B for battle
            else:
                actions.append(np.random.randint(0, 4))  # Movement
        return actions
    
    def _badge_pursuit_sequence(self, steps: int) -> List[int]:
        """Generate sequence for badge pursuit stage."""
        # More strategic actions focused on progression
        actions = []
        for i in range(steps):
            if i % 20 == 0:  # Every 20 steps, try to interact with gym
                actions.append(np.random.choice([4, 5, 6, 7]))  # A, B, SELECT, START
            else:
                actions.append(np.random.randint(0, 4))  # Movement
        return actions

# Utility functions for action analysis
def analyze_action_distribution(actions: List[int]) -> dict:
    """
    Analyze distribution of actions taken.
    
    Args:
        actions: List of action indices
        
    Returns:
        Dictionary with action distribution statistics
    """
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    total = len(actions)
    action_probs = {action: count/total for action, count in action_counts.items()}
    
    return {
        'counts': action_counts,
        'probabilities': action_probs,
        'most_frequent': max(action_counts.items(), key=lambda x: x[1]),
        'total_actions': total
    }

def get_action_name(action_index: int) -> str:
    """
    Get action name from index.
    
    Args:
        action_index: Index of action (0-11)
        
    Returns:
        Name of action
    """
    return ACTIONS.get(action_index, "UNKNOWN")

def get_action_index(action_name: str) -> int:
    """
    Get action index from name.
    
    Args:
        action_name: Name of action
        
    Returns:
        Index of action
    """
    for index, name in ACTIONS.items():
        if name == action_name:
            return index
    return -1
