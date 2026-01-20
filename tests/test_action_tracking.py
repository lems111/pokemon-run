"""
Test script to verify action tracking in Pokémon Yellow RL environment.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pokemon_yellow_env import PokemonYellowEnv
from actions import ACTIONS

def test_action_tracking():
    """Test that actions are properly tracked in the environment."""
    print("Testing action tracking in Pokémon Yellow environment...")
    
    # Create environment
    env = PokemonYellowEnv()
    
    # Check initial state
    print(f"Initial action counts: {env.action_counts}")
    print(f"Initial total actions: {env.total_actions}")
    
    # Test a few actions
    test_actions = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]  # Test some basic actions
    
    for i, action in enumerate(test_actions):
        print(f"Step {i}: Taking action {action} ({ACTIONS[action]})")
        
        # Reset for each test to avoid state interference
        obs, info = env.reset()
        
        # Execute action
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"  Action counts: {env.action_counts}")
        print(f"  Total actions: {env.total_actions}")
        
        # Verify action was recorded
        action_name = ACTIONS[action]
        if action_name in env.action_counts:
            print(f"  ✓ Action {action_name} was recorded {env.action_counts[action_name]} times")
        else:
            print(f"  ✗ Action {action_name} was NOT recorded")
    
    print("\nAction tracking test completed!")

if __name__ == "__main__":
    test_action_tracking()
