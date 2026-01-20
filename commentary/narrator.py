"""
Narrator module that builds prompts for LLM commentary generation.
"""
import time
from typing import Dict, List, Optional
from memory_map import decode_player_position, decode_badge_count, decode_party_stats, decode_in_battle, decode_menu_state
from actions import get_action_name

class Narrator:
    """
    Builds structured prompts for LLM commentary generation.
    """
    
    def __init__(self):
        self.last_commentary_time = 0
        self.commentary_interval = 5  # seconds between commentaries
        self.event_queue = []
        
    def build_prompt(self, 
                    game_state: Dict,
                    telemetry: Dict,
                    action_probs: List[float] = None,
                    recent_actions: List[int] = None) -> str:
        """
        Build a structured prompt for the LLM.
        
        Args:
            game_state: Current game state information
            telemetry: Telemetry data from the environment
            action_probs: Probability distribution over actions
            recent_actions: Recent actions taken
            
        Returns:
            str: Formatted prompt for LLM
        """
        # Format game state
        location = self._format_location(game_state.get('position', (0, 0, 0)))
        recent_events = self._format_recent_events(telemetry.get('events', []))
        current_status = self._format_current_status(telemetry)
        action_distribution = self._format_action_distribution(action_probs, recent_actions)
        goal_metrics = self._format_goal_metrics(telemetry.get('metrics', {}))
        
        # Build prompt
        prompt = f"""
You are a Pokémon Yellow game commentator. Provide natural-language commentary for a stream.
Keep responses to 1-2 sentences maximum, and use a Twitch-friendly tone.

Current game state:
{location}
{current_status}
{recent_events}

Action distribution (last 50 steps):
{action_distribution}

Goal metrics:
{goal_metrics}

Provide commentary that fits this context. Focus on:
- What the agent is doing right now
- Progress toward goals
- Interesting events
- Game state changes

Commentary:
"""
        
        return prompt.strip()
    
    def _format_location(self, position: tuple) -> str:
        """Format player position information."""
        if not position:
            return "Location: Unknown"
            
        map_id, x, y = position
        # Convert map_id to a readable name (this would be expanded in a real implementation)
        map_name = self._get_map_name(map_id)
        return f"Location: {map_name}, x={x} y={y}"
    
    def _get_map_name(self, map_id: int) -> str:
        """Get readable name for map ID."""
        # This would be expanded with actual map names
        map_names = {
            0: "Pallet Town",
            1: "Viridian Forest",
            2: "Pewter City",
            3: "Mt. Moon",
            4: "Cerulean City",
            5: "Route 1",
            6: "Route 2",
            7: "Route 3",
            # Add more map names as needed
        }
        return map_names.get(map_id, f"Map {map_id}")
    
    def _format_recent_events(self, events: List[Dict]) -> str:
        """Format recent game events."""
        if not events:
            return "Recent: No events"
            
        event_descriptions = []
        for event in events[-5:]:  # Show last 5 events
            event_type = event.get('type', 'unknown')
            event_desc = event.get('description', '')
            event_descriptions.append(f"- {event_type}: {event_desc}")
            
        return "Recent events:\n" + "\n".join(event_descriptions)
    
    def _format_current_status(self, telemetry: Dict) -> str:
        """Format current game status."""
        in_battle = telemetry.get('in_battle', False)
        in_menu = telemetry.get('in_menu', False)
        battle_won = telemetry.get('battle_won', 0)
        battle_lost = telemetry.get('battle_lost', 0)
        
        status_parts = []
        status_parts.append(f"in_menu={in_menu}")
        status_parts.append(f"in_battle={in_battle}")
        status_parts.append(f"battles_won={battle_won}")
        status_parts.append(f"battles_lost={battle_lost}")
        
        return "Current status: " + ", ".join(status_parts)
    
    def _format_action_distribution(self, action_probs: List[float], recent_actions: List[int]) -> str:
        """Format action distribution."""
        if not action_probs:
            return "No action data available"
            
        # Get top actions
        action_indices = sorted(range(len(action_probs)), key=lambda i: action_probs[i], reverse=True)
        top_actions = action_indices[:5]  # Top 5 actions
        
        action_strings = []
        for i, action_idx in enumerate(top_actions):
            prob = action_probs[action_idx]
            action_name = get_action_name(action_idx)
            action_strings.append(f"{action_name} ({prob:.2%})")
            
        return ", ".join(action_strings)
    
    def _format_goal_metrics(self, metrics: Dict) -> str:
        """Format goal metrics."""
        if not metrics:
            return "Goal metrics: None"
            
        metric_strings = []
        for key, value in metrics.items():
            metric_strings.append(f"{key}={value}")
            
        return "Goal metrics: " + ", ".join(metric_strings)
    
    def should_generate_commentary(self, current_time: float = None) -> bool:
        """
        Check if enough time has passed to generate new commentary.
        
        Args:
            current_time: Current timestamp (optional, uses time.time() if not provided)
            
        Returns:
            bool: True if commentary should be generated
        """
        if current_time is None:
            current_time = time.time()
            
        if current_time - self.last_commentary_time > self.commentary_interval:
            self.last_commentary_time = current_time
            return True
            
        return False
    
    def add_event(self, event_type: str, description: str):
        """Add an event to the event queue."""
        self.event_queue.append({
            'type': event_type,
            'description': description,
            'timestamp': time.time()
        })
        
        # Keep only last 20 events
        if len(self.event_queue) > 20:
            self.event_queue = self.event_queue[-20:]

# Example usage
def example_usage():
    """Example of how to use the narrator."""
    narrator = Narrator()
    
    # Example game state
    game_state = {
        'position': (0, 10, 15),  # map_id, x, y
    }
    
    # Example telemetry
    telemetry = {
        'events': [
            {'type': 'entered_map', 'description': 'Entered Pallet Town'},
            {'type': 'visited_tile', 'description': 'Visited new tile (0, 10, 15)'},
        ],
        'in_battle': False,
        'in_menu': False,
        'battle_won': 0,
        'battle_lost': 0,
        'metrics': {
            'tiles_visited': 42,
            'badges': 0,
            'money': 150,
        }
    }
    
    # Example action probabilities
    action_probs = [0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    
    # Build prompt
    prompt = narrator.build_prompt(game_state, telemetry, action_probs)
    print("Generated prompt:")
    print(prompt)

if __name__ == "__main__":
    example_usage()
