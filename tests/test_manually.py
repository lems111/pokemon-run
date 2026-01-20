import os
import sys

# Add current directory to Python path to find pokemon_yellow_env
project_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(project_root)
from pokemon_yellow_env import PokemonYellowEnv
from memory_map import (
    decode_player_position, decode_badge_count, decode_party_stats, get_observation_vector,
    pretty_print_battle, type_id_to_name, move_id_to_name, parse_player_name
)
from pyboy import PyBoy

# Initialize PyBoy with your ROM
env = PokemonYellowEnv()

try:
    # Keep the emulator running (e.g., for 1000 ticks)
    while env.emulator.tick():
        ram_data = env._get_observation()
        position = decode_player_position(ram_data)
    env.emulator.stop()
except KeyboardInterrupt:
    print("Evaluation interrupted")
    env.emulator.stop()
