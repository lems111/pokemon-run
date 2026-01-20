"""
Integration smoke tests that run only if PyBoy and a save state/ROM are available.
These tests are non-fatal and will skip if the environment isn't prepared.
Run with: python3 tests/test_integration_with_pyboy.py
"""
import os
import sys

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_map import (
    decode_player_position,
    decode_options,
    decode_badge_count,
    decode_money,
)

SKIP_REASON = None
try:
    from pokemon_yellow_env import PokemonYellowEnv
except Exception as e:
    SKIP_REASON = f"Cannot import PokemonYellowEnv: {e}"

ROM_EXISTS = os.path.exists("pokemon_yellow.gb")
STATE_EXISTS = os.path.exists("states/pallet_town.state")


def run_smoke():
    if SKIP_REASON:
        print(f"SKIPPED: {SKIP_REASON}")
        return
    if not ROM_EXISTS:
        print("SKIPPED: ROM not found at pokemon_yellow.gb")
        return
    if not STATE_EXISTS:
        print("SKIPPED: Save state not found at states/pallet_town.state")
        return

    print("Starting integration smoke test with real emulator and save state...")
    env = PokemonYellowEnv(save_state_path="states/pallet_town.state")
    obs, info = env.reset()
    print("Observation length:", len(obs))

    pos = decode_player_position(obs)
    opts = decode_options(obs)
    badges = decode_badge_count(obs)
    money = decode_money(obs)

    print("Position:", pos)
    print("Options:", opts)
    print("Badges:", badges)
    print("Money:", money)

    print("Integration test complete (no assertions).")


if __name__ == '__main__':
    run_smoke()
