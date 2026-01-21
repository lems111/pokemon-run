
import os
import sys
import time
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(project_root)

from pokemon_yellow_env import PokemonYellowEnv
from memory_map import decode_player_position

load_dotenv()

# Helper: find an action id by its name from actions.ACTIONS mapping

def _find_action_id(name: str, default: int) -> int:
    try:
        from actions import ACTIONS
        for k, v in ACTIONS.items():
            if str(v).upper() == name.upper():
                return int(k)
    except Exception:
        pass
    return int(default)


def main():
    env = PokemonYellowEnv()

    # Make debug printing very frequent for this manual test
    env.debug_pos = True
    env.debug_pos_every = 1

    # Resolve action ids (fallbacks are best-guess; your actions.py is source of truth)
    ACT_RIGHT = _find_action_id("RIGHT", 3)
    ACT_LEFT = _find_action_id("LEFT", 2)
    ACT_UP = _find_action_id("UP", 0)
    ACT_DOWN = _find_action_id("DOWN", 1)
    ACT_A = _find_action_id("A", 4)
    ACT_B = _find_action_id("B", 5)
    ACT_START = _find_action_id("START", 7)

    try:
        from actions import ACTIONS
        def an(a: int) -> str:
            return str(ACTIONS.get(int(a), a))
    except Exception:
        def an(a: int) -> str:
            return str(a)

    # Sequence:
    # 1) Try dismissing any dialogs/menus (B/A/START)
    # 2) Try moving RIGHT then DOWN
    # 3) Then idle in a loop so you can press keys and keep the window open
    seq = []
    seq += [ACT_B] * 20
    seq += [ACT_A] * 10
    seq += [ACT_START] * 5
    seq += [ACT_B] * 10
    seq += [ACT_RIGHT] * 120
    seq += [ACT_DOWN] * 120
    seq += [ACT_LEFT] * 120
    seq += [ACT_UP] * 120

    last = None

    print("\nDone with scripted inputs. Window will stay open; press Ctrl+C to quit.\n")

    # Keep window open and emulator running so you can press keys and observe.
    while True:
        env.emulator.tick(1, env.shouldRender)
        ram = env._get_wram_c000_dfff()
        cands = decode_player_position(ram)
        cur = cands.get('current', {})
        p1 = cands.get('plus1', {})
        m1 = cands.get('minus1', {})

        tup = (
            cur.get('map_id'), cur.get('x'), cur.get('y'),
            p1.get('map_id'), p1.get('x'), p1.get('y'),
            m1.get('map_id'), m1.get('x'), m1.get('y'),
        )

        if tup != last:
            print(
                f"cur=({cur.get('map_id')},{cur.get('x')},{cur.get('y')}) "
                f"+1=({p1.get('map_id')},{p1.get('x')},{p1.get('y')}) "
                f"-1=({m1.get('map_id')},{m1.get('x')},{m1.get('y')})"
            )
            last = tup

        time.sleep(0.01)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        try:
            env = locals().get('env')
            if env is not None:
                env._release_all_buttons()
        except Exception:
            pass
    finally:
        try:
            # Be defensive: env might not exist if init failed
            pass
        except Exception:
            pass
