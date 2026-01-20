from pokemon_yellow_env import PokemonYellowEnv
from memory_map import RAM_ADDRESSES

env = PokemonYellowEnv.__new__(PokemonYellowEnv)
env.episode_steps = 0
env.tiles_visited = set()
env.last_position = None
env._prev_in_battle = True
env._prev_badges = 0
env._prev_money = 0
env._prev_party_hp = [100.0] * 6
env._prev_battle_type = 1
env._last_battle_result = None
env.battles_won = 0
env.battles_lost = 0

# Simulate observation after battle: not in battle, party HP dropped significantly
raw_after = bytearray(0x1000)
raw_after[RAM_ADDRESSES['IN_BATTLE_FLAG']] = 0
raw_after[RAM_ADDRESSES['BATTLE_TYPE']] = 0
cur_base = RAM_ADDRESSES['PARTY_CURRENT_HP']
max_base = RAM_ADDRESSES['PARTY_MAX_HP']
raw_after[RAM_ADDRESSES['PARTY_COUNT']] = 1
raw_after[cur_base] = 1  # cur hp low
raw_after[cur_base + 1] = 0
raw_after[max_base] = 100
raw_after[max_base + 1] = 0

env._get_observation = lambda: raw_after
r = env._compute_reward(0)
print('reward', r)
print('_last_battle_result', env._last_battle_result)
print('battles_lost', env.battles_lost)
print('_prev_in_battle', env._prev_in_battle)
print('_prev_opponent_roster_count', getattr(env, '_prev_opponent_roster_count', None))
print('prev_party_hp', env._prev_party_hp)

# Also compute raw derived values
from memory_map import decode_in_battle, decode_party_stats, decode_badge_count, decode_money, get_byte
ram = env._get_observation()
print('decode_in_battle', bool(decode_in_battle(ram)))
print('party_stats', decode_party_stats(ram))
print('badges', decode_badge_count(ram))
print('money', decode_money(ram))
try:
    print('opponent_roster', get_byte(ram, RAM_ADDRESSES['OPPONENT_ROSTER_COUNT']))
except Exception as e:
    print('opponent_roster read failed', e)
