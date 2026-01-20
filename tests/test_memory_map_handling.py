"""
Simple tests to validate memory-map handling and observation extraction.
Run with: python -m pytest tests/test_memory_map_handling.py (if you use pytest)
Or run directly: python tests/test_memory_map_handling.py
"""
import sys
import os
from types import SimpleNamespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_map import (
    RAM_ADDRESSES,
    decode_player_position,
    decode_money,
    decode_badge_count,
    decode_party_stats,
)

from pokemon_yellow_env import PokemonYellowEnv


def test_decode_helpers():
    # Create a fake WRAM slice (0xD000..0xDFFF => 0x1000 bytes)
    ram = bytearray(0x1000)

    # Player position
    ram[RAM_ADDRESSES['PLAYER_MAP_ID']] = 7
    ram[RAM_ADDRESSES['PLAYER_MAP_X']] = 12
    ram[RAM_ADDRESSES['PLAYER_MAP_Y']] = 34
    assert decode_player_position(ram) == (7, 12, 34)

    # Money (3-byte little endian)
    start = RAM_ADDRESSES['MONEY']
    ram[start:start+3] = bytes([0x12, 0x34, 0x56])
    assert decode_money(ram) == (0x12 | (0x34 << 8) | (0x56 << 16))

    # Badges count (bit count)
    ram[RAM_ADDRESSES['BADGES']] = 0b10101010
    assert decode_badge_count(ram) == bin(0b10101010).count('1')

    # Party stats (current/max hp little-endian)
    ram[RAM_ADDRESSES['PARTY_COUNT']] = 2
    cur_off = RAM_ADDRESSES['PARTY_CURRENT_HP']
    max_off = RAM_ADDRESSES['PARTY_MAX_HP']
    # First mon: cur=50, max=100
    ram[cur_off] = 50 & 0xFF
    ram[cur_off+1] = 0
    ram[max_off] = 100 & 0xFF
    ram[max_off+1] = 0
    stats = decode_party_stats(ram)
    assert abs(stats[0] - 0.5) < 1e-6


class FakeEmu:
    def get_memory_value(self, addr):
        # Simple deterministic pseudo-memory: low byte of address
        return addr & 0xFF


def test_env_observation_mapping():
    # Create minimal env instance without initializing PyBoy
    env = PokemonYellowEnv.__new__(PokemonYellowEnv)
    env.emulator = FakeEmu()

    obs = env._get_observation()
    assert len(obs) == 0x1000

    # Check that value at PLAYER_MAP_ID offset corresponds to WRAM address 0xD000 + offset
    offset = RAM_ADDRESSES['PLAYER_MAP_ID']
    expected = FakeEmu().get_memory_value(0xD000 + offset)
    assert obs[offset] == expected


def test_to_d000_slice_from_c000():
    # Simulate an 8KB WRAM dump starting at 0xC000
    raw = bytearray(0x2000)
    # Place a marker at the absolute D000+PLAYER_MAP_ID address (which is index 0x1000 + offset in this buffer)
    raw[0x1000 + RAM_ADDRESSES['PLAYER_MAP_ID']] = 0xAB
    from memory_map import to_d000_slice

    d000 = to_d000_slice(raw)
    assert len(d000) == 0x1000
    assert d000[RAM_ADDRESSES['PLAYER_MAP_ID']] == 0xAB


def test_to_d000_slice_with_base_address():
    # Simulate a buffer where index 0 corresponds to 0xD000 directly
    raw = bytearray(0x1000)
    raw[RAM_ADDRESSES['PLAYER_MAP_ID']] = 0x12
    from memory_map import to_d000_slice

    d000 = to_d000_slice(raw, base_address=0xD000)
    assert len(d000) == 0x1000
    assert d000[RAM_ADDRESSES['PLAYER_MAP_ID']] == 0x12


def test_decoders_handle_c000_dump():
    # Simulate an 8KB dump starting at 0xC000
    raw = bytearray(0x2000)
    # Put values at addresses relative to D000 window
    raw[0x1000 + RAM_ADDRESSES['PLAYER_MAP_ID']] = 0x22
    raw[0x1000 + RAM_ADDRESSES['BADGES']] = 0b11110000
    raw[0x1000 + RAM_ADDRESSES['MONEY']] = 0x01
    raw[0x1000 + RAM_ADDRESSES['MONEY'] + 1] = 0x02
    raw[0x1000 + RAM_ADDRESSES['MONEY'] + 2] = 0x03

    from memory_map import (
        decode_player_position,
        decode_badge_count,
        decode_money
    )

    pos = decode_player_position(raw)
    assert pos[0] == 0x22
    assert decode_badge_count(raw) == bin(0b11110000).count('1')
    assert decode_money(raw) == (0x01 | (0x02 << 8) | (0x03 << 16))


if __name__ == '__main__':
    # Run simple checks when executed directly
    test_decode_helpers()
    print("✓ memory_map helper tests passed")
    test_env_observation_mapping()
    print("✓ environment WRAM mapping test passed")
    test_to_d000_slice_from_c000()
    print("✓ to_d000_slice from 0xC000 test passed")
    test_to_d000_slice_with_base_address()
    print("✓ to_d000_slice with base address test passed")
    test_decoders_handle_c000_dump()
    print("✓ decoders accept 0xC000 dump test passed")

    # Battle outcome detection tests (win/loss heuristics)
    def test_battle_win_detection():
        env = PokemonYellowEnv.__new__(PokemonYellowEnv)
        env.episode_steps = 0
        env.tiles_visited = set()
        env.last_position = None
        env._prev_in_battle = True
        env._prev_badges = 0
        env._prev_money = 0
        env._prev_party_hp = [50.0] * 6
        env._prev_battle_type = 1
        env._last_battle_result = None
        env.battles_won = 0
        env.battles_lost = 0

        # Simulate observation after battle: not in battle, badges increased
        raw_after = bytearray(0x1000)
        # Set IN_BATTLE_FLAG (D057) to 0, BATTLE_TYPE to 0
        raw_after[RAM_ADDRESSES['IN_BATTLE_FLAG']] = 0
        raw_after[RAM_ADDRESSES['BATTLE_TYPE']] = 0
        # Badges increased
        raw_after[RAM_ADDRESSES['BADGES']] = 0b00000001

        env._get_observation = lambda: raw_after
        r = env._compute_reward(0)
        assert env._last_battle_result == 'win'
        assert env.battles_won >= 1

    def test_battle_loss_detection():
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
        # Set party current HP entries to low values (cur 0, max 100)
        cur_base = RAM_ADDRESSES['PARTY_CURRENT_HP']
        max_base = RAM_ADDRESSES['PARTY_MAX_HP']
        raw_after[RAM_ADDRESSES['PARTY_COUNT']] = 1
        raw_after[cur_base] = 1  # cur hp low
        raw_after[cur_base + 1] = 0
        raw_after[max_base] = 100
        raw_after[max_base + 1] = 0

        env._get_observation = lambda: raw_after
        r = env._compute_reward(0)
        assert env._last_battle_result == 'loss'
        assert env.battles_lost >= 1

    test_battle_win_detection()
    print("✓ battle win detection test passed")
    test_battle_loss_detection()
    print("✓ battle loss detection test passed")

    # New environment-level test: done detection when all party HP are zero
    def test_env_done_when_all_fainted():
        env = PokemonYellowEnv.__new__(PokemonYellowEnv)
        # Ensure required attributes exist (mimic minimal init)
        env.episode_steps = 0
        env.tiles_visited = set()
        env.last_position = None
        env._prev_in_battle = False
        env._prev_badges = 0
        env._prev_money = 0
        env._prev_party_hp = [0.0] * 6
        env._last_battle_result = None
        # Monkeypatch observation to an all-zero 0x1000 slice
        env.emulator = None
        env._get_observation = lambda: bytes([0] * 0x1000)
        assert env._is_done() is True
    test_env_done_when_all_fainted()
    print("✓ env done detection test passed")

    # Tests for new battle outcome inspection helpers
    def test_is_battle_win_fallback():
        env = PokemonYellowEnv.__new__(PokemonYellowEnv)
        # Minimal init
        env.episode_steps = 0
        env.tiles_visited = set()
        env.last_position = None
        env._prev_in_battle = True
        env._prev_badges = 0
        env._prev_money = 0
        env._prev_party_hp = [50.0] * 6
        env._last_battle_result = None
        env.battles_won = 0
        env.battles_lost = 0

        # Simulate observation after battle: not in battle, badges increased
        raw_after = bytearray(0x1000)
        raw_after[RAM_ADDRESSES['IN_BATTLE_FLAG']] = 0
        raw_after[RAM_ADDRESSES['BADGES']] = 0b00000001
        env._get_observation = lambda: raw_after
        assert env._is_battle_win() is True
        assert env._is_battle_loss() is False

    test_is_battle_win_fallback()
    print("✓ _is_battle_win fallback test passed")

    def test_is_battle_loss_fallback():
        env = PokemonYellowEnv.__new__(PokemonYellowEnv)
        env.episode_steps = 0
        env.tiles_visited = set()
        env.last_position = None
        env._prev_in_battle = True
        env._prev_badges = 0
        env._prev_money = 0
        env._prev_party_hp = [100.0] * 6
        env._last_battle_result = None

        # Simulate observation after battle: not in battle, party HP dropped significantly
        raw_after = bytearray(0x1000)
        raw_after[RAM_ADDRESSES['IN_BATTLE_FLAG']] = 0
        raw_after[RAM_ADDRESSES['BATTLE_TYPE']] = 0
        # Set party current HP entries to low values (cur 0, max 100)
        cur_base = RAM_ADDRESSES['PARTY_CURRENT_HP']
        max_base = RAM_ADDRESSES['PARTY_MAX_HP']
        raw_after[RAM_ADDRESSES['PARTY_COUNT']] = 1
        raw_after[cur_base] = 1  # cur hp low
        raw_after[cur_base + 1] = 0
        raw_after[max_base] = 100
        raw_after[max_base + 1] = 0

        env._get_observation = lambda: raw_after
        assert env._is_battle_loss() is True
        assert env._is_battle_win() is False

    test_is_battle_loss_fallback()
    print("✓ _is_battle_loss fallback test passed")

    def test_battle_methods_honor_recorded_result():
        env = PokemonYellowEnv.__new__(PokemonYellowEnv)
        env._last_battle_result = 'win'
        assert env._is_battle_win() is True
        assert env._is_battle_loss() is False
        env._last_battle_result = 'loss'
        assert env._is_battle_win() is False
        assert env._is_battle_loss() is True

    test_battle_methods_honor_recorded_result()
    print("✓ battle methods honor recorded result test passed")

    def test_battle_end_by_roster_count_win():
        env = PokemonYellowEnv.__new__(PokemonYellowEnv)
        env.episode_steps = 0
        env.tiles_visited = set()
        env.last_position = None
        env._prev_in_battle = True
        env._prev_opponent_roster_count = 1
        env._prev_badges = 0
        env._prev_money = 0
        env._prev_party_hp = [50.0] * 6
        env._last_battle_result = None

        raw_after = bytearray(0x1000)
        raw_after[RAM_ADDRESSES['IN_BATTLE_FLAG']] = 0
        raw_after[RAM_ADDRESSES['OPPONENT_ROSTER_COUNT']] = 0
        raw_after[RAM_ADDRESSES['PARTY_COUNT']] = 1
        cur_base = RAM_ADDRESSES['PARTY_CURRENT_HP']
        max_base = RAM_ADDRESSES['PARTY_MAX_HP']
        raw_after[cur_base] = 50 & 0xFF
        raw_after[cur_base + 1] = 0
        raw_after[max_base] = 100
        raw_after[max_base + 1] = 0

        env._get_observation = lambda: raw_after
        assert env._is_battle_win() is True
        assert env._is_battle_loss() is False

    test_battle_end_by_roster_count_win()

    def test_battle_end_by_roster_count_small_survivor():
        env = PokemonYellowEnv.__new__(PokemonYellowEnv)
        env.episode_steps = 0
        env.tiles_visited = set()
        env.last_position = None
        env._prev_in_battle = True
        env._prev_opponent_roster_count = 1
        env._prev_badges = 0
        env._prev_money = 0
        env._prev_party_hp = [100.0] * 6
        env._last_battle_result = None

        raw_after = bytearray(0x1000)
        raw_after[RAM_ADDRESSES['IN_BATTLE_FLAG']] = 0
        raw_after[RAM_ADDRESSES['OPPONENT_ROSTER_COUNT']] = 0
        raw_after[RAM_ADDRESSES['PARTY_COUNT']] = 1
        cur_base = RAM_ADDRESSES['PARTY_CURRENT_HP']
        max_base = RAM_ADDRESSES['PARTY_MAX_HP']
        # Tiny surviving HP (1/100 => 0.01 normalized)
        raw_after[cur_base] = 1
        raw_after[cur_base + 1] = 0
        raw_after[max_base] = 100
        raw_after[max_base + 1] = 0

        env._get_observation = lambda: raw_after
        # Should NOT be a win due to tiny survivor; large HP drop implies loss
        assert env._is_battle_win() is False
        assert env._is_battle_loss() is True

    test_battle_end_by_roster_count_small_survivor()
    print("✓ battle end by roster count small survivor test passed")
    print("✓ battle end by roster count win test passed")

    def test_battle_end_by_roster_count_loss():
        env = PokemonYellowEnv.__new__(PokemonYellowEnv)
        env.episode_steps = 0
        env.tiles_visited = set()
        env.last_position = None
        env._prev_in_battle = True
        env._prev_opponent_roster_count = 1
        env._prev_badges = 0
        env._prev_money = 0
        env._prev_party_hp = [100.0] * 6
        env._last_battle_result = None

        raw_after = bytearray(0x1000)
        raw_after[RAM_ADDRESSES['IN_BATTLE_FLAG']] = 0
        raw_after[RAM_ADDRESSES['OPPONENT_ROSTER_COUNT']] = 0
        raw_after[RAM_ADDRESSES['PARTY_COUNT']] = 1
        cur_base = RAM_ADDRESSES['PARTY_CURRENT_HP']
        max_base = RAM_ADDRESSES['PARTY_MAX_HP']
        # Set party current HP to zero (all fainted)
        raw_after[cur_base] = 0
        raw_after[cur_base + 1] = 0
        raw_after[max_base] = 100
        raw_after[max_base + 1] = 0

        env._get_observation = lambda: raw_after
        assert env._is_battle_loss() is True
        assert env._is_battle_win() is False

    test_battle_end_by_roster_count_loss()
    print("✓ battle end by roster count loss test passed")

    # Small unit tests for new decoders
    def test_decode_options_and_ids():
        raw = bytearray(0x1000)
        # D355 options: set battle animation off (bit7) and text speed to 3
        raw[RAM_ADDRESSES['OPTIONS']] = (1 << 7) | 3
        opts = __import__('memory_map').decode_options(raw)
        assert opts['battle_animation_off'] is True
        assert opts['text_speed'] == 3

        raw[RAM_ADDRESSES['PLAYER_ID_LOW']] = 0x34
        raw[RAM_ADDRESSES['PLAYER_ID_HIGH']] = 0x12
        pid = __import__('memory_map').decode_player_id(raw)
        assert pid == 0x1234

        raw[RAM_ADDRESSES['PLAYER_TILE_Y']] = 0x55
        raw[RAM_ADDRESSES['PLAYER_TILE_X']] = 0x66
        x, y = __import__('memory_map').decode_player_tile_block(raw)
        assert (x, y) == (0x66, 0x55)

    test_decode_options_and_ids()
    print("✓ decode options and ids test passed")

    def test_game_time_and_constants_present():
        # Verify some newly-added constants exist and can be read
        from memory_map import RAM_ADDRESSES, get_byte, get_word_le, decode_game_time
        assert 'GAME_TIME_HOURS' in RAM_ADDRESSES
        assert 'PLAYER_NAME' in RAM_ADDRESSES

        raw = bytearray(0x1000)
        # Set hours (word little-endian)
        raw[RAM_ADDRESSES['GAME_TIME_HOURS']] = 0x2A
        raw[RAM_ADDRESSES['GAME_TIME_HOURS'] + 1] = 0x00
        gt = decode_game_time(raw)
        assert gt['hours'] == 0x2A

        # get_byte and get_word_le helpers
        raw[RAM_ADDRESSES['PLAYER_ID_LOW']] = 0x11
        raw[RAM_ADDRESSES['PLAYER_ID_HIGH']] = 0x22
        assert get_byte(raw, RAM_ADDRESSES['PLAYER_ID_LOW']) == 0x11
        assert get_word_le(raw, RAM_ADDRESSES['GAME_TIME_HOURS']) == 0x2A

    test_game_time_and_constants_present()
    print("✓ game time and constants presence test passed")

    def test_party_helpers_and_opponent_decoding():
        raw = bytearray(0x1000)
        # Party count
        raw[RAM_ADDRESSES['PARTY_COUNT']] = 2
        # Party 0: cur=40, max=100
        cur0 = RAM_ADDRESSES['PARTY_CURRENT_HP']
        max0 = RAM_ADDRESSES['PARTY_MAX_HP']
        raw[cur0] = 40 & 0xFF
        raw[cur0 + 1] = 0
        raw[max0] = 100 & 0xFF
        raw[max0 + 1] = 0
        # Party 1: cur=10, max=80
        cur1 = cur0 + RAM_ADDRESSES['PARTY_BLOCK_SIZE']
        max1 = max0 + RAM_ADDRESSES['PARTY_BLOCK_SIZE']
        raw[cur1] = 10 & 0xFF
        raw[cur1 + 1] = 0
        raw[max1] = 80 & 0xFF
        raw[max1 + 1] = 0

        # Opponent roster: count 1, first pokemon id and HP
        raw[RAM_ADDRESSES['OPPONENT_ROSTER_COUNT']] = 1
        base = RAM_ADDRESSES['OPPONENT_POKEMON_BASE']
        raw[base] = 55  # poke id
        raw[base + 1] = 0x2A  # cur hp low
        raw[base + 2] = 0x00  # cur hp high
        raw[base + 3] = 4  # status

        from memory_map import get_party_count, get_party_mon_hp, decode_opponent_pokemon, parse_player_name, is_event_flag_set
        assert get_party_count(raw) == 2
        assert get_party_mon_hp(raw, 0) == (40, 100)
        assert get_party_mon_hp(raw, 1) == (10, 80)

        opp0 = decode_opponent_pokemon(raw, 0)
        assert opp0['id'] == 55
        assert opp0['current_hp'] == 0x2A
        assert opp0['status'] == 4

        # Player name parsing
        raw[RAM_ADDRESSES['PLAYER_NAME']] = ord('A')
        raw[RAM_ADDRESSES['PLAYER_NAME'] + 1] = ord('B')
        raw[RAM_ADDRESSES['PLAYER_NAME'] + 2] = 0x00
        name = __import__('memory_map').parse_player_name(raw)
        assert name.startswith('AB')

        # Event flag test
        # Set flag index 10 -> should set bit 2 of byte at base + 1
        flag_idx = 10
        byte_idx = RAM_ADDRESSES['EVENT_FLAGS_BASE'] + (flag_idx // 8)
        raw[byte_idx] = 1 << (flag_idx % 8)
        assert is_event_flag_set(raw, flag_idx) is True

    test_party_helpers_and_opponent_decoding()
    print("✓ party helpers and opponent decoding tests passed")

    def test_extended_opponent_parsing_and_inspector():
        raw = bytearray(0x1000)
        # Opponent roster: count 2
        raw[RAM_ADDRESSES['OPPONENT_ROSTER_COUNT']] = 2
        base = RAM_ADDRESSES['OPPONENT_POKEMON_BASE']
        # Opponent 0: id=55, cur=42, max=100, level=12, type1=3, type2=4, moves=[1,2,3,4]
        raw[base + 0] = 55
        raw[base + 1] = 42
        raw[base + 2] = 0
        raw[base + 4] = 100
        raw[base + 5] = 0
        raw[base + 6] = 12
        raw[base + 7] = 3
        raw[base + 8] = 4
        raw[base + 9] = 1
        raw[base + 10] = 2
        raw[base + 11] = 3
        raw[base + 12] = 4
        # Opponent 1: id=99
        b1 = base + RAM_ADDRESSES['OPPONENT_POKEMON_BLOCK_SIZE']
        raw[b1 + 0] = 99
        raw[b1 + 1] = 7
        raw[b1 + 2] = 0

        from memory_map import decode_opponent_pokemon, pretty_print_battle
        o0 = decode_opponent_pokemon(raw, 0)
        assert o0['id'] == 55
        assert o0['current_hp'] == 42
        assert o0['max_hp'] == 100
        assert o0['level'] == 12
        assert o0['type1'] == 3
        assert o0['moves'][0] == 1

        summary = pretty_print_battle(raw)
        assert isinstance(summary, dict)
        assert len(summary['opponents']) == 2
        assert summary['opponents'][0]['id'] == 55

    test_extended_opponent_parsing_and_inspector()
    print("✓ extended opponent parsing and inspector tests passed")

    # New tests: type/move name resolution and variant support
    def test_type_and_move_name_resolution():
        from memory_map import type_id_to_name, move_id_to_name
        assert type_id_to_name(3) == 'Poison'
        assert move_id_to_name(1) == 'Pound'
        # Unknown move ids fall back to Move_#
        assert move_id_to_name(9999).startswith('Move_')

    test_type_and_move_name_resolution()
    print("✓ type and move name resolution tests passed")

    def test_set_game_variant_yellow():
        from memory_map import set_game_variant, get_game_variant, RAM_ADDRESSES
        set_game_variant('yellow')
        assert get_game_variant() == 'yellow'
        assert RAM_ADDRESSES['PLAYER_NAME_LEN'] == 7

    test_set_game_variant_yellow()
    print("✓ set_game_variant('yellow') test passed")

    def test_shift_offsets_for_yellow():
        from memory_map import shift_offsets, RAM_ADDRESSES
        new_map = shift_offsets(RAM_ADDRESSES, ['PLAYER_MAP_ID'], -1)
        assert new_map['PLAYER_MAP_ID'] == (RAM_ADDRESSES['PLAYER_MAP_ID'] - 1)
        # Ensure original map not mutated
        assert RAM_ADDRESSES['PLAYER_MAP_ID'] != new_map['PLAYER_MAP_ID']

    test_shift_offsets_for_yellow()
    print("✓ shift_offsets for yellow test passed")

    print("All tests passed")
