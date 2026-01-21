"""
Memory map constants and helper decoders for Pokémon Yellow.
"""
from typing import Optional

# RAM Addresses - These are offsets relative to the WRAM window at 0xD000..0xDFFF (0x1000 bytes)
# The memory_map constants are provided as offsets from 0xD000 (e.g. 0x35E -> absolute 0xD35E)
# Values derived from the Pokémon R/B/Y RAM map (DataCrystal / pret disassembly)
# Pokemon Yellow - Most of the RAM map (but not all) for this game has an offset of -1 from the one on Red and Blue.
# Offsets below are already adjusted for Yellow where applicable
RAM_ADDRESSES = {
    # Current map and player coordinates (relative to 0xD000)
    # D35E = Current map ID
    # D361 = Player Y-position (1 byte)
    # D362 = Player X-position (1 byte)
    'PLAYER_MAP_ID': 0x35D,  # absolute 0xD35E
    'PLAYER_MAP_Y': 0x360,   # absolute 0xD361
    'PLAYER_MAP_X': 0x361,   # absolute 0xD362

    # Player tile/block coordinates (D363/D364) - finer-grained block positions
    'PLAYER_TILE_Y': 0x362,
    'PLAYER_TILE_X': 0x363,

    # Player ID bytes (D359..D35A)
    'PLAYER_ID_LOW': 0x358,
    'PLAYER_ID_HIGH': 0x359,

    # Player name (D158..D162, inclusive) - useful for diagnostics
    'PLAYER_NAME': 0x157,
    'PLAYER_NAME_LEN': 5,

    # Map header (current loaded map)
    # R/B symbols are D367..; Yellow often shifts many D000-window offsets by -1
    'MAP_TILESET': 0x366,       # D367 tileset (Yellow -1)
    'MAP_HEIGHT':  0x367,       # D368 height  (Yellow -1)
    'MAP_WIDTH':   0x368,       # D369 width   (Yellow -1)
    'MAP_CONNECTIONS': 0x36F,   # D370 connection byte (Yellow -1)

    # Battle state
    # D057 = Type of battle (0 when not in battle)
    # D05A = Battle type (normal/safari/old man...)
    'IN_BATTLE_FLAG': 0x056,  # absolute 0xD057
    'BATTLE_TYPE': 0x059,     # absolute 0xD05A

    # Battle-related helper
    'BATTLE_MUSIC': 0x05B,
    'BATTLE_STATUS': 0x061,  # D062-D064 battle status bytes (player)
    'BATTLE_CRITICAL_FLAG': 0x05D,
    'BATTLE_HOOK_FLAG': 0x05E,

    # Battle context (D000 window)
    'BATTLE_PLAYER_PARTY_SLOT': 0x004,  # D005 (Yellow -1)
    'BATTLE_ENEMY_PARTY_SLOT':  0x008,  # D009 (Yellow -1)
    'BATTLE_ATTACK_MISSED':     0x06D,  # D06E (Yellow -1)
    'DAMAGE_TO_BE_DEALT':       0x0D7,  # D0D8 (Yellow -1)

    # Escape flag (Gen 1): non-zero when an item/move that allows escape was used.
    # Community maps often label this `wEscapedFromBattle` / `wd078`.
    # Yellow offsets are frequently -1 from R/B in the D000 window, so we keep a fallback.
    'ESCAPED_FROM_BATTLE': 0x077,      # absolute 0xD078

    # Party data (arrays of Pokémon data blocks)
    # D163 = Party size (# Pokémon in party)
    # D16C.. = current HP (2 bytes per mon), D18D.. = max HP (2 bytes per mon)
    'PARTY_COUNT': 0x162,
    'PARTY_CURRENT_HP': 0x16B,
    'PARTY_MAX_HP': 0x18C,
    'PARTY_BLOCK_SIZE': 0x2C,  # bytes between successive party entries

    # Party extended signals (base fields; per-mon stride = PARTY_BLOCK_SIZE)
    # Base offsets are for mon #0; add i*PARTY_BLOCK_SIZE for mon i
    'PARTY_STATUS': 0x16E,      # D16F status (Yellow -1)
    'PARTY_TYPE1':  0x170,      # D171 type1  (Yellow -1)
    'PARTY_TYPE2':  0x171,      # D172 type2  (Yellow -1)
    'PARTY_MOVES':  0x172,      # D173..D176 moves (Yellow -1)
    'PARTY_EXP':    0x178,      # D179..D17B exp (Yellow -1)
    'PARTY_LEVEL':  0x18B,      # D18C level  (Yellow -1)

    # Game progress
    # D356 contains badges as binary switches (bits set = badge owned)
    'BADGES': 0x355,
    # D347-D349 = money bytes (3-byte BCD, 6 decimal digits)
    'MONEY': 0x346,

    # Text/display-related (see notes) - D358/D359 are text-related flags/IDs
    'TEXT_STATE': 0x357,
    # Misc options (D355) - encodes battle animation, battle style, text speed nybble
    'OPTIONS': 0x354,

    # Event displacement / small event offsets
    'EVENT_DISPLACEMENT_1': 0x35E,
    'EVENT_DISPLACEMENT_2': 0x35F,
    
    # Event flags base (D5A6..D5C5 contains many region-specific flags)
    'EVENT_FLAGS_BASE': 0x5A5,

    # Opponent/trainer data and in-battle structures start in higher D offsets
    # (e.g. opponent roster at ~0x8A4). Add on demand.
    'OPPONENT_ROSTER_COUNT': 0x89B,
    'OPPONENT_POKEMON_BASE': 0x8A3,
    'OPPONENT_POKEMON_BLOCK_SIZE': 0x2C,

    # Game time (DA40..DA45)
    'GAME_TIME_HOURS': 0xA3F,
    'GAME_TIME_MINUTES': 0xA41,
    'GAME_TIME_SECONDS': 0xA43,
    'GAME_TIME_FRAMES': 0xA44,

    # Note: many additional addresses exist (player direction, battle internals, etc.)
    # See DataCrystal / Red & Blue RAM map for more details; add on demand.
}


# --- Absolute WRAM addresses (NOT relative to the 0xD000 window) ---
# Yellow shares most of the same absolute WRAM symbols as R/B; the common
# -1 offset note mainly affects many D000-window offsets in some community maps.
# These are absolute addresses, so they do not use the -1 adjustment.
ABS_WRAM_ADDRESSES = {
    # Gen 1 (R/B/Y) battle result.
    # 0x00 = battle started OR win
    # 0x01 = loss
    # 0x02 = draw (running away counts as draw)
    'BATTLE_RESULT': 0xCF0B,
    'BATTLE_TURN_COUNTER': 0xCCD5,
    'PLAYER_SUBSTITUTE_HP': 0xCCD7,
    'ENEMY_SUBSTITUTE_HP': 0xCCD8,    
}


def get_abs_byte(ram_data: bytes, abs_addr: int, base_address: Optional[int] = None) -> int:
    """Read a byte at an absolute WRAM address.

    Supports:
      - a full 0xC000..0xDFFF dump (len >= 0x2000), where index 0 == 0xC000
      - a buffer with an explicit `base_address` indicating address of ram_data[0]

    Returns 0 on failure.
    """
    try:
        if base_address is not None:
            i = int(abs_addr) - int(base_address)
            if i < 0 or i >= len(ram_data):
                return 0
            return int(ram_data[i]) & 0xFF

        # Common case: env provides 0xC000..0xDFFF
        if len(ram_data) >= 0x2000:
            i = int(abs_addr) - 0xC000
            if i < 0 or i >= len(ram_data):
                return 0
            return int(ram_data[i]) & 0xFF

        # If caller provided only the 0xD000 slice, we cannot read Cxxx addresses.
        return 0
    except Exception:
        return 0


def decode_battle_result(ram_data: bytes) -> int:
    """Return wBattleResult: 0=win/start, 1=loss, 2=draw/run-away. Returns -1 if unavailable."""
    try:
        if len(ram_data) < 0x2000:
            return -1
        return int(get_abs_byte(ram_data, ABS_WRAM_ADDRESSES['BATTLE_RESULT'])) & 0xFF
    except Exception:
        return -1

def decode_player_position(ram_data: bytes) -> tuple:
    """Decode player position from RAM data.

    Accepts either a 0xD000..0xDFFF WRAM slice (preferred) or a full 0xC000..0xDFFF dump.
    The function will normalize inputs using `to_d000_slice` when possible.

    Uses authoritative WRAM offsets (relative to 0xD000):
      - PLAYER_MAP_ID at D35E
      - PLAYER_MAP_Y  at D361
      - PLAYER_MAP_X  at D362
    """
    try:
        ram = to_d000_slice(ram_data)
        map_id = ram[RAM_ADDRESSES['PLAYER_MAP_ID']]
        x = ram[RAM_ADDRESSES['PLAYER_MAP_X']]
        y = ram[RAM_ADDRESSES['PLAYER_MAP_Y']]
        return (map_id, x, y)
    except (IndexError, ValueError):
        return (0, 0, 0)


def decode_in_battle(ram_data: bytes) -> bool:
    """Return True if a battle is active.

    Accepts WRAM slice or full dump; normalizes via `to_d000_slice`.
    Checks D057 (Type of battle) — non-zero indicates a battle context.
    """
    try:
        ram = to_d000_slice(ram_data)
        return bool(ram[RAM_ADDRESSES['IN_BATTLE_FLAG']])
    except (IndexError, ValueError):
        return False


def decode_escaped_from_battle(ram_data: bytes) -> bool:
    """Return True if an escape-from-battle item/move was used.

    This corresponds to `wEscapedFromBattle` (often labeled `wd078`) which is
    non-zero when an item or move that allows escape from battle was used.

    """
    try:
        ram = to_d000_slice(ram_data)
        off1 = RAM_ADDRESSES.get('ESCAPED_FROM_BATTLE')

        if off1 is not None and off1 < len(ram) and (int(ram[off1]) & 0xFF) != 0:
            return True

        return False
    except Exception:
        return False


def decode_badge_count(ram_data: bytes) -> int:
    """Return the number of badges owned (counts set bits in D356).

    Accepts WRAM slice or full dump; normalizes via `to_d000_slice`.
    D356 stores badges as individual bits (one bit per badge).
    """
    try:
        ram = to_d000_slice(ram_data)
        badges_byte = ram[RAM_ADDRESSES['BADGES']]
        # Count set bits
        return bin(badges_byte).count('1')
    except (IndexError, ValueError):
        return 0



def _bcd_byte_to_int(b: int) -> int:
    """Convert one BCD byte (two decimal digits) to int.

    Example: 0x42 -> 42
    """
    return (((b >> 4) & 0x0F) * 10) + (b & 0x0F)


def money_bcd3_to_int(b0: int, b1: int, b2: int) -> int:
    """Convert 3-byte BCD money (6 digits) to int.

    Example: 0x12 0x34 0x56 -> 123456
    """
    return (_bcd_byte_to_int(b0) * 10000) + (_bcd_byte_to_int(b1) * 100) + _bcd_byte_to_int(b2)


def decode_money(ram_data: bytes) -> int:
    """Decode money value from D347..D349.

    Gen 1 stores money as 3-byte BCD (6 decimal digits).
    Example: 0x12 0x34 0x56 -> 123456

    Accepts WRAM slice or full dump; normalizes via `to_d000_slice`.
    Returns an integer amount (0 on out-of-bounds or malformed data).
    """
    try:
        ram = to_d000_slice(ram_data)
        start = RAM_ADDRESSES['MONEY']
        b0 = int(ram[start]) & 0xFF
        b1 = int(ram[start + 1]) & 0xFF
        b2 = int(ram[start + 2]) & 0xFF
        value = money_bcd3_to_int(b0, b1, b2)
        # Defensive clamp
        if value < 0:
            return 0
        if value > 999999:
            return 999999
        return int(value)
    except (IndexError, KeyError, ValueError):
        return 0

def decode_basic_game_info(ram_data: bytes) -> dict:
    """Return a small, safe summary of commonly-used game state.

    Keys: map_id, x, y, in_battle, badges, money
    """
    try:
        ram = to_d000_slice(ram_data)
        map_id, x, y = decode_player_position(ram)
        in_battle = bool(decode_in_battle(ram))
        badges = int(decode_badge_count(ram))
        money = int(decode_money(ram))
        return {
            "map_id": int(map_id),
            "x": int(x),
            "y": int(y),
            "in_battle": bool(in_battle),
            "badges": int(badges),
            "money": int(money),
        }
    except Exception:
        return {
            "map_id": 0,
            "x": 0,
            "y": 0,
            "in_battle": False,
            "badges": 0,
            "money": 0,
        }


def get_opponent_roster_count(ram_data: bytes) -> int:
    """Best-effort read of opponent roster count from WRAM. Returns 0 on failure."""
    try:
        ram = to_d000_slice(ram_data)
        off = RAM_ADDRESSES.get("OPPONENT_ROSTER_COUNT")
        if off is None or len(ram) <= off:
            return 0
        return int(ram[off]) & 0xFF
    except Exception:
        return 0

def decode_party_stats(ram_data: bytes) -> list:
    """Return list of up to 6 party HP percentages (current/max) as floats.

    Accepts WRAM slice or full dump; normalizes via `to_d000_slice`.
    Uses the party block layout from the RAM map:
      - PARTY_COUNT at D163 (number of Pokémon in party)
      - Current HP stored as 16-bit LE at D16C + i*0x2C
      - Max HP stored as 16-bit LE at D18D + i*0x2C
    If memory is out-of-bounds or values invalid, returns zeros for missing entries.
    """
    stats = [0.0] * 6
    try:
        ram = to_d000_slice(ram_data)
        count = ram[RAM_ADDRESSES['PARTY_COUNT']]
        count = min(count, 6)
        cur_base = RAM_ADDRESSES['PARTY_CURRENT_HP']
        max_base = RAM_ADDRESSES['PARTY_MAX_HP']
        stride = RAM_ADDRESSES['PARTY_BLOCK_SIZE']
        for i in range(count):
            cur_off = cur_base + i * stride
            max_off = max_base + i * stride
            # Little-endian 16-bit values
            cur_hp = ram[cur_off] | (ram[cur_off + 1] << 8)
            max_hp = ram[max_off] | (ram[max_off + 1] << 8)
            stats[i] = (cur_hp / max_hp) if max_hp > 0 else 0.0
        return stats
    except (IndexError, KeyError, ValueError):
        return [0.0] * 6


def get_party_count(ram_data: bytes) -> int:
    """Return party size (number of Pokémon in party)."""
    try:
        ram = to_d000_slice(ram_data)
        return int(ram[RAM_ADDRESSES['PARTY_COUNT']])
    except (IndexError, ValueError):
        return 0


def get_party_mon_hp(ram_data: bytes, index: int) -> tuple:
    """Return (current_hp, max_hp) for the party Pokémon at `index` (0-based).

    Raises IndexError if index out-of-range of party block layout.
    """
    ram = to_d000_slice(ram_data)
    count = min(ram[RAM_ADDRESSES['PARTY_COUNT']], 6)
    if index < 0 or index >= count:
        raise IndexError("party index out of range")
    cur_base = RAM_ADDRESSES['PARTY_CURRENT_HP']
    max_base = RAM_ADDRESSES['PARTY_MAX_HP']
    stride = RAM_ADDRESSES['PARTY_BLOCK_SIZE']
    cur_off = cur_base + index * stride
    max_off = max_base + index * stride
    cur_hp = ram[cur_off] | (ram[cur_off + 1] << 8)
    max_hp = ram[max_off] | (ram[max_off + 1] << 8)
    return cur_hp, max_hp


def decode_opponent_pokemon(ram_data: bytes, index: int = 0) -> dict:
    """Decode opponent party slot `index` using the Gen 1 enemy party struct (0x2C bytes each).

    Layout (from enemy party list):
      base+0x00 species
      base+0x01..0x02 current HP (LE)
      base+0x04 status
      base+0x05 type1
      base+0x06 type2
      base+0x08..0x0B moves (4)
      base+0x0E..0x10 exp (3, BE)
      base+0x21 level
      base+0x22..0x23 max HP (LE)
    """
    try:
        ram = to_d000_slice(ram_data)
        roster_count = int(ram[RAM_ADDRESSES['OPPONENT_ROSTER_COUNT']]) & 0xFF
        if index < 0 or index >= roster_count:
            raise IndexError("opponent index out of range")

        stride = RAM_ADDRESSES['OPPONENT_POKEMON_BLOCK_SIZE']
        base = RAM_ADDRESSES['OPPONENT_POKEMON_BASE'] + index * stride

        species = int(ram[base + 0x00]) & 0xFF
        cur_hp = int(ram[base + 0x01]) | (int(ram[base + 0x02]) << 8)
        status = int(ram[base + 0x04]) & 0xFF
        type1 = int(ram[base + 0x05]) & 0xFF
        type2 = int(ram[base + 0x06]) & 0xFF
        moves = [int(ram[base + 0x08 + j]) & 0xFF for j in range(4)]
        exp = _u24_be(ram[base + 0x0E], ram[base + 0x0F], ram[base + 0x10])
        level = int(ram[base + 0x21]) & 0xFF
        max_hp = int(ram[base + 0x22]) | (int(ram[base + 0x23]) << 8)

        raw_block = bytes(ram[base: base + stride])

        return {
            'id': species,
            'current_hp': cur_hp,
            'max_hp': max_hp,
            'level': level,
            'status': status,
            'type1': type1,
            'type2': type2,
            'moves': moves,
            'exp': int(exp),
            'raw_block': raw_block,
        }
    except Exception:
        return {
            'id': 0, 'current_hp': 0, 'max_hp': 0, 'level': 0,
            'status': 0, 'type1': 0, 'type2': 0, 'moves': [],
            'exp': 0, 'raw_block': b''
        }
    
def decode_text_state(ram_data: bytes) -> int:
    """Return text state / text-speed flags (D358).

    Accepts WRAM slice or full dump; normalizes via `to_d000_slice`.
    """
    try:
        ram = to_d000_slice(ram_data)
        return ram[RAM_ADDRESSES['TEXT_STATE']]
    except (IndexError, ValueError):
        return 0


def decode_options(ram_data: bytes) -> dict:
    """Decode the D355 options byte into useful flags.

    Returns a dict with:
      - battle_animation_off (bool)
      - battle_style_set (bool)
      - text_speed (0..15, where lower is faster)
    """
    try:
        ram = to_d000_slice(ram_data)
        opts = ram[RAM_ADDRESSES['OPTIONS']]
        battle_animation_off = bool((opts >> 7) & 1)
        battle_style_set = bool((opts >> 6) & 1)
        text_speed = opts & 0x0F
        return {
            'battle_animation_off': battle_animation_off,
            'battle_style_set': battle_style_set,
            'text_speed': text_speed,
        }
    except (IndexError, ValueError):
        return {'battle_animation_off': False, 'battle_style_set': False, 'text_speed': 0}


def decode_player_tile_block(ram_data: bytes) -> tuple:
    """Return player's current block/tile coordinates (D363/D364)."""
    try:
        ram = to_d000_slice(ram_data)
        y = ram[RAM_ADDRESSES['PLAYER_TILE_Y']]
        x = ram[RAM_ADDRESSES['PLAYER_TILE_X']]
        return (x, y)
    except (IndexError, ValueError):
        return (0, 0)


def decode_player_id(ram_data: bytes) -> int:
    """Return player's ID (combines D359/D35A).

    Also returns 0 on malformed data.
    """
    try:
        ram = to_d000_slice(ram_data)
        low = ram[RAM_ADDRESSES['PLAYER_ID_LOW']]
        high = ram[RAM_ADDRESSES['PLAYER_ID_HIGH']]
        return (high << 8) | low
    except (IndexError, ValueError):
        return 0


def parse_player_name(ram_data: bytes) -> str:
    """Return player's name as a best-effort ASCII string.

    The in-ROM text encoding isn't plain ASCII; for simplicity this helper
    decodes the name bytes as Latin-1 and strips common terminators (0x00, 0x50).
    """
    try:
        ram = to_d000_slice(ram_data)
        start = RAM_ADDRESSES['PLAYER_NAME']
        length = RAM_ADDRESSES.get('PLAYER_NAME_LEN', 5)
        raw = bytes(ram[start:start + length])
        # Trim at first 0x00 or 0x50 if present
        for term in (b'\x00', b'\x50'):
            if term in raw:
                raw = raw.split(term, 1)[0]
        try:
            return raw.decode('latin-1')
        except Exception:
            return ''.join(chr(b) for b in raw)
    except (IndexError, ValueError):
        return ''


def is_event_flag_set(ram_data: bytes, flag_index: int) -> bool:
    """Return True if the event flag with `flag_index` (absolute index) is set.

    Flags are stored as bits starting at `EVENT_FLAGS_BASE`.
    """
    try:
        ram = to_d000_slice(ram_data)
        byte_index = RAM_ADDRESSES['EVENT_FLAGS_BASE'] + (flag_index // 8)
        bit = flag_index % 8
        val = ram[byte_index]
        return bool((val >> bit) & 1)
    except (IndexError, ValueError):
        return False


def decode_game_time(ram_data: bytes) -> dict:
    """Return game time components (hours, minutes, seconds, frames).

    Values are read from DA40..DA45 (offsets relative to 0xD000).
    """
    try:
        ram = to_d000_slice(ram_data)
        hours = ram[RAM_ADDRESSES['GAME_TIME_HOURS']] | (ram[RAM_ADDRESSES['GAME_TIME_HOURS'] + 1] << 8)
        minutes = ram[RAM_ADDRESSES['GAME_TIME_MINUTES']] | (ram[RAM_ADDRESSES['GAME_TIME_MINUTES'] + 1] << 8)
        seconds = ram[RAM_ADDRESSES['GAME_TIME_SECONDS']]
        frames = ram[RAM_ADDRESSES['GAME_TIME_FRAMES']]
        return {'hours': hours, 'minutes': minutes, 'seconds': seconds, 'frames': frames}
    except (IndexError, ValueError):
        return {'hours': 0, 'minutes': 0, 'seconds': 0, 'frames': 0}


def pretty_print_battle(ram_data: bytes) -> dict:
    """Return a concise dict summarizing the current in-battle state.

    Summary includes: 'in_battle' (bool), 'battle_type', 'party' (list of dicts with
    per-mon current/max HP), and 'opponents' (list of decoded opponent dicts).

    This helper is useful for quick diagnostics and in-showcase overlays.
    """
    try:
        ram = to_d000_slice(ram_data)
        from math import inf
        in_battle = bool(ram[RAM_ADDRESSES['IN_BATTLE_FLAG']])
        battle_type = decode_battle_type(ram)
        party_list = []
        party_count = get_party_count(ram)
        for i in range(party_count):
            cur, maxhp = get_party_mon_hp(ram, i)
            party_list.append({'index': i, 'current_hp': cur, 'max_hp': maxhp})
        opponents = []
        roster = get_opponent_roster_count(ram)
        for i in range(roster):
            opponents.append(decode_opponent_pokemon(ram, i))
        return {
            'in_battle': in_battle,
            'battle_type': battle_type,
            'party': party_list,
            'opponents': opponents,
        }
    except (IndexError, ValueError):
        return {'in_battle': False, 'battle_type': 0, 'party': [], 'opponents': []}


def get_observation_vector(ram_data: bytes) -> list:
    """Create compact normalized observation vector from RAM data.

    NOTE: Upgraded to v2 (43-dim) for richer training signals.
    """
    return get_observation_vector_v2(ram_data)

def get_observation_vector_v2(ram_data: bytes) -> list:
    """Expanded normalized observation vector (43 floats).

    Layout:
      0  map_id/255
      1  x/255
      2  y/255
      3  in_battle (0/1)
      4  badges/8
      5  money/999999
      6  text_state/255

      7  tileset/255
      8  map_height/255
      9  map_width/255
      10 connections/255

      11 party_count/6

      12..17  party_hp[0..5] (0..1)
      18..23  party_levels[0..5] / 100
      24..29  party_statuses[0..5] / 255

      30 player_party_slot/5
      31 enemy_party_slot/5
      32 attack_missed (0/1)
      33 pending_damage/255
      34 battle_turn_counter/255
      35 player_sub_hp/255
      36 enemy_sub_hp/255

      37 opponent_roster_count/6
      38 opponent0_hp_ratio (0..1)
      39 opponent0_level/100
      40 opponent0_status/255
      41 opponent0_type1/15
      42 opponent0_type2/15
    """
    try:
        # normalize to 0xD000 slice for Dxxx offsets
        ram = to_d000_slice(ram_data)

        map_id, x, y = decode_player_position(ram)
        in_battle = 1.0 if decode_in_battle(ram) else 0.0
        badges = float(decode_badge_count(ram))
        money = float(decode_money(ram))
        text_state = float(decode_text_state(ram))

        mh = decode_map_header(ram)
        tileset = float(mh.get("tileset", 0))
        mheight = float(mh.get("height", 0))
        mwidth = float(mh.get("width", 0))
        mconn = float(mh.get("connections", 0))

        party_count = float(get_party_count(ram))
        if party_count < 0:
            party_count = 0.0
        if party_count > 6:
            party_count = 6.0

        party_hp = decode_party_stats(ram)              # 6 floats
        party_lv = decode_party_levels(ram)             # 6 ints
        party_st = decode_party_statuses(ram)           # 6 ints

        bc = decode_battle_context(ram)
        pslot = float(bc.get("player_party_slot", 0))
        eslot = float(bc.get("enemy_party_slot", 0))
        attack_missed = 1.0 if int(bc.get("attack_missed", 0)) != 0 else 0.0
        pending_damage = float(bc.get("pending_damage", 0))

        # Absolute WRAM reads require 0xC000..0xDFFF buffer; if not available they return 0.
        turn_counter = float(decode_battle_turn_counter(ram_data))
        sub = decode_substitute_hp(ram_data)
        sub_p = float(sub.get("player", 0))
        sub_e = float(sub.get("enemy", 0))

        roster = float(get_opponent_roster_count(ram))
        if roster < 0:
            roster = 0.0
        if roster > 6:
            roster = 6.0

        opp0 = decode_opponent_pokemon(ram, 0)
        opp0_hp_ratio = 0.0
        try:
            chp = float(opp0.get("current_hp", 0))
            mhp = float(opp0.get("max_hp", 0))
            opp0_hp_ratio = (chp / mhp) if mhp > 0 else 0.0
        except Exception:
            opp0_hp_ratio = 0.0

        opp0_level = float(opp0.get("level", 0))
        opp0_status = float(opp0.get("status", 0))
        opp0_type1 = float(opp0.get("type1", 0))
        opp0_type2 = float(opp0.get("type2", 0))

        obs = [
            float(map_id) / 255.0,
            float(x) / 255.0,
            float(y) / 255.0,
            in_battle,
            badges / 8.0,
            min(money, 999999.0) / 999999.0,
            text_state / 255.0,

            tileset / 255.0,
            mheight / 255.0,
            mwidth / 255.0,
            mconn / 255.0,

            party_count / 6.0,
        ]

        # party hp already 0..1
        obs.extend([float(v) for v in party_hp[:6]])

        # party levels normalize by 100 (cap defensively)
        for v in party_lv[:6]:
            vv = float(v)
            if vv < 0:
                vv = 0.0
            if vv > 100:
                vv = 100.0
            obs.append(vv / 100.0)

        # party status bytes /255
        obs.extend([float(int(v) & 0xFF) / 255.0 for v in party_st[:6]])

        # battle context
        obs.append(min(pslot, 5.0) / 5.0)
        obs.append(min(eslot, 5.0) / 5.0)
        obs.append(attack_missed)
        obs.append(min(pending_damage, 255.0) / 255.0)
        obs.append(min(turn_counter, 255.0) / 255.0)
        obs.append(min(sub_p, 255.0) / 255.0)
        obs.append(min(sub_e, 255.0) / 255.0)

        # opponent summary
        obs.append(roster / 6.0)
        obs.append(float(opp0_hp_ratio))
        obs.append(min(opp0_level, 100.0) / 100.0)
        obs.append(min(opp0_status, 255.0) / 255.0)
        obs.append(min(opp0_type1, 15.0) / 15.0)
        obs.append(min(opp0_type2, 15.0) / 15.0)

        # Ensure exact length
        if len(obs) < 43:
            obs.extend([0.0] * (43 - len(obs)))
        elif len(obs) > 43:
            obs = obs[:43]

        return obs
    except Exception:
        return [0.0] * 43

# --- Human-friendly name mappings and game-variant helpers (Yellow quirks) ---
# Note: Move IDs and Type IDs in Gen 1 can be extracted from the ROM's text tables.
# We include compact maps for common IDs and a fallback resolution function. For
# full accuracy, use `load_move_table_from_rom(rom_path)` when you have the ROM
# available.

# Type ID mapping (Gen 1 ordering, best-effort mapping -- can be overridden)
TYPE_ID_TO_NAME = {
    0: 'Normal', 1: 'Fighting', 2: 'Flying', 3: 'Poison', 4: 'Ground',
    5: 'Rock', 6: 'Bug', 7: 'Ghost', 8: 'Fire', 9: 'Water', 10: 'Grass',
    11: 'Electric', 12: 'Psychic', 13: 'Ice', 14: 'Dragon'
}

# Minimal move ID map for common moves used in tests / debugging. This is
# intentionally small; users can load a complete table from the ROM if needed.
MOVE_ID_TO_NAME = {
    1: 'Pound', 2: 'Karate Chop', 3: 'Double Slap', 4: 'Comet Punch',
    15: 'Thunderbolt', 17: 'Thunder', 33: 'Quick Attack', 45: 'Growl'
}

def type_id_to_name(type_id: int) -> str:
    """Return the human-readable name for a type id (falls back to 'Type #')."""
    return TYPE_ID_TO_NAME.get(type_id, f"Type_{type_id}")


def move_id_to_name(move_id: int) -> str:
    """Return the human-readable name for a move id (falls back to 'Move #')."""
    return MOVE_ID_TO_NAME.get(move_id, f"Move_{move_id}")


def load_move_table_from_rom(rom_path: str) -> dict:
    """Attempt to read the move names from the ROM's text table.

    This is best-effort. If it fails, returns an empty dict. It is useful for
    making the inspector fully human-readable using the actual ROM being used
    for emulation.
    """
    try:
        with open(rom_path, 'rb') as f:
            rom = f.read()
        # Heuristic: Search for the familiar 'MOVE_NAMES' table signature by
        # scanning expected text entries. This is a conservative approach and
        # may require refinement for all ROM variants.
        # If successful, update MOVE_ID_TO_NAME with discovered names.
        # For now, this is a placeholder that returns an empty mapping.
        return {}
    except Exception:
        return {}


def get_raw_ram_slice(ram_data: bytes, start: int = 0, length: int = 0x1000) -> bytes:
    """Get a slice of RAM data from the 0xD000 WRAM window.

    By default this returns a 0x1000 (4KB) slice consistent with the
    offsets used in this module (offsets are relative to 0xD000).
    """
    try:
        ram = to_d000_slice(ram_data)
        return ram[start:start + length]
    except Exception:
        return b""


# Utility accessors
def get_byte(ram_data: bytes, offset: int) -> int:
    """Return a single byte at the given offset (relative to 0xD000).

    The function accepts either a 0xD000 slice or a full 0xC000..0xDFFF dump
    and will normalize automatically.
    """
    ram = to_d000_slice(ram_data)
    return ram[offset]


def get_word_le(ram_data: bytes, offset: int) -> int:
    """Read a little-endian 16-bit value at the given offset (relative to 0xD000)."""
    ram = to_d000_slice(ram_data)
    lo = ram[offset]
    hi = ram[offset + 1]
    return lo | (hi << 8)


def decode_battle_type(ram_data: bytes) -> int:
    """Return the current battle type value from D05A (0 when not in battle).

    See `RAM_ADDRESSES['BATTLE_TYPE']` for the offset. This is useful to
    differentiate between normal battles and special contexts (safari, old man, etc.).
    """
    try:
        return get_byte(ram_data, RAM_ADDRESSES['BATTLE_TYPE'])
    except (IndexError, ValueError):
        return 0


def to_d000_slice(ram_data: bytes, base_address: Optional[int] = None) -> bytes:
    """Return a 0x1000 slice corresponding to addresses 0xD000..0xDFFF.

    Accepts either:
      - a buffer where index 0 corresponds to address 0xD000 (length >= 0x1000)
      - a buffer that starts at 0xC000 (length >= 0x2000) and extracts the
        0xD000..0xDFFF portion automatically
      - a buffer with an explicit ``base_address`` specifying the absolute
        address of `ram_data[0]` (useful for custom memory dumps)

    Raises:
      - ValueError if the function cannot map to a valid 0xD000..0xDFFF slice.
    """
    # If caller provided a base address, compute the index of 0xD000
    if base_address is not None:
        idx = 0xD000 - base_address
        if idx < 0 or idx + 0x1000 > len(ram_data):
            raise ValueError("Provided base_address and ram_data length cannot map to 0xD000..0xDFFF")
        return ram_data[idx:idx + 0x1000]

    # Try to infer from length
    if len(ram_data) >= 0x1000 and len(ram_data) < 0x2000:
        # Assume this is already the 0xD000..0xDFFF window (or equivalent 4KB slice)
        return ram_data[:0x1000]

    if len(ram_data) >= 0x2000:
        # Common case: buffer covers 0xC000..0xDFFF (8KB), so 0xD000 starts at offset 0x1000
        d000_offset = 0xD000 - 0xC000
        return ram_data[d000_offset:d000_offset + 0x1000]

    raise ValueError("ram_data length not long enough to infer a 0xD000..0xDFFF slice")

def decode_map_header(ram_data: bytes) -> dict:
    """Decode basic map header fields (tileset, height/width, connection byte)."""
    try:
        ram = to_d000_slice(ram_data)
        return {
            "tileset": int(ram[RAM_ADDRESSES["MAP_TILESET"]]) & 0xFF,
            "height": int(ram[RAM_ADDRESSES["MAP_HEIGHT"]]) & 0xFF,
            "width": int(ram[RAM_ADDRESSES["MAP_WIDTH"]]) & 0xFF,
            "connections": int(ram[RAM_ADDRESSES["MAP_CONNECTIONS"]]) & 0xFF,
        }
    except Exception:
        return {"tileset": 0, "height": 0, "width": 0, "connections": 0}
    
def _u24_be(b0: int, b1: int, b2: int) -> int:
    """Gen 1 EXP is commonly stored as 3 bytes, big-endian."""
    return ((int(b0) & 0xFF) << 16) | ((int(b1) & 0xFF) << 8) | (int(b2) & 0xFF)

def decode_party_levels(ram_data: bytes) -> list:
    """Return up to 6 party levels."""
    out = [0] * 6
    try:
        ram = to_d000_slice(ram_data)
        count = min(int(ram[RAM_ADDRESSES['PARTY_COUNT']]) & 0xFF, 6)
        stride = RAM_ADDRESSES['PARTY_BLOCK_SIZE']
        base = RAM_ADDRESSES['PARTY_LEVEL']
        for i in range(count):
            out[i] = int(ram[base + i * stride]) & 0xFF
    except Exception:
        pass
    return out

def decode_party_statuses(ram_data: bytes) -> list:
    """Return up to 6 party status bytes."""
    out = [0] * 6
    try:
        ram = to_d000_slice(ram_data)
        count = min(int(ram[RAM_ADDRESSES['PARTY_COUNT']]) & 0xFF, 6)
        stride = RAM_ADDRESSES['PARTY_BLOCK_SIZE']
        base = RAM_ADDRESSES['PARTY_STATUS']
        for i in range(count):
            out[i] = int(ram[base + i * stride]) & 0xFF
    except Exception:
        pass
    return out

def decode_party_moves(ram_data: bytes) -> list:
    """Return up to 6 party move lists (each 4 move IDs)."""
    out = [[0, 0, 0, 0] for _ in range(6)]
    try:
        ram = to_d000_slice(ram_data)
        count = min(int(ram[RAM_ADDRESSES['PARTY_COUNT']]) & 0xFF, 6)
        stride = RAM_ADDRESSES['PARTY_BLOCK_SIZE']
        base = RAM_ADDRESSES['PARTY_MOVES']
        for i in range(count):
            o = base + i * stride
            out[i] = [int(ram[o + j]) & 0xFF for j in range(4)]
    except Exception:
        pass
    return out

def decode_party_exp(ram_data: bytes) -> list:
    """Return up to 6 party EXP values as ints."""
    out = [0] * 6
    try:
        ram = to_d000_slice(ram_data)
        count = min(int(ram[RAM_ADDRESSES['PARTY_COUNT']]) & 0xFF, 6)
        stride = RAM_ADDRESSES['PARTY_BLOCK_SIZE']
        base = RAM_ADDRESSES['PARTY_EXP']
        for i in range(count):
            o = base + i * stride
            out[i] = _u24_be(ram[o], ram[o + 1], ram[o + 2])
    except Exception:
        pass
    return out
    
def decode_battle_context(ram_data: bytes) -> dict:
    """Battle context signals useful for shaping + debugging."""
    try:
        ram = to_d000_slice(ram_data)
        return {
            "player_party_slot": int(ram[RAM_ADDRESSES["BATTLE_PLAYER_PARTY_SLOT"]]) & 0xFF,
            "enemy_party_slot": int(ram[RAM_ADDRESSES["BATTLE_ENEMY_PARTY_SLOT"]]) & 0xFF,
            "attack_missed": int(ram[RAM_ADDRESSES["BATTLE_ATTACK_MISSED"]]) & 0xFF,
            "pending_damage": int(ram[RAM_ADDRESSES["DAMAGE_TO_BE_DEALT"]]) & 0xFF,
        }
    except Exception:
        return {"player_party_slot": 0, "enemy_party_slot": 0, "attack_missed": 0, "pending_damage": 0}    
    
def get_abs_word_le(ram_data: bytes, abs_addr: int, base_address: Optional[int] = None) -> int:
    """Read a little-endian 16-bit value at an absolute WRAM address."""
    lo = get_abs_byte(ram_data, abs_addr, base_address=base_address)
    hi = get_abs_byte(ram_data, abs_addr + 1, base_address=base_address)
    return (hi << 8) | lo

def decode_battle_turn_counter(ram_data: bytes) -> int:
    """Return battle turn counter (0 if unavailable)."""
    try:
        return int(get_abs_byte(ram_data, ABS_WRAM_ADDRESSES['BATTLE_TURN_COUNTER'])) & 0xFF
    except Exception:
        return 0

def decode_substitute_hp(ram_data: bytes) -> dict:
    """Return substitute HP bytes (player/enemy)."""
    try:
        return {
            "player": int(get_abs_byte(ram_data, ABS_WRAM_ADDRESSES['PLAYER_SUBSTITUTE_HP'])) & 0xFF,
            "enemy": int(get_abs_byte(ram_data, ABS_WRAM_ADDRESSES['ENEMY_SUBSTITUTE_HP'])) & 0xFF,
        }
    except Exception:
        return {"player": 0, "enemy": 0}    