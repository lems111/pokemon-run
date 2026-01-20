"""
Memory map constants and helper decoders for Pokémon Yellow.
"""
import struct
from typing import Optional

# RAM Addresses - These are offsets relative to the WRAM window at 0xD000..0xDFFF (0x1000 bytes)
# The memory_map constants are provided as offsets from 0xD000 (e.g. 0x35E -> absolute 0xD35E)
# Values derived from the Pokémon R/B/Y RAM map (DataCrystal / pret disassembly)
# Pokemon Yellow - Most of the RAM map (but not all) for this game has an offset of -1 from the one on Red and Blue.
RAM_ADDRESSES = {
    # Current map and player coordinates (relative to 0xD000)
    # D35E = Current map ID
    # D361 = Player Y-position (1 byte)
    # D362 = Player X-position (1 byte)
    'PLAYER_MAP_ID': 0x35D,  # absolute 0xD35E
    'PLAYER_MAP_Y': 0x360,   # absolute 0xD361
    'PLAYER_MAP_X': 0x361,   # absolute 0xD362

    # Battle state
    # D057 = Type of battle (0 when not in battle)
    # D05A = Battle type (normal/safari/old man...)
    'IN_BATTLE_FLAG': 0x056,  # absolute 0xD057
    'BATTLE_TYPE': 0x059,     # absolute 0xD05A

    # Party data (arrays of Pokémon data blocks)
    # D163 = Party size (# Pokémon in party)
    # D16C.. = current HP (2 bytes per mon), D18D.. = max HP (2 bytes per mon)
    'PARTY_COUNT': 0x162,
    'PARTY_CURRENT_HP': 0x16B,
    'PARTY_MAX_HP': 0x18C,
    'PARTY_BLOCK_SIZE': 0x2C,  # bytes between successive party entries

    # Game progress
    # D356 contains badges as binary switches (bits set = badge owned)
    'BADGES': 0x355,
    # D347-D349 = money bytes (little-endian 3-byte integer)
    'MONEY': 0x346,

    # Text/display-related (see notes) - D358/D359 are text-related flags/IDs
    'TEXT_STATE': 0x357,
    # Misc options (D355) - encodes battle animation, battle style, text speed nybble
    'OPTIONS': 0x354,

    # Player tile/block coordinates (D363/D364) - finer-grained block positions
    'PLAYER_TILE_Y': 0x362,
    'PLAYER_TILE_X': 0x363,

    # Event displacement / small event offsets
    'EVENT_DISPLACEMENT_1': 0x35E,
    'EVENT_DISPLACEMENT_2': 0x35F,

    # Player ID bytes (D359..D35A)
    'PLAYER_ID_LOW': 0x358,
    'PLAYER_ID_HIGH': 0x359,

    # Player name (D158..D162, inclusive) - useful for diagnostics
    'PLAYER_NAME': 0x157,
    'PLAYER_NAME_LEN': 5,

    # Battle-related helper
    'BATTLE_MUSIC': 0x05B,
    'BATTLE_STATUS': 0x061,  # D062-D064 battle status bytes (player)
    'BATTLE_CRITICAL_FLAG': 0x05D,
    'BATTLE_HOOK_FLAG': 0x05E,

    # Event flags base (D5A6..D5C5 contains many region-specific flags)
    'EVENT_FLAGS_BASE': 0x5A5,

    # Opponent/trainer data and in-battle structures start in higher D offsets
    # (e.g. opponent roster at ~0x8A4). Add on demand.
    'OPPONENT_ROSTER_COUNT': 0x89B,
    'OPPONENT_POKEMON_BASE': 0x8A3,
    'OPPONENT_POKEMON_BLOCK_SIZE': 0x10,

    # Game time (DA40..DA45)
    'GAME_TIME_HOURS': 0xA3F,
    'GAME_TIME_MINUTES': 0xA41,
    'GAME_TIME_SECONDS': 0xA43,
    'GAME_TIME_FRAMES': 0xA44,

    # Note: many additional addresses exist (player direction, battle internals, etc.)
    # See DataCrystal / Red & Blue RAM map for more details; add on demand.
}

# Helper functions to decode RAM values

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


def decode_money(ram_data: bytes) -> int:
    """Read 3-byte little-endian money value from D347..D349.

    Accepts WRAM slice or full dump; normalizes via `to_d000_slice`.
    Returns an integer amount (0 on out-of-bounds or malformed data).
    """
    try:
        ram = to_d000_slice(ram_data)
        start = RAM_ADDRESSES['MONEY']
        money_bytes = ram[start:start + 3]
        if len(money_bytes) != 3:
            return 0
        return money_bytes[0] | (money_bytes[1] << 8) | (money_bytes[2] << 16)
    except (IndexError, ValueError):
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
    """Decode basic info about the opponent's Pokémon at `index`.

    The opponent roster is located at D89C (count) and D8A4.. (each slot ~0x10 bytes).
    This function returns a dict with keys: 'id', 'current_hp', 'max_hp', 'level',
    'status', 'type1', 'type2', 'moves' (list), and 'raw_block'.

    Offsets within the block are conservative and chosen for compatibility with
    common Red/Blue/Yellow mappings. Tests use these offsets directly.
    """
    try:
        ram = to_d000_slice(ram_data)
        roster_count = ram[RAM_ADDRESSES['OPPONENT_ROSTER_COUNT']]
        if index < 0 or index >= roster_count:
            raise IndexError("opponent index out of range")
        base = RAM_ADDRESSES['OPPONENT_POKEMON_BASE'] + index * RAM_ADDRESSES['OPPONENT_POKEMON_BLOCK_SIZE']
        poke_id = ram[base]
        cur_hp = ram[base + 1] | (ram[base + 2] << 8)
        # Max HP commonly resides a few bytes later in the block
        max_hp = ram[base + 4] | (ram[base + 5] << 8)
        level = ram[base + 6]
        status = ram[base + 3]
        type1 = ram[base + 7]
        type2 = ram[base + 8]
        moves = [ram[base + 9], ram[base + 10], ram[base + 11], ram[base + 12]]
        raw_block = ram[base: base + RAM_ADDRESSES['OPPONENT_POKEMON_BLOCK_SIZE']]
        return {
            'id': poke_id,
            'current_hp': cur_hp,
            'max_hp': max_hp,
            'level': level,
            'status': status,
            'type1': type1,
            'type2': type2,
            'moves': moves,
            'raw_block': raw_block,
        }
    except (IndexError, ValueError):
        return {'id': 0, 'current_hp': 0, 'max_hp': 0, 'level': 0, 'status': 0, 'type1': 0, 'type2': 0, 'moves': [], 'raw_block': b''}


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
        roster = ram[RAM_ADDRESSES['OPPONENT_ROSTER_COUNT']]
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
    """Create compact observation vector from RAM data.

    Vector layout:
      [map_id, player_x, player_y, in_battle, badges_count, money, text_state, party_hp[0..5]]

    Accepts WRAM slice or full dump; normalizes via `to_d000_slice`.
    """
    ram = to_d000_slice(ram_data)
    position = decode_player_position(ram)
    in_battle = decode_in_battle(ram)
    badges = decode_badge_count(ram)
    money = decode_money(ram)
    party_hp = decode_party_stats(ram)
    text_state = decode_text_state(ram)

    obs_vector = [
        position[0],  # map_id
        position[1],  # player_x
        position[2],  # player_y
        int(in_battle),
        badges,
        money,
        text_state,
    ]
    obs_vector.extend(party_hp)
    return obs_vector


def get_observation_vector(ram_data: bytes) -> list:
    """Create compact observation vector from RAM data.

    Vector layout:
      [map_id, player_x, player_y, in_battle, badges_count, money, text_state, party_hp[0..5]]
    """
    position = decode_player_position(ram_data)
    in_battle = decode_in_battle(ram_data)
    badges = decode_badge_count(ram_data)
    money = decode_money(ram_data)
    party_hp = decode_party_stats(ram_data)
    text_state = decode_text_state(ram_data)

    obs_vector = [
        position[0],  # map_id
        position[1],  # player_x
        position[2],  # player_y
        int(in_battle),
        badges,
        money,
        text_state,
    ]
    obs_vector.extend(party_hp)
    return obs_vector


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

GAME_VARIANT = None


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


def set_game_variant(variant: str):
    """Set game variant to apply any known quirks for parsing RAM.

    Supported variants: 'yellow' (applies Yellow-specific quirks such as
    certain offset adjustments and name lengths). This function mutates
    `RAM_ADDRESSES` in-place only for known adjustments and records the
    selected variant in `GAME_VARIANT`.
    """
    global GAME_VARIANT
    variant = variant.lower().strip()
    if variant == 'yellow':
        apply_yellow_quirks()
        GAME_VARIANT = 'yellow'
    else:
        GAME_VARIANT = variant


def apply_yellow_quirks():
    """Apply a small set of Yellow-specific corrections to `RAM_ADDRESSES`.

    DataCrystal notes that many offsets in Yellow are shifted by -1 relative
    to Red/Blue. This function applies a conservative set of overrides that
    are known to differ in Yellow. It avoids global blind shifts to prevent
    regressions; use `apply_yellow_shift_all` if you explicitly want to
    apply a -1 offset to a conservative set of addresses.
    """
    # Conservative, well-documented adjustments for Yellow (do not change
    # values that are already correct to avoid regressions)
    overrides = {
        # Player name length in Yellow is commonly 7 bytes (vs 5 in some tables)
        'PLAYER_NAME_LEN': 7,
    }
    RAM_ADDRESSES.update(overrides)


# Keys that are known to commonly be shifted by -1 in Yellow vs Red/Blue.
YELLOW_SHIFTED_KEYS = [
    'PLAYER_MAP_ID', 'PLAYER_MAP_Y', 'PLAYER_MAP_X',
    'PARTY_COUNT', 'PARTY_CURRENT_HP', 'PARTY_MAX_HP', 'PARTY_BLOCK_SIZE',
    'BADGES', 'MONEY', 'TEXT_STATE', 'OPTIONS', 'PLAYER_TILE_Y', 'PLAYER_TILE_X',
    'PLAYER_ID_LOW', 'PLAYER_ID_HIGH', 'PLAYER_NAME'
]


def shift_offsets(offset_map: dict, keys: list, delta: int = -1) -> dict:
    """Return a new offset map with the given keys shifted by `delta`.

    Does not mutate the original map; returns a shallow copy with adjustments.
    """
    new_map = dict(offset_map)
    for k in keys:
        if k in new_map and isinstance(new_map[k], int):
            new_map[k] = new_map[k] + delta
    return new_map


def apply_yellow_shift_all():
    """Apply a -1 shift to a conservative set of addresses in-place.

    WARNING: This mutates global `RAM_ADDRESSES`. Use only if you are
    certain your offsets are Red/Blue-derived and need Yellow adjustments.
    """
    global RAM_ADDRESSES
    RAM_ADDRESSES = shift_offsets(RAM_ADDRESSES, YELLOW_SHIFTED_KEYS, -1)


def get_game_variant() -> Optional[str]:
    return GAME_VARIANT


def get_raw_ram_slice(ram_data: bytes, start: int = 0, length: int = 0x1000) -> bytes:
    """Get a slice of RAM data from the 0xD000 WRAM window.

    By default this returns a 0x1000 (4KB) slice consistent with the
    offsets used in this module (offsets are relative to 0xD000).
    """
    return ram_data[start:start + length]


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
