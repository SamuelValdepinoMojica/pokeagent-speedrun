"""
Fast lightweight state reader for DRL training
Only reads the minimal data needed for observations (not the full game state)
"""

from typing import Dict, Any, Tuple
import numpy as np


def get_fast_observation_data(memory_reader) -> Dict[str, Any]:
    """
    Read ONLY the data needed for DRL observations (much faster than get_comprehensive_state)
    
    Returns dict with:
    - position: (x, y)
    - tiles: 7x7 local map
    - party: Basic party info (levels, HP)
    - badges: Badge count
    - money: Money
    - in_battle: Boolean
    """
    # Read player position (fast - just 2 bytes)
    x = memory_reader.read_byte(0x02037318)  # X position
    y = memory_reader.read_byte(0x02037319)  # Y position
    
    # Read local 7x7 map (medium speed - need to extract tiles)
    # For now, get minimal tile data
    tiles = []  # TODO: Implement fast tile reading
    
    # Read party size (1 byte)
    party_count = memory_reader.read_byte(0x02024284)
    
    # Read party Pokemon data (only what we need: level, HP)
    party = []
    for i in range(min(party_count, 6)):
        offset = 0x02024284 + 4 + (i * 100)  # Party structure offset
        level = memory_reader.read_byte(offset + 0x38)
        current_hp = memory_reader.read_word(offset + 0x56)
        max_hp = memory_reader.read_word(offset + 0x58)
        party.append({
            'level': level,
            'current_hp': current_hp,
            'max_hp': max_hp
        })
    
    # Read badges (2 bytes)
    badges_raw = memory_reader.read_word(0x020244EC)
    badge_count = bin(badges_raw).count('1')
    
    # Read money (4 bytes)
    money = memory_reader.read_dword(0x020244E0)
    
    # Read battle flag (1 byte)
    in_battle = memory_reader.read_byte(0x02022B4C) != 0
    
    return {
        'position': (x, y),
        'tiles': tiles,  # Empty for now - can optimize later
        'party': party,
        'badges': badge_count,
        'money': money,
        'in_battle': in_battle,
        'party_count': party_count
    }
