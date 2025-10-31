"""
Compare actual state data returned by comprehensive vs lightweight readers.
Shows EXACTLY what data the agent receives in each case.
"""

import sys
import numpy as np
from pokemon_env.emulator import EmeraldEmulator
from agent.lightweight_state_reader import LightweightStateReader


def format_dict(d, indent=0):
    """Pretty print dictionary with indentation"""
    lines = []
    for key, value in d.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(format_dict(value, indent + 1))
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            lines.append(f"{prefix}{key}: [{len(value)} items]")
            if len(value) > 0:
                lines.append(f"{prefix}  Example item:")
                lines.append(format_dict(value[0], indent + 2))
        elif isinstance(value, list) and len(value) > 3:
            lines.append(f"{prefix}{key}: [{len(value)} items] {value[:3]}...")
        elif isinstance(value, np.ndarray):
            lines.append(f"{prefix}{key}: ndarray shape={value.shape} dtype={value.dtype}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


def compare_states():
    """Compare comprehensive vs lightweight state data"""
    
    print("="*80)
    print("STATE COMPARISON: Comprehensive vs Lightweight")
    print("="*80)
    print()
    
    # Initialize emulator
    print("Initializing emulator...")
    emulator = EmeraldEmulator(
        rom_path="Emerald-GBAdvance/rom.gba",
        headless=True
    )
    emulator.initialize()
    emulator.load_state("Emerald-GBAdvance/quick_start_save.state")
    print("✓ Emulator ready\n")
    
    # === COMPREHENSIVE STATE ===
    print("="*80)
    print("1. COMPREHENSIVE STATE (get_comprehensive_state)")
    print("="*80)
    print()
    
    import time
    start = time.time()
    comprehensive = emulator.get_comprehensive_state()
    comp_time = (time.time() - start) * 1000
    
    print(f"⏱️  Read time: {comp_time:.2f}ms\n")
    
    # Show structure
    print("📋 Top-level keys:")
    for key in comprehensive.keys():
        value = comprehensive[key]
        if isinstance(value, dict):
            print(f"  • {key}: dict with {len(value)} keys")
        elif isinstance(value, list):
            print(f"  • {key}: list with {len(value)} items")
        else:
            print(f"  • {key}: {type(value).__name__}")
    print()
    
    # Show player info
    print("👤 Player Info (comprehensive):")
    player = comprehensive.get('player', {})
    print(f"  • Position: {player.get('position')}")
    print(f"  • Location: {player.get('location')}")
    print(f"  • Name: {player.get('name')}")
    if player.get('party'):
        print(f"  • Party size: {len(player['party'])} Pokemon")
        if len(player['party']) > 0:
            pokemon = player['party'][0]
            print(f"    - {pokemon.get('species_name')} Lv.{pokemon.get('level')}")
            print(f"      HP: {pokemon.get('current_hp')}/{pokemon.get('max_hp')}")
            print(f"      Status: {pokemon.get('status')}")
            print(f"      Types: {pokemon.get('types')}")
            print(f"      Moves: {pokemon.get('moves')}")
            print(f"      Move PP: {pokemon.get('move_pp')}")
    print()
    
    # Show game info
    print("🎮 Game Info (comprehensive):")
    game = comprehensive.get('game', {})
    print(f"  • Money: ${game.get('money')}")
    print(f"  • Badges: {game.get('badges')}")
    print(f"  • Game state: {game.get('game_state')}")
    print(f"  • In battle: {game.get('is_in_battle')}")
    print(f"  • Items: {game.get('item_count')} items")
    print(f"  • Pokedex caught: {game.get('pokedex_caught')}")
    print(f"  • Pokedex seen: {game.get('pokedex_seen')}")
    dialog_text = game.get('dialog_text')
    if dialog_text:
        print(f"  • Dialog text: {dialog_text[:50]}...")
    else:
        print(f"  • Dialog text: None")
    print()
    
    # Show map info
    print("🗺️  Map Info (comprehensive):")
    map_data = comprehensive.get('map', {})
    tiles = map_data.get('tiles', [])
    if tiles:
        print(f"  • Tiles grid: {len(tiles)}x{len(tiles[0])}")
        print(f"  • Total tiles: {len(tiles) * len(tiles[0])}")
        if len(tiles) > 0 and len(tiles[0]) > 0:
            center_tile = tiles[len(tiles)//2][len(tiles[0])//2]
            print(f"  • Center tile example: {center_tile[:4] if len(center_tile) >= 4 else center_tile}")
    print()
    
    # === LIGHTWEIGHT STATE ===
    print("="*80)
    print("2. LIGHTWEIGHT STATE (LightweightStateReader)")
    print("="*80)
    print()
    
    state_reader = LightweightStateReader(emulator.memory_reader)
    
    start = time.time()
    lightweight = state_reader.get_drl_state(map_radius=3)
    light_time = (time.time() - start) * 1000
    
    print(f"⏱️  Read time: {light_time:.2f}ms ({comp_time/light_time:.1f}x faster!)\n")
    
    # Show structure
    print("📋 Top-level keys:")
    for key in lightweight.keys():
        value = lightweight[key]
        if isinstance(value, dict):
            print(f"  • {key}: dict with {len(value)} keys")
        elif isinstance(value, list):
            print(f"  • {key}: list with {len(value)} items")
        else:
            print(f"  • {key}: {type(value).__name__}")
    print()
    
    # Show detailed content
    print("📦 Complete lightweight state:")
    print(format_dict(lightweight, indent=1))
    print()
    
    # === LIGHTWEIGHT OBSERVATION ===
    print("="*80)
    print("3. LIGHTWEIGHT OBSERVATION (for DRL agent)")
    print("="*80)
    print()
    
    start = time.time()
    observation = state_reader.get_observation_for_drl(map_radius=3)
    obs_time = (time.time() - start) * 1000
    
    print(f"⏱️  Read time: {obs_time:.2f}ms\n")
    
    print("🤖 Observation format (what agent actually sees):")
    print(f"  • Type: Dict with 2 keys: 'map' and 'vector'")
    print()
    print(f"  • map: shape={observation['map'].shape} dtype={observation['map'].dtype}")
    print(f"    - Channel 0: Metatile ID (normalized 0-1)")
    print(f"    - Channel 1: Behavior (normalized 0-1)")
    print(f"    - Channel 2: Collision (0 or 1)")
    print()
    print(f"  • vector: shape={observation['vector'].shape} dtype={observation['vector'].dtype}")
    print(f"    - Elements 0-1: Position (x, y) normalized")
    print(f"    - Elements 2-13: Party Pokemon (4 features × 3 Pokemon)")
    print(f"    - Elements 14-17: Game state (badges, in_battle, reserved)")
    print()
    
    # Show vector values
    print("📊 Vector values:")
    vec = observation['vector']
    print(f"  Position: [{vec[0]:.3f}, {vec[1]:.3f}]")
    print(f"  Pokemon 1: [{vec[2]:.3f}, {vec[3]:.3f}, {vec[4]:.3f}, {vec[5]:.3f}]")
    print(f"  Pokemon 2: [{vec[6]:.3f}, {vec[7]:.3f}, {vec[8]:.3f}, {vec[9]:.3f}]")
    print(f"  Pokemon 3: [{vec[10]:.3f}, {vec[11]:.3f}, {vec[12]:.3f}, {vec[13]:.3f}]")
    print(f"  Game state: [{vec[14]:.3f}, {vec[15]:.3f}, {vec[16]:.3f}, {vec[17]:.3f}]")
    print()
    
    # Show map sample
    print("🗺️  Map sample (center 3x3):")
    map_arr = observation['map']
    for i in range(3, 4):  # Just show center row
        for j in range(2, 5):  # Center 3 tiles
            tile = map_arr[i, j]
            print(f"  Tile[{i},{j}]: id={tile[0]:.3f}, behavior={tile[1]:.3f}, collision={tile[2]:.0f}")
    print()
    
    # === COMPARISON SUMMARY ===
    print("="*80)
    print("4. COMPARISON SUMMARY")
    print("="*80)
    print()
    
    print("⏱️  SPEED:")
    print(f"  • Comprehensive: {comp_time:.2f}ms → ~{1000/comp_time:.0f} FPS")
    print(f"  • Lightweight: {light_time:.2f}ms → ~{1000/light_time:.0f} FPS")
    print(f"  • Speedup: {comp_time/light_time:.1f}x faster")
    print()
    
    print("📦 DATA SIZE:")
    comp_tiles = len(comprehensive.get('map', {}).get('tiles', [])) ** 2
    light_tiles = len(lightweight.get('map_tiles', [])) ** 2 if lightweight.get('map_tiles') else 0
    print(f"  • Comprehensive map: {comp_tiles} tiles")
    print(f"  • Lightweight map: {light_tiles} tiles ({100*light_tiles/comp_tiles:.0f}% of comprehensive)")
    print()
    
    print("ℹ️  INFORMATION CONTENT:")
    print("  COMPREHENSIVE has:")
    print("    ✓ Location names (strings)")
    print("    ✓ Dialog text (with OCR)")
    print("    ✓ Full battle info")
    print("    ✓ Items inventory")
    print("    ✓ Pokedex counts")
    print("    ✓ Money")
    print("    ✓ Full Pokemon stats (moves, PP, attack, defense, etc.)")
    print("    ✓ 15x15 map with full metadata")
    print()
    print("  LIGHTWEIGHT has:")
    print("    ✓ Position (x, y)")
    print("    ✓ Party basics (species, level, HP, status)")
    print("    ✓ Badges count")
    print("    ✓ In battle flag")
    print("    ✓ 7x7 map with essential data")
    print()
    
    print("🎯 FOR DRL TRAINING:")
    print("  • Lightweight is SUFFICIENT")
    print("  • 11x speed improvement enables practical training")
    print("  • Agent learns through exploration, not by 'reading' game text")
    print()
    
    # Cleanup
    emulator.stop()
    print("✓ Done!")


if __name__ == "__main__":
    compare_states()
