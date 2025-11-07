"""
Lightweight state reader for DRL training - reads ONLY what's needed for observations.
This dramatically reduces memory read overhead compared to get_comprehensive_state().
"""
from typing import Dict, Any, List, Tuple
import numpy as np


class LightweightStateReader:
    """Minimal state reader for DRL environment - fast but limited functionality"""
    
    def __init__(self, memory_reader):
        """
        Args:
            memory_reader: The full MemoryReader instance for fallback
        """
        self.mem = memory_reader
    
    def get_drl_state(self, map_radius: int = 3) -> Dict[str, Any]:
        """
        Get minimal state needed for DRL observations.
        
        Reads only:
        - Position (x, y)
        - Map tiles around player (radius × radius grid)
        - Party Pokemon basics (species, HP, level)
        - Key game flags (badges count, in_battle)
        
        This is MUCH faster than get_comprehensive_state() because it:
        1. Reads smaller map (7x7 vs 15x15)
        2. Skips dialog detection / OCR
        3. Skips battle details
        4. Skips items, pokedex, money, time
        5. Reads minimal party info
        
        Args:
            map_radius: Radius around player to read (3 = 7x7 grid)
            
        Returns:
            Dict with keys: position, map_tiles, party, badges, in_battle
        """
        state = {
            "position": None,
            "map_tiles": None,
            "party": [],
            "badges": 0,
            "in_battle": False,
            "milestone_count": 0  # 🆕 Agregar contador de milestones
        }
        
        try:
            # 1. Position - fast, always works
            coords = self.mem.read_coordinates()
            state["position"] = {"x": coords[0], "y": coords[1]}
            
            # 2. Map tiles - read smaller radius for speed
            # CRITICAL: Validate map buffer before accessing to prevent segfault
            # Check if map buffer is valid and dimensions are reasonable
            map_buffer_valid = False
            if (hasattr(self.mem, '_map_buffer_addr') and 
                self.mem._map_buffer_addr and 
                hasattr(self.mem, '_map_width') and 
                hasattr(self.mem, '_map_height')):
                # Check dimensions are reasonable (not 0 or corrupted)
                if (self.mem._map_width > 0 and 
                    self.mem._map_height > 0 and 
                    self.mem._map_width < 500 and 
                    self.mem._map_height < 500):
                    map_buffer_valid = True
            
            # Only try to read map if buffer is valid
            if map_buffer_valid:
                try:
                    tiles = self.mem.read_map_around_player(radius=map_radius)
                    if tiles:
                        state["map_tiles"] = tiles
                except:
                    # Map read failed, leave as None
                    pass
            
            # 3. Party Pokemon - minimal info only
            try:
                party = self.mem.read_party_pokemon()
                if party:
                    state["party"] = [
                        {
                            "species_name": pokemon.species_name,
                            "level": pokemon.level,
                            "current_hp": pokemon.current_hp,
                            "max_hp": pokemon.max_hp,
                            "status": pokemon.status.get_status_name() if pokemon.status else "OK"
                        }
                        for pokemon in party[:3]  # Only first 3 Pokemon for speed
                    ]
            except:
                pass
            
            # 4. Badges - just count
            try:
                badges = self.mem.read_badges()
                if badges:
                    state["badges"] = len(badges)
            except:
                pass
            
            # 5. Battle flag - quick check
            try:
                state["in_battle"] = self.mem.is_in_battle()
            except:
                pass
            
            # 6. Milestone count - cuántos hitos completados (🆕)
            try:
                # Acceder al milestone_tracker del emulator si existe
                if (hasattr(self.mem, 'core') and 
                    hasattr(self.mem.core, 'milestone_tracker')):
                    milestone_tracker = self.mem.core.milestone_tracker
                    state["milestone_count"] = len(milestone_tracker.milestones)
            except:
                pass
            
        except Exception as e:
            # Fail silently, return partial state
            pass
        
        return state
    
    def get_observation_for_drl(self, map_radius: int = 3) -> Dict[str, np.ndarray]:
        """
        Get observation in the exact format needed by DRL environment.
        
        Returns:
            Dict with:
                - 'map': ndarray of shape (map_size, map_size, 3) - normalized [0, 1]
                - 'vector': ndarray of shape (19,) - normalized features (incluye milestone_count)
        """
        state = self.get_drl_state(map_radius=map_radius)
        
        # Map size (e.g., 7x7 for radius=3)
        map_size = 2 * map_radius + 1
        
        # Initialize map array (metatile_id, behavior, collision)
        map_array = np.zeros((map_size, map_size, 3), dtype=np.float32)
        
        # Fill map array from tiles
        if state["map_tiles"]:
            tiles = state["map_tiles"]
            for i, row in enumerate(tiles):
                for j, tile_data in enumerate(row):
                    if i < map_size and j < map_size:
                        if len(tile_data) >= 3:
                            tile_id, behavior, collision = tile_data[:3]
                            # Normalize to [0, 1]
                            map_array[i, j, 0] = min(tile_id / 1024.0, 1.0)  # Metatile ID
                            map_array[i, j, 1] = min(behavior / 255.0, 1.0)  # Behavior
                            map_array[i, j, 2] = 1.0 if collision else 0.0    # Collision
        
        # Initialize vector features (19 elements - aumentado para milestone_count)
        vector = np.zeros(19, dtype=np.float32)
        
        # Position (2 features)
        if state["position"]:
            vector[0] = state["position"]["x"] / 1024.0  # Normalize to ~[0, 1]
            vector[1] = state["position"]["y"] / 1024.0
        
        # Party Pokemon (12 features = 4 per Pokemon × 3 Pokemon)
        for i, pokemon in enumerate(state["party"][:3]):
            idx = 2 + i * 4
            # Species ID (approximate from name hash - simple encoding)
            vector[idx] = hash(pokemon.get("species_name", "")) % 256 / 255.0
            # Level
            vector[idx + 1] = pokemon.get("level", 0) / 100.0
            # HP ratio
            current_hp = pokemon.get("current_hp", 0)
            max_hp = pokemon.get("max_hp", 1)
            vector[idx + 2] = current_hp / max(max_hp, 1)
            # Status (0 = OK, 1 = not OK)
            vector[idx + 3] = 0.0 if pokemon.get("status") == "OK" else 1.0
        
        # Game state features (5 features - agregado milestone_count)
        vector[14] = state["badges"] / 8.0  # Badges count (0-8)
        vector[15] = 1.0 if state["in_battle"] else 0.0
        vector[16] = state.get("milestone_count", 0) / 100.0  # 🆕 Milestone count (0-100 normalizado)
        vector[17] = 0.0  # Reserved
        vector[18] = 0.0  # Reserved
        
        return {
            "map": map_array,
            "vector": vector
        }
