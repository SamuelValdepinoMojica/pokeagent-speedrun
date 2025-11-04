"""
Deep Reinforcement Learning Environment for Pokemon Emerald
Compatible with Stable Baselines3 (PPO, DQN, etc.)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

from pokemon_env.emulator import EmeraldEmulator
from agent.lightweight_state_reader import LightweightStateReader
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)


class PokemonEmeraldEnv(gym.Env):
    """
    Gymnasium environment for training DRL agents on Pokemon Emerald.
    
    Features:
    - Uses structured game state (not raw pixels)
    - Automatic reset to saved state
    - Reward shaping for speedrun objectives
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        rom_path: str = "Emerald-GBAdvance/rom.gba",
        initial_state_path: str = "Emerald-GBAdvance/quick_start_save.state",
        render_mode: Optional[str] = None,
        max_steps: int = 10000,
        frame_skip: int = 36,  # Execute action for N frames (6 frames = 10 decisions/sec at 60fps)
        turbo_mode: bool = True,  # If True, runs emulator at max speed (no throttling)
        state_read_interval: int = 1  # Read full game state every N steps (1=every step, 5=every 5 steps)
    ):
        """
        Initialize the Pokemon Emerald DRL environment.
        
        Args:
            rom_path: Path to the ROM file
            initial_state_path: Path to the initial save state
            render_mode: 'human' or 'rgb_array' for visualization
            max_steps: Maximum steps per episode
            frame_skip: Number of frames to execute each action
                       - Lower = more decisions/sec but buttons may not register
                       - Higher = fewer decisions/sec but more reliable
                       - 6 is a good balance (10 decisions/sec at 60fps)
            turbo_mode: Run emulator at maximum speed (no sleep/throttling)
            state_read_interval: Read full game state every N steps for speed
                       - 1 = read every step (slow but accurate)
                       - 5 = read every 5 steps (5x faster, slightly stale data)
        """
        super().__init__()
        
        self.rom_path = rom_path
        self.initial_state_path = initial_state_path
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.turbo_mode = turbo_mode
        self.state_read_interval = state_read_interval
        
        # Initialize OCR
        logger.info("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
        logger.info("PaddleOCR initialized successfully")
        
        # Initialize emulator
        logger.info(f"Initializing emulator with ROM: {rom_path}")
        self.emulator = EmeraldEmulator(rom_path=rom_path, headless=(render_mode != 'human'))
        self.emulator.initialize()
        logger.info("Emulator initialized successfully")
        
        # Initialize lightweight state reader for fast observations
        logger.info("Initializing lightweight state reader for DRL...")
        self.state_reader = LightweightStateReader(self.emulator.memory_reader)
        logger.info("Lightweight reader ready")
        
        # Load initial state ONCE - critical for training speed!
        logger.info(f"Loading initial state from: {initial_state_path}")
        self.emulator.load_state(initial_state_path)
        # Now save it to memory so we can restore quickly
        logger.info("Saving state to memory for fast resets...")
        self.initial_state_bytes = self.emulator.save_state()
        logger.info(f"State cached in memory: {len(self.initial_state_bytes)} bytes")
        
        # Get initial game state and cache it too
        logger.info("Getting initial game state...")
        self.initial_game_state = self.emulator.get_comprehensive_state()
        logger.info(f"Initial game state cached")
        
        # Action space: 8 GBA buttons
        # 0=A, 1=B, 2=SELECT, 3=START, 4=RIGHT, 5=LEFT, 6=UP, 7=DOWN
        self.action_space = spaces.Discrete(8)
        
        self._action_map = {
            0: "a",
            1: "b", 
            2: "select",
            3: "start",
            4: "right",
            5: "left",
            6: "up",
            7: "down"
        }
        
        # Observation space: Dictionary with map image and game state vector
        # Map: 7x7x3 channels (metatile_id, collision, behavior) - for CNN
        # Vector: player position (2) + party info (12) + game state (4) = 18 features
        self.observation_space = spaces.Dict({
            'map': spaces.Box(
                low=0,
                high=1.0,
                shape=(7, 7, 3),  # Height x Width x Channels
                dtype=np.float32
            ),
            'vector': spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(18,),
                dtype=np.float32
            )
        })
        
        # Internal state tracking
        self.prev_game_state = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.cached_game_state = None  # Cache for state_read_interval optimization
        
        # Tracking for reward calculation
        self.prev_location = None
        self.visited_locations = set()
        self.prev_position = None
        self.stationary_steps = 0
        
        logger.info(f"Environment created - Action space: {self.action_space}, Observation space: {self.observation_space.shape}")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to the initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            observation: Initial observation vector
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Get lightweight state for fast reset
        logger.info(f"Resetting environment - Episode complete")
        lightweight_state = self.state_reader.get_drl_state(map_radius=3)
        self.prev_game_state = lightweight_state
        logger.info(f"Reset complete - soft reset (continuing from current position)")
        
        # Reset tracking variables
        logger.info("Resetting tracking variables...")
        self.current_step = 0
        self.episode_reward = 0.0
        position = lightweight_state.get('position', {})
        self.prev_position = (position.get('x', 0), position.get('y', 0))
        self.stationary_steps = 0
        logger.info(f"Initial position: {self.prev_position}")
        
        # Extract observation using lightweight reader
        logger.info("Extracting observation...")
        observation = self.state_reader.get_observation_for_drl(map_radius=3)
        logger.info(f"Observation extracted - Map shape: {observation['map'].shape}, Vector shape: {observation['vector'].shape}")
        
        info = {
            'location': 'Unknown',
            'badges': lightweight_state.get('badges', 0),
            'party_size': len(lightweight_state.get('party', []))
        }
        
        logger.info(f"Environment reset - Party size: {info['party_size']}")
        
        return observation, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one action in the environment with frame skip (Atari-style).
        
        Args:
            action: Action index (0-7)
            
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional information
        """
        self.current_step += 1
        
        # Execute action for frame_skip frames
        button = self._action_map[action]
        total_reward = 0.0
        
        # Repeat action for frame_skip frames (Atari-style)
        for _ in range(self.frame_skip):
            self.emulator.run_frame_with_buttons([button])
        
        # Get lightweight observation directly (much faster than get_comprehensive_state!)
        observation = self.state_reader.get_observation_for_drl(map_radius=3)
        
        # For reward calculation, we still need some game state info
        # But we can get it from the lightweight reader
        lightweight_state = self.state_reader.get_drl_state(map_radius=3)
        
        # Calculate reward using lightweight state
        reward = self._calculate_reward_from_lightweight(self.prev_game_state, lightweight_state)
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = self._check_terminated_from_lightweight(lightweight_state)
        truncated = self.current_step >= self.max_steps
        
        # Prepare info
        info = {
            'location': 'Unknown',  # Location reading is slow, skip for now
            'badges': lightweight_state.get('badges', 0),
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'action_taken': button,
            'frames_executed': self.frame_skip
        }
        
        # Update previous state
        self.prev_game_state = lightweight_state
        
        # Check for text every 100 steps
        if self.current_step % 10 == 0:
            # Get screenshot for OCR
            screenshot = self.emulator.get_screenshot()
            if screenshot:
                # Convert PIL Image to numpy array if needed
                img_array = np.array(screenshot)
                # Run OCR
                response = self.ocr.predict(img_array)
                
                text_list = response[0]['rec_texts']
                
                extracted_text = ' '.join(text_list)
                
                logger.error(f"📝 OCR Text at step {self.current_step}: {extracted_text}")
                    
        return observation, reward, terminated, truncated, info
    
    def _extract_observation(self, game_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert game state to observation for the agent.
        
        Returns a dictionary with:
        - 'map': 7x7x3 array for CNN (metatile_id, collision, behavior)
        - 'vector': 18-element vector (position, party, game state)
        """
        player = game_state.get('player', {})
        position = player.get('position', {})
        
        # === Extract map as 7x7x3 image for CNN ===
        map_data = game_state.get('map', {})
        tiles = map_data.get('tiles', [])
        
        # Initialize 7x7x3 map array
        map_array = np.zeros((7, 7, 3), dtype=np.float32)
        
        if tiles and len(tiles) > 0:
            # Get center 7x7 from the full tile grid
            center_i = len(tiles) // 2
            center_j = len(tiles[0]) // 2 if len(tiles) > 0 else 0
            start_i = max(0, center_i - 3)
            start_j = max(0, center_j - 3)
            
            for i in range(7):
                for j in range(7):
                    tile_i = start_i + i
                    tile_j = start_j + j
                    
                    if tile_i < len(tiles) and tile_j < len(tiles[tile_i]):
                        tile = tiles[tile_i][tile_j]
                        if tile and len(tile) >= 3:
                            # Channel 0: metatile_id (normalized)
                            map_array[i, j, 0] = tile[0] / 1000.0
                            
                            # Channel 1: behavior (extract value from enum if needed)
                            behavior = tile[1]
                            if hasattr(behavior, 'value'):
                                behavior_val = behavior.value
                            else:
                                behavior_val = int(behavior) if behavior is not None else 0
                            map_array[i, j, 1] = behavior_val / 255.0
                            
                            # Channel 2: collision (0 or 1)
                            map_array[i, j, 2] = float(tile[2])
        
        # === Extract vector features (18 total) ===
        vector_features = []
        
        # Player position (2 features)
        vector_features.append(position.get('x', 0) / 100.0)
        vector_features.append(position.get('y', 0) / 100.0)
        
        # Party Pokemon (12 features: 6 Pokemon x 2 values)
        party = player.get('party', [])
        for i in range(6):
            if i < len(party):
                pokemon = party[i]
                vector_features.append(pokemon.get('level', 0) / 100.0)
                current_hp = pokemon.get('current_hp', 0)
                max_hp = pokemon.get('max_hp', 1)
                hp_ratio = current_hp / max(max_hp, 1)
                vector_features.append(hp_ratio)
            else:
                vector_features.extend([0.0, 0.0])
        
        # Game state (4 features)
        game = game_state.get('game', {})
        vector_features.append(min(game.get('money', 0) / 999999.0, 1.0))
        
        # Handle badges - can be int or list
        badges = game.get('badges', 0)
        if isinstance(badges, list):
            badge_count = sum(1 for b in badges if b)
        else:
            badge_count = int(badges) if isinstance(badges, (int, float)) else 0
        vector_features.append(badge_count / 8.0)
        
        vector_features.append(1.0 if game.get('is_in_battle', False) else 0.0)
        vector_features.append(min(game.get('pokedex_caught', 0) / 200.0, 1.0))
        
        return {
            'map': map_array,
            'vector': np.array(vector_features, dtype=np.float32)
        }
    
    def _calculate_reward(
        self,
        prev_state: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward based on state transition.
        """
        reward = 0.0
        
        # Check if game is in a "blocked" state (dialogue, cutscene, etc.)
        # In these states, player can't move, so don't penalize
        is_blocked = self._is_game_blocked(current_state)
        
        # Badge rewards (primary objective)
        prev_badges = self._get_badges(prev_state)
        curr_badges = self._get_badges(current_state)
        if curr_badges > prev_badges:
            reward += 1000.0
            logger.info(f"🏆 Badge obtained! Total badges: {curr_badges}")
        
        # Level up rewards
        prev_levels = self._get_total_party_level(prev_state)
        curr_levels = self._get_total_party_level(current_state)
        if curr_levels > prev_levels:
            reward += 50.0
        
        # Location exploration rewards
        curr_location = self._get_location(current_state)
        if curr_location != self.prev_location:
            if curr_location not in self.visited_locations:
                reward += 20.0
                self.visited_locations.add(curr_location)
            else:
                reward += 5.0
            self.prev_location = curr_location
        
        # Movement rewards (only if not blocked)
        curr_position = self._get_position(current_state)
        if not is_blocked:
            if curr_position != self.prev_position:
                reward += 0.5
                self.stationary_steps = 0
            else:
                self.stationary_steps += 1
                # Penalize being stuck, but less harshly at first
                reward -= 0.05 * min(self.stationary_steps, 20)
        else:
            # In blocked state, reset stationary counter
            self.stationary_steps = 0
        
        self.prev_position = curr_position
        
        # HP penalties (only if not in blocked state to avoid unfair penalties during battles/cutscenes)
        if not is_blocked:
            party = current_state.get('player', {}).get('party', [])
            for pokemon in party:
                current_hp = pokemon.get('current_hp', 0)
                max_hp = pokemon.get('max_hp', 1)
                hp_ratio = current_hp / max(max_hp, 1)
                if hp_ratio < 0.2:
                    reward -= 5.0
                elif hp_ratio < 0.5:
                    reward -= 1.0
        
        return reward
    
    def _is_game_blocked(self, game_state: Dict[str, Any]) -> bool:
        """
        Check if the game is in a state where player input is blocked.
        This includes dialogues, cutscenes, menus, etc.
        """
        game = game_state.get('game_state', {})
        
        # Check for dialogue
        if game.get('in_dialogue', False):
            return True
        
        # Check for battle (partially blocked - different action space)
        if game.get('is_in_battle', False):
            return True
        
        # Check for menu
        if game.get('in_menu', False):
            return True
        
        # Could add more checks:
        # - Animations playing
        # - Cutscenes
        # - Scrolling text
        # For now, these three are the main ones
        
        return False
    
    def _check_terminated(self, game_state: Dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        party = game_state.get('player', {}).get('party', [])
        if party:
            all_fainted = all(p.get('current_hp', 0) == 0 for p in party)
            if all_fainted:
                return True
        
        badges = self._get_badges(game_state)
        if badges >= 8:
            return True
        
        return False
    
    def _get_position(self, game_state: Dict[str, Any]) -> Tuple[int, int]:
        """Get player position as tuple."""
        pos = game_state.get('player', {}).get('position', {})
        return (pos.get('x', 0), pos.get('y', 0))
    
    def _get_location(self, game_state: Dict[str, Any]) -> str:
        """Get current location name."""
        location = game_state.get('player', {}).get('location', '')
        if isinstance(location, dict):
            return location.get('map_name', 'UNKNOWN')
        return str(location) if location else 'UNKNOWN'
    
    def _get_badges(self, game_state: Dict[str, Any]) -> int:
        """Get number of badges."""
        badges = game_state.get('game', {}).get('badges', 0)
        if isinstance(badges, list):
            return sum(1 for b in badges if b)
        return int(badges) if isinstance(badges, int) else 0
    
    def _get_total_party_level(self, game_state: Dict[str, Any]) -> int:
        """Get sum of all party Pokemon levels."""
        party = game_state.get('player', {}).get('party', [])
        return sum(p.get('level', 0) for p in party)
    
    # === Lightweight methods for fast DRL training ===
    
    def _calculate_reward_from_lightweight(
        self,
        prev_state: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward based on lightweight state (faster than full state).
        Simplified reward function that doesn't require expensive state reads.
        """
        reward = 0.0
        
        # Badge rewards (primary objective)
        prev_badges = prev_state.get('badges', 0) if prev_state else 0
        curr_badges = current_state.get('badges', 0)
        if curr_badges > prev_badges:
            reward += 1000.0
            logger.info(f"🏆 Badge obtained! Total badges: {curr_badges}")
        
        # Level up rewards
        prev_party = prev_state.get('party', []) if prev_state else []
        curr_party = current_state.get('party', [])
        prev_levels = sum(p.get('level', 0) for p in prev_party)
        curr_levels = sum(p.get('level', 0) for p in curr_party)
        if curr_levels > prev_levels:
            reward += 50.0
        
        # Movement rewards
        prev_pos = prev_state.get('position', {}) if prev_state else {}
        curr_pos = current_state.get('position', {})
        prev_coords = (prev_pos.get('x', 0), prev_pos.get('y', 0))
        curr_coords = (curr_pos.get('x', 0), curr_pos.get('y', 0))
        
        # Check if in battle (movement doesn't matter in battle)
        in_battle = current_state.get('in_battle', False)
        
        if not in_battle:
            if curr_coords != prev_coords:
                reward += 0.5
                self.stationary_steps = 0
            else:
                self.stationary_steps += 1
                # Penalize being stuck
                reward -= 0.05 * min(self.stationary_steps, 20)
        else:
            # In battle, reset stationary counter
            self.stationary_steps = 0
        
        # HP penalties (only if not in battle)
        if not in_battle:
            for pokemon in curr_party:
                current_hp = pokemon.get('current_hp', 0)
                max_hp = pokemon.get('max_hp', 1)
                hp_ratio = current_hp / max(max_hp, 1)
                if hp_ratio < 0.2:
                    reward -= 5.0
                elif hp_ratio < 0.5:
                    reward -= 1.0
        
        return reward
    
    def _check_terminated_from_lightweight(self, lightweight_state: Dict[str, Any]) -> bool:
        """Check if episode should terminate using lightweight state."""
        # Check if all Pokemon fainted
        party = lightweight_state.get('party', [])
        if party:
            all_fainted = all(p.get('current_hp', 0) == 0 for p in party)
            if all_fainted:
                return True
        
        # Check if got all badges
        badges = lightweight_state.get('badges', 0)
        if badges >= 8:
            return True
        
        return False
    
    # === End lightweight methods ===
    
    def render(self):
        """Render the environment.
        
        Note: For training with visualization, use PygameRenderCallback in train_ppo.py
        instead of calling this method directly. This avoids the PIL.Image.show() 
        issue which spawns external processes and causes system resource exhaustion.
        """
        if self.render_mode == 'human':
            # Don't use screenshot.show() - it spawns external processes
            # Instead, return the screenshot for external rendering (e.g., pygame)
            screenshot = self.emulator.get_screenshot()
            if screenshot:
                return np.array(screenshot)
        elif self.render_mode == 'rgb_array':
            screenshot = self.emulator.get_screenshot()
            if screenshot:
                return np.array(screenshot)
        return None
    
    def close(self):
        """Clean up resources."""
        self.emulator.stop()