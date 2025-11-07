"""
LLM Reward Shaping Callback for Stable Baselines3

This callback periodically analyzes the training state using a local LLM (Ollama)
and adjusts the reward multiplier in each environment to guide learning.
"""

import logging
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

logger = logging.getLogger(__name__)


class LLMRewardCallback(BaseCallback):
    """
    Callback that uses LLM to dynamically adjust reward multipliers based on game state.
    
    The LLM analyzes:
    - Recent reward trends
    - Milestone progress
    - Episode statistics
    - Agent behavior patterns (stuck, wandering, etc.)
    
    Then provides:
    - Reward multiplier (0.5 to 2.0)
    - Strategic advice
    """
    
    def __init__(
        self, 
        check_frequency: int = 1000,  # Check every N steps
        use_llm: bool = False,  # Start with rule-based, then enable LLM
        verbose: int = 1,
        llm_timeout: int = 30  # 🆕 Timeout configurable (default 30s)
    ):
        super().__init__(verbose)
        self.check_frequency = check_frequency
        self.use_llm = use_llm
        self.llm_timeout = llm_timeout
        self.last_check_step = 0
        
        # 🆕 EXPLORACIÓN SIN TRAMPA - solo lugares visitados
        self.visited_maps = {}  # {env_id: set(map_names)}
        self.visited_positions = {}  # {env_id: {map_name: set((x,y))}}
        self.talked_npcs = {}  # {env_id: set(npc_names)}
        
        # 🆕 OBJETIVOS PERSISTENTES CON TRACKING
        self.active_objectives = {}  # {env_id: {'name': str, 'milestone': str, 'step_set': int, 'last_pos': (x,y), 'last_map': str}}
        
        # 🆕 MAPA DE UBICACIONES APRENDIDAS (sin trampa - el agente las descubre)
        # IMPORTANTE: NO se borra en reset - el agente "recuerda" ubicaciones aprendidas
        self.learned_locations_global = {}  # {milestone_name: (map_name, (x, y))} - GLOBAL entre episodes
        self.learned_locations = {}  # {env_id: {milestone_name: (map_name, (x, y))}} - Por episode (temporal)
        
        # 🆕 EXPLORACIÓN VISUAL - qué ha visto el agente
        self.seen_visual_hashes = {}  # {env_id: set(hash(screenshot_region))} - Para detectar nuevas vistas
        
        # 🆕 HISTORIAL DE DIÁLOGOS (el LLM decide cuáles guardar)
        self.dialog_history = {}  # {env_id: [{'text': str, 'step': int, 'useful': bool}]}
        
        # Track statistics for LLM analysis
        self.recent_rewards = []
        self.episode_lengths = []
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        
        # Check if it's time to update
        if self.num_timesteps - self.last_check_step < self.check_frequency:
            return True
        
        self.last_check_step = self.num_timesteps
        
        # Get environments
        envs = self.training_env.envs if hasattr(self.training_env, 'envs') else [self.training_env]
        
        # Log that we're checking rewards
        if self.verbose > 0:
            logger.warning(f"🔍 LLM Callback: Checking {len(envs)} environments at step {self.num_timesteps}")
            
            # 🆕 Mostrar ubicaciones aprendidas (memoria global)
            if self.learned_locations_global and self.num_timesteps % 1000 == 0:
                logger.info(f"🧠 Global memory: {len(self.learned_locations_global)} locations learned: {list(self.learned_locations_global.keys())}")
        
        if self.use_llm:
            # Use LLM for analysis (to implement)
            self._update_with_llm(envs)
        else:
            # Use simple rule-based system
            self._update_with_rules(envs)
        
        return True
    
    def _update_with_rules(self, envs):
        """Simple rule-based reward adjustment (no LLM needed)."""
        
        # Check if we have SubprocVecEnv (need to use env_method)
        num_envs = self.training_env.num_envs if hasattr(self.training_env, 'num_envs') else len(envs)
        
        for i in range(num_envs):
            try:
                # For SubprocVecEnv, use env_method to access attributes remotely
                if hasattr(self.training_env, 'env_method'):
                    # Get stationary_steps from remote environment
                    stationary_steps_list = self.training_env.env_method('_get_stationary_steps', indices=[i])
                    stationary_steps = stationary_steps_list[0] if stationary_steps_list else 0
                    
                    # Get milestone count
                    milestone_count_list = self.training_env.env_method('_get_milestone_count', indices=[i])
                    completed = milestone_count_list[0] if milestone_count_list else 0
                    
                    # Get last milestone count
                    last_count_list = self.training_env.env_method('_get_last_milestone_count', indices=[i])
                    last_milestone_count = last_count_list[0] if last_count_list else 0
                else:
                    # For DummyVecEnv, access directly
                    actual_env = envs[i].unwrapped if hasattr(envs[i], 'unwrapped') else envs[i]
                    stationary_steps = getattr(actual_env, 'stationary_steps', 0)
                    
                    has_milestone_tracker = hasattr(actual_env, 'emulator') and \
                                           hasattr(actual_env.emulator, 'milestone_tracker')
                    completed = len(actual_env.emulator.milestone_tracker.completed_milestones) if has_milestone_tracker else 0
                    last_milestone_count = getattr(actual_env, 'last_milestone_count', 0)
                
                # Apply rules
                if stationary_steps > 100:
                    # Severely stuck - reduce rewards
                    multiplier = 0.3
                    advice = "Agent is stuck! Penalizing rewards to force exploration."
                elif stationary_steps > 50:
                    # Somewhat stuck
                    multiplier = 0.6
                    advice = "Agent showing low movement. Reducing rewards."
                elif completed > last_milestone_count:
                    # New milestone! Boost rewards
                    multiplier = 1.8
                    advice = f"NEW MILESTONE! Boosting rewards. Total: {completed}"
                else:
                    # Normal operation
                    multiplier = 1.0
                    advice = "Normal operation"
                
                # Update environment using env_method for SubprocVecEnv
                if hasattr(self.training_env, 'env_method'):
                    # El método _set_llm_multiplier ahora también actualiza last_milestone_count
                    self.training_env.env_method('_set_llm_multiplier', multiplier, advice, completed, indices=[i])
                else:
                    # For DummyVecEnv, set directly
                    actual_env = envs[i].unwrapped if hasattr(envs[i], 'unwrapped') else envs[i]
                    actual_env.llm_reward_multiplier = multiplier
                    actual_env.llm_advice = advice
                    actual_env.llm_last_update_step = self.num_timesteps
                    # SIEMPRE actualizar last_milestone_count, no solo cuando hay nuevo milestone
                    actual_env.last_milestone_count = completed
                
                # Log con más detalle para debugging
                if self.verbose > 0:
                    if multiplier == 1.8:
                        # Nuevo milestone - boost activo
                        logger.warning(f"🎯 [Env {i}] Step {self.num_timesteps}: {advice} (multiplier={multiplier:.2f}x)")
                    elif multiplier < 1.0:
                        # Penalización por estar atascado
                        logger.warning(f"⚠️ [Env {i}] Step {self.num_timesteps}: {advice} (multiplier={multiplier:.2f}x)")
                    elif stationary_steps > 20:
                        # Log cuando vuelve a normal después de boost
                        logger.info(f"↩️ [Env {i}] Multiplier reset to {multiplier:.2f}x (milestones: {completed}, stationary: {stationary_steps})")
                elif self.verbose > 1:
                    # Verbose mode: show even normal operation (for debugging)
                    logger.info(f"[Env {i}] {advice} (×{multiplier:.2f})")
                    
            except Exception as e:
                logger.error(f"Error updating env {i}: {e}")
    
    def _update_with_llm(self, envs):
        """Use Ollama LLM for intelligent reward shaping (advanced) con lectura de diálogos."""
        try:
            import requests
            
            # Check if we have SubprocVecEnv
            num_envs = self.training_env.num_envs if hasattr(self.training_env, 'num_envs') else len(envs)
            
            for i in range(num_envs):
                try:
                    # Get environment reference
                    actual_env = None  # 🆕 Inicializar para evitar error
                    
                    if hasattr(self.training_env, 'env_method'):
                        # SubprocVecEnv - need to get state remotely
                        stationary_steps_list = self.training_env.env_method('_get_stationary_steps', indices=[i])
                        stationary_steps = stationary_steps_list[0] if stationary_steps_list else 0
                        
                        milestone_count_list = self.training_env.env_method('_get_milestone_count', indices=[i])
                        completed = milestone_count_list[0] if milestone_count_list else 0
                        
                        last_count_list = self.training_env.env_method('_get_last_milestone_count', indices=[i])
                        last_milestone_count = last_count_list[0] if last_count_list else 0
                        
                        # 🆕 LEER DIÁLOGO del juego
                        dialog_list = self.training_env.env_method('_get_current_dialog', indices=[i])
                        dialog_text = dialog_list[0] if dialog_list else ""
                    else:
                        # DummyVecEnv
                        actual_env = envs[i].unwrapped if hasattr(envs[i], 'unwrapped') else envs[i]
                        stationary_steps = getattr(actual_env, 'stationary_steps', 0)
                        
                        has_milestone_tracker = hasattr(actual_env, 'emulator') and \
                                               hasattr(actual_env.emulator, 'milestone_tracker')
                        completed = len(actual_env.emulator.milestone_tracker.completed_milestones) if has_milestone_tracker else 0
                        last_milestone_count = getattr(actual_env, 'last_milestone_count', 0)
                        
                        # 🆕 LEER DIÁLOGO del juego
                        dialog_text = self._read_dialog_from_env(actual_env)
                    
                    # Log solo si hay diálogo
                    if dialog_text:
                        logger.info(f"📜 [Env {i}] Dialog available: '{dialog_text[:60]}...'")
                    
                    # Prepare state summary for LLM (MEJORADO con diálogos + mapa + historial)
                    # Solo pasar actual_env si está disponible (DummyVecEnv)
                    state_summary = self._get_state_summary_with_dialog(
                        stationary_steps, completed, dialog_text, env=actual_env
                    )
                    
                    # Create prompt for LLM (MEJORADO con mapa y contexto)
                    prompt = f"""You are an AI coach for a Pokemon Emerald reinforcement learning agent.

Current State:
{state_summary}

CRITICAL CONTEXT AWARENESS:

📜 DIALOGUE CHAIN (most important):
   - Look at "Recent Dialogue Chain" above
   - Multiple dialogues may form a CONNECTED STORY
   - Example chain:
     1. "Go see PROF. BIRCH outside town"
     2. "Be careful, there are wild POKéMON!"
     3. "LITTLEROOT TOWN - A town that can't be shaded red"
   - All 3 relate to SAME objective: leaving town to find PROF. BIRCH
   - If dialogues are RELATED, keep the SAME objective through the chain
   - Only change objective when dialogue gives CLEARLY DIFFERENT direction

🎯 If there is an ACTIVE OBJECTIVE shown above:
   - This is what the agent is CURRENTLY trying to do
   - Check if current dialogue is PART OF THE SAME CHAIN
   - If yes → KEEP objective (dialogue continues the story)
   - If no (new quest) → REPLACE with new objective

📍 Current LOCATION + Learned Locations:
   - Agent only knows locations it has VISITED before
   - If objective mentions "ROUTE_101" but agent never visited → can't calculate distance
   - Agent must EXPLORE to find new locations first
   - Once found, location is remembered for future episodes

CRITICAL RULES FOR OBJECTIVE DETECTION:

🚫 ABSOLUTE RULE #1: NO DIALOGUE = NO NEW OBJECTIVE
   - If there is NO dialogue text (empty or "❌ No dialogue")
   - You MUST return: objective="null"
   - Agent cannot invent objectives from game state alone
   - ONLY NPCs can give objectives through dialogue

⚠️ IMPORTANT: Most ambient/system messages are already FILTERED before reaching you.
If you receive a message, it's likely important. However, still check:

IGNORE these dialogues (set objective="null", multiplier=1.0):
- Farewells ONLY: "See you", "Goodbye", "Take care", "Come back soon" (and NOTHING else)
- Pure ambient text (rare): "There is a movie on TV", "It's a poster", "It's a bookshelf"
  * Note: Most ambient text is pre-filtered, so if it reaches you, double-check context
- System messages (rare): "Saved", "Loading", "Press START"
  * Note: Most system messages are pre-filtered

ALWAYS RECOGNIZE as OBJECTIVES (extract meaningful goal):
- **ANY NPC dialogue** = potential objective (even simple greetings can hint at location/progress)
- **Direction/Location hints**: "Go to X", "Find X", "X is outside", "Head to X" → objective=X
- **Quest dialogue**: "Bring me X", "Defeat X", "Talk to X", "Have you been to see X?" → objective=action
- **Story progression**: First meeting, receiving items, battles → describe the milestone
- **Current location matching milestone**: If location matches next uncompleted milestone → objective=milestone_name

STRATEGY (CRITICAL - preserve active objectives):
1. Is this EXACTLY a farewell/ambient/system message? → YES: objective="null", multiplier=1.0

2. **IF THERE IS AN ACTIVE OBJECTIVE** (shown above):
   - Does dialogue COMPLETE the objective? → Keep objective name (will be marked complete)
   - Does dialogue relate to objective? → KEEP THE SAME objective name
   - Does dialogue give EXPLICIT new quest/direction? → ONLY THEN replace
   - Generic NPC chat ("Hi!", "Nice weather")? → KEEP active objective, DON'T replace
   
   ⚠️ IMPORTANT: Don't replace active objective with generic "explore_and_talk_to_npcs"
   Only replace if dialogue gives CLEAR NEW DIRECTION like:
   - "Go to [specific place]"
   - "Find [specific person/item]"
   - "Defeat [specific trainer]"

3. **IF NO ACTIVE OBJECTIVE**:
   - Look at NEXT uncompleted milestone (marked ⏳)
   - Does dialogue relate to that milestone?
     * YES, directly → objective=milestone_name
     * YES, preparing for it → objective=milestone_name
     * NO, but NPC gave specific quest → objective=description_of_quest
     * NO, generic chat → objective="explore_and_talk_to_npcs" (ONLY if no other objective)

4. Special: New milestone just completed → multiplier=1.8-2.0

CONCRETE EXAMPLES:

Example 1 - MOM dialogue at start:
- Dialogue: "MOM: See you, honey!"
- Analysis: This is EXACTLY a farewell
- Response: {{"multiplier": 1.0, "reason": "farewell dialogue", "detected_objective": "null"}}

Example 2 - MOM dialogue BEFORE leaving:
- Dialogue: "MOM: Your very own POKéMON legend is about to unfold!"
- Next milestone: ⏳ ROUTE_101
- Analysis: Story progression, hints at going outside
- Response: {{"multiplier": 1.4, "reason": "story progression before ROUTE_101", "detected_objective": "ROUTE_101"}}

Example 3 - NPC in lab:
- Dialogue: "PROF. BIRCH: You should go meet the neighbors!"
- Next milestone: ⏳ LITTLEROOT_TOWN exploration
- Response: {{"multiplier": 1.5, "reason": "NPC directing to explore town", "detected_objective": "explore_LITTLEROOT_TOWN"}}

Example 4 - Location matches milestone:
- Location: ROUTE_101
- Next milestone: ⏳ ROUTE_101
- Dialogue: (none)
- Response: {{"multiplier": 1.6, "reason": "reached next milestone location", "detected_objective": "ROUTE_101"}}

Example 5 - Generic NPC:
- Dialogue: "Trainer: Nice weather today!"
- Analysis: Not farewell, not system, = NPC interaction
- Response: {{"multiplier": 1.1, "reason": "NPC interaction - exploration", "detected_objective": "talk_to_npcs"}}

Example 6 - WITH ACTIVE OBJECTIVE CONTEXT (IMPORTANT):
- ACTIVE OBJECTIVE: 'visit_PROF_BIRCH_house' (set 50 steps ago, started at LITTLEROOT_TOWN)
- Current Location: BIRCH_HOUSE
- Dialogue: "Hi, neighbor! Do you already have a POKéMON?"
- Analysis: Agent REACHED the objective location! Dialogue confirms arrival at Prof Birch's house
- Response: {{"multiplier": 1.7, "reason": "arrived at objective location - Prof Birch house", "detected_objective": "visit_PROF_BIRCH_house"}}

Example 7 - ACTIVE OBJECTIVE COMPLETED:
- ACTIVE OBJECTIVE: 'get_starter_pokemon' (set 200 steps ago)
- Current Location: BIRCH_LAB
- Dialogue: "PROF. BIRCH: Here, take this POKéMON!"
- Analysis: Objective about to be completed!
- Response: {{"multiplier": 1.9, "reason": "receiving starter - objective completed", "detected_objective": "STARTER_CHOSEN"}}

Example 8 - ACTIVE OBJECTIVE BUT NEW DIRECTION:
- ACTIVE OBJECTIVE: 'explore_town' (set 100 steps ago)
- Dialogue: "NPC: Have you been to see PROF. BIRCH? He's looking for you!"
- Analysis: New SPECIFIC direction given - override old objective
- Response: {{"multiplier": 1.4, "reason": "new objective - visit Prof Birch", "detected_objective": "visit_PROF_BIRCH"}}

Example 9 - ACTIVE OBJECTIVE + GENERIC CHAT (CRITICAL):
- ACTIVE OBJECTIVE: 'visit_PROF_BIRCH' (set 200 steps ago)
- Dialogue: "NPC: Um, hi! There are scary POKéMON out there!"
- Analysis: Generic NPC chat, NOT a new quest. KEEP active objective!
- Response: {{"multiplier": 1.0, "reason": "generic NPC chat while pursuing objective", "detected_objective": "visit_PROF_BIRCH"}}
  ⚠️ DO NOT change to "explore_and_talk_to_npcs" - that would lose the main objective!

Example 10 - ACTIVE OBJECTIVE + NO DIALOGUE:
- ACTIVE OBJECTIVE: 'ROUTE_101' (set 300 steps ago)
- Dialogue: (none)
- Analysis: No new information, keep pursuing active objective
- Response: {{"multiplier": 1.0, "reason": "no dialogue detected", "detected_objective": "ROUTE_101"}}

Example 11 - DIALOGUE CHAIN (CRITICAL):
- Recent Dialogue Chain:
  1. "Go see PROF. BIRCH outside town!"
  2. "Be careful, wild POKéMON are dangerous!"
  3. "LITTLEROOT TOWN - A town that can't be shaded red"
- ACTIVE OBJECTIVE: 'find_PROF_BIRCH_outside_town' (set 100 steps ago)
- Current Dialogue: "LITTLEROOT TOWN - A town that..."
- Analysis: All 3 dialogues are PART OF SAME CHAIN (leaving town to find Birch)
  * Dialogue 3 is town sign, relates to leaving town
  * NOT a new objective, continues the same story
- Response: {{"multiplier": 1.0, "reason": "town sign - part of leaving town chain", "detected_objective": "find_PROF_BIRCH_outside_town"}}
  ⚠️ KEEP the objective because dialogue chain continues!

Example 12 - AGENT DOESN'T KNOW LOCATION:
- ACTIVE OBJECTIVE: 'reach_ROUTE_101' (set 50 steps ago)
- Dialogue: "It's dangerous out there!"
- Analysis: Agent has NEVER visited ROUTE_101 before
  * Agent doesn't know WHERE it is
  * Must explore to find it
  * Dialogue confirms danger outside (related to objective)
- Response: {{"multiplier": 1.0, "reason": "exploring to find ROUTE_101", "detected_objective": "reach_ROUTE_101"}}
  ⚠️ KEEP objective even though location unknown - agent needs to explore!

Example 13 - NO DIALOGUE, NO ACTIVE OBJECTIVE (CRITICAL):
- ACTIVE OBJECTIVE: (none)
- Dialogue: (none) or "❌ No dialogue"
- Current Location: LITTLEROOT_TOWN
- Analysis: NO NPC spoke, NO information extracted
  * Cannot invent objectives from game state alone
  * Agent must explore until NPC gives direction
- Response: {{"multiplier": 1.0, "reason": "no dialogue detected", "detected_objective": "null"}}
  ⚠️ NEVER invent objectives like "ROUTE_101" without NPC mentioning it!

CRITICAL RULES:
1. NO DIALOGUE = NO NEW OBJECTIVE (always return "null") - Example 13
2. If ACTIVE OBJECTIVE exists, only replace with SPECIFIC new quest (Example 8)
3. DO NOT replace with generic "explore_and_talk_to_npcs" (Example 9)
4. Use Dialogue Chain to see if current dialogue CONTINUES the story (Example 11)
5. Agent doesn't know locations until visited - keep objective during exploration (Example 12)

Respond with ONLY a JSON object:
{{"multiplier": 1.0, "reason": "short explanation", "detected_objective": "MILESTONE_NAME or description or null"}}

Multiplier guide:
- 1.8-2.0: New milestone completed or at completion point
- 1.5-1.8: Dialogue/location directly matches next milestone OR active objective reached
- 1.3-1.5: Dialogue suggests progress toward milestone
- 1.2-1.3: NPC gave useful info or quest-related
- 1.1: Generic NPC interaction (better than nothing)
- 1.0: ONLY farewell/ambient/system messages
- 0.6-0.8: Stationary > 50 steps
- 0.3-0.5: Stationary > 100 steps
"""
                    
                    # 🆕 DEBUG: Ver el prompt completo que se envía
                    if self.verbose > 0 and (self.num_timesteps % 10000 == 0 or dialog_text):
                        logger.info(f"🤖 [Env {i}] LLM Prompt Summary:\n{state_summary}")
                    
                    # Call Ollama API (local)
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "llama3",
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": 0.3}
                        },
                        timeout=self.llm_timeout  # 🆕 Usar timeout configurable (default 30s)
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        # Parse LLM response with robust error handling
                        import json
                        import re
                        
                        raw_response = result['response']
                        
                        # 🆕 LIMPIEZA: Extraer JSON de respuesta que puede tener texto extra
                        try:
                            # Intentar parsear directo
                            llm_output = json.loads(raw_response)
                        except json.JSONDecodeError:
                            # Si falla, buscar JSON entre el texto
                            logger.debug(f"Raw LLM response: {raw_response[:200]}")
                            json_match = re.search(r'\{[^{}]*"multiplier"[^{}]*\}', raw_response)
                            if json_match:
                                try:
                                    llm_output = json.loads(json_match.group(0))
                                    logger.debug(f"Extracted JSON from text: {llm_output}")
                                except json.JSONDecodeError:
                                    logger.warning(f"Could not parse JSON even after extraction. Using defaults.")
                                    llm_output = {'multiplier': 1.0, 'reason': 'parsing error', 'detected_objective': 'null'}
                            else:
                                logger.warning(f"No JSON found in LLM response: {raw_response[:200]}")
                                llm_output = {'multiplier': 1.0, 'reason': 'no json in response', 'detected_objective': 'null'}
                        
                        # Validar que tenga los campos necesarios
                        if 'multiplier' not in llm_output:
                            logger.warning(f"LLM response missing 'multiplier' field: {llm_output}")
                            llm_output['multiplier'] = 1.0
                        if 'reason' not in llm_output:
                            llm_output['reason'] = 'unknown'
                        if 'detected_objective' not in llm_output:
                            llm_output['detected_objective'] = 'null'
                        
                        # 🆕 DEBUG: Mostrar respuesta completa del LLM cuando hay diálogo
                        if dialog_text and dialog_text.strip():
                            logger.info(f"🔍 [Env {i}] DIALOG: '{dialog_text}'")
                            logger.info(f"🤖 [Env {i}] LLM RESPONSE: {llm_output}")
                        
                        # 🎯 ARQUITECTURA REDISEÑADA:
                        # - LLM NO decide boost directamente
                        # - LLM solo EXTRAE información (objetivo)
                        # - Sistema MIDE progreso y DA boost
                        
                        # Ignorar multiplier del LLM - solo usar para extraer objetivo
                        multiplier = 1.0  # Empezar en neutral
                        
                        # GUARDAR NUEVO OBJETIVO si el LLM lo detectó
                        objective = llm_output.get('detected_objective', None)
                        
                        # 🆕 ACTUALIZAR EXPLORACIÓN (registrar mapa/posición actual)
                        if actual_env:
                            env_id = id(actual_env)
                            self._update_exploration_map(actual_env, env_id)
                            
                            # 🆕 CHEQUEAR PROGRESO DE OBJETIVO ACTIVO (si existe)
                            progress = self._check_objective_progress(actual_env, env_id)
                            
                            # 🆕 DETECCIÓN VISUAL - ¿Vio algo nuevo?
                            new_visual_discovery = self._check_new_visual_discovery(actual_env, env_id)
                            
                            # 🆕 AJUSTAR MULTIPLIER BASADO EN PROGRESO REAL
                            if progress['has_objective']:
                                if progress['milestone_completed']:
                                    # ✅ MILESTONE COMPLETADO → Mega boost y limpiar
                                    multiplier = 2.0
                                    logger.warning(f"🏆 MILESTONE COMPLETED: {progress['objective_name']} → BOOST 2.0x")
                                    self.active_objectives[env_id] = None
                                    
                                elif progress['time_active'] > 10000:
                                    # ⏰ Objetivo muy viejo sin completar → Reducir y limpiar
                                    # AUMENTADO a 10000 steps (antes 5000) para no olvidar tan rápido
                                    multiplier = 1.0
                                    logger.info(f"⏰ Objetivo '{progress['objective_name']}' activo {progress['time_active']} steps sin completar → neutral 1.0x")
                                    self.active_objectives[env_id] = None
                                
                                # 🆕 PRIORIDAD MÁXIMA: ¿Se acerca al objetivo?
                                elif progress.get('moving_toward_goal'):
                                    # ✨ SE ACERCA AL OBJETIVO → ¡Gran boost!
                                    distance = progress.get('distance_to_goal', '?')
                                    multiplier = 1.8
                                    logger.warning(f"✨ ACERCÁNDOSE al objetivo '{progress['objective_name']}'! Distancia: {distance} → BOOST 1.8x")
                                
                                elif progress.get('moving_toward_goal') is False and progress.get('distance_to_goal') is not None:
                                    # ❌ SE ALEJA DEL OBJETIVO → Penalizar
                                    distance = progress.get('distance_to_goal', '?')
                                    multiplier = 0.8
                                    logger.info(f"❌ ALEJÁNDOSE del objetivo '{progress['objective_name']}'. Distancia: {distance} → BOOST 0.8x")
                                    
                                elif progress['changed_map']:
                                    # 🗺️ CAMBIÓ DE MAPA → Exploración, puede ser progreso
                                    multiplier = 1.3
                                    logger.info(f"🗺️ Cambió de mapa trabajando en objetivo '{progress['objective_name']}' → BOOST 1.3x (exploración)")
                                    
                                elif new_visual_discovery:
                                    # 👁️ VIO ALGO NUEVO → Bonus exploración
                                    multiplier = 1.2
                                    logger.info(f"👁️ Nueva vista descubierta mientras busca '{progress['objective_name']}' → BOOST 1.2x (exploración)")
                                    
                                else:
                                    # ⏳ Objetivo activo pero SIN progreso medible → Neutral
                                    # (No conocemos ubicación O no se mueve con dirección clara)
                                    multiplier = 1.0
                                    logger.debug(f"⏳ [Env {i}] Objetivo activo: {progress['objective_name']} ({progress['time_active']} steps) - sin progreso medible → 1.0x")
                            
                            else:
                                # 📉 NO HAY OBJETIVO ACTIVO
                                # Chequear si el LLM acaba de detectar uno NUEVO
                                new_objective = llm_output.get('detected_objective', None)
                                if new_objective and new_objective != 'None' and new_objective != 'null':
                                    # ✨ NUEVO OBJETIVO DETECTADO → Boost por obtener información
                                    multiplier = 1.3
                                    logger.info(f"✨ NUEVO objetivo detectado: '{new_objective}' → BOOST 1.3x")
                                elif new_visual_discovery:
                                    # 👁️ Nueva vista sin objetivo
                                    multiplier = 1.1
                                    logger.info(f"👁️ Nueva vista descubierta (sin objetivo) → BOOST 1.1x (exploración libre)")
                                else:
                                    multiplier = 1.0  # Neutral
                                    logger.debug(f"📉 Sin objetivo, sin nueva vista → 1.0x")

                        
                        # 🆕 GUARDAR NUEVO OBJETIVO CON CONTEXTO COMPLETO
                        objective = llm_output.get('detected_objective', None)
                        if objective and objective != 'None' and objective != 'null':
                            if actual_env:
                                env_id = id(actual_env)
                                
                                # Chequear si ya existe este objetivo
                                existing_obj = self.active_objectives.get(env_id)
                                should_update = True
                                
                                if existing_obj and existing_obj.get('name') == objective:
                                    # MISMO OBJETIVO → NO sobrescribir, solo actualizar posición
                                    should_update = False
                                    logger.debug(f"🔄 [Env {i}] Objetivo '{objective}' ya activo - manteniendo contexto original")
                                
                                elif existing_obj and objective in ['explore_and_talk_to_npcs', 'talk_to_npcs']:
                                    # 🛡️ PROTECCIÓN: NO reemplazar objetivo específico con genérico
                                    existing_name = existing_obj.get('name', '')
                                    if existing_name and existing_name not in ['explore_and_talk_to_npcs', 'talk_to_npcs']:
                                        should_update = False
                                        logger.warning(f"🛡️ [Env {i}] PROTECCIÓN: NO reemplazar objetivo '{existing_name}' con genérico '{objective}' - manteniendo objetivo específico")
                                        # Mantener el objetivo existente
                                        objective = existing_name
                                
                                # Leer posición y mapa actual
                                current_pos = None
                                current_map = None
                                try:
                                    if hasattr(actual_env, 'emulator') and hasattr(actual_env.emulator, 'memory_reader'):
                                        mem = actual_env.emulator.memory_reader
                                        pos_data = mem.read_position()
                                        map_data = mem.read_current_map()
                                        if pos_data and 'x' in pos_data and 'y' in pos_data:
                                            current_pos = (pos_data['x'], pos_data['y'])
                                        if map_data and 'map_name' in map_data:
                                            current_map = map_data['map_name']
                                except Exception as e:
                                    logger.debug(f"Error reading position/map: {e}")
                                
                                if should_update:
                                    # NUEVO OBJETIVO → Guardar con TODO el contexto
                                    self.active_objectives[env_id] = {
                                        'name': objective,
                                        'milestone': objective if objective in ['ROUTE_101', 'STARTER_CHOSEN', 'BIRCH_LAB_VISITED', 'LITTLEROOT_TOWN'] else None,  # Detectar si es milestone
                                        'step_set': self.num_timesteps,
                                        'initial_pos': current_pos,
                                        'initial_map': current_map,
                                        'last_pos': current_pos,
                                        'last_map': current_map
                                    }
                                    logger.warning(f"🎯 [Env {i}] NEW OBJECTIVE SET: '{objective}' at {current_map} {current_pos}")
                                    
                                    # 🆕 APRENDER UBICACIÓN: Si hay diálogo en esta ubicación, probablemente sea relevante
                                    if dialog_text and dialog_text.strip():
                                        self._learn_objective_location(actual_env, env_id, objective)
                                else:
                                    # MISMO OBJETIVO → Solo actualizar última posición Y refrescar timer
                                    if existing_obj:
                                        existing_obj['last_pos'] = current_pos
                                        existing_obj['last_map'] = current_map
                                        
                                        # 🆕 APRENDER/CONFIRMAR UBICACIÓN: Si hay diálogo aquí, esta es la ubicación correcta
                                        if dialog_text and dialog_text.strip():
                                            self._learn_objective_location(actual_env, env_id, objective)
                                        
                                        # 🆕 REFRESCAR TIMER: Si el LLM lo menciona de nuevo, el objetivo sigue siendo válido
                                        # Esto evita que se "olvide" por timeout
                                        time_since_set = self.num_timesteps - existing_obj.get('step_set', 0)
                                        if time_since_set > 2000:  # Solo refrescar si han pasado >2000 steps
                                            old_step = existing_obj.get('step_set', 0)
                                            existing_obj['step_set'] = self.num_timesteps - 1000  # Retroceder 1000 steps
                                            logger.info(f"🔄 [Env {i}] Objetivo '{objective}' RE-CONFIRMADO por LLM → timer refrescado ({time_since_set} steps activo)")
                                        logger.debug(f"🔄 [Env {i}] Objetivo '{objective}' ya activo - manteniendo contexto")
                        
                        # Update environment
                        if hasattr(self.training_env, 'env_method'):
                            self.training_env.env_method('_set_llm_multiplier', multiplier, 
                                                        llm_output['reason'], completed, indices=[i])
                        else:
                            # Solo actualizar si actual_env está disponible (DummyVecEnv)
                            if actual_env is not None:
                                actual_env.llm_reward_multiplier = multiplier
                                actual_env.llm_advice = llm_output['reason']
                                actual_env.llm_last_update_step = self.num_timesteps
                                actual_env.last_milestone_count = completed
                        
                        if self.verbose > 0:
                            objective = llm_output.get('detected_objective', 'None')
                            # Mostrar si no hay diálogo o si el LLM no detectó objetivo
                            if not dialog_text or not dialog_text.strip():
                                dialog_status = "❌ No dialogue"
                            elif objective == 'None' or objective is None or objective == 'null':
                                dialog_status = f"⚠️ Dialogue present but no objective extracted"
                                # 🆕 MOSTRAR RESPUESTA COMPLETA DEL LLM cuando no detecta objetivo
                                logger.warning(f"📋 [Env {i}] Full LLM output: {llm_output}")
                                logger.warning(f"📋 [Env {i}] Dialogue was: '{dialog_text}'")
                            else:
                                dialog_status = f"✅ Dialogue: '{dialog_text[:30]}...'"
                            
                            logger.warning(f"🤖 [Env {i}] LLM: {llm_output['reason']} | {dialog_status} | Objective: {objective} | (multiplier={multiplier:.2f}x)")
                    else:
                        logger.warning(f"Ollama API error: {response.status_code}")
                        # Fallback to rules if LLM fails
                        if actual_env is not None:
                            self._update_with_rules([envs[i]])
                        else:
                            self._update_with_rules(envs)
                        
                except Exception as e:
                    logger.error(f"Error updating env {i} with LLM: {e}")
                    # 🆕 Mostrar más detalles del error para debug
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    
                    # Fallback: usar multiplier por defecto
                    if actual_env is not None:
                        actual_env.llm_reward_multiplier = 1.0
                        actual_env.llm_advice = f"LLM error: {str(e)[:50]}"
                    
        except Exception as e:
            logger.error(f"LLM update failed: {e}. Falling back to rules.")
            self._update_with_rules(envs)
    
    def _read_dialog_from_env(self, env):
        """Leer diálogo del environment usando OCR fallback."""
        try:
            if hasattr(env, 'emulator') and hasattr(env.emulator, 'memory_reader'):
                mem_reader = env.emulator.memory_reader
                
                # Intentar obtener screenshot para OCR
                screenshot = None
                if hasattr(env.emulator, 'get_screenshot'):
                    screenshot = env.emulator.get_screenshot()
                
                # Usar read_dialog_with_ocr_fallback (memoria + OCR)
                dialog = mem_reader.read_dialog_with_ocr_fallback(screenshot)
                
                # 🆕 FILTRAR SOLO MENSAJES DE SISTEMA (no ambiente - dejar que LLM decida)
                if dialog and dialog.strip():
                    dialog_lower = dialog.lower().strip()
                    
                    # Patrones de mensajes de sistema/UI que deben filtrarse
                    # IMPORTANTE: Filtrado MÍNIMO - dejar que el LLM decida sobre ambiente
                    system_message_patterns = [
                        "no item assigned",          # "There is no item assigned to SELECT"
                        "no registered item",        # Variante del mensaje
                        "press start",               # Instrucciones genéricas
                        "press select",              # Instrucciones genéricas
                        "saving",                    # Mensajes de guardado
                        "save completed",            # Mensajes de guardado
                        "now loading",               # Mensajes de carga
                    ]
                    
                    # Solo filtrar mensajes de SISTEMA, NO ambiente
                    # El LLM puede decidir si "TV" o "bookshelf" es relevante o no
                    if any(pattern in dialog_lower for pattern in system_message_patterns):
                        logger.debug(f"Filtered system message: '{dialog[:50]}...'")
                        return ""  # No guardar en historial ni retornar
                
                # 🆕 Guardar en historial si es nuevo texto ÚTIL
                if dialog and dialog.strip():
                    env_id = id(env)
                    if env_id not in self.dialog_history:
                        self.dialog_history[env_id] = []
                    
                    # Solo guardar si es diferente al último
                    if not self.dialog_history[env_id] or self.dialog_history[env_id][-1] != dialog:
                        self.dialog_history[env_id].append(dialog)
                        # Mantener solo últimos 10 diálogos
                        if len(self.dialog_history[env_id]) > 10:
                            self.dialog_history[env_id].pop(0)
                
                return dialog if dialog else ""
        except Exception as e:
            logger.debug(f"Could not read dialog: {e}")
        return ""
    
    def _get_state_summary_with_dialog(self, stationary_steps, completed, dialog_text, env=None) -> str:
        """Create a summary with dialogue context for LLM analysis."""
        
        summary = f"- Current step: {self.num_timesteps}\n"
        summary += f"- Stationary steps: {stationary_steps}\n"
        summary += f"- Milestones completed: {completed}\n"
        
        # 🆕 MOSTRAR MILESTONES CON ESTADO (✅ completados, ⏳ pendientes)
        if env and hasattr(env, 'emulator') and hasattr(env.emulator, 'milestone_tracker'):
            try:
                from pokemon_env.emulator import MilestoneTracker
                tracker = env.emulator.milestone_tracker
                
                # Encontrar último completado
                last_completed_idx = -1
                for idx, milestone_name in enumerate(MilestoneTracker.MILESTONE_ORDER):
                    if tracker.is_completed(milestone_name):
                        last_completed_idx = idx
                
                # Mostrar últimos 2 completados + próximos 5 sin completar
                if last_completed_idx >= 0:
                    start_idx = max(0, last_completed_idx - 1)
                    end_idx = min(len(MilestoneTracker.MILESTONE_ORDER), last_completed_idx + 6)
                else:
                    start_idx = 0
                    end_idx = 5
                
                summary += f"\n🎯 Game Milestones (Recent Progress):\n"
                for milestone_name in MilestoneTracker.MILESTONE_ORDER[start_idx:end_idx]:
                    status = "✅" if tracker.is_completed(milestone_name) else "⏳"
                    summary += f"  {status} {milestone_name}\n"
            except Exception as e:
                logger.debug(f"Could not read milestones: {e}")
        
        # 🆕 AGREGAR INFORMACIÓN DEL MAPA (si disponible)
        if env:
            env_id = id(env)
            
            # 🆕 MOSTRAR OBJETIVO ACTIVO ACTUAL (si existe)
            if env_id in self.active_objectives and self.active_objectives[env_id]:
                active_obj = self.active_objectives[env_id]
                obj_name = active_obj.get('name', 'Unknown')
                steps_active = self.num_timesteps - active_obj.get('step_set', 0)
                initial_map = active_obj.get('initial_map', '?')
                
                summary += f"\n🎯 ACTIVE OBJECTIVE: '{obj_name}'\n"
                summary += f"   - Set {steps_active} steps ago\n"
                summary += f"   - Started at: {initial_map}\n"
                
                # Chequear si conocemos la ubicación del objetivo
                goal_location = self._get_milestone_location(obj_name, env_id)
                if goal_location:
                    goal_map, goal_pos = goal_location
                    summary += f"   - Known location: {goal_map} at {goal_pos}\n"
                else:
                    summary += f"   - Location UNKNOWN - must explore to find it\n"
            
            try:
                # Leer ubicación actual
                if hasattr(env, 'emulator') and hasattr(env.emulator, 'memory_reader'):
                    mem = env.emulator.memory_reader
                    location = mem.read_current_map()
                    position = mem.read_position()
                    
                    if location:
                        summary += f"\n- Location: {location.get('map_name', 'Unknown')}\n"
                    if position:
                        summary += f"- Position: ({position.get('x', '?')}, {position.get('y', '?')})\n"
                    
                    # Ver qué hay alrededor (tiles cercanos)
                    tiles_info = self._get_nearby_tiles_info(env)
                    if tiles_info:
                        summary += f"- Nearby: {tiles_info}\n"
            except Exception as e:
                logger.debug(f"Could not read map info: {e}")
        
        # 🆕 AGREGAR HISTORIAL DE DIÁLOGOS (últimos 5 para cadena de contexto)
        if env:
            env_id = id(env)
            if env_id in self.dialog_history and len(self.dialog_history[env_id]) > 0:
                summary += f"\n📜 Recent Dialogue Chain (last {min(5, len(self.dialog_history[env_id]))} dialogues):\n"
                summary += "   Use this to understand the FULL CONTEXT of the current situation.\n"
                summary += "   If multiple dialogues relate to the same quest, they form a CHAIN.\n"
                for i, hist_dialog in enumerate(self.dialog_history[env_id][-5:], 1):
                    summary += f"  {i}. \"{hist_dialog[:80]}...\"\n"
        
        # 🆕 AGREGAR DIÁLOGO ACTUAL
        if dialog_text and len(dialog_text.strip()) > 0:
            summary += f"\n📜 Current Game Dialogue:\n\"{dialog_text}\"\n"
        else:
            summary += f"\n📜 Current Game Dialogue: (none detected)\n"
        
        # Recent rewards
        if hasattr(self, 'recent_rewards') and len(self.recent_rewards) > 0:
            avg_reward = np.mean(self.recent_rewards[-100:])
            summary += f"- Avg recent reward: {avg_reward:.2f}\n"
        
        return summary
    
    def _get_nearby_tiles_info(self, env) -> str:
        """🆕 Obtener descripción de tiles cercanos al jugador."""
        try:
            if hasattr(env, 'emulator') and hasattr(env.emulator, 'memory_reader'):
                mem = env.emulator.memory_reader
                # Leer mapa 3x3 alrededor del jugador
                map_data = mem.read_map_around_player(radius=1)  # 3x3 grid
                
                if map_data and 'tiles' in map_data:
                    tiles = map_data['tiles']
                    
                    # Contar tipos de tiles
                    grass_count = sum(1 for row in tiles for tile in row if 'grass' in str(tile).lower())
                    water_count = sum(1 for row in tiles for tile in row if 'water' in str(tile).lower())
                    path_count = sum(1 for row in tiles for tile in row if 'path' in str(tile).lower() or 'road' in str(tile).lower())
                    
                    info_parts = []
                    if grass_count > 0:
                        info_parts.append(f"{grass_count} grass")
                    if water_count > 0:
                        info_parts.append(f"{water_count} water")
                    if path_count > 0:
                        info_parts.append(f"{path_count} path")
                    
                    return ", ".join(info_parts) if info_parts else "open area"
        except Exception as e:
            logger.debug(f"Could not read nearby tiles: {e}")
        return ""
    
    def _get_state_summary(self, env) -> str:
        """Create a summary of current state for LLM analysis."""
        
        stationary = getattr(env, 'stationary_steps', 0)
        step = getattr(env, 'current_step', 0)
        
        summary = f"- Current step: {step}\n"
        summary += f"- Stationary steps: {stationary}\n"
        
        # Milestone info
        if hasattr(env, 'emulator') and hasattr(env.emulator, 'milestone_tracker'):
            tracker = env.emulator.milestone_tracker
            completed = len(tracker.completed_milestones)
            total = len(tracker.milestones)
            summary += f"- Milestones: {completed}/{total}\n"
            if tracker.completed_milestones:
                summary += f"- Last milestone: {list(tracker.completed_milestones)[-1]}\n"
        
        # Recent rewards
        if hasattr(self, 'recent_rewards') and len(self.recent_rewards) > 0:
            avg_reward = np.mean(self.recent_rewards[-100:])
            summary += f"- Avg recent reward: {avg_reward:.2f}\n"
        
        return summary

    def _update_exploration_map(self, env, env_id):
        """🆕 Actualizar mapa de exploración (solo lugares visitados - sin trampa)."""
        try:
            if not hasattr(env, 'emulator') or not hasattr(env.emulator, 'memory_reader'):
                return
            
            mem = env.emulator.memory_reader
            location = mem.read_current_map()
            position = mem.read_position()
            
            # Inicializar estructuras si no existen
            if env_id not in self.visited_maps:
                self.visited_maps[env_id] = set()
                self.visited_positions[env_id] = {}
                self.talked_npcs[env_id] = set()
                self.seen_visual_hashes[env_id] = set()  # 🆕 Inicializar tracking visual
            
            # Registrar mapa visitado
            if location and 'map_name' in location:
                map_name = location['map_name']
                self.visited_maps[env_id].add(map_name)
                
                # Registrar posición visitada en este mapa
                if map_name not in self.visited_positions[env_id]:
                    self.visited_positions[env_id][map_name] = set()
                
                if position and 'x' in position and 'y' in position:
                    pos_tuple = (position['x'], position['y'])
                    self.visited_positions[env_id][map_name].add(pos_tuple)
            
        except Exception as e:
            logger.debug(f"Error updating exploration map: {e}")
    
    def _check_new_visual_discovery(self, env, env_id) -> bool:
        """🆕 Detectar si la vista actual es NUEVA (exploración visual).
        
        Usa hash simple del screenshot para detectar vistas nunca antes vistas.
        Esto recompensa: ver nuevas casas, NPCs, edificios, rutas, etc.
        
        Returns:
            True si es una vista completamente nueva
        """
        try:
            if not hasattr(env, 'emulator') or not hasattr(env.emulator, 'get_screenshot'):
                return False
            
            # Obtener screenshot
            screenshot = env.emulator.get_screenshot()
            if screenshot is None or not hasattr(screenshot, 'tobytes'):
                return False
            
            # Hash simple del screenshot (centro de la imagen, donde está el personaje)
            # Tomamos solo el centro para ignorar UI/bordes
            import hashlib
            h, w = screenshot.shape[:2] if len(screenshot.shape) > 1 else (0, 0)
            if h == 0 or w == 0:
                return False
            
            # Centro de la imagen (área visible alrededor del jugador)
            center_region = screenshot[h//4:3*h//4, w//4:3*w//4]
            
            # Reducir resolución para hash más robusto (ignorar cambios pequeños)
            try:
                import cv2
                small = cv2.resize(center_region, (32, 32))
                visual_hash = hashlib.md5(small.tobytes()).hexdigest()[:16]  # Hash corto
            except:
                visual_hash = hashlib.md5(center_region.tobytes()).hexdigest()[:16]
            
            # Inicializar si no existe
            if env_id not in self.seen_visual_hashes:
                self.seen_visual_hashes[env_id] = set()
            
            # Chequear si es nueva
            is_new = visual_hash not in self.seen_visual_hashes[env_id]
            
            if is_new:
                self.seen_visual_hashes[env_id].add(visual_hash)
                logger.info(f"👁️ NUEVA VISTA descubierta! (hash: {visual_hash}) - Total vistas únicas: {len(self.seen_visual_hashes[env_id])}")
            
            return is_new
            
        except Exception as e:
            logger.debug(f"Error checking visual discovery: {e}")
            return False
    
    def _get_exploration_summary(self, env_id) -> str:
        """�� Obtener resumen de exploración (solo lo visitado)."""
        if env_id not in self.visited_maps:
            return "No exploration data yet"
        
        maps_visited = len(self.visited_maps[env_id])
        total_positions = sum(len(positions) for positions in self.visited_positions[env_id].values())
        
        # Últimos 3 mapas visitados
        recent_maps = list(self.visited_maps[env_id])[-3:]
        
        summary = f"Maps explored: {maps_visited} | Positions visited: {total_positions}\n"
        if recent_maps:
            summary += f"Recent areas: {', '.join(recent_maps)}"
        
        return summary
    
    def _check_objective_progress(self, env, env_id) -> dict:
        """🆕 Chequear progreso hacia objetivo activo.
        
        Retorna información sobre el progreso del objetivo activo:
        - Si el milestone asociado fue completado
        - Tiempo que el objetivo ha estado activo
        - Si el agente se está moviendo (cambió mapa o posición)
        - Si se está ACERCANDO al objetivo (dirección correcta)
        """
        result = {
            'has_objective': False,
            'objective_name': None,
            'milestone_completed': False,
            'time_active': 0,
            'is_moving': False,
            'changed_map': False,
            'moving_toward_goal': False,  # 🆕 Nuevo: se acerca al objetivo
            'distance_to_goal': None       # 🆕 Distancia actual al objetivo
        }
        
        if env_id not in self.active_objectives or not self.active_objectives[env_id]:
            return result
        
        objective = self.active_objectives[env_id]
        result['has_objective'] = True
        result['objective_name'] = objective.get('name')
        result['time_active'] = self.num_timesteps - objective.get('step_set', 0)
        
        # Chequear si el milestone asociado fue completado
        milestone_name = objective.get('milestone')
        if milestone_name and hasattr(env, 'emulator') and hasattr(env.emulator, 'milestone_tracker'):
            tracker = env.emulator.milestone_tracker
            result['milestone_completed'] = tracker.is_completed(milestone_name)
        
        # 🆕 CHEQUEAR SI EL AGENTE SE ESTÁ MOVIENDO Y ACERCANDO
        try:
            if hasattr(env, 'emulator') and hasattr(env.emulator, 'memory_reader'):
                mem = env.emulator.memory_reader
                current_pos_data = mem.read_position()
                current_map_data = mem.read_current_map()
                
                current_pos = None
                current_map = None
                
                if current_pos_data and 'x' in current_pos_data and 'y' in current_pos_data:
                    current_pos = (current_pos_data['x'], current_pos_data['y'])
                    last_pos = objective.get('last_pos')
                    
                    # Comparar posición
                    if last_pos and current_pos != last_pos:
                        result['is_moving'] = True
                        # Actualizar última posición conocida
                        objective['last_pos'] = current_pos
                
                if current_map_data and 'map_name' in current_map_data:
                    current_map = current_map_data['map_name']
                    last_map = objective.get('last_map')
                    
                    # Comparar mapa
                    if last_map and current_map != last_map:
                        result['changed_map'] = True
                        result['is_moving'] = True  # Cambiar de mapa implica movimiento
                        # Actualizar último mapa conocido
                        objective['last_map'] = current_map
                        logger.info(f"🗺️ Map changed: {last_map} → {current_map}")
                
                # � APRENDER ubicaciones de milestones que se completan
                self._learn_milestone_location(env, env_id)
                
                # �🆕 CALCULAR SI SE ACERCA AL OBJETIVO (solo si ya conocemos la ubicación)
                # Primero intentar con milestone, si no existe usar el nombre del objetivo directamente
                objective_name_to_find = milestone_name if milestone_name else result['objective_name']
                goal_location = self._get_milestone_location(objective_name_to_find, env_id)
                
                if goal_location and current_map and current_pos:
                    goal_map, goal_pos = goal_location
                    
                    # Si estamos en el mapa correcto
                    if current_map == goal_map:
                        # Calcular distancia Manhattan
                        current_distance = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
                        result['distance_to_goal'] = current_distance
                        
                        # Comparar con distancia anterior
                        last_distance = objective.get('last_distance_to_goal')
                        if last_distance is not None:
                            if current_distance < last_distance:
                                result['moving_toward_goal'] = True
                                logger.debug(f"🎯 Moving toward goal! Distance: {last_distance} → {current_distance}")
                            elif current_distance > last_distance:
                                result['moving_toward_goal'] = False
                                logger.debug(f"⬅️ Moving AWAY from goal. Distance: {last_distance} → {current_distance}")
                        
                        # Actualizar distancia para próxima comparación
                        objective['last_distance_to_goal'] = current_distance
                    
                    elif current_map != goal_map:
                        # Si cambió al mapa correcto → ¡progreso!
                        if result['changed_map'] and current_map == goal_map:
                            result['moving_toward_goal'] = True
                            logger.info(f"✅ Reached goal map: {current_map}")
                else:
                    # 🔍 NO CONOCEMOS LA UBICACIÓN → Modo exploración
                    # El agente debe explorar para encontrarla
                    if milestone_name:
                        logger.debug(f"🔍 Exploring to find '{milestone_name}' - location not yet learned")
        
        except Exception as e:
            logger.debug(f"Error checking movement: {e}")
        
        return result
    
    def _get_milestone_location(self, milestone_name: str, env_id=None) -> Optional[tuple]:
        """🆕 Obtener ubicación objetivo de un milestone O objetivo arbitrario.
        
        IMPORTANTE: No usa coordenadas hardcodeadas (trampa).
        Solo retorna ubicaciones que el agente YA visitó y aprendió.
        
        PERSISTENCIA: Usa learned_locations_global que NO se borra en resets.
        El agente "recuerda" ubicaciones entre episodes.
        
        Retorna: (map_name, (x, y)) o None si aún no conoce esa ubicación
        
        AHORA también busca objetivos arbitrarios como "visit_PROF_BIRCH_house"
        """
        if not milestone_name:
            return None
        
        # 🎯 REGLA ANTI-TRAMPA: Solo usar ubicaciones que el agente YA visitó
        # PRIMERO: Buscar en memoria GLOBAL (persiste entre episodes)
        if milestone_name in self.learned_locations_global:
            location = self.learned_locations_global[milestone_name]
            logger.debug(f"📍 Using learned location for {milestone_name}: {location} (from global memory)")
            return location
        
        # SEGUNDO: Buscar en memoria del episode actual
        if env_id and env_id in self.learned_locations and milestone_name in self.learned_locations[env_id]:
            location = self.learned_locations[env_id][milestone_name]
            logger.debug(f"📍 Using learned location for {milestone_name}: {location} (from current episode)")
            return location
        
        # TERCERO: Buscar por NOMBRE PARCIAL (ej: "BIRCH" en "visit_PROF_BIRCH_house")
        # Útil para conectar objetivos con ubicaciones aprendidas
        for learned_name, location in self.learned_locations_global.items():
            # Si el milestone_name contiene parte del learned_name o viceversa
            if learned_name.lower() in milestone_name.lower() or milestone_name.lower() in learned_name.lower():
                logger.debug(f"📍 Found partial match: {milestone_name} ≈ {learned_name} → {location}")
                return location
        
        # Si no conoce la ubicación → modo exploración
        logger.debug(f"❓ Location for {milestone_name} not yet learned - exploration mode")
        return None
    
    def _learn_objective_location(self, env, env_id, objective_name: str):
        """🆕 Aprender ubicación de un objetivo cuando el agente lo menciona/visita.
        
        Cuando el LLM extrae un objetivo y hay diálogo en esa ubicación,
        asumimos que ESA es la ubicación correcta para ese objetivo.
        
        Ejemplo:
        - Objetivo: "visit_PROF_BIRCH_house"
        - Diálogo: "Hi, neighbor!" en BIRCH_HOUSE
        - → Guardar: "visit_PROF_BIRCH_house" = (BIRCH_HOUSE, (x, y))
        """
        try:
            if not hasattr(env, 'emulator') or not hasattr(env.emulator, 'memory_reader'):
                return
            
            mem = env.emulator.memory_reader
            current_map_data = mem.read_current_map()
            current_pos_data = mem.read_position()
            
            if not current_map_data or not current_pos_data:
                return
            
            current_map = current_map_data.get('map_name')
            current_pos = (current_pos_data.get('x'), current_pos_data.get('y'))
            
            if not current_map or None in current_pos:
                return
            
            # Inicializar si no existe
            if env_id not in self.learned_locations:
                self.learned_locations[env_id] = {}
            
            # Guardar ubicación del objetivo
            if objective_name not in self.learned_locations[env_id]:
                self.learned_locations[env_id][objective_name] = (current_map, current_pos)
                logger.info(f"📍 LEARNED objective location: '{objective_name}' = {current_map} at {current_pos}")
            
            # También guardar en global (persiste entre episodes)
            if objective_name not in self.learned_locations_global:
                self.learned_locations_global[objective_name] = (current_map, current_pos)
                logger.info(f"🎓 LEARNED objective (GLOBAL): '{objective_name}' = {current_map} at {current_pos}")
        
        except Exception as e:
            logger.debug(f"Error learning objective location: {e}")
    
    def _learn_milestone_location(self, env, env_id):
        """🆕 Aprender ubicaciones de milestones cuando el agente los visita.
        
        Cada vez que se completa un milestone, guardamos dónde estaba el agente.
        IMPORTANTE: Guarda en learned_locations_global (persiste entre episodes)
        y en learned_locations[env_id] (para el episode actual).
        
        Esto permite que el agente "recuerde" lugares importantes sin hardcodear.
        """
        try:
            if not hasattr(env, 'emulator') or not hasattr(env.emulator, 'milestone_tracker'):
                return
            
            tracker = env.emulator.milestone_tracker
            mem = env.emulator.memory_reader
            
            # Leer ubicación actual
            current_map_data = mem.read_current_map()
            current_pos_data = mem.read_position()
            
            if not current_map_data or not current_pos_data:
                return
            
            current_map = current_map_data.get('map_name')
            current_pos = (current_pos_data.get('x'), current_pos_data.get('y'))
            
            if not current_map or None in current_pos:
                return
            
            # Inicializar diccionario del episode si no existe
            if env_id not in self.learned_locations:
                self.learned_locations[env_id] = {}
            
            # Chequear cada milestone completado
            from pokemon_env.emulator import MilestoneTracker
            for milestone_name in MilestoneTracker.MILESTONE_ORDER:
                # Si el milestone está completo pero NO lo hemos aprendido aún
                if tracker.is_completed(milestone_name):
                    # Guardar en memoria del EPISODE
                    if milestone_name not in self.learned_locations[env_id]:
                        self.learned_locations[env_id][milestone_name] = (current_map, current_pos)
                    
                    # 🎓 APRENDER GLOBAL: Guardar para TODOS los episodes futuros
                    if milestone_name not in self.learned_locations_global:
                        self.learned_locations_global[milestone_name] = (current_map, current_pos)
                        logger.info(f"🎓 LEARNED location for '{milestone_name}': {current_map} at {current_pos} (GLOBAL - persists across resets)")
            
            # 🆕 CASO ESPECIAL: Aprender ubicación de mapa actual si coincide con nombre de milestone
            # Ejemplo: Si estás en "ROUTE_101" y existe milestone "ROUTE_101"
            for milestone_name in MilestoneTracker.MILESTONE_ORDER:
                if current_map and milestone_name in current_map:
                    # Guardar en episode
                    if milestone_name not in self.learned_locations[env_id]:
                        self.learned_locations[env_id][milestone_name] = (current_map, current_pos)
                    
                    # Guardar en global
                    if milestone_name not in self.learned_locations_global:
                        self.learned_locations_global[milestone_name] = (current_map, current_pos)
                        logger.info(f"🗺️ LEARNED map location for '{milestone_name}': {current_map} at {current_pos} (GLOBAL)")
        
        except Exception as e:
            logger.debug(f"Error learning milestone location: {e}")
