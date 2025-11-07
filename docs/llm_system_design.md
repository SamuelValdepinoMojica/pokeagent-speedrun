# Sistema LLM para Reward Shaping - Diseño Arquitectónico

## Problema Original

El usuario identificó varios problemas críticos con el sistema LLM:

### 1. **Evaluación sin LLM**
> "cuando quiera hacer la evaluacion con mi agente de DRL con LLM este ya no funcionaria el LLM no?"

**Respuesta**: ¡Correcto! El LLM es solo para **TRAINING**, no para evaluación/competencia.

**Flujo Correcto**:
```
TRAINING (con LLM):
  DRL Agent → Acción → Environment
      ↓
  Reward Base
      ↓
  LLM analiza estado → Multiplica reward (0.3x - 2.0x)
      ↓
  Reward Shaped → PPO aprende mejor política

EVALUACIÓN (sin LLM):
  DRL Agent → Acción → Environment
      ↓
  Solo Reward Base
      ↓  
  El agente usa lo que APRENDIÓ durante training
```

**El LLM es un "profesor"** durante entrenamiento. El agente debe **generalizar** lo aprendido para funcionar sin él.

### 2. **Tracking de Progreso Inexistente**
> "por ejemplo leyo uno del lab del profesor pero no lo toma en considerancion? como sabe si se acerca o no?"

**Problema Actual**:
```python
Step 200: LLM ve "Go find PROF BIRCH" → multiplier=1.6
Step 400: LLM chequea de nuevo → diálogo expiró (500 steps) → multiplier=1.0
         ❌ Aunque el agente esté yendo hacia el objetivo, pierde el boost
```

**Solución Implementada**: Sistema de objetivos persistentes

```python
# Cuando LLM detecta objetivo:
self.active_objectives[env_id] = {
    'name': "ROUTE_101",               # Objetivo detectado
    'milestone': "ROUTE_101",          # Milestone asociado (del emulator)
    'step_set': 200,                   # Cuándo se detectó
    'last_pos': (5, 10),               # Posición cuando se detectó
    'last_map': "LITTLEROOT_TOWN"      # Mapa cuando se detectó
}

# En cada check del LLM (cada 200 steps):
progress = self._check_objective_progress(env, env_id)

if progress['milestone_completed']:
    # ✅ MILESTONE COMPLETADO → Mega boost y limpiar objetivo
    multiplier = 2.0
    self.active_objectives[env_id] = None
    
elif progress['has_objective'] and progress['time_active'] < 5000:
    # ⏳ OBJETIVO ACTIVO y no ha pasado mucho tiempo → Mantener boost
    multiplier = 1.5
    
elif progress['time_active'] > 5000:
    # ⏰ Mucho tiempo sin completar → Reducir boost gradualmente
    multiplier = 1.0
    self.active_objectives[env_id] = None  # Limpiar objetivo obsoleto
```

### 3. **Exploración sin Trampa**
> "no podriamos guardarlos lugares donde ha estado con la posicion sin hacer trampa osea solo los que hayan visitado?"

**¡Brillante idea!** Implementado sistema de "fog of war":

```python
# Solo registra lo que el agente HA VISITADO
self.visited_maps[env_id] = set()           # {map_name1, map_name2, ...}
self.visited_positions[env_id] = {          # {map_name: {(x,y), (x2,y2), ...}}
    "LITTLEROOT_TOWN": {(5,10), (6,10), (7,10)},
    "ROUTE_101": {(0,5), (1,5)}
}
self.talked_npcs[env_id] = set()            # {"MOM", "PROF_BIRCH"}

# El LLM solo puede ver esta información
exploration_summary = self._get_exploration_summary(env_id)
# → "Maps explored: 2 | Positions visited: 5"
# → "Recent areas: LITTLEROOT_TOWN, ROUTE_101"
```

**NO revela**:
- ❌ Mapas no visitados
- ❌ Posiciones no exploradas
- ❌ NPCs no encontrados

### 4. **LLM Decide Qué es Útil (sin bias)**
> "a lo mejor que el LLM decidad si los dialogos son utiles o no? osea evitar de meterle bias"

**¡Excelente punto!** En lugar de filtrar con reglas hardcodeadas:

**Antes (con bias)**:
```python
# Reglas hardcodeadas
if "See you" in dialog or "Goodbye" in dialog:
    return ""  # Filtrar despedidas
if "There is a movie on TV" in dialog:
    return ""  # Filtrar ambiente
```

**Ahora (LLM decide)**:
```python
# El LLM clasifica cada diálogo
prompt = """
Classify this dialogue:
1. Is it useful for progressing in the game? (yes/no)
2. Does it relate to a milestone or quest? (milestone_name or null)
3. Type: farewell, ambient, quest, story, system

Dialogue: "{dialog_text}"

Respond JSON: {{"useful": bool, "milestone": str, "type": str, "reason": str}}
"""

# Guardar diálogos con clasificación del LLM
self.dialog_history[env_id].append({
    'text': dialog,
    'step': current_step,
    'useful': llm_response['useful'],      # LLM decide si es útil
    'type': llm_response['type'],          # LLM clasifica tipo
    'milestone': llm_response['milestone']  # LLM identifica milestone
})
```

**Ventajas**:
- ✅ Sin reglas hardcodeadas
- ✅ LLM aprende patrones por contexto
- ✅ Más flexible a diferentes juegos/situaciones

## Arquitectura Completa

### Estado Persistente por Environment

```python
class LLMRewardCallback:
    def __init__(self):
        # EXPLORACIÓN (solo visitado - sin trampa)
        self.visited_maps = {}        # {env_id: set(map_names)}
        self.visited_positions = {}   # {env_id: {map: set((x,y))}}
        self.talked_npcs = {}         # {env_id: set(npc_names)}
        
        # OBJETIVOS (persistentes con tracking)
        self.active_objectives = {}   # {env_id: dict}
        
        # DIÁLOGOS (LLM clasifica utilidad)
        self.dialog_history = {}      # {env_id: [dict]}
```

### Flujo en Training

```
Step 0-199:
  - Agent explora
  - Se registran mapas/posiciones visitadas
  - Dialogs se cachean

Step 200 (LLM Check):
  1. Actualizar exploración: _update_exploration_map()
  2. Leer diálogo del caché
  3. LLM analiza:
     - Estado actual (milestones, posición, mapa)
     - Exploración (solo lo visitado)
     - Objetivo activo (si existe)
     - Diálogo nuevo
  4. LLM responde:
     - ¿Diálogo útil? → Guardar en historial
     - ¿Nuevo objetivo? → Persistir con milestone
     - Multiplier (0.3-2.0x)
  5. Aplicar multiplier a rewards

Step 201-399:
  - Rewards se multiplican por último valor del LLM
  - Objetivo sigue activo
  - Si milestone se completa → reward boost automático

Step 400 (LLM Check):
  1. Chequear progreso de objetivo activo:
     - ¿Milestone completado? → 2.0x, limpiar objetivo
     - ¿Aún activo? → mantener boost (1.5x)
     - ¿Muy viejo (>5000 steps)? → reducir (1.0x)
  2. Nuevo análisis de estado...
```

### Flujo en Evaluation

```
Step N:
  - Agent → Acción (usa política aprendida)
  - Environment → Reward base
  - NO LLM callback
  - Agent generaliza lo aprendido:
    * "Aprendí que explorar nuevos mapas da reward"
    * "Aprendí que hablar con NPCs da reward"
    * "Aprendí que avanzar hacia objetivos da reward"
```

## Beneficios del Sistema

### 1. **Objetivos Persistentes**
- ✅ No se pierden cuando el diálogo expira del caché
- ✅ Se mantienen hasta completar el milestone asociado
- ✅ Se limpian automáticamente cuando se completan

### 2. **Tracking de Progreso**
- ✅ Mide si el milestone asociado fue completado
- ✅ Tiempo que lleva el objetivo activo
- ✅ (Futuro) Distancia geométrica al objetivo

### 3. **Exploración Sin Trampa**
- ✅ Solo registra lugares visitados
- ✅ LLM no conoce el mapa completo
- ✅ Simula "fog of war" real

### 4. **LLM Decide Utilidad**
- ✅ Sin bias de reglas hardcodeadas
- ✅ LLM aprende por contexto
- ✅ Más flexible y generalizable

### 5. **Separación Training/Eval**
- ✅ LLM solo en training (profesor)
- ✅ Agent generaliza para eval (examen)
- ✅ No depende del LLM en producción

## Próximos Pasos

### Mejoras Pendientes

1. **Distancia Geométrica a Objetivos**
   - Medir si agente se acerca/aleja del objetivo
   - Ajustar multiplier según progreso espacial

2. **Clasificación Automática de Diálogos**
   - LLM clasifica cada diálogo como útil/no-útil
   - Guardar solo diálogos útiles para historial

3. **Decay de Objetivos Antiguos**
   - Reducir boost gradualmente si no hay progreso
   - Limpiar objetivos obsoletos automáticamente

4. **Tracking de NPCs Interactuados**
   - Registrar qué NPCs has hablado
   - Evitar diálogos repetidos innecesarios

## Conclusión

El sistema ahora:
- ✅ Funciona solo con LLM en training, no en eval
- ✅ Mantiene objetivos persistentes con tracking
- ✅ Registra exploración sin trampa (fog of war)
- ✅ LLM decide qué diálogos son útiles (sin bias)
- ✅ Mide progreso hacia milestones
- ✅ Limpia objetivos completados automáticamente

**El agente DRL aprende mejor con el "profesor" LLM, pero puede funcionar solo después.**
