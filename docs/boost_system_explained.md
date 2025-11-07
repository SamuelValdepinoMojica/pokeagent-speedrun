# Sistema de Boost - Cómo Funciona

## Pregunta Original
> "este boost como se da? el LLM lo decide no? es lo pone solo hay milestone activo? o este lo pone dependiendo si esta cerca de algunos lugares que visito y que le sirven o si va siguiendo la posición o por ejemplo en donde esta el estado anterior y lo compara con el actual?"

## Respuesta: **Ambos - Sistema Híbrido**

El boost se decide en **múltiples capas**:

### 1. **LLM Sugiere Boost Inicial** (basado en diálogo/contexto)

Cuando el LLM ve un diálogo nuevo:

```python
# El LLM analiza el estado y sugiere multiplier
Diálogo: "Go find PROF. BIRCH on ROUTE 101!"
Milestones: ✅ LITTLEROOT_TOWN, ⏳ ROUTE_101

LLM Response:
{
  "multiplier": 1.6,  # ← LLM sugiere basado en contexto
  "reason": "NPC directing to next milestone",
  "detected_objective": "ROUTE_101"
}
```

**El LLM decide basándose en**:
- Si el diálogo menciona un milestone próximo
- Si parece quest importante vs. texto ambiente
- Contexto del mapa actual vs. milestones pendientes

### 2. **Sistema Ajusta Basado en Progreso Real** (trackea posición/mapa)

El sistema luego **SOBRESCRIBE** el boost del LLM basándose en **progreso real**:

```python
# DESPUÉS de que el LLM sugiere 1.6x, el sistema chequea:

progress = _check_objective_progress(env)

if milestone_completed:
    multiplier = 2.0  # ✅ SOBRESCRIBE - Milestone logrado
    
elif changed_map:
    multiplier = max(1.6, 1.6)  # 🗺️ Mantiene/aumenta - Cambió de mapa
    
elif is_moving:
    multiplier = max(1.6, 1.4)  # 🚶 Mantiene - Se está moviendo
    
elif time_active > 5000:
    multiplier = 1.0  # ⏰ SOBRESCRIBE - Objetivo obsoleto
```

### 3. **Comparación Estado Anterior vs Actual**

El sistema compara **cada check** (cada 200 steps):

```python
# Cuando se detecta objetivo (Step 200):
objective = {
    'name': "ROUTE_101",
    'initial_pos': (5, 10),
    'initial_map': "LITTLEROOT_TOWN",
    'last_pos': (5, 10),      # ← Última posición conocida
    'last_map': "LITTLEROOT_TOWN"  # ← Último mapa conocido
}

# Siguiente check (Step 400):
current_pos = (8, 15)  # Leyó de memoria del juego
current_map = "LITTLEROOT_TOWN"

if current_pos != last_pos:  # (8,15) != (5,10)
    is_moving = True  # ✅ Se movió!
    objective['last_pos'] = (8, 15)  # Actualiza para próximo check

# Siguiente check (Step 600):
current_map = "ROUTE_101"  # ¡Cambió de mapa!

if current_map != last_map:  # "ROUTE_101" != "LITTLEROOT_TOWN"
    changed_map = True  # ✅ Progresó!
    multiplier = 1.6  # Boost alto por cambiar de mapa
```

## Flujo Completo - Ejemplo Real

### Step 0-199: Explorando sin objetivo
```
Agent explora LITTLEROOT_TOWN
Multiplier: 1.0 (neutral)
```

### Step 200: LLM Check + Detección de Objetivo
```
1. Agent lee diálogo: "Go find PROF. BIRCH on ROUTE 101!"
2. LLM analiza:
   - Diálogo menciona ROUTE_101
   - Próximo milestone sin completar: ROUTE_101
   - Sugiere: multiplier = 1.6
   
3. Sistema guarda objetivo:
   objective = {
       'name': "ROUTE_101",
       'milestone': "ROUTE_101",
       'step_set': 200,
       'initial_pos': (5, 10),
       'initial_map': "LITTLEROOT_TOWN",
       'last_pos': (5, 10),
       'last_map': "LITTLEROOT_TOWN"
   }
   
4. No hay progreso previo → mantiene multiplier = 1.6
5. Rewards: base_reward × 1.6 = shaped_reward
```

### Step 201-399: Moviéndose con objetivo activo
```
Agent se mueve por LITTLEROOT_TOWN
Multiplier: 1.6 (se mantiene - objetivo activo)
```

### Step 400: LLM Check + Medición de Progreso
```
1. LLM re-analiza estado
   - No hay diálogo nuevo
   - Sugiere: multiplier = 1.0 (sin diálogo)
   
2. Sistema chequea objetivo activo:
   progress = _check_objective_progress()
   
   current_pos = (8, 15)  # Leyó de memoria
   last_pos = (5, 10)     # Del objetivo guardado
   
   if (8,15) != (5,10):
       is_moving = True ✅
       
3. Sistema SOBRESCRIBE multiplier:
   multiplier = max(1.0, 1.4) = 1.4  # Se está moviendo
   
4. Actualiza objetivo:
   objective['last_pos'] = (8, 15)
   
5. Rewards: base_reward × 1.4 = shaped_reward
```

### Step 600: Cambio de Mapa
```
1. LLM analiza (sin diálogo nuevo)
   - Sugiere: 1.0
   
2. Sistema chequea:
   current_map = "ROUTE_101"
   last_map = "LITTLEROOT_TOWN"
   
   if "ROUTE_101" != "LITTLEROOT_TOWN":
       changed_map = True ✅
       
3. Sistema SOBRESCRIBE:
   multiplier = max(1.0, 1.6) = 1.6  # Cambió de mapa!
   
4. Rewards: base_reward × 1.6
```

### Step 800: Milestone Completado
```
1. Sistema chequea milestone tracker:
   tracker.is_completed("ROUTE_101") → True ✅
   
2. Sistema SOBRESCRIBE:
   multiplier = 2.0  # 🏆 MEGA BOOST
   
3. Limpia objetivo:
   active_objectives[env_id] = None
   
4. Rewards: base_reward × 2.0  # Recompensa máxima!
```

### Step 1000+: Sin objetivo activo
```
Multiplier vuelve a 1.0 (neutral)
Esperando próximo diálogo para nuevo objetivo
```

## Ventajas del Sistema

### 1. **LLM para Contexto Semántico**
- ✅ Entiende diálogos complejos
- ✅ Relaciona texto con milestones
- ✅ Detecta objetivos importantes

### 2. **Sistema para Medición Objetiva**
- ✅ Compara posiciones reales (no alucinaciones)
- ✅ Detecta cambios de mapa
- ✅ Mide tiempo de objetivo activo
- ✅ Valida milestone completion

### 3. **Persistencia Entre Checks**
- ✅ Objetivo no se pierde cuando diálogo expira
- ✅ Mantiene boost mientras hay progreso
- ✅ Limpia automáticamente objetivos completados/obsoletos

## Datos que se Comparan

### En Cada Check (cada 200 steps):

**Estado Anterior** (guardado en `active_objectives`):
```python
{
    'last_pos': (5, 10),
    'last_map': "LITTLEROOT_TOWN",
    'step_set': 200
}
```

**Estado Actual** (leído de memoria del juego):
```python
current_pos = memory_reader.read_position()  # (8, 15)
current_map = memory_reader.read_current_map()  # "ROUTE_101"
current_step = self.num_timesteps  # 400
```

**Comparación**:
```python
# Posición cambió?
if current_pos != last_pos:
    is_moving = True
    
# Mapa cambió?
if current_map != last_map:
    changed_map = True
    is_moving = True  # Cambiar mapa implica movimiento
    
# Mucho tiempo activo?
time_active = current_step - step_set  # 400 - 200 = 200 steps
if time_active > 5000:
    obsolete = True
```

## Decisión Final del Boost

```python
# Prioridad de sobrescritura (de mayor a menor):

1. Milestone completado → 2.0x (máxima prioridad)
2. Objetivo obsoleto (>5000 steps) → 1.0x (limitar)
3. Cambió de mapa → max(llm_multiplier, 1.6x)
4. Se está moviendo → max(llm_multiplier, 1.4x)
5. Objetivo activo pero quieto → max(llm_multiplier, 1.2x)
6. Sin objetivo → llm_multiplier (lo que sugirió el LLM)
```

## Resumen

**¿Cómo se da el boost?**

1. **LLM sugiere boost inicial** basado en análisis de diálogo/contexto
2. **Sistema mide progreso real** comparando estado anterior vs actual
3. **Sistema SOBRESCRIBE** el boost del LLM si hay progreso medible
4. **Se mantiene el boost** mientras el objetivo esté activo y haya movimiento
5. **Se limpia automáticamente** cuando se completa o se vuelve obsoleto

**El LLM da la dirección (qué es importante), el sistema mide la ejecución (qué tanto progresa).**
