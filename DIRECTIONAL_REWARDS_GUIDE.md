# 🧭 Guía de Recompensas Direccionales

## ¿Qué es Reward Shaping Direccional?

Es un sistema que **recompensa progresivamente** al agente cuando se acerca a objetivos importantes, **antes** de completarlos.

---

## 🎯 Problema Anterior vs Solución Actual

### ❌ Antes (Solo Milestones)

```
Agente en ROUTE 101:
├─ Objetivo: Rescatar a Prof Birch (en posición 13, 7)
├─ Posición actual: (5, 5) → distancia = 15 tiles
│
├─ Agente camina hacia objetivo:
│  └─ (6, 5) → (7, 5) → (8, 6) → (9, 7)...
│     └─ Reward: +0.5 cada paso (movimiento)
│     └─ Multiplier: 1.0× (normal, sin guía)
│
└─ Finalmente llega al objetivo:
   └─ ¡Rescata a Birch! → +1000 reward
   └─ Milestone completado → Multiplier: 1.8×

PROBLEMA: Solo recompensa al FINAL, no durante el camino
```

### ✅ Ahora (Direccional + Milestones)

```
Agente en ROUTE 101:
├─ Objetivo detectado: Prof Birch en (13, 7)
├─ Posición inicial: (5, 5) → distancia = 15 tiles
│
├─ Agente se mueve HACIA el objetivo:
│  ├─ (6, 5) → distancia = 14 tiles
│  │  └─ 🧭 "Acercándose" → Multiplier: 1.5×
│  │  └─ Reward: 0.5 → 0.75 (boosted!)
│  │
│  ├─ (7, 5) → distancia = 13 tiles
│  │  └─ 🧭 "Acercándose" → Multiplier: 1.5×
│  │  └─ Reward: 0.5 → 0.75 (boosted!)
│  │
│  └─ (12, 7) → distancia = 1 tile
│     └─ 🎯 "¡MUY CERCA!" → Multiplier: 1.8×
│     └─ Reward: 0.5 → 0.90 (gran boost!)
│
├─ Agente se mueve LEJOS del objetivo:
│  └─ (6, 4) → distancia = 16 tiles (+1)
│     └─ ⚠️ "Alejándose" → Multiplier: 0.8×
│     └─ Reward: 0.5 → 0.40 (penalizado)
│
└─ Finalmente rescata a Birch:
   └─ Milestone → Multiplier LLM: 1.8×
   └─ Muy cerca → Multiplier Dir: 1.8×
   └─ COMBINADO: 1.8 × 1.8 = 3.24× ¡Super boost!

SOLUCIÓN: Recompensa CONTINUAMENTE el progreso correcto
```

---

## 📊 Ejemplo Real de Training

### Sin Direccional (solo LLM):
```bash
Step 1000: 
🔍 LLM Callback: NEW MILESTONE! (×1.80)
💰 Reward: 0.50 → 0.90 (×1.80)  # Solo cuando completa milestone

Steps 1001-1999:
💰 Reward: 0.50 → 0.50 (×1.00)  # Normal, sin guía
💰 Reward: 0.50 → 0.50 (×1.00)
💰 Reward: 0.50 → 0.50 (×1.00)
...
# Agente camina sin saber si va bien o mal
```

### Con Direccional (LLM + Proximidad):
```bash
Step 100:
🧭 [Env 0] ✅ Acercándose a objetivo en ROUTE_101 (-2.0 tiles) (×1.50)
💰 Reward: 0.50 → 0.75 (Dir:1.50)

Step 200:
🧭 [Env 0] ✅ Acercándose a objetivo en ROUTE_101 (-1.5 tiles) (×1.50)
💰 Reward: 0.50 → 0.75 (Dir:1.50)

Step 300:
🧭 [Env 0] 🎯 ¡MUY CERCA del objetivo! (2.0 tiles) (×1.80)
💰 Reward: 0.50 → 0.90 (Dir:1.80)

Step 400:
🧭 [Env 0] ⚠️ Alejándose de objetivo en ROUTE_101 (+1.0 tiles) (×0.80)
💰 Reward: 0.50 → 0.40 (Dir:0.80)

Step 500:
🧭 [Env 0] ✅ Acercándose a objetivo en ROUTE_101 (-1.0 tiles) (×1.50)
💰 Reward: 0.50 → 0.75 (Dir:1.50)

Step 1000:
🔍 LLM Callback: NEW MILESTONE! (×1.80)
🧭 [Env 0] 🎯 ¡MUY CERCA del objetivo! (0.5 tiles) (×1.80)
💰 Reward: 0.50 → 1.62 (LLM:1.80 × Dir:1.80)  # ¡3.24× combinado!
```

---

## 🗺️ Objetivos Configurados

El sistema conoce objetivos importantes en cada mapa:

```python
ROUTE_101: 
  └─ (13, 7) "Prof Birch rescue"

LITTLEROOT_TOWN:
  ├─ (14, 8) "Player's house"
  └─ (7, 8) "Rival's house"

ROUTE_103:
  └─ (4, 4) "Rival battle"

OLDALE_TOWN:
  ├─ (8, 8) "Pokemon Center"
  └─ (13, 7) "Mart"

PETALBURG_CITY:
  └─ (13, 13) "Gym entrance"
```

**¿Cómo funciona?**
- Cada 100 steps, el sistema calcula la distancia al objetivo más cercano
- Si la distancia disminuye → **Boost** (1.5×)
- Si la distancia aumenta → **Penalización** (0.8×)
- Si está muy cerca (<3 tiles) → **Super Boost** (1.8×)

---

## 🎮 Cómo Usar

### Entrenar con Direccional:
```bash
python train_ppo.py --mode train --timesteps 100000 --n-envs 4
```

El sistema ahora tiene **2 capas de reward shaping**:

1. **LLM Callback** (cada 1000 steps):
   - Detecta milestones completados
   - Detecta si está atascado
   - Multiplier: 0.3× a 1.8×

2. **Directional Callback** (cada 100 steps):
   - Detecta si se acerca a objetivos
   - Guía continuamente al agente
   - Multiplier: 0.8× a 1.8×

**Multiplicadores se COMBINAN:**
```
Final reward = base_reward × llm_multiplier × directional_multiplier

Ejemplo:
- Base: 0.5 (movimiento)
- LLM: 1.8× (milestone)
- Direccional: 1.5× (acercándose)
- FINAL: 0.5 × 1.8 × 1.5 = 1.35 🚀
```

---

## 🔧 Agregar Nuevos Objetivos

Edita `agent/directional_reward_callback.py`:

```python
self.known_objectives = {
    "NOMBRE_MAPA": [(x, y, "descripcion")],
    
    # Ejemplo: Agregar Rustboro City
    "RUSTBORO_CITY": [
        (15, 20, "Gym entrance"),
        (10, 15, "Devon Corp"),
        (8, 8, "Pokemon Center")
    ],
}
```

---

## 📈 Beneficios

### Aprendizaje más Eficiente:
- ✅ Agente recibe feedback **cada 100 steps** (antes: solo cuando completa milestone)
- ✅ Aprende **dirección correcta** más rápido
- ✅ Menos tiempo "perdido" caminando sin propósito

### Menos Tiempo Atascado:
- ✅ Si se aleja del objetivo → penalización inmediata
- ✅ Fuerza al agente a **probar nuevas direcciones**

### Mejor Progreso en Historia:
- ✅ Objetivos alineados con progreso del juego
- ✅ Sistema guía hacia milestones naturalmente
- ✅ Combinación con LLM multiplica efectividad

---

## 🧪 Testing

Ver los logs en tiempo real:
```bash
# Filtrar solo reward shaping
tail -f training.log | grep -E "🧭|💰|🔍"
```

Ejemplo de output esperado:
```
🧭 [Env 0] ✅ Acercándose a objetivo en ROUTE_101 (×1.50)
💰 Reward: 0.50 → 0.75 (Dir:1.50)
🧭 [Env 1] 🎯 ¡MUY CERCA del objetivo! (×1.80)
💰 Reward: 0.50 → 0.90 (Dir:1.80)
🔍 LLM Callback: NEW MILESTONE! Total: 8 (×1.80)
💰 Reward: 0.50 → 1.62 (LLM:1.80 × Dir:1.80)
```

---

## ⚙️ Configuración Avanzada

Ajustar multiplicadores en `train_ppo.py`:

```python
directional_callback = DirectionalRewardCallback(
    check_frequency=100,      # Revisar cada N steps
    proximity_boost=1.5,      # Multiplicador cuando se acerca
    proximity_penalty=0.8,    # Multiplicador cuando se aleja
    verbose=1
)
```

**Recomendaciones:**
- `check_frequency=100`: Balance entre precisión y performance
- `proximity_boost=1.5`: Suficiente incentivo sin dominar otros rewards
- `proximity_penalty=0.8`: Penalización suave (no destruye el learning)

---

## 🎯 Conclusión

El sistema ahora responde a tu pregunta original:

> "¿No sería mejor que sepa si está en el lugar correcto?"

**Respuesta: ¡SÍ!** Y ahora lo sabe. El sistema:

1. ✅ Detecta objetivos importantes en cada mapa
2. ✅ Calcula distancia al objetivo más cercano
3. ✅ Recompensa continuamente cuando se acerca
4. ✅ Penaliza cuando se aleja sin razón
5. ✅ Combina con LLM para máxima efectividad

¡Todo sin necesidad de visión compleja o LLM costoso!
