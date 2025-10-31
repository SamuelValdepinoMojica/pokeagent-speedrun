# 📦 Guía de Archivos del Proyecto DRL para Pokemon Emerald

## 🎯 Archivos Creados/Modificados para DRL Training

### ✅ **ARCHIVOS ESENCIALES** (para entrenar el agente)

#### 1. **`agent/lightweight_state_reader.py`** ⭐ NUEVO
**Propósito:** Lee estado del juego de forma optimizada (30x más rápido)
**Uso:** El environment DRL lo usa para obtener observaciones rápidas
**Necesario para:** Training con velocidad práctica

```python
# Funciones principales:
- get_drl_state() → Estado básico para DRL
- get_observation_for_drl() → Observación en formato (map, vector)
```

#### 2. **`agent/drl_env.py`** ⭐ MODIFICADO
**Propósito:** Environment Gymnasium para Stable Baselines3
**Cambios:** 
- Integra `LightweightStateReader` 
- Métodos optimizados: `_calculate_reward_from_lightweight()`, `_check_terminated_from_lightweight()`
**Necesario para:** Todo el training

```python
# Uso:
env = PokemonEmeraldEnv(
    rom_path="Emerald-GBAdvance/rom.gba",
    initial_state_path="Emerald-GBAdvance/quick_start_save.state",
    frame_skip=6,
    max_steps=10000
)
```

#### 3. **`train_ppo.py`** (YA EXISTÍA)
**Propósito:** Script principal para entrenar con PPO
**Uso:** `python train_ppo.py --mode train --timesteps 100000`
**Necesario para:** Iniciar training

---

### 📊 **ARCHIVOS DE ANÁLISIS** (útiles pero no esenciales)

#### 4. **`benchmark_speed.py`** ⭐ NUEVO
**Propósito:** Medir velocidad del environment (FPS)
**Uso:** `python benchmark_speed.py --steps 500 --frame-skip 6`
**Utilidad:** Verificar que la optimización funciona

#### 5. **`visualize_observations.py`** ⭐ NUEVO
**Propósito:** Visualizar qué ve el agente (map 7x7x3 + vector 18)
**Uso:** `python visualize_observations.py`
**Utilidad:** Debug - entender las observaciones

#### 6. **`watch_training.py`** ⭐ NUEVO
**Propósito:** Ver al agente jugando (con o sin modelo entrenado)
**Uso:** 
```bash
python watch_training.py --model models/ppo_pokemon_100000_steps.zip
python watch_training.py --random  # Ver acciones aleatorias
```

#### 7. **`compare_state_data.py`** ⭐ NUEVO
**Propósito:** Comparar Comprehensive vs Lightweight state
**Uso:** `python compare_state_data.py`
**Utilidad:** Documentación - mostrar diferencias

#### 8. **`visualize_map_sizes.py`** ⭐ NUEVO
**Propósito:** Crear gráfica de 15x15 vs 7x7 mapa
**Uso:** `python visualize_map_sizes.py`
**Utilidad:** Documentación visual

---

### 📝 **ARCHIVOS DE DOCUMENTACIÓN**

#### 9. **`docs/state_comparison.md`** ⭐ NUEVO
**Propósito:** Explicación detallada de diferencias entre estados
**Utilidad:** Entender qué lee cada método

---

## 🚀 Para Compartir con Compañeros

### **Opción 1: Archivos Mínimos (Solo para entrenar)**

Si tus compañeros solo quieren **entrenar el agente**, necesitan:

```
📦 Archivos esenciales:
├── agent/
│   ├── lightweight_state_reader.py  ⭐ NUEVO
│   ├── drl_env.py                   ⭐ MODIFICADO
│   ├── __init__.py
│   ├── action.py
│   ├── perception.py
│   └── ... (resto sin cambios)
├── pokemon_env/
│   └── ... (todo sin cambios)
├── train_ppo.py                     (sin cambios)
├── requirements.txt
├── Emerald-GBAdvance/
│   ├── rom.gba
│   ├── quick_start_save.state
│   └── ... 
└── README.md
```

**Comando para crear paquete mínimo:**
```bash
# Desde la raíz del proyecto
tar -czf drl_training_minimal.tar.gz \
    agent/lightweight_state_reader.py \
    agent/drl_env.py \
    agent/__init__.py \
    agent/action.py \
    agent/perception.py \
    agent/simple.py \
    agent/memory.py \
    agent/planning.py \
    agent/system_prompt.py \
    pokemon_env/ \
    utils/ \
    train_ppo.py \
    requirements.txt \
    Emerald-GBAdvance/rom.gba \
    Emerald-GBAdvance/quick_start_save.state \
    README.md
```

**Instrucciones para compañeros:**
```bash
# 1. Extraer
tar -xzf drl_training_minimal.tar.gz
cd pokeagent-speedrun

# 2. Instalar dependencias
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Entrenar
python train_ppo.py --mode train --timesteps 100000 --state Emerald-GBAdvance/quick_start_save.state
```

---

### **Opción 2: Archivos Completos (Con análisis y debugging)**

Si quieren **entender y analizar** el proyecto:

```bash
# Crear paquete completo
tar -czf drl_training_full.tar.gz \
    agent/ \
    pokemon_env/ \
    utils/ \
    train_ppo.py \
    benchmark_speed.py \
    visualize_observations.py \
    watch_training.py \
    compare_state_data.py \
    visualize_map_sizes.py \
    docs/state_comparison.md \
    requirements.txt \
    Emerald-GBAdvance/ \
    README.md
```

**Scripts disponibles:**
```bash
# 1. Benchmark de velocidad
python benchmark_speed.py --steps 500 --frame-skip 6

# 2. Ver observaciones del agente
python visualize_observations.py

# 3. Ver agente jugando
python watch_training.py --random

# 4. Comparar estados
python compare_state_data.py

# 5. Visualizar mapas
python visualize_map_sizes.py

# 6. Entrenar
python train_ppo.py --mode train --timesteps 100000
```

---

### **Opción 3: Solo los Cambios (Para revisar)**

Si tus compañeros **ya tienen el proyecto base** y solo quieren ver tus cambios:

```bash
# Crear patch con solo los cambios
git diff > drl_optimization.patch

# O crear zip solo con archivos nuevos/modificados
zip -r drl_changes.zip \
    agent/lightweight_state_reader.py \
    agent/drl_env.py \
    benchmark_speed.py \
    visualize_observations.py \
    watch_training.py \
    compare_state_data.py \
    visualize_map_sizes.py \
    docs/state_comparison.md
```

**Instrucciones:**
```bash
# Aplicar cambios sobre proyecto existente
unzip drl_changes.zip

# O con git patch:
git apply drl_optimization.patch
```

---

## 📋 Lista de Verificación para Compartir

### **Antes de compartir, verifica que incluyes:**

- [x] **ROM file**: `Emerald-GBAdvance/rom.gba` (¡importante!)
- [x] **Save state**: `Emerald-GBAdvance/quick_start_save.state`
- [x] **Archivos Python**: Todos los `.py` necesarios
- [x] **Requirements**: `requirements.txt` con:
  ```
  stable-baselines3[extra]
  gymnasium
  torch
  numpy
  pillow
  mgba
  ... (resto de dependencias)
  ```
- [x] **README**: Con instrucciones de uso
- [ ] **Modelos entrenados** (opcional): `models/*.zip` si tienes

### **Archivos que NO necesitan:**

- ❌ `__pycache__/` (generados automáticamente)
- ❌ `.venv/` (cada uno crea su propio virtualenv)
- ❌ `llm_logs/` (logs viejos)
- ❌ `.git/` (si compartes como ZIP/TAR)
- ❌ Archivos temporales (`.pyc`, `.log`, etc.)

---

## 🔑 Archivos Clave por Funcionalidad

### **Para Training (ESENCIALES):**
```
1. agent/lightweight_state_reader.py  ← Optimización de velocidad
2. agent/drl_env.py                   ← Environment con lightweight reader
3. train_ppo.py                       ← Script de training
4. Emerald-GBAdvance/rom.gba         ← Juego
5. Emerald-GBAdvance/*.state         ← Save states
```

### **Para Debugging:**
```
1. benchmark_speed.py           ← Medir FPS
2. visualize_observations.py    ← Ver qué ve el agente
3. watch_training.py            ← Ver agente jugando
4. compare_state_data.py        ← Comparar estados
```

### **Para Documentación:**
```
1. docs/state_comparison.md     ← Explicación técnica
2. visualize_map_sizes.py       ← Gráficas
3. README.md                    ← Instrucciones generales
```

---

## 💡 Recomendación Final

**Para compartir con compañeros de equipo:**

1. **Crear branch en Git:**
   ```bash
   git checkout -b feature/drl-optimization
   git add agent/lightweight_state_reader.py agent/drl_env.py benchmark_speed.py
   git commit -m "Add lightweight state reader for 30x training speedup"
   git push origin feature/drl-optimization
   ```

2. **O crear paquete completo:**
   ```bash
   # Ejecuta el script de empaquetado
   ./create_package.sh  # (créalo basándote en la sección de arriba)
   ```

3. **Incluir documentación:**
   - Link al `docs/state_comparison.md`
   - Resultados del benchmark (22 FPS → 239 FPS)
   - Instrucciones de uso

**¿Necesitas que:**
1. **Cree un script de empaquetado automático?**
2. **Genere un README específico para compartir?**
3. **Cree un documento de "Release Notes"?**

¿Qué prefieres?
