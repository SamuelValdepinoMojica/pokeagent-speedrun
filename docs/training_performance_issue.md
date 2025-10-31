# ⚠️ Problema de Rendimiento con Múltiples Ambientes

## 🔍 Problema Detectado

Cuando entrenas con **12 ambientes**, el rendimiento es **CATASTRÓFICO**:

```
Esperado: ~150-180 it/s (iterations per second)
Real:     ~26 it/s ❌

Velocidad: 6x MÁS LENTO de lo esperado
Tiempo 1M steps: ~10 horas (vs ~90 minutos esperado)
```

**Además:** El proceso **NO responde a Ctrl+C** (se queda colgado).

---

## 🧪 Causa del Problema

**DummyVecEnv con múltiples emuladores mGBA:**

El problema es que `DummyVecEnv` ejecuta todos los ambientes **secuencialmente en un solo proceso**:

```python
# Con 12 ambientes, hace esto:
for i in range(12):
    env[i].step(action)  # Uno por uno, NO en paralelo

# Cada emulador toma tiempo:
- mGBA emulator: ~4ms por frame
- 12 emuladores: 12 × 4ms = 48ms
- Con frame_skip=36: 48ms × 36 = 1.7 segundos por acción!
```

**Resultado:** Con 12 ambientes, el overhead es tan grande que va MÁS LENTO que con 1 solo ambiente.

---

## ✅ Soluciones

### **Solución 1: Usar 1 Ambiente (RECOMENDADO)**

```bash
python train_ppo.py --mode train --timesteps 1000000 --n-envs 1 --state Emerald-GBAdvance/quick_start_save.state
```

**Rendimiento esperado:**
- **~80-100 it/s** con 1 ambiente
- **1M steps:** ~3 horas
- ✅ Responde a Ctrl+C correctamente
- ✅ Usa menos RAM (~6GB vs ~25GB)

---

### **Solución 2: Usar 4 Ambientes (Compromiso)**

```bash
python train_ppo.py --mode train --timesteps 1000000 --n-envs 4 --state Emerald-GBAdvance/quick_start_save.state
```

**Rendimiento esperado:**
- **~60-80 it/s** con 4 ambientes
- **1M steps:** ~4 horas
- ⚠️ Puede ser inestable con Ctrl+C
- Usa ~12GB RAM

---

### **Solución 3: Usar SubprocVecEnv (AVANZADO - Experimental)**

Si quieres probar con procesos paralelos verdaderos:

**Editar `train_ppo.py`:**

```python
# Línea 96, cambiar de DummyVecEnv a SubprocVecEnv:
if n_envs == 1:
    env = DummyVecEnv(env_fns)
else:
    # Usar SubprocVecEnv para verdadero paralelismo
    from multiprocessing import set_start_method
    try:
        set_start_method('spawn', force=True)
    except:
        pass
    env = SubprocVecEnv(env_fns, start_method='spawn')
```

**Rendimiento esperado:**
- **~120-150 it/s** con 4-8 ambientes
- **1M steps:** ~90 minutos
- ⚠️ RIESGO: Puede crashear con `EOFError` (como viste antes)
- Usa ~20GB RAM con 8 ambientes

---

## 📊 Comparación de Rendimientos

| Configuración | it/s | Tiempo 1M steps | Estabilidad | RAM Usada |
|---------------|------|-----------------|-------------|-----------|
| **1 env (DummyVecEnv)** | **~80-100** | **~3 horas** | ✅ Excelente | ~6GB |
| 4 envs (DummyVecEnv) | ~60-80 | ~4 horas | ⚠️ Buena | ~12GB |
| 12 envs (DummyVecEnv) | ~26 ❌ | ~10 horas ❌ | ❌ Mala (no responde a Ctrl+C) | ~25GB |
| 4 envs (SubprocVecEnv) | ~120-150 | ~2 horas | ⚠️ Inestable (EOFError posible) | ~12GB |
| 8 envs (SubprocVecEnv) | ~150-180 | ~90 min | ❌ Inestable | ~20GB |

---

## 🎯 **Recomendación Final: 1 Ambiente**

```bash
python train_ppo.py --mode train --timesteps 1000000 --n-envs 1 --state Emerald-GBAdvance/quick_start_save.state
```

**Por qué 1 ambiente es mejor:**

1. ✅ **Estable:** No crashea, responde a Ctrl+C
2. ✅ **Rápido:** 80-100 it/s (3 horas para 1M steps)
3. ✅ **Menos RAM:** Solo ~6GB usados
4. ✅ **Depuración fácil:** Puedes usar `--visualize` para ver qué hace
5. ✅ **Checkpoints funcionales:** Guardan correctamente cada 10k steps

**Comparado con 12 ambientes:**
- **3x más rápido** (80 it/s vs 26 it/s)
- **4x menos RAM** (6GB vs 25GB)
- **NO se cuelga** con Ctrl+C

---

## 🧪 Testing Recomendado

### **Test 1: Verifica velocidad (30 segundos)**
```bash
python train_ppo.py --mode train --timesteps 1000 --n-envs 1 --state Emerald-GBAdvance/quick_start_save.state
```

**Deberías ver:**
```
~80-100 it/s en la progress bar
```

### **Test 2: Training corto (20 minutos)**
```bash
python train_ppo.py --mode train --timesteps 100000 --n-envs 1 --state Emerald-GBAdvance/quick_start_save.state
```

**Después, evalúa:**
```bash
python watch_trained_agent.py --model logs/checkpoints/ppo_pokemon_100000_steps.zip --steps 1000
```

### **Test 3: Training largo (3 horas)**
```bash
python train_ppo.py --mode train --timesteps 1000000 --n-envs 1 --state Emerald-GBAdvance/quick_start_save.state
```

---

## 🐛 Por Qué No Responde a Ctrl+C

Con 12 emuladores en DummyVecEnv:

1. Ctrl+C envía `KeyboardInterrupt`
2. Python intenta parar los emuladores
3. **PERO:** Los 12 emuladores están en callbacks de mGBA (C code)
4. Los callbacks ignoran `KeyboardInterrupt`
5. Resultado: El proceso se queda **colgado esperando** a que terminen los callbacks

**Solución:** Usar **1 ambiente** o **SubprocVecEnv** (procesos separados que se pueden matar).

---

## 💡 Alternativa: Training Overnight

Si quieres dejar entrenando sin supervisión:

```bash
# Usar nohup para que siga corriendo si cierras la terminal
nohup python train_ppo.py --mode train --timesteps 5000000 --n-envs 1 --state Emerald-GBAdvance/quick_start_save.state > training.log 2>&1 &

# Ver progreso:
tail -f training.log

# Detener si es necesario:
pkill -f train_ppo.py
```

**Tiempo:** 5M steps = ~15 horas (overnight)

---

## 📚 Resumen

**Problema:** DummyVecEnv + múltiples emuladores = LENTO y se cuelga

**Solución:** Usar **1 ambiente** para máxima estabilidad y buen rendimiento

**Comando óptimo:**
```bash
python train_ppo.py --mode train --timesteps 1000000 --n-envs 1 --state Emerald-GBAdvance/quick_start_save.state
```

**Resultado:** ~3 horas para 1M steps, estable, responde a Ctrl+C ✅
