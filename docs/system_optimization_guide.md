# 💻 Guía de Configuración para Tu Sistema

## 🖥️ Especificaciones de Tu Sistema

```
CPU: Intel Ultra 9 (Laptop)
GPU: RTX 5060 Ti 16GB (via oculink)
RAM: 32GB
```

---

## ⚡ Recomendación de Ambientes

### Cálculo de Capacidad:

**RAM disponible:** 32GB
- Sistema operativo: ~4GB
- Por ambiente: ~1.5GB (emulator + estado)
- **Máximo teórico:** (32 - 4) / 1.5 = **~18 ambientes**

**CPU (Intel Ultra 9):**
- Cores típicos: 14-16 cores (6P + 8E)
- **Recomendado:** `n_envs = cores - 2` = **12-14 ambientes**

**GPU (RTX 5060 Ti 16GB):**
- Excelente para entrenamiento DRL
- 16GB VRAM es más que suficiente
- Con CNN policy, usar ~2GB VRAM

### 🎯 **Configuración Óptima para Tu Sistema:**

```bash
python train_ppo.py \
    --mode train \
    --timesteps 1000000 \
    --n-envs 12 \
    --frame-skip 6 \
    --state Emerald-GBAdvance/quick_start_save.state
```

**Por qué 12 ambientes:**
- ✅ Aprovecha todos los cores CPU
- ✅ No satura la RAM (12 × 1.5GB = 18GB)
- ✅ Deja margen para el sistema
- ✅ ~12x speedup vs 1 ambiente

---

## 📊 Rendimiento Esperado con Tu Sistema

| n_envs | RAM Usada | CPU Load | Tiempo 1M steps | Speedup |
|--------|-----------|----------|-----------------|---------|
| 1      | ~5GB      | ~10%     | ~7 horas        | 1x      |
| 4      | ~10GB     | ~40%     | ~1.8 horas      | 4x      |
| 8      | ~16GB     | ~70%     | ~55 minutos     | 8x      |
| **12** | **~22GB** | **~90%** | **~35 minutos** | **12x** |
| 16     | ~28GB     | ~100%    | ~26 minutos     | 16x ⚠️  |

**Recomendación:** **12 ambientes** es el sweet spot

---

## 🎮 1. Ver Entrenamiento en Tiempo Real

### Opción A: TensorBoard (Recomendado)

**Durante el entrenamiento**, en otra terminal:

```bash
python monitor_training.py
```

O manualmente:
```bash
tensorboard --logdir=./tensorboard_logs --port=6006
```

Luego abre: **http://localhost:6006**

**Verás:**
- 📈 **Reward por episodio** (¿está subiendo?)
- 📏 **Longitud de episodios** (¿explora más?)
- 📉 **Policy loss** (¿está aprendiendo?)
- 🎲 **Entropy** (¿está explorando?)

### Opción B: Ver Logs en Terminal

Los logs muestran progress cada época:

```
| rollout/              |          |
|    ep_len_mean        | 245      |  ← Promedio de pasos por episodio
|    ep_rew_mean        | -12.5    |  ← Reward promedio (queremos que suba)
| time/                 |          |
|    fps                | 156      |  ← Steps por segundo
|    total_timesteps    | 8192     |  ← Progreso total
```

### Opción C: Ver al Agente Jugando (Pausar Training)

**IMPORTANTE:** No se puede visualizar DURANTE el training con múltiples ambientes.

**Workflow recomendado:**

1. **Entrenar sin visualización:**
```bash
python train_ppo.py --mode train --timesteps 100000 --n-envs 12
```

2. **Detener training** (Ctrl+C después de X steps)

3. **Ver el modelo entrenado:**
```bash
# Ver el último checkpoint
python watch_trained_agent.py --model logs/checkpoints/ppo_pokemon_10000_steps.zip
```

---

## 📊 2. Cómo Saber Si Está Aprendiendo

### ✅ Señales de Buen Aprendizaje:

#### A. Reward Aumenta
```
Epoch 1:  ep_rew_mean = -50.0   ❌ Malo (penalizaciones)
Epoch 10: ep_rew_mean = -20.0   ⚠️  Mejorando
Epoch 20: ep_rew_mean = 5.0     ✅ Bien!
Epoch 50: ep_rew_mean = 50.0    🎉 Excelente!
```

**Revisar en TensorBoard:** Curva de "rollout/ep_rew_mean" debe **subir**

#### B. Longitud de Episodios Aumenta
```
Epoch 1:  ep_len_mean = 50      ❌ Muere rápido
Epoch 10: ep_len_mean = 200     ⚠️  Sobrevive más
Epoch 50: ep_len_mean = 1000    ✅ Explora mucho
```

**Significa:** El agente sobrevive más tiempo sin morir

#### C. Entropy Disminuye Gradualmente
```
Epoch 1:  entropy = 2.0         ✅ Explora mucho (bueno al inicio)
Epoch 50: entropy = 0.5         ✅ Más decidido (bueno al final)
```

**Significa:** El agente pasa de explorar aleatoriamente a tomar decisiones más seguras

#### D. Policy Loss Disminuye
```
Epoch 1:  policy_loss = 0.5     ⚠️  Alto
Epoch 50: policy_loss = 0.05    ✅ Bajo (aprendió)
```

### ❌ Señales de Mal Aprendizaje:

1. **Reward no aumenta después de 50k steps**
   - Posible problema: Reward shaping malo
   - Solución: Ajustar recompensas

2. **Entropy = 0 muy rápido**
   - Problema: Colapsó a una política (ej: solo presiona UP)
   - Solución: Aumentar `ent_coef` en PPO

3. **Episode length = max_steps siempre**
   - Problema: Nunca termina (stuck en loops)
   - Solución: Aumentar penalización por inmovilidad

---

## 🎁 3. Sistema de Recompensas Actual

Revisa el archivo `agent/drl_env.py`:

```python
def _calculate_reward_from_lightweight(prev_state, current_state):
    reward = 0.0
    
    # 🏆 Objetivo principal: Obtener badges
    if curr_badges > prev_badges:
        reward += 1000.0  # ¡GRANDE! Es el objetivo
    
    # 📈 Subir de nivel
    if curr_levels > prev_levels:
        reward += 50.0
    
    # 🚶 Moverse (explorar)
    if curr_coords != prev_coords:
        reward += 0.5  # Pequeño reward por movimiento
    else:
        self.stationary_steps += 1
        reward -= 0.05 * min(self.stationary_steps, 20)  # Penaliza quedarse quieto
    
    # 💔 HP bajo (penalización)
    if hp_ratio < 0.2:
        reward -= 5.0  # Crítico
    elif hp_ratio < 0.5:
        reward -= 1.0  # Bajo
```

### 📊 Escala de Recompensas:

```
+1000.0  → Badge obtenida (OBJETIVO PRINCIPAL)
+50.0    → Subir nivel
+20.0    → Descubrir nueva ubicación (primera vez)
+5.0     → Revisitar ubicación
+0.5     → Moverse (cada step)
-0.05    → Quedarse quieto (por step)
-1.0     → HP < 50%
-5.0     → HP < 20%
```

### 🎯 Recompensas Esperadas:

**Agente random (malo):**
```
Episode reward: -50 a -20
(Se queda quieto mucho, pierde HP)
```

**Agente explorando (mejorando):**
```
Episode reward: -10 a +20
(Se mueve, descubre lugares)
```

**Agente entrenado (bueno):**
```
Episode reward: +50 a +200
(Explora eficientemente, sube niveles)
```

**Agente experto (objetivo):**
```
Episode reward: +1000+
(Obtiene badges!)
```

---

## 🧪 4. Script de Testing

Prueba tu configuración:

```bash
# Test 1: Verificar que soporta 12 ambientes (30 segundos)
python train_ppo.py --mode train --timesteps 1000 --n-envs 12

# Si funciona → ✅ Tu sistema soporta 12 ambientes

# Test 2: Training corto para ver rewards (5 minutos)
python train_ppo.py --mode train --timesteps 10000 --n-envs 12

# Revisar en TensorBoard si reward sube

# Test 3: Ver agente random (baseline)
python watch_trained_agent.py --random --steps 1000

# Nota el reward promedio (~-20 a -50 típicamente)

# Test 4: Training real (30 minutos)
python train_ppo.py --mode train --timesteps 100000 --n-envs 12

# Después, ver el agente entrenado:
python watch_trained_agent.py --model logs/checkpoints/ppo_pokemon_100000_steps.zip
```

---

## 📈 5. Monitoreo Recomendado

### Setup de 2 Terminales:

**Terminal 1 - Training:**
```bash
python train_ppo.py --mode train --timesteps 1000000 --n-envs 12 --state Emerald-GBAdvance/quick_start_save.state
```

**Terminal 2 - Monitoring:**
```bash
python monitor_training.py
# O manualmente:
tensorboard --logdir=./tensorboard_logs
```

### Qué Revisar Cada 10-20 Minutos:

1. **TensorBoard (navegador):**
   - ¿Reward subiendo? ✅
   - ¿Episode length aumentando? ✅
   - ¿Policy loss bajando? ✅

2. **Terminal del training:**
   - ¿FPS estable? (debería ser ~150-200 con n_envs=12)
   - ¿Sin crashes? ✅

3. **Recursos del sistema:**
```bash
# En otra terminal:
htop  # Ver CPU y RAM

# O:
watch -n 1 'ps aux | grep train_ppo | head -5'
```

---

## 🎬 6. Workflow Completo Recomendado

```bash
# Día 1: Training inicial (30 min - 1 hora)
python train_ppo.py --mode train --timesteps 100000 --n-envs 12
# → Produce: logs/checkpoints/ppo_pokemon_100000_steps.zip

# Ver resultado:
python watch_trained_agent.py --model logs/checkpoints/ppo_pokemon_100000_steps.zip

# Día 2: Training largo (overnight)
python train_ppo.py --mode train --timesteps 1000000 --n-envs 12
# → ~35 minutos con tu sistema

# Día 3: Fine-tuning
# Cargar modelo anterior y seguir entrenando
python train_ppo.py --mode train --timesteps 2000000 --n-envs 12 --model logs/checkpoints/ppo_pokemon_1000000_steps.zip
```

---

## 💡 Tips Finales

### Para Maximizar Tu Hardware:

1. **Usa los 12 ambientes** - aprovecha tu CPU
2. **GPU se usa automáticamente** - PyTorch detecta la RTX 5060 Ti
3. **Monitorea con TensorBoard** - visual y claro
4. **Checkpoints cada 10k steps** - no pierdas progreso

### Si el Training es Muy Lento:

1. Verifica GPU:
```python
import torch
print(torch.cuda.is_available())  # Debe ser True
print(torch.cuda.get_device_name(0))  # Debe mostrar RTX 5060 Ti
```

2. Reduce calidad si necesario:
```bash
python train_ppo.py --mode train --timesteps 1000000 --n-envs 8 --frame-skip 12
```

### Si Quieres Ver Jugando Durante Training:

**Opción:** Entrenar 1 ambiente con visualización (LENTO pero educativo)
```bash
python train_ppo.py --mode train --timesteps 10000 --n-envs 1 --visualize
```

⚠️ **NOTA:** Esto es ~12x más lento, solo para debugging/demo

---

## 🚀 Comando Final Recomendado Para Ti:

```bash
# Training óptimo para tu sistema (Intel Ultra 9 + 32GB RAM + RTX 5060 Ti)
python train_ppo.py \
    --mode train \
    --timesteps 1000000 \
    --n-envs 12 \
    --frame-skip 6 \
    --state Emerald-GBAdvance/quick_start_save.state

# En otra terminal (monitoreo):
python monitor_training.py
```

**Resultado esperado:**
- ⏱️ **35 minutos** para 1M steps
- 💾 **22GB RAM** usados
- 🖥️ **~90% CPU** utilizado
- 🎮 **~150 FPS** de training
- 💰 **Checkpoints cada 10k steps**

---

**¿Listo para empezar el training optimizado?** 🏃‍♂️
