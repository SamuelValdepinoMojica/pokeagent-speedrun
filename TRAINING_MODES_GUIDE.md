# 🎮 Guía de Comparación: 3 Modos de Entrenamiento

## 📋 Resumen

Ahora puedes entrenar **3 modelos diferentes** para compararlos:

| Modo | Descripción | Observation | Reward Shaping |
|------|-------------|-------------|----------------|
| 🔵 **Pure DRL** | DRL puro sin ayuda | Vector [19] con milestone_count | ❌ Ninguno |
| 📊 **Rule-based** | Milestones con reglas | Vector [19] con milestone_count | ✅ Reglas fijas |
| 🤖 **LLM+Dialogue** | LLM lee diálogos | Vector [19] con milestone_count | ✅ LLM inteligente |

---

## 🚀 Uso Rápido

### **Opción 1: Script Automático**

```bash
# Entrena los 3 modelos en secuencia
./train_comparison.sh
```

El script te preguntará cuáles quieres entrenar.

---

### **Opción 2: Manual (un modelo a la vez)**

#### **1️⃣ Pure DRL (sin reward shaping)**

```bash
python train_ppo.py \
    --mode train \
    --state Emerald-GBAdvance/quick_start_save.state \
    --timesteps 100000 \
    --n-envs 4 \
    --model-path ./models/ppo_pure_drl \
    --pure-drl
```

**Características**:
- ✅ Milestone count en observation (vector[16])
- ❌ Sin callbacks de reward shaping
- ⚡ Más rápido (sin LLM overhead)

---

#### **2️⃣ Rule-based Milestones**

```bash
python train_ppo.py \
    --mode train \
    --state Emerald-GBAdvance/quick_start_save.state \
    --timesteps 100000 \
    --n-envs 4 \
    --model-path ./models/ppo_rule_based
```

**Características**:
- ✅ Milestone count en observation
- ✅ LLM Callback con reglas fijas:
  - `stationary > 100` → 0.3× penalty
  - `new milestone` → 1.8× boost
- ⚡ Rápido (no usa Ollama)

---

#### **3️⃣ LLM + Diálogos**

```bash
# Primero iniciar Ollama
ollama serve

# En otra terminal:
python train_ppo.py \
    --mode train \
    --state Emerald-GBAdvance/quick_start_save.state \
    --timesteps 100000 \
    --n-envs 4 \
    --model-path ./models/ppo_llm_dialogue \
    --use-llm
```

**Características**:
- ✅ Milestone count en observation
- ✅ LLM lee diálogos con OCR
- ✅ Decisiones inteligentes basadas en texto del juego
- 🐌 Más lento (llamadas a LLM cada 1000 steps)

---

## 📊 Comparar Resultados

### **Con TensorBoard**

```bash
# Ver todos los entrenamientos juntos
tensorboard --logdir ./tensorboard_logs

# Solo ver modelos específicos
tensorboard --logdir_spec \
    pure:./tensorboard_logs/PPO_pure_drl,\
    rules:./tensorboard_logs/PPO_rule_based,\
    llm:./tensorboard_logs/PPO_llm_dialogue
```

Abre: http://localhost:6006

---

### **Métricas a Comparar**

| Métrica | Significado | Qué Buscar |
|---------|-------------|------------|
| `rollout/ep_rew_mean` | Reward promedio | ⬆️ Más alto = mejor |
| `rollout/ep_len_mean` | Longitud de episodio | Context-dependent |
| `train/approx_kl` | Divergencia KL | Estabilidad |
| `train/explained_variance` | Qué tan bien predice valor | ⬆️ Cerca de 1.0 |

**Hipótesis**:
- **Pure DRL**: Aprende lento pero robusto
- **Rule-based**: Aprende medio-rápido, puede sobre-optimizar milestones
- **LLM+Dialogue**: Aprende rápido si LLM da buenos consejos

---

## 🎯 Diferencias Técnicas

### **Observation Space**

**TODOS usan el mismo observation**:
```python
observation = {
    'map': np.array([7, 7, 3]),  # Mapa 7x7 con 3 canales
    'vector': np.array([19])     # Features incluye milestone_count
}
```

Esto significa que **los modelos son compatibles** - puedes:
- Cargar un modelo pre-entrenado con cualquier modo
- Cambiar de modo durante el entrenamiento
- Comparar directamente el rendimiento

---

### **Reward Function**

**Base reward (igual para todos)**:
```python
# En drl_env.py _calculate_reward_from_lightweight()
reward = 0
reward += 1000 * badges_obtenidos
reward += 50 * level_ups
reward += 0.5 * movimiento
reward -= 0.05 * stuck_penalty
```

**Reward Shaping (diferencias)**:

| Modo | Multiplier | Fuente de Decisión |
|------|------------|-------------------|
| Pure DRL | 1.0 (siempre) | Ninguna |
| Rule-based | 0.3 - 1.8 | Reglas if/else |
| LLM+Dialogue | 0.3 - 2.0 | LLM analiza texto |

---

## 🔬 Experimento Sugerido

### **Plan de Prueba**

1. **Entrenar los 3 modelos** con los mismos parámetros:
   ```bash
   ./train_comparison.sh
   ```

2. **Dejar entrenar** por al menos 100k timesteps cada uno

3. **Evaluar** cada modelo:
   ```bash
   python train_ppo.py --mode test \
       --model-path ./models/ppo_pure_drl \
       --test-episodes 10
   
   python train_ppo.py --mode test \
       --model-path ./models/ppo_rule_based \
       --test-episodes 10
   
   python train_ppo.py --mode test \
       --model-path ./models/ppo_llm_dialogue \
       --test-episodes 10
   ```

4. **Comparar**:
   - Badges obtenidos
   - Milestones completados
   - Tiempo de entrenamiento
   - Estabilidad

---

## 📝 Logs Esperados

### **Pure DRL**
```bash
🔵 Pure DRL mode - ALL reward shaping disabled
Step 10000: reward=5.2, badges=0, milestones=3
Step 20000: reward=8.5, badges=0, milestones=5
```

### **Rule-based**
```bash
📊 Rule-based milestone reward shaping
🎯 [Env 0] Step 10000: NEW MILESTONE! Boosting rewards. Total: 5 (multiplier=1.80x)
💰 Reward shaping: 0.50 → 0.90 (LLM:1.80)
```

### **LLM+Dialogue**
```bash
🤖 LLM + Dialogue-based reward shaping
🤖 [Env 0] LLM: Dialogue says go north to find Birch | Objective: Find Professor Birch | (multiplier=1.50x)
💰 Reward shaping: 0.50 → 0.75 (LLM:1.50)
```

---

## ⚠️ Consideraciones

### **Tiempo de Entrenamiento**

| Modo | Tiempo estimado (100k steps, 4 envs) |
|------|--------------------------------------|
| Pure DRL | ~30 minutos |
| Rule-based | ~35 minutos |
| LLM+Dialogue | ~50-60 minutos (por LLM calls) |

### **Requisitos**

- **Pure DRL**: Solo Python + mGBA
- **Rule-based**: Solo Python + mGBA
- **LLM+Dialogue**: Python + mGBA + **Ollama** + Modelo LLM descargado

---

## 🎉 Conclusión

Con este sistema puedes:

✅ **Experimentar** con diferentes enfoques de reward shaping  
✅ **Comparar** resultados objetivamente  
✅ **Publicar** findings científicos (qué método funciona mejor)  
✅ **Aprender** qué papel juega el reward shaping en DRL  

¡Buena suerte con tus experimentos! 🚀
