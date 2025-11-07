# 🤖 Sistema LLM con Lectura de Diálogos

## 📋 Resumen

El sistema de entrenamiento ahora incluye **lectura inteligente de diálogos** para que el LLM tome decisiones basadas en lo que el juego DICE, no solo en reglas fijas.

---

## 🎯 Características Nuevas

### **1. Milestone Count en Observation**

El agente ahora puede "ver" cuántos milestones ha completado:

```python
observation = {
    'map': np.array([7, 7, 3]),     # Mapa visual
    'vector': np.array([19]),        # Features
}

# vector[16] = milestone_count / 100.0  # 🆕 NUEVO!
```

**Beneficio**: El agente aprende que `milestone_count` subiendo = progreso.

---

### **2. Lectura de Diálogos (Memoria + OCR)**

El LLM ahora lee el texto del juego usando:

```python
dialog = memory_reader.read_dialog_with_ocr_fallback(screenshot)
```

**Prioridad**:
1. **Memoria + OCR detectan** → Usa memoria (más preciso)
2. **Solo OCR detecta** → Usa OCR (memoria falló)
3. **Solo memoria detecta** → Suprime (probablemente texto residual)

---

### **3. LLM Toma Decisiones Inteligentes**

El LLM recibe:

```
Current State:
- Stationary steps: 30
- Milestones completed: 5

📜 Game Dialogue:
"Go north to find PROFESSOR BIRCH!"
```

Y decide:

```json
{
  "multiplier": 1.5,
  "reason": "Dialogue shows objective: go north. Agent moving north.",
  "detected_objective": "Find Professor Birch to the north"
}
```

---

## 🔧 Configuración

### **Requisitos**

1. **Ollama** corriendo localmente:
   ```bash
   # Instalar Ollama
   curl https://ollama.ai/install.sh | sh
   
   # Descargar modelo
   ollama pull llama3
   
   # Iniciar servidor
   ollama serve
   ```

2. **OCR** habilitado (opcional pero recomendado):
   ```python
   # Ya está configurado en memory_reader.py
   self._ocr_detector = create_ocr_detector()
   ```

---

### **Activar el Sistema**

En `train_ppo.py` línea 264:

```python
llm_callback = LLMRewardCallback(
    check_frequency=1000,  # Cada 1000 steps
    use_llm=True,          # 🆕 ACTIVADO!
    verbose=1
)
```

---

## 📊 Comparación de Modos

| Modo | Usa LLM | Lee Diálogos | Decisión | Velocidad |
|------|---------|--------------|----------|-----------|
| **Rule-Based** | ❌ | ❌ | Reglas fijas | ⚡ Rápido |
| **LLM + Diálogos** | ✅ | ✅ | Inteligente | 🐌 Lento |

---

## 🎮 Ejemplos de Uso del LLM

### **Ejemplo 1: Objetivo Detectado**

```
📜 Diálogo: "The DEVON GOODS were stolen! Go to RUSTBORO WOODS!"

🤖 LLM decide:
- Multiplier: 1.6
- Reason: "Clear objective: recover Devon Goods in Rustboro Woods"
- Detected objective: "Go to Rustboro Woods"
```

### **Ejemplo 2: Progreso Confirmado**

```
📜 Diálogo: "You obtained the STONE BADGE!"

🤖 LLM decide:
- Multiplier: 2.0
- Reason: "Major milestone achieved: first gym badge!"
- Detected objective: "Badge obtained, look for next objective"
```

### **Ejemplo 3: Agente Atascado**

```
📜 Diálogo: (none detected)
Stationary steps: 150

🤖 LLM decide:
- Multiplier: 0.3
- Reason: "Agent stuck with no dialogue guidance"
- Detected objective: null
```

---

## 🔍 Cómo Funciona (Interno)

### **Paso 1: Lectura de Diálogos**

```python
# En cada check (1000 steps):
screenshot = emulator.get_screenshot()
dialog = memory_reader.read_dialog_with_ocr_fallback(screenshot)
```

### **Paso 2: Análisis del LLM**

```python
prompt = f"""
Current State:
- Milestones: {completed}
- Stationary: {stationary_steps}

📜 Game Dialogue:
"{dialog_text}"

Analyze: What is the objective? Is agent progressing?
"""

response = ollama.generate(prompt)
```

### **Paso 3: Aplicar Multiplicador**

```python
multiplier = llm_output['multiplier']  # 0.3 - 2.0
reward_final = reward_base * multiplier

# Ejemplo:
0.50 × 1.8 = 0.90  # Boost por milestone
```

---

## ⚙️ Ventajas vs Desventajas

### ✅ **Ventajas**

1. **Objetivos del Juego**: El LLM lee lo que el juego DICE hacer
2. **Adaptativo**: Aprende patrones de diálogo → objetivo
3. **Legal**: Solo usa información que el jugador ve
4. **Milestone Awareness**: El agente sabe cuánto progreso lleva

### ❌ **Desventajas**

1. **Lento**: Llamada a LLM cada 1000 steps (~3-5 segundos)
2. **Requiere Ollama**: Debe estar corriendo en localhost
3. **OCR Imperfecto**: Puede malinterpretar texto
4. **Cambio de Observation**: Requiere reentrenar modelo (vector[19] vs vector[18])

---

## 🚀 Próximos Pasos

### **Para Probar**:

```bash
# 1. Iniciar Ollama
ollama serve

# 2. Entrenar con LLM + Diálogos
python train_ppo.py \
    --mode train \
    --state Emerald-GBAdvance/quick_start_save.state \
    --timesteps 100000 \
    --n-envs 4

# 3. Monitorear logs
tail -f training.log | grep "🤖"
```

### **Logs Esperados**:

```
🤖 [Env 0] LLM: Dialogue shows objective: go to Route 101 | Objective: Find Professor Birch | (multiplier=1.50x)
💰 Reward shaping: 0.50 → 0.75 (LLM:1.50)
```

---

## 📈 Resultados Esperados

**Hipótesis**: 
- El agente aprenderá **más rápido** porque el LLM interpreta objetivos del juego
- Menos tiempo atascado en lugares sin objetivo
- Mejor alineación con progreso real del juego

**Métricas a Monitorear**:
- Milestones por episodio
- Tiempo hasta primer badge
- Reward promedio
- Objetivos detectados por el LLM

---

## 🔄 Volver a Rule-Based

Si el LLM es muy lento o no funciona bien:

```python
# En train_ppo.py línea 266:
use_llm=False  # Desactivar LLM
```

Volverá al sistema de reglas fijas (más rápido).

---

## 📝 Notas Técnicas

### **Cambios en Observation Space**

```python
# ANTES:
'vector': spaces.Box(shape=(18,))

# AHORA:
'vector': spaces.Box(shape=(19,))  # +1 para milestone_count
```

**Implicación**: Modelos pre-entrenados con vector[18] **NO funcionarán**. Hay que reentrenar desde cero.

### **Acceso a Milestone Tracker**

```python
# En lightweight_state_reader.py:
if hasattr(self.mem, 'core') and hasattr(self.mem.core, 'milestone_tracker'):
    milestone_count = len(milestone_tracker.milestones)
```

---

## ✨ Conclusión

Este sistema representa un **híbrido inteligente**:

- **PPO** aprende política (qué botones presionar)
- **Milestone Count** en observation (sabe si progresa)
- **LLM + Diálogos** para reward shaping (interpreta objetivos)

Es como tener un **entrenador humano** que lee el juego y dice "bien hecho" o "estás atascado".

¡Buena suerte con el entrenamiento! 🎮🤖
