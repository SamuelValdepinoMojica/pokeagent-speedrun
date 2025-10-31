# 🎮 Pokemon Emerald DRL con CNN

Entrenamiento de agentes de Deep Reinforcement Learning (PPO) para Pokemon Emerald usando CNNs para procesar el mapa.

## 🏗️ Arquitectura

### **Observación (Observation Space)**

El agente recibe dos tipos de información:

1. **Mapa (7x7x3)** - Procesado con CNN
   - Canal 0: **Metatile ID** (tipo de tile: grass, water, door, etc.)
   - Canal 1: **Behavior** (comportamiento: walkable, surf, encounter, etc.)
   - Canal 2: **Collision** (0 = puede caminar, 1 = bloqueado)

2. **Vector (18 features)** - Procesado con MLP
   - Posición del jugador (x, y)
   - Party Pokémon (6 x 2 = 12): nivel y HP% de cada Pokémon
   - Estado del juego (4): dinero, badges, en batalla, pokédex

### **Política CNN Personalizada**

```
Map (7x7x3) ──► CNN ──────┐
                          ├──► Fusion ──► Actor/Critic
Vector (18) ──► MLP ──────┘

CNN: Conv2D(3→32) → Conv2D(32→64) → Flatten
MLP: Linear(18→64) → Linear(64→128)
Fusion: Concat → Linear(combined→256)
```

## 📦 Instalación

```bash
# 1. Instalar dependencias básicas
pip install gymnasium stable-baselines3[extra] tensorboard

# 2. Instalar dependencias del proyecto
pip install mgba-py pillow numpy

# 3. (Opcional) Para GPU
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Uso Rápido

### **1. Probar el entorno**

```bash
python test_cnn_env.py
```

Esto verificará:
- ✅ El entorno se inicializa correctamente
- ✅ Las observaciones tienen la forma correcta
- ✅ Los canales del mapa contienen datos válidos
- ✅ Las acciones funcionan

### **2. Entrenar el agente**

```bash
# Entrenamiento básico (1M steps, ~1-2 horas en GPU)
python train_ppo.py --mode train --timesteps 1000000

# Entrenamiento largo (10M steps, ~10-20 horas en GPU)
python train_ppo.py --mode train --timesteps 10000000

# Con estado personalizado
python train_ppo.py --mode train \
    --state Emerald-GBAdvance/truck_start.state \
    --timesteps 5000000
```

### **3. Monitorear entrenamiento**

En otra terminal:

```bash
tensorboard --logdir ./tensorboard_logs
```

Abre http://localhost:6006 para ver:
- Reward por episodio
- Pérdidas del actor/critic
- Badges obtenidas
- Locaciones visitadas

### **4. Probar modelo entrenado**

```bash
python train_ppo.py --mode test \
    --model-path ./models/ppo_pokemon.zip \
    --test-episodes 10
```

## 🎯 Recompensas (Reward Shaping)

El agente aprende con estas recompensas:

| Evento | Recompensa | Descripción |
|--------|------------|-------------|
| 🏆 Badge obtenida | +1000.0 | Objetivo principal |
| 📈 Subir de nivel | +50.0 | Por cada nivel ganado (suma de party) |
| 🗺️ Nueva ubicación | +20.0 | Primera vez en ubicación |
| 🚶 Regresar a ubicación | +5.0 | Ubicación ya visitada |
| 🏃 Moverse | +0.5 | Por cada paso dado |
| 🧍 Quedarse quieto | -0.1 a -1.0 | Penalización creciente |
| 💔 HP crítico (<20%) | -5.0 | Por Pokémon con HP bajo |
| 🩹 HP bajo (20-50%) | -1.0 | Por Pokémon con HP medio |

## 📊 Arquitectura CNN Detallada

### **Feature Extractor**

```python
PokemonCNNExtractor(
    # CNN para mapa
    map_cnn: Sequential(
        Conv2d(3, 32, kernel_size=3, padding=1),  # 7x7x3 → 7x7x32
        ReLU(),
        BatchNorm2d(32),
        Conv2d(32, 64, kernel_size=3, padding=1), # 7x7x32 → 7x7x64
        ReLU(),
        BatchNorm2d(64),
        Flatten()  # 7x7x64 = 3136 features
    ),
    
    # MLP para vector
    vector_mlp: Sequential(
        Linear(18, 64),
        ReLU(),
        Linear(64, 128),
        ReLU()
    ),
    
    # Fusion
    fusion: Sequential(
        Linear(3136+128=3264, 256),  # Combina ambas ramas
        ReLU()
    )
)
```

### **Actor-Critic**

```python
Policy Network:
  features (256) → actor_net → action_logits (8)
  features (256) → critic_net → value (1)
```

## 🔧 Hiperparámetros

```python
PPO(
    learning_rate=3e-4,      # Tasa de aprendizaje
    n_steps=2048,            # Steps por actualización
    batch_size=64,           # Tamaño de batch
    n_epochs=10,             # Épocas por actualización
    gamma=0.99,              # Factor de descuento
    gae_lambda=0.95,         # GAE lambda
    clip_range=0.2,          # PPO clip range
    ent_coef=0.01,           # Coeficiente de entropía
)
```

## 📁 Estructura de Archivos

```
agent/
  ├── drl_env.py           # Entorno Gymnasium con CNN observations
  ├── cnn_policy.py        # Política CNN personalizada
  └── __init__.py

train_ppo.py               # Script de entrenamiento
test_cnn_env.py            # Script de prueba

models/                    # Modelos entrenados guardados aquí
logs/                      # Logs de entrenamiento
  └── checkpoints/         # Checkpoints cada 10k steps
tensorboard_logs/          # Logs para TensorBoard
```

## 🎮 Por qué usar CNN para el mapa?

### **Ventajas:**

1. **Patrones espaciales**: La CNN aprende a reconocer:
   - Caminos vs obstáculos
   - Puertas y entradas
   - Agua para Surf
   - Hierba alta (encuentros)

2. **Invarianza a traslación**: Reconoce un camino sin importar dónde esté en el grid

3. **Features jerárquicas**: 
   - Primeras capas: bordes, tiles individuales
   - Capas profundas: patrones complejos (habitaciones, corredores)

4. **Mejor que vectorizar**: Un vector plano pierde información espacial

### **Ejemplo visual:**

```
Mapa crudo:          CNN ve:
░░░░███░░           [Camino vertical]
░░░░███░░           [Paredes a los lados]
████▓▓▓████  ───►   [Puerta en el centro]
░░░░███░░           [Camino continúa]
░░░░███░░           [Navegable hacia arriba]
```

## 🚦 Próximos Pasos

### **Mejoras posibles:**

1. **Curriculum Learning**: Entrenar progresivamente en objetivos más difíciles
2. **Reward Shaping con LLM**: Usar LLM local para objetivos dinámicos
3. **Multi-Objetivo**: Entrenar para múltiples badges simultáneamente
4. **Attention Mechanism**: Agregar attention sobre el mapa
5. **Recurrencia**: Usar LSTM/GRU para memoria temporal

### **Experimentar con arquitectura:**

```python
# En cnn_policy.py, puedes modificar:

# Más capas convolucionales
Conv2d(64, 128, kernel_size=3),
Conv2d(128, 256, kernel_size=3),

# Pooling para reducir dimensionalidad
MaxPool2d(2, 2),

# Residual connections
x = x + conv_block(x)
```

## 🐛 Troubleshooting

**Error: "No module named 'mgba'"**
```bash
pip install mgba-py
```

**Error: CUDA out of memory**
```python
# Reducir batch_size en train_ppo.py
batch_size=32  # En lugar de 64
```

**Entrenamiento muy lento**
```bash
# Verificar que usa GPU
python -c "import torch; print(torch.cuda.is_available())"

# Si no hay GPU, reducir n_steps
n_steps=512  # En lugar de 2048
```

**Agente se queda atascado**
```python
# Aumentar penalización por quedarse quieto
reward -= 0.5 * min(self.stationary_steps, 10)  # En _calculate_reward
```

## 📝 Notas

- El entorno usa `load_state()` para reset, no reinicia el juego completo
- Los checkpoints se guardan cada 10,000 steps por defecto
- El entrenamiento es determinista si usas `seed` en `reset()`
- La CNN procesa tiles en formato (H, W, C), PyTorch usa (C, H, W)

## 🤝 Contribuir

Mejoras sugeridas:
- [ ] Agregar más canales al mapa (NPC positions, items)
- [ ] Implementar HER (Hindsight Experience Replay)
- [ ] Multi-agent training con self-play
- [ ] Visualización en tiempo real del training

---

¡Buena suerte entrenando! 🚀🎮
