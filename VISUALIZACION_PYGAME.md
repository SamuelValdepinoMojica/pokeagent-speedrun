# 🎮 Visualización con Pygame Durante Entrenamiento

## Problema Resuelto

**Antes:** `--visualize` causaba **Segmentation Fault** después de ~800-3600 steps
- Usaba `PIL.Image.show()` que lanza procesos externos (xdg-open, display, etc.)
- Creaba cientos de archivos temporales PNG
- Intentaba abrir cientos de ventanas simultáneamente
- Sistema colapsaba por agotamiento de recursos

**Ahora:** Visualización estable con pygame sin crashes
- ✅ UNA ventana persistente de pygame
- ✅ Actualización eficiente sin procesos externos
- ✅ Control de FPS (30 FPS por defecto)
- ✅ Estadísticas en tiempo real (steps, episodes, rewards)
- ✅ Cierre graceful (cerrar ventana detiene entrenamiento)

## Uso

### Entrenamiento CON visualización (más lento, ~25-30 it/s)
```bash
python train_ppo.py --mode train --timesteps 100000 --visualize --n-envs 1 --state Emerald-GBAdvance/quick_start_save.state
```

### Entrenamiento SIN visualización (más rápido, ~35 it/s)
```bash
python train_ppo.py --mode train --timesteps 100000 --n-envs 1 --state Emerald-GBAdvance/quick_start_save.state
```

### Ver agente entrenado (después de completar entrenamiento)
```bash
python watch_trained_agent.py --model logs/checkpoints/ppo_pokemon_100000_steps.zip --steps 2000
```

## Implementación Técnica

### 1. Nuevo Callback: `PygameRenderCallback`
- Hereda de `stable_baselines3.common.callbacks.BaseCallback`
- Se ejecuta en cada step del entrenamiento
- Renderiza frame actual en ventana pygame
- Muestra estadísticas overlay (steps, episodes, rewards)
- Maneja eventos de pygame (cierre de ventana)

### 2. Modificación de `agent/drl_env.py`
**Antes:**
```python
def render(self):
    if self.render_mode == 'human':
        screenshot = self.emulator.get_screenshot()
        screenshot.show()  # ❌ Lanza procesos externos
```

**Después:**
```python
def render(self):
    if self.render_mode == 'human':
        screenshot = self.emulator.get_screenshot()
        return np.array(screenshot)  # ✅ Retorna array para pygame
```

### 3. Integración en `train_ppo.py`
- Import de pygame y numpy
- Clase `PygameRenderCallback` con rendering eficiente
- Callbacks condicionales (pygame solo si `--visualize`)
- Unwrapping correcto de Monitor wrapper

## Características de la Ventana Pygame

### Tamaño
- 720x480 pixels (3x escala del GBA original 240x160)

### Estadísticas Mostradas
- **Steps:** Total de pasos de entrenamiento
- **Episodes:** Episodios completados
- **Avg Reward:** Promedio de reward de últimos 10 episodios
- **Episode Length:** Longitud del episodio actual

### Control de FPS
- **30 FPS:** Balance entre fluidez y rendimiento
- Configurable en el código (parámetro `fps`)

### Cierre
- Cerrar ventana pygame → Detiene entrenamiento gracefully
- Guarda modelo con sufijo `_interrupted`

## Comparación de Rendimiento

| Modo | it/s | Tiempo 100k steps | Ventana | Estabilidad |
|------|------|-------------------|---------|-------------|
| Sin visualización | ~35 it/s | ~48 minutos | ❌ No | ✅ 100% estable |
| Con pygame (nuevo) | ~25-30 it/s | ~60-70 minutos | ✅ Sí | ✅ 100% estable |
| Con PIL.show() (viejo) | ~33 it/s | ❌ Crash en 800-3600 steps | ❌ Múltiples | ❌ CRASH |

## Diferencias: train_ppo.py vs watch_trained_agent.py

### `train_ppo.py --visualize` (NUEVO)
- **Propósito:** Ver entrenamiento en tiempo real
- **Framework:** Pygame con callback de Stable-Baselines3
- **Velocidad:** 25-30 it/s (un poco más lento)
- **Estadísticas:** Steps, episodes, avg reward
- **Uso:** Durante entrenamiento activo

### `watch_trained_agent.py`
- **Propósito:** Ver agente YA entrenado jugando
- **Framework:** Pygame standalone
- **Velocidad:** 60 FPS (más fluido)
- **Estadísticas:** Reward acumulado, acción actual
- **Uso:** Después de completar entrenamiento

## Verificación de Funcionamiento

### Test Exitoso (15000 steps)
```bash
python train_ppo.py --mode train --timesteps 15000 --visualize --n-envs 1 --state Emerald-GBAdvance/quick_start_save.state
```

**Resultado:**
- ✅ Exit Code: 0 (éxito)
- ✅ Sin Segmentation Fault
- ✅ Ventana pygame funcionando correctamente
- ✅ ~29-30 it/s estable
- ✅ Warnings de "Map buffer corruption" son normales (no causan crash)

## Notas Importantes

### ⚠️ `--visualize` solo funciona con `n_envs=1`
Si intentas `--visualize --n-envs 2`, automáticamente se cambia a `n_envs=1`.

### ⚠️ Map buffer corruption warnings
Los warnings de "Map buffer corruption" son **normales** y **NO causan crashes**:
- Aparecen cuando el juego cambia de mapa/ubicación
- El código se recupera automáticamente
- No afectan el entrenamiento

### ⚠️ Rendimiento
Visualización reduce velocidad ~15-20%:
- Sin visualización: 35 it/s
- Con visualización: 25-30 it/s
- Para entrenamiento largo (1M steps), usa sin visualización

## Recomendación de Uso

### Para Experimentación/Debug (cortos)
```bash
# Ver qué hace el agente (5k-20k steps)
python train_ppo.py --mode train --timesteps 10000 --visualize --n-envs 1 --state Emerald-GBAdvance/quick_start_save.state
```

### Para Entrenamiento Serio (largos)
```bash
# Sin visualización para máxima velocidad (100k-1M steps)
python train_ppo.py --mode train --timesteps 1000000 --n-envs 1 --state Emerald-GBAdvance/quick_start_save.state

# Luego ver con watch_trained_agent.py
python watch_trained_agent.py --model logs/checkpoints/ppo_pokemon_1000000_steps.zip --steps 5000
```

## Créditos

- Implementación basada en `watch_trained_agent.py` existente
- Adaptado para funcionar como callback de Stable-Baselines3
- Resuelve el problema de resource exhaustion de PIL.Image.show()
