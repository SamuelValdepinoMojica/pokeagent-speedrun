# 🗑️ Archivos que Puedes Borrar de Forma Segura

## ✅ ARCHIVOS CREADOS PARA PRUEBA/ANÁLISIS (Puedes borrar)

### 📊 Scripts de Visualización/Análisis (OPCIONALES - para debugging)
```bash
# Estos son útiles pero no necesarios para training
benchmark_speed.py                    # ✅ BORRABLE - Medir velocidad
compare_state_data.py                 # ✅ BORRABLE - Comparar estados
visualize_observations.py             # ✅ BORRABLE - Ver observaciones
visualize_map_sizes.py                # ✅ BORRABLE - Gráficas de mapas
watch_training.py                     # ✅ BORRABLE - Ver agente jugando
visualize_agent.py                    # ✅ BORRABLE - Visualización antigua
visualize_fast.py                     # ✅ BORRABLE - Visualización antigua
visualize_pygame.py                   # ✅ BORRABLE - Visualización antigua
visualize_simple.py                   # ✅ BORRABLE - Visualización antigua
```

**Total:** ~9 archivos (50KB)

### 🖼️ Imágenes Generadas (TEMPORALES)
```bash
agent_observation_initial.png         # ✅ BORRABLE - Output de visualize_observations.py
agent_observation_after_steps.png     # ✅ BORRABLE - Output de visualize_observations.py
map_size_comparison.png               # ✅ BORRABLE - Output de visualize_map_sizes.py
emerald.png                           # ✅ BORRABLE - Screenshot de prueba
```

**Total:** 4 imágenes (~500KB)

### 📦 Archivos Comprimidos (DUPLICADOS)
```bash
mGBA-0.10.5-ubuntu64-focal.tar.xz     # ✅ BORRABLE - Ya extraído
mGBA-0.10.5-ubuntu64-focal.tar.xz.1   # ✅ BORRABLE - Descarga duplicada
mGBA-0.10.5-ubuntu64-focal.tar.xz.2   # ✅ BORRABLE - Descarga duplicada
```

**Total:** 3 archivos (~300MB - ¡mucho espacio!)

### 🧪 Scripts de Prueba/Desarrollo Antiguos
```bash
test_cnn_env.py                       # ✅ BORRABLE - Test viejo
train_drl.py                          # ✅ BORRABLE - Script antiguo (usa train_ppo.py)
grab_map.py                           # ✅ BORRABLE - Utilidad de desarrollo
manual.py                             # ✅ BORRABLE - Script manual de prueba
```

**Total:** 4 archivos (~20KB)

### 📄 Documentación Generada (OPCIONAL)
```bash
create_package.sh                     # ⚠️ ÚTIL - Pero puedes borrar si no vas a compartir
docs/sharing_guide.md                 # ⚠️ ÚTIL - Documentación
docs/file_structure.md                # ⚠️ ÚTIL - Documentación
docs/state_comparison.md              # ⚠️ ÚTIL - Documentación técnica
```

---

## ❌ ARCHIVOS QUE **NO DEBES BORRAR**

### 🔴 CRÍTICOS (El proyecto no funciona sin estos)
```bash
agent/lightweight_state_reader.py     # ❌ NO BORRAR - Optimización clave
agent/drl_env.py                      # ❌ NO BORRAR - Environment DRL
agent/ (resto)                        # ❌ NO BORRAR - Código del agente
pokemon_env/                          # ❌ NO BORRAR - Emulador
utils/                                # ❌ NO BORRAR - Utilidades
train_ppo.py                          # ❌ NO BORRAR - Script principal de training
run.py                                # ❌ NO BORRAR - Runner principal
requirements.txt                      # ❌ NO BORRAR - Dependencias
Emerald-GBAdvance/rom.gba            # ❌ NO BORRAR - Juego
Emerald-GBAdvance/*.state            # ❌ NO BORRAR - Save states
```

### 🟡 ÚTILES (No críticos pero recomendados mantener)
```bash
tests/                                # 🟡 MANTENER - Tests del proyecto
server/                               # 🟡 MANTENER - Si usas el servidor web
README.md                             # 🟡 MANTENER - Documentación principal
TRAINING_CNN.md                       # 🟡 MANTENER - Guía de CNN training
.gitignore                            # 🟡 MANTENER - Config de Git
```

### 🔵 GENERADOS (Puedes borrar pero se regeneran)
```bash
logs/                                 # 🔵 OPCIONAL - Logs de training (se regeneran)
tensorboard_logs/                     # 🔵 OPCIONAL - Logs de TensorBoard (se regeneran)
models/                               # 🔵 MANTENER - Modelos entrenados (valiosos!)
.pokeagent_cache/                     # 🔵 OPCIONAL - Cache (se regenera)
training.log                          # 🔵 OPCIONAL - Log (se regenera)
submission.log                        # 🔵 MANTENER - Para submission del concurso
```

---

## 🚀 COMANDO PARA LIMPIAR TODO LO BORRABLE

```bash
#!/bin/bash
# Ejecuta este comando para borrar todos los archivos de prueba de forma segura

cd /home/samuel-valdespino/pokeagent-speedrun

# Visualización/Análisis opcionales
rm -f benchmark_speed.py
rm -f compare_state_data.py
rm -f visualize_observations.py
rm -f visualize_map_sizes.py
rm -f watch_training.py
rm -f visualize_agent.py
rm -f visualize_fast.py
rm -f visualize_pygame.py
rm -f visualize_simple.py

# Imágenes generadas
rm -f agent_observation_initial.png
rm -f agent_observation_after_steps.png
rm -f map_size_comparison.png
rm -f emerald.png

# Archivos comprimidos duplicados (¡libera 300MB!)
rm -f mGBA-0.10.5-ubuntu64-focal.tar.xz
rm -f mGBA-0.10.5-ubuntu64-focal.tar.xz.1
rm -f mGBA-0.10.5-ubuntu64-focal.tar.xz.2

# Scripts de prueba antiguos
rm -f test_cnn_env.py
rm -f train_drl.py
rm -f grab_map.py
rm -f manual.py

# Opcional: Documentación generada (si no la necesitas)
# rm -f create_package.sh
# rm -rf docs/

echo "✅ Limpieza completada!"
echo "Espacio liberado: ~300MB"
echo ""
echo "Archivos esenciales conservados:"
echo "  ✓ agent/lightweight_state_reader.py"
echo "  ✓ agent/drl_env.py"
echo "  ✓ train_ppo.py"
echo "  ✓ pokemon_env/"
echo "  ✓ utils/"
echo "  ✓ Emerald-GBAdvance/"
```

---

## 📋 RESUMEN POR CATEGORÍA

### ✅ BORRAR (Seguros - ~300MB)
1. **Comprimidos duplicados:** `mGBA-*.tar.xz*` → **~300MB**
2. **Scripts de análisis:** 9 archivos → **~50KB**
3. **Imágenes generadas:** 4 archivos → **~500KB**
4. **Scripts antiguos:** 4 archivos → **~20KB**

### ⚠️ OPCIONAL (Útiles para debugging)
1. **Documentación:** `docs/`, `create_package.sh`
2. **Logs antiguos:** `logs/`, `training.log`

### ❌ NO BORRAR (Esenciales)
1. **Agent:** `agent/lightweight_state_reader.py`, `agent/drl_env.py`, etc.
2. **Environment:** `pokemon_env/`, `utils/`
3. **Training:** `train_ppo.py`, `run.py`
4. **Assets:** `Emerald-GBAdvance/rom.gba`, `*.state`
5. **Modelos:** `models/` (si tienes modelos entrenados)

---

## 💡 MI RECOMENDACIÓN

**Ejecuta este comando conservador (solo borra lo más seguro):**

```bash
cd /home/samuel-valdespino/pokeagent-speedrun

# Solo borra comprimidos duplicados (libera 300MB)
rm -f mGBA-0.10.5-ubuntu64-focal.tar.xz*

# Y scripts de visualización antiguos
rm -f visualize_agent.py visualize_fast.py visualize_pygame.py visualize_simple.py

# Y archivos de prueba antiguos
rm -f test_cnn_env.py train_drl.py grab_map.py manual.py

echo "✅ Limpieza básica completada (~300MB liberados)"
```

**Luego decide si quieres:**
- Mantener `benchmark_speed.py`, `visualize_observations.py`, `watch_training.py` (útiles para debugging)
- Mantener `docs/` (documentación que acabamos de crear)
- Borrar logs antiguos si no los necesitas

---

**¿Quieres que:**
1. **Ejecute el comando de limpieza básica?** (solo borra lo 100% seguro)
2. **Ejecute limpieza completa?** (borra todo lo opcional)
3. **Te ayude a decidir caso por caso?**
