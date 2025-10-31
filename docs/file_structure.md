# 📂 Estructura de Archivos del Proyecto DRL

## 🌟 RESUMEN VISUAL

```
pokeagent-speedrun/
│
├── 🚀 ARCHIVOS ESENCIALES (para entrenar)
│   ├── agent/
│   │   ├── ⭐ lightweight_state_reader.py  [NUEVO] 30x speedup
│   │   ├── ⭐ drl_env.py                   [MODIFICADO] Usa lightweight reader
│   │   ├── __init__.py
│   │   ├── action.py
│   │   ├── perception.py
│   │   ├── planning.py
│   │   ├── memory.py
│   │   └── simple.py
│   │
│   ├── pokemon_env/                        [Sin cambios]
│   │   ├── emulator.py
│   │   ├── memory_reader.py
│   │   ├── emerald_utils.py
│   │   ├── enums.py
│   │   └── ...
│   │
│   ├── utils/                              [Sin cambios]
│   │   ├── map_formatter.py
│   │   ├── state_formatter.py
│   │   ├── checkpoint.py
│   │   └── ...
│   │
│   ├── Emerald-GBAdvance/                  [Sin cambios]
│   │   ├── rom.gba                         ⚠️ REQUERIDO
│   │   ├── quick_start_save.state          ⚠️ REQUERIDO
│   │   └── ...
│   │
│   ├── train_ppo.py                        [Sin cambios]
│   └── requirements.txt                    [Sin cambios]
│
├── 📊 HERRAMIENTAS DE ANÁLISIS (útiles pero opcionales)
│   ├── ⭐ benchmark_speed.py               [NUEVO] Medir FPS
│   ├── ⭐ visualize_observations.py        [NUEVO] Ver observaciones
│   ├── ⭐ watch_training.py                [NUEVO] Ver agente jugando
│   ├── ⭐ compare_state_data.py            [NUEVO] Comparar estados
│   └── ⭐ visualize_map_sizes.py           [NUEVO] Gráficas de mapas
│
├── 📝 DOCUMENTACIÓN
│   ├── docs/
│   │   ├── ⭐ state_comparison.md          [NUEVO] Explicación técnica
│   │   └── ⭐ sharing_guide.md             [NUEVO] Guía de archivos
│   ├── README.md
│   ├── ⭐ INSTALLATION.md                  [AUTO-GENERADO]
│   └── ⭐ CHANGELOG.md                     [AUTO-GENERADO]
│
├── 🛠️ SCRIPTS DE EMPAQUETADO
│   └── ⭐ create_package.sh                [NUEVO] Script automático
│
└── ❌ NO COMPARTIR
    ├── __pycache__/                        [Auto-generado]
    ├── .venv/                              [Cada uno crea el suyo]
    ├── llm_logs/                           [Logs viejos]
    ├── models/                             [Muy grande, opcional]
    └── .git/                               [Si compartes ZIP/TAR]
```

---

## 📋 LISTA DE ARCHIVOS POR CATEGORÍA

### ⚡ CORE (Indispensables)

| Archivo | Tamaño | Descripción | Cambios |
|---------|--------|-------------|---------|
| `agent/lightweight_state_reader.py` | ~7KB | Lector optimizado | ⭐ NUEVO |
| `agent/drl_env.py` | ~20KB | Environment DRL | ⭐ MODIFICADO |
| `train_ppo.py` | ~15KB | Script principal | Sin cambios |
| `Emerald-GBAdvance/rom.gba` | ~16MB | ROM del juego | ⚠️ REQUERIDO |
| `Emerald-GBAdvance/*.state` | ~500KB | Save states | ⚠️ REQUERIDO |

**Total Core:** ~17MB

---

### 📊 ANÁLISIS (Opcionales)

| Archivo | Tamaño | Propósito |
|---------|--------|-----------|
| `benchmark_speed.py` | ~5KB | Medir velocidad (FPS) |
| `visualize_observations.py` | ~8KB | Ver qué ve el agente |
| `watch_training.py` | ~10KB | Ver agente jugando |
| `compare_state_data.py` | ~12KB | Comparar estados |
| `visualize_map_sizes.py` | ~6KB | Gráficas de mapas |

**Total Análisis:** ~41KB

---

### 📚 DOCUMENTACIÓN (Útiles)

| Archivo | Tamaño | Contenido |
|---------|--------|-----------|
| `docs/state_comparison.md` | ~15KB | Explicación técnica detallada |
| `docs/sharing_guide.md` | ~10KB | Esta guía |
| `INSTALLATION.md` | ~5KB | Instrucciones de uso |
| `CHANGELOG.md` | ~4KB | Historial de cambios |

**Total Docs:** ~34KB

---

## 🎯 CASOS DE USO

### 1️⃣ Compañero quiere ENTRENAR solamente

**Archivos necesarios:**
```
✅ agent/lightweight_state_reader.py
✅ agent/drl_env.py
✅ agent/__init__.py (y resto de agent/)
✅ pokemon_env/ (completo)
✅ utils/ (completo)
✅ train_ppo.py
✅ requirements.txt
✅ Emerald-GBAdvance/rom.gba
✅ Emerald-GBAdvance/quick_start_save.state
```

**Comando:**
```bash
./create_package.sh minimal
```

**Tamaño:** ~20MB

---

### 2️⃣ Compañero quiere ANALIZAR y DEBUGGEAR

**Archivos necesarios:**
```
✅ Todo lo anterior +
✅ benchmark_speed.py
✅ visualize_observations.py
✅ watch_training.py
✅ compare_state_data.py
✅ visualize_map_sizes.py
✅ docs/state_comparison.md
✅ docs/sharing_guide.md
```

**Comando:**
```bash
./create_package.sh full
```

**Tamaño:** ~20.5MB

---

### 3️⃣ Compañero YA TIENE el proyecto base

**Solo necesita:**
```
✅ agent/lightweight_state_reader.py
✅ agent/drl_env.py (reemplazar)
✅ benchmark_speed.py
✅ visualize_observations.py
✅ watch_training.py
✅ compare_state_data.py
✅ docs/state_comparison.md
```

**Comando:**
```bash
zip drl_changes.zip \
    agent/lightweight_state_reader.py \
    agent/drl_env.py \
    benchmark_speed.py \
    visualize_observations.py \
    watch_training.py \
    compare_state_data.py \
    docs/state_comparison.md
```

**Tamaño:** ~60KB

---

## 🚀 COMANDOS RÁPIDOS

### Crear paquete mínimo
```bash
./create_package.sh minimal
```

### Crear paquete completo
```bash
./create_package.sh full
```

### Solo cambios (para actualizar)
```bash
zip -r drl_changes.zip \
    agent/lightweight_state_reader.py \
    agent/drl_env.py \
    benchmark_speed.py \
    visualize_observations.py \
    watch_training.py \
    compare_state_data.py \
    visualize_map_sizes.py \
    docs/
```

### Crear patch de Git
```bash
git diff main > drl_optimization.patch
```

---

## ✅ CHECKLIST ANTES DE COMPARTIR

- [ ] ROM incluido (`Emerald-GBAdvance/rom.gba`)
- [ ] Save state incluido (`quick_start_save.state`)
- [ ] Requirements actualizado
- [ ] No incluir `__pycache__/`
- [ ] No incluir `.venv/`
- [ ] No incluir modelos grandes (opcional)
- [ ] Incluir INSTALLATION.md
- [ ] Incluir CHANGELOG.md
- [ ] Probar que el paquete funciona:
  ```bash
  tar -xzf package.tar.gz
  cd pokeagent-speedrun-drl
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  python benchmark_speed.py --steps 100
  ```

---

## 📤 MÉTODOS DE COMPARTIR

### Opción 1: Git (Recomendado para equipo)
```bash
git checkout -b feature/drl-optimization
git add agent/lightweight_state_reader.py agent/drl_env.py
git commit -m "Add lightweight state reader (30x speedup)"
git push origin feature/drl-optimization
```

### Opción 2: Archivo comprimido
```bash
./create_package.sh full
# Enviar el archivo .tar.gz por email/drive/etc
```

### Opción 3: Google Drive / Dropbox
```bash
./create_package.sh full
# Subir a Drive y compartir link
```

### Opción 4: GitHub Release
```bash
./create_package.sh full
# Crear release en GitHub y adjuntar el .tar.gz
```

---

## 💡 TIPS FINALES

1. **Para compañeros técnicos:** Comparte solo los cambios (60KB)
2. **Para nuevos usuarios:** Comparte paquete completo (20MB)
3. **Para debugging:** Incluye todas las herramientas de análisis
4. **Para producción:** Solo archivos esenciales

---

## 🔗 LINKS ÚTILES

- Documentación técnica: `docs/state_comparison.md`
- Guía de instalación: `INSTALLATION.md`
- Historial de cambios: `CHANGELOG.md`
- Script de empaquetado: `create_package.sh`

---

**Última actualización:** 2025-10-28  
**Versión:** v1.0-lightweight  
**Mantenedor:** Samuel Valdespino
