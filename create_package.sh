#!/bin/bash
# Script para crear paquetes compartibles del proyecto DRL optimizado

set -e  # Exit on error

PROJECT_NAME="pokeagent-speedrun-drl"
VERSION="v1.0-lightweight"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=================================================="
echo "📦 Empaquetando Proyecto DRL Pokemon Emerald"
echo "=================================================="
echo ""

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para crear directorio temporal
create_temp_dir() {
    TEMP_DIR=$(mktemp -d)
    echo -e "${BLUE}📂 Directorio temporal: ${TEMP_DIR}${NC}"
    mkdir -p "${TEMP_DIR}/${PROJECT_NAME}"
}

# Función para copiar archivos esenciales
copy_essential_files() {
    echo ""
    echo -e "${GREEN}✓ Copiando archivos esenciales...${NC}"
    
    # Directorios completos
    cp -r agent "${TEMP_DIR}/${PROJECT_NAME}/"
    cp -r pokemon_env "${TEMP_DIR}/${PROJECT_NAME}/"
    cp -r utils "${TEMP_DIR}/${PROJECT_NAME}/"
    cp -r Emerald-GBAdvance "${TEMP_DIR}/${PROJECT_NAME}/"
    
    # Archivos de configuración
    cp requirements.txt "${TEMP_DIR}/${PROJECT_NAME}/"
    cp pyproject.toml "${TEMP_DIR}/${PROJECT_NAME}/" 2>/dev/null || true
    cp pytest.ini "${TEMP_DIR}/${PROJECT_NAME}/" 2>/dev/null || true
    
    # Scripts principales
    cp train_ppo.py "${TEMP_DIR}/${PROJECT_NAME}/"
    cp run.py "${TEMP_DIR}/${PROJECT_NAME}/" 2>/dev/null || true
    
    echo -e "${GREEN}✓ Archivos esenciales copiados${NC}"
}

# Función para copiar herramientas de análisis
copy_analysis_tools() {
    echo ""
    echo -e "${GREEN}✓ Copiando herramientas de análisis...${NC}"
    
    cp benchmark_speed.py "${TEMP_DIR}/${PROJECT_NAME}/" 2>/dev/null || true
    cp visualize_observations.py "${TEMP_DIR}/${PROJECT_NAME}/" 2>/dev/null || true
    cp watch_training.py "${TEMP_DIR}/${PROJECT_NAME}/" 2>/dev/null || true
    cp compare_state_data.py "${TEMP_DIR}/${PROJECT_NAME}/" 2>/dev/null || true
    cp visualize_map_sizes.py "${TEMP_DIR}/${PROJECT_NAME}/" 2>/dev/null || true
    
    echo -e "${GREEN}✓ Herramientas de análisis copiadas${NC}"
}

# Función para copiar documentación
copy_documentation() {
    echo ""
    echo -e "${GREEN}✓ Copiando documentación...${NC}"
    
    mkdir -p "${TEMP_DIR}/${PROJECT_NAME}/docs"
    cp docs/state_comparison.md "${TEMP_DIR}/${PROJECT_NAME}/docs/" 2>/dev/null || true
    cp docs/sharing_guide.md "${TEMP_DIR}/${PROJECT_NAME}/docs/" 2>/dev/null || true
    cp README.md "${TEMP_DIR}/${PROJECT_NAME}/" 2>/dev/null || true
    
    echo -e "${GREEN}✓ Documentación copiada${NC}"
}

# Función para limpiar archivos innecesarios
clean_unnecessary_files() {
    echo ""
    echo -e "${YELLOW}🧹 Limpiando archivos innecesarios...${NC}"
    
    # Eliminar __pycache__
    find "${TEMP_DIR}/${PROJECT_NAME}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    
    # Eliminar .pyc
    find "${TEMP_DIR}/${PROJECT_NAME}" -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Eliminar logs viejos
    rm -rf "${TEMP_DIR}/${PROJECT_NAME}/llm_logs" 2>/dev/null || true
    
    # Eliminar modelos entrenados (muy grandes)
    rm -rf "${TEMP_DIR}/${PROJECT_NAME}/models" 2>/dev/null || true
    
    echo -e "${GREEN}✓ Limpieza completada${NC}"
}

# Función para crear README de instalación
create_installation_readme() {
    echo ""
    echo -e "${GREEN}✓ Creando README de instalación...${NC}"
    
    cat > "${TEMP_DIR}/${PROJECT_NAME}/INSTALLATION.md" << 'EOF'
# 🚀 Instalación y Uso - DRL Pokemon Emerald

## 📋 Requisitos Previos

- Python 3.10 o superior
- 4GB RAM mínimo (8GB recomendado)
- GPU opcional (entrenamiento más rápido con CUDA)

## 🔧 Instalación

### 1. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Verificar instalación

```bash
python -c "import torch; import stable_baselines3; print('✓ Instalación correcta')"
```

## 🎮 Uso Rápido

### Entrenar agente (recomendado empezar aquí)

```bash
python train_ppo.py --mode train --timesteps 100000 --state Emerald-GBAdvance/quick_start_save.state
```

### Benchmark de velocidad

```bash
python benchmark_speed.py --steps 500 --frame-skip 6
```

**Resultado esperado:** ~200-300 FPS (con lightweight state reader)

### Visualizar observaciones del agente

```bash
python visualize_observations.py
```

### Ver agente jugando (aleatorio)

```bash
python watch_training.py --random --steps 1000
```

### Ver agente entrenado

```bash
python watch_training.py --model models/ppo_pokemon_100000_steps.zip
```

## 📊 Herramientas de Análisis

### Comparar estados (Comprehensive vs Lightweight)

```bash
python compare_state_data.py
```

### Visualizar tamaños de mapa

```bash
python visualize_map_sizes.py
```

## ⚙️ Configuración de Training

Editar parámetros en `train_ppo.py` o usar flags:

```bash
python train_ppo.py \
    --mode train \
    --timesteps 1000000 \
    --state Emerald-GBAdvance/quick_start_save.state \
    --n-envs 4 \
    --frame-skip 6
```

**Parámetros importantes:**
- `--timesteps`: Total de pasos de training (100k = ~30 min, 1M = ~5 horas)
- `--n-envs`: Entornos paralelos (4 = 4x throughput)
- `--frame-skip`: Frames por acción (6 = 10 decisiones/seg)

## 🐛 Troubleshooting

### Error: "No module named 'mgba'"

```bash
pip install mgba
```

### Error: "ROM not found"

Verifica que existe `Emerald-GBAdvance/rom.gba`

### Training muy lento (<50 FPS)

1. Verifica que estás usando `lightweight_state_reader`
2. Reduce `n_envs` si tienes poca RAM
3. Cierra otros programas

## 📚 Documentación

- `docs/state_comparison.md` - Diferencias entre estados
- `docs/sharing_guide.md` - Guía de archivos del proyecto
- `README.md` - Documentación general

## 🎯 Resultados Esperados

**Velocidad de training:**
- Con optimización: 200-300 FPS
- Sin optimización: 20-30 FPS

**Training time para 1M steps:**
- Con optimización: ~90 minutos
- Sin optimización: ~12 horas

## 💡 Tips

1. **Empieza con 100k steps** para probar que funciona
2. **Usa n_envs=4** para aprovechar CPU
3. **Monitor el log** para ver progreso
4. **Guarda modelos cada 100k steps** (automático)

## ❓ Soporte

Si encuentras problemas, revisa:
1. Logs en la terminal
2. `docs/state_comparison.md` para entender el sistema
3. Ejecuta `compare_state_data.py` para verificar lectura de estado

---

**Versión:** v1.0-lightweight  
**Optimización:** 30x speedup con lightweight state reader
EOF

    echo -e "${GREEN}✓ README de instalación creado${NC}"
}

# Función para crear changelog
create_changelog() {
    echo ""
    echo -e "${GREEN}✓ Creando CHANGELOG...${NC}"
    
    cat > "${TEMP_DIR}/${PROJECT_NAME}/CHANGELOG.md" << 'EOF'
# 📝 Changelog - DRL Optimization

## [v1.0-lightweight] - 2025-10-28

### ⚡ Optimizaciones Principales

#### Velocidad de Training: 30x Speedup
- **Antes:** 22 FPS (~12 horas para 1M steps)
- **Después:** 240 FPS (~90 minutos para 1M steps)

### ✨ Nuevos Archivos

#### `agent/lightweight_state_reader.py`
- Lector optimizado de estado del juego
- Lee solo información esencial para DRL
- Métodos:
  - `get_drl_state()` - Estado mínimo
  - `get_observation_for_drl()` - Observaciones (map 7x7x3 + vector 18)

#### `agent/drl_env.py` (MODIFICADO)
- Integra `LightweightStateReader`
- Nuevos métodos optimizados:
  - `_calculate_reward_from_lightweight()`
  - `_check_terminated_from_lightweight()`
- Reset y step optimizados

#### Herramientas de Análisis
- `benchmark_speed.py` - Medir FPS del environment
- `visualize_observations.py` - Ver observaciones del agente
- `watch_training.py` - Ver agente jugando
- `compare_state_data.py` - Comparar estados
- `visualize_map_sizes.py` - Visualizar mapas

#### Documentación
- `docs/state_comparison.md` - Comparación técnica detallada
- `docs/sharing_guide.md` - Guía de archivos
- `INSTALLATION.md` - Instrucciones de instalación
- `CHANGELOG.md` - Este archivo

### 🔄 Cambios en Observaciones

#### Antes (Comprehensive State):
- Map: 15x15 tiles (225 tiles)
- Read time: ~380ms
- Incluye: Dialog text, items, pokedex, etc.

#### Después (Lightweight State):
- Map: 7x7 tiles (49 tiles)
- Read time: ~12ms
- Incluye: Solo esencial (position, party, badges, map)

### 📊 Comparación de Información

#### ✅ Mantenido:
- Position (x, y)
- Party Pokemon (species, level, HP, status)
- Badges count
- In battle flag
- Map local (7x7 con 3 canales)

#### ❌ Removido (para velocidad):
- Location names (strings)
- Dialog text / OCR
- Items inventory
- Money
- Pokedex counts
- Full battle details
- Pokemon stats completos (moves, PP, etc.)

### 🎯 Impacto

**Para DRL Training:**
- ✅ Información suficiente para aprender navegación
- ✅ Velocidad permite training práctico
- ✅ Mantiene objetivos principales (badges, battles)

**Trade-offs:**
- ⚠️ Menos contexto estratégico
- ⚠️ No lee texto de NPCs
- ⚠️ Visión más corta (3 tiles vs 7 tiles)

### 🔧 Instalación

Ver `INSTALLATION.md` para instrucciones completas.

### 📚 Referencias

- Stable Baselines3: https://stable-baselines3.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/
- mGBA: https://mgba.io/

---

**Mantenedor:** Samuel Valdespino  
**Fecha:** October 28, 2025  
**Versión:** v1.0-lightweight
EOF

    echo -e "${GREEN}✓ CHANGELOG creado${NC}"
}

# Función principal para crear paquete
create_package() {
    local package_type=$1
    
    create_temp_dir
    copy_essential_files
    
    if [ "$package_type" == "full" ]; then
        copy_analysis_tools
        copy_documentation
    fi
    
    clean_unnecessary_files
    create_installation_readme
    create_changelog
    
    # Crear archivo comprimido
    local output_file="${PROJECT_NAME}_${VERSION}_${TIMESTAMP}.tar.gz"
    echo ""
    echo -e "${BLUE}📦 Comprimiendo paquete...${NC}"
    
    cd "${TEMP_DIR}"
    tar -czf "${output_file}" "${PROJECT_NAME}"
    cd - > /dev/null
    
    # Mover a directorio actual
    mv "${TEMP_DIR}/${output_file}" .
    
    # Limpiar
    rm -rf "${TEMP_DIR}"
    
    # Resultado
    local size=$(du -h "${output_file}" | cut -f1)
    echo ""
    echo -e "${GREEN}✅ Paquete creado exitosamente!${NC}"
    echo -e "${BLUE}📦 Archivo: ${output_file}${NC}"
    echo -e "${BLUE}💾 Tamaño: ${size}${NC}"
    echo ""
    echo "Para compartir:"
    echo "  1. Envía el archivo: ${output_file}"
    echo "  2. Instrucciones de uso en: INSTALLATION.md (dentro del paquete)"
    echo ""
}

# Menú principal
show_menu() {
    echo ""
    echo "Selecciona el tipo de paquete:"
    echo "  1) Mínimo - Solo archivos esenciales para training"
    echo "  2) Completo - Con herramientas de análisis y documentación"
    echo "  3) Salir"
    echo ""
    read -p "Opción [1-3]: " choice
    
    case $choice in
        1)
            echo ""
            echo "Creando paquete MÍNIMO..."
            create_package "minimal"
            ;;
        2)
            echo ""
            echo "Creando paquete COMPLETO..."
            create_package "full"
            ;;
        3)
            echo "Saliendo..."
            exit 0
            ;;
        *)
            echo "Opción inválida"
            show_menu
            ;;
    esac
}

# Ejecutar menú si no hay argumentos
if [ $# -eq 0 ]; then
    show_menu
else
    # Permitir especificar tipo como argumento
    case $1 in
        minimal|min)
            create_package "minimal"
            ;;
        full|complete)
            create_package "full"
            ;;
        *)
            echo "Uso: $0 [minimal|full]"
            exit 1
            ;;
    esac
fi
