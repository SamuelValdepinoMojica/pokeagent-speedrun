#!/bin/bash
# Quick demo of parallel environments

echo "🚀 DEMO: Múltiples Ambientes Paralelos"
echo "======================================"
echo ""

# Check current setup
echo "📊 Sistema Actual:"
echo "  CPU Cores: $(nproc)"
echo "  RAM Total: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  RAM Libre: $(free -h | awk '/^Mem:/ {print $7}')"
echo ""

# Calculate recommended n_envs
cpu_cores=$(nproc)
recommended_envs=$((cpu_cores > 2 ? cpu_cores - 2 : 1))
echo "✅ Recomendación: --n-envs $recommended_envs"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EJEMPLOS DE USO:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "1️⃣  UN AMBIENTE (con visualización):"
echo "   python train_ppo.py --mode train --timesteps 10000 --n-envs 1 --visualize"
echo "   • Ver el agente jugando"
echo "   • Bueno para debugging"
echo "   • ~4 minutos para 10k steps"
echo ""

echo "2️⃣  CUATRO AMBIENTES (recomendado):"
echo "   python train_ppo.py --mode train --timesteps 100000 --n-envs 4"
echo "   • 4x más rápido"
echo "   • Balance RAM/velocidad"
echo "   • ~10 minutos para 100k steps"
echo ""

echo "3️⃣  OCHO AMBIENTES (máxima velocidad):"
echo "   python train_ppo.py --mode train --timesteps 1000000 --n-envs 8"
echo "   • 8x más rápido"
echo "   • Requiere 8+ GB RAM"
echo "   • ~50 minutos para 1M steps"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "COMPARACIÓN DE VELOCIDAD:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cat << 'TABLE'
┌─────────┬───────────┬─────────────┬──────────────────┐
│ n_envs  │ FPS Total │  Steps/sec  │  1M steps time   │
├─────────┼───────────┼─────────────┼──────────────────┤
│    1    │  239 FPS  │  ~40 s/s    │    ~7 horas      │
│    2    │  478 FPS  │  ~80 s/s    │    ~3.5 horas    │
│    4    │  956 FPS  │  ~160 s/s   │    ~1.7 horas    │
│    8    │ 1912 FPS  │  ~320 s/s   │    ~52 minutos   │
└─────────┴───────────┴─────────────┴──────────────────┘
TABLE

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST RÁPIDO:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Ask user if they want to test
read -p "¿Quieres probar con 1000 steps para comparar velocidades? (y/N): " test_choice

if [[ "$test_choice" =~ ^[Yy]$ ]]; then
    echo ""
    echo "🧪 Test 1: 1 ambiente"
    echo "Running: python train_ppo.py --mode train --timesteps 1000 --n-envs 1"
    echo ""
    time python train_ppo.py --mode train --timesteps 1000 --n-envs 1 2>&1 | grep -E "(FPS|steps/sec|Time|Timestep)"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    echo "🧪 Test 2: 4 ambientes"
    echo "Running: python train_ppo.py --mode train --timesteps 1000 --n-envs 4"
    echo ""
    time python train_ppo.py --mode train --timesteps 1000 --n-envs 4 2>&1 | grep -E "(FPS|steps/sec|Time|Timestep)"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Test completado!"
    echo ""
else
    echo ""
    echo "⏭️  Saltando test. Puedes ejecutarlo manualmente:"
    echo "   ./demo_parallel_envs.sh"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "COMANDO RECOMENDADO PARA TI:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "python train_ppo.py \\"
echo "    --mode train \\"
echo "    --timesteps 1000000 \\"
echo "    --n-envs $recommended_envs \\"
echo "    --frame-skip 6 \\"
echo "    --state Emerald-GBAdvance/quick_start_save.state"
echo ""
echo "Tiempo estimado: ~$(echo "scale=1; 100 / $recommended_envs" | bc) minutos para 100k steps"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 Más información:"
echo "   docs/parallel_envs_guide.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
