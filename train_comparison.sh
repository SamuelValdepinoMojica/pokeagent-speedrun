#!/bin/bash

# Script para entrenar y comparar 3 modos de entrenamiento:
# 1. DRL Puro (sin reward shaping)
# 2. DRL + Milestones (rule-based)
# 3. DRL + LLM + Diálogos

# Configuración común
ROM="Emerald-GBAdvance/rom.gba"
STATE="Emerald-GBAdvance/quick_start_save.state"
TIMESTEPS=100000
N_ENVS=4
SAVE_FREQ=10000

echo "=========================================="
echo "🎮 Pokemon Emerald - Training Comparison"
echo "=========================================="
echo ""
echo "This script will train 3 models:"
echo "  1️⃣  Pure DRL (no reward shaping)"
echo "  2️⃣  DRL + Rule-based milestones"
echo "  3️⃣  DRL + LLM + Dialogues"
echo ""
echo "Each model will train for ${TIMESTEPS} timesteps"
echo "Using ${N_ENVS} parallel environments"
echo ""
echo "=========================================="
echo ""

# Preguntar qué modelos entrenar
read -p "Train Pure DRL? (y/n) [y]: " train_pure
train_pure=${train_pure:-y}

read -p "Train Rule-based? (y/n) [y]: " train_rules
train_rules=${train_rules:-y}

read -p "Train LLM-based? (y/n) [n]: " train_llm
train_llm=${train_llm:-n}

echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

# Modelo 1: DRL Puro
if [ "$train_pure" = "y" ]; then
    echo ""
    echo "🔵 [1/3] Training Pure DRL Model..."
    echo "Model will be saved to: ./models/ppo_pure_drl"
    echo "Logs: ./logs_pure_drl/"
    echo "Tensorboard: ./tensorboard_logs/PPO_pure_drl/"
    echo ""
    
    python train_ppo.py \
        --mode train \
        --rom "$ROM" \
        --state "$STATE" \
        --timesteps "$TIMESTEPS" \
        --n-envs "$N_ENVS" \
        --save-freq "$SAVE_FREQ" \
        --model-path "./models/ppo_pure_drl" \
        --pure-drl \
        2>&1 | tee training_pure_drl.log
    
    echo ""
    echo "✅ Pure DRL training complete!"
    echo ""
fi

# Modelo 2: DRL + Rule-based milestones
if [ "$train_rules" = "y" ]; then
    echo ""
    echo "📊 [2/3] Training Rule-based Milestone Model..."
    echo "Model will be saved to: ./models/ppo_rule_based"
    echo "Logs: ./logs_rule_based/"
    echo "Tensorboard: ./tensorboard_logs/PPO_rule_based/"
    echo ""
    
    python train_ppo.py \
        --mode train \
        --rom "$ROM" \
        --state "$STATE" \
        --timesteps "$TIMESTEPS" \
        --n-envs "$N_ENVS" \
        --save-freq "$SAVE_FREQ" \
        --model-path "./models/ppo_rule_based" \
        2>&1 | tee training_rule_based.log
    
    echo ""
    echo "✅ Rule-based training complete!"
    echo ""
fi

# Modelo 3: DRL + LLM + Diálogos
if [ "$train_llm" = "y" ]; then
    echo ""
    echo "🤖 [3/3] Training LLM + Dialogue Model..."
    echo "⚠️  Make sure Ollama is running: ollama serve"
    echo "Model will be saved to: ./models/ppo_llm_dialogue"
    echo "Logs: ./logs_llm_dialogue/"
    echo "Tensorboard: ./tensorboard_logs/PPO_llm_dialogue/"
    echo ""
    
    # Verificar que Ollama esté corriendo
    if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        echo "❌ ERROR: Ollama is not running!"
        echo "Please start Ollama first: ollama serve"
        echo "Skipping LLM training..."
    else
        echo "✅ Ollama detected, starting training..."
        
        python train_ppo.py \
            --mode train \
            --rom "$ROM" \
            --state "$STATE" \
            --timesteps "$TIMESTEPS" \
            --n-envs "$N_ENVS" \
            --save-freq "$SAVE_FREQ" \
            --model-path "./models/ppo_llm_dialogue" \
            --use-llm \
            2>&1 | tee training_llm_dialogue.log
        
        echo ""
        echo "✅ LLM-based training complete!"
        echo ""
    fi
fi

echo ""
echo "=========================================="
echo "🎉 All selected trainings complete!"
echo "=========================================="
echo ""
echo "To view results with TensorBoard:"
echo "  tensorboard --logdir ./tensorboard_logs"
echo ""
echo "To compare models:"
echo "  python compare_models.py"
echo ""
echo "Model locations:"
if [ "$train_pure" = "y" ]; then
    echo "  🔵 Pure DRL:        ./models/ppo_pure_drl.zip"
fi
if [ "$train_rules" = "y" ]; then
    echo "  📊 Rule-based:      ./models/ppo_rule_based.zip"
fi
if [ "$train_llm" = "y" ]; then
    echo "  🤖 LLM+Dialogue:    ./models/ppo_llm_dialogue.zip"
fi
echo ""
