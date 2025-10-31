#!/usr/bin/env python3
"""
Monitor training progress in real-time using TensorBoard
Shows rewards, episode length, and other metrics while training
"""

import os
import subprocess
import sys
import time

def main():
    print("=" * 60)
    print("📊 MONITOREO DE ENTRENAMIENTO EN TIEMPO REAL")
    print("=" * 60)
    print()
    
    # Check if tensorboard logs exist
    log_dir = "./tensorboard_logs"
    if not os.path.exists(log_dir):
        print(f"❌ No se encontró el directorio: {log_dir}")
        print("   Primero inicia el entrenamiento con:")
        print("   python train_ppo.py --mode train --timesteps 1000000 --n-envs 4")
        return
    
    print("✅ Directorio de logs encontrado")
    print()
    print("🚀 Iniciando TensorBoard...")
    print()
    print("Una vez que inicie TensorBoard:")
    print("  1. Abre tu navegador en: http://localhost:6006")
    print("  2. Verás gráficas de:")
    print("     • Reward promedio por episodio")
    print("     • Longitud de episodios")
    print("     • Loss de la política")
    print("     • Value loss")
    print("     • Entropy")
    print()
    print("💡 Para detener TensorBoard: Ctrl+C")
    print("=" * 60)
    print()
    
    try:
        # Start tensorboard
        subprocess.run([
            sys.executable, "-m", "tensorboard.main",
            "--logdir", log_dir,
            "--port", "6006",
            "--bind_all"
        ])
    except KeyboardInterrupt:
        print("\n👋 TensorBoard detenido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print()
        print("Instala TensorBoard con:")
        print("  pip install tensorboard")

if __name__ == "__main__":
    main()
