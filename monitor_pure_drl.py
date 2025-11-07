"""
Monitor Pure DRL Training Progress
Muestra estadísticas en tiempo real sin reward shaping direccional
"""

import time
import os
from pathlib import Path

def get_latest_log_dir():
    """Encuentra el directorio de logs más reciente"""
    tensorboard_dir = Path("./tensorboard_logs")
    if not tensorboard_dir.exists():
        return None
    
    # Encontrar el PPO_* más reciente
    subdirs = [d for d in tensorboard_dir.iterdir() if d.is_dir() and d.name.startswith("PPO_")]
    if not subdirs:
        return None
    
    latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest

def parse_monitor_file(filepath):
    """Lee el archivo monitor.csv y extrae estadísticas"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 3:  # Header + metadata + at least 1 episode
            return None
        
        # Última línea es el episodio más reciente
        last_episode = lines[-1].strip().split(',')
        if len(last_episode) < 3:
            return None
        
        reward = float(last_episode[0])
        length = int(last_episode[1])
        time_elapsed = float(last_episode[2])
        
        # Calcular promedio de últimos 10 episodios
        recent_rewards = []
        for line in lines[-11:-1]:  # Últimos 10 episodios (sin contar el actual)
            try:
                parts = line.strip().split(',')
                if len(parts) >= 1:
                    recent_rewards.append(float(parts[0]))
            except:
                pass
        
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else reward
        
        return {
            'last_reward': reward,
            'avg_reward_10': avg_reward,
            'last_length': length,
            'total_episodes': len(lines) - 2,
            'time_elapsed': time_elapsed
        }
    except Exception as e:
        print(f"Error parsing monitor file: {e}")
        return None

def main():
    print("=" * 80)
    print("🧪 MONITORING PURE DRL TRAINING (No Directional Reward Shaping)")
    print("=" * 80)
    print("\nEste entrenamiento usa:")
    print("  ✅ PPO (Deep Reinforcement Learning)")
    print("  ✅ LLM Reward Shaping (milestone detection)")
    print("  ❌ Directional Reward Shaping (DESACTIVADO para prueba)")
    print("\nObjetivo: Ver hasta dónde aprende el agente sin guía direccional\n")
    print("-" * 80)
    
    while True:
        log_dir = get_latest_log_dir()
        if not log_dir:
            print("⏳ Esperando que inicie el entrenamiento...")
            time.sleep(5)
            continue
        
        # Buscar archivos monitor_*.csv
        monitor_files = list(Path("./logs").glob("monitor_*.csv"))
        
        if not monitor_files:
            print("⏳ Esperando archivos de monitoreo...")
            time.sleep(5)
            continue
        
        os.system('clear')
        print("=" * 80)
        print(f"📊 PURE DRL TRAINING - {log_dir.name}")
        print("=" * 80)
        print()
        
        total_episodes = 0
        total_reward = 0
        best_reward = float('-inf')
        
        # Leer todos los environments
        for monitor_file in sorted(monitor_files):
            env_id = monitor_file.stem.split('_')[-1]
            stats = parse_monitor_file(monitor_file)
            
            if stats:
                total_episodes += stats['total_episodes']
                total_reward += stats['last_reward']
                best_reward = max(best_reward, stats['last_reward'])
                
                print(f"🎮 Environment {env_id}:")
                print(f"   Episodes: {stats['total_episodes']}")
                print(f"   Last Reward: {stats['last_reward']:.1f}")
                print(f"   Avg (10 ep): {stats['avg_reward_10']:.1f}")
                print(f"   Last Length: {stats['last_length']} steps")
                print()
        
        print("-" * 80)
        print(f"📈 OVERALL:")
        print(f"   Total Episodes: {total_episodes}")
        print(f"   Best Reward: {best_reward:.1f}")
        print(f"   Avg Recent Reward: {total_reward / len(monitor_files):.1f}")
        print()
        print("💡 TIP: Si los rewards no mejoran mucho, puede ser señal de que")
        print("       el agente necesita guía (como reward shaping direccional)")
        print()
        print("🔄 Actualizando en 10 segundos... (Ctrl+C para detener)")
        
        time.sleep(10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ Monitoreo detenido")
