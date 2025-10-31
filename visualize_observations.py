"""
Visualize what the RL agent actually sees
Shows the map tensor and vector features
"""

import numpy as np
import matplotlib.pyplot as plt
from agent.drl_env import PokemonEmeraldEnv

def visualize_observation(obs, title="Agent Observation"):
    """
    Visualize the observation dictionary that the agent receives
    """
    fig = plt.figure(figsize=(15, 5))
    
    # === 1. Visualize the 7x7x3 map tensor ===
    map_data = obs['map']
    
    # Plot each channel separately
    for channel in range(3):
        ax = plt.subplot(1, 4, channel + 1)
        
        # Show heatmap
        im = ax.imshow(map_data[:, :, channel], cmap='viridis', interpolation='nearest')
        
        # Add values as text
        for i in range(7):
            for j in range(7):
                text = ax.text(j, i, f'{map_data[i, j, channel]:.2f}',
                             ha="center", va="center", color="white", fontsize=8)
        
        if channel == 0:
            ax.set_title('Channel 0: Metatile ID')
        elif channel == 1:
            ax.set_title('Channel 1: Behavior')
        else:
            ax.set_title('Channel 2: Collision')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # === 2. Visualize the 18-element vector ===
    ax = plt.subplot(1, 4, 4)
    vector_data = obs['vector']
    
    # Feature names
    feature_names = [
        'Pos X', 'Pos Y',
        'P1 Lvl', 'P1 HP',
        'P2 Lvl', 'P2 HP',
        'P3 Lvl', 'P3 HP',
        'P4 Lvl', 'P4 HP',
        'P5 Lvl', 'P5 HP',
        'P6 Lvl', 'P6 HP',
        'Money', 'Badges', 'Battle', 'Dex'
    ]
    
    # Bar plot
    colors = ['red', 'red'] + ['blue']*12 + ['green']*4
    bars = ax.barh(range(18), vector_data, color=colors, alpha=0.7)
    
    ax.set_yticks(range(18))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel('Normalized Value')
    ax.set_title('Vector Features (18)')
    ax.set_xlim([-0.1, 1.1])
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, vector_data)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=7)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    """
    Create environment, get observation, and visualize it
    """
    print("🎮 Creating Pokemon Emerald environment...")
    env = PokemonEmeraldEnv(
        rom_path="Emerald-GBAdvance/rom.gba",
        initial_state_path="Emerald-GBAdvance/quick_start_save.state",
        render_mode=None
    )
    
    print("🔄 Resetting environment...")
    obs, info = env.reset()
    
    print("\n" + "="*60)
    print("📊 OBSERVATION STRUCTURE")
    print("="*60)
    print(f"Type: {type(obs)}")
    print(f"Keys: {obs.keys()}")
    print(f"\n📐 Map shape: {obs['map'].shape}")
    print(f"   - 7x7 spatial grid")
    print(f"   - 3 channels: [metatile_id, behavior, collision]")
    print(f"   - Range: [{obs['map'].min():.3f}, {obs['map'].max():.3f}]")
    
    print(f"\n📊 Vector shape: {obs['vector'].shape}")
    print(f"   - 18 features total:")
    print(f"   - Position (2): x={obs['vector'][0]:.3f}, y={obs['vector'][1]:.3f}")
    print(f"   - Party (12): 6 Pokemon × (level, hp_ratio)")
    print(f"   - Game (4): money, badges, battle, pokedex")
    print(f"   - Range: [{obs['vector'].min():.3f}, {obs['vector'].max():.3f}]")
    
    print(f"\n🎯 Info: {info}")
    print("="*60)
    
    # Visualize initial observation
    print("\n📈 Creating visualization...")
    fig = visualize_observation(obs, title="Initial Observation (Agent's View)")
    plt.savefig('agent_observation_initial.png', dpi=150, bbox_inches='tight')
    print("✅ Saved to: agent_observation_initial.png")
    
    # Take a few steps and visualize again
    print("\n🎮 Taking 5 random actions...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: action={action}, reward={reward:.3f}, location={info['location']}")
    
    # Visualize after movement
    fig = visualize_observation(obs, title="After 5 Steps (Agent's View)")
    plt.savefig('agent_observation_after_steps.png', dpi=150, bbox_inches='tight')
    print("✅ Saved to: agent_observation_after_steps.png")
    
    print("\n🖼️  Visualization complete! Check the PNG files.")
    print("\n💡 The agent sees:")
    print("   1. 7x7 map around player (3 channels)")
    print("   2. 18 numeric features (position, party, game state)")
    print("   - CNN processes the map")
    print("   - MLP processes the vector")
    print("   - Both are combined for decision making")
    
    env.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
