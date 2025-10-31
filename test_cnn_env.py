"""
Test the CNN-based Pokemon DRL environment
"""

import logging
import numpy as np
from agent.drl_env import PokemonEmeraldEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_environment():
    """Test the environment with CNN observations"""
    print("=" * 60)
    print("🧪 Testing Pokemon Emerald CNN Environment")
    print("=" * 60)
    
    # Create environment
    print("\n1️⃣ Creating environment...")
    env = PokemonEmeraldEnv(
        rom_path="Emerald-GBAdvance/rom.gba",
        initial_state_path="Emerald-GBAdvance/start.state",
        max_steps=1000
    )
    
    print(f"✅ Environment created!")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Map shape: {env.observation_space['map'].shape}")
    print(f"   Vector shape: {env.observation_space['vector'].shape}")
    
    # Test reset
    print("\n2️⃣ Testing reset...")
    obs, info = env.reset()
    
    print(f"✅ Reset successful!")
    print(f"   Map observation shape: {obs['map'].shape}")
    print(f"   Map value range: [{obs['map'].min():.3f}, {obs['map'].max():.3f}]")
    print(f"   Vector observation shape: {obs['vector'].shape}")
    print(f"   Location: {info['location']}")
    print(f"   Badges: {info['badges']}")
    print(f"   Party size: {info['party_size']}")
    
    # Display map statistics
    print("\n   📊 Map channels:")
    for i, channel_name in enumerate(['Metatile ID', 'Behavior', 'Collision']):
        channel = obs['map'][:, :, i]
        print(f"      Channel {i} ({channel_name}):")
        print(f"         Range: [{channel.min():.3f}, {channel.max():.3f}]")
        print(f"         Mean: {channel.mean():.3f}")
        print(f"         Non-zero tiles: {np.count_nonzero(channel)}/49")
    
    # Display vector features
    print("\n   📊 Vector features:")
    feature_names = [
        "Player X", "Player Y",
        "P1 Level", "P1 HP%", "P2 Level", "P2 HP%",
        "P3 Level", "P3 HP%", "P4 Level", "P4 HP%",
        "P5 Level", "P5 HP%", "P6 Level", "P6 HP%",
        "Money", "Badges", "In Battle", "Pokedex"
    ]
    for i, (name, value) in enumerate(zip(feature_names, obs['vector'])):
        print(f"      {name:12s}: {value:.3f}")
    
    # Test random actions
    print("\n3️⃣ Testing random actions...")
    total_reward = 0
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        button_name = env._action_map[action].upper()
        print(f"   Step {step+1:2d}: Action={button_name:6s} | Reward={reward:6.2f} | "
              f"Location={info['location'][:20]:20s} | Total={total_reward:6.2f}")
        
        if terminated or truncated:
            print("   ⚠️ Episode ended!")
            break
    
    print(f"\n✅ Steps completed! Total reward: {total_reward:.2f}")
    
    # Test observation space
    print("\n4️⃣ Verifying observation space compliance...")
    assert env.observation_space['map'].contains(obs['map']), "❌ Map out of bounds!"
    assert env.observation_space['vector'].contains(obs['vector']), "❌ Vector out of bounds!"
    print("✅ All observations within valid ranges!")
    
    # Close environment
    print("\n5️⃣ Closing environment...")
    env.close()
    print("✅ Environment closed!")
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)


def visualize_map_channel(env, channel_idx=2):
    """Visualize a specific channel of the map"""
    obs, _ = env.reset()
    map_data = obs['map']
    
    print(f"\n📍 Map Channel {channel_idx} (7x7 grid):")
    print("   " + "-" * 29)
    
    channel = map_data[:, :, channel_idx]
    for i in range(7):
        row_str = "   | "
        for j in range(7):
            val = channel[i, j]
            if val == 0:
                row_str += "  "
            elif val < 0.5:
                row_str += "░░"
            else:
                row_str += "██"
            row_str += " "
        row_str += "|"
        print(row_str)
    
    print("   " + "-" * 29)
    print("   Legend: ░░ = passable, ██ = blocked")


if __name__ == "__main__":
    try:
        test_environment()
        
        # Optional: Visualize collision map
        print("\n🗺️  Visualizing collision map...")
        env = PokemonEmeraldEnv(
            rom_path="Emerald-GBAdvance/rom.gba",
            initial_state_path="Emerald-GBAdvance/start.state"
        )
        visualize_map_channel(env, channel_idx=2)
        env.close()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
