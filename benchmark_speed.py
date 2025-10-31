"""
Benchmark emulator speed to verify turbo mode is working
"""

import time
from agent.drl_env import PokemonEmeraldEnv

def benchmark_env(steps=1000, frame_skip=6):
    """
    Benchmark environment speed
    """
    print(f"🏁 Benchmarking environment speed...")
    print(f"   Steps: {steps}")
    print(f"   Frame skip: {frame_skip}")
    print(f"   Expected frames: {steps * frame_skip}")
    print()
    
    # Create environment
    env = PokemonEmeraldEnv(
        rom_path="Emerald-GBAdvance/rom.gba",
        initial_state_path="Emerald-GBAdvance/quick_start_save.state",
        render_mode=None,  # Headless for speed
        max_steps=steps + 100,
        frame_skip=frame_skip,
        turbo_mode=True
    )
    
    print("🔄 Resetting environment...")
    obs, info = env.reset()
    
    print(f"🎮 Running {steps} steps...")
    start_time = time.time()
    
    for i in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    env.close()
    
    # Calculate metrics
    steps_per_sec = steps / elapsed
    frames_per_sec = (steps * frame_skip) / elapsed
    
    print()
    print("=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Steps executed: {steps}")
    print(f"Frames executed: {steps * frame_skip}")
    print()
    print(f"⚡ Steps per second: {steps_per_sec:.2f}")
    print(f"⚡ Frames per second: {frames_per_sec:.2f}")
    print()
    print("🎯 Interpretation:")
    if frames_per_sec > 500:
        print("   ✅ EXCELLENT - Turbo mode working! (~600-1000 FPS)")
    elif frames_per_sec > 200:
        print("   ✅ GOOD - Fast but could be faster (~200-500 FPS)")
    elif frames_per_sec > 60:
        print("   ⚠️  SLOW - Faster than real-time but not turbo (~60-200 FPS)")
    else:
        print("   ❌ VERY SLOW - Running at real-time speed (~60 FPS)")
        print("      Problem: Turbo mode not working correctly")
    print("=" * 60)
    
    return steps_per_sec, frames_per_sec


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark emulator speed")
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps to run')
    parser.add_argument('--frame-skip', type=int, default=6, help='Frame skip value')
    
    args = parser.parse_args()
    
    benchmark_env(steps=args.steps, frame_skip=args.frame_skip)
