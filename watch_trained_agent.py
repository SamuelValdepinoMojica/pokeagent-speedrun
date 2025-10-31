#!/usr/bin/env python3
"""
Watch the trained agent play in real-time
Shows visual game window while the agent plays
"""

import argparse
import logging
import time
import pygame
import numpy as np
from stable_baselines3 import PPO
from agent.drl_env import PokemonEmeraldEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def watch_agent(
    model_path: str = None,
    state_path: str = "Emerald-GBAdvance/quick_start_save.state",
    steps: int = 10000,
    random: bool = False
):
    """
    Watch an agent play Pokemon Emerald
    
    Args:
        model_path: Path to trained model (None = random actions)
        state_path: Initial game state
        steps: Number of steps to watch
        random: Use random actions instead of model
    """
    
    print("=" * 70)
    if random or model_path is None:
        print("🎮 WATCHING RANDOM AGENT")
        print("   (Taking random actions - no learning)")
    else:
        print(f"🤖 WATCHING TRAINED AGENT")
        print(f"   Model: {model_path}")
    print("=" * 70)
    print()
    
    # Initialize pygame for visualization
    pygame.init()
    screen = pygame.display.set_mode((240 * 3, 160 * 3))  # 3x scale
    pygame.display.set_caption("Pokemon Emerald - Agent Playing")
    clock = pygame.time.Clock()
    
    # Create environment with visualization
    print("🎮 Creating environment with visualization...")
    env = PokemonEmeraldEnv(
        rom_path="Emerald-GBAdvance/rom.gba",
        initial_state_path=state_path,
        render_mode='human',
        max_steps=100000,
        frame_skip=36
    )
    
    # Load model if provided
    model = None
    if not random and model_path:
        print(f"📦 Loading model from: {model_path}")
        try:
            model = PPO.load(model_path, env=env)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Falling back to random actions")
            model = None
    
    print()
    print("▶️  Starting gameplay...")
    print("   Close the pygame window or press Ctrl+C to stop")
    print()
    print("📊 Stats Display:")
    print("   • Reward: Cumulative reward")
    print("   • Steps: Steps taken")
    print("   • Episodes: Episodes completed")
    print("=" * 70)
    print()
    
    obs, info = env.reset()
    total_reward = 0
    episode_count = 0
    step_count = 0
    episode_reward = 0
    
    # Font for stats
    font = pygame.font.Font(None, 36)
    
    try:
        for step in range(steps):
            # Check for pygame events (window close)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n👋 Window closed by user")
                    return
            
            # Get action
            if model is not None:
                action, _states = model.predict(obs, deterministic=False)
                # Convert numpy array to integer
                action = int(action)
            else:
                action = env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_reward += reward
            step_count += 1
            
            # Get screenshot and display
            screenshot = env.emulator.get_screenshot()
            if screenshot:
                # Convert PIL to pygame surface
                screenshot_array = np.array(screenshot)
                # Scale up 3x
                screenshot_array = np.repeat(np.repeat(screenshot_array, 3, axis=0), 3, axis=1)
                surface = pygame.surfarray.make_surface(screenshot_array.swapaxes(0, 1))
                screen.blit(surface, (0, 0))
                
                # Draw stats overlay
                stats_text = [
                    f"Steps: {step_count}",
                    f"Episodes: {episode_count}",
                    f"Episode Reward: {episode_reward:.1f}",
                    f"Total Reward: {total_reward:.1f}",
                    f"Action: {env._action_map[int(action)]}"
                ]
                
                y_offset = 10
                for text in stats_text:
                    # Background for text
                    text_surface = font.render(text, True, (255, 255, 255))
                    bg_rect = text_surface.get_rect()
                    bg_rect.topleft = (10, y_offset)
                    bg_rect.inflate_ip(10, 5)
                    pygame.draw.rect(screen, (0, 0, 0), bg_rect)
                    # Text
                    screen.blit(text_surface, (10, y_offset))
                    y_offset += 35
                
                pygame.display.flip()
                clock.tick(60)  # 60 FPS display
            
            # Reset if episode ended
            if terminated or truncated:
                episode_count += 1
                print(f"📋 Episode {episode_count} finished:")
                print(f"   Steps: {step_count}")
                print(f"   Reward: {episode_reward:.2f}")
                print(f"   Reason: {'Terminated' if terminated else 'Truncated'}")
                print()
                
                obs, info = env.reset()
                episode_reward = 0
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"⏱️  Step {step}/{steps} | Reward: {total_reward:.2f} | Episodes: {episode_count}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    
    finally:
        print()
        print("=" * 70)
        print("📊 FINAL STATISTICS")
        print("=" * 70)
        print(f"Total steps: {step_count}")
        print(f"Total episodes: {episode_count}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average reward per episode: {total_reward / max(episode_count, 1):.2f}")
        print()
        
        env.close()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained agent play Pokemon Emerald")
    parser.add_argument("--model", type=str, help="Path to trained model (.zip)")
    parser.add_argument("--state", type=str, default="Emerald-GBAdvance/quick_start_save.state",
                       help="Initial game state")
    parser.add_argument("--steps", type=int, default=10000, help="Number of steps to watch")
    parser.add_argument("--random", action="store_true", help="Use random actions instead of model")
    
    args = parser.parse_args()
    
    watch_agent(
        model_path=args.model,
        state_path=args.state,
        steps=args.steps,
        random=args.random
    )
