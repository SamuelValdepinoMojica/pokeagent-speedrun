"""
Train a PPO agent to play Pokemon Emerald with CNN map processing
"""

import os
import logging
import pygame
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

from agent.drl_env import PokemonEmeraldEnv
from agent.cnn_policy import PokemonCNNPolicy
from agent.llm_reward_callback import LLMRewardCallback

# Setup logging
# Use WARNING during training to reduce terminal spam
# Change to INFO for debugging: level=logging.INFO
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PygameRenderCallback(BaseCallback):
    """
    Callback for rendering the environment with pygame during training.
    Uses pygame to display the game screen efficiently without spawning external processes.
    """
    
    def __init__(self, env, render_freq: int = 1, fps: int = 30, verbose: int = 0):
        """
        Args:
            env: The vectorized environment
            render_freq: Render every N steps (1 = every step)
            fps: Target FPS for display (controls rendering speed)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.env = env
        self.render_freq = render_freq
        self.fps = fps
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((240 * 3, 160 * 3))  # 3x scale
        pygame.display.set_caption("Pokemon Emerald - PPO Training")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        
        # Stats tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        """
        Called after each step. Renders the game screen.
        Returns:
            True to continue training, False to stop
        """
        # IMPORTANT: Always pump events to keep window responsive
        # This prevents OS from thinking the window is frozen
        pygame.event.pump()
        
        # Check for pygame events (window close)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info("\n👋 Window closed by user - stopping training")
                return False  # Stop training
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    logger.info("\n⏸️  ESC pressed - stopping training")
                    return False
        
        # Only render every N steps to avoid slowdown
        if self.num_timesteps % self.render_freq == 0:
            try:
                # Get the base environment (unwrap VecMonitor and DummyVecEnv)
                base_env = self.env.envs[0]
                
                # Unwrap Monitor wrapper to get the actual PokemonEmeraldEnv
                if hasattr(base_env, 'env'):
                    base_env = base_env.env  # Unwrap Monitor
                
                # Get screenshot from emulator (skip if emulator is busy)
                if not hasattr(base_env, 'emulator') or not base_env.emulator.core:
                    return True  # Continue training even if we can't render
                
                screenshot = base_env.emulator.get_screenshot()
                
                if screenshot:
                    # Convert PIL to numpy array
                    screenshot_array = np.array(screenshot)
                    
                    # Scale up 3x
                    screenshot_array = np.repeat(np.repeat(screenshot_array, 3, axis=0), 3, axis=1)
                    
                    # Convert to pygame surface
                    surface = pygame.surfarray.make_surface(screenshot_array.swapaxes(0, 1))
                    self.screen.blit(surface, (0, 0))
                    
                    # Draw stats overlay
                    stats_text = [
                        f"Steps: {self.num_timesteps:,}",
                        f"Episodes: {len(self.episode_rewards)}",
                        f"Avg Reward: {np.mean(self.episode_rewards[-10:]):.1f}" if self.episode_rewards else "Avg Reward: N/A",
                        f"Episode Length: {self.current_episode_length}",
                    ]
                    
                    y_offset = 10
                    for text in stats_text:
                        # Background for text
                        text_surface = self.font.render(text, True, (255, 255, 255))
                        bg_rect = text_surface.get_rect()
                        bg_rect.topleft = (10, y_offset)
                        bg_rect.inflate_ip(10, 5)
                        pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
                        # Text
                        self.screen.blit(text_surface, (10, y_offset))
                        y_offset += 30
                    
                    pygame.display.flip()
                    
                    # Don't use clock.tick() - it can cause hangs during checkpoints
                    # Instead, just update display as fast as possible
                    # pygame.event.pump() at the start keeps window responsive
                    
            except Exception as e:
                # Don't stop training on render errors
                if self.verbose > 0:
                    logger.debug(f"Render skipped: {e}")
                pass  # Continue training
        
        # Track episode stats
        self.current_episode_length += 1
        
        # Check if episode ended (for stats tracking)
        if hasattr(self.locals, 'dones') and self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        else:
            # Accumulate reward
            if hasattr(self.locals, 'rewards'):
                self.current_episode_reward += self.locals.get('rewards', [0])[0]
        
        return True  # Continue training
    
    def _on_training_end(self) -> None:
        """Called at the end of training. Clean up pygame."""
        pygame.quit()
        logger.info("Pygame window closed")


def make_env(rom_path, state_path, rank=0, visualize=False):
    """
    Utility function for creating a monitored environment.
    
    Args:
        rom_path: Path to ROM
        state_path: Path to save state
        rank: Environment rank for logging
        visualize: If True, create with render_mode='human' (shows pygame window)
    """
    def _init():
        logger.info(f"[Env {rank}] Creating environment...")
        env = PokemonEmeraldEnv(
            rom_path=rom_path,
            initial_state_path=state_path,
            render_mode='human' if visualize else None,
            max_steps=10000,
            frame_skip=12,  # 12 frames = 0.2s per action (same as server/app.py)
        )
        logger.info(f"[Env {rank}] Wrapping with Monitor...")
        env = Monitor(env, f"./logs/monitor_{rank}")
        logger.info(f"[Env {rank}] Environment ready!")
        return env
    return _init


def train_ppo(
    rom_path: str = "Emerald-GBAdvance/rom.gba",
    initial_state_path: str = "Emerald-GBAdvance/quick_start_save.state",
    total_timesteps: int = 1_000_000,
    save_freq: int = 10_000,
    model_save_path: str = "./models/ppo_pokemon",
    log_dir: str = "./logs",
    tensorboard_log: str = "./tensorboard_logs",
    n_envs: int = 1,  # Number of parallel environments
    visualize: bool = False,  # If True, show pygame window (only works with n_envs=1)
    use_llm: bool = False,  # 🆕 Enable LLM-based reward shaping
    pure_drl: bool = False  # 🆕 Pure DRL mode (no reward shaping at all)
):
    """
    Train a PPO agent on Pokemon Emerald.
    
    Args:
        rom_path: Path to Pokemon Emerald ROM
        initial_state_path: Path to initial save state
        total_timesteps: Total training steps
        save_freq: How often to save checkpoints
        model_save_path: Where to save the final model
        log_dir: Directory for logs
        tensorboard_log: Directory for tensorboard logs
        n_envs: Number of parallel environments (more = faster training)
        visualize: Show pygame window during training (only works with n_envs=1)
        use_llm: Enable LLM-based reward shaping with dialogue reading
        pure_drl: Pure DRL mode - disable ALL reward shaping callbacks
    """
    # Validate visualize mode
    if visualize and n_envs > 1:
        logger.warning("⚠️  visualize=True only works with n_envs=1. Setting n_envs=1.")
        n_envs = 1
    
    # Validate conflicting flags
    if use_llm and pure_drl:
        logger.warning("⚠️  Cannot use --use-llm and --pure-drl together. Using --pure-drl (no reward shaping).")
        use_llm = False
    
    # Create directories
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("🎮 Pokemon Emerald PPO Training")
    logger.info("=" * 60)
    logger.info(f"ROM: {rom_path}")
    logger.info(f"Initial State: {initial_state_path}")
    logger.info(f"Parallel Environments: {n_envs}")
    logger.info(f"Total Timesteps: {total_timesteps:,}")
    logger.info(f"Save Frequency: {save_freq:,}")
    
    # 🆕 Mostrar modo de entrenamiento
    if pure_drl:
        logger.info(f"Training Mode: 🔵 PURE DRL (no reward shaping)")
    elif use_llm:
        logger.info(f"Training Mode: 🤖 LLM + Dialogue-based reward shaping")
    else:
        logger.info(f"Training Mode: 📊 Rule-based milestone reward shaping")
    
    logger.info("=" * 60)
    
    # Create environment(s) - multiple for faster training
    logger.info(f"Creating {n_envs} parallel environment(s)...")
    
    # Use SubprocVecEnv for TRUE parallel execution (each env in separate process)
    # Note: With START button removed, save crashes should be eliminated
    if n_envs > 1:
        logger.info(f"Using SubprocVecEnv ({n_envs} processes, TRUE parallel)")
        env_fns = [make_env(rom_path, initial_state_path, rank=i, visualize=(visualize and i == 0)) for i in range(n_envs)]
        env = SubprocVecEnv(env_fns, start_method='spawn')  # spawn method is safer than fork for mGBA
    else:
        logger.info("Using DummyVecEnv (single environment)")
        env_fns = [make_env(rom_path, initial_state_path, rank=0, visualize=visualize)]
        env = DummyVecEnv(env_fns)
    
    env = VecMonitor(env, log_dir)
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback (always enabled)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_pokemon",
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    callbacks.append(checkpoint_callback)
    
    # 🆕 REWARD SHAPING CALLBACKS - Configurables según modo
    if not pure_drl:
        # LLM reward shaping callback
        llm_callback = LLMRewardCallback(
            check_frequency=500,  # Check every 500 steps para análisis LLM
            use_llm=use_llm,  # Usar LLM si --use-llm está activo
            llm_timeout=60,  # 🆕 Timeout de 60 segundos para LLM (era 15s)
            verbose=1
        )
        callbacks.append(llm_callback)
        
        if use_llm:
            logger.info("✅ LLM reward shaping enabled (DIALOGUE-BASED MODE - reads game text!)")
        else:
            logger.info("✅ LLM reward shaping enabled (RULE-BASED MODE - milestone detection)")
    else:
        logger.info("🔵 Pure DRL mode - ALL reward shaping disabled")
    
    # Directional reward shaping callback (proximity-based) - DESACTIVADO PARA PRUEBA
    # Recompensa progresivamente cuando el agente se acerca a objetivos
    # Más frecuente que LLM (cada 100 steps) para guiar continuamente
    # NOTA: Desactivado temporalmente para probar DRL puro sin reward shaping direccional
    """
    from agent.directional_reward_callback import DirectionalRewardCallback
    directional_callback = DirectionalRewardCallback(
        check_frequency=100,  # Check every 100 steps (más frecuente que LLM)
        proximity_boost=1.5,  # 1.5× reward cuando se acerca a objetivos
        proximity_penalty=0.8,  # 0.8× reward cuando se aleja
        verbose=1
    )
    callbacks.append(directional_callback)
    logger.info("🧭 Directional reward shaping enabled (proximity-based guidance)")
    """
    logger.info("⚠️  Directional reward shaping DISABLED - Testing pure DRL learning")
    
    # Pygame visualization callback (only if visualize=True)

    if visualize:
        logger.info("🎮 Enabling pygame visualization...")
        render_callback = PygameRenderCallback(
            env=env,
            render_freq=1,  # Render every step
            fps=30,  # 30 FPS display (smooth but not too slow)
            verbose=0
        )
        callbacks.append(render_callback)
        logger.info("✅ Pygame window will open during training")
        logger.info("   Close the window to stop training gracefully")
    
    logger.info("Creating PPO model with CNN policy...")
    
    # Create PPO model with custom CNN policy
    model = PPO(
        PokemonCNNPolicy,  # Use custom CNN policy instead of MlpPolicy
        env,
        learning_rate=3e-4,
        n_steps=4096,      # Collect more experiences before update (was 2048)
        batch_size=128,    # Larger batches for better GPU usage (was 64)
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device='auto'  # Use GPU if available
    )
    
    logger.info("Model created successfully!")
    logger.info(f"Policy: {model.policy}")
    logger.info(f"Device: {model.device}")
    logger.info("")
    logger.info("🚀 Starting training...")
    logger.info("=" * 60)
    
    try:
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        logger.info("=" * 60)
        logger.info("✅ Training completed successfully!")
        logger.info("=" * 60)
        
        # Save final model
        logger.info(f"Saving final model to: {model_save_path}")
        model.save(model_save_path)
        logger.info("✅ Model saved!")
        
        return model
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        logger.info("Saving current model...")
        model.save(model_save_path + "_interrupted")
        logger.info("✅ Model saved!")
        return model
    
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        raise
    
    finally:
        env.close()
        logger.info("Environment closed.")


def test_model(
    model_path: str,
    rom_path: str = "Emerald-GBAdvance/rom.gba",
    initial_state_path: str = "Emerald-GBAdvance/start.state",
    n_episodes: int = 5
):
    """
    Test a trained PPO model.
    
    Args:
        model_path: Path to saved model
        rom_path: Path to ROM
        initial_state_path: Path to initial state
        n_episodes: Number of test episodes
    """
    logger.info("=" * 60)
    logger.info("🎮 Testing Trained Model")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Episodes: {n_episodes}")
    logger.info("=" * 60)
    
    # Load model
    logger.info("Loading model...")
    model = PPO.load(model_path)
    logger.info("✅ Model loaded!")
    
    # Create test environment
    env = PokemonEmeraldEnv(
        rom_path=rom_path,
        initial_state_path=initial_state_path,
        render_mode='human',
        max_steps=10000
    )
    
    # Run test episodes
    for episode in range(n_episodes):
        logger.info(f"\n--- Episode {episode + 1}/{n_episodes} ---")
        obs, info = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated
            
            # Log progress
            if step % 100 == 0:
                logger.info(f"Step {step}: Location={info['location']}, Badges={info['badges']}, Reward={episode_reward:.2f}")
        
        logger.info(f"Episode finished! Total reward: {episode_reward:.2f}, Steps: {step}")
    
    env.close()
    logger.info("\n✅ Testing complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or test PPO agent on Pokemon Emerald")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Mode: train or test")
    parser.add_argument("--rom", type=str, default="Emerald-GBAdvance/rom.gba",
                        help="Path to ROM file")
    parser.add_argument("--state", type=str, default="Emerald-GBAdvance/quick_start_save.state",
                        help="Path to initial state file")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--save-freq", type=int, default=10_000,
                        help="Checkpoint save frequency")
    parser.add_argument("--model-path", type=str, default="./models/ppo_pokemon",
                        help="Path to save/load model")
    parser.add_argument("--test-episodes", type=int, default=5,
                        help="Number of test episodes")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments (recommended: 1-4 for DummyVecEnv)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show pygame window during training (only works with --n-envs 1)")
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable LLM-based reward shaping with dialogue reading (requires Ollama)")
    parser.add_argument("--pure-drl", action="store_true",
                        help="Pure DRL mode: disable ALL reward shaping (no milestones, no LLM)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_ppo(
            rom_path=args.rom,
            initial_state_path=args.state,
            total_timesteps=args.timesteps,
            save_freq=args.save_freq,
            model_save_path=args.model_path,
            n_envs=args.n_envs,
            visualize=args.visualize,
            use_llm=args.use_llm,
            pure_drl=args.pure_drl
        )
    else:
        test_model(
            model_path=args.model_path,
            rom_path=args.rom,
            initial_state_path=args.state,
            n_episodes=args.test_episodes
        )
