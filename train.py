"""
MARL FJSP Training Script
=========================

Training entry point for Multi-Agent Reinforcement Learning
Flexible Job Shop Scheduling Problem.

Usage:
    python train.py
    python train.py --timesteps 100000 --batch_size 500
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from FJSPParallelEnvWrapper import FJSPParallelEnv
from a2c import MultiAgentA2C
from typing import List


def plot_rewards_vs_timesteps(timesteps: List[int], rewards: List[float], out_path: str = "reward_curve.png"):
    """
    Plot episode rewards against cumulative timesteps and save to disk.
    Uses symlog on y-axis to handle wide/negative reward ranges while keeping zero visible.
    """
    if not timesteps or not rewards:
        print("No reward data to plot.")
        return
    
    # Align lengths in case of mismatch
    n = min(len(timesteps), len(rewards))
    ts = timesteps[:n]
    rs = rewards[:n]
    
    plt.figure(figsize=(8, 4.5))
    plt.plot(ts, rs, marker='o', linewidth=1, markersize=3)
    plt.xlabel("Cumulative Timesteps")
    plt.ylabel("Episode Reward")
    plt.yscale("symlog")
    plt.title("Reward vs. Timesteps")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved reward plot to {out_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MARL FJSP')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Total training timesteps')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for updates')

    # A2C hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lamb', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--lr_actor', type=float, default=0.0003,
                        help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=0.001,
                        help='Critic learning rate')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient for exploration')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Max gradient norm for clipping')
    
    # Environment parameters
    parser.add_argument('--num_initial_orders', type=int, default=5,
                        help='Number of orders to start with')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Max steps per episode')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_path', type=str, default='./checkpoints',
                        help='Path to save model checkpoints')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Enable grid visualization during training')

    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable detailed simulation logging')

    # Train/Test split
    parser.add_argument('--train_orders', type=int, default=25,
                        help='Number of orders for training (default: 25)')
    parser.add_argument('--test_orders', type=int, default=5,
                        help='Number of orders for testing (default: 5)')
    parser.add_argument('--test_only_viz', action='store_true',
                        help='Only visualize during test phase (not training)')
    parser.add_argument('--heuristic', action='store_true',
                        help='Use rule-based heuristic policy for testing instead of learned policy')

    # Test-only mode
    parser.add_argument('--test_only', action='store_true',
                        help='Skip training and only run test with a loaded model')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to load a pre-trained model checkpoint')

    return parser.parse_args()


def train(args):
    """Main training function."""

    # Configure logging
    if args.verbose:
        from utils.Logger import get_logger
        logger = get_logger()
        logger.enable()
        log_file = logger.setup_file_logging("logs")
        print(f"Logging to: {log_file}")

    print("=" * 60)
    print("MARL FJSP Training")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Set random seed
    np.random.seed(args.seed)
    
    # Create environment
    print("Creating environment...")
    env = FJSPParallelEnv()
    
    print(f"  Agents: {env.possible_agents}")
    print(f"  Number of agents: {len(env.possible_agents)}")
    print()
    
    # Print observation and action spaces
    print("Observation spaces:")
    for agent in env.possible_agents:
        obs_space = env.observation_space(agent)
        print(f"  {agent}: {obs_space}")
    print()
    
    print("Action spaces:")
    for agent in env.possible_agents:
        act_space = env.action_space(agent)
        print(f"  {agent}: {act_space}")
    print()
    
    # Determine if we should visualize during training
    # If --test_only_viz is set, don't visualize training even if --visualize is set
    visualize_training = args.visualize and not args.test_only_viz

    # Create Multi-Agent A2C
    print("Creating Multi-Agent A2C...")
    ma2c = MultiAgentA2C(
        env=env,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lamb=args.lamb,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        use_gae=True,
        visualize=visualize_training,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm
    )

    print(f"  Global state dimension (critic): {ma2c.global_obs_dim}")
    print(f"  Observation dimensions: {ma2c.obs_dims}")
    print(f"  Action dimensions: {ma2c.act_dims}")
    print()

    # Load pre-trained model if specified
    if args.load_path:
        print(f"Loading model from {args.load_path}...")
        ma2c.load_model(args.load_path)

    # Skip training if test_only mode
    episode_rewards = []
    if args.test_only:
        print("=" * 60)
        print("TEST-ONLY MODE: Skipping training")
        print("=" * 60)
        if not args.load_path:
            print("WARNING: No model loaded (--load_path not specified)")
            print("         Using untrained/random policy!")
    else:
        # Training
        print("=" * 60)
        print("Starting training...")
        print(f"  Total timesteps: {args.timesteps}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Gamma: {args.gamma}")
        print(f"  Lambda: {args.lamb}")
        print(f"  Actor LR: {args.lr_actor}")
        print(f"  Critic LR: {args.lr_critic}")
        print(f"  Entropy coef: {args.entropy_coef}")
        print(f"  Max grad norm: {args.max_grad_norm}")
        print(f"  Training orders: {args.train_orders}")
        print(f"  Test orders: {args.test_orders}")
        print(f"  Visualization (train): {visualize_training}")
        print(f"  Visualization (test): {args.visualize or args.test_only_viz}")
        print("=" * 60)
        print()

        # Run training with train_orders
        episode_rewards = ma2c.learn(total_timesteps=args.timesteps, num_orders=args.train_orders)

        # Plot rewards vs timesteps
        plot_rewards_vs_timesteps(ma2c.episode_end_timesteps, episode_rewards, out_path="reward_curve.png")

        # Training complete
        print()
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total episodes: {len(episode_rewards)}")

        if len(episode_rewards) > 0:
            print(f"Final avg reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
            print(f"Best episode reward: {max(episode_rewards):.2f}")
            print(f"Worst episode reward: {min(episode_rewards):.2f}")

        # Save model after training
        import os
        os.makedirs(args.save_path, exist_ok=True)
        model_path = os.path.join(args.save_path, "model.pt")
        ma2c.save_model(model_path)

    # === TEST PHASE ===
    # Run test with test_orders and visualization if requested
    visualize_test = args.visualize or args.test_only_viz
    test_results = ma2c.test(
        num_orders=args.test_orders,
        visualize=visualize_test,
        max_steps=args.max_steps,
        use_heuristic=args.heuristic
    )


    return ma2c, episode_rewards, test_results


def test_environment():
    """Quick test to verify environment works."""
    print("Testing environment...")
    
    env = FJSPParallelEnv()
    observations, infos = env.reset()
    
    print(f"Initial agents: {env.agents}")
    print(f"Initial observations keys: {observations.keys()}")
    
    # Run a few steps with random actions
    for step in range(5):
        if not env.agents:
            break
        
        # Random actions
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        print(f"Step {step + 1}: rewards = {rewards}")
    
    print("Environment test passed!\n")


if __name__ == "__main__":
    args = parse_args()

    # Optional: test environment first
    # test_environment()

    # Run training and testing
    model, rewards, test_results = train(args)

    # model.plot_losses(save_path='losses.png')
