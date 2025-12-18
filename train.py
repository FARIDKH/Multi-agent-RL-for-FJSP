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
from datetime import datetime

from FJSPParallelEnvWrapper import FJSPParallelEnv
from a2c import MultiAgentA2C


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MARL FJSP')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Total training timesteps')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size for updates')
    
    # A2C hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lamb', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--lr_actor', type=float, default=0.001,
                        help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=0.001,
                        help='Critic learning rate')
    
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
    
    return parser.parse_args()


def train(args):
    """Main training function."""
    
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
    
    # Create Multi-Agent A2C
    print("Creating Multi-Agent A2C...")
    ma2c = MultiAgentA2C(
        env=env,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lamb=args.lamb,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        use_gae=True
    )
    
    print(f"  Global state dimension (critic): {ma2c.global_obs_dim}")
    print(f"  Observation dimensions: {ma2c.obs_dims}")
    print(f"  Action dimensions: {ma2c.act_dims}")
    print()
    
    # Training
    print("=" * 60)
    print("Starting training...")
    print(f"  Total timesteps: {args.timesteps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Lambda: {args.lamb}")
    print(f"  Actor LR: {args.lr_actor}")
    print(f"  Critic LR: {args.lr_critic}")
    print("=" * 60)
    print()
    
    # Run training
    episode_rewards = ma2c.learn(total_timesteps=args.timesteps)
    
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
    
    # Save model (optional)
    # TODO: Implement model saving
    # save_model(ma2c, args.save_path)
    
    return ma2c, episode_rewards


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
    
    # Run training
    model, rewards = train(args)