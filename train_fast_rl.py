"""
Fast RL training script using simplified environment.
Trains in minutes instead of hours.
"""

import os
import random
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List

from src.fast_env import FastMultiScenarioEnv


class ProgressCallback(BaseCallback):
    """Callback to print training progress."""
    
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                print(f"Step {self.n_calls}: reward={ep_info.get('r', 0):.1f}, episodes={self.num_episodes}")
        return True


def train_fast_rl(
    total_timesteps: int = 100000,
    save_path: str = "rl_models",
    learning_rate: float = 3e-4
):
    """
    Train RL agent using fast simulated environment.
    """
    
    print("="*60)
    print("FAST RL TRAINING - Hyper-Heuristic Scheduler")
    print("="*60)
    
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\nTotal timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Save path: {save_path}")
    
    # Create environment
    env = FastMultiScenarioEnv(
        max_steps=50,
        max_time_seconds=60.0
    )
    
    vec_env = DummyVecEnv([lambda: env])
    
    # Create PPO agent
    print("\nInitializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        device="cpu",
        tensorboard_log=os.path.join(save_path, "tensorboard_fast")
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix="fast_rl_scheduler"
    )
    
    progress_callback = ProgressCallback(check_freq=2000)
    
    # Train
    print("\nStarting training...")
    print("-"*60)
    
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, progress_callback],
        progress_bar=False
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save final model
    final_model_path = os.path.join(save_path, "fast_rl_scheduler_final.zip")
    model.save(final_model_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Timesteps per second: {total_timesteps/duration:.1f}")
    print(f"Final model saved: {final_model_path}")
    
    return model


def evaluate_policy(model_path: str = "rl_models/fast_rl_scheduler_final.zip", n_episodes: int = 10):
    """
    Evaluate the trained policy.
    """
    from stable_baselines3 import PPO
    
    print("\n" + "="*60)
    print("EVALUATING TRAINED POLICY")
    print("="*60)
    
    model = PPO.load(model_path)
    env = FastMultiScenarioEnv(max_steps=50)
    
    total_rewards = []
    actions_taken = {'GA': 0, 'CP-SAT': 0, 'SA': 0}
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            action_name = info.get('action_name', 'Unknown')
            actions_taken[action_name] = actions_taken.get(action_name, 0) + 1
            
            done = terminated or truncated
        
        total_rewards.append(total_reward)
    
    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Mean reward: {np.mean(total_rewards):.2f} (+/- {np.std(total_rewards):.2f})")
    print(f"  Min/Max: {np.min(total_rewards):.2f} / {np.max(total_rewards):.2f}")
    print(f"\nAction distribution:")
    for action, count in actions_taken.items():
        print(f"  {action}: {count} times ({count/sum(actions_taken.values())*100:.1f}%)")


if __name__ == "__main__":
    # Train with fast environment
    model = train_fast_rl(
        total_timesteps=100000,  # Should complete in 2-5 minutes
        save_path="rl_models"
    )
    
    # Evaluate the trained policy
    evaluate_policy(n_episodes=20)
