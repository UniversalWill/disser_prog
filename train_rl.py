import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List
import os
from datetime import datetime

from src.models import ScheduleContext, Teacher, Group, Room, Slot, Event, RoomType, WeekType
from src.rl_env import SchedulingEnv


class MultiScenarioEnv(gym.Env):
    """
    Environment wrapper that randomizes scenarios during training.
    This helps the agent generalize to different scheduling problems.
    """
    
    def __init__(self, 
                 scenario_configs: Optional[List[dict]] = None,
                 max_steps: int = 50,
                 max_time_seconds: float = 60.0,
                 fast_mode: bool = True):
        
        super().__init__()
        
        self.scenario_configs = scenario_configs or [
            {'n_groups': 8, 'n_teachers': 8, 'n_events': 70},
            {'n_groups': 10, 'n_teachers': 10, 'n_events': 100},
            {'n_groups': 12, 'n_teachers': 10, 'n_events': 120},
        ]
        
        self.max_steps = max_steps
        self.max_time_seconds = max_time_seconds
        self.fast_mode = fast_mode
        
        # Action and observation spaces (same as SchedulingEnv)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        
        self.current_env: Optional[SchedulingEnv] = None
        self.current_config = None
        
    def _generate_scenario(self, config: dict) -> ScheduleContext:
        """Generate a random scheduling scenario."""
        n_groups = config['n_groups']
        n_teachers = config['n_teachers']
        n_events = config['n_events']
        
        # Teachers
        teachers = []
        for i in range(1, n_teachers + 1):
            prefs = {}
            if random.random() > 0.5:
                prefs[random.randint(1, 25)] = 10
            if random.random() > 0.5:
                prefs[random.randint(1, 25)] = -10
            teachers.append(Teacher(
                id=i, 
                name=f"Teacher_{i}", 
                max_hours=40, 
                preferences=prefs
            ))
        
        # Groups
        groups = []
        for i in range(1, n_groups + 1):
            groups.append(Group(
                id=i, 
                name=f"Group-{i}", 
                size=random.randint(15, 30)
            ))
        
        # Rooms (proportional to number of groups)
        n_lecture_rooms = max(5, n_groups // 2)
        n_lab_rooms = max(2, n_groups // 4)
        rooms = []
        for i in range(1, n_lecture_rooms + 1):
            rooms.append(Room(id=i, capacity=80, type=RoomType.LECTURE, building="Main"))
        for i in range(n_lecture_rooms + 1, n_lecture_rooms + n_lab_rooms + 1):
            rooms.append(Room(id=i, capacity=40, type=RoomType.LAB, building="Lab"))
        
        # Slots (5 days, 5 pairs)
        slots = []
        slot_id = 1
        for day in range(1, 6):
            for time in range(1, 6):
                slots.append(Slot(id=slot_id, day=day, time=time))
                slot_id += 1
        
        # Events
        subjects = ["Math", "Physics", "Programming", "Databases", "Networks", "History", "Philosophy", "English"]
        events = []
        for i in range(1, n_events + 1):
            subject = random.choice(subjects)
            teacher = random.choice(teachers)
            
            if random.random() < 0.2:
                g_ids = random.sample([g.id for g in groups], 2)
                r_type = RoomType.LECTURE
            else:
                g_ids = [random.choice(groups).id]
                r_type = random.choice([RoomType.LECTURE, RoomType.LAB])
            
            events.append(Event(
                id=i,
                subject_name=subject,
                teacher_id=teacher.id,
                group_ids=g_ids,
                duration=1,
                room_type_required=r_type
            ))
        
        return ScheduleContext(
            teachers=teachers,
            groups=groups,
            rooms=rooms,
            slots=slots,
            events=events
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Randomly select a scenario
        self.current_config = random.choice(self.scenario_configs)
        context = self._generate_scenario(self.current_config)
        
        self.current_env = SchedulingEnv(
            context=context,
            max_steps=self.max_steps,
            max_time_seconds=self.max_time_seconds,
            fast_mode=self.fast_mode
        )
        
        return self.current_env.reset(seed=seed)
    
    def step(self, action: int):
        return self.current_env.step(action)
    
    def get_current_schedule(self):
        return self.current_env.get_current_schedule()


class TrainingProgressCallback(BaseCallback):
    """Callback to log training progress."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log info from info dict
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if self.n_calls % 1000 == 0:
                print(f"Step {self.n_calls}: reward={ep_info.get('r', 0):.2f}, len={ep_info.get('l', 0)}")
        return True


def train_rl_agent(
    total_timesteps: int = 100000,
    save_path: str = "rl_models",
    scenario_configs: Optional[List[dict]] = None,
    fast_mode: bool = True
):
    """
    Train an RL agent using PPO on multiple scheduling scenarios.
    
    Args:
        total_timesteps: Total number of training steps
        save_path: Directory to save the trained model
        scenario_configs: List of scenario configurations
        fast_mode: Use fast iterations for quick training
    """
    
    print("="*60)
    print("RL AGENT TRAINING - Hyper-Heuristic Scheduler")
    print("="*60)
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Default scenario configs
    if scenario_configs is None:
        scenario_configs = [
            {'n_groups': 6, 'n_teachers': 6, 'n_events': 50},
            {'n_groups': 8, 'n_teachers': 8, 'n_events': 70},
            {'n_groups': 10, 'n_teachers': 10, 'n_events': 100},
        ]
    
    print(f"\nTraining scenarios:")
    for i, cfg in enumerate(scenario_configs):
        print(f"  {i+1}. Groups: {cfg['n_groups']}, Teachers: {cfg['n_teachers']}, Events: {cfg['n_events']}")
    
    print(f"\nTotal timesteps: {total_timesteps}")
    print(f"Save path: {save_path}")
    
    # Create environment
    env = MultiScenarioEnv(
        scenario_configs=scenario_configs,
        max_steps=50,
        max_time_seconds=60.0,
        fast_mode=fast_mode
    )
    
    # Wrap in DummyVecEnv for stable-baselines3
    vec_env = DummyVecEnv([lambda: env])
    
    # Create PPO agent
    print("\nInitializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        device="cpu",  # CPU is faster for MLP policy
        tensorboard_log=os.path.join(save_path, "tensorboard")
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix="rl_scheduler"
    )
    
    progress_callback = TrainingProgressCallback()
    
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
    final_model_path = os.path.join(save_path, "rl_scheduler_final.zip")
    model.save(final_model_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Duration: {duration:.1f} seconds")
    print(f"Final model saved: {final_model_path}")
    print(f"Checkpoints saved in: {save_path}")
    
    return model


if __name__ == "__main__":
    # Train with default settings
    model = train_rl_agent(
        total_timesteps=50000,  # 50k steps with fast mode
        save_path="rl_models",
        fast_mode=True
    )
    
    print("\nTo use the trained model:")
    print("  from stable_baselines3 import PPO")
    print("  model = PPO.load('rl_models/rl_scheduler_final.zip')")
