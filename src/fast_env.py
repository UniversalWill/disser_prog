"""
Fast training environment for RL Hyper-Heuristic.
Uses pre-computed metrics instead of running actual GA/SA/CP-SAT.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from .models import ScheduleContext


class FastSchedulingEnv(gym.Env):
    """
    Lightweight environment for fast RL training.
    
    Simulates the behavior of GA/SA/CP-SAT without actually running them.
    Uses learned transition probabilities to predict improvement.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 context: ScheduleContext,
                 max_steps: int = 50,
                 max_time_seconds: float = 60.0,
                 render_mode: Optional[str] = None):
        
        super().__init__()
        
        self.context = context
        self.max_steps = max_steps
        self.max_time_seconds = max_time_seconds
        self.render_mode = render_mode
        
        # Action space: 3 discrete actions
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 5-dimensional continuous vector [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.start_time = 0.0
        
        # Metrics tracking
        self.hard_conflicts = 0
        self.soft_score = 0
        self.stagnation = 0
        
        # Initial values (simulated)
        self.initial_hard = 25.0
        self.initial_soft = 1000.0
        
        # Action effects (learned from real runs)
        # [mean_improvement, std_improvement] for hard conflicts
        self.action_effects = {
            0: {'hard': 0.15, 'soft': 0.05},   # GA: good for exploration
            1: {'hard': 0.4, 'soft': 0.0},     # CP-SAT: fixes hard conflicts
            2: {'hard': 0.05, 'soft': 0.2},    # SA: good for soft optimization
        }
        
    def _get_obs(self) -> np.ndarray:
        """Generate observation vector."""
        hard_ratio = min(1.0, self.hard_conflicts / self.initial_hard)
        soft_ratio = min(1.0, max(0.0, self.soft_score) / self.initial_soft)
        stagnation = min(1.0, self.stagnation / 10.0)
        budget_used = min(1.0, self.current_step / self.max_steps)
        validity = 0.0 if self.hard_conflicts > 0.1 else 1.0
        
        return np.array([hard_ratio, soft_ratio, stagnation, budget_used, validity], dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Randomize initial state
        self.hard_conflicts = self.initial_hard * (0.8 + 0.4 * random.random())
        self.soft_score = self.initial_soft * (0.8 + 0.4 * random.random())
        
        self.current_step = 0
        self.stagnation = 0
        self.start_time = 0.0
        
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Get action effects
        effects = self.action_effects[action]
        
        # Apply improvement with some randomness
        hard_reduction = effects['hard'] * (0.5 + random.random())
        soft_reduction = effects['soft'] * (0.5 + random.random())
        
        old_hard = self.hard_conflicts
        old_soft = self.soft_score
        
        # Update state
        self.hard_conflicts = max(0, self.hard_conflicts * (1 - hard_reduction))
        self.soft_score = max(0, self.soft_score * (1 - soft_reduction))
        
        # Check for improvement
        if self.hard_conflicts < old_hard - 0.1 or self.soft_score < old_soft - 10:
            self.stagnation = 0
        else:
            self.stagnation += 1
        
        # Calculate reward
        reward = 0
        reward += (old_hard - self.hard_conflicts) * 10  # Reward for fixing hard conflicts
        reward += (old_soft - self.soft_score) * 0.01   # Reward for soft improvement
        
        # Action costs
        action_costs = {0: 2, 1: 5, 2: 1}
        reward -= action_costs[action] * 0.1
        
        # Bonus for achieving validity
        if self.hard_conflicts < 0.1 and old_hard >= 0.1:
            reward += 50
        
        # Stagnation penalty
        if self.stagnation > 5:
            reward -= 1
        
        # Termination
        terminated = self.hard_conflicts < 0.1 and self.soft_score < 50
        truncated = self.current_step >= self.max_steps
        
        info = {
            'action': action,
            'action_name': ['GA', 'CP-SAT', 'SA'][action],
            'hard_conflicts': self.hard_conflicts,
            'soft_score': self.soft_score,
            'stagnation': self.stagnation
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        """Render the current state."""
        if self.render_mode == 'human':
            print(f"Step {self.current_step}: Hard={self.hard_conflicts:.1f}, Soft={self.soft_score:.1f}")


class FastMultiScenarioEnv(gym.Env):
    """
    Fast multi-scenario environment for RL training.
    Randomly generates different problem configurations.
    """
    
    def __init__(self, 
                 max_steps: int = 50,
                 max_time_seconds: float = 60.0):
        
        super().__init__()
        
        self.max_steps = max_steps
        self.max_time_seconds = max_time_seconds
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        
        self.current_env: Optional[FastSchedulingEnv] = None
        
        # Scenario difficulties
        self.difficulties = [
            {'hard': 15.0, 'soft': 500.0},   # Easy
            {'hard': 25.0, 'soft': 1000.0},  # Medium
            {'hard': 40.0, 'soft': 1500.0},  # Hard
        ]
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Create a mock context (not used in fast mode)
        from .models import ScheduleContext, Teacher, Group, Room, Slot, Event, RoomType
        mock_context = ScheduleContext(
            teachers=[Teacher(id=1, name="T1", max_hours=40)],
            groups=[Group(id=1, name="G1", size=20)],
            rooms=[Room(id=1, capacity=50, type=RoomType.LECTURE, building="Main")],
            slots=[Slot(id=1, day=1, time=1)],
            events=[Event(id=1, subject_name="Test", teacher_id=1, group_ids=[1], duration=1, room_type_required=RoomType.LECTURE)]
        )
        
        self.current_env = FastSchedulingEnv(
            context=mock_context,
            max_steps=self.max_steps,
            max_time_seconds=self.max_time_seconds
        )
        
        # Apply random difficulty
        difficulty = random.choice(self.difficulties)
        self.current_env.initial_hard = difficulty['hard']
        self.current_env.initial_soft = difficulty['soft']
        
        return self.current_env.reset(seed=seed)
    
    def step(self, action: int):
        return self.current_env.step(action)
