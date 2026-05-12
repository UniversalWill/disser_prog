from typing import List, Tuple, Dict, Any
import os
from .models import ScheduleContext
from .rl_env import SchedulingEnv
from .validator import ScheduleValidator


class RLScheduler:
    """
    RL-based Hyper-Heuristic Scheduler.
    Uses a trained RL agent to dynamically select and apply optimization operators.
    """
    
    def __init__(self, 
                 context: ScheduleContext, 
                 model_path: str = "rl_models/rl_scheduler_final.zip",
                 max_steps: int = 50,
                 max_time_seconds: float = 60.0):
        """
        Initialize the RL scheduler.
        
        Args:
            context: Scheduling context with all entities
            model_path: Path to trained RL model (PPO)
            max_steps: Maximum number of actions the agent can take
            max_time_seconds: Maximum time budget in seconds
        """
        self.context = context
        self.model_path = model_path
        self.max_steps = max_steps
        self.max_time_seconds = max_time_seconds
        self.model = None
        
    def load_model(self):
        """Load the trained RL model."""
        from stable_baselines3 import PPO
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"RL model not found at {self.model_path}. "
                f"Please train the model first using train_rl.py"
            )
        
        self.model = PPO.load(self.model_path)
        print(f"RL model loaded from {self.model_path}")
    
    def solve(self) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
        """
        Solve the scheduling problem using the RL agent.
        
        Returns:
            Tuple of (schedule, metrics)
        """
        # Load model if not loaded
        if self.model is None:
            self.load_model()
        
        # Create environment
        env = SchedulingEnv(
            context=self.context,
            max_steps=self.max_steps,
            max_time_seconds=self.max_time_seconds
        )
        
        # Reset and run
        obs, _ = env.reset()
        
        metrics = {
            'actions_taken': [],
            'hard_conflicts_history': [],
            'soft_score_history': [],
            'total_steps': 0,
            'terminated': False,
            'truncated': False
        }
        
        print("\n=== RL Agent Execution ===")
        print(f"Max steps: {self.max_steps}, Max time: {self.max_time_seconds}s")
        print("-" * 40)
        
        done = False
        step = 0
        
        while not done:
            # Get action from RL agent
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Log
            action_names = {0: 'GA', 1: 'CP-SAT', 2: 'SA'}
            action_name = action_names.get(action, f'Action {action}')
            
            metrics['actions_taken'].append(action_name)
            metrics['hard_conflicts_history'].append(info.get('hard_conflicts', 0))
            metrics['soft_score_history'].append(info.get('soft_score', 0))
            
            # Progress output
            if step % 5 == 0 or terminated or truncated:
                print(f"Step {step+1}: Action={action_name}, "
                      f"Hard={info.get('hard_conflicts', '?')}, "
                      f"Soft={info.get('soft_score', '?'):.0f}")
            
            done = terminated or truncated
            step += 1
        
        metrics['total_steps'] = step
        metrics['terminated'] = terminated
        metrics['truncated'] = truncated
        
        # Get final schedule
        final_schedule = env.get_current_schedule()
        
        # Final validation
        validator = ScheduleValidator(self.context, final_schedule)
        is_valid, errors, v_metrics = validator.validate()
        metrics['validation'] = {
            'is_valid': is_valid,
            'errors': errors,
            'metrics': v_metrics
        }
        
        print("-" * 40)
        print(f"Execution completed in {step} steps")
        print(f"Final schedule valid: {is_valid}")
        print(f"Hard conflicts: {v_metrics['hard_conflicts']}")
        print(f"Soft score: {v_metrics.get('teacher_preference_score', 0)}")
        
        return final_schedule, metrics
    
    def get_action_distribution(self) -> Dict[str, int]:
        """Analyze which actions the agent prefers."""
        if not hasattr(self, '_action_counts'):
            return {}
        return self._action_counts


class HybridSchedulerWithRL:
    """
    Hybrid approach: Uses RL agent to orchestrate GA, CP-SAT, and SA.
    Falls back to static pipeline if RL model is not available.
    """
    
    def __init__(self, context: ScheduleContext, model_path: str = None, use_rl: bool = True):
        self.context = context
        self.model_path = model_path or "rl_models/rl_scheduler_final.zip"
        self.use_rl = use_rl
    
    def solve(self) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
        """Solve using RL if available, otherwise use static pipeline."""
        
        if self.use_rl and os.path.exists(self.model_path):
            print("Using RL-based Hyper-Heuristic Scheduler")
            scheduler = RLScheduler(self.context, self.model_path)
            return scheduler.solve()
        else:
            print("Using Static Hybrid Pipeline (RL model not found)")
            from .solver import HybridScheduler
            scheduler = HybridScheduler(self.context)
            return scheduler.solve()
