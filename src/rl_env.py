import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import copy
import random
from typing import List, Tuple, Dict, Any, Optional
from .models import ScheduleContext
from .phase1_ga import ScheduleGA
from .phase2_cpsat import repair_with_cpsat
from .phase3_sa import optimize_soft_constraints, check_hard_conflicts, evaluate_soft_score

class SchedulingEnv(gym.Env):
    """
    Gymnasium Environment for Schedule Optimization using RL as Hyper-Heuristic.
    
    Actions:
        0: Run GA for 10 generations (exploration)
        1: Run CP-SAT Repair (fix hard conflicts)
        2: Run SA for 100 iterations (local optimization)
    
    Observation:
        [hard_conflicts_ratio, soft_penalty_ratio, stagnation, budget_used, validity]
    
    Reward:
        Potential-based shaping for stable learning.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 context: ScheduleContext,
                 max_steps: int = 50,
                 max_time_seconds: float = 60.0,
                 fast_mode: bool = True,
                 render_mode: Optional[str] = None):
        
        super().__init__()
        
        self.context = context
        self.max_steps = max_steps
        self.max_time_seconds = max_time_seconds
        self.render_mode = render_mode
        self.fast_mode = fast_mode
        
        # Action space: 3 discrete actions
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 5-dimensional continuous vector [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        
        # Internal state
        self.current_schedule: Optional[List[Tuple[int, int]]] = None
        self.current_step = 0
        self.start_time = 0.0
        self.stagnation_counter = 0
        
        # Tracking metrics
        self.prev_hard_conflicts = 0
        self.prev_soft_score = 0
        self.best_hard_conflicts = float('inf')
        self.best_soft_score = float('inf')
        
        # Reference values for normalization
        self.initial_hard_conflicts = 0
        self.initial_soft_score = 0
        self.max_soft_score = 10000  # Approximate upper bound
        
        # GA state (for incremental execution)
        self.ga: Optional[ScheduleGA] = None
        self.ga_population = None
        
        # Action costs
        self.action_costs = {0: 2, 1: 5, 2: 1}
        
    def _get_obs(self) -> np.ndarray:
        """Generate observation vector."""
        # 1. Hard conflicts ratio
        if self.initial_hard_conflicts > 0:
            hard_ratio = min(1.0, self.prev_hard_conflicts / self.initial_hard_conflicts)
        else:
            hard_ratio = 0.0
            
        # 2. Soft penalty ratio
        soft_ratio = min(1.0, max(0.0, self.prev_soft_score) / self.max_soft_score)
        
        # 3. Stagnation (normalized by max steps)
        stagnation = min(1.0, self.stagnation_counter / 10.0)
        
        # 4. Budget used
        budget_used = min(1.0, self.current_step / self.max_steps)
        
        # 5. Current validity (0 or 1)
        validity = 0.0 if self.prev_hard_conflicts > 0 else 1.0
        
        return np.array([hard_ratio, soft_ratio, stagnation, budget_used, validity], dtype=np.float32)
    
    def _calculate_potential(self) -> float:
        """Calculate potential function for reward shaping."""
        # Potential = -hard_conflicts * weight - soft_score * weight
        potential = -self.prev_hard_conflicts * 10.0 - max(0, self.prev_soft_score) * 0.01
        return potential
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Initialize random schedule
        self.current_schedule = self._generate_random_schedule()
        
        # Calculate initial metrics
        self.initial_hard_conflicts = self._count_hard_conflicts(self.current_schedule)
        self.initial_soft_score = evaluate_soft_score(self.context, self.current_schedule)
        
        self.prev_hard_conflicts = self.initial_hard_conflicts
        self.prev_soft_score = self.initial_soft_score
        self.best_hard_conflicts = self.initial_hard_conflicts
        self.best_soft_score = self.initial_soft_score
        
        # Reset counters
        self.current_step = 0
        self.start_time = time.time()
        self.stagnation_counter = 0
        
        # Initialize GA
        self.ga = ScheduleGA(self.context)
        self.ga_population = None
        
        return self._get_obs(), {}
    
    def _generate_random_schedule(self) -> List[Tuple[int, int]]:
        """Generate a completely random schedule."""
        import random
        schedule = []
        for event in self.context.events:
            slot = random.choice(self.context.slots)
            room = random.choice(self.context.rooms)
            schedule.append((slot.id, room.id))
        return schedule
    
    def _count_hard_conflicts(self, schedule: List[Tuple[int, int]]) -> int:
        """Count number of hard conflicts in schedule."""
        slot_to_events = {}
        conflicts = 0
        
        for idx, (slot_id, room_id) in enumerate(schedule):
            if slot_id not in slot_to_events:
                slot_to_events[slot_id] = []
            slot_to_events[slot_id].append(idx)
            
        for idx, (slot_id, room_id) in enumerate(schedule):
            event = self.context.events[idx]
            room = self.context.rooms_dict[room_id]
            
            # Capacity and type check
            total_students = sum(self.context.groups_dict[gid].size for gid in event.group_ids)
            if room.capacity < total_students:
                conflicts += 1
            if room.type != event.room_type_required:
                conflicts += 1
                
            # Overlap check
            for other_idx in slot_to_events.get(slot_id, []):
                if other_idx <= idx:
                    continue
                other_event = self.context.events[other_idx]
                _, other_room_id = schedule[other_idx]
                
                if room_id == other_room_id:
                    conflicts += 1
                if event.teacher_id == other_event.teacher_id:
                    conflicts += 1
                if set(event.group_ids).intersection(set(other_event.group_ids)):
                    conflicts += 1
                    
        return conflicts
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Calculate potential before action
        potential_before = self._calculate_potential()
        
        # Execute action
        info = {'action': action}
        
        if self.fast_mode:
            # Fast mode: minimal iterations for quick training
            if action == 0:
                self._run_ga_step(n_gen=3)
                info['action_name'] = 'GA'
            elif action == 1:
                self._run_cpsat_repair()
                info['action_name'] = 'CP-SAT'
            elif action == 2:
                self._run_sa_step(n_iter=30)
                info['action_name'] = 'SA'
        else:
            # Full mode: more thorough optimization
            if action == 0:
                self._run_ga_step(n_gen=10)
                info['action_name'] = 'GA'
            elif action == 1:
                self._run_cpsat_repair()
                info['action_name'] = 'CP-SAT'
            elif action == 2:
                self._run_sa_step(n_iter=100)
                info['action_name'] = 'SA'
        
        # Update metrics
        new_hard_conflicts = self._count_hard_conflicts(self.current_schedule)
        new_soft_score = evaluate_soft_score(self.context, self.current_schedule)
        
        self.prev_hard_conflicts = new_hard_conflicts
        self.prev_soft_score = new_soft_score
        
        # Update best tracking
        if new_hard_conflicts < self.best_hard_conflicts or \
           (new_hard_conflicts == self.best_hard_conflicts and new_soft_score < self.best_soft_score):
            self.best_hard_conflicts = new_hard_conflicts
            self.best_soft_score = new_soft_score
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        # Calculate reward (potential-based shaping)
        potential_after = self._calculate_potential()
        reward = potential_after - potential_before - self.action_costs[action] * 0.1
        
        # Bonus for achieving validity
        if new_hard_conflicts == 0 and self.best_hard_conflicts > 0:
            reward += 50.0
            info['achieved_validity'] = True
        
        # Check termination conditions
        elapsed_time = time.time() - self.start_time
        truncated = (self.current_step >= self.max_steps or elapsed_time >= self.max_time_seconds)
        terminated = (new_hard_conflicts == 0 and new_soft_score <= 0)
        
        info.update({
            'hard_conflicts': new_hard_conflicts,
            'soft_score': new_soft_score,
            'stagnation': self.stagnation_counter,
            'elapsed_time': elapsed_time,
            'truncated': truncated
        })
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _run_ga_step(self, n_gen: int = 10):
        """Run GA for a few generations - optimized for RL training."""
        if self.ga_population is None:
            # First run - initialize population with smaller size
            pop, best_ind = self.ga.run(ngen=n_gen, pop_size=20)  # Reduced from 50
            self.ga_population = pop
        else:
            # Continue evolution - simplified for speed
            for gen in range(n_gen):
                # Select and clone
                offspring = self.ga.toolbox.select(self.ga_population, len(self.ga_population))
                offspring = list(map(self.ga.toolbox.clone, offspring))
                
                # Apply crossover and mutation (simplified)
                for i in range(1, len(offspring), 2):
                    if len(offspring) > i + 1:
                        self.ga.toolbox.mate(offspring[i-1], offspring[i])
                        del offspring[i-1].fitness.values
                        del offspring[i].fitness.values
                
                for i in range(len(offspring)):
                    if random.random() < 0.2:
                        self.ga.toolbox.mutate(offspring[i])
                        del offspring[i].fitness.values
                
                # Evaluate only invalid individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.ga.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                self.ga_population = offspring
        
        # Update current schedule with best individual
        from deap import tools
        hof = tools.HallOfFame(1)
        hof.update(self.ga_population)
        if len(hof) > 0:
            self.current_schedule = list(hof[0])
    
    def _run_cpsat_repair(self):
        """Run CP-SAT repair on current schedule."""
        try:
            self.current_schedule = repair_with_cpsat(self.context, self.current_schedule)
        except Exception as e:
            # If CP-SAT fails, keep current schedule
            pass
    
    def _run_sa_step(self, n_iter: int = 100):
        """Run SA for a few iterations."""
        self.current_schedule = optimize_soft_constraints(
            self.context,
            self.current_schedule,
            initial_temp=100.0,
            cooling_rate=0.99,
            min_temp=50.0,  # Higher min_temp for short runs
            iterations_per_temp=n_iter // 10
        )
    
    def get_current_schedule(self) -> List[Tuple[int, int]]:
        """Return the current best schedule."""
        return self.current_schedule
    
    def render(self):
        """Render the current state."""
        if self.render_mode == 'human':
            print(f"Step {self.current_step}/{self.max_steps}")
            print(f"  Hard Conflicts: {self.prev_hard_conflicts}")
            print(f"  Soft Score: {self.prev_soft_score}")
            print(f"  Stagnation: {self.stagnation_counter}")
