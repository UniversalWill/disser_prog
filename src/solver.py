from typing import List, Tuple
from .models import ScheduleContext
from .phase1_ga import ScheduleGA
from .phase2_cpsat import repair_with_cpsat
from .phase3_sa import optimize_soft_constraints, check_hard_conflicts

class HybridScheduler:
    def __init__(self, context: ScheduleContext):
        self.context = context

    def solve(self) -> Tuple[List[Tuple[int, int]], dict]:
        """
        Runs the full 3-phase hybrid scheduling pipeline.
        Returns the final schedule and a dictionary with execution metrics.
        """
        metrics = {}
        
        print("=== Phase 1: Global Search (GA) ===")
        ga = ScheduleGA(self.context)
        # Using fewer generations for faster testing, but in real use, use 200+
        pop, best_ga_ind = ga.run(ngen=200, pop_size=100) 
        
        ga_hard_conflicts, ga_soft_penalty = best_ga_ind.fitness.values
        metrics['ga_hard_conflicts'] = ga_hard_conflicts
        metrics['ga_soft_penalty'] = ga_soft_penalty
        
        print(f"GA Result -> Hard Conflicts: {ga_hard_conflicts}, Soft Penalty: {ga_soft_penalty}")
        
        # Convert DEAP individual to pure list of tuples
        current_schedule = list(best_ga_ind)
        
        # Phase 2: CP-SAT Repair
        if ga_hard_conflicts > 0 or check_hard_conflicts(self.context, current_schedule):
            print("\n=== Phase 2: Intelligent Repair (CP-SAT) ===")
            print("Hard conflicts detected. Initiating repair...")
            current_schedule = repair_with_cpsat(self.context, current_schedule)
            
            # Verify repair success
            has_conflicts = check_hard_conflicts(self.context, current_schedule)
            metrics['cpsat_success'] = not has_conflicts
            if has_conflicts:
                print("WARNING: CP-SAT could not resolve all hard conflicts.")
            else:
                print("CP-SAT successfully repaired the schedule!")
        else:
            print("\n=== Phase 2: Skipped (No hard conflicts) ===")
            metrics['cpsat_success'] = True
            
        # Phase 3: Simulated Annealing
        if not check_hard_conflicts(self.context, current_schedule):
            print("\n=== Phase 3: Local Optimization (Simulated Annealing) ===")
            from src.phase3_sa import evaluate_soft_score
            initial_score = evaluate_soft_score(self.context, current_schedule)
            print(f"Initial Soft Score before SA: {initial_score}")
            
            current_schedule = optimize_soft_constraints(
                self.context, 
                current_schedule,
                initial_temp=100.0,
                cooling_rate=0.95,
                min_temp=0.1,
                iterations_per_temp=500
            )
            final_score = evaluate_soft_score(self.context, current_schedule)
            print(f"Final Soft Score after SA: {final_score}")
            print("Simulated Annealing complete.")
        else:
            print("\n=== Phase 3: Skipped (Schedule is invalid) ===")
            
        metrics['final_valid'] = not check_hard_conflicts(self.context, current_schedule)
        
        return current_schedule, metrics
