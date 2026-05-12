import time
import os
import pandas as pd
from typing import Dict, Any

from src.data_generator import FacultyDatasetGenerator
from src.solver import HybridScheduler
from src.rl_solver import RLScheduler
from src.validator import ScheduleValidator

def run_benchmark():
    print("="*80)
    print("FACULTY SCALE BENCHMARK: Hybrid vs RL Scheduler")
    print("="*80)
    
    # 1. Generate Dataset
    generator = FacultyDatasetGenerator(seed=42)
    
    context = generator.generate_faculty_dataset()
    print("\nDataset ready. Starting benchmark...\n")
    
    results = []
    
    # 2. Run Static Hybrid Scheduler
    print("--- 1. STATIC HYBRID SCHEDULER ---")
    start_time = time.time()
    hybrid_scheduler = HybridScheduler(context)
    hybrid_schedule, hybrid_metrics = hybrid_scheduler.solve()
    hybrid_time = time.time() - start_time
    
    validator_h = ScheduleValidator(context, hybrid_schedule)
    is_valid_h, errors_h, metrics_h = validator_h.validate()
    
    print(f"\nHybrid Results: Time = {hybrid_time:.2f}s, Valid = {is_valid_h}, Hard = {metrics_h['hard_conflicts']}, Soft = {metrics_h['group_windows'] + metrics_h['teacher_windows']}")
    
    results.append({
        'Algorithm': 'Static Hybrid (GA->CP-SAT->SA)',
        'Execution Time (s)': round(hybrid_time, 2),
        'Is Valid': is_valid_h,
        'Hard Conflicts': metrics_h['hard_conflicts'],
        'Group Windows': metrics_h['group_windows'],
        'Teacher Windows': metrics_h['teacher_windows'],
        'Empty Days': metrics_h['group_empty_days'],
        'Late Starts Penalty': metrics_h['late_starts_penalty'],
        'Teacher Sat.': metrics_h['teacher_preference_score']
    })
    
    # 3. Run RL Scheduler
    print("\n--- 2. RL HYPER-HEURISTIC SCHEDULER ---")
    model_path = "rl_models/rl_scheduler_final.zip"
    if not os.path.exists(model_path):
        print(f"Skipping RL test: Model not found at {model_path}")
    else:
        start_time = time.time()
        # Give it a bit more time for the faculty scale problem
        rl_scheduler = RLScheduler(context, model_path, max_steps=20, max_time_seconds=600.0)
        rl_schedule, rl_metrics = rl_scheduler.solve()
        rl_time = time.time() - start_time
        
        validator_rl = ScheduleValidator(context, rl_schedule)
        is_valid_rl, errors_rl, metrics_rl = validator_rl.validate()
        
        print(f"\nRL Results: Time = {rl_time:.2f}s, Valid = {is_valid_rl}, Hard = {metrics_rl['hard_conflicts']}, Soft = {metrics_rl['group_windows'] + metrics_rl['teacher_windows']}")
        
        results.append({
            'Algorithm': 'RL Agent (PPO)',
            'Execution Time (s)': round(rl_time, 2),
            'Is Valid': is_valid_rl,
            'Hard Conflicts': metrics_rl['hard_conflicts'],
            'Group Windows': metrics_rl['group_windows'],
            'Teacher Windows': metrics_rl['teacher_windows'],
            'Empty Days': metrics_rl['group_empty_days'],
            'Late Starts Penalty': metrics_rl['late_starts_penalty'],
            'Teacher Sat.': metrics_rl['teacher_preference_score']
        })
        
    # 4. Save and display metrics
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
