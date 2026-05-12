import random
import math
import copy
from typing import List, Tuple, Dict, Set
from .models import ScheduleContext

def evaluate_soft_score(context: ScheduleContext, chromosome: List[Tuple[int, int]]) -> int:
    """
    Calculates the penalty for soft constraints (windows for students and teachers).
    Lower score is better.
    """
    penalty = 0
    
    # 1. Evaluate teacher preferences
    for idx, (slot_id, _) in enumerate(chromosome):
        event = context.events[idx]
        teacher = context.teachers_dict[event.teacher_id]
        if slot_id in teacher.preferences:
            penalty -= teacher.preferences[slot_id]
            
    # Calculate windows (empty slots between classes on the same day)
    
    # Map entities to their daily slots
    # day -> group_id -> list of time indices
    group_daily_schedule: Dict[int, Dict[int, List[int]]] = {}
    teacher_daily_schedule: Dict[int, Dict[int, List[int]]] = {}
    
    for idx, (slot_id, _) in enumerate(chromosome):
        event = context.events[idx]
        slot = context.slots_dict[slot_id]
        day = slot.day
        time_idx = slot.time
        
        # Teacher
        if day not in teacher_daily_schedule:
            teacher_daily_schedule[day] = {}
        if event.teacher_id not in teacher_daily_schedule[day]:
            teacher_daily_schedule[day][event.teacher_id] = []
        teacher_daily_schedule[day][event.teacher_id].append(time_idx)
        
        # Groups
        if day not in group_daily_schedule:
            group_daily_schedule[day] = {}
        for gid in event.group_ids:
            if gid not in group_daily_schedule[day]:
                group_daily_schedule[day][gid] = []
            group_daily_schedule[day][gid].append(time_idx)
            
    # Calculate penalty for windows
    window_penalty_weight = 10 # penalty per window slot
    
    def calculate_windows(daily_schedule: Dict[int, Dict[int, List[int]]], is_group: bool) -> int:
        windows_penalty = 0
        
        # Track days with classes to penalize empty days
        all_days = set(range(1, 6)) # Assuming 5 working days (Mon-Fri)
        
        for entity_id in set(e_id for day in daily_schedule.values() for e_id in day.keys()):
            days_with_classes = set()
            
            for day in all_days:
                if day in daily_schedule and entity_id in daily_schedule[day]:
                    times = daily_schedule[day][entity_id]
                    days_with_classes.add(day)
                    
                    if len(times) > 1:
                        times.sort()
                        for i in range(len(times) - 1):
                            gap = times[i+1] - times[i] - 1
                            if gap > 0:
                                # Non-linear penalty: gap of 1 = 10, gap of 2 = 30, gap of 3 = 60
                                # Formula: gap * (gap + 1) / 2 * base_weight
                                windows_penalty += int((gap * (gap + 1)) / 2 * window_penalty_weight)
                    elif len(times) == 1:
                        # Minor penalty for having only 1 class in a day (inefficient commuting)
                        windows_penalty += 5
                        
                    # --- Morning Compactness Penalty ---
                    # We want to encourage classes to start as early as possible.
                    # Penalize based on the starting time of the FIRST class of the day.
                    first_class_time = min(times)
                    # If first class starts at pair 1, penalty is 0. 
                    # If it starts at pair 2, penalty is 2. If pair 5, penalty is 8.
                    # We use a minor penalty weight here to not override the window fixing.
                    windows_penalty += (first_class_time - 1) * 2
                        
            # If it's a student group, having completely empty days in the middle of the week is usually bad
            # (unless it's an intended study day, but we penalize it here to spread classes evenly).
            if is_group:
                empty_days = all_days - days_with_classes
                # Heavy penalty for each completely empty day to force spreading out the load
                windows_penalty += len(empty_days) * 50
                
        return windows_penalty

    penalty += calculate_windows(group_daily_schedule, is_group=True)
    penalty += calculate_windows(teacher_daily_schedule, is_group=False)
    
    return penalty

def check_hard_conflicts(context: ScheduleContext, chromosome: List[Tuple[int, int]]) -> bool:
    """
    Checks if a chromosome has any hard conflicts. Returns True if conflicts exist, False if valid.
    """
    slot_to_events: Dict[int, List[int]] = {}
    for idx, (slot_id, room_id) in enumerate(chromosome):
        if slot_id not in slot_to_events:
            slot_to_events[slot_id] = []
        slot_to_events[slot_id].append(idx)
        
    for idx, (slot_id, room_id) in enumerate(chromosome):
        event = context.events[idx]
        room = context.rooms_dict[room_id]
        
        # Room constraints
        total_students = sum([context.groups_dict[gid].size for gid in event.group_ids])
        if room.capacity < total_students or room.type != event.room_type_required:
            return True
            
        # Overlap constraints
        events_in_same_slot = slot_to_events.get(slot_id, [])
        for other_idx in events_in_same_slot:
            if other_idx <= idx:
                continue
                
            other_event = context.events[other_idx]
            _, other_room_id = chromosome[other_idx]
            
            if room_id == other_room_id:
                return True
            if event.teacher_id == other_event.teacher_id:
                return True
            if set(event.group_ids).intersection(set(other_event.group_ids)):
                return True
                
    return False

def optimize_soft_constraints(context: ScheduleContext, initial_schedule: List[Tuple[int, int]], 
                              initial_temp: float = 100.0, cooling_rate: float = 0.98, min_temp: float = 0.1,
                              iterations_per_temp: int = 100) -> List[Tuple[int, int]]:
    """
    Implements Simulated Annealing to minimize soft constraints (windows, preferences)
    while strictly preserving hard constraints. Uses Markov chains (multiple iterations per temperature).
    """
    current_schedule = copy.deepcopy(initial_schedule)
    current_score = evaluate_soft_score(context, current_schedule)
    
    best_schedule = copy.deepcopy(current_schedule)
    best_score = current_score
    
    temp = initial_temp
    
    num_events = len(context.events)
    all_slot_ids = [s.id for s in context.slots]
    all_room_ids = [r.id for r in context.rooms]
    
    # Pre-calculate valid rooms per event to speed up random selections
    valid_rooms_per_event = {}
    for idx, event in enumerate(context.events):
        total_students = sum([context.groups_dict[gid].size for gid in event.group_ids])
        valid_rooms = [r.id for r in context.rooms if r.capacity >= total_students and r.type == event.room_type_required]
        valid_rooms_per_event[idx] = valid_rooms

    while temp > min_temp:
        for _ in range(iterations_per_temp):
            # Generate neighbor
            neighbor = copy.deepcopy(current_schedule)
            
            operation = random.random()
            
            if operation < 0.4:
                # Operation 1: Move single event to empty slot/room (Intelligent move)
                event_idx = random.randint(0, num_events - 1)
                valid_rooms = valid_rooms_per_event[event_idx]
                
                if not valid_rooms:
                    continue
                    
                # Try to find a valid slot and room (limit attempts)
                found_valid = False
                for _ in range(20):
                    new_slot_id = random.choice(all_slot_ids)
                    new_room_id = random.choice(valid_rooms) # Only pick valid rooms!
                    
                    original_pos = neighbor[event_idx]
                    neighbor[event_idx] = (new_slot_id, new_room_id)
                    
                    if not check_hard_conflicts(context, neighbor):
                        found_valid = True
                        break
                    else:
                        neighbor[event_idx] = original_pos
                        
            elif operation < 0.8:
                # Operation 2: Same-day swap (Highly likely to be valid and useful for windows)
                idx1 = random.randint(0, num_events - 1)
                slot1_id, room1_id = neighbor[idx1]
                slot1 = context.slots_dict[slot1_id]
                
                # Find another event on the SAME day
                same_day_events = [i for i, (s_id, _) in enumerate(neighbor) if context.slots_dict[s_id].day == slot1.day and i != idx1]
                
                if same_day_events:
                    idx2 = random.choice(same_day_events)
                    
                    # Swap their slots AND rooms (effectively swapping the time blocks)
                    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                    
                    if check_hard_conflicts(context, neighbor):
                        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                        
            else:
                # Operation 3: General Swap two events
                idx1 = random.randint(0, num_events - 1)
                idx2 = random.randint(0, num_events - 1)
                
                if idx1 != idx2:
                    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                    
                    if check_hard_conflicts(context, neighbor):
                        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                        
            # Evaluate neighbor
            neighbor_score = evaluate_soft_score(context, neighbor)
            
            # Decide whether to accept
            if neighbor_score < current_score:
                current_schedule = neighbor
                current_score = neighbor_score
                
                if neighbor_score < best_score:
                    best_schedule = copy.deepcopy(neighbor)
                    best_score = neighbor_score
            else:
                delta = neighbor_score - current_score
                acceptance_probability = math.exp(-delta / temp)
                if random.random() < acceptance_probability:
                    current_schedule = neighbor
                    current_score = neighbor_score
                    
        # Cool down
        temp *= cooling_rate
        
    return best_schedule
