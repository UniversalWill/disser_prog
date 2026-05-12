from typing import List, Tuple
from ortools.sat.python import cp_model
from .models import ScheduleContext

def repair_with_cpsat(context: ScheduleContext, ga_chromosome: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Takes a potentially invalid chromosome from GA and uses CP-SAT to find a valid
    schedule that is as close as possible to the GA schedule.
    Returns the repaired chromosome.
    """
    model = cp_model.CpModel()
    
    num_events = len(context.events)
    all_slot_ids = [s.id for s in context.slots]
    all_room_ids = [r.id for r in context.rooms]
    
    # Create variables
    # We will use indices in the lists as variables to easily enforce constraints
    # slot_vars[i] represents the slot index for event i
    # room_vars[i] represents the room index for event i
    
    slot_vars = []
    room_vars = []
    
    for i in range(num_events):
        slot_vars.append(model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(all_slot_ids), f'slot_e{i}'
        ))
        
        # Room domain must only include rooms that satisfy capacity and type requirements
        event = context.events[i]
        total_students = sum([context.groups_dict[gid].size for gid in event.group_ids])
        
        valid_room_ids = [
            r.id for r in context.rooms 
            if r.capacity >= total_students and r.type == event.room_type_required
        ]
        
        if not valid_room_ids:
            # If no valid rooms exist for an event, the problem is infeasible
            raise ValueError(f"No valid room found for event {event.id}")
            
        room_vars.append(model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(valid_room_ids), f'room_e{i}'
        ))

    # Add constraints
    for i in range(num_events):
        event_i = context.events[i]
        for j in range(i + 1, num_events):
            event_j = context.events[j]
            
            # Check if events share resources
            share_teacher = (event_i.teacher_id == event_j.teacher_id)
            share_group = len(set(event_i.group_ids).intersection(set(event_j.group_ids))) > 0
            
            if share_teacher or share_group:
                # If they share a teacher or a group, they MUST be in different slots
                model.Add(slot_vars[i] != slot_vars[j])
            else:
                # If they don't share teacher/group, they CAN be in the same slot,
                # BUT if they are in the same slot, they MUST be in different rooms.
                
                # We need an indicator variable to represent (slot_vars[i] == slot_vars[j])
                same_slot = model.NewBoolVar(f'same_slot_{i}_{j}')
                
                # Link indicator variable to the equality condition
                model.Add(slot_vars[i] == slot_vars[j]).OnlyEnforceIf(same_slot)
                model.Add(slot_vars[i] != slot_vars[j]).OnlyEnforceIf(same_slot.Not())
                
                # If same_slot is true, then rooms must be different
                model.Add(room_vars[i] != room_vars[j]).OnlyEnforceIf(same_slot)

    # Add Hints and Objective
    # We want to minimize the differences from the GA solution
    difference_vars = []
    
    for i in range(num_events):
        ga_slot_id, ga_room_id = ga_chromosome[i]
        
        # Hinting
        model.AddHint(slot_vars[i], ga_slot_id)
        model.AddHint(room_vars[i], ga_room_id)
        
        # Objective: minimize distance between new slot/room and ga slot/room
        # Create abs difference variables
        slot_diff = model.NewIntVar(0, max(all_slot_ids), f'diff_slot_e{i}')
        model.Add(slot_diff >= slot_vars[i] - ga_slot_id)
        model.Add(slot_diff >= ga_slot_id - slot_vars[i])
        
        room_diff = model.NewIntVar(0, max(all_room_ids), f'diff_room_e{i}')
        model.Add(room_diff >= room_vars[i] - ga_room_id)
        model.Add(room_diff >= ga_room_id - room_vars[i])
        
        difference_vars.extend([slot_diff, room_diff])

    model.Minimize(sum(difference_vars))
    
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0 # Set a timeout
    
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        repaired_chromosome = []
        for i in range(num_events):
            repaired_chromosome.append(
                (solver.Value(slot_vars[i]), solver.Value(room_vars[i]))
            )
        return repaired_chromosome
    else:
        # If infeasible, we return the original (this might mean the constraints are too tight
        # and no valid schedule exists at all)
        print("CP-SAT could not find a feasible solution to repair the schedule.")
        return ga_chromosome
