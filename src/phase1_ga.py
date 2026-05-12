import random
from typing import List, Tuple, Dict, Any
from deap import base, creator, tools, algorithms
from .models import ScheduleContext, Event, Room, Slot

# Fitness definition: (hard_conflicts, soft_penalty)
# We want to minimize both, so weights are negative.
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


class ScheduleGA:
    def __init__(self, context: ScheduleContext):
        self.context = context
        self.toolbox = base.Toolbox()
        self.setup_toolbox()

    def setup_toolbox(self):
        # Attribute generator
        # Generate a random (slot_id, room_id)
        def attr_generator():
            slot_id = random.choice(self.context.slots).id
            room_id = random.choice(self.context.rooms).id
            return (slot_id, room_id)

        self.toolbox.register("attr_slot_room", attr_generator)

        # Structure initializers
        # An individual is a list of (slot_id, room_id) with length = len(events)
        def init_individual(icls):
            return icls([attr_generator() for _ in range(len(self.context.events))])

        self.toolbox.register("individual", init_individual, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Operators
        self.toolbox.register("evaluate", self.evaluate)
        
        # We use cxUniform but adapt it for our list of tuples. 
        # cxUniform normally works on simple lists. Since our individual is a list of tuples,
        # cxUniform will swap the entire (slot_id, room_id) tuple between parents, which is correct.
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        
        # Mutation
        self.toolbox.register("mutate", self.mutate, indpb=0.2)
        
        # Selection
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def mutate(self, individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                # Randomly change either slot or room or both
                old_slot, old_room = individual[i]
                new_slot = old_slot
                new_room = old_room
                
                choice = random.random()
                if choice < 0.33:
                    new_slot = random.choice(self.context.slots).id
                elif choice < 0.66:
                    new_room = random.choice(self.context.rooms).id
                else:
                    new_slot = random.choice(self.context.slots).id
                    new_room = random.choice(self.context.rooms).id
                
                individual[i] = (new_slot, new_room)
        return individual,

    def evaluate(self, individual) -> Tuple[int, int]:
        hard_conflicts = 0
        soft_penalty = 0

        # Mappings to check conflicts
        # slot_id -> list of events happening in this slot
        slot_to_events: Dict[int, List[int]] = {}
        for idx, (slot_id, room_id) in enumerate(individual):
            if slot_id not in slot_to_events:
                slot_to_events[slot_id] = []
            slot_to_events[slot_id].append(idx)

        # Evaluate constraints
        for idx, (slot_id, room_id) in enumerate(individual):
            event = self.context.events[idx]
            room = self.context.rooms_dict[room_id]
            teacher = self.context.teachers_dict[event.teacher_id]
            
            # 1. Capacity Hard Constraint
            total_students = sum([self.context.groups_dict[gid].size for gid in event.group_ids])
            if room.capacity < total_students:
                hard_conflicts += 1
                
            # 2. Room Type Hard Constraint
            if room.type != event.room_type_required:
                hard_conflicts += 1

            # 3. Soft Constraints (Teacher Preferences)
            # If slot_id is in preferences, add its negative value to penalty (minimize penalty)
            # If weight is positive (preferred), penalty decreases. If negative (avoid), penalty increases.
            if slot_id in teacher.preferences:
                soft_penalty -= teacher.preferences[slot_id] 
            
            # Check overlap conflicts with other events in the same slot
            events_in_same_slot = slot_to_events.get(slot_id, [])
            for other_idx in events_in_same_slot:
                if other_idx <= idx:
                    continue # check pairs only once
                
                other_event = self.context.events[other_idx]
                _, other_room_id = individual[other_idx]

                # 4. Same Room overlap
                if room_id == other_room_id:
                    hard_conflicts += 1
                
                # 5. Same Teacher overlap
                if event.teacher_id == other_event.teacher_id:
                    hard_conflicts += 1
                    
                # 6. Same Group overlap
                if set(event.group_ids).intersection(set(other_event.group_ids)):
                    hard_conflicts += 1

        return hard_conflicts, soft_penalty

    def run(self, ngen: int = 200, pop_size: int = 100) -> Tuple[List[Any], Any]:
        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1) # Keep the best individual
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: (sum([v[0] for v in x])/len(x), sum([v[1] for v in x])/len(x)))
        stats.register("min", lambda x: (min([v[0] for v in x]), min([v[1] for v in x])))

        # Use eaSimple for evolution
        algorithms.eaSimple(
            pop, 
            self.toolbox, 
            cxpb=0.5, 
            mutpb=0.2, 
            ngen=ngen, 
            stats=stats, 
            halloffame=hof, 
            verbose=False
        )

        return pop, hof[0]
