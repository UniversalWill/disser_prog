from collections import defaultdict
from typing import List, Tuple, Dict, Any
from .models import ScheduleContext

class ScheduleValidator:
    """
    Validates a schedule against all hard constraints and calculates quality metrics (soft constraints).
    """
    def __init__(self, context: ScheduleContext, schedule: List[Tuple[int, int]]):
        self.context = context
        self.schedule = schedule
        self.errors = []
        self.metrics = {
            "hard_conflicts": 0,
            "teacher_preference_score": 0,
            "group_windows": 0,
            "teacher_windows": 0,
            "group_empty_days": 0,
            "late_starts_penalty": 0
        }

    def validate(self) -> Tuple[bool, List[str], Dict[str, Any]]:
        self._check_hard_constraints()
        self._calculate_metrics()
        self.metrics["hard_conflicts"] = len(self.errors)
        return len(self.errors) == 0, self.errors, self.metrics

    def _check_hard_constraints(self):
        if len(self.schedule) != len(self.context.events):
            self.errors.append(f"Несоответствие количества занятий: ожидалось {len(self.context.events)}, получено {len(self.schedule)}")

        slot_to_events = defaultdict(list)
        
        for idx, (slot_id, room_id) in enumerate(self.schedule):
            event = self.context.events[idx]
            room = self.context.rooms_dict[room_id]
            slot = self.context.slots_dict[slot_id]

            # 1. Capacity check
            total_students = sum(self.context.groups_dict[gid].size for gid in event.group_ids)
            if room.capacity < total_students:
                self.errors.append(f"[Вместимость] Занятие {event.id} (группы {event.group_ids}, {total_students} чел) в аудитории {room.id} (вмещает {room.capacity})")

            # 2. Room Type check
            if room.type != event.room_type_required:
                self.errors.append(f"[Тип аудитории] Занятие {event.id} требует {event.room_type_required.value}, но назначена {room.type.value}")

            slot_to_events[slot_id].append((event, room, slot))

        for slot_id, items in slot_to_events.items():
            seen_rooms = set()
            seen_teachers = set()
            seen_groups = set()

            for event, room, slot in items:
                # 3. Room double-booking
                if room.id in seen_rooms:
                    self.errors.append(f"[Накладка] Аудитория {room.id} занята дважды в слот {slot_id} (день {slot.day}, пара {slot.time})")
                seen_rooms.add(room.id)

                # 4. Teacher double-booking
                if event.teacher_id in seen_teachers:
                    self.errors.append(f"[Накладка] Преподаватель {event.teacher_id} ведет два занятия в слот {slot_id} (день {slot.day}, пара {slot.time})")
                seen_teachers.add(event.teacher_id)

                # 5. Group double-booking
                for gid in event.group_ids:
                    if gid in seen_groups:
                        self.errors.append(f"[Накладка] Группа {gid} имеет два занятия в слот {slot_id} (день {slot.day}, пара {slot.time})")
                    seen_groups.add(gid)

    def _calculate_metrics(self):
        group_daily_schedule = defaultdict(lambda: defaultdict(list))
        teacher_daily_schedule = defaultdict(lambda: defaultdict(list))
        
        for idx, (slot_id, room_id) in enumerate(self.schedule):
            event = self.context.events[idx]
            slot = self.context.slots_dict[slot_id]
            teacher = self.context.teachers_dict[event.teacher_id]

            # Teacher preferences
            if slot_id in teacher.preferences:
                self.metrics["teacher_preference_score"] += teacher.preferences[slot_id]

            # Daily tracking
            teacher_daily_schedule[slot.day][teacher.id].append(slot.time)
            for gid in event.group_ids:
                group_daily_schedule[slot.day][gid].append(slot.time)

        all_days = set(range(1, 6)) # Assume 5 working days (Mon-Fri)

        # Calculate Group metrics
        all_groups = set(g.id for g in self.context.groups)
        for gid in all_groups:
            days_with_classes = set()
            for day in all_days:
                if day in group_daily_schedule and gid in group_daily_schedule[day]:
                    times = sorted(group_daily_schedule[day][gid])
                    days_with_classes.add(day)
                    
                    # Windows
                    if len(times) > 1:
                        for i in range(len(times) - 1):
                            gap = times[i+1] - times[i] - 1
                            if gap > 0:
                                self.metrics["group_windows"] += gap
                                
                    # Late starts
                    first_time = times[0]
                    if first_time > 1:
                        self.metrics["late_starts_penalty"] += (first_time - 1)
                        
            # Empty days
            empty_days = len(all_days - days_with_classes)
            self.metrics["group_empty_days"] += empty_days

        # Calculate Teacher metrics
        for teacher in self.context.teachers:
            for day in all_days:
                if day in teacher_daily_schedule and teacher.id in teacher_daily_schedule[day]:
                    times = sorted(teacher_daily_schedule[day][teacher.id])
                    if len(times) > 1:
                        for i in range(len(times) - 1):
                            gap = times[i+1] - times[i] - 1
                            if gap > 0:
                                self.metrics["teacher_windows"] += gap
