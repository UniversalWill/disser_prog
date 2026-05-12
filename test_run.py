from src.models import ScheduleContext, Teacher, Group, Room, Slot, Event, RoomType, WeekType
from src.solver import HybridScheduler

def create_mock_data() -> ScheduleContext:
    # 2 Teachers
    t1 = Teacher(id=1, name="Ivanov I.I.", max_hours=20, preferences={1: 10, 2: -5}) # Prefers slot 1, avoids slot 2
    t2 = Teacher(id=2, name="Petrov P.P.", max_hours=15, preferences={})

    # 2 Groups
    g1 = Group(id=1, name="CS-101", size=25)
    g2 = Group(id=2, name="CS-102", size=20)

    # 3 Rooms (2 Lecture, 1 Lab)
    r1 = Room(id=1, capacity=50, type=RoomType.LECTURE, building="Main")
    r2 = Room(id=2, capacity=30, type=RoomType.LECTURE, building="Main")
    r3 = Room(id=3, capacity=25, type=RoomType.LAB, building="Annex") # Ensure capacity is enough for Group 2 (size 20)

    # 4 Slots (2 days, 2 pairs each)
    s1 = Slot(id=1, day=1, time=1)
    s2 = Slot(id=2, day=1, time=2)
    s3 = Slot(id=3, day=2, time=1)
    s4 = Slot(id=4, day=2, time=2)

    # 4 Events
    e1 = Event(id=1, subject_name="Math Lecture", teacher_id=1, group_ids=[1, 2], duration=1, room_type_required=RoomType.LECTURE)
    e2 = Event(id=2, subject_name="Physics Lecture", teacher_id=2, group_ids=[1, 2], duration=1, room_type_required=RoomType.LECTURE)
    e3 = Event(id=3, subject_name="Math Practice", teacher_id=1, group_ids=[1], duration=1, room_type_required=RoomType.LECTURE)
    e4 = Event(id=4, subject_name="Physics Lab", teacher_id=2, group_ids=[2], duration=1, room_type_required=RoomType.LAB)

    return ScheduleContext(
        teachers=[t1, t2],
        groups=[g1, g2],
        rooms=[r1, r2, r3],
        slots=[s1, s2, s3, s4],
        events=[e1, e2, e3, e4]
    )

if __name__ == "__main__":
    context = create_mock_data()
    print("Initializing Hybrid Scheduler with mock data...")
    scheduler = HybridScheduler(context)
    
    final_schedule, metrics = scheduler.solve()
    
    print("\n=== Final Results ===")
    print(f"Valid schedule found: {metrics['final_valid']}")
    
    print("\nSchedule Details:")
    for idx, (slot_id, room_id) in enumerate(final_schedule):
        event = context.events[idx]
        slot = context.slots_dict[slot_id]
        room = context.rooms_dict[room_id]
        print(f"Event {event.subject_name} (Teacher: {event.teacher_id}, Groups: {event.group_ids}) -> Day {slot.day}, Pair {slot.time} in Room {room.id} ({room.type.value})")
