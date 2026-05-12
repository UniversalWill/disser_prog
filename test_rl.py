import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ScheduleContext, Teacher, Group, Room, Slot, Event, RoomType, WeekType
from src.rl_solver import RLScheduler
from src.validator import ScheduleValidator
from src.formatter import generate_text_schedule, generate_json_schedule, generate_html_table
import random

def create_test_context() -> ScheduleContext:
    """Create a test scheduling context."""
    random.seed(123)  # For reproducibility
    
    # 10 Teachers
    teachers = []
    for i in range(1, 11):
        prefs = {}
        if random.random() > 0.5:
            prefs[random.randint(1, 25)] = 10
        if random.random() > 0.5:
            prefs[random.randint(1, 25)] = -10
        teachers.append(Teacher(id=i, name=f"Teacher_{i}", max_hours=40, preferences=prefs))
    
    # 10 Groups
    groups = []
    for i in range(1, 11):
        groups.append(Group(id=i, name=f"Group-{i}", size=random.randint(15, 30)))
    
    # Rooms
    rooms = []
    for i in range(1, 11):
        rooms.append(Room(id=i, capacity=80, type=RoomType.LECTURE, building="Main"))
    for i in range(11, 16):
        rooms.append(Room(id=i, capacity=40, type=RoomType.LAB, building="Lab"))
    
    # Slots (5 days, 5 pairs)
    slots = []
    slot_id = 1
    for day in range(1, 6):
        for time in range(1, 6):
            slots.append(Slot(id=slot_id, day=day, time=time))
            slot_id += 1
    
    # 100 Events
    events = []
    subjects = ["Math", "Physics", "Programming", "Databases", "Networks", "History", "Philosophy", "English"]
    
    for i in range(1, 101):
        subject = random.choice(subjects)
        teacher = random.choice(teachers)
        
        if random.random() < 0.2:
            g_ids = random.sample([g.id for g in groups], 2)
            r_type = RoomType.LECTURE
        else:
            g_ids = [random.choice(groups).id]
            r_type = random.choice([RoomType.LECTURE, RoomType.LAB])
        
        events.append(Event(
            id=i,
            subject_name=subject,
            teacher_id=teacher.id,
            group_ids=g_ids,
            duration=1,
            room_type_required=r_type
        ))
    
    return ScheduleContext(
        teachers=teachers,
        groups=groups,
        rooms=rooms,
        slots=slots,
        events=events
    )


def main():
    print("="*60)
    print("RL SCHEDULER TEST")
    print("="*60)
    
    # Check if model exists
    model_path = "rl_models/rl_scheduler_final.zip"
    
    if not os.path.exists(model_path):
        print("\nWARNING: Trained model not found!")
        print(f"Expected path: {model_path}")
        print("\nPlease train the model first:")
        print("  python train_rl.py")
        print("\nFalling back to static pipeline for demonstration...")
        
        # Use static pipeline
        from src.solver import HybridScheduler
        context = create_test_context()
        scheduler = HybridScheduler(context)
        schedule, metrics = scheduler.solve()
    else:
        print("\nLoading trained RL model...")
        context = create_test_context()
        scheduler = RLScheduler(context, model_path)
        schedule, metrics = scheduler.solve()
    
    # Validate
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    validator = ScheduleValidator(context, schedule)
    is_valid, errors, v_metrics = validator.validate()
    
    if is_valid:
        print("✅ Schedule is VALID (no hard conflicts)")
    else:
        print(f"❌ Found {len(errors)} hard conflicts:")
        for err in errors[:5]:
            print(f"  - {err}")
    
    print("\nQuality Metrics:")
    print(f"  - Group windows: {v_metrics['group_windows']}")
    print(f"  - Teacher windows: {v_metrics['teacher_windows']}")
    print(f"  - Empty days: {v_metrics['group_empty_days']}")
    print(f"  - Late starts penalty: {v_metrics['late_starts_penalty']}")
    print(f"  - Teacher satisfaction: {v_metrics['teacher_preference_score']}")
    
    # Generate outputs
    print("\nGenerating output files...")
    
    text_output = generate_text_schedule(context, schedule)
    with open("rl_schedule_output.txt", "w", encoding="utf-8") as f:
        f.write(text_output)
    
    json_output = generate_json_schedule(context, schedule)
    with open("rl_schedule_output.json", "w", encoding="utf-8") as f:
        f.write(json_output)
    
    html_output = generate_html_table(json_output, context)
    with open("rl_schedule_output.html", "w", encoding="utf-8") as f:
        f.write(html_output)
    
    print("Files saved: rl_schedule_output.txt/json/html")
    
    # Print action summary if RL was used
    if 'actions_taken' in metrics:
        print("\n" + "="*60)
        print("ACTION SUMMARY")
        print("="*60)
        from collections import Counter
        action_counts = Counter(metrics['actions_taken'])
        for action, count in action_counts.items():
            print(f"  {action}: {count} times")


if __name__ == "__main__":
    main()
