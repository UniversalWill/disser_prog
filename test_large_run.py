import random
from src.models import ScheduleContext, Teacher, Group, Room, Slot, Event, RoomType, WeekType
from src.solver import HybridScheduler
from src.formatter import generate_text_schedule, generate_json_schedule, generate_html_table
from src.validator import ScheduleValidator

def create_large_mock_data() -> ScheduleContext:
    random.seed(42) # For reproducibility
    
    # 1. 10 Teachers
    teachers = []
    for i in range(1, 11):
        prefs = {}
        # Give some random preferences
        if random.random() > 0.5:
            prefs[random.randint(1, 30)] = 10 # Prefer some slot
        if random.random() > 0.5:
            prefs[random.randint(1, 30)] = -10 # Avoid some slot
        teachers.append(Teacher(id=i, name=f"Преподаватель_{i}", max_hours=40, preferences=prefs))

    # 2. 10 Groups
    groups = []
    for i in range(1, 11):
        groups.append(Group(id=i, name=f"Группа-{i}", size=random.randint(15, 30)))

    # 3. 15 Rooms (10 Lecture, 5 Lab)
    rooms = []
        # For 2 groups, max size is ~60 (30+30)
        # For labs, we need to ensure capacity is at least max group size (30)
    for i in range(1, 11):
        rooms.append(Room(id=i, capacity=80, type=RoomType.LECTURE, building="Главный корпус"))
    for i in range(11, 16):
        rooms.append(Room(id=i, capacity=40, type=RoomType.LAB, building="Лабораторный корпус"))

    # 4. Slots: 5 days (Mon-Fri), 5 pairs a day = 25 slots
    slots = []
    slot_id = 1
    for day in range(1, 6): # Removed Saturday
        for time in range(1, 6):
            slots.append(Slot(id=slot_id, day=day, time=time))
            slot_id += 1

    # 5. 100 Events
    events = []
    subjects = ["Математика", "Физика", "Программирование", "Базы данных", "Сети", "История", "Философия", "Английский"]
    
    for i in range(1, 101):
        subject = random.choice(subjects)
        teacher = random.choice(teachers)
        
        # 80% single group, 20% combined lecture for two groups
        if random.random() < 0.2:
            g_ids = random.sample([g.id for g in groups], 2)
            r_type = RoomType.LECTURE
        else:
            g_ids = [random.choice(groups).id]
            # Labs usually for smaller/single groups
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

if __name__ == "__main__":
    print("Создание контекста (100 занятий, 10 групп, 10 преподавателей, 25 слотов)...")
    context = create_large_mock_data()
    
    print("\nИнициализация гибридного планировщика...")
    scheduler = HybridScheduler(context)
    
    final_schedule, metrics = scheduler.solve()
    
    print("\n=== ВЕРИФИКАЦИЯ РАСПИСАНИЯ ===")
    validator = ScheduleValidator(context, final_schedule)
    is_valid, errors, v_metrics = validator.validate()
    
    if is_valid:
        print("✅ Расписание полностью ВАЛИДНО (жестких конфликтов нет).")
    else:
        print(f"❌ НАЙДЕНО {len(errors)} ЖЕСТКИХ КОНФЛИКТОВ:")
        for err in errors[:10]: # Print top 10 errors
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... и еще {len(errors) - 10} ошибок.")
            
    print("\nМетрики качества (Soft Constraints):")
    print(f"  - Окна у групп (сумма слотов): {v_metrics['group_windows']}")
    print(f"  - Окна у преподавателей (сумма слотов): {v_metrics['teacher_windows']}")
    print(f"  - Пустые дни у групп (сумма по всем группам): {v_metrics['group_empty_days']}")
    print(f"  - Поздние начала пар (штрафной балл): {v_metrics['late_starts_penalty']}")
    print(f"  - Удовлетворенность преподавателей: {v_metrics['teacher_preference_score']}")

    print("\nГенерация отчетов (Text, JSON, HTML)...")
    
    # 1. Text
    text_output = generate_text_schedule(context, final_schedule)
    with open("schedule_output.txt", "w", encoding="utf-8") as f:
        f.write(text_output)
        
    # 2. JSON
    json_output = generate_json_schedule(context, final_schedule)
    with open("schedule_output.json", "w", encoding="utf-8") as f:
        f.write(json_output)
        
    # 3. HTML
    html_output = generate_html_table(json_output, context)
    with open("schedule_output.html", "w", encoding="utf-8") as f:
        f.write(html_output)
        
    print("Расписание успешно сохранено в файлы:")
    print(" - schedule_output.txt")
    print(" - schedule_output.json")
    print(" - schedule_output.html")
    
    # Print a preview
    print("\nПревью JSON расписания (первые 20 строк):")
    lines = json_output.split("\n")
    print("\n".join(lines[:20]))
    print("...")
