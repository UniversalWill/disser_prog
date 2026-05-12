import random
from typing import List, Dict
from .models import ScheduleContext, Teacher, Group, Room, Slot, Event, RoomType


class FacultyDatasetGenerator:
    """
    Generates realistic scheduling datasets at the scale of a university faculty.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random = random.Random(seed)

    def _generate_slots(self, days: int = 5, pairs_per_day: int = 6) -> List[Slot]:
        slots = []
        slot_id = 1
        for day in range(1, days + 1):
            for time in range(1, pairs_per_day + 1):
                slots.append(Slot(id=slot_id, day=day, time=time))
                slot_id += 1
        return slots

    def _generate_rooms(
        self, num_lecture: int, num_lab: int, num_seminar: int
    ) -> List[Room]:
        rooms = []
        room_id = 1

        # Big lecture halls (capacity 100-150)
        for _ in range(num_lecture):
            rooms.append(
                Room(
                    id=room_id,
                    capacity=self.random.choice([100, 120, 150]),
                    type=RoomType.LECTURE,
                    building="Main Building",
                )
            )
            room_id += 1

        # Computer Labs (capacity 20-35)
        for _ in range(num_lab):
            rooms.append(
                Room(
                    id=room_id,
                    capacity=self.random.choice([25, 30, 35]),
                    type=RoomType.LAB,
                    building="Lab Wing",
                )
            )
            room_id += 1

        # Seminar/Practice rooms (capacity 30-40)
        for _ in range(num_seminar):
            rooms.append(
                Room(
                    id=room_id,
                    capacity=self.random.choice([30, 35, 40]),
                    type=RoomType.LECTURE,
                    building="Main Building",
                )
            )
            room_id += 1

        return rooms

    def _generate_teachers(self, num_teachers: int, slots: List[Slot]) -> List[Teacher]:
        teachers = []
        for i in range(1, num_teachers + 1):
            prefs = {}
            # 30% of teachers have strong preferences for specific days
            if self.random.random() < 0.3:
                preferred_day = self.random.randint(1, 5)
                for s in slots:
                    if s.day == preferred_day:
                        prefs[s.id] = 15  # Strong preference
                    elif s.day == (preferred_day % 5) + 1:
                        prefs[s.id] = -20  # Strong avoidance of next day

            # 50% of teachers dislike the first pair
            if self.random.random() < 0.5:
                for s in slots:
                    if s.time == 1:
                        prefs[s.id] = -10

            max_hours = self.random.choice([20, 30, 40])
            teachers.append(
                Teacher(id=i, name=f"Prof. {i}", max_hours=max_hours, preferences=prefs)
            )

        return teachers

    def _generate_groups(self, courses: int, groups_per_course: int) -> List[Group]:
        groups = []
        group_id = 1
        for course in range(1, courses + 1):
            for g in range(1, groups_per_course + 1):
                size = self.random.randint(20, 30)
                groups.append(Group(id=group_id, name=f"CS-{course}0{g}", size=size))
                group_id += 1
        return groups

    def generate_faculty_dataset(self) -> ScheduleContext:
        """
        Generates a faculty-scale dataset:
        - 4 courses (years), 5 groups per course = 20 groups
        - ~40 teachers
        - ~25 rooms
        - ~400 events
        """
        self.random.seed(self.seed)

        # 1. Base entities
        slots = self._generate_slots(
            days=6, pairs_per_day=6
        )  # 6 days, 6 pairs max (36 slots)
        rooms = self._generate_rooms(
            num_lecture=5, num_lab=10, num_seminar=10
        )  # 25 rooms
        groups = self._generate_groups(courses=4, groups_per_course=5)  # 20 groups
        teachers = self._generate_teachers(num_teachers=40, slots=slots)  # 40 teachers

        # 2. Generate curriculum (Events)
        events = []
        event_id = 1

        subjects = {
            1: ["Calculus", "Linear Algebra", "Programming 101", "Physics", "English"],
            2: [
                "Algorithms",
                "Databases",
                "Computer Architecture",
                "Discrete Math",
                "Statistics",
            ],
            3: [
                "OS",
                "Networks",
                "Machine Learning",
                "Software Engineering",
                "Web Dev",
            ],
            4: [
                "Distributed Systems",
                "AI",
                "Cryptography",
                "Security",
                "Project Management",
            ],
        }

        # Map teachers to subjects (roughly 10 teachers per course year)
        teacher_pools = {
            1: teachers[0:10],
            2: teachers[10:20],
            3: teachers[20:30],
            4: teachers[30:40],
        }

        for course in range(1, 5):
            course_groups = [g for g in groups if g.name.startswith(f"CS-{course}")]
            course_subjects = subjects[course]
            course_teachers = teacher_pools[course]

            for subject in course_subjects:
                # Assign a primary lecturer for this subject
                lecturer = self.random.choice(course_teachers)

                # 1. Lecture: All groups in the course together (1 pair per week)
                events.append(
                    Event(
                        id=event_id,
                        subject_name=f"{subject} (Lec)",
                        teacher_id=lecturer.id,
                        group_ids=[g.id for g in course_groups],
                        duration=1,
                        room_type_required=RoomType.LECTURE,
                    )
                )
                event_id += 1

                # Assign practice/lab teachers
                practice_teachers = self.random.sample(
                    course_teachers, min(3, len(course_teachers))
                )

                # 2. Practices/Seminars: Individual groups (1-2 pairs per week)
                num_practices = self.random.choice([1, 2])
                for p in range(num_practices):
                    for g in course_groups:
                        p_teacher = self.random.choice(practice_teachers)
                        events.append(
                            Event(
                                id=event_id,
                                subject_name=f"{subject} (Prac)",
                                teacher_id=p_teacher.id,
                                group_ids=[g.id],
                                duration=1,
                                room_type_required=RoomType.LECTURE,  # Seminars can be in standard rooms
                            )
                        )
                        event_id += 1

                # 3. Labs (for tech subjects): Half groups (1 pair per week)
                if subject in [
                    "Programming 101",
                    "Algorithms",
                    "Databases",
                    "Networks",
                    "Machine Learning",
                    "Web Dev",
                ]:
                    for g in course_groups:
                        l_teacher = self.random.choice(practice_teachers)
                        events.append(
                            Event(
                                id=event_id,
                                subject_name=f"{subject} (Lab)",
                                teacher_id=l_teacher.id,
                                group_ids=[g.id],
                                duration=1,
                                room_type_required=RoomType.LAB,
                            )
                        )
                        event_id += 1

        print("Generated Faculty Dataset:")
        print(f" - Groups: {len(groups)}")
        print(f" - Teachers: {len(teachers)}")
        print(f" - Rooms: {len(rooms)}")
        print(f" - Slots: {len(slots)}")
        print(f" - Total Events: {len(events)}")

        return ScheduleContext(
            teachers=teachers, groups=groups, rooms=rooms, slots=slots, events=events
        )


if __name__ == "__main__":
    generator = FacultyDatasetGenerator(seed=123)
    context = generator.generate_faculty_dataset()
