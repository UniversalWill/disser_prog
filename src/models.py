from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class RoomType(str, Enum):
    LECTURE = "Lecture"
    LAB = "Lab"
    SEMINAR = "Seminar"

class WeekType(str, Enum):
    ODD = "Odd"
    EVEN = "Even"
    BOTH = "Both"

class Teacher(BaseModel):
    id: int
    name: str
    max_hours: int = Field(gt=0)
    preferences: Dict[int, int] = Field(default_factory=dict) # slot_id -> weight (e.g., +1 for preferred, -1 for avoiding)

class Group(BaseModel):
    id: int
    name: str
    size: int = Field(gt=0)

class Room(BaseModel):
    id: int
    capacity: int = Field(gt=0)
    type: RoomType
    building: str

class Slot(BaseModel):
    id: int
    day: int = Field(ge=1, le=7) # 1=Monday, 7=Sunday
    time: int = Field(ge=1) # E.g., 1 for 1st pair (08:00-09:30), 2 for 2nd pair, etc.
    week_type: WeekType = WeekType.BOTH

class Event(BaseModel):
    id: int
    subject_name: str
    teacher_id: int
    group_ids: List[int]
    duration: int = Field(gt=0) # Total duration for the event in slots
    room_type_required: RoomType

# Helper structure to hold all input data
class ScheduleContext(BaseModel):
    teachers: List[Teacher]
    groups: List[Group]
    rooms: List[Room]
    slots: List[Slot]
    events: List[Event]

    @property
    def teachers_dict(self) -> Dict[int, Teacher]:
        return {t.id: t for t in self.teachers}
    
    @property
    def groups_dict(self) -> Dict[int, Group]:
        return {g.id: g for g in self.groups}

    @property
    def rooms_dict(self) -> Dict[int, Room]:
        return {r.id: r for r in self.rooms}

    @property
    def slots_dict(self) -> Dict[int, Slot]:
        return {s.id: s for s in self.slots}

    @property
    def events_dict(self) -> Dict[int, Event]:
        return {e.id: e for e in self.events}
