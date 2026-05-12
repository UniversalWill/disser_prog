import json
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from .models import ScheduleContext

def generate_text_schedule(context: ScheduleContext, schedule: List[Tuple[int, int]]) -> str:
    """
    Generates a human-readable text representation of the schedule.
    """
    output = []
    
    # 1. Map schedule by Day -> Group -> Time -> Event
    # day -> time -> group_id -> event_info
    schedule_by_day = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for idx, (slot_id, room_id) in enumerate(schedule):
        event = context.events[idx]
        slot = context.slots_dict[slot_id]
        room = context.rooms_dict[room_id]
        teacher = context.teachers_dict[event.teacher_id]
        
        event_str = f"[{room.type.value}] {event.subject_name} (Преп: {teacher.name}, Ауд: {room.id}-{room.building})"
        
        for group_id in event.group_ids:
            schedule_by_day[slot.day][slot.time][group_id].append(event_str)

    # 2. Format the output
    days_of_week = {1: "Понедельник", 2: "Вторник", 3: "Среда", 4: "Четверг", 5: "Пятница", 6: "Суббота", 7: "Воскресенье"}
    
    output.append("="*50)
    output.append("РАСПИСАНИЕ ЗАНЯТИЙ")
    output.append("="*50)
    
    # Sort days
    for day in sorted(schedule_by_day.keys()):
        output.append(f"\n--- {days_of_week.get(day, f'День {day}')} ---")
        
        day_schedule = schedule_by_day[day]
        # Sort times
        for time_idx in sorted(day_schedule.keys()):
            time_str = f"Пара {time_idx}"
            output.append(f"\n  {time_str}:")
            
            time_groups = day_schedule[time_idx]
            # Sort groups
            for group_id in sorted(time_groups.keys()):
                group = context.groups_dict[group_id]
                events = time_groups[group_id]
                
                # If there are multiple events for the same group in the same slot, it's a conflict
                if len(events) > 1:
                    events_str = " !!! КОНФЛИКТ !!! " + " И ".join(events)
                else:
                    events_str = events[0]
                    
                output.append(f"    Группа {group.name}: {events_str}")
                
    return "\n".join(output)

def generate_json_schedule(context: ScheduleContext, schedule: List[Tuple[int, int]]) -> str:
    """
    Generates a structured JSON representation of the schedule.
    Format:
    [
        {
            "day": 1,
            "day_name": "Понедельник",
            "time": 1,
            "events": [
                {
                    "group_id": 1,
                    "group_name": "Группа-1",
                    "subject": "Математика",
                    "teacher": "Иванов И.И.",
                    "room": "1-Главный корпус",
                    "room_type": "Lecture"
                }, ...
            ]
        }, ...
    ]
    """
    days_of_week = {1: "Понедельник", 2: "Вторник", 3: "Среда", 4: "Четверг", 5: "Пятница", 6: "Суббота", 7: "Воскресенье"}
    
    # Structure: day -> time -> list of events
    schedule_tree = defaultdict(lambda: defaultdict(list))
    
    for idx, (slot_id, room_id) in enumerate(schedule):
        event = context.events[idx]
        slot = context.slots_dict[slot_id]
        room = context.rooms_dict[room_id]
        teacher = context.teachers_dict[event.teacher_id]
        
        for group_id in event.group_ids:
            group = context.groups_dict[group_id]
            
            event_data = {
                "group_id": group.id,
                "group_name": group.name,
                "subject": event.subject_name,
                "teacher": teacher.name,
                "room": f"{room.id}-{room.building}",
                "room_type": room.type.value,
                "is_conflict": False # To be evaluated if needed
            }
            schedule_tree[slot.day][slot.time].append(event_data)

    # Mark conflicts (if multiple events for same group in same slot)
    for day in schedule_tree:
        for time in schedule_tree[day]:
            group_counts = defaultdict(int)
            for e in schedule_tree[day][time]:
                group_counts[e["group_id"]] += 1
            
            for e in schedule_tree[day][time]:
                if group_counts[e["group_id"]] > 1:
                    e["is_conflict"] = True

    # Flatten into list of dictionaries
    json_output = []
    for day in sorted(schedule_tree.keys()):
        for time in sorted(schedule_tree[day].keys()):
            slot_data = {
                "day": day,
                "day_name": days_of_week.get(day, f"День {day}"),
                "time": time,
                "events": sorted(schedule_tree[day][time], key=lambda x: x["group_id"])
            }
            json_output.append(slot_data)
            
    return json.dumps(json_output, ensure_ascii=False, indent=2)

def generate_html_table(json_data_str: str, context: ScheduleContext) -> str:
    """
    Renders an HTML table from the JSON schedule data.
    Rows: Time slots (Day + Pair)
    Columns: Groups
    """
    data = json.loads(json_data_str)
    
    # Extract unique days, times, and groups
    groups = sorted(context.groups, key=lambda g: g.id)
    
    # Build HTML structure
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Расписание занятий</title>",
        "<style>",
        "  body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f7fa; }",
        "  h1 { color: #2c3e50; text-align: center; }",
        "  table { width: 100%; border-collapse: collapse; background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }",
        "  th, td { border: 1px solid #dfe6e9; padding: 12px; text-align: center; vertical-align: top; }",
        "  th { background-color: #34495e; color: white; position: sticky; top: 0; }",
        "  .day-header { background-color: #bdc3c7; color: #2c3e50; font-weight: bold; text-align: left; padding-left: 20px; }",
        "  .time-col { font-weight: bold; background-color: #ecf0f1; width: 80px; }",
        "  .event { background-color: #e8f4f8; border-radius: 4px; padding: 8px; margin-bottom: 5px; font-size: 0.9em; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }",
        "  .event.conflict { background-color: #ffcccc; border-left: 4px solid #e74c3c; }",
        "  .event-subject { font-weight: bold; color: #2980b9; margin-bottom: 4px; }",
        "  .event-details { color: #7f8c8d; font-size: 0.85em; }",
        "  .type-lab { border-left: 4px solid #27ae60; }",
        "  .type-lecture { border-left: 4px solid #f39c12; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Сводное расписание занятий</h1>",
        "<table>",
        "  <thead>",
        "    <tr>",
        "      <th>Время</th>"
    ]
    
    # Headers for groups
    for g in groups:
        html.append(f"      <th>{g.name}</th>")
    html.append("    </tr>")
    html.append("  </thead>")
    html.append("  <tbody>")
    
    # Reorganize data for easy rendering: dict[day_name][time][group_id] = list of events
    render_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for slot in data:
        day_name = slot["day_name"]
        time = slot["time"]
        for event in slot["events"]:
            render_data[day_name][time][event["group_id"]].append(event)
            
    # Render rows
    for day_name in render_data:
        html.append(f"    <tr><td colspan='{len(groups) + 1}' class='day-header'>{day_name}</td></tr>")
        
        for time in sorted(render_data[day_name].keys()):
            html.append("    <tr>")
            html.append(f"      <td class='time-col'>Пара {time}</td>")
            
            for g in groups:
                events = render_data[day_name][time].get(g.id, [])
                html.append("      <td>")
                
                if not events:
                    html.append("        <span style='color: #bdc3c7;'>-</span>")
                else:
                    for e in events:
                        conflict_class = "conflict" if e.get("is_conflict") else ""
                        type_class = "type-lab" if e["room_type"] == "Lab" else "type-lecture"
                        html.append(f"        <div class='event {type_class} {conflict_class}'>")
                        html.append(f"          <div class='event-subject'>{e['subject']}</div>")
                        html.append(f"          <div class='event-details'>Преп: {e['teacher']}</div>")
                        html.append(f"          <div class='event-details'>Ауд: {e['room']}</div>")
                        html.append("        </div>")
                        
                html.append("      </td>")
            html.append("    </tr>")
            
    html.append("  </tbody>")
    html.append("</table>")
    html.append("</body>")
    html.append("</html>")
    
    return "\n".join(html)
