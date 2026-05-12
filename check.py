import json
data = json.load(open('schedule_output.json', encoding='utf-8'))
for d in data:
    for e in d['events']:
        if e['group_id'] == 8:
            print(f"Day {d['day']} Time {d['time']}: {e['subject']}")
