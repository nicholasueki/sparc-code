#!/usr/bin/env bash
# Print Lucas's conversation history (what was heard + what Lucas said).
# Usage: scripts/transcript.sh [N]   (default: last 40 lines; N=0 for everything)
N="${1:-40}"
ssh robot-vision@robot-vision.local "~/lucas_venv/bin/python - <<'EOF'
import sqlite3, time
db = sqlite3.connect('/home/robot-vision/lucas_data/world.db')
n = $N
q = (\"SELECT ts, type, description FROM events \"
     \"WHERE type IN ('user_said','lucas_said') ORDER BY ts\")
rows = db.execute(q).fetchall()
if n:
    rows = rows[-n:]
last_day = None
for ts, t, desc in rows:
    day = time.strftime('%A %b %d', time.localtime(ts))
    if day != last_day:
        print(f'\n--- {day} ---')
        last_day = day
    who = 'THEM ' if t == 'user_said' else 'LUCAS'
    text = desc.split(': \"', 1)[-1].rstrip('\"')
    print(f\"{time.strftime('%H:%M', time.localtime(ts))} {who} {text}\")
EOF"
