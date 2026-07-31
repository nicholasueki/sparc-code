"""Narrative time: epochs in storage, human phrasing at read time (design §7.3)."""
from __future__ import annotations

import time
from datetime import datetime


_BUCKETS: list[tuple[float, str]] = [
    (45, "just now"),
    (150, "a couple of minutes ago"),
    (600, "a few minutes ago"),
    (2700, "earlier this hour"),
    (10800, "a couple of hours ago"),
    (43200, "earlier today"),
]


def ago(epoch: float, now: float | None = None) -> str:
    now = now or time.time()
    delta = max(0.0, now - epoch)
    for limit, phrase in _BUCKETS:
        if delta < limit:
            return phrase
    then, cur = datetime.fromtimestamp(epoch), datetime.fromtimestamp(now)
    days = (cur.date() - then.date()).days
    if days == 0:
        return "earlier today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 30:
        return "a while back"
    return "a long time ago"


def scene_clock(now: float | None = None) -> str:
    """'Tuesday evening, just after 7' — the snapshot's opening phrase."""
    dt = datetime.fromtimestamp(now or time.time())
    hour = dt.hour
    part = (
        "early morning" if hour < 7 else "morning" if hour < 12
        else "afternoon" if hour < 17 else "evening" if hour < 22 else "late night"
    )
    return f"{dt.strftime('%A')} {part}"
