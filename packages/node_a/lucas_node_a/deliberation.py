"""Deliberation package + prose serializer + validator (design §8-§10, MVP-collapsed).

Stages: PERCEIVED -> ENRICHED -> DECIDED (OPTIONED+DECIDED folded into one
cortex call). `partial_result` is always executable (INV-1). The Deliberation
never leaves Node A (INV-3).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from lucas_common import config, narrative
from lucas_common.types import Action, new_id


class Tier(str, Enum):
    INTERRUPT = "interrupt"
    COMPETE = "compete"
    BACKGROUND = "background"


class Stage(str, Enum):
    PERCEIVED = "perceived"
    ENRICHED = "enriched"
    DECIDED = "decided"
    EXECUTED = "executed"
    ABANDONED = "abandoned"


@dataclass
class Deliberation:
    event_type: str
    trigger_desc: str
    tier: Tier = Tier.COMPETE
    priority: float = 0.5
    id: str = field(default_factory=new_id)
    created_at: float = field(default_factory=time.time)
    deadline: float | None = None
    stage: Stage = Stage.PERCEIVED
    partial_result: Action = field(default_factory=Action)  # INV-1: starts as wait
    entity_ids: list[str] = field(default_factory=list)
    snapshot: str = ""  # frozen prose at creation (compute against this)
    scratch: dict = field(default_factory=dict)

    def age(self) -> float:
        return time.time() - self.created_at


# ------------------------------------------------------------- serializer

def serialize_scene(world) -> str:
    """~80-token prose snapshot per design §8.3. No IDs, uncertainty in words."""
    bits = [f"It's {narrative.scene_clock()}. Lucas is in the apartment, on his stand."]
    if world.present:
        people = []
        for eid, info in world.present.items():
            who = info["name"] or "someone Lucas doesn't recognize"
            attend = ", looking at Lucas" if info.get("attending", 0) > 0.5 else ""
            people.append(f"{who} is here (came in {narrative.ago(info['since'])}{attend})")
        bits.append(" ".join(people) + ".")
    else:
        bits.append("Nobody is in view right now.")
    recent = world.recent_notable_events(3)
    if recent:
        bits.append(
            "Recently: "
            + "; ".join(f"{_render_event(t, d)} ({narrative.ago(ts)})"
                        for ts, t, d in recent)
            + "."
        )
    return " ".join(bits)


def _render_event(type_: str, description: str) -> str:
    """Lucas's OWN utterances are summarized, never quoted, in the scene prose.

    Verbatim self-quotes proved to be a feedback loop: anything Lucas once said
    (including mistakes) reappears in 'Recently:' and gets pattern-matched into
    new replies. Others' words stay verbatim — they're context, not a template.
    The database keeps full text either way (transcripts, search, distillation)."""
    if type_ == "lucas_said":
        return "Lucas spoke to them"
    return description


# -------------------------------------------------------------- validator

def validate(world, delib: Deliberation, action: Action) -> tuple[bool, str]:
    """Deterministic re-check against LIVE state (LIM-M2-3). -> (ok, reason)."""
    if action.kind in ("say", "ask_user"):
        text = (action.args or {}).get("text", "")
        if not text or len(text) > 400:
            return False, "say/ask text missing or too long"
        if delib.event_type == "person_enters":
            # person must still be present
            if delib.entity_ids and not any(e in world.present for e in delib.entity_ids):
                return False, "person already left"
            cooldown = config.get("node_a.cooldowns.greet_same_person_s", 300)
            key = f"greet:{delib.entity_ids[0] if delib.entity_ids else 'unknown'}"
            if time.time() - world.last_action_ts.get(key, 0) < cooldown:
                return False, "greeting cooldown"
            # global greet cooldown: identity churn must never cause rapid re-greeting
            global_cd = config.get("node_a.cooldowns.greet_anyone_s", 60)
            if time.time() - world.last_action_ts.get("greet:*", 0) < global_cd:
                return False, "global greeting cooldown"
    if action.kind == "remember":
        if not (action.args or {}).get("statement"):
            return False, "remember without statement"
    if action.kind == "enroll_face":
        name = str((action.args or {}).get("name", "")).strip()
        if not (2 <= len(name) <= 24 and name.replace(" ", "").replace("-", "").isalpha()):
            return False, "enroll_face: implausible name"
        if name.lower() in (n.lower() for n in world.known_names()):
            return False, f"enroll_face: {name} already enrolled"
        unknowns = [e for e, i in world.present.items() if not i.get("name")]
        if len(world.present) != 1 or len(unknowns) != 1:
            return False, "enroll_face: need exactly one unknown person present"
        if len(world.present[unknowns[0]].get("embs", [])) < 3:
            return False, "enroll_face: not enough face samples yet"
    if action.kind == "set_reminder":
        if not (action.args or {}).get("text"):
            return False, "reminder without text"
    return True, "ok"


def mark_executed(world, delib: Deliberation, action: Action) -> None:
    if action.kind in ("say", "ask_user") and delib.event_type == "person_enters":
        key = f"greet:{delib.entity_ids[0] if delib.entity_ids else 'unknown'}"
        world.last_action_ts[key] = time.time()
        world.last_action_ts["greet:*"] = time.time()
