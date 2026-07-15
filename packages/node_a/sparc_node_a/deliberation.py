"""Deliberation package + prose serializer + validator (design §8-§10, MVP-collapsed).

Stages: PERCEIVED -> ENRICHED -> DECIDED (OPTIONED+DECIDED folded into one
cortex call). `partial_result` is always executable (INV-1). The Deliberation
never leaves Node A (INV-3).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from sparc_common import config, narrative
from sparc_common.types import MOTION_KINDS, Action, new_id


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
    bits = [f"It's {narrative.scene_clock()}. SPARC is in the apartment, on his stand."]
    if world.present:
        people = []
        for eid, info in world.present.items():
            state = info.get("identity_state", "unknown")
            if state == "uncertain":
                who = "someone whose identity SPARC is uncertain about"
            else:
                who = world.live_name(eid) or "someone SPARC doesn't recognize"
            attend = ", looking at SPARC" if info.get("attending", 0) > 0.5 else ""
            people.append(f"{who} is here (came in {narrative.ago(info['since'])}{attend})")
        bits.append(" ".join(people) + ".")
        # ground truth about face memory — prevents false "I remember your face"
        # claims and signals when enrollment is possible (design: code owns reality).
        unknown_ids = [
            e for e, i in world.present.items()
            if not i.get("name") and i.get("identity_state", "unknown") == "unknown"
        ]
        if len(unknown_ids) == 1:
            has_face = len(world.face_samples(unknown_ids[0])) >= 3
            bits.append(
                "SPARC has NOT saved the unrecognized person's face. "
                + ("If they tell SPARC their name, SPARC can save their face now."
                   if has_face
                   else "SPARC cannot get a clear look at their face yet."))
        elif len(unknown_ids) > 1:
            bits.append(
                f"There are {len(unknown_ids)} people here SPARC doesn't recognize; "
                "SPARC can only save one new face at a time, when it's clear whose "
                "name was given.")
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
    """SPARC's OWN utterances are summarized, never quoted, in the scene prose.

    Verbatim self-quotes proved to be a feedback loop: anything SPARC once said
    (including mistakes) reappears in 'Recently:' and gets pattern-matched into
    new replies. Others' words stay verbatim — they're context, not a template.
    The database keeps full text either way (transcripts, search, distillation)."""
    if type_ == "sparc_said":
        return "SPARC spoke to them"
    return description


# -------------------------------------------------------------- validator

GENERIC_GREETING = "Oh — hi there!"


def grounded_greeting_text(world, delib: Deliberation) -> str | None:
    """Return the only greeting text authorized for this exact live target."""
    if delib.event_type != "person_enters" or len(delib.entity_ids) != 1:
        return None
    target = delib.entity_ids[0]
    if target not in world.present:
        return None
    name = world.live_name(target)
    return f"Hi {name}!" if name else GENERIC_GREETING


def ground_greeting(world, delib: Deliberation, action: Action) -> Action:
    """Replace unconstrained arrival prose with target-authorized greeting text."""
    if (delib.event_type != "person_enters"
            or action.kind not in ("say", "ask_user")):
        return action
    text = grounded_greeting_text(world, delib)
    if text is None:
        return action
    why = "; ".join(filter(None, (action.why, "identity-grounded greeting")))
    return action.model_copy(update={
        "kind": "say", "args": {"text": text}, "why": why,
    })


def validate(world, delib: Deliberation, action: Action) -> tuple[bool, str]:
    """Deterministic re-check against LIVE state (LIM-M2-3). -> (ok, reason)."""
    if action.kind in MOTION_KINDS:
        # Fail closed: only the literal YAML boolean true enables motion. Even then,
        # no action is executable until a real executor is wired into Node A.
        if config.get("motion.enabled", False) is not True:
            return False, "motion disabled"
        return False, "motion executor unavailable"
    if action.kind in ("say", "ask_user"):
        text = (action.args or {}).get("text", "")
        if not text or len(text) > 400:
            return False, "say/ask text missing or too long"
        if delib.event_type == "person_enters":
            expected = grounded_greeting_text(world, delib)
            if expected is None:
                return False, "greeting target unavailable"
            if action.kind != "say" or text != expected:
                return False, "greeting is not grounded to its live target"
            cooldown = config.get("node_a.cooldowns.greet_same_person_s", 300)
            key = f"greet:{delib.entity_ids[0]}"
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
        unknowns = [
            e for e, i in world.present.items()
            if not i.get("name") and i.get("identity_state", "unknown") == "unknown"
        ]
        # v0.5: known people may be present; the NAME just needs an unambiguous owner
        if len(unknowns) == 0:
            return False, "enroll_face: nobody unrecognized is present"
        if len(unknowns) > 1:
            return False, "enroll_face: two unrecognized people here — unclear whose name"
        if len(world.face_samples(unknowns[0])) < 3:
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
