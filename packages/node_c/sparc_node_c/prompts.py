"""Prompt assembly + response parsing for cortexd.

The model sees prose in, and must return one JSON object. Parsing is defensive:
strip <think>, extract the first balanced JSON object, Pydantic-validate,
deterministically repair (inject wait/ask, clamp indices). §15/§16 compliant.
"""
from __future__ import annotations

import json
import logging
import re

from sparc_common import config
from sparc_common.types import Action, OptionMeta, OptionSet, ThinkRequest

log = logging.getLogger("sparc.prompts")

# Memory prompt genes live in config (personality/memory_prompts) so the evolution
# harness can vary them per-request without editing source. The literals here are
# the last-resort fallback when the config section is absent.
_FALLBACK_BRIEFING_HEADER = (
    "MEMORY (the COMPLETE list of what SPARC knows from the past — if an "
    "answer is not here or in CONVERSATION, SPARC does NOT know it and says so): {memory}"
)
_FALLBACK_EMPTY_HEADER = (
    "MEMORY: (empty — SPARC has no stored knowledge; he must not claim to "
    "remember anything)"
)


def _render_memory(template: str, memory: str) -> str:
    """Substitute the briefing. Plain replace, not str.format — genome text may
    contain JSON braces, and a stray brace must never raise."""
    return template.replace("{memory}", memory)

THINK_TASK = """\
TASK: You are deciding SPARC's next move. Propose {max_options} DISTINCT candidate \
actions from this exact set: {action_set}. Options must \
differ in kind or intent, not wording. Then choose the best one for SPARC's personality \
and a backup. Reply with ONLY this JSON, no other text:
{{"options":[{{"idx":0,"action":"say","args":{{"text":"..."}},"tone":"warm","risk":"low","novelty":0.3}}],
"choice":0,"backup":1,"why":"one short sentence"}}
Rules: args for say/ask_user = {{"text": ...}} (1-2 short sentences, in character); \
remember = {{"statement": ...}}; set_reminder = {{"text": ..., "in_minutes": N}}; wait = {{}}; \
enroll_face = {{"name": ...}} — ONLY when someone SPARC doesn't recognize has just told \
you their name in this conversation and they are visible right now.{motion_rules}"""

MOTION_RULES = """ \
Motion args: look_at = {"target": "<person name|sound|door|window>"}; \
approach = {"target": "<person name>", "standoff_m": 0.5-2.0}; \
back_up = {"distance_m": 0.1-1.0}; stop_moving = {}. \
Move only when it clearly helps; never move toward someone without a reason they'd welcome."""

BASE_ACTIONS = "say, ask_user, wait, remember, set_reminder, enroll_face"
MOTION_ACTIONS = BASE_ACTIONS + ", look_at, approach, back_up, stop_moving"


def build_think_user(req: ThinkRequest) -> str:
    parts = [f"SCENE: {req.scene}"]
    if req.memory:
        header = req.memory_header_override or config.get(
            "memory_prompts.briefing_header", _FALLBACK_BRIEFING_HEADER)
        parts.append(_render_memory(header, req.memory))
    else:
        parts.append(req.empty_header_override or config.get(
            "memory_prompts.empty_header", _FALLBACK_EMPTY_HEADER))
    if req.conversation:
        turns = "\n".join(f"{t['role']}: {t['text']}" for t in req.conversation[-12:])
        parts.append(f"CONVERSATION:\n{turns}")
    parts.append(f"EVENT: {req.event}")
    if req.image_b64:
        parts.append(
            "CAMERA: a live frame from your camera is attached. Use what you can "
            "actually SEE — who is there, gestures (like waving), what they're doing, "
            "notable objects — in your reaction. Mention only what is visible; if the "
            "frame is unclear, say nothing about it."
        )
    parts.append(THINK_TASK.format(
        max_options=req.max_options,
        action_set=MOTION_ACTIONS if req.motion else BASE_ACTIONS,
        motion_rules=MOTION_RULES if req.motion else "",
    ))
    return "\n\n".join(parts)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """First balanced {...} in the text, after stripping think blocks/fences."""
    text = _THINK_RE.sub("", text)
    text = text.replace("```json", "```")
    if "```" in text:
        # prefer fenced content when present
        for chunk in text.split("```"):
            obj = _balanced_object(chunk)
            if obj is not None:
                return obj
    return _balanced_object(text)


def _balanced_object(text: str) -> dict | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, escaped = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_think(raw: str, max_options: int) -> tuple[OptionSet, int, int, str] | None:
    """Returns (options, choice, backup, why) or None when unrecoverable."""
    data = extract_json(raw)
    if not data or "options" not in data:
        return None
    options: list[OptionMeta] = []
    for i, o in enumerate(data.get("options", [])[: max_options + 2]):
        try:
            o["idx"] = i
            options.append(OptionMeta.model_validate(o))
        except Exception:
            log.warning("dropping malformed option: %.120s", o)
    if not options:
        return None
    kinds = {o.action for o in options}
    if "wait" not in kinds:
        options.append(OptionMeta(idx=len(options), action="wait", tone="patient"))
    if "ask_user" not in kinds:
        options.append(
            OptionMeta(
                idx=len(options),
                action="ask_user",
                args={"text": "Sorry — should I do something here?"},
                tone="unsure",
            )
        )
    n = len(options)
    choice = data.get("choice", 0)
    backup = data.get("backup", 0)
    choice = choice if isinstance(choice, int) and 0 <= choice < n else 0
    backup = backup if isinstance(backup, int) and 0 <= backup < n and backup != choice else (
        (choice + 1) % n
    )
    return OptionSet(options=options), choice, backup, str(data.get("why", ""))[:200]


def option_to_action(opt: OptionMeta, why: str, fallback_level: int = 0) -> Action:
    return Action(
        kind=opt.action,
        args=opt.args,
        option_idx=opt.idx,
        why=why,
        fallback_level=fallback_level,
    )


DISTILL_SYSTEM = """You extract durable facts about people and the home from an \
event log. Only facts likely true next week. Reply ONLY with JSON: \
{"proposals":[{"statement":"...","kind":"fact|preference|habit","confidence":0.0}]}"""


def distill_system(override: str | None = None) -> str:
    """Gene G3. override (eval harness) > config > module fallback."""
    return override or config.get("memory_prompts.distill_system", DISTILL_SYSTEM)


VLM_SYSTEM = """You answer questions about a camera image with structured data. \
Reply ONLY with JSON: {"answer":"...","confidence":0.0}. Confidence in [0,1]. \
Answer only what is visible; never invent."""
