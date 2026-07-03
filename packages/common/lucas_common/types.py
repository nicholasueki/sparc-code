"""Shared boundary types. Every cross-module message is one of these models.

Normative per MVP_System_Design.md §4. Keep wire-compatible: additive changes only.
"""
from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field


def now() -> float:
    return time.time()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Msg(BaseModel):
    """Base for every bus message."""

    ts: float = Field(default_factory=now)
    msg_id: str = Field(default_factory=new_id)


# ---------------------------------------------------------------- perception

class Detection(BaseModel):
    track_id: str
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]  # normalized xyxy
    face_embedding: Optional[list[float]] = None  # 512-d, hailo8 only
    pose: Optional[dict[str, tuple[float, float, float]]] = None
    attending: Optional[float] = None  # 0..1 looking-at-Lucas


class DetectionFrame(Msg):
    source: Literal["imx500", "hailo8"]
    detections: list[Detection]
    scene_delta: Literal["new_track", "lost_track", "motion", "class_change", "periodic"]


# --------------------------------------------------------------------- audio

class SpeechEvent(Msg):
    kind: Literal["onset", "offset", "wake"]
    energy: float = 0.0


class SoundEvent(Msg):
    cls: str
    conf: float


class Transcript(Msg):
    text: str
    conf: float = 1.0
    segment_ts: list[tuple[float, float]] = Field(default_factory=list)
    speaker_track_id: Optional[str] = None
    voice_embedding: Optional[list[float]] = None


class SpeakRequest(Msg):
    text: str
    priority: int = 5
    interruptible: bool = True


# ----------------------------------------------------------------- cognition

ActionKind = Literal[
    "say", "ask_user", "wait", "remember", "set_reminder",
    # motion intents (schema ships pre-hardware; execution gated by motion.enabled)
    "look_at", "approach", "back_up", "stop_moving",
]
MOTION_KINDS = {"look_at", "approach", "back_up", "stop_moving"}


class OptionMeta(BaseModel):
    idx: int
    action: ActionKind
    args: dict = Field(default_factory=dict)
    tone: str = "neutral"
    risk: Literal["low", "med", "high"] = "low"
    novelty: float = 0.0


class OptionSet(BaseModel):
    options: list[OptionMeta]


class Selection(BaseModel):
    choice: int
    backup: int
    raw: str = ""
    source: Literal["mac", "h10_selector", "validator"] = "mac"


class SceneFacts(BaseModel):
    people: int = 0
    activity: str = ""
    objects: list[str] = Field(default_factory=list)
    anomaly: Optional[str] = None
    confidence: float = 0.0


class ToolCall(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    deliberation_id: str = ""


class MemoryProposal(BaseModel):
    statement: str
    kind: Literal["fact", "preference", "habit"] = "fact"
    entity_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.6
    evidence_event_ids: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None


class Action(BaseModel):
    """Always-executable result carried by a Deliberation (INV-1)."""

    kind: ActionKind = "wait"
    args: dict = Field(default_factory=dict)
    option_idx: Optional[int] = None
    fallback_level: int = 0  # 0 = full pipeline, higher = degraded
    why: str = ""


# ------------------------------------------------------------ cortexd wire

class ThinkRequest(BaseModel):
    deliberation_id: str
    scene: str  # prose snapshot (~80 tok)
    memory: str = ""  # retrieval briefing (<=120 tok)
    conversation: list[dict] = Field(default_factory=list)  # [{role, text}]
    event: str  # one-line trigger
    image_b64: Optional[str] = None
    max_options: int = 5
    motion: bool = False  # offer motion intents in the action menu
    # experiment overrides (eval harness only; None = use config defaults)
    persona_override: Optional[str] = None
    temperature_override: Optional[float] = None


class ThinkResponse(BaseModel):
    options: OptionSet
    choice: int
    backup: int
    why: str = ""
    action: Action  # resolved chosen action, validated server-side
    timing_ms: dict = Field(default_factory=dict)
    fallback_level: int = 0
    thinking: str = ""  # model's <think> block, for the live watcher


class VlmQuery(BaseModel):
    image_b64: str
    question: str


class VlmAnswer(BaseModel):
    answer: str
    confidence: float = 0.5


class RetrieveRequest(BaseModel):
    situation: str
    entity_ids: list[str] = Field(default_factory=list)
    k: int = 4


class RetrievedFact(BaseModel):
    fact_id: str
    statement: str
    score: float


class UpsertFactRequest(BaseModel):
    fact_id: str
    statement: str


# --------------------------------------------------------------- selector

class SelectorRequest(BaseModel):
    persona_line: str
    mood: str
    options: list[str]  # one-line rendered options, order = idx


class SalienceRequest(BaseModel):
    event_line: str
    context_line: str = ""


class SalienceDecision(BaseModel):
    verdict: Literal["SKIP", "NOTE", "ESCALATE"]
    raw: str = ""
