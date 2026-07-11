"""M2 — orchestrator: the deterministic spine (Node A).

Subscribes to perception/audio topics, owns the world model, creates
Deliberations, runs the 3-stage anytime pipeline against cortexd/genaid with
full fallback layers, validates against live state, executes (speak/remember),
and traces everything.

Run: python -m lucas_node_a.orchestrator  (mosquitto must be running locally)
"""
from __future__ import annotations

import heapq
import logging
import threading
import time

import httpx

from lucas_common import config
from lucas_common.bus import Bus
from lucas_common.types import (
    Action,
    DetectionFrame,
    SoundEvent,
    SpeakRequest,
    ThinkRequest,
    ThinkResponse,
    Transcript,
)

from .deliberation import Deliberation, Stage, Tier, mark_executed, serialize_scene, validate
from .world_model import WorldModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("lucas.orchestrator")


def _is_storable_fact(text: str) -> bool:
    """Guard the deterministic remember rule against non-facts:
    'remember my face' is an ENROLLMENT request (enroll_face handles it), and
    questions / garbled STT fragments must never become durable facts."""
    low = text.strip().lower()
    if low.endswith("?"):
        return False
    first = low.split()[0] if low.split() else ""
    if first in ("do", "did", "can", "could", "will", "would", "are", "is", "does", "who"):
        return False
    # enrollment / appearance intents are not facts
    if any(p in low for p in ("my face", "who i am", "what i look like", "my appearance")):
        return False
    return True


class Orchestrator:
    def __init__(self) -> None:
        self.world = WorldModel(config.get("node_a.db_path"))
        self.bus = Bus(client_id="orchestrator")
        self.cortex = httpx.Client(
            base_url=config.get("endpoints.cortexd"),
            timeout=config.get("node_a.think_timeout_s", 12),
        )
        self._queue: list[tuple[float, float, Deliberation]] = []  # (-prio, ts, delib)
        self._lock = threading.Lock()
        self._pending_person: dict[str, str] = {}  # track_id -> deliberation id (merge)
        # v0.4: births wait briefly for face identity before greeting (ENRICHED stage)
        self._pending_births: dict[str, float] = {}  # entity_id -> greet deadline

    # ------------------------------------------------------------ ingest

    def on_tier0(self, frame: DetectionFrame) -> None:
        for det in frame.detections:
            if det.cls != "person":
                continue
            if frame.scene_delta == "new_track":
                eid, name, identity, reappeared = self.world.person_appeared(det.track_id)
                who = name or "someone new"
                if reappeared:
                    # object permanence: same person, brief tracking gap — no re-greet
                    self.world.trace("-", "reappearance", {"entity": eid})
                    self.bus.publish_json("lucas/debug/thought", {
                        "kind": "note",
                        "text": "same person returned — no re-greet (object permanence)"})
                    continue
                # v0.4: don't greet yet — give face enrichment a moment to name them
                wait_s = float(config.get("node_a.face.identity_wait_s", 2.5))
                self._pending_births[eid] = time.time() + wait_s
                self.bus.publish_json("lucas/debug/thought", {
                    "kind": "note",
                    "text": f"person detected — waiting up to {wait_s:.0f}s for face identity"})
            elif frame.scene_delta == "lost_track":
                eid = self.world.person_left(det.track_id)
                self._pending_person.pop(det.track_id, None)
                if eid:
                    self._pending_births.pop(eid, None)
                    self.world.add_event("person_left", "they left Lucas's view", [eid], 0.3)

    def on_rich(self, frame: DetectionFrame) -> None:
        """Face embeddings from the Hailo-8 enrichment loop -> identity."""
        for det in frame.detections:
            if det.face_embedding is None:
                continue
            if len(self.world.present) != 1:
                return  # MVP: only resolve identity when unambiguous (v0.5: association)
            eid = next(iter(self.world.present))
            # churn-proof rolling buffer for seamless enrollment (enroll_face action)
            self.world.buffer_face(eid, det.face_embedding)
            new_eid, name, quality = self.world.update_identity(eid, det.face_embedding)
            if name and quality == "known":
                if eid in self._pending_births:  # keep the greet pending under the merged id
                    self._pending_births[new_eid] = self._pending_births.pop(eid)
                if not self.world.db.execute(
                    "SELECT 1 FROM trace WHERE kind='identified' AND payload LIKE ? "
                    "AND ts > ?", (f"%{new_eid}%", time.time() - 300)).fetchone():
                    self.world.trace("-", "identified", {"entity": new_eid, "name": name})
                    self.bus.publish_json("lucas/debug/thought", {
                        "kind": "note", "text": f"face recognized: {name}"})

    def _drain_births(self) -> None:
        """Submit greeting deliberations once identity arrives or the wait expires."""
        now = time.time()
        for eid, deadline in list(self._pending_births.items()):
            info = self.world.present.get(eid)
            if info is None:
                self._pending_births.pop(eid, None)
                continue
            name = info.get("name")
            if not name and now < deadline:
                continue
            self._pending_births.pop(eid, None)
            who = name or "someone new"
            self.world.add_event(
                "person_entered", f"{who} came into view",
                [eid], config.get("node_a.salience.person_priority", 0.7))
            d = Deliberation(
                event_type="person_enters",
                trigger_desc=f"{who} just came into view"
                + ("" if name else " (Lucas doesn't recognize them)"),
                tier=Tier.COMPETE,
                priority=config.get("node_a.salience.person_priority", 0.7),
                entity_ids=[eid],
            )
            self._pending_person[info["track_id"]] = d.id
            self.submit(d)

    def on_transcript(self, tr: Transcript) -> None:
        self.world.conversation.append({"role": "user", "text": tr.text, "ts": tr.ts})
        self.world.add_event("user_said", f'someone said: "{tr.text}"', [], 0.8)
        # Deterministic memory rule: explicit "remember ..." is committed by code,
        # never left to the model's option choice (design §1.4 / §2.4).
        import re as _re

        m = _re.search(r"\bremember\b[,:]?\s*(?:that\s+)?(.+)", tr.text, _re.IGNORECASE)
        if m and len(m.group(1)) > 3 and _is_storable_fact(m.group(1)):
            stmt = m.group(1).strip().rstrip(".!")
            self._commit_fact(stmt, source="user_told", confidence=0.9)
            log.info("DETERMINISTIC REMEMBER: %s", stmt)
        d = Deliberation(
            event_type="person_speaks",
            trigger_desc=f'they said: "{tr.text}"',
            tier=Tier.INTERRUPT,  # direct address preempts
            priority=0.9,
        )
        self.submit(d)

    def _commit_fact(self, stmt: str, source: str, confidence: float = 0.7) -> str:
        """Commit with semantic dedupe: near-duplicates reinforce instead of pile up."""
        similar = None
        try:
            r = self.cortex.post("/memory/retrieve",
                                 json={"situation": stmt, "k": 1}, timeout=4)
            hits = r.json() if r.status_code == 200 else []
            if hits:
                similar = (hits[0]["fact_id"], hits[0]["score"])
        except Exception:
            pass
        fid = self.world.commit_fact(stmt, source=source, confidence=confidence,
                                     similar=similar)
        if not (similar and similar[0] == fid):  # new fact -> mirror it
            try:
                self.cortex.post("/memory/upsert",
                                 json={"fact_id": fid, "statement": stmt}, timeout=5)
            except Exception:
                log.warning("memory mirror offline for fact %s", fid)
        return fid

    def on_sound(self, snd: SoundEvent) -> None:
        prio_map = config.get("node_a.salience.sound_priority", {})
        prio = float(prio_map.get(snd.cls, prio_map.get("default", 0.3)))
        self.world.add_event("sound", f"a {snd.cls} sound was heard", [], prio)
        if prio >= 0.6:
            self.submit(Deliberation(
                event_type="sound_event",
                trigger_desc=f"Lucas heard what sounded like {snd.cls}",
                tier=Tier.INTERRUPT if prio >= 0.9 else Tier.COMPETE,
                priority=prio,
            ))

    # --------------------------------------------------------- scheduler

    def submit(self, d: Deliberation) -> None:
        d.snapshot = serialize_scene(self.world)  # frozen snapshot for computing
        with self._lock:
            heapq.heappush(self._queue, (-d.priority, d.created_at, d))
        self.world.trace(d.id, "created", {
            "event": d.event_type, "priority": d.priority, "trigger": d.trigger_desc})

    def run(self) -> None:
        self.bus.subscribe("lucas/vision/tier0", DetectionFrame, self.on_tier0)
        self.bus.subscribe("lucas/vision/rich", DetectionFrame, self.on_rich)
        self.bus.subscribe("lucas/audio/transcript", Transcript, self.on_transcript)
        self.bus.subscribe("lucas/audio/sound", SoundEvent, self.on_sound)
        self.bus.start()
        log.info("orchestrator live; world has %d facts",
                 len(self.world.facts_for_prompt(99)))
        period = 1.0 / float(config.get("node_a.scheduler_hz", 20))
        while True:
            self._drain_births()
            d = None
            with self._lock:
                if self._queue:
                    _, _, d = heapq.heappop(self._queue)
            if d is None:
                time.sleep(period)
                continue
            # staleness check before working a parked deliberation
            if d.event_type == "person_enters" and d.entity_ids and not any(
                e in self.world.present for e in d.entity_ids
            ):
                self.world.trace(d.id, "abandoned", {"reason": "stale: person left"})
                continue
            try:
                self.pipeline(d)
            except Exception:
                log.exception("pipeline crashed for %s", d.id)
                self.world.trace(d.id, "error", {"stage": d.stage})

    # ---------------------------------------------------------- pipeline

    def _maybe_frame(self, d: Deliberation) -> str | None:
        """One live JPEG (base64) for events where seeing helps. Fail-open:
        no frame is never an error, just a text-only think. Never persisted."""
        if not config.get("node_a.vision_in_loop", True):
            return None
        wants = d.event_type in ("person_enters", "sound_event") or (
            d.event_type == "person_speaks" and self.world.present
        )
        if not wants:
            return None
        try:
            import base64

            r = httpx.get(
                f"http://127.0.0.1:{config.get('node_a.frame_port', 8600)}/frame.jpg",
                timeout=2.5,
            )
            if r.status_code == 200:
                return base64.b64encode(r.content).decode()
        except Exception:
            pass
        return None

    def pipeline(self, d: Deliberation) -> None:
        # ENRICHED: hot-state read (identity/attending already in world model)
        d.stage = Stage.ENRICHED
        # DECIDED: one folded cortex call (options + choice) with L1/L2 fallbacks
        memory_brief = self._memory_briefing(d)
        image_b64 = self._maybe_frame(d)
        req = ThinkRequest(
            deliberation_id=d.id,
            scene=d.snapshot,
            memory=memory_brief,
            conversation=[
                {"role": t["role"], "text": t["text"]}
                for t in self.world.conversation[-12:]
            ],
            event=d.trigger_desc,
            image_b64=image_b64,
            max_options=3,  # eval: 3 options = same pass rate, ~30% faster than 5
        )
        t0 = time.time()
        action = None
        try:
            r = self.cortex.post("/think", json=req.model_dump())
            r.raise_for_status()
            resp = ThinkResponse.model_validate(r.json())
            action = resp.action
            self.world.trace(d.id, "think", {
                "ms": int((time.time() - t0) * 1000),
                "choice": resp.choice, "why": resp.why,
                "fallback_level": resp.fallback_level,
                "n_options": len(resp.options.options),
                "has_image": image_b64 is not None})
            # publish the full thought to the live watcher via the LOCAL broker
            # (cortexd's own cross-node MQTT publish is unreliable under MLX; the
            # orchestrator owns the broker box, so this hop never flaps)
            self.bus.publish_json("lucas/debug/thought", {
                "kind": "think", "deliberation_id": d.id, "event": d.trigger_desc,
                "scene": d.snapshot, "memory": memory_brief, "thinking": resp.thinking,
                "has_image": image_b64 is not None,
                "options": [{"idx": o.idx, "action": o.action, "args": o.args, "tone": o.tone}
                            for o in resp.options.options],
                "choice": resp.choice, "backup": resp.backup, "why": resp.why,
                "gen_ms": resp.timing_ms.get("generate"),
                "attempts": resp.timing_ms.get("attempts"),
            })
        except Exception as e:
            # L2: cortex unreachable -> deterministic reflex partial
            log.warning("cortexd unavailable (%s); using reflex partial", e)
            action = self._reflex_action(d)
            self.world.trace(d.id, "think_fallback", {"error": str(e)[:200]})

        d.partial_result = action
        d.stage = Stage.DECIDED

        # VALIDATE against live state, then execute (backup = deterministic wait)
        ok, reason = validate(self.world, d, action)
        if not ok:
            self.world.trace(d.id, "vetoed", {"reason": reason, "action": action.kind})
            self.bus.publish_json("lucas/debug/thought", {
                "kind": "note",
                "text": f"vetoed {action.kind} ({reason}) — doing nothing instead"})
            action = Action(kind="wait", why=f"vetoed: {reason}", fallback_level=3)
        self.execute(d, action)

    def _memory_briefing(self, d: Deliberation) -> str:
        lines = [s for s in self.world.facts_for_prompt(3)]
        try:  # semantic recall via cortexd (degrades to local facts if down)
            r = self.cortex.post("/memory/retrieve", json={
                "situation": d.trigger_desc, "entity_ids": d.entity_ids, "k": 4},
                timeout=3)
            if r.status_code == 200:
                for hit in r.json():
                    if hit["statement"] not in lines:
                        lines.append(hit["statement"])
        except Exception:
            pass
        return " ".join(lines[:6])[:600]

    def _reflex_action(self, d: Deliberation) -> Action:
        if d.event_type == "person_enters":
            return Action(kind="say", args={"text": "Oh — hi there!"},
                          why="reflex greeting (cortex offline)", fallback_level=2)
        if d.event_type == "person_speaks":
            return Action(kind="say",
                          args={"text": "I heard you — my brain's a bit offline, give me a moment."},
                          why="reflex ack (cortex offline)", fallback_level=2)
        return Action(kind="wait", why="reflex default", fallback_level=2)

    # ----------------------------------------------------------- execute

    def execute(self, d: Deliberation, action: Action) -> None:
        d.stage = Stage.EXECUTED
        if action.kind in ("say", "ask_user"):
            text = action.args.get("text", "")
            self.bus.publish("lucas/tts/say", SpeakRequest(text=text))
            self.world.conversation.append({"role": "lucas", "text": text, "ts": time.time()})
            self.world.add_event("lucas_said", f'Lucas said: "{text}"', d.entity_ids, 0.4)
            log.info("LUCAS SAYS: %s", text)
        elif action.kind == "remember":
            stmt = action.args.get("statement", "")
            self._commit_fact(stmt, source="observed")
            log.info("LUCAS REMEMBERS: %s", stmt)
            # a remember with no reply feels like being ignored — brief deterministic ack
            ack = "Got it — I'll remember that."
            self.bus.publish("lucas/tts/say", SpeakRequest(text=ack))
            self.world.conversation.append({"role": "lucas", "text": ack, "ts": time.time()})
        elif action.kind == "enroll_face":
            name = str(action.args.get("name", "")).strip()
            unknowns = [e for e, i in self.world.present.items() if not i.get("name")]
            eid = self.world.enroll_present(unknowns[0], name) if unknowns else None
            if eid:
                self.world.add_event("enrolled", f"Lucas learned {name}'s face",
                                     [eid], 0.7)
                self._commit_fact(f"{name} is someone Lucas knows by sight",
                                  source="observed", confidence=0.8)
                ack = f"Nice to meet you, {name} — I'll remember your face."
                self.bus.publish(
                    "lucas/tts/say", SpeakRequest(text=ack))
                self.world.conversation.append(
                    {"role": "lucas", "text": ack, "ts": time.time()})
                log.info("LUCAS ENROLLED FACE: %s (%s)", name, eid)
            else:
                # outcome truth: a silent failure would let Lucas later claim success
                self.world.add_event(
                    "enroll_failed",
                    f"Lucas tried to save {name}'s face but couldn't get a clear "
                    f"enough look — it was not saved", d.entity_ids, 0.6)
                self.bus.publish("lucas/tts/say", SpeakRequest(
                    text=f"Sorry {name}, I couldn't get a clear look — can you face me for a second?"))
                log.warning("enroll_face failed post-validation (samples inconsistent)")
        elif action.kind == "set_reminder":
            self.world.add_event("reminder_set",
                                 f"reminder: {action.args.get('text','')}", d.entity_ids, 0.5)
        mark_executed(self.world, d, action)
        self.world.trace(d.id, "executed", {
            "action": action.kind, "args": action.args,
            "fallback_level": action.fallback_level, "age_s": round(d.age(), 2)})


def main() -> None:
    Orchestrator().run()


if __name__ == "__main__":
    main()
