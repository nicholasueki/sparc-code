"""Runner: one genome x the suite -> fitness components + raw traces.

Every run uses ISOLATED temp stores. The production world.db must never learn
about Maya — scripts/eval_memory.py established this rule and it is absolute.

The pipeline under test is the real one: events -> WorldModel.add_event ->
/distill (gene G3) -> commit_fact + mirror.upsert -> briefing (gene G4) ->
/think (genes G0/G1/G2). Nothing is stubbed, which is the point: a prompt that
wins here wins on the path the rover actually runs.

A "seed" is an independent repetition, not an RNG seed — cortexd exposes no
sampler seed and temperature is frozen at 0.3. Repetitions are what the
reliability metric actually wants: run-to-run variance under production settings.
"""
from __future__ import annotations

import logging
import random
import re
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import fitness as fit
from .genome import Genome
from .grader import Aggregate, ProbeOutcome, aggregate, approx_tokens, grade
from .llm import LLMClient
from .suite import CaptureProbe, Event, Probe, Scenario, SCENARIOS

log = logging.getLogger("evo.runner")

ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("common", "node_a", "node_c"):
    p = str(ROOT / "packages" / _pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from sparc_common import narrative  # noqa: E402
from sparc_node_a.world_model import WorldModel  # noqa: E402
from sparc_node_c.memory import SemanticMemory  # noqa: E402

REMEMBER_RE = re.compile(r"\bremember\b[,:]?\s*(?:that\s+)?(.+)", re.IGNORECASE)


# --------------------------------------------------------------- cortex client

@dataclass
class CortexClient:
    base_url: str
    timeout: float = 120.0

    def __post_init__(self) -> None:
        self._c = httpx.Client(timeout=self.timeout)

    def health(self) -> dict:
        return self._c.get(f"{self.base_url}/health").json()

    def distill(self, episodes: list[str], system_override: str) -> list[dict]:
        r = self._c.post(f"{self.base_url}/distill",
                         json={"episodes": episodes, "system_override": system_override})
        r.raise_for_status()
        return r.json().get("proposals", [])

    def think(self, *, scene: str, memory: str, event: str, genome: Genome,
              max_options: int = 3) -> tuple[dict, int]:
        req = {
            "deliberation_id": f"evo-{uuid.uuid4().hex[:8]}",
            "scene": scene, "memory": memory, "conversation": [], "event": event,
            "max_options": max_options,
            "persona_override": genome.persona,                  # G0
            "memory_header_override": genome.briefing_header,    # G1
            "empty_header_override": genome.empty_header,        # G2
        }
        t0 = time.time()
        r = self._c.post(f"{self.base_url}/think", json=req)
        r.raise_for_status()
        return r.json(), int((time.time() - t0) * 1000)

    def close(self) -> None:
        self._c.close()


# ------------------------------------------------------------------ world build

def _backdate(world: WorldModel, event_id: str, days_ago: float) -> None:
    if days_ago <= 0:
        return
    ts = time.time() - days_ago * 86400
    world.db.execute("UPDATE events SET ts=? WHERE id=?", (ts, event_id))
    world.db.commit()


def _backdate_fact(world: WorldModel, fact_id: str, days_ago: float) -> None:
    if days_ago <= 0:
        return
    world.db.execute("UPDATE facts SET created=? WHERE id=?",
                     (time.time() - days_ago * 86400, fact_id))
    world.db.commit()


DISTRACTOR_TEMPLATES = [
    "The {room} light switch is {pos} the door.",
    "The {appliance} makes a {sound} noise when it finishes.",
    "{name} usually leaves for work around {hour}.",
    "There is a {colour} {object} on the {surface}.",
    "The {plant} on the windowsill needs water every {n} days.",
    "{name} prefers the thermostat at {temp} degrees.",
    "Bin collection is on {day} mornings.",
    "The {room} window sticks when it rains.",
]
_FILL = {
    "room": ["hallway", "bathroom", "bedroom", "study", "porch", "landing", "pantry"],
    "pos": ["to the left of", "to the right of", "just inside", "behind"],
    "appliance": ["dishwasher", "dryer", "kettle", "toaster", "washing machine"],
    "sound": ["rattling", "high", "clunking", "buzzing", "chiming"],
    "name": ["Sam", "Priya", "Dev", "Ana", "Tom", "Iris"],
    "hour": ["7:15", "8:00", "8:40", "9:05", "6:50"],
    "colour": ["green", "grey", "wooden", "striped", "cracked", "chipped"],
    "object": ["mug", "lamp", "basket", "notebook", "candle", "clock", "vase"],
    "surface": ["shelf", "counter", "sideboard", "radiator", "bookcase"],
    "plant": ["fern", "orchid", "cactus", "spider plant", "basil"],
    "n": ["3", "4", "5", "7", "10"],
    "temp": ["18", "19", "20", "21", "22"],
    "day": ["Monday", "Tuesday", "Thursday", "Friday"],
}


def _distractors(n: int, rng: random.Random) -> list[str]:
    """Filler facts that are plausible for this apartment and deliberately do not
    collide with any probe target — F6 measures retrieval precision under load,
    not the harness's ability to invent trick answers."""
    out: set[str] = set()
    while len(out) < n:
        t = rng.choice(DISTRACTOR_TEMPLATES)
        out.add(re.sub(r"\{(\w+)\}", lambda m: rng.choice(_FILL[m.group(1)]), t))
    return sorted(out)


@dataclass
class World:
    world: WorldModel
    mirror: SemanticMemory
    tmpdir: str
    proposals: list[dict] = field(default_factory=list)


def build_world(scenario: Scenario, genome: Genome, cortex: CortexClient,
                embed_model: str, rng: random.Random) -> World:
    """Live the scenario for real: events -> distillation -> reconciled facts."""
    tmp = tempfile.mkdtemp(prefix=f"sparc_evo_{scenario.name}_")
    world = WorldModel(f"{tmp}/world.db")
    mirror = SemanticMemory(f"{tmp}/mirror.db", embed_model)
    assert world.facts_for_prompt(9) == [] and mirror.count() == 0, "store not empty"

    for ev in scenario.events:
        eid = world.add_event(ev.type, ev.desc, [], 0.5)
        _backdate(world, eid, ev.days_ago)
        # the deterministic "SPARC, remember X" rule, same as production
        m = REMEMBER_RE.search(ev.desc)
        if scenario.deterministic_remembers and m and "said:" in ev.desc:
            stmt = m.group(1).strip().rstrip('."!')
            fid = world.commit_fact(stmt, source="user_told", confidence=0.9)
            mirror.upsert(fid, stmt)
            _backdate_fact(world, fid, ev.days_ago)

    if scenario.distractors:
        for stmt in _distractors(scenario.distractors, rng):
            fid = world.commit_fact(stmt, source="observed", confidence=0.55)
            mirror.upsert(fid, stmt)

    episodes = [e.desc for e in scenario.events]
    proposals = cortex.distill(episodes, genome.distill_system)
    for p in proposals:
        stmt = p.get("statement", "").strip()
        if not stmt:
            continue
        hits = mirror.retrieve(stmt, k=1, min_score=0.35)
        similar = (hits[0][0], hits[0][2]) if hits else None
        fid = world.commit_fact(stmt, source="distillation",
                                confidence=float(p.get("confidence", 0.6)),
                                similar=similar)
        if not (similar and similar[0] == fid):
            mirror.upsert(fid, stmt)
    return World(world, mirror, tmp, proposals)


# ------------------------------------------------------- G4 briefing composition

def _fetch_facts(world: WorldModel, order: str, k: int) -> list[tuple[str, str, float, str, float]]:
    sql = ("SELECT id, statement, confidence, source, created FROM facts "
           "WHERE invalidated IS NULL ORDER BY ")
    sql += {"recency": "created DESC", "confidence": "confidence DESC, created DESC"}.get(
        order, "confidence DESC, created DESC")
    return world.db.execute(sql + " LIMIT ?", (k,)).fetchall()


_SOURCE_PHRASE = {"user_told": "you asked me to remember",
                  "distillation": "I picked up", "observed": "I noticed"}


def _decorate(stmt: str, conf: float, source: str, created: float, ff: dict) -> str:
    prefix = ""
    if ff.get("confidence_marks"):
        prefix = "" if conf >= 0.8 else ("probably: " if conf >= 0.55 else "I think: ")
    if ff.get("provenance"):
        when = narrative.ago(created) if created else "a while back"
        prefix = f"({_SOURCE_PHRASE.get(source, 'I noticed')}, {when}) " + prefix
    return prefix + stmt


_NAME_RE = re.compile(r"\b(Maya|Nicholas|Priya|Sam|Dev|Ana|Tom|Iris)\b")


def compose_briefing(genome: Genome, w: World, situation: str) -> tuple[str, list[str]]:
    """Gene G4. Returns (briefing_string, lines) — lines feed the density metric."""
    ff, r = genome.fact_format, genome.retrieval
    seen: set[str] = set()
    entries: list[tuple[str, str, float, str, float]] = []

    if ff["order"] != "relevance":
        for row in _fetch_facts(w.world, ff["order"], r["recent_k"]):
            if row[1] not in seen:
                seen.add(row[1])
                entries.append(row)

    for fid, stmt, _score in w.mirror.retrieve(situation, k=r["vector_k"],
                                               min_score=r["min_score"]):
        if stmt in seen:
            continue
        seen.add(stmt)
        row = w.world.db.execute(
            "SELECT id, statement, confidence, source, created FROM facts WHERE id=?",
            (fid,)).fetchone()
        entries.append(row or (fid, stmt, 0.6, "distillation", time.time()))

    if ff["order"] == "relevance":
        for row in _fetch_facts(w.world, "confidence", r["recent_k"]):
            if row[1] not in seen:
                seen.add(row[1])
                entries.append(row)

    lines = [_decorate(s, c, src, cr, ff) for _fid, s, c, src, cr in entries][:r["max_lines"]]

    if ff.get("group_by_entity"):
        groups: dict[str, list[str]] = {}
        for ln in lines:
            m = _NAME_RE.search(ln)
            groups.setdefault(m.group(1) if m else "Home", []).append(ln)
        lines = [f"{who}: {'; '.join(items)}" for who, items in groups.items()]

    brief = ff["separator"].join(lines)[:r["max_chars"]]
    return brief, lines


# ------------------------------------------------------------------- run genome

@dataclass
class RunResult:
    genome_id: str
    components: fit.Components
    fitness: int
    outcomes: list[list[ProbeOutcome]]        # per rep
    capture: dict[str, bool]
    agg: Aggregate
    proposals: dict[str, list[dict]]          # scenario -> distiller output
    error: str | None = None


def run_genome(genome: Genome, probes: list[Probe], captures: list[CaptureProbe], *,
               cortex: CortexClient, embed_model: str, reps: int = 3,
               ref_density_rate: float | None = None,
               judge: LLMClient | None = None,
               rng_seed: int = 0) -> RunResult:
    genome.validate()
    rng = random.Random(rng_seed)
    needed = {p.scenario for p in probes} | {c.scenario for c in captures}
    all_outcomes: list[list[ProbeOutcome]] = []
    capture_flags: dict[str, bool] = {}
    proposals: dict[str, list[dict]] = {}

    for rep in range(reps):
        worlds: dict[str, World] = {}
        try:
            for name in needed:
                worlds[name] = build_world(SCENARIOS[name], genome, cortex,
                                           embed_model, rng)
                proposals[name] = worlds[name].proposals

            # capture grades G3 alone — no briefing, no persona, no think call
            for c in captures:
                facts = " ".join(worlds[c.scenario].world.facts_for_prompt(200)).lower()
                ok = all(w.lower() in facts for w in c.words)
                capture_flags[c.id] = capture_flags.get(c.id, True) and ok

            rep_outcomes: list[ProbeOutcome] = []
            for p in probes:
                w = worlds[p.scenario]
                brief, lines = compose_briefing(genome, w, p.situation)
                try:
                    resp, ms = cortex.think(scene=p.scene, memory=brief,
                                            event=p.event, genome=genome)
                except Exception as e:  # noqa: BLE001
                    log.error("think failed on %s: %s", p.id, e)
                    rep_outcomes.append(ProbeOutcome(
                        p.id, p.family, False, "error", str(e)[:200], brief, lines,
                        0, 2, gates=["transport_error"], reasons=[str(e)[:200]]))
                    continue
                chosen = resp["options"]["options"][resp["choice"]]
                rep_outcomes.append(grade(
                    p, SCENARIOS[p.scenario],
                    action=chosen["action"], text=chosen.get("args", {}).get("text", ""),
                    briefing=brief, briefing_lines=lines, latency_ms=ms,
                    fallback_level=int(resp.get("fallback_level", 0)), judge=judge))
            all_outcomes.append(rep_outcomes)
        finally:
            for w in worlds.values():
                w.world.db.close()
                w.mirror.db.close()

    agg = aggregate(all_outcomes, list(capture_flags.values()))
    rates = list(agg.pass_rates.values())
    D, raw_rate = fit.density(agg.facts_used, agg.briefing_tokens, ref_density_rate)
    comps = fit.Components(
        A=fit.accuracy(rates), R=fit.reliability(rates), D=D,
        L=fit.latency(agg.median_ms), n_probes=len(probes), n_seeds=reps,
        density_rate=raw_rate,
        precision=(agg.relevant_lines / agg.total_lines) if agg.total_lines else 0.0,
        median_ms=agg.median_ms,
        mean_brief_tokens=agg.briefing_tokens / max(1, len(rates) * reps),
        schema_retries=agg.schema_retries, gates=agg.gates)
    return RunResult(genome.id, comps, fit.fitness(comps), all_outcomes,
                     capture_flags, agg, proposals)


def measure_reference_rate(result: RunResult) -> float:
    """The control genome's facts-per-briefing-token, which anchors D=1.0."""
    return result.components.density_rate
