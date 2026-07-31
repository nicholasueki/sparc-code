#!/usr/bin/env python3
"""Long-horizon memory eval — zero knowledge -> episodes -> distillation -> recall.

Uses ISOLATED stores (temp world.db + temp vector mirror on this machine) so the
production robot never learns about fictional people. Model calls (think/distill)
go to the real cortexd — they are stateless w.r.t. memory.

Pipeline under test = the real one:
  events -> WorldModel.add_event -> /distill -> reconcile (commit_fact) ->
  mirror.upsert -> retrieval briefing (recent + vector + facts) -> /think

Usage: .venv/bin/python scripts/eval_memory.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
for pkg in ("common", "node_a", "node_c"):
    sys.path.insert(0, str(ROOT / "packages" / pkg))

from sparc_node_a.world_model import WorldModel  # noqa: E402
from sparc_node_c.memory import SemanticMemory  # noqa: E402

CORTEX = "http://10.1.215.33:8800"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------------------------------------------------------ episode script

DAY1_EVENTS = [
    ("person_entered", "someone new came into view"),
    ("user_said", 'someone said: "Hi! I\'m Maya, Nicholas\'s sister — I\'m visiting for the week."'),
    ("sparc_said", 'SPARC said: "Nice to meet you, Maya! Welcome."'),
    ("user_said", 'someone said: "Nicholas is stuck at work until late tonight."'),
    ("person_left", "they left SPARC's view"),
    ("sound", "a doorbell sound was heard"),
    ("person_entered", "Maya came into view"),
    ("user_said", 'someone said: "Ugh, I can\'t eat this — I really hate cilantro."'),
    ("sparc_said", 'SPARC said: "Noted — no cilantro fan in the house this week!"'),
    ("user_said", 'someone said: "My flight back home leaves Friday at 9 in the morning."'),
    ("user_said", 'someone said: "SPARC, remember that the spare key is under the blue flowerpot."'),
    ("sound", "a microwave beep was heard"),
    ("person_left", "they left SPARC's view"),
    ("person_entered", "Maya came into view"),
    ("user_said", 'someone said: "Goodnight SPARC, see you tomorrow."'),
    ("person_left", "they left SPARC's view"),
]

TARGET_FACTS = {
    "maya_sister": ["maya", "sister"],
    "cilantro": ["cilantro"],
    "flight": ["flight", "friday"],
}

PROBES = [
    dict(id="P1_name_greeting",
         event="Maya just came into view and glanced at SPARC",
         situation="Maya entered the room in the morning",
         scene="It's Wednesday morning. SPARC is in the apartment, on his stand. "
               "A familiar guest is here, came in just now, and looked at SPARC.",
         allowed={"say"}, must_any=["maya"], must_not=["nicholas!"]),
    dict(id="P2_flight_recall",
         event='they said: "SPARC, when does my flight leave again?"',
         situation="Maya asks when her flight leaves",
         scene="It's Wednesday morning. SPARC is in the apartment. Maya is here, "
               "drinking coffee near the couch.",
         allowed={"say"}, must_any=["friday"], must_not=["saturday", "sunday", "monday"],
         bonus=["9", "nine"]),
    dict(id="P3_indirect_use",
         event='they said: "I\'m making salad for Maya tonight — should I add cilantro?"',
         situation="Nicholas asks about adding cilantro to Maya's salad",
         scene="It's Wednesday evening. SPARC is in the apartment. Nicholas is here "
               "(sure it's him), chopping vegetables in the kitchen.",
         allowed={"say"}, must_any=["hate", "doesn't like", "does not like", "no cilantro",
                                    "skip the cilantro", "leave it out", "avoid"],
         must_not=["she loves cilantro", "great idea"]),
    dict(id="P4_no_fabrication",
         event='they said: "What does Maya do for work, again?"',
         situation="Nicholas asks what Maya does for work",
         scene="It's Wednesday evening. SPARC is in the apartment. Nicholas is here, "
               "relaxing on the couch.",
         allowed={"say", "ask_user"},
         must_any=["know", "didn't mention", "never said", "not sure", "no idea",
                   "hasn't told", "didn't say", "don't think she", "don't recall",
                   "only heard", "haven't heard"],  # negated-knowledge phrasings vary
         must_not=["engineer", "teacher", "doctor", "designer", "nurse", "lawyer"]),
    dict(id="P5_deterministic_key",
         event='they said: "Where did we put the spare key?"',
         situation="someone asks where the spare key is hidden",
         scene="It's Thursday morning. SPARC is in the apartment. Maya is here, by the door.",
         allowed={"say"}, must_any=["blue flowerpot", "flowerpot", "flower pot"],
         must_not=["under the mat", "in the drawer"]),
]


def briefing(world: WorldModel, mirror: SemanticMemory, situation: str) -> str:
    """Replica of orchestrator._memory_briefing composition, using local stores."""
    lines = list(world.facts_for_prompt(3))
    for _, stmt, _ in mirror.retrieve(situation, k=4, min_score=0.35):
        if stmt not in lines:
            lines.append(stmt)
    return " ".join(lines[:6])[:600]


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="sparc_memeval_")
    world = WorldModel(f"{tmp}/world.db")
    mirror = SemanticMemory(f"{tmp}/mirror.db", EMBED_MODEL)
    client = httpx.Client(timeout=90)
    print(f"isolated stores in {tmp}")
    results = {}

    # ---- phase 0: zero knowledge sanity
    assert world.facts_for_prompt(9) == [], "world not empty"
    assert mirror.count() == 0, "mirror not empty"
    print("phase 0: zero knowledge confirmed")

    # ---- phase 1: live the day (events + the deterministic remember rule)
    import re
    for type_, desc in DAY1_EVENTS:
        world.add_event(type_, desc, [], 0.5)
        m = re.search(r"\bremember\b[,:]?\s*(?:that\s+)?(.+)", desc, re.IGNORECASE)
        if m and "said:" in desc:
            stmt = m.group(1).strip().rstrip('."!')
            fid = world.commit_fact(stmt, source="user_told", confidence=0.9)
            mirror.upsert(fid, stmt)
            print(f"  deterministic remember: {stmt}")
    print(f"phase 1: {len(DAY1_EVENTS)} events lived")

    # ---- phase 2: idle distillation (real model)
    episodes = [d for _, d in DAY1_EVENTS]
    r = client.post(f"{CORTEX}/distill", json={"episodes": episodes})
    r.raise_for_status()
    proposals = r.json()["proposals"]
    print(f"phase 2: distiller proposed {len(proposals)} facts:")
    for p in proposals:
        print(f"   - [{p['kind']} {p['confidence']:.2f}] {p['statement']}")
        fid = world.commit_fact(p["statement"], source="distillation",
                                confidence=p["confidence"])
        mirror.upsert(fid, p["statement"])

    all_facts = " ".join(world.facts_for_prompt(50)).lower()
    capture = {k: all(w in all_facts for w in words) for k, words in TARGET_FACTS.items()}
    results["capture"] = capture
    print(f"phase 2 capture: {capture}")

    # ---- phase 3: recall probes through the real think path
    probe_results = []
    for p in PROBES:
        brief = briefing(world, mirror, p["situation"])
        req = dict(deliberation_id=f"memeval-{uuid.uuid4().hex[:6]}",
                   scene=p["scene"], memory=brief, conversation=[],
                   event=p["event"], max_options=3)
        t0 = time.time()
        resp = client.post(f"{CORTEX}/think", json=req).json()
        chosen = resp["options"]["options"][resp["choice"]]
        text = (chosen["args"].get("text") or "").lower()
        kind = chosen["action"]
        ok_kind = kind in p["allowed"]
        ok_any = any(m in text for m in p["must_any"])
        ok_not = not any(m in text for m in p["must_not"])
        ok = ok_kind and ok_any and ok_not
        bonus = any(b in text for b in p.get("bonus", [])) if p.get("bonus") else None
        probe_results.append(dict(id=p["id"], ok=ok, kind=kind, text=text[:110],
                                  briefing=brief[:160], bonus=bonus,
                                  ms=int((time.time() - t0) * 1000)))
        flag = "PASS" if ok else "FAIL"
        extra = f" bonus={'yes' if bonus else 'no'}" if bonus is not None else ""
        print(f"[{flag}] {p['id']:<22}{extra}\n       brief: \"{brief[:120]}\"\n       said:  \"{text}\"")
    results["probes"] = probe_results

    # ---- phase 4: dedup / reinforcement check (same lookup path as orchestrator)
    def commit_deduped(stmt: str, conf: float) -> str:
        hits = mirror.retrieve(stmt, k=1, min_score=0.35)
        similar = (hits[0][0], hits[0][2]) if hits else None
        fid = world.commit_fact(stmt, source="distillation", confidence=conf,
                                similar=similar)
        if not (similar and similar[0] == fid):
            mirror.upsert(fid, stmt)
        return fid

    commit_deduped("Maya really hates cilantro.", 0.6)  # near-dup of distilled fact
    facts_now = world.facts_for_prompt(50)
    cilantro_facts = [f for f in facts_now if "cilantro" in f.lower()]
    results["dedup"] = dict(cilantro_fact_count=len(cilantro_facts),
                            semantic_dedup="OK" if len(cilantro_facts) == 1 else "GAP")
    print(f"phase 4 dedup: {len(cilantro_facts)} cilantro fact(s) -> "
          f"{'semantic dedup GAP' if len(cilantro_facts) > 1 else 'reinforced, no duplicate — ok'}")

    n_pass = sum(r["ok"] for r in probe_results)
    print(f"\n==== MEMORY EVAL: capture {sum(capture.values())}/{len(capture)}, "
          f"recall {n_pass}/{len(PROBES)} ====")
    with open("/tmp/memory_eval.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved: /tmp/memory_eval.json")


if __name__ == "__main__":
    main()
