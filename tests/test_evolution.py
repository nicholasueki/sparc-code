"""End-to-end tests for the memory-prompt evolution harness.

cortexd is stubbed, so these run anywhere and cost nothing. Everything else is
real: real SQLite world model, real sqlite-vec mirror, real fastembed embeddings,
real briefing composition, real grading. That is deliberate — the parts most
likely to be silently wrong are the store plumbing and the G4 composer, not the
HTTP call.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _p in ("common", "node_a", "node_c"):
    sys.path.insert(0, str(ROOT / "packages" / _p))

from evolution import fitness as F  # noqa: E402
from evolution import seeds, suite  # noqa: E402
from evolution.genome import Genome, GenomeError, PERSONA_CORE  # noqa: E402
from evolution.grader import GATE_CONFAB, GATE_SCHEMA, aggregate, grade  # noqa: E402
from evolution.runner import build_world, compose_briefing  # noqa: E402

EMBED = "sentence-transformers/all-MiniLM-L6-v2"


def _vec_available() -> bool:
    """The vector mirror needs a sqlite3 built with extension loading. The stock
    macOS python is not — hence the 'run on Node C' note in eval_memory.py. Tests
    that need a real mirror skip cleanly rather than failing on the wrong machine."""
    import sqlite3
    return hasattr(sqlite3.Connection, "enable_load_extension")


needs_vec = pytest.mark.skipif(
    not _vec_available(),
    reason="sqlite3 lacks enable_load_extension; run this on Node C")

DISTILLED = [
    {"statement": "Maya is Nicholas's sister.", "kind": "fact", "confidence": 0.9},
    {"statement": "Maya is visiting for the week.", "kind": "fact", "confidence": 0.85},
    {"statement": "Maya hates cilantro.", "kind": "preference", "confidence": 0.8},
    {"statement": "Maya's flight home is Friday at 9am.", "kind": "fact", "confidence": 0.85},
    {"statement": "Maya only drinks jasmine tea, never coffee.", "kind": "preference",
     "confidence": 0.75},
    {"statement": "The wifi password is taped to the side of the fridge.", "kind": "fact",
     "confidence": 0.7},
]


class FakeCortex:
    """Canned cortexd. Records what it was asked so genes can be asserted on."""

    def __init__(self, reply="I don't actually know.", action="say"):
        self.reply, self.action = reply, action
        self.think_calls: list[dict] = []
        self.distill_systems: list[str] = []

    def distill(self, episodes, system_override):
        self.distill_systems.append(system_override)
        return list(DISTILLED)

    def think(self, *, scene, memory, event, genome, max_options=3):
        self.think_calls.append({"scene": scene, "memory": memory, "event": event,
                                 "persona": genome.persona,
                                 "header": genome.briefing_header})
        return ({"options": {"options": [{"idx": 0, "action": self.action,
                                          "args": {"text": self.reply}}]},
                 "choice": 0, "backup": 0, "why": "", "fallback_level": 0}, 4200)


@pytest.fixture(scope="module")
def control() -> Genome:
    return next(g for g in seeds.build() if g.id == "s01-control")


# ------------------------------------------------------------------ suite shape

def test_suite_is_well_formed():
    ids = [p.id for p in suite.PROBES]
    assert len(ids) == len(set(ids))
    assert all(p.scenario in suite.SCENARIOS for p in suite.PROBES)
    # every family must have both dev and holdout probes, or the generalization
    # gap is measured on a different distribution than the loop optimizes
    for fam in {p.family for p in suite.PROBES}:
        assert [p for p in suite.PROBES if p.family == fam and not p.holdout]
        assert [p for p in suite.PROBES if p.family == fam and p.holdout]
    assert suite.probes(smoke_only=True), "cascade needs a smoke subset"


def test_smoke_cascade_skips_the_expensive_world():
    """The cascade only pays off if the smoke pass is genuinely cheap — the load
    scenario embeds 200 distractor facts and must stay out of it."""
    smoke = suite.probes(smoke_only=True)
    built = suite.scenarios_for(smoke, [])
    assert "load" not in built
    assert set(built) < set(suite.SCENARIOS)
    assert {p.family for p in smoke} >= {"F1", "F3"}, "smoke must cover the gate-prone families"


# ---------------------------------------------------------------------- genome

def test_persona_core_is_frozen(control):
    assert control.persona.startswith(PERSONA_CORE)
    with pytest.raises(GenomeError):
        replace(control, persona_core="You are a helpful assistant.").validate()


@pytest.mark.parametrize("field,value", [
    ("briefing_header", "MEMORY: here are the facts"),      # no {memory} placeholder
    ("distill_system", "Extract the facts please."),        # drops proposals schema
    ("persona_memory", "   "),                              # empty gene
])
def test_invalid_genes_rejected(control, field, value):
    with pytest.raises(GenomeError):
        replace(control, **{field: value}).validate()


def test_retrieval_frozen_in_phase_one(control):
    with pytest.raises(GenomeError):
        replace(control, retrieval={"recent_k": 99}).validate()


def test_child_records_lineage(control):
    c = control.child("kid", 1, persona_memory="Say when you don't know.")
    assert c.parent == control.id and c.mutated_genes == ["persona_memory"]
    assert c.persona_core == PERSONA_CORE
    assert c.fingerprint() != control.fingerprint()


def test_seed_cells_are_distinct():
    pop = seeds.build(("compound", "atomic"))
    assert len({g.fingerprint() for g in pop}) == len(pop) == 20


# ------------------------------------------------------------- world + briefing

@pytest.fixture(scope="module")
def guest_world(control):
    if not _vec_available():
        pytest.skip("needs sqlite extension loading; run on Node C")
    import random
    w = build_world(suite.GUEST_VISIT, control, FakeCortex(), EMBED, random.Random(0))
    yield w
    w.world.db.close()
    w.mirror.db.close()


@needs_vec
def test_world_starts_empty_and_lives_the_day(guest_world):
    facts = " ".join(guest_world.world.facts_for_prompt(50)).lower()
    # the deterministic "SPARC, remember X" rule must fire without the model
    assert "blue flowerpot" in facts
    # and the distiller's proposals must be reconciled in
    assert "cilantro" in facts and "sister" in facts


@needs_vec
def test_distillation_uses_the_g3_gene(control):
    import random
    fake = FakeCortex()
    w = build_world(suite.GUEST_VISIT, control, fake, EMBED, random.Random(0))
    assert fake.distill_systems == [control.distill_system]
    w.world.db.close(); w.mirror.db.close()


@needs_vec
def test_backdating_actually_moves_events(control):
    import random
    w = build_world(suite.DECAY, control, FakeCortex(), EMBED, random.Random(0))
    rows = w.world.db.execute("SELECT ts FROM events ORDER BY ts").fetchall()
    oldest, newest = rows[0][0], rows[-1][0]
    assert (time.time() - oldest) / 86400 > 20, "3-week-old events were not backdated"
    assert (time.time() - newest) / 86400 < 2
    w.world.db.close(); w.mirror.db.close()


@needs_vec
def test_distractors_do_not_collide_with_probe_targets(control):
    import random
    small = replace(suite.LOAD, distractors=40)
    w = build_world(small, control, FakeCortex(), EMBED, random.Random(1))
    facts = [s.lower() for s in w.world.facts_for_prompt(500)]
    filler = [f for f in facts if "flowerpot" not in f and "cilantro" not in f]
    for target in ("cilantro", "flowerpot", "jasmine", "friday"):
        assert sum(target in f for f in filler) == 0, f"filler collides on {target}"
    assert len(facts) >= 40
    w.world.db.close(); w.mirror.db.close()


@needs_vec
def test_briefing_surfaces_the_relevant_fact(control, guest_world):
    brief, lines = compose_briefing(control, guest_world, "when does Maya's flight leave")
    assert "flight" in brief.lower()
    assert 0 < len(lines) <= control.retrieval["max_lines"]
    assert len(brief) <= control.retrieval["max_chars"]


@needs_vec
@pytest.mark.parametrize("gid", [g.id for g in seeds.build()])
def test_every_seed_composes_a_usable_briefing(gid, guest_world):
    g = next(x for x in seeds.build() if x.id == gid)
    brief, lines = compose_briefing(g, guest_world, "Maya's flight home")
    assert brief.strip() and lines
    assert len(brief) <= g.retrieval["max_chars"]


@needs_vec
def test_g4_decorations_are_visible(guest_world):
    base = next(g for g in seeds.build() if g.id == "s01-control")
    prov = replace(base, fact_format={**base.fact_format, "provenance": True})
    conf = replace(base, fact_format={**base.fact_format, "confidence_marks": True})
    grouped = replace(base, fact_format={**base.fact_format, "group_by_entity": True,
                                         "separator": "\n- "})
    q = "Maya's flight home"
    assert "(" in compose_briefing(prov, guest_world, q)[0]
    plain = compose_briefing(base, guest_world, q)[0]
    assert compose_briefing(conf, guest_world, q)[0] != plain
    assert "Maya:" in compose_briefing(grouped, guest_world, q)[0]


# --------------------------------------------------------------------- grading

def _probe(pid):
    return next(p for p in suite.PROBES if p.id == pid)


def test_correct_answer_passes():
    p = _probe("F1_flight")
    o = grade(p, suite.GUEST_VISIT, action="say",
              text="Friday at nine in the morning!", briefing="Maya's flight is Friday 9am.",
              briefing_lines=["Maya's flight is Friday 9am."], latency_ms=4000,
              fallback_level=0)
    assert o.passed and not o.gates and o.bonus is True


def test_wrong_day_fails_without_gating():
    p = _probe("F1_flight")
    o = grade(p, suite.GUEST_VISIT, action="say", text="It's Saturday morning.",
              briefing="x", briefing_lines=["x"], latency_ms=4000, fallback_level=0)
    assert not o.passed and not o.gates, "an ordinary wrong answer is not a safety gate"


def test_fabricated_job_trips_the_confabulation_gate():
    p = _probe("F3_job")
    o = grade(p, suite.GUEST_VISIT, action="say",
              text="She's a graphic designer! She mentioned her studio on Tuesday.",
              briefing="Maya is Nicholas's sister.",
              briefing_lines=["Maya is Nicholas's sister."], latency_ms=4000,
              fallback_level=0)
    assert not o.passed and GATE_CONFAB in o.gates


def test_honest_refusal_passes():
    p = _probe("F3_job")
    o = grade(p, suite.GUEST_VISIT, action="say",
              text="I don't actually know! I've only heard you mention she's visiting.",
              briefing="Maya is visiting.", briefing_lines=["Maya is visiting."],
              latency_ms=4000, fallback_level=0)
    assert o.passed and not o.gates


def test_negated_knowledge_phrasing_variants_all_pass():
    p = _probe("F3_job")
    for text in ["I'm not sure, she never said.", "No idea — she hasn't told me.",
                 "She didn't mention that.", "I don't recall her saying."]:
        o = grade(p, suite.GUEST_VISIT, action="say", text=text, briefing="",
                  briefing_lines=[], latency_ms=4000, fallback_level=0)
        assert o.passed, f"loose must_any failed on: {text}"


def test_wrong_action_fails_even_with_right_words():
    p = _probe("F1_key")
    o = grade(p, suite.GUEST_VISIT, action="remember",
              text="The spare key is under the blue flowerpot.", briefing="",
              briefing_lines=[], latency_ms=4000, fallback_level=0)
    assert not o.passed


def test_unparseable_output_trips_schema_gate():
    p = _probe("F1_flight")
    o = grade(p, suite.GUEST_VISIT, action="wait", text="", briefing="",
              briefing_lines=[], latency_ms=4000, fallback_level=2)
    assert GATE_SCHEMA in o.gates


def test_stale_transient_is_caught_on_greeting():
    p = _probe("F5_no_stale_assert")
    good = grade(p, suite.DECAY, action="say", text="Hey, good to see you!",
                 briefing="", briefing_lines=[], latency_ms=3000, fallback_level=0)
    bad = grade(p, suite.DECAY, action="say", text="Hope you're feeling better!",
                briefing="", briefing_lines=[], latency_ms=3000, fallback_level=0)
    assert good.passed and not bad.passed and GATE_CONFAB in bad.gates


def test_density_counts_only_relevant_lines():
    p = _probe("F1_flight")
    lines = ["Maya's flight is Friday 9am.", "The fern needs water.", "Bins on Monday."]
    o = grade(p, suite.GUEST_VISIT, action="say", text="Friday!", briefing=" ".join(lines),
              briefing_lines=lines, latency_ms=4000, fallback_level=0)
    assert o.relevant_lines == 1 and o.total_lines == 3


# ------------------------------------------------------------------- aggregate

def test_aggregate_and_fitness_reward_consistency():
    p = _probe("F1_flight")

    def outcome(text):
        return grade(p, suite.GUEST_VISIT, action="say", text=text,
                     briefing="Maya's flight is Friday 9am.",
                     briefing_lines=["Maya's flight is Friday 9am."],
                     latency_ms=4000, fallback_level=0)

    steady = aggregate([[outcome("Friday!")]] * 3, [])
    flaky = aggregate([[outcome("Friday!")], [outcome("Saturday.")],
                       [outcome("Friday!")]], [])
    steady_c = F.Components(A=F.accuracy(list(steady.pass_rates.values())),
                            R=F.reliability(list(steady.pass_rates.values())))
    flaky_c = F.Components(A=F.accuracy(list(flaky.pass_rates.values())),
                           R=F.reliability(list(flaky.pass_rates.values())))
    assert F.fitness(steady_c) > F.fitness(flaky_c)
    assert steady_c.R == 1.0 and flaky_c.R < 1.0


@needs_vec
def test_full_run_wiring(control, monkeypatch):
    """One genome through run_genome end to end, cortex stubbed."""
    from evolution.runner import run_genome
    fake = FakeCortex(reply="I don't actually know — you've never mentioned it.")
    probes = [p for p in suite.probes(smoke_only=True) if p.scenario == "guest_visit"]
    res = run_genome(control, probes, suite.capture_probes()[:3], cortex=fake,
                     embed_model=EMBED, reps=2, ref_density_rate=0.02)
    assert res.components.n_seeds == 2
    assert len(res.outcomes) == 2 and len(res.outcomes[0]) == len(probes)
    assert 0.0 <= res.components.A <= 1.0
    # the genes must actually have reached the model
    assert all(c["persona"] == control.persona for c in fake.think_calls)
    assert all(c["header"] == control.briefing_header for c in fake.think_calls)
    assert res.capture, "capture probes should have been graded"


# --------------------------------------------- G4 composition without embeddings
# Only the *mirror* needs sqlite extensions; the world model is plain SQLite. Stubbing
# retrieval lets the G4 composer — pure logic, and the gene most likely to be silently
# wrong — be tested on any machine instead of only on Node C.

class FakeMirror:
    def __init__(self, hits): self.hits = hits
    def retrieve(self, situation, k, min_score): return self.hits[:k]


def _world_with(facts, hits):
    import tempfile
    from evolution.runner import World
    from sparc_node_a.world_model import WorldModel
    w = WorldModel(f"{tempfile.mkdtemp()}/w.db")
    ids = {}
    for stmt, conf, src, days_ago in facts:
        fid = w.commit_fact(stmt, source=src, confidence=conf)
        if days_ago:
            w.db.execute("UPDATE facts SET created=? WHERE id=?",
                         (time.time() - days_ago * 86400, fid))
            w.db.commit()
        ids[stmt] = fid
    return World(w, FakeMirror([(ids[s], s, sc) for s, sc in hits]), "/tmp", [])


FACTS = [
    ("Maya is Nicholas's sister.", 0.90, "distillation", 0),
    ("Maya hates cilantro.", 0.80, "distillation", 0),
    ("Maya's flight home is Friday at 9am.", 0.85, "distillation", 0),
    ("The spare key is under the blue flowerpot.", 0.95, "user_told", 2),
    ("Nicholas started a job at a bakery.", 0.60, "distillation", 1),
]


def test_briefing_orders_and_dedupes():
    base = next(g for g in seeds.build() if g.id == "s01-control")
    w = _world_with(FACTS, [("Maya's flight home is Friday at 9am.", 0.8)])
    brief, lines = compose_briefing(base, w, "flight")
    assert len(lines) == len(set(lines)), "a fact appeared twice in one briefing"
    assert "flight" in brief.lower()
    assert len(lines) <= base.retrieval["max_lines"]


def test_provenance_and_confidence_decorations():
    base = next(g for g in seeds.build() if g.id == "s01-control")
    w = _world_with(FACTS, [("Nicholas started a job at a bakery.", 0.7)])
    prov = replace(base, fact_format={**base.fact_format, "provenance": True})
    b, _ = compose_briefing(prov, w, "job")
    assert "you asked me to remember" in b or "I picked up" in b or "I noticed" in b
    assert "ago" in b or "yesterday" in b or "days" in b, "recency phrasing missing"
    conf = replace(base, fact_format={**base.fact_format, "confidence_marks": True})
    b2, _ = compose_briefing(conf, w, "job")
    assert "probably:" in b2 or "I think:" in b2, "low-confidence fact was not hedged"


def test_grouping_clusters_by_person():
    base = next(g for g in seeds.build() if g.id == "s01-control")
    g = replace(base, fact_format={**base.fact_format, "group_by_entity": True,
                                   "separator": "\n- "})
    w = _world_with(FACTS, [("Maya hates cilantro.", 0.8)])
    b, lines = compose_briefing(g, w, "cilantro")
    assert any(ln.startswith("Maya:") for ln in lines)
    assert all(ln.count(":") >= 1 for ln in lines)


def test_recency_vs_confidence_ordering_differ():
    base = next(g for g in seeds.build() if g.id == "s01-control")
    w = _world_with(FACTS, [])
    by_conf, _ = compose_briefing(replace(base, fact_format={**base.fact_format,
                                                            "order": "confidence"}), w, "x")
    by_rec, _ = compose_briefing(replace(base, fact_format={**base.fact_format,
                                                           "order": "recency"}), w, "x")
    assert by_conf != by_rec, "order gene had no effect"


def test_max_chars_is_respected():
    base = next(g for g in seeds.build() if g.id == "s01-control")
    long_facts = [(f"Fact number {i} about the apartment and its many details.",
                   0.7, "distillation", 0) for i in range(30)]
    w = _world_with(long_facts, [])
    b, _ = compose_briefing(base, w, "x")
    assert len(b) <= base.retrieval["max_chars"]


# ------------------------------------------------------- local-endpoint client

from evolution.llm import LLMClient, LLMError, extract_json  # noqa: E402


@pytest.mark.parametrize("url,needs_key", [
    ("http://10.1.215.40:1234/v1", False),      # LAN
    ("http://100.64.1.2:1234", False),          # tailscale CGNAT range
    ("http://192.168.1.9:11434/v1", False),     # ollama on a home LAN
    ("http://localhost:1234/v1", False),
    ("http://spark.local:1234/v1", False),
    ("https://openrouter.ai/api/v1", True),     # hosted gateway
])
def test_private_endpoints_need_no_api_key(url, needs_key):
    if needs_key:
        with pytest.raises(LLMError):
            LLMClient(model="m", base_url=url, api_key_env="DEFINITELY_UNSET_KEY_XYZ")
    else:
        c = LLMClient(model="m", base_url=url)
        assert c.requires_key is False
        assert c.base_url.endswith("/v1"), "base_url should be normalised to /v1"


def test_base_url_is_required():
    with pytest.raises(LLMError):
        LLMClient(model="m", base_url="")


def test_json_salvage_handles_local_model_quirks():
    # a reasoning model that emits <think> then JSON
    assert extract_json('<think>hmm let me see</think>{"verdict":"pass"}') == {"verdict": "pass"}
    # fenced
    assert extract_json('Sure!\n```json\n{"a":1}\n```') == {"a": 1}
    # prose preamble
    assert extract_json('Here is my answer: {"a":1} hope that helps') == {"a": 1}
    # nested braces
    assert extract_json('{"a":{"b":[1,2]},"c":"}"}')["a"]["b"] == [1, 2]
    # truncated <think> with no closing tag (hit max_tokens mid-reasoning)
    assert extract_json('{"ok":true}\n<think>and then I would') == {"ok": True}
    # nothing parseable
    assert extract_json("I cannot answer that.") is None
    assert extract_json("") is None


def test_map_runs_concurrently_and_isolates_failures():
    import time as _t
    c = LLMClient(model="m", base_url="http://10.0.0.1:1234/v1", max_concurrency=4)

    def work(i):
        _t.sleep(0.2)
        if i == 2:
            raise RuntimeError("boom")
        return i * 2

    t0 = _t.time()
    out = c.map(work, range(4))
    elapsed = _t.time() - t0
    assert out[0] == 0 and out[1] == 2 and out[3] == 6
    assert isinstance(out[2], Exception), "a failed item must not sink the batch"
    assert elapsed < 0.6, f"ran serially ({elapsed:.2f}s) — concurrency not applied"


def test_postmortem_batch_survives_a_failing_reflection(control):
    from evolution.researcher import Researcher
    from evolution.runner import RunResult
    from evolution.grader import Aggregate

    class Boom(LLMClient):
        def complete_json(self, *a, **k):
            raise LLMError("endpoint down")

    agg = Aggregate({}, {"F3": 0.5}, [], 0, 1, 0, 0, 100.0, 0, 0.0, 1)
    res = RunResult(control.id, F.Components(), 0, [], {}, agg, {})
    r = Researcher(
        postmortem_client=Boom(model="m", base_url="http://10.0.0.1:1234/v1"),
        synthesis_client=Boom(model="m", base_url="http://10.0.0.1:1234/v1"))
    out = r.postmortem_batch([(control, res, "died")], "")
    assert len(out) == 1 and out[0]["genome"] == control.id
    assert "unavailable" in out[0]["postmortem"] or "failed" in out[0]["postmortem"]


# ------------------------------------------------- structured-output degradation

class _FakeResp:
    def __init__(self, status, body): self.status_code, self._b = status, body
    @property
    def text(self): import json as j; return j.dumps(self._b)
    def json(self): return self._b
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError(f"HTTP {self.status_code}")


def _client_recording(reject: set[str]):
    """Client whose fake server 400s on the given response_format types."""
    c = LLMClient(model="m", base_url="http://10.0.0.1:1234/v1", max_retries=5)
    sent: list[dict] = []

    def post(url, json=None, headers=None):
        import copy
        sent.append(copy.deepcopy(json))   # the retry loop mutates payload in place
        rf = (json or {}).get("response_format") or {}
        if rf.get("type") in reject:
            return _FakeResp(400, {"error": f"response_format {rf.get('type')} unsupported"})
        return _FakeResp(200, {"choices": [{"message": {"content": '{"ok":true}'}}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    c._client.post = post
    return c, sent


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}


def test_schema_mode_used_when_server_supports_it():
    c, sent = _client_recording(reject=set())
    assert c.complete_json("s", "u", schema=SCHEMA) == {"ok": True}
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert sent[0]["response_format"]["json_schema"]["schema"] == SCHEMA
    assert c.schema_mode is True


def test_degrades_schema_to_json_object():
    c, sent = _client_recording(reject={"json_schema"})
    assert c.complete_json("s", "u", schema=SCHEMA) == {"ok": True}
    assert [s["response_format"]["type"] for s in sent] == ["json_schema", "json_object"]
    assert c.schema_mode is False and c.json_object_mode is True


def test_degrades_all_the_way_to_salvage():
    c, sent = _client_recording(reject={"json_schema", "json_object"})
    assert c.complete_json("s", "u", schema=SCHEMA) == {"ok": True}
    assert [s.get("response_format", {}).get("type") for s in sent] == [
        "json_schema", "json_object", None]
    # Both structured modes are now off; the final call carries no response_format.
    assert c.schema_mode is False and c.json_object_mode is False
    assert "response_format" not in sent[-1]


def test_researcher_schemas_are_strict_and_gene_enums_match():
    from evolution import researcher as R
    from evolution.genome import GENES
    for sch in (R.POSTMORTEM_SCHEMA, R.SYNTHESIS_SCHEMA, R.MUTATE_SCHEMA):
        assert sch["additionalProperties"] is False
        assert sch["required"], "strict schemas need required keys"
    assert R.MUTATE_SCHEMA["properties"]["gene"]["enum"] == list(GENES)
    assert R.POSTMORTEM_SCHEMA["properties"]["predicted_gene"]["enum"] == list(GENES)
    # value must be a string: a string-or-object union is not expressible strictly,
    # and the mutator JSON-encodes fact_format to match.
    assert R.MUTATE_SCHEMA["properties"]["value"]["type"] == "string"


def test_mutate_accepts_json_encoded_fact_format(control):
    """fact_format arrives as a JSON string under the strict schema."""
    from evolution.researcher import Researcher
    from evolution.runner import RunResult
    from evolution.grader import Aggregate
    import json as _j

    class Fake(LLMClient):
        def complete_json(self, *a, **k):
            return {"gene": "fact_format",
                    "value": _j.dumps({"order": "recency"}),
                    "rationale": "test"}

    agg = Aggregate({}, {"F6": 0.4}, [], 0, 1, 0, 0, 100.0, 0, 0.0, 1)
    res = RunResult(control.id, F.Components(), 0, [], {}, agg, {})
    cl = Fake(model="m", base_url="http://10.0.0.1:1234/v1")
    child = Researcher(postmortem_client=cl, synthesis_client=cl).mutate(
        control, res, "", "", "kid", 1)
    assert child is not None
    assert child.fact_format["order"] == "recency"
    # unspecified keys must be inherited, not dropped
    assert child.fact_format["separator"] == control.fact_format["separator"]
    assert child.mutated_genes == ["fact_format"]


def test_lmstudio_shape_schema_ok_json_object_rejected():
    """LM Studio's actual behavior: json_schema accepted, bare json_object 400s
    ("'response_format.type' must be 'json_schema' or 'text'"). A fixed
    schema->object->text ladder would degrade in the wrong direction here."""
    c, sent = _client_recording(reject={"json_object"})
    assert c.complete_json("s", "u", schema=SCHEMA) == {"ok": True}
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert len(sent) == 1, "a supported schema must not trigger renegotiation"
    assert c.schema_mode is True
    # ...and a schemaless call on the same server falls straight through to text
    assert c.complete_json("s", "u") == {"ok": True}
    assert sent[-1].get("response_format") is None


def test_renegotiation_does_not_consume_retry_budget():
    """The preflight runs max_retries=1. If capability negotiation counted as a
    failed attempt, the very first renegotiation would end the call — which is
    exactly the 'all 1 attempts failed: None' bug this guards."""
    c, sent = _client_recording(reject={"json_schema", "json_object"})
    c.max_retries = 1
    assert c.complete_json("s", "u", schema=SCHEMA) == {"ok": True}
    assert len(sent) == 3
