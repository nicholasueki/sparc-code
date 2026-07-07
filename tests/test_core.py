"""Fixture tests — no hardware, no network. Run: pytest tests/ from repo root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for pkg in ("common", "node_a", "node_c"):
    sys.path.insert(0, str(ROOT / "packages" / pkg))

from lucas_common.types import ThinkRequest, OptionMeta  # noqa: E402
from lucas_node_c import prompts  # noqa: E402
from lucas_node_a.perception.imx500_tripwire import CentroidTracker  # noqa: E402
from lucas_common import narrative  # noqa: E402


def test_parse_think_happy_path():
    raw = """<think>reasoning here</think>
    {"options":[{"idx":0,"action":"say","args":{"text":"Hi Nicholas!"},"tone":"warm","risk":"low","novelty":0.2},
    {"idx":1,"action":"wait","args":{}}],"choice":0,"backup":1,"why":"friendly greeting"}"""
    parsed = prompts.parse_think(raw, 5)
    assert parsed is not None
    options, choice, backup, why = parsed
    assert choice == 0 and backup == 1
    assert options.options[0].action == "say"
    kinds = {o.action for o in options.options}
    assert "wait" in kinds and "ask_user" in kinds  # deterministic injection


def test_parse_think_clamps_bad_indices():
    raw = '{"options":[{"idx":0,"action":"wait","args":{}}],"choice":9,"backup":9,"why":"x"}'
    options, choice, backup, _ = prompts.parse_think(raw, 5)
    n = len(options.options)
    assert 0 <= choice < n and 0 <= backup < n and choice != backup


def test_parse_think_garbage_returns_none():
    assert prompts.parse_think("no json here at all", 5) is None
    assert prompts.parse_think('{"not_options": true}', 5) is None


def test_extract_json_fenced_and_nested():
    raw = 'text ```json\n{"a": {"b": [1,2]}, "c": "}"}\n``` trailing'
    assert prompts.extract_json(raw) == {"a": {"b": [1, 2]}, "c": "}"}


def test_build_think_user_sections():
    req = ThinkRequest(deliberation_id="d1", scene="A room.", memory="Nicholas likes tea.",
                       conversation=[{"role": "user", "text": "hello"}], event="someone came in")
    text = prompts.build_think_user(req)
    for section in ("SCENE:", "MEMORY", "CONVERSATION:", "EVENT:", "TASK:"):
        assert section in text
    assert "COMPLETE list" in text  # closed-world memory declaration (anti-fabrication)


def test_tracker_new_and_lost():
    tr = CentroidTracker(lost_after_s=5.0)
    box = (0.1, 0.1, 0.3, 0.5)
    born = []
    for _ in range(CentroidTracker.MIN_HITS):
        current, new_ids, _ = tr.update([box])
        born += new_ids
    assert len(born) == 1 and current[0][0] == born[0]
    # same box next frame -> same id, no new birth
    current2, new2, _ = tr.update([box])
    assert not new2 and current2[0][0] == born[0]


def test_narrative_buckets():
    import time
    now = time.time()
    assert narrative.ago(now - 10, now) == "just now"
    assert narrative.ago(now - 300, now) == "a few minutes ago"
    assert "day" in narrative.ago(now - 86400 * 3, now)


def test_option_to_action():
    opt = OptionMeta(idx=2, action="say", args={"text": "hi"})
    act = prompts.option_to_action(opt, "why", 0)
    assert act.kind == "say" and act.option_idx == 2


def test_on_transcript_queues_deliberation(tmp_path, monkeypatch):
    """Regression: a transcript must BOTH ingest the event and queue a deliberation
    (a bad edit once stranded the submit() as dead code — nothing ever thought)."""
    cfg = tmp_path / "lucas.yaml"
    cfg.write_text(
        "bus: {host: 127.0.0.1, port: 1883}\n"
        "endpoints: {cortexd: 'http://127.0.0.1:1'}\n"  # unreachable -> fast fail paths
        f"node_a: {{db_path: {tmp_path}/world.db, think_timeout_s: 1}}\n"
    )
    monkeypatch.setenv("LUCAS_CONFIG", str(cfg))
    from lucas_common import config
    config.load.cache_clear()
    sys.path.insert(0, str(ROOT / "packages" / "node_a"))
    from lucas_node_a.orchestrator import Orchestrator
    from lucas_common.types import Transcript

    o = Orchestrator()
    o.on_transcript(Transcript(text="remember that the towels live in the hall closet"))
    assert len(o._queue) == 1, "transcript must queue a person_speaks deliberation"
    assert o._queue[0][2].event_type == "person_speaks"
    facts = o.world.facts_for_prompt(5)
    assert any("towels" in f for f in facts), "deterministic remember must commit"
    config.load.cache_clear()


def test_tracker_debounce_and_merge():
    """One-frame jitter must not birth tracks; overlapping boxes are one person."""
    tr = CentroidTracker(lost_after_s=5.0)
    # single flicker frame -> nothing born
    _, born, _ = tr.update([(0.4, 0.2, 0.6, 0.9)])
    assert born == []
    # persists 2 more frames -> born once
    _, born2, _ = tr.update([(0.41, 0.2, 0.61, 0.9)])
    _, born3, _ = tr.update([(0.42, 0.2, 0.62, 0.9)])
    assert born2 == [] and len(born3) == 1
    # duplicate overlapping detection -> still one track
    current, born4, _ = tr.update([(0.42, 0.2, 0.62, 0.9), (0.43, 0.22, 0.63, 0.88)])
    assert born4 == [] and len(current) == 1


def test_person_reappearance_object_permanence(tmp_path):
    from lucas_node_a.world_model import WorldModel
    w = WorldModel(str(tmp_path / "w.db"))
    eid1, _, _, re1 = w.person_appeared("trk_a")
    assert re1 is False
    w.person_left("trk_a")
    eid2, _, identity, re2 = w.person_appeared("trk_b")  # seconds later, new track id
    assert re2 is True and eid2 == eid1 and identity == "reappeared"
