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
    tr = CentroidTracker(lost_after_s=0.0)
    current, new_ids, lost = tr.update([(0.1, 0.1, 0.3, 0.5)])
    assert len(new_ids) == 1 and not lost
    # same box next frame -> same id
    current2, new2, _ = tr.update([(0.11, 0.1, 0.31, 0.5)])
    assert not new2 and current2[0][0] == current[0][0]
    # far box -> new id
    _, new3, _ = tr.update([(0.8, 0.8, 0.95, 0.99)])
    assert len(new3) == 1


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
