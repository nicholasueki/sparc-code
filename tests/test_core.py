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
    """Sub-threshold jitter must not birth tracks; overlapping boxes are one person."""
    tr = CentroidTracker(lost_after_s=5.0)
    box = (0.4, 0.2, 0.6, 0.9)
    # up to MIN_HITS-1 consecutive frames -> nothing born
    for _ in range(CentroidTracker.MIN_HITS - 1):
        _, born, _ = tr.update([box])
        assert born == []
    # one more frame -> born exactly once
    _, born_final, _ = tr.update([box])
    assert len(born_final) == 1
    # duplicate overlapping detection -> still one track, no second birth
    current, born_dup, _ = tr.update([box, (0.43, 0.22, 0.63, 0.88)])
    assert born_dup == [] and len(current) == 1
    # tiny box (below MIN_AREA) is ignored entirely
    _, born_tiny, _ = tr.update([(0.5, 0.5, 0.55, 0.58)])
    assert born_tiny == []


def test_person_reappearance_object_permanence(tmp_path):
    from lucas_node_a.world_model import WorldModel
    w = WorldModel(str(tmp_path / "w.db"))
    eid1, _, _, re1 = w.person_appeared("trk_a")
    assert re1 is False
    w.person_left("trk_a")
    eid2, _, identity, re2 = w.person_appeared("trk_b")  # seconds later, new track id
    assert re2 is True and eid2 == eid1 and identity == "reappeared"


def test_camera_section_only_when_image_attached():
    req = ThinkRequest(deliberation_id="d2", scene="A room.", event="someone waved",
                       image_b64="aGVsbG8=")
    assert "CAMERA:" in prompts.build_think_user(req)
    req2 = ThinkRequest(deliberation_id="d3", scene="A room.", event="idle check")
    assert "CAMERA:" not in prompts.build_think_user(req2)


def test_scene_summarizes_own_speech_keeps_others_verbatim(tmp_path):
    """Lucas's own words never appear verbatim in the scene (self-echo guard);
    other people's words do."""
    from lucas_node_a.world_model import WorldModel
    from lucas_node_a.deliberation import serialize_scene
    w = WorldModel(str(tmp_path / "w.db"))
    w.add_event("user_said", 'someone said: "I love jasmine tea"', [], 0.8)
    w.add_event("lucas_said", 'Lucas said: "A very unique phrase xyzzy"', [], 0.4)
    scene = serialize_scene(w)
    assert "jasmine tea" in scene            # others verbatim
    assert "xyzzy" not in scene              # own words summarized
    assert "Lucas spoke to them" in scene


def test_identity_merge_and_greet_wait(tmp_path):
    """Face embedding resolves a temp entity into the enrolled person."""
    from lucas_node_a.world_model import WorldModel
    w = WorldModel(str(tmp_path / "w.db"))
    emb = [1.0] + [0.0] * 511
    known = w.enroll_face("Nicholas", emb)
    eid, name, identity, _ = w.person_appeared("trk_x")   # unknown at birth
    assert name is None
    new_eid, name2, quality = w.update_identity(eid, emb)
    assert new_eid == known and name2 == "Nicholas" and quality == "known"
    assert w.present[known]["name"] == "Nicholas"
    assert eid not in w.present  # temp entity absorbed


def test_scrfd_decode_shapes():
    from lucas_node_a.perception.face_enrich import decode_scrfd, SCRFD_BRANCHES
    import numpy as np
    outs = {}
    for stride, (s, b, k) in SCRFD_BRANCHES.items():
        h = 640 // stride
        outs[s] = np.full((h, h, 2), -8.0, np.float32)   # logits ~ 0 prob
        outs[b] = np.ones((h, h, 8), np.float32)
        outs[k] = np.ones((h, h, 20), np.float32)
    assert decode_scrfd(outs, conf_t=0.5) is None        # nothing confident
    outs[SCRFD_BRANCHES[16][0]][10, 10, 0] = 8.0          # one hot face
    det = decode_scrfd(outs, conf_t=0.5)
    assert det is not None
    box, lm, score = det
    assert score > 0.99 and lm.shape == (5, 2)
    assert box[0] < box[2] and box[1] < box[3]
