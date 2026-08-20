"""Fixture tests — no hardware, no network. Run: pytest tests/ from repo root."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for pkg in ("common", "node_a", "node_c"):
    sys.path.insert(0, str(ROOT / "packages" / pkg))

from sparc_common.types import MOTION_KINDS, Action, ThinkRequest, OptionMeta  # noqa: E402
from sparc_node_c import prompts  # noqa: E402
from sparc_node_a.perception.imx500_tripwire import CentroidTracker  # noqa: E402
from sparc_common import narrative  # noqa: E402


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
    cfg = tmp_path / "sparc.yaml"
    cfg.write_text(
        "bus: {host: 127.0.0.1, port: 1883}\n"
        "endpoints: {cortexd: 'http://127.0.0.1:1'}\n"  # unreachable -> fast fail paths
        f"node_a: {{db_path: {tmp_path}/world.db, think_timeout_s: 1}}\n"
    )
    monkeypatch.setenv("SPARC_CONFIG", str(cfg))
    from sparc_common import config
    config.load.cache_clear()
    sys.path.insert(0, str(ROOT / "packages" / "node_a"))
    from sparc_node_a.orchestrator import Orchestrator
    from sparc_common.types import Transcript

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
    from sparc_node_a.world_model import WorldModel
    w = WorldModel(str(tmp_path / "w.db"))
    eid1, _, _, re1 = w.person_appeared("trk_a")
    assert re1 is False
    w.person_left("trk_a")
    eid2, _, identity, re2 = w.person_appeared("trk_b")  # seconds later, new track id
    assert re2 is True and eid2 == eid1 and identity == "unknown"
    assert w.present[eid2]["identity_provenance"] == "timestamp"


def test_camera_section_only_when_image_attached():
    req = ThinkRequest(deliberation_id="d2", scene="A room.", event="someone waved",
                       image_b64="aGVsbG8=")
    assert "CAMERA:" in prompts.build_think_user(req)
    req2 = ThinkRequest(deliberation_id="d3", scene="A room.", event="idle check")
    assert "CAMERA:" not in prompts.build_think_user(req2)


def test_scene_summarizes_own_speech_keeps_others_verbatim(tmp_path):
    """SPARC's own words never appear verbatim in the scene (self-echo guard);
    other people's words do."""
    from sparc_node_a.world_model import WorldModel
    from sparc_node_a.deliberation import serialize_scene
    w = WorldModel(str(tmp_path / "w.db"))
    w.add_event("user_said", 'someone said: "I love jasmine tea"', [], 0.8)
    w.add_event("sparc_said", 'SPARC said: "A very unique phrase xyzzy"', [], 0.4)
    scene = serialize_scene(w)
    assert "jasmine tea" in scene            # others verbatim
    assert "xyzzy" not in scene              # own words summarized
    assert "SPARC spoke to them" in scene


def test_identity_merge_and_greet_wait(tmp_path):
    """Face embedding resolves a temp entity into the enrolled person."""
    from sparc_node_a.world_model import WorldModel
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
    from sparc_node_a.perception.face_enrich import decode_scrfd, SCRFD_BRANCHES
    import numpy as np
    outs = {}
    for stride, (s, b, k) in SCRFD_BRANCHES.items():
        h = 640 // stride
        outs[s] = np.full((h, h, 2), -8.0, np.float32)   # logits ~ 0 prob
        outs[b] = np.ones((h, h, 8), np.float32)
        outs[k] = np.ones((h, h, 20), np.float32)
    assert decode_scrfd(outs, conf_t=0.5) == []         # nothing confident
    outs[SCRFD_BRANCHES[16][0]][10, 10, 0] = 8.0          # one hot face
    dets = decode_scrfd(outs, conf_t=0.5)
    assert len(dets) == 1
    box, lm, score = dets[0]
    assert score > 0.99 and lm.shape == (5, 2)
    assert box[0] < box[2] and box[1] < box[3]


def test_enroll_face_validation_and_commit(tmp_path):
    """enroll_face: vetoed without samples/valid name; commits with them."""
    from sparc_node_a.world_model import WorldModel
    from sparc_node_a.deliberation import Deliberation, validate
    from sparc_common.types import Action
    w = WorldModel(str(tmp_path / "w.db"))
    eid, *_ = w.person_appeared("trk_m")
    d = Deliberation(event_type="person_speaks", trigger_desc="x", entity_ids=[eid])

    ok, why = validate(w, d, Action(kind="enroll_face", args={"name": "Maya"}))
    assert not ok and "samples" in why                      # no embeddings yet
    ok, _ = validate(w, d, Action(kind="enroll_face", args={"name": "x9!!"}))
    assert not ok                                           # implausible name

    for _ in range(4): w.buffer_face(eid, [1.0] + [0.0] * 511)  # stable samples
    ok, why = validate(w, d, Action(kind="enroll_face", args={"name": "Maya"}))
    assert ok, why
    assert w.enroll_present(eid, "Maya") == eid
    assert w.present[eid]["name"] == "Maya"
    assert "Maya" in w.known_names()
    # second enrollment of the same name gets vetoed
    ok, why = validate(w, d, Action(kind="enroll_face", args={"name": "maya"}))
    assert not ok and "already enrolled" in why


def test_storable_fact_rejects_nonfacts():
    from sparc_node_a.orchestrator import _is_storable_fact
    assert _is_storable_fact("that I like jasmine tea")
    assert _is_storable_fact("my name is Nicholas")
    assert not _is_storable_fact("my face")          # enrollment intent
    assert not _is_storable_fact("my face?")
    assert not _is_storable_fact("do you know me?")  # question
    assert not _is_storable_fact("what I look like")


def test_recent_notable_excludes_presence_churn(tmp_path):
    from sparc_node_a.world_model import WorldModel
    w = WorldModel(str(tmp_path / "w.db"))
    w.add_event("person_entered", "someone new came into view", [], 0.7)
    w.add_event("person_left", "they left SPARC's view", [], 0.3)
    w.add_event("person_left", "they left SPARC's view", [], 0.3)  # dup
    w.add_event("user_said", 'someone said: "hello there"', [], 0.8)
    notable = w.recent_notable_events(3)
    descs = [d for _, _, d in notable]
    assert any("hello there" in d for d in descs)
    assert not any("left SPARC" in d for d in descs)      # presence churn gone
    assert not any("came into view" in d for d in descs)


def test_scene_states_face_not_saved_for_unknown(tmp_path):
    """Scene must tell the model the truth: unknown person's face isn't saved."""
    from sparc_node_a.world_model import WorldModel
    from sparc_node_a.deliberation import serialize_scene
    w = WorldModel(str(tmp_path / "w.db"))
    eid, *_ = w.person_appeared("trk_z")
    scene = serialize_scene(w)
    assert "NOT saved the unrecognized person's face" in scene
    assert "clear look" in scene            # no samples yet
    for _ in range(3):
        w.buffer_face(eid, [1.0] + [0.0] * 511)
    assert "can save their face now" in serialize_scene(w)


def test_face_buffer_survives_presence_churn(tmp_path):
    from sparc_node_a.world_model import WorldModel
    w = WorldModel(str(tmp_path / "w.db"))
    eid, *_ = w.person_appeared("trk_a")
    for _ in range(4):
        w.buffer_face(eid, [1.0] + [0.0] * 511)
    w.person_left("trk_a")                  # churn: left view
    assert len(w.face_samples(eid)) == 4    # buffer survived


@pytest.fixture
def motion_config(tmp_path, monkeypatch):
    from sparc_common import config

    def set_config(motion_yaml: str) -> None:
        cfg = tmp_path / "sparc.yaml"
        cfg.write_text(motion_yaml)
        monkeypatch.setenv("SPARC_CONFIG", str(cfg))
        config.load.cache_clear()

    yield set_config
    config.load.cache_clear()


@pytest.mark.parametrize("motion_kind", sorted(MOTION_KINDS))
def test_disabled_motion_is_vetoed_and_only_fallback_executes(
    tmp_path, motion_config, motion_kind
):
    from sparc_node_a.deliberation import Deliberation, Stage
    from sparc_node_a.orchestrator import Orchestrator
    from sparc_node_a.world_model import WorldModel

    motion_config("motion: {enabled: false}\n")
    world = WorldModel(str(tmp_path / "world.db"))
    orchestrator = object.__new__(Orchestrator)
    orchestrator.world = world
    orchestrator.bus = type(
        "StubBus", (), {"publish_json": lambda self, *args, **kwargs: None}
    )()
    deliberation = Deliberation(event_type="test", trigger_desc="motion request")

    orchestrator._execute_requested(deliberation, Action(kind=motion_kind))

    rows = world.db.execute(
        "SELECT kind, payload FROM trace WHERE deliberation_id=? ORDER BY rowid",
        (deliberation.id,),
    ).fetchall()
    assert [kind for kind, _ in rows] == [
        "requested", "vetoed", "fallback_executed"
    ]
    payloads = [json.loads(payload) for _, payload in rows]
    assert payloads[0]["action"] == motion_kind
    assert payloads[1] == {
        "reason": "motion disabled", "action": motion_kind, "args": {}
    }
    assert payloads[2]["action"] == "wait"
    assert payloads[2]["fallback_level"] == 3
    assert deliberation.stage == Stage.EXECUTED
    assert deliberation.partial_result.kind == "wait"


def test_motion_wire_kind_is_stop_moving_not_stop():
    assert "stop_moving" in MOTION_KINDS
    assert "stop" not in MOTION_KINDS


@pytest.mark.parametrize(
    "motion_yaml",
    ["{}\n", "motion: true\n", "motion: {enabled: 'true'}\n"],
)
def test_absent_or_malformed_motion_config_fails_closed(
    tmp_path, motion_config, motion_yaml
):
    from sparc_node_a.deliberation import Deliberation, validate
    from sparc_node_a.world_model import WorldModel

    motion_config(motion_yaml)
    world = WorldModel(str(tmp_path / "world.db"))
    action = Action(kind=sorted(MOTION_KINDS)[0])
    ok, reason = validate(
        world, Deliberation(event_type="test", trigger_desc="motion request"), action
    )
    assert not ok
    assert reason == "motion disabled"


def test_enabled_motion_without_executor_is_vetoed_and_not_traced_executed(
    tmp_path, motion_config
):
    from sparc_node_a.deliberation import Deliberation
    from sparc_node_a.orchestrator import Orchestrator
    from sparc_node_a.world_model import WorldModel

    motion_config("motion: {enabled: true}\n")
    world = WorldModel(str(tmp_path / "world.db"))
    orchestrator = object.__new__(Orchestrator)
    orchestrator.world = world
    orchestrator.bus = type(
        "StubBus", (), {"publish_json": lambda self, *args, **kwargs: None}
    )()
    action = Action(kind=sorted(MOTION_KINDS)[0])
    deliberation = Deliberation(event_type="test", trigger_desc="motion request")

    orchestrator._execute_requested(deliberation, action)

    rows = world.db.execute(
        "SELECT kind, payload FROM trace WHERE deliberation_id=? ORDER BY rowid",
        (deliberation.id,),
    ).fetchall()
    assert [kind for kind, _ in rows] == [
        "requested", "vetoed", "fallback_executed"
    ]
    assert json.loads(rows[1][1])["reason"] == "motion executor unavailable"
    assert all(kind != "executed" for kind, _ in rows)


def test_motion_fallback_survives_debug_telemetry_failure(
    tmp_path, motion_config
):
    from sparc_node_a.deliberation import Deliberation, Stage
    from sparc_node_a.orchestrator import Orchestrator
    from sparc_node_a.world_model import WorldModel

    class RaisingBus:
        def publish_json(self, *args, **kwargs):
            raise RuntimeError("debug broker unavailable")

    motion_config("motion: {enabled: false}\n")
    world = WorldModel(str(tmp_path / "world.db"))
    orchestrator = object.__new__(Orchestrator)
    orchestrator.world = world
    orchestrator.bus = RaisingBus()
    deliberation = Deliberation(event_type="test", trigger_desc="motion request")

    orchestrator._execute_requested(
        deliberation, Action(kind=sorted(MOTION_KINDS)[0])
    )

    rows = world.db.execute(
        "SELECT kind FROM trace WHERE deliberation_id=? ORDER BY rowid",
        (deliberation.id,),
    ).fetchall()
    assert [kind for kind, in rows] == [
        "requested", "vetoed", "fallback_executed"
    ]
    assert deliberation.stage == Stage.EXECUTED
    assert deliberation.partial_result.kind == "wait"


def test_face_track_association():
    from sparc_node_a.perception.face_enrich import match_face_to_track
    boxes = {"near": (0.4, 0.2, 0.7, 0.95), "far": (0.35, 0.3, 0.8, 1.0),
             "other": (0.0, 0.1, 0.25, 0.9)}
    # face center inside both 'near' and 'far' -> smallest box wins
    assert match_face_to_track((0.55, 0.4), boxes) == "near"
    assert match_face_to_track((0.1, 0.5), boxes) == "other"
    assert match_face_to_track((0.95, 0.5), boxes) is None  # nobody there


def test_enroll_allowed_with_known_person_present(tmp_path):
    """v0.5: enrolling the one unknown works even while a known person is in view."""
    from sparc_node_a.world_model import WorldModel
    from sparc_node_a.deliberation import Deliberation, validate
    from sparc_common.types import Action
    w = WorldModel(str(tmp_path / "w.db"))
    w.enroll_face("Nicholas", [1.0] + [0.0] * 511)
    # Nicholas present (known) + one stranger
    nid, *_ = w.person_appeared("trk_nick", [1.0] + [0.0] * 511)
    sid, name, *_ = w.person_appeared("trk_parent")
    assert w.present[nid]["name"] == "Nicholas" and name is None
    for _ in range(4):
        w.buffer_face(sid, [0.0, 1.0] + [0.0] * 510)
    d = Deliberation(event_type="person_speaks", trigger_desc="x", entity_ids=[sid])
    ok, why = validate(w, d, Action(kind="enroll_face", args={"name": "Luciana"}))
    assert ok, why
    # two unknowns -> ambiguous -> veto
    w.person_appeared("trk_stranger2")
    ok, why = validate(w, d, Action(kind="enroll_face", args={"name": "Bob"}))
    assert not ok and "unclear" in why


def test_fresh_face_evidence_has_conservative_states(tmp_path):
    """Transition rows: no face, positive, uncertain, and below-threshold."""
    import math
    from sparc_node_a.world_model import WorldModel
    from sparc_node_a.deliberation import serialize_scene

    w = WorldModel(str(tmp_path / "w.db"))
    nicholas = w.enroll_face("Nicholas", [1.0, 0.0, 0.0])

    anonymous, name, state, _ = w.person_appeared("no-face")
    assert name is None and state == "unknown"
    assert w.present[anonymous]["identity_provenance"] == "none"
    w.person_left("no-face")

    known, name, state, reappeared = w.person_appeared(
        "positive", [1.0, 0.0, 0.0])
    assert (known, name, state, reappeared) == (
        nicholas, "Nicholas", "known", False)
    assert w.live_name(known) == "Nicholas"
    w.person_left("positive")

    uncertain_vec = [0.5, math.sqrt(0.75), 0.0]
    uncertain, name, state, _ = w.person_appeared("uncertain", uncertain_vec)
    assert uncertain != nicholas and name is None and state == "uncertain"
    scene = serialize_scene(w)
    assert "identity SPARC is uncertain about" in scene
    assert "Nicholas" not in scene
    w.person_left("uncertain")

    negative, name, state, _ = w.person_appeared("negative", [0.0, 0.0, 1.0])
    assert negative not in (nicholas, uncertain)
    assert name is None and state == "unknown"


def test_timestamp_reappearance_never_reuses_enrollment(tmp_path):
    """Transition rows: timestamp continuity is anonymous-only."""
    from sparc_node_a.world_model import WorldModel
    from sparc_node_a.deliberation import serialize_scene

    w = WorldModel(str(tmp_path / "w.db"))
    known = w.enroll_face("Nicholas", [1.0, 0.0, 0.0])
    live_known, *_ = w.person_appeared("known-track", [1.0, 0.0, 0.0])
    assert live_known == known
    w.person_left("known-track")

    stranger, name, state, reappeared = w.person_appeared("stranger-track")
    assert stranger != known
    assert name is None and state == "unknown" and reappeared is False
    assert known not in w.present
    assert "Nicholas" not in serialize_scene(w)

    w.person_left("stranger-track")
    same_stranger, _, state, reappeared = w.person_appeared("stranger-flap")
    assert same_stranger == stranger and state == "unknown" and reappeared is True


def test_known_confirmation_clears_contradiction_and_no_face_retains_binding(tmp_path):
    """Transition rows: contradiction clears on confirmation; no-face is inert."""
    from sparc_node_a.world_model import WorldModel

    w = WorldModel(str(tmp_path / "w.db"))
    known = w.enroll_face("Nicholas", [1.0, 0.0, 0.0])
    w.person_appeared("trk", [1.0, 0.0, 0.0])

    eid, name, state = w.update_identity(known, [0.0, 1.0, 0.0], 0.9)
    assert eid == known and name is None and state == "uncertain"
    assert len(w.contradiction_buffer[known]) == 1

    eid, name, state = w.update_identity(known, [1.0, 0.0, 0.0], 0.9)
    assert (eid, name, state) == (known, "Nicholas", "known")
    assert known not in w.contradiction_buffer
    # No update is made when a rich frame contains no face; the confirmed live
    # binding therefore remains authoritative for this continuous track.
    assert w.live_name(known) == "Nicholas"


def test_two_consistent_other_known_samples_rebind(tmp_path):
    """Transition row: a different enrollment needs two consistent positives."""
    from sparc_node_a.world_model import WorldModel

    w = WorldModel(str(tmp_path / "w.db"))
    nicholas = w.enroll_face("Nicholas", [1.0, 0.0, 0.0])
    maya = w.enroll_face("Maya", [0.0, 1.0, 0.0])
    w.person_appeared("trk", [1.0, 0.0, 0.0])

    eid, name, state = w.update_identity(nicholas, [0.0, 1.0, 0.0], 0.9)
    assert eid == nicholas and name is None and state == "uncertain"
    assert w.live_name(nicholas) is None

    eid, name, state = w.update_identity(nicholas, [0.0, 1.0, 0.0], 0.9)
    assert (eid, name, state) == (maya, "Maya", "known")
    assert nicholas not in w.present
    assert w.present[maya]["track_id"] == "trk"
    assert w.live_name(maya) == "Maya"


def test_uncertain_live_known_suppresses_scene_and_named_greeting(tmp_path):
    """Uncertain evidence retains enrollment but forbids named presentation."""
    import math
    from sparc_node_a.world_model import WorldModel
    from sparc_node_a.deliberation import Deliberation, serialize_scene, validate
    from sparc_common.types import Action

    w = WorldModel(str(tmp_path / "w.db"))
    known = w.enroll_face("Nicholas", [1.0, 0.0, 0.0])
    w.person_appeared("trk", [1.0, 0.0, 0.0])
    uncertain_vec = [0.5, math.sqrt(0.75), 0.0]
    eid, name, state = w.update_identity(known, uncertain_vec, 0.9)
    assert (eid, name, state) == (known, None, "uncertain")
    assert "Nicholas" in w.known_names()  # durable enrollment remains

    scene = serialize_scene(w)
    assert "identity SPARC is uncertain about" in scene
    assert "Nicholas" not in scene
    d = Deliberation(
        event_type="person_enters", trigger_desc="arrival", entity_ids=[known])
    ok, why = validate(
        w, d, Action(kind="say", args={"text": "Hi Nicholas!"}))
    assert not ok and "live target" in why
    ok, why = validate(w, d, Action(kind="say", args={"text": "Oh — hi there!"}))
    assert ok, why


def test_three_stable_high_quality_negatives_split_without_corruption(tmp_path):
    """Three strong negatives move only the live presentation and its evidence."""
    import json
    from sparc_node_a.world_model import WorldModel

    w = WorldModel(str(tmp_path / "w.db"))
    known = w.enroll_face("Nicholas", [1.0, 0.0, 0.0])
    w.person_appeared("trk", [1.0, 0.0, 0.0])
    w.buffer_face(known, [1.0, 0.0, 0.0])
    old_event = w.add_event("observed", "Nicholas waved", [known], 0.6)
    w.last_action_ts[f"greet:{known}"] = 123.0

    negative = [0.0, 1.0, 0.0]
    for _ in range(2):
        eid, name, state = w.update_identity(known, negative, 0.9)
        assert eid == known and name is None and state == "uncertain"
    new_eid, name, state = w.update_identity(known, negative, 0.9)

    assert new_eid != known and name is None and state == "unknown"
    assert known not in w.present and w.present[new_eid]["track_id"] == "trk"
    assert w.db.execute(
        "SELECT name, present FROM entities WHERE id=?", (known,)).fetchone() == (
            "Nicholas", 0)
    assert w.db.execute(
        "SELECT name FROM known_faces WHERE entity_id=?", (known,)).fetchone() == (
            "Nicholas",)
    assert len(w.face_samples(new_eid)) == 3
    assert w.face_samples(known) == [[1.0, 0.0, 0.0]]
    assert w.last_action_ts[f"greet:{known}"] == 123.0

    old_ids = json.loads(w.db.execute(
        "SELECT entity_ids FROM events WHERE id=?", (old_event,)).fetchone()[0])
    assert old_ids == [known]
    new_event = w.add_event("observed", "the visitor waved", [new_eid], 0.6)
    new_ids = json.loads(w.db.execute(
        "SELECT entity_ids FROM events WHERE id=?", (new_event,)).fetchone()[0])
    assert new_ids == [new_eid]


def test_negative_split_requires_detection_quality_and_embedding_agreement(tmp_path):
    """Low-confidence or mutually-inconsistent negatives cannot force a split."""
    from sparc_node_a.world_model import WorldModel

    w = WorldModel(str(tmp_path / "w.db"))
    known = w.enroll_face("Nicholas", [1.0, 0.0, 0.0, 0.0])
    w.person_appeared("trk", [1.0, 0.0, 0.0, 0.0])

    for _ in range(4):
        eid, _, _ = w.update_identity(known, [0.0, 1.0, 0.0, 0.0], 0.2)
        assert eid == known
    inconsistent = (
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    for sample in inconsistent:
        eid, _, _ = w.update_identity(known, sample, 0.9)
        assert eid == known
    assert len(w.contradiction_buffer[known]) == 1


def test_identity_transition_trace_records_evidence_and_entities(tmp_path):
    from sparc_node_a.world_model import WorldModel

    w = WorldModel(str(tmp_path / "w.db"))
    known = w.enroll_face("Nicholas", [1.0, 0.0, 0.0])
    w.person_appeared("trk", [1.0, 0.0, 0.0])
    w.update_identity(known, [0.0, 1.0, 0.0], 0.9)
    rows = w.db.execute(
        "SELECT payload FROM trace WHERE kind='identity_transition' ORDER BY id"
    ).fetchall()
    assert rows
    payloads = [__import__("json").loads(row[0]) for row in rows]
    assert any(p["transition"] == "confirm" and p["to_entity"] == known
               and p["evidence_class"] == "known_match" for p in payloads)
    assert any(p["transition"] == "uncertain" and p["from_entity"] == known
               and p["evidence_class"] == "strong_negative" for p in payloads)


def test_orchestrator_split_schedules_one_unnamed_greeting(tmp_path):
    """A corrected anonymous arrival is handed to the generic greeting path."""
    import time
    from sparc_node_a.world_model import WorldModel
    from sparc_node_a.orchestrator import Orchestrator
    from sparc_common.types import Detection, DetectionFrame

    w = WorldModel(str(tmp_path / "w.db"))
    known = w.enroll_face("Nicholas", [1.0, 0.0, 0.0])
    w.person_appeared("trk", [1.0, 0.0, 0.0])

    o = Orchestrator.__new__(Orchestrator)
    o.world = w
    o._pending_births = {}
    o._pending_person = {}
    submitted = []
    o.submit = submitted.append
    frame = DetectionFrame(
        source="hailo8", scene_delta="periodic",
        detections=[Detection(
            track_id="trk", cls="face", conf=0.9,
            bbox=(0.1, 0.1, 0.2, 0.2),
            face_embedding=[0.0, 1.0, 0.0])])
    for _ in range(3):
        o.on_rich(frame)

    anonymous = w.entity_by_track("trk")
    assert anonymous != known
    assert len(w.face_samples(anonymous)) == 3  # split samples, no duplicate append
    assert anonymous in o._pending_births
    assert o._pending_births[anonymous] > time.time()

    o._pending_births[anonymous] = 0.0
    o._drain_births()
    assert len(submitted) == 1
    assert "someone new" in submitted[0].trigger_desc
    assert "Nicholas" not in submitted[0].trigger_desc


def test_anonymous_tracker_flap_does_not_schedule_second_greeting(tmp_path):
    from sparc_node_a.world_model import WorldModel
    from sparc_node_a.orchestrator import Orchestrator
    from sparc_common.types import Detection, DetectionFrame

    class FakeBus:
        def publish_json(self, *_args, **_kwargs):
            pass

    o = Orchestrator.__new__(Orchestrator)
    o.world = WorldModel(str(tmp_path / "w.db"))
    o.bus = FakeBus()
    o._pending_births = {}
    o._pending_person = {}
    box = (0.1, 0.1, 0.2, 0.2)
    o.on_tier0(DetectionFrame(
        source="imx500", scene_delta="new_track",
        detections=[Detection(
            track_id="first", cls="person", conf=0.9, bbox=box)]))
    anonymous = o.world.entity_by_track("first")
    assert anonymous in o._pending_births

    o.world.last_action_ts[f"greet:{anonymous}"] = 123.0
    o.on_tier0(DetectionFrame(
        source="imx500", scene_delta="lost_track",
        detections=[Detection(
            track_id="first", cls="person", conf=0.0, bbox=box)]))
    assert o._pending_births == {}
    o.on_tier0(DetectionFrame(
        source="imx500", scene_delta="new_track",
        detections=[Detection(
            track_id="flap", cls="person", conf=0.9, bbox=box)]))

    assert o.world.entity_by_track("flap") == anonymous
    assert o._pending_births == {}
    assert o.world.last_action_ts[f"greet:{anonymous}"] == 123.0


@pytest.mark.parametrize("requested", ["Hi Nicholas!", "Hi Beatrice!"])
def test_anonymous_arrival_replaces_ungrounded_name_with_generic_greeting(
        tmp_path, requested):
    """Absent enrollment and arbitrary prose are never executable identities."""
    from sparc_node_a.deliberation import Deliberation, validate
    from sparc_node_a.orchestrator import Orchestrator
    from sparc_node_a.world_model import WorldModel

    class RecordingBus:
        def __init__(self):
            self.spoken = []

        def publish(self, _topic, payload):
            self.spoken.append(payload.text)

        def publish_json(self, *_args, **_kwargs):
            pass

    w = WorldModel(str(tmp_path / "w.db"))
    known = w.enroll_face("Nicholas", [1.0, 0.0, 0.0])
    w.person_appeared("known", [1.0, 0.0, 0.0])
    w.person_left("known")
    stranger, *_ = w.person_appeared("stranger")
    d = Deliberation(
        event_type="person_enters", trigger_desc="arrival", entity_ids=[stranger])
    raw = Action(kind="say", args={"text": requested})

    ok, why = validate(w, d, raw)
    assert not ok and "live target" in why
    o = Orchestrator.__new__(Orchestrator)
    o.world = w
    o.bus = RecordingBus()
    o._execute_requested(d, raw)

    assert o.bus.spoken == ["Oh — hi there!"]
    assert all(name not in o.bus.spoken[0] for name in ("Nicholas", "Beatrice"))
    assert d.partial_result.args["text"] == "Oh — hi there!"
    trace = w.db.execute(
        "SELECT payload FROM trace WHERE deliberation_id=? AND kind='greeting_grounded'",
        (d.id,),
    ).fetchone()
    assert json.loads(trace[0])["live_name"] is None
    assert known not in w.present


def test_confirmed_target_gets_only_its_grounded_named_greeting(tmp_path):
    from sparc_node_a.deliberation import Deliberation, validate
    from sparc_node_a.orchestrator import Orchestrator
    from sparc_node_a.world_model import WorldModel

    class RecordingBus:
        def __init__(self):
            self.spoken = []

        def publish(self, _topic, payload):
            self.spoken.append(payload.text)

        def publish_json(self, *_args, **_kwargs):
            pass

    w = WorldModel(str(tmp_path / "w.db"))
    known = w.enroll_face("Nicholas", [1.0, 0.0, 0.0])
    w.person_appeared("known", [1.0, 0.0, 0.0])
    d = Deliberation(
        event_type="person_enters", trigger_desc="arrival", entity_ids=[known])
    correct = Action(kind="say", args={"text": "Hi Nicholas!"})
    assert validate(w, d, correct) == (True, "ok")
    assert validate(
        w, d, Action(kind="say", args={"text": "Hi Beatrice!"})
    )[0] is False

    o = Orchestrator.__new__(Orchestrator)
    o.world = w
    o.bus = RecordingBus()
    o._execute_requested(
        d, Action(kind="say", args={"text": "Welcome, Beatrice!"}))
    assert o.bus.spoken == ["Hi Nicholas!"]


def test_anonymous_bind_conflict_preserves_incumbent_and_source(tmp_path):
    from sparc_node_a.deliberation import serialize_scene
    from sparc_node_a.world_model import WorldModel

    w = WorldModel(str(tmp_path / "w.db"))
    known = w.enroll_face("Nicholas", [1.0, 0.0, 0.0])
    w.person_appeared("nick-track", [1.0, 0.0, 0.0])
    source, *_ = w.person_appeared("stranger-track")
    w.buffer_face(known, [1.0, 0.0, 0.0])
    w.buffer_face(source, [0.0, 0.0, 1.0])
    old_event = w.add_event("observed", "the stranger waved", [source], 0.6)

    eid, name, state = w.update_identity(source, [1.0, 0.0, 0.0], 0.9)

    assert (eid, name, state) == (source, None, "uncertain")
    assert w.entity_by_track("nick-track") == known
    assert w.entity_by_track("stranger-track") == source
    assert w.present[known]["track_id"] == "nick-track"
    assert w.live_name(known) == "Nicholas" and w.live_name(source) is None
    assert w.face_samples(known) == [[1.0, 0.0, 0.0]]
    assert w.face_samples(source) == [[0.0, 0.0, 1.0]]
    assert "Nicholas" in serialize_scene(w)
    assert "identity SPARC is uncertain about" in serialize_scene(w)
    assert json.loads(w.db.execute(
        "SELECT entity_ids FROM events WHERE id=?", (old_event,)).fetchone()[0]
    ) == [source]
    assert w.db.execute(
        "SELECT track_id, present FROM entities WHERE id=?", (known,)
    ).fetchone() == ("nick-track", 1)
    payload = json.loads(w.db.execute(
        "SELECT payload FROM trace WHERE kind='identity_transition' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()[0])
    assert payload["transition"] == "ownership_conflict"
    assert payload["candidate_entity"] == known

    assert w.person_left("stranger-track") == source
    assert w.entity_by_track("stranger-track") is None
    assert w.entity_by_track("nick-track") == known


def test_known_rebind_conflict_preserves_both_tracks_and_later_leaves(tmp_path):
    from sparc_node_a.deliberation import serialize_scene
    from sparc_node_a.world_model import WorldModel

    w = WorldModel(str(tmp_path / "w.db"))
    nicholas = w.enroll_face("Nicholas", [1.0, 0.0, 0.0])
    maya = w.enroll_face("Maya", [0.0, 1.0, 0.0])
    w.person_appeared("nick-track", [1.0, 0.0, 0.0])
    w.person_appeared("maya-track", [0.0, 1.0, 0.0])
    w.buffer_face(nicholas, [1.0, 0.0, 0.0])
    w.buffer_face(maya, [0.0, 1.0, 0.0])
    event = w.add_event("observed", "both people waved", [nicholas, maya], 0.6)

    first = w.update_identity(nicholas, [0.0, 1.0, 0.0], 0.9)
    second = w.update_identity(nicholas, [0.0, 1.0, 0.0], 0.9)

    assert first == (nicholas, None, "uncertain")
    assert second == (nicholas, None, "uncertain")
    assert w.entity_by_track("nick-track") == nicholas
    assert w.entity_by_track("maya-track") == maya
    assert w.present[maya]["track_id"] == "maya-track"
    assert w.live_name(nicholas) is None and w.live_name(maya) == "Maya"
    assert w.face_samples(nicholas) == [[1.0, 0.0, 0.0]]
    assert w.face_samples(maya) == [[0.0, 1.0, 0.0]]
    scene = serialize_scene(w)
    assert "identity SPARC is uncertain about" in scene and "Maya" in scene
    assert json.loads(w.db.execute(
        "SELECT entity_ids FROM events WHERE id=?", (event,)).fetchone()[0]
    ) == [nicholas, maya]
    payload = json.loads(w.db.execute(
        "SELECT payload FROM trace WHERE kind='identity_transition' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()[0])
    assert payload["transition"] == "ownership_conflict"
    assert payload["evidence_class"] == "different_known_live_conflict"

    assert w.person_left("maya-track") == maya
    assert w.entity_by_track("maya-track") is None
    assert w.entity_by_track("nick-track") == nicholas
    assert w.person_left("nick-track") == nicholas
    assert not w.present


def test_direct_positive_arrival_cannot_claim_already_live_enrollment(tmp_path):
    from sparc_node_a.world_model import WorldModel

    w = WorldModel(str(tmp_path / "w.db"))
    known = w.enroll_face("Nicholas", [1.0, 0.0, 0.0])
    w.person_appeared("incumbent", [1.0, 0.0, 0.0])

    source, name, state, _ = w.person_appeared(
        "conflicting", [1.0, 0.0, 0.0])
    assert source != known and name is None and state == "uncertain"
    assert w.entity_by_track("incumbent") == known
    assert w.entity_by_track("conflicting") == source
