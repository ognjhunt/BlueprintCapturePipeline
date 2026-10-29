# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_task_target_choice.py
#   src/blueprint_pipeline/website_task_masks.py
#   src/blueprint_pipeline/clean_plate_stage.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_preparation_terminal.py
#   src/blueprint_pipeline/website_task_context.py
"""An ambiguous object to move asks the site once, then resumes the same job.

The actual listener, lease, ledger and status reader run. run_e2e stands in for
the orchestrator's stages before the clean-plate refusal; the task-mask
selection is exercised directly on synthetic segmenter tracks. No provider call.
"""
import base64
import json
from datetime import timedelta
from pathlib import Path

import pytest

from blueprint_pipeline import capture_original_owner_observer as observer  # noqa: F401
from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_preparation_terminal as terminal
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline import website_task_target_choice as choice
from blueprint_pipeline.common import PipelineError, StageError, write_json
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_scene_retirement_generations import capture_birth_source_projection
from blueprint_pipeline.website_preparation_status import (
    DELIVERY_FILE, STATUS_V1, STATUS_V4, read_preparation_status, requested_status_version, retain_preparation_wakeup,
)
from blueprint_pipeline.website_task_masks import select_task_track
from tests.test_website_assessment_resume import WIRE_FIXTURE
from tests.test_website_preparation_status import prepared_source  # noqa: F401

SCHEMA = Path(__file__).resolve().parents[1] / "docs/schemas/website_task_target_choice.v1.schema.json"
TARGET = {"target_id": "target-1", "semantic_label": "tote", "task_effect": "manipulated", "disposition": "remove",
          "spatial_evidence": [{"timestamp_seconds": 4.0, "box_xywh_normalized": [0.0, 0.0, 1.0, 1.0]}]}
FRAMES = [{"frame_id": "f-4", "timestamp_seconds": 4.0}]


def _track(track_id, start):
    # Two equally sized instances in one 10x10 view: the anchor box cannot tell them apart.
    runs = [{"start": row * 10 + start, "length": 3} for row in range(4, 7)]
    return {"track_id": track_id, "label": "target-1",
            "observations": [{"source_frame_id": "f-4", "width": 10, "height": 10, "runs": runs}]}


TRACKS = [_track("sam-left", 1), _track("sam-right", 6)]


def _question(scene_id="site-req-1", capture_id="walkthrough-req-1"):
    return choice.build_question(target=TARGET, candidates=TRACKS, frames=FRAMES, scene_id=scene_id,
                                 capture_id=capture_id, source_geometry={"digest": "sha256:" + "6" * 64,
                                 "binding": {"source_video_digest": "sha256:" + "7" * 64}})


def _answer(question, selectors, picked):
    answer = {"schema_version": choice.ANSWER_SCHEMA, **selectors, "question_digest": question["question_digest"],
              "target_id": question["target_id"], "candidate_set_digest": question["candidate_set_digest"],
              "choice": picked, "answered_at": "2026-10-29T12:00:00.000Z"}
    answer["answer_digest"] = cross_runtime_canonical_digest(answer)
    return answer


def test_ambiguous_instances_are_carried_and_only_a_bound_answer_selects_one():
    with pytest.raises(choice.TaskTargetAmbiguous, match="^task_target_track_ambiguous:target-1$") as raised:
        select_task_track(target=TARGET, tracks=TRACKS, frames=FRAMES)
    assert {track["track_id"] for track in raised.value.candidates} == {"sam-left", "sam-right"}
    question = _question()
    right = next(row["candidate_id"] for row in question["candidates"] if "right side" in row["label"])
    site_choice = {"target_id": "target-1", "candidate_set_digest": question["candidate_set_digest"],
                   "candidate_id": right, "question_digest": question["question_digest"]}
    picked = select_task_track(target=TARGET, tracks=TRACKS, frames=FRAMES, site_choice=site_choice)
    assert picked["track_id"] == "sam-right" and picked["selection_basis"] == "site_choice"
    # A different reconstruction (any changed mask) is a different candidate set: the answer never applies.
    changed = [TRACKS[0], {**TRACKS[1], "observations": [{**TRACKS[1]["observations"][0], "runs": [
        {"start": 66, "length": 2}]}]}]
    with pytest.raises(choice.TaskTargetAmbiguous):
        select_task_track(target=TARGET, tracks=changed, frames=FRAMES, site_choice=site_choice)
    with pytest.raises(choice.TaskTargetAmbiguous):
        select_task_track(target={**TARGET, "target_id": "target-2"}, tracks=[{**t, "label": "target-2"}
                          for t in TRACKS], frames=FRAMES, site_choice=site_choice)


def test_question_lists_found_instances_without_pixels_and_only_for_the_moved_object(tmp_path):
    question = _question()
    assert question["semantic_label"] == "Tote" and len(question["candidates"]) == 2
    assert [row["label"] for row in question["candidates"]] == [
        "Tote 1: left side, center of the view", "Tote 2: right side, center of the view"]
    assert all(set(row) == {"candidate_id", "label", "reference_time_ms"} and row["reference_time_ms"] == 4000
               for row in question["candidates"])
    assert "image" not in json.dumps(question) and "runs" not in json.dumps(question)
    pipeline = tmp_path / "walkthrough-req-1" / "pipeline"
    pipeline.mkdir(parents=True)
    plan = {"targets": [TARGET]}
    ambiguous = choice.TaskTargetAmbiguous("target-1", candidates=TRACKS, frames=FRAMES)
    wrapped = StageError("clean_plate", "x")
    wrapped.__cause__ = ambiguous
    assert choice.retain_question(pipeline, exc=wrapped, plan=plan, source_geometry={"digest": "sha256:" + "6" * 64},
                                  scene_id="site-req-1", capture_id="walkthrough-req-1") == (
        "website_task_target_choice_required:target-1")
    assert choice.read_question(pipeline)["candidate_set_digest"] == question["candidate_set_digest"]
    for other in ({**TARGET, "task_effect": "static_contact"}, {**TARGET, "target_class": "person"}):
        assert choice.retain_question(pipeline, exc=ambiguous, plan={"targets": [other]}, source_geometry={},
                                      scene_id="site-req-1", capture_id="walkthrough-req-1") is None
    assert choice.retain_question(pipeline, exc=ValueError("task_target_track_ambiguous:target-1"), plan=plan,
                                  source_geometry={}, scene_id="site-req-1", capture_id="walkthrough-req-1") is None


def test_shared_contract_examples_validate_on_this_side():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text())
    question, answer, decline = schema["examples"]
    for example in schema["examples"]:
        jsonschema.validate(example, schema)
    public = {"kind": "task_target_choice", **{key: _question()[key] for key in (
        "question_digest", "target_id", "semantic_label", "candidate_set_digest", "candidates")}}
    jsonschema.validate(public, schema)
    full = {**question, "capture_id": answer["capture_id"], "scene_id": answer["scene_id"]}
    for item in (answer, decline):
        assert choice.validate_answer(item, question=full) == item
    with pytest.raises(ValueError):
        choice.validate_answer({**answer, "question_digest": "sha256:" + "9" * 64}, question=full)


@pytest.fixture
def parked(prepared_source, monkeypatch):  # noqa: F811 - imported pytest fixture
    root, owner, context, _, _, _ = prepared_source
    birth = capture_birth_source_projection(root)
    payload = json.loads(WIRE_FIXTURE.read_text().replace("walkthrough-req-1", owner["capture_id"]))
    payload["source_finalize"] = {"bucket": owner["bucket"], "object_name": owner["completion_marker"]["object_name"],
        "generation": owner["completion_marker"]["generation"], "event_id": "original-event",
        "event_source": "original-source"}
    payload["source_membership_selector"] = birth["source_membership_selector"]
    payload["pipeline_status_event"]["source_finalize"] = payload["source_finalize"]
    payload["pipeline_handoff_uri"] = (f"gs://{owner['bucket']}/"
        f"{birth['source_membership_selector']['object_name'].rsplit('/', 1)[0]}/pipeline_handoff.json")
    payload["pipeline_status_event"]["pipeline_handoff_uri"] = payload["pipeline_handoff_uri"]
    raw = json.dumps(payload, indent=2).encode()
    (root / listener.JOB_LEDGER_FILENAME).unlink()
    write_json(root / "raw/manifest.json", {"capture_source": "browser_self_capture"})
    monkeypatch.setattr(listener, "stage_handoff_capture", lambda *_, **__: root)
    proposal = {"schema_version": "site_assessment_preparation_proposal.v1", "request_id": owner["request_id"],
        "capture_id": owner["capture_id"], "job_id": "advisory-" + "a" * 64, "run_id": "site-assessment-test",
        "source_key": "sha256:" + "b" * 64, "context_digest": "c" * 64, "packet_sha256": "d" * 64,
        "questions_pending": True, "scope": "scene_preparation_only", "robot_suitability_verified": False,
        "physical_trial_authorized": False}
    monkeypatch.setattr(context_reader, "load_website_scene_sponsorship",
                        lambda **_: {"assessment_preparation_proposal": proposal})
    question = _question(owner["scene_id"], owner["capture_id"])
    runs = []

    def ambiguous_run(**kwargs):
        runs.append(kwargs)
        write_json(root / "pipeline" / choice.QUESTION_FILE, question)
        error = StageError("clean_plate", "website_task_target_choice_required:target-1")
        raise PipelineError(str(error)) from error
    result = listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
                                              run_e2e=ambiguous_run)
    # Parked, not raised: the message is acknowledged and the ledger keeps the exact bytes.
    assert result["status"] == "awaiting_site_target_choice" and result["queue_disposition"] == "parked_for_site_answer"
    ledger = listener._read_job_ledger(root)
    assert choice.armed(ledger) and ledger["attempt_count"] == 1
    assert base64.b64decode(ledger[choice.FIELD]["payload_base64"], validate=True) == raw
    assert ledger[choice.FIELD]["question_digest"] == question["question_digest"]
    retain_preparation_wakeup(root)
    selectors = json.loads((root / DELIVERY_FILE).read_text())["selectors"]
    return root, raw, question, selectors, runs, ambiguous_run


def test_parked_job_asks_once_and_spends_nothing_on_redelivery(parked):
    root, raw, question, selectors, runs, ambiguous_run = parked
    before = listener._read_job_ledger(root)
    again = listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=ambiguous_run)
    assert again["status"] == "awaiting_site_target_choice" and len(runs) == 1
    assert listener._read_job_ledger(root) == before
    status = read_preparation_status(capture_root=root, selectors=selectors, schema_version=STATUS_V4)
    assert status["state"] == "awaiting_inputs" and status["code"] == "preparation_inputs_pending"
    assert status["question"]["question_digest"] == question["question_digest"]
    assert [row["candidate_id"] for row in status["question"]["candidates"]] == [
        row["candidate_id"] for row in question["candidates"]]
    assert len(json.dumps(status)) < 8192 and requested_status_version("4") == STATUS_V4
    assert read_preparation_status(capture_root=root, selectors=selectors, schema_version=STATUS_V1)["state"] == (
        "awaiting_inputs")
    # Waiting on the site is not silence: no overdue/retry-exhausted observation.
    stopped = terminal._parse_utc(before["updated_at"])
    assert terminal.record_unretried_preparation(root, alive_since=stopped, now=stopped + timedelta(days=30)) is None


def test_drain_waits_for_an_answer_and_refuses_a_stale_one(parked, monkeypatch):
    root, _, question, selectors, runs, _ = parked
    calls = []

    def webapp(**kwargs):
        calls.append(kwargs)
        raise ValueError("website_control_task-target-choice_http_404:website_task_target_choice_pending")
    monkeypatch.setattr(context_reader, "website_webapp_request", webapp)
    process_args = {"provider": "openai", "run_e2e": lambda **_: pytest.fail("replayed without an answer")}
    assert choice.reconcile_waiting_target_choices(listener, storage_root=root.parents[4], process_args=process_args,
                                                   now=1000.0) == {"resumed": 0, "waiting": 1, "declined": 0}
    assert calls[0]["operation"] == "task-target-choice" and calls[0]["payload"]["question_digest"] == (
        question["question_digest"])
    # Backed off: the next tick does not ask again yet.
    assert choice.reconcile_waiting_target_choices(listener, storage_root=root.parents[4], process_args=process_args,
                                                   now=1001.0)["waiting"] == 1 and len(calls) == 1
    stale = _answer({**question, "question_digest": "sha256:" + "8" * 64}, selectors,
                    {"kind": "candidate", "candidate_id": question["candidates"][0]["candidate_id"]})
    monkeypatch.setattr(context_reader, "website_webapp_request", lambda **_: stale)
    assert choice.reconcile_waiting_target_choices(listener, storage_root=root.parents[4], process_args=process_args,
                                                   now=5000.0)["resumed"] == 0
    assert not (root / "pipeline" / choice.ANSWER_FILE).exists() and len(runs) == 1


def test_none_of_these_ends_the_wait_without_a_replay(parked, monkeypatch):
    root, _, question, selectors, runs, _ = parked
    monkeypatch.setattr(context_reader, "website_webapp_request",
                        lambda **_: _answer(question, selectors, {"kind": "none_of_these"}))
    process_args = {"provider": "openai", "run_e2e": lambda **_: pytest.fail("declined question replayed")}
    assert choice.reconcile_waiting_target_choices(listener, storage_root=root.parents[4],
                                                   process_args=process_args) == {"resumed": 0, "waiting": 0, "declined": 1}
    status = read_preparation_status(capture_root=root, selectors=selectors, schema_version=STATUS_V4)
    # Still parked on the same question; the WebApp shows the site's own answer for it.
    assert status["state"] == "awaiting_inputs" and status["question"]["question_digest"] == question["question_digest"]
    assert choice.load_target_choices(root / "pipeline") == {}
    assert choice.reconcile_waiting_target_choices(listener, storage_root=root.parents[4],
                                                   process_args=process_args)["declined"] == 1


def test_a_bound_answer_resumes_the_same_job_from_retained_work(parked, monkeypatch):
    root, _, question, selectors, runs, _ = parked
    picked = question["candidates"][1]["candidate_id"]
    monkeypatch.setattr(context_reader, "website_webapp_request",
                        lambda **_: _answer(question, selectors, {"kind": "candidate", "candidate_id": picked}))
    replays = []

    def run_e2e(**kwargs):
        replays.append({**kwargs, "choices": choice.load_target_choices(root / "pipeline")})
        return {"status": "completed"}
    process_args = {"provider": "openai", "run_evaluation_prep": False, "run_e2e": run_e2e}
    assert choice.reconcile_waiting_target_choices(listener, storage_root=root.parents[4],
                                                   process_args=process_args)["resumed"] == 1
    assert len(replays) == 1 and replays[0]["resume_completed_stages"] is True
    assert replays[0]["capture_root"] == str(root)
    assert replays[0]["choices"] == {"target-1": {"target_id": "target-1", "candidate_id": picked,
        "candidate_set_digest": question["candidate_set_digest"], "question_digest": question["question_digest"]}}
    ledger = listener._read_job_ledger(root)
    # Same capture, same handoff: the claim consumed the record; the failure row keeps the bytes.
    assert ledger["status"] == "completed" and ledger["attempt_count"] == 2 and choice.FIELD not in ledger
    assert ledger["attempt_history"][0][choice.FIELD]["question_digest"] == question["question_digest"]
    assert choice.reconcile_waiting_target_choices(listener, storage_root=root.parents[4],
                                                   process_args=process_args)["resumed"] == 0
    assert len(replays) == 1


def test_only_the_whole_choice_failure_parks_the_job():
    whole = StageError("clean_plate", "website_task_target_choice_required:target-1")
    assert choice.required_choice(PipelineError(str(whole))) == "target-1"
    assert choice.required_choice(StageError("clean_plate", "privacy_blocked,website_task_target_choice_required:t")) is None
    assert choice.required_choice(ValueError("task_target_track_ambiguous:target-1")) is None


def test_clean_plate_asks_instead_of_failing_and_hands_the_answer_to_the_masks(tmp_path, monkeypatch):
    from blueprint_pipeline.clean_plate_removal_analysis_gemini import build_removal_plan
    from blueprint_pipeline.clean_plate_stage import run_clean_plate_stage
    from tests.test_clean_plate_stage import _ANALYSIS_ATTR, _make_capture
    capture_root = _make_capture(tmp_path)
    source = capture_root / "raw/walkthrough.mp4"
    plan = build_removal_plan(targets=[{**TARGET, "target_class": "movable_object", "target_role": "task_object",
        "rebuild_intent": "rebuild_and_compose", "confidence": 0.9}], status="completed", model="test",
        processing="agentic")
    geometry = {"status": "estimated", "digest": "sha256:" + "6" * 64, "frames": [{"frame_id": "f-4"}],
                "binding": {"source_video_digest": "sha256:" + "7" * 64}}
    seen = []

    def masks(**kwargs):
        seen.append(kwargs["target_choices"])
        if not kwargs["target_choices"]:
            raise choice.TaskTargetAmbiguous("target-1", candidates=TRACKS, frames=FRAMES)
        return {"targets": []}
    monkeypatch.setattr(_ANALYSIS_ATTR, lambda **_: plan)
    monkeypatch.setattr("blueprint_pipeline.clean_plate_stage.prepare_website_source_frames", lambda **_: geometry)
    monkeypatch.setattr("blueprint_pipeline.clean_plate_stage.run_website_task_masks", masks)
    result = run_clean_plate_stage(capture_root=capture_root, website_source_video=source,
                                   privacy_processing={"status": "pending_website_review"})
    assert result["status"] == "blocked" and result["blockers"] == ["website_task_target_choice_required:target-1"]
    question = choice.read_question(capture_root / "pipeline")
    assert question["capture_id"] == "c1" and question["scene_id"] == "s1"
    selectors = {"request_id": "req-1", "scene_id": "s1", "capture_id": "c1",
                 "completion_marker_generation": "1", "producer_delivery_key": "sha256:" + "3" * 64,
                 "source_payload_sha256": "4" * 64, "task_context_digest": "sha256:" + "5" * 64}
    picked = question["candidates"][0]["candidate_id"]
    write_json(capture_root / "pipeline" / choice.ANSWER_FILE,
               _answer(question, selectors, {"kind": "candidate", "candidate_id": picked}))
    result = run_clean_plate_stage(capture_root=capture_root, website_source_video=source,
                                   privacy_processing={"status": "pending_website_review"})
    assert result["blockers"] == [] and seen[-1]["target-1"]["candidate_id"] == picked
