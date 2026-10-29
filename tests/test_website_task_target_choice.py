# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_task_target_choice.py
#   src/blueprint_pipeline/website_task_masks.py
#   src/blueprint_pipeline/clean_plate_stage.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_reconstruction_resume.py
#   src/blueprint_pipeline/website_task_context.py
#   src/blueprint_pipeline/run_e2e.py
"""An ambiguous object to move is guessed, the job continues, and the site may correct it.

The actual listener, lease, ledger, status reader and drains run. run_e2e stands
in for the orchestrator's stages; task-mask selection runs directly on synthetic
segmenter tracks. No provider call.
"""
import base64
import json
from pathlib import Path

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_reconstruction_resume as reconstruction
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline import website_task_target_choice as choice
from blueprint_pipeline.common import PipelineError, StageError, write_json
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_scene_retirement_generations import capture_birth_source_projection
from blueprint_pipeline.website_atlas_preview_store import JOB_FILE, SCHEMA as JOB_SCHEMA
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
GEOMETRY = {"digest": "sha256:" + "6" * 64, "binding": {"source_video_digest": "sha256:" + "7" * 64}}


def _track(track_id, start, label="target-1", frames=("f-4",)):
    # Equally sized instances in a 10x10 view: the anchor box cannot tell them apart.
    runs = [{"start": row * 10 + start, "length": 3} for row in range(4, 7)]
    return {"track_id": track_id, "label": label,
            "observations": [{"source_frame_id": frame, "width": 10, "height": 10, "runs": runs} for frame in frames]}


TRACKS = [_track("sam-left", 1), _track("sam-right", 6)]


def _guess(target=TARGET, tracks=TRACKS):
    guessed = choice.best_supported_guess(target_id=target["target_id"], candidates=tracks, anchor_scores={})
    return {"target": target, "candidates": tracks, "frames": FRAMES, "guessed_track_id": guessed["track_id"]}


def _question(scene_id="site-req-1", capture_id="walkthrough-req-1", guess=None):
    guess = guess or _guess()
    return choice.build_question(target=guess["target"], candidates=guess["candidates"], frames=FRAMES,
        guessed_track_id=guess["guessed_track_id"], scene_id=scene_id, capture_id=capture_id, source_geometry=GEOMETRY)


def _answer(question, selectors, picked):
    answer = {"schema_version": choice.ANSWER_SCHEMA, **selectors, "question_digest": question["question_digest"],
              "target_id": question["target_id"], "candidate_set_digest": question["candidate_set_digest"],
              "guessed_candidate_id": question["guessed_candidate_id"], "choice": picked,
              "answered_at": "2026-10-29T12:00:00.000Z"}
    answer["answer_digest"] = cross_runtime_canonical_digest(answer)
    return answer


def _other(question):
    return next(row["candidate_id"] for row in question["candidates"] if row["candidate_id"] != question["guessed_candidate_id"])


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


def test_best_supported_guess_follows_the_documented_evidence_order():
    left, right = TRACKS
    wider = _track("sam-wide", 1, frames=("f-4", "f-5"))
    # 1. anchor overlap decides first.
    assert choice.best_supported_guess(target_id="target-1", candidates=[left, right],
                                       anchor_scores={"sam-right": 0.2, "sam-left": 0.1})["track_id"] == "sam-right"
    # 2. then the instance visible in more frames.
    assert choice.best_supported_guess(target_id="target-1", candidates=[left, wider],
                                       anchor_scores={})["track_id"] == "sam-wide"
    # 3. then the larger mean visible area.
    big = {**right, "observations": [{**right["observations"][0], "runs": [{"start": 0, "length": 40}]}]}
    assert choice.best_supported_guess(target_id="target-1", candidates=[left, big], anchor_scores={})["track_id"] == "sam-right"
    # 4. then a stable digest order, whatever order the segmenter returned them in.
    first = choice.best_supported_guess(target_id="target-1", candidates=[left, right], anchor_scores={})
    assert choice.best_supported_guess(target_id="target-1", candidates=[right, left], anchor_scores={})["track_id"] == (
        first["track_id"])
    assert first["selection_basis"] == "best_supported_guess"


def test_task_masks_continue_on_the_best_guess_and_report_it(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_masks as masks
    registry = [{"source_frame_id": "f-4", "decoded_pts_seconds": 4.0, "width": 10, "height": 10},
                {"source_frame_id": "f-5", "decoded_pts_seconds": 5.0, "width": 10, "height": 10}]
    tracks = [_track("sam-left", 1), _track("sam-right", 6, frames=("f-4", "f-5"))]
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    monkeypatch.setattr(masks, "prepare_continuous_video", lambda **kw: (registry, {"path": "prepared.mp4"}))
    monkeypatch.setattr(masks, "run_meta_sam31", lambda **kw: {"tracks": tracks})

    def run(**extra):
        return masks.run_website_task_masks(plan={"targets": [TARGET], "task_context_sha256": "task"},
            source_geometry={"digest": "source", "geometry_available": False, "binding": {"source_video_digest": "video"},
                             "frames": [{"frame_id": "f-4", "timestamp_seconds": 4.0, "width": 10, "height": 10}]},
            source_video=tmp_path / "source.mov", output_root=tmp_path / "masks", **extra)
    guesses = []
    result = run(target_guesses=guesses)
    # Not a stop: the run goes on with the instance seen in more frames, and says it guessed.
    assert result["status"] == "completed" and result["targets"][0]["track"]["track_id"] == "sam-right"
    assert result["targets"][0]["source_track"]["selection_basis"] == "best_supported_guess"
    assert [row["guessed_track_id"] for row in guesses] == ["sam-right"]
    question = choice.build_question(target=TARGET, candidates=guesses[0]["candidates"], frames=guesses[0]["frames"],
        guessed_track_id="sam-right", scene_id="s", capture_id="c", source_geometry=GEOMETRY)
    left = next(row["candidate_id"] for row in question["candidates"] if row["candidate_id"] != question["guessed_candidate_id"])
    again = []
    corrected = run(target_guesses=again, target_choices={"target-1": {"target_id": "target-1",
        "candidate_set_digest": question["candidate_set_digest"], "candidate_id": left,
        "question_digest": question["question_digest"]}})
    assert corrected["targets"][0]["track"]["track_id"] == "sam-left" and again == []
    assert corrected["targets"][0]["source_track"]["selection_basis"] == "site_choice"


def test_questions_are_kept_per_target_without_pixels(tmp_path):
    pipeline = tmp_path / "walkthrough-req-1" / "pipeline"
    second = {**TARGET, "target_id": "target-2", "semantic_label": "lid"}
    digests = choice.retain_questions(pipeline, guesses=[_guess(), _guess(second, [_track(t["track_id"], 1 if t is TRACKS[0]
        else 6, label="target-2") for t in TRACKS]), _guess({**TARGET, "target_id": "kept", "task_effect": "static_contact"})],
        source_geometry=GEOMETRY, scene_id="site-req-1", capture_id="walkthrough-req-1")
    questions = choice.current_questions(pipeline)
    assert [q["target_id"] for q in questions] == ["target-1", "target-2"] and len(digests) == 2
    question = questions[0]
    assert question["guessed_candidate_id"] in {row["candidate_id"] for row in question["candidates"]}
    assert [row["label"] for row in question["candidates"]] == [
        "Tote 1: left side, center of the view", "Tote 2: right side, center of the view"]
    assert "image" not in json.dumps(questions) and "runs" not in json.dumps(questions)
    # A later run with no guess leaves no current question, though the files stay.
    choice.retain_questions(pipeline, guesses=[], source_geometry=GEOMETRY, scene_id="site-req-1",
                            capture_id="walkthrough-req-1")
    assert choice.current_questions(pipeline) == [] and choice.read_question(pipeline, "target-1") == question


def test_shared_contract_examples_validate_on_this_side():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text())
    question, other, confirm, decline = schema["examples"]
    for example in schema["examples"]:
        jsonschema.validate(example, schema)
    public = choice.public_question  # the projection keys match the contract
    assert public is not None
    full = {**question, "capture_id": other["capture_id"], "scene_id": other["scene_id"]}
    for item in (other, confirm, decline):
        assert choice.validate_answer(item, question=full) == item
    for forged in ({**confirm, "choice": {"kind": "confirm_guess", "candidate_id": other["choice"]["candidate_id"]}},
                   {**other, "choice": {"kind": "other_candidate", "candidate_id": question["guessed_candidate_id"]}},
                   {**other, "question_digest": "sha256:" + "9" * 64}):
        with pytest.raises(ValueError):
            choice.validate_answer(forged, question=full)


def _job(root, state, **fields):
    write_json(root / "pipeline/atlas" / JOB_FILE, {"schema_version": JOB_SCHEMA, "state": state, **fields})


@pytest.fixture
def guessed(prepared_source, monkeypatch):  # noqa: F811 - imported pytest fixture
    """A delivery whose run guessed the object to move and then stopped on an ordinary retryable refusal."""
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
    runs = []

    def guessing_run(guesses=(_guess(),), error="website_native_intake_pending"):
        def run_e2e(**kwargs):
            runs.append({**kwargs, "choices": choice.load_target_choices(root / "pipeline")})
            choice.retain_questions(root / "pipeline", guesses=list(guesses), source_geometry=GEOMETRY,
                                    scene_id=owner["scene_id"], capture_id=owner["capture_id"])
            if error is None:
                return {"status": "completed"}
            raise PipelineError(str(StageError("website_scene_preparation", error)))
        return run_e2e
    with pytest.raises(PipelineError, match="website_native_intake_pending"):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=guessing_run())
    ledger = listener._read_job_ledger(root)
    assert ledger["status"] == "failed_retryable" and ledger["attempt_count"] == 1
    assert base64.b64decode(ledger[choice.REPLAY_FIELD]["payload_base64"], validate=True) == raw
    retain_preparation_wakeup(root)
    selectors = json.loads((root / DELIVERY_FILE).read_text())["selectors"]
    return root, raw, choice.current_questions(root / "pipeline")[0], selectors, runs, guessing_run


def _webapp(monkeypatch, answers):
    calls = []

    def respond(**kwargs):
        calls.append(kwargs)
        found = answers.get(kwargs["payload"]["question_digest"])
        if found is None:
            raise ValueError("website_control_task-target-choice_http_404:website_task_target_choice_pending")
        return found
    monkeypatch.setattr(context_reader, "website_webapp_request", respond)
    return calls


def test_the_guess_is_published_without_holding_the_job(guessed):
    root, raw, question, selectors, runs, guessing_run = guessed
    status = read_preparation_status(capture_root=root, selectors=selectors, schema_version=STATUS_V4)
    # Optional correction: the state is the job's own, not a wait on the site.
    assert status["state"] == "failed_retryable" and status["question"]["guessed_candidate_id"] == (
        question["guessed_candidate_id"])
    assert len(json.dumps(status)) < 8192 and requested_status_version("4") == STATUS_V4
    assert read_preparation_status(capture_root=root, selectors=selectors, schema_version=STATUS_V1)["state"] == (
        "failed_retryable")
    # A redelivery claims and runs as before: no gate on the answer.
    with pytest.raises(PipelineError):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=guessing_run())
    assert len(runs) == 2 and listener._read_job_ledger(root)["attempt_count"] == 2


def test_confirming_the_guess_records_it_and_reruns_nothing(guessed, monkeypatch):
    root, _, question, selectors, runs, _ = guessed
    _webapp(monkeypatch, {question["question_digest"]: _answer(question, selectors,
        {"kind": "confirm_guess", "candidate_id": question["guessed_candidate_id"]})})
    process_args = {"provider": "openai", "run_e2e": lambda **_: pytest.fail("a confirmation reran the job")}
    counts = choice.reconcile_target_choices(listener, storage_root=root.parents[4], process_args=process_args)
    assert counts["confirmed"] == 1 and counts["replayed"] == 0
    assert choice.load_target_choices(root / "pipeline")["target-1"]["candidate_id"] == question["guessed_candidate_id"]
    assert choice.correction_binding_digest(root / "pipeline") is None and choice.public_question(root) is None


def test_a_stale_answer_is_refused_and_asked_again_later(guessed, monkeypatch):
    root, _, question, selectors, runs, _ = guessed
    stale = _answer({**question, "question_digest": "sha256:" + "8" * 64}, selectors,
                    {"kind": "other_candidate", "candidate_id": _other(question)})
    calls = _webapp(monkeypatch, {question["question_digest"]: stale})
    process_args = {"provider": "openai", "run_e2e": lambda **_: pytest.fail("stale answer reran the job")}
    assert choice.reconcile_target_choices(listener, storage_root=root.parents[4], process_args=process_args,
                                           now=1000.0)["waiting"] == 1
    assert choice.read_answer(root / "pipeline", question) is None and len(calls) == 1
    choice.reconcile_target_choices(listener, storage_root=root.parents[4], process_args=process_args, now=1001.0)
    assert len(calls) == 1  # backed off


def test_negative_answer_replays_original_job_once_to_a_truthful_evidence_limit(guessed, monkeypatch):
    from types import SimpleNamespace
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    root, raw, question, selectors, runs, guessing_run = guessed
    source = root / "raw/retained-selection-source.mp4"
    source.write_bytes(b"synthetic retained source for negative-answer recovery")
    question = {**question, "source_video_digest": _sha256_file(source)}
    question["question_digest"] = canonical_digest(question, digest_field="question_digest")
    write_json(choice._path(root / "pipeline", question["target_id"], "question"), question)
    write_json(root / "pipeline" / choice.QUESTION_DIR / choice.INDEX_FILE, {
        "schema_version": choice.INDEX_SCHEMA, "questions": {question["target_id"]: question["question_digest"]}})
    answers = {question["question_digest"]: _answer(question, selectors, {"kind": "none_of_these"})}
    _webapp(monkeypatch, answers)
    before = listener._read_job_ledger(root)
    def no_provider(*_args, **_kwargs):
        pytest.fail("a rejected guessed target must not buy or run another model/native scene")
    def process_negative(**kwargs):
        runs.append(kwargs)
        context = context_reader.load_current_website_task_context(request_id=selectors["request_id"],
            scene_id=selectors["scene_id"], capture_id=selectors["capture_id"], purpose="scene_preparation")
        return run_website_capture_preparation(descriptor=SimpleNamespace(metadata={"site_task_context": context}),
            capture_root=root, pipeline_dir=root / "pipeline", source_video=source,
            derived_scene_generation_allowed=True, gates=[], prepare_media=no_provider,
            check_privacy=no_provider, make_gate=no_provider)
    process_args = {"provider": "openai", "run_e2e": process_negative}
    counts = choice.reconcile_target_choices(listener, storage_root=root.parents[4], process_args=process_args)
    assert counts["declined"] == 1 and counts["replayed"] == 1
    after = listener._read_job_ledger(root)
    assert after["status"] == "completed" and after["attempt_count"] == before["attempt_count"] + 1
    assert after["attempt_history"][0] == before["attempt_history"][0]
    assert base64.b64decode(after[choice.REPLAY_FIELD]["payload_base64"], validate=True) == raw
    status = read_preparation_status(capture_root=root, selectors=selectors, schema_version=STATUS_V4)
    assert status["state"] == "awaiting_inputs" and status["outcome"]["category"] == "insufficient_evidence"
    assert choice.reconcile_target_choices(listener, storage_root=root.parents[4],
                                          process_args=process_args)["replayed"] == 0
    assert len(runs) == 2  # Original attempt and one no-spend rejection outcome.
    answers[question['question_digest']] = _answer(question, selectors, {
        'kind': 'confirm_guess', 'candidate_id': question['guessed_candidate_id']})
    original = guessing_run(guesses=(), error=None)
    def restore_original(**kwargs):
        from blueprint_pipeline.website_preparation_revision import workspace
        assert workspace(root) == root / 'pipeline/website_scene_preparation'
        value = {'schema_version': 'website_scene_handoff.v1', 'scene_id': selectors['scene_id'],
            'capture_id': selectors['capture_id'], 'status': 'intake_ready',
            'website_intake_outbox': {'state': 'forward_pending', 'submission_id': selectors['capture_id']}}
        value['digest'] = canonical_digest(value, digest_field='digest')
        write_json(root / 'pipeline/website_scene_preparation/handoff.json', value)
        return original(**kwargs)
    import time
    restored = choice.reconcile_target_choices(listener, storage_root=root.parents[4],
        process_args={'provider': 'openai', 'run_e2e': restore_original}, now=time.time() + 120)
    assert restored['confirmed'] == 1 and restored['replayed'] == 1
    assert choice.correction_binding_digest(root / 'pipeline') is None
    assert choice.correction_pending(listener._read_job_ledger(root), root / 'pipeline') is None
    assert len(runs) == 3


def test_a_correction_reruns_the_same_job_from_retained_work_once(guessed, monkeypatch):
    root, _, question, selectors, runs, guessing_run = guessed
    picked = _other(question)
    _webapp(monkeypatch, {question["question_digest"]: _answer(question, selectors,
        {"kind": "other_candidate", "candidate_id": picked})})
    process_args = {"provider": "openai", "run_evaluation_prep": False, "run_e2e": guessing_run(guesses=(), error=None)}
    counts = choice.reconcile_target_choices(listener, storage_root=root.parents[4], process_args=process_args)
    assert counts == {"replayed": 1, "confirmed": 0, "corrected": 1, "declined": 0, "waiting": 0}
    assert len(runs) == 2 and runs[1]["resume_completed_stages"] is True and runs[1]["capture_root"] == str(root)
    assert runs[1]["choices"]["target-1"]["candidate_id"] == picked
    ledger = listener._read_job_ledger(root)
    digest = choice.correction_binding_digest(root / "pipeline")
    assert ledger["status"] == "completed" and ledger["attempt_count"] == 2
    assert ledger["attempt_history"][-1][choice.CORRECTION_MARK] == digest
    assert choice.correction_pending(ledger, root / "pipeline") is None
    assert choice.reconcile_target_choices(listener, storage_root=root.parents[4], process_args=process_args)["replayed"] == 0
    assert len(runs) == 2


def test_a_correction_that_reaches_the_atlas_wait_is_armed_and_resumed(guessed, monkeypatch):
    root, _, question, selectors, runs, guessing_run = guessed
    _webapp(monkeypatch, {question["question_digest"]: _answer(question, selectors,
        {"kind": "other_candidate", "candidate_id": _other(question)})})
    _job(root, "processing", stage="pose")

    def waits(**kwargs):
        runs.append(kwargs)
        choice.retain_questions(root / "pipeline", guesses=[], source_geometry=GEOMETRY,
                                scene_id=root.parent.parent.name, capture_id=root.name)
        raise PipelineError(reconstruction.WAIT_ERROR)
    assert choice.reconcile_target_choices(listener, storage_root=root.parents[4],
        process_args={"provider": "openai", "run_e2e": waits})["replayed"] == 1
    ledger = listener._read_job_ledger(root)
    # The correction run's earlier, unrelated failure does not stop the wait from arming.
    assert reconstruction.armed(ledger) and ledger["attempt_history"][-1][choice.CORRECTION_MARK]
    assert choice.correction_pending(ledger, root / "pipeline") is None
    _job(root, "ready")
    done = []
    assert reconstruction.reconcile_waiting_reconstructions(listener, storage_root=root.parents[4], process_args={
        "provider": "openai", "run_evaluation_prep": False,
        "run_e2e": lambda **kw: done.append(kw) or {"status": "completed"}})["resumed"] == 1
    assert len(done) == 1 and listener._read_job_ledger(root)["status"] == "completed"


def test_a_correction_run_that_stops_short_stays_owed_and_bounded(guessed, monkeypatch):
    root, _, question, selectors, runs, guessing_run = guessed
    _webapp(monkeypatch, {question["question_digest"]: _answer(question, selectors,
        {"kind": "other_candidate", "candidate_id": _other(question)})})
    calls = []

    def fails(**kwargs):
        calls.append(kwargs)
        raise PipelineError("website_scene_preparation: transient_before_task_masks")
    process_args = {"provider": "openai", "run_e2e": fails}
    for now in (1000.0, 1001.0, 1100.0):
        choice.reconcile_target_choices(listener, storage_root=root.parents[4], process_args=process_args, now=now)
    # Owed after the swallowed failure, retried after its backoff, never in a tight loop.
    assert len(calls) == 2 and choice.correction_pending(listener._read_job_ledger(root), root / "pipeline")
    for step in range(10):
        choice.reconcile_target_choices(listener, storage_root=root.parents[4], process_args=process_args,
                                        now=10_000.0 + step * 2000)
    assert len(calls) == choice.MAX_CORRECTION_REPLAYS


def test_two_ambiguous_targets_keep_their_own_choices(guessed, monkeypatch):
    root, _, first, selectors, runs, guessing_run = guessed
    second_target = {**TARGET, "target_id": "target-2", "semantic_label": "lid"}
    second_tracks = [_track("lid-a", 1, label="target-2"), _track("lid-b", 6, label="target-2")]
    both = (_guess(), _guess(second_target, second_tracks))
    choice.retain_questions(root / "pipeline", guesses=list(both), source_geometry=GEOMETRY,
                            scene_id=root.parent.parent.name, capture_id=root.name)
    second = choice.read_question(root / "pipeline", "target-2")
    # One question at a time, in target order.
    assert choice.public_question(root)["target_id"] == "target-1"
    answers = {first["question_digest"]: _answer(first, selectors, {"kind": "other_candidate", "candidate_id": _other(first)})}
    _webapp(monkeypatch, answers)
    idle = {"provider": "openai", "run_e2e": lambda **_: pytest.fail("not yet")}
    replay_due = choice._replay_due
    monkeypatch.setattr(choice, "_replay_due", lambda *_: False)
    choice.reconcile_target_choices(listener, storage_root=root.parents[4], process_args=idle, now=1000.0)
    assert choice.public_question(root)["target_id"] == "target-2"
    answers[second["question_digest"]] = _answer(second, selectors, {"kind": "other_candidate", "candidate_id": _other(second)})
    choice.reconcile_target_choices(listener, storage_root=root.parents[4], process_args=idle, now=10_000.0)
    monkeypatch.setattr(choice, "_replay_due", replay_due)
    assert choice.public_question(root) is None
    process_args = {"provider": "openai", "run_evaluation_prep": False, "run_e2e": guessing_run(guesses=(), error=None)}
    assert choice.reconcile_target_choices(listener, storage_root=root.parents[4], process_args=process_args)["replayed"] == 1
    # Neither answer overwrote the other: the rerun applies both.
    assert set(runs[-1]["choices"]) == {"target-1", "target-2"}
    assert [row["target_id"] for row in choice.corrections(root / "pipeline")] == ["target-1", "target-2"]


def test_a_completed_job_reopens_only_for_its_correction(guessed, monkeypatch):
    root, raw, question, selectors, runs, guessing_run = guessed
    listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
                                     run_e2e=guessing_run(error=None))
    ledger = listener._read_job_ledger(root)
    assert ledger["status"] == "completed" and choice.REPLAY_FIELD in ledger
    claim = dict(scene_id=ledger["scene_id"], capture_id=ledger["capture_id"], owner="w", lease_seconds=60,
                 payload_sha256=ledger["source_payload_sha256"], producer_delivery_key=ledger["producer_delivery_key"])
    assert listener._claim_job_lease(root, **claim)[0] == "completed"
    stale = {"revision": ledger["revision"] - 1, "corrections_digest": "sha256:" + "1" * 64}
    assert listener._claim_job_lease(root, **claim, expected_target_choice_correction=stale)[0] == (
        "target_choice_correction_changed")
    state, claimed = listener._claim_job_lease(root, **claim, expected_target_choice_correction={
        "revision": ledger["revision"], "corrections_digest": "sha256:" + "1" * 64})
    assert state == "claimed" and claimed["status"] == "processing"
    commit = json.loads((root / listener.JOB_OUTPUT_COMMIT_FILENAME).read_text())
    assert commit["status"] == "superseded_target_choice_correction"


def test_only_explicit_waits_since_the_correction_arm_a_replay():
    unknown = {"status": "failed_retryable", "error": "unknown_provider_charge"}
    assert reconstruction.safe_to_arm(2, [unknown], False) is False
    assert reconstruction.safe_to_arm(2, [unknown], False, correction=True) is True
    assert reconstruction.safe_to_arm(2, [unknown], True, correction=True) is False
    wait = {"error": "w", reconstruction.FIELD: {"schema_version": reconstruction.SCHEMA, "error": "w"},
            choice.CORRECTION_MARK: "sha256:" + "1" * 64}
    assert reconstruction.safe_to_arm(3, [unknown, wait], False) is True
    assert reconstruction.safe_to_arm(3, [unknown, {**wait, reconstruction.FIELD: None}], False) is False


def test_clean_plate_continues_on_the_guess_and_hands_answers_to_the_masks(tmp_path, monkeypatch):
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
            kwargs["target_guesses"].append(_guess())
        return {"targets": []}
    monkeypatch.setattr(_ANALYSIS_ATTR, lambda **_: plan)
    monkeypatch.setattr("blueprint_pipeline.clean_plate_stage.prepare_website_source_frames", lambda **_: geometry)
    monkeypatch.setattr("blueprint_pipeline.clean_plate_stage.run_website_task_masks", masks)
    result = run_clean_plate_stage(capture_root=capture_root, website_source_video=source,
                                   privacy_processing={"status": "pending_website_review"})
    assert result["blockers"] == [] and result["mode"] == "task_masks_ready"
    question = choice.current_questions(capture_root / "pipeline")[0]
    assert question["capture_id"] == "c1" and question["scene_id"] == "s1"
    selectors = {"request_id": "req-1", "scene_id": "s1", "capture_id": "c1",
                 "completion_marker_generation": "1", "producer_delivery_key": "sha256:" + "3" * 64,
                 "source_payload_sha256": "4" * 64, "task_context_digest": "sha256:" + "5" * 64}
    picked = _other(question)
    write_json(capture_root / "pipeline" / choice.QUESTION_DIR / "target-1.answer.json",
               _answer(question, selectors, {"kind": "other_candidate", "candidate_id": picked}))
    result = run_clean_plate_stage(capture_root=capture_root, website_source_video=source,
                                   privacy_processing={"status": "pending_website_review"})
    assert result["blockers"] == [] and seen[-1]["target-1"]["candidate_id"] == picked
    assert choice.current_questions(capture_root / "pipeline") == []
