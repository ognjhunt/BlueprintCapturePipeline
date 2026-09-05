from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_sam31_preparation_queue as precursor
from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_contract import launch_preparation_request_digest
from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
    launch_preparation_status, stage_launch_preparation_request,
)
from tests.test_task_evaluation_launch_preparation_worker import (
    SERVICE_ACCOUNT, _rebind_recipe, fetcher, production_request_with_fetchable_bytes,
)


def _record(path):
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


@pytest.fixture
def job(tmp_path, request):
    reviewer_kind = getattr(request, "param", "ai")
    request, payloads = production_request_with_fetchable_bytes()
    recipe = json.loads(payloads[request["construction"]["recipe"]["uri"]])
    ref = recipe["stage_sequence"][0]["configuration"]
    config = json.loads(payloads[ref["uri"]])
    config["required_views"] = {"mask_source": "sam31_reviewed_calibrated_object_masks"}
    config["sam31_preparation_plan"] = {"reviewer_kind": reviewer_kind}
    config["sam31_review_kind"] = reviewer_kind
    data = json.dumps(config).encode()
    payloads[ref["uri"]] = data
    ref.update(digest="sha256:" + hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    _rebind_recipe(request, payloads, recipe)
    queue = tmp_path / "queue"
    stage_launch_preparation_request(value=request, queue_root=queue, submitted_by="blueprint-webapp")
    def forbidden_renderer(**kwargs):
        pytest.fail("Construction renderer cannot run before validated SAM evidence")
    kwargs = {
        "queue_root": queue, "input_root": tmp_path / "inputs",
        "allowed_uri_prefixes": ["s3://blueprint-production-inputs/"],
        "service_account": SERVICE_ACCOUNT, "source_commit": request["expected_production_commit"],
        "fetcher": fetcher(payloads), "scene_render_input_materializer": forbidden_renderer,
        "construction_queue_root": tmp_path / "construction",
    }
    return tmp_path, request, queue, kwargs


def _wait(job):
    _, request, queue, kwargs = job
    result = worker.process_launch_preparation_queue(
        **kwargs, sam31_preparation_advancer=lambda context: {
            "status": "waiting_for_child", "evidence_refs": [],
            "child_kind": "independent_sdk_review",
        },
    )
    status = launch_preparation_status(preparation_id=request["preparation_id"], queue_root=queue)
    return result["results"][0], status


def _resume(job, progress, *, content=b'{"status":"child_completed"}'):
    root, request, queue, _ = job
    source = root / "inputs" / "child-result.json"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(content)
    return precursor.stage_resume_signal(
        queue_root=queue, preparation_id=request["preparation_id"],
        request_digest=launch_preparation_request_digest(request),
        progress_digest=progress["progress_digest"], source_commit=request["expected_production_commit"],
        kind="child_result", evidence_ref=_record(source), approved_roots=(root,),
    )


def test_ai_child_wait_retains_parent_identity_without_final_result_or_human_pause(job):
    progress, status = _wait(job)
    _, request, queue, kwargs = job
    assert progress["status"] == "waiting_for_child"
    assert status["status"] == precursor.WAITING_STATE
    assert status["worker_status"] == "waiting_for_child"
    assert status["human_review_required"] is False
    assert status["request_digest"] == launch_preparation_request_digest(request)
    assert not list((queue / "results").glob("*.json"))
    assert not (job[0] / "construction").exists()
    repeated = worker.process_launch_preparation_queue(
        **kwargs, sam31_preparation_advancer=lambda context: pytest.fail("No wake-up signal"),
    )
    assert repeated["processed_count"] == 0
    intake = stage_launch_preparation_request(value=request, queue_root=queue, submitted_by="blueprint-webapp")
    assert intake["already_exists"] is True


def test_digest_bound_child_signal_resumes_same_job_without_fabricating_ready(job):
    progress, _ = _wait(job)
    signal = _resume(job, progress)
    seen = []
    def advance(context):
        seen.append(context)
        return {"status": "waiting_for_child", "evidence_refs": [],
                "child_kind": "waiting_for_exact_mask_validation"}
    result = worker.process_launch_preparation_queue(**job[3], sam31_preparation_advancer=advance)
    assert result["resume_results"] == [{"status": "resumed", "signal_digest": signal["signal_digest"]}]
    assert seen[0]["validated_resume_receipt"] == signal
    assert seen[0]["request"] == job[1]
    assert seen[0]["expected_source_commit"] == job[1]["expected_production_commit"]
    assert result["results"][0]["status"] == "waiting_for_child"
    assert result["results"][0]["sequence"] == 2
    assert not list((job[2] / "results").glob("*.json"))


def test_caller_ready_boolean_without_receipts_is_blocked_before_render_or_construction(job):
    _wait(job)
    progress = precursor.load_progress(
        job[2], next((job[2] / precursor.WAITING_STATE).glob("*.json")).name,
        launch_preparation_request_digest(job[1]),
    )
    _resume(job, progress)
    result = worker.process_launch_preparation_queue(
        **job[3], sam31_preparation_advancer=lambda context: {"status": "ready", "evidence_refs": []},
    )
    assert result["results"][0]["status"] == "blocked"
    assert result["results"][0]["blockers"] == ["sam31_preparation_driver_evidence_missing"]
    assert not (job[0] / "construction").exists()


def test_unknown_unvalidated_signal_is_quarantined_and_does_not_wake_parent(job):
    _wait(job)
    queue = job[2]
    unknown = queue / "source-resume-pending" / "unknown.json"
    unknown.write_text('{"status":"ready","accepted":true}')
    result = worker.process_launch_preparation_queue(
        **job[3], sam31_preparation_advancer=lambda context: pytest.fail("Invalid signal must not advance"),
    )
    assert result["processed_count"] == 0
    assert result["resume_results"][0]["status"] == "rejected"
    assert (queue / "source-resume-blocked" / "unknown.json").is_file()
    assert list((queue / precursor.WAITING_STATE).glob("*.json"))
    assert not list((queue / "results").glob("*.json"))


def test_changed_child_receipt_is_rejected_on_resume_readback(job):
    progress, _ = _wait(job)
    signal = _resume(job, progress)
    Path(signal["evidence_ref"]["path"]).write_bytes(b"changed")
    result = worker.process_launch_preparation_queue(
        **job[3], sam31_preparation_advancer=lambda context: pytest.fail("Changed child evidence"),
    )
    assert result["processed_count"] == 0
    assert result["resume_results"][0]["blocker"] == "sam31_preparation_evidence_readback_mismatch"
    assert list((job[2] / precursor.WAITING_STATE).glob("*.json"))


def test_signal_cannot_change_parent_request_or_progress(job):
    progress, _ = _wait(job)
    root, request, queue, _ = job
    artifact = root / "inputs" / "child.json"
    artifact.write_bytes(b"child")
    with pytest.raises(precursor.Sam31PreparationQueueError, match="resume_progress_mismatch"):
        precursor.stage_resume_signal(
            queue_root=queue, preparation_id=request["preparation_id"],
            request_digest=launch_preparation_request_digest(request),
            progress_digest="sha256:" + "0" * 64, source_commit=request["expected_production_commit"],
            kind="child_result", evidence_ref=_record(artifact), approved_roots=(root,),
        )
    assert not list((queue / "source-resume-pending").glob("*.json"))


@pytest.mark.parametrize("job", ["human"], indirect=True)
def test_legacy_optional_human_wait_does_not_accept_child_signal(job):
    root, request, queue, kwargs = job
    candidate = root / "inputs" / "candidate.json"
    candidate.parent.mkdir()
    value = {"schema_version": "public_scene_sam31_track_selection_review_candidate.v1"}
    value["candidate_digest"] = canonical_digest(value, digest_field="candidate_digest")
    candidate.write_text(json.dumps(value))
    result = worker.process_launch_preparation_queue(
        **kwargs, sam31_preparation_advancer=lambda context: {
            "status": "awaiting_human_review", "reviewer_kind": "human", "evidence_refs": [_record(candidate)],
            "review_candidate_ref": _record(candidate),
        },
    )
    progress = result["results"][0]
    with pytest.raises(precursor.Sam31PreparationQueueError, match="resume_kind_invalid"):
        _resume(job, progress)
    status = launch_preparation_status(preparation_id=request["preparation_id"], queue_root=queue)
    assert status["human_review_required"] is True


def test_progress_chain_cannot_be_rewritten(job):
    _wait(job)
    path = next((job[2] / "source-progress").rglob("*.json"))
    progress = json.loads(path.read_text())
    progress["status"] = "ready"
    path.chmod(0o600)  # Deliberate tamper of this test-owned immutable fixture.
    path.write_text(json.dumps(progress))
    with pytest.raises(precursor.Sam31PreparationQueueError, match="progress_chain_invalid"):
        launch_preparation_status(preparation_id=job[1]["preparation_id"], queue_root=job[2])

def test_ai_plan_cannot_silently_introduce_human_pause(job):
    root = job[0] / "inputs"
    root.mkdir()
    candidate = root / "candidate.json"
    candidate.write_text("{}")
    result = worker.process_launch_preparation_queue(
        **job[3], sam31_preparation_advancer=lambda context: {
            "status": "awaiting_human_review", "reviewer_kind": "human",
            "evidence_refs": [_record(candidate)], "review_candidate_ref": _record(candidate),
        },
    )
    assert result["results"][0]["status"] == "blocked"
    assert result["results"][0]["blockers"] == ["sam31_preparation_human_pause_not_explicit"]


def test_repeated_child_progress_still_gets_distinct_resume_bound_checkpoints(job):
    progress, _ = _wait(job)
    same_advancement = progress["advancement"]
    for sequence in (2, 3):
        _resume(job, progress)
        result = worker.process_launch_preparation_queue(
            **job[3], sam31_preparation_advancer=lambda context: same_advancement,
        )
        progress = result["results"][0]
        assert progress["sequence"] == sequence
        assert progress["status"] == "waiting_for_child"
    assert not list((job[2] / "results").glob("*.json"))
