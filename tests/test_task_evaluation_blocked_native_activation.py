import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_completed_placement_adoption import native_submission_absent


def seal(path, document, field):
    document[field] = canonical_digest(document, digest_field=field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "pending",
        "processing",
        "completed",
        "paid",
        "provider",
        "result_digest",
        "owner",
        "current_release",
        "launch",
    ],
)
def test_only_terminal_blocked_prelaunch_can_retire_native_hold(tmp_path, monkeypatch, defect):
    old, new = "a" * 40, "b" * 40
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_release_identity.running_release_commit",
        lambda: old if defect == "current_release" else new,
    )
    queue = tmp_path / "activation-queue"
    state = tmp_path / "progression"
    auth = tmp_path / "authorization.json"
    authorization = {"reference": "same-sealed-owner"}
    auth.write_text(json.dumps(authorization))
    plan = {
        "expected_production_commit": old,
        "source_launch_id": "scene-launch",
        "future_outputs": {"destination": {"expected_activation_id": "activation"}},
        "phases": {"destination": {"authorization_path": str(auth)}},
    }
    request = {
        "activation_id": "activation",
        "expected_production_commit": old,
        "authorization": authorization,
    }
    if defect == "owner":
        request["authorization"] = {"reference": "different-owner"}
    marker = {
        "schema_version": "task_evaluation_configured_controls_progression.v1",
        "status": "destination_qualification_activation_queued",
        "expected_production_commit": old,
        "activation_request": request,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "activation_executed_provider": False,
    }
    root = state / "scene-launch" / ("franka-controls-" + old[:12])
    seal(root / "destination_activation_progression.json", marker, "progression_digest")
    filename = "activation-" + canonical_digest(request).removeprefix("sha256:") + ".json"
    envelope = {
        "schema_version": "task_evaluation_launch_activation_envelope.v1",
        "request": request,
        "request_digest": canonical_digest(request),
        "provider_mutation_performed_inside_intake": False,
        "paid_execution_requested": False,
    }
    seal(queue / "blocked" / filename, envelope, "envelope_digest")
    result = {
        "schema_version": "task_evaluation_launch_activation_result.v1",
        "activation_id": "activation",
        "status": "blocked",
        "blockers": ["packaging_failed"],
        "provider_mutation_performed": defect == "provider",
        "paid_execution_requested": defect == "paid",
    }
    seal(queue / "results" / filename, result, "result_digest")
    if defect == "result_digest":
        result["result_digest"] = "sha256:" + "c" * 64
        (queue / "results" / filename).write_text(json.dumps(result))
    if defect in ("pending", "processing", "completed"):
        seal(queue / defect / filename, envelope, "envelope_digest")
    launches = tmp_path / "launches"
    if defect == "launch":
        (launches / "activation-auto-launch").mkdir(parents=True)
    config = {
        "scene_root": str(tmp_path / "owners"),
        "progression_root": str(state),
        "launch_state_root": str(launches),
        "activation_queue_root": str(queue),
    }
    assert native_submission_absent(config=config, plan=plan) is (defect is None)
