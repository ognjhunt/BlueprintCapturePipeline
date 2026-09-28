"""Recovery failures must reach the allocator writer with teardown truth."""
import json

import pytest

from blueprint_pipeline import native_task_arena_feedback_allocator_adapter as adapter
from blueprint_pipeline import task_evaluation_robot_placement_warm_executor as warm
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_allocator import _write_native_task_arena_adapter_output
from blueprint_pipeline.task_evaluation_collision_aware_candidate_generation import (
    CollisionAwareCandidateGenerationError,
)


@pytest.mark.parametrize("closeout_state", ["absent", "unknown", "exception"])
def test_failed_search_retains_native_feedback_and_teardown(tmp_path, monkeypatch, closeout_state):
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "native_task_arena_packet_request.v1.json").write_text(json.dumps({
        "native_construction_feedback": {"maximum_rounds": 8},
    }))
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"instance_id": 51956468}))
    native = tmp_path / "native.json"
    native.write_text(json.dumps({"status": "blocked", "native_isaac_executed": True}))
    cold = {
        "status": "blocked", "blockers": ["native_grasp_failed"],
        "warm_session": {"instance_id": 51956468},
        "warm_session_receipt_path": str(session),
        "native_control_result_path": str(native),
        "native_control_result_digest": "sha256:" + "a" * 64,
        "continuing_spend_from_this_run": True,
    }
    before = native.read_bytes()
    closeouts = []

    def fail(**kwargs):
        raise CollisionAwareCandidateGenerationError("curobo_remote_process_failed")

    def close(*, warm_session):
        closeouts.append(warm_session["instance_id"])
        if closeout_state == "exception":
            raise RuntimeError("provider unavailable")
        return {"status": "completed" if closeout_state == "absent" else "blocked",
                "provider_instance_absent": closeout_state == "absent",
                "continuing_spend_from_this_run": closeout_state != "absent"}

    monkeypatch.setattr(warm, "_run_retained_native_construction_feedback", fail)
    monkeypatch.setattr(warm, "close_native_task_arena_warm_instance", close)
    result = adapter.continue_retained_feedback_if_requested(
        execute=True, construction_requested=True, retain_warm_session=True,
        result=cold, packet_dir=packet, runtime_source_packet_receipt_path=tmp_path / "runtime.json",
        prepared_bundle={"implementation_commit": "a" * 40}, native_authority={},
        job_dir=tmp_path / "job", max_hourly_rate_usd=0.8, hard_cap_usd=0.45,
        hard_ttl_seconds=2025,
    )
    output = tmp_path / "allocator-result.json"
    sealed = _write_native_task_arena_adapter_output(output, result)
    assert json.loads(output.read_text()) == sealed
    assert sealed["result_digest"] == canonical_digest(sealed, digest_field="result_digest")
    assert sealed["status"] == "blocked"
    assert sealed["blockers"] == [
        "native_construction_feedback_failed:CollisionAwareCandidateGenerationError",
        "native_grasp_failed",
    ]
    assert sealed["native_control_result_path"] == str(native)
    assert native.read_bytes() == before
    assert sealed["continuing_spend_from_this_run"] is (closeout_state != "absent")
    assert (sealed["warm_session"] is None) is (closeout_state == "absent")
    assert sealed["retry_cap"] == 0
    assert closeouts == [51956468]
    failure = json.loads((tmp_path / "job/native-construction-feedback/feedback-failure.v1.json").read_text())
    assert failure == sealed["native_construction_feedback_controller"]
    assert failure["failure_digest"] == canonical_digest(failure, digest_field="failure_digest")
