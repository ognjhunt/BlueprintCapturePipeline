"""A passed replay can request only its existing owner's controller intent."""
import hashlib
import json
import time

import pytest

from blueprint_pipeline.agent_execution.controller_recovery import ControllerRecoveryBinding, ControllerRecoveryTools
from blueprint_pipeline.agent_execution.contracts import AgentTool, ToolContext, digest
from blueprint_pipeline.agent_execution.operations import OperationPending, ToolRefused
from blueprint_pipeline.agent_execution.prepare import prepare_retained_failure
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from tests.test_agent_production_service import fixture, write
from tests.test_task_evaluation_stage_replay import _queue, CHILD


def prepare(tmp_path):
    service, _, _, _ = fixture(tmp_path)
    queue, _, job = _queue(tmp_path / "saved")
    root = tmp_path / "intents"
    intent = {"schema_version": "task_evaluation_scene_intent.v1", "intent_id": "scene-fixture",
        "request": {"consent": {"spend_authorized": True, "task_confirmed": True},
            "execution": {"max_total_spend_usd": 30.0, "max_retries": 1, "expires_at_epoch": time.time() + 300}}}
    intent["intent_digest"] = cross_runtime_canonical_digest(intent)
    write(root / "scene-fixture/intent.json", intent)
    config = {"schema_version": "task_evaluation_scene_progression_config.v1", "intent_root": str(root)}
    config["config_digest"] = canonical_digest(config)
    config_path = tmp_path / "controller.json"
    write(config_path, config)
    link = {"schema_version": "task_evaluation_scene_preparation_link.v1", "intent_id": intent["intent_id"],
        "intent_digest": intent["intent_digest"], "request_digest": job["parent_request_digest"]}
    link["link_digest"] = canonical_digest(link)
    link_path = tmp_path / "link.json"
    write(link_path, link)
    binding = ControllerRecoveryBinding(recovery_id="resume_scene", intent_id=intent["intent_id"],
        intent_digest=intent["intent_digest"], controller_config_path=str(config_path),
        controller_config_sha256="sha256:" + hashlib.sha256(config_path.read_bytes()).hexdigest(),
        required_replay_id="failed_boundary", parent_request_digest=job["parent_request_digest"],
        preparation_link_path=str(link_path), preparation_link_sha256="sha256:" + hashlib.sha256(link_path.read_bytes()).hexdigest())
    record = prepare_retained_failure(service, task_id="recovery-task", run_id="recovery-run", child_id=CHILD,
        owner_client_id="fixture-client", inference_budget_usd=1, queue_root=queue,
        parent_queue_root=tmp_path / "parents", input_root=tmp_path, approved_roots=(tmp_path,), controller_recovery=binding)
    service.journal.register(record.task)
    return service, record, binding


def test_requires_successful_same_task_replay_and_deduplicates_controller_handoff(tmp_path):
    service, record, binding = prepare(tmp_path)
    bridge = ControllerRecoveryTools(service.journal, (binding,), record.task.source_commit)
    tool = bridge.tools()[0]
    context = ToolContext(record.task.run_id, record.task.task_id, record.task.context_revision,
        digest({"operation": 1}), record.task.admission.authority_digest, record.task.deadline)
    report_digest = digest({"replay": "completed"})
    args = {"recovery_id": "resume_scene", "replay_report_digest": report_digest}
    with pytest.raises(ToolRefused, match="successful_replay_required"):
        tool.invoke(args, context)
    replay = AgentTool("replay_retained_stage", "1", "fixture", {"type": "object"}, "read_only", lambda *_: {})
    op = service.journal.prepare_call(record.task, replay, {"replay_id": "failed_boundary"}, turn_id="turn", call_id="replay")
    service.journal.complete_operation(op["operation_id"], {"success": True,
        "output": {"status": "completed", "report_digest": report_digest}})
    with pytest.raises(OperationPending):
        tool.invoke(args, context)
    request_path, _ = bridge.paths(context.operation_id)
    original = request_path.read_bytes()
    with pytest.raises(OperationPending):
        tool.invoke(args, context)
    assert request_path.read_bytes() == original
    assert json.loads(original)["binding"]["intent_digest"] == binding.intent_digest
    service.validate_admission(record.task)


def test_cancellation_committed_before_controller_handoff_wins(tmp_path):
    service, record, _ = prepare(tmp_path)
    service.journal.request_cancel(record.task.task_id, "cancel")
    from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
    with pytest.raises(AgentExecutionError, match="cancelled_before_handoff"):
        service.journal.claim_controller_request(record.task, digest({"operation": 1}), {"request": "admitted"})
