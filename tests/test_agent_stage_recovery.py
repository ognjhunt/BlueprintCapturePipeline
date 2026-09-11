"""No-model/no-GPU replay queue and stable same-call deferred SDK results."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.agent_execution import stage_recovery as replay
from blueprint_pipeline.agent_execution.contracts import AgentTool, ToolContext, ToolReconciliation, digest
from blueprint_pipeline.agent_execution.journal import AgentJournal
from blueprint_pipeline.agent_execution.operations import OperationPending
from tests.test_agent_execution_sdk import make_task, response, setup_runtime
from tests.test_agent_production_service import fixture as production_fixture
from tests.test_task_evaluation_stage_replay import _queue, CHILD
from blueprint_pipeline.agent_execution.prepare import prepare_retained_failure


def binding(tmp_path):
    return replay.StageReplayBinding(replay_id="retained_failure", child_id="sam31-fixture",
                                     job_sha256=digest({"job": 1}), queue_root=str(tmp_path / "children"),
                                     parent_queue_root=str(tmp_path / "parents"), input_root=str(tmp_path / "inputs"),
                                     approved_roots=(str(tmp_path),))


def test_replay_queue_reconciles_without_redispatch_and_omits_raw_report(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "_job", lambda _: (None, {"job": 1}))
    tools = replay.StageReplayTools(journal=AgentJournal(tmp_path / "journal"), bindings=(binding(tmp_path),), source_commit="a" * 40)
    tool = tools.tools()[0]
    context = ToolContext("run", "task", digest({"context": 1}), digest({"operation": 1}), digest({"authority": 1}), time.time() + 60)
    args = {"replay_id": "retained_failure"}
    with pytest.raises(OperationPending):
        tool.invoke(args, context)
    request_path, result_path = tools._paths(context)
    original = request_path.read_bytes()
    with pytest.raises(OperationPending):
        tool.invoke(args, context)
    assert request_path.read_bytes() == original
    assert tool.reconcile(args, context).status == "pending"
    report = {"status": "job_refused", "blocker": "job_source_commit_mismatch",
              "phase": "sam31_review", "fired_predicates": ["detected != 16", "url == 'https://secret.example/key'"],
              "job_path": "/private/source/job", "saved_result": {"signed_uri": "https://secret.example/key"}}
    summary = replay.summarize_report(report, replay_id="retained_failure", job_sha256=binding(tmp_path).job_sha256, source_commit="a" * 40)
    replay.write_json(result_path, {"request_digest": digest(json.loads(original)), "source_commit": "a" * 40, "summary": summary})
    assert tool.reconcile(args, context).output == summary
    assert "secret.example" not in json.dumps(summary)
    assert "/private" not in json.dumps(summary)
    assert summary["fired_predicates"] == ["detected != 16"]
    assert summary["redacted_predicate_count"] == 1


def test_replay_command_cannot_request_paid_phase_or_inherit_model_environment(tmp_path):
    command = replay.replay_command({"binding": binding(tmp_path).model_dump()}, output_root=tmp_path)
    assert command[1:3] == ["-m", "blueprint_pipeline.task_evaluation_stage_replay"]
    assert "--allow-paid" not in command
    assert "--environment-file" not in command
    assert "--isolate" not in command  # Isolation belongs to the qualified systemd worker.


def test_sdk_waits_for_same_pending_operation_without_second_dispatch(tmp_path, monkeypatch):
    ready = threading.Event()
    invocations = []
    reconciliations = []

    def invoke(args, context):
        invocations.append(context.operation_id)
        raise OperationPending("offline_fixture_pending")

    def reconcile(args, context):
        reconciliations.append(context.operation_id)
        if len(reconciliations) == 1:
            return ToolReconciliation("pending")
        ready.set()
        return ToolReconciliation("completed", {"replayed": True})

    tool = AgentTool("fixture_replay", "1", "Replay fixture", {"type": "object", "properties": {}, "additionalProperties": False},
                     "idempotent_write", invoke, reconcile)
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return response([{"type": "function_call", "id": "fc1", "call_id": "call1", "name": tool.tool_id,
                              "arguments": "{}", "status": "completed"}])
        assert ready.is_set()
        return response()
    runtime = setup_runtime(tmp_path / "sdk", monkeypatch, handler, tools=(tool,))
    task = make_task((tool,), max_model_turns=2)
    runtime.start(task)
    assert runtime.step(task.task_id)["state"] == "completed"
    assert len(calls) == 2
    assert len(invocations) == 1
    assert set(reconciliations) == set(invocations)


def prepare_fixture(tmp_path, owner_client_id="fixture-client"):
    service, _, _, _ = production_fixture(tmp_path)
    root, job_path, _ = _queue(tmp_path / "retained")
    record = prepare_retained_failure(
        service, task_id="recovery_task", run_id="recovery_run", child_id=CHILD,
        owner_client_id=owner_client_id, inference_budget_usd=1,
        queue_root=root, parent_queue_root=tmp_path / "retained" / "parents",
        input_root=tmp_path / "retained" / "inputs", approved_roots=(tmp_path,),
    )
    service.enqueue(record.task.task_id, owner_client_id)
    runtime = service.runtime_for_task(record.task)
    with pytest.raises(OperationPending):
        runtime.operations.execute(record.task, turn_id="fixture_turn", call_id="fixture_call",
                                   name="replay_retained_stage", arguments={"replay_id": "failed_boundary"})
    return service, record, runtime, job_path


@pytest.mark.parametrize("outcome", ["completed", "interrupted", "cancelled", "revoked"])
def test_offline_worker_consumes_bound_request_once(tmp_path, monkeypatch, outcome):
    service, record, runtime, job_path = prepare_fixture(tmp_path)
    monkeypatch.setattr(replay, "require_offline_isolation", lambda _: None)  # OS boundary only.
    source_bytes = job_path.read_bytes()
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        assert argv[1:3] == ["-m", "blueprint_pipeline.task_evaluation_stage_replay"]
        assert set(kwargs["env"]) == {"PATH", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE"}
        assert "--allow-paid" not in argv
        destination = Path(argv[argv.index("--json-out") + 1])
        assert (destination.parent / "intent.json").is_file()
        replay.write_json(destination, {"status": "job_refused", "phase": "sam31_review", "blocker": "fixture_input_refused"})
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(replay.subprocess, "run", execute)
    request_path = next((service.journal.root / "stage-replays" / "requests").glob("*.json"))
    if outcome == "interrupted":
        intent = service.journal.root / "stage-replays" / "runs" / request_path.stem / "intent.json"
        replay.write_json(intent, {"request_digest": digest(json.loads(request_path.read_text()))})
    if outcome == "cancelled":
        service.service.cancel(record.task.task_id)
    if outcome == "revoked":
        path = Path(service.config.task_store_root) / (record.task.task_id + ".json")
        value = json.loads(path.read_text())
        value["enabled"] = False
        replay.write_json(path, value)
    result = replay.process_one(service)
    assert result["summary"]["status"] == {"completed": "job_refused", "interrupted": "interrupted",
                                            "cancelled": "cancelled", "revoked": "refused"}[outcome]
    assert len(calls) == (1 if outcome == "completed" else 0)
    assert replay.process_one(service) is None
    assert job_path.read_bytes() == source_bytes
    if outcome == "completed":
        saved = runtime.operations.execute(record.task, turn_id="fixture_turn", call_id="fixture_call",
                                           name="replay_retained_stage", arguments={"replay_id": "failed_boundary"})
        assert saved["output"] == result["summary"]
    else:
        runtime.operations.reconcile_cancelled(record.task)
    assert service.journal.unsettled_operations(record.task.task_id) == []


def test_offline_worker_refuses_configured_secret_outside_canonical_directory(tmp_path):
    key = tmp_path / "outside-provider-secrets" / "key"
    key.parent.mkdir()
    key.write_text("sk-only-a-fixture")
    key.chmod(0o600)
    with pytest.raises(replay.AgentExecutionError, match="configured_secret_accessible"):
        replay.require_secret_isolation(SimpleNamespace(credential_file=str(key), webhook_secret_file=None))
    with pytest.raises(replay.AgentExecutionError, match="configured_secret_accessible"):
        replay.require_secret_isolation(SimpleNamespace(credential_file="/missing", webhook_secret_file=None,
                                                        webapp_sync_token_file=str(key)))
