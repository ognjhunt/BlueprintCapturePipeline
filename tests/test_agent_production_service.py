"""Production composition, real intake authentication and server-owned scope."""

from dataclasses import asdict
import json
import time

from fastapi.testclient import TestClient
import httpx
import pytest

from blueprint_pipeline import live_pipeline_intake_service as intake
from blueprint_pipeline.agent_execution import http_routes, production
from blueprint_pipeline.agent_execution.contracts import AgentAdmission, AgentExecutionError, AgentTask, digest
from blueprint_pipeline.agent_execution.journal import AgentJournal
from blueprint_pipeline.agent_execution.production import OperationalDiagnosis, ProductionAgentService
from blueprint_pipeline.agent_execution.supervisor_bridge import SupervisorCapabilityBridge, context_revision
from blueprint_pipeline.task_evaluation_supervisor.capabilities import SupervisorContext
from blueprint_pipeline.task_evaluation_supervisor.contracts import AutonomyMode
from blueprint_pipeline.task_evaluation_supervisor.supervisor import default_authority_envelope
from blueprint_pipeline.task_evaluation_supervisor.tools import ToolRegistry
from tests.test_agent_execution_sdk import response
from tests.test_live_pipeline_intake_service import _signed_intake_headers


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    path.chmod(0o600)


def fixture(tmp_path):
    registry = ToolRegistry.default()
    status = {"status": "waiting", "first_blocker": "sam_review_pending", "next_required_stage": "sam_review"}
    status["status_digest"] = digest(status)
    authority = default_authority_envelope(
        run_id="prod_fixture", mode=AutonomyMode.EXECUTE_NON_SPEND, tool_registry=registry,
        immutable_input_digests=[status["status_digest"]], agent_inference_budget_usd=10,
        allow_agent_inference=True,
    ).to_mapping()
    context = SupervisorContext(run_id="prod_fixture", customer_question="Inspect preparation status.",
                                fresh_scene_preparation_status=status, authority_envelope=authority,
                                supervisor_output_dir=str(tmp_path / "state" / "supervisor"))
    journal = AgentJournal(tmp_path / "state")
    bridge = SupervisorCapabilityBridge(capability="capture_testbed_supervisor", registry=registry,
                                        journal=journal, load_context=lambda _: context)
    tool = next(tool for tool in bridge.tools() if tool.tool_id == "inspect_fresh_scene_preparation")
    inputs = [{"role": "user", "content": json.dumps({"question": context.customer_question,
                "context_revision": context_revision(context), "status_digest": status["status_digest"]})}]
    admission = AgentAdmission(
        authority_digest=authority["authority_digest"], authority_reference="server:prod_task",
        project_id="proj_fixture", runtime="openai_agents_sdk", disclosure_scope="sanitized_operations",
        allowed_input_digests=(digest(inputs), status["status_digest"]), allowed_tool_ids=(tool.tool_id,),
        budget_policy="strict_per_call", inference_budget_usd=10, expires_at=time.time() + 600,
    )
    task = AgentTask(task_id="prod_task", run_id=context.run_id, capability="capture_testbed_supervisor",
                     context_revision=context_revision(context), source_commit="a" * 40,
                     instructions="Inspect the requested status and report supported next actions.",
                     model="gpt-5.6-terra", input=inputs, input_digests=(status["status_digest"],),
                     output_schema=OperationalDiagnosis.model_json_schema(), tool_ids=(tool.tool_id,),
                     tool_digests={tool.tool_id: tool.tool_digest}, admission=admission,
                     deadline=time.time() + 300, max_model_turns=2, max_tool_output_bytes=10_000)
    record_path = tmp_path / "admitted" / "prod_task.json"
    write(record_path, {"schema_version": "blueprint_agent_admitted_task.v1", "enabled": True,
                        "autostart": True, "owner_client_ids": ["fixture-client"],
                        "task": task.model_dump(mode="json"), "context": asdict(context)})
    key_path = tmp_path / "key"
    key_path.write_text("sk-fixture")
    key_path.chmod(0o600)
    config_path = tmp_path / "config.json"
    write(config_path, {"schema_version": "blueprint_agent_production_config.v1",
                        "state_root": str(tmp_path / "state"), "task_store_root": str(tmp_path / "admitted"),
                        "source_commit": "a" * 40, "project_id": "proj_fixture", "credential_id": "key_fixture",
                        "credential_file": str(key_path), "allowed_models": ["gpt-5.6-terra"],
                        "max_task_budget_usd": 10})
    service = ProductionAgentService(config_path, source_commit="a" * 40)
    return service, task, record_path, config_path


def test_production_autostart_real_sdk_and_real_registry_inspection(tmp_path, monkeypatch):
    service, task, _, _ = fixture(tmp_path)
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "1")
    calls = []
    output = {"disposition": "awaiting_input", "summary": "SAM review remains pending.",
              "evidence_references": [], "next_actions": ["Inspect the admitted review status."], "uncertainty": []}

    def handler(request):
        calls.append(request)
        assert request.headers["OpenAI-Project"] == "proj_fixture"
        if len(calls) == 1:
            return response([{"type": "function_call", "name": "inspect_fresh_scene_preparation",
                              "call_id": "call_fixture", "id": "fc_fixture", "status": "completed",
                              "arguments": json.dumps({"context_revision": task.context_revision,
                                  "arguments": {"status_digest": task.input_digests[0]}})}])
        assert "sam_review_pending" in request.content.decode()
        return response([{"type": "message", "id": "msg_end", "role": "assistant", "status": "completed",
                          "content": [{"type": "output_text", "text": json.dumps(output), "annotations": []}]}], number=2)

    original = service.runtime_for_task
    def runtime(current):
        value = original(current)
        value._hermetic_transport = httpx.MockTransport(handler)
        return value
    service.service.runtime_for_task = runtime
    assert service.autostart() == 1
    assert service.autostart() == 0
    assert calls == []
    assert service.service.tick()["state"] == "completed"
    assert len(calls) == 2
    result = service.status(task.task_id, "fixture-client")
    assert result["result"]["output"] == output
    assert result["result"]["hermetic"]
    assert "credential_file" not in json.dumps(result)
    restarted = ProductionAgentService(service.config_path, source_commit="a" * 40)
    assert restarted.autostart() == 0
    assert restarted.service.recover() == 0
    assert restarted.status(task.task_id, "fixture-client")["state"] == "completed"


def test_changed_server_record_revokes_before_inference(tmp_path):
    service, task, record_path, _ = fixture(tmp_path)
    service.enqueue(task.task_id, "fixture-client")
    value = json.loads(record_path.read_text())
    value["enabled"] = False
    write(record_path, value)
    assert service.service.tick()["state"] == "cancelled"
    assert service.journal.task(task.task_id)["cancel_reason"] == "agent_admission_revoked"


@pytest.mark.parametrize("changed", ["release", "budget", "project", "context", "config", "client", "symlink"])
def test_invalid_admission_never_enqueues(tmp_path, changed):
    service, task, record_path, config_path = fixture(tmp_path)
    if changed == "release":
        with pytest.raises(AgentExecutionError, match="release_mismatch"):
            ProductionAgentService(config_path, source_commit="b" * 40)
        return
    if changed == "client":
        with pytest.raises(AgentExecutionError, match="client_not_authorized"):
            service.enqueue(task.task_id, "different-client")
    elif changed == "symlink":
        original = record_path.with_suffix(".retained")
        record_path.rename(original)
        record_path.symlink_to(original)
        with pytest.raises(AgentExecutionError, match="file_unavailable"):
            service.enqueue(task.task_id, "fixture-client")
    else:
        value = json.loads(record_path.read_text())
        if changed in {"budget", "project"}:
            value["task"]["admission"]["inference_budget_usd" if changed == "budget" else "project_id"] = (
                20 if changed == "budget" else "other_project")
        elif changed == "context":
            value["context"]["customer_question"] = "Unadmitted changed question"
        elif changed == "config":
            config = json.loads(config_path.read_text())
            config["max_task_budget_usd"] = 1
            write(config_path, config)
        write(record_path, value)
        with pytest.raises(AgentExecutionError):
            service.enqueue(task.task_id, "fixture-client")
    assert service.journal.tasks(active_only=False) == []


def test_routes_use_real_intake_hmac_and_task_owner(tmp_path, monkeypatch):
    service, task, _, _ = fixture(tmp_path)
    monkeypatch.setenv(intake.INTAKE_CLIENT_SECRETS_ENV, json.dumps({"fixture-client": "fixture-secret", "other": "other-secret"}))
    monkeypatch.setenv(intake.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.setenv(intake.INTAKE_WORK_DIR_ENV, str(tmp_path / "http"))
    original = http_routes.register_agent_execution_routes
    monkeypatch.setattr(http_routes, "register_agent_execution_routes",
                        lambda app, **kwargs: original(app, service_factory=lambda: service, **kwargs))
    client = TestClient(intake.create_app())
    url = f"/api/live-pipeline/agents/tasks/{task.task_id}"
    assert client.post(url + "/enqueue", content="{}").status_code == 401
    headers = _signed_intake_headers("fixture-secret", "{}", client_id="fixture-client")
    assert client.post(url + "/enqueue", content="{}", headers=headers).json()["state"] == "queued"
    assert client.post(url + "/enqueue", content="{}", headers=headers).status_code == 401
    other = _signed_intake_headers("other-secret", "", client_id="other", nonce="other-nonce")
    assert client.get(url, headers=other).status_code == 403
    extra = '{"prompt":"Expand authority"}'
    headers = _signed_intake_headers("fixture-secret", extra, client_id="fixture-client", nonce="extra-nonce")
    assert client.post(url + "/enqueue", content=extra, headers=headers).status_code == 400
    headers = _signed_intake_headers("fixture-secret", "{}", client_id="fixture-client", nonce="cancel-nonce")
    assert client.post(url + "/cancel", content="{}", headers=headers).json()["cancel_requested"]


def test_config_is_release_bound_and_has_no_managed_default(tmp_path):
    service, _, _, _ = fixture(tmp_path)
    assert service.health()["managed_api_enabled"] is False
    assert service.health()["active_tasks_up_to_1000"] == 0
    assert production.CONFIG_ENV == "BLUEPRINT_AGENT_EXECUTION_CONFIG"


@pytest.mark.parametrize("change_task", [False, True])
def test_replacement_server_owner_cannot_read_or_cancel_saved_task(tmp_path, change_task):
    service, task, record_path, _ = fixture(tmp_path)
    service.enqueue(task.task_id, "fixture-client")
    value = json.loads(record_path.read_text())
    value["owner_client_ids"] = ["replacement-owner"]
    if change_task:
        value["task"]["instructions"] = "Different replacement task"
    write(record_path, value)
    for action in (lambda: service.status(task.task_id, "replacement-owner"),
                   lambda: service.act(task.task_id, "replacement-owner", "cancel")):
        with pytest.raises(AgentExecutionError, match="ownership_record_changed"):
            action()
    assert not service.journal.task(task.task_id)["cancel_requested"]


@pytest.mark.parametrize("secret_kind", ["credential", "webhook"])
def test_world_readable_secret_is_refused(tmp_path, secret_kind):
    service, task, _, _ = fixture(tmp_path)
    path = tmp_path / ("key" if secret_kind == "credential" else "webhook")
    path.write_text("fixture-secret")
    path.chmod(0o644)
    with pytest.raises(AgentExecutionError, match="secret_file_permissions_unsafe"):
        if secret_kind == "credential":
            service._credential(task)
        else:
            production._read_private(path, secret=True)


def test_production_units_bind_sdk_and_offline_replay_boundaries():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "deploy" / "systemd"
    worker = (root / "blueprint-agent-execution.service").read_text()
    assert "Environment=BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=1" in worker
    assert "Environment=BLUEPRINT_AGENT_EXECUTION_CONFIG=" in worker
    offline = (root / "blueprint-agent-stage-replay.service").read_text()
    for boundary in ("User=blueprint", "PrivateNetwork=true", "PrivateDevices=true", "ProtectSystem=strict",
                     "InaccessiblePaths=-/etc/blueprint/provider-secrets"):
        assert boundary in offline
