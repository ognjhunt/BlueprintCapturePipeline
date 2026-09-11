"""The real installed SDK receives images and retains independently checked results."""
import base64
import hashlib
import io
import json
from pathlib import Path

import httpx
from PIL import Image
import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from blueprint_pipeline.agent_execution import episode_tasks as ep
from blueprint_pipeline.agent_execution.production import ProductionAgentService
from blueprint_pipeline.agent_execution.visual_tasks import prepare_visual_task, collect_visual_task, VisualTaskBinding
from blueprint_pipeline.agent_execution.contracts import digest
from blueprint_pipeline.task_evaluation_supervisor.sdk_image_tools import encode_tool_output
from blueprint_pipeline.episode_interpretation import materialize_episode_interpretation_rights
from tests.test_agent_episode_tasks import setup
from tests.test_agent_visual_tasks import build
from tests.test_agent_production_service import write
from tests.test_agent_execution_sdk import response
from tests.test_episode_interpretation import _request, _output


def sdk_service(service):
    config = service.config.model_dump(mode="json")
    config["managed_api_enabled"] = False
    # SDK must work without the managed project or retention admission.
    config["project_guard_receipt_file"] = None
    config["project_guard_receipt_digest"] = None
    write(service.config_path, config)
    return ProductionAgentService(service.config_path, source_commit="a" * 40)


def drive_sdk(service, record, calls, output, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "1")
    requests = []
    def handler(request):
        value = json.loads(request.content)
        requests.append(value)
        assert value["store"] is False
        if len(requests) == 1:
            return response([{"type": "function_call", "name": name, "call_id": f"call_{index}",
                "id": f"fc_{index}", "status": "completed", "arguments": json.dumps(arguments)}
                for index, (name, arguments) in enumerate(calls)])
        assert "data:image/png;base64," in request.content.decode()
        return response([{"type": "message", "id": "msg_end", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": json.dumps(output), "annotations": []}]}], number=2)
    runtime = service.runtime_for_task(record.task)
    runtime._hermetic_transport = httpx.MockTransport(handler)
    runtime.admit(record.task)
    state = runtime.step(record.task.task_id)
    assert state["state"] == "completed"
    assert state["result"]["runtime"] == "openai_agents_sdk"
    audit = runtime._audit(record.task).manifest()
    assert audit["reserved_max_cost_usd"] <= record.task.admission.inference_budget_usd
    assert not audit["in_flight_unknown_count"]
    assert len(requests) == 2
    return runtime, requests


def test_episode_sdk_uses_images_and_collects_sealed_interpretation(tmp_path, monkeypatch):
    service, original, _, data = setup(tmp_path)
    service = sdk_service(service)
    request = _request(data)
    rights = tmp_path / "sdk-rights.json"
    materialize_episode_interpretation_rights(episode_id=request.episode_id,
        input_bundle_digest=request.input_receipt["input_bundle_digest"],
        identity=ep.interpreter_identity("openai_agents_sdk", original.task.model), allowed_artifact_roles=ep.ROLES,
        external_disclosure_authorized=True, accepted_by="fixture-owner", accepted_on="2026-09-11",
        authority_reference="test-only", source_rights_admission_digest=digest({"owned_fixture": True}), output_path=rights)
    record = ep.prepare_episode_task(service, task_id="sdk_episode", run_id="sdk_run", request=request,
        rights_path=rights, owner_client_id="fixture-client", runtime="openai_agents_sdk", inference_budget_usd=1)
    before = data["score_path"].read_bytes()
    calls = [("read_episode_context", {}),
        ("read_episode_trace", {"role": "state_trace", "start_step": 0, "end_step": 20, "limit": 100}),
        ("read_episode_trace", {"role": "contact_force_trace", "start_step": 0, "end_step": 20, "limit": 100}),
        ("inspect_episode_interval", {"start_seconds": 0, "end_seconds": 1, "max_observations": 8})]
    runtime, _ = drive_sdk(service, record, calls, _output(data).model_dump(mode="json"), monkeypatch)
    receipt = ep.collect_episode_task(service, record.task.task_id)
    assert receipt["interpreter"]["runtime"] == "openai_agents_sdk"
    assert data["score_path"].read_bytes() == before
    assert ep.collect_episode_task(sdk_service(service), record.task.task_id) == receipt
    assert runtime.cleanup(record.task.task_id)["cleanup_state"] == "deleted"


@pytest.mark.parametrize("missing", [False, True])
def test_sdk_visual_inspection_keeps_all_required_views(tmp_path, monkeypatch, missing):
    original_service, original, _, _ = build(tmp_path)
    service = sdk_service(original_service)
    binding = original.visual_investigation
    path = tmp_path / "sdk-visual-rights.json"
    rights = json.loads(Path(binding.rights_path).read_text())
    rights.update(runtime="openai_agents_sdk", agent_runtime_policy={"project_id": service.config.project_id,
        "disclosure_scope": "rights_admitted_visual_review_evidence", "budget_policy": "strict_per_call",
        "session_retention": "not_admitted", "trace_retention": "not_admitted", "region": "default", "project_guard_receipt_digest": None})
    rights["rights_digest"] = digest({k:v for k,v in rights.items() if k != "rights_digest"})
    write(path, rights)
    binding = VisualTaskBinding.model_validate({**binding.model_dump(), "rights_path": str(path),
        "rights_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()})
    record = prepare_visual_task(service, binding=binding, task_id="sdk_visual", run_id="sdk_run",
        owner_client_id="fixture-client", inference_budget_usd=1, runtime="openai_agents_sdk")
    calls = [("inspect_evidence_image", {"image_id": f"view_{i}", "crop": None}) for i in range(15 if missing else 16)]
    output = {"status": "inspected", "summary": "Inspected admitted views", "findings": [], "evidence_gaps": [], "final_acceptance_granted": False}
    runtime, _ = drive_sdk(service, record, calls, output, monkeypatch)
    if missing:
        with pytest.raises(AgentExecutionError, match="required_views_missing"):
            collect_visual_task(service, record.task.task_id)
    else:
        receipt = collect_visual_task(service, record.task.task_id)
        assert not receipt["missing_mandatory_view_ids"]
        assert not receipt["final_acceptance_granted"] and receipt["independent_final_review_required"]
    assert runtime.cleanup(record.task.task_id)["cleanup_state"] == "deleted"


def test_image_context_uses_pixels_not_compressed_bytes_and_rejects_external_urls():
    data = io.BytesIO()
    Image.new("RGB", (1024,1024), "white").save(data, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
    output, tokens = encode_tool_output([{"type": "input_image", "image_url": url}], model="gpt-5.6-terra")
    assert tokens == 1229 + 1 + 256
    assert output[0].image_url == url
    with pytest.raises(ValueError, match="source_invalid"):
        encode_tool_output([{"type": "input_image", "image_url": "https://private.invalid/a.png"}], model="gpt-5.6-terra")
    with pytest.raises(ValueError, match="model_or_content_invalid"):
        encode_tool_output([{"type": "input_image", "image_url": url}], model="unqualified-model")


def test_context_exhaustion_stops_before_a_second_model_request(tmp_path, monkeypatch):
    from blueprint_pipeline.agent_execution.contracts import AgentTool
    from tests.test_agent_execution_sdk import make_task, setup_runtime
    calls = []
    tool = AgentTool("oversized_evidence", "1", "Read oversized fixture", {
        "type": "object", "properties": {}, "additionalProperties": False}, "read_only",
        lambda *_: {"text": "x" * 50_000})
    def handler(request):
        calls.append(request)
        return response([{"type": "function_call", "id": "fc_1", "call_id": "call_1",
            "name": tool.tool_id, "arguments": "{}", "status": "completed"}])
    runtime = setup_runtime(tmp_path, monkeypatch, handler, tools=(tool,))
    task = make_task((tool,), capability="episode_investigation", max_model_turns=3,
        max_input_tokens=100_000, max_output_tokens=4000, max_tool_output_bytes=100_000)
    runtime.output_models[task.capability] = runtime.output_models.pop("test_investigation")
    runtime.admit(task)
    with pytest.raises(Exception, match="tool_context_ceiling_exceeded"):
        runtime.step(task.task_id)
    assert len(calls) == 1
    assert runtime.inspect(task.task_id)["state"] == "reconciling"
    runtime.step(task.task_id)
    assert len(calls) == 1


def test_visual_producer_selects_sdk_under_its_exact_runtime_policy(tmp_path):
    import time
    from blueprint_pipeline.agent_execution.visual_producer import schedule_visual_investigation
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    original_service, original, _, _ = build(tmp_path)
    service = sdk_service(original_service)
    binding = original.visual_investigation
    source = json.loads(Path(binding.rights_path).read_text())["source_rights_admission_digest"]
    policy = {"project_id": service.config.project_id, "disclosure_scope": "rights_admitted_visual_review_evidence",
        "budget_policy": "strict_per_call", "session_retention": "not_admitted", "trace_retention": "not_admitted",
        "region": "default", "project_guard_receipt_digest": None}
    authority = {"schema_version": "blueprint_managed_visual_batch_authority.v1", "runtime": "openai_agents_sdk",
        "review_kind": "sam31", "source_rights_admission_digest": source, "run_ids": ["sdk-producer"],
        "external_disclosure_authorized": True, "provider_training_authorized": False, "public_redistribution_authorized": False,
        "accepted_by": "fixture-owner", "expires_at": time.time() + 900, "maximum_tasks": 1, "per_task_budget_usd": 1,
        "model": original.task.model, "agent_runtime_policy": policy}
    authority["authority_digest"] = canonical_digest(authority)
    path = tmp_path / "visual-batch.json"
    write(path, authority)
    kwargs = dict(review_kind="sam31", run_id="sdk-producer", candidate_digest=binding.candidate_digest,
        final_review_input_digest=binding.final_review_input_digest,
        frames=[{"path": str(Path(binding.evidence_root) / v.relative_path), "sha256": v.sha256,
            "camera_id": v.camera_id, "role": v.role} for v in binding.views],
        source_rights_admission_digest=source, authority_path=path, service=service)
    queued = schedule_visual_investigation(**kwargs)
    assert schedule_visual_investigation(**kwargs) == queued
    task = service.record(queued["task_id"]).task
    assert task.admission.runtime == "openai_agents_sdk"
    assert task.admission.budget_policy == "strict_per_call"
    assert not queued["final_acceptance_granted"]
