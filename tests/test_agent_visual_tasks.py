"""Mandatory SAM views cannot be replaced by adaptive crops or model prose."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError, digest
from blueprint_pipeline.agent_execution.production import ProductionAgentService
from blueprint_pipeline.agent_execution.visual_tasks import VisualTaskBinding, VisualView, prepare_visual_task, collect_visual_task
from tests.test_agent_episode_tasks import setup
from tests.test_agent_production_service import write


def build(tmp_path):
    service, _, api, _ = setup(tmp_path)
    config = service.config.model_dump(mode="json")
    guard = json.loads(Path(config["project_guard_receipt_file"]).read_text())
    guard["disclosure_scope"] = "rights_admitted_visual_review_evidence"
    write(Path(config["project_guard_receipt_file"]), guard)
    config["project_guard_receipt_digest"] = digest(guard)
    write(service.config_path, config)
    service = ProductionAgentService(service.config_path, source_commit="a" * 40)
    root = tmp_path / "views"
    root.mkdir()
    views = []
    for index in range(16):
        path = root / f"{index}.png"
        Image.new("RGB", (4, 4), (index, 20, 30)).save(path)
        views.append(VisualView(view_id=f"view_{index}", relative_path=path.name,
            sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(), camera_id=f"camera_{index}", role="sam_overlay"))
    rights_path = tmp_path / "visual-rights.json"
    binding = VisualTaskBinding(review_kind="sam31", candidate_digest=digest({"candidate": 1}),
        evidence_root=str(root), views=tuple(views), mandatory_view_ids=tuple(view.view_id for view in views),
        final_review_input_digest=digest({"original_16_views": [view.sha256 for view in views]}),
        rights_path=str(rights_path), rights_sha256=digest({"placeholder": True}))
    rights = {"schema_version": "blueprint_visual_investigation_rights.v1", "input_manifest_digest": binding.input_manifest_digest,
        "candidate_digest": binding.candidate_digest, "runtime": "openai_agents_api", "model": "gpt-5.6-terra",
        "agent_runtime_policy": {"project_id": config["project_id"], "disclosure_scope": guard["disclosure_scope"],
            "budget_policy": guard["budget_policy"], "session_retention": guard["session_retention"],
            "trace_retention": guard["trace_retention"], "region": "us", "project_guard_receipt_digest": digest(guard)},
        "allowed_image_sha256": sorted(view.sha256 for view in views), "external_disclosure_authorized": True,
        "provider_training_authorized": False, "public_redistribution_authorized": False,
        "accepted_by": "fixture-owner", "source_rights_admission_digest": digest({"owned_fixture": True})}
    rights["rights_digest"] = digest(rights)
    write(rights_path, rights)
    binding = VisualTaskBinding.model_validate({**binding.model_dump(),
        "rights_sha256": "sha256:" + hashlib.sha256(rights_path.read_bytes()).hexdigest()})
    record = prepare_visual_task(service, binding=binding, task_id="visual_task", run_id="visual_run",
        owner_client_id="fixture-client", inference_budget_usd=1)
    runtime = service.runtime_for_task(record.task)
    runtime.transport = api
    runtime.admit(record.task)
    runtime.step(record.task.task_id)
    return service, record, api, runtime


@pytest.mark.parametrize("missing", [False, True])
def test_sam_collection_requires_every_full_view_before_inspected_result(tmp_path, missing):
    service, record, api, runtime = build(tmp_path)
    api.actions = [{"type": "function_call", "turn_id": "turn_1", "call_id": f"call_{i}",
        "name": "inspect_evidence_image", "arguments": {"image_id": f"view_{i}", "crop": [0, 0, 2, 2] if missing and i == 15 else None}}
        for i in range(16)]
    runtime.step(record.task.task_id)
    api.actions = []
    api.output = {"status": "inspected", "summary": "Mandatory views inspected.", "findings": [], "evidence_gaps": [], "final_acceptance_granted": False}
    api.turn_status = "completed"
    runtime.step(record.task.task_id)
    if missing:
        with pytest.raises(AgentExecutionError, match="required_views_missing"):
            collect_visual_task(service, record.task.task_id)
    else:
        receipt = collect_visual_task(service, record.task.task_id)
        assert receipt["mandatory_view_ids"] == list(record.visual_investigation.mandatory_view_ids)
        assert receipt["final_review_input_digest"] == record.visual_investigation.final_review_input_digest
        assert not receipt["final_acceptance_granted"]
        assert receipt["independent_final_review_required"]
