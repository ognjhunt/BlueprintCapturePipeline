"""Controller-native retries retain exact failed-child and owner-budget proof."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from blueprint_pipeline.agent_execution.controller_recovery import ControllerRecoveryBinding
from blueprint_pipeline.agent_execution.production import ProductionAgentService
from blueprint_pipeline.agent_execution.recovery_lineage import resolve_recovery_binding, recovery_binding_authorized
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_progression_state import advance
from tests.test_agent_production_service import fixture, write
from tests.test_agent_recovery_lineage import sealed, record


def same_release_case(tmp_path, *, defect=None):
    service, _, _, _ = fixture(tmp_path)
    now = time.time() - 30
    intent = {"schema_version": "task_evaluation_scene_intent.v1", "intent_id": "scene-retry",
        "request": {"owner": {"user_id": "owner"}, "task": {"id": "frozen-task"},
            "consent": {"spend_authorized": True, "task_confirmed": True},
            "execution": {"allowed_providers": ["vast"], "max_total_spend_usd": 50,
                "max_retries": 0 if defect == "retry_limit" else 2, "expires_at_epoch": now + 1800}}}
    directory = tmp_path / "intents" / intent["intent_id"]
    sealed(directory / "intent.json", intent, "intent_digest", cross=True)
    intent = json.loads((directory / "intent.json").read_text())
    queue, children = tmp_path / "parents", tmp_path / "children"
    config = {"schema_version": "task_evaluation_scene_progression_config.v1", "intent_root": str(directory.parent),
        "only_intent_id": intent["intent_id"], "preparation_queue_root": str(queue),
        "child_queue_root": str(children), "factory_output_root": str(tmp_path / "factory")}
    controller = tmp_path / "controller.json"
    sealed(controller, config, "config_digest")
    old = {"schema_version": "task_evaluation_scene_attempt.v1", "intent_id": intent["intent_id"],
        "intent_digest": intent["intent_digest"], "attempt_id": "source-old", "source_commit": "a" * 40,
        "input_digest": "sha256:" + "d" * 64, "provider": "vast", "maximum_spend_usd": 4.5,
        "runtime_digest": "sha256:" + "e" * 64,
        "reserved_at_epoch": now}
    old_ref = sealed(directory / "attempts/source-old.json", old, "attempt_digest", cross=True)
    old = json.loads(Path(old_ref["path"]).read_text())

    def parent(label, attempt):
        namespace = "scene-" + canonical_digest({"intent_digest": intent["intent_digest"],
            "attempt_digest": attempt["attempt_digest"]})[7:55]
        request = {"preparation_id": label, "expected_production_commit": attempt["source_commit"],
            "team_namespace": namespace, "scene": {"identity": {"id": "scene-frozen"}},
            "task": {"identity": {"id": "task-frozen"}}}
        request_digest = canonical_digest(request)
        filename = label + "-" + request_digest[7:] + ".json"
        sealed(queue / "blocked" / filename, {"schema_version": "task_evaluation_launch_preparation_envelope.v1",
            "request": request, "request_digest": request_digest}, "envelope_digest")
        link = {"schema_version": "task_evaluation_scene_preparation_link.v1", "intent_id": intent["intent_id"],
            "intent_digest": intent["intent_digest"], "request_digest": request_digest, "preparation_id": label,
            "expected_production_commit": attempt["source_commit"], "scene_id": "scene-frozen", "task_id": "task-frozen",
            "team_namespace": namespace, "result_filename": filename}
        return sealed(directory / "preparations" / (request_digest[7:] + ".json"), link, "link_digest"), request_digest

    old_link, old_request = parent("old-parent", old)
    output = tmp_path / "factory" / intent["intent_id"] / old["attempt_id"]
    producer = sealed(output / "producer.json", {"status": "failed", "allocation_created": defect == "producer"}, "receipt_digest")
    child = {"schema_version": "task_evaluation_sam31_preparation_execution_job.v1", "parent_request_digest": old_request,
        "parent_preparation_id": "old-parent", "expected_source_commit": "a" * 40,
        "plan_digest": "sha256:" + "e" * 64, "inputs_digest": "sha256:" + "f" * 64, "phase": "sam31_tracking"}
    child["child_id"] = "sam31-" + canonical_digest({k: child[k] for k in ("parent_request_digest", "plan_digest", "phase", "inputs_digest")})[7:]
    if defect == "child_parent":
        child["parent_request_digest"] = "sha256:" + "0" * 64
    job_ref = sealed(children / "failed" / (child["child_id"] + ".json"), child, "job_digest")
    child = json.loads(Path(job_ref["path"]).read_text())
    result_ref = sealed(children / "results" / (child["child_id"] + ".json"), {
        "schema_version": "task_evaluation_sam31_preparation_execution_result.v1", "status": "failed",
        "child_id": child["child_id"], "job_digest": child["job_digest"],
        "artifacts": {"sam31_allocator_result": producer}}, "result_digest")
    failure_ref = sealed(output / "failure.json", {"schema_version": "task_evaluation_scene_attempt_failure.v1",
        "status": "failed", "failure_kind": "create_refused", "attempt_digest": old["attempt_digest"],
        "observed_at_epoch": now + 5, "producer_result": producer, "child_job": job_ref, "child_result": result_ref,
        "child_id": child["child_id"], "parent_request_digest": old_request}, "failure_digest")
    guard = {"schema_version": "gpu_spend_guard.v1", "status": "passed",
        "generated_at": datetime.fromtimestamp(now - 400 if defect == "stale_zero" else now + 7, timezone.utc).isoformat(),
        "reap_mode": True, "provider_zero_verified": True, "live_instance_count": 0, "total_burn_per_hour_usd": 0,
        "reap_candidate_ids": [], "reap_results": [], "provider_zero": {"status": "verified", "global_live_instance_count": 0,
            "global_total_burn_per_hour_usd": 0, "required_provider_ids": ["vast"]},
        "inventory_results": [{"provider": "vast", "status": "succeeded", "row_count": 0, "required": True}]}
    guard_ref = sealed(output / "guard.json", guard, "guard_digest")
    owner_ref = sealed(output / "ownership.json", {"schema_version": "task_evaluation_scene_attempt_ownership.v1",
        "status": "closed_without_resource", "attempt_digest": old["attempt_digest"], "active_writer_count": 0,
        "unresolved_create_count": 0, "provider_guard": guard_ref, "observed_at_epoch": now + 8}, "ownership_digest")
    evidence = {"failure": failure_ref, "provider_guard": guard_ref, "ownership_reconciliation": owner_ref}
    new = {**old, "attempt_id": "source-new", "reserved_at_epoch": now + 10,
        "recovery": {"prior_attempt_id": old["attempt_id"], "prior_attempt_digest": old["attempt_digest"],
            "failure_digest": json.loads(Path(failure_ref["path"]).read_text())["failure_digest"], "evidence": evidence}}
    if defect == "runtime":
        new["runtime_digest"] = "sha256:" + "f" * 64
    new_ref = sealed(directory / "attempts/source-new.json", new, "attempt_digest", cross=True)
    new = json.loads(Path(new_ref["path"]).read_text())
    new_link, new_request = parent("new-parent", new)
    first = advance(directory, intent, None, status="blocked", phase="preparation",
        state={"attempt": old_ref, "preparation_link": old_link}, now=now + 9)
    state = {"attempt": new_ref, "preparation_link": new_link,
             "recovery_predecessors": [] if defect == "history" else [{"attempt": old_ref, "evidence": evidence}]}
    advance(directory, intent, first, status="blocked", phase="preparation", state=state, now=now + 11)
    anchor = ControllerRecoveryBinding(recovery_id="owner-anchor", intent_id=intent["intent_id"], intent_digest=intent["intent_digest"],
        controller_config_path=str(controller), controller_config_sha256=record(controller)["sha256"], required_replay_id="failed_boundary",
        parent_request_digest=old_request, preparation_link_path=old_link["path"], preparation_link_sha256=old_link["sha256"], allow_controller_successors=True)
    installed = service.config.model_dump(mode="json")
    installed.update(automatic_run_supervision=True, automatic_failure_investigation=True, automatic_recovery_bindings=[anchor.model_dump(mode="json")])
    write(service.config_path, installed)
    return ProductionAgentService(service.config_path, source_commit="a" * 40), anchor, directory, queue, new_request


def test_same_release_recovery_reopens_canonical_grant_and_keeps_original_records(tmp_path):
    service, anchor, directory, queue, request = same_release_case(tmp_path)
    before = {p: p.read_bytes() for p in directory.rglob("*.json")}
    derived = resolve_recovery_binding(service, intent_id=anchor.intent_id, parent_request_digest=request, parent_queue_root=str(queue))
    assert derived.parent_request_digest == request and recovery_binding_authorized(service, derived)
    receipt = json.loads(next((service.journal.root / "controller-recovery-bindings").glob("*.json")).read_text())
    assert receipt["lineage_edges"][0]["kind"] == "same_release_recovery"
    assert receipt["unchanged_authority"]["execution_bounds"]["max_retries"] == 2
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.parametrize("defect", ["retry_limit", "producer", "child_parent", "stale_zero", "history", "runtime"])
def test_same_release_recovery_refuses_unadmitted_or_unrelated_evidence(tmp_path, defect):
    service, anchor, _, queue, request = same_release_case(tmp_path, defect=defect)
    with pytest.raises((AgentExecutionError, ValueError)):
        resolve_recovery_binding(service, intent_id=anchor.intent_id, parent_request_digest=request, parent_queue_root=str(queue))
    assert not list((service.journal.root / "controller-recovery-bindings").glob("*.json"))
