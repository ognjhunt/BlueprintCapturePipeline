"""Only controller-recorded compatible successors inherit exact recovery scope."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from blueprint_pipeline.agent_execution.controller_recovery import ControllerRecoveryBinding
from blueprint_pipeline.agent_execution.production import ProductionAgentService
from blueprint_pipeline.agent_execution.recovery_lineage import resolve_recovery_binding, recovery_binding_authorized
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_scene_progression_state import advance
from tests.test_agent_production_service import fixture, write


def record(path):
    raw = path.read_bytes()
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def sealed(path, value, field, *, cross=False):
    value = deepcopy(value)
    value[field] = (cross_runtime_canonical_digest if cross else canonical_digest)(value, digest_field=field)
    if path.exists():
        path.chmod(0o600)
    write(path, value)
    return record(path)


def lineage_case(tmp_path):
    service, _, _, _ = fixture(tmp_path)
    intent = {"schema_version": "task_evaluation_scene_intent.v1", "intent_id": "scene-owner",
        "request": {"owner": {"user_id": "owner"}, "task": {"id": "frozen-task"},
            "consent": {"spend_authorized": True, "task_confirmed": True},
            "execution": {"max_total_spend_usd": 50, "max_retries": 2, "expires_at_epoch": time.time() + 1800}}}
    directory = tmp_path / "intents" / intent["intent_id"]
    sealed(directory / "intent.json", intent, "intent_digest", cross=True)
    intent = json.loads((directory / "intent.json").read_text())
    queue = tmp_path / "owned-queue"
    config = {"schema_version": "task_evaluation_scene_progression_config.v1", "intent_root": str(directory.parent),
        "only_intent_id": intent["intent_id"], "preparation_queue_root": str(queue), "factory_output_root": str(tmp_path / "factory")}
    controller = tmp_path / "controller.json"
    sealed(controller, config, "config_digest")

    def parent(label, commit):
        request = {"preparation_id": label, "expected_production_commit": commit,
                   "scene": {"identity": {"id": "scene-frozen"}}, "task": {"identity": {"id": "task-frozen"}}}
        request_digest = canonical_digest(request)
        filename = label + "-" + request_digest[7:] + ".json"
        envelope = sealed(queue / "blocked" / filename,
            {"schema_version": "task_evaluation_launch_preparation_envelope.v1", "request_digest": request_digest,
             "request": request}, "envelope_digest")
        link = {"schema_version": "task_evaluation_scene_preparation_link.v1", "intent_id": intent["intent_id"],
            "intent_digest": intent["intent_digest"], "request_digest": request_digest, "preparation_id": label,
            "expected_production_commit": commit, "scene_id": "scene-frozen", "task_id": "task-frozen",
            "team_namespace": "owner-team", "result_filename": filename}
        return sealed(directory / "preparations" / (request_digest[7:] + ".json"), link, "link_digest"), envelope, request_digest

    old_link, old_envelope, old_request = parent("old-parent", "a" * 40)
    new_link, _, new_request = parent("new-parent", "b" * 40)
    attempts = []
    for label, commit in (("source-old", "a" * 40), ("source-new", "b" * 40)):
        attempts.append(sealed(directory / "attempts" / (label + ".json"),
            {"schema_version": "task_evaluation_scene_attempt.v1", "intent_id": intent["intent_id"],
             "intent_digest": intent["intent_digest"], "attempt_id": label, "source_commit": commit,
             "input_digest": "sha256:" + "d" * 64, "provider": "vast", "maximum_spend_usd": 10},
            "attempt_digest", cross=True))
    old_attempt = json.loads(Path(attempts[0]["path"]).read_text())
    output = tmp_path / "factory" / intent["intent_id"] / "source-old"
    guard = sealed(output / "guard.json", {"schema_version": "gpu_spend_guard.v1", "status": "passed"}, "guard_digest")
    transition = sealed(output / "transition.json", {"schema_version": "task_evaluation_scene_release_transition.v1",
        "attempt_digest": old_attempt["attempt_digest"], "parent_state": "blocked", "parent_envelope": old_envelope}, "failure_digest")
    ownership = sealed(output / "ownership.json", {"schema_version": "task_evaluation_scene_attempt_ownership.v1",
        "attempt_digest": old_attempt["attempt_digest"], "status": "closed_without_resource",
        "active_writer_count": 0, "unresolved_create_count": 0, "provider_guard": guard}, "ownership_digest")
    state = {"attempt": attempts[0], "preparation_link": old_link}
    prior = advance(directory, intent, None, status="blocked", phase="preparation", state=state, now=time.time())
    next_state = {"attempt": attempts[1], "preparation_link": new_link, "release_predecessors": [{
        "attempt": attempts[0], "new_source_commit": "b" * 40,
        "basis": "terminal_preparation_and_reconciled_global_ownership",
        "reconciliation": {"failure": transition, "provider_guard": guard, "ownership_reconciliation": ownership}}]}
    advance(directory, intent, prior, status="blocked", phase="preparation", state=next_state, now=time.time())
    anchor = ControllerRecoveryBinding(recovery_id="owner-anchor", intent_id=intent["intent_id"],
        intent_digest=intent["intent_digest"], controller_config_path=str(controller), controller_config_sha256=record(controller)["sha256"],
        required_replay_id="failed_boundary", parent_request_digest=old_request,
        preparation_link_path=old_link["path"], preparation_link_sha256=old_link["sha256"], allow_controller_successors=True)
    installed = service.config.model_dump(mode="json")
    installed.update(source_commit="b" * 40, automatic_run_supervision=True,
                     automatic_failure_investigation=True, automatic_recovery_bindings=[anchor.model_dump(mode="json")])
    write(service.config_path, installed)
    service = ProductionAgentService(service.config_path, source_commit="b" * 40)
    return service, anchor, directory, queue, new_request


def test_exact_successor_derivation_is_immutable_and_requires_current_anchor(tmp_path):
    service, anchor, directory, queue, request = lineage_case(tmp_path)
    originals = {p: p.read_bytes() for p in directory.rglob("*.json")}
    derived = resolve_recovery_binding(service, intent_id=anchor.intent_id, parent_request_digest=request, parent_queue_root=str(queue))
    assert derived.parent_request_digest == request and derived != anchor
    assert derived.allow_controller_successors is False
    assert derived.required_replay_id == "failed_boundary"
    assert recovery_binding_authorized(service, derived)
    assert resolve_recovery_binding(service, intent_id=anchor.intent_id, parent_request_digest=request, parent_queue_root=str(queue)) == derived
    assert len(list((service.journal.root / "controller-recovery-bindings").glob("*.json"))) == 1
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    config = service.config.model_dump(mode="json")
    config["automatic_recovery_bindings"] = []
    write(service.config_path, config)
    revoked = ProductionAgentService(service.config_path, source_commit="b" * 40)
    assert not recovery_binding_authorized(revoked, derived)


def test_new_approved_allowance_gets_distinct_derivation_without_rewriting_history(tmp_path):
    service, anchor, _, queue, request = lineage_case(tmp_path)
    old = resolve_recovery_binding(service, intent_id=anchor.intent_id, parent_request_digest=request, parent_queue_root=str(queue))
    old_path = next((service.journal.root / "controller-recovery-bindings").glob("*.json"))
    before = old_path.read_bytes()
    config = service.config.model_dump(mode="json")
    config["automatic_supervision_allowances"] = [{"intent_id": anchor.intent_id, "intent_digest": anchor.intent_digest,
        "maximum_revisions": 6, "maximum_reserved_inference_usd": 6, "expires_at": time.time() + 900,
        "authorization_reference": "fixture explicit owner amendment"}]
    write(service.config_path, config)
    changed = ProductionAgentService(service.config_path, source_commit="b" * 40)
    new = resolve_recovery_binding(changed, intent_id=anchor.intent_id, parent_request_digest=request, parent_queue_root=str(queue))
    assert new.recovery_id != old.recovery_id
    assert old_path.read_bytes() == before
    assert not recovery_binding_authorized(changed, old) and recovery_binding_authorized(changed, new)
    assert len(list((service.journal.root / "controller-recovery-bindings").glob("*.json"))) == 2


def test_derived_binding_reaches_real_failure_task_admission_and_revokes(tmp_path):
    from blueprint_pipeline.agent_execution.prepare import prepare_retained_failure
    from tests.test_task_evaluation_stage_replay import _queue, CHILD
    service, anchor, _, queue, request = lineage_case(tmp_path)
    derived = resolve_recovery_binding(service, intent_id=anchor.intent_id, parent_request_digest=request, parent_queue_root=str(queue))
    children, path, job = _queue(tmp_path / "retained-child")
    job.update(parent_request_digest=request, parent_preparation_id="new-parent", expected_source_commit="b" * 40)
    write(path, job)
    record_value = prepare_retained_failure(service, task_id="auto-failure-current-successor", run_id=anchor.intent_id,
        child_id=CHILD, owner_client_id="fixture-client", inference_budget_usd=1, queue_root=children,
        parent_queue_root=queue, input_root=tmp_path, approved_roots=(tmp_path,), controller_recovery=derived, autostart=False)
    service.validate_admission(record_value.task)
    assert set(record_value.task.tool_ids) == {"replay_retained_stage", "request_preauthorized_scene_progression"}
    config = service.config.model_dump(mode="json")
    config["automatic_recovery_bindings"][0]["allow_controller_successors"] = False
    write(service.config_path, config)
    revoked = ProductionAgentService(service.config_path, source_commit="b" * 40)
    with pytest.raises(AgentExecutionError, match="automatic_recovery_scope_revoked"):
        revoked.validate_admission(record_value.task)


@pytest.mark.parametrize("defect", ["opt_out", "queue", "event", "lineage", "task", "traversal", "guard", "revoked", "expiry", "config"])
def test_unapproved_or_tampered_successors_never_inherit_recovery(tmp_path, defect):
    service, anchor, directory, queue, request = lineage_case(tmp_path)
    if defect == "opt_out":
        config = service.config.model_dump(mode="json")
        config["automatic_recovery_bindings"][0]["allow_controller_successors"] = False
        write(service.config_path, config)
        opted_out = ProductionAgentService(service.config_path, source_commit="b" * 40)
        assert resolve_recovery_binding(opted_out, intent_id=anchor.intent_id, parent_request_digest=request, parent_queue_root=str(queue)) is None
        return
    if defect == "queue":
        queue = tmp_path / "foreign-queue"
    elif defect == "event":
        (directory / "progression-events/000002.json").chmod(0o600)
        (directory / "progression-events/000002.json").write_text("{}")
    elif defect in {"lineage", "task", "traversal"}:
        event_path = directory / "progression-events/000002.json"
        event = json.loads(event_path.read_text())
        if defect == "lineage":
            event["state"]["release_predecessors"] = []
        elif defect == "task":
            ref = event["state"]["preparation_link"]
            value = json.loads(Path(ref["path"]).read_text())
            value["task_id"] = "another-task"
            event["state"]["preparation_link"] = sealed(Path(ref["path"]), value, "link_digest")
        else:
            ref = event["state"]["preparation_link"]
            ref["path"] = str(directory / "preparations" / ".." / "preparations" / Path(ref["path"]).name)
        event.pop("event_digest")
        sealed(event_path, event, "event_digest", cross=True)
        event = json.loads(event_path.read_text())
        projection = json.loads((directory / "progression.json").read_text())
        projection.update(state=event["state"], last_event_digest=event["event_digest"])
        sealed(directory / "progression.json", projection, "progression_digest", cross=True)
    elif defect == "guard":
        (tmp_path / "factory" / anchor.intent_id / "source-old/guard.json").write_text("{}")
    elif defect == "revoked":
        write(directory / "revoked.json", {})
    elif defect == "expiry":
        import blueprint_pipeline.agent_execution.recovery_lineage as module
        original = module.time.time
        try:
            module.time.time = lambda: original() + 3600
            with pytest.raises(AgentExecutionError, match="owner_authority_revoked"):
                resolve_recovery_binding(service, intent_id=anchor.intent_id, parent_request_digest=request, parent_queue_root=str(queue))
        finally:
            module.time.time = original
        return
    else:
        write(Path(anchor.controller_config_path), {"changed": True})
    with pytest.raises((AgentExecutionError, ValueError, KeyError)):
        resolve_recovery_binding(service, intent_id=anchor.intent_id, parent_request_digest=request, parent_queue_root=str(queue))
    assert not list((service.journal.root / "controller-recovery-bindings").glob("*.json"))
