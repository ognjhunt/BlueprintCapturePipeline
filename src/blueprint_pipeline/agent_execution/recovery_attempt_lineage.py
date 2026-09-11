"""Read-only proof of a controller's already admitted same-release retry."""
from __future__ import annotations

from pathlib import Path

from ..decision_evidence_contracts import canonical_digest
from .recovery_lineage import _attempt, _attempt_namespace, _link, _reference, _require


def recovery_edge(previous_ref, current_ref, previous_link_ref, state, directory, anchor, config, intent):
    from ..task_evaluation_scene_recovery import validate_recovery_evidence
    before, after = (_attempt(ref, directory, anchor) for ref in (previous_ref, current_ref))
    recovery = after.get("recovery", {})
    refs = recovery.get("evidence", {})
    edges = [row for row in state.get("recovery_predecessors", [])
             if row.get("attempt") == previous_ref and row.get("evidence") == refs]
    execution = intent["request"]["execution"]
    _require(len(edges) == 1 and recovery.get("prior_attempt_id") == before["attempt_id"]
             and recovery.get("prior_attempt_digest") == before["attempt_digest"]
             and all(before.get(key) == after.get(key) for key in
                     ("source_commit", "input_digest", "provider", "maximum_spend_usd"))
             and after["provider"] in execution.get("allowed_providers", [])
             and 0 < after["maximum_spend_usd"] <= execution["max_total_spend_usd"], "recorded_recovery_required")
    paths = sorted((directory / "attempts").glob("*.json"))
    _require(len(paths) <= 10000, "attempt_inventory_unbounded")
    attempts = [_attempt({"path": str(path), **_file_identity(path)}, directory, anchor) for path in paths]
    _require(sum("recovery" in row for row in attempts) <= execution["max_retries"]
             and execution["max_retries"] > 0, "recovery_limit_exceeded")
    _require(set(refs) == {"failure", "provider_guard", "ownership_reconciliation"}, "recovery_evidence_missing")
    output = Path(config["factory_output_root"]) / anchor.intent_id / before["attempt_id"]
    failure = _reference(refs["failure"], root=output, field="failure_digest")
    for name in ("provider_guard", "ownership_reconciliation"):
        _reference(refs[name], root=output)
    prior_link = _link(anchor, previous_link_ref, directory)
    _require(prior_link["team_namespace"] == _attempt_namespace(anchor, before)
             and failure.get("parent_request_digest") == prior_link["request_digest"], "recovery_parent_changed")
    queue = Path(config["child_queue_root"])
    child = _reference(failure.get("child_job"), root=queue / "failed", field="job_digest")
    result = _reference(failure.get("child_result"), root=queue / "results", field="result_digest")
    identity = {key: child.get(key) for key in ("parent_request_digest", "plan_digest", "phase", "inputs_digest")}
    _require(child.get("schema_version") == "task_evaluation_sam31_preparation_execution_job.v1"
             and child.get("parent_request_digest") == prior_link["request_digest"]
             and child.get("parent_preparation_id") == prior_link["preparation_id"]
             and child.get("expected_source_commit") == before["source_commit"]
             and child.get("child_id") == failure.get("child_id") == "sam31-" + canonical_digest(identity)[7:]
             and Path(failure["child_job"]["path"]).name == child["child_id"] + ".json"
             and Path(failure["child_result"]["path"]).name == child["child_id"] + ".json"
             and result.get("status") == "failed" and result.get("child_id") == child["child_id"]
             and result.get("job_digest") == child["job_digest"], "recovery_failed_child_changed")
    producer_refs = [failure["child_result"], *[ref for name, ref in result.get("artifacts", {}).items()
                                               if name.endswith("allocator_result")]]
    _require(failure.get("producer_result") in producer_refs, "recovery_producer_not_owned")
    # Reopen the same canonical grant at its recorded reservation time. Current
    # consent is checked separately on every use; no historical zero is promoted
    # into authority to allocate now.
    validated = validate_recovery_evidence(refs, prior_attempt=before, provider=after["provider"],
                                           now=after["reserved_at_epoch"])
    _require(validated == recovery, "recovery_grant_changed")
    return {"kind": "same_release_recovery", "previous_attempt": previous_ref,
            "successor_attempt": current_ref, "reconciliation": refs}


def _file_identity(path):
    from .recovery_lineage import _read
    _, ref = _read(path)
    return {key: ref[key] for key in ("sha256", "size_bytes")}
