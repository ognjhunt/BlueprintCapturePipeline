"""Read real failed child/ownership records before reserving a new attempt."""
from __future__ import annotations

from datetime import datetime, timezone

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_public_scene_attempt_factory import record
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
from .task_evaluation_scene_progression_state import require, safe_path
from . import task_evaluation_scene_intake as intake


def failure_kind(value):
    text = ";".join(str(x) for x in value.get("blockers", [])) + ";" + str(value.get("blocker") or "")
    if "create_outcome_ambiguous" in text or value.get("allocation_outcome_ambiguous") is True:
        return "create_ambiguous"
    if "started_without_terminal_reconciliation" in text or "create_returned_null" in text:
        return "create_null"
    if (value.get("allocation_created") is False or "create_refused" in text or "create_rejected" in text
            or ("instance_not_created" in text and value.get("provider_mutations_performed") == 0
                and value.get("instance_id") in (None, "") and value.get("provider_mutation_outcome_ambiguous") is False)):
        return "create_refused"
    return None


def retain_failure(*, attempt, link, child_queue_root, output_root, now):
    """Select only a real terminal failure bound to this exact preparation."""
    queue, output = safe_path(child_queue_root), safe_path(output_root)
    candidates = []
    for path in sorted((queue / "failed").glob("*.json")):
        job = read(path, digest_field="job_digest")
        if job.get("parent_request_digest") != link["request_digest"]:
            continue
        key = {name: job.get(name) for name in ("parent_request_digest", "plan_digest", "phase", "inputs_digest")}
        require(job.get("expected_source_commit") == attempt["source_commit"]
                and job.get("parent_preparation_id") == link["preparation_id"]
                and job.get("child_id") == "sam31-" + canonical_digest(key)[7:], "failure_release_mismatch")
        result_path = queue / "results" / path.name
        result = read(result_path, digest_field="result_digest")
        require(result.get("job_digest") == job["job_digest"] and result.get("status") == "failed"
                and result.get("child_id") == job["child_id"], "failure_child_binding_invalid")
        paths = [result_path]
        for name, ref in result.get("artifacts", {}).items():
            if name.endswith("allocator_result"):
                paths.insert(0, checked_file(ref["path"], ref))
        for producer_path in paths:
            producer = read(producer_path)
            kind = failure_kind(producer)
            if kind is not None:
                candidates.append((job, producer_path, kind, path, result_path))
                break
    if not candidates:
        return None
    require(len(candidates) == 1, "failure_identity_ambiguous")
    job, producer, kind, job_path, result_path = candidates[0]
    path = output / (attempt["attempt_id"] + "-failure.json")
    if path.exists():
        value = read(path, digest_field="failure_digest")
        require(value.get("attempt_digest") == attempt["attempt_digest"]
                and value.get("producer_result") == record(producer), "failure_record_changed")
        return path
    value = {"schema_version": "task_evaluation_scene_attempt_failure.v1", "status": "failed",
        "attempt_digest": attempt["attempt_digest"], "failure_kind": kind, "observed_at_epoch": now,
        "producer_result": record(producer), "child_job": record(job_path), "child_result": record(result_path),
        "child_id": job["child_id"], "parent_request_digest": link["request_digest"]}
    value["failure_digest"] = canonical_digest(value, digest_field="failure_digest")
    output.mkdir(parents=True, exist_ok=True, mode=0o750)
    intake.write_exclusive(path, value)
    return path


def reconcile_ownership(*, attempt, failure_path, config, output_root, now):
    """Observe all configured ownership roots and global inventory; never reap.

    Unresolved pending teardowns continue to block. This function never closes
    an ambiguous-create obligation just because a lease's clock expired.
    """
    from .paid_provider_lane_lease import LEASE_SCHEMA_VERSION, _lease_is_stale
    from .paid_lane_guard import PENDING_TEARDOWN_SCHEMA_VERSION
    from .task_evaluation_launch_reconciler import _guard_provider_zero
    failure = read(failure_path, digest_field="failure_digest")
    guard_path = safe_path(config["provider_guard_path"])
    guard = read(guard_path)
    verified, blockers = _guard_provider_zero(guard=guard, required_providers=[attempt["provider"]],
        max_age_seconds=300, now=datetime.fromtimestamp(now, timezone.utc),
        not_before=datetime.fromtimestamp(failure["observed_at_epoch"], timezone.utc))
    require(verified, "recovery_provider_zero_required:" + ";".join(blockers))
    roots = [safe_path(p) for p in config.get("ownership_roots", [])]
    require(bool(roots) and all(p.is_dir() for p in roots), "ownership_scope_missing")
    required = [safe_path(config[key]) for key in ("child_execution_root", "launch_execution_root")]
    require(all(any(path == root or path.is_relative_to(root) for root in roots) for path in required),
            "ownership_scope_incomplete")
    paths = set()
    for root in roots:
        paths.update(root.rglob("*.lease.json"))
        paths.update(root.glob("**/pending_teardowns/*.json"))
        paths.update(root.glob("**/pending-teardowns/*.json"))
    require(len(paths) <= 20000, "ownership_scan_bound_exceeded")
    rows, active, unresolved = [], 0, 0
    for path in sorted(paths):
        value = read(safe_path(path))
        if path.name.endswith(".lease.json"):
            require(value.get("schema_version") == LEASE_SCHEMA_VERSION, "ownership_lease_invalid")
            stale, reason = _lease_is_stale(value)
            active += int(not stale)
            rows.append({"record": record(path), "kind": "lease", "stale": stale, "reason": reason})
        else:
            require(value.get("schema_version") == PENDING_TEARDOWN_SCHEMA_VERSION, "ownership_pending_record_invalid")
            require(value.get("status") in {"open", "closed", "cancelled_no_allocation"}, "ownership_pending_status_invalid")
            unresolved += int(value["status"] == "open")
            rows.append({"record": record(path), "kind": "pending_teardown", "status": value["status"]})
    queue_rows = []
    for root_name, states in (("child_queue_root", ("pending", "processing", "waiting_external")),
                             ("launch_queue_root", ("processing",))):
        root = safe_path(config[root_name])
        require(root.is_dir(), "ownership_queue_missing")
        for state in states:
            for path in sorted((root / state).glob("*.json")):
                read(safe_path(path))
                queue_rows.append(record(path))
    active += len(queue_rows)
    value = {"schema_version": "task_evaluation_scene_attempt_ownership.v1",
        "attempt_digest": attempt["attempt_digest"], "status": "closed_without_resource" if active == unresolved == 0 else "unresolved",
        "active_writer_count": active, "unresolved_create_count": unresolved,
        "provider_guard": record(guard_path), "observed_at_epoch": now,
        "ownership_roots": [str(root) for root in roots], "observed_records": rows, "active_queue_records": queue_rows,
        "provider_mutation_performed": False, "historical_failure_modified": False}
    value["ownership_digest"] = canonical_digest(value, digest_field="ownership_digest")
    output = safe_path(output_root)
    output.mkdir(parents=True, exist_ok=True, mode=0o750)
    # Snapshot a mutable guard before binding a retry, so future guard refreshes
    # cannot invalidate the retained basis or change what was reconciled.
    guard_snapshot = output / (record(guard_path)["sha256"][7:] + "-guard.json")
    if not guard_snapshot.exists():
        intake.write_exclusive(guard_snapshot, guard)
    value["provider_guard"] = record(guard_snapshot)
    value["ownership_digest"] = canonical_digest(value, digest_field="ownership_digest")
    path = output / (value["ownership_digest"][7:] + ".json")
    if not path.exists():
        intake.write_exclusive(path, value)
    require(active == unresolved == 0, "ownership_reconciliation_required")
    return {"failure": record(failure_path), "provider_guard": record(guard_snapshot), "ownership_reconciliation": record(path)}
