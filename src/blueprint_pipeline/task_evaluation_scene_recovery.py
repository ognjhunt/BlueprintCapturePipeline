"""Reconcile a failed immutable attempt before reserving a distinct successor.

The original jobs, provider receipts and maximum exposure stay intact. These
predicates grant no allocation authority; the canonical paid gate still runs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, require


def validate_recovery_evidence(evidence, *, prior_attempt, provider, now):
    """Reopen server-retained failure and global guard bytes, never trust labels."""
    require(isinstance(evidence, dict) and set(evidence) == {
        "failure", "provider_guard", "ownership_reconciliation"}, "scene_recovery_evidence_invalid")
    values = {}
    for name, ref in evidence.items():
        require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "size_bytes"},
                "scene_recovery_reference_invalid")
        path = Path(ref["path"])
        require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)),
                "scene_recovery_reference_unsafe")
        checked_file(path, ref)
        values[name] = read(path)
    failure = values["failure"]
    require(failure.get("schema_version") == "task_evaluation_scene_attempt_failure.v1"
            and failure.get("failure_digest") == canonical_digest(failure, digest_field="failure_digest")
            and failure.get("attempt_digest") == prior_attempt["attempt_digest"]
            and failure.get("status") == "failed"
            and failure.get("failure_kind") in {"create_refused", "create_null", "create_ambiguous"},
            "scene_recovery_failure_not_recoverable")
    failed_at = failure.get("observed_at_epoch")
    require(type(failed_at) in (int, float) and prior_attempt["reserved_at_epoch"] <= failed_at <= now,
            "scene_recovery_failure_time_invalid")
    # Reopen the original outcome too. A manufactured wrapper with no retained
    # producer failure is insufficient evidence for autonomous recovery.
    original = failure.get("producer_result")
    require(isinstance(original, dict), "scene_recovery_producer_evidence_missing")
    original_path = Path(original["path"])
    require(original_path.is_absolute() and not any(p.is_symlink() for p in (original_path, *original_path.parents)),
            "scene_recovery_producer_path_unsafe")
    checked_file(original_path, original)
    producer = read(original_path)
    require(producer.get("status") in {"failed", "blocked", "refused"}
            and producer.get("allocation_created") is not True,
            "scene_recovery_producer_not_failed")
    blockers = [str(b) for b in producer.get("blockers", [])]
    blockers.append(str(producer.get("blocker") or ""))
    create_evidence = (producer.get("allocation_created") is False
                       or producer.get("allocation_outcome_ambiguous") is True
                       or (any("instance_not_created" in b for b in blockers)
                           and producer.get("provider_mutations_performed") == 0
                           and producer.get("instance_id") in (None, "")
                           and producer.get("provider_mutation_outcome_ambiguous") is False)
                       or any(marker in blocker for blocker in blockers for marker in (
                           "create_outcome_ambiguous", "create_refused", "create_rejected",
                           "create_returned_null", "started_without_terminal_reconciliation")))
    require(create_evidence, "scene_recovery_create_failure_evidence_missing")
    from .task_evaluation_launch_reconciler import _guard_provider_zero
    zero, blockers = _guard_provider_zero(
        guard=values["provider_guard"], required_providers=[provider], max_age_seconds=300,
        now=datetime.fromtimestamp(now, timezone.utc),
        not_before=datetime.fromtimestamp(failed_at, timezone.utc))
    require(zero, "scene_recovery_authoritative_zero_required:" + ";".join(blockers))
    ownership = values["ownership_reconciliation"]
    require(ownership.get("schema_version") == "task_evaluation_scene_attempt_ownership.v1"
            and ownership.get("ownership_digest") == canonical_digest(ownership, digest_field="ownership_digest")
            and ownership.get("attempt_digest") == prior_attempt["attempt_digest"]
            and ownership.get("status") == "closed_without_resource"
            and ownership.get("active_writer_count") == 0
            and ownership.get("unresolved_create_count") == 0
            and ownership.get("provider_guard") == evidence["provider_guard"]
            and type(ownership.get("observed_at_epoch")) in (int, float)
            and failed_at <= ownership["observed_at_epoch"] <= now
            and now - ownership["observed_at_epoch"] <= 300,
            "scene_recovery_ownership_reconciliation_required")
    return {"prior_attempt_id": prior_attempt["attempt_id"],
            "prior_attempt_digest": prior_attempt["attempt_digest"],
            "failure_digest": failure["failure_digest"], "evidence": evidence}
