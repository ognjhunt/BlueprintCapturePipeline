"""Settle the current attempt of a website scene intent whose authority ended.

Moved out of ``task_evaluation_terminal_scene_attempt_settlement`` (2026-11-02,
beta tracker E-06). Before the settlement it can record, it must reconcile global
provider ownership (``task_evaluation_scene_progression_recovery.reconcile_ownership``),
which reads the GPU provider inventory. The control-plane intake service imports
the terminal-settlement module (for ``budget_retained_hold``), and
``tests/test_live_pipeline_import_isolation.py`` refuses any import reachable
from it that reaches the hot lane, including one nested in a function. Only scene
progression calls this, so the ownership reconciliation lives here, where the
intake never loads it. Re-exporting it from the settlement module would recreate
the edge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_retained_controls_evidence import _file, _read, validated_cancellation
from .task_evaluation_terminal_scene_attempt_settlement import (
    AUTHORITY_ENDED_SCHEMA,
    OWNERSHIP_RETRY_SECONDS,
    OWNERSHIP_SCHEMA,
    _launch_state,
    _require,
    _sealed,
    launch_id_for_preparation,
    settle_retired_attempt_rows,
    unallocated_launch_evidence,
)


def _retained_ownership(root: Path, attempt_digest: str) -> tuple[Path | None, float | None]:
    """A closed ownership reconciliation already retained for this retirement, and the newest one's time."""
    closed, latest = None, None
    for path in sorted(root.glob("*.json")) if root.is_dir() else ():
        if path.name.endswith("-guard.json"):
            continue
        try:
            value = _sealed(path, "ownership_digest")
        except (OSError, ValueError):
            continue
        observed = value.get("observed_at_epoch")
        if value.get("schema_version") != OWNERSHIP_SCHEMA or value.get("attempt_digest") != attempt_digest \
                or isinstance(observed, bool) or not isinstance(observed, (int, float)):
            continue
        latest = observed if latest is None else max(latest, observed)
        if (value.get("status") == "closed_without_resource" and value.get("active_writer_count") == 0
                and value.get("unresolved_create_count") == 0):
            closed = path
    return closed, latest


def settle_ended_authority_attempt(*, directory: Path, intent: Mapping[str, Any], state: Mapping[str, Any],
                                   config: Mapping[str, Any], now: float) -> dict[str, Any] | None:
    """Settle the current attempt of an intent whose authority ended, if it provably never allocated.

    Progression stops at the authority gate once ``revoked.json`` exists or the
    execution window closes, and it only retires an attempt when it starts a
    successor, so the current attempt's holds stayed held for good
    (capture-447ae4f4, 2026-11-02: $6.01 after a $0 disk refusal at 13:39Z,
    then revocation at 14:08Z). This settles them as any retired attempt is
    settled, under a retirement record naming why: the intent's authority
    ended and the attempt's only launch is terminal with no allocation.

    Accounting only: it never launches, retries or reserves. Returns ``None``
    when there is nothing to settle, a waiting status while global ownership is
    not yet reconciled, else ``settle_retired_attempt_rows``'s outcome. Raises
    ``ValueError`` when the evidence is missing, changed or may show spend.
    """
    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
    from .task_evaluation_scene_preparation_identity import SCHEMA as PREPARATION_SCHEMA

    directory = Path(directory)
    revocation = directory / "revoked.json"
    expiry = intake.effective_execution_expiry(directory, intent)
    _require(revocation.exists() or now >= expiry, "authority_not_ended")
    attempt_ref, factory_ref, activation_ref = state.get("attempt"), state.get("factory"), state.get("activation_link")
    if not (isinstance(attempt_ref, Mapping) and isinstance(factory_ref, Mapping)
            and isinstance(activation_ref, Mapping)):
        return None  # No current attempt with an issued activation: nothing was reserved for it.
    attempt = intake._read(checked_file(attempt_ref["path"], attempt_ref), "attempt_digest")
    if attempt.get("schema_version") != PREPARATION_SCHEMA:
        return None  # A paid source attempt keeps its own hold; only website preparations retire here.
    _require(attempt.get("maximum_spend_usd") == 0 and attempt.get("paid_authority_granted") is False
             and attempt.get("intent_digest") == intent["intent_digest"], "ended_attempt_invalid")
    activation = read(checked_file(activation_ref["path"], activation_ref), digest_field="link_digest")
    row_ref = activation["scene_configuration_attempt"]
    row = intake._read(checked_file(row_ref["path"], row_ref), "attempt_digest")
    if validated_cancellation(directory, row) is not None:
        return None  # Already settled.
    factory = read(checked_file(factory_ref["path"], factory_ref), digest_field="factory_digest")
    request = read(checked_file(factory["submission_request"]["path"], factory["submission_request"]))
    _require(factory.get("schema_version") == "website_scene_attempt_factory.v1"
             and factory.get("attempt_digest") == attempt["attempt_digest"]
             and factory.get("intent_digest") == intent["intent_digest"] == row.get("intent_digest")
             and factory.get("source_commit") == attempt["source_commit"] == row.get("source_commit")
             and canonical_digest(request) == row.get("input_digest") == activation.get("request_digest")
             and request.get("preparation_id") == activation.get("preparation_id"), "ended_attempt_binding_invalid")
    launch_id = launch_id_for_preparation(str(request["preparation_id"]))
    execution = _launch_state(launch_id=launch_id, launch_execution_root=Path(config["launch_execution_root"]),
                              launch_queue_root=Path(config["launch_queue_root"]))
    # Only a launch with a terminal receipt: one never queued may still be in transit.
    _require(execution is not None and execution.get("launch_receipt") is not None
             and execution.get("launch_status") == "blocked", "ended_attempt_launch_not_terminal")
    launch = _read(Path(execution["launch_receipt"]["path"]))
    refusal = unallocated_launch_evidence(launch=launch, request=request, source_commit=row["source_commit"])
    _require(refusal is not None, "ended_attempt_launch_may_have_allocated")

    output = Path(config["factory_output_root"]) / intent["intent_id"] / attempt["attempt_id"]
    retirement_path = output / "authority-ended-retirement.json"
    if retirement_path.exists():
        retirement = _sealed(retirement_path, "failure_digest")
        _require(retirement.get("schema_version") == AUTHORITY_ENDED_SCHEMA
                 and retirement.get("attempt_digest") == attempt["attempt_digest"]
                 and retirement.get("launch_receipt") == execution["launch_receipt"], "ended_retirement_changed")
    else:
        retirement = {
            "schema_version": AUTHORITY_ENDED_SCHEMA, "status": "retired_after_authority_ended",
            "intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
            "attempt_id": attempt["attempt_id"], "attempt_digest": attempt["attempt_digest"],
            "authority": {"revocation": _file(revocation) if revocation.exists() else None,
                          "effective_execution_expiry_epoch": expiry},
            "configuration_attempt": dict(row_ref), "launch_id": launch_id,
            "launch_receipt": execution["launch_receipt"], "unallocated_launch_evidence": refusal,
            "provider_allocation_performed": False, "new_execution_authorized": False,
            "observed_at_epoch": now,
        }
        retirement["failure_digest"] = canonical_digest(retirement, digest_field="failure_digest")
        output.mkdir(parents=True, exist_ok=True, mode=0o750)
        intake.write_exclusive(retirement_path, retirement)
    # The same global reconciliation every retired attempt needs before its rows settle.
    reconciliations = output / "authority-ended-reconciliation"
    ownership, latest = _retained_ownership(reconciliations, attempt["attempt_digest"])
    if ownership is None:
        if latest is not None and now - latest < OWNERSHIP_RETRY_SECONDS:
            return {"status": "waiting_for_ownership_reconciliation", "retired_attempt_id": attempt["attempt_id"]}
        from .task_evaluation_scene_progression_recovery import reconcile_ownership
        ownership = Path(reconcile_ownership(attempt=attempt, failure_path=retirement_path, config=config,
                                             output_root=reconciliations, now=now,
                                             execution_attempt=row)["ownership_reconciliation"]["path"])
    return settle_retired_attempt_rows(
        directory=directory, retired_attempt=attempt, retirement_record={"path": str(retirement_path)},
        ownership_record={"path": str(ownership)}, launch_execution_root=Path(config["launch_execution_root"]),
        launch_queue_root=Path(config["launch_queue_root"]), source_factory=factory_ref)


__all__ = ["settle_ended_authority_attempt"]
