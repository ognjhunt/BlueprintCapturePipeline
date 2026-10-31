"""Report each closed World Labs Atlas attempt's outcome to WebApp's ledger.

World Labs exposes no per-operation bill, so an Atlas reservation stays fully
held until its attempt's outcome is known. Through the existing signed
``preparation-settlement`` transport, the same route other providers settle
on, a completed operation settles at the reviewed price it was reserved at,
and a failed operation or a refused submission (which bought nothing) settles
at zero, so its hold stops counting against the job's ceiling. An attempt
whose outcome is unknown, such as a send that may have bought an operation or
an operation that can no longer be found, reports nothing and stays held.

Settlement only records accounting. It never authorizes a purchase, never
raises into the attempt that called it, and a refused or unreachable
settlement leaves the hold in place for the next pass.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

PROVIDER = "world_labs_atlas"
RESOURCE_CLASS = "provider_reconstruction_api"
SETTLEMENT_FILENAME = "world_labs_settlement.json"
#: Retry attempts nest one level per resubmission; a few more than the task cap.
MAX_RETRY_DEPTH = 8

_LOG = logging.getLogger(__name__)


def _read(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def atlas_attempt_outcome(state: Mapping[str, Any], *, closed: bool = False) -> dict[str, Any] | None:
    """The settled outcome of one attempt's retained provider record, or None while unknown.

    ``closed`` marks an attempt that can never send again (an archived pose
    attempt, or any task attempt), so even a refusal that was retryable is final.
    """
    operation_id = state.get("operation_id")
    terminal = state.get("terminal_operation")
    # The operation itself finished with an error; its id is the one this hold bought.
    # A task whose retry later completed keeps this failure in its own record.
    if (isinstance(terminal, Mapping) and terminal.get("done") is True and terminal.get("error") is not None
            and isinstance(operation_id, str) and terminal.get("id") == operation_id):
        return {"outcome": "failed", "operation_id": operation_id, "outcome_receipt_digest": canonical_digest(terminal)}
    if (state.get("status") == "completed" and isinstance(operation_id, str)
            and isinstance(state.get("result_digest"), str)):
        return {"outcome": "completed", "operation_id": operation_id, "outcome_receipt_digest": state["result_digest"]}
    if state.get("status") == "rejected" and not operation_id and not state.get("possibly_bought"):
        rejection = state.get("provider_rejection")
        from .website_atlas_pose import RESEND_LIMIT
        if isinstance(rejection, Mapping) and (closed or not rejection.get("retryable")
                                                or state.get("resends", 0) >= RESEND_LIMIT):
            return {"outcome": "refused", "operation_id": None, "outcome_receipt_digest": canonical_digest({
                "request_digest": state.get("request_digest"), "provider_rejection": dict(rejection),
                "send_statuses": [row.get("status") for row in state.get("sends") or [] if isinstance(row, Mapping)]})}
    return None


def settle_closed_atlas_attempt(*, state_dir: Path, admission_path: Path, task_context: Mapping[str, Any],
                                closed: bool = False) -> dict[str, Any] | None:
    """Settle one attempt once; returns the WebApp receipt, or None when nothing was settled."""
    record_path = state_dir / SETTLEMENT_FILENAME
    retained = _read(record_path)
    if retained is not None:
        return retained
    state, admission = _read(state_dir / "submission.json"), _read(admission_path)
    if state is None or admission is None:
        return None
    outcome = atlas_attempt_outcome(state, closed=closed)
    binding = admission.get("allocation_binding_digest")
    if (outcome is None or admission.get("provider") != PROVIDER or admission.get("resource_class") != RESOURCE_CLASS
            or not isinstance(binding, str) or binding != state.get("request_digest")
            or not isinstance(admission.get("task_context_digest"), str)):
        return None
    # The reservation's own task digest: an Atlas hold survives a later task edit.
    command = {"task_context_digest": admission["task_context_digest"], "allocation_binding_digest": binding,
               "provider": PROVIDER, **outcome}
    from .website_atlas_pose import _write_state
    from .website_task_context import preparation_webapp_request
    try:
        receipt = preparation_webapp_request(task_context=task_context, operation="preparation-settlement",
            payload={"request_id": task_context.get("request_id"), "scene_id": task_context.get("scene_id"),
                     "settlement": command})
        if receipt.get("status") != "settled" or any(receipt.get(key) != value for key, value in command.items()):
            raise ValueError("atlas_settlement_receipt_invalid")
        _write_state(record_path, receipt)
    except Exception as exc:  # noqa: BLE001 - accounting only: never changes the attempt's own outcome
        # The hold stays in place and a later pass reports it again.
        _LOG.warning("atlas_settlement_deferred binding=%s outcome=%s reason=%s",
                     binding, outcome["outcome"], str(exc)[:200])
        return None
    return receipt


def settle_closed_atlas_task_attempts(root: Path, *, task_context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A task attempt and each retry it started (``root/retry/...``), each with its own reservation."""
    settled, node = [], Path(root)
    for _ in range(MAX_RETRY_DEPTH):
        if not (node / "atlas_pose" / "submission.json").is_file():
            break
        receipt = settle_closed_atlas_attempt(state_dir=node / "atlas_pose",
            admission_path=node / "controller_admission.json", task_context=task_context, closed=True)
        if receipt is not None:
            settled.append(receipt)
        node = node / "retry"
    return settled


def settle_closed_atlas_pose_attempts(output_root: Path, *, task_context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The current pose attempt and every attempt set aside under ``atlas_pose_attempts/``."""
    settled = []
    rows = [(Path(output_root) / "atlas_pose", False)]
    archived = Path(output_root) / "atlas_pose_attempts"
    if archived.is_dir() and not archived.is_symlink():
        rows += [(path, True) for path in sorted(archived.iterdir()) if path.is_dir() and not path.is_symlink()]
    for state_dir, closed in rows:
        receipt = settle_closed_atlas_attempt(state_dir=state_dir, admission_path=state_dir / "controller_admission.json",
                                              task_context=task_context, closed=closed)
        if receipt is not None:
            settled.append(receipt)
    return settled


def settle_closed_atlas_attempts_under(root: Path, *, task_context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every Atlas attempt retained anywhere under one job's output root.

    Run before the job admits anything new, so a hold World Labs never charged
    (capture-447ae4f4's failed splat) stops counting against the job's ceiling
    even when the step that bought it is not revisited.
    """
    settled, base = [], Path(root).resolve()
    if not base.is_dir():
        return settled
    for path in sorted(base.rglob("submission.json")):
        state_dir = path.parent
        if path.is_symlink() or not path.resolve().is_relative_to(base):
            continue
        if state_dir.name == "atlas_pose":
            inside = state_dir / "controller_admission.json"
            # A pose keeps its admission beside its state and may resend a credit refusal;
            # a task keeps it one level up, and its refusals are final.
            receipt = (settle_closed_atlas_attempt(state_dir=state_dir, admission_path=inside, task_context=task_context)
                       if inside.is_file() else
                       settle_closed_atlas_attempt(state_dir=state_dir, admission_path=state_dir.parent
                                                   / "controller_admission.json", task_context=task_context, closed=True))
        elif state_dir.parent.name == "atlas_pose_attempts":
            receipt = settle_closed_atlas_attempt(state_dir=state_dir, admission_path=state_dir / "controller_admission.json",
                                                  task_context=task_context, closed=True)
        else:
            continue
        if receipt is not None:
            settled.append(receipt)
    return settled
