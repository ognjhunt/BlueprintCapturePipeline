"""Resume the frozen full schedule through existing admitted native episode runners.

This driver owns retained intent and reconciliation, never provider allocation,
execution authority, scoring or qualification. Quick-10 remains separate.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .exact_workcell_variation_matrix import ExactWorkcellVariationError
from .exact_workcell_variation_runtime import (
    collect_evaluation_evidence,
    validate_matrix_and_schedule,
)


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def safe_execution_blockers(error: Exception) -> list[str]:
    """Retain bounded machine codes, never arbitrary provider error bodies."""
    return sorted(
        {
            code
            for code in getattr(error, "blockers", ())
            if isinstance(code, str)
            and re.fullmatch(r"[a-z][a-z0-9_]*(?::[A-Za-z0-9_.-]{1,100})?", code)
        }
    )[:50]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def artifact_record(root: Path, path: Path) -> dict[str, Any]:
    root, path = root.resolve(), path.resolve()
    if root not in path.parents or not path.is_file():
        raise ExactWorkcellVariationError(["evaluation_artifact_outside_root"])
    body = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": len(body),
        "sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
    }


def _verify_record(root: Path, record: Mapping[str, Any]) -> None:
    artifact = record["artifact"]
    observed = artifact_record(root, root / artifact["relative_path"])
    if observed != artifact:
        raise ExactWorkcellVariationError(["evaluation_retained_episode_bytes_changed"])


def execute_evaluation_schedule(
    *,
    request: Mapping[str, Any],
    schedule_request: Mapping[str, Any],
    matrix: Mapping[str, Any],
    schedule: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    evidence_root: Path,
    run_cell: Callable[..., None],
    assert_execution_allowed: Callable[[], None],
    execution_inputs_digest: str | None = None,
) -> dict[str, Any]:
    """Run every frozen row once; an uncertain effect requires reconciliation.

    ``run_cell`` is the existing provider's isolated native cell executor. It
    calls ``begin(row)`` immediately before each effect and ``commit(row,
    record)`` only after retaining the real episode bytes. The production
    ``NativeCellProcessRunner`` uses native CellRuntime, real control/policy
    episode loops, and a separate Isaac interpreter per cell. The authority
    callback must recheck the caller's existing permission/deadline/spend/hold;
    this function cannot grant or renew any of those.
    """
    validate_matrix_and_schedule(
        request=request, schedule_request=schedule_request, matrix=matrix, schedule=schedule
    )
    required_binding = {
        "job_id",
        "site_id",
        "capture_id",
        "source_capture_digest",
        "source_plan_digest",
        "configuration_digest",
    }
    if (
        set(source_binding) != required_binding
        or any(not isinstance(value, str) or not value for value in source_binding.values())
        or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", source_binding[key]) is None
            for key in ("source_capture_digest", "source_plan_digest", "configuration_digest")
        )
        or any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", source_binding[key]) is None
            for key in ("job_id", "site_id", "capture_id")
        )
        or (
            execution_inputs_digest is not None
            and re.fullmatch(r"sha256:[0-9a-f]{64}", execution_inputs_digest) is None
        )
    ):
        raise ExactWorkcellVariationError(["evaluation_source_binding_invalid"])
    root = evidence_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "source_binding": dict(source_binding),
        "matrix_digest": matrix["matrix_digest"],
        "schedule_digest": schedule["schedule_digest"],
    }
    if execution_inputs_digest is not None:
        identity["execution_inputs_digest"] = execution_inputs_digest
    identity_digest = canonical_digest(identity)
    state_path = root / "exact_workcell_session_state.v1.json"
    with (root / ".exact_workcell_execution.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExactWorkcellVariationError(["evaluation_session_already_owned"]) from exc
        if state_path.exists():
            state = _read(state_path)
            if state.get("identity_digest") != identity_digest or state.get("identity") != identity:
                raise ExactWorkcellVariationError(["evaluation_session_source_or_scope_changed"])
        else:
            state = {
                "schema_version": "exact_workcell_session_state.v1",
                "identity": identity,
                "identity_digest": identity_digest,
                "episodes": {},
                "failure": None,
            }
            _atomic(state_path, state)
        frozen = {row["episode_id"]: row for row in schedule["rows"]}
        if set(state["episodes"]) - set(frozen):
            raise ExactWorkcellVariationError(["evaluation_retained_unscheduled_episode"])
        for episode_id, retained in state["episodes"].items():
            if retained["episode_binding_digest"] != frozen[episode_id]["episode_binding_digest"]:
                raise ExactWorkcellVariationError(["evaluation_retained_binding_changed"])
            if retained["state"] == "completed":
                _verify_record(root, retained["record"])
        # A child may seal its episode and lose the pipe acknowledgement.
        # Adopt its immutable sidecar before deciding an effect is ambiguous.
        for episode_id, retained in list(state["episodes"].items()):
            sidecar = (
                root
                / "receipts"
                / (canonical_digest({"episode_id": episode_id})[7:] + ".record.json")
            )
            if retained["state"] != "completed" and sidecar.exists():
                record = _read(sidecar)
                if (
                    record.get("episode_id") != episode_id
                    or record.get("episode_binding_digest")
                    != frozen[episode_id]["episode_binding_digest"]
                ):
                    raise ExactWorkcellVariationError(
                        ["evaluation_reconciled_receipt_binding_mismatch"]
                    )
                _verify_record(root, record)
                raw = _read(root / record["artifact"]["relative_path"])
                if raw.get("episode_id") != episode_id:
                    raise ExactWorkcellVariationError(
                        ["evaluation_reconciled_episode_identity_mismatch"]
                    )
                state["episodes"][episode_id] = {**retained, "state": "completed", "record": record}
                _atomic(state_path, state)
        unresolved = [
            name for name, row in state["episodes"].items() if row["state"] != "completed"
        ]
        if unresolved:
            # A killed worker, provider timeout or unknown callback cannot buy
            # another episode. Retain the exact uncertain identities for the
            # owning reconciliation path; no automatic reset to pending.
            return {
                "status": "reconciliation_required",
                "episode_ids": sorted(unresolved),
                "identity_digest": identity_digest,
                "paid_execution_retried": False,
            }

        def begin(row: Mapping[str, Any]) -> None:
            episode_id = row["episode_id"]
            if dict(row) != frozen.get(episode_id) or episode_id in state["episodes"]:
                raise ExactWorkcellVariationError(["evaluation_duplicate_or_unbound_begin"])
            assert_execution_allowed()
            state["episodes"][episode_id] = {
                "state": "effect_pending",
                "episode_binding_digest": row["episode_binding_digest"],
            }
            _atomic(state_path, state)

        def commit(row: Mapping[str, Any], record: Mapping[str, Any]) -> None:
            episode_id = row["episode_id"]
            if (
                dict(row) != frozen.get(episode_id)
                or state["episodes"].get(episode_id, {}).get("state") != "effect_pending"
                or record.get("episode_id") != episode_id
                or record.get("episode_binding_digest") != row["episode_binding_digest"]
            ):
                raise ExactWorkcellVariationError(["evaluation_commit_binding_mismatch"])
            _verify_record(root, record)
            receipt = _read(root / record["artifact"]["relative_path"])
            if receipt.get("episode_id") != episode_id:
                raise ExactWorkcellVariationError(["evaluation_episode_receipt_identity_mismatch"])
            state["episodes"][episode_id] = {
                "state": "completed",
                "episode_binding_digest": row["episode_binding_digest"],
                "record": dict(record),
            }
            _atomic(state_path, state)

        try:
            for cell in matrix["cells"]:
                rows = [row for row in schedule["rows"] if row["cell_id"] == cell["cell_id"]]
                remaining = [row for row in rows if row["episode_id"] not in state["episodes"]]
                if not remaining:
                    continue
                assert_execution_allowed()
                run_cell(cell=cell, rows=remaining, begin=begin, commit=commit)
                if any(
                    state["episodes"].get(row["episode_id"], {}).get("state") != "completed"
                    for row in rows
                ):
                    raise ExactWorkcellVariationError(
                        ["evaluation_cell_returned_without_retained_receipts"]
                    )
        except Exception as exc:
            # Keep failure classes only; exception bodies may contain signed
            # video URLs, private paths or provider response content.
            codes = safe_execution_blockers(exc)
            state["failure"] = {
                "type": type(exc).__name__,
                "automatic_retry_allowed": False,
                "blockers": codes,
                "unresolved_episode_ids": sorted(
                    name for name, row in state["episodes"].items() if row["state"] != "completed"
                ),
            }
            _atomic(state_path, state)
            return {
                "status": "blocked" if codes else "interrupted",
                "blockers": codes,
                "identity_digest": identity_digest,
                "completed_episode_count": sum(
                    row["state"] == "completed" for row in state["episodes"].values()
                ),
                "failure_type": type(exc).__name__,
                "paid_execution_retried": False,
            }
        records = [state["episodes"][row["episode_id"]]["record"] for row in schedule["rows"]]
        evidence = collect_evaluation_evidence(
            request=request,
            schedule_request=schedule_request,
            matrix=matrix,
            schedule=schedule,
            episode_records=records,
            evidence_root=root,
        )
        _atomic(root / "exact_workcell_execution_evidence.v1.json", evidence)
        return {
            "status": evidence["status"],
            "identity_digest": identity_digest,
            "completed_episode_count": len(records),
            "evidence": evidence,
            "paid_execution_retried": False,
        }
