"""Settle every reservation row of a retired scene attempt once nothing can spend it.

A scene attempt reserves up to five rows: its own source row, one
scene-configuration row keyed by the preparation request digest, and three
controls rows. ``reserve_scene_attempt`` counts every non-cancelled row at its
full hold against the owner's spend cap and the paid-attempt count, so an
intent that retries fifteen times holds ~$150 while having spent a few dollars
(InteriorGS 840938, 2026-09-13 06:44Z: 31 rows, $148.38 held, cap exhausted).

The progression already proves an attempt terminal before minting a successor:
a sealed release transition or failure record plus a global ownership
reconciliation (provider zero, no leases, no pending teardowns). This module
turns that proof into a settlement receipt per row, honoured by
``validated_cancellation`` exactly like an unstarted-controls cancellation.
Dependent rows settle only when their launch is terminal or was never queued.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_retained_controls_evidence import DIRECTORY, _file, _read, validated_cancellation

SCHEMA = "task_evaluation_terminal_scene_attempt_settlement.v1"
STATUS = "settled_after_terminal_attempt"
RETIREMENT_SCHEMAS = {
    "task_evaluation_scene_release_transition.v1",
    "task_evaluation_scene_attempt_failure.v1",
}
OWNERSHIP_SCHEMA = "task_evaluation_scene_attempt_ownership.v1"
TERMINAL_LAUNCH_STATUSES = {"blocked", "completed", "failed"}
CONTROLS_PHASES = ("construction", "controls", "placement")
_ATTEMPT_FIELDS = ("attempt_id", "attempt_digest", "intent_digest", "provider", "maximum_spend_usd", "source_commit")
_PREPARATION_SUFFIX = "-scene-configuration-preparation"
_ACTIVATION_SUFFIX = "-scene-configuration-activation-auto"


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise ValueError("terminal_settlement_" + code)


def _sealed(path: Path, field: str) -> dict[str, Any]:
    value = _read(path)
    _require(value.get(field) == canonical_digest(value, digest_field=field), "record_seal_invalid")
    return value


def launch_id_for_preparation(preparation_id: str) -> str:
    _require(preparation_id.endswith(_PREPARATION_SUFFIX), "preparation_id_invalid")
    return preparation_id[: -len(_PREPARATION_SUFFIX)] + _ACTIVATION_SUFFIX + "-launch"


def dependent_row_ids(request_digest: str) -> dict[str, str]:
    _require(re.fullmatch(r"sha256:[0-9a-f]{64}", str(request_digest)) is not None, "request_digest_invalid")
    stem = "controls-" + request_digest[7:47]
    rows = {"scene-configuration-" + request_digest[7:31]: "scene_configuration"}
    rows.update({f"{stem}-{phase}": phase for phase in CONTROLS_PHASES})
    return rows


def validate_terminal_settlement(*, receipt: Mapping[str, Any], attempt: Mapping[str, Any]) -> None:
    """Fail closed unless the settlement still binds live, byte-identical evidence."""
    _require(
        receipt.get("schema_version") == SCHEMA
        and receipt.get("status") == STATUS
        and receipt.get("receipt_digest") == canonical_digest(receipt, digest_field="receipt_digest")
        and all(receipt.get(k) == attempt.get(k) for k in _ATTEMPT_FIELDS)
        and receipt.get("provider_mutation_performed") is False
        and receipt.get("downstream_execution_eligible") is False,
        "receipt_invalid",
    )
    retired = receipt.get("retired_attempt") or {}
    retirement_ref = receipt.get("retirement_record") or {}
    ownership_ref = receipt.get("ownership_reconciliation") or {}
    _require(_file(Path(str(retirement_ref.get("path") or ""))) == retirement_ref, "retirement_record_changed")
    _require(_file(Path(str(ownership_ref.get("path") or ""))) == ownership_ref, "ownership_record_changed")
    retirement = _sealed(Path(retirement_ref["path"]), "failure_digest")
    ownership = _sealed(Path(ownership_ref["path"]), "ownership_digest")
    _require(
        retirement.get("schema_version") in RETIREMENT_SCHEMAS
        and retirement.get("attempt_digest") == retired.get("attempt_digest")
        and ownership.get("schema_version") == OWNERSHIP_SCHEMA
        and ownership.get("status") == "closed_without_resource"
        and ownership.get("attempt_digest") == retired.get("attempt_digest")
        and ownership.get("active_writer_count") == 0
        and ownership.get("unresolved_create_count") == 0,
        "terminal_evidence_invalid",
    )
    if attempt["attempt_id"] == retired.get("attempt_id"):
        _require(receipt.get("dependency") is None and receipt.get("execution_terminal") is None, "source_row_shape_invalid")
        return
    dependency = receipt.get("dependency") or {}
    link_ref = dependency.get("preparation_link") or {}
    _require(_file(Path(str(link_ref.get("path") or ""))) == link_ref, "preparation_link_changed")
    link = _sealed(Path(link_ref["path"]), "link_digest")
    rows = dependent_row_ids(str(link.get("request_digest")))
    _require(
        attempt["attempt_id"] in rows
        and ("-" + str(retired.get("attempt_id")) + "-") in str(link.get("preparation_id"))
        and (rows[attempt["attempt_id"]] != "scene_configuration" or attempt.get("input_digest") == link["request_digest"]),
        "dependency_binding_invalid",
    )
    execution = receipt.get("execution_terminal") or {}
    launch_id = launch_id_for_preparation(str(link["preparation_id"]))
    _require(execution.get("launch_id") == launch_id, "execution_launch_id_invalid")
    launch_ref = execution.get("launch_receipt")
    if launch_ref is not None:
        _require(_file(Path(str(launch_ref.get("path") or ""))) == launch_ref, "launch_receipt_changed")
        launch = _read(Path(launch_ref["path"]))
        _require(
            launch.get("launch_id") == launch_id and launch.get("status") in TERMINAL_LAUNCH_STATUSES,
            "launch_not_terminal",
        )
    else:
        _require(execution.get("launch_never_queued") is True, "execution_evidence_missing")


def _launch_state(*, launch_id: str, launch_execution_root: Path, launch_queue_root: Path) -> dict[str, Any] | None:
    """Terminal evidence for a dependent row, or None while a launch may still spend."""
    receipt_path = launch_execution_root / launch_id / "launch_receipt.json"
    if receipt_path.is_file():
        launch = _read(receipt_path)
        if launch.get("launch_id") != launch_id or launch.get("status") not in TERMINAL_LAUNCH_STATUSES:
            return None
        return {"launch_id": launch_id, "launch_receipt": _file(receipt_path), "launch_never_queued": False}
    if (launch_execution_root / launch_id).exists():
        return None  # A launch directory without a terminal receipt is still running or unreconciled.
    for state in ("pending", "processing"):
        if any(path.name.startswith(launch_id) for path in (launch_queue_root / state).glob("*.json")):
            return None
    return {"launch_id": launch_id, "launch_receipt": None, "launch_never_queued": True}


def settle_retired_attempt_rows(*, directory: Path, retired_attempt: Mapping[str, Any],
        retirement_record: Mapping[str, Any], ownership_record: Mapping[str, Any],
        launch_execution_root: Path, launch_queue_root: Path, dry_run: bool = False) -> dict[str, Any]:
    """Write settlement receipts for the retired attempt's rows; idempotent and lock-free by design."""
    from . import task_evaluation_scene_intake as intake

    directory = Path(directory)
    retirement_ref = _file(Path(str(retirement_record["path"])))
    ownership_ref = _file(Path(str(ownership_record["path"])))
    retired = {"attempt_id": retired_attempt["attempt_id"], "attempt_digest": retired_attempt["attempt_digest"]}
    settled, skipped = [], []

    def settle(attempt: Mapping[str, Any], *, dependency: dict[str, Any] | None, execution: dict[str, Any] | None) -> None:
        existing = validated_cancellation(directory, attempt)
        if existing is not None:
            settled.append({"attempt_id": attempt["attempt_id"], "status": "already_released", "schema_version": existing.get("schema_version")})
            return
        receipt = {
            "schema_version": SCHEMA, "status": STATUS,
            **{k: attempt[k] for k in _ATTEMPT_FIELDS},
            "retired_attempt": retired, "retirement_record": retirement_ref,
            "ownership_reconciliation": ownership_ref, "dependency": dependency,
            "execution_terminal": execution, "downstream_execution_eligible": False,
            "provider_mutation_performed": False,
        }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        validate_terminal_settlement(receipt=receipt, attempt=attempt)
        if not dry_run:
            target = directory / DIRECTORY / (attempt["attempt_id"] + ".json")
            target.parent.mkdir(mode=0o750, exist_ok=True)
            intake.write_exclusive(target, receipt)
        settled.append({"attempt_id": attempt["attempt_id"], "status": "settled"})

    source_path = directory / "attempts" / (retired["attempt_id"] + ".json")
    source_row = intake._read(source_path, "attempt_digest")
    _require(source_row["attempt_digest"] == retired["attempt_digest"], "retired_attempt_digest_mismatch")
    settle(source_row, dependency=None, execution=None)
    marker = "-" + retired["attempt_id"] + "-"
    for link_path in sorted((directory / "preparations").glob("*.json")):
        if link_path.name.endswith(".activation.json"):
            continue
        link = _read(link_path)
        if link.get("schema_version") != "task_evaluation_scene_preparation_link.v1" or marker not in str(link.get("preparation_id", "")):
            continue
        if link.get("link_digest") != canonical_digest(link, digest_field="link_digest"):
            skipped.append({"preparation_link": link_path.name, "reason": "link_seal_invalid"})
            continue
        launch_id = launch_id_for_preparation(str(link["preparation_id"]))
        execution = _launch_state(launch_id=launch_id, launch_execution_root=Path(launch_execution_root),
                                  launch_queue_root=Path(launch_queue_root))
        if execution is None:
            skipped.append({"preparation_link": link_path.name, "reason": "launch_not_terminal", "launch_id": launch_id})
            continue
        dependency = {"preparation_link": _file(link_path), "request_digest": link["request_digest"]}
        for row_id in dependent_row_ids(str(link["request_digest"])):
            row_path = directory / "attempts" / (row_id + ".json")
            if not row_path.is_file():
                continue
            settle(intake._read(row_path, "attempt_digest"), dependency=dependency, execution=execution)
    return {"status": "would_settle" if dry_run else "settled", "retired_attempt_id": retired["attempt_id"],
            "rows": settled, "skipped": skipped, "provider_mutation_performed": False}


def sweep_retired_attempts(*, directory: Path, state: Mapping[str, Any], config: Mapping[str, Any],
                           dry_run: bool = False) -> dict[str, Any]:
    """Settle rows for every predecessor the progression already retired; never raises."""
    directory = Path(directory)
    summary: dict[str, Any] = {"settled_rows": 0, "already_released_rows": 0, "skipped": []}
    entries = []
    for lineage in state.get("release_predecessors") or []:
        entries.append((lineage.get("attempt"), lineage.get("reconciliation")) if isinstance(lineage, Mapping) else (None, None))
    for lineage in state.get("recovery_predecessors") or []:
        entries.append((lineage.get("attempt"), lineage.get("evidence")) if isinstance(lineage, Mapping) else (None, None))
    for attempt_ref, evidence in entries:
        try:
            if not isinstance(attempt_ref, Mapping) or not isinstance(evidence, Mapping):
                raise ValueError("terminal_settlement_lineage_shape_invalid")
            from . import task_evaluation_scene_intake as intake
            retired = intake._read(Path(str(attempt_ref["path"])), "attempt_digest")
            outcome = settle_retired_attempt_rows(
                directory=directory, retired_attempt=retired,
                retirement_record=evidence["failure"], ownership_record=evidence["ownership_reconciliation"],
                launch_execution_root=Path(config["launch_execution_root"]),
                launch_queue_root=Path(config["launch_queue_root"]), dry_run=dry_run)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            summary["skipped"].append({"attempt": str((attempt_ref or {}).get("path", ""))[-60:], "reason": str(exc)[:120]})
            continue
        summary["settled_rows"] += sum(1 for row in outcome["rows"] if row["status"] == "settled")
        summary["already_released_rows"] += sum(1 for row in outcome["rows"] if row["status"] == "already_released")
        summary["skipped"].extend(outcome["skipped"])
    summary["skipped"] = sorted({json.dumps(row, sort_keys=True) for row in summary["skipped"]})
    summary["skipped"] = [json.loads(row) for row in summary["skipped"]]
    summary["summary_digest"] = "sha256:" + hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()
    return summary


def settlement_releases_budget(receipt: Mapping[str, Any]) -> bool:
    """Only dependent work proven never queued has a zero-cost settlement.

    Completed/failed/blocked executions retain their conservative reservation
    until a separate billing reconciliation can establish incurred spend. Legacy
    v1 receipts are interpreted the same way; no historical file is rewritten.
    A source parent being idle is not proof that its paid children cost zero.
    """
    execution = receipt.get("execution_terminal") or {}
    return (receipt.get("schema_version") == SCHEMA
            and execution.get("launch_never_queued") is True
            and execution.get("launch_receipt") is None)
