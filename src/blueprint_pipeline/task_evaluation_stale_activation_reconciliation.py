"""Fail-closed recovery for one crash-stranded activation queue claim.

The activation worker claims a request by moving its immutable envelope from
``pending`` to ``processing``.  A host/process interruption before result
publication historically left that envelope live forever.  Release retention
then correctly protected the source commit even after a later sibling
activation for the exact preparation had terminated.

This module does not infer closure from age.  It requires a digest-valid
terminal sibling for the same preparation, proves the stranded activation has
no activation root or downstream reference, binds a fresh all-provider zero
report, and takes the same per-envelope lease as the worker before moving the
original envelope to ``blocked``.  All queue/identity/result evidence remains.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_activation_queue import (
    ENVELOPE_SCHEMA_VERSION,
    IDENTITY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
)
from .task_evaluation_launch_preparation_queue import (
    write_launch_preparation_record_exclusive,
)


PLAN_SCHEMA_VERSION = "task_evaluation_stale_activation_reconciliation_plan.v1"
RECEIPT_SCHEMA_VERSION = (
    "task_evaluation_stale_activation_reconciliation_receipt.v1"
)
APPLY_ACKNOWLEDGEMENT = "reconcile-stale-activation-processing-envelope"
DEFAULT_MAX_PROVIDER_ZERO_AGE_SECONDS = 300
DEFAULT_MINIMUM_PROCESSING_AGE_SECONDS = 60 * 60
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")


class StaleActivationReconciliationError(ValueError):
    """The processing activation could not be proven terminal and inert."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _absolute(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise StaleActivationReconciliationError(
            f"stale_activation_reconciliation_{field}_must_be_absolute"
        )
    return path


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _file_snapshot(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        info = path.lstat()
        payload = path.read_bytes()
    except OSError as exc:
        raise StaleActivationReconciliationError(blocker) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise StaleActivationReconciliationError(blocker)
    return {
        "path": str(path),
        "sha256": _sha256_bytes(payload),
        "size_bytes": int(info.st_size),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mtime_ns": int(info.st_mtime_ns),
    }


def _read_sealed(
    path: Path, *, schema_version: str, digest_field: str, blocker: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = _file_snapshot(path, blocker=blocker)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StaleActivationReconciliationError(blocker) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != schema_version
        or value.get(digest_field)
        != canonical_digest(value, digest_field=digest_field)
    ):
        raise StaleActivationReconciliationError(blocker)
    return dict(value), snapshot


def _parse_time(value: Any, *, blocker: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise StaleActivationReconciliationError(blocker) from exc
    if parsed.tzinfo is None:
        raise StaleActivationReconciliationError(blocker)
    return parsed.astimezone(timezone.utc)


def _validate_provider_zero(
    path: Path,
    *,
    now: datetime,
    maximum_age_seconds: int,
) -> dict[str, Any]:
    snapshot = _file_snapshot(
        path, blocker="stale_activation_reconciliation_provider_zero_invalid"
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_provider_zero_invalid"
        ) from exc
    provider_zero = value.get("provider_zero") if isinstance(value, Mapping) else None
    inventories = value.get("inventory_results") if isinstance(value, Mapping) else None
    generated_at = _parse_time(
        value.get("generated_at") if isinstance(value, Mapping) else None,
        blocker="stale_activation_reconciliation_provider_zero_invalid",
    )
    age_seconds = (now - generated_at).total_seconds()
    required_inventory_providers = {
        str(row.get("provider"))
        for row in inventories or []
        if isinstance(row, Mapping) and row.get("required") is True
    }
    if (
        value.get("schema_version") != "gpu_spend_guard.v1"
        or value.get("status") != "passed"
        or value.get("blockers") != []
        or value.get("provider_zero_verified") is not True
        or value.get("live_instance_count") != 0
        or value.get("total_burn_per_hour_usd") != 0
        or not isinstance(provider_zero, Mapping)
        or provider_zero.get("status") != "verified"
        or provider_zero.get("blockers") != []
        or provider_zero.get("global_live_instance_count") != 0
        or provider_zero.get("global_total_burn_per_hour_usd") != 0
        or set(provider_zero.get("required_provider_ids") or [])
        != required_inventory_providers
        or "vast" not in required_inventory_providers
        or not isinstance(inventories, list)
        or not inventories
        or any(
            not isinstance(row, Mapping)
            or (row.get("required") is True and row.get("status") != "succeeded")
            or (row.get("required") is True and row.get("row_count") != 0)
            or (row.get("required") is True and row.get("blockers") != [])
            for row in inventories
        )
        or age_seconds < 0
        or age_seconds > maximum_age_seconds
    ):
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_provider_zero_invalid"
        )
    return {
        **snapshot,
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "age_seconds": age_seconds,
        "required_provider_ids": sorted(
            str(row.get("provider"))
            for row in inventories
            if isinstance(row, Mapping) and row.get("required") is True
        ),
        "provider_zero_verified": True,
    }


def _request_binding(request: Mapping[str, Any]) -> dict[str, Any]:
    preparation = request.get("preparation")
    if not isinstance(preparation, Mapping):
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_request_binding_invalid"
        )
    binding = {
        "team_namespace": request.get("team_namespace"),
        "lane": request.get("lane"),
        "expected_production_commit": request.get("expected_production_commit"),
        "preparation_id": preparation.get("preparation_id"),
        "preparation_request_digest": preparation.get("request_digest"),
        "preparation_result_digest": preparation.get("result_digest"),
    }
    if (
        not all(isinstance(value, str) and value for value in binding.values())
        or re.fullmatch(
            r"[0-9a-f]{40}", str(binding["expected_production_commit"])
        )
        is None
        or not _DIGEST_RE.fullmatch(str(binding["preparation_request_digest"]))
        or not _DIGEST_RE.fullmatch(str(binding["preparation_result_digest"]))
    ):
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_request_binding_invalid"
        )
    return binding


def _reference_hits(
    *, roots: Sequence[Path], needles: Sequence[str], excluded: set[Path]
) -> list[str]:
    hits: list[str] = []
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise StaleActivationReconciliationError(
                "stale_activation_reconciliation_reference_root_invalid"
            )
        for path in sorted(root.rglob("*.json")):
            if path.is_symlink() or not path.is_file():
                raise StaleActivationReconciliationError(
                    "stale_activation_reconciliation_reference_invalid"
                )
            resolved = path.resolve()
            if resolved in excluded:
                continue
            try:
                encoded = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise StaleActivationReconciliationError(
                    "stale_activation_reconciliation_reference_invalid"
                ) from exc
            if any(needle in encoded for needle in needles):
                hits.append(str(path))
    return hits


def build_stale_activation_reconciliation_plan(
    *,
    target_envelope: str | Path,
    terminal_sibling_envelope: str | Path,
    activation_queue_root: str | Path,
    activation_base_root: str | Path,
    reference_roots: Sequence[str | Path],
    provider_zero_report: str | Path,
    now: datetime | None = None,
    minimum_processing_age_seconds: int = DEFAULT_MINIMUM_PROCESSING_AGE_SECONDS,
    maximum_provider_zero_age_seconds: int = DEFAULT_MAX_PROVIDER_ZERO_AGE_SECONDS,
) -> dict[str, Any]:
    """Build a non-mutating plan for one exact processing orphan."""

    if (
        not isinstance(minimum_processing_age_seconds, int)
        or isinstance(minimum_processing_age_seconds, bool)
        or minimum_processing_age_seconds < 0
        or not isinstance(maximum_provider_zero_age_seconds, int)
        or isinstance(maximum_provider_zero_age_seconds, bool)
        or maximum_provider_zero_age_seconds <= 0
    ):
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_age_policy_invalid"
        )
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    queue = _absolute(activation_queue_root, field="activation_queue_root").resolve()
    base = _absolute(activation_base_root, field="activation_base_root").resolve()
    target_path = _absolute(target_envelope, field="target_envelope").resolve()
    sibling_path = _absolute(
        terminal_sibling_envelope, field="terminal_sibling_envelope"
    ).resolve()
    if queue.is_symlink() or not queue.is_dir() or base.is_symlink() or not base.is_dir():
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_managed_root_invalid"
        )
    if target_path.parent != queue / "processing" or sibling_path.parent != queue / "blocked":
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_queue_state_invalid"
        )
    target, target_snapshot = _read_sealed(
        target_path,
        schema_version=ENVELOPE_SCHEMA_VERSION,
        digest_field="envelope_digest",
        blocker="stale_activation_reconciliation_target_invalid",
    )
    sibling, sibling_snapshot = _read_sealed(
        sibling_path,
        schema_version=ENVELOPE_SCHEMA_VERSION,
        digest_field="envelope_digest",
        blocker="stale_activation_reconciliation_terminal_sibling_invalid",
    )
    target_request = target.get("request")
    sibling_request = sibling.get("request")
    if not isinstance(target_request, Mapping) or not isinstance(sibling_request, Mapping):
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_request_binding_invalid"
        )
    target_activation_id = str(target_request.get("activation_id") or "")
    sibling_activation_id = str(sibling_request.get("activation_id") or "")
    if (
        not target_activation_id
        or not sibling_activation_id
        or target_activation_id == sibling_activation_id
        or _request_binding(target_request) != _request_binding(sibling_request)
        or target.get("provider_mutation_performed_inside_intake") is not False
        or target.get("catalog_mutation_performed_inside_intake") is not False
        or target.get("standing_authorization_published_inside_intake") is not False
        or target.get("paid_execution_requested") is not False
        or sibling.get("provider_mutation_performed_inside_intake") is not False
        or sibling.get("catalog_mutation_performed_inside_intake") is not False
        or sibling.get("standing_authorization_published_inside_intake") is not False
        or sibling.get("paid_execution_requested") is not False
    ):
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_sibling_binding_mismatch"
        )
    expected_target_name = (
        f"{target_activation_id}-{str(target['request_digest']).removeprefix('sha256:')}.json"
    )
    if target_path.name != expected_target_name:
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_target_invalid"
        )
    sibling_result_path = queue / "results" / sibling_path.name
    sibling_result, sibling_result_snapshot = _read_sealed(
        sibling_result_path,
        schema_version=RESULT_SCHEMA_VERSION,
        digest_field="result_digest",
        blocker="stale_activation_reconciliation_terminal_result_invalid",
    )
    if (
        sibling_result.get("activation_id") != sibling_activation_id
        or sibling_result.get("status") != "blocked"
        or sibling_result.get("provider_mutation_performed") is not False
        or sibling_result.get("paid_execution_requested") is not False
        or not sibling_result.get("blockers")
    ):
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_terminal_result_invalid"
        )
    submitted_at = _parse_time(
        target.get("submitted_at_iso"),
        blocker="stale_activation_reconciliation_target_invalid",
    )
    sibling_submitted_at = _parse_time(
        sibling.get("submitted_at_iso"),
        blocker="stale_activation_reconciliation_terminal_sibling_invalid",
    )
    sibling_observed_at = _parse_time(
        sibling_result.get("observed_at_iso"),
        blocker="stale_activation_reconciliation_terminal_result_invalid",
    )
    if sibling_submitted_at <= submitted_at or sibling_observed_at <= submitted_at:
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_terminal_sibling_not_newer"
        )
    age_seconds = (moment - submitted_at).total_seconds()
    if age_seconds < minimum_processing_age_seconds:
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_processing_lease_not_expired"
        )
    target_root = base / target_activation_id
    if os.path.lexists(target_root):
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_activation_artifacts_present"
        )
    identity_path = queue / "identities" / f"{target_activation_id}.json"
    identity, identity_snapshot = _read_sealed(
        identity_path,
        schema_version=IDENTITY_SCHEMA_VERSION,
        digest_field="identity_digest",
        blocker="stale_activation_reconciliation_identity_invalid",
    )
    if (
        identity.get("activation_id") != target_activation_id
        or identity.get("request_digest") != target.get("request_digest")
    ):
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_identity_invalid"
        )
    roots = tuple(_absolute(path, field="reference_root").resolve() for path in reference_roots)
    hits = _reference_hits(
        roots=roots,
        needles=(
            target_activation_id,
            str(target.get("request_digest")),
            str(target.get("envelope_digest")),
        ),
        excluded={target_path, identity_path},
    )
    if hits:
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_downstream_reference_present:"
            + hits[0]
        )
    zero = _validate_provider_zero(
        _absolute(provider_zero_report, field="provider_zero_report").resolve(),
        now=moment,
        maximum_age_seconds=maximum_provider_zero_age_seconds,
    )
    result: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "ready_to_reconcile",
        "planned_at": moment.isoformat().replace("+00:00", "Z"),
        "minimum_processing_age_seconds": minimum_processing_age_seconds,
        "maximum_provider_zero_age_seconds": maximum_provider_zero_age_seconds,
        "activation_queue_root": str(queue),
        "activation_base_root": str(base),
        "reference_roots": sorted(str(path) for path in roots),
        "target": {
            **target_snapshot,
            "activation_id": target_activation_id,
            "request_digest": target["request_digest"],
            "envelope_digest": target["envelope_digest"],
            "processing_age_seconds": age_seconds,
            "binding": _request_binding(target_request),
        },
        "terminal_sibling": {
            **sibling_snapshot,
            "activation_id": sibling_activation_id,
            "request_digest": sibling["request_digest"],
            "envelope_digest": sibling["envelope_digest"],
            "result": {
                **sibling_result_snapshot,
                "result_digest": sibling_result["result_digest"],
                "blockers": list(sibling_result["blockers"]),
            },
        },
        "target_identity": identity_snapshot,
        "provider_zero": zero,
        "downstream_reference_count": 0,
        "provider_mutation_performed": False,
        "production_artifact_deletion_performed": False,
        "evidence_deletion_performed": False,
        "plan_digest": "",
    }
    result["plan_digest"] = canonical_digest(result, digest_field="plan_digest")
    return result


def _read_plan(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_plan_invalid"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != PLAN_SCHEMA_VERSION
        or value.get("status") != "ready_to_reconcile"
        or value.get("plan_digest")
        != canonical_digest(value, digest_field="plan_digest")
    ):
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_plan_invalid"
        )
    return dict(value)


def _stable_projection(plan: Mapping[str, Any]) -> dict[str, Any]:
    target = dict(plan.get("target") or {})
    target.pop("processing_age_seconds", None)
    return {
        "activation_queue_root": plan.get("activation_queue_root"),
        "activation_base_root": plan.get("activation_base_root"),
        "reference_roots": plan.get("reference_roots"),
        "target": target,
        "terminal_sibling": plan.get("terminal_sibling"),
        "target_identity": plan.get("target_identity"),
        "downstream_reference_count": plan.get("downstream_reference_count"),
    }


def apply_stale_activation_reconciliation_plan(
    *, dry_run_plan_path: str | Path, acknowledgement: str, receipt_out: str | Path
) -> dict[str, Any]:
    """Revalidate and terminalize one exact orphan without deleting evidence."""

    if acknowledgement != APPLY_ACKNOWLEDGEMENT:
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_acknowledgement_missing"
        )
    plan_path = _absolute(dry_run_plan_path, field="dry_run_plan")
    plan = _read_plan(plan_path)
    target = Path(str(plan["target"]["path"]))
    sibling = Path(str(plan["terminal_sibling"]["path"]))
    provider_zero = Path(str(plan["provider_zero"]["path"]))
    current = build_stale_activation_reconciliation_plan(
        target_envelope=target,
        terminal_sibling_envelope=sibling,
        activation_queue_root=plan["activation_queue_root"],
        activation_base_root=plan["activation_base_root"],
        reference_roots=plan["reference_roots"],
        provider_zero_report=provider_zero,
        minimum_processing_age_seconds=plan["minimum_processing_age_seconds"],
        maximum_provider_zero_age_seconds=plan["maximum_provider_zero_age_seconds"],
    )
    if _stable_projection(current) != _stable_projection(plan):
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_plan_changed"
        )
    queue = Path(str(plan["activation_queue_root"]))
    receipt_path = _absolute(receipt_out, field="receipt_out")
    if receipt_path == plan_path or receipt_path.parent != queue / "reconciliations":
        raise StaleActivationReconciliationError(
            "stale_activation_reconciliation_receipt_path_invalid"
        )
    with target.open("rb") as lease:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise StaleActivationReconciliationError(
                "stale_activation_reconciliation_worker_lease_active"
            ) from exc
        current = build_stale_activation_reconciliation_plan(
            target_envelope=target,
            terminal_sibling_envelope=sibling,
            activation_queue_root=plan["activation_queue_root"],
            activation_base_root=plan["activation_base_root"],
            reference_roots=plan["reference_roots"],
            provider_zero_report=provider_zero,
            minimum_processing_age_seconds=plan["minimum_processing_age_seconds"],
            maximum_provider_zero_age_seconds=plan["maximum_provider_zero_age_seconds"],
        )
        if _stable_projection(current) != _stable_projection(plan):
            raise StaleActivationReconciliationError(
                "stale_activation_reconciliation_plan_changed"
            )
        activation_id = str(plan["target"]["activation_id"])
        terminal_result_digest = str(
            plan["terminal_sibling"]["result"]["result_digest"]
        )
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "activation_id": activation_id,
            "blockers": [
                "launch_activation_crash_stranded_processing_reconciled",
                f"terminal_sibling_result:{terminal_result_digest}",
            ],
            "catalog_mutation_state": "proven_absent_by_reconciliation",
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
            "observed_at_iso": datetime.now(timezone.utc).isoformat(),
            "result_digest": "",
        }
        result["result_digest"] = canonical_digest(
            result, digest_field="result_digest"
        )
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "reconciled_terminal_blocked",
            "activation_id": activation_id,
            "target_envelope_digest": plan["target"]["envelope_digest"],
            "terminal_sibling_activation_id": plan["terminal_sibling"][
                "activation_id"
            ],
            "terminal_sibling_envelope_digest": plan["terminal_sibling"][
                "envelope_digest"
            ],
            "terminal_sibling_result_digest": terminal_result_digest,
            "provider_zero_report_sha256": current["provider_zero"]["sha256"],
            "provider_zero_generated_at": current["provider_zero"]["generated_at"],
            "result_digest": result["result_digest"],
            "provider_mutation_performed": False,
            "production_artifact_deletion_performed": False,
            "evidence_deletion_performed": False,
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        receipt_path.parent.mkdir(mode=0o750, exist_ok=True)
        try:
            write_launch_preparation_record_exclusive(receipt_path, receipt)
            write_launch_preparation_record_exclusive(
                queue / "results" / target.name, result
            )
        except FileExistsError as exc:
            raise StaleActivationReconciliationError(
                "stale_activation_reconciliation_immutable_result_conflict"
            ) from exc
        os.replace(target, queue / "blocked" / target.name)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--target-envelope", required=True)
    plan_parser.add_argument("--terminal-sibling-envelope", required=True)
    plan_parser.add_argument("--activation-queue-root", required=True)
    plan_parser.add_argument("--activation-base-root", required=True)
    plan_parser.add_argument("--reference-root", action="append", required=True)
    plan_parser.add_argument("--provider-zero-report", required=True)
    plan_parser.add_argument("--output", required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--dry-run-plan", required=True)
    apply_parser.add_argument("--acknowledgement", required=True)
    apply_parser.add_argument("--receipt-out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = build_stale_activation_reconciliation_plan(
                target_envelope=args.target_envelope,
                terminal_sibling_envelope=args.terminal_sibling_envelope,
                activation_queue_root=args.activation_queue_root,
                activation_base_root=args.activation_base_root,
                reference_roots=args.reference_root,
                provider_zero_report=args.provider_zero_report,
            )
            output = _absolute(args.output, field="output")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(_canonical_bytes(result))
        else:
            result = apply_stale_activation_reconciliation_plan(
                dry_run_plan_path=args.dry_run_plan,
                acknowledgement=args.acknowledgement,
                receipt_out=args.receipt_out,
            )
    except (OSError, StaleActivationReconciliationError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "APPLY_ACKNOWLEDGEMENT",
    "StaleActivationReconciliationError",
    "apply_stale_activation_reconciliation_plan",
    "build_stale_activation_reconciliation_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
