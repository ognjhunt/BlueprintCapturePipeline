"""Fail-closed reconciliation for terminal no-allocation watchdog reservations."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import utc_now_iso
from .production_gpu_campaign_budget import ProductionGpuCampaignBudget


SCHEMA_VERSION = "no_allocation_budget_reconciliation.v1"


def _read_json_file(path_value: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label}_missing") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label}_unsafe_file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}_unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label}_not_object")
    return path, value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: object, *, label: str) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label}_timestamp_invalid") from exc
    if result.tzinfo is None:
        raise ValueError(f"{label}_timestamp_timezone_missing")
    return result


def reconcile_no_allocation_watchdog_budget(
    *,
    watchdog_evidence: str | Path,
    provider_zero_snapshot: str | Path,
    campaign_budget_ledger: str | Path,
    reservation_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Settle a full reservation only after terminal zero/no-ID evidence."""

    watchdog_path, watchdog = _read_json_file(watchdog_evidence, label="watchdog_evidence")
    zero_path, zero = _read_json_file(provider_zero_snapshot, label="provider_zero_snapshot")
    ledger_path, ledger = _read_json_file(campaign_budget_ledger, label="campaign_budget_ledger")
    expected_id = str(reservation_id or "").strip()
    blockers: list[str] = []
    if watchdog.get("schema_version") != "groot_oscar_runpod_canary_watchdog.v1":
        blockers.append("watchdog_schema_invalid")
    if watchdog.get("status") != "provider_terminal_budget_reservation_exceeded":
        blockers.append("watchdog_not_terminal_budget_breach")
    if watchdog.get("provider") != "vast":
        blockers.append("watchdog_provider_not_vast")
    if watchdog.get("provider_absence_confirmed") is not True:
        blockers.append("watchdog_provider_absence_unproven")
    if watchdog.get("control_plane_terminal") is not True:
        blockers.append("watchdog_control_plane_open")
    if watchdog.get("provider_mutations_performed") != 0:
        blockers.append("watchdog_provider_mutation_observed")
    recorded = watchdog.get("recorded_vast_instance")
    if not isinstance(recorded, Mapping) or not (
        recorded.get("status") == "not_recorded" and recorded.get("required") is False
    ):
        blockers.append("watchdog_no_instance_id_not_proven")
    pending_close = watchdog.get("pod_pending_teardown_close")
    if not isinstance(pending_close, Mapping) or pending_close.get("status") not in {
        "closed",
        "cancelled_no_allocation",
    }:
        blockers.append("watchdog_pending_teardown_not_terminal")
    lane_release = watchdog.get("provider_lane_terminal_release")
    if not isinstance(lane_release, Mapping) or lane_release.get("status") != "released":
        blockers.append("watchdog_paid_lane_not_released")
    breach = watchdog.get("campaign_budget_settlement")
    if not isinstance(breach, Mapping) or breach.get("status") != "retained_open_budget_breach":
        blockers.append("watchdog_retained_open_breach_missing")

    if zero.get("schema_version") != "gpu_spend_guard.v1" or zero.get("status") != "passed":
        blockers.append("provider_zero_snapshot_invalid")
    if zero.get("live_instance_count") != 0 or zero.get("total_burn_per_hour_usd") != 0:
        blockers.append("global_provider_zero_unproven")
    inventory_rows = zero.get("inventory_results")
    vast_rows = (
        [
            row
            for row in inventory_rows
            if isinstance(row, Mapping) and row.get("provider") == "vast"
        ]
        if isinstance(inventory_rows, list)
        else []
    )
    if len(vast_rows) != 1 or not (
        vast_rows[0].get("status") == "succeeded" and vast_rows[0].get("row_count") == 0
    ):
        blockers.append("vast_global_inventory_zero_unproven")
    try:
        if _timestamp(zero.get("generated_at"), label="provider_zero") < _timestamp(
            watchdog.get("completed_at"), label="watchdog_completed"
        ):
            blockers.append("provider_zero_snapshot_predates_watchdog_terminal")
    except ValueError as exc:
        blockers.append(str(exc))

    if ledger.get("schema_version") != "production_gpu_campaign_budget.v1":
        blockers.append("campaign_budget_ledger_schema_invalid")
    reservations = ledger.get("reservations")
    matching = (
        [
            row
            for row in reservations
            if isinstance(row, Mapping) and row.get("reservation_id") == expected_id
        ]
        if isinstance(reservations, list)
        else []
    )
    if len(matching) != 1 or matching[0].get("status") != "open":
        blockers.append("campaign_budget_open_reservation_not_found")
    if isinstance(breach, Mapping) and matching:
        if breach.get("reserved_gpu_seconds") != matching[0].get("reserved_gpu_seconds"):
            blockers.append("watchdog_reservation_seconds_mismatch")
        elapsed = breach.get("elapsed_gpu_seconds")
        reserved = matching[0].get("reserved_gpu_seconds")
        if type(elapsed) is not int or type(reserved) is not int or elapsed != reserved + 1:
            blockers.append("watchdog_breach_not_exactly_one_second")
    if blockers:
        raise ValueError("no_allocation_budget_reconciliation_blocked:" + blockers[0])

    reservation = dict(matching[0])
    budget = ProductionGpuCampaignBudget(
        ledger_path,
        initial_spent_usd=float(ledger["initial_spent_usd"]),
        initial_used_gpu_seconds=int(ledger["initial_used_gpu_seconds"]),
        total_spend_cap_usd=float(ledger["total_spend_cap_usd"]),
        combined_gpu_wall_cap_seconds=int(ledger["combined_gpu_wall_cap_seconds"]),
    )
    settlement = budget.settle(
        reservation_id=expected_id,
        charged_gpu_seconds=int(reservation["reserved_gpu_seconds"]),
        charged_usd=float(reservation["reserved_usd"]),
        outcome="no_allocation_watchdog_rounding_breach_full_reservation",
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "settled_conservative_full_reservation",
        "created_at": utc_now_iso(),
        "reservation_id": expected_id,
        "watchdog_evidence_sha256": _file_sha256(watchdog_path),
        "provider_zero_snapshot_sha256": _file_sha256(zero_path),
        "campaign_budget_ledger_sha256_after_settlement": _file_sha256(ledger_path),
        "settlement": settlement,
        "actual_provider_allocation_observed": False,
        "actual_attributable_provider_spend_usd": 0,
        "conservative_charged_gpu_seconds": settlement["charged_gpu_seconds"],
        "conservative_charged_usd": settlement["charged_usd"],
        "provider_mutations_performed": 0,
        "claim_boundary": (
            "Budget reconciliation only; not GPU execution, policy inference, WAM "
            "execution, ranking evidence, or physical evidence."
        ),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reconcile-no-allocation-watchdog",))
    parser.add_argument("--watchdog-evidence", required=True)
    parser.add_argument("--provider-zero-snapshot", required=True)
    parser.add_argument("--campaign-budget-ledger", required=True)
    parser.add_argument("--reservation-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = reconcile_no_allocation_watchdog_budget(
        watchdog_evidence=args.watchdog_evidence,
        provider_zero_snapshot=args.provider_zero_snapshot,
        campaign_budget_ledger=args.campaign_budget_ledger,
        reservation_id=args.reservation_id,
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "reconcile_no_allocation_watchdog_budget"]
