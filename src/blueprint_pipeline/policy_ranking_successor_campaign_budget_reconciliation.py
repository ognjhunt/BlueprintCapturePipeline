"""Settle a successor GPU reservation after a late terminal watchdog proof.

This recovery path is provider-read-only.  It preserves the allocator's original
fail-closed settlement receipt, requires exact owned-instance absence, and then
reuses the allocator's canonical campaign-budget settlement function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import utc_now_iso, write_json
from .paid_resource_allocator import _settle_successor_campaign_budget
from .production_gpu_campaign_budget import ProductionGpuCampaignBudget

SCHEMA_VERSION = "policy_ranking_successor_campaign_budget_reconciliation.v1"
RECEIPT_NAME = "successor_campaign_budget_reconciliation_v1.json"
PRESERVED_SETTLEMENT_NAME = "successor_campaign_budget_settlement_before_reconciliation_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, maximum_bytes: int = 2 * 1024 * 1024) -> dict[str, Any]:
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file() or metadata.st_size > maximum_bytes:
        raise ValueError(f"reconciliation_input_unsafe:{path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"reconciliation_input_not_object:{path.name}")
    return dict(payload)


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"reconciliation_{name}_invalid")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"reconciliation_{name}_invalid")
    return number


def reconcile_successor_campaign_budget_after_watchdog(
    *, job_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    root = Path(job_dir).expanduser().resolve()
    receipt_path = root / RECEIPT_NAME
    if receipt_path.exists():
        existing = _json_object(receipt_path)
        if existing.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("reconciliation_existing_receipt_schema_invalid")
        return existing

    adapter_path = root / "adapter_output.json"
    reservation_path = root / "successor_campaign_budget_reservation.json"
    settlement_path = root / "successor_campaign_budget_settlement.json"
    ledger_path = root / "production_campaign_budget_ledger.json"
    watchdog_path = root / "independent_vast_watchdog" / "groot_oscar_runpod_canary_watchdog.json"
    teardown_path = root / "vast_teardown_manifest.json"
    adapter = _json_object(adapter_path)
    reservation = _json_object(reservation_path)
    original_settlement = _json_object(settlement_path)
    ledger_before = _json_object(ledger_path)
    watchdog = _json_object(watchdog_path)
    teardown = _json_object(teardown_path)

    reservation_row = reservation.get("reservation")
    reservation_row = dict(reservation_row) if isinstance(reservation_row, Mapping) else {}
    reservation_id = str(reservation_row.get("reservation_id") or "")
    ledger_rows = ledger_before.get("reservations")
    ledger_rows = ledger_rows if isinstance(ledger_rows, list) else []
    matching_rows = [
        row
        for row in ledger_rows
        if isinstance(row, Mapping) and row.get("reservation_id") == reservation_id
    ]
    exact_instance = watchdog.get("recorded_vast_instance")
    exact_instance = dict(exact_instance) if isinstance(exact_instance, Mapping) else {}
    exact_teardown = watchdog.get("recorded_vast_instance_teardown")
    exact_teardown = dict(exact_teardown) if isinstance(exact_teardown, Mapping) else {}
    final_owned = watchdog.get("final_inventory")
    final_owned = dict(final_owned) if isinstance(final_owned, Mapping) else {}
    teardown_ids = teardown.get("vast_instance_ids")
    teardown_ids = teardown_ids if isinstance(teardown_ids, list) else []
    recorded_instance_id = str(exact_instance.get("instance_id") or "")

    blockers: list[str] = []
    if reservation.get("status") != "reserved" or not reservation_id:
        blockers.append("successor_campaign_reservation_invalid")
    if len(matching_rows) != 1 or matching_rows[0].get("status") != "open":
        blockers.append("successor_campaign_open_reservation_not_exact")
    if original_settlement.get("status") != "open_reservation_retained_fail_closed":
        blockers.append("successor_original_settlement_not_fail_closed_open")
    if (
        adapter.get("status") != "completed"
        or int(adapter.get("provider_mutations_performed") or 0) <= 0
        or adapter.get("continuing_spend_from_this_run") is not False
    ):
        blockers.append("successor_adapter_terminal_accounting_invalid")
    try:
        _number(adapter.get("runtime_seconds"), name="runtime_seconds")
        _number(adapter.get("estimated_gpu_cost_usd"), name="estimated_gpu_cost_usd")
    except (TypeError, ValueError):
        blockers.append("successor_adapter_runtime_or_cost_invalid")
    if (
        watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider") != "vast"
        or watchdog.get("provider_absence_confirmed") is not True
        or watchdog.get("provider_mutations_performed") != 0
        or watchdog.get("global_inventory_within_authorized_residual_limit") is not True
        or final_owned.get("api_confirmed") is not True
        or final_owned.get("live_resource_count") != 0
    ):
        blockers.append("successor_terminal_watchdog_proof_invalid")
    if (
        exact_instance.get("status") != "recorded"
        or not recorded_instance_id
        or exact_teardown.get("instance_id") != recorded_instance_id
        or exact_teardown.get("provider_absence_confirmed") is not True
        or int(recorded_instance_id) not in teardown_ids
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("runner_gpu_teardown_completed") is not True
    ):
        blockers.append("successor_exact_owned_instance_absence_invalid")
    if blockers:
        blocked = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "generated_at": utc_now_iso(),
            "blockers": blockers,
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
        write_json(receipt_path, blocked)
        return blocked

    preserved_path = root / PRESERVED_SETTLEMENT_NAME
    if preserved_path.exists():
        raise FileExistsError("successor_preserved_settlement_already_exists")
    shutil.copyfile(settlement_path, preserved_path)
    preserved_path.chmod(0o600)
    ledger_sha256_before = _sha256(ledger_path)
    original_settlement_sha256 = _sha256(preserved_path)
    budget = ProductionGpuCampaignBudget(
        ledger_path,
        initial_spent_usd=_number(ledger_before.get("initial_spent_usd"), name="initial_spent_usd"),
        initial_used_gpu_seconds=int(ledger_before.get("initial_used_gpu_seconds")),
        total_spend_cap_usd=_number(
            ledger_before.get("total_spend_cap_usd"), name="total_spend_cap_usd"
        ),
        combined_gpu_wall_cap_seconds=int(ledger_before.get("combined_gpu_wall_cap_seconds")),
    )
    settlement_result = _settle_successor_campaign_budget(
        budget=budget,
        reservation=reservation,
        result={**adapter, "independent_watchdog_close": watchdog},
        job_dir=root,
    )
    if settlement_result.get("status") != "settled":
        raise ValueError("successor_campaign_reconciliation_did_not_settle")
    ledger_after = _json_object(ledger_path)
    if ledger_after.get("open_reservation_count") != 0:
        raise ValueError("successor_campaign_reconciliation_reservation_still_open")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "settled",
        "generated_at": utc_now_iso(),
        "reservation_id": reservation_id,
        "recorded_vast_instance_id": recorded_instance_id,
        "charged_gpu_seconds": settlement_result["settlement"]["charged_gpu_seconds"],
        "charged_usd": settlement_result["settlement"]["charged_usd"],
        "ledger_sha256_before": ledger_sha256_before,
        "ledger_sha256_after": _sha256(ledger_path),
        "original_fail_closed_settlement_path": str(preserved_path),
        "original_fail_closed_settlement_sha256": original_settlement_sha256,
        "terminal_watchdog_sha256": _sha256(watchdog_path),
        "adapter_output_sha256_before_reconciliation": _sha256(adapter_path),
        "settlement_receipt_sha256": _sha256(settlement_path),
        "global_provider_zero_claimed": False,
        "owned_provider_instance_absence_confirmed": True,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "settles adapter-estimated WAM cost after exact owned-instance absence; "
            "not provider billing-export reconciliation or global provider zero"
        ),
    }
    write_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    result = reconcile_successor_campaign_budget_after_watchdog(job_dir=args.job_dir)
    print(json.dumps({"success": result.get("status") == "settled"}, sort_keys=True))
    return 0 if result.get("status") == "settled" else 2


if __name__ == "__main__":
    raise SystemExit(main())
