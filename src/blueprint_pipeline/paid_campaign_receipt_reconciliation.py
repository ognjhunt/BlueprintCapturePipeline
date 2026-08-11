"""Digest-bound historical spend reconciliation for paid campaigns.

The atomic campaign ledger prevents *new* controllers from overspending, but a
campaign can predate that ledger or contain receipts written by several
independent worktrees.  This module reconstructs the conservative amount that
must seed the ledger from retained provider receipts.  It is deliberately
strict about provider allocation identity: a paid-looking receipt without an
allocation ID is an ambiguity, not permission to count or ignore it manually.

The first admitted receipt adapter is Vast's ``vast_budget_ledger.v1``.  Its
``estimated_cost_usd`` is an elapsed-rate estimate, never an invoice.  Only an
``actual_cost_usd`` explicitly sourced from ``provider_billing_api`` is labeled
actual.  Repeated estimates for one allocation are deduplicated by provider and
allocation ID and conservatively resolved to their maximum observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "paid_campaign_receipt_reconciliation.v1"
VAST_LEDGER_SCHEMA_VERSION = "vast_budget_ledger.v1"
GPU_SPEND_GUARD_SCHEMA_VERSION = "gpu_spend_guard.v1"
AUTHORITATIVE_ACTUAL_SOURCE = "provider_billing_api"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _money(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return round(number, 6)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _path_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _receipt_paths(
    roots: Sequence[Path], *, include_path_substrings: Sequence[str]
) -> tuple[list[Path], list[dict[str, Any]], list[str]]:
    paths: set[Path] = set()
    bindings: list[dict[str, Any]] = []
    blockers: list[str] = []
    filters = tuple(str(item).strip() for item in include_path_substrings if str(item).strip())
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        if root.is_symlink() or not root.is_dir():
            blockers.append(f"campaign_receipt_root_invalid:{root}")
            continue
        bindings.append(
            {
                "path": str(root),
                "include_path_substrings": list(filters),
            }
        )
        for candidate in root.rglob("vast_budget_ledger.json"):
            resolved = candidate.resolve()
            candidate_text = str(resolved)
            if not all(token in candidate_text for token in filters):
                continue
            if (
                candidate.is_symlink()
                or not resolved.is_file()
                or not resolved.is_relative_to(root)
            ):
                blockers.append(f"campaign_receipt_file_invalid:{candidate}")
                continue
            paths.add(resolved)
    return sorted(paths), bindings, blockers


def _vast_observation(
    path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str], list[str]]:
    """Return (allocation observation, no-allocation receipt, blockers, warnings)."""

    binding = _path_binding(path)
    payload = _load_json(path)
    if payload is None:
        return None, None, [f"campaign_receipt_json_invalid:{path}"], []
    if payload.get("schema_version") != VAST_LEDGER_SCHEMA_VERSION:
        return None, None, [f"campaign_receipt_schema_invalid:{path}"], []

    estimated = _money(payload.get("estimated_cost_usd"))
    if estimated is None:
        return None, None, [f"campaign_receipt_estimated_cost_invalid:{path}"], []
    actual_value = payload.get("actual_cost_usd")
    actual = None if actual_value is None else _money(actual_value)
    if actual_value is not None and actual is None:
        return None, None, [f"campaign_receipt_actual_cost_invalid:{path}"], []
    actual_source = str(payload.get("actual_cost_source") or "").strip()

    raw_ids = payload.get("vast_instance_ids")
    if not isinstance(raw_ids, list):
        return None, None, [f"campaign_receipt_allocation_ids_invalid:{path}"], []
    ids = []
    for value in raw_ids:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            return None, None, [f"campaign_receipt_allocation_ids_invalid:{path}"], []
        identifier = str(value).strip()
        if not identifier:
            return None, None, [f"campaign_receipt_allocation_ids_invalid:{path}"], []
        ids.append(identifier)
    ids = sorted(set(ids))

    if not ids:
        if estimated != 0 or (actual is not None and actual != 0):
            return (
                None,
                None,
                [f"campaign_receipt_paid_cost_without_allocation_id:{path}"],
                [],
            )
        return (
            None,
            {
                **binding,
                "schema_version": payload["schema_version"],
                "receipt_status": payload.get("status"),
                "reason": "zero_cost_no_provider_allocation",
            },
            [],
            [],
        )
    if len(ids) != 1:
        return (
            None,
            None,
            [f"campaign_receipt_multi_allocation_cost_not_apportionable:{path}"],
            [],
        )

    warnings: list[str] = []
    if actual is not None and actual_source == AUTHORITATIVE_ACTUAL_SOURCE:
        selected_cost = actual
        cost_basis = "provider_reported_actual_usd"
        authoritative = True
    else:
        selected_cost = estimated
        cost_basis = "elapsed_rate_upper_bound_estimate"
        authoritative = False
        if actual is not None:
            warnings.append(
                f"non_authoritative_actual_cost_not_labeled_actual:{path}"
            )
    return (
        {
            "provider": "vast",
            "provider_allocation_id": ids[0],
            "provider_allocation_key": f"vast:{ids[0]}",
            "selected_cost_usd": selected_cost,
            "cost_basis": cost_basis,
            "cost_is_authoritative_actual": authoritative,
            "estimated_cost_usd": estimated,
            "actual_cost_usd": actual,
            "actual_cost_source": actual_source or None,
            "receipt_status": payload.get("status"),
            "actual_live_runtime_seconds_observed_by_adapter": payload.get(
                "actual_live_runtime_seconds_observed_by_adapter"
            ),
            "receipt": binding,
        },
        None,
        [],
        warnings,
    )


def _inventory_observation(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    resolved = path.expanduser().resolve()
    if path.is_symlink() or not resolved.is_file():
        return None, [f"campaign_inventory_receipt_invalid:{path}"]
    binding = _path_binding(resolved)
    payload = _load_json(resolved)
    if payload is None or payload.get("schema_version") != GPU_SPEND_GUARD_SCHEMA_VERSION:
        return None, [f"campaign_inventory_receipt_schema_invalid:{path}"]
    vast_rows = [
        dict(row)
        for row in payload.get("inventory_results") or []
        if isinstance(row, Mapping) and row.get("provider") == "vast"
    ]
    blockers: list[str] = []
    if len(vast_rows) != 1 or vast_rows[0].get("status") != "succeeded":
        blockers.append(f"campaign_inventory_vast_query_not_succeeded:{path}")
    live_count = payload.get("live_instance_count")
    if isinstance(live_count, bool) or not isinstance(live_count, int) or live_count < 0:
        blockers.append(f"campaign_inventory_live_count_invalid:{path}")
        live_count = None
    return (
        {
            **binding,
            "schema_version": payload.get("schema_version"),
            "generated_at": payload.get("generated_at"),
            "status": payload.get("status"),
            "vast_query_succeeded": not blockers,
            "live_instance_count": live_count,
            "provider_zero": live_count == 0 and not blockers,
        },
        blockers,
    )


def reconcile_paid_campaign_receipts(
    *,
    roots: Sequence[str | Path],
    include_path_substrings: Sequence[str] = (),
    inventory_receipts: Sequence[str | Path] = (),
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Reconcile retained paid receipts and latest provider inventory.

    Roots and path filters are explicit campaign-selection inputs.  The output
    remains blocked when any selected paid receipt cannot be joined to exactly
    one provider allocation or when the latest inventory does not prove zero.
    """

    receipt_paths, root_bindings, blockers = _receipt_paths(
        [Path(item) for item in roots],
        include_path_substrings=include_path_substrings,
    )
    observations_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_allocation_receipts: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in receipt_paths:
        observation, no_allocation, receipt_blockers, receipt_warnings = (
            _vast_observation(path)
        )
        blockers.extend(receipt_blockers)
        warnings.extend(receipt_warnings)
        if observation is not None:
            observations_by_key[observation["provider_allocation_key"]].append(
                observation
            )
        if no_allocation is not None:
            no_allocation_receipts.append(no_allocation)

    allocations: list[dict[str, Any]] = []
    for key, observations in sorted(observations_by_key.items()):
        actuals = [
            item
            for item in observations
            if item["cost_is_authoritative_actual"] is True
        ]
        if actuals:
            actual_values = {item["selected_cost_usd"] for item in actuals}
            if len(actual_values) != 1:
                blockers.append(f"campaign_allocation_actual_cost_conflict:{key}")
            selected = max(actuals, key=lambda item: item["selected_cost_usd"])
        else:
            selected = max(observations, key=lambda item: item["selected_cost_usd"])
            estimated_values = {item["selected_cost_usd"] for item in observations}
            if len(estimated_values) > 1:
                warnings.append(
                    f"campaign_allocation_estimate_conservative_max_selected:{key}"
                )
        allocations.append(
            {
                "provider": selected["provider"],
                "provider_allocation_id": selected["provider_allocation_id"],
                "provider_allocation_key": key,
                "selected_cost_usd": selected["selected_cost_usd"],
                "cost_basis": selected["cost_basis"],
                "cost_is_authoritative_actual": selected[
                    "cost_is_authoritative_actual"
                ],
                "observation_count": len(observations),
                "observations": sorted(
                    observations, key=lambda item: item["receipt"]["path"]
                ),
            }
        )

    inventory: list[dict[str, Any]] = []
    for raw_path in inventory_receipts:
        observation, inventory_blockers = _inventory_observation(Path(raw_path))
        blockers.extend(inventory_blockers)
        if observation is not None:
            inventory.append(observation)
    inventory.sort(key=lambda item: str(item.get("generated_at") or ""))
    if not inventory:
        blockers.append("campaign_latest_provider_inventory_missing")
    elif inventory[-1].get("provider_zero") is not True:
        blockers.append("campaign_latest_provider_inventory_not_zero")

    if not receipt_paths:
        blockers.append("campaign_paid_receipts_missing")
    total = round(sum(item["selected_cost_usd"] for item in allocations), 6)
    authoritative_total = round(
        sum(
            item["selected_cost_usd"]
            for item in allocations
            if item["cost_is_authoritative_actual"] is True
        ),
        6,
    )
    estimated_total = round(total - authoritative_total, 6)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": "qualified" if not blockers else "blocked",
        "selection": {
            "roots": root_bindings,
            "include_path_substrings": [
                str(item) for item in include_path_substrings
            ],
            "admitted_receipt_schemas": [VAST_LEDGER_SCHEMA_VERSION],
        },
        "selected_receipt_count": len(receipt_paths),
        "provider_allocation_count": len(allocations),
        "zero_cost_no_allocation_receipt_count": len(no_allocation_receipts),
        "campaign_spend_accounting_usd": total,
        "provider_reported_actual_usd": authoritative_total,
        "elapsed_rate_upper_bound_estimate_usd": estimated_total,
        "allocations": allocations,
        "zero_cost_no_allocation_receipts": no_allocation_receipts,
        "provider_inventory_receipts": inventory,
        "latest_provider_zero_proven": bool(
            inventory and inventory[-1].get("provider_zero") is True
        ),
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "claim_boundary": (
            "campaign_spend_accounting_usd is the conservative amount used for "
            "campaign-cap admission. Its elapsed-rate component is not an invoice. "
            "Provider-zero is a point-in-time API inventory fact, not run success."
        ),
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--include-path-substring", action="append", default=[])
    parser.add_argument("--inventory-receipt", action="append", default=[])
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = reconcile_paid_campaign_receipts(
        roots=args.root,
        include_path_substrings=args.include_path_substring,
        inventory_receipts=args.inventory_receipt,
    )
    write_json(Path(args.output).expanduser().resolve(), manifest)
    return 0 if manifest["status"] == "qualified" else 2


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())
