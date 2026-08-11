from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import paid_campaign_receipt_reconciliation as R
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _ledger(
    path: Path,
    *,
    instance_ids: list[int] | None,
    estimated: float,
    actual: float | None = None,
    actual_source: str = "not_available_from_instance_probe_api",
) -> Path:
    return _write(
        path / "vast_budget_ledger.json",
        {
            "schema_version": "vast_budget_ledger.v1",
            "status": "completed" if instance_ids else "planned",
            "vast_instance_ids": instance_ids or [],
            "estimated_cost_usd": estimated,
            "actual_cost_usd": actual,
            "actual_cost_source": actual_source,
        },
    )


def _inventory(path: Path, *, live: int, when: str) -> Path:
    return _write(
        path,
        {
            "schema_version": "gpu_spend_guard.v1",
            "generated_at": when,
            "status": "passed" if live == 0 else "blocked",
            "live_instance_count": live,
            "inventory_results": [
                {
                    "provider": "vast",
                    "status": "succeeded",
                    "required": True,
                    "credential_present": True,
                    "row_count": live,
                    "blockers": [],
                }
            ],
        },
    )


def test_reconciles_both_scene_fixtures_without_scene_specific_code(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    _ledger(root / "840313_canned_beverage/run", instance_ids=[101], estimated=0.25)
    _ledger(root / "840796_fridge/run", instance_ids=[202], estimated=0.5)
    _ledger(root / "840796_fridge/dry_run", instance_ids=[], estimated=0.0)
    zero = _inventory(tmp_path / "provider-zero.json", live=0, when="2026-08-10T02:00:00Z")

    manifest = R.reconcile_paid_campaign_receipts(
        roots=[root],
        include_path_substrings=["840796"],
        inventory_receipts=[zero],
        generated_at="fixed",
    )

    assert manifest["status"] == "qualified"
    assert manifest["provider_allocation_count"] == 1
    assert manifest["campaign_spend_accounting_usd"] == 0.5
    assert manifest["elapsed_rate_upper_bound_estimate_usd"] == 0.5
    assert manifest["provider_reported_actual_usd"] == 0.0
    assert manifest["zero_cost_no_allocation_receipt_count"] == 1
    assert manifest["allocations"][0]["provider_allocation_id"] == "202"
    assert manifest["manifest_digest"] == canonical_digest(
        manifest, digest_field="manifest_digest"
    )


def test_deduplicates_allocation_and_selects_conservative_max_estimate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    _ledger(root / "copy-a", instance_ids=[303], estimated=0.4)
    _ledger(root / "copy-b", instance_ids=[303], estimated=0.6)
    zero = _inventory(tmp_path / "zero.json", live=0, when="2026-08-10T02:00:00Z")

    manifest = R.reconcile_paid_campaign_receipts(
        roots=[root], inventory_receipts=[zero], generated_at="fixed"
    )

    assert manifest["status"] == "qualified"
    assert manifest["provider_allocation_count"] == 1
    assert manifest["campaign_spend_accounting_usd"] == 0.6
    assert manifest["allocations"][0]["observation_count"] == 2
    assert manifest["warnings"] == [
        "campaign_allocation_estimate_conservative_max_selected:vast:303"
    ]


def test_only_provider_billing_api_cost_is_labeled_actual(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    _ledger(
        root / "untrusted",
        instance_ids=[401],
        estimated=0.7,
        actual=0.1,
        actual_source="caller_assertion",
    )
    _ledger(
        root / "authoritative",
        instance_ids=[402],
        estimated=0.8,
        actual=0.2,
        actual_source="provider_billing_api",
    )
    zero = _inventory(tmp_path / "zero.json", live=0, when="2026-08-10T02:00:00Z")

    manifest = R.reconcile_paid_campaign_receipts(
        roots=[root], inventory_receipts=[zero], generated_at="fixed"
    )

    assert manifest["status"] == "qualified"
    assert manifest["campaign_spend_accounting_usd"] == 0.9
    assert manifest["provider_reported_actual_usd"] == 0.2
    assert manifest["elapsed_rate_upper_bound_estimate_usd"] == 0.7
    assert len(manifest["warnings"]) == 1


def test_paid_cost_without_allocation_identity_blocks(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    receipt = _ledger(root / "broken", instance_ids=[], estimated=0.3)
    zero = _inventory(tmp_path / "zero.json", live=0, when="2026-08-10T02:00:00Z")

    manifest = R.reconcile_paid_campaign_receipts(
        roots=[root], inventory_receipts=[zero], generated_at="fixed"
    )

    assert manifest["status"] == "blocked"
    assert manifest["campaign_spend_accounting_usd"] == 0.0
    assert manifest["blockers"] == [
        f"campaign_receipt_paid_cost_without_allocation_id:{receipt}"
    ]


def test_latest_inventory_must_prove_provider_zero(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    _ledger(root / "run", instance_ids=[501], estimated=0.3)
    older_zero = _inventory(
        tmp_path / "older-zero.json", live=0, when="2026-08-10T01:00:00Z"
    )
    newer_live = _inventory(
        tmp_path / "newer-live.json", live=1, when="2026-08-10T02:00:00Z"
    )

    manifest = R.reconcile_paid_campaign_receipts(
        roots=[root],
        inventory_receipts=[newer_live, older_zero],
        generated_at="fixed",
    )

    assert manifest["status"] == "blocked"
    assert manifest["latest_provider_zero_proven"] is False
    assert manifest["blockers"] == ["campaign_latest_provider_inventory_not_zero"]


def test_cli_writes_digest_bound_manifest(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    _ledger(root / "run", instance_ids=[601], estimated=0.3)
    zero = _inventory(tmp_path / "zero.json", live=0, when="2026-08-10T02:00:00Z")
    output = tmp_path / "manifest.json"

    assert (
        R.main(
            [
                "--root",
                str(root),
                "--inventory-receipt",
                str(zero),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["status"] == "qualified"
    assert manifest["campaign_spend_accounting_usd"] == 0.3

