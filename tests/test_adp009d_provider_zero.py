from __future__ import annotations

from datetime import datetime, timezone

from blueprint_pipeline.adp009d_provider_zero import build_provider_zero_receipt
from blueprint_pipeline import adp009d_provider_zero
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _inventories() -> dict[str, dict]:
    return {
        provider: {
            "provider": provider,
            "status": "observed",
            "api_confirmed": True,
            "resources": [],
            "blockers": [],
            "raw_provider_response_recorded": False,
        }
        for provider in ("runpod", "vast", "digitalocean")
    }


def test_provider_zero_receipt_requires_all_api_confirmed_global_inventories() -> None:
    receipt = build_provider_zero_receipt(
        _inventories(), now=datetime(2026, 8, 12, tzinfo=timezone.utc)
    )

    assert receipt["status"] == "passed"
    assert receipt["provider_zero_verified"] is True
    assert receipt["live_instance_count"] == 0
    assert receipt["instances"] == []
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_provider_zero_receipt_blocks_live_or_unconfirmed_inventory() -> None:
    inventories = _inventories()
    inventories["vast"]["resources"] = [{"instance_id": "47501029"}]
    inventories["runpod"]["api_confirmed"] = False

    receipt = build_provider_zero_receipt(inventories)

    assert receipt["status"] == "blocked"
    assert receipt["provider_zero_verified"] is False
    assert receipt["live_instance_count"] == 1
    assert "provider_zero_live_resources_detected" in receipt["blockers"]
    assert any(item.startswith("provider_inventory:runpod:") for item in receipt["blockers"])


def test_provider_zero_cli_writes_only_compact_sanitized_summary(
    tmp_path, monkeypatch
) -> None:
    receipt = build_provider_zero_receipt(_inventories())
    monkeypatch.setattr(
        adp009d_provider_zero, "collect_provider_zero_receipt", lambda: receipt
    )
    output = tmp_path / "provider_zero.json"

    assert adp009d_provider_zero.main(["--output", str(output)]) == 0
    assert output.is_file()
