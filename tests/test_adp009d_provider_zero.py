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


def test_keyless_retired_runpod_leaves_the_required_scope_only_on_its_marker(monkeypatch) -> None:
    """RunPod is retired (2026-11-05): with no RunPod key it is recorded as retired and
    the receipt needs Vast and DigitalOcean; with a key it is still queried and required."""

    from blueprint_pipeline import gpu_render_providers
    from blueprint_pipeline.spend_admission_lock import retired_provider_inventory_row

    inventories = _inventories()
    inventories["runpod"] = {"provider": "runpod", "status": adp009d_provider_zero.RETIRED_CREDENTIAL_ABSENT}
    receipt = build_provider_zero_receipt(inventories)
    assert receipt["status"] == "passed" and receipt["provider_zero_verified"] is True
    assert receipt["provider_zero"]["required_provider_ids"] == ["vast", "digitalocean"]
    assert retired_provider_inventory_row("runpod") in receipt["inventory_results"]
    # Any other RunPod answer is a required inventory, as before.
    inventories["runpod"] = {"provider": "runpod", "status": "blocked", "blockers": ["runpod_api_key_missing"]}
    blocked = build_provider_zero_receipt(inventories)
    assert blocked["provider_zero_verified"] is False
    assert blocked["provider_zero"]["required_provider_ids"] == ["runpod", "vast", "digitalocean"]

    queried = []

    class Provider:
        def __init__(self, name, available):
            self.name, self._available = name, available

        def available(self):
            return {"provider": self.name, "available": self._available}

        def billable_inventory(self, *, name_prefix):
            queried.append(self.name)
            return _inventories()[self.name]

    for runpod_key in (False, True):
        queried.clear()
        monkeypatch.setattr(gpu_render_providers, "get_render_provider",
                            lambda name, runpod_key=runpod_key: Provider(name, name != "runpod" or runpod_key))
        collected = adp009d_provider_zero.collect_provider_zero_receipt()
        assert collected["provider_zero_verified"] is True
        assert ("runpod" in queried) is runpod_key
        assert ("runpod" in collected["provider_zero"]["required_provider_ids"]) is runpod_key
