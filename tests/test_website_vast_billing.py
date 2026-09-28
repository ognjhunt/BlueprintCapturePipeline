from __future__ import annotations

import json

import pytest

from blueprint_pipeline import website_vast_billing as billing
from blueprint_pipeline import gpu_render_providers, vast_provider_adapter


def test_provider_charge_reads_exact_instance_and_all_pages(monkeypatch):
    monkeypatch.setattr(gpu_render_providers, "_read_secret", lambda _name: "test-key")
    paths = []

    def read(*, method, path, api_key, timeout_seconds):
        assert method == "GET" and api_key == "test-key" and timeout_seconds == 30
        paths.append(path)
        if len(paths) == 1:
            return 200, {"success": True, "results": [
                {"source": "instance-42", "type": "instance", "start": 100, "end": 100,
                 "amount": .228, "items": [{"type": "gpu", "amount": .221}]},
                {"source": "instance-43", "type": "instance", "start": 100, "end": 100,
                 "amount": 9, "items": []}], "next_token": "page-2"}
        return 200, {"success": True, "results": [
            {"source": "instance-42", "type": "instance", "start": 200, "end": 200,
             "amount": .01, "items": [{"type": "disk", "amount": .01}]}], "next_token": None}

    monkeypatch.setattr(vast_provider_adapter, "_api_json", read)
    charge = billing._provider_charge("42", preflight_epoch=100, teardown_epoch=200)
    assert charge["source"] == "instance-42"
    assert charge["amount_usd"] == .238
    assert len(charge["rows"]) == 2
    assert charge["provider_charge_receipt_digest"].startswith("sha256:")
    assert "after_token=page-2" in paths[1]
    assert "test-key" not in json.dumps(charge)


def test_provider_charge_missing_and_duplicate_fail_closed(monkeypatch):
    monkeypatch.setattr(gpu_render_providers, "_read_secret", lambda _name: "test-key")
    monkeypatch.setattr(vast_provider_adapter, "_api_json", lambda **_kw: (
        200, {"success": True, "results": [], "next_token": None}))
    with pytest.raises(ValueError, match="charge_missing_or_duplicate"):
        billing._provider_charge("42", preflight_epoch=100, teardown_epoch=200)
