"""Malformed successful HTTP bodies are not evidence of provider absence."""
from types import SimpleNamespace

import pytest

from blueprint_pipeline import groot_oscar_runpod_watchdog as watchdog
from blueprint_pipeline import task_evaluation_policy_canary_dispatcher as dispatcher
from blueprint_pipeline import vast_provider_adapter as adapter
from blueprint_pipeline.gpu_render_providers import VastRenderProvider


@pytest.mark.parametrize("payload", [{}, {"instances": None}, {"instances": "truncated"},
    {"instances": [None]}, {"success": False, "instances": []}])
def test_malformed_inventory_never_proves_teardown_or_zero(monkeypatch, payload):
    reads = []
    def transport(**kwargs):
        reads.append((kwargs["method"], kwargs["path"]))
        assert reads[-1] == ("GET", "/instances/")
        return 200, payload
    monkeypatch.setattr(adapter, "_api_json", transport)
    monkeypatch.setattr(VastRenderProvider, "_key", lambda _self: "synthetic-key")
    def no_delete(_instance):
        pytest.fail("malformed inventory authorized a deletion")
    result = watchdog.terminate_canary_resources(provider=SimpleNamespace(name="vast",
        _key=lambda: "synthetic-key", terminate=no_delete), provider_name="vast",
        pod_name_prefix="blueprint-offline-fixture-", armed={"status": "armed", "provider": "vast"})
    assert result["provider_absence_confirmed"] is False
    assert result["status"] == "teardown_unverified"
    zero = dispatcher.collect_policy_canary_vast_provider_zero()
    assert zero["provider_zero_verified"] is False
    assert zero["live_instance_count"] is None
    assert reads
