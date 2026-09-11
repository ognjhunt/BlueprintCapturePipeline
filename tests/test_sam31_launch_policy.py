"""Replay the observed capacity/create mismatch without a provider call."""
from __future__ import annotations

import itertools
import json

import pytest

from blueprint_pipeline import sam31_vast_source_track_canary as canary
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.sam31_gpu_admission import collect_sam31_vast_preflight
from blueprint_pipeline.vast_provider_adapter import _select_offer
from tests.test_sam31_vast_source_track_canary import (
    _Provider, _bound_request, _preflight, _grant, _runtime_result,
    INPUT_URL, PUT_URL, GET_URL, TOKEN,
    recovery_identity as recovery_identity,
)


def test_saved_64gb_offer_does_not_replace_resource_requirements():
    # Minimal reconstruction of the saved offer's compute/default-8GB-disk
    # prices. The raw provider response was not retained; this is an offline
    # pricing reproduction, not a claim to recover that missing API response.
    offer = {"id": 50631939, "gpu_name": "CMP 170HX", "gpu_ram": 65536,
        "dph_base": .4, "dph_total": .4022222222222222, "storage_total_cost": .0022222222222222222,
        "storage_cost": .2, "disk_space": 135, "compute_cap": 800, "num_gpus": 1,
        "reliability": .981603, "direct_port_count": 49, "has_avx": True,
        "geolocation": "Missouri, US", "driver_version": "610.43.03"}
    old_quote = _select_offer([offer], max_hourly_rate=.5, min_gpu_ram_mb=24000,
                             allowed_geolocation_country_codes=["US"])
    assert old_quote["hourly_rate_usd"] == pytest.approx(.4022222222222222)
    assert _select_offer([offer], max_hourly_rate=old_quote["hourly_rate_usd"],
        min_gpu_ram_mb=65536, disk_gb=80, required_provider_disk_gb=80) is None
    calls = []

    def capacity(request):
        calls.append(request)
        chosen = _select_offer([offer], max_hourly_rate=request["max_hourly_rate_usd"],
            min_gpu_ram_mb=request["min_gpu_ram_mb"], disk_gb=request["container_disk_gb"],
            required_provider_disk_gb=request["required_provider_disk_gb"],
            require_avx=request["require_avx"], min_reliability=request["min_reliability"],
            require_direct_port=request["require_direct_port"],
            allowed_geolocation_country_codes=request["allowed_geolocation_country_codes"])
        return {"status": "available" if chosen else "blocked", "selected_offer": chosen}

    preflight = collect_sam31_vast_preflight(name_prefix="fixture", container_disk_bytes=80 * 1024**3,
        watchdog={"status": "armed", "independent_process": True}, conflicting_owner_present=False,
        capacity_probe=capacity, inventory_probe=lambda _: {"api_confirmed": True, "live_resource_count": 0},
        max_hourly_rate_usd=.5, clock=lambda: 1000)
    assert preflight["status"] == "verified"
    assert preflight["on_demand_price_usd_per_hour"] == pytest.approx(.4222222222222222)
    request = {**_bound_request(), "bound_preflight_digest": canonical_digest(preflight)}
    policy = canary.frozen_sam31_launch_policy(request, preflight)
    assert policy == calls[0]
    assert policy["max_hourly_rate_usd"] == .5
    assert policy["min_gpu_ram_mb"] == 25770
    assert policy["container_disk_gb"] == policy["required_provider_disk_gb"] == 80


def test_launch_reopens_exact_bound_preflight_before_provider_access():
    snapshot = _preflight()
    snapshot["capacity_request"]["max_hourly_rate_usd"] = 1
    with pytest.raises(canary.Sam31VastCanaryError, match="preflight_binding_changed"):
        canary.frozen_sam31_launch_policy(_bound_request(), snapshot)


def test_legacy_unpriced_disk_snapshot_cannot_authorize_another_create():
    snapshot = _preflight()
    snapshot["capacity_request"].pop("container_disk_gb")
    snapshot["capacity_request"].pop("required_provider_disk_gb")
    request = {**_bound_request(), "bound_preflight_digest": canonical_digest(snapshot)}
    with pytest.raises(canary.Sam31VastCanaryError, match="capacity_policy_changed"):
        canary.frozen_sam31_launch_policy(request, snapshot)


def _run(tmp_path, provider):
    times = itertools.count(1000.0)
    return canary.run_sam31_vast_source_track_canary(bound_request=_bound_request(), preflight=_preflight(),
        job_dir=tmp_path, input_bundle_get_url=INPUT_URL, output_put_url=PUT_URL, output_get_url=GET_URL,
        hf_token=TOKEN, provider=provider, paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _: _runtime_result(), sleeper=lambda _: None, clock=lambda: next(times),
        watchdog_validator=lambda *_: True)


def test_provider_refusal_is_retained_without_runtime_secrets(tmp_path):
    class Refused(_Provider):
        def launch(self, job_dir, request, **kwargs):
            self.requests.append(request)
            return {"status": "blocked", "blockers": ["no_vast_offer_matching_rate_and_gpu_memory", INPUT_URL],
                "attempts": [{"offer_search_status": 200, "offer_count": 0,
                              "create_error_body": TOKEN + PUT_URL}]}

    result = _run(tmp_path, Refused())
    assert result["status"] == "failed"
    assert "no_vast_offer_matching_rate_and_gpu_memory" in result["blockers"]
    assert result["provider_zero_verified"] and result["cost_usd"] == 0
    diagnostic = json.loads((tmp_path / "provider_launch_outcome.json").read_text())
    assert diagnostic["attempts"] == [{"offer_search_status": 200, "offer_count": 0}]
    for path in tmp_path.rglob("*.json"):
        assert all(secret not in path.read_text() for secret in (TOKEN, INPUT_URL, PUT_URL, GET_URL))


def test_diagnostic_write_failure_cannot_skip_exact_teardown(tmp_path, monkeypatch):
    provider = _Provider()
    def full_disk(*_):
        raise OSError("ENOSPC diagnostic write")
    monkeypatch.setattr(canary, "_retain_launch_outcome", full_disk)
    with pytest.raises(OSError, match="ENOSPC"):
        _run(tmp_path, provider)
    assert provider.requests and not provider.launched
    assert json.loads((tmp_path / "teardown_receipt.json").read_text())["instance_id"] == "42"
    assert not list((tmp_path / "leases").glob("*.lease.json"))


def test_admission_cost_uses_frozen_rate_ceiling_instead_of_cheaper_offer():
    from tests.test_sam31_gpu_admission import _build, _preflight as admission_preflight
    snapshot = admission_preflight()
    snapshot["capacity_request"]["max_hourly_rate_usd"] = 10
    snapshot["preflight_digest"] = canonical_digest(snapshot, digest_field="preflight_digest")
    admission, bound = _build(preflight=snapshot, execute=True, qualified=True)
    assert "sam31_gpu_budget_below_worst_case_cost" in admission["blockers"]
    assert bound["provider_mutation_authorized"] is False
