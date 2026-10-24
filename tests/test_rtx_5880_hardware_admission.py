"""Vendor-supported RT metadata cannot waive geography, driver or memory gates."""

from __future__ import annotations

import pytest

from blueprint_pipeline.gpu_selection_policy import _is_isaac_rt_candidate
from blueprint_pipeline.vast_provider_adapter import _offer_summary, _select_offer


def _offer(**overrides):
    # Retained read-only UK offer, 2026-10-24; no allocation is made by this test.
    return {
        "id": 28619335,
        "machine_id": 49863,
        "gpu_name": "RTX 5880Ada",
        "gpu_ram": 49140,
        "num_gpus": 1,
        "compute_cap": 890,
        "driver_version": "580.159.03",
        "geolocation": "United Kingdom, GB",
        "dph_base": 0.8,
        "storage_cost": 1 / 3,
        "disk_space": 984.125,
        "inet_down_cost": 0.0390625,
        "inet_up_cost": 0.0390625,
        **overrides,
    }


def _select(offer, **overrides):
    criteria = dict(
        max_hourly_rate=1,
        min_gpu_ram_mb=46000,
        min_compute_cap=800,
        max_compute_cap=900,
        require_known_supported_isaac_driver=True,
        require_isaac_rt_candidate=True,
        minimum_driver_version="580.65.06",
        disk_gb=200,
        required_provider_disk_gb=200,
        allowed_geolocation_country_codes=("US",),
        hard_cap_usd=8,
        max_live_minutes=120,
        expected_provider_download_bytes=52_473_322_184,
        expected_provider_upload_bytes=9_234_866_908,
    )
    criteria.update(overrides)
    return _select_offer([offer], **criteria)


@pytest.mark.parametrize("name", ["RTX 5880Ada", "NVIDIA RTX 5880 Ada"])
def test_vendor_rt_and_vram_metadata_for_both_provider_spellings(name):
    summary = _offer_summary(_offer(gpu_name=name, gpu_ram=128000), disk_gb=200)
    assert _is_isaac_rt_candidate(name)
    assert summary["isaac_rt_candidate"] is True
    assert summary["known_model_vram_cap_mb"] == 49152
    assert summary["gpu_ram_mb"] == 49152


def test_actual_offer_keeps_geography_and_transfer_budget_gates():
    assert _select(_offer()) is None
    assert _select(_offer(), allowed_geolocation_country_codes=("GB",)) is not None
    assert _select(_offer(), allowed_geolocation_country_codes=("GB",), hard_cap_usd=2) is None


def test_supported_metadata_does_not_admit_bad_driver_or_inadequate_memory():
    gb = {"allowed_geolocation_country_codes": ("GB",)}
    assert _select(_offer(driver_version="595.71.05"), **gb) is None
    assert _select(_offer(gpu_ram=24576), **gb) is None
    assert _select(_offer(gpu_name="RTX 4090", gpu_ram=49140), **gb) is None
    assert _select(_offer(gpu_name="A100", gpu_ram=81920), **gb) is None
