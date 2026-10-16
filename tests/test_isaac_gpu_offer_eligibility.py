"""Beta L5: reject non-rendering rentals before spend; keep compute workloads open."""

import pytest

from blueprint_pipeline.vast_provider_adapter import _select_offer


@pytest.mark.parametrize("name", ["A800 PCIE", "A100 SXM4", "Unregistered accelerator"])
def test_isaac_offer_selection_never_falls_back_to_non_rt_gpu(name):
    offer = {"ask_contract_id": 1, "gpu_name": name, "gpu_ram": 81920,
             "dph_total": .4, "compute_cap": 800, "driver_version": "580.178.04"}
    assert _select_offer([offer], max_hourly_rate=.9) is None
    for policy in ("generation", "training", "open"):
        assert _select_offer([offer], max_hourly_rate=.9, gpu_selection_policy=policy) is not None


def test_isaac_offer_selection_keeps_supported_l40s_eligible():
    offer = {"ask_contract_id": 2, "gpu_name": "L40S", "gpu_ram": 46068,
             "dph_total": .8, "compute_cap": 890, "driver_version": "580.126.09"}
    selected = _select_offer([offer], max_hourly_rate=.9, min_gpu_ram_mb=46000,
                             require_known_supported_isaac_driver=True)
    assert selected is not None and selected["ask_contract_id"] == 2
    assert selected["isaac_rt_candidate"] is True
