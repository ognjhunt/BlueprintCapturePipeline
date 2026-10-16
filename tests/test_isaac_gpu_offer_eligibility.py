"""Beta L5: reject non-rendering rentals before spend; keep compute workloads open."""

import json

import pytest

from blueprint_pipeline.vast_provider_adapter import _offer_selection_manifest, _select_offer


@pytest.mark.parametrize("name", ["A800 PCIE", "A100 SXM4", "Unregistered accelerator"])
def test_isaac_offer_selection_never_falls_back_to_non_rt_gpu(name):
    offer = {"ask_contract_id": 1, "gpu_name": name, "gpu_ram": 81920,
             "dph_total": .4, "compute_cap": 800, "driver_version": "580.178.04"}
    assert _select_offer([offer], max_hourly_rate=.9, require_isaac_rt_candidate=True) is None
    assert _select_offer([offer], max_hourly_rate=.9, require_isaac_rt_candidate=True,
                         gpu_selection_policy="open") is None
    for policy in ("generation", "training", "open"):
        assert _select_offer([offer], max_hourly_rate=.9, gpu_selection_policy=policy) is not None


def test_isaac_offer_selection_keeps_supported_l40s_eligible():
    offer = {"ask_contract_id": 2, "gpu_name": "L40S", "gpu_ram": 46068,
             "dph_total": .8, "compute_cap": 890, "driver_version": "580.126.09"}
    selected = _select_offer([offer], max_hourly_rate=.9, min_gpu_ram_mb=46000,
                             require_known_supported_isaac_driver=True, require_isaac_rt_candidate=True)
    assert selected is not None and selected["ask_contract_id"] == 2
    assert selected["isaac_rt_candidate"] is True


def test_compute_only_legacy_selection_keeps_cmp170hx_eligible():
    offer = {"ask_contract_id": 3, "gpu_name": "CMP 170HX", "gpu_ram": 65536,
             "dph_total": .4, "compute_cap": 800, "driver_version": "580.178.04"}
    assert _select_offer([offer], max_hourly_rate=.9)["ask_contract_id"] == 3
    assert _select_offer([offer], max_hourly_rate=.9, require_isaac_rt_candidate=True) is None


def test_offer_manifest_records_the_native_requirement_and_filtered_count(tmp_path):
    offer = {"ask_contract_id": 7, "gpu_name": "A800 PCIE", "gpu_ram": 81920,
             "dph_total": .4, "compute_cap": 800, "driver_version": "580.178.04"}
    kwargs = dict(generated_at="fixed", status_code=200, offers=[offer], selected_offer=None,
                  max_hourly_rate=.9, min_gpu_ram_mb=46000, require_known_supported_isaac_driver=False,
                  excluded_machine_ids=(), allowed_machine_ids=(), machine_avoidlist_path=tmp_path / "avoid.json",
                  avoidlist_status=None, blockers=[])
    native = _offer_selection_manifest(**kwargs, require_isaac_rt_candidate=True)
    assert native["require_isaac_rt_candidate"] is True and native["quality_filtered_offer_count"] == 0
    legacy = _offer_selection_manifest(**kwargs)
    assert legacy["require_isaac_rt_candidate"] is False and legacy["quality_filtered_offer_count"] == 1


def test_real_adapter_forwards_native_requirement_before_rental(tmp_path, monkeypatch):
    from blueprint_pipeline import vast_provider_adapter as adapter
    from tests.test_vast_provider_adapter import _run_with_staged_output, _runtime_output_archive

    observed = []
    original = adapter._select_offer

    def select(offers, **kwargs):
        observed.append(kwargs["require_isaac_rt_candidate"])
        return original(offers, **kwargs)

    def download(**kwargs):
        path = kwargs["output_path"]
        path.write_bytes(_runtime_output_archive())
        return {"status": "completed", "downloaded_size_bytes": path.stat().st_size}

    monkeypatch.setattr(adapter, "_select_offer", select)
    monkeypatch.setattr(adapter, "_download_provider_output_with_capacity_guard", download)
    events = []
    _run_with_staged_output(tmp_path, monkeypatch, events, provider_bundle_kind="isaac")
    assert observed == [True] and "teardown" in events
    manifest = json.loads((tmp_path / "vast_offer_selection_manifest.json").read_text())
    assert manifest["require_isaac_rt_candidate"] is True


def test_native_price_refusal_keeps_price_reason_when_rt_is_not_the_cause(tmp_path, monkeypatch):
    from tests.test_vast_provider_adapter import _run_with_staged_output

    events = []
    result, _ = _run_with_staged_output(tmp_path, monkeypatch, events,
                                       provider_bundle_kind="isaac", max_hourly_rate=.1)
    manifest = json.loads((tmp_path / "vast_offer_selection_manifest.json").read_text())
    assert manifest["blockers"] == ["no_vast_offer_at_or_below_max_hourly_rate"]
    assert result["blockers"] == manifest["blockers"] and not events
