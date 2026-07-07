from __future__ import annotations

import pytest

from blueprint_pipeline.lane_hardware_requirements import (
    build_lane_hardware_contract,
    hardware_contract_or_raise,
    resolve_gpu_vram_gb,
)
from blueprint_pipeline.paid_lane_guard import (
    PreSpendPreflightBlocked,
    require_pre_spend_preflight,
)
from blueprint_pipeline.provider_reliability_manifest import (
    build_pre_spend_preflight,
)

T4_LANE = "kitchen_g1_groot_sonic_eval"


def test_4090_fails_the_oscar_groot_lane_floor() -> None:
    """The exact 2026-07-06 failure: OSCAR-2B + GR00T on a 24GB card."""
    contract = build_lane_hardware_contract(
        lane=T4_LANE,
        gpu_type_id="NVIDIA GeForce RTX 4090",
        disk_gb=175,
    )
    assert contract["status"] == "FAIL"
    assert any(
        blocker.startswith("gpu_vram_below_lane_floor:24gb_lt_40gb")
        for blocker in contract["blockers"]
    )


def test_a6000_with_disk_passes_the_oscar_groot_lane() -> None:
    contract = build_lane_hardware_contract(
        lane=T4_LANE, gpu_type_id="NVIDIA RTX A6000", disk_gb=175
    )
    assert contract["status"] == "PASS"
    assert contract["blockers"] == []
    assert contract["vram_gb"] == 48.0
    assert "resolution reduction" in contract["requirements"]["notes"]


def test_unregistered_lane_fails_closed() -> None:
    contract = build_lane_hardware_contract(
        lane="brand_new_lane", gpu_type_id="NVIDIA RTX A6000", disk_gb=200
    )
    assert contract["status"] == "FAIL"
    assert "lane_hardware_requirements_unregistered:brand_new_lane" in (
        contract["blockers"]
    )


def test_unknown_gpu_fails_closed_unless_vram_supplied() -> None:
    unknown = build_lane_hardware_contract(
        lane=T4_LANE, gpu_type_id="Mystery GPU 9000", disk_gb=175
    )
    assert unknown["status"] == "FAIL"
    assert "gpu_vram_unknown:Mystery GPU 9000" in unknown["blockers"]

    explicit = build_lane_hardware_contract(
        lane=T4_LANE, gpu_type_id="Mystery GPU 9000", vram_gb=80.0, disk_gb=175
    )
    assert explicit["status"] == "PASS"


def test_missing_disk_fails_closed() -> None:
    contract = build_lane_hardware_contract(
        lane=T4_LANE, gpu_type_id="NVIDIA RTX A6000"
    )
    assert contract["status"] == "FAIL"
    assert "container_disk_size_missing" in contract["blockers"]


def test_hardware_contract_or_raise_refuses_under_provisioned_pod() -> None:
    with pytest.raises(RuntimeError, match="lane_hardware_contract_failed"):
        hardware_contract_or_raise(
            lane=T4_LANE, gpu_type_id="NVIDIA GeForce RTX 4090", disk_gb=175
        )
    assert resolve_gpu_vram_gb("NVIDIA GeForce RTX 4090") == 24.0


def _valid_preflight_inputs() -> dict:
    return {
        "provider": "runpod",
        "credential_present": True,
        "capacity_evidence": {"available": True, "detail": "stock=High"},
        "image_contract": {"image_ref": "img:1.2.3", "pinned": True},
        "runtime_contract": {
            "startup_marker": "STARTED",
            "progress_marker": "PROGRESS",
            "startup_timeout_seconds": 600,
            "no_progress_timeout_seconds": 1200,
        },
        "spend_gate_open": True,
    }


def test_preflight_fails_when_hardware_contract_fails() -> None:
    hardware = build_lane_hardware_contract(
        lane=T4_LANE, gpu_type_id="NVIDIA GeForce RTX 4090", disk_gb=175
    )
    preflight = build_pre_spend_preflight(
        **_valid_preflight_inputs(), hardware_contract=hardware
    )
    assert preflight["status"] == "FAIL"
    assert any(
        blocker.startswith("hardware_contract_invalid:gpu_vram_below_lane_floor")
        for blocker in preflight["blockers"]
    )


def test_preflight_passes_with_passing_hardware_contract() -> None:
    hardware = build_lane_hardware_contract(
        lane=T4_LANE, gpu_type_id="NVIDIA RTX A6000", disk_gb=175
    )
    preflight = build_pre_spend_preflight(
        **_valid_preflight_inputs(), hardware_contract=hardware
    )
    assert preflight["status"] == "PASS"
    assert preflight["hardware_contract"]["status"] == "PASS"


def test_chokepoint_raises_before_spend_on_bad_hardware(tmp_path) -> None:
    hardware = build_lane_hardware_contract(
        lane=T4_LANE, gpu_type_id="NVIDIA GeForce RTX 4090", disk_gb=175
    )
    with pytest.raises(PreSpendPreflightBlocked):
        require_pre_spend_preflight(
            lane=T4_LANE,
            **_valid_preflight_inputs(),
            hardware_contract=hardware,
            record_dir=tmp_path,
        )
    assert (tmp_path / "pre_spend_preflight.json").is_file()
