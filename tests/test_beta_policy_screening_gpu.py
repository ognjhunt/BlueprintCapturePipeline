"""Screening planning refuses a memory floor the screening GPU lane cannot rent."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import beta_policy_screening_gpu as gpu
from blueprint_pipeline import gpu_selection_policy
from blueprint_pipeline.beta_policy_screening_coordinator import INFEASIBLE_BLOCKER, _initial_state
from blueprint_pipeline.native_task_arena_vast import policy_canary_session_offer_hardware
from blueprint_pipeline.policy_canary_session_bundle_identity import minimum_gpu_ram_mb
from blueprint_pipeline.vast_compute_capability import MAX_COMPUTE_CAP_ENV
from blueprint_pipeline.vast_provider_adapter import _known_gpu_vram_cap_mb, _select_offer

CATALOG = Path(__file__).resolve().parents[1] / "config" / "beta_policy_candidates.v1.json"


@pytest.fixture(autouse=True)
def _default_ceiling(monkeypatch):
    monkeypatch.delenv(MAX_COMPUTE_CAP_ENV, raising=False)


def _runtime(candidate_id: str) -> dict:
    rows = json.loads(CATALOG.read_text(encoding="utf-8"))["candidates"]
    return next(row for row in rows if row["candidate_id"] == candidate_id)["runtime"]


def test_catalog_floor_the_lane_cannot_rent_is_refused_and_the_rest_stay_feasible():
    assert _runtime("cosmos3_nano_policy_droid")["minimum_gpu_memory_gb"] == 80
    assert gpu.gpu_infeasible_blockers(_runtime("cosmos3_nano_policy_droid")) == [gpu.GPU_MEMORY_FLOOR_UNRENTABLE]
    for candidate in ("cosmos3_edge_policy_droid", "groot_n17_droid", "molmoact2_droid", "pi05_droid"):
        assert gpu.gpu_infeasible_blockers(_runtime(candidate)) == [], candidate
    # No declared floor is the bundle's 46 GB default; a runtime-less row is never refused here.
    assert gpu.gpu_infeasible_blockers({"minimum_gpu_memory_gb": None}) == []
    assert gpu.gpu_infeasible_blockers(None) == []
    assert gpu.rentable_parts(48) == ["L40", "L40S", "RTX 6000 Ada", "RTX 6000Ada", "RTX A6000"]


def test_lifting_the_lane_ceiling_admits_the_large_rt_part(monkeypatch):
    monkeypatch.setenv(MAX_COMPUTE_CAP_ENV, "0")
    assert gpu.rentable_parts(80) == ["RTX PRO 6000 Max-Q", "RTX PRO 6000 S", "RTX PRO 6000 WS"]
    assert gpu.gpu_infeasible_blockers({"minimum_gpu_memory_gb": 80}) == []


def test_an_undescribed_rt_keyword_leaves_the_market_unknown(monkeypatch):
    monkeypatch.setattr(gpu_selection_policy, "ISAAC_RT_GPU_KEYWORDS",
                        (*gpu_selection_policy.ISAAC_RT_GPU_KEYWORDS, "RTX 9000 FUTURE"))
    assert gpu.rentable_parts(80) is None
    assert gpu.gpu_infeasible_blockers({"minimum_gpu_memory_gb": 80}) == []


def test_every_rt_keyword_is_described_and_memory_agrees_with_the_allocator_registry():
    names = [name.upper() for name in gpu.RT_GPU_PARTS]
    for keyword in gpu_selection_policy.ISAAC_RT_GPU_KEYWORDS:
        assert any(keyword in name for name in names), keyword
    for name, (memory_gb, _capability) in gpu.RT_GPU_PARTS.items():
        known = _known_gpu_vram_cap_mb(name)
        assert known is None or known == memory_gb * 1024, name


@pytest.mark.parametrize("floor_gb", [24, 48, 80])
def test_planning_verdict_matches_the_allocator_offer_selection(floor_gb):
    """The allocator's own selection over every described part, plus the denied datacenter parts."""
    parts = {**gpu.RT_GPU_PARTS, "A100 SXM4": (80, 800), "H100 SXM": (80, 900), "H200": (141, 900)}
    offers = [{"id": index + 1, "dph_total": 0.5, "gpu_name": name, "gpu_ram": memory * 1024,
               "compute_cap": capability, "num_gpus": 1}
              for index, (name, (memory, capability)) in enumerate(parts.items())]
    lane = policy_canary_session_offer_hardware(minimum_gpu_ram_mb({"minimum_gpu_memory_gb": floor_gb}))
    selected = _select_offer(offers, max_hourly_rate=10.0, min_gpu_ram_mb=lane["min_gpu_ram_mb"],
                             min_compute_cap=lane["min_compute_cap"], prefer_isaac_rt=True,
                             require_isaac_rt_candidate=True, gpu_selection_policy=lane["gpu_selection_policy"])
    rentable = gpu.rentable_parts(floor_gb)
    assert (selected is None) is (rentable == [])
    if selected is not None:
        assert selected["gpu_name"] in rentable


def test_published_state_marks_only_measured_geometry_as_measured():
    geometry = _initial_state({"status": "infeasible", "blockers": ["team_embodiment_station_out_of_reach"]})
    assert geometry == {"status": "unavailable",
                        "blockers": sorted([INFEASIBLE_BLOCKER, "team_embodiment_station_out_of_reach"])}
    unrentable = _initial_state({"status": "infeasible", "blockers": [gpu.GPU_MEMORY_FLOOR_UNRENTABLE]})
    assert unrentable == {"status": "unavailable", "blockers": [gpu.GPU_MEMORY_FLOOR_UNRENTABLE]}
