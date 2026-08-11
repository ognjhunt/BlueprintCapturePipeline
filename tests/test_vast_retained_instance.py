from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.vast_retained_instance import bind_all_in_cost


def test_all_in_cost_uses_nested_created_instance_rate(tmp_path: Path) -> None:
    selected_offer = {"hourly_rate_usd": 0.65}

    binding = bind_all_in_cost(
        tmp_path,
        selected_offer=selected_offer,
        instance_payload={
            "instances": {
                "id": 123,
                "dph_total": 0.81,
                "storage_total_cost": 0.16,
            }
        },
        instance_id=123,
        disk_gb=200,
        max_live_minutes=120,
        max_hourly_rate=0.80,
        hard_cap_usd=2.0,
    )

    assert binding["compute_hourly_rate_usd"] == 0.65
    assert binding["storage_hourly_rate_usd"] == 0.16
    assert binding["all_in_hourly_rate_usd"] == 0.81
    assert binding["all_in_hourly_rate_under_max"] is False
    assert binding["projected_all_in_cost_usd"] == 1.62
    assert binding["projected_all_in_cost_under_hard_cap"] is True
    assert selected_offer["hourly_rate_usd"] == 0.81
    assert json.loads((tmp_path / "vast_all_in_cost_binding.json").read_text()) == binding


def test_all_in_cost_preserves_flat_payload_compatibility(tmp_path: Path) -> None:
    binding = bind_all_in_cost(
        tmp_path,
        selected_offer={"hourly_rate_usd": 0.65},
        instance_payload={"dph_total": 0.70},
        instance_id=456,
        disk_gb=100,
        max_live_minutes=60,
        max_hourly_rate=0.80,
        hard_cap_usd=1.0,
    )

    assert binding["all_in_hourly_rate_usd"] == 0.70
    assert binding["all_in_hourly_rate_under_max"] is True
    assert binding["projected_all_in_cost_under_hard_cap"] is True
