from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_g1_catalog import (
    G1_CANDIDATE_IDS,
    INVENTORY_PATH,
    unavailable_g1_preset,
)


def test_g1_catalog_lists_exact_pinned_pairs_without_readiness() -> None:
    preset = unavailable_g1_preset()
    assert preset["readiness"]["status"] == "unavailable"
    assert preset["readiness"]["receipt"] is None
    policies = preset["policy_candidates"]
    assert [policy["candidate_id"] for policy in policies] == list(G1_CANDIDATE_IDS)
    assert [policy["evaluation_objective_id"] for policy in policies] == [
        "task_success", "task_success", "g1_navigation_goal", "g1_navigation_goal",
    ]
    assert all(policy["readiness"]["status"] == "unavailable" for policy in policies)
    assert len({policy["checkpoint"]["digest"] for policy in policies}) == 4
    assert all(policy["checkpoint"]["size_bytes"] > 1_000_000_000 for policy in policies)


def test_g1_catalog_rejects_unpinned_candidate_inventory(tmp_path: Path) -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    inventory["candidates"][0]["files"][0]["size_bytes"] += 1
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError, match="g1_preflight_candidate_inventory_digest_invalid"):
        unavailable_g1_preset(inventory_path=path)
