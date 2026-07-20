from __future__ import annotations

import json
from pathlib import Path


REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "external_anchor_candidate_registry_2026-07-20.json"
)


def _registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_external_anchor_registry_never_inherits_paper_metrics() -> None:
    registry = _registry()
    boundary = registry["claim_boundary"]

    assert boundary["blueprint_collects_real_world_task_outcomes"] is False
    assert boundary["blueprint_real_world_ordering_correlation_status"] == (
        "correlation_not_measured"
    )
    assert boundary["paper_metrics_are_inherited"] is False
    assert "pearson_ranking_fidelity" in boundary["sc3_0_929_semantics"]
    assert "accuracy" in boundary["sc3_0_929_semantics"]


def test_external_anchor_candidates_remain_blocked_until_raw_rights_and_review_pass() -> None:
    candidates = {row["candidate_id"]: row for row in _registry()["candidates"]}

    assert set(candidates) == {
        "sc3_eval_v3_published_artifacts",
        "oscar_policy_rollout_2026_06_16",
        "roboarena_data_dump_2026_07_17",
    }
    assert all(row["status"].startswith("blocked_candidate") for row in candidates.values())
    assert candidates["sc3_eval_v3_published_artifacts"]["raw_per_cell_outcomes_available"] is False
    assert candidates["oscar_policy_rollout_2026_06_16"]["dataset_license_declared"] is False
    roboarena = candidates["roboarena_data_dump_2026_07_17"]
    assert roboarena["raw_per_session_outcomes_available"] is True
    assert roboarena["direct_identifiers_observed_in_session_metadata"] is True
    assert roboarena["privacy_and_consent_scope_verified"] is False
    assert roboarena["independently_accepted"] is False


def test_oscar_runtime_license_inventory_keeps_benchmark_assets_separate() -> None:
    inventory = _registry()["oscar_runtime_license_inventory"]

    assert inventory["source"]["revision"] == (
        "4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb"
    )
    assert inventory["checkpoint"]["revision"] == (
        "c9781ffa7dd8556d862d7d9f338a2ea008a58ca6"
    )
    assert inventory["official_benchmark_assets_commercial_use_verified"] is False
    assert inventory["official_benchmark_assets_may_be_used_for_commercial_decision_rows"] is False
