from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.dual_task_scenario_suite import (
    COUSIN_BLOCKER,
    DualTaskScenarioSuiteError,
    REQUIRED_FAMILIES,
    validate_dual_task_scenario_suite,
)


def _manifests() -> list[dict]:
    root = Path(__file__).parents[1] / "docs/arm_decision_proof_v1/manifests"
    return [
        json.loads((root / name).read_text())
        for name in (
            "third_scene_840920_task_a_scenario_suite.v1.json",
            "third_scene_840920_task_b_scenario_suite.v1.json",
        )
    ]


def test_checked_dual_task_suites_freeze_same_candidates_and_all_families() -> None:
    task_a, task_b = [validate_dual_task_scenario_suite(row) for row in _manifests()]

    assert task_a["scene_id"] == task_b["scene_id"] == "840920"
    assert task_a["shared_scene_freeze_digest"] == task_b["shared_scene_freeze_digest"]
    assert task_a["task_id"] != task_b["task_id"]
    assert task_a["candidate_ids"] == task_b["candidate_ids"] == [
        "pi05_droid",
        "groot_n17_droid",
    ]
    assert {cell["family"] for cell in task_a["cells"]} == REQUIRED_FAMILIES
    assert {cell["family"] for cell in task_b["cells"]} == REQUIRED_FAMILIES
    assert sum(cell["powered_diagnostic"] for cell in task_a["cells"]) == 1
    assert sum(cell["powered_diagnostic"] for cell in task_b["cells"]) == 1
    for suite in (task_a, task_b):
        cousin = next(
            cell for cell in suite["cells"] if cell["family"] == "admitted_object_cousin"
        )
        assert cousin["execution_admission"] == {
            "status": "blocked",
            "blocker_code": COUSIN_BLOCKER,
            "cousin_asset": None,
        }
        assert cousin["resolved_parameters"] == {}
        assert cousin["factor_records"] == []
        assert cousin["applied_parameter_readback_required"] is False


def test_suite_rejects_outcome_leakage_or_initial_matrix_expansion() -> None:
    suite = _manifests()[0]
    suite["learned_policy_outcomes_consulted"] = True
    suite["cells"][1]["scheduled_initially"] = True
    suite["suite_digest"] = canonical_digest(suite, digest_field="suite_digest")

    with pytest.raises(DualTaskScenarioSuiteError) as excinfo:
        validate_dual_task_scenario_suite(suite)

    assert "dual_task_scenario_suite_policy_neutrality_invalid" in excinfo.value.errors
    assert "dual_task_scenario_initial_scope_expanded" in excinfo.value.errors


def test_suite_rejects_index_only_or_runtime_unsupported_cousin() -> None:
    suite = _manifests()[0]
    cousin = next(
        cell for cell in suite["cells"] if cell["family"] == "admitted_object_cousin"
    )
    cousin.pop("execution_admission")
    cousin["resolved_parameters"] = {"object_cousin_index": 1}
    cousin["factor_records"] = [
        {
            "parameter_id": "object_cousin_index",
            "runtime_target": "AssetResolver.admitted_object_cousin.index",
            "unit": "index",
            "nominal_value": 0,
            "resolved_value": 1,
            "application_tolerance": 0.1,
        }
    ]
    cousin["applied_parameter_readback_required"] = True
    suite["suite_digest"] = canonical_digest(suite, digest_field="suite_digest")

    with pytest.raises(DualTaskScenarioSuiteError) as excinfo:
        validate_dual_task_scenario_suite(suite)
    assert "dual_task_scenario_cousin_admission_missing" in excinfo.value.errors

    cousin["execution_admission"] = {
        "status": "admitted",
        "blocker_code": None,
        "cousin_asset": {
            "asset_id": "exact_cousin",
            "sha256": "sha256:" + "1" * 64,
            "size_bytes": 123,
            "rights_receipt_digest": "sha256:" + "2" * 64,
        },
    }
    suite["suite_digest"] = canonical_digest(suite, digest_field="suite_digest")
    with pytest.raises(DualTaskScenarioSuiteError) as excinfo:
        validate_dual_task_scenario_suite(suite)
    assert "dual_task_scenario_cousin_asset_invalid" in excinfo.value.errors

    cousin["factor_records"][0]["runtime_target"] = (
        "EventManager.reset.object_start_position_m.y"
    )
    suite["suite_digest"] = canonical_digest(suite, digest_field="suite_digest")
    with pytest.raises(DualTaskScenarioSuiteError) as excinfo:
        validate_dual_task_scenario_suite(suite)
    assert "dual_task_scenario_cousin_asset_invalid" in excinfo.value.errors
