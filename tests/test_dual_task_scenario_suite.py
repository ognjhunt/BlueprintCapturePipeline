from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.dual_task_scenario_suite import (
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


def test_suite_rejects_outcome_leakage_or_initial_matrix_expansion() -> None:
    suite = _manifests()[0]
    suite["learned_policy_outcomes_consulted"] = True
    suite["cells"][1]["scheduled_initially"] = True
    suite["suite_digest"] = canonical_digest(suite, digest_field="suite_digest")

    with pytest.raises(DualTaskScenarioSuiteError) as excinfo:
        validate_dual_task_scenario_suite(suite)

    assert "dual_task_scenario_suite_policy_neutrality_invalid" in excinfo.value.errors
    assert "dual_task_scenario_initial_scope_expanded" in excinfo.value.errors
