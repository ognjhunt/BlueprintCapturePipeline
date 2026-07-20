from __future__ import annotations

from copy import deepcopy

from blueprint_pipeline.oscar_action_control_contracts import (
    EXECUTED_CONTROL_KINDS,
    REJECTION_CONTROL_KINDS,
    SCHEMA_VERSION,
    validate_oscar_action_control_suite,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _suite() -> dict:
    controls = []
    for index, kind in enumerate(sorted(EXECUTED_CONTROL_KINDS), 1):
        controls.append(
            {
                "control_kind": kind,
                "control_action_sha256": _digest(100 + index),
                "transformation_or_replay_condition_verified": True,
                "status": "fresh_counterfactual_completed",
                "fresh_official_oscar_model_execution_proven": True,
                "fresh_oscar_provider_model_run_steps": 1,
                "skeleton_conditioning_sha256": _digest(200 + index),
                "model_output_sha256": _digest(300 + index),
                "provider_execution_sha256": _digest(400 + index),
                "next_policy_query_sha256": _digest(500 + index),
            }
        )
    for index, kind in enumerate(sorted(REJECTION_CONTROL_KINDS), 1):
        controls.append(
            {
                "control_kind": kind,
                "control_action_sha256": _digest(600 + index),
                "transformation_or_replay_condition_verified": True,
                "status": "admission_rejected_before_ranking",
                "rejection_blocker": f"{kind}_action_control_rejected",
                "decision_grade_eligible": False,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "base_commanded_action_sha256": _digest(1),
        "base_skeleton_conditioning_sha256": _digest(2),
        "base_model_output_sha256": _digest(3),
        "oscar_checkpoint_sha256": _digest(4),
        "provider_execution_manifest_sha256": _digest(5),
        "base_execution_fresh_official_oscar_model": True,
        "controls_are_excluded_from_decision_rows": True,
        "controls": controls,
    }


def test_oscar_action_control_suite_covers_executed_and_rejected_controls() -> None:
    result = validate_oscar_action_control_suite(_suite())

    assert result["status"] == "passed"
    assert result["executed_control_count"] == 4
    assert result["rejection_control_count"] == 3


def test_oscar_action_control_suite_blocks_reused_outputs_and_admitted_stale_actions() -> None:
    candidate = deepcopy(_suite())
    candidate["controls"][0]["model_output_sha256"] = candidate["base_model_output_sha256"]
    stale = next(row for row in candidate["controls"] if row["control_kind"] == "stale")
    stale["status"] = "fresh_counterfactual_completed"
    stale["decision_grade_eligible"] = True

    result = validate_oscar_action_control_suite(candidate)

    assert result["status"] == "blocked"
    assert "oscar_action_control_model_output_reused:0" in result["blockers"]
    assert any("oscar_action_replay_control_not_rejected" in item for item in result["blockers"])
    assert any("oscar_action_replay_control_not_decision_blocked" in item for item in result["blockers"])


def test_oscar_action_controls_normalize_digests_before_reuse_checks() -> None:
    candidate = deepcopy(_suite())
    candidate["controls"][0]["control_action_sha256"] = (
        "sha256:" + candidate["base_commanded_action_sha256"]
    )
    candidate["controls"][1]["model_output_sha256"] = (
        "sha256:" + candidate["base_model_output_sha256"]
    )

    result = validate_oscar_action_control_suite(candidate)

    assert result["status"] == "blocked"
    assert "oscar_action_control_action_not_distinct_from_base:0" in result["blockers"]
    assert "oscar_action_control_model_output_reused:1" in result["blockers"]
