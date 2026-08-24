from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_scene_policy_readiness import (
    CONTROLS_PREDECESSOR,
    READY_VERDICT,
    ScenePolicyReadinessError,
    load_scene_policy_readiness,
    validate_scene_policy_readiness,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


MANIFESTS = Path(__file__).parents[1] / "docs/arm_decision_proof_v1/manifests"
REPORT = MANIFESTS / "adp009d_scene_840920_policy_readiness.v1.json"
SCENARIO = MANIFESTS / "third_scene_840920_task_a_scenario_suite.v1.json"


def _values() -> tuple[dict, dict]:
    return json.loads(REPORT.read_text()), json.loads(SCENARIO.read_text())


def test_committed_scene_policy_readiness_waits_only_for_controls() -> None:
    report = load_scene_policy_readiness(REPORT, scenario_suite_path=SCENARIO)

    assert report["candidate_ids"] == ["pi05_droid", "groot_n17_droid"]
    assert report["external_blockers"] == []
    assert report["verdict"] == READY_VERDICT
    assert report["controls_predecessor"]["blocker_code"] == CONTROLS_PREDECESSOR
    assert report["controls_predecessor"]["required_schema"] == (
        "native_task_arena_control_result.v1"
    )
    assert report["controls_predecessor"]["receipt_digest"] is None
    assert report["scenario_matrix"]["cell_count"] == 7
    assert report["scenario_matrix"]["seeds"] == [3101, 3102]
    assert all(
        candidate["observation_adapter_ready"]
        and candidate["action_adapter_ready"]
        and candidate["rights_ready"]
        and candidate["candidate_can_grade_itself"] is False
        for candidate in report["candidates"]
    )


def test_readiness_rejects_floating_checkpoint_or_controls_bypass() -> None:
    report, scenario = _values()
    report["candidates"][1]["checkpoint"]["revision"] = "main"
    report["controls_predecessor"]["bypass_permitted"] = True
    report["readiness_digest"] = canonical_digest(
        report, digest_field="readiness_digest"
    )

    with pytest.raises(ScenePolicyReadinessError) as excinfo:
        validate_scene_policy_readiness(report, scenario_suite=scenario)

    assert (
        "scene_policy_readiness_groot_n17_droid_checkpoint_invalid"
        in excinfo.value.errors
    )
    assert "scene_policy_readiness_controls_predecessor_invalid" in excinfo.value.errors


def test_readiness_rejects_candidate_whose_rights_are_not_ready() -> None:
    report, scenario = _values()
    report["candidates"][0]["rights_ready"] = False
    report["readiness_digest"] = canonical_digest(
        report, digest_field="readiness_digest"
    )

    with pytest.raises(ScenePolicyReadinessError) as excinfo:
        validate_scene_policy_readiness(report, scenario_suite=scenario)

    assert "scene_policy_readiness_pi05_droid_rights_ready_invalid" in (
        excinfo.value.errors
    )


def test_readiness_rejects_unavailable_checkpoint_or_unrehearsed_terminal() -> None:
    report, scenario = _values()
    report["candidates"][0]["checkpoint"]["checkpoint_ready"] = False
    report["candidates"][1]["terminal_rehearsal_status"] = "not_run"
    report["readiness_digest"] = canonical_digest(
        report, digest_field="readiness_digest"
    )

    with pytest.raises(ScenePolicyReadinessError) as excinfo:
        validate_scene_policy_readiness(report, scenario_suite=scenario)

    assert "scene_policy_readiness_pi05_droid_checkpoint_availability_invalid" in (
        excinfo.value.errors
    )
    assert "scene_policy_readiness_groot_n17_droid_terminal_rehearsal_invalid" in (
        excinfo.value.errors
    )


def test_readiness_rejects_a_scenario_target_without_runtime_application() -> None:
    report, scenario = _values()
    scenario["cells"][2]["factor_records"][0]["runtime_target"] = (
        "EventManager.reset.unimplemented_light.value"
    )
    scenario["suite_digest"] = canonical_digest(scenario, digest_field="suite_digest")
    report["scenario_suite_digest"] = scenario["suite_digest"]
    report["readiness_digest"] = canonical_digest(
        report, digest_field="readiness_digest"
    )

    with pytest.raises((ScenePolicyReadinessError, ValueError)):
        validate_scene_policy_readiness(report, scenario_suite=scenario)
