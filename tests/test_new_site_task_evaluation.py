from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from blueprint_pipeline.capture_intake import CaptureIntakeError
from blueprint_pipeline.new_site_task_evaluation import (
    NewSiteTaskEvaluationError,
    run_new_site_task_evaluation,
)

FIXTURE = Path(__file__).parent / "fixtures" / "new_site_loading_bay_v1"


def _claim_plan(result: dict[str, object]) -> dict[str, object]:
    return result["production_evidence_plan"]["claim_plans"][0]  # type: ignore[index]


def _copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "new-site-fixture"
    shutil.copytree(FIXTURE, target)
    return target


def test_complete_new_site_runs_development_evaluation_without_upgrading_claims(
    tmp_path: Path,
) -> None:
    result = run_new_site_task_evaluation(
        fixture_root=FIXTURE,
        state_root=tmp_path / "state",
    )

    assert result["terminal_state"] == "decided"
    assert result["capture_materialization_validation"]["all_declared_raw_hashes_verified"]
    assert result["capture_materialization_validation"]["frame_count"] == 2
    assert result["task_specification"]["task_spec_digest"].startswith("sha256:")
    assert result["task_specification"]["target_region"]["observation_manifest_digest"].startswith(
        "sha256:"
    )
    assert result["site_evidence_audit"]["gap_count"] == 0
    site_evidence = result["site_evidence_profile"]["evidence"]
    required_observed_evidence = {
        "articulation_actuation",
        "articulation_model",
        "calibrated_rgb",
        "camera_poses",
        "friction_contact",
        "mass_inertia",
        "material_parameters",
        "metric_scale",
        "robot_site_registration",
        "sensor_calibration",
        "sensor_timing",
        "validated_collider",
        "validated_mesh",
    }
    assert required_observed_evidence == set(site_evidence)
    assert all(site_evidence[key]["available"] for key in required_observed_evidence)
    assert all(site_evidence[key]["validated"] for key in required_observed_evidence)
    assert result["site_evidence_compilation_report"]["fabricated_records"] == 0
    assert result["site_evidence_compilation_report"]["unmapped_artifacts"] == []
    assert _claim_plan(result)["status"] == "abstention_planned"
    assert _claim_plan(result)["next_cheapest_experiment"] == "qualification_benchmark"
    assert result["development_route"]["status"] == "development_route_selected"
    assert result["development_route"]["candidate_rejection_codes"] == [
        "no_exact_verified_qualification"
    ]
    assert result["development_route"]["candidate_capability_digest_matches"]
    assert result["execution_manifest"]["status"] == "completed"
    assert result["execution_manifest"]["registered_adapters"] == ["local://captured-visibility-v1"]
    assert result["execution_manifest"]["provider_discovery_from_defaults"] is False
    assert result["execution_manifest"]["physical_robot_run_initiated"] is False
    assert len(result["evidence_results"]) == 1
    assert result["evidence_results"][0]["status"] == "valid"
    assert result["decision_envelope"]["overall_outcome"] == "decision"
    assert result["decision_envelope"]["per_claim_verdicts"][0]["verdict"] == "supported"
    assert all(result["digest_joins"].values())
    assert result["proof_boundary"] == {
        "development_evidence_only": True,
        "capture_evidence_tier": "fixture_only",
        "control_plane_fixture_only": True,
        "production_measurement_route_selected": False,
        "r7_catalog_entry_created": False,
        "physical_task_success": False,
        "deployment_readiness": False,
        "safety_certification": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
        "paid_compute_used": False,
    }
    assert result["webapp_projection"]["raw_paths_included"] is False
    assert result["webapp_projection"]["raw_frames_included"] is False
    assert result["webapp_projection"]["credentials_included"] is False


def test_incomplete_new_site_abstains_at_smallest_missing_measurement(
    tmp_path: Path,
) -> None:
    result = run_new_site_task_evaluation(
        fixture_root=FIXTURE,
        state_root=tmp_path / "state",
        site_artifacts_name="site_evidence_incomplete.json",
    )

    claim_plan = _claim_plan(result)
    routing = claim_plan["measurement_routing_decision"]
    next_action = routing["abstention"]["smallest_next_action"]
    assert result["terminal_state"] == "abstained"
    assert result["site_evidence_audit"]["gap_count"] == 2
    assert {row["evidence_id"] for row in result["site_evidence_audit"]["gaps"]} == {
        "sensor_calibration",
        "sensor_timing",
    }
    assert next_action["action_type"] == "sensor_calibration"
    assert next_action["exact_scope"] == ["sensor_calibration", "sensor_timing"]
    assert result["development_route"]["status"] == "development_route_abstained"
    assert result["execution_manifest"] is None
    assert result["evidence_results"] == []
    assert result["decision_envelope"]["overall_outcome"] == "abstention"
    assert result["decision_envelope"]["next_cheapest_experiment"] == "sensor_calibration"
    assert all(result["digest_joins"].values())


def test_raw_capture_tamper_fails_before_site_audit(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    (fixture / "raw" / "frame-000.rgb.bin").write_bytes(b"tampered")

    with pytest.raises(CaptureIntakeError, match="(?:size|digest)_mismatch"):
        run_new_site_task_evaluation(
            fixture_root=fixture,
            state_root=tmp_path / "state",
        )


def test_observed_site_evidence_digest_tamper_fails_closed(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    path = fixture / "site_evidence_complete.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["artifacts"]["sensor_calibration"]["calibration_id"] = "forged"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        NewSiteTaskEvaluationError,
        match="digest_mismatch:artifact_digest",
    ):
        run_new_site_task_evaluation(
            fixture_root=fixture,
            state_root=tmp_path / "state",
        )


def test_task_target_cannot_reference_an_unobserved_frame(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    task_path = fixture / "task_spec.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["target_region"]["supporting_frames"].append("invented-frame")
    task_path.write_text(json.dumps(task), encoding="utf-8")

    with pytest.raises(
        NewSiteTaskEvaluationError,
        match="task_target_supporting_frames_not_observed",
    ):
        run_new_site_task_evaluation(
            fixture_root=fixture,
            state_root=tmp_path / "state",
        )


def test_supervisor_cannot_lower_requirements_or_substitute_route(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    task_path = fixture / "task_spec.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["development_method"]["declared_measurement_capabilities"].remove(
        "sensor_timing_supported"
    )
    task_path.write_text(json.dumps(task), encoding="utf-8")

    result = run_new_site_task_evaluation(
        fixture_root=fixture,
        state_root=tmp_path / "state",
    )
    supervisor = result["supervisor_proposals"]
    missing_capabilities = result["development_route"]["missing_required_capabilities"]

    assert supervisor["agent_may_lower_requirements"] is False
    assert supervisor["agent_may_forge_qualification"] is False
    assert supervisor["agent_may_authorize_spend"] is False
    assert supervisor["agent_may_substitute_method"] is False
    assert all(row["authoritative"] is False for row in supervisor["results"])
    assert all(row["proof_effect"] == "none" for row in supervisor["results"])
    assert missing_capabilities == ["sensor_timing_supported"]
    assert (
        "required_capability_missing:sensor_timing_supported"
        in result["development_route"]["blockers"]
    )
    assert result["development_route"]["status"] == "development_route_abstained"
    assert result["execution_manifest"] is None


def test_result_write_is_digest_stable_and_idempotent(tmp_path: Path) -> None:
    state = tmp_path / "state"
    first = run_new_site_task_evaluation(fixture_root=FIXTURE, state_root=state)
    second = run_new_site_task_evaluation(fixture_root=FIXTURE, state_root=state)

    assert first["result_digest"] == second["result_digest"]
    run_files = list((state / "runs" / first["run_id"]).glob("*.json"))
    assert len(run_files) == 1
    assert json.loads(run_files[0].read_text(encoding="utf-8")) == first
