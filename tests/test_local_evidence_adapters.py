from __future__ import annotations

import pytest

from blueprint_pipeline.local_evidence_adapters import (
    ANALYTIC_REACHABILITY_ADAPTER,
    CAPTURED_VISIBILITY_ADAPTER,
    AnalyticReachabilityAdapter,
    CapturedVisibilityAdapter,
    authorized_local_evidence_adapter_registry,
)


def _testbed() -> dict:
    return {
        "robot_sensor_controller_bindings": {
            "embodiment": {
                "reach_envelope": {"minimum_m": 0.1, "maximum_m": 1.0},
            },
            "selected_robot_placement": {
                "candidate_id": "base-1",
                "base_position_site_m": [0.0, 0.0, 0.0],
                "captured_coverage": 0.95,
                "calibration_uncertainty_m": 0.01,
                "method_qualification_status": "analytic_only",
            },
        },
        "target_regions": [{
            "region_id": "tote-1",
            "position_site_m": [0.6, 0.1, 0.7],
            "supporting_frames": ["frame-2", "frame-1", "frame-1"],
            "captured_coverage": 0.9,
        }],
        "validation_envelope": {"robot_placement_digest": "sha256:" + "a" * 64},
    }


def test_analytic_reachability_uses_only_explicit_metric_inputs() -> None:
    result = AnalyticReachabilityAdapter().execute(
        claim={
            "claim_type": "analytic_reachability",
            "subject": "tote-1",
        },
        testbed=_testbed(),
    )
    assert result["status"] == "valid"
    assert result["supports_claim"] is True
    assert result["claim_ceiling"]["physical_success"] is False
    assert result["provenance"]["physical_robot_run_initiated"] is False

    missing = _testbed()
    missing["robot_sensor_controller_bindings"]["selected_robot_placement"].pop(
        "base_position_site_m"
    )
    abstention = AnalyticReachabilityAdapter().execute(
        claim={
            "claim_type": "analytic_reachability",
            "subject": "tote-1",
        },
        testbed=missing,
    )
    assert abstention["status"] == "unavailable"
    assert abstention["supports_claim"] is None
    assert "robot_base_metric_position_missing" in abstention["blockers"]


def test_analytic_reachability_abstains_at_uncertain_boundary() -> None:
    result = AnalyticReachabilityAdapter().execute(
        claim={
            "claim_type": "reachability",
            "subject": {"target_position_site_m": [0.995, 0.0, 0.0]},
        },
        testbed=_testbed(),
    )
    assert result["status"] == "uncertain"
    assert result["supports_claim"] is None
    assert result["categorical_finding"] == "reach_boundary_uncertain"


def test_captured_visibility_binds_exact_region_and_retained_frames() -> None:
    result = CapturedVisibilityAdapter().execute(
        claim={
            "claim_type": "captured_visibility",
            "subject": {"target_region_id": "tote-1"},
        },
        testbed=_testbed(),
    )
    assert result["status"] == "valid"
    assert result["coverage"] == 0.9
    assert result["raw_artifact_references"] == [
        {"uri": "capture-frame://frame-1", "frame_id": "frame-1"},
        {"uri": "capture-frame://frame-2", "frame_id": "frame-2"},
    ]
    assert result["claim_ceiling"]["metric_geometry"] is False

    missing = CapturedVisibilityAdapter().execute(
        claim={
            "claim_type": "captured_visibility",
            "subject": {"target_region_id": "occluded-region"},
        },
        testbed=_testbed(),
    )
    assert missing["status"] == "unavailable"
    assert missing["supports_claim"] is None


def test_local_registry_is_empty_by_default_and_rejects_unknown_authority() -> None:
    empty = authorized_local_evidence_adapter_registry([])
    assert empty.manifest() == []
    authorized = authorized_local_evidence_adapter_registry([
        CAPTURED_VISIBILITY_ADAPTER,
        ANALYTIC_REACHABILITY_ADAPTER,
    ])
    assert authorized.manifest() == [
        ANALYTIC_REACHABILITY_ADAPTER,
        CAPTURED_VISIBILITY_ADAPTER,
    ]
    with pytest.raises(ValueError, match="local_evidence_adapter_not_registered"):
        authorized_local_evidence_adapter_registry(["provider://live-not-authorized"])
