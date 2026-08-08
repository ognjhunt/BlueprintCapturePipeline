from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.articulated_workspace_clearance import (
    ArticulatedWorkspaceClearanceError,
    evaluate_revolute_member_sweep,
    validate_articulated_workspace_clearance,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


ROOT = Path(__file__).resolve().parents[1]
REJECTION = (
    ROOT
    / "docs/arm_decision_proof_v1/manifests"
    / "second_scene_candidate_840411_clearance_rejection.v1.json"
)


def _obstacle(*, minimum: list[float], maximum: list[float], obstacle_id: str = "chair") -> dict:
    return {
        "obstacle_id": obstacle_id,
        "world_aabb_min_m": minimum,
        "world_aabb_max_m": maximum,
        "source_receipt_digest": "sha256:fixture",
    }


def test_original_rigid_fixture_has_no_articulated_sweep_contract() -> None:
    with pytest.raises(ArticulatedWorkspaceClearanceError) as caught:
        evaluate_revolute_member_sweep(
            hinge_origin_world_m=[0.0, 0.0, 0.0],
            closed_endpoint_world_m=[0.0, 0.0, 0.0],
            member_vertical_interval_m=[0.0, 1.0],
            start_angle_degrees=0.0,
            end_angle_degrees=45.0,
            obstacles=[],
        )

    assert caught.value.errors == ("sweep_member_radius_invalid",)


def test_840411_right_door_centerline_hits_chair_before_45_degrees() -> None:
    result = evaluate_revolute_member_sweep(
        hinge_origin_world_m=[4.475898768, 1.452210456, 0.0],
        closed_endpoint_world_m=[4.475898768, 1.9413037, 0.0],
        member_vertical_interval_m=[0.0, 1.826109993],
        start_angle_degrees=0.0,
        end_angle_degrees=45.0,
        obstacles=[
            _obstacle(
                obstacle_id="chair:227",
                minimum=[3.697225536, 1.345660441, 0.001594507],
                maximum=[4.274605183, 1.871812323, 0.856457869],
            )
        ],
        angular_resolution_degrees=0.25,
        member_half_thickness_m=0.0,
    )

    assert result["status"] == "blocked_by_observed_obstacle"
    assert result["first_collision"]["obstacle_id"] == "chair:227"
    assert 25.0 < result["first_collision"]["angle_degrees"] < 27.0
    assert result["first_collision"]["angle_degrees"] < 45.0
    assert result["claim_boundary"][
        "zero_thickness_centerline_collision_is_strong_rejection"
    ]
    assert validate_articulated_workspace_clearance(result) == result


def test_clear_sweep_remains_candidate_only() -> None:
    result = evaluate_revolute_member_sweep(
        hinge_origin_world_m=[0.0, 0.0, 0.0],
        closed_endpoint_world_m=[0.0, 0.5, 0.0],
        member_vertical_interval_m=[0.0, 1.0],
        start_angle_degrees=0.0,
        end_angle_degrees=45.0,
        obstacles=[
            _obstacle(minimum=[2.0, 2.0, 0.0], maximum=[3.0, 3.0, 1.0])
        ],
    )

    assert result["status"] == "clearance_candidate_only"
    assert result["first_collision"] is None
    assert result["claim_boundary"]["ik_or_contact_qualified"] is False


def test_840411_left_door_negative_sweep_hits_other_chair() -> None:
    result = evaluate_revolute_member_sweep(
        hinge_origin_world_m=[4.475898768, 2.430396944, 0.0],
        closed_endpoint_world_m=[4.475898768, 1.9413037, 0.0],
        member_vertical_interval_m=[0.0, 1.826109993],
        start_angle_degrees=0.0,
        end_angle_degrees=-45.0,
        obstacles=[
            _obstacle(
                obstacle_id="chair:226",
                minimum=[3.697225536, 1.980155103, 0.001594507],
                maximum=[4.274605183, 2.506306986, 0.856457869],
            )
        ],
    )

    assert result["status"] == "blocked_by_observed_obstacle"
    assert -25.0 < result["first_collision"]["angle_degrees"] < -24.0


def test_vertical_separation_does_not_false_positive() -> None:
    result = evaluate_revolute_member_sweep(
        hinge_origin_world_m=[0.0, 0.0, 2.0],
        closed_endpoint_world_m=[0.0, 0.5, 2.0],
        member_vertical_interval_m=[2.0, 3.0],
        start_angle_degrees=0.0,
        end_angle_degrees=90.0,
        obstacles=[
            _obstacle(minimum=[-1.0, -1.0, 0.0], maximum=[1.0, 1.0, 1.0])
        ],
    )

    assert result["status"] == "clearance_candidate_only"


def test_checked_rejection_binds_both_chairs_and_resumes_candidate_order() -> None:
    rejection = json.loads(REJECTION.read_text(encoding="utf-8"))

    validate_articulated_workspace_clearance(rejection["right_door_sweep"])
    validate_articulated_workspace_clearance(rejection["left_door_sweep"])
    assert rejection["status"] == "rejected_before_task_freeze"
    assert rejection["selection_effect"] == "resume_frozen_candidate_order_at_840796"
    assert rejection["learned_policy_outcomes_accessed"] is False
    assert rejection["record_digest"] == canonical_digest(
        rejection, digest_field="record_digest"
    )
