from __future__ import annotations

import copy
import math

import pytest

from blueprint_pipeline.deformable_transfer_scoring import (
    DeformableTransferScoringError,
    OUTCOME_LADDER,
    score_deformable_transfer,
)


DEFORMABLE_ID = "task_towel"
DESTINATION_ID = "task_basket"
ROBOT_ID = "franka"


def _yaw_quaternion(radians: float) -> list[float]:
    return [0.0, 0.0, math.sin(radians / 2.0), math.cos(radians / 2.0)]


def _task_spec() -> dict:
    return {
        "deformable_entity_id": DEFORMABLE_ID,
        "destination_entity_id": DESTINATION_ID,
        "robot_entity_id": ROBOT_ID,
        "destination_interior_obb": {
            "center_world_m": [1.0, 2.0, 0.5],
            "half_extents_m": [0.5, 0.4, 0.3],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "receptacle_reference_pose_world": {
            "position_m": [1.0, 2.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "minimum_particle_fraction_inside": 0.75,
        "settle_window_samples": 3,
        "maximum_node_speed_mps": 0.02,
        "maximum_principal_strain": 0.25,
        "minimum_grasp_contact_force_n": 0.1,
        "maximum_release_contact_force_n": 0.0,
        "minimum_robot_clearance_m": 0.15,
        "maximum_receptacle_translation_drift_m": 0.01,
        "maximum_receptacle_rotation_drift_rad": 0.03,
        "maximum_receptacle_linear_speed_mps": 0.01,
        "maximum_receptacle_angular_speed_radps": 0.03,
    }


def _sample(sample_index: int) -> dict:
    positions = [
        [0.8, 1.8, 0.45],
        [1.2, 1.8, 0.45],
        [0.8, 2.2, 0.55],
        [1.2, 2.2, 0.55],
    ]
    return {
        "sample_index": sample_index,
        "time_seconds": 0.1 * sample_index,
        "entities": {
            DEFORMABLE_ID: {
                "nodal_positions_world_m": positions,
                "nodal_velocities_world_mps": [[0.0, 0.0, 0.0] for _ in positions],
                "deformation_gradients": [
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ],
                "nodal_kinematic_flags": [1.0 for _ in positions],
                "state_write_count_after_episode_start": 0,
                "solver_divergence_count": 0,
            },
            DESTINATION_ID: {
                "pose_world": {
                    "position_m": [1.0, 2.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "linear_velocity_world_mps": [0.0, 0.0, 0.0],
                "angular_velocity_world_radps": [0.0, 0.0, 0.0],
            },
            ROBOT_ID: {
                "gripper_clearance_points_world_m": [[2.0, 3.0, 1.5]],
                "gripper_contact_pair_count_by_entity_id": {DEFORMABLE_ID: 0},
                "gripper_contact_normal_force_n_by_entity_id": {
                    DEFORMABLE_ID: 0.0
                },
            },
        },
    }


def _samples() -> list[dict]:
    samples = [_sample(index) for index in range(4)]
    grasp_robot = samples[0]["entities"][ROBOT_ID]
    grasp_robot["gripper_contact_pair_count_by_entity_id"][DEFORMABLE_ID] = 2
    grasp_robot["gripper_contact_normal_force_n_by_entity_id"][DEFORMABLE_ID] = 0.2
    return samples


def _score(samples: list[dict], *, task_spec: dict | None = None) -> dict:
    return score_deformable_transfer(
        task_spec=task_spec or _task_spec(),
        samples=samples,
    )


def test_positive_uses_raw_entity_state_and_reaches_terminal_ladder_rung() -> None:
    result = _score(_samples())

    assert result["deterministic_success"] is True
    assert result["outcome"] == "succeeded"
    assert result["outcome_ladder"] == list(OUTCOME_LADDER)
    assert result["ladder_truncated_at"] is None
    assert result["failure_reasons"] == []
    assert result["measurements"]["particle_fraction_inside"] == 1.0
    assert result["measurements"]["centroid_inside"] is True
    assert result["measurements"]["maximum_grasp_contact_pair_count"] == 2
    assert result["predicates"]["grasp_contact_observed"] is True
    assert result["result_digest"].startswith("sha256:")


def test_destination_obb_boundary_is_inclusive() -> None:
    spec = _task_spec()
    spec["destination_interior_obb"] = {
        "center_world_m": [0.0, 0.0, 0.0],
        "half_extents_m": [1.0, 0.5, 0.25],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    samples = _samples()
    boundary_positions = [
        [1.0, 0.5, 0.25],
        [-1.0, -0.5, -0.25],
        [1.0, -0.5, 0.25],
        [-1.0, 0.5, -0.25],
    ]
    for sample in samples:
        sample["entities"][DEFORMABLE_ID][
            "nodal_positions_world_m"
        ] = boundary_positions
        sample["entities"][ROBOT_ID]["gripper_clearance_points_world_m"] = [
            [3.0, 3.0, 3.0]
        ]

    result = _score(samples, task_spec=spec)

    assert result["measurements"]["particle_fraction_inside"] == 1.0
    assert result["measurements"]["centroid_inside"] is True
    assert result["deterministic_success"] is True


def test_rotated_destination_obb_is_evaluated_in_its_local_frame() -> None:
    spec = _task_spec()
    spec["destination_interior_obb"] = {
        "center_world_m": [1.0, 2.0, 0.5],
        "half_extents_m": [0.9, 0.2, 0.2],
        "orientation_xyzw": _yaw_quaternion(math.pi / 2.0),
    }
    # These nodes extend primarily along world Y.  They fit only after applying
    # the frozen 90-degree OBB rotation; the equivalent axis-aligned box fails.
    rotated_positions = [
        [0.9, 1.2, 0.45],
        [1.1, 1.2, 0.55],
        [0.9, 2.8, 0.45],
        [1.1, 2.8, 0.55],
    ]
    samples = _samples()
    for sample in samples:
        sample["entities"][DEFORMABLE_ID][
            "nodal_positions_world_m"
        ] = rotated_positions

    rotated = _score(samples, task_spec=spec)
    axis_aligned_spec = copy.deepcopy(spec)
    axis_aligned_spec["destination_interior_obb"]["orientation_xyzw"] = [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    axis_aligned = _score(samples, task_spec=axis_aligned_spec)

    assert rotated["measurements"]["particle_fraction_inside"] == 1.0
    assert rotated["deterministic_success"] is True
    assert axis_aligned["measurements"]["particle_fraction_inside"] == 0.0
    assert axis_aligned["deterministic_success"] is False


def test_nan_native_state_is_retained_as_non_success_not_a_caller_failure_flag() -> None:
    samples = _samples()
    samples[-1]["entities"][DEFORMABLE_ID]["nodal_positions_world_m"][0][0] = (
        float("nan")
    )

    result = _score(samples)

    assert result["deterministic_success"] is False
    assert result["predicates"]["finite_without_divergence"] is False
    assert result["measurements"]["particle_fraction_inside"] is None
    assert "non_finite_or_invalid_native_state" in result["failure_reasons"]


def test_native_solver_divergence_is_not_misclassified_as_task_failure() -> None:
    samples = _samples()
    samples[1]["entities"][DEFORMABLE_ID]["solver_divergence_count"] = 1

    result = _score(samples)

    assert result["measurements"]["solver_divergence_count"] == 1
    assert result["predicates"]["finite_without_divergence"] is False
    assert "solver_divergence_observed" in result["failure_reasons"]


def test_insufficient_particle_containment_stops_the_ladder() -> None:
    spec = _task_spec()
    spec["minimum_particle_fraction_inside"] = 1.0
    samples = _samples()
    samples[-1]["entities"][DEFORMABLE_ID]["nodal_positions_world_m"][0] = [
        3.0,
        3.0,
        3.0,
    ]

    result = _score(samples, task_spec=spec)

    assert result["measurements"]["particle_fraction_inside"] == 0.75
    assert result["predicates"]["contained"] is False
    assert result["ladder_truncated_at"] == "contained"
    assert "insufficient_particle_containment" in result["failure_reasons"]


def test_centroid_must_be_inside_even_when_particle_fraction_passes() -> None:
    samples = _samples()
    samples[-1]["entities"][DEFORMABLE_ID]["nodal_positions_world_m"][0] = [
        4.0,
        2.0,
        0.5,
    ]

    result = _score(samples)

    assert result["measurements"]["particle_fraction_inside"] == 0.75
    assert result["measurements"]["centroid_inside"] is False
    assert result["predicates"]["contained"] is False
    assert "centroid_outside_destination" in result["failure_reasons"]


def test_one_transient_velocity_fails_the_full_settle_window() -> None:
    samples = _samples()
    samples[1]["entities"][DEFORMABLE_ID]["nodal_velocities_world_mps"][2] = [
        0.03,
        0.0,
        0.0,
    ]

    result = _score(samples)

    assert result["measurements"]["maximum_settle_node_speed_mps"] == 0.03
    assert result["predicates"]["settled"] is False
    assert "settle_velocity_exceeded" in result["failure_reasons"]


def test_maximum_principal_strain_is_derived_from_deformation_gradient() -> None:
    samples = _samples()
    samples[0]["entities"][DEFORMABLE_ID]["deformation_gradients"][0] = [
        [1.4, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    result = _score(samples)

    assert result["measurements"][
        "maximum_absolute_principal_engineering_strain"
    ] == pytest.approx(0.4)
    assert result["predicates"]["strain_within_bound"] is False
    assert "maximum_principal_strain_exceeded" in result["failure_reasons"]


def test_release_requires_zero_gripper_pairs_and_force_throughout_settle() -> None:
    samples = _samples()
    robot = samples[1]["entities"][ROBOT_ID]
    robot["gripper_contact_pair_count_by_entity_id"][DEFORMABLE_ID] = 1
    robot["gripper_contact_normal_force_n_by_entity_id"][DEFORMABLE_ID] = 0.2

    result = _score(samples)

    assert result["measurements"]["release_contact_pair_count"] == 1
    assert result["measurements"]["maximum_release_contact_force_n"] == 0.2
    assert result["predicates"]["released"] is False
    assert "gripper_contact_not_released" in result["failure_reasons"]


def test_success_requires_prior_qualified_gripper_contact() -> None:
    samples = _samples()
    robot = samples[0]["entities"][ROBOT_ID]
    robot["gripper_contact_pair_count_by_entity_id"][DEFORMABLE_ID] = 0
    robot["gripper_contact_normal_force_n_by_entity_id"][DEFORMABLE_ID] = 0.0

    result = _score(samples)

    assert result["deterministic_success"] is False
    assert result["predicates"]["grasp_contact_observed"] is False
    assert result["ladder_truncated_at"] == "grasp_contact_observed"
    assert "qualified_gripper_deformable_contact_not_observed" in result[
        "failure_reasons"
    ]


def test_robot_retreat_is_computed_from_clearance_points_and_geometry() -> None:
    samples = _samples()
    for sample in samples:
        sample["entities"][ROBOT_ID]["gripper_clearance_points_world_m"] = [
            [1.0, 2.0, 0.5]
        ]

    result = _score(samples)

    assert result["measurements"]["minimum_robot_clearance_m"] == 0.0
    assert result["predicates"]["robot_retreated"] is False
    assert "robot_retreat_clearance_not_met" in result["failure_reasons"]


def test_receptacle_pose_drift_fails_stability() -> None:
    samples = _samples()
    samples[-1]["entities"][DESTINATION_ID]["pose_world"]["position_m"] = [
        1.02,
        2.0,
        0.0,
    ]

    result = _score(samples)

    assert result["measurements"][
        "maximum_receptacle_translation_drift_m"
    ] == pytest.approx(0.02)
    assert result["predicates"]["receptacle_stable"] is False
    assert "receptacle_pose_or_drift_unstable" in result["failure_reasons"]


def test_post_start_direct_write_is_an_integrity_failure() -> None:
    samples = _samples()
    samples[1]["entities"][DEFORMABLE_ID][
        "state_write_count_after_episode_start"
    ] = 1

    result = _score(samples)

    assert result["deterministic_success"] is False
    assert result["predicates"]["integrity_preserved"] is False
    assert result["ladder_truncated_at"] == "integrity_preserved"
    assert result["failure_reasons"][0] == "post_start_direct_state_write_observed"


def test_native_kinematic_flag_cannot_hide_an_attachment() -> None:
    samples = _samples()
    samples[0]["entities"][DEFORMABLE_ID]["nodal_kinematic_flags"][0] = 0.0

    result = _score(samples)

    assert result["measurements"]["maximum_kinematic_node_count"] == 1
    assert result["predicates"]["integrity_preserved"] is False
    assert "kinematic_attachment_observed" in result["failure_reasons"]


def test_caller_boolean_cannot_substitute_for_native_contact_readback() -> None:
    samples = _samples()
    samples[0]["entities"][ROBOT_ID][
        "gripper_contact_normal_force_n_by_entity_id"
    ][DEFORMABLE_ID] = False

    with pytest.raises(DeformableTransferScoringError) as exc_info:
        _score(samples)

    assert "deformable_transfer_gripper_contact_force_invalid:0" in exc_info.value.errors
