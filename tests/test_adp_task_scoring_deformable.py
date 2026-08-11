from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.adp_task_scoring import (
    TASK_KIND_DEFORMABLE_TRANSFER,
    TASK_SPEC_SCHEMA_VERSION,
    TaskNeutralScoringError,
    score_task_episode_from_spec,
    validate_deformable_task_spec,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


DEFORMABLE_ID = "deformable"
DESTINATION_ID = "receptacle"
ROBOT_ID = "robot"


def _task_spec() -> dict:
    return {
        "schema_version": TASK_SPEC_SCHEMA_VERSION,
        "task_kind": TASK_KIND_DEFORMABLE_TRANSFER,
        "prompt": "Transfer the deformable into the receptacle and retreat.",
        "deformable_entity_id": DEFORMABLE_ID,
        "destination_entity_id": DESTINATION_ID,
        "robot_entity_id": ROBOT_ID,
        "destination_interior_obb": {
            "center_world_m": [0.0, 0.0, 0.5],
            "half_extents_m": [0.5, 0.5, 0.5],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "receptacle_reference_pose_world": {
            "position_m": [0.0, 0.0, 0.0],
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
        "control_frequency_hz": 15,
        "maximum_action_steps": 20,
    }


def _sample(sample_index: int, *, contained: bool = True) -> dict:
    positions = (
        [
            [-0.2, -0.2, 0.4],
            [0.2, -0.2, 0.4],
            [-0.2, 0.2, 0.6],
            [0.2, 0.2, 0.6],
        ]
        if contained
        else [
            [1.4, -0.2, 0.4],
            [1.8, -0.2, 0.4],
            [1.4, 0.2, 0.6],
            [1.8, 0.2, 0.6],
        ]
    )
    return {
        "sample_index": sample_index,
        "time_seconds": sample_index * 0.1,
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
                    "position_m": [0.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "linear_velocity_world_mps": [0.0, 0.0, 0.0],
                "angular_velocity_world_radps": [0.0, 0.0, 0.0],
            },
            ROBOT_ID: {
                "gripper_clearance_points_world_m": [[2.0, 2.0, 2.0]],
                "gripper_contact_pair_count_by_entity_id": {DEFORMABLE_ID: 0},
                "gripper_contact_normal_force_n_by_entity_id": {DEFORMABLE_ID: 0.0},
            },
        },
    }


def _samples(*, contained: bool = True) -> list[dict]:
    samples = [_sample(index, contained=contained) for index in range(4)]
    robot = samples[0]["entities"][ROBOT_ID]
    robot["gripper_contact_pair_count_by_entity_id"][DEFORMABLE_ID] = 1
    robot["gripper_contact_normal_force_n_by_entity_id"][DEFORMABLE_ID] = 0.2
    return samples


def _dispatch(samples: list[dict]) -> dict:
    return score_task_episode_from_spec(task_spec=_task_spec(), samples=samples)


def test_deformable_zero_action_is_a_scored_deterministic_negative() -> None:
    samples = _samples(contained=False)
    robot = samples[0]["entities"][ROBOT_ID]
    robot["gripper_contact_pair_count_by_entity_id"][DEFORMABLE_ID] = 0
    robot["gripper_contact_normal_force_n_by_entity_id"][DEFORMABLE_ID] = 0.0
    report = _dispatch(samples)

    assert report["status"] == "scored"
    assert report["task_succeeded"] is False
    assert report["outcome"] == "finite_without_divergence"
    assert report["outcome_rank"] == 2


def test_deformable_scripted_positive_uses_the_common_result_contract() -> None:
    report = _dispatch(_samples())

    assert report["status"] == "scored"
    assert report["task_kind"] == TASK_KIND_DEFORMABLE_TRANSFER
    assert report["task_succeeded"] is True
    assert report["outcome"] == "succeeded"
    assert report["outcome_rank"] == 10
    assert report["result_digest"] == canonical_digest(report, digest_field="result_digest")


@pytest.mark.parametrize("native_failure", ["nan", "divergence"])
def test_deformable_nan_or_divergence_cannot_be_scored_as_success(
    native_failure: str,
) -> None:
    samples = _samples()
    deformable = samples[1]["entities"][DEFORMABLE_ID]
    if native_failure == "nan":
        deformable["nodal_velocities_world_mps"][0][0] = float("nan")
    else:
        deformable["solver_divergence_count"] = 1

    report = _dispatch(samples)

    assert report["status"] == "scored"
    assert report["task_succeeded"] is False
    assert report["predicates"]["finite_without_divergence"] is False
    assert report["outcome"] == "integrity_preserved"
    assert report["outcome_rank"] == 1


@pytest.mark.parametrize("integrity_failure", ["post_start_write", "attachment"])
def test_deformable_post_start_write_or_hidden_attachment_is_rejected(
    integrity_failure: str,
) -> None:
    samples = _samples()
    deformable = samples[1]["entities"][DEFORMABLE_ID]
    if integrity_failure == "post_start_write":
        deformable["state_write_count_after_episode_start"] = 1
    else:
        deformable["nodal_kinematic_flags"][0] = 0.0

    report = _dispatch(samples)

    assert report["status"] == "scored"
    assert report["task_succeeded"] is False
    assert report["predicates"]["integrity_preserved"] is False
    assert report["outcome"] == "native_state_observed"
    assert report["outcome_rank"] == 0


def test_deformable_spec_rejects_caller_verdicts_and_wrong_envelope() -> None:
    caller_graded = _task_spec()
    caller_graded["task_succeeded"] = True
    with pytest.raises(TaskNeutralScoringError) as caught:
        validate_deformable_task_spec(caller_graded)
    assert caught.value.errors == ("deformable_task_spec_fields_invalid",)

    wrong_envelope = copy.deepcopy(_task_spec())
    wrong_envelope["schema_version"] = "adp_task_spec.v0"
    wrong_envelope["task_kind"] = "rigid_pick_place"
    with pytest.raises(TaskNeutralScoringError) as caught:
        validate_deformable_task_spec(wrong_envelope)
    assert caught.value.errors == (
        "deformable_task_kind_invalid",
        "task_spec_schema_invalid",
    )
