from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.adp009d_task_scoring import CAN_START_POSITION_M
from blueprint_pipeline.adp_task_scoring import (
    OUTCOME_NEVER_MOVED,
    OUTCOME_NON_TASK_JOINT_MOVED,
    OUTCOME_OPENED_AND_SETTLED,
    OUTCOME_OPENED_THEN_REBOUNDED,
    OUTCOME_RELEASE_OR_RETREAT_INCOMPLETE,
    TaskNeutralScoringError,
    score_task_episode_from_spec,
)


def _articulated_spec() -> dict:
    return {
        "schema_version": "adp_task_spec.v1",
        "task_kind": "articulated_open_close",
        "target_joint_id": "right_door",
        "joint_reset_positions_rad": {"right_door": 0.0, "left_door": 0.0},
        "target_success_interval_rad": [0.785398163, 1.396263402],
        "joint_hard_limits_rad": {
            "right_door": [0.0, 1.919862177],
            "left_door": [-0.01, 1.919862177],
        },
        "settle_window_samples": 3,
        "maximum_settled_target_speed_rad_s": 0.05,
        "non_task_joint_motion_tolerance_rad": 0.001,
        "movement_epsilon_rad": 0.0001,
        "reset_tolerance_rad": 0.0001,
    }


def _sample(step: int, target: float, *, other: float = 0.0, speed: float = 0.0) -> dict:
    return {
        "step_index": step,
        "joint_positions_rad": {"right_door": target, "left_door": other},
        "joint_velocities_rad_s": {"right_door": speed, "left_door": 0.0},
        "task_contact_active": False,
        "joint_limit_violation": False,
        "containment_violation": False,
        "robot_collision_failure": False,
        "scene_collision_failure": False,
        "retreat_completed": step >= 3,
    }


def test_task_neutral_dispatch_preserves_original_rigid_fixture() -> None:
    samples = [
        {
            "step_index": index,
            "can_pose_world": [*CAN_START_POSITION_M, 0.0, 0.0, 0.0, 1.0],
        }
        for index in range(40)
    ]
    report = score_task_episode_from_spec(
        task_spec={
            "schema_version": "adp_task_spec.v1",
            "task_kind": "rigid_pick_place",
            "destination_position_world_m": [
                CAN_START_POSITION_M[0] + 0.2,
                CAN_START_POSITION_M[1],
                CAN_START_POSITION_M[2],
            ],
            "support_plane_z_m": CAN_START_POSITION_M[2],
            "settle_window_samples": 40,
            "require_sealed_start_pose": True,
        },
        samples=samples,
    )

    assert report["outcome"] == OUTCOME_NEVER_MOVED
    assert report["task_succeeded"] is False


def test_articulated_zero_action_is_a_deterministic_negative() -> None:
    report = score_task_episode_from_spec(
        task_spec=_articulated_spec(),
        samples=[_sample(index, 0.0) for index in range(4)],
    )

    assert report["outcome"] == OUTCOME_NEVER_MOVED
    assert report["task_succeeded"] is False
    assert report["judgement_source"] == "deterministic_native_simulator_joint_state"


def test_articulated_scripted_positive_requires_release_retreat_and_settle() -> None:
    samples = [
        _sample(0, 0.0),
        _sample(1, 0.4, speed=0.4),
        _sample(2, 0.9, speed=0.2),
        _sample(3, 0.9),
        _sample(4, 0.9),
        _sample(5, 0.9),
    ]

    report = score_task_episode_from_spec(task_spec=_articulated_spec(), samples=samples)

    assert report["outcome"] == OUTCOME_OPENED_AND_SETTLED
    assert report["outcome_rank"] == 4
    assert report["task_succeeded"] is True
    assert all(report["predicates"].values())


def test_opened_then_rebounded_cannot_pass() -> None:
    samples = [
        _sample(0, 0.0),
        _sample(1, 0.9),
        _sample(2, 0.4),
        _sample(3, 0.3),
        _sample(4, 0.2),
    ]

    report = score_task_episode_from_spec(task_spec=_articulated_spec(), samples=samples)

    assert report["outcome"] == OUTCOME_OPENED_THEN_REBOUNDED
    assert report["outcome_rank"] == 2
    assert report["task_succeeded"] is False


@pytest.mark.parametrize(("contact_active", "retreat_completed"), [(True, True), (False, False)])
def test_stable_open_without_release_or_retreat_has_its_frozen_failure_rung(
    contact_active: bool, retreat_completed: bool
) -> None:
    samples = [
        _sample(0, 0.0),
        _sample(1, 0.9),
        _sample(2, 0.9),
        _sample(3, 0.9),
        _sample(4, 0.9),
    ]
    for sample in samples[-3:]:
        sample["task_contact_active"] = contact_active
        sample["retreat_completed"] = retreat_completed

    report = score_task_episode_from_spec(task_spec=_articulated_spec(), samples=samples)

    assert report["outcome"] == OUTCOME_RELEASE_OR_RETREAT_INCOMPLETE
    assert report["outcome_rank"] == 3
    assert report["task_succeeded"] is False


def test_non_task_joint_motion_blocks_otherwise_successful_open() -> None:
    samples = [
        _sample(0, 0.0),
        _sample(1, 0.9, other=0.002),
        _sample(2, 0.9, other=0.002),
        _sample(3, 0.9, other=0.002),
    ]

    report = score_task_episode_from_spec(task_spec=_articulated_spec(), samples=samples)

    assert report["outcome"] == OUTCOME_NON_TASK_JOINT_MOVED
    assert report["task_succeeded"] is False


def test_missing_native_velocity_fails_closed() -> None:
    sample = _sample(0, 0.0)
    del sample["joint_velocities_rad_s"]

    with pytest.raises(TaskNeutralScoringError) as caught:
        score_task_episode_from_spec(
            task_spec=_articulated_spec(), samples=[sample, copy.deepcopy(sample)]
        )

    assert any("joint_velocities_invalid" in error for error in caught.value.errors)


def _graph_spec() -> dict:
    link_ids = ["body", "door", "latch", "drum", "selector", "drawer", "panel"]

    def joint(
        joint_id: str,
        child: str,
        joint_type: str,
        role: str,
        limits: list[float],
        *,
        parent: str = "body",
        dependency: dict | None = None,
    ) -> dict:
        return {
            "joint_id": joint_id,
            "parent_link_id": parent,
            "child_link_id": child,
            "joint_type": joint_type,
            "role": role,
            "axis": [0.0, 0.0, 0.0] if joint_type == "fixed" else [0.0, 0.0, 1.0],
            "limits": limits,
            "reset_position": 0.0,
            "reset_tolerance": 0.0001,
            "drive": {
                "drive_type": "none" if role == "passive" else "force",
                "stiffness": 0.0 if role in {"target", "passive"} else 20.0,
                "damping": 0.1,
                "maximum_force": 0.0 if role == "passive" else 100.0,
            },
            "dependency": dependency,
        }

    graph = {
        "schema_version": "adp_articulation_graph.v1",
        "links": [
            {
                "link_id": link_id,
                "is_root": link_id == "body",
                "semantic_role": link_id,
            }
            for link_id in link_ids
        ],
        "joints": [
            joint("door_hinge", "door", "revolute", "target", [0.0, 1.2]),
            joint(
                "latch_coupler",
                "latch",
                "revolute",
                "dependent",
                [-0.2, 0.2],
                parent="door",
                dependency={
                    "driver_joint_id": "door_hinge",
                    "multiplier": 0.1,
                    "offset": 0.0,
                    "tolerance": 0.001,
                },
            ),
            joint("drum_bearing", "drum", "continuous", "passive", [-100.0, 100.0]),
            joint("selector_axis", "selector", "revolute", "locked", [-3.2, 3.2]),
            joint("detergent_slide", "drawer", "prismatic", "locked", [0.0, 0.2]),
            joint("service_panel_weld", "panel", "fixed", "locked", [0.0, 0.0]),
        ],
        "collision_pairs": [
            {"link_a": "body", "link_b": "door", "collision_enabled": True},
            {"link_a": "door", "link_b": "latch", "collision_enabled": False},
        ],
        "success_predicate": {
            "combination": "all",
            "joint_intervals": {"door_hinge": [0.7, 1.0]},
        },
    }
    return {
        "schema_version": "adp_task_spec.v2",
        "task_kind": "articulated_open_close",
        "articulation_graph": graph,
        "settle_window_samples": 3,
        "maximum_settled_target_speed": 0.05,
        "locked_joint_motion_tolerance": 0.001,
        "movement_epsilon": 0.0001,
    }


def _graph_sample(
    step: int,
    door: float,
    *,
    latch_error: float = 0.0,
    drum: float = 0.0,
) -> dict:
    positions = {
        "door_hinge": door,
        "latch_coupler": door * 0.1 + latch_error,
        "drum_bearing": drum,
        "selector_axis": 0.0,
        "detergent_slide": 0.0,
        "service_panel_weld": 0.0,
    }
    return {
        "step_index": step,
        "joint_positions": positions,
        "joint_velocities_per_s": {joint_id: 0.0 for joint_id in positions},
        "task_contact_active": False,
        "joint_limit_violation": False,
        "containment_violation": False,
        "robot_collision_failure": False,
        "scene_collision_failure": False,
        "retreat_completed": step >= 2,
    }


def test_general_graph_scores_target_dependent_passive_and_locked_roles() -> None:
    report = score_task_episode_from_spec(
        task_spec=_graph_spec(),
        samples=[
            _graph_sample(0, 0.0),
            _graph_sample(1, 0.8, drum=0.1),
            _graph_sample(2, 0.8, drum=0.2),
            _graph_sample(3, 0.8, drum=0.3),
            _graph_sample(4, 0.8, drum=0.4),
        ],
    )

    assert report["outcome"] == OUTCOME_OPENED_AND_SETTLED
    assert report["task_succeeded"] is True
    assert report["predicates"]["dependent_joints_consistent"] is True
    assert report["predicates"]["locked_joints_stable"] is True


def test_dependent_joint_violation_blocks_otherwise_successful_target() -> None:
    report = score_task_episode_from_spec(
        task_spec=_graph_spec(),
        samples=[
            _graph_sample(0, 0.0),
            _graph_sample(1, 0.8, latch_error=0.01),
            _graph_sample(2, 0.8, latch_error=0.01),
            _graph_sample(3, 0.8, latch_error=0.01),
        ],
    )

    assert report["outcome"] == OUTCOME_NON_TASK_JOINT_MOVED
    assert report["task_succeeded"] is False
    assert report["predicates"]["dependent_joints_consistent"] is False


def _rigid_v2_spec() -> dict:
    return {
        "schema_version": "adp_task_spec.v2",
        "task_kind": "rigid_pick_place",
        "subject_asset_id": "notebook_replacement",
        "start_pose_world": [1.0, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0],
        "destination_position_bounds_world_m": {
            "minimum": [1.14, 1.99, 0.79],
            "maximum": [1.16, 2.01, 0.81],
        },
        "destination_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "destination_orientation_tolerance_rad": 0.1,
        "support_height_interval_m": [0.79, 0.81],
        "minimum_translation_m": 0.14,
        "minimum_lift_m": 0.02,
        "movement_epsilon_m": 0.001,
        "reset_translation_tolerance_m": 0.001,
        "reset_orientation_tolerance_rad": 0.01,
        "settle_window_samples": 3,
        "settle_position_tolerance_m": 0.002,
        "settle_orientation_tolerance_rad": 0.01,
        "release_required": True,
        "release_gripper_width_min_m": 0.07,
    }


def _rigid_v2_sample(step: int, position: list[float], *, safety: bool = True) -> dict:
    sample = {
        "step_index": step,
        "task_object_pose_world": [*position, 0.0, 0.0, 0.0, 1.0],
        "gripper_width_m": 0.08 if step >= 3 else 0.04,
        "task_contact_active": False if step >= 3 else True,
    }
    if safety:
        sample.update(
            robot_collision_failure=False,
            scene_collision_failure=False,
            containment_violation=False,
        )
    return sample


def test_scene_neutral_rigid_task_scores_pose_volume_release_and_settle() -> None:
    report = score_task_episode_from_spec(
        task_spec=_rigid_v2_spec(),
        samples=[
            _rigid_v2_sample(0, [1.0, 2.0, 0.8]),
            _rigid_v2_sample(1, [1.0, 2.0, 0.83]),
            _rigid_v2_sample(2, [1.15, 2.0, 0.83]),
            _rigid_v2_sample(3, [1.15, 2.0, 0.8]),
            _rigid_v2_sample(4, [1.15, 2.0, 0.8]),
            _rigid_v2_sample(5, [1.15, 2.0, 0.8]),
        ],
    )

    assert report["status"] == "scored"
    assert report["outcome"] == "placed_and_settled"
    assert report["task_succeeded"] is True
    assert report["subject_asset_id"] == "notebook_replacement"


def test_scene_neutral_rigid_thresholds_accept_machine_roundoff_at_boundary() -> None:
    task_spec = _rigid_v2_spec()
    task_spec["minimum_lift_m"] = 0.08
    task_spec["minimum_translation_m"] = 0.15

    report = score_task_episode_from_spec(
        task_spec=task_spec,
        samples=[
            _rigid_v2_sample(0, [1.0, 2.0, 0.8]),
            _rigid_v2_sample(1, [1.0, 2.0, 0.88]),
            _rigid_v2_sample(2, [1.15, 2.0, 0.88]),
            _rigid_v2_sample(3, [1.15, 2.0, 0.8]),
            _rigid_v2_sample(4, [1.15, 2.0, 0.8]),
            _rigid_v2_sample(5, [1.15, 2.0, 0.8]),
        ],
    )

    assert report["measurements"]["maximum_lift_m"] < 0.08
    assert report["measurements"]["maximum_translation_m"] < 0.15
    assert report["task_succeeded"] is True


def test_scene_neutral_rigid_task_abstains_without_native_safety_readback() -> None:
    report = score_task_episode_from_spec(
        task_spec=_rigid_v2_spec(),
        samples=[
            _rigid_v2_sample(0, [1.0, 2.0, 0.8], safety=False),
            _rigid_v2_sample(1, [1.15, 2.0, 0.83], safety=False),
            _rigid_v2_sample(2, [1.15, 2.0, 0.8], safety=False),
            _rigid_v2_sample(3, [1.15, 2.0, 0.8], safety=False),
            _rigid_v2_sample(4, [1.15, 2.0, 0.8], safety=False),
        ],
    )

    assert report["status"] == "undetermined"
    assert report["outcome"] == "native_safety_readback_missing"
    assert report["task_succeeded"] is False
