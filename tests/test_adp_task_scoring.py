from __future__ import annotations

import copy
import math

import pytest

from blueprint_pipeline.adp009d_task_scoring import CAN_START_POSITION_M
from blueprint_pipeline.adp_task_scoring import (
    OUTCOME_NEVER_MOVED,
    OUTCOME_NON_TASK_JOINT_MOVED,
    OUTCOME_LIMIT_OR_CONTAINMENT_VIOLATION,
    OUTCOME_OPENED_AND_SETTLED,
    OUTCOME_OPENED_THEN_REBOUNDED,
    OUTCOME_RELEASE_OR_RETREAT_INCOMPLETE,
    TaskNeutralScoringError,
    confirm_rigid_task_success_contract,
    score_task_episode_from_spec,
    seal_rigid_task_success_contract,
)
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
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
        "task_contact_minimum_force_n": 0.5,
    }


def _rigid_v2_sample(step: int, position: list[float], *, safety: bool = True) -> dict:
    sample = {
        "step_index": step,
        "task_object_pose_world": [*position, 0.0, 0.0, 0.0, 1.0],
        "gripper_width_m": 0.08 if step >= 3 else 0.04,
        "task_contact_active": False if step >= 3 else True,
        "support_contact_active": step >= 3,
    }
    if safety:
        sample.update(
            robot_collision_failure=False,
            scene_collision_failure=False,
            containment_violation=False,
            forbidden_robot_task_collision_failure=False,
            locked_joint_containment_violation=False,
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
    assert report["manipulation_strategy"] == "pick_and_place"


def test_rigid_destination_is_scored_in_its_live_frame_and_must_stay_stable() -> None:
    task_spec = _rigid_v2_spec()
    half_sqrt = math.sqrt(0.5)
    destination_pose = [1.15, 2.0, 0.8, 0.0, 0.0, half_sqrt, half_sqrt]
    task_spec.update(
        destination_relation="inside",
        destination_pose_world=destination_pose,
        destination_position_bounds_destination_frame_m={
            "minimum": [0.04, -0.01, -0.01],
            "maximum": [0.06, 0.01, 0.01],
        },
        subject_collision_bounds_scoring_frame_m={
            "minimum": [-0.02, -0.005, -0.005],
            "maximum": [0.02, 0.005, 0.005],
        },
        destination_interior_bounds_body_frame_m={
            "minimum": [0.02, -0.03, -0.02],
            "maximum": [0.08, 0.03, 0.02],
        },
        destination_reset_translation_tolerance_m=0.002,
        destination_reset_rotation_tolerance_rad=0.01,
    )
    samples = [
        _rigid_v2_sample(0, [1.0, 2.0, 0.8]),
        _rigid_v2_sample(1, [1.0, 2.0, 0.83]),
        _rigid_v2_sample(2, [1.15, 2.05, 0.83]),
        _rigid_v2_sample(3, [1.15, 2.05, 0.8]),
        _rigid_v2_sample(4, [1.15, 2.05, 0.8]),
        _rigid_v2_sample(5, [1.15, 2.05, 0.8]),
    ]
    for sample in samples:
        sample["destination_pose_world"] = destination_pose

    report = score_task_episode_from_spec(task_spec=task_spec, samples=samples)

    assert report["task_succeeded"] is True
    assert report["measurements"]["settle_destination_inside"] is True
    assert report["measurements"]["destination_pose_stable"] is True

    samples[-1]["destination_pose_world"] = [
        1.16,
        2.0,
        0.8,
        0.0,
        0.0,
        half_sqrt,
        half_sqrt,
    ]
    moved_destination = score_task_episode_from_spec(
        task_spec=task_spec, samples=samples
    )
    assert moved_destination["task_succeeded"] is False
    assert "destination_pose_stability" in moved_destination["failed_criteria"]


def test_rigid_destination_pose_readback_is_required() -> None:
    task_spec = _rigid_v2_spec()
    task_spec.update(
        destination_relation="inside",
        destination_pose_world=[1.15, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0],
        destination_position_bounds_destination_frame_m={
            "minimum": [-0.01, -0.01, -0.01],
            "maximum": [0.01, 0.01, 0.01],
        },
        subject_collision_bounds_scoring_frame_m={
            "minimum": [-0.005, -0.005, -0.005],
            "maximum": [0.005, 0.005, 0.005],
        },
        destination_interior_bounds_body_frame_m={
            "minimum": [-0.02, -0.02, -0.02],
            "maximum": [0.02, 0.02, 0.02],
        },
        destination_reset_translation_tolerance_m=0.002,
        destination_reset_rotation_tolerance_rad=0.01,
    )
    report = score_task_episode_from_spec(
        task_spec=task_spec,
        samples=[
            _rigid_v2_sample(0, [1.0, 2.0, 0.8]),
            _rigid_v2_sample(1, [1.15, 2.0, 0.8]),
            _rigid_v2_sample(2, [1.15, 2.0, 0.8]),
            _rigid_v2_sample(3, [1.15, 2.0, 0.8]),
        ],
    )

    assert report["status"] == "undetermined"
    assert report["outcome"] == "native_destination_pose_readback_missing"


def _held_out_groot_push_samples() -> list[dict]:
    """Reproduce the terminal measurements from Quick-10 held-out GR00T."""

    samples = [
        _rigid_v2_sample(0, [1.0, 2.0, 0.8]),
        _rigid_v2_sample(1, [1.08, 2.0, 0.8]),
        _rigid_v2_sample(2, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(3, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(4, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(5, [1.15, 2.0, 0.8]),
    ]
    for sample in samples:
        # A planar push keeps the fingers closed and does not control the
        # object's terminal yaw.  Neither fact invalidates a completed push.
        sample["gripper_width_m"] = 0.0
    for sample in samples[2:]:
        sample["task_object_pose_world"][3:] = [
            0.0,
            0.0,
            0.7071067811865476,
            0.7071067811865476,
        ]
    return samples


def test_planar_push_scores_held_out_groot_destination_and_settle_as_success() -> None:
    task_spec = _rigid_v2_spec()
    task_spec["manipulation_strategy"] = "planar_push"

    report = score_task_episode_from_spec(
        task_spec=task_spec,
        samples=_held_out_groot_push_samples(),
    )

    assert report["status"] == "scored"
    assert report["manipulation_strategy"] == "planar_push"
    assert report["outcome"] == "pushed_and_settled"
    assert report["task_succeeded"] is True
    assert report["measurements"]["settle_destination_inside"] is True
    assert report["measurements"]["settle_orientation_ok"] is False
    assert report["measurements"]["settle_support_height_ok"] is True
    assert report["measurements"]["settle_support_contact_ok"] is True
    assert report["measurements"]["settled"] is True
    assert report["measurements"]["settle_task_contact_cleared"] is True
    assert report["measurements"]["released"] is False
    assert report["measurements"]["maximum_lift_m"] == 0.0


def test_pick_and_place_keeps_release_orientation_and_lift_requirements() -> None:
    report = score_task_episode_from_spec(
        task_spec=_rigid_v2_spec(),
        samples=_held_out_groot_push_samples(),
    )

    assert report["manipulation_strategy"] == "pick_and_place"
    assert report["outcome"] == "release_incomplete"
    assert report["task_succeeded"] is False


def test_planar_push_requires_task_contact_to_clear_during_settle() -> None:
    task_spec = _rigid_v2_spec()
    task_spec["manipulation_strategy"] = "planar_push"
    samples = _held_out_groot_push_samples()
    samples[-1]["task_contact_active"] = True

    report = score_task_episode_from_spec(task_spec=task_spec, samples=samples)

    assert report["status"] == "scored"
    assert report["outcome"] == "push_contact_not_cleared"
    assert report["task_succeeded"] is False
    assert report["measurements"]["settle_task_contact_cleared"] is False


def test_team_confirmed_contract_can_define_task_specific_terminal_success() -> None:
    task_spec = _rigid_v2_spec()
    task_spec.update(site_id="scene839873", task_id="move_cup_to_green_target")
    compatibility = seal_rigid_task_success_contract(
        task_spec=task_spec,
        site_id=task_spec["site_id"],
        task_id=task_spec["task_id"],
        author_source="compatibility_default",
        author_id="test-default",
        confirmation_status="confirmed",
    )
    criteria = copy.deepcopy(compatibility["criteria"])
    criteria["orientation"]["mode"] = "ignored"
    criteria["gripper_state"] = {"mode": "ignored", "threshold_m": None}
    criteria["terminal_task_contact"]["mode"] = "cleared"
    criteria["motion"]["minimum_lift_m"] = None
    task_spec["task_success_contract"] = seal_rigid_task_success_contract(
        task_spec=task_spec,
        site_id=task_spec["site_id"],
        task_id=task_spec["task_id"],
        author_source="site_robot_team",
        author_id="robot-team:relocation-owners",
        confirmation_status="confirmed",
        confirmed_by_team_id="robot-team:relocation-owners",
        criteria=criteria,
    )

    report = score_task_episode_from_spec(
        task_spec=task_spec,
        samples=_held_out_groot_push_samples(),
    )

    assert report["task_succeeded"] is True
    assert report["failed_criteria"] == []
    assert report["failure_reason_plain_english"] is None
    assert report["task_success_contract_digest"] == (
        task_spec["task_success_contract"]["contract_digest"]
    )
    assert report["task_success_contract"]["provenance"]["author_source"] == (
        "site_robot_team"
    )


def test_agent_contract_is_proposal_only_until_a_team_confirms_new_document() -> None:
    task_spec = _rigid_v2_spec()
    task_spec.update(site_id="scene839873", task_id="move_cup_to_green_target")
    proposal = seal_rigid_task_success_contract(
        task_spec=task_spec,
        site_id=task_spec["site_id"],
        task_id=task_spec["task_id"],
        author_source="agent_proposal",
        author_id="agent:criteria-drafter",
        confirmation_status="proposal_only",
    )
    task_spec["task_success_contract"] = proposal

    with pytest.raises(
        TaskNeutralScoringError,
        match="rigid_task_success_contract_unconfirmed",
    ):
        score_task_episode_from_spec(
            task_spec=task_spec,
            samples=_held_out_groot_push_samples(),
        )

    confirmed = confirm_rigid_task_success_contract(
        proposal, confirmed_by_team_id="robot-team:relocation-owners"
    )
    assert proposal["provenance"]["confirmation_status"] == "proposal_only"
    assert confirmed["contract_digest"] != proposal["contract_digest"]
    assert confirmed["provenance"]["proposal_digest"] == proposal["contract_digest"]
    task_spec["task_success_contract"] = confirmed
    report = score_task_episode_from_spec(
        task_spec=task_spec,
        samples=[
            _rigid_v2_sample(0, [1.0, 2.0, 0.8]),
            _rigid_v2_sample(1, [1.0, 2.0, 0.83]),
            _rigid_v2_sample(2, [1.15, 2.0, 0.83]),
            _rigid_v2_sample(3, [1.15, 2.0, 0.8]),
            _rigid_v2_sample(4, [1.15, 2.0, 0.8]),
            _rigid_v2_sample(5, [1.15, 2.0, 0.8]),
        ],
    )
    assert report["task_succeeded"] is True


def test_task_success_contract_digest_detects_post_confirmation_drift() -> None:
    task_spec = _rigid_v2_spec()
    contract = seal_rigid_task_success_contract(
        task_spec=task_spec,
        site_id="scene839873",
        task_id="move_cup_to_green_target",
        author_source="site_robot_team",
        author_id="robot-team:relocation-owners",
        confirmation_status="confirmed",
        confirmed_by_team_id="robot-team:relocation-owners",
    )
    contract["criteria"]["orientation"]["mode"] = "ignored"
    task_spec["task_success_contract"] = contract

    with pytest.raises(
        TaskNeutralScoringError,
        match="rigid_task_success_contract_digest_mismatch",
    ):
        score_task_episode_from_spec(
            task_spec=task_spec,
            samples=_held_out_groot_push_samples(),
        )


def test_task_success_contract_digest_uses_browser_compatible_number_encoding() -> None:
    contract = seal_rigid_task_success_contract(
        task_spec=_rigid_v2_spec(),
        site_id="interiorgs-839873",
        task_id="move-cup",
        author_source="compatibility_default",
        author_id="blueprint:manipulation_strategy_defaults.v1",
        confirmation_status="confirmed",
    )

    assert contract["contract_digest"] == cross_runtime_canonical_digest(
        contract, digest_field="contract_digest"
    )
    assert contract["contract_digest"] != canonical_digest(
        contract, digest_field="contract_digest"
    )


def _dropped_then_placed_samples() -> list[dict]:
    samples = [
        _rigid_v2_sample(0, [1.0, 2.0, 0.8]),
        _rigid_v2_sample(1, [1.0, 2.0, 0.86]),
        _rigid_v2_sample(2, [1.15, 2.0, 0.86]),
        _rigid_v2_sample(3, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(4, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(5, [1.15, 2.0, 0.8]),
    ]
    # Contact is lost while the object is still unsupported at step 2; it
    # falls 6 cm and regains support inside the destination at step 3.
    samples[2]["task_contact_active"] = False
    return samples


def test_no_drop_is_distinct_from_eventual_placement_success() -> None:
    task_spec = _rigid_v2_spec()
    eventual = score_task_episode_from_spec(
        task_spec=task_spec, samples=_dropped_then_placed_samples()
    )
    assert eventual["task_succeeded"] is True
    assert eventual["event_ledger"]["drop_events"][0]["fall_m"] == pytest.approx(
        0.06
    )
    assert eventual["event_ledger"]["drop_events"][0]["contact_lost_step"] == 2
    assert eventual["event_ledger"]["drop_events"][0]["minimum_height_m"] == 0.8
    assert eventual["event_ledger"]["drop_events"][0]["support_recontact_step"] == 3
    assert (
        eventual["event_ledger"]["drop_events"][0][
            "destination_inside_at_recontact"
        ]
        is True
    )

    default_contract = eventual["task_success_contract"]
    criteria = copy.deepcopy(default_contract["criteria"])
    criteria["temporal_invariants"]["no_drop"]["mode"] = "required"
    task_spec.update(site_id="scene839873", task_id="move_cup_to_green_target")
    task_spec["task_success_contract"] = seal_rigid_task_success_contract(
        task_spec=task_spec,
        site_id=task_spec["site_id"],
        task_id=task_spec["task_id"],
        author_source="site_robot_team",
        author_id="robot-team:relocation-owners",
        confirmation_status="confirmed",
        confirmed_by_team_id="robot-team:relocation-owners",
        criteria=criteria,
    )
    no_drop = score_task_episode_from_spec(
        task_spec=task_spec, samples=_dropped_then_placed_samples()
    )

    assert no_drop["task_succeeded"] is False
    assert no_drop["failed_criteria"] == ["no_drop"]
    assert no_drop["failure_reason_plain_english"].startswith(
        "The object was dropped"
    )


def test_no_drop_allows_contact_clear_after_supported_controlled_placement() -> None:
    task_spec = _rigid_v2_spec()
    samples = [
        _rigid_v2_sample(0, [1.0, 2.0, 0.8]),
        _rigid_v2_sample(1, [1.0, 2.0, 0.86]),
        _rigid_v2_sample(2, [1.15, 2.0, 0.86]),
        _rigid_v2_sample(3, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(4, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(5, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(6, [1.15, 2.0, 0.8]),
    ]
    # The object is placed onto support before task contact clears; it remains
    # supported throughout the release, so the 6 cm controlled lowering is
    # not a drop event.
    samples[3]["task_contact_active"] = True
    baseline = score_task_episode_from_spec(task_spec=task_spec, samples=samples)
    criteria = copy.deepcopy(baseline["task_success_contract"]["criteria"])
    criteria["temporal_invariants"]["no_drop"]["mode"] = "required"
    task_spec.update(site_id="scene839873", task_id="move_cup_to_green_target")
    task_spec["task_success_contract"] = seal_rigid_task_success_contract(
        task_spec=task_spec,
        site_id=task_spec["site_id"],
        task_id=task_spec["task_id"],
        author_source="site_robot_team",
        author_id="robot-team:relocation-owners",
        confirmation_status="confirmed",
        confirmed_by_team_id="robot-team:relocation-owners",
        criteria=criteria,
    )

    report = score_task_episode_from_spec(task_spec=task_spec, samples=samples)

    assert report["task_succeeded"] is True
    assert report["criteria_satisfied"]["no_drop"] is True
    assert report["event_ledger"]["drop_events"] == []


def test_scoped_temporal_event_limits_are_deterministically_enforced() -> None:
    task_spec = _rigid_v2_spec()
    baseline = score_task_episode_from_spec(
        task_spec=task_spec,
        samples=_dropped_then_placed_samples(),
    )
    criteria = copy.deepcopy(baseline["task_success_contract"]["criteria"])
    temporal = criteria["temporal_invariants"]
    temporal.update(
        maximum_task_contact_force_n=20.0,
        forbidden_contact_classes=["table_edge"],
        workspace_excursions="forbidden",
        maximum_retries=1,
        maximum_regrasps=0,
    )
    task_spec.update(site_id="scene839873", task_id="move_cup_to_green_target")
    task_spec["task_success_contract"] = seal_rigid_task_success_contract(
        task_spec=task_spec,
        site_id=task_spec["site_id"],
        task_id=task_spec["task_id"],
        author_source="site_robot_team",
        author_id="robot-team:relocation-owners",
        confirmation_status="confirmed",
        confirmed_by_team_id="robot-team:relocation-owners",
        criteria=criteria,
    )
    samples = _dropped_then_placed_samples()
    for sample in samples:
        sample.update(
            task_contact_force_n=5.0,
            contact_classes_active=[],
            workspace_excursion=False,
            retry_count=0,
            regrasp_count=0,
        )
    samples[2].update(
        task_contact_force_n=25.0,
        contact_classes_active=["table_edge"],
        workspace_excursion=True,
    )
    for sample in samples[2:]:
        sample.update(retry_count=2, regrasp_count=1)

    report = score_task_episode_from_spec(task_spec=task_spec, samples=samples)

    assert report["task_succeeded"] is False
    assert report["failed_criteria"] == [
        "maximum_task_contact_force",
        "forbidden_contact_classes",
        "workspace_excursions",
        "maximum_retries",
        "maximum_regrasps",
    ]
    assert report["event_ledger"]["peak_task_contact_force_n"] == 25.0
    assert report["event_ledger"]["observed_forbidden_contact_classes"] == [
        "table_edge"
    ]


def test_contact_force_limit_uses_retained_top_level_native_readback() -> None:
    task_spec = _rigid_v2_spec()
    baseline = score_task_episode_from_spec(
        task_spec=task_spec,
        samples=_dropped_then_placed_samples(),
    )
    criteria = copy.deepcopy(baseline["task_success_contract"]["criteria"])
    criteria["temporal_invariants"]["maximum_task_contact_force_n"] = 10.0
    task_spec.update(site_id="scene839873", task_id="move_cup_to_green_target")
    task_spec["task_success_contract"] = seal_rigid_task_success_contract(
        task_spec=task_spec,
        site_id=task_spec["site_id"],
        task_id=task_spec["task_id"],
        author_source="site_robot_team",
        author_id="robot-team:relocation-owners",
        confirmation_status="confirmed",
        confirmed_by_team_id="robot-team:relocation-owners",
        criteria=criteria,
    )
    samples = _dropped_then_placed_samples()
    for sample in samples:
        sample.update(
            task_robot_contact_peak_force_n=5.0,
            task_contact_force_n=1.0,
            native_readback={"task_robot_contact_peak_force_n": 2.0},
        )
    samples[2]["task_robot_contact_peak_force_n"] = 12.0

    report = score_task_episode_from_spec(task_spec=task_spec, samples=samples)

    assert report["task_succeeded"] is False
    assert report["event_ledger"]["peak_task_contact_force_n"] == 12.0
    assert report["event_ledger"]["task_contact_force_sources"] == [
        "task_robot_contact_peak_force_n"
    ]
    assert report["failed_criteria"] == ["maximum_task_contact_force"]


def test_scoped_temporal_limit_abstains_when_event_readback_is_missing() -> None:
    task_spec = _rigid_v2_spec()
    baseline = score_task_episode_from_spec(
        task_spec=task_spec,
        samples=_dropped_then_placed_samples(),
    )
    criteria = copy.deepcopy(baseline["task_success_contract"]["criteria"])
    criteria["temporal_invariants"]["maximum_retries"] = 0
    task_spec.update(site_id="scene839873", task_id="move_cup_to_green_target")
    task_spec["task_success_contract"] = seal_rigid_task_success_contract(
        task_spec=task_spec,
        site_id=task_spec["site_id"],
        task_id=task_spec["task_id"],
        author_source="site_robot_team",
        author_id="robot-team:relocation-owners",
        confirmation_status="confirmed",
        confirmed_by_team_id="robot-team:relocation-owners",
        criteria=criteria,
    )

    report = score_task_episode_from_spec(
        task_spec=task_spec, samples=_dropped_then_placed_samples()
    )

    assert report["status"] == "undetermined"
    assert report["outcome"] == "native_temporal_event_readback_missing"
    assert report["event_ledger"]["required_readback_gaps"] == [
        "retry_event_ledger"
    ]


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


def test_scene_neutral_rigid_task_cannot_pass_without_support_contact() -> None:
    samples = [
        _rigid_v2_sample(0, [1.0, 2.0, 0.8]),
        _rigid_v2_sample(1, [1.0, 2.0, 0.83]),
        _rigid_v2_sample(2, [1.15, 2.0, 0.83]),
        _rigid_v2_sample(3, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(4, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(5, [1.15, 2.0, 0.8]),
    ]
    for sample in samples[-3:]:
        sample["support_contact_active"] = False

    report = score_task_episode_from_spec(
        task_spec=_rigid_v2_spec(), samples=samples
    )

    assert report["status"] == "scored"
    assert report["task_succeeded"] is False
    assert report["measurements"]["settle_support_contact_ok"] is False


def test_scene_neutral_rigid_task_cannot_pass_when_locked_joint_moves() -> None:
    samples = [
        _rigid_v2_sample(0, [1.0, 2.0, 0.8]),
        _rigid_v2_sample(1, [1.0, 2.0, 0.83]),
        _rigid_v2_sample(2, [1.15, 2.0, 0.83]),
        _rigid_v2_sample(3, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(4, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(5, [1.15, 2.0, 0.8]),
    ]
    samples[2]["locked_joint_containment_violation"] = True

    report = score_task_episode_from_spec(
        task_spec=_rigid_v2_spec(), samples=samples
    )

    assert report["status"] == "scored"
    assert report["outcome"] == "collision_or_containment_failure"
    assert report["task_succeeded"] is False


def test_forbidden_robot_object_collision_emits_specific_safety_event() -> None:
    task_spec = _rigid_v2_spec()
    task_spec["control_frequency_hz"] = 15.0
    samples = [
        _rigid_v2_sample(0, [1.0, 2.0, 0.8]),
        _rigid_v2_sample(1, [1.0, 2.0, 0.83]),
        _rigid_v2_sample(72, [1.15, 2.0, 0.83]),
        _rigid_v2_sample(73, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(74, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(75, [1.15, 2.0, 0.8]),
    ]
    samples[2].update(
        robot_collision_failure=True,
        forbidden_robot_task_collision_failure=True,
        robot_task_forbidden_collision_peak_force_n=4.519003553,
        collision_failure_minimum_force_n=1.0,
    )

    report = score_task_episode_from_spec(task_spec=task_spec, samples=samples)

    assert report["task_succeeded"] is False
    assert report["failure_reason_plain_english"] == (
        "Forbidden robot-object contact reached 4.519 N, exceeding 1 N at step 72."
    )
    assert report["event_ledger"]["safety_events"] == [
        {
            "event_type": "forbidden_robot_object_contact_force_exceeded",
            "step_index": 72,
            "simulation_time_seconds": 4.8,
            "measured_force_n": 4.519003553,
            "threshold_n": 1.0,
            "contact_pair_identities": [],
            "contact_pair_identity_status": "contact_pair_identity_missing",
        }
    ]


def test_scene_neutral_rigid_task_abstains_without_support_contact_readback() -> None:
    samples = [
        _rigid_v2_sample(0, [1.0, 2.0, 0.8]),
        _rigid_v2_sample(1, [1.0, 2.0, 0.83]),
        _rigid_v2_sample(2, [1.15, 2.0, 0.83]),
        _rigid_v2_sample(3, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(4, [1.15, 2.0, 0.8]),
        _rigid_v2_sample(5, [1.15, 2.0, 0.8]),
    ]
    for sample in samples[-3:]:
        del sample["support_contact_active"]

    report = score_task_episode_from_spec(
        task_spec=_rigid_v2_spec(), samples=samples
    )

    assert report["status"] == "undetermined"
    assert report["outcome"] == "native_support_contact_readback_missing"
    assert report["task_succeeded"] is False


def test_solver_residual_at_a_hard_stop_is_not_a_limit_violation() -> None:
    """C29's measured -5.7e-8 rad would have failed a successful grasp.

    A joint resting against its own hard stop reports a tiny excursion past
    it in any physics engine.  This recomputation compared that residual with
    exact arithmetic, so the closed washer door -- merely nudged, and reported
    violation-free by the simulator's own flag in the same trace -- failed the
    positive control on about 34 nanometres at the handle.
    """

    from blueprint_pipeline.adp_task_scoring import (
        JOINT_HARD_LIMIT_SOLVER_RESIDUAL_RAD,
    )

    samples = [
        _sample(0, -5.716e-08),
        _sample(1, 0.4, speed=0.4),
        _sample(2, 0.9, speed=0.2),
        _sample(3, 0.9),
        _sample(4, 0.9),
        _sample(5, 0.9),
    ]

    report = score_task_episode_from_spec(task_spec=_articulated_spec(), samples=samples)

    assert report["predicates"]["joint_hard_limits_respected"] is True
    assert report["outcome"] == OUTCOME_OPENED_AND_SETTLED
    assert report["task_succeeded"] is True
    # The excursion is sealed either way, so a real violation creeping up on
    # the allowance stays visible rather than silently absorbed.
    measurements = report["measurements"]
    assert measurements["joint_hard_limit_max_excursion_rad"] == pytest.approx(
        5.716e-08
    )
    assert measurements["joint_hard_limit_solver_residual_rad"] == (
        JOINT_HARD_LIMIT_SOLVER_RESIDUAL_RAD
    )


def test_a_real_excursion_past_a_hard_stop_still_fails() -> None:
    """The allowance is solver residual, not a loosened task gate."""

    from blueprint_pipeline.adp_task_scoring import (
        JOINT_HARD_LIMIT_SOLVER_RESIDUAL_RAD,
    )

    samples = [
        _sample(0, -10.0 * JOINT_HARD_LIMIT_SOLVER_RESIDUAL_RAD),
        _sample(1, 0.4, speed=0.4),
        _sample(2, 0.9, speed=0.2),
        _sample(3, 0.9),
        _sample(4, 0.9),
        _sample(5, 0.9),
    ]

    report = score_task_episode_from_spec(task_spec=_articulated_spec(), samples=samples)

    assert report["predicates"]["joint_hard_limits_respected"] is False
    assert report["outcome"] == OUTCOME_LIMIT_OR_CONTAINMENT_VIOLATION
    assert report["task_succeeded"] is False


def test_the_simulators_own_violation_flag_stays_authoritative() -> None:
    """The native readback knows the real limits; it is never overridden."""

    samples = [
        _sample(0, 0.0),
        _sample(1, 0.4, speed=0.4),
        _sample(2, 0.9, speed=0.2),
        _sample(3, 0.9),
        _sample(4, 0.9),
        _sample(5, 0.9),
    ]
    samples[2]["joint_limit_violation"] = True

    report = score_task_episode_from_spec(task_spec=_articulated_spec(), samples=samples)

    assert report["predicates"]["joint_hard_limits_respected"] is False
    assert report["measurements"]["joint_hard_limit_max_excursion_rad"] == 0.0
    assert report["task_succeeded"] is False
