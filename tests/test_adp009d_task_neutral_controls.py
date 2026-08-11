from __future__ import annotations

import copy
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_control_episode import (
    ControlEpisodeError,
    run_task_neutral_controls,
    validate_task_control_plan,
)
from blueprint_pipeline.adp009d_task_scoring import (
    CAN_START_POSITION_M,
    GRIPPER_FULL_OPENING_M,
    SUPPORT_PLANE_Z_M,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


_DESTINATION = [
    CAN_START_POSITION_M[0] + 0.2,
    CAN_START_POSITION_M[1],
    SUPPORT_PLANE_Z_M,
]


class _Environment:
    def __init__(self, task_kind: str):
        self.task_kind = task_kind
        self.reset_count = 0
        self.steps = []
        self.joints = [0.0] * 7
        self.gripper = 0.0

    def reset(self):
        self.reset_count += 1
        self.steps = []
        self.joints = [0.0] * 7
        self.gripper = 0.0

    def hold_action(self, *, gripper_command):
        return [*self.joints, float(gripper_command)]

    def step(self, action):
        self.steps.append(list(action))
        self.joints = [float(value) for value in action[:7]]
        self.gripper = float(action[7])

    def read_arm_joint_positions(self):
        return list(self.joints)

    def read_task_sample(self):
        return {
            "joint_positions_rad": {
                "refrigerator_upper_door_hinge": max(0.0, self.joints[0]),
                "refrigerator_lower_door_hinge": 0.0,
            },
            "joint_velocities_rad_s": {
                "refrigerator_upper_door_hinge": 0.0,
                "refrigerator_lower_door_hinge": 0.0,
            },
            "task_contact_active": self.gripper > 0.5,
            "joint_limit_violation": False,
            "containment_violation": False,
            "robot_collision_failure": False,
            "scene_collision_failure": False,
            "retreat_completed": self.joints[1] > 0.5,
            "grasp_frame_position_world_m": self.joints[:3],
        }

    def read_object_sample(self):
        progress = self.joints[0]
        if progress <= 0.0:
            position = list(CAN_START_POSITION_M)
            width = GRIPPER_FULL_OPENING_M
            sample = {"can_pose_world": [*position, 0.0, 0.0, 0.0, 1.0]}
        elif progress < 1.0:
            position = [
                CAN_START_POSITION_M[0],
                CAN_START_POSITION_M[1],
                SUPPORT_PLANE_Z_M + 0.1,
            ]
            width = 0.07
            sample = {
                "can_pose_world": [*position, 0.0, 0.0, 0.0, 1.0],
                "grasp_frame_position_world_m": position,
                "finger_contact_forces_n": [2.5, 2.5],
            }
        else:
            position = list(_DESTINATION)
            width = GRIPPER_FULL_OPENING_M
            sample = {"can_pose_world": [*position, 0.0, 0.0, 0.0, 1.0]}
        sample["gripper_width_m"] = width
        return sample


def _task(task_kind: str) -> dict:
    if task_kind == "rigid_pick_place":
        return {
            "schema_version": "adp_task_spec.v1",
            "task_kind": task_kind,
            "destination_position_world_m": _DESTINATION,
            "support_plane_z_m": SUPPORT_PLANE_Z_M,
            "settle_window_samples": 3,
            "require_sealed_start_pose": True,
            "maximum_action_steps": 8,
        }
    return {
        "schema_version": "adp_task_spec.v1",
        "task_kind": "articulated_open_close",
        "target_joint_id": "refrigerator_upper_door_hinge",
        "joint_reset_positions_rad": {
            "refrigerator_upper_door_hinge": 0.0,
            "refrigerator_lower_door_hinge": 0.0,
        },
        "target_success_interval_rad": [0.785398163, 0.959931089],
        "joint_hard_limits_rad": {
            "refrigerator_upper_door_hinge": [0.0, 1.570796327],
            "refrigerator_lower_door_hinge": [0.0, 1.570796327],
        },
        "settle_window_samples": 3,
        "maximum_settled_target_speed_rad_s": 0.05,
        "non_task_joint_motion_tolerance_rad": 0.001,
        "movement_epsilon_rad": 0.0001,
        "reset_tolerance_rad": 0.0001,
        "maximum_action_steps": 8,
    }


def _plan(task: dict) -> dict:
    actions = (
        [
            {"phase_id": "lift", "isaac_action": [0.5, 0, 0, 0, 0, 0, 0, 1]},
            {"phase_id": "place", "isaac_action": [1.0, 0, 0, 0, 0, 0, 0, 0]},
        ]
        if task["task_kind"] == "rigid_pick_place"
        else [
            {"phase_id": "open", "isaac_action": [0.9, 0, 0, 0, 0, 0, 0, 1]},
            {"phase_id": "retreat", "isaac_action": [0.9, 1, 0, 0, 0, 0, 0, 0]},
        ]
    )
    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "cell_id": f"{task['task_kind']}-canonical",
        "task_spec_digest": canonical_digest(task),
        "trajectory_source": "native_ik_preflight",
        "planner_receipt_digest": "sha256:" + "a" * 64,
        "zero_action_steps": 3,
        "scripted_positive_actions": actions,
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _semantic_articulated_plan(task: dict) -> dict:
    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "cell_id": "articulated-open-close-semantic-gripper",
        "task_spec_digest": canonical_digest(task),
        "trajectory_source": "native_ik_preflight",
        "planner_receipt_digest": "sha256:" + "b" * 64,
        "zero_action_steps": 3,
        "scripted_positive_actions": [
            {
                "phase_id": "open",
                "arm_joint_positions": [0.9, 0, 0, 0, 0, 0, 0],
                "gripper_state": "closed",
            },
            {
                "phase_id": "retreat",
                "arm_joint_positions": [0.9, 1, 0, 0, 0, 0, 0],
                "gripper_state": "open",
            },
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _cartesian_articulated_plan(task: dict) -> dict:
    def phase(phase_id, target, gripper_state):
        return {
            "phase_id": phase_id,
            "mode": "ik_pose",
            "target_position_world_m": target,
            "target_quaternion_world_xyzw": None,
            "gripper_state": gripper_state,
            "minimum_steps": 1,
            "maximum_steps": 2,
            "arrival_tolerance_m": 1.0e-6,
            "arrival_stability_steps": 1,
            "max_joint_delta_rad": 0.03,
            "max_joint_setpoint_lead_rad": 0.2,
        }

    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "cell_id": "articulated-open-close-cartesian",
        "task_spec_digest": canonical_digest(task),
        "trajectory_source": "native_ik_preflight",
        "planner_receipt_digest": "sha256:" + "c" * 64,
        "zero_action_steps": 3,
        "scripted_positive_actions": [
            phase("open", [0.9, 0.0, 0.0], "closed"),
            phase("retreat", [0.9, 1.0, 0.0], "open"),
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _deformable_task() -> dict:
    return {
        "schema_version": "adp_task_spec.v1",
        "task_kind": "deformable_transfer",
        "prompt": "Transfer the deformable into the destination and retreat.",
        "deformable_entity_id": "movable",
        "destination_entity_id": "destination",
        "robot_entity_id": "robot",
        "destination_interior_obb": {
            "center_world_m": [0.5, 0.0, 0.2],
            "half_extents_m": [0.2, 0.15, 0.1],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "receptacle_reference_pose_world": {
            "position_m": [0.5, 0.0, 0.1],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "minimum_particle_fraction_inside": 0.75,
        "settle_window_samples": 3,
        "maximum_node_speed_mps": 0.02,
        "maximum_principal_strain": 0.25,
        "minimum_grasp_contact_force_n": 0.1,
        "maximum_release_contact_force_n": 0.0,
        "minimum_robot_clearance_m": 0.1,
        "maximum_receptacle_translation_drift_m": 0.01,
        "maximum_receptacle_rotation_drift_rad": 0.03,
        "maximum_receptacle_linear_speed_mps": 0.01,
        "maximum_receptacle_angular_speed_radps": 0.03,
        "control_frequency_hz": 15,
        "maximum_action_steps": 10,
    }


def _deformable_plan(task: dict) -> dict:
    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "cell_id": "deformable-canonical",
        "task_spec_digest": canonical_digest(task),
        "trajectory_source": "native_ik_preflight",
        "planner_receipt_digest": "sha256:" + "d" * 64,
        "zero_action_steps": 3,
        "scripted_positive_actions": [
            {
                "phase_id": "grasp",
                "isaac_action": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            },
            {
                "phase_id": "release_and_retreat",
                "isaac_action": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def test_task_control_plan_accepts_the_shared_deformable_scorer_contract() -> None:
    task = _deformable_task()
    plan = _deformable_plan(task)

    assert validate_task_control_plan(plan, task_spec=task) == plan


def test_task_control_plan_rejects_a_deformable_scorer_contract_drift() -> None:
    task = _deformable_task()
    plan = _deformable_plan(task)
    task["minimum_grasp_contact_force_n"] = 0.0

    with pytest.raises(
        ControlEpisodeError,
        match="deformable_transfer_grasp_contact_force_invalid",
    ):
        validate_task_control_plan(plan, task_spec=task)


class _CartesianEnvironment(_Environment):
    def scripted_action_for_pose(
        self,
        *,
        target_position_world_m,
        target_quaternion_world_xyzw,
        gripper_command,
        max_joint_delta_rad,
        max_joint_setpoint_lead_rad,
    ):
        del target_quaternion_world_xyzw
        assert max_joint_delta_rad == pytest.approx(0.03)
        assert max_joint_setpoint_lead_rad == pytest.approx(0.2)
        return [*target_position_world_m, 0.0, 0.0, 0.0, 0.0, gripper_command]


@pytest.mark.parametrize("task_kind", ["rigid_pick_place", "articulated_open_close"])
def test_same_control_contract_passes_original_and_second_scene_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_kind: str
) -> None:
    from blueprint_pipeline import adp009d_control_episode as module

    observation_index = {"value": 0}

    def fake_observation(*_args, **kwargs):
        row = {
            "observation_index": observation_index["value"],
            "kind": kwargs["kind"],
            "views": {},
        }
        observation_index["value"] += 1
        return row

    monkeypatch.setattr(module, "_persist_observation", fake_observation)
    monkeypatch.setattr(
        module,
        "finalize_manipulation_evaluation_visual_evidence",
        lambda **_kwargs: (
            {
                "status": "complete",
                "required_camera_ids": ["external", "wrist", "overview"],
                "review_only_camera_ids": ["overview"],
            },
            [],
        ),
    )
    task = _task(task_kind)
    environment = _Environment(task_kind)

    pair = run_task_neutral_controls(
        environment=environment,
        task_spec=task,
        control_plan=_plan(task),
        gripper_open_command=0.0,
        output_dir=tmp_path,
    )

    assert pair["cell_admitted_for_policy_execution"] is True
    assert pair["execution_order"] == [
        "zero_action_negative",
        "deterministic_scripted_positive",
    ]
    assert [row["observed_outcome"] for row in pair["controls"]] == [
        "never_moved",
        "placed" if task_kind == "rigid_pick_place" else "opened_and_settled",
    ]
    assert environment.reset_count == 2


def test_task_control_plan_rejects_unbound_or_over_budget_trajectory(
    tmp_path: Path,
) -> None:
    task = _task("articulated_open_close")
    plan = _plan(task)
    plan["scripted_positive_actions"] *= 4
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    with pytest.raises(ControlEpisodeError, match="task_control_action_budget_exceeds_task_spec"):
        run_task_neutral_controls(
            environment=_Environment("articulated_open_close"),
            task_spec=copy.deepcopy(task),
            control_plan=plan,
            gripper_open_command=0.0,
            output_dir=tmp_path,
        )


def test_semantic_gripper_states_use_the_native_measured_command_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import adp009d_control_episode as module

    monkeypatch.setattr(
        module,
        "_persist_observation",
        lambda *_args, **kwargs: {
            "observation_index": 0,
            "kind": kwargs["kind"],
            "views": {},
        },
    )
    monkeypatch.setattr(
        module,
        "finalize_manipulation_evaluation_visual_evidence",
        lambda **_kwargs: ({"status": "complete"}, []),
    )
    task = _task("articulated_open_close")
    environment = _Environment("articulated_open_close")

    pair = run_task_neutral_controls(
        environment=environment,
        task_spec=task,
        control_plan=_semantic_articulated_plan(task),
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    assert pair["cell_admitted_for_policy_execution"] is True
    assert environment.steps[0][-1] == 1.0
    assert environment.steps[1][-1] == 0.0


def test_cartesian_articulated_phases_use_measured_arrival_and_native_action_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import adp009d_control_episode as module

    monkeypatch.setattr(
        module,
        "_persist_observation",
        lambda *_args, **kwargs: {
            "observation_index": 0,
            "kind": kwargs["kind"],
            "views": {},
        },
    )
    monkeypatch.setattr(
        module,
        "finalize_manipulation_evaluation_visual_evidence",
        lambda **_kwargs: ({"status": "complete"}, []),
    )
    task = _task("articulated_open_close")
    pair = run_task_neutral_controls(
        environment=_CartesianEnvironment("articulated_open_close"),
        task_spec=task,
        control_plan=_cartesian_articulated_plan(task),
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    assert pair["cell_admitted_for_policy_execution"] is True
    positive = copy.deepcopy(
        __import__("json").loads(
            (tmp_path / "adp_task_control_episode.deterministic_scripted_positive.json").read_text()
        )
    )
    assert [row["phase_id"] for row in positive["phase_arrivals"]] == [
        "open",
        "retreat",
    ]
    assert all(row["target_reached"] for row in positive["phase_arrivals"])
