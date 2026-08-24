from __future__ import annotations

import copy
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_control_episode import (
    ControlEpisodeError,
    run_task_neutral_controls,
)
from blueprint_pipeline.adp009d_physics_backend_comparison import (
    build_backend_contact_configuration,
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

    def read_arm_dynamics_observation(self):
        zeros = [0.0] * 7
        return {
            "schema_version": "adp009d_arm_dynamics_observation.v2",
            "joint_position_rad": list(self.joints),
            "joint_velocity_rad_s": zeros,
            "joint_position_target_rad": list(self.joints),
            "computed_torque_nm": zeros,
            "applied_torque_nm": zeros,
            "joint_effort_limit_nm": [87.0] * 4 + [12.0] * 3,
            "joint_effort_utilization": zeros,
            "body_contact_force_world_n": None,
            "body_incoming_joint_wrench_body": {},
            "contact_envelope": None,
            "backend_contact_configuration": build_backend_contact_configuration(
                "physx"
            ),
        }

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
            "grasp_frame_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
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


class _CartesianEnvironment(_Environment):
    def scripted_action_for_pose(
        self,
        *,
        phase_id=None,
        target_position_world_m,
        target_quaternion_world_xyzw,
        gripper_command,
        max_joint_delta_rad,
        max_joint_setpoint_lead_rad,
    ):
        assert phase_id is not None
        del target_quaternion_world_xyzw
        assert max_joint_delta_rad == pytest.approx(0.03)
        assert max_joint_setpoint_lead_rad == pytest.approx(0.2)
        return [*target_position_world_m, 0.0, 0.0, 0.0, 0.0, gripper_command]


class _SolvedJointCartesianEnvironment(_CartesianEnvironment):
    def __init__(self, task_kind: str):
        super().__init__(task_kind)
        self.bounded_calls = []
        self.cartesian_targets = []

    def bounded_joint_action(self, **kwargs):
        self.bounded_calls.append(dict(kwargs))
        return [
            *[float(value) for value in kwargs["target_joint_positions_rad"]],
            float(kwargs["gripper_command"]),
        ]

    def scripted_action_for_pose(self, **kwargs):
        self.cartesian_targets.append(list(kwargs["target_position_world_m"]))
        return super().scripted_action_for_pose(**kwargs)

    def predict_grasp_frame_pose_world(
        self, joint_positions_rad, *, gripper_command=None
    ):
        assert gripper_command is not None
        return [
            *[float(value) for value in joint_positions_rad[:3]],
            0.0,
            0.0,
            0.0,
            1.0,
        ]


class _BilateralContactCartesianEnvironment(_CartesianEnvironment):
    def read_task_sample(self):
        sample = super().read_task_sample()
        if self.gripper <= 0.5:
            return sample
        closed_steps = sum(row[7] > 0.5 for row in self.steps)
        sides = ["left_inner_finger"]
        if closed_steps >= 2:
            sides.append("right_inner_finger")
        sample["native_readback"] = {
            "contact_sensor_instance_readback": {
                "task_robot_contact": [
                    {
                        "nonzero_filter_forces": [
                            {
                                "filter_prim_path_expr": (
                                    "{ENV_REGEX_NS}/Robot/Gripper/" + side
                                ),
                                "force_magnitude_n": 2.0,
                            }
                            for side in sides
                        ]
                    }
                ]
            }
        }
        return sample


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

    with pytest.raises(
        ControlEpisodeError, match="task_control_action_budget_exceeds_task_spec"
    ):
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


def test_task_neutral_control_dispatches_solved_vector_through_bounded_seam(
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
    plan = _cartesian_articulated_plan(task)
    solved = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    plan["scripted_positive_actions"][0][
        "hold_solved_arm_joint_positions_rad"
    ] = solved
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    environment = _SolvedJointCartesianEnvironment("articulated_open_close")

    pair = run_task_neutral_controls(
        environment=environment,
        task_spec=task,
        control_plan=plan,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    assert pair["cell_admitted_for_policy_execution"] is True
    assert environment.bounded_calls
    assert all(
        call["target_joint_positions_rad"] == solved
        for call in environment.bounded_calls
    )
    assert [0.9, 0.0, 0.0] not in environment.cartesian_targets
    positive = __import__("json").loads(
        (
            tmp_path
            / "adp_task_control_episode.deterministic_scripted_positive.json"
        ).read_text()
    )
    open_actions = [
        row for row in positive["action_trace"] if row["phase_id"] == "open"
    ]
    assert all(row["isaac_action"][:7] == solved for row in open_actions)
    arrival = next(
        row for row in positive["phase_arrivals"] if row["phase_id"] == "open"
    )
    assert arrival["selected_joint_positions_rad"] == solved
    assert arrival["terminal_commanded_joint_positions_rad"] == solved
    assert arrival["terminal_reached_joint_positions_rad"] == solved
    assert arrival["selected_to_commanded_joint_l2_rad"] == pytest.approx(0.0)
    assert arrival["commanded_to_reached_joint_l2_rad"] == pytest.approx(0.0)
    assert arrival["terminal_fk_grasp_frame_position_world_m"] == [0.9, 0.0, 0.0]
    assert arrival["terminal_fk_to_measured_tcp_error_m"] == pytest.approx(0.0)
    assert arrival["terminal_fk_status"] == "measured"


def test_task_neutral_control_dispatches_physics_admitted_close_through_live_dls(
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

    class _LiveDlsEnvironment(_CartesianEnvironment):
        def __init__(self):
            super().__init__("articulated_open_close")
            self.pose_calls = []

        def scripted_action_for_pose(self, **kwargs):
            self.pose_calls.append(dict(kwargs))
            preferred = kwargs.pop(
                "preferred_posture_joint_positions_rad", None
            )
            if preferred is not None:
                assert preferred == [0.1] * 7
            return super().scripted_action_for_pose(**kwargs)

    task = _task("articulated_open_close")
    plan = _cartesian_articulated_plan(task)
    close = plan["scripted_positive_actions"][0]
    close["phase_id"] = "contact_close"
    close["physx_dls_preferred_posture_joint_positions_rad"] = [0.1] * 7
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    environment = _LiveDlsEnvironment()

    pair = run_task_neutral_controls(
        environment=environment,
        task_spec=task,
        control_plan=plan,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    assert pair["cell_admitted_for_policy_execution"] is True
    assert environment.pose_calls
    positive = __import__("json").loads(
        (
            tmp_path
            / "adp_task_control_episode.deterministic_scripted_positive.json"
        ).read_text()
    )
    close_arrival = next(
        row
        for row in positive["phase_arrivals"]
        if row["phase_id"] == "contact_close"
    )
    assert close_arrival["arm_command_source"] == (
        "live_physx_dls_with_preferred_posture"
    )


def test_task_neutral_control_refuses_missing_solved_joint_dispatch(
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
    plan = _cartesian_articulated_plan(task)
    plan["scripted_positive_actions"][0][
        "hold_solved_arm_joint_positions_rad"
    ] = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    with pytest.raises(ControlEpisodeError) as excinfo:
        run_task_neutral_controls(
            environment=_CartesianEnvironment("articulated_open_close"),
            task_spec=task,
            control_plan=plan,
            gripper_open_command=0.0,
            gripper_closed_command=1.0,
            output_dir=tmp_path,
        )

    assert excinfo.value.errors == (
        "task_control_solved_joint_dispatch_unavailable:open",
    )


def test_gripper_transition_holds_arm_targets_and_runs_full_dwell(
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
    task["maximum_action_steps"] = 20

    def phase(phase_id, target, gripper_state, *, hold=False, steps=2):
        return {
            "phase_id": phase_id,
            "mode": "ik_pose",
            "target_position_world_m": target,
            "target_quaternion_world_xyzw": None,
            "gripper_state": gripper_state,
            "minimum_steps": steps if hold else 1,
            "maximum_steps": steps,
            "arrival_tolerance_m": 1.0e-6,
            "arrival_stability_steps": 1,
            "max_joint_delta_rad": 0.03,
            "max_joint_setpoint_lead_rad": 0.2,
            "hold_arm_joint_positions_during_gripper_transition": hold,
        }

    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "cell_id": "articulated-fixed-arm-gripper-dwell",
        "task_spec_digest": canonical_digest(task),
        "trajectory_source": "native_ik_preflight",
        "planner_receipt_digest": "sha256:" + "d" * 64,
        "zero_action_steps": 3,
        "scripted_positive_actions": [
            phase("contact_open", [0.0, 0.0, 0.0], "open"),
            phase("contact_close", [0.0, 0.0, 0.0], "closed", hold=True),
            phase("open", [0.9, 0.0, 0.0], "closed"),
            phase("retreat", [0.9, 1.0, 0.0], "open"),
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    environment = _CartesianEnvironment("articulated_open_close")

    pair = run_task_neutral_controls(
        environment=environment,
        task_spec=task,
        control_plan=plan,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    assert pair["cell_admitted_for_policy_execution"] is True
    positive = __import__("json").loads(
        (
            tmp_path
            / "adp_task_control_episode.deterministic_scripted_positive.json"
        ).read_text()
    )
    held = [
        row
        for row in positive["action_trace"]
        if row["phase_id"] == "contact_close"
    ]
    assert len(held) == 2
    assert [row["isaac_action"] for row in held] == [[0.0] * 7 + [1.0]] * 2
    assert [row["action_recomputed"] for row in held] == [False, False]
    assert [row["action_hold_index"] for row in held] == [0, 1]
    close_arrival = next(
        row
        for row in positive["phase_arrivals"]
        if row["phase_id"] == "contact_close"
    )
    assert close_arrival["arrival_target_source"] == (
        "previous_phase_qualified_entry_pose_held_during_gripper_transition"
    )
    assert close_arrival["target_position_world_m"] == [0.0, 0.0, 0.0]
    assert close_arrival["target_reached"] is True


def test_gripper_transition_preserves_explicit_sealed_arrival_target(
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
    task["maximum_action_steps"] = 12

    def phase(phase_id, target, gripper_state, *, hold=False):
        row = {
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
            "hold_arm_joint_positions_during_gripper_transition": hold,
        }
        if hold:
            row["arrival_target_position_world_m"] = [0.1, 0.0, 0.0]
        return row

    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "cell_id": "articulated-sealed-close-arrival",
        "task_spec_digest": canonical_digest(task),
        "trajectory_source": "native_ik_preflight",
        "planner_receipt_digest": "sha256:" + "e" * 64,
        "zero_action_steps": 3,
        "scripted_positive_actions": [
            phase("contact_open", [0.0, 0.0, 0.0], "open"),
            phase(
                "contact_close",
                [0.0, 0.0, 0.0],
                "closed",
                hold=True,
            ),
            phase("open", [0.9, 0.0, 0.0], "closed"),
            phase("retreat", [0.9, 1.0, 0.0], "open"),
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    run_task_neutral_controls(
        environment=_CartesianEnvironment("articulated_open_close"),
        task_spec=task,
        control_plan=plan,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    positive = __import__("json").loads(
        (
            tmp_path
            / "adp_task_control_episode.deterministic_scripted_positive.json"
        ).read_text()
    )
    close_arrival = next(
        row
        for row in positive["phase_arrivals"]
        if row["phase_id"] == "contact_close"
    )
    assert close_arrival["arrival_target_source"] == "sealed_arrival_pose_override"
    assert close_arrival["target_position_world_m"] == [0.1, 0.0, 0.0]
    assert close_arrival["terminal_position_world_m"] == [0.0, 0.0, 0.0]
    assert close_arrival["target_reached"] is False


def test_contact_close_requires_simultaneous_bilateral_native_contact(
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
    task["maximum_action_steps"] = 12

    def phase(phase_id, target, gripper_state, *, steps=2):
        return {
            "phase_id": phase_id,
            "mode": "ik_pose",
            "target_position_world_m": target,
            "target_quaternion_world_xyzw": None,
            "gripper_state": gripper_state,
            "minimum_steps": 1,
            "maximum_steps": steps,
            "arrival_tolerance_m": 1.0e-6,
            "arrival_stability_steps": 1,
            "max_joint_delta_rad": 0.03,
            "max_joint_setpoint_lead_rad": 0.2,
        }

    close = phase("contact_close", [0.9, 0.0, 0.0], "closed", steps=3)
    close.update(
        {
            "require_bilateral_task_contact": True,
            "bilateral_task_contact_minimum_force_n": 0.5,
        }
    )
    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "cell_id": "articulated-bilateral-contact-close",
        "task_spec_digest": canonical_digest(task),
        "trajectory_source": "native_ik_preflight",
        "planner_receipt_digest": "sha256:" + "e" * 64,
        "zero_action_steps": 3,
        "scripted_positive_actions": [
            phase("contact_open", [0.0, 0.0, 0.0], "open"),
            close,
            phase("retreat", [0.9, 1.0, 0.0], "open"),
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    pair = run_task_neutral_controls(
        environment=_BilateralContactCartesianEnvironment(
            "articulated_open_close"
        ),
        task_spec=task,
        control_plan=plan,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    assert pair["cell_admitted_for_policy_execution"] is True
    positive = __import__("json").loads(
        (
            tmp_path
            / "adp_task_control_episode.deterministic_scripted_positive.json"
        ).read_text()
    )
    close_actions = [
        row
        for row in positive["action_trace"]
        if row["phase_id"] == "contact_close"
    ]
    assert len(close_actions) == 2
    arrival = next(
        row
        for row in positive["phase_arrivals"]
        if row["phase_id"] == "contact_close"
    )
    assert arrival["target_reached"] is True
    assert arrival["terminal_bilateral_task_contact_active"] is True
    assert arrival["terminal_task_contact_pad_forces_n"] == {
        "left_inner_finger": 2.0,
        "right_inner_finger": 2.0,
    }


class _OffsetCartesianEnvironment(_CartesianEnvironment):
    """The live arm lands a fixed distance from wherever it is commanded.

    This replays the C20c/C25 paid-run class: the controller converges, but
    the measured PhysX fingertip midpoint sits a systematic ~14 mm from the
    commanded pose, so a single open-loop attempt can never satisfy the gate.
    """

    OFFSET_M = (0.014, 0.0, 0.0)

    def scripted_action_for_pose(
        self,
        *,
        phase_id=None,
        target_position_world_m,
        target_quaternion_world_xyzw,
        gripper_command,
        max_joint_delta_rad,
        max_joint_setpoint_lead_rad,
    ):
        assert phase_id is not None
        del target_quaternion_world_xyzw
        del max_joint_delta_rad, max_joint_setpoint_lead_rad
        reached = [
            float(value) + offset
            for value, offset in zip(target_position_world_m, self.OFFSET_M)
        ]
        return [*reached, 0.0, 0.0, 0.0, 0.0, gripper_command]


class _TrackingResidualSolvedJointEnvironment(_CartesianEnvironment):
    TRACKING_RESIDUAL_RAD = 0.014

    def __init__(self, task_kind: str):
        super().__init__(task_kind)
        self.bounded_calls = []
        self.commanded_joints = [0.0] * 7

    def reset(self):
        super().reset()
        self.commanded_joints = [0.0] * 7

    def bounded_joint_action(self, **kwargs):
        self.bounded_calls.append(dict(kwargs))
        return [
            *[float(value) for value in kwargs["target_joint_positions_rad"]],
            float(kwargs["gripper_command"]),
        ]

    def step(self, action):
        self.steps.append(list(action))
        self.commanded_joints = [float(value) for value in action[:7]]
        self.joints = list(self.commanded_joints)
        self.gripper = float(action[7])
        if self.gripper > 0.5:
            self.joints[0] += self.TRACKING_RESIDUAL_RAD

    def read_arm_dynamics_observation(self):
        result = super().read_arm_dynamics_observation()
        result["joint_position_target_rad"] = list(self.commanded_joints)
        return result


class _StuckCartesianEnvironment(_CartesianEnvironment):
    """The arm parks at one wrong pose no matter what is commanded."""

    STUCK_AT_M = (0.4, 0.0, 0.0)

    def scripted_action_for_pose(self, *, gripper_command, **_kwargs):
        return [*self.STUCK_AT_M, 0.0, 0.0, 0.0, 0.0, gripper_command]


def _patched_media(monkeypatch: pytest.MonkeyPatch) -> None:
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
        lambda **_kwargs: ({"status": "complete"}, []),
    )


def _recovery_plan(task: dict) -> dict:
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
        "cell_id": "articulated-open-close-recovery",
        "task_spec_digest": canonical_digest(task),
        "trajectory_source": "native_ik_preflight",
        "planner_receipt_digest": "sha256:" + "e" * 64,
        "zero_action_steps": 3,
        "scripted_positive_actions": [
            phase("open", [0.9, 0.0, 0.0], "closed"),
            phase("retreat", [0.9, 1.0, 0.0], "open"),
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def test_pose_phase_recovers_from_systematic_offset_within_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A measured 14 mm miss must trigger a biased retry, not end the run.

    Paid runs C20c and C25 each burned a full GPU cycle to learn only that
    the live arm parks ~14-15 mm from the commanded contact pose.  The
    executor now retries the phase inside the same episode, re-commanding
    the pose biased by the measured miss while the arrival gate still
    measures against the original sealed target.
    """

    _patched_media(monkeypatch)
    task = _task("articulated_open_close")
    plan = _recovery_plan(task)
    environment = _OffsetCartesianEnvironment("articulated_open_close")

    pair = run_task_neutral_controls(
        environment=environment,
        task_spec=task,
        control_plan=plan,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    assert pair["cell_admitted_for_policy_execution"] is True
    positive = __import__("json").loads(
        (
            tmp_path
            / "adp_task_control_episode.deterministic_scripted_positive.json"
        ).read_text()
    )
    open_arrivals = [
        row
        for row in positive["phase_arrivals"]
        if row["phase_id"] == "open"
    ]
    assert [row["attempt"] for row in open_arrivals] == [1, 2]
    assert open_arrivals[0]["target_reached"] is False
    assert open_arrivals[0]["recovery_strategy"] is None
    assert open_arrivals[1]["target_reached"] is True
    assert open_arrivals[1]["recovery_strategy"] == "measured_miss_compensation"
    # The gate still measures against the original sealed target; only the
    # command is biased, by exactly the measured miss.
    assert open_arrivals[1]["target_position_world_m"] == [0.9, 0.0, 0.0]
    bias = open_arrivals[1]["commanded_position_bias_m"]
    assert bias[0] == pytest.approx(-0.014)
    assert bias[1] == pytest.approx(0.0)
    assert open_arrivals[1]["terminal_position_error_m"] == pytest.approx(
        0.0, abs=1.0e-9
    )


def test_solved_joint_anchor_compensates_measured_joint_tracking_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A solved first attempt must not make every recovery attempt identical."""

    _patched_media(monkeypatch)
    task = _task("articulated_open_close")
    plan = _recovery_plan(task)
    solved = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    plan["scripted_positive_actions"][0][
        "hold_solved_arm_joint_positions_rad"
    ] = solved
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    environment = _TrackingResidualSolvedJointEnvironment(
        "articulated_open_close"
    )

    pair = run_task_neutral_controls(
        environment=environment,
        task_spec=task,
        control_plan=plan,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    assert pair["cell_admitted_for_policy_execution"] is True
    positive = __import__("json").loads(
        (
            tmp_path
            / "adp_task_control_episode.deterministic_scripted_positive.json"
        ).read_text()
    )
    open_arrivals = [
        row for row in positive["phase_arrivals"] if row["phase_id"] == "open"
    ]
    assert [row["arm_command_source"] for row in open_arrivals] == [
        "solved_joint_target",
        "joint_tracking_recovery_from_solved_branch",
    ]
    assert open_arrivals[0]["terminal_position_error_m"] == pytest.approx(0.014)
    assert open_arrivals[1]["terminal_position_error_m"] == pytest.approx(
        0.0, abs=1.0e-9
    )
    assert environment.bounded_calls
    assert environment.bounded_calls[0]["target_joint_positions_rad"] == solved
    assert environment.bounded_calls[-1]["target_joint_positions_rad"] == [
        0.886,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert open_arrivals[1]["solved_joint_command_bias_rad"] == pytest.approx(
        [-0.014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )


def test_pose_phase_recovery_escalates_the_ladder_then_fails_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pose no strategy can reach escalates every rung, then fails.

    A stuck arm never improves, so no strategy is allowed to repeat: the
    executor walks the whole ladder once and stops with the exhaustion
    sealed, rather than burning the full attempt cap on one dead strategy.
    """

    _patched_media(monkeypatch)
    task = _task("articulated_open_close")
    plan = _recovery_plan(task)
    environment = _StuckCartesianEnvironment("articulated_open_close")

    pair = run_task_neutral_controls(
        environment=environment,
        task_spec=task,
        control_plan=plan,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    assert pair["cell_admitted_for_policy_execution"] is False
    positive = __import__("json").loads(
        (
            tmp_path
            / "adp_task_control_episode.deterministic_scripted_positive.json"
        ).read_text()
    )
    open_arrivals = [
        row
        for row in positive["phase_arrivals"]
        if row["phase_id"] == "open"
    ]
    from blueprint_pipeline.adp009d_control_episode import (
        TASK_CONTROL_RECOVERY_LADDER,
    )

    assert [row["attempt"] for row in open_arrivals] == [1, 2, 3, 4, 5]
    assert all(row["target_reached"] is False for row in open_arrivals)
    # Nominal first, then each distinct rung exactly once.
    assert [row["recovery_strategy"] for row in open_arrivals] == [
        None,
        *TASK_CONTROL_RECOVERY_LADDER,
    ]
    blocker = positive["phase_execution_blocker"]
    assert blocker.startswith("scripted_control_phase_not_reached:open:")
    assert ":attempts=5" in blocker
    assert ":strategies_exhausted=True" in blocker
    # A stuck arm never improves, so best and final coincide here.
    assert ":best_attempt=1" in blocker


class _ConvergingCartesianEnvironment(_CartesianEnvironment):
    """Each distinct command lands closer, but not close enough at first.

    This is the case the old fixed cap of three destroyed: the run was
    converging and would have landed, and the cap ended it anyway -- paying a
    whole fresh GPU cycle to resume a trend this run had already established.
    """

    def __init__(self, task_kind: str) -> None:
        super().__init__(task_kind)
        self.offset = 0.04
        self._last_command: tuple[float, ...] | None = None

    def scripted_action_for_pose(
        self,
        *,
        phase_id=None,
        target_position_world_m,
        target_quaternion_world_xyzw,
        gripper_command,
        max_joint_delta_rad,
        max_joint_setpoint_lead_rad,
    ):
        assert phase_id is not None
        del target_quaternion_world_xyzw
        del max_joint_delta_rad, max_joint_setpoint_lead_rad
        command = tuple(float(value) for value in target_position_world_m)
        if self._last_command is not None and command != self._last_command:
            self.offset *= 0.5
        self._last_command = command
        reached = [command[0] + self.offset, command[1], command[2]]
        return [*reached, 0.0, 0.0, 0.0, 0.0, gripper_command]


def test_a_converging_strategy_keeps_going_instead_of_burning_a_gpu_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Improvement earns another attempt of the same strategy, past three."""

    _patched_media(monkeypatch)
    task = _task("articulated_open_close")
    task["maximum_action_steps"] = 400
    plan = _recovery_plan(task)
    # A gate the environment can actually satisfy once it has converged.
    for row in plan["scripted_positive_actions"]:
        row["arrival_tolerance_m"] = 0.002
    plan["plan_digest"] = ""
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    environment = _ConvergingCartesianEnvironment("articulated_open_close")

    pair = run_task_neutral_controls(
        environment=environment,
        task_spec=task,
        control_plan=plan,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    assert pair["cell_admitted_for_policy_execution"] is True
    positive = __import__("json").loads(
        (
            tmp_path
            / "adp_task_control_episode.deterministic_scripted_positive.json"
        ).read_text()
    )
    open_arrivals = [
        row for row in positive["phase_arrivals"] if row["phase_id"] == "open"
    ]
    # It kept going past the old three-attempt cap...
    assert len(open_arrivals) > 3
    # ...never escalated, because the trend said this strategy was working...
    assert [row["recovery_strategy"] for row in open_arrivals[1:]] == [
        "measured_miss_compensation"
    ] * (len(open_arrivals) - 1)
    # ...the error strictly improved every attempt...
    errors = [row["terminal_position_error_m"] for row in open_arrivals]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    # ...and the run landed inside the unchanged gate, in one session.
    assert open_arrivals[-1]["target_reached"] is True
    assert open_arrivals[-1]["target_position_world_m"] == [0.9, 0.0, 0.0]


def test_recovery_selector_projects_whether_a_strategy_will_land_in_time() -> None:
    """The selector is the whole policy: measured trend in, next strategy out."""

    from blueprint_pipeline.adp009d_control_episode import (
        TASK_CONTROL_RECOVERY_LADDER,
        _next_recovery_strategy,
    )

    first, second, third, fourth = TASK_CONTROL_RECOVERY_LADDER

    def choose(history, *, tolerance=0.005, remaining=5):
        return _next_recovery_strategy(
            history,
            arrival_tolerance_m=tolerance,
            remaining_attempts=remaining,
        )

    # The nominal attempt always escalates onto the first rung.
    assert choose([]) == first
    assert choose([{"strategy": None, "error_m": 0.02}]) == first

    # C28's real numbers: diverging, so escalate rather than repeat.
    assert (
        choose(
            [
                {"strategy": None, "error_m": 0.015422},
                {"strategy": first, "error_m": 0.028500},
            ]
        )
        == second
    )

    # C29's real numbers, attempt 2: improving 1.84 mm against an 8.63 mm
    # excess needs ~5 more attempts and 6 remain, so keep going.
    assert (
        choose(
            [
                {"strategy": None, "error_m": 0.015470},
                {"strategy": first, "error_m": 0.013630},
            ],
            remaining=6,
        )
        == first
    )

    # C29 attempt 3: still improving, but only 0.69 mm against a 7.94 mm
    # excess -- about eleven more attempts, with five left.  A sign check
    # would have repeated this and burned the budget; the projection escalates.
    assert (
        choose(
            [
                {"strategy": first, "error_m": 0.013630},
                {"strategy": first, "error_m": 0.012940},
            ],
            remaining=5,
        )
        == second
    )

    # A stalled strategy -- no worse, but no better -- also escalates.
    assert (
        choose(
            [
                {"strategy": second, "error_m": 0.02},
                {"strategy": third, "error_m": 0.02},
            ]
        )
        == fourth
    )

    # The ladder ends rather than looping forever.
    assert (
        choose(
            [
                {"strategy": third, "error_m": 0.02},
                {"strategy": fourth, "error_m": 0.02},
            ]
        )
        is None
    )


def test_a_plan_may_reorder_the_ladder_but_never_invent_a_rung() -> None:
    """The hypothesis ranking is per-run data, not a frozen constant.

    Whoever just read the previous run's telemetry -- an operator or an agent
    -- ranks the rungs for the next launch, with no code change.  A plan can
    reorder or narrow them; it cannot promise physics this executor has no
    implementation for.
    """

    from blueprint_pipeline.adp009d_control_episode import (
        TASK_CONTROL_RECOVERY_LADDER,
        _next_recovery_strategy,
        recovery_ladder_for_plan,
    )

    first, second, third, _fourth = TASK_CONTROL_RECOVERY_LADDER

    # Absent, malformed, unknown, or duplicated entries fall back whole --
    # never to a silently disabled recovery.
    assert recovery_ladder_for_plan({}) == TASK_CONTROL_RECOVERY_LADDER
    assert recovery_ladder_for_plan(
        {"recovery_strategy_ladder": "not_a_list"}
    ) == TASK_CONTROL_RECOVERY_LADDER
    assert recovery_ladder_for_plan(
        {"recovery_strategy_ladder": []}
    ) == TASK_CONTROL_RECOVERY_LADDER
    assert recovery_ladder_for_plan(
        {"recovery_strategy_ladder": [first, "teleport_the_gripper"]}
    ) == TASK_CONTROL_RECOVERY_LADDER
    assert recovery_ladder_for_plan(
        {"recovery_strategy_ladder": [first, first]}
    ) == TASK_CONTROL_RECOVERY_LADDER

    # A reordered, narrowed ladder is honoured, and drives escalation order.
    reordered = recovery_ladder_for_plan(
        {"recovery_strategy_ladder": [third, first]}
    )
    assert reordered == (third, first)
    assert _next_recovery_strategy([], ladder=reordered) == third
    assert (
        _next_recovery_strategy(
            [
                {"strategy": None, "error_m": 0.02},
                {"strategy": third, "error_m": 0.02},
            ],
            ladder=reordered,
        )
        == first
    )
    # A rung the run did not carry is not reachable by escalation.
    assert (
        _next_recovery_strategy(
            [{"strategy": second, "error_m": 0.02}], ladder=reordered
        )
        is None
    )


class _BestFirstThenWorseEnvironment(_CartesianEnvironment):
    """Attempt 1 is the closest; every later attempt drifts further.

    This is C32: contact_open reached 11.63 mm on its first attempt, then
    13.68, 13.30, 15.43 and 15.39 as the ladder escalated -- and the run
    reported 15.39, discarding the best result it had produced.
    """

    TARGET_X = 0.9
    ERRORS_M = (0.01163, 0.01368, 0.01330, 0.01543, 0.01539)

    def __init__(self, task_kind: str) -> None:
        super().__init__(task_kind)
        self._phase_commands: list[tuple[float, ...]] = []

    def scripted_action_for_pose(
        self,
        *,
        phase_id=None,
        target_position_world_m,
        target_quaternion_world_xyzw,
        gripper_command,
        max_joint_delta_rad,
        max_joint_setpoint_lead_rad,
    ):
        assert phase_id is not None
        del target_quaternion_world_xyzw
        del max_joint_delta_rad, max_joint_setpoint_lead_rad
        command = tuple(float(value) for value in target_position_world_m)
        # Retreats aim back at the entry pose near the origin and are tracked
        # normally; anything out at the phase target is a fresh attempt at it.
        if command[0] < 0.5:
            reached = list(command)
        else:
            if command not in self._phase_commands:
                self._phase_commands.append(command)
            index = min(
                len(self._phase_commands) - 1, len(self.ERRORS_M) - 1
            )
            # The arm settles where physics puts it, not where it was aimed:
            # biasing the command does not move the endpoint.  That is what
            # C32 measured, and it is why compensation could not converge.
            reached = [self.TARGET_X + self.ERRORS_M[index], 0.0, 0.0]
        return [*reached, 0.0, 0.0, 0.0, 0.0, gripper_command]


def test_a_phase_reports_its_best_attempt_not_its_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The blocker must carry what the phase achieved, not where it ended."""

    _patched_media(monkeypatch)
    task = _task("articulated_open_close")
    task["maximum_action_steps"] = 400
    plan = _recovery_plan(task)
    environment = _BestFirstThenWorseEnvironment("articulated_open_close")

    run_task_neutral_controls(
        environment=environment,
        task_spec=task,
        control_plan=plan,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    positive = __import__("json").loads(
        (
            tmp_path
            / "adp_task_control_episode.deterministic_scripted_positive.json"
        ).read_text()
    )
    blocker = positive["phase_execution_blocker"]
    errors = [
        row["terminal_position_error_m"]
        for row in positive["phase_arrivals"]
        if row["phase_id"] == "open"
    ]
    best = min(errors)

    assert f"error_m={best:.6f}" in blocker
    assert best < errors[-1], "the fixture must actually end worse than its best"
    assert f":final_error_m={errors[-1]:.6f}" in blocker
    assert ":best_attempt=1" in blocker
