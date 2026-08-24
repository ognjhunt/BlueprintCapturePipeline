from __future__ import annotations

import math

import pytest

from blueprint_pipeline.native_task_arena_actuator_sweep import (
    CLOSE_POSTURE_SWEEP_SCHEMA_VERSION,
    CONTACT_ACQUISITION_SWEEP_SCHEMA_VERSION,
    DEFAULT_CONTACT_APPROACH_OFFSETS_M,
    DEFAULT_WRIST_GAIN_CANDIDATES,
    DOWNSTREAM_PHASE_POSTURE_MATRIX_SCHEMA_VERSION,
    SWEEP_SCHEMA_VERSION,
    candidate_postures,
    run_actuator_posture_sweep,
    run_contact_acquisition_sweep,
    run_contact_close_posture_sweep,
    run_downstream_phase_posture_matrix,
)


TARGET = [1.0, 0.0, 0.0]


class _SweepEnvironment:
    """A wrist whose reach depends on the gains it is given.

    Models the measured mechanism rather than a number: an implicit PD joint
    delivers ``stiffness * error`` up to its effort limit, so a stiffer joint
    saturates at a smaller error and stops tracking.  With the shipped 400
    N-m/rad against 12 N-m the usable error is 0.03 rad, which is what left
    every controller short of the handle.
    """

    EFFORT_LIMIT_NM = 12.0

    def __init__(self) -> None:
        self.stiffness = 400.0
        self.damping = 80.0
        self.joints = [0.0] * 7
        self.reset_count = 0
        self.gain_writes: list[tuple[float, float]] = []
        self._peak_utilization = 0.0

    # -- gain surface -------------------------------------------------
    def write_joint_stiffness_to_sim(self, value, joint_ids=None):
        del joint_ids
        self.stiffness = float(value)
        self.gain_writes.append((self.stiffness, self.damping))

    def write_joint_damping_to_sim(self, value, joint_ids=None):
        del joint_ids
        self.damping = float(value)

    # -- episode surface ----------------------------------------------
    def reset(self) -> None:
        self.reset_count += 1
        self.joints = [0.0] * 7
        self._peak_utilization = 0.0

    def bounded_joint_action(
        self,
        *,
        target_joint_positions_rad,
        gripper_command,
        max_joint_delta_rad,
        max_joint_setpoint_lead_rad,
    ):
        del max_joint_setpoint_lead_rad
        command = []
        for target, current in zip(target_joint_positions_rad, self.joints):
            step = max(-max_joint_delta_rad, min(max_joint_delta_rad, target - current))
            command.append(current + step)
        return [*command, float(gripper_command)]

    CONTROL_PERIOD_S = 1.0 / 15.0

    def step(self, action) -> None:
        # Both PD terms draw on the same limited torque: stiffness pays for
        # position error and damping pays for the speed that closes it, so
        # the reachable travel per step is
        # effort_limit / (stiffness + damping / dt).  At the shipped 400/80
        # that is 0.0075 rad; at 40/8 it is ten times more.
        per_step = self.EFFORT_LIMIT_NM / (
            self.stiffness + self.damping / self.CONTROL_PERIOD_S
        )
        moved = []
        for commanded, current in zip(action[:7], self.joints):
            gap = commanded - current
            allowed = max(-per_step, min(per_step, gap))
            moved.append(current + allowed)
            if abs(gap) > per_step:
                self._peak_utilization = 1.0
            elif abs(gap) > 0.0:
                self._peak_utilization = max(
                    self._peak_utilization, abs(gap) / per_step
                )
        self.joints = moved

    def read_arm_joint_positions(self):
        return list(self.joints)

    def read_arm_dynamics_observation(self):
        return {
            "joint_effort_utilization": [self._peak_utilization] * 7,
        }

    def read_object_sample(self):
        # The fingertip sits where joint 5 actually got to, so a joint that
        # cannot track lands short in exactly the way the paid runs measured.
        return {"grasp_frame_position_world_m": [self.joints[4], 0.0, 0.0]}


def _postures():
    return [
        {"posture_index": 0, "seed_index": 1, "joint_positions_rad": [0.0] * 4 + [1.0, 0.0, 0.0]},
        {"posture_index": 1, "seed_index": 7, "joint_positions_rad": [0.0] * 4 + [0.5, 0.0, 0.0]},
    ]


def _sweep(environment, **overrides):
    kwargs = dict(
        environment=environment,
        robot=environment,
        arm_joint_ids=[0, 1, 2, 3, 4, 5, 6],
        target_position_world_m=TARGET,
        postures=_postures(),
        gripper_open_command=0.0,
        max_joint_delta_rad=0.05,
        max_joint_setpoint_lead_rad=0.2,
        settle_steps=60,
    )
    kwargs.update(overrides)
    return run_actuator_posture_sweep(**kwargs)


def test_one_run_returns_a_gain_by_posture_surface() -> None:
    """The sweep replaces one hypothesis per paid run with a measurement."""

    environment = _SweepEnvironment()

    report = _sweep(environment)

    assert report["schema_version"] == SWEEP_SCHEMA_VERSION
    assert report["status"] == "measured"
    assert report["cell_count"] == len(DEFAULT_WRIST_GAIN_CANDIDATES) * 2
    # Every cell reports what the arm did, never whether it passed.
    for cell in report["cells"]:
        assert cell["joint_tracking_error_rad"] is not None
        assert cell["measured_distance_to_target_m"] is not None
        assert "task_succeeded" not in cell
        assert "outcome" not in cell


def test_actuator_sweep_scores_each_parallel_jaw_posture_against_its_own_target() -> None:
    nominal = [-2**-0.5, 0.0, 0.0, 2**-0.5]
    equivalent = [0.0, 2**-0.5, 2**-0.5, 0.0]

    class _VariantEnvironment(_SweepEnvironment):
        def read_object_sample(self):
            return {
                "grasp_frame_position_world_m": [self.joints[4], 0.0, 0.0],
                "grasp_frame_orientation_world_xyzw": equivalent,
                "task_contact_active": False,
                "joint_limit_violation": False,
                "robot_collision_failure": False,
                "scene_collision_failure": False,
            }

    postures = [
        {
            "posture_index": index,
            "variant_id": variant_id,
            "joint_positions_rad": [0.0] * 4 + [1.0, 0.0, 0.0],
            "authoritative_target_position_world_m": TARGET,
            "authoritative_target_quaternion_world_xyzw": quaternion,
        }
        for index, (variant_id, quaternion) in enumerate(
            (
                ("normalized_nominal", nominal),
                ("parallel_jaw_equivalent", equivalent),
            )
        )
    ]

    report = _sweep(
        _VariantEnvironment(),
        target_orientation_world_xyzw=nominal,
        postures=postures,
        wrist_gain_candidates=((40.0, 8.0),),
        settle_steps=60,
    )

    by_variant = {cell["variant_id"]: cell for cell in report["cells"]}
    assert by_variant["normalized_nominal"][
        "measured_orientation_error_rad"
    ] == pytest.approx(math.pi)
    assert by_variant["parallel_jaw_equivalent"][
        "measured_orientation_error_rad"
    ] == pytest.approx(0.0)


def test_bounded_command_bias_is_scored_only_against_authoritative_target() -> None:
    authoritative = [0.0, 0.0, 0.0, 1.0]
    command = [0.0, math.sin(0.02), 0.0, math.cos(0.02)]
    metadata = {
        "rotation_vector_body_rad": [0.0, 0.04, 0.0],
        "close_joint_positions_rad": [0.2] * 7,
    }

    class _AuthorityEnvironment(_SweepEnvironment):
        def read_object_sample(self):
            return {
                "grasp_frame_position_world_m": [self.joints[4], 0.0, 0.0],
                "grasp_frame_orientation_world_xyzw": authoritative,
                "task_contact_active": False,
                "joint_limit_violation": False,
                "robot_collision_failure": False,
                "scene_collision_failure": False,
            }

    report = _sweep(
        _AuthorityEnvironment(),
        target_orientation_world_xyzw=authoritative,
        postures=[
            {
                "posture_index": 0,
                "joint_positions_rad": [0.0] * 4 + [1.0, 0.0, 0.0],
                "candidate_command_target_position_world_m": TARGET,
                "candidate_command_target_quaternion_world_xyzw": command,
                "authoritative_target_position_world_m": TARGET,
                "authoritative_target_quaternion_world_xyzw": authoritative,
                "bounded_orientation_candidate": metadata,
            }
        ],
        wrist_gain_candidates=((40.0, 8.0),),
        settle_steps=60,
    )

    cell = report["cells"][0]
    assert cell["bounded_orientation_candidate"] == metadata
    assert cell["measured_orientation_error_rad"] == pytest.approx(0.0)
    assert cell[
        "measured_orientation_error_to_authoritative_target_rad"
    ] == pytest.approx(0.0)
    assert cell[
        "measured_orientation_error_to_candidate_command_target_rad"
    ] == pytest.approx(0.04)


def test_actuator_sweep_publishes_incremental_cell_progress() -> None:
    progress = []

    report = _sweep(
        _SweepEnvironment(),
        progress_callback=lambda row: progress.append(row),
    )

    assert len(progress) == report["cell_count"]
    assert progress[-1]["completed_cell_count"] == report["cell_count"]
    assert progress[-1]["total_cell_count"] == report["cell_count"]
    assert progress[-1]["last_cell"] == report["cells"][-1]


def test_actuator_sweep_restores_both_original_pd_gains() -> None:
    environment = _SweepEnvironment()
    environment.data = type(
        "Data",
        (),
        {
            "joint_stiffness": [[321.0] * 7],
            "joint_damping": [[54.0] * 7],
        },
    )()

    _sweep(
        environment,
        wrist_gain_candidates=((100.0, 20.0),),
        settle_steps=1,
    )

    assert environment.stiffness == 321.0
    assert environment.damping == 54.0


def test_downstream_matrix_measures_many_branches_before_phase_five() -> None:
    class _DownstreamEnvironment(_SweepEnvironment):
        def __init__(self):
            super().__init__()
            self.checkpoint_resets = []

        def reset_to_diagnostic_checkpoint(
            self,
            *,
            arm_joint_positions_rad,
            task_joint_positions_rad,
        ):
            self.reset()
            self.joints = list(arm_joint_positions_rad)
            self.checkpoint_resets.append(
                {
                    "arm": list(arm_joint_positions_rad),
                    "task": dict(task_joint_positions_rad),
                }
            )

        def read_object_sample(self):
            return {
                "grasp_frame_position_world_m": [self.joints[4], 0.0, 0.0],
                "grasp_frame_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "task_contact_active": False,
            }

    environment = _DownstreamEnvironment()
    progress_rows = []
    phase_ids = ("joint_path_01", "retreat")
    plan = {
        "scripted_positive_actions": [
            {
                "phase_id": phase_id,
                "target_position_world_m": [1.0, 0.0, 0.0],
                "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "arrival_tolerance_m": 0.02,
                "arrival_orientation_tolerance_rad": 0.08,
                "gripper_state": (
                    "closed"
                    if phase_id in {"contact_close", "joint_path_01"}
                    else "open"
                ),
                "max_joint_delta_rad": 1.0,
                "max_joint_setpoint_lead_rad": 1.0,
                **(
                    {"expected_joint_positions": {"door": 0.8}}
                    if phase_id == "release"
                    else {}
                ),
            }
            for phase_id in (
                "contact_close",
                "joint_path_01",
                "release",
                "retreat",
            )
        ]
    }
    global_ik = {
        "phases": [
            {
                "phase_id": phase_id,
                "attempts": [
                    {
                        "solved": True,
                        "seed_index": index,
                        "joint_positions_rad": [0.0] * 4
                        + [value, 0.0, 0.0],
                        "position_error_m": 0.001,
                    }
                    for index, value in enumerate((1.0, 0.5))
                ],
            }
            for phase_id in (
                "contact_close",
                "joint_path_01",
                "release",
                "retreat",
            )
        ]
    }

    report = run_downstream_phase_posture_matrix(
        environment=environment,
        robot=environment,
        arm_joint_ids=list(range(7)),
        control_plan=plan,
        global_ik=global_ik,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        phase_ids=phase_ids,
        wrist_gain_candidates=((400.0, 80.0), (40.0, 8.0)),
        settle_steps=60,
        progress_callback=progress_rows.append,
    )

    assert report["schema_version"] == (
        DOWNSTREAM_PHASE_POSTURE_MATRIX_SCHEMA_VERSION
    )
    assert report["status"] == "measured"
    assert report["represented_configuration_count"] == 8
    assert report["executed_cell_count"] == 8
    assert [row["phase_id"] for row in report["phase_reports"]] == list(
        phase_ids
    )
    assert all(
        "task_succeeded" not in cell
        for phase in report["phase_reports"]
        for cell in phase["cells"]
    )
    assert "asserts_no_phase_admission_or_task_outcome" in report[
        "claim_boundary"
    ]
    assert len(progress_rows) == 2
    assert [row["completed_phase_count"] for row in progress_rows] == [1, 2]
    assert progress_rows[-1]["represented_configuration_count"] == 8
    assert progress_rows[-1]["executed_cell_count"] == 8
    assert progress_rows[-1]["last_phase"]["phase_id"] == "retreat"
    assert "task_succeeded" not in progress_rows[-1]
    assert len(environment.checkpoint_resets) == 8
    assert all(
        cell["checkpoint_source_phase_id"] == "contact_close"
        for cell in report["phase_reports"][0]["cells"]
    )
    assert all(
        cell["checkpoint_gripper_command"] == 1.0
        and cell["checkpoint_settle_steps"] == 8
        for cell in report["phase_reports"][0]["cells"]
    )
    assert all(
        cell["checkpoint_source_phase_id"] == "release"
        for cell in report["phase_reports"][1]["cells"]
    )
    assert all(
        cell["checkpoint_gripper_command"] == 0.0
        for cell in report["phase_reports"][1]["cells"]
    )
    assert environment.checkpoint_resets[-1]["task"] == {"door": 0.8}


def test_close_sweep_measures_every_branch_and_admits_only_physical_grasp() -> None:
    class _CloseEnvironment:
        def __init__(self) -> None:
            self.joints = [0.0] * 7
            self.gripper = 0.0

        def reset(self):
            self.joints = [0.0] * 7
            self.gripper = 0.0

        def bounded_joint_action(self, **kwargs):
            return [
                *[float(value) for value in kwargs["target_joint_positions_rad"]],
                float(kwargs["gripper_command"]),
            ]

        def step(self, action):
            self.joints = list(action[:7])
            self.gripper = float(action[7])

        def read_arm_joint_positions(self):
            return list(self.joints)

        def predict_grasp_frame_pose_world(self, joints, *, gripper_command=None):
            del gripper_command
            return [float(joints[0]), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        def read_task_sample(self):
            good = abs(self.joints[0] - 0.2) < 1.0e-9 and self.gripper == 1.0
            forces = []
            if good:
                forces = [
                    {
                        "filter_prim_path_expr": side,
                        "force_magnitude_n": 1.0,
                    }
                    for side in ("left_inner_finger", "right_inner_finger")
                ]
            return {
                "grasp_frame_position_world_m": [self.joints[0], 0.0, 0.0],
                "grasp_frame_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "native_readback": {
                    "contact_sensor_instance_readback": {
                        "task_robot_contact": [
                            {"nonzero_filter_forces": forces}
                        ]
                    }
                },
            }

    report = run_contact_close_posture_sweep(
        environment=_CloseEnvironment(),
        target_position_world_m=[0.2, 0.0, 0.0],
        target_orientation_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        postures=[
            {"posture_index": 0, "seed_index": 3, "joint_positions_rad": [0.8] * 7},
            {"posture_index": 1, "seed_index": 7, "joint_positions_rad": [0.2] * 7},
        ],
        preposition_joint_positions_rad=[0.0] * 7,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        max_joint_delta_rad=1.0,
        max_joint_setpoint_lead_rad=1.0,
        arrival_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
        bilateral_contact_minimum_force_n=0.5,
        preposition_steps=1,
        settle_steps=2,
    )

    assert report["schema_version"] == CLOSE_POSTURE_SWEEP_SCHEMA_VERSION
    assert report["cell_count"] == 2
    assert report["admitted_cell_count"] == 1
    assert report["best_cell"]["seed_index"] == 7
    assert report["best_cell"]["admitted"] is True
    assert report["best_cell"]["commanded_to_reached_joint_l2_rad"] == 0.0
    assert report["best_cell"]["fk_to_measured_tcp_error_m"] == 0.0


def test_close_sweep_compares_joint_replay_with_live_physx_dls_per_branch() -> None:
    class _ControllerEnvironment:
        def __init__(self) -> None:
            self.joints = [0.0] * 7
            self.gripper = 0.0
            self.mode = "reset"
            self.dls_calls: list[dict[str, object]] = []

        def reset(self):
            self.joints = [0.0] * 7
            self.gripper = 0.0
            self.mode = "reset"

        def bounded_joint_action(self, **kwargs):
            self.mode = "joint"
            return [
                *[float(value) for value in kwargs["target_joint_positions_rad"]],
                float(kwargs["gripper_command"]),
            ]

        def scripted_action_for_pose(self, **kwargs):
            self.mode = "dls"
            self.dls_calls.append(dict(kwargs))
            # The live measured-TCP controller closes the residual that the
            # same preferred joint posture leaves behind under replay.
            return [0.2] * 7 + [float(kwargs["gripper_command"])]

        def step(self, action):
            self.joints = list(action[:7])
            self.gripper = float(action[7])

        def read_arm_joint_positions(self):
            return list(self.joints)

        def predict_grasp_frame_pose_world(self, joints, *, gripper_command=None):
            del gripper_command
            return [float(joints[0]), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        def read_task_sample(self):
            measured_x = 0.2 if self.mode == "dls" else self.joints[0] + 0.01
            forces = (
                [
                    {
                        "filter_prim_path_expr": side,
                        "force_magnitude_n": 1.0,
                    }
                    for side in ("left_inner_finger", "right_inner_finger")
                ]
                if self.mode == "dls"
                else []
            )
            return {
                "grasp_frame_position_world_m": [measured_x, 0.0, 0.0],
                "grasp_frame_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "native_readback": {
                    "contact_sensor_instance_readback": {
                        "task_robot_contact": [
                            {"nonzero_filter_forces": forces}
                        ]
                    }
                },
            }

    environment = _ControllerEnvironment()
    report = run_contact_close_posture_sweep(
        environment=environment,
        target_position_world_m=[0.2, 0.0, 0.0],
        target_orientation_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        postures=[
            {
                "posture_index": 0,
                "seed_index": 7,
                "joint_positions_rad": [0.19] * 7,
            }
        ],
        preposition_joint_positions_rad=[0.0] * 7,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        max_joint_delta_rad=1.0,
        max_joint_setpoint_lead_rad=1.0,
        arrival_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
        bilateral_contact_minimum_force_n=0.5,
        preposition_steps=1,
        settle_steps=2,
        compare_physx_dls=True,
    )

    assert report["controller_modes"] == [
        "bounded_joint_replay",
        "live_physx_dls",
    ]
    assert report["cell_count"] == 2
    by_mode = {cell["controller_mode"]: cell for cell in report["cells"]}
    assert by_mode["bounded_joint_replay"]["admitted"] is False
    assert by_mode["live_physx_dls"]["admitted"] is True
    assert report["best_cell"]["controller_mode"] == "live_physx_dls"
    assert environment.dls_calls[0]["target_position_world_m"] == [0.2, 0.0, 0.0]
    assert environment.dls_calls[0][
        "preferred_posture_joint_positions_rad"
    ] == [0.19] * 7


def test_close_sweep_folds_measured_closed_tcp_residual_back_into_ik() -> None:
    class _ClosedCalibrationEnvironment:
        def __init__(self) -> None:
            self.joints = [0.0] * 7
            self.gripper = 0.0

        def reset(self):
            self.joints = [0.0] * 7
            self.gripper = 0.0

        def bounded_joint_action(self, **kwargs):
            return [
                *[float(value) for value in kwargs["target_joint_positions_rad"]],
                float(kwargs["gripper_command"]),
            ]

        def step(self, action):
            self.joints = list(action[:7])
            self.gripper = float(action[7])

        def read_arm_joint_positions(self):
            return list(self.joints)

        def predict_grasp_frame_pose_world(self, joints, *, gripper_command=None):
            del gripper_command
            return [float(joints[0]), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        def read_task_sample(self):
            # PhysX's closed-pad frame is +10 mm from the solver model. The
            # first posture therefore misses; one measured-residual update
            # asks IK for 0.19 m and lands the physical frame at 0.20 m.
            measured_x = self.joints[0] + (0.01 if self.gripper == 1.0 else 0.0)
            forces = (
                [
                    {
                        "filter_prim_path_expr": side,
                        "force_magnitude_n": 1.0,
                    }
                    for side in ("left_inner_finger", "right_inner_finger")
                ]
                if self.gripper == 1.0
                else []
            )
            return {
                "grasp_frame_position_world_m": [measured_x, 0.0, 0.0],
                "grasp_frame_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "native_readback": {
                    "contact_sensor_instance_readback": {
                        "task_robot_contact": [
                            {"nonzero_filter_forces": forces}
                        ]
                    }
                },
            }

    solve_calls: list[tuple[list[float], list[float]]] = []

    def _solve(target, seed):
        solve_calls.append((list(target), list(seed)))
        return [float(target[0]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    report = run_contact_close_posture_sweep(
        environment=_ClosedCalibrationEnvironment(),
        target_position_world_m=[0.2, 0.0, 0.0],
        target_orientation_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        postures=[
            {
                "posture_index": 0,
                "seed_index": 7,
                "joint_positions_rad": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ],
        preposition_joint_positions_rad=[0.0] * 7,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        max_joint_delta_rad=1.0,
        max_joint_setpoint_lead_rad=1.0,
        arrival_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
        bilateral_contact_minimum_force_n=0.5,
        preposition_steps=1,
        settle_steps=2,
        solve=_solve,
        solver_target_position_world_m=[0.2, 0.0, 0.0],
        max_calibration_iterations=4,
        bilateral_stability_steps=2,
    )

    assert report["calibration_enabled"] is True
    assert report["branch_count"] == 1
    assert report["cell_count"] == 2
    assert report["admitted_cell_count"] == 1
    assert report["cells"][0]["measured_residual_to_target_m"] == pytest.approx(
        [0.01, 0.0, 0.0]
    )
    assert report["cells"][1]["solver_target_position_world_m"] == pytest.approx(
        [0.19, 0.0, 0.0]
    )
    assert report["best_cell"]["commanded_joint_positions_rad"][0] == pytest.approx(
        0.19
    )
    assert report["best_cell"]["maximum_consecutive_bilateral_steps"] == 2
    assert solve_calls[0][0] == pytest.approx([0.19, 0.0, 0.0])


def test_contact_acquisition_represents_125_cells_in_one_loaded_scene() -> None:
    class _AcquisitionEnvironment:
        def __init__(self) -> None:
            self.joints = [0.0] * 7
            self.gripper = 0.0
            self.reset_count = 0

        def reset(self):
            self.joints = [0.0] * 7
            self.gripper = 0.0
            self.reset_count += 1

        def bounded_joint_action(self, **kwargs):
            return [
                *[float(value) for value in kwargs["target_joint_positions_rad"]],
                float(kwargs["gripper_command"]),
            ]

        def scripted_action_for_pose(self, **kwargs):
            target = [float(value) for value in kwargs["target_position_world_m"]]
            return [*target, 0.0, 0.0, 0.0, 0.0, float(kwargs["gripper_command"])]

        def step(self, action):
            self.joints = [float(value) for value in action[:7]]
            self.gripper = float(action[7])

        def read_arm_joint_positions(self):
            return list(self.joints)

        def read_task_sample(self):
            measured_position = [
                self.joints[0] + (0.1 if self.gripper == 1.0 else 0.0),
                self.joints[1],
                self.joints[2],
            ]
            bilateral = self.gripper == 1.0 and measured_position == [1.0, 0.0, 0.0]
            forces = (
                [
                    {
                        "filter_prim_path_expr": side,
                        "force_magnitude_n": 1.0,
                        "force_world_n": [
                            0.0,
                            1.0 if side == "left_inner_finger" else -1.0,
                            0.0,
                        ],
                    }
                    for side in ("left_inner_finger", "right_inner_finger")
                ]
                if bilateral
                else []
            )
            return {
                "grasp_frame_position_world_m": measured_position,
                "grasp_frame_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "gripper_width_m": 0.02 if self.gripper == 1.0 else 0.10,
                "native_readback": {
                    "contact_sensor_instance_readback": {
                        "task_robot_contact": [
                            {"nonzero_filter_forces": forces}
                        ]
                    }
                },
            }

    environment = _AcquisitionEnvironment()
    progress: list[dict[str, object]] = []
    report = run_contact_acquisition_sweep(
        environment=environment,
        authored_target_position_world_m=[1.0, 0.0, 0.0],
        command_target_position_world_m=[0.9, 0.0, 0.0],
        target_orientation_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preposition_joint_positions_rad=[0.0] * 7,
        approach_axis_world=[1.0, 0.0, 0.0],
        jaw_axis_world=[0.0, 1.0, 0.0],
        lateral_axis_world=[0.0, 0.0, 1.0],
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        max_joint_delta_rad=1.0,
        max_joint_setpoint_lead_rad=1.0,
        arrival_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
        bilateral_contact_minimum_force_n=0.5,
        preposition_steps=1,
        advance_steps=1,
        close_steps=3,
        stop_after_admitted_cells=1,
        progress_callback=lambda value: progress.append(dict(value)),
    )

    assert report["schema_version"] == CONTACT_ACQUISITION_SWEEP_SCHEMA_VERSION
    assert report["represented_cell_count"] == 225
    assert report["executed_cell_count"] == 1
    assert report["admitted_cell_count"] == 1
    assert report["best_cell"]["admitted"] is True
    assert report["best_cell"]["maximum_consecutive_bilateral_steps"] == 2
    assert report["best_cell"]["open_pose_gate_triggered"] is True
    assert report["best_cell"]["open_contact_triggered"] is False
    assert report["best_cell"]["open_advance_trigger_reasons"] == [
        "pose_gate"
    ]
    assert report["best_cell"]["executed_close_steps"] == 2
    assert report["best_cell"]["close_phase_gate_triggered"] is True
    force_evidence = report["best_cell"]["best_bilateral_force_evidence"]
    assert force_evidence["opposed_jaw_contact_active"] is True
    assert force_evidence["opposed_jaw_force_min_n"] == 1.0
    assert force_evidence["same_direction_approach_contact_active"] is False
    assert report["best_cell"]["reached_open_joint_positions_rad"][:3] == [
        0.9,
        0.0,
        0.0,
    ]
    assert report["best_cell"]["candidate_command_target_position_world_m"] == [
        0.9,
        0.0,
        0.0,
    ]
    assert report["best_cell"]["terminal_grasp_frame_shift_from_open_m"] == (
        pytest.approx(0.1)
    )
    # One reset for the cell plus a cleanup reset; no provider or scene reload.
    assert environment.reset_count == 2
    assert [item["status"] for item in progress] == ["running", "measured"]
    assert progress[0]["executed_cell_count"] == 1
    assert progress[0]["admitted_cell_count"] == 1
    assert progress[0]["last_cell"]["cell_index"] == 0
    assert progress[-1]["best_cell"]["admitted"] is True


def test_contact_acquisition_default_approach_grid_is_symmetric() -> None:
    assert DEFAULT_CONTACT_APPROACH_OFFSETS_M == (
        -0.020,
        -0.015,
        -0.010,
        -0.005,
        0.0,
        0.005,
        0.010,
        0.015,
        0.020,
    )


def test_contact_acquisition_closes_at_first_task_pad_contact() -> None:
    class _ContactOnsetEnvironment:
        def __init__(self) -> None:
            self.joints = [0.0] * 7
            self.gripper = 0.0
            self.pose_calls = 0

        def reset(self):
            self.joints = [0.0] * 7
            self.gripper = 0.0

        def bounded_joint_action(self, **kwargs):
            return [
                *[float(value) for value in kwargs["target_joint_positions_rad"]],
                float(kwargs["gripper_command"]),
            ]

        def scripted_action_for_pose(self, **kwargs):
            self.pose_calls += 1
            # The second advance step is the useful first-pad-contact moment.
            # A fixed-horizon controller would execute the third step, drive
            # through that moment, and close after contact has been lost.
            x = (0.99, 1.0, 1.02)[min(self.pose_calls - 1, 2)]
            return [x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(kwargs["gripper_command"])]

        def step(self, action):
            self.joints = [float(value) for value in action[:7]]
            self.gripper = float(action[7])

        def read_arm_joint_positions(self):
            return list(self.joints)

        def read_task_sample(self):
            at_contact = self.joints[0] == 1.0
            sides = (
                ("left_inner_finger", "right_inner_finger")
                if self.gripper == 1.0
                else ("left_inner_finger",)
            )
            forces = [
                {
                    "filter_prim_path_expr": side,
                    "force_magnitude_n": 2.0,
                }
                for side in sides
            ] if at_contact else []
            return {
                "grasp_frame_position_world_m": list(self.joints[:3]),
                "grasp_frame_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "gripper_width_m": 0.02 if self.gripper == 1.0 else 0.10,
                "native_readback": {
                    "contact_sensor_instance_readback": {
                        "task_robot_contact": [
                            {"nonzero_filter_forces": forces}
                        ]
                    }
                },
            }

    environment = _ContactOnsetEnvironment()
    report = run_contact_acquisition_sweep(
        environment=environment,
        authored_target_position_world_m=[1.0, 0.0, 0.0],
        target_orientation_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preposition_joint_positions_rad=[0.0] * 7,
        approach_axis_world=[1.0, 0.0, 0.0],
        jaw_axis_world=[0.0, 1.0, 0.0],
        lateral_axis_world=[0.0, 0.0, 1.0],
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        max_joint_delta_rad=1.0,
        max_joint_setpoint_lead_rad=1.0,
        arrival_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
        bilateral_contact_minimum_force_n=0.5,
        approach_offsets_m=[0.0],
        jaw_offsets_m=[0.0],
        lateral_offsets_m=[0.0],
        preposition_steps=1,
        advance_steps=3,
        close_steps=2,
    )

    cell = report["best_cell"]
    assert report["admitted_cell_count"] == 1
    assert environment.pose_calls == 2
    assert cell["open_contact_triggered"] is True
    assert cell["open_pose_gate_triggered"] is True
    assert cell["open_advance_trigger_reasons"] == [
        "task_pad_contact",
        "pose_gate",
    ]
    assert cell["open_contact_trigger_step_index"] == 1
    assert cell["executed_open_advance_steps"] == 2
    assert cell["maximum_consecutive_open_contact_steps"] == 1
    assert cell["maximum_consecutive_open_bilateral_steps"] == 0
    assert cell["open_contact_trigger_pad_forces_n"] == {
        "left_inner_finger": 2.0,
    }
    assert cell["terminal_maximum_joint_drift_from_frozen_rad"] == 0.0
    assert cell["terminal_grasp_frame_shift_from_open_m"] == 0.0
    assert cell["terminal_open_advance_gripper_width_m"] == 0.10
    assert cell["commanded_close_gripper_value"] == 1.0
    assert cell["minimum_close_gripper_width_m"] == 0.02
    assert cell["maximum_close_gripper_width_m"] == 0.02
    assert cell["first_bilateral_close_step_index"] == 0
    assert cell["last_bilateral_close_step_index"] == 1
    assert cell["executed_close_steps"] == 2
    assert cell["close_phase_gate_triggered"] is True


def test_contact_acquisition_rejects_one_finger_contact() -> None:
    class _OneFingerEnvironment:
        def __init__(self) -> None:
            self.joints = [0.0] * 7
            self.gripper = 0.0

        def reset(self):
            self.joints = [0.0] * 7
            self.gripper = 0.0

        def bounded_joint_action(self, **kwargs):
            return [
                *[float(value) for value in kwargs["target_joint_positions_rad"]],
                float(kwargs["gripper_command"]),
            ]

        def scripted_action_for_pose(self, **kwargs):
            target = [float(value) for value in kwargs["target_position_world_m"]]
            return [*target, 0.0, 0.0, 0.0, 0.0, float(kwargs["gripper_command"])]

        def step(self, action):
            self.joints = [float(value) for value in action[:7]]
            self.gripper = float(action[7])

        def read_arm_joint_positions(self):
            return list(self.joints)

        def read_task_sample(self):
            forces = (
                [
                    {
                        "filter_prim_path_expr": "left_inner_finger",
                        "force_magnitude_n": 2.0,
                    }
                ]
                if self.gripper == 1.0
                else []
            )
            return {
                "grasp_frame_position_world_m": list(self.joints[:3]),
                "grasp_frame_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "native_readback": {
                    "contact_sensor_instance_readback": {
                        "task_robot_contact": [
                            {"nonzero_filter_forces": forces}
                        ]
                    }
                },
            }

    report = run_contact_acquisition_sweep(
        environment=_OneFingerEnvironment(),
        authored_target_position_world_m=[1.0, 0.0, 0.0],
        target_orientation_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preposition_joint_positions_rad=[0.0] * 7,
        approach_axis_world=[1.0, 0.0, 0.0],
        jaw_axis_world=[0.0, 1.0, 0.0],
        lateral_axis_world=[0.0, 0.0, 1.0],
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        max_joint_delta_rad=1.0,
        max_joint_setpoint_lead_rad=1.0,
        arrival_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
        bilateral_contact_minimum_force_n=0.5,
        approach_offsets_m=[0.0],
        jaw_offsets_m=[0.0],
        lateral_offsets_m=[0.0],
        preposition_steps=1,
        advance_steps=1,
        close_steps=2,
    )

    assert report["executed_cell_count"] == 1
    assert report["admitted_cell_count"] == 0
    assert report["cells"][0]["peak_close_pad_forces_n"] == {
        "left_inner_finger": 2.0
    }
    assert report["cells"][0]["terminal_bilateral_task_contact_active"] is False


def test_contact_acquisition_rejects_bilateral_contact_outside_arrival_gate() -> None:
    class _BilateralMissEnvironment:
        def __init__(self) -> None:
            self.joints = [0.0] * 7
            self.gripper = 0.0

        def reset(self):
            self.joints = [0.0] * 7
            self.gripper = 0.0

        def bounded_joint_action(self, **kwargs):
            return [
                *[float(value) for value in kwargs["target_joint_positions_rad"]],
                float(kwargs["gripper_command"]),
            ]

        def scripted_action_for_pose(self, **kwargs):
            target = [float(value) for value in kwargs["target_position_world_m"]]
            target[0] += 0.010
            return [*target, 0.0, 0.0, 0.0, 0.0, float(kwargs["gripper_command"])]

        def step(self, action):
            self.joints = [float(value) for value in action[:7]]
            self.gripper = float(action[7])

        def read_arm_joint_positions(self):
            return list(self.joints)

        def read_task_sample(self):
            forces = (
                [
                    {
                        "filter_prim_path_expr": side,
                        "force_magnitude_n": 1.0,
                        "force_world_n": [1.0, 0.0, 0.0],
                    }
                    for side in ("left_inner_finger", "right_inner_finger")
                ]
                if self.gripper == 1.0
                else []
            )
            return {
                "grasp_frame_position_world_m": list(self.joints[:3]),
                "grasp_frame_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "native_readback": {
                    "contact_sensor_instance_readback": {
                        "task_robot_contact": [
                            {"nonzero_filter_forces": forces}
                        ]
                    }
                },
            }

    report = run_contact_acquisition_sweep(
        environment=_BilateralMissEnvironment(),
        authored_target_position_world_m=[1.0, 0.0, 0.0],
        target_orientation_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preposition_joint_positions_rad=[0.0] * 7,
        approach_axis_world=[1.0, 0.0, 0.0],
        jaw_axis_world=[0.0, 1.0, 0.0],
        lateral_axis_world=[0.0, 0.0, 1.0],
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        max_joint_delta_rad=1.0,
        max_joint_setpoint_lead_rad=1.0,
        arrival_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
        bilateral_contact_minimum_force_n=0.5,
        approach_offsets_m=[0.0],
        jaw_offsets_m=[0.0],
        lateral_offsets_m=[0.0],
        preposition_steps=1,
        advance_steps=1,
        close_steps=2,
    )

    assert report["executed_cell_count"] == 1
    assert report["admitted_cell_count"] == 0
    assert report["cells"][0]["terminal_bilateral_task_contact_active"] is True
    force_evidence = report["cells"][0]["best_bilateral_force_evidence"]
    assert force_evidence["opposed_jaw_contact_active"] is False
    assert force_evidence["same_direction_approach_contact_active"] is True
    assert force_evidence["same_direction_approach_force_min_n"] == 1.0
    assert report["cells"][0][
        "terminal_distance_to_candidate_target_m"
    ] == pytest.approx(0.010)


def test_contact_acquisition_rejects_shifted_candidate_outside_authored_gate() -> None:
    class _ShiftedCandidateEnvironment:
        def __init__(self) -> None:
            self.joints = [0.0] * 7
            self.gripper = 0.0

        def reset(self):
            self.joints = [0.0] * 7
            self.gripper = 0.0

        def bounded_joint_action(self, **kwargs):
            return [
                *[float(value) for value in kwargs["target_joint_positions_rad"]],
                float(kwargs["gripper_command"]),
            ]

        def scripted_action_for_pose(self, **kwargs):
            target = [float(value) for value in kwargs["target_position_world_m"]]
            return [*target, 0.0, 0.0, 0.0, 0.0, float(kwargs["gripper_command"])]

        def step(self, action):
            self.joints = [float(value) for value in action[:7]]
            self.gripper = float(action[7])

        def read_arm_joint_positions(self):
            return list(self.joints)

        def read_task_sample(self):
            forces = (
                [
                    {
                        "filter_prim_path_expr": side,
                        "force_magnitude_n": 1.0,
                    }
                    for side in ("left_inner_finger", "right_inner_finger")
                ]
                if self.gripper == 1.0
                else []
            )
            return {
                "grasp_frame_position_world_m": list(self.joints[:3]),
                "grasp_frame_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "native_readback": {
                    "contact_sensor_instance_readback": {
                        "task_robot_contact": [
                            {"nonzero_filter_forces": forces}
                        ]
                    }
                },
            }

    report = run_contact_acquisition_sweep(
        environment=_ShiftedCandidateEnvironment(),
        authored_target_position_world_m=[1.0, 0.0, 0.0],
        target_orientation_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        preposition_joint_positions_rad=[0.0] * 7,
        approach_axis_world=[1.0, 0.0, 0.0],
        jaw_axis_world=[0.0, 1.0, 0.0],
        lateral_axis_world=[0.0, 0.0, 1.0],
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        max_joint_delta_rad=1.0,
        max_joint_setpoint_lead_rad=1.0,
        arrival_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
        bilateral_contact_minimum_force_n=0.5,
        approach_offsets_m=[-0.010],
        jaw_offsets_m=[0.0],
        lateral_offsets_m=[0.0],
        preposition_steps=1,
        advance_steps=1,
        close_steps=2,
    )

    cell = report["cells"][0]
    assert cell["candidate_target_gate_passed"] is True
    assert cell["authored_target_gate_passed"] is False
    assert cell["terminal_bilateral_task_contact_active"] is True
    assert cell["maximum_consecutive_bilateral_steps"] == 2
    assert cell["terminal_distance_to_candidate_target_m"] == pytest.approx(0.0)
    assert cell["terminal_distance_to_authored_target_m"] == pytest.approx(0.010)
    assert cell["admitted"] is False
    assert report["admitted_cell_count"] == 0


def test_close_sweep_isolates_one_bad_branch_instead_of_losing_the_surface() -> None:
    class _OneBadBranch(_SweepEnvironment):
        def step(self, action):
            if float(action[0]) > 0.7:
                raise TypeError(
                    "float() argument must be a string or a real number, not 'NoneType'"
                )
            self.joints = [float(value) for value in action[:7]]

        def read_task_sample(self):
            return {
                "grasp_frame_position_world_m": [self.joints[0], 0.0, 0.0],
                "grasp_frame_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            }

        def predict_grasp_frame_pose_world(self, joints, *, gripper_command=None):
            del gripper_command
            return [float(joints[0]), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    report = run_contact_close_posture_sweep(
        environment=_OneBadBranch(),
        target_position_world_m=[0.2, 0.0, 0.0],
        target_orientation_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        postures=[
            {"posture_index": 0, "joint_positions_rad": [0.8] * 7},
            {"posture_index": 1, "joint_positions_rad": [0.2] * 7},
        ],
        preposition_joint_positions_rad=[0.0] * 7,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        max_joint_delta_rad=1.0,
        max_joint_setpoint_lead_rad=1.0,
        arrival_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
        bilateral_contact_minimum_force_n=0.5,
        preposition_steps=1,
        settle_steps=2,
    )

    assert report["cell_count"] == 2
    assert report["cells"][0]["status"] == "cell_error"
    assert report["cells"][0]["error"].startswith("close:TypeError:")
    assert report["cells"][1]["measured_distance_to_target_m"] == 0.0


def test_close_sweep_preserves_cells_when_final_cleanup_fails() -> None:
    class _CleanupFails(_SweepEnvironment):
        def reset(self) -> None:
            super().reset()
            if self.reset_count == 3:
                raise TypeError(
                    "float() argument must be a string or a real number, not 'NoneType'"
                )

        def read_task_sample(self):
            return {
                "grasp_frame_position_world_m": [self.joints[0], 0.0, 0.0],
                "grasp_frame_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            }

    report = run_contact_close_posture_sweep(
        environment=_CleanupFails(),
        target_position_world_m=[0.2, 0.0, 0.0],
        target_orientation_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        postures=[
            {"posture_index": 0, "joint_positions_rad": [0.1] * 7},
            {"posture_index": 1, "joint_positions_rad": [0.2] * 7},
        ],
        preposition_joint_positions_rad=[0.0] * 7,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        max_joint_delta_rad=1.0,
        max_joint_setpoint_lead_rad=1.0,
        arrival_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
        bilateral_contact_minimum_force_n=0.5,
        preposition_steps=1,
        settle_steps=1,
    )

    assert report["status"] == "measured"
    assert report["cell_count"] == 2
    assert len(report["cells"]) == 2
    assert report["cleanup_error"].startswith("final_reset:TypeError:")


def test_the_sweep_separates_gains_that_can_track_from_gains_that_cannot() -> None:
    """The shipped stiffness is measurably the worst cell, not a guess."""

    environment = _SweepEnvironment()

    report = _sweep(environment)

    shipped = [
        cell
        for cell in report["cells"]
        if cell["wrist_stiffness_nm_per_rad"] == 400.0 and cell["posture_index"] == 0
    ][0]
    softer = [
        cell
        for cell in report["cells"]
        if cell["wrist_stiffness_nm_per_rad"] == 40.0 and cell["posture_index"] == 0
    ][0]

    # 12 / 400 = 0.03 rad of usable error against 12 / 40 = 0.3 rad.
    assert shipped["joint_tracking_error_rad"] > softer["joint_tracking_error_rad"]
    assert shipped["measured_distance_to_target_m"] > softer[
        "measured_distance_to_target_m"
    ]
    assert shipped["wrist_peak_effort_utilization"] == pytest.approx(1.0)
    # And the surface names the best cell rather than leaving it to be eyeballed.
    best = report["best_cell"]
    assert best["measured_distance_to_target_m"] == min(
        cell["measured_distance_to_target_m"] for cell in report["cells"]
    )


def test_the_sweep_restores_the_gains_it_borrowed() -> None:
    """A diagnostic must not retune the robot the controls then measure."""

    environment = _SweepEnvironment()

    report = _sweep(environment)

    assert report["gains_restored"] is True
    assert environment.stiffness == pytest.approx(400.0)
    assert environment.damping == pytest.approx(80.0)
    # And it leaves the arm reset rather than parked at the last cell.
    assert environment.joints == [0.0] * 7


def test_a_runtime_that_cannot_retune_is_reported_not_fatal() -> None:
    """Measurement is optional; the controls behind it are not."""

    class _NoGains(_SweepEnvironment):
        write_joint_stiffness_to_sim = None

    report = _sweep(_NoGains())

    assert report["status"] == "unavailable"
    assert report["cells"] == []
    assert "gain_write" in report["reason"]


def test_every_solved_branch_is_measured_not_only_the_selected_one() -> None:
    """A posture rejected for margin may still be the one the arm can hold."""

    global_ik = {
        "phases": [
            {
                "phase_id": "contact_open",
                "selected": {"joint_positions_rad": [0.1] * 7, "seed_index": 1},
                "solutions": [
                    {
                        "joint_positions_rad": [0.1] * 7,
                        "seed_index": 1,
                        "position_error_m": 0.0048,
                        "minimum_joint_limit_margin_rad": 0.0020,
                    },
                    {
                        "joint_positions_rad": [0.2] * 7,
                        "seed_index": 7,
                        "position_error_m": 0.0051,
                        "minimum_joint_limit_margin_rad": 0.0801,
                    },
                ],
            },
            {"phase_id": "approach", "solutions": [{"joint_positions_rad": [0.3] * 7}]},
        ]
    }

    postures = candidate_postures(global_ik, phase_id="contact_open")

    assert [row["seed_index"] for row in postures] == [1, 7]
    assert postures[1]["minimum_joint_limit_margin_rad"] == pytest.approx(0.0801)
    # Falls back to the selected branch when a run sealed only that one.
    only_selected = candidate_postures(
        {"phases": [{"phase_id": "contact_open", "selected": {"joint_positions_rad": [0.4] * 7}}]},
        phase_id="contact_open",
    )
    assert len(only_selected) == 1
    assert only_selected[0]["joint_positions_rad"] == [0.4] * 7


def test_a_cell_that_cannot_be_measured_does_not_poison_the_surface() -> None:
    class _NoFingertip(_SweepEnvironment):
        def read_object_sample(self):
            return {}

    report = _sweep(_NoFingertip())

    assert report["status"] == "measured"
    assert all(
        cell["measured_distance_to_target_m"] is None for cell in report["cells"]
    )
    assert report["best_cell"] is None
    # Tracking is still measurable without a fingertip readback.
    assert all(
        isinstance(cell["joint_tracking_error_rad"], float)
        and math.isfinite(cell["joint_tracking_error_rad"])
        for cell in report["cells"]
    )


def test_an_articulated_cell_is_measured_through_its_own_sampler() -> None:
    """C35 reported `unavailable` on a perfectly measurable arm.

    An articulated cell carries no rigid task object, so asking for the rigid
    sample raises instead of returning nothing -- and the whole sweep was
    discarded on a run whose fingertip was readable the entire time.
    """

    class _Articulated(_SweepEnvironment):
        def read_object_sample(self):
            raise RuntimeError("isaac_episode_rigid_task_object_missing")

        def read_task_sample(self):
            return {"grasp_frame_position_world_m": [self.joints[4], 0.0, 0.0]}

    report = _sweep(_Articulated())

    assert report["status"] == "measured"
    assert all(
        cell["measured_distance_to_target_m"] is not None for cell in report["cells"]
    )
    assert report["best_cell"] is not None


class _ModelOffsetEnvironment(_SweepEnvironment):
    """A solver whose model of the fingertip is off by a constant.

    C36's measurement: at the solved posture, across a tenfold stiffness
    range and with joint tracking at 0.007 rad, the fingertip sat a constant
    +13.0 mm off in one axis.  The solver hits its own target exactly; the
    real fingertip lands 13 mm past it, every time.
    """

    MODEL_ERROR_M = 0.013

    def read_task_sample(self):
        return {"grasp_frame_position_world_m": [self.joints[4] + self.MODEL_ERROR_M, 0.0, 0.0]}

    def read_object_sample(self):
        raise RuntimeError("isaac_episode_rigid_task_object_missing")

    def solve(self, target_position_world_m, seed_joint_positions_rad):
        del seed_joint_positions_rad
        # A perfect solver in its own model's terms.
        return [0.0] * 4 + [float(target_position_world_m[0]), 0.0, 0.0]


def _calibrate(environment, **overrides):
    from blueprint_pipeline.native_task_arena_actuator_sweep import (
        calibrate_posture_to_measured_target,
    )

    kwargs = dict(
        environment=environment,
        solve=environment.solve,
        target_position_world_m=[0.5, 0.0, 0.0],
        seed_joint_positions_rad=[0.0] * 7,
        gripper_open_command=0.0,
        max_joint_delta_rad=0.05,
        max_joint_setpoint_lead_rad=0.2,
        arrival_tolerance_m=0.005,
        settle_steps=80,
    )
    kwargs.update(overrides)
    return calibrate_posture_to_measured_target(**kwargs)


def test_calibration_finds_the_posture_whose_measured_tip_reaches_the_target() -> None:
    """The gate asks where the real fingertip is, so solve for that."""

    environment = _ModelOffsetEnvironment()

    report = _calibrate(environment)

    assert report["status"] == "measured"
    assert report["converged"] is True
    first, last = report["iterations"][0], report["iterations"][-1]
    # The uncalibrated solve reproduces the measured defect exactly...
    assert first["measured_distance_to_target_m"] == pytest.approx(0.013, abs=1e-6)
    # ...and folding the residual back into the solver's target removes it.
    assert last["measured_distance_to_target_m"] < 0.005
    assert last["solver_target_position_world_m"][0] == pytest.approx(
        0.5 - 0.013, abs=1e-6
    )
    assert report["best"]["measured_distance_to_target_m"] == min(
        row["measured_distance_to_target_m"] for row in report["iterations"]
    )


def test_calibration_is_bounded_and_keeps_its_best_when_it_cannot_converge() -> None:
    """An unreachable target stops rather than iterating forever."""

    class _Unreachable(_ModelOffsetEnvironment):
        def solve(self, target_position_world_m, seed_joint_positions_rad):
            del target_position_world_m, seed_joint_positions_rad
            return [0.0] * 7  # ignores the target entirely

    report = _calibrate(_Unreachable(), max_iterations=3)

    assert report["converged"] is False
    assert report["iteration_count"] == 3
    assert report["best"] is not None


def test_calibration_reports_a_runtime_it_cannot_drive() -> None:
    report = _calibrate(_ModelOffsetEnvironment(), solve=None)

    assert report["status"] == "unavailable"
    assert report["iterations"] == []


def test_the_sweep_reads_every_seed_the_multistart_already_sealed() -> None:
    """C36 measured one posture because the alternatives were under `attempts`.

    The solver seals each seed it tried -- solved and unsolved -- so the
    branches the selector passed over were in the receipt the whole time.
    Reading the wrong key silently narrowed a sweep to a single cell.
    """

    global_ik = {
        "phases": [
            {
                "phase_id": "contact_open",
                "selected": {"joint_positions_rad": [0.1] * 7, "seed_index": 1},
                "attempts": [
                    {
                        "solved": True,
                        "seed_index": 1,
                        "joint_positions_rad": [0.1] * 7,
                        "minimum_joint_limit_margin_rad": 0.0020,
                    },
                    {
                        "solved": True,
                        "seed_index": 7,
                        "joint_positions_rad": [0.2] * 7,
                        "minimum_joint_limit_margin_rad": 0.0801,
                    },
                    # A seed that failed carries a seed pose, not a solution.
                    {"solved": False, "seed_index": 9, "joint_positions_rad": [0.9] * 7},
                ],
            }
        ]
    }

    postures = candidate_postures(global_ik, phase_id="contact_open")

    assert [row["seed_index"] for row in postures] == [1, 7]
    assert postures[1]["minimum_joint_limit_margin_rad"] == pytest.approx(0.0801)
    # And a receipt carrying neither key still yields the selected branch.
    fallback = candidate_postures(
        {"phases": [{"phase_id": "contact_open", "selected": {"joint_positions_rad": [0.4] * 7}}]},
        phase_id="contact_open",
    )
    assert len(fallback) == 1


def test_unsolved_terminal_postures_require_explicit_physics_diagnostic_opt_in() -> None:
    """A failed IK attempt is evidence to measure, never a silent solution."""

    global_ik = {
        "phases": [
            {
                "phase_id": "contact_open",
                "selected": None,
                "attempts": [
                    {
                        "solved": False,
                        "seed_index": 4,
                        "joint_positions_rad": [0.4] * 7,
                        "position_error_m": 0.0044,
                        "orientation_error_rad": 0.087,
                    }
                ],
            }
        ]
    }

    assert candidate_postures(global_ik, phase_id="contact_open") == []
    diagnostic = candidate_postures(
        global_ik,
        phase_id="contact_open",
        include_unsolved_attempts=True,
    )

    assert len(diagnostic) == 1
    assert diagnostic[0]["offsim_solved"] is False
    assert diagnostic[0]["offsim_orientation_error_rad"] == pytest.approx(
        0.087
    )


class _WallEnvironment(_SweepEnvironment):
    """The pad midpoint can go anywhere except through a wall in X.

    C37's shape: the measured point stops dead at a surface while the target
    keeps moving past it.  Everything the arm can reach, it reaches.
    """

    WALL_X = 0.30

    def __init__(self) -> None:
        super().__init__()
        self.commanded = [0.0, 0.0, 0.0]

    def solve(self, target_position_world_m, seed_joint_positions_rad):
        del seed_joint_positions_rad
        self.commanded = [float(v) for v in target_position_world_m]
        return [0.0] * 7

    def read_task_sample(self):
        blocked = self.commanded[0] < self.WALL_X
        return {
            "grasp_frame_position_world_m": [
                max(self.commanded[0], self.WALL_X),
                self.commanded[1],
                self.commanded[2],
            ],
            "task_contact_active": blocked,
        }

    def read_object_sample(self):
        raise RuntimeError("isaac_episode_rigid_task_object_missing")


class _GhostFrameEnvironment(_WallEnvironment):
    """The measured point never follows the target at all."""

    def read_task_sample(self):
        return {"grasp_frame_position_world_m": [0.30, 0.0, 0.0], "task_contact_active": False}


def _probe(environment, **overrides):
    from blueprint_pipeline.native_task_arena_actuator_sweep import (
        probe_target_reachability,
    )

    kwargs = dict(
        environment=environment,
        solve=environment.solve,
        base_target_position_world_m=[0.29, 0.0, 0.0],
        seed_joint_positions_rad=[0.0] * 7,
        gripper_open_command=0.0,
        max_joint_delta_rad=0.05,
        max_joint_setpoint_lead_rad=0.2,
        settle_steps=3,
    )
    kwargs.update(overrides)
    return probe_target_reachability(**kwargs)


def test_the_probe_separates_an_obstruction_from_a_frame_problem() -> None:
    """An obstruction moves some axes and stalls others; a ghost moves none."""

    wall = _probe(_WallEnvironment())

    assert wall["status"] == "measured"
    following = wall["axis_following"]
    # Y and Z follow the target one-for-one...
    assert following["y"]["measured_span_m"] == pytest.approx(
        following["y"]["requested_span_m"], abs=1e-9
    )
    assert following["z"]["measured_span_m"] == pytest.approx(
        following["z"]["requested_span_m"], abs=1e-9
    )
    # ...while X is clipped at the wall, so it spans strictly less than asked.
    assert following["x"]["measured_span_m"] < following["x"]["requested_span_m"]
    # And the blocked cells are exactly the ones reporting contact.
    blocked = [c for c in wall["cells"] if c["requested_target_position_world_m"][0] < 0.30]
    assert blocked and all(c["contact_steps"] > 0 for c in blocked)
    clear = [c for c in wall["cells"] if c["requested_target_position_world_m"][0] > 0.30]
    assert clear and all(c["contact_steps"] == 0 for c in clear)

    ghost = _probe(_GhostFrameEnvironment())
    # Nothing follows: that is a frame problem, not an obstruction.
    for name in ("x", "y", "z"):
        assert ghost["axis_following"][name]["measured_span_m"] == pytest.approx(0.0)
        assert ghost["axis_following"][name]["requested_span_m"] >= 0.0


def test_the_probe_records_targets_the_solver_cannot_reach() -> None:
    class _PickySolver(_WallEnvironment):
        def solve(self, target_position_world_m, seed_joint_positions_rad):
            if target_position_world_m[0] > 0.32:
                return None
            return super().solve(target_position_world_m, seed_joint_positions_rad)

    report = _probe(_PickySolver())

    unsolved = [c for c in report["cells"] if c["status"] == "unsolved"]
    assert unsolved
    assert all("measured_grasp_frame_position_world_m" not in c for c in unsolved)
    assert report["status"] == "measured"


def test_the_probe_starts_each_cell_from_the_known_anchor_and_stops_on_force() -> None:
    class _ContactFrontier(_SweepEnvironment):
        def solve(self, target_position_world_m, seed_joint_positions_rad):
            del seed_joint_positions_rad
            return [
                float(target_position_world_m[0]),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]

        def step(self, action) -> None:
            self.joints = [float(value) for value in action[:7]]

        def read_task_sample(self):
            force = 75.0 if self.joints[0] < 0.30 else 0.0
            filtered = (
                [
                    {
                        "filter_prim_path_expr": "/Robot/right_inner_finger",
                        "force_magnitude_n": force,
                    }
                ]
                if force
                else []
            )
            return {
                "grasp_frame_position_world_m": [self.joints[0], 0.0, 0.0],
                "task_contact_active": bool(force),
                "task_robot_contact_peak_force_n": force,
                "native_readback": {
                    "contact_sensor_instance_readback": {
                        "task_robot_contact": [
                            {"nonzero_filter_forces": filtered}
                        ]
                    }
                },
            }

    environment = _ContactFrontier()
    report = _probe(
        environment,
        base_target_position_world_m=[0.29, 0.0, 0.0],
        offsets_m=[[0.11, 0.0, 0.0], [0.0, 0.0, 0.0]],
        preposition_target_position_world_m=[0.40, 0.0, 0.0],
        preposition_settle_steps=2,
        settle_steps=4,
        abort_contact_force_n=50.0,
        max_joint_delta_rad=1.0,
    )

    clear, blocked = report["cells"]
    assert report["preposition_target_position_world_m"] == [0.40, 0.0, 0.0]
    assert environment.reset_count == 3  # two cells plus the final reset
    assert clear["executed_steps"] == 4
    assert clear["aborted_on_contact_force"] is False
    assert blocked["executed_steps"] == 1
    assert blocked["aborted_on_contact_force"] is True
    assert blocked["peak_task_contact_force_n"] == pytest.approx(75.0)
    assert blocked["peak_pad_contact_forces_n"] == {
        "right_inner_finger": pytest.approx(75.0)
    }


def test_ordered_frontier_stops_after_the_first_contact_cell() -> None:
    class _OrderedFrontier(_WallEnvironment):
        pass

    report = _probe(
        _OrderedFrontier(),
        base_target_position_world_m=[0.34, 0.0, 0.0],
        offsets_m=[
            [0.02, 0.0, 0.0],
            [0.00, 0.0, 0.0],
            [-0.02, 0.0, 0.0],
            [-0.04, 0.0, 0.0],
            [-0.06, 0.0, 0.0],
            [-0.08, 0.0, 0.0],
        ],
        stop_after_first_contact_cell=True,
    )

    assert [cell["offset_m"] for cell in report["cells"]] == [
        [0.02, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.02, 0.0, 0.0],
        [-0.04, 0.0, 0.0],
        [-0.06, 0.0, 0.0],
    ]
    assert report["cells"][-1]["contact_steps"] > 0
    assert report["stop_after_first_contact_cell"] is True
    assert report["stopped_after_first_contact_cell"] is True


def test_the_sweep_measures_the_model_versus_physics_gap() -> None:
    """C42 ruled out everything else; this is what was left, unmeasured.

    All five contact branches predicted 4.5-4.9 mm off-sim and measured 12.9,
    12.9, 13.8, 15.2 and 204 mm, and the executed branch was within 0.06 mm of
    the best available.  Gains, branch, posture and obstruction are all
    excluded by measurement.  What remains is that the solver and the simulator
    disagree about where the gripper is at a given set of joints -- and that
    was being inferred by subtracting two error magnitudes rather than
    measured as a vector with a direction.
    """

    class _ModelDisagrees(_SweepEnvironment):
        OFFSET_M = 0.008

        def step(self, action) -> None:
            # This arm tracks perfectly, so the only thing separating the
            # prediction from the measurement is the disagreement itself --
            # not a tracking shortfall wearing its clothes.
            self.joints = [float(value) for value in action[:7]]

        def read_task_sample(self):
            return {
                "grasp_frame_position_world_m": [
                    self.joints[4] + self.OFFSET_M,
                    0.0,
                    0.0,
                ]
            }

        def read_object_sample(self):
            raise RuntimeError("isaac_episode_rigid_task_object_missing")

    postures = [
        {
            "posture_index": 0,
            "seed_index": 1,
            "joint_positions_rad": [0.0] * 4 + [1.0, 0.0, 0.0],
            # Where the solver believes this posture lands the grasp frame.
            "predicted_grasp_frame_position_world_m": [1.0, 0.0, 0.0],
        }
    ]

    report = _sweep(_ModelDisagrees(), postures=postures)

    assert report["status"] == "measured"
    for cell in report["cells"]:
        gap = cell["measured_minus_model_m"]
        assert gap is not None
        # The gap is a vector with a direction, not a difference of magnitudes.
        assert gap[0] > 0.0
        assert cell["measured_minus_model_distance_m"] == pytest.approx(
            abs(gap[0]), abs=1e-9
        )
        assert cell["predicted_grasp_frame_position_world_m"] == [1.0, 0.0, 0.0]


def test_a_posture_without_a_prediction_still_measures() -> None:
    """Older receipts carry no prediction; the sweep must not lose the cell."""

    postures = [
        {
            "posture_index": 0,
            "seed_index": 1,
            "joint_positions_rad": [0.0] * 4 + [1.0, 0.0, 0.0],
        }
    ]

    report = _sweep(_SweepEnvironment(), postures=postures)

    assert report["status"] == "measured"
    for cell in report["cells"]:
        assert cell["measured_minus_model_m"] is None
        assert cell["measured_distance_to_target_m"] is not None


def test_each_cell_records_the_posture_the_arm_actually_reached() -> None:
    """C43 could not settle its own decisive binary for want of this.

    The solver moved its predicted fingertip 1.90 mm across four postures
    and physics moved 0.24 mm -- a slope of -0.88 that eats every correction
    the calibration makes.  Either the arm is not differentiating the
    commands, or it is and the two frames disagree about where it ended up.
    The worst single joint cannot tell those apart; the whole vector can.
    """

    environment = _SweepEnvironment()

    report = _sweep(environment)

    for cell in report["cells"]:
        commanded = cell["commanded_joint_positions_rad"]
        reached = cell["measured_joint_positions_rad"]
        residual = cell["joint_tracking_residual_rad"]
        assert len(commanded) == 7
        assert reached is not None and len(reached) == 7
        # The residual is the vector, and its worst element is the scalar the
        # sweep already reported -- so the two can never disagree.
        assert residual == pytest.approx(
            [a - b for a, b in zip(commanded, reached)]
        )
        assert max(abs(v) for v in residual) == pytest.approx(
            cell["joint_tracking_error_rad"]
        )
