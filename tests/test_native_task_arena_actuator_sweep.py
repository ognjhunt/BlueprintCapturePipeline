from __future__ import annotations

import math

import pytest

from blueprint_pipeline.native_task_arena_actuator_sweep import (
    DEFAULT_WRIST_GAIN_CANDIDATES,
    SWEEP_SCHEMA_VERSION,
    candidate_postures,
    run_actuator_posture_sweep,
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
