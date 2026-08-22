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
