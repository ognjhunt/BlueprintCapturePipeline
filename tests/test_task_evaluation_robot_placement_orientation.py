"""Contract tests for CPU-decidable gripper-orientation feasibility.

These pin the CPU-decidable part of a defect observed on scene 839873: candidate
base/reset pairs reached native execution before their authored orientation slew
was screened.  They do not assign every native rejection the same root cause.
"""

from __future__ import annotations

import math

import pytest

from blueprint_pipeline.task_evaluation_robot_placement_orientation import (
    RobotPlacementOrientationError,
    SCHEMA_VERSION,
    evaluate_orientation_slew_feasibility,
    evaluate_task_aware_reset_orientation_feasibility,
    quaternion_angle_rad,
    required_orientation_slew_rad,
    solve_base_yaw_for_orientation,
    world_rest_grasp_orientation_xyzw,
)


IDENTITY = (0.0, 0.0, 0.0, 1.0)
# Franka + Robotiq rest grasp frame, expressed in the robot base frame: the
# gripper points down at the reset joint pose.  Measured from the native
# readback on scene 839873 (candidate9 / r33).
FRANKA_REST_GRASP_BASE = (0.0, -1.0, 0.0, 0.0)
# Authored planar-push tool orientation: gripper horizontal, 90 deg about +Y.
PUSH_TARGET = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
YAW_180 = (0.0, 0.0, 1.0, 0.0)


def _phase(phase_id: str, orientation) -> dict:
    return {
        "phase_id": phase_id,
        "position_world_m": [1.0, 0.0, 0.8],
        "orientation_world_xyzw": list(orientation),
    }


def test_quaternion_angle_is_symmetric_and_sign_invariant() -> None:
    a = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
    b = IDENTITY
    assert quaternion_angle_rad(a, b) == pytest.approx(math.pi / 2, abs=1e-9)
    assert quaternion_angle_rad(b, a) == pytest.approx(math.pi / 2, abs=1e-9)
    negated = tuple(-value for value in a)
    # q and -q are the same rotation; the metric must not report pi difference.
    assert quaternion_angle_rad(negated, b) == pytest.approx(math.pi / 2, abs=1e-9)


def test_world_rest_grasp_composes_base_with_rest_frame() -> None:
    """Reproduces the exact native readback: base(180 deg yaw) * rest == 180 deg about X."""

    world = world_rest_grasp_orientation_xyzw(
        base_orientation_xyzw=YAW_180,
        rest_grasp_orientation_base_xyzw=FRANKA_REST_GRASP_BASE,
    )
    assert world[0] == pytest.approx(1.0, abs=1e-9)
    assert world[1] == pytest.approx(0.0, abs=1e-9)
    assert world[2] == pytest.approx(0.0, abs=1e-9)
    assert world[3] == pytest.approx(0.0, abs=1e-9)


def test_required_slew_reproduces_the_paid_839873_measurement() -> None:
    """The 180 deg yaw the harness chose demands a full pi wrist rotation."""

    slew = required_orientation_slew_rad(
        base_orientation_xyzw=YAW_180,
        rest_grasp_orientation_base_xyzw=FRANKA_REST_GRASP_BASE,
        target_orientation_xyzw=PUSH_TARGET,
    )
    assert slew == pytest.approx(math.pi, abs=1e-6)


def test_base_yaw_materially_changes_required_slew() -> None:
    """Yaw is a real lever: the same task needs half the rotation at yaw 0."""

    at_zero = required_orientation_slew_rad(
        base_orientation_xyzw=IDENTITY,
        rest_grasp_orientation_base_xyzw=FRANKA_REST_GRASP_BASE,
        target_orientation_xyzw=PUSH_TARGET,
    )
    assert at_zero == pytest.approx(math.pi / 2, abs=1e-6)
    assert at_zero < math.pi - 1e-6


def test_infeasible_slew_fails_closed_with_named_blocker() -> None:
    """The 839873 configuration must be rejected on CPU, before any allocation."""

    report = evaluate_orientation_slew_feasibility(
        base_orientation_xyzw=YAW_180,
        rest_grasp_orientation_base_xyzw=FRANKA_REST_GRASP_BASE,
        phases=[_phase("precontact", PUSH_TARGET), _phase("push_contact", PUSH_TARGET)],
        maximum_steps_per_phase=64,
        orientation_slew_rad_per_step=0.0267,
    )
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["feasible"] is False
    assert (
        "robot_placement_orientation_slew_exceeds_phase_budget:precontact"
        in report["blockers"]
    )
    worst = report["worst_required_slew_rad"]
    assert worst == pytest.approx(math.pi, abs=1e-6)
    # 118 steps required against a 64 step budget.
    assert report["phases"][0]["required_steps"] == 118
    assert report["phases"][0]["step_budget"] == 64


def test_feasible_slew_passes_and_reports_margin() -> None:
    report = evaluate_orientation_slew_feasibility(
        base_orientation_xyzw=IDENTITY,
        rest_grasp_orientation_base_xyzw=FRANKA_REST_GRASP_BASE,
        phases=[_phase("precontact", PUSH_TARGET)],
        maximum_steps_per_phase=64,
        orientation_slew_rad_per_step=0.0267,
    )
    assert report["feasible"] is True
    assert report["blockers"] == []
    assert report["phases"][0]["required_steps"] == 59
    assert 0.0 < report["phases"][0]["budget_utilization"] < 1.0


def test_solver_finds_a_feasible_yaw_and_reports_the_worst_phase() -> None:
    """The adaptive lever: solve yaw for orientation feasibility, for any task."""

    solved = solve_base_yaw_for_orientation(
        rest_grasp_orientation_base_xyzw=FRANKA_REST_GRASP_BASE,
        phases=[_phase("precontact", PUSH_TARGET)],
        maximum_steps_per_phase=64,
        orientation_slew_rad_per_step=0.0267,
    )
    assert solved["feasible"] is True
    assert solved["best_worst_slew_rad"] == pytest.approx(math.pi / 2, abs=1e-3)
    # The yaw the harness actually used must be reported as strictly worse.
    assert solved["best_worst_slew_rad"] < math.pi - 1e-3


def test_solver_reports_infeasible_when_no_yaw_can_satisfy_the_budget() -> None:
    solved = solve_base_yaw_for_orientation(
        rest_grasp_orientation_base_xyzw=FRANKA_REST_GRASP_BASE,
        phases=[_phase("precontact", PUSH_TARGET)],
        maximum_steps_per_phase=4,
        orientation_slew_rad_per_step=0.0267,
    )
    assert solved["feasible"] is False
    assert solved["blockers"]


@pytest.mark.parametrize(
    "quaternion",
    [(0.0, 0.0, 0.0, 0.0), (1.0, float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0)],
)
def test_degenerate_quaternions_fail_closed(quaternion) -> None:
    with pytest.raises(RobotPlacementOrientationError):
        required_orientation_slew_rad(
            base_orientation_xyzw=IDENTITY,
            rest_grasp_orientation_base_xyzw=FRANKA_REST_GRASP_BASE,
            target_orientation_xyzw=quaternion,
        )


def test_missing_phase_orientation_fails_closed() -> None:
    with pytest.raises(RobotPlacementOrientationError):
        evaluate_orientation_slew_feasibility(
            base_orientation_xyzw=IDENTITY,
            rest_grasp_orientation_base_xyzw=FRANKA_REST_GRASP_BASE,
            phases=[{"phase_id": "precontact", "position_world_m": [0, 0, 0]}],
            maximum_steps_per_phase=64,
            orientation_slew_rad_per_step=0.0267,
        )


def test_non_positive_budget_or_rate_fails_closed() -> None:
    for budget, rate in ((0, 0.0267), (64, 0.0), (64, -1.0), (-1, 0.0267)):
        with pytest.raises(RobotPlacementOrientationError):
            evaluate_orientation_slew_feasibility(
                base_orientation_xyzw=IDENTITY,
                rest_grasp_orientation_base_xyzw=FRANKA_REST_GRASP_BASE,
                phases=[_phase("precontact", PUSH_TARGET)],
                maximum_steps_per_phase=budget,
                orientation_slew_rad_per_step=rate,
            )


def test_scene_839873_regression_is_refused_using_the_shipped_franka_profile() -> None:
    """End-to-end regression on the real numbers, via profile data only.

    Eleven paid Arena allocations accepted this configuration analytically and
    were each rejected natively.  With the profile's measured rest grasp frame
    and slew rate, it must now be refused locally, at no cost.
    """

    from blueprint_pipeline.scene_placement.robot_profile import get_robot_profile

    profile = get_robot_profile("franka_panda")
    phases = [
        _phase(name, PUSH_TARGET)
        for name in ("precontact", "push_contact", "push_01", "push_02")
    ]

    refused = evaluate_orientation_slew_feasibility(
        base_orientation_xyzw=YAW_180,  # the yaw every failing run used
        rest_grasp_orientation_base_xyzw=profile.rest_grasp_orientation_base_xyzw,
        phases=phases,
        maximum_steps_per_phase=64,
        orientation_slew_rad_per_step=profile.orientation_slew_rad_per_step,
    )
    assert refused["feasible"] is False
    assert refused["worst_required_slew_rad"] == pytest.approx(math.pi, abs=1e-6)

    # And the solver must surface that a better yaw existed all along.
    solved = solve_base_yaw_for_orientation(
        rest_grasp_orientation_base_xyzw=profile.rest_grasp_orientation_base_xyzw,
        phases=phases,
        maximum_steps_per_phase=64,
        orientation_slew_rad_per_step=profile.orientation_slew_rad_per_step,
    )
    assert solved["feasible"] is True
    assert solved["best_worst_slew_rad"] < refused["worst_required_slew_rad"]


def test_franka_reach_ceiling_drops_the_affordance_slack() -> None:
    """A fixed-base arm must arrive AT the tool pose, so slack overstates reach."""

    from blueprint_pipeline.scene_placement.robot_profile import get_robot_profile

    franka = get_robot_profile("franka_panda")
    assert franka.effector_reaches_tool_pose is True
    assert franka.max_shoulder_to_affordance_m() == pytest.approx(0.855, abs=1e-9)
    # The pose that was executed three times sat at 93.6% of usable span; the
    # abandoned one at 57.9%.  The gate must now separate them.
    assert franka.reach_utilization(0.800) > franka.reach_utilization(0.495)
    assert franka.reach_utilization(0.800) == pytest.approx(0.9357, abs=1e-3)


def test_approach_embodiments_keep_their_affordance_slack() -> None:
    """Humanoids approach an affordance; this change must not narrow their reach."""

    from blueprint_pipeline.scene_placement.robot_profile import get_robot_profile

    g1 = get_robot_profile("unitree_g1")
    assert g1.effector_reaches_tool_pose is False
    assert g1.max_shoulder_to_affordance_m() == pytest.approx(
        g1.arm_span_m + g1.max_effector_to_affordance_m, abs=1e-9
    )


def _gate(status: str = "passed") -> dict:
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    gate = {
        "schema_version": "task_evaluation_robot_placement_geometry_gate.v1",
        "status": status,
        "blockers": [],
        "geometry_gate_digest": "",
    }
    gate["geometry_gate_digest"] = canonical_digest(
        gate, digest_field="geometry_gate_digest"
    )
    return gate


def _trajectory(steps: int = 64) -> dict:
    return {
        "phases": [_phase("precontact", PUSH_TARGET)],
        "maximum_steps_per_phase": steps,
    }


def test_agent_gate_derives_a_task_aware_reset_before_any_execution() -> None:
    """The static home refusal is repaired before it can waste provider spend."""

    from blueprint_pipeline.task_evaluation_robot_placement_agent import (
        _reject_infeasible_orientation_slew,
    )

    admitted = _reject_infeasible_orientation_slew(
        gate=_gate("passed"),
        proposal={"pose": {"orientation_xyzw": list(YAW_180)}},
        trajectory=_trajectory(),
        robot_id="franka_panda",
        maximum_steps_per_phase=64,
    )
    assert admitted["status"] == "passed"
    report = admitted["orientation_slew_feasibility"]
    assert report["feasible"] is True
    assert report["task_aware_reset"]["residual_slew_rad"] < 0.08
    assert report["phases"][0]["reference"] == "derived_reset"
    assert report["native_full_pose_ik_required"] is True


def test_agent_gate_admits_a_slewable_candidate_and_attaches_the_report() -> None:
    from blueprint_pipeline.task_evaluation_robot_placement_agent import (
        _reject_infeasible_orientation_slew,
        _validated_gate,
    )

    admitted = _reject_infeasible_orientation_slew(
        gate=_gate("passed"),
        proposal={"pose": {"orientation_xyzw": list(IDENTITY)}},
        trajectory=_trajectory(),
        robot_id="franka_panda",
        maximum_steps_per_phase=64,
    )
    assert admitted["status"] == "passed"
    assert admitted["orientation_slew_feasibility"]["feasible"] is True
    assert _validated_gate(admitted) == admitted


def test_task_aware_gate_carries_prior_phase_orientation_sequentially() -> None:
    from blueprint_pipeline.franka_kinematics import (
        FRANKA_JOINT_LIMITS_RAD,
        forward_kinematics,
    )

    report = evaluate_task_aware_reset_orientation_feasibility(
        base_orientation_xyzw=YAW_180,
        phases=[
            _phase("precontact", PUSH_TARGET),
            _phase(
                "turnaround",
                (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5)),
            ),
        ],
        maximum_steps_per_phase=64,
        orientation_slew_rad_per_step=0.0267,
        nominal_joint_positions=FRANKA_RESET_JOINTS,
        joint_limits_rad=FRANKA_JOINT_LIMITS_RAD,
        forward_kinematics=forward_kinematics,
        flange_to_grasp_orientation_xyzw=(0.0, 0.0, 1.0, 0.0),
    )

    assert report["phases"][0]["reference"] == "derived_reset"
    assert report["phases"][1]["reference"] == "prior_phase"
    assert report["feasible"] is False
    assert report["blockers"] == [
        "robot_placement_orientation_slew_exceeds_phase_budget:turnaround"
    ]


def test_agent_gate_is_inert_without_a_trajectory_or_known_profile() -> None:
    """Never invent an analytic verdict the inputs cannot support."""

    from blueprint_pipeline.task_evaluation_robot_placement_agent import (
        _reject_infeasible_orientation_slew,
    )

    for kwargs in (
        {"trajectory": None, "robot_id": "franka_panda"},
        {"trajectory": _trajectory(), "robot_id": ""},
        {"trajectory": _trajectory(), "robot_id": "no_such_robot_profile"},
    ):
        result = _reject_infeasible_orientation_slew(
            gate=_gate("passed"),
            proposal={"pose": {"orientation_xyzw": list(YAW_180)}},
            maximum_steps_per_phase=64,
            **kwargs,
        )
        assert result["status"] == "passed"
        assert "orientation_slew_feasibility" not in result


def test_guidance_recommends_a_feasible_yaw_for_the_authored_plan() -> None:
    """Adaptivity: the proposer is told which yaws this task admits."""

    from blueprint_pipeline.task_evaluation_robot_placement_agent import (
        _orientation_slew_guidance,
    )

    guidance = _orientation_slew_guidance(
        trajectory=_trajectory(), robot_id="franka_panda"
    )
    assert guidance is not None
    assert guidance["any_yaw_is_feasible"] is True
    assert guidance["advisory_only_deterministic_gate_decides"] is True
    assert 0.0 < guidance["admissible_yaw_fraction"] < 1.0
    assert (
        guidance["recommended_worst_phase_required_steps"]
        <= guidance["step_budget"]
    )


def test_guidance_is_absent_when_it_cannot_be_computed() -> None:
    from blueprint_pipeline.task_evaluation_robot_placement_agent import (
        _orientation_slew_guidance,
    )

    assert _orientation_slew_guidance(trajectory=_trajectory(), robot_id="") is None
    assert (
        _orientation_slew_guidance(
            trajectory={"phases": [], "maximum_steps_per_phase": 64},
            robot_id="franka_panda",
        )
        is None
    )
    assert (
        _orientation_slew_guidance(
            trajectory={"phases": [_phase("precontact", PUSH_TARGET)]},
            robot_id="franka_panda",
        )
        is None
    )


FRANKA_RESET_JOINTS = [
    0.0,
    -0.6283185307179586,
    0.0,
    -2.5132741228718345,
    0.0,
    1.8849555921538759,
    0.0,
]


def test_forward_kinematics_reproduces_the_native_grasp_frame_readback() -> None:
    """The flange->grasp offset is calibrated, not guessed.

    Published-DH forward kinematics at the shipped reset pose, composed with the
    profile's flange-to-grasp rotation, must equal the grasp orientation the GPU
    actually reported.  If this drifts, every reset-pose decision below is void.
    """

    from blueprint_pipeline.franka_kinematics import forward_kinematics
    from blueprint_pipeline.scene_placement.robot_profile import get_robot_profile
    from blueprint_pipeline.task_evaluation_robot_placement_orientation import (
        grasp_orientation_base_xyzw,
    )

    profile = get_robot_profile("franka_panda")
    computed = grasp_orientation_base_xyzw(
        joint_positions=FRANKA_RESET_JOINTS,
        forward_kinematics=forward_kinematics,
        flange_to_grasp_orientation_xyzw=(
            profile.flange_to_grasp_orientation_xyzw
        ),
    )
    assert quaternion_angle_rad(
        computed, profile.rest_grasp_orientation_base_xyzw
    ) == pytest.approx(0.0, abs=1e-6)


def test_task_aware_reset_turns_the_infeasible_839873_slew_into_a_free_one() -> None:
    """The real fix: derive the reset pose from the task, not from a constant."""

    from blueprint_pipeline.franka_kinematics import (
        FRANKA_JOINT_LIMITS_RAD,
        forward_kinematics,
    )
    from blueprint_pipeline.scene_placement.robot_profile import get_robot_profile
    from blueprint_pipeline.task_evaluation_robot_placement_orientation import (
        quaternion_multiply_xyzw,
        solve_task_aware_reset_joints,
    )

    profile = get_robot_profile("franka_panda")
    base_inverse = [-YAW_180[0], -YAW_180[1], -YAW_180[2], YAW_180[3]]
    target_base = quaternion_multiply_xyzw(base_inverse, PUSH_TARGET)

    solved = solve_task_aware_reset_joints(
        target_orientation_base_xyzw=target_base,
        nominal_joint_positions=FRANKA_RESET_JOINTS,
        joint_limits_rad=FRANKA_JOINT_LIMITS_RAD,
        forward_kinematics=forward_kinematics,
        flange_to_grasp_orientation_xyzw=(
            profile.flange_to_grasp_orientation_xyzw
        ),
        coarse_samples=13,
        refine_rounds=6,
    )
    # The shipped constant is a full pi away from what the task asks for.
    assert solved["nominal_slew_rad"] == pytest.approx(math.pi, abs=1e-6)
    # A task-derived reset is within a couple of degrees.
    assert solved["residual_slew_rad"] < math.radians(2.0)
    assert solved["improvement_rad"] > math.radians(170.0)

    # And that turns an infeasible phase into a comfortably feasible one.
    infeasible = math.ceil(
        solved["nominal_slew_rad"] / profile.orientation_slew_rad_per_step
    )
    feasible = math.ceil(
        solved["residual_slew_rad"] / profile.orientation_slew_rad_per_step
    )
    assert infeasible > 64
    assert feasible <= 8

    # The shoulder posture is preserved: only elbow and wrist move.
    for index in (0, 1, 2):
        assert solved["joint_positions_rad"][index] == pytest.approx(
            FRANKA_RESET_JOINTS[index], abs=1e-9
        )


def test_reset_solver_respects_joint_limits() -> None:
    from blueprint_pipeline.franka_kinematics import (
        FRANKA_JOINT_LIMITS_RAD,
        forward_kinematics,
    )
    from blueprint_pipeline.task_evaluation_robot_placement_orientation import (
        solve_task_aware_reset_joints,
    )

    solved = solve_task_aware_reset_joints(
        target_orientation_base_xyzw=PUSH_TARGET,
        nominal_joint_positions=FRANKA_RESET_JOINTS,
        joint_limits_rad=FRANKA_JOINT_LIMITS_RAD,
        forward_kinematics=forward_kinematics,
        flange_to_grasp_orientation_xyzw=(0.0, 0.0, 1.0, 0.0),
    )
    for value, (low, high) in zip(
        solved["joint_positions_rad"], FRANKA_JOINT_LIMITS_RAD
    ):
        assert low - 1e-9 <= value <= high + 1e-9


def test_reset_solver_fails_closed_on_malformed_search_inputs() -> None:
    from blueprint_pipeline.franka_kinematics import forward_kinematics
    from blueprint_pipeline.task_evaluation_robot_placement_orientation import (
        solve_task_aware_reset_joints,
    )

    for kwargs in (
        {"nominal_joint_positions": [], "joint_limits_rad": []},
        {
            "nominal_joint_positions": [0.0, 0.0],
            "joint_limits_rad": [(-1.0, 1.0)],
        },
        {
            "nominal_joint_positions": [0.0],
            "joint_limits_rad": [(1.0, -1.0)],
        },
    ):
        with pytest.raises(RobotPlacementOrientationError):
            solve_task_aware_reset_joints(
                target_orientation_base_xyzw=PUSH_TARGET,
                forward_kinematics=forward_kinematics,
                flange_to_grasp_orientation_xyzw=(0.0, 0.0, 1.0, 0.0),
                searchable_joint_indices=(0,),
                **kwargs,
            )
