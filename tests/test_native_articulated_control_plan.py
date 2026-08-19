from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_articulated_control_plan import (
    MAX_JOINT_DELTA_RAD,
    MAX_JOINT_SETPOINT_LEAD_RAD,
    NativeArticulatedControlPlanError,
    materialize_native_articulated_control_plan,
)


def _scene() -> dict:
    motion = {
        "hinge_point_world_m": [0.0, 0.0, 1.0],
        "hinge_axis_world_unit": [0.0, 0.0, 1.0],
        "handle_grasp_point_closed_world_m": [0.5, 0.0, 1.0],
        "authored_limits_degrees": [0.0, 90.0],
        "scripted_sweep_angle_degrees": 50.0,
    }
    result = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "task_kind": "articulated_open_close",
        "scenario": {"cell_id": "articulated-canonical", "seed": 17},
        "articulation": {"motion_geometry": motion},
        "task_spec": {
            "schema_version": "adp_task_spec.v1",
            "task_kind": "articulated_open_close",
            "settle_window_samples": 40,
            "maximum_action_steps": 450,
        },
        "plan_digest": "",
    }
    result["plan_digest"] = canonical_digest(result, digest_field="plan_digest")
    return result


def _construction(scene: dict) -> dict:
    clearance = {
        "scene_plan_digest": scene["plan_digest"],
        "phases": [{"phase_id": "approach"}],
        "plan_digest": "",
    }
    clearance["plan_digest"] = canonical_digest(
        clearance, digest_field="plan_digest"
    )
    result = {
        "schema_version": "native_task_arena_construction_result.v1",
        "status": "completed",
        "construction_gate_qualified": True,
        "blockers": [],
        "scene_plan_digest": scene["plan_digest"],
        "phase_results": [{"phase_id": "approach", "target_reached": True}],
        "camera_gates": {
            role: {"passed": True} for role in ("external", "wrist", "overview")
        },
        "reset_replay": {"passed": True},
        "construction_phase_plan": clearance,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def test_qualified_native_construction_freezes_bounded_contact_arc() -> None:
    scene = _scene()
    result = materialize_native_articulated_control_plan(
        scene_plan=scene, construction_result=_construction(scene)
    )

    phases = result["scripted_positive_actions"]
    assert [row["phase_id"] for row in phases[:3]] == [
        "approach",
        "grasp_open",
        "grasp_close",
    ]
    assert [row["phase_id"] for row in phases[-2:]] == ["release", "retreat"]
    assert len([row for row in phases if row["phase_id"].startswith("sweep_")]) == 7
    assert all(row["mode"] == "ik_pose" for row in phases)
    assert result["maximum_scripted_and_settle_steps"] == 430
    assert result["maximum_scripted_and_settle_steps"] <= 450
    assert result["candidate_policy_queried"] is False
    assert result["plan_digest"] == canonical_digest(
        result, digest_field="plan_digest"
    )


@pytest.mark.parametrize(
    "mutation,expected",
    (
        (lambda value: value["camera_gates"]["wrist"].update(passed=False), "camera_preflight_incomplete"),
        (lambda value: value["reset_replay"].update(passed=False), "reset_preflight_incomplete"),
        (lambda value: value["phase_results"][0].update(target_reached=False), "ik_preflight_incomplete"),
    ),
)
def test_unqualified_native_evidence_never_materializes_a_control_plan(
    mutation, expected
) -> None:
    scene = _scene()
    construction = _construction(scene)
    mutation(construction)
    construction["result_digest"] = canonical_digest(
        construction, digest_field="result_digest"
    )

    with pytest.raises(NativeArticulatedControlPlanError, match=expected):
        materialize_native_articulated_control_plan(
            scene_plan=copy.deepcopy(scene), construction_result=construction
        )


def test_scene_or_construction_digest_drift_fails_closed() -> None:
    scene = _scene()
    construction = _construction(scene)
    scene["scenario"]["seed"] = 18

    with pytest.raises(NativeArticulatedControlPlanError, match="scene_plan_invalid"):
        materialize_native_articulated_control_plan(
            scene_plan=scene, construction_result=construction
        )


def test_setpoint_lead_does_not_throttle_the_arm_below_its_own_limits() -> None:
    """The lead cap, not the slew cap, decides how fast the arm can move.

    The implicit actuator is a PD: its velocity comes from the position error
    it is shown, so capping how far the commanded setpoint may lead the
    measured position caps the achievable speed. Measured in r17 with a
    0.20 rad lead cap:

        total_action_steps        400   (the full budget, exhausted)
        joint travel achieved  10.717 rad
        travel needed, phase 1  8.721 rad   (of nine phases)
        achieved rate          0.0038 rad/joint/step

    That rate is an eighth of what the 0.03 rad slew already allowed, which is
    how we know the slew was never the binding constraint. Every phase
    reported native_task_phase_ik_unreached even though the IK solution was
    correct on the first step -- the arm could not be driven there in time.

    A Panda's joint velocity limit is ~2.6 rad/s. At 20 Hz control the slew
    must stay under 0.13 rad/step to respect that, and the lead must not sit
    so far below it that the actuator is starved.
    """

    control_hz = 20.0
    panda_joint_velocity_limit_rad_s = 2.6

    assert MAX_JOINT_DELTA_RAD * control_hz <= panda_joint_velocity_limit_rad_s, (
        "the per-step slew must not command faster than the joint can move"
    )
    assert MAX_JOINT_DELTA_RAD * control_hz >= 1.0, (
        "commanding under 1 rad/s throttles a Panda to a fraction of its speed"
    )
    # The lead must exceed the slew, or the actuator is shown a smaller error
    # than the setpoint is allowed to advance and the slew never binds.
    assert MAX_JOINT_SETPOINT_LEAD_RAD > MAX_JOINT_DELTA_RAD, (
        "a lead below the slew starves the PD and stalls the arm"
    )
    # Phase 1 alone needed 8.721 rad of travel inside a 400 step budget.
    reachable_rad = MAX_JOINT_DELTA_RAD * 400
    assert reachable_rad >= 8.721 * 2, (
        "the budget must cover more than the first of nine phases"
    )
