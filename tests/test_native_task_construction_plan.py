from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_construction_plan import (
    MAX_JOINT_DELTA_RAD,
    MAX_JOINT_SETPOINT_LEAD_RAD,
    NativeTaskConstructionPlanError,
    evaluate_rigid_construction_gates,
    materialize_native_task_construction_phase_plan,
)


def _articulated_840796_fixture() -> dict:
    motion = {
        "schema_version": "native_articulated_motion_geometry.v1",
        "target_joint_id": "refrigerator_upper_door_hinge",
        "hinge_point_world_m": [1.974, 1.479, 1.02],
        "hinge_axis_world_unit": [0.0, 0.0, 1.0],
        "handle_grasp_point_closed_world_m": [2.094, 1.807, 1.02],
        "authored_limits_degrees": [0.0, 90.0],
        "scripted_sweep_angle_degrees": 50.0,
        "motion_geometry_digest": "",
    }
    motion["motion_geometry_digest"] = canonical_digest(
        motion, digest_field="motion_geometry_digest"
    )
    return {
        "scene_id": "840796",
        "task_kind": "articulated_open_close",
        "plan_digest": "sha256:" + "a" * 64,
        "task_spec": {
            "schema_version": "adp_task_spec.v1",
            "task_kind": "articulated_open_close",
        },
        "articulation": {"motion_geometry": motion},
    }


def _rigid_fixture(*, asset_id: str, scene_id: str = "840313") -> dict:
    affordance = {
        "schema_version": "native_rigid_interaction_affordance.v1",
        "subject_asset_id": asset_id,
        "scoring_frame_id": "task_scoring_frame",
        "asset_root_from_scoring_frame": {
            "position_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "contact_point_scoring_frame_m": [0.0, 0.0, 0.06],
        "approach_unit_scoring_frame": [0.0, -1.0, 0.0],
        "lift_unit_world": [0.0, 0.0, 1.0],
        "gripper_orientation_scoring_frame_xyzw": [
            0.0,
            -0.7071067811865475,
            0.0,
            0.7071067811865476,
        ],
        "pregrasp_clearance_m": 0.12,
        "arrival_orientation_tolerance_rad": 0.05,
        "allowed_contact_prim_paths": ["/Asset/body"],
        "intended_support_prim_paths": ["/Scene/support"],
        "affordance_digest": "",
    }
    affordance["affordance_digest"] = canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    return {
        "scene_id": scene_id,
        "task_kind": "rigid_pick_place",
        "plan_digest": "sha256:" + "b" * 64,
        "cadence": {"maximum_action_steps": 240},
        "task_spec": {
            "schema_version": "adp_task_spec.v2",
            "task_kind": "rigid_pick_place",
            "subject_asset_id": asset_id,
            "start_pose_world": [
                3.4681748,
                -3.3100837,
                0.526465,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "destination_position_bounds_world_m": {
                "minimum": [3.70, -3.45, 0.51],
                "maximum": [3.80, -3.35, 0.54],
            },
            "support_height_interval_m": [0.50, 0.55],
            "destination_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "destination_orientation_tolerance_rad": 0.05,
            "minimum_lift_m": 0.08,
            "minimum_translation_m": 0.15,
            "settle_window_samples": 15,
            "task_contact_minimum_force_n": 0.5,
            "collision_failure_minimum_force_n": 1.0,
            "reset_translation_tolerance_m": 0.001,
            "reset_orientation_tolerance_rad": 0.01,
            "settle_position_tolerance_m": 0.002,
            "settle_orientation_tolerance_rad": 0.02,
            "relocation_tracking_tolerance_m": 0.02,
            "workspace_position_bounds_world_m": {
                "minimum": [3.0, -4.0, 0.4],
                "maximum": [4.2, -3.0, 0.8],
            },
            "interaction_affordance": affordance,
        },
        "objects": [
            {
                "semantic_role": "task_object",
                "task_subject": True,
                "asset_id": asset_id,
                "pose_world": {
                    "position_world_m": [3.4681748, -3.3100837, 0.526465],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "reset_state": {
                    "root_pose_world": {
                        "position_world_m": [3.4681748, -3.3100837, 0.526465],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "joint_positions": {},
                },
            }
        ],
    }


def _planar_push_fixture() -> dict:
    scene = _rigid_fixture(asset_id="admitted_can", scene_id="841007")
    spec = scene["task_spec"]
    spec["manipulation_strategy"] = "planar_push"
    spec["minimum_lift_m"] = 0.0
    spec["start_pose_world"] = [-1.5, -7.15, 0.801008339, 0.0, 0.0, 0.0, 1.0]
    spec["destination_position_bounds_world_m"] = {
        "minimum": [-1.72, -7.20, 0.791008339],
        "maximum": [-1.62, -7.10, 0.811008339],
    }
    spec["support_height_interval_m"] = [0.791008339, 0.811008339]
    spec["workspace_position_bounds_world_m"] = {
        "minimum": [-1.91, -7.60, 0.78],
        "maximum": [-1.37, -6.30, 1.00],
    }
    spec["interaction_affordance"]["contact_point_scoring_frame_m"] = [
        0.031094726,
        0.0,
        0.084713997,
    ]
    spec["interaction_affordance"]["approach_unit_scoring_frame"] = [1.0, 0.0, 0.0]
    spec["interaction_affordance"]["gripper_orientation_scoring_frame_xyzw"] = [
        0.0,
        -0.7071067811865475,
        0.0,
        0.7071067811865476,
    ]
    affordance = spec["interaction_affordance"]
    affordance["affordance_digest"] = canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    root_pose = scene["objects"][0]["pose_world"]
    reset_pose = scene["objects"][0]["reset_state"]["root_pose_world"]
    root_pose["position_world_m"] = list(spec["start_pose_world"][:3])
    reset_pose["position_world_m"] = list(spec["start_pose_world"][:3])
    return scene


def test_840796_articulated_fixture_preserves_compatibility_plan() -> None:
    scene = _articulated_840796_fixture()

    dispatched = materialize_native_task_construction_phase_plan(scene)

    assert dispatched["schema_version"] == (
        "native_articulated_construction_phase_plan.v1"
    )
    assert dispatched["target_joint_id"] == "refrigerator_upper_door_hinge"
    assert dispatched["phase_count"] == 12
    assert dispatched["plan_digest"] == canonical_digest(
        dispatched, digest_field="plan_digest"
    )


@pytest.mark.parametrize(
    "scene,label",
    [
        (_rigid_fixture(asset_id="canned_beverage_replacement"), "rigid"),
        (_articulated_840796_fixture(), "legacy_articulated"),
    ],
)
def test_every_construction_phase_plan_publishes_the_command_limits(
    scene: dict, label: str
) -> None:
    """The worker executes ``execution_parameters``; a plan that omits these two
    bounds silently hands the run to the servo's own defaults."""

    execution = materialize_native_task_construction_phase_plan(scene)[
        "execution_parameters"
    ]

    assert execution["max_joint_delta_rad"] == pytest.approx(MAX_JOINT_DELTA_RAD)
    assert execution["max_joint_setpoint_lead_rad"] == pytest.approx(
        MAX_JOINT_SETPOINT_LEAD_RAD
    )


def test_rigid_phase_plan_reserves_the_controls_settle_window() -> None:
    """The qualifying controls episode replays every qualified phase and then
    appends the settle window inside the same ``maximum_action_steps`` cap, so
    a construction budget without the reserve can qualify a paid construction
    that ``native_rigid_control_action_budget_exceeded`` then refuses."""

    scene = _rigid_fixture(asset_id="canned_beverage_replacement")
    execution = materialize_native_task_construction_phase_plan(scene)[
        "execution_parameters"
    ]
    assert execution["maximum_construction_total_steps"] == (
        scene["cadence"]["maximum_action_steps"]
        - scene["task_spec"]["settle_window_samples"]
    )


def test_rigid_phase_plan_refuses_an_unqualifiable_budget() -> None:
    scene = _rigid_fixture(asset_id="canned_beverage_replacement")
    scene["cadence"]["maximum_action_steps"] = (
        scene["task_spec"]["settle_window_samples"] + 1
    )
    with pytest.raises(
        NativeTaskConstructionPlanError,
        match="native_rigid_construction_total_budget_infeasible",
    ):
        materialize_native_task_construction_phase_plan(scene)


def test_construction_dispatch_forwards_an_overridden_command_limit_pair() -> None:
    execution = materialize_native_task_construction_phase_plan(
        _rigid_fixture(asset_id="canned_beverage_replacement"),
        max_joint_delta_rad=0.10,
        max_joint_setpoint_lead_rad=1.00,
    )["execution_parameters"]

    assert execution["max_joint_delta_rad"] == pytest.approx(0.10)
    assert execution["max_joint_setpoint_lead_rad"] == pytest.approx(1.00)


def test_840313_rigid_fixture_has_complete_construction_gate_sequence() -> None:
    plan = materialize_native_task_construction_phase_plan(
        _rigid_fixture(asset_id="canned_beverage_replacement"),
        rigid_waypoint_count=3,
    )

    assert plan["schema_version"] == "native_rigid_construction_phase_plan.v1"
    assert [row["phase_id"] for row in plan["phases"]] == [
        "pregrasp",
        "grasp_contact",
        "lift_clearance",
        "relocate_01",
        "relocate_02",
        "relocate_03",
        "place",
        "release",
        "settle_observe",
        "retreat",
        "recovery",
    ]
    assert plan["phases"][1]["gripper_state"] == "closed"
    assert plan["phases"][7]["gripper_state"] == "open"
    assert {
        "pregrasp_reachability",
        "grasp_contact",
        "relocation_path",
        "release",
        "retreat",
        "support_stability",
        "destination_containment",
        "reset_readback",
    }.issubset(plan["required_gate_ids"])
    assert plan["plan_digest"] == canonical_digest(
        plan, digest_field="plan_digest"
    )


def test_planar_push_compiles_without_a_fake_lift_or_grasp() -> None:
    plan = materialize_native_task_construction_phase_plan(
        _planar_push_fixture(), rigid_waypoint_count=3
    )

    assert plan["manipulation_strategy"] == "planar_push"
    assert plan["thresholds"]["minimum_lift_m"] == 0.0
    assert [row["phase_id"] for row in plan["phases"]] == [
        "precontact",
        "push_contact",
        "push_01",
        "push_02",
        "push_03",
        "push_release",
        "settle_observe",
        "retreat",
        "recovery",
    ]
    assert "grasp_contact" not in plan["required_gate_ids"]
    assert {
        "push_contact",
        "push_contact_maintained",
        "push_path",
        "support_contact",
        "destination_containment",
    }.issubset(plan["required_gate_ids"])
    assert plan["plan_digest"] == canonical_digest(
        plan, digest_field="plan_digest"
    )


def test_rigid_plan_refuses_an_unauthored_gripper_orientation() -> None:
    scene = _planar_push_fixture()
    affordance = scene["task_spec"]["interaction_affordance"]
    affordance["gripper_orientation_scoring_frame_xyzw"] = [0.0, 0.0, 0.0, 1.0]
    affordance["affordance_digest"] = canonical_digest(
        affordance, digest_field="affordance_digest"
    )

    with pytest.raises(NativeTaskConstructionPlanError) as excinfo:
        materialize_native_task_construction_phase_plan(scene)

    assert excinfo.value.errors == (
        "native_rigid_construction_gripper_orientation_unauthored",
    )


def test_planar_push_gate_uses_native_motion_contact_and_support_readback() -> None:
    plan = materialize_native_task_construction_phase_plan(
        _planar_push_fixture(), rigid_waypoint_count=3
    )
    phase_results = []
    for phase in plan["phases"]:
        pushing = phase["phase_id"] == "push_contact" or phase[
            "phase_id"
        ].startswith("push_0")
        sample = {
            "task_scoring_pose_world": [
                *phase["expected_scoring_position_world_m"],
                *phase["expected_scoring_orientation_world_xyzw"],
            ],
            "grasp_frame_position_world_m": list(phase["position_world_m"]),
            "task_robot_contact_peak_force_n": 1.0 if pushing else 0.0,
            "task_support_contact_peak_force_n": (
                1.0 if pushing or phase["phase_id"] == "settle_observe" else 0.0
            ),
            "task_scene_collision_peak_force_n": 0.0,
            "robot_scene_contact_peak_force_n": 0.0,
            "robot_task_forbidden_collision_peak_force_n": 0.0,
            "locked_joint_containment_violation": False,
            "finger_separation_m": 0.01 if pushing else 0.08,
        }
        samples = [sample]
        if phase["phase_id"] == "settle_observe":
            samples = [dict(sample) for _ in range(plan["settle_window_samples"])]
        phase_results.append(
            {
                "phase_id": phase["phase_id"],
                "target_reached": True,
                "gripper_state": phase["gripper_state"],
                "task_sample": sample,
                "task_samples": samples,
            }
        )

    # The worker retains every approach step while moving from precontact to
    # first contact. Those early samples must not be interpreted as a failure
    # to maintain contact during the subsequent push path.
    push_contact = next(
        row for row in phase_results if row["phase_id"] == "push_contact"
    )
    approach_sample = dict(push_contact["task_sample"])
    approach_sample["task_robot_contact_peak_force_n"] = 0.0
    push_contact["task_samples"].insert(0, approach_sample)

    passed = evaluate_rigid_construction_gates(
        phase_plan=plan,
        phase_results=phase_results,
        reset_replay={"passed": True},
    )
    assert passed["passed"] is True

    push_02 = next(row for row in phase_results if row["phase_id"] == "push_02")
    push_02["task_sample"]["task_robot_contact_peak_force_n"] = 0.0
    failed = evaluate_rigid_construction_gates(
        phase_plan=plan,
        phase_results=phase_results,
        reset_replay={"passed": True},
    )
    assert failed["passed"] is False
    assert (
        "native_rigid_construction_gate_failed:push_contact_maintained"
        in failed["blockers"]
    )


def test_generic_dual_task_rigid_fixture_changes_only_bound_subject_and_geometry() -> None:
    first = materialize_native_task_construction_phase_plan(
        _rigid_fixture(asset_id="rigid_a", scene_id="shared_scene")
    )
    second_scene = _rigid_fixture(asset_id="rigid_b", scene_id="shared_scene")
    second_scene["objects"][0]["pose_world"]["position_world_m"] = [1.0, 2.0, 0.8]
    second_scene["objects"][0]["reset_state"]["root_pose_world"][
        "position_world_m"
    ] = [1.0, 2.0, 0.8]
    second_scene["task_spec"]["destination_position_bounds_world_m"] = {
        "minimum": [1.14, 1.99, 0.79],
        "maximum": [1.16, 2.01, 0.81],
    }
    second_scene["task_spec"]["support_height_interval_m"] = [0.79, 0.81]
    second_scene["task_spec"]["minimum_translation_m"] = 0.10
    affordance = second_scene["task_spec"]["interaction_affordance"]
    affordance["asset_root_from_scoring_frame"]["position_m"] = [0.02, 0.0, 0.0]
    affordance["allowed_contact_prim_paths"] = ["/Asset/links/base"]
    affordance["affordance_digest"] = canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    second_scene["task_spec"]["start_pose_world"] = [
        1.02,
        2.0,
        0.8,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    second = materialize_native_task_construction_phase_plan(second_scene)

    assert first["subject_asset_id"] == "rigid_a"
    assert second["subject_asset_id"] == "rigid_b"
    assert first["required_gate_ids"] == second["required_gate_ids"]
    assert first["destination_position_world_m"] != second[
        "destination_position_world_m"
    ]


def test_rigid_destination_is_required_instead_of_invented() -> None:
    scene = _rigid_fixture(asset_id="rigid_a")
    del scene["task_spec"]["destination_position_bounds_world_m"]

    with pytest.raises(NativeTaskConstructionPlanError) as excinfo:
        materialize_native_task_construction_phase_plan(scene)

    assert excinfo.value.errors == (
        "native_rigid_construction_destination_missing",
    )


def test_dispatch_rejects_unknown_task_kind() -> None:
    scene = copy.deepcopy(_rigid_fixture(asset_id="rigid_a"))
    scene["task_kind"] = "force_insertion"

    with pytest.raises(NativeTaskConstructionPlanError) as excinfo:
        materialize_native_task_construction_phase_plan(scene)

    assert excinfo.value.errors == (
        "native_task_construction_task_kind_unsupported:force_insertion",
    )


def test_new_articulated_graph_never_falls_through_legacy_handle_adapter() -> None:
    scene = _articulated_840796_fixture()
    scene["task_spec"] = {
        "schema_version": "adp_task_spec.v2",
        "task_kind": "articulated_open_close",
        "articulation_graph_digest": "sha256:" + "f" * 64,
    }

    with pytest.raises(NativeTaskConstructionPlanError) as excinfo:
        materialize_native_task_construction_phase_plan(scene)

    assert excinfo.value.errors == (
        "native_articulated_construction_general_interaction_affordance_missing",
    )


def test_rigid_gate_evaluator_requires_native_object_motion_contact_and_release() -> None:
    plan = materialize_native_task_construction_phase_plan(
        _rigid_fixture(asset_id="rigid_evaluation_fixture"),
        rigid_waypoint_count=3,
    )
    phase_results = []
    for phase in plan["phases"]:
        position = phase["expected_scoring_position_world_m"]
        closed = phase["gripper_state"] == "closed"
        sample = {
            "task_scoring_pose_world": [*position, 0.0, 0.0, 0.0, 1.0],
            "grasp_frame_position_world_m": [
                position[0],
                position[1],
                position[2] + 0.06,
            ],
            "task_robot_contact_peak_force_n": 1.0 if closed else 0.0,
            "task_support_contact_peak_force_n": (
                1.0 if phase["phase_id"] == "settle_observe" else 0.0
            ),
            "task_scene_collision_peak_force_n": 0.0,
            "robot_scene_contact_peak_force_n": 0.0,
            "robot_task_forbidden_collision_peak_force_n": 0.0,
            "locked_joint_containment_violation": False,
            "finger_separation_m": 0.01 if closed else 0.08,
        }
        task_samples = [sample]
        if phase["phase_id"] == "settle_observe":
            task_samples = [dict(sample) for _ in range(plan["settle_window_samples"])]
        phase_results.append(
            {
                "phase_id": phase["phase_id"],
                "target_reached": True,
                "gripper_state": phase["gripper_state"],
                "task_sample": sample,
                "task_samples": task_samples,
            }
        )

    result = evaluate_rigid_construction_gates(
        phase_plan=plan,
        phase_results=phase_results,
        reset_replay={"passed": True},
    )

    assert result["passed"] is True
    assert result["blockers"] == []

    lift = next(row for row in phase_results if row["phase_id"] == "lift_clearance")
    lift["task_samples"][0]["locked_joint_containment_violation"] = True
    result = evaluate_rigid_construction_gates(
        phase_plan=plan,
        phase_results=phase_results,
        reset_replay={"passed": True},
    )
    assert result["passed"] is False
    assert "native_rigid_construction_gate_failed:base_collision_clearance" in result[
        "blockers"
    ]
    lift["task_samples"][0]["locked_joint_containment_violation"] = False

    relocate = next(
        row for row in phase_results if row["phase_id"].startswith("relocate_")
    )
    relocate["task_sample"]["task_scoring_pose_world"][:3] = plan[
        "start_position_world_m"
    ]
    result = evaluate_rigid_construction_gates(
        phase_plan=plan,
        phase_results=phase_results,
        reset_replay={"passed": True},
    )
    assert result["passed"] is False
    assert "native_rigid_construction_gate_failed:relocation_path" in result[
        "blockers"
    ]
