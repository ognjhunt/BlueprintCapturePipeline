from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.adp009d_control_episode import run_task_neutral_controls
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_articulated_control_plan import (
    materialize_native_articulated_control_plan,
)
from blueprint_pipeline.native_task_construction_plan import (
    evaluate_rigid_construction_gates,
    materialize_native_task_construction_phase_plan,
)
from blueprint_pipeline.native_task_control_plan import (
    NativeTaskControlPlanError,
    materialize_native_task_control_plan,
)


def _articulated_scene() -> dict:
    motion = {
        "hinge_point_world_m": [0.0, 0.0, 1.0],
        "hinge_axis_world_unit": [0.0, 0.0, 1.0],
        "handle_grasp_point_closed_world_m": [0.5, 0.0, 1.0],
        "authored_limits_degrees": [0.0, 90.0],
        "scripted_sweep_angle_degrees": 50.0,
    }
    scene = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "scene_id": "840796",
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
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")
    return scene


def _articulated_construction(scene: dict) -> dict:
    clearance = {
        "scene_plan_digest": scene["plan_digest"],
        "phases": [{"phase_id": "approach"}],
        "plan_digest": "",
    }
    clearance["plan_digest"] = canonical_digest(
        clearance, digest_field="plan_digest"
    )
    construction = {
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
    construction["result_digest"] = canonical_digest(
        construction, digest_field="result_digest"
    )
    return construction


def _rigid_scene(
    *,
    scene_id: str,
    asset_id: str,
    root_position: list[float] | None = None,
    scoring_offset: list[float] | None = None,
) -> dict:
    root = list(root_position or [3.4681748, -3.3100837, 0.526465])
    offset = list(scoring_offset or [0.0, 0.0, 0.0])
    start = [root[index] + offset[index] for index in range(3)]
    destination = [start[0] + 0.30, start[1], start[2]]
    affordance = {
        "schema_version": "native_rigid_interaction_affordance.v1",
        "subject_asset_id": asset_id,
        "scoring_frame_id": "task_scoring_frame",
        "asset_root_from_scoring_frame": {
            "position_m": offset,
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "contact_point_scoring_frame_m": [0.0, 0.0, 0.06],
        "approach_unit_scoring_frame": [0.0, -1.0, 0.0],
        "lift_unit_world": [0.0, 0.0, 1.0],
        "gripper_orientation_scoring_frame_xyzw": [0.0, 0.0, 0.0, 1.0],
        "pregrasp_clearance_m": 0.12,
        "arrival_orientation_tolerance_rad": 0.05,
        "allowed_contact_prim_paths": ["/Asset/links/base"],
        "intended_support_prim_paths": ["/Scene/support"],
        "affordance_digest": "",
    }
    affordance["affordance_digest"] = canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    task_spec = {
        "schema_version": "adp_task_spec.v2",
        "task_kind": "rigid_pick_place",
        "subject_asset_id": asset_id,
        "start_pose_world": [*start, 0.0, 0.0, 0.0, 1.0],
        "destination_position_bounds_world_m": {
            "minimum": [destination[0] - 0.02, destination[1] - 0.02, destination[2] - 0.02],
            "maximum": [destination[0] + 0.02, destination[1] + 0.02, destination[2] + 0.02],
        },
        "destination_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "destination_orientation_tolerance_rad": 0.05,
        "support_height_interval_m": [destination[2] - 0.03, destination[2] + 0.03],
        "workspace_position_bounds_world_m": {
            "minimum": [min(start[0], destination[0]) - 0.2, start[1] - 0.2, start[2] - 0.1],
            "maximum": [max(start[0], destination[0]) + 0.2, start[1] + 0.2, start[2] + 0.3],
        },
        "minimum_lift_m": 0.08,
        "minimum_translation_m": 0.15,
        "settle_window_samples": 8,
        "task_contact_minimum_force_n": 0.5,
        "collision_failure_minimum_force_n": 1.0,
        "reset_translation_tolerance_m": 0.001,
        "reset_orientation_tolerance_rad": 0.01,
        "settle_position_tolerance_m": 0.002,
        "settle_orientation_tolerance_rad": 0.02,
        "relocation_tracking_tolerance_m": 0.02,
        "release_gripper_width_min_m": 0.05,
        "release_required": True,
        "movement_epsilon_m": 0.001,
        "maximum_action_steps": 512,
        "interaction_affordance": affordance,
    }
    scene = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "scene_id": scene_id,
        "task_kind": "rigid_pick_place",
        "scenario": {"cell_id": f"{scene_id}-canonical", "seed": 19},
        "task_spec": task_spec,
        "objects": [
            {
                "semantic_role": "task_object",
                "task_subject": True,
                "asset_id": asset_id,
                "pose_world": {
                    "position_world_m": root,
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "reset_state": {
                    "root_pose_world": {
                        "position_world_m": root,
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "joint_positions": {},
                },
            }
        ],
        "plan_digest": "",
    }
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")
    return scene


def _rigid_construction(scene: dict) -> dict:
    phase_plan = materialize_native_task_construction_phase_plan(
        scene, rigid_waypoint_count=3, maximum_steps_per_phase=32
    )
    phase_results = []
    for phase in phase_plan["phases"]:
        position = list(phase["expected_scoring_position_world_m"])
        orientation = list(phase["expected_scoring_orientation_world_xyzw"])
        closed = phase["gripper_state"] == "closed"
        sample = {
            "task_scoring_pose_world": [*position, *orientation],
            "grasp_frame_position_world_m": list(phase["position_world_m"]),
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
        count = (
            phase_plan["settle_window_samples"]
            if phase["phase_id"] == "settle_observe"
            else phase_plan["execution_parameters"]["stable_samples"]
        )
        phase_results.append(
            {
                "phase_id": phase["phase_id"],
                "target_reached": True,
                "gripper_state": phase["gripper_state"],
                "steps": count,
                "task_sample": dict(sample),
                "task_samples": [dict(sample) for _ in range(count)],
            }
        )
    gate_evaluation = evaluate_rigid_construction_gates(
        phase_plan=phase_plan,
        phase_results=phase_results,
        reset_replay={"passed": True},
    )
    assert gate_evaluation["passed"] is True
    construction = {
        "schema_version": "native_task_arena_construction_result.v1",
        "status": "completed",
        "construction_gate_qualified": True,
        "blockers": [],
        "scene_plan_digest": scene["plan_digest"],
        "construction_phase_plan": phase_plan,
        "phase_results": phase_results,
        "rigid_construction_gates": gate_evaluation,
        "camera_gates": {
            role: {"passed": True} for role in ("external", "wrist", "overview")
        },
        "reset_replay": {"passed": True},
        "result_digest": "",
    }
    construction["result_digest"] = canonical_digest(
        construction, digest_field="result_digest"
    )
    return construction


def test_articulated_compatibility_adapter_is_byte_semantically_unchanged() -> None:
    scene = _articulated_scene()
    construction = _articulated_construction(scene)

    expected = materialize_native_articulated_control_plan(
        scene_plan=scene, construction_result=construction
    )
    dispatched = materialize_native_task_control_plan(
        scene_plan=scene, construction_result=construction
    )

    assert dispatched == expected


def test_840313_rigid_fixture_replays_only_qualified_construction_phases() -> None:
    scene = _rigid_scene(scene_id="840313", asset_id="rigid_fixture")
    construction = _rigid_construction(scene)

    plan = materialize_native_task_control_plan(
        scene_plan=scene, construction_result=construction
    )

    phases = construction["construction_phase_plan"]["phases"]
    assert [row["phase_id"] for row in plan["scripted_positive_actions"]] == [
        row["phase_id"] for row in phases
    ]
    assert [row["target_position_world_m"] for row in plan["scripted_positive_actions"]] == [
        row["position_world_m"] for row in phases
    ]
    assert [row["target_quaternion_world_xyzw"] for row in plan["scripted_positive_actions"]] == [
        row["orientation_world_xyzw"] for row in phases
    ]
    assert plan["zero_action_steps"] == scene["task_spec"]["settle_window_samples"]
    assert (
        plan[
            "positive_trajectory_reexecutes_exact_qualified_phase_targets_and_budgets"
        ]
        is True
    )
    assert plan["plan_digest"] == canonical_digest(
        plan, digest_field="plan_digest"
    )


def test_generic_rigid_fixture_preserves_nonidentity_scoring_frame_affordance() -> None:
    scene = _rigid_scene(
        scene_id="unseen_admitted_room",
        asset_id="multi_link_locked_subject",
        root_position=[1.0, 2.0, 0.8],
        scoring_offset=[0.02, -0.01, 0.03],
    )
    construction = _rigid_construction(scene)

    plan = materialize_native_task_control_plan(
        scene_plan=scene, construction_result=construction
    )

    assert plan["task_kind"] == "rigid_pick_place"
    assert plan["interaction_affordance_digest"] == scene["task_spec"][
        "interaction_affordance"
    ]["affordance_digest"]
    assert plan["construction_gate_evaluation_digest"] == construction[
        "rigid_construction_gates"
    ]["evaluation_digest"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (
            lambda scene, _construction: scene["task_spec"].pop(
                "interaction_affordance"
            ),
            "native_rigid_control_interaction_affordance_invalid",
        ),
        (
            lambda _scene, construction: construction[
                "rigid_construction_gates"
            ].update(passed=False),
            "native_rigid_control_gate_evaluation",
        ),
        (
            lambda _scene, construction: construction["phase_results"][0].pop(
                "steps"
            ),
            "native_rigid_control_phase_steps_invalid:0",
        ),
        (
            lambda _scene, construction: construction["camera_gates"][
                "wrist"
            ].update(passed=False),
            "native_rigid_control_camera_preflight_incomplete",
        ),
        (
            lambda _scene, construction: construction["reset_replay"].update(
                passed=False
            ),
            "native_rigid_control_reset_preflight_incomplete",
        ),
    ),
)
def test_rigid_controls_fail_closed_on_missing_exact_evidence(
    mutation, expected
) -> None:
    scene = _rigid_scene(scene_id="generic", asset_id="rigid_subject")
    construction = _rigid_construction(scene)
    mutation(scene, construction)
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")
    construction["scene_plan_digest"] = scene["plan_digest"]
    construction["result_digest"] = canonical_digest(
        construction, digest_field="result_digest"
    )

    with pytest.raises(NativeTaskControlPlanError, match=expected):
        materialize_native_task_control_plan(
            scene_plan=copy.deepcopy(scene), construction_result=construction
        )


def test_legacy_rigid_task_without_exact_affordance_fails_closed() -> None:
    scene = _rigid_scene(scene_id="840313", asset_id="legacy_rigid")
    scene["task_spec"] = {
        "schema_version": "adp_task_spec.v1",
        "task_kind": "rigid_pick_place",
        "settle_window_samples": 8,
        "maximum_action_steps": 64,
    }
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")
    construction = _rigid_construction(
        _rigid_scene(scene_id="source", asset_id="legacy_rigid")
    )
    construction["scene_plan_digest"] = scene["plan_digest"]
    construction["result_digest"] = canonical_digest(
        construction, digest_field="result_digest"
    )

    with pytest.raises(
        NativeTaskControlPlanError, match="native_rigid_control_task_spec_invalid"
    ):
        materialize_native_task_control_plan(
            scene_plan=scene, construction_result=construction
        )


class _RigidControlEnvironment:
    def __init__(
        self, *, scene: dict, construction: dict, force_wrong_orientation: bool = False
    ):
        self._scene = scene
        self._force_wrong_orientation = force_wrong_orientation
        self._phase_by_target = {
            tuple(phase["position_world_m"]): phase
            for phase in construction["construction_phase_plan"]["phases"]
        }
        self._pending_target = None
        self._pending_gripper = 0.0
        self.reset()

    def reset(self) -> None:
        self.joints = [0.0] * 7
        self.gripper = 0.0
        self.pose = list(self._scene["task_spec"]["start_pose_world"])
        self.grasp = [0.0, 0.0, 0.0]
        self.grasp_orientation = [0.0, 0.0, 0.0, 1.0]
        self._has_grasped = False
        self._pending_target = None
        self._pending_orientation = None

    def read_arm_joint_positions(self) -> list[float]:
        return list(self.joints)

    def hold_action(self, *, gripper_command: float) -> list[float]:
        self._pending_target = None
        self._pending_gripper = float(gripper_command)
        return [*self.joints, float(gripper_command)]

    def scripted_action_for_pose(
        self,
        *,
        target_position_world_m,
        target_quaternion_world_xyzw,
        gripper_command,
        max_joint_delta_rad,
        max_joint_setpoint_lead_rad,
    ) -> list[float]:
        assert target_quaternion_world_xyzw is not None
        assert max_joint_delta_rad == pytest.approx(0.03)
        assert max_joint_setpoint_lead_rad == pytest.approx(0.20)
        self._pending_target = [float(value) for value in target_position_world_m]
        self._pending_orientation = [
            float(value) for value in target_quaternion_world_xyzw
        ]
        self._pending_gripper = float(gripper_command)
        return [*self.joints, self._pending_gripper]

    def step(self, action) -> None:
        self.gripper = float(action[-1])
        if self._pending_target is None:
            return
        self.grasp = list(self._pending_target)
        self.grasp_orientation = (
            [1.0, 0.0, 0.0, 0.0]
            if self._force_wrong_orientation
            else list(self._pending_orientation)
        )
        phase = self._phase_by_target[tuple(self._pending_target)]
        if self.gripper > 0.5:
            self._has_grasped = True
            self.pose = [
                *phase["expected_scoring_position_world_m"],
                *phase["expected_scoring_orientation_world_xyzw"],
            ]

    def read_object_sample(self) -> dict:
        closed = self.gripper > 0.5
        return {
            "task_object_pose_world": list(self.pose),
            "gripper_width_m": 0.01 if closed else 0.08,
            "task_contact_active": closed and self._has_grasped,
            "support_contact_active": (not closed) and self._has_grasped,
            "robot_collision_failure": False,
            "scene_collision_failure": False,
            "containment_violation": False,
            "forbidden_robot_task_collision_failure": False,
            "locked_joint_containment_violation": False,
            "grasp_frame_position_world_m": list(self.grasp),
            "grasp_frame_orientation_world_xyzw": list(self.grasp_orientation),
        }


def test_generic_rigid_plan_runs_zero_then_positive_through_shared_scorer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import adp009d_control_episode as controls_module

    scene = _rigid_scene(scene_id="generic", asset_id="rigid_subject")
    construction = _rigid_construction(scene)
    plan = materialize_native_task_control_plan(
        scene_plan=scene, construction_result=construction
    )
    observation_index = {"value": 0}

    def fake_observation(*_args, **kwargs):
        row = {
            "observation_index": observation_index["value"],
            "kind": kwargs["kind"],
            "views": {},
        }
        observation_index["value"] += 1
        return row

    monkeypatch.setattr(controls_module, "_persist_observation", fake_observation)
    monkeypatch.setattr(
        controls_module,
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

    pair = run_task_neutral_controls(
        environment=_RigidControlEnvironment(
            scene=scene, construction=construction
        ),
        task_spec=scene["task_spec"],
        control_plan=plan,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    assert pair["execution_order"] == [
        "zero_action_negative",
        "deterministic_scripted_positive",
    ]
    assert [row["control_passed"] for row in pair["controls"]] == [True, True]
    assert pair["cell_admitted_for_policy_execution"] is True


def test_generic_rigid_control_rejects_correct_position_with_wrong_orientation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import adp009d_control_episode as controls_module

    scene = _rigid_scene(scene_id="generic", asset_id="rigid_subject")
    construction = _rigid_construction(scene)
    plan = materialize_native_task_control_plan(
        scene_plan=scene, construction_result=construction
    )
    counter = {"value": 0}

    def fake_observation(*_args, **kwargs):
        row = {
            "observation_index": counter["value"],
            "kind": kwargs["kind"],
            "views": {},
        }
        counter["value"] += 1
        return row

    monkeypatch.setattr(controls_module, "_persist_observation", fake_observation)
    monkeypatch.setattr(
        controls_module,
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

    pair = run_task_neutral_controls(
        environment=_RigidControlEnvironment(
            scene=scene,
            construction=construction,
            force_wrong_orientation=True,
        ),
        task_spec=scene["task_spec"],
        control_plan=plan,
        gripper_open_command=0.0,
        gripper_closed_command=1.0,
        output_dir=tmp_path,
    )

    positive = next(
        row
        for row in pair["controls"]
        if row["control_id"] == "deterministic_scripted_positive"
    )
    assert positive["control_passed"] is False
    assert pair["cell_admitted_for_policy_execution"] is False
