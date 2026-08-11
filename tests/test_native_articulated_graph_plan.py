from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.adp009d_control_episode import run_task_neutral_controls
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_construction_plan import (
    NativeTaskConstructionPlanError,
    evaluate_graph_articulated_construction_gates,
    materialize_native_task_construction_phase_plan,
)
from blueprint_pipeline.native_task_arena_construction_worker import (
    _evaluate_task_construction_gates,
)
from blueprint_pipeline.native_task_control_plan import (
    NativeTaskControlPlanError,
    materialize_native_task_control_plan,
)


def _graph() -> dict:
    return {
        "schema_version": "adp_articulation_graph.v1",
        "links": [
            {"link_id": "base", "is_root": True, "semantic_role": "fixed_body"},
            {"link_id": "panel", "is_root": False, "semantic_role": "target"},
            {
                "link_id": "follower",
                "is_root": False,
                "semantic_role": "dependent",
            },
            {"link_id": "roller", "is_root": False, "semantic_role": "passive"},
            {"link_id": "dial", "is_root": False, "semantic_role": "locked"},
        ],
        "joints": [
            {
                "joint_id": "panel_hinge",
                "parent_link_id": "base",
                "child_link_id": "panel",
                "joint_type": "revolute",
                "role": "target",
                "axis": [0.0, 0.0, 1.0],
                "limits": [0.0, 1.2],
                "reset_position": 0.0,
                "reset_tolerance": 0.001,
                "drive": {
                    "drive_type": "force",
                    "stiffness": 0.0,
                    "damping": 2.0,
                    "maximum_force": 80.0,
                },
                "dependency": None,
            },
            {
                "joint_id": "follower_hinge",
                "parent_link_id": "panel",
                "child_link_id": "follower",
                "joint_type": "revolute",
                "role": "dependent",
                "axis": [0.0, 0.0, 1.0],
                "limits": [-0.1, 0.1],
                "reset_position": 0.0,
                "reset_tolerance": 0.001,
                "drive": {
                    "drive_type": "force",
                    "stiffness": 4.0,
                    "damping": 1.0,
                    "maximum_force": 20.0,
                },
                "dependency": {
                    "driver_joint_id": "panel_hinge",
                    "multiplier": 0.05,
                    "offset": 0.0,
                    "tolerance": 0.001,
                },
            },
            {
                "joint_id": "roller_axis",
                "parent_link_id": "base",
                "child_link_id": "roller",
                "joint_type": "continuous",
                "role": "passive",
                "axis": [0.0, 1.0, 0.0],
                "limits": [-10.0, 10.0],
                "reset_position": 0.0,
                "reset_tolerance": 0.01,
                "drive": {
                    "drive_type": "none",
                    "stiffness": 0.0,
                    "damping": 0.2,
                    "maximum_force": 0.0,
                },
                "dependency": None,
            },
            {
                "joint_id": "dial_axis",
                "parent_link_id": "base",
                "child_link_id": "dial",
                "joint_type": "revolute",
                "role": "locked",
                "axis": [0.0, 1.0, 0.0],
                "limits": [-1.0, 1.0],
                "reset_position": 0.0,
                "reset_tolerance": 0.001,
                "drive": {
                    "drive_type": "force",
                    "stiffness": 100.0,
                    "damping": 10.0,
                    "maximum_force": 20.0,
                },
                "dependency": None,
            },
        ],
        "collision_pairs": [
            {"link_a": "base", "link_b": "panel", "collision_enabled": True}
        ],
        "success_predicate": {
            "combination": "all",
            "joint_intervals": {"panel_hinge": [0.7, 0.95]},
        },
    }


def _scene() -> dict:
    graph = _graph()
    graph_digest = canonical_digest(graph)
    path = []
    for index, target in enumerate((0.0, 0.4, 0.8)):
        path.append(
            {
                "waypoint_id": f"mechanism_path_{index:02d}",
                "joint_positions": {
                    "panel_hinge": target,
                    "follower_hinge": target * 0.05,
                    "roller_axis": 0.0,
                    "dial_axis": 0.0,
                },
                "contact_pose_asset_root": {
                    "position_m": [0.5, 0.1 * index, 0.6],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "clearance_unit_asset_root": [0.0, -1.0, 0.0],
            }
        )
    affordance = {
        "schema_version": "native_articulated_graph_interaction_affordance.v1",
        "subject_asset_id": "generic_multi_joint_asset",
        "articulation_graph_digest": graph_digest,
        "kinematic_path_receipt_digest": "sha256:" + "a" * 64,
        "contact_link_id": "panel",
        "contact_body_prim_paths": ["/Asset/links/panel/contact"],
        "contact_point_link_m": [0.4, 0.0, 0.0],
        "approach_unit_asset_root": [0.0, -1.0, 0.0],
        "retreat_unit_asset_root": [0.0, -1.0, 0.0],
        "gripper_orientation_contact_xyzw": [0.0, 0.0, 0.0, 1.0],
        "precontact_clearance_m": 0.12,
        "sweep_clearance_m": 0.025,
        "retreat_clearance_m": 0.12,
        "arrival_tolerance_m": 0.02,
        "arrival_orientation_tolerance_rad": 0.04,
        "arrival_stability_steps": 2,
        "motion_minimum_steps": 1,
        "motion_maximum_steps": 25,
        "gripper_dwell_minimum_steps": 5,
        "gripper_dwell_maximum_steps": 12,
        "max_joint_delta_rad": 0.03,
        "max_joint_setpoint_lead_rad": 0.2,
        "joint_contact_path": path,
        "affordance_digest": "",
    }
    affordance["affordance_digest"] = canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    task_spec = {
        "schema_version": "adp_task_spec.v2",
        "task_kind": "articulated_open_close",
        "articulation_graph": graph,
        "articulation_graph_digest": graph_digest,
        "interaction_affordance": affordance,
        "settle_window_samples": 5,
        "maximum_settled_target_speed": 0.03,
        "locked_joint_motion_tolerance": 0.01,
        "movement_epsilon": 0.01,
        "maximum_action_steps": 512,
    }
    scene = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "scene_id": "generic_unseen_room",
        "task_kind": "articulated_open_close",
        "scenario": {"cell_id": "generic-canonical", "seed": 31},
        "task_spec": task_spec,
        "objects": [
            {
                "semantic_role": "task_object",
                "task_subject": True,
                "asset_id": "generic_multi_joint_asset",
                "object_type": "ARTICULATION",
                "pose_world": {
                    "position_world_m": [1.0, 2.0, 0.5],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "reset_state": {
                    "root_pose_world": {
                        "position_world_m": [1.0, 2.0, 0.5],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "joint_positions": {
                        "panel_hinge": 0.0,
                        "follower_hinge": 0.0,
                        "roller_axis": 0.0,
                        "dial_axis": 0.0,
                    },
                },
            }
        ],
        "plan_digest": "",
    }
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")
    return scene


def _construction(scene: dict) -> dict:
    phase_plan = materialize_native_task_construction_phase_plan(scene)
    reset = dict(phase_plan["joint_reset_positions"])
    sample = {
        "joint_positions": reset,
        "task_contact_active": False,
        "joint_limit_violation": False,
        "containment_violation": False,
        "robot_collision_failure": False,
        "scene_collision_failure": False,
        "retreat_completed": True,
    }
    phase_results = [
        {
            "phase_id": phase["phase_id"],
            "target_reached": True,
            "task_sample": copy.deepcopy(sample),
            "task_samples": [copy.deepcopy(sample), copy.deepcopy(sample)],
        }
        for phase in phase_plan["phases"]
    ]
    gates = evaluate_graph_articulated_construction_gates(
        phase_plan=phase_plan,
        phase_results=phase_results,
        reset_replay={"passed": True},
    )
    assert gates["passed"] is True
    dispatched = _evaluate_task_construction_gates(
        phase_plan=phase_plan,
        phase_results=phase_results,
        reset_replay={"passed": True},
    )
    assert dispatched is not None
    assert dispatched[0] == "articulated_graph_construction_gates"
    assert dispatched[1] == gates
    result = {
        "schema_version": "native_task_arena_construction_result.v1",
        "status": "completed",
        "construction_gate_qualified": True,
        "blockers": [],
        "scene_plan_digest": scene["plan_digest"],
        "construction_phase_plan": phase_plan,
        "phase_results": phase_results,
        "articulated_graph_construction_gates": gates,
        "camera_gates": {
            role: {"passed": True} for role in ("external", "wrist", "overview")
        },
        "reset_replay": {"passed": True},
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def _redigest_affordance(scene: dict) -> None:
    affordance = scene["task_spec"]["interaction_affordance"]
    affordance["affordance_digest"] = canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")


def test_graph_articulated_construction_binds_complete_graph_and_exact_paths() -> None:
    plan = materialize_native_task_construction_phase_plan(_scene())

    assert plan["schema_version"] == (
        "native_articulated_graph_construction_phase_plan.v1"
    )
    assert plan["joint_ids_by_role"] == {
        "target": ["panel_hinge"],
        "dependent": ["follower_hinge"],
        "passive": ["roller_axis"],
        "locked": ["dial_axis"],
    }
    assert [row["phase_id"] for row in plan["exact_contact_phases"]] == [
        "approach",
        "contact_open",
        "contact_close",
        "joint_path_01",
        "joint_path_02",
        "release",
        "retreat",
    ]
    assert plan["exact_contact_phases"][4]["expected_joint_positions"][
        "panel_hinge"
    ] == pytest.approx(0.8)
    assert plan["interaction_affordance"]["contact_body_prim_paths"] == [
        "/Asset/links/panel/contact"
    ]
    assert all(
        row["arrival_orientation_tolerance_rad"] == pytest.approx(0.04)
        for row in plan["phases"]
        + plan["exact_contact_phases"]
    )
    assert plan["plan_digest"] == canonical_digest(plan, digest_field="plan_digest")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (
            lambda scene: scene["task_spec"]["interaction_affordance"][
                "joint_contact_path"
            ][1]["joint_positions"].pop("roller_axis"),
            "native_articulated_graph_construction_joint_path_set_invalid",
        ),
        (
            lambda scene: scene["task_spec"]["interaction_affordance"][
                "joint_contact_path"
            ][1]["joint_positions"].update(follower_hinge=0.09),
            "native_articulated_graph_construction_dependent_joint_path_invalid",
        ),
        (
            lambda scene: scene["task_spec"]["interaction_affordance"][
                "joint_contact_path"
            ][1]["joint_positions"].update(dial_axis=0.1),
            "native_articulated_graph_construction_locked_joint_path_invalid",
        ),
        (
            lambda scene: scene["task_spec"]["interaction_affordance"][
                "joint_contact_path"
            ][-1]["joint_positions"].update(
                panel_hinge=0.5, follower_hinge=0.025
            ),
            "native_articulated_graph_construction_path_target_mismatch",
        ),
        (
            lambda scene: scene["task_spec"]["interaction_affordance"].update(
                contact_link_id="roller"
            ),
            "native_articulated_graph_construction_contact_link_not_target_driven",
        ),
        (
            lambda scene: scene["task_spec"]["interaction_affordance"].pop(
                "arrival_orientation_tolerance_rad"
            ),
            (
                "native_articulated_graph_construction_"
                "arrival_orientation_tolerance_rad_invalid"
            ),
        ),
    ),
)
def test_graph_articulated_construction_fails_closed_on_unbound_mechanism_path(
    mutation, expected
) -> None:
    scene = _scene()
    mutation(scene)
    _redigest_affordance(scene)

    with pytest.raises(NativeTaskConstructionPlanError, match=expected):
        materialize_native_task_construction_phase_plan(scene)


def test_graph_articulated_clearance_gate_uses_every_native_sample() -> None:
    scene = _scene()
    construction = _construction(scene)
    plan = construction["construction_phase_plan"]
    phase_results = construction["phase_results"]

    phase_results[2]["task_samples"][0]["robot_collision_failure"] = True
    evaluation = evaluate_graph_articulated_construction_gates(
        phase_plan=plan,
        phase_results=phase_results,
        reset_replay={"passed": True},
    )
    assert evaluation["passed"] is False
    assert (
        "native_articulated_graph_construction_gate_failed:base_collision_clearance"
        in evaluation["blockers"]
    )

    phase_results[2]["task_samples"] = []
    with pytest.raises(
        NativeTaskConstructionPlanError,
        match="native_articulated_graph_construction_path_readback_missing",
    ):
        evaluate_graph_articulated_construction_gates(
            phase_plan=plan,
            phase_results=phase_results,
            reset_replay={"passed": True},
        )


def test_graph_articulated_control_replays_only_qualified_exact_contact_path() -> None:
    scene = _scene()
    construction = _construction(scene)

    control = materialize_native_task_control_plan(
        scene_plan=scene, construction_result=construction
    )

    phase_plan = construction["construction_phase_plan"]
    assert control["task_kind"] == "articulated_open_close"
    assert control["target_joint_ids"] == ["panel_hinge"]
    assert control["dependent_joint_ids"] == ["follower_hinge"]
    assert control["passive_joint_ids"] == ["roller_axis"]
    assert control["locked_joint_ids"] == ["dial_axis"]
    assert [row["phase_id"] for row in control["scripted_positive_actions"]] == [
        row["phase_id"] for row in phase_plan["exact_contact_phases"]
    ]
    assert control["scripted_positive_actions"][4]["expected_joint_positions"][
        "panel_hinge"
    ] == pytest.approx(0.8)
    assert all(
        row["arrival_orientation_tolerance_rad"] == pytest.approx(0.04)
        for row in control["scripted_positive_actions"]
    )
    assert control["construction_gate_evaluation_digest"]
    assert control["plan_digest"] == canonical_digest(
        control, digest_field="plan_digest"
    )


def test_graph_articulated_control_rejects_incomplete_clearance_readback() -> None:
    scene = _scene()
    construction = _construction(scene)
    construction["phase_results"][1]["task_samples"] = []
    construction["result_digest"] = canonical_digest(
        construction, digest_field="result_digest"
    )

    with pytest.raises(
        NativeTaskControlPlanError,
        match="native_articulated_graph_control_gate_evaluation_failed",
    ):
        materialize_native_task_control_plan(
            scene_plan=scene, construction_result=construction
        )


class _GraphControlEnvironment:
    def __init__(self, construction: dict):
        phases = construction["construction_phase_plan"]["exact_contact_phases"]
        self._joint_positions_by_target = {
            tuple(row["position_world_m"]): dict(row["expected_joint_positions"])
            for row in phases
            if "expected_joint_positions" in row
        }
        self._retreat_target = list(phases[-1]["position_world_m"])
        self.reset()

    def reset(self) -> None:
        self.arm = [0.0] * 7
        self.gripper = 0.0
        self.grasp = [0.0, 0.0, 0.0]
        self.joints = {
            "panel_hinge": 0.0,
            "follower_hinge": 0.0,
            "roller_axis": 0.0,
            "dial_axis": 0.0,
        }

    def read_arm_joint_positions(self) -> list[float]:
        return list(self.arm)

    def hold_action(self, *, gripper_command: float) -> list[float]:
        return [*self.arm, float(gripper_command)]

    def scripted_action_for_pose(
        self,
        *,
        target_position_world_m,
        target_quaternion_world_xyzw,
        gripper_command,
        max_joint_delta_rad,
        max_joint_setpoint_lead_rad,
    ) -> list[float]:
        assert target_quaternion_world_xyzw == [0.0, 0.0, 0.0, 1.0]
        assert max_joint_delta_rad == pytest.approx(0.03)
        assert max_joint_setpoint_lead_rad == pytest.approx(0.2)
        return [
            *[float(value) for value in target_position_world_m],
            0.0,
            0.0,
            0.0,
            0.0,
            float(gripper_command),
        ]

    def step(self, action) -> None:
        self.arm = [float(value) for value in action[:7]]
        self.gripper = float(action[-1])
        self.grasp = list(self.arm[:3])
        expected = self._joint_positions_by_target.get(tuple(self.grasp))
        if expected is not None:
            self.joints = dict(expected)

    def read_task_sample(self) -> dict:
        return {
            "joint_positions": dict(self.joints),
            "joint_velocities_per_s": {joint_id: 0.0 for joint_id in self.joints},
            "task_contact_active": self.gripper > 0.5,
            "joint_limit_violation": False,
            "containment_violation": False,
            "robot_collision_failure": False,
            "scene_collision_failure": False,
            "retreat_completed": (
                self.gripper <= 0.5 and self.grasp == self._retreat_target
            ),
            "grasp_frame_position_world_m": list(self.grasp),
            "grasp_frame_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
        }


def test_graph_articulated_control_runs_zero_then_positive_through_shared_scorer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import adp009d_control_episode as controls_module

    scene = _scene()
    construction = _construction(scene)
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
        environment=_GraphControlEnvironment(construction),
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
