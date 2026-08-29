from __future__ import annotations

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_diagnostic_native_arena_compiler import (
    TaskEvaluationDiagnosticNativeArenaCompilerError,
    _derive_task_aware_franka_reset,
    _legacy_robot_placement_is_clear,
    _runtime_subject_task_spec,
)
from blueprint_pipeline.task_evaluation_robot_placement_trajectory import (
    placement_trajectory_from_native_plan,
)


def _task_spec() -> dict:
    affordance = {
        "schema_version": "native_rigid_interaction_affordance.v1",
        "subject_asset_id": "scene-839873-mug-replacement",
        "affordance_digest": "",
    }
    affordance["affordance_digest"] = canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    return {
        "task_kind": "rigid_pick_place",
        "subject_asset_id": "scene-839873-mug-replacement",
        "interaction_affordance": affordance,
    }


def test_runtime_subject_alias_updates_task_and_affordance_together() -> None:
    source = _task_spec()
    result = _runtime_subject_task_spec(source)

    assert result["subject_asset_id"] == "scene_839873_mug_replacement"
    assert result["source_subject_identity"] == "scene-839873-mug-replacement"
    assert result["interaction_affordance"]["subject_asset_id"] == result[
        "subject_asset_id"
    ]
    assert result["interaction_affordance"]["affordance_digest"] == canonical_digest(
        result["interaction_affordance"], digest_field="affordance_digest"
    )
    assert source["subject_asset_id"] == "scene-839873-mug-replacement"


def test_runtime_subject_alias_refuses_cross_bound_affordance() -> None:
    source = _task_spec()
    source["interaction_affordance"]["subject_asset_id"] = "other-object"

    with pytest.raises(
        TaskEvaluationDiagnosticNativeArenaCompilerError,
        match="diagnostic_native_compiler_interaction_affordance_invalid",
    ):
        _runtime_subject_task_spec(source)


def test_blocked_overlap_placement_cannot_feed_native_compiler() -> None:
    workspace = {"status": "blocked"}
    placement = {
        "status": "abstained",
        "mesh_triangle_aabb_overlap_probe_clear": False,
        "base_support_coverage": {"full_sample_support_candidate": True},
        "analytic_reach_candidate": True,
    }

    assert _legacy_robot_placement_is_clear(workspace, placement) is False


def test_exact_legacy_pose_requires_all_analytic_gates() -> None:
    workspace = {"status": "placement_candidate_materialized"}
    placement = {
        "status": "runtime_visualization_candidate_only",
        "mesh_triangle_aabb_overlap_probe_clear": True,
        "base_support_coverage": {"full_sample_support_candidate": True},
        "analytic_reach_candidate": True,
    }

    assert _legacy_robot_placement_is_clear(workspace, placement) is True
    placement["analytic_reach_candidate"] = False
    assert _legacy_robot_placement_is_clear(workspace, placement) is False


def _trajectory() -> dict:
    plan = {
        "schema_version": "native_rigid_construction_phase_plan.v1",
        "task_kind": "rigid_pick_place",
        "manipulation_strategy": "planar_push",
        "phase_count": 1,
        "execution_parameters": {
            "arrival_tolerance_m": 0.02,
            "arrival_orientation_tolerance_rad": 0.08,
            "maximum_steps_per_phase": 64,
        },
        "phases": [
            {
                "phase_id": "precontact",
                "position_world_m": [2.79, -6.76, 0.818],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "gripper_state": "open",
                "gate_ids": ["precontact_reachability"],
            }
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return placement_trajectory_from_native_plan(plan)


def _native_profile() -> dict:
    reset = {
        f"panda_joint{index}": value
        for index, value in enumerate(
            [0.0, -0.628318530718, 0.0, -2.513274122872, 0.0, 1.884955592154, 0.0],
            start=1,
        )
    }
    reset["finger_joint"] = 0.104255385697
    return {"robot_joint_reset_positions_rad": reset}


def test_task_aware_reset_is_base_and_trajectory_bound() -> None:
    reset, report = _derive_task_aware_franka_reset(
        profile=_native_profile(),
        base_pose={
            "position_world_m": [3.544, -6.7605, 0.752958],
            "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
        },
        task_trajectory=_trajectory(),
    )

    assert reset["finger_joint"] == pytest.approx(0.104255385697)
    assert [reset[f"panda_joint{index}"] for index in range(1, 8)] == pytest.approx(
        report["derived_arm_joint_positions_rad"]
    )
    assert report["residual_slew_rad"] < report["nominal_slew_rad"]
    assert report["source_trajectory_digest"] == _trajectory()["trajectory_digest"]
    assert report["source_phase_id"] == "precontact"
    assert report["native_full_pose_ik_required"] is True
    assert report["native_collision_and_contact_required"] is True
    assert report["native_reset_application_and_readback_required"] is True
    assert report["derivation_digest"] == canonical_digest(
        report, digest_field="derivation_digest"
    )


def test_task_aware_reset_rejects_unbound_trajectory() -> None:
    trajectory = _trajectory()
    trajectory["phases"][0]["orientation_world_xyzw"] = [0.0, 0.0, 0.0, 1.0]

    with pytest.raises(
        TaskEvaluationDiagnosticNativeArenaCompilerError,
        match="diagnostic_native_compiler_task_trajectory_invalid",
    ):
        _derive_task_aware_franka_reset(
            profile=_native_profile(),
            base_pose={
                "position_world_m": [3.544, -6.7605, 0.752958],
                "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
            },
            task_trajectory=trajectory,
        )
