from __future__ import annotations

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_diagnostic_native_arena_compiler import (
    TaskEvaluationDiagnosticNativeArenaCompilerError,
    _legacy_robot_placement_is_clear,
    _runtime_subject_task_spec,
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
