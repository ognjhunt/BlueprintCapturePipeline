from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paired_target_interaction_affordance_candidate import (
    PairedTargetInteractionAffordanceError,
    materialize_paired_target_interaction_affordance_candidate,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze(path: Path, *, task_kind: str) -> Path:
    articulated = task_kind == "articulated_interaction"
    value = {
        "schema_version": "dual_task_task_freeze.v1",
        "scene_freeze_digest": "sha256:" + "a" * 64,
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "frozen_before_learned_policy_execution": True,
        "learned_policy_outcomes_accessed": False,
        "overview_camera_policy_input": False,
        "overview_camera_deterministic_scoring_input": False,
        "task_id": "task_a" if articulated else "task_b",
        "prompt": "Open the object." if articulated else "Relocate the object.",
        "task_kind": task_kind,
        "source_object": {
            "instance_id": "source",
            "semantic_label": "generic_object",
            "observed_bounds_world_m": {
                "minimum": [0.0, 0.0, 0.0],
                "maximum": [0.4, 0.4, 0.4],
            },
            "observed_pose_world": {
                "position_world_m": [0.2, 0.2, 0.2],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "support_or_attachment_id": "support",
            "collision_identity_receipt_digest": "sha256:" + "b" * 64,
            "support_receipt_digest": "sha256:" + "c" * 64,
            "franka_placement_packet_digest": "sha256:" + "d" * 64,
            "visibility_receipt_digest": "sha256:" + "e" * 64,
        },
        "removal_plan": {
            "removal_id": "removal",
            "mask_set_id": "mask",
            "source_collider_prim_path": "/Scene/object",
            "collider_deletion_id": "collider",
            "replacement_asset_id": "asset",
            "replacement_qualification_id": "qualification",
        },
        "cameras": {"external": "external", "wrist": "wrist", "overview": "overview"},
        "execution_contract": {
            "control_frequency_hz": 20,
            "maximum_steps": 400,
            "settle_window_steps": 20,
            "seeds": [31],
            "canonical_scenario_cell_id": "canonical",
            "reset_state": {"robot": "home", "object": "source"},
        },
        "deterministic_success_predicates": ["done"],
        "failure_rungs": ["never_moved"],
        "target_configuration": (
            {"kind": "joint_interval", "target_joint_ids": ["hinge"], "joint_intervals": {"hinge": [0.4, 0.8]}}
            if articulated
            else {
                "kind": "pose_volume",
                "position_bounds_world_m": {"minimum": [0.5, 0.5, 0.0], "maximum": [0.6, 0.6, 0.1]},
                "orientation_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
                "maximum_orientation_error_rad": 0.1,
                "support_id": "support",
                "release_required": True,
            }
        ),
        "articulation_graph": (
            {
                "schema_version": "adp_articulation_graph.v1",
                "links": [
                    {"link_id": "body", "is_root": True, "semantic_role": "fixed"},
                    {"link_id": "panel", "is_root": False, "semantic_role": "moving"},
                ],
                "joints": [
                    {
                        "joint_id": "hinge", "parent_link_id": "body", "child_link_id": "panel",
                        "joint_type": "revolute", "role": "target", "axis": [0, 0, 1],
                        "limits": [0, 1.0], "reset_position": 0, "reset_tolerance": 0.001,
                        "drive": {"drive_type": "force", "stiffness": 0, "damping": 2, "maximum_force": 20},
                        "dependency": None,
                    }
                ],
                "collision_pairs": [{"link_a": "body", "link_b": "panel", "collision_enabled": True}],
                "success_predicate": {"combination": "all", "joint_intervals": {"hinge": [0.4, 0.8]}},
            }
            if articulated
            else {
                "schema_version": "adp_articulation_graph.v1",
                "links": [{"link_id": "base", "is_root": True, "semantic_role": "rigid_task_body"}],
                "joints": [], "collision_pairs": [],
                "success_predicate": {"combination": "all", "joint_intervals": {}},
            }
        ),
        "mechanism_provenance": "candidate geometry",
        "task_freeze_digest": "",
    }
    value["task_freeze_digest"] = canonical_digest(value, digest_field="task_freeze_digest")
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _usd(path: Path, *, articulated: bool) -> Path:
    body = (
        '''def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
        def Xform "panel" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
        {
            def Cube "collision" (prepend apiSchemas = ["PhysicsCollisionAPI"])
            {
                double size = 1
                double3 xformOp:scale = (0.30, 0.025, 0.25)
                double3 xformOp:translate = (0.30, 0.0, 0.30)
                uniform token[] xformOpOrder = ["xformOp:scale", "xformOp:translate"]
            }
        }'''
        if articulated
        else '''def Xform "base" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
        {
            def Cube "collision" (prepend apiSchemas = ["PhysicsCollisionAPI"])
            {
                double size = 1
                double3 xformOp:scale = (0.30, 0.35, 0.02)
                uniform token[] xformOpOrder = ["xformOp:scale"]
            }
        }'''
    )
    joint = (
        '''def "joints"
        {
            def PhysicsRevoluteJoint "hinge"
            {
                uniform token physics:axis = "X"
                rel physics:body0 = </Asset/links/body>
                rel physics:body1 = </Asset/links/panel>
                point3f physics:localPos0 = (0, 0, 0.3)
                point3f physics:localPos1 = (0, 0, 0.3)
                quatf physics:localRot0 = (0.70710677, 0, -0.70710677, 0)
                quatf physics:localRot1 = (0.70710677, 0, -0.70710677, 0)
                float physics:lowerLimit = 0
                float physics:upperLimit = 60
            }
        }'''
        if articulated
        else ""
    )
    path.write_text(
        f'''#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Asset"
{{
    def Xform "links"
    {{
        {body}
    }}
    {joint}
}}
''',
        encoding="utf-8",
    )
    return path


def _registered(path: Path, freeze: Path, usd: Path, *, task_id: str) -> Path:
    frozen = json.loads(freeze.read_text())
    value = {
        "schema_version": "registered_replacement_asset.v1",
        "scene_id": "840920",
        "task_id": task_id,
        "asset_id": "asset",
        "task_freeze_digest": frozen["task_freeze_digest"],
        "output_usd": {"path": str(usd), "size_bytes": usd.stat().st_size, "sha256": _sha(usd)},
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("task_kind", "expected_method", "expected_link"),
    [
        ("articulated_interaction", "target_driven_link_far_edge_pinch", "panel"),
        ("rigid_object_manipulation", "rigid_root_thinnest_axis_pinch", "base"),
    ],
)
def test_graph_roles_select_geometry_without_object_names(
    tmp_path: Path, task_kind: str, expected_method: str, expected_link: str
) -> None:
    articulated = task_kind == "articulated_interaction"
    freeze = _freeze(tmp_path / "freeze.json", task_kind=task_kind)
    usd = _usd(tmp_path / "asset.usda", articulated=articulated)
    registered = _registered(
        tmp_path / "registered.json",
        freeze,
        usd,
        task_id="task_a" if articulated else "task_b",
    )
    result = materialize_paired_target_interaction_affordance_candidate(
        task_freeze_path=freeze,
        registered_asset_receipt_path=registered,
        robot_base_position_world_m=[0.0, -1.0, 0.0],
        output_path=tmp_path / "result.json",
    )

    assert result["selection_contract"]["method"] == expected_method
    assert result["selection_contract"]["object_label_or_task_id_geometry_shortcut_used"] is False
    assert result["candidate"]["link_id"] == expected_link
    assert result["candidate"]["contact_body_prim_paths"] == [
        f"/Asset/links/{expected_link}"
    ]
    assert result["candidate"]["measured_collision_prim_paths"] == [
        f"/Asset/links/{expected_link}/collision"
    ]
    assert result["candidate"]["pinch_span_within_stroke"] is True
    contact = result["candidate"]["contact_point_registered_stage_m"]
    # The selected point must remain on the transformed selected-link envelope;
    # a parent-frame/local-frame mixup would move it outside this simple fixture.
    assert all(abs(value) <= 1.0 for value in contact)
    assert result["native_contact_executed"] is False
    assert "native_reach_and_two_finger_contact_unproven" in result["blockers"]


def test_registered_usd_tamper_fails_closed(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path / "freeze.json", task_kind="rigid_object_manipulation")
    usd = _usd(tmp_path / "asset.usda", articulated=False)
    registered = _registered(tmp_path / "registered.json", freeze, usd, task_id="task_b")
    usd.write_text(usd.read_text() + "\n# tamper\n", encoding="utf-8")

    with pytest.raises(
        PairedTargetInteractionAffordanceError,
        match="paired_target_affordance_registered_usd_mismatch",
    ):
        materialize_paired_target_interaction_affordance_candidate(
            task_freeze_path=freeze,
            registered_asset_receipt_path=registered,
            robot_base_position_world_m=[0.0, -1.0, 0.0],
            output_path=tmp_path / "result.json",
        )
