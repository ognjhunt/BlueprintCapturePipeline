from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.common import sha256_file
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_packet import (
    NativeTaskArenaPacketError,
    materialize_native_task_arena_packet,
)
from blueprint_pipeline.native_task_runtime_contract import (
    DROID_FRANKA_RESET_JOINT_NAMES,
)


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> dict:
    return {
        "position_world_m": [x, y, z],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _camera(role: str) -> dict:
    wrist = role == "wrist"
    return {
        "role": role,
        "policy_input": role in {"external", "wrist"},
        "scoring_input": False,
        "pose_frame": "robot_body" if wrist else "world",
        "parent_prim_path": (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
            if wrist
            else "{ENV_REGEX_NS}"
        ),
        "optical_convention": "opencv",
        "frame_from_camera_matrix": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "intrinsics": {
            "fx": 300.0,
            "fy": 300.0,
            "cx": 159.5,
            "cy": 89.5,
            "width": 320,
            "height": 180,
        },
    }


def _request(evidence: Path, *, articulated: bool) -> dict:
    articulated_asset = b'''#usda 1.0
(
    defaultPrim = "FixtureRoot"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "FixtureRoot"
{
    def Xform "cabinet" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "door" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Mesh "handle" {}
    }
    def Xform "locked" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def "joints"
    {
        def PhysicsRevoluteJoint "door_hinge"
        {
            uniform token physics:axis = "Z"
            rel physics:body0 = </FixtureRoot/cabinet>
            rel physics:body1 = </FixtureRoot/door>
            point3f physics:localPos0 = (0, 0, 0.8)
            point3f physics:localPos1 = (0, 0, 0.8)
            float physics:lowerLimit = 0
            float physics:upperLimit = 90
        }
        def PhysicsRevoluteJoint "locked_hinge"
        {
            uniform token physics:axis = "Z"
            rel physics:body0 = </FixtureRoot/cabinet>
            rel physics:body1 = </FixtureRoot/locked>
            point3f physics:localPos0 = (0, 0, 0.2)
            point3f physics:localPos1 = (0, 0, 0.2)
            float physics:lowerLimit = 0
            float physics:upperLimit = 90
        }
    }
}
'''
    files = {
        "scene_collision": ("scene_collision.usda", b"collision"),
        "scene_appearance": ("scene_appearance.usdc", b"appearance"),
        "task_object": (
            "task_object.usda",
            articulated_asset if articulated else b"rigid",
        ),
    }
    assets = []
    for role, (filename, content) in files.items():
        path = evidence / role / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        assets.append(
            {
                "semantic_role": role,
                "filename": filename,
                "source": {
                    "root": "evidence",
                    "relative_path": f"{role}/{filename}",
                    "size_bytes": len(content),
                    "sha256": f"sha256:{sha256_file(path)}",
                },
                "pose_world": _pose(1.0, 2.0, 0.0)
                if role == "task_object"
                else _pose(),
            }
        )
    task_spec = {
        "schema_version": "adp_task_spec.v1",
        "task_kind": "articulated_open_close",
        "prompt": "Open the door.",
        "target_joint_id": "door_hinge",
        "joint_reset_positions_rad": {"door_hinge": 0.0, "locked_hinge": 0.0},
        "target_success_interval_rad": [0.7, 1.0],
        "joint_hard_limits_rad": {
            "door_hinge": [0.0, 1.5],
            "locked_hinge": [0.0, 1.5],
        },
        "scripted_positive_target_rad": 0.8,
        "settle_window_samples": 4,
        "control_frequency_hz": 15,
        "maximum_action_steps": 20,
    }
    bindings = [
        {
            "joint_id": "door_hinge",
            "joint_prim_path": "/FixtureRoot/joints/door_hinge",
            "native_joint_name": "door_hinge",
            "role": "task_joint",
        },
        {
            "joint_id": "locked_hinge",
            "joint_prim_path": "/FixtureRoot/joints/locked_hinge",
            "native_joint_name": "locked_hinge",
            "role": "locked_joint",
        },
    ]
    state = {
        "moving_link_prim_path": "/FixtureRoot/door",
        "moving_link_native_body_name": "door",
        "handle_prim_paths": ["/FixtureRoot/door/handle"],
        "handle_grasp_point_link_m": [0.1, 0.2, 0.3],
        "robot_gripper_contact_prim_pattern": "{ENV_REGEX_NS}/Robot/gripper/.*",
        "robot_collision_prim_pattern": "{ENV_REGEX_NS}/Robot/.*",
        "task_contact_minimum_force_n": 0.5,
        "collision_failure_minimum_force_n": 1.0,
        "retreat_minimum_separation_m": 0.1,
        "root_translation_tolerance_m": 0.002,
        "root_orientation_tolerance_rad": 0.01,
    }
    if not articulated:
        task_spec = {
            "task_kind": "rigid_pick_place",
            "prompt": "Pick and place the object.",
            "control_frequency_hz": 15,
            "maximum_action_steps": 20,
            "settle_window_samples": 4,
        }
        bindings = []
        state = None
    context_kind = "construction_canary" if articulated else "evaluation_cell"
    context_digest_field = (
        "context_digest" if context_kind == "construction_canary" else "instance_digest"
    )
    context_document = {
        "schema_version": (
            "native_task_construction_canary.v1"
            if context_kind == "construction_canary"
            else "adp009d_scenario_instance.v1"
        ),
        "program_id": "arm-decision-proof-v1",
        "cell_id": "canonical.seed_17",
        "seed": 17,
        "policy_neutral": True,
        "caller_asserted_success": False,
        "learned_policy_outcomes_consulted": False,
        context_digest_field: "",
    }
    context_document[context_digest_field] = canonical_digest(
        context_document, digest_field=context_digest_field
    )
    request = {
        "schema_version": "native_task_arena_packet_request.v1",
        "scene_id": "840796" if articulated else "840313",
        "task_id": "articulated_fixture" if articulated else "rigid_fixture",
        "task_spec": task_spec,
        "task_joint_bindings": bindings,
        "task_state_binding": state,
        "assets": assets,
        "robot_base_pose_world": _pose(1.75, 1.99, 0.0),
        "robot_joint_reset_positions_rad": {
            name: float(index) / 100.0
            for index, name in enumerate(DROID_FRANKA_RESET_JOINT_NAMES)
        },
        "cameras": [_camera(role) for role in ("external", "wrist", "overview")],
        "scenario": {
            "context_kind": context_kind,
            "cell_id": "canonical.seed_17",
            "instance_digest": context_document[context_digest_field],
            "seed": 17,
            "context_document": context_document,
        },
        "physics_frequency_hz": 120,
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    return request


@pytest.mark.parametrize("articulated", [False, True])
def test_original_and_second_scene_share_one_packet_materializer(
    tmp_path: Path, articulated: bool
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    request = _request(evidence, articulated=articulated)
    output = tmp_path / "packet"

    receipt = materialize_native_task_arena_packet(
        request=request, evidence_root=evidence, output_dir=output
    )

    assert receipt["status"] == "construction_packet_completed"
    assert receipt["native_application_claimed"] is False
    assert receipt["request_digest"] == request["request_digest"]
    assert len(receipt["source_bindings"]) == 3
    persisted = json.loads(
        (output / "native_task_arena_packet_receipt.v1.json").read_text()
    )
    assert persisted == receipt
    contract = json.loads(
        (output / "native_task_runtime_contract.v1.json").read_text()
    )
    assert contract["scenario"]["context_kind"] == (
        "construction_canary" if articulated else "evaluation_cell"
    )
    task_object = next(
        row for row in contract["objects"] if row["semantic_role"] == "task_object"
    )
    assert task_object["object_type"] == (
        "ARTICULATION" if articulated else "RIGID"
    )
    plan = json.loads((output / "native_task_arena_scene_plan.v1.json").read_text())
    assert plan["asset_directory"] == "assets"
    assert {row["usd_path"] for row in plan["objects"]} == {
        f"assets/{row['filename']}" for row in request["assets"]
    }


def test_tampered_source_fails_and_removes_partial_packet(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    request = _request(evidence, articulated=True)
    (evidence / "task_object" / "task_object.usda").write_bytes(b"tampered")
    output = tmp_path / "packet"

    with pytest.raises(NativeTaskArenaPacketError) as excinfo:
        materialize_native_task_arena_packet(
            request=request, evidence_root=evidence, output_dir=output
        )

    assert excinfo.value.errors == (
        "native_task_arena_packet_asset_identity_mismatch:task_object",
    )
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    request = _request(evidence, articulated=True)
    output = tmp_path / "packet"
    output.mkdir()
    sentinel = output / "user-owned"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(NativeTaskArenaPacketError, match="output_exists"):
        materialize_native_task_arena_packet(
            request=request, evidence_root=evidence, output_dir=output
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_forged_construction_context_fails_before_copy(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    request = _request(evidence, articulated=True)
    request["scenario"]["context_document"]["seed"] += 1
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    output = tmp_path / "packet"

    with pytest.raises(NativeTaskArenaPacketError) as excinfo:
        materialize_native_task_arena_packet(
            request=request, evidence_root=evidence, output_dir=output
        )

    assert excinfo.value.errors == (
        "native_task_arena_packet_scenario_binding_mismatch",
        "native_task_arena_packet_scenario_digest_invalid",
    )
    assert not output.exists()


def test_checked_second_scene_request_and_receipt_are_self_consistent() -> None:
    manifest_root = (
        Path(__file__).parents[1] / "docs/arm_decision_proof_v1/manifests"
    )
    request = json.loads(
        (
            manifest_root
            / "second_scene_840796_native_arena_packet_request.v1.json"
        ).read_text()
    )
    receipt = json.loads(
        (
            manifest_root
            / "second_scene_840796_native_arena_packet_receipt.v1.json"
        ).read_text()
    )
    orientation = json.loads(
        (
            manifest_root
            / "second_scene_840796_franka_base_orientation.v1.json"
        ).read_text()
    )

    assert request["request_digest"] == canonical_digest(
        request, digest_field="request_digest"
    )
    assert request["scenario"]["context_kind"] == "construction_canary"
    assert request["robot_base_pose_world"]["orientation_xyzw"] == orientation[
        "resolved_orientation_xyzw"
    ]
    assert receipt["request_digest"] == request["request_digest"]
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    assert receipt["native_application_claimed"] is False
