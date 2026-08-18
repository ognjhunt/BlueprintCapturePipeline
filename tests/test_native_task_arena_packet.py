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
    NativeTaskRuntimeContractError,
)
from blueprint_pipeline.replacement_construction_bindings import (
    GAUSSIAN_REMOVAL_QUALIFICATION_SCHEMA_VERSION,
    MASK_SET_QUALIFICATION_SCHEMA_VERSION,
    REPLACEMENT_QUALIFICATION_SCHEMA_VERSION,
    SOURCE_COLLIDER_DELETION_SCHEMA_VERSION,
    seal_replacement_construction_bindings,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _evidence_record(name: str, digest: str, schema_version: str) -> dict:
    return {
        "path": f"/fixture/{name}.json",
        "size_bytes": 1,
        "sha256": _sha("f"),
        "schema_version": schema_version,
        "canonical_digest": digest,
    }


def _construction_row_with_evidence(row: dict) -> dict:
    row = json.loads(json.dumps(row))
    row["evidence_receipts"] = {
        "task_freeze": _evidence_record(
            "task_freeze", row["task_freeze_digest"], "dual_task_task_freeze.v1"
        ),
        "mask_set": _evidence_record(
            "mask_set",
            row["mask_set_receipt_digest"],
            MASK_SET_QUALIFICATION_SCHEMA_VERSION,
        ),
        "gaussian_removal": _evidence_record(
            "gaussian_removal",
            row["source_removal_receipt_digest"],
            GAUSSIAN_REMOVAL_QUALIFICATION_SCHEMA_VERSION,
        ),
        "source_collider_deletion": {
            "selected_deletion_id": row["collider_deletion_id"],
            "independent": _evidence_record(
                "source_collider_deletion",
                row["collider_deletion_receipt_digest"],
                SOURCE_COLLIDER_DELETION_SCHEMA_VERSION,
            ),
        },
        "replacement_qualification": _evidence_record(
            "replacement_qualification",
            row["replacement_qualification_receipt_digest"],
            REPLACEMENT_QUALIFICATION_SCHEMA_VERSION,
        ),
    }
    return row


def _materialized_construction(value: dict) -> dict:
    result = json.loads(json.dumps(value))
    result["scene_freeze_receipt"] = _evidence_record(
        "scene_freeze",
        result["scene_freeze_digest"],
        "dual_task_scene_freeze.v1",
    )
    result["construction_digest"] = canonical_digest(
        result,
        digest_field="construction_digest",
    )
    return result


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
    def Xform "cabinet" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Mesh "body" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {
            uniform token physics:approximation = "convexHull"
            point3f[] points = [(-0.3, -0.2, 0), (0.3, 0.2, 1.6)]
        }
    }
    def Xform "door" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Mesh "handle" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {
            uniform token physics:approximation = "convexHull"
            point3f[] points = [(0, 0, 0), (0.1, 0.2, 0.3)]
        }
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
    collision_asset = b'''#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Root"
{
    def Mesh "floor" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
}
'''
    files = {
        "scene_collision": (
            "scene_collision.usda",
            collision_asset if articulated else b"#usda 1.0\n# collision\n",
        ),
        # Placeholder bytes are not accepted for a staged USD payload: the
        # packet now refuses a file whose magic is not the format its filename
        # declares, so these fixtures carry the real leading bytes.
        "scene_appearance": ("scene_appearance.usdc", b"PXR-USDC\x00appearance"),
        "task_object": (
            "task_object.usda",
            articulated_asset if articulated else b"#usda 1.0\n# rigid\n",
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


def test_two_task_packets_preserve_one_shared_repeatable_replacement_set(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    articulated_request = _request(evidence, articulated=True)
    task_asset = articulated_request["assets"][-1]
    task_asset.update(
        {
            "semantic_role": "replacement",
            "asset_id": "articulated_a",
            "object_type": "ARTICULATION",
            "reset_state": {
                "joint_positions": {"door_hinge": 0.0, "locked_hinge": 0.0}
            },
        }
    )
    rigid_path = evidence / "rigid_b" / "rigid_b.usda"
    rigid_path.parent.mkdir()
    rigid_path.write_bytes(b"#usda 1.0\n# rigid-b\n")
    articulated_request["assets"].append(
        {
            "semantic_role": "replacement",
            "asset_id": "rigid_b",
            "object_type": "RIGID",
            "filename": rigid_path.name,
            "source": {
                "root": "evidence",
                "relative_path": "rigid_b/rigid_b.usda",
                "size_bytes": rigid_path.stat().st_size,
                "sha256": f"sha256:{sha256_file(rigid_path)}",
            },
            "pose_world": _pose(2.0, 3.0, 0.8),
            "reset_state": {"joint_positions": {}},
        }
    )
    articulated_request["task_spec"]["subject_asset_id"] = "articulated_a"
    articulated_freeze_digest = "sha256:" + "1" * 64
    rigid_freeze_digest = "sha256:" + "2" * 64
    shared_construction = _materialized_construction(
        seal_replacement_construction_bindings(
            scene_freeze_digest="sha256:" + "3" * 64,
            task_freeze_join_digest="sha256:" + "4" * 64,
            bindings=[
                _construction_row_with_evidence(
                    {
                        "task_id": "articulated_fixture",
                        "asset_id": "articulated_a",
                        "task_freeze_digest": articulated_freeze_digest,
                        "source_object_instance_id": "source_a",
                        "removal_id": "removal_a",
                        "mask_set_id": "masks_a",
                        "mask_set_receipt_digest": "sha256:" + "5" * 64,
                        "source_removal_receipt_digest": "sha256:" + "6" * 64,
                        "source_removal_qualified": True,
                        "collider_deletion_id": "collider_a",
                        "source_collider_prim_path": "/Root/source_a",
                        "collider_deletion_receipt_digest": "sha256:" + "7" * 64,
                        "collider_deletion_qualified": True,
                        "replacement_qualification_id": "qualification_a",
                        "replacement_qualification_receipt_digest": "sha256:"
                        + "8" * 64,
                        "replacement_asset_sha256": task_asset["source"]["sha256"],
                        "replacement_simulator_import_qualified": True,
                    }
                ),
                _construction_row_with_evidence(
                    {
                        "task_id": "rigid_task_b",
                        "asset_id": "rigid_b",
                        "task_freeze_digest": rigid_freeze_digest,
                        "source_object_instance_id": "source_b",
                        "removal_id": "removal_b",
                        "mask_set_id": "masks_b",
                        "mask_set_receipt_digest": "sha256:" + "9" * 64,
                        "source_removal_receipt_digest": "sha256:" + "a" * 64,
                        "source_removal_qualified": True,
                        "collider_deletion_id": "collider_b",
                        "source_collider_prim_path": "/Root/source_b",
                        "collider_deletion_receipt_digest": "sha256:" + "b" * 64,
                        "collider_deletion_qualified": True,
                        "replacement_qualification_id": "qualification_b",
                        "replacement_qualification_receipt_digest": "sha256:"
                        + "c" * 64,
                        "replacement_asset_sha256": articulated_request["assets"][
                            -1
                        ]["source"]["sha256"],
                        "replacement_simulator_import_qualified": True,
                    }
                ),
            ],
        )
    )
    articulated_request["construction_bindings"] = shared_construction
    articulated_request["task_freeze_digest"] = articulated_freeze_digest
    articulated_request["request_digest"] = canonical_digest(
        articulated_request, digest_field="request_digest"
    )
    rigid_request = json.loads(json.dumps(articulated_request))
    rigid_request.update(
        task_id="rigid_task_b",
        task_spec={
            "task_kind": "rigid_pick_place",
            "prompt": "Relocate the rigid subject.",
            "subject_asset_id": "rigid_b",
            "control_frequency_hz": 15,
            "maximum_action_steps": 20,
            "settle_window_samples": 4,
        },
        task_joint_bindings=[],
        task_state_binding=None,
        task_freeze_digest=rigid_freeze_digest,
    )
    rigid_request["request_digest"] = canonical_digest(
        rigid_request, digest_field="request_digest"
    )

    receipts = []
    contracts = []
    for name, request in (("task_a", articulated_request), ("task_b", rigid_request)):
        output = tmp_path / name
        receipts.append(
            materialize_native_task_arena_packet(
                request=request,
                evidence_root=evidence,
                output_dir=output,
            )
        )
        contracts.append(
            json.loads((output / "native_task_runtime_contract.v1.json").read_text())
        )

    assert all(len(receipt["source_bindings"]) == 4 for receipt in receipts)
    assert [contract["task_subject_asset_id"] for contract in contracts] == [
        "articulated_a",
        "rigid_b",
    ]
    assert {receipt["shared_construction_digest"] for receipt in receipts} == {
        shared_construction["construction_digest"]
    }
    assert [receipt["task_freeze_digest"] for receipt in receipts] == [
        articulated_freeze_digest,
        rigid_freeze_digest,
    ]
    assert {
        row["asset_id"]
        for row in contracts[0]["objects"]
        if row["source_semantic_role"] == "replacement"
    } == {
        row["asset_id"]
        for row in contracts[1]["objects"]
        if row["source_semantic_role"] == "replacement"
    } == {"articulated_a", "rigid_b"}


def test_resolved_scenario_parameter_reaches_native_scene_plan(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    request = _request(evidence, articulated=False)
    context = request["scenario"]["context_document"]
    context["resolved_parameters"] = {"external_camera_extrinsic_dx_m": 0.02}
    context["factor_records"] = [
        {
            "parameter_id": "external_camera_extrinsic_dx_m",
            "runtime_target": "EventManager.reset.external_camera.pose.position.x",
            "unit": "m",
            "nominal_value": 0.0,
            "resolved_value": 0.02,
            "application_tolerance": 1.0e-4,
        }
    ]
    context["instance_digest"] = canonical_digest(
        context, digest_field="instance_digest"
    )
    request["scenario"]["instance_digest"] = context["instance_digest"]
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    output = tmp_path / "packet"

    materialize_native_task_arena_packet(
        request=request, evidence_root=evidence, output_dir=output
    )

    contract = json.loads(
        (output / "native_task_runtime_contract.v1.json").read_text()
    )
    plan = json.loads((output / "native_task_arena_scene_plan.v1.json").read_text())
    assert contract["scenario"]["parameter_bindings"][0]["resolved_value"] == 0.02
    application = plan["scenario"]["parameter_applications"][0]
    assert application["expected_native_value"] == pytest.approx(0.02)
    external = next(camera for camera in plan["cameras"] if camera["role"] == "external")
    assert external["frame_from_camera_matrix"][3] == pytest.approx(0.02)


def test_unsupported_scenario_target_fails_before_packet_copy_is_admitted(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    request = _request(evidence, articulated=False)
    context = request["scenario"]["context_document"]
    context["resolved_parameters"] = {"object_mass_kg": 0.4}
    context["factor_records"] = [
        {
            "parameter_id": "object_mass_kg",
            "runtime_target": "EventManager.reset.object_rigid_body.mass_kg",
            "unit": "kg",
            "nominal_value": 0.355,
            "resolved_value": 0.4,
            "application_tolerance": 1.0e-4,
        }
    ]
    context["instance_digest"] = canonical_digest(
        context, digest_field="instance_digest"
    )
    request["scenario"]["instance_digest"] = context["instance_digest"]
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    output = tmp_path / "packet"

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_arena_packet(
            request=request, evidence_root=evidence, output_dir=output
        )

    assert excinfo.value.errors == (
        "native_task_runtime_scenario_target_unsupported:"
        "EventManager.reset.object_rigid_body.mass_kg",
    )
    assert not output.exists()


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


def test_a_json_receipt_named_as_a_usd_asset_fails_before_packet_copy(
    tmp_path: Path,
) -> None:
    """The packet refuses bytes that are not the format their name declares.

    Digesting the copy proves only that the packet staged what it was handed.
    A receipt supplied in place of the exported asset digests perfectly and
    then fails inside ``UsdStage::Open`` on a rented GPU, which is exactly how
    the 2026-08-18 arena construction attempt was spent.
    """

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    request = _request(evidence, articulated=True)
    receipt = evidence / "scene_appearance" / "scene_appearance.usdc"
    receipt.write_text('{"schema_version": "appearance_export.v1"}\n', encoding="utf-8")
    for asset in request["assets"]:
        if asset["semantic_role"] == "scene_appearance":
            asset["source"]["size_bytes"] = receipt.stat().st_size
            asset["source"]["sha256"] = f"sha256:{sha256_file(receipt)}"
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    output = tmp_path / "packet"

    with pytest.raises(NativeTaskArenaPacketError) as excinfo:
        materialize_native_task_arena_packet(
            request=request, evidence_root=evidence, output_dir=output
        )

    # the identity checks all pass -- only the format check catches this
    assert excinfo.value.errors == (
        "native_task_arena_packet_asset_format_invalid:scene_appearance",
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


def test_checked_second_scene_packet_v3_binding_is_canonical() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs/arm_decision_proof_v1/manifests"
        / "second_scene_840796_native_arena_packet_v3_binding.v1.json"
    )
    binding = json.loads(path.read_text(encoding="utf-8"))

    assert binding["binding_digest"] == canonical_digest(
        binding, digest_field="binding_digest"
    )
    assert binding["native_application_claimed"] is False
    assert binding["motion_geometry"]["derived_from_exact_task_usd"] is True


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
