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


_ENTITY_PHYSICS = {
    "movable_rigid": "rigid_body",
    "articulated_fixture": "articulation",
    "movable_deformable": "deformable_volume",
    "destination_receptacle": "rigid_body",
    "support_surface": "static_collider",
    "obstacle": "static_collider",
    "robot": "robot_articulation",
}
_ENTITY_RESET = {
    "deformable_volume": "native_deformable_state",
    "rigid_body": "native_rigid_state",
    "articulation": "native_articulation_state",
    "static_collider": "immutable_scene_state",
    "robot_articulation": "native_robot_state",
}
_ENTITY_CONTACT = {
    "movable_rigid": "manipulated_rigid",
    "articulated_fixture": "manipulated_articulation",
    "movable_deformable": "manipulated_deformable",
    "destination_receptacle": "destination_volume",
    "support_surface": "supporting_surface",
    "obstacle": "collision_obstacle",
    "robot": "manipulator",
}
_ENTITY_SCORING = {
    "movable_rigid": "movable_target",
    "articulated_fixture": "articulated_target",
    "movable_deformable": "deformable_target",
    "destination_receptacle": "destination",
    "support_surface": "support_context",
    "obstacle": "collision_context",
    "robot": "robot_context",
}


def _fixed_sha(character: str) -> str:
    return "sha256:" + character * 64


def _task_entity(
    entity_id: str,
    role: str,
    *,
    pose: dict,
    runtime_asset_digest: str,
    runtime_asset_reference: str,
    binding_kind: str,
    legacy_task_object_alias: bool = False,
) -> dict:
    physics_type = _ENTITY_PHYSICS[role]
    staged = binding_kind == "usd_asset"
    registered = binding_kind == "registered_scene_geometry"
    source_digest = _fixed_sha("a")
    state_digest = _fixed_sha("b")
    return {
        "entity_id": entity_id,
        "semantic_role": "task_object" if legacy_task_object_alias else role,
        "source_observation": {
            "observation_id": f"observation:{entity_id}",
            "source_kind": (
                "runtime_embodiment"
                if role == "robot"
                else "registered_scene_geometry"
                if registered
                else "observed_dataset_entity"
            ),
            "source_reference": f"source/{entity_id}",
            "source_sha256": source_digest,
            "observed": role != "robot",
        },
        "physics_type": physics_type,
        "runtime_asset": {
            "asset_id": f"asset:{entity_id}",
            "binding_kind": binding_kind,
            "source_reference": runtime_asset_reference,
            "sha256": runtime_asset_digest,
        },
        "initial_state": {
            "pose_world": pose,
            "state_sha256": state_digest,
            "settled_state_required": True,
            "initial_penetration_allowed": False,
        },
        "reset_method": {
            "kind": _ENTITY_RESET[physics_type],
            "state_id": f"reset:{entity_id}",
            "native_readback_required": True,
            "direct_state_write_after_episode_start_allowed": False,
        },
        "contact_role": {
            "kind": _ENTITY_CONTACT[role],
            "native_contact_readback_required": True,
        },
        "scoring_role": {
            "kind": _ENTITY_SCORING[role],
            "deterministic_state_readback_required": True,
            "policy_self_grading_allowed": False,
        },
        "removal_policy": {
            "source_entity_action": "not_present" if staged else "retain",
            "gaussian_action": "not_applicable" if staged else "retain",
            "collider_action": "not_applicable" if staged else "retain",
            "receipt_sha256": _fixed_sha("c"),
        },
        "replacement_policy": {
            "action": (
                "insert_runtime_asset" if staged else "retain_registered_source"
            ),
            "replacement_required": staged,
            "receipt_sha256": _fixed_sha("d"),
        },
        "provenance": {
            "source_id": f"source:{entity_id}",
            "source_revision": "fixture-revision-1",
            "source_path": f"fixture/{entity_id}",
            "source_size_bytes": 32,
            "license_id": "fixture-license",
            "public_source_rights_id": "fixture-public-rights",
            "derived_processing_authority_id": "fixture-processing-authority",
            "provider_terms_id": "fixture-provider-terms",
            "output_rights_id": "fixture-output-rights",
            "attribution": "Hermetic fixture",
            "disclosure_class": (
                "runtime_bundled" if role == "robot" else "generated_derivative"
            ),
            "upload_permitted": role == "robot",
            "raw_redistribution_permitted": role == "robot",
            "provider_retention_permitted": False,
            "provider_training_permitted": False,
        },
        "digests": {
            "source_sha256": source_digest,
            "runtime_asset_sha256": runtime_asset_digest,
            "initial_state_sha256": state_digest,
            "configuration_sha256": _fixed_sha("e"),
        },
    }


def _deformable_request(evidence: Path) -> dict:
    request = _request(evidence, articulated=False)
    request["scene_id"] = "deformable_fixture"
    request["task_id"] = "cloth_into_basket"
    request["task_spec"] = {
        "task_kind": "deformable_transfer",
        "prompt": "Pick up the cloth, place it inside the basket, release it, and retreat.",
        "deformable_entity_id": "cloth",
        "destination_entity_id": "basket",
        "robot_entity_id": "franka",
        "destination_interior_obb": {
            "center_world_m": [1.0, 2.0, 0.5],
            "half_extents_m": [0.3, 0.2, 0.2],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "receptacle_reference_pose_world": {
            "position_m": [1.0, 2.0, 0.3],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "minimum_particle_fraction_inside": 0.75,
        "settle_window_samples": 4,
        "maximum_node_speed_mps": 0.02,
        "maximum_principal_strain": 0.25,
        "maximum_release_contact_force_n": 0.0,
        "minimum_robot_clearance_m": 0.15,
        "maximum_receptacle_translation_drift_m": 0.01,
        "maximum_receptacle_rotation_drift_rad": 0.03,
        "maximum_receptacle_linear_speed_mps": 0.01,
        "maximum_receptacle_angular_speed_radps": 0.03,
        "control_frequency_hz": 15,
        "maximum_action_steps": 20,
    }
    request["task_joint_bindings"] = []
    request["task_state_binding"] = None

    retained_scene_assets = request["assets"][:2]
    staged_entities = {
        "cloth": ("movable_deformable", _pose(0.8, 1.8, 0.7), b"cloth"),
        "basket": ("destination_receptacle", _pose(1.0, 2.0, 0.3), b"basket"),
        "wall": ("obstacle", _pose(1.5, 2.5, 0.5), b"wall"),
        "chair": ("obstacle", _pose(0.4, 2.2, 0.4), b"chair"),
    }
    entity_assets = []
    task_entities = []
    for entity_id, (role, pose, content) in staged_entities.items():
        filename = f"{entity_id}.usda"
        path = evidence / entity_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        digest = f"sha256:{sha256_file(path)}"
        entity_assets.append(
            {
                "entity_id": entity_id,
                "semantic_role": role,
                "filename": filename,
                "source": {
                    "root": "evidence",
                    "relative_path": f"{entity_id}/{filename}",
                    "size_bytes": len(content),
                    "sha256": digest,
                },
                "pose_world": pose,
            }
        )
        task_entities.append(
            _task_entity(
                entity_id,
                role,
                pose=pose,
                runtime_asset_digest=digest,
                runtime_asset_reference=f"assets/{filename}",
                binding_kind="usd_asset",
            )
        )

    task_entities.extend(
        [
            _task_entity(
                "counter",
                "support_surface",
                pose=_pose(0.0, 0.0, 0.0),
                runtime_asset_digest=_fixed_sha("6"),
                runtime_asset_reference="/Scene/counter",
                binding_kind="registered_scene_geometry",
            ),
            _task_entity(
                "franka",
                "robot",
                pose=request["robot_base_pose_world"],
                runtime_asset_digest=_fixed_sha("7"),
                runtime_asset_reference="arena/franka_panda",
                binding_kind="runtime_embodiment",
            ),
        ]
    )
    request["assets"] = [*retained_scene_assets, *entity_assets]
    request["task_entities"] = task_entities
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    return request


def _entity_migrated_legacy_request(evidence: Path, *, articulated: bool) -> dict:
    request = _request(evidence, articulated=articulated)
    target_asset = next(
        row for row in request["assets"] if row["semantic_role"] == "task_object"
    )
    target_role = "articulated_fixture" if articulated else "movable_rigid"
    target_id = "840796_refrigerator" if articulated else "840313_pick_object"
    target_asset["entity_id"] = target_id
    request["task_entities"] = [
        _task_entity(
            target_id,
            target_role,
            pose=target_asset["pose_world"],
            runtime_asset_digest=target_asset["source"]["sha256"],
            runtime_asset_reference=f"assets/{target_asset['filename']}",
            binding_kind="usd_asset",
            legacy_task_object_alias=True,
        ),
        _task_entity(
            "franka",
            "robot",
            pose=request["robot_base_pose_world"],
            runtime_asset_digest=_fixed_sha("7"),
            runtime_asset_reference="arena/franka_panda",
            binding_kind="runtime_embodiment",
        ),
    ]
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
    assert "task_entities" not in contract
    assert "task_entity_contract_digest" not in receipt
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
    assert "task_entities" not in plan
    assert plan["asset_directory"] == "assets"
    assert {row["usd_path"] for row in plan["objects"]} == {
        f"assets/{row['filename']}" for row in request["assets"]
    }


@pytest.mark.parametrize("articulated", [False, True])
def test_legacy_task_object_alias_migrates_through_entity_keyed_plan(
    tmp_path: Path,
    articulated: bool,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    request = _entity_migrated_legacy_request(evidence, articulated=articulated)
    output = tmp_path / "packet"

    materialize_native_task_arena_packet(
        request=request,
        evidence_root=evidence,
        output_dir=output,
    )

    contract = json.loads(
        (output / "native_task_runtime_contract.v1.json").read_text()
    )
    expected_role = "articulated_fixture" if articulated else "movable_rigid"
    expected_entity_id = (
        "840796_refrigerator" if articulated else "840313_pick_object"
    )
    assert contract["task_entity_role_index"][expected_role] == [expected_entity_id]
    assert not any(
        row["semantic_role"] == "task_object"
        for row in contract["task_entities"]
    )
    task_object = next(
        row for row in contract["objects"] if row.get("entity_id") == expected_entity_id
    )
    assert task_object["semantic_role"] == expected_role
    assert task_object["object_type"] == (
        "ARTICULATION" if articulated else "RIGID"
    )

    plan = json.loads(
        (output / "native_task_arena_scene_plan.v1.json").read_text()
    )
    planned = next(
        row for row in plan["objects"] if row.get("entity_id") == expected_entity_id
    )
    assert planned["prim_path"].startswith("{ENV_REGEX_NS}/task_entities/")
    if articulated:
        assert plan["articulation"]["task_joint_prim_paths"]["door_hinge"].startswith(
            planned["prim_path"] + "/"
        )


def test_deformable_multi_entity_request_crosses_packet_contract_and_plan(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    request = _deformable_request(evidence)
    output = tmp_path / "packet"

    receipt = materialize_native_task_arena_packet(
        request=request,
        evidence_root=evidence,
        output_dir=output,
    )

    contract = json.loads(
        (output / "native_task_runtime_contract.v1.json").read_text()
    )
    plan = json.loads(
        (output / "native_task_arena_scene_plan.v1.json").read_text()
    )
    expected_entity_ids = {"basket", "chair", "cloth", "counter", "franka", "wall"}
    assert {row["entity_id"] for row in contract["task_entities"]} == (
        expected_entity_ids
    )
    assert contract["task_entity_role_index"]["obstacle"] == ["chair", "wall"]
    assert receipt["task_entity_contract_digest"] == contract[
        "task_entity_contract_digest"
    ]
    assert receipt["task_entity_contract_digest"].startswith("sha256:")
    assert contract["runtime_readback_required"][
        "deformable_nodal_state_and_strain"
    ] is True
    assert contract["runtime_readback_required"]["deformable_native_contact"] is True
    assert contract["runtime_readback_required"][
        "destination_pose_and_velocity"
    ] is True

    contract_objects = {
        row["entity_id"]: row
        for row in contract["objects"]
        if "entity_id" in row
    }
    assert contract_objects["cloth"]["object_type"] == "DEFORMABLE"
    assert contract_objects["basket"]["object_type"] == "RIGID"
    assert {
        entity_id
        for entity_id, row in contract_objects.items()
        if row["semantic_role"] == "obstacle"
    } == {"chair", "wall"}
    assert "counter" not in contract_objects
    assert "franka" not in contract_objects

    assert plan["task_entities"] == contract["task_entities"]
    plan_entities = {
        row["entity_id"]: row for row in plan["objects"] if "entity_id" in row
    }
    assert set(plan_entities) == {"basket", "chair", "cloth", "wall"}
    assert len({row["prim_path"] for row in plan_entities.values()}) == 4
    assert all(
        row["prim_path"].startswith("{ENV_REGEX_NS}/task_entities/")
        for row in plan_entities.values()
    )
    assert plan_entities["cloth"]["activate_contact_sensors"] is False
    assert plan_entities["cloth"][
        "requires_native_deformable_contact_readback"
    ] is True
    assert "requires_native_deformable_contact_readback" not in plan_entities[
        "basket"
    ]
    assert {
        binding["entity_id"]
        for binding in receipt["source_bindings"]
        if "entity_id" in binding
    } == {"basket", "chair", "cloth", "wall"}


def test_multi_entity_asset_identity_mismatch_fails_by_entity_id(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    request = _deformable_request(evidence)
    cloth = next(
        row for row in request["task_entities"] if row["entity_id"] == "cloth"
    )
    cloth["runtime_asset"]["sha256"] = _fixed_sha("0")
    cloth["digests"]["runtime_asset_sha256"] = _fixed_sha("0")
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    output = tmp_path / "packet"

    with pytest.raises(NativeTaskRuntimeContractError) as exc_info:
        materialize_native_task_arena_packet(
            request=request,
            evidence_root=evidence,
            output_dir=output,
        )

    assert exc_info.value.errors == (
        "native_task_runtime_asset_digest_invalid:cloth",
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
