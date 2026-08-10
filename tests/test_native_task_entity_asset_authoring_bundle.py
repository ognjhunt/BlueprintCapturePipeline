from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_entity_asset_authoring_bundle import (
    BUNDLE_FILENAME,
    DEFORMABLE_AUTHORING_API,
    DEFORMABLE_COOKING_API,
    DEFORMABLE_EXPECTED_PRIM_TYPE,
    DEFORMABLE_REQUIRED_SCHEMAS,
    DEFORMABLE_RUNTIME_CLASS,
    PENDING_NATIVE_GAPS,
    RECEIPT_FILENAME,
    RIGID_AUTHORING_API,
    RIGID_EXPECTED_PRIM_TYPE,
    RIGID_RUNTIME_CLASS,
    RUNTIME_IDENTITY_SCHEMA_VERSION,
    NativeTaskEntityAssetAuthoringBundleError,
    build_native_task_entity_asset_authoring_bundle,
    materialize_native_asset_authoring_runtime_identity,
    verify_native_task_entity_asset_authoring_bundle,
)
from blueprint_pipeline.native_task_entity_contract import (
    TASK_KIND_DEFORMABLE_TRANSFER,
    materialize_native_task_entity_contract,
)
from blueprint_pipeline.native_task_runtime_source_packet import (
    ARENA_COMMIT,
    ARENA_REPOSITORY,
    ARENA_TREE,
    ISAACLAB_COMMIT,
    ISAACLAB_REPOSITORY,
    ISAACLAB_TREE,
)
from blueprint_pipeline.task_entity_asset_candidate import (
    RIGID_COLLISION_REPRESENTATIONS,
    materialize_task_entity_asset_candidate,
)


def _sha_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _files(root: Path, roles: tuple[str, ...]) -> list[dict]:
    rows = []
    for index, role in enumerate(roles):
        content = f"fixture-{role}-{index}\n".encode()
        path = root / "assets" / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        rows.append(
            {
                "role": role,
                "path": f"assets/{role}.bin",
                "sha256": _sha_bytes(content),
                "size_bytes": len(content),
            }
        )
    return rows


def _candidate(tmp_path: Path, asset_class: str) -> tuple[dict, Path]:
    deformable = asset_class == "deformable_volume"
    entity_id = "cloth" if deformable else "basket"
    root = tmp_path / f"candidate-{entity_id}"
    roles = (
        (
            "rest_geometry",
            "material_definition",
            "texture",
            "physics_configuration",
            "runtime_usd",
        )
        if deformable
        else (
            "visual_geometry",
            "collision_geometry",
            "material_definition",
            "texture",
            "physics_configuration",
            "runtime_usd",
        )
    )
    files = _files(root, roles)
    source_digest = _sha("a" if deformable else "b")
    raw = {
        "schema_version": "task_entity_asset_candidate.v1",
        "entity_id": entity_id,
        "asset_id": f"asset:{entity_id}",
        "asset_class": asset_class,
        "source_observation": {
            "observation_id": f"observation:{entity_id}",
            "source_reference": f"InteriorGS/SAGE:fixture:{entity_id}",
            "source_sha256": source_digest,
            "source_size_bytes": 1024,
            "bounds_world": {
                "minimum_m": [0.0, 0.0, 1.0],
                "maximum_m": [0.3, 0.2, 1.1],
            },
            "metric_dimensions_m": [0.3, 0.2, 0.1],
            "coverage": {
                "metric_bounds_observed": True,
                "rest_state_bounded": True,
                "full_surface_observed": False,
                "interior_collision_observed": not deformable,
                "interior_appearance_observed": False,
                "engineered_interior_not_factual": not deformable,
                "unobserved_regions": ["underside"],
            },
        },
        "rights": {
            "source_revision": "fixture-source-revision",
            "license_id": "fixture-license",
            "license_reference": "https://example.invalid/license",
            "license_sha256": _sha("c"),
            "attribution": "Hermetic fixture",
            "derived_processing_authority_id": "fixture-authority",
            "provider_terms_id": "fixture-provider-terms",
            "output_rights_id": "fixture-output-rights",
            "raw_source_private_upload_permitted": False,
            "derived_asset_private_upload_permitted": True,
            "raw_redistribution_permitted": False,
            "provider_retention_permitted": False,
            "provider_training_permitted": False,
        },
        "authoring": {
            "method": "released_code_parametric",
            "source_repository": "https://example.invalid/released-authoring",
            "source_revision": "d" * 40,
            "source_tree": "e" * 40,
            "package_name": "fixture-authoring",
            "package_version": "1.0.0",
            "generated_geometry_used": True,
            "generated_physics_used": False,
        },
        "files": files,
        "transform": {
            "authored_origin_m": [0.0, 0.0, 0.0],
            "pivot_m": [0.0, 0.0, 0.0],
            "scale_xyz": [1.0, 1.0, 1.0],
            "world_pose": {
                "position_world_m": [0.8, 1.8, 1.1],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "meters_per_unit": 1.0,
            "up_axis": "Z",
        },
        "simulator_import": {
            "simulator": "Isaac Sim",
            "simulator_version": "6.0.1",
            "source_repository": ISAACLAB_REPOSITORY,
            "source_revision": ISAACLAB_COMMIT,
            "importer_module": (DEFORMABLE_RUNTIME_CLASS if deformable else RIGID_RUNTIME_CLASS),
            "expected_prim_type": (
                DEFORMABLE_EXPECTED_PRIM_TYPE if deformable else RIGID_EXPECTED_PRIM_TYPE
            ),
        },
        "retained_diagnostic_requirements": [
            "native_import",
            "stable_support_and_no_initial_penetration",
            "native_contact",
            "native_reset_readback",
            "native_render_coverage",
            *(
                [
                    "native_deformable_settling",
                    "native_strain_and_solver_stability",
                ]
                if deformable
                else []
            ),
        ],
    }
    if deformable:
        raw["deformable_configuration"] = {
            "representation": "closed_tetrahedral_volumetric_fem",
            "rest_topology": {
                "vertex_count": 100,
                "tetrahedron_count": 240,
                "closed_volume": True,
                "manifold_surface": True,
                "topology_sha256": _sha("f"),
            },
            "material": {
                "mass_kg": 0.25,
                "volume_density_kg_m3": 250.0,
                "effective_thickness_m": 0.02,
                "youngs_modulus_pa": 50_000.0,
                "poissons_ratio": 0.35,
                "elasticity_damping": 0.01,
                "velocity_damping": 0.1,
                "dynamic_friction": 0.6,
                "independent_bend_parameter_available": False,
                "independent_shear_parameter_available": False,
                "thin_shell_cloth_claimed": False,
            },
            "solver": {
                "mesh_resolution": 16,
                "particle_or_vertex_spacing_m": 0.01,
                "position_iterations": 16,
                "substeps": 4,
                "maximum_admitted_principal_strain": 0.5,
            },
            "collision": {
                "self_collision_enabled": True,
                "contact_offset_m": 0.005,
                "rest_offset_m": 0.002,
                "requested_grasp_contact_representation": ("native_finger_to_deformable_collision"),
                "hidden_kinematic_attachment_allowed": False,
            },
            "reset": {
                "reset_kind": "native_default_nodal_state",
                "write_default_nodal_state_before_episode": True,
                "zero_nodal_velocities": True,
                "free_kinematic_flag_value": 1.0,
                "native_readback_required": True,
                "direct_state_write_after_episode_start_allowed": False,
            },
        }
    else:
        raw["receptacle_configuration"] = {
            "geometry": {
                "open_interior": True,
                "top_cap_present": False,
                "interior_dimensions_m": [0.25, 0.16, 0.08],
                "wall_thickness_m": 0.01,
                "floor_thickness_m": 0.01,
                "engineered_interior": True,
            },
            "collision": {
                "representation": "multi_part_convex_open_receptacle",
                "collision_sha256": _sha("1"),
                "contact_offset_m": 0.002,
                "rest_offset_m": 0.0,
            },
            "material": {
                "static_friction": 0.6,
                "dynamic_friction": 0.5,
                "restitution": 0.0,
                "material_provenance_sha256": _sha("2"),
            },
            "anchoring": {
                "static_anchored": True,
                "mass_kg": 0.0,
                "inertia_diagonal_kg_m2": [0.0, 0.0, 0.0],
                "stable_support_readback_required": True,
                "native_collision_readback_required": True,
            },
        }
    return materialize_task_entity_asset_candidate(raw), root


_PHYSICS = {
    "movable_deformable": "deformable_volume",
    "destination_receptacle": "static_collider",
    "support_surface": "static_collider",
    "obstacle": "static_collider",
    "robot": "robot_articulation",
}
_RESET = {
    "deformable_volume": "native_deformable_state",
    "static_collider": "immutable_scene_state",
    "robot_articulation": "native_robot_state",
}
_CONTACT = {
    "movable_deformable": "manipulated_deformable",
    "destination_receptacle": "destination_volume",
    "support_surface": "supporting_surface",
    "obstacle": "collision_obstacle",
    "robot": "manipulator",
}
_SCORING = {
    "movable_deformable": "deformable_target",
    "destination_receptacle": "destination",
    "support_surface": "support_context",
    "obstacle": "collision_context",
    "robot": "robot_context",
}


def _entity(
    entity_id: str,
    role: str,
    *,
    candidate: dict | None = None,
) -> dict:
    physics = _PHYSICS[role]
    source_digest = candidate["source_observation"]["source_sha256"] if candidate else _sha("3")
    runtime_usd = (
        next(row for row in candidate["files"] if row["role"] == "runtime_usd")
        if candidate
        else None
    )
    runtime_digest = runtime_usd["sha256"] if runtime_usd else _sha("4")
    configuration_digest = (
        canonical_digest(
            candidate[
                "deformable_configuration"
                if role == "movable_deformable"
                else "receptacle_configuration"
            ]
        )
        if candidate
        else _sha("5")
    )
    inserted = candidate is not None
    robot = role == "robot"
    return {
        "entity_id": entity_id,
        "semantic_role": role,
        "source_observation": {
            "observation_id": f"observation:{entity_id}",
            "source_kind": (
                "runtime_embodiment"
                if robot
                else "observed_dataset_entity"
                if inserted
                else "registered_scene_geometry"
            ),
            "source_reference": (
                candidate["source_observation"]["source_reference"]
                if candidate
                else f"sources/{entity_id}"
            ),
            "source_sha256": source_digest,
            "observed": not robot,
        },
        "physics_type": physics,
        "runtime_asset": {
            "asset_id": candidate["asset_id"] if candidate else f"asset:{entity_id}",
            "binding_kind": (
                "runtime_embodiment"
                if robot
                else "usd_asset"
                if inserted
                else "registered_scene_geometry"
            ),
            "source_reference": f"assets/{entity_id}.usd",
            "sha256": runtime_digest,
        },
        "initial_state": {
            "pose_world": {
                "position_world_m": [0.8, 1.8, 1.1],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "state_sha256": _sha("6"),
            "settled_state_required": True,
            "initial_penetration_allowed": False,
        },
        "reset_method": {
            "kind": _RESET[physics],
            "state_id": f"reset:{entity_id}",
            "native_readback_required": True,
            "direct_state_write_after_episode_start_allowed": False,
        },
        "contact_role": {
            "kind": _CONTACT[role],
            "native_contact_readback_required": True,
        },
        "scoring_role": {
            "kind": _SCORING[role],
            "deterministic_state_readback_required": True,
            "policy_self_grading_allowed": False,
        },
        "removal_policy": {
            "source_entity_action": "not_present" if inserted else "retain",
            "gaussian_action": "not_applicable" if inserted else "retain",
            "collider_action": "not_applicable" if inserted else "retain",
            "receipt_sha256": _sha("7"),
        },
        "replacement_policy": {
            "action": "insert_runtime_asset" if inserted else "retain_registered_source",
            "replacement_required": inserted,
            "receipt_sha256": _sha("8"),
        },
        "provenance": {
            "source_id": f"source:{entity_id}",
            "source_revision": "fixture-source-revision",
            "source_path": f"fixture/{entity_id}",
            "source_size_bytes": 1024,
            "license_id": "fixture-license",
            "public_source_rights_id": "fixture-public-rights",
            "derived_processing_authority_id": "fixture-authority",
            "provider_terms_id": "fixture-provider-terms",
            "output_rights_id": "fixture-output-rights",
            "attribution": "Hermetic fixture",
            "disclosure_class": ("runtime_bundled" if robot else "restricted_private_processing"),
            "upload_permitted": inserted or robot,
            "raw_redistribution_permitted": robot,
            "provider_retention_permitted": False,
            "provider_training_permitted": False,
        },
        "digests": {
            "source_sha256": source_digest,
            "runtime_asset_sha256": runtime_digest,
            "initial_state_sha256": _sha("6"),
            "configuration_sha256": configuration_digest,
        },
    }


def _contract(cloth: dict, basket: dict) -> dict:
    return materialize_native_task_entity_contract(
        task_kind=TASK_KIND_DEFORMABLE_TRANSFER,
        task_entities=[
            _entity("cloth", "movable_deformable", candidate=cloth),
            _entity("basket", "destination_receptacle", candidate=basket),
            _entity("counter", "support_surface"),
            _entity("wall", "obstacle"),
            _entity("franka", "robot"),
        ],
    )


def _runtime_identity() -> dict:
    return materialize_native_asset_authoring_runtime_identity(
        {
            "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
            "runtime_id": "fixture-isaac-native-runtime",
            "simulator": {
                "name": "Isaac Sim",
                "version": "6.0.1",
                "install_root": "/isaac-sim",
                "container_image": ("fixture.invalid/isaac-sim@sha256:" + "9" * 64),
            },
            "runtime_sources": {
                "isaac_lab": {
                    "repository": ISAACLAB_REPOSITORY,
                    "revision": ISAACLAB_COMMIT,
                    "tree": ISAACLAB_TREE,
                },
                "arena": {
                    "repository": ARENA_REPOSITORY,
                    "revision": ARENA_COMMIT,
                    "tree": ARENA_TREE,
                },
                "source_packet_receipt_digest": _sha("a"),
            },
            "python": {
                "python_tag": "cp312",
                "abi_tag": "cp312",
                "platform_tag": "manylinux_2_28_x86_64",
            },
            "selected_robot": {
                "robot_id": "franka_panda",
                "module": "isaaclab_arena.embodiments.droid.droid",
            },
            "bindings": {
                "deformable_volume": {
                    "representation": "closed_tetrahedral_volumetric_fem",
                    "authoring_api": DEFORMABLE_AUTHORING_API,
                    "cooking_api": DEFORMABLE_COOKING_API,
                    "runtime_class": DEFORMABLE_RUNTIME_CLASS,
                    "expected_prim_type": DEFORMABLE_EXPECTED_PRIM_TYPE,
                    "required_schemas": list(DEFORMABLE_REQUIRED_SCHEMAS),
                    "thin_shell_supported": False,
                    "independent_bend_shear_supported": False,
                },
                "rigid_receptacle": {
                    "authoring_api": RIGID_AUTHORING_API,
                    "runtime_class": RIGID_RUNTIME_CLASS,
                    "expected_prim_type": RIGID_EXPECTED_PRIM_TYPE,
                    "collision_representations": list(RIGID_COLLISION_REPRESENTATIONS),
                    "open_interior_required": True,
                    "top_cap_forbidden": True,
                },
            },
        }
    )


def _fixture(tmp_path: Path) -> tuple[dict, dict, dict, dict[str, Path]]:
    cloth, cloth_root = _candidate(tmp_path, "deformable_volume")
    basket, basket_root = _candidate(tmp_path, "rigid_receptacle")
    return (
        _contract(cloth, basket),
        cloth,
        basket,
        {"cloth": cloth_root, "basket": basket_root},
    )


def test_bundle_is_deterministic_entity_keyed_and_pre_canary(tmp_path: Path) -> None:
    contract, cloth, basket, roots = _fixture(tmp_path)
    identity = _runtime_identity()

    first = build_native_task_entity_asset_authoring_bundle(
        output_dir=tmp_path / "first",
        task_entity_contract=contract,
        asset_candidates=[cloth, basket],
        asset_source_roots=roots,
        runtime_identity=identity,
        generated_at="2026-08-10T15:00:00Z",
    )
    second = build_native_task_entity_asset_authoring_bundle(
        output_dir=tmp_path / "second",
        task_entity_contract=contract,
        asset_candidates=[basket, cloth],
        asset_source_roots={"basket": roots["basket"], "cloth": roots["cloth"]},
        runtime_identity=identity,
        generated_at="2026-08-10T15:00:00Z",
    )

    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["receipt_digest"] == second["receipt_digest"]
    assert first["native_simulator_executed"] is False
    assert first["native_qualification_claimed"] is False
    assert first["raw_dataset_source_bytes_included"] is False
    assert first["pending_native_gaps"] == list(PENDING_NATIVE_GAPS)
    assert set(first["asset_candidate_digests"]) == {"basket", "cloth"}

    persisted = verify_native_task_entity_asset_authoring_bundle(
        tmp_path / "first" / RECEIPT_FILENAME,
        expected_task_entity_contract_digest=contract["contract_digest"],
        expected_runtime_identity_digest=identity["runtime_identity_digest"],
    )
    assert persisted == first

    with zipfile.ZipFile(tmp_path / "first" / BUNDLE_FILENAME) as archive:
        names = archive.namelist()
        assert all("InteriorGS" not in name and "SAGE" not in name for name in names)
        manifest_name = (
            "native_task_entity_asset_authoring_source/"
            "native_task_entity_asset_authoring_input.v1.json"
        )
        manifest = json.loads(archive.read(manifest_name))
        assert manifest["asset_entity_ids"] == ["basket", "cloth"]
        assert manifest["raw_dataset_source_bytes_included"] is False
        assert manifest["native_simulator_executed"] is False
        plans = {row["entity_id"]: row for row in manifest["entity_authoring_plans"]}
        assert plans["cloth"]["operation"]["operation_kind"] == (
            "compose_closed_volumetric_fem_candidate"
        )
        assert plans["cloth"]["operation"]["cooking_api"] == (DEFORMABLE_COOKING_API)
        assert plans["cloth"]["operation"]["thin_shell_cloth_claimed"] is False
        assert plans["basket"]["operation"]["operation_kind"] == (
            "compose_open_rigid_receptacle_candidate"
        )
        assert plans["basket"]["operation"]["top_cap_forbidden"] is True


def test_thin_shell_claim_fails_before_any_output(tmp_path: Path) -> None:
    contract, cloth, basket, roots = _fixture(tmp_path)
    cloth = copy.deepcopy(cloth)
    cloth["deformable_configuration"]["material"]["thin_shell_cloth_claimed"] = True
    cloth["candidate_digest"] = canonical_digest(cloth, digest_field="candidate_digest")

    with pytest.raises(
        NativeTaskEntityAssetAuthoringBundleError,
        match="native_asset_authoring_unsupported_thin_shell_claim:cloth",
    ):
        build_native_task_entity_asset_authoring_bundle(
            output_dir=tmp_path / "bundle",
            task_entity_contract=contract,
            asset_candidates=[cloth, basket],
            asset_source_roots=roots,
            runtime_identity=_runtime_identity(),
        )
    assert not (tmp_path / "bundle").exists()


def test_missing_runtime_identity_is_a_typed_blocker(tmp_path: Path) -> None:
    contract, cloth, basket, roots = _fixture(tmp_path)

    with pytest.raises(
        NativeTaskEntityAssetAuthoringBundleError,
        match="native_asset_authoring_runtime_identity_missing",
    ):
        build_native_task_entity_asset_authoring_bundle(
            output_dir=tmp_path / "bundle",
            task_entity_contract=contract,
            asset_candidates=[cloth, basket],
            asset_source_roots=roots,
            runtime_identity=None,
        )


def test_runtime_revision_drift_fails_closed(tmp_path: Path) -> None:
    contract, cloth, basket, roots = _fixture(tmp_path)
    identity = _runtime_identity()
    identity["runtime_sources"]["isaac_lab"]["revision"] = "0" * 40
    identity["runtime_identity_digest"] = canonical_digest(
        identity, digest_field="runtime_identity_digest"
    )

    with pytest.raises(
        NativeTaskEntityAssetAuthoringBundleError,
        match=("native_asset_authoring_runtime_identity_source_mismatch:isaac_lab:revision"),
    ):
        build_native_task_entity_asset_authoring_bundle(
            output_dir=tmp_path / "bundle",
            task_entity_contract=contract,
            asset_candidates=[cloth, basket],
            asset_source_roots=roots,
            runtime_identity=identity,
        )


def test_candidate_bytes_are_reverified_before_bundle_creation(tmp_path: Path) -> None:
    contract, cloth, basket, roots = _fixture(tmp_path)
    (roots["cloth"] / "assets/runtime_usd.bin").write_bytes(b"tampered")

    with pytest.raises(
        NativeTaskEntityAssetAuthoringBundleError,
        match="native_asset_authoring_source_file_identity_mismatch:cloth:runtime_usd",
    ):
        build_native_task_entity_asset_authoring_bundle(
            output_dir=tmp_path / "bundle",
            task_entity_contract=contract,
            asset_candidates=[cloth, basket],
            asset_source_roots=roots,
            runtime_identity=_runtime_identity(),
        )
    assert not (tmp_path / "bundle").exists()


def test_candidate_set_must_match_supported_task_entities(tmp_path: Path) -> None:
    contract, cloth, _basket, roots = _fixture(tmp_path)

    with pytest.raises(NativeTaskEntityAssetAuthoringBundleError) as exc_info:
        build_native_task_entity_asset_authoring_bundle(
            output_dir=tmp_path / "bundle",
            task_entity_contract=contract,
            asset_candidates=[cloth],
            asset_source_roots={"cloth": roots["cloth"]},
            runtime_identity=_runtime_identity(),
        )

    assert "native_asset_authoring_candidate_missing:basket" in exc_info.value.errors
    assert "native_asset_authoring_asset_class_missing:rigid_receptacle" in exc_info.value.errors


def test_configuration_digest_join_cannot_be_bypassed(tmp_path: Path) -> None:
    contract, cloth, basket, roots = _fixture(tmp_path)
    entities = copy.deepcopy(contract["task_entities"])
    cloth_entity = next(row for row in entities if row["entity_id"] == "cloth")
    cloth_entity["digests"]["configuration_sha256"] = _sha("0")
    mismatched_contract = materialize_native_task_entity_contract(
        task_kind=TASK_KIND_DEFORMABLE_TRANSFER,
        task_entities=entities,
    )

    with pytest.raises(
        NativeTaskEntityAssetAuthoringBundleError,
        match="native_asset_authoring_configuration_mismatch:cloth",
    ):
        build_native_task_entity_asset_authoring_bundle(
            output_dir=tmp_path / "bundle",
            task_entity_contract=mismatched_contract,
            asset_candidates=[cloth, basket],
            asset_source_roots=roots,
            runtime_identity=_runtime_identity(),
        )


def test_private_derived_upload_authority_is_required(tmp_path: Path) -> None:
    contract, cloth, basket, roots = _fixture(tmp_path)
    basket = copy.deepcopy(basket)
    basket["rights"]["derived_asset_private_upload_permitted"] = False
    basket["candidate_digest"] = canonical_digest(basket, digest_field="candidate_digest")

    with pytest.raises(
        NativeTaskEntityAssetAuthoringBundleError,
        match="native_asset_authoring_derived_upload_not_permitted:basket",
    ):
        build_native_task_entity_asset_authoring_bundle(
            output_dir=tmp_path / "bundle",
            task_entity_contract=contract,
            asset_candidates=[cloth, basket],
            asset_source_roots=roots,
            runtime_identity=_runtime_identity(),
        )


def test_source_and_provenance_joins_are_entity_scoped(tmp_path: Path) -> None:
    contract, cloth, basket, roots = _fixture(tmp_path)
    basket = copy.deepcopy(basket)
    basket["source_observation"]["source_reference"] = "wrong/source/reference"
    basket["rights"]["provider_terms_id"] = "wrong-provider-terms"
    basket["candidate_digest"] = canonical_digest(basket, digest_field="candidate_digest")

    with pytest.raises(NativeTaskEntityAssetAuthoringBundleError) as exc_info:
        build_native_task_entity_asset_authoring_bundle(
            output_dir=tmp_path / "bundle",
            task_entity_contract=contract,
            asset_candidates=[cloth, basket],
            asset_source_roots=roots,
            runtime_identity=_runtime_identity(),
        )

    assert "native_asset_authoring_source_observation_mismatch:basket" in exc_info.value.errors
    assert "native_asset_authoring_provenance_mismatch:basket" in exc_info.value.errors


def test_receipt_verifier_rejects_bundle_tamper(tmp_path: Path) -> None:
    contract, cloth, basket, roots = _fixture(tmp_path)
    build_native_task_entity_asset_authoring_bundle(
        output_dir=tmp_path / "bundle",
        task_entity_contract=contract,
        asset_candidates=[cloth, basket],
        asset_source_roots=roots,
        runtime_identity=_runtime_identity(),
        generated_at="2026-08-10T15:00:00Z",
    )
    with (tmp_path / "bundle" / BUNDLE_FILENAME).open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(
        NativeTaskEntityAssetAuthoringBundleError,
        match="native_asset_authoring_bundle_bytes_identity_mismatch",
    ):
        verify_native_task_entity_asset_authoring_bundle(tmp_path / "bundle" / RECEIPT_FILENAME)
