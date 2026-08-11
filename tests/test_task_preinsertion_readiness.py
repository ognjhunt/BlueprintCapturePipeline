from __future__ import annotations

import base64
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline import deformable_native_capability_preflight as preflight
from blueprint_pipeline.composed_paired_entity_placement import (
    plan_composed_paired_entity_placement,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.deformable_native_capability_preflight import (
    ALL_REQUIRED_SYMBOLS,
    FROZEN_CANDIDATES,
    MINIMUM_INTERNAL_RUNTIME_MODULES,
    POLICY_ADAPTER_MODULES,
    REQUEST_SCHEMA_VERSION as PREFLIGHT_REQUEST_SCHEMA_VERSION,
    build_deformable_native_capability_preflight,
)
from blueprint_pipeline.native_task_runtime_source_packet import (
    ARENA_COMMIT,
    ARENA_REPOSITORY,
    ARENA_TREE,
    ISAACLAB_COMMIT,
    ISAACLAB_REPOSITORY,
    ISAACLAB_TREE,
)
from blueprint_pipeline.registered_static_receptacle_asset import (
    CANDIDATE_FILENAME as REGISTERED_RECEPTACLE_CANDIDATE_FILENAME,
    RECEIPT_FILENAME as REGISTERED_RECEPTACLE_RECEIPT_FILENAME,
    VISUAL_BASIS_FILENAME as REGISTERED_RECEPTACLE_VISUAL_BASIS_FILENAME,
    build_registered_static_receptacle_asset,
)
from blueprint_pipeline.sage_collision_component_topology import (
    inspect_sage_collision_component_topology,
)
from blueprint_pipeline.semantic_review_attestation import (
    TRUSTED_PUBLIC_KEY_SHA256_ENV,
    canonical_semantic_authority_selection_bytes,
    canonical_semantic_review_attestation_bytes,
    materialize_semantic_authority_selection,
    materialize_semantic_review_attestation,
    materialize_semantic_review_payload,
    semantic_frame_evidence_digest,
    semantic_review_signature_message,
)
from blueprint_pipeline.task_preinsertion_readiness import (
    CAMERA_SCHEMA_VERSION,
    CAMERA_EXTRINSICS_SCHEMA_VERSION,
    ENTITY_SCHEMA_VERSION,
    ENGINEERED_ASSET_EVIDENCE_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION,
    PLACEMENT_SCHEMA_VERSION,
    PREFLIGHT_OBSERVATIONS_SCHEMA_VERSION,
    REGISTRATION_EVIDENCE_SCHEMA_VERSION,
    REGISTRATION_TRANSFORM_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION,
    RIGHTS_SCHEMA_VERSION,
    RIGHTS_EVIDENCE_SCHEMA_VERSION,
    RIGHTS_AUTHORITY_PUBLIC_KEY_SHA256_ENV,
    RIGHTS_INTERPRETATION_VERSION,
    REGISTERED_RECEPTACLE_REPLAY_REQUEST_SCHEMA_VERSION,
    RUNTIME_SCHEMA_VERSION,
    SCENARIO_SCHEMA_VERSION,
    RESOLVED_SCENARIO_CELL_SCHEMA_VERSION,
    SCENE_SCHEMA_VERSION,
    SCORER_SCHEMA_VERSION,
    SOURCE_EVIDENCE_SCHEMA_VERSION,
    TASK_FREEZE_AUTHORITY_PUBLIC_KEY_SHA256_ENV,
    TASK_SCHEMA_VERSION,
    TRUST_SCHEMA_VERSION,
    TOPOLOGY_EVIDENCE_SCHEMA_VERSION,
    TOPOLOGY_SURVEY_SCHEMA_VERSION,
    VISUAL_OBSERVATION_PROVENANCE_SCHEMA_VERSION,
    TaskPreinsertionReadinessError,
    collect_task_preinsertion_readiness,
    prompt_task_spec_freeze_digest,
    rights_evidence_signature_message,
    source_observation_signature_message,
    task_freeze_signature_message,
)
from blueprint_pipeline.task_entity_asset_candidate import (
    materialize_task_entity_asset_candidate,
)


_TARGET = {
    "rigid_pick_place": ("can", "movable_rigid", "rigid_body"),
    "articulated_open_close": ("refrigerator", "articulated_fixture", "articulation"),
    "deformable_transfer": ("cloth", "movable_deformable", "deformable_volume"),
}
_RIGHTS_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"R" * 32)
_SEMANTIC_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"S" * 32)
_TASK_FREEZE_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"T" * 32)


def _public_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


@pytest.fixture(autouse=True)
def _trusted_fixture_authorities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        RIGHTS_AUTHORITY_PUBLIC_KEY_SHA256_ENV,
        _digest_bytes(_public_key_bytes(_RIGHTS_PRIVATE_KEY)),
    )
    monkeypatch.setenv(
        TRUSTED_PUBLIC_KEY_SHA256_ENV,
        _digest_bytes(_public_key_bytes(_SEMANTIC_PRIVATE_KEY)),
    )
    monkeypatch.setenv(
        TASK_FREEZE_AUTHORITY_PUBLIC_KEY_SHA256_ENV,
        _digest_bytes(_public_key_bytes(_TASK_FREEZE_PRIVATE_KEY)),
    )


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_json(path: Path, value: dict[str, Any]) -> str:
    content = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return _digest_bytes(content)


def _write_bytes(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return _digest_bytes(content)


def _seal(value: dict[str, Any], field: str = "receipt_digest") -> dict[str, Any]:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _source_evidence(
    *,
    entity_id: str,
    source_id: str,
    source_sha256: str,
    classification: str,
    observed: bool,
    design_basis_only: bool,
    source_instance_id: str,
    coordinate_frame_id: str,
    bounds_world: dict[str, list[float]],
    rest_state: str,
    support_relation: dict[str, Any],
    cited_visual_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    dimensions = [
        round(float(maximum) - float(minimum), 9)
        for minimum, maximum in zip(
            bounds_world["minimum_m"], bounds_world["maximum_m"], strict=True
        )
    ]
    semantic_authority: dict[str, Any] | None = None
    if classification == "observed_source":
        public_key = _public_key_bytes(_SEMANTIC_PRIVATE_KEY)
        semantic_authority = {
            "authority_id": "fixture-source-observation-authority",
            "key_id": "fixture-source-observation-key",
            "public_key_base64": base64.b64encode(public_key).decode("ascii"),
            "public_key_sha256": _digest_bytes(public_key),
            "signature_base64": "",
        }
    evidence = {
        "schema_version": SOURCE_EVIDENCE_SCHEMA_VERSION,
        "evidence_id": f"source-evidence:{entity_id}",
        "entity_id": entity_id,
        "source_id": source_id,
        "source_sha256": source_sha256,
        "source_instance_id": source_instance_id,
        "coordinate_frame_id": coordinate_frame_id,
        "bounds_world": bounds_world,
        "metric_dimensions_m": dimensions,
        "rest_state": rest_state,
        "support_relation": support_relation,
        "cited_visual_evidence": cited_visual_evidence,
        "cited_visual_evidence_digest": canonical_digest(
            {"cited_visual_evidence": cited_visual_evidence}
        ),
        "classification": classification,
        "observed": observed,
        "design_basis_only": design_basis_only,
        "semantic_authority": semantic_authority,
        "receipt_digest": "",
    }
    if semantic_authority is not None:
        semantic_authority["signature_base64"] = base64.b64encode(
            _SEMANTIC_PRIVATE_KEY.sign(source_observation_signature_message(evidence))
        ).decode("ascii")
    return _seal(evidence)


def _preflight_fixture(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime = root / "runtime"
    isaaclab_root = runtime / "isaaclab"
    arena_root = runtime / "arena"
    simulator_root = runtime / "isaac-sim"
    dependency_root = runtime / "dependencies"
    internal_root = runtime / "bundle"
    for directory in (
        isaaclab_root,
        arena_root,
        simulator_root,
        dependency_root,
        internal_root,
    ):
        directory.mkdir(parents=True)
    for relative in preflight.ISAACLAB_REQUIRED_RELATIVE_PATHS:
        _write_bytes(isaaclab_root / relative, f"# {relative}\n".encode())
    for relative in preflight.ARENA_REQUIRED_RELATIVE_PATHS:
        _write_bytes(arena_root / relative, f"# {relative}\n".encode())

    dependencies = [
        {
            "package": "fixture-pure",
            "version": "1.0.0",
            "wheel_tag": "py3-none-any",
            "wheel_sha256": _sha("a"),
            "import_module": "fixture_pure",
        },
        {
            "package": "fixture-binary",
            "version": "2.0.0",
            "wheel_tag": "cp312-cp312-manylinux_2_28_x86_64",
            "wheel_sha256": _sha("b"),
            "import_module": "fixture_binary",
        },
    ]
    installed = []
    for row in dependencies:
        module_file = dependency_root / str(row["import_module"]) / "__init__.py"
        _write_bytes(module_file, b"# dependency\n")
        installed.append({**row, "module_file": str(module_file)})

    internal_requirements = []
    internal_observations = []
    internal_paths: dict[str, Path] = {}
    for module in MINIMUM_INTERNAL_RUNTIME_MODULES:
        path = internal_root / (module.replace(".", "/") + ".py")
        digest = _write_bytes(path, module.encode())
        internal_paths[module] = path
        internal_requirements.append({"module": module, "sha256": digest})
        internal_observations.append({"module": module, "sha256": digest, "file": str(path)})

    checkpoints = {
        "pi05_droid": {
            "checkpoint_uri": "gs://fixture/pi05",
            "checkpoint_inventory_sha256": _sha("c"),
            "object_count": 3,
            "size_bytes": 1234,
        },
        "groot_n17_droid": {
            "model_id": "nvidia/GR00T-N1.7-DROID",
            "checkpoint_revision": "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5",
            "checkpoint_files_sha256": _sha("d"),
        },
    }
    policy_requirements = []
    policy_observations = []
    for candidate_id in FROZEN_CANDIDATES:
        module = POLICY_ADAPTER_MODULES[candidate_id]
        path = internal_paths[module]
        digest = "sha256:" + __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        policy_requirements.append(
            {
                "candidate_id": candidate_id,
                "adapter_module": module,
                "adapter_sha256": digest,
                "checkpoint_identity": checkpoints[candidate_id],
            }
        )
        policy_observations.append(
            {
                "candidate_id": candidate_id,
                "adapter_module": module,
                "adapter_file": str(path),
                "adapter_sha256": digest,
                "checkpoint_identity": checkpoints[candidate_id],
            }
        )

    simulator_identity = {
        "runtime_id": "isaac-sim-fixture",
        "container_image": "fixture.invalid/isaac@sha256:" + "e" * 64,
    }
    request = {
        "schema_version": PREFLIGHT_REQUEST_SCHEMA_VERSION,
        "preflight_id": "fixture-deformable-preflight",
        "selected_robot_id": "franka_panda",
        "simulator_runtime_identity": simulator_identity,
        "runtime_python": {
            "implementation": "CPython",
            "version": "3.12.11",
            "python_tag": "cp312",
            "abi_tag": "cp312",
            "platform_tags": ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64"],
        },
        "dependency_closure": dependencies,
        "cuda_warp_requirements": {
            "torch_cuda_version": "12.8",
            "warp_version": "1.8.1",
            "selected_device": "cuda:0",
            "minimum_compute_capability": [8, 0],
        },
        "policy_identities": policy_requirements,
        "internal_runtime_modules": internal_requirements,
    }

    known_module_paths = {
        "isaaclab.sim.schemas.schemas": "source/isaaclab/isaaclab/sim/schemas/schemas.py",
        "isaaclab.sim.spawners.materials.physics_materials": "source/isaaclab/isaaclab/sim/spawners/materials/physics_materials.py",
        "isaaclab.sensors.contact_sensor.contact_sensor": "source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py",
        "isaaclab.sensors.contact_sensor.contact_sensor_cfg": "source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py",
        "isaaclab.sensors.camera.camera": "source/isaaclab/isaaclab/sensors/camera/camera.py",
        "isaaclab.sensors.camera.camera_cfg": "source/isaaclab/isaaclab/sensors/camera/camera_cfg.py",
        "isaaclab.sim.simulation_context": "source/isaaclab/isaaclab/sim/simulation_context.py",
        "isaaclab_physx.assets.deformable_object.deformable_object": "source/isaaclab_physx/isaaclab_physx/assets/deformable_object/deformable_object.py",
        "isaaclab_physx.assets.deformable_object.deformable_object_cfg": "source/isaaclab_physx/isaaclab_physx/assets/deformable_object/deformable_object_cfg.py",
        "isaaclab_physx.assets.deformable_object.deformable_object_data": "source/isaaclab_physx/isaaclab_physx/assets/deformable_object/deformable_object_data.py",
    }
    module_files: dict[str, str] = {}
    for module in {symbol.split(":", 1)[0] for symbol in ALL_REQUIRED_SYMBOLS}:
        if module in known_module_paths:
            path = isaaclab_root / known_module_paths[module]
        elif module in {"torch", "warp"}:
            path = dependency_root / module / "__init__.py"
            _write_bytes(path, f"# {module}\n".encode())
        else:
            path = simulator_root / "python" / (module.replace(".", "/") + ".py")
            _write_bytes(path, f"# {module}\n".encode())
        module_files[module] = str(path)

    observations = {
        "schema_version": PREFLIGHT_OBSERVATIONS_SCHEMA_VERSION,
        "source_roots": {
            "isaaclab": {
                "root_path": str(isaaclab_root),
                "repository": ISAACLAB_REPOSITORY,
                "revision": ISAACLAB_COMMIT,
                "tree": ISAACLAB_TREE,
                "source_receipt_digest": _sha("1"),
            },
            "arena": {
                "root_path": str(arena_root),
                "repository": ARENA_REPOSITORY,
                "revision": ARENA_COMMIT,
                "tree": ARENA_TREE,
                "source_receipt_digest": _sha("2"),
            },
        },
        "simulator_runtime_identity": {**simulator_identity, "root_path": str(simulator_root)},
        "python_runtime": deepcopy(request["runtime_python"]),
        "dependency_root": str(dependency_root),
        "installed_dependencies": installed,
        "selected_embodiment": {
            "robot_id": "franka_panda",
            "selected_module": "isaaclab_arena.embodiments.droid.droid",
            "selected_module_file": str(arena_root / "isaaclab_arena/embodiments/droid/droid.py"),
        },
        "imported_modules": [
            "isaaclab_arena.embodiments",
            "isaaclab_arena.embodiments.droid",
            "isaaclab_arena.embodiments.droid.droid",
        ],
        "available_symbols": list(ALL_REQUIRED_SYMBOLS),
        "module_files": module_files,
        "cuda_warp_declarations": {
            "torch_version": "2.7.0",
            "torch_cuda_version": "12.8",
            "warp_version": "1.8.1",
            "cuda_driver_version": "570.86.15",
            "selected_device": "cuda:0",
            "warp_devices": ["cpu", "cuda:0"],
            "device_name": "Fixture GPU",
            "compute_capability": [8, 9],
        },
        "media_tools": [
            {
                "name": name,
                "executable": f"/fixture/bin/{name}",
                "returncode": 0,
                "version_line": f"{name} version fixture",
            }
            for name in ("ffmpeg", "ffprobe")
        ],
        "policy_identities": policy_observations,
        "internal_runtime_modules": internal_observations,
        "deformable_model": "volumetric_fem",
        "claimed_capabilities": ["volumetric_fem"],
    }
    matrix = build_deformable_native_capability_preflight(
        request=request, observations=observations
    )
    return request, observations, matrix


def _basket_candidate(
    root: Path,
) -> tuple[dict[str, Any], list[tuple[str, str, str, str, str, None]]]:
    file_rows = []
    bindings: list[tuple[str, str, str, str, str, None]] = []
    obj = b"""# open receptacle fixture
v -0.15 -0.10 0.00
v 0.15 -0.10 0.00
v 0.15 0.10 0.00
v -0.15 0.10 0.00
v -0.15 -0.10 0.10
v 0.15 -0.10 0.10
v 0.15 0.10 0.10
v -0.15 0.10 0.10
f 1 4 3 2
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""
    runtime_usda = b"""#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Asset"
{
    def Mesh "Geometry" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        point3f[] points = [(-0.15, -0.10, 0.00), (0.15, -0.10, 0.00), (0.15, 0.10, 0.00), (-0.15, 0.10, 0.00), (-0.15, -0.10, 0.10), (0.15, -0.10, 0.10), (0.15, 0.10, 0.10), (-0.15, 0.10, 0.10)]
        int[] faceVertexCounts = [4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 3, 2, 1, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
        uniform token subdivisionScheme = "none"
    }
}
"""
    contents = {
        "visual_geometry": ("visual_geometry.obj", obj),
        "collision_geometry": ("collision_geometry.obj", obj),
        "material_definition": (
            "material_definition.json",
            b'{"display_color_rgb":[0.2,0.4,0.6],"schema_version":"fixture_material.v1"}\n',
        ),
        "texture": ("texture.ppm", b"P3\n1 1\n255\n51 102 153\n"),
        "physics_configuration": (
            "physics_configuration.json",
            b'{"dynamic_friction":0.5,"schema_version":"fixture_physics.v1","static_friction":0.6}\n',
        ),
        "runtime_usd": ("runtime_asset.usda", runtime_usda),
    }
    for role, (filename, content) in contents.items():
        relative = f"assets/basket/{filename}"
        digest = _write_bytes(root / relative, content)
        file_rows.append(
            {
                "role": role,
                "path": relative,
                "sha256": digest,
                "size_bytes": len(content),
            }
        )
        bindings.append(
            (f"basket_file_{role}", "supporting_evidence", relative, digest, "opaque", None)
        )
    source = {
        "schema_version": "task_entity_asset_candidate.v1",
        "entity_id": "basket",
        "asset_id": "asset:basket",
        "asset_class": "rigid_receptacle",
        "source_observation": {
            "observation_id": "sage-component:87",
            "source_reference": "InteriorGS/SAGE:scene-new",
            "source_sha256": _sha("1"),
            "source_size_bytes": 4096,
            "bounds_world": {
                "minimum_m": [0.0, 0.0, 0.0],
                "maximum_m": [0.3, 0.2, 0.1],
            },
            "metric_dimensions_m": [0.3, 0.2, 0.1],
            "coverage": {
                "metric_bounds_observed": True,
                "rest_state_bounded": True,
                "full_surface_observed": False,
                "interior_collision_observed": False,
                "interior_appearance_observed": False,
                "engineered_interior_not_factual": True,
                "unobserved_regions": ["source occupied interior"],
            },
        },
        "rights": {
            "source_revision": "a" * 40,
            "license_id": "fixture-license",
            "license_reference": "https://example.invalid/license",
            "license_sha256": _sha("4"),
            "attribution": "fixture attribution",
            "derived_processing_authority_id": "fixture-derived-authority",
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
            "source_revision": "a" * 40,
            "source_tree": "b" * 40,
            "package_name": "released-authoring",
            "package_version": "1.0.0",
            "generated_geometry_used": True,
            "generated_physics_used": False,
        },
        "files": file_rows,
        "transform": {
            "authored_origin_m": [0.0, 0.0, 0.0],
            "pivot_m": [0.0, 0.0, 0.0],
            "scale_xyz": [1.0, 1.0, 1.0],
            "world_pose": {
                "position_world_m": [1.0, 1.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "meters_per_unit": 1.0,
            "up_axis": "Z",
        },
        "simulator_import": {
            "simulator": "Isaac Sim",
            "simulator_version": "6.0.0-dev2",
            "source_repository": "https://github.com/isaac-sim/IsaacLab",
            "source_revision": "a" * 40,
            "importer_module": "isaaclab.assets.RigidObject",
            "expected_prim_type": "UsdGeom.Xform+collision",
        },
        "receptacle_configuration": {
            "geometry": {
                "open_interior": True,
                "top_cap_present": False,
                "interior_dimensions_m": [0.28, 0.18, 0.08],
                "wall_thickness_m": 0.01,
                "floor_thickness_m": 0.01,
                "engineered_interior": True,
            },
            "collision": {
                "representation": "multi_part_convex_open_receptacle",
                "collision_sha256": file_rows[1]["sha256"],
                "contact_offset_m": 0.002,
                "rest_offset_m": 0.0,
            },
            "material": {
                "static_friction": 0.6,
                "dynamic_friction": 0.5,
                "restitution": 0.0,
                "material_provenance_sha256": file_rows[2]["sha256"],
            },
            "anchoring": {
                "static_anchored": True,
                "mass_kg": 0.0,
                "inertia_diagonal_kg_m2": [0.0, 0.0, 0.0],
                "stable_support_readback_required": True,
                "native_collision_readback_required": True,
            },
        },
        "retained_diagnostic_requirements": [
            "native_import",
            "stable_support_and_no_initial_penetration",
            "native_contact",
            "native_reset_readback",
            "native_render_coverage",
        ],
    }
    return materialize_task_entity_asset_candidate(source), bindings


def _registered_basket_evidence(
    root: Path,
    *,
    reference_position_world_m: list[float],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[tuple[str, str, str, str, str, str | None]],
    dict[str, Any],
]:
    bindings: list[tuple[str, str, str, str, str, str | None]] = []
    source_root = root / "evidence/basket_source"
    source_root.mkdir(parents=True)
    labels_path = source_root / "labels.json"
    labels_path.write_text(
        json.dumps(
            [
                {
                    "ins_id": "87",
                    "label": "basket",
                    "bounding_box": [
                        {"x": x, "y": y, "z": z}
                        for z in (0.0, 0.1)
                        for x, y in (
                            (0.0, 0.0),
                            (0.0, 0.2),
                            (0.3, 0.2),
                            (0.3, 0.0),
                        )
                    ],
                }
            ],
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    collision_path = source_root / "collision.usda"
    stage = Usd.Stage.CreateNew(str(collision_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    mesh = UsdGeom.Mesh.Define(stage, "/Root/OpenBasket")
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    def add_box(
        minimum: tuple[float, float, float],
        maximum: tuple[float, float, float],
    ) -> None:
        start = len(points)
        x0, y0, z0 = minimum
        x1, y1, z1 = maximum
        points.extend(
            [
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            ]
        )
        faces.extend(
            tuple(start + index for index in face)
            for face in (
                (0, 3, 2, 1),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
            )
        )

    add_box((0.0, 0.0, 0.0), (0.3, 0.2, 0.01))
    add_box((0.0, 0.0, 0.01), (0.01, 0.2, 0.1))
    add_box((0.29, 0.0, 0.01), (0.3, 0.2, 0.1))
    add_box((0.01, 0.0, 0.01), (0.29, 0.01, 0.1))
    add_box((0.01, 0.19, 0.01), (0.29, 0.2, 0.1))
    # The source topology contract operates on connected mesh components, so
    # keep the five authored pieces in one explicit component.  These floor-
    # band triangles join indices without adding geometry inside the opening.
    faces.extend(
        (
            (4, 8, 9),
            (5, 16, 17),
            (4, 24, 25),
            (7, 34, 35),
        )
    )
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    topology = inspect_sage_collision_component_topology(
        labels_path=labels_path,
        target_instance_ids=["87"],
        opening_probe_instance_ids=["87"],
        sage_collision_usd_path=collision_path,
    )
    assert topology["targets"][0]["opening_probe"]["open_collision_cavity_passed"]
    topology_relative = "evidence/basket_source/topology.json"
    topology_sha256 = _write_json(root / topology_relative, topology)

    frame_root = source_root / "frames"
    frame_root.mkdir()
    frame_path = frame_root / "basket_closeup.png"
    image = Image.new("RGB", (8, 6))
    image.putdata([(20 + index * 3, 40 + index * 2, 60 + index) for index in range(48)])
    image.save(frame_path, format="PNG")
    frame_bytes = frame_path.read_bytes()
    frame_sha256 = _digest_bytes(frame_bytes)
    with Image.open(frame_path) as decoded:
        decoded_rgb_sha256 = _digest_bytes(decoded.convert("RGB").tobytes())
    render_manifest = {
        "schema_version": "splat_scene_render.v1",
        "source_digest": _sha("1"),
        "renderer_identity": {
            "name": "fixture_renderer",
            "harness_sha256": _sha("2"),
            "entry_sha256": _sha("3"),
            "width": 8,
            "height": 6,
            "pixel_ratio": 1,
            "supersampling": 1,
            "alpha": False,
            "background_rgb_hex": "0x000000",
            "output_format": "lossless_png",
            "color_space": "srgb",
            "node_version": "v1.0.0",
            "dependency_versions": {"renderer": "1.0.0"},
        },
        "render": {"status": "completed", "returncode": 0},
        "appearance_fidelity": {
            "source_splat_count": 100,
            "retained_splat_count": 100,
            "appearance_fidelity_qualified": False,
            "evaluation_input_authorized": False,
        },
        "cameras": [
            {
                "id": "camera-external",
                "path": frame_path.name,
                "bytes": len(frame_bytes),
                "nonblank": True,
                "digest": frame_sha256,
            }
        ],
        "camera_calibration": [
            {
                "id": "camera-external",
                "pose_convention": "world_position_target_up_look_at_z_up",
                "position_world_m": [1.0, 2.0, 3.0],
                "target_world_m": [0.0, 0.0, 0.0],
                "up_world": [0.0, 0.0, 1.0],
                "intrinsics": {
                    "model": "pinhole_centered_square_pixels",
                    "fx": 8.0,
                    "fy": 8.0,
                    "cx": 4.0,
                    "cy": 3.0,
                    "width": 8,
                    "height": 6,
                    "vertical_fov_deg": 45.0,
                },
            }
        ],
        "render_manifest_digest": "",
    }
    render_manifest["render_manifest_digest"] = canonical_digest(
        render_manifest, digest_field="render_manifest_digest"
    )
    review = {
        "schema_version": "adp_deformable_scene_visual_review.v1",
        "scene_id": "scene-new",
        "reviewer_id": "fixture-reviewer",
        "reviewed_at": "2026-08-10T00:00:00Z",
        "learned_policy_outcomes_inspected": False,
        "reconnaissance_only": True,
        "render_manifest_digest": render_manifest["render_manifest_digest"],
        "collision_topology_receipt_digest": topology["receipt_digest"],
        "targets": [
            {
                "target_id": "basket-target",
                "publisher_instance_id": "87",
                "target_kind": "destination_receptacle",
                "material_class": "not_applicable",
                "material_class_supported_by_observation": False,
                "rest_state": "not_applicable",
                "support_relation": "observed_container_contents",
                "rigid_exterior_observed": True,
                "open_rim_observed": True,
                "interior_occupied": True,
                "complete_interior_appearance_observed": False,
                "collision_component_identity_passed": True,
                "open_collision_cavity_passed": True,
                "source_destination_admitted": False,
                "engineered_twin_design_basis_admitted": True,
                "selection_role": "engineered_twin_design_basis",
                "cited_frames": [
                    {
                        "camera_id": "camera-external",
                        "sha256": frame_sha256,
                        "size_bytes": len(frame_bytes),
                    }
                ],
                "review_notes": "Fixture engineered-twin basis.",
            }
        ],
        "selected_movable_instance_id": "fixture-cloth",
        "selected_destination_design_basis_instance_id": "87",
        "source_destination_is_occupied": True,
        "source_destination_complete_interior_appearance_observed": False,
        "composition_required": True,
        "claim_boundary": {
            "virtual_closeup_recovers_missing_source_observations": False,
            "collision_cavity_establishes_hidden_appearance": False,
            "engineered_twin_hidden_geometry_is_source_truth": False,
            "review_is_evaluation_policy_media": False,
            "physical_material_equivalence_proven": False,
        },
        "review_digest": "",
    }
    review["review_digest"] = canonical_digest(review, digest_field="review_digest")
    render_relative = "evidence/basket_source/render_manifest.json"
    review_relative = "evidence/basket_source/visual_review.json"
    render_sha256 = _write_json(root / render_relative, render_manifest)
    review_sha256 = _write_json(root / review_relative, review)
    semantic_rows = [
        {
            "target_id": "basket-target",
            "camera_id": "camera-external",
            "sha256": frame_sha256,
            "size_bytes": len(frame_bytes),
            "decoded_rgb_sha256": decoded_rgb_sha256,
        }
    ]
    semantic_payload = materialize_semantic_review_payload(
        attestation_id="basket-semantic-attestation",
        selection_id="basket-semantic-selection",
        authority_id="fixture-review-authority",
        authority_key_id="fixture-review-key",
        scene_id="scene-new",
        target_id="basket-target",
        source_instance_id="87",
        semantic_role="destination_receptacle",
        visual_review_digest=review["review_digest"],
        render_manifest_digest=render_manifest["render_manifest_digest"],
        collision_topology_receipt_digest=topology["receipt_digest"],
        cited_frames_digest=semantic_frame_evidence_digest(semantic_rows),
        learned_policy_outcomes_inspected=False,
        semantic_assertions={
            "rigid_exterior_observed": True,
            "open_rim_observed": True,
            "interior_occupied": True,
            "complete_interior_appearance_observed": False,
            "source_destination_admitted": False,
            "engineered_twin_design_basis_admitted": True,
            "selection_role": "engineered_twin_design_basis",
        },
    )
    public_key = _public_key_bytes(_SEMANTIC_PRIVATE_KEY)
    attestation = materialize_semantic_review_attestation(
        payload=semantic_payload,
        public_key_base64=base64.b64encode(public_key).decode("ascii"),
        signature_base64=base64.b64encode(
            _SEMANTIC_PRIVATE_KEY.sign(semantic_review_signature_message(semantic_payload))
        ).decode("ascii"),
    )
    selection = materialize_semantic_authority_selection(attestation=attestation)
    attestation_relative = "evidence/basket_source/semantic_attestation.json"
    selection_relative = "evidence/basket_source/semantic_selection.json"
    attestation_sha256 = _write_bytes(
        root / attestation_relative,
        canonical_semantic_review_attestation_bytes(attestation),
    )
    selection_sha256 = _write_bytes(
        root / selection_relative,
        canonical_semantic_authority_selection_bytes(selection),
    )
    builder_rights = {
        "source_revision": "fixture-source-revision",
        "license_id": "CC-BY-NC-4.0",
        "license_reference": "https://example.invalid/license",
        "license_sha256": _sha("4"),
        "attribution": "Fixture attribution",
        "derived_processing_authority_id": "fixture-derived-authority",
        "provider_terms_id": "fixture-provider-terms",
        "output_rights_id": "fixture-output-rights",
        "raw_source_private_upload_permitted": False,
        "derived_asset_private_upload_permitted": True,
        "raw_redistribution_permitted": False,
        "provider_retention_permitted": False,
        "provider_training_permitted": False,
    }
    authoring_identity = {
        "source_repository": "https://example.invalid/BlueprintCapturePipeline",
        "source_revision": "b" * 40,
        "source_tree": "c" * 40,
        "package_name": "blueprint-pipeline-fixture",
        "package_version": "1.0.0",
    }
    physics = {
        "static_friction": 0.6,
        "dynamic_friction": 0.5,
        "restitution": 0.0,
        "contact_offset_m": 0.002,
        "rest_offset_m": 0.0,
        "diagnostic_display_color_rgb": [0.7, 0.55, 0.35],
    }
    builder_arguments = {
        "target_instance_id": "87",
        "entity_id": "basket",
        "asset_id": "asset:basket",
        "reference_world_pose": {
            "position_world_m": reference_position_world_m,
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "rights": builder_rights,
        "authoring_identity": authoring_identity,
        "physics_configuration": physics,
        "simulator_name": "Isaac Sim",
        "simulator_version": "6.0.1",
    }
    # The retained replay request is canonical JSON.  Feed the initial build
    # the same round-tripped mapping order so byte-for-byte replay also covers
    # the builder's emitted diagnostic JSON files.
    builder_arguments = json.loads(json.dumps(builder_arguments, sort_keys=True))
    asset_root = root / "assets/basket"
    result = build_registered_static_receptacle_asset(
        labels_path=labels_path,
        sage_collision_usd_path=collision_path,
        topology_receipt=topology,
        visual_review_receipt_path=root / review_relative,
        render_manifest_path=root / render_relative,
        frame_root=frame_root,
        semantic_review_attestation_path=root / attestation_relative,
        semantic_authority_selection_path=root / selection_relative,
        output_root=asset_root,
        **builder_arguments,
    )
    candidate = result["candidate"]
    registered_receipt = result["receipt"]

    def bind(
        binding_id: str,
        relative: str,
        *,
        content_type: str,
        schema_version: str | None,
    ) -> str:
        digest = _digest_bytes((root / relative).read_bytes())
        bindings.append(
            (
                binding_id,
                "supporting_evidence",
                relative,
                digest,
                content_type,
                schema_version,
            )
        )
        return digest

    labels_relative = "evidence/basket_source/labels.json"
    collision_relative = "evidence/basket_source/collision.usda"
    frame_relative = "evidence/basket_source/frames/basket_closeup.png"
    labels_sha256 = bind(
        "basket_source_labels", labels_relative, content_type="opaque", schema_version=None
    )
    collision_sha256 = bind(
        "basket_source_collision",
        collision_relative,
        content_type="opaque",
        schema_version=None,
    )
    bind(
        "basket_collision_topology",
        topology_relative,
        content_type="json",
        schema_version="interiorgs_sage_collision_component_topology.v2",
    )
    bind(
        "basket_visual_review",
        review_relative,
        content_type="json",
        schema_version="adp_deformable_scene_visual_review.v1",
    )
    bind(
        "basket_render_manifest",
        render_relative,
        content_type="json",
        schema_version="splat_scene_render.v1",
    )
    bind(
        "basket_semantic_attestation",
        attestation_relative,
        content_type="json",
        schema_version="semantic_review_attestation.v1",
    )
    bind(
        "basket_semantic_selection",
        selection_relative,
        content_type="json",
        schema_version="semantic_review_authority_selection.v1",
    )
    bind(
        "basket_source_visual_external",
        frame_relative,
        content_type="opaque",
        schema_version=None,
    )
    candidate_relative = f"assets/basket/{REGISTERED_RECEPTACLE_CANDIDATE_FILENAME}"
    registered_relative = f"assets/basket/{REGISTERED_RECEPTACLE_RECEIPT_FILENAME}"
    visual_relative = f"assets/basket/{REGISTERED_RECEPTACLE_VISUAL_BASIS_FILENAME}"
    candidate_sha256 = bind(
        "basket_candidate",
        candidate_relative,
        content_type="json",
        schema_version="task_entity_asset_candidate.v1",
    )
    registered_sha256 = bind(
        "basket_registered_asset_receipt",
        registered_relative,
        content_type="json",
        schema_version="registered_static_receptacle_asset.v1",
    )
    visual_sha256 = bind(
        "basket_visual_basis",
        visual_relative,
        content_type="json",
        schema_version="engineered_receptacle_visual_design_basis.v1",
    )
    for row in candidate["files"]:
        bind(
            f"basket_file_{row['role']}",
            f"assets/basket/{row['path']}",
            content_type="opaque",
            schema_version=None,
        )
    module_rows = []
    module_specs = (
        ("registered_static_receptacle_asset", "basket_registered_asset_builder"),
        ("sage_collision_component_topology", "basket_topology_verifier"),
        ("engineered_receptacle_visual_basis", "basket_visual_verifier"),
        ("semantic_review_attestation", "basket_semantic_verifier"),
    )
    for module, binding_id in module_specs:
        relative = f"evidence/modules/{module}.py"
        module_sha256 = _write_bytes(
            root / relative,
            (Path(__file__).parents[1] / f"src/blueprint_pipeline/{module}.py").read_bytes(),
        )
        bindings.append(
            (binding_id, "supporting_evidence", relative, module_sha256, "opaque", None)
        )
        module_rows.append({"module": module, "binding_id": binding_id, "sha256": module_sha256})
    provenance = _seal(
        {
            "schema_version": VISUAL_OBSERVATION_PROVENANCE_SCHEMA_VERSION,
            "evidence_id": "visual-provenance:basket",
            "entity_id": "basket",
            "source_instance_id": "87",
            "source_id": "interiorgs",
            "source_sha256": _sha("1"),
            "coordinate_frame_id": "shared-world",
            "camera_id": "camera-external",
            "frame_binding_id": "basket_source_visual_external",
            "frame_sha256": frame_sha256,
            "frame_size_bytes": len(frame_bytes),
            "width": 8,
            "height": 6,
            "decoded_rgb_sha256": decoded_rgb_sha256,
            "producer_identity": {
                "kind": "registered_scene_render",
                "producer": "fixture_renderer",
                "version": "1.0.0",
                "configuration_sha256": render_manifest["render_manifest_digest"],
            },
            "receipt_digest": "",
        }
    )
    provenance_relative = "evidence/basket_source/visual_provenance.json"
    provenance_sha256 = _write_json(root / provenance_relative, provenance)
    bindings.append(
        (
            "basket_source_visual_provenance",
            "supporting_evidence",
            provenance_relative,
            provenance_sha256,
            "json",
            VISUAL_OBSERVATION_PROVENANCE_SCHEMA_VERSION,
        )
    )
    source_citation = {
        "binding_id": "basket_source_visual_external",
        "camera_id": "camera-external",
        "sha256": frame_sha256,
        "size_bytes": len(frame_bytes),
        "width": 8,
        "height": 6,
        "decoded_rgb_sha256": decoded_rgb_sha256,
        "provenance_binding_id": "basket_source_visual_provenance",
        "provenance_sha256": provenance_sha256,
    }
    replay_request = _seal(
        {
            "schema_version": REGISTERED_RECEPTACLE_REPLAY_REQUEST_SCHEMA_VERSION,
            "replay_id": "basket-registered-builder-replay",
            "input_bindings": {
                "labels": {"binding_id": "basket_source_labels", "sha256": labels_sha256},
                "collision": {
                    "binding_id": "basket_source_collision",
                    "sha256": collision_sha256,
                },
                "topology": {
                    "binding_id": "basket_collision_topology",
                    "sha256": topology_sha256,
                },
                "visual_review": {
                    "binding_id": "basket_visual_review",
                    "sha256": review_sha256,
                },
                "render_manifest": {
                    "binding_id": "basket_render_manifest",
                    "sha256": render_sha256,
                },
                "semantic_attestation": {
                    "binding_id": "basket_semantic_attestation",
                    "sha256": attestation_sha256,
                },
                "semantic_selection": {
                    "binding_id": "basket_semantic_selection",
                    "sha256": selection_sha256,
                },
            },
            "frame_bindings": [
                {
                    "camera_id": "camera-external",
                    "binding_id": "basket_source_visual_external",
                    "sha256": frame_sha256,
                }
            ],
            "module_sources": module_rows,
            "builder_arguments": builder_arguments,
            "receipt_digest": "",
        }
    )
    replay_relative = "evidence/basket_source/builder_replay_request.json"
    replay_sha256 = _write_json(root / replay_relative, replay_request)
    bindings.append(
        (
            "basket_builder_replay_request",
            "supporting_evidence",
            replay_relative,
            replay_sha256,
            "json",
            REGISTERED_RECEPTACLE_REPLAY_REQUEST_SCHEMA_VERSION,
        )
    )
    authoring = _seal(
        {
            "schema_version": ENGINEERED_ASSET_EVIDENCE_SCHEMA_VERSION,
            "evidence_id": "engineered-asset-evidence:basket",
            "entity_id": "basket",
            "asset_id": "asset:basket",
            "candidate_binding_id": "basket_candidate",
            "candidate_digest": candidate["candidate_digest"],
            "registered_asset_receipt_binding_id": "basket_registered_asset_receipt",
            "registered_asset_receipt_sha256": registered_sha256,
            "topology_receipt_binding_id": "basket_collision_topology",
            "topology_receipt_sha256": topology_sha256,
            "visual_basis_binding_id": "basket_visual_basis",
            "visual_basis_sha256": visual_sha256,
            "semantic_attestation_binding_id": "basket_semantic_attestation",
            "semantic_attestation_sha256": attestation_sha256,
            "semantic_selection_binding_id": "basket_semantic_selection",
            "semantic_selection_sha256": selection_sha256,
            "builder_source_binding_id": "basket_registered_asset_builder",
            "builder_source_sha256": module_rows[0]["sha256"],
            "builder_replay_request_binding_id": "basket_builder_replay_request",
            "builder_replay_request_sha256": replay_sha256,
            "contract_replay_passed": True,
            "all_candidate_files_bound": True,
            "static_asset_structure_readback_passed": True,
            "native_simulator_qualified": False,
            "receipt_digest": "",
        }
    )
    authoring_relative = "evidence/basket_authoring_evidence.json"
    authoring_sha256 = _write_json(root / authoring_relative, authoring)
    bindings.append(
        (
            "basket_authoring_evidence",
            "supporting_evidence",
            authoring_relative,
            authoring_sha256,
            "json",
            ENGINEERED_ASSET_EVIDENCE_SCHEMA_VERSION,
        )
    )
    assert candidate_sha256 == _digest_bytes((root / candidate_relative).read_bytes())
    assert registered_receipt["receipt_digest"] == registered_receipt["receipt_digest"]
    return candidate, authoring, bindings, source_citation


def _placement(
    target_id: str,
    *,
    task_center_m: list[float] | None = None,
    frozen_seed: int = 2026081001,
) -> dict[str, Any]:
    return plan_composed_paired_entity_placement(
        support_regions=[
            {
                "support_region_id": "observed_support",
                "aabb_min_m": [0.0, 0.0, 0.0],
                "aabb_max_m": [3.0, 2.0, 0.0],
                "supports_entities": True,
                "supports_robot_base": True,
            }
        ],
        obstacle_aabbs=[],
        entity_specs=[
            {
                "entity_id": "basket",
                "footprint_xy_m": [0.3, 0.2],
                "height_m": 0.1,
            },
            {
                "entity_id": target_id,
                "footprint_xy_m": [0.2, 0.2],
                "height_m": 0.05,
            },
        ],
        canonical_task_centers_m=[task_center_m or [0.0, 0.0, 0.0]],
        robot_spec={
            "base_footprint_xy_m": [0.4, 0.4],
            "base_clearance_height_m": 0.25,
            "reach_annulus_m": [0.4, 1.6],
        },
        minimum_separations_m={
            "canonical_region": 0.6,
            "entity_entity": 0.1,
            "entity_obstacle": 0.05,
            "robot_entity": 0.05,
            "robot_obstacle": 0.05,
            "support_edge": 0.05,
        },
        grid_spacing_m=0.5,
        frozen_seed=frozen_seed,
    )


def _task_spec(task_kind: str, target_id: str) -> dict[str, Any]:
    if task_kind == "deformable_transfer":
        return {
            "schema_version": "adp_task_spec.v1",
            "task_kind": task_kind,
            "prompt": "Pick up the cloth, place it inside the basket, release it, and retreat.",
            "deformable_entity_id": target_id,
            "destination_entity_id": "basket",
            "robot_entity_id": "franka",
            "destination_interior_obb": {
                "center_world_m": [1.0, 1.0, 0.2],
                "half_extents_m": [0.2, 0.2, 0.2],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "receptacle_reference_pose_world": {
                "position_m": [1.0, 1.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "minimum_particle_fraction_inside": 0.75,
            "settle_window_samples": 3,
            "maximum_node_speed_mps": 0.02,
            "maximum_principal_strain": 0.25,
            "minimum_grasp_contact_force_n": 0.1,
            "maximum_release_contact_force_n": 0.0,
            "minimum_robot_clearance_m": 0.15,
            "maximum_receptacle_translation_drift_m": 0.01,
            "maximum_receptacle_rotation_drift_rad": 0.03,
            "maximum_receptacle_linear_speed_mps": 0.01,
            "maximum_receptacle_angular_speed_radps": 0.03,
            "control_frequency_hz": 15,
            "maximum_action_steps": 32,
        }
    if task_kind == "articulated_open_close":
        return {
            "schema_version": "adp_task_spec.v1",
            "task_kind": task_kind,
            "target_joint_id": "right_door",
            "joint_reset_positions_rad": {"left_door": 0.0, "right_door": 0.0},
            "target_success_interval_rad": [0.785398163, 1.396263402],
            "joint_hard_limits_rad": {
                "left_door": [-0.01, 1.919862177],
                "right_door": [0.0, 1.919862177],
            },
            "settle_window_samples": 3,
            "maximum_settled_target_speed_rad_s": 0.05,
            "non_task_joint_motion_tolerance_rad": 0.001,
            "movement_epsilon_rad": 0.0001,
            "reset_tolerance_rad": 0.0001,
        }
    return {
        "schema_version": "adp_task_spec.v1",
        "task_kind": task_kind,
        "destination_position_world_m": [1.0, 1.0, 0.1],
        "support_plane_z_m": 0.0,
        "settle_window_samples": 40,
        "require_sealed_start_pose": True,
    }


def _entity(
    entity_id: str,
    role: str,
    physics_type: str,
    *,
    source_id: str,
    source_sha256: str,
    source_binding_id: str,
    runtime_origin: str,
    runtime_binding_id: str | None,
    runtime_sha256: str | None,
    authoring_binding_id: str | None = None,
    design_binding_id: str | None = None,
    pending: bool = False,
) -> dict[str, Any]:
    runtime = (
        {
            "origin": "pending_asset_slot",
            "status": "pending_asset_slot",
            "asset_id": None,
            "sha256": None,
            "evidence_binding_id": None,
            "authoring_receipt_binding_id": None,
            "design_basis_observation_binding_id": None,
            "observed_source_truth": False,
            "physical_equivalence_claimed": False,
        }
        if pending
        else {
            "origin": runtime_origin,
            "status": (
                "candidate_ready_pending_native"
                if runtime_origin == "engineered_composed_asset"
                else "ready"
            ),
            "asset_id": f"asset:{entity_id}",
            "sha256": runtime_sha256,
            "evidence_binding_id": runtime_binding_id,
            "authoring_receipt_binding_id": authoring_binding_id,
            "design_basis_observation_binding_id": design_binding_id,
            "observed_source_truth": runtime_origin == "registered_source",
            "physical_equivalence_claimed": False,
        }
    )
    classification = "runtime_embodiment" if role == "robot" else "observed_source"
    return {
        "entity_id": entity_id,
        "semantic_role": role,
        "physics_type": physics_type,
        "source_observation": {
            "classification": classification,
            "source_id": source_id,
            "source_sha256": source_sha256,
            "observed": classification == "observed_source",
            "evidence_binding_id": source_binding_id,
        },
        "runtime_asset": runtime,
    }


def _fixture(tmp_path: Path, task_kind: str = "deformable_transfer") -> dict[str, Any]:
    target_id, target_role, target_physics = _TARGET[task_kind]
    root = tmp_path / "packet"
    root.mkdir()

    placement_receipt = _placement(target_id)
    heldout_placement_receipt = _placement(
        target_id,
        task_center_m=[2.0, 1.0, 0.0],
        frozen_seed=2026081002,
    )
    assert (
        heldout_placement_receipt["selection"]["entity_placements"]
        != placement_receipt["selection"]["entity_placements"]
    )
    basket_placement = next(
        row
        for row in placement_receipt["selection"]["entity_placements"]
        if row["subject_id"] == "basket"
    )
    basket_reference_position = [
        basket_placement["center_world_m"][0],
        basket_placement["center_world_m"][1],
        basket_placement["aabb_min_m"][2],
    ]
    (
        basket_candidate,
        basket_authoring_evidence,
        basket_registered_evidence_bindings,
        basket_source_citation,
    ) = _registered_basket_evidence(
        root,
        reference_position_world_m=basket_reference_position,
    )
    registered_scene_asset_sha = _write_bytes(
        root / "evidence" / "registered_scene.usda",
        b'#usda 1.0\ndef Xform "RegisteredScene" {}\n',
    )
    robot_asset_sha = _write_bytes(
        root / "evidence" / "franka.usda",
        b'#usda 1.0\ndef Xform "Franka" {}\n',
    )
    scorer_source_sha = _write_bytes(
        root / "evidence" / "adp_task_scoring.py",
        (Path(__file__).parents[1] / "src/blueprint_pipeline/adp_task_scoring.py").read_bytes(),
    )
    trust_source_sha = _write_bytes(
        root / "evidence" / "trusted_execution_envelope.py",
        (
            Path(__file__).parents[1] / "src/blueprint_pipeline/trusted_execution_envelope.py"
        ).read_bytes(),
    )

    registration_transform = _seal(
        {
            "schema_version": REGISTRATION_TRANSFORM_SCHEMA_VERSION,
            "evidence_id": "registration-transform:scene-new",
            "scene_id": "scene-new",
            "appearance_source_id": "appearance",
            "collision_source_id": "collision",
            "coordinate_frame_id": "shared-world",
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "appearance_to_collision_matrix_row_major": [
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
            "receipt_digest": "",
        }
    )
    transform_sha = _write_json(
        root / "evidence" / "registration_transform.json",
        registration_transform,
    )
    registration_evidence = _seal(
        {
            "schema_version": REGISTRATION_EVIDENCE_SCHEMA_VERSION,
            "evidence_id": "registration-evidence:scene-new",
            "scene_id": "scene-new",
            "status": "verified_registration",
            "appearance_source_sha256": _sha("1"),
            "collision_source_sha256": _sha("2"),
            "coordinate_frame_id": "shared-world",
            "transform_binding_id": "registration_transform",
            "transform_sha256": transform_sha,
            "receipt_digest": "",
        }
    )
    registration_evidence_sha = _write_json(
        root / "evidence" / "registration_evidence.json", registration_evidence
    )
    topology_survey = _seal(
        {
            "schema_version": TOPOLOGY_SURVEY_SCHEMA_VERSION,
            "evidence_id": "topology-survey:scene-new",
            "scene_id": "scene-new",
            "appearance_source_sha256": _sha("1"),
            "collision_source_sha256": _sha("2"),
            "surveyed_region_ids": ["complete-known-room"],
            "unseen_or_occluded_regions": ["behind-fixed-wall"],
            "completed_at": "2026-08-10T00:00:00Z",
            "receipt_digest": "",
        }
    )
    topology_survey_sha = _write_json(
        root / "evidence" / "topology_survey.json",
        topology_survey,
    )
    topology_evidence = _seal(
        {
            "schema_version": TOPOLOGY_EVIDENCE_SCHEMA_VERSION,
            "evidence_id": "topology-evidence:scene-new",
            "scene_id": "scene-new",
            "survey_binding_id": "topology_survey",
            "survey_sha256": topology_survey_sha,
            "complete_known_topology_surveyed": True,
            "source_observation_limits_recorded": True,
            "unseen_or_occluded_regions": ["behind-fixed-wall"],
            "receipt_digest": "",
        }
    )
    topology_evidence_sha = _write_json(
        root / "evidence" / "topology_evidence.json", topology_evidence
    )

    scene = _seal(
        {
            "schema_version": SCENE_SCHEMA_VERSION,
            "scene_id": "scene-new",
            "status": "frozen",
            "appearance": {
                "source_id": "appearance",
                "source_kind": "metric_gaussian_appearance",
                "revision": "appearance-r1",
                "source_path": "private/source/appearance",
                "size_bytes": 4096,
                "sha256": _sha("1"),
                "coordinate_frame_id": "shared-world",
                "rights_source_id": "interiorgs",
            },
            "collision": {
                "source_id": "collision",
                "source_kind": "registered_collision_geometry",
                "revision": "collision-r1",
                "source_path": "private/source/collision",
                "size_bytes": 2048,
                "sha256": _sha("2"),
                "coordinate_frame_id": "shared-world",
                "rights_source_id": "sage",
            },
            "registration": {
                "status": "passed",
                "shared_coordinates_proved": True,
                "scale_axes_transform_proved": True,
                "transform_sha256": transform_sha,
                "evidence_binding_id": "registration_evidence",
            },
            "topology": {
                "complete_known_topology_surveyed": True,
                "source_observation_limits_recorded": True,
                "unseen_or_occluded_regions": ["behind-fixed-wall"],
                "evidence_binding_id": "topology_evidence",
            },
            "receipt_digest": "",
        }
    )
    scene_sha = _write_json(root / "core" / "scene.json", scene)

    def _resolve_task_spec_for_placement(
        placement: dict[str, Any],
    ) -> dict[str, Any]:
        spec = _task_spec(task_kind, target_id)
        placement_row = next(
            row
            for row in placement["selection"]["entity_placements"]
            if row["subject_id"] == "basket"
        )
        if task_kind == "deformable_transfer":
            basket_geometry = basket_candidate["receptacle_configuration"]["geometry"]
            interior_dimensions = basket_geometry["interior_dimensions_m"]
            floor_thickness = basket_geometry["floor_thickness_m"]
            reference_position = [
                placement_row["center_world_m"][0],
                placement_row["center_world_m"][1],
                placement_row["aabb_min_m"][2],
            ]
            spec["receptacle_reference_pose_world"] = {
                "position_m": reference_position,
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
            spec["destination_interior_obb"] = {
                "center_world_m": [
                    reference_position[0],
                    reference_position[1],
                    reference_position[2] + floor_thickness + interior_dimensions[2] / 2.0,
                ],
                "half_extents_m": [value / 2.0 for value in interior_dimensions],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        elif task_kind == "rigid_pick_place":
            spec["destination_position_world_m"] = [
                float(value) for value in placement_row["center_world_m"]
            ]
        return spec

    frozen_task_spec = _resolve_task_spec_for_placement(placement_receipt)
    heldout_task_spec = _resolve_task_spec_for_placement(heldout_placement_receipt)
    frozen_prompt = (
        frozen_task_spec["prompt"]
        if task_kind == "deformable_transfer"
        else "Manipulate the target into the destination, release, and retreat."
    )
    cell_task_specs = [
        {
            "cell_id": "canonical",
            "task_spec": frozen_task_spec,
            "task_spec_digest": canonical_digest(frozen_task_spec),
        },
        {
            "cell_id": "heldout-composed",
            "task_spec": heldout_task_spec,
            "task_spec_digest": canonical_digest(heldout_task_spec),
        },
    ]
    prompt_task_spec_digest = prompt_task_spec_freeze_digest(
        task_kind=task_kind,
        prompt=frozen_prompt,
        cell_task_spec_digests={row["cell_id"]: row["task_spec_digest"] for row in cell_task_specs},
    )
    task_freeze_public_key = _public_key_bytes(_TASK_FREEZE_PRIVATE_KEY)
    task = {
        "schema_version": TASK_SCHEMA_VERSION,
        "scene_id": "scene-new",
        "task_id": "task-new",
        "task_kind": task_kind,
        "status": "frozen",
        "prompt": frozen_prompt,
        "task_spec_digest": canonical_digest(frozen_task_spec),
        "prompt_task_spec_digest": prompt_task_spec_digest,
        "candidate_ids": ["groot_n17_droid", "pi05_droid"],
        "outcome_blind": True,
        "entities_frozen": True,
        "start_state_frozen": True,
        "destination_frozen": True,
        "controls_frozen": True,
        "seeds_frozen": True,
        "matrix_subset_frozen": True,
        "freeze_authority": {
            "authority_id": "fixture-task-freeze-authority",
            "key_id": "fixture-task-freeze-key",
            "public_key_base64": base64.b64encode(task_freeze_public_key).decode("ascii"),
            "public_key_sha256": _digest_bytes(task_freeze_public_key),
            "signature_base64": "",
        },
        "receipt_digest": "",
    }
    task["freeze_authority"]["signature_base64"] = base64.b64encode(
        _TASK_FREEZE_PRIVATE_KEY.sign(task_freeze_signature_message(task))
    ).decode("ascii")
    _seal(task)
    task_sha = _write_json(root / "core" / "task.json", task)

    rights_supporting_bindings: list[tuple[str, str, str, str, str, str | None]] = []
    rights_verifier_relative = "evidence/rights/task_preinsertion_readiness.py"
    rights_verifier_sha256 = _write_bytes(
        root / rights_verifier_relative,
        (
            Path(__file__).parents[1] / "src/blueprint_pipeline/task_preinsertion_readiness.py"
        ).read_bytes(),
    )
    rights_supporting_bindings.append(
        (
            "rights_interpretation_verifier",
            "supporting_evidence",
            rights_verifier_relative,
            rights_verifier_sha256,
            "opaque",
            None,
        )
    )

    def _rights_receipt(
        *,
        binding_id: str,
        evidence_kind: str,
        subject_id: str,
        document_id: str,
        source_revision: str | None,
        private_derived_processing_permitted: bool,
        raw_upload_permitted: bool,
        provider_retention_permitted: bool,
        provider_training_permitted: bool,
        output_rights_bound: bool,
    ) -> str:
        relative = f"evidence/rights/{binding_id}.json"
        document_binding_id = f"rights_document_{binding_id}"
        document_relative = f"evidence/rights/documents/{binding_id}.txt"
        document_bytes = (
            f"fixture retained rights document\nkind={evidence_kind}\n"
            f"subject={subject_id}\ndocument={document_id}\n"
        ).encode("utf-8")
        document_sha256 = _write_bytes(root / document_relative, document_bytes)
        rights_supporting_bindings.append(
            (
                document_binding_id,
                "supporting_evidence",
                document_relative,
                document_sha256,
                "opaque",
                None,
            )
        )
        public_key = _public_key_bytes(_RIGHTS_PRIVATE_KEY)
        payload = {
            "schema_version": RIGHTS_EVIDENCE_SCHEMA_VERSION,
            "evidence_id": f"rights-evidence:{binding_id}",
            "evidence_kind": evidence_kind,
            "subject_id": subject_id,
            "document_id": document_id,
            "document_binding_id": document_binding_id,
            "document_sha256": document_sha256,
            "document_size_bytes": len(document_bytes),
            "source_revision": source_revision,
            "private_derived_processing_permitted": (private_derived_processing_permitted),
            "raw_upload_permitted": raw_upload_permitted,
            "provider_retention_permitted": provider_retention_permitted,
            "provider_training_permitted": provider_training_permitted,
            "output_rights_bound": output_rights_bound,
            "interpretation_version": RIGHTS_INTERPRETATION_VERSION,
            "verifier_source_binding_id": "rights_interpretation_verifier",
            "verifier_source_sha256": rights_verifier_sha256,
            "authority": {
                "authority_id": "fixture-rights-authority",
                "key_id": "fixture-rights-key",
                "public_key_base64": base64.b64encode(public_key).decode("ascii"),
                "public_key_sha256": _digest_bytes(public_key),
                "signature_base64": "",
            },
            "receipt_digest": "",
        }
        payload["authority"]["signature_base64"] = base64.b64encode(
            _RIGHTS_PRIVATE_KEY.sign(rights_evidence_signature_message(payload))
        ).decode("ascii")
        _seal(payload)
        digest = _write_json(root / relative, payload)
        rights_supporting_bindings.append(
            (
                binding_id,
                "supporting_evidence",
                relative,
                digest,
                "json",
                RIGHTS_EVIDENCE_SCHEMA_VERSION,
            )
        )
        return digest

    rights_source_specs = (
        (
            "franka",
            "runtime-r1",
            "runtime-license",
            "runtime/franka",
            1024,
            _sha("6"),
            "Runtime fixture",
            "runtime_bundled",
            True,
        ),
        (
            "interiorgs",
            "appearance-r1",
            "interiorgs-license",
            "private/source/appearance",
            4096,
            _sha("1"),
            "Interior scene fixture",
            "restricted_nonredistributable",
            False,
        ),
        (
            "sage",
            "collision-r1",
            "sage-license",
            "private/source/collision",
            2048,
            _sha("2"),
            "Collision fixture",
            "restricted_nonredistributable",
            False,
        ),
    )
    rights_sources = []
    for (
        source_id,
        revision,
        license_id,
        source_path,
        size_bytes,
        source_sha256,
        attribution,
        disclosure_class,
        raw_upload_permitted,
    ) in rights_source_specs:
        license_binding_id = f"license_{source_id}"
        output_binding_id = f"output_rights_{source_id}"
        license_sha256 = _rights_receipt(
            binding_id=license_binding_id,
            evidence_kind="source_license",
            subject_id=source_id,
            document_id=license_id,
            source_revision=revision,
            private_derived_processing_permitted=True,
            raw_upload_permitted=raw_upload_permitted,
            provider_retention_permitted=False,
            provider_training_permitted=False,
            output_rights_bound=False,
        )
        output_sha256 = _rights_receipt(
            binding_id=output_binding_id,
            evidence_kind="source_output_rights",
            subject_id=source_id,
            document_id=f"{source_id}-output-rights",
            source_revision=revision,
            private_derived_processing_permitted=True,
            raw_upload_permitted=raw_upload_permitted,
            provider_retention_permitted=False,
            provider_training_permitted=False,
            output_rights_bound=True,
        )
        rights_sources.append(
            {
                "source_id": source_id,
                "revision": revision,
                "license_id": license_id,
                "license_binding_id": license_binding_id,
                "license_sha256": license_sha256,
                "source_path": source_path,
                "size_bytes": size_bytes,
                "sha256": source_sha256,
                "attribution": attribution,
                "disclosure_class": disclosure_class,
                "raw_upload_permitted": raw_upload_permitted,
                "private_derived_processing_permitted": True,
                "provider_retention_permitted": False,
                "provider_training_permitted": False,
                "output_rights_id": f"{source_id}-output-rights",
                "output_rights_binding_id": output_binding_id,
                "output_rights_sha256": output_sha256,
            }
        )
    authority_sha = _rights_receipt(
        binding_id="private_derived_upload_authority",
        evidence_kind="private_derived_processing_authority",
        subject_id="run-private-derived-processing",
        document_id="private-derived-processing-authority",
        source_revision=None,
        private_derived_processing_permitted=True,
        raw_upload_permitted=False,
        provider_retention_permitted=False,
        provider_training_permitted=False,
        output_rights_bound=True,
    )
    provider_terms_sha = _rights_receipt(
        binding_id="provider_terms",
        evidence_kind="provider_terms",
        subject_id="vast",
        document_id="vast-private-processing-terms",
        source_revision=None,
        private_derived_processing_permitted=True,
        raw_upload_permitted=False,
        provider_retention_permitted=False,
        provider_training_permitted=False,
        output_rights_bound=True,
    )
    rights = _seal(
        {
            "schema_version": RIGHTS_SCHEMA_VERSION,
            "status": "admitted",
            "sources": rights_sources,
            "provider_processing": {
                "provider_id": "vast",
                "private_derived_upload_authority_id": ("private-derived-processing-authority"),
                "private_derived_upload_authority_binding_id": ("private_derived_upload_authority"),
                "private_derived_upload_authority_sha256": authority_sha,
                "provider_terms_id": "vast-private-processing-terms",
                "provider_terms_binding_id": "provider_terms",
                "provider_terms_sha256": provider_terms_sha,
            },
            "receipt_digest": "",
        }
    )
    rights_sha = _write_json(root / "core" / "rights.json", rights)

    source_evidence_bindings: list[tuple[str, str, str, str, str, str]] = []
    source_visual_bindings: list[tuple[str, str, str, str, str, str | None]] = []
    source_geometry = {
        target_id: (
            "target-instance",
            {"minimum_m": [0.0, 0.0, 0.0], "maximum_m": [0.2, 0.2, 0.05]},
            "observed_resting_on_counter",
            {"relation": "supported_by", "support_entity_id": "counter"},
        ),
        "basket": (
            "87",
            {"minimum_m": [0.0, 0.0, 0.0], "maximum_m": [0.3, 0.2, 0.1]},
            "observed_resting_on_counter",
            {"relation": "supported_by", "support_entity_id": "counter"},
        ),
        "counter": (
            "counter-instance",
            {"minimum_m": [0.0, 0.0, 0.0], "maximum_m": [3.0, 2.0, 0.1]},
            "static_registered_support",
            {"relation": "static_scene_anchor", "support_entity_id": None},
        ),
        "wall": (
            "wall-instance",
            {"minimum_m": [0.0, 0.0, 0.0], "maximum_m": [0.1, 2.0, 2.0]},
            "static_registered_obstacle",
            {"relation": "static_scene_anchor", "support_entity_id": None},
        ),
        "franka": (
            "franka-runtime-instance",
            {"minimum_m": [0.0, 0.0, 0.0], "maximum_m": [0.6, 0.6, 1.2]},
            "runtime_reset_pose",
            {"relation": "runtime_mount", "support_entity_id": None},
        ),
    }
    visual_source_identity = {
        target_id: ("interiorgs", _sha("1")),
        "counter": ("sage", _sha("2")),
        "wall": ("sage", _sha("2")),
    }
    source_citations: dict[str, list[dict[str, Any]]] = {
        "basket": [basket_source_citation],
        "franka": [],
    }
    for entity_id in (target_id, "counter", "wall"):
        binding_id = f"source_visual_{entity_id}"
        relative = f"evidence/observations/{entity_id}.png"
        image = Image.new("RGB", (4, 3))
        image.putdata([(20 + index * 10, 30 + index * 5, 40 + index * 3) for index in range(12)])
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        image.save(root / relative, format="PNG")
        content = (root / relative).read_bytes()
        digest = _digest_bytes(content)
        with Image.open(root / relative) as decoded:
            decoded_rgb_sha256 = _digest_bytes(decoded.convert("RGB").tobytes())
        source_visual_bindings.append(
            (binding_id, "supporting_evidence", relative, digest, "opaque", None)
        )
        source_id, source_sha256 = visual_source_identity[entity_id]
        source_instance_id = source_geometry[entity_id][0]
        provenance = _seal(
            {
                "schema_version": VISUAL_OBSERVATION_PROVENANCE_SCHEMA_VERSION,
                "evidence_id": f"visual-provenance:{entity_id}",
                "entity_id": entity_id,
                "source_instance_id": source_instance_id,
                "source_id": source_id,
                "source_sha256": source_sha256,
                "coordinate_frame_id": "shared-world",
                "camera_id": "camera-external",
                "frame_binding_id": binding_id,
                "frame_sha256": digest,
                "frame_size_bytes": len(content),
                "width": 4,
                "height": 3,
                "decoded_rgb_sha256": decoded_rgb_sha256,
                "producer_identity": {
                    "kind": "registered_scene_render",
                    "producer": "fixture_renderer",
                    "version": "1.0.0",
                    "configuration_sha256": _sha("e"),
                },
                "receipt_digest": "",
            }
        )
        provenance_binding_id = f"source_visual_provenance_{entity_id}"
        provenance_relative = f"evidence/observations/{entity_id}_provenance.json"
        provenance_sha256 = _write_json(root / provenance_relative, provenance)
        source_visual_bindings.append(
            (
                provenance_binding_id,
                "supporting_evidence",
                provenance_relative,
                provenance_sha256,
                "json",
                VISUAL_OBSERVATION_PROVENANCE_SCHEMA_VERSION,
            )
        )
        source_citations[entity_id] = [
            {
                "binding_id": binding_id,
                "camera_id": "camera-external",
                "sha256": digest,
                "size_bytes": len(content),
                "width": 4,
                "height": 3,
                "decoded_rgb_sha256": decoded_rgb_sha256,
                "provenance_binding_id": provenance_binding_id,
                "provenance_sha256": provenance_sha256,
            }
        ]
    source_evidence_specs = (
        (target_id, "interiorgs", _sha("1"), "observed_source", True, False),
        ("basket", "interiorgs", _sha("1"), "observed_source", True, True),
        ("counter", "sage", _sha("2"), "observed_source", True, False),
        ("wall", "sage", _sha("2"), "observed_source", True, False),
        ("franka", "franka", _sha("6"), "runtime_embodiment", False, False),
    )
    for (
        entity_id,
        source_id,
        source_sha256,
        classification,
        observed,
        design_basis_only,
    ) in source_evidence_specs:
        binding_id = f"source_evidence_{entity_id}"
        source_instance_id, bounds_world, rest_state, support_relation = source_geometry[entity_id]
        receipt = _source_evidence(
            entity_id=entity_id,
            source_id=source_id,
            source_sha256=source_sha256,
            classification=classification,
            observed=observed,
            design_basis_only=design_basis_only,
            source_instance_id=source_instance_id,
            coordinate_frame_id="shared-world",
            bounds_world=bounds_world,
            rest_state=rest_state,
            support_relation=support_relation,
            cited_visual_evidence=source_citations[entity_id],
        )
        relative = f"evidence/{binding_id}.json"
        digest = _write_json(root / relative, receipt)
        source_evidence_bindings.append(
            (
                binding_id,
                "supporting_evidence",
                relative,
                digest,
                "json",
                SOURCE_EVIDENCE_SCHEMA_VERSION,
            )
        )

    entity_rows = [
        _entity(
            target_id,
            target_role,
            target_physics,
            source_id="interiorgs",
            source_sha256=_sha("1"),
            source_binding_id=f"source_evidence_{target_id}",
            runtime_origin="pending_asset_slot",
            runtime_binding_id=None,
            runtime_sha256=None,
            pending=True,
        ),
        _entity(
            "basket",
            "destination_receptacle",
            "rigid_body",
            source_id="interiorgs",
            source_sha256=_sha("1"),
            source_binding_id="source_evidence_basket",
            runtime_origin="engineered_composed_asset",
            runtime_binding_id="basket_candidate",
            runtime_sha256=basket_candidate["candidate_digest"],
            authoring_binding_id="basket_authoring_evidence",
            design_binding_id="source_evidence_basket",
        ),
        _entity(
            "counter",
            "support_surface",
            "static_collider",
            source_id="sage",
            source_sha256=_sha("2"),
            source_binding_id="source_evidence_counter",
            runtime_origin="registered_source",
            runtime_binding_id="registered_scene_asset",
            runtime_sha256=registered_scene_asset_sha,
        ),
        _entity(
            "wall",
            "obstacle",
            "static_collider",
            source_id="sage",
            source_sha256=_sha("2"),
            source_binding_id="source_evidence_wall",
            runtime_origin="registered_source",
            runtime_binding_id="registered_scene_asset",
            runtime_sha256=registered_scene_asset_sha,
        ),
        _entity(
            "franka",
            "robot",
            "robot_articulation",
            source_id="franka",
            source_sha256=_sha("6"),
            source_binding_id="source_evidence_franka",
            runtime_origin="runtime_embodiment",
            runtime_binding_id="robot_asset",
            runtime_sha256=robot_asset_sha,
        ),
    ]
    entities = {
        "schema_version": ENTITY_SCHEMA_VERSION,
        "scene_id": "scene-new",
        "task_id": "task-new",
        "task_kind": task_kind,
        "entities": entity_rows,
        "inventory_digest": "",
    }
    entities["inventory_digest"] = canonical_digest(entities, digest_field="inventory_digest")
    entities_sha = _write_json(root / "core" / "entities.json", entities)

    placement_receipt_sha = _write_json(
        root / "evidence" / "placement_receipt.json", placement_receipt
    )
    heldout_placement_receipt_sha = _write_json(
        root / "evidence" / "heldout_placement_receipt.json",
        heldout_placement_receipt,
    )
    placement = _seal(
        {
            "schema_version": PLACEMENT_SCHEMA_VERSION,
            "scene_id": "scene-new",
            "task_id": "task-new",
            "status": "passed",
            "placements": [
                {
                    "cell_id": "canonical",
                    "receipt_binding_id": "placement_receipt",
                    "receipt_digest": placement_receipt["receipt_digest"],
                    "robot_entity_id": "franka",
                },
                {
                    "cell_id": "heldout-composed",
                    "receipt_binding_id": "heldout_placement_receipt",
                    "receipt_digest": heldout_placement_receipt["receipt_digest"],
                    "robot_entity_id": "franka",
                },
            ],
            "receipt_digest": "",
        }
    )
    placement_sha = _write_json(root / "core" / "placement.json", placement)

    scorer = _seal(
        {
            "schema_version": SCORER_SCHEMA_VERSION,
            "task_id": "task-new",
            "task_kind": task_kind,
            "target_entity_id": target_id,
            "destination_entity_id": ("basket" if task_kind != "articulated_open_close" else None),
            "deterministic": True,
            "policy_self_grading_allowed": False,
            "caller_asserted_outcomes_accepted": False,
            "prompt": frozen_prompt,
            "task_spec": frozen_task_spec,
            "cell_task_specs": cell_task_specs,
            "scorer_source_binding_id": "scorer_source",
            "receipt_digest": "",
        }
    )
    scorer_sha = _write_json(root / "core" / "scorer.json", scorer)

    camera_extrinsics_bindings: list[tuple[str, str, str, str, str, None]] = []
    camera_extrinsics: dict[str, tuple[str, str]] = {}
    for index, role in enumerate(("external", "overview", "wrist")):
        binding_id = f"camera_extrinsics_{role}"
        relative = f"evidence/cameras/{role}.json"
        extrinsics = _seal(
            {
                "schema_version": CAMERA_EXTRINSICS_SCHEMA_VERSION,
                "evidence_id": f"camera-extrinsics:{role}",
                "camera_id": f"camera-{role}",
                "pose_frame": "shared-world",
                "translation_m": [float(index), 0.0, 1.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "calibrated_at": "2026-08-10T00:00:00Z",
                "receipt_digest": "",
            }
        )
        digest = _write_json(root / relative, extrinsics)
        camera_extrinsics[role] = (binding_id, digest)
        camera_extrinsics_bindings.append(
            (
                binding_id,
                "supporting_evidence",
                relative,
                digest,
                "json",
                CAMERA_EXTRINSICS_SCHEMA_VERSION,
            )
        )

    cameras = _seal(
        {
            "schema_version": CAMERA_SCHEMA_VERSION,
            "scene_id": "scene-new",
            "task_id": "task-new",
            "status": "frozen_pending_native_application",
            "native_application_claimed": False,
            "cameras": [
                {
                    "camera_id": f"camera-{role}",
                    "role": role,
                    "policy_input": role in {"external", "wrist"},
                    "review_only": role == "overview",
                    "scoring_input": False,
                    "pose_frame": "shared-world",
                    "extrinsics_binding_id": camera_extrinsics[role][0],
                    "extrinsics_sha256": camera_extrinsics[role][1],
                    "intrinsics": {
                        "fx": 500.0,
                        "fy": 500.0,
                        "cx": 320.0,
                        "cy": 240.0,
                        "width": 640,
                        "height": 480,
                    },
                    "visibility_thresholds": {
                        "minimum_target_fraction": 0.01,
                        "minimum_destination_fraction": 0.01,
                    },
                }
                for role in ("external", "overview", "wrist")
            ],
            "receipt_digest": "",
        }
    )
    cameras_sha = _write_json(root / "core" / "cameras.json", cameras)

    resolved_parameters = _seal(
        {
            "schema_version": RESOLVED_SCENARIO_CELL_SCHEMA_VERSION,
            "evidence_id": "resolved-scenario-cell:canonical",
            "cell_id": "canonical",
            "seed": 2026081001,
            "family": "canonical",
            "resolved_parameters": {
                "placement_variant": "canonical",
                "approach_variant": "canonical",
                "illumination_variant": "canonical",
                "camera_sensor_variant": "canonical",
                "bounded_physics_variant": "canonical",
                "appearance_material_cousin_variant": "canonical",
            },
            "receipt_digest": "",
        }
    )
    resolved_parameters_sha = _write_json(
        root / "evidence" / "canonical_resolved_parameters.json",
        resolved_parameters,
    )
    heldout_resolved_parameters = _seal(
        {
            "schema_version": RESOLVED_SCENARIO_CELL_SCHEMA_VERSION,
            "evidence_id": "resolved-scenario-cell:heldout-composed",
            "cell_id": "heldout-composed",
            "seed": 2026081002,
            "family": "held_out_composed",
            "resolved_parameters": {
                "placement_variant": "heldout-composed",
                "approach_variant": "canonical",
                "illumination_variant": "dimmed-85-percent",
                "camera_sensor_variant": "canonical",
                "bounded_physics_variant": "canonical",
                "appearance_material_cousin_variant": "canonical",
            },
            "receipt_digest": "",
        }
    )
    heldout_resolved_parameters_sha = _write_json(
        root / "evidence" / "heldout_resolved_parameters.json",
        heldout_resolved_parameters,
    )

    scenario = _seal(
        {
            "schema_version": SCENARIO_SCHEMA_VERSION,
            "scene_id": "scene-new",
            "task_id": "task-new",
            "status": "frozen",
            "candidate_ids": ["groot_n17_droid", "pi05_droid"],
            "controls_required_in_every_scored_cell": True,
            "upper_bound_matrix_launched": False,
            "cells": [
                {
                    "cell_id": "canonical",
                    "seed": 2026081001,
                    "family": "canonical",
                    "placement_receipt_digest": placement_receipt["receipt_digest"],
                    "resolved_parameters_binding_id": "canonical_resolved_parameters",
                    "resolved_parameters_sha256": resolved_parameters_sha,
                },
                {
                    "cell_id": "heldout-composed",
                    "seed": 2026081002,
                    "family": "held_out_composed",
                    "placement_receipt_digest": heldout_placement_receipt["receipt_digest"],
                    "resolved_parameters_binding_id": "heldout_resolved_parameters",
                    "resolved_parameters_sha256": heldout_resolved_parameters_sha,
                },
            ],
            "receipt_digest": "",
        }
    )
    scenario_sha = _write_json(root / "core" / "scenario.json", scenario)

    preflight_request, preflight_observations, preflight_matrix = _preflight_fixture(
        root / "evidence" / "preflight_fixture"
    )
    preflight_request_sha = _write_json(
        root / "evidence" / "preflight_request.json", preflight_request
    )
    preflight_observations_sha = _write_json(
        root / "evidence" / "preflight_observations.json", preflight_observations
    )
    preflight_matrix_sha = _write_json(
        root / "evidence" / "preflight_matrix.json", preflight_matrix
    )
    runtime = _seal(
        {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "task_kind": task_kind,
            "status": "static_preflight_passed_dynamic_native_required",
            "request_binding_id": "preflight_request",
            "observations_binding_id": "preflight_observations",
            "matrix_binding_id": "preflight_matrix",
            "native_execution_completed": False,
            "native_qualified": False,
            "scene_run_admitted": False,
            "receipt_digest": "",
        }
    )
    runtime_sha = _write_json(root / "core" / "runtime.json", runtime)

    trust = _seal(
        {
            "schema_version": TRUST_SCHEMA_VERSION,
            "status": "configured_pending_signed_execution",
            "verifier_source_binding_id": "trust_source",
            "envelope_schema_version": "trusted_execution_envelope.v1",
            "runner_public_key_sha256": _sha("b"),
            "signed_return_required": True,
            "configured_key_match_required": True,
            "lifecycle_artifacts_verifier_owned": True,
            "provider_zero_verifier_owned": True,
            "native_execution_claimed": False,
            "receipt_digest": "",
        }
    )
    trust_sha = _write_json(root / "core" / "trust.json", trust)

    binding_rows = [
        ("cameras", "cameras", "core/cameras.json", cameras_sha, "json", CAMERA_SCHEMA_VERSION),
        ("entities", "entities", "core/entities.json", entities_sha, "json", ENTITY_SCHEMA_VERSION),
        (
            "placement",
            "placement",
            "core/placement.json",
            placement_sha,
            "json",
            PLACEMENT_SCHEMA_VERSION,
        ),
        (
            "placement_receipt",
            "supporting_evidence",
            "evidence/placement_receipt.json",
            placement_receipt_sha,
            "json",
            "composed_paired_entity_placement_receipt.v1",
        ),
        (
            "heldout_placement_receipt",
            "supporting_evidence",
            "evidence/heldout_placement_receipt.json",
            heldout_placement_receipt_sha,
            "json",
            "composed_paired_entity_placement_receipt.v1",
        ),
        (
            "registered_scene_asset",
            "supporting_evidence",
            "evidence/registered_scene.usda",
            registered_scene_asset_sha,
            "opaque",
            None,
        ),
        (
            "registration_evidence",
            "supporting_evidence",
            "evidence/registration_evidence.json",
            registration_evidence_sha,
            "json",
            REGISTRATION_EVIDENCE_SCHEMA_VERSION,
        ),
        (
            "registration_transform",
            "supporting_evidence",
            "evidence/registration_transform.json",
            transform_sha,
            "json",
            REGISTRATION_TRANSFORM_SCHEMA_VERSION,
        ),
        ("rights", "rights", "core/rights.json", rights_sha, "json", RIGHTS_SCHEMA_VERSION),
        (
            "robot_asset",
            "supporting_evidence",
            "evidence/franka.usda",
            robot_asset_sha,
            "opaque",
            None,
        ),
        ("runtime", "runtime", "core/runtime.json", runtime_sha, "json", RUNTIME_SCHEMA_VERSION),
        (
            "preflight_matrix",
            "supporting_evidence",
            "evidence/preflight_matrix.json",
            preflight_matrix_sha,
            "json",
            preflight.MATRIX_SCHEMA_VERSION,
        ),
        (
            "preflight_observations",
            "supporting_evidence",
            "evidence/preflight_observations.json",
            preflight_observations_sha,
            "json",
            PREFLIGHT_OBSERVATIONS_SCHEMA_VERSION,
        ),
        (
            "preflight_request",
            "supporting_evidence",
            "evidence/preflight_request.json",
            preflight_request_sha,
            "json",
            PREFLIGHT_REQUEST_SCHEMA_VERSION,
        ),
        (
            "scenario",
            "scenario",
            "core/scenario.json",
            scenario_sha,
            "json",
            SCENARIO_SCHEMA_VERSION,
        ),
        (
            "canonical_resolved_parameters",
            "supporting_evidence",
            "evidence/canonical_resolved_parameters.json",
            resolved_parameters_sha,
            "json",
            RESOLVED_SCENARIO_CELL_SCHEMA_VERSION,
        ),
        (
            "heldout_resolved_parameters",
            "supporting_evidence",
            "evidence/heldout_resolved_parameters.json",
            heldout_resolved_parameters_sha,
            "json",
            RESOLVED_SCENARIO_CELL_SCHEMA_VERSION,
        ),
        ("scene", "scene", "core/scene.json", scene_sha, "json", SCENE_SCHEMA_VERSION),
        ("scorer", "scorer", "core/scorer.json", scorer_sha, "json", SCORER_SCHEMA_VERSION),
        (
            "scorer_source",
            "supporting_evidence",
            "evidence/adp_task_scoring.py",
            scorer_source_sha,
            "opaque",
            None,
        ),
        ("task", "task", "core/task.json", task_sha, "json", TASK_SCHEMA_VERSION),
        ("trust", "trust", "core/trust.json", trust_sha, "json", TRUST_SCHEMA_VERSION),
        (
            "topology_evidence",
            "supporting_evidence",
            "evidence/topology_evidence.json",
            topology_evidence_sha,
            "json",
            TOPOLOGY_EVIDENCE_SCHEMA_VERSION,
        ),
        (
            "topology_survey",
            "supporting_evidence",
            "evidence/topology_survey.json",
            topology_survey_sha,
            "json",
            TOPOLOGY_SURVEY_SCHEMA_VERSION,
        ),
        (
            "trust_source",
            "supporting_evidence",
            "evidence/trusted_execution_envelope.py",
            trust_source_sha,
            "opaque",
            None,
        ),
    ]
    binding_rows.extend(basket_registered_evidence_bindings)
    binding_rows.extend(camera_extrinsics_bindings)
    binding_rows.extend(rights_supporting_bindings)
    binding_rows.extend(source_evidence_bindings)
    binding_rows.extend(source_visual_bindings)
    manifest = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "run_id": "preinsertion-run",
        "scene_id": "scene-new",
        "task_id": "task-new",
        "task_kind": task_kind,
        "candidate_ids": ["groot_n17_droid", "pi05_droid"],
        "asset_slot": {
            "entity_id": target_id,
            "semantic_role": target_role,
            "physics_type": target_physics,
            "status": "unresolved",
            "blocker_code": f"simready_{target_role}_asset_and_native_insertion_required",
        },
        "bindings": [
            {
                "binding_id": binding_id,
                "purpose": purpose,
                "relative_path": relative_path,
                "sha256": sha256,
                "content_type": content_type,
                "schema_version": schema_version,
            }
            for (
                binding_id,
                purpose,
                relative_path,
                sha256,
                content_type,
                schema_version,
            ) in binding_rows
        ],
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    manifest_path = root / "preinsertion_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "root": root,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "scene": scene,
        "task": task,
        "rights": rights,
        "basket_candidate": basket_candidate,
        "basket_authoring_evidence": basket_authoring_evidence,
        "entities": entities,
        "placement": placement,
        "scorer": scorer,
        "runtime": runtime,
        "preflight_matrix": preflight_matrix,
        "trust": trust,
        "cameras": cameras,
        "scenario": scenario,
        "target_id": target_id,
        "target_role": target_role,
    }


def _replace_core_artifact(fixture: dict[str, Any], binding_id: str, value: dict[str, Any]) -> None:
    manifest = deepcopy(fixture["manifest"])
    binding = next(row for row in manifest["bindings"] if row["binding_id"] == binding_id)
    binding["sha256"] = _write_json(fixture["root"] / binding["relative_path"], value)
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    fixture["manifest"] = manifest
    _write_json(fixture["manifest_path"], manifest)


def _resign_task_freeze_for_scorer(
    *, task: dict[str, Any], scorer: dict[str, Any]
) -> dict[str, Any]:
    task = deepcopy(task)
    task["prompt"] = scorer["prompt"]
    task["task_spec_digest"] = canonical_digest(scorer["task_spec"])
    task["prompt_task_spec_digest"] = prompt_task_spec_freeze_digest(
        task_kind=task["task_kind"],
        prompt=task["prompt"],
        cell_task_spec_digests={
            row["cell_id"]: row["task_spec_digest"] for row in scorer["cell_task_specs"]
        },
    )
    task["freeze_authority"]["signature_base64"] = ""
    task["freeze_authority"]["signature_base64"] = base64.b64encode(
        _TASK_FREEZE_PRIVATE_KEY.sign(task_freeze_signature_message(task))
    ).decode("ascii")
    task["receipt_digest"] = canonical_digest(task, digest_field="receipt_digest")
    return task


@pytest.mark.parametrize(
    ("task_kind", "expected_role"),
    [
        ("rigid_pick_place", "movable_rigid"),
        ("articulated_open_close", "articulated_fixture"),
        ("deformable_transfer", "movable_deformable"),
    ],
)
def test_collector_preserves_all_task_families_and_emits_one_typed_slot(
    tmp_path: Path, task_kind: str, expected_role: str
) -> None:
    fixture = _fixture(tmp_path, task_kind)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert receipt["schema_version"] == RECEIPT_SCHEMA_VERSION
    assert receipt["status"] == "preinsertion_ready_one_asset_slot_unresolved"
    assert receipt["all_non_movable_asset_prerequisites_passed"] is True
    assert receipt["blockers"] == []
    assert len(receipt["unresolved_slots"]) == 1
    slot = receipt["unresolved_slots"][0]
    assert slot["semantic_role"] == expected_role
    assert slot["native_qualification_claimed"] is False
    assert slot["dependent_native_gate_ids"]
    if task_kind == "deformable_transfer":
        assert any("deformable" in gate_id for gate_id in slot["dependent_native_gate_ids"])
    else:
        assert all("deformable" not in gate_id for gate_id in slot["dependent_native_gate_ids"])
    assert receipt["claim_boundary"]["native_execution_observed"] is False
    assert receipt["claim_boundary"]["provider_zero_proved"] is False
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")


def test_engineered_asset_is_separate_from_observed_design_basis(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    basket = next(row for row in receipt["entity_lineage"] if row["entity_id"] == "basket")
    assert basket["source_observation_classification"] == "observed_source"
    assert basket["source_observed"] is True
    assert basket["runtime_asset_origin"] == "engineered_composed_asset"
    assert basket["runtime_asset_is_observed_source_truth"] is False
    assert basket["physical_equivalence_claimed"] is False


def test_mapping_only_evidence_has_no_api_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(TypeError):
        collect_task_preinsertion_readiness(fixture["manifest"])  # type: ignore[arg-type]


def test_changed_artifact_bytes_block_before_exposing_slot(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["root"] / "core" / "task.json"
    path.write_bytes(path.read_bytes() + b" ")

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_binding_digest_mismatch:task" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []
    assert receipt["ready_for_asset_insertion"] is False


def test_camera_pose_frame_must_join_registered_scene_frame(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    cameras = deepcopy(fixture["cameras"])
    cameras["cameras"][0]["pose_frame"] = "unregistered-world"
    cameras["receipt_digest"] = canonical_digest(cameras, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "cameras", cameras)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert any(
        blocker.startswith("task_preinsertion_camera_invalid:") for blocker in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


def test_symlinked_artifact_is_rejected_even_when_target_bytes_match(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    core = fixture["root"] / "core"
    task = core / "task.json"
    target = core / "task-real.json"
    task.rename(target)
    try:
        task.symlink_to(target.name)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks unavailable")

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_binding_task_file_invalid" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


def test_duplicate_json_keys_fail_even_with_rebound_file_digest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    task_path = fixture["root"] / "core" / "task.json"
    content = task_path.read_bytes().replace(
        b'{\n  "candidate_ids"', b'{\n  "task_id": "task-new",\n  "candidate_ids"', 1
    )
    task_path.write_bytes(content)
    import hashlib

    manifest = deepcopy(fixture["manifest"])
    binding = next(row for row in manifest["bindings"] if row["binding_id"] == "task")
    binding["sha256"] = "sha256:" + hashlib.sha256(content).hexdigest()
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    _write_json(fixture["manifest_path"], manifest)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_binding_json_invalid:task" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


def test_second_pending_entity_prevents_single_slot_claim(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    entities = deepcopy(fixture["entities"])
    support = next(row for row in entities["entities"] if row["entity_id"] == "counter")
    support["runtime_asset"] = {
        "origin": "pending_asset_slot",
        "status": "pending_asset_slot",
        "asset_id": None,
        "sha256": None,
        "evidence_binding_id": None,
        "authoring_receipt_binding_id": None,
        "design_basis_observation_binding_id": None,
        "observed_source_truth": False,
        "physical_equivalence_claimed": False,
    }
    entities["inventory_digest"] = canonical_digest(entities, digest_field="inventory_digest")
    _replace_core_artifact(fixture, "entities", entities)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_entity_asset_slot_join_invalid" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


def test_engineered_asset_cannot_be_laundered_as_observed_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    entities = deepcopy(fixture["entities"])
    basket = next(row for row in entities["entities"] if row["entity_id"] == "basket")
    basket["runtime_asset"]["observed_source_truth"] = True
    entities["inventory_digest"] = canonical_digest(entities, digest_field="inventory_digest")
    _replace_core_artifact(fixture, "entities", entities)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_entity_engineered_asset_invalid:basket" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


@pytest.mark.parametrize("mutation", ["source_digest", "runtime_asset_digest"])
def test_entity_provenance_and_runtime_bytes_must_join_exact_digests(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    entities = deepcopy(fixture["entities"])
    basket = next(row for row in entities["entities"] if row["entity_id"] == "basket")
    if mutation == "source_digest":
        basket["source_observation"]["source_sha256"] = _sha("c")
        expected = "task_preinsertion_entity_source_invalid:basket"
    else:
        basket["runtime_asset"]["sha256"] = _sha("d")
        expected = "task_preinsertion_entity_engineered_candidate_invalid:basket"
    entities["inventory_digest"] = canonical_digest(entities, digest_field="inventory_digest")
    _replace_core_artifact(fixture, "entities", entities)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert expected in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


@pytest.mark.parametrize(
    ("binding_id", "expected_blocker"),
    [
        (
            "basket_registered_asset_builder",
            "task_preinsertion_entity_engineered_registered_receipt_invalid:basket",
        ),
        ("scorer_source", "task_preinsertion_scorer_contract_invalid"),
        ("trust_source", "task_preinsertion_trust_policy_invalid"),
    ],
)
def test_rebound_substitute_verifier_source_bytes_are_not_accepted(
    tmp_path: Path, binding_id: str, expected_blocker: str
) -> None:
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["manifest"])
    binding = next(row for row in manifest["bindings"] if row["binding_id"] == binding_id)
    content = b"# plausible but not the bound implementation\n"
    binding["sha256"] = _write_bytes(fixture["root"] / binding["relative_path"], content)
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    _write_json(fixture["manifest_path"], manifest)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert expected_blocker in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


def test_unhashable_manifest_candidate_is_a_typed_boundary_error(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["manifest"])
    manifest["candidate_ids"] = [{"candidate": "not-an-id"}, "pi05_droid"]
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    _write_json(fixture["manifest_path"], manifest)

    with pytest.raises(TaskPreinsertionReadinessError) as caught:
        collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_manifest_candidate_ids_invalid" in caught.value.errors


@pytest.mark.parametrize(
    ("artifact_id", "field", "blocker"),
    [
        ("runtime", "native_qualified", "task_preinsertion_runtime_contract_invalid"),
        ("trust", "native_execution_claimed", "task_preinsertion_trust_policy_invalid"),
    ],
)
def test_caller_authored_native_claims_fail_closed(
    tmp_path: Path, artifact_id: str, field: str, blocker: str
) -> None:
    fixture = _fixture(tmp_path)
    value = deepcopy(fixture[artifact_id])
    value[field] = True
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    _replace_core_artifact(fixture, artifact_id, value)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert blocker in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


def test_intermediate_directory_symlink_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["root"]
    core = root / "core"
    moved = root / "core-real"
    core.rename(moved)
    try:
        core.symlink_to(moved.name, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks unavailable")

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert any(
        blocker.startswith("task_preinsertion_binding_") and blocker.endswith("_file_invalid")
        for blocker in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


def test_artifact_binding_snapshot_survives_path_replacement_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    target = fixture["root"] / "core" / "task.json"
    original = target.read_bytes()
    replaced = False
    real_read = os.read

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        result = real_read(descriptor, size)
        if not replaced and result == original:
            replacement = target.with_suffix(".replacement")
            replacement.write_bytes(b'{"schema_version":"attacker"}\n')
            os.replace(replacement, target)
            replaced = True
        return result

    monkeypatch.setattr("blueprint_pipeline.task_preinsertion_readiness.os.read", replacing_read)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert replaced is True
    assert receipt["status"] == "preinsertion_ready_one_asset_slot_unresolved"
    assert len(receipt["unresolved_slots"]) == 1


def test_semantic_and_registered_replays_use_loaded_byte_snapshots_after_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import task_preinsertion_readiness as readiness

    fixture = _fixture(tmp_path)
    replay_request_binding_id = fixture["basket_authoring_evidence"][
        "builder_replay_request_binding_id"
    ]
    replay_request_binding = next(
        row
        for row in fixture["manifest"]["bindings"]
        if row["binding_id"] == replay_request_binding_id
    )
    replay_request = json.loads(
        (fixture["root"] / replay_request_binding["relative_path"]).read_text(encoding="utf-8")
    )
    replaced_binding_ids = {
        str(row["binding_id"]) for row in replay_request["input_bindings"].values()
    }
    replaced_binding_ids.update(str(row["binding_id"]) for row in replay_request["frame_bindings"])
    original_load = readiness._load_bindings
    replacement_count = 0

    def load_then_replace_original_paths(**kwargs: Any) -> Any:
        nonlocal replacement_count
        artifacts, receipts, blockers = original_load(**kwargs)
        for binding_id in sorted(replaced_binding_ids):
            artifact = artifacts[binding_id]
            assert artifact["content"] != b"post-load-attacker-bytes"
            artifact["path"].write_bytes(b"post-load-attacker-bytes")
            replacement_count += 1
        return artifacts, receipts, blockers

    monkeypatch.setattr(readiness, "_load_bindings", load_then_replace_original_paths)

    receipt = readiness.collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert replacement_count == len(replaced_binding_ids)
    assert receipt["status"] == "preinsertion_ready_one_asset_slot_unresolved"
    assert receipt["ready_for_asset_insertion"] is True
    assert receipt["blockers"] == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("task_kind", "task_preinsertion_manifest_task_kind_invalid"),
        (
            "binding_purpose",
            "task_preinsertion_manifest_binding_purpose_invalid:task",
        ),
        (
            "binding_content_type",
            "task_preinsertion_manifest_binding_content_type_invalid:task",
        ),
    ],
)
def test_malformed_json_types_raise_only_typed_boundary_errors(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["manifest"])
    if mutation == "task_kind":
        manifest["task_kind"] = []
    else:
        task_binding = next(row for row in manifest["bindings"] if row["binding_id"] == "task")
        task_binding["purpose" if mutation == "binding_purpose" else "content_type"] = []
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    _write_json(fixture["manifest_path"], manifest)

    with pytest.raises(TaskPreinsertionReadinessError) as caught:
        collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert expected in caught.value.errors


def test_entity_physics_list_is_a_typed_fail_closed_blocker(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    entities = deepcopy(fixture["entities"])
    entities["entities"][0]["physics_type"] = []
    entities["inventory_digest"] = canonical_digest(entities, digest_field="inventory_digest")
    _replace_core_artifact(fixture, "entities", entities)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert any(
        blocker.startswith("task_preinsertion_entity_invalid:") for blocker in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


@pytest.mark.parametrize(
    ("artifact_id", "list_field", "digest_field", "expected_prefix"),
    [
        ("entities", "entities", "inventory_digest", "task_preinsertion_entity_"),
        ("rights", "sources", "receipt_digest", "task_preinsertion_rights_"),
        ("cameras", "cameras", "receipt_digest", "task_preinsertion_camera_"),
        ("scenario", "cells", "receipt_digest", "task_preinsertion_scenario_"),
        (
            "placement",
            "placements",
            "receipt_digest",
            "task_preinsertion_placement_",
        ),
    ],
)
def test_non_mapping_rows_are_rejected_instead_of_silently_discarded(
    tmp_path: Path,
    artifact_id: str,
    list_field: str,
    digest_field: str,
    expected_prefix: str,
) -> None:
    fixture = _fixture(tmp_path)
    artifact = deepcopy(fixture[artifact_id])
    artifact[list_field].append("ignored-attacker-row")
    artifact[digest_field] = canonical_digest(artifact, digest_field=digest_field)
    _replace_core_artifact(fixture, artifact_id, artifact)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert any(blocker.startswith(expected_prefix) for blocker in receipt["blockers"])
    assert receipt["unresolved_slots"] == []


def test_runtime_preflight_rejects_junk_rows_even_when_matrix_is_resealed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    matrix = deepcopy(fixture["preflight_matrix"])
    matrix["static_checks"].append("ignored-attacker-row")
    matrix["receipt_digest"] = canonical_digest(matrix, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "preflight_matrix", matrix)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_runtime_preflight_replay_invalid" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


@pytest.mark.parametrize("field", ["source", "design_basis"])
def test_entity_observation_receipts_cannot_be_rebound_to_unrelated_evidence(
    tmp_path: Path, field: str
) -> None:
    fixture = _fixture(tmp_path)
    entities = deepcopy(fixture["entities"])
    basket = next(row for row in entities["entities"] if row["entity_id"] == "basket")
    if field == "source":
        basket["source_observation"]["evidence_binding_id"] = "trust_source"
        expected = "task_preinsertion_entity_source_invalid:basket"
    else:
        basket["runtime_asset"]["design_basis_observation_binding_id"] = "trust_source"
        expected = "task_preinsertion_entity_engineered_asset_invalid:basket"
    entities["inventory_digest"] = canonical_digest(entities, digest_field="inventory_digest")
    _replace_core_artifact(fixture, "entities", entities)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert expected in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


def test_engineered_candidate_requires_exact_replayed_authoring_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    entities = deepcopy(fixture["entities"])
    basket = next(row for row in entities["entities"] if row["entity_id"] == "basket")
    basket["runtime_asset"]["authoring_receipt_binding_id"] = "trust_source"
    entities["inventory_digest"] = canonical_digest(entities, digest_field="inventory_digest")
    _replace_core_artifact(fixture, "entities", entities)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert (
        "task_preinsertion_entity_engineered_authoring_evidence_invalid:basket"
        in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


def test_engineered_candidate_cannot_rebind_registered_builder_receipt(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    authoring = deepcopy(fixture["basket_authoring_evidence"])
    trust_binding = next(
        row for row in fixture["manifest"]["bindings"] if row["binding_id"] == "trust_source"
    )
    authoring["registered_asset_receipt_binding_id"] = "trust_source"
    authoring["registered_asset_receipt_sha256"] = trust_binding["sha256"]
    authoring["receipt_digest"] = canonical_digest(authoring, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "basket_authoring_evidence", authoring)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert (
        "task_preinsertion_entity_engineered_registered_receipt_invalid:basket"
        in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


def test_registered_receptacle_builder_replay_rejects_changed_arguments(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["manifest"])
    request = json.loads(
        (fixture["root"] / "evidence/basket_source/builder_replay_request.json").read_text(
            encoding="utf-8"
        )
    )
    request["builder_arguments"]["physics_configuration"]["dynamic_friction"] = 0.9
    request["receipt_digest"] = canonical_digest(request, digest_field="receipt_digest")
    request_binding = next(
        row for row in manifest["bindings"] if row["binding_id"] == "basket_builder_replay_request"
    )
    request_binding["sha256"] = _write_json(
        fixture["root"] / request_binding["relative_path"], request
    )
    authoring = deepcopy(fixture["basket_authoring_evidence"])
    authoring["builder_replay_request_sha256"] = request_binding["sha256"]
    authoring["receipt_digest"] = canonical_digest(authoring, digest_field="receipt_digest")
    authoring_binding = next(
        row for row in manifest["bindings"] if row["binding_id"] == "basket_authoring_evidence"
    )
    authoring_binding["sha256"] = _write_json(
        fixture["root"] / authoring_binding["relative_path"], authoring
    )
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    _write_json(fixture["manifest_path"], manifest)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert (
        "task_preinsertion_entity_engineered_registered_replay_invalid:basket"
        in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


@pytest.mark.parametrize(
    "substitution",
    [
        "not_usd",
        "closed_top_cap",
        "tiled_closed_top_cap",
        "hidden_cube_cap",
        "unbound_sublayer",
    ],
)
def test_rehashed_arbitrary_runtime_usd_cannot_become_an_engineered_asset(
    tmp_path: Path, substitution: str
) -> None:
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["manifest"])
    runtime_file_binding = next(
        row for row in manifest["bindings"] if row["binding_id"] == "basket_file_runtime_usd"
    )
    runtime_path = fixture["root"] / runtime_file_binding["relative_path"]
    if substitution == "not_usd":
        substitute = b"this is not an OpenUSD stage\n"
    elif substitution in {"closed_top_cap", "tiled_closed_top_cap", "hidden_cube_cap"}:
        layer = Sdf.Layer.CreateAnonymous("tampered-runtime.usda")
        assert layer.ImportFromString(runtime_path.read_text(encoding="utf-8"))
        stage = Usd.Stage.Open(layer)
        geometry = UsdGeom.Mesh.Get(stage, "/Asset/Geometry")
        assert geometry
        if substitution == "hidden_cube_cap":
            cap = UsdGeom.Cube.Define(stage, "/Asset/HiddenCollisionCap")
            cap.CreateSizeAttr(1.0)
            cap.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.1))
            cap.AddScaleOp().Set(Gf.Vec3d(0.276, 0.184, 0.002))
            UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
        else:
            points = list(geometry.GetPointsAttr().Get())
            counts = list(geometry.GetFaceVertexCountsAttr().Get())
            indices = list(geometry.GetFaceVertexIndicesAttr().Get())
            cap_start = len(points)
            points.extend(
                Gf.Vec3f(x, y, 0.1)
                for x, y in (
                    (-0.138, -0.092),
                    (0.138, -0.092),
                    (0.138, 0.092),
                    (-0.138, 0.092),
                )
            )
            if substitution == "closed_top_cap":
                counts.append(4)
                indices.extend(cap_start + index for index in (0, 1, 2, 3))
            else:
                center = len(points)
                points.append(Gf.Vec3f(0.0, 0.0, 0.1))
                for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
                    counts.append(3)
                    indices.extend((cap_start + first, cap_start + second, center))
            geometry.GetPointsAttr().Set(points)
            geometry.GetFaceVertexCountsAttr().Set(counts)
            geometry.GetFaceVertexIndicesAttr().Set(indices)
        substitute = layer.ExportToString().encode("utf-8")
    else:
        (runtime_path.parent / "unbound.usda").write_bytes(b'#usda 1.0\ndef Xform "Unbound" {}\n')
        substitute = runtime_path.read_bytes().replace(
            b'    defaultPrim = "Asset"',
            b'    defaultPrim = "Asset"\n    subLayers = [@unbound.usda@]',
        )
    substitute_sha = _write_bytes(runtime_path, substitute)
    runtime_file_binding["sha256"] = substitute_sha

    previous = deepcopy(fixture["basket_candidate"])
    runtime_file = next(row for row in previous["files"] if row["role"] == "runtime_usd")
    runtime_file["sha256"] = substitute_sha
    runtime_file["size_bytes"] = len(substitute)
    candidate = materialize_task_entity_asset_candidate(
        {
            key: previous[key]
            for key in (
                "schema_version",
                "entity_id",
                "asset_id",
                "asset_class",
                "source_observation",
                "rights",
                "authoring",
                "files",
                "transform",
                "simulator_import",
                "receptacle_configuration",
                "retained_diagnostic_requirements",
            )
        }
    )
    candidate_binding = next(
        row for row in manifest["bindings"] if row["binding_id"] == "basket_candidate"
    )
    candidate_binding["sha256"] = _write_json(
        fixture["root"] / candidate_binding["relative_path"], candidate
    )

    authoring = deepcopy(fixture["basket_authoring_evidence"])
    authoring["candidate_digest"] = candidate["candidate_digest"]
    authoring["receipt_digest"] = canonical_digest(authoring, digest_field="receipt_digest")
    authoring_binding = next(
        row for row in manifest["bindings"] if row["binding_id"] == "basket_authoring_evidence"
    )
    authoring_binding["sha256"] = _write_json(
        fixture["root"] / authoring_binding["relative_path"], authoring
    )

    entities = deepcopy(fixture["entities"])
    basket = next(row for row in entities["entities"] if row["entity_id"] == "basket")
    basket["runtime_asset"]["sha256"] = candidate["candidate_digest"]
    entities["inventory_digest"] = canonical_digest(entities, digest_field="inventory_digest")
    entities_binding = next(row for row in manifest["bindings"] if row["binding_id"] == "entities")
    entities_binding["sha256"] = _write_json(
        fixture["root"] / entities_binding["relative_path"], entities
    )
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    _write_json(fixture["manifest_path"], manifest)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert (
        "task_preinsertion_entity_engineered_static_structure_invalid:basket" in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


@pytest.mark.parametrize(
    "field",
    ["request_binding_id", "observations_binding_id", "matrix_binding_id"],
)
def test_runtime_static_preflight_cannot_rebind_checks_to_arbitrary_artifacts(
    tmp_path: Path, field: str
) -> None:
    fixture = _fixture(tmp_path)
    runtime = deepcopy(fixture["runtime"])
    runtime[field] = "basket_candidate"
    runtime["receipt_digest"] = canonical_digest(runtime, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "runtime", runtime)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_runtime_preflight_replay_invalid" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("registration", "task_preinsertion_scene_registration_evidence_invalid"),
        ("topology", "task_preinsertion_scene_topology_evidence_invalid"),
        ("rights", "task_preinsertion_rights_not_admitted"),
        ("camera", "task_preinsertion_camera_invalid:external"),
        ("scenario", "task_preinsertion_scenario_cell_invalid:canonical"),
    ],
)
def test_frozen_fact_evidence_cannot_be_rebound_to_unrelated_valid_bytes(
    tmp_path: Path, surface: str, expected: str
) -> None:
    fixture = _fixture(tmp_path)
    if surface in {"registration", "topology"}:
        scene = deepcopy(fixture["scene"])
        scene[surface]["evidence_binding_id"] = "basket_candidate"
        scene["receipt_digest"] = canonical_digest(scene, digest_field="receipt_digest")
        _replace_core_artifact(fixture, "scene", scene)
    elif surface == "rights":
        rights = deepcopy(fixture["rights"])
        provider_terms = next(
            row for row in fixture["manifest"]["bindings"] if row["binding_id"] == "provider_terms"
        )
        rights["provider_processing"]["private_derived_upload_authority_binding_id"] = (
            "provider_terms"
        )
        rights["provider_processing"]["private_derived_upload_authority_sha256"] = provider_terms[
            "sha256"
        ]
        rights["receipt_digest"] = canonical_digest(rights, digest_field="receipt_digest")
        _replace_core_artifact(fixture, "rights", rights)
    elif surface == "camera":
        cameras = deepcopy(fixture["cameras"])
        external = next(row for row in cameras["cameras"] if row["role"] == "external")
        wrist_binding = next(
            row
            for row in fixture["manifest"]["bindings"]
            if row["binding_id"] == "camera_extrinsics_wrist"
        )
        external["extrinsics_binding_id"] = "camera_extrinsics_wrist"
        external["extrinsics_sha256"] = wrist_binding["sha256"]
        cameras["receipt_digest"] = canonical_digest(cameras, digest_field="receipt_digest")
        _replace_core_artifact(fixture, "cameras", cameras)
    else:
        scenario = deepcopy(fixture["scenario"])
        cell = scenario["cells"][0]
        basket_binding = next(
            row
            for row in fixture["manifest"]["bindings"]
            if row["binding_id"] == "basket_candidate"
        )
        cell["resolved_parameters_binding_id"] = "basket_candidate"
        cell["resolved_parameters_sha256"] = basket_binding["sha256"]
        scenario["receipt_digest"] = canonical_digest(scenario, digest_field="receipt_digest")
        _replace_core_artifact(fixture, "scenario", scenario)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert expected in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


def test_rehashed_rights_document_cannot_bypass_signed_interpretation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["manifest"])
    evidence_binding = next(
        row for row in manifest["bindings"] if row["binding_id"] == "license_interiorgs"
    )
    evidence_path = fixture["root"] / evidence_binding["relative_path"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    document_binding = next(
        row for row in manifest["bindings"] if row["binding_id"] == evidence["document_binding_id"]
    )
    substituted_document = b"substituted terms with an attacker-selected interpretation\n"
    substituted_sha256 = _write_bytes(
        fixture["root"] / document_binding["relative_path"],
        substituted_document,
    )
    document_binding["sha256"] = substituted_sha256
    evidence["document_sha256"] = substituted_sha256
    evidence["document_size_bytes"] = len(substituted_document)
    evidence["receipt_digest"] = canonical_digest(evidence, digest_field="receipt_digest")
    evidence_binding["sha256"] = _write_json(evidence_path, evidence)
    rights = deepcopy(fixture["rights"])
    interiorgs = next(row for row in rights["sources"] if row["source_id"] == "interiorgs")
    interiorgs["license_sha256"] = evidence_binding["sha256"]
    rights["receipt_digest"] = canonical_digest(rights, digest_field="receipt_digest")
    rights_binding = next(row for row in manifest["bindings"] if row["binding_id"] == "rights")
    rights_binding["sha256"] = _write_json(
        fixture["root"] / rights_binding["relative_path"], rights
    )
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    _write_json(fixture["manifest_path"], manifest)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_rights_source_invalid:interiorgs" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


def test_signed_denial_of_private_derived_processing_is_not_admitted(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["manifest"])
    rights = deepcopy(fixture["rights"])
    interiorgs = next(row for row in rights["sources"] if row["source_id"] == "interiorgs")
    interiorgs["private_derived_processing_permitted"] = False
    for binding_field, digest_field in (
        ("license_binding_id", "license_sha256"),
        ("output_rights_binding_id", "output_rights_sha256"),
    ):
        binding = next(
            row for row in manifest["bindings"] if row["binding_id"] == interiorgs[binding_field]
        )
        evidence_path = fixture["root"] / binding["relative_path"]
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["private_derived_processing_permitted"] = False
        evidence["authority"]["signature_base64"] = base64.b64encode(
            _RIGHTS_PRIVATE_KEY.sign(rights_evidence_signature_message(evidence))
        ).decode("ascii")
        evidence["receipt_digest"] = canonical_digest(evidence, digest_field="receipt_digest")
        binding["sha256"] = _write_json(evidence_path, evidence)
        interiorgs[digest_field] = binding["sha256"]
    rights["receipt_digest"] = canonical_digest(rights, digest_field="receipt_digest")
    rights_binding = next(row for row in manifest["bindings"] if row["binding_id"] == "rights")
    rights_binding["sha256"] = _write_json(
        fixture["root"] / rights_binding["relative_path"], rights
    )
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    _write_json(fixture["manifest_path"], manifest)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_rights_source_invalid:interiorgs" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


@pytest.mark.parametrize(
    "task_kind",
    ["rigid_pick_place", "articulated_open_close", "deformable_transfer"],
)
def test_frozen_task_prompt_cannot_diverge_from_exact_scorer_task_spec(
    tmp_path: Path, task_kind: str
) -> None:
    fixture = _fixture(tmp_path, task_kind)
    task = deepcopy(fixture["task"])
    task["prompt"] = "Use a different unfrozen prompt for this evaluation."
    task["receipt_digest"] = canonical_digest(task, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "task", task)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_scorer_prompt_freeze_join_invalid" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


@pytest.mark.parametrize(
    "task_kind",
    ["rigid_pick_place", "articulated_open_close", "deformable_transfer"],
)
def test_frozen_task_spec_digest_must_join_scorer_for_every_task_family(
    tmp_path: Path, task_kind: str
) -> None:
    fixture = _fixture(tmp_path, task_kind)
    task = deepcopy(fixture["task"])
    task["task_spec_digest"] = _sha("f")
    task["receipt_digest"] = canonical_digest(task, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "task", task)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_scorer_task_spec_freeze_join_invalid" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


def test_scene_wide_sha_without_entity_visual_citations_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_binding = next(
        row
        for row in fixture["manifest"]["bindings"]
        if row["binding_id"] == "source_evidence_basket"
    )
    source_path = fixture["root"] / source_binding["relative_path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["cited_visual_evidence"] = []
    source["cited_visual_evidence_digest"] = canonical_digest({"cited_visual_evidence": []})
    source["receipt_digest"] = canonical_digest(source, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "source_evidence_basket", source)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_entity_source_invalid:basket" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


def test_engineered_candidate_and_placement_join_observed_entity_bounds(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_binding = next(
        row
        for row in fixture["manifest"]["bindings"]
        if row["binding_id"] == "source_evidence_basket"
    )
    source_path = fixture["root"] / source_binding["relative_path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["bounds_world"]["maximum_m"][0] = 0.31
    source["metric_dimensions_m"][0] = 0.31
    source["receipt_digest"] = canonical_digest(source, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "source_evidence_basket", source)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_entity_engineered_candidate_invalid:basket" in receipt["blockers"]
    assert (
        "task_preinsertion_placement_entity_dimensions_join_invalid:canonical"
        in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


def test_observed_support_relation_must_join_a_support_surface(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    binding_id = f"source_evidence_{fixture['target_id']}"
    source_binding = next(
        row for row in fixture["manifest"]["bindings"] if row["binding_id"] == binding_id
    )
    source_path = fixture["root"] / source_binding["relative_path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["support_relation"]["support_entity_id"] = "wall"
    source["receipt_digest"] = canonical_digest(source, digest_field="receipt_digest")
    _replace_core_artifact(fixture, binding_id, source)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert f"task_preinsertion_entity_source_invalid:{fixture['target_id']}" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


def test_deformable_scorer_destination_must_join_every_placement_cell(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    scorer = deepcopy(fixture["scorer"])
    canonical_spec = next(
        row["task_spec"] for row in scorer["cell_task_specs"] if row["cell_id"] == "canonical"
    )
    heldout_row = next(
        row for row in scorer["cell_task_specs"] if row["cell_id"] == "heldout-composed"
    )
    heldout_row["task_spec"] = deepcopy(canonical_spec)
    heldout_row["task_spec_digest"] = canonical_digest(canonical_spec)
    scorer["receipt_digest"] = canonical_digest(scorer, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "scorer", scorer)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert (
        "task_preinsertion_scorer_destination_placement_join_invalid:heldout-composed"
        in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


@pytest.mark.parametrize("task_kind", ["rigid_pick_place", "articulated_open_close"])
def test_coordinated_unrelated_prompt_reseal_cannot_replace_trusted_task_freeze(
    tmp_path: Path,
    task_kind: str,
) -> None:
    fixture = _fixture(tmp_path, task_kind)
    unrelated_prompt = "Fold all laundry and then turn off the room lights."
    scorer = deepcopy(fixture["scorer"])
    scorer["prompt"] = unrelated_prompt
    scorer["receipt_digest"] = canonical_digest(scorer, digest_field="receipt_digest")
    task = deepcopy(fixture["task"])
    task["prompt"] = unrelated_prompt
    task["prompt_task_spec_digest"] = prompt_task_spec_freeze_digest(
        task_kind=task_kind,
        prompt=unrelated_prompt,
        cell_task_spec_digests={
            row["cell_id"]: row["task_spec_digest"] for row in scorer["cell_task_specs"]
        },
    )
    task["receipt_digest"] = canonical_digest(task, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "scorer", scorer)
    _replace_core_artifact(fixture, "task", task)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_task_freeze_authority_invalid" in receipt["blockers"]
    assert (
        "task_preinsertion_scorer_prompt_task_spec_freeze_join_invalid" not in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


def test_distinct_deformable_placements_have_cell_resolved_destination_specs(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    specs = {row["cell_id"]: row["task_spec"] for row in fixture["scorer"]["cell_task_specs"]}

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert (
        specs["canonical"]["destination_interior_obb"]
        != specs["heldout-composed"]["destination_interior_obb"]
    )
    assert receipt["ready_for_asset_insertion"] is True
    scorer_gate = next(row for row in receipt["prerequisite_gates"] if row["gate_id"] == "scorer")
    assert set(scorer_gate["evidence"]["cell_task_spec_digests"]) == {
        "canonical",
        "heldout-composed",
    }


def test_heldout_composed_cannot_coordinate_reuse_of_canonical_placement(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    canonical_placement = fixture["placement"]["placements"][0]
    placement = deepcopy(fixture["placement"])
    heldout_placement = next(
        row for row in placement["placements"] if row["cell_id"] == "heldout-composed"
    )
    heldout_placement["receipt_binding_id"] = canonical_placement["receipt_binding_id"]
    heldout_placement["receipt_digest"] = canonical_placement["receipt_digest"]
    placement["receipt_digest"] = canonical_digest(placement, digest_field="receipt_digest")

    scorer = deepcopy(fixture["scorer"])
    canonical_spec = deepcopy(scorer["task_spec"])
    heldout_spec = next(
        row for row in scorer["cell_task_specs"] if row["cell_id"] == "heldout-composed"
    )
    heldout_spec["task_spec"] = canonical_spec
    heldout_spec["task_spec_digest"] = canonical_digest(canonical_spec)
    scorer["receipt_digest"] = canonical_digest(scorer, digest_field="receipt_digest")
    task = _resign_task_freeze_for_scorer(task=fixture["task"], scorer=scorer)

    scenario = deepcopy(fixture["scenario"])
    heldout_scenario = next(
        row for row in scenario["cells"] if row["cell_id"] == "heldout-composed"
    )
    heldout_scenario["placement_receipt_digest"] = canonical_placement["receipt_digest"]
    scenario["receipt_digest"] = canonical_digest(scenario, digest_field="receipt_digest")

    _replace_core_artifact(fixture, "placement", placement)
    _replace_core_artifact(fixture, "scorer", scorer)
    _replace_core_artifact(fixture, "task", task)
    _replace_core_artifact(fixture, "scenario", scenario)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert (
        "task_preinsertion_scenario_placement_distinctness_invalid:heldout-composed"
        in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


def test_heldout_composed_rejects_robot_only_placement_difference(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    canonical_receipt_path = fixture["root"] / "evidence/placement_receipt.json"
    canonical_receipt = json.loads(canonical_receipt_path.read_text(encoding="utf-8"))
    robot_only_receipt = _placement(
        fixture["target_id"],
        frozen_seed=2026081127,
    )
    assert (
        robot_only_receipt["selection"]["entity_placements"]
        == canonical_receipt["selection"]["entity_placements"]
    )
    assert (
        robot_only_receipt["selection"]["robot_base_placement"]
        != canonical_receipt["selection"]["robot_base_placement"]
    )
    _replace_core_artifact(
        fixture,
        "heldout_placement_receipt",
        robot_only_receipt,
    )

    placement = deepcopy(fixture["placement"])
    heldout_placement = next(
        row for row in placement["placements"] if row["cell_id"] == "heldout-composed"
    )
    heldout_placement["receipt_digest"] = robot_only_receipt["receipt_digest"]
    placement["receipt_digest"] = canonical_digest(placement, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "placement", placement)

    scorer = deepcopy(fixture["scorer"])
    heldout_spec = next(
        row for row in scorer["cell_task_specs"] if row["cell_id"] == "heldout-composed"
    )
    heldout_spec["task_spec"] = deepcopy(scorer["task_spec"])
    heldout_spec["task_spec_digest"] = canonical_digest(heldout_spec["task_spec"])
    scorer["receipt_digest"] = canonical_digest(scorer, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "scorer", scorer)
    _replace_core_artifact(
        fixture,
        "task",
        _resign_task_freeze_for_scorer(task=fixture["task"], scorer=scorer),
    )

    resolved_binding = next(
        row
        for row in fixture["manifest"]["bindings"]
        if row["binding_id"] == "heldout_resolved_parameters"
    )
    resolved = json.loads(
        (fixture["root"] / resolved_binding["relative_path"]).read_text(encoding="utf-8")
    )
    resolved["seed"] = 2026081127
    resolved["receipt_digest"] = canonical_digest(resolved, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "heldout_resolved_parameters", resolved)
    resolved_binding = next(
        row
        for row in fixture["manifest"]["bindings"]
        if row["binding_id"] == "heldout_resolved_parameters"
    )

    scenario = deepcopy(fixture["scenario"])
    heldout_scenario = next(
        row for row in scenario["cells"] if row["cell_id"] == "heldout-composed"
    )
    heldout_scenario["seed"] = 2026081127
    heldout_scenario["placement_receipt_digest"] = robot_only_receipt["receipt_digest"]
    heldout_scenario["resolved_parameters_sha256"] = resolved_binding["sha256"]
    scenario["receipt_digest"] = canonical_digest(scenario, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "scenario", scenario)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert (
        "task_preinsertion_scenario_task_placement_distinctness_invalid:heldout-composed"
        in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


def test_scenario_seed_must_join_resolved_receipt_and_placement_request(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    rebound_seed = 2026081999
    resolved_binding = next(
        row
        for row in fixture["manifest"]["bindings"]
        if row["binding_id"] == "heldout_resolved_parameters"
    )
    resolved = json.loads(
        (fixture["root"] / resolved_binding["relative_path"]).read_text(encoding="utf-8")
    )
    resolved["seed"] = rebound_seed
    resolved["receipt_digest"] = canonical_digest(resolved, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "heldout_resolved_parameters", resolved)
    resolved_binding = next(
        row
        for row in fixture["manifest"]["bindings"]
        if row["binding_id"] == "heldout_resolved_parameters"
    )

    scenario = deepcopy(fixture["scenario"])
    heldout = next(row for row in scenario["cells"] if row["cell_id"] == "heldout-composed")
    heldout["seed"] = rebound_seed
    heldout["resolved_parameters_sha256"] = resolved_binding["sha256"]
    scenario["receipt_digest"] = canonical_digest(scenario, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "scenario", scenario)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert (
        "task_preinsertion_scenario_placement_seed_join_invalid:heldout-composed"
        in receipt["blockers"]
    )
    assert not any(
        blocker.startswith("task_preinsertion_scenario_cell_invalid:heldout-composed")
        for blocker in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


def test_scenario_family_label_must_match_resolved_parameter_deltas(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["manifest"])
    resolved_binding = next(
        row for row in manifest["bindings"] if row["binding_id"] == "heldout_resolved_parameters"
    )
    resolved_path = fixture["root"] / resolved_binding["relative_path"]
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["resolved_parameters"] = {
        "placement_variant": "canonical",
        "approach_variant": "canonical",
        "illumination_variant": "canonical",
        "camera_sensor_variant": "canonical",
        "bounded_physics_variant": "canonical",
        "appearance_material_cousin_variant": "canonical",
    }
    resolved["receipt_digest"] = canonical_digest(resolved, digest_field="receipt_digest")
    resolved_binding["sha256"] = _write_json(resolved_path, resolved)

    scenario = deepcopy(fixture["scenario"])
    heldout = next(row for row in scenario["cells"] if row["cell_id"] == "heldout-composed")
    heldout["resolved_parameters_sha256"] = resolved_binding["sha256"]
    scenario["receipt_digest"] = canonical_digest(scenario, digest_field="receipt_digest")
    scenario_binding = next(row for row in manifest["bindings"] if row["binding_id"] == "scenario")
    scenario_binding["sha256"] = _write_json(
        fixture["root"] / scenario_binding["relative_path"], scenario
    )
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    _write_json(fixture["manifest_path"], manifest)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert (
        "task_preinsertion_scenario_family_semantics_invalid:heldout-composed"
        in receipt["blockers"]
    )
    assert receipt["unresolved_slots"] == []


def test_self_resealed_observation_png_and_provenance_cannot_replace_signed_semantics(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = deepcopy(fixture["manifest"])
    entity_id = fixture["target_id"]
    source_binding_id = f"source_evidence_{entity_id}"
    source_binding = next(
        row for row in manifest["bindings"] if row["binding_id"] == source_binding_id
    )
    source_path = fixture["root"] / source_binding["relative_path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    original_signature = source["semantic_authority"]["signature_base64"]
    citation = source["cited_visual_evidence"][0]

    frame_binding = next(
        row for row in manifest["bindings"] if row["binding_id"] == citation["binding_id"]
    )
    frame_path = fixture["root"] / frame_binding["relative_path"]
    attacker_image = Image.new("RGB", (4, 3))
    attacker_image.putdata(
        [(230 - index * 7, 10 + index * 11, 80 + index * 5) for index in range(12)]
    )
    attacker_image.save(frame_path, format="PNG")
    frame_bytes = frame_path.read_bytes()
    frame_sha256 = _digest_bytes(frame_bytes)
    with Image.open(frame_path) as decoded:
        decoded_rgb_sha256 = _digest_bytes(decoded.convert("RGB").tobytes())
    frame_binding["sha256"] = frame_sha256

    provenance_binding = next(
        row
        for row in manifest["bindings"]
        if row["binding_id"] == citation["provenance_binding_id"]
    )
    provenance_path = fixture["root"] / provenance_binding["relative_path"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["frame_sha256"] = frame_sha256
    provenance["frame_size_bytes"] = len(frame_bytes)
    provenance["decoded_rgb_sha256"] = decoded_rgb_sha256
    provenance["receipt_digest"] = canonical_digest(provenance, digest_field="receipt_digest")
    provenance_binding["sha256"] = _write_json(provenance_path, provenance)

    citation["sha256"] = frame_sha256
    citation["size_bytes"] = len(frame_bytes)
    citation["decoded_rgb_sha256"] = decoded_rgb_sha256
    citation["provenance_sha256"] = provenance_binding["sha256"]
    source["cited_visual_evidence_digest"] = canonical_digest(
        {"cited_visual_evidence": source["cited_visual_evidence"]}
    )
    source["receipt_digest"] = canonical_digest(source, digest_field="receipt_digest")
    source_binding["sha256"] = _write_json(source_path, source)
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    _write_json(fixture["manifest_path"], manifest)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert source["semantic_authority"]["signature_base64"] == original_signature
    assert f"task_preinsertion_entity_source_invalid:{entity_id}" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


@pytest.mark.parametrize("mutation", ["runtime_embodiment", "unsigned_observed_source"])
def test_non_robot_entities_require_signed_observed_source_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path)
    entities = deepcopy(fixture["entities"])
    basket = next(row for row in entities["entities"] if row["entity_id"] == "basket")
    source_binding_id = basket["source_observation"]["evidence_binding_id"]
    source_binding = next(
        row for row in fixture["manifest"]["bindings"] if row["binding_id"] == source_binding_id
    )
    source = json.loads(
        (fixture["root"] / source_binding["relative_path"]).read_text(encoding="utf-8")
    )
    if mutation == "runtime_embodiment":
        basket["source_observation"].update(
            {
                "classification": "runtime_embodiment",
                "observed": False,
            }
        )
        basket["runtime_asset"].update(
            {
                "origin": "runtime_embodiment",
                "status": "ready",
                "authoring_receipt_binding_id": None,
                "design_basis_observation_binding_id": None,
                "observed_source_truth": False,
            }
        )
        source.update(
            {
                "classification": "runtime_embodiment",
                "observed": False,
                "design_basis_only": False,
                "cited_visual_evidence": [],
                "cited_visual_evidence_digest": canonical_digest({"cited_visual_evidence": []}),
                "semantic_authority": None,
            }
        )
    else:
        source["semantic_authority"] = None
    source["receipt_digest"] = canonical_digest(source, digest_field="receipt_digest")
    _replace_core_artifact(fixture, source_binding_id, source)
    entities["inventory_digest"] = canonical_digest(
        entities,
        digest_field="inventory_digest",
    )
    _replace_core_artifact(fixture, "entities", entities)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_entity_source_invalid:basket" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


def test_placement_robot_identity_must_join_unique_robot_entity(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    placement = deepcopy(fixture["placement"])
    placement["placements"][0]["robot_entity_id"] = "unrelated-robot"
    placement["receipt_digest"] = canonical_digest(placement, digest_field="receipt_digest")
    _replace_core_artifact(fixture, "placement", placement)

    receipt = collect_task_preinsertion_readiness(fixture["manifest_path"])

    assert "task_preinsertion_placement_entity_join_invalid:canonical" in receipt["blockers"]
    assert receipt["unresolved_slots"] == []


def test_manifest_path_with_symlinked_ancestor_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    alias = tmp_path / "packet-alias"
    try:
        alias.symlink_to(fixture["root"], target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks unavailable")

    with pytest.raises(TaskPreinsertionReadinessError) as caught:
        collect_task_preinsertion_readiness(alias / "preinsertion_manifest.json")

    assert "task_preinsertion_manifest_file_invalid" in caught.value.errors
