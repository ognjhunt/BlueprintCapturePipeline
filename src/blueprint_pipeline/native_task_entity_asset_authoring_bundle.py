"""Freeze task-entity assets for one native deformable authoring canary.

The bundle produced here is an immutable *input* to a later Isaac native
canary.  It joins a normalized :mod:`native_task_entity_contract` to normalized
``task_entity_asset_candidate`` records, verifies every staged candidate byte,
and binds the exact Isaac Lab/Arena source revisions and authoring APIs that a
native worker is expected to use.

No simulator is imported or executed here.  A successful bundle therefore
does not prove USD composition, PhysX cooking, contacts, reset behavior,
rendering, or physical cloth equivalence.  The supported deformable is the
released volumetric-FEM path; thin-shell, independent bend/shear, and hidden
attachment claims fail closed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_import_scope import ROBOT_EMBODIMENT_MODULES
from .native_task_entity_contract import (
    SCHEMA_VERSION as TASK_ENTITY_CONTRACT_SCHEMA_VERSION,
)
from .native_task_entity_contract import (
    TASK_KIND_DEFORMABLE_TRANSFER,
    NativeTaskEntityContractError,
    materialize_native_task_entity_contract,
)
from .native_task_runtime_source_packet import (
    ARENA_COMMIT,
    ARENA_REPOSITORY,
    ARENA_TREE,
    ISAACLAB_COMMIT,
    ISAACLAB_REPOSITORY,
    ISAACLAB_TREE,
)
from .task_entity_asset_candidate import (
    DEFORMABLE_REPRESENTATION,
    RIGID_COLLISION_REPRESENTATIONS,
    SCHEMA_VERSION as ASSET_CANDIDATE_SCHEMA_VERSION,
)
from .task_entity_asset_candidate import (
    TaskEntityAssetCandidateError,
    materialize_task_entity_asset_candidate,
)


RUNTIME_IDENTITY_SCHEMA_VERSION = "native_task_entity_asset_authoring_runtime_identity.v1"
INPUT_SCHEMA_VERSION = "native_task_entity_asset_authoring_input.v1"
RECEIPT_SCHEMA_VERSION = "native_task_entity_asset_authoring_bundle_receipt.v1"
BUNDLE_FILENAME = "native_task_entity_asset_authoring_source_bundle.v1.zip"
RECEIPT_FILENAME = "native_task_entity_asset_authoring_bundle_receipt.v1.json"
SOURCE_ROOT_NAME = "native_task_entity_asset_authoring_source"

DEFORMABLE_RUNTIME_CLASS = (
    "isaaclab_physx.assets.deformable_object.deformable_object:DeformableObject"
)
DEFORMABLE_AUTHORING_API = "isaaclab.sim.schemas.schemas:define_deformable_body_properties"
DEFORMABLE_COOKING_API = "omni.physx.scripts.deformableUtils:add_physx_deformable_body"
DEFORMABLE_EXPECTED_PRIM_TYPE = "pxr.OmniPhysicsSchema.OmniPhysicsDeformableBodyAPI"
DEFORMABLE_REQUIRED_SCHEMAS = (
    "pxr.PhysxSchema.PhysxCollisionAPI",
    "pxr.PhysxSchema.PhysxBaseDeformableBodyAPI",
    "pxr.OmniPhysicsSchema.OmniPhysicsDeformableBodyAPI",
    "pxr.OmniPhysicsSchema.OmniPhysicsDeformableMaterialAPI",
    "pxr.PhysxSchema.PhysxDeformableMaterialAPI",
)
RIGID_RUNTIME_CLASS = "isaaclab.assets.rigid_object.rigid_object:RigidObject"
RIGID_AUTHORING_API = "isaaclab.sim.spawners.from_files.from_files:spawn_from_usd"
RIGID_EXPECTED_PRIM_TYPE = "UsdGeom.Xform+collision"

PENDING_NATIVE_GAPS = (
    "native_usd_composition_and_schema_readback",
    "native_physx_deformable_cooking_and_cuda_warp_execution",
    "native_deformable_settling_strain_and_solver_stability",
    "native_genuine_gripper_deformable_contact_lift_and_release",
    "native_nodal_reset_repeatability_and_no_post_start_state_writes",
    "native_receptacle_support_collision_and_no_initial_penetration",
    "native_applied_material_solver_contact_and_pose_parameter_readback",
    "native_external_wrist_and_overview_render_alignment_and_coverage",
)

_DIGEST_PREFIX = "sha256:"


class NativeTaskEntityAssetAuthoringBundleError(ValueError):
    """Stable, sorted failures before a native provider can be considered."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _json_clone(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeTaskEntityAssetAuthoringBundleError([error]) from exc
    if not isinstance(cloned, dict):
        raise NativeTaskEntityAssetAuthoringBundleError([error])
    return cloned


def _valid_digest(value: Any) -> bool:
    text = str(value or "")
    return bool(
        text.startswith(_DIGEST_PREFIX)
        and len(text) == len(_DIGEST_PREFIX) + 64
        and all(character in "0123456789abcdef" for character in text[len(_DIGEST_PREFIX) :])
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _has_symlink_component(path: Path, *, root: Path) -> bool:
    current = root
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    for part in parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _digest_without(value: Mapping[str, Any], *fields: str) -> str:
    normalized = dict(value)
    for field in fields:
        normalized.pop(field, None)
    return canonical_digest(normalized)


def _required_mapping(value: Any, *, error: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(error)
        return {}
    return value


def _required_string(value: Any, *, error: str, errors: list[str]) -> str:
    result = str(value or "").strip()
    if not result:
        errors.append(error)
    return result


def materialize_native_asset_authoring_runtime_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize the exact static runtime binding consumed by the bundle."""

    source = _json_clone(value, error="native_asset_authoring_runtime_identity_invalid")
    errors: list[str] = []
    if source.get("schema_version") != RUNTIME_IDENTITY_SCHEMA_VERSION:
        errors.append("native_asset_authoring_runtime_identity_schema_invalid")

    simulator = _required_mapping(
        source.get("simulator"),
        error="native_asset_authoring_runtime_identity_simulator_missing",
        errors=errors,
    )
    runtime_sources = _required_mapping(
        source.get("runtime_sources"),
        error="native_asset_authoring_runtime_identity_sources_missing",
        errors=errors,
    )
    python = _required_mapping(
        source.get("python"),
        error="native_asset_authoring_runtime_identity_python_missing",
        errors=errors,
    )
    selected_robot = _required_mapping(
        source.get("selected_robot"),
        error="native_asset_authoring_runtime_identity_robot_missing",
        errors=errors,
    )
    bindings = _required_mapping(
        source.get("bindings"),
        error="native_asset_authoring_runtime_identity_bindings_missing",
        errors=errors,
    )
    deformable = _required_mapping(
        bindings.get("deformable_volume"),
        error="native_asset_authoring_runtime_identity_deformable_binding_missing",
        errors=errors,
    )
    receptacle = _required_mapping(
        bindings.get("rigid_receptacle"),
        error="native_asset_authoring_runtime_identity_receptacle_binding_missing",
        errors=errors,
    )

    container_image = _required_string(
        simulator.get("container_image"),
        error="native_asset_authoring_runtime_identity_container_image_missing",
        errors=errors,
    )
    if (
        "@sha256:" not in container_image
        or len(container_image.rsplit("@sha256:", 1)[-1]) != 64
        or not all(
            character in "0123456789abcdef"
            for character in container_image.rsplit("@sha256:", 1)[-1]
        )
    ):
        errors.append("native_asset_authoring_runtime_identity_image_not_digest_pinned")

    expected_sources = {
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
    }
    normalized_sources: dict[str, Any] = {}
    for source_id, expected in expected_sources.items():
        observed = _required_mapping(
            runtime_sources.get(source_id),
            error=f"native_asset_authoring_runtime_identity_source_missing:{source_id}",
            errors=errors,
        )
        normalized_sources[source_id] = {
            key: str(observed.get(key) or "").strip() for key in ("repository", "revision", "tree")
        }
        for key, expected_value in expected.items():
            if normalized_sources[source_id][key] != expected_value:
                errors.append(
                    f"native_asset_authoring_runtime_identity_source_mismatch:{source_id}:{key}"
                )

    source_receipt_digest = str(runtime_sources.get("source_packet_receipt_digest") or "")
    if not _valid_digest(source_receipt_digest):
        errors.append("native_asset_authoring_runtime_identity_source_packet_receipt_invalid")

    robot_id = str(selected_robot.get("robot_id") or "").strip()
    robot_module = str(selected_robot.get("module") or "").strip()
    expected_robot_module = ROBOT_EMBODIMENT_MODULES.get(robot_id)
    if robot_id != "franka_panda" or robot_module != expected_robot_module:
        errors.append("native_asset_authoring_runtime_identity_robot_mismatch")

    required_schemas = sorted(
        str(item).strip() for item in deformable.get("required_schemas") or [] if str(item).strip()
    )
    if required_schemas != sorted(DEFORMABLE_REQUIRED_SCHEMAS):
        errors.append("native_asset_authoring_runtime_identity_schema_set_mismatch")
    if (
        deformable.get("representation") != DEFORMABLE_REPRESENTATION
        or deformable.get("authoring_api") != DEFORMABLE_AUTHORING_API
        or deformable.get("cooking_api") != DEFORMABLE_COOKING_API
        or deformable.get("runtime_class") != DEFORMABLE_RUNTIME_CLASS
        or deformable.get("expected_prim_type") != DEFORMABLE_EXPECTED_PRIM_TYPE
        or deformable.get("thin_shell_supported") is not False
        or deformable.get("independent_bend_shear_supported") is not False
    ):
        errors.append("native_asset_authoring_runtime_identity_deformable_binding_invalid")

    collision_representations = sorted(
        str(item).strip()
        for item in receptacle.get("collision_representations") or []
        if str(item).strip()
    )
    if (
        receptacle.get("authoring_api") != RIGID_AUTHORING_API
        or receptacle.get("runtime_class") != RIGID_RUNTIME_CLASS
        or receptacle.get("expected_prim_type") != RIGID_EXPECTED_PRIM_TYPE
        or receptacle.get("open_interior_required") is not True
        or receptacle.get("top_cap_forbidden") is not True
        or collision_representations != sorted(RIGID_COLLISION_REPRESENTATIONS)
    ):
        errors.append("native_asset_authoring_runtime_identity_receptacle_binding_invalid")

    normalized_python = {
        key: _required_string(
            python.get(key),
            error=f"native_asset_authoring_runtime_identity_python_{key}_missing",
            errors=errors,
        )
        for key in ("python_tag", "abi_tag", "platform_tag")
    }

    if errors:
        raise NativeTaskEntityAssetAuthoringBundleError(errors)

    result: dict[str, Any] = {
        "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
        "runtime_id": _required_string(
            source.get("runtime_id"),
            error="native_asset_authoring_runtime_identity_id_missing",
            errors=errors,
        ),
        "simulator": {
            "name": _required_string(
                simulator.get("name"),
                error="native_asset_authoring_runtime_identity_simulator_name_missing",
                errors=errors,
            ),
            "version": _required_string(
                simulator.get("version"),
                error="native_asset_authoring_runtime_identity_simulator_version_missing",
                errors=errors,
            ),
            "install_root": _required_string(
                simulator.get("install_root"),
                error="native_asset_authoring_runtime_identity_simulator_root_missing",
                errors=errors,
            ),
            "container_image": container_image,
        },
        "runtime_sources": {
            **normalized_sources,
            "source_packet_receipt_digest": source_receipt_digest,
        },
        "python": normalized_python,
        "selected_robot": {"robot_id": robot_id, "module": robot_module},
        "bindings": {
            "deformable_volume": {
                "representation": DEFORMABLE_REPRESENTATION,
                "authoring_api": DEFORMABLE_AUTHORING_API,
                "cooking_api": DEFORMABLE_COOKING_API,
                "runtime_class": DEFORMABLE_RUNTIME_CLASS,
                "expected_prim_type": DEFORMABLE_EXPECTED_PRIM_TYPE,
                "required_schemas": sorted(DEFORMABLE_REQUIRED_SCHEMAS),
                "thin_shell_supported": False,
                "independent_bend_shear_supported": False,
            },
            "rigid_receptacle": {
                "authoring_api": RIGID_AUTHORING_API,
                "runtime_class": RIGID_RUNTIME_CLASS,
                "expected_prim_type": RIGID_EXPECTED_PRIM_TYPE,
                "collision_representations": sorted(RIGID_COLLISION_REPRESENTATIONS),
                "open_interior_required": True,
                "top_cap_forbidden": True,
            },
        },
        "claim_boundary": {
            "static_identity_binding_only": True,
            "native_simulator_qualified": False,
            "thin_shell_cloth_supported": False,
            "physical_material_equivalence": False,
        },
        "runtime_identity_digest": "",
    }
    if errors:
        raise NativeTaskEntityAssetAuthoringBundleError(errors)
    result["runtime_identity_digest"] = canonical_digest(
        result, digest_field="runtime_identity_digest"
    )
    return result


def _verify_runtime_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeTaskEntityAssetAuthoringBundleError(
            ["native_asset_authoring_runtime_identity_missing"]
        )
    normalized = _json_clone(value, error="native_asset_authoring_runtime_identity_invalid")
    raw = dict(normalized)
    raw.pop("runtime_identity_digest", None)
    raw.pop("claim_boundary", None)
    materialized = materialize_native_asset_authoring_runtime_identity(raw)
    if materialized != normalized:
        raise NativeTaskEntityAssetAuthoringBundleError(
            ["native_asset_authoring_runtime_identity_not_normalized"]
        )
    return normalized


def _verify_task_entity_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeTaskEntityAssetAuthoringBundleError(
            ["native_asset_authoring_task_entity_contract_missing"]
        )
    contract = _json_clone(value, error="native_asset_authoring_task_entity_contract_invalid")
    if contract.get("schema_version") != TASK_ENTITY_CONTRACT_SCHEMA_VERSION:
        raise NativeTaskEntityAssetAuthoringBundleError(
            ["native_asset_authoring_task_entity_contract_schema_invalid"]
        )
    try:
        materialized = materialize_native_task_entity_contract(
            task_kind=str(contract.get("task_kind") or ""),
            task_entities=contract.get("task_entities") or [],
        )
    except NativeTaskEntityContractError as exc:
        raise NativeTaskEntityAssetAuthoringBundleError(
            [f"native_asset_authoring_task_entity_contract_invalid:{error}" for error in exc.errors]
        ) from exc
    if materialized != contract:
        raise NativeTaskEntityAssetAuthoringBundleError(
            ["native_asset_authoring_task_entity_contract_not_normalized"]
        )
    if contract["task_kind"] != TASK_KIND_DEFORMABLE_TRANSFER:
        raise NativeTaskEntityAssetAuthoringBundleError(
            ["native_asset_authoring_task_kind_unsupported"]
        )
    return contract


def _verify_candidate(value: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeTaskEntityAssetAuthoringBundleError(
            [f"native_asset_authoring_candidate_invalid:{index}"]
        )
    candidate = _json_clone(value, error=f"native_asset_authoring_candidate_invalid:{index}")
    entity_id = str(candidate.get("entity_id") or index)
    if candidate.get("schema_version") != ASSET_CANDIDATE_SCHEMA_VERSION:
        raise NativeTaskEntityAssetAuthoringBundleError(
            [f"native_asset_authoring_candidate_schema_invalid:{entity_id}"]
        )
    deformable = candidate.get("deformable_configuration")
    if isinstance(deformable, Mapping):
        material = deformable.get("material")
        if (
            deformable.get("representation") != DEFORMABLE_REPRESENTATION
            or not isinstance(material, Mapping)
            or material.get("thin_shell_cloth_claimed") is not False
            or material.get("independent_bend_parameter_available") is not False
            or material.get("independent_shear_parameter_available") is not False
        ):
            raise NativeTaskEntityAssetAuthoringBundleError(
                [f"native_asset_authoring_unsupported_thin_shell_claim:{entity_id}"]
            )

    raw = dict(candidate)
    for field in (
        "status",
        "claims",
        "pending_gates",
        "physically_unresolved",
        "candidate_digest",
    ):
        raw.pop(field, None)
    try:
        materialized = materialize_task_entity_asset_candidate(raw)
    except TaskEntityAssetCandidateError as exc:
        raise NativeTaskEntityAssetAuthoringBundleError(
            [
                f"native_asset_authoring_candidate_invalid:{entity_id}:{error}"
                for error in exc.errors
            ]
        ) from exc
    if materialized != candidate:
        raise NativeTaskEntityAssetAuthoringBundleError(
            [f"native_asset_authoring_candidate_not_normalized:{entity_id}"]
        )
    return candidate


def _candidate_configuration(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    key = (
        "deformable_configuration"
        if candidate["asset_class"] == "deformable_volume"
        else "receptacle_configuration"
    )
    value = candidate.get(key)
    return value if isinstance(value, Mapping) else {}


def _runtime_usd(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = [
        row
        for row in candidate.get("files") or []
        if isinstance(row, Mapping) and row.get("role") == "runtime_usd"
    ]
    return rows[0] if len(rows) == 1 else {}


def _validate_entity_candidate_join(
    *,
    entity: Mapping[str, Any],
    candidate: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    entity_id = str(entity["entity_id"])
    role = entity["semantic_role"]
    expected = {
        "movable_deformable": ("deformable_volume", "deformable_volume"),
        "destination_receptacle": ("static_collider", "rigid_receptacle"),
    }
    expected_physics, expected_class = expected[role]
    if role == "destination_receptacle" and entity["physics_type"] == "rigid_body":
        expected_physics = "rigid_body"
    if entity["physics_type"] != expected_physics or candidate["asset_class"] != expected_class:
        errors.append(f"native_asset_authoring_entity_candidate_class_mismatch:{entity_id}")

    runtime_asset = entity["runtime_asset"]
    if (
        runtime_asset["binding_kind"] != "usd_asset"
        or entity["replacement_policy"]["action"] != "insert_runtime_asset"
        or entity["replacement_policy"]["replacement_required"] is not True
    ):
        errors.append(f"native_asset_authoring_entity_not_runtime_insert:{entity_id}")
    if runtime_asset["asset_id"] != candidate["asset_id"]:
        errors.append(f"native_asset_authoring_asset_id_mismatch:{entity_id}")
    runtime_usd = _runtime_usd(candidate)
    if not runtime_usd or runtime_asset["sha256"] != runtime_usd.get("sha256"):
        errors.append(f"native_asset_authoring_runtime_usd_mismatch:{entity_id}")
    if (
        entity["source_observation"]["observation_id"]
        != candidate["source_observation"]["observation_id"]
        or entity["source_observation"]["source_reference"]
        != candidate["source_observation"]["source_reference"]
        or entity["source_observation"]["source_sha256"]
        != candidate["source_observation"]["source_sha256"]
    ):
        errors.append(f"native_asset_authoring_source_observation_mismatch:{entity_id}")
    if entity["digests"]["configuration_sha256"] != canonical_digest(
        _candidate_configuration(candidate)
    ):
        errors.append(f"native_asset_authoring_configuration_mismatch:{entity_id}")

    rights = candidate["rights"]
    provenance = entity["provenance"]
    if (
        rights["derived_asset_private_upload_permitted"] is not True
        or provenance["upload_permitted"] is not True
    ):
        errors.append(f"native_asset_authoring_derived_upload_not_permitted:{entity_id}")
    provenance_join = {
        "source_revision": "source_revision",
        "license_id": "license_id",
        "derived_processing_authority_id": "derived_processing_authority_id",
        "provider_terms_id": "provider_terms_id",
        "output_rights_id": "output_rights_id",
        "attribution": "attribution",
    }
    if any(
        rights[rights_key] != provenance[provenance_key]
        for rights_key, provenance_key in provenance_join.items()
    ) or (candidate["source_observation"]["source_size_bytes"] != provenance["source_size_bytes"]):
        errors.append(f"native_asset_authoring_provenance_mismatch:{entity_id}")
    if (
        rights["raw_redistribution_permitted"] != provenance["raw_redistribution_permitted"]
        or rights["provider_retention_permitted"] != provenance["provider_retention_permitted"]
        or rights["provider_training_permitted"] != provenance["provider_training_permitted"]
        or rights["provider_training_permitted"] is not False
    ):
        errors.append(f"native_asset_authoring_provider_terms_unadmitted:{entity_id}")

    simulator_import = candidate["simulator_import"]
    binding_key = (
        "deformable_volume"
        if candidate["asset_class"] == "deformable_volume"
        else "rigid_receptacle"
    )
    binding = runtime_identity["bindings"][binding_key]
    if (
        simulator_import["simulator"] != runtime_identity["simulator"]["name"]
        or simulator_import["simulator_version"] != runtime_identity["simulator"]["version"]
        or simulator_import["source_repository"] != ISAACLAB_REPOSITORY
        or simulator_import["source_revision"] != ISAACLAB_COMMIT
        or simulator_import["importer_module"] != binding["runtime_class"]
        or simulator_import["expected_prim_type"] != binding["expected_prim_type"]
    ):
        errors.append(f"native_asset_authoring_simulator_import_mismatch:{entity_id}")

    if candidate["asset_class"] == "rigid_receptacle":
        geometry = candidate["receptacle_configuration"]["geometry"]
        if geometry["open_interior"] is not True or geometry["top_cap_present"] is not False:
            errors.append(f"native_asset_authoring_receptacle_not_open:{entity_id}")
    return errors


def _operation_for(*, entity: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    if candidate["asset_class"] == "deformable_volume":
        configuration = candidate["deformable_configuration"]
        topology = configuration["rest_topology"]
        topology_stage = topology.get("topology_stage", "explicit_tetrahedral_mesh")
        return {
            "operation_kind": (
                "cook_closed_surface_to_volumetric_fem_candidate"
                if topology_stage == "surface_mesh_pending_native_cook"
                else "compose_closed_volumetric_fem_candidate"
            ),
            "authoring_api": DEFORMABLE_AUTHORING_API,
            "cooking_api": DEFORMABLE_COOKING_API,
            "runtime_class": DEFORMABLE_RUNTIME_CLASS,
            "expected_prim_type": DEFORMABLE_EXPECTED_PRIM_TYPE,
            "required_schemas": sorted(DEFORMABLE_REQUIRED_SCHEMAS),
            "configuration": configuration,
            "topology_stage": topology_stage,
            "native_topology_readback_required": True,
            "candidate_authored_transform": candidate["transform"],
            "initial_pose_world": entity["initial_state"]["pose_world"],
            "native_schema_and_parameter_readback_required": True,
            "thin_shell_cloth_claimed": False,
            "hidden_kinematic_attachment_allowed": False,
        }
    return {
        "operation_kind": "compose_open_rigid_receptacle_candidate",
        "authoring_api": RIGID_AUTHORING_API,
        "runtime_class": RIGID_RUNTIME_CLASS,
        "expected_prim_type": RIGID_EXPECTED_PRIM_TYPE,
        "configuration": candidate["receptacle_configuration"],
        "candidate_authored_transform": candidate["transform"],
        "initial_pose_world": entity["initial_state"]["pose_world"],
        "native_collision_pose_and_support_readback_required": True,
        "open_interior_required": True,
        "top_cap_forbidden": True,
    }


def build_native_task_entity_asset_authoring_bundle(
    *,
    output_dir: str | Path,
    task_entity_contract: Mapping[str, Any],
    asset_candidates: Sequence[Mapping[str, Any]],
    asset_source_roots: Mapping[str, str | Path],
    runtime_identity: Mapping[str, Any] | None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Create one deterministic pre-canary source bundle and receipt."""

    contract = _verify_task_entity_contract(task_entity_contract)
    identity = _verify_runtime_identity(runtime_identity)
    if (
        isinstance(asset_candidates, (str, bytes, Mapping))
        or not isinstance(asset_candidates, Sequence)
        or not asset_candidates
    ):
        raise NativeTaskEntityAssetAuthoringBundleError(
            ["native_asset_authoring_candidates_invalid"]
        )
    candidates = [
        _verify_candidate(value, index=index) for index, value in enumerate(asset_candidates)
    ]
    if not isinstance(asset_source_roots, Mapping):
        raise NativeTaskEntityAssetAuthoringBundleError(
            ["native_asset_authoring_source_roots_invalid"]
        )
    by_candidate_id: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for candidate in candidates:
        entity_id = candidate["entity_id"]
        if entity_id in by_candidate_id:
            errors.append(f"native_asset_authoring_candidate_duplicate:{entity_id}")
        by_candidate_id[entity_id] = candidate

    supported_entities = {
        entity["entity_id"]: entity
        for entity in contract["task_entities"]
        if entity["semantic_role"] in {"movable_deformable", "destination_receptacle"}
    }
    if set(supported_entities) != set(by_candidate_id):
        for entity_id in sorted(set(supported_entities) - set(by_candidate_id)):
            errors.append(f"native_asset_authoring_candidate_missing:{entity_id}")
        for entity_id in sorted(set(by_candidate_id) - set(supported_entities)):
            errors.append(f"native_asset_authoring_candidate_unbound:{entity_id}")
    asset_classes = {candidate["asset_class"] for candidate in candidates}
    for asset_class in ("deformable_volume", "rigid_receptacle"):
        if asset_class not in asset_classes:
            errors.append(f"native_asset_authoring_asset_class_missing:{asset_class}")
    if set(asset_source_roots) != set(by_candidate_id):
        errors.append("native_asset_authoring_source_root_set_mismatch")

    for entity_id in sorted(set(supported_entities) & set(by_candidate_id)):
        errors.extend(
            _validate_entity_candidate_join(
                entity=supported_entities[entity_id],
                candidate=by_candidate_id[entity_id],
                runtime_identity=identity,
            )
        )
    if errors:
        raise NativeTaskEntityAssetAuthoringBundleError(errors)

    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise NativeTaskEntityAssetAuthoringBundleError(["native_asset_authoring_output_exists"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        source_root = staging / SOURCE_ROOT_NAME
        source_root.mkdir()
        asset_rows_by_entity: dict[str, list[dict[str, Any]]] = {}
        for entity_id, candidate in sorted(by_candidate_id.items()):
            raw_root = Path(asset_source_roots[entity_id]).expanduser()
            if raw_root.is_symlink():
                raise NativeTaskEntityAssetAuthoringBundleError(
                    [f"native_asset_authoring_source_root_invalid:{entity_id}"]
                )
            asset_root = raw_root.resolve()
            if not asset_root.is_dir():
                raise NativeTaskEntityAssetAuthoringBundleError(
                    [f"native_asset_authoring_source_root_invalid:{entity_id}"]
                )
            rows: list[dict[str, Any]] = []
            for file_record in candidate["files"]:
                relative = PurePosixPath(file_record["path"])
                source = asset_root / Path(*relative.parts)
                if (
                    _has_symlink_component(source, root=asset_root)
                    or not source.is_file()
                    or source.stat().st_size != file_record["size_bytes"]
                    or _sha256(source) != file_record["sha256"]
                ):
                    raise NativeTaskEntityAssetAuthoringBundleError(
                        [
                            "native_asset_authoring_source_file_identity_mismatch:"
                            f"{entity_id}:{file_record['role']}"
                        ]
                    )
                archive_relative = PurePosixPath("candidate_assets", entity_id, *relative.parts)
                destination = source_root / Path(*archive_relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if (
                    destination.stat().st_size != file_record["size_bytes"]
                    or _sha256(destination) != file_record["sha256"]
                ):
                    raise NativeTaskEntityAssetAuthoringBundleError(
                        [
                            "native_asset_authoring_staged_file_identity_mismatch:"
                            f"{entity_id}:{file_record['role']}"
                        ]
                    )
                rows.append(
                    {
                        "role": file_record["role"],
                        "archive_relative_path": archive_relative.as_posix(),
                        "size_bytes": file_record["size_bytes"],
                        "sha256": file_record["sha256"],
                    }
                )
            asset_rows_by_entity[entity_id] = sorted(
                rows, key=lambda row: (row["role"], row["archive_relative_path"])
            )

        authoring_plans = []
        for entity_id, entity in sorted(supported_entities.items()):
            candidate = by_candidate_id[entity_id]
            authoring_plans.append(
                {
                    "entity_id": entity_id,
                    "semantic_role": entity["semantic_role"],
                    "physics_type": entity["physics_type"],
                    "asset_id": candidate["asset_id"],
                    "candidate_digest": candidate["candidate_digest"],
                    "candidate_record": candidate,
                    "staged_files": asset_rows_by_entity[entity_id],
                    "operation": _operation_for(entity=entity, candidate=candidate),
                }
            )

        manifest: dict[str, Any] = {
            "schema_version": INPUT_SCHEMA_VERSION,
            "generated_at": generated_at or utc_now_iso(),
            "program_id": "arm-decision-proof-v1",
            "status": "ready_for_native_authoring_canary",
            "task_kind": contract["task_kind"],
            "task_entity_contract_digest": contract["contract_digest"],
            "runtime_identity": identity,
            "runtime_identity_digest": identity["runtime_identity_digest"],
            "entity_authoring_plans": authoring_plans,
            "asset_entity_ids": sorted(supported_entities),
            "raw_dataset_source_bytes_included": False,
            "derived_candidate_asset_bytes_included": True,
            "candidate_policy_queried": False,
            "candidate_outcomes_accessed": False,
            "native_simulator_executed": False,
            "native_qualification_claimed": False,
            "thin_shell_cloth_claimed": False,
            "physical_material_equivalence_claimed": False,
            "pending_native_gaps": list(PENDING_NATIVE_GAPS),
            "input_digest": "",
        }
        manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
        manifest_path = source_root / "native_task_entity_asset_authoring_input.v1.json"
        write_json(manifest_path, manifest)

        bundle_path = staging / BUNDLE_FILENAME
        with zipfile.ZipFile(bundle_path, "w", allowZip64=True) as archive:
            for path in sorted(source_root.rglob("*")):
                if not path.is_file():
                    continue
                relative = PurePosixPath(SOURCE_ROOT_NAME, path.relative_to(source_root).as_posix())
                info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (0o100644 & 0xFFFF) << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_STORED)

        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "ready_for_native_authoring_canary",
            "bundle_name": BUNDLE_FILENAME,
            "bundle_path": str(output / BUNDLE_FILENAME),
            "bundle_size_bytes": bundle_path.stat().st_size,
            "bundle_sha256": _sha256(bundle_path),
            "input_digest": manifest["input_digest"],
            "task_entity_contract_digest": contract["contract_digest"],
            "runtime_identity_digest": identity["runtime_identity_digest"],
            "asset_candidate_digests": {
                entity_id: by_candidate_id[entity_id]["candidate_digest"]
                for entity_id in sorted(by_candidate_id)
            },
            "raw_dataset_source_bytes_included": False,
            "native_simulator_executed": False,
            "native_qualification_claimed": False,
            "pending_native_gaps": list(PENDING_NATIVE_GAPS),
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = _digest_without(receipt, "bundle_path", "receipt_digest")
        write_json(staging / RECEIPT_FILENAME, receipt)
        shutil.rmtree(source_root)
        staging.replace(output)
        return receipt
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_native_task_entity_asset_authoring_bundle(
    receipt_path: str | Path,
    *,
    expected_task_entity_contract_digest: str | None = None,
    expected_runtime_identity_digest: str | None = None,
) -> dict[str, Any]:
    """Reverify a frozen source bundle without extracting or executing it."""

    path = Path(receipt_path).expanduser().resolve()
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeTaskEntityAssetAuthoringBundleError(
            ["native_asset_authoring_receipt_invalid"]
        ) from exc
    if not isinstance(receipt, Mapping):
        raise NativeTaskEntityAssetAuthoringBundleError(["native_asset_authoring_receipt_invalid"])
    receipt = dict(receipt)
    errors: list[str] = []
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "ready_for_native_authoring_canary"
        or receipt.get("bundle_name") != BUNDLE_FILENAME
        or receipt.get("native_simulator_executed") is not False
        or receipt.get("native_qualification_claimed") is not False
        or receipt.get("raw_dataset_source_bytes_included") is not False
        or receipt.get("pending_native_gaps") != list(PENDING_NATIVE_GAPS)
    ):
        errors.append("native_asset_authoring_receipt_contract_invalid")
    if receipt.get("receipt_digest") != _digest_without(receipt, "bundle_path", "receipt_digest"):
        errors.append("native_asset_authoring_receipt_digest_invalid")
    if (
        expected_task_entity_contract_digest
        and receipt.get("task_entity_contract_digest") != expected_task_entity_contract_digest
    ):
        errors.append("native_asset_authoring_task_entity_contract_digest_mismatch")
    if (
        expected_runtime_identity_digest
        and receipt.get("runtime_identity_digest") != expected_runtime_identity_digest
    ):
        errors.append("native_asset_authoring_runtime_identity_digest_mismatch")

    bundle_path = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    try:
        observed_size = bundle_path.stat().st_size
        observed_digest = _sha256(bundle_path)
    except OSError:
        observed_size = -1
        observed_digest = ""
    if observed_size != receipt.get("bundle_size_bytes") or observed_digest != receipt.get(
        "bundle_sha256"
    ):
        errors.append("native_asset_authoring_bundle_bytes_identity_mismatch")
    if errors:
        raise NativeTaskEntityAssetAuthoringBundleError(errors)
    return receipt


__all__ = [
    "BUNDLE_FILENAME",
    "DEFORMABLE_AUTHORING_API",
    "DEFORMABLE_COOKING_API",
    "DEFORMABLE_EXPECTED_PRIM_TYPE",
    "DEFORMABLE_REQUIRED_SCHEMAS",
    "DEFORMABLE_RUNTIME_CLASS",
    "INPUT_SCHEMA_VERSION",
    "NativeTaskEntityAssetAuthoringBundleError",
    "PENDING_NATIVE_GAPS",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA_VERSION",
    "RIGID_AUTHORING_API",
    "RIGID_EXPECTED_PRIM_TYPE",
    "RIGID_RUNTIME_CLASS",
    "RUNTIME_IDENTITY_SCHEMA_VERSION",
    "build_native_task_entity_asset_authoring_bundle",
    "materialize_native_asset_authoring_runtime_identity",
    "verify_native_task_entity_asset_authoring_bundle",
]
