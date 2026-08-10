"""Normalize task-neutral native-runtime entities without importing a simulator.

The native runtime historically accepted one asset with the semantic role
``task_object``.  That alias cannot describe a deformable, its receptacle, the
supporting scene, obstacles, and the robot independently.  This module defines
the pure-data boundary that later packet and runtime changes can consume.

Validation here proves only that a request is complete and internally
consistent.  Native composition, contact behavior, state application, and
readback remain separate simulator gates.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "native_task_entity_contract.v1"

TASK_KIND_RIGID_PICK_PLACE = "rigid_pick_place"
TASK_KIND_ARTICULATED_OPEN_CLOSE = "articulated_open_close"
TASK_KIND_DEFORMABLE_TRANSFER = "deformable_transfer"

TASK_KINDS = (
    TASK_KIND_RIGID_PICK_PLACE,
    TASK_KIND_ARTICULATED_OPEN_CLOSE,
    TASK_KIND_DEFORMABLE_TRANSFER,
)

SEMANTIC_ROLES = (
    "movable_rigid",
    "articulated_fixture",
    "movable_deformable",
    "destination_receptacle",
    "support_surface",
    "obstacle",
    "robot",
)

_LEGACY_TASK_OBJECT_ROLE_BY_TASK_KIND = {
    TASK_KIND_RIGID_PICK_PLACE: "movable_rigid",
    TASK_KIND_ARTICULATED_OPEN_CLOSE: "articulated_fixture",
}

_REQUIRED_ROLES_BY_TASK_KIND = {
    TASK_KIND_RIGID_PICK_PLACE: frozenset({"movable_rigid", "robot"}),
    TASK_KIND_ARTICULATED_OPEN_CLOSE: frozenset(
        {"articulated_fixture", "robot"}
    ),
    TASK_KIND_DEFORMABLE_TRANSFER: frozenset(
        {
            "movable_deformable",
            "destination_receptacle",
            "support_surface",
            "obstacle",
            "robot",
        }
    ),
}

_PHYSICS_TYPES_BY_ROLE = {
    "movable_rigid": frozenset({"rigid_body"}),
    "articulated_fixture": frozenset({"articulation"}),
    "movable_deformable": frozenset({"deformable_volume"}),
    "destination_receptacle": frozenset({"rigid_body", "static_collider"}),
    "support_surface": frozenset({"rigid_body", "static_collider"}),
    "obstacle": frozenset({"rigid_body", "articulation", "static_collider"}),
    "robot": frozenset({"robot_articulation"}),
}

_RESET_KINDS_BY_PHYSICS_TYPE = {
    "rigid_body": frozenset({"native_rigid_state"}),
    "articulation": frozenset({"native_articulation_state"}),
    "deformable_volume": frozenset({"native_deformable_state"}),
    "static_collider": frozenset({"immutable_scene_state"}),
    "robot_articulation": frozenset({"native_robot_state"}),
}

_CONTACT_KIND_BY_ROLE = {
    "movable_rigid": "manipulated_rigid",
    "articulated_fixture": "manipulated_articulation",
    "movable_deformable": "manipulated_deformable",
    "destination_receptacle": "destination_volume",
    "support_surface": "supporting_surface",
    "obstacle": "collision_obstacle",
    "robot": "manipulator",
}

_SCORING_KIND_BY_ROLE = {
    "movable_rigid": "movable_target",
    "articulated_fixture": "articulated_target",
    "movable_deformable": "deformable_target",
    "destination_receptacle": "destination",
    "support_surface": "support_context",
    "obstacle": "collision_context",
    "robot": "robot_context",
}

_SOURCE_KINDS = frozenset(
    {
        "observed_dataset_entity",
        "registered_scene_geometry",
        "runtime_embodiment",
        "legacy_runtime_contract",
        "generated_runtime_asset",
    }
)
_ASSET_BINDING_KINDS = frozenset(
    {"usd_asset", "registered_scene_geometry", "runtime_embodiment"}
)
_DISCLOSURE_CLASSES = frozenset(
    {
        "public_redistributable",
        "restricted_private_processing",
        "runtime_bundled",
        "generated_derivative",
    }
)
_SOURCE_ENTITY_ACTIONS = frozenset({"retain", "remove", "not_present"})
_GAUSSIAN_ACTIONS = frozenset({"retain", "delete_owned", "not_applicable"})
_COLLIDER_ACTIONS = frozenset({"retain", "delete_subtree", "not_applicable"})
_REPLACEMENT_ACTIONS = frozenset(
    {"retain_registered_source", "insert_runtime_asset", "none"}
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class NativeTaskEntityContractError(ValueError):
    """Fail-closed validation error with stable, sorted identifiers."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _valid_identifier(value: Any) -> bool:
    return bool(_IDENTIFIER.fullmatch(_string(value)))


def _valid_digest(value: Any) -> bool:
    return bool(_SHA256.fullmatch(_string(value)))


def _clone_mapping(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeTaskEntityContractError([error]) from exc
    if not isinstance(cloned, dict):
        raise NativeTaskEntityContractError([error])
    return cloned


def _mapping(
    value: Any, *, field: str, entity_label: str, errors: list[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"native_task_entity_{field}_invalid:{entity_label}")
        return {}
    return value


def _required_string(
    value: Mapping[str, Any],
    *,
    key: str,
    field: str,
    entity_label: str,
    errors: list[str],
) -> str:
    result = _string(value.get(key))
    if not result:
        errors.append(f"native_task_entity_{field}_invalid:{entity_label}")
    return result


def _required_bool(
    value: Mapping[str, Any],
    *,
    key: str,
    field: str,
    entity_label: str,
    errors: list[str],
) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        errors.append(f"native_task_entity_{field}_invalid:{entity_label}")
        return False
    return result


def _required_digest(
    value: Mapping[str, Any],
    *,
    key: str,
    field: str,
    entity_label: str,
    errors: list[str],
) -> str:
    result = _string(value.get(key))
    if not _valid_digest(result):
        errors.append(f"native_task_entity_{field}_invalid:{entity_label}")
    return result


def _pose(value: Any, *, entity_label: str, errors: list[str]) -> dict[str, Any]:
    error = f"native_task_entity_initial_state_pose_invalid:{entity_label}"
    if not isinstance(value, Mapping):
        errors.append(error)
        return {"position_world_m": [], "orientation_xyzw": []}

    vectors: dict[str, list[float]] = {}
    for key, length in (("position_world_m", 3), ("orientation_xyzw", 4)):
        source = value.get(key)
        if isinstance(source, (str, bytes)):
            errors.append(error)
            vectors[key] = []
            continue
        try:
            vector = [float(item) for item in source]
        except (TypeError, ValueError):
            errors.append(error)
            vector = []
        if len(vector) != length or not all(math.isfinite(item) for item in vector):
            errors.append(error)
            vector = []
        vectors[key] = vector

    orientation = vectors["orientation_xyzw"]
    if orientation:
        norm = math.sqrt(sum(component * component for component in orientation))
        if abs(norm - 1.0) > 1e-6:
            errors.append(error)
    return vectors


def project_legacy_task_object_entity(
    *, task_kind: str, task_object: Mapping[str, Any]
) -> dict[str, Any]:
    """Project the legacy ``task_object`` role without fabricating evidence.

    The legacy record must already carry every field required by the new
    contract.  This compatibility projection only maps its role and, when
    available, its old ``name`` into ``entity_id``.  Missing provenance,
    policies, or readback requirements therefore still fail closed.
    """

    if task_kind not in _LEGACY_TASK_OBJECT_ROLE_BY_TASK_KIND:
        raise NativeTaskEntityContractError(
            [f"native_task_entity_legacy_task_kind_invalid:{task_kind or 'missing'}"]
        )
    projected = _clone_mapping(
        task_object, error="native_task_entity_legacy_task_object_invalid"
    )
    if projected.get("semantic_role") != "task_object":
        raise NativeTaskEntityContractError(
            ["native_task_entity_legacy_semantic_role_invalid"]
        )
    if not _string(projected.get("entity_id")) and _string(projected.get("name")):
        projected["entity_id"] = _string(projected["name"])
    projected["semantic_role"] = _LEGACY_TASK_OBJECT_ROLE_BY_TASK_KIND[task_kind]
    return projected


def _normalize_entity(
    source: Mapping[str, Any], *, index: int, errors: list[str]
) -> dict[str, Any]:
    entity_id = _string(source.get("entity_id"))
    entity_label = entity_id or str(index)
    if not _valid_identifier(entity_id):
        errors.append(f"native_task_entity_id_invalid:{entity_label}")

    semantic_role = _string(source.get("semantic_role"))
    if semantic_role not in SEMANTIC_ROLES:
        errors.append(f"native_task_entity_semantic_role_invalid:{entity_label}")

    source_observation = _mapping(
        source.get("source_observation"),
        field="source_observation",
        entity_label=entity_label,
        errors=errors,
    )
    observation_id = _required_string(
        source_observation,
        key="observation_id",
        field="source_observation",
        entity_label=entity_label,
        errors=errors,
    )
    source_kind = _required_string(
        source_observation,
        key="source_kind",
        field="source_observation",
        entity_label=entity_label,
        errors=errors,
    )
    source_reference = _required_string(
        source_observation,
        key="source_reference",
        field="source_observation",
        entity_label=entity_label,
        errors=errors,
    )
    observation_digest = _required_digest(
        source_observation,
        key="source_sha256",
        field="source_observation",
        entity_label=entity_label,
        errors=errors,
    )
    observed = _required_bool(
        source_observation,
        key="observed",
        field="source_observation",
        entity_label=entity_label,
        errors=errors,
    )
    if source_kind not in _SOURCE_KINDS:
        errors.append(f"native_task_entity_source_observation_invalid:{entity_label}")

    physics_type = _string(source.get("physics_type"))
    if physics_type not in _PHYSICS_TYPES_BY_ROLE.get(semantic_role, frozenset()):
        errors.append(f"native_task_entity_physics_type_invalid:{entity_label}")

    runtime_asset = _mapping(
        source.get("runtime_asset"),
        field="runtime_asset",
        entity_label=entity_label,
        errors=errors,
    )
    asset_id = _required_string(
        runtime_asset,
        key="asset_id",
        field="runtime_asset",
        entity_label=entity_label,
        errors=errors,
    )
    binding_kind = _required_string(
        runtime_asset,
        key="binding_kind",
        field="runtime_asset",
        entity_label=entity_label,
        errors=errors,
    )
    asset_reference = _required_string(
        runtime_asset,
        key="source_reference",
        field="runtime_asset",
        entity_label=entity_label,
        errors=errors,
    )
    runtime_asset_digest = _required_digest(
        runtime_asset,
        key="sha256",
        field="runtime_asset",
        entity_label=entity_label,
        errors=errors,
    )
    if binding_kind not in _ASSET_BINDING_KINDS:
        errors.append(f"native_task_entity_runtime_asset_invalid:{entity_label}")

    initial_state = _mapping(
        source.get("initial_state"),
        field="initial_state",
        entity_label=entity_label,
        errors=errors,
    )
    pose_world = _pose(
        initial_state.get("pose_world"), entity_label=entity_label, errors=errors
    )
    state_digest = _required_digest(
        initial_state,
        key="state_sha256",
        field="initial_state",
        entity_label=entity_label,
        errors=errors,
    )
    settled_state_required = _required_bool(
        initial_state,
        key="settled_state_required",
        field="initial_state",
        entity_label=entity_label,
        errors=errors,
    )
    initial_penetration_allowed = _required_bool(
        initial_state,
        key="initial_penetration_allowed",
        field="initial_state",
        entity_label=entity_label,
        errors=errors,
    )
    if not settled_state_required or initial_penetration_allowed:
        errors.append(f"native_task_entity_initial_state_invalid:{entity_label}")

    reset_method = _mapping(
        source.get("reset_method"),
        field="reset_method",
        entity_label=entity_label,
        errors=errors,
    )
    reset_kind = _required_string(
        reset_method,
        key="kind",
        field="reset_method",
        entity_label=entity_label,
        errors=errors,
    )
    reset_state_id = _required_string(
        reset_method,
        key="state_id",
        field="reset_method",
        entity_label=entity_label,
        errors=errors,
    )
    reset_readback_required = _required_bool(
        reset_method,
        key="native_readback_required",
        field="reset_method",
        entity_label=entity_label,
        errors=errors,
    )
    post_start_write_allowed = _required_bool(
        reset_method,
        key="direct_state_write_after_episode_start_allowed",
        field="reset_method",
        entity_label=entity_label,
        errors=errors,
    )
    if (
        reset_kind not in _RESET_KINDS_BY_PHYSICS_TYPE.get(physics_type, frozenset())
        or not reset_readback_required
        or post_start_write_allowed
    ):
        errors.append(f"native_task_entity_reset_method_invalid:{entity_label}")

    contact_role = _mapping(
        source.get("contact_role"),
        field="contact_role",
        entity_label=entity_label,
        errors=errors,
    )
    contact_kind = _required_string(
        contact_role,
        key="kind",
        field="contact_role",
        entity_label=entity_label,
        errors=errors,
    )
    contact_readback_required = _required_bool(
        contact_role,
        key="native_contact_readback_required",
        field="contact_role",
        entity_label=entity_label,
        errors=errors,
    )
    if (
        contact_kind != _CONTACT_KIND_BY_ROLE.get(semantic_role)
        or not contact_readback_required
    ):
        errors.append(f"native_task_entity_contact_role_invalid:{entity_label}")

    scoring_role = _mapping(
        source.get("scoring_role"),
        field="scoring_role",
        entity_label=entity_label,
        errors=errors,
    )
    scoring_kind = _required_string(
        scoring_role,
        key="kind",
        field="scoring_role",
        entity_label=entity_label,
        errors=errors,
    )
    deterministic_state_required = _required_bool(
        scoring_role,
        key="deterministic_state_readback_required",
        field="scoring_role",
        entity_label=entity_label,
        errors=errors,
    )
    policy_self_grading_allowed = _required_bool(
        scoring_role,
        key="policy_self_grading_allowed",
        field="scoring_role",
        entity_label=entity_label,
        errors=errors,
    )
    if (
        scoring_kind != _SCORING_KIND_BY_ROLE.get(semantic_role)
        or not deterministic_state_required
        or policy_self_grading_allowed
    ):
        errors.append(f"native_task_entity_scoring_role_invalid:{entity_label}")

    removal_policy = _mapping(
        source.get("removal_policy"),
        field="removal_policy",
        entity_label=entity_label,
        errors=errors,
    )
    source_entity_action = _required_string(
        removal_policy,
        key="source_entity_action",
        field="removal_policy",
        entity_label=entity_label,
        errors=errors,
    )
    gaussian_action = _required_string(
        removal_policy,
        key="gaussian_action",
        field="removal_policy",
        entity_label=entity_label,
        errors=errors,
    )
    collider_action = _required_string(
        removal_policy,
        key="collider_action",
        field="removal_policy",
        entity_label=entity_label,
        errors=errors,
    )
    removal_receipt_digest = _required_digest(
        removal_policy,
        key="receipt_sha256",
        field="removal_policy",
        entity_label=entity_label,
        errors=errors,
    )
    removal_values_valid = (
        source_entity_action in _SOURCE_ENTITY_ACTIONS
        and gaussian_action in _GAUSSIAN_ACTIONS
        and collider_action in _COLLIDER_ACTIONS
    )
    removal_combination_valid = (
        source_entity_action == "retain"
        and gaussian_action == "retain"
        and collider_action == "retain"
    ) or (
        source_entity_action == "remove"
        and (gaussian_action == "delete_owned" or collider_action == "delete_subtree")
    ) or (
        source_entity_action == "not_present"
        and gaussian_action == "not_applicable"
        and collider_action == "not_applicable"
    )
    if not removal_values_valid or not removal_combination_valid:
        errors.append(f"native_task_entity_removal_policy_invalid:{entity_label}")

    replacement_policy = _mapping(
        source.get("replacement_policy"),
        field="replacement_policy",
        entity_label=entity_label,
        errors=errors,
    )
    replacement_action = _required_string(
        replacement_policy,
        key="action",
        field="replacement_policy",
        entity_label=entity_label,
        errors=errors,
    )
    replacement_required = _required_bool(
        replacement_policy,
        key="replacement_required",
        field="replacement_policy",
        entity_label=entity_label,
        errors=errors,
    )
    replacement_receipt_digest = _required_digest(
        replacement_policy,
        key="receipt_sha256",
        field="replacement_policy",
        entity_label=entity_label,
        errors=errors,
    )
    replacement_combination_valid = (
        replacement_action == "insert_runtime_asset" and replacement_required
    ) or (
        replacement_action in {"retain_registered_source", "none"}
        and not replacement_required
    )
    removal_replacement_join_valid = (
        source_entity_action == "retain"
        and replacement_action == "retain_registered_source"
    ) or (
        source_entity_action == "remove"
        and replacement_action == "insert_runtime_asset"
    ) or (
        source_entity_action == "not_present"
        and replacement_action in {"insert_runtime_asset", "none"}
    )
    if (
        replacement_action not in _REPLACEMENT_ACTIONS
        or not replacement_combination_valid
        or not removal_replacement_join_valid
    ):
        errors.append(f"native_task_entity_replacement_policy_invalid:{entity_label}")

    provenance = _mapping(
        source.get("provenance"),
        field="provenance",
        entity_label=entity_label,
        errors=errors,
    )
    normalized_provenance = {
        key: _required_string(
            provenance,
            key=key,
            field="provenance",
            entity_label=entity_label,
            errors=errors,
        )
        for key in (
            "source_id",
            "source_revision",
            "source_path",
            "license_id",
            "public_source_rights_id",
            "derived_processing_authority_id",
            "provider_terms_id",
            "output_rights_id",
            "attribution",
            "disclosure_class",
        )
    }
    source_size_bytes = provenance.get("source_size_bytes")
    if (
        isinstance(source_size_bytes, bool)
        or not isinstance(source_size_bytes, int)
        or source_size_bytes <= 0
    ):
        errors.append(f"native_task_entity_provenance_invalid:{entity_label}")
        source_size_bytes = 0
    upload_permitted = _required_bool(
        provenance,
        key="upload_permitted",
        field="provenance",
        entity_label=entity_label,
        errors=errors,
    )
    raw_redistribution_permitted = _required_bool(
        provenance,
        key="raw_redistribution_permitted",
        field="provenance",
        entity_label=entity_label,
        errors=errors,
    )
    provider_retention_permitted = _required_bool(
        provenance,
        key="provider_retention_permitted",
        field="provenance",
        entity_label=entity_label,
        errors=errors,
    )
    provider_training_permitted = _required_bool(
        provenance,
        key="provider_training_permitted",
        field="provenance",
        entity_label=entity_label,
        errors=errors,
    )
    if normalized_provenance["disclosure_class"] not in _DISCLOSURE_CLASSES:
        errors.append(f"native_task_entity_provenance_invalid:{entity_label}")

    digests = _mapping(
        source.get("digests"),
        field="digests",
        entity_label=entity_label,
        errors=errors,
    )
    normalized_digests = {
        key: _required_digest(
            digests,
            key=key,
            field="digests",
            entity_label=entity_label,
            errors=errors,
        )
        for key in (
            "source_sha256",
            "runtime_asset_sha256",
            "initial_state_sha256",
            "configuration_sha256",
        )
    }
    if (
        normalized_digests["source_sha256"] != observation_digest
        or normalized_digests["runtime_asset_sha256"] != runtime_asset_digest
        or normalized_digests["initial_state_sha256"] != state_digest
    ):
        errors.append(f"native_task_entity_digests_invalid:{entity_label}")

    return {
        "entity_id": entity_id,
        "semantic_role": semantic_role,
        "source_observation": {
            "observation_id": observation_id,
            "source_kind": source_kind,
            "source_reference": source_reference,
            "source_sha256": observation_digest,
            "observed": observed,
        },
        "physics_type": physics_type,
        "runtime_asset": {
            "asset_id": asset_id,
            "binding_kind": binding_kind,
            "source_reference": asset_reference,
            "sha256": runtime_asset_digest,
        },
        "initial_state": {
            "pose_world": pose_world,
            "state_sha256": state_digest,
            "settled_state_required": settled_state_required,
            "initial_penetration_allowed": initial_penetration_allowed,
        },
        "reset_method": {
            "kind": reset_kind,
            "state_id": reset_state_id,
            "native_readback_required": reset_readback_required,
            "direct_state_write_after_episode_start_allowed": post_start_write_allowed,
        },
        "contact_role": {
            "kind": contact_kind,
            "native_contact_readback_required": contact_readback_required,
        },
        "scoring_role": {
            "kind": scoring_kind,
            "deterministic_state_readback_required": deterministic_state_required,
            "policy_self_grading_allowed": policy_self_grading_allowed,
        },
        "removal_policy": {
            "source_entity_action": source_entity_action,
            "gaussian_action": gaussian_action,
            "collider_action": collider_action,
            "receipt_sha256": removal_receipt_digest,
        },
        "replacement_policy": {
            "action": replacement_action,
            "replacement_required": replacement_required,
            "receipt_sha256": replacement_receipt_digest,
        },
        "provenance": {
            **normalized_provenance,
            "source_size_bytes": source_size_bytes,
            "upload_permitted": upload_permitted,
            "raw_redistribution_permitted": raw_redistribution_permitted,
            "provider_retention_permitted": provider_retention_permitted,
            "provider_training_permitted": provider_training_permitted,
        },
        "digests": normalized_digests,
    }


def materialize_native_task_entity_contract(
    *, task_kind: str, task_entities: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return one canonical entity contract or raise stable typed blockers."""

    errors: list[str] = []
    if task_kind not in TASK_KINDS:
        errors.append(f"native_task_entity_task_kind_invalid:{task_kind or 'missing'}")

    if (
        isinstance(task_entities, (str, bytes, Mapping))
        or not isinstance(task_entities, Sequence)
        or not task_entities
    ):
        raise NativeTaskEntityContractError(
            ["native_task_entity_task_entities_invalid"] + errors
        )

    normalized: list[dict[str, Any]] = []
    legacy_projection_count = 0
    for index, raw_entity in enumerate(task_entities):
        if not isinstance(raw_entity, Mapping):
            errors.append(f"native_task_entity_invalid:{index}")
            continue
        entity = raw_entity
        if raw_entity.get("semantic_role") == "task_object":
            try:
                entity = project_legacy_task_object_entity(
                    task_kind=task_kind, task_object=raw_entity
                )
            except NativeTaskEntityContractError as exc:
                errors.extend(exc.errors)
                continue
            legacy_projection_count += 1
        normalized.append(_normalize_entity(entity, index=index, errors=errors))

    entity_ids: set[str] = set()
    semantic_role_index: dict[str, list[str]] = {}
    for entity in normalized:
        entity_id = entity["entity_id"]
        if entity_id in entity_ids:
            errors.append(f"native_task_entity_id_duplicate:{entity_id}")
        entity_ids.add(entity_id)
        semantic_role_index.setdefault(entity["semantic_role"], []).append(entity_id)

    for role in sorted(_REQUIRED_ROLES_BY_TASK_KIND.get(task_kind, frozenset())):
        if role not in semantic_role_index:
            errors.append(f"native_task_entity_role_missing:{role}")

    if errors:
        raise NativeTaskEntityContractError(errors)

    normalized.sort(key=lambda row: row["entity_id"])
    normalized_role_index = {
        role: sorted(entity_ids)
        for role, entity_ids in sorted(semantic_role_index.items())
    }
    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_kind": task_kind,
        "task_entities": normalized,
        "semantic_role_index": normalized_role_index,
        "legacy_task_object_projection_count": legacy_projection_count,
        "claim_boundary": (
            "structural_contract_only_native_application_contact_and_state_readback_"
            "remain_required"
        ),
    }
    contract["contract_digest"] = canonical_digest(contract)
    return contract


__all__ = [
    "NativeTaskEntityContractError",
    "SCHEMA_VERSION",
    "SEMANTIC_ROLES",
    "TASK_KINDS",
    "TASK_KIND_ARTICULATED_OPEN_CLOSE",
    "TASK_KIND_DEFORMABLE_TRANSFER",
    "TASK_KIND_RIGID_PICK_PLACE",
    "materialize_native_task_entity_contract",
    "project_legacy_task_object_entity",
]
