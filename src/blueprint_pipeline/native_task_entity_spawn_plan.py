"""Compile entity-keyed native spawn bindings without importing a simulator.

The Arena scene plan deliberately describes both legacy role-keyed assets and
new task entities.  This module is the pure-data boundary between that sealed
plan, any deformable/receptacle authoring input, and the native adapters.  It
fails before Isaac starts when an entity, candidate, authored operation, or
runtime asset identity cannot be joined exactly.

The result is still only a spawn plan.  It does not claim that USD schemas were
composed, PhysX cooked a deformable, reset was applied, or native readback
succeeded.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_scene_plan import SCHEMA_VERSION as ARENA_PLAN_SCHEMA
from .native_task_entity_asset_authoring_bundle import (
    DEFORMABLE_RUNTIME_CLASS,
    INPUT_SCHEMA_VERSION as AUTHORING_INPUT_SCHEMA,
    RIGID_RUNTIME_CLASS,
)
from .task_entity_asset_candidate import SCHEMA_VERSION as ASSET_CANDIDATE_SCHEMA


SCHEMA_VERSION = "native_task_entity_spawn_plan.v1"

ADAPTER_ARENA_OBJECT = "arena_object"
ADAPTER_ISAAC_DEFORMABLE_OBJECT = "isaac_deformable_object"

_OBJECT_TYPE_BY_PHYSICS_TYPE = {
    "rigid_body": "RIGID",
    "articulation": "ARTICULATION",
    "deformable_volume": "DEFORMABLE",
    "static_collider": "BASE",
}
_AUTHORING_ROLES = frozenset({"movable_deformable", "destination_receptacle"})
_LEGACY_TARGET_ROLE_BY_TASK_KIND = {
    "rigid_pick_place": "movable_rigid",
    "articulated_open_close": "articulated_fixture",
}


class NativeTaskEntitySpawnPlanError(ValueError):
    """Stable, sorted failures raised before native imports or GPU work."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _clone(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeTaskEntitySpawnPlanError([error]) from exc
    if not isinstance(result, dict):
        raise NativeTaskEntitySpawnPlanError([error])
    return result


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _basename(value: Any) -> str:
    path = PurePosixPath(str(value or ""))
    return path.name if path.name not in {"", ".", ".."} else ""


def _reset_recipe(
    *,
    entity: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    errors: list[str],
) -> dict[str, Any]:
    entity_id = str(entity["entity_id"])
    reset = entity.get("reset_method")
    if not isinstance(reset, Mapping):
        errors.append(f"native_task_entity_spawn_reset_invalid:{entity_id}")
        reset = {}
    reset_kind = str(reset.get("kind") or "")
    state_id = str(reset.get("state_id") or "")
    native_readback_required = reset.get("native_readback_required") is True
    post_start_allowed = reset.get(
        "direct_state_write_after_episode_start_allowed"
    )
    if not state_id or not native_readback_required or post_start_allowed is not False:
        errors.append(f"native_task_entity_spawn_reset_invalid:{entity_id}")

    if reset_kind == "native_deformable_state":
        configuration = (
            candidate.get("deformable_configuration")
            if isinstance(candidate, Mapping)
            else None
        )
        candidate_reset = (
            configuration.get("reset") if isinstance(configuration, Mapping) else None
        )
        if (
            not isinstance(candidate_reset, Mapping)
            or candidate_reset.get("reset_kind") != "native_default_nodal_state"
            or candidate_reset.get("write_default_nodal_state_before_episode")
            is not True
            or candidate_reset.get("zero_nodal_velocities") is not True
            or candidate_reset.get("free_kinematic_flag_value") != 1.0
            or candidate_reset.get("native_readback_required") is not True
            or candidate_reset.get(
                "direct_state_write_after_episode_start_allowed"
            )
            is not False
        ):
            errors.append(f"native_task_entity_spawn_reset_invalid:{entity_id}")
        steps = [
            {
                "order": 1,
                "operation": "load_default_nodal_state",
                "source": "deformable_object.data.default_nodal_state_w",
            },
            {
                "order": 2,
                "operation": "zero_nodal_velocities",
                "velocity_mps": [0.0, 0.0, 0.0],
            },
            {
                "order": 3,
                "operation": "write_nodal_state_to_sim_index",
            },
            {
                "order": 4,
                "operation": "write_nodal_kinematic_target_to_sim_index",
                "free_flag_value": 1.0,
            },
            {
                "order": 5,
                "operation": "readback_data_nodal_state_and_kinematic_target",
            },
        ]
    else:
        operations = {
            "native_rigid_state": (
                "write_default_root_state_to_sim",
                "readback_root_state",
            ),
            "native_articulation_state": (
                "write_default_root_and_joint_state_to_sim",
                "readback_root_and_joint_state",
            ),
            "immutable_scene_state": (
                "verify_immutable_scene_state",
                "readback_root_pose",
            ),
        }.get(reset_kind)
        if operations is None:
            errors.append(f"native_task_entity_spawn_reset_invalid:{entity_id}")
            operations = ()
        steps = [
            {"order": index, "operation": operation}
            for index, operation in enumerate(operations, start=1)
        ]

    return {
        "reset_kind": reset_kind,
        "state_id": state_id,
        "write_scope": "before_episode_start_only",
        "direct_state_write_after_episode_start_allowed": False,
        "native_readback_required": native_readback_required,
        "steps": steps,
    }


def _authoring_bindings(
    *,
    manifest_value: Mapping[str, Any] | None,
    task_kind: str,
    task_entity_contract_digest: str,
    required_entity_ids: set[str],
    entities: Mapping[str, Mapping[str, Any]],
    scene_objects: Mapping[str, Mapping[str, Any]],
    errors: list[str],
) -> tuple[dict[str, dict[str, Any]], str | None]:
    if manifest_value is None:
        for entity_id in sorted(required_entity_ids):
            errors.append(f"native_task_entity_spawn_authoring_plan_missing:{entity_id}")
        return {}, None
    manifest = _clone(
        manifest_value, error="native_task_entity_spawn_authoring_manifest_invalid"
    )
    if (
        manifest.get("schema_version") != AUTHORING_INPUT_SCHEMA
        or manifest.get("input_digest")
        != canonical_digest(manifest, digest_field="input_digest")
    ):
        errors.append("native_task_entity_spawn_authoring_manifest_invalid")
    if (
        manifest.get("task_kind") != task_kind
        or manifest.get("task_entity_contract_digest")
        != task_entity_contract_digest
    ):
        errors.append("native_task_entity_spawn_authoring_contract_mismatch")

    rows = manifest.get("entity_authoring_plans")
    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Sequence):
        errors.append("native_task_entity_spawn_authoring_plans_invalid")
        rows = []
    by_entity: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"native_task_entity_spawn_authoring_plan_invalid:{index}")
            continue
        entity_id = str(row.get("entity_id") or index)
        if entity_id in by_entity:
            errors.append(
                f"native_task_entity_spawn_authoring_plan_duplicate:{entity_id}"
            )
            continue
        by_entity[entity_id] = row
    for entity_id in sorted(required_entity_ids - set(by_entity)):
        errors.append(f"native_task_entity_spawn_authoring_plan_missing:{entity_id}")
    for entity_id in sorted(set(by_entity) - required_entity_ids):
        errors.append(f"native_task_entity_spawn_authoring_plan_unbound:{entity_id}")
    if manifest.get("asset_entity_ids") != sorted(required_entity_ids):
        errors.append("native_task_entity_spawn_authoring_entity_set_mismatch")

    bindings: dict[str, dict[str, Any]] = {}
    for entity_id in sorted(required_entity_ids & set(by_entity)):
        row = by_entity[entity_id]
        entity = entities[entity_id]
        runtime_asset = entity["runtime_asset"]
        initial_state = entity["initial_state"]
        scene_object = scene_objects.get(entity_id)
        if scene_object is None:
            errors.append(f"native_task_entity_spawn_scene_object_missing:{entity_id}")
            continue
        candidate = row.get("candidate_record")
        operation = row.get("operation")
        staged_files = row.get("staged_files")
        if not isinstance(candidate, Mapping) or not isinstance(operation, Mapping):
            errors.append(f"native_task_entity_spawn_authoring_plan_invalid:{entity_id}")
            continue
        if (
            candidate.get("schema_version") != ASSET_CANDIDATE_SCHEMA
            or candidate.get("candidate_digest")
            != canonical_digest(candidate, digest_field="candidate_digest")
            or row.get("candidate_digest") != candidate.get("candidate_digest")
        ):
            errors.append(f"native_task_entity_spawn_candidate_digest_invalid:{entity_id}")
        expected_asset_class = (
            "deformable_volume"
            if entity["semantic_role"] == "movable_deformable"
            else "rigid_receptacle"
        )
        if (
            row.get("semantic_role") != entity["semantic_role"]
            or row.get("physics_type") != entity["physics_type"]
            or row.get("asset_id") != runtime_asset.get("asset_id")
            or candidate.get("entity_id") != entity_id
            or candidate.get("asset_id") != runtime_asset.get("asset_id")
            or candidate.get("asset_class") != expected_asset_class
        ):
            errors.append(f"native_task_entity_spawn_candidate_join_invalid:{entity_id}")

        candidate_files = candidate.get("files")
        candidate_files = (
            candidate_files
            if isinstance(candidate_files, Sequence)
            and not isinstance(candidate_files, (str, bytes, Mapping))
            else []
        )
        staged_files = (
            staged_files
            if isinstance(staged_files, Sequence)
            and not isinstance(staged_files, (str, bytes, Mapping))
            else []
        )
        candidate_runtime = [
            item
            for item in candidate_files
            if isinstance(item, Mapping) and item.get("role") == "runtime_usd"
        ]
        staged_runtime = [
            item
            for item in staged_files
            if isinstance(item, Mapping) and item.get("role") == "runtime_usd"
        ]
        if len(candidate_runtime) != 1 or len(staged_runtime) != 1:
            errors.append(f"native_task_entity_spawn_runtime_usd_missing:{entity_id}")
            continue
        candidate_file = candidate_runtime[0]
        staged_file = staged_runtime[0]
        expected_digest = runtime_asset.get("sha256")
        if (
            candidate_file.get("sha256") != expected_digest
            or staged_file.get("sha256") != expected_digest
            or scene_object.get("sha256") != expected_digest
            or candidate_file.get("size_bytes") != staged_file.get("size_bytes")
            or _basename(candidate_file.get("path"))
            != _basename(staged_file.get("archive_relative_path"))
            or _basename(candidate_file.get("path"))
            != _basename(scene_object.get("usd_path"))
        ):
            errors.append(f"native_task_entity_spawn_asset_digest_mismatch:{entity_id}")

        configuration_key = (
            "deformable_configuration"
            if expected_asset_class == "deformable_volume"
            else "receptacle_configuration"
        )
        configuration = candidate.get(configuration_key)
        expected_runtime_class = (
            DEFORMABLE_RUNTIME_CLASS
            if expected_asset_class == "deformable_volume"
            else RIGID_RUNTIME_CLASS
        )
        if (
            not isinstance(configuration, Mapping)
            or canonical_digest(configuration)
            != entity.get("digests", {}).get("configuration_sha256")
            or operation.get("configuration") != configuration
            or operation.get("candidate_authored_transform")
            != candidate.get("transform")
            or operation.get("initial_pose_world")
            != initial_state.get("pose_world")
            or operation.get("runtime_class") != expected_runtime_class
        ):
            errors.append(f"native_task_entity_spawn_authoring_join_invalid:{entity_id}")

        bindings[entity_id] = {
            "candidate_digest": candidate.get("candidate_digest"),
            "runtime_usd_sha256": candidate_file.get("sha256"),
            "runtime_usd_size_bytes": candidate_file.get("size_bytes"),
            "staged_runtime_usd": dict(staged_file),
            "operation": dict(operation),
            "_candidate_record": dict(candidate),
        }
    return bindings, str(manifest.get("input_digest") or "")


def materialize_native_task_entity_spawn_plan(
    *,
    scene_plan: Mapping[str, Any],
    authoring_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Join a sealed scene plan to stable entity-keyed native adapters."""

    plan = _clone(scene_plan, error="native_task_entity_spawn_scene_plan_invalid")
    if (
        plan.get("schema_version") != ARENA_PLAN_SCHEMA
        or plan.get("plan_digest")
        != canonical_digest(plan, digest_field="plan_digest")
    ):
        raise NativeTaskEntitySpawnPlanError(
            ["native_task_entity_spawn_scene_plan_invalid"]
        )
    errors: list[str] = []
    entity_rows = plan.get("task_entities")
    if entity_rows is None:
        entity_rows = []
    if isinstance(entity_rows, (str, bytes, Mapping)) or not isinstance(
        entity_rows, Sequence
    ):
        errors.append("native_task_entity_spawn_entities_invalid")
        entity_rows = []
    entities: dict[str, Mapping[str, Any]] = {}
    role_index: dict[str, list[str]] = {}
    for index, entity in enumerate(entity_rows):
        if not isinstance(entity, Mapping):
            errors.append(f"native_task_entity_spawn_entity_invalid:{index}")
            continue
        entity_id = str(entity.get("entity_id") or index)
        role = str(entity.get("semantic_role") or "")
        if entity_id in entities:
            errors.append(f"native_task_entity_spawn_entity_duplicate:{entity_id}")
            continue
        normalized_entity = dict(entity)
        for field in (
            "runtime_asset",
            "initial_state",
            "reset_method",
            "replacement_policy",
            "digests",
        ):
            value = entity.get(field)
            if not isinstance(value, Mapping):
                errors.append(
                    f"native_task_entity_spawn_entity_field_invalid:{entity_id}:{field}"
                )
                value = {}
            normalized_entity[field] = dict(value)
        entities[entity_id] = normalized_entity
        role_index.setdefault(role, []).append(entity_id)
    normalized_role_index = {
        role: sorted(entity_ids) for role, entity_ids in sorted(role_index.items())
    }
    if entities and plan.get("task_entity_role_index") != normalized_role_index:
        errors.append("native_task_entity_spawn_role_index_mismatch")
    contract_digest = str(plan.get("task_entity_contract_digest") or "")
    if entities and not _digest(contract_digest):
        errors.append("native_task_entity_spawn_contract_digest_invalid")
    robot_entity_ids = normalized_role_index.get("robot", [])
    if entities:
        if len(robot_entity_ids) != 1:
            errors.append("native_task_entity_spawn_robot_cardinality_invalid")
        else:
            robot_entity_id = robot_entity_ids[0]
            robot_entity = entities[robot_entity_id]
            robot_plan = plan.get("robot")
            if (
                not isinstance(robot_plan, Mapping)
                or robot_entity.get("physics_type") != "robot_articulation"
                or robot_entity["runtime_asset"].get("binding_kind")
                != "runtime_embodiment"
                or robot_entity["initial_state"].get("pose_world")
                != robot_plan.get("base_pose_world")
            ):
                errors.append(
                    f"native_task_entity_spawn_robot_join_invalid:{robot_entity_id}"
                )

    object_rows = plan.get("objects")
    if isinstance(object_rows, (str, bytes, Mapping)) or not isinstance(
        object_rows, Sequence
    ):
        raise NativeTaskEntitySpawnPlanError(
            ["native_task_entity_spawn_scene_objects_invalid"]
        )
    scene_objects: dict[str, Mapping[str, Any]] = {}
    runtime_names: set[str] = set()
    prim_paths: set[str] = set()
    asset_rows: list[dict[str, Any]] = []
    role_names: dict[str, list[str]] = {}
    for index, row in enumerate(object_rows):
        if not isinstance(row, Mapping):
            errors.append(f"native_task_entity_spawn_scene_object_invalid:{index}")
            continue
        entity_id = str(row.get("entity_id") or "")
        role = str(row.get("semantic_role") or "")
        identity = entity_id or role or str(index)
        runtime_name = str(row.get("name") or role)
        prim_path = str(row.get("prim_path") or "")
        if not runtime_name or runtime_name in runtime_names:
            errors.append(f"native_task_entity_spawn_runtime_name_duplicate:{identity}")
        runtime_names.add(runtime_name)
        if not prim_path or prim_path in prim_paths:
            errors.append(f"native_task_entity_spawn_prim_path_duplicate:{identity}")
        prim_paths.add(prim_path)

        entity: Mapping[str, Any] | None = None
        adapter_kind = ADAPTER_ARENA_OBJECT
        semantic_tags = [["class", role]]
        if entity_id:
            entity = entities.get(entity_id)
            if entity is None:
                errors.append(f"native_task_entity_spawn_entity_unbound:{entity_id}")
                continue
            if entity_id in scene_objects:
                errors.append(f"native_task_entity_spawn_scene_object_duplicate:{entity_id}")
                continue
            scene_objects[entity_id] = row
            expected_type = _OBJECT_TYPE_BY_PHYSICS_TYPE.get(
                str(entity.get("physics_type") or "")
            )
            if (
                role != entity.get("semantic_role")
                or row.get("object_type") != expected_type
                or row.get("sha256") != entity.get("runtime_asset", {}).get("sha256")
                or row.get("pose_world") != entity.get("initial_state", {}).get("pose_world")
                or _basename(row.get("usd_path"))
                != _basename(entity.get("runtime_asset", {}).get("source_reference"))
            ):
                errors.append(f"native_task_entity_spawn_scene_join_invalid:{entity_id}")
            if not prim_path.startswith("{ENV_REGEX_NS}/task_entities/"):
                errors.append(f"native_task_entity_spawn_prim_path_invalid:{entity_id}")
            adapter_kind = (
                ADAPTER_ISAAC_DEFORMABLE_OBJECT
                if entity.get("physics_type") == "deformable_volume"
                else ADAPTER_ARENA_OBJECT
            )
            semantic_tags = [["class", role], ["entity_id", entity_id]]

        role_names.setdefault(role, []).append(runtime_name)
        asset_rows.append(
            {
                "source_object_index": index,
                "runtime_name": runtime_name,
                "prim_path": prim_path,
                "semantic_role": role,
                **({"entity_id": entity_id} if entity_id else {}),
                "adapter_kind": adapter_kind,
                "object_type": row.get("object_type"),
                "usd_path": row.get("usd_path"),
                "sha256": row.get("sha256"),
                "size_bytes": row.get("size_bytes"),
                "visible": row.get("visible") is True,
                "pose_world": row.get("pose_world"),
                "semantic_tags": semantic_tags,
                "activate_contact_sensors": row.get("activate_contact_sensors")
                is True,
            }
        )

    expected_scene_entity_ids = {
        entity_id
        for entity_id, entity in entities.items()
        if entity.get("semantic_role") != "robot"
        and entity.get("runtime_asset", {}).get("binding_kind") == "usd_asset"
    }
    for entity_id in sorted(expected_scene_entity_ids - set(scene_objects)):
        errors.append(f"native_task_entity_spawn_scene_object_missing:{entity_id}")
    for entity_id in sorted(set(scene_objects) - expected_scene_entity_ids):
        errors.append(f"native_task_entity_spawn_scene_object_unexpected:{entity_id}")

    required_authoring_ids = {
        entity_id
        for entity_id, entity in entities.items()
        if entity.get("semantic_role") in _AUTHORING_ROLES
        and entity.get("replacement_policy", {}).get("action")
        == "insert_runtime_asset"
    }
    if required_authoring_ids or authoring_manifest is not None:
        authoring_bindings, authoring_input_digest = _authoring_bindings(
            manifest_value=authoring_manifest,
            task_kind=str(plan.get("task_kind") or ""),
            task_entity_contract_digest=contract_digest,
            required_entity_ids=required_authoring_ids,
            entities=entities,
            scene_objects=scene_objects,
            errors=errors,
        )
    else:
        authoring_bindings, authoring_input_digest = {}, None

    for row in asset_rows:
        entity_id = row.get("entity_id")
        entity = entities.get(str(entity_id)) if entity_id else None
        if entity is not None:
            binding = authoring_bindings.get(str(entity_id))
            candidate_record = (binding or {}).get("_candidate_record")
            row["authoring_binding"] = (
                {
                    key: value
                    for key, value in binding.items()
                    if not key.startswith("_")
                }
                if binding is not None
                else None
            )
            row["reset_recipe"] = _reset_recipe(
                entity=entity,
                candidate=candidate_record,
                errors=errors,
            )

    if errors:
        raise NativeTaskEntitySpawnPlanError(errors)

    role_aliases = {
        role: names[0]
        for role, names in sorted(role_names.items())
        if len(names) == 1
    }
    legacy_target_role = _LEGACY_TARGET_ROLE_BY_TASK_KIND.get(plan.get("task_kind"))
    if legacy_target_role in role_aliases:
        role_aliases["task_object"] = role_aliases[legacy_target_role]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scene_plan_digest": plan["plan_digest"],
        "task_entity_contract_digest": contract_digest or None,
        "authoring_input_digest": authoring_input_digest,
        "assets": asset_rows,
        "entity_asset_names": dict(
            sorted(
                (str(row["entity_id"]), str(row["runtime_name"]))
                for row in asset_rows
                if row.get("entity_id")
            )
        ),
        "entity_prim_paths": dict(
            sorted(
                (str(row["entity_id"]), str(row["prim_path"]))
                for row in asset_rows
                if row.get("entity_id")
            )
        ),
        "role_aliases": dict(sorted(role_aliases.items())),
        "claim_boundary": {
            "spawn_plan_is_not_native_import_proof": True,
            "reset_recipe_is_not_native_reset_readback": True,
            "simulator_execution_is_not_physical_truth": True,
        },
        "spawn_plan_digest": "",
    }
    result["spawn_plan_digest"] = canonical_digest(
        result, digest_field="spawn_plan_digest"
    )
    return json.loads(json.dumps(result))


__all__ = [
    "ADAPTER_ARENA_OBJECT",
    "ADAPTER_ISAAC_DEFORMABLE_OBJECT",
    "NativeTaskEntitySpawnPlanError",
    "SCHEMA_VERSION",
    "materialize_native_task_entity_spawn_plan",
]
