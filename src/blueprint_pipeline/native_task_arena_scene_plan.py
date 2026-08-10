"""Compile a native Arena scene from the task-neutral runtime contract.

This module deliberately imports no Isaac or Arena package.  It settles the
expensive-to-discover facts before a GPU launch: exact staged bytes, spawn
types and prim paths, articulation reset names, contact filters, robot reset
state, camera roles, and the physics/control cadence.  The GPU worker consumes
the resulting plan and must still retain native application/readback evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_articulated_motion_geometry import (
    NativeArticulatedMotionGeometryError,
    derive_native_articulated_motion_geometry,
)
from .native_task_runtime_contract import SCHEMA_VERSION as RUNTIME_CONTRACT_SCHEMA


SCHEMA_VERSION = "native_task_arena_scene_plan.v1"
ENV_ROOT = "{ENV_REGEX_NS}"


class NativeTaskArenaScenePlanError(ValueError):
    """Stable, sorted failures raised before the native runtime starts."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _source_to_spawned_prim(
    source_path: str, *, spawned_root_prim_path: str, source_root_prim_path: str
) -> str:
    if source_path == source_root_prim_path:
        suffix = ""
    elif source_path.startswith(source_root_prim_path + "/"):
        suffix = source_path[len(source_root_prim_path) :]
    else:
        raise NativeTaskArenaScenePlanError(
            [f"native_task_arena_source_prim_invalid:{source_path}"]
        )
    return f"{spawned_root_prim_path}{suffix}"


def _entity_runtime_name(entity_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", entity_id)
    if not normalized or normalized[0].isdigit():
        normalized = f"entity_{normalized}"
    suffix = hashlib.sha256(entity_id.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}_{suffix}"


def _entity_prim_path(entity_id: str) -> str:
    return f"{ENV_ROOT}/task_entities/{_entity_runtime_name(entity_id)}"


def _validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        contract = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_runtime_contract_invalid"]
        ) from exc
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != RUNTIME_CONTRACT_SCHEMA
    ):
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_runtime_contract_invalid"]
        )
    expected = canonical_digest(contract, digest_field="contract_digest")
    if contract.get("contract_digest") != expected:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_runtime_contract_digest_invalid"]
        )
    return contract


def _stage_assets(
    objects: list[dict[str, Any]],
    *,
    provider_asset_directory: Path,
    published_asset_directory: str | None,
) -> list[dict[str, Any]]:
    errors: list[str] = []
    if (
        not provider_asset_directory.is_dir()
        or provider_asset_directory.is_symlink()
    ):
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_asset_directory_invalid"]
        )
    rows: list[dict[str, Any]] = []
    for row in objects:
        role = str(row["semantic_role"])
        entity_id = str(row.get("entity_id") or "")
        identity = entity_id or role
        candidate = provider_asset_directory / str(row["filename"])
        path = candidate.resolve()
        if provider_asset_directory != path.parent:
            errors.append(f"native_task_arena_asset_outside_directory:{identity}")
            continue
        if candidate.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"native_task_arena_asset_missing:{identity}")
            continue
        observed = _sha256(path)
        if observed != row["sha256"]:
            errors.append(f"native_task_arena_asset_digest_mismatch:{identity}")
            continue
        staged = {
            "name": _entity_runtime_name(entity_id) if entity_id else identity,
            "semantic_role": role,
            "prim_path": (
                _entity_prim_path(entity_id)
                if entity_id
                else f"{ENV_ROOT}/{role}"
            ),
            "object_type": row["object_type"],
            "usd_path": (
                f"{published_asset_directory}/{row['filename']}"
                if published_asset_directory is not None
                else str(path)
            ),
            "sha256": observed,
            "size_bytes": path.stat().st_size,
            "visible": bool(row["visible"]),
            "pose_world": row["pose_world"],
            # Arena's existing ContactSensor seam is rigid-body based.  A
            # deformable entity must use a separately qualified native
            # soft-body contact readback instead of being silently routed
            # through that sensor.
            "activate_contact_sensors": row["object_type"]
            in {"RIGID", "ARTICULATION"},
        }
        if entity_id:
            staged["entity_id"] = entity_id
        if row["object_type"] == "DEFORMABLE":
            staged["requires_native_deformable_contact_readback"] = True
        rows.append(staged)
    if errors:
        raise NativeTaskArenaScenePlanError(errors)
    return rows


def _cadence(contract: Mapping[str, Any], *, physics_frequency_hz: float) -> dict[str, Any]:
    try:
        control_frequency = float(contract["task_spec"]["control_frequency_hz"])
        physics_frequency = float(physics_frequency_hz)
        ratio = physics_frequency / control_frequency
        maximum_action_steps = int(contract["task_spec"]["maximum_action_steps"])
        settle_window_samples = int(
            contract["task_spec"].get("settle_window_samples", 0)
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_control_cadence_invalid"]
        ) from exc
    rounded = round(ratio)
    if (
        not math.isfinite(control_frequency)
        or not math.isfinite(physics_frequency)
        or control_frequency <= 0.0
        or physics_frequency <= 0.0
        or rounded < 1
        or maximum_action_steps <= 0
        or settle_window_samples < 0
        or not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_control_cadence_invalid"]
        )
    return {
        "control_frequency_hz": control_frequency,
        "physics_frequency_hz": physics_frequency,
        "physics_dt_seconds": 1.0 / physics_frequency,
        "control_decimation": int(rounded),
        "maximum_action_steps": maximum_action_steps,
        "settle_window_samples": settle_window_samples,
        "episode_length_seconds": (
            maximum_action_steps + settle_window_samples + 1
        )
        / control_frequency,
    }


def _articulation_plan(
    contract: Mapping[str, Any],
    *,
    staged_objects: list[dict[str, Any]],
    asset_directory: Path,
) -> dict[str, Any]:
    task_kind = contract["task_kind"]
    if task_kind != "articulated_open_close":
        return {
            "task_joint_reset_positions_rad": {},
            "task_joint_prim_paths": {},
            "task_joint_roles": {},
            "contact_sensors": [],
        }
    sample_binding = contract["task_sample_binding"]
    state_binding = contract["task_state_binding"]
    native_names = dict(sample_binding["native_joint_names"])
    scorer_reset = dict(contract["task_spec"]["joint_reset_positions_rad"])
    try:
        native_reset = {
            native_names[joint_id]: float(value)
            for joint_id, value in sorted(scorer_reset.items())
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_task_joint_reset_invalid"]
        ) from exc
    articulated_entity_ids = contract.get("task_entity_role_index", {}).get(
        "articulated_fixture", []
    )
    if articulated_entity_ids:
        if len(articulated_entity_ids) != 1:
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_articulated_entity_cardinality_invalid"]
            )
        articulated_entity_id = str(articulated_entity_ids[0])
        contract_object = next(
            (
                row
                for row in contract["objects"]
                if row.get("entity_id") == articulated_entity_id
            ),
            None,
        )
        staged_object = next(
            (
                row
                for row in staged_objects
                if row.get("entity_id") == articulated_entity_id
            ),
            None,
        )
    else:
        contract_object = next(
            (
                row
                for row in contract["objects"]
                if row["semantic_role"] == "task_object"
            ),
            None,
        )
        staged_object = next(
            (
                row
                for row in staged_objects
                if row["semantic_role"] == "task_object"
            ),
            None,
        )
    if contract_object is None or staged_object is None:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_task_object_asset_missing"]
        )
    task_object_asset_path = asset_directory / str(contract_object["filename"])
    spawned_root_prim_path = str(staged_object["prim_path"])
    target_joint_id = str(contract["task_spec"]["target_joint_id"])
    try:
        motion_geometry = derive_native_articulated_motion_geometry(
            task_object_usd_path=task_object_asset_path,
            task_object_sha256=contract_object["sha256"],
            target_joint_id=target_joint_id,
            target_joint_prim_path=sample_binding["joint_prim_paths"][
                target_joint_id
            ],
            moving_link_prim_path=state_binding["moving_link_prim_path"],
            handle_grasp_point_moving_link_m=state_binding[
                "handle_grasp_point_link_m"
            ],
            task_object_pose_world=contract_object["pose_world"],
            reset_angle_rad=contract["task_spec"]["joint_reset_positions_rad"][
                target_joint_id
            ],
            scripted_target_angle_rad=contract["task_spec"][
                "scripted_positive_target_rad"
            ],
        )
    except (KeyError, StopIteration, NativeArticulatedMotionGeometryError) as exc:
        errors = (
            list(exc.errors)
            if isinstance(exc, NativeArticulatedMotionGeometryError)
            else ["native_task_arena_motion_geometry_binding_invalid"]
        )
        raise NativeTaskArenaScenePlanError(errors) from exc
    source_root = motion_geometry["source_asset_root_prim_path"]
    moving_link = _source_to_spawned_prim(
        str(state_binding["moving_link_prim_path"]),
        spawned_root_prim_path=spawned_root_prim_path,
        source_root_prim_path=source_root,
    )
    scene_filter = f"{ENV_ROOT}/scene_collision/.*"
    return {
        "task_joint_reset_positions_rad": native_reset,
        "task_joint_prim_paths": {
            joint_id: _source_to_spawned_prim(
                path,
                spawned_root_prim_path=spawned_root_prim_path,
                source_root_prim_path=source_root,
            )
            for joint_id, path in sorted(
                sample_binding["joint_prim_paths"].items()
            )
        },
        "task_joint_roles": dict(sample_binding["joint_roles"]),
        "moving_link_native_body_name": state_binding[
            "moving_link_native_body_name"
        ],
        "handle_prim_paths": [
            _source_to_spawned_prim(
                path,
                spawned_root_prim_path=spawned_root_prim_path,
                source_root_prim_path=source_root,
            )
            for path in state_binding["handle_prim_paths"]
        ],
        "handle_grasp_point_link_m": state_binding["handle_grasp_point_link_m"],
        "motion_geometry": motion_geometry,
        "contact_sensors": [
            {
                "sensor_id": "task_robot_contact",
                "prim_path": moving_link,
                "filter_prim_paths_expr": [
                    state_binding["robot_gripper_contact_prim_pattern"]
                ],
            },
            {
                "sensor_id": "task_scene_contact",
                "prim_path": moving_link,
                "filter_prim_paths_expr": [scene_filter],
            },
            {
                "sensor_id": "robot_scene_contact",
                "prim_path": state_binding["robot_collision_prim_pattern"],
                "filter_prim_paths_expr": [scene_filter],
            },
        ],
        "state_thresholds": {
            key: state_binding[key]
            for key in (
                "task_contact_minimum_force_n",
                "collision_failure_minimum_force_n",
                "retreat_minimum_separation_m",
                "root_translation_tolerance_m",
                "root_orientation_tolerance_rad",
            )
        },
    }


def materialize_native_task_arena_scene_plan(
    *,
    runtime_contract: Mapping[str, Any],
    provider_asset_directory: str | Path,
    physics_frequency_hz: float,
    published_asset_directory: str | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Verify staged bytes and freeze one deterministic native scene plan."""

    contract = _validate_contract(runtime_contract)
    raw_asset_directory = Path(provider_asset_directory).expanduser()
    if raw_asset_directory.is_symlink():
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_asset_directory_invalid"]
        )
    asset_directory = raw_asset_directory.resolve()
    if published_asset_directory is not None:
        pure = PurePosixPath(published_asset_directory)
        if (
            not published_asset_directory
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.name in {"", ".", ".."}
        ):
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_published_asset_directory_invalid"]
            )
        published_asset_directory = pure.as_posix().rstrip("/")
    objects = _stage_assets(
        list(contract["objects"]),
        provider_asset_directory=asset_directory,
        published_asset_directory=published_asset_directory,
    )
    articulation = _articulation_plan(
        contract,
        staged_objects=objects,
        asset_directory=asset_directory,
    )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runtime_contract_digest": contract["contract_digest"],
        "scene_id": contract["scene_id"],
        "task_id": contract["task_id"],
        "task_kind": contract["task_kind"],
        "task_spec": contract["task_spec"],
        "task_sample_binding": contract["task_sample_binding"],
        "task_state_binding": contract["task_state_binding"],
        "scenario": contract["scenario"],
        "asset_directory": published_asset_directory or str(asset_directory),
        "objects": objects,
        "robot": contract["robot"],
        "cameras": contract["cameras"],
        "cadence": _cadence(contract, physics_frequency_hz=physics_frequency_hz),
        "articulation": articulation,
        "reset": {
            "robot_joint_positions_rad": contract["robot"][
                "joint_reset_positions_rad"
            ],
            "task_joint_positions_rad": articulation[
                "task_joint_reset_positions_rad"
            ],
            "root_pose_reset_required": True,
            "native_readback_required_after_reset": True,
        },
        "application_readback_required": contract["runtime_readback_required"],
        "claim_boundary": {
            "plan_is_not_native_application_proof": True,
            "plan_is_not_policy_episode_evidence": True,
            "simulator_execution_is_not_physical_truth": True,
        },
        "plan_digest": "",
    }
    if "task_entities" in contract:
        plan.update(
            {
                "task_entities": contract["task_entities"],
                "task_entity_role_index": contract["task_entity_role_index"],
                "task_entity_contract_digest": contract[
                    "task_entity_contract_digest"
                ],
            }
        )
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    if destination is not None:
        write_json(Path(destination), plan)
    return json.loads(json.dumps(plan))


__all__ = [
    "NativeTaskArenaScenePlanError",
    "SCHEMA_VERSION",
    "materialize_native_task_arena_scene_plan",
]
