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
from pathlib import Path
from typing import Any, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest
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


def _source_to_spawned_prim(source_path: str, *, role: str) -> str:
    if source_path == "/Asset":
        suffix = ""
    elif source_path.startswith("/Asset/"):
        suffix = source_path[len("/Asset") :]
    else:
        raise NativeTaskArenaScenePlanError(
            [f"native_task_arena_source_prim_invalid:{source_path}"]
        )
    return f"{ENV_ROOT}/{role}{suffix}"


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
    objects: list[dict[str, Any]], *, provider_asset_directory: Path
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
        candidate = provider_asset_directory / str(row["filename"])
        path = candidate.resolve()
        if provider_asset_directory != path.parent:
            errors.append(f"native_task_arena_asset_outside_directory:{role}")
            continue
        if candidate.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"native_task_arena_asset_missing:{role}")
            continue
        observed = _sha256(path)
        if observed != row["sha256"]:
            errors.append(f"native_task_arena_asset_digest_mismatch:{role}")
            continue
        rows.append(
            {
                "name": role,
                "semantic_role": role,
                "prim_path": f"{ENV_ROOT}/{role}",
                "object_type": row["object_type"],
                "usd_path": str(path),
                "sha256": observed,
                "size_bytes": path.stat().st_size,
                "visible": bool(row["visible"]),
                "pose_world": row["pose_world"],
                "activate_contact_sensors": row["object_type"]
                in {"RIGID", "ARTICULATION"},
            }
        )
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


def _articulation_plan(contract: Mapping[str, Any]) -> dict[str, Any]:
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
    moving_link = _source_to_spawned_prim(
        str(state_binding["moving_link_prim_path"]), role="task_object"
    )
    scene_filter = f"{ENV_ROOT}/scene_collision/.*"
    return {
        "task_joint_reset_positions_rad": native_reset,
        "task_joint_prim_paths": {
            joint_id: _source_to_spawned_prim(path, role="task_object")
            for joint_id, path in sorted(
                sample_binding["joint_prim_paths"].items()
            )
        },
        "task_joint_roles": dict(sample_binding["joint_roles"]),
        "moving_link_native_body_name": state_binding[
            "moving_link_native_body_name"
        ],
        "handle_prim_paths": [
            _source_to_spawned_prim(path, role="task_object")
            for path in state_binding["handle_prim_paths"]
        ],
        "handle_grasp_point_link_m": state_binding["handle_grasp_point_link_m"],
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
    objects = _stage_assets(
        list(contract["objects"]), provider_asset_directory=asset_directory
    )
    articulation = _articulation_plan(contract)
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
        "asset_directory": str(asset_directory),
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
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    if destination is not None:
        write_json(Path(destination), plan)
    return json.loads(json.dumps(plan))


__all__ = [
    "NativeTaskArenaScenePlanError",
    "SCHEMA_VERSION",
    "materialize_native_task_arena_scene_plan",
]
