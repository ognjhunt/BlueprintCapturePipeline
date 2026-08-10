"""Seal the task-neutral inputs a native arm evaluation runtime must consume.

The original ADP-009D runtime learned these facts from Python constants: which
USD was the task object, whether it was rigid, where the Franka stood, and what
the cameras observed.  That works for one canned beverage and silently becomes
wrong for the next scene.  This contract makes those choices data, validates
their cross-links before a GPU launch, and gives the runtime one digest-bound
document to read.

This module deliberately does not claim the values were applied by Isaac.
Application and readback remain runtime gates; a valid document is only the
immutable request for those gates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .articulated_runtime_composition import plan_articulated_runtime_composition
from .common import write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "native_task_runtime_contract.v1"
PROGRAM_ID = "arm-decision-proof-v1"
FROZEN_CANDIDATES = ("pi05_droid", "groot_n17_droid")
ASSET_ROLES = ("scene_collision", "scene_appearance", "task_object")
CAMERA_ROLES = ("external", "wrist", "overview")
TASK_KINDS = ("rigid_pick_place", "articulated_open_close")


class NativeTaskRuntimeContractError(ValueError):
    """Stable, sorted contract failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _finite_vector(
    value: Any, *, length: int, error: str, errors: list[str]
) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        errors.append(error)
        return []
    if len(result) != length or not all(math.isfinite(item) for item in result):
        errors.append(error)
        return []
    return result


def _pose(value: Any, *, error: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(error)
        return {}
    position = _finite_vector(
        value.get("position_world_m"), length=3, error=error, errors=errors
    )
    orientation = _finite_vector(
        value.get("orientation_xyzw"), length=4, error=error, errors=errors
    )
    if orientation:
        norm = math.sqrt(sum(item * item for item in orientation))
        if abs(norm - 1.0) > 1e-6:
            errors.append(error)
    return {
        "position_world_m": position,
        "orientation_xyzw": orientation,
    }


def _asset_rows(
    assets: Sequence[Mapping[str, Any]], *, errors: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    by_role: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(assets):
        if not isinstance(row, Mapping):
            errors.append(f"native_task_runtime_asset_invalid:{index}")
            continue
        role = str(row.get("semantic_role") or "")
        if role not in ASSET_ROLES or role in by_role:
            errors.append(f"native_task_runtime_asset_role_invalid:{role or index}")
            continue
        by_role[role] = row
    required = {"scene_collision", "task_object"}
    for role in sorted(required - set(by_role)):
        errors.append(f"native_task_runtime_asset_missing:{role}")

    rows: list[dict[str, Any]] = []
    for role in ASSET_ROLES:
        if role not in by_role:
            continue
        source = by_role[role]
        filename = str(source.get("filename") or "")
        if (
            not filename
            or PurePosixPath(filename).name != filename
            or filename in {".", ".."}
        ):
            errors.append(f"native_task_runtime_asset_filename_invalid:{role}")
        digest = str(source.get("sha256") or "")
        if not _digest(digest):
            errors.append(f"native_task_runtime_asset_digest_invalid:{role}")
        pose = _pose(
            source.get("pose_world"),
            error=f"native_task_runtime_asset_pose_invalid:{role}",
            errors=errors,
        )
        rows.append(
            {
                "name": str(source.get("name") or role),
                "semantic_role": role,
                "filename": filename,
                "sha256": digest,
                "pose_world": pose,
            }
        )
    return rows, by_role


def _camera_rows(
    cameras: Sequence[Mapping[str, Any]], *, errors: list[str]
) -> list[dict[str, Any]]:
    by_role: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(cameras):
        if not isinstance(row, Mapping):
            errors.append(f"native_task_runtime_camera_invalid:{index}")
            continue
        role = str(row.get("role") or "")
        if role not in CAMERA_ROLES or role in by_role:
            errors.append(f"native_task_runtime_camera_role_invalid:{role or index}")
            continue
        by_role[role] = row
    for role in sorted(set(CAMERA_ROLES) - set(by_role)):
        errors.append(f"native_task_runtime_camera_missing:{role}")

    rows: list[dict[str, Any]] = []
    for role in CAMERA_ROLES:
        if role not in by_role:
            continue
        source = by_role[role]
        expected_policy = role in {"external", "wrist"}
        if bool(source.get("policy_input")) is not expected_policy:
            errors.append(f"native_task_runtime_camera_policy_role_invalid:{role}")
        if bool(source.get("scoring_input")):
            errors.append(f"native_task_runtime_camera_scoring_forbidden:{role}")
        frame = str(source.get("pose_frame") or "")
        expected_frame = "panda_hand" if role == "wrist" else "world"
        if frame != expected_frame:
            errors.append(f"native_task_runtime_camera_pose_frame_invalid:{role}")
        matrix = _finite_vector(
            source.get("frame_from_camera_matrix"),
            length=16,
            error=f"native_task_runtime_camera_pose_invalid:{role}",
            errors=errors,
        )
        intrinsics = source.get("intrinsics")
        if not isinstance(intrinsics, Mapping):
            errors.append(f"native_task_runtime_camera_intrinsics_invalid:{role}")
            intrinsics = {}
        intrinsic_values: dict[str, float | int] = {}
        try:
            intrinsic_values = {
                "fx": float(intrinsics["fx"]),
                "fy": float(intrinsics["fy"]),
                "cx": float(intrinsics["cx"]),
                "cy": float(intrinsics["cy"]),
                "width": int(intrinsics["width"]),
                "height": int(intrinsics["height"]),
            }
        except (KeyError, TypeError, ValueError):
            errors.append(f"native_task_runtime_camera_intrinsics_invalid:{role}")
        else:
            numeric = list(intrinsic_values.values())
            if (
                any(not math.isfinite(float(item)) for item in numeric)
                or intrinsic_values["fx"] <= 0
                or intrinsic_values["fy"] <= 0
                or intrinsic_values["width"] <= 0
                or intrinsic_values["height"] <= 0
            ):
                errors.append(f"native_task_runtime_camera_intrinsics_invalid:{role}")
        rows.append(
            {
                "role": role,
                "policy_input": expected_policy,
                "review_only": role == "overview",
                "scoring_input": False,
                "pose_frame": frame,
                "frame_from_camera_matrix": matrix,
                "intrinsics": intrinsic_values,
            }
        )
    return rows


def materialize_native_task_runtime_contract(
    *,
    scene_id: str,
    task_id: str,
    task_spec: Mapping[str, Any],
    assets: Sequence[Mapping[str, Any]],
    robot_base_pose_world: Mapping[str, Any],
    cameras: Sequence[Mapping[str, Any]],
    scenario_cell_id: str,
    scenario_instance_digest: str,
    seed: int,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and freeze one native scene/task request before execution."""

    errors: list[str] = []
    scene = str(scene_id or "").strip()
    task = str(task_id or "").strip()
    if not scene:
        errors.append("native_task_runtime_scene_id_missing")
    if not task:
        errors.append("native_task_runtime_task_id_missing")
    if not isinstance(task_spec, Mapping):
        raise NativeTaskRuntimeContractError(["native_task_runtime_task_spec_invalid"])
    task_kind = str(task_spec.get("task_kind") or "")
    if task_kind not in TASK_KINDS:
        errors.append("native_task_runtime_task_kind_invalid")
    if not str(scenario_cell_id or "").strip():
        errors.append("native_task_runtime_scenario_cell_missing")
    if not _digest(scenario_instance_digest):
        errors.append("native_task_runtime_scenario_digest_invalid")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errors.append("native_task_runtime_seed_invalid")

    asset_rows, by_asset_role = _asset_rows(assets, errors=errors)
    composition: dict[str, Any] = {}
    if {"scene_collision", "task_object"}.issubset(by_asset_role):
        try:
            composition = plan_articulated_runtime_composition(
                task_spec=task_spec,
                twin_usd_filename=str(by_asset_role["task_object"].get("filename") or ""),
                scene_collision_filename=str(
                    by_asset_role["scene_collision"].get("filename") or ""
                ),
                appearance_filename=(
                    str(by_asset_role["scene_appearance"].get("filename") or "")
                    if "scene_appearance" in by_asset_role
                    else None
                ),
                twin_position_world_m=(
                    by_asset_role["task_object"].get("pose_world") or {}
                ).get("position_world_m"),
            )
        except ValueError as exc:
            errors.append(f"native_task_runtime_composition_invalid:{exc}")
    if composition:
        planned = {row["semantic_role"]: row for row in composition["objects"]}
        for row in asset_rows:
            role = row["semantic_role"]
            row["object_type"] = planned[role]["object_type"]
            row["visible"] = planned[role]["visible"]

    robot_pose = _pose(
        robot_base_pose_world,
        error="native_task_runtime_robot_base_pose_invalid",
        errors=errors,
    )
    camera_rows = _camera_rows(cameras, errors=errors)
    if errors:
        raise NativeTaskRuntimeContractError(errors)

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "scene_id": scene,
        "task_id": task,
        "task_kind": task_kind,
        "task_spec": json.loads(json.dumps(task_spec, sort_keys=True)),
        "task_spec_digest": canonical_digest(dict(task_spec)),
        "scenario": {
            "cell_id": str(scenario_cell_id),
            "instance_digest": str(scenario_instance_digest),
            "seed": seed,
        },
        "candidate_ids": list(FROZEN_CANDIDATES),
        "objects": asset_rows,
        "robot": {
            "robot_id": "franka_panda",
            "base_pose_world": robot_pose,
            "action_seam": {
                "kind": "joint_position_with_gripper",
                "arm_joint_count": 7,
                "action_dimension": 8,
                "gripper_command_source": "native_readback_resolved",
            },
        },
        "cameras": camera_rows,
        "task_sample_binding": composition["task_sample_binding"],
        "reset_contract": {
            "same_scene_bytes": True,
            "same_object_bytes": True,
            "same_cameras": True,
            "same_scorer": True,
            "native_state_readback_required_after_reset": True,
        },
        "scoring_contract": {
            "source": "deterministic_simulator_state",
            "policy_may_grade_itself": False,
            "overview_camera_may_score": False,
        },
        "runtime_readback_required": {
            "object_types": True,
            "asset_digests": True,
            "world_poses": True,
            "camera_transforms_and_intrinsics": True,
            "scenario_parameters": True,
            "task_joint_indices": task_kind == "articulated_open_close",
        },
        "claim_boundary": {
            "valid_contract_is_not_native_application_proof": True,
            "simulator_execution_is_not_physical_truth": True,
            "public_dataset_rehearsal_only": True,
        },
        "contract_digest": "",
    }
    document["contract_digest"] = canonical_digest(
        document, digest_field="contract_digest"
    )
    if destination is not None:
        write_json(Path(destination), document)
    return json.loads(json.dumps(document))


def load_native_task_runtime_contract(path: str | Path) -> dict[str, Any]:
    """Load a materialized contract and reject schema or digest drift."""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeTaskRuntimeContractError(
            ["native_task_runtime_contract_unreadable"]
        ) from exc
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise NativeTaskRuntimeContractError(["native_task_runtime_contract_schema_invalid"])
    expected = canonical_digest(document, digest_field="contract_digest")
    if document.get("contract_digest") != expected:
        raise NativeTaskRuntimeContractError(["native_task_runtime_contract_digest_invalid"])
    return document


__all__ = [
    "FROZEN_CANDIDATES",
    "NativeTaskRuntimeContractError",
    "SCHEMA_VERSION",
    "load_native_task_runtime_contract",
    "materialize_native_task_runtime_contract",
]
