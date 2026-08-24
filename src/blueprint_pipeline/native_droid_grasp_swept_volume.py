"""Conservatively qualify a DROID open-gripper pose against a grasp patch."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_franka_action_math import grasp_orientation_contact_xyzw
from .rigid_frame_transforms import rotate_vector_xyzw


SCHEMA_VERSION = "native_droid_grasp_swept_volume.v1"
REGISTERED_SCHEMA = "registered_replacement_asset.v1"
AFFORDANCE_SCHEMA = "paired_target_interaction_affordance_candidate.v1"
CONTROLLED_BODY_PATH = "/panda/Gripper/Robotiq_2F_85/base_link"
GRIPPER_PREFIX = "/panda/Gripper/Robotiq_2F_85/"
PAD_PATHS = {
    side: (
        f"{GRIPPER_PREFIX}{side}_inner_finger/"
        "Defeatured_2F_85_PAD_OPEN_fingertipsstep_01"
    )
    for side in ("left", "right")
}


class NativeDroidGraspSweptVolumeError(ValueError):
    """Stable provider-zero grasp-clearance refusal."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: str | Path, code: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeDroidGraspSweptVolumeError(code) from exc
    if source.is_symlink() or not isinstance(value, dict):
        raise NativeDroidGraspSweptVolumeError(code)
    return source, value


def _record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise NativeDroidGraspSweptVolumeError("droid_grasp_file_invalid")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _vector(value: Any, length: int, code: str) -> list[float]:
    try:
        row = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise NativeDroidGraspSweptVolumeError(code) from exc
    if len(row) != length or not all(math.isfinite(item) for item in row):
        raise NativeDroidGraspSweptVolumeError(code)
    return row


def _unit(value: Any, code: str) -> list[float]:
    row = _vector(value, 3, code)
    norm = math.sqrt(sum(item * item for item in row))
    if norm <= 1.0e-9:
        raise NativeDroidGraspSweptVolumeError(code)
    return [item / norm for item in row]


def _rotation(quaternion_xyzw: Sequence[float], np: Any) -> Any:
    return np.column_stack(
        [
            rotate_vector_xyzw(quaternion_xyzw, axis)
            for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        ]
    )


def _box_corners(lower: Any, upper: Any, np: Any) -> Any:
    return np.asarray(
        [
            [x, y, z, 1.0]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=float,
    )


def materialize_native_droid_grasp_swept_volume(
    *,
    robot_usd_path: str | Path,
    expected_robot_sha256: str,
    robot_asset_uri: str,
    registered_asset_receipt_path: str | Path,
    interaction_affordance_path: str | Path,
    output_path: str | Path,
    search_step_m: float = 0.001,
    clearance_margin_m: float = 0.004,
    maximum_standoff_m: float = 0.03,
) -> dict[str, Any]:
    """Find the first conservative open-gripper collision-free standoff."""

    try:
        import numpy as np
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - environment guard
        raise NativeDroidGraspSweptVolumeError(
            "droid_grasp_openusd_runtime_missing"
        ) from exc
    robot_path = Path(robot_usd_path).expanduser().resolve()
    if _sha256(robot_path) != expected_robot_sha256 or not str(robot_asset_uri):
        raise NativeDroidGraspSweptVolumeError("droid_grasp_robot_identity_invalid")
    registered_path, registered = _read(
        registered_asset_receipt_path, "droid_grasp_registered_asset_invalid"
    )
    affordance_path, affordance = _read(
        interaction_affordance_path, "droid_grasp_affordance_invalid"
    )
    output_usd = registered.get("output_usd") or {}
    task_path = Path(str(output_usd.get("path") or "")).expanduser().resolve()
    candidate = affordance.get("candidate") or {}
    if (
        registered.get("schema_version") != REGISTERED_SCHEMA
        or registered.get("receipt_digest")
        != canonical_digest(registered, digest_field="receipt_digest")
        or task_path.is_symlink()
        or not task_path.is_file()
        or task_path.stat().st_size != output_usd.get("size_bytes")
        or _sha256(task_path) != output_usd.get("sha256")
        or affordance.get("schema_version") != AFFORDANCE_SCHEMA
        or affordance.get("receipt_digest")
        != canonical_digest(affordance, digest_field="receipt_digest")
        or (affordance.get("registered_asset") or {}).get("receipt_digest")
        != registered.get("receipt_digest")
        or not str(candidate.get("grasp_collision_patch_prim_path") or "")
        or float(candidate.get("contact_point_to_grasp_collider_surface_m") or 0.0)
        > 0.001
    ):
        raise NativeDroidGraspSweptVolumeError("droid_grasp_input_binding_invalid")
    try:
        step = float(search_step_m)
        margin = float(clearance_margin_m)
        maximum = float(maximum_standoff_m)
    except (TypeError, ValueError) as exc:
        raise NativeDroidGraspSweptVolumeError("droid_grasp_search_invalid") from exc
    if not all(math.isfinite(value) and value > 0.0 for value in (step, margin, maximum)):
        raise NativeDroidGraspSweptVolumeError("droid_grasp_search_invalid")

    robot = Usd.Stage.Open(str(robot_path))
    task = Usd.Stage.Open(str(task_path))
    if robot is None or task is None:
        raise NativeDroidGraspSweptVolumeError("droid_grasp_usd_unreadable")
    purposes = [
        UsdGeom.Tokens.default_,
        UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy,
        UsdGeom.Tokens.guide,
    ]
    cache = UsdGeom.XformCache()
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), purposes, useExtentsHint=False
    )
    body = robot.GetPrimAtPath(CONTROLLED_BODY_PATH)
    if not body.IsValid():
        raise NativeDroidGraspSweptVolumeError("droid_grasp_controlled_body_missing")
    body_world = np.asarray(cache.GetLocalToWorldTransform(body), dtype=float).T
    body_inverse = np.linalg.inv(body_world)
    body_rotation = body_world[:3, :3]
    body_position = body_world[:3, 3]
    pad_centers: dict[str, Any] = {}
    for side, path in PAD_PATHS.items():
        prim = robot.GetPrimAtPath(path)
        box = bounds.ComputeWorldBound(prim).ComputeAlignedBox()
        pad_centers[side] = (
            np.asarray(box.GetMin(), dtype=float)
            + np.asarray(box.GetMax(), dtype=float)
        ) / 2.0
    midpoint = (pad_centers["left"] + pad_centers["right"]) / 2.0
    jaw = pad_centers["left"] - pad_centers["right"]
    open_separation = float(np.linalg.norm(jaw))
    if not 0.09 <= open_separation <= 0.12:
        raise NativeDroidGraspSweptVolumeError("droid_grasp_open_width_invalid")
    measured_quaternion = grasp_orientation_contact_xyzw(
        approach_axis=(midpoint - body_position).tolist(),
        jaw_axis=jaw.tolist(),
    )
    grasp_rotation = _rotation(measured_quaternion, np)
    body_from_grasp_rotation = body_rotation.T @ grasp_rotation
    body_to_grasp_position = body_rotation.T @ (midpoint - body_position)

    approach = np.asarray(
        _unit(candidate.get("gripper_approach_axis_registered_stage"), "droid_grasp_axes_invalid")
    )
    pinch = np.asarray(
        _unit(candidate.get("pinch_axis_registered_stage"), "droid_grasp_axes_invalid")
    )
    lateral_outward = np.asarray(
        _unit(
            candidate.get("grasp_lateral_outward_unit_registered_stage"),
            "droid_grasp_axes_invalid",
        )
    )
    if (
        abs(float(np.dot(approach, pinch))) > 1.0e-6
        or abs(float(np.dot(approach, lateral_outward))) > 1.0e-6
        or abs(float(np.dot(pinch, lateral_outward))) > 1.0e-6
    ):
        raise NativeDroidGraspSweptVolumeError("droid_grasp_axes_invalid")
    grasp_x = np.cross(pinch, approach)
    target_grasp_rotation = np.column_stack([grasp_x, pinch, approach])
    if abs(float(np.linalg.det(target_grasp_rotation)) - 1.0) > 1.0e-6:
        raise NativeDroidGraspSweptVolumeError("droid_grasp_axes_invalid")
    target_body_rotation = target_grasp_rotation @ body_from_grasp_rotation.T
    contact = np.asarray(
        _vector(candidate.get("contact_point_registered_stage_m"), 3, "droid_grasp_contact_invalid")
    )
    outward = -approach

    gripper_rows: list[tuple[str, Any]] = []
    for prim in robot.Traverse():
        path = str(prim.GetPath())
        if not (
            path.startswith(GRIPPER_PREFIX)
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        ):
            continue
        box = bounds.ComputeWorldBound(prim).ComputeAlignedBox()
        corners = _box_corners(
            np.asarray(box.GetMin(), dtype=float),
            np.asarray(box.GetMax(), dtype=float),
            np,
        )
        gripper_rows.append((path, (body_inverse @ corners.T).T))
    task_rows: list[tuple[str, Any, Any]] = []
    for prim in task.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        box = bounds.ComputeWorldBound(prim).ComputeAlignedBox()
        task_rows.append(
            (
                str(prim.GetPath()),
                np.asarray(box.GetMin(), dtype=float),
                np.asarray(box.GetMax(), dtype=float),
            )
        )
    if not gripper_rows or not task_rows:
        raise NativeDroidGraspSweptVolumeError("droid_grasp_collision_set_missing")

    def sample(standoff: float) -> dict[str, Any]:
        grasp_target = contact + outward * standoff
        target_body_position = (
            grasp_target - target_body_rotation @ body_to_grasp_position
        )
        transform = np.eye(4)
        transform[:3, :3] = target_body_rotation
        transform[:3, 3] = target_body_position
        collisions = []
        transformed: dict[str, tuple[Any, Any]] = {}
        for path, relative_corners in gripper_rows:
            world = (transform @ relative_corners.T).T[:, :3]
            lower, upper = world.min(axis=0), world.max(axis=0)
            transformed[path] = (lower, upper)
            if "/left_inner_finger/" in path or "/right_inner_finger/" in path:
                continue
            for task_prim, task_lower, task_upper in task_rows:
                overlap = np.minimum(upper, task_upper) - np.maximum(
                    lower, task_lower
                )
                if bool(np.all(overlap > 0.0)):
                    collisions.append(
                        {
                            "gripper_collision_prim_path": path,
                            "task_collision_prim_path": task_prim,
                            "overlap_m": [float(value) for value in overlap],
                        }
                    )
        return {
            "standoff_m": standoff,
            "forbidden_collisions": collisions,
            "forbidden_collision_count": len(collisions),
            "transformed": transformed,
        }

    samples = []
    clear = None
    count = int(math.floor(maximum / step + 1.0e-9))
    for index in range(count + 1):
        row = sample(round(index * step, 12))
        samples.append(row)
        if row["forbidden_collision_count"] == 0:
            clear = row
            break
    if clear is None:
        raise NativeDroidGraspSweptVolumeError("droid_grasp_no_clear_standoff")
    minimum_clear = float(clear["standoff_m"])
    selected = math.ceil((minimum_clear + margin) / step - 1.0e-9) * step
    if selected > maximum + 1.0e-9:
        raise NativeDroidGraspSweptVolumeError("droid_grasp_margin_unavailable")
    selected_row = sample(selected)
    if selected_row["forbidden_collision_count"]:
        raise NativeDroidGraspSweptVolumeError("droid_grasp_selected_standoff_collides")

    patch_path = str(candidate["grasp_collision_patch_prim_path"])
    patch_prim = task.GetPrimAtPath(patch_path)
    patch_box = bounds.ComputeWorldBound(patch_prim).ComputeAlignedBox()
    patch_corners = _box_corners(
        np.asarray(patch_box.GetMin(), dtype=float),
        np.asarray(patch_box.GetMax(), dtype=float),
        np,
    )[:, :3]
    patch_approach = patch_corners @ approach
    patch_pinch = patch_corners @ pinch
    # ``contact`` is a point on the source-derived rim collider, while the
    # control TCP is the midpoint between the two fingertip pads.  Placing
    # that midpoint on the surface embeds half of each pad in the rim.  PhysX
    # then resolves the overlap by pushing the hand outward: C75 measured an
    # 11.65 mm repeatable lateral residual, and the exact shipped pad geometry
    # below independently measures an 11.00 mm support distance.
    #
    # Determine which end of the patch the authored contact occupies, then
    # move the TCP beyond that free edge by the conservative pad AABB support.
    # The sign is derived from the patch itself rather than assuming a
    # left- or right-hand rim.
    grasp_lateral = target_grasp_rotation[:, 0]
    patch_lateral = patch_corners @ grasp_lateral
    contact_lateral = float(contact @ grasp_lateral)
    distance_to_minimum = abs(contact_lateral - float(patch_lateral.min()))
    distance_to_maximum = abs(contact_lateral - float(patch_lateral.max()))
    if min(distance_to_minimum, distance_to_maximum) > 0.001:
        raise NativeDroidGraspSweptVolumeError(
            "droid_grasp_contact_not_on_lateral_patch_boundary"
        )
    lateral_sign = -1.0 if distance_to_minimum <= distance_to_maximum else 1.0
    patch_derived_lateral_outward = grasp_lateral * lateral_sign
    if float(np.dot(patch_derived_lateral_outward, lateral_outward)) < 1.0 - 1.0e-6:
        raise NativeDroidGraspSweptVolumeError(
            "droid_grasp_lateral_axis_binding_mismatch"
        )
    selected_grasp_target = contact + outward * selected
    pad_lateral_support: dict[str, float] = {}
    for side, path in PAD_PATHS.items():
        lower, upper = selected_row["transformed"][path]
        pad_corners = _box_corners(lower, upper, np)[:, :3]
        projections = (pad_corners - selected_grasp_target) @ lateral_outward
        support = max(0.0, -float(projections.min()))
        if not math.isfinite(support) or support <= 0.0:
            raise NativeDroidGraspSweptVolumeError(
                "droid_grasp_lateral_pad_support_invalid"
            )
        pad_lateral_support[side] = support
    selected_lateral_tcp_offset = max(pad_lateral_support.values())
    if selected_lateral_tcp_offset > maximum + 1.0e-9:
        raise NativeDroidGraspSweptVolumeError(
            "droid_grasp_lateral_pad_support_exceeds_bound"
        )
    pad_reach = {}
    for side, path in PAD_PATHS.items():
        lower, upper = selected_row["transformed"][path]
        pad_corners = _box_corners(lower, upper, np)[:, :3]
        values = pad_corners @ approach
        overlap = min(float(values.max()), float(patch_approach.max())) - max(
            float(values.min()), float(patch_approach.min())
        )
        pad_reach[side] = overlap
    if any(value <= 0.0 for value in pad_reach.values()):
        raise NativeDroidGraspSweptVolumeError("droid_grasp_pads_do_not_reach_patch")
    patch_span = float(patch_pinch.max() - patch_pinch.min())
    if patch_span >= open_separation:
        raise NativeDroidGraspSweptVolumeError("droid_grasp_patch_not_straddled")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "conservative_open_gripper_standoff_qualified",
        "robot_asset": {**_record(robot_path), "uri": str(robot_asset_uri)},
        "registered_asset": {
            **_record(registered_path),
            "receipt_digest": registered["receipt_digest"],
        },
        "interaction_affordance": {
            **_record(affordance_path),
            "receipt_digest": affordance["receipt_digest"],
        },
        "controlled_body_prim_path": CONTROLLED_BODY_PATH,
        "selected_pad_prim_paths": dict(PAD_PATHS),
        "open_pad_separation_m": open_separation,
        "patch_pinch_span_m": patch_span,
        "contact_point_world_m": [float(value) for value in contact],
        "approach_unit_world": [float(value) for value in approach],
        "pinch_axis_world": [float(value) for value in pinch],
        "search_step_m": step,
        "clearance_margin_m": margin,
        "minimum_collision_free_outward_standoff_m": minimum_clear,
        "selected_outward_standoff_m": selected,
        "last_blocked_sample": (
            None
            if len(samples) < 2
            else {
                key: value
                for key, value in samples[-2].items()
                if key != "transformed"
            }
        ),
        "selected_sample": {
            key: value
            for key, value in selected_row.items()
            if key != "transformed"
        },
        "pad_patch_approach_overlap_m": pad_reach,
        "lateral_outward_unit_world": [
            float(value) for value in lateral_outward
        ],
        "lateral_outward_grasp_frame_unit": [lateral_sign, 0.0, 0.0],
        "pad_lateral_surface_support_m": pad_lateral_support,
        "selected_lateral_tcp_surface_offset_m": (
            selected_lateral_tcp_offset
        ),
        "lateral_contact_method": (
            "source_patch_boundary_plus_conservative_pad_aabb_support"
        ),
        "collision_method": "conservative_transformed_world_aabb",
        "native_contact_or_closure_executed": False,
        "provider_mutation_performed": False,
        "claim_boundary": (
            "provider_zero_open_gripper_aabb_clearance_only;not_closed_gripper_"
            "contact_grasp_retention_native_simulator_or_task_success"
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise NativeDroidGraspSweptVolumeError("droid_grasp_destination_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


__all__ = [
    "NativeDroidGraspSweptVolumeError",
    "SCHEMA_VERSION",
    "materialize_native_droid_grasp_swept_volume",
]
