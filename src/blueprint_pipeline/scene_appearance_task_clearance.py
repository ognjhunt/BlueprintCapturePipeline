"""Press the room's particles down out of the task object and its mark, keeping the table there (CPU).

The room's particles under a placed task object reach up through the plane it rests on, and those
under the target mark reach up over the flat mark, so the robots' cameras see table colour in front
of the object's lower edge and over the mark (capture-447ae4f4: 5.4 % of the tray and 14.2 % of the
mark in the external camera). Two solids cannot share a place, so those particles are moved
straight down, keeping their colour and size, until their extent (``SIGMA_REACH`` of their largest
standard deviation) stays ``CLEARANCE_M`` under the plane. The table is then still there when the
object is lifted away; nothing is cut out.

Which particles: those whose centre lies inside the object's start footprint, at most its own height
above the plane, or inside the mark's own outline (its radius, no margin) at most ``MARK_BAND_M``
above its plane, and whose extent reaches above the plane. Anything higher (a real thing above the
object or standing on the mark) is left. A particle that would have to move further than
``MAXIMUM_PRESS_M`` is left as it is and counted. Nothing outside those volumes changes.

The bound: pressing more than ``MAXIMUM_PRESSED_SHARE`` of the field (1 %) presses nothing and raises
``scene_appearance_clearance_too_large`` (a flag, never a stop).

Arithmetic only: it reads arrays and returns new positions and a receipt.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SCHEMA_VERSION = "scene_appearance_task_clearance.v3"
SIGMA_REACH = 3.0
CLEARANCE_M = 0.001
MARK_BAND_M = 0.005
MAXIMUM_PRESS_M = 0.10
MAXIMUM_PRESSED_SHARE = 0.01
MARKER_THICKNESS_M = 0.002
FLAG_TOO_LARGE = "scene_appearance_clearance_too_large"
METHOD = "pressed_down_keeping_colour"


class SceneAppearanceClearanceError(ValueError):
    """The arrays or the volumes cannot be read as clearance inputs."""


def _array(value: Any, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != len(shape) or any(want is not None and got != want for got, want in zip(array.shape, shape)):
        raise SceneAppearanceClearanceError(f"scene_appearance_clearance_{name}_shape_invalid")
    if not np.isfinite(array).all():
        raise SceneAppearanceClearanceError(f"scene_appearance_clearance_{name}_not_finite")
    return array


def arrays_digest(positions: Any, opacities: Any) -> str:
    """The digest of a field's positions and opacities as the float32 arrays USD stores."""
    digest = hashlib.sha256()
    for value in (positions, opacities):
        digest.update(np.ascontiguousarray(np.asarray(value, dtype="<f4")).tobytes())
    return "sha256:" + digest.hexdigest()


def task_volumes(task_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The object's start footprint and height on its plane, and the visible mark, from a compiled task spec."""
    volumes: list[dict[str, Any]] = []
    pose, bounds = task_spec.get("start_pose_world"), task_spec.get("subject_collision_bounds_scoring_frame_m")
    if pose is not None and bounds is not None:
        volumes.append({"kind": "box", "role": "task_object", "pose_world": [float(v) for v in pose],
                        "minimum_m": [float(v) for v in bounds["minimum"]],
                        "maximum_m": [float(v) for v in bounds["maximum"]]})
    marker = task_spec.get("visible_target_marker")
    if isinstance(marker, Mapping) and marker.get("surface_position_world_m") is not None:
        volumes.append({"kind": "disc", "role": "task_target_marker",
                        "surface_position_world_m": [float(v) for v in marker["surface_position_world_m"]],
                        "radius_m": float(marker["radius_m"]), "thickness_m": MARKER_THICKNESS_M})
    return volumes


def _validated(volumes: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not volumes:
        raise SceneAppearanceClearanceError("scene_appearance_clearance_volumes_missing")
    for row in volumes:
        if row.get("kind") == "box":
            pose, low, high = (np.asarray(row.get(key), float) for key in ("pose_world", "minimum_m", "maximum_m"))
            if (pose.shape != (7,) or low.shape != (3,) or high.shape != (3,) or not (low < high).all()
                    or not np.isfinite([*pose, *low, *high]).all() or np.linalg.norm(pose[3:]) < 1e-9):
                raise SceneAppearanceClearanceError("scene_appearance_clearance_box_invalid")
        elif row.get("kind") == "disc":
            centre = np.asarray(row.get("surface_position_world_m"), float)
            radius, thickness = float(row.get("radius_m", 0)), float(row.get("thickness_m", 0))
            if centre.shape != (3,) or not np.isfinite(centre).all() or not radius > 0 or not thickness > 0:
                raise SceneAppearanceClearanceError("scene_appearance_clearance_disc_invalid")
        else:
            raise SceneAppearanceClearanceError("scene_appearance_clearance_volume_kind_invalid")
    return list(volumes)


def _footprint_and_band(row: Mapping[str, Any], positions: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Centres over the volume's own outline, and its plane and the highest centre it presses."""
    if row["kind"] == "box":
        pose = np.asarray(row["pose_world"], float)
        x, y, z, w = pose[3:] / np.linalg.norm(pose[3:])
        rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                             [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                             [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        local = (positions - pose[:3]) @ rotation
        low, high = np.asarray(row["minimum_m"], float), np.asarray(row["maximum_m"], float)
        over = np.all((local[:, :2] >= low[:2]) & (local[:, :2] <= high[:2]), axis=1)
        plane = float(pose[2] + low[2])
        return over, plane, float(pose[2] + high[2])
    centre = np.asarray(row["surface_position_world_m"], float)
    over = np.linalg.norm(positions[:, :2] - centre[:2], axis=1) <= float(row["radius_m"])
    return over, float(centre[2]), float(centre[2]) + float(row["thickness_m"]) + MARK_BAND_M


def clearance_plan(*, positions_world: Any, sigma_world: Any, opacities: Any,
                   volumes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The pressed-down positions and the receipt (opacities are never changed).

    ``positions_world`` (N×3, simulator frame, Z up, metres), ``sigma_world`` (N, each particle's
    largest standard deviation, metres), ``opacities`` (N, in [0, 1], for the digests only).
    """
    positions = _array(positions_world, (None, 3), "positions")
    count = positions.shape[0]
    sigma = _array(sigma_world, (count,), "sigma")
    opacity = _array(opacities, (count,), "opacities")
    if (sigma < 0).any() or (opacity < 0).any() or (opacity > 1).any():
        raise SceneAppearanceClearanceError("scene_appearance_clearance_values_out_of_range")
    rows = _validated(volumes)
    drop = np.zeros(count)
    per_volume = []
    for row in rows:
        over, plane, top = _footprint_and_band(row, positions)
        reach = positions[:, 2] + SIGMA_REACH * sigma
        hit = over & (positions[:, 2] <= top) & (reach > plane - CLEARANCE_M)
        needed = np.where(hit, reach - (plane - CLEARANCE_M), 0.0)
        drop = np.maximum(drop, needed)
        per_volume.append({**dict(row), "plane_z_m": plane, "highest_pressed_centre_z_m": top,
                           "reaching_count": int(hit.sum())})
    too_far = drop > MAXIMUM_PRESS_M
    pressed = (drop > 0) & ~too_far
    share = float(pressed.sum()) / max(count, 1)
    flags: list[str] = []
    if not pressed.any():
        status = "nothing_reaches"
    elif share > MAXIMUM_PRESSED_SHARE:
        status, pressed = "too_large_not_pressed", np.zeros(count, dtype=bool)
        flags.append(FLAG_TOO_LARGE)
    else:
        status = "pressed"
    derived = positions.copy()
    derived[pressed, 2] -= drop[pressed]
    moved = drop[pressed]
    source_digest = arrays_digest(positions, opacity)
    receipt = {
        "schema_version": SCHEMA_VERSION, "status": status, "flags": flags, "method": METHOD,
        "particle_count": int(count), "pressed_count": int(pressed.sum()), "pressed_share": share,
        "left_too_far_count": int(too_far.sum()),
        "press_m": ({"median": float(np.median(moved)), "p90": float(np.percentile(moved, 90)),
                     "maximum": float(moved.max())} if len(moved) else None),
        "rule": {"sigma_reach": SIGMA_REACH, "clearance_m": CLEARANCE_M, "mark_band_m": MARK_BAND_M,
                 "maximum_press_m": MAXIMUM_PRESS_M, "maximum_pressed_share": MAXIMUM_PRESSED_SHARE},
        "volumes": per_volume,
        "source_arrays_digest": source_digest,
        "derived_arrays_digest": arrays_digest(derived, opacity) if pressed.any() else source_digest,
        "arrays_digest_basis": "sha256 of positions then opacities as little-endian float32",
        "colour_and_opacity_changed": False, "outside_volumes_changed": False, "source_assets_mutated": False,
    }
    return {"positions_world": derived, "pressed": pressed, "receipt": receipt}


def cleared_quality_is_qualified(quality: Any, clearance: Any, *, appearance_sha256: Any) -> bool:
    """A field whose positions the clearance moved: qualified only with that clearance's own receipt.

    The quality must be otherwise qualified, say it was mutated by this clearance, and name the
    receipt; the receipt must verify by its digest, be ``pressed``, and its derived digest must be
    the appearance file the packet carries. Anything else that changed the arrays is refused.
    """
    from .decision_evidence_contracts import canonical_digest
    from .gaussian_field_quality import SCHEMA_VERSION as QUALITY_SCHEMA

    if not isinstance(quality, Mapping) or not isinstance(clearance, Mapping):
        return False
    mutation = quality.get("mutation")
    file_digest = str(appearance_sha256 or "")
    file_digest = file_digest if file_digest.startswith("sha256:") else "sha256:" + file_digest
    try:
        receipt_ok = clearance.get("receipt_digest") == canonical_digest(dict(clearance), digest_field="receipt_digest")
    except (TypeError, ValueError):
        receipt_ok = False
    return bool(
        quality.get("schema_version") == QUALITY_SCHEMA and quality.get("status") == "qualified"
        and quality.get("blockers") == [] and quality.get("learned_tensors_mutated") is True
        and isinstance(mutation, Mapping) and mutation.get("kind") == "scene_appearance_task_clearance"
        and receipt_ok and clearance.get("status") == "pressed"
        and mutation.get("clearance_receipt_digest") == clearance.get("receipt_digest")
        and mutation.get("derived_particlefield_digest") == clearance.get("derived_particlefield_digest") == file_digest
        and mutation.get("source_particlefield_digest") == clearance.get("source_particlefield_digest")
        and isinstance(clearance.get("source_particlefield_digest"), str)
        and clearance.get("source_particlefield_digest") != file_digest)


__all__ = ["cleared_quality_is_qualified", "CLEARANCE_M", "FLAG_TOO_LARGE", "MARKER_THICKNESS_M", "MARK_BAND_M", "MAXIMUM_PRESSED_SHARE",
           "MAXIMUM_PRESS_M", "METHOD", "SCHEMA_VERSION", "SIGMA_REACH", "SceneAppearanceClearanceError",
           "arrays_digest", "clearance_plan", "task_volumes"]
