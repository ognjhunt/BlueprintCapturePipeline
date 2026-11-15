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

Arithmetic only: it reads arrays and returns new positions and a receipt. ``streamed_clearance_plan``
reads the field in blocks of ``CHUNK_ROWS`` particles and keeps only the pressed particles' indices
and drops (at most ``MAXIMUM_PRESSED_SHARE`` of the field), so a large field is measured in a fixed
working set; ``clearance_plan`` is the same computation over whole arrays.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
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
CHUNK_ROWS = 1 << 18

# Rows ``[start, stop)`` of one field array, as float64.
Rows = Callable[[int, int], np.ndarray]


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


def _plane_and_top(row: Mapping[str, Any]) -> tuple[float, float]:
    """The volume's plane and the highest centre it presses."""
    if row["kind"] == "box":
        pose = np.asarray(row["pose_world"], float)
        return float(pose[2] + np.asarray(row["minimum_m"], float)[2]), float(pose[2] + np.asarray(row["maximum_m"], float)[2])
    centre = np.asarray(row["surface_position_world_m"], float)
    return float(centre[2]), float(centre[2]) + float(row["thickness_m"]) + MARK_BAND_M


def _over(row: Mapping[str, Any], positions: np.ndarray) -> np.ndarray:
    """Centres over the volume's own outline."""
    if row["kind"] == "box":
        pose = np.asarray(row["pose_world"], float)
        x, y, z, w = pose[3:] / np.linalg.norm(pose[3:])
        rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                             [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                             [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        local = (positions - pose[:3]) @ rotation
        low, high = np.asarray(row["minimum_m"], float), np.asarray(row["maximum_m"], float)
        return np.all((local[:, :2] >= low[:2]) & (local[:, :2] <= high[:2]), axis=1)
    centre = np.asarray(row["surface_position_world_m"], float)
    return np.linalg.norm(positions[:, :2] - centre[:2], axis=1) <= float(row["radius_m"])


def _spans(count: int, rows: int) -> list[tuple[int, int]]:
    return [(start, min(start + rows, count)) for start in range(0, count, rows)]


def apply_press(positions: np.ndarray, start: int, plan: Mapping[str, Any]) -> np.ndarray:
    """Rows ``start..`` of the field's world positions with the plan's press applied (a new array)."""
    indices, drops = plan["pressed_indices"], plan["drops"]
    derived = np.array(positions, dtype=np.float64, copy=True)
    low, high = np.searchsorted(indices, [start, start + len(derived)])
    derived[indices[low:high] - start, 2] -= drops[low:high]
    return derived


def _digest(blocks: Sequence[Callable[[], np.ndarray]]) -> str:
    """``arrays_digest`` of arrays read block by block (all positions, then all opacities)."""
    digest = hashlib.sha256()
    for block in blocks:
        digest.update(np.ascontiguousarray(np.asarray(block(), dtype="<f4")).tobytes())
    return "sha256:" + digest.hexdigest()


def streamed_clearance_plan(*, count: int, positions: Rows, sigma: Rows, sigma_shape: tuple[int, ...],
                            opacities: Rows, opacity_shape: tuple[int, ...], volumes: Sequence[Mapping[str, Any]],
                            rows: int = CHUNK_ROWS) -> dict[str, Any]:
    """``clearance_plan``'s receipt, read in blocks: ``{"receipt", "pressed_indices", "drops"}``.

    ``positions(start, stop)`` gives world positions (rows × 3), ``sigma`` and ``opacities`` the
    matching rows of the other two arrays, whose whole shapes are ``sigma_shape`` and
    ``opacity_shape``. The checks run in ``clearance_plan``'s order and raise its errors. Only the
    pressed particles' indices and drops are kept; ``apply_press`` gives the derived positions.
    """
    spans = _spans(count, rows)

    def every(read: Rows, test: Callable[[np.ndarray], bool]) -> bool:
        return all(test(read(start, stop)) for start, stop in spans)

    def finite(block: np.ndarray) -> bool:
        return bool(np.isfinite(block).all())

    if not every(positions, finite):
        raise SceneAppearanceClearanceError("scene_appearance_clearance_positions_not_finite")
    if tuple(sigma_shape) != (count,):
        raise SceneAppearanceClearanceError("scene_appearance_clearance_sigma_shape_invalid")
    if not every(sigma, finite):
        raise SceneAppearanceClearanceError("scene_appearance_clearance_sigma_not_finite")
    if tuple(opacity_shape) != (count,):
        raise SceneAppearanceClearanceError("scene_appearance_clearance_opacities_shape_invalid")
    if not every(opacities, finite):
        raise SceneAppearanceClearanceError("scene_appearance_clearance_opacities_not_finite")
    if (not every(sigma, lambda block: not (block < 0).any())
            or not every(opacities, lambda block: not ((block < 0).any() or (block > 1).any()))):
        raise SceneAppearanceClearanceError("scene_appearance_clearance_values_out_of_range")
    volume_rows = _validated(volumes)
    bands = [_plane_and_top(row) for row in volume_rows]
    reaching = [0] * len(volume_rows)
    pressed_total = too_far_total = 0
    kept_indices: list[np.ndarray] = []
    kept_drops: list[np.ndarray] = []
    over_bound = False
    for start, stop in spans:
        block, block_sigma = positions(start, stop), sigma(start, stop)
        drop = np.zeros(stop - start)
        for number, (row, (plane, top)) in enumerate(zip(volume_rows, bands)):
            over = _over(row, block)
            reach = block[:, 2] + SIGMA_REACH * block_sigma
            hit = over & (block[:, 2] <= top) & (reach > plane - CLEARANCE_M)
            needed = np.where(hit, reach - (plane - CLEARANCE_M), 0.0)
            drop = np.maximum(drop, needed)
            reaching[number] += int(hit.sum())
        too_far = drop > MAXIMUM_PRESS_M
        pressed = (drop > 0) & ~too_far
        too_far_total += int(too_far.sum())
        pressed_total += int(pressed.sum())
        # Past the share bound nothing is pressed, so nothing more needs keeping.
        over_bound = over_bound or float(pressed_total) / max(count, 1) > MAXIMUM_PRESSED_SHARE
        if not over_bound:
            kept_indices.append(np.flatnonzero(pressed) + start)
            kept_drops.append(drop[pressed])
    per_volume = [{**dict(row), "plane_z_m": plane, "highest_pressed_centre_z_m": top, "reaching_count": hits}
                  for row, (plane, top), hits in zip(volume_rows, bands, reaching)]
    share = float(pressed_total) / max(count, 1)
    flags: list[str] = []
    indices, moved = np.zeros(0, dtype=np.int64), np.zeros(0)
    if not pressed_total:
        status = "nothing_reaches"
    elif share > MAXIMUM_PRESSED_SHARE:
        status, pressed_total = "too_large_not_pressed", 0
        flags.append(FLAG_TOO_LARGE)
    else:
        status = "pressed"
        indices, moved = np.concatenate(kept_indices).astype(np.int64), np.concatenate(kept_drops)
    plan = {"pressed_indices": indices, "drops": moved}
    source_digest = _digest([*(lambda a=a, b=b: positions(a, b) for a, b in spans),
                             *(lambda a=a, b=b: opacities(a, b) for a, b in spans)])
    derived_digest = source_digest
    if pressed_total:
        derived_digest = _digest([*(lambda a=a, b=b: apply_press(positions(a, b), a, plan) for a, b in spans),
                                  *(lambda a=a, b=b: opacities(a, b) for a, b in spans)])
    receipt = {
        "schema_version": SCHEMA_VERSION, "status": status, "flags": flags, "method": METHOD,
        "particle_count": int(count), "pressed_count": int(pressed_total), "pressed_share": share,
        "left_too_far_count": int(too_far_total),
        "press_m": ({"median": float(np.median(moved)), "p90": float(np.percentile(moved, 90)),
                     "maximum": float(moved.max())} if len(moved) else None),
        "rule": {"sigma_reach": SIGMA_REACH, "clearance_m": CLEARANCE_M, "mark_band_m": MARK_BAND_M,
                 "maximum_press_m": MAXIMUM_PRESS_M, "maximum_pressed_share": MAXIMUM_PRESSED_SHARE},
        "volumes": per_volume,
        "source_arrays_digest": source_digest,
        "derived_arrays_digest": derived_digest,
        "arrays_digest_basis": "sha256 of positions then opacities as little-endian float32",
        "colour_and_opacity_changed": False, "outside_volumes_changed": False, "source_assets_mutated": False,
    }
    return {**plan, "receipt": receipt}


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
    plan = streamed_clearance_plan(
        count=count, positions=lambda a, b: positions[a:b], sigma=lambda a, b: sigma[a:b], sigma_shape=sigma.shape,
        opacities=lambda a, b: opacity[a:b], opacity_shape=opacity.shape, volumes=volumes)
    pressed = np.zeros(count, dtype=bool)
    pressed[plan["pressed_indices"]] = True
    return {"positions_world": apply_press(positions, 0, plan), "pressed": pressed, "receipt": plan["receipt"]}


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


__all__ = ["CHUNK_ROWS", "cleared_quality_is_qualified", "CLEARANCE_M", "FLAG_TOO_LARGE", "MARKER_THICKNESS_M", "MARK_BAND_M", "MAXIMUM_PRESSED_SHARE",
           "MAXIMUM_PRESS_M", "METHOD", "SCHEMA_VERSION", "SIGMA_REACH", "SceneAppearanceClearanceError",
           "apply_press", "arrays_digest", "clearance_plan", "streamed_clearance_plan", "task_volumes"]
