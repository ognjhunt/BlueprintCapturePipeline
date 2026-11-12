"""How well the placed collision box fits the object's own masks in the site's frames (a box-fit check).

What this is, and what it is not. The placed box is built from the masks lifted through the
registration (``website_task_preparation``: the subject's bounds are the masks' points mapped by
``source_to_runtime``), and this check projects it back through the inverse of the same matrix.
The registration's error cancels, so this cannot see where the registration put the object in the
room. It measures only how well an axis-aligned box fits the masks: overlap, the offset between
the box's and the masks' centroids read on the support plane at the box's height, the image offset,
and the size ratio per horizontal axis (frames whose mask touches the border give no centre or
size). Its flags say that, and nothing about placement in the room:

- ``task_object_box_fit_offset``: the median centroid offset exceeds the threshold;
- ``task_object_box_fit_size``: a median axis ratio is outside 1 ± ``SIZE_RATIO_TOLERANCE``.

The threshold is the preparation's stated placement uncertainty, never less than
``MINIMUM_THRESHOLD_M`` (so a tiny stated figure cannot flag every job). What is known about
placement in the room is the registration's own residual near the task, published separately
(``placement_uncertainty``, 7.2 cm on capture-447ae4f4). The independent check of height is the
collider against the particles and the site's depth (``website_support_height_agreement``).
Flags only: a check that cannot run is recorded as not assessed, never a stop.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SCHEMA_VERSION = "website_task_box_fit_check.v1"
#: The threshold never falls below this: half of the 10 cm error the check is meant to see.
MINIMUM_THRESHOLD_M = 0.05
DEFAULT_PLACEMENT_UNCERTAINTY_M = 0.05
SIZE_RATIO_TOLERANCE = 0.15
#: Below this the frames do not pin the centre down; size is then read with the box as placed.
MINIMUM_TRIANGULATION_CONDITIONING = 0.01
MINIMUM_AXIS_PIXELS = 40.0
MINIMUM_USABLE_FRAMES = 3
#: Frames whose offset lies within this angle of the mean offset count as "the same direction".
SAME_DIRECTION_DEGREES = 45.0
FLAG_PLACEMENT = "task_object_box_fit_offset"
FLAG_SIZE = "task_object_box_fit_size"


def _hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(map(tuple, points))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], float)


def _fill(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pixels whose centre lies inside a convex counter-clockwise polygon."""
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    lo = np.floor(polygon.min(axis=0)).astype(int)
    hi = np.ceil(polygon.max(axis=0)).astype(int)
    x0, x1, y0, y1 = max(lo[0], 0), min(hi[0], width), max(lo[1], 0), min(hi[1], height)
    if x0 >= x1 or y0 >= y1:
        return mask
    xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    inside = np.ones(xs.shape, dtype=bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        inside &= (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0]) >= 0
    mask[y0:y1, x0:x1] = inside
    return mask


def _extent_along(pixels: np.ndarray, direction: np.ndarray) -> float:
    values = pixels @ direction
    low, high = np.quantile(values, [0.01, 0.99])
    return float(high - low)


def _triangulate(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, float]:
    """The point nearest every ray (least squares) and how well the rays pin it (0 none, 1 best)."""
    system, right = np.zeros((3, 3)), np.zeros(3)
    for origin, direction in zip(origins, directions, strict=True):
        projector = np.eye(3) - np.outer(direction, direction)
        system += projector
        right += projector @ origin
    eigen = np.linalg.eigvalsh(system)
    return np.linalg.solve(system, right), float(eigen[0] / eigen[-1])


def check_video_placement(*, subject_min_sim: Sequence[float], subject_max_sim: Sequence[float],
                          frames: Sequence[Mapping[str, Any]], masks: Mapping[str, np.ndarray],
                          source_to_runtime: Any, runtime_to_simulator: Any,
                          placement_uncertainty_m: float | None = None) -> dict[str, Any]:
    """The placed box against the object's masks in the source frames; see the module docstring.

    ``frames`` carry ``frame_id``, ``intrinsics`` and ``world_from_camera`` (source frame,
    OpenCV camera: +z forward, +y down); ``masks`` maps a frame id to its boolean mask in that
    frame's pixels. The box is axis-aligned in the simulator frame (Z up, metres).

    The centre offset is triangulated from every frame's two centroid rays, so a height
    difference is not mistaken for a horizontal one when the cameras look down at a shallow
    angle. Size is then measured with the box moved onto the observed centre, so a placement
    error is not mistaken for a size error.
    """
    arguments = dict(frames=frames, masks=masks, source_to_runtime=source_to_runtime,
                     runtime_to_simulator=runtime_to_simulator)
    low, high = np.asarray(subject_min_sim, float), np.asarray(subject_max_sim, float)
    placed = _frame_rows(low, high, **arguments)
    usable = [row for row in placed if "_rays" in row]
    offset = None
    if len(usable) >= MINIMUM_USABLE_FRAMES:
        origins = np.array([row["_rays"][0] for row in usable])
        observed, pinned = _triangulate(origins, np.array([row["_rays"][1] for row in usable]))
        box_centre, _ = _triangulate(origins, np.array([row["_rays"][2] for row in usable]))
        offset = {"offset_m": (observed - box_centre).tolist(), "triangulation_conditioning": pinned}
    pinned_down = offset is not None and offset["triangulation_conditioning"] >= MINIMUM_TRIANGULATION_CONDITIONING
    aligned = (_frame_rows(low + offset["offset_m"], high + offset["offset_m"], **arguments)
               if pinned_down else placed)
    for row in placed:
        row.pop("_rays", None)
    for row in aligned:
        row.pop("_rays", None)
    threshold = (float(placement_uncertainty_m) if placement_uncertainty_m is not None
                 and np.isfinite(placement_uncertainty_m) and placement_uncertainty_m > 0
                 else DEFAULT_PLACEMENT_UNCERTAINTY_M)
    threshold = max(threshold, MINIMUM_THRESHOLD_M)
    return summarize(placed, offset=offset, aligned=aligned, placement_threshold_m=threshold)


def _frame_rows(low: np.ndarray, high: np.ndarray, *, frames: Sequence[Mapping[str, Any]],
                masks: Mapping[str, np.ndarray], source_to_runtime: Any,
                runtime_to_simulator: Any) -> list[dict[str, Any]]:
    sim_from_source = np.asarray(runtime_to_simulator, float) @ np.asarray(source_to_runtime, float)
    source_from_sim = np.linalg.inv(sim_from_source)
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    centre = (low + high) / 2
    half = (high - low) / 2
    rows = []
    for frame in frames:
        mask = masks.get(frame["frame_id"])
        if mask is None or not mask.any():
            continue
        K, camera = np.asarray(frame["intrinsics"], float), np.asarray(frame["world_from_camera"], float)
        camera_from_sim = np.linalg.inv(camera) @ source_from_sim

        def project(points: np.ndarray) -> np.ndarray | None:
            local = np.c_[points, np.ones(len(points))] @ camera_from_sim.T
            if (local[:, 2] <= 1e-6).any():
                return None
            image = local[:, :3] @ K.T
            return image[:, :2] / image[:, 2:3]

        projected = project(corners)
        if projected is None:
            continue
        box = _fill(_hull(projected), mask.shape)
        union = int((box | mask).sum())
        row: dict[str, Any] = {"frame_id": frame["frame_id"], "mask_pixels": int(mask.sum()),
                               "box_pixels": int(box.sum()), "overlap": (int((box & mask).sum()) / union) if union else 0.0}
        border = mask[:2].any() or mask[-2:].any() or mask[:, :2].any() or mask[:, -2:].any()
        row["mask_cut_by_frame"] = bool(border)
        if not border and box.any():
            # Centroid rays, met with the plane at the box's mid-height.
            origin = (sim_from_source @ np.r_[camera[:3, 3], 1.0])[:3]
            offsets, rays = [], []
            for silhouette in (mask, box):
                ys, xs = np.nonzero(silhouette)
                ray_camera = np.linalg.inv(K) @ [xs.mean() + 0.5, ys.mean() + 0.5, 1.0]
                ray = sim_from_source[:3, :3] @ (camera[:3, :3] @ ray_camera)
                rays.append(ray / np.linalg.norm(ray))
                if abs(ray[2]) < 1e-9:
                    break
                t = (centre[2] - origin[2]) / ray[2]
                offsets.append(origin + t * ray)
            if len(offsets) == 2:
                row["centre_offset_m"] = (offsets[0] - offsets[1])[:2].tolist()
                row["_rays"] = (origin, *rays)
                (my, mx), (by, bx) = (np.mean(np.nonzero(item), axis=1) for item in (mask, box))
                distance = float(np.linalg.norm(offsets[1] - origin))
                row["image_offset_px"] = [float(mx - bx), float(my - by)]
                row["image_offset_m"] = [float(mx - bx) * distance / K[0, 0], float(my - by) * distance / K[1, 1]]
            ratios = {}
            mask_pixels = np.c_[np.nonzero(mask)[1], np.nonzero(mask)[0]] + 0.5
            box_pixels = np.c_[np.nonzero(box)[1], np.nonzero(box)[0]] + 0.5
            for axis, name in ((0, "x"), (1, "y")):
                step = np.zeros(3)
                step[axis] = half[axis]
                ends = project(np.array([centre - step, centre + step]))
                if ends is None:
                    continue
                span = ends[1] - ends[0]
                if np.linalg.norm(span) < MINIMUM_AXIS_PIXELS:
                    continue
                direction = span / np.linalg.norm(span)
                ratios[name] = _extent_along(mask_pixels, direction) / max(_extent_along(box_pixels, direction), 1e-9)
            row["size_ratio"] = ratios
        rows.append(row)
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], *, offset: Mapping[str, Any] | None = None,
              aligned: Sequence[Mapping[str, Any]] | None = None,
              placement_threshold_m: float = DEFAULT_PLACEMENT_UNCERTAINTY_M) -> dict[str, Any]:
    """Medians over the frames, the triangulated offset and its direction consistency, and the flags."""
    aligned = rows if aligned is None else aligned
    overlaps = [row["overlap"] for row in rows]
    plane = np.array([row["centre_offset_m"] for row in rows if "centre_offset_m" in row]).reshape(-1, 2)
    ratios = {name: [row["size_ratio"][name] for row in aligned if name in (row.get("size_ratio") or {})]
              for name in ("x", "y")}
    summary: dict[str, Any] = {"frames_compared": len(rows), "frames_with_centre": int(len(plane)),
                               "median_overlap": float(np.median(overlaps)) if overlaps else None,
                               "median_overlap_after_centring": (float(np.median([row["overlap"] for row in aligned]))
                                                                 if aligned else None)}
    flags: list[str] = []
    if offset is not None:
        vector = np.asarray(offset["offset_m"], float)
        summary.update(triangulated_offset_m=vector.tolist(),
                       triangulation_conditioning=float(offset["triangulation_conditioning"]))
    if len(plane) >= MINIMUM_USABLE_FRAMES:
        # Per frame, the offset read on the support plane at the box's height: the same direction
        # in most frames is what a correctable placement error looks like.
        distances, mean = np.linalg.norm(plane, axis=1), plane.mean(axis=0)
        summary["median_plane_offset_m"] = float(np.median(distances))
        summary["median_image_offset_px"] = np.median([row["image_offset_px"] for row in rows
                                                       if "image_offset_px" in row], axis=0).tolist()
        summary["median_image_offset_m"] = np.median([row["image_offset_m"] for row in rows
                                                      if "image_offset_m" in row], axis=0).tolist()
        if summary["median_plane_offset_m"] > placement_threshold_m:
            flags.append(FLAG_PLACEMENT)
        if np.linalg.norm(mean) > 1e-9:
            cosines = plane @ mean / np.maximum(distances * np.linalg.norm(mean), 1e-12)
            summary["share_of_frames_offset_in_the_mean_direction"] = float(
                np.mean(cosines >= np.cos(np.radians(SAME_DIRECTION_DEGREES))))
    summary["median_size_ratio"] = {name: float(np.median(values)) for name, values in ratios.items()
                                    if len(values) >= MINIMUM_USABLE_FRAMES}
    if any(abs(value - 1.0) > SIZE_RATIO_TOLERANCE for value in summary["median_size_ratio"].values()):
        flags.append(FLAG_SIZE)
    assessed = len(rows) >= MINIMUM_USABLE_FRAMES
    return {"schema_version": SCHEMA_VERSION,
            "status": "assessed" if assessed else "not_assessed_too_few_masked_frames",
            "flags": flags if assessed else [], "summary": summary, "frames": [dict(row) for row in rows],
            "frames_after_centring": [dict(row) for row in aligned] if aligned is not rows else None,
            "thresholds": {"placement_offset_m": placement_threshold_m,
                           "size_ratio_tolerance": SIZE_RATIO_TOLERANCE, "minimum_axis_pixels": MINIMUM_AXIS_PIXELS,
                           "minimum_usable_frames": MINIMUM_USABLE_FRAMES},
            "basis": "box_fit_to_masks_registration_error_cancels",
            "measures_room_placement": False,
            "stops_preparation": False}


def box_fit_record(preparation: Mapping[str, Any], *, track: Mapping[str, Any],
                          frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How well the placed box fits the object's masks in the site's frames (``website_task_placement_check``).

    Flags only: a check that cannot run is recorded as not assessed, never a stop.
    """
    from .website_track_mask_codec import decode_track_mask

    try:
        masks = {row["source_frame_id"]: decode_track_mask(row) for row in track.get("observations") or []}
        return check_video_placement(
            subject_min_sim=preparation["subject"]["aabb_min_xyz"], subject_max_sim=preparation["subject"]["aabb_max_xyz"],
            frames=frames, masks=masks, source_to_runtime=preparation["registration"]["source_to_runtime"],
            runtime_to_simulator=preparation["coordinate_frame"]["runtime_to_simulator"],
            placement_uncertainty_m=preparation["coordinate_frame"].get("placement_uncertainty_m"))
    except Exception as exc:  # noqa: BLE001 - a flag-only check must never stop a job
        return {"status": "not_assessed", "reason": type(exc).__name__ + ":" + str(exc)[:200], "flags": [],
                "stops_preparation": False}


__all__ = ["box_fit_record", "DEFAULT_PLACEMENT_UNCERTAINTY_M", "FLAG_PLACEMENT", "MINIMUM_THRESHOLD_M", "FLAG_SIZE", "SCHEMA_VERSION", "SIZE_RATIO_TOLERANCE",
           "check_video_placement", "summarize"]
