"""An object's size from the points that belong to it, when its mask spilled onto what lies behind or under it.

Beta journey step: compile (``website_task_preparation``), as the job's captured rigid object is sized. Its box is
the 1st to 99th percentile of every point its masks cover (``website_task_masks.estimate_target_bounds``). A mask
that also takes in the table it stands on, the wall it touches or the room behind its silhouette makes that box
the size of the spill: test job 4's bedside lamp, about 0.2 m across, came out 0.40 x 0.375 x 0.67 m, because 3
of its 33 masks took in the round table under it.

Only where the box-fit check (``website_task_placement_check``) already finds the box's size disagreeing with the
masks (``task_object_box_fit_size``) is the size read again, from the points that belong to the object:

- In each frame, mask pixels within ``EDGE_ERODE_PX`` of a depth discontinuity (depths ``EDGE_SPAN_PX`` either
  side more than ``DISCONTINUITY_RELATIVE`` apart: Atlas depth is smooth across a silhouette, so no one-pixel step
  shows it) are left out: there the object and what is behind it mix.
- In each frame, only the depth cluster nearest the camera that holds at least ``CLUSTER_MIN_SHARE`` of the mask's
  points is kept: what is behind the object's silhouette is farther away. Clusters part where the mask's depth
  histogram (``CLUSTER_BIN_RELATIVE`` bins) falls below ``VALLEY_SHARE`` of the cluster's peak.
- Across frames, a point is kept only when at least ``AGREE_SHARE`` of the other frames that can see it, and at
  least ``AGREE_MIN_FRAMES`` of them, place it inside the object's mask: every point of the object lies inside
  every view's outline of it, and a patch of table or wall one mask took in lies outside the others. A frame whose
  own depth puts a nearer surface at the point's pixel cannot see it and does not vote (a patch of table behind
  the object is behind it from other views too); one whose depth shows a farther surface there looks through the
  point's place, and votes against it, mask or not (``DEPTH_TOLERANCE`` either way).

The kept points are read exactly as before (each view where the object rests, the same percentiles), so nothing
is shrunk below what the object's own points show. The new extent is used only when the views it is read from (at
least the box-fit check's ``MINIMUM_USABLE_FRAMES``) give at least ``MIN_POINTS`` points, and the box-fit check
measures its size and finds it fits the masks better than the old one: a size it cannot measure is not adopted.
Both extents and both box fits are recorded, and an adopted size is flagged as estimated from the object's own
points. An object whose box already fits its masks keeps
its box as it is, with no record.

A published size for the exact model (``website_object_spec_research``) is a cross-check only: one that disagrees
with the box beyond ``DIMENSION_RELATIVE_ERROR`` is flagged, never used in place of what the video shows.
Geometry never stops a job; every value is an estimate from the video, never a measurement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .atlas_scene_estimates import OBJECT_SIZE_DISAGREES_WITH_PUBLISHED, OBJECT_SIZE_FROM_ITS_POINTS
from .local_reconstruction_adapters import _sha256_file
from .website_task_masks import resting_views
from .website_task_placement_check import FLAG_SIZE, MINIMUM_USABLE_FRAMES, check_video_placement
from .website_track_mask_codec import decode_track_mask, mask_runs  # noqa: F401 - mask_runs for callers

REVISION = "website_object_extent.v1"
#: Depths ``EDGE_SPAN_PX`` either side of a pixel further apart than this share of their depth are a silhouette
#: edge (test job 4's lamp masks: 0 to 0.6% of their pixels; their surroundings' steepest 1% go past it).
DISCONTINUITY_RELATIVE, EDGE_SPAN_PX = 0.10, 3
#: Mask pixels this near a silhouette edge mix the object with what is behind it.
EDGE_ERODE_PX = 3
#: The depth histogram's bins (a share of depth: Atlas depth comes in steps of about 1%), where a cluster ends
#: (a bin below this share of its peak), and the share of the mask's points the object's cluster holds at least.
CLUSTER_BIN_RELATIVE, VALLEY_SHARE, CLUSTER_MIN_SHARE = 0.02, 0.1, 0.2
#: A point is the object's when it lies inside the object's mask (grown by ``AGREE_TOLERANCE_PX``) in at least this
#: share of the other frames that see it, and in at least ``AGREE_MIN_FRAMES`` of them.
AGREE_SHARE, AGREE_MIN_FRAMES, AGREE_TOLERANCE_PX = 0.5, 2, 1
#: A point this share nearer or farther than a frame's depth at its pixel is in front of, or behind, what it sees.
DEPTH_TOLERANCE = 0.05
#: Points per frame tested, an even stride over its mask.
MAX_POINTS_PER_FRAME = 4000
#: As the raw estimate reads its points (``estimate_target_bounds``).
QUANTILES = (0.01, 0.99)
#: Fewer points than this have no 1st percentile apart from their least: the read above would be their extremes.
MIN_POINTS = round(1 / QUANTILES[0])
#: The preparation's stated relative error of an estimated dimension (``website_task_preparation``).
DIMENSION_RELATIVE_ERROR = 0.25
BASIS = "object_points_agreeing_across_frames"
#: The outline carve (``outline_hull``): the box is searched this share of its size beyond each side, on this many
#: cells a side (refined over at most ``HULL_PASSES`` passes), and a cell is the object's when all but ``1 - HULL_AGREE_SHARE`` of the frames that see it put it
#: inside the object's mask (grown by ``AGREE_TOLERANCE_PX``). A simple majority is not enough: frames that all
#: look from one side agree on cells far along their line of sight. The tenth left over tolerates a failed mask or
#: an occluded view.
HULL_MARGIN, HULL_STEPS, HULL_AGREE_SHARE, HULL_PASSES = 0.5, 48, 0.9, 4
HULL_BASIS = "object_outline_agreeing_across_frames"


def _frame_points(observation: Mapping[str, Any], frame: Mapping[str, Any]
                  ) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """One frame's mask points that are the object's own surface (edges and farther surfaces left out), in the
    source frame, how many of its pixels each step kept, and its depth (NaN where it has none)."""
    from scipy import ndimage

    if _sha256_file(Path(frame["geometry_path"])) != frame["geometry_digest"]:
        raise ValueError("task_target_geometry_changed")
    mask = decode_track_mask(observation)
    with np.load(frame["geometry_path"], allow_pickle=False) as geometry:
        depth, valid = geometry["depth_m"].astype(float), geometry["valid_mask"]
    if mask.shape != depth.shape:
        raise ValueError("task_target_mask_geometry_mismatch")
    counts = {"mask": int(mask.sum()), "with_depth": int((mask & valid).sum())}
    edge = np.zeros_like(mask)
    padded = np.pad(np.where(valid, depth, np.nan), EDGE_SPAN_PX, mode="edge")
    for axis in (0, 1):
        before, after = (np.take(padded, np.arange(start, start + depth.shape[axis]), axis=axis)
                         for start in (0, 2 * EDGE_SPAN_PX))
        trim = [slice(EDGE_SPAN_PX, -EDGE_SPAN_PX)] * 2
        trim[axis] = slice(None)
        before, after = before[tuple(trim)], after[tuple(trim)]
        with np.errstate(invalid="ignore"):
            edge |= np.abs(after - before) > DISCONTINUITY_RELATIVE * np.fmin(after, before)
    usable = mask & valid & ~ndimage.binary_dilation(edge, iterations=EDGE_ERODE_PX)
    counts["off_edges"] = int(usable.sum())
    ys, xs = np.nonzero(usable)
    chosen = _nearest_cluster(depth[ys, xs]) if len(xs) else None
    counts["nearest_surface"] = 0 if chosen is None else int(chosen.sum())
    known = np.where(valid, depth, np.nan).astype(np.float32)  # every view's depth is held at once
    if chosen is None:
        return np.zeros((0, 3)), counts, known
    xs, ys, values = xs[chosen], ys[chosen], depth[ys, xs][chosen]
    camera = (np.stack([xs, ys, np.ones_like(xs)], axis=1) @ np.linalg.inv(frame["intrinsics"]).T) * values[:, None]
    pose = np.asarray(frame["world_from_camera"], dtype=float)
    return camera @ pose[:3, :3].T + pose[:3, 3], counts, known


def _nearest_cluster(values: np.ndarray) -> np.ndarray | None:
    """Which of one mask's depths are its nearest surface holding at least ``CLUSTER_MIN_SHARE`` of them; None when
    no surface does."""
    logs = np.log(values)
    bins = np.floor((logs - logs.min()) / CLUSTER_BIN_RELATIVE).astype(int)
    counts = np.bincount(bins)
    start, peak = 0, 0
    for index, count in enumerate(np.r_[counts, 0]):
        peak = max(peak, count)
        if count < VALLEY_SHARE * peak:
            members = (bins >= start) & (bins < index)
            if members.sum() >= CLUSTER_MIN_SHARE * len(values):
                return members
            start, peak = index + 1, 0
    return None


def own_points(track: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]) -> tuple[list[np.ndarray], list[dict]]:
    """Each frame's points of the object that the other frames able to see them agree on, in the source frame, and
    per frame how many each step kept."""
    from scipy import ndimage

    by_id = {frame["frame_id"]: frame for frame in frames}
    views = []
    for observation in track.get("observations") or []:
        frame = by_id.get(observation.get("source_frame_id"))
        if frame is None:
            continue
        points, counts, depth = _frame_points(observation, frame)
        step = max(1, -(-len(points) // MAX_POINTS_PER_FRAME))
        grown = ndimage.binary_dilation(decode_track_mask(observation), iterations=AGREE_TOLERANCE_PX)
        views.append({"frame_id": frame["frame_id"], "points": points[::step], "counts": counts, "grown": grown,
                      "depth": depth,
                      "intrinsics": np.asarray(frame["intrinsics"], dtype=float),
                      "camera_from_source": np.linalg.inv(np.asarray(frame["world_from_camera"], dtype=float))})
    kept, rows = [], []
    for index, view in enumerate(views):
        points = view["points"]
        seen, inside = np.zeros(len(points), dtype=int), np.zeros(len(points), dtype=int)
        for other_index, other in enumerate(views):
            if other_index == index or not len(points):
                continue
            camera = points @ other["camera_from_source"][:3, :3].T + other["camera_from_source"][:3, 3]
            front = camera[:, 2] > 1e-6
            pixels = camera @ other["intrinsics"].T
            with np.errstate(divide="ignore", invalid="ignore"):
                u, v = np.rint(pixels[:, 0] / camera[:, 2]), np.rint(pixels[:, 1] / camera[:, 2])
            height, width = other["grown"].shape
            at = np.flatnonzero(front & (u >= 0) & (u < width) & (v >= 0) & (v < height))
            rows_at, columns_at = v[at].astype(int), u[at].astype(int)
            there, distance = other["depth"][rows_at, columns_at], camera[at, 2]
            with np.errstate(invalid="ignore"):
                hidden = distance > there * (1 + DEPTH_TOLERANCE)
                through = distance < there * (1 - DEPTH_TOLERANCE)
            at, rows_at, columns_at, through = at[~hidden], rows_at[~hidden], columns_at[~hidden], through[~hidden]
            seen[at] += 1
            inside[at] += other["grown"][rows_at, columns_at] & ~through
        agreed = (seen >= AGREE_MIN_FRAMES) & (inside >= AGREE_MIN_FRAMES) & (inside >= AGREE_SHARE * seen)
        kept.append(points[agreed])
        rows.append({"frame_id": view["frame_id"], **view["counts"], "tested": int(len(points)),
                     "agreed": int(agreed.sum())})
    return kept, rows


def _sim_box(low: Sequence[float], high: Sequence[float], runtime_to_sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    moved = corners @ runtime_to_sim[:3, :3].T + runtime_to_sim[:3, 3]
    return moved.min(axis=0), moved.max(axis=0)


def _fit(low, high, *, frames, masks, matrix, runtime_to_sim, placement_uncertainty_m) -> dict[str, Any]:
    """The box-fit check's flags and medians for a box (runtime units); a check that cannot run flags nothing."""
    low_sim, high_sim = _sim_box(low, high, runtime_to_sim)
    try:
        fit = check_video_placement(subject_min_sim=low_sim, subject_max_sim=high_sim, frames=frames, masks=masks,
                                    source_to_runtime=matrix, runtime_to_simulator=runtime_to_sim,
                                    placement_uncertainty_m=placement_uncertainty_m)
    except Exception as exc:  # noqa: BLE001 - a flag-only check never stops a job, as box_fit_record
        return {"flags": [], "not_assessed": type(exc).__name__, "size_error": None}
    ratios = fit["summary"].get("median_size_ratio") or {}
    return {"flags": list(fit["flags"]), "median_size_ratio": dict(ratios),
            "median_overlap": fit["summary"].get("median_overlap"),
            "size_error": max((abs(value - 1.0) for value in ratios.values()), default=None)}


def _box(low, high, runtime_to_sim) -> dict[str, Any]:
    low_sim, high_sim = _sim_box(low, high, runtime_to_sim)
    return {"aabb_min_xyz": low_sim.tolist(), "aabb_max_xyz": high_sim.tolist(),
            "dimensions_m": (high_sim - low_sim).tolist()}


def published_cross_check(object_spec: Mapping[str, Any] | None, dimensions_m: Sequence[float]) -> dict[str, Any] | None:
    """The exact model's published size against the box's (simulator metres, z up), or None without one.

    Height against height and the two plan sides against depth and width, both sorted (the object's turn on its
    support is not known). Recorded and flagged when any differs by more than ``DIMENSION_RELATIVE_ERROR``; never
    used as the size."""
    from .website_object_spec_research import published_body_size

    published = published_body_size(object_spec, max_relative_half_range=DIMENSION_RELATIVE_ERROR)
    if published is None or published.get("dimension_match") != "exact_model":
        return None
    estimated = [*sorted(float(value) for value in dimensions_m[:2]), float(dimensions_m[2])]
    stated = [*sorted(float(published[key]) for key in ("depth_m", "width_m")), float(published["height_m"])]
    ratios = [value / reference if reference > 0 else float("inf") for value, reference in zip(estimated, stated)]
    agrees = all(abs(ratio - 1.0) <= DIMENSION_RELATIVE_ERROR for ratio in ratios)
    return {"status": "agrees" if agrees else "disagrees", "dimension_match": "exact_model",
            "published_plan_and_height_m": stated, "estimated_plan_and_height_m": estimated,
            "ratios": [round(ratio, 4) for ratio in ratios], "tolerance": DIMENSION_RELATIVE_ERROR,
            "object_spec_digest": published.get("object_spec_digest"), "used_as_size": False,
            **({} if agrees else {"flag": OBJECT_SIZE_DISAGREES_WITH_PUBLISHED})}


def outline_hull(low: Sequence[float], high: Sequence[float], track: Mapping[str, Any],
                 frames: Sequence[Mapping[str, Any]], matrix: Any) -> dict[str, Any] | None:
    """``_carve`` coarse to fine: each pass carves the last pass's bounds (grown by one of its cells) on a grid as
    many cells a side, until a cell no longer shrinks by half or ``HULL_PASSES`` are done. A spilled box can be
    ten times the object, so one grid over it leaves only a few cells inside the object itself."""
    hull = _carve(low, high, track, frames, matrix, HULL_MARGIN)
    for _ in range(HULL_PASSES - 1):
        if hull is None:
            return None
        cell = np.asarray(hull["cell_runtime_units"])
        finer = _carve(np.asarray(hull["minimum"]) - cell, np.asarray(hull["maximum"]) + cell, track, frames, matrix, 0.0)
        if finer is None or max(finer["cell_runtime_units"]) > 0.5 * max(cell):
            break
        hull = finer
    return hull


def _carve(low: Sequence[float], high: Sequence[float], track: Mapping[str, Any],
           frames: Sequence[Mapping[str, Any]], matrix: Any, margin: float) -> dict[str, Any] | None:
    """The bounds (runtime units) of what nearly every frame that sees it puts inside the object's mask.

    Every point of the object lies inside every view's outline of it, whatever the depth says; a band of table or
    wall one mask took in lies outside the other views' outlines. So the cells inside ``low..high`` grown by
    ``HULL_MARGIN`` that at least ``HULL_AGREE_SHARE`` of the frames seeing them put inside the mask bound the
    object from above, free of depth noise and of spill. Along the cameras' line of sight a carve cannot close in
    (a video from one side never sees behind the object): there it bounds no tighter than the mask's own box. None
    with fewer than ``MINIMUM_USABLE_FRAMES`` frames, or nothing kept."""
    from scipy.ndimage import binary_dilation

    low, high, matrix = np.asarray(low, dtype=float), np.asarray(high, dtype=float), np.asarray(matrix, dtype=float)
    size = np.maximum(high - low, 1e-9)
    axes = [np.linspace(low[i] - size[i] * margin, high[i] + size[i] * margin, HULL_STEPS) for i in range(3)]
    cell = np.array([axis[1] - axis[0] for axis in axes])
    cells = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    inverse = np.linalg.inv(matrix)
    source = cells @ inverse[:3, :3].T + inverse[:3, 3]
    by_id = {str(frame.get("frame_id")): frame for frame in frames}
    seen, inside, views = np.zeros(len(cells), dtype=np.int32), np.zeros(len(cells), dtype=np.int32), 0
    framing = np.zeros(len(cells), dtype=np.int32)
    for row in track.get("observations") or []:
        frame = by_id.get(str(row.get("source_frame_id")))
        if frame is None:
            continue
        mask = decode_track_mask(row)
        if not mask.any():
            continue
        mask = binary_dilation(mask, iterations=AGREE_TOLERANCE_PX) if AGREE_TOLERANCE_PX else mask
        if frame.get("intrinsics") is None or (frame.get("camera_from_world") is None
                                                  and frame.get("world_from_camera") is None):
            continue
        camera = (np.asarray(frame["camera_from_world"], dtype=float) if frame.get("camera_from_world") is not None
                  else np.linalg.inv(np.asarray(frame["world_from_camera"], dtype=float)))
        points = source @ camera[:3, :3].T + camera[:3, 3]
        intrinsics = np.asarray(frame["intrinsics"], dtype=float)
        # Masks may be stored at another resolution than the frame's geometry.
        scale_u = mask.shape[1] / float(frame.get("width") or mask.shape[1])
        scale_v = mask.shape[0] / float(frame.get("height") or mask.shape[0])
        front = points[:, 2] > 1e-9
        depth = np.where(front, points[:, 2], 1.0)
        column, row_at = intrinsics[0, 0] * points[:, 0] / depth + intrinsics[0, 2], intrinsics[1, 1] * points[:, 1] / depth + intrinsics[1, 2]
        u, v = np.rint(column * scale_u).astype(np.int64), np.rint(row_at * scale_v).astype(np.int64)
        framed = front & (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        if not framed.any():
            continue
        framing += framed
        hit = np.zeros(len(cells), dtype=bool)
        hit[framed] = mask[v[framed], u[framed]]
        # A frame whose depth, at a cell's pixel outside the mask, shows something in front of the object (nearer
        # than the object's own nearest depth in that frame) cannot see the cell: an occluder hides it, and the
        # frame does not vote on it. The table under the object or the wall behind it is no nearer than the object,
        # so a frame showing them still votes against a cell inside them (``DEPTH_TOLERANCE`` either way).
        surface = _frame_depth(frame)
        if surface is not None:
            rows_in, columns_in = np.nonzero(mask)
            own = surface[np.clip(np.rint(rows_in / scale_v).astype(np.int64), 0, surface.shape[0] - 1),
                          np.clip(np.rint(columns_in / scale_u).astype(np.int64), 0, surface.shape[1] - 1)]
            own = own[np.isfinite(own)]
            if len(own):
                nearest = float(np.quantile(own, 0.05))
                du, dv = np.rint(column).astype(np.int64), np.rint(row_at).astype(np.int64)
                ask = framed & ~hit & (du >= 0) & (du < surface.shape[1]) & (dv >= 0) & (dv < surface.shape[0])
                there = np.full(len(cells), np.nan)
                there[ask] = surface[dv[ask], du[ask]]
                with np.errstate(invalid="ignore"):
                    framed &= ~((there < nearest * (1 - DEPTH_TOLERANCE)) & (depth > there * (1 + DEPTH_TOLERANCE)))
        views += 1
        seen += framed
        inside += hit & framed
    if views < MINIMUM_USABLE_FRAMES:
        return None
    # A cell most frames cannot see lies inside something else (a table, a wall): a few spilled masks cannot make
    # it the object's. One the object's own outline covers is seen wherever it is framed.
    kept = ((seen >= MINIMUM_USABLE_FRAMES) & (seen >= AGREE_SHARE * framing) & (inside >= HULL_AGREE_SHARE * seen))
    if not kept.any():
        return None
    held = cells[kept]
    # Each kept cell stands for its whole extent: the bounds reach half a cell past the outermost centres.
    return {"minimum": (held.min(axis=0) - cell / 2).tolist(), "maximum": (held.max(axis=0) + cell / 2).tolist(),
            "cells": int(kept.sum()), "views": views, "cell_runtime_units": cell.tolist()}


def _frame_depth(frame: Mapping[str, Any]) -> np.ndarray | None:
    """A frame's depth (NaN where invalid), or None when its geometry is not at hand."""
    try:
        with np.load(str(frame["geometry_path"])) as geometry:
            depth = np.asarray(geometry["depth_m"], dtype=np.float32)
            valid = np.asarray(geometry["valid_mask"], dtype=bool) if "valid_mask" in geometry.files else depth > 0
    except (KeyError, OSError, ValueError, TypeError):
        return None
    return np.where(valid, depth, np.nan)


def own_points_extent(subject_min: Sequence[float], subject_max: Sequence[float], target: Mapping[str, Any],
                      frames: Sequence[Mapping[str, Any]], matrix: Any, runtime_to_sim: Any,
                      placement_uncertainty_m: float | None) -> tuple[list[float], list[float], dict[str, Any]]:
    """Compile's one call: the object's box (runtime units) and the record to keep (``subject_extent``), empty when
    its box already fits its masks and no exact model's size is published."""
    matrix, runtime_to_sim = np.asarray(matrix, dtype=float), np.asarray(runtime_to_sim, dtype=float)
    low, high = list(map(float, subject_min)), list(map(float, subject_max))
    track = target.get("track") or {}
    record: dict[str, Any] = {}
    masks = {row["source_frame_id"]: decode_track_mask(row) for row in track.get("observations") or []}
    options = {"frames": frames, "masks": masks, "matrix": matrix, "runtime_to_sim": runtime_to_sim,
               "placement_uncertainty_m": placement_uncertainty_m}
    hull = outline_hull(low, high, track, frames, matrix)
    if hull is not None:
        # Both boxes bound the object from above: it lies in their overlap. On an axis where they do not
        # overlap the outline carve, read free of depth, is kept.
        new_low = [max(a, b) for a, b in zip(low, hull["minimum"])]
        new_high = [min(a, b) for a, b in zip(high, hull["maximum"])]
        for axis in range(3):
            if new_high[axis] <= new_low[axis]:
                new_low[axis], new_high[axis] = hull["minimum"][axis], hull["maximum"][axis]
        moved = max(max(abs(a - b) for a, b in zip(new_low, low)), max(abs(a - b) for a, b in zip(new_high, high)))
        if moved > max(hull["cell_runtime_units"]):
            record = {"schema_version": REVISION, "basis": HULL_BASIS, "searched_because": "always",
                      "before": _box(low, high, runtime_to_sim), "outline_hull": {**hull, **_box(
                          hull["minimum"], hull["maximum"], runtime_to_sim)}, "after": _box(new_low, new_high, runtime_to_sim),
                      "status": "from_its_own_points", "flag": OBJECT_SIZE_FROM_ITS_POINTS, "physical_measurement": False}
            low, high = new_low, new_high
    before = _fit(low, high, **options)
    if FLAG_SIZE in before["flags"]:
        carved = record.get("outline_hull") and record
        record = {"schema_version": REVISION, "basis": BASIS, "searched_because": FLAG_SIZE,
                  "before": {**_box(low, high, runtime_to_sim), "box_fit": before}, "physical_measurement": False,
                  **({"outline_carve": carved} if carved else {})}
        kept, rows = own_points(track, frames)
        record["frames"] = rows
        views = [points for points in kept if len(points)]
        views = resting_views(views) if len(views) >= MINIMUM_USABLE_FRAMES else views
        if len(views) < MINIMUM_USABLE_FRAMES or sum(len(points) for points in views) < MIN_POINTS:
            record.update(status="kept", reason="too_few_views_or_points_of_the_object",
                          frames_with_points=len(views), points=int(sum(len(points) for points in views)))
        else:
            agreed = np.concatenate(views) @ matrix[:3, :3].T + matrix[:3, 3]
            new_low, new_high = (value.tolist() for value in np.quantile(agreed, QUANTILES, axis=0))
            after = _fit(new_low, new_high, **options)
            record["after"] = {**_box(new_low, new_high, runtime_to_sim), "box_fit": after,
                               "points": int(len(agreed)), "frames_with_points": len(views)}
            if after["size_error"] is None:
                record.update(status="kept", reason="own_points_size_not_measured_in_the_frames")
            elif after["size_error"] >= before["size_error"]:
                record.update(status="kept", reason="own_points_fit_the_masks_no_better")
            else:
                record.update(status="from_its_own_points", flag=OBJECT_SIZE_FROM_ITS_POINTS)
                low, high = new_low, new_high
    if record.get("outline_carve") and record.get("status") == "kept":
        # The points' read was not adopted, but the outline carve before it was: that size is in use, flagged.
        record.update(status="from_its_own_points", flag=OBJECT_SIZE_FROM_ITS_POINTS, points_read=record.pop("reason", None))
    check = published_cross_check(target.get("object_spec"), _box(low, high, runtime_to_sim)["dimensions_m"])
    if check is not None:
        record = {**(record or {"schema_version": REVISION}), "published_cross_check": check}
    return low, high, ({"subject_extent": record} if record else {})
