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
is shrunk below what the object's own points show. The new extent is used when at least ``MIN_FRAMES`` frames
contribute and its box fits the masks no worse than the old one; both extents and both box fits are recorded, and
the size is flagged as estimated from the object's own points. An object whose box already fits its masks keeps
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
from .website_task_placement_check import FLAG_SIZE, check_video_placement
from .website_track_mask_codec import decode_track_mask

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
#: Frames that must keep points of the object for its new extent to be read.
MIN_FRAMES = 3
#: As the raw estimate reads its points (``estimate_target_bounds``).
QUANTILES = (0.01, 0.99)
#: The preparation's stated relative error of an estimated dimension (``website_task_preparation``).
DIMENSION_RELATIVE_ERROR = 0.25
BASIS = "object_points_agreeing_across_frames"


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
    known = np.where(valid, depth, np.nan)
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
    before = _fit(low, high, **options)
    if FLAG_SIZE in before["flags"]:
        record = {"schema_version": REVISION, "basis": BASIS, "searched_because": FLAG_SIZE,
                  "before": {**_box(low, high, runtime_to_sim), "box_fit": before}, "physical_measurement": False}
        kept, rows = own_points(track, frames)
        record["frames"] = rows
        views = [points for points in kept if len(points)]
        if len(views) < MIN_FRAMES:
            record.update(status="kept", reason="too_few_frames_with_points_of_the_object")
        else:
            agreed = np.concatenate(resting_views(views)) @ matrix[:3, :3].T + matrix[:3, 3]
            new_low, new_high = (value.tolist() for value in np.quantile(agreed, QUANTILES, axis=0))
            after = _fit(new_low, new_high, **options)
            record["after"] = {**_box(new_low, new_high, runtime_to_sim), "box_fit": after,
                               "points": int(len(agreed)), "frames_with_points": len(views)}
            # Only a measured worse fit keeps the old box: a box too small in the frames to measure is no evidence.
            if after["size_error"] is not None and after["size_error"] > before["size_error"]:
                record.update(status="kept", reason="own_points_fit_the_masks_worse")
            else:
                record.update(status="from_its_own_points", flag=OBJECT_SIZE_FROM_ITS_POINTS)
                low, high = new_low, new_high
    check = published_cross_check(target.get("object_spec"), _box(low, high, runtime_to_sim)["dimensions_m"])
    if check is not None:
        record = {**(record or {"schema_version": REVISION}), "published_cross_check": check}
    return low, high, ({"subject_extent": record} if record else {})
