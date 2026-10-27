"""Automatic approximate scene/object sizes from paired Atlas source observations.

Scale priors stay named assumptions. Original cameras, meshes and depths are never
rescaled in place. The existing scene analyzer supplies robust room coverage.
Height priors are more such assumptions: the capture camera's height above the
floor its views observed, and the standard height of a task surface the task
names. They vote only when the references alone cannot settle the scale.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import cv2
import numpy as np
from PIL import Image

from .atlas_depth_splat import load_posed_rgbd
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .splat_scene_analysis import _robust_minmax


def source_to_posed(source: Path, rgb: np.ndarray) -> dict:
    """Check pixel alignment; never assume the provider preserved the input crop."""
    with Image.open(source) as im:
        original = np.asarray(im.convert("RGB"))
    h, w = rgb.shape[:2]
    sh, sw = original.shape[:2]
    resized = cv2.resize(original, (w, h))
    error = float(np.abs(resized.astype(float) - rgb).mean())
    if abs(sw / sh - w / h) < 0.002 and error < 8:
        matrix = np.array([[w / sw, 0, 0], [0, h / sh, 0], [0, 0, 1.0]])
        return {
            "matrix": matrix.tolist(),
            "source_grid": [sw, sh],
            "method": "pixel_checked_resize",
            "mean_rgb_error": error,
        }
    detector = cv2.ORB_create(nfeatures=3000)
    a, da = detector.detectAndCompute(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY), None)
    b, db = detector.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
    matches = (
        []
        if da is None or db is None
        else [
            x
            for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
            for x in pair[:1]
        ]
    )
    if len(matches) < 12:
        raise ValueError("atlas_source_to_posed_alignment_unavailable")
    p = np.array([a[m.queryIdx].pt for m in matches])
    q = np.array([b[m.trainIdx].pt for m in matches])
    affine, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=2)
    if affine is None or int(inliers.sum()) < 12 or np.linalg.det(affine[:, :2]) <= 0:
        raise ValueError("atlas_source_to_posed_alignment_unavailable")
    matrix = np.vstack([affine, [0, 0, 1.0]]) @ np.diag([w / sw, h / sh, 1])
    residual = float(
        np.median(
            np.linalg.norm(
                p[inliers[:, 0] > 0] @ affine[:, :2].T + affine[:, 2] - q[inliers[:, 0] > 0], axis=1
            )
        )
    )
    return {
        "matrix": matrix.tolist(),
        "source_grid": [sw, sh],
        "method": "source_rgb_feature_registration",
        "median_pixel_residual": residual,
        "inliers": int(inliers.sum()),
    }


def load_views(pose_root: Path, inputs: dict) -> tuple[dict, list[dict]]:
    from .atlas_context_fov import full_view_sources
    result, frames = load_posed_rgbd(pose_root)
    by_id = {r["frame_id"]: r for r in full_view_sources(inputs, pose_root)}
    usage = result["operation_receipt"]["input_usage"]["frames"]
    for frame, row in zip(frames, usage, strict=True):
        source = by_id[frame["frame_id"]]
        if _sha256_file(Path(source["image_path"])) != source["image_digest"]:
            raise ValueError("atlas_automation_source_changed")
        image = next(r for r in row["outputs"] if r["role"] == "image")
        frame.update(image_path=str(pose_root / image["path"]), image_digest=image["sha256"])
        try:
            frame["pixel_alignment"] = source_to_posed(Path(source["image_path"]), frame["rgb"])
        except ValueError as exc:
            # Other views can still supply useful dimensions. A lost alignment
            # never becomes a guessed source box or reference endpoint.
            frame["pixel_alignment"] = {"status": "unavailable", "reason": str(exc)}
    return result, frames


def mapped_pixels(frame: dict, normalized: list) -> np.ndarray:
    alignment = frame["pixel_alignment"]
    if "matrix" not in alignment:
        raise ValueError("atlas_source_to_posed_alignment_unavailable")
    xy = np.asarray(normalized, dtype=float) * alignment["source_grid"]
    return np.column_stack([xy, np.ones(len(xy))]) @ np.asarray(alignment["matrix"]).T[:, :2]


def unproject(frame: dict, xy: np.ndarray, depth: np.ndarray) -> np.ndarray:
    k = frame["intrinsics"]
    local = np.column_stack(
        [
            (xy[:, 0] + 0.5 - k["cx"]) / k["fx"] * depth,
            -(xy[:, 1] + 0.5 - k["cy"]) / k["fy"] * depth,
            -depth,
        ]
    )
    return local @ frame["rotation"].T + frame["position"]


def reference_point(frame: dict, xy: np.ndarray) -> np.ndarray:
    x, y = np.rint(xy).astype(int)
    h, w = frame["depth"].shape
    if not 0 <= x < w or not 0 <= y < h:
        raise ValueError("atlas_reference_outside_returned_view")
    radius = max(1, round(min(w, h) * 0.005))
    sl = np.s_[
        max(0, y - radius) : min(h, y + radius + 1), max(0, x - radius) : min(w, x + radius + 1)
    ]
    values = frame["depth"][sl]
    keep = np.isfinite(values) & (values > 0) & (frame["confidence"][sl] > 0)
    if not keep.any():
        raise ValueError("atlas_reference_depth_missing")
    return unproject(frame, np.array([[x, y]]), np.array([np.median(values[keep])]))[0]


def frame_points(frame: dict, mask=None) -> np.ndarray:
    depth = frame["depth"]
    keep = np.isfinite(depth) & (depth > 0) & (frame["confidence"] > 0)
    if mask is not None:
        if mask.shape != depth.shape:
            raise ValueError("atlas_measurement_mask_grid_mismatch")
        keep &= mask
    y, x = np.nonzero(keep)
    # Bound CPU/memory per view, not the number of input views.
    step = max(1, math.ceil(len(x) / 100_000))
    return unproject(frame, np.column_stack([x[::step], y[::step]]), depth[y[::step], x[::step]])


def _object_key(label: str) -> str:
    return " ".join(label.lower().split())


def reference_agreement(references: list[dict], *, unique_majority: bool = False
                        ) -> tuple[list[dict], list[dict], dict]:
    """The references that agree on one scale, the ones set aside, and how they were judged.

    Several views of one labelled object count once. An object agrees when its
    plausible range contains the consensus of the largest agreeing set; a
    disagreeing minority is set aside, and without a strict majority of objects
    none is kept: two priors that disagree say the scale is unknown.

    With ``unique_majority`` (a wide prior joining the vote), every largest
    agreeing set must settle on the same majority: a prior that agrees with two
    references that disagree with each other picks neither.
    """
    groups: dict[str, list[dict]] = {}
    for reference in references:
        groups.setdefault(_object_key(reference["label"]), []).append(reference)
    objects = [
        (
            float(np.median([r["meters_per_unit"] for r in rows])),
            min(r["meters_per_unit_range"][0] for r in rows),
            max(r["meters_per_unit_range"][1] for r in rows),
            rows,
        )
        for rows in groups.values()
    ]
    if not objects:
        return [], [], {"basis": "no_reference", "object_count": 0, "agreeing_object_count": 0}

    def agreeing_with(value):
        return [row for row in objects if row[1] <= value <= row[2]]

    def settled(seed):
        return agreeing_with(float(np.median([row[0] for row in seed])))

    seeds = [agreeing_with(row[0]) for row in objects]
    largest = max(seeds, key=len)
    agreeing = settled(largest)
    if unique_majority:
        majorities = {tuple(map(id, found)): found
                      for found in (settled(seed) for seed in seeds if len(seed) == len(largest))
                      if 2 * len(found) > len(objects)}
        if len(majorities) > 1:
            return [], list(references), {"object_count": len(objects), "agreeing_object_count": len(agreeing),
                                          "basis": "no_majority", "competing_majorities": len(majorities)}
        agreeing = next(iter(majorities.values()), agreeing)
    record = {"object_count": len(objects), "agreeing_object_count": len(agreeing)}
    if len(objects) > 1 and 2 * len(agreeing) <= len(objects):
        return [], list(references), {**record, "basis": "no_majority"}
    kept = [r for row in agreeing for r in row[3]]
    basis = (
        "single_object"
        if len(objects) == 1
        else "agreeing_objects"
        if len(agreeing) == len(objects)
        else "majority_after_outliers"
    )
    return kept, [r for r in references if r not in kept], {**record, "basis": basis}


#: Height priors: named, ranged assumptions about how high something stands above
#: the floor the views observed. They vote only when the references alone do not
#: agree across several objects, and they can only add support.
HEIGHT_PRIOR_BASES = ("camera_height_prior_only", "surface_height_prior_only", "height_priors_only",
                      "single_object_with_height_priors", "majority_with_height_priors")
#: The bases of a scale no reference stands behind: a guesstimate on the priors alone.
PRIOR_ONLY_BASES = HEIGHT_PRIOR_BASES[:3]
#: References that already agree across several objects never consult the priors.
REFERENCE_MAJORITY_BASES = ("agreeing_objects", "majority_after_outliers")

#: The capture camera's height above the floor: a prior every handheld walkthrough
#: carries. A phone held to film while walking sits in front of the chest or face,
#: below the eyes. Standing eye height of US adults runs from 1.42 m (5th-percentile
#: woman) to 1.75 m (95th-percentile man); chest to eye height for most adults is
#: the wide band below.
CAMERA_HEIGHT_LABEL = "capture camera height above floor"
CAMERA_HEIGHT_ASSUMED_M = 1.4
CAMERA_HEIGHT_RANGE_M = (1.2, 1.6)
CAMERA_HEIGHT_ASSUMPTION = "a phone held while walking, between chest and eye height"
CAMERA_HEIGHT_SOURCE = (
    "Standing eye height, US adults: 1.42 m (5th-percentile woman) to 1.75 m (95th-percentile man), "
    "medians 1.52 m and 1.64 m; 2012 ANSUR II survey, tabulated in NC State Ergonomics Center, "
    "Anthropometric Summary Data Tables. A phone held to film sits below the eyes.")
CAMERA_HEIGHT_MIN_VIEWS = 3
FLOOR_MIN_POINTS = 500
#: Observed points farther from the cameras than this many median distances are depth outliers.
FLOOR_MAX_REACH = 20.0
#: The floor rules are metric; each pass refits at the scale the last one implied.
FLOOR_FIT_PASSES = 4
FLOOR_FIT_SETTLED = 0.01

#: Standard heights of named task support and destination surfaces, top above the
#: floor in metres: (kind, phrases that name it, assumed, plausible range, source).
#: Matching is by whole words and conservative; a label naming none, or more than
#: one, casts no vote.
SURFACE_HEIGHTS = (
    ("coffee table", ("coffee table", "coffee tables"), 0.43, (0.36, 0.50),
     "Coffee tables 14-18 in (0.36-0.46 m), most 16-18 in: furniture sizing guides."),
    ("nightstand", ("nightstand", "nightstands", "night stand", "night table", "bedside table"), 0.66,
     (0.56, 0.71), "Nightstands 22-28 in (0.56-0.71 m), most often 26 in: furniture sizing guides."),
    ("bar counter", ("bar counter", "bar counters", "bar top", "bar table"), 1.06, (1.00, 1.10),
     "Bar height 40-42 in (1.02-1.07 m), 42 in standard: furniture sizing guides."),
    ("kitchen counter", ("kitchen counter", "kitchen counters", "countertop", "countertops", "counter top",
                         "worktop", "worktops", "work top"), 0.91, (0.85, 0.95),
     "EN 1116:2018 kitchen worktop heights 850, 900 and 950 mm; US kitchen counters 36 in (0.914 m)."),
    ("workbench", ("workbench", "workbenches", "work bench"), 0.91, (0.86, 1.00),
     "Standing workbenches 34-38 in (0.86-0.97 m), industrial benches 36-37 in: ergonomics guidance."),
    ("dining table", ("dining table", "dining tables", "kitchen table", "kitchen tables", "dinner table"), 0.75,
     (0.71, 0.78), "Dining tables 28-30 in (0.71-0.76 m), 30 in typical: furniture sizing guides."),
    ("desk", ("desk", "desks"), 0.74, (0.70, 0.77),
     "Fixed-height office desks 720-760 mm (EN 527-1 products); US desks 29-30 in (0.74-0.76 m)."),
)
#: A label with any of these words names no standard top: floors, shelves, the
#: inside or underside of something, seating, or a desk or counter of no fixed height.
SURFACE_EXCLUDED_WORDS = frozenset({
    "floor", "floors", "ground", "shelf", "shelves", "shelving", "rack", "racks", "drawer", "drawers",
    "cabinet", "cabinets", "cupboard", "under", "below", "beneath", "underneath", "inside", "stool", "chair",
    "seat", "step", "ladder", "standing", "adjustable", "reception", "kids", "child", "children", "toy",
    "miniature", "model"})
SURFACE_ROLES = ("support", "destination")
SURFACE_MIN_POINTS = 200


def observed_floor(frames: list[dict], points: np.ndarray) -> dict:
    """The floor under the capture cameras, fitted on the views' observed depth, in provider units.

    The levelling rules (``atlas_world_levelling``) find it: the dominant level
    direction, then its lowest dense layer, wider than a counter with almost
    nothing below it. Those rules are in metres, so they run on the provider-unit
    points at the scale the camera-height prior implies, refitted until that
    scale settles. Every camera must stand above it. A table or counter seen
    alone, with nothing visible below it, still passes for a floor: priors on it
    are votes, never measurements. Cameras and depths are only read.
    """
    from .atlas_world_levelling import SAMPLE_CAP, _floor, _level_direction

    def unavailable(reason: str, **details) -> dict:
        return {"status": "unavailable", "reason": reason, **details}

    centres = np.array([np.asarray(f["position"], dtype=np.float64) for f in frames]).reshape(-1, 3)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    rng = np.random.default_rng(607)
    sample = points[rng.choice(len(points), SAMPLE_CAP, replace=False)] if len(points) > SAMPLE_CAP else points
    if len(sample) and len(centres):
        # Depth far beyond everything else (a window, a stray pixel) is not floor evidence.
        reach = np.linalg.norm(sample - np.median(centres, axis=0), axis=1)
        sample = sample[reach <= FLOOR_MAX_REACH * np.median(reach)]
    if not len(centres) or len(sample) < FLOOR_MIN_POINTS:
        return unavailable("floor_points_too_few", sampled_point_count=len(sample))
    # Atlas +Y is roughly up: the height over the lowest observed points sets the first pass.
    first = float(np.median(centres[:, 1]) - np.percentile(sample[:, 1], 2))
    if not first > 0:
        return unavailable("camera_not_above_floor")
    scale = CAMERA_HEIGHT_ASSUMED_M / first
    for _ in range(FLOOR_FIT_PASSES):
        metres = sample * scale
        normal = _level_direction(metres, np.random.default_rng(607))
        floor = None if normal is None else _floor(metres, normal)
        if floor is None:
            return unavailable("floor_unobserved")
        heights = (centres * scale @ floor["normal"] - floor["height_m"]) / scale
        if heights.min() <= 0:
            return unavailable("camera_not_above_floor")
        implied = CAMERA_HEIGHT_ASSUMED_M / float(np.median(heights))
        settled = abs(implied / scale - 1.0) <= FLOOR_FIT_SETTLED
        floor_scale, scale = scale, implied
        if settled:
            break
    else:
        return unavailable("floor_fit_unsettled")
    floor_points = round(floor["inlier_share"] * len(sample))
    if floor_points < FLOOR_MIN_POINTS:
        return unavailable("floor_points_too_few", floor_point_count=floor_points)
    return {"status": "observed", "normal": [float(v) for v in floor["normal"]],
            "offset_provider_units": float(floor["height_m"] / floor_scale),
            "point_count": floor_points, "sampled_point_count": len(sample),
            "inlier_share": float(floor["inlier_share"]), "below_share": float(floor["below_share"]),
            "rms_provider_units": float(floor["rms_m"] / floor_scale),
            "extent_provider_units": [float(v / floor_scale) for v in floor["extent_m"]],
            "camera_heights_provider_units": [float(h) for h in heights]}


def _prior_row(*, label: str, basis: str, assumed: float, plausible: tuple, span: float, **fields) -> dict:
    low, high = plausible
    return {"status": "measured", "label": label, "basis": basis, "assumed_distance_m": assumed,
            "plausible_distance_range_m": [low, high], **fields, "span_provider_units": span,
            "meters_per_unit": assumed / span, "meters_per_unit_range": [low / span, high / span]}


def camera_height_reference(frames: list[dict], floor: dict) -> dict:
    """The capture cameras' median height above ``floor`` (``observed_floor``), as a named prior.

    ``unavailable`` names why the prior does not apply: no floor, too few views,
    or heights one chest-to-eye hold cannot explain (stairs, a drone, a phone held
    up for part of the walk).
    """
    if floor["status"] != "observed":
        return {"status": "unavailable", "reason": floor["reason"]}
    heights = np.asarray(floor["camera_heights_provider_units"])
    if len(heights) < CAMERA_HEIGHT_MIN_VIEWS:
        return {"status": "unavailable", "reason": "camera_height_too_few_views", "camera_count": len(heights)}
    span = float(np.median(heights))
    low, high = CAMERA_HEIGHT_RANGE_M
    # A strict majority of the views must fit one chest-to-eye hold around their median.
    band = (span * low / CAMERA_HEIGHT_ASSUMED_M, span * high / CAMERA_HEIGHT_ASSUMED_M)
    supporting = int(np.count_nonzero((heights >= band[0]) & (heights <= band[1])))
    if 2 * supporting <= len(heights):
        return {"status": "unavailable", "reason": "camera_height_spread_implausible",
                "camera_count": len(heights), "supporting_camera_count": supporting}
    return _prior_row(label=CAMERA_HEIGHT_LABEL, basis="camera_height_prior", assumed=CAMERA_HEIGHT_ASSUMED_M,
                      plausible=CAMERA_HEIGHT_RANGE_M, span=span, assumption=CAMERA_HEIGHT_ASSUMPTION,
                      source=CAMERA_HEIGHT_SOURCE, camera_count=len(heights), supporting_camera_count=supporting)


def _words(text: str) -> str:
    return " " + " ".join("".join(c if c.isalnum() else " " for c in text.lower()).split()) + " "


def standard_surface(*texts: str | None) -> tuple | None:
    """The one ``SURFACE_HEIGHTS`` row these labels name, or None (no match, several, or excluded)."""
    words = [_words(text) for text in texts if text]
    if any(word in text.split() for text in words for word in SURFACE_EXCLUDED_WORDS):
        return None
    kinds = {row for row in SURFACE_HEIGHTS for text in words if any(f" {p} " in text for p in row[1])}
    return next(iter(kinds)) if len(kinds) == 1 else None


def surface_height_references(frames: list[dict], surfaces, floor: dict) -> list[dict]:
    """Each task support or destination surface's observed top above ``floor``, as a named prior.

    A surface votes when its label or segmentation prompt names one standard
    height (``SURFACE_HEIGHTS``) and its own track is observed on these views.
    Its top is measured from its observed points the way object bounds are: the
    99th percentile of their heights above the floor. Anything else casts no
    vote and says why; an observation is never assumed.
    """
    from .website_task_masks import decode_track_mask

    by_id = {f["frame_id"]: f for f in frames}
    rows = []
    for target in surfaces:
        if target.get("target_role") not in SURFACE_ROLES:
            continue
        identity = {"target_id": target.get("target_id"), "target_role": target.get("target_role"),
                    "semantic_label": target.get("semantic_label"),
                    "segmentation_prompt": target.get("segmentation_prompt")}
        surface = standard_surface(target.get("semantic_label"), target.get("segmentation_prompt"))
        if surface is None:
            rows.append({**identity, "status": "unmatched"})
            continue
        if floor["status"] != "observed":
            rows.append({**identity, "status": "unavailable", "reason": floor["reason"]})
            continue
        observed = [o for o in ((target.get("track") or {}).get("observations") or [])
                    if o.get("source_frame_id") in by_id]
        try:
            parts = [frame_points(by_id[o["source_frame_id"]], decode_track_mask(o)) for o in observed]
        except ValueError as exc:
            rows.append({**identity, "status": "unavailable", "reason": str(exc)})
            continue
        points = np.concatenate(parts) if parts else np.empty((0, 3))
        if len(points) < SURFACE_MIN_POINTS:
            rows.append({**identity, "status": "unavailable", "reason": "surface_unobserved",
                         "point_count": len(points)})
            continue
        top = float(np.percentile(points @ np.asarray(floor["normal"]) - floor["offset_provider_units"], 99))
        if not top > 0:
            rows.append({**identity, "status": "unavailable", "reason": "surface_not_above_floor"})
            continue
        kind, _, assumed, plausible, source = surface
        rows.append(_prior_row(label=f"task surface height: {kind}", basis="standard_surface_height_prior",
                               assumed=assumed, plausible=plausible, span=top, **identity, kind=kind,
                               assumption=f"a standard {kind} top", source=source, point_count=len(points),
                               observed_frame_ids=sorted({o["source_frame_id"] for o in observed})))
    return rows


#: The clean plate's registry and an Atlas input name one decoded frame within this presentation time.
SURFACE_FRAME_PTS_TOLERANCE_S = 0.002


def _same_aspect(width: float, height: float, other_width: float, other_height: float) -> bool:
    return abs(width * other_height - height * other_width) <= 0.01 * width * other_height


def surfaces_on_posed_views(frames: list[dict], inputs: dict, selection: dict) -> list[dict]:
    """Retained task-surface tracks (``website_task_masks.retained_task_surfaces``) on these posed views.

    A view takes a surface's upright source mask only when the clean plate's
    decoded-frame registry names the same frame, at the same presentation time,
    with the aspect of the view's own checked source; the mask then follows that
    view's source-to-posed pixel alignment. Any other view is not observed:
    nothing is guessed. ``frames`` come from ``load_views`` on the full-view sources.
    """
    if selection.get("status") != "selected":
        return []
    import cv2
    from .website_task_masks import decode_track_mask

    registry = {row.get("source_frame_id"): row for row in selection.get("source_frame_registry") or []}
    times = {row.get("frame_id"): row.get("timestamp_seconds") for row in inputs.get("frames") or []}
    by_id = {frame["frame_id"]: frame for frame in frames}
    centre = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    surfaces = []
    for target in selection.get("targets") or []:
        observations = []
        for observation in target["source_track"].get("observations") or []:
            frame_id = observation.get("source_frame_id")
            frame, row, time = by_id.get(frame_id), registry.get(frame_id), times.get(frame_id)
            alignment = (frame or {}).get("pixel_alignment") or {}
            if frame is None or row is None or time is None or "matrix" not in alignment:
                continue
            try:
                mask = decode_track_mask(observation)
            except ValueError:
                continue
            (mask_height, mask_width), (source_width, source_height) = mask.shape, alignment["source_grid"]
            if (abs(float(row["decoded_pts_seconds"]) - float(time)) > SURFACE_FRAME_PTS_TOLERANCE_S
                    or not _same_aspect(mask_width, mask_height, source_width, source_height)
                    or not _same_aspect(row["width"], row["height"], source_width, source_height)):
                continue
            height, width = frame["depth"].shape
            # Mask pixel centres -> source grid -> posed grid, in OpenCV's centre convention.
            to_source = np.diag([source_width / mask_width, source_height / mask_height, 1.0])
            matrix = np.linalg.inv(centre) @ np.asarray(alignment["matrix"]) @ to_source @ centre
            posed = cv2.warpAffine(mask.astype(np.uint8), matrix[:2], (width, height), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
            edges = np.diff(np.pad(posed.reshape(-1).astype(np.int8), (1, 1)))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            observations.append({"source_frame_id": frame_id, "width": width, "height": height,
                                 "source_mask_digest": canonical_digest(observation),
                                 "resampling": "source_to_posed_nearest",
                                 "runs": [{"start": int(s), "length": int(e - s)} for s, e in zip(starts, ends)]})
        surfaces.append({**{key: target.get(key) for key in (
            "target_id", "target_role", "semantic_label", "segmentation_prompt")},
            "track": {"track_id": target["source_track"].get("track_id"), "observations": observations}})
    return surfaces


def _vote_with_height_priors(references: list[dict], priors: list[dict], alone: dict
                             ) -> tuple[list[dict], list[dict], list[dict], dict]:
    """The references' vote again, with each height prior as one more object.

    The priors can only add support: a scale stands when one of them is in the
    one strict majority. The references they side with set the scale; with no
    reference, the agreeing priors set it, with their own wide ranges.
    """
    kept, _, record = reference_agreement([*references, *priors], unique_majority=True)
    prior_ids = {id(row) for row in priors}
    agreeing_priors = [row for row in kept if id(row) in prior_ids]
    kept = [row for row in kept if id(row) not in prior_ids] if agreeing_priors else []
    kinds = {row["basis"] for row in agreeing_priors}
    if not agreeing_priors:
        basis = "no_majority"
    elif kept:
        basis = "single_object_with_height_priors" if alone["basis"] == "single_object" else \
            "majority_with_height_priors"
    else:
        basis = ("camera_height_prior_only" if kinds == {"camera_height_prior"} else
                 "surface_height_prior_only" if kinds == {"standard_surface_height_prior"} else
                 "height_priors_only")
    cues: dict[str, str] = {}
    for row in [*kept, *agreeing_priors]:
        cues.setdefault(_object_key(row["label"]), row["label"])
    kept_ids = {id(row) for row in kept}
    return kept, [row for row in references if id(row) not in kept_ids], agreeing_priors, {
        **record, "basis": basis, "agreeing_cues": list(cues.values()), "references_alone": alone}


#: A reference endpoint marks its object's edge; SAM's mask edge may sit a few
#: pixels inside it. Farther than this from the mask, the endpoint is on something else.
ENDPOINT_MASK_FRACTION = 0.005
ENDPOINT_MASK_MIN_PX = 3


def _on_named_object(row: dict | None, pixels: np.ndarray, frame: dict) -> None:
    """Both endpoints lie on one SAM 3.1 instance of the object the reference names.

    On 2026-10-23 one endpoint of the only reference sat on paper towels in front
    of the dishwasher opening; its longer span made the whole scene small.
    """
    from .website_task_masks import decode_track_mask

    if row is None:
        raise ValueError("atlas_reference_object_unnamed")
    height, width = frame["depth"].shape
    radius = max(ENDPOINT_MASK_MIN_PX, round(ENDPOINT_MASK_FRACTION * math.hypot(width, height)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    ends = np.rint(pixels).astype(int)
    found = False
    for observation in row["instances"]:
        mask = decode_track_mask(observation)
        if mask.shape != (height, width):
            raise ValueError("atlas_reference_mask_grid_mismatch")
        if not mask.any():
            continue
        found = True
        near = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        if all(0 <= x < width and 0 <= y < height and near[y, x] for x, y in ends):
            return
    raise ValueError("atlas_reference_endpoint_off_object" if found else "atlas_reference_object_unsegmented")


def estimate_sizes(
    *, frames: list[dict], automation: dict, pose_digest: str, output: Path, tracks: list[dict] = (),
    reference_masks: list[dict] | None = None, camera_height: bool = False,
    surfaces: list[dict] | tuple = (), surface_source: dict | None = None,
) -> dict:
    """Metres per unit from the agreeing references, and the sizes they give.

    With ``reference_masks`` (``atlas_reference_masks``), a reference counts only
    when its endpoints lie on the object it names; without, references are taken
    as placed (retained plans that predate naming them).

    Height priors join the vote whenever the references alone do not agree across
    several objects, for frames that are the capture's own original views (never
    edited or generated ones): with ``camera_height``, the camera's height above
    the observed floor, and each task support or destination in ``surfaces``
    (``website_task_masks`` targets tracked on these views) whose label names a
    standard height. Each breaks a tie, confirms or contradicts a single
    reference, or, with no reference, gives a guesstimate with its own wide
    range; disagreeing votes give no scale. ``height_priors`` records what each
    found, and ``surface_source`` where the surfaces came from. Surfaces are an
    opportunistic input, not part of the binding: a retained result is read back
    as it was written, with whatever surfaces it measured.
    """
    binding = {
        "pose_digest": pose_digest,
        "automation": automation,
        "tracks_digest": canonical_digest({"tracks": tracks}),
    }
    if reference_masks is not None:
        binding["reference_masks_digest"] = canonical_digest({"reference_masks": reference_masks})
    if output.is_file():
        value = json.loads(output.read_text())
        if value.get("binding") != binding or value.get("digest") != canonical_digest(
            value, digest_field="digest"
        ):
            raise ValueError("atlas_measurements_binding_changed")
        return value
    by_id = {f["frame_id"]: f for f in frames}
    named = None if reference_masks is None else {row["reference_index"]: row for row in reference_masks}
    accepted, gaps = [], []
    for index, reference in enumerate(automation.get("scale_references", [])):
        try:
            f = by_id[reference["frame_id"]]
            pixels = mapped_pixels(f, reference["endpoints_normalized"])
            if named is not None:
                _on_named_object(named.get(index), pixels, f)
            points = np.array([reference_point(f, xy) for xy in pixels])
            distance = float(np.linalg.norm(points[1] - points[0]))
            if not math.isfinite(distance) or distance <= 1e-6:
                raise ValueError("atlas_reference_span_invalid")
            accepted.append(
                {
                    **reference,
                    "world_endpoints": points.tolist(),
                    "span_provider_units": distance,
                    "meters_per_unit": reference["assumed_distance_m"] / distance,
                    "meters_per_unit_range": [
                        v / distance for v in reference["plausible_distance_range_m"]
                    ],
                    "basis": "model_assumed_reference",
                    "public_reference_readback": next(
                        (
                            r
                            for r in automation.get("reference_sources", [])
                            if r["frame_id"] == reference["frame_id"]
                            and r["label"] == reference["label"]
                        ),
                        None,
                    ),
                    "pixel_alignment": f["pixel_alignment"],
                }
            )
        except (KeyError, ValueError) as exc:
            gaps.append({"reference": reference, "reason": str(exc)})
    clouds = [frame_points(f) for f in frames]
    nonempty = [p for p in clouds if len(p)]
    points = np.concatenate(nonempty) if nonempty else np.empty((0, 3))
    measured = accepted
    accepted, disagreeing, agreement = reference_agreement(measured)
    priors, agreeing_priors = None, []
    if (camera_height or surfaces) and agreement["basis"] not in REFERENCE_MAJORITY_BASES:
        floor = observed_floor(frames, points)
        priors = {"floor": floor,
                  **({"camera": camera_height_reference(frames, floor)} if camera_height else {}),
                  **({"surfaces": surface_height_references(frames, surfaces, floor)} if surfaces else {}),
                  **({"surface_source": dict(surface_source)} if surface_source is not None else {})}
        votes = [row for row in [priors.get("camera"), *priors.get("surfaces", [])]
                 if row is not None and row["status"] == "measured"]
        if votes:
            accepted, disagreeing, agreeing_priors, agreement = _vote_with_height_priors(
                measured, votes, agreement)
            agreeing_ids = {id(row) for row in agreeing_priors}
            for row in votes:
                row["agrees_with_consensus"] = id(row) in agreeing_ids
    gaps.extend(
        {"reference": r, "reason": "atlas_reference_disagrees_with_consensus"} for r in disagreeing
    )
    scale = float(np.median([r["meters_per_unit"] for r in accepted])) if accepted else None
    scale_range = (
        [
            min(r["meters_per_unit_range"][0] for r in accepted),
            max(r["meters_per_unit_range"][1] for r in accepted),
        ]
        if accepted
        else None
    )
    if scale is None and agreeing_priors:
        # No reference stands: the agreeing priors' guesstimate, with their own wide ranges.
        scale = float(np.median([r["meters_per_unit"] for r in agreeing_priors]))
        scale_range = [min(r["meters_per_unit_range"][0] for r in agreeing_priors),
                       max(r["meters_per_unit_range"][1] for r in agreeing_priors)]

    def bounds(points):
        if not len(points):
            return {"status": "unavailable", "reason": "no_finite_observed_depth"}
        lo, hi = _robust_minmax(points, 1.0)
        extent = hi - lo
        return {
            "status": "estimated",
            "point_count": len(points),
            "bounds_provider_units": [lo.tolist(), hi.tolist()],
            "extent_xyz_provider_units": extent.tolist(),
            "extent_xyz_m_estimate": (extent * scale).tolist() if scale else None,
            "extent_xyz_m_range": [(extent * s).tolist() for s in scale_range] if scale else None,
            "extent_xyz_feet_estimate": (extent * scale / 0.3048).tolist() if scale else None,
            "coverage": "central_98_percent_visible_points_not_full_hidden_object_or_room",
        }

    objects = []
    from .website_task_masks import decode_track_mask

    for target in tracks:
        parts = [
            frame_points(by_id[o["source_frame_id"]], decode_track_mask(o))
            for o in target["track"]["observations"]
            if o["source_frame_id"] in by_id
        ]
        objects.append(
            {
                "target_id": target["target_id"],
                "label": target["label"],
                **bounds(np.concatenate(parts) if parts else np.empty((0, 3))),
            }
        )
    result = {
        "schema_version": "atlas_scene_measurements.v1",
        "binding": binding,
        "status": "estimated" if scale else "relative_sizes_only",
        "meters_per_unit_estimate": scale,
        "meters_per_unit_range": scale_range,
        "scale_references": accepted,
        "reference_agreement": agreement,
        "reference_gaps": gaps,
        **({"height_priors": priors} if priors is not None else {}),
        "scene": bounds(points),
        "objects": objects,
        "scale_status": "estimated_from_named_assumptions"
        if scale
        else "unscaled_provider_world_units",
        "uncaptured_regions": automation.get("uncaptured_regions", []),
        "raw_geometry_rescaled": False,
        "metric_measurement_proven": False,
        "physical_evidence": False,
        "simulation_qualified": False,
        "claim_ceiling": "development_only",
    }
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(output, result)
    return result
