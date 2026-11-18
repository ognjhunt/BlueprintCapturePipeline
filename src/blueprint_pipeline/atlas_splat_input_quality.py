"""Which posed views a refined splat trains on: consistent ones with parallax, a few held back.

capture-5b067498's refined splat of a library aisle (41 frames) came out blurry,
while a 16-frame kitchen on the same request form came out clear. The library
job's retained records showed why (2026-11-08):

- The first ~3 s (10 of 41 views) had poses and depth that disagree. 39 of the
  draft's 287 cross-view pairs were over the 0.05 limit (worst 10.7), and every
  one involved those views. The only check was the pooled median (0.007), which
  hid them. World Labs had not accepted pose refinement for that pose.
- 22 views stood at one point, panning, with no parallax. The pose then jumped
  3.3 units in 0.16 s into a straight dolly, which agreed within 1 % on its own.

Before the refined-splat purchase, ``select_splat_frames`` reads the draft's own
cross-view agreement (``atlas_depth_splat.cross_view_agreement``) and the posed
cameras, then:

1. Drops, worst first, every view whose median residual against its measured
   neighbours is over the limit (``MAX_FRAME_RESIDUAL``). Each view is judged
   on its own pairs, not the pool. A view with no measured pair stays and is named.
2. Splits the rest in time at pose jumps and around no-parallax runs (the camera
   stayed within ``STATIONARY_DEPTH_FRACTION`` of the scene depth for at least
   ``STATIONARY_MINIMUM_SECONDS``). It keeps the moving segment holding most of
   the reviewed task route (then the most baseline), every other moving segment
   on the same side of any jump, and every view of each no-parallax run but its
   near-duplicates (``NEAR_DUPLICATE_DEGREES``), never one that saw a weak region of
   the coverage map. Measured pairs never
   compare views across a jump (they look at the nearest cameras only), so a
   segment beyond one is left out unless measured pairs tie it to the kept side.
3. Holds back a few sharp, consistent views of the kept segment
   (``HELDOUT_FRAME_COUNT``). They are never sent, so the returned splat can be
   rendered and scored against photographs it did not train on, and looked at
   from the coverage map's weak regions (``heldout_quality``).

Nothing here stops a job. Too few consistent views is flagged and the best
consistent subset goes on, never fewer than the provider's two. Every decision is
recorded. The draft, gap completion and mesh registration still use every posed view.
"""
from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from .atlas_splat_render import REVISION as RENDERER_REVISION
from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "atlas_splat_input_selection.v1"
REVISION = "per-view-consistency-motion-segments-heldout-near-duplicates-v2"
#: Beside the refined-splat roots in the job's output root; written once, before any purchase.
SELECTION_FILE = "refined_splat_input_selection.json"
#: Beside the adopted refined splat's manifest; never part of its digest.
SCORE_FILE = "heldout_render_score.json"
#: The draft's own limit (``atlas_depth_splat.MAX_MEDIAN_CROSS_VIEW_RESIDUAL``), applied per view.
MAX_FRAME_RESIDUAL = 0.05
PROVIDER_MINIMUM_FRAMES = 2  # posedImages2Splats: "Provide 2 to 512 ordered posedFrames".
CONSISTENT_FRAME_FLOOR = 8
#: A step is a jump when it moves this much of the scene depth at this many times the clip's median speed.
JUMP_DEPTH_FRACTION = 0.25
JUMP_SPEED_FACTOR = 10.0
JUMP_ROTATION_DEGREES = 45.0
STATIONARY_DEPTH_FRACTION = 0.02
STATIONARY_MINIMUM_SECONDS = 1.0
#: In a no-parallax run (the camera stood still), a view whose viewing direction is within this many degrees
#: of a view already kept is a near-duplicate, left out: its image shifts by under about 2% of its width at a
#: 60-degree field of view. A planning choice (owner, 2026-11-18: only true near-duplicates are cut), not a
#: measurement. Every other view of the run is kept, as is any view that saw a weak region of the coverage map.
NEAR_DUPLICATE_DEGREES = 1.0
HELDOUT_FRAME_COUNT = 2
#: Views are held back only while at least this many are still sent.
HELDOUT_MINIMUM_SENT_FRAMES = 24
#: The held-out score's render width in pixels, and the gap that flags held-out views below training views.
SCORE_WIDTH = 640
HELDOUT_BELOW_TRAINING_DB = 3.0
#: The look check (planning thresholds, not measurements): SSIM blocks in render pixels, the block SSIM under
#: which a weak region's held-out blocks flag it, and the held-out floors under which the whole look is weak.
LOOK_BLOCK = 32
LOOK_WEAK_BLOCK_SSIM = 0.5
LOOK_HELDOUT_SSIM_FLOOR = 0.6
LOOK_HELDOUT_PSNR_FLOOR_DB = 20.0
#: A Gaussian is needle-like when its longest axis is this many times its middle one; a weak region
#: with more than this share of them is flagged, as is an orbit view whose region footprint is this empty.
NEEDLE_RATIO = 6.0
NEEDLE_SHARE = 0.25
STRETCH_DEGREES = 20.0
OFF_PATH_VIEWS = 4
OFF_PATH_HOLE_SHARE = 0.25
#: Orbit views stop after this long; held-out views are always scored.
LOOK_TIME_BUDGET_SECONDS = 480


def limits() -> dict[str, Any]:
    return {"max_frame_residual": MAX_FRAME_RESIDUAL, "provider_minimum_frames": PROVIDER_MINIMUM_FRAMES,
            "consistent_frame_floor": CONSISTENT_FRAME_FLOOR, "jump_depth_fraction": JUMP_DEPTH_FRACTION,
            "jump_speed_factor": JUMP_SPEED_FACTOR, "jump_rotation_degrees": JUMP_ROTATION_DEGREES,
            "stationary_depth_fraction": STATIONARY_DEPTH_FRACTION,
            "stationary_minimum_seconds": STATIONARY_MINIMUM_SECONDS, "near_duplicate_degrees": NEAR_DUPLICATE_DEGREES,
            "heldout_frame_count": HELDOUT_FRAME_COUNT, "heldout_minimum_sent_frames": HELDOUT_MINIMUM_SENT_FRAMES}


def frame_residuals(pairs: Sequence[Mapping[str, Any]], frame_ids) -> dict[str, tuple[float, int]]:
    """Each view's median relative residual over the measured pairs it shares with other kept views."""
    kept = set(frame_ids)
    values: dict[str, list[float]] = {}
    for pair in pairs:
        residual = pair.get("median_relative_residual")
        if (pair.get("source") in kept and pair.get("target") in kept and isinstance(residual, (int, float))
                and not isinstance(residual, bool) and math.isfinite(residual)):
            values.setdefault(pair["source"], []).append(float(residual))
            values.setdefault(pair["target"], []).append(float(residual))
    return {key: (float(median(rows)), len(rows)) for key, rows in values.items()}


def consistent_frames(frame_ids: Sequence[str], pairs: Sequence[Mapping[str, Any]],
                      limit: float = MAX_FRAME_RESIDUAL) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Drop the worst view while any view's own median residual is over ``limit``; keep input order.

    Removing a view removes its pairs, so a good view that only disagreed with a
    bad neighbour is judged again on the pairs left. A group left agreeing only
    with itself, every member of which had disagreed with its neighbours before
    (capture-5b067498's views 029 and 058 matched each other at 0.007 and nothing
    else), is dropped too. Returns the kept views, the dropped ones in order (with
    the residual that dropped them) and the kept views no measured pair covers.
    """
    initial = frame_residuals(pairs, frame_ids)
    kept, dropped = list(frame_ids), []
    while True:
        residuals = frame_residuals(pairs, kept)
        over = [key for key in kept if key in residuals and residuals[key][0] > limit]
        if not over:
            break
        worst = max(over, key=lambda key: (residuals[key][0], key))
        kept.remove(worst)
        dropped.append({"frame_id": worst, "reason": "depth_disagrees_with_measured_neighbours",
                        "median_relative_residual": round(residuals[worst][0], 6),
                        "pair_count": residuals[worst][1]})
    group = {key: key for key in kept}

    def root(key):
        while group[key] != key:
            group[key] = group[group[key]]
            key = group[key]
        return key
    for pair in pairs:
        residual = pair.get("median_relative_residual")
        if (pair.get("source") in group and pair.get("target") in group and type(residual) in (int, float)
                and residual <= limit):
            group[root(pair["source"])] = root(pair["target"])
    members: dict[str, list[str]] = {}
    for key in kept:
        members.setdefault(root(key), []).append(key)
    for rows in members.values():
        if all(key in initial and initial[key][0] > limit for key in rows):
            for key in rows:
                kept.remove(key)
                dropped.append({"frame_id": key, "reason": "agrees_only_with_views_that_disagree_elsewhere",
                                "median_relative_residual": round(initial[key][0], 6), "pair_count": initial[key][1]})
    return kept, dropped, [key for key in kept if key not in frame_residuals(pairs, kept)]


def _rotation(quaternion_xyzw) -> list[list[float]]:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def camera_rows(frames: Sequence[Mapping[str, Any]], timestamps: Mapping[str, float]) -> list[dict[str, Any]]:
    """Frame id, time, centre, rotation and viewing direction of each posed view with a timestamp."""
    rows = []
    for frame in frames:
        time = timestamps.get(frame["frame_id"])
        if not isinstance(time, (int, float)) or isinstance(time, bool) or not math.isfinite(time):
            continue
        extrinsics = frame["extrinsics"]
        rotation = _rotation(extrinsics["quaternion"])
        rows.append({"frame_id": frame["frame_id"], "time": float(time),
                     "position": [float(value) for value in extrinsics["position"]], "rotation": rotation,
                     # RUB cameras look down -Z.
                     "forward": [-rotation[0][2], -rotation[1][2], -rotation[2][2]]})
    return sorted(rows, key=lambda row: (row["time"], row["frame_id"]))


def _distance(a, b) -> float:
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def _angle(a, b) -> float:
    """Degrees between two rotation matrices."""
    trace = sum(a[k][i] * b[k][i] for i in range(3) for k in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1) / 2))))


def _direction_angle(a, b) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(p * q for p, q in zip(a, b))))))


def motion_segments(rows: Sequence[Mapping[str, Any]], scene_depth: float) -> tuple[list[dict[str, Any]], list[dict]]:
    """Time-ordered segments, each ``moving`` or ``no_parallax``, with the jump group it belongs to.

    A jump is a step that moves ``JUMP_DEPTH_FRACTION`` of the scene depth at
    ``JUMP_SPEED_FACTOR`` times the clip's median speed (or turns
    ``JUMP_ROTATION_DEGREES`` that much faster than its median turn). Within each
    jump-free group, a run of views whose centres stay within
    ``STATIONARY_DEPTH_FRACTION`` of the scene depth of its first view for at
    least ``STATIONARY_MINIMUM_SECONDS`` adds no parallax.
    """
    steps = []
    for a, b in zip(rows, rows[1:]):
        seconds = max(b["time"] - a["time"], 1e-3)
        steps.append((_distance(a["position"], b["position"]), _angle(a["rotation"], b["rotation"]), seconds))
    speed = median([d / s for d, _, s in steps]) if steps else 0.0
    turn = median([r / s for _, r, s in steps]) if steps else 0.0
    jumps, groups, current = [], [], list(rows[:1])
    for index, (moved, turned, seconds) in enumerate(steps):
        if ((moved > JUMP_DEPTH_FRACTION * scene_depth and moved / seconds > JUMP_SPEED_FACTOR * max(speed, 1e-9))
                or (turned > JUMP_ROTATION_DEGREES and turned / seconds > JUMP_SPEED_FACTOR * max(turn, 1e-9))):
            jumps.append({"after_frame_id": rows[index]["frame_id"], "before_frame_id": rows[index + 1]["frame_id"],
                          "translation_units": round(moved, 6), "rotation_degrees": round(turned, 3),
                          "seconds": round(rows[index + 1]["time"] - rows[index]["time"], 6)})
            groups.append(current)
            current = []
        current.append(rows[index + 1])
    if current:
        groups.append(current)
    radius = STATIONARY_DEPTH_FRACTION * scene_depth
    segments = []
    for group_index, group in enumerate(groups):
        labels, start = ["moving"] * len(group), 0
        while start < len(group):
            end = start
            while end + 1 < len(group) and _distance(group[end + 1]["position"], group[start]["position"]) <= radius:
                end += 1
            if end > start and group[end]["time"] - group[start]["time"] >= STATIONARY_MINIMUM_SECONDS:
                labels[start:end + 1] = ["no_parallax"] * (end + 1 - start)
                start = end + 1
            else:
                start += 1
        for index, row in enumerate(group):
            if not segments or segments[-1]["jump_group"] != group_index or segments[-1]["kind"] != labels[index]:
                segments.append({"kind": labels[index], "jump_group": group_index, "rows": []})
            segments[-1]["rows"].append(row)
    for segment in segments:
        rows_ = segment["rows"]
        segment["baseline_units"] = round(sum(_distance(a["position"], b["position"])
                                              for a, b in zip(rows_, rows_[1:])), 6)
    return segments, jumps


def _distinct_views(rows: Sequence[Mapping[str, Any]], seed_time: float, *, protected=frozenset()) -> list[str]:
    """A no-parallax run's views less its near-duplicates (``NEAR_DUPLICATE_DEGREES``), widest first from the seed.

    A ``protected`` view (one that saw a weak region of the coverage map) is never left out.
    """
    forward = np.array([row["forward"] for row in rows], float)
    seed = min(range(len(rows)), key=lambda i: (abs(rows[i]["time"] - seed_time), rows[i]["frame_id"]))
    kept = [seed]
    # Each step keeps the view farthest in direction from every view kept so far, while it is not a
    # near-duplicate of one; the running nearest angle keeps this quadratic in the run's length.
    nearest = np.degrees(np.arccos(np.clip(forward @ forward[seed], -1, 1)))
    nearest[seed] = -1
    while True:
        candidate = max(range(len(rows)), key=lambda i: (nearest[i], -rows[i]["time"]))
        if nearest[candidate] < NEAR_DUPLICATE_DEGREES:
            break
        kept.append(candidate)
        nearest = np.minimum(nearest, np.degrees(np.arccos(np.clip(forward @ forward[candidate], -1, 1))))
        nearest[kept] = -1
    kept += [i for i, row in enumerate(rows) if row["frame_id"] in protected and i not in kept]
    return [rows[i]["frame_id"] for i in kept]


def _heldout(rows, kept_ids, *, route, residuals, sharpness, sent_total) -> tuple[list[str], str | None]:
    """Sharp, measured, consistent views spread inside the kept segment, never its ends or the task route."""
    if sent_total - HELDOUT_FRAME_COUNT < HELDOUT_MINIMUM_SENT_FRAMES:
        return [], "too_few_views_to_hold_back"
    ordered = [row for row in rows if row["frame_id"] in kept_ids]
    candidates = [row for row in ordered[2:-2] if row["frame_id"] not in route and row["frame_id"] in residuals]
    if len(candidates) < HELDOUT_FRAME_COUNT:
        return [], "no_interior_consistent_views"
    start, end = ordered[0]["time"], ordered[-1]["time"]
    chosen: list[str] = []
    for k in range(1, HELDOUT_FRAME_COUNT + 1):
        target = start + k * (end - start) / (HELDOUT_FRAME_COUNT + 1)
        near = sorted((row for row in candidates if row["frame_id"] not in chosen),
                      key=lambda row: (abs(row["time"] - target), row["frame_id"]))[:3]
        chosen.append(max(near, key=lambda row: (sharpness.get(row["frame_id"], 0.0), -abs(row["time"] - target)))
                      ["frame_id"])
    return chosen, None


def select_splat_frames(*, frames: Sequence[Mapping[str, Any]], agreement: Mapping[str, Any] | None,
                        timestamps: Mapping[str, float], scene_depth: float | None,
                        pose_refinement_accepted: bool | None, route_frame_ids=(),
                        sharpness: Mapping[str, float] | None = None, weak_region_frame_ids=()) -> dict[str, Any]:
    """The views to send, the views held back, and why every other posed view was left out.

    ``frames`` are posed views (``frame_id`` and ``extrinsics``), ``agreement``
    the draft's ``cross_view_agreement`` for exactly these views, ``timestamps``
    each view's source time, ``scene_depth`` the median observed depth in pose
    units. Never raises for too few views: it flags and keeps the best ones.
    """
    sharpness = dict(sharpness or {})
    route = set(route_frame_ids)
    weak = frozenset(weak_region_frame_ids)
    ids = [frame["frame_id"] for frame in frames]
    flags, left_out = [], []
    if pose_refinement_accepted is False:
        flags.append("pose_refinement_not_accepted")
    pairs = list((agreement or {}).get("pairs") or [])
    if agreement is None:
        flags.append("splat_cross_view_agreement_unavailable")
    kept, dropped, unmeasured = consistent_frames(ids, pairs)
    restored = []
    if len(kept) < PROVIDER_MINIMUM_FRAMES:
        # Never stop: the least inconsistent views make up the provider's minimum.
        for row in sorted(dropped, key=lambda row: (row["median_relative_residual"], row["frame_id"])):
            if len(kept) + len(restored) >= min(PROVIDER_MINIMUM_FRAMES, len(ids)):
                break
            restored.append(row["frame_id"])
        kept = [key for key in ids if key in set(kept) | set(restored)]
        flags.append("splat_consistent_frames_below_provider_minimum")
    elif len(kept) < min(CONSISTENT_FRAME_FLOOR, len(ids)):
        flags.append("splat_consistent_frames_below_floor")
    if dropped:
        flags.append("splat_inconsistent_frames_dropped")
    left_out += [{"frame_id": row["frame_id"], "reason": row["reason"]} for row in dropped
                 if row["frame_id"] not in restored]
    residuals = frame_residuals(pairs, kept)
    rows = camera_rows([frame for frame in frames if frame["frame_id"] in set(kept)], timestamps)
    motion: dict[str, Any] = {"untimed_frame_ids": [key for key in kept if key not in {row["frame_id"] for row in rows}]}
    primary_ids: set[str] = set()
    if scene_depth is None or not math.isfinite(scene_depth) or scene_depth <= 0 or len(rows) < 3:
        motion["status"] = "not_segmented"
        motion["reason"] = "scene_depth_unavailable" if len(rows) >= 3 else "too_few_timed_views"
    else:
        segments, jumps = motion_segments(rows, scene_depth)
        for segment in segments:
            segment["route_frame_count"] = sum(row["frame_id"] in route for row in segment["rows"])
        moving = [segment for segment in segments if segment["kind"] == "moving" and segment["baseline_units"] > 0]
        motion.update(status="segmented", jumps=jumps)
        if not moving:
            # A capture that never moved keeps every view: there is nothing better to send.
            flags.append("splat_no_parallax_capture")
            for segment in segments:
                segment["decision"] = "kept_no_moving_segment"
            primary_ids = set(kept)
        else:
            primary = max(moving, key=lambda s: (s["route_frame_count"], s["baseline_units"], len(s["rows"])))
            primary_ids = {row["frame_id"] for row in primary["rows"]}
            group_ids = {}
            for segment in segments:
                group_ids.setdefault(segment["jump_group"], set()).update(row["frame_id"] for row in segment["rows"])
            linked = {primary["jump_group"]}
            for group, members in group_ids.items():
                kept_side = group_ids[primary["jump_group"]]
                links = [pair["median_relative_residual"] for pair in pairs
                         if type(pair.get("median_relative_residual")) in (int, float)
                         and ((pair.get("source") in members and pair.get("target") in kept_side)
                              or (pair.get("target") in members and pair.get("source") in kept_side))]
                if group != primary["jump_group"] and links and median(links) <= MAX_FRAME_RESIDUAL:
                    linked.add(group)
            primary_span = (primary["rows"][0]["time"], primary["rows"][-1]["time"])
            for segment in segments:
                members = [row["frame_id"] for row in segment["rows"]]
                if segment is primary:
                    segment["decision"] = "primary"
                elif segment["jump_group"] not in linked:
                    segment["decision"] = "left_out_across_pose_jump"
                    left_out += [{"frame_id": key, "reason": "registration_unconfirmed_across_pose_jump"}
                                 for key in members]
                    flags.append("splat_pose_jump_segment_left_out")
                elif segment["kind"] == "moving":
                    segment["decision"] = "kept"
                else:
                    seed = primary_span[0] if segment["rows"][-1]["time"] <= primary_span[0] else primary_span[1]
                    keep = set(_distinct_views(segment["rows"], seed, protected=weak))
                    if len(keep) < len(members):
                        segment["decision"] = "capped"
                        segment["kept_frame_ids"] = [key for key in members if key in keep]
                        left_out += [{"frame_id": key, "reason": "no_parallax_near_duplicate"}
                                     for key in members if key not in keep]
                        flags.append("splat_no_parallax_frames_capped")
                    else:
                        segment["decision"] = "kept"
        motion["segments"] = [{"kind": s["kind"], "jump_group": s["jump_group"], "decision": s["decision"],
                               "frame_ids": [row["frame_id"] for row in s["rows"]],
                               "baseline_units": s["baseline_units"], "route_frame_count": s["route_frame_count"],
                               **({"kept_frame_ids": s["kept_frame_ids"]} if "kept_frame_ids" in s else {})}
                              for s in segments]
    removed = {row["frame_id"] for row in left_out}
    kept = [key for key in kept if key not in removed]
    heldout, heldout_reason = _heldout(rows, primary_ids & set(kept) if primary_ids else set(kept), route=route,
                                       residuals=residuals, sharpness=sharpness, sent_total=len(kept))
    sent = [key for key in kept if key not in set(heldout)]
    route_left = sorted(route & removed)
    if route_left:
        flags.append("task_route_frames_left_out")
    record = {"schema_version": SCHEMA_VERSION, "revision": REVISION,
              "basis": "draft_cross_view_agreement_and_posed_cameras", "limits": limits(),
              "pose_refinement_accepted": pose_refinement_accepted,
              "scene_depth_units": None if scene_depth is None else round(float(scene_depth), 6),
              "posed_frame_ids": ids, "sent_frame_ids": sent, "heldout_frame_ids": heldout,
              **({"heldout_unavailable_reason": heldout_reason} if heldout_reason else {}),
              "left_out": left_out,
              "consistency": {"dropped": dropped, "restored_frame_ids": restored, "unmeasured_frame_ids": unmeasured,
                              "pair_count": len(pairs)},
              "motion": motion, "route_frame_ids": sorted(route), "route_frames_left_out": route_left,
              **({"weak_region_frame_ids": sorted(weak)} if weak else {}),
              "flags": sorted(set(flags)), "claim_scope": "development_only"}
    record["digest"] = canonical_digest(record, digest_field="digest")
    return record


def scene_depth(frames: Sequence[Mapping[str, Any]]) -> float | None:
    """The median, over views, of each view's median confident depth, in pose units."""
    import numpy as np
    from .atlas_depth_splat import frame_arrays

    values = []
    for frame in frames:  # one view's depth at a time: a retained pose's views stay in their files
        depth, confidence = (np.asarray(value) for value in frame_arrays(frame, "depth", "confidence"))
        valid = np.isfinite(depth) & (depth > 0) & (confidence > 0)
        if valid.any():
            values.append(float(np.median(depth[valid])))
    return float(np.median(values)) if values else None


def ssim_map(left, right):
    """Wang et al. SSIM per pixel (11x11 Gaussian, sigma 1.5, L=1), channel mean, 'valid' window positions.

    The same window and constants as ``heldout_appearance_evaluation_v2.windowed_ssim``, whose
    value is this map's mean.
    """
    import cv2
    import numpy as np
    from .heldout_appearance_evaluation_v2 import _KERNEL, SSIM_K1, SSIM_K2, SSIM_WINDOW

    c1, c2 = SSIM_K1 ** 2, SSIM_K2 ** 2
    cut = (SSIM_WINDOW - 1) // 2
    kernel = _KERNEL.astype(np.float64)

    def blur(value):
        return cv2.sepFilter2D(value, cv2.CV_64F, kernel, kernel, borderType=cv2.BORDER_REFLECT)[
            cut:value.shape[0] - cut, cut:value.shape[1] - cut]
    maps = []
    for channel in range(left.shape[2]):
        a, b = left[..., channel].astype(np.float64), right[..., channel].astype(np.float64)
        mu_a, mu_b = blur(a), blur(b)
        var_a, var_b, cov = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
        maps.append(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)))
    return np.mean(maps, axis=0)


def _photo(frame, width, height):
    import numpy as np
    from PIL import Image
    from .atlas_depth_splat import frame_arrays

    return np.asarray(Image.fromarray(np.asarray(frame_arrays(frame, "rgb")[0], dtype=np.uint8)).resize(
        (width, height), Image.Resampling.BOX), dtype=np.float64) / 255.0


def _scored(splat, frame, width: int = SCORE_WIDTH):
    """The score row of one posed view, with its SSIM map and render kept for the block pass."""
    import numpy as np
    from .atlas_splat_render import render

    rendered = render(splat, frame, width)
    image, height = rendered["rgb"].astype(np.float64), rendered["height"]
    photo = _photo(frame, width, height)
    covered = rendered["alpha"] > 0.5
    result = {"frame_id": frame["frame_id"], "coverage": round(float(covered.mean()), 6),
              "visible_gaussians": rendered["visible_gaussians"]}
    mse = float(np.mean((image - photo) ** 2))
    result["psnr_db"] = round(10 * math.log10(1.0 / max(mse, 1e-10)), 3)
    local = ssim_map(image, photo) if min(image.shape[:2]) >= 11 else None
    result["ssim"] = None if local is None else round(float(local.mean()), 6)
    both = covered[:, 1:] & covered[:, :-1]
    observed = np.abs(np.diff(photo, axis=1))[both].mean() if both.any() else 0.0
    result["edge_energy_ratio"] = (round(float(np.abs(np.diff(image, axis=1))[both].mean() / observed), 6)
                                   if both.any() and observed > 0 else None)
    return result, local, rendered


def render_photo_score(splat, frame, width: int = SCORE_WIDTH) -> dict[str, Any]:
    """PSNR, SSIM, coverage and edge-energy ratio of an alpha-blended render against the view's own photograph."""
    return _scored(splat, frame, width)[0]


def _in_box(points, region, pad):
    import numpy as np

    centre, half = np.asarray(region["centroid"], float), np.asarray(region["extent"], float) / 2 + pad
    return np.all(np.abs(np.asarray(points, float) - centre) <= half, axis=-1)


def _block_points(frame, local, width, height):
    """Each LOOK_BLOCK px block of the SSIM map: its mean SSIM and the world point its median depth shows."""
    import numpy as np
    from .atlas_depth_splat import frame_arrays

    depth, confidence = frame_arrays(frame, "depth", "confidence")
    k, factor, cut = frame["intrinsics"], width / float(frame["intrinsics"]["width"]), 5
    rows = []
    for y0 in range(0, local.shape[0] - LOOK_BLOCK // 2, LOOK_BLOCK):
        for x0 in range(0, local.shape[1] - LOOK_BLOCK // 2, LOOK_BLOCK):
            block = local[y0:y0 + LOOK_BLOCK, x0:x0 + LOOK_BLOCK]
            # The map starts ``cut`` px in; full-resolution pixels of the same block.
            ys = slice(int((y0 + cut) / factor), int((y0 + cut + block.shape[0]) / factor))
            xs = slice(int((x0 + cut) / factor), int((x0 + cut + block.shape[1]) / factor))
            d = depth[ys, xs][(confidence[ys, xs] > 0) & np.isfinite(depth[ys, xs]) & (depth[ys, xs] > 0)]
            if not d.size:
                continue
            z = float(np.median(d))
            u, v = (xs.start + xs.stop) / 2, (ys.start + ys.stop) / 2
            local_point = np.array([(u - k["cx"]) / k["fx"] * z, -(v - k["cy"]) / k["fy"] * z, -z])
            rows.append({"frame_id": frame["frame_id"], "pixel_box": [x0 + cut, y0 + cut, block.shape[1], block.shape[0]],
                         "ssim": float(block.mean()),
                         "world_point": (np.asarray(frame["rotation"]) @ local_point + frame["position"]).tolist()})
    return rows


def _region_gaussians(splat, region, voxel):
    """Needle-like and view-stretched shares of the Gaussians centred in a weak region."""
    import numpy as np
    from .atlas_splat_render import _quaternion_matrices

    inside = np.flatnonzero(_in_box(splat.xyz, region, voxel))
    if not len(inside):
        return {"gaussian_count": 0, "needle_share": None, "stretched_along_filmed_direction_share": None}
    scale = np.exp(np.asarray(splat.scales, np.float64)[inside])
    order = np.argsort(scale, axis=1)
    sorted_scale = np.take_along_axis(scale, order, axis=1)
    needle = sorted_scale[:, 2] > NEEDLE_RATIO * np.maximum(sorted_scale[:, 1], 1e-12)
    axes = _quaternion_matrices(np.asarray(splat.quats, np.float64)[inside])
    longest = np.take_along_axis(axes, order[:, None, 2:3].repeat(3, axis=1), axis=2)[:, :, 0]
    filmed = np.asarray(((region.get("suggested_orbit") or {}).get("filmed_direction")) or [0, 0, 0], float)
    along = np.abs(longest @ filmed) >= math.cos(math.radians(STRETCH_DEGREES)) if np.any(filmed) else needle & False
    return {"gaussian_count": int(len(inside)), "needle_share": round(float(needle.mean()), 4),
            "stretched_along_filmed_direction_share": round(float((needle & along).mean()), 4)}


def _off_path(splat, region, camera, width):
    """Holes where an orbit view of a weak region shows nothing; no photograph exists there to compare."""
    from .atlas_splat_render import camera_of, render

    rendered = render(splat, camera, width)
    centre, half = np.asarray(region["centroid"], float), np.asarray(region["extent"], float) / 2
    corners = centre + half * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    position, rotation, k = camera_of(camera)
    factor = width / float(k["width"])
    local = (corners - position) @ rotation
    z = -local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (local[:, 0] / z * k["fx"] + k["cx"]) * factor
        v = (-local[:, 1] / z * k["fy"] + k["cy"]) * factor
    row = {"camera_id": camera["camera_id"], "orbit_degrees": camera.get("orbit_degrees"),
           "visible_gaussians": rendered["visible_gaussians"]}
    if not (z > 0).all():
        return {**row, "region_pixel_count": 0, "hole_share": None}
    x0, x1 = (int(np.clip(math.floor(u.min()), 0, width)), int(np.clip(math.ceil(u.max()), 0, width)))
    y0, y1 = (int(np.clip(math.floor(v.min()), 0, rendered["height"])),
              int(np.clip(math.ceil(v.max()), 0, rendered["height"])))
    alpha = rendered["alpha"][y0:y1, x0:x1]
    return {**row, "region_pixel_count": int(alpha.size),
            "hole_share": round(float((alpha < 0.5).mean()), 4) if alpha.size else None}


def heldout_quality(*, splat, frames: Sequence[Mapping[str, Any]], selection: Mapping[str, Any],
                    weak_regions: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Render the returned splat at the held-out views, their nearest sent views and the weak regions' orbits.

    The render is an alpha-blended Gaussian render (``atlas_splat_render``) at
    ``SCORE_WIDTH`` px. Held-out views are scored with PSNR and SSIM against
    their photographs, and each ``LOOK_BLOCK`` px block's SSIM is placed in the
    world through the photograph's depth, so a weak region's blocks are named.
    Orbit views of the weak regions (``atlas_coverage_map``) have no photograph:
    there the record counts holes in the region's footprint and needle-like
    Gaussians in the region instead. ``look_quality`` is ``good``, ``weak`` with
    the regions named, or ``weak_regions_not_assessed`` (flagged) when the held-out
    views pass but no coverage map was made: information and a flag, never a stop. Held-out views well
    below the training views (``HELDOUT_BELOW_TRAINING_DB``) are flagged too.
    """
    import time

    started = time.monotonic()
    by_id = {frame["frame_id"]: frame for frame in frames}
    heldout = [key for key in selection.get("heldout_frame_ids") or [] if key in by_id]
    sent = [key for key in selection.get("sent_frame_ids") or [] if key in by_id]
    order = {key: index for index, key in enumerate(selection.get("posed_frame_ids") or [])}
    training = []
    for key in heldout:
        nearest = min((other for other in sent if other not in training),
                      key=lambda other: (abs(order.get(other, 0) - order.get(key, 0)), other), default=None)
        if nearest is not None:
            training.append(nearest)
    regions = list((weak_regions or {}).get("regions") or [])
    voxel = float(((weak_regions or {}).get("grid") or {}).get("voxel_size_units") or 0.0)
    blocks, scored = [], {"heldout": [], "training": []}
    for role, keys in (("heldout", heldout), ("training", training)):
        for key in keys:
            row, local, rendered = _scored(splat, by_id[key])
            scored[role].append(row)
            if role == "heldout" and local is not None:
                blocks += _block_points(by_id[key], local, rendered["width"], rendered["height"])
    medians = {role: (median([row["psnr_db"] for row in rows]) if rows else None) for role, rows in scored.items()}
    ssims = [row["ssim"] for row in scored["heldout"] if row["ssim"] is not None]
    flags = (["heldout_render_below_training"] if None not in medians.values()
             and medians["heldout"] < medians["training"] - HELDOUT_BELOW_TRAINING_DB else [])
    region_rows, off_path, named = [], [], []
    for region in regions:
        inside = [row for row in blocks if _in_box(row["world_point"], region, voxel)]
        region_rows.append({"id": region["id"], "heldout_block_count": len(inside),
                            "heldout_block_ssim": round(float(np.mean([row["ssim"] for row in inside])), 4)
                            if inside else None, **_region_gaussians(splat, region, voxel), "off_path": []})
    for region, row in zip(regions, region_rows):
        for camera in (region.get("suggested_orbit") or {}).get("cameras") or []:
            if len(off_path) >= OFF_PATH_VIEWS or abs(camera.get("orbit_degrees") or 0) != min(region["orbit_deg"]):
                continue
            if time.monotonic() - started > LOOK_TIME_BUDGET_SECONDS:
                flags.append("look_check_time_budget_reached")
                break
            view = _off_path(splat, region, camera, SCORE_WIDTH)
            row["off_path"].append(view)
            off_path.append(view["camera_id"])
    for row in region_rows:
        holes = [view["hole_share"] for view in row["off_path"] if view["hole_share"] is not None]
        row["flags"] = sorted({*(["low_heldout_block_ssim"] if row["heldout_block_ssim"] is not None
                                 and row["heldout_block_ssim"] < LOOK_WEAK_BLOCK_SSIM else []),
                               *(["needle_like_gaussians"] if (row["needle_share"] or 0) > NEEDLE_SHARE else []),
                               *(["holes_off_path"] if holes and max(holes) > OFF_PATH_HOLE_SHARE else [])})
        if row["flags"]:
            named.append(row["id"])
    # Without a map (none written, or one that could not be made) the regions were not looked at.
    assessed = weak_regions is not None and weak_regions.get("status") == "mapped"
    if not assessed:
        flags.append("weak_regions_not_assessed")
    below = bool((ssims and median(ssims) < LOOK_HELDOUT_SSIM_FLOOR)
                 or (medians["heldout"] is not None and medians["heldout"] < LOOK_HELDOUT_PSNR_FLOOR_DB))
    verdict = "weak" if named or below else "good" if assessed else "weak_regions_not_assessed"
    look = {"verdict": verdict, "weak_region_ids": named, "weak_regions_assessed": assessed,
            **({} if assessed else {"weak_regions_not_assessed_reason": (weak_regions or {}).get("reason")
                                    or "coverage_map_unavailable"}),
            "heldout_below_floor": bool(below), "off_path_camera_ids": off_path, "regions": region_rows,
            "worst_heldout_blocks": [{**row, "ssim": round(row["ssim"], 4),
                                      "world_point": [round(x, 6) for x in row["world_point"]],
                                      "region_id": next((region["id"] for region in regions
                                                         if _in_box(row["world_point"], region, voxel)), None)}
                                     for row in sorted(blocks, key=lambda row: row["ssim"])[:8]],
            "limits": {"block_pixels": LOOK_BLOCK, "weak_block_ssim": LOOK_WEAK_BLOCK_SSIM,
                       "heldout_ssim_floor": LOOK_HELDOUT_SSIM_FLOOR, "heldout_psnr_floor_db": LOOK_HELDOUT_PSNR_FLOOR_DB,
                       "needle_ratio": NEEDLE_RATIO, "needle_share": NEEDLE_SHARE, "off_path_hole_share": OFF_PATH_HOLE_SHARE,
                       "off_path_views": OFF_PATH_VIEWS},
            "weak_regions_digest": (weak_regions or {}).get("digest")}
    if verdict == "weak":
        flags.append("look_quality_weak")
    record = {"schema_version": "atlas_splat_heldout_score.v2", "status": "scored",
              "basis": "alpha_blended_gaussian_render", "renderer": RENDERER_REVISION,
              "render_width_pixels": SCORE_WIDTH, "selection_digest": selection.get("digest"), **scored,
              "median_heldout_psnr_db": medians["heldout"], "median_training_psnr_db": medians["training"],
              "median_heldout_ssim": median(ssims) if ssims else None, "look_quality": look,
              "seconds": round(time.monotonic() - started, 3), "flags": sorted(set(flags)),
              "claim_scope": "development_only", "image_quality_measurement": True}
    record["digest"] = canonical_digest(record, digest_field="digest")
    return record
