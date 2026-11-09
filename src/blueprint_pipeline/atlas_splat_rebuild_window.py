"""Limit a rebuild's retry pose to the views its earlier pose found consistent and moving.

A rebuild (``atlas_splat_rebuild``) re-selects views across the whole clip before
its pose. World Labs refines a pose (``poseRefinementAccepted``) or not, and gives
no reason. ``atlas_splat_input_quality.select_splat_frames`` finds what the
earlier pose's own cameras show (views that disagree, pose jumps, stationary runs
without parallax), but only after a pose: the first rebuild sent the next pose the
same footage again. capture-5b067498 (2026-11-08) is the case that showed this.
Both its poses were not refined.

The windowed retry reads that earlier pose here, before anything is bought. It runs
``select_splat_frames`` on the earlier pose's cameras, timestamps, depth and (when
one was fused) its draft's cross-view pairs, with the reviewed task route. From the
segments that choice keeps it derives:

- the time intervals the next selection may draw candidates from: each run of kept
  segments (primary, kept, capped), from its first kept view to its last;
- the views inside them it may not use: those the consistency check dropped, and
  every source frame of a capped no-parallax run except the views the cap kept;
- a derived scene review (``pose_window`` inside it names the original review's
  digest). Its intervals and exclusions are the window's. A reviewed coverage frame
  outside the window is left out, named in ``uncaptured_regions`` and flagged
  (``task_route_frames_left_out`` for working-area frames). An anchor outside moves
  to the in-window reviewed frame nearest it in time, recorded as
  ``original_task_anchor_frame_id``.

No window is made, and the retry stops by name with nothing bought, when:

- the earlier pose cannot be read or segmented (missing timestamps or depth);
- it has no moving segment (a pan or a tripod shot);
- the kept segments already span every timed view with nothing excluded (one clean
  moving segment, so a retry would send the same footage);
- the window would leave a reviewed coverage role with no frame.

Every decision is recorded once, in the retry's pose root (``WINDOW_FILE``), and
read back on later ticks, never re-derived. Nothing here buys or uploads anything.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest

#: Segment decisions of ``select_splat_frames`` whose views it sends (or holds back to score).
KEPT_DECISIONS = frozenset({"primary", "kept", "capped"})
#: Task coverage roles: ``atlas_video_inputs.select_coverage_views`` needs one of them (and a surrounding scene).
TASK_ROLES = frozenset({"task_object", "task_region"})


def _timed(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def window_from_selection(record: Mapping[str, Any], timestamps: Mapping[str, float]) -> dict[str, Any]:
    """The intervals and excluded views a selection record keeps; ``status`` says why there is no window.

    ``record`` is ``select_splat_frames``'s, on the earlier pose. ``timestamps`` are its views' source times.
    """
    motion = record.get("motion") or {}
    if motion.get("status") != "segmented":
        return {"status": "unavailable", "reason": "earlier_pose_" + str(motion.get("reason") or "not_segmented")}
    segments = motion.get("segments") or []
    if not segments or all(row.get("decision") == "kept_no_moving_segment" for row in segments):
        return {"status": "unavailable", "reason": "earlier_pose_no_moving_segment"}
    dropped = {row["frame_id"] for row in (record.get("consistency") or {}).get("dropped") or []}
    restored = set((record.get("consistency") or {}).get("restored_frame_ids") or [])
    runs: list[list[Mapping[str, Any]]] = [[]]
    for segment in segments:
        if segment.get("decision") in KEPT_DECISIONS:
            runs[-1].append(segment)
        elif runs[-1]:
            runs.append([])
    intervals, capped_spans = [], []
    for run in (run for run in runs if run):
        kept = [key for segment in run for key in segment.get("kept_frame_ids", segment["frame_ids"])]
        times = [timestamps[key] for key in kept if _timed(timestamps.get(key))]
        if times:
            intervals.append([min(times), max(times)])
        for segment in run:
            if segment.get("decision") == "capped":
                span = [timestamps[key] for key in segment["frame_ids"] if _timed(timestamps.get(key))]
                if span:
                    capped_spans.append({"start": min(span), "end": max(span),
                                         "kept_frame_ids": list(segment.get("kept_frame_ids") or [])})
    if not intervals:
        return {"status": "unavailable", "reason": "earlier_pose_kept_views_untimed"}
    excluded = sorted(dropped - restored)
    timed = [timestamps[key] for key in record.get("posed_frame_ids") or [] if _timed(timestamps.get(key))]
    whole = bool(timed) and len(intervals) == 1 and intervals[0] == [min(timed), max(timed)]
    if whole and not excluded and not capped_spans:
        return {"status": "unchanged", "reason": "earlier_pose_kept_every_view"}
    return {"status": "windowed", "intervals_seconds": intervals, "excluded_frame_ids": excluded,
            "capped_no_parallax_spans": capped_spans}


def _inside(time: Any, intervals: Sequence[Sequence[float]]) -> bool:
    return _timed(time) and any(start <= time <= end for start, end in intervals)


def windowed_review(review: Mapping[str, Any], survey_frames: Sequence[Mapping[str, Any]],
                    window: Mapping[str, Any]) -> dict[str, Any]:
    """The scene review the retry's selection uses, derived from the reviewed one; ``status`` says if none."""
    from .atlas_video_inputs import WORKING_AREA_ROLES

    times = {row["frame_id"]: row.get("timestamp_seconds") for row in survey_frames}
    original = [list(row) for row in review.get("static_intervals_seconds") or []]
    intervals = [[max(a, c), min(b, d)] for a, b in original for c, d in window["intervals_seconds"]
                 if max(a, c) < min(b, d)]
    excluded = set(window["excluded_frame_ids"])
    for span in window["capped_no_parallax_spans"]:
        keep = set(span["kept_frame_ids"])
        excluded |= {key for key, time in times.items()
                     if key not in keep and _timed(time) and span["start"] <= time <= span["end"]}

    def usable(key: str) -> bool:
        return key not in excluded and _inside(times.get(key), intervals)

    requirements, left_out, uncaptured, task_left = [], [], [], []
    for row in review.get("coverage_requirements") or []:
        ids = list(row.get("required_frame_ids") or [])
        kept = [key for key in ids if usable(key)]
        lost = [key for key in ids if key not in kept]
        if lost:
            left_out += [key for key in lost if key not in left_out]
            uncaptured.append(f"{row.get('label')}: frames {', '.join(lost)} left out of the rebuilt pose, "
                              "outside the consistent moving footage it was limited to.")
            if row.get("role") in WORKING_AREA_ROLES:
                task_left += lost
        if kept:
            requirements.append({**row, "required_frame_ids": kept})
    roles_before = {row.get("role") for row in review.get("coverage_requirements") or []}
    roles_after = {row.get("role") for row in requirements}
    if not intervals or ("surrounding_scene" in roles_before and "surrounding_scene" not in roles_after) or (
            roles_before & TASK_ROLES and not roles_after & TASK_ROLES):
        return {"status": "unavailable", "reason": "pose_window_omits_reviewed_coverage",
                "left_out_required_frame_ids": left_out}
    anchor = review.get("anchor_frame_id")
    moved = None
    candidates = [key for row in requirements for key in row["required_frame_ids"]]
    if not candidates:
        return {"status": "unavailable", "reason": "pose_window_omits_reviewed_coverage",
                "left_out_required_frame_ids": left_out}
    if not usable(str(anchor)):
        reference = times.get(anchor)
        anchor_time = reference if _timed(reference) else min((times[key] for key in candidates), default=0.0)
        moved, anchor = anchor, min(candidates, key=lambda key: (abs(times[key] - anchor_time), times[key], key))
    flags = ["pose_window_applied"]
    if left_out:
        flags.append("pose_window_required_frames_left_out")
    if task_left:
        flags.append("task_route_frames_left_out")
    if moved is not None:
        flags.append("pose_window_anchor_moved")
    derived = {**{key: value for key, value in review.items() if key != "additional_input_frame_ids"},
               "static_intervals_seconds": intervals,
               "excluded_frame_ids": sorted(set(review.get("excluded_frame_ids") or []) | excluded),
               "coverage_requirements": requirements, "anchor_frame_id": anchor,
               "uncaptured_regions": [*(review.get("uncaptured_regions") or []), *uncaptured]}
    additional = [key for key in review.get("additional_input_frame_ids") or [] if usable(key)]
    if additional:
        derived["additional_input_frame_ids"] = additional
    if moved is not None and "original_task_anchor_frame_id" not in review:
        derived["original_task_anchor_frame_id"] = moved
    derived["pose_window"] = {"derived_by": "atlas_splat_rebuild_window", "original_review_digest":
                              canonical_digest(dict(review)), "intervals_seconds": intervals,
                              "left_out_required_frame_ids": left_out, "flags": flags}
    return {"status": "windowed", "review": derived, "intervals_seconds": intervals,
            "excluded_frame_ids": sorted(excluded), "left_out_required_frame_ids": left_out,
            "anchor_frame_id": anchor, "original_anchor_frame_id": moved, "flags": flags}


def _earlier_selection(previous_root: Path, review: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """``select_splat_frames`` on the earlier attempt's pose: its cameras, times, depth and any draft pairs."""
    from .atlas_depth_splat import load_posed_rgbd
    from .atlas_splat_input_quality import scene_depth, select_splat_frames
    from .atlas_splat_rebuild import earlier_pose
    from .atlas_video_inputs import GEOMETRY_INPUTS_FILE, WORKING_AREA_ROLES

    found = earlier_pose(previous_root / "pose")
    if found is None:
        raise ValueError("earlier_pose_unreadable")
    source_root = Path(found["_source_root"])
    pose, frames = load_posed_rgbd(source_root)
    inputs = source_root / GEOMETRY_INPUTS_FILE
    rows = json.loads(inputs.read_text()).get("frames", []) if inputs.is_file() else []
    timestamps = {row["frame_id"]: row.get("timestamp_seconds") for row in rows}
    draft_path = previous_root / "splat/atlas_depth_splat_manifest.json"
    draft = json.loads(draft_path.read_text()) if draft_path.is_file() and not draft_path.is_symlink() else {}
    try:
        depth = scene_depth(frames)
    except (OSError, ValueError, KeyError):
        depth = None  # Unreadable depth: the selection says it could not segment, and no window is made.
    record = select_splat_frames(frames=frames,
        agreement=draft.get("cross_view_agreement") if draft.get("pose_result_digest") == pose["digest"] else None,
        timestamps=timestamps, scene_depth=depth,
        sharpness={row["frame_id"]: row["sharpness"] for row in rows if _timed(row.get("sharpness"))},
        pose_refinement_accepted=pose["operation"]["response"].get("poseRefinementAccepted"),
        route_frame_ids={key for group in review.get("coverage_requirements") or []
                         if group.get("role") in WORKING_AREA_ROLES for key in group.get("required_frame_ids", [])})
    return {**record, "pose_result_digest": pose["digest"]}, timestamps


def pose_window(*, job_dir: Path, request: Mapping[str, Any], review: Mapping[str, Any],
                survey: Mapping[str, Any], pose_root: Path) -> dict[str, Any]:
    """The retry's window and derived review, recorded once in ``pose_root`` and read back; never raises.

    ``status`` is ``windowed`` (use ``review``) or names why no window was made.
    """
    from .atlas_splat_rebuild import WINDOW_FILE, WINDOW_SCHEMA, _inside as inside_job

    path = Path(pose_root) / WINDOW_FILE
    if path.is_file() and not path.is_symlink():
        record = json.loads(path.read_text())
        if (record.get("schema_version") != WINDOW_SCHEMA or record.get("request_digest") != request["digest"]
                or record.get("digest") != canonical_digest(record, digest_field="digest")):
            raise ValueError("atlas_splat_rebuild_pose_window_changed")
        return record
    flags: list[str] = []
    selection = None
    try:
        previous_root = inside_job(Path(job_dir), str(request.get("previous_root") or "missing"))
        selection, timestamps = _earlier_selection(previous_root, review)
        window = window_from_selection(selection, timestamps)
        if window["status"] == "windowed":
            window = {**window, **windowed_review(review, survey.get("frames") or [], window)}
    except Exception as exc:  # noqa: BLE001 - an unreadable earlier pose makes no window; it never raises
        window = {"status": "unavailable", "reason": "earlier_pose_unreadable",
                  "error": type(exc).__name__ + ":" + str(exc).partition(":")[0][:120]}
    if window["status"] != "windowed":
        flags.append("pose_window_" + window["status"])
    record = {"schema_version": WINDOW_SCHEMA, "request_digest": request["digest"],
              "previous_request_digest": request.get("previous_request_digest"),
              "status": window["status"], "reason": window.get("reason"),
              "earlier_pose_result_digest": (selection or {}).get("pose_result_digest"),
              "earlier_selection": None if selection is None else {
                  "digest": selection["digest"], "flags": selection["flags"],
                  "motion": {key: (selection.get("motion") or {}).get(key) for key in ("status", "reason", "jumps")},
                  "segments": [{key: row.get(key) for key in ("kind", "jump_group", "decision", "baseline_units",
                                                               "route_frame_count")}
                               | {"frame_count": len(row.get("frame_ids") or [])}
                               for row in (selection.get("motion") or {}).get("segments") or []]},
              "original_review_digest": canonical_digest(dict(review)),
              "flags": sorted(set(flags) | set(window.get("flags") or []))}
    if window["status"] == "windowed":
        record.update({key: window[key] for key in ("intervals_seconds", "excluded_frame_ids",
                                                     "left_out_required_frame_ids", "anchor_frame_id",
                                                     "original_anchor_frame_id", "review")},
                      review_digest=canonical_digest(window["review"]))
    elif "error" in window:
        record["error"] = window["error"]
    if "left_out_required_frame_ids" in window and "left_out_required_frame_ids" not in record:
        record["left_out_required_frame_ids"] = window["left_out_required_frame_ids"]
    record["digest"] = canonical_digest(record, digest_field="digest")
    Path(pose_root).mkdir(parents=True, exist_ok=True)
    write_json(path, record)
    return record


def window_for_pose_root(pose_root: Path) -> dict[str, Any] | None:
    """A windowed retry's recorded window in this pose root, else None (every other pose root)."""
    from .atlas_splat_rebuild import WINDOW_FILE, WINDOW_SCHEMA

    path = Path(pose_root) / WINDOW_FILE
    if not path.is_file() or path.is_symlink():
        return None
    record = json.loads(path.read_text())
    if (record.get("schema_version") != WINDOW_SCHEMA
            or record.get("digest") != canonical_digest(record, digest_field="digest")):
        raise ValueError("atlas_splat_rebuild_pose_window_changed")
    return record if record.get("status") == "windowed" else None


def frame_in_window(row: Mapping[str, Any], window: Mapping[str, Any]) -> bool:
    """Whether a source frame (``frame_id``, ``timestamp_seconds``) is one the window lets the pose use."""
    return (row.get("frame_id") not in set(window["review"]["excluded_frame_ids"])
            and _inside(row.get("timestamp_seconds"), window["intervals_seconds"]))
