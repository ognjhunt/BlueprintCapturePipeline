"""Blueprint's own measurement of a website scene's movable furniture, when it can matter, kept as a sidecar.

Beta journey step: L5 team policy evaluation. A team half whose Franka finds no
stance it carries from may be blocked by chairs (capture-447ae4f4). Moving one
in the simulation needs the chairs' 3-D bounds, and a scene's preparation holds
only the targets its task needed: capture-447's task masks hold the tray and the
table, and its Atlas scene review names the chairs as one grouped entry with 2-D
boxes in two frames. Measuring them is Blueprint's own preparation spend, like
any other preparation analysis call: never a step for the site.

- **When it runs** (``measurement_offer``, read by ``team_scene_packet``). Only
  when the carry search found no stance it carries from, the station search
  refused nearer stances on geometry that is not at the mount's height (a seat
  below it, a chair's back above it: ``nearer_refusals.blocker_tops``), and the
  scene's own review names movable furniture (a chair or a stool,
  ``robot_access_furniture.FURNITURE_KINDS``). The half then records the
  measurement as due (``measurement_due``) with its most: the analysis call's
  (``clean_plate_removal_analysis_gemini.movable_obstacle_request``) plus the
  segmentation's for at most ``MAXIMUM_SEGMENTATION_PROMPTS`` concepts over every
  source frame (``meta_sam31``), and the worker runs it once per such binding
  (a held one again, bounded: see Spending).
- **What is bound.** The intent's own registered preparation, which for a
  renewed job is the renewal's (``current_preparation``: the intent's source
  registration names it, its request is the intent's, and its renewal block
  names its parent), its task context, the revision, the review's bytes, the
  source video and geometry, and the most it may cost. The sealed preparation is
  only read.
- **Spending** (``run_measurement``). The binding is recomputed and must be the
  one the half recorded, or nothing is spent. Every local check that can refuse
  runs before anything is bought, so such a refusal costs nothing: the scene's
  geometry is reopened and must still be the preparation's (an Atlas result
  republished after the preparation, as the one-time splat rebuild does, refuses
  ``geometry_changed``), the preparation's registration matrices are read, and
  every source frame SAM would track is decoded, each geometry frame among them
  at its own time (``geometry_frame_mapping_invalid``). The analysis is asked once more,
  only for the movable furniture around the support, through the retained paid
  call (``analyze_movable_obstacles``); SAM masks are made for just those pieces
  (``meta_sam31.run_meta_sam31``). Both reserve through the job's
  preparation-spend path within its already-approved caps, each within its share
  of the recorded most, and keep their admissions in their receipts. A refused
  reservation (caps, window) is the request's typed hold
  (``team_scene_packet_worker``); the site is never asked. A held measurement
  is tried again at most once per release: a transient cause (the network, a
  timeout, a busy or failing service) after a wait, a refusal only under a
  newly deployed release. A paid call whose outcome is uncertain needs
  reconciliation first and is never bought again by a retry, except a SAM
  request that got no answer: it is counted as spent and sent once more, only
  while the segmentation share still covers it. Each piece takes the
  track its own boxes pick; an ambiguous one is left unmeasured, never guessed.
  Its 3-D bounds are lifted on the CPU from the scene's own estimated depth and
  cameras, as the task masks are (``website_task_masks.estimate_target_bounds``),
  into the simulator frame with the preparation's own registration.
- **The sidecar** (``SIDECAR_FILENAME``, under the capture's
  ``pipeline/robot_access/<preparation digest>/``) is sealed and bound to that
  preparation digest, the intent and the measurement's binding, and records
  what each call reserved. The team half reads it only through the same
  registration (``furniture_for_scope``); a renewed preparation is a new digest.

Nothing here changes the intent, the preparation, the success contract or the
probe. Every figure is an estimate; nothing is a physical measurement.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

SIDECAR_SCHEMA = "blueprint.robot_access_movable_furniture_sidecar.v1"
BINDING_KIND = "movable_furniture_measurement"
DIRECTORY = "robot_access"
SIDECAR_FILENAME = "movable_furniture.v1.json"
#: Distinct segmentation concepts the measurement may ask SAM for; one concept finds every instance of it.
MAXIMUM_SEGMENTATION_PROMPTS = 2
#: An analysis track window: a box timestamp and a mask this far apart still name the same moment.
_TRACK_WINDOW_S = 0.5
REVIEW_PATH = ("pipeline", "atlas", "atlas_scene_review.json")
CLEAN_PLATE_MANIFEST = ("pipeline", "clean_plate", "clean_plate_stage_manifest.json")
ATLAS_OUTPUT = ("pipeline", "website_scene_preparation", "atlas")


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise ValueError("robot_access_measurement_" + code)


def _sha(path: Path) -> str:
    from .local_reconstruction_adapters import _sha256_file

    return _sha256_file(path)


def current_preparation(scope: Mapping[str, Any]) -> dict[str, Any]:
    """The intent's own registered preparation (a renewal's for a renewed job), its context and lineage; read only.

    ``scope`` is the team packet request's own intent (``owner_success_confirmation_scope``). The
    intent's source registration (``website_scene_dispatch``) names the preparation and task context
    the intent was admitted from, each by digest; the preparation's request must be the intent's.
    """

    from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
    from .website_preparation_revision import capture_root_for_preparation
    from .website_scene_source_registry_paths import _index_path, binding_root

    intent = scope["intent"]
    root = Path(scope["directory"]).parent
    index = _index_path(binding_root({"intent_root": str(root)}), intent["request"])
    _require(index.is_file(), "registration_missing")
    registration = read(index, digest_field="registration_digest")
    _require(registration.get("schema_version") == "website_scene_source_registration.v1"
             and registration.get("request_digest") == cross_runtime_canonical_digest(intent["request"]),
             "registration_not_this_intent")
    references = registration["references"]
    for name in ("preparation", "task_context"):
        checked_file(references[name]["path"], references[name])
    path = Path(references["preparation"]["path"])
    preparation = read(path, digest_field="digest")
    context = read(references["task_context"]["path"], digest_field="context_digest")
    _require(preparation.get("intake_request") == intent["request"]
             and (preparation.get("binding") or {}).get("task_context_digest") == context.get("context_digest"),
             "preparation_not_this_intent")
    capture_root = capture_root_for_preparation(path)
    _require(capture_root is not None, "capture_root_unknown")
    renewal = preparation.get("website_preparation_renewal") if isinstance(
        preparation.get("website_preparation_renewal"), Mapping) else None
    lineage = {"intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
               "registration_digest": registration["registration_digest"],
               "preparation_digest": preparation["digest"], "preparation_sha256": references["preparation"]["sha256"],
               "task_context_digest": context["context_digest"],
               "renewal": None if renewal is None else {
                   key: renewal.get(key) for key in ("parent_preparation_digest", "parent_intent_id",
                                                     "authority_digest")}}
    return {"preparation": preparation, "preparation_path": str(path), "task_context": context,
            "capture_root": str(capture_root), "lineage": lineage}


def review_furniture(capture_root: str | Path) -> dict[str, Any]:
    """The scene review's own entries that name movable furniture (``FURNITURE_KINDS``), with its bytes' digest."""

    from .robot_access_furniture import furniture_kind

    path = Path(capture_root).joinpath(*REVIEW_PATH)
    if path.is_symlink() or not path.is_file():
        return {"status": "review_unavailable", "items": []}
    try:
        review = json.loads(path.read_text(encoding="utf-8"))
        objects = ((review.get("automation") or {}).get("objects") or []) if isinstance(review, Mapping) else []
    except (OSError, ValueError, AttributeError):
        return {"status": "review_unreadable", "items": []}
    items = []
    for row in objects:
        if not isinstance(row, Mapping):
            continue
        kind = furniture_kind({"semantic_label": row.get("label"), "segmentation_prompt": row.get("segmentation_prompt")})
        if kind is not None:
            items.append({"obstacle_id": str(row.get("target_id") or "")[:128], "label": str(row.get("label") or kind)[:160],
                          "kind": kind, "segmentation_prompt": str(row.get("segmentation_prompt") or "")[:160],
                          "disposition": row.get("disposition"),
                          "regions": len(row.get("regions") or []) if isinstance(row.get("regions"), list) else 0})
    return {"status": "named" if items else "none_named", "items": items, "sha256": _sha(path)}


def _frame_count(video: Path) -> int:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                             "-show_entries", "stream=nb_read_packets", "-of", "json", str(video)],
                            check=True, timeout=600, capture_output=True)
    return int(json.loads(result.stdout)["streams"][0]["nb_read_packets"])


def measurement_sources(capture_root: str | Path, current: Mapping[str, Any]) -> dict[str, Any]:
    """The capture's retained source video, its duration, processing mode and frame count (CPU, no spend).

    The video is the one the clean-plate stage analysed (``input_video_path``), inside the capture.
    """

    from .clean_plate_removal_analysis_gemini import DEFAULT_PROCESSING, PROCESSING_ENV, _video_processing

    root = Path(capture_root)
    manifest = json.loads(root.joinpath(*CLEAN_PLATE_MANIFEST).read_text(encoding="utf-8"))
    video = Path(str(manifest.get("input_video_path") or ""))
    _require(video.is_absolute() and not video.is_symlink() and video.is_file()
             and video.resolve().is_relative_to(root.resolve()), "source_video_unavailable")
    processing, duration = _video_processing(video, os.getenv(PROCESSING_ENV, "").strip() or DEFAULT_PROCESSING)
    return {"video_path": str(video), "source_video_digest": _sha(video), "processing": processing,
            "duration_seconds": duration, "frames": _frame_count(video)}


def _cents(value: float) -> float:
    return math.ceil(round(value * 100, 6)) / 100


def measurement_binding(*, current: Mapping[str, Any], sources: Mapping[str, Any], review: Mapping[str, Any],
                        revision_digest: str) -> dict[str, Any]:
    """What one measurement is: these exact inputs, these calls, and at most this many USD of preparation spend."""

    from .clean_plate_removal_analysis_gemini import movable_obstacle_request
    from .meta_sam31 import PRICE_PER_FRAME_USD

    request, analysis_usd = movable_obstacle_request(
        source_digest=sources["source_video_digest"], duration_seconds=float(sources["duration_seconds"]),
        processing=str(sources["processing"]), task_context=current["task_context"],
        review_items=[{key: row[key] for key in ("obstacle_id", "label", "segmentation_prompt")}
                      for row in review["items"]])
    unit = max(50, int(sources["frames"])) * PRICE_PER_FRAME_USD
    segmentation_usd = _cents(unit * MAXIMUM_SEGMENTATION_PROMPTS)
    items = [{key: row[key] for key in ("obstacle_id", "label", "kind")} for row in review["items"]]
    return {"kind": BINDING_KIND, "configured_scene_revision_digest": revision_digest,
            "lineage": json.loads(json.dumps(current["lineage"])), "review_sha256": review["sha256"], "items": items,
            "source_video_digest": sources["source_video_digest"],
            "source_geometry_digest": (current["preparation"].get("binding") or {}).get("source_geometry_digest"),
            "analysis": {"request_digest": canonical_digest(request), "maximum_cost_usd": analysis_usd},
            "segmentation": {"frames": int(sources["frames"]), "unit_cost_usd": unit,
                             "maximum_prompts": MAXIMUM_SEGMENTATION_PROMPTS, "maximum_cost_usd": segmentation_usd},
            "maximum_cost_usd": _cents(analysis_usd + segmentation_usd)}


def measurement_offer(scope: Mapping[str, Any] | None, *, revision_digest: str,
                      sources: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Whether the furniture can be measured, and the exact binding the measurement would be.

    ``{"status": "available", "binding", "binding_digest", "review"}``, or ``{"status": <why not>}``. Reads
    only: the registration, the preparation, the review and the video's metadata. ``sources`` replaces
    ``measurement_sources`` (tests).
    """

    if scope is None:
        return {"status": "intent_unbound"}
    try:
        current = current_preparation(scope)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "preparation_unresolved", "reason": str(exc)[:160]}
    if current["preparation"].get("coordinate_frame", {}).get("scale_authority") == "model_estimated_object_frame":
        return {"status": "object_local_frame"}
    review = review_furniture(current["capture_root"])
    if review["status"] != "named":
        return {"status": "review_" + review["status"].removeprefix("review_")}
    try:
        found = (sources or measurement_sources)(current["capture_root"], current)
        binding = measurement_binding(current=current, sources=found, review=review, revision_digest=revision_digest)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        return {"status": "sources_unavailable", "reason": str(exc)[:160] or type(exc).__name__}
    from .robot_access_furniture import MAXIMUM_MEASUREMENT_USD

    if not 0 < binding["maximum_cost_usd"] <= MAXIMUM_MEASUREMENT_USD:
        return {"status": "quote_outside_bound", "maximum_cost_usd": binding["maximum_cost_usd"]}
    return {"status": "available", "binding": binding, "binding_digest": canonical_digest(binding),
            "review": review}


def plausibly_blocked(stances: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Whether the stances nearer than the chosen one were refused on something other than the support's top.

    A room mesh counts the blockers' tops against the mount (``nearer_refusals.blocker_tops``): a seat is
    below it, a chair's back above it, the table's own top at it. A box index names clipping boxes.
    """

    refusals = ((stances[0].get("search") or {}).get("nearer_refusals") or {}) if stances else {}
    tops = refusals.get("blocker_tops") or {}
    reasons = refusals.get("by_reason") or {}
    off_mount = int(tops.get("below_mount") or 0) + int(tops.get("above_mount") or 0)
    boxes = int(reasons.get("clips") or 0) + int(reasons.get("clearance") or 0)
    return {"plausible": bool(off_mount or boxes), "off_mount_blocked_stances": off_mount,
            "box_blocked_stances": boxes, "nearer_refused_stances": int(refusals.get("stances") or 0)}


# ----------------------------------------------------------------------------- the paid measurement


def _sidecar_path(current: Mapping[str, Any]) -> Path:
    return Path(current["capture_root"]) / "pipeline" / DIRECTORY / current["lineage"]["preparation_digest"][7:23] / (
        SIDECAR_FILENAME)


def _valid_sidecar(path: Path, current: Mapping[str, Any]) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not (isinstance(value, dict) and value.get("schema_version") == SIDECAR_SCHEMA
            and value.get("sidecar_digest") == canonical_digest(value, digest_field="sidecar_digest")
            and value.get("lineage") == current["lineage"]):
        return None
    return value


def _to_world(preparation: Mapping[str, Any]) -> Callable[[Mapping[str, Any]], tuple[list[float], list[float]]]:
    import numpy as np

    from .website_task_preparation import _runtime_bounds

    matrix = np.asarray(preparation["registration"]["source_to_runtime"], dtype=float)
    to_sim = np.asarray(preparation["coordinate_frame"]["runtime_to_simulator"], dtype=float)
    return lambda bounds: _runtime_bounds(dict(zip(("minimum", "maximum"), _runtime_bounds(bounds, matrix))), to_sim)


def _segment(*, video_path: str, source_video_digest: str, prompts: list[dict[str, Any]], output_root: Path,
             task_context: Mapping[str, Any], maximum_cost_usd: float) -> dict[str, Any]:
    """SAM tracks for just these prompts over every source frame, reserved through the preparation-spend path."""

    from .meta_sam31 import PRICE_PER_FRAME_USD, prepare_continuous_video, run_meta_sam31

    registry, video = prepare_continuous_video(source=Path(video_path), source_digest=source_video_digest,
                                               root=output_root / "video")
    unit = max(50, len(registry)) * PRICE_PER_FRAME_USD
    _require(unit * len(prompts) <= maximum_cost_usd + 1e-9, "segmentation_exceeds_binding")
    # The share bounds every dispatch, including a resend of one that got no answer. The tracks
    # span every frame at full resolution and only ``lift`` reads them (selection, decoding and
    # resampling all take compact runs; the sidecar keeps bounds and track ids, never runs).
    result = run_meta_sam31(frame_registry=registry, frame_artifacts=[], prompts=prompts,
                            output_root=output_root / "sam", admission={}, task_context=task_context,
                            video_artifact=video, maximum_cost_usd=maximum_cost_usd, compact_runs=True)
    return {"registry": registry, "tracks": result["tracks"], "reserved_cost_usd": unit * len(prompts)}


def _frames(*, video_path: str, source_video_digest: str, output_root: Path) -> list[dict[str, Any]]:
    """Every source frame SAM would track, decoded on the CPU (no spend); ``_segment`` reads the same receipt."""

    from .meta_sam31 import prepare_continuous_video

    registry, _ = prepare_continuous_video(source=Path(video_path), source_digest=source_video_digest,
                                           root=output_root / "video")
    return registry


def _frames_mapped(registry: Sequence[Mapping[str, Any]], frames: Sequence[Mapping[str, Any]]) -> None:
    """Every geometry frame is one of the source video's decoded frames, at the same time."""

    by_id = {row["source_frame_id"]: row for row in registry}
    for frame in frames:
        row = by_id.get(frame["frame_id"])
        _require(row is not None and abs(float(row["decoded_pts_seconds"]) - float(frame["timestamp_seconds"])) <= 0.002,
                 "geometry_frame_mapping_invalid")


def _geometry(capture_root: str, current: Mapping[str, Any]) -> dict[str, Any]:
    """The scene's bound source geometry, reopened as the handoff and the renewal reopen it (CPU)."""

    from .website_atlas_scene_source import atlas_scene_source

    root = Path(capture_root)
    atlas = atlas_scene_source(capture_root=root, output_root=root.joinpath(*ATLAS_OUTPUT))
    _require(atlas.get("status") == "ready", "geometry_unavailable")
    geometry = atlas["source_geometry"]
    _require(geometry.get("digest") == (current["preparation"].get("binding") or {}).get("source_geometry_digest"),
             "geometry_changed")
    return geometry


def lift(targets: Sequence[Mapping[str, Any]], *, registry: Sequence[Mapping[str, Any]],
         tracks: Sequence[Mapping[str, Any]], prompts: Sequence[Mapping[str, Any]],
         geometry: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Each piece's own track (picked by its own boxes) lifted to its visible bounds on the CPU; and those left out.

    The tracks are named by the source video's decoded frames; every geometry frame must be one of
    them at the same time. A piece whose boxes pick no single instance, or whose mask has no depth, is
    left out with why, never guessed.
    """

    from .website_task_masks import (
        TaskTargetAmbiguous,
        estimate_target_bounds,
        select_task_track,
        track_at_geometry_resolution,
    )

    frames = list(geometry["frames"])
    _frames_mapped(registry, frames)
    identity = [{"frame_id": row["source_frame_id"], "timestamp_seconds": row["decoded_pts_seconds"]} for row in registry]
    label = {prompt["text"].casefold(): prompt["output_label"] for prompt in prompts}
    ids = {frame["frame_id"] for frame in frames}
    measured, left = [], []
    for target in targets:
        prompt = label.get(str(target.get("segmentation_prompt") or target.get("semantic_label") or "").casefold())
        if prompt is None:
            left.append({"target_id": target["target_id"], "reason": "segmentation_concept_over_the_bound_count"})
            continue
        candidates = [{**track, "label": target["target_id"]} for track in tracks if track.get("label") == prompt]
        try:
            track = select_task_track(target=target, tracks=candidates, frames=identity)
            observed = track_at_geometry_resolution(
                {**track, "observations": [row for row in track["observations"] if row["source_frame_id"] in ids]},
                frames)
            bounds = estimate_target_bounds(observed, frames)
        except TaskTargetAmbiguous:
            left.append({"target_id": target["target_id"], "reason": "instance_ambiguous"})
            continue
        except ValueError as exc:
            left.append({"target_id": target["target_id"], "reason": str(exc)[:120]})
            continue
        measured.append({**target, "estimated_visible_bounds": bounds, "track_id": track.get("track_id")})
    return measured, left


def run_measurement(scope: Mapping[str, Any], *, binding_digest: str, revision_digest: str,
                    sources: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
                    analyze: Callable[..., Mapping[str, Any]] | None = None,
                    segment: Callable[..., Mapping[str, Any]] | None = None,
                    geometry: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
                    frames: Callable[..., Sequence[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    """Measure the movable furniture a team half recorded as due, once; return the sealed sidecar.

    ``binding_digest`` is the half's ``measurement_due`` binding. It is recomputed from the current
    preparation, review and video and must be the same, or nothing is spent. Every local check that can
    refuse (the geometry still the preparation's, its frames among the decoded source frames) runs before
    the first paid call, so such a refusal costs nothing. An existing sidecar for this preparation is
    returned as is. ``sources``, ``analyze``, ``segment``, ``geometry`` and ``frames`` replace the host
    readers and the two paid calls (tests).
    """

    from .clean_plate_removal_analysis_gemini import analyze_movable_obstacles, movable_obstacle_request
    from .robot_access_furniture import _refusal, furniture_kind, movable_furniture_record

    current = current_preparation(scope)
    path = _sidecar_path(current)
    existing = _valid_sidecar(path, current)
    if existing is not None:
        return existing
    review = review_furniture(current["capture_root"])
    _require(review["status"] == "named", "review_changed")
    found = (sources or measurement_sources)(current["capture_root"], current)
    binding = measurement_binding(current=current, sources=found, review=review, revision_digest=revision_digest)
    _require(binding_digest == canonical_digest(binding), "binding_changed")
    request, analysis_usd = movable_obstacle_request(
        source_digest=found["source_video_digest"], duration_seconds=float(found["duration_seconds"]),
        processing=str(found["processing"]), task_context=current["task_context"],
        review_items=[{key: row[key] for key in ("obstacle_id", "label", "segmentation_prompt")}
                      for row in review["items"]])
    _require(canonical_digest(request) == binding["analysis"]["request_digest"]
             and analysis_usd <= binding["analysis"]["maximum_cost_usd"], "analysis_exceeds_binding")
    # Everything local that can refuse, before anything is bought: an Atlas result republished after the
    # preparation (the one-time splat rebuild) refuses here at USD 0, not after the analysis and SAM.
    to_world = _to_world(current["preparation"])
    source_geometry = (geometry or _geometry)(current["capture_root"], current)
    work = path.parent
    work.mkdir(parents=True, exist_ok=True, mode=0o750)
    _frames_mapped((frames or _frames)(video_path=found["video_path"], source_video_digest=found["source_video_digest"],
                                       output_root=work / "segmentation"), source_geometry["frames"])
    analysis = (analyze or analyze_movable_obstacles)(
        video_path=found["video_path"], task_context=current["task_context"], output_root=work / "analysis",
        binding=request, maximum_cost_usd=analysis_usd)
    targets = [row for row in analysis.get("targets") or [] if isinstance(row, Mapping)
               and furniture_kind(row) is not None and _refusal(row) is None and row.get("spatial_evidence")]
    prompts: list[dict[str, Any]] = []
    for row in targets:
        text = str(row.get("segmentation_prompt") or row.get("semantic_label") or "").strip()
        if (text and len(prompts) < MAXIMUM_SEGMENTATION_PROMPTS
                and all(prompt["text"].casefold() != text.casefold() for prompt in prompts)):
            prompts.append({"prompt_id": str(row["target_id"]), "text": text, "output_label": str(row["target_id"]),
                            "anchor_frame_index": 0})
    segmentation: Mapping[str, Any] = {"registry": [], "tracks": [], "reserved_cost_usd": 0.0}
    measured: list[dict[str, Any]] = []
    left: list[dict[str, Any]] = []
    if prompts:
        segmentation = (segment or _segment)(
            video_path=found["video_path"], source_video_digest=found["source_video_digest"], prompts=prompts,
            output_root=work / "segmentation", task_context=current["task_context"],
            maximum_cost_usd=binding["segmentation"]["maximum_cost_usd"])
        measured, left = lift(targets, registry=segmentation["registry"], tracks=segmentation["tracks"],
                              prompts=prompts, geometry=source_geometry)
    record = movable_furniture_record(measured, to_world)
    value = {"schema_version": SIDECAR_SCHEMA, "lineage": dict(current["lineage"]),
             "measurement": {"binding_digest": binding_digest, "maximum_cost_usd": binding["maximum_cost_usd"],
                             "spend": "website_preparation_spend_within_the_jobs_approved_caps"},
             "analysis": {"request_digest": canonical_digest(request), "reserved_cost_usd": analysis_usd,
                          "targets": len(analysis.get("targets") or []), "blockers": list(analysis.get("blockers") or [])},
             "segmentation": {"prompts": [prompt["text"] for prompt in prompts],
                              "reserved_cost_usd": float(segmentation.get("reserved_cost_usd") or 0.0)},
             "movable_furniture": record, "left_unmeasured": left,
             "basis": "visible_masked_surfaces_with_model_estimated_depth_and_cameras",
             "physical_measurement_proven": False, "sidecar_digest": ""}
    value["sidecar_digest"] = canonical_digest(value, digest_field="sidecar_digest")
    from . import task_evaluation_scene_intake as intake

    if not path.exists():
        intake.write_exclusive(path, value)
    written = _valid_sidecar(path, current)
    _require(written is not None, "sidecar_conflict")
    return written


def measurement_due(path: Any) -> dict[str, Any] | None:
    """The furniture measurement the compiled half at ``path`` records as due, with its sealed binding; else None."""

    from .robot_access_furniture import MEASUREMENT_DUE
    from .team_scene_packet import read_team_scene_inputs

    try:
        if not path:
            return None
        read_team_scene_inputs(str(path))
        value = json.loads(Path(str(path)).read_text(encoding="utf-8"))
        access = value.get("robot_access_removal") if isinstance(value, Mapping) else None
        if (not isinstance(access, Mapping) or access.get("status") != MEASUREMENT_DUE
                or canonical_digest(access["binding"]) != access["binding_digest"]):
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return json.loads(json.dumps(dict(access)))


def furniture_for_scope(scope: Mapping[str, Any] | None, *, revision_digest: str) -> dict[str, Any]:
    """What a team half knows of its scene's movable furniture: the sidecar's measurement, or that none was made.

    ``{"status": "measured", "record", "sidecar_digest", "preparation_digest"}`` when the sidecar of the
    intent's current preparation exists; else ``{"status": "not_measured", "reason"}``. The offer to
    measure is read only when a carry fails (``measurement_offer``). ``revision_digest`` is the half's.
    """

    del revision_digest  # the sidecar is the preparation's; the revision is in the measurement's binding
    if scope is None:
        return {"status": "not_measured", "reason": "intent_unbound"}
    try:
        current = current_preparation(scope)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "not_measured", "reason": "preparation_unresolved:" + str(exc)[:120]}
    sidecar = _valid_sidecar(_sidecar_path(current), current)
    if sidecar is None:
        return {"status": "not_measured", "reason": "no_sidecar_for_the_current_preparation"}
    return {"status": "measured", "record": sidecar.get("movable_furniture"), "sidecar_digest": sidecar["sidecar_digest"],
            "preparation_digest": current["lineage"]["preparation_digest"]}


__all__ = [
    "BINDING_KIND",
    "MAXIMUM_SEGMENTATION_PROMPTS",
    "SIDECAR_FILENAME",
    "SIDECAR_SCHEMA",
    "current_preparation",
    "furniture_for_scope",
    "lift",
    "measurement_binding",
    "measurement_due",
    "measurement_offer",
    "measurement_sources",
    "plausibly_blocked",
    "review_furniture",
    "run_measurement",
]
