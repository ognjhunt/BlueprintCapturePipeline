"""Track task objects in source pixels; bind estimated geometry when available."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .meta_sam31 import PROFILE as META_PROFILE, prepare_continuous_video, run_meta_sam31
from .sam31_source_track_provider_stage import run_sam31_source_track_stage
from .scene_placement.sam31_source_track_provider import RUN_REQUEST_SCHEMA_VERSION
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest


def decode_track_mask(observation: Mapping[str, Any]) -> np.ndarray:
    height, width = int(observation["height"]), int(observation["width"])
    if height <= 0 or width <= 0 or height * width > 100_000_000:
        raise ValueError("website_track_mask_dimensions_invalid")
    mask = np.zeros(height * width, dtype=bool)
    for run in observation["runs"]:
        start, length = int(run["start"]), int(run["length"])
        if start < 0 or length <= 0 or start + length > mask.size:
            raise ValueError("website_track_mask_run_invalid")
        mask[start:start + length] = True
    return mask.reshape(height, width)


def track_at_geometry_resolution(track: Mapping[str, Any], frames: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Map upright source masks to the same uncropped grid as estimated depth."""
    by_id = {frame["frame_id"]: frame for frame in frames}
    observations = []
    for observation in track["observations"]:
        frame = by_id[observation["source_frame_id"]]
        mask = decode_track_mask(observation)
        resized = np.asarray(Image.fromarray(mask).resize(
            (frame["width"], frame["height"]), Image.Resampling.NEAREST))
        edges = np.diff(np.pad(resized.reshape(-1).astype(np.int8), (1, 1)))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        observations.append({**observation, "width": frame["width"], "height": frame["height"],
                             "source_mask_digest": canonical_digest(observation),
                             "source_mask_width": observation["width"],
                             "source_mask_height": observation["height"],
                             "resampling": "upright_uncropped_nearest",
                             "runs": [{"start": int(start), "length": int(end - start)}
                                      for start, end in zip(starts, ends)]})
    return {**track, "observations": observations}


def select_task_track(*, target: Mapping[str, Any], tracks: list[Mapping[str, Any]],
                      frames: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    """A text match alone must not remove every instance of the same object class."""
    evidence = [row for row in target.get("spatial_evidence", [])
                if isinstance(row, Mapping) and row.get("box_xywh_normalized") is not None]
    if not evidence:
        raise ValueError("task_target_spatial_anchor_missing")
    by_id = {frame["frame_id"]: frame for frame in frames}
    scores: list[tuple[float, Mapping[str, Any]]] = []
    for track in tracks:
        if track.get("label") != target["target_id"]:
            continue
        overlaps = []
        for observation in evidence:
            # Video-analysis timestamps are coarse (static analysis samples at
            # 2 FPS). Use the nearest observed mask within that bounded window,
            # not a sparse geometry frame or an unobserved exact timestamp.
            exact_frame = (target.get("grounding") or {}).get("source_frame_id")
            nearby = [row for row in track["observations"] if row["source_frame_id"] in by_id
                      and (exact_frame is None or row["source_frame_id"] == exact_frame)
                      and abs(float(by_id[row["source_frame_id"]]["timestamp_seconds"])
                              - observation["timestamp_seconds"]) <= 0.5]
            if not nearby:
                continue
            support = min(nearby, key=lambda row: abs(
                float(by_id[row["source_frame_id"]]["timestamp_seconds"]) - observation["timestamp_seconds"]))
            mask = decode_track_mask(support)
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            x, y, w, h = np.asarray(observation["box_xywh_normalized"], dtype=float)
            if not np.isfinite([x, y, w, h]).all() or min(x, y) < 0 or min(w, h) <= 0 or max(x + w, y + h) > 1.00001:
                raise ValueError("task_target_spatial_anchor_invalid")
            left, top, right, bottom = x * mask.shape[1], y * mask.shape[0], (x + w) * mask.shape[1], (y + h) * mask.shape[0]
            ml, mt, mr, mb = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            intersection = max(0, min(right, mr) - max(left, ml)) * max(0, min(bottom, mb) - max(top, mt))
            union = (right - left) * (bottom - top) + (mr - ml) * (mb - mt) - intersection
            overlaps.append(float(intersection / union) if union else 0.0)
        scores.append((sum(overlaps) / len(overlaps) if overlaps else 0.0, track))
    scores.sort(key=lambda item: item[0], reverse=True)
    if not scores or scores[0][0] < 0.25 or (len(scores) > 1 and scores[0][0] - scores[1][0] < 0.1):
        raise ValueError(f"task_target_track_ambiguous:{target['target_id']}")
    return scores[0][1]


# A concept that cannot find the target is worth one image price to discover
# and one frame price per decoded frame to discover the expensive way. Four
# nouns is a bounded search, not a retry loop: each one must be a new concept
# the model supported with the crop it was shown.
#
# Four because three is provably too few. On the drawer scene the model spends
# one noun that finds nothing, one that finds a part, and only then has been
# told about both failures; the noun that can answer is the next one.
MAXIMUM_CONCEPT_PROBES = 4

# A sub-part of the target always overlaps the target's own box, so overlap
# alone cannot tell "the cabinet" from "one of its drawer fronts". A concept is
# only the target's when its mask also covers most of the box the grounding
# model drew around the whole thing.
MINIMUM_CONCEPT_BOX_COVERAGE = 0.6

CONCEPT_RESOLVED = "resolved"
CONCEPT_NO_INSTANCE = "no_instance"
CONCEPT_MATCHED_PART = "matched_part"


def _grounded_box_coverage(track: Mapping[str, Any], *, box: Sequence[float],
                           width: int, height: int) -> float:
    """How much of the grounded box the selected mask actually spans."""
    mask = decode_track_mask(track["observations"][0])
    rows, columns = np.nonzero(mask)
    if not len(columns):
        return 0.0
    x, y, box_width, box_height = (float(value) for value in box)
    left, top = x * width, y * height
    right, bottom = (x + box_width) * width, (y + box_height) * height
    scale_x, scale_y = width / mask.shape[1], height / mask.shape[0]
    mask_left, mask_top = columns.min() * scale_x, rows.min() * scale_y
    mask_right, mask_bottom = (columns.max() + 1) * scale_x, (rows.max() + 1) * scale_y
    overlap = (max(0.0, min(right, mask_right) - max(left, mask_left))
               * max(0.0, min(bottom, mask_bottom) - max(top, mask_top)))
    area = (right - left) * (bottom - top)
    return overlap / area if area > 0 else 0.0


def probe_segmentation_concept(*, target: Mapping[str, Any], concept: str,
                               task_context: Mapping[str, Any], output_root: Path) -> str:
    """Prove on one frame that a concept resolves this target before buying the clip.

    The grounded frame holds the same pixels the video call sees, so a noun that
    finds nothing here will find nothing there. Two ways to fail are worth
    telling apart: the concept found no instance at all, or it found a part of
    the target and not the target. `drawers` returns three drawer fronts on a
    three-drawer cabinet, each one overlapping the cabinet's own box, and the
    largest would be selected on overlap alone.
    """
    grounding = target["grounding"]
    path = Path(grounding["source_image_path"])
    if _sha256_file(path) != grounding["image_digest"]:
        raise ValueError("website_grounded_source_image_changed")
    with Image.open(path) as image:
        width, height = image.size
    frame_id = grounding["source_frame_id"]
    timestamp = float(target["spatial_evidence"][0]["timestamp_seconds"])
    registry = [{"source_frame_id": frame_id, "model_frame_index": 0, "width": width, "height": height,
                 "decoded_pts_seconds": timestamp, "concept_probe": True}]
    result = run_meta_sam31(frame_registry=registry,
        frame_artifacts=[{"source_frame_id": frame_id, "path": str(path), "sha256": grounding["image_digest"]}],
        prompts=[{"prompt_id": target["target_id"], "output_label": target["target_id"], "text": concept}],
        output_root=output_root, admission={}, task_context=task_context)
    candidates = [{**track, "label": target["target_id"]} for track in result["tracks"]]
    try:
        selected = select_task_track(target=target, tracks=candidates,
                                     frames=[{"frame_id": frame_id, "timestamp_seconds": timestamp}])
    except ValueError as exc:
        if str(exc) != f"task_target_track_ambiguous:{target['target_id']}":
            raise
        return CONCEPT_NO_INSTANCE
    coverage = _grounded_box_coverage(selected, box=target["spatial_evidence"][0]["box_xywh_normalized"],
                                      width=width, height=height)
    return CONCEPT_RESOLVED if coverage >= MINIMUM_CONCEPT_BOX_COVERAGE else CONCEPT_MATCHED_PART


def resolve_video_segmentation_concept(*, target: Mapping[str, Any], tracks: list[Mapping[str, Any]],
                                       registry: list[Mapping[str, Any]], video: Mapping[str, Any],
                                       task_context: Mapping[str, Any], grounding_root: Path,
                                       probe_root: Path, failed_concept: str,
                                       maximum_probes: int = MAXIMUM_CONCEPT_PROBES) -> Mapping[str, Any]:
    """Return a grounded target whose concept the segmenter actually resolves.

    The video-analysis noun already bought a full clip and found nothing. Each
    task-grounded noun or model proposal is spent on one frame before it is
    spent on the whole clip.
    """
    from .website_task_grounding import ground_task_target as _ground

    def tried(value: Mapping[str, Any]) -> bool:
        concept = value["segmentation_prompt"].strip().casefold()
        return any(concept == row.casefold() for row in rejected)

    grounded, rejected, outcome = target, [failed_concept.strip()], CONCEPT_NO_INSTANCE
    # The task itself can supply a more specific, visually testable noun than
    # the grounding model. A cabinet explicitly described as under the desk is
    # probed as one assembly before asking for more free-form synonyms. The
    # one-frame coverage gate below still decides whether it is actually this
    # object; the task phrase alone never authorizes a full-video call.
    task_words = str(target.get("task_basis_quote", "")).casefold()
    task_noun = ("under-desk cabinet" if "cabinet" in task_words
                 and ("under the desk" in task_words or "under-desk" in task_words)
                 else None)
    if not 0 <= maximum_probes <= MAXIMUM_CONCEPT_PROBES:
        raise ValueError("website_concept_probe_budget_invalid")
    remaining_probes = maximum_probes
    if remaining_probes and task_noun and task_noun.casefold() not in (row.casefold() for row in rejected):
        grounded = {**grounded, "segmentation_prompt": task_noun}
        outcome = probe_segmentation_concept(target=grounded, concept=task_noun,
                                             task_context=task_context, output_root=probe_root)
        if outcome == CONCEPT_RESOLVED:
            return grounded
        rejected.append(task_noun)
        remaining_probes -= 1
    # The budget counts probes, not turns. A grounding that hands back a noun
    # already tried costs nothing and must not use one up, or the two nouns
    # already known to fail would spend the whole search before the model is
    # ever told what went wrong with them.
    for _ in range(remaining_probes):
        if tried(grounded):
            grounded = _ground(target=grounded, tracks=tracks, registry=registry, video=video,
                task_context=task_context, output_root=grounding_root,
                failed_segmentation_prompt=rejected[-1], also_rejected=rejected[:-1],
                matched_only_part=outcome == CONCEPT_MATCHED_PART)
            if tried(grounded):
                # Named every noun that failed and shown the crop, the model
                # still has no other supported reading of these pixels.
                break
        concept = grounded["segmentation_prompt"].strip()
        # Rooted by the request digest alone, so the same concept is never
        # bought twice however late in the search it is proposed.
        outcome = probe_segmentation_concept(target=grounded, concept=concept,
                                             task_context=task_context, output_root=probe_root)
        if outcome == CONCEPT_RESOLVED:
            return grounded
        rejected.append(concept)
    raise ValueError(f"task_target_track_ambiguous:{target['target_id']}")


def segment_grounded_static_target(*, target: Mapping[str, Any], registry: list[Mapping[str, Any]],
                                   task_context: Mapping[str, Any], output_root: Path) -> Mapping[str, Any]:
    """One source-image mask for a static contact surface missed by video SAM.

    This never substitutes a single-frame mask for an object that must be
    removed throughout the video. Crop pixels map back to the original grid.
    """
    if target.get("task_effect") != "static_contact" or target.get("disposition") != "keep":
        raise ValueError("website_single_frame_mask_requires_static_contact")
    grounding = target["grounding"]
    path = Path(grounding["source_image_path"])
    if _sha256_file(path) != grounding["image_digest"]:
        raise ValueError("website_grounded_source_image_changed")
    frame = next(row for row in registry if row["source_frame_id"] == grounding["source_frame_id"])
    x, y, width, height = target["spatial_evidence"][0]["box_xywh_normalized"]
    output_root.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as image:
        if image.size != (frame["width"], frame["height"]):
            raise ValueError("website_grounded_source_dimensions_changed")
        crop_box = (max(0, math.floor((x - width * 0.2) * image.width)),
                    max(0, math.floor((y - height * 0.2) * image.height)),
                    min(image.width, math.ceil((x + width * 1.2) * image.width)),
                    min(image.height, math.ceil((y + height * 1.2) * image.height)))
        crop_path = output_root / "source-crop.png"
        image.crop(crop_box).save(crop_path)
    left, top, right, bottom = crop_box
    crop_registry = [{**frame, "model_frame_index": 0, "width": right - left, "height": bottom - top,
                      "source_crop_box_pixels": crop_box, "source_image_digest": grounding["image_digest"]}]
    # Location already identifies the target in this crop. Drop an optional
    # leading color adjective, which can exclude a shaded/cream-colored book.
    concept = re.sub(r"^(?:white|black|blue|red|green|yellow|orange|purple|brown|gray|grey|cream)(?:-colou?red)?\s+",
                     "", target["segmentation_prompt"], count=1, flags=re.IGNORECASE).strip()
    result = run_meta_sam31(frame_registry=crop_registry,
        frame_artifacts=[{"source_frame_id": frame["source_frame_id"], "path": str(crop_path), "sha256": _sha256_file(crop_path)}],
        prompts=[{"prompt_id": target["target_id"], "output_label": target["target_id"], "text": concept}],
        output_root=output_root, admission={}, task_context=task_context)
    tracks = []
    for track in result["tracks"]:
        observations = []
        for row in track["observations"]:
            mask = decode_track_mask(row)
            if row["source_frame_id"] != frame["source_frame_id"] or mask.shape != (bottom - top, right - left):
                raise ValueError("website_grounded_crop_mask_mapping_invalid")
            full = np.zeros((frame["height"], frame["width"]), dtype=bool)
            full[top:bottom, left:right] = mask
            edges = np.diff(np.pad(full.reshape(-1).astype(np.int8), (1, 1)))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            observations.append({**row, "width": frame["width"], "height": frame["height"],
                "source_crop_box_pixels": crop_box, "crop_mask_digest": canonical_digest(row),
                "runs": [{"start": int(start), "length": int(end - start)} for start, end in zip(starts, ends)]})
        tracks.append({**track, "track_id": track["track_id"] + "-static-frame", "observations": observations,
                       "coverage": "single_observed_frame", "provider_binding_digest": result["binding_digest"]})
    return select_task_track(target=target, tracks=tracks, frames=[
        {"frame_id": frame["source_frame_id"], "timestamp_seconds": frame["decoded_pts_seconds"]}])


def masked_source_points(track: Mapping[str, Any], frames: list[Mapping[str, Any]]) -> np.ndarray:
    by_id = {frame["frame_id"]: frame for frame in frames}
    points = []
    for observation in track["observations"]:
        frame = by_id[observation["source_frame_id"]]
        if _sha256_file(Path(frame["geometry_path"])) != frame["geometry_digest"]:
            raise ValueError("task_target_geometry_changed")
        mask = decode_track_mask(observation)
        with np.load(frame["geometry_path"], allow_pickle=False) as geometry:
            depth, valid = geometry["depth_m"], geometry["valid_mask"]
        if mask.shape != depth.shape:
            raise ValueError("task_target_mask_geometry_mismatch")
        ys, xs = np.nonzero(mask & valid)
        if not len(xs):
            continue
        pixels = np.stack([xs, ys, np.ones_like(xs)], axis=1)
        camera_points = (pixels @ np.linalg.inv(frame["intrinsics"]).T) * depth[ys, xs, None]
        pose = np.asarray(frame["world_from_camera"])
        points.append(camera_points @ pose[:3, :3].T + pose[:3, 3])
    if not points:
        raise ValueError("task_target_has_no_estimated_geometry")
    return np.concatenate(points)


def estimate_target_bounds(track: Mapping[str, Any], frames: list[Mapping[str, Any]],
                           *, source_to_target: np.ndarray | None = None) -> dict[str, Any]:
    observed = masked_source_points(track, frames)
    if source_to_target is not None:
        transform = np.asarray(source_to_target, dtype=float)
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])):
            raise ValueError("task_target_coordinate_transform_invalid")
        # Rotate the observed points before enclosing them. Rotating an AABB
        # invents empty corners and can make a shallow object appear tall.
        observed = observed @ transform[:3, :3].T + transform[:3, 3]
    minimum, maximum = np.quantile(observed, [0.01, 0.99], axis=0)
    return {"minimum": minimum.tolist(), "maximum": maximum.tolist(),
            "center": ((minimum + maximum) / 2).tolist(), "unit": "estimated_meters",
            "basis": "visible_masked_surfaces_with_model_estimated_depth_and_cameras",
            "complete_object_dimensions": False, "metric_measurement_proven": False}


def run_website_task_masks(*, plan: Mapping[str, Any], source_geometry: Mapping[str, Any],
                          output_root: Path, meta_admission: Mapping[str, Any] | None = None,
                          meta_admission_grant: Any = None, task_context: Mapping[str, Any] | None = None,
                          source_video: Path | None = None, defer_kept_static: bool = False,
                          view_plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    targets = [target for target in plan.get("targets", [])
               if target.get("task_effect") in {"manipulated", "static_contact", "static_obstacle"}]
    if not targets:
        raise ValueError("website_task_targets_missing")
    provider = os.getenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    if provider == "meta":
        profile = {**META_PROFILE, "profile_digest": canonical_digest(META_PROFILE)}
    elif provider == "local":
        profile_path = Path(os.getenv("BLUEPRINT_WEBSITE_SAM31_PROFILE") or "")
        if not profile_path.is_file():
            raise ValueError("website_sam31_profile_missing")
        profile = json.loads(profile_path.read_text())
    else:
        raise ValueError("website_sam31_provider_invalid")
    geometry_available = source_geometry.get("geometry_available") is not False
    binding = {"geometry_digest": source_geometry["digest"] if geometry_available else None,
               "source_frames_digest": source_geometry["digest"],
               "geometry_input_digest": source_geometry.get("geometry_input_digest"), "task_targets": targets,
               "task_context_sha256": plan["task_context_sha256"], "profile_digest": profile["profile_digest"],
               "mask_input_pixels": ("selected_original_views_v1" if view_plan else
                                     "continuous_source_video_v1" if source_video else "upright_source_v1")
                                    if provider == "meta" else "geometry_v1"}
    if view_plan is not None:
        if (provider != "meta" or source_video is None or not task_context
                or view_plan.get("digest") != canonical_digest(view_plan, digest_field="digest")
                or view_plan.get("source_frame_registry") is None
                or view_plan.get("video", {}).get("source_video_digest") !=
                    source_geometry["binding"]["source_video_digest"]):
            raise ValueError("website_mask_view_plan_invalid")
        binding["view_plan_digest"] = view_plan["digest"]
    request_key = canonical_digest(binding)
    root = output_root.resolve() / request_key[7:23]
    root.mkdir(parents=True, exist_ok=True)
    frames = list(source_geometry["frames"])
    registry, artifacts, prompts = [], [], []
    full_registry = []
    prompt_labels = {}
    grounded_targets = {}
    video_artifact = None
    if view_plan is not None:
        registry, video_artifact = list(view_plan["sparse_registry"]), dict(view_plan["video"])
        full_registry = list(view_plan["source_frame_registry"])
        if any(row["source_frame_id"] not in {full["source_frame_id"] for full in full_registry}
               for row in registry):
            raise ValueError("website_mask_view_registry_invalid")
    elif provider == "meta" and source_video is not None:
        registry, video_artifact = prepare_continuous_video(source=source_video,
            source_digest=source_geometry["binding"]["source_video_digest"], root=root)
        full_registry = registry
    if provider == "meta" and source_video is not None:
        by_id = {row["source_frame_id"]: row for row in registry}
        for frame in frames:
            row = by_id.get(frame["frame_id"])
            if row is not None and abs(row["decoded_pts_seconds"] - frame["timestamp_seconds"]) > 0.002:
                raise ValueError("website_sam31_geometry_frame_mapping_invalid")
            if view_plan is None and row is None:
                raise ValueError("website_sam31_geometry_frame_mapping_invalid")
    for index, frame in enumerate([] if video_artifact else frames):
        source = Path(frame["source_image_path"] if provider == "meta" else frame["image_path"])
        expected_digest = frame["source_image_digest"] if provider == "meta" else frame["image_digest"]
        if _sha256_file(source) != expected_digest:
            raise ValueError("website_sam31_source_frame_changed")
        jpeg = root / f"{index:06d}.jpg"
        with Image.open(source) as image:
            image = image.convert("RGB")
            if provider == "meta":
                rotation = float(frame["display_rotation_degrees"])
                if not np.isfinite(rotation) or rotation % 90:
                    raise ValueError("website_source_rotation_not_supported")
                image = image.rotate(rotation, expand=True)
            width, height = image.size
            image.save(jpeg, quality=95, subsampling=0)
        digest = _sha256_file(jpeg)
        registry.append({"source_frame_id": frame["frame_id"], "model_frame_index": index,
                         "source_frame_digest": frame["source_image_digest"],
                         "retained_video_digest": source_geometry["binding"]["source_video_digest"],
                         "decoded_pts_seconds": frame["timestamp_seconds"],
                         "sync_map_row_digest": canonical_json_digest({"frame_id": frame["frame_id"], "pts": frame["timestamp_seconds"]}),
                         "camera_record_digest": canonical_json_digest(frame), "encoder_retained": True,
                         "width": width, "height": height, "analysis_jpeg_digest": digest})
        artifacts.append({"source_frame_id": frame["frame_id"], "path": str(jpeg), "sha256": digest,
                          "size_bytes": jpeg.stat().st_size, "media_type": "image/jpeg"})
    for target in targets:
        evidence = target.get("spatial_evidence") or []
        if not evidence or evidence[0].get("timestamp_seconds") is None:
            raise ValueError("task_target_spatial_anchor_missing")
        anchor = min(range(len(registry)), key=lambda i: abs(registry[i]["decoded_pts_seconds"] - evidence[0]["timestamp_seconds"]))
        text = target.get("segmentation_prompt") or target["semantic_label"]
        if view_plan is not None and target.get("task_effect") == "manipulated":
            from .website_task_grounding import ground_task_target
            grounded = ground_task_target(target=target, tracks=[], registry=registry,
                video=video_artifact, task_context=task_context, output_root=root / "grounding")
            task_words = str(target.get("task_basis_quote", "")).casefold()
            concept = ("under-desk cabinet" if "cabinet" in task_words
                       and ("under the desk" in task_words or "under-desk" in task_words)
                       else grounded["segmentation_prompt"])
            initial = probe_segmentation_concept(target=grounded, concept=concept,
                task_context=task_context, output_root=root / "concept_probes")
            if initial != CONCEPT_RESOLVED:
                grounded = resolve_video_segmentation_concept(target=grounded, tracks=[], registry=registry,
                    video=video_artifact, task_context=task_context, grounding_root=root / "grounding",
                    probe_root=root / "concept_probes", failed_concept=concept,
                    maximum_probes=MAXIMUM_CONCEPT_PROBES - 1)
            else:
                grounded = {**grounded, "segmentation_prompt": concept}
            grounded_targets[target["target_id"]] = grounded
            text = grounded["segmentation_prompt"]
        shared = next((p for p in prompts if p["text"].strip().casefold() == text.strip().casefold()), None) if provider == "meta" else None
        if shared is None:
            shared = {"prompt_id": target["target_id"], "text": text,
                      "output_label": target["target_id"], "anchor_frame_index": anchor}
            prompts.append(shared)
        prompt_labels[target["target_id"]] = shared["output_label"]
    request = {"schema_version": RUN_REQUEST_SCHEMA_VERSION, "provider_profile": profile,
               "bindings": {"capture_digest": source_geometry["binding"]["source_video_digest"],
                            "retained_video_digest": source_geometry["binding"]["source_video_digest"],
                            "camera_solution_digest": source_geometry["digest"],
                            "frame_registry_digest": canonical_json_digest(registry)},
               "frame_registry": registry, "frame_artifacts": artifacts, "prompts": prompts,
               "allowed_evidence_uses": ["semantic_analysis"]}
    if provider == "meta":
        result = run_meta_sam31(frame_registry=registry, frame_artifacts=artifacts, prompts=prompts,
                               output_root=root, admission=meta_admission or {}, admission_grant=meta_admission_grant,
                               task_context=task_context, video_artifact=video_artifact)
        geometry_ids = {frame["frame_id"] for frame in frames}
        # Full tracks stay in the provider receipt. Sampled source frames drive
        # editing now; the same tracks gain geometry later without a second call.
        tracks = [track_at_geometry_resolution({**track, "observations": [row for row in track["observations"]
                  if row["source_frame_id"] in geometry_ids]}, frames) for track in result["tracks"]]
        identity_tracks = result["tracks"]
        identity_frames = [{"frame_id": row["source_frame_id"], "timestamp_seconds": row["decoded_pts_seconds"]}
                           for row in registry]
    else:
        request_path, result_path, tracks_path = root / "request.json", root / "result.json", root / "tracks.json"
        write_json(request_path, request)
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get("run_request_artifact", {}).get("sha256") != _sha256_file(request_path):
                raise ValueError("website_sam31_retained_request_changed")
        else:
            result = run_sam31_source_track_stage(request_path=request_path, run_result_path=result_path,
                                                 provider_result_path=tracks_path, import_request_path=root / "import.json")
        if result.get("status") != "completed":
            raise ValueError("website_sam31_tracks_unavailable")
        if result.get("provider_result_artifact", {}).get("sha256") != _sha256_file(tracks_path):
            raise ValueError("website_sam31_retained_tracks_changed")
        tracks = json.loads(tracks_path.read_text())["tracks"]
        identity_tracks, identity_frames = tracks, frames
    selected = []
    deferred_target_ids = []
    selected_track_ids = set()
    for target in targets:
        if defer_kept_static and target.get("disposition") == "keep" and target.get("task_effect") != "manipulated":
            deferred_target_ids.append(target["target_id"])
            continue
        selected_prompt = next(p["text"] for p in prompts
                               if p["output_label"] == prompt_labels[target["target_id"]])
        candidates = [{**track, "label": target["target_id"]} for track in identity_tracks
                      if track.get("label") == prompt_labels[target["target_id"]]]
        grounding = None
        if target["target_id"] in grounded_targets:
            target = grounded_targets[target["target_id"]]
            grounding = target["grounding"]
        try:
            identity = select_task_track(target=target, tracks=candidates, frames=identity_frames)
        except ValueError as exc:
            if (str(exc) != f"task_target_track_ambiguous:{target['target_id']}"
                    or provider != "meta" or not video_artifact or not task_context
                    or view_plan is not None):
                raise
            from .website_task_grounding import ground_task_target
            grounded = ground_task_target(target=target, tracks=candidates, registry=registry,
                video=video_artifact, task_context=task_context, output_root=root / "grounding")
            grounding = grounded["grounding"]
            try:
                identity = select_task_track(target=grounded, tracks=candidates, frames=identity_frames)
            except ValueError as grounded_exc:
                if str(grounded_exc) != f"task_target_track_ambiguous:{target['target_id']}":
                    raise
                previous_prompt = next(p["text"] for p in prompts if p["output_label"] == prompt_labels[target["target_id"]])
                ambiguous = f"task_target_track_ambiguous:{target['target_id']}"
                # A surface the task only rests on still has the single-frame
                # rescue; an object that must be removed from every frame does
                # not, and fails here rather than shipping one frame of evidence.
                single_frame_rescue = (target.get("task_effect") == "static_contact"
                                       and target.get("disposition") == "keep")
                try:
                    # Bounded concept search, each candidate proved on the
                    # grounded frame first. Reuse full-video transport and
                    # admission; never buy repeated identical attempts, and never
                    # buy a whole clip for a noun the segmenter has not resolved.
                    grounded = resolve_video_segmentation_concept(target=grounded, tracks=candidates,
                        registry=registry, video=video_artifact, task_context=task_context,
                        grounding_root=root / "grounding", probe_root=root / "concept_probes",
                        failed_concept=previous_prompt)
                except ValueError as concept_exc:
                    if str(concept_exc) != ambiguous or not single_frame_rescue:
                        raise
                    identity = segment_grounded_static_target(target=grounded, registry=registry,
                        task_context=task_context, output_root=root / "grounded_static_masks" / target["target_id"])
                    selected_prompt = grounded["segmentation_prompt"]
                else:
                    grounding = grounded["grounding"]
                    refined = run_meta_sam31(frame_registry=registry, frame_artifacts=[],
                        prompts=[{"prompt_id": target["target_id"], "output_label": target["target_id"],
                                  "text": grounded["segmentation_prompt"]}], output_root=root / "grounded_masks",
                        admission={}, task_context=task_context, video_artifact=video_artifact)
                    try:
                        identity = select_task_track(target=grounded, tracks=refined["tracks"], frames=identity_frames)
                    except ValueError as refined_exc:
                        if str(refined_exc) != ambiguous or not single_frame_rescue:
                            raise
                        identity = segment_grounded_static_target(target=grounded, registry=registry,
                            task_context=task_context, output_root=root / "grounded_static_masks" / target["target_id"])
                    selected_prompt = grounded["segmentation_prompt"]
                # The concept can change while target id stays fixed; replace the
                # sampled candidate below from the exact selected full track.
            tracks = [row for row in tracks if row["track_id"] != identity["track_id"]]
            tracks.append(track_at_geometry_resolution({**identity, "observations": [row for row in identity["observations"]
                if row["source_frame_id"] in geometry_ids]}, frames))
        track = {**next(row for row in tracks if row["track_id"] == identity["track_id"]),
                 "label": target["target_id"]}
        if track["track_id"] in selected_track_ids:
            raise ValueError("website_task_targets_resolve_to_same_instance")
        selected_track_ids.add(track["track_id"])
        selected.append({"target_id": target["target_id"], "target_role": target.get("target_role"),
                         "semantic_label": target["semantic_label"], "task_effect": target["task_effect"],
                         "segmentation_prompt": selected_prompt,
                         "placement_relation": target.get("placement_relation"),
                         "articulated_part": target.get("articulated_part") or "",
                         "articulation_kind": target.get("articulation_kind") or "",
                         "disposition": target["disposition"], "track": track,
                         "estimated_visible_bounds": estimate_target_bounds(track, frames) if geometry_available else None})
        if provider == "meta":
            selected[-1]["source_track"] = identity
            if grounding is not None:
                selected[-1]["grounding"] = grounding
    manifest = {"schema_version": "website_task_masks.v1", "status": "completed", "binding": binding,
                "claim_ceiling": "development_only", "targets": selected,
                "source_geometry_digest": source_geometry["digest"] if geometry_available else None}
    if deferred_target_ids:
        manifest.update(status="object_removal_ready", deferred_target_ids=deferred_target_ids)
    if video_artifact:
        manifest["source_frame_registry"] = full_registry
        manifest["source_video_digest"] = source_geometry["binding"]["source_video_digest"]
    manifest["digest"] = canonical_digest(manifest, digest_field="digest")
    write_json(root / ("task_masks.object_removal.json" if deferred_target_ids else "task_masks.json"), manifest)
    return manifest


def complete_retained_static_task_masks(*, plan: Mapping[str, Any],
                                        source_geometry: Mapping[str, Any],
                                        task_masks: Mapping[str, Any],
                                        output_root: Path, view_plan_root: Path) -> dict[str, Any]:
    """Select kept objects from the first, paid SAM response without a provider call.

    The view-first clip already includes the kept-object prompts. Its tracks were
    deferred for object removal, not absent from the provider response. A missing
    or ambiguous retained observation is a preparation refusal, never a reason
    to purchase another clip implicitly.
    """
    deferred = task_masks.get("deferred_target_ids")
    targets = [target for target in plan.get("targets", [])
               if target.get("task_effect") in {"manipulated", "static_contact", "static_obstacle"}]
    if (task_masks.get("status") != "object_removal_ready" or not isinstance(deferred, list) or not deferred
            or len(set(deferred)) != len(deferred)
            or task_masks.get("digest") != canonical_digest(task_masks, digest_field="digest")
            or source_geometry.get("digest") != canonical_digest(source_geometry, digest_field="digest")
            or task_masks.get("source_video_digest") != source_geometry.get("binding", {}).get("source_video_digest")):
        raise ValueError("website_retained_static_masks_binding_invalid")
    binding = task_masks["binding"]
    if (binding.get("source_frames_digest") != source_geometry["digest"]
            or binding.get("geometry_input_digest") != source_geometry.get("geometry_input_digest")
            or binding.get("task_context_sha256") != plan.get("task_context_sha256")
            or binding.get("task_targets") != targets
            or binding.get("mask_input_pixels") not in {
                "selected_original_views_v1", "continuous_source_video_v1"}
            or {row["target_id"] for row in targets if row.get("disposition") == "keep"
                and row.get("task_effect") != "manipulated"} != set(deferred)
            or {row["target_id"] for row in task_masks.get("targets", [])} !=
                {row["target_id"] for row in targets} - set(deferred)):
        raise ValueError("website_retained_static_masks_binding_invalid")
    request_key = canonical_digest(binding)
    root = output_root.resolve() / request_key[7:23]
    first_manifest = root / "task_masks.object_removal.json"
    if not first_manifest.is_file() or json.loads(first_manifest.read_text()) != task_masks:
        raise ValueError("website_retained_static_masks_manifest_changed")
    if binding["mask_input_pixels"] == "selected_original_views_v1":
        plans = list(view_plan_root.glob("*/view_plan.json"))
        matching = [row for path in plans if (row := json.loads(path.read_text())).get("digest") ==
                    binding.get("view_plan_digest")]
        if len(matching) != 1:
            raise ValueError("website_retained_static_view_plan_missing")
        view_plan = matching[0]
        video = view_plan.get("video") or {}
        registry = view_plan.get("sparse_registry")
        video_root = view_plan_root
        if (view_plan.get("digest") != canonical_digest(view_plan, digest_field="digest")
                or view_plan.get("binding", {}).get("source_geometry_digest") != source_geometry["digest"]
                or view_plan.get("binding", {}).get("task_context_sha256") != plan.get("task_context_sha256")
                or view_plan.get("binding", {}).get("source_video_digest") != task_masks["source_video_digest"]
                or task_masks.get("source_frame_registry") != view_plan.get("source_frame_registry")):
            raise ValueError("website_retained_static_view_plan_changed")
        input_digest = view_plan["digest"]
    else:
        receipt_path = root / "continuous-video.json"
        if not receipt_path.is_file() or binding.get("view_plan_digest") is not None:
            raise ValueError("website_retained_static_video_receipt_missing")
        receipt = json.loads(receipt_path.read_text())
        video = receipt.get("video") or {}
        registry = receipt.get("registry")
        video_root = root
        if (receipt.get("digest") != canonical_digest(receipt, digest_field="digest")
                or registry != task_masks.get("source_frame_registry")):
            raise ValueError("website_retained_static_video_receipt_changed")
        input_digest = receipt["digest"]
    if (not isinstance(registry, list) or not registry
            or not isinstance(video.get("path"), str)
            or not Path(video["path"]).resolve().is_relative_to(video_root.resolve())
            or video.get("source_video_digest") != task_masks["source_video_digest"]
            or _sha256_file(Path(video["path"])) != video.get("sha256")):
        raise ValueError("website_retained_static_video_changed")
    candidates = list(root.glob("*/tracks.json"))
    if len(candidates) != 1:
        raise ValueError("website_retained_static_provider_result_missing")
    result_path = candidates[0]
    paid_root = result_path.parent
    request_binding = json.loads((paid_root / "binding.json").read_text())
    result = json.loads(result_path.read_text())
    provider_digest = canonical_digest(request_binding)
    if (result.get("schema_version") != "website_meta_sam31_tracks.v1"
            or result.get("status") != "completed" or result.get("binding_digest") != provider_digest
            or paid_root.name != provider_digest[7:]
            or request_binding.get("profile") != META_PROFILE or result.get("profile") != META_PROFILE
            or binding.get("profile_digest") != canonical_digest(META_PROFILE)
            or request_binding.get("frame_registry") != registry
            or request_binding.get("frame_artifacts") != []
            or request_binding.get("continuous_video") !=
                {key: video[key] for key in ("sha256", "source_video_digest", "encoding")}
            or request_binding.get("video_transport") != "meta_files_v1"):
        raise ValueError("website_retained_static_provider_binding_changed")
    prompts = request_binding.get("prompts") or []
    receipts = result.get("responses") or []
    if len(prompts) != len(receipts) or not prompts:
        raise ValueError("website_retained_static_provider_receipts_invalid")
    verified_tracks = []
    for index, receipt in enumerate(receipts):
        response_path = paid_root / f"response-{index}.json"
        parsed_path = paid_root / f"parsed-response-{index}.json"
        if (Path(receipt.get("path", "")).resolve() != response_path.resolve()
                or receipt.get("sha256") != _sha256_file(response_path)):
            raise ValueError("website_retained_static_provider_receipts_invalid")
        response = json.loads(response_path.read_text())
        parsed = json.loads(parsed_path.read_text())
        if (response.get("binding_digest") != provider_digest or response.get("clip_digest") != video["sha256"]
                or response.get("response", {}).get("status") != "completed"
                or parsed.get("digest") != canonical_digest(parsed, digest_field="digest")
                or parsed.get("binding") != {"response_digest": receipt["sha256"],
                    "request_digest": provider_digest, "parser_revision": 1, "profile": META_PROFILE}
                or any(row.get("label") != prompts[index].get("output_label")
                       for row in parsed.get("tracks", []))):
            raise ValueError("website_retained_static_provider_receipts_invalid")
        verified_tracks.extend(parsed.get("tracks") or [])
    if verified_tracks != result.get("tracks"):
        raise ValueError("website_retained_static_provider_tracks_changed")
    selected = list(task_masks["targets"])
    selected_ids = {row["source_track"]["track_id"] for row in selected}
    frames = [{"frame_id": row["source_frame_id"], "timestamp_seconds": row["decoded_pts_seconds"]}
              for row in registry]
    geometry_ids = {frame["frame_id"] for frame in source_geometry["frames"]}
    for target in targets:
        if target["target_id"] not in deferred:
            continue
        text = (target.get("segmentation_prompt") or target["semantic_label"]).strip().casefold()
        matching_prompts = [row for row in prompts if row.get("text", "").strip().casefold() == text]
        if len(matching_prompts) != 1:
            raise ValueError(f"website_retained_static_prompt_missing:{target['target_id']}")
        label = matching_prompts[0]["output_label"]
        identity = select_task_track(target=target,
            tracks=[{**track, "label": target["target_id"]} for track in verified_tracks
                    if track.get("label") == label], frames=frames)
        if identity["track_id"] in selected_ids:
            raise ValueError("website_task_targets_resolve_to_same_instance")
        selected_ids.add(identity["track_id"])
        sampled = {**identity, "observations": [row for row in identity["observations"]
                     if row["source_frame_id"] in geometry_ids]}
        if not sampled["observations"]:
            raise ValueError(f"website_retained_static_geometry_observation_missing:{target['target_id']}")
        selected.append({"target_id": target["target_id"], "target_role": target.get("target_role"),
            "semantic_label": target["semantic_label"], "task_effect": target["task_effect"],
            "segmentation_prompt": matching_prompts[0]["text"],
            "placement_relation": target.get("placement_relation"),
            "articulated_part": target.get("articulated_part") or "",
            "articulation_kind": target.get("articulation_kind") or "",
            "disposition": target["disposition"], "source_track": identity,
            "track": track_at_geometry_resolution(sampled, source_geometry["frames"]),
            "estimated_visible_bounds": None})
    complete = {**task_masks, "status": "completed", "targets": selected,
        "retained_static_source": {"provider_binding_digest": provider_digest,
                                   "provider_result_digest": _sha256_file(result_path),
                                   "retained_video_receipt_digest": input_digest}}
    complete.pop("deferred_target_ids")
    complete["digest"] = canonical_digest(complete, digest_field="digest")
    completed_path = root / "task_masks.json"
    if completed_path.is_file() and json.loads(completed_path.read_text()) != complete:
        raise ValueError("website_retained_static_masks_manifest_changed")
    write_json(completed_path, complete)
    return complete


def bind_task_masks_to_geometry(*, task_masks: Mapping[str, Any], source_geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Lift the retained selected tracks after reconstruction; never call SAM again."""
    if task_masks.get("deferred_target_ids"):
        raise ValueError("website_static_task_masks_pending")
    if (task_masks.get("digest") != canonical_digest(task_masks, digest_field="digest")
            or source_geometry.get("digest") != canonical_digest(source_geometry, digest_field="digest")
            or task_masks.get("source_video_digest") != source_geometry["binding"].get("source_video_digest")
            or task_masks["binding"].get("geometry_input_digest") != source_geometry["binding"].get(
                "tracking_input_digest", source_geometry["binding"].get("input_digest"))
            or (source_geometry["binding"].get("tracking_input_digest") is not None
                and source_geometry["binding"].get("task_masks_digest") != task_masks["digest"])):
        raise ValueError("website_task_geometry_binding_mismatch")
    frames = source_geometry["frames"]
    ids = {frame["frame_id"] for frame in frames}
    targets = []
    for target in task_masks["targets"]:
        source_track = target["source_track"]
        track = track_at_geometry_resolution({**source_track, "observations": [row for row in source_track["observations"]
                                            if row["source_frame_id"] in ids]}, frames)
        targets.append({**target, "track": track, "estimated_visible_bounds": estimate_target_bounds(track, frames)})
    value = {**task_masks, "targets": targets, "source_geometry_digest": source_geometry["digest"],
             "binding": {**task_masks["binding"], "geometry_digest": source_geometry["digest"]}}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value
