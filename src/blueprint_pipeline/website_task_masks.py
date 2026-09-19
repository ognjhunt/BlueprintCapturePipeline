"""Bind task-specific SAM 3.1 tracks to original-view estimated geometry."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

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
    scores: list[tuple[float, Mapping[str, Any]]] = []
    for track in tracks:
        if track.get("label") != target["target_id"]:
            continue
        overlaps = []
        for observation in evidence:
            frame = min(frames, key=lambda f: abs(float(f["timestamp_seconds"]) - observation["timestamp_seconds"]))
            support = next((row for row in track["observations"] if row["source_frame_id"] == frame["frame_id"]), None)
            if support is None:
                continue
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


def estimate_target_bounds(track: Mapping[str, Any], frames: list[Mapping[str, Any]]) -> dict[str, Any]:
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
    minimum, maximum = np.quantile(np.concatenate(points), [0.01, 0.99], axis=0)
    return {"minimum": minimum.tolist(), "maximum": maximum.tolist(),
            "center": ((minimum + maximum) / 2).tolist(), "unit": "estimated_meters",
            "basis": "visible_masked_surfaces_with_model_estimated_depth_and_cameras",
            "complete_object_dimensions": False, "metric_measurement_proven": False}


def run_website_task_masks(*, plan: Mapping[str, Any], source_geometry: Mapping[str, Any],
                          output_root: Path, meta_admission: Mapping[str, Any] | None = None,
                          meta_admission_grant: Any = None, task_context: Mapping[str, Any] | None = None,
                          source_video: Path | None = None) -> dict[str, Any]:
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
    binding = {"geometry_digest": source_geometry["digest"], "task_targets": targets,
               "task_context_sha256": plan["task_context_sha256"], "profile_digest": profile["profile_digest"],
               "mask_input_pixels": ("continuous_source_video_v1" if source_video else "upright_source_v1")
                                    if provider == "meta" else "geometry_v1"}
    request_key = canonical_digest(binding)
    root = output_root.resolve() / request_key[7:23]
    root.mkdir(parents=True, exist_ok=True)
    frames = list(source_geometry["frames"])
    registry, artifacts, prompts = [], [], []
    prompt_labels = {}
    video_artifact = None
    if provider == "meta" and source_video is not None:
        registry, video_artifact = prepare_continuous_video(source=source_video,
            source_digest=source_geometry["binding"]["source_video_digest"], root=root)
        by_id = {row["source_frame_id"]: row for row in registry}
        for frame in frames:
            row = by_id.get(frame["frame_id"])
            if row is None or abs(row["decoded_pts_seconds"] - frame["timestamp_seconds"]) > 0.002:
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
        # Full tracks stay in the provider receipt; only geometry-backed frames
        # enter placement and background recovery. No interpolation of missing masks.
        tracks = [track_at_geometry_resolution({**track, "observations": [row for row in track["observations"]
                  if row["source_frame_id"] in geometry_ids]}, frames) for track in result["tracks"]]
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
    selected = []
    selected_track_ids = set()
    for target in targets:
        candidates = [{**track, "label": target["target_id"]} for track in tracks
                      if track.get("label") == prompt_labels[target["target_id"]]]
        track = select_task_track(target=target, tracks=candidates, frames=frames)
        if track["track_id"] in selected_track_ids:
            raise ValueError("website_task_targets_resolve_to_same_instance")
        selected_track_ids.add(track["track_id"])
        selected.append({"target_id": target["target_id"], "target_role": target.get("target_role"),
                         "semantic_label": target["semantic_label"], "task_effect": target["task_effect"],
                         "placement_relation": target.get("placement_relation"),
                         "disposition": target["disposition"], "track": track,
                         "estimated_visible_bounds": estimate_target_bounds(track, frames)})
        if provider == "meta":
            source_track = next(row for row in result["tracks"] if row["track_id"] == track["track_id"])
            selected[-1]["source_track"] = {**source_track, "observations": [row for row in source_track["observations"]
                if row["source_frame_id"] in geometry_ids]}
    manifest = {"schema_version": "website_task_masks.v1", "status": "completed", "binding": binding,
                "claim_ceiling": "development_only", "targets": selected,
                "source_geometry_digest": source_geometry["digest"]}
    manifest["digest"] = canonical_digest(manifest, digest_field="digest")
    write_json(root / "task_masks.json", manifest)
    return manifest
