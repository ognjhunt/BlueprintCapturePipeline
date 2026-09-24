"""Prepare original-resolution frames and task masks for direct image editing."""
from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from .local_reconstruction_adapters import _sha256_file, _extract_frames
from .website_task_masks import decode_track_mask


def select_reconstruction_frames(*, frames: Sequence[Mapping[str, Any]], task_masks: Mapping[str, Any],
                                 limit: int) -> list[dict[str, Any]]:
    """Use the provider's full view allowance, including wider room context."""
    visible = {row["source_frame_id"] for target in task_masks["targets"]
               if target.get("task_effect") == "manipulated"
               for row in (target.get("source_track") or target["track"])["observations"]}
    anchors = [index for index, frame in enumerate(frames) if frame["frame_id"] in visible]
    if len({frame["image_digest"] for frame in frames}) < 2:
        raise ValueError("at_least_two_distinct_reconstruction_views_required")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 2:
        raise ValueError("website_reconstruction_frame_limit_invalid")
    quality = {}
    for index, frame in enumerate(frames):
        path = Path(frame["image_path"])
        if _sha256_file(path) != frame["image_digest"]:
            raise ValueError("website_reconstruction_source_changed")
        with Image.open(path) as image:
            gray = image.convert("L")
            gray.thumbnail((256, 256))
            pixels = np.asarray(gray, dtype=float)
        laplacian = (4 * pixels[1:-1, 1:-1] - pixels[:-2, 1:-1] - pixels[2:, 1:-1]
                     - pixels[1:-1, :-2] - pixels[1:-1, 2:])
        quality[index] = float(laplacian.var()) if laplacian.size else 0.0
    # Preserve views of the actual work; object absence is not a reason to
    # discard useful room context. Review still checks every selected view.
    selected = list(dict.fromkeys([anchors[0], anchors[-1]])) if anchors else []
    if len(selected) == 2 and frames[selected[0]]["image_digest"] == frames[selected[1]]["image_digest"]:
        selected.pop()
    peak = max(quality.values())
    candidates = set(quality) - set(selected)
    while candidates and len(selected) < limit:
        seen = {frames[i]["image_digest"] for i in selected}
        candidates = {i for i in candidates if frames[i]["image_digest"] not in seen}
        if not candidates:
            break
        best = max(sorted(candidates), key=lambda i: min((abs(i - j) for j in selected), default=len(frames))
                   * (0.5 + 0.5 * quality[i] / max(peak, 1e-12)))
        selected.append(best)
        candidates.remove(best)
    return [dict(frames[i]) for i in sorted(selected)]


def replace_unmasked_task_views(*, selected: Sequence[Mapping[str, Any]],
                                frames: Sequence[Mapping[str, Any]], task_masks: Mapping[str, Any],
                                targets: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    """A missing SAM mask does not override positive task-object evidence.

    Keep completed edits byte-for-byte and replace unedited, positively observed
    task views with other context views. The whole resulting set still requires
    visual review; this does not certify object absence in replacement views.
    """
    registry = task_masks.get("source_frame_registry") or []
    visible = set()
    timestamps = {row["source_frame_id"]: row["decoded_pts_seconds"] for row in registry}
    tracks = {row.get("target_id"): row for row in task_masks.get("targets", []) if row.get("target_id")}
    for target in targets:
        if target.get("task_effect") != "manipulated" or target.get("disposition") != "remove":
            continue
        observed_times = []
        for evidence in target.get("spatial_evidence", []):
            timestamp = evidence.get("timestamp_seconds")
            if registry and isinstance(timestamp, (float, int)) and np.isfinite(timestamp):
                nearest = min(registry, key=lambda row: abs(row["decoded_pts_seconds"] - timestamp))
                visible.add(nearest["source_frame_id"])
                observed_times.append(nearest["decoded_pts_seconds"])
        tracked = tracks.get(target.get("target_id"), {})
        observations = (tracked.get("source_track") or tracked.get("track") or {}).get("observations", [])
        observed_times.extend(timestamps[row["source_frame_id"]] for row in observations
                              if row["source_frame_id"] in timestamps)
        # A tracker can start late or drop an object between observations. An
        # absent mask inside its observed visibility interval is not clearance.
        # Replace these views, without inventing masks or editing other objects.
        if observed_times:
            visible.update(frame_id for frame_id, timestamp in timestamps.items()
                           if min(observed_times) <= timestamp <= max(observed_times))
    unsafe = {frame["frame_id"] for frame in frames
              if frame["frame_id"] in visible and not frame["remaining_pixel_count"]}
    if not unsafe.intersection(frame["frame_id"] for frame in selected):
        return [dict(frame) for frame in selected]
    retained = [dict(frame) for frame in selected if frame["frame_id"] not in unsafe]
    order = {frame["frame_id"]: i for i, frame in enumerate(frames)}
    # Never add a frame needing an edit: only completed task views and unchanged
    # context may enter this reuse path. Keep using the provider's actual limit.
    candidates = [dict(frame) for frame in frames
                  if frame["frame_id"] not in unsafe and not frame["remaining_pixel_count"]]
    while len(retained) < limit:
        seen = {frame["image_digest"] for frame in retained}
        ids = {frame["frame_id"] for frame in retained}
        candidates = [frame for frame in candidates if frame["frame_id"] not in ids and frame["image_digest"] not in seen]
        if not candidates:
            break
        replacement = max(candidates, key=lambda frame: min(
            (abs(order[frame["frame_id"]] - order[row["frame_id"]]) for row in retained), default=len(frames)))
        retained.append(replacement)
    if len(retained) < 2:
        raise ValueError("at_least_two_distinct_reconstruction_views_required")
    return sorted(retained, key=lambda frame: order[frame["frame_id"]])


def reconstruction_source_frames(*, source_geometry: Mapping[str, Any], task_masks: Mapping[str, Any],
                                 source_video: Path, limit: int, output_root: Path) -> list[dict[str, Any]]:
    """CPU-decode extra context without increasing the geometry/GPU frame batch."""
    existing = {f["frame_id"]: dict(f) for f in source_geometry["frames"]}
    # A short-lived target can fall between the sparse geometry samples.
    # Retain its actual observed frames even when the context budget is full.
    required = set()
    for target in task_masks.get("targets", []):
        if target.get("task_effect") != "manipulated":
            continue
        observations = (target.get("source_track") or target.get("track") or {}).get("observations", [])
        if observations:
            required.update((observations[0]["source_frame_id"], observations[-1]["source_frame_id"]))
    if len(existing) >= limit and required <= set(existing):
        return list(existing.values())
    registry = task_masks.get("source_frame_registry") or []
    if not registry:
        raise ValueError("website_full_video_frame_registry_required")
    if (_sha256_file(source_video) != task_masks.get("source_video_digest")
            or task_masks["source_video_digest"] != source_geometry["binding"]["source_video_digest"]):
        raise ValueError("website_reconstruction_video_changed")
    count = min(limit, len(registry))
    indexes = ({round(i * (len(registry) - 1) / (count - 1)) for i in range(count)}
               if len(existing) < limit else set())
    indexes.update(i for i, row in enumerate(registry) if row["source_frame_id"] in required)
    indexes = sorted(indexes)
    indexes = [i for i in indexes if registry[i]["source_frame_id"] not in existing]
    if any(row["source_frame_id"] != f"decoded-{i:09d}" for i, row in enumerate(registry)):
        raise ValueError("website_reconstruction_frame_mapping_invalid")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("website_reconstruction_ffmpeg_missing")
    rows = _extract_frames(video_path=source_video, ffmpeg=ffmpeg, indexes=indexes,
        presentation_times=[row["decoded_pts_seconds"] for row in registry],
        decoded_frame_metadata=registry, frame_root=output_root)
    rotation = source_geometry["frames"][0]["display_rotation_degrees"]
    for row in rows:
        path = output_root / f'{row["frame_id"]}.png'
        existing[row["frame_id"]] = {"frame_id": row["frame_id"], "timestamp_seconds": row["t_video_sec"],
            "source_image_path": str(path), "source_image_digest": row["digest"],
            "display_rotation_degrees": rotation}
    return sorted(existing.values(), key=lambda frame: frame["timestamp_seconds"])


def prepare_object_removal_frames(*, frames: Sequence[Mapping[str, Any]], task_masks: Mapping[str, Any],
                                  output_root: Path) -> list[dict[str, Any]]:
    """Keep original pixels; only the editor supplies the newly exposed background."""
    output_root.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, frame in enumerate(frames):
        source = Path(frame["source_image_path"])
        if _sha256_file(source) != frame["source_image_digest"]:
            raise ValueError("website_object_removal_source_changed")
        rotation = float(frame["display_rotation_degrees"])
        if not np.isfinite(rotation) or rotation % 90:
            raise ValueError("website_source_rotation_not_supported")
        with Image.open(source) as image:
            original = image.convert("RGB").rotate(rotation, expand=True)
        mask = np.zeros((original.height, original.width), dtype=bool)
        for target in task_masks["targets"]:
            # People are removed with the manipulated task objects.
            if target["disposition"] != "remove" or (target["task_effect"] != "manipulated"
                                                     and target.get("target_class") != "person"):
                continue
            track = target.get("source_track") or target["track"]
            for observation in track["observations"]:
                if observation["source_frame_id"] != frame["frame_id"]:
                    continue
                source_mask = decode_track_mask(observation)
                if target.get("source_track") and source_mask.shape != mask.shape:
                    raise ValueError("website_object_removal_source_mask_mismatch")
                mask |= np.asarray(Image.fromarray(source_mask).resize(original.size, Image.Resampling.NEAREST))
        # Cover motion-blurred object edges and leave a narrow background-only
        # transition for compositing. Scale the margin with the source resolution.
        edge_margin = max(2, round(min(original.size) * 0.01))
        feather_pixels = max(1, round(min(original.size) * 0.005))
        if mask.any():
            mask = binary_dilation(mask, iterations=edge_margin + feather_pixels)
        image_path, mask_path = output_root / f"{index:06d}.png", output_root / f"{index:06d}.mask.png"
        original.save(image_path)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        digest = _sha256_file(image_path)
        prepared.append({"frame_id": frame["frame_id"], "image_path": str(image_path), "image_digest": digest,
                         "original_image_path": str(image_path), "original_image_digest": digest,
                         "original_source_digest": frame["source_image_digest"],
                         "remaining_mask_path": str(mask_path), "remaining_mask_digest": _sha256_file(mask_path),
                         "edge_feather_pixels": feather_pixels,
                         "remaining_pixel_count": int(mask.sum()), "generated_pixels_present": False,
                         "metric_measurement_proven": False})
    return prepared
