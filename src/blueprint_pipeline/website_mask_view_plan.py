"""Choose original task and context views before hosted SAM is paid to track them."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _extract_frames, _sha256_file
from .meta_sam31 import encode_clip

SCHEMA_VERSION = "website_mask_view_plan.v1"


def prepare_mask_view_plan(*, source_video: Path, source_geometry: Mapping[str, Any],
                           plan: Mapping[str, Any], source_geometry_root: Path,
                           output_root: Path, limit: int) -> dict[str, Any]:
    """Bind a short, lossless SAM clip to the full original-frame timeline.

    The selected views include the two room endpoints, the task's observed
    moments, and nearby geometry inputs. Remaining slots maximize time spread.
    Hidden held-out frames never become method inputs. The full timeline stays
    available for downstream frame identities and uncertainty checks, but only
    selected original pixels are sent to hosted SAM.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 2:
        raise ValueError("website_mask_view_limit_invalid")
    source_digest = source_geometry["binding"]["source_video_digest"]
    if _sha256_file(source_video) != source_digest:
        raise ValueError("website_mask_view_source_changed")
    index_paths = list(source_geometry_root.rglob("decoded_observation_index.json"))
    split_paths = list(source_geometry_root.rglob("frozen_split_manifest.json"))
    if len(index_paths) != 1 or len(split_paths) != 1:
        raise ValueError("website_mask_view_source_index_missing")
    index_path, split_path = index_paths[0], split_paths[0]
    index, split = json.loads(index_path.read_text()), json.loads(split_path.read_text())
    if (index.get("capture_digest") != source_digest or split.get("capture_digest") != source_digest
            or index.get("frozen_split_digest") != split.get("split_digest")
            or split.get("split_digest") != canonical_digest(split, digest_field="split_digest")):
        raise ValueError("website_mask_view_source_index_changed")
    times = index.get("decoded_presentation_times_seconds")
    if (not isinstance(times, list) or not 2 <= len(times) <= 15000
            or any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("website_mask_view_timeline_invalid")
    heldout = {row["decoded_frame_index"] for row in split["assignments"] if row["split"] == "held_out"}
    geometry = {row["frame_id"]: row for row in source_geometry["frames"]}
    if len(geometry) != len(source_geometry["frames"]) or len(geometry) < 2:
        raise ValueError("website_mask_view_geometry_invalid")
    ordered = sorted(geometry.values(), key=lambda row: row["timestamp_seconds"])
    required = {ordered[0]["frame_id"], ordered[-1]["frame_id"]}
    allowed = [i for i in range(len(times)) if i not in heldout]
    for target in plan.get("targets", []):
        if target.get("task_effect") != "manipulated" or target.get("disposition") != "remove":
            continue
        observations = target.get("spatial_evidence") or []
        if not observations:
            raise ValueError("website_mask_view_task_anchor_missing")
        for anchor in (observations[0], observations[-1]):
            timestamp = float(anchor["timestamp_seconds"])
            nearest = min(allowed, key=lambda i: abs(times[i] - timestamp))
            if abs(times[nearest] - timestamp) > 0.5:
                raise ValueError("website_mask_view_task_anchor_missing")
            required.add(f"decoded-{nearest:09d}")
            # At least one task view also needs the already admitted geometry.
            required.add(min(ordered, key=lambda row: abs(row["timestamp_seconds"] - timestamp))["frame_id"])
    if len(required) > limit:
        raise ValueError("website_mask_view_provider_capacity_exceeded")
    candidates = set(geometry) | required
    selected = set(required)
    def timestamp(frame_id: str) -> float:
        return float(times[int(frame_id.removeprefix("decoded-"))])
    while len(selected) < min(limit, len(candidates)):
        selected.add(max(sorted(candidates - selected), key=lambda item: min(
            abs(timestamp(item) - timestamp(chosen)) for chosen in selected)))
    selected_ids = sorted(selected, key=timestamp)
    binding = {"source_geometry_digest": source_geometry["digest"], "source_video_digest": source_digest,
               "task_context_sha256": plan["task_context_sha256"], "provider_max_input_images": limit,
               "decoded_index_digest": _sha256_file(index_path), "frozen_split_digest": split["split_digest"],
               "selected_frame_ids": selected_ids}
    root = output_root / canonical_digest(binding)[7:23]
    manifest_path = root / "view_plan.json"
    if manifest_path.is_file():
        retained = json.loads(manifest_path.read_text())
        if (retained.get("binding") != binding
                or retained.get("digest") != canonical_digest(retained, digest_field="digest")
                or _sha256_file(Path(retained["video"]["path"])) != retained["video"]["sha256"]
                or any(_sha256_file(Path(row["source_image_path"])) != row["source_image_digest"]
                       for row in retained["frames"])):
            raise ValueError("website_mask_view_retained_changed")
        return retained
    root.mkdir(parents=True, exist_ok=True)
    missing = [int(frame_id.removeprefix("decoded-")) for frame_id in selected_ids if frame_id not in geometry]
    extracted = {}
    if missing:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ValueError("website_mask_view_ffmpeg_missing")
        for row in _extract_frames(video_path=source_video, ffmpeg=ffmpeg, indexes=missing,
                                   presentation_times=times, decoded_frame_metadata=[{} for _ in times],
                                   frame_root=root / "extracted"):
            extracted[row["frame_id"]] = row
    rotation = float(ordered[0]["display_rotation_degrees"])
    if not math.isfinite(rotation) or rotation % 90:
        raise ValueError("website_mask_view_rotation_invalid")
    frames, artifacts, sparse_registry = [], [], []
    for model_index, frame_id in enumerate(selected_ids):
        if frame_id in geometry:
            frame = dict(geometry[frame_id])
        else:
            row = extracted[frame_id]
            frame = {"frame_id": frame_id, "timestamp_seconds": timestamp(frame_id),
                     "source_image_path": str(root / "extracted" / f"{frame_id}.png"),
                     "source_image_digest": row["digest"], "display_rotation_degrees": rotation}
        source = Path(frame["source_image_path"])
        if _sha256_file(source) != frame["source_image_digest"]:
            raise ValueError("website_mask_view_frame_changed")
        upright_path = root / "upright" / f"{frame_id}.png"
        upright_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            upright = image.convert("RGB").rotate(rotation, expand=True)
            upright.save(upright_path)
            width, height = upright.size
        frames.append(frame)
        artifacts.append({"source_frame_id": frame_id, "path": str(upright_path),
                          "sha256": _sha256_file(upright_path)})
        sparse_registry.append({"source_frame_id": frame_id, "model_frame_index": model_index,
                                "decoded_pts_seconds": timestamp(frame_id), "width": width,
                                "height": height, "retained_video_digest": source_digest})
    clip_root = root / "clip"
    clip_root.mkdir(parents=True, exist_ok=True)
    clip = encode_clip(registry=sparse_registry, artifacts=artifacts, root=clip_root)
    video = {"path": str(clip), "sha256": _sha256_file(clip), "source_video_digest": source_digest,
             "encoding": "selected_original_views_lossless_h264_v1"}
    full_registry = [{"source_frame_id": f"decoded-{i:09d}", "model_frame_index": i,
                      "decoded_pts_seconds": float(t), "width": sparse_registry[0]["width"],
                      "height": sparse_registry[0]["height"], "retained_video_digest": source_digest}
                     for i, t in enumerate(times)]
    result = {"schema_version": SCHEMA_VERSION, "binding": binding, "frames": frames,
              "sparse_registry": sparse_registry, "source_frame_registry": full_registry,
              "video": video, "claim_ceiling": "development_only"}
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(manifest_path, result)
    return result
