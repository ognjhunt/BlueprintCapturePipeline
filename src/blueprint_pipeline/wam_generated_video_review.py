"""Generated WAM video validation helpers.

These checks distinguish a generated video file that can be decoded for review
from a mere path or placeholder byte blob. Higher-level visual quality checks
still decide whether the decoded rollout is useful for task-success judging.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "wam_generated_video_review_validation.v1"
VISUAL_SMOKE_SCHEMA_VERSION = "wam_generated_rollout_visual_smoke.v1"
SOURCE_POLICY_OBSERVATION_VISUAL_QA_SCHEMA_VERSION = "source_policy_observation_visual_qa.v1"
PERSISTENT_WAM_VISUAL_QUALITY_SCHEMA_VERSION = "persistent_policy_wam_visual_quality_report.v1"
PERSISTENT_WAM_FRAME_STATS_SCHEMA_VERSION = "persistent_policy_wam_frame_stats.v1"
NEXT_OBSERVATION_SELECTION_SCHEMA_VERSION = "oscar_next_observation_selection.v1"
NEXT_OBSERVATION_ADVISORY_SIGNAL_BLOCKERS = {
    "next_observation_candidate_flat_or_low_contrast",
    "next_observation_candidate_low_scene_structure",
}

REVIEW_QUALITY_MIN_WIDTH = 320
REVIEW_QUALITY_MIN_HEIGHT = 256
REVIEW_QUALITY_MIN_FPS = 8.0
REVIEW_QUALITY_MIN_NUM_FRAMES = 12
TARGET_CENTER_MIN_X = 0.15
TARGET_CENTER_MAX_X = 0.85
TARGET_CENTER_MIN_Y = 0.12
TARGET_CENTER_MAX_Y = 0.88
TARGET_VISIBLE_AREA_RATIO_MIN = 0.55
TARGET_MIN_FRAME_AREA_RATIO = 0.00035
PROJECTED_ROBOT_MATERIAL_ROLES = {
    "hand",
    "wrist",
    "gripper",
    "end_effector",
    "forearm",
    "lower_arm",
}
PROJECTED_ROBOT_MATERIAL_MIN_REVIEW_POINTS = 2
PROJECTED_ROBOT_MATERIAL_MIN_REVIEW_EDGE_DENSITY = 0.16
PROJECTED_ROBOT_MATERIAL_MIN_REVIEW_STD_LUMA = 16.0
PROJECTED_ROBOT_MATERIAL_REVIEW_REQUIRE_ENV = (
    "BLUEPRINT_REQUIRE_PROJECTED_ROBOT_MATERIAL_DETAIL_FOR_REVIEW"
)


def _blocked(path: Path, blockers: list[str], **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "blockers": blockers,
        **fields,
    }


def validate_generated_mp4_for_review(
    path: str | Path,
    *,
    sample_frame_count: int = 6,
) -> dict[str, Any]:
    """Return decode-level review validation for a generated MP4 path."""
    video_path = Path(path).expanduser()
    if not video_path.is_file():
        return _blocked(video_path, ["generated_video_missing"])
    size_bytes = video_path.stat().st_size
    if size_bytes <= 0:
        return _blocked(video_path, ["generated_video_empty"])
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent.
        return _blocked(video_path, [f"opencv_import_failed:{type(exc).__name__}"])

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return _blocked(video_path, ["generated_video_unreadable"])
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        blockers: list[str] = []
        if frame_count <= 0:
            blockers.append("generated_video_frame_count_unavailable")
        if width <= 0 or height <= 0:
            blockers.append("generated_video_dimensions_unavailable")

        sample_indices: list[int] = []
        if frame_count > 0:
            last = max(0, frame_count - 1)
            wanted = max(1, sample_frame_count)
            sample_indices = sorted(
                {round(index * last / max(wanted - 1, 1)) for index in range(wanted)}
            )
        readable_samples = 0
        sampled_frames: list[dict[str, Any]] = []
        for frame_index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            readable_samples += 1
            sampled_frames.append(
                {
                    "frame_index": int(frame_index),
                    "height": int(frame.shape[0]),
                    "width": int(frame.shape[1]),
                }
            )
        if frame_count > 0 and readable_samples <= 0:
            blockers.append("generated_video_sample_frames_unreadable")
        status = "completed" if not blockers else "blocked"
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "path": str(video_path),
            "exists": True,
            "size_bytes": size_bytes,
            "frame_count": frame_count,
            "fps": round(fps, 6),
            "width": width,
            "height": height,
            "readable_sampled_frame_count": readable_samples,
            "sampled_frames": sampled_frames,
            "blockers": blockers,
        }
    finally:
        capture.release()


def _safe_component(value: Any, *, fallback: str = "rollout") -> str:
    text = str(value or fallback).strip().lower()
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _rollout_video_path(rollout: Mapping[str, Any]) -> Path | None:
    for key in (
        "generated_video_path",
        "video_path",
        "output_video_path",
        "path",
    ):
        value = rollout.get(key)
        if value:
            return Path(str(value)).expanduser()
    return None


def _round_float(value: Any, digits: int = 6) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _string_list(value: Any) -> list[str]:
    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _normalize_text(*parts: Any) -> str:
    text = " ".join(_string(part).lower() for part in parts if _string(part))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _tokens(*parts: Any) -> set[str]:
    return {token for token in _normalize_text(*parts).split() if token}


def _semantic_index_objects(object_index: Mapping[str, Any] | Sequence[Any] | None) -> list[dict[str, Any]]:
    if object_index is None:
        return []
    raw_objects: Any
    if isinstance(object_index, Mapping):
        raw_objects = object_index.get("objects") or object_index.get("object_index_entries")
    else:
        raw_objects = object_index
    if not isinstance(raw_objects, Sequence) or isinstance(raw_objects, (str, bytes)):
        return []
    return [dict(item) for item in raw_objects if isinstance(item, Mapping)]


def _semantic_object_id(entry: Mapping[str, Any]) -> str:
    for key in ("object_id", "instance_id", "id", "uuid", "name"):
        text = _string(entry.get(key))
        if text:
            return text
    return ""


def _semantic_object_label(entry: Mapping[str, Any]) -> str:
    for key in ("label", "class_name", "category", "name", "source_prompt"):
        text = _string(entry.get(key))
        if text:
            return text
    return _semantic_object_id(entry) or "object"


def _semantic_object_crops(entry: Mapping[str, Any]) -> list[str]:
    crops: list[str] = []
    for key in ("reference_crop", "crop_path"):
        value = _string(entry.get(key))
        if value and value not in crops:
            crops.append(value)
    for key in ("all_crops", "crop_paths", "image_paths"):
        for value in _string_list(entry.get(key)):
            if value not in crops:
                crops.append(value)
    return crops


def _semantic_object_mask_path(entry: Mapping[str, Any]) -> str:
    for key in ("mask_path", "mask_uri", "mask"):
        value = _string(entry.get(key))
        if value:
            return value
    raw_entry = _mapping(entry.get("raw_entry"))
    for key in ("mask_path", "mask_uri", "mask"):
        value = _string(raw_entry.get(key))
        if value:
            return value
    return ""


def _semantic_object_keypoints(entry: Mapping[str, Any]) -> Any:
    if entry.get("keypoints"):
        return entry.get("keypoints")
    return _mapping(entry.get("raw_entry")).get("keypoints")


def _semantic_object_bbox(entry: Mapping[str, Any]) -> Any:
    for key in (
        "source_frame_bbox",
        "bbox_xyxy",
        "bbox",
        "boundingBox",
        "box",
        "mean_box_px",
    ):
        value = entry.get(key)
        if value:
            return value
    raw_entry = _mapping(entry.get("raw_entry"))
    for key in ("source_frame_bbox", "bbox_xyxy", "bbox", "boundingBox", "box", "mean_box_px"):
        value = raw_entry.get(key)
        if value:
            return value
    return None


def _semantic_object_is_synthetic(entry: Mapping[str, Any]) -> bool:
    if bool(entry.get("synthetic_label") or entry.get("synthetic_labeled_frame")):
        return True
    raw_entry = _mapping(entry.get("raw_entry"))
    if bool(raw_entry.get("synthetic_label") or raw_entry.get("synthetic_labeled_frame")):
        return True
    source_text = _normalize_text(
        entry.get("source"),
        entry.get("label_source"),
        entry.get("provenance"),
        raw_entry.get("source"),
        raw_entry.get("label_source"),
    )
    return "synthetic" in source_text


def _candidate_match_score(
    entry: Mapping[str, Any],
    *,
    target_object_id: str | None,
    task_id: str | None,
) -> float:
    target_text = _normalize_text(target_object_id, task_id)
    if not target_text:
        return 0.0
    object_id = _semantic_object_id(entry)
    label = _semantic_object_label(entry)
    entry_text = _normalize_text(
        object_id,
        label,
        entry.get("source_prompt"),
        entry.get("description"),
        _mapping(entry.get("raw_entry")).get("source_prompt"),
    )
    if not entry_text:
        return 0.0
    score = 0.0
    if entry_text == target_text:
        score += 1.0
    if target_text in entry_text or entry_text in target_text:
        score += 0.7
    target_tokens = _tokens(target_object_id, task_id)
    entry_tokens = _tokens(object_id, label, entry.get("source_prompt"))
    overlap = target_tokens & entry_tokens
    if overlap:
        score += min(0.6, 0.12 * len(overlap))
    if {"handle", "knob", "button", "switch", "lever"} & target_tokens & entry_tokens:
        score += 0.25
    if {"sink", "faucet", "stovetop", "stove", "dishwasher", "cabinet"} & target_tokens & entry_tokens:
        score += 0.15
    return score


def _selected_semantic_target(
    *,
    object_index: Mapping[str, Any] | Sequence[Any] | None,
    eval_ready_task_grounding: Mapping[str, Any] | None,
    target_object_id: str | None,
    task_id: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    grounding = _mapping(eval_ready_task_grounding)
    selected = _mapping(grounding.get("selected_task_target"))
    if selected:
        score = _candidate_match_score(
            selected,
            target_object_id=target_object_id,
            task_id=task_id,
        )
        if score > 0.0 or not target_object_id:
            return selected, {
                "source": "eval_ready_task_grounding.selected_task_target",
                "match_score": _round_float(score),
            }

    objects = _semantic_index_objects(object_index)
    if not objects:
        return None, {"source": "none", "match_score": 0.0}
    scored = [
        (
            _candidate_match_score(
                entry,
                target_object_id=target_object_id,
                task_id=task_id,
            ),
            entry,
        )
        for entry in objects
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_entry = scored[0]
    if best_score <= 0.0:
        return None, {"source": "object_index", "match_score": 0.0}
    return dict(best_entry), {
        "source": "object_index",
        "match_score": _round_float(best_score),
    }


def _bbox_xyxy(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, Mapping):
        lowered = {str(key).lower(): item for key, item in value.items()}
        if {"x", "y", "width", "height"} <= set(lowered):
            x0 = _safe_float(lowered.get("x"))
            y0 = _safe_float(lowered.get("y"))
            width = _safe_float(lowered.get("width"))
            height = _safe_float(lowered.get("height"))
            return (x0, y0, x0 + width, y0 + height) if width > 0 and height > 0 else None
        if {"left", "top", "width", "height"} <= set(lowered):
            x0 = _safe_float(lowered.get("left"))
            y0 = _safe_float(lowered.get("top"))
            width = _safe_float(lowered.get("width"))
            height = _safe_float(lowered.get("height"))
            return (x0, y0, x0 + width, y0 + height) if width > 0 and height > 0 else None
        x0 = lowered.get("x0", lowered.get("xmin", lowered.get("x_min")))
        y0 = lowered.get("y0", lowered.get("ymin", lowered.get("y_min")))
        x1 = lowered.get("x1", lowered.get("xmax", lowered.get("x_max")))
        y1 = lowered.get("y1", lowered.get("ymax", lowered.get("y_max")))
        if x0 is not None and y0 is not None and x1 is not None and y1 is not None:
            box = (_safe_float(x0), _safe_float(y0), _safe_float(x1), _safe_float(y1))
            return box if box[2] > box[0] and box[3] > box[1] else None
        if {"cx", "cy", "width", "height"} <= set(lowered):
            cx = _safe_float(lowered.get("cx"))
            cy = _safe_float(lowered.get("cy"))
            width = _safe_float(lowered.get("width"))
            height = _safe_float(lowered.get("height"))
            if width > 0 and height > 0:
                return (cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5)
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 4:
        x0 = _safe_float(value[0])
        y0 = _safe_float(value[1])
        third = _safe_float(value[2])
        fourth = _safe_float(value[3])
        if third > x0 and fourth > y0:
            return (x0, y0, third, fourth)
        return (x0, y0, x0 + third, y0 + fourth) if third > 0 and fourth > 0 else None
    return None


def _point2(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        for key in ("center", "center_px", "target_center_px", "uv", "point"):
            point = _point2(value.get(key))
            if point is not None:
                return point
        if "x" in value and "y" in value:
            return [_safe_float(value.get("x")), _safe_float(value.get("y"))]
        if "u" in value and "v" in value:
            return [_safe_float(value.get("u")), _safe_float(value.get("v"))]
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) >= 2 and not isinstance(value[0], (Mapping, list, tuple)):
            return [_safe_float(value[0]), _safe_float(value[1])]
        for item in value:
            point = _point2(item)
            if point is not None:
                return point
    return None


def _target_center_and_radius(entry: Mapping[str, Any]) -> tuple[list[float] | None, float | None]:
    bbox = _bbox_xyxy(_semantic_object_bbox(entry))
    if bbox is not None:
        width = max(1.0, bbox[2] - bbox[0])
        height = max(1.0, bbox[3] - bbox[1])
        return [bbox[0] + width * 0.5, bbox[1] + height * 0.5], max(width, height) * 0.6
    point = _point2(_semantic_object_keypoints(entry))
    return (point, 40.0) if point is not None else (None, None)


def _resolve_local_artifact_path(value: Any, *, base_dir: str | Path | None) -> Path | None:
    text = _string(value)
    if not text or "://" in text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir).expanduser() / path
    return path


def _luma_region_stats(array: Any) -> dict[str, Any]:
    import numpy as np

    luma = 0.2126 * array[:, :, 0] + 0.7152 * array[:, :, 1] + 0.0722 * array[:, :, 2]
    height, width = luma.shape
    histogram, _ = np.histogram(luma, bins=128, range=(0, 255))
    total = int(histogram.sum())
    probabilities = histogram[histogram > 0] / max(total, 1)
    entropy_bits = float(-(probabilities * np.log2(probabilities)).sum()) if total else 0.0
    gradient_y, gradient_x = np.gradient(luma)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    return {
        "status": "completed",
        "width": int(width),
        "height": int(height),
        "mean_luma": _round_float(float(luma.mean()), 3),
        "std_luma": _round_float(float(luma.std()), 3),
        "luma_range": _round_float(float(luma.max() - luma.min()), 3),
        "dark_pixel_ratio": _round_float(float((luma < 32.0).mean()), 6),
        "entropy_bits": _round_float(entropy_bits, 6),
        "edge_density": _round_float(float((gradient_magnitude > 18.0).mean()), 6),
    }


def _target_region_stats(
    frame_path: str | Path | None,
    *,
    target: Mapping[str, Any],
    base_dir: str | Path | None,
) -> dict[str, Any]:
    if frame_path is None:
        return {"status": "blocked", "blockers": ["source_policy_observation_frame_missing"]}
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency/environment edge.
        return {"status": "blocked", "blockers": [f"target_region_dependency_import_failed:{type(exc).__name__}"]}
    try:
        with Image.open(Path(frame_path).expanduser()) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb).astype("float32")
    except Exception as exc:
        return {"status": "blocked", "blockers": [f"source_policy_observation_frame_unreadable:{type(exc).__name__}"]}

    height, width = array.shape[:2]
    bbox = _bbox_xyxy(_semantic_object_bbox(target))
    center, radius = _target_center_and_radius(target)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        pad_x = max(4.0, (x1 - x0) * 0.15)
        pad_y = max(4.0, (y1 - y0) * 0.15)
        region = (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)
    elif center is not None and radius is not None:
        region = (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        )
    else:
        region = None

    if region is not None:
        x0 = max(0, int(math.floor(region[0])))
        y0 = max(0, int(math.floor(region[1])))
        x1 = min(width, int(math.ceil(region[2])))
        y1 = min(height, int(math.ceil(region[3])))
        if x1 > x0 and y1 > y0:
            stats = _luma_region_stats(array[y0:y1, x0:x1])
            stats["source"] = "source_frame_target_region"
            stats["region_xyxy"] = [x0, y0, x1, y1]
            return stats

    for crop in _semantic_object_crops(target):
        crop_path = _resolve_local_artifact_path(crop, base_dir=base_dir)
        if crop_path is None or not crop_path.is_file():
            continue
        try:
            with Image.open(crop_path) as crop_image:
                crop_rgb = crop_image.convert("RGB")
                crop_array = np.asarray(crop_rgb).astype("float32")
        except Exception:
            continue
        stats = _luma_region_stats(crop_array)
        stats["source"] = "object_index_crop"
        stats["crop_path"] = str(crop_path)
        return stats

    return {"status": "blocked", "blockers": ["target_task_region_visual_probe_unavailable"]}


def _mask_visibility_stats(
    target: Mapping[str, Any],
    *,
    frame_width: int,
    frame_height: int,
    base_dir: str | Path | None,
) -> dict[str, Any]:
    mask_path_text = _semantic_object_mask_path(target)
    if not mask_path_text:
        return {"available": False}
    path = _resolve_local_artifact_path(mask_path_text, base_dir=base_dir)
    if path is None or not path.is_file():
        return {"available": True, "path": mask_path_text, "local_file_exists": False}
    try:
        from PIL import Image
        import numpy as np

        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"))
    except Exception as exc:
        return {
            "available": True,
            "path": str(path),
            "local_file_exists": True,
            "status": f"unreadable:{type(exc).__name__}",
        }
    nonzero = mask > 0
    nonzero_ratio = float(nonzero.mean()) if mask.size else 0.0
    frame_area = max(1, int(frame_width) * int(frame_height))
    source_frame_ratio = float(nonzero.sum()) / float(frame_area)
    return {
        "available": True,
        "path": str(path),
        "local_file_exists": True,
        "width": int(mask.shape[1]) if mask.ndim >= 2 else 0,
        "height": int(mask.shape[0]) if mask.ndim >= 2 else 0,
        "mask_nonzero_ratio": _round_float(nonzero_ratio, 6),
        "source_frame_area_ratio": _round_float(source_frame_ratio, 6),
    }


def _target_region_quality_passed(stats: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if stats.get("status") != "completed":
        return False, _string_list(stats.get("blockers")) or [
            "target_task_region_visual_probe_unavailable"
        ]
    mean_luma = float(stats.get("mean_luma") or 0.0)
    std_luma = float(stats.get("std_luma") or 0.0)
    luma_range = float(stats.get("luma_range") or 0.0)
    dark_ratio = float(stats.get("dark_pixel_ratio") or 0.0)
    entropy = float(stats.get("entropy_bits") or 0.0)
    edge_density = float(stats.get("edge_density") or 0.0)
    blockers: list[str] = []
    if mean_luma < 35.0 or dark_ratio > 0.60:
        blockers.append("target_task_region_too_dark_or_low_information")
    if std_luma < 8.0 or luma_range < 35.0 or entropy < 1.35 or edge_density < 0.002:
        blockers.append("target_task_region_too_flat_or_low_detail")
    return not blockers, blockers


def _semantic_target_quality(
    *,
    frame_path: str | Path | None,
    stats: Mapping[str, Any],
    target_object_id: str | None,
    task_id: str | None,
    object_index: Mapping[str, Any] | Sequence[Any] | None,
    eval_ready_task_grounding: Mapping[str, Any] | None,
    semantic_artifact_base_dir: str | Path | None,
) -> dict[str, Any]:
    objects = _semantic_index_objects(object_index)
    grounding = _mapping(eval_ready_task_grounding)
    semantic_available = bool(objects or grounding)
    result: dict[str, Any] = {
        "schema_version": "source_policy_observation_semantic_target_quality.v1",
        "status": "not_available",
        "available": semantic_available,
        "target_object_id": target_object_id,
        "task_id": task_id,
        "eval_ready_task_grounding_used": bool(grounding),
        "object_index_used": bool(objects),
        "semantic_artifact_base_dir": str(Path(semantic_artifact_base_dir).expanduser())
        if semantic_artifact_base_dir
        else None,
        "gates": {},
        "warnings": [],
        "blockers": [],
    }
    if not semantic_available:
        result["warnings"] = ["semantic_target_artifacts_not_supplied"]
        return result
    if stats.get("status") != "completed":
        result["status"] = "failed"
        result["blockers"] = _string_list(stats.get("blockers")) or [
            "source_policy_observation_visual_probe_failed"
        ]
        return result

    target, selection = _selected_semantic_target(
        object_index=object_index,
        eval_ready_task_grounding=grounding,
        target_object_id=target_object_id,
        task_id=task_id,
    )
    result["selection"] = selection
    if target is None:
        result["status"] = "failed"
        result["blockers"] = ["target_object_not_found_in_semantic_index"]
        result["gates"] = {
            "target_object_visibility": {
                "passed": False,
                "reason": "target_object_not_found_in_semantic_index",
            }
        }
        return result

    width = int(stats.get("width") or 0)
    height = int(stats.get("height") or 0)
    bbox = _bbox_xyxy(_semantic_object_bbox(target))
    center, radius = _target_center_and_radius(target)
    crops = _semantic_object_crops(target)
    mask_path = _semantic_object_mask_path(target)
    keypoints = _semantic_object_keypoints(target)
    mask_stats = _mask_visibility_stats(
        target,
        frame_width=width,
        frame_height=height,
        base_dir=semantic_artifact_base_dir,
    )
    synthetic_label = _semantic_object_is_synthetic(target)
    target_summary = {
        "object_id": _semantic_object_id(target),
        "label": _semantic_object_label(target),
        "bbox_xyxy": [_round_float(value, 3) for value in bbox] if bbox is not None else None,
        "target_center_px": [_round_float(value, 3) for value in center] if center else None,
        "target_radius_px": _round_float(radius, 3) if radius is not None else None,
        "crop_paths": crops,
        "crop_available": bool(crops),
        "mask_path": mask_path or None,
        "mask_available": bool(mask_path or mask_stats.get("available")),
        "keypoints_available": bool(keypoints),
        "synthetic_label_evidence": synthetic_label,
    }
    result["selected_target"] = target_summary
    blockers: list[str] = []
    warnings: list[str] = []
    gates: dict[str, Any] = {}

    semantic_evidence_passed = bool(bbox or center or mask_path or crops)
    gates["target_semantic_evidence"] = {
        "passed": semantic_evidence_passed,
        "bbox_available": bbox is not None,
        "crop_available": bool(crops),
        "mask_available": bool(mask_path or mask_stats.get("available")),
        "keypoints_available": bool(keypoints),
        "synthetic_label_evidence": synthetic_label,
    }
    if not semantic_evidence_passed:
        blockers.append("target_object_index_lacks_crop_mask_keypoint_or_bbox")

    visible_passed = False
    visible_ratio = None
    frame_area_ratio = None
    if bbox is not None and width > 0 and height > 0:
        x0, y0, x1, y1 = bbox
        box_area = max(0.0, (x1 - x0) * (y1 - y0))
        ix0 = min(max(0.0, x0), float(width))
        iy0 = min(max(0.0, y0), float(height))
        ix1 = min(max(0.0, x1), float(width))
        iy1 = min(max(0.0, y1), float(height))
        visible_area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        visible_ratio = visible_area / box_area if box_area > 0 else 0.0
        frame_area_ratio = visible_area / float(max(1, width * height))
        visible_passed = bool(
            visible_area > 0
            and visible_ratio >= TARGET_VISIBLE_AREA_RATIO_MIN
            and frame_area_ratio >= TARGET_MIN_FRAME_AREA_RATIO
        )
    elif center is not None and width > 0 and height > 0:
        visible_passed = 0.0 <= center[0] < float(width) and 0.0 <= center[1] < float(height)
    gates["target_object_visibility"] = {
        "passed": visible_passed,
        "visible_area_ratio": _round_float(visible_ratio) if visible_ratio is not None else None,
        "source_frame_area_ratio": _round_float(frame_area_ratio)
        if frame_area_ratio is not None
        else None,
    }
    if not visible_passed:
        if center is not None and (center[0] < 0 or center[0] >= width or center[1] < 0 or center[1] >= height):
            blockers.append("target_object_offscreen_in_source_observation")
        else:
            blockers.append("target_object_not_visible_in_source_observation")
    if frame_area_ratio is not None and frame_area_ratio < TARGET_MIN_FRAME_AREA_RATIO:
        blockers.append("target_object_too_small_for_initial_observation")

    centered_passed = bool(
        center is not None
        and width > 0
        and height > 0
        and TARGET_CENTER_MIN_X <= center[0] / float(width) <= TARGET_CENTER_MAX_X
        and TARGET_CENTER_MIN_Y <= center[1] / float(height) <= TARGET_CENTER_MAX_Y
    )
    gates["target_centering"] = {
        "passed": centered_passed,
        "center_px": target_summary["target_center_px"],
        "center_normalized": [
            _round_float(center[0] / float(width)),
            _round_float(center[1] / float(height)),
        ]
        if center is not None and width > 0 and height > 0
        else None,
        "accepted_normalized_bounds": {
            "min_x": TARGET_CENTER_MIN_X,
            "max_x": TARGET_CENTER_MAX_X,
            "min_y": TARGET_CENTER_MIN_Y,
            "max_y": TARGET_CENTER_MAX_Y,
        },
    }
    if center is None:
        blockers.append("target_region_center_unavailable")
    elif not centered_passed:
        blockers.append("target_object_not_centered_for_initial_wam_observation")

    occlusion_text = _normalize_text(
        target.get("occlusion"),
        target.get("occlusion_status"),
        target.get("visibility"),
        target.get("visibility_status"),
    )
    explicit_occluded = any(
        token in occlusion_text
        for token in ("occluded", "blocked", "hidden", "not visible", "invisible", "covered")
    ) and not any(token in occlusion_text for token in ("not occluded", "visible", "clear"))
    mask_area_ratio = float(mask_stats.get("source_frame_area_ratio") or 0.0)
    mask_too_small = bool(mask_stats.get("local_file_exists") and mask_area_ratio < TARGET_MIN_FRAME_AREA_RATIO)
    occlusion_passed = bool(visible_passed and not explicit_occluded and not mask_too_small)
    gates["target_occlusion"] = {
        "passed": occlusion_passed,
        "explicit_occlusion_text": occlusion_text or None,
        "mask_visibility": mask_stats,
    }
    if explicit_occluded:
        blockers.append("target_object_marked_occluded_by_semantic_evidence")
    if mask_too_small:
        blockers.append("target_mask_too_small_for_visibility")

    region_stats = _target_region_stats(
        frame_path,
        target=target,
        base_dir=semantic_artifact_base_dir,
    )
    region_passed, region_blockers = _target_region_quality_passed(region_stats)
    gates["task_region_quality"] = {
        "passed": region_passed,
        "metrics": region_stats,
    }
    blockers.extend(region_blockers)

    readiness = _mapping(grounding.get("readiness"))
    grounding_blockers = _string_list(readiness.get("blockers"))
    if grounding and any("missing_task_target" in blocker for blocker in grounding_blockers):
        blockers.append("eval_ready_task_grounding_target_blocked")
    if grounding and readiness.get("target_crop_available") is False and not crops:
        warnings.append("eval_ready_task_grounding_reports_target_crop_unavailable")
    if grounding and readiness.get("target_mask_or_keypoint_available") is False and not (
        keypoints or mask_path
    ):
        warnings.append("eval_ready_task_grounding_reports_target_mask_or_keypoint_unavailable")

    result["gates"] = gates
    result["warnings"] = sorted(set(warnings))
    result["blockers"] = sorted(set(blockers))
    result["status"] = "passed" if not result["blockers"] else "failed"
    result["synthetic_label_evidence_used"] = synthetic_label
    return result


def _frame_visual_stats(
    path: str | Path,
    *,
    role: str,
    frame_index: int | None = None,
    source_frame_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    frame_path = Path(path).expanduser()
    result: dict[str, Any] = {
        "schema_version": PERSISTENT_WAM_FRAME_STATS_SCHEMA_VERSION,
        "role": role,
        "frame_index": frame_index,
        "path": str(frame_path),
        "status": "blocked",
        "blockers": [],
    }
    if not frame_path.is_file():
        result["blockers"] = ["frame_missing_for_visual_quality"]
        return result
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency/environment edge.
        result["blockers"] = [f"visual_quality_dependency_import_failed:{type(exc).__name__}"]
        return result
    try:
        with Image.open(frame_path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb).astype("float32")
    except Exception as exc:
        result["blockers"] = [f"frame_unreadable_for_visual_quality:{type(exc).__name__}"]
        return result

    luma = 0.2126 * array[:, :, 0] + 0.7152 * array[:, :, 1] + 0.0722 * array[:, :, 2]
    height, width = luma.shape
    histogram, _ = np.histogram(luma, bins=256, range=(0, 255))
    total = int(histogram.sum())
    probabilities = histogram[histogram > 0] / max(total, 1)
    entropy_bits = float(-(probabilities * np.log2(probabilities)).sum()) if total else 0.0
    gradient_y, gradient_x = np.gradient(luma)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    edge_density = float((gradient_magnitude > 18.0).mean())
    if height >= 3 and width >= 3:
        laplacian = (
            -4.0 * luma[1:-1, 1:-1]
            + luma[:-2, 1:-1]
            + luma[2:, 1:-1]
            + luma[1:-1, :-2]
            + luma[1:-1, 2:]
        )
        sharpness_laplacian_variance = float(laplacian.var())
    else:
        sharpness_laplacian_variance = 0.0
    center_y0 = max(0, int(height * 0.25))
    center_y1 = min(height, max(center_y0 + 1, int(height * 0.75)))
    center_x0 = max(0, int(width * 0.25))
    center_x1 = min(width, max(center_x0 + 1, int(width * 0.75)))
    center = luma[center_y0:center_y1, center_x0:center_x1]
    center_gradient = gradient_magnitude[center_y0:center_y1, center_x0:center_x1]
    center_histogram, _ = np.histogram(center, bins=128, range=(0, 255))
    center_probabilities = center_histogram[center_histogram > 0] / max(
        int(center_histogram.sum()),
        1,
    )
    center_entropy = (
        float(-(center_probabilities * np.log2(center_probabilities)).sum())
        if center_histogram.sum()
        else 0.0
    )
    dominant_luma_bin_ratio = float(histogram.max() / max(total, 1)) if total else 0.0

    source = dict(source_frame_stats or {})
    source_edge_density = float(source.get("edge_density") or 0.0)
    source_mean_luma = float(source.get("mean_luma") or 0.0)
    source_std_luma = float(source.get("std_luma") or 0.0)
    source_entropy_bits = float(source.get("entropy_bits") or 0.0)
    drift = {
        "mean_luma_delta_from_source": _round_float(float(luma.mean()) - source_mean_luma, 3)
        if source
        else None,
        "std_luma_ratio_to_source": _round_float(float(luma.std()) / source_std_luma, 6)
        if source_std_luma > 0.0
        else None,
        "edge_density_ratio_to_source": _round_float(edge_density / source_edge_density, 6)
        if source_edge_density > 0.0
        else None,
        "entropy_delta_from_source": _round_float(entropy_bits - source_entropy_bits, 3)
        if source
        else None,
    }
    result.update(
        {
            "status": "completed",
            "width": int(width),
            "height": int(height),
            "mean_luma": _round_float(float(luma.mean()), 3),
            "std_luma": _round_float(float(luma.std()), 3),
            "luma_min": _round_float(float(luma.min()), 3),
            "luma_max": _round_float(float(luma.max()), 3),
            "luma_range": _round_float(float(luma.max() - luma.min()), 3),
            "dark_pixel_ratio": _round_float(float((luma < 32.0).mean()), 6),
            "near_black_pixel_ratio": _round_float(float((luma < 16.0).mean()), 6),
            "bright_pixel_ratio": _round_float(float((luma > 224.0).mean()), 6),
            "dominant_luma_bin_ratio": _round_float(dominant_luma_bin_ratio, 6),
            "entropy_bits": _round_float(entropy_bits, 6),
            "edge_density": _round_float(edge_density, 6),
            "sharpness_laplacian_variance": _round_float(sharpness_laplacian_variance, 3),
            "center_crop": {
                "x0": int(center_x0),
                "y0": int(center_y0),
                "x1": int(center_x1),
                "y1": int(center_y1),
                "mean_luma": _round_float(float(center.mean()), 3),
                "std_luma": _round_float(float(center.std()), 3),
                "dark_pixel_ratio": _round_float(float((center < 32.0).mean()), 6),
                "entropy_bits": _round_float(center_entropy, 6),
                "edge_density": _round_float(float((center_gradient > 18.0).mean()), 6),
            },
            "drift_from_source": drift,
            "blockers": [],
        }
    )
    return result


def _cv2_frame_signal_stats(frame: Any, cv2: Any) -> dict[str, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    edges = cv2.Canny(gray, 50, 150)
    return {
        "mean_luma": _round_float(float(gray.mean()), 3),
        "std_luma": _round_float(float(gray.std()), 3),
        "luma_min": int(gray.min()),
        "luma_max": int(gray.max()),
        "luma_range": int(gray.max()) - int(gray.min()),
        "dark_pixel_ratio": _round_float(float((gray < 32).mean()), 6),
        "bright_pixel_ratio": _round_float(float((gray > 224).mean()), 6),
        "edge_density": _round_float(float((edges > 0).mean()), 6),
    }


def _next_observation_signal_blockers(stats: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    mean_luma = float(stats.get("mean_luma") or 0.0)
    std_luma = float(stats.get("std_luma") or 0.0)
    luma_range = float(stats.get("luma_range") or 0.0)
    dark_ratio = float(stats.get("dark_pixel_ratio") or 0.0)
    bright_ratio = float(stats.get("bright_pixel_ratio") or 0.0)
    edge_density = float(stats.get("edge_density") or 0.0)
    if mean_luma < 25.0 or dark_ratio > 0.78:
        blockers.append("next_observation_candidate_too_dark")
    if mean_luma > 245.0 and bright_ratio > 0.90:
        blockers.append("next_observation_candidate_overexposed")
    if std_luma < 8.0 or luma_range < 32.0:
        blockers.append("next_observation_candidate_flat_or_low_contrast")
    if edge_density < 0.002:
        blockers.append("next_observation_candidate_low_scene_structure")
    if edge_density > 0.12 and std_luma < 28.0:
        blockers.append("next_observation_candidate_static_noise_artifact")
    return blockers


def _write_next_observation_selection_manifest(
    out_dir: Path,
    *,
    status: str,
    video_path: Path,
    candidates: Sequence[Mapping[str, Any]],
    selected_frame_index: int | None,
    blockers: Sequence[str],
    extraction_method: str,
    selection_quality_status: str | None = None,
    selected_frame_signal_blockers: Sequence[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "next_observation_selection.json").write_text(
        json.dumps(
            {
                "schema_version": NEXT_OBSERVATION_SELECTION_SCHEMA_VERSION,
                "status": status,
                "video_path": str(video_path),
                "selected_frame_index": selected_frame_index,
                "extraction_method": extraction_method,
                "selection_quality_status": selection_quality_status
                or ("passed_signal_gate" if status == "completed" else "blocked"),
                "selected_frame_signal_blockers": list(selected_frame_signal_blockers or []),
                "candidate_count": len(candidates),
                "candidates": list(candidates),
                "blockers": list(blockers),
                "claim_boundary": {
                    "selected_frame_is_generated_next_observation_candidate": (
                        status == "completed"
                    ),
                    "visual_signal_gate_is_not_task_success_evidence": True,
                    "scene_or_task_specific_pixels_used": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def extract_next_observation_frame_from_video(
    video_path: str | Path,
    out_dir: str | Path,
) -> Path | None:
    """Select a non-seed frame from a generated rollout video.

    Signal-valid future frames are preferred. If no future frame passes the
    visual signal gate, the earliest decodable non-terminal future frame is
    still materialized and labeled with signal warnings so downstream gates can
    judge WAM quality without pretending frame 0 was a rollout result.
    """
    resolved_out = Path(out_dir).expanduser()
    resolved_out.mkdir(parents=True, exist_ok=True)
    resolved_video = Path(video_path).expanduser()
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        cv2 = None
    if cv2 is not None:
        capture = cv2.VideoCapture(str(resolved_video))
        candidates: list[dict[str, Any]] = []
        selected_index: int | None = None
        selected_frame = None
        warning_index: int | None = None
        warning_frame = None
        warning_blockers: list[str] = []
        try:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                stats = _cv2_frame_signal_stats(frame, cv2)
                blockers = (
                    ["next_observation_candidate_is_seed_frame"]
                    if frame_index == 0
                    else _next_observation_signal_blockers(stats)
                )
                terminal_blockers = [
                    blocker
                    for blocker in blockers
                    if blocker not in NEXT_OBSERVATION_ADVISORY_SIGNAL_BLOCKERS
                    and blocker != "next_observation_candidate_is_seed_frame"
                ]
                materializable_future_frame = bool(frame_index > 0 and not terminal_blockers)
                candidates.append(
                    {
                        "frame_index": frame_index,
                        "metrics": stats,
                        "blockers": blockers,
                        "usable_future_frame": bool(frame_index > 0 and not blockers),
                        "materializable_future_frame": materializable_future_frame,
                        "terminal_signal_blockers": terminal_blockers,
                    }
                )
                if frame_index > 0 and not blockers and selected_frame is None:
                    selected_index = frame_index
                    selected_frame = frame
                    break
                if (
                    frame_index > 0
                    and materializable_future_frame
                    and warning_frame is None
                ):
                    warning_index = frame_index
                    warning_frame = frame
                    warning_blockers = list(blockers)
                frame_index += 1
        finally:
            capture.release()
        if selected_frame is not None and selected_index is not None:
            selected_path = resolved_out / f"next_observation_frame_{selected_index:04d}.jpg"
            if cv2.imwrite(str(selected_path), selected_frame):
                _write_next_observation_selection_manifest(
                    resolved_out,
                    status="completed",
                    video_path=resolved_video,
                    candidates=candidates,
                    selected_frame_index=selected_index,
                    blockers=[],
                    extraction_method="cv2_earliest_signal_valid_future_frame",
                    selection_quality_status="passed_signal_gate",
                )
                return selected_path
        if warning_frame is not None and warning_index is not None:
            selected_path = resolved_out / f"next_observation_frame_{warning_index:04d}.jpg"
            if cv2.imwrite(str(selected_path), warning_frame):
                _write_next_observation_selection_manifest(
                    resolved_out,
                    status="completed",
                    video_path=resolved_video,
                    candidates=candidates,
                    selected_frame_index=warning_index,
                    blockers=[],
                    extraction_method="cv2_earliest_decodable_future_frame_with_signal_warnings",
                    selection_quality_status="degraded_visual_signal",
                    selected_frame_signal_blockers=warning_blockers,
                )
                return selected_path
        blockers = ["no_usable_future_next_observation_frame"]
        if not candidates:
            blockers.append("generated_video_has_no_readable_frames")
        _write_next_observation_selection_manifest(
            resolved_out,
            status="blocked",
            video_path=resolved_video,
            candidates=candidates,
            selected_frame_index=None,
            blockers=blockers,
            extraction_method="cv2_earliest_signal_valid_future_frame",
        )
        return None

    try:
        from PIL import Image, ImageSequence
    except ImportError:
        _write_next_observation_selection_manifest(
            resolved_out,
            status="blocked",
            video_path=resolved_video,
            candidates=[],
            selected_frame_index=None,
            blockers=["next_observation_video_decode_dependency_missing"],
            extraction_method="pillow_sequence_fallback",
        )
        return None
    try:
        with Image.open(resolved_video) as image:
            for frame_index, frame in enumerate(ImageSequence.Iterator(image)):
                if frame_index == 0:
                    continue
                selected_path = resolved_out / f"next_observation_frame_{frame_index:04d}.jpg"
                frame.convert("RGB").save(selected_path)
                _write_next_observation_selection_manifest(
                    resolved_out,
                    status="completed",
                    video_path=resolved_video,
                    candidates=[{"frame_index": frame_index, "usable_future_frame": True}],
                    selected_frame_index=frame_index,
                    blockers=[],
                    extraction_method="pillow_sequence_fallback",
                )
                return selected_path
    except Exception as exc:
        _write_next_observation_selection_manifest(
            resolved_out,
            status="blocked",
            video_path=resolved_video,
            candidates=[],
            selected_frame_index=None,
            blockers=[f"next_observation_video_decode_failed:{type(exc).__name__}"],
            extraction_method="pillow_sequence_fallback",
        )
        return None
    _write_next_observation_selection_manifest(
        resolved_out,
        status="blocked",
        video_path=resolved_video,
        candidates=[],
        selected_frame_index=None,
        blockers=["generated_video_has_no_future_frames"],
        extraction_method="pillow_sequence_fallback",
    )
    return None


def _trace_image_size(row: Mapping[str, Any]) -> tuple[float, float] | None:
    size = row.get("image_size_px") or row.get("image_size")
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)) and len(size) >= 2:
        try:
            return float(size[0]), float(size[1])
        except (TypeError, ValueError):
            return None
    if isinstance(size, Mapping):
        try:
            return float(size.get("width") or 0), float(size.get("height") or 0)
        except (TypeError, ValueError):
            return None
    return None


def _first_projected_robot_trace_row(path: Path) -> dict[str, Any] | None:
    first_projectable: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, Mapping):
                continue
            landmarks = row.get("landmarks")
            if not isinstance(landmarks, Sequence) or isinstance(landmarks, (str, bytes)):
                continue
            has_projected = False
            for landmark in landmarks:
                if not isinstance(landmark, Mapping):
                    continue
                projection = landmark.get("image_projection")
                if isinstance(projection, Mapping) and projection.get("available") is True:
                    has_projected = True
                    break
            if not has_projected:
                continue
            if int(row.get("frame_index") or 0) == 0:
                return dict(row)
            if first_projectable is None:
                first_projectable = dict(row)
    return first_projectable


def _projected_robot_material_quality(
    frame_path: str | Path | None,
    *,
    projected_skeleton_trace_path: str | Path | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "projected_robot_material_quality.v1",
        "status": "not_available",
        "projected_skeleton_trace_path": str(Path(projected_skeleton_trace_path).expanduser())
        if projected_skeleton_trace_path
        else None,
        "projected_skeleton_trace_used": False,
        "projected_skeleton_trace_available": False,
        "review_quality_gate_available": False,
        "blockers": [],
    }
    if frame_path is None:
        return result
    if projected_skeleton_trace_path is None:
        return result
    trace_path = Path(projected_skeleton_trace_path).expanduser()
    if not trace_path.is_file():
        result.update(
            {
                "status": "blocked",
                "blockers": ["projected_skeleton_trace_missing_for_robot_material_quality"],
            }
        )
        return result
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency/environment edge.
        result.update(
            {
                "status": "blocked",
                "blockers": [
                    f"projected_robot_material_quality_dependency_import_failed:{type(exc).__name__}"
                ],
            }
        )
        return result
    try:
        row = _first_projected_robot_trace_row(trace_path)
    except (OSError, json.JSONDecodeError) as exc:
        result.update(
            {
                "status": "blocked",
                "blockers": [f"projected_skeleton_trace_unreadable:{type(exc).__name__}"],
            }
        )
        return result
    if not row:
        result.update(
            {
                "status": "blocked",
                "projected_skeleton_trace_available": True,
                "blockers": ["projected_skeleton_trace_has_no_projected_robot_points"],
            }
        )
        return result
    try:
        with Image.open(Path(frame_path).expanduser()) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb).astype("float32")
    except Exception as exc:
        result.update(
            {
                "status": "blocked",
                "projected_skeleton_trace_available": True,
                "blockers": [f"source_policy_observation_frame_unreadable:{type(exc).__name__}"],
            }
        )
        return result
    height, width = array.shape[:2]
    trace_size = _trace_image_size(row)
    trace_width = trace_size[0] if trace_size and trace_size[0] > 0 else float(width)
    trace_height = trace_size[1] if trace_size and trace_size[1] > 0 else float(height)
    scale_x = float(width) / trace_width
    scale_y = float(height) / trace_height
    radius = max(12, int(round(min(width, height) * 0.045)))
    luma = 0.2126 * array[:, :, 0] + 0.7152 * array[:, :, 1] + 0.0722 * array[:, :, 2]
    gradient_y, gradient_x = np.gradient(luma)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    landmarks = row.get("landmarks")
    roi_metrics: list[dict[str, Any]] = []
    if isinstance(landmarks, Sequence) and not isinstance(landmarks, (str, bytes)):
        for landmark in landmarks:
            if not isinstance(landmark, Mapping):
                continue
            role = _string(landmark.get("link_role") or landmark.get("role")).lower()
            landmark_id = _string(landmark.get("landmark_id") or landmark.get("link_name"))
            role_or_name = f"{role} {landmark_id}".lower()
            if not (
                role in PROJECTED_ROBOT_MATERIAL_ROLES
                or any(token in role_or_name for token in PROJECTED_ROBOT_MATERIAL_ROLES)
            ):
                continue
            projection = landmark.get("image_projection")
            if not isinstance(projection, Mapping) or projection.get("available") is not True:
                continue
            try:
                u_px = float(projection.get("u_px")) * scale_x
                v_px = float(projection.get("v_px")) * scale_y
            except (TypeError, ValueError):
                continue
            if not (0.0 <= u_px < width and 0.0 <= v_px < height):
                continue
            x0 = max(0, int(round(u_px)) - radius)
            y0 = max(0, int(round(v_px)) - radius)
            x1 = min(width, int(round(u_px)) + radius + 1)
            y1 = min(height, int(round(v_px)) + radius + 1)
            roi = luma[y0:y1, x0:x1]
            roi_gradient = gradient_magnitude[y0:y1, x0:x1]
            if roi.size <= 0:
                continue
            roi_metrics.append(
                {
                    "landmark_id": landmark_id or None,
                    "link_role": role or None,
                    "u_px": _round_float(u_px, 3),
                    "v_px": _round_float(v_px, 3),
                    "roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "std_luma": _round_float(float(roi.std()), 3),
                    "edge_density": _round_float(float((roi_gradient > 18.0).mean()), 6),
                }
            )
    if len(roi_metrics) < PROJECTED_ROBOT_MATERIAL_MIN_REVIEW_POINTS:
        result.update(
            {
                "status": "blocked",
                "projected_skeleton_trace_used": True,
                "projected_skeleton_trace_available": True,
                "review_quality_gate_available": False,
                "frame_index": row.get("frame_index"),
                "projected_robot_point_count": len(roi_metrics),
                "minimum_projected_robot_point_count": PROJECTED_ROBOT_MATERIAL_MIN_REVIEW_POINTS,
                "roi_radius_px": radius,
                "roi_metrics": roi_metrics,
                "blockers": ["projected_robot_material_quality_insufficient_projected_points"],
            }
        )
        return result
    mean_edge_density = sum(float(metric["edge_density"]) for metric in roi_metrics) / len(
        roi_metrics
    )
    mean_std_luma = sum(float(metric["std_luma"]) for metric in roi_metrics) / len(roi_metrics)
    blockers: list[str] = []
    if mean_edge_density < PROJECTED_ROBOT_MATERIAL_MIN_REVIEW_EDGE_DENSITY:
        blockers.append("source_policy_observation_projected_robot_material_low_detail")
    if mean_std_luma < PROJECTED_ROBOT_MATERIAL_MIN_REVIEW_STD_LUMA:
        blockers.append("source_policy_observation_projected_robot_material_flat")
    result.update(
        {
            "status": "passed" if not blockers else "failed",
            "projected_skeleton_trace_used": True,
            "projected_skeleton_trace_available": True,
            "review_quality_gate_available": True,
            "frame_index": row.get("frame_index"),
            "image_size_px": [int(width), int(height)],
            "trace_image_size_px": [_round_float(trace_width, 3), _round_float(trace_height, 3)],
            "roi_radius_px": radius,
            "projected_robot_point_count": len(roi_metrics),
            "minimum_projected_robot_point_count": PROJECTED_ROBOT_MATERIAL_MIN_REVIEW_POINTS,
            "mean_projected_robot_roi_edge_density": _round_float(mean_edge_density, 6),
            "minimum_mean_projected_robot_roi_edge_density": (
                PROJECTED_ROBOT_MATERIAL_MIN_REVIEW_EDGE_DENSITY
            ),
            "mean_projected_robot_roi_std_luma": _round_float(mean_std_luma, 3),
            "minimum_mean_projected_robot_roi_std_luma": (
                PROJECTED_ROBOT_MATERIAL_MIN_REVIEW_STD_LUMA
            ),
            "roi_metrics": roi_metrics,
            "blockers": blockers,
        }
    )
    return result


def _source_policy_observation_blockers(
    stats: Mapping[str, Any],
    *,
    target_object_id: str | None,
    review_quality_required: bool,
) -> list[str]:
    blockers: list[str] = []
    if stats.get("status") != "completed":
        return list(stats.get("blockers") or ["source_policy_observation_visual_probe_failed"])
    width = int(stats.get("width") or 0)
    height = int(stats.get("height") or 0)
    mean_luma = float(stats.get("mean_luma") or 0.0)
    std_luma = float(stats.get("std_luma") or 0.0)
    luma_range = float(stats.get("luma_range") or 0.0)
    dark_ratio = float(stats.get("dark_pixel_ratio") or 0.0)
    near_black_ratio = float(stats.get("near_black_pixel_ratio") or 0.0)
    entropy = float(stats.get("entropy_bits") or 0.0)
    edge_density = float(stats.get("edge_density") or 0.0)
    sharpness = float(stats.get("sharpness_laplacian_variance") or 0.0)
    center = stats.get("center_crop") if isinstance(stats.get("center_crop"), Mapping) else {}
    center_dark_ratio = float(center.get("dark_pixel_ratio") or 0.0)
    center_edge_density = float(center.get("edge_density") or 0.0)
    center_entropy = float(center.get("entropy_bits") or 0.0)
    if review_quality_required and (
        width < REVIEW_QUALITY_MIN_WIDTH or height < REVIEW_QUALITY_MIN_HEIGHT
    ):
        blockers.append("source_policy_observation_resolution_too_low_for_review_quality")
    if mean_luma < 38.0 or dark_ratio > 0.50 or near_black_ratio > 0.45:
        blockers.append("source_policy_observation_too_dark_for_review")
    if std_luma < 12.0 or luma_range < 50.0 or entropy < 2.5:
        blockers.append("source_policy_observation_flat_or_low_contrast")
    if edge_density < 0.012 or sharpness < 20.0:
        blockers.append("source_policy_observation_blurry_or_low_detail")
    if (
        review_quality_required
        and edge_density > 0.45
        and center_edge_density > 0.35
        and sharpness > 5000.0
    ):
        blockers.append("source_policy_observation_speckled_or_noisy_for_review_quality")
    if center_dark_ratio > 0.65 or center_edge_density < 0.004 or center_entropy < 1.8:
        blockers.append("source_policy_observation_task_region_low_information")
    if dark_ratio > 0.40 and edge_density < 0.015:
        blockers.append("source_policy_observation_mostly_wall_cabinet_counter_or_occlusion")
    if target_object_id and any(
        item in blockers
        for item in {
            "source_policy_observation_too_dark_for_review",
            "source_policy_observation_blurry_or_low_detail",
            "source_policy_observation_task_region_low_information",
            "source_policy_observation_mostly_wall_cabinet_counter_or_occlusion",
        }
    ):
        blockers.append("target_object_visibility_failed_visual_proxy")
    return sorted(set(blockers))


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def assess_source_policy_observation_visual_qa(
    frame_path: str | Path | None,
    *,
    generated_at: str,
    target_object_id: str | None = None,
    task_id: str | None = None,
    object_index: Mapping[str, Any] | Sequence[Any] | None = None,
    eval_ready_task_grounding: Mapping[str, Any] | None = None,
    semantic_artifact_base_dir: str | Path | None = None,
    projected_skeleton_trace_path: str | Path | None = None,
    visual_profile: str = "smoke",
    review_quality_required: bool = False,
) -> dict[str, Any]:
    """Assess the initial policy POV before WAM spend or rollout claims."""
    if frame_path is None:
        stats = {
            "status": "blocked",
            "path": None,
            "blockers": ["source_policy_observation_frame_missing"],
        }
    else:
        stats = _frame_visual_stats(frame_path, role="source_policy_observation", frame_index=0)
    blockers = _source_policy_observation_blockers(
        stats,
        target_object_id=target_object_id,
        review_quality_required=review_quality_required,
    )
    semantic_target_quality = _semantic_target_quality(
        frame_path=frame_path,
        stats=stats,
        target_object_id=target_object_id,
        task_id=task_id,
        object_index=object_index,
        eval_ready_task_grounding=eval_ready_task_grounding,
        semantic_artifact_base_dir=semantic_artifact_base_dir,
    )
    if semantic_target_quality.get("status") == "failed":
        blockers.extend(_string_list(semantic_target_quality.get("blockers")))
    projected_robot_material_quality = _projected_robot_material_quality(
        frame_path,
        projected_skeleton_trace_path=projected_skeleton_trace_path,
    )
    material_quality_enforced = bool(
        review_quality_required
        and projected_skeleton_trace_path is not None
        and _truthy_env(PROJECTED_ROBOT_MATERIAL_REVIEW_REQUIRE_ENV)
    )
    if material_quality_enforced:
        material_status = _string(projected_robot_material_quality.get("status"))
        if material_status in {"blocked", "failed"}:
            blockers.extend(_string_list(projected_robot_material_quality.get("blockers")))
    passed = bool(stats.get("status") == "completed" and not blockers)
    semantic_status = _string(semantic_target_quality.get("status"))
    return {
        "schema_version": SOURCE_POLICY_OBSERVATION_VISUAL_QA_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed_visual_quality_gate" if passed else "failed_visual_quality_gate",
        "visual_success": passed,
        "visual_profile": visual_profile,
        "review_quality_required": review_quality_required,
        "source_frame_path": str(Path(frame_path).expanduser()) if frame_path else None,
        "target_object_id": target_object_id,
        "task_id": task_id,
        "target_visibility_status": (
            "failed_semantic_gate"
            if semantic_status == "failed"
            else "passed_semantic_gate"
            if semantic_status == "passed" and passed
            else "not_proven"
            if semantic_status == "passed"
            else
            "failed_visual_proxy"
            if "target_object_visibility_failed_visual_proxy" in blockers
            else "not_declared"
            if not target_object_id
            else "passed_visual_proxy"
            if passed
            else "not_proven"
        ),
        "metrics": stats,
        "semantic_target_quality": semantic_target_quality,
        "projected_robot_material_quality": projected_robot_material_quality,
        "projected_robot_material_quality_enforced": material_quality_enforced,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "visual_qa_is_not_task_success_proof": True,
            "target_visibility_is_heuristic_without_detector": semantic_status == "not_available",
            "semantic_target_gates_are_initial_observation_quality_checks": True,
            "projected_robot_material_quality_is_heuristic": bool(
                projected_robot_material_quality.get("projected_skeleton_trace_used")
            ),
            "projected_robot_material_quality_is_advisory_unless_strict_env_enabled": (
                not material_quality_enforced
            ),
            "projected_skeleton_trace_is_support_evidence_not_robot_sensor_truth": bool(
                projected_robot_material_quality.get("projected_skeleton_trace_used")
            ),
            "synthetic_labels_are_support_evidence_not_raw_capture_truth": bool(
                semantic_target_quality.get("synthetic_label_evidence_used")
            ),
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "raw_secret_values_recorded": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def _rational_to_float(value: Any) -> float:
    text = str(value or "").strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            return float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _video_metadata_from_status(video_status: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = (
        video_status.get("ffprobe_metadata")
        if isinstance(video_status, Mapping)
        and isinstance(video_status.get("ffprobe_metadata"), Mapping)
        else {}
    )
    streams = metadata.get("streams") if isinstance(metadata.get("streams"), list) else []
    stream = streams[0] if streams and isinstance(streams[0], Mapping) else {}
    format_metadata = (
        metadata.get("format") if isinstance(metadata.get("format"), Mapping) else {}
    )
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "fps": _rational_to_float(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "nb_frames": int(stream.get("nb_frames") or 0) if str(stream.get("nb_frames") or "").isdigit() else 0,
        "duration_seconds": _round_float(
            stream.get("duration") or format_metadata.get("duration"),
            6,
        ),
        "size_bytes": int(format_metadata.get("size") or 0)
        if str(format_metadata.get("size") or "").isdigit()
        else 0,
    }


def _profile_quality_blockers(
    *,
    visual_profile: str,
    video_metadata: Mapping[str, Any],
    requested_settings: Mapping[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    requested = dict(requested_settings or {})
    width = int(video_metadata.get("width") or requested.get("width") or 0)
    height = int(video_metadata.get("height") or requested.get("height") or 0)
    fps = float(video_metadata.get("fps") or requested.get("fps") or 0.0)
    num_frames = int(video_metadata.get("nb_frames") or requested.get("num_frames") or 0)
    review_quality_profile = visual_profile == "review_quality"
    below_minimum = bool(
        width < REVIEW_QUALITY_MIN_WIDTH
        or height < REVIEW_QUALITY_MIN_HEIGHT
        or fps < REVIEW_QUALITY_MIN_FPS
        or num_frames < REVIEW_QUALITY_MIN_NUM_FRAMES
    )
    blockers: list[str] = []
    if review_quality_profile and below_minimum:
        blockers.append("review_quality_profile_media_below_minimum")
    profile_contract = {
        "visual_profile": visual_profile,
        "review_quality_profile": review_quality_profile,
        "review_quality_minimum": {
            "width": REVIEW_QUALITY_MIN_WIDTH,
            "height": REVIEW_QUALITY_MIN_HEIGHT,
            "fps": REVIEW_QUALITY_MIN_FPS,
            "num_frames": REVIEW_QUALITY_MIN_NUM_FRAMES,
        },
        "observed_or_requested": {
            "width": width,
            "height": height,
            "fps": _round_float(fps, 3),
            "num_frames": num_frames,
        },
        "review_quality_minimum_satisfied": not below_minimum,
        "smoke_only": bool(not review_quality_profile or below_minimum),
        "bounded_compromise_resolution_used": bool(
            review_quality_profile
            and not below_minimum
            and (width < 640 or height < 480 or fps < 15.0)
        ),
    }
    return blockers, profile_contract


def _generated_frame_quality_blockers(frame_stats: Sequence[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for stats in frame_stats:
        if stats.get("status") != "completed":
            blockers.extend(str(item) for item in stats.get("blockers") or [])
            continue
        if float(stats.get("mean_luma") or 0.0) < 35.0 or float(
            stats.get("dark_pixel_ratio") or 0.0
        ) > 0.55:
            blockers.append("wam_generated_frame_too_dark_for_review")
        if (
            float(stats.get("std_luma") or 0.0) < 10.0
            or float(stats.get("entropy_bits") or 0.0) < 2.4
            or float(stats.get("dominant_luma_bin_ratio") or 0.0) > 0.70
        ):
            blockers.append("wam_generated_frame_flat_or_low_detail")
        drift = stats.get("drift_from_source") if isinstance(stats.get("drift_from_source"), Mapping) else {}
        edge_ratio = drift.get("edge_density_ratio_to_source")
        entropy_delta = drift.get("entropy_delta_from_source")
        mean_delta = drift.get("mean_luma_delta_from_source")
        if edge_ratio is not None:
            parsed_edge_ratio = float(edge_ratio)
            if parsed_edge_ratio < 0.25:
                blockers.append("wam_generated_frame_edge_structure_drift")
            if parsed_edge_ratio > 2.0:
                blockers.append("wam_generated_frame_edge_structure_explosion")
        if entropy_delta is not None and float(entropy_delta) < -1.5:
            blockers.append("wam_generated_frame_entropy_drift")
        if mean_delta is not None and float(mean_delta) < -35.0:
            blockers.append("wam_generated_frame_darkening_drift")
    return sorted(set(blockers))


def _write_contact_sheet(
    *,
    frame_paths: Sequence[Path],
    output_path: Path,
    labels: Sequence[str],
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - dependency/environment edge.
        return {"status": "blocked", "blockers": [f"pillow_import_failed:{type(exc).__name__}"]}
    thumbnails = []
    for frame_path in frame_paths:
        try:
            with Image.open(frame_path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((220, 140))
                canvas = Image.new("RGB", (220, 160), (245, 245, 245))
                x = (220 - thumb.width) // 2
                y = 18 + (140 - thumb.height) // 2
                canvas.paste(thumb, (x, y))
                thumbnails.append(canvas)
        except Exception:
            continue
    if not thumbnails:
        return {"status": "blocked", "blockers": ["no_readable_frames_for_contact_sheet"]}
    columns = min(4, max(1, len(thumbnails)))
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 220, rows * 160), (235, 238, 241))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover - Pillow always provides default in normal envs.
        font = None
    for index, thumb in enumerate(thumbnails):
        x = (index % columns) * 220
        y = (index // columns) * 160
        sheet.paste(thumb, (x, y))
        label = labels[index] if index < len(labels) else f"frame {index}"
        draw.rectangle((x, y, x + 219, y + 17), fill=(28, 34, 42))
        draw.text((x + 6, y + 4), label[:36], fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return {
        "status": "completed",
        "path": str(output_path),
        "frame_count": len(thumbnails),
        "width": sheet.width,
        "height": sheet.height,
    }


def write_persistent_wam_visual_quality_artifacts(
    *,
    job_dir: str | Path,
    generated_at: str,
    source_frame_path: str | Path | None,
    generated_frame_paths: Sequence[str | Path],
    review_video_path: str | Path | None = None,
    video_status: Mapping[str, Any] | None = None,
    visual_profile: str = "smoke",
    requested_settings: Mapping[str, Any] | None = None,
    provider_status: str | None = None,
    live_wam_generation_success_count: int = 0,
    learned_wam_model_success_count: int = 0,
    structural_fallback_used: bool = False,
    target_object_id: str | None = None,
    task_id: str | None = None,
    projected_skeleton_trace_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write review-quality visual QA artifacts for a persistent policy/WAM rollout."""
    job = Path(job_dir).expanduser().resolve()
    job.mkdir(parents=True, exist_ok=True)
    normalized_profile = visual_profile if visual_profile in {"smoke", "review_quality"} else "smoke"
    source_qa = assess_source_policy_observation_visual_qa(
        source_frame_path,
        generated_at=generated_at,
        target_object_id=target_object_id,
        task_id=task_id,
        projected_skeleton_trace_path=projected_skeleton_trace_path,
        visual_profile=normalized_profile,
        review_quality_required=normalized_profile == "review_quality",
    )
    source_qa_path = job / "source_policy_observation_visual_qa.json"
    source_qa_path.write_text(json.dumps(source_qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_stats = source_qa.get("metrics") if isinstance(source_qa.get("metrics"), Mapping) else {}
    frame_paths = [Path(path).expanduser() for path in generated_frame_paths]
    frame_stats = [
        _frame_visual_stats(
            frame_path,
            role="wam_generated_next_observation",
            frame_index=index + 1,
            source_frame_stats=source_stats,
        )
        for index, frame_path in enumerate(frame_paths)
    ]
    frame_stats_path = job / "wam_rollout_frame_stats.jsonl"
    with frame_stats_path.open("w", encoding="utf-8") as handle:
        for row in frame_stats:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    contact_frame_paths = []
    contact_labels = []
    if source_frame_path:
        contact_frame_paths.append(Path(source_frame_path).expanduser())
        contact_labels.append("source")
    for index, frame_path in enumerate(frame_paths[:15], start=1):
        contact_frame_paths.append(frame_path)
        contact_labels.append(f"step {index}")
    contact_sheet_path = job / "wam_rollout_contact_sheet.jpg"
    contact_sheet = _write_contact_sheet(
        frame_paths=contact_frame_paths,
        output_path=contact_sheet_path,
        labels=contact_labels,
    )

    video_metadata = _video_metadata_from_status(video_status)
    if review_video_path:
        video_metadata["path"] = str(Path(review_video_path).expanduser())
    profile_blockers, profile_contract = _profile_quality_blockers(
        visual_profile=normalized_profile,
        video_metadata=video_metadata,
        requested_settings=requested_settings,
    )
    generated_blockers = _generated_frame_quality_blockers(frame_stats)
    source_blockers = list(source_qa.get("blockers") or [])
    blockers = sorted(set(source_blockers + generated_blockers + profile_blockers))
    generated_frames_pass = bool(frame_stats) and not generated_blockers
    source_pass = source_qa.get("status") == "passed_visual_quality_gate"
    provider_completed = provider_status == "completed"
    visual_success = bool(source_pass and generated_frames_pass and not profile_blockers)
    first_two_frame_blockers = _generated_frame_quality_blockers(frame_stats[:2])
    autoregressive_guard = {
        "autoregressive_chain_used": len(frame_stats) > 1,
        "generated_frame_count": len(frame_stats),
        "first_two_transition_visual_success": bool(frame_stats[:2] and not first_two_frame_blockers),
        "first_two_transition_blockers": first_two_frame_blockers,
        "periodic_reanchor_from_clean_render_used": False,
        "long_horizon_visual_drift_blocker": bool(len(frame_stats) > 2 and not visual_success),
        "long_rollout_should_not_be_overclaimed": bool(len(frame_stats) > 2 and not visual_success),
    }
    if autoregressive_guard["long_horizon_visual_drift_blocker"]:
        blockers.append("autoregressive_chain_visual_drift_or_quality_blocked_long_rollout")
        blockers = sorted(set(blockers))
    report = {
        "schema_version": PERSISTENT_WAM_VISUAL_QUALITY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed_visual_quality_gate" if visual_success else "failed_visual_quality_gate",
        "visual_success": visual_success,
        "visually_useful_rollout": visual_success,
        "visual_profile": normalized_profile,
        "profile_contract": profile_contract,
        "provider_status": provider_status,
        "provider_completed": provider_completed,
        "provider_completed_visual_quality_failed": bool(provider_completed and not visual_success),
        "live_wam_generation_success_count": int(live_wam_generation_success_count),
        "learned_wam_model_success_count": int(learned_wam_model_success_count),
        "live_wam_generation_success": bool(live_wam_generation_success_count > 0),
        "structural_fallback_used": bool(structural_fallback_used),
        "source_policy_observation_visual_qa_path": str(source_qa_path),
        "frame_stats_jsonl_path": str(frame_stats_path),
        "contact_sheet_path": str(contact_sheet_path) if contact_sheet_path.is_file() else None,
        "contact_sheet": contact_sheet,
        "review_video_path": str(Path(review_video_path).expanduser()) if review_video_path else None,
        "review_video_metadata": video_metadata,
        "generated_frame_count": len(frame_stats),
        "generated_frame_paths": [str(path) for path in frame_paths],
        "quality_summary": {
            "source_passed": source_pass,
            "source_mean_luma": source_stats.get("mean_luma"),
            "source_dark_pixel_ratio": source_stats.get("dark_pixel_ratio"),
            "source_edge_density": source_stats.get("edge_density"),
            "generated_frames_passed": generated_frames_pass,
            "minimum_generated_mean_luma": min(
                (float(row.get("mean_luma") or 0.0) for row in frame_stats),
                default=None,
            ),
            "maximum_generated_dark_pixel_ratio": max(
                (float(row.get("dark_pixel_ratio") or 0.0) for row in frame_stats),
                default=None,
            ),
        },
        "autoregressive_chain_guard": autoregressive_guard,
        "blockers": blockers,
        "claim_boundary": {
            "valid_mp4_or_provider_completed_is_not_visual_success": True,
            "live_wam_generation_success_can_coexist_with_visually_useful_rollout_false": True,
            "generated_observation_review_support_only": True,
            "review_quality_gate_is_not_scale_up_approval": True,
            "visual_quality_is_not_task_success_proof": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "accepted_anchor_manipulation_success_proven": False,
            "raw_secret_values_recorded": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    report_path = job / "wam_rollout_visual_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def visual_smoke_generated_rollouts_for_review(
    *,
    rollouts: Sequence[Mapping[str, Any]],
    output_dir: Path,
    generated_at: str,
    require_review_quality_profile: bool = True,
) -> dict[str, Any]:
    """Return a lightweight visual sanity check for generated rollout videos."""
    frame_dir = output_dir / "generated_rollout_frame_review" / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rollout_results: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - dependency/environment edge.
        return {
            "schema_version": VISUAL_SMOKE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked_visual_probe_failed",
            "blockers": [f"opencv_import_failed:{type(exc).__name__}"],
            "rollout_count": len(rollouts),
            "rollouts": [],
            "claim_boundary": {
                "valid_mp4_file_generated": bool(rollouts),
                "visual_rollout_useful_for_task_success_review": False,
                "generated_observation_review_support_only": True,
                "valid_media_artifact_is_not_task_success_review_evidence": True,
                "task_success_review_requires_visual_smoke_pass": True,
                "raw_secret_values_recorded": False,
                "secret_hashes_recorded": False,
            },
        }

    for rollout_index, rollout in enumerate(rollouts):
        video_path = _rollout_video_path(rollout)
        rollout_id = str(rollout.get("rollout_id") or f"rollout_{rollout_index + 1:04d}")
        result: dict[str, Any] = {
            "rollout_id": rollout_id,
            "generated_video_path": str(video_path) if video_path else None,
            "status": "blocked_visual_probe_failed",
            "sampled_frames": [],
            "visual_quality_flags": {
                "first_frame_preserves_source_scene": False,
                "later_frames_flat_or_dark": False,
                "success_review_not_reliable_from_this_rollout": True,
            },
        }
        if not video_path or not video_path.is_file():
            result["blockers"] = ["generated_video_missing_for_visual_smoke"]
            blockers.append("generated_video_missing_for_visual_smoke")
            rollout_results.append(result)
            continue
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            result["blockers"] = ["generated_video_unreadable_for_visual_smoke"]
            blockers.append("generated_video_unreadable_for_visual_smoke")
            rollout_results.append(result)
            continue
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            media_profile_blockers: list[str] = []
            if width < REVIEW_QUALITY_MIN_WIDTH or height < REVIEW_QUALITY_MIN_HEIGHT:
                media_profile_blockers.append(
                    "generated_rollout_video_resolution_too_low_for_task_success_review"
                )
            if fps < REVIEW_QUALITY_MIN_FPS:
                media_profile_blockers.append(
                    "generated_rollout_video_fps_too_low_for_task_success_review"
                )
            if frame_count < REVIEW_QUALITY_MIN_NUM_FRAMES:
                media_profile_blockers.append(
                    "generated_rollout_video_too_short_for_task_success_review"
                )
            media_profile_reviewable = not media_profile_blockers
            result["media_profile"] = {
                "width": width,
                "height": height,
                "fps": _round_float(fps, 3),
                "frame_count": frame_count,
                "review_quality_minimum": {
                    "width": REVIEW_QUALITY_MIN_WIDTH,
                    "height": REVIEW_QUALITY_MIN_HEIGHT,
                    "fps": REVIEW_QUALITY_MIN_FPS,
                    "num_frames": REVIEW_QUALITY_MIN_NUM_FRAMES,
                },
                "reviewable_for_task_success_evidence": media_profile_reviewable,
                "blockers": media_profile_blockers,
            }
            if frame_count <= 0:
                result["blockers"] = ["generated_video_frame_count_unavailable"]
                blockers.append("generated_video_frame_count_unavailable")
                rollout_results.append(result)
                continue
            sample_indices = sorted(
                {
                    0,
                    min(frame_count - 1, max(0, frame_count // 5)),
                    min(frame_count - 1, max(0, (frame_count * 2) // 5)),
                    min(frame_count - 1, max(0, (frame_count * 3) // 5)),
                    min(frame_count - 1, max(0, (frame_count * 4) // 5)),
                    frame_count - 1,
                }
            )
            samples: list[dict[str, Any]] = []
            safe_rollout_id = _safe_component(rollout_id)
            first_hist = None
            first_edge_density = 0.0
            first_entropy_bits = 0.0
            for sample_order, frame_index in enumerate(sample_indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                luma_min = int(gray.min())
                luma_max = int(gray.max())
                std_luma = float(gray.std())
                edges = cv2.Canny(gray, 50, 150)
                edge_density = float((edges > 0).mean())
                gray_hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
                gray_total = float(gray_hist.sum())
                gray_probabilities = [
                    float(value) / max(gray_total, 1.0)
                    for value in gray_hist.flatten()
                    if float(value) > 0.0
                ]
                entropy_bits = float(
                    -sum(probability * math.log2(probability) for probability in gray_probabilities)
                )
                hist = cv2.calcHist(
                    [frame],
                    [0, 1, 2],
                    None,
                    [8, 8, 8],
                    [0, 256, 0, 256, 0, 256],
                )
                cv2.normalize(hist, hist)
                if first_hist is None:
                    first_hist = hist
                    first_edge_density = edge_density
                    first_entropy_bits = entropy_bits
                hist_correlation_to_first = float(
                    cv2.compareHist(first_hist, hist, cv2.HISTCMP_CORREL)
                )
                edge_density_ratio_to_first = (
                    edge_density / first_edge_density
                    if first_edge_density > 0.0
                    else 0.0
                )
                sample_path = frame_dir / f"{safe_rollout_id}_frame_{sample_order:03d}.jpg"
                cv2.imwrite(str(sample_path), frame)
                samples.append(
                    {
                        "sample_index": sample_order,
                        "frame_index": int(frame_index),
                        "path": str(sample_path),
                        "mean_luma": round(float(gray.mean()), 3),
                        "std_luma": round(std_luma, 3),
                        "luma_min": luma_min,
                        "luma_max": luma_max,
                        "luma_range": luma_max - luma_min,
                        "entropy_bits": round(entropy_bits, 6),
                        "entropy_delta_to_first": round(
                            entropy_bits - first_entropy_bits,
                            6,
                        ),
                        "edge_density": round(edge_density, 6),
                        "edge_density_ratio_to_first": round(
                            edge_density_ratio_to_first,
                            6,
                        ),
                        "histogram_correlation_to_first": round(
                            hist_correlation_to_first,
                            6,
                        ),
                    }
                )
            if not samples:
                result["blockers"] = ["generated_video_frames_unreadable_for_visual_smoke"]
                blockers.append("generated_video_frames_unreadable_for_visual_smoke")
                rollout_results.append(result)
                continue
            first_preserves_scene = (
                samples[0]["luma_range"] >= 100
                and samples[0].get("edge_density", 0.0) >= 0.005
            )
            later_samples = samples[1:]
            later_flat_or_dark = bool(
                later_samples
                and all(sample["luma_range"] < 40 for sample in later_samples)
            )
            later_lost_scene_structure = bool(
                later_samples
                and all(
                    sample.get("edge_density_ratio_to_first", 0.0) < 0.10
                    and sample.get("histogram_correlation_to_first", 0.0) < 0.25
                    for sample in later_samples
                )
            )
            later_edge_structure_drift = bool(
                later_samples
                and any(
                    sample.get("edge_density_ratio_to_first", 0.0) < 0.25
                    and sample.get("edge_density", 0.0) < 0.01
                    for sample in later_samples
                )
            )
            later_entropy_drift = bool(
                later_samples
                and any(
                    float(sample.get("entropy_delta_to_first") or 0.0) < -1.5
                    for sample in later_samples
                )
            )
            later_static_or_noise_artifact = bool(
                later_samples
                and any(
                    sample.get("edge_density_ratio_to_first", 0.0) > 3.0
                    and sample.get("edge_density", 0.0) > 0.12
                    and sample.get("std_luma", 0.0) < 28.0
                    for sample in later_samples
                )
            )
            first_failed_future_sample = next(
                (
                    sample
                    for sample in later_samples
                    if (
                        sample.get("luma_range", 0) < 40
                        or sample.get("mean_luma", 0.0) < 35.0
                        or (
                            sample.get("edge_density_ratio_to_first", 0.0) < 0.10
                            and sample.get("histogram_correlation_to_first", 0.0) < 0.25
                        )
                        or (
                            sample.get("edge_density_ratio_to_first", 0.0) < 0.25
                            and sample.get("edge_density", 0.0) < 0.01
                        )
                        or float(sample.get("entropy_delta_to_first") or 0.0) < -1.5
                        or (
                            sample.get("edge_density_ratio_to_first", 0.0) > 3.0
                            and sample.get("edge_density", 0.0) > 0.12
                            and sample.get("std_luma", 0.0) < 28.0
                        )
                    )
                ),
                None,
            )
            immediate_future_collapse = bool(
                first_failed_future_sample
                and int(first_failed_future_sample.get("sample_index") or -1) == 1
            )
            future_quality_diagnostic = {
                "source_sample_frame_index": samples[0].get("frame_index"),
                "sampled_future_frame_count": len(later_samples),
                "first_failed_future_frame_index": (
                    first_failed_future_sample.get("frame_index")
                    if first_failed_future_sample
                    else None
                ),
                "first_failed_future_sample_index": (
                    first_failed_future_sample.get("sample_index")
                    if first_failed_future_sample
                    else None
                ),
                "minimum_future_mean_luma": (
                    min(float(sample.get("mean_luma") or 0.0) for sample in later_samples)
                    if later_samples
                    else None
                ),
                "minimum_future_edge_density_ratio_to_first": (
                    min(
                        float(sample.get("edge_density_ratio_to_first") or 0.0)
                        for sample in later_samples
                    )
                    if later_samples
                    else None
                ),
                "first_future_frame_collapsed": immediate_future_collapse,
                "diagnostic_label": (
                    "immediate_future_frame_collapse"
                    if immediate_future_collapse
                    else "future_frame_quality_degraded"
                    if first_failed_future_sample
                    else "future_frames_pass_sampled_signal_gates"
                ),
                "likely_debug_focus": (
                    [
                        "wam_runtime_input_contract",
                        "action_or_skeleton_conditioning",
                        "image_video_normalization_or_decoding",
                        "guidance_or_sampling_settings",
                    ]
                    if immediate_future_collapse
                    else []
                ),
                "diagnostic_only_not_success_label": True,
            }
            first_frame_not_scene_like = not first_preserves_scene
            quality_blockers = []
            if first_frame_not_scene_like:
                quality_blockers.append(
                    "generated_rollout_first_frame_not_scene_like"
                )
            if immediate_future_collapse:
                quality_blockers.append(
                    "generated_rollout_first_future_frame_collapsed"
                )
            if later_flat_or_dark:
                quality_blockers.append("generated_rollout_later_frames_flat_or_dark")
            if later_lost_scene_structure:
                quality_blockers.append(
                    "generated_rollout_later_frames_lost_scene_structure"
                )
            if later_edge_structure_drift:
                quality_blockers.append(
                    "generated_rollout_later_frames_edge_structure_drift"
                )
            if later_entropy_drift:
                quality_blockers.append("generated_rollout_later_frames_entropy_drift")
            if later_static_or_noise_artifact:
                quality_blockers.append(
                    "generated_rollout_later_frames_static_noise_artifact"
                )
            review_usefulness_blockers = list(media_profile_blockers)
            if require_review_quality_profile:
                quality_blockers.extend(media_profile_blockers)
            result.update(
                {
                    "status": "failed_visual_quality_smoke"
                    if quality_blockers
                    else "passed_visual_quality_smoke",
                    "frame_count": frame_count,
                    "sampled_frames": samples,
                    "future_frame_quality_diagnostic": future_quality_diagnostic,
                    "visual_quality_flags": {
                        "first_frame_preserves_source_scene": first_preserves_scene,
                        "first_future_frame_collapsed": immediate_future_collapse,
                        "later_frames_flat_or_dark": later_flat_or_dark,
                        "later_frames_lost_scene_structure": later_lost_scene_structure,
                        "later_frames_edge_structure_drift": later_edge_structure_drift,
                        "later_frames_entropy_drift": later_entropy_drift,
                        "later_frames_static_noise_artifact": later_static_or_noise_artifact,
                        "media_profile_reviewable_for_task_success": (
                            media_profile_reviewable
                        ),
                        "visual_rollout_useful_for_task_success_review": bool(
                            not quality_blockers and not review_usefulness_blockers
                        ),
                        "success_review_not_reliable_from_this_rollout": bool(
                            quality_blockers or review_usefulness_blockers
                        ),
                    },
                    "blockers": quality_blockers,
                    "review_usefulness_blockers": review_usefulness_blockers,
                }
            )
            blockers.extend(quality_blockers)
        finally:
            capture.release()
        rollout_results.append(result)

    smoke_passed = bool(rollout_results) and all(
        row.get("status") == "passed_visual_quality_smoke" for row in rollout_results
    )
    review_usefulness_blockers = sorted(
        {
            str(item)
            for row in rollout_results
            for item in row.get("review_usefulness_blockers", []) or []
            if str(item)
        }
    )
    useful = bool(smoke_passed) and all(
        row.get("visual_quality_flags", {}).get(
            "visual_rollout_useful_for_task_success_review"
        )
        is True
        for row in rollout_results
    )
    status = (
        "not_applicable_missing_rollouts"
        if not rollouts
        else "passed_visual_quality_smoke"
        if smoke_passed
        else "failed_visual_quality_smoke"
        if {
            "generated_rollout_first_frame_not_scene_like",
            "generated_rollout_first_future_frame_collapsed",
            "generated_rollout_later_frames_flat_or_dark",
            "generated_rollout_later_frames_lost_scene_structure",
            "generated_rollout_later_frames_edge_structure_drift",
            "generated_rollout_later_frames_entropy_drift",
            "generated_rollout_later_frames_static_noise_artifact",
            "generated_rollout_video_resolution_too_low_for_task_success_review",
            "generated_rollout_video_fps_too_low_for_task_success_review",
            "generated_rollout_video_too_short_for_task_success_review",
        }.intersection(blockers)
        else "blocked_visual_probe_failed"
    )
    return {
        "schema_version": VISUAL_SMOKE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "rollout_count": len(rollouts),
        "rollouts": rollout_results,
        "blockers": sorted(set(blockers)),
        "review_usefulness_status": (
            "reviewable_for_task_success"
            if useful
            else "not_reviewable_for_task_success"
        ),
        "review_usefulness_blockers": review_usefulness_blockers,
        "claim_boundary": {
            "valid_mp4_file_generated": bool(rollouts),
            "visual_rollout_useful_for_task_success_review": useful,
            "generated_observation_review_support_only": True,
            "valid_media_artifact_is_not_task_success_review_evidence": not useful,
            "task_success_review_requires_visual_smoke_pass": require_review_quality_profile,
            "review_quality_profile_required": require_review_quality_profile,
            "visual_smoke_is_not_forward_inverse_consistency": True,
            "raw_secret_values_recorded": False,
            "secret_hashes_recorded": False,
        },
    }
