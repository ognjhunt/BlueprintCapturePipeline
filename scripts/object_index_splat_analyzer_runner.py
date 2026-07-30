#!/usr/bin/env python3
"""Splat Analyzer adapter for the object-index stage.

This runner keeps Splat Analyzer behind the existing object-index backend
interface. It discovers a local .ply/.spz splat, runs an external Splat Analyzer
CLI when configured, and converts interactions.json into Blueprint object-index
objects. The output is model-derived support metadata, not capture truth or
robot-readiness proof.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SPLAT_SUFFIXES = {".ply", ".spz"}
DEFAULT_MAX_PROMPTS = 32
DEFAULT_QUALITY = "medium"
MIN_EXTENT_M = 0.02
MAX_PLY_HEADER_BYTES = 1024 * 1024
STANDARD_3DGS_VERTEX_PROPERTIES = {
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
}
CLAIM_BOUNDARY = {
    "artifact_purpose": "model_derived_3dgs_object_index_support",
    "source_is_gaussian_splat": True,
    "raw_capture_truth_preserved": True,
    "objects_are_model_derived_candidates": True,
    "box_center_depth_sampling_is_approximate": True,
    "depth_extent_is_inferred_not_observed": True,
    "object_orientation_is_not_estimated": True,
    "metric_oriented_box_validated": False,
    "input_axis_convention_validated": False,
    "relationships_are_advisory_candidates": True,
    "robot_spawn_validated": False,
    "collision_or_contact_validated": False,
    "articulation_state_validated": False,
    "simulator_execution_proven": False,
    "robot_policy_execution_proven": False,
    "physical_robot_readiness_proven": False,
    "public_claim_upgrade_allowed": False,
}
LABEL_BUCKETS = {
    "door": ("door",),
    "drawer": ("drawer",),
    "cabinet": ("cabinet", "cupboard", "closet", "wardrobe"),
    "fridge": ("fridge", "refrigerator"),
    "container": ("box", "container", "bin", "tote", "basket", "package", "crate"),
    "desk": ("desk", "table", "workstation"),
    "chair": ("chair", "stool"),
    "monitor": ("monitor", "tv", "screen"),
    "shelf": ("shelf", "rack"),
    "sink": ("sink", "faucet", "tap"),
    "handle": ("handle", "knob", "lever"),
}
INTERACTIVE_BUCKETS = {
    "door": ("door", 0.82),
    "drawer": ("drawer", 0.84),
    "cabinet": ("cabinet", 0.76),
    "fridge": ("refrigerator_door", 0.84),
    "handle": ("handle", 0.78),
}


def _read_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _slug(text: Any, *, fallback: str = "object") -> str:
    out = []
    for char in _string(text).lower():
        out.append(char if char.isalnum() else "_")
    value = "".join(out).strip("_")
    while "__" in value:
        value = value.replace("__", "_")
    return value or fallback


def _capture_root(payload: Mapping[str, Any]) -> Path:
    capture_root = _string(payload.get("capture_root"))
    if capture_root:
        return Path(capture_root).expanduser().resolve()
    raw_root = _string(payload.get("raw_root"))
    if raw_root:
        raw_path = Path(raw_root).expanduser().resolve()
        return raw_path.parent if raw_path.name == "raw" else raw_path
    return Path.cwd().resolve()


def _pipeline_root(capture_root: Path) -> Path:
    return capture_root / "pipeline"


def _resolve_candidate_path(value: Any, *, capture_root: Path, base_dirs: Sequence[Path]) -> Path | None:
    text = _string(value)
    if not text or text.startswith(("http://", "https://", "gs://", "s3://")):
        return None
    path = Path(text).expanduser()
    candidates = [path] if path.is_absolute() else [base / path for base in base_dirs]
    if not path.is_absolute():
        candidates.extend([capture_root / text, capture_root / "pipeline" / text, capture_root / "raw" / text])
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved.suffix.lower() in SPLAT_SUFFIXES:
            return resolved
    return None


def _append_asset(
    assets: List[Dict[str, Any]],
    seen: set[str],
    value: Any,
    *,
    capture_root: Path,
    base_dirs: Sequence[Path],
    source: str,
    kind: str = "",
    quality: str = "",
) -> None:
    path = _resolve_candidate_path(value, capture_root=capture_root, base_dirs=base_dirs)
    if path is None:
        return
    key = str(path)
    if key in seen:
        return
    seen.add(key)
    assets.append(
        {
            "path": key,
            "source": source,
            "kind": kind or path.suffix.lower().lstrip("."),
            "quality": quality,
            "suffix": path.suffix.lower(),
        }
    )


def _iter_mapping_paths(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, raw in value.items():
            if isinstance(raw, Mapping):
                yield str(key), raw.get("local_path") or raw.get("path") or raw.get("uri") or raw.get("url")
            else:
                yield str(key), raw
    elif isinstance(value, list):
        for index, raw in enumerate(value):
            if isinstance(raw, Mapping):
                yield (
                    str(raw.get("quality") or raw.get("name") or index),
                    raw.get("local_path") or raw.get("path") or raw.get("uri") or raw.get("url"),
                )
            else:
                yield str(index), raw
    elif isinstance(value, str):
        yield "default", value


def inspect_splat_analyzer_input(path: Path) -> Dict[str, Any]:
    """Classify whether upstream Splat Analyzer can directly load an exact local asset."""

    suffix = path.suffix.lower()
    if suffix == ".spz":
        return {
            "input_profile": "spz",
            "direct_splat_analyzer_compatible": True,
            "world_axis_convention_verified": False,
            "conversion_required": False,
        }
    if suffix != ".ply":
        return {
            "input_profile": "unsupported_splat_container",
            "direct_splat_analyzer_compatible": False,
            "world_axis_convention_verified": False,
            "conversion_required": True,
            "reason": "unsupported_splat_container",
        }
    try:
        with path.open("rb") as handle:
            header = handle.read(MAX_PLY_HEADER_BYTES + 1)
    except OSError as exc:
        return {
            "input_profile": "unreadable_ply",
            "direct_splat_analyzer_compatible": False,
            "world_axis_convention_verified": False,
            "conversion_required": True,
            "reason": f"ply_header_read_failed:{exc}",
        }
    marker = b"end_header"
    marker_index = header.find(marker)
    if marker_index < 0 or marker_index > MAX_PLY_HEADER_BYTES:
        return {
            "input_profile": "invalid_or_oversized_ply_header",
            "direct_splat_analyzer_compatible": False,
            "world_axis_convention_verified": False,
            "conversion_required": True,
            "reason": "ply_end_header_missing_within_limit",
        }
    header_text = header[: marker_index + len(marker)].decode("ascii", errors="replace")
    properties = {
        parts[2]
        for line in header_text.splitlines()
        if len(parts := line.strip().split()) == 3 and parts[0] == "property"
    }
    is_supersplat_compressed = {
        "packed_position",
        "packed_rotation",
        "packed_scale",
        "packed_color",
    }.issubset(properties)
    standard_missing = sorted(STANDARD_3DGS_VERTEX_PROPERTIES - properties)
    compatible = not standard_missing and not is_supersplat_compressed
    profile = (
        "supersplat_chunked_compressed_ply"
        if is_supersplat_compressed
        else "standard_3dgs_ply"
        if compatible
        else "unknown_or_incomplete_ply"
    )
    result: Dict[str, Any] = {
        "input_profile": profile,
        "direct_splat_analyzer_compatible": compatible,
        "world_axis_convention_verified": False,
        "conversion_required": not compatible,
        "missing_standard_properties": standard_missing,
    }
    if not compatible:
        result["reason"] = "splat_analyzer_requires_standard_uncompressed_3dgs_properties"
        result["required_next_step"] = (
            "create_hash_bound_standard_3dgs_ply_derivative_and_record_source_digest_"
            "conversion_runtime_digest_and_explicit_world_axis_transform"
        )
    return result


def discover_splat_assets(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return local .ply/.spz splat assets in priority order."""

    capture_root = _capture_root(payload)
    pipeline_root = _pipeline_root(capture_root)
    raw_root = (
        Path(_string(payload.get("raw_root"))).expanduser().resolve()
        if _string(payload.get("raw_root"))
        else capture_root / "raw"
    )
    base_dirs = [pipeline_root, pipeline_root / "worldlabs_assets", raw_root, capture_root]
    assets: List[Dict[str, Any]] = []
    seen: set[str] = set()

    _append_asset(
        assets,
        seen,
        os.getenv("SPLAT_ANALYZER_ASSET_PATH") or payload.get("splat_analyzer_asset_path"),
        capture_root=capture_root,
        base_dirs=base_dirs,
        source="explicit_splat_analyzer_asset_path",
    )

    materialized = _read_optional_mapping(pipeline_root / "worldlabs_assets" / "materialized_assets_manifest.json")
    for item in materialized.get("downloads", []) if isinstance(materialized.get("downloads"), list) else []:
        if not isinstance(item, Mapping):
            continue
        kind = _string(item.get("kind"))
        if kind not in {"splat_ply", "splat_spz"}:
            continue
        _append_asset(
            assets,
            seen,
            item.get("local_path") or item.get("relative_path"),
            capture_root=capture_root,
            base_dirs=base_dirs,
            source="worldlabs_materialized_assets_manifest",
            kind=kind,
            quality=_string(item.get("quality")),
        )

    export_manifest = _read_optional_mapping(pipeline_root / "worldlabs_export_manifest.json")
    for key, kind in (("ply_urls", "splat_ply"), ("spz_urls", "splat_spz")):
        for quality, value in _iter_mapping_paths(export_manifest.get(key)):
            _append_asset(
                assets,
                seen,
                value,
                capture_root=capture_root,
                base_dirs=base_dirs,
                source="worldlabs_export_manifest",
                kind=kind,
                quality=quality,
            )

    world_manifest = _read_optional_mapping(pipeline_root / "worldlabs_world_manifest.json")
    splats = _mapping(_mapping(world_manifest.get("assets")).get("splats"))
    for key, kind in (
        ("ply_urls", "splat_ply"),
        ("ply_url", "splat_ply"),
        ("spz_urls", "splat_spz"),
        ("spz_url", "splat_spz"),
    ):
        for quality, value in _iter_mapping_paths(splats.get(key)):
            _append_asset(
                assets,
                seen,
                value,
                capture_root=capture_root,
                base_dirs=base_dirs,
                source="worldlabs_world_manifest",
                kind=kind,
                quality=quality,
            )

    inventory = _read_optional_mapping(pipeline_root / "simulation_automation" / "scene_asset_inventory.json")
    inventory_items = inventory.get("assets") or inventory.get("files") or inventory.get("inventory")
    if isinstance(inventory_items, list):
        for item in inventory_items:
            if not isinstance(item, Mapping):
                continue
            value = item.get("local_path") or item.get("path") or item.get("asset_path")
            _append_asset(
                assets,
                seen,
                value,
                capture_root=capture_root,
                base_dirs=base_dirs,
                source="scene_asset_inventory",
                kind=_string(item.get("kind") or item.get("asset_type")),
                quality=_string(item.get("quality")),
            )

    for search_root in (pipeline_root / "worldlabs_assets", pipeline_root / "simready", raw_root):
        if not search_root.is_dir():
            continue
        for suffix in ("*.ply", "*.spz"):
            for path in sorted(search_root.rglob(suffix)):
                _append_asset(
                    assets,
                    seen,
                    str(path),
                    capture_root=capture_root,
                    base_dirs=base_dirs,
                    source="local_asset_scan",
                    kind=f"splat_{path.suffix.lower().lstrip('.')}",
                )

    return sorted(
        assets,
        key=lambda item: (
            0 if item.get("source") == "explicit_splat_analyzer_asset_path" else 1,
            0 if item.get("suffix") == ".ply" else 1,
        ),
    )


def _prompt_values(payload: Mapping[str, Any]) -> List[str]:
    override = _string(os.getenv("SPLAT_ANALYZER_PROMPT") or payload.get("splat_analyzer_prompt"))
    if override:
        raw_values = override.split(",")
    else:
        bank = _mapping(payload.get("prompt_bank"))
        raw_values = []
        for key in ("task_specific", "broad", "all"):
            values = bank.get(key)
            if isinstance(values, list):
                raw_values.extend(values)
    max_prompts = max(1, _safe_int(os.getenv("SPLAT_ANALYZER_MAX_PROMPTS"), DEFAULT_MAX_PROMPTS))
    out: List[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        text = " ".join(_string(raw).replace("/", " ").split())
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(text)
        if len(out) >= max_prompts:
            break
    return out


def _label_bucket(label: str) -> str:
    lowered = label.strip().lower()
    for bucket, tokens in LABEL_BUCKETS.items():
        if any(token in lowered for token in tokens):
            return bucket
    return lowered or "object"


def _task_text(payload: Mapping[str, Any]) -> str:
    descriptor = _mapping(payload.get("descriptor"))
    metadata = _mapping(descriptor.get("metadata"))
    task_zone = _mapping(metadata.get("task_zone"))
    parts = [
        metadata.get("task_statement"),
        metadata.get("workflow_context"),
        task_zone.get("label"),
        _mapping(payload.get("raw_manifest")).get("workflowName"),
        _mapping(payload.get("raw_manifest")).get("special_task_type"),
    ]
    return " ".join(_string(part).lower() for part in parts if _string(part))


def _task_relevance(label: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    bucket = _label_bucket(label)
    haystack = _task_text(payload)
    matched: List[str] = []
    for token in {label.lower(), bucket, *LABEL_BUCKETS.get(bucket, ())}:
        if token and token in haystack and token not in matched:
            matched.append(token)
    score = 0.25 + (0.45 if matched else 0.0)
    if bucket in {"door", "drawer", "cabinet", "fridge", "handle"} and any(
        token in haystack for token in ("open", "close", "turn", "pull", "push")
    ):
        score += 0.2
    if bucket in {"container", "shelf", "desk"} and any(
        token in haystack for token in ("pick", "place", "move", "organize", "inventory")
    ):
        score += 0.15
    return {
        "score": round(min(1.0, score), 4),
        "matched_terms": matched,
        "reason": "splat_analyzer_label_task_overlap" if matched else "splat_analyzer_scene_context",
    }


def _articulation_hints(label: str) -> Dict[str, Any]:
    bucket = _label_bucket(label)
    if bucket in INTERACTIVE_BUCKETS:
        kind, confidence = INTERACTIVE_BUCKETS[bucket]
        return {
            "interactive": True,
            "kind": kind,
            "confidence": confidence,
            "reason": "splat_analyzer_label_prior",
        }
    return {
        "interactive": False,
        "kind": "static_or_unproven",
        "confidence": 0.25,
        "reason": "no_interactive_label_prior",
    }


def _xyz_from_value(value: Any) -> List[float] | None:
    if isinstance(value, Mapping):
        return [_safe_float(value.get(axis), 0.0) for axis in ("x", "y", "z")]
    if isinstance(value, list) and len(value) >= 3:
        return [_safe_float(value[index], 0.0) for index in range(3)]
    return None


def _position(item: Mapping[str, Any]) -> List[float] | None:
    bbox = _mapping(item.get("boundingBox") or item.get("bbox3d") or item.get("obb"))
    return (
        _xyz_from_value(item.get("position"))
        or _xyz_from_value(item.get("center"))
        or _xyz_from_value(item.get("world_center"))
        or _xyz_from_value(bbox.get("center"))
    )


def _extents(item: Mapping[str, Any]) -> List[float]:
    bbox = _mapping(item.get("boundingBox") or item.get("bbox3d") or item.get("obb"))
    raw = (
        _xyz_from_value(item.get("size"))
        or _xyz_from_value(item.get("scale"))
        or _xyz_from_value(item.get("extents"))
        or _xyz_from_value(bbox.get("extents"))
        or [0.45, 0.45, 0.45]
    )
    return [round(max(MIN_EXTENT_M, abs(value)), 6) for value in raw[:3]]


def _frames(item: Mapping[str, Any]) -> List[Dict[str, Any]]:
    values = item.get("frames")
    frames = [dict(frame) for frame in values if isinstance(frame, Mapping)] if isinstance(values, list) else []
    if not frames and isinstance(item.get("frame_annotations"), list):
        frames = [dict(frame) for frame in item.get("frame_annotations") if isinstance(frame, Mapping)]
    return frames


def _frame_scores(frames: Sequence[Mapping[str, Any]], item: Mapping[str, Any]) -> List[float]:
    scores = []
    for frame in frames:
        score = max(_safe_float(frame.get("score"), -1.0), _safe_float(frame.get("confidence"), -1.0))
        if score >= 0.0:
            scores.append(score)
    item_score = max(
        _safe_float(item.get("score"), -1.0),
        _safe_float(item.get("confidence"), -1.0),
        _safe_float(item.get("mean_confidence"), -1.0),
    )
    if item_score >= 0.0:
        scores.append(item_score)
    return scores


def _mean_box_px(frames: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    widths = []
    heights = []
    areas = []
    for frame in frames:
        box = frame.get("box") or frame.get("bbox") or frame.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) < 4:
            continue
        width = max(0.0, _safe_float(box[2]) - _safe_float(box[0]))
        height = max(0.0, _safe_float(box[3]) - _safe_float(box[1]))
        widths.append(width)
        heights.append(height)
        areas.append(width * height)
    if not areas:
        return {"width": 0.0, "height": 0.0, "area": 0.0}
    return {
        "width": round(sum(widths) / len(widths), 4),
        "height": round(sum(heights) / len(heights), 4),
        "area": round(sum(areas) / len(areas), 4),
    }


def normalize_interactions(
    interactions_payload: Mapping[str, Any],
    *,
    input_payload: Mapping[str, Any],
    asset_record: Mapping[str, Any],
    interactions_path: Path,
    job_dir: Path,
) -> Dict[str, Any]:
    raw_objects = interactions_payload.get("objects")
    if not isinstance(raw_objects, list):
        raw_objects = []
    objects: List[Dict[str, Any]] = []
    seen_by_label: Dict[str, int] = {}
    for raw in raw_objects:
        if not isinstance(raw, Mapping):
            continue
        label = _string(raw.get("label") or raw.get("name") or raw.get("class_name") or "object") or "object"
        center = _position(raw)
        if center is None:
            continue
        bucket = _label_bucket(label)
        seen_by_label[bucket] = seen_by_label.get(bucket, 0) + 1
        object_id = _string(raw.get("id") or raw.get("object_id") or raw.get("instance_id"))
        if not object_id:
            object_id = f"splat_{_slug(bucket)}_{seen_by_label[bucket]:04d}"
        frames = _frames(raw)
        scores = _frame_scores(frames, raw)
        confidence = max(scores) if scores else 0.0
        mean_confidence = (sum(scores) / len(scores)) if scores else confidence
        frame_indices = []
        for frame in frames:
            frame_index = frame.get("frame_idx", frame.get("frame_index", frame.get("frameIndex")))
            frame_indices.append(_safe_int(frame_index, -1))
        frame_indices = [index for index in frame_indices if index >= 0]
        source_prompt = _string(raw.get("source_prompt") or raw.get("prompt") or label)
        objects.append(
            {
                "id": object_id,
                "object_id": object_id,
                "label": label,
                "name": label,
                "boundingBox": {
                    "center": [round(_safe_float(value), 6) for value in center[:3]],
                    "extents": _extents(raw),
                    "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "orientationQuaternion": [1.0, 0.0, 0.0, 0.0],
                    "kind": "rough_axis_aligned_interaction_volume_candidate",
                },
                "bounding_box_role": "visualization_only_rough_interaction_volume",
                "metric_placement_ready": False,
                "physics_ready": False,
                "geometry_source_class": "box_center_depth_proxy",
                "mean_confidence": round(max(0.0, min(1.0, mean_confidence)), 4),
                "confidence": round(max(0.0, min(1.0, confidence)), 4),
                "n_total_detections": max(1, len(frames)),
                "n_frame_detections": max(1, len(set(frame_indices))) if frames else 1,
                "evidence_frames": sorted(set(frame_indices)),
                "source_prompts": [source_prompt] if source_prompt else [label],
                "task_relevance": _task_relevance(label, input_payload),
                "articulation_hints": _articulation_hints(label),
                "mean_box_px": _mean_box_px(frames),
                "all_crops": [],
                "reference_crop": "",
                "merged_object_ids": [],
                "provenance": {
                    "source": "splat_analyzer",
                    "grounding_level": "model_derived_from_gaussian_splat",
                    "asset_path": asset_record.get("path"),
                    "asset_source": asset_record.get("source"),
                    "asset_kind": asset_record.get("kind"),
                    "interactions_path": str(interactions_path),
                    "job_dir": str(job_dir),
                    "observation_coverage": {
                        "frame_vote_count": len(frames),
                        "score_count": len(scores),
                    },
                    "canonical_truth": False,
                    "presentation_only": True,
                    "claim_boundary": dict(CLAIM_BOUNDARY),
                },
            }
        )
    relationships = build_relationship_candidates(objects)
    return {
        "objects": objects,
        "scene_relationship_candidates": relationships,
    }


def _center_extents(obj: Mapping[str, Any]) -> tuple[List[float], List[float]]:
    bbox = _mapping(obj.get("boundingBox"))
    center = [_safe_float(value) for value in bbox.get("center", [0.0, 0.0, 0.0])[:3]]
    extents = [_safe_float(value, 0.1) for value in bbox.get("extents", [0.1, 0.1, 0.1])[:3]]
    return center, extents


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))


def build_relationship_candidates(objects: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    relationships: List[Dict[str, Any]] = []
    for left_index, left in enumerate(objects):
        left_id = _string(left.get("object_id") or left.get("id"))
        left_center, left_extents = _center_extents(left)
        left_diag = max(0.25, math.sqrt(sum(value * value for value in left_extents)))
        for right in objects[left_index + 1 :]:
            right_id = _string(right.get("object_id") or right.get("id"))
            right_center, right_extents = _center_extents(right)
            dist = _distance(left_center, right_center)
            threshold = max(1.0, left_diag + math.sqrt(sum(value * value for value in right_extents)))
            if dist <= threshold:
                relationships.append(
                    {
                        "subject_id": left_id,
                        "object_id": right_id,
                        "relationship": "near",
                        "confidence": round(max(0.25, 1.0 - dist / max(threshold, 1e-6)), 4),
                        "source": "splat_analyzer_geometry_adjacency",
                        "review_required": True,
                        "claim_boundary": "relationship_is_model_derived_candidate_not_capture_truth",
                    }
                )
            vertical_gap = right_center[2] - left_center[2]
            horizontal_dist = math.sqrt(
                (right_center[0] - left_center[0]) ** 2 + (right_center[1] - left_center[1]) ** 2
            )
            support_radius = max(left_extents[0], left_extents[1], right_extents[0], right_extents[1])
            if (
                0.0 < vertical_gap <= max(0.75, left_extents[2] + right_extents[2])
                and horizontal_dist <= support_radius
            ):
                relationships.append(
                    {
                        "subject_id": right_id,
                        "object_id": left_id,
                        "relationship": "on_or_above_candidate",
                        "confidence": 0.45,
                        "source": "splat_analyzer_bbox_vertical_overlap",
                        "review_required": True,
                        "claim_boundary": "support_relationship_is_advisory_not_physics_or_contact_proof",
                    }
                )
    return relationships[:100]


def _template_command(template: str, *, splat_path: Path, prompt: str, job_dir: Path, output_json: Path) -> List[str]:
    replacements = {
        "SPLAT_PATH": shlex.quote(str(splat_path)),
        "PROMPT": shlex.quote(prompt),
        "JOB_DIR": shlex.quote(str(job_dir)),
        "OUTPUT_JSON": shlex.quote(str(output_json)),
    }
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{" + key + "}", value)
    return shlex.split(rendered)


def _configured_command(
    *,
    splat_path: Path,
    prompt: str,
    job_dir: Path,
    output_json: Path,
) -> tuple[List[str], Path | None, str]:
    template = _string(os.getenv("SPLAT_ANALYZER_COMMAND"))
    if template:
        return (
            _template_command(
                template,
                splat_path=splat_path,
                prompt=prompt,
                job_dir=job_dir,
                output_json=output_json,
            ),
            None,
            "template",
        )
    run_local = _string(os.getenv("SPLAT_ANALYZER_RUN_LOCAL"))
    repo = _string(os.getenv("SPLAT_ANALYZER_REPO"))
    run_local_path = Path(run_local).expanduser() if run_local else None
    if run_local_path is None and repo:
        run_local_path = Path(repo).expanduser() / "run_local.py"
    if run_local_path is not None and run_local_path.is_file():
        quality = _string(os.getenv("SPLAT_ANALYZER_QUALITY")) or DEFAULT_QUALITY
        return (
            [
                sys.executable,
                str(run_local_path.resolve()),
                "--ply",
                str(splat_path),
                "--prompt",
                prompt,
                "--quality",
                quality,
                "--job_dir",
                str(job_dir),
            ],
            run_local_path.resolve().parent,
            "run_local",
        )
    return [], None, "not_configured"


def _load_interactions_after_run(
    *,
    asset_record: Mapping[str, Any],
    prompt: str,
    job_dir: Path,
    output_json: Path,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    fixture = _string(os.getenv("SPLAT_ANALYZER_INTERACTIONS_JSON"))
    if fixture:
        fixture_path = Path(fixture).expanduser().resolve()
        if not fixture_path.is_file():
            return {}, {"backend_status": "failed", "reason": f"interactions_fixture_missing:{fixture_path}"}
        return _read_optional_mapping(fixture_path), {
            "backend_status": "ok",
            "execution_mode": "interactions_fixture",
            "interactions_path": str(fixture_path),
        }

    splat_path = Path(str(asset_record.get("path"))).resolve()
    command, cwd, mode = _configured_command(
        splat_path=splat_path,
        prompt=prompt,
        job_dir=job_dir,
        output_json=output_json,
    )
    if not command:
        return {}, {
            "backend_status": "skipped",
            "reason": "splat_analyzer_command_not_configured",
            "execution_mode": mode,
        }
    input_preflight = inspect_splat_analyzer_input(splat_path)
    if input_preflight.get("direct_splat_analyzer_compatible") is not True:
        return {}, {
            "backend_status": "skipped",
            "reason": "splat_analyzer_input_conversion_required",
            "execution_mode": "input_preflight",
            "input_preflight": input_preflight,
        }
    try:
        proc = subprocess.run(command, check=False, text=True, capture_output=True, cwd=str(cwd) if cwd else None)
    except OSError as exc:
        return {}, {
            "backend_status": "failed",
            "reason": f"splat_analyzer_failed_to_launch:{exc}",
            "command": command,
            "execution_mode": mode,
        }
    interactions_path = job_dir / "interactions.json"
    if proc.returncode != 0:
        return {}, {
            "backend_status": "failed",
            "reason": "splat_analyzer_command_failed",
            "return_code": proc.returncode,
            "command": command,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "execution_mode": mode,
        }
    if not interactions_path.is_file():
        return {}, {
            "backend_status": "failed",
            "reason": f"interactions_json_missing:{interactions_path}",
            "return_code": proc.returncode,
            "command": command,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "execution_mode": mode,
        }
    return _read_optional_mapping(interactions_path), {
        "backend_status": "ok",
        "return_code": proc.returncode,
        "command": command,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "execution_mode": mode,
        "interactions_path": str(interactions_path),
    }


def run_splat_analyzer_backend(payload: Mapping[str, Any], *, output_path: Path) -> Dict[str, Any]:
    capture_root = _capture_root(payload)
    output_dir = output_path.parent
    job_dir = output_dir / "splat_analyzer_job"
    job_dir.mkdir(parents=True, exist_ok=True)
    assets = discover_splat_assets(payload)
    if not assets:
        return {
            "objects": [],
            "scene_relationship_candidates": [],
            "backend_status": "skipped",
            "reason": "missing_local_splat_asset",
            "searched_capture_root": str(capture_root),
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    prompts = _prompt_values(payload)
    if not prompts:
        return {
            "objects": [],
            "scene_relationship_candidates": [],
            "backend_status": "skipped",
            "reason": "missing_splat_analyzer_prompts",
            "splat_asset": assets[0],
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    prompt = ", ".join(prompts)
    asset_record = dict(assets[0])
    asset_record["input_preflight"] = inspect_splat_analyzer_input(
        Path(str(asset_record.get("path"))).resolve()
    )
    interactions_payload, execution = _load_interactions_after_run(
        asset_record=asset_record,
        prompt=prompt,
        job_dir=job_dir,
        output_json=output_path,
    )
    if execution.get("backend_status") != "ok":
        return {
            "objects": [],
            "scene_relationship_candidates": [],
            "backend_status": execution.get("backend_status"),
            "reason": execution.get("reason"),
            "splat_asset": asset_record,
            "splat_asset_candidates": assets,
            "prompt": prompt,
            "execution": execution,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    interactions_path = Path(_string(execution.get("interactions_path")) or job_dir / "interactions.json")
    normalized = normalize_interactions(
        interactions_payload,
        input_payload=payload,
        asset_record=asset_record,
        interactions_path=interactions_path,
        job_dir=job_dir,
    )
    objects = normalized["objects"]
    return {
        "backend_status": "ok" if objects else "skipped",
        "reason": "" if objects else "no_splat_analyzer_objects",
        "backend_mode": "splat_analyzer",
        "objects": objects,
        "scene_relationship_candidates": normalized["scene_relationship_candidates"],
        "splat_asset": asset_record,
        "splat_asset_candidates": assets,
        "prompt": prompt,
        "prompt_count": len(prompts),
        "execution": execution,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def main(argv: List[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 2:
        print("usage: object_index_splat_analyzer_runner.py <input_json> <output_json>", file=sys.stderr)
        return 2
    input_path = Path(args[0])
    output_path = Path(args[1])
    payload = _read_payload(input_path)
    result = run_splat_analyzer_backend(payload, output_path=output_path)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
