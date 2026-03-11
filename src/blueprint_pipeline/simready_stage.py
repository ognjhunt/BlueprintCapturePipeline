"""Task-scoped SimReady workcell generation for Isaac-oriented local twins."""

from __future__ import annotations

import argparse
import math
import shutil
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .common import (
    PipelineError,
    ensure_dir,
    has_nonempty_file,
    optional_read_json,
    read_json,
    read_json_any,
    utc_now_iso,
    write_json,
    write_text,
)
from .local_capture import resolve_local_capture_context
from .object_geometry_stage import run_object_geometry_stage

_ARTICULATED_LABEL_TOKENS = {
    "door": "door",
    "drawer": "drawer",
    "cabinet": "cabinet",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
}
_SUPPORT_LABEL_TOKENS = {
    "table",
    "desk",
    "counter",
    "countertop",
    "shelf",
    "cabinet",
    "nightstand",
    "dresser",
    "island",
}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class SimReadyStageResult:
    capture_root: Path
    provider_name: str
    runtime: str
    scene_path: str
    manifest_path: str
    validation_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "v1",
            "capture_root": str(self.capture_root),
            "provider_name": self.provider_name,
            "runtime": self.runtime,
            "scene_path": self.scene_path,
            "manifest_path": self.manifest_path,
            "validation_path": self.validation_path,
        }


def _read_optional_json_any(path: Path) -> Any:
    if not path.is_file():
        return None
    return read_json_any(path)


def _string_list(*values: Any) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, (list, tuple, set)):
            items = [str(item) for item in value]
        elif value is None:
            items = []
        else:
            items = [str(value)]
        for item in items:
            text = item.strip()
            if text and text not in seen:
                out.append(text)
                seen.add(text)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_articulated_label(label: str) -> bool:
    lowered = label.strip().lower()
    return any(token in lowered for token in _ARTICULATED_LABEL_TOKENS)


def _articulation_type(label: str) -> str:
    lowered = label.strip().lower()
    for token, kind in _ARTICULATED_LABEL_TOKENS.items():
        if token in lowered:
            return kind
    return "fixed"


def _normalize_bbox(entry: Mapping[str, Any]) -> Dict[str, Any]:
    raw = entry.get("boundingBox")
    if not isinstance(raw, Mapping):
        raw = {}
    center_raw = raw.get("center") if isinstance(raw.get("center"), list) else []
    extents_raw = raw.get("extents") if isinstance(raw.get("extents"), list) else []
    axes_raw = raw.get("axes") if isinstance(raw.get("axes"), list) else []
    quat_raw = (
        raw.get("orientationQuaternion")
        if isinstance(raw.get("orientationQuaternion"), list)
        else []
    )
    center = [_safe_float(center_raw[idx] if idx < len(center_raw) else 0.0) for idx in range(3)]
    extents = [
        max(0.05, _safe_float(extents_raw[idx] if idx < len(extents_raw) else 0.25))
        for idx in range(3)
    ]
    axes: List[List[float]] = []
    for idx in range(3):
        row = axes_raw[idx] if idx < len(axes_raw) and isinstance(axes_raw[idx], list) else None
        if isinstance(row, list):
            axes.append([_safe_float(row[col] if col < len(row) else 0.0) for col in range(3)])
        else:
            axes.append([1.0 if idx == col else 0.0 for col in range(3)])
    quat = [_safe_float(quat_raw[idx] if idx < len(quat_raw) else (1.0 if idx == 0 else 0.0)) for idx in range(4)]
    return {
        "center": [round(value, 6) for value in center],
        "extents": [round(value, 6) for value in extents],
        "axes": axes,
        "orientationQuaternion": [round(value, 6) for value in quat],
    }


def _normalize_object_index(path: Path) -> List[Dict[str, Any]]:
    payload = _read_optional_json_any(path)
    if payload is None:
        return []
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, Mapping):
        entries = payload.get("objects") or payload.get("items") or payload.get("summaries") or []
    else:
        entries = []

    normalized: List[Dict[str, Any]] = []
    for raw in entries:
        if not isinstance(raw, Mapping):
            continue
        object_id = str(raw.get("id") or raw.get("object_id") or raw.get("instance_id") or "").strip()
        if not object_id:
            continue
        label = str(raw.get("label") or raw.get("name") or raw.get("class_name") or "object").strip() or "object"
        item = {
            "object_id": object_id,
            "label": label,
            "name": str(raw.get("name") or label),
            "boundingBox": _normalize_bbox(raw),
            "reference_crop": str(raw.get("reference_crop") or "").strip() or None,
            "all_crops": [
                str(value).strip()
                for value in raw.get("all_crops", [])
                if str(value).strip()
            ] if isinstance(raw.get("all_crops"), list) else [],
            "pointCloudFile": str(raw.get("pointCloudFile") or "").strip() or None,
            "source_entry": dict(raw),
        }
        normalized.append(item)
    return normalized


def _load_object_entries(capture_root: Path) -> List[Dict[str, Any]]:
    candidates = [
        capture_root / "raw" / "object_index.json",
        capture_root / "raw" / "arkit" / "objects" / "index.json",
    ]
    for path in candidates:
        entries = _normalize_object_index(path)
        if entries:
            return entries
    return []


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[idx]) - float(b[idx])) ** 2 for idx in range(3)))


def _xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
    )


def _select_support_object(
    *,
    target: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    target_bbox = target.get("placement_bbox") or target.get("boundingBox") or {}
    center = target_bbox["center"]
    extents = target_bbox["extents"]
    target_bottom = float(center[2]) - float(extents[2]) * 0.5
    candidates: List[tuple[float, str]] = []
    for other in objects:
        other_id = str(other.get("object_id") or "")
        if not other_id or other_id == str(target.get("object_id") or ""):
            continue
        other_bbox = other.get("placement_bbox") or other.get("boundingBox") or {}
        other_center = other_bbox["center"]
        other_extents = other_bbox["extents"]
        other_top = float(other_center[2]) + float(other_extents[2]) * 0.5
        gap = target_bottom - other_top
        if gap < -0.2 or gap > 0.75:
            continue
        if _xy_distance(center, other_center) > 1.75:
            continue
        label = str(other.get("label") or "").lower()
        if not any(token in label for token in _SUPPORT_LABEL_TOKENS):
            continue
        score = abs(gap) + 0.25 * _xy_distance(center, other_center)
        candidates.append((score, other_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _task_category(text: str) -> str:
    lowered = text.strip().lower()
    if "open and close" in lowered or lowered.startswith("open "):
        return "open_close"
    if "navigate to" in lowered or lowered.startswith("navigate "):
        return "navigate"
    if "pick up" in lowered or "place" in lowered:
        return "pick"
    return "generic"


def _resolve_real_crop_paths(entry: Mapping[str, Any], capture_root: Path) -> List[Path]:
    raw_values = _string_list(entry.get("reference_crop"), entry.get("all_crops"))
    out: List[Path] = []
    for value in raw_values:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (capture_root / candidate).resolve()
        if candidate.suffix.lower() in _IMAGE_EXTENSIONS and candidate.is_file() and candidate not in out:
            out.append(candidate)
    return out


def _synthetic_view_score(*, azimuth_deg: float, elevation_deg: float, bbox_diag: float) -> Dict[str, float]:
    visibility = max(0.45, 1.0 - min(abs(azimuth_deg - 180.0), 180.0) / 420.0)
    centered = max(0.55, 1.0 - abs(elevation_deg - 18.0) / 90.0)
    scale = min(0.98, 0.55 + bbox_diag / 4.0)
    occlusion = max(0.02, min(0.2, abs(azimuth_deg - 180.0) / 1000.0))
    hole_ratio = max(0.01, min(0.08, abs(elevation_deg - 18.0) / 550.0))
    task_relevance = 0.9 if azimuth_deg in {150.0, 180.0, 210.0} else 0.78
    total = round(
        visibility * 0.28
        + centered * 0.2
        + scale * 0.2
        + (1.0 - occlusion) * 0.17
        + (1.0 - hole_ratio) * 0.08
        + task_relevance * 0.07,
        4,
    )
    return {
        "visibility": round(visibility, 4),
        "centeredness": round(centered, 4),
        "scale": round(scale, 4),
        "occlusion": round(occlusion, 4),
        "hole_ratio": round(hole_ratio, 4),
        "task_relevance": round(task_relevance, 4),
        "total": total,
    }


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    payload = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + payload
        + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    )


def _write_simple_png(
    path: Path,
    *,
    width: int,
    height: int,
    background: tuple[int, int, int] = (24, 28, 34),
    box: Optional[tuple[int, int, int, int]] = None,
    box_fill: tuple[int, int, int] = (78, 132, 214),
    grayscale: Optional[int] = None,
) -> None:
    width = max(1, int(width))
    height = max(1, int(height))
    rows: List[bytes] = []
    if grayscale is not None:
        value = max(0, min(255, int(grayscale)))
        for y in range(height):
            row = bytearray()
            for x in range(width):
                inside = (
                    box is None
                    or (box[0] <= x < box[2] and box[1] <= y < box[3])
                )
                row.append(value if inside else 0)
            rows.append(b"\x00" + bytes(row))
        color_type = 0
        ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    else:
        for y in range(height):
            row = bytearray()
            for x in range(width):
                if box is not None and box[0] <= x < box[2] and box[1] <= y < box[3]:
                    pixel = box_fill
                else:
                    pixel = background
                row.extend(bytes((pixel[0], pixel[1], pixel[2])))
            rows.append(b"\x00" + bytes(row))
        color_type = 2
        ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    encoded = b"".join(rows)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(encoded))
        + _png_chunk(b"IEND", b"")
    )
    ensure_dir(path.parent)
    path.write_bytes(payload)


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    return 256, 256


def _write_real_mask(image_path: Path, mask_path: Path) -> None:
    width, height = _png_dimensions(image_path)
    _write_simple_png(mask_path, width=width, height=height, grayscale=255)


def _write_synthetic_view(
    image_path: Path,
    mask_path: Path,
    *,
    selected: bool,
    color: tuple[int, int, int],
) -> Dict[str, Any]:
    box = (52, 56, 204, 210)
    bg = (28, 40, 56) if selected else (24, 28, 34)
    _write_simple_png(image_path, width=256, height=256, background=bg, box=box, box_fill=color)
    _write_simple_png(mask_path, width=256, height=256, grayscale=255, box=box)
    return {"prompt_box": [52, 56, 204, 210]}


def _build_real_views(
    *,
    object_id: str,
    label: str,
    crop_paths: Sequence[Path],
    views_dir: Path,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for index, crop_path in enumerate(crop_paths[:4]):
        mask_path = views_dir / object_id / f"real_mask_{index:02d}.png"
        _write_real_mask(crop_path, mask_path)
        width, height = _png_dimensions(crop_path)
        score = round(max(0.62, 0.95 - index * 0.08), 4)
        candidates.append(
            {
                "view_id": f"{object_id}-real-{index:02d}",
                "source_mode": "real_capture",
                "image_path": str(crop_path),
                "crop_path": str(crop_path),
                "mask_path": str(mask_path),
                "prompt_box": [0, 0, width, height],
                "camera_pose": {"mode": "crop_reference"},
                "metrics": {
                    "visibility": score,
                    "centeredness": 0.92,
                    "scale": 0.95,
                    "occlusion": 0.04,
                    "hole_ratio": 0.0,
                    "task_relevance": 0.88,
                    "total": score,
                },
                "selected": True,
                "accepted_mask": True,
                "accepted_crop": True,
                "provenance": {"kind": "reference_crop", "object_id": object_id, "label": label},
            }
        )
    return {"source_mode": "real_capture", "candidates": candidates}


def _build_synthetic_views(
    *,
    entry: Mapping[str, Any],
    views_dir: Path,
) -> Dict[str, Any]:
    object_id = str(entry.get("object_id") or "")
    center = entry["boundingBox"]["center"]
    extents = entry["boundingBox"]["extents"]
    bbox_diag = math.sqrt(sum(float(value) ** 2 for value in extents))
    radius = max(0.6, min(2.5, bbox_diag * 1.8))
    candidates: List[Dict[str, Any]] = []
    specs: List[tuple[float, float]] = []
    for elevation in (12.0, 24.0):
        for azimuth in range(0, 360, 30):
            specs.append((float(azimuth), elevation))

    for index, (azimuth_deg, elevation_deg) in enumerate(specs):
        radians = math.radians(azimuth_deg)
        image_path = views_dir / object_id / f"synthetic_view_{index:02d}.png"
        mask_path = views_dir / object_id / f"synthetic_mask_{index:02d}.png"
        score = _synthetic_view_score(
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            bbox_diag=bbox_diag,
        )
        prompt = _write_synthetic_view(
            image_path,
            mask_path,
            selected=False,
            color=(78, 132, 214),
        )
        candidates.append(
            {
                "view_id": f"{object_id}-synthetic-{index:02d}",
                "source_mode": "synthetic_virtual",
                "image_path": str(image_path),
                "crop_path": str(image_path),
                "mask_path": str(mask_path),
                "prompt_box": prompt["prompt_box"],
                "camera_pose": {
                    "look_at": list(center),
                    "position": [
                        round(float(center[0]) + radius * math.cos(radians), 6),
                        round(float(center[1]) + radius * math.sin(radians), 6),
                        round(float(center[2]) + radius * math.sin(math.radians(elevation_deg)), 6),
                    ],
                    "azimuth_deg": azimuth_deg,
                    "elevation_deg": elevation_deg,
                    "radius_m": round(radius, 4),
                },
                "metrics": score,
                "selected": False,
                "accepted_mask": False,
                "accepted_crop": False,
                "provenance": {"kind": "synthetic_view", "object_id": object_id},
            }
        )

    ranked = sorted(candidates, key=lambda item: float(item["metrics"]["total"]), reverse=True)
    selected_ids = {item["view_id"] for item in ranked[:4]}
    for candidate in candidates:
        selected = candidate["view_id"] in selected_ids
        candidate["selected"] = selected
        candidate["accepted_mask"] = selected
        candidate["accepted_crop"] = selected
        if selected:
            _write_synthetic_view(
                Path(candidate["image_path"]),
                Path(candidate["mask_path"]),
                selected=True,
                color=(104, 188, 126),
            )
    return {"source_mode": "synthetic_virtual", "candidates": candidates}


def _layer_for_object(
    *,
    object_id: str,
    label: str,
    primary_ids: Sequence[str],
    articulation_ids: Sequence[str],
) -> str:
    if object_id in articulation_ids or _is_articulated_label(label):
        return "fixture_layer"
    if object_id in primary_ids:
        return "interactive_layer"
    return "physics_layer"


def _select_workcell_objects(
    *,
    objects: Sequence[Mapping[str, Any]],
    primary_ids: Sequence[str],
    articulation_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    by_id = {str(item.get("object_id")): dict(item) for item in objects}
    if not by_id:
        return []

    selected: set[str] = set(primary_ids) | set(articulation_ids)
    if not selected:
        selected = {str(item.get("object_id")) for item in list(objects)[: min(8, len(objects))]}

    primary_centers = []
    for object_id in primary_ids:
        if object_id not in by_id:
            continue
        bbox = by_id[object_id].get("placement_bbox") or by_id[object_id].get("boundingBox") or {}
        primary_centers.append(bbox["center"])
    if primary_centers:
        centroid = [
            sum(float(center[idx]) for center in primary_centers) / float(len(primary_centers))
            for idx in range(3)
        ]
    else:
        centroid = [0.0, 0.0, 0.0]

    for object_id in list(primary_ids):
        target = by_id.get(object_id)
        if not target:
            continue
        support_id = _select_support_object(target=target, objects=objects)
        if support_id:
            selected.add(support_id)

    for item in objects:
        object_id = str(item.get("object_id") or "")
        bbox = item.get("placement_bbox") or item.get("boundingBox") or {}
        center = bbox["center"]
        if object_id in selected:
            continue
        if _distance(center, centroid) <= 2.5:
            selected.add(object_id)

    ordered = [by_id[object_id] for object_id in primary_ids if object_id in by_id]
    ordered.extend([by_id[object_id] for object_id in articulation_ids if object_id in by_id and object_id not in primary_ids])
    ordered.extend(
        [
            by_id[object_id]
            for object_id in sorted(selected)
            if object_id in by_id and object_id not in primary_ids and object_id not in articulation_ids
        ]
    )
    return ordered


def _build_object_packets(
    *,
    objects: Sequence[Mapping[str, Any]],
    primary_ids: Sequence[str],
    articulation_ids: Sequence[str],
    task_text: str,
) -> List[Dict[str, Any]]:
    by_id = {str(item.get("object_id")): item for item in objects}
    packets: List[Dict[str, Any]] = []
    for item in objects:
        object_id = str(item.get("object_id") or "")
        label = str(item.get("label") or "object")
        layer = _layer_for_object(
            object_id=object_id,
            label=label,
            primary_ids=primary_ids,
            articulation_ids=articulation_ids,
        )
        bbox = item.get("placement_bbox") or item.get("boundingBox") or {}
        support_id = str(item.get("support_object_id") or "") or _select_support_object(target=item, objects=objects)
        nearby_fixtures = [
            other_id
            for other_id in articulation_ids
            if other_id in by_id and other_id != object_id and _distance(
                bbox["center"],
                (by_id[other_id].get("placement_bbox") or by_id[other_id].get("boundingBox"))["center"],
            ) <= 2.5
        ]
        nearby_context = [
            str(other.get("object_id") or "")
            for other in objects
            if str(other.get("object_id") or "") not in {object_id, *nearby_fixtures}
            and _distance(
                bbox["center"],
                (other.get("placement_bbox") or other.get("boundingBox"))["center"],
            ) <= 2.0
        ]
        packet = {
            "object_id": object_id,
            "label": label,
            "task_role": (
                "primary_target"
                if object_id in primary_ids
                else "required_fixture"
                if object_id in articulation_ids
                else "context_object"
            ),
            "task_category": _task_category(task_text),
            "layer": layer,
            "articulation_required": object_id in articulation_ids or _is_articulated_label(label),
            "articulation_type": _articulation_type(label),
            "boundingBox": bbox,
            "support_object_id": support_id,
            "nearby_articulated_fixture_ids": _string_list(item.get("nearby_articulated_fixture_ids"), nearby_fixtures),
            "nearby_context_ids": _string_list(item.get("nearby_context_ids"), nearby_context),
            "mesh_glb_path": str(item.get("mesh_glb_path") or "").strip() or None,
            "collision_hulls": [
                dict(hull) for hull in item.get("collision_hulls", []) if isinstance(hull, Mapping)
            ],
            "support_surfaces": [
                dict(surface) for surface in item.get("support_surfaces", []) if isinstance(surface, Mapping)
            ],
            "visual_replacement_masks": [
                dict(mask) for mask in item.get("visual_replacement_masks", []) if isinstance(mask, Mapping)
            ],
            "ai_hints": dict(item.get("ai_hints") or {}) if isinstance(item.get("ai_hints"), Mapping) else {},
            "material_physics_hints": {
                "graspable": object_id in primary_ids and not _is_articulated_label(label),
                "collision_kind": "decomposed_hulls" if item.get("collision_hulls") else "box_proxy",
                "dynamic": object_id in primary_ids and not _is_articulated_label(label),
                "support_surface_count": len(item.get("support_surfaces", [])) if isinstance(item.get("support_surfaces"), list) else 0,
            },
        }
        packets.append(packet)
    return packets


def _build_asset_requests(
    *,
    object_packets: Sequence[Mapping[str, Any]],
    object_views: Mapping[str, Mapping[str, Any]],
    provider_name: str,
) -> Dict[str, Any]:
    requests: List[Dict[str, Any]] = []
    for packet in object_packets:
        object_id = str(packet.get("object_id") or "")
        view_payload = object_views.get(object_id) or {}
        selected_views = [
            dict(item)
            for item in view_payload.get("selected_views", [])
            if isinstance(item, Mapping)
        ]
        requests.append(
            {
                "object_id": object_id,
                "label": str(packet.get("label") or "object"),
                "task_role": str(packet.get("task_role") or "context_object"),
                "provider_name": provider_name,
                "requested_mode": (
                    "functional_proxy"
                    if bool(packet.get("articulation_required"))
                    else "external_generation_with_proxy_fallback"
                ),
                "articulation_required": bool(packet.get("articulation_required")),
                "articulation_type": str(packet.get("articulation_type") or "fixed"),
                "dimensions_m": list(packet.get("boundingBox", {}).get("extents", [])),
                "selected_view_ids": [str(item.get("view_id") or "") for item in selected_views],
                "selected_view_images": [str(item.get("image_path") or "") for item in selected_views],
                "selected_view_masks": [str(item.get("mask_path") or "") for item in selected_views],
                "source_mesh_glb": str(packet.get("mesh_glb_path") or ""),
                "collision_hull_paths": [
                    str(item.get("path") or "")
                    for item in packet.get("collision_hulls", [])
                    if isinstance(item, Mapping)
                ],
                "support_surfaces": [
                    dict(item) for item in packet.get("support_surfaces", []) if isinstance(item, Mapping)
                ],
                "visual_replacement_masks": [
                    dict(item) for item in packet.get("visual_replacement_masks", []) if isinstance(item, Mapping)
                ],
                "ai_hints": dict(packet.get("ai_hints") or {}) if isinstance(packet.get("ai_hints"), Mapping) else {},
                "world_transform": {
                    "position": list(packet.get("boundingBox", {}).get("center", [])),
                    "orientation_quaternion": list(
                        packet.get("boundingBox", {}).get("orientationQuaternion", [])
                    ),
                },
            }
        )
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "provider_name": provider_name,
        "requests": requests,
    }


def _copy_reference_image(source_path: Path, target_path: Path) -> Optional[str]:
    if not source_path.is_file():
        return None
    ensure_dir(target_path.parent)
    shutil.copy2(source_path, target_path)
    return str(target_path)


def _write_reference_model_usd(path: Path, reference_rel_path: str) -> None:
    ref = reference_rel_path.replace("\\", "/")
    write_text(
        path,
        f'#usda 1.0\n(\n    defaultPrim = "Root"\n)\n\ndef Xform "Root" (\n    prepend references = @{ref}@\n)\n{{\n}}\n',
    )


def _usd_quat(quat: Sequence[float]) -> str:
    values = [_safe_float(quat[idx] if idx < len(quat) else (1.0 if idx == 0 else 0.0)) for idx in range(4)]
    return f"({values[0]:.6f}, {values[1]:.6f}, {values[2]:.6f}, {values[3]:.6f})"


def _write_proxy_asset(
    asset_dir: Path,
    *,
    packet: Mapping[str, Any],
    source_view: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    ensure_dir(asset_dir)
    model_path = asset_dir / "model.usda"
    metadata_path = asset_dir / "metadata.json"
    extents = packet["boundingBox"]["extents"]
    articulation_required = bool(packet.get("articulation_required"))
    articulation_type = str(packet.get("articulation_type") or "fixed")
    label = str(packet.get("label") or "object")
    joint_names = ["joint_main"] if articulation_required and articulation_type != "fixed" else []
    rigid_mode = "kinematic" if articulation_required else "dynamic"
    source_mesh_path = Path(str(packet.get("mesh_glb_path") or ""))
    asset_payload = {
        "schema_version": "v1",
        "object_id": str(packet.get("object_id") or ""),
        "label": label,
        "asset_kind": "functional_proxy" if articulation_required else "proxy_fallback",
        "articulation_required": articulation_required,
        "articulation_type": articulation_type,
        "joint_names": joint_names,
        "dimensions_m": list(extents),
        "rigid_mode": rigid_mode,
        "provider_status": "pending_external_generation" if not articulation_required else "not_requested",
        "source_view_id": str(source_view.get("view_id") or "") if isinstance(source_view, Mapping) else None,
        "collision_hulls": [
            dict(item) for item in packet.get("collision_hulls", []) if isinstance(item, Mapping)
        ],
        "support_surfaces": [
            dict(item) for item in packet.get("support_surfaces", []) if isinstance(item, Mapping)
        ],
    }
    write_json(metadata_path, asset_payload)
    size = [max(0.05, _safe_float(value)) for value in extents]
    joint_names_text = ", ".join(f'"{name}"' for name in joint_names)
    copied_hulls: List[str] = []
    for idx, item in enumerate(packet.get("collision_hulls", [])):
        if not isinstance(item, Mapping):
            continue
        hull_path = Path(str(item.get("path") or ""))
        if not hull_path.is_file():
            continue
        copied = asset_dir / "collision" / f"hull_{idx:02d}{hull_path.suffix.lower() or '.glb'}"
        ensure_dir(copied.parent)
        shutil.copy2(hull_path, copied)
        copied_hulls.append(str(copied))
    if source_mesh_path.is_file():
        mesh_target = asset_dir / "mesh.glb"
        shutil.copy2(source_mesh_path, mesh_target)
        _write_reference_model_usd(model_path, "./mesh.glb")
    else:
        write_text(
            model_path,
            "\n".join(
                [
                    "#usda 1.0",
                    "(",
                    '    defaultPrim = "Asset"',
                    ")",
                    "",
                    'def Xform "Asset" (',
                    "    customData = {",
                    f'        string label = "{label}"',
                    f'        string articulation_type = "{articulation_type}"',
                    f'        bool articulation_required = {"true" if articulation_required else "false"}',
                    f'        string rigid_mode = "{rigid_mode}"',
                    f"        string[] joint_names = [{joint_names_text}]",
                    "    }",
                    ")",
                    "{",
                    '    def Cube "Geometry"',
                    "    {",
                    "        double size = 1",
                    f"        double3 xformOp:scale = ({size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f})",
                    '        uniform token[] xformOpOrder = ["xformOp:scale"]',
                    "    }",
                    "}",
                    "",
                ]
            ),
        )
    reference_image = None
    if isinstance(source_view, Mapping):
        image_path = Path(str(source_view.get("crop_path") or source_view.get("image_path") or ""))
        if image_path.is_file():
            reference_image = _copy_reference_image(image_path, asset_dir / f"reference{image_path.suffix.lower() or '.png'}")
    return {
        "object_id": str(packet.get("object_id") or ""),
        "label": label,
        "asset_kind": asset_payload["asset_kind"],
        "asset_status": "ready",
        "provider_status": asset_payload["provider_status"],
        "provider_name": "functional_proxy" if articulation_required else "manual_stub",
        "articulation_required": articulation_required,
        "articulation_type": articulation_type,
        "joint_names": joint_names,
        "dimensions_m": list(extents),
        "collision_hull_paths": copied_hulls,
        "asset_dir": str(asset_dir),
        "asset_usd_path": str(model_path),
        "metadata_path": str(metadata_path),
        "reference_image_path": reference_image,
    }


def _build_assets(
    *,
    object_packets: Sequence[Mapping[str, Any]],
    object_views: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    assets_root = output_dir / "assets"
    ensure_dir(assets_root)
    assets: List[Dict[str, Any]] = []
    for packet in object_packets:
        object_id = str(packet.get("object_id") or "")
        view_payload = object_views.get(object_id) or {}
        selected_views = view_payload.get("selected_views", [])
        source_view = selected_views[0] if isinstance(selected_views, list) and selected_views else None
        asset_dir = assets_root / f"obj_{object_id}"
        assets.append(_write_proxy_asset(asset_dir, packet=packet, source_view=source_view))
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "assets": assets,
    }


def _detect_visual_source(capture_root: Path) -> Optional[str]:
    candidates = [
        capture_root / "raw" / "3dgs_compressed.ply",
        capture_root / "raw" / "gaussian_splat.ply",
        capture_root / "pipeline" / "advanced_geometry" / "3dgs_compressed.ply",
    ]
    for path in candidates:
        if has_nonempty_file(path):
            return str(path)
    return None


def _task_anchor(
    *,
    primary_objects: Sequence[Mapping[str, Any]],
    fallback_objects: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    source = primary_objects[0] if primary_objects else (fallback_objects[0] if fallback_objects else None)
    if not source:
        return {"start_zone": [0.0, 0.0, 0.0], "goal_zone": [0.0, 0.0, 0.0]}
    bbox = source.get("placement_bbox") or source.get("boundingBox") or {}
    center = bbox["center"]
    goal = [round(float(center[idx]), 6) for idx in range(3)]
    return {
        "start_zone": [round(goal[0] - 1.0, 6), round(goal[1], 6), round(goal[2], 6)],
        "goal_zone": goal,
    }


def _write_scene_usda(
    scene_path: Path,
    *,
    object_packets: Sequence[Mapping[str, Any]],
    assets_by_id: Mapping[str, Mapping[str, Any]],
    visual_source: Optional[str],
) -> None:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "SimReadyScene"',
        ")",
        "",
        'def Xform "SimReadyScene"',
        "{",
    ]
    if visual_source:
        lines.extend(
            [
                '    def Xform "VisualLayer" (',
                "        customData = {",
                f'            string source_path = "{visual_source}"',
                '            string layer = "visual_layer"',
                "        }",
                "    )",
                "    {",
                "    }",
                "",
            ]
        )
    lines.extend(
        [
            '    def Xform "PhysicsLayer" (',
            "        customData = {",
            '            string layer = "physics_layer"',
            "        }",
            "    )",
            "    {",
            "    }",
            "",
            '    def Xform "InteractiveLayer"',
            "    {",
        ]
    )
    for packet in object_packets:
        object_id = str(packet.get("object_id") or "")
        asset = assets_by_id.get(object_id) or {}
        asset_path = Path(str(asset.get("asset_usd_path") or ""))
        rel_path = asset_path.relative_to(scene_path.parent).as_posix() if asset_path.is_file() else ""
        center = packet["boundingBox"]["center"]
        quat = packet["boundingBox"]["orientationQuaternion"]
        lines.extend(
            [
                f'        def Xform "obj_{object_id}" (',
                f'            references = @{rel_path}@' if rel_path else "            kind = \"component\"",
                "        )",
                "        {",
                f"            double3 xformOp:translate = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})",
                f"            quatf xformOp:orient = {_usd_quat(quat)}",
                '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
                "        }",
            ]
        )
    lines.extend(["    }", "}"])
    write_text(scene_path, "\n".join(lines) + "\n")


def _build_validation(
    *,
    object_packets: Sequence[Mapping[str, Any]],
    assets_payload: Mapping[str, Any],
    scene_path: Path,
    task_text: str,
) -> Dict[str, Any]:
    assets = [
        dict(item) for item in assets_payload.get("assets", []) if isinstance(item, Mapping)
    ]
    assets_by_id = {str(item.get("object_id") or ""): item for item in assets}
    object_results: List[Dict[str, Any]] = []
    passed_objects = 0
    for packet in object_packets:
        object_id = str(packet.get("object_id") or "")
        asset = assets_by_id.get(object_id) or {}
        dims_source = packet["boundingBox"]["extents"]
        dims_asset = asset.get("dimensions_m") if isinstance(asset.get("dimensions_m"), list) else []
        max_delta = max(
            abs(_safe_float(dims_asset[idx] if idx < len(dims_asset) else 0.0) - float(dims_source[idx]))
            for idx in range(3)
        )
        articulation_required = bool(packet.get("articulation_required"))
        articulation_ok = (not articulation_required) or bool(asset.get("joint_names"))
        passed = bool(asset.get("asset_usd_path")) and articulation_ok and max_delta <= 0.1
        if passed:
            passed_objects += 1
        object_results.append(
            {
                "object_id": object_id,
                "passed": passed,
                "max_dimension_delta_m": round(max_delta, 6),
                "articulation_required": articulation_required,
                "articulation_ok": articulation_ok,
                "asset_usd_path": asset.get("asset_usd_path"),
            }
        )

    scene_ok = scene_path.is_file() and all(
        Path(str(item.get("asset_usd_path") or "")).is_file() for item in assets
    )
    category = _task_category(task_text)
    task_ok = scene_ok and passed_objects > 0
    task_detail = "scene and objects validated"
    if category == "pick":
        has_primary = any(str(item.get("task_role") or "") == "primary_target" for item in object_packets)
        has_support = any(str(item.get("support_object_id") or "") for item in object_packets)
        task_ok = task_ok and has_primary
        task_detail = "pick workcell has primary object"
        if not has_support:
            task_detail += "; no explicit support surface inferred"
    elif category == "open_close":
        has_fixture = any(bool(item.get("articulation_required")) for item in object_packets)
        task_ok = task_ok and has_fixture
        task_detail = "open/close workcell has articulated proxy"
    elif category == "navigate":
        task_detail = "navigate workcell has local scene geometry"

    overall_status = "passed" if scene_ok and task_ok else "degraded"
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "overall_status": overall_status,
        "scene_validation": {
            "passed": scene_ok,
            "scene_path": str(scene_path),
            "asset_reference_count": len(assets),
        },
        "object_validation": object_results,
        "task_validation": {
            "passed": task_ok,
            "category": category,
            "detail": task_detail,
        },
    }


def run_simready_stage(
    *,
    capture_root: str | Path,
    provider_name: str = "manual",
    runtime: str = "isaac_sim",
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    descriptor = read_json(context.descriptor_path)
    task_scope = optional_read_json(context.pipeline_root / "task_scope_record.json") or {}
    task_targets = optional_read_json(context.pipeline_root / "task_targets.json") or {}
    geometry_evidence = optional_read_json(context.pipeline_root / "geometry_evidence.json") or {}
    geometry_stage_result = run_object_geometry_stage(
        capture_root=context.capture_root,
        provider_name=provider_name,
    )
    geometry_manifest_path = Path(str(geometry_stage_result.get("manifest_path") or ""))
    geometry_manifest = read_json_any(geometry_manifest_path)
    geometry_objects = geometry_manifest.get("objects", []) if isinstance(geometry_manifest, Mapping) else []
    if not isinstance(geometry_objects, list) or not geometry_objects:
        raise PipelineError(f"SimReady stage requires object geometry artifacts at {geometry_manifest_path}")

    task_text = str(
        task_scope.get("task_statement")
        or descriptor.get("metadata", {}).get("task_statement")
        or context.capture_id
    ).strip()
    primary_ids = _string_list(task_scope.get("target_object_ids"), task_targets.get("target_object_ids"))
    articulation_ids = _string_list(
        task_scope.get("articulation_required_ids"),
        task_targets.get("articulation_required_ids"),
    )
    workcell_objects = _select_workcell_objects(objects=geometry_objects, primary_ids=primary_ids, articulation_ids=articulation_ids)
    object_packets = _build_object_packets(
        objects=workcell_objects,
        primary_ids=primary_ids,
        articulation_ids=articulation_ids,
        task_text=task_text,
    )

    output_dir = context.pipeline_root / "simready"
    ensure_dir(output_dir)
    object_views_by_id: Dict[str, Dict[str, Any]] = {
        str(item.get("object_id") or ""): {
            "object_id": str(item.get("object_id") or ""),
            "label": str(item.get("label") or "object"),
            "source_mode": str(item.get("source_mode") or "unknown"),
            "candidate_count": len(item.get("selected_views", [])) if isinstance(item.get("selected_views"), list) else 0,
            "selected_views": [dict(view) for view in item.get("selected_views", []) if isinstance(view, Mapping)],
            "candidates": [dict(view) for view in item.get("selected_views", []) if isinstance(view, Mapping)],
        }
        for item in workcell_objects
    }

    write_json(
        output_dir / "object_packets.json",
        {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "task_text": task_text,
            "objects": object_packets,
        },
    )
    write_json(
        output_dir / "simready_object_views.json",
        {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "task_text": task_text,
            "objects": list(object_views_by_id.values()),
        },
    )

    asset_requests = _build_asset_requests(
        object_packets=object_packets,
        object_views=object_views_by_id,
        provider_name=provider_name,
    )
    write_json(output_dir / "simready_asset_requests.json", asset_requests)

    assets_payload = _build_assets(
        object_packets=object_packets,
        object_views=object_views_by_id,
        output_dir=output_dir,
    )
    write_json(output_dir / "simready_assets.json", assets_payload)
    assets_by_id = {
        str(item.get("object_id") or ""): item
        for item in assets_payload.get("assets", [])
        if isinstance(item, Mapping)
    }

    primary_objects = [item for item in workcell_objects if str(item.get("object_id") or "") in primary_ids]
    anchors = _task_anchor(primary_objects=primary_objects, fallback_objects=workcell_objects)
    visual_source = _detect_visual_source(context.capture_root)
    scene_path = output_dir / "simready_scene.usda"
    _write_scene_usda(
        scene_path,
        object_packets=object_packets,
        assets_by_id=assets_by_id,
        visual_source=visual_source,
    )

    scene_manifest = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "runtime": runtime,
        "provider_name": provider_name,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "task_text": task_text,
        "task_category": _task_category(task_text),
        "geometry_manifest_path": str(geometry_manifest_path),
        "layers": {
            "visual_layer": {"source_path": visual_source, "mode": "splat_reference" if visual_source else "none"},
            "physics_layer": {
                "mode": "static_collision_shell",
                "navigation_clearance_envelope_m": geometry_evidence.get("measured_route_width_m"),
            },
            "interactive_layer": {
                "object_ids": [
                    str(item.get("object_id") or "")
                    for item in object_packets
                    if str(item.get("layer") or "") == "interactive_layer"
                ]
            },
            "fixture_layer": {
                "object_ids": [
                    str(item.get("object_id") or "")
                    for item in object_packets
                    if str(item.get("layer") or "") == "fixture_layer"
                ]
            },
        },
        "scene_path": str(scene_path),
        "task_start_zone": anchors["start_zone"],
        "task_goal_zone": anchors["goal_zone"],
        "workcell_object_count": len(object_packets),
        "assets_path": str(output_dir / "simready_assets.json"),
        "object_views_path": str(output_dir / "simready_object_views.json"),
        "object_packets_path": str(output_dir / "object_packets.json"),
    }
    write_json(output_dir / "simready_scene_manifest.json", scene_manifest)

    validation = _build_validation(
        object_packets=object_packets,
        assets_payload=assets_payload,
        scene_path=scene_path,
        task_text=task_text,
    )
    write_json(output_dir / "simready_validation.json", validation)

    result = SimReadyStageResult(
        capture_root=context.capture_root,
        provider_name=provider_name,
        runtime=runtime,
        scene_path=str(scene_path),
        manifest_path=str(output_dir / "simready_scene_manifest.json"),
        validation_path=str(output_dir / "simready_validation.json"),
    )
    return result.to_dict()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a task-scoped SimReady workcell from a capture")
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument("--provider", default="manual", help="Provider adapter name")
    parser.add_argument("--runtime", default="isaac_sim", help="Target runtime")
    args = parser.parse_args(argv)

    try:
        result = run_simready_stage(
            capture_root=args.capture_root,
            provider_name=args.provider,
            runtime=args.runtime,
        )
    except Exception as exc:
        print(f"[simready-stage] FAILED: {exc}")
        return 1

    print(f"[simready-stage] scene={result['scene_path']}")
    print(f"[simready-stage] manifest={result['manifest_path']}")
    print(f"[simready-stage] validation={result['validation_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
