"""Per-object geometry extraction for task-scoped SimReady workcells."""

from __future__ import annotations

import argparse
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .common import PipelineError, ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context

try:
    import numpy as np
except Exception:  # pragma: no cover - environment-dependent
    np = None  # type: ignore[assignment]

try:
    import trimesh
except Exception:  # pragma: no cover - environment-dependent
    trimesh = None  # type: ignore[assignment]

AIHintRunner = Callable[[Dict[str, Any]], Optional[Mapping[str, Any]]]


@dataclass(frozen=True)
class ObjectGeometryStageResult:
    capture_root: Path
    manifest_path: str
    object_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "v1",
            "capture_root": str(self.capture_root),
            "manifest_path": self.manifest_path,
            "object_count": self.object_count,
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
                seen.add(text)
                out.append(text)
    return out


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
                inside = box is None or (box[0] <= x < box[2] and box[1] <= y < box[3])
                row.append(value if inside else 0)
            rows.append(b"\x00" + bytes(row))
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    else:
        for y in range(height):
            row = bytearray()
            for x in range(width):
                pixel = box_fill if box is not None and box[0] <= x < box[2] and box[1] <= y < box[3] else background
                row.extend(bytes((pixel[0], pixel[1], pixel[2])))
            rows.append(b"\x00" + bytes(row))
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + _png_chunk(b"IEND", b"")
    )
    ensure_dir(path.parent)
    path.write_bytes(payload)


def _png_dimensions(path: Path) -> tuple[int, int]:
    try:
        data = path.read_bytes()
    except Exception:
        return 256, 256
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    return 256, 256


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
    quat = [
        _safe_float(quat_raw[idx] if idx < len(quat_raw) else (1.0 if idx == 0 else 0.0))
        for idx in range(4)
    ]
    return {
        "center": [round(value, 6) for value in center],
        "extents": [round(value, 6) for value in extents],
        "axes": axes,
        "orientationQuaternion": [round(value, 6) for value in quat],
    }


def _candidate_object_index_paths(capture_root: Path) -> List[Path]:
    return [
        capture_root / "raw" / "object_index.json",
        capture_root / "raw" / "arkit" / "objects" / "index.json",
    ]


def _load_object_entries(capture_root: Path) -> Tuple[List[Dict[str, Any]], Path]:
    for index_path in _candidate_object_index_paths(capture_root):
        if not index_path.is_file():
            continue
        payload = read_json_any(index_path)
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
            normalized.append(
                {
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
            )
        if normalized:
            return normalized, index_path
    return [], _candidate_object_index_paths(capture_root)[0]


def _resolve_object_file(path_text: Optional[str], *, index_path: Path, capture_root: Path) -> Optional[Path]:
    if not path_text:
        return None
    candidate = Path(str(path_text))
    candidates: List[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                (index_path.parent / candidate).resolve(),
                (capture_root / candidate).resolve(),
                (capture_root / "raw" / candidate).resolve(),
            ]
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _resolve_real_crop_paths(entry: Mapping[str, Any], *, capture_root: Path) -> List[Path]:
    out: List[Path] = []
    for value in _string_list(entry.get("reference_crop"), entry.get("all_crops")):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (capture_root / candidate).resolve()
        if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and candidate not in out:
            out.append(candidate)
    return out


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[idx]) - float(b[idx])) ** 2 for idx in range(3)))


def _quaternion_inverse(quat: Sequence[float]) -> List[float]:
    return [float(quat[0]), -float(quat[1]), -float(quat[2]), -float(quat[3])]


def _quaternion_matrix(quat: Sequence[float]) -> List[List[float]]:
    w, x, y, z = [float(quat[idx] if idx < len(quat) else (1.0 if idx == 0 else 0.0)) for idx in range(4)]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-8:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _rotate_points(points: Any, quat: Sequence[float]) -> Any:
    if np is None:
        raise PipelineError("numpy is required for object geometry processing")
    matrix = np.array(_quaternion_matrix(quat), dtype=float)
    return np.asarray(points, dtype=float) @ matrix.T


def _load_mesh_or_points(
    *,
    entry: Mapping[str, Any],
    index_path: Path,
    capture_root: Path,
) -> Tuple[Any, str]:
    if trimesh is None or np is None:
        return None, "geometry_lib_unavailable"

    resolved = _resolve_object_file(
        str(entry.get("pointCloudFile") or ""),
        index_path=index_path,
        capture_root=capture_root,
    )
    if resolved is None:
        return None, "missing_point_cloud"

    loaded = trimesh.load(str(resolved), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geom for geom in loaded.geometry.values() if geom is not None]
        if geometries:
            return trimesh.util.concatenate(geometries), "source_scene_mesh"
        return None, "empty_scene_mesh"
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy(), "source_mesh"

    loaded_pc = trimesh.load(str(resolved), process=False)
    vertices = getattr(loaded_pc, "vertices", None)
    if vertices is None or len(vertices) < 4:
        return None, "insufficient_point_cloud"

    points = np.asarray(vertices, dtype=float)
    extents = entry["boundingBox"]["extents"]
    pitch = max(0.04, min(extents) / 8.0)
    try:
        mesh = trimesh.voxel.ops.points_to_marching_cubes(points, pitch=pitch)
        return mesh, "point_cloud_marching_cubes"
    except Exception:
        try:
            mesh = trimesh.points.PointCloud(points).convex_hull
            return mesh, "point_cloud_convex_hull"
        except Exception:
            return None, "point_cloud_reconstruction_failed"


def _box_mesh_from_bbox(bbox: Mapping[str, Any]) -> Any:
    if trimesh is None:
        return None
    extents = [max(0.05, _safe_float(value)) for value in bbox.get("extents", [0.25, 0.25, 0.25])]
    return trimesh.creation.box(extents=extents)


def _normalize_mesh_to_local(mesh: Any, bbox: Mapping[str, Any]) -> Any:
    if trimesh is None or np is None:
        return mesh
    normalized = mesh.copy()
    center = np.asarray(bbox["center"], dtype=float)
    inv_quat = _quaternion_inverse(bbox.get("orientationQuaternion", [1.0, 0.0, 0.0, 0.0]))
    vertices = np.asarray(normalized.vertices, dtype=float) - center
    normalized.vertices = _rotate_points(vertices, inv_quat)
    return normalized


def _export_mesh(mesh: Any, path: Path) -> str:
    ensure_dir(path.parent)
    if mesh is None:
        path.write_bytes(b"glb")
        return str(path)
    mesh.export(path)
    return str(path)


def _mesh_bounds(mesh: Any) -> Dict[str, Any]:
    if trimesh is None or np is None or mesh is None or getattr(mesh, "bounds", None) is None:
        return {"center": [0.0, 0.0, 0.0], "extents": [0.25, 0.25, 0.25]}
    bounds = np.asarray(mesh.bounds, dtype=float)
    mins = bounds[0]
    maxs = bounds[1]
    center = ((mins + maxs) * 0.5).tolist()
    extents = (maxs - mins).tolist()
    return {
        "center": [round(float(value), 6) for value in center],
        "extents": [round(max(0.05, float(value)), 6) for value in extents],
    }


def _mesh_components(mesh: Any) -> List[Any]:
    if trimesh is None or mesh is None:
        return []
    try:
        components = list(mesh.split(only_watertight=False))
        return [component for component in components if getattr(component, "vertices", None) is not None and len(component.vertices) >= 4]
    except Exception:
        return []


def _kmeans2(points_xy: Any) -> Optional[Tuple[Any, Any]]:
    if np is None or len(points_xy) < 8:
        return None
    points = np.asarray(points_xy, dtype=float)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    axis = 0 if (maxs[0] - mins[0]) >= (maxs[1] - mins[1]) else 1
    order = np.argsort(points[:, axis])
    seed_a = points[order[0]]
    seed_b = points[order[-1]]
    if np.linalg.norm(seed_a - seed_b) < 0.1:
        return None
    centroids = np.vstack([seed_a, seed_b])
    for _ in range(6):
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        if labels.min() == labels.max():
            return None
        centroids = np.vstack([points[labels == idx].mean(axis=0) for idx in range(2)])
    return labels, centroids


def _collision_hull_meshes(mesh: Any) -> Tuple[List[Any], str]:
    if trimesh is None or np is None or mesh is None:
        return [], "geometry_unavailable"

    components = _mesh_components(mesh)
    if len(components) > 1:
        hulls: List[Any] = []
        for component in components[:4]:
            try:
                hulls.append(component.convex_hull)
            except Exception:
                hulls.append(component.bounding_box.to_mesh())
        return hulls, "component_convex_hulls"

    vertices = np.asarray(mesh.vertices, dtype=float)
    clustered = _kmeans2(vertices[:, :2]) if len(vertices) >= 20 else None
    if clustered is not None:
        labels, centroids = clustered
        hulls = []
        for idx in range(2):
            subset = vertices[labels == idx]
            if len(subset) < 8:
                continue
            try:
                hulls.append(trimesh.points.PointCloud(subset).convex_hull)
            except Exception:
                pass
        if len(hulls) >= 2:
            return hulls, "kmeans_convex_hulls"

    try:
        return [mesh.convex_hull], "single_convex_hull"
    except Exception:
        return [mesh.bounding_box.to_mesh()], "bounding_box_hull"


def _surface_polygon(points: Any) -> Dict[str, Any]:
    if np is None:
        return {"center": [0.0, 0.0, 0.0], "bounds_xy": [0.0, 0.0, 0.0, 0.0]}
    array = np.asarray(points, dtype=float)
    center = array.mean(axis=0)
    mins = array.min(axis=0)
    maxs = array.max(axis=0)
    return {
        "center": [round(float(value), 6) for value in center.tolist()],
        "bounds_xy": [round(float(mins[0]), 6), round(float(mins[1]), 6), round(float(maxs[0]), 6), round(float(maxs[1]), 6)],
    }


def _support_surfaces(mesh: Any, bbox: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    if trimesh is None or np is None or mesh is None or getattr(mesh, "faces", None) is None or len(mesh.faces) == 0:
        extents = bbox["extents"]
        center = bbox["center"]
        top_z = float(extents[2]) * 0.5
        return (
            [
                {
                    "surface_id": "top_plane",
                    "normal_local": [0.0, 0.0, 1.0],
                    "center_local": [0.0, 0.0, round(top_z, 6)],
                    "center_world": [round(float(center[0]), 6), round(float(center[1]), 6), round(float(center[2] + top_z), 6)],
                    "area_estimate_m2": round(float(extents[0]) * float(extents[1]), 6),
                    "support_score": 0.5,
                    "method": "bbox_fallback",
                }
            ],
            "bbox_fallback",
        )

    normals = np.asarray(mesh.face_normals, dtype=float)
    centers = np.asarray(mesh.triangles_center, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    top_mask = normals[:, 2] >= 0.85
    if not np.any(top_mask):
        return _support_surfaces(None, bbox)

    top_centers = centers[top_mask]
    top_areas = areas[top_mask]
    top_faces = np.asarray(mesh.faces, dtype=int)[top_mask]
    grouped: List[Dict[str, Any]] = []
    bins: List[float] = []
    for face_idx, center, area in zip(top_faces, top_centers, top_areas):
        z = float(center[2])
        matched_idx = None
        for idx, bin_z in enumerate(bins):
            if abs(bin_z - z) <= 0.08:
                matched_idx = idx
                break
        vertices = np.asarray(mesh.vertices[face_idx], dtype=float)
        if matched_idx is None:
            bins.append(z)
            grouped.append({"points": [vertices], "area": float(area)})
        else:
            grouped[matched_idx]["points"].append(vertices)
            grouped[matched_idx]["area"] += float(area)
    surfaces: List[Dict[str, Any]] = []
    world_center = bbox["center"]
    quat = bbox.get("orientationQuaternion", [1.0, 0.0, 0.0, 0.0])
    rotation = np.array(_quaternion_matrix(quat), dtype=float)
    for idx, group in enumerate(grouped):
        points = np.concatenate(group["points"], axis=0)
        centroid = points.mean(axis=0)
        centroid_world = (centroid @ rotation.T) + np.asarray(world_center, dtype=float)
        polygon = _surface_polygon(points)
        surfaces.append(
            {
                "surface_id": f"surface_{idx:02d}",
                "normal_local": [0.0, 0.0, 1.0],
                "center_local": [round(float(value), 6) for value in centroid.tolist()],
                "center_world": [round(float(value), 6) for value in centroid_world.tolist()],
                "area_estimate_m2": round(float(group["area"]), 6),
                "support_score": round(min(1.0, max(0.35, float(group["area"]) * 2.0)), 4),
                "bounds_xy_local": polygon["bounds_xy"],
                "method": "mesh_top_faces",
            }
        )
    surfaces.sort(key=lambda item: (-float(item["support_score"]), -float(item["area_estimate_m2"])))
    return surfaces[:4], "mesh_top_faces"


def _write_real_mask(image_path: Path, mask_path: Path) -> None:
    width, height = _png_dimensions(image_path)
    _write_simple_png(mask_path, width=width, height=height, grayscale=255)


def _build_real_views(
    *,
    object_id: str,
    crop_paths: Sequence[Path],
    views_dir: Path,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for index, crop_path in enumerate(crop_paths[:4]):
        mask_path = views_dir / f"real_mask_{index:02d}.png"
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
            }
        )
    return {"source_mode": "real_capture", "candidates": candidates}


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


def _build_synthetic_views(
    *,
    object_id: str,
    bbox: Mapping[str, Any],
    views_dir: Path,
) -> Dict[str, Any]:
    center = bbox["center"]
    extents = bbox["extents"]
    bbox_diag = math.sqrt(sum(float(value) ** 2 for value in extents))
    radius = max(0.6, min(2.5, bbox_diag * 1.8))
    candidates: List[Dict[str, Any]] = []
    specs: List[tuple[float, float]] = []
    for elevation in (12.0, 24.0):
        for azimuth in range(0, 360, 30):
            specs.append((float(azimuth), elevation))

    for index, (azimuth_deg, elevation_deg) in enumerate(specs):
        radians = math.radians(azimuth_deg)
        image_path = views_dir / f"synthetic_view_{index:02d}.png"
        mask_path = views_dir / f"synthetic_mask_{index:02d}.png"
        box = (52, 56, 204, 210)
        _write_simple_png(image_path, width=256, height=256, background=(24, 28, 34), box=box, box_fill=(78, 132, 214))
        _write_simple_png(mask_path, width=256, height=256, grayscale=255, box=box)
        score = _synthetic_view_score(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, bbox_diag=bbox_diag)
        candidates.append(
            {
                "view_id": f"{object_id}-synthetic-{index:02d}",
                "source_mode": "synthetic_virtual",
                "image_path": str(image_path),
                "crop_path": str(image_path),
                "mask_path": str(mask_path),
                "prompt_box": [52, 56, 204, 210],
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
            }
        )
    ranked = sorted(candidates, key=lambda item: float(item["metrics"]["total"]), reverse=True)
    selected_ids = {item["view_id"] for item in ranked[:4]}
    for candidate in candidates:
        if candidate["view_id"] in selected_ids:
            candidate["selected"] = True
            _write_simple_png(
                Path(candidate["image_path"]),
                width=256,
                height=256,
                background=(28, 40, 56),
                box=(52, 56, 204, 210),
                box_fill=(104, 188, 126),
            )
    return {"source_mode": "synthetic_virtual", "candidates": candidates}


def _support_link_for_target(
    *,
    target: Mapping[str, Any],
    other_objects: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    target_bbox = target["placement_bbox"]
    center = target_bbox["center"]
    bottom_z = float(center[2]) - float(target_bbox["extents"][2]) * 0.5
    best: Optional[tuple[float, str]] = None
    for other in other_objects:
        other_id = str(other.get("object_id") or "")
        if not other_id or other_id == str(target.get("object_id") or ""):
            continue
        for surface in other.get("support_surfaces", []):
            if not isinstance(surface, Mapping):
                continue
            surf_center = surface.get("center_world") if isinstance(surface.get("center_world"), list) else []
            if len(surf_center) < 3:
                continue
            dz = bottom_z - float(surf_center[2])
            if dz < -0.2 or dz > 0.75:
                continue
            planar = math.sqrt((float(center[0]) - float(surf_center[0])) ** 2 + (float(center[1]) - float(surf_center[1])) ** 2)
            if planar > 1.75:
                continue
            score = abs(dz) + 0.25 * planar
            if best is None or score < best[0]:
                best = (score, other_id)
    return best[1] if best else None


def _heuristic_ai_hints(
    *,
    entry: Mapping[str, Any],
    mesh_source: str,
    collision_count: int,
    support_count: int,
    selected_views: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    label = str(entry.get("label") or "object").lower()
    articulation_type = "hinge" if any(token in label for token in ("door", "cabinet", "refrigerator", "fridge")) else "slider" if "drawer" in label else "none"
    return {
        "source": "heuristic_fallback",
        "articulation_guess": articulation_type,
        "shape_complexity": "irregular" if collision_count > 1 else "simple",
        "support_surface_confidence": round(min(1.0, 0.35 + 0.2 * support_count), 4),
        "mesh_source": mesh_source,
        "selected_view_count": len(selected_views),
    }


def run_object_geometry_stage(
    *,
    capture_root: str | Path,
    provider_name: str = "manual",
    ai_hint_runner: Optional[AIHintRunner] = None,
) -> Dict[str, Any]:
    if np is None:
        raise PipelineError("numpy is required for object geometry processing")
    context = resolve_local_capture_context(capture_root)
    task_scope = read_json_any(context.pipeline_root / "task_scope_record.json") if (context.pipeline_root / "task_scope_record.json").is_file() else {}
    task_targets = read_json_any(context.pipeline_root / "task_targets.json") if (context.pipeline_root / "task_targets.json").is_file() else {}
    object_entries, index_path = _load_object_entries(context.capture_root)
    if not object_entries:
        raise PipelineError(f"Object geometry stage requires an object index under {context.raw_root}")

    primary_ids = _string_list(task_scope.get("target_object_ids"), task_targets.get("target_object_ids"))
    articulation_ids = _string_list(task_scope.get("articulation_required_ids"), task_targets.get("articulation_required_ids"))
    output_root = context.pipeline_root / "object_geometry"
    ensure_dir(output_root)

    geometry_objects: List[Dict[str, Any]] = []
    for entry in object_entries:
        object_id = str(entry.get("object_id") or "")
        object_dir = output_root / f"obj_{object_id}"
        views_dir = object_dir / "views"
        real_crops = _resolve_real_crop_paths(entry, capture_root=context.capture_root)
        view_payload = (
            _build_real_views(object_id=object_id, crop_paths=real_crops, views_dir=views_dir)
            if real_crops
            else _build_synthetic_views(object_id=object_id, bbox=entry["boundingBox"], views_dir=views_dir)
        )
        selected_views = [dict(item) for item in view_payload["candidates"] if bool(item.get("selected"))]
        mask_entries = [
            {
                "view_id": str(item.get("view_id") or ""),
                "mask_path": str(item.get("mask_path") or ""),
                "image_path": str(item.get("image_path") or ""),
                "source_mode": str(item.get("source_mode") or view_payload["source_mode"]),
            }
            for item in selected_views
        ]

        source_mesh, mesh_source = _load_mesh_or_points(entry=entry, index_path=index_path, capture_root=context.capture_root)
        if source_mesh is None:
            source_mesh = _box_mesh_from_bbox(entry["boundingBox"])
            mesh_source = "bbox_proxy_mesh"
        local_mesh = _normalize_mesh_to_local(source_mesh, entry["boundingBox"])
        mesh_glb_path = object_dir / "mesh.glb"
        mesh_path = _export_mesh(local_mesh, mesh_glb_path)
        mesh_bounds = _mesh_bounds(local_mesh)

        hull_meshes, hull_method = _collision_hull_meshes(local_mesh)
        hull_entries: List[Dict[str, Any]] = []
        for idx, hull_mesh in enumerate(hull_meshes):
            hull_path = object_dir / "collision_hulls" / f"hull_{idx:02d}.glb"
            _export_mesh(hull_mesh, hull_path)
            hull_entries.append(
                {
                    "hull_id": f"hull_{idx:02d}",
                    "path": str(hull_path),
                    "bounds_local": _mesh_bounds(hull_mesh),
                }
            )

        support_surfaces, support_method = _support_surfaces(local_mesh, entry["boundingBox"])
        ai_hints = _heuristic_ai_hints(
            entry=entry,
            mesh_source=mesh_source,
            collision_count=len(hull_entries),
            support_count=len(support_surfaces),
            selected_views=selected_views,
        )
        if ai_hint_runner is not None:
            try:
                custom_hints = ai_hint_runner(
                    {
                        "object_id": object_id,
                        "label": str(entry.get("label") or "object"),
                        "selected_views": selected_views,
                        "mesh_source": mesh_source,
                        "collision_hulls": hull_entries,
                        "support_surfaces": support_surfaces,
                    }
                )
            except Exception:
                custom_hints = None
            if isinstance(custom_hints, Mapping):
                ai_hints.update({str(key): value for key, value in custom_hints.items()})
                ai_hints["source"] = "ai_runner"

        geometry_record = {
            "object_id": object_id,
            "label": str(entry.get("label") or "object"),
            "task_role": (
                "primary_target"
                if object_id in primary_ids
                else "required_fixture"
                if object_id in articulation_ids
                else "context_object"
            ),
            "source_mode": view_payload["source_mode"],
            "provider_name": provider_name,
            "source_bbox": entry["boundingBox"],
            "placement_bbox": entry["boundingBox"],
            "mesh_glb_path": mesh_path,
            "mesh_source": mesh_source,
            "mesh_bounds_local": mesh_bounds,
            "collision_hulls": hull_entries,
            "collision_method": hull_method,
            "support_surfaces": support_surfaces,
            "support_method": support_method,
            "selected_views": selected_views,
            "visual_replacement_masks": mask_entries,
            "ai_hints": ai_hints,
        }
        write_json(object_dir / "support_surfaces.json", {"surfaces": support_surfaces, "method": support_method})
        write_json(object_dir / "ai_hints.json", ai_hints)
        geometry_objects.append(geometry_record)

    by_id = {str(item.get("object_id") or ""): item for item in geometry_objects}
    for item in geometry_objects:
        object_id = str(item.get("object_id") or "")
        item["nearby_articulated_fixture_ids"] = [
            other_id
            for other_id in articulation_ids
            if other_id in by_id and other_id != object_id and _distance(
                item["placement_bbox"]["center"],
                by_id[other_id]["placement_bbox"]["center"],
            ) <= 2.5
        ]
        item["nearby_context_ids"] = [
            str(other.get("object_id") or "")
            for other in geometry_objects
            if str(other.get("object_id") or "") != object_id
            and _distance(item["placement_bbox"]["center"], other["placement_bbox"]["center"]) <= 2.0
        ]
        item["support_object_id"] = _support_link_for_target(target=item, other_objects=geometry_objects)

    manifest_path = output_root / "object_geometry_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "provider_name": provider_name,
            "scene_id": context.scene_id,
            "capture_id": context.capture_id,
            "objects": geometry_objects,
        },
    )
    result = ObjectGeometryStageResult(
        capture_root=context.capture_root,
        manifest_path=str(manifest_path),
        object_count=len(geometry_objects),
    )
    return result.to_dict()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract per-object geometry packages for SimReady")
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument("--provider", default="manual", help="Provider adapter name")
    args = parser.parse_args(argv)

    try:
        result = run_object_geometry_stage(
            capture_root=args.capture_root,
            provider_name=args.provider,
        )
    except Exception as exc:
        print(f"[object-geometry-stage] FAILED: {exc}")
        return 1

    print(f"[object-geometry-stage] manifest={result['manifest_path']}")
    print(f"[object-geometry-stage] object_count={result['object_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
