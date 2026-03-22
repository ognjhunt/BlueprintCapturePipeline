"""Shared helpers for site-memory retrieval, alignment, and fusion."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), separators=(",", ":")) + "\n")


def p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(arr, 95))


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def pose_matrix(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.shape != (4, 4):
        return None
    return arr


def mat_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def transform_translation(T: Any) -> np.ndarray:
    arr = pose_matrix(T)
    if arr is None:
        return np.zeros(3, dtype=np.float64)
    return arr[:3, 3]


def effective_pose(record: Mapping[str, Any]) -> Optional[np.ndarray]:
    direct_site = pose_matrix(record.get("T_site_camera"))
    if direct_site is not None:
        return direct_site
    raw = pose_matrix(record.get("T_world_camera"))
    if raw is None:
        return None
    site_xform = pose_matrix(record.get("site_frame_transform"))
    if site_xform is None:
        return raw
    return site_xform @ raw


def pose_distance_m(a: Any, b: Any) -> float:
    ta = transform_translation(a)
    tb = transform_translation(b)
    return float(np.linalg.norm(ta - tb))


def rotation_cosine(T_a: np.ndarray, T_b: np.ndarray) -> float:
    cos_angle = float((np.trace(T_a[:3, :3].T @ T_b[:3, :3]) - 1.0) / 2.0)
    return float(np.clip(cos_angle, -1.0, 1.0))


def gs_uri_to_local(uri: str, *, storage_root: Optional[Path]) -> Optional[Path]:
    if not uri:
        return None
    if not uri.startswith("gs://"):
        return Path(uri).expanduser()
    if storage_root is None:
        return None
    trimmed = uri[5:]
    bucket, _, key = trimmed.partition("/")
    if not bucket or not key:
        return None
    return storage_root / bucket / key


def load_embedding(
    *,
    embedding_uri: str,
    storage_root: Optional[Path],
    expected_dim: int = 1024,
) -> Optional[np.ndarray]:
    local = gs_uri_to_local(embedding_uri, storage_root=storage_root)
    if local is None or not local.is_file():
        return None
    try:
        vec = np.fromfile(str(local), dtype=np.float32)
    except Exception:
        return None
    if vec.shape[0] != expected_dim:
        return None
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return vec / norm


def _load_png_array(path: Path) -> Optional[np.ndarray]:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            arr = np.asarray(image)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def load_numeric_array(
    path_value: Any,
    *,
    storage_root: Optional[Path] = None,
) -> Optional[np.ndarray]:
    text = str(path_value or "").strip()
    if not text:
        return None
    path = gs_uri_to_local(text, storage_root=storage_root)
    if path is None or not path.is_file():
        return None
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            return np.asarray(np.load(path), dtype=np.float32)
        if suffix == ".png":
            arr = _load_png_array(path)
            if arr is None:
                return None
            arr = np.asarray(arr)
            if arr.dtype == np.uint16 or float(arr.max()) > 255.0:
                out = arr.astype(np.float32) * 0.001
                out[arr == 0] = 0.0
                return out
            return arr.astype(np.float32)
    except Exception:
        return None
    return None


def geometry_fingerprint(
    *,
    depth_path: Any,
    confidence_path: Any,
    storage_root: Optional[Path] = None,
    intrinsics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    depth = load_numeric_array(depth_path, storage_root=storage_root)
    confidence = load_numeric_array(confidence_path, storage_root=storage_root)
    if depth is None:
        return {
            "available": False,
            "representation": "none",
        }

    depth = np.asarray(depth, dtype=np.float32)
    valid_mask = np.isfinite(depth) & (depth > 0.0)
    valid = depth[valid_mask]
    if valid.size == 0:
        return {
            "available": False,
            "representation": "depth_unusable",
        }

    lo = float(np.percentile(valid, 5))
    hi = float(np.percentile(valid, 95))
    if hi <= lo:
        hi = lo + 1e-3
    hist, _ = np.histogram(valid, bins=8, range=(lo, hi))
    hist_sum = int(hist.sum()) or 1
    hist_norm = [round(float(item) / hist_sum, 6) for item in hist.tolist()]
    median_depth = float(np.median(valid))
    depth_std = float(np.std(valid))
    plane_support = float(np.mean(np.abs(valid - median_depth) <= 0.15))

    result: Dict[str, Any] = {
        "available": True,
        "representation": "depth_histogram_v1",
        "valid_fraction": round(float(valid.size) / float(depth.size or 1), 6),
        "min_depth_m": round(float(valid.min()), 4),
        "max_depth_m": round(float(valid.max()), 4),
        "mean_depth_m": round(float(valid.mean()), 4),
        "median_depth_m": round(median_depth, 4),
        "std_depth_m": round(depth_std, 4),
        "free_space_extent_m": round(float(np.percentile(valid, 90)), 4),
        "depth_histogram_8": hist_norm,
        "plane_support_ratio": round(plane_support, 4),
        "surface_complexity": round(min(depth_std / max(median_depth, 1e-3), 4.0), 4),
    }

    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32)
        if confidence.shape == depth.shape:
            conf_values = confidence[valid_mask]
            if conf_values.size > 0:
                result["confidence_mean"] = round(float(np.mean(conf_values)), 4)
                result["confidence_high_fraction"] = round(float(np.mean(conf_values >= 0.75)), 4)
    if intrinsics:
        fx = intrinsics.get("fx")
        fy = intrinsics.get("fy")
        if fx and fy:
            result["projective_scale"] = {
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(intrinsics.get("cx") or 0.0),
                "cy": float(intrinsics.get("cy") or 0.0),
            }
    return result


def fingerprint_similarity(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    if not a or not b:
        return 0.0
    hist_a = np.asarray(list(a.get("depth_histogram_8") or []), dtype=np.float64)
    hist_b = np.asarray(list(b.get("depth_histogram_8") or []), dtype=np.float64)
    histogram_score = 0.0
    if hist_a.size == hist_b.size and hist_a.size > 0:
        diff = np.abs(hist_a - hist_b).sum()
        histogram_score = max(0.0, 1.0 - float(diff) / 2.0)
    median_a = float(a.get("median_depth_m") or 0.0)
    median_b = float(b.get("median_depth_m") or 0.0)
    median_score = max(0.0, 1.0 - abs(median_a - median_b) / 4.0)
    complexity_a = float(a.get("surface_complexity") or 0.0)
    complexity_b = float(b.get("surface_complexity") or 0.0)
    complexity_score = max(0.0, 1.0 - abs(complexity_a - complexity_b))
    plane_a = float(a.get("plane_support_ratio") or 0.0)
    plane_b = float(b.get("plane_support_ratio") or 0.0)
    plane_score = max(0.0, 1.0 - abs(plane_a - plane_b))
    return round(
        (0.45 * histogram_score) + (0.25 * median_score) + (0.15 * complexity_score) + (0.15 * plane_score),
        4,
    )


def visibility_cells_from_record(
    record: Mapping[str, Any],
    *,
    cell_size_m: float = 0.5,
) -> List[str]:
    T = effective_pose(record)
    if T is None:
        return []
    origin = T[:3, 3]
    forward = T[:3, 2]
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        return []
    forward = forward / norm
    geometry = record.get("geometry_fingerprint")
    if not isinstance(geometry, Mapping):
        geometry = {}
    extent = float(geometry.get("free_space_extent_m") or 2.0)
    steps = max(2, min(8, int(math.ceil(extent / max(cell_size_m, 0.1)))))
    cells: List[str] = []
    for step in range(1, steps + 1):
        point = origin + forward * (step * cell_size_m)
        cell_x = int(math.floor(point[0] / cell_size_m))
        cell_z = int(math.floor(point[2] / cell_size_m))
        key = f"{cell_x},{cell_z}"
        if key not in cells:
            cells.append(key)
    return cells


def backproject_depth_points(
    *,
    depth: np.ndarray,
    intrinsics: Mapping[str, Any],
    T_world_camera: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    sample_step: int = 16,
    min_confidence: float = 0.5,
    static_weight: float = 1.0,
) -> np.ndarray:
    fx = float(intrinsics.get("fx") or 0.0)
    fy = float(intrinsics.get("fy") or 0.0)
    cx = float(intrinsics.get("cx") or 0.0)
    cy = float(intrinsics.get("cy") or 0.0)
    if fx <= 0.0 or fy <= 0.0:
        return np.zeros((0, 4), dtype=np.float32)

    rows: List[List[float]] = []
    height, width = depth.shape[:2]
    confidence_arr = confidence if confidence is not None and confidence.shape == depth.shape else None
    for v in range(0, height, max(1, sample_step)):
        for u in range(0, width, max(1, sample_step)):
            z = float(depth[v, u])
            if not math.isfinite(z) or z <= 0.0:
                continue
            if confidence_arr is not None and float(confidence_arr[v, u]) < min_confidence:
                continue
            x = ((float(u) - cx) * z) / fx
            y = ((float(v) - cy) * z) / fy
            camera_point = np.array([x, y, z, 1.0], dtype=np.float64)
            world = T_world_camera @ camera_point
            weight = static_weight
            if confidence_arr is not None:
                weight *= clamp01(float(confidence_arr[v, u]), default=0.0)
            rows.append([float(world[0]), float(world[1]), float(world[2]), float(weight)])
    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def write_ascii_pointcloud(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {int(points.shape[0])}",
        "property float x",
        "property float y",
        "property float z",
        "property float weight",
        "end_header",
    ]
    lines = [" ".join(f"{float(value):.6f}" for value in row[:4]) for row in points]
    path.write_text("\n".join(header + lines) + "\n", encoding="utf-8")


def aggregate_chunk_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    storage_root: Optional[Path],
) -> Dict[str, Any]:
    embeddings: List[np.ndarray] = []
    anchor_ids: List[str] = []
    zone_id = None
    static_scores: List[float] = []
    geometry_fingerprints: List[Mapping[str, Any]] = []

    for record in records:
        emb_uri = str(record.get("embedding_uri") or "")
        if emb_uri:
            emb = load_embedding(embedding_uri=emb_uri, storage_root=storage_root)
            if emb is not None:
                embeddings.append(emb)
        zone_id = zone_id or record.get("zone_id")
        static_scores.append(clamp01(record.get("staticness_score"), default=0.0))
        anchors = record.get("anchor_observations") or []
        for anchor in anchors:
            text = str(anchor or "").strip()
            if text and text not in anchor_ids:
                anchor_ids.append(text)
        geometry = record.get("geometry_fingerprint")
        if isinstance(geometry, Mapping) and geometry:
            geometry_fingerprints.append(geometry)

    centroid = None
    if embeddings:
        centroid_arr = np.mean(np.stack(embeddings, axis=0), axis=0)
        norm = float(np.linalg.norm(centroid_arr))
        if norm > 1e-8:
            centroid = (centroid_arr / norm).astype(np.float32)

    geometry_summary: Dict[str, Any] = {}
    if geometry_fingerprints:
        geometry_summary = {
            "median_depth_m": round(
                float(np.mean([float(item.get("median_depth_m") or 0.0) for item in geometry_fingerprints])),
                4,
            ),
            "plane_support_ratio": round(
                float(np.mean([float(item.get("plane_support_ratio") or 0.0) for item in geometry_fingerprints])),
                4,
            ),
            "surface_complexity": round(
                float(np.mean([float(item.get("surface_complexity") or 0.0) for item in geometry_fingerprints])),
                4,
            ),
            "depth_histogram_8": _mean_histograms(
                [item.get("depth_histogram_8") or [] for item in geometry_fingerprints]
            ),
        }

    return {
        "record_count": len(records),
        "zone_id": zone_id,
        "anchor_ids": anchor_ids,
        "staticness_score": round(float(np.mean(static_scores or [0.0])), 4),
        "geometry_fingerprint": geometry_summary,
        "embedding_centroid": centroid,
    }


def _mean_histograms(items: Sequence[Sequence[float]]) -> List[float]:
    arrays = [np.asarray(list(item), dtype=np.float64) for item in items if item]
    if not arrays:
        return []
    min_size = min(arr.size for arr in arrays)
    if min_size <= 0:
        return []
    stacked = np.stack([arr[:min_size] for arr in arrays], axis=0)
    return [round(float(value), 6) for value in np.mean(stacked, axis=0).tolist()]


def plane_summaries(points: np.ndarray) -> List[Dict[str, Any]]:
    if points.size == 0:
        return []
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    if xyz.shape[0] < 12:
        return []
    planes: List[Dict[str, Any]] = []
    y_values = xyz[:, 1]
    for label, target in (("floor_like", float(np.percentile(y_values, 10))), ("ceiling_like", float(np.percentile(y_values, 90)))):
        support_mask = np.abs(y_values - target) <= 0.15
        support = xyz[support_mask]
        if support.shape[0] < 6:
            continue
        mins = support.min(axis=0)
        maxs = support.max(axis=0)
        planes.append(
            {
                "plane_id": label,
                "orientation": "horizontal",
                "y_m": round(target, 4),
                "support_count": int(support.shape[0]),
                "extent_x_m": round(float(maxs[0] - mins[0]), 4),
                "extent_z_m": round(float(maxs[2] - mins[2]), 4),
            }
        )
    return planes


def iter_groups(records: Sequence[Mapping[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        group_id = str(record.get(key) or "").strip()
        if not group_id:
            continue
        groups.setdefault(group_id, []).append(dict(record))
    return groups
