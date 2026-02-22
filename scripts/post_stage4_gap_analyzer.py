#!/usr/bin/env python3
"""Post-Stage-4 gap observability analysis for Gaussian outputs.

This stage inspects rendered images, estimates hole regions, and proposes
pseudo-view candidates for image-space repair.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pil_image_module():
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for gap analysis image IO") from exc
    return Image


def _load_image_rgb_alpha(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    Image = _pil_image_module()
    img = Image.open(path)
    bands = img.getbands()
    alpha: np.ndarray | None = None
    if "A" in bands:
        alpha_img = np.asarray(img.getchannel("A"), dtype=np.uint8)
        alpha = alpha_img
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return rgb, alpha


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    return 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]


def compute_hole_mask(
    rgb: np.ndarray,
    *,
    alpha: np.ndarray | None = None,
    dark_threshold: int = 18,
    low_contrast_threshold: int = 8,
) -> np.ndarray:
    """Estimate missing/invalid regions from rendered RGB(+alpha)."""
    gray = _to_gray(rgb)
    dark = gray <= float(dark_threshold)
    contrast = np.max(rgb, axis=2).astype(np.int16) - np.min(rgb, axis=2).astype(np.int16)
    flat_dark = np.logical_and(dark, contrast <= int(low_contrast_threshold))
    if alpha is not None:
        alpha_hole = alpha <= 8
        return np.logical_or(flat_dark, alpha_hole)
    return flat_dark


def _connected_components(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    count = 0
    for y in range(h):
        for x in range(w):
            if not bool(mask[y, x]) or visited[y, x] != 0:
                continue
            count += 1
            stack = [(y, x)]
            visited[y, x] = 1
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and visited[ny, nx] == 0 and bool(mask[ny, nx]):
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
    return count


def _laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    center = g[1:-1, 1:-1]
    up = g[:-2, 1:-1]
    down = g[2:, 1:-1]
    left = g[1:-1, :-2]
    right = g[1:-1, 2:]
    lap = (4.0 * center) - up - down - left - right
    if lap.size == 0:
        return 0.0
    return float(np.var(lap))


def _qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = [float(v) for v in qvec]
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def _view_dir_from_qvec(qvec: Sequence[float]) -> np.ndarray:
    rot_wc = _qvec_to_rotmat(qvec).T
    forward = rot_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return forward / norm


def _rotate_yaw_deg(vec: np.ndarray, yaw_deg: float) -> np.ndarray:
    theta = math.radians(float(yaw_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    out = rot @ vec
    norm = float(np.linalg.norm(out))
    if norm <= 1e-8:
        return vec
    return out / norm


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    cosine = float(np.dot(a, b) / denom)
    cosine = max(-1.0, min(1.0, cosine))
    return float(math.degrees(math.acos(cosine)))


def _load_poses_from_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    poses: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return poses
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        name = str(row.get("image") or row.get("name") or "").strip()
        qvec = row.get("qvec") if isinstance(row.get("qvec"), list) else None
        tvec = row.get("tvec") if isinstance(row.get("tvec"), list) else None
        if name and qvec and len(qvec) == 4 and tvec and len(tvec) == 3:
            poses[name] = {
                "qvec": [float(v) for v in qvec],
                "tvec": [float(v) for v in tvec],
            }
    return poses


def _load_colmap_images_txt(path: Path) -> Dict[str, Dict[str, Any]]:
    poses: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return poses
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    idx = 0
    while idx < len(lines):
        text = lines[idx].strip()
        idx += 1
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) < 10:
            continue
        try:
            image_id = int(parts[0])
            qvec = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            tvec = [float(parts[5]), float(parts[6]), float(parts[7])]
            name = parts[9]
        except Exception:
            continue
        poses[name] = {"qvec": qvec, "tvec": tvec, "image_id": image_id}
        # Skip 2D points line.
        if idx < len(lines):
            idx += 1
    return poses


def _load_colmap_images_bin(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read camera poses from COLMAP images.bin (binary format)."""
    poses: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return poses
    try:
        data = path.read_bytes()
    except Exception:
        return poses
    if len(data) < 8:
        return poses
    try:
        (num_images,) = struct.unpack_from("<Q", data, 0)
    except struct.error:
        return poses
    offset = 8
    for _ in range(int(num_images)):
        if offset + 64 > len(data):
            break
        try:
            image_id = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            qw, qx, qy, qz = struct.unpack_from("<dddd", data, offset)
            offset += 32
            tx, ty, tz = struct.unpack_from("<ddd", data, offset)
            offset += 24
            camera_id = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            # Read null-terminated image name.
            name_bytes = bytearray()
            while offset < len(data) and data[offset] != 0:
                name_bytes.append(data[offset])
                offset += 1
            offset += 1  # skip null terminator
            name = name_bytes.decode("utf-8", errors="replace")
            # Read number of 2D points and skip them.
            if offset + 8 > len(data):
                break
            (num_points2d,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            # Each 2D point: x(double) + y(double) + point3D_id(int64) = 24 bytes
            offset += int(num_points2d) * 24
            poses[name] = {
                "qvec": [float(qw), float(qx), float(qy), float(qz)],
                "tvec": [float(tx), float(ty), float(tz)],
                "image_id": int(image_id),
            }
        except (struct.error, UnicodeDecodeError):
            break
    return poses


def _build_render_index_pose_map(pose_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map 3DGRUT render indices (00000.png) to ordered COLMAP poses."""
    indexed: List[tuple[int, Dict[str, Any]]] = []
    for pose in pose_map.values():
        try:
            image_id = int(pose.get("image_id"))
        except Exception:
            continue
        indexed.append((image_id, pose))
    if not indexed:
        return {}

    indexed.sort(key=lambda item: item[0])
    index_pose_map: Dict[str, Dict[str, Any]] = {}
    for idx, (_image_id, pose) in enumerate(indexed):
        key = f"{idx:05d}.png"
        index_pose_map[key] = pose
    return index_pose_map


def _rotmat_to_qvec(R: np.ndarray) -> List[float]:
    """Convert 3x3 rotation matrix to COLMAP quaternion [qw, qx, qy, qz]."""
    # Shepperd's method — numerically stable for all rotations.
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [qw / norm, qx / norm, qy / norm, qz / norm]


def _camera_center_from_pose(qvec: Sequence[float], tvec: Sequence[float]) -> np.ndarray:
    """World-space camera center: C = -R^T @ t."""
    R = _qvec_to_rotmat(qvec)
    t = np.array([float(v) for v in tvec], dtype=np.float64)
    return -R.T @ t


def _look_at_qvec(eye: np.ndarray, target: np.ndarray) -> List[float]:
    """COLMAP quaternion for a camera at *eye* looking toward *target*.

    COLMAP camera convention: +Z points away from the scene (into the sensor),
    +X right, +Y down. The world-to-camera rotation R satisfies R @ forward = [0,0,1]
    where forward = normalize(target - eye).
    """
    fwd = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    norm = float(np.linalg.norm(fwd))
    if norm < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    fwd = fwd / norm

    # World up hint — use +Y if forward isn't nearly parallel to it
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # COLMAP Y-down
    if abs(float(np.dot(fwd, world_up))) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    right = np.cross(fwd, world_up)
    rnorm = float(np.linalg.norm(right))
    if rnorm < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    right = right / rnorm
    down = np.cross(fwd, right)  # Y axis in camera frame

    # Camera-from-world rotation: rows = camera axes expressed in world
    # COLMAP: R maps world→camera, camera Z = -forward (looks opposite to fwd)
    # Actually: camera +Z = fwd direction in COLMAP convention for the viewing direction
    # R_cw = [[right], [down], [fwd]]  →  R @ fwd = [0,0,1]
    R_cw = np.stack([right, down, fwd], axis=0)
    return _rotmat_to_qvec(R_cw)


def _load_colmap_points3d_bin(path: Path) -> np.ndarray:
    """Read 3D point XYZ from COLMAP points3D.bin. Returns (N, 3) float64 array."""
    if not path.is_file():
        return np.zeros((0, 3), dtype=np.float64)
    data = path.read_bytes()
    if len(data) < 8:
        return np.zeros((0, 3), dtype=np.float64)
    (num_points,) = struct.unpack_from("<Q", data, 0)
    pts = []
    offset = 8
    for _ in range(int(num_points)):
        if offset + 43 > len(data):
            break
        # point3D_id(8) + xyz(24) + rgb(3) + error(8) = 43 bytes, then track(variable)
        _pid = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        x, y, z = struct.unpack_from("<ddd", data, offset)
        offset += 24
        offset += 3  # rgb
        offset += 8  # error
        # track length + track entries
        if offset + 8 > len(data):
            pts.append([x, y, z])
            break
        (track_len,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        offset += int(track_len) * 8  # each entry: image_id(4) + point2D_idx(4)
        pts.append([x, y, z])
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array(pts, dtype=np.float64)


def compute_scene_bounds(
    points3d: np.ndarray,
    camera_centers: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return (scene_center, scene_radius) from SfM points and camera positions."""
    all_pts = np.concatenate([points3d, camera_centers], axis=0) if len(points3d) > 0 else camera_centers
    if len(all_pts) == 0:
        return np.zeros(3, dtype=np.float64), 1.0
    center = np.median(all_pts, axis=0)
    dists = np.linalg.norm(all_pts - center, axis=1)
    # Use 95th percentile to exclude extreme outliers
    radius = float(np.percentile(dists, 95)) if len(dists) > 0 else 1.0
    return center, max(radius, 0.1)


def _build_coverage_map(
    camera_dirs: np.ndarray,
    n_phi: int = 36,
    n_theta: int = 18,
) -> np.ndarray:
    """2D histogram of camera viewing directions on the unit sphere.

    phi (azimuth) ∈ [0, 2π), theta (elevation) ∈ [0, π].
    Returns (n_theta, n_phi) int array of view counts per bin.
    """
    coverage = np.zeros((n_theta, n_phi), dtype=np.int32)
    for d in camera_dirs:
        norm = float(np.linalg.norm(d))
        if norm < 1e-8:
            continue
        dn = d / norm
        theta = math.acos(max(-1.0, min(1.0, float(dn[1]))))  # Y = up/down
        phi = math.atan2(float(dn[0]), float(dn[2]))  # XZ plane
        if phi < 0:
            phi += 2.0 * math.pi
        ti = min(int(theta / math.pi * n_theta), n_theta - 1)
        pi = min(int(phi / (2.0 * math.pi) * n_phi), n_phi - 1)
        coverage[ti, pi] += 1
    return coverage


def generate_void_filling_candidates(
    scene_center: np.ndarray,
    scene_radius: float,
    existing_poses: Dict[str, Dict[str, Any]],
    *,
    max_candidates: int = 48,
    n_phi: int = 36,
    n_theta: int = 18,
    orbit_radius_factor: float = 1.5,
    exclude_poles: bool = False,
    pole_exclusion_fraction: float = 0.05,
) -> List[Dict[str, Any]]:
    """Place virtual cameras on a sphere in under-covered viewing directions.

    Returns list of candidate dicts with is_virtual=True, compatible with
    the existing candidate JSONL format.
    """
    # Extract existing camera centers and viewing directions
    centers = []
    dirs = []
    for pose in existing_poses.values():
        qvec = pose["qvec"]
        tvec = pose["tvec"]
        c = _camera_center_from_pose(qvec, tvec)
        centers.append(c)
        dirs.append(_view_dir_from_qvec(qvec))

    if not centers:
        return []

    camera_centers = np.array(centers, dtype=np.float64)
    camera_dirs = np.array(dirs, dtype=np.float64)

    coverage = _build_coverage_map(camera_dirs, n_phi=n_phi, n_theta=n_theta)

    # Generate candidate positions in under-covered bins
    orbit_r = scene_radius * orbit_radius_factor
    candidates: List[Dict[str, Any]] = []

    # Score each bin by how under-covered it is
    bin_scores: List[tuple[float, int, int]] = []
    for ti in range(n_theta):
        # Optional compatibility mode for callers that want to skip poles.
        theta_center = (ti + 0.5) / n_theta * math.pi
        if exclude_poles:
            frac = max(0.0, min(0.49, float(pole_exclusion_fraction)))
            if theta_center < frac * math.pi or theta_center > (1.0 - frac) * math.pi:
                continue
        for pi in range(n_phi):
            count = int(coverage[ti, pi])
            # Inverse coverage = higher score for less-covered bins
            score = 1.0 / (1.0 + count)
            bin_scores.append((score, ti, pi))

    # Sort by score descending (least covered first)
    bin_scores.sort(key=lambda x: x[0], reverse=True)

    existing_view_dirs = [np.array(d, dtype=np.float64) for d in dirs]

    for score, ti, pi in bin_scores:
        if len(candidates) >= max_candidates:
            break
        if score <= 0.25:
            # Already well-covered — skip but keep processing other bins
            continue

        theta = (ti + 0.5) / n_theta * math.pi
        phi = (pi + 0.5) / n_phi * 2.0 * math.pi

        # Camera position on sphere, looking inward
        eye = scene_center + orbit_r * np.array([
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
            math.sin(theta) * math.cos(phi),
        ], dtype=np.float64)

        # View direction: from eye toward scene center
        view_dir = scene_center - eye
        vn = float(np.linalg.norm(view_dir))
        if vn < 1e-8:
            continue
        view_dir = view_dir / vn

        # Check parallax to existing views
        min_angle = 180.0
        for ed in existing_view_dirs:
            angle = _angle_between_deg(view_dir, ed)
            min_angle = min(min_angle, angle)

        qvec = _look_at_qvec(eye, scene_center)
        R = _qvec_to_rotmat(qvec)
        tvec_arr = -R @ eye
        tvec = [float(tvec_arr[0]), float(tvec_arr[1]), float(tvec_arr[2])]

        candidates.append({
            "id": f"virtual_theta{ti}_phi{pi}",
            "source_image": "",
            "render_image": "",
            "hole_ratio": 0.0,
            "hole_pixels": 0,
            "cluster_count": 0,
            "sharpness": 0.0,
            "score": float(score) * 1000.0,
            "yaw_offset_deg": 0.0,
            "parallax_to_nearest_captured_deg": float(min_angle),
            "view_dir": [float(view_dir[0]), float(view_dir[1]), float(view_dir[2])],
            "qvec": [float(v) for v in qvec],
            "tvec": tvec,
            "camera_center": [float(eye[0]), float(eye[1]), float(eye[2])],
            "is_virtual": True,
            "coverage_bin": [ti, pi],
            "coverage_count": int(coverage[ti, pi]),
        })

    return candidates


def _fallback_pose_for_index(index: int, total: int) -> tuple[List[float], List[float], np.ndarray]:
    total_safe = max(1, total)
    theta = 2.0 * math.pi * (float(index) / float(total_safe))
    view_dir = np.array([math.sin(theta), -0.05, math.cos(theta)], dtype=np.float64)
    norm = float(np.linalg.norm(view_dir))
    if norm > 1e-8:
        view_dir = view_dir / norm
    # Identity quaternion and synthetic camera center ring.
    qvec = [1.0, 0.0, 0.0, 0.0]
    tvec = [2.0 * math.sin(theta), 1.5, 2.0 * math.cos(theta)]
    return qvec, tvec, view_dir


def _collect_render_images(renders_dir: Path) -> List[Path]:
    images: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        images.extend(sorted(renders_dir.rglob(ext)))
    return sorted({p.resolve() for p in images})


def rank_candidate_views(
    candidates: Sequence[Dict[str, Any]],
    *,
    max_candidates: int,
    min_parallax_deg: float,
) -> List[Dict[str, Any]]:
    """Greedy candidate selection with angular diversity."""
    ordered = sorted(
        candidates,
        key=lambda c: (
            float(c.get("score", 0.0)),
            float(c.get("hole_ratio", 0.0)),
            float(c.get("cluster_count", 0.0)),
            -float(c.get("sharpness", 0.0)),
        ),
        reverse=True,
    )
    selected: List[Dict[str, Any]] = []
    selected_dirs: List[np.ndarray] = []
    min_parallax = float(max(0.0, min_parallax_deg))
    for cand in ordered:
        if len(selected) >= max_candidates:
            break
        parallax_to_capture = float(cand.get("parallax_to_nearest_captured_deg", 0.0))
        if parallax_to_capture < min_parallax:
            continue
        vec_raw = cand.get("view_dir")
        if not isinstance(vec_raw, list) or len(vec_raw) != 3:
            continue
        vec = np.array([float(v) for v in vec_raw], dtype=np.float64)
        if selected_dirs:
            min_angle = min(_angle_between_deg(vec, prev) for prev in selected_dirs)
            if min_angle < min_parallax:
                continue
        selected.append(cand)
        selected_dirs.append(vec)
    return selected


def analyze_gap_observability(
    *,
    renders_dir: Path,
    output_dir: Path,
    max_candidate_views: int,
    min_parallax_deg: float,
    poses_jsonl: Path | None = None,
    colmap_images_txt: Path | None = None,
    colmap_images_bin: Path | None = None,
    colmap_points3d_bin: Path | None = None,
    max_virtual_candidates: int = 48,
    exclude_poles: bool = False,
) -> Dict[str, Any]:
    images = _collect_render_images(renders_dir)
    if not images:
        raise RuntimeError(f"No render images found under {renders_dir}")

    pose_map: Dict[str, Dict[str, Any]] = {}
    if poses_jsonl is not None:
        pose_map.update(_load_poses_from_jsonl(poses_jsonl))
    if colmap_images_bin is not None:
        pose_map.update(_load_colmap_images_bin(colmap_images_bin))
    if colmap_images_txt is not None:
        pose_map.update(_load_colmap_images_txt(colmap_images_txt))
    render_index_pose_map = _build_render_index_pose_map(pose_map)

    preview_dir = output_dir / "gap_mask_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    frame_stats: List[Dict[str, Any]] = []
    total_pixels = 0
    total_hole_pixels = 0
    total_clusters = 0
    pose_match_count = 0
    pose_index_match_count = 0

    for idx, image_path in enumerate(images):
        rgb, alpha = _load_image_rgb_alpha(image_path)
        mask = compute_hole_mask(rgb, alpha=alpha)
        hole_pixels = int(mask.sum())
        pixels = int(mask.size)
        total_pixels += pixels
        total_hole_pixels += hole_pixels
        clusters = _connected_components(mask)
        total_clusters += clusters

        gray = _to_gray(rgb)
        sharpness = _laplacian_variance(gray)

        pose = pose_map.get(image_path.name)
        if pose is None and render_index_pose_map:
            pose = render_index_pose_map.get(image_path.name)
            if pose is None:
                stem = image_path.stem.strip()
                if stem.isdigit():
                    pose = render_index_pose_map.get(f"{int(stem):05d}.png")
            if pose is not None:
                pose_index_match_count += 1
        if pose is None:
            qvec, tvec, view_dir = _fallback_pose_for_index(idx, len(images))
        else:
            pose_match_count += 1
            qvec = [float(v) for v in pose.get("qvec", [1.0, 0.0, 0.0, 0.0])]
            tvec = [float(v) for v in pose.get("tvec", [0.0, 0.0, 0.0])]
            view_dir = _view_dir_from_qvec(qvec)

        frame_stats.append(
            {
                "image_path": image_path,
                "hole_ratio": float(hole_pixels) / float(max(1, pixels)),
                "hole_pixels": int(hole_pixels),
                "cluster_count": int(clusters),
                "sharpness": float(sharpness),
                "view_dir": view_dir,
                "qvec": qvec,
                "tvec": tvec,
            }
        )

        Image = _pil_image_module()
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask_img.save(preview_dir / f"{image_path.stem}_mask.png")

    pseudo_candidates: List[Dict[str, Any]] = []
    for stat in frame_stats:
        # Generate pseudo viewpoints as yaw perturbations around captured poses.
        for yaw_offset in (-14.0, -7.0, 7.0, 14.0):
            view_dir = _rotate_yaw_deg(stat["view_dir"], yaw_offset)
            pseudo_candidates.append(
                {
                    "id": f"{stat['image_path'].stem}_yaw_{yaw_offset:+.1f}",
                    "source_image": stat["image_path"].name,
                    "render_image": str(stat["image_path"]),
                    "hole_ratio": float(stat["hole_ratio"]),
                    "hole_pixels": int(stat["hole_pixels"]),
                    "cluster_count": int(stat["cluster_count"]),
                    "sharpness": float(stat["sharpness"]),
                    "score": float(stat["hole_ratio"]) * 1000.0 + float(stat["cluster_count"]) * 0.5,
                    "yaw_offset_deg": float(yaw_offset),
                    "parallax_to_nearest_captured_deg": float(abs(yaw_offset)),
                    "view_dir": [float(v) for v in view_dir.tolist()],
                    "qvec": [float(v) for v in stat["qvec"]],
                    "tvec": [float(v) for v in stat["tvec"]],
                }
            )

    # Generate void-filling virtual cameras from under-covered sphere directions.
    virtual_candidates: List[Dict[str, Any]] = []
    if colmap_points3d_bin is not None and pose_map:
        points3d = _load_colmap_points3d_bin(colmap_points3d_bin)
        cam_centers = np.array(
            [_camera_center_from_pose(p["qvec"], p["tvec"]) for p in pose_map.values()],
            dtype=np.float64,
        )
        scene_center, scene_radius = compute_scene_bounds(points3d, cam_centers)
        virtual_candidates = generate_void_filling_candidates(
            scene_center,
            scene_radius,
            pose_map,
            max_candidates=max(1, max_virtual_candidates),
            exclude_poles=bool(exclude_poles),
        )

    # Merge yaw-perturbation and virtual candidates
    all_candidates = pseudo_candidates + virtual_candidates

    selected = rank_candidate_views(
        all_candidates,
        max_candidates=max_candidate_views,
        min_parallax_deg=min_parallax_deg,
    )

    candidates_path = output_dir / "gap_candidate_views.jsonl"
    with candidates_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    report: Dict[str, Any] = {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "renders_dir": str(renders_dir),
        "input_render_count": int(len(frame_stats)),
        "pose_mapping_mode": "name_or_colmap_index" if render_index_pose_map else "name_only",
        "pose_match_count": int(pose_match_count),
        "pose_fallback_count": int(max(0, len(frame_stats) - pose_match_count)),
        "pose_index_match_count": int(pose_index_match_count),
        "global_hole_pixel_ratio": float(total_hole_pixels) / float(max(1, total_pixels)),
        "total_hole_pixels": int(total_hole_pixels),
        "total_pixels": int(total_pixels),
        "total_cluster_count": int(total_clusters),
        "candidate_view_count": int(len(selected)),
        "max_candidate_views": int(max_candidate_views),
        "min_parallax_deg": float(min_parallax_deg),
        "virtual_candidate_count": int(len(virtual_candidates)),
        "virtual_candidates_selected": int(sum(1 for c in selected if c.get("is_virtual"))),
        "candidate_views_path": str(candidates_path),
        "mask_preview_dir": str(preview_dir),
        "top_hole_frames": [
            {
                "image": stat["image_path"].name,
                "hole_ratio": float(stat["hole_ratio"]),
                "cluster_count": int(stat["cluster_count"]),
                "sharpness": float(stat["sharpness"]),
            }
            for stat in sorted(frame_stats, key=lambda s: float(s["hole_ratio"]), reverse=True)[:10]
        ],
    }
    report_path = output_dir / "gap_analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Stage-4 render gaps and propose pseudo-view repairs")
    parser.add_argument("--renders-dir", required=True, help="Path to 3DGRUT renders directory")
    parser.add_argument("--output-dir", required=True, help="Path to NuRec output directory")
    parser.add_argument(
        "--max-candidate-views",
        type=int,
        default=int(os.getenv("POST_STAGE4_MAX_PSEUDOVIEWS", "96")),
        help="Maximum pseudo-views to propose",
    )
    parser.add_argument(
        "--min-parallax-deg",
        type=float,
        default=float(os.getenv("POST_STAGE4_MIN_PARALLAX_DEG", "7.0")),
        help="Minimum angular separation from nearest captured camera",
    )
    parser.add_argument(
        "--poses-jsonl",
        default="",
        help="Optional pose JSONL path with fields {image|name,qvec,tvec}",
    )
    parser.add_argument(
        "--colmap-images-txt",
        default="",
        help="Optional COLMAP images.txt path for camera poses",
    )
    parser.add_argument(
        "--colmap-images-bin",
        default="",
        help="Optional COLMAP images.bin (binary) path for camera poses",
    )
    parser.add_argument(
        "--colmap-points3d-bin",
        default="",
        help="Optional COLMAP points3D.bin path for scene bounds (enables void-fill cameras)",
    )
    parser.add_argument(
        "--max-virtual-candidates",
        type=int,
        default=int(os.getenv("POST_STAGE4_MAX_VIRTUAL_CANDIDATES", "48")),
        help="Maximum virtual void-fill camera candidates to generate",
    )
    parser.add_argument(
        "--exclude-poles",
        action="store_true",
        default=str(os.getenv("POST_STAGE4_EXCLUDE_POLES", "false")).strip().lower() in {"1", "true", "yes", "on"},
        help="Exclude near-pole directions when generating virtual void-fill candidates",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    renders_dir = Path(args.renders_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    poses_jsonl = Path(args.poses_jsonl) if str(args.poses_jsonl).strip() else None
    colmap_images_txt = Path(args.colmap_images_txt) if str(args.colmap_images_txt).strip() else None
    colmap_images_bin = Path(args.colmap_images_bin) if str(args.colmap_images_bin).strip() else None
    colmap_points3d_bin = Path(args.colmap_points3d_bin) if str(args.colmap_points3d_bin).strip() else None

    analyze_gap_observability(
        renders_dir=renders_dir,
        output_dir=output_dir,
        max_candidate_views=max(1, int(args.max_candidate_views)),
        min_parallax_deg=max(0.0, float(args.min_parallax_deg)),
        poses_jsonl=poses_jsonl,
        colmap_images_txt=colmap_images_txt,
        colmap_images_bin=colmap_images_bin,
        colmap_points3d_bin=colmap_points3d_bin,
        max_virtual_candidates=max(1, int(args.max_virtual_candidates)),
        exclude_poles=bool(args.exclude_poles),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
