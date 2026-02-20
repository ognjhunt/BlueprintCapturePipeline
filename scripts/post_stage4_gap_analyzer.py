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
            qvec = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            tvec = [float(parts[5]), float(parts[6]), float(parts[7])]
            name = parts[9]
        except Exception:
            continue
        poses[name] = {"qvec": qvec, "tvec": tvec}
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
            }
        except (struct.error, UnicodeDecodeError):
            break
    return poses


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

    preview_dir = output_dir / "gap_mask_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    frame_stats: List[Dict[str, Any]] = []
    total_pixels = 0
    total_hole_pixels = 0
    total_clusters = 0

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
        if pose is None:
            qvec, tvec, view_dir = _fallback_pose_for_index(idx, len(images))
        else:
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

    selected = rank_candidate_views(
        pseudo_candidates,
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
        "global_hole_pixel_ratio": float(total_hole_pixels) / float(max(1, total_pixels)),
        "total_hole_pixels": int(total_hole_pixels),
        "total_pixels": int(total_pixels),
        "total_cluster_count": int(total_clusters),
        "candidate_view_count": int(len(selected)),
        "max_candidate_views": int(max_candidate_views),
        "min_parallax_deg": float(min_parallax_deg),
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

    analyze_gap_observability(
        renders_dir=renders_dir,
        output_dir=output_dir,
        max_candidate_views=max(1, int(args.max_candidate_views)),
        min_parallax_deg=max(0.0, float(args.min_parallax_deg)),
        poses_jsonl=poses_jsonl,
        colmap_images_txt=colmap_images_txt,
        colmap_images_bin=colmap_images_bin,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
