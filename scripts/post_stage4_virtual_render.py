#!/usr/bin/env python3
"""Render the current 3DGRUT checkpoint from arbitrary virtual camera poses.

Builds a synthetic COLMAP dataset (text format) with the virtual cameras and
invokes 3DGRUT in render-only mode (n_iterations=0 or 1) to produce images.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from post_stage4_gap_analyzer import compute_hole_mask as _compute_hole_mask
except Exception:
    _compute_hole_mask = None  # type: ignore[assignment]


def _log(msg: str) -> None:
    print(f"[virtual-render] {msg}", flush=True)


MAX_CAMERA_DIMENSION = 8192
MAX_CAMERA_PIXELS = 8192 * 8192


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _load_rgb_alpha(path: Path) -> tuple[Any, Any]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for virtual-render hole-ratio estimation") from exc
    img = Image.open(path)
    alpha = None
    if "A" in img.getbands():
        alpha = img.getchannel("A")
    return img.convert("RGB"), alpha


def _estimate_hole_ratio(path: Path) -> float:
    rgb_img, alpha_img = _load_rgb_alpha(path)
    import numpy as np
    rgb = np.asarray(rgb_img, dtype=np.uint8)
    alpha = np.asarray(alpha_img, dtype=np.uint8) if alpha_img is not None else None
    if _compute_hole_mask is not None:
        hole = _compute_hole_mask(rgb, alpha=alpha)
    else:
        gray = 0.299 * rgb[..., 0].astype(np.float32) + 0.587 * rgb[..., 1].astype(np.float32) + 0.114 * rgb[..., 2].astype(np.float32)
        dark = gray <= 18.0
        contrast = np.max(rgb, axis=2).astype(np.int16) - np.min(rgb, axis=2).astype(np.int16)
        hole = np.logical_and(dark, contrast <= 8)
        if alpha is not None:
            hole = np.logical_or(hole, alpha <= 8)
    return float(hole.sum()) / float(max(1, hole.size))


def _read_colmap_cameras_bin(path: Path) -> Dict[str, Any] | None:
    """Read the first camera entry from cameras.bin to reuse intrinsics."""
    if not path.is_file():
        return None
    data = path.read_bytes()
    if len(data) < 8:
        return None
    (num_cameras,) = struct.unpack_from("<Q", data, 0)
    if num_cameras < 1:
        return None
    offset = 8
    camera_id = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    model_id = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    width = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    height = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    # Model parameter counts: https://colmap.github.io/cameras.html
    model_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5, 6: 8, 7: 12, 8: 4, 9: 5}
    n_params = model_params.get(model_id, 4)
    params = []
    for _ in range(n_params):
        (p,) = struct.unpack_from("<d", data, offset)
        offset += 8
        params.append(p)
    model_names = {
        0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
        3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE",
        6: "FULL_OPENCV", 7: "FOV", 8: "SIMPLE_RADIAL_FISHEYE",
        9: "RADIAL_FISHEYE",
    }
    return {
        "camera_id": camera_id,
        "model": model_names.get(model_id, "PINHOLE"),
        "width": int(width),
        "height": int(height),
        "params": params,
    }


def _read_colmap_cameras_txt(path: Path) -> Dict[str, Any] | None:
    """Read the first camera entry from cameras.txt."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        return {
            "camera_id": int(parts[0]),
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": [float(p) for p in parts[4:]],
        }
    return None


def _is_valid_camera_dims(width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    if width > MAX_CAMERA_DIMENSION or height > MAX_CAMERA_DIMENSION:
        return False
    return (width * height) <= MAX_CAMERA_PIXELS


def write_colmap_cameras_txt(path: Path, camera: Dict[str, Any]) -> None:
    """Write cameras.txt with a single camera entry."""
    params_str = " ".join(f"{p:.10f}" for p in camera["params"])
    camera_id = int(camera.get("camera_id", 1))
    with path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        f.write(f"{camera_id} {camera['model']} {camera['width']} {camera['height']} {params_str}\n")


def write_colmap_images_txt(path: Path, entries: List[Dict[str, Any]]) -> None:
    """Write images.txt with virtual camera poses.

    Each entry: {image_id, qvec: [qw,qx,qy,qz], tvec: [tx,ty,tz], name: str}
    """
    with path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(entries)}\n")
        for entry in entries:
            qvec = entry["qvec"]
            tvec = entry["tvec"]
            f.write(
                f"{entry['image_id']} "
                f"{qvec[0]:.10f} {qvec[1]:.10f} {qvec[2]:.10f} {qvec[3]:.10f} "
                f"{tvec[0]:.10f} {tvec[1]:.10f} {tvec[2]:.10f} "
                f"{int(entry.get('camera_id', 1))} {entry['name']}\n"
            )
            f.write("\n")  # Empty 2D points line


def write_colmap_points3d_txt(path: Path) -> None:
    """Write empty points3D.txt (no SfM points needed for rendering)."""
    with path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# Number of points: 0\n")


def build_virtual_colmap_dataset(
    output_dir: Path,
    candidates: List[Dict[str, Any]],
    reference_camera: Dict[str, Any],
) -> Path:
    """Build a synthetic COLMAP dataset directory for virtual cameras.

    Creates: output_dir/sparse/0/{cameras,images,points3D}.txt + output_dir/images/
    Returns the dataset root (output_dir).
    """
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    width = int(reference_camera["width"])
    height = int(reference_camera["height"])
    if not _is_valid_camera_dims(width, height):
        raise ValueError(f"Invalid reference camera dimensions: {width}x{height}")

    write_colmap_cameras_txt(sparse_dir / "cameras.txt", reference_camera)
    write_colmap_points3d_txt(sparse_dir / "points3D.txt")

    entries = []
    primary_camera_id = int(reference_camera.get("camera_id", 1))
    for idx, cand in enumerate(candidates):
        image_name = f"virtual_{idx:05d}.png"
        entries.append({
            "image_id": idx + 1,
            "qvec": cand["qvec"],
            "tvec": cand["tvec"],
            "camera_id": int(cand.get("camera_id", primary_camera_id)),
            "name": image_name,
        })
        # Create placeholder image (3DGRUT requires images to exist)
        _create_placeholder_image(images_dir / image_name, width, height)

    write_colmap_images_txt(sparse_dir / "images.txt", entries)
    return output_dir


def _create_placeholder_image(path: Path, width: int, height: int) -> None:
    """Create a minimal black PNG placeholder."""
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), (0, 0, 0))
        img.save(path)
    except ImportError:
        # Fallback: write a minimal 1x1 PNG (3DGRUT may still work)
        import zlib
        def _minimal_png(w: int, h: int) -> bytes:
            raw = b""
            for _ in range(h):
                raw += b"\x00" + b"\x00\x00\x00" * w
            compressed = zlib.compress(raw)
            def chunk(ctype: bytes, data: bytes) -> bytes:
                import struct as st
                c = ctype + data
                return st.pack(">I", len(data)) + c + st.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            import struct as st
            header = b"\x89PNG\r\n\x1a\n"
            ihdr = st.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
            return header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
        path.write_bytes(_minimal_png(min(width, 4), min(height, 4)))


def render_virtual_views(
    virtual_dataset_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    threedgrut_python: str = "python3.11",
    threedgrut_dir: Path = Path("/opt/3dgrut"),
    timeout_sec: int = 1800,
) -> Tuple[bool, Path, str]:
    """Invoke 3DGRUT to render from virtual camera poses.

    Uses resume from checkpoint with n_iterations=0 (render-only).
    Returns (success, renders_dir, log_tail).
    """
    train_script = threedgrut_dir / "train.py"
    render_out = output_dir / "virtual_render_run"
    render_out.mkdir(parents=True, exist_ok=True)

    # Try n_iterations=0 first (render-only, no training)
    for n_iters in [0, 1]:
        cmd = [
            threedgrut_python,
            str(train_script),
            "--config-name", "apps/colmap_3dgut_mcmc",
            f"path={virtual_dataset_dir}/",
            f"out_dir={render_out}/",
            "experiment_name=virtual_render",
            f"n_iterations={n_iters}",
            f"resume={checkpoint_path}",
            "with_gui=false",
            "with_viser_gui=false",
            "num_workers=2",
            "export_ply.enabled=false",
            "export_usdz.enabled=false",
        ]
        _log(f"Running 3DGRUT render (n_iterations={n_iters}): {' '.join(cmd)}")

        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        proc = subprocess.run(
            cmd,
            cwd=str(threedgrut_dir),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
            env=env,
        )

        # Find renders directory
        renders_dirs = sorted(render_out.rglob("renders"), key=lambda p: p.stat().st_mtime, reverse=True)
        if renders_dirs:
            render_images = list(renders_dirs[0].glob("*.png"))
            if render_images:
                _log(f"Rendered {len(render_images)} virtual views (n_iterations={n_iters})")
                return True, renders_dirs[0], (proc.stdout or "")[-2000:]

        if n_iters == 0:
            _log("n_iterations=0 produced no renders; trying n_iterations=1")

    log_tail = (proc.stderr or "")[-2000:] if proc else ""
    _log(f"Virtual render failed: {log_tail[-500:]}")
    return False, render_out, log_tail


def render_and_collect_virtual_views(
    *,
    candidates_jsonl: Path,
    checkpoint_path: Path,
    reference_sparse_dir: Path,
    work_dir: Path,
    threedgrut_python: str = "python3.11",
    threedgrut_dir: Path = Path("/opt/3dgrut"),
) -> Dict[str, Any]:
    """Full pipeline: load candidates → build dataset → render → collect results.

    Only processes virtual candidates (is_virtual=True).
    Returns report dict with renders_dir, rendered_count, etc.
    """
    started = time.time()
    work_dir.mkdir(parents=True, exist_ok=True)

    candidates = [c for c in _load_jsonl(candidates_jsonl) if c.get("is_virtual")]
    if not candidates:
        return {
            "status": "skipped_no_virtual_candidates",
            "rendered_count": 0,
            "elapsed_sec": 0.0,
        }

    _log(f"Processing {len(candidates)} virtual camera candidates")

    # Load reference camera intrinsics
    ref_camera = (
        _read_colmap_cameras_bin(reference_sparse_dir / "cameras.bin")
        or _read_colmap_cameras_txt(reference_sparse_dir / "cameras.txt")
    )
    if ref_camera is None:
        return {"status": "error_no_reference_camera", "rendered_count": 0, "elapsed_sec": 0.0}
    width = int(ref_camera.get("width", 0))
    height = int(ref_camera.get("height", 0))
    if not _is_valid_camera_dims(width, height):
        _log(f"Rejected suspicious camera dimensions from reference sparse data: {width}x{height}")
        return {
            "status": "error_invalid_reference_camera_dimensions",
            "rendered_count": 0,
            "elapsed_sec": float(time.time() - started),
        }

    # Build synthetic COLMAP dataset
    dataset_dir = work_dir / "virtual_dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    build_virtual_colmap_dataset(dataset_dir, candidates, ref_camera)

    # Render
    success, renders_dir, log_tail = render_virtual_views(
        dataset_dir,
        checkpoint_path,
        work_dir,
        threedgrut_python=threedgrut_python,
        threedgrut_dir=threedgrut_dir,
    )

    rendered_images = sorted(renders_dir.glob("*.png")) if success else []

    report: Dict[str, Any] = {
        "status": "ok" if success else "render_failed",
        "rendered_count": len(rendered_images),
        "candidates_count": len(candidates),
        "renders_dir": str(renders_dir),
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(checkpoint_path),
        "elapsed_sec": float(time.time() - started),
    }

    # Write candidate-to-render mapping
    mapping_path = work_dir / "virtual_render_mapping.jsonl"
    with mapping_path.open("w", encoding="utf-8") as f:
        primary_camera_id = int(ref_camera.get("camera_id", 1))
        for idx, cand in enumerate(candidates):
            render_name = f"{idx:05d}.png"
            render_path = renders_dir / render_name
            predicted_hole_ratio = _estimate_hole_ratio(render_path) if render_path.is_file() else 1.0
            entry = {
                "candidate_id": cand.get("id", f"virtual_{idx}"),
                "render_name": render_name,
                "render_exists": render_path.is_file(),
                "render_image": str(render_path.resolve()) if render_path.is_file() else "",
                "source_image": str(cand.get("source_image") or ""),
                "predicted_hole_ratio": float(predicted_hole_ratio),
                "camera_id": int(cand.get("camera_id", primary_camera_id)),
                "qvec": cand["qvec"],
                "tvec": cand["tvec"],
                "is_virtual": True,
            }
            if cand.get("camera_center"):
                entry["camera_center"] = cand["camera_center"]
            f.write(json.dumps(entry) + "\n")

    report["mapping_path"] = str(mapping_path)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render 3DGRUT from virtual camera poses")
    parser.add_argument("--candidates-jsonl", required=True, help="Path to gap_candidate_views.jsonl")
    parser.add_argument("--checkpoint", required=True, help="Path to 3DGRUT ckpt_last.pt")
    parser.add_argument("--reference-sparse-dir", required=True, help="Path to COLMAP sparse/0 with cameras.bin")
    parser.add_argument("--work-dir", required=True, help="Working directory for virtual render artifacts")
    parser.add_argument("--threedgrut-python", default=os.getenv("THREEDGRUT_PYTHON", "python3.11"))
    parser.add_argument("--threedgrut-dir", default=os.getenv("THREEDGRUT_DIR", "/opt/3dgrut"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = render_and_collect_virtual_views(
        candidates_jsonl=Path(args.candidates_jsonl),
        checkpoint_path=Path(args.checkpoint),
        reference_sparse_dir=Path(args.reference_sparse_dir),
        work_dir=Path(args.work_dir),
        threedgrut_python=args.threedgrut_python,
        threedgrut_dir=Path(args.threedgrut_dir),
    )
    report_path = Path(args.work_dir) / "virtual_render_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"Report: {report_path}")
    return 0 if report.get("status") in ("ok", "skipped_no_virtual_candidates") else 1


if __name__ == "__main__":
    raise SystemExit(main())
