"""Phase 4 synthesis orchestrator and CLI.

Full pipeline from a target camera pose to a synthesised view:

  1. Load site reference index for the site
  2. Query K nearest reference frames (spatial in site frame, or embedding-based)
  3. Load the best reference frame's depth map
  4. Forward-splat the reference frame into the target viewpoint
  5. Optionally pass the splatted image to Cosmos-Predict2.5-2B Image2World
  6. Write the output image (and optional MP4) to disk

Usage (CLI):

  python -m blueprint_pipeline.synthesis.synthesize \\
    --site-id <site_id> \\
    --storage-root /mnt/gcs \\
    --bucket my-bucket \\
    --target-pose '[[r00,r01,r02,tx],[...],[...],[0,0,0,1]]' \\
    --target-intrinsics '{"fx":1462,"fy":1462,"cx":960,"cy":720,"width":1920,"height":1440}' \\
    --output /tmp/synthesis/view.jpg \\
    --mode splat_only

  # With Cosmos generation:
  --mode cosmos_i2w

Python API:

  from blueprint_pipeline.synthesis.synthesize import synthesize_view

  result = synthesize_view(
      site_id="abc123",
      storage_root=Path("/mnt/gcs"),
      bucket="my-bucket",
      target_T_world_camera=T,      # [4, 4] numpy, in site frame
      target_intrinsics={"fx": 1462, "fy": 1462, "cx": 960, "cy": 720},
      target_h=1440,
      target_w=1920,
      output_path=Path("/tmp/view.jpg"),
      mode="splat_only",
  )
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .depth_splat import depth_splat, load_depth_png
from .plucker_rays import compute_plucker_map, normalise_plucker
from .retrieval_query import query_site


def synthesize_view(
    *,
    site_id: str,
    storage_root: Path,
    bucket: str,
    target_T_world_camera: np.ndarray,   # [4, 4] in site frame (after Phase 3B alignment)
    target_intrinsics: Dict[str, float], # fx, fy, cx, cy
    target_h: int,
    target_w: int,
    output_path: Path,
    mode: str = "splat_only",            # "splat_only" | "cosmos_i2w"
    k: int = 1,                          # number of reference frames to retrieve
    query_mode: str = "spatial",         # "spatial" | "embedding" | "hybrid"
    cosmos_model: Optional[Any] = None,  # pre-loaded Cosmos model
    depth_scale: float = 0.001,          # 0.001 for 16-bit mm PNG; 1.0 for float32 m
    fill_holes: bool = True,
    num_frames: Optional[int] = None,
    previous_tail_path: Optional[Path] = None,
    previous_tail_alpha: float = 0.35,
    lookahead_target_T_world_camera: Optional[np.ndarray] = None,
    lookahead_k: int = 1,
) -> Dict[str, Any]:
    """
    Synthesise a novel view from a target camera pose.

    Returns a dict with:
      status          — "completed" or "failed"
      output_path     — Path to the output JPEG
      reference_used  — frame_id and capture_id of the best reference frame
      coverage_frac   — fraction of output pixels covered by depth splatting
      retrieval_dist_m — camera-centre distance to retrieved reference (spatial mode)
    """
    site_index_path = (
        storage_root / bucket / "sites" / site_id / "reference_memory"
        / "site_reference_index.jsonl"
    )
    if not site_index_path.is_file():
        return {
            "status": "failed",
            "reason": f"site_reference_index not found: {site_index_path}",
        }

    # --- 1. Retrieve K nearest reference frames ---
    refs = query_site(
        site_index_path=site_index_path,
        target_T_world_camera=target_T_world_camera,
        k=k,
        mode=query_mode,
        storage_root=storage_root,
        bucket=bucket,
    )
    if not refs:
        return {"status": "failed", "reason": "no_reference_frames_found"}

    best_ref = refs[0]
    retrieved_references = [_reference_summary(item) for item in refs]
    lookahead_references: List[Dict[str, Any]] = []
    if lookahead_target_T_world_camera is not None:
        lookahead_refs = query_site(
            site_index_path=site_index_path,
            target_T_world_camera=lookahead_target_T_world_camera,
            k=lookahead_k,
            mode=query_mode,
            storage_root=storage_root,
            bucket=bucket,
        )
        lookahead_references = [_reference_summary(item) for item in lookahead_refs]

    # --- 2. Load reference frame image and depth ---
    ref_image = _load_ref_image(best_ref, storage_root=storage_root, bucket=bucket)
    if ref_image is None:
        return {
            "status": "failed",
            "reason": "could_not_load_reference_image",
            "reference_id": best_ref.get("reference_id"),
        }

    ref_depth = _load_ref_depth(
        best_ref, storage_root=storage_root, bucket=bucket, depth_scale=depth_scale
    )
    if ref_depth is None:
        # Fall back to a flat depth plane at 3m if no depth available
        ref_depth = np.full(ref_image.shape[:2], 3.0, dtype=np.float32)

    # --- 3. Build reference pose in site frame ---
    T_world_ref = _effective_pose(best_ref)
    if T_world_ref is None:
        return {"status": "failed", "reason": "reference_has_no_pose"}

    ref_intrinsics = best_ref.get("intrinsics") or {}
    if not ref_intrinsics or "fx" not in ref_intrinsics:
        # Estimate from image dimensions
        H_r, W_r = ref_image.shape[:2]
        ref_intrinsics = {
            "fx": max(W_r, H_r) * 1.0,
            "fy": max(W_r, H_r) * 1.0,
            "cx": W_r / 2.0,
            "cy": H_r / 2.0,
        }

    # --- 4. Depth splat: reference → target viewpoint ---
    warped_image, coverage_mask = depth_splat(
        ref_image=ref_image,
        ref_depth=ref_depth,
        T_world_ref=T_world_ref,
        K_ref=ref_intrinsics,
        T_world_target=target_T_world_camera,
        K_target=target_intrinsics,
        target_h=target_h,
        target_w=target_w,
        depth_scale=1.0,   # already converted during load
        fill_holes=fill_holes,
    )

    coverage_frac = float(coverage_mask.mean())
    conditioning_image = _blend_previous_tail(
        warped_image,
        previous_tail_path=previous_tail_path,
        alpha=previous_tail_alpha,
    )

    # --- 5. Compute Plücker ray map for target (for Cosmos conditioning or logging) ---
    target_plucker = compute_plucker_map(
        T_world_camera=target_T_world_camera,
        intrinsics=target_intrinsics,
        height=target_h,
        width=target_w,
    )
    target_plucker_norm = normalise_plucker(target_plucker)

    # --- 6. Generate view ---
    from .cosmos_inference import generate_view
    output_path = Path(output_path)
    generate_view(
        splatted_image=conditioning_image,
        coverage_mask=coverage_mask,
        target_plucker_map=target_plucker_norm,
        output_path=output_path,
        mode=mode,
        cosmos_model=cosmos_model,
        num_frames=num_frames or 57,
    )
    video_path = output_path.with_suffix(".mp4")

    # Camera-centre distance to best reference (spatial retrieval quality metric)
    retrieval_dist_m: Optional[float] = None
    if T_world_ref is not None:
        t_ref = T_world_ref[:3, 3]
        t_target = target_T_world_camera[:3, 3]
        retrieval_dist_m = float(np.linalg.norm(t_ref - t_target))

    return {
        "status": "completed",
        "output_path": str(output_path),
        "reference_used": {
            "reference_id": best_ref.get("reference_id"),
            "capture_id": best_ref.get("capture_id"),
            "frame_id": best_ref.get("frame_id"),
        },
        "retrieved_references": retrieved_references,
        "lookahead_references": lookahead_references,
        "coverage_frac": round(coverage_frac, 3),
        "retrieval_dist_m": round(retrieval_dist_m, 3) if retrieval_dist_m is not None else None,
        "mode": mode,
        "video_path": str(video_path) if video_path.is_file() else None,
        "conditioning": {
            "previous_tail_path": str(previous_tail_path) if previous_tail_path and previous_tail_path.is_file() else None,
            "previous_tail_alpha": previous_tail_alpha if previous_tail_path else 0.0,
        },
    }


def synthesize_route(
    *,
    site_id: str,
    storage_root: Path,
    bucket: str,
    target_poses: List[np.ndarray],      # ordered list of [4, 4] poses along desired route
    target_intrinsics: Dict[str, float],
    target_h: int,
    target_w: int,
    output_dir: Path,
    mode: str = "splat_only",
    depth_scale: float = 0.001,
    fill_holes: bool = True,
    cosmos_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Synthesise a sequence of views along a route (list of ordered poses).
    Saves each frame as {output_dir}/{frame_idx:06d}.jpg.
    Returns summary with per-frame coverage and retrieval distances.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_results: List[Dict[str, Any]] = []
    for i, T in enumerate(target_poses):
        frame_path = output_dir / f"{i:06d}.jpg"
        result = synthesize_view(
            site_id=site_id,
            storage_root=storage_root,
            bucket=bucket,
            target_T_world_camera=T,
            target_intrinsics=target_intrinsics,
            target_h=target_h,
            target_w=target_w,
            output_path=frame_path,
            mode=mode,
            cosmos_model=cosmos_model,
            depth_scale=depth_scale,
            fill_holes=fill_holes,
        )
        result["frame_index"] = i
        frame_results.append(result)

    completed = [r for r in frame_results if r.get("status") == "completed"]
    mean_coverage = float(np.mean([r["coverage_frac"] for r in completed])) if completed else 0.0

    # Stitch frames into video
    if completed and mode == "splat_only":
        _stitch_frames(output_dir, output_dir / "route.mp4")

    return {
        "status": "completed",
        "frame_count": len(target_poses),
        "frames_synthesised": len(completed),
        "mean_coverage_frac": round(mean_coverage, 3),
        "output_dir": str(output_dir),
        "frames": frame_results,
    }


# ---------------------------------------------------------------------------
# Reference frame loading
# ---------------------------------------------------------------------------


def _load_ref_image(
    rec: Dict[str, Any],
    *,
    storage_root: Path,
    bucket: str,
) -> Optional[np.ndarray]:
    from PIL import Image

    frame_uri = rec.get("frame_uri")
    if not frame_uri:
        return None
    local = _uri_to_local(frame_uri, storage_root=storage_root, bucket=bucket)
    if local is None or not local.is_file():
        return None
    try:
        image = Image.open(local).convert("RGB")
        intr = rec.get("intrinsics") or {}
        target_width = int(intr.get("width") or 0)
        target_height = int(intr.get("height") or 0)
        if (
            target_width > 0
            and target_height > 0
            and image.size == (target_height, target_width)
        ):
            # ARKit-derived reference frames may be materialized in display
            # orientation (portrait), while the indexed intrinsics/depth remain
            # in encoded camera orientation (landscape). Rotate back so RGB,
            # depth, and intrinsics share the same pixel frame.
            image = image.transpose(Image.Transpose.ROTATE_270)
        return np.array(image)
    except Exception:
        return None


def _reference_summary(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "reference_id": str(rec.get("reference_id") or ""),
        "capture_id": str(rec.get("capture_id") or ""),
        "frame_id": str(rec.get("frame_id") or ""),
        "frame_uri": str(rec.get("frame_uri") or ""),
    }


def _blend_previous_tail(
    warped_image: np.ndarray,
    *,
    previous_tail_path: Optional[Path],
    alpha: float,
) -> np.ndarray:
    if previous_tail_path is None:
        return warped_image
    tail_path = Path(previous_tail_path)
    if not tail_path.is_file():
        return warped_image
    try:
        previous_tail = Image.open(tail_path).convert("RGB").resize(
            (warped_image.shape[1], warped_image.shape[0]),
            Image.Resampling.LANCZOS,
        )
        previous_tail_np = np.array(previous_tail, dtype=np.float32)
        warped_np = warped_image.astype(np.float32)
        blend_alpha = float(min(0.9, max(0.0, alpha)))
        blended = previous_tail_np * blend_alpha + warped_np * (1.0 - blend_alpha)
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)
    except Exception:
        return warped_image


def _load_ref_depth(
    rec: Dict[str, Any],
    *,
    storage_root: Path,
    bucket: str,
    depth_scale: float,
) -> Optional[np.ndarray]:
    depth_uri = rec.get("depth_uri")
    if not depth_uri:
        return None
    local = _uri_to_local(depth_uri, storage_root=storage_root, bucket=bucket)
    if local is None or not local.is_file():
        return None
    try:
        depth = load_depth_png(local, depth_scale=depth_scale)
    except Exception:
        return None

    # ARKit depth maps are captured at 256×192 while reference frames are
    # full-resolution video frames (e.g. 1920×1440).  depth_splat() requires
    # depth.shape == ref_image.shape[:2], so upsample here using NEAREST to
    # preserve hard depth edges at object boundaries.
    intr = rec.get("intrinsics") or {}
    tgt_w = int(intr.get("width") or depth.shape[1])
    tgt_h = int(intr.get("height") or depth.shape[0])
    if depth.shape != (tgt_h, tgt_w):
        from PIL import Image as PILImage
        depth_mm = (depth / depth_scale).clip(0, 65535).astype(np.uint16)
        depth_img = PILImage.fromarray(depth_mm, mode="I;16")
        depth_img = depth_img.resize((tgt_w, tgt_h), resample=PILImage.NEAREST)
        depth = np.array(depth_img, dtype=np.float32) * depth_scale

    return depth


def _effective_pose(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    T_raw = rec.get("T_world_camera")
    if T_raw is None:
        return None
    T = np.array(T_raw, dtype=np.float64)
    if T.shape != (4, 4):
        return None
    site_xform = rec.get("site_frame_transform")
    if site_xform is not None:
        S = np.array(site_xform, dtype=np.float64)
        if S.shape == (4, 4):
            return S @ T
    return T


def _uri_to_local(uri: str, *, storage_root: Path, bucket: str) -> Optional[Path]:
    if not uri.startswith("gs://"):
        return Path(uri)
    remainder = uri[5:]
    bkt, _, key = remainder.partition("/")
    candidate = storage_root / bkt / key
    if candidate.is_file():
        return candidate
    flat = storage_root / key
    if flat.is_file():
        return flat
    return candidate


# ---------------------------------------------------------------------------
# Video stitching
# ---------------------------------------------------------------------------


def _stitch_frames(frames_dir: Path, output_path: Path, fps: int = 10) -> None:
    """Stitch numbered JPEG frames into an MP4 using FFmpeg (best-effort)."""
    import subprocess
    pattern = str(frames_dir / "%06d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        pass  # ffmpeg not installed; frame JPEGs are the primary deliverable


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthesise a novel view from a Blueprint site retrieval index"
    )
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--storage-root", default=os.getenv("GCS_ROOT", "/mnt/gcs"))
    parser.add_argument("--bucket", default=(os.getenv("PIPELINE_BUCKET") or "").strip() or None)
    parser.add_argument(
        "--target-pose",
        required=True,
        help="4x4 T_world_camera JSON array in site frame (after alignment)",
    )
    parser.add_argument(
        "--target-intrinsics",
        required=True,
        help='JSON object: {"fx":1462,"fy":1462,"cx":960,"cy":720}',
    )
    parser.add_argument("--target-height", type=int, default=1440)
    parser.add_argument("--target-width", type=int, default=1920)
    parser.add_argument("--output", required=True, help="Output JPEG path")
    parser.add_argument(
        "--mode",
        default="splat_only",
        choices=["splat_only", "cosmos_i2w"],
    )
    parser.add_argument("--k", type=int, default=1, help="Number of reference frames to retrieve")
    parser.add_argument(
        "--query-mode",
        default="spatial",
        choices=["spatial", "embedding", "hybrid"],
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.001,
        help="Multiply raw depth PNG values by this to get metres (default 0.001 for 16-bit mm)",
    )
    parser.add_argument("--no-fill-holes", action="store_true")

    args = parser.parse_args(argv)

    if not args.bucket:
        print("ERROR: --bucket is required (or set PIPELINE_BUCKET env var)", file=sys.stderr)
        return 1

    try:
        T = np.array(json.loads(args.target_pose), dtype=np.float64)
        if T.shape != (4, 4):
            print(f"ERROR: --target-pose must be a 4x4 matrix, got shape {T.shape}", file=sys.stderr)
            return 1
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: Invalid --target-pose JSON: {e}", file=sys.stderr)
        return 1

    try:
        intrinsics = json.loads(args.target_intrinsics)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid --target-intrinsics JSON: {e}", file=sys.stderr)
        return 1

    result = synthesize_view(
        site_id=args.site_id,
        storage_root=Path(args.storage_root),
        bucket=args.bucket,
        target_T_world_camera=T,
        target_intrinsics=intrinsics,
        target_h=args.target_height,
        target_w=args.target_width,
        output_path=Path(args.output),
        mode=args.mode,
        k=args.k,
        query_mode=args.query_mode,
        depth_scale=args.depth_scale,
        fill_holes=not args.no_fill_holes,
    )

    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
