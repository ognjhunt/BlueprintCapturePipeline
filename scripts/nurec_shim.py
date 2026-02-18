#!/usr/bin/env python3
"""NuRec reconstruction shim using COLMAP + 3DGRUT + Fixer + SAM3.

This script replaces the external NuRec service for local/VM runs.
It takes a video file, runs Structure-from-Motion via COLMAP, trains
3DGRUT (3D Gaussian Unscented Transform) for neural reconstruction,
optionally refines renders with NVIDIA Fixer, runs SAM3 for object
detection (replacing ARKit), and produces the required pipeline outputs:
  - export_last.usdz  (neural scene for Isaac Sim)
  - export_last.ply   (Gaussian splat point cloud)
  - nvblox_mesh.ply   (collision mesh from dense reconstruction)
  - occupancy.bin     (voxel occupancy grid)
  - object_point_cloud_index.json  (SAM3-detected objects for swap pipeline)

Usage as NUREC_PIPELINE_COMMAND:
  export NUREC_PIPELINE_COMMAND="python3 /app/scripts/nurec_shim.py \
    --job-spec {JOB_SPEC_PATH} --output-dir {NUREC_OUTPUT_DIR} \
    --raw-prefix {RAW_PREFIX_URI}"

Optional Fixer routing:
  --skip-fixer                     # disable stage 5
  --fixer-mode auto|local|h100    # default auto (H100 first, then local)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ---------------------------------------------------------------------------
# Configuration (paths set by VM provisioning / Docker snapshot)
# ---------------------------------------------------------------------------
THREEDGRUT_DIR = os.getenv("THREEDGRUT_DIR", "/opt/3dgrut")
FIXER_DIR = os.getenv("FIXER_DIR", "/opt/Fixer")
FIXER_WEIGHTS_DIR = os.getenv("FIXER_WEIGHTS_DIR", "/opt/Fixer/weights")
DEFAULT_FIXER_H100_SCRIPT = os.getenv("FIXER_H100_SCRIPT", "/app/scripts/fixer_h100_stage.sh")


def _log(msg: str) -> None:
    print(f"[nurec-shim] {msg}", flush=True)


def _run(cmd: list[str] | str, **kwargs) -> subprocess.CompletedProcess:
    _log(f"  $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    # Ensure headless operation for COLMAP (no Qt GUI)
    env = kwargs.pop("env", None)
    if env is None:
        env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    result = subprocess.run(
        cmd, shell=isinstance(cmd, str), text=True,
        capture_output=True, env=env, **kwargs,
    )
    if result.returncode != 0:
        _log(f"  STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"Command failed (code {result.returncode}): {cmd}")
    return result


# ---------------------------------------------------------------------------
# Stage 1: Frame Extraction
# ---------------------------------------------------------------------------
def extract_frames(video_path: Path, frames_dir: Path,
                   max_frames: int = 300, target_fps: int = 5) -> int:
    """Extract frames from video at reduced FPS for SfM."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Extracting frames from {video_path} at {target_fps} fps (max {max_frames})...")
    _run([
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={target_fps}",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        str(frames_dir / "frame_%05d.jpg"),
    ])
    count = len(list(frames_dir.glob("frame_*.jpg")))
    _log(f"Extracted {count} frames.")
    return count


# ---------------------------------------------------------------------------
# Stage 2: COLMAP SfM
# ---------------------------------------------------------------------------
def _colmap_has_cuda() -> bool:
    """Best-effort detection for CUDA-enabled COLMAP binary."""
    def _query_output(args: list[str]) -> str:
        result = subprocess.run(
            args,
            check=False,
            text=True,
            capture_output=True,
        )
        return (result.stdout + "\n" + result.stderr).lower()

    try:
        output = _query_output(["colmap", "version"])
    except FileNotFoundError:
        _log("WARNING: COLMAP not found in PATH")
        return False

    if "command `version` not recognized" in output:
        # Older COLMAP builds print CUDA status in the main banner/help.
        output = _query_output(["colmap", "help"])
        if "without cuda" not in output and "with cuda" not in output:
            output = _query_output(["colmap"])

    if "without cuda" in output:
        return False
    if "with cuda" in output:
        return True
    return ("cuda" in output) and ("without cuda" not in output)


def _colmap_supports_option(subcommand: str, option_name: str) -> bool:
    """Return True if a COLMAP subcommand help includes the given option."""
    try:
        result = subprocess.run(
            ["colmap", subcommand, "-h"],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return False

    output = (result.stdout + "\n" + result.stderr).lower()
    return option_name.lower() in output


def run_colmap_sfm(frames_dir: Path, workspace: Path, *, sift_use_gpu: bool) -> Path:
    """Run COLMAP Structure-from-Motion pipeline."""
    db_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    sift_gpu_flag = "1" if sift_use_gpu else "0"
    feature_gpu_option = (
        "--FeatureExtraction.use_gpu"
        if _colmap_supports_option("feature_extractor", "--FeatureExtraction.use_gpu")
        else "--SiftExtraction.use_gpu"
    )
    matching_gpu_option = (
        "--FeatureMatching.use_gpu"
        if _colmap_supports_option("sequential_matcher", "--FeatureMatching.use_gpu")
        else "--SiftMatching.use_gpu"
    )

    _log(f"Running COLMAP feature extraction (SIFT GPU={sift_gpu_flag})...")
    _run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--SiftExtraction.max_num_features", "8192",
        feature_gpu_option, sift_gpu_flag,
    ])

    _log(f"Running COLMAP sequential matching (SIFT GPU={sift_gpu_flag})...")
    _run([
        "colmap", "sequential_matcher",
        "--database_path", str(db_path),
        "--SequentialMatching.overlap", "10",
        matching_gpu_option, sift_gpu_flag,
    ])

    _log("Running COLMAP sparse reconstruction (mapper)...")
    _run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse_dir),
    ])

    # Find the best reconstruction (most registered images)
    recon_dirs = sorted(d for d in sparse_dir.iterdir() if d.is_dir())
    if not recon_dirs:
        raise RuntimeError("COLMAP mapper produced no reconstruction")

    best_dir = recon_dirs[0]
    best_count = 0
    for d in recon_dirs:
        images_bin = d / "images.bin"
        if images_bin.exists() and images_bin.stat().st_size >= 8:
            n_images = struct.unpack("<Q", images_bin.read_bytes()[:8])[0]
            _log(f"  {d.name}: {n_images} registered images")
            if n_images > best_count:
                best_count = n_images
                best_dir = d
        else:
            _log(f"  {d.name}: no images.bin")

    _log(f"Selected reconstruction: {best_dir} ({best_count} images)")
    return best_dir


# ---------------------------------------------------------------------------
# Stage 3: COLMAP Undistortion (required for 3DGRUT - PINHOLE only)
# ---------------------------------------------------------------------------
def run_colmap_undistort(frames_dir: Path, sparse_dir: Path,
                         workspace: Path) -> Path:
    """Undistort images to convert camera model to PINHOLE for 3DGRUT."""
    undistorted_dir = workspace / "undistorted"
    undistorted_dir.mkdir(parents=True, exist_ok=True)

    _log("Running COLMAP image undistortion (SIMPLE_RADIAL → PINHOLE)...")
    _run([
        "colmap", "image_undistorter",
        "--image_path", str(frames_dir),
        "--input_path", str(sparse_dir),
        "--output_path", str(undistorted_dir),
        "--output_type", "COLMAP",
        "--max_image_size", "1920",
    ])

    # 3DGRUT expects sparse/0/ but undistorter puts files in sparse/
    sparse_0_dir = undistorted_dir / "sparse" / "0"
    if not sparse_0_dir.exists():
        sparse_0_dir.mkdir(parents=True, exist_ok=True)
        sparse_flat = undistorted_dir / "sparse"
        for f in ["cameras.bin", "images.bin", "points3D.bin"]:
            src = sparse_flat / f
            if src.exists():
                src.rename(sparse_0_dir / f)

    _log(f"Undistorted output at: {undistorted_dir}")
    return undistorted_dir


# ---------------------------------------------------------------------------
# Stage 4: 3DGRUT Training → USDZ + PLY export
# ---------------------------------------------------------------------------
def run_3dgrut_training(undistorted_dir: Path, output_dir: Path,
                         n_iterations: int = 7000) -> dict:
    """Train 3DGRUT on undistorted COLMAP data and export USDZ + PLY."""
    threedgrut_dir = Path(THREEDGRUT_DIR)
    train_script = threedgrut_dir / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(f"3DGRUT not found at {threedgrut_dir}")

    grut_out = output_dir / "3dgrut"
    grut_out.mkdir(parents=True, exist_ok=True)

    _log(f"Starting 3DGRUT training ({n_iterations} iterations)...")
    _run([
        sys.executable, str(train_script),
        "--config-name", "apps/colmap_3dgut_mcmc",
        f"path={undistorted_dir}/",
        f"out_dir={grut_out}/",
        "experiment_name=nurec_scene",
        "export_usdz.enabled=true",
        "export_usdz.apply_normalizing_transform=true",
        "export_ply.enabled=true",
        f"n_iterations={n_iterations}",
        "with_gui=false",
        "with_viser_gui=false",
        "num_workers=4",
    ], cwd=str(threedgrut_dir))

    # Find the output directory (3DGRUT creates a nested structure)
    experiment_dirs = list(grut_out.rglob("export_last.usdz"))
    if not experiment_dirs:
        raise RuntimeError("3DGRUT did not produce export_last.usdz")

    result_dir = experiment_dirs[0].parent
    _log(f"3DGRUT output at: {result_dir}")

    # Read metrics if available
    metrics = {}
    metrics_file = result_dir / "metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        _log(f"  PSNR: {metrics.get('mean_psnr', 'N/A'):.2f}")
        _log(f"  SSIM: {metrics.get('mean_ssim', 'N/A'):.3f}")
        _log(f"  LPIPS: {metrics.get('mean_lpips', 'N/A'):.3f}")

    return {
        "result_dir": result_dir,
        "usdz": result_dir / "export_last.usdz",
        "ply": result_dir / "export_last.ply",
        "ingp": result_dir / "export_last.ingp",
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Stage 5: Fixer image refinement (optional, requires Cosmos/TE)
# ---------------------------------------------------------------------------
def _has_image_outputs(directory: Path) -> bool:
    if not directory.exists():
        return False
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.exr")
    return any(any(directory.rglob(pattern)) for pattern in patterns)


def _run_fixer_local_stage(renders_dir: Path, fixed_dir: Path) -> bool:
    """Run Fixer locally on the current machine."""
    fixer_dir = Path(FIXER_DIR)
    fixer_weights = Path(FIXER_WEIGHTS_DIR)
    inference_script = fixer_dir / "src" / "inference_pretrained_model.py"
    pretrained_path = fixer_weights / "pretrained" / "pretrained_fixer.pkl"

    if not inference_script.exists():
        _log("WARNING: Fixer source not found locally; skipping local Fixer")
        return False
    if not pretrained_path.exists():
        _log("WARNING: Fixer pretrained weights not found locally; skipping local Fixer")
        return False

    fixed_dir.mkdir(parents=True, exist_ok=True)
    _log("Running Fixer image refinement locally...")
    _run(
        [
            sys.executable,
            str(inference_script),
            "--input_folder",
            str(renders_dir),
            "--output_folder",
            str(fixed_dir),
            "--pretrained_path",
            str(pretrained_path),
        ],
        cwd=str(fixer_dir / "src"),
    )
    if not _has_image_outputs(fixed_dir):
        _log("WARNING: Fixer completed but produced no refined images")
        return False
    return True


def _run_fixer_h100_stage(
    renders_dir: Path,
    fixed_dir: Path,
    *,
    h100_script: Path,
    h100_instance_id: str,
    h100_keep_instance: bool,
    h100_max_hourly: float,
    h100_disk_gb: int,
) -> bool:
    """Run Fixer on a remote H100 stage runner (Vast.ai script)."""
    if not h100_script.exists():
        _log(f"WARNING: H100 Fixer script not found: {h100_script}")
        return False

    fixed_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash",
        str(h100_script),
        "--input-renders",
        str(renders_dir),
        "--output-dir",
        str(fixed_dir),
        "--max-hourly",
        str(h100_max_hourly),
        "--disk-gb",
        str(h100_disk_gb),
    ]
    if h100_instance_id:
        cmd.extend(["--instance-id", h100_instance_id])
    if h100_keep_instance:
        cmd.append("--keep-instance")

    _log("Running Fixer on H100 stage runner...")
    _run(cmd)
    if not _has_image_outputs(fixed_dir):
        _log("WARNING: H100 Fixer stage completed but no refined images were returned")
        return False
    return True


def run_fixer_refinement(
    renders_dir: Path,
    output_dir: Path,
    *,
    mode: str = "auto",
    h100_script: Path = Path(DEFAULT_FIXER_H100_SCRIPT),
    h100_instance_id: str = "",
    h100_keep_instance: bool = False,
    h100_max_hourly: float = 2.50,
    h100_disk_gb: int = 80,
) -> Path:
    """Run NVIDIA Fixer refinement using local or H100 backend.

    Modes:
      - auto: try H100 stage first, then local, then skip
      - h100: try only H100 stage
      - local: try only local stage
    """
    fixed_dir = output_dir / "fixer_output"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    mode_normalized = mode.strip().lower()

    if mode_normalized not in {"auto", "h100", "local"}:
        _log(f"WARNING: Unknown fixer mode '{mode}', falling back to auto")
        mode_normalized = "auto"

    if mode_normalized in {"auto", "h100"}:
        try:
            if _run_fixer_h100_stage(
                renders_dir,
                fixed_dir,
                h100_script=h100_script,
                h100_instance_id=h100_instance_id,
                h100_keep_instance=h100_keep_instance,
                h100_max_hourly=h100_max_hourly,
                h100_disk_gb=h100_disk_gb,
            ):
                _log(f"Fixer output at: {fixed_dir} (backend=h100)")
                return fixed_dir
        except RuntimeError as exc:
            _log(f"WARNING: H100 Fixer stage failed ({exc})")
        if mode_normalized == "h100":
            _log("WARNING: H100 Fixer requested but unavailable; using unrefined renders")
            return renders_dir

    if mode_normalized in {"auto", "local"}:
        try:
            if _run_fixer_local_stage(renders_dir, fixed_dir):
                _log(f"Fixer output at: {fixed_dir} (backend=local)")
                return fixed_dir
        except RuntimeError as exc:
            _log(f"WARNING: Local Fixer stage failed ({exc})")

    _log("WARNING: Fixer unavailable; using unrefined renders")
    return renders_dir


# ---------------------------------------------------------------------------
# Stage 6: Dense reconstruction → collision mesh (nvblox_mesh.ply)
# ---------------------------------------------------------------------------
def _read_ply_mesh_counts(ply_path: Path) -> tuple[int, int]:
    """Read vertex/face counts from PLY header without external dependencies."""
    with open(ply_path, "rb") as f:
        first = f.readline().decode("ascii", errors="ignore").strip().lower()
        if first != "ply":
            raise RuntimeError(f"Invalid PLY header in {ply_path}")

        vertex_count = 0
        face_count = 0
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF while reading PLY header: {ply_path}")
            text = line.decode("ascii", errors="ignore").strip().lower()
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            elif text.startswith("element face "):
                face_count = int(text.split()[-1])
            elif text == "end_header":
                break

    return vertex_count, face_count


def _validate_collision_mesh(output_ply: Path) -> None:
    """Hard quality gate for collision meshes."""
    if not output_ply.exists() or output_ply.stat().st_size == 0:
        raise RuntimeError(f"Collision mesh missing or empty: {output_ply}")
    vertex_count, face_count = _read_ply_mesh_counts(output_ply)
    if face_count <= 0:
        raise RuntimeError(
            f"Collision mesh has no faces ({vertex_count} vertices, {face_count} faces): {output_ply}"
        )
    _log(f"  Collision mesh validated: {vertex_count} vertices, {face_count} faces")


def _read_ply_vertex_count(ply_path: Path) -> int:
    """Read vertex count from PLY header (works for point clouds or meshes)."""
    vertex_count, _ = _read_ply_mesh_counts(ply_path)
    return vertex_count


def run_dense_reconstruction(frames_dir: Path, sparse_dir: Path,
                              workspace: Path, output_ply: Path) -> str:
    """Run COLMAP dense reconstruction for collision mesh."""
    dense_dir = workspace / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)

    _log("Running COLMAP image undistortion for dense...")
    _run([
        "colmap", "image_undistorter",
        "--image_path", str(frames_dir),
        "--input_path", str(sparse_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ])

    _log("Running COLMAP PatchMatch stereo (GPU-accelerated)...")
    try:
        _run([
            "colmap", "patch_match_stereo",
            "--workspace_path", str(dense_dir),
            "--PatchMatchStereo.geom_consistency", "true",
        ])
    except RuntimeError as exc:
        raise RuntimeError(
            "PatchMatch stereo failed; refusing point-cloud fallback for collision mesh"
        ) from exc

    _log("Running COLMAP stereo fusion...")
    fused_ply = dense_dir / "fused.ply"
    _run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--output_path", str(fused_ply),
    ])

    if fused_ply.exists() and fused_ply.stat().st_size > 0:
        mesh_method = pointcloud_to_mesh(fused_ply, dense_dir, output_ply)
        _validate_collision_mesh(output_ply)
        return mesh_method
    else:
        raise RuntimeError("Dense stereo fusion produced no output mesh candidates")


def _mesh_with_open3d_poisson(fused_ply: Path, output_ply: Path) -> bool:
    """Attempt Open3D Poisson meshing; return True on success."""
    try:
        import open3d as o3d
        import numpy as np
    except ImportError:
        _log("  Open3D unavailable; using COLMAP meshing fallback")
        return False

    force_poisson = _env_flag("OPEN3D_POISSON_FORCE", False)
    max_poisson_points = max(1, _env_int("OPEN3D_POISSON_MAX_POINTS", 900000))
    poisson_depth = max(6, min(12, _env_int("OPEN3D_POISSON_DEPTH", 9)))
    poisson_depth_large = max(6, min(12, _env_int("OPEN3D_POISSON_DEPTH_LARGE", 8)))
    downsample_target = max(0, _env_int("OPEN3D_POISSON_DOWNSAMPLE_TARGET", 450000))

    header_points = 0
    try:
        header_points = _read_ply_vertex_count(fused_ply)
    except Exception as exc:
        _log(f"  WARNING: Could not read fused PLY header count ({exc}); continuing")

    if header_points > 0:
        _log(f"  Fused cloud header points: {header_points}")
        if header_points > max_poisson_points and not force_poisson:
            _log(
                "  Skipping Open3D Poisson due to large fused cloud "
                f"({header_points} > {max_poisson_points}); using COLMAP delaunay fallback"
            )
            return False

    _log("Running Open3D Poisson mesh reconstruction...")
    try:
        pcd = o3d.io.read_point_cloud(str(fused_ply))
        point_count = len(pcd.points)
        _log(f"  Point cloud: {point_count} points")

        if point_count > max_poisson_points and not force_poisson:
            _log(
                "  Skipping Open3D Poisson after load due to point count "
                f"({point_count} > {max_poisson_points}); using COLMAP delaunay fallback"
            )
            return False

        effective_depth = poisson_depth_large if point_count > downsample_target > 0 else poisson_depth
        if point_count > downsample_target > 0:
            ratio = max(0.05, min(1.0, float(downsample_target) / float(max(1, point_count))))
            _log(
                "  Downsampling point cloud for Poisson "
                f"(target={downsample_target}, ratio={ratio:.3f})..."
            )
            pcd = pcd.random_down_sample(ratio)
            point_count = len(pcd.points)
            _log(f"  Downsampled point cloud: {point_count} points")

        if not pcd.has_normals():
            _log("  Estimating normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(30)

        _log(f"  Running Poisson reconstruction (depth={effective_depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=effective_depth, width=0, scale=1.1, linear_fit=False,
        )

        densities_arr = np.asarray(densities)
        density_threshold = np.quantile(densities_arr, 0.05)
        vertices_to_remove = densities_arr < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        _log(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
        o3d.io.write_triangle_mesh(str(output_ply), mesh)
        return True
    except Exception as exc:
        _log(f"  Open3D meshing failed ({exc}); using COLMAP meshing fallback")
        return False


def _mesh_with_colmap_delaunay(dense_dir: Path, output_ply: Path) -> None:
    """Mesh from COLMAP dense workspace using delaunay mesher."""
    input_candidates = [
        dense_dir,
        dense_dir / "sparse",
        dense_dir / "sparse" / "0",
    ]
    tried: list[tuple[Path, str]] = []

    for input_path in input_candidates:
        if not input_path.exists():
            continue
        if output_ply.exists():
            output_ply.unlink()
        _log(f"Running COLMAP delaunay mesher (input={input_path})...")
        try:
            _run([
                "colmap", "delaunay_mesher",
                "--input_path", str(input_path),
                "--output_path", str(output_ply),
            ])
        except RuntimeError as exc:
            tried.append((input_path, str(exc)))
            continue

        if output_ply.exists() and output_ply.stat().st_size > 0:
            return
        tried.append((input_path, "delaunay_mesher completed but output was empty"))

    details = "; ".join(f"{path}: {msg}" for path, msg in tried) or "no valid input path candidates"
    raise RuntimeError(f"COLMAP delaunay mesher failed for all candidates ({details})")


def pointcloud_to_mesh(fused_ply: Path, dense_dir: Path, output_ply: Path) -> str:
    """Convert dense point cloud to collision mesh with robust fallbacks."""
    if _mesh_with_open3d_poisson(fused_ply, output_ply):
        return "poisson_open3d"
    _mesh_with_colmap_delaunay(dense_dir, output_ply)
    return "delaunay_colmap"


# ---------------------------------------------------------------------------
# Stage 7: Occupancy grid from PLY
# ---------------------------------------------------------------------------
def _build_robust_occupancy_grid(xyz, resolution: int):
    """Build occupancy grid with percentile clipping to suppress outliers."""
    import numpy as np

    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise RuntimeError("No valid XYZ points for occupancy generation")

    low_q = np.percentile(xyz, 1.0, axis=0)
    high_q = np.percentile(xyz, 99.0, axis=0)
    robust_mask = np.all((xyz >= low_q) & (xyz <= high_q), axis=1)
    robust_xyz = xyz[robust_mask]

    # If clipping is too aggressive, fall back to all points.
    min_kept = max(1024, int(xyz.shape[0] * 0.25))
    if robust_xyz.shape[0] < min_kept:
        robust_xyz = xyz
        robust_mask = np.ones(xyz.shape[0], dtype=bool)

    bounds_min = robust_xyz.min(axis=0)
    bounds_max = robust_xyz.max(axis=0)
    extent = bounds_max - bounds_min
    max_extent = float(np.max(extent))
    if max_extent <= 1e-6:
        max_extent = 1.0
    voxel_size = max_extent / float(resolution)

    grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    idx = ((robust_xyz - bounds_min) / voxel_size).astype(int)
    idx = np.clip(idx, 0, resolution - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    center = (bounds_min + bounds_max) / 2.0
    stats = {
        "total_points": int(xyz.shape[0]),
        "kept_points": int(robust_xyz.shape[0]),
    }
    return grid, center, float(voxel_size), stats


def generate_occupancy(ply_path: Path, output_bin: Path,
                        resolution: int = 64) -> None:
    """Generate a voxel occupancy grid from the Gaussian splat PLY."""
    _log(f"Generating occupancy grid ({resolution}^3)...")
    try:
        import numpy as np
        from plyfile import PlyData

        ply = PlyData.read(str(ply_path))
        vertices = ply["vertex"]
        xyz = np.column_stack([
            np.array(vertices["x"]),
            np.array(vertices["y"]),
            np.array(vertices["z"]),
        ])

        grid, center, voxel_size, stats = _build_robust_occupancy_grid(xyz, resolution)

        with open(output_bin, "wb") as f:
            f.write(struct.pack("<iii", resolution, resolution, resolution))
            f.write(struct.pack("<fff", *center))
            f.write(struct.pack("<f", voxel_size))
            f.write(grid.tobytes())

        occupied = int(grid.sum())
        _log(f"  Occupancy: {occupied}/{resolution**3} voxels ({100*occupied/resolution**3:.1f}%)")
        _log(f"  Robust occupancy points: {stats['kept_points']}/{stats['total_points']}")

    except ImportError:
        _log("  plyfile not available, trying trimesh...")
        try:
            import trimesh
            import numpy as np

            mesh = trimesh.load(str(ply_path))
            if hasattr(mesh, 'vertices'):
                xyz = np.asarray(mesh.vertices)
            else:
                xyz = np.asarray(mesh.points) if hasattr(mesh, 'points') else np.zeros((1, 3))

            grid, center, voxel_size, stats = _build_robust_occupancy_grid(xyz, resolution)

            with open(output_bin, "wb") as f:
                f.write(struct.pack("<iii", resolution, resolution, resolution))
                f.write(struct.pack("<fff", *center))
                f.write(struct.pack("<f", voxel_size))
                f.write(grid.tobytes())

            occupied = int(grid.sum())
            _log(f"  Occupancy: {occupied}/{resolution**3} voxels ({100*occupied/resolution**3:.1f}%)")
            _log(f"  Robust occupancy points: {stats['kept_points']}/{stats['total_points']}")
        except ImportError:
            _log("  No PLY reader available, writing placeholder occupancy...")
            with open(output_bin, "wb") as f:
                f.write(struct.pack("<iii", 32, 32, 32))
                f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
                f.write(struct.pack("<f", 0.1))
                f.write(b"\x00" * (32 * 32 * 32))


# ---------------------------------------------------------------------------
# Video finder
# ---------------------------------------------------------------------------
def find_video(raw_prefix: str, storage_root: Path) -> Path:
    """Find the video file from the raw prefix path."""
    if raw_prefix.startswith("gs://"):
        parts = raw_prefix.replace("gs://", "").split("/", 1)
        relative = parts[1] if len(parts) == 2 else parts[0]
        raw_dir = storage_root / relative
    else:
        raw_dir = Path(raw_prefix)

    _log(f"Looking for video in: {raw_dir}")

    # If raw_prefix is a file directly
    if raw_dir.is_file():
        return raw_dir

    video_extensions = [".mov", ".MOV", ".mp4", ".MP4", ".m4v", ".avi"]
    for ext in video_extensions:
        videos = list(raw_dir.rglob(f"*{ext}"))
        if videos:
            _log(f"Found video: {videos[0]}")
            return videos[0]

    manifest_path = raw_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        video_uri = manifest.get("video_uri", "")
        if video_uri:
            video_path = raw_dir / video_uri
            if video_path.exists():
                return video_path

    raise FileNotFoundError(f"No video file found in {raw_dir}")


# ---------------------------------------------------------------------------
# Env parsing helpers
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        _log(f"WARNING: Invalid float in {name}={raw!r}; using {default}")
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        _log(f"WARNING: Invalid int in {name}={raw!r}; using {default}")
        return default


def _scene_semantics_fallback_report(
    *,
    requested_environment: str,
    reason: str,
) -> dict:
    requested = str(requested_environment or "").strip().lower()
    explicit = requested in {"warehouse", "kitchen", "bedroom"}
    if explicit:
        resolved = requested
        source = "manual_override"
        prompt_source = "environment_override"
        confidence = 1.0
    else:
        resolved = "default"
        source = "local_auto_fallback"
        prompt_source = "auto_fallback"
        confidence = 0.35
    return {
        "schema_version": "v1",
        "requested_environment": requested or "auto",
        "resolved_environment": resolved,
        "environment_source": source,
        "environment_confidence": confidence,
        "prompt_source": prompt_source,
        "detection_prompts": [],
        "fallback_reason": reason,
    }


def _infer_scene_semantics_report(*, frames_dir: Path, requested_environment: str) -> dict:
    timeout_sec = max(5, _env_int("SCENE_SEMANTICS_TIMEOUT_SEC", 30))
    try:
        from blueprint_pipeline.scene_semantics import infer_scene_semantics
    except Exception as exc:
        return _scene_semantics_fallback_report(
            requested_environment=requested_environment,
            reason=f"scene_semantics_import_failed:{exc}",
        )

    try:
        report = infer_scene_semantics(
            frames_dir=frames_dir,
            requested_environment=requested_environment,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        return _scene_semantics_fallback_report(
            requested_environment=requested_environment,
            reason=f"scene_semantics_inference_failed:{exc}",
        )
    if not isinstance(report, dict):
        return _scene_semantics_fallback_report(
            requested_environment=requested_environment,
            reason="scene_semantics_invalid_payload",
        )
    return report


def _resolve_sam3_settings(
    *,
    environment: str,
    frame_count: int,
    requested_n_frames: int,
    requested_min_frame_detections: int,
) -> tuple[int, int]:
    """Resolve robust SAM3 sampling/filter settings for the current scene."""
    env = environment.strip().lower()
    if env == "warehouse":
        auto_n_frames = 12
        auto_min_detections = 2
    elif env == "kitchen":
        auto_n_frames = 10
        auto_min_detections = 2
    elif env == "bedroom":
        auto_n_frames = 12
        auto_min_detections = 2
    elif env == "auto":
        auto_n_frames = 14
        auto_min_detections = 3
    else:
        auto_n_frames = 8
        auto_min_detections = 2

    # Scale sampling with capture length to avoid sparse sampling on long clips.
    if frame_count > 0:
        auto_n_frames = max(auto_n_frames, min(32, max(8, frame_count // 10)))

    n_frames = requested_n_frames if requested_n_frames > 0 else auto_n_frames
    min_frame_detections = (
        requested_min_frame_detections
        if requested_min_frame_detections > 0
        else auto_min_detections
    )

    if frame_count > 0:
        n_frames = max(1, min(n_frames, frame_count))
    min_frame_detections = max(1, min_frame_detections)
    return n_frames, min_frame_detections


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="NuRec reconstruction shim (COLMAP + 3DGRUT + Fixer)"
    )
    parser.add_argument("--job-spec", required=True, help="Path to nurec_job_spec.json")
    parser.add_argument("--output-dir", required=True, help="NuRec output directory")
    parser.add_argument("--raw-prefix", default="", help="Raw data prefix URI or video path")
    parser.add_argument("--storage-root", default=os.getenv("GCS_ROOT", "/mnt/gcs"))
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames to extract")
    parser.add_argument("--extract-fps", type=int, default=5, help="Frame extraction FPS")
    parser.add_argument("--n-iterations", type=int, default=7000,
                        help="3DGRUT training iterations")
    parser.add_argument(
        "--colmap-sift-gpu",
        default=(os.getenv("COLMAP_SIFT_GPU", "auto").strip().lower() or "auto"),
        choices=["auto", "on", "off"],
        help="SIFT GPU mode for COLMAP feature extraction/matching",
    )
    parser.add_argument("--skip-fixer", action="store_true",
                        help="Skip Fixer image refinement")
    parser.add_argument(
        "--fixer-mode",
        default=os.getenv("FIXER_MODE", "auto"),
        choices=["auto", "local", "h100"],
        help="Fixer backend mode: auto (h100->local), local, or h100 only",
    )
    parser.add_argument(
        "--fixer-h100-script",
        default=os.getenv("FIXER_H100_SCRIPT", DEFAULT_FIXER_H100_SCRIPT),
        help="Path to H100 stage runner script (used for --fixer-mode h100/auto)",
    )
    parser.add_argument(
        "--fixer-h100-instance-id",
        default=os.getenv("FIXER_H100_INSTANCE_ID", ""),
        help="Optional existing Vast.ai instance ID for Fixer H100 stage",
    )
    parser.add_argument(
        "--fixer-h100-keep-instance",
        action="store_true",
        default=_env_flag("FIXER_H100_KEEP_INSTANCE", False),
        help="Keep H100 instance alive after Fixer stage (default destroys temp instance)",
    )
    parser.add_argument(
        "--fixer-h100-max-hourly",
        type=float,
        default=_env_float("FIXER_H100_MAX_HOURLY", 2.50),
        help="Max hourly price when provisioning H100 for Fixer",
    )
    parser.add_argument(
        "--fixer-h100-disk-gb",
        type=int,
        default=_env_int("FIXER_H100_DISK_GB", 80),
        help="Disk size (GB) when provisioning H100 for Fixer",
    )
    parser.add_argument("--skip-dense", action="store_true",
                        help="Skip dense reconstruction (use Gaussian PLY as mesh)")
    parser.add_argument("--skip-sam3", action="store_true",
                        help="Skip SAM3 object detection")
    parser.add_argument("--environment", default="auto",
                        choices=["auto", "default", "warehouse", "kitchen", "bedroom"],
                        help="Environment type for SAM3 detection prompts (auto recommended)")
    parser.add_argument(
        "--sam3-n-frames",
        type=int,
        default=_env_int("SAM3_N_FRAMES", 0),
        help="Frames to sample for SAM3 detection (0=auto)",
    )
    parser.add_argument(
        "--sam3-min-frame-detections",
        type=int,
        default=_env_int("SAM3_MIN_FRAME_DETECTIONS", 0),
        help="Minimum detections per object across frames (0=auto)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_root = Path(args.storage_root)
    workspace = output_dir / "_colmap_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # Load job spec for raw_prefix if not provided
    raw_prefix = args.raw_prefix
    if not raw_prefix:
        spec = json.loads(Path(args.job_spec).read_text(encoding="utf-8"))
        raw_prefix = spec.get("capture", {}).get("raw_prefix_uri", "")

    # -----------------------------------------------------------------------
    # Stage 1: Frame Extraction
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("STAGE 1: Frame Extraction")
    _log("=" * 60)
    video_path = find_video(raw_prefix, storage_root)
    frames_dir = workspace / "frames"
    frame_count = extract_frames(video_path, frames_dir,
                                  args.max_frames, args.extract_fps)

    if frame_count < 10:
        _log(f"WARNING: Only {frame_count} frames extracted. Reconstruction may fail.")

    # -----------------------------------------------------------------------
    # Stage 2: COLMAP SfM
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("STAGE 2: Structure-from-Motion (COLMAP)")
    _log("=" * 60)
    colmap_cuda = _colmap_has_cuda()
    if args.colmap_sift_gpu == "auto":
        sift_use_gpu = colmap_cuda
    elif args.colmap_sift_gpu == "on":
        if not colmap_cuda:
            _log("WARNING: --colmap-sift-gpu=on requested, but COLMAP reports no CUDA. Using CPU.")
            sift_use_gpu = False
        else:
            sift_use_gpu = True
    else:
        sift_use_gpu = False

    _log(f"COLMAP CUDA detected: {colmap_cuda}. Effective SIFT GPU: {sift_use_gpu}.")
    sparse_dir = run_colmap_sfm(frames_dir, workspace, sift_use_gpu=sift_use_gpu)

    # -----------------------------------------------------------------------
    # Stage 3: Undistort for 3DGRUT (PINHOLE cameras required)
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("STAGE 3: Image Undistortion (→ PINHOLE)")
    _log("=" * 60)
    undistorted_dir = run_colmap_undistort(frames_dir, sparse_dir, workspace)

    # -----------------------------------------------------------------------
    # Stage 4: 3DGRUT Training → USDZ + PLY
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("STAGE 4: 3DGRUT Neural Reconstruction")
    _log("=" * 60)
    grut_result = run_3dgrut_training(undistorted_dir, output_dir,
                                       args.n_iterations)

    # Copy 3DGRUT outputs to the expected locations
    usdz_src = grut_result["usdz"]
    ply_src = grut_result["ply"]
    usdz_dst = output_dir / "export_last.usdz"
    ply_dst = output_dir / "export_last.ply"

    if usdz_src != usdz_dst:
        shutil.copy2(str(usdz_src), str(usdz_dst))
    if ply_src != ply_dst:
        shutil.copy2(str(ply_src), str(ply_dst))

    # Also copy INGP checkpoint
    ingp_src = grut_result.get("ingp")
    if ingp_src and ingp_src.exists():
        shutil.copy2(str(ingp_src), str(output_dir / "export_last.ingp"))

    # -----------------------------------------------------------------------
    # Stage 5: Fixer Image Refinement (optional)
    # -----------------------------------------------------------------------
    if not args.skip_fixer:
        _log("=" * 60)
        _log("STAGE 5: Fixer Image Refinement")
        _log("=" * 60)
        _log(f"Fixer backend mode: {args.fixer_mode}")
        # Find rendered images from 3DGRUT training
        renders_dirs = list(grut_result["result_dir"].rglob("renders"))
        if renders_dirs:
            run_fixer_refinement(
                renders_dirs[0],
                output_dir,
                mode=args.fixer_mode,
                h100_script=Path(args.fixer_h100_script),
                h100_instance_id=args.fixer_h100_instance_id.strip(),
                h100_keep_instance=args.fixer_h100_keep_instance,
                h100_max_hourly=args.fixer_h100_max_hourly,
                h100_disk_gb=args.fixer_h100_disk_gb,
            )
        else:
            _log("WARNING: No rendered images found, skipping Fixer")
    else:
        _log("Skipping Fixer refinement (--skip-fixer)")

    # -----------------------------------------------------------------------
    # Stage 6: Dense Reconstruction → Collision Mesh
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("STAGE 6: Collision Mesh (nvblox_mesh.ply)")
    _log("=" * 60)
    mesh_ply = output_dir / "nvblox_mesh.ply"
    if args.skip_dense:
        raise RuntimeError(
            "--skip-dense is incompatible with collision mesh quality gate "
            "(triangulated mesh required)"
        )
    mesh_method = run_dense_reconstruction(frames_dir, sparse_dir, workspace, mesh_ply)
    mesh_method_path = output_dir / "mesh_method.txt"
    mesh_method_path.write_text(f"{mesh_method}\n", encoding="utf-8")
    _log(f"  Collision mesh method: {mesh_method}")
    quality_profile = "delaunay_relaxed" if mesh_method == "delaunay_colmap" else "default"
    quality_profile_path = output_dir / "quality_profile.txt"
    quality_profile_path.write_text(f"{quality_profile}\n", encoding="utf-8")
    _log(f"  Suggested quality profile: {quality_profile}")

    # -----------------------------------------------------------------------
    # Stage 7: Occupancy Grid
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("STAGE 7: Occupancy Grid")
    _log("=" * 60)
    occupancy_bin = output_dir / "occupancy.bin"
    generate_occupancy(ply_dst, occupancy_bin)

    # -----------------------------------------------------------------------
    # Stage 8: SAM3 Object Detection (replaces ARKit)
    # -----------------------------------------------------------------------
    object_index_path = None
    if not args.skip_sam3:
        _log("=" * 60)
        _log("STAGE 8: SAM3 Object Detection")
        _log("=" * 60)
        scene_semantics_report = _infer_scene_semantics_report(
            frames_dir=frames_dir,
            requested_environment=args.environment,
        )
        scene_semantics_path = output_dir / "scene_semantics_report.json"
        scene_semantics_path.write_text(
            json.dumps(scene_semantics_report, indent=2),
            encoding="utf-8",
        )

        resolved_environment = (
            str(scene_semantics_report.get("resolved_environment") or args.environment)
            .strip()
            .lower()
        )
        if resolved_environment not in {"default", "warehouse", "kitchen", "bedroom"}:
            resolved_environment = "default"
        detection_prompts_override = (
            scene_semantics_report.get("detection_prompts")
            if isinstance(scene_semantics_report.get("detection_prompts"), list)
            else None
        )
        prompt_source_override = str(scene_semantics_report.get("prompt_source") or "").strip() or None
        environment_source = str(scene_semantics_report.get("environment_source") or "").strip() or None
        environment_confidence = scene_semantics_report.get("environment_confidence")
        _log(
            "Scene semantics: "
            f"requested={args.environment} resolved={resolved_environment} "
            f"source={environment_source or 'unknown'} "
            f"confidence={environment_confidence if environment_confidence is not None else 'n/a'}"
        )

        sam3_n_frames, sam3_min_frame_detections = _resolve_sam3_settings(
            environment=resolved_environment,
            frame_count=frame_count,
            requested_n_frames=args.sam3_n_frames,
            requested_min_frame_detections=args.sam3_min_frame_detections,
        )
        _log(
            "SAM3 settings: "
            f"n_frames={sam3_n_frames}, "
            f"min_frame_detections={sam3_min_frame_detections}"
        )
        try:
            from sam3_detect import run_sam3_detection

            # Write index to the raw/arkit/objects/ path expected by the pipeline
            index_output = output_dir / "object_point_cloud_index.json"

            # Use COLMAP sparse dir for 3D refinement if available
            colmap_sparse = None
            undist_sparse = workspace / "undistorted" / "sparse" / "0"
            if undist_sparse.exists():
                colmap_sparse = undist_sparse

            # Pass Gaussian PLY for accurate 3D back-projection
            gaussian_ply = ply_dst if ply_dst.exists() else None

            sam3_result = run_sam3_detection(
                frames_dir=frames_dir,
                output_path=index_output,
                environment=resolved_environment,
                detection_prompts_override=detection_prompts_override,
                prompt_source_override=prompt_source_override,
                environment_source=environment_source,
                environment_confidence=environment_confidence,
                colmap_sparse_dir=colmap_sparse,
                gaussian_ply_path=gaussian_ply,
                n_sample_frames=sam3_n_frames,
                min_frame_detections=sam3_min_frame_detections,
            )
            n_objects = len(sam3_result.get("objects", []))
            _log(f"SAM3 detected {n_objects} objects")
            object_index_path = index_output
        except Exception as e:
            _log(f"WARNING: SAM3 detection failed ({e}), no object index generated")
    else:
        _log("Skipping SAM3 detection (--skip-sam3)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("RECONSTRUCTION COMPLETE")
    _log("=" * 60)
    required = ["export_last.usdz", "export_last.ply", "nvblox_mesh.ply", "occupancy.bin"]
    all_ok = True
    for artifact in required:
        path = output_dir / artifact
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            _log(f"  ✓ {artifact}: {size_mb:.1f}MB")
        else:
            _log(f"  ✗ {artifact}: MISSING")
            all_ok = False

    # Optional outputs
    for artifact in [
        "export_last.ingp",
        "object_point_cloud_index.json",
        "scene_semantics_report.json",
        "mesh_method.txt",
        "quality_profile.txt",
    ]:
        path = output_dir / artifact
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            _log(f"  ○ {artifact}: {size_mb:.1f}MB")

    if grut_result.get("metrics"):
        m = grut_result["metrics"]
        _log(f"  Quality: PSNR={m.get('mean_psnr', 0):.2f} "
             f"SSIM={m.get('mean_ssim', 0):.3f} "
             f"LPIPS={m.get('mean_lpips', 0):.3f}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
