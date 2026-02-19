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
  - visual_mesh.glb   (viewer-friendly visual mesh, textured when available)
  - visual_pointcloud.ply (colored point cloud for debugging/inspection)
  - mesh_manifest.json (artifact role manifest: volume/visual/collision)
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
import concurrent.futures
import json
import os
import shutil
import statistics
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ---------------------------------------------------------------------------
# Configuration (paths set by VM provisioning / Docker snapshot)
# ---------------------------------------------------------------------------
THREEDGRUT_DIR = os.getenv("THREEDGRUT_DIR", "/opt/3dgrut")
# 3DGRUT requires Python >=3.11; use THREEDGRUT_PYTHON env var to override,
# defaulting to python3.11 (installed alongside the image's default python3.10).
THREEDGRUT_PYTHON = os.getenv("THREEDGRUT_PYTHON", "python3.11")
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


def _frame_blur_scores(frames_dir: Path) -> list[tuple[Path, float]]:
    """Compute per-frame blur scores using ffmpeg blurdetect (higher is blurrier)."""
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        return []
    ffmpeg_input = str(frames_dir / "frame_%05d.jpg")
    blur_report = frames_dir / ".blurdetect_report.txt"
    try:
        _run([
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-start_number",
            "1",
            "-i",
            ffmpeg_input,
            "-vf",
            f"blurdetect,metadata=mode=print:file={blur_report}",
            "-f",
            "null",
            "-",
        ])
    except Exception as exc:
        _log(f"WARNING: blurdetect failed ({exc}); skipping blur filtering")
        return []

    if not blur_report.is_file() or blur_report.stat().st_size <= 0:
        return []
    text = blur_report.read_text(encoding="utf-8", errors="ignore")
    frame_to_score: dict[int, float] = {}
    current_frame: int | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("frame:"):
            try:
                token = line.split()[0]
                current_frame = int(token.split(":", 1)[1])
            except Exception:
                current_frame = None
            continue
        if line.startswith("lavfi.blur=") and current_frame is not None:
            try:
                frame_to_score[current_frame + 1] = float(line.split("=", 1)[1])
            except Exception:
                pass

    out: list[tuple[Path, float]] = []
    for path in frames:
        try:
            frame_num = int(path.stem.split("_")[-1])
        except Exception:
            continue
        if frame_num in frame_to_score:
            out.append((path, frame_to_score[frame_num]))
    return out


def _apply_blur_frame_filter(frames_dir: Path, *, keep_ratio: float, min_keep: int) -> int:
    """Keep the sharpest subset of extracted frames in-place."""
    entries = _frame_blur_scores(frames_dir)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        return 0
    if not entries:
        return len(frames)

    ratio = max(0.0, min(1.0, float(keep_ratio)))
    if ratio <= 0.0:
        return len(frames)
    keep_target = max(int(min_keep), int(round(len(entries) * ratio)))
    keep_target = max(1, min(len(entries), keep_target))
    sorted_entries = sorted(entries, key=lambda item: item[1])  # lower blur score = sharper
    keep_paths = {path for path, _ in sorted_entries[:keep_target]}
    drop_paths = [path for path in frames if path not in keep_paths]
    if not drop_paths:
        return len(frames)

    dropped = 0
    for path in drop_paths:
        try:
            path.unlink()
            dropped += 1
        except Exception as exc:
            _log(f"WARNING: could not drop blurred frame {path.name} ({exc})")

    remaining = len(sorted(frames_dir.glob("frame_*.jpg")))
    _log(
        f"Blur filter kept {remaining}/{len(frames)} frames "
        f"(dropped={dropped}, keep_ratio={ratio:.2f})"
    )
    return remaining


def _percentiles(values: Sequence[float], percentiles: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {}

    sorted_values = sorted(float(v) for v in values)
    count = len(sorted_values)
    out: Dict[str, float] = {}
    for pct in percentiles:
        if count == 1:
            out[f"p{pct}"] = float(sorted_values[0])
            continue
        rank = max(0.0, min(100.0, float(pct))) / 100.0 * float(count - 1)
        low = int(rank)
        high = min(low + 1, count - 1)
        if low == high:
            out[f"p{pct}"] = float(sorted_values[low])
            continue
        weight = rank - float(low)
        value = (1.0 - weight) * sorted_values[low] + weight * sorted_values[high]
        out[f"p{pct}"] = float(value)
    return out


def _frame_signal_stats(frames_dir: Path) -> Dict[str, Dict[int, float]]:
    """Extract frame-level brightness/motion stats via ffmpeg signalstats."""
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        return {"yavg": {}, "ydif": {}}

    ffmpeg_input = str(frames_dir / "frame_%05d.jpg")
    stats_report = frames_dir / ".signalstats_report.txt"
    try:
        _run([
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-start_number",
            "1",
            "-i",
            ffmpeg_input,
            "-vf",
            f"signalstats,metadata=mode=print:file={stats_report}",
            "-f",
            "null",
            "-",
        ])
    except Exception as exc:
        _log(f"WARNING: signalstats failed ({exc}); capture quality report will omit brightness/motion details")
        return {"yavg": {}, "ydif": {}}

    if not stats_report.is_file() or stats_report.stat().st_size <= 0:
        return {"yavg": {}, "ydif": {}}

    text = stats_report.read_text(encoding="utf-8", errors="ignore")
    current_frame: int | None = None
    yavg: Dict[int, float] = {}
    ydif: Dict[int, float] = {}
    for line in text.splitlines():
        entry = line.strip()
        if entry.startswith("frame:"):
            try:
                token = entry.split()[0]
                current_frame = int(token.split(":", 1)[1]) + 1
            except Exception:
                current_frame = None
            continue
        if current_frame is None:
            continue
        if entry.startswith("lavfi.signalstats.YAVG="):
            try:
                yavg[current_frame] = float(entry.split("=", 1)[1])
            except Exception:
                pass
        elif entry.startswith("lavfi.signalstats.YDIF="):
            try:
                ydif[current_frame] = float(entry.split("=", 1)[1])
            except Exception:
                pass
    return {"yavg": yavg, "ydif": ydif}


def build_capture_quality_report(frames_dir: Path) -> Dict[str, Any]:
    """Compute objective capture-quality stats from extracted frames."""
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    blur_entries = _frame_blur_scores(frames_dir)
    blur_scores = [score for _, score in blur_entries]
    signal = _frame_signal_stats(frames_dir)
    yavg_values = list(signal.get("yavg", {}).values())
    ydif_values = list(signal.get("ydif", {}).values())

    report: Dict[str, Any] = {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "frame_count": int(len(frames)),
        "blur": {
            "count": int(len(blur_scores)),
            "percentiles": _percentiles(blur_scores, [5, 50, 95]),
            "min": float(min(blur_scores)) if blur_scores else None,
            "max": float(max(blur_scores)) if blur_scores else None,
        },
        "brightness": {
            "count": int(len(yavg_values)),
            "percentiles": _percentiles(yavg_values, [5, 50, 95]),
            "dark_frames_yavg_lt_45": int(sum(1 for value in yavg_values if value < 45.0)),
            "bright_frames_yavg_gt_200": int(sum(1 for value in yavg_values if value > 200.0)),
        },
        "motion": {
            "count": int(len(ydif_values)),
            "percentiles": _percentiles(ydif_values, [5, 50, 95]),
            "very_low_change_ydif_lt_1": int(sum(1 for value in ydif_values if value < 1.0)),
        },
        "blurriest_frames": [],
    }

    if blur_entries:
        sorted_blur = sorted(blur_entries, key=lambda item: item[1])
        report["blurriest_frames"] = [
            {"frame": path.name, "score": float(score)}
            for path, score in sorted_blur[:10]
        ]
        report["sharpest_frames"] = [
            {"frame": path.name, "score": float(score)}
            for path, score in sorted(blur_entries, key=lambda item: item[1], reverse=True)[:10]
        ]

    if blur_scores:
        report["blur"]["mean"] = float(statistics.mean(blur_scores))
    if yavg_values:
        report["brightness"]["mean"] = float(statistics.mean(yavg_values))
    if ydif_values:
        report["motion"]["mean"] = float(statistics.mean(ydif_values))
    return report


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


def run_colmap_sfm(
    frames_dir: Path,
    workspace: Path,
    *,
    sift_use_gpu: bool,
    mapper_num_threads: int = 0,
    matcher_mode: str = "sequential",
    sequential_overlap: int = 10,
) -> Path:
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
    matcher = matcher_mode.strip().lower()
    if matcher not in {"sequential", "exhaustive"}:
        _log(f"WARNING: Unknown matcher_mode={matcher_mode!r}; falling back to sequential")
        matcher = "sequential"
    matcher_subcommand = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
    matching_gpu_option = (
        "--FeatureMatching.use_gpu"
        if _colmap_supports_option(matcher_subcommand, "--FeatureMatching.use_gpu")
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

    if matcher == "sequential":
        overlap = max(1, int(sequential_overlap))
        _log(
            f"Running COLMAP sequential matching "
            f"(SIFT GPU={sift_gpu_flag}, overlap={overlap})..."
        )
        _run([
            "colmap", "sequential_matcher",
            "--database_path", str(db_path),
            "--SequentialMatching.overlap", str(overlap),
            matching_gpu_option, sift_gpu_flag,
        ])
    else:
        _log(f"Running COLMAP exhaustive matching (SIFT GPU={sift_gpu_flag})...")
        _run([
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
            matching_gpu_option, sift_gpu_flag,
        ])

    mapper_cmd = [
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse_dir),
    ]
    if mapper_num_threads > 0:
        if _colmap_supports_option("mapper", "--Mapper.num_threads"):
            mapper_cmd.extend(["--Mapper.num_threads", str(mapper_num_threads)])
        else:
            _log("WARNING: COLMAP mapper does not expose --Mapper.num_threads on this build")

    _log("Running COLMAP sparse reconstruction (mapper)...")
    _run(mapper_cmd)

    # Find the best reconstruction (most registered images)
    best_dir, best_count = _select_best_reconstruction(sparse_dir, emit_logs=True)
    if best_dir is None:
        raise RuntimeError("COLMAP mapper produced no reconstruction")

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
        # Use a very large max_image_size to avoid COLMAP's internal rounding
        # truncating the camera params to a different resolution than the output
        # images — which causes 3DGRUT's dimension assertion to fail.
        "--max_image_size", "9999",
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
        THREEDGRUT_PYTHON, str(train_script),
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


def run_dense_reconstruction(
    frames_dir: Path,
    sparse_dir: Path,
    workspace: Path,
    output_ply: Path,
) -> Dict[str, Any]:
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
        return {
            "mesh_method": mesh_method,
            "fused_ply": fused_ply,
            "dense_dir": dense_dir,
        }
    else:
        raise RuntimeError("Dense stereo fusion produced no output mesh candidates")


def _mesh_with_open3d_poisson(fused_ply: Path, output_ply: Path, *, force: bool = False) -> bool:
    """Attempt Open3D Poisson meshing; return True on success."""
    _apply_open3d_thread_overrides()
    try:
        import open3d as o3d
        import numpy as np
    except ImportError:
        _log("  Open3D unavailable; using COLMAP meshing fallback")
        return False

    force_poisson = force or _env_flag("OPEN3D_POISSON_FORCE", False)
    max_poisson_points = max(1, _env_int("OPEN3D_POISSON_MAX_POINTS", 2000000))
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
# Collision mesh hardening (spike filtering + fallback)
# ---------------------------------------------------------------------------
def _collision_spike_metrics(mesh) -> Dict[str, Any]:
    try:
        import numpy as np
    except ImportError:
        return {
            "enabled": False,
            "reason": "numpy_unavailable",
        }

    faces = np.asarray(getattr(mesh, "faces", []))
    vertices = np.asarray(getattr(mesh, "vertices", []))
    if faces.size == 0 or vertices.size == 0:
        return {
            "enabled": True,
            "face_count": 0,
            "long_edge_face_count": 0,
            "long_edge_face_ratio": 0.0,
            "edge_length_m": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0},
            "thresholds": {
                "max_edge_m": max(0.01, _env_float("COLLISION_MAX_EDGE_M", 5.0)),
                "max_long_edge_ratio": max(0.0, _env_float("COLLISION_SPIKE_MAX_RATIO", 0.02)),
            },
        }

    tri = vertices[faces]
    edge_01 = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
    edge_12 = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
    edge_20 = np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)
    edge_all = np.concatenate([edge_01, edge_12, edge_20])

    max_edge_m = max(0.01, _env_float("COLLISION_MAX_EDGE_M", 5.0))
    long_edge_mask = (edge_01 > max_edge_m) | (edge_12 > max_edge_m) | (edge_20 > max_edge_m)
    long_edge_faces = int(long_edge_mask.sum())
    face_count = int(len(faces))
    long_edge_ratio = float(long_edge_faces / float(face_count)) if face_count > 0 else 0.0

    return {
        "enabled": True,
        "face_count": face_count,
        "long_edge_face_count": long_edge_faces,
        "long_edge_face_ratio": long_edge_ratio,
        "edge_length_m": {
            "p50": float(np.percentile(edge_all, 50)),
            "p95": float(np.percentile(edge_all, 95)),
            "p99": float(np.percentile(edge_all, 99)),
            "max": float(edge_all.max()),
        },
        "thresholds": {
            "max_edge_m": max_edge_m,
            "max_long_edge_ratio": max(0.0, _env_float("COLLISION_SPIKE_MAX_RATIO", 0.02)),
        },
    }


def _postprocess_collision_mesh(mesh_path: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema_version": "v1",
        "mesh_path": str(mesh_path),
        "enabled": False,
        "steps": [],
    }
    try:
        import numpy as np
        import trimesh
    except Exception as exc:
        report["reason"] = f"postprocess_deps_unavailable:{exc}"
        return report

    try:
        mesh = trimesh.load_mesh(str(mesh_path), process=True)
        if mesh is None:
            report["reason"] = "failed_to_load_mesh"
            return report

        report["enabled"] = True
        report["before"] = {
            "vertex_count": int(len(getattr(mesh, "vertices", []))),
            "face_count": int(len(getattr(mesh, "faces", []))),
        }

        # Remove tiny disconnected components while preserving the largest piece.
        min_component_faces = max(1, _env_int("COLLISION_MIN_COMPONENT_FACES", 300))
        largest_kept_faces = 0
        if hasattr(mesh, "split"):
            parts = list(mesh.split(only_watertight=False))
            if len(parts) > 1:
                parts_sorted = sorted(parts, key=lambda p: len(getattr(p, "faces", [])), reverse=True)
                largest_kept_faces = int(len(getattr(parts_sorted[0], "faces", [])))
                kept_parts = [parts_sorted[0]]
                for part in parts_sorted[1:]:
                    if len(getattr(part, "faces", [])) >= min_component_faces:
                        kept_parts.append(part)
                if len(kept_parts) != len(parts):
                    mesh = trimesh.util.concatenate(kept_parts)
                    report["steps"].append(
                        {
                            "name": "component_filter",
                            "total_components": int(len(parts)),
                            "kept_components": int(len(kept_parts)),
                            "min_component_faces": min_component_faces,
                            "largest_component_faces": largest_kept_faces,
                        }
                    )

        # Remove pathological long-edge faces.
        max_edge_m = max(0.01, _env_float("COLLISION_MAX_EDGE_M", 5.0))
        faces = np.asarray(mesh.faces)
        vertices = np.asarray(mesh.vertices)
        if faces.size > 0 and vertices.size > 0:
            tri = vertices[faces]
            edge_01 = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
            edge_12 = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
            edge_20 = np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)
            long_edge_mask = (edge_01 > max_edge_m) | (edge_12 > max_edge_m) | (edge_20 > max_edge_m)
            long_edge_faces = int(long_edge_mask.sum())
            if long_edge_faces > 0 and long_edge_faces < len(faces):
                keep_idx = np.flatnonzero(~long_edge_mask)
                mesh = mesh.submesh([keep_idx], append=True, repair=True)
                report["steps"].append(
                    {
                        "name": "spike_face_filter",
                        "long_edge_faces_removed": long_edge_faces,
                        "long_edge_faces_before": int(len(faces)),
                        "max_edge_m": max_edge_m,
                    }
                )

        # trimesh 4.x removed remove_degenerate_faces(); use nondegenerate_faces() mask instead.
        if hasattr(mesh, "remove_degenerate_faces"):
            mesh.remove_degenerate_faces()
        elif hasattr(mesh, "nondegenerate_faces"):
            nd_mask = mesh.nondegenerate_faces()
            if nd_mask is not None and len(nd_mask) > 0:
                mesh.update_faces(nd_mask)
        mesh.remove_unreferenced_vertices()

        smooth_iters = max(0, _env_int("COLLISION_SMOOTH_ITERS", 2))
        if smooth_iters > 0:
            try:
                trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth_iters)
                report["steps"].append({"name": "taubin_smoothing", "iterations": smooth_iters})
            except Exception as exc:
                report["steps"].append({"name": "taubin_smoothing_skipped", "reason": str(exc)})

        mesh.export(str(mesh_path))
        report["after"] = {
            "vertex_count": int(len(getattr(mesh, "vertices", []))),
            "face_count": int(len(getattr(mesh, "faces", []))),
        }
        report["spike_metrics"] = _collision_spike_metrics(mesh)
        return report
    except Exception as exc:
        report["reason"] = f"postprocess_failed:{exc}"
        return report


def _enforce_collision_spike_gate(collision_report: Mapping[str, Any]) -> None:
    metrics = (
        collision_report.get("spike_metrics")
        if isinstance(collision_report.get("spike_metrics"), Mapping)
        else {}
    )
    ratio = float(metrics.get("long_edge_face_ratio", 0.0) or 0.0)
    max_ratio = max(0.0, _env_float("COLLISION_SPIKE_MAX_RATIO", 0.02))
    if ratio > max_ratio:
        raise RuntimeError(
            "Collision spike gate failed: "
            f"long_edge_face_ratio={ratio:.4f} exceeds threshold={max_ratio:.4f}"
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Stage 7: visual mesh exports for generic viewers
# ---------------------------------------------------------------------------
def _build_visual_mesh_quick(fused_ply: Path, output_glb: Path, target_faces: int) -> Dict[str, Any]:
    """Generate a viewer-friendly mesh from dense fused point cloud."""
    _apply_open3d_thread_overrides()
    try:
        import open3d as o3d
        import numpy as np
    except Exception as exc:
        try:
            import trimesh
        except Exception as tri_exc:
            raise RuntimeError(
                f"visual mesh export requires open3d or trimesh ({exc}; {tri_exc})"
            ) from tri_exc

        cloud = trimesh.load(str(fused_ply))
        output_glb.parent.mkdir(parents=True, exist_ok=True)
        cloud.export(str(output_glb))
        return {
            "ok": True,
            "method": "quick_passthrough_trimesh",
            "target_faces": int(target_faces),
        }

    pcd = o3d.io.read_point_cloud(str(fused_ply))
    point_count = len(pcd.points)
    if point_count <= 0:
        raise RuntimeError(f"No points found in fused cloud: {fused_ply}")

    max_points = max(50000, _env_int("VISUAL_MESH_MAX_POINTS", 700000))
    if point_count > max_points:
        ratio = max(0.05, min(1.0, float(max_points) / float(point_count)))
        pcd = pcd.random_down_sample(ratio)
        point_count = len(pcd.points)

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=50)
        )
        pcd.orient_normals_consistent_tangent_plane(50)

    depth = _resolve_visual_mesh_poisson_depth(point_count)
    _log(f"  Visual mesh Poisson depth={depth} (points={point_count})")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=0,
        scale=1.1,
        linear_fit=False,
    )
    if len(mesh.triangles) <= 0:
        raise RuntimeError("Open3D Poisson returned an empty mesh")

    density_quantile = min(0.2, max(0.0, _env_float("VISUAL_MESH_DENSITY_QUANTILE", 0.03)))
    if density_quantile > 0.0:
        densities_arr = np.asarray(densities)
        density_threshold = np.quantile(densities_arr, density_quantile)
        mesh.remove_vertices_by_mask(densities_arr < density_threshold)

    if target_faces > 0 and len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)

    if pcd.has_colors() and len(pcd.colors) > 0 and len(mesh.vertices) > 0:
        tree = o3d.geometry.KDTreeFlann(pcd)
        pcd_colors = np.asarray(pcd.colors)
        vtx = np.asarray(mesh.vertices)
        out_colors = np.zeros((len(vtx), 3), dtype=np.float64)
        for i, vert in enumerate(vtx):
            _, idx, _ = tree.search_knn_vector_3d(vert, 1)
            out_colors[i] = pcd_colors[idx[0]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(out_colors)

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(output_glb), mesh, write_vertex_colors=True)
    if not ok:
        raise RuntimeError(f"Failed to write visual mesh GLB: {output_glb}")
    return {
        "ok": True,
        "method": "quick_poisson_open3d",
        "point_count": int(point_count),
        "faces": int(len(mesh.triangles)),
        "target_faces": int(target_faces),
        "textured": False,
        "texture_image_count": 0,
        "atlas_resolution": 0,
        "uv_coverage_ratio": 0.0,
        "path": str(output_glb),
    }


def _estimate_uv_coverage_ratio(scene) -> float:
    try:
        import numpy as np
    except Exception:
        return 0.0

    total = 0
    valid = 0
    for geometry in getattr(scene, "geometry", {}).values():
        visual = getattr(geometry, "visual", None)
        uv = getattr(visual, "uv", None)
        if uv is None:
            continue
        uv_arr = np.asarray(uv, dtype=np.float64)
        if uv_arr.ndim != 2 or uv_arr.shape[1] < 2:
            continue
        mask = (
            np.isfinite(uv_arr[:, 0])
            & np.isfinite(uv_arr[:, 1])
            & (uv_arr[:, 0] >= 0.0)
            & (uv_arr[:, 0] <= 1.0)
            & (uv_arr[:, 1] >= 0.0)
            & (uv_arr[:, 1] <= 1.0)
        )
        total += int(uv_arr.shape[0])
        valid += int(mask.sum())
    return float(valid / total) if total > 0 else 0.0


def _texture_files(base_dir: Path) -> list[Path]:
    out: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.ktx2"):
        out.extend(sorted(base_dir.glob(ext)))
    return out


def _max_texture_resolution(texture_paths: Sequence[Path]) -> int:
    if not texture_paths:
        return 0
    try:
        from PIL import Image
    except Exception:
        return 0

    max_side = 0
    for path in texture_paths:
        try:
            with Image.open(path) as img:
                width, height = img.size
            max_side = max(max_side, int(width), int(height))
        except Exception:
            continue
    return int(max_side)


def _find_texrecon_output_obj(work_dir: Path, prefix: str) -> Path | None:
    preferred = [
        work_dir / f"{prefix}.obj",
        work_dir / f"{prefix}_out.obj",
        work_dir / f"{prefix}_model.obj",
        work_dir / "model.obj",
    ]
    for candidate in preferred:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    dynamic = sorted(work_dir.glob(f"{prefix}*.obj"), key=lambda p: p.stat().st_size, reverse=True)
    for candidate in dynamic:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _run_texrecon(scene_dir: Path, mesh_obj: Path, out_prefix: str, work_dir: Path) -> tuple[Path | None, str]:
    attempts = [
        [
            "texrecon",
            str(scene_dir),
            str(mesh_obj),
            out_prefix,
            "--keep_unseen_faces",
            "--max_texture_size",
            str(max(512, _env_int("VISUAL_MESH_TEXTURE_SIZE", 4096))),
            "--num_textures",
            str(max(1, _env_int("VISUAL_MESH_TEXTURE_MAX_ATLASES", 2))),
        ],
        [
            "texrecon",
            str(scene_dir),
            str(mesh_obj),
            out_prefix,
        ],
    ]
    errors: list[str] = []
    for cmd in attempts:
        try:
            _run(cmd, cwd=str(work_dir))
        except Exception as exc:
            errors.append(str(exc))
            continue
        output_obj = _find_texrecon_output_obj(work_dir, out_prefix)
        if output_obj is not None:
            return output_obj, ""
        errors.append("texrecon completed but produced no OBJ output")
    return None, "; ".join(errors)[:400]


def _build_visual_mesh_textured_colmap(
    *,
    fused_ply: Path,
    output_glb: Path,
    workspace: Path | None,
    target_faces: int,
) -> Dict[str, Any]:
    if workspace is None:
        return {
            "ok": False,
            "method": "textured_colmap",
            "reason": "workspace_missing",
            "textured": False,
            "texture_image_count": 0,
            "atlas_resolution": 0,
            "uv_coverage_ratio": 0.0,
        }
    if shutil.which("texrecon") is None:
        return {
            "ok": False,
            "method": "textured_colmap",
            "reason": "texrecon_unavailable",
            "textured": False,
            "texture_image_count": 0,
            "atlas_resolution": 0,
            "uv_coverage_ratio": 0.0,
        }

    _apply_open3d_thread_overrides()
    try:
        import numpy as np
        import open3d as o3d
        import trimesh
    except Exception as exc:
        return {
            "ok": False,
            "method": "textured_colmap",
            "reason": f"deps_unavailable:{exc}",
            "textured": False,
            "texture_image_count": 0,
            "atlas_resolution": 0,
            "uv_coverage_ratio": 0.0,
        }

    scene_candidates = [
        workspace / "dense",
        workspace / "undistorted",
    ]
    scene_dir: Path | None = None
    for candidate in scene_candidates:
        if (candidate / "images").is_dir() and (candidate / "sparse").exists():
            scene_dir = candidate
            break
    if scene_dir is None:
        return {
            "ok": False,
            "method": "textured_colmap",
            "reason": "colmap_scene_missing",
            "textured": False,
            "texture_image_count": 0,
            "atlas_resolution": 0,
            "uv_coverage_ratio": 0.0,
        }

    pcd = o3d.io.read_point_cloud(str(fused_ply))
    point_count = int(len(pcd.points))
    if point_count <= 0:
        return {
            "ok": False,
            "method": "textured_colmap",
            "reason": f"empty_fused_cloud:{fused_ply}",
            "textured": False,
            "texture_image_count": 0,
            "atlas_resolution": 0,
            "uv_coverage_ratio": 0.0,
        }

    max_points = max(100000, _env_int("VISUAL_MESH_TEXTURE_MAX_POINTS", 1200000))
    if point_count > max_points:
        ratio = max(0.05, min(1.0, float(max_points) / float(point_count)))
        pcd = pcd.random_down_sample(ratio)
        point_count = int(len(pcd.points))

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=64)
        )
        pcd.orient_normals_consistent_tangent_plane(50)

    depth = max(8, min(12, _env_int("VISUAL_MESH_TEXTURE_POISSON_DEPTH", 11)))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=0,
        scale=1.1,
        linear_fit=False,
    )
    if len(mesh.triangles) <= 0:
        return {
            "ok": False,
            "method": "textured_colmap",
            "reason": "poisson_empty_mesh",
            "textured": False,
            "texture_image_count": 0,
            "atlas_resolution": 0,
            "uv_coverage_ratio": 0.0,
        }

    density_quantile = min(0.2, max(0.0, _env_float("VISUAL_MESH_TEXTURE_DENSITY_QUANTILE", 0.02)))
    if density_quantile > 0.0:
        density_values = np.asarray(densities)
        threshold = np.quantile(density_values, density_quantile)
        mesh.remove_vertices_by_mask(density_values < threshold)
    if target_faces > 0 and len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(int(target_faces))

    texturing_dir = workspace / "visual_texturing"
    if texturing_dir.exists():
        shutil.rmtree(texturing_dir)
    texturing_dir.mkdir(parents=True, exist_ok=True)
    base_mesh_obj = texturing_dir / "mesh_base.obj"
    if not o3d.io.write_triangle_mesh(str(base_mesh_obj), mesh):
        return {
            "ok": False,
            "method": "textured_colmap",
            "reason": f"mesh_write_failed:{base_mesh_obj}",
            "textured": False,
            "texture_image_count": 0,
            "atlas_resolution": 0,
            "uv_coverage_ratio": 0.0,
        }

    out_prefix = "visual_textured"
    textured_obj, texrecon_error = _run_texrecon(scene_dir, base_mesh_obj, out_prefix, texturing_dir)
    if textured_obj is None:
        return {
            "ok": False,
            "method": "textured_colmap",
            "reason": f"texrecon_failed:{texrecon_error or 'unknown'}",
            "textured": False,
            "texture_image_count": 0,
            "atlas_resolution": 0,
            "uv_coverage_ratio": 0.0,
        }

    loaded = trimesh.load(str(textured_obj), force="scene", process=False)
    scene = loaded if hasattr(loaded, "geometry") else loaded.scene()
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_glb))
    texture_paths = _texture_files(texturing_dir)
    atlas_resolution = _max_texture_resolution(texture_paths)
    if atlas_resolution <= 0:
        atlas_resolution = max(512, _env_int("VISUAL_MESH_TEXTURE_SIZE", 4096))

    return {
        "ok": True,
        "method": "textured_colmap_texrecon",
        "path": str(output_glb),
        "faces": int(sum(len(getattr(g, "faces", [])) for g in scene.geometry.values())),
        "point_count": int(point_count),
        "target_faces": int(target_faces),
        "textured": True,
        "texture_image_count": int(len(texture_paths)),
        "atlas_resolution": int(atlas_resolution),
        "uv_coverage_ratio": float(_estimate_uv_coverage_ratio(scene)),
    }


def _build_visual_mesh_gaussian_tsdf(
    *,
    gaussian_ply: Path,
    output_glb: Path,
    target_faces: int,
) -> Dict[str, Any]:
    try:
        from blueprint_pipeline.gaussian_visual_mesh import build_gaussian_visual_mesh
    except Exception as exc:
        return {
            "ok": False,
            "method": "gaussian_tsdf",
            "reason": f"gaussian_visual_mesh_import_failed:{exc}",
        }

    return build_gaussian_visual_mesh(
        gaussian_ply=gaussian_ply,
        output_glb=output_glb,
        target_faces=target_faces,
    )


def build_visual_mesh_artifacts(
    *,
    output_dir: Path,
    fused_ply: Path,
    gaussian_ply: Path,
    workspace: Path | None = None,
) -> Dict[str, Any]:
    enabled = _env_flag("VISUAL_MESH_ENABLED", True)
    target_faces = _env_int("VISUAL_MESH_TARGET_FACES", 0)
    method = (os.getenv("VISUAL_MESH_METHOD", "textured_colmap") or "textured_colmap").strip().lower()

    visual_pointcloud = output_dir / "visual_pointcloud.ply"
    visual_mesh = output_dir / "visual_mesh.glb"
    robust_mesh = output_dir / "visual_mesh_robust.glb"
    report: Dict[str, Any] = {
        "enabled": enabled,
        "configured_method": method,
        "target_faces": target_faces,
        "visual_pointcloud": str(visual_pointcloud),
        "textured": False,
        "texture_image_count": 0,
        "atlas_resolution": 0,
        "uv_coverage_ratio": 0.0,
        "fallback_reason": "",
    }
    if not enabled:
        report["status"] = "disabled"
        return report

    shutil.copy2(str(fused_ply), str(visual_pointcloud))
    robust_report: Dict[str, Any] = {}
    quick_report: Dict[str, Any] = {}
    textured_report: Dict[str, Any] = {}
    fallback_reasons: list[str] = []

    if method == "textured_colmap":
        textured_report = _build_visual_mesh_textured_colmap(
            fused_ply=fused_ply,
            output_glb=visual_mesh,
            workspace=workspace,
            target_faces=target_faces,
        )
        report["textured_colmap"] = textured_report
        if textured_report.get("ok") and visual_mesh.exists():
            report["status"] = "ok"
            report["selected_method"] = str(textured_report.get("method") or "textured_colmap")
            report["visual_mesh"] = str(visual_mesh)
            report["textured"] = bool(textured_report.get("textured", True))
            report["texture_image_count"] = int(textured_report.get("texture_image_count", 0))
            report["atlas_resolution"] = int(textured_report.get("atlas_resolution", 0))
            report["uv_coverage_ratio"] = float(textured_report.get("uv_coverage_ratio", 0.0))
            return report
        reason = str(textured_report.get("reason") or "textured_colmap_failed")
        fallback_reasons.append(reason)

    if method in {"gaussian_tsdf", "textured_colmap"}:
        robust_report = _build_visual_mesh_gaussian_tsdf(
            gaussian_ply=gaussian_ply,
            output_glb=robust_mesh,
            target_faces=target_faces,
        )
        report["robust"] = robust_report
        if robust_report.get("ok") and robust_mesh.exists():
            if robust_mesh != visual_mesh:
                shutil.copy2(str(robust_mesh), str(visual_mesh))
            report["status"] = "ok"
            report["selected_method"] = str(robust_report.get("method") or "gaussian_tsdf")
            report["visual_mesh"] = str(visual_mesh)
            report["visual_mesh_robust"] = str(robust_mesh)
            report["textured"] = bool(robust_report.get("textured", False))
            report["texture_image_count"] = int(robust_report.get("texture_image_count", 0))
            report["atlas_resolution"] = int(robust_report.get("atlas_resolution", 0))
            report["uv_coverage_ratio"] = float(robust_report.get("uv_coverage_ratio", 0.0))
            if fallback_reasons:
                report["fallback_reason"] = "; ".join(fallback_reasons)
            return report
        if method in {"textured_colmap", "gaussian_tsdf"}:
            fallback_reasons.append(str(robust_report.get("reason") or "gaussian_tsdf_failed"))

    quick_report = _build_visual_mesh_quick(
        fused_ply=fused_ply,
        output_glb=visual_mesh,
        target_faces=target_faces,
    )
    report["quick"] = quick_report
    report["status"] = "ok"
    report["selected_method"] = str(quick_report.get("method") or "quick_poisson")
    report["visual_mesh"] = str(visual_mesh)
    report["textured"] = bool(quick_report.get("textured", False))
    report["texture_image_count"] = int(quick_report.get("texture_image_count", 0))
    report["atlas_resolution"] = int(quick_report.get("atlas_resolution", 0))
    report["uv_coverage_ratio"] = float(quick_report.get("uv_coverage_ratio", 0.0))
    if fallback_reasons:
        report["fallback_reason"] = "; ".join(fallback_reasons)
    if robust_mesh.exists():
        report["visual_mesh_robust"] = str(robust_mesh)
    return report


def write_mesh_manifest(
    *,
    output_dir: Path,
    visual_usdz: Path,
    gaussian_ply: Path,
    collision_mesh_ply: Path,
    occupancy: Path,
    visual_report: Mapping[str, Any],
    collision_method: str,
    collision_report: Mapping[str, Any],
) -> Path:
    visual_mesh_path = (
        output_dir / "visual_mesh.glb"
        if (output_dir / "visual_mesh.glb").is_file() and (output_dir / "visual_mesh.glb").stat().st_size > 0
        else None
    )
    visual_pointcloud_path = (
        output_dir / "visual_pointcloud.ply"
        if (output_dir / "visual_pointcloud.ply").is_file()
        and (output_dir / "visual_pointcloud.ply").stat().st_size > 0
        else None
    )
    robust_mesh_path = (
        output_dir / "visual_mesh_robust.glb"
        if (output_dir / "visual_mesh_robust.glb").is_file()
        and (output_dir / "visual_mesh_robust.glb").stat().st_size > 0
        else None
    )

    def _entry(path: Path, *, role: str, kind: str, viewer_hint: str) -> Dict[str, Any]:
        return {
            "path": path.name,
            "role": role,
            "kind": kind,
            "size_bytes": int(path.stat().st_size),
            "viewer_hint": viewer_hint,
        }

    assets = [
        _entry(
            visual_usdz,
            role="volume_visual",
            kind="usdz_nurec_volume",
            viewer_hint="Use Isaac Sim / Omniverse renderer for neural volume visuals",
        ),
        _entry(
            gaussian_ply,
            role="gaussian_pointcloud",
            kind="ply_gaussian",
            viewer_hint="Debug/training artifact, not final viewer mesh",
        ),
        _entry(
            collision_mesh_ply,
            role="collision",
            kind="ply_triangle_mesh",
            viewer_hint="Physics/collision mesh; may look coarse or white in viewers",
        ),
        _entry(
            occupancy,
            role="occupancy",
            kind="binary_voxel_grid",
            viewer_hint="Used for occupancy checks; not a visual asset",
        ),
    ]
    if visual_mesh_path is not None:
        visual_is_textured = bool(visual_report.get("textured", False))
        assets.append(
            _entry(
                visual_mesh_path,
                role="visual",
                kind="glb_triangle_mesh_textured" if visual_is_textured else "glb_triangle_mesh_vertex_color",
                viewer_hint=(
                    "Primary generic-viewer photoreal textured mesh"
                    if visual_is_textured
                    else "Primary generic-viewer asset (vertex color fallback)"
                ),
            )
        )
    if robust_mesh_path is not None:
        assets.append(
            _entry(
                robust_mesh_path,
                role="visual_optional",
                kind="glb_triangle_mesh_vertex_color",
                viewer_hint="Robust visual mesh candidate (gaussian_tsdf mode)",
            )
        )
    if visual_pointcloud_path is not None:
        assets.append(
            _entry(
                visual_pointcloud_path,
                role="visual_pointcloud",
                kind="ply_pointcloud_color",
                viewer_hint="Colored dense point cloud for visual debugging",
            )
        )

    payload = {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "collision_method": collision_method,
        "visual_method": str(visual_report.get("selected_method") or ""),
        "primary_visual_asset": "",
        "viewer_compatibility": [],
        "assets": assets,
        "reports": {
            "visual": dict(visual_report),
            "collision": dict(collision_report),
        },
    }

    visual_preference = (os.getenv("NUREC_VISUAL_PRIMARY", "usdz") or "usdz").strip().lower()
    if visual_preference not in {"usdz", "mesh", "auto"}:
        visual_preference = "usdz"

    has_visual_mesh = visual_mesh_path is not None
    visual_is_textured = bool(visual_report.get("textured", False))
    mesh_compat = "generic_textured_mesh" if visual_is_textured else "fallback_vertex_mesh"
    payload["viewer_compatibility"] = ["omniverse_neural", mesh_compat]
    if visual_preference == "mesh" and has_visual_mesh:
        payload["primary_visual_asset"] = visual_mesh_path.name if visual_mesh_path is not None else visual_usdz.name
    elif visual_preference == "auto" and has_visual_mesh and visual_is_textured:
        payload["primary_visual_asset"] = visual_mesh_path.name if visual_mesh_path is not None else visual_usdz.name
    else:
        payload["primary_visual_asset"] = visual_usdz.name

    manifest_path = output_dir / "mesh_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Stage 8: Occupancy grid from PLY
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


def _env_choice(name: str, default: str, allowed: Sequence[str]) -> str:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in allowed:
        return raw
    _log(f"WARNING: Invalid choice in {name}={raw!r}; using {default}")
    return default


def _quality_profile() -> str:
    return _env_choice("NUREC_QUALITY_PROFILE", "quality_first", ("quality_first", "balanced", "fast"))


def _quality_profile_defaults(profile: str) -> Dict[str, Any]:
    table: Dict[str, Dict[str, Any]] = {
        "quality_first": {
            "max_frames": 450,
            "extract_fps": 6,
            "n_iterations": 12000,
            "colmap_matcher_mode": "exhaustive",
            "colmap_sequential_overlap": 30,
            "resume": False,
        },
        "balanced": {
            "max_frames": 320,
            "extract_fps": 5,
            "n_iterations": 9000,
            "colmap_matcher_mode": "sequential",
            "colmap_sequential_overlap": 20,
            "resume": False,
        },
        "fast": {
            "max_frames": 240,
            "extract_fps": 4,
            "n_iterations": 7000,
            "colmap_matcher_mode": "sequential",
            "colmap_sequential_overlap": 10,
            "resume": True,
        },
    }
    return table.get(profile, table["quality_first"])


def _apply_open3d_thread_overrides() -> None:
    thread_count = max(0, _env_int("OPEN3D_CPU_THREADS", 0))
    if thread_count <= 0:
        return
    value = str(thread_count)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = value


def _resolve_visual_mesh_poisson_depth(point_count: int) -> int:
    base_depth = max(6, min(12, _env_int("VISUAL_MESH_POISSON_DEPTH", 12)))
    large_threshold = max(1, _env_int("VISUAL_MESH_POISSON_LARGE_THRESHOLD", 500000))
    large_depth = max(6, min(base_depth, _env_int("VISUAL_MESH_POISSON_DEPTH_LARGE", 9)))
    return large_depth if point_count > large_threshold else base_depth


def _scene_semantics_fallback_report(
    *,
    requested_environment: str,
    reason: str,
) -> dict:
    requested = str(requested_environment or "").strip().lower()
    explicit = requested in {"warehouse", "kitchen", "bedroom"}
    if explicit:
        resolved = requested
        source = "explicit_hint_fallback"
        prompt_source = "explicit_hint_fallback"
        confidence = 0.7
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


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _has_colmap_model(model_dir: Path) -> bool:
    return all(_is_nonempty_file(model_dir / name) for name in ("cameras.bin", "images.bin", "points3D.bin"))


def _count_extracted_frames(frames_dir: Path) -> int:
    if not frames_dir.is_dir():
        return 0
    return len(list(frames_dir.glob("frame_*.jpg")))


def _read_registered_image_count(model_dir: Path) -> int:
    images_bin = model_dir / "images.bin"
    if not _is_nonempty_file(images_bin):
        return 0
    try:
        return int(struct.unpack("<Q", images_bin.read_bytes()[:8])[0])
    except Exception:
        return 0


def _registration_ratio(*, registered_images: int, extracted_frames: int) -> float:
    denom = max(1, int(extracted_frames))
    return float(max(0, int(registered_images))) / float(denom)


def _select_best_reconstruction(sparse_dir: Path, *, emit_logs: bool = False) -> tuple[Path | None, int]:
    if not sparse_dir.exists():
        return None, 0
    recon_dirs = sorted(d for d in sparse_dir.iterdir() if d.is_dir())
    if not recon_dirs:
        return None, 0

    best_dir: Path | None = None
    best_count = 0
    for recon_dir in recon_dirs:
        images_bin = recon_dir / "images.bin"
        if not _is_nonempty_file(images_bin):
            if emit_logs:
                _log(f"  {recon_dir.name}: no images.bin")
            continue
        try:
            n_images = struct.unpack("<Q", images_bin.read_bytes()[:8])[0]
        except Exception as exc:
            if emit_logs:
                _log(f"  {recon_dir.name}: unreadable images.bin ({exc})")
            continue
        if emit_logs:
            _log(f"  {recon_dir.name}: {n_images} registered images")
        if n_images >= best_count:
            best_count = n_images
            best_dir = recon_dir
    if best_dir is None:
        if emit_logs:
            _log(f"  No readable images.bin found; falling back to {recon_dirs[0].name}")
        return recon_dirs[0], 0
    return best_dir, best_count


def _load_existing_grut_result(output_dir: Path) -> Dict[str, Any] | None:
    usdz_path = output_dir / "export_last.usdz"
    ply_path = output_dir / "export_last.ply"
    if not (_is_nonempty_file(usdz_path) and _is_nonempty_file(ply_path)):
        return None

    grut_root = output_dir / "3dgrut"
    result_dir = output_dir
    if grut_root.exists():
        export_candidates = sorted(
            grut_root.rglob("export_last.usdz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if export_candidates:
            result_dir = export_candidates[0].parent

    metrics: Dict[str, Any] = {}
    metrics_path = result_dir / "metrics.json"
    if _is_nonempty_file(metrics_path):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _log(f"WARNING: Failed to read existing 3DGRUT metrics ({exc})")

    ingp_path = output_dir / "export_last.ingp"
    return {
        "result_dir": result_dir,
        "usdz": usdz_path,
        "ply": ply_path,
        "ingp": ingp_path if _is_nonempty_file(ingp_path) else None,
        "metrics": metrics,
    }


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not _is_nonempty_file(path):
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_existing_visual_report(output_dir: Path) -> Dict[str, Any] | None:
    report_candidates = [
        output_dir / "visual_mesh_report.json",
    ]
    for report_path in report_candidates:
        report = _load_json_dict(report_path)
        if report:
            enabled = bool(report.get("enabled", False))
            if not enabled:
                return report
            visual_mesh = output_dir / "visual_mesh.glb"
            visual_pointcloud = output_dir / "visual_pointcloud.ply"
            if _is_nonempty_file(visual_mesh) and _is_nonempty_file(visual_pointcloud):
                return report

    manifest = _load_json_dict(output_dir / "mesh_manifest.json")
    reports = manifest.get("reports") if isinstance(manifest.get("reports"), dict) else {}
    visual_report = reports.get("visual") if isinstance(reports.get("visual"), dict) else {}
    if visual_report:
        enabled = bool(visual_report.get("enabled", False))
        if not enabled:
            return dict(visual_report)
        visual_mesh = output_dir / "visual_mesh.glb"
        visual_pointcloud = output_dir / "visual_pointcloud.ply"
        if _is_nonempty_file(visual_mesh) and _is_nonempty_file(visual_pointcloud):
            return dict(visual_report)

    visual_mesh = output_dir / "visual_mesh.glb"
    visual_pointcloud = output_dir / "visual_pointcloud.ply"
    if _is_nonempty_file(visual_mesh) and _is_nonempty_file(visual_pointcloud):
        return {
            "enabled": True,
            "configured_method": "resume",
            "selected_method": "resume_existing_artifacts",
            "status": "ok",
            "visual_mesh": str(visual_mesh),
            "visual_pointcloud": str(visual_pointcloud),
        }
    return None


def _save_visual_report(output_dir: Path, visual_report: Mapping[str, Any]) -> None:
    report_path = output_dir / "visual_mesh_report.json"
    report_path.write_text(json.dumps(dict(visual_report), indent=2), encoding="utf-8")


def _run_stage9_sam3(
    *,
    output_dir: Path,
    workspace: Path,
    frames_dir: Path,
    frame_count: int,
    requested_environment: str,
    requested_n_frames: int,
    requested_min_frame_detections: int,
    gaussian_ply: Path,
    resume: bool,
) -> Path | None:
    scene_semantics_path = output_dir / "scene_semantics_report.json"
    index_output = output_dir / "object_point_cloud_index.json"
    if resume and _is_nonempty_file(scene_semantics_path) and _is_nonempty_file(index_output):
        _log("Resuming Stage 9: using existing scene semantics + SAM3 object index")
        return index_output

    scene_semantics_report = _infer_scene_semantics_report(
        frames_dir=frames_dir,
        requested_environment=requested_environment,
    )
    scene_semantics_path.write_text(
        json.dumps(scene_semantics_report, indent=2),
        encoding="utf-8",
    )

    resolved_environment = (
        str(scene_semantics_report.get("resolved_environment") or requested_environment)
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
        f"requested={requested_environment} resolved={resolved_environment} "
        f"source={environment_source or 'unknown'} "
        f"confidence={environment_confidence if environment_confidence is not None else 'n/a'}"
    )

    sam3_n_frames, sam3_min_frame_detections = _resolve_sam3_settings(
        environment=resolved_environment,
        frame_count=frame_count,
        requested_n_frames=requested_n_frames,
        requested_min_frame_detections=requested_min_frame_detections,
    )
    _log(
        "SAM3 settings: "
        f"n_frames={sam3_n_frames}, "
        f"min_frame_detections={sam3_min_frame_detections}"
    )
    try:
        from sam3_detect import run_sam3_detection

        colmap_sparse = None
        undist_sparse = workspace / "undistorted" / "sparse" / "0"
        if undist_sparse.exists():
            colmap_sparse = undist_sparse

        gaussian_ply_path = gaussian_ply if gaussian_ply.exists() else None

        sam3_result = run_sam3_detection(
            frames_dir=frames_dir,
            output_path=index_output,
            environment=resolved_environment,
            detection_prompts_override=detection_prompts_override,
            prompt_source_override=prompt_source_override,
            environment_source=environment_source,
            environment_confidence=environment_confidence,
            colmap_sparse_dir=colmap_sparse,
            gaussian_ply_path=gaussian_ply_path,
            n_sample_frames=sam3_n_frames,
            min_frame_detections=sam3_min_frame_detections,
        )
        n_objects = len(sam3_result.get("objects", []))
        _log(f"SAM3 detected {n_objects} objects")
        return index_output
    except Exception as exc:
        _log(f"WARNING: SAM3 detection failed ({exc}), no object index generated")
        return None


def _run_dependency_preflight(*, check_fused_ssim: bool = True) -> None:
    """Fail fast on known runtime dependency gaps before expensive stages."""
    threedgrut_dir = Path(THREEDGRUT_DIR)
    train_script = threedgrut_dir / "train.py"
    tiny_cuda_header = threedgrut_dir / "thirdparty" / "tiny-cuda-nn" / "include" / "tiny-cuda-nn" / "common.h"
    missing: list[str] = []
    if not train_script.exists():
        missing.append(f"missing 3DGRUT training script: {train_script}")
    if not tiny_cuda_header.exists():
        missing.append(
            "missing tiny-cuda-nn submodule header "
            f"(expected {tiny_cuda_header})"
        )
    if missing:
        details = "; ".join(missing)
        raise RuntimeError(
            "Dependency preflight failed before reconstruction: "
            f"{details}. Bake these dependencies into the runtime image."
        )

    if check_fused_ssim:
        probe = subprocess.run(
            [THREEDGRUT_PYTHON, "-c", "import fused_ssim"],
            check=False,
            text=True,
            capture_output=True,
        )
        if probe.returncode != 0:
            stderr_tail = (probe.stderr or "").strip()[-400:]
            raise RuntimeError(
                "Dependency preflight failed: could not import fused_ssim with "
                f"{THREEDGRUT_PYTHON}. Rebuild fused_ssim against the current torch ABI. "
                f"stderr_tail={stderr_tail!r}"
            )


def _resolve_hf_token() -> str:
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        token = (os.getenv(name) or "").strip()
        if token:
            return token
    return ""


def _sam3_local_cache_present() -> bool:
    # Check for pre-baked weights first (Docker image layer)
    weights_path = Path(os.getenv("SAM3_WEIGHTS_PATH", "/opt/sam3_weights/sam3.pt"))
    if weights_path.is_file():
        return True

    roots: list[Path] = []
    hf_home = (os.getenv("HF_HOME") or "").strip()
    if hf_home:
        roots.append(Path(hf_home))
    roots.append(Path.home() / ".cache" / "huggingface")

    for root in roots:
        if not root.exists():
            continue
        candidates = [root, root / "hub"]
        for candidate in candidates:
            if not candidate.exists():
                continue
            for pattern in ("models--facebook--sam3*", "models--facebook--segment-anything-3*"):
                if any(candidate.glob(pattern)):
                    return True
    return False


def _run_sam3_preflight(*, strict: bool) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "enabled": True,
        "strict": bool(strict),
        "status": "ok",
        "reason": "",
        "token_present": False,
        "cache_present": False,
        "import_ok": False,
    }
    token = _resolve_hf_token()
    cache_present = _sam3_local_cache_present()
    report["token_present"] = bool(token)
    report["cache_present"] = bool(cache_present)

    try:
        __import__("sam3")
        report["import_ok"] = True
    except Exception as exc:
        report["import_ok"] = False
        report["status"] = "skip"
        report["reason"] = f"sam3_import_failed:{exc}"

    if report["status"] == "ok" and not token and not cache_present:
        report["status"] = "skip"
        report["reason"] = "sam3_model_access_unavailable:missing_hf_token_and_cache"

    if strict and report["status"] != "ok":
        reason = str(report.get("reason") or "sam3_preflight_failed")
        raise RuntimeError(
            "SAM3 preflight failed in strict mode: "
            f"{reason}. Provide HF_TOKEN with gated-model access or pre-bake SAM3 weights."
        )
    return report


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> int:
    profile = _quality_profile()
    profile_defaults = _quality_profile_defaults(profile)
    parser = argparse.ArgumentParser(
        description="NuRec reconstruction shim (COLMAP + 3DGRUT + Fixer)"
    )
    parser.add_argument("--job-spec", required=True, help="Path to nurec_job_spec.json")
    parser.add_argument("--output-dir", required=True, help="NuRec output directory")
    parser.add_argument("--raw-prefix", default="", help="Raw data prefix URI or video path")
    parser.add_argument("--storage-root", default=os.getenv("GCS_ROOT", "/mnt/gcs"))
    parser.add_argument(
        "--max-frames",
        type=int,
        default=_env_int("MAX_FRAMES", int(profile_defaults["max_frames"])),
        help="Max frames to extract",
    )
    parser.add_argument(
        "--extract-fps",
        type=int,
        default=_env_int("EXTRACT_FPS", int(profile_defaults["extract_fps"])),
        help="Frame extraction FPS",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=_env_int("N_ITERATIONS", int(profile_defaults["n_iterations"])),
        help="3DGRUT training iterations",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=_env_flag("NUREC_RESUME", bool(profile_defaults["resume"])),
        help="Reuse completed stage outputs from --output-dir when valid",
    )
    parser.add_argument(
        "--dependency-preflight",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("NUREC_DEPENDENCY_PREFLIGHT", True),
        help="Fail fast on missing runtime deps before expensive stages",
    )
    parser.add_argument(
        "--preflight-check-fused-ssim",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("NUREC_PREFLIGHT_CHECK_FUSED_SSIM", True),
        help="Include fused_ssim import ABI check in dependency preflight",
    )
    parser.add_argument(
        "--colmap-sift-gpu",
        default=(os.getenv("COLMAP_SIFT_GPU", "auto").strip().lower() or "auto"),
        choices=["auto", "on", "off"],
        help="SIFT GPU mode for COLMAP feature extraction/matching",
    )
    parser.add_argument(
        "--colmap-mapper-threads",
        type=int,
        default=_env_int("COLMAP_MAPPER_NUM_THREADS", 0),
        help="Mapper CPU threads (0=auto/all available)",
    )
    parser.add_argument(
        "--colmap-matcher-mode",
        default=(
            os.getenv("COLMAP_MATCHER_MODE", str(profile_defaults["colmap_matcher_mode"]))
            .strip()
            .lower()
            or str(profile_defaults["colmap_matcher_mode"])
        ),
        choices=["sequential", "exhaustive"],
        help="COLMAP matcher mode for SfM correspondence search",
    )
    parser.add_argument(
        "--colmap-sequential-overlap",
        type=int,
        default=_env_int("COLMAP_SEQUENTIAL_OVERLAP", int(profile_defaults["colmap_sequential_overlap"])),
        help="Temporal overlap window for sequential matcher",
    )
    parser.add_argument(
        "--colmap-min-registered-ratio",
        type=float,
        default=_env_float("COLMAP_MIN_REGISTERED_RATIO", 0.80),
        help="Minimum registered/extracted frame ratio before triggering SfM retry",
    )
    parser.add_argument(
        "--colmap-retry-min-registered-ratio",
        type=float,
        default=_env_float("COLMAP_RETRY_MIN_REGISTERED_RATIO", 0.75),
        help="Hard minimum registered/extracted frame ratio after forced retry",
    )
    parser.add_argument(
        "--blur-filter-keep-ratio",
        type=float,
        default=_env_float("BLUR_FILTER_KEEP_RATIO", 1.0),
        help="Keep ratio of sharpest frames before SfM (1.0 disables filtering)",
    )
    parser.add_argument(
        "--blur-filter-min-frames",
        type=int,
        default=_env_int("BLUR_FILTER_MIN_FRAMES", 120),
        help="Minimum number of frames to keep when blur filtering is enabled",
    )
    parser.add_argument("--skip-fixer", action="store_true", help="Skip Fixer image refinement")
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
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Skip dense reconstruction (use Gaussian PLY as mesh)",
    )
    parser.add_argument("--skip-sam3", action="store_true", help="Skip SAM3 object detection")
    parser.add_argument(
        "--sam3-strict-preflight",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("SAM3_PREFLIGHT_STRICT", False),
        help="Fail fast before reconstruction if SAM3 gated model access is unavailable",
    )
    parser.add_argument(
        "--parallel-post-stage6",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("NUREC_PARALLEL_POST_STAGE6", True),
        help="Run Stage 7 visual mesh and Stage 9 SAM3 concurrently after Stage 6",
    )
    parser.add_argument(
        "--environment",
        default="auto",
        choices=["auto", "default", "warehouse", "kitchen", "bedroom"],
        help="Environment type for SAM3 detection prompts (auto recommended)",
    )
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
    existing_grut_result: Dict[str, Any] | None = (
        _load_existing_grut_result(output_dir) if args.resume else None
    )

    # Load job spec for raw_prefix if not provided
    raw_prefix = args.raw_prefix
    if not raw_prefix:
        spec = json.loads(Path(args.job_spec).read_text(encoding="utf-8"))
        raw_prefix = spec.get("capture", {}).get("raw_prefix_uri", "")

    if args.dependency_preflight:
        _log("Running dependency preflight checks...")
        check_fused_ssim = args.preflight_check_fused_ssim and existing_grut_result is None
        if args.preflight_check_fused_ssim and existing_grut_result is not None:
            _log("Preflight: skipping fused_ssim ABI check (Stage 4 outputs already present for resume)")
        _run_dependency_preflight(check_fused_ssim=check_fused_ssim)

    _log(f"Quality profile: {profile}")

    sam3_preflight_path = output_dir / "sam3_preflight_report.json"
    sam3_skip_reason = ""
    if args.skip_sam3:
        sam3_preflight = {
            "schema_version": "v1",
            "generated_at": _utc_now_iso(),
            "enabled": False,
            "strict": bool(args.sam3_strict_preflight),
            "status": "disabled_by_flag",
            "reason": "--skip-sam3",
        }
    else:
        sam3_preflight = _run_sam3_preflight(strict=bool(args.sam3_strict_preflight))
        if str(sam3_preflight.get("status")) == "skip":
            sam3_skip_reason = str(sam3_preflight.get("reason") or "sam3_preflight_skip")
            _log(f"SAM3 preflight: skipping Stage 9 ({sam3_skip_reason})")
    sam3_preflight_path.write_text(json.dumps(sam3_preflight, indent=2), encoding="utf-8")

    cpu_cores = max(1, int(os.cpu_count() or 1))
    mapper_threads = max(0, int(args.colmap_mapper_threads))
    if mapper_threads == 0:
        mapper_threads = cpu_cores
    _log(f"CPU cores visible: {cpu_cores}; mapper threads target: {mapper_threads}")

    # -----------------------------------------------------------------------
    # Stage 1: Frame Extraction
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("STAGE 1: Frame Extraction")
    _log("=" * 60)
    video_path = find_video(raw_prefix, storage_root)
    frames_dir = workspace / "frames"
    if args.resume:
        existing_frame_count = _count_extracted_frames(frames_dir)
        if existing_frame_count > 0:
            frame_count = existing_frame_count
            _log(f"Resuming Stage 1: using existing extracted frames ({frame_count})")
        else:
            frame_count = extract_frames(video_path, frames_dir, args.max_frames, args.extract_fps)
    else:
        frame_count = extract_frames(video_path, frames_dir, args.max_frames, args.extract_fps)

    if frame_count < 10:
        _log(f"WARNING: Only {frame_count} frames extracted. Reconstruction may fail.")

    if args.blur_filter_keep_ratio < 1.0:
        filtered_count = _apply_blur_frame_filter(
            frames_dir,
            keep_ratio=args.blur_filter_keep_ratio,
            min_keep=args.blur_filter_min_frames,
        )
        if filtered_count > 0:
            frame_count = filtered_count

    capture_quality_report = build_capture_quality_report(frames_dir)
    capture_quality_path = output_dir / "capture_quality_report.json"
    capture_quality_path.write_text(json.dumps(capture_quality_report, indent=2), encoding="utf-8")

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
    sparse_root = workspace / "sparse"
    sparse_dir: Path
    registered_images = 0
    if args.resume:
        existing_sparse_dir, existing_sparse_count = _select_best_reconstruction(sparse_root, emit_logs=True)
        if existing_sparse_dir is not None and existing_sparse_count > 0:
            sparse_dir = existing_sparse_dir
            registered_images = int(existing_sparse_count)
            _log(
                "Resuming Stage 2: using existing COLMAP sparse model "
                f"{existing_sparse_dir} ({existing_sparse_count} images)"
            )
        else:
            sparse_dir = run_colmap_sfm(
                frames_dir,
                workspace,
                sift_use_gpu=sift_use_gpu,
                mapper_num_threads=mapper_threads,
                matcher_mode=args.colmap_matcher_mode,
                sequential_overlap=args.colmap_sequential_overlap,
            )
            registered_images = _read_registered_image_count(sparse_dir)
    else:
        sparse_dir = run_colmap_sfm(
            frames_dir,
            workspace,
            sift_use_gpu=sift_use_gpu,
            mapper_num_threads=mapper_threads,
            matcher_mode=args.colmap_matcher_mode,
            sequential_overlap=args.colmap_sequential_overlap,
        )
        registered_images = _read_registered_image_count(sparse_dir)

    min_registered_ratio = max(0.0, min(1.0, float(args.colmap_min_registered_ratio)))
    retry_min_ratio = max(0.0, min(1.0, float(args.colmap_retry_min_registered_ratio)))
    registered_ratio = _registration_ratio(
        registered_images=registered_images,
        extracted_frames=frame_count,
    )
    _log(
        "SfM coverage: "
        f"registered={registered_images}/{frame_count} "
        f"(ratio={registered_ratio:.3f})"
    )

    if registered_ratio < min_registered_ratio:
        _log(
            "SfM coverage below target; forcing retry with exhaustive matching "
            f"(threshold={min_registered_ratio:.3f})"
        )
        db_path = workspace / "database.db"
        if db_path.exists():
            db_path.unlink()
        if sparse_root.exists():
            shutil.rmtree(sparse_root)
        sparse_dir = run_colmap_sfm(
            frames_dir,
            workspace,
            sift_use_gpu=sift_use_gpu,
            mapper_num_threads=mapper_threads,
            matcher_mode="exhaustive",
            sequential_overlap=max(30, int(args.colmap_sequential_overlap)),
        )
        registered_images = _read_registered_image_count(sparse_dir)
        registered_ratio = _registration_ratio(
            registered_images=registered_images,
            extracted_frames=frame_count,
        )
        _log(
            "SfM retry coverage: "
            f"registered={registered_images}/{frame_count} "
            f"(ratio={registered_ratio:.3f})"
        )

    if registered_ratio < retry_min_ratio:
        raise RuntimeError(
            "COLMAP registration quality gate failed: "
            f"registered_ratio={registered_ratio:.3f} < retry_min={retry_min_ratio:.3f}. "
            "Capture likely has excessive blur/coverage gaps; rerun with steadier motion and more overlap."
        )

    capture_quality_report["sfm"] = {
        "registered_images": int(registered_images),
        "extracted_frames": int(frame_count),
        "registered_ratio": float(registered_ratio),
        "min_registered_ratio": float(min_registered_ratio),
        "retry_min_registered_ratio": float(retry_min_ratio),
    }
    capture_quality_path.write_text(json.dumps(capture_quality_report, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Stage 3: Undistort for 3DGRUT (PINHOLE cameras required)
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("STAGE 3: Image Undistortion (→ PINHOLE)")
    _log("=" * 60)
    undistorted_dir = workspace / "undistorted"
    undistorted_model_dir = undistorted_dir / "sparse" / "0"
    undistorted_images_dir = undistorted_dir / "images"
    has_undistorted_images = undistorted_images_dir.is_dir() and any(
        p.is_file() for p in undistorted_images_dir.rglob("*")
    )
    if args.resume and _has_colmap_model(undistorted_model_dir) and has_undistorted_images:
        _log("Resuming Stage 3: using existing undistorted COLMAP workspace")
    else:
        undistorted_dir = run_colmap_undistort(frames_dir, sparse_dir, workspace)

    # -----------------------------------------------------------------------
    # Stage 4: 3DGRUT Training → USDZ + PLY
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("STAGE 4: 3DGRUT Neural Reconstruction")
    _log("=" * 60)
    if existing_grut_result is not None:
        grut_result = existing_grut_result
        _log("Resuming Stage 4: using existing 3DGRUT exports in output directory")
    else:
        grut_result = run_3dgrut_training(
            undistorted_dir,
            output_dir,
            args.n_iterations,
        )

    # Copy 3DGRUT outputs to the expected locations
    usdz_src = Path(str(grut_result["usdz"]))
    ply_src = Path(str(grut_result["ply"]))
    usdz_dst = output_dir / "export_last.usdz"
    ply_dst = output_dir / "export_last.ply"

    if usdz_src != usdz_dst:
        shutil.copy2(str(usdz_src), str(usdz_dst))
    if ply_src != ply_dst:
        shutil.copy2(str(ply_src), str(ply_dst))

    # Also copy INGP checkpoint
    ingp_src_raw = grut_result.get("ingp")
    if ingp_src_raw:
        ingp_src = Path(str(ingp_src_raw))
        if ingp_src.exists():
            shutil.copy2(str(ingp_src), str(output_dir / "export_last.ingp"))

    # -----------------------------------------------------------------------
    # Stage 5: Fixer Image Refinement (optional)
    # -----------------------------------------------------------------------
    if not args.skip_fixer:
        _log("=" * 60)
        _log("STAGE 5: Fixer Image Refinement")
        _log("=" * 60)
        _log(f"Fixer backend mode: {args.fixer_mode}")
        fixed_dir = output_dir / "fixer_output"
        if args.resume and _has_image_outputs(fixed_dir):
            _log("Resuming Stage 5: using existing Fixer outputs")
        else:
            renders_dirs = list(Path(str(grut_result["result_dir"])).rglob("renders"))
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

    dense_dir = workspace / "dense"
    fused_ply_resume = dense_dir / "fused.ply"
    mesh_method_path = output_dir / "mesh_method.txt"
    dense_result: Dict[str, Any] | None = None
    reused_dense_stage6 = False
    if (
        args.resume
        and _is_nonempty_file(mesh_ply)
        and _is_nonempty_file(mesh_method_path)
        and _is_nonempty_file(fused_ply_resume)
    ):
        try:
            _validate_collision_mesh(mesh_ply)
            existing_mesh_method = mesh_method_path.read_text(encoding="utf-8").strip().lower()
            if existing_mesh_method in {"poisson_open3d", "delaunay_colmap"}:
                dense_result = {
                    "mesh_method": existing_mesh_method,
                    "fused_ply": fused_ply_resume,
                    "dense_dir": dense_dir,
                }
                reused_dense_stage6 = True
                _log(
                    "Resuming Stage 6: using existing fused cloud + collision mesh "
                    f"(method={existing_mesh_method})"
                )
            else:
                _log(
                    "Resume check: invalid mesh method marker "
                    f"{existing_mesh_method!r}; rerunning dense reconstruction"
                )
        except Exception as exc:
            _log(f"Resume check: existing collision mesh unusable ({exc}); rerunning Stage 6")

    if dense_result is None:
        dense_result = run_dense_reconstruction(frames_dir, sparse_dir, workspace, mesh_ply)

    mesh_method = str(dense_result.get("mesh_method") or "")
    fused_ply = Path(str(dense_result.get("fused_ply") or ""))
    collision_report_path = output_dir / "collision_mesh_report.json"
    if args.resume and reused_dense_stage6 and _is_nonempty_file(collision_report_path):
        collision_report = _load_json_dict(collision_report_path)
        if collision_report:
            _log("Resuming Stage 6: using existing collision postprocess report")
        else:
            collision_report = _postprocess_collision_mesh(mesh_ply)
            collision_report_path.write_text(json.dumps(collision_report, indent=2), encoding="utf-8")
    else:
        collision_report = _postprocess_collision_mesh(mesh_ply)
        collision_report_path.write_text(json.dumps(collision_report, indent=2), encoding="utf-8")

    try:
        _enforce_collision_spike_gate(collision_report)
    except RuntimeError as spike_error:
        if mesh_method == "delaunay_colmap" and fused_ply.exists():
            _log(f"Collision spike gate failed for Delaunay mesh ({spike_error})")
            _log("Attempting collision fallback: forced Open3D Poisson from fused cloud...")
            if _mesh_with_open3d_poisson(fused_ply, mesh_ply, force=True):
                _validate_collision_mesh(mesh_ply)
                mesh_method = "poisson_open3d"
                collision_report = _postprocess_collision_mesh(mesh_ply)
                collision_report_path.write_text(
                    json.dumps(collision_report, indent=2), encoding="utf-8"
                )
                _enforce_collision_spike_gate(collision_report)
            else:
                raise RuntimeError(
                    "Collision spike gate failed and fallback Poisson meshing was unavailable"
                ) from spike_error
        else:
            raise

    mesh_method_path.write_text(f"{mesh_method}\n", encoding="utf-8")
    _log(f"  Collision mesh method: {mesh_method}")
    quality_profile = "delaunay_relaxed" if mesh_method == "delaunay_colmap" else "default"
    quality_profile_path = output_dir / "quality_profile.txt"
    quality_profile_path.write_text(f"{quality_profile}\n", encoding="utf-8")
    _log(f"  Suggested quality profile: {quality_profile}")

    def _run_stage7_visual() -> Dict[str, Any]:
        _log("=" * 60)
        _log("STAGE 7: Visual Mesh Exports")
        _log("=" * 60)
        if args.resume:
            existing_report = _load_existing_visual_report(output_dir)
            if existing_report is not None:
                _log("Resuming Stage 7: using existing visual mesh artifacts")
                _save_visual_report(output_dir, existing_report)
                return existing_report
        visual = build_visual_mesh_artifacts(
            output_dir=output_dir,
            fused_ply=fused_ply,
            gaussian_ply=ply_dst,
            workspace=workspace,
        )
        _save_visual_report(output_dir, visual)
        if bool(visual.get("enabled", False)) and str(visual.get("status")) != "ok":
            raise RuntimeError(f"visual mesh export failed: {visual}")
        return visual

    def _run_stage9() -> Path | None:
        _log("=" * 60)
        _log("STAGE 9: SAM3 Object Detection")
        _log("=" * 60)
        return _run_stage9_sam3(
            output_dir=output_dir,
            workspace=workspace,
            frames_dir=frames_dir,
            frame_count=frame_count,
            requested_environment=args.environment,
            requested_n_frames=args.sam3_n_frames,
            requested_min_frame_detections=args.sam3_min_frame_detections,
            gaussian_ply=ply_dst,
            resume=args.resume,
        )

    object_index_path: Path | None = None
    sam3_enabled = (not args.skip_sam3) and (not sam3_skip_reason)
    if sam3_enabled and args.parallel_post_stage6:
        _log("Running Stage 7 and Stage 9 concurrently...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            visual_future = executor.submit(_run_stage7_visual)
            sam3_future = executor.submit(_run_stage9)
            visual_report = visual_future.result()
            object_index_path = sam3_future.result()
    else:
        visual_report = _run_stage7_visual()
        if sam3_enabled:
            object_index_path = _run_stage9()
        else:
            if sam3_skip_reason:
                _log(f"Skipping SAM3 detection ({sam3_skip_reason})")
            else:
                _log("Skipping SAM3 detection (--skip-sam3)")

    if object_index_path is None:
        placeholder_index = output_dir / "object_point_cloud_index.json"
        if not _is_nonempty_file(placeholder_index):
            placeholder_payload = {
                "schema_version": "v1",
                "generated_at": _utc_now_iso(),
                "objects": [],
                "skip_reason": sam3_skip_reason or ("--skip-sam3" if args.skip_sam3 else "sam3_no_objects"),
            }
            placeholder_index.write_text(json.dumps(placeholder_payload, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Stage 8: Occupancy Grid
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("STAGE 8: Occupancy Grid")
    _log("=" * 60)
    occupancy_bin = output_dir / "occupancy.bin"
    if args.resume and _is_nonempty_file(occupancy_bin):
        _log("Resuming Stage 8: using existing occupancy grid")
    else:
        generate_occupancy(ply_dst, occupancy_bin)

    # -----------------------------------------------------------------------
    # Stage 8.5: Mesh Manifest (artifact roles + viewer guidance)
    # -----------------------------------------------------------------------
    write_mesh_manifest(
        output_dir=output_dir,
        visual_usdz=usdz_dst,
        gaussian_ply=ply_dst,
        collision_mesh_ply=mesh_ply,
        occupancy=occupancy_bin,
        visual_report=visual_report,
        collision_method=mesh_method,
        collision_report=collision_report,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("RECONSTRUCTION COMPLETE")
    _log("=" * 60)
    required = [
        "export_last.usdz",
        "export_last.ply",
        "nvblox_mesh.ply",
        "occupancy.bin",
        "mesh_manifest.json",
    ]
    if bool(visual_report.get("enabled", False)):
        required.append("visual_mesh.glb")
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
        "sam3_preflight_report.json",
        "capture_quality_report.json",
        "mesh_method.txt",
        "quality_profile.txt",
        "collision_mesh_report.json",
        "visual_mesh_report.json",
        "visual_pointcloud.ply",
        "visual_mesh_robust.glb",
    ]:
        path = output_dir / artifact
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            _log(f"  ○ {artifact}: {size_mb:.1f}MB")

    if grut_result.get("metrics"):
        m = grut_result["metrics"]
        _log(
            f"  Quality: PSNR={m.get('mean_psnr', 0):.2f} "
            f"SSIM={m.get('mean_ssim', 0):.3f} "
            f"LPIPS={m.get('mean_lpips', 0):.3f}"
        )

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
