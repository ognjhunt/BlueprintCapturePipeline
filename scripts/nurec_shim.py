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
  - export_last_refined.usdz / export_last_refined.ply (optional refined visual assets)
  - gap_analysis_report.json / view_repair_report.json / post_stage4_distill_report.json
  - refinement_quality_gate.json (auto-rollback quality gate decision)
  - occupancy.bin     (voxel occupancy grid)
  - object_point_cloud_index.json  (SAM3-detected objects for swap pipeline)

Usage as NUREC_PIPELINE_COMMAND:
  export NUREC_PIPELINE_COMMAND="python3 /app/scripts/nurec_shim.py \
    --job-spec {JOB_SPEC_PATH} --output-dir {NUREC_OUTPUT_DIR} \
    --raw-prefix {RAW_PREFIX_URI}"

Optional Fixer routing:
  --skip-fixer                     # disable stage 5
  --fixer-mode auto|local|h100    # default local (auto aliases local)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import statistics
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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
FIXER_WEIGHTS_DIR = os.getenv("FIXER_WEIGHTS_DIR", "/opt/fixer_weights")
DEFAULT_FIXER_H100_SCRIPT = os.getenv("FIXER_H100_SCRIPT", "/app/scripts/fixer_h100_stage.sh")
STAGE14_RESUME_METADATA = ".stage14_resume_metadata.json"
POST_STAGE4_GAP_ANALYZER_SCRIPT = REPO_ROOT / "scripts" / "post_stage4_gap_analyzer.py"
POST_STAGE4_VIEW_REPAIR_SCRIPT = REPO_ROOT / "scripts" / "post_stage4_view_repair.py"
POST_STAGE4_DISTILL_SCRIPT = REPO_ROOT / "scripts" / "post_stage4_distill.py"
POST_STAGE4_VIRTUAL_RENDER_SCRIPT = REPO_ROOT / "scripts" / "post_stage4_virtual_render.py"


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
def _probe_video_duration_seconds(video_path: Path) -> float | None:
    """Return media duration in seconds, or None if ffprobe fails."""
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if probe.returncode != 0:
            _log(f"WARNING: ffprobe failed for {video_path} ({probe.stderr.strip()[:200]})")
            return None
        duration = float((probe.stdout or "").strip())
        if duration > 0:
            return duration
    except Exception as exc:
        _log(f"WARNING: Could not read video duration via ffprobe ({exc})")
    return None


def _resolve_effective_max_frames(video_duration_sec: float | None, requested_max_frames: int) -> tuple[int, str]:
    """Resolve frame budget for long videos while keeping a hard upper bound."""
    requested = max(1, int(requested_max_frames))
    if not _env_flag("ADAPTIVE_MAX_FRAMES", True):
        return requested, "adaptive_max_frames=disabled"
    if video_duration_sec is None or video_duration_sec <= 0:
        return requested, "adaptive_max_frames=duration_unknown"

    target_density_fps = max(0.01, _env_float("ADAPTIVE_MAX_FRAMES_TARGET_FPS", 3.0))
    hard_cap = max(requested, _env_int("ADAPTIVE_MAX_FRAMES_HARD_CAP", 6000))
    proposed = int(math.ceil(video_duration_sec * target_density_fps))
    resolved = max(requested, min(hard_cap, proposed))
    reason = (
        "adaptive_max_frames=enabled "
        f"(duration={video_duration_sec:.1f}s target_density_fps={target_density_fps:.3f} "
        f"proposed={proposed} hard_cap={hard_cap} resolved={resolved})"
    )
    return resolved, reason


def _resolve_effective_extract_fps(
    video_duration_sec: float | None,
    requested_extract_fps: int,
    effective_max_frames: int,
) -> tuple[float, str]:
    """Resolve extraction FPS so long videos are sampled across full duration."""
    requested = float(max(1, int(requested_extract_fps)))
    if not _env_flag("ADAPTIVE_EXTRACT_FPS", True):
        return requested, "adaptive_extract_fps=disabled"
    if video_duration_sec is None or video_duration_sec <= 0:
        return requested, "adaptive_extract_fps=duration_unknown"

    budget_fps = max(0.01, float(effective_max_frames) / float(video_duration_sec))
    effective = min(requested, budget_fps)
    warn_floor = max(0.01, _env_float("ADAPTIVE_EXTRACT_FPS_WARN_FLOOR", 0.15))
    reason = (
        "adaptive_extract_fps=enabled "
        f"(requested_fps={requested:.3f} budget_fps={budget_fps:.3f} effective_fps={effective:.3f} "
        f"max_frames={effective_max_frames})"
    )
    if effective < warn_floor:
        _log(
            "WARNING: Effective extraction FPS is very low "
            f"({effective:.3f} < warn_floor={warn_floor:.3f}); "
            "spatial coverage is preserved but local detail may degrade."
        )
    return effective, reason


def _resolve_effective_max_n_gaussians(
    *,
    video_duration_sec: float | None,
    registered_frame_count: int,
    sfm_point_count: int,
    n_iterations: int,
    requested_max_n_gaussians: int,
) -> tuple[int, int, str]:
    """Resolve max Gaussian count and add-end-iteration for 3DGRUT MCMC strategy.

    Uses two signals (SfM point count × multiplier, frame count × per-frame budget)
    and takes the conservative minimum.  All tuning knobs are overridable via env vars.

    Returns:
        (max_n_gaussians, add_end_iteration, reason_string)
    """
    requested = max(0, int(requested_max_n_gaussians))
    refinement_tail = max(0.0, min(0.5, _env_float("GRUT_REFINEMENT_TAIL_RATIO", 0.15)))
    end_iter = max(1, int(n_iterations * (1.0 - refinement_tail)))

    if not _env_flag("ADAPTIVE_MAX_N_GAUSSIANS", True):
        effective = requested if requested > 0 else 1_000_000
        return effective, end_iter, "adaptive_max_n_gaussians=disabled"

    # Tuning knobs
    sfm_multiplier = max(1.0, _env_float("GRUT_SFM_POINT_MULTIPLIER", 20.0))
    per_frame_budget = max(100, _env_int("GRUT_PER_FRAME_GAUSSIAN_BUDGET", 2000))
    hard_floor = max(10_000, _env_int("GRUT_MAX_N_GAUSSIANS_FLOOR", 100_000))
    hard_ceiling = max(hard_floor, _env_int("GRUT_MAX_N_GAUSSIANS_CEILING", 2_000_000))

    # Primary signal: SfM 3D point count
    sfm_signal = int(sfm_point_count * sfm_multiplier) if sfm_point_count > 0 else 0

    # Secondary signal: per-frame Gaussian budget
    frame_signal = int(registered_frame_count * per_frame_budget) if registered_frame_count > 0 else 0

    # Combine: conservative minimum of available signals
    if sfm_signal > 0 and frame_signal > 0:
        proposed = min(sfm_signal, frame_signal)
        signal_source = "min(sfm,frame)"
    elif sfm_signal > 0:
        proposed = sfm_signal
        signal_source = "sfm_only"
    elif frame_signal > 0:
        proposed = frame_signal
        signal_source = "frame_only"
    else:
        proposed = hard_floor
        signal_source = "fallback_floor"

    # Clamp to [floor, ceiling]
    resolved = max(hard_floor, min(hard_ceiling, proposed))

    # If user explicitly requested a value (>0), honor it as an override
    if requested > 0:
        resolved = requested
        signal_source = "user_override"

    reason = (
        "adaptive_max_n_gaussians=enabled "
        f"(sfm_points={sfm_point_count} sfm_mult={sfm_multiplier:.1f} sfm_signal={sfm_signal} "
        f"frames={registered_frame_count} per_frame={per_frame_budget} frame_signal={frame_signal} "
        f"proposed={proposed} signal={signal_source} "
        f"floor={hard_floor} ceiling={hard_ceiling} resolved={resolved} "
        f"refinement_tail={refinement_tail:.2f} add_end_iter={end_iter})"
    )
    return resolved, end_iter, reason


def _resolve_effective_min_registered_ratio(
    *,
    requested_ratio: float,
    registered_images: int,
    extracted_frames: int,
) -> tuple[float, str]:
    """Adaptively lower the SfM retry threshold when absolute frame count is healthy.

    For short videos, sequential matching may register only 50-65% of frames,
    but 120-150 well-separated frames produce better 3DGRUT output than 224
    redundant frames from exhaustive matching.  When the absolute number of
    registered frames exceeds a minimum, we accept a lower ratio rather than
    forcing a retry that adds redundant viewpoints.

    For long videos (1000+ frames) the ratio threshold stays strict because
    low registration genuinely indicates reconstruction gaps.
    """
    requested = max(0.0, min(1.0, float(requested_ratio)))

    if not _env_flag("ADAPTIVE_MIN_REGISTERED_RATIO", True):
        return requested, "adaptive_min_registered_ratio=disabled"

    # Absolute floor: if we have this many registered frames, the
    # reconstruction is healthy regardless of the ratio.
    absolute_min_frames = max(20, _env_int("SFM_ABSOLUTE_MIN_FRAMES", 100))

    # Below this extracted frame count, don't relax — every frame matters.
    small_set_threshold = max(20, _env_int("SFM_SMALL_SET_THRESHOLD", 60))

    # Relaxed ratio used when absolute count is healthy.
    relaxed_ratio = max(0.10, min(requested, _env_float("SFM_RELAXED_RATIO", 0.50)))

    if extracted_frames <= small_set_threshold:
        # Small captures: keep the strict ratio — losing frames hurts.
        return requested, (
            "adaptive_min_registered_ratio=strict "
            f"(extracted={extracted_frames} <= small_set={small_set_threshold})"
        )

    if registered_images >= absolute_min_frames:
        reason = (
            "adaptive_min_registered_ratio=relaxed "
            f"(registered={registered_images} >= absolute_min={absolute_min_frames} "
            f"requested_ratio={requested:.3f} relaxed_ratio={relaxed_ratio:.3f})"
        )
        return relaxed_ratio, reason

    return requested, (
        "adaptive_min_registered_ratio=strict "
        f"(registered={registered_images} < absolute_min={absolute_min_frames})"
    )


def extract_frames(video_path: Path, frames_dir: Path,
                   max_frames: int = 300, target_fps: float = 5) -> int:
    """Extract frames from video at reduced FPS for SfM."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Avoid stale frame tails from prior runs (e.g., previous higher frame count).
    for stale in frames_dir.glob("frame_*.jpg"):
        try:
            stale.unlink()
        except Exception:
            pass
    for report_name in (".blurdetect_report.txt", ".signalstats_report.txt"):
        report_path = frames_dir / report_name
        if report_path.exists():
            try:
                report_path.unlink()
            except Exception:
                pass
    _log(f"Extracting frames from {video_path} at {target_fps:.3f} fps (max {max_frames})...")
    _run([
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={target_fps:.6f}",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        str(frames_dir / "frame_%05d.jpg"),
    ])
    count = len(list(frames_dir.glob("frame_*.jpg")))
    _log(f"Extracted {count} frames.")
    return count


def _frame_blur_scores(frames_dir: Path, *, fail_on_error: bool = False) -> list[tuple[Path, float]]:
    """Compute per-frame blur scores using ffmpeg blurdetect (higher is blurrier)."""
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        return []
    ffmpeg_input = str(frames_dir / "frame_*.jpg")
    blur_report = frames_dir / ".blurdetect_report.txt"
    try:
        _run([
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-pattern_type",
            "glob",
            "-i",
            ffmpeg_input,
            "-vf",
            f"blurdetect,metadata=mode=print:file={blur_report}",
            "-f",
            "null",
            "-",
        ])
    except Exception as exc:
        if fail_on_error:
            raise RuntimeError(
                "blurdetect failed while blur filtering is required; "
                "install an ffmpeg build with blurdetect or disable strict blur filter"
            ) from exc
        _log(f"WARNING: blurdetect failed ({exc}); skipping blur filtering")
        return []

    if not blur_report.is_file() or blur_report.stat().st_size <= 0:
        if fail_on_error:
            raise RuntimeError(
                "blurdetect produced no report while blur filtering is required; "
                "cannot continue in strict mode"
            )
        return []
    text = blur_report.read_text(encoding="utf-8", errors="ignore")
    frame_to_score: dict[int, float] = {}
    current_frame_idx: int | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("frame:"):
            try:
                token = line.split()[0]
                current_frame_idx = int(token.split(":", 1)[1])
            except Exception:
                current_frame_idx = None
            continue
        if line.startswith("lavfi.blur=") and current_frame_idx is not None:
            try:
                frame_to_score[current_frame_idx] = float(line.split("=", 1)[1])
            except Exception:
                pass

    out: list[tuple[Path, float]] = []
    for idx, path in enumerate(frames):
        score = frame_to_score.get(idx)
        if score is None:
            continue
        out.append((path, score))
    return out


def _apply_blur_frame_filter(
    frames_dir: Path,
    *,
    keep_ratio: float,
    min_keep: int,
    fail_on_error: bool = False,
    status_out: Dict[str, Any] | None = None,
) -> int:
    """Keep the sharpest subset of extracted frames in-place."""
    if status_out is not None:
        status_out.clear()
    entries = _frame_blur_scores(frames_dir, fail_on_error=fail_on_error)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        if status_out is not None:
            status_out.update({
                "status": "no_frames",
                "input_frames": 0,
                "scores_count": 0,
            })
        return 0
    if not entries:
        if status_out is not None:
            status_out.update({
                "status": "unavailable",
                "input_frames": int(len(frames)),
                "scores_count": 0,
                "kept_frames": int(len(frames)),
                "dropped_frames": 0,
                "keep_ratio": float(keep_ratio),
                "min_keep": int(min_keep),
            })
        if fail_on_error:
            raise RuntimeError(
                "blur filtering is required but blur scores are unavailable; "
                "refusing to continue with unfiltered frames"
            )
        return len(frames)

    ratio = max(0.0, min(1.0, float(keep_ratio)))
    if ratio <= 0.0:
        if status_out is not None:
            status_out.update({
                "status": "disabled",
                "input_frames": int(len(frames)),
                "scores_count": int(len(entries)),
                "kept_frames": int(len(frames)),
                "dropped_frames": 0,
                "keep_ratio": float(ratio),
                "min_keep": int(min_keep),
            })
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
    if status_out is not None:
        status_out.update({
            "status": "ok",
            "input_frames": int(len(frames)),
            "scores_count": int(len(entries)),
            "kept_frames": int(remaining),
            "dropped_frames": int(dropped),
            "keep_ratio": float(ratio),
            "min_keep": int(min_keep),
        })
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

    ffmpeg_input = str(frames_dir / "frame_*.jpg")
    frame_numbers: list[int] = []
    for frame_path in frames:
        try:
            frame_numbers.append(int(frame_path.stem.split("_")[-1]))
        except Exception:
            frame_numbers.append(0)
    stats_report = frames_dir / ".signalstats_report.txt"
    try:
        _run([
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-pattern_type",
            "glob",
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
    current_frame_number: int | None = None
    yavg: Dict[int, float] = {}
    ydif: Dict[int, float] = {}
    for line in text.splitlines():
        entry = line.strip()
        if entry.startswith("frame:"):
            try:
                token = entry.split()[0]
                current_idx = int(token.split(":", 1)[1])
                if 0 <= current_idx < len(frame_numbers):
                    current_frame_number = frame_numbers[current_idx]
                else:
                    current_frame_number = None
            except Exception:
                current_frame_number = None
            continue
        if current_frame_number is None:
            continue
        if entry.startswith("lavfi.signalstats.YAVG="):
            try:
                yavg[current_frame_number] = float(entry.split("=", 1)[1])
            except Exception:
                pass
        elif entry.startswith("lavfi.signalstats.YDIF="):
            try:
                ydif[current_frame_number] = float(entry.split("=", 1)[1])
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
        # ffmpeg blurdetect: higher score means blurrier frame.
        sorted_sharpest = sorted(blur_entries, key=lambda item: item[1])
        sorted_blurriest = sorted(blur_entries, key=lambda item: item[1], reverse=True)
        report["blurriest_frames"] = [
            {"frame": path.name, "score": float(score)}
            for path, score in sorted_blurriest[:10]
        ]
        report["sharpest_frames"] = [
            {"frame": path.name, "score": float(score)}
            for path, score in sorted_sharpest[:10]
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
    if db_path.exists():
        db_path.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
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
        loop_detection = _env_flag("COLMAP_LOOP_DETECTION", True)
        _log(
            f"Running COLMAP sequential matching "
            f"(SIFT GPU={sift_gpu_flag}, overlap={overlap}, "
            f"loop_detection={loop_detection})..."
        )
        seq_cmd = [
            "colmap", "sequential_matcher",
            "--database_path", str(db_path),
            "--SequentialMatching.overlap", str(overlap),
            "--SequentialMatching.loop_detection", "1" if loop_detection else "0",
            matching_gpu_option, sift_gpu_flag,
        ]
        _run(seq_cmd)
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


def _resolve_chunked_sfm_enabled(requested_mode: str, frame_count: int, min_frames: int) -> tuple[bool, str]:
    mode = (requested_mode or "").strip().lower()
    threshold = max(1, int(min_frames))
    if mode == "on":
        return True, f"requested=on (frame_count={frame_count})"
    if mode == "off":
        return False, "requested=off"
    if mode in {"auto", ""}:
        enabled = int(frame_count) >= threshold
        return enabled, (
            f"requested=auto (frame_count={frame_count} "
            f"min_frames={threshold} -> {'enabled' if enabled else 'disabled'})"
        )
    _log(f"WARNING: Unknown COLMAP chunked mode {requested_mode!r}; falling back to auto")
    enabled = int(frame_count) >= threshold
    return enabled, (
        f"requested={requested_mode!r} invalid -> auto "
        f"(frame_count={frame_count} min_frames={threshold} -> {'enabled' if enabled else 'disabled'})"
    )


def _resolve_colmap_retry_matcher_mode(requested_mode: str, frame_count: int) -> tuple[str, str]:
    mode = (requested_mode or "").strip().lower()
    if mode in {"sequential", "exhaustive"}:
        return mode, f"requested={mode}"
    threshold = max(50, _env_int("COLMAP_AUTO_EXHAUSTIVE_MAX_FRAMES", 600))
    resolved = "exhaustive" if int(frame_count) <= threshold else "sequential"
    if mode in {"auto", ""}:
        return resolved, (
            "requested=auto "
            f"(frame_count={frame_count} threshold={threshold} -> {resolved})"
        )
    _log(f"WARNING: Unknown COLMAP retry matcher mode {requested_mode!r}; falling back to auto")
    return resolved, (
        f"requested={requested_mode!r} invalid -> auto "
        f"(frame_count={frame_count} threshold={threshold} -> {resolved})"
    )


def _build_colmap_chunk_ranges(
    total_frames: int,
    *,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
) -> list[tuple[int, int]]:
    total = max(0, int(total_frames))
    if total <= 0:
        return []

    size = max(20, int(chunk_size))
    size = min(size, total)
    overlap = max(0, int(chunk_overlap))
    overlap = min(overlap, size - 1)
    step = max(1, size - overlap)
    max_allowed = max(1, int(max_chunks))

    if total > size:
        min_step = max(1, math.ceil((total - size) / max(1, max_allowed - 1)))
        step = max(step, min_step)

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total, start + size)
        ranges.append((start, end))
        if end >= total:
            break
        start += step

    # Ensure tail coverage with a full-size final window whenever possible.
    if ranges and ranges[-1][1] < total:
        tail_start = max(0, total - size)
        if ranges[-1][0] != tail_start:
            ranges.append((tail_start, total))

    return ranges


def _populate_chunk_frames(chunk_frames_dir: Path, frame_paths: Sequence[Path]) -> None:
    chunk_frames_dir.mkdir(parents=True, exist_ok=True)
    for src in frame_paths:
        dst = chunk_frames_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def _copy_colmap_model(src_model_dir: Path, dst_model_dir: Path) -> None:
    dst_model_dir.mkdir(parents=True, exist_ok=True)
    copied_any = False
    for name in (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "cameras.txt",
        "images.txt",
        "points3D.txt",
    ):
        src = src_model_dir / name
        if src.exists():
            shutil.copy2(src, dst_model_dir / name)
            copied_any = True
    if not copied_any:
        raise RuntimeError(f"No COLMAP model files found in {src_model_dir}")


def _symlink_or_copy_tree(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copytree(src, dst)


def _export_undistorted_artifacts(*, output_dir: Path, undistorted_dir: Path) -> None:
    """Expose undistorted COLMAP assets as stable output-side artifacts."""
    src_images = undistorted_dir / "images"
    src_sparse = undistorted_dir / "sparse" / "0"
    if not src_images.is_dir() or not src_sparse.is_dir():
        return
    export_root = output_dir / "colmap_undistorted"
    _symlink_or_copy_tree(src_images, export_root / "images")
    _symlink_or_copy_tree(src_sparse, export_root / "sparse" / "0")


def run_colmap_sfm_chunked(
    frames_dir: Path,
    workspace: Path,
    *,
    sift_use_gpu: bool,
    mapper_num_threads: int = 0,
    chunk_size_frames: int = 600,
    chunk_overlap_frames: int = 120,
    chunk_max_chunks: int = 24,
    chunk_matcher_mode: str = "sequential",
    sequential_overlap: int = 30,
) -> tuple[Path, Dict[str, Any]]:
    """Run chunked SfM and merge chunk models into workspace/sparse/0."""
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No extracted frames found in {frames_dir}")

    ranges = _build_colmap_chunk_ranges(
        len(frame_paths),
        chunk_size=chunk_size_frames,
        chunk_overlap=chunk_overlap_frames,
        max_chunks=chunk_max_chunks,
    )
    if not ranges:
        raise RuntimeError("Failed to compute chunk ranges for COLMAP chunked SfM")

    chunk_root = workspace / "chunked_sfm"
    if chunk_root.exists():
        shutil.rmtree(chunk_root)
    chunk_root.mkdir(parents=True, exist_ok=True)

    successful_chunks: list[Dict[str, Any]] = []
    failed_chunks: list[Dict[str, Any]] = []
    for idx, (start, end) in enumerate(ranges):
        chunk_dir = chunk_root / f"chunk_{idx:03d}"
        chunk_frames_dir = chunk_dir / "frames"
        chunk_workspace = chunk_dir / "workspace"
        chunk_frames = frame_paths[start:end]
        _populate_chunk_frames(chunk_frames_dir, chunk_frames)

        chunk_size = end - start
        per_chunk_seq_overlap = min(max(1, int(sequential_overlap)), max(1, chunk_size - 1))
        _log(
            f"Chunked SfM {idx + 1}/{len(ranges)}: "
            f"frames {start + 1}-{end} ({chunk_size} frames)"
        )
        try:
            model_dir = run_colmap_sfm(
                chunk_frames_dir,
                chunk_workspace,
                sift_use_gpu=sift_use_gpu,
                mapper_num_threads=mapper_num_threads,
                matcher_mode=chunk_matcher_mode,
                sequential_overlap=per_chunk_seq_overlap,
            )
            registered_images = _read_registered_image_count(model_dir)
            successful_chunks.append(
                {
                    "chunk_index": idx,
                    "start_frame_idx": start,
                    "end_frame_idx_exclusive": end,
                    "model_dir": model_dir,
                    "registered_images": int(registered_images),
                }
            )
        except Exception as exc:
            message = str(exc)
            _log(f"WARNING: Chunked SfM failed for chunk {idx + 1}/{len(ranges)} ({message})")
            failed_chunks.append(
                {
                    "chunk_index": idx,
                    "start_frame_idx": start,
                    "end_frame_idx_exclusive": end,
                    "error": message,
                }
            )

    if not successful_chunks:
        raise RuntimeError("Chunked SfM produced no successful chunk reconstructions")

    successful_chunks.sort(key=lambda item: int(item["chunk_index"]))
    current_model = Path(str(successful_chunks[0]["model_dir"]))
    current_registered = int(successful_chunks[0]["registered_images"])
    selected_source = f"chunk_{int(successful_chunks[0]['chunk_index']):03d}"
    merge_successes = 0
    merge_failures = 0

    merge_root = chunk_root / "merged"
    merge_root.mkdir(parents=True, exist_ok=True)
    for merge_idx, chunk in enumerate(successful_chunks[1:], start=1):
        next_model = Path(str(chunk["model_dir"]))
        next_registered = int(chunk["registered_images"])
        output_model = merge_root / f"merge_{merge_idx:03d}"
        output_model.mkdir(parents=True, exist_ok=True)
        try:
            _run(
                [
                    "colmap",
                    "model_merger",
                    "--input_path1",
                    str(current_model),
                    "--input_path2",
                    str(next_model),
                    "--output_path",
                    str(output_model),
                ]
            )
            try:
                _run(
                    [
                        "colmap",
                        "bundle_adjuster",
                        "--input_path",
                        str(output_model),
                        "--output_path",
                        str(output_model),
                    ]
                )
            except RuntimeError as exc:
                _log(f"WARNING: bundle_adjuster after merge {merge_idx} failed ({exc})")
            current_model = output_model
            current_registered = _read_registered_image_count(current_model)
            selected_source = f"merge_{merge_idx:03d}"
            merge_successes += 1
            _log(
                f"Chunked SfM merge {merge_idx}/{len(successful_chunks) - 1}: "
                f"registered_images={current_registered}"
            )
        except RuntimeError as exc:
            merge_failures += 1
            _log(
                f"WARNING: model_merger failed for chunk {int(chunk['chunk_index']) + 1} ({exc})"
            )
            # If merge fails, keep the model with better registration coverage.
            if next_registered > current_registered:
                current_model = next_model
                current_registered = next_registered
                selected_source = f"chunk_{int(chunk['chunk_index']):03d}_fallback_best"
                _log(
                    f"Chunked SfM fallback: switched to chunk model with "
                    f"{current_registered} registered images"
                )

    sparse_root = workspace / "sparse"
    if sparse_root.exists():
        shutil.rmtree(sparse_root)
    final_sparse_dir = sparse_root / "0"
    _copy_colmap_model(current_model, final_sparse_dir)
    final_registered = _read_registered_image_count(final_sparse_dir)

    report: Dict[str, Any] = {
        "enabled": True,
        "chunk_count_planned": int(len(ranges)),
        "chunk_count_successful": int(len(successful_chunks)),
        "chunk_count_failed": int(len(failed_chunks)),
        "merge_successes": int(merge_successes),
        "merge_failures": int(merge_failures),
        "chunk_size_frames": int(chunk_size_frames),
        "chunk_overlap_frames": int(chunk_overlap_frames),
        "chunk_max_chunks": int(chunk_max_chunks),
        "chunk_matcher_mode": str(chunk_matcher_mode),
        "selected_model_source": selected_source,
        "selected_registered_images": int(final_registered),
        "failed_chunks": failed_chunks,
    }
    return final_sparse_dir, report


def _run_sfm_with_optional_chunking(
    *,
    frames_dir: Path,
    workspace: Path,
    sift_use_gpu: bool,
    mapper_num_threads: int,
    matcher_mode: str,
    sequential_overlap: int,
    frame_count: int,
    chunked_mode: str,
    chunk_min_frames: int,
    chunk_size_frames: int,
    chunk_overlap_frames: int,
    chunk_max_chunks: int,
    chunk_matcher_mode: str,
) -> tuple[Path, int, Dict[str, Any]]:
    chunk_enabled, chunk_reason = _resolve_chunked_sfm_enabled(
        chunked_mode,
        frame_count,
        chunk_min_frames,
    )
    sfm_report: Dict[str, Any] = {
        "chunking_requested_mode": str(chunked_mode),
        "chunking_enabled": bool(chunk_enabled),
        "chunking_reason": chunk_reason,
    }

    if chunk_enabled:
        _log(f"COLMAP chunked SfM enabled ({chunk_reason})")
        try:
            sparse_dir, chunk_report = run_colmap_sfm_chunked(
                frames_dir,
                workspace,
                sift_use_gpu=sift_use_gpu,
                mapper_num_threads=mapper_num_threads,
                chunk_size_frames=chunk_size_frames,
                chunk_overlap_frames=chunk_overlap_frames,
                chunk_max_chunks=chunk_max_chunks,
                chunk_matcher_mode=chunk_matcher_mode,
                sequential_overlap=sequential_overlap,
            )
            registered_images = _read_registered_image_count(sparse_dir)
            sfm_report["chunking_applied"] = True
            sfm_report["chunking"] = chunk_report
            return sparse_dir, int(registered_images), sfm_report
        except Exception as exc:
            _log(f"WARNING: Chunked SfM failed ({exc}); falling back to single-pass SfM")
            sfm_report["chunking_applied"] = False
            sfm_report["chunking_fallback"] = "single_pass"
            sfm_report["chunking_error"] = str(exc)
    else:
        _log(f"COLMAP chunked SfM disabled ({chunk_reason})")
        sfm_report["chunking_applied"] = False

    sparse_dir = run_colmap_sfm(
        frames_dir,
        workspace,
        sift_use_gpu=sift_use_gpu,
        mapper_num_threads=mapper_num_threads,
        matcher_mode=matcher_mode,
        sequential_overlap=sequential_overlap,
    )
    registered_images = _read_registered_image_count(sparse_dir)
    return sparse_dir, int(registered_images), sfm_report


# ---------------------------------------------------------------------------
# Stage 3: COLMAP Undistortion (required for 3DGRUT - PINHOLE only)
# ---------------------------------------------------------------------------
def run_colmap_undistort(frames_dir: Path, sparse_dir: Path,
                         workspace: Path) -> Path:
    """Undistort images to convert camera model to PINHOLE for 3DGRUT."""
    undistorted_dir = workspace / "undistorted"
    if undistorted_dir.exists():
        shutil.rmtree(undistorted_dir)
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
                         n_iterations: int = 7000, *,
                         max_n_gaussians: int = 0,
                         add_end_iteration: int = 0) -> dict:
    """Train 3DGRUT on undistorted COLMAP data and export USDZ + PLY."""
    threedgrut_dir = Path(THREEDGRUT_DIR)
    train_script = threedgrut_dir / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(f"3DGRUT not found at {threedgrut_dir}")

    grut_out = output_dir / "3dgrut"
    if grut_out.exists():
        shutil.rmtree(grut_out)
    grut_out.mkdir(parents=True, exist_ok=True)

    _log(f"Starting 3DGRUT training ({n_iterations} iterations)...")
    cmd = [
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
    ]
    if max_n_gaussians > 0:
        cmd.append(f"strategy.add.max_n_gaussians={max_n_gaussians}")
        _log(f"  max_n_gaussians override: {max_n_gaussians}")
    if add_end_iteration > 0:
        cmd.append(f"strategy.add.end_iteration={add_end_iteration}")
        _log(f"  add_end_iteration override: {add_end_iteration}")
    _run(cmd, cwd=str(threedgrut_dir))

    # Find the output directory (3DGRUT creates a nested structure)
    experiment_dirs = sorted(
        grut_out.rglob("export_last.usdz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
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
        "max_n_gaussians": max_n_gaussians,
        "add_end_iteration": add_end_iteration,
    }


# ---------------------------------------------------------------------------
# Stage 5: Fixer image refinement (optional, requires Cosmos/TE)
# ---------------------------------------------------------------------------
FIXER_IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.exr")
FIXER_COMPLETION_MARKER = ".fixer_stage_complete.json"


def _iter_image_outputs(directory: Path):
    for pattern in FIXER_IMAGE_PATTERNS:
        yield from directory.rglob(pattern)


def _has_image_outputs(directory: Path) -> bool:
    if not directory.exists():
        return False
    return any(True for _ in _iter_image_outputs(directory))


def _count_image_outputs(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in _iter_image_outputs(directory))


def _fixer_completion_marker_path(fixed_dir: Path) -> Path:
    return fixed_dir / FIXER_COMPLETION_MARKER


def _load_fixer_completion_marker(fixed_dir: Path) -> Dict[str, Any]:
    marker_path = _fixer_completion_marker_path(fixed_dir)
    if not _is_nonempty_file(marker_path):
        return {}
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_fixer_completion_marker(fixed_dir: Path, *, backend: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "backend": str(backend),
        "image_count": int(_count_image_outputs(fixed_dir)),
    }
    _fixer_completion_marker_path(fixed_dir).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def _clear_image_outputs(directory: Path) -> int:
    if not directory.exists():
        return 0
    removed = 0
    for path in _iter_image_outputs(directory):
        try:
            path.unlink()
            removed += 1
        except Exception:
            pass
    return removed


def _finalize_fixer_success(fixed_dir: Path, *, backend: str) -> bool:
    image_count = _count_image_outputs(fixed_dir)
    if image_count <= 0:
        _log("WARNING: Fixer completed but produced no refined images")
        return False
    marker = _write_fixer_completion_marker(fixed_dir, backend=backend)
    _log(
        "Fixer completion marker written "
        f"(backend={marker.get('backend')}, image_count={marker.get('image_count')})"
    )
    return True


def _run_fixer_local_stage(renders_dir: Path, fixed_dir: Path) -> bool:
    """Run Fixer locally on the current machine.

    Uses the nv-tlabs/Fixer inference script (Difix3D+ single-step diffusion).
    Expects:
      - FIXER_DIR (/opt/Fixer) with cloned nv-tlabs/Fixer repo
      - FIXER_WEIGHTS_DIR (/opt/fixer_weights) with HF download of nvidia/Fixer
    """
    fixer_dir = Path(FIXER_DIR)
    fixer_weights = Path(FIXER_WEIGHTS_DIR)
    inference_script = fixer_dir / "src" / "inference_pretrained_model.py"
    pretrained_path = fixer_weights / "pretrained" / "pretrained_fixer.pkl"

    if not inference_script.exists():
        _log(f"WARNING: Fixer source not found at {inference_script}; skipping local Fixer")
        return False
    if not pretrained_path.exists():
        _log(f"WARNING: Fixer weights not found at {pretrained_path}; skipping local Fixer")
        return False

    # Verify Cosmos base model files (DIT + VAE tokenizer)
    base_dit = fixer_weights / "base" / "model_fast_tokenizer.pt"
    base_vae = fixer_weights / "base" / "tokenizer_fast.pth"
    if not base_dit.exists() or not base_vae.exists():
        _log(f"WARNING: Fixer base models missing ({base_dit}, {base_vae}); skipping")
        return False

    # Fixer's Cosmos pipeline hardcodes weights at /work/models/{base,pretrained}.
    # Create symlinks so the inference script finds them regardless of our layout.
    # Best-effort: may fail on read-only root filesystems (e.g. macOS); Dockerfile
    # already bakes these symlinks so this is a runtime fallback only.
    try:
        work_models = Path("/work/models")
        work_models.mkdir(parents=True, exist_ok=True)
        for sub in ("base", "pretrained"):
            link = work_models / sub
            target = fixer_weights / sub
            if not link.exists() and target.is_dir():
                link.symlink_to(target)
                _log(f"  symlinked {link} -> {target}")
    except OSError as exc:
        _log(f"  NOTE: could not create /work/models symlinks ({exc}); "
             "assuming Dockerfile already set them up")

    fixed_dir.mkdir(parents=True, exist_ok=True)
    timestep = _env_int("FIXER_TIMESTEP", 250)
    resolution = _env_int("FIXER_RESOLUTION", 1024)
    fixer_python = os.getenv("FIXER_PYTHON", "python3")

    # Fail fast with a precise diagnostic when TE's compiled torch extension is missing.
    try:
        _run(
            [
                fixer_python,
                "-c",
                "import transformer_engine.pytorch as _te; print('TRANSFORMER_ENGINE_OK')",
            ],
            cwd=str(fixer_dir / "src"),
        )
    except RuntimeError:
        _log(
            "WARNING: Fixer runtime preflight failed: transformer_engine PyTorch extension is unavailable "
            "(missing transformer_engine_torch*.so). Install both transformer-engine and "
            "transformer-engine-cu12, then retry."
        )
        return False

    _log(f"Running Fixer image refinement locally (timestep={timestep}, res={resolution})...")

    # Use the system python3 (not sys.executable which may be python3.11 for 3DGRUT)
    # because cosmos-predict2, flash-attn, transformer-engine etc. are installed under python3.10.
    _run(
        [
            fixer_python,
            str(inference_script),
            "--model",
            str(pretrained_path),
            "--input",
            str(renders_dir),
            "--output",
            str(fixed_dir),
            "--timestep",
            str(timestep),
            "--resolution",
            str(resolution),
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
        "--fixer-dir",
        str(Path(FIXER_DIR)),
        "--fixer-weights-dir",
        str(Path(FIXER_WEIGHTS_DIR)),
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
      - auto: alias for local (H100 auto-routing disabled)
      - h100: explicit opt-in H100 stage
      - local: try only local stage
    """
    fixed_dir = output_dir / "fixer_output"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    marker_path = _fixer_completion_marker_path(fixed_dir)
    removed_images = _clear_image_outputs(fixed_dir)
    if removed_images > 0:
        _log(f"Removed {removed_images} stale image(s) from previous Fixer attempts")
    if marker_path.exists():
        try:
            marker_path.unlink()
        except Exception:
            pass
    mode_normalized = mode.strip().lower()

    if mode_normalized not in {"auto", "h100", "local"}:
        _log(f"WARNING: Unknown fixer mode '{mode}', falling back to local")
        mode_normalized = "local"

    # Keep auto behavior deterministic on single-GPU runners (e.g., RTX 4090):
    # auto now always uses local Fixer, without attempting H100 staging.
    if mode_normalized == "auto":
        _log("Fixer auto mode resolved to local backend (H100 auto-routing disabled)")
        mode_normalized = "local"

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
                if _finalize_fixer_success(fixed_dir, backend="h100"):
                    _log(f"Fixer output at: {fixed_dir} (backend=h100)")
                    return fixed_dir
                _log("WARNING: H100 Fixer stage returned success but outputs are incomplete")
        except RuntimeError as exc:
            _log(f"WARNING: H100 Fixer stage failed ({exc})")
        if mode_normalized == "h100":
            _log("WARNING: H100 Fixer requested but unavailable; using unrefined renders")
            return renders_dir

    if mode_normalized in {"auto", "local"}:
        try:
            if _run_fixer_local_stage(renders_dir, fixed_dir):
                if _finalize_fixer_success(fixed_dir, backend="local"):
                    _log(f"Fixer output at: {fixed_dir} (backend=local)")
                    return fixed_dir
                _log("WARNING: Local Fixer stage returned success but outputs are incomplete")
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
    """Generate a viewer-friendly mesh from dense fused point cloud.

    Quality strategy:
    - Use voxel downsampling (not random) to preserve spatial coverage.
    - Keep up to 2M points by default for room-scale detail.
    - Always use Poisson depth 12 for high-resolution reconstruction.
    - Transfer color via weighted KNN average (K=5) for smooth vertex colors.
    """
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

    # Voxel downsample instead of random to preserve spatial coverage.
    max_points = max(50000, _env_int("VISUAL_MESH_MAX_POINTS", 2000000))
    if point_count > max_points:
        # Estimate voxel size from bounding box to hit target point count.
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        volume = float(extent[0] * extent[1] * extent[2])
        voxel_size = max(0.001, (volume / max_points) ** (1.0 / 3.0))
        pcd = pcd.voxel_down_sample(voxel_size)
        point_count = len(pcd.points)
        _log(f"  Voxel downsampled to {point_count} points (voxel={voxel_size:.4f})")

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=64)
        )
        pcd.orient_normals_consistent_tangent_plane(100)

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

    # Weighted KNN color transfer (K=5) for smoother vertex colors.
    knn_k = max(1, _env_int("VISUAL_MESH_COLOR_KNN", 5))
    if pcd.has_colors() and len(pcd.colors) > 0 and len(mesh.vertices) > 0:
        tree = o3d.geometry.KDTreeFlann(pcd)
        pcd_colors = np.asarray(pcd.colors)
        vtx = np.asarray(mesh.vertices)
        out_colors = np.zeros((len(vtx), 3), dtype=np.float64)
        for i, vert in enumerate(vtx):
            _, idx, dist = tree.search_knn_vector_3d(vert, knn_k)
            if knn_k == 1 or len(idx) <= 1:
                out_colors[i] = pcd_colors[idx[0]]
            else:
                dist_arr = np.asarray(dist, dtype=np.float64)
                # Inverse-distance weights; guard against zero distance.
                weights = 1.0 / np.maximum(dist_arr, 1e-12)
                weights /= weights.sum()
                out_colors[i] = (pcd_colors[idx] * weights[:, None]).sum(axis=0)
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


def _prepare_texrecon_scene_with_refined_images(
    *,
    scene_dir: Path,
    refined_images_dir: Path,
    workspace: Path,
) -> tuple[Path | None, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "requested_dir": str(refined_images_dir),
        "used": False,
        "matched_images": 0,
        "total_images": 0,
        "coverage_ratio": 0.0,
        "reason": "",
    }
    scene_images = scene_dir / "images"
    if not scene_images.is_dir():
        report["reason"] = "scene_images_missing"
        return None, report
    if not refined_images_dir.is_dir():
        report["reason"] = "refined_images_missing"
        return None, report

    originals = sorted(p for p in scene_images.rglob("*") if p.is_file())
    if not originals:
        report["reason"] = "scene_images_empty"
        return None, report

    refined_index: Dict[str, Path] = {}
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for path in refined_images_dir.rglob(ext):
            if path.is_file():
                refined_index[path.name.lower()] = path

    matches: Dict[Path, Path] = {}
    for original in originals:
        replacement = refined_index.get(original.name.lower())
        if replacement is not None:
            matches[original] = replacement

    total_images = len(originals)
    matched_images = len(matches)
    coverage_ratio = float(matched_images / total_images) if total_images > 0 else 0.0
    report["total_images"] = int(total_images)
    report["matched_images"] = int(matched_images)
    report["coverage_ratio"] = float(coverage_ratio)

    min_coverage = min(1.0, max(0.0, _env_float("VISUAL_MESH_FIXER_IMAGE_COVERAGE_MIN", 0.95)))
    if coverage_ratio < min_coverage:
        report["reason"] = f"coverage_below_threshold:{coverage_ratio:.3f}<{min_coverage:.3f}"
        return None, report

    override_scene = workspace / "visual_texturing_scene_override"
    if override_scene.exists():
        shutil.rmtree(override_scene)
    override_images = override_scene / "images"
    override_images.mkdir(parents=True, exist_ok=True)

    for original in originals:
        rel = original.relative_to(scene_images)
        destination = override_images / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = matches.get(original, original)
        shutil.copy2(str(source), str(destination))

    sparse_src = scene_dir / "sparse"
    sparse_dst = override_scene / "sparse"
    if sparse_src.is_dir():
        shutil.copytree(sparse_src, sparse_dst)
    else:
        report["reason"] = "scene_sparse_missing"
        shutil.rmtree(override_scene, ignore_errors=True)
        return None, report

    report["used"] = True
    report["reason"] = "ok"
    report["scene_dir"] = str(override_scene)
    return override_scene, report


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
    refined_images_dir: Path | None = None,
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

    texrecon_scene_dir = scene_dir
    fixer_override_report: Dict[str, Any] = {
        "requested_dir": str(refined_images_dir) if refined_images_dir is not None else "",
        "used": False,
        "matched_images": 0,
        "total_images": 0,
        "coverage_ratio": 0.0,
        "reason": "not_requested" if refined_images_dir is None else "not_applied",
    }
    if refined_images_dir is not None:
        prepared_scene, override_report = _prepare_texrecon_scene_with_refined_images(
            scene_dir=scene_dir,
            refined_images_dir=refined_images_dir,
            workspace=workspace,
        )
        fixer_override_report = override_report
        if prepared_scene is not None:
            texrecon_scene_dir = prepared_scene

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

    max_points = max(100000, _env_int("VISUAL_MESH_TEXTURE_MAX_POINTS", 2000000))
    if point_count > max_points:
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        volume = float(extent[0] * extent[1] * extent[2])
        voxel_size = max(0.001, (volume / max_points) ** (1.0 / 3.0))
        pcd = pcd.voxel_down_sample(voxel_size)
        point_count = int(len(pcd.points))

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=64)
        )
        pcd.orient_normals_consistent_tangent_plane(100)

    depth = max(8, min(13, _env_int("VISUAL_MESH_TEXTURE_POISSON_DEPTH", 12)))
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
    textured_obj, texrecon_error = _run_texrecon(
        texrecon_scene_dir, base_mesh_obj, out_prefix, texturing_dir
    )
    if textured_obj is None:
        return {
            "ok": False,
            "method": "textured_colmap",
            "reason": f"texrecon_failed:{texrecon_error or 'unknown'}",
            "textured": False,
            "texture_image_count": 0,
            "atlas_resolution": 0,
            "uv_coverage_ratio": 0.0,
            "fixer_image_override": fixer_override_report,
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
        "fixer_image_override": fixer_override_report,
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
    refined_images_dir: Path | None = None,
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
        textured_kwargs: Dict[str, Any] = {
            "fused_ply": fused_ply,
            "output_glb": visual_mesh,
            "workspace": workspace,
            "target_faces": target_faces,
        }
        if refined_images_dir is not None:
            textured_kwargs["refined_images_dir"] = refined_images_dir
        textured_report = _build_visual_mesh_textured_colmap(
            **textured_kwargs,
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
    refinement_report: Mapping[str, Any] | None = None,
    hallucinated_region_mask: Path | None = None,
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
    refined_usdz_path = (
        output_dir / "export_last_refined.usdz"
        if (output_dir / "export_last_refined.usdz").is_file()
        and (output_dir / "export_last_refined.usdz").stat().st_size > 0
        else None
    )
    refined_ply_path = (
        output_dir / "export_last_refined.ply"
        if (output_dir / "export_last_refined.ply").is_file()
        and (output_dir / "export_last_refined.ply").stat().st_size > 0
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
    if refined_usdz_path is not None and refined_usdz_path.name != visual_usdz.name:
        assets.append(
            _entry(
                refined_usdz_path,
                role="volume_visual_refined",
                kind="usdz_nurec_volume_refined",
                viewer_hint="Refined neural volume visual with gap-filling distillation",
            )
        )
    if refined_ply_path is not None and refined_ply_path.name != gaussian_ply.name:
        assets.append(
            _entry(
                refined_ply_path,
                role="gaussian_pointcloud_refined",
                kind="ply_gaussian_refined",
                viewer_hint="Refined Gaussian pointcloud after pseudo-view distillation",
            )
        )
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
        "hallucinated_region_mask": (
            hallucinated_region_mask.name
            if hallucinated_region_mask is not None
            and hallucinated_region_mask.is_file()
            and hallucinated_region_mask.stat().st_size > 0
            else ""
        ),
        "assets": assets,
        "reports": {
            "visual": dict(visual_report),
            "collision": dict(collision_report),
            "refinement": dict(refinement_report) if isinstance(refinement_report, Mapping) else {},
        },
    }

    visual_preference = (os.getenv("NUREC_VISUAL_PRIMARY", "usdz") or "usdz").strip().lower()
    if visual_preference not in {"usdz", "mesh", "auto"}:
        visual_preference = "usdz"

    has_visual_mesh = visual_mesh_path is not None
    visual_is_textured = bool(visual_report.get("textured", False))
    mesh_compat = "generic_textured_mesh" if visual_is_textured else "fallback_vertex_mesh"
    payload["viewer_compatibility"] = ["omniverse_neural", mesh_compat]
    refined_primary_asset = ""
    if isinstance(refinement_report, Mapping):
        if str(refinement_report.get("status") or "").strip().lower() == "passed":
            refined_primary_asset = str(refinement_report.get("active_visual_asset") or "").strip()
    refined_primary_name = Path(refined_primary_asset).name if refined_primary_asset else ""
    if refined_primary_name and (output_dir / refined_primary_name).is_file():
        payload["primary_visual_asset"] = refined_primary_name
    elif visual_preference == "mesh" and has_visual_mesh:
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


def _prune_gaussian_splat(src_ply: Path, dst_ply: Path) -> bool:
    """Remove low-quality Gaussians: low opacity, oversized, and outliers.

    Returns True if pruning succeeded and *dst_ply* was written.
    """
    try:
        import numpy as np
        from plyfile import PlyData, PlyElement  # type: ignore[import-untyped]
    except ImportError:
        _log("  WARNING: plyfile not available; skipping Gaussian pruning")
        return False

    try:
        plydata = PlyData.read(str(src_ply))
    except Exception as exc:
        _log(f"  WARNING: could not read PLY for pruning: {exc}")
        return False

    vertex = plydata["vertex"]
    n_before = len(vertex.data)
    if n_before < 100:
        return False

    # Read key Gaussian properties if they exist.
    names = set(vertex.data.dtype.names or [])
    keep = np.ones(n_before, dtype=bool)

    # 1. Remove low-opacity Gaussians (< 0.01 after sigmoid).
    if "opacity" in names:
        opacity_raw = np.array(vertex["opacity"], dtype=np.float32)
        # Gaussian PLY stores opacity as inverse-sigmoid (logit).
        opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
        min_opacity = _env_float("PRUNE_MIN_OPACITY", 0.01)
        keep &= opacity >= min_opacity

    # 2. Remove oversized Gaussians (scale > 5× median).
    scale_names = [n for n in ("scale_0", "scale_1", "scale_2") if n in names]
    if len(scale_names) == 3:
        scales = np.stack(
            [np.array(vertex[s], dtype=np.float32) for s in scale_names],
            axis=1,
        )
        # Scales are stored as log-scale; exp to get actual.
        scales_actual = np.exp(scales)
        max_per_gaussian = np.max(scales_actual, axis=1)
        median_scale = float(np.median(max_per_gaussian))
        scale_multiplier = _env_float("PRUNE_MAX_SCALE_MULTIPLIER", 5.0)
        if median_scale > 0:
            keep &= max_per_gaussian <= median_scale * scale_multiplier

    # 3. Remove spatial outliers (> 1.5× IQR from median position).
    if "x" in names and "y" in names and "z" in names:
        xyz = np.stack(
            [
                np.array(vertex["x"], dtype=np.float32),
                np.array(vertex["y"], dtype=np.float32),
                np.array(vertex["z"], dtype=np.float32),
            ],
            axis=1,
        )
        centroid = np.median(xyz, axis=0)
        dists = np.linalg.norm(xyz - centroid, axis=1)
        q75 = float(np.percentile(dists, 75))
        q25 = float(np.percentile(dists, 25))
        iqr = q75 - q25
        outlier_factor = _env_float("PRUNE_OUTLIER_IQR_FACTOR", 3.0)
        cutoff = q75 + outlier_factor * iqr
        keep &= dists <= cutoff

    n_after = int(np.sum(keep))
    n_removed = n_before - n_after
    _log(
        f"  Gaussian pruning: {n_before} → {n_after} "
        f"(removed {n_removed}, {100.0 * n_removed / max(1, n_before):.1f}%)"
    )

    if n_after < 100:
        _log("  WARNING: pruning would remove almost all Gaussians; skipping")
        return False

    if n_removed == 0:
        return False

    pruned_data = vertex.data[keep]
    pruned_element = PlyElement.describe(pruned_data, "vertex")
    PlyData([pruned_element], text=False).write(str(dst_ply))
    return True


def _normalize_scene_cleaning_mode(raw: str) -> str:
    mode = (raw or "").strip().lower()
    if mode in {"off", "auto", "force"}:
        return mode
    _log(f"WARNING: Invalid scene cleaning mode {raw!r}; using 'off'")
    return "off"


def _normalize_mask_export_space(raw: str) -> str:
    mode = (raw or "").strip().lower()
    if mode in {"raw", "undistorted"}:
        return mode
    _log(f"WARNING: Invalid SAM3 mask export space {raw!r}; using 'raw'")
    return "raw"


def _quality_profile() -> str:
    return _env_choice("NUREC_QUALITY_PROFILE", "quality_first", ("quality_first", "balanced", "fast"))


def _quality_profile_defaults(profile: str) -> Dict[str, Any]:
    table: Dict[str, Dict[str, Any]] = {
        "quality_first": {
            "max_frames": 500,
            "extract_fps": 8,
            "n_iterations": 15000,
            "colmap_matcher_mode": "auto",
            "colmap_sequential_overlap": 30,
            "blur_filter_keep_ratio": 0.85,
            "blur_filter_min_frames": 120,
            "resume": False,
            "max_n_gaussians": 0,
        },
        "balanced": {
            "max_frames": 320,
            "extract_fps": 5,
            "n_iterations": 9000,
            "colmap_matcher_mode": "sequential",
            "colmap_sequential_overlap": 20,
            "blur_filter_keep_ratio": 0.90,
            "blur_filter_min_frames": 120,
            "resume": False,
            "max_n_gaussians": 0,
        },
        "fast": {
            "max_frames": 240,
            "extract_fps": 4,
            "n_iterations": 7000,
            "colmap_matcher_mode": "sequential",
            "colmap_sequential_overlap": 10,
            "blur_filter_keep_ratio": 1.0,
            "blur_filter_min_frames": 120,
            "resume": True,
            "max_n_gaussians": 500_000,
        },
    }
    return table.get(profile, table["quality_first"])


def _resolve_colmap_matcher_mode(requested_mode: str, frame_count: int) -> tuple[str, str]:
    mode = (requested_mode or "").strip().lower()
    if mode in {"sequential", "exhaustive"}:
        return mode, f"requested={mode}"
    if mode == "auto" or not mode:
        threshold = max(50, _env_int("COLMAP_AUTO_EXHAUSTIVE_MAX_FRAMES", 600))
        resolved = "exhaustive" if frame_count <= threshold else "sequential"
        return resolved, (
            "requested=auto "
            f"(frame_count={frame_count} threshold={threshold} -> {resolved})"
        )
    threshold = max(50, _env_int("COLMAP_AUTO_EXHAUSTIVE_MAX_FRAMES", 600))
    resolved = "exhaustive" if frame_count <= threshold else "sequential"
    _log(
        f"WARNING: Unknown COLMAP matcher mode {requested_mode!r}; "
        f"falling back to auto -> {resolved}"
    )
    return resolved, f"requested={requested_mode!r} fallback=auto(frame_count={frame_count})"


def _apply_open3d_thread_overrides() -> None:
    thread_count = max(0, _env_int("OPEN3D_CPU_THREADS", 0))
    if thread_count <= 0:
        return
    value = str(thread_count)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = value


def _resolve_visual_mesh_poisson_depth(point_count: int) -> int:
    base_depth = max(6, min(13, _env_int("VISUAL_MESH_POISSON_DEPTH", 12)))
    large_threshold = max(1, _env_int("VISUAL_MESH_POISSON_LARGE_THRESHOLD", 500000))
    large_depth = max(6, min(base_depth, _env_int("VISUAL_MESH_POISSON_DEPTH_LARGE", 12)))
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


def _read_3d_point_count(model_dir: Path) -> int:
    """Read number of 3D points from COLMAP's points3D.bin (uint64 LE header)."""
    points3d_bin = model_dir / "points3D.bin"
    if not _is_nonempty_file(points3d_bin):
        return 0
    try:
        data = points3d_bin.read_bytes()[:8]
        if len(data) < 8:
            return 0
        return int(struct.unpack("<Q", data)[0])
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


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not _is_nonempty_file(path):
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _video_signature(video_path: Path) -> Dict[str, Any]:
    signature: Dict[str, Any] = {
        "path": str(video_path),
    }
    try:
        stat = video_path.stat()
        signature["size_bytes"] = int(stat.st_size)
        signature["mtime_ns"] = int(stat.st_mtime_ns)
    except Exception:
        pass
    return signature


def _float_match(left: Any, right: Any, *, atol: float = 1e-6) -> bool:
    try:
        return abs(float(left) - float(right)) <= float(atol)
    except Exception:
        return False


def _load_stage14_resume_metadata(output_dir: Path) -> Dict[str, Any]:
    return _load_json_dict(output_dir / STAGE14_RESUME_METADATA)


def _write_stage14_resume_metadata(output_dir: Path, payload: Mapping[str, Any]) -> None:
    metadata_path = output_dir / STAGE14_RESUME_METADATA
    metadata_path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _validate_stage14_resume_metadata(
    metadata: Mapping[str, Any],
    *,
    profile: str,
    video_signature: Mapping[str, Any],
    requested_max_frames: int,
    effective_max_frames: int,
    requested_extract_fps: int,
    effective_extract_fps: float,
    blur_filter_keep_ratio: float,
    blur_filter_min_frames: int,
    n_iterations: int,
    max_n_gaussians: int = 0,
) -> list[str]:
    reasons: list[str] = []
    if str(metadata.get("schema_version") or "") != "v1":
        reasons.append("resume_metadata_schema_mismatch")
    if str(metadata.get("quality_profile") or "") != str(profile):
        reasons.append("quality_profile_changed")

    stage1 = metadata.get("stage1") if isinstance(metadata.get("stage1"), Mapping) else {}
    stage2 = metadata.get("stage2") if isinstance(metadata.get("stage2"), Mapping) else {}
    stage4 = metadata.get("stage4") if isinstance(metadata.get("stage4"), Mapping) else {}
    video_meta = metadata.get("video") if isinstance(metadata.get("video"), Mapping) else {}
    blur_meta = stage1.get("blur_filter") if isinstance(stage1.get("blur_filter"), Mapping) else {}

    expected_video_size = video_signature.get("size_bytes")
    actual_video_size = video_meta.get("size_bytes")
    if expected_video_size is not None and actual_video_size is not None:
        if int(expected_video_size) != int(actual_video_size):
            reasons.append("video_size_changed")

    expected_video_mtime = video_signature.get("mtime_ns")
    actual_video_mtime = video_meta.get("mtime_ns")
    if expected_video_mtime is not None and actual_video_mtime is not None:
        if int(expected_video_mtime) != int(actual_video_mtime):
            reasons.append("video_mtime_changed")

    if int(stage1.get("requested_max_frames", -1)) != int(requested_max_frames):
        reasons.append("requested_max_frames_changed")
    if int(stage1.get("effective_max_frames", -1)) != int(effective_max_frames):
        reasons.append("effective_max_frames_changed")
    if int(stage1.get("requested_extract_fps", -1)) != int(requested_extract_fps):
        reasons.append("requested_extract_fps_changed")
    if not _float_match(stage1.get("effective_extract_fps"), effective_extract_fps, atol=1e-4):
        reasons.append("effective_extract_fps_changed")
    if not _float_match(blur_meta.get("keep_ratio"), blur_filter_keep_ratio, atol=1e-6):
        reasons.append("blur_filter_keep_ratio_changed")
    saved_min = blur_meta.get("min_frames", blur_meta.get("min_keep", -1))
    if int(saved_min) != int(blur_filter_min_frames):
        reasons.append("blur_filter_min_frames_changed")
    if float(blur_filter_keep_ratio) < 1.0:
        blur_status = str(blur_meta.get("status") or "")
        if blur_status != "ok":
            reasons.append(f"prior_blur_filter_not_ok:{blur_status or 'missing'}")

    if int(stage4.get("n_iterations", -1)) != int(n_iterations):
        reasons.append("n_iterations_changed")
    # Compare against the requested value when available. This avoids false
    # mismatches when adaptive mode is requested (max_n_gaussians=0) and the
    # effective cached value is scene-dependent.
    cached_requested_gaussians = stage4.get("max_n_gaussians_requested")
    if cached_requested_gaussians is not None:
        if int(cached_requested_gaussians) != int(max_n_gaussians):
            reasons.append("max_n_gaussians_changed")
    elif int(max_n_gaussians) > 0:
        # Backward compatibility with older metadata that only stored effective
        # values: only compare when current run explicitly requests a fixed cap.
        cached_max_gaussians = stage4.get("max_n_gaussians")
        if cached_max_gaussians is not None and int(cached_max_gaussians) != int(max_n_gaussians):
            reasons.append("max_n_gaussians_changed")

    # Sanity-check cached workspace consistency (frames/sparse) against metadata.
    cached_frame_count = int(_count_extracted_frames(Path(str(metadata.get("frames_dir") or "")))) \
        if str(metadata.get("frames_dir") or "").strip() else -1
    if cached_frame_count >= 0 and int(stage1.get("frame_count", -1)) >= 0:
        if cached_frame_count != int(stage1.get("frame_count")):
            reasons.append("cached_frame_count_mismatch")
    cached_registered = int(_read_registered_image_count(Path(str(metadata.get("sparse_model_dir") or "")))) \
        if str(metadata.get("sparse_model_dir") or "").strip() else -1
    if cached_registered >= 0 and int(stage2.get("registered_images", -1)) >= 0:
        if cached_registered != int(stage2.get("registered_images")):
            reasons.append("cached_registered_images_mismatch")
    return reasons


def _resolve_stage14_resume(
    *,
    resume_requested: bool,
    quality_guardrails: bool,
    output_dir: Path,
    workspace: Path,
    profile: str,
    video_signature: Mapping[str, Any],
    requested_max_frames: int,
    effective_max_frames: int,
    requested_extract_fps: int,
    effective_extract_fps: float,
    blur_filter_keep_ratio: float,
    blur_filter_min_frames: int,
    n_iterations: int,
    max_n_gaussians: int = 0,
) -> tuple[bool, Dict[str, Any] | None, list[str]]:
    if not resume_requested:
        return False, None, ["resume_not_requested"]

    existing_grut_result = _load_existing_grut_result(output_dir)
    if existing_grut_result is None:
        return False, None, ["missing_stage4_exports"]

    if not quality_guardrails:
        return True, existing_grut_result, ["guardrails_disabled"]

    metadata = _load_stage14_resume_metadata(output_dir)
    if not metadata:
        return False, None, ["missing_stage14_resume_metadata"]

    metadata = dict(metadata)
    metadata["frames_dir"] = str(workspace / "frames")
    sparse_dir, _ = _select_best_reconstruction(workspace / "sparse")
    if sparse_dir is None:
        return False, None, ["cached_sparse_model_missing"]
    metadata["sparse_model_dir"] = str(sparse_dir) if sparse_dir is not None else ""

    reasons = _validate_stage14_resume_metadata(
        metadata,
        profile=profile,
        video_signature=video_signature,
        requested_max_frames=requested_max_frames,
        effective_max_frames=effective_max_frames,
        requested_extract_fps=requested_extract_fps,
        effective_extract_fps=effective_extract_fps,
        blur_filter_keep_ratio=blur_filter_keep_ratio,
        blur_filter_min_frames=blur_filter_min_frames,
        n_iterations=n_iterations,
        max_n_gaussians=max_n_gaussians,
    )
    if reasons:
        return False, None, reasons
    return True, existing_grut_result, ["metadata_match"]


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


def _resolve_post_stage4_refine_mode(requested_mode: str) -> str:
    mode = (requested_mode or "").strip().lower()
    if mode in {"off", "auto", "force"}:
        return mode
    _log(f"WARNING: Unknown --post-stage4-refine mode {requested_mode!r}; falling back to auto")
    return "auto"


def _apply_pipeline_mode_overrides(args: argparse.Namespace) -> None:
    mode = str(getattr(args, "pipeline_mode", "full") or "full").strip().lower()
    if mode not in {"full", "photorealistic_scene", "photoreal_hallucination"}:
        _log(f"WARNING: Unknown PIPELINE_MODE={mode!r}; falling back to 'full'")
        mode = "full"
    args.pipeline_mode = mode

    if mode == "photorealistic_scene":
        args.scene_cleaning_mode = "off"
        _log("PIPELINE_MODE=photorealistic_scene: scene cleaning disabled, 3DGRUT PLY is primary output")
        return

    if mode == "photoreal_hallucination":
        args.scene_cleaning_mode = "off"
        args.max_frames = max(int(args.max_frames), _env_int("HALLUCINATION_MIN_MAX_FRAMES", 500))
        args.extract_fps = max(int(args.extract_fps), _env_int("HALLUCINATION_MIN_EXTRACT_FPS", 8))
        args.n_iterations = max(int(args.n_iterations), _env_int("HALLUCINATION_MIN_ITERATIONS", 22000))
        desired_max_gaussians = max(10_000, _env_int("HALLUCINATION_MIN_MAX_N_GAUSSIANS", 500_000))
        args.max_n_gaussians = max(int(args.max_n_gaussians), desired_max_gaussians)
        args.blur_filter_keep_ratio = min(
            float(args.blur_filter_keep_ratio),
            max(0.0, min(1.0, _env_float("HALLUCINATION_MAX_BLUR_KEEP_RATIO", 0.70))),
        )
        args.colmap_matcher_mode = "sequential"
        args.colmap_sequential_overlap = max(
            int(args.colmap_sequential_overlap),
            _env_int("HALLUCINATION_MIN_COLMAP_OVERLAP", 40),
        )
        args.post_stage4_refine = "force"
        args.post_stage4_refine_model = "fixer+gsfix3d"
        args.post_stage4_max_pseudoviews = max(
            int(args.post_stage4_max_pseudoviews),
            _env_int("HALLUCINATION_MIN_MAX_PSEUDOVIEWS", 160),
        )
        args.post_stage4_distill_iters = max(
            int(args.post_stage4_distill_iters),
            _env_int("HALLUCINATION_MIN_DISTILL_ITERS", 6000),
        )
        args.post_stage4_time_budget_min = max(
            int(args.post_stage4_time_budget_min),
            _env_int("HALLUCINATION_MIN_TIME_BUDGET_MIN", 120),
        )
        # Keep void-fill loop disabled in this mode; hallucination comes from post-stage4 synthetic repair.
        args.void_fill_rounds = 0
        _log(
            "PIPELINE_MODE=photoreal_hallucination: "
            "applied clarity-first overrides (high-capacity baseline + forced synthetic repair)"
        )


def _resolve_refinement_quality_gate_profile(*, pipeline_mode: str) -> Dict[str, Any]:
    requested = (os.getenv("REFINEMENT_QUALITY_GATE_PROFILE", "auto").strip().lower() or "auto")
    if requested in {"default", "strict"}:
        resolved = "strict"
    elif requested in {"relaxed", "hallucination"}:
        resolved = "hallucination"
    elif requested == "auto":
        resolved = "hallucination" if pipeline_mode == "photoreal_hallucination" else "strict"
    else:
        _log(
            "WARNING: Unknown REFINEMENT_QUALITY_GATE_PROFILE="
            f"{requested!r}; falling back to auto"
        )
        resolved = "hallucination" if pipeline_mode == "photoreal_hallucination" else "strict"

    if resolved == "hallucination":
        defaults = {
            "min_hole_improvement_ratio": -0.20,
            "max_sharpness_drop_ratio": 0.30,
            "max_psnr_drop_db": 4.0,
            "enforce_psnr": False,
        }
    else:
        defaults = {
            "min_hole_improvement_ratio": 0.30,
            "max_sharpness_drop_ratio": 0.05,
            "max_psnr_drop_db": 0.50,
            "enforce_psnr": True,
        }

    min_hole_improvement_ratio = max(
        -1.0,
        min(
            1.0,
            _env_float(
                "REFINEMENT_GATE_MIN_HOLE_IMPROVEMENT_RATIO",
                defaults["min_hole_improvement_ratio"],
            ),
        ),
    )
    max_sharpness_drop_ratio = max(
        0.0,
        min(
            1.0,
            _env_float(
                "REFINEMENT_GATE_MAX_SHARPNESS_DROP_RATIO",
                defaults["max_sharpness_drop_ratio"],
            ),
        ),
    )
    max_psnr_drop_db = max(
        0.0,
        _env_float(
            "REFINEMENT_GATE_MAX_PSNR_DROP_DB",
            defaults["max_psnr_drop_db"],
        ),
    )
    enforce_psnr = _env_flag("REFINEMENT_GATE_ENFORCE_PSNR", defaults["enforce_psnr"])
    return {
        "requested_profile": requested,
        "resolved_profile": resolved,
        "min_hole_improvement_ratio": float(min_hole_improvement_ratio),
        "max_sharpness_drop_ratio": float(max_sharpness_drop_ratio),
        "max_psnr_drop_db": float(max_psnr_drop_db),
        "enforce_psnr": bool(enforce_psnr),
    }


def _has_valid_post_stage4_refine_cache(output_dir: Path) -> bool:
    gate = _load_json_dict(output_dir / "refinement_quality_gate.json")
    if str(gate.get("status") or "").strip().lower() != "passed":
        return False
    refined_usdz = output_dir / "export_last_refined.usdz"
    refined_ply = output_dir / "export_last_refined.ply"
    return _is_nonempty_file(refined_usdz) and _is_nonempty_file(refined_ply)


def _select_primary_renders_dir(result_dir: Path | str | None) -> Path | None:
    """Pick the most recent renders directory deterministically."""
    if result_dir is None:
        return None
    root = Path(str(result_dir))
    if not root.exists():
        return None
    candidates = [p for p in root.rglob("renders") if p.is_dir()]
    if not candidates:
        return None

    def _safe_mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except Exception:
            return 0.0

    def _newest_image_mtime(path: Path) -> float:
        newest = 0.0
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for image in path.glob(ext):
                newest = max(newest, _safe_mtime(image))
        return newest if newest > 0.0 else _safe_mtime(path)

    candidates.sort(
        key=lambda p: (_newest_image_mtime(p), _safe_mtime(p), -len(str(p))),
        reverse=True,
    )
    return candidates[0]


def _find_latest_checkpoint_in_result_dir(result_dir: Path | str | None) -> Path | None:
    if result_dir is None:
        return None
    candidates = sorted(Path(str(result_dir)).rglob("ckpt_last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _run_void_fill_loop(
    *,
    output_dir: Path,
    workspace: Path,
    undistorted_dir: Path,
    active_gaussian_ply: Path,
    active_visual_usdz: Path,
    active_ingp: Path | None,
    grut_result: Dict[str, Any],
    void_fill_rounds: int,
    void_fill_distill_iters: int,
    void_fill_target_hole_ratio: float,
    max_n_gaussians: int,
    time_budget_min: int,
) -> Dict[str, Any]:
    """Iterative void-fill loop: render virtual views → Fixer inpaint → distill back.

    Each round:
      1. Run gap analyzer with points3D to generate virtual cameras
      2. Render 3DGRUT from those virtual poses
      3. Inpaint the rendered images with Fixer (via view_repair)
      4. Distill repaired + virtual views back into 3DGRUT

    Stops based on virtual probe render statistics (p90 hole ratio), not
    pre-render global hole ratio.
    """
    import numpy as np
    import time as _time

    loop_started = _time.time()
    round_reports: List[Dict[str, Any]] = []
    min_hole_ratio = max(0.0, min(1.0, _env_float("VOID_FILL_MIN_HOLE_RATIO", 0.03)))
    max_hole_ratio = max(min_hole_ratio, min(1.0, _env_float("VOID_FILL_MAX_HOLE_RATIO", 0.98)))
    max_repair_per_round = max(1, _env_int("VOID_FILL_MAX_REPAIR_PER_ROUND", 24))

    # Track current checkpoint and PLY across rounds
    grut_result_dir = grut_result.get("result_dir")
    current_ckpt = _find_latest_checkpoint_in_result_dir(grut_result_dir)

    if current_ckpt is None:
        _log("WARNING: No 3DGRUT checkpoint found; cannot run void-fill loop")
        return {"status": "skipped_no_checkpoint", "rounds": []}

    sparse_dir = workspace / "undistorted" / "sparse" / "0"
    points3d_bin = sparse_dir / "points3D.bin"

    current_ply = active_gaussian_ply
    current_usdz = active_visual_usdz
    current_ingp = active_ingp
    best_ply = active_gaussian_ply
    best_usdz = active_visual_usdz

    for round_idx in range(void_fill_rounds):
        round_num = round_idx + 1
        round_dir = output_dir / f"void_fill_round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        _log(
            f"--- Void Fill Round {round_num}/{void_fill_rounds} "
            f"(hole_filter=[{min_hole_ratio:.0%},{max_hole_ratio:.0%}], max_repair={max_repair_per_round}) ---"
        )

        try:
            # 1. Gap analysis with virtual camera generation
            renders_dir = _select_primary_renders_dir(grut_result_dir)
            if renders_dir is None:
                _log(f"  Round {round_num}: No renders directory found; stopping loop")
                break

            gap_args: List[str] = [
                sys.executable,
                str(POST_STAGE4_GAP_ANALYZER_SCRIPT),
                "--renders-dir", str(renders_dir),
                "--output-dir", str(round_dir),
                "--max-candidate-views", "96",
                "--min-parallax-deg", "5.0",
                "--max-virtual-candidates", "48",
            ]
            colmap_images_txt = sparse_dir / "images.txt"
            colmap_images_bin = sparse_dir / "images.bin"
            if colmap_images_txt.is_file():
                gap_args.extend(["--colmap-images-txt", str(colmap_images_txt)])
            if colmap_images_bin.is_file():
                gap_args.extend(["--colmap-images-bin", str(colmap_images_bin)])
            if points3d_bin.is_file():
                gap_args.extend(["--colmap-points3d-bin", str(points3d_bin)])
            _run(gap_args)

            gap_report = _load_json_dict(round_dir / "gap_analysis_report.json")
            current_hole_ratio = float(gap_report.get("global_hole_pixel_ratio", 1.0))
            virtual_count = int(gap_report.get("virtual_candidates_selected", 0))

            _log(f"  Round {round_num}: hole_ratio={current_hole_ratio:.3f}, virtual_candidates={virtual_count}")

            if virtual_count == 0:
                status = "no_virtual_candidates"
                _log(f"  Round {round_num}: No virtual candidates generated; stopping ({status})")
                round_reports.append(
                    {
                        "round": round_num,
                        "status": status,
                        "hole_ratio": current_hole_ratio,
                        "probe_hole_ratio_mean": None,
                        "probe_hole_ratio_p90": None,
                        "probe_render_count": 0,
                    }
                )
                break

            # 2. Render virtual views
            _log(f"  Round {round_num}: Rendering {virtual_count} virtual views...")
            virtual_render_dir = round_dir / "virtual_renders"
            virtual_render_dir.mkdir(parents=True, exist_ok=True)

            vrender_args: List[str] = [
                sys.executable,
                str(POST_STAGE4_VIRTUAL_RENDER_SCRIPT),
                "--candidates-jsonl", str(round_dir / "gap_candidate_views.jsonl"),
                "--checkpoint", str(current_ckpt),
                "--reference-sparse-dir", str(sparse_dir),
                "--work-dir", str(virtual_render_dir),
                "--threedgrut-python", str(THREEDGRUT_PYTHON),
                "--threedgrut-dir", str(THREEDGRUT_DIR),
            ]
            _run(vrender_args)
            vrender_report = _load_json_dict(virtual_render_dir / "virtual_render_report.json")
            rendered_count = int(vrender_report.get("rendered_count", 0))

            if rendered_count == 0:
                _log(f"  Round {round_num}: No virtual views rendered; stopping")
                round_reports.append({"round": round_num, "status": "render_failed", "hole_ratio": current_hole_ratio})
                break

            mapping_path = Path(str(vrender_report.get("mapping_path", virtual_render_dir / "virtual_render_mapping.jsonl")))
            mapping_rows = _load_jsonl_rows(mapping_path)

            probe_hole_values: List[float] = []
            for row in mapping_rows:
                if not bool(row.get("render_exists")):
                    continue
                try:
                    hole_ratio = float(row.get("predicted_hole_ratio", 1.0))
                except Exception:
                    hole_ratio = 1.0
                if not math.isfinite(hole_ratio):
                    continue
                probe_hole_values.append(max(0.0, min(1.0, hole_ratio)))

            probe_render_count = len(probe_hole_values)
            probe_hole_ratio_mean = float(np.mean(probe_hole_values)) if probe_hole_values else None
            probe_hole_ratio_p90 = float(np.percentile(probe_hole_values, 90.0)) if probe_hole_values else None

            if probe_hole_ratio_p90 is not None and probe_hole_ratio_p90 <= void_fill_target_hole_ratio:
                _log(
                    f"  Round {round_num}: target met by probe p90 "
                    f"({probe_hole_ratio_p90:.3f} <= {void_fill_target_hole_ratio:.3f}); stopping"
                )
                round_reports.append(
                    {
                        "round": round_num,
                        "status": "target_met",
                        "hole_ratio": current_hole_ratio,
                        "probe_hole_ratio_mean": probe_hole_ratio_mean,
                        "probe_hole_ratio_p90": probe_hole_ratio_p90,
                        "probe_render_count": probe_render_count,
                        "virtual_rendered": int(rendered_count),
                        "virtual_selected_for_repair": 0,
                        "filtered_low_hole_count": 0,
                        "filtered_high_hole_count": 0,
                    }
                )
                break

            gap_candidates_path = round_dir / "gap_candidate_views.jsonl"
            gap_candidates = _load_jsonl_rows(gap_candidates_path)
            virtual_candidates_by_id: Dict[str, Dict[str, Any]] = {}
            for cand in gap_candidates:
                if not bool(cand.get("is_virtual")):
                    continue
                cand_id = str(cand.get("id") or "").strip()
                if cand_id:
                    virtual_candidates_by_id[cand_id] = cand

            filtered_mapping_rows: List[Dict[str, Any]] = []
            filtered_candidates: List[Dict[str, Any]] = []
            bounded_rows: List[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
            filtered_low_hole_count = 0
            filtered_high_hole_count = 0
            for row in mapping_rows:
                cand_id = str(row.get("candidate_id") or "").strip()
                if not cand_id:
                    continue
                render_exists = bool(row.get("render_exists"))
                try:
                    hole_ratio = float(row.get("predicted_hole_ratio", 1.0))
                except Exception:
                    hole_ratio = 1.0
                if not render_exists:
                    continue
                if hole_ratio < min_hole_ratio:
                    filtered_low_hole_count += 1
                    continue
                if hole_ratio > max_hole_ratio:
                    filtered_high_hole_count += 1
                    continue
                base = dict(virtual_candidates_by_id.get(cand_id, {}))
                base.update({
                    "id": cand_id,
                    "is_virtual": True,
                    "source_image": str(row.get("source_image") or base.get("source_image") or ""),
                    "render_image": "",
                    "render_name": str(row.get("render_name") or ""),
                    "qvec": row.get("qvec", base.get("qvec")),
                    "tvec": row.get("tvec", base.get("tvec")),
                    "camera_id": row.get("camera_id", base.get("camera_id")),
                    "predicted_hole_ratio": hole_ratio,
                })
                bounded_rows.append((hole_ratio, row, base))

            bounded_rows.sort(key=lambda item: item[0], reverse=True)
            selected_rows = bounded_rows[:max_repair_per_round]
            filtered_mapping_rows = [row for _hole, row, _base in selected_rows]
            filtered_candidates = [base for _hole, _row, base in selected_rows]

            filtered_candidates_path = round_dir / "gap_candidate_views_filtered.jsonl"
            filtered_mapping_path = round_dir / "virtual_render_mapping_filtered.jsonl"
            with filtered_candidates_path.open("w", encoding="utf-8") as f:
                for row in filtered_candidates:
                    f.write(json.dumps(row, ensure_ascii=True) + "\n")
            with filtered_mapping_path.open("w", encoding="utf-8") as f:
                for row in filtered_mapping_rows:
                    f.write(json.dumps(row, ensure_ascii=True) + "\n")

            if not filtered_candidates:
                _log(f"  Round {round_num}: No virtual candidates remained after bounded filtering; stopping")
                round_reports.append({
                    "round": round_num,
                    "status": "no_candidates_after_threshold",
                    "hole_ratio": current_hole_ratio,
                    "round_threshold": float(max_hole_ratio),
                    "virtual_rendered": int(rendered_count),
                    "virtual_selected_for_repair": 0,
                    "probe_hole_ratio_mean": probe_hole_ratio_mean,
                    "probe_hole_ratio_p90": probe_hole_ratio_p90,
                    "probe_render_count": probe_render_count,
                    "filtered_low_hole_count": int(filtered_low_hole_count),
                    "filtered_high_hole_count": int(filtered_high_hole_count),
                })
                break

            # 3. Inpaint virtual renders with Fixer
            _log(
                f"  Round {round_num}: Inpainting {len(filtered_candidates)} bounded-filtered virtual renders with Fixer..."
            )
            actual_renders_dir = Path(str(vrender_report.get("renders_dir", virtual_render_dir)))

            repair_args: List[str] = [
                sys.executable,
                str(POST_STAGE4_VIEW_REPAIR_SCRIPT),
                "--renders-dir", str(actual_renders_dir),
                "--candidate-views", str(filtered_candidates_path),
                "--virtual-render-mapping", str(filtered_mapping_path),
                "--output-dir", str(round_dir),
                "--model", "fixer",
            ]
            _run(repair_args)

            # 4. Distill back with virtual camera augmentation
            _log(f"  Round {round_num}: Distilling repaired views back into 3DGRUT...")
            distill_args: List[str] = [
                sys.executable,
                str(POST_STAGE4_DISTILL_SCRIPT),
                "--output-dir", str(round_dir),
                "--undistorted-dir", str(undistorted_dir),
                "--base-usdz", str(current_usdz),
                "--base-ply", str(current_ply),
                "--accepted-views-jsonl", str(round_dir / "accepted_repaired_views.jsonl"),
                "--repaired-views-dir", str(round_dir / "post_stage4_repaired_views"),
                "--distill-iters", str(void_fill_distill_iters),
                "--max-n-gaussians", str(max(0, max_n_gaussians)),
                "--time-budget-min", str(max(1, time_budget_min)),
                "--threedgrut-python", str(THREEDGRUT_PYTHON),
                "--threedgrut-dir", str(THREEDGRUT_DIR),
                "--virtual-renders-dir", str(actual_renders_dir),
                "--virtual-candidates-jsonl", str(filtered_candidates_path),
            ]
            if current_ingp is not None and _is_nonempty_file(current_ingp):
                distill_args.extend(["--base-ingp", str(current_ingp)])
            _run(distill_args)

            distill_report = _load_json_dict(round_dir / "post_stage4_distill_report.json")

            # Strict round acceptance: distill status + virtual append + refined assets + PSNR gate.
            refined_ply_path = round_dir / "export_last_refined.ply"
            refined_usdz_path = round_dir / "export_last_refined.usdz"
            refined_ingp_path = round_dir / "export_last_refined.ingp"

            baseline_metrics = grut_result.get("metrics") if isinstance(grut_result.get("metrics"), Mapping) else {}
            refined_metrics = distill_report.get("refined_metrics") if isinstance(distill_report.get("refined_metrics"), Mapping) else {}

            baseline_psnr = None
            refined_psnr = None
            try:
                baseline_psnr = float(baseline_metrics.get("mean_psnr"))
            except Exception:
                pass
            try:
                refined_psnr = float(refined_metrics.get("mean_psnr"))
            except Exception:
                pass

            psnr_drop = None
            if baseline_psnr is not None and refined_psnr is not None:
                psnr_drop = baseline_psnr - refined_psnr

            distill_status = str(distill_report.get("status") or "").strip().lower()
            distill_ok_flag = bool(distill_report.get("distill_ok", distill_status == "ok"))
            distill_ok = distill_ok_flag and distill_status == "ok"
            virtual_appended_count = int(distill_report.get("virtual_appended_count", 0) or 0)
            refined_ply_ok = _is_nonempty_file(refined_ply_path)

            rejection_reason = ""
            if not distill_ok:
                rejection_reason = f"distill_not_ok:{distill_status or 'missing'}"
            elif not refined_ply_ok:
                rejection_reason = "missing_refined_ply"
            elif virtual_appended_count <= 0:
                rejection_reason = "no_virtual_appended"
            elif baseline_psnr is not None and refined_psnr is None:
                rejection_reason = "missing_refined_psnr"
            elif psnr_drop is not None and psnr_drop > 0.5:
                rejection_reason = f"psnr_drop_exceeded:{psnr_drop:.3f}"

            round_status = "ok" if not rejection_reason else "rejected"
            if round_status == "ok":
                # Accept this round's output.
                current_ply = refined_ply_path
                best_ply = refined_ply_path
                if _is_nonempty_file(refined_usdz_path):
                    current_usdz = refined_usdz_path
                    best_usdz = refined_usdz_path
                if _is_nonempty_file(refined_ingp_path):
                    current_ingp = refined_ingp_path
                # Update checkpoint for next round
                new_ckpts = sorted(round_dir.rglob("ckpt_last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if new_ckpts:
                    current_ckpt = new_ckpts[0]
                # Update result_dir so next round's renders_dir lookup works
                new_result = distill_report.get("result_dir")
                if new_result:
                    grut_result_dir = new_result
                drop_text = f"{psnr_drop:.2f}" if psnr_drop is not None else "N/A"
                _log(
                    f"  Round {round_num}: Accepted (virtual_appended={virtual_appended_count}, PSNR drop={drop_text} dB)"
                )
            else:
                _log(f"  Round {round_num}: Rejected ({rejection_reason}); stopping loop")

            round_reports.append({
                "round": round_num,
                "status": round_status,
                "rejection_reason": rejection_reason,
                "hole_ratio": current_hole_ratio,
                "virtual_rendered": rendered_count,
                "virtual_selected_for_repair": len(filtered_candidates),
                "round_threshold": float(max_hole_ratio),
                "distill_status": distill_status,
                "distill_ok": bool(distill_ok),
                "virtual_appended_count": int(virtual_appended_count),
                "baseline_psnr": baseline_psnr,
                "refined_psnr": refined_psnr,
                "psnr_drop": psnr_drop,
                "refined_ply": str(refined_ply_path) if _is_nonempty_file(refined_ply_path) else "",
                "probe_hole_ratio_mean": probe_hole_ratio_mean,
                "probe_hole_ratio_p90": probe_hole_ratio_p90,
                "probe_render_count": probe_render_count,
                "filtered_low_hole_count": int(filtered_low_hole_count),
                "filtered_high_hole_count": int(filtered_high_hole_count),
            })

            if round_status != "ok":
                break

        except Exception as exc:
            _log(f"  Round {round_num}: Error: {exc}")
            round_reports.append({"round": round_num, "status": f"error:{exc}", "hole_ratio": 0.0})
            break

    report = {
        "status": "completed",
        "rounds_requested": void_fill_rounds,
        "rounds_completed": len(round_reports),
        "rounds": round_reports,
        "best_ply": str(best_ply),
        "best_usdz": str(best_usdz),
        "elapsed_sec": float(_time.time() - loop_started),
    }
    report_path = output_dir / "void_fill_loop_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _evaluate_refinement_quality_gate(
    *,
    baseline_hole_ratio: float,
    refined_hole_ratio: float,
    pre_sharpness: float,
    post_sharpness: float,
    baseline_psnr: float | None,
    refined_psnr: float | None,
    metric_basis: str = "candidate_pre_post_repair",
    min_hole_improvement_ratio: float = 0.30,
    max_sharpness_drop_ratio: float = 0.05,
    max_psnr_drop_db: float = 0.50,
    enforce_psnr_gate: bool = True,
    gate_profile: str = "strict",
) -> Dict[str, Any]:
    min_hole_improvement = max(-1.0, min(1.0, float(min_hole_improvement_ratio)))
    max_sharpness_drop = max(0.0, min(1.0, float(max_sharpness_drop_ratio)))
    max_psnr_drop = max(0.0, float(max_psnr_drop_db))

    baseline_hole = max(0.0, float(baseline_hole_ratio))
    refined_hole = max(0.0, float(refined_hole_ratio))
    hole_improvement = 0.0
    if baseline_hole > 1e-8:
        hole_improvement = (baseline_hole - refined_hole) / baseline_hole
    hole_gate_pass = hole_improvement >= min_hole_improvement

    sharp_drop = 0.0
    if pre_sharpness > 1e-8:
        sharp_drop = (float(pre_sharpness) - float(post_sharpness)) / float(pre_sharpness)
    sharpness_gate_pass = sharp_drop <= max_sharpness_drop

    psnr_gate_enforced = bool(enforce_psnr_gate) and baseline_psnr is not None and refined_psnr is not None
    psnr_drop = None
    psnr_gate_pass = True
    if psnr_gate_enforced:
        psnr_drop = float(baseline_psnr) - float(refined_psnr)
        psnr_gate_pass = psnr_drop <= max_psnr_drop

    status = "passed" if hole_gate_pass and sharpness_gate_pass and psnr_gate_pass else "failed_safe_rollback"
    return {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "status": status,
        "gate_profile": str(gate_profile or "strict"),
        "metric_basis": str(metric_basis or "candidate_pre_post_repair"),
        "thresholds": {
            "min_hole_improvement_ratio": min_hole_improvement,
            "max_sharpness_drop_ratio": max_sharpness_drop,
            "max_psnr_drop_db": max_psnr_drop,
        },
        "metrics": {
            "baseline_hole_ratio": baseline_hole,
            "refined_hole_ratio": refined_hole,
            "hole_improvement_ratio": hole_improvement,
            "pre_sharpness": float(pre_sharpness),
            "post_sharpness": float(post_sharpness),
            "sharpness_drop_ratio": sharp_drop,
            "baseline_psnr": float(baseline_psnr) if baseline_psnr is not None else None,
            "refined_psnr": float(refined_psnr) if refined_psnr is not None else None,
            "psnr_drop_db": float(psnr_drop) if psnr_drop is not None else None,
        },
        "gates": {
            "hole_improvement": bool(hole_gate_pass),
            "sharpness": bool(sharpness_gate_pass),
            "psnr": bool(psnr_gate_pass),
            "psnr_enforced": bool(psnr_gate_enforced),
        },
    }


def _run_stage9_sam3(
    *,
    output_dir: Path,
    workspace: Path,
    frames_dir: Path,
    undistorted_images_dir: Path,
    frame_count: int,
    requested_environment: str,
    requested_n_frames: int,
    requested_min_frame_detections: int,
    gaussian_ply: Path,
    resume: bool,
    scene_cleaning_mode: str,
    sam3_mask_export_space: str,
) -> Path | None:
    scene_semantics_path = output_dir / "scene_semantics_report.json"
    index_output = output_dir / "object_point_cloud_index.json"
    scene_cleaning_enabled = scene_cleaning_mode != "off"
    mask_export_space = _normalize_mask_export_space(sam3_mask_export_space)
    instance_masks_dir = output_dir / "instance_masks"

    sam3_frames_dir = frames_dir
    if scene_cleaning_enabled and mask_export_space == "undistorted":
        has_undistorted = undistorted_images_dir.is_dir() and any(
            p.is_file() for p in undistorted_images_dir.rglob("*")
        )
        if has_undistorted:
            sam3_frames_dir = undistorted_images_dir
        else:
            message = (
                "Stage 9 scene-cleaning prerequisites require undistorted images "
                f"at {undistorted_images_dir}, but none were found."
            )
            if scene_cleaning_mode == "force":
                raise RuntimeError(message)
            _log(f"WARNING: {message} Falling back to raw-frame SAM3 without mask export.")
            scene_cleaning_enabled = False

    if resume and _is_nonempty_file(scene_semantics_path) and _is_nonempty_file(index_output):
        if scene_cleaning_enabled:
            has_masks = instance_masks_dir.is_dir() and any(instance_masks_dir.glob("*.png"))
            if has_masks:
                _log("Resuming Stage 9: using existing scene semantics + SAM3 object index + instance masks")
                return index_output
            _log("Resume miss: instance masks required for scene cleaning are missing; rerunning Stage 9")
        else:
            _log("Resuming Stage 9: using existing scene semantics + SAM3 object index")
            return index_output

    scene_semantics_report = _infer_scene_semantics_report(
        frames_dir=sam3_frames_dir,
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
        colmap_sparse = None
        undist_sparse = workspace / "undistorted" / "sparse" / "0"
        if undist_sparse.exists():
            colmap_sparse = undist_sparse

        gaussian_ply_path = gaussian_ply if gaussian_ply.exists() else None
        sam3_frame_count = len(
            list(sam3_frames_dir.glob("*.jpg")) + list(sam3_frames_dir.glob("*.png"))
        )
        if sam3_frame_count > 0 and sam3_frame_count != frame_count:
            sam3_n_frames, sam3_min_frame_detections = _resolve_sam3_settings(
                environment=resolved_environment,
                frame_count=sam3_frame_count,
                requested_n_frames=requested_n_frames,
                requested_min_frame_detections=requested_min_frame_detections,
            )

        from sam3_detect import run_sam3_detection
        sam3_result = run_sam3_detection(
            frames_dir=sam3_frames_dir,
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
            save_instance_masks=scene_cleaning_enabled,
            instance_masks_dir=instance_masks_dir if scene_cleaning_enabled else None,
            force_full_video_masks=scene_cleaning_enabled,
        )

        n_objects = len(sam3_result.get("objects", []))
        _log(f"SAM3 detected {n_objects} objects")
        return index_output
    except Exception as exc:
        if scene_cleaning_mode == "force":
            raise
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
        "--max-n-gaussians",
        type=int,
        default=_env_int("MAX_N_GAUSSIANS", int(profile_defaults["max_n_gaussians"])),
        help="Max Gaussians for 3DGRUT MCMC (0=adaptive from SfM point count)",
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
        choices=["auto", "sequential", "exhaustive"],
        help="COLMAP matcher mode for SfM correspondence search",
    )
    parser.add_argument(
        "--colmap-sequential-overlap",
        type=int,
        default=_env_int("COLMAP_SEQUENTIAL_OVERLAP", int(profile_defaults["colmap_sequential_overlap"])),
        help="Temporal overlap window for sequential matcher",
    )
    parser.add_argument(
        "--colmap-chunked-mode",
        default=(os.getenv("COLMAP_CHUNKED_MODE", "auto").strip().lower() or "auto"),
        choices=["auto", "off", "on"],
        help="Chunked COLMAP SfM mode for long captures",
    )
    parser.add_argument(
        "--colmap-chunk-min-frames",
        type=int,
        default=_env_int("COLMAP_CHUNK_MIN_FRAMES", 900),
        help="Minimum extracted frames before chunked SfM auto-enables",
    )
    parser.add_argument(
        "--colmap-chunk-size-frames",
        type=int,
        default=_env_int("COLMAP_CHUNK_SIZE_FRAMES", 600),
        help="Chunk size (frames) for chunked SfM windows",
    )
    parser.add_argument(
        "--colmap-chunk-overlap-frames",
        type=int,
        default=_env_int("COLMAP_CHUNK_OVERLAP_FRAMES", 120),
        help="Chunk overlap (frames) between adjacent SfM windows",
    )
    parser.add_argument(
        "--colmap-chunk-max-chunks",
        type=int,
        default=_env_int("COLMAP_CHUNK_MAX_CHUNKS", 24),
        help="Maximum chunk windows allowed for chunked SfM",
    )
    parser.add_argument(
        "--colmap-chunk-matcher-mode",
        default=(os.getenv("COLMAP_CHUNK_MATCHER_MODE", "sequential").strip().lower() or "sequential"),
        choices=["sequential", "exhaustive"],
        help="Matcher mode used inside each chunk window",
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
        "--colmap-retry-matcher-mode",
        default=(os.getenv("COLMAP_RETRY_MATCHER_MODE", "auto").strip().lower() or "auto"),
        choices=["auto", "sequential", "exhaustive"],
        help="Matcher mode for SfM quality-gate retry",
    )
    parser.add_argument(
        "--blur-filter-keep-ratio",
        type=float,
        default=_env_float("BLUR_FILTER_KEEP_RATIO", float(profile_defaults["blur_filter_keep_ratio"])),
        help="Keep ratio of sharpest frames before SfM (1.0 disables filtering)",
    )
    parser.add_argument(
        "--blur-filter-min-frames",
        type=int,
        default=_env_int("BLUR_FILTER_MIN_FRAMES", int(profile_defaults["blur_filter_min_frames"])),
        help="Minimum number of frames to keep when blur filtering is enabled",
    )
    parser.add_argument(
        "--blur-filter-required",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("BLUR_FILTER_REQUIRED", profile == "quality_first"),
        help="Fail Stage 1 when blur filtering is enabled but blur scores are unavailable",
    )
    parser.add_argument(
        "--quality-guardrails",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("NUREC_QUALITY_GUARDRAILS", profile == "quality_first"),
        help="Strictly validate resume cache inputs/settings; recompute Stage 1-4 on mismatch",
    )
    parser.add_argument("--skip-fixer", action="store_true", help="Skip Fixer image refinement")
    parser.add_argument(
        "--fixer-mode",
        default=os.getenv("FIXER_MODE", "local"),
        choices=["auto", "local", "h100"],
        help="Fixer backend mode: local (default), auto (alias local), or explicit h100",
    )
    parser.add_argument(
        "--fixer-rerun",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("FIXER_RERUN", False),
        help="Force rerun of Stage 5 Fixer even when resume outputs exist",
    )
    parser.add_argument(
        "--fixer-required",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("FIXER_REQUIRED", False),
        help="Fail pipeline when Stage 5 Fixer does not produce refined outputs",
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
        "--post-stage4-refine",
        default=(os.getenv("POST_STAGE4_REFINE", "auto").strip().lower() or "auto"),
        choices=["off", "auto", "force"],
        help="Post-Stage-4 refinement mode: off, auto, or force",
    )
    parser.add_argument(
        "--post-stage4-refine-model",
        default=(os.getenv("POST_STAGE4_REFINE_MODEL", "fixer+gsfix3d").strip().lower() or "fixer+gsfix3d"),
        choices=["fixer", "fixer+gsfix3d"],
        help="Model stack for pseudo-view repair",
    )
    parser.add_argument(
        "--post-stage4-max-pseudoviews",
        type=int,
        default=_env_int("POST_STAGE4_MAX_PSEUDOVIEWS", 96),
        help="Maximum pseudo-views for gap-filling candidates",
    )
    parser.add_argument(
        "--post-stage4-distill-iters",
        type=int,
        default=_env_int("POST_STAGE4_DISTILL_ITERS", 3000),
        help="Distillation iterations for refined Stage-4 outputs",
    )
    parser.add_argument(
        "--post-stage4-time-budget-min",
        type=int,
        default=_env_int("POST_STAGE4_TIME_BUDGET_MIN", 90),
        help="Time budget (minutes) for post-Stage-4 distillation",
    )
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Skip dense reconstruction (use Gaussian PLY as mesh)",
    )
    parser.add_argument("--skip-sam3", action="store_true", help="Skip SAM3 object detection")
    parser.add_argument(
        "--scene-cleaning-mode",
        default=_normalize_scene_cleaning_mode(os.getenv("SCENE_CLEANING_MODE", "off")),
        choices=["off", "auto", "force"],
        help="Scene-cleaning integration mode: off (disabled), auto (best effort), force (hard fail)",
    )
    parser.add_argument(
        "--sam3-mask-export-space",
        default=_normalize_mask_export_space(os.getenv("SAM3_MASK_EXPORT_SPACE", "undistorted")),
        choices=["raw", "undistorted"],
        help="Image space for SAM3 instance-mask export when scene cleaning is enabled",
    )
    parser.add_argument(
        "--skip-scene-cleaning",
        action="store_true",
        help=argparse.SUPPRESS,
    )
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
        "--pipeline-mode",
        default=(os.getenv("PIPELINE_MODE", "full").strip().lower() or "full"),
        choices=["full", "photorealistic_scene", "photoreal_hallucination"],
        help=(
            "Pipeline mode: 'full' (default, all stages incl. scene cleaning), "
            "'photorealistic_scene' (skip Inpaint360GS, use 3DGRUT PLY directly, "
            "focus on gap-fill quality), "
            "'photoreal_hallucination' (clarity-first high-capacity baseline then "
            "forced synthetic repair with relaxed quality gates)"
        ),
    )
    parser.add_argument(
        "--void-fill-rounds",
        type=int,
        default=_env_int("VOID_FILL_ROUNDS", 0),
        help="Number of void-fill iterative rounds (0=disabled). Each round renders virtual cameras, inpaints with Fixer, and distills back.",
    )
    parser.add_argument(
        "--void-fill-target-hole-ratio",
        type=float,
        default=_env_float("VOID_FILL_TARGET_HOLE_RATIO", 0.05),
        help="Stop void-fill loop when virtual probe p90 hole ratio drops below this threshold",
    )
    parser.add_argument(
        "--void-fill-distill-iters",
        type=int,
        default=_env_int("VOID_FILL_DISTILL_ITERS", 5000),
        help="3DGRUT training iterations per void-fill distill round",
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
    args.scene_cleaning_mode = _normalize_scene_cleaning_mode(str(args.scene_cleaning_mode))
    args.sam3_mask_export_space = _normalize_mask_export_space(str(args.sam3_mask_export_space))
    if getattr(args, "skip_scene_cleaning", False):
        args.scene_cleaning_mode = "off"
    _apply_pipeline_mode_overrides(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_root = Path(args.storage_root)
    workspace = output_dir / "_colmap_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    existing_grut_result: Dict[str, Any] | None = None

    # Load job spec for raw_prefix if not provided
    raw_prefix = args.raw_prefix
    if not raw_prefix:
        spec = json.loads(Path(args.job_spec).read_text(encoding="utf-8"))
        raw_prefix = spec.get("capture", {}).get("raw_prefix_uri", "")

    video_path = find_video(raw_prefix, storage_root)
    video_duration_sec = _probe_video_duration_seconds(video_path)
    effective_max_frames, effective_max_frames_reason = _resolve_effective_max_frames(
        video_duration_sec,
        int(args.max_frames),
    )
    effective_extract_fps, effective_extract_fps_reason = _resolve_effective_extract_fps(
        video_duration_sec,
        int(args.extract_fps),
        int(effective_max_frames),
    )
    video_signature = _video_signature(video_path)
    effective_resume, existing_grut_result, resume_reasons = _resolve_stage14_resume(
        resume_requested=bool(args.resume),
        quality_guardrails=bool(args.quality_guardrails),
        output_dir=output_dir,
        workspace=workspace,
        profile=profile,
        video_signature=video_signature,
        requested_max_frames=int(args.max_frames),
        effective_max_frames=int(effective_max_frames),
        requested_extract_fps=int(args.extract_fps),
        effective_extract_fps=float(effective_extract_fps),
        blur_filter_keep_ratio=float(args.blur_filter_keep_ratio),
        blur_filter_min_frames=int(args.blur_filter_min_frames),
        n_iterations=int(args.n_iterations),
        max_n_gaussians=int(args.max_n_gaussians),
    )

    _log(f"Quality profile: {profile}")
    _log(
        "Quality guardrails: "
        f"{'enabled' if args.quality_guardrails else 'disabled'} "
        f"(resume_requested={bool(args.resume)} effective_resume={effective_resume})"
    )
    if args.resume:
        _log(f"Resume decision: {'accepted' if effective_resume else 'recompute Stage 1-4'}")
        if resume_reasons:
            _log(f"  Resume reason(s): {', '.join(resume_reasons)}")

    if args.dependency_preflight:
        _log("Running dependency preflight checks...")
        check_fused_ssim = args.preflight_check_fused_ssim and existing_grut_result is None
        if args.preflight_check_fused_ssim and existing_grut_result is not None:
            _log("Preflight: skipping fused_ssim ABI check (Stage 4 outputs already present for resume)")
        _run_dependency_preflight(check_fused_ssim=check_fused_ssim)

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
    if video_duration_sec is not None:
        _log(f"Video duration: {video_duration_sec:.1f}s")
    _log(f"Frame budget: requested={args.max_frames}, effective={effective_max_frames}")
    _log(f"Extraction FPS: requested={args.extract_fps}, effective={effective_extract_fps:.3f}")
    _log(f"  {effective_max_frames_reason}")
    _log(f"  {effective_extract_fps_reason}")
    frames_dir = workspace / "frames"
    if effective_resume:
        existing_frame_count = _count_extracted_frames(frames_dir)
        if existing_frame_count > 0:
            frame_count = existing_frame_count
            _log(f"Resuming Stage 1: using existing extracted frames ({frame_count})")
        else:
            frame_count = extract_frames(
                video_path,
                frames_dir,
                effective_max_frames,
                effective_extract_fps,
            )
    else:
        frame_count = extract_frames(
            video_path,
            frames_dir,
            effective_max_frames,
            effective_extract_fps,
        )

    if frame_count < 10:
        _log(f"WARNING: Only {frame_count} frames extracted. Reconstruction may fail.")

    blur_filter_status: Dict[str, Any] = {
        "enabled": bool(float(args.blur_filter_keep_ratio) < 1.0),
        "required": bool(args.blur_filter_required and float(args.blur_filter_keep_ratio) < 1.0),
        "status": "disabled",
        "keep_ratio": float(args.blur_filter_keep_ratio),
        "min_frames": int(args.blur_filter_min_frames),
    }
    if args.blur_filter_keep_ratio < 1.0:
        filtered_count = _apply_blur_frame_filter(
            frames_dir,
            keep_ratio=args.blur_filter_keep_ratio,
            min_keep=args.blur_filter_min_frames,
            fail_on_error=bool(args.blur_filter_required),
            status_out=blur_filter_status,
        )
        if filtered_count > 0:
            frame_count = filtered_count
    else:
        blur_filter_status["status"] = "disabled"

    capture_quality_report = build_capture_quality_report(frames_dir)
    capture_quality_report["frame_extraction"] = {
        "video_duration_sec": float(video_duration_sec) if video_duration_sec is not None else None,
        "requested_max_frames": int(args.max_frames),
        "effective_max_frames": int(effective_max_frames),
        "requested_extract_fps": int(args.extract_fps),
        "effective_extract_fps": float(effective_extract_fps),
        "adaptive_max_frames_reason": effective_max_frames_reason,
        "adaptive_extract_fps_reason": effective_extract_fps_reason,
        "blur_filter": dict(blur_filter_status),
    }
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
    effective_matcher_mode, matcher_mode_reason = _resolve_colmap_matcher_mode(
        args.colmap_matcher_mode,
        frame_count,
    )
    _log(f"COLMAP matcher: {effective_matcher_mode} ({matcher_mode_reason})")
    sparse_root = workspace / "sparse"
    sparse_dir: Path
    registered_images = 0
    sfm_run_report: Dict[str, Any] = {}
    if effective_resume:
        existing_sparse_dir, existing_sparse_count = _select_best_reconstruction(sparse_root, emit_logs=True)
        if existing_sparse_dir is not None and existing_sparse_count > 0:
            sparse_dir = existing_sparse_dir
            registered_images = int(existing_sparse_count)
            sfm_run_report = {
                "chunking_requested_mode": str(args.colmap_chunked_mode),
                "chunking_enabled": False,
                "chunking_reason": "resume_existing_sparse_model",
                "chunking_applied": False,
            }
            _log(
                "Resuming Stage 2: using existing COLMAP sparse model "
                f"{existing_sparse_dir} ({existing_sparse_count} images)"
            )
        else:
            sparse_dir, registered_images, sfm_run_report = _run_sfm_with_optional_chunking(
                frames_dir=frames_dir,
                workspace=workspace,
                sift_use_gpu=sift_use_gpu,
                mapper_num_threads=mapper_threads,
                matcher_mode=effective_matcher_mode,
                sequential_overlap=args.colmap_sequential_overlap,
                frame_count=frame_count,
                chunked_mode=args.colmap_chunked_mode,
                chunk_min_frames=args.colmap_chunk_min_frames,
                chunk_size_frames=args.colmap_chunk_size_frames,
                chunk_overlap_frames=args.colmap_chunk_overlap_frames,
                chunk_max_chunks=args.colmap_chunk_max_chunks,
                chunk_matcher_mode=args.colmap_chunk_matcher_mode,
            )
    else:
        sparse_dir, registered_images, sfm_run_report = _run_sfm_with_optional_chunking(
            frames_dir=frames_dir,
            workspace=workspace,
            sift_use_gpu=sift_use_gpu,
            mapper_num_threads=mapper_threads,
            matcher_mode=effective_matcher_mode,
            sequential_overlap=args.colmap_sequential_overlap,
            frame_count=frame_count,
            chunked_mode=args.colmap_chunked_mode,
            chunk_min_frames=args.colmap_chunk_min_frames,
            chunk_size_frames=args.colmap_chunk_size_frames,
            chunk_overlap_frames=args.colmap_chunk_overlap_frames,
            chunk_max_chunks=args.colmap_chunk_max_chunks,
            chunk_matcher_mode=args.colmap_chunk_matcher_mode,
        )

    raw_min_registered_ratio = max(0.0, min(1.0, float(args.colmap_min_registered_ratio)))
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

    min_registered_ratio, adaptive_ratio_reason = _resolve_effective_min_registered_ratio(
        requested_ratio=raw_min_registered_ratio,
        registered_images=registered_images,
        extracted_frames=frame_count,
    )
    _log(f"SfM retry threshold: {min_registered_ratio:.3f} ({adaptive_ratio_reason})")

    retry_matcher_mode, retry_matcher_reason = _resolve_colmap_retry_matcher_mode(
        args.colmap_retry_matcher_mode,
        frame_count,
    )
    retry_triggered = False
    if registered_ratio < min_registered_ratio:
        retry_triggered = True
        _log(
            "SfM coverage below target; forcing retry "
            f"(threshold={min_registered_ratio:.3f}, matcher={retry_matcher_mode})"
        )
        _log(f"  Retry matcher reason: {retry_matcher_reason}")
        db_path = workspace / "database.db"
        if db_path.exists():
            db_path.unlink()
        if sparse_root.exists():
            shutil.rmtree(sparse_root)

        retry_sequential_overlap = max(30, int(args.colmap_sequential_overlap))
        if retry_matcher_mode == "sequential":
            retry_sequential_overlap = max(
                retry_sequential_overlap,
                _env_int("COLMAP_RETRY_SEQUENTIAL_OVERLAP", 60),
            )
        sparse_dir, registered_images, retry_sfm_report = _run_sfm_with_optional_chunking(
            frames_dir=frames_dir,
            workspace=workspace,
            sift_use_gpu=sift_use_gpu,
            mapper_num_threads=mapper_threads,
            matcher_mode=retry_matcher_mode,
            sequential_overlap=retry_sequential_overlap,
            frame_count=frame_count,
            chunked_mode=args.colmap_chunked_mode,
            chunk_min_frames=args.colmap_chunk_min_frames,
            chunk_size_frames=args.colmap_chunk_size_frames,
            chunk_overlap_frames=args.colmap_chunk_overlap_frames,
            chunk_max_chunks=args.colmap_chunk_max_chunks,
            chunk_matcher_mode=args.colmap_chunk_matcher_mode,
        )
        sfm_run_report["retry"] = retry_sfm_report
        registered_ratio = _registration_ratio(
            registered_images=registered_images,
            extracted_frames=frame_count,
        )
        _log(
            "SfM retry coverage: "
            f"registered={registered_images}/{frame_count} "
            f"(ratio={registered_ratio:.3f})"
        )

    # Apply the same adaptive logic to the hard-fail gate: if the absolute
    # frame count is healthy, the reconstruction is viable.
    effective_retry_min, adaptive_fail_reason = _resolve_effective_min_registered_ratio(
        requested_ratio=retry_min_ratio,
        registered_images=registered_images,
        extracted_frames=frame_count,
    )
    if registered_ratio < effective_retry_min:
        raise RuntimeError(
            "COLMAP registration quality gate failed: "
            f"registered_ratio={registered_ratio:.3f} < effective_min={effective_retry_min:.3f} "
            f"({adaptive_fail_reason}). "
            "Capture likely has excessive blur/coverage gaps; rerun with steadier motion and more overlap."
        )

    capture_quality_report["sfm"] = {
        "registered_images": int(registered_images),
        "extracted_frames": int(frame_count),
        "matcher_mode_requested": str(args.colmap_matcher_mode),
        "matcher_mode_effective": str(effective_matcher_mode),
        "matcher_mode_reason": matcher_mode_reason,
        "retry_matcher_mode_requested": str(args.colmap_retry_matcher_mode),
        "retry_matcher_mode_effective": str(retry_matcher_mode),
        "retry_matcher_mode_reason": retry_matcher_reason,
        "retry_triggered": bool(retry_triggered),
        "registered_ratio": float(registered_ratio),
        "min_registered_ratio_requested": float(raw_min_registered_ratio),
        "min_registered_ratio_effective": float(min_registered_ratio),
        "min_registered_ratio_reason": adaptive_ratio_reason,
        "retry_min_registered_ratio": float(retry_min_ratio),
        "run_report": sfm_run_report,
    }
    capture_quality_path.write_text(json.dumps(capture_quality_report, indent=2), encoding="utf-8")

    # Read SfM 3D point count for adaptive Gaussian budget
    sfm_point_count = _read_3d_point_count(sparse_dir)
    _log(f"SfM 3D points: {sfm_point_count}")

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
    if effective_resume and _has_colmap_model(undistorted_model_dir) and has_undistorted_images:
        _log("Resuming Stage 3: using existing undistorted COLMAP workspace")
    else:
        undistorted_dir = run_colmap_undistort(frames_dir, sparse_dir, workspace)
    _export_undistorted_artifacts(output_dir=output_dir, undistorted_dir=undistorted_dir)

    # -----------------------------------------------------------------------
    # Stage 4: 3DGRUT Training → USDZ + PLY
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("STAGE 4: 3DGRUT Neural Reconstruction")
    _log("=" * 60)

    effective_max_n_gaussians, effective_add_end_iter, gaussians_reason = \
        _resolve_effective_max_n_gaussians(
            video_duration_sec=video_duration_sec,
            registered_frame_count=registered_images,
            sfm_point_count=sfm_point_count,
            n_iterations=int(args.n_iterations),
            requested_max_n_gaussians=int(args.max_n_gaussians),
        )
    _log(f"  {gaussians_reason}")

    capture_quality_report["grut"] = {
        "max_n_gaussians": int(effective_max_n_gaussians),
        "add_end_iteration": int(effective_add_end_iter),
        "sfm_point_count": int(sfm_point_count),
        "adaptive_reason": gaussians_reason,
    }
    capture_quality_path.write_text(json.dumps(capture_quality_report, indent=2), encoding="utf-8")

    if existing_grut_result is not None:
        grut_result = existing_grut_result
        _log("Resuming Stage 4: using existing 3DGRUT exports in output directory")
    else:
        grut_result = run_3dgrut_training(
            undistorted_dir,
            output_dir,
            args.n_iterations,
            max_n_gaussians=effective_max_n_gaussians,
            add_end_iteration=effective_add_end_iter,
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
    ingp_dst = (output_dir / "export_last.ingp").resolve()
    ingp_src_raw = grut_result.get("ingp")
    if ingp_src_raw:
        ingp_src = Path(str(ingp_src_raw)).resolve()
        if ingp_src.exists() and ingp_src != ingp_dst:
            shutil.copy2(str(ingp_src), str(ingp_dst))

    active_visual_usdz = usdz_dst
    active_gaussian_ply = ply_dst
    active_ingp = ingp_dst if _is_nonempty_file(ingp_dst) else None
    refinement_report: Dict[str, Any] = {
        "enabled": False,
        "mode": "off",
        "status": "skipped",
        "reason": "post_stage4_refine_disabled",
        "active_visual_asset": usdz_dst.name,
        "active_gaussian_asset": ply_dst.name,
    }
    hallucinated_region_mask: Path | None = None

    _write_stage14_resume_metadata(
        output_dir,
        {
            "schema_version": "v1",
            "generated_at": _utc_now_iso(),
            "quality_profile": str(profile),
            "video": dict(video_signature),
            "stage1": {
                "frame_count": int(frame_count),
                "requested_max_frames": int(args.max_frames),
                "effective_max_frames": int(effective_max_frames),
                "requested_extract_fps": int(args.extract_fps),
                "effective_extract_fps": float(effective_extract_fps),
                "blur_filter": dict(blur_filter_status),
            },
            "stage2": {
                "registered_images": int(registered_images),
                "registered_ratio": float(registered_ratio),
                "matcher_mode_effective": str(effective_matcher_mode),
            },
            "stage4": {
                "n_iterations": int(args.n_iterations),
                "max_n_gaussians_requested": int(args.max_n_gaussians),
                "max_n_gaussians": int(effective_max_n_gaussians),
                "add_end_iteration": int(effective_add_end_iter),
                "sfm_point_count": int(sfm_point_count),
                "result_dir": str(grut_result.get("result_dir") or ""),
                "export_usdz_bytes": int(usdz_dst.stat().st_size) if usdz_dst.exists() else 0,
                "export_ply_bytes": int(ply_dst.stat().st_size) if ply_dst.exists() else 0,
            },
        },
    )

    # -----------------------------------------------------------------------
    # Stage 5: Fixer Image Refinement (optional)
    # -----------------------------------------------------------------------
    fixer_refined_images_dir: Path | None = None
    if not args.skip_fixer:
        _log("=" * 60)
        _log("STAGE 5: Fixer Image Refinement")
        _log("=" * 60)
        _log(f"Fixer backend mode: {args.fixer_mode}")
        fixed_dir = output_dir / "fixer_output"
        completion_marker = _load_fixer_completion_marker(fixed_dir)
        has_valid_fixer_resume = bool(completion_marker) and _has_image_outputs(fixed_dir)
        should_run_fixer = True
        if effective_resume and args.fixer_rerun:
            _log("Resume override: forcing Stage 5 rerun (--fixer-rerun)")
        elif effective_resume and has_valid_fixer_resume:
            _log(
                "Resuming Stage 5: using existing Fixer outputs "
                f"(backend={completion_marker.get('backend')}, "
                f"image_count={completion_marker.get('image_count')})"
            )
            fixer_refined_images_dir = fixed_dir
            should_run_fixer = False
        elif effective_resume and _has_image_outputs(fixed_dir):
            _log(
                "Resume check: found Fixer images without completion marker; "
                "rerunning Stage 5 to avoid partial outputs"
            )
        if should_run_fixer:
            renders_dir = _select_primary_renders_dir(grut_result.get("result_dir"))
            if renders_dir is not None:
                fixer_result = run_fixer_refinement(
                    renders_dir,
                    output_dir,
                    mode=args.fixer_mode,
                    h100_script=Path(args.fixer_h100_script),
                    h100_instance_id=args.fixer_h100_instance_id.strip(),
                    h100_keep_instance=args.fixer_h100_keep_instance,
                    h100_max_hourly=args.fixer_h100_max_hourly,
                    h100_disk_gb=args.fixer_h100_disk_gb,
                )
                if fixer_result != renders_dir and _has_image_outputs(fixer_result):
                    fixer_refined_images_dir = fixer_result
                elif args.fixer_required:
                    raise RuntimeError(
                        "Fixer required but refinement outputs were unavailable; "
                        "set --no-fixer-required to allow fallback to unrefined renders"
                    )
            else:
                _log("WARNING: No rendered images found, skipping Fixer")
    else:
        _log("Skipping Fixer refinement (--skip-fixer)")

    # -----------------------------------------------------------------------
    # Stage 4.5/5A/5B/5C: Gap analysis + pseudo-view repair + distill + gate
    # -----------------------------------------------------------------------
    post_stage4_mode = _resolve_post_stage4_refine_mode(args.post_stage4_refine)
    gate_profile = _resolve_refinement_quality_gate_profile(
        pipeline_mode=str(getattr(args, "pipeline_mode", "full")),
    )
    _log(
        "Refinement quality gate profile: "
        f"{gate_profile['resolved_profile']} "
        f"(requested={gate_profile['requested_profile']}, "
        f"min_hole_improvement={gate_profile['min_hole_improvement_ratio']:.2f}, "
        f"max_sharpness_drop={gate_profile['max_sharpness_drop_ratio']:.2f}, "
        f"max_psnr_drop={gate_profile['max_psnr_drop_db']:.2f}, "
        f"enforce_psnr={gate_profile['enforce_psnr']})"
    )
    refinement_report["mode"] = post_stage4_mode
    if post_stage4_mode == "off":
        _log("Skipping post-Stage-4 refinement (--post-stage4-refine=off)")
    else:
        refined_usdz = output_dir / "export_last_refined.usdz"
        refined_ply = output_dir / "export_last_refined.ply"
        refined_ingp = output_dir / "export_last_refined.ingp"
        if (
            effective_resume
            and post_stage4_mode == "auto"
            and _has_valid_post_stage4_refine_cache(output_dir)
        ):
            active_visual_usdz = refined_usdz
            active_gaussian_ply = refined_ply
            active_ingp = refined_ingp if _is_nonempty_file(refined_ingp) else active_ingp
            gate = _load_json_dict(output_dir / "refinement_quality_gate.json")
            refinement_report = {
                "enabled": True,
                "mode": post_stage4_mode,
                "status": "passed",
                "reason": "resume_existing_refinement",
                "active_visual_asset": active_visual_usdz.name,
                "active_gaussian_asset": active_gaussian_ply.name,
                "quality_gate": gate,
            }
            mask_path = output_dir / "hallucinated_region_mask.png"
            if _is_nonempty_file(mask_path):
                hallucinated_region_mask = mask_path
            _log("Resuming post-Stage-4 refinement: using existing refined assets")
        else:
            _log("=" * 60)
            _log("STAGE 4.5/5A/5B/5C: Post-Stage-4 Gap Fill + Distill")
            _log("=" * 60)
            renders_dir = _select_primary_renders_dir(grut_result.get("result_dir"))
            required_scripts = [
                POST_STAGE4_GAP_ANALYZER_SCRIPT,
                POST_STAGE4_VIRTUAL_RENDER_SCRIPT,
                POST_STAGE4_VIEW_REPAIR_SCRIPT,
                POST_STAGE4_DISTILL_SCRIPT,
            ]
            missing_scripts = [str(p) for p in required_scripts if not p.is_file()]
            if missing_scripts:
                msg = f"missing post-stage4 scripts: {', '.join(missing_scripts)}"
                if post_stage4_mode == "force":
                    raise RuntimeError(msg)
                _log(f"WARNING: {msg}; skipping post-stage4 refinement")
                refinement_report = {
                    "enabled": False,
                    "mode": post_stage4_mode,
                    "status": "skipped",
                    "reason": msg,
                    "active_visual_asset": active_visual_usdz.name,
                    "active_gaussian_asset": active_gaussian_ply.name,
                }
            elif renders_dir is None:
                msg = "stage4_renders_missing"
                if post_stage4_mode == "force":
                    raise RuntimeError("Post-Stage-4 refinement requested but Stage-4 renders are unavailable")
                _log("WARNING: Stage-4 renders not found; skipping post-stage4 refinement")
                refinement_report = {
                    "enabled": False,
                    "mode": post_stage4_mode,
                    "status": "skipped",
                    "reason": msg,
                    "active_visual_asset": active_visual_usdz.name,
                    "active_gaussian_asset": active_gaussian_ply.name,
                }
            else:
                gap_report: Dict[str, Any] = {}
                view_repair_report: Dict[str, Any] = {}
                distill_report: Dict[str, Any] = {}
                gate_report: Dict[str, Any] = {}
                stage4_virtual_mapping_path: Path | None = None
                stage4_virtual_renders_dir: Path | None = None
                try:
                    # Stage 4.5: identify high-value hole regions and pseudo-view candidates.
                    gap_args: List[str] = [
                        sys.executable,
                        str(POST_STAGE4_GAP_ANALYZER_SCRIPT),
                        "--renders-dir",
                        str(renders_dir),
                        "--output-dir",
                        str(output_dir),
                        "--max-candidate-views",
                        str(max(1, int(args.post_stage4_max_pseudoviews))),
                        "--min-parallax-deg",
                        str(max(0.0, _env_float("POST_STAGE4_MIN_PARALLAX_DEG", 7.0))),
                        "--max-virtual-candidates",
                        str(max(1, _env_int("POST_STAGE4_MAX_VIRTUAL_CANDIDATES", 48))),
                    ]
                    colmap_images_txt = workspace / "undistorted" / "sparse" / "0" / "images.txt"
                    colmap_images_bin = workspace / "undistorted" / "sparse" / "0" / "images.bin"
                    if colmap_images_txt.is_file():
                        gap_args.extend(["--colmap-images-txt", str(colmap_images_txt)])
                    if colmap_images_bin.is_file():
                        gap_args.extend(["--colmap-images-bin", str(colmap_images_bin)])
                    colmap_points3d_bin = workspace / "undistorted" / "sparse" / "0" / "points3D.bin"
                    if colmap_points3d_bin.is_file():
                        gap_args.extend(["--colmap-points3d-bin", str(colmap_points3d_bin)])
                    _run(gap_args)
                    gap_report = _load_json_dict(output_dir / "gap_analysis_report.json")

                    virtual_selected = int(gap_report.get("virtual_candidates_selected", 0) or 0)
                    if virtual_selected > 0:
                        stage4_ckpt = _find_latest_checkpoint_in_result_dir(grut_result.get("result_dir"))
                        if stage4_ckpt is None:
                            _log("WARNING: no checkpoint found for Stage 4.5 virtual renders; virtual candidates may be rejected")
                        else:
                            stage4_virtual_work_dir = output_dir / "post_stage4_virtual_renders"
                            stage4_virtual_work_dir.mkdir(parents=True, exist_ok=True)
                            vrender_args: List[str] = [
                                sys.executable,
                                str(POST_STAGE4_VIRTUAL_RENDER_SCRIPT),
                                "--candidates-jsonl",
                                str(output_dir / "gap_candidate_views.jsonl"),
                                "--checkpoint",
                                str(stage4_ckpt),
                                "--reference-sparse-dir",
                                str(workspace / "undistorted" / "sparse" / "0"),
                                "--work-dir",
                                str(stage4_virtual_work_dir),
                                "--threedgrut-python",
                                str(THREEDGRUT_PYTHON),
                                "--threedgrut-dir",
                                str(THREEDGRUT_DIR),
                            ]
                            _run(vrender_args)
                            vrender_report = _load_json_dict(stage4_virtual_work_dir / "virtual_render_report.json")
                            stage4_virtual_mapping_path = Path(
                                str(vrender_report.get("mapping_path", stage4_virtual_work_dir / "virtual_render_mapping.jsonl"))
                            )
                            stage4_virtual_renders_dir = Path(
                                str(vrender_report.get("renders_dir", stage4_virtual_work_dir))
                            )

                    # Stage 5A: repair candidate pseudo-views with Fixer (+GSFix3D fallback).
                    view_repair_args: List[str] = [
                        sys.executable,
                        str(POST_STAGE4_VIEW_REPAIR_SCRIPT),
                        "--renders-dir",
                        str(renders_dir),
                        "--candidate-views",
                        str(output_dir / "gap_candidate_views.jsonl"),
                        "--output-dir",
                        str(output_dir),
                        "--model",
                        str(args.post_stage4_refine_model),
                    ]
                    if stage4_virtual_mapping_path is not None and stage4_virtual_mapping_path.is_file():
                        view_repair_args.extend(["--virtual-render-mapping", str(stage4_virtual_mapping_path)])
                    _run(view_repair_args)

                    # Stage 5B: distill accepted repaired views back into refined Gaussian outputs.
                    distill_args: List[str] = [
                        sys.executable,
                        str(POST_STAGE4_DISTILL_SCRIPT),
                        "--output-dir",
                        str(output_dir),
                        "--undistorted-dir",
                        str(undistorted_dir),
                        "--base-usdz",
                        str(usdz_dst),
                        "--base-ply",
                        str(ply_dst),
                        "--accepted-views-jsonl",
                        str(output_dir / "accepted_repaired_views.jsonl"),
                        "--repaired-views-dir",
                        str(output_dir / "post_stage4_repaired_views"),
                        "--distill-iters",
                        str(max(1, int(args.post_stage4_distill_iters))),
                        "--max-n-gaussians",
                        str(max(0, int(effective_max_n_gaussians))),
                        "--time-budget-min",
                        str(max(1, int(args.post_stage4_time_budget_min))),
                        "--threedgrut-python",
                        str(THREEDGRUT_PYTHON),
                        "--threedgrut-dir",
                        str(THREEDGRUT_DIR),
                    ]
                    if stage4_virtual_renders_dir is not None and stage4_virtual_renders_dir.is_dir():
                        distill_args.extend(["--virtual-renders-dir", str(stage4_virtual_renders_dir)])
                    if (output_dir / "gap_candidate_views.jsonl").is_file():
                        distill_args.extend(["--virtual-candidates-jsonl", str(output_dir / "gap_candidate_views.jsonl")])
                    if active_ingp is not None and _is_nonempty_file(active_ingp):
                        distill_args.extend(["--base-ingp", str(active_ingp)])
                    _run(distill_args)

                    gap_report = _load_json_dict(output_dir / "gap_analysis_report.json")
                    view_repair_report = _load_json_dict(output_dir / "view_repair_report.json")
                    distill_report = _load_json_dict(output_dir / "post_stage4_distill_report.json")

                    baseline_metrics = (
                        grut_result.get("metrics")
                        if isinstance(grut_result.get("metrics"), Mapping)
                        else {}
                    )
                    refined_metrics = (
                        distill_report.get("refined_metrics")
                        if isinstance(distill_report.get("refined_metrics"), Mapping)
                        else {}
                    )
                    baseline_psnr = None
                    refined_psnr = None
                    try:
                        baseline_psnr = float(baseline_metrics.get("mean_psnr"))
                    except Exception:
                        baseline_psnr = None
                    try:
                        refined_psnr = float(refined_metrics.get("mean_psnr"))
                    except Exception:
                        refined_psnr = None

                    candidate_pre_hole = float(
                        view_repair_report.get(
                            "pre_repair_hole_ratio_mean",
                            gap_report.get("global_hole_pixel_ratio", 1.0),
                        )
                    )
                    candidate_post_hole = float(
                        view_repair_report.get(
                            "post_repair_hole_ratio_mean",
                            candidate_pre_hole,
                        )
                    )
                    gate_report = _evaluate_refinement_quality_gate(
                        baseline_hole_ratio=candidate_pre_hole,
                        refined_hole_ratio=candidate_post_hole,
                        pre_sharpness=float(view_repair_report.get("pre_sharpness_mean", 0.0)),
                        post_sharpness=float(view_repair_report.get("post_sharpness_mean", 0.0)),
                        baseline_psnr=baseline_psnr,
                        refined_psnr=refined_psnr,
                        metric_basis="candidate_pre_post_repair",
                        min_hole_improvement_ratio=float(gate_profile["min_hole_improvement_ratio"]),
                        max_sharpness_drop_ratio=float(gate_profile["max_sharpness_drop_ratio"]),
                        max_psnr_drop_db=float(gate_profile["max_psnr_drop_db"]),
                        enforce_psnr_gate=bool(gate_profile["enforce_psnr"]),
                        gate_profile=str(gate_profile["resolved_profile"]),
                    )
                    gate_report["reports"] = {
                        "gap_analysis": "gap_analysis_report.json",
                        "view_repair": "view_repair_report.json",
                        "distill": "post_stage4_distill_report.json",
                    }
                    gate_report["refined_assets"] = {
                        "usdz": str(refined_usdz.name),
                        "ply": str(refined_ply.name),
                        "ingp": str(refined_ingp.name) if _is_nonempty_file(refined_ingp) else "",
                    }
                    (output_dir / "refinement_quality_gate.json").write_text(
                        json.dumps(gate_report, indent=2),
                        encoding="utf-8",
                    )

                    mask_path = output_dir / "hallucinated_region_mask.png"
                    if _is_nonempty_file(mask_path):
                        hallucinated_region_mask = mask_path

                    if (
                        str(gate_report.get("status") or "").strip().lower() == "passed"
                        and _is_nonempty_file(refined_usdz)
                        and _is_nonempty_file(refined_ply)
                    ):
                        active_visual_usdz = refined_usdz
                        active_gaussian_ply = refined_ply
                        active_ingp = refined_ingp if _is_nonempty_file(refined_ingp) else active_ingp
                        refinement_report = {
                            "enabled": True,
                            "mode": post_stage4_mode,
                            "status": "passed",
                            "reason": "quality_gate_passed",
                            "active_visual_asset": active_visual_usdz.name,
                            "active_gaussian_asset": active_gaussian_ply.name,
                            "quality_gate": gate_report,
                        }
                        _log("Post-Stage-4 refinement accepted: using refined Gaussian outputs")
                    else:
                        refinement_report = {
                            "enabled": True,
                            "mode": post_stage4_mode,
                            "status": "failed_safe_rollback",
                            "reason": str(gate_report.get("status") or "quality_gate_failed"),
                            "active_visual_asset": active_visual_usdz.name,
                            "active_gaussian_asset": active_gaussian_ply.name,
                            "quality_gate": gate_report,
                        }
                        _log("Post-Stage-4 refinement rejected by quality gate; rolling back to baseline Stage-4 outputs")
                except Exception as exc:
                    if post_stage4_mode == "force":
                        raise
                    _log(f"WARNING: Post-Stage-4 refinement failed ({exc}); rolling back to baseline Stage-4 outputs")
                    refinement_report = {
                        "enabled": True,
                        "mode": post_stage4_mode,
                        "status": "failed_safe_rollback",
                        "reason": f"runtime_error:{exc}",
                        "active_visual_asset": active_visual_usdz.name,
                        "active_gaussian_asset": active_gaussian_ply.name,
                    }
        gate_path = output_dir / "refinement_quality_gate.json"
        if not _is_nonempty_file(gate_path):
            gate_fallback = {
                "schema_version": "v1",
                "generated_at": _utc_now_iso(),
                "status": str(refinement_report.get("status") or "skipped"),
                "reason": str(refinement_report.get("reason") or "post_stage4_refine_not_run"),
            }
            gate_path.write_text(json.dumps(gate_fallback, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Stage 4.6: Iterative Void Fill Loop (optional)
    # -----------------------------------------------------------------------
    void_fill_rounds = getattr(args, "void_fill_rounds", 0)
    if void_fill_rounds > 0 and grut_result is not None:
        _log("=" * 60)
        _log(f"STAGE 4.6: Void Fill Loop ({void_fill_rounds} rounds)")
        _log("=" * 60)
        try:
            void_fill_report = _run_void_fill_loop(
                output_dir=output_dir,
                workspace=workspace,
                undistorted_dir=undistorted_dir,
                active_gaussian_ply=active_gaussian_ply,
                active_visual_usdz=active_visual_usdz,
                active_ingp=active_ingp,
                grut_result=grut_result,
                void_fill_rounds=void_fill_rounds,
                void_fill_distill_iters=max(1, getattr(args, "void_fill_distill_iters", 5000)),
                void_fill_target_hole_ratio=max(0.0, getattr(args, "void_fill_target_hole_ratio", 0.05)),
                max_n_gaussians=max(0, int(effective_max_n_gaussians)),
                time_budget_min=max(1, int(args.post_stage4_time_budget_min)),
            )
            best_ply = Path(str(void_fill_report.get("best_ply", "")))
            best_usdz = Path(str(void_fill_report.get("best_usdz", "")))
            if _is_nonempty_file(best_ply):
                active_gaussian_ply = best_ply
                _log(f"Void fill complete: using {best_ply.name}")
            if _is_nonempty_file(best_usdz):
                active_visual_usdz = best_usdz
        except Exception as exc:
            _log(f"WARNING: Void fill loop failed ({exc}); continuing with existing assets")

    # -----------------------------------------------------------------------
    # Stage 6: Dense Reconstruction → Collision Mesh
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("STAGE 6: Collision Mesh (nvblox_mesh.ply)")
    _log("=" * 60)
    mesh_ply = output_dir / "nvblox_mesh.ply"
    dense_dir = workspace / "dense"
    fused_ply_resume = dense_dir / "fused.ply"
    mesh_method_path = output_dir / "mesh_method.txt"
    collision_report_path = output_dir / "collision_mesh_report.json"
    dense_result: Dict[str, Any] | None = None
    reused_dense_stage6 = False

    if args.skip_dense:
        _log("--skip-dense: skipping PatchMatch; generating collision mesh from Gaussian PLY")
        if _mesh_with_open3d_poisson(active_gaussian_ply, mesh_ply, force=True):
            _validate_collision_mesh(mesh_ply)
            mesh_method = "poisson_open3d"
            # Use Gaussian PLY as the visual pointcloud source (no fused.ply available).
            fused_ply = active_gaussian_ply
            _log(f"  Collision mesh from Gaussian PLY: {mesh_ply}")
        else:
            raise RuntimeError(
                "--skip-dense collision mesh generation failed "
                "(Open3D Poisson could not mesh the Gaussian PLY)"
            )
        collision_report = _postprocess_collision_mesh(mesh_ply)
        collision_report_path.write_text(json.dumps(collision_report, indent=2), encoding="utf-8")
        _enforce_collision_spike_gate(collision_report)
    else:
        if (
            effective_resume
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
        if effective_resume and reused_dense_stage6 and _is_nonempty_file(collision_report_path):
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
        if effective_resume:
            existing_report = _load_existing_visual_report(output_dir)
            if existing_report is not None:
                _log("Resuming Stage 7: using existing visual mesh artifacts")
                _save_visual_report(output_dir, existing_report)
                return existing_report
        visual = build_visual_mesh_artifacts(
            output_dir=output_dir,
            fused_ply=fused_ply,
            gaussian_ply=active_gaussian_ply,
            workspace=workspace,
            refined_images_dir=fixer_refined_images_dir,
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
            undistorted_images_dir=undistorted_dir / "images",
            frame_count=frame_count,
            requested_environment=args.environment,
            requested_n_frames=args.sam3_n_frames,
            requested_min_frame_detections=args.sam3_min_frame_detections,
            gaussian_ply=active_gaussian_ply,
            resume=effective_resume,
            scene_cleaning_mode=args.scene_cleaning_mode,
            sam3_mask_export_space=args.sam3_mask_export_space,
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
    if effective_resume and _is_nonempty_file(occupancy_bin):
        _log("Resuming Stage 8: using existing occupancy grid")
    else:
        generate_occupancy(active_gaussian_ply, occupancy_bin)

    # -----------------------------------------------------------------------
    # Stage 8.5: Mesh Manifest (artifact roles + viewer guidance)
    # -----------------------------------------------------------------------
    write_mesh_manifest(
        output_dir=output_dir,
        visual_usdz=active_visual_usdz,
        gaussian_ply=active_gaussian_ply,
        collision_mesh_ply=mesh_ply,
        occupancy=occupancy_bin,
        visual_report=visual_report,
        collision_method=mesh_method,
        collision_report=collision_report,
        refinement_report=refinement_report,
        hallucinated_region_mask=hallucinated_region_mask,
    )

    # -----------------------------------------------------------------------
    # Stage 8.7: Photorealistic scene mode — promote 3DGRUT PLY + post-process
    # -----------------------------------------------------------------------
    if getattr(args, "pipeline_mode", "full") == "photorealistic_scene":
        _log("=" * 60)
        _log("STAGE 8.7: Photorealistic Scene — 3DGRUT PLY Promotion + Pruning")
        _log("=" * 60)

        # Promote the best available 3DGRUT PLY as the primary Gaussian artifact.
        primary_ply = output_dir / "photorealistic_gaussian.ply"
        source_ply = active_gaussian_ply  # refined or base
        _log(f"  Primary Gaussian source: {source_ply.name}")
        if source_ply.is_file():
            # Prune low-quality Gaussians from the PLY before promoting.
            pruned = _prune_gaussian_splat(source_ply, primary_ply)
            if pruned:
                _log(f"  Pruned PLY written to {primary_ply.name}")
                active_gaussian_ply = primary_ply
            else:
                shutil.copy2(str(source_ply), str(primary_ply))
                active_gaussian_ply = primary_ply
                _log(f"  Copied (no pruning) to {primary_ply.name}")
        else:
            _log(f"  WARNING: source PLY not found: {source_ply}")

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
        "export_last_refined.usdz",
        "export_last_refined.ply",
        "export_last_refined.ingp",
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
        "gap_analysis_report.json",
        "gap_candidate_views.jsonl",
        "view_repair_report.json",
        "accepted_repaired_views.jsonl",
        "post_stage4_distill_report.json",
        "refinement_quality_gate.json",
        "hallucinated_region_mask.png",
        "inpainted_visual_mesh.glb",
        "scene_cleaning_report.json",
        "photorealistic_gaussian.ply",
    ]:
        path = output_dir / artifact
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            _log(f"  ○ {artifact}: {size_mb:.1f}MB")

    _log(
        "  Active visual/canonical outputs: "
        f"visual={active_visual_usdz.name} gaussian={active_gaussian_ply.name}"
    )

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
