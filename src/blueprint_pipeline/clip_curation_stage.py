"""OSCAR-grade per-clip curation gates (SPEC-02, launch-audit-2026-07-02).

Runs after materialization and before retrieval/package export. Applies
per-clip quality gates modeled on OSCAR's data pipeline (arXiv 2606.04463),
which rejected 91.6% of source data with mechanism-specific filters:

1. ``min_frames``        — minimum clip length (OSCAR: >= 70 frames per clip).
2. ``camera_stability``  — pose-jitter bound from the camera trajectory. For
   robot-POV clips a static-camera constraint is enforced instead (OSCAR uses
   static-camera conditioning data). If poses are missing, the gate is marked
   ``not_measurable`` and fails closed unless config allows.
3. ``content_novelty``   — walkthrough clips must show non-trivial scene
   coverage: pose travel above a floor OR view-direction diversity above a
   floor (extends the existing stationary-pan dedup in
   ``retrieval_index_stage``).
4. ``sharpness``         — real Laplacian-variance sharpness. Metadata that is
   absent or a stamped constant (the geometry/video lanes currently write
   ``sharpness_score: 100.0`` / ``blur_score: 0.0`` for every frame) is
   treated as *unmeasured* — it is NEVER trusted. When frame image files are
   available the stage re-measures sharpness from pixels (cv2 when
   importable, else a pure-numpy Laplacian); otherwise the gate is
   ``not_measurable`` and fails closed unless config allows.
5. ``exposure``          — luminance-histogram clipping check: reject clips
   whose sampled frames have more than a configurable fraction of pixels
   crushed to black or clipped to white.

Doctrine: raw capture inputs are read-only. This stage only *reads* clip
records/frames and writes derived artifacts (curation + rejection manifests)
to a ``derived/clip_curation`` output directory. No measurement is ever
fabricated: anything that cannot be measured is reported as
``not_measurable`` and, by default, rejected (fail closed).

Usage (library):

    from blueprint_pipeline.clip_curation_stage import run_clip_curation_stage
    result = run_clip_curation_stage(bundle_dir=..., config=...)

Usage (CLI):

    python -m blueprint_pipeline.clip_curation_stage <bundle_dir> \
        [--config thresholds.yaml] [--output-dir OUT]

Input contract: ``<bundle_dir>/clips_manifest.json`` with
``{"clips": [{"clip_id", "clip_kind"?, "frames": [{"frame_id",
"timestamp"?, "T_world_camera"?, "sharpness_score"?, "blur_score"?,
"image_path"?}, ...]}, ...]}`` — or pass clip records directly to
:func:`curate_clips`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .common import PipelineError, read_json, utc_now_iso, write_json
from .logging_utils import log_event

logger = logging.getLogger("blueprint.clip_curation")

CURATION_MANIFEST_SCHEMA_VERSION = "clip_curation_manifest.v1"
CLIPS_MANIFEST_FILENAME = "clips_manifest.json"
CLIPS_JSONL_FILENAME = "clips.jsonl"
DEFAULT_OUTPUT_SUBDIR = Path("derived") / "clip_curation"

GATE_MIN_FRAMES = "min_frames"
GATE_CAMERA_STABILITY = "camera_stability"
GATE_CONTENT_NOVELTY = "content_novelty"
GATE_SHARPNESS = "sharpness"
GATE_EXPOSURE = "exposure"
ALL_GATES = (
    GATE_MIN_FRAMES,
    GATE_CAMERA_STABILITY,
    GATE_CONTENT_NOVELTY,
    GATE_SHARPNESS,
    GATE_EXPOSURE,
)

GATE_STATUS_PASSED = "passed"
GATE_STATUS_FAILED = "failed"
GATE_STATUS_NOT_MEASURABLE = "not_measurable"
GATE_STATUS_SKIPPED = "skipped"

# Stamped constants that geometry/video lanes are known to write today
# (geometry_stage.py:1190, geometry_sources.py:208 stamp sharpness 100.0;
# geometry_stage.py:679, video_to_world_service_runtime.py:181 stamp
# blur_score 0.0). These values are never trusted as measurements.
KNOWN_STAMPED_SHARPNESS_CONSTANTS = (100.0,)
KNOWN_STAMPED_BLUR_CONSTANTS = (0.0,)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipCurationConfig:
    """Thresholds for the per-clip curation gates.

    Defaults reference OSCAR (arXiv 2606.04463) where the paper pins a value,
    and this repo's existing ARKit-lane heuristics otherwise. Ops can tune
    per capture modality by loading overrides from YAML/JSON
    (:meth:`from_file` / :meth:`from_dict`).
    """

    profile_name: str = "oscar_static_robot_pov"

    # --- min clip length --------------------------------------------------
    # OSCAR: clips shorter than 70 frames are rejected outright.
    min_clip_frames: int = 70

    # --- camera stability -------------------------------------------------
    # RMS of the second difference of camera positions (metres). A smooth
    # constant-velocity walkthrough has ~0 jitter; handheld shake shows up
    # directly in this metric. OSCAR enforces a hard static-camera
    # constraint for world-model conditioning clips; for walkthrough capture
    # we bound motion smoothness instead of rejecting motion outright.
    max_pose_jitter_m: float = 0.02
    # Static-camera constraint for robot-POV clips (clip_kind ==
    # "robot_pov"): total pose travel must stay under this bound.
    max_static_camera_travel_m: float = 0.05
    enforce_static_camera_for_robot_pov: bool = True
    # Fail-closed policy: clips without pose trajectories are rejected
    # (gate not_measurable) unless explicitly allowed.
    allow_unmeasured_stability: bool = False

    # --- content / novelty (walkthrough clips) ----------------------------
    # A walkthrough clip must show non-trivial scene coverage: total pose
    # travel >= min_pose_travel_m OR view-direction spread >=
    # min_view_direction_spread_deg. (Extends the per-frame
    # _MIN_TRAVEL_M=0.07 / stationary-pan heuristics in
    # retrieval_index_stage to the clip level, per OSCAR's non-trivial
    # content requirement.)
    min_pose_travel_m: float = 0.5
    min_view_direction_spread_deg: float = 15.0
    allow_unmeasured_novelty: bool = False

    # --- sharpness ---------------------------------------------------------
    # Laplacian-variance floor; matches the existing ARKit-lane
    # _MIN_SHARPNESS=40.0 gate in retrieval_index_stage.
    min_sharpness_laplacian_var: float = 40.0
    # Metadata sharpness that is constant across the whole clip is treated
    # as a stamped placeholder (unmeasured), never as a measurement.
    treat_constant_metadata_as_unmeasured: bool = True
    # Always re-measure from pixels even when varied metadata is present.
    always_remeasure_sharpness: bool = False
    # Number of frames (evenly spaced) sampled for pixel measurements.
    max_measured_frames_per_clip: int = 8
    allow_unmeasured_sharpness: bool = False

    # --- exposure ----------------------------------------------------------
    # Luminance histogram clipping: a pixel is "crushed" when luma <=
    # crushed_luma_max and "clipped" when luma >= clipped_luma_min
    # (0..255 scale). The clip fails when the median sampled-frame
    # crushed+clipped fraction exceeds max_clipped_pixel_fraction.
    crushed_luma_max: float = 8.0
    clipped_luma_min: float = 247.0
    max_clipped_pixel_fraction: float = 0.25
    allow_unmeasured_exposure: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClipCurationConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise PipelineError(
                f"Unknown clip curation config keys: {unknown}; known keys: {sorted(known)}"
            )
        return cls(**dict(payload))

    @classmethod
    def from_file(cls, path: str | Path) -> "ClipCurationConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            payload = yaml.safe_load(text) or {}
        else:
            payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise PipelineError(f"Clip curation config at {path} must be a mapping")
        return cls.from_dict(payload)

    @classmethod
    def industrial_mobile_robot_pov(cls) -> "ClipCurationConfig":
        """Preset for roaming industrial humanoid/mobile-base robot POV clips."""

        return cls(
            profile_name="industrial_mobile_robot_pov",
            enforce_static_camera_for_robot_pov=False,
            max_pose_jitter_m=0.05,
        )

    @classmethod
    def from_profile(cls, profile: str | None) -> "ClipCurationConfig":
        normalized = (profile or "").strip().lower().replace("-", "_")
        if not normalized or normalized in {"default", "oscar", "oscar_static_robot_pov"}:
            return cls()
        if normalized in {"industrial_mobile_robot_pov", "industrial_mobile", "mobile_robot_pov"}:
            return cls.industrial_mobile_robot_pov()
        raise PipelineError(
            "Unknown clip curation profile "
            f"{profile!r}; known profiles: oscar_static_robot_pov, industrial_mobile_robot_pov"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Image helpers (shared with semantic_dedup_stage)
# ---------------------------------------------------------------------------


def _try_import_cv2() -> Optional[Any]:
    try:  # pragma: no cover - depends on environment
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def load_image_gray(path: Path) -> Optional[np.ndarray]:
    """Load an image as a float64 grayscale array on a 0..255 scale.

    Supports ``.npy`` arrays directly (2-D grayscale or HxWx3 RGB) so tests
    and offline lanes need no image codec, plus regular image files via cv2
    when importable, else Pillow. Returns None when the file is missing or
    unreadable — callers must treat that as *not measurable*, never as a
    default measurement.
    """
    if not path.is_file():
        return None
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 3:
                arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            if arr.ndim != 2:
                return None
            if arr.size and float(arr.max()) <= 1.0:
                arr = arr * 255.0
            return arr
        cv2 = _try_import_cv2()
        if cv2 is not None:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            return np.asarray(img, dtype=np.float64)
        from PIL import Image

        with Image.open(path) as img:
            return np.asarray(img.convert("L"), dtype=np.float64)
    except Exception:
        return None


def laplacian_variance(gray: np.ndarray) -> float:
    """Laplacian-variance sharpness of a grayscale image (0..255 scale).

    Uses ``cv2.Laplacian(...).var()`` when opencv is importable, otherwise a
    pure-numpy 4-neighbour Laplacian over the image interior so the gate
    works in codec-free test/CI environments.
    """
    gray = np.asarray(gray, dtype=np.float64)
    if gray.ndim != 2 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    cv2 = _try_import_cv2()
    if cv2 is not None:  # pragma: no cover - depends on environment
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(lap.var())


def luminance_clipped_fraction(
    gray: np.ndarray,
    *,
    crushed_luma_max: float,
    clipped_luma_min: float,
) -> float:
    """Fraction of pixels crushed to black or clipped to white."""
    gray = np.asarray(gray, dtype=np.float64)
    if gray.size == 0:
        return 1.0
    crushed = np.count_nonzero(gray <= crushed_luma_max)
    clipped = np.count_nonzero(gray >= clipped_luma_min)
    return float(crushed + clipped) / float(gray.size)


# ---------------------------------------------------------------------------
# Clip record helpers
# ---------------------------------------------------------------------------


def load_clip_records(bundle_dir: Path) -> List[Dict[str, Any]]:
    """Load clip records from a bundle directory (read-only)."""
    manifest_path = bundle_dir / CLIPS_MANIFEST_FILENAME
    if manifest_path.is_file():
        payload = read_json(manifest_path)
        clips = payload.get("clips")
        if not isinstance(clips, list):
            raise PipelineError(
                f"{manifest_path} must contain a 'clips' list; got {type(clips).__name__}"
            )
        return [dict(clip) for clip in clips]
    jsonl_path = bundle_dir / CLIPS_JSONL_FILENAME
    if jsonl_path.is_file():
        records: List[Dict[str, Any]] = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
    raise PipelineError(
        f"No {CLIPS_MANIFEST_FILENAME} or {CLIPS_JSONL_FILENAME} found in {bundle_dir}"
    )


def _clip_positions(frames: Sequence[Mapping[str, Any]]) -> Optional[np.ndarray]:
    positions: List[Tuple[float, float, float]] = []
    for frame in frames:
        T = frame.get("T_world_camera")
        if T is None:
            continue
        try:
            mat = np.asarray(T, dtype=np.float64)
            if mat.shape != (4, 4):
                continue
            positions.append((float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])))
        except Exception:
            continue
    if len(positions) < 2:
        return None
    return np.asarray(positions, dtype=np.float64)


def _clip_forward_vectors(frames: Sequence[Mapping[str, Any]]) -> Optional[np.ndarray]:
    forwards: List[np.ndarray] = []
    for frame in frames:
        T = frame.get("T_world_camera")
        if T is None:
            continue
        try:
            mat = np.asarray(T, dtype=np.float64)
            if mat.shape != (4, 4):
                continue
            # Camera convention: -Z is the viewing direction (ARKit/OpenGL).
            fwd = -mat[:3, 2]
            norm = float(np.linalg.norm(fwd))
            if norm <= 0.0:
                continue
            forwards.append(fwd / norm)
        except Exception:
            continue
    if len(forwards) < 2:
        return None
    return np.asarray(forwards, dtype=np.float64)


def _pose_travel_m(positions: np.ndarray) -> float:
    deltas = np.diff(positions, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _pose_jitter_rms_m(positions: np.ndarray) -> Optional[float]:
    if positions.shape[0] < 3:
        return None
    second_diff = positions[2:] - 2.0 * positions[1:-1] + positions[:-2]
    magnitudes = np.linalg.norm(second_diff, axis=1)
    return float(math.sqrt(float(np.mean(magnitudes**2))))


def _view_direction_spread_deg(forwards: np.ndarray, *, max_samples: int = 64) -> float:
    if forwards.shape[0] > max_samples:
        idx = np.linspace(0, forwards.shape[0] - 1, max_samples).round().astype(int)
        forwards = forwards[idx]
    sims = np.clip(forwards @ forwards.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(float(sims.min()))))


def _sample_frame_images(
    frames: Sequence[Mapping[str, Any]],
    bundle_dir: Optional[Path],
    max_frames: int,
) -> List[Tuple[str, np.ndarray]]:
    """Evenly sample frames that have readable image files. Read-only."""
    candidates = [f for f in frames if f.get("image_path")]
    if not candidates:
        return []
    if len(candidates) > max_frames:
        idx = np.linspace(0, len(candidates) - 1, max_frames).round().astype(int)
        candidates = [candidates[i] for i in idx]
    loaded: List[Tuple[str, np.ndarray]] = []
    for frame in candidates:
        raw_path = Path(str(frame["image_path"]))
        path = raw_path if raw_path.is_absolute() or bundle_dir is None else bundle_dir / raw_path
        gray = load_image_gray(path)
        if gray is not None:
            loaded.append((str(frame.get("frame_id", raw_path.name)), gray))
    return loaded


def _metadata_sharpness_status(
    frames: Sequence[Mapping[str, Any]],
    config: ClipCurationConfig,
) -> Tuple[Optional[float], str]:
    """Classify metadata sharpness as a trusted median or unmeasured.

    Returns (median_or_None, reason). Absent metadata, known stamped
    constants (100.0 sharpness / 0.0 blur), and clip-wide constant values
    are all classified as unmeasured — stamped scores from the
    geometry/video lanes must never pass the gate as if they were real.
    """
    scores: List[float] = []
    for frame in frames:
        value = frame.get("sharpness_score")
        if value is None:
            quality = frame.get("quality") or {}
            value = quality.get("sharpness_score")
        if value is None:
            continue
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            continue
    if not scores:
        return None, "metadata_absent"
    unique = set(scores)
    if unique <= set(KNOWN_STAMPED_SHARPNESS_CONSTANTS):
        return None, "metadata_stamped_constant"
    blur_scores: List[float] = []
    for frame in frames:
        value = frame.get("blur_score")
        if value is not None:
            try:
                blur_scores.append(float(value))
            except (TypeError, ValueError):
                continue
    if blur_scores and set(blur_scores) <= set(KNOWN_STAMPED_BLUR_CONSTANTS):
        return None, "metadata_stamped_constant"
    if (
        config.treat_constant_metadata_as_unmeasured
        and len(scores) >= 2
        and len(unique) == 1
    ):
        return None, "metadata_constant_across_clip"
    return float(np.median(np.asarray(scores, dtype=np.float64))), "metadata_measured"


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def _gate_result(
    status: str,
    *,
    value: Optional[float] = None,
    threshold: Optional[float] = None,
    reason: Optional[str] = None,
    detail: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"status": status}
    if value is not None:
        result["value"] = round(float(value), 6)
    if threshold is not None:
        result["threshold"] = float(threshold)
    if reason:
        result["reason"] = reason
    if detail:
        result["detail"] = dict(detail)
    return result


def _evaluate_min_frames(
    frames: Sequence[Mapping[str, Any]], config: ClipCurationConfig
) -> Dict[str, Any]:
    count = len(frames)
    if count < config.min_clip_frames:
        return _gate_result(
            GATE_STATUS_FAILED,
            value=count,
            threshold=config.min_clip_frames,
            reason=f"clip has {count} frames; floor is {config.min_clip_frames} (OSCAR >=70)",
        )
    return _gate_result(GATE_STATUS_PASSED, value=count, threshold=config.min_clip_frames)


def _evaluate_camera_stability(
    frames: Sequence[Mapping[str, Any]],
    clip_kind: str,
    config: ClipCurationConfig,
) -> Dict[str, Any]:
    positions = _clip_positions(frames)
    if positions is None:
        return _gate_result(
            GATE_STATUS_NOT_MEASURABLE,
            threshold=config.max_pose_jitter_m,
            reason="no usable pose trajectory; camera stability cannot be measured",
        )
    if clip_kind == "robot_pov" and config.enforce_static_camera_for_robot_pov:
        travel = _pose_travel_m(positions)
        if travel > config.max_static_camera_travel_m:
            return _gate_result(
                GATE_STATUS_FAILED,
                value=travel,
                threshold=config.max_static_camera_travel_m,
                reason="robot_pov clip violates static-camera constraint (OSCAR conditioning data)",
            )
        return _gate_result(
            GATE_STATUS_PASSED,
            value=travel,
            threshold=config.max_static_camera_travel_m,
            detail={"constraint": "static_camera"},
        )
    jitter = _pose_jitter_rms_m(positions)
    if jitter is None:
        return _gate_result(
            GATE_STATUS_NOT_MEASURABLE,
            threshold=config.max_pose_jitter_m,
            reason="fewer than 3 posed frames; jitter cannot be measured",
        )
    if jitter > config.max_pose_jitter_m:
        return _gate_result(
            GATE_STATUS_FAILED,
            value=jitter,
            threshold=config.max_pose_jitter_m,
            reason="pose jitter exceeds motion-smoothness bound",
        )
    return _gate_result(GATE_STATUS_PASSED, value=jitter, threshold=config.max_pose_jitter_m)


def _evaluate_content_novelty(
    frames: Sequence[Mapping[str, Any]],
    clip_kind: str,
    config: ClipCurationConfig,
) -> Dict[str, Any]:
    if clip_kind == "robot_pov":
        # Static-camera robot-POV clips are exempt from walkthrough scene
        # coverage; non-trivial *action* gating is SPEC-04 territory.
        return _gate_result(GATE_STATUS_SKIPPED, reason="novelty gate applies to walkthrough clips")
    positions = _clip_positions(frames)
    forwards = _clip_forward_vectors(frames)
    if positions is None and forwards is None:
        return _gate_result(
            GATE_STATUS_NOT_MEASURABLE,
            reason="no usable pose trajectory; scene-coverage novelty cannot be measured",
        )
    travel = _pose_travel_m(positions) if positions is not None else 0.0
    spread = _view_direction_spread_deg(forwards) if forwards is not None else 0.0
    detail = {
        "pose_travel_m": round(travel, 4),
        "min_pose_travel_m": config.min_pose_travel_m,
        "view_direction_spread_deg": round(spread, 2),
        "min_view_direction_spread_deg": config.min_view_direction_spread_deg,
    }
    if travel >= config.min_pose_travel_m or spread >= config.min_view_direction_spread_deg:
        return _gate_result(GATE_STATUS_PASSED, detail=detail)
    return _gate_result(
        GATE_STATUS_FAILED,
        reason="clip shows neither sufficient pose travel nor view-direction diversity",
        detail=detail,
    )


def _evaluate_sharpness(
    frames: Sequence[Mapping[str, Any]],
    bundle_dir: Optional[Path],
    config: ClipCurationConfig,
) -> Dict[str, Any]:
    metadata_median, metadata_reason = _metadata_sharpness_status(frames, config)
    if metadata_median is not None and not config.always_remeasure_sharpness:
        if metadata_median < config.min_sharpness_laplacian_var:
            return _gate_result(
                GATE_STATUS_FAILED,
                value=metadata_median,
                threshold=config.min_sharpness_laplacian_var,
                reason="metadata sharpness below floor",
                detail={"source": "metadata"},
            )
        return _gate_result(
            GATE_STATUS_PASSED,
            value=metadata_median,
            threshold=config.min_sharpness_laplacian_var,
            detail={"source": "metadata"},
        )
    images = _sample_frame_images(frames, bundle_dir, config.max_measured_frames_per_clip)
    if not images:
        return _gate_result(
            GATE_STATUS_NOT_MEASURABLE,
            threshold=config.min_sharpness_laplacian_var,
            reason=(
                f"sharpness unmeasured ({metadata_reason}) and no frame images available "
                "to re-measure; stamped constants are never trusted"
            ),
            detail={"metadata_classification": metadata_reason},
        )
    measured = float(np.median([laplacian_variance(gray) for _, gray in images]))
    detail = {
        "source": "measured_laplacian_variance",
        "metadata_classification": metadata_reason,
        "measured_frame_count": len(images),
    }
    if measured < config.min_sharpness_laplacian_var:
        return _gate_result(
            GATE_STATUS_FAILED,
            value=measured,
            threshold=config.min_sharpness_laplacian_var,
            reason="measured Laplacian-variance sharpness below floor",
            detail=detail,
        )
    return _gate_result(
        GATE_STATUS_PASSED,
        value=measured,
        threshold=config.min_sharpness_laplacian_var,
        detail=detail,
    )


def _evaluate_exposure(
    frames: Sequence[Mapping[str, Any]],
    bundle_dir: Optional[Path],
    config: ClipCurationConfig,
) -> Dict[str, Any]:
    images = _sample_frame_images(frames, bundle_dir, config.max_measured_frames_per_clip)
    if not images:
        return _gate_result(
            GATE_STATUS_NOT_MEASURABLE,
            threshold=config.max_clipped_pixel_fraction,
            reason="no frame images available; luminance histogram cannot be measured",
        )
    fractions = [
        luminance_clipped_fraction(
            gray,
            crushed_luma_max=config.crushed_luma_max,
            clipped_luma_min=config.clipped_luma_min,
        )
        for _, gray in images
    ]
    median_fraction = float(np.median(fractions))
    detail = {
        "measured_frame_count": len(images),
        "crushed_luma_max": config.crushed_luma_max,
        "clipped_luma_min": config.clipped_luma_min,
    }
    if median_fraction > config.max_clipped_pixel_fraction:
        return _gate_result(
            GATE_STATUS_FAILED,
            value=median_fraction,
            threshold=config.max_clipped_pixel_fraction,
            reason="crushed/clipped pixel fraction exceeds exposure bound",
            detail=detail,
        )
    return _gate_result(
        GATE_STATUS_PASSED,
        value=median_fraction,
        threshold=config.max_clipped_pixel_fraction,
        detail=detail,
    )


_UNMEASURED_ALLOW_FLAGS = {
    GATE_CAMERA_STABILITY: "allow_unmeasured_stability",
    GATE_CONTENT_NOVELTY: "allow_unmeasured_novelty",
    GATE_SHARPNESS: "allow_unmeasured_sharpness",
    GATE_EXPOSURE: "allow_unmeasured_exposure",
}


def evaluate_clip(
    clip: Mapping[str, Any],
    *,
    config: ClipCurationConfig,
    bundle_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate every curation gate for a single clip record (read-only)."""
    clip_id = str(clip.get("clip_id") or clip.get("id") or "unknown_clip")
    clip_kind = str(clip.get("clip_kind") or "walkthrough")
    frames = clip.get("frames") or []
    if not isinstance(frames, list):
        frames = []

    gate_results: Dict[str, Dict[str, Any]] = {
        GATE_MIN_FRAMES: _evaluate_min_frames(frames, config),
        GATE_CAMERA_STABILITY: _evaluate_camera_stability(frames, clip_kind, config),
        GATE_CONTENT_NOVELTY: _evaluate_content_novelty(frames, clip_kind, config),
        GATE_SHARPNESS: _evaluate_sharpness(frames, bundle_dir, config),
        GATE_EXPOSURE: _evaluate_exposure(frames, bundle_dir, config),
    }

    rejection_reasons: List[str] = []
    for gate, result in gate_results.items():
        status = result["status"]
        if status == GATE_STATUS_FAILED:
            rejection_reasons.append(f"{gate}: {result.get('reason', 'failed')}")
        elif status == GATE_STATUS_NOT_MEASURABLE:
            allow_flag = _UNMEASURED_ALLOW_FLAGS.get(gate)
            allowed = bool(allow_flag and getattr(config, allow_flag))
            result["fail_closed"] = not allowed
            if not allowed:
                rejection_reasons.append(
                    f"{gate}: not measurable and fail-closed "
                    f"(set {allow_flag}=true to allow)"
                )

    return {
        "clip_id": clip_id,
        "clip_kind": clip_kind,
        "frame_count": len(frames),
        "status": "rejected" if rejection_reasons else "accepted",
        "gate_results": gate_results,
        "rejection_reasons": rejection_reasons,
    }


def curate_clips(
    clips: Sequence[Mapping[str, Any]],
    *,
    config: Optional[ClipCurationConfig] = None,
    bundle_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Pure curation over clip records; returns the curation manifest payload.

    Reads clip records (and frame images when referenced) but writes nothing.
    """
    config = config or ClipCurationConfig()
    evaluations = [
        evaluate_clip(clip, config=config, bundle_dir=bundle_dir) for clip in clips
    ]
    accepted = [e for e in evaluations if e["status"] == "accepted"]
    rejected = [e for e in evaluations if e["status"] == "rejected"]

    gate_rejection_counts: Dict[str, int] = {gate: 0 for gate in ALL_GATES}
    for evaluation in rejected:
        for gate, result in evaluation["gate_results"].items():
            status = result["status"]
            if status == GATE_STATUS_FAILED or (
                status == GATE_STATUS_NOT_MEASURABLE and result.get("fail_closed")
            ):
                gate_rejection_counts[gate] += 1

    return {
        "schema_version": CURATION_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "config": config.to_dict(),
        "input_clip_count": len(evaluations),
        "accepted_clip_count": len(accepted),
        "rejected_clip_count": len(rejected),
        "accepted_clip_ids": [e["clip_id"] for e in accepted],
        "rejection_manifest": {
            "gate_rejection_counts": gate_rejection_counts,
            "rejected_clips": [
                {
                    "clip_id": e["clip_id"],
                    "rejection_reasons": e["rejection_reasons"],
                }
                for e in rejected
            ],
        },
        "clips": evaluations,
    }


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_clip_curation_stage(
    *,
    bundle_dir: str | Path,
    config: Optional[ClipCurationConfig] = None,
    config_path: Optional[str | Path] = None,
    profile: str | None = None,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Curate the clips of a bundle directory and write derived manifests.

    Raw capture inputs under ``bundle_dir`` are treated as read-only; the
    curation manifest and rejection manifest are written to
    ``<bundle_dir>/derived/clip_curation`` (or ``output_dir``).
    """
    bundle_dir = Path(bundle_dir)
    if len([item for item in (config, config_path, profile) if item is not None]) > 1:
        raise PipelineError("Pass only one of config, config_path, or profile")
    if config_path is not None:
        config = ClipCurationConfig.from_file(config_path)
    elif profile is not None:
        config = ClipCurationConfig.from_profile(profile)
    config = config or ClipCurationConfig()

    clips = load_clip_records(bundle_dir)
    manifest = curate_clips(clips, config=config, bundle_dir=bundle_dir)

    out_dir = Path(output_dir) if output_dir is not None else bundle_dir / DEFAULT_OUTPUT_SUBDIR
    manifest_path = out_dir / "clip_curation_manifest.json"
    rejection_path = out_dir / "clip_rejection_manifest.json"
    write_json(manifest_path, manifest)
    write_json(
        rejection_path,
        {
            "schema_version": CURATION_MANIFEST_SCHEMA_VERSION,
            "generated_at": manifest["generated_at"],
            **manifest["rejection_manifest"],
        },
    )

    log_event(
        logger,
        logging.INFO,
        "clip_curation_stage_complete",
        bundle_dir=bundle_dir,
        input_clip_count=manifest["input_clip_count"],
        accepted_clip_count=manifest["accepted_clip_count"],
        rejected_clip_count=manifest["rejected_clip_count"],
        gate_rejection_counts=manifest["rejection_manifest"]["gate_rejection_counts"],
        manifest_path=manifest_path,
    )

    return {
        "status": "completed",
        "bundle_dir": str(bundle_dir),
        "manifest_path": str(manifest_path),
        "rejection_manifest_path": str(rejection_path),
        "input_clip_count": manifest["input_clip_count"],
        "accepted_clip_count": manifest["accepted_clip_count"],
        "rejected_clip_count": manifest["rejected_clip_count"],
        "accepted_clip_ids": manifest["accepted_clip_ids"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m blueprint_pipeline.clip_curation_stage",
        description="Apply OSCAR-grade per-clip curation gates to a clip bundle.",
    )
    parser.add_argument("bundle_dir", type=Path, help="Bundle directory with clips_manifest.json")
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON threshold overrides")
    parser.add_argument(
        "--profile",
        choices=("oscar_static_robot_pov", "industrial_mobile_robot_pov"),
        default=None,
        help="Built-in curation profile; mutually exclusive with --config",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Derived artifact directory")
    args = parser.parse_args(argv)

    result = run_clip_curation_stage(
        bundle_dir=args.bundle_dir,
        config_path=args.config,
        profile=args.profile,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
