"""Tier-1 rollout-reliability gate for generated WAM evidence.

Flow-based screen answering one question before any evaluator spend: does a
generated rollout's visible motion agree with the commanded action stream in
presence and timing? A rollout that fails forces product-level abstention —
"abstaining when generated evidence is unreliable" at the evidence layer, in
the spirit of SC3-Eval's action-consistency termination but without a learned
inverse-dynamics model. A preregistered diagnostic evaluator may still score a
retained, technically valid failed rollout to measure how this gate relates to
rank fidelity; that diagnostic score cannot erase the reliability failure.

Scope limits, stated so callers do not over-claim:
- No camera calibration is used, so image-space *direction* agreement is not
  asserted — only motion presence, degeneracy, and timing correlation.
- Passing this gate is necessary, not sufficient, for causal action
  following; the paired counterfactual screen remains the scientific test.
- Threshold defaults were anchored on the campaign-3 validation set
  (real DROID compose vs. Cosmos3 clips) and must be recalibrated and frozen
  per experiment before scientific use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

TRANSLATION_SLICE = slice(0, 3)
ROT6D_SLICE = slice(3, 9)
GRIPPER_INDEX = 9
ROT6D_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
ACTION_DIM = 10

FLAG_DECODE_FAILED = "decode_failed"
FLAG_TOO_FEW_FRAMES = "too_few_frames"
FLAG_BLANK_FRAMES = "blank_frames"
FLAG_STATIC_UNDER_COMMAND = "static_under_command"
FLAG_MOTION_WITHOUT_COMMAND = "motion_without_command"
FLAG_TIMING_DISAGREEMENT = "timing_disagreement"
FLAG_TIMING_EVIDENCE_INSUFFICIENT = "timing_evidence_insufficient"
FLAG_INVALID_ACTION_ROT6D = "invalid_action_rot6d"

TIMING_SCOPE_WINDOW = "window"
TIMING_SCOPE_SESSION = "session"


@dataclass(frozen=True)
class ReliabilityThresholds:
    """Mechanism defaults; calibrate and freeze before scientific use."""

    downscale_width: int = 160
    blank_spatial_std_min: float = 4.0
    command_active_energy_min: float = 0.05
    command_null_energy_max: float = 0.01
    static_motion_max: float = 0.05
    null_motion_max: float = 0.60
    timing_correlation_min: float = 0.15
    timing_check_min_energy_std: float = 0.05
    rot6d_orthonormal_tol: float = 0.15


@dataclass(frozen=True)
class RolloutReliabilityReport:
    video_path: str
    n_frames: int
    n_action_steps: int
    flags: tuple[str, ...]
    reliable: bool
    command_energy_mean: float
    command_energy_std: float
    motion_mean: float
    motion_max: float
    timing_correlation: float | None
    timing_flag_scope: str
    spatial_std_mean: float

    def as_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "n_frames": self.n_frames,
            "n_action_steps": self.n_action_steps,
            "flags": list(self.flags),
            "reliable": self.reliable,
            "command_energy_mean": self.command_energy_mean,
            "command_energy_std": self.command_energy_std,
            "motion_mean": self.motion_mean,
            "motion_max": self.motion_max,
            "timing_correlation": self.timing_correlation,
            "timing_flag_scope": self.timing_flag_scope,
            "spatial_std_mean": self.spatial_std_mean,
        }


@dataclass(frozen=True)
class SessionReliabilityThresholds:
    """Prospective session-level timing rule; freeze explicitly per experiment."""

    timing_correlation_min: float = 0.15
    minimum_eligible_timing_windows: int = 3


@dataclass(frozen=True)
class SessionReliabilityReport:
    session_id: str
    n_windows: int
    hard_failure_window_count: int
    timing_eligible_window_count: int
    timing_correlation_median: float | None
    timing_correlation_values: tuple[float, ...]
    flags: tuple[str, ...]
    reliable: bool

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "n_windows": self.n_windows,
            "hard_failure_window_count": self.hard_failure_window_count,
            "timing_eligible_window_count": self.timing_eligible_window_count,
            "timing_correlation_median": self.timing_correlation_median,
            "timing_correlation_values": list(self.timing_correlation_values),
            "flags": list(self.flags),
            "reliable": self.reliable,
            "timing_aggregation_unit": "session",
            "timing_aggregation_statistic": "median_of_eligible_window_correlations",
        }


def action_energy_series(actions: np.ndarray) -> np.ndarray:
    """Per-step commanded-motion energy from a [T, 10] action chunk.

    Components (translation norm, rot6d identity deviation, gripper
    transition) are each max-normalized across the chunk before summing so a
    single unit convention cannot dominate the temporal shape.
    """
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
        raise ValueError(f"expected [T, {ACTION_DIM}] action chunk, got {arr.shape}")
    trans = np.linalg.norm(arr[:, TRANSLATION_SLICE], axis=1)
    rot = np.linalg.norm(arr[:, ROT6D_SLICE] - ROT6D_IDENTITY[None, :], axis=1)
    grip = np.abs(np.diff(arr[:, GRIPPER_INDEX], prepend=arr[0, GRIPPER_INDEX]))
    parts = []
    for comp in (trans, rot, grip):
        peak = float(np.max(comp))
        parts.append(comp / peak if peak > 0 else comp)
    return np.sum(parts, axis=0) / 3.0


def _read_frames_gray(video_path: Path, downscale_width: int) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            if w > downscale_width:
                nh = max(1, int(round(h * downscale_width / w)))
                gray = cv2.resize(gray, (downscale_width, nh), interpolation=cv2.INTER_AREA)
            frames.append(gray)
    finally:
        cap.release()
    return frames


def video_motion_series(video_path: Path, downscale_width: int = 160) -> tuple[np.ndarray, float, int]:
    """Camera-compensated per-frame-pair motion energy.

    Dense Farneback flow with the spatial-median flow vector subtracted
    before aggregation, so global camera translation does not read as scene
    motion. Returns (series, mean spatial std, frame count).
    """
    import cv2

    frames = _read_frames_gray(Path(video_path), downscale_width)
    if len(frames) < 2:
        return np.array([]), 0.0, len(frames)
    spatial_std = float(np.mean([float(np.std(f)) for f in frames]))
    series = []
    for a, b in zip(frames[:-1], frames[1:], strict=True):
        flow = cv2.calcOpticalFlowFarneback(
            a, b, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        med = np.median(flow.reshape(-1, 2), axis=0)
        residual = flow - med[None, None, :]
        series.append(float(np.mean(np.linalg.norm(residual, axis=2))))
    return np.asarray(series), spatial_std, len(frames)


def assess_rollout_reliability(
    video_path: str | Path,
    actions: np.ndarray,
    thresholds: ReliabilityThresholds | None = None,
    *,
    timing_flag_scope: Literal["window", "session"] = TIMING_SCOPE_SESSION,
) -> RolloutReliabilityReport:
    """Screen one generated rollout against its commanded action chunk.

    Phase-B callers use the default ``session`` scope: correlation is recorded
    here but cannot reject a single noisy window.  Historical reproduction of
    the lab gate can request ``window`` explicitly.
    """
    if timing_flag_scope not in {TIMING_SCOPE_WINDOW, TIMING_SCOPE_SESSION}:
        raise ValueError(f"unsupported_timing_flag_scope:{timing_flag_scope}")
    th = thresholds or ReliabilityThresholds()
    arr = np.asarray(actions, dtype=np.float64)
    energy = action_energy_series(arr)
    semantic_flags: list[str] = []
    r1, r2 = arr[:, 3:6], arr[:, 6:9]
    if (
        np.any(np.abs(np.linalg.norm(r1, axis=1) - 1.0) > th.rot6d_orthonormal_tol)
        or np.any(np.abs(np.linalg.norm(r2, axis=1) - 1.0) > th.rot6d_orthonormal_tol)
        or np.any(np.abs(np.sum(r1 * r2, axis=1)) > th.rot6d_orthonormal_tol)
    ):
        semantic_flags.append(FLAG_INVALID_ACTION_ROT6D)
    motion, spatial_std, n_frames = video_motion_series(Path(video_path), th.downscale_width)

    flags: list[str] = list(semantic_flags)
    if n_frames == 0:
        flags.append(FLAG_DECODE_FAILED)
    elif n_frames < 2 or motion.size == 0:
        flags.append(FLAG_TOO_FEW_FRAMES)

    corr: float | None = None
    motion_mean = float(np.mean(motion)) if motion.size else 0.0
    motion_max = float(np.max(motion)) if motion.size else 0.0
    energy_mean = float(np.mean(energy))
    energy_std = float(np.std(energy))

    if not flags:
        if spatial_std < th.blank_spatial_std_min:
            flags.append(FLAG_BLANK_FRAMES)
        commanded_active = energy_mean >= th.command_active_energy_min
        commanded_null = float(np.max(energy)) <= th.command_null_energy_max
        if commanded_active and motion_max < th.static_motion_max:
            flags.append(FLAG_STATIC_UNDER_COMMAND)
        if commanded_null and motion_mean > th.null_motion_max:
            flags.append(FLAG_MOTION_WITHOUT_COMMAND)
        n = min(len(energy), len(motion))
        if (
            n >= 4
            and energy_std >= th.timing_check_min_energy_std
            and float(np.std(motion[:n])) > 0
        ):
            corr = float(np.corrcoef(energy[:n], motion[:n])[0, 1])
            if (
                timing_flag_scope == TIMING_SCOPE_WINDOW
                and commanded_active
                and corr < th.timing_correlation_min
                and FLAG_STATIC_UNDER_COMMAND not in flags
            ):
                flags.append(FLAG_TIMING_DISAGREEMENT)

    return RolloutReliabilityReport(
        video_path=str(video_path),
        n_frames=n_frames,
        n_action_steps=int(np.asarray(actions).shape[0]),
        flags=tuple(flags),
        reliable=not flags,
        command_energy_mean=energy_mean,
        command_energy_std=energy_std,
        motion_mean=motion_mean,
        motion_max=motion_max,
        timing_correlation=corr,
        timing_flag_scope=timing_flag_scope,
        spatial_std_mean=spatial_std,
    )


def assess_session_reliability(
    session_id: str,
    windows: Sequence[RolloutReliabilityReport],
    thresholds: SessionReliabilityThresholds,
) -> SessionReliabilityReport:
    """Aggregate timing at the independent session unit.

    Non-timing defects remain immediate hard failures. Timing correlation is
    evaluated only from otherwise-valid windows, using a prospectively frozen
    minimum count and median threshold. Insufficient timing evidence abstains.
    """
    if not session_id:
        raise ValueError("session_id_required")
    if thresholds.minimum_eligible_timing_windows <= 0:
        raise ValueError("minimum_eligible_timing_windows_must_be_positive")
    if not windows:
        raise ValueError("session_windows_required")
    if any(window.timing_flag_scope != TIMING_SCOPE_SESSION for window in windows):
        raise ValueError("session_aggregation_requires_session_scoped_windows")

    hard_flags: list[str] = []
    hard_failure_windows = 0
    correlations: list[float] = []
    for window in windows:
        window_hard_flags = [flag for flag in window.flags if flag != FLAG_TIMING_DISAGREEMENT]
        if window_hard_flags:
            hard_failure_windows += 1
            hard_flags.extend(window_hard_flags)
            continue
        if window.timing_correlation is not None and np.isfinite(window.timing_correlation):
            correlations.append(float(window.timing_correlation))

    flags = list(dict.fromkeys(hard_flags))
    median = float(np.median(correlations)) if correlations else None
    if len(correlations) < thresholds.minimum_eligible_timing_windows:
        flags.append(FLAG_TIMING_EVIDENCE_INSUFFICIENT)
    elif median is not None and median < thresholds.timing_correlation_min:
        flags.append(FLAG_TIMING_DISAGREEMENT)

    return SessionReliabilityReport(
        session_id=session_id,
        n_windows=len(windows),
        hard_failure_window_count=hard_failure_windows,
        timing_eligible_window_count=len(correlations),
        timing_correlation_median=median,
        timing_correlation_values=tuple(correlations),
        flags=tuple(flags),
        reliable=not flags,
    )
