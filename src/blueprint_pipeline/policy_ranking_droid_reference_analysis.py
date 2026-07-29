"""Reusable, label-free analysis for paired Cosmos3 DROID reference clips.

The native Cosmos DROID interface encodes wrist, left-shoulder, and
right-shoulder views into one ``concat_view`` frame.  Whole-frame motion can
therefore hide a frozen external view or let one noisy view dominate.  This
module keeps the views attributable and compares a recorded-action rollout
against its matched valid no-motion counterfactual.

The report is diagnostic unless its numerical thresholds and independent
session count were frozen prospectively.  It never turns decodability or
motion alone into WAM qualification credit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np

from .policy_ranking_thesis import canonical_sha256
from .common import write_json
from .wam_rollout_reliability import (
    TIMING_SCOPE_SESSION,
    ReliabilityThresholds,
    SessionReliabilityThresholds,
    assess_rollout_reliability,
    assess_session_reliability,
)


def _decode_concat_view(video_path: Path) -> dict[str, list[np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) < 2:
        raise ValueError(f"droid_reference_video_too_short:{video_path}")
    height, width = frames[0].shape[:2]
    if width % 2 or height % 3:
        raise ValueError(f"droid_reference_concat_geometry_invalid:{width}x{height}")
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise ValueError("droid_reference_video_geometry_changed")
    wrist_height = 2 * height // 3
    half_width = width // 2
    return {
        "wrist": [frame[:wrist_height, :] for frame in frames],
        "left": [frame[wrist_height:, :half_width] for frame in frames],
        "right": [frame[wrist_height:, half_width:] for frame in frames],
    }


def _residual_flow_metrics(frames: list[np.ndarray]) -> dict[str, float | int]:
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    series: list[float] = []
    for before, after in zip(gray[:-1], gray[1:], strict=True):
        flow = cv2.calcOpticalFlowFarneback(
            before,
            after,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        median = np.median(flow.reshape(-1, 2), axis=0)
        residual = flow - median[None, None, :]
        series.append(float(np.mean(np.linalg.norm(residual, axis=2))))
    values = np.asarray(series, dtype=np.float64)
    return {
        "frame_count": len(frames),
        "residual_flow_mean": float(values.mean()),
        "residual_flow_max": float(values.max()),
        "residual_flow_std": float(values.std()),
    }


def analyze_droid_reference_pair(
    *,
    recorded_video: str | Path,
    no_motion_video: str | Path,
    recorded_actions: np.ndarray,
    no_motion_actions: np.ndarray,
    reliability_thresholds: ReliabilityThresholds,
    session_thresholds: SessionReliabilityThresholds,
    session_id: str,
) -> dict[str, Any]:
    """Analyze a matched active/null pair without granting causal credit.

    Thresholds are required arguments so future scientific callers cannot
    silently inherit development defaults.  A single window normally causes
    session timing to abstain; that is the intended fail-closed behavior.
    """

    recorded_path = Path(recorded_video).expanduser().resolve()
    no_motion_path = Path(no_motion_video).expanduser().resolve()
    recorded_views = _decode_concat_view(recorded_path)
    no_motion_views = _decode_concat_view(no_motion_path)
    if {name: len(value) for name, value in recorded_views.items()} != {
        name: len(value) for name, value in no_motion_views.items()
    }:
        raise ValueError("droid_reference_pair_frame_count_mismatch")

    view_comparison: dict[str, Any] = {}
    for view_name in ("wrist", "left", "right"):
        active = _residual_flow_metrics(recorded_views[view_name])
        null = _residual_flow_metrics(no_motion_views[view_name])
        active_mean = float(active["residual_flow_mean"])
        null_mean = float(null["residual_flow_mean"])
        view_comparison[view_name] = {
            "recorded": active,
            "no_motion": null,
            "recorded_minus_no_motion_mean": active_mean - null_mean,
            "recorded_exceeds_no_motion": active_mean > null_mean,
        }

    recorded_report = assess_rollout_reliability(
        recorded_path,
        recorded_actions,
        reliability_thresholds,
        timing_flag_scope=TIMING_SCOPE_SESSION,
    )
    no_motion_report = assess_rollout_reliability(
        no_motion_path,
        no_motion_actions,
        reliability_thresholds,
        timing_flag_scope=TIMING_SCOPE_SESSION,
    )
    session_report = assess_session_reliability(
        session_id,
        [recorded_report],
        session_thresholds,
    )
    active_view_count = sum(
        int(bool(report["recorded_exceeds_no_motion"])) for report in view_comparison.values()
    )
    reasons = list(session_report.flags)
    if active_view_count != len(view_comparison):
        reasons.append("recorded_did_not_exceed_no_motion_in_every_view")
    payload: dict[str, Any] = {
        "schema_version": "policy_ranking_cosmos3_droid_reference_pair_analysis.v1",
        "session_id": session_id,
        "recorded_rollout_reliability": recorded_report.as_dict(),
        "no_motion_rollout_reliability": no_motion_report.as_dict(),
        "session_reliability": session_report.as_dict(),
        "view_comparison": view_comparison,
        "recorded_exceeds_no_motion_view_count": active_view_count,
        "view_count": len(view_comparison),
        "abstain": bool(reasons),
        "abstention_reasons": reasons,
        "cosmos_wam_qualification_credit": False,
        "thresholds": {
            "rollout": asdict(reliability_thresholds),
            "session": asdict(session_thresholds),
        },
        "claim_boundary": (
            "Paired label-free diagnostic only. Qualification requires prospectively frozen "
            "thresholds and the registered independent-session count; decodability or motion "
            "alone is insufficient."
        ),
    }
    payload["analysis_sha256"] = canonical_sha256(payload)
    return payload


__all__ = ["analyze_droid_reference_pair"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-video", required=True)
    parser.add_argument("--no-motion-video", required=True)
    parser.add_argument("--action-streams", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--diagnostic-development-defaults",
        action="store_true",
        help="Explicitly acknowledge that development thresholds cannot earn qualification credit.",
    )
    args = parser.parse_args()
    if not args.diagnostic_development_defaults:
        parser.error(
            "--diagnostic-development-defaults is required until a threshold lock is supplied"
        )
    action_payload = json.loads(Path(args.action_streams).read_text(encoding="utf-8"))
    report = analyze_droid_reference_pair(
        recorded_video=args.recorded_video,
        no_motion_video=args.no_motion_video,
        recorded_actions=np.asarray(action_payload["recorded"]["actions"], dtype=np.float64),
        no_motion_actions=np.asarray(action_payload["no_motion"]["actions"], dtype=np.float64),
        reliability_thresholds=ReliabilityThresholds(),
        session_thresholds=SessionReliabilityThresholds(),
        session_id=args.session_id,
    )
    report["threshold_source"] = "diagnostic_development_defaults_not_prospectively_frozen"
    report["analysis_sha256"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "analysis_sha256"}
    )
    write_json(Path(args.output), report)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through focused subprocess use
    raise SystemExit(main())
