"""Label-free collapse diagnostics over audited generated-only Phase-A frames."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np

from .common import write_json
from .policy_ranking_roboarena_calibration import canonical_sha256, file_sha256


STATIC_MEAN_CONSECUTIVE_DIFFERENCE_MAX = 0.003
LOOP_MATCH_DIFFERENCE_MAX = 0.003
LOOP_MATCH_FRACTION_MIN = 0.80
DISCONTINUITY_DIFFERENCE_MIN = 0.15
DISCONTINUITY_MEDIAN_MULTIPLIER = 6.0


def _load_frame(path: Path, expected_sha256: str) -> np.ndarray:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError("audited_frame_missing_or_changed")
    frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise ValueError("audited_frame_decode_failed")
    return cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def _episode_metrics(row: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    frames = [
        _load_frame(output_root / str(frame["relative_output_path"]), str(frame["encoded_jpeg_sha256"]))
        for frame in row.get("sampled_frames") or []
    ]
    if len(frames) != 32:
        raise ValueError("collapse_audit_requires_32_frames")
    differences = [float(np.mean(np.abs(frames[i] - frames[i - 1]))) for i in range(1, 32)]
    mean_difference = float(np.mean(differences))
    median_difference = float(np.median(differences))
    max_difference = max(differences)
    static = mean_difference <= STATIC_MEAN_CONSECUTIVE_DIFFERENCE_MAX
    first_future_collapse = bool(
        differences[0] >= DISCONTINUITY_DIFFERENCE_MIN
        and float(np.mean(differences[1:])) <= STATIC_MEAN_CONSECUTIVE_DIFFERENCE_MAX
    )
    loop_period: int | None = None
    loop_fraction = 0.0
    if not static:
        for period in range(2, 5):
            matches = [
                float(np.mean(np.abs(frames[index] - frames[index - period])))
                <= LOOP_MATCH_DIFFERENCE_MAX
                for index in range(period, len(frames))
            ]
            fraction = float(np.mean(matches)) if matches else 0.0
            if fraction > loop_fraction:
                loop_fraction = fraction
                loop_period = period
        if loop_fraction < LOOP_MATCH_FRACTION_MIN:
            loop_period = None
    discontinuity = bool(
        max_difference >= DISCONTINUITY_DIFFERENCE_MIN
        and max_difference
        >= DISCONTINUITY_MEDIAN_MULTIPLIER * median_difference + STATIC_MEAN_CONSECUTIVE_DIFFERENCE_MAX
    )
    flags: list[str] = []
    if static:
        flags.append("static_or_frozen_future")
    if first_future_collapse:
        flags.append("first_future_frame_collapse")
    if loop_period is not None:
        flags.append("repeated_frame_loop")
    if discontinuity:
        flags.append("sudden_visual_discontinuity")
    return {
        "request_id": row["request_id"],
        "session_id": row["session_id"],
        "policy_id_internal_only": row["policy_id_internal_only"],
        "short_episode_source": row["short_episode_source"],
        "mean_consecutive_grayscale_difference": mean_difference,
        "median_consecutive_grayscale_difference": median_difference,
        "max_consecutive_grayscale_difference": max_difference,
        "best_loop_period": loop_period,
        "best_loop_match_fraction": loop_fraction,
        "deterministic_collapse_flags": flags,
        "safety_abstention_recommended": bool(flags),
        "retained_in_dataset": True,
    }


def build_collapse_report(crop_manifest: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if crop_manifest.get("status") != "ready_for_manual_visual_review":
        blockers.append("crop_manifest_not_complete")
    root = Path(str(crop_manifest.get("output_root") or ""))
    rows: list[dict[str, Any]] = []
    if not blockers:
        for source in crop_manifest.get("requests") or []:
            try:
                rows.append(_episode_metrics(source, root))
            except Exception as exc:
                blockers.append(
                    f"collapse_metric_failed:{source.get('request_id')}:{type(exc).__name__}:{exc}"
                )
    counts = Counter(flag for row in rows for flag in row["deterministic_collapse_flags"])
    result: dict[str, Any] = {
        "schema_version": "policy_ranking_roboarena_collapse_audit.v1",
        "status": "completed" if len(rows) == 441 and not blockers else "blocked",
        "crop_audit_sha256": crop_manifest.get("audit_sha256"),
        "thresholds": {
            "static_mean_consecutive_difference_max": STATIC_MEAN_CONSECUTIVE_DIFFERENCE_MAX,
            "loop_match_difference_max": LOOP_MATCH_DIFFERENCE_MAX,
            "loop_match_fraction_min": LOOP_MATCH_FRACTION_MIN,
            "discontinuity_difference_min": DISCONTINUITY_DIFFERENCE_MIN,
            "discontinuity_median_multiplier": DISCONTINUITY_MEDIAN_MULTIPLIER,
        },
        "episode_count": len(rows),
        "flagged_episode_count": sum(bool(row["deterministic_collapse_flags"]) for row in rows),
        "flag_counts": dict(sorted(counts.items())),
        "episodes": rows,
        "evaluator_required_flags": [
            "robot_skeleton_divergence",
            "object_disappearance",
            "scene_corruption",
            "robot_out_of_view",
            "uncertainty_increases_with_depth",
            "action_following_degrades_with_depth",
        ],
        "retention_rule": "All flagged episodes remain in the matrix and count against reliability; flags recommend safety abstention rather than row removal.",
        "blockers": blockers,
        "provider_called": False,
        "outcome_labels_accessed": False,
    }
    result["report_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    crop = json.loads(Path(args.crop_manifest).read_text(encoding="utf-8"))
    result = build_collapse_report(crop)
    write_json(Path(args.output), result)
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
