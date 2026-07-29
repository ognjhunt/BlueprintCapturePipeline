from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.robot_skeleton_projection import (
    build_projected_robot_skeleton_trace,
)


def _calibration() -> dict:
    return {
        "intrinsics": {
            "fx": 100.0,
            "fy": 100.0,
            "cx": 50.0,
            "cy": 40.0,
            "width": 100,
            "height": 80,
        },
        "camera_from_world": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "reference_frame": "franka_base",
        "camera_frame": "external_left_optical",
        "translation_unit": "meters",
        "optical_convention": "opencv",
        "reprojection_error_px": 0.4,
    }


def test_builds_camera_aligned_embodiment_neutral_trace(tmp_path: Path) -> None:
    result = build_projected_robot_skeleton_trace(
        landmark_frames=[
            {"base": [0.0, 0.0, 1.0], "elbow": [0.1, 0.0, 1.0], "hand": [0.2, 0.0, 1.0]},
            {"base": [0.0, 0.0, 1.0], "elbow": [0.1, 0.05, 1.0], "hand": [0.2, 0.1, 1.0]},
        ],
        segments=[("base", "elbow"), ("elbow", "hand")],
        camera_calibration=_calibration(),
        embodiment="franka_panda_droid",
        episode_id="fixture-episode",
        output_dir=tmp_path,
    )

    assert result["status"] == "passed"
    assert result["all_frames_have_projected_landmark"] is True
    assert result["provenance"]["physical_future_observation_used"] is False
    rows = [json.loads(line) for line in Path(result["trace_path"]).read_text().splitlines()]
    hand = next(row for row in rows[0]["landmarks"] if row["landmark_id"] == "hand")
    assert hand["image_projection"]["u_px"] == pytest.approx(70.0)
    assert hand["image_projection"]["v_px"] == pytest.approx(40.0)


def test_fails_closed_on_uncalibrated_or_non_opencv_camera(tmp_path: Path) -> None:
    calibration = _calibration()
    calibration.pop("reprojection_error_px")
    with pytest.raises(ValueError, match="reprojection_error_missing"):
        build_projected_robot_skeleton_trace(
            landmark_frames=[{"hand": [0.0, 0.0, 1.0]}],
            segments=[],
            camera_calibration=calibration,
            embodiment="franka_panda_droid",
            episode_id="fixture",
            output_dir=tmp_path,
        )
    calibration = _calibration()
    calibration["optical_convention"] = "unknown"
    with pytest.raises(ValueError, match="optical_convention"):
        build_projected_robot_skeleton_trace(
            landmark_frames=[{"hand": [0.0, 0.0, 1.0]}],
            segments=[],
            camera_calibration=calibration,
            embodiment="franka_panda_droid",
            episode_id="fixture",
            output_dir=tmp_path,
        )


def test_fails_closed_on_landmark_identity_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identity_drift"):
        build_projected_robot_skeleton_trace(
            landmark_frames=[{"hand": [0.0, 0.0, 1.0]}, {"wrist": [0.0, 0.0, 1.0]}],
            segments=[],
            camera_calibration=_calibration(),
            embodiment="franka_panda_droid",
            episode_id="fixture",
            output_dir=tmp_path,
        )
