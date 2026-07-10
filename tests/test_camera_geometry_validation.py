from __future__ import annotations

import pytest

from blueprint_pipeline.camera_geometry_validation import (
    validate_camera_calibration,
    validate_camera_intrinsics,
    validate_se3_matrix,
)


def _calibration() -> dict[str, object]:
    return {
        "intrinsics": {
            "fx": 500.0,
            "fy": 505.0,
            "cx": 320.0,
            "cy": 240.0,
            "width": 640,
            "height": 480,
        },
        "camera_from_world": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "reference_frame": "world",
        "camera_frame": "head_camera",
        "translation_unit": "meters",
        "reprojection_error_px": 0.7,
    }


def _strict(payload: dict[str, object], **kwargs: object) -> dict[str, object]:
    return validate_camera_calibration(
        payload,
        require_extrinsics=True,
        require_frame_metadata=True,
        require_translation_units=True,
        require_reprojection_error=True,
        **kwargs,
    )


def test_calibrated_camera_golden_fixture_is_projection_ready() -> None:
    result = _strict(
        _calibration(),
        expected_reference_frame="world",
        expected_camera_frame="head_camera",
    )

    assert result["status"] == "passed"
    assert result["projection_ready"] is True
    assert result["intrinsics"] == {
        "fx": 500.0,
        "fy": 505.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480,
    }
    assert result["camera_from_reference"] == _calibration()["camera_from_world"]


@pytest.mark.parametrize(
    ("mutation", "expected_blocker"),
    [
        (lambda payload: payload.pop("camera_from_world"), "camera_extrinsics_missing"),
        (
            lambda payload: payload["intrinsics"].update({"fx": 1e-4}),
            "camera_horizontal_fov_implausible",
        ),
        (
            lambda payload: payload["intrinsics"].update({"cx": 900.0}),
            "camera_principal_point_outside_image_x",
        ),
        (
            lambda payload: payload["camera_from_world"].__setitem__(0, [-1.0, 0.0, 0.0, 0.0]),
            "camera_from_reference_rotation_not_right_handed",
        ),
        (
            lambda payload: payload["camera_from_world"].__setitem__(0, [1.0, 0.2, 0.0, 0.0]),
            "camera_from_reference_rotation_not_orthonormal",
        ),
        (lambda payload: payload.pop("reprojection_error_px"), "camera_reprojection_error_missing"),
        (
            lambda payload: payload.update({"reference_frame": "camera"}),
            "camera_reference_frame_mismatch",
        ),
        (
            lambda payload: payload.update({"translation_unit": "millimeters"}),
            "camera_extrinsics_translation_unit_missing_or_not_meters",
        ),
    ],
)
def test_projection_ready_camera_rejects_bad_calibration(
    mutation,
    expected_blocker: str,
) -> None:
    payload = _calibration()
    mutation(payload)
    result = _strict(payload, expected_reference_frame="world")

    assert result["projection_ready"] is False
    assert expected_blocker in result["blockers"]


def test_se3_and_intrinsics_reject_nan_and_invalid_last_row() -> None:
    pose = _calibration()["camera_from_world"]
    pose[0][0] = float("nan")
    assert validate_se3_matrix(pose, field="pose")["valid"] is False

    bad_last_row = _calibration()["camera_from_world"]
    bad_last_row[3] = [0.0, 0.0, 1.0, 1.0]
    assert "pose_invalid_homogeneous_last_row" in validate_se3_matrix(
        bad_last_row,
        field="pose",
    )["blockers"]
    assert validate_camera_intrinsics(
        {"fx": float("nan"), "fy": 500, "cx": 1, "cy": 1, "width": 10, "height": 10}
    )["valid"] is False
