from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import g1_microwave_grasp_arc_seed as grasp


def test_axis_angle_rotation_is_right_handed_and_orthonormal() -> None:
    rotation = grasp.axis_angle_rotation([0.0, 0.0, 1.0], np.pi / 2.0)

    np.testing.assert_allclose(
        rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_oriented_grasp_basis_aligns_hand_axis_and_approach() -> None:
    basis = grasp.oriented_grasp_basis(
        pelvis_model_xyz_m=[0.0, 0.0, 0.8],
        handle_model_xyz_m=[0.4, 0.0, 1.2],
        hinge_axis_model_xyz=[0.0, 0.0, 1.0],
        hand_axis_polarity=-1.0,
        grasp_yaw_rad=0.0,
    )

    np.testing.assert_allclose(basis[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(basis[:, 2], [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)


def test_head_camera_is_orthonormal_and_task_directed() -> None:
    axes = np.asarray(grasp.EGOCENTRIC_CAMERA_XYAXES).reshape(2, 3)
    camera_x, camera_y = axes
    camera_z = np.cross(camera_x, camera_y)
    view_direction = -camera_z

    np.testing.assert_allclose(np.linalg.norm(camera_x), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(camera_y), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(camera_x, camera_y), 0.0, atol=1e-12)
    # Torso +X is forward, -Y is toward the right-hand microwave target, and
    # negative Z aims down from head height toward the handle.
    assert view_direction[0] > 0.7
    assert view_direction[1] < -0.6
    assert view_direction[2] < -0.2


def test_grasp_arc_seed_rejects_unbound_model_before_loading_inputs(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.xml"
    model.write_text("<mujoco/>", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="g1_microwave_grasp_arc_model_missing_or_sha256_mismatch",
    ):
        grasp.solve_grasp_arc_seed(
            model_path=model,
            standing_initialization_path=tmp_path / "missing-standing.json",
            initial_policy_observation_path=tmp_path / "missing-observation.json",
            target_focus_report_path=tmp_path / "missing-focus.json",
            expected_model_sha256="0" * 64,
        )
