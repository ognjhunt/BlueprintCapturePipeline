from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.policy_ranking_cosmos_causal_screen import (
    _correlation,
    action_intensity,
    camera_compensated_motion,
)


def test_action_intensity_rejects_wrong_shape_and_preserves_zero_control() -> None:
    assert np.array_equal(action_intensity([[0.0] * 10] * 16), np.zeros(16))
    with pytest.raises(ValueError, match="action_shape_invalid"):
        action_intensity([[0.0] * 10] * 15)


def test_action_intensity_sees_translation_rotation_and_gripper_changes() -> None:
    actions = np.zeros((16, 10), dtype=np.float64)
    actions[:, 3] = 1.0
    actions[:, 7] = 1.0
    actions[2, 0] = 0.25
    actions[7, 4] = 0.2
    actions[11, 9] = 1.0

    signal = action_intensity(actions.tolist())

    assert signal[2] > signal[1]
    assert signal[7] > signal[6]
    assert signal[11] > signal[10]


def test_correlation_fails_closed_for_constant_signals() -> None:
    assert _correlation(np.ones(16), np.arange(16)) == 0.0


def test_camera_compensated_motion_rejects_global_translation() -> None:
    base = np.zeros((136, 160), dtype=np.uint8)
    base[35:100, 45:120] = 180
    frames = []
    for offset in range(17):
        matrix = np.float32([[1, 0, offset], [0, 1, 0]])
        frames.append(__import__("cv2").warpAffine(base, matrix, (160, 136), borderMode=1))

    residual = camera_compensated_motion(np.asarray(frames))

    assert residual.shape == (16,)
    assert float(np.mean(residual)) < 0.25
