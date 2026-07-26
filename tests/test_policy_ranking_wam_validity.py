from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from blueprint_pipeline.policy_ranking_wam_validity import (
    analyze_action_motion,
    load_restricted_roboarena_npz,
)


def _npz(path: Path) -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "cartesian_position": [index / 100, 0, 0, 0, 0, 0],
                "joint_position": [0.0] * 7,
                "gripper_position": [0.0],
                "action": [index / 10] * 7 + [0.0],
            }
        )
    np.savez(path, data=np.asarray(rows, dtype=object))


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 32))
    assert writer.isOpened()
    for index in range(12):
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        frame[:, :32] = index * 10
        frame[:, 32:] = (0, 255, 0)
        writer.write(frame)
    writer.release()


def test_restricted_loader_and_action_motion_diagnostic(tmp_path: Path) -> None:
    npz = tmp_path / "actions.npz"
    video = tmp_path / "rollout.mp4"
    _npz(npz)
    _video(video)
    arrays = load_restricted_roboarena_npz(npz)
    assert arrays["action"].shape == (12, 8)
    result = analyze_action_motion(video, npz)
    assert result["action_step_count"] == 12
    assert result["third_party_physical_pixels_decoded_for_metric"] is False
    assert result["security"]["numpy_allow_pickle_used"] is False


def test_restricted_loader_rejects_non_numpy_pickle_global(tmp_path: Path) -> None:
    target = tmp_path / "bad.npz"
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer, {"descr": "|O", "fortran_order": False, "shape": (1,)}
    )
    buffer.write(pickle.dumps(Path("forbidden"), protocol=3))
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("data.npy", buffer.getvalue())
    with pytest.raises((ValueError, pickle.UnpicklingError)):
        load_restricted_roboarena_npz(target)
