from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from blueprint_pipeline.policy_ranking_causal_conditioning import (
    _controls,
    build_causal_report,
    generated_motion_channels,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256


def _npz(path: Path, scale: float) -> None:
    rows = []
    for index in range(30):
        magnitude = scale * (0.05 + (index % 7) / 10)
        rows.append(
            {
                "cartesian_position": [index / 100, 0, 0, 0, 0, 0],
                "joint_position": [0.0] * 7,
                "gripper_position": [0.0],
                "action": [magnitude] + [0.0] * 7,
            }
        )
    np.savez(path, data=np.asarray(rows, dtype=object))


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 32))
    assert writer.isOpened()
    value = 0
    for index in range(30):
        value = min(250, value + 1 + index % 7)
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        frame[:, :32] = value
        frame[4:8, 4:28] = (0, 255, 255)
        frame[:, 32:] = (0, 255, 0)
        writer.write(frame)
    writer.release()


def test_motion_channels_never_decode_physical_half(tmp_path: Path) -> None:
    video = tmp_path / "paired.mp4"
    _video(video)
    channels, metadata = generated_motion_channels(video)
    assert set(channels) == {"full_generated", "overlay_region", "overlay_masked_residual"}
    assert metadata["third_party_physical_pixels_decoded"] is False
    assert metadata["generated_crop_pixels"] == [0, 0, 32, 32]


def test_controls_are_deterministic_and_include_required_placebos() -> None:
    action = np.arange(20, dtype=np.float64)
    first = _controls(action, seed_material="fixed", swapped=action[::-1])
    second = _controls(action, seed_material="fixed", swapped=action[::-1])
    assert set(first) == {
        "zero_actions",
        "shuffled_action_order",
        "temporally_reversed_actions",
        "circularly_shifted_actions",
        "within_session_swapped_policy_actions",
    }
    assert all(np.array_equal(first[key], second[key]) for key in first)


def test_causal_report_is_label_blind_and_clustered(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout"
    roboarena = tmp_path / "roboarena"
    requests = []
    for session in ("s1", "s2"):
        for policy_index, policy in enumerate(("p1", "p2"), start=1):
            video = rollout / session / policy / "left" / "compare_overlay_vs_gt.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            _video(video)
            npz = (
                roboarena
                / "evaluation_sessions"
                / session
                / f"A_{policy}"
                / f"{policy}_npz_file.npz"
            )
            npz.parent.mkdir(parents=True, exist_ok=True)
            _npz(npz, float(policy_index))
            requests.append(
                {
                    "request_id": f"{session}-{policy}",
                    "session_id": session,
                    "policy_id": policy,
                    "video_path": str(video),
                    "video_sha256": file_sha256(video),
                    "method": "temporal",
                }
            )
    report = build_causal_report(
        {"inventory_sha256": "a" * 64, "requests": requests},
        roboarena_root=roboarena,
        bootstrap_replicates=100,
    )
    assert report["status"] == "completed"
    assert report["row_count"] == 4
    assert report["benchmark_labels_seen"] is False
    assert report["third_party_physical_video_pixels_decoded"] is False
    assert report["clustered_summaries"]["overlay_masked_residual"]["session_cluster_count"] == 2
