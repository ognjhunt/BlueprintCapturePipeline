from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from blueprint_pipeline.ctrl_world_public_initial_observation import (
    build_ctrl_world_public_initial_observation,
)


def _video(path: Path, value: int) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (320, 192)
    )
    assert writer.isOpened()
    writer.write(np.full((192, 320, 3), value, dtype=np.uint8))
    writer.release()


def test_public_initial_observation_excludes_exposed_outcome(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    videos = dataset / "videos/val/899"
    videos.mkdir(parents=True)
    for index in range(3):
        _video(videos / f"{index}.mp4", 20 + index)
    annotation = dataset / "annotation.json"
    annotation.write_text(
        json.dumps(
            {
                "episode_id": 899,
                "texts": ["Move the banana to the right"],
                "success": 0,
                "joints": [[0.0] * 8],
                "states": [[0.0] * 7],
                "videos": [
                    {"video_path": f"videos/val/899/{index}.mp4"}
                    for index in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )
    result = build_ctrl_world_public_initial_observation(
        annotation_path=annotation,
        dataset_root=dataset,
        output_dir=tmp_path / "output",
        ctrl_world_revision="9" * 40,
        expected_trajectory_id="899",
    )
    serialized = json.dumps(result, sort_keys=True)
    assert result["source"]["source_outcome_field_present"] is True
    assert result["source"]["source_outcome_value_recorded"] is False
    assert result["confirmation_eligible"] is False
    assert '"success": 0' not in serialized
    assert len(result["views"]) == 3
    assert all(row["native_shape"] == [192, 320, 3] for row in result["views"].values())
    assert result["state"]["joint_position"]["shape"] == [7]
    assert result["state"]["gripper_position"]["shape"] == [1]
