"""Contract tests for the LeRobot v2.1 export of ADP-009D episodes.

LeRobot's dataset layout is the interchange robotics teams actually load --
openpi fine-tunes from it and GR00T consumes the same schema -- so the export
is the difference between "here is our receipt JSON" and "here is a dataset
your tooling already reads".  These tests pin the tree, the parquet columns,
and the fail-closed refusal when a receipt predates step-trace retention.
"""

from __future__ import annotations

import json

import pytest

from blueprint_pipeline.adp009d_dataset_capture import DatasetCaptureRecorder
from blueprint_pipeline.adp009d_droid_observation import (
    DROID_EXTERIOR_VIEW_1,
    DROID_WRIST_VIEW,
)
from blueprint_pipeline.adp009d_lerobot_export import (
    LEROBOT_CODEBASE_VERSION,
    LeRobotExportError,
    export_lerobot_dataset,
)
from tests.test_adp009d_policy_episode import _run

pyarrow = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _episode_receipt(tmp_path, index: int) -> dict:
    episode_id = f"pi05_droid-episode-{index:03d}"
    recorder = DatasetCaptureRecorder(
        output_dir=tmp_path / "media_root",
        episode_id=episode_id,
        view_keys=(DROID_EXTERIOR_VIEW_1, DROID_WRIST_VIEW),
    )
    return _run(
        media_output_dir=tmp_path / "media_root",
        episode_id=episode_id,
        dataset_capture=recorder,
    )


def test_export_writes_a_loadable_lerobot_v21_tree(tmp_path) -> None:
    receipts = [_episode_receipt(tmp_path, 0), _episode_receipt(tmp_path, 1)]
    dataset_dir = tmp_path / "lerobot"

    report = export_lerobot_dataset(
        episode_receipts=receipts,
        output_dir=dataset_dir,
        media_root=tmp_path / "media_root",
    )

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    assert info["codebase_version"] == LEROBOT_CODEBASE_VERSION == "v2.1"
    assert info["fps"] == 15
    assert info["robot_type"] == "franka_panda"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == sum(r["environment_steps"] for r in receipts)
    assert info["features"]["observation.state"]["shape"] == [8]
    assert info["features"]["action"]["shape"] == [8]
    assert info["features"]["observation.state"]["names"][-1] == "gripper_width_m"
    video_key = "observation.images.exterior_image_1_left"
    assert info["features"][video_key]["dtype"] == "video"
    assert info["features"][video_key]["info"]["video.fps"] == 15.0
    assert info["total_videos"] == 4

    tasks = [
        json.loads(line)
        for line in (dataset_dir / "meta" / "tasks.jsonl").read_text().splitlines()
    ]
    assert len(tasks) == 1
    assert tasks[0]["task"] == receipts[0]["prompt"]

    episodes = [
        json.loads(line)
        for line in (dataset_dir / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert [row["length"] for row in episodes] == [
        r["environment_steps"] for r in receipts
    ]

    stats_lines = (dataset_dir / "meta" / "episodes_stats.jsonl").read_text().splitlines()
    assert len(stats_lines) == 2
    first_stats = json.loads(stats_lines[0])["stats"]
    assert len(first_stats["observation.state"]["mean"]) == 8

    table = pq.read_table(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet")
    assert table.num_rows == receipts[0]["environment_steps"]
    state = table.column("observation.state").to_pylist()
    action = table.column("action").to_pylist()
    timestamp = table.column("timestamp").to_pylist()
    assert len(state[0]) == 8
    assert action[0][0] == pytest.approx(0.25)
    assert timestamp[0] == 0.0
    assert timestamp[1] == pytest.approx(1 / 15)
    assert table.column("episode_index").to_pylist()[0] == 0
    assert table.column("next.done").to_pylist()[-1] is True

    second = pq.read_table(dataset_dir / "data" / "chunk-000" / "episode_000001.parquet")
    global_index = table.column("index").to_pylist() + second.column("index").to_pylist()
    assert global_index == list(range(info["total_frames"]))

    for camera in ("exterior_image_1_left", "wrist_image_left"):
        video = (
            dataset_dir
            / "videos"
            / "chunk-000"
            / f"observation.images.{camera}"
            / "episode_000000.mp4"
        )
        assert video.is_file()
        assert video.read_bytes()[4:8] == b"ftyp"

    assert report["episodes_exported"] == 2
    assert report["video_streams_exported"] == 4


def test_export_refuses_receipts_without_step_trace(tmp_path) -> None:
    receipt = _episode_receipt(tmp_path, 0)
    stripped = {k: v for k, v in receipt.items() if k != "step_trace"}

    with pytest.raises(LeRobotExportError, match="step_trace_missing"):
        export_lerobot_dataset(
            episode_receipts=[stripped],
            output_dir=tmp_path / "lerobot",
            media_root=tmp_path / "media_root",
        )


def test_export_without_capture_emits_state_action_only(tmp_path) -> None:
    receipt = _run()

    report = export_lerobot_dataset(
        episode_receipts=[receipt],
        output_dir=tmp_path / "lerobot",
        media_root=None,
    )

    info = json.loads((tmp_path / "lerobot" / "meta" / "info.json").read_text())
    assert info["total_videos"] == 0
    assert not any(key.startswith("observation.images.") for key in info["features"])
    assert report["video_streams_exported"] == 0
    table = pq.read_table(
        tmp_path / "lerobot" / "data" / "chunk-000" / "episode_000000.parquet"
    )
    assert table.num_rows == receipt["environment_steps"]


def test_export_refuses_mixed_capture_presence(tmp_path) -> None:
    with_capture = _episode_receipt(tmp_path, 0)
    without_capture = _run()

    with pytest.raises(LeRobotExportError, match="capture_inconsistent"):
        export_lerobot_dataset(
            episode_receipts=[with_capture, without_capture],
            output_dir=tmp_path / "lerobot",
            media_root=tmp_path / "media_root",
        )
