from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from blueprint_pipeline.lerobot_export_validation import validate_lerobot_export
from tests.video_codec import require_video_codec_or_skip

FPS = 5


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_fixture_video(path: Path, *, frame_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        path.write_bytes(b"not-a-real-mp4")
        return
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (32, 32)
    )
    if not writer.isOpened():
        path.write_bytes(b"not-a-real-mp4")
        return
    try:
        for index in range(frame_count):
            writer.write(np.full((32, 32, 3), 32 + index, dtype=np.uint8))
    finally:
        writer.release()


def _default_rows(frames_per_episode: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_index = 0
    for episode_index, length in enumerate(frames_per_episode):
        for frame_index in range(length):
            rows.append(
                {
                    "observation.state": [0.0] * 7,
                    "action": [0.01 * frame_index] * 7,
                    "timestamp": frame_index / FPS,
                    "task_index": 0,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "index": global_index,
                    "next.done": frame_index == length - 1,
                }
            )
            global_index += 1
    return rows


def _default_episodes(frames_per_episode: tuple[int, ...]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    start = 0
    for episode_index, length in enumerate(frames_per_episode):
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [0],
                "length": length,
                "start_index": start,
                "end_index": start + length,
                "videos/observation.images.ego_view/chunk_index": 0,
                "videos/observation.images.ego_view/file_index": episode_index,
                "videos/observation.images.ego_view/from_timestamp": 0.0,
                "videos/observation.images.ego_view/to_timestamp": length / FPS,
            }
        )
        start += length
    return episodes


def _lerobot_v3_fixture(
    tmp_path: Path,
    *,
    frames_per_episode: tuple[int, ...] = (3,),
    mutate_rows: Callable[[list[dict[str, Any]]], None] | None = None,
    mutate_episodes: Callable[[list[dict[str, Any]]], None] | None = None,
    info_overrides: dict[str, Any] | None = None,
    write_videos: bool = True,
    video_bytes: bytes | None = None,
) -> Path:
    root = tmp_path / "exports" / "lerobot_v3"
    rows = _default_rows(frames_per_episode)
    if mutate_rows:
        mutate_rows(rows)
    episodes = _default_episodes(frames_per_episode)
    if mutate_episodes:
        mutate_episodes(episodes)
    tasks = [{"task_index": 0, "task": "task-1"}]
    _write_jsonl(root / "data" / "chunk-000" / "file-000.parquet.jsonl", rows)
    _write_jsonl(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet.jsonl", episodes
    )
    _write_jsonl(root / "meta" / "tasks.parquet.jsonl", tasks)
    _write_jsonl(root / "meta" / "tasks.jsonl", tasks)
    info = {
        "schema_version": "lerobot_v3_info.v1",
        "fps": FPS,
        "total_episodes": len(frames_per_episode),
        "total_frames": len(rows),
        "features": {
            "observation.state": {"dtype": "float32", "shape": [7]},
            "action": {"dtype": "float32", "shape": [7]},
            "observation.images.ego_view": {"dtype": "video", "shape": [0, 0, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }
    info.update(info_overrides or {})
    _write_json(root / "meta" / "info.json", info)
    if write_videos:
        for episode_index in range(len(frames_per_episode)):
            video = (
                root
                / "videos"
                / "observation.images.ego_view"
                / "chunk-000"
                / f"file-{episode_index:03d}.mp4"
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            if video_bytes is None:
                _write_fixture_video(
                    video, frame_count=frames_per_episode[episode_index]
                )
            else:
                video.write_bytes(video_bytes)
    return root


def test_missing_export_dir_fails_closed(tmp_path: Path) -> None:
    report = validate_lerobot_export(tmp_path / "does-not-exist")

    assert report["status"] == "blocked"
    assert "export_dir_missing" in report["blockers"]


def test_good_export_passes_with_hermetic_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_find_spec = __import__("importlib").util.find_spec

    def without_lerobot(name: str, *args: object, **kwargs: object) -> object:
        if name == "lerobot":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(
        "blueprint_pipeline.lerobot_export_validation.importlib.util.find_spec",
        without_lerobot,
    )
    root = _lerobot_v3_fixture(tmp_path, frames_per_episode=(3, 2))

    report = validate_lerobot_export(root)

    assert report["status"] == "passed"
    assert report["blockers"] == []
    assert report["layout"] == "lerobot_v3"
    assert "hermetic" in report["loader"]
    assert report["counts"]["episode_count"] == 2
    assert report["counts"]["frame_count"] == 5
    assert report["counts"]["task_count"] == 1
    assert report["checks"]["timestamps_monotonic_per_episode"] == "passed"
    assert report["checks"]["task_index_referential_integrity"] == "passed"
    assert report["checks"]["video_frame_alignment_declared"] == "passed"
    # Passing validation is a loadability claim, never a data-quality claim.
    boundary = report["claim_boundary"]
    assert boundary["validation_passed_is_not_data_quality_or_success_claim"] is True


def test_undecodable_video_file_blocks_when_decoder_available(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    root = _lerobot_v3_fixture(
        tmp_path,
        frames_per_episode=(3,),
        video_bytes=b"not-a-real-mp4",
    )

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "video_file_undecodable:episode_0" in report["blockers"]
    assert report["checks"]["video_frame_alignment_decoded"] == "failed"


def test_duplicate_timestamps_block(tmp_path: Path) -> None:
    def clone_timestamp(rows: list[dict[str, Any]]) -> None:
        rows[2]["timestamp"] = rows[1]["timestamp"]

    root = _lerobot_v3_fixture(
        tmp_path, frames_per_episode=(3,), mutate_rows=clone_timestamp
    )

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "timestamps_not_monotonic:episode_0" in report["blockers"]
    assert report["checks"]["timestamps_monotonic_per_episode"] == "failed"


def test_missing_timestamps_block(tmp_path: Path) -> None:
    def drop_timestamp(rows: list[dict[str, Any]]) -> None:
        del rows[1]["timestamp"]

    root = _lerobot_v3_fixture(
        tmp_path, frames_per_episode=(3,), mutate_rows=drop_timestamp
    )

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "timestamps_missing:episode_0" in report["blockers"]


def test_fps_inconsistent_with_info_blocks(tmp_path: Path) -> None:
    root = _lerobot_v3_fixture(
        tmp_path, frames_per_episode=(3,), info_overrides={"fps": 30}
    )

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "fps_inconsistent:episode_0" in report["blockers"]
    assert report["checks"]["fps_consistent"] == "failed"


def test_dangling_task_index_blocks(tmp_path: Path) -> None:
    def dangle(rows: list[dict[str, Any]]) -> None:
        rows[1]["task_index"] = 7

    root = _lerobot_v3_fixture(tmp_path, frames_per_episode=(3,), mutate_rows=dangle)

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "task_index_dangling:episode_0" in report["blockers"]
    assert report["checks"]["task_index_referential_integrity"] == "failed"


def test_declared_video_frames_must_match_row_count(tmp_path: Path) -> None:
    def stretch_video(episodes: list[dict[str, Any]]) -> None:
        episodes[0]["videos/observation.images.ego_view/to_timestamp"] = 5 / FPS

    root = _lerobot_v3_fixture(
        tmp_path, frames_per_episode=(3,), mutate_episodes=stretch_video
    )

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "video_frame_count_mismatch:episode_0" in report["blockers"]
    assert report["checks"]["video_frame_alignment_declared"] == "failed"


def test_missing_video_file_blocks_when_feature_declared(tmp_path: Path) -> None:
    root = _lerobot_v3_fixture(tmp_path, frames_per_episode=(3,), write_videos=False)

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "video_file_missing:episode_0" in report["blockers"]
    assert report["checks"]["video_files_present"] == "failed"


def test_unstable_action_dim_blocks(tmp_path: Path) -> None:
    def truncate_action(rows: list[dict[str, Any]]) -> None:
        rows[2]["action"] = rows[2]["action"][:6]

    root = _lerobot_v3_fixture(
        tmp_path, frames_per_episode=(3,), mutate_rows=truncate_action
    )

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "action_dim_unstable" in report["blockers"]
    assert report["checks"]["feature_dims_stable"] == "failed"


def test_action_dim_must_match_declared_features(tmp_path: Path) -> None:
    root = _lerobot_v3_fixture(
        tmp_path,
        frames_per_episode=(3,),
        info_overrides={
            "features": {
                "observation.state": {"dtype": "float32", "shape": [7]},
                "action": {"dtype": "float32", "shape": [9]},
                "observation.images.ego_view": {"dtype": "video", "shape": [0, 0, 3]},
            }
        },
    )

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "action_dim_mismatch_with_info" in report["blockers"]


def test_frame_index_gap_blocks(tmp_path: Path) -> None:
    def gap(rows: list[dict[str, Any]]) -> None:
        rows[2]["frame_index"] = 5

    root = _lerobot_v3_fixture(tmp_path, frames_per_episode=(3,), mutate_rows=gap)

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "frame_index_not_sequential:episode_0" in report["blockers"]


def test_episode_length_mismatch_blocks(tmp_path: Path) -> None:
    def lie_about_length(episodes: list[dict[str, Any]]) -> None:
        episodes[0]["length"] = 5

    root = _lerobot_v3_fixture(
        tmp_path, frames_per_episode=(3,), mutate_episodes=lie_about_length
    )

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "episode_length_mismatch:episode_0" in report["blockers"]


def test_empty_dataset_blocks(tmp_path: Path) -> None:
    root = _lerobot_v3_fixture(tmp_path, frames_per_episode=())

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "dataset_empty_no_frames" in report["blockers"]


def test_decoded_video_frame_count_mismatch_blocks(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    root = _lerobot_v3_fixture(tmp_path, frames_per_episode=(3,), write_videos=False)
    video = (
        root / "videos" / "observation.images.ego_view" / "chunk-000" / "file-000.mp4"
    )
    video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (32, 32)
    )
    if not writer.isOpened():
        require_video_codec_or_skip("cv2 mp4 writer unavailable")
    for index in range(5):
        writer.write(np.full((32, 32, 3), 40 * index, dtype=np.uint8))
    writer.release()

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "video_frame_count_mismatch_decoded:episode_0" in report["blockers"]
    assert report["checks"]["video_frame_alignment_decoded"] == "failed"


def test_gr00t_episodes_jsonl_layout_supported(tmp_path: Path) -> None:
    root = tmp_path / "exports" / "gr00t_lerobot"
    rows = _default_rows((3,))
    _write_jsonl(root / "data" / "chunk-000" / "episode_000000.parquet.jsonl", rows)
    episodes = _default_episodes((3,))
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "task-1"}])
    _write_json(
        root / "meta" / "modality.json",
        {
            "video": {
                "ego_view": {"original_key": "observation.images.ego_view"},
            },
        },
    )
    _write_json(
        root / "meta" / "info.json",
        {
            "schema_version": "gr00t_lerobot_info.v1",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [7]},
                "action": {"dtype": "float32", "shape": [7]},
            },
        },
    )
    _write_fixture_video(
        root
        / "videos"
        / "chunk-000"
        / "observation.images.ego_view"
        / "episode_000000.mp4",
        frame_count=3,
    )

    report = validate_lerobot_export(root)

    assert report["status"] == "passed"
    assert report["blockers"] == []
    assert report["layout"] == "gr00t_lerobot"
    assert report["counts"]["frame_count"] == 3
    assert report["checks"]["video_files_present"] == "passed"


def test_gr00t_modality_video_file_missing_blocks(tmp_path: Path) -> None:
    root = tmp_path / "exports" / "gr00t_lerobot"
    rows = _default_rows((3,))
    _write_jsonl(root / "data" / "chunk-000" / "episode_000000.parquet.jsonl", rows)
    episodes = _default_episodes((3,))
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "task-1"}])
    _write_json(
        root / "meta" / "modality.json",
        {
            "video": {
                "ego_view": {"original_key": "observation.images.ego_view"},
            },
        },
    )
    _write_json(
        root / "meta" / "info.json",
        {
            "schema_version": "gr00t_lerobot_info.v1",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [7]},
                "action": {"dtype": "float32", "shape": [7]},
            },
        },
    )

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert "video_file_missing:episode_0" in report["blockers"]
    assert report["checks"]["video_files_present"] == "failed"


def test_native_parquet_without_pyarrow_fails_closed(tmp_path: Path) -> None:
    if __import__("importlib").util.find_spec("pyarrow") is not None:
        pytest.skip("pyarrow installed; hermetic-unreadable branch not reachable")
    root = _lerobot_v3_fixture(tmp_path, frames_per_episode=(3,))
    # Native parquet only, no jsonl mirror: hermetic reader cannot prove anything.
    data_dir = root / "data" / "chunk-000"
    (data_dir / "file-000.parquet.jsonl").unlink()
    (data_dir / "file-000.parquet").write_bytes(b"PAR1-fake")

    report = validate_lerobot_export(root)

    assert report["status"] == "blocked"
    assert any(
        blocker.startswith("parquet_unreadable_missing_pyarrow")
        for blocker in report["blockers"]
    )
