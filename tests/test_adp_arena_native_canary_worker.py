from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.adp_arena_native_canary_worker import (
    ARENA_REVISION,
    ARENA_UV_LOCK_SHA256,
    GROOT_REVISION,
    ISAAC_LAB_REVISION,
    OPENPI_REVISION,
    _artifact_manifest,
    _episode_rows,
)


def test_native_worker_source_identities_are_exact() -> None:
    for revision in (ARENA_REVISION, ISAAC_LAB_REVISION, GROOT_REVISION, OPENPI_REVISION):
        assert len(revision) == 40
        int(revision, 16)
    assert len(ARENA_UV_LOCK_SHA256) == 64
    int(ARENA_UV_LOCK_SHA256, 16)


def test_native_worker_reads_episode_truth_and_hashes_artifacts(tmp_path: Path) -> None:
    result_dir = tmp_path / "zero_action_control" / "run"
    result_dir.mkdir(parents=True)
    episode = {"success": False, "env_id": 0, "episode_in_env": 0}
    (result_dir / "episode_results_rank0.jsonl").write_text(
        json.dumps(episode) + "\n", encoding="utf-8"
    )
    (result_dir / "camera.mp4").write_bytes(b"review-video")

    assert _episode_rows(tmp_path) == [episode]
    artifacts = _artifact_manifest(tmp_path)
    assert {row["path"] for row in artifacts} == {
        "zero_action_control/run/camera.mp4",
        "zero_action_control/run/episode_results_rank0.jsonl",
    }
    assert all(row["sha256"].startswith("sha256:") for row in artifacts)
