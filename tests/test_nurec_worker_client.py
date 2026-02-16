"""Tests for NuRec worker client dispatch behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline.nurec_worker_client import NurecWorkerClient, NurecWorkerConfig


def _descriptor() -> CaptureDescriptor:
    return CaptureDescriptor.from_dict(
        {
            "schema_version": "v1",
            "scene_id": "scene_1",
            "capture_id": "cap_1",
            "capture_source": "iphone",
            "capture_tier": "tier1_iphone",
            "raw_prefix_uri": "gs://bucket/scenes/scene_1/iphone/cap_1/raw",
            "frames_index_uri": "gs://bucket/scenes/scene_1/captures/cap_1/frames/index.jsonl",
            "nurec_mode": "mono_pose_assisted",
        }
    )


def test_local_worker_dispatch_includes_repo_pythonpath(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(command, **kwargs):  # noqa: ANN001
        captured["command"] = command
        captured["env"] = kwargs.get("env", {})
        return _Proc()

    monkeypatch.setattr("blueprint_pipeline.nurec_worker_client.subprocess.run", _fake_run)

    client = NurecWorkerClient(
        storage_root=tmp_path,
        bucket="bucket",
        pipeline_prefix="scenes/scene_1/captures/cap_1/pipeline",
        config=NurecWorkerConfig(worker_mode="local_worker"),
    )
    spec_path = client.pipeline_dir / "nurec_job_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("{}", encoding="utf-8")

    client.dispatch(spec_path=spec_path)

    env = captured["env"]
    assert "PYTHONPATH" in env
    assert str(client._repo_src) in env["PYTHONPATH"]


def test_run_clears_stale_markers_before_dispatch(monkeypatch, tmp_path: Path) -> None:
    client = NurecWorkerClient(
        storage_root=tmp_path,
        bucket="bucket",
        pipeline_prefix="scenes/scene_1/captures/cap_1/pipeline",
        config=NurecWorkerConfig(worker_mode="external_markers"),
    )

    complete_marker = client.pipeline_dir / ".nurec_complete"
    failed_marker = client.pipeline_dir / ".nurec_failed"
    complete_marker.parent.mkdir(parents=True, exist_ok=True)
    complete_marker.write_text("stale", encoding="utf-8")
    failed_marker.write_text("stale", encoding="utf-8")

    def _fake_wait() -> None:
        (client.pipeline_dir / ".nurec_complete").write_text("fresh", encoding="utf-8")

    monkeypatch.setattr(client, "wait_for_completion", _fake_wait)
    monkeypatch.setattr(client, "collect_outputs", lambda: {"status": "completed"})

    out = client.run(
        descriptor=_descriptor(),
        descriptor_uri="gs://bucket/scenes/scene_1/captures/cap_1/capture_descriptor.json",
        object_index_uri="gs://bucket/scenes/scene_1/iphone/cap_1/raw/arkit/objects/index.json",
    )

    assert out["status"] == "completed"
    assert complete_marker.read_text(encoding="utf-8") == "fresh"
    assert not failed_marker.exists()
