import json
from pathlib import Path

import pytest

from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.pubsub_handoff_listener import (
    HandoffMessage,
    parse_handoff_payload,
    process_handoff_payload,
)


class FakeBlob:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def download_to_filename(self, destination: str) -> None:
        Path(destination).write_bytes(self._data)


class FakeStorageClient:
    def __init__(self, blobs: list[FakeBlob]) -> None:
        self._blobs = blobs

    def bucket(self, _name: str):
        return object()

    def list_blobs(self, _bucket: str, prefix: str):
        return [blob for blob in self._blobs if blob.name.startswith(prefix)]


def test_parse_handoff_payload_requires_identity_consistency() -> None:
    payload = {
        "bucket": "capture-bucket",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        "pipeline_handoff_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/pipeline_handoff.json",
    }

    handoff = parse_handoff_payload(json.dumps(payload).encode("utf-8"))

    assert handoff == HandoffMessage(
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        raw_prefix_uri="gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        pipeline_handoff_uri="gs://capture-bucket/scenes/scene-1/captures/capture-1/pipeline_handoff.json",
    )


def test_parse_handoff_payload_blocks_mismatched_raw_prefix() -> None:
    with pytest.raises(PipelineError, match="raw_prefix_uri does not match"):
        parse_handoff_payload(
            {
                "bucket": "capture-bucket",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "raw_prefix_uri": "gs://capture-bucket/scenes/other/captures/capture-1/raw",
            }
        )


def test_process_handoff_stages_capture_and_runs_e2e(tmp_path: Path) -> None:
    prefix = "scenes/scene-1/captures/capture-1"
    client = FakeStorageClient(
        [
            FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
            FakeBlob(f"{prefix}/raw/manifest.json", b"{}"),
            FakeBlob(f"{prefix}/pipeline_handoff.json", b"{}"),
            FakeBlob(f"{prefix}/capture_descriptor.json", b"{}"),
        ]
    )
    calls = []

    def fake_run_e2e(**kwargs):
        calls.append(kwargs)
        return {"status": "ok"}

    result = process_handoff_payload(
        {
            "bucket": "capture-bucket",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        },
        storage_root=tmp_path,
        provider="openai",
        run_e2e=fake_run_e2e,
        storage_client=client,  # type: ignore[arg-type]
    )

    capture_root = tmp_path / "capture-bucket" / prefix
    assert result["status"] == "processed"
    assert calls == [
        {
            "capture_root": str(capture_root),
            "provider": "openai",
            "run_evaluation_prep": True,
        }
    ]
    assert (capture_root / "raw" / "capture_upload_complete.json").is_file()
    assert (capture_root / "pipeline_handoff.json").is_file()
