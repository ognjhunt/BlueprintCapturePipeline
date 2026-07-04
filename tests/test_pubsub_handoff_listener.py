import json
from pathlib import Path

import pytest

from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.pubsub_handoff_listener import (
    HandoffMessage,
    parse_handoff_payload,
    process_handoff_payload,
    stage_handoff_capture,
)


# Real iOS raw bundle namelist per CaptureRawContractV3Validator (no pipeline_handoff.json).
_IOS_MANIFEST = {
    "scene_id": "scene-1",
    "capture_id": "capture-1",
    "site_submission_id": "site-submission-scene-1",
    "buyer_request_id": "req-scene-1",
    "capture_job_id": "capture-job-scene-1",
    "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
}
_IOS_CONTEXT = {
    "scene_id": "scene-1",
    "capture_id": "capture-1",
    "site_submission_id": "site-submission-scene-1",
    "buyer_request_id": "req-scene-1",
    "capture_job_id": "capture-job-scene-1",
}


def _ios_bundle_blobs(prefix: str) -> "list[FakeBlob]":
    return [
        FakeBlob(f"{prefix}/raw/manifest.json", json.dumps(_IOS_MANIFEST).encode("utf-8")),
        FakeBlob(f"{prefix}/raw/capture_context.json", json.dumps(_IOS_CONTEXT).encode("utf-8")),
        FakeBlob(f"{prefix}/raw/hashes.json", b"{}"),
        FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
        FakeBlob(f"{prefix}/raw/arkit/frames.jsonl", b"{}\n"),
        FakeBlob(f"{prefix}/raw/walkthrough.mov", b"\x00\x00"),
    ]


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


def test_redelivered_completed_handoff_is_idempotent(tmp_path: Path) -> None:
    prefix = "scenes/scene-1/captures/capture-1"
    client = FakeStorageClient(
        [
            FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
            FakeBlob(f"{prefix}/pipeline_handoff.json", b"{}"),
        ]
    )
    payload = {
        "bucket": "capture-bucket",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
    }
    calls: list[dict] = []

    def fake_run_e2e(**kwargs):
        calls.append(kwargs)
        return {"status": "ok"}

    first = process_handoff_payload(
        payload,
        storage_root=tmp_path,
        provider="openai",
        run_e2e=fake_run_e2e,
        storage_client=client,  # type: ignore[arg-type]
    )
    second = process_handoff_payload(
        payload,
        storage_root=tmp_path,
        provider="openai",
        run_e2e=fake_run_e2e,
        storage_client=client,  # type: ignore[arg-type]
    )

    assert first["status"] == "processed"
    assert second["status"] == "skipped_already_processed"
    assert len(calls) == 1
    capture_root = tmp_path / "capture-bucket" / prefix
    ledger = json.loads(
        (capture_root / "pipeline_job_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "completed"
    assert ledger["attempt_count"] == 1


def test_crashed_processing_run_is_retried_not_skipped(tmp_path: Path) -> None:
    prefix = "scenes/scene-1/captures/capture-1"
    client = FakeStorageClient(
        [
            FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
            FakeBlob(f"{prefix}/pipeline_handoff.json", b"{}"),
        ]
    )
    payload = {
        "bucket": "capture-bucket",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
    }
    boom_calls: list[dict] = []

    def crashing_run_e2e(**kwargs):
        boom_calls.append(kwargs)
        raise RuntimeError("pod died mid-run")

    with pytest.raises(RuntimeError):
        process_handoff_payload(
            payload,
            storage_root=tmp_path,
            provider="openai",
            run_e2e=crashing_run_e2e,
            storage_client=client,  # type: ignore[arg-type]
        )
    capture_root = tmp_path / "capture-bucket" / prefix
    ledger = json.loads(
        (capture_root / "pipeline_job_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "processing"

    def ok_run_e2e(**kwargs):
        return {"status": "ok"}

    retried = process_handoff_payload(
        payload,
        storage_root=tmp_path,
        provider="openai",
        run_e2e=ok_run_e2e,
        storage_client=client,  # type: ignore[arg-type]
    )
    assert retried["status"] == "processed"
    ledger = json.loads(
        (capture_root / "pipeline_job_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "completed"
    assert ledger["attempt_count"] == 2


def test_stage_handoff_synthesizes_missing_pipeline_handoff(tmp_path: Path) -> None:
    """XR-03: a real iOS bundle (no hand-authored pipeline_handoff.json) stages without error."""

    prefix = "scenes/scene-1/captures/capture-1"
    handoff = HandoffMessage(
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        raw_prefix_uri="gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        pipeline_handoff_uri=(
            "gs://capture-bucket/scenes/scene-1/captures/capture-1/pipeline_handoff.json"
        ),
    )
    client = FakeStorageClient(_ios_bundle_blobs(prefix))

    capture_root = stage_handoff_capture(handoff, storage_root=tmp_path, storage_client=client)  # type: ignore[arg-type]

    synthesized = capture_root / "pipeline_handoff.json"
    assert synthesized.is_file(), "stage must synthesize pipeline_handoff.json for real iOS bundles"
    payload = json.loads(synthesized.read_text())
    assert payload["scene_id"] == "scene-1"
    assert payload["capture_id"] == "capture-1"
    assert payload["site_submission_id"] == "site-submission-scene-1"
    assert payload["buyer_request_id"] == "req-scene-1"
    assert payload["capture_job_id"] == "capture-job-scene-1"
    assert payload["owner_system"]["request_id"] == "req-scene-1"
    assert payload["synthesized"] is True


def test_stage_handoff_preserves_hand_authored_pipeline_handoff(tmp_path: Path) -> None:
    """A bundle that already carries pipeline_handoff.json is never overwritten by synthesis."""

    prefix = "scenes/scene-1/captures/capture-1"
    handoff = HandoffMessage(
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        raw_prefix_uri="gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        pipeline_handoff_uri=None,
    )
    hand_authored = {"owner_system": {"request_id": "hand-authored"}, "hand_authored": True}
    blobs = _ios_bundle_blobs(prefix) + [
        FakeBlob(f"{prefix}/pipeline_handoff.json", json.dumps(hand_authored).encode("utf-8")),
    ]
    client = FakeStorageClient(blobs)

    capture_root = stage_handoff_capture(handoff, storage_root=tmp_path, storage_client=client)  # type: ignore[arg-type]

    payload = json.loads((capture_root / "pipeline_handoff.json").read_text())
    assert payload == hand_authored


def test_stage_handoff_missing_upload_complete_still_raises(tmp_path: Path) -> None:
    """Synthesis must not paper over a genuinely broken bundle (no capture_upload_complete.json)."""

    prefix = "scenes/scene-1/captures/capture-1"
    handoff = HandoffMessage(
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        raw_prefix_uri="gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        pipeline_handoff_uri=None,
    )
    blobs = [
        FakeBlob(f"{prefix}/raw/manifest.json", json.dumps(_IOS_MANIFEST).encode("utf-8")),
        FakeBlob(f"{prefix}/raw/capture_context.json", json.dumps(_IOS_CONTEXT).encode("utf-8")),
    ]
    client = FakeStorageClient(blobs)

    with pytest.raises(PipelineError, match="capture_upload_complete.json"):
        stage_handoff_capture(handoff, storage_root=tmp_path, storage_client=client)  # type: ignore[arg-type]
