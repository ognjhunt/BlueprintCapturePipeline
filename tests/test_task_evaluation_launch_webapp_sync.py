from __future__ import annotations

import json

from blueprint_pipeline import task_evaluation_launch_webapp_sync as sync_module


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_args):  # type: ignore[no-untyped-def]
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _progress() -> dict[str, object]:
    return {
        "schema_version": "task_evaluation_launch_progress.v1",
        "launch_id": "launch-1",
        "run_id": "run-1",
        "request_digest": "sha256:" + "a" * 64,
        "phase": "intake_webapp_record_binding",
        "phase_status": "verified",
        "observed_at_iso": "2026-08-13T14:00:00+00:00",
        "elapsed_seconds": 0.0,
    }


def test_progress_sync_requires_a_matching_webapp_binding_receipt(monkeypatch) -> None:
    progress = _progress()
    monkeypatch.setattr(
        sync_module.urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            {
                "schema_version": (
                    "task_evaluation_launch_progress_web_sync_receipt.v1"
                ),
                "status": "recorded",
                "launch_id": progress["launch_id"],
                "run_id": progress["run_id"],
                "request_digest": progress["request_digest"],
                "phase": progress["phase"],
            }
        ),
    )

    result = sync_module.sync_launch_progress_to_webapp(
        progress=progress,
        endpoint_url="https://webapp.test/api/internal/task-evaluation-launch-progress",
        token="test-token",
    )

    assert result["status"] == "succeeded"
    assert result["response"]["launch_id"] == "launch-1"


def test_progress_sync_rejects_a_different_webapp_record_binding(monkeypatch) -> None:
    progress = _progress()
    monkeypatch.setattr(
        sync_module.urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            {
                "schema_version": (
                    "task_evaluation_launch_progress_web_sync_receipt.v1"
                ),
                "status": "recorded",
                "launch_id": "different-launch",
                "run_id": progress["run_id"],
                "request_digest": progress["request_digest"],
                "phase": progress["phase"],
            }
        ),
    )

    result = sync_module.sync_launch_progress_to_webapp(
        progress=progress,
        endpoint_url="https://webapp.test/api/internal/task-evaluation-launch-progress",
        token="test-token",
    )

    assert result == {
        "schema_version": "task_evaluation_launch_webapp_progress_result.v1",
        "launch_id": "launch-1",
        "run_id": "run-1",
        "request_digest": "sha256:" + "a" * 64,
        "status": "failed",
        "reason": "response_binding_mismatch",
    }


def test_scene_configuration_sync_requires_atomic_launch_ready_offering_ack(
    monkeypatch,
) -> None:
    offering_digest = "sha256:" + "e" * 64
    receipt = {
        "launch_id": "launch-1",
        "run_id": "run-1",
        "request_digest": "sha256:" + "a" * 64,
        "receipt_digest": "sha256:" + "b" * 64,
        "terminal_evidence": {
            "scene_configuration": {
                "configured_scene_offering": {
                    "offering_digest": offering_digest,
                }
            }
        },
    }
    response = {
        "launch_id": "launch-1",
        "run_id": "run-1",
        "request_digest": "sha256:" + "a" * 64,
        "receipt_digest": "sha256:" + "b" * 64,
        "configured_scene_offering_digest": offering_digest,
        "configured_scene_offering_status": "launch_ready",
    }
    monkeypatch.setattr(
        sync_module.urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(response),
    )

    result = sync_module.sync_launch_receipt_to_webapp(
        receipt=receipt,
        endpoint_url="https://webapp.test/api/internal/task-evaluation-launches",
        token="test-token",
    )

    assert result["status"] == "succeeded"
    assert result["configured_scene_offering_status"] == "launch_ready"

    response.pop("configured_scene_offering_digest")
    refused = sync_module.sync_launch_receipt_to_webapp(
        receipt=receipt,
        endpoint_url="https://webapp.test/api/internal/task-evaluation-launches",
        token="test-token",
    )
    assert refused["status"] == "failed"
    assert refused["reason"] == "configured_scene_offering_binding_mismatch"
