from __future__ import annotations

import json

import pytest

from blueprint_pipeline.webapp_sync import WebappSyncError, sync_webapp_pipeline_attachment


def _minimal_payload() -> dict[str, object]:
    return {
        "site_submission_id": "site-1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        "qualification_state": "qualified_ready",
        "opportunity_state": "handoff_ready",
        "artifacts": {"qualification_summary_uri": "gs://bucket/path.json"},
    }


def test_sync_webapp_pipeline_attachment_skips_when_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_REQUIRED", raising=False)

    result = sync_webapp_pipeline_attachment(**_minimal_payload())

    assert result == {"status": "skipped", "reason": "sync_not_configured", "attempts": 0}


def test_sync_webapp_pipeline_attachment_raises_when_required(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.setenv("PIPELINE_SYNC_REQUIRED", "true")

    with pytest.raises(WebappSyncError, match="sync_not_configured"):
        sync_webapp_pipeline_attachment(**_minimal_payload())


def test_sync_webapp_pipeline_attachment_forwards_capture_completion_timestamp(
    monkeypatch, tmp_path
) -> None:
    raw_complete_path = tmp_path / "capture_upload_complete.json"
    raw_complete_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "raw_prefix": "scenes/scene-1/captures/capture-1/raw",
                "completed_at": "2026-03-20T14:03:02Z",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://example.test/webapp-sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "secret")
    monkeypatch.setenv("PIPELINE_SYNC_REQUIRED", "true")

    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"ok":true}'

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("blueprint_pipeline.webapp_sync.urllib_request.urlopen", fake_urlopen)

    result = sync_webapp_pipeline_attachment(
        **_minimal_payload(),
        raw_capture_complete_path=raw_complete_path,
    )

    assert result["status"] == "succeeded"
    assert captured["payload"]["latest_capture_completed_at"] == "2026-03-20T14:03:02Z"
