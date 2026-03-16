from __future__ import annotations

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
