from __future__ import annotations

import json

import pytest

from blueprint_pipeline.webapp_sync import WebappSyncError, sync_webapp_pipeline_attachment


def _minimal_payload() -> dict[str, object]:
    return {
        "site_submission_id": "site-1",
        "request_id": "request-1",
        "buyer_request_id": "buyer-request-1",
        "capture_job_id": "capture-job-1",
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

    assert result["status"] == "skipped"
    assert result["reason"] == "sync_not_configured"
    assert result["attempts"] == 0
    assert result["attachment_payload"]["qualification_state"] == "qualified_ready"
    assert result["attachment_payload"]["placeholder_fallback_allowed"] is False
    assert result["attachment_payload"]["upstream_links_verified"] is True


def test_sync_webapp_pipeline_attachment_raises_when_required(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.setenv("PIPELINE_SYNC_REQUIRED", "true")

    with pytest.raises(WebappSyncError, match="sync_not_configured"):
        sync_webapp_pipeline_attachment(**_minimal_payload())


@pytest.mark.parametrize(
    "field",
    ["site_submission_id", "request_id", "buyer_request_id", "capture_job_id"],
)
def test_sync_payload_requires_upstream_request_job_and_bootstrap_records(monkeypatch, field: str) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.setenv("PIPELINE_SYNC_REQUIRED", "true")
    payload = _minimal_payload()
    payload[field] = ""

    with pytest.raises(ValueError, match=field):
        sync_webapp_pipeline_attachment(**payload)


def test_sync_rejects_generated_capture_ids_as_upstream_links(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.setenv("PIPELINE_SYNC_REQUIRED", "true")
    payload = _minimal_payload()
    payload["site_submission_id"] = "scene-1:capture-1"
    payload["request_id"] = "scene-1:capture-1"

    with pytest.raises(ValueError, match="generated capture ids"):
        sync_webapp_pipeline_attachment(**payload)


def test_sync_rejects_placeholder_upstream_links(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.setenv("PIPELINE_SYNC_REQUIRED", "true")
    payload = _minimal_payload()
    payload["buyer_request_id"] = "example-buyer-request"

    with pytest.raises(ValueError, match="placeholder upstream ids"):
        sync_webapp_pipeline_attachment(**payload)


def test_sync_without_upstream_links_fails_closed_when_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_REQUIRED", raising=False)
    payload = _minimal_payload()
    payload["capture_job_id"] = ""

    result = sync_webapp_pipeline_attachment(**payload)

    assert result["status"] == "failed"
    assert result["blocker"] == "missing_upstream_pipeline_records"
    assert result["attempts"] == 0
    assert result["attachment_payload"]["upstream_links_verified"] is False
    assert result["attachment_payload"]["missing_upstream_links"] == ["capture_job_id"]
    assert result["attachment_payload"]["placeholder_fallback_allowed"] is False
    assert result["buyer_access_check"]["blocker"] == "missing_upstream_pipeline_records"


def test_placeholder_request_fallback_requires_explicit_internal_flag(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_REQUIRED", raising=False)
    monkeypatch.setenv("PIPELINE_SYNC_ALLOW_PLACEHOLDER_REQUESTS", "true")

    result = sync_webapp_pipeline_attachment(**_minimal_payload())

    assert result["status"] == "skipped"
    assert result["attachment_payload"]["placeholder_fallback_allowed"] is True
    assert result["attachment_payload"]["upstream_links_verified"] is True


def test_sync_webapp_pipeline_attachment_returns_buyer_access_and_checksums(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.test/api/pipeline-sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "token")
    monkeypatch.setenv("PIPELINE_BUYER_ACCESS_CHECK_URL", "https://webapp.test/api/buyer-access")

    class _Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

    def _fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        assert timeout > 0
        if request.full_url.endswith("/buyer-access"):
            return _Response({"buyer_accessible": True})
        return _Response(
            {
                "attachment_id": "att-1",
                "listing_id": "listing-1",
                "artifact_id": "artifact-1",
            }
        )

    monkeypatch.setattr("blueprint_pipeline.webapp_sync.urllib_request.urlopen", _fake_urlopen)

    result = sync_webapp_pipeline_attachment(**_minimal_payload())

    assert result["status"] == "succeeded"
    assert result["webapp_response_ids"]["attachment_id"] == "att-1"
    assert result["webapp_response_ids"]["listing_id"] == "listing-1"
    assert result["artifact_uri_checksums"]["qualification_summary_uri"]
    assert result["buyer_access_check"]["buyer_access_checked"] is True
    assert result["buyer_access_check"]["buyer_accessible"] is True


def test_production_sync_is_required_and_disables_placeholder_fallback(monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_REQUIRED", raising=False)

    with pytest.raises(WebappSyncError, match="sync_not_configured"):
        sync_webapp_pipeline_attachment(**_minimal_payload())
