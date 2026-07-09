from __future__ import annotations

import pytest

from blueprint_pipeline import cloud_run_iam_auth as auth


def test_cloud_run_iam_auth_disabled_preserves_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BLUEPRINT_CLOUD_RUN_IAM_AUTH_ENABLED", raising=False)

    headers = auth.cloud_run_id_token_headers(
        {"Authorization": "Bearer runner-token"},
        url="https://runner.example/run",
    )

    assert headers == {"Authorization": "Bearer runner-token"}


def test_cloud_run_iam_auth_adds_serverless_header(monkeypatch: pytest.MonkeyPatch) -> None:
    audiences: list[str] = []

    def fetch_token(audience: str) -> str:
        audiences.append(audience)
        return "google-id-token"

    monkeypatch.setenv("BLUEPRINT_CLOUD_RUN_IAM_AUTH_ENABLED", "true")
    monkeypatch.setattr(auth, "_fetch_google_id_token", fetch_token)

    headers = auth.cloud_run_id_token_headers(
        {"Authorization": "Bearer runner-token", "Content-Type": "application/json"},
        url="https://sam3-detect-abc-uc.a.run.app/run",
    )

    assert audiences == ["https://sam3-detect-abc-uc.a.run.app"]
    assert headers["Authorization"] == "Bearer runner-token"
    assert headers["X-Serverless-Authorization"] == "Bearer google-id-token"


def test_cloud_run_iam_auth_rejects_invalid_audience_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_CLOUD_RUN_IAM_AUTH_ENABLED", "true")

    with pytest.raises(auth.CloudRunIamAuthError, match="invalid_audience_url"):
        auth.cloud_run_id_token_headers({}, url="not-a-url")
