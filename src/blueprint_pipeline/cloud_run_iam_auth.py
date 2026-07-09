"""Helpers for calling private Cloud Run HTTP services."""

from __future__ import annotations

import os
from urllib import parse as urllib_parse

from .common import parse_bool


class CloudRunIamAuthError(RuntimeError):
    """Raised when Cloud Run IAM auth is enabled but an ID token cannot be built."""


def cloud_run_iam_auth_enabled() -> bool:
    return parse_bool(os.getenv("BLUEPRINT_CLOUD_RUN_IAM_AUTH_ENABLED"), default=False)


def cloud_run_id_token_audience(url: str) -> str:
    parsed = urllib_parse.urlsplit(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CloudRunIamAuthError("cloud_run_iam_auth_invalid_audience_url")
    return f"{parsed.scheme}://{parsed.netloc}"


def _fetch_google_id_token(audience: str) -> str:
    try:
        from google.auth.transport.requests import Request as GoogleAuthRequest
        from google.oauth2 import id_token
    except ImportError as exc:  # pragma: no cover - covered by package contract tests.
        raise CloudRunIamAuthError("cloud_run_iam_auth_google_auth_missing") from exc

    try:
        token = id_token.fetch_id_token(GoogleAuthRequest(), audience)
    except Exception as exc:  # pragma: no cover - depends on live ADC metadata/credentials.
        raise CloudRunIamAuthError(f"cloud_run_iam_auth_token_fetch_failed:{exc.__class__.__name__}") from exc
    token_text = str(token or "").strip()
    if not token_text:
        raise CloudRunIamAuthError("cloud_run_iam_auth_token_fetch_empty")
    return token_text


def cloud_run_id_token_headers(headers: dict[str, str], *, url: str) -> dict[str, str]:
    if not cloud_run_iam_auth_enabled():
        return dict(headers)
    audience = cloud_run_id_token_audience(url)
    token = _fetch_google_id_token(audience)
    merged = dict(headers)
    merged["X-Serverless-Authorization"] = f"Bearer {token}"
    return merged
