from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

from blueprint_pipeline.provider_bundle_staging_common import (
    BUNDLE_ROUTE,
    read_or_create_staging_token,
    staging_url_with_token,
)


def test_staging_token_is_persisted_mode_0600_without_manifest_disclosure(
    tmp_path: Path,
) -> None:
    token_path = tmp_path / "secrets" / "staging-token"

    token, status = read_or_create_staging_token(token_path)
    reread_token, reread_status = read_or_create_staging_token(token_path)

    assert token
    assert reread_token == token
    assert token_path.read_text(encoding="utf-8").strip() == token
    assert oct(token_path.stat().st_mode & 0o777) == "0o600"
    assert status["created"] is True
    assert reread_status["created"] is False
    assert status["token_recorded_in_manifest"] is False
    assert token not in str(status)


def test_staging_url_normalizes_route_and_places_token_only_in_query() -> None:
    url = staging_url_with_token(
        "https://staging.example/base?old=value",
        BUNDLE_ROUTE,
        "secret-token",
    )
    parsed = urlparse(url)

    assert parsed.scheme == "https"
    assert parsed.netloc == "staging.example"
    assert parsed.path == "/bundle.zip"
    assert parse_qs(parsed.query) == {"token": ["secret-token"]}
    assert parsed.fragment == ""
