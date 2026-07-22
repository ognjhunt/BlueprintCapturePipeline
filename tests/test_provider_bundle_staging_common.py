from __future__ import annotations

import threading
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from blueprint_pipeline.provider_bundle_staging_common import (
    BUNDLE_ROUTE,
    OUTPUT_ROUTE,
    create_staging_server,
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


def test_provider_neutral_server_serves_bundle_and_accepts_bounded_output(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "bundle.zip"
    output_path = tmp_path / "output.zip"
    bundle_path.write_bytes(b"bundle-bytes")
    token = "provider-neutral-token"
    server = create_staging_server(
        bundle_path=bundle_path,
        output_path=output_path,
        token=token,
        max_output_bytes=64,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    try:
        bundle_url = staging_url_with_token(base_url, BUNDLE_ROUTE, token)
        with urllib.request.urlopen(bundle_url, timeout=5) as response:
            assert response.status == 200
            assert response.read() == b"bundle-bytes"

        output_url = staging_url_with_token(base_url, OUTPUT_ROUTE, token)
        request = urllib.request.Request(
            output_url,
            data=b"output-bytes",
            method="PUT",
            headers={"Content-Type": "application/zip"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
        assert output_path.read_bytes() == b"output-bytes"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
