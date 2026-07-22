from __future__ import annotations

import threading
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from blueprint_pipeline.provider_bundle_staging_common import (
    BUNDLE_ROUTE,
    OUTPUT_ROUTE,
    create_staging_server,
    prepare_provider_bundle_staging,
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


def test_provider_neutral_staging_manifest_preserves_secret_boundary(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as archive:
        archive.writestr("request.json", "{}")
    token_path = tmp_path / "secrets" / "token"
    secret_env_path = tmp_path / "secrets" / "staging.env"

    manifest = prepare_provider_bundle_staging(
        job_dir=tmp_path / "job",
        bundle_path=bundle_path,
        public_base_url="https://staging.example/base?discard=this",
        token_file=token_path,
        secret_env_file=secret_env_path,
        generated_at="2026-07-22T00:00:00+00:00",
    )

    assert manifest["schema_version"] == "provider_bundle_staging_manifest.v1"
    assert manifest["status"] == "ready"
    assert manifest["base_url_redacted"] == "https://staging.example/base?REDACTED_QUERY"
    assert manifest["bundle_url_path"] == "/bundle.zip?token=<redacted-token>"
    assert manifest["output_put_url_path"] == "/output.zip?token=<redacted-token>"
    assert manifest["raw_secret_values_recorded"] is False
    token = token_path.read_text(encoding="utf-8").strip()
    assert token
    assert token not in str(manifest)
    assert token in secret_env_path.read_text(encoding="utf-8")
    assert oct(secret_env_path.stat().st_mode & 0o777) == "0o600"
    assert (tmp_path / "job" / "provider_bundle_staging_manifest.json").is_file()


def test_provider_neutral_staging_manifest_fails_closed_for_bad_bundle(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "not-a-zip.zip"
    bundle_path.write_text("not a zip", encoding="utf-8")

    manifest = prepare_provider_bundle_staging(
        job_dir=tmp_path / "job",
        bundle_path=bundle_path,
        public_base_url=None,
        token_file=tmp_path / "secrets" / "token",
        secret_env_file=tmp_path / "secrets" / "staging.env",
    )

    assert manifest["status"] == "blocked"
    assert "public_base_url_missing" in manifest["blockers"]
    assert any(
        str(blocker).startswith("provider_runtime_bundle_zip_inspection_failed:")
        for blocker in manifest["blockers"]
    )
    assert manifest["provider_fetchable_bundle_uri_ready"] is False


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
