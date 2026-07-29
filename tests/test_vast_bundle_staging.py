from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import provider_bundle_staging_common as staging_common
from blueprint_pipeline import vast_bundle_staging as staging


pytestmark = pytest.mark.slow


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_zip_bundle(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/manifest.json", "{}\n")


def test_prepare_vast_bundle_staging_writes_redacted_manifest_and_secret_env(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "isaac_provider_runtime_bundle.zip"
    _write_zip_bundle(bundle)
    token_file = tmp_path / "vast_bundle_staging_token"
    secret_env = tmp_path / "urls.env"

    manifest = staging.prepare_vast_bundle_staging(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=token_file,
        secret_env_file=secret_env,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert manifest["status"] == "ready"
    assert manifest["bundle_zip_integrity_passed"] is True
    assert manifest["bundle_zip_parse_error"] is None
    assert manifest["provider_fetchable_bundle_uri_ready"] is True
    assert manifest["provider_output_callback_ready"] is True
    assert manifest["bundle_url_path"] == "/bundle.zip?token=<redacted-token>"
    assert manifest["output_put_url_path"] == "/output.zip?token=<redacted-token>"
    assert token_file.is_file()
    assert oct(token_file.stat().st_mode & 0o777) == "0o600"
    assert secret_env.is_file()
    assert oct(secret_env.stat().st_mode & 0o777) == "0o600"
    raw_token = token_file.read_text(encoding="utf-8").strip()
    persisted_manifest = (tmp_path / "vast_bundle_staging_manifest.json").read_text(
        encoding="utf-8"
    )
    assert manifest["token_file"]["path_redacted"] is True
    assert manifest["secret_env_file"]["path_redacted"] is True
    assert "path" not in manifest["token_file"]
    assert "path" not in manifest["secret_env_file"]
    assert manifest["secret_artifact_policy"]["local_secret_file_paths_recorded"] is False
    assert str(token_file) not in persisted_manifest
    assert str(secret_env) not in persisted_manifest
    assert raw_token not in persisted_manifest
    assert raw_token in secret_env.read_text(encoding="utf-8")


def test_prepare_vast_bundle_staging_blocks_without_public_base_url(tmp_path: Path) -> None:
    bundle = tmp_path / "isaac_provider_runtime_bundle.zip"
    _write_zip_bundle(bundle)

    manifest = staging.prepare_vast_bundle_staging(
        job_dir=tmp_path,
        bundle_path=bundle,
        token_file=tmp_path / "token",
    )

    assert manifest["status"] == "blocked"
    assert manifest["blockers"] == ["public_base_url_missing"]
    assert manifest["provider_bundle_url_present"] is False
    assert manifest["provider_output_put_url_present"] is False
    assert manifest["raw_secret_values_recorded"] is False


def test_prepare_vast_bundle_staging_blocks_malformed_bundle_zip(tmp_path: Path) -> None:
    bundle = tmp_path / "isaac_provider_runtime_bundle.zip"
    bundle.write_bytes(b"not a zip")

    manifest = staging.prepare_vast_bundle_staging(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert manifest["status"] == "blocked"
    assert manifest["bundle_zip_integrity_passed"] is False
    assert manifest["bundle_zip_parse_error"].startswith("BadZipFile:")
    assert "provider_runtime_bundle_zip_inspection_failed:BadZipFile" in manifest[
        "blockers"
    ]


def test_vast_bundle_staging_server_serves_bundle_and_accepts_put(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"zip bundle")
    output = tmp_path / "output.zip"
    token = "unit-test-token"
    server = staging.create_staging_server(
        bundle_path=bundle,
        output_path=output,
        token=token,
        max_output_bytes=1024,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        base_url = f"http://{host}:{port}"
        health = urllib.request.urlopen(f"{base_url}/health", timeout=5)
        assert health.status == 200
        head = urllib.request.urlopen(
            urllib.request.Request(
                f"{base_url}/bundle.zip?token={token}",
                method="HEAD",
            ),
            timeout=5,
        )
        assert head.status == 200
        assert int(head.headers["Content-Length"]) == bundle.stat().st_size
        downloaded = urllib.request.urlopen(
            f"{base_url}/bundle.zip?token={token}",
            timeout=5,
        ).read()
        assert downloaded == b"zip bundle"
        upload = urllib.request.urlopen(
            urllib.request.Request(
                f"{base_url}/output.zip?token={token}",
                data=b"runtime output",
                method="PUT",
                headers={"Content-Type": "application/zip"},
            ),
            timeout=5,
        )
        assert upload.status == 200
        assert output.read_bytes() == b"runtime output"
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(
                f"{base_url}/bundle.zip?token=wrong",
                timeout=5,
            )
        assert excinfo.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_vast_bundle_staging_self_test_writes_manifest_without_token(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"zip bundle")
    token_file = tmp_path / "token"

    result = staging.run_local_staging_self_test(
        job_dir=tmp_path,
        bundle_path=bundle,
        token_file=token_file,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "passed"
    assert result["provider_public_base_url_ready"] is False
    assert result["provider_public_base_url_blocker"] == "public_tunnel_not_started"
    assert (tmp_path / "vast_staging_self_test_output.zip").is_file()
    persisted = (tmp_path / "vast_bundle_staging_self_test.json").read_text(
        encoding="utf-8"
    )
    assert result["token_file"]["path_redacted"] is True
    assert "path" not in result["token_file"]
    assert result["secret_artifact_policy"]["local_secret_file_paths_recorded"] is False
    assert str(token_file) not in persisted
    assert token_file.read_text(encoding="utf-8").strip() not in persisted
    written = _read_json(tmp_path / "vast_bundle_staging_self_test.json")
    assert written["status"] == "passed"


def test_start_cloudflared_tunnel_records_public_url_without_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloudflared = tmp_path / "cloudflared"
    cloudflared.write_text("#!/bin/sh\n", encoding="utf-8")
    cloudflared.chmod(0o755)

    popen_calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeProcess:
        pid = 4242

        def __init__(self, *_args: object, **kwargs: object) -> None:
            popen_calls.append((list(_args[0]), dict(kwargs)))
            stdout = kwargs["stdout"]
            stdout.write("INF starting quick tunnel\n")
            stdout.write("INF https://stable-unit-test.trycloudflare.com\n")
            stdout.flush()

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            raise AssertionError("process should stay running when URL is observed")

    monkeypatch.setattr(staging.subprocess, "Popen", FakeProcess)

    manifest = staging.start_cloudflared_tunnel(
        job_dir=tmp_path,
        local_base_url="http://127.0.0.1:8819",
        cloudflared_path=cloudflared,
        transport_protocol="http2",
        startup_timeout_seconds=5,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert manifest["status"] == "running"
    assert manifest["public_base_url"] == "https://stable-unit-test.trycloudflare.com"
    assert manifest["pid"] == 4242
    assert manifest["transport_protocol"] == "http2"
    assert manifest["detached_process_session"] is True
    assert popen_calls[0][0][1:4] == ["tunnel", "--protocol", "http2"]
    assert popen_calls[0][1]["start_new_session"] is True
    assert manifest["cleanup_command"] == "kill 4242"
    persisted = (tmp_path / "vast_cloudflared_tunnel_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "token=" not in persisted


def test_start_cloudflared_tunnel_blocks_when_binary_missing(tmp_path: Path) -> None:
    manifest = staging.start_cloudflared_tunnel(
        job_dir=tmp_path,
        local_base_url="http://127.0.0.1:8819",
        cloudflared_path=tmp_path / "missing-cloudflared",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert manifest["status"] == "blocked"
    assert manifest["blockers"] == ["cloudflared_binary_missing"]


def test_verify_public_staging_urls_retries_and_cleans_output_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"zip bundle")
    output = tmp_path / "output.zip"
    token = "unit-test-token"
    calls: list[str] = []

    class FakeResponse:
        def __init__(
            self,
            *,
            status: int = 200,
            headers: dict[str, str] | None = None,
            body: bytes = b'{"ok": true}',
        ) -> None:
            self.status = status
            self.headers = headers or {}
            self._body = body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return self._body

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        method = request.get_method() if hasattr(request, "get_method") else "GET"
        calls.append(method)
        if calls == ["HEAD"]:
            raise urllib.error.URLError(OSError("dns not ready"))
        if method == "HEAD":
            return FakeResponse(
                headers={
                    "Content-Length": str(bundle.stat().st_size),
                    "Content-Type": "application/zip",
                }
            )
        if method == "PUT":
            output.write_bytes(getattr(request, "data", b""))
            return FakeResponse()
        raise AssertionError(f"unexpected method: {method}")

    monkeypatch.setattr(staging.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(staging.time, "sleep", lambda _seconds: None)

    result = staging.verify_public_staging_urls(
        job_dir=tmp_path,
        provider_bundle_url=f"https://example.trycloudflare.com/bundle.zip?token={token}",
        provider_output_put_url=f"https://example.trycloudflare.com/output.zip?token={token}",
        bundle_path=bundle,
        output_path=output,
        max_wait_seconds=2,
        retry_interval_seconds=0,
        timeout_seconds=1,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "passed"
    assert result["attempt_count"] == 2
    assert calls == ["HEAD", "HEAD", "PUT"]
    assert result["output_probe_cleanup"]["status"] == "removed"
    assert not output.exists()
    persisted = (tmp_path / "vast_public_staging_verification.json").read_text(
        encoding="utf-8"
    )
    assert token not in persisted
    assert "REDACTED_QUERY" in persisted


def test_verify_public_staging_urls_allows_inline_bundle_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_zip_bundle(bundle)
    output = tmp_path / "output.zip"
    token = "secret-token"
    calls: list[str] = []

    class FakeResponse:
        status = 200
        headers: dict[str, str] = {}

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"ok"

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        method = request.get_method() if hasattr(request, "get_method") else "GET"
        calls.append(method)
        if method == "PUT":
            output.write_bytes(getattr(request, "data", b""))
            return FakeResponse()
        raise AssertionError(f"unexpected method: {method}")

    monkeypatch.setattr(staging.urllib.request, "urlopen", fake_urlopen)

    result = staging.verify_public_staging_urls(
        job_dir=tmp_path,
        provider_bundle_url=f"https://example.trycloudflare.com/bundle.zip?token={token}",
        provider_output_put_url=f"https://example.trycloudflare.com/output.zip?token={token}",
        bundle_path=bundle,
        output_path=output,
        max_wait_seconds=1,
        retry_interval_seconds=0,
        timeout_seconds=1,
        require_bundle_fetch_probe=False,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "passed"
    assert result["require_bundle_fetch_probe"] is False
    assert result["attempts"][0]["bundle_probe"]["reason"] == (
        "bundle_fetch_replaced_by_inline_transport"
    )
    assert calls == ["PUT"]
    assert result["output_probe_cleanup"]["status"] == "removed"
    persisted = (tmp_path / "vast_public_staging_verification.json").read_text(
        encoding="utf-8"
    )
    assert token not in persisted


def test_public_staging_probe_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"zip bundle")
    http_error = urllib.error.HTTPError("https://example.test", 503, "nope", {}, None)
    assert staging._probe_exception(http_error)["http_status_code"] == 503

    class FakeResponse:
        def __init__(
            self,
            *,
            status: int = 200,
            headers: dict[str, str] | None = None,
            body: bytes = b"response",
        ) -> None:
            self.status = status
            self.headers = headers or {}
            self._body = body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return self._body

    responses = [
        FakeResponse(status=503),
        FakeResponse(headers={"Content-Length": str(bundle.stat().st_size + 1)}),
        FakeResponse(status=500),
    ]

    def fake_urlopen(_request: object, timeout: float) -> FakeResponse:
        return responses.pop(0)

    monkeypatch.setattr(staging.urllib.request, "urlopen", fake_urlopen)

    unreachable = staging._head_bundle_url(
        bundle_url="https://example.test/bundle.zip",
        bundle_path=bundle,
        timeout_seconds=1,
    )
    assert unreachable["blocker"] == "provider_bundle_fetch_url_unreachable"

    mismatch = staging._head_bundle_url(
        bundle_url="https://example.test/bundle.zip",
        bundle_path=bundle,
        timeout_seconds=1,
    )
    assert mismatch["blocker"] == "provider_bundle_fetch_url_size_mismatch"

    unwritable = staging._put_output_probe(
        output_put_url="https://example.test/output.zip",
        probe_zip=b"PK",
        timeout_seconds=1,
    )
    assert unwritable["blocker"] == "provider_output_put_url_unwritable"

    def raising_urlopen(_request: object, timeout: float) -> FakeResponse:
        raise urllib.error.URLError(OSError("offline"))

    monkeypatch.setattr(staging.urllib.request, "urlopen", raising_urlopen)
    failed_put = staging._put_output_probe(
        output_put_url="https://example.test/output.zip",
        probe_zip=b"PK",
        timeout_seconds=1,
    )
    assert failed_put["blocker"] == "provider_output_put_url_unwritable"
    assert failed_put["reason_type"] == "OSError"


def test_public_staging_cleanup_edges(tmp_path: Path) -> None:
    probe_zip = b"PK"
    missing = tmp_path / "missing-output.zip"
    assert staging._cleanup_output_probe(
        output_path=missing,
        probe_zip=probe_zip,
        cleanup_output_probe=False,
    ) == {"status": "skipped", "reason": "cleanup_output_probe_false"}
    assert staging._cleanup_output_probe(
        output_path=None,
        probe_zip=probe_zip,
        cleanup_output_probe=True,
    ) == {"status": "skipped", "reason": "output_path_missing"}
    assert staging._cleanup_output_probe(
        output_path=missing,
        probe_zip=probe_zip,
        cleanup_output_probe=True,
    ) == {"status": "skipped", "reason": "output_probe_file_not_present"}

    wrong = tmp_path / "wrong-output.zip"
    wrong.write_bytes(b"not probe")
    mismatch = staging._cleanup_output_probe(
        output_path=wrong,
        probe_zip=probe_zip,
        cleanup_output_probe=True,
    )
    assert mismatch["reason"] == "output_path_does_not_match_probe_bytes"
    assert wrong.exists()

    class ReadFailPath:
        def exists(self) -> bool:
            return True

        def read_bytes(self) -> bytes:
            raise OSError("cannot read")

    read_failed = staging._cleanup_output_probe(
        output_path=ReadFailPath(),  # type: ignore[arg-type]
        probe_zip=probe_zip,
        cleanup_output_probe=True,
    )
    assert read_failed["reason"] == "output_probe_file_read_failed"

    class UnlinkFailPath:
        def exists(self) -> bool:
            return True

        def read_bytes(self) -> bytes:
            return probe_zip

        def unlink(self) -> None:
            raise OSError("cannot remove")

    unlink_failed = staging._cleanup_output_probe(
        output_path=UnlinkFailPath(),  # type: ignore[arg-type]
        probe_zip=probe_zip,
        cleanup_output_probe=True,
    )
    assert unlink_failed["reason"] == "output_probe_file_cleanup_failed"


def test_verify_public_staging_urls_missing_urls_and_deadline_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"zip bundle")

    missing = staging.verify_public_staging_urls(
        job_dir=tmp_path / "missing",
        provider_bundle_url="",
        provider_output_put_url="",
        bundle_path=bundle,
        output_path=tmp_path / "output.zip",
    )
    assert missing["status"] == "blocked"
    assert missing["blockers"] == [
        "provider_bundle_fetch_url_missing",
        "provider_output_put_url_missing",
    ]

    class FakeResponse:
        status = 503
        headers: dict[str, str] = {}

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(staging.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    deadline = staging.verify_public_staging_urls(
        job_dir=tmp_path / "deadline",
        provider_bundle_url="https://example.test/bundle.zip",
        provider_output_put_url="https://example.test/output.zip",
        bundle_path=bundle,
        output_path=tmp_path / "output.zip",
        max_wait_seconds=0,
        retry_interval_seconds=0,
        timeout_seconds=1,
    )
    assert deadline["status"] == "blocked"
    assert "provider_bundle_fetch_url_unreachable" in deadline["blockers"]
    assert "provider_output_put_url_not_checked" in deadline["blockers"]


def test_verify_public_staging_urls_blocks_relative_urls_without_exception(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"zip bundle")

    relative_bundle = staging.verify_public_staging_urls(
        job_dir=tmp_path / "relative-bundle",
        provider_bundle_url="/bundle.zip?token=local-token",
        provider_output_put_url="https://example.test/output.zip",
        bundle_path=bundle,
        output_path=tmp_path / "output.zip",
        max_wait_seconds=0,
        retry_interval_seconds=0,
        timeout_seconds=1,
    )
    assert relative_bundle["status"] == "blocked"
    assert "provider_bundle_fetch_url_unreachable" in relative_bundle["blockers"]
    bundle_probe = relative_bundle["attempts"][0]["bundle_probe"]  # type: ignore[index]
    assert bundle_probe["error_type"] == "ValueError"  # type: ignore[index]

    relative_output = staging.verify_public_staging_urls(
        job_dir=tmp_path / "relative-output",
        provider_bundle_url="/bundle.zip?token=local-token",
        provider_output_put_url="/output.zip?token=local-token",
        bundle_path=bundle,
        output_path=tmp_path / "output.zip",
        max_wait_seconds=0,
        retry_interval_seconds=0,
        timeout_seconds=1,
        require_bundle_fetch_probe=False,
    )
    assert relative_output["status"] == "blocked"
    assert "provider_output_put_url_unwritable" in relative_output["blockers"]
    output_probe = relative_output["attempts"][0]["output_put_probe"]  # type: ignore[index]
    assert output_probe["error_type"] == "ValueError"  # type: ignore[index]


def test_verify_public_staging_urls_cleanup_block_and_skipped_put_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"zip bundle")

    monkeypatch.setattr(
        staging,
        "_head_bundle_url",
        lambda *args, **kwargs: {"status": "passed", "method": "HEAD"},
    )
    monkeypatch.setattr(
        staging,
        "_put_output_probe",
        lambda *args, **kwargs: {"status": "passed", "method": "PUT"},
    )
    monkeypatch.setattr(
        staging,
        "_cleanup_output_probe",
        lambda *args, **kwargs: {"status": "blocked", "reason": "cannot_remove_probe"},
    )
    cleanup_blocked = staging.verify_public_staging_urls(
        job_dir=tmp_path / "cleanup-blocked",
        provider_bundle_url="https://example.test/bundle.zip",
        provider_output_put_url="https://example.test/output.zip",
        bundle_path=bundle,
        output_path=tmp_path / "output.zip",
        max_wait_seconds=0,
        retry_interval_seconds=0,
        required_consecutive_successes=1,
    )
    assert cleanup_blocked["status"] == "blocked"
    assert cleanup_blocked["attempts"][0]["output_put_probe"]["blocker"] == "cannot_remove_probe"

    monkeypatch.setattr(
        staging,
        "_cleanup_output_probe",
        lambda *args, **kwargs: {"status": "not_requested"},
    )
    skipped_put = staging.verify_public_staging_urls(
        job_dir=tmp_path / "skipped-put",
        provider_bundle_url="https://example.test/bundle.zip",
        provider_output_put_url="https://example.test/output.zip",
        bundle_path=bundle,
        output_path=tmp_path / "output.zip",
        max_wait_seconds=0,
        retry_interval_seconds=0,
        required_consecutive_successes=1,
        allow_output_put_probe=False,
    )
    assert skipped_put["status"] == "passed"
    assert skipped_put["warnings"] == ["provider_output_put_url_not_mutation_probed"]
    assert skipped_put["attempts"][0]["output_put_probe"]["reason"] == (
        "output_put_probe_requires_explicit_allow"
    )

    stability_blocked = staging.verify_public_staging_urls(
        job_dir=tmp_path / "stability-blocked",
        provider_bundle_url="https://example.test/bundle.zip",
        provider_output_put_url="https://example.test/output.zip",
        bundle_path=bundle,
        output_path=tmp_path / "output.zip",
        max_wait_seconds=0,
        retry_interval_seconds=0,
        required_consecutive_successes=2,
        allow_output_put_probe=False,
    )
    assert stability_blocked["status"] == "blocked"
    assert stability_blocked["blockers"] == ["public_staging_url_stability_not_proven"]


def test_prepare_vast_bundle_staging_edge_blockers_and_redaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_zip_bundle(bundle)
    token_file = tmp_path / "token"
    token_file.write_text("existing-token\n", encoding="utf-8")

    manifest = staging.prepare_vast_bundle_staging(
        job_dir=tmp_path / "existing-token",
        bundle_path=bundle,
        public_base_url="http://localhost:8765",
        token_file=token_file,
        secret_env_file=tmp_path / "existing-token.env",
    )
    assert manifest["token_file"]["created"] is False  # type: ignore[index]
    assert manifest["warnings"] == ["local_base_url_not_provider_fetchable_from_vast"]
    assert staging._redact_url("https://example.test/path?token=secret&x=1") == (
        "https://example.test/path?REDACTED_QUERY"
    )

    missing_bundle = staging.prepare_vast_bundle_staging(
        job_dir=tmp_path / "missing-bundle",
        bundle_path=tmp_path / "missing.zip",
        public_base_url="ftp://example.test",
        token_file=tmp_path / "missing-token",
    )
    assert missing_bundle["blockers"] == [
        "provider_runtime_bundle_missing",
        "public_base_url_scheme_not_http",
    ]

    empty_token = tmp_path / "empty-token"
    empty_token.write_text("\n", encoding="utf-8")
    token_blocked = staging.prepare_vast_bundle_staging(
        job_dir=tmp_path / "empty-token-job",
        bundle_path=bundle,
        public_base_url="https://example.test",
        token_file=empty_token,
        secret_env_file=tmp_path / "empty-token.env",
    )
    assert "staging_token_missing" in token_blocked["blockers"]

    class FakeZipFile:
        def __init__(self, path: Path) -> None:
            self.path = path

        def __enter__(self) -> "FakeZipFile":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def namelist(self) -> list[str]:
            return ["provider_runtime/manifest.json"]

        def testzip(self) -> str:
            return "provider_runtime/manifest.json"

    monkeypatch.setattr(staging_common.zipfile, "ZipFile", FakeZipFile)
    corrupt_member = staging.prepare_vast_bundle_staging(
        job_dir=tmp_path / "corrupt-member",
        bundle_path=bundle,
        public_base_url="https://example.test",
        token_file=tmp_path / "corrupt-token",
        secret_env_file=tmp_path / "corrupt.env",
    )
    assert "provider_runtime_bundle_zip_integrity_failed" in corrupt_member["blockers"]
    assert corrupt_member["bundle_zip_testzip_result"] == "provider_runtime/manifest.json"


def test_vast_bundle_staging_server_error_responses(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"bundle")
    output = tmp_path / "output.zip"
    token = "unit-test-token"
    server = staging.create_staging_server(
        bundle_path=bundle,
        output_path=output,
        token=token,
        max_output_bytes=4,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        base_url = f"http://{host}:{port}"
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(
                urllib.request.Request(f"{base_url}/missing", method="HEAD"),
                timeout=5,
            )
        assert excinfo.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"{base_url}/missing", timeout=5)
        assert excinfo.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base_url}/wrong.zip?token={token}",
                    data=b"ok",
                    method="PUT",
                ),
                timeout=5,
            )
        assert excinfo.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base_url}/output.zip?token=wrong",
                    data=b"ok",
                    method="PUT",
                ),
                timeout=5,
            )
        assert excinfo.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base_url}/output.zip?token={token}",
                    data=b"",
                    method="PUT",
                ),
                timeout=5,
            )
        assert excinfo.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base_url}/output.zip?token={token}",
                    data=b"too-large",
                    method="PUT",
                ),
                timeout=5,
            )
        assert excinfo.value.code == 413
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    missing_bundle_server = staging.create_staging_server(
        bundle_path=tmp_path / "missing.zip",
        output_path=output,
        token=token,
    )
    thread = threading.Thread(target=missing_bundle_server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = missing_bundle_server.server_address
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"http://{host}:{port}/bundle.zip?token={token}", timeout=5)
        assert excinfo.value.code == 404
    finally:
        missing_bundle_server.shutdown()
        missing_bundle_server.server_close()
        thread.join(timeout=5)


def test_vast_bundle_staging_put_short_upload_branch(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"bundle")
    output = tmp_path / "output.zip"
    token = "unit-test-token"
    server = staging.create_staging_server(
        bundle_path=bundle,
        output_path=output,
        token=token,
    )

    class PartialBody:
        def __init__(self) -> None:
            self.calls = 0

        def read(self, _size: int) -> bytes:
            self.calls += 1
            return b"abc" if self.calls == 1 else b""

    class HarnessHandler(staging.VastBundleStagingRequestHandler):
        def __init__(self) -> None:
            self.path = f"{staging.OUTPUT_ROUTE}?token={token}"
            self.headers = {"Content-Length": "5"}
            self.rfile = PartialBody()
            self.server = server
            self.responses: list[tuple[int, dict[str, object]]] = []

        def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
            self.responses.append((status, payload))

    handler = HarnessHandler()
    handler.do_PUT()

    assert handler.responses == [(400, {"ok": False, "error": "short_upload"})]
    assert not output.exists()
    server.server_close()


def test_local_staging_self_test_records_mismatches_and_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"bundle")

    class FakeServer:
        server_address = ("127.0.0.1", 9999)

        def serve_forever(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

        def server_close(self) -> None:
            return None

    class FakeResponse:
        def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
            self.status = status
            self.headers = headers or {}

    monkeypatch.setattr(staging, "create_staging_server", lambda **kwargs: FakeServer())
    responses = [
        FakeResponse(200),
        FakeResponse(200, {"Content-Length": "999", "Content-Type": "application/zip"}),
        FakeResponse(200),
    ]

    def fake_urlopen(_request: object, timeout: int) -> FakeResponse:
        return responses.pop(0)

    monkeypatch.setattr(staging.urllib.request, "urlopen", fake_urlopen)
    mismatch = staging.run_local_staging_self_test(
        job_dir=tmp_path / "mismatch",
        bundle_path=bundle,
        token_file=tmp_path / "mismatch-token",
    )
    assert mismatch["status"] == "blocked"
    assert mismatch["blockers"] == [
        "bundle_head_content_length_mismatch",
        "output_put_file_not_written",
    ]

    def failing_urlopen(_request: object, timeout: int) -> FakeResponse:
        raise TimeoutError("offline")

    monkeypatch.setattr(staging.urllib.request, "urlopen", failing_urlopen)
    failed = staging.run_local_staging_self_test(
        job_dir=tmp_path / "failed",
        bundle_path=bundle,
        token_file=tmp_path / "failed-token",
    )
    assert failed["status"] == "blocked"
    assert failed["blockers"] == ["local_staging_self_test_failed:TimeoutError"]


def test_serve_wrapper_closes_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"bundle")
    token_file = tmp_path / "token"
    closed: list[str] = []

    class FakeServer:
        def serve_forever(self) -> None:
            closed.append("served")
            raise KeyboardInterrupt

        def server_close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr(staging, "create_staging_server", lambda **kwargs: FakeServer())
    with pytest.raises(KeyboardInterrupt):
        staging.serve_vast_bundle_staging(
            bundle_path=bundle,
            output_path=tmp_path / "output.zip",
            token_file=token_file,
            port=0,
        )

    assert closed == ["served", "closed"]


def test_cloudflared_tunnel_blockers_and_process_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = staging.start_cloudflared_tunnel(
        job_dir=tmp_path / "missing",
        local_base_url="",
        cloudflared_path=tmp_path / "missing-cloudflared",
        generated_at="now",
    )
    assert missing["status"] == "blocked"
    assert "cloudflared_binary_missing" in missing["blockers"]
    assert "local_base_url_missing" in missing["blockers"]

    bad_scheme = staging.start_cloudflared_tunnel(
        job_dir=tmp_path / "bad-scheme",
        local_base_url="ftp://localhost:8000",
        cloudflared_path=tmp_path / "missing-cloudflared",
        generated_at="now",
    )
    assert "local_base_url_scheme_not_http" in bad_scheme["blockers"]

    executable = tmp_path / "cloudflared"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")

    class TimeoutProcess:
        pid = 123

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            raise RuntimeError("terminate failed")

    monkeypatch.setattr(staging.subprocess, "Popen", lambda *_args, **_kwargs: TimeoutProcess())
    times = iter([0.0, 0.5, 2.0])
    monkeypatch.setattr(staging.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(staging.time, "sleep", lambda _seconds: None)
    original_read_text = Path.read_text

    def flaky_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == "vast_cloudflared_tunnel.log":
            raise OSError("log unavailable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)
    timeout = staging.start_cloudflared_tunnel(
        job_dir=tmp_path / "timeout",
        local_base_url="http://127.0.0.1:8000",
        cloudflared_path=executable,
        startup_timeout_seconds=1,
        generated_at="now",
    )
    assert timeout["blockers"] == ["cloudflared_public_url_not_observed_before_timeout"]

    class ExitedProcess:
        pid = 456

        def poll(self) -> int:
            return 1

        def terminate(self) -> None:
            return None

    monkeypatch.setattr(staging.subprocess, "Popen", lambda *_args, **_kwargs: ExitedProcess())
    times = iter([0.0, 0.5])
    monkeypatch.setattr(staging.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(Path, "read_text", original_read_text)
    exited = staging.start_cloudflared_tunnel(
        job_dir=tmp_path / "exited",
        local_base_url="http://127.0.0.1:8000",
        cloudflared_path=executable,
        startup_timeout_seconds=1,
        generated_at="now",
    )
    assert exited["blockers"] == ["cloudflared_process_exited_before_public_url"]


def test_vast_bundle_staging_cli_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def fake_prepare_vast_bundle_staging(**kwargs: object) -> dict[str, object]:
        calls.append("prepare")
        return {"status": "blocked", "blockers": ["missing_public_url"]}

    def fake_run_local_staging_self_test(**kwargs: object) -> dict[str, object]:
        calls.append("self-test")
        return {"status": "passed"}

    verify_results: list[dict[str, object]] = [
        {"status": "passed", "blockers": []},
        {"status": "blocked", "blockers": ["public_url_unreachable"]},
    ]

    def fake_verify_public_staging_urls(**kwargs: object) -> dict[str, object]:
        calls.append("verify-public")
        return verify_results.pop(0)

    def fake_serve_vast_bundle_staging(**kwargs: object) -> None:
        calls.append("serve")

    cloudflared_results: list[dict[str, object]] = [
        {"status": "running", "public_base_url": "https://public.trycloudflare.com", "blockers": []},
        {"status": "blocked", "public_base_url": None, "blockers": ["cloudflared_binary_missing"]},
    ]

    def fake_start_cloudflared_tunnel(**kwargs: object) -> dict[str, object]:
        calls.append("start-cloudflared")
        return cloudflared_results.pop(0)

    monkeypatch.setattr(staging, "prepare_vast_bundle_staging", fake_prepare_vast_bundle_staging)
    monkeypatch.setattr(staging, "run_local_staging_self_test", fake_run_local_staging_self_test)
    monkeypatch.setattr(staging, "verify_public_staging_urls", fake_verify_public_staging_urls)
    monkeypatch.setattr(staging, "serve_vast_bundle_staging", fake_serve_vast_bundle_staging)
    monkeypatch.setattr(staging, "start_cloudflared_tunnel", fake_start_cloudflared_tunnel)

    assert staging.main(
        [
            "prepare",
            "--job-dir",
            str(tmp_path),
            "--bundle-path",
            str(tmp_path / "bundle.zip"),
            "--public-base-url",
            "https://example.test",
            "--token-file",
            str(tmp_path / "token"),
            "--secret-env-file",
            str(tmp_path / "urls.env"),
            "--output-path",
            str(tmp_path / "output.zip"),
        ]
    ) == 1
    assert staging.main(
        [
            "self-test",
            "--job-dir",
            str(tmp_path),
            "--bundle-path",
            str(tmp_path / "bundle.zip"),
            "--output-path",
            str(tmp_path / "output.zip"),
            "--token-file",
            str(tmp_path / "token"),
        ]
    ) == 0
    assert staging.main(
        [
            "verify-public",
            "--job-dir",
            str(tmp_path),
            "--bundle-path",
            str(tmp_path / "bundle.zip"),
            "--public-base-url",
            "https://example.test",
            "--token-file",
            str(tmp_path / "token"),
            "--output-path",
            str(tmp_path / "output.zip"),
            "--max-wait-seconds",
            "1",
            "--retry-interval-seconds",
            "0",
            "--timeout-seconds",
            "1",
        ]
    ) == 0
    assert staging.main(
        [
            "verify-public",
            "--job-dir",
            str(tmp_path),
            "--bundle-path",
            str(tmp_path / "bundle.zip"),
            "--public-base-url",
            "https://example.test",
            "--token-file",
            str(tmp_path / "token"),
            "--output-path",
            str(tmp_path / "output.zip"),
            "--no-cleanup-output-probe",
        ]
    ) == 1
    assert staging.main(
        [
            "serve",
            "--bundle-path",
            str(tmp_path / "bundle.zip"),
            "--output-path",
            str(tmp_path / "output.zip"),
            "--token-file",
            str(tmp_path / "token"),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
        ]
    ) == 0
    assert staging.main(
        [
            "start-cloudflared",
            "--job-dir",
            str(tmp_path),
            "--local-base-url",
            "http://127.0.0.1:8000",
            "--cloudflared-path",
            str(tmp_path / "cloudflared"),
            "--startup-timeout-seconds",
            "1",
        ]
    ) == 0
    assert staging.main(
        [
            "start-cloudflared",
            "--job-dir",
            str(tmp_path),
            "--local-base-url",
            "http://127.0.0.1:8000",
        ]
    ) == 1

    output = capsys.readouterr().out
    assert "blockers=missing_public_url" in output
    assert "blockers=public_url_unreachable" in output
    assert "public_base_url=https://public.trycloudflare.com" in output
    assert "blockers=cloudflared_binary_missing" in output
    assert calls == [
        "prepare",
        "self-test",
        "verify-public",
        "verify-public",
        "serve",
        "start-cloudflared",
        "start-cloudflared",
    ]
