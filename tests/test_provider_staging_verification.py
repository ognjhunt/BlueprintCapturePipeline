from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline import provider_staging_verification as staging_verification
from blueprint_pipeline.provider_staging_verification import (
    get_bundle_url,
    head_bundle_url,
    put_output_probe,
    verify_provider_staging_urls,
)


def test_signed_get_bundle_probe_checks_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"exact-bundle")

    class Response:
        status = 200
        headers = {"Content-Type": "application/zip"}

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _max_bytes: int) -> bytes:
            return b"exact-bundle"

    monkeypatch.setattr(staging_verification.urllib.request, "urlopen", lambda *_a, **_k: Response())

    result = get_bundle_url(
        bundle_url="https://objects.example/bundle.zip?signed=secret",
        bundle_path=bundle,
        timeout_seconds=1,
    )

    assert result["status"] == "passed"
    assert result["method"] == "GET"
    assert result["received_sha256"] == result["expected_sha256"]


def test_provider_staging_url_probes_reject_non_http_transports(tmp_path: Path) -> None:
    bundle_probe = head_bundle_url(
        bundle_url="file:///tmp/bundle.zip",
        bundle_path=tmp_path / "bundle.zip",
        timeout_seconds=1,
    )
    output_probe = put_output_probe(
        output_put_url="/relative/output.zip",
        timeout_seconds=1,
        probe_zip=b"PK",
    )

    assert bundle_probe["status"] == "blocked"
    assert bundle_probe["error_type"] == "ValueError"
    assert output_probe["status"] == "blocked"
    assert output_probe["error_type"] == "ValueError"


def test_provider_staging_verification_is_provider_neutral_and_redacted(
    tmp_path: Path,
) -> None:
    cleanup_calls: list[bool] = []

    def passing_head_probe(**_kwargs: object) -> dict[str, object]:
        return {"status": "passed", "method": "HEAD"}

    def passing_output_probe(**_kwargs: object) -> dict[str, object]:
        return {"status": "passed", "method": "PUT"}

    def cleanup_probe(**kwargs: object) -> dict[str, object]:
        cleanup_calls.append(bool(kwargs["cleanup_output_probe"]))
        return {"status": "removed"}

    ticks = iter([10.0, 10.1])
    result = verify_provider_staging_urls(
        job_dir=tmp_path,
        provider_bundle_url="https://staging.example/bundle.zip?token=bundle-secret",
        provider_output_put_url="https://staging.example/output.zip?token=output-secret",
        max_wait_seconds=1,
        retry_interval_seconds=0,
        timeout_seconds=1,
        head_probe=passing_head_probe,
        output_probe=passing_output_probe,
        cleanup_probe=cleanup_probe,
        monotonic=lambda: next(ticks),
        sleep=lambda _seconds: None,
        now_iso=lambda: "2026-07-22T00:00:00+00:00",
    )

    assert result["schema_version"] == "provider_staging_verification.v1"
    assert result["status"] == "passed"
    assert result["provider_bundle_url_redacted"].endswith("?REDACTED_QUERY")
    assert result["provider_output_put_url_redacted"].endswith("?REDACTED_QUERY")
    assert "bundle-secret" not in str(result)
    assert "output-secret" not in str(result)
    assert cleanup_calls == [True]
    assert (tmp_path / "provider_staging_verification.json").is_file()


def test_provider_staging_verification_missing_urls_fails_without_probing(
    tmp_path: Path,
) -> None:
    result = verify_provider_staging_urls(
        job_dir=tmp_path,
        provider_bundle_url="",
        provider_output_put_url="",
        now_iso=lambda: "2026-07-22T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert result["attempt_count"] == 0
    assert result["blockers"] == [
        "provider_bundle_fetch_url_missing",
        "provider_output_put_url_missing",
    ]
