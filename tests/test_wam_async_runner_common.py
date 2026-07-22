from __future__ import annotations

import json

import pytest

from blueprint_pipeline.wam_async_runner_common import (
    download_url_to_file,
    read_json_mapping,
    read_sensitive_url_file,
    redact_provider_url,
)


def test_redact_provider_url_removes_query_and_fragment() -> None:
    value = redact_provider_url("https://objects.example/bundle.zip?secret=1#token")

    assert value == "https://objects.example/bundle.zip?REDACTED_QUERY#REDACTED_FRAGMENT"
    assert "secret=1" not in value
    assert "token" not in value


def test_read_json_mapping_rejects_non_object_shape(tmp_path) -> None:
    path = tmp_path / "value.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    assert read_json_mapping(path) == {}


def test_read_sensitive_url_file_reports_metadata_without_echoing_value(tmp_path) -> None:
    path = tmp_path / "signed-url.txt"
    secret_url = "https://objects.example/bundle.zip?credential=do-not-record"
    path.write_text(secret_url, encoding="utf-8")
    path.chmod(0o600)

    value, metadata = read_sensitive_url_file(str(path), label="provider_bundle_url")

    assert value == secret_url
    assert metadata["mode_is_0600"] is True
    assert metadata["value_present"] is True
    assert metadata["raw_secret_values_recorded"] is False
    assert secret_url not in json.dumps(metadata)


def test_download_url_to_file_does_not_record_signed_url(tmp_path, monkeypatch) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b"provider-output"

    monkeypatch.setattr(
        "blueprint_pipeline.wam_async_runner_common.urllib.request.urlopen",
        lambda *args, **kwargs: Response(),
    )
    url = "https://objects.example/output.zip?credential=do-not-record"
    output = tmp_path / "output.zip"

    result = download_url_to_file(
        url=url,
        output_path=output,
        user_agent="BlueprintTest/1.0",
        timeout_seconds=10,
    )

    assert result["status"] == "completed"
    assert result["downloaded_size_bytes"] == len(b"provider-output")
    assert output.read_bytes() == b"provider-output"
    assert url not in json.dumps(result)


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/provider-output.zip",
        "http://objects.example/output.zip",
        "https://user:password@objects.example/output.zip",
    ],
)
def test_download_url_to_file_rejects_unsafe_url_schemes_and_credentials(
    tmp_path,
    url: str,
) -> None:
    result = download_url_to_file(
        url=url,
        output_path=tmp_path / "output.zip",
        user_agent="BlueprintTest/1.0",
        timeout_seconds=10,
    )

    assert result["status"] == "blocked"
    assert result["error_type"] == "ValueError"
