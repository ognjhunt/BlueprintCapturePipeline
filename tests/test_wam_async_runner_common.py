from __future__ import annotations

import json

from blueprint_pipeline.wam_async_runner_common import (
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
