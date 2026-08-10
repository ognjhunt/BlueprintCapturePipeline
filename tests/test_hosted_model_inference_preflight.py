from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.hosted_model_inference_preflight import (
    materialize_hosted_model_inference_preflight,
)


def test_inference_preflight_requires_actual_bounded_completion(tmp_path: Path) -> None:
    observed = {}

    def fake_post(endpoint, headers, payload):
        observed.update(endpoint=endpoint, headers=headers, payload=json.loads(payload))
        return 200, b'{"model":"gpt-4.1","choices":[{"message":{"content":"OK"}}]}'

    output = tmp_path / "receipt.json"
    result = materialize_hosted_model_inference_preflight(
        output_path=output,
        backend="openai",
        generated_at="2026-08-10T00:00:00+00:00",
        secret_loader=lambda backend: ("secret-value", "fixture"),
        http_post=fake_post,
    )

    assert result["status"] == "qualified"
    assert result["credential_validated"] is True
    assert result["max_output_tokens"] == 1
    assert observed["payload"]["max_tokens"] == 1
    assert observed["headers"]["Authorization"] == "Bearer secret-value"
    assert "secret-value" not in output.read_text()


def test_public_catalog_visibility_cannot_qualify_inference(tmp_path: Path) -> None:
    result = materialize_hosted_model_inference_preflight(
        output_path=tmp_path / "receipt.json",
        backend="nvidia_nim",
        secret_loader=lambda backend: ("catalog-only-key", "fixture"),
        http_post=lambda endpoint, headers, payload: (401, b'{"data":[]}'),
    )

    assert result["status"] == "blocked"
    assert result["credential_validated"] is False
    assert result["blockers"] == ["hosted_model_inference_response_invalid"]


def test_missing_key_abstains_before_network(tmp_path: Path) -> None:
    called = False

    def forbidden_post(endpoint, headers, payload):
        nonlocal called
        called = True
        raise AssertionError

    result = materialize_hosted_model_inference_preflight(
        output_path=tmp_path / "receipt.json",
        backend="openai",
        secret_loader=lambda backend: ("", "missing"),
        http_post=forbidden_post,
    )

    assert result["status"] == "blocked"
    assert called is False
