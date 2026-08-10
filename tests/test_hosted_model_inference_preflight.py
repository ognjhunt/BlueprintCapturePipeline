from __future__ import annotations

import io
import json
from pathlib import Path
import urllib.error

from blueprint_pipeline.hosted_model_inference_preflight import (
    BACKENDS,
    materialize_hosted_model_inference_preflight,
)


def test_inference_preflight_requires_actual_bounded_completion(tmp_path: Path) -> None:
    observed = {}

    def fake_post(endpoint, headers, payload):
        observed.update(endpoint=endpoint, headers=headers, payload=json.loads(payload))
        return (
            200,
            b'{"model":"gpt-5.6-luna","choices":[{"message":{"content":'
            b"\"{\\\"dominant_color\\\":\\\"red\\\"}\"}}],"
            b'"usage":{"prompt_tokens":20,"completion_tokens":4,"total_tokens":24}}',
        )

    output = tmp_path / "receipt.json"
    result = materialize_hosted_model_inference_preflight(
        output_path=output,
        backend="openai",
        model="gpt-5.6-luna",
        reasoning_effort="max",
        generated_at="2026-08-10T00:00:00+00:00",
        secret_loader=lambda backend: ("secret-value", "fixture"),
        http_post=fake_post,
    )

    assert result["status"] == "qualified"
    assert result["credential_validated"] is True
    assert result["model"] == "gpt-5.6-luna"
    assert result["reasoning_effort"] == "max"
    assert result["probe_response_validated"] is True
    assert result["verified_capabilities"] == ["image_input", "structured_json"]
    assert result["max_output_tokens"] == 256
    assert observed["payload"]["max_completion_tokens"] == 256
    assert observed["payload"]["reasoning_effort"] == "max"
    assert "temperature" not in observed["payload"]
    assert observed["payload"]["response_format"]["type"] == "json_schema"
    assert observed["payload"]["messages"][0]["content"][1]["type"] == "image_url"
    assert result["probe_image"]["uploaded_scene_bytes"] is False
    assert result["usage"] == {
        "input_tokens": 20,
        "output_tokens": 4,
        "total_tokens": 24,
    }
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


def test_text_only_or_non_json_completion_cannot_claim_joint_capabilities(
    tmp_path: Path,
) -> None:
    result = materialize_hosted_model_inference_preflight(
        output_path=tmp_path / "receipt.json",
        backend="openai",
        model="arbitrary-model",
        secret_loader=lambda backend: ("inference-key", "fixture"),
        http_post=lambda endpoint, headers, payload: (
            200,
            b'{"model":"arbitrary-model","choices":[{"message":{"content":"red"}}]}',
        ),
    )

    assert result["status"] == "blocked"
    assert result["credential_validated"] is True
    assert result["verified_capabilities"] == []
    assert result["blockers"] == ["hosted_model_capability_response_invalid"]


def test_http_error_retains_only_safe_routing_fields(tmp_path: Path) -> None:
    def reject(endpoint, headers, payload):
        raise urllib.error.HTTPError(
            endpoint,
            400,
            "bad request",
            {},
            io.BytesIO(
                b'{"error":{"message":"do not retain me","type":"invalid_request_error",'
                b'"param":"reasoning_effort","code":"unsupported_value"}}'
            ),
        )

    result = materialize_hosted_model_inference_preflight(
        output_path=tmp_path / "receipt.json",
        backend="openai",
        secret_loader=lambda backend: ("inference-key", "fixture"),
        http_post=reject,
    )

    assert result["provider_error"] == {
        "type": "invalid_request_error",
        "code": "unsupported_value",
        "param": "reasoning_effort",
    }
    assert "do not retain me" not in json.dumps(result)


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


def test_nim_backend_uses_served_vision_model_and_inference_key() -> None:
    assert BACKENDS["nvidia_nim"]["model"] == (
        "meta/llama-3.2-11b-vision-instruct"
    )
    assert BACKENDS["nvidia_nim"]["secret_file"] == "nvidia_nim_api_key"


def test_openai_default_is_luna_xhigh_not_legacy_gpt41(tmp_path: Path) -> None:
    observed = {}

    def fake_post(endpoint, headers, payload):
        observed.update(json.loads(payload))
        return (
            200,
            b'{"model":"gpt-5.6-luna","choices":[{"message":{"content":'
            b"\"{\\\"dominant_color\\\":\\\"red\\\"}\"}}]}"
        )

    result = materialize_hosted_model_inference_preflight(
        output_path=tmp_path / "receipt.json",
        backend="openai",
        secret_loader=lambda backend: ("inference-key", "fixture"),
        http_post=fake_post,
    )

    assert result["model"] == "gpt-5.6-luna"
    assert result["reasoning_effort"] == "xhigh"
    assert observed["model"] == "gpt-5.6-luna"
    assert observed["reasoning_effort"] == "xhigh"
