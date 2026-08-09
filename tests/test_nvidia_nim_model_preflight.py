from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.nvidia_nim_model_preflight import (
    DEFAULT_MODEL,
    materialize_nvidia_nim_model_preflight,
)


def test_nim_preflight_qualifies_exact_model_without_inference_or_secret(
    tmp_path: Path,
) -> None:
    observed: dict = {}

    def fake_get(endpoint, headers):
        observed["endpoint"] = endpoint
        observed["headers"] = headers
        return 200, json.dumps(
            {"data": [{"id": "other/model"}, {"id": DEFAULT_MODEL}]}
        ).encode()

    output = tmp_path / "preflight.json"
    result = materialize_nvidia_nim_model_preflight(
        output_path=output,
        generated_at="2026-08-08T00:00:00+00:00",
        secret_loader=lambda: ("super-secret", "fixture"),
        http_get=fake_get,
    )

    assert result["status"] == "qualified"
    assert result["credential_validated"] is True
    assert result["required_model_present"] is True
    assert result["paid_inference_performed"] is False
    assert result["provider_mutations_performed"] == 0
    assert observed["headers"]["Authorization"] == "Bearer super-secret"
    assert "super-secret" not in output.read_text()


def test_nim_preflight_abstains_when_model_is_not_in_catalog(tmp_path: Path) -> None:
    result = materialize_nvidia_nim_model_preflight(
        output_path=tmp_path / "preflight.json",
        secret_loader=lambda: ("secret", "fixture"),
        http_get=lambda endpoint, headers: (200, b'{"data":[{"id":"other"}]}'),
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["nvidia_nim_required_model_unavailable"]
    assert result["paid_inference_performed"] is False


def test_nim_preflight_abstains_before_network_without_key(tmp_path: Path) -> None:
    called = False

    def forbidden_get(endpoint, headers):
        nonlocal called
        called = True
        raise AssertionError("network must not be called")

    result = materialize_nvidia_nim_model_preflight(
        output_path=tmp_path / "preflight.json",
        secret_loader=lambda: ("", "missing"),
        http_get=forbidden_get,
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["nvidia_nim_api_key_missing"]
    assert called is False
