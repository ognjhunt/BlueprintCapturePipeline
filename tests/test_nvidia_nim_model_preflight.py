from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.nvidia_nim_model_preflight import (
    DEFAULT_MODEL,
    materialize_nvidia_nim_model_preflight,
)


def test_nim_preflight_qualifies_exact_model_and_never_records_the_secret(
    tmp_path: Path,
) -> None:
    """Qualification now requires a one-token call, not just a catalog listing.

    This test previously asserted the gate qualified *without* inference. That
    contract is what let a key entitled only to the catalog reach a paid
    provider, where the Joint Agent died on its first model call.
    """

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
        http_post=lambda endpoint, headers, payload: (
            200,
            b'{"choices":[{"message":{"content":"ok"}}]}',
        ),
    )

    assert result["status"] == "qualified"
    assert result["credential_validated"] is True
    assert result["required_model_present"] is True
    assert result["inference_probe"]["authorized"] is True
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


def test_preflight_proves_the_key_can_actually_infer(tmp_path) -> None:
    """The catalog endpoint answers 200 to an invalid key.

    That is how a key with no inference entitlement reached a paid provider:
    the preflight asked the one NIM endpoint that does not enforce auth, saw
    200, and reported qualified. The Joint Agent run then died at its first
    model call with 'Model authentication failed'. The gate now issues a
    minimal authenticated inference request, which is the thing the agent
    actually needs to work.
    """

    calls: list[tuple[str, str]] = []

    def catalog(endpoint, headers):
        calls.append(("GET", endpoint))
        return 200, json.dumps({"data": [{"id": "google/gemma-4-31b-it"}]}).encode()

    def unauthorized_inference(endpoint, headers, payload):
        calls.append(("POST", endpoint))
        return 401, b'{"status":401,"title":"Unauthorized"}'

    receipt = materialize_nvidia_nim_model_preflight(
        output_path=tmp_path / "preflight.json",
        secret_loader=lambda: ("key", "test"),
        http_get=catalog,
        http_post=unauthorized_inference,
    )

    assert receipt["status"] == "blocked"
    assert "nvidia_nim_inference_unauthorized" in receipt["blockers"]
    assert ("POST", "https://integrate.api.nvidia.com/v1/chat/completions") in calls
    assert receipt["inference_probe"]["http_status"] == 401


def test_preflight_qualifies_when_inference_is_authorized(tmp_path) -> None:
    receipt = materialize_nvidia_nim_model_preflight(
        output_path=tmp_path / "preflight.json",
        secret_loader=lambda: ("key", "test"),
        http_get=lambda e, h: (
            200,
            json.dumps({"data": [{"id": "google/gemma-4-31b-it"}]}).encode(),
        ),
        http_post=lambda e, h, p: (200, b'{"choices":[{"message":{"content":"ok"}}]}'),
    )

    assert receipt["status"] == "qualified"
    assert receipt["blockers"] == []
    assert receipt["inference_probe"]["authorized"] is True
    assert receipt["inference_probe"]["max_tokens"] == 1
