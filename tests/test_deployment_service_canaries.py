from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from scripts.run_deployment_service_canaries import (
    CanaryError,
    execute_canaries,
    validate_topology,
)


IMAGE = f"gcr.io/blueprint/image@sha256:{'a' * 64}"


def _topology() -> dict[str, object]:
    return {
        "schema_version": "blueprint.terraform_topology_evidence.v1",
        "generated_at": "2026-07-09T00:00:00Z",
        "release_id": "release-a",
        "git_sha": "b" * 40,
        "provider_refresh_zero_drift": True,
        "terraform_outputs": {
            "privacy_runner_services": {
                "value": {
                    "sam3": "https://sam3-abc-uc.a.run.app",
                    "vip": "https://vip-abc-uc.a.run.app",
                    "deepprivacy2": "https://deepprivacy-abc-uc.a.run.app",
                    "video_to_world": "https://video-world-abc-uc.a.run.app",
                }
            },
            "deployed_image_digests": {
                "value": {
                    "pipeline": IMAGE,
                    "privacy_sam3": IMAGE,
                    "privacy_vip": IMAGE,
                    "privacy_deepprivacy2": IMAGE,
                    "video_to_world": IMAGE,
                }
            },
        },
    }


@dataclass
class _Response:
    status_code: int
    payload: dict[str, object]

    @property
    def content(self) -> bytes:
        return b"{}"

    def json(self) -> dict[str, object]:
        return dict(self.payload)


class _Session:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append(("GET", url, kwargs))
        if "video-world" in url:
            return _Response(200, {"status": "ok", "runner": "video_to_world"})
        if "deepprivacy" in url:
            runner_kind = "deepprivacy2"
        elif "sam3" in url:
            runner_kind = "sam3"
        else:
            runner_kind = "vip"
        return _Response(200, {"status": "ok", "runner_kind": runner_kind})

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append(("POST", url, kwargs))
        authorization = kwargs["headers"]["Authorization"]
        if authorization == "Bearer blueprint-invalid-canary-token":
            return _Response(401, {"status": "failed", "reason": "unauthorized"})
        return _Response(
            200,
            {
                "status": "ok",
                "authentication": "verified",
                "model_execution_performed": False,
            },
        )


def test_deployment_canary_requires_digest_topology_and_cloud_run_urls() -> None:
    validated = validate_topology(_topology())
    assert set(validated["services"]) == {
        "sam3",
        "vip",
        "deepprivacy2",
        "video_to_world",
    }

    topology = _topology()
    topology["terraform_outputs"]["privacy_runner_services"]["value"][  # type: ignore[index]
        "sam3"
    ] = "https://attacker.example/"
    with pytest.raises(CanaryError, match="cloud_run_service_url_invalid"):
        validate_topology(topology)

    topology = _topology()
    topology["terraform_outputs"]["deployed_image_digests"]["value"][  # type: ignore[index]
        "pipeline"
    ] = "gcr.io/blueprint/image:latest"
    with pytest.raises(CanaryError, match="deployed_image_not_digest_pinned"):
        validate_topology(topology)


def test_deployment_canary_separates_iam_and_application_auth_without_model_work() -> None:
    session = _Session()
    secrets_requested: list[tuple[str, str]] = []

    def secret_value(project: str, name: str) -> str:
        secrets_requested.append((project, name))
        return "s" * 40

    result = execute_canaries(
        topology=_topology(),
        project_id="blueprint-8c1ca",
        privacy_secret_name="privacy-runner",
        video_secret_name="video-runner",
        session=session,
        identity_token_for_audience=lambda audience: f"identity-for:{audience}",
        secret_value=secret_value,
    )

    assert result["status"] == "passed"
    assert result["release_id"] == "release-a"
    assert result["git_sha"] == "b" * 40
    assert result["secret_payloads_persisted"] is False
    assert result["claim_boundary"]["model_task_success_proven"] is False
    assert secrets_requested == [
        ("blueprint-8c1ca", "privacy-runner"),
        ("blueprint-8c1ca", "video-runner"),
    ]
    assert len(session.calls) == 12
    for method, _url, kwargs in session.calls:
        assert kwargs["allow_redirects"] is False
        assert kwargs["timeout"] == (5.0, 30.0)
        assert kwargs["headers"]["X-Serverless-Authorization"].startswith("Bearer ")
        if method == "GET":
            assert "Authorization" not in kwargs["headers"]


def test_deployment_canary_rejects_short_secret_before_network() -> None:
    session = _Session()
    with pytest.raises(CanaryError, match="canary_secret_payload_too_short"):
        execute_canaries(
            topology=_topology(),
            project_id="blueprint-8c1ca",
            privacy_secret_name="privacy-runner",
            video_secret_name="video-runner",
            session=session,
            identity_token_for_audience=lambda _audience: "identity",
            secret_value=lambda _project, _name: "short",
        )
    assert session.calls == []
