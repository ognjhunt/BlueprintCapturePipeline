#!/usr/bin/env python3
"""Run IAM-authenticated, application-authenticated no-op service canaries."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, cast
from urllib.parse import urlsplit


SECRET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,255}$")
DIGEST_IMAGE_PATTERN = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
MAX_RESPONSE_BYTES = 64 * 1024
SERVICE_KINDS = {
    "sam3": "privacy",
    "vip": "privacy",
    "deepprivacy2": "privacy",
    "video_to_world": "video_to_world",
}


class CanaryError(RuntimeError):
    """Raised when topology or a live authenticated canary is invalid."""


class ResponseLike(Protocol):
    status_code: int
    content: bytes

    def json(self) -> Any: ...


class SessionLike(Protocol):
    def get(self, url: str, **kwargs: Any) -> ResponseLike: ...

    def post(self, url: str, **kwargs: Any) -> ResponseLike: ...


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _terraform_output(outputs: Mapping[str, Any], key: str) -> Any:
    value = outputs.get(key)
    if not isinstance(value, Mapping) or "value" not in value:
        raise CanaryError(f"terraform_output_missing:{key}")
    return value["value"]


def _validate_service_url(value: Any, *, service: str) -> str:
    raw = str(value or "").strip()
    parsed = urlsplit(raw)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".run.app")
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise CanaryError(f"cloud_run_service_url_invalid:{service}")
    return raw.rstrip("/")


def validate_topology(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "blueprint.terraform_topology_evidence.v1":
        raise CanaryError("topology_schema_invalid")
    if payload.get("provider_refresh_zero_drift") is not True:
        raise CanaryError("topology_provider_refresh_not_zero_drift")
    git_sha = str(payload.get("git_sha") or "")
    release_id = str(payload.get("release_id") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        raise CanaryError("topology_git_sha_invalid")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", release_id):
        raise CanaryError("topology_release_id_invalid")
    outputs = _mapping(payload.get("terraform_outputs"))
    raw_services = _mapping(_terraform_output(outputs, "privacy_runner_services"))
    services = {
        service: _validate_service_url(raw_services.get(service), service=service)
        for service in SERVICE_KINDS
    }
    raw_images = _mapping(_terraform_output(outputs, "deployed_image_digests"))
    expected_images = {
        "pipeline",
        "privacy_sam3",
        "privacy_vip",
        "privacy_deepprivacy2",
        "video_to_world",
    }
    if set(raw_images) != expected_images:
        raise CanaryError("deployed_image_digest_set_invalid")
    images = {key: str(value or "") for key, value in raw_images.items()}
    for key, image in images.items():
        if not DIGEST_IMAGE_PATTERN.fullmatch(image):
            raise CanaryError(f"deployed_image_not_digest_pinned:{key}")
    return {
        "services": services,
        "images": images,
        "git_sha": git_sha,
        "release_id": release_id,
        "topology_generated_at": str(payload.get("generated_at") or ""),
    }


def _bounded_json(response: ResponseLike, *, label: str) -> Mapping[str, Any]:
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise CanaryError(f"response_too_large:{label}")
    try:
        payload = response.json()
    except (TypeError, ValueError) as exc:
        raise CanaryError(f"response_not_json:{label}") from exc
    if not isinstance(payload, Mapping):
        raise CanaryError(f"response_not_object:{label}")
    return payload


def execute_canaries(
    *,
    topology: Mapping[str, Any],
    project_id: str,
    privacy_secret_name: str,
    video_secret_name: str,
    session: SessionLike,
    identity_token_for_audience: Callable[[str], str],
    secret_value: Callable[[str, str], str],
) -> dict[str, Any]:
    validated = validate_topology(topology)
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project_id):
        raise CanaryError("project_id_invalid")
    for name in (privacy_secret_name, video_secret_name):
        if not SECRET_NAME_PATTERN.fullmatch(name):
            raise CanaryError("canary_secret_name_invalid")

    secrets = {
        "privacy": secret_value(project_id, privacy_secret_name),
        "video_to_world": secret_value(project_id, video_secret_name),
    }
    if any(len(value) < 32 for value in secrets.values()):
        raise CanaryError("canary_secret_payload_too_short")

    service_results: dict[str, Any] = {}
    for service, service_kind in SERVICE_KINDS.items():
        base_url = validated["services"][service]
        identity_token = identity_token_for_audience(base_url)
        if not identity_token:
            raise CanaryError(f"identity_token_empty:{service}")
        serverless_headers = {
            "X-Serverless-Authorization": f"Bearer {identity_token}",
            "Accept": "application/json",
        }
        health = session.get(
            f"{base_url}/healthz",
            headers=serverless_headers,
            timeout=(5.0, 30.0),
            allow_redirects=False,
        )
        health_payload = _bounded_json(health, label=f"{service}:health")
        if health.status_code != 200 or health_payload.get("status") != "ok":
            raise CanaryError(f"authenticated_health_failed:{service}")
        reported_service = (
            health_payload.get("runner_kind")
            if service_kind == "privacy"
            else health_payload.get("runner")
        )
        if reported_service != service:
            raise CanaryError(f"authenticated_health_identity_mismatch:{service}")

        invalid = session.post(
            f"{base_url}/canary",
            content=b"",
            headers={
                **serverless_headers,
                "Authorization": "Bearer blueprint-invalid-canary-token",
            },
            timeout=(5.0, 30.0),
            allow_redirects=False,
        )
        invalid_payload = _bounded_json(invalid, label=f"{service}:negative")
        if invalid.status_code != 401 or invalid_payload.get("reason") != "unauthorized":
            raise CanaryError(f"application_auth_negative_canary_failed:{service}")

        positive = session.post(
            f"{base_url}/canary",
            content=b"",
            headers={
                **serverless_headers,
                "Authorization": f"Bearer {secrets[service_kind]}",
            },
            timeout=(5.0, 30.0),
            allow_redirects=False,
        )
        positive_payload = _bounded_json(positive, label=f"{service}:positive")
        if (
            positive.status_code != 200
            or positive_payload.get("status") != "ok"
            or positive_payload.get("authentication") != "verified"
            or positive_payload.get("model_execution_performed") is not False
        ):
            raise CanaryError(f"application_auth_positive_canary_failed:{service}")
        service_results[service] = {
            "service_url": base_url,
            "iam_authenticated_health": "passed",
            "invalid_application_token_rejected": True,
            "valid_application_token_canary": "passed",
            "model_execution_performed": False,
        }

    return {
        "schema_version": "blueprint.deployment_service_canaries.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "release_id": validated["release_id"],
        "git_sha": validated["git_sha"],
        "topology_generated_at": validated["topology_generated_at"],
        "services": service_results,
        "deployed_image_digests": validated["images"],
        "secret_payloads_persisted": False,
        "claim_boundary": {
            "service_identity_and_application_auth_proven": True,
            "model_task_success_proven": False,
            "provider_execution_proven": False,
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-evidence", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--privacy-secret-name", required=True)
    parser.add_argument("--video-secret-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        import google.auth
        import requests
        from google.auth.transport.requests import Request
        from google.oauth2 import id_token

        topology = json.loads(args.topology_evidence.read_text(encoding="utf-8"))
        if not isinstance(topology, Mapping):
            raise CanaryError("topology_evidence_not_object")
        credentials, _project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        auth_request = Request()

        def fetch_identity_token(audience: str) -> str:
            return str(id_token.fetch_id_token(auth_request, audience))

        def fetch_secret(project_id: str, secret_name: str) -> str:
            credentials.refresh(auth_request)
            response = requests.get(
                "https://secretmanager.googleapis.com/v1/"
                f"projects/{project_id}/secrets/{secret_name}/versions/latest:access",
                headers={"Authorization": f"Bearer {credentials.token}"},
                timeout=(5.0, 30.0),
                allow_redirects=False,
            )
            if response.status_code != 200 or len(response.content) > MAX_RESPONSE_BYTES:
                raise CanaryError(f"secret_manager_access_failed:{secret_name}")
            response_payload = _mapping(response.json())
            encoded = str(
                _mapping(response_payload.get("payload")).get("data") or ""
            )
            try:
                return base64.b64decode(encoded, validate=True).decode("utf-8").strip()
            except (UnicodeError, ValueError) as exc:
                raise CanaryError(f"secret_manager_payload_invalid:{secret_name}") from exc

        result = execute_canaries(
            topology=topology,
            project_id=args.project_id,
            privacy_secret_name=args.privacy_secret_name,
            video_secret_name=args.video_secret_name,
            session=cast(SessionLike, requests.Session()),
            identity_token_for_audience=fetch_identity_token,
            secret_value=fetch_secret,
        )
        _write_json_atomic(args.output.resolve(), result)
    except (OSError, UnicodeError, ValueError, CanaryError) as exc:
        print(f"[deployment-canary] ERROR {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # Provider/credential transport failure, fail closed.
        print(
            f"[deployment-canary] ERROR external_call_failed:{type(exc).__name__}",
            file=sys.stderr,
        )
        return 1
    print(f"[deployment-canary] passed services={len(result['services'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
