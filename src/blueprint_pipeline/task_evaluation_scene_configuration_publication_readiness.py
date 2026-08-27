"""No-spend readiness gate for scene-configuration result publication."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from .task_evaluation_configured_scene_object_store import (
    TaskEvaluationConfiguredSceneObjectStoreError,
    validate_configured_scene_object_store_configuration,
)
from .task_evaluation_launch_webapp_sync import (
    PipelineSyncTokenError,
    load_pipeline_sync_token,
)
from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url


PUBLICATION_READINESS_URL_ENV = "PIPELINE_TASK_EVALUATION_LAUNCH_PUBLICATION_READINESS_URL"
READINESS_REQUEST_SCHEMA_VERSION = "task_evaluation_launch_publication_readiness_request.v1"
READINESS_RECEIPT_SCHEMA_VERSION = "task_evaluation_launch_publication_readiness_receipt.v1"
TERMINAL_RECEIPT_SCHEMA_VERSION = "task_evaluation_launch_receipt.v1"
WEB_SYNC_RECEIPT_SCHEMA_VERSION = "task_evaluation_launch_web_sync_receipt.v1"
CONFIGURED_SCENE_OFFERING_SCHEMA_VERSION = "task_evaluation_configured_scene_offering.v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class SceneConfigurationPublicationReadinessError(RuntimeError):
    """A no-spend publication prerequisite is not satisfied."""


def _validated_binding(
    *, launch_id: str, run_id: str, request_digest: str, team_namespace: str
) -> dict[str, str]:
    values = {
        "launch_id": str(launch_id or "").strip(),
        "run_id": str(run_id or "").strip(),
        "request_digest": str(request_digest or "").strip(),
        "team_namespace": str(team_namespace or "").strip(),
    }
    if (
        any(
            _IDENTIFIER.fullmatch(values[field]) is None
            for field in ("launch_id", "run_id", "team_namespace")
        )
        or _DIGEST.fullmatch(values["request_digest"]) is None
    ):
        raise SceneConfigurationPublicationReadinessError(
            "scene_configuration_publication_binding_invalid"
        )
    return values


def verify_scene_configuration_publication_readiness(
    *,
    launch_id: str,
    run_id: str,
    request_digest: str,
    team_namespace: str,
    endpoint_url: str | None = None,
    token_file_path: str | Path | None = None,
    timeout_seconds: float = 10.0,
    object_store_validator: Callable[[], Mapping[str, Any]] = (
        validate_configured_scene_object_store_configuration
    ),
    opener: Callable[..., Any] = urllib_request.urlopen,
) -> dict[str, Any]:
    """Prove local object-store setup and exact read-only WebApp readiness.

    The returned receipt is not spend authority. Remote object-store authority
    is intentionally left unclaimed until the terminal publisher performs its
    content-addressed upload and full-byte readback.
    """

    binding = _validated_binding(
        launch_id=launch_id,
        run_id=run_id,
        request_digest=request_digest,
        team_namespace=team_namespace,
    )
    try:
        object_store = dict(object_store_validator())
    except TaskEvaluationConfiguredSceneObjectStoreError as exc:
        code = str(exc)
        if not code.startswith("configured_scene_object_store_"):
            code = "configured_scene_object_store_configuration_invalid"
        raise SceneConfigurationPublicationReadinessError(code) from exc
    if (
        object_store.get("status") != "locally_configured"
        or object_store.get("client_constructed") is not True
        or object_store.get("provider_mutation_performed") is not False
        or object_store.get("remote_bucket_authority_verified") is not False
    ):
        raise SceneConfigurationPublicationReadinessError(
            "configured_scene_object_store_configuration_invalid"
        )
    resolved_url = str(endpoint_url or os.getenv(PUBLICATION_READINESS_URL_ENV) or "").strip()
    if not resolved_url:
        raise SceneConfigurationPublicationReadinessError(
            "scene_configuration_publication_readiness_url_missing"
        )
    try:
        url = validated_https_sync_url(resolved_url)
    except ValueError as exc:
        raise SceneConfigurationPublicationReadinessError(
            "scene_configuration_publication_readiness_url_invalid"
        ) from exc
    try:
        token = load_pipeline_sync_token(token_file_path=token_file_path, require_file=True)
    except PipelineSyncTokenError as exc:
        raise SceneConfigurationPublicationReadinessError(str(exc)) from exc
    payload = {
        "schema_version": READINESS_REQUEST_SCHEMA_VERSION,
        **binding,
        "expected_terminal_receipt_schema_version": (TERMINAL_RECEIPT_SCHEMA_VERSION),
        "expected_web_sync_receipt_schema_version": (WEB_SYNC_RECEIPT_SCHEMA_VERSION),
        "expected_configured_scene_offering_schema_version": (
            CONFIGURED_SCENE_OFFERING_SCHEMA_VERSION
        ),
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    outbound = urllib_request.Request(
        url,
        data=body,
        headers=_pipeline_sync_headers(token, body),
        method="POST",
    )
    try:
        with opener(outbound, timeout=max(0.1, timeout_seconds)) as response:
            raw = response.read(64 * 1024 + 1)
    except urllib_error.HTTPError as exc:
        raise SceneConfigurationPublicationReadinessError(
            f"scene_configuration_publication_readiness_http_error:{exc.code}"
        ) from exc
    except urllib_error.URLError as exc:
        raise SceneConfigurationPublicationReadinessError(
            "scene_configuration_publication_readiness_unreachable"
        ) from exc
    except (OSError, TimeoutError, ValueError) as exc:
        raise SceneConfigurationPublicationReadinessError(
            "scene_configuration_publication_readiness_unreachable"
        ) from exc
    if len(raw) > 64 * 1024:
        raise SceneConfigurationPublicationReadinessError(
            "scene_configuration_publication_readiness_response_invalid"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SceneConfigurationPublicationReadinessError(
            "scene_configuration_publication_readiness_response_invalid"
        ) from exc
    expected = {
        "schema_version": READINESS_RECEIPT_SCHEMA_VERSION,
        "status": "ready",
        **binding,
        "terminal_receipt_schema_version": TERMINAL_RECEIPT_SCHEMA_VERSION,
        "web_sync_receipt_schema_version": WEB_SYNC_RECEIPT_SCHEMA_VERSION,
        "configured_scene_offering_schema_version": (CONFIGURED_SCENE_OFFERING_SCHEMA_VERSION),
        "launch_record_read_succeeded": True,
        "team_namespace_binding_passed": True,
        "firestore_mutation_performed": False,
    }
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise SceneConfigurationPublicationReadinessError(
            "scene_configuration_publication_readiness_response_mismatch"
        )
    return {
        **expected,
        "object_store_configuration": object_store,
        "remote_object_store_authority_verified": False,
        "spend_authority_granted": False,
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
    }


def scene_configuration_publication_readiness_decision(
    *,
    request: Mapping[str, Any],
    profile: Mapping[str, Any],
    live_requested: bool,
    existing_blockers: bool,
    probe: Callable[..., Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Evaluate the dispatcher prerequisite without owning launch mutation.

    The dispatcher remains responsible for recording the returned receipt and
    blocker before immutable staging. Keeping the transport/error policy here
    prevents the launcher's paid-mutation spine from absorbing another service
    client while preserving the exact same fail-closed ordering.
    """

    not_applicable = {
        "schema_version": "task_evaluation_launch_publication_preflight.v1",
        "status": "not_applicable",
        "provider_mutation_performed": False,
        "spend_authority_granted": False,
        "raw_secret_values_recorded": False,
    }
    task_evaluation_run = profile.get("task_evaluation_run")
    if not isinstance(task_evaluation_run, Mapping):
        task_evaluation_run = {}
    if (
        existing_blockers
        or not live_requested
        or task_evaluation_run.get("run_mode") != "scene_configuration"
    ):
        return not_applicable, None

    try:
        receipt = dict(
            (probe or verify_scene_configuration_publication_readiness)(
                launch_id=str(request.get("launch_id") or ""),
                run_id=str(request.get("run_id") or ""),
                request_digest=str(request.get("request_digest") or ""),
                team_namespace=str(task_evaluation_run.get("team_namespace") or ""),
            )
        )
        if (
            receipt.get("status") != "ready"
            or receipt.get("provider_mutation_performed") is not False
            or receipt.get("spend_authority_granted") is not False
        ):
            raise SceneConfigurationPublicationReadinessError(
                "scene_configuration_publication_readiness_receipt_invalid"
            )
        return receipt, None
    except SceneConfigurationPublicationReadinessError as exc:
        code = str(exc)
        if not (
            code.startswith("scene_configuration_publication_")
            or code.startswith("configured_scene_object_store_")
            or code.startswith("pipeline_sync_token_")
        ):
            code = "scene_configuration_publication_readiness_internal_error"
    except Exception:  # fail closed without recording transport/client detail
        code = "scene_configuration_publication_readiness_internal_error"
    return {
        "schema_version": "task_evaluation_launch_publication_preflight.v1",
        "status": "blocked",
        "blockers": [code],
        "provider_mutation_performed": False,
        "spend_authority_granted": False,
        "raw_secret_values_recorded": False,
    }, code


__all__ = [
    "CONFIGURED_SCENE_OFFERING_SCHEMA_VERSION",
    "PUBLICATION_READINESS_URL_ENV",
    "READINESS_RECEIPT_SCHEMA_VERSION",
    "READINESS_REQUEST_SCHEMA_VERSION",
    "SceneConfigurationPublicationReadinessError",
    "TERMINAL_RECEIPT_SCHEMA_VERSION",
    "WEB_SYNC_RECEIPT_SCHEMA_VERSION",
    "scene_configuration_publication_readiness_decision",
    "verify_scene_configuration_publication_readiness",
]
