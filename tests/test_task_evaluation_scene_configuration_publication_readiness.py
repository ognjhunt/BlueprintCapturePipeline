from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import (
    task_evaluation_scene_configuration_publication_readiness as readiness,
)


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_args):  # type: ignore[no-untyped-def]
        return False

    def read(self, _maximum: int) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _binding() -> dict[str, str]:
    return {
        "launch_id": "launch-001",
        "run_id": "run-001",
        "request_digest": "sha256:" + "a" * 64,
        "team_namespace": "robot-team-001",
    }


def _object_store() -> dict[str, object]:
    return {
        "status": "locally_configured",
        "client_constructed": True,
        "remote_bucket_authority_verified": False,
        "provider_mutation_performed": False,
    }


def _webapp_receipt() -> dict[str, object]:
    return {
        "schema_version": readiness.READINESS_RECEIPT_SCHEMA_VERSION,
        "status": "ready",
        **_binding(),
        "terminal_receipt_schema_version": (
            readiness.TERMINAL_RECEIPT_SCHEMA_VERSION
        ),
        "web_sync_receipt_schema_version": (
            readiness.WEB_SYNC_RECEIPT_SCHEMA_VERSION
        ),
        "configured_scene_offering_schema_version": (
            readiness.CONFIGURED_SCENE_OFFERING_SCHEMA_VERSION
        ),
        "launch_record_read_succeeded": True,
        "team_namespace_binding_passed": True,
        "firestore_mutation_performed": False,
    }


def _token_file(tmp_path: Path) -> Path:
    path = tmp_path / "pipeline-sync-token"
    path.write_text("test-only-token\n", encoding="utf-8")
    path.chmod(0o640)
    return path


def test_publication_readiness_binds_webapp_schemas_without_remote_store_mutation(
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def open_request(request, **_kwargs):  # type: ignore[no-untyped-def]
        observed["method"] = request.method
        observed["payload"] = json.loads(request.data)
        return _Response(_webapp_receipt())

    result = readiness.verify_scene_configuration_publication_readiness(
        **_binding(),
        endpoint_url=(
            "https://webapp.test/api/internal/pipeline/"
            "task-evaluation-launch-publication-readiness"
        ),
        token_file_path=_token_file(tmp_path),
        object_store_validator=_object_store,
        opener=open_request,
    )

    assert observed == {
        "method": "POST",
        "payload": {
            "schema_version": readiness.READINESS_REQUEST_SCHEMA_VERSION,
            **_binding(),
            "expected_terminal_receipt_schema_version": (
                readiness.TERMINAL_RECEIPT_SCHEMA_VERSION
            ),
            "expected_web_sync_receipt_schema_version": (
                readiness.WEB_SYNC_RECEIPT_SCHEMA_VERSION
            ),
            "expected_configured_scene_offering_schema_version": (
                readiness.CONFIGURED_SCENE_OFFERING_SCHEMA_VERSION
            ),
        },
    }
    assert result["status"] == "ready"
    assert result["remote_object_store_authority_verified"] is False
    assert result["spend_authority_granted"] is False
    assert result["provider_mutation_performed"] is False
    assert result["raw_secret_values_recorded"] is False


def test_publication_readiness_refuses_team_or_schema_drift(
    tmp_path: Path,
) -> None:
    response = _webapp_receipt()
    response["team_namespace"] = "other-team"
    with pytest.raises(
        readiness.SceneConfigurationPublicationReadinessError,
        match="scene_configuration_publication_readiness_response_mismatch",
    ):
        readiness.verify_scene_configuration_publication_readiness(
            **_binding(),
            endpoint_url=(
                "https://webapp.test/api/internal/pipeline/"
                "task-evaluation-launch-publication-readiness"
            ),
            token_file_path=_token_file(tmp_path),
            object_store_validator=_object_store,
            opener=lambda *_args, **_kwargs: _Response(response),
        )


def test_publication_readiness_refuses_nonlocal_object_store_claim(
    tmp_path: Path,
) -> None:
    invalid = {**_object_store(), "remote_bucket_authority_verified": True}
    with pytest.raises(
        readiness.SceneConfigurationPublicationReadinessError,
        match="configured_scene_object_store_configuration_invalid",
    ):
        readiness.verify_scene_configuration_publication_readiness(
            **_binding(),
            endpoint_url=(
                "https://webapp.test/api/internal/pipeline/"
                "task-evaluation-launch-publication-readiness"
            ),
            token_file_path=_token_file(tmp_path),
            object_store_validator=lambda: invalid,
            opener=lambda *_args, **_kwargs: pytest.fail("WebApp called"),
        )
