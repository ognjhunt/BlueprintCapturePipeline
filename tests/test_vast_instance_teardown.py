from __future__ import annotations

import urllib.error

from blueprint_pipeline.vast_instance_teardown import (
    build_vast_teardown_manifest,
    destroy_vast_instance_with_retry,
)


def _redact(value: object, secrets: list[str]) -> object:
    return "<redacted>" if any(secret and secret in str(value) for secret in secrets) else value


def test_destroy_vast_instance_retries_then_redacts_success() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def api_request(**_kwargs: object) -> tuple[int, object]:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return 200, {"credential": "secret-key"}

    continuing_spend, actions = destroy_vast_instance_with_retry(
        instance_id=42,
        api_key="secret-key",
        api_request=api_request,
        redact_runtime_value=_redact,
        sleep=sleeps.append,
        attempts=3,
        backoff_seconds=2,
    )

    assert continuing_spend is False
    assert [action["status"] for action in actions] == ["failed", "completed"]
    assert actions[-1]["response"] == "<redacted>"
    assert sleeps == [2]


def test_destroy_vast_instance_treats_404_as_absent() -> None:
    def api_request(**_kwargs: object) -> tuple[int, object]:
        raise urllib.error.HTTPError("https://vast.example", 404, "missing", {}, None)

    continuing_spend, actions = destroy_vast_instance_with_retry(
        instance_id=7,
        api_key="secret",
        api_request=api_request,
        redact_runtime_value=_redact,
        sleep=lambda _seconds: None,
    )

    assert continuing_spend is False
    assert actions == [
        {
            "instance_id": 7,
            "action": "destroy_instance",
            "attempt": 1,
            "http_status_code": 404,
            "status": "completed",
            "reason": "instance_already_absent",
        }
    ]


def test_destroy_vast_instance_exhaustion_preserves_spend_warning() -> None:
    sleeps: list[float] = []

    def api_request(**_kwargs: object) -> tuple[int, object]:
        raise OSError("offline")

    continuing_spend, actions = destroy_vast_instance_with_retry(
        instance_id=9,
        api_key="secret",
        api_request=api_request,
        redact_runtime_value=_redact,
        sleep=sleeps.append,
        attempts=3,
        backoff_seconds=8,
    )

    assert continuing_spend is True
    assert len(actions) == 3
    assert all(action["error_type"] == "OSError" for action in actions)
    assert sleeps == [8, 15.0]


def test_teardown_manifest_never_promotes_continuing_spend_to_completed() -> None:
    manifest = build_vast_teardown_manifest(
        schema_version="vast_teardown_manifest.v1",
        generated_at="2026-07-22T00:00:00+00:00",
        instance_id=9,
        status="blocked",
        teardown_actions=[{"status": "failed"}],
        continuing_spend=True,
        zero_continuing_spend_scope="manual verification required",
        extra_fields={"blockers": ["vast_instance_destroy_failed"]},
    )

    assert manifest["runner_gpu_teardown_completed"] is False
    assert manifest["continuing_spend_from_this_run"] is True
    assert manifest["blockers"] == ["vast_instance_destroy_failed"]
    assert manifest["raw_secret_values_recorded"] is False
