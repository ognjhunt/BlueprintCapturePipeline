"""Fail-closed Vast instance teardown primitives shared by paid runners."""

from __future__ import annotations

import urllib.error
from collections.abc import Callable, Mapping, Sequence
from typing import Any


ApiRequest = Callable[..., tuple[int, Any]]
RuntimeRedactor = Callable[[Any, Sequence[str]], Any]
Sleeper = Callable[[float], None]


def destroy_vast_instance_with_retry(
    *,
    instance_id: int,
    api_key: str,
    api_request: ApiRequest,
    redact_runtime_value: RuntimeRedactor,
    sleep: Sleeper,
    attempts: int = 3,
    backoff_seconds: float = 3.0,
) -> tuple[bool, list[dict[str, Any]]]:
    """Destroy one Vast instance and report whether billing may continue.

    A 404 proves the instance is already absent. Every other failure is retried
    with bounded linear backoff. The returned boolean is deliberately named by
    meaning: it is true only when every destroy attempt failed.
    """

    teardown_actions: list[dict[str, Any]] = []
    total = max(1, int(attempts))
    for attempt in range(1, total + 1):
        try:
            delete_status, delete_response = api_request(
                method="DELETE",
                path=f"/instances/{instance_id}/",
                api_key=api_key,
                timeout_seconds=30,
            )
            teardown_actions.append(
                {
                    "instance_id": instance_id,
                    "action": "destroy_instance",
                    "attempt": attempt,
                    "http_status_code": delete_status,
                    "response": redact_runtime_value(delete_response, [api_key]),
                    "status": "completed",
                }
            )
            return False, teardown_actions
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                teardown_actions.append(
                    {
                        "instance_id": instance_id,
                        "action": "destroy_instance",
                        "attempt": attempt,
                        "http_status_code": exc.code,
                        "status": "completed",
                        "reason": "instance_already_absent",
                    }
                )
                return False, teardown_actions
            teardown_actions.append(
                {
                    "instance_id": instance_id,
                    "action": "destroy_instance",
                    "attempt": attempt,
                    "http_status_code": exc.code,
                    "status": "failed",
                }
            )
        except Exception as exc:
            teardown_actions.append(
                {
                    "instance_id": instance_id,
                    "action": "destroy_instance",
                    "attempt": attempt,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
            )
        if attempt < total:
            sleep(min(15.0, backoff_seconds * attempt))
    return True, teardown_actions


def build_vast_teardown_manifest(
    *,
    schema_version: str,
    generated_at: str,
    instance_id: int,
    status: str,
    teardown_actions: Sequence[Mapping[str, Any]],
    continuing_spend: bool,
    zero_continuing_spend_scope: str | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared teardown proof without weakening provider-specific fields."""

    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "generated_at": generated_at,
        "status": status,
        "vast_instance_ids": [instance_id] if instance_id else [],
        "teardown_actions_performed": [dict(action) for action in teardown_actions],
        "runner_gpu_teardown_completed": not continuing_spend,
        "continuing_spend_from_this_run": continuing_spend,
    }
    if zero_continuing_spend_scope is not None:
        manifest["zero_continuing_spend_scope"] = zero_continuing_spend_scope
    if extra_fields:
        manifest.update(dict(extra_fields))
    manifest["raw_secret_values_recorded"] = False
    return manifest
