"""Bounded, retrying Vast teardown execution for asynchronous WAM runs."""

from __future__ import annotations

import time
import urllib.error
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def destroy_vast_instance_with_retry(
    *,
    instance_id: int,
    api_key: str,
    api_call: Callable[..., tuple[int, Any]],
    redact_response: Callable[[Any, Sequence[str]], Any],
    attempts: int = 3,
    backoff_seconds: float = 3.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[bool, list[dict[str, Any]]]:
    """Destroy one instance, proving absence or returning continuing-spend risk.

    A 404 is authoritative already-absent evidence. Other HTTP and transport failures
    retry with bounded linear backoff. ``continuing_spend`` is true only when every
    attempt fails.
    """

    teardown_actions: list[dict[str, Any]] = []
    total = max(1, int(attempts))
    for attempt in range(1, total + 1):
        try:
            delete_status, raw_response = api_call(
                method="DELETE",
                path=f"/instances/{instance_id}/",
                api_key=api_key,
                timeout_seconds=30,
            )
            response = (
                dict(raw_response) if isinstance(raw_response, Mapping) else raw_response
            )
            teardown_actions.append(
                {
                    "instance_id": instance_id,
                    "action": "destroy_instance",
                    "attempt": attempt,
                    "http_status_code": delete_status,
                    "response": redact_response(response, [api_key]),
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
        except Exception as exc:  # noqa: BLE001 - retries remain fail-closed
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
            sleeper(min(15.0, backoff_seconds * attempt))
    return True, teardown_actions
