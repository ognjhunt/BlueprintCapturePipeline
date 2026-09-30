"""Existing Vast REST transport shared by adapters and read-only account observations."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VAST_API_BASE = "https://console.vast.ai/api/v0"


def _api_json(
    *,
    method: str,
    path: str,
    api_key: str,
    payload: Mapping[str, Any] | None = None,
    timeout_seconds: int = 30,
) -> tuple[int, dict[str, Any]]:
    url = (
        path
        if path.startswith("http://") or path.startswith("https://")  # noqa: PIE810 - preserve existing compatibility/body semantics
        else f"{VAST_API_BASE}{path}"
    )
    from .provider_transport import provider_json_request

    read_options = {}
    if method.upper() == "GET" and path in {"/instances", "/instances/"}:
        from .vast_inventory_read_retry import inventory_read_retry
        read_options["read_retry"] = inventory_read_retry()

    return provider_json_request(
        url=url,
        method=method,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        body_json=payload,
        timeout_seconds=timeout_seconds,
        **read_options,
    )
