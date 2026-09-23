"""Bounded, sanitized live log sampling for website MapAnything workers."""

from __future__ import annotations

import re


def worker_log_diagnostic(instance_id: str) -> dict:
    from .gpu_render_providers import _read_secret
    from .vast_provider_adapter import _api_json, _fetch_text

    if not instance_id.isdecimal() or int(instance_id) <= 0:
        return {"status": "unavailable", "reason": "instance_invalid"}
    key = _read_secret("vast_api_key")
    if not key:
        return {"status": "unavailable", "reason": "key_unavailable"}
    try:
        status, response = _api_json(
            method="PUT", path=f"/instances/request_logs/{instance_id}/",
            api_key=key, payload={"tail": "150", "daemon_logs": "false"},
            timeout_seconds=15,
        )
        url = response.get("result_url") or response.get("temp_download_url")
        if status != 200 or not isinstance(url, str) or not url:
            return {"status": "unavailable", "reason": "log_transport_unavailable"}
        log = _fetch_text(url, timeout_seconds=15)
    except Exception:  # noqa: BLE001 - never retain provider URLs or raw errors.
        return {"status": "unavailable", "reason": "log_transport_unavailable"}
    markers = []
    for line in log.splitlines():
        match = re.search(
            r"BLUEPRINT_WEBSITE_MAPANYTHING_BOOTSTRAP_FAILURE(?:_REPORT_FAILED)?:"
            r"[a-z][a-z0-9_]{2,80}(?::[a-z][a-z0-9_]{2,100})?", line,
        )
        if match and match.group() not in markers:
            markers.append(match.group())
    exception_types = sorted(set(re.findall(
        r"(?m)^\s*(CalledProcessError|FileExistsError|ValueError|RuntimeError|TimeoutError|OSError):", log,
    )))
    return {"status": "observed", "typed_markers": markers[:8],
            "exception_types": exception_types[:8], "log_bytes_observed": len(log.encode())}
