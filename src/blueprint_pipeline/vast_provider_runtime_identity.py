"""Parse retained-worker runtime identity from bounded Vast evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def runtime_dependency_cache_ready(
    *, startup_log_text: str, isaac_smoke: Mapping[str, Any]
) -> bool:
    container_log = isaac_smoke.get("container_log_result")
    durable_lines = (
        container_log.get("observed_blueprint_marker_lines")
        if isinstance(container_log, Mapping)
        else []
    )
    durable_lines = durable_lines if isinstance(durable_lines, list) else []
    markers = (
        "BLUEPRINT_VAST_RUNTIME_DEPENDENCY_CACHE_HIT:",
        "BLUEPRINT_VAST_RUNTIME_DEPENDENCY_CACHE_FILLED:",
    )
    return any(
        marker in startup_log_text
        or any(isinstance(line, str) and marker in line for line in durable_lines)
        for marker in markers
    )


def provider_remote_work_dir(startup_log_text: str) -> str | None:
    matches = re.findall(
        r"(?m)^BLUEPRINT_VAST_WORK_DIR:(/[^\r\n]+)$", startup_log_text
    )
    unique = sorted(set(matches))
    if len(unique) != 1 or unique[0] not in {
        "/workspace",
        "/tmp/blueprint_vast_work",
    }:
        return None
    return unique[0]


__all__ = ["provider_remote_work_dir", "runtime_dependency_cache_ready"]
