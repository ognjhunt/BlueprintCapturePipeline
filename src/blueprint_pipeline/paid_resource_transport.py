"""Shared file-backed signed-URL resolution for canonical paid allocators."""

from __future__ import annotations

from typing import Sequence

from .wam_async_runner_common import read_sensitive_url_file


def resolve_paid_transport_urls(
    inputs: Sequence[tuple[str, str | None]],
    *,
    blocker_prefix: str,
) -> tuple[dict[str, str], list[str]]:
    resolved: dict[str, str] = {}
    blockers: list[str] = []
    for label, path_value in inputs:
        value, metadata = read_sensitive_url_file(str(path_value or ""), label=label)
        if not value:
            blockers.append(f"{blocker_prefix}_{label}_missing")
        elif not value.startswith("https://"):
            blockers.append(f"{blocker_prefix}_{label}_not_https")
        elif metadata.get("mode_is_0600") is not True:
            blockers.append(f"{blocker_prefix}_{label}_file_permissions_not_0600")
        resolved[label] = value
    return resolved, blockers


__all__ = ["resolve_paid_transport_urls"]
