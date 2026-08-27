"""Fail-closed control-plane capacity checks for returned provider archives."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .wam_async_runner_common import download_url_to_file


DiskUsageProvider = Callable[[Path], Any]
Downloader = Callable[..., Mapping[str, Any]]


def observe_provider_output_disk_capacity(
    *,
    destination_directory: Path,
    required_free_bytes: int,
    phase: str,
    schema_version: str,
    blocker_prefix: str,
    disk_usage_provider: DiskUsageProvider | None = None,
) -> dict[str, Any]:
    """Measure the destination filesystem and refuse unknown capacity."""

    base: dict[str, Any] = {
        "schema_version": schema_version,
        "status": "blocked",
        "phase": phase,
        "measurement_path": str(destination_directory),
        "required_free_bytes": required_free_bytes,
        "observed_free_bytes": None,
        "blockers": [],
    }
    if (
        isinstance(required_free_bytes, bool)
        or not isinstance(required_free_bytes, int)
        or required_free_bytes <= 0
    ):
        base["blockers"] = [f"{blocker_prefix}_disk_requirement_invalid"]
        return base
    try:
        usage = (disk_usage_provider or shutil.disk_usage)(destination_directory)
        free_bytes = getattr(usage, "free", None)
    except (OSError, TypeError, ValueError) as exc:
        base["measurement_error_type"] = type(exc).__name__
        base["blockers"] = [f"{blocker_prefix}_disk_capacity_unavailable"]
        return base
    if isinstance(free_bytes, bool) or not isinstance(free_bytes, int) or free_bytes < 0:
        base["blockers"] = [f"{blocker_prefix}_disk_capacity_unavailable"]
        return base
    base["observed_free_bytes"] = free_bytes
    if free_bytes < required_free_bytes:
        base["blockers"] = [f"{blocker_prefix}_disk_capacity_insufficient"]
        return base
    base["status"] = "ready"
    return base


def download_provider_output_with_capacity_guard(
    *,
    url: str,
    output_path: Path,
    minimum_free_bytes: int,
    disk_usage_provider: DiskUsageProvider | None = None,
    downloader: Downloader | None = None,
) -> dict[str, Any]:
    """Re-stat capacity immediately before performing the provider-output GET."""

    if minimum_free_bytes:
        capacity = observe_provider_output_disk_capacity(
            destination_directory=output_path.parent,
            required_free_bytes=minimum_free_bytes,
            phase="before_provider_output_get",
            schema_version="vast_provider_output_disk_capacity.v1",
            blocker_prefix="provider_output",
            disk_usage_provider=disk_usage_provider,
        )
        if capacity["status"] != "ready":
            return {
                "status": "blocked",
                "download_attempted": False,
                "disk_capacity": capacity,
                "blockers": list(capacity["blockers"]),
            }
    else:
        capacity = {
            "schema_version": "vast_provider_output_disk_capacity.v1",
            "status": "not_required",
            "phase": "before_provider_output_get",
            "measurement_path": str(output_path.parent),
            "required_free_bytes": 0,
            "observed_free_bytes": None,
            "blockers": [],
        }
    transfer = dict(
        (downloader or download_url_to_file)(
            url=url,
            output_path=output_path,
            user_agent="BlueprintVastProviderAdapter/1.0",
            timeout_seconds=60,
        )
    )
    transfer.update(download_attempted=True, disk_capacity=capacity)
    return transfer


__all__ = [
    "DiskUsageProvider",
    "download_provider_output_with_capacity_guard",
    "observe_provider_output_disk_capacity",
]
