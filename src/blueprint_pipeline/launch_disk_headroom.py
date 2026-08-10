"""Refuse a launch that would fill the disk staging its own bundle.

Twice in one session the laptop ran out of space partway through a run. Both
times the instance had already been created, so the cost was not a failed
launch - it was an orphan billing on a provider while the harness lay dead,
unable even to write the file that would have recorded the obligation.

Checking before the create is free. Dying after it costs whatever the instance
bills until somebody looks at a dashboard.

The margin matters as much as the bundle size. A run writes container logs, a
result, manifests and an output zip after staging, so sizing the check to the
bundle alone passes a launch that then dies writing its own receipt - the same
failure with extra steps.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence


LAUNCH_DISK_HEADROOM_SCHEMA_VERSION = "launch_disk_headroom.v1"
# Room for the run's own output after the bundle is staged: logs, receipts,
# manifests and the downloaded output zip.
DEFAULT_SAFETY_MARGIN_BYTES = 1024 * 1024 * 1024


class LaunchDiskHeadroomError(ValueError):
    """Stable, sorted disk-headroom failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def require_launch_disk_headroom(
    *,
    path: str | Path,
    required_bytes: int,
    safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES,
    free_bytes_override: int | None = None,
) -> dict[str, Any]:
    """Fail before the create call if the run cannot fit on disk."""

    target = Path(path).expanduser()
    if free_bytes_override is None:
        stat = os.statvfs(target if target.exists() else target.parent)
        free = int(stat.f_bavail) * int(stat.f_frsize)
        measured_from = "statvfs"
    else:
        free = int(free_bytes_override)
        measured_from = "override"

    needed = int(required_bytes) + int(safety_margin_bytes)
    if free < needed:
        raise LaunchDiskHeadroomError(
            [
                "launch_disk_headroom_insufficient_disk:"
                f"needed_bytes={needed}:free_bytes={free}:"
                f"shortfall_bytes={needed - free}:path={target}"
            ]
        )

    return {
        "schema_version": LAUNCH_DISK_HEADROOM_SCHEMA_VERSION,
        "sufficient": True,
        "measured_path": str(target),
        "measured_from": measured_from,
        "free_bytes": free,
        "needed_bytes": int(required_bytes),
        "safety_margin_bytes": int(safety_margin_bytes),
        "claim_boundary": {
            "headroom_now_is_not_headroom_for_the_whole_run": True,
            "another_process_may_consume_the_same_disk": True,
        },
    }


__all__ = [
    "DEFAULT_SAFETY_MARGIN_BYTES",
    "LAUNCH_DISK_HEADROOM_SCHEMA_VERSION",
    "LaunchDiskHeadroomError",
    "require_launch_disk_headroom",
]
