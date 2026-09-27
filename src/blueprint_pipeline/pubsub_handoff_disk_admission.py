"""Disk admission and retryable capacity results for Pub/Sub capture staging."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import PipelineError, utc_now_iso
from .control_plane_disk_budget import (
    ControlPlaneDiskBudgetError,
    DEFAULT_RESERVATION_ROOT,
    reserve_control_plane_disk,
)
from .control_plane_disk_reservation_heartbeat import keep_reservation_live


class HandoffStagingCapacityError(PipelineError):
    """Capture download waits for bulk disk headroom."""


def staging_manifest_row(blob: Any, *, name: str, relative_path: str) -> dict[str, Any]:
    """The cloud identity and size of one listed object."""

    size = getattr(blob, "size", None)
    generation = getattr(blob, "generation", None)
    md5_hash = getattr(blob, "md5_hash", None)
    crc32c = getattr(blob, "crc32c", None)
    return {
        "name": name,
        "relative_path": relative_path,
        "size": size if isinstance(size, int) and not isinstance(size, bool) and size >= 0 else None,
        "generation": str(generation) if generation is not None and str(generation).strip() else None,
        "md5_hash": md5_hash if isinstance(md5_hash, str) and md5_hash else None,
        "crc32c": crc32c if isinstance(crc32c, str) and crc32c else None,
    }


def download_with_reservation(
    *,
    downloads: Sequence[tuple[Any, Path]],
    manifest_rows: Sequence[Mapping[str, Any]],
    storage_root: Path,
    capture_root: Path,
) -> None:
    listed_sizes = {row["name"]: row["size"] or 0 for row in manifest_rows}
    expected_bytes = sum(
        listed_sizes[str(blob.name)] for blob, _destination in downloads
    ) + 64 * 1024 * 1024
    try:
        reservation = reserve_control_plane_disk(
            "handoff_staging",
            target_root=storage_root,
            reservation_root=os.getenv(
                "BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT",
                str(DEFAULT_RESERVATION_ROOT),
            ),
            expected_bytes=expected_bytes,
            workspace=capture_root,
            workload="handoff_staging",
        )
    except ControlPlaneDiskBudgetError as exc:
        raise HandoffStagingCapacityError(
            "pubsub_handoff_staging_capacity_insufficient"
        ) from exc
    with reservation, keep_reservation_live(reservation):
        for blob, destination in downloads:
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(destination))


def finish_staging_capacity_blocked(
    *, capture_root: Path, handoff: Any, owner: str, token: str,
    attempt_count: int, attempt_started_at: str, previous_history: Sequence[Mapping[str, Any]],
    failure_stage: str, finish_job_lease: Callable[..., Any],
) -> dict[str, Any]:
    blocked_at = utc_now_iso()
    blocker = "pubsub_handoff_staging_capacity_insufficient"
    finish_job_lease(
        capture_root,
        owner=owner,
        token=token,
        update={
            "status": "retryable_blocked",
            "updated_at": blocked_at,
            "last_error_type": "HandoffStagingCapacityError",
            "last_error": blocker,
            "retry_blockers": [blocker],
            "queue_disposition": "retryable",
            "attempt_history": [*previous_history, {
                "attempt_number": attempt_count,
                "status": "retryable_blocked",
                "stage": failure_stage,
                "started_at": attempt_started_at,
                "completed_at": blocked_at,
                "blockers": [blocker],
            }],
        },
    )
    return {
        "schema_version": "v1",
        "status": "retryable_blocked",
        "queue_disposition": "retryable",
        "blockers": [blocker],
        "bucket": handoff.bucket,
        "scene_id": handoff.scene_id,
        "capture_id": handoff.capture_id,
        "capture_root": str(capture_root),
    }
