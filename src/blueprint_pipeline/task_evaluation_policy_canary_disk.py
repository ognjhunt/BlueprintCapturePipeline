"""Disk reservation for one policy-canary dispatch and its resumed passes."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .control_plane_disk_budget import reserve_control_plane_disk


RUN_START_MARKERS = (
    "policy_canary_session_authority.json",
    "allocator_invocation_started.json",
)
DOWNLOAD_WORKLOAD = "policy_canary"
# A streamed run directory holds about 1.1 GB against download mode's 7.75 GB
# (plan 15): its samples form their own measured group.
STREAMED_WORKLOAD = "policy_canary_streamed"


def canary_workload(environ: Mapping[str, str] | None = None) -> str:
    """The footprint label for a run the unit's delivery mode will launch.

    The dispatcher launches the allocator with its own environment, so the mode
    it resolves is the one the Quick-10 session resolves: unset streams exactly
    when the dedicated B2 store is configured. An invalid mode is refused by the
    session before any spend and is labelled as download.
    """
    from .policy_canary_output_members import (
        STREAM,
        PolicyCanaryOutputDeliveryError,
        resolve_output_delivery,
    )

    try:
        return STREAMED_WORKLOAD if resolve_output_delivery(environ).mode == STREAM else DOWNLOAD_WORKLOAD
    except PolicyCanaryOutputDeliveryError:
        return DOWNLOAD_WORKLOAD


def canary_disk_reservation(*, output: Path, outputs: Path, reservation_root: Path | None,
                            environ: Mapping[str, str] | None = None) -> Any:
    """Reserve the run's workspace, never treating a resumed pass as a fresh sample."""

    if reservation_root is None:
        return contextlib.nullcontext()
    resumes_run = any((output / name).exists() for name in RUN_START_MARKERS)
    return reserve_control_plane_disk(
        "policy_canary_dispatch",
        target_root=outputs,
        reservation_root=reservation_root,
        workspace=output,
        workload=canary_workload(environ),
        fresh=False if resumes_run else None,
    )
