"""Disk reservation for one policy-canary dispatch and its resumed passes."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from .control_plane_disk_budget import reserve_control_plane_disk


RUN_START_MARKERS = (
    "policy_canary_session_authority.json",
    "allocator_invocation_started.json",
)


def canary_disk_reservation(*, output: Path, outputs: Path, reservation_root: Path | None) -> Any:
    """Reserve the run's workspace, never treating a resumed pass as a fresh sample."""

    if reservation_root is None:
        return contextlib.nullcontext()
    resumes_run = any((output / name).exists() for name in RUN_START_MARKERS)
    return reserve_control_plane_disk(
        "policy_canary_dispatch",
        target_root=outputs,
        reservation_root=reservation_root,
        workspace=output,
        workload="policy_canary",
        fresh=False if resumes_run else None,
    )
