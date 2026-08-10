"""Fail-closed local storage admission for paid evidence-producing runs.

Paid provider teardown can succeed while the local orchestrator is unable to
write the receipt that proves it.  This module keeps that failure mode out of
individual scene/task launchers: measure enough headroom before any mutation
and reserve a small amount of real (non-sparse) storage for closeout evidence.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "adp_paid_local_evidence_capacity.v1"
DEFAULT_MINIMUM_FREE_BYTES = 2 * 1024**3
DEFAULT_CLOSEOUT_RESERVE_BYTES = 16 * 1024**2
DiskUsage = Callable[[Path], Any]


def measure_paid_local_evidence_capacity(
    *,
    evidence_root: str | Path,
    immutable_input_paths: Sequence[str | Path],
    blocker: str,
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES,
    input_replica_multiplier: int = 2,
    closeout_reserve_bytes: int = DEFAULT_CLOSEOUT_RESERVE_BYTES,
    disk_usage: DiskUsage = shutil.disk_usage,
) -> dict[str, Any]:
    """Measure local capacity before provider/object-store mutation.

    ``input_replica_multiplier`` accounts for retained download and extraction
    copies of immutable inputs.  The absolute floor covers logs, media, and
    receipts even for small bundles.  Every input must already exist so a
    caller cannot get a favorable estimate from an unresolved path.
    """

    root = Path(evidence_root).expanduser().resolve()
    inputs = [Path(path).expanduser().resolve() for path in immutable_input_paths]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise ValueError("adp_paid_local_evidence_input_missing:" + ",".join(missing))
    if min(minimum_free_bytes, input_replica_multiplier, closeout_reserve_bytes) < 0:
        raise ValueError("adp_paid_local_evidence_capacity_contract_invalid")
    input_bytes = sum(path.stat().st_size for path in inputs)
    required = (
        max(int(minimum_free_bytes), input_bytes * int(input_replica_multiplier))
        + int(closeout_reserve_bytes)
    )
    usage = disk_usage(root)
    passed = int(usage.free) >= required
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "blocked",
        "filesystem_path": str(root),
        "observed_free_bytes": int(usage.free),
        "minimum_free_bytes": required,
        "immutable_input_bytes": input_bytes,
        "immutable_input_count": len(inputs),
        "input_replica_multiplier": int(input_replica_multiplier),
        "closeout_reserve_bytes": int(closeout_reserve_bytes),
        "blockers": [] if passed else [str(blocker)],
        "raw_secret_values_recorded": False,
    }


def materialize_local_closeout_reserve(
    path: str | Path,
    *,
    size_bytes: int = DEFAULT_CLOSEOUT_RESERVE_BYTES,
) -> Path:
    """Allocate and fsync real bytes that can later be traded for receipts."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if size_bytes < 0:
        raise ValueError("adp_paid_local_closeout_reserve_size_invalid")
    block = b"\0" * min(1024 * 1024, max(1, int(size_bytes)))
    remaining = int(size_bytes)
    with destination.open("wb") as stream:
        while remaining:
            chunk = block[: min(len(block), remaining)]
            stream.write(chunk)
            remaining -= len(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    if destination.stat().st_size != int(size_bytes):
        raise OSError("adp_paid_local_closeout_reserve_size_mismatch")
    return destination


def release_local_closeout_reserve(path: str | Path) -> None:
    """Release the reserved bytes immediately before closeout writes."""

    Path(path).expanduser().resolve().unlink(missing_ok=True)


__all__ = [
    "DEFAULT_CLOSEOUT_RESERVE_BYTES",
    "DEFAULT_MINIMUM_FREE_BYTES",
    "SCHEMA_VERSION",
    "materialize_local_closeout_reserve",
    "measure_paid_local_evidence_capacity",
    "release_local_closeout_reserve",
]
