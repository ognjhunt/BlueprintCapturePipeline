"""Non-terminal progress for an in-flight Task Evaluation launch.

A launch takes roughly twenty-five minutes, almost all of it before any
terminal receipt exists, so the WebApp previously showed nothing at all while
boot, the pinned dependency closure, scene construction, and the runtime phases
went by. This reads the evidence the run is already writing and shapes it into
one small progress record.

It is strictly observational. It never asserts success, never writes a terminal
status, and reports only what the run has already recorded, so a progress
record can never be mistaken for a result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROGRESS_SCHEMA_VERSION = "task_evaluation_launch_progress.v1"
# The worker writes one JSON object per line as it advances; the last line is
# the current phase. Globbed rather than hardcoded so a differently shaped job
# directory degrades to "no phase yet" instead of raising.
PHASE_LOG_GLOB = "**/vast_runtime_phase_log.jsonl"
MAX_PHASE_LOG_BYTES = 512 * 1024


def _tail_phase(run_root: Path) -> tuple[str | None, str | None]:
    """Return the most recent (phase, status) the worker recorded."""

    try:
        candidates = sorted(run_root.glob(PHASE_LOG_GLOB))
    except OSError:
        return None, None
    for path in reversed(candidates):
        try:
            if path.stat().st_size > MAX_PHASE_LOG_BYTES:
                with path.open("rb") as stream:
                    stream.seek(-MAX_PHASE_LOG_BYTES, 2)
                    text = stream.read().decode("utf-8", errors="ignore")
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping) and row.get("phase"):
                return str(row.get("phase")), (
                    str(row.get("status")) if row.get("status") else None
                )
    return None, None


def _lane_instance(guard: Mapping[str, Any], *, name_prefix: str) -> dict[str, Any] | None:
    """Find this lane's instance without attributing another operator's spend."""

    for row in guard.get("instances") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("name") or "").startswith(name_prefix):
            return dict(row)
    return None


def build_launch_progress(
    *,
    run_root: str | Path,
    request: Mapping[str, Any],
    guard: Mapping[str, Any] | None = None,
    elapsed_seconds: float,
    observed_at: datetime | None = None,
    lane_instance_prefix: str = "blueprint-adp009d-",
) -> dict[str, Any]:
    """Shape one observational progress record for an in-flight launch."""

    moment = (observed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    phase, phase_status = _tail_phase(Path(run_root).expanduser())
    progress: dict[str, Any] = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "launch_id": request.get("launch_id"),
        "run_id": request.get("run_id"),
        "request_digest": request.get("request_digest"),
        "phase": phase or "starting",
        "phase_status": phase_status or "running",
        "observed_at_iso": moment.isoformat(),
        "elapsed_seconds": round(max(0.0, float(elapsed_seconds)), 3),
    }
    instance = _lane_instance(guard or {}, name_prefix=lane_instance_prefix)
    if instance is not None:
        age = instance.get("age_seconds")
        rate = instance.get("cost_per_hr_usd")
        cost = None
        if isinstance(age, (int, float)) and isinstance(rate, (int, float)):
            cost = round(float(age) * float(rate) / 3600.0, 6)
        progress["provider"] = {
            "instance_state": str(instance.get("state") or "unknown"),
            "instance_age_seconds": round(float(age), 3)
            if isinstance(age, (int, float))
            else None,
            # Derived from the observed rate and age. An estimate, never the
            # billed figure, which only the terminal receipt carries.
            "estimated_cost_usd": cost,
        }
    return progress


__all__ = ["PROGRESS_SCHEMA_VERSION", "build_launch_progress"]
