"""Persist proven-bad Vast machines across sibling Task Evaluation launches."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from .common import write_json


CONTENT_AGENTS_MACHINE_AVOIDLIST_FILENAME = "adp-content-agents-vast-machine-avoidlist.json"
SIMREADY_ISAAC_MACHINE_AVOIDLIST_FILENAME = "adp009b-simready-isaac-vast-machine-avoidlist.json"


def _lane_machine_avoidlist_path(
    *,
    job_dir: str | Path,
    explicit_path: str | Path | None,
    job_dir_name: str,
    shared_filename: str,
    retained_filename: str,
) -> Path | None:
    """Merge one lane's retained avoidlists into its control-plane shared path."""

    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    resolved_job = Path(job_dir).expanduser().resolve()
    if resolved_job.name != job_dir_name or resolved_job.parent.name != "allocator":
        return None
    state_root = resolved_job.parent.parent.parent
    shared = state_root / "provider-machine-avoidlists" / shared_filename

    machine_ids: set[int] = set()
    entries: list[dict[str, Any]] = []
    seen_entries: set[str] = set()
    candidates = [shared]
    candidates.extend(
        sorted(state_root.glob(f"*/allocator/{job_dir_name}/{retained_filename}"))
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != "vast_machine_avoidlist.v1"
        ):
            continue
        for value in payload.get("machine_ids") or []:
            if isinstance(value, int) and not isinstance(value, bool):
                machine_ids.add(value)
        for value in payload.get("entries") or []:
            if not isinstance(value, Mapping):
                continue
            row = dict(value)
            machine_id = row.get("machine_id")
            if isinstance(machine_id, int) and not isinstance(machine_id, bool):
                machine_ids.add(machine_id)
            identity = json.dumps(row, sort_keys=True, separators=(",", ":"))
            if identity not in seen_entries:
                seen_entries.add(identity)
                entries.append(row)
    if machine_ids or entries:
        write_json(
            shared,
            {
                "schema_version": "vast_machine_avoidlist.v1",
                "status": "completed",
                "machine_ids": sorted(machine_ids),
                "entries": entries,
                "raw_secret_values_recorded": False,
            },
        )
    return shared


def content_agents_machine_avoidlist_path(
    *, job_dir: str | Path, explicit_path: str | Path | None
) -> Path | None:
    return _lane_machine_avoidlist_path(
        job_dir=job_dir,
        explicit_path=explicit_path,
        job_dir_name="content-agents-job",
        shared_filename=CONTENT_AGENTS_MACHINE_AVOIDLIST_FILENAME,
        retained_filename="vast_machine_avoidlist.json",
    )


def simready_isaac_machine_avoidlist_path(
    *, job_dir: str | Path, explicit_path: str | Path | None
) -> Path | None:
    return _lane_machine_avoidlist_path(
        job_dir=job_dir,
        explicit_path=explicit_path,
        job_dir_name="simready-isaac-job",
        shared_filename=SIMREADY_ISAAC_MACHINE_AVOIDLIST_FILENAME,
        retained_filename="adp009b_simready_isaac_machine_avoidlist.json",
    )


__all__ = [
    "CONTENT_AGENTS_MACHINE_AVOIDLIST_FILENAME",
    "SIMREADY_ISAAC_MACHINE_AVOIDLIST_FILENAME",
    "content_agents_machine_avoidlist_path",
    "simready_isaac_machine_avoidlist_path",
]
