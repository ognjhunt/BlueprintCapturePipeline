"""Persist proven-bad Vast machines across sibling Task Evaluation launches."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

from .common import write_json


CONTENT_AGENTS_MACHINE_AVOIDLIST_FILENAME = "adp-content-agents-vast-machine-avoidlist.json"
SIMREADY_ISAAC_MACHINE_AVOIDLIST_FILENAME = "adp009b-simready-isaac-vast-machine-avoidlist.json"


def load_machine_avoidlist(path: str | Path) -> dict[str, Any]:
    """Read one avoidlist without turning corrupt bytes into an empty list."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {
            "schema_version": "vast_machine_avoidlist.v1",
            "status": "empty",
            "machine_ids": [],
            "entries": [],
            "raw_secret_values_recorded": False,
        }
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "schema_version": "vast_machine_avoidlist.v1",
            "status": "blocked_parse_failed",
            "machine_ids": [],
            "entries": [],
            "parse_error": f"{type(exc).__name__}:{str(exc)[:200]}",
            "raw_secret_values_recorded": False,
        }
    return (
        dict(data)
        if isinstance(data, Mapping)
        else {
            "schema_version": "vast_machine_avoidlist.v1",
            "status": "blocked_invalid_shape",
            "machine_ids": [],
            "entries": [],
            "raw_secret_values_recorded": False,
        }
    )


def machine_avoidlist_ids(data: Mapping[str, Any]) -> set[int]:
    """Read every admitted avoidlist shape without silently dropping bad rows."""

    status = str(data.get("status") or "").strip()
    if status.startswith("blocked_"):
        raise ValueError(f"vast_machine_avoidlist_invalid:{status}")
    schema_version = data.get("schema_version")
    if schema_version not in (None, "vast_machine_avoidlist.v1"):
        raise ValueError("vast_machine_avoidlist_invalid:schema_version")

    def machine_id(value: Any, *, field: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"vast_machine_avoidlist_invalid:{field}")
        if isinstance(value, int):
            resolved = value
        elif isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value.strip()):
            resolved = int(value.strip())
        else:
            raise ValueError(f"vast_machine_avoidlist_invalid:{field}")
        if resolved <= 0:
            raise ValueError(f"vast_machine_avoidlist_invalid:{field}")
        return resolved

    ids: set[int] = set()
    direct_ids = data.get("machine_ids")
    if direct_ids is not None:
        if not isinstance(direct_ids, list):
            raise ValueError("vast_machine_avoidlist_invalid:machine_ids")
        for index, value in enumerate(direct_ids):
            ids.add(machine_id(value, field=f"machine_ids.{index}"))

    # ``machines`` is the original arena-lane v1 publication shape;
    # ``entries`` is the current adapter-written shape.
    for field in ("entries", "machines"):
        rows = data.get(field)
        if rows is None:
            continue
        if not isinstance(rows, list):
            raise ValueError(f"vast_machine_avoidlist_invalid:{field}")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"vast_machine_avoidlist_invalid:{field}.{index}")
            ids.add(
                machine_id(
                    row.get("machine_id"),
                    field=f"{field}.{index}.machine_id",
                )
            )
    return ids


def avoidlist_machine_ids(path: str | Path) -> set[int]:
    return machine_avoidlist_ids(load_machine_avoidlist(path))


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


def stage_machine_avoidlist_for_attempt(
    *, source_path: str | Path | None, destination_path: str | Path
) -> Path | None:
    """Copy one admission-bound snapshot to a provider-mutable attempt path."""

    if source_path is None:
        return None
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        return None
    destination = Path(destination_path).expanduser().resolve()
    if source == destination:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if destination.exists():
        if not destination.is_file() or hashlib.sha256(
            destination.read_bytes()
        ).hexdigest() != source_digest:
            raise ValueError("provider_machine_avoidlist_attempt_path_conflict")
        return destination
    shutil.copy2(source, destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != source_digest:
        destination.unlink(missing_ok=True)
        raise ValueError("provider_machine_avoidlist_attempt_copy_mismatch")
    return destination


__all__ = [
    "CONTENT_AGENTS_MACHINE_AVOIDLIST_FILENAME",
    "SIMREADY_ISAAC_MACHINE_AVOIDLIST_FILENAME",
    "avoidlist_machine_ids",
    "content_agents_machine_avoidlist_path",
    "load_machine_avoidlist",
    "machine_avoidlist_ids",
    "simready_isaac_machine_avoidlist_path",
    "stage_machine_avoidlist_for_attempt",
]
