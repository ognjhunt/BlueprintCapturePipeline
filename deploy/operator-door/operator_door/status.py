"""One status document: what a run snapshot on the host used to take a dozen commands.

Each section is computed independently; a section that fails reports
``{"error": ...}`` instead of failing the whole document, because a status call
is most useful precisely when something on the host is broken.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Callable

from . import VERSION
from .admitted_controls import DISPATCHER_HOLD_ONLY
from .requests import _SCOPES
from . import holds as hold_records
from .config import DoorConfig
from .hostinfo import HostInfo
from .secrets_guard import scan_bytes

SCHEMA = "blueprint_operator_door_status.v1"
_VERSION_KEYS = ("source_commit", "commit_proven", "blockers", "disk_headroom", "claim_ceiling")
_RECEIPT_KEYS = ("schema", "status", "source_commit", "commit", "mode", "finished_at", "completed_at")
_CAPACITY_KEYS = (
    "schema_version", "observed_at_epoch", "level", "report_digest", "alerts", "mounts", "usage",
    "volume_resize", "truncated",
)
_MAX_JSON_BYTES = 256 * 1024
_MAX_BREAK_GLASS_LEDGER_BYTES = 16 * 1024 * 1024


def _section(builder: Callable[[], Any], code: str) -> Any:
    try:
        return builder()
    except Exception as error:  # noqa: BLE001 - one broken section must not hide the rest
        return {"error": f"{code}:{type(error).__name__}"}


class SecretContentRefused(PermissionError):
    """The file holds credential-shaped content, so none of it is served."""


def _small_json(path: Path) -> Any:
    data = path.read_bytes()[: _MAX_JSON_BYTES + 1]
    if len(data) > _MAX_JSON_BYTES:
        raise ValueError("too_large")
    if scan_bytes(data) is not None:
        raise SecretContentRefused("secret_content")
    return json.loads(data)


def _regular_bytes(path: Path, limit: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("unsafe_file")
        data = os.read(fd, limit + 1)
    finally:
        os.close(fd)
    if len(data) > limit:
        raise ValueError("too_large")
    return data


def _break_glass(config: DoorConfig) -> dict[str, Any]:
    root = Path(config.control_plane_state) / "cleanup-receipts"
    if not root.exists() and not root.is_symlink():
        return {"unreported": 0, "latest": None}
    if root.is_symlink() or not root.is_dir():
        raise ValueError("notes_root_unsafe")
    reported: set[str] = set()
    ledger = root / "reported.jsonl"
    if ledger.exists() or ledger.is_symlink():
        for line in _regular_bytes(ledger, _MAX_BREAK_GLASS_LEDGER_BYTES).splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and isinstance(row.get("name"), str):
                reported.add(row["name"])
    notes = sorted(root.glob("*.json"))
    if len(notes) > 10_000:
        raise ValueError("too_many_notes")
    unreported = [path for path in notes if path.name not in reported]
    latest: tuple[int, dict[str, str]] | None = None
    for path in unreported:
        raw = _regular_bytes(path, _MAX_JSON_BYTES)
        if scan_bytes(raw) is not None:
            continue
        try:
            note = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(note, dict) or type(note.get("created_at_epoch")) is not int:
            continue
        if not all(isinstance(note.get(key), str) for key in ("created_at", "operator", "reason")):
            continue
        summary = {key: note[key] for key in ("created_at", "operator", "reason")}
        if latest is None or note["created_at_epoch"] > latest[0]:
            latest = (note["created_at_epoch"], summary)
    return {"unreported": len(unreported), "latest": latest[1] if latest is not None else None}


def _deployed(host: HostInfo) -> dict[str, Any]:
    try:
        payload = host.version()
    except Exception as error:  # noqa: BLE001
        return {"error": f"version_unavailable:{type(error).__name__}"}
    return {key: payload[key] for key in _VERSION_KEYS if key in payload}


def _active_release(config: DoorConfig) -> dict[str, Any]:
    link = Path(config.active_release_link)
    target = os.readlink(link)
    return {"link": str(link), "target": target, "commit": Path(target).name}


def _recent_receipts(config: DoorConfig, limit: int = 5) -> list[dict[str, Any]]:
    directory = Path(config.control_plane_state) / "deploy-receipts"
    receipts = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    summaries: list[dict[str, Any]] = []
    for path in receipts[:limit]:
        entry: dict[str, Any] = {"name": path.name, "mtime": int(path.stat().st_mtime)}
        try:
            document = _small_json(path)
            if isinstance(document, dict):
                entry.update({key: document[key] for key in _RECEIPT_KEYS if key in document})
        except SecretContentRefused:
            entry["error"] = "secret_content_refused"
        except Exception as error:  # noqa: BLE001
            entry["error"] = type(error).__name__
        summaries.append(entry)
    return summaries


def _capacity(config: DoorConfig) -> dict[str, Any]:
    """The capacity controller's summary: levels, alerts and usage attribution."""

    document = _small_json(Path(config.capacity_summary))
    if not isinstance(document, dict):
        raise ValueError("not_an_object")
    return {key: document[key] for key in _CAPACITY_KEYS if key in document}


def _door_requests(config: DoorConfig) -> dict[str, int]:
    spool = Path(config.spool_root)
    return {state: len(list((spool / state).glob("*.json"))) for state in ("pending", "processing")}


def _holds(config: DoorConfig) -> list[dict[str, Any]]:
    now = time.time()
    records = hold_records.active(Path(config.spool_root) / "holds")
    keys = ("unit", "owner", "reason", "requested_by", "request_id", "created_at", "expires_at")
    return [{**{key: record[key] for key in keys},
             "remaining_seconds": max(0, int(record["expires_at_epoch"] - now)),
             "expired": record["expires_at_epoch"] <= now} for record in records]


def build_status(config: DoorConfig, host: HostInfo, *, caller: dict[str, Any]) -> dict[str, Any]:
    state = Path(config.control_plane_state)
    return {
        "schema": SCHEMA,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "door": {"version": VERSION, "caller": caller,
                 "request_kinds": sorted(_SCOPES),
                 "hold_target": "blueprint-agent-run-dispatcher.timer" if DISPATCHER_HOLD_ONLY else None},
        "deployed": _deployed(host),
        "active_release": _section(lambda: _active_release(config), "active_release_unavailable"),
        "deploys": {
            "active_units": _section(
                lambda: [row["unit"] for row in host.list_units(
                    "blueprint-*deploy*", states=("active", "activating", "deactivating", "reloading"))],
                "deploy_units_unavailable",
            ),
            "recent_receipts": _section(lambda: _recent_receipts(config), "receipts_unavailable"),
        },
        "paid_launch_locks": _section(host.paid_launch_locks, "locks_unavailable"),
        "spend_guard": _section(lambda: _small_json(state / "gpu_spend_guard" / "latest.json"),
                                "spend_guard_unavailable"),
        "failed_units": _section(
            lambda: [row["unit"] for row in host.list_units("blueprint-*", states=("failed",))],
            "failed_units_unavailable",
        ),
        "controller_units": _section(lambda: host.unit_properties(config.controller_units),
                                     "controller_units_unavailable"),
        "disk": _section(lambda: host.disk(("/", "/var/lib/blueprint", "/mnt/blueprint-work")),
                         "disk_unavailable"),
        "capacity": _section(lambda: _capacity(config), "capacity_unavailable"),
        "load": _section(host.load, "load_unavailable"),
        "door_requests": _section(lambda: _door_requests(config), "door_requests_unavailable"),
        "holds": _section(lambda: _holds(config), "holds_unavailable"),
        "break_glass": _section(lambda: _break_glass(config), "break_glass_unavailable"),
    }
