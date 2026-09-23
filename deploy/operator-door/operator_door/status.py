"""One status document: what a run snapshot on the host used to take a dozen commands.

Each section is computed independently; a section that fails reports
``{"error": ...}`` instead of failing the whole document, because a status call
is most useful precisely when something on the host is broken.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Callable

from . import VERSION
from .config import DoorConfig
from .hostinfo import HostInfo
from .secrets_guard import scan_bytes

SCHEMA = "blueprint_operator_door_status.v1"
_VERSION_KEYS = ("source_commit", "commit_proven", "blockers", "disk_headroom", "claim_ceiling")
_RECEIPT_KEYS = ("schema", "status", "source_commit", "commit", "mode", "finished_at", "completed_at")
_MAX_JSON_BYTES = 256 * 1024


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


def _door_requests(config: DoorConfig) -> dict[str, int]:
    spool = Path(config.spool_root)
    return {state: len(list((spool / state).glob("*.json"))) for state in ("pending", "processing")}


def build_status(config: DoorConfig, host: HostInfo, *, caller: dict[str, Any]) -> dict[str, Any]:
    state = Path(config.control_plane_state)
    return {
        "schema": SCHEMA,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "door": {"version": VERSION, "caller": caller},
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
        "load": _section(host.load, "load_unavailable"),
        "door_requests": _section(lambda: _door_requests(config), "door_requests_unavailable"),
    }
