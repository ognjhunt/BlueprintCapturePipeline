"""Privileged requests and the spool that carries them to the root runner.

The door process cannot deploy or control units: it only validates a request
and writes it to ``pending/``. A root oneshot, started by a ``.path`` unit,
revalidates each file with the same function before acting, so a file placed
in the spool by anything else still has to pass these schemas.

Spool layout under ``<state_root>/requests``::

    pending/<id>.json       written by the door (group-writable by the door only)
    processing/<id>.json    claimed by the runner   (root-owned from here on)
    completed/<id>.json     after the runner acted or refused
    results/<id>.json       the runner's result
    results/<id>.outcome.json, results/<id>.log   written by transient scripts

Only commits already on ``origin/main`` can be deployed: a ``deploy`` token lets
its holder get merged code running as root, and nothing more. Arbitrary pushed
branches (canary deploys) and candidate-code stage replays are deliberately not
offered, because either would run unreviewed code as root.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import secrets
import stat
import tempfile
from pathlib import Path
from typing import Any

from .config import DoorConfig
from .hostinfo import UNIT_NAME
from .secrets_guard import redact_lines

SCHEMA = "blueprint_operator_door_request.v1"
MAX_SPOOL_FILE = 64 * 1024
STATES = ("pending", "processing", "completed")
_SCOPES = {"deploy": "deploy", "unit": "operate", "door-upgrade": "deploy"}
_COMMIT = re.compile(r"[0-9a-f]{40}")
_REQUEST_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z-(deploy|unit|door-upgrade)-[0-9a-f]{8}")
_UNIT_ACTIONS = ("start", "reset-failed", "stop", "restart")
_TRIGGER_ONLY_ACTIONS = ("stop", "restart")
# Timers that protect money or cleanup are never paused through the door.
_SAFETY_CRITICAL = re.compile(r"spend-guard|watchdog|teardown|reaper|provider-zero")
_LOG_TAIL_LINES = 200


class RequestRefused(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def required_scope(kind: str) -> str:
    try:
        return _SCOPES[kind]
    except KeyError as error:
        raise RequestRefused("kind_unknown") from error


def _only(body: dict[str, Any], allowed: tuple[str, ...]) -> None:
    for key in body:
        if key not in allowed:
            raise RequestRefused(f"request_key_unknown:{key}")


def _commit(body: dict[str, Any]) -> str:
    commit = body.get("commit")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise RequestRefused("commit_invalid")
    return commit


def validate_request(body: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized request or raise ``RequestRefused``."""

    if not isinstance(body, dict):
        raise RequestRefused("request_not_object")
    kind = body.get("kind")
    if kind not in _SCOPES:
        raise RequestRefused("kind_unknown")
    if kind == "deploy":
        _only(body, ("kind", "commit", "wait_for_idle"))
        wait = body.get("wait_for_idle", True)
        if not isinstance(wait, bool):
            raise RequestRefused("wait_for_idle_invalid")
        return {"kind": kind, "commit": _commit(body), "wait_for_idle": wait}
    if kind == "unit":
        _only(body, ("kind", "unit", "action"))
        unit, action = body.get("unit"), body.get("action")
        if not isinstance(unit, str) or not UNIT_NAME.fullmatch(unit):
            raise RequestRefused("unit_name_invalid")
        if unit.startswith("blueprint-operator-door"):
            raise RequestRefused("unit_is_door")
        if action not in _UNIT_ACTIONS:
            raise RequestRefused("unit_action_invalid")
        if action in _TRIGGER_ONLY_ACTIONS:
            # Pausing a trigger never kills a running job; stopping a service could.
            if not unit.endswith((".timer", ".path")):
                raise RequestRefused("unit_action_not_allowed")
            if _SAFETY_CRITICAL.search(unit):
                raise RequestRefused("unit_safety_critical")
        return {"kind": kind, "unit": unit, "action": action}
    _only(body, ("kind", "commit"))
    return {"kind": kind, "commit": _commit(body)}


def new_request_id(kind: str, now: _dt.datetime | None = None) -> str:
    moment = (now or _dt.datetime.now(_dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{moment}-{kind}-{secrets.token_hex(4)}"


def validate_request_id(request_id: str) -> str:
    if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
        raise RequestRefused("request_id_invalid")
    return request_id


def enqueue(config: DoorConfig, request: dict[str, Any], *, requested_by: str) -> str:
    normalized = validate_request(request)
    request_id = new_request_id(normalized["kind"])
    document = {
        "schema": SCHEMA,
        "id": request_id,
        "request": normalized,
        "requested_by": requested_by,
        "requested_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    pending = Path(config.spool_root) / "pending"
    handle, temporary = tempfile.mkstemp(dir=pending, prefix=f".{request_id}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
        os.chmod(temporary, 0o644)
        os.replace(temporary, pending / f"{request_id}.json")
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return request_id


def _open_regular(path: Path) -> tuple[int, os.stat_result]:
    """Open without following symlinks or blocking on FIFOs; regular files only."""

    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise RequestRefused("spool_file_unsafe") from error
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise RequestRefused("spool_file_unsafe")
    return fd, info


def load_request_file(path: Path) -> dict[str, Any]:
    """Read a spool file defensively: no symlinks, regular files, bounded size."""

    fd, info = _open_regular(path)
    try:
        if info.st_size > MAX_SPOOL_FILE:
            raise RequestRefused("spool_file_too_large")
        data = os.read(fd, MAX_SPOOL_FILE + 1)
    finally:
        os.close(fd)
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RequestRefused("spool_file_not_json") from error
    if not isinstance(document, dict):
        raise RequestRefused("spool_file_not_object")
    return document


def _json_or_none(path: Path) -> Any:
    try:
        return load_request_file(path)
    except RequestRefused:
        return None


def _tail(path: Path) -> str | None:
    try:
        fd, info = _open_regular(path)
    except RequestRefused:
        return None
    try:
        start = max(0, info.st_size - 256 * 1024)
        os.lseek(fd, start, os.SEEK_SET)
        text = os.read(fd, 256 * 1024 + 1).decode("utf-8", "replace")
    finally:
        os.close(fd)
    return redact_lines("\n".join(text.splitlines()[-_LOG_TAIL_LINES:]) + "\n")


def request_state(config: DoorConfig, request_id: str) -> dict[str, Any]:
    validate_request_id(request_id)
    spool = Path(config.spool_root)
    state, document = "unknown", None
    for candidate in ("completed", "processing", "pending"):
        path = spool / candidate / f"{request_id}.json"
        if path.exists():
            state, document = candidate, _json_or_none(path)
            break
    results = spool / "results"
    return {
        "id": request_id,
        "state": state,
        "request": (document or {}).get("request"),
        "requested_by": (document or {}).get("requested_by"),
        "requested_at": (document or {}).get("requested_at"),
        "result": _json_or_none(results / f"{request_id}.json"),
        "outcome": _json_or_none(results / f"{request_id}.outcome.json"),
        "log_tail": _tail(results / f"{request_id}.log"),
    }


def list_requests(config: DoorConfig, limit: int = 20) -> list[dict[str, Any]]:
    spool = Path(config.spool_root)
    found: list[tuple[float, str, str]] = []
    for state in STATES:
        for path in (spool / state).glob("*.json"):
            if not _REQUEST_ID.fullmatch(path.stem):
                continue
            try:
                found.append((path.stat().st_mtime, path.stem, state))
            except FileNotFoundError:
                continue  # moved by the runner between listing and stat
    found.sort(reverse=True)
    return [{"id": request_id, "state": state, "kind": request_id.split("-", 1)[1].rsplit("-", 1)[0]}
            for _, request_id, state in found[:limit]]
