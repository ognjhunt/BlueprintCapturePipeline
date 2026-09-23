"""Privileged requests and the spool that carries them to the root runner.

The door process cannot deploy or control units: it only validates a request
and writes it to ``pending/``. A root oneshot, started by a ``.path`` unit,
revalidates each file with the same function before acting, so a file placed
in the spool by anything else still has to pass these schemas.

Spool layout under ``<state_root>/requests``::

    pending/<id>.json       written by the door (atomic rename, mode 0644)
    processing/<id>.json    claimed by the runner
    completed/<id>.json     after the runner acted or refused
    results/<id>.json       the runner's result
    results/<id>.outcome.json, results/<id>.log   written by transient scripts
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
_SCOPES = {"deploy": "deploy", "unit": "operate", "stage-replay": "operate", "door-upgrade": "deploy"}
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CHILD = re.compile(r"^sam31-[a-f0-9]{8,64}$")
_PARENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{3,159}$")
_REQUEST_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-(deploy|unit|stage-replay|door-upgrade)-[0-9a-f]{8}$")
_UNIT_ACTIONS = ("start", "reset-failed", "stop", "restart")
_TRIGGER_ONLY_ACTIONS = ("stop", "restart")
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
    if not isinstance(commit, str) or not _COMMIT.match(commit):
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
        _only(body, ("kind", "commit", "mode", "wait_for_idle"))
        mode = body.get("mode", "main")
        if mode not in ("main", "canary"):
            raise RequestRefused("mode_invalid")
        wait = body.get("wait_for_idle", True)
        if not isinstance(wait, bool):
            raise RequestRefused("wait_for_idle_invalid")
        return {"kind": kind, "commit": _commit(body), "mode": mode, "wait_for_idle": wait}
    if kind == "unit":
        _only(body, ("kind", "unit", "action"))
        unit, action = body.get("unit"), body.get("action")
        if not isinstance(unit, str) or not UNIT_NAME.match(unit):
            raise RequestRefused("unit_name_invalid")
        if unit.startswith("blueprint-operator-door"):
            raise RequestRefused("unit_is_door")
        if action not in _UNIT_ACTIONS:
            raise RequestRefused("unit_action_invalid")
        if action in _TRIGGER_ONLY_ACTIONS and not unit.endswith((".timer", ".path")):
            # Pausing a trigger never kills a running job; stopping a service could.
            raise RequestRefused("unit_action_not_allowed")
        return {"kind": kind, "unit": unit, "action": action}
    if kind == "stage-replay":
        _only(body, ("kind", "commit", "child", "parent"))
        child, parent = body.get("child"), body.get("parent")
        if (child is None) == (parent is None):
            raise RequestRefused("replay_target_invalid")
        if child is not None and (not isinstance(child, str) or not _CHILD.match(child)):
            raise RequestRefused("replay_target_invalid")
        if parent is not None and (not isinstance(parent, str) or not _PARENT.match(parent)):
            raise RequestRefused("replay_target_invalid")
        return {"kind": kind, "commit": _commit(body), "child": child, "parent": parent}
    _only(body, ("kind", "commit"))
    return {"kind": kind, "commit": _commit(body)}


def new_request_id(kind: str, now: _dt.datetime | None = None) -> str:
    moment = (now or _dt.datetime.now(_dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{moment}-{kind}-{secrets.token_hex(4)}"


def validate_request_id(request_id: str) -> str:
    if not isinstance(request_id, str) or not _REQUEST_ID.match(request_id):
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


def load_request_file(path: Path) -> dict[str, Any]:
    """Read a spool file defensively: no symlinks, regular files, bounded size."""

    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise RequestRefused("spool_file_unsafe") from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RequestRefused("spool_file_unsafe")
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
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 256 * 1024))
            text = stream.read().decode("utf-8", "replace")
    except OSError:
        return None
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
            if _REQUEST_ID.match(path.stem):
                found.append((path.stat().st_mtime, path.stem, state))
    found.sort(reverse=True)
    return [{"id": request_id, "state": state, "kind": request_id.split("-", 1)[1].rsplit("-", 1)[0]}
            for _, request_id, state in found[:limit]]
