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
offered, because either would run unreviewed code as root. ``retire-scene-workspace``
(``operate``) runs the active release's own retention module for one scene: a plan,
or with ``apply`` a retirement that deletes nothing it cannot restore.
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
#: Every request kind and the token scope it needs. ``validate_request`` has one explicit
#: branch per kind, the runner one launch per non-unit kind, and ids name exactly these kinds.
_SCOPES = {
    "deploy": "deploy",
    "unit": "operate",
    "hold": "operate",
    "release-hold": "operate",
    "door-upgrade": "deploy",
    # Plans or retires one website scene workspace with the active release's own module; it
    # runs no new code, and the module deletes nothing it cannot restore.
    "retire-scene-workspace": "operate",
    "restore-scene-workspace": "operate",
}
_COMMIT = re.compile(r"[0-9a-f]{40}")
# The grammar the Pub/Sub listener accepts for a scene id and a GCS bucket.
_SCENE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]")
_REQUEST_ID = re.compile(
    r"[0-9]{8}T[0-9]{6}Z-(" + "|".join(re.escape(kind) for kind in sorted(_SCOPES)) + r")-[0-9a-f]{8}"
)
_UNIT_ACTIONS = ("start", "reset-failed", "stop", "restart")
_TRIGGER_ONLY_ACTIONS = ("stop", "restart")
_HOLD_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]{0,63}\Z")
# Timers that protect money or cleanup are never paused through the door.
_SAFETY_CRITICAL = re.compile(
    r"spend-guard|watchdog|teardown|reaper|provider-zero|"
    r"terminal-resource-release|storage-gc|capacity|replay-cache-gc|preflight"
)
_LOG_TAIL_LINES = 200


class RequestRefused(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def required_scope(kind: str) -> str:
    if not isinstance(kind, str) or kind not in _SCOPES:
        raise RequestRefused("kind_unknown")
    return _SCOPES[kind]


def _only(body: dict[str, Any], allowed: tuple[str, ...]) -> None:
    for key in body:
        if key not in allowed:
            raise RequestRefused(f"request_key_unknown:{key}")


def _commit(body: dict[str, Any]) -> str:
    commit = body.get("commit")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise RequestRefused("commit_invalid")
    return commit


def _hold_unit(unit: Any) -> str:
    if not isinstance(unit, str) or not UNIT_NAME.fullmatch(unit) or not unit.endswith((".timer", ".path")):
        raise RequestRefused("hold_unit_invalid")
    if unit.startswith("blueprint-operator-door"):
        raise RequestRefused("unit_is_door")
    if _SAFETY_CRITICAL.search(unit):
        raise RequestRefused("unit_safety_critical")
    return unit


def validate_request(body: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized request or raise ``RequestRefused``."""

    if not isinstance(body, dict):
        raise RequestRefused("request_not_object")
    kind = body.get("kind")
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
            if action == "stop":
                raise RequestRefused("unit_stop_requires_hold")
        return {"kind": kind, "unit": unit, "action": action}
    if kind == "hold":
        _only(body, ("kind", "unit", "owner", "reason", "expires_in_seconds"))
        unit = _hold_unit(body.get("unit"))
        owner, reason, duration = body.get("owner"), body.get("reason"), body.get("expires_in_seconds")
        if not isinstance(owner, str) or not _HOLD_OWNER.fullmatch(owner):
            raise RequestRefused("hold_owner_invalid")
        if not isinstance(reason, str) or not 1 <= len(reason) <= 200 or not reason.isprintable():
            raise RequestRefused("hold_reason_invalid")
        if type(duration) is not int or not 60 <= duration <= 86400:
            raise RequestRefused("hold_expiry_invalid")
        return {"kind": kind, "unit": unit, "owner": owner, "reason": reason,
                "expires_in_seconds": duration}
    if kind == "release-hold":
        _only(body, ("kind", "unit"))
        return {"kind": kind, "unit": _hold_unit(body.get("unit"))}
    if kind == "door-upgrade":
        _only(body, ("kind", "commit"))
        return {"kind": kind, "commit": _commit(body)}
    if kind == "retire-scene-workspace":
        _only(body, ("kind", "scene_id", "bucket", "apply"))
        scene_id = body.get("scene_id")
        if not isinstance(scene_id, str) or not _SCENE_ID.fullmatch(scene_id) or scene_id in {".", ".."}:
            raise RequestRefused("scene_id_invalid")
        apply = body.get("apply", False)
        if not isinstance(apply, bool):
            raise RequestRefused("apply_invalid")
        normalized: dict[str, Any] = {"kind": kind, "scene_id": scene_id, "apply": apply}
        if "bucket" in body:
            bucket = body["bucket"]
            if not isinstance(bucket, str) or not _BUCKET.fullmatch(bucket) or ".." in bucket:
                raise RequestRefused("bucket_invalid")
            normalized["bucket"] = bucket
        return normalized
    if kind == "restore-scene-workspace":
        _only(body, ("kind", "scene_id", "bucket"))
        scene_id, bucket = body.get("scene_id"), body.get("bucket")
        if not isinstance(scene_id, str) or not _SCENE_ID.fullmatch(scene_id) or scene_id in {".", ".."}:
            raise RequestRefused("scene_id_invalid")
        if not isinstance(bucket, str) or not _BUCKET.fullmatch(bucket) or ".." in bucket:
            raise RequestRefused("bucket_invalid")
        return {"kind": kind, "scene_id": scene_id, "bucket": bucket}
    # Anything else, however it is shaped, is not a request the door knows.
    raise RequestRefused("kind_unknown")


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
