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
``provider-output-resume`` (``operate``) runs the active release's
``provider_output_promotion resume`` for one streamed policy-canary attempt,
as the service user: promotion, the gated cleanup, the absence proof and, with
``ingest``, the needed members' ingestion. It runs no new code.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
from pathlib import Path
from typing import Any

from .config import DoorConfig
from .admitted_controls import DISPATCHER_HOLD_ONLY
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
    "lane-scratch": "operate",
    "owner-census-decision": "operate",
    "retire-scene": "operate",
    "restore-scene": "operate",
    "legacy-owner-census": "operate",
    # Resumes one streamed canary attempt's promotion/ingestion with the active release's module.
    "provider-output-resume": "operate",
}
if DISPATCHER_HOLD_ONLY:
    # Preserve the installed d78ee479 controls plus the approved dispatcher gate.
    # The API and privileged spool reader share this exact request allowlist.
    _SCOPES = {kind: scope for kind, scope in _SCOPES.items()
               if kind in {"deploy", "unit", "door-upgrade", "hold", "release-hold"}}
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
_LANE_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_LEASE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
# A policy-canary dispatch directory name (the activation id).
_CANARY_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
# Timers that protect money or cleanup are never paused through the door.
_SAFETY_CRITICAL = re.compile(
    r"spend-guard|watchdog|teardown|reaper|provider-zero|"
    r"terminal-resource-release|storage-gc|capacity|replay-cache-gc|preflight"
)
_BASELINE_TRIGGER_SAFETY = re.compile(r"spend-guard|watchdog|teardown|reaper|provider-zero")
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
    if DISPATCHER_HOLD_ONLY and unit != "blueprint-agent-run-dispatcher.timer":
        raise RequestRefused("hold_unit_profile_refused")
    return unit


def _lane_part(value: Any, field: str) -> str:
    if not isinstance(value, str) or _LANE_PART.fullmatch(value) is None:
        raise RequestRefused(f"lane_scratch_{field}_invalid")
    return value


def validate_request(body: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized request or raise ``RequestRefused``."""

    if not isinstance(body, dict):
        raise RequestRefused("request_not_object")
    kind = body.get("kind")
    required_scope(kind)
    if kind in ("retire-scene", "restore-scene"):
        fields = {"kind", "intent_id", "consent_id", "expected_sha256", "expected_size_bytes"}
        if kind == "retire-scene":
            fields.add("apply")
        consent_id, digest, size = (body.get(key) for key in
                                  ("consent_id", "expected_sha256", "expected_size_bytes"))
        intent = body.get("intent_id")
        if (set(body) != fields or not isinstance(intent, str) or not _SCENE_ID.fullmatch(intent)
                or not isinstance(consent_id, str) or not re.fullmatch("[0-9a-f]{32}", consent_id)
                or not isinstance(digest, str) or not _LEASE_DIGEST.fullmatch(digest)
                or type(size) is not int or not 1 <= size <= 512 * 1024
                or (kind == "retire-scene" and type(body["apply"]) is not bool)):
            raise RequestRefused("scene_lifecycle_options_invalid")
        return {key: body[key] for key in fields}
    if kind == "legacy-owner-census":
        if set(body) != {"kind"}:
            raise RequestRefused("legacy_owner_options_invalid")
        return {"kind": kind}
    if kind == "owner-census-decision":
        allowed = {"kind", "consent_id", "expected_sha256", "expected_size_bytes"}
        consent_id, digest, size = (body.get(k) for k in ("consent_id", "expected_sha256", "expected_size_bytes"))
        if (set(body) != allowed or not isinstance(consent_id, str) or len(consent_id) != 32
                or re.fullmatch(r"[0-9a-f]{32}", consent_id) is None
                or not isinstance(digest, str) or len(digest) != 71 or _LEASE_DIGEST.fullmatch(digest) is None
                or type(size) is not int or not 1 <= size <= 512 * 1024):
            raise RequestRefused("owner_consent_options_invalid")
        return {"kind": kind, "consent_id": consent_id, "expected_sha256": digest, "expected_size_bytes": size}
    if kind == "deploy":
        _only(body, ("kind", "commit", "wait_for_idle"))
        wait = body.get("wait_for_idle", True)
        if not isinstance(wait, bool):
            raise RequestRefused("wait_for_idle_invalid")
        return {"kind": kind, "commit": _commit(body), "wait_for_idle": wait}
    if kind == "unit":
        if body.get("action") == "repair-notifier-binding":
            _only(body, ("kind", "unit", "action", "expected_postcheck_sha256", "expected_source_commit"))
            digest, commit = body.get("expected_postcheck_sha256"), body.get("expected_source_commit")
            if (body.get("unit") != "blueprint-pipeline-control-plane.service"
                    or not isinstance(digest, str) or not _LEASE_DIGEST.fullmatch(digest)
                    or not isinstance(commit, str) or not _COMMIT.fullmatch(commit)):
                raise RequestRefused("notifier_repair_identity_invalid")
            return {"kind": kind, "unit": body["unit"], "action": body["action"],
                    "expected_postcheck_sha256": digest, "expected_source_commit": commit}
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
            safety = _BASELINE_TRIGGER_SAFETY if DISPATCHER_HOLD_ONLY else _SAFETY_CRITICAL
            if safety.search(unit):
                raise RequestRefused("unit_safety_critical")
            if action == "stop" and (
                not DISPATCHER_HOLD_ONLY or unit == "blueprint-agent-run-dispatcher.timer"
            ):
                raise RequestRefused("unit_stop_requires_hold")
        return {"kind": kind, "unit": unit, "action": action}
    if kind == "hold":
        _only(body, ("kind", "unit", "owner", "reason", "expires_in_seconds", "require_explicit_release"))
        unit = _hold_unit(body.get("unit"))
        owner, reason, duration = body.get("owner"), body.get("reason"), body.get("expires_in_seconds")
        if not isinstance(owner, str) or not _HOLD_OWNER.fullmatch(owner):
            raise RequestRefused("hold_owner_invalid")
        if not isinstance(reason, str) or not 1 <= len(reason) <= 200 or not reason.isprintable():
            raise RequestRefused("hold_reason_invalid")
        if type(duration) is not int or not 60 <= duration <= 86400:
            raise RequestRefused("hold_expiry_invalid")
        explicit = body.get("require_explicit_release", False)
        if type(explicit) is not bool:
            raise RequestRefused("hold_release_policy_invalid")
        if explicit and unit != "blueprint-agent-run-dispatcher.timer":
            raise RequestRefused("hold_explicit_release_unit_refused")
        return {"kind": kind, "unit": unit, "owner": owner, "reason": reason,
                "expires_in_seconds": duration, **({"require_explicit_release": True} if explicit else {})}
    if kind == "release-hold":
        _only(body, ("kind", "unit"))
        return {"kind": kind, "unit": _hold_unit(body.get("unit"))}
    if kind == "door-upgrade":
        _only(body, ("kind", "commit"))
        return {"kind": kind, "commit": _commit(body)}
    if kind == "lane-scratch":
        action = body.get("action")
        if action == "ls":
            _only(body, ("kind", "action", "root", "lane", "limit", "offset"))
        elif action in ("renew", "release"):
            _only(body, ("kind", "action", "root", "lane", "name", "owner", "expected_digest",
                         "ttl_seconds") if action == "renew" else
                  ("kind", "action", "root", "lane", "name", "owner", "expected_digest"))
        else:
            raise RequestRefused("lane_scratch_action_invalid")
        root = body.get("root")
        if root not in ("work", "inputs"):
            raise RequestRefused("lane_scratch_root_invalid")
        normalized = {"kind": kind, "action": action, "root": root,
                      "lane": _lane_part(body.get("lane"), "lane")}
        if action == "ls":
            limit, offset = body.get("limit", 50), body.get("offset", 0)
            if type(limit) is not int or not 1 <= limit <= 100:
                raise RequestRefused("lane_scratch_limit_invalid")
            if type(offset) is not int or not 0 <= offset <= 10000:
                raise RequestRefused("lane_scratch_offset_invalid")
            return {**normalized, "limit": limit, "offset": offset}
        normalized["name"] = _lane_part(body.get("name"), "name")
        normalized["owner"] = _lane_part(body.get("owner"), "owner")
        digest = body.get("expected_digest")
        if not isinstance(digest, str) or _LEASE_DIGEST.fullmatch(digest) is None:
            raise RequestRefused("lane_scratch_digest_invalid")
        normalized["expected_digest"] = digest
        if action == "renew":
            ttl = body.get("ttl_seconds")
            if type(ttl) is not int or not 0 < ttl <= 14 * 86400:
                raise RequestRefused("lane_scratch_ttl_invalid")
            normalized["ttl_seconds"] = ttl
        return normalized
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
    if kind == "provider-output-resume":
        _only(body, ("kind", "run", "attempt", "ingest"))
        run, attempt, ingest = body.get("run"), body.get("attempt"), body.get("ingest", False)
        if not isinstance(run, str) or not _CANARY_RUN.fullmatch(run) or run in {".", ".."}:
            raise RequestRefused("provider_output_resume_run_invalid")
        if type(attempt) is not int or not 1 <= attempt <= 999:
            raise RequestRefused("provider_output_resume_attempt_invalid")
        if not isinstance(ingest, bool):
            raise RequestRefused("provider_output_resume_ingest_invalid")
        return {"kind": kind, "run": run, "attempt": attempt, "ingest": ingest}
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


def _enqueue(config: DoorConfig, request: dict[str, Any], *, requested_by: str, request_id: str | None = None) -> str:
    normalized = validate_request(request)
    request_id = request_id or new_request_id(normalized["kind"])
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
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, pending / f"{request_id}.json")
        _sync_directory(pending)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return request_id


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def enqueue(config: DoorConfig, request: dict[str, Any], *, requested_by: str,
            operation_key: str | None = None) -> str:
    """Retain operation identity beyond spool retention; retry cannot relaunch it."""
    normalized = validate_request(request)
    if operation_key is None:
        return _enqueue(config, normalized, requested_by=requested_by)
    if not isinstance(operation_key, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{16,128}", operation_key):
        raise RequestRefused("operation_key_invalid")
    root = Path(config.spool_root) / "pending" / ".operations"
    root.mkdir(mode=0o750, exist_ok=True)
    if root.is_symlink():
        raise RequestRefused("operation_store_unsafe")
    identity = hashlib.sha256((requested_by + "\0" + operation_key).encode()).hexdigest()
    path = root / f"{identity}.json"
    lock = os.open(root / f"{identity}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o640)
    try:
        if not stat.S_ISREG(os.fstat(lock).st_mode):
            raise RequestRefused("operation_store_unsafe")
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            operation = load_request_file(path)
            if operation.get("requested_by") != requested_by or operation.get("request") != normalized:
                raise RequestRefused("operation_key_conflict")
            if operation.get("published"):
                return validate_request_id(operation["id"])
        else:
            operation = {"id": new_request_id(normalized["kind"]), "request": normalized,
                         "requested_by": requested_by, "published": False}
            _write_operation(path, operation)
        request_id = validate_request_id(operation["id"])
        # A crash after publication but before the marker is repaired with the
        # same id. The runner also refuses an already-completed id.
        if not any((Path(config.spool_root) / state / f"{request_id}.json").exists() for state in STATES):
            _enqueue(config, normalized, requested_by=requested_by, request_id=request_id)
        _write_operation(path, {**operation, "published": True})
        return request_id
    finally:
        os.close(lock)


def _write_operation(path: Path, value: dict[str, Any]) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".operation-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


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


def observed_unit_outcome(state: dict[str, Any]) -> dict[str, str] | None:
    """Observe only the recorded invocation. A later restart proves nothing
    about this operation, and historical acceptance is never promoted."""
    result = state.get("result") or {}
    bound = result.get("unit_observation") or {}
    invocation = bound.get("InvocationID")
    units = state.get("unit_state") or []
    if not invocation or len(units) != 1:
        return None
    unit = units[0]
    if unit.get("InvocationID") != invocation:
        return {"status": "unknown", "reason": "unit_invocation_changed"}
    if unit.get("ActiveState") == "failed":
        return {"status": "observed_failed", "invocation_id": invocation}
    if unit.get("ActiveState") == "inactive" and unit.get("Result") == "success" and unit.get("ExecMainStatus") == "0":
        return {"status": "observed_completed", "invocation_id": invocation}
    return None


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
