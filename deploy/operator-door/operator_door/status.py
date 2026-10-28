"""One status document: what a run snapshot on the host used to take a dozen commands.

Each section is computed independently; a section that fails reports
``{"error": ...}`` instead of failing the whole document, because a status call
is most useful precisely when something on the host is broken.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
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
_PREFLIGHT_SCHEMA = "task_evaluation_production_chain_preflight_summary.v1"
_PREFLIGHT_KEYS = frozenset(("schema_version", "generated_at", "observed_at_epoch", "active_sha",
                           "report_sha256", "configuration_sha256", "lane", "coverage", "counts",
                           "findings", "proof_boundary", "summary_sha256"))
_BETA_TRIGGERS = frozenset((
    "blueprint-pubsub-handoff-listener.timer", "blueprint-team-scene-packet.path",
    "blueprint-rigid-robot-free-probe-runner.path", "blueprint-rigid-robot-free-probe-runner.timer",
    "blueprint-beta-policy-screening.timer",
))


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


class PreflightSummaryRefused(ValueError):
    """A fixed reason code, with no report content in the error."""


def _anchored_summary_bytes(path: Path, limit: int) -> bytes:
    """Walk from a directory FD; no component is ever reopened by full path."""
    if not path.is_absolute() or ".." in path.parts or not path.name:
        raise PreflightSummaryRefused("preflight_summary_path_invalid")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directories: list[tuple[int | None, str, int]] = []
    descriptor = None

    def check_links() -> None:
        for parent_fd, name, child_fd in directories[1:]:
            linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            opened = os.fstat(child_fd)
            if not stat.S_ISDIR(linked.st_mode) or (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino):
                raise PreflightSummaryRefused("preflight_summary_directory_identity_changed")

    try:
        directories.append((None, "/", os.open("/", flags)))
        for name in path.parent.parts[1:]:
            parent_fd = directories[-1][2]
            directories.append((parent_fd, name, os.open(name, flags, dir_fd=parent_fd)))
        check_links()
        parent_fd = directories[-1][2]
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent_fd)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise PreflightSummaryRefused("preflight_summary_file_unsafe")
        data = os.read(descriptor, limit + 1)
        after = os.fstat(descriptor)
        linked = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        check_links()
        if (not stat.S_ISREG(linked.st_mode) or (linked.st_dev, linked.st_ino) != (before.st_dev, before.st_ino)
                or (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns)
                or len(data) != before.st_size or len(data) > limit):
            raise PreflightSummaryRefused("preflight_summary_file_identity_changed")
        return data
    except OSError:
        raise PreflightSummaryRefused("preflight_summary_symlink_or_open_refused") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for _parent, _name, directory_fd in reversed(directories):
            os.close(directory_fd)


def _preflight(config: DoorConfig) -> dict[str, Any]:
    """Read only the explicit safe projection, never the root-owned raw report."""
    try:
        path = Path(config.preflight_summary)
        raw = _anchored_summary_bytes(path, 128 * 1024)
        if scan_bytes(raw) is not None:
            raise PreflightSummaryRefused("preflight_summary_secret_shaped")
        document = json.loads(raw)
        if not isinstance(document, dict) or set(document) != _PREFLIGHT_KEYS or document.get("schema_version") != _PREFLIGHT_SCHEMA:
            raise PreflightSummaryRefused("preflight_summary_schema_invalid")
        for field in ("report_sha256", "configuration_sha256", "summary_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(document.get(field) or "")):
                raise PreflightSummaryRefused("preflight_summary_digest_invalid")
        content = {key: value for key, value in document.items() if key != "summary_sha256"}
        encoded = (json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if hashlib.sha256(encoded).hexdigest() != document["summary_sha256"]:
            raise PreflightSummaryRefused("preflight_summary_digest_mismatch")
        observed = document.get("observed_at_epoch")
        if type(observed) not in (int, float) or not math.isfinite(observed) or not -300 <= time.time() - observed <= 3600:
            raise PreflightSummaryRefused("preflight_summary_stale_or_future")
        generated = _dt.datetime.strptime(document["generated_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc).timestamp()
        if generated != observed:
            raise PreflightSummaryRefused("preflight_summary_timestamp_mismatch")
        release = _active_release(config).get("commit")
        if not re.fullmatch(r"[0-9a-f]{40}", str(release or "")) or document["active_sha"] != release:
            raise PreflightSummaryRefused("preflight_summary_release_mismatch")
        if document["lane"] not in {"diagnostic", "beta_rigid", "unknown"} or document["proof_boundary"] != "selected_route_plumbing_only":
            raise PreflightSummaryRefused("preflight_summary_scope_invalid")
        counts, coverage, rows = document["counts"], document["coverage"], document["findings"]
        if not isinstance(counts, dict) or set(counts) != {"blocker", "warning", "info"} or not all(type(value) is int and value >= 0 for value in counts.values()):
            raise PreflightSummaryRefused("preflight_summary_counts_invalid")
        if not isinstance(coverage, dict) or set(coverage) != {"status", "unit_count", "unknown_units"} or coverage.get("status") not in {"selected_route_contract", "unknown_or_partial"} or not all(type(coverage.get(key)) is int and coverage[key] >= 0 for key in ("unit_count", "unknown_units")):
            raise PreflightSummaryRefused("preflight_summary_coverage_invalid")
        if not isinstance(rows, list) or len(rows) > 1000:
            raise PreflightSummaryRefused("preflight_summary_findings_invalid")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"severity", "code", "count", "units", "category", "owner", "next_action"}:
                raise PreflightSummaryRefused("preflight_summary_finding_invalid")
            if row["severity"] not in counts or type(row["count"]) is not int or row["count"] < 1:
                raise PreflightSummaryRefused("preflight_summary_finding_invalid")
            if not all(isinstance(row[key], str) and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", row[key]) for key in ("code", "category", "owner", "next_action")):
                raise PreflightSummaryRefused("preflight_summary_finding_invalid")
            if not isinstance(row["units"], list) or len(row["units"]) > 128 or not all(isinstance(unit, str) and (unit in {"host", "unrecognized_unit"} or re.fullmatch(r"blueprint-[a-z0-9-]{1,100}\.(?:service|timer|path)", unit)) for unit in row["units"]):
                raise PreflightSummaryRefused("preflight_summary_units_invalid")
        return {"status": "observed", **document}
    except PreflightSummaryRefused as error:
        return {"status": "refused", "reason": str(error)}
    except (OSError, ValueError, TypeError, KeyError):
        return {"status": "unavailable", "reason": "preflight_summary_unreadable_or_invalid"}


def _handoff_delivery(config: DoorConfig) -> dict[str, Any]:
    raw = _regular_bytes(Path(config.handoff_delivery_metrics), _MAX_JSON_BYTES)
    if scan_bytes(raw) is not None:
        raise SecretContentRefused("secret_content")
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get("schema_version") != "website_preparation_delivery_metrics.v1":
        raise ValueError("delivery_metrics_schema_invalid")
    keys = ("observed_at_epoch", "attempted", "delivered", "unavailable", "due_backlog",
            "oldest_due_age_seconds", "hint_count_scanned", "hint_scan_limit_reached", "invalid_hint_count",
            "overloaded", "repair_scanned", "successful_delivery_lag_seconds")
    safe: dict[str, Any] = {"schema_version": document["schema_version"]}
    for key in keys:
        value = document.get(key)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0 or type(value) is bool:
            safe[key] = value
    return safe


def _screening_progress(config: DoorConfig) -> dict[str, Any]:
    raw = _regular_bytes(Path(config.screening_worker_status), 32 * 1024)
    if scan_bytes(raw) is not None:
        raise SecretContentRefused("secret_content")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema_version") != "blueprint.beta_policy_screening_worker_status.v1":
        raise ValueError("screening_status_schema_invalid")
    safe = {"schema_version": value["schema_version"]}
    for key in ("status", "coverage", "reason"):
        candidate = value.get(key)
        if candidate is None or isinstance(candidate, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", candidate):
            safe[key] = candidate
    for key in ("observed_at_epoch", "campaign_count", "publication_pending", "oldest_eligible_wait_seconds",
                "last_ledger_progress_at_epoch", "last_successful_publication_at_epoch"):
        candidate = value.get(key)
        if candidate is None or type(candidate) in (int, float) and math.isfinite(candidate) and candidate >= 0:
            safe[key] = candidate
    for key in ("candidate_counts", "blocker_categories"):
        candidates = value.get(key)
        if isinstance(candidates, dict):
            safe[key] = {name: count for name, count in candidates.items()
                         if isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", name) and type(count) is int and count >= 0}
    failure = value.get("last_publication_failure")
    if isinstance(failure, dict) and failure.get("category") in {"auth", "source", "ack", "unavailable", "unknown"}:
        safe["last_publication_failure"] = {key: failure.get(key) for key in ("category", "owner", "next_action", "next_retry_at_epoch", "observed_at_epoch")
            if failure.get(key) is None or type(failure.get(key)) in (int, float) and math.isfinite(failure[key])
            or isinstance(failure.get(key), str) and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", failure[key])}
    return safe


def _unit_health(controller_units: Any, failed_units: Any) -> dict[str, Any]:
    if not isinstance(controller_units, list) or not isinstance(failed_units, list):
        return {"status": "unknown", "reason": "unit_inventory_unavailable"}
    known = {row["Id"] for row in controller_units if isinstance(row, dict) and isinstance(row.get("Id"), str)}
    rows = []
    for row in controller_units:
        if not isinstance(row, dict):
            continue
        name = row.get("Id")
        if row.get("ActiveState") == "failed" or name in failed_units:
            state = "required_unit_failed"
        elif name in _BETA_TRIGGERS and row.get("ActiveState") != "active":
            state = "required_trigger_inactive"
        elif row.get("ActiveState") == "inactive" and row.get("Type") == "oneshot" and row.get("Result") == "success":
            state = "successful_inactive_oneshot"
        elif row.get("ActiveState") == "active":
            state = "active"
        else:
            state = "unknown_or_inactive"
        rows.append({"unit": name, "state": state, "invocation_id": row.get("InvocationID")})
    return {"status": "observed", "current": rows,
            "historical_diagnostics": [name for name in failed_units if name not in known]}


def _door_requests(config: DoorConfig) -> dict[str, int]:
    spool = Path(config.spool_root)
    return {state: len(list((spool / state).glob("*.json"))) for state in ("pending", "processing")}


def _holds(config: DoorConfig) -> list[dict[str, Any]]:
    now = time.time()
    records = hold_records.active(Path(config.spool_root) / "holds")
    keys = ("unit", "owner", "reason", "requested_by", "request_id", "created_at", "expires_at")
    return [{**{key: record[key] for key in keys},
             "require_explicit_release": record.get("require_explicit_release", False),
             "remaining_seconds": max(0, int(record["expires_at_epoch"] - now)),
             "expired": record["expires_at_epoch"] <= now} for record in records]


def build_status(config: DoorConfig, host: HostInfo, *, caller: dict[str, Any]) -> dict[str, Any]:
    state = Path(config.control_plane_state)
    failed = _section(lambda: [row["unit"] for row in host.list_units("blueprint-*", states=("failed",))], "failed_units_unavailable")
    controllers = _section(lambda: host.unit_properties(config.controller_units), "controller_units_unavailable")
    return {
        "schema": SCHEMA,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "door": {"version": VERSION, "caller": caller,
                 "request_kinds": sorted(_SCOPES),
                 "hold_target": "blueprint-agent-run-dispatcher.timer" if DISPATCHER_HOLD_ONLY else None},
        "deployed": _deployed(host),
        "notifier_binding": _section(lambda: host.notifier_binding(), "notifier_binding_unavailable"),
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
        "failed_units": failed,
        "controller_units": controllers,
        "unit_health": _unit_health(controllers, failed),
        "preflight": _preflight(config),
        "preparation_delivery": _section(lambda: _handoff_delivery(config), "preparation_delivery_unavailable"),
        "beta_screening": _section(lambda: _screening_progress(config), "beta_screening_unavailable"),
        "disk": _section(lambda: host.disk(("/", "/var/lib/blueprint", "/mnt/blueprint-work")),
                         "disk_unavailable"),
        "capacity": _section(lambda: _capacity(config), "capacity_unavailable"),
        "load": _section(host.load, "load_unavailable"),
        "door_requests": _section(lambda: _door_requests(config), "door_requests_unavailable"),
        "holds": _section(lambda: _holds(config), "holds_unavailable"),
        "break_glass": _section(lambda: _break_glass(config), "break_glass_unavailable"),
    }
