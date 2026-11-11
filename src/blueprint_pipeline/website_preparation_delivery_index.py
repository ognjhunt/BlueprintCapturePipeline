"""Recoverable due hints for the existing preparation delivery records.

This private index carries no source or delivery authority. Dispatch always
reopens the canonical delivery file, ledger and current signed source. Losing
an index entry is repaired by the listener's separate bounded ledger sweep.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .common import write_json

DIRECTORY = ".website_preparation_pending"
SCHEMA_VERSION = "website_preparation_delivery_hint.v1"
METRICS_FILE = ".website_preparation_delivery_metrics.json"
METRICS_SCHEMA_VERSION = "website_preparation_delivery_metrics.v1"
MAX_HINT_SCAN = 20000
_UNAVAILABLE_FIELDS = ("unavailable_failure_count", "unavailable_last_attempt_at_epoch",
                       "unavailable_next_attempt_at_epoch")
# D-05: the capacity tick pages (PAGE_ALERT_CODES) when a live, unparked callback
# has waited over an hour, whether it is due now or waiting out a retry backoff.
# Parked callbacks only warn. Nothing here admits, refuses or changes delivery.
STALLED_CODE = "preparation_delivery_stalled"
PARKED_CODE = "preparation_delivery_parked"
METRICS_UNREADABLE_CODE = "preparation_delivery_metrics_unreadable"
PAGE_CODES = frozenset({STALLED_CODE})
LIVE_PAGE_SECONDS = 3600
# The listener writes the metrics every minute; an hour without them means it stopped.
METRICS_STALE_SECONDS = 3600
MAX_METRICS_BYTES = 256 * 1024
STORAGE_ROOT_ENV = "BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT"
DEFAULT_STORAGE_ROOT = Path("/var/lib/blueprint/pubsub-handoffs")


@dataclass(frozen=True)
class DueDelivery:
    capture_root: Path
    # Exact selected hint for compare-and-swap deferral. It conveys no authority.
    hint: Mapping[str, Any]


def _location(capture_root: Path) -> tuple[Path, str]:
    root = Path(capture_root).absolute()
    if (root.parent.name != "captures" or root.parents[2].name != "scenes"
            or not root.parents[1].name.startswith("site-")
            or not root.name.startswith(("walkthrough-", "supplement-"))
            or any(p.is_symlink() for p in (root, *root.parents))):
        raise ValueError("preparation_delivery_index_root_invalid")
    storage = root.parents[4]
    return storage, root.relative_to(storage).as_posix()


@contextmanager
def _locked(storage_root: Path):
    directory = Path(storage_root) / DIRECTORY
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("preparation_delivery_index_unsafe")
    descriptor = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("preparation_delivery_index_unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield directory
    finally:
        os.close(descriptor)


def _number(value: Any, default: float) -> float:
    if type(value) not in (int, float):
        return default
    try:
        number = float(value)
    except OverflowError:
        return default
    return number if math.isfinite(number) and number >= 0 else default


def _unavailable_delay(failures: int) -> int:
    return min(1800, 60 * 2 ** (failures - 1))


def _unavailable_backoff(hint: Mapping[str, Any], *, now: float) -> dict[str, Any]:
    """Only a bounded, self-consistent retained failure may delay selection."""
    failures = hint.get("unavailable_failure_count")
    attempted = _number(hint.get("unavailable_last_attempt_at_epoch"), -1.0)
    deadline = _number(hint.get("unavailable_next_attempt_at_epoch"), -1.0)
    if (type(failures) is not int or not 1 <= failures <= 6 or attempted < 0
            or attempted > now or deadline > now + 1800
            or deadline != attempted + _unavailable_delay(failures)):
        return {}
    return {key: hint[key] for key in _UNAVAILABLE_FIELDS}


def update(capture_root: Path, delivery: Mapping[str, Any]) -> None:
    storage, relative = _location(capture_root)
    key = hashlib.sha256(relative.encode()).hexdigest()
    with _locked(storage) as directory:
        path = directory / (key + ".json")
        if path.is_symlink():
            raise ValueError("preparation_delivery_index_unsafe")
        if delivery.get("state") != "pending":
            # A newer revision's hint must survive a late acknowledgement.
            try:
                old = json.loads(path.read_text())
            except (FileNotFoundError, ValueError):
                return
            if old.get("revision") == delivery.get("revision") or (
                    # A retired record supersedes an older revision's hint too (website_preparation_status).
                    delivery.get("state") == "retired" and type(old.get("revision")) is int
                    and type(delivery.get("revision")) is int and old["revision"] < delivery["revision"]):
                path.unlink()
            return
        hint = {"schema_version": SCHEMA_VERSION, "capture_relative_path": relative,
                "revision": delivery.get("revision"),
                "next_attempt_at_epoch": _number(delivery.get("next_attempt_at_epoch"), 0.0),
                "pending_since_epoch": _number(delivery.get("pending_since_epoch"), 0.0)}
        parked_at = _number(delivery.get("parked_at_epoch"), -1.0)
        if parked_at >= 0:
            # Parked: counted apart from the due backlog (website_preparation_status).
            hint["parked_at_epoch"] = parked_at
        try:
            previous = json.loads(path.read_text())
        except (FileNotFoundError, ValueError):
            previous = None
        if (type(previous) is dict and previous.get("schema_version") == SCHEMA_VERSION
                and previous.get("capture_relative_path") == relative
                and previous.get("revision") == hint["revision"]):
            # The bounded ledger sweep must not reset an unreadable record's
            # scheduling backoff. A new canonical revision starts promptly.
            hint.update(_unavailable_backoff(previous, now=time.time()))
        write_json(path, hint)


def defer_unavailable(selected: DueDelivery, *, now: float) -> None:
    """Defer only the selected hint; never repair or acknowledge delivery bytes."""
    storage, relative = _location(selected.capture_root)
    key = hashlib.sha256(relative.encode()).hexdigest()
    with _locked(storage) as directory:
        path = directory / (key + ".json")
        if path.is_symlink():
            raise ValueError("preparation_delivery_index_unsafe")
        try:
            current = json.loads(path.read_text())
        except (FileNotFoundError, ValueError):
            return
        if current != selected.hint:
            return  # A newer revision/selection owns its own schedule.
        backoff = _unavailable_backoff(current, now=now)
        failures = min(6, backoff.get("unavailable_failure_count", 0) + 1)
        write_json(path, {**current, "unavailable_failure_count": failures,
                         "unavailable_last_attempt_at_epoch": now,
                         "unavailable_next_attempt_at_epoch": now + _unavailable_delay(failures)})


def due(storage_root: Path, *, now: float, limit: int) -> tuple[list[DueDelivery], dict[str, Any]]:
    """Due rows, live before parked, and backlog metrics.

    ``due_backlog``, ``oldest_due_age_seconds`` and ``overloaded`` count live rows
    due now. ``live_count`` and ``oldest_live_age_seconds`` add live rows waiting
    out a retry backoff. Parked rows count only in ``parked_count`` and
    ``oldest_parked_age_seconds``. Every age runs from ``pending_since_epoch``.
    """
    storage = Path(storage_root).absolute()
    rows = []
    live_ages: list[float] = []
    parked_ages: list[float] = []
    invalid = scanned = 0
    with _locked(storage) as directory:
        for path in directory.iterdir():
            if not re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
                continue
            if scanned >= MAX_HINT_SCAN:
                break
            scanned += 1
            try:
                if path.is_symlink():
                    raise ValueError("unsafe")
                hint = json.loads(path.read_text())
                relative = hint["capture_relative_path"]
                root = storage / relative
                actual_storage, actual_relative = _location(root)
                if (hint.get("schema_version") != SCHEMA_VERSION or actual_storage != storage
                        or actual_relative != relative
                        or path.stem != hashlib.sha256(relative.encode()).hexdigest()):
                    raise ValueError("invalid")
                backoff = _unavailable_backoff(hint, now=now)
                if not backoff and any(key in hint for key in _UNAVAILABLE_FIELDS):
                    invalid += 1
                next_attempt = max(_number(hint.get("next_attempt_at_epoch"), 0.0),
                                   backoff.get("unavailable_next_attempt_at_epoch", 0.0))
                pending_since = _number(hint.get("pending_since_epoch"), now)
                parked = _number(hint.get("parked_at_epoch"), -1.0) >= 0
                (parked_ages if parked else live_ages).append(max(0.0, now - pending_since))
                if next_attempt <= now:
                    # Zero means immediately eligible, not older than every
                    # retained retry. Compare real eligibility times so a
                    # stream of new revisions cannot starve an already-due one.
                    # A parked retry waits behind every live row that is due.
                    rows.append((parked, max(next_attempt, pending_since), pending_since, root, hint))
            except (OSError, ValueError, KeyError, TypeError):
                invalid += 1
    rows.sort(key=lambda row: (row[0], row[1], row[2], str(row[3])))
    live_due = [row for row in rows if not row[0]]
    return [DueDelivery(row[3], row[4]) for row in rows[:limit]], {
        "due_backlog": len(live_due), "oldest_due_age_seconds": max([0.0, *(now - row[2] for row in live_due)]),
        "live_count": len(live_ages), "oldest_live_age_seconds": max([0.0, *live_ages]),
        "parked_count": len(parked_ages), "oldest_parked_age_seconds": max([0.0, *parked_ages]),
        "hint_count_scanned": scanned, "hint_scan_limit_reached": scanned >= MAX_HINT_SCAN,
        "invalid_hint_count": invalid,
        "overloaded": len(live_due) > limit or scanned >= MAX_HINT_SCAN}


def _read_metrics(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_METRICS_BYTES:
            raise ValueError("preparation_delivery_metrics_unsafe")
        raw = os.read(descriptor, MAX_METRICS_BYTES + 1)
    finally:
        os.close(descriptor)
    value = json.loads(raw)
    if type(value) is not dict or value.get("schema_version") != METRICS_SCHEMA_VERSION:
        raise ValueError("preparation_delivery_metrics_invalid")
    return value


def observe_delivery_backlog(storage_root: Path, *, now: float) -> dict[str, Any]:
    """Read-only: the listener's last delivery metrics as capacity-report alert rows.

    A live callback older than ``LIVE_PAGE_SECONDS`` pages; parked callbacks
    warn. Metrics the listener stopped writing age by the silence, and say so.
    """
    try:
        metrics = _read_metrics(Path(storage_root) / METRICS_FILE)
    except FileNotFoundError:
        return {"status": "absent", "alerts": []}
    except (OSError, ValueError):
        return {"status": "unreadable",
                "alerts": [{"code": METRICS_UNREADABLE_CODE, "reason": "metrics_unreadable"}]}
    counts = {key: metrics.get(key) for key in ("live_count", "parked_count")}
    ages = {key: metrics.get(key) for key in
            ("observed_at_epoch", "oldest_live_age_seconds", "oldest_parked_age_seconds")}
    if (metrics.get("index_unavailable") is True
            or not all(type(value) is int and value >= 0 for value in counts.values())
            or not all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                       for value in ages.values())):
        reason = "index_unavailable" if metrics.get("index_unavailable") is True else "metrics_incomplete"
        return {"status": "incomplete", "alerts": [{"code": METRICS_UNREADABLE_CODE, "reason": reason}]}
    silent = max(0.0, now - float(ages["observed_at_epoch"]))
    live_age = float(ages["oldest_live_age_seconds"]) + silent if counts["live_count"] else 0.0
    parked_age = float(ages["oldest_parked_age_seconds"]) + silent if counts["parked_count"] else 0.0
    alerts: list[dict[str, Any]] = []
    if counts["live_count"] and live_age > LIVE_PAGE_SECONDS:
        alerts.append({"code": STALLED_CODE, "count": counts["live_count"], "age_seconds": round(live_age),
                       "threshold_seconds": LIVE_PAGE_SECONDS, "parked_count": counts["parked_count"]})
    if counts["parked_count"]:
        alerts.append({"code": PARKED_CODE, "count": counts["parked_count"], "age_seconds": round(parked_age)})
    if silent > METRICS_STALE_SECONDS:
        alerts.append({"code": METRICS_UNREADABLE_CODE, "reason": "metrics_stale", "age_seconds": round(silent)})
    return {"status": "observed", "observed_at_epoch": ages["observed_at_epoch"], **counts,
            "oldest_live_age_seconds": live_age, "oldest_parked_age_seconds": parked_age, "alerts": alerts}


def configured_delivery_check() -> Callable[[float], dict[str, Any]]:
    """The capacity service's check, on the listener's storage root."""
    configured = str(os.getenv(STORAGE_ROOT_ENV) or "").strip()
    root = Path(configured) if configured.startswith("/") else DEFAULT_STORAGE_ROOT
    return lambda now: observe_delivery_backlog(root, now=now)
