"""Vast credit held by admitted launches, so concurrent launch slots share one balance.

Before (G18), Vast credit admission (``provider_credit_admission``) compared
each launch's hard cap with the balance alone. With three launch slots, three
launches could each see the same balance and together be admitted for three
times what it covers.

Now every admitted launch leaves a hold for its hard cap in one small file
beside the launch slot locks (``vast_credit_holds.v1.json``, or
``BLUEPRINT_VAST_CREDIT_HOLD_ROOT``). Admission reads the open holds, checks
``balance >= cap + reserve + open holds`` and records its own hold, all under
one exclusive lock, so the sum of admitted caps never passes the balance less
the reserve. A check that only reads (an early check before staging) subtracts
the open holds without taking one.

A hold closes by itself when its launch's own teardown manifest, written after
the hold, says nothing continues to spend
(``continuing_spend_from_this_run: false``): a launch refused before
allocation writes one too. A retained instance (``true``) keeps its hold. A
hold whose launch left no such manifest (a process killed mid-launch) closes
after ``MAX_HOLD_AGE_SECONDS`` and is reported as ``expired_without_teardown``:
by then Vast has debited what the instance ran from the balance itself.
``release_hold`` closes one at once (for a ledger that settles the launch).

A holds file that cannot be read refuses new credit while it could still hold
an open hold. Once its own modification time is older than
``MAX_HOLD_AGE_SECONDS``, every hold it could hold has expired: it is set aside
(renamed ``vast_credit_holds.v1.json.unreadable-<epoch>``, kept for inspection)
and admission proceeds with no holds, saying so (``holds_file_set_aside``).

Holds over-count: a running launch's spend so far is already out of the
balance and still held at its whole cap. That can only refuse earlier. No
amount here is a money value of its own; caps and the reserve come from the
caller.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .common import write_json

HOLD_ROOT_ENV = "BLUEPRINT_VAST_CREDIT_HOLD_ROOT"
HOLDS_FILENAME = "vast_credit_holds.v1.json"
LOCK_FILENAME = "vast_credit_holds.lock"
SCHEMA = "blueprint.vast_credit_holds.v1"
TEARDOWN_FILENAME = "vast_teardown_manifest.json"
#: A launch's own watchdog and TTL end it well inside a day; a hold never outlives one.
MAX_HOLD_AGE_SECONDS = 24 * 60 * 60
HOLDS_UNREADABLE = "provider_credit_holds_unreadable"
_LAUNCH_LOCK_FILE_ENV = "VAST_LAUNCH_LOCK_FILE"
_API_KEY_FILE_ENV = "VAST_API_KEY_FILE"
_DEFAULT_API_KEY_FILE = "~/.blueprint-secrets/vast_api_key"


def hold_root() -> Path:
    """Beside the launch slot locks, which every launching unit already shares."""

    configured = str(os.environ.get(HOLD_ROOT_ENV) or "").strip()
    if configured:
        return Path(configured)
    # The rule of ``vast_launch_slots._vast_launch_lock_path``, read here without importing the
    # provider adapter (the control plane must not); a test pins the two together.
    lock = str(os.environ.get(_LAUNCH_LOCK_FILE_ENV) or "").strip()
    if lock:
        return Path(lock).expanduser().resolve().parent
    return Path(os.environ.get(_API_KEY_FILE_ENV, _DEFAULT_API_KEY_FILE)).expanduser().resolve().parent


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / LOCK_FILENAME, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _read(root: Path) -> list[dict[str, Any]]:
    """The recorded holds; ValueError when the file exists but is not a holds file."""

    path = root / HOLDS_FILENAME
    if not path.exists() and not path.is_symlink():
        return []
    if path.is_symlink():
        raise ValueError(HOLDS_UNREADABLE)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA or not isinstance(value.get("holds"), list):
        raise ValueError(HOLDS_UNREADABLE)
    return [dict(row) for row in value["holds"] if isinstance(row, Mapping)]


def _amount(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def hold_state(row: Mapping[str, Any], *, now: float) -> str:
    """``open``, ``closed`` (its teardown proves no continuing spend) or ``expired_without_teardown``."""

    held_at = _amount(row.get("held_at_epoch"))
    if held_at is None or _amount(row.get("held_usd")) is None:
        return "open"  # an unreadable row keeps counting: money fails closed
    manifest = Path(str(row.get("job_dir") or "")) / TEARDOWN_FILENAME
    try:
        written = manifest.stat().st_mtime
        teardown = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        teardown = None
    if (isinstance(teardown, Mapping) and written >= held_at
            and teardown.get("continuing_spend_from_this_run") is False):
        return "closed"
    return "expired_without_teardown" if now - held_at > MAX_HOLD_AGE_SECONDS else "open"


def _open_rows(rows: list[dict[str, Any]], *, now: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept, expired = [], []
    for row in rows:
        state = hold_state(row, now=now)
        if state == "open":
            kept.append(row)
        elif state == "expired_without_teardown":
            expired.append(row)
    return kept, expired


def _total(rows: list[dict[str, Any]]) -> float:
    return round(sum(_amount(row.get("held_usd")) or 0.0 for row in rows), 6)


def _set_aside_if_expired(root: Path, *, now: float) -> str | None:
    """Under the lock: rename an unreadable holds file no open hold could still be in; its new name."""

    path = root / HOLDS_FILENAME
    try:
        modified = path.lstat().st_mtime
    except OSError:
        return None
    if now - modified <= MAX_HOLD_AGE_SECONDS:
        return None
    target = root / f"{HOLDS_FILENAME}.unreadable-{int(now)}"
    os.replace(path, target)
    return str(target)


def _load(root: Path, *, now: float) -> tuple[list[dict[str, Any]], str | None]:
    """Under the lock: the holds, or none when an old unreadable file was set aside; ValueError if young."""

    try:
        return _read(root), None
    except (OSError, ValueError, UnicodeDecodeError):
        aside = _set_aside_if_expired(root, now=now)
        if aside is None:
            raise ValueError(HOLDS_UNREADABLE) from None
        return [], aside


def open_holds_usd(*, root: Path | None = None, now: float | None = None) -> tuple[float | None, list[str]]:
    """What open holds keep from the balance, read only; ``(None, [blocker])`` when unreadable."""

    base = hold_root() if root is None else root
    moment = time.time() if now is None else now
    try:
        rows = _read(base)
    except (OSError, ValueError, UnicodeDecodeError):
        try:
            with _locked(base):
                rows, _aside = _load(base, now=moment)
        except (OSError, ValueError):
            return None, [HOLDS_UNREADABLE]
    return _total(_open_rows(rows, now=moment)[0]), []


def admit_and_hold(decide: Any, *, job_dir: Path, required_usd: float, root: Path | None = None,
                   now: float | None = None) -> dict[str, Any]:
    """Run ``decide(held_usd)`` under the holds lock and record a hold when it admits.

    ``decide`` is the credit check with the open holds subtracted
    (``provider_credit_admission.credit_admission(..., held_usd=...)``). A hold
    this same job directory left earlier is replaced, never counted twice.
    """

    base = hold_root() if root is None else root
    moment = time.time() if now is None else now
    job = str(Path(job_dir).resolve())
    try:
        with _locked(base):
            try:
                rows, aside = _load(base, now=moment)
            except (OSError, ValueError):
                return {"status": "blocked", "blockers": [HOLDS_UNREADABLE], "credit_reserved": False}
            others = [row for row in rows if row.get("job_dir") != job]
            kept, expired = _open_rows(others, now=moment)
            held, counted = _total(kept), len(kept)
            result = dict(decide(held))
            if result.get("status") == "admitted":
                kept.append({"job_dir": job, "held_usd": float(required_usd), "held_at_epoch": moment,
                             "pid": os.getpid()})
                result["credit_reserved"] = True
            write_json(base / HOLDS_FILENAME, {"schema_version": SCHEMA, "holds": kept})
    except OSError:
        return {"status": "blocked", "blockers": ["provider_credit_holds_unavailable"], "credit_reserved": False}
    if aside is not None:
        result["holds_file_set_aside"] = aside
    result.update(open_holds_usd=held, open_holds=counted,
                  expired_holds=[{"job_dir": row.get("job_dir"), "held_usd": row.get("held_usd"),
                                  "state": "expired_without_teardown"} for row in expired])
    return result


def release_hold(job_dir: Path, *, root: Path | None = None) -> bool:
    """Close this launch's hold now (its spend is settled elsewhere); ``True`` when one was open."""

    base = hold_root() if root is None else root
    job = str(Path(job_dir).resolve())
    with _locked(base):
        rows = _read(base)
        kept = [row for row in rows if row.get("job_dir") != job]
        if len(kept) == len(rows):
            return False
        write_json(base / HOLDS_FILENAME, {"schema_version": SCHEMA, "holds": kept})
        return True


__all__ = ["HOLD_ROOT_ENV", "HOLDS_UNREADABLE", "MAX_HOLD_AGE_SECONDS", "admit_and_hold", "hold_root",
           "hold_state", "open_holds_usd", "release_hold"]
