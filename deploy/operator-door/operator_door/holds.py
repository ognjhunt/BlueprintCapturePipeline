"""Root-owned timer hold records shared by the runner, expiry unit and status."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from .requests import RequestRefused, validate_request, validate_request_id

SCHEMA = "blueprint_operator_door_hold.v1"
MAX_RECORD_BYTES = 16 * 1024


class HoldError(ValueError):
    """A hold record or directory is unsafe or malformed."""


def timestamp(epoch: float | None = None) -> str:
    return dt.datetime.fromtimestamp(epoch if epoch is not None else time.time(), dt.timezone.utc).isoformat(
        timespec="seconds"
    )


def _directory(root: Path, *, create: bool) -> None:
    if create:
        root.mkdir(mode=0o755, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise HoldError("holds_directory_unsafe")


@contextlib.contextmanager
def locked(root: Path) -> Iterator[None]:
    """Serialize runner changes with old expiry timers from a renewed hold."""

    _directory(root, create=True)
    fd = os.open(root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise HoldError("holds_lock_unsafe")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def read(root: Path, unit: str) -> dict[str, Any] | None:
    path = root / f"{unit}.json"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise HoldError("hold_record_unsafe") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RECORD_BYTES:
            raise HoldError("hold_record_unsafe")
        raw = os.read(fd, MAX_RECORD_BYTES + 1)
    finally:
        os.close(fd)
    if len(raw) > MAX_RECORD_BYTES:
        raise HoldError("hold_record_unsafe")
    try:
        record = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HoldError("hold_record_invalid") from exc
    if (
        not isinstance(record, dict)
        or record.get("schema") != SCHEMA
        or record.get("unit") != unit
        or record.get("status") not in {"active", "releasing", "released", "expired_released", "failed_released"}
        or not isinstance(record.get("owner"), str)
        or type(record.get("expires_at_epoch")) is not int
        or ("enabled_before" in record and type(record["enabled_before"]) is not bool)
        or (record.get("status") == "releasing" and record.get("release_status")
            not in {"released", "expired_released", "failed_released"})
    ):
        raise HoldError("hold_record_invalid")
    try:
        validate_request_id(record.get("request_id"))
    except RequestRefused as exc:
        raise HoldError("hold_record_invalid") from exc
    return record


def write(root: Path, unit: str, record: dict[str, Any]) -> None:
    _directory(root, create=True)
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    if len(payload) > MAX_RECORD_BYTES:
        raise HoldError("hold_record_too_large")
    fd, temporary = tempfile.mkstemp(prefix=f".{unit}.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / f"{unit}.json")
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def active(root: Path) -> list[dict[str, Any]]:
    """Records still marked active, including an overdue failed expiry."""

    if not root.exists():
        return []
    _directory(root, create=False)
    records = []
    for path in sorted(root.glob("*.json")):
        unit = path.name.removesuffix(".json")
        try:
            validate_request({"kind": "release-hold", "unit": unit})
        except RequestRefused as exc:
            raise HoldError("hold_record_invalid") from exc
        record = read(root, unit)
        if record is not None and record["status"] == "active":
            records.append(record)
    return records


def begin_release(
    root: Path, unit: str, record: dict[str, Any], *, released_by: str,
    status: str, now: float | None = None,
) -> None:
    """Persist recovery intent, then remove the active systemd guard."""

    if status not in {"released", "expired_released", "failed_released"}:
        raise HoldError("hold_release_status_invalid")
    current = read(root, unit)
    if current is None or current["request_id"] != record["request_id"] or current["status"] != "active":
        raise HoldError("hold_release_generation_changed")
    if read(root / "releasing", unit) is not None:
        raise HoldError("hold_release_in_progress")
    pending = {**record, "status": "releasing", "release_status": status,
               "released_by": released_by, "released_at": timestamp(now)}
    write(root / "releasing", unit, pending)
    (root / f"{unit}.json").unlink(missing_ok=True)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def finish_release(
    root: Path, unit: str, *, command: Any = None,
) -> int:
    """Idempotently finish a release begun before a crash or reboot."""

    try:
        validate_request({"kind": "release-hold", "unit": unit})
    except RequestRefused as exc:
        raise HoldError("hold_record_invalid") from exc
    if (root / "releasing").exists():
        _directory(root / "releasing", create=False)
    record = read(root / "releasing", unit)
    if record is None:
        return 0
    current = read(root, unit)
    if current is not None:
        if current["request_id"] != record["request_id"]:
            return 1
        (root / f"{unit}.json").unlink()
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    run = command or (lambda argv: subprocess.run(argv, check=False))
    if record.get("enabled_before", True):
        enabled = run(["systemctl", "enable", "--", unit])
        if enabled.returncode != 0:
            return enabled.returncode
    started = run(["systemctl", "--no-block", "start", "--", unit])
    if started.returncode != 0:
        return started.returncode
    record["status"] = record.pop("release_status")
    write(root / "history", f"{unit}.{record['request_id']}", record)
    (root / "releasing" / f"{unit}.json").unlink(missing_ok=True)
    directory = os.open(root / "releasing", os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return 0


def expire(root: Path, unit: str, request_id: str, *, now: float | None = None) -> int:
    """Release only the matching expired generation; renewed holds are untouched."""

    validate_request({"kind": "release-hold", "unit": unit})
    validate_request_id(request_id)
    if "-hold-" not in request_id:
        raise HoldError("hold_request_id_invalid")
    with locked(root):
        record = read(root, unit)
        moment = time.time() if now is None else now
        if record is None or record["request_id"] != request_id or record["status"] != "active":
            return 0
        if record["expires_at_epoch"] > moment:
            return 0
        begin_release(root, unit, record, released_by="expiry", status="expired_released", now=moment)
        return finish_release(root, unit)
    return 0


def sweep(root: Path, *, now: float | None = None) -> int:
    """Restore expiry after reboot and keep every unexpired hold off at boot."""

    if not root.exists():
        return 0
    moment = time.time() if now is None else now
    failed = False
    pending = root / "releasing"
    for path in sorted(pending.glob("*.json")) if pending.exists() else []:
        unit = path.name.removesuffix(".json")
        with locked(root):
            failed |= finish_release(root, unit) != 0
    for snapshot in active(root):
        unit = snapshot["unit"]
        if snapshot["expires_at_epoch"] <= moment:
            if expire(root, unit, snapshot["request_id"], now=moment) != 0:
                failed = True
            continue
        with locked(root):
            record = read(root, unit)
            if record is None or record["status"] != "active" or record["request_id"] != snapshot["request_id"]:
                continue
            if record["expires_at_epoch"] <= moment:
                continue  # the next sweep or the existing expiry job releases it
            disabled = subprocess.run(["systemctl", "disable", "--", unit], check=False)
            stopped = subprocess.run(["systemctl", "stop", "--", unit], check=False)
            failed |= disabled.returncode != 0 or stopped.returncode != 0
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holds-dir", type=Path, required=True)
    parser.add_argument("--unit")
    parser.add_argument("--request-id")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.sweep:
            if args.unit or args.request_id:
                raise HoldError("hold_sweep_arguments_invalid")
            return sweep(args.holds_dir)
        if not args.unit or not args.request_id:
            raise HoldError("hold_expiry_arguments_missing")
        return expire(args.holds_dir, args.unit, args.request_id)
    except (OSError, HoldError, RequestRefused):
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
