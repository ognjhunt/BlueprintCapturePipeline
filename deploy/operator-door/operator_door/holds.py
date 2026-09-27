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
        or record.get("status") not in {"active", "released", "expired_released", "failed_released"}
        or not isinstance(record.get("owner"), str)
        or type(record.get("expires_at_epoch")) is not int
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
        result = subprocess.run(["systemctl", "--no-block", "start", "--", unit], check=False)
        if result.returncode != 0:
            return result.returncode
        record.update(status="expired_released", released_at=timestamp(moment), released_by="expiry")
        write(root, unit, record)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holds-dir", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--request-id", required=True)
    args = parser.parse_args(argv)
    try:
        return expire(args.holds_dir, args.unit, args.request_id)
    except (OSError, HoldError, RequestRefused):
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
