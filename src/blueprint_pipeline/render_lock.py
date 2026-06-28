"""File-based advisory single-flight lock for GPU render / OSCAR launches.

Two agents or sessions on the same machine must not fire concurrent GPU
render or OSCAR launches: doing so starves each other's processes and piles up
orphan pods. This module provides a small, dependency-free advisory lock so the
*second* launcher learns a render is already in flight and backs off (or waits)
instead of double-firing.

The lock is a single file under a state directory whose contents record the
holder's PID, the acquisition (start) time, the job label, and the host. It is
*advisory*: it only constrains code that goes through ``render_lock(...)``. It
is not a kernel mutex and makes no cross-machine guarantee — it single-flights
per machine, which is exactly where pods pile up.

Stale holders are reclaimed automatically: if the recorded PID is no longer
alive (and the lock was taken on this host), the lockfile is treated as
abandoned and replaced. An optional ``max_age`` reclaims locks held longer than
a TTL, for wedged holders whose PID lingers.

Usage::

    from blueprint_pipeline.render_lock import render_lock, RenderLockTimeout

    try:
        with render_lock("isaac-render"):
            launch_gpu_render(...)
    except RenderLockTimeout:
        log.info("another render is already running; skipping double-launch")

By default a contended acquire **fails fast** (``timeout=0``) so launchers do
not queue up a second pod. Pass ``timeout=`` to block: ``timeout=None`` blocks
indefinitely, ``timeout=N`` blocks up to ``N`` seconds.

This module is standalone — it deliberately does not import or modify any
runner/job module. Callers opt in.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "LockInfo",
    "RenderLock",
    "RenderLockError",
    "RenderLockTimeout",
    "render_lock",
    "default_state_dir",
    "main",
]

#: Environment variable overriding the default lock state directory.
STATE_DIR_ENV = "BLUEPRINT_RENDER_LOCK_DIR"

_LOCK_VERSION = 1
#: Brief grace before an unparseable lockfile is judged stale rather than an
#: in-progress write by a racing acquirer.
_MALFORMED_GRACE_SECONDS = 0.05
_SAFE_LABEL = re.compile(r"[^A-Za-z0-9._-]+")


class RenderLockError(RuntimeError):
    """Base class for render-lock failures."""


class RenderLockTimeout(RenderLockError):
    """Raised when the lock could not be acquired within the timeout."""


@dataclass(frozen=True)
class LockInfo:
    """Identity of a lock holder, as recorded in the lockfile."""

    pid: int
    label: str
    start_time: float
    start_time_iso: str
    hostname: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "label": self.label,
            "start_time": self.start_time,
            "start_time_iso": self.start_time_iso,
            "hostname": self.hostname,
        }


def default_state_dir() -> Path:
    """Return the directory lockfiles live in (``$BLUEPRINT_RENDER_LOCK_DIR``).

    Defaults to ``~/.blueprint-state/render-locks`` — a stable per-user path so
    independent sessions on the same machine see each other's locks.
    """

    override = os.environ.get(STATE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".blueprint-state" / "render-locks"


def _hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover - extremely unusual
        return ""


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _slugify(label: str) -> str:
    slug = _SAFE_LABEL.sub("-", label.strip()).strip("-._")
    return (slug or "lock")[:120]


def _pid_is_alive(pid: int) -> bool:
    """Return whether ``pid`` names a live process on this host."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but is owned by another user.
        return True
    except OSError:
        return False
    return True


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_create_with_content(path: Path, data: bytes) -> bool:
    """Atomically create ``path`` carrying ``data``.

    Returns ``True`` if we created it, ``False`` if it already existed. The file
    becomes visible *with its content already in place* (via a hardlink from a
    temp file), so a racing reader never observes an empty/partial lockfile.
    """

    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(tmp), str(path))
        except FileExistsError:
            return False
        return True
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _read_lock_info(path: Path) -> LockInfo | None:
    """Parse the lockfile at ``path``; ``None`` if absent, empty, or malformed."""

    try:
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        pid = int(data["pid"])
    except (KeyError, TypeError, ValueError):
        return None
    try:
        start_time = float(data.get("start_time", 0.0) or 0.0)
    except (TypeError, ValueError):
        start_time = 0.0
    return LockInfo(
        pid=pid,
        label=str(data.get("label", "")),
        start_time=start_time,
        start_time_iso=str(data.get("start_time_iso", "")),
        hostname=str(data.get("hostname", "")),
    )


class RenderLock:
    """An advisory single-flight lock keyed by ``label``.

    Not re-entrant: a second acquire of the same label — even from the same
    process — contends, because single-flight is the whole point.
    """

    def __init__(
        self,
        label: str,
        *,
        state_dir: str | os.PathLike[str] | None = None,
        timeout: float | None = 0.0,
        poll_interval: float = 0.1,
        max_age: float | None = None,
    ) -> None:
        if not label or not label.strip():
            raise ValueError("render lock label must be a non-empty string")
        base = Path(state_dir).expanduser() if state_dir is not None else default_state_dir()
        self._label = label
        self._path = base / f"{_slugify(label)}.lock"
        self._timeout = timeout
        self._poll_interval = max(float(poll_interval), 0.001)
        self._max_age = max_age
        self._held = False
        self._token: tuple[int, float] | None = None

    # -- introspection ----------------------------------------------------

    @property
    def label(self) -> str:
        return self._label

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        return self._held

    def read_holder(self) -> LockInfo | None:
        """Return the current on-disk holder, or ``None`` if unlocked."""

        return _read_lock_info(self._path)

    # -- core -------------------------------------------------------------

    def _build_payload(self) -> bytes:
        start_time = time.time()
        self._token = (os.getpid(), start_time)
        payload = {
            "pid": os.getpid(),
            "label": self._label,
            "start_time": start_time,
            "start_time_iso": _iso(start_time),
            "hostname": _hostname(),
            "version": _LOCK_VERSION,
        }
        return json.dumps(payload, indent=2).encode("utf-8")

    def _is_stale(self, holder: LockInfo) -> bool:
        same_host = (not holder.hostname) or holder.hostname == _hostname()
        if same_host and not _pid_is_alive(holder.pid):
            return True
        if self._max_age is not None and holder.start_time > 0:
            if (time.time() - holder.start_time) > self._max_age:
                return True
        return False

    def _reclaim(self, expected: LockInfo | None) -> None:
        """Remove the lockfile, but only if it still matches ``expected``.

        Guards the narrow window where another acquirer replaced a stale lock
        between our read and our unlink — we must not delete *their* fresh lock.
        """

        if expected is not None:
            current = _read_lock_info(self._path)
            if current is not None and not (
                current.pid == expected.pid and current.start_time == expected.start_time
            ):
                return
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass

    def acquire(self) -> "RenderLock":
        """Acquire the lock, reclaiming stale holders. Returns ``self``.

        Raises :class:`RenderLockTimeout` if a live holder keeps the lock past
        the configured timeout (``timeout=0`` => fail fast immediately).
        """

        if self._held:
            return self
        _ensure_dir(self._path.parent)
        payload = self._build_payload()
        deadline: float | None = None
        if self._timeout is not None and self._timeout > 0:
            deadline = time.monotonic() + self._timeout

        while True:
            if _atomic_create_with_content(self._path, payload):
                self._held = True
                return self

            holder = _read_lock_info(self._path)
            if holder is None:
                if not self._path.exists():
                    continue  # vanished between create and read: retry
                # Exists but unparseable. Give a racing writer a grace window,
                # then treat a persistently bad lockfile as abandoned.
                time.sleep(min(self._poll_interval, _MALFORMED_GRACE_SECONDS))
                holder = _read_lock_info(self._path)
                if holder is None:
                    if self._path.exists():
                        self._reclaim(expected=None)
                    continue

            if self._is_stale(holder):
                self._reclaim(expected=holder)
                continue

            # Live holder: contention.
            if self._timeout == 0:
                raise RenderLockTimeout(self._contention_message(holder))
            if deadline is not None and time.monotonic() >= deadline:
                raise RenderLockTimeout(self._contention_message(holder))
            time.sleep(self._poll_interval)

    def release(self) -> None:
        """Release the lock if we still hold it. Safe to call more than once."""

        if not self._held:
            return
        self._held = False
        if self._token is None:
            return
        holder = _read_lock_info(self._path)
        if holder is None:
            return
        if holder.pid == self._token[0] and holder.start_time == self._token[1]:
            try:
                os.unlink(self._path)
            except FileNotFoundError:
                pass

    def _contention_message(self, holder: LockInfo) -> str:
        age = max(0.0, time.time() - holder.start_time) if holder.start_time else 0.0
        return (
            f"render lock '{self._label}' is held by pid {holder.pid} "
            f"on {holder.hostname or 'this host'} (held {age:.1f}s) at {self._path}"
        )

    # -- context manager --------------------------------------------------

    def __enter__(self) -> "RenderLock":
        return self.acquire()

    def __exit__(self, *_exc: object) -> bool:
        self.release()
        return False


def render_lock(
    label: str,
    *,
    state_dir: str | os.PathLike[str] | None = None,
    timeout: float | None = 0.0,
    poll_interval: float = 0.1,
    max_age: float | None = None,
) -> RenderLock:
    """Build a :class:`RenderLock` for ``label`` (use as a context manager).

    Example::

        with render_lock("isaac-render"):
            ...
    """

    return RenderLock(
        label,
        state_dir=state_dir,
        timeout=timeout,
        poll_interval=poll_interval,
        max_age=max_age,
    )


# --------------------------------------------------------------------------
# CLI: inspect / break a lock for operators unsticking a wedged machine.
# --------------------------------------------------------------------------


def _holder_dict(holder: LockInfo | None) -> dict[str, Any] | None:
    return holder.as_dict() if holder is not None else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="render_lock",
        description="Inspect or break the advisory render single-flight lock.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Report the current holder of a lock.")
    p_status.add_argument("label")
    p_status.add_argument("--state-dir", default=None)
    p_status.add_argument("--max-age", type=float, default=None)

    p_break = sub.add_parser("break", help="Remove a stale (or forced) lock.")
    p_break.add_argument("label")
    p_break.add_argument("--state-dir", default=None)
    p_break.add_argument("--max-age", type=float, default=None)
    p_break.add_argument(
        "--force",
        action="store_true",
        help="Remove the lock even if its holder appears alive.",
    )

    args = parser.parse_args(argv)
    lock = RenderLock(args.label, state_dir=args.state_dir, max_age=args.max_age)
    holder = lock.read_holder()

    if args.command == "status":
        stale = holder is not None and lock._is_stale(holder)
        print(
            json.dumps(
                {
                    "label": args.label,
                    "path": str(lock.path),
                    "held": holder is not None and not stale,
                    "stale": stale,
                    "holder": _holder_dict(holder),
                },
                indent=2,
            )
        )
        return 0

    # command == "break"
    if holder is None:
        print(json.dumps({"label": args.label, "broken": False, "reason": "not_held"}))
        return 0
    stale = lock._is_stale(holder)
    if not stale and not args.force:
        print(
            json.dumps(
                {
                    "label": args.label,
                    "broken": False,
                    "reason": "holder_alive",
                    "holder": _holder_dict(holder),
                }
            )
        )
        return 1
    try:
        os.unlink(lock.path)
    except FileNotFoundError:
        pass
    print(
        json.dumps(
            {"label": args.label, "broken": True, "stale": stale, "holder": _holder_dict(holder)}
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
