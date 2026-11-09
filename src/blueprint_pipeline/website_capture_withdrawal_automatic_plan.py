"""Bounds for the withdrawal deletion plan the intake starts by itself after an acknowledgement.

The plan reads every run's records under the run roots, so it must never compete
with the intake that starts it: the intake answers acknowledgements and takes
uploads in the same systemd unit (deploy/systemd/blueprint-pipeline-intake.service:
MemoryMax=3G, CPUQuota=200%, OOMScoreAdjust=-200; measured up to 1.5G). A child
process stays in that unit's cgroup, so it is bounded here, from inside, with what
an unprivileged process may do:

- one automatic plan at a time on the host: a host-wide lock (``HOST_LOCK``). A
  request while a fresh holder runs starts nothing and leaves the plan owed; the
  WebApp resends each pending acknowledgement every hour, which asks again;
- a holder that is dead, or alive past ``WALL_CLOCK_SECONDS`` plus a grace, is
  stale: a hung one is killed (only if its pid still has the recorded start time)
  and its site gets a failure record;
- the child: wall clock ``WALL_CLOCK_SECONDS`` (SIGALRM, and RLIMIT_CPU as a
  backstop), address space ``ADDRESS_SPACE_BYTES``, nice 19, the idle IO class,
  and oom_score_adj 1000 so the kernel takes the scan before the intake. Each
  limit is best effort (the unit's seccomp filter answers EPERM rather than
  killing), and the record says which ones held;
- the child's environment is only what the plan needs (``child_environment``);
- a failure, timeout or death is recorded beside the tombstone
  (``automatic_deletion_plan_failure.json``), so there is never no plan and no trace.

Standard library only: the intake's withdrawal route imports this module.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PLAN_FILE = "automatic_deletion_plan.json"
FAILURE_FILE = "automatic_deletion_plan_failure.json"
FAILURE_SCHEMA = "website_capture_withdrawal_automatic_plan_failure.v1"
LOCK_ENV = "BLUEPRINT_WEBSITE_WITHDRAWAL_PLAN_LOCK"
CHILD_ENV = "BLUEPRINT_WEBSITE_WITHDRAWAL_PLAN_CHILD"
HOST_LOCK = Path("/var/lib/blueprint/pipeline-control-plane/website-withdrawal-automatic-plan.lock")
# 30 minutes: a full scan of the run roots is a quiet-host job; the hourly resend retries a timeout.
WALL_CLOCK_SECONDS = 1800
STALE_GRACE_SECONDS = 300
# 1 GiB of address space: the plan streams records in 1 MiB windows and holds one run's
# inventory at a time (the synthetic drill's child peaks at 36 MB of address space). With the intake's measured 1.5G this stays
# under the unit's MemoryHigh=2G-plus headroom and well under MemoryMax=3G.
ADDRESS_SPACE_BYTES = 1 << 30
OOM_SCORE_ADJ = 1000
_IOPRIO_CLASS_IDLE, _IOPRIO_WHO_PROCESS, _IOPRIO_CLASS_SHIFT = 3, 1, 13
# Only roots and paths the plan resolves; never a secret, token, key or credential file.
_PASSED = re.compile(r"BLUEPRINT_[A-Z0-9_]*(_ROOT|_ROOTS|_PATH|_REGISTRY)")
_SECRET = re.compile(r"SECRET|TOKEN|KEY|CREDENTIAL|PASSWORD|PRIVATE|_FILE")


def host_lock(path: Path | None = None) -> Path:
    return Path(path or os.environ.get(LOCK_ENV) or HOST_LOCK)


def _holder_path(lock: Path) -> Path:
    return lock.with_name(lock.name + ".holder.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_start(pid: int) -> str | None:
    """The kernel's start time of ``pid`` (field 22 of /proc/<pid>/stat), or None if it is gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat.rsplit(")", 1)[1].split()[19]


def child_environment(parent: Mapping[str, str] | None = None) -> dict[str, str]:
    """PATH, a C.UTF-8 locale, PYTHONPATH to this package, the host lock, and ``BLUEPRINT_*`` roots and paths.

    The plan reads its default roots from ``BLUEPRINT_*_ROOT``/``_PATH``/``_REGISTRY``
    overrides (screening queue and work roots, team registry and queues, G1 site
    registrations), so those pass; everything else the intake holds (provider key
    files, sync tokens, webhook URLs) does not.
    """
    parent = os.environ if parent is None else parent
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                   CHILD_ENV: "1", LOCK_ENV: str(host_lock(Path(parent[LOCK_ENV]) if parent.get(LOCK_ENV) else None))}
    for name, value in parent.items():
        if _PASSED.fullmatch(name) and not _SECRET.search(name):
            environment[name] = value
    return environment


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace ``path`` with ``value``; refuses a symlinked directory and never follows a symlinked file."""
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        temporary = f".{path.name}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600,
                             dir_fd=directory)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)  # replaces a link, never follows
        os.fsync(directory)
    finally:
        os.close(directory)


def record_failure(journal: Path, reason: str, **details: Any) -> dict[str, Any]:
    """Record (replacing the last one) why the automatic plan did not land; the next acknowledgement retries."""
    path = journal / FAILURE_FILE
    previous: dict[str, Any] = {}
    if path.is_file() and not path.is_symlink():
        try:
            previous = json.loads(path.read_text())
        except (OSError, ValueError):
            previous = {}
    value = {"schema_version": FAILURE_SCHEMA, "reason": reason, "at_iso": _now_iso(),
             "attempts": int(previous.get("attempts") or 0) + 1, "retried_by": "the next acknowledgement", **details}
    _write_json_atomic(path, value)
    return value


def _read_holder(lock: Path) -> dict[str, Any] | None:
    path = _holder_path(lock)
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _clear_holder(lock: Path, holder: Mapping[str, Any] | None = None) -> None:
    path = _holder_path(lock)
    if holder is None or _read_holder(lock) == holder:
        path.unlink(missing_ok=True)


def reap_stale(lock: Path, *, now: float | None = None) -> str:
    """``free``, ``fresh`` (a live holder within its limit), or ``reaped`` (a dead or overdue holder, recorded).

    Reads the holder record beside the lock. A dead holder (no process with its pid
    and recorded start time) never recorded its own end, so its site gets a failure
    record. A live holder past the wall clock plus grace is killed first.
    """
    holder = _read_holder(lock)
    if holder is None:
        return "free"
    now = time.time() if now is None else now
    pid, started = int(holder.get("pid") or 0), float(holder.get("started_at_epoch") or 0)
    alive = pid > 0 and _process_start(pid) == holder.get("process_start")
    overdue = now - started > WALL_CLOCK_SECONDS + STALE_GRACE_SECONDS
    if alive and not overdue:
        return "fresh"
    if alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            return "fresh"  # not ours to kill; it stays the holder
    journal = Path(str(holder.get("journal") or ""))
    if journal.is_dir() and not journal.is_symlink() and not (journal / PLAN_FILE).is_file():
        record_failure(journal, "automatic_plan_timed_out_killed" if alive else "automatic_plan_process_died",
                       pid=pid, started_at_epoch=started)
    _clear_holder(lock, holder)
    return "reaped"


@contextmanager
def hold_host_lock(lock: Path, journal: Path) -> Iterator[bool]:
    """Yield True while this process is the host's one automatic plan, False if another fresh one is."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # The flock holder is alive (a dead one releases it); kill it only when overdue, then try once more.
            if reap_stale(lock) != "reaped":
                yield False
                return
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
        else:
            reap_stale(lock)  # a holder record left by a process that died holding the flock
        holder = {"pid": os.getpid(), "process_start": _process_start(os.getpid()),
                  "started_at_epoch": time.time(), "journal": str(journal)}
        _write_json_atomic(_holder_path(lock), holder)
        try:
            yield True
        finally:
            _clear_holder(lock, holder)


class PlanTimedOut(Exception):
    """The automatic plan passed its wall clock."""


def apply_child_limits() -> dict[str, Any]:
    """Bound this process (the intake's automatic plan child); each limit is best effort and reported."""
    import ctypes
    import resource

    applied: dict[str, Any] = {}

    def attempt(name: str, action: Any) -> None:
        try:
            action()
            applied[name] = True
        except (OSError, ValueError, AttributeError):
            applied[name] = False

    attempt("address_space_bytes", lambda: resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES,) * 2))
    attempt("cpu_seconds", lambda: resource.setrlimit(resource.RLIMIT_CPU, (WALL_CLOCK_SECONDS, WALL_CLOCK_SECONDS + 60)))
    attempt("nice_19", lambda: os.setpriority(os.PRIO_PROCESS, 0, 19))

    def idle_io() -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        number = {"x86_64": 251, "aarch64": 30}[os.uname().machine]
        if libc.syscall(number, _IOPRIO_WHO_PROCESS, 0, _IOPRIO_CLASS_IDLE << _IOPRIO_CLASS_SHIFT) != 0:
            raise OSError(ctypes.get_errno(), "ioprio_set")

    attempt("io_class_idle", idle_io)
    attempt("oom_score_adj_1000", lambda: Path("/proc/self/oom_score_adj").write_text(str(OOM_SCORE_ADJ)))

    def timed_out(_signum: int, _frame: Any) -> None:
        raise PlanTimedOut()

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(WALL_CLOCK_SECONDS)
    applied["wall_clock_seconds"] = WALL_CLOCK_SECONDS
    return applied


def launch_detached(command: list[str]) -> None:
    """Start the plan child with only its environment; reaped by a thread, never awaited by the request."""
    import subprocess
    import threading
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               env=child_environment(), cwd="/", start_new_session=True, close_fds=True)
    threading.Thread(target=process.wait, daemon=True).start()
