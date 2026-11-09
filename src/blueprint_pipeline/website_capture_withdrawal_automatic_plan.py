"""Bounds and records for the withdrawal deletion plan the intake starts by itself after an acknowledgement.

The plan reads every run's records under the run roots, so it must never compete
with the intake that starts it: the intake answers acknowledgements and takes
uploads in the same systemd unit (deploy/systemd/blueprint-pipeline-intake.service:
MemoryMax=3G, CPUQuota=200%, OOMScoreAdjust=-200, TasksMax=512; measured up to
1.5G). A child stays in that unit's cgroup, so it is bounded from inside, with
what an unprivileged process may do:

- one automatic plan at a time on the host. The launch decision is taken under a
  host-wide lock (``HOST_LOCK``) and the child's pid is written as the holder
  before the lock is released, so a burst of acknowledgements starts one child;
- the child bounds itself before anything heavy is imported (``main``):
  oom_score_adj 1000 (the kernel takes the scan before the intake), nice 19, the
  idle IO class, ``ADDRESS_SPACE_BYTES`` of address space, and a
  ``WALL_CLOCK_SECONDS`` wall clock (SIGALRM, RLIMIT_CPU as a backstop). Each is
  best effort (the unit's seccomp filter answers EPERM rather than killing), and
  the plan record says which held;
- a holder that is dead, or alive past the wall clock plus a grace, is stale. An
  overdue one is killed; one that survives the kill (uninterruptible IO) stays
  the holder and its site is recorded as ``stuck``;
- the child gets only ``PLAN_ROOT_VARIABLES`` and the paths it needs to start.

Every request leaves a record a reader can check, in
``<site>/website_withdrawal/automatic_deletion_plan_state.json`` (``STATE_FILE``):

- ``launched``: the intake started the child (pid, time, attempt); ``planning``:
  the child holds the host lock and is scanning. Either one whose pid is gone,
  with no plan, is recorded as a failure on the next request;
- ``owed``: another site's plan holds the host lock (``owed_behind`` names it);
  the next acknowledgement asks again, and this is not a failure;
- ``failed``: the last attempt's reason and time, and when the next one may start
  (``BACKOFF_SECONDS`` after the 1st, 2nd and 3rd failure);
- ``stopped_needs_operator``: after ``MAX_FAILED_ATTEMPTS`` failures no automatic
  attempt starts again; the intake logs a warning on every acknowledgement and an
  operator runs the plan by hand;
- ``stuck``: this site's plan survived SIGKILL and still holds the host lock;
- ``recorded``: the plan is ``automatic_deletion_plan.json``; any earlier failure
  is kept as ``superseded_failure``.

The backoff and the stop concern only this plan. The tombstone, the
acknowledgement and the refusal of new work for the site do not wait on it.

Standard library only: the intake's withdrawal route imports this module.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import signal
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PLAN_FILE = "automatic_deletion_plan.json"
STATE_FILE = "automatic_deletion_plan_state.json"
STATE_SCHEMA = "website_capture_withdrawal_automatic_plan_state.v1"
LOCK_ENV = "BLUEPRINT_WEBSITE_WITHDRAWAL_PLAN_LOCK"
HOST_LOCK = Path("/var/lib/blueprint/pipeline-control-plane/website-withdrawal-automatic-plan.lock")
# 30 minutes: a full scan of the run roots is a quiet-host job.
WALL_CLOCK_SECONDS = 1800
STALE_GRACE_SECONDS = 300
# 1 GiB of address space: the plan streams records in 1 MiB windows and holds one run's
# inventory at a time (the synthetic drill's child peaks at 36 MB of address space). With the
# intake's measured 1.5G this stays well under the unit's MemoryMax=3G.
ADDRESS_SPACE_BYTES = 1 << 30
OOM_SCORE_ADJ = 1000
# Wait this long after the 1st, 2nd and 3rd failure; the 4th failure stops automatic attempts.
BACKOFF_SECONDS = (3600, 4 * 3600, 24 * 3600)
MAX_FAILED_ATTEMPTS = len(BACKOFF_SECONDS) + 1
# Every environment variable the plan reads for a root, queue or registry (recorded from the plan's
# own reads; tests/test_website_capture_withdrawal_drill.py fails when it reads one not listed).
PLAN_ROOT_VARIABLES = (
    "BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT", "BLUEPRINT_BETA_POLICY_SCREENING_WORK_ROOT",
    "BLUEPRINT_NATIVE_G1_SITE_REGISTRATION_ROOT", "BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE",
    "BLUEPRINT_RIGID_ROBOT_FREE_PROBE_WORK_ROOT", "BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY",
    "BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT")
CHILD_MODULE = "blueprint_pipeline.website_capture_withdrawal_automatic_plan"
_IOPRIO_CLASS_IDLE, _IOPRIO_WHO_PROCESS, _IOPRIO_CLASS_SHIFT = 3, 1, 13
_LOCK_WAIT_SECONDS = 10.0
# What the child process applied to itself (None in any other process).
APPLIED_LIMITS: dict[str, Any] | None = None


def host_lock(path: Path | None = None) -> Path:
    return Path(path or os.environ.get(LOCK_ENV) or HOST_LOCK)


def _holder_path(lock: Path) -> Path:
    return lock.with_name(lock.name + ".holder.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process(pid: int) -> tuple[str, str] | None:
    """``(state, start time)`` of ``pid`` from /proc/<pid>/stat, or None if it is gone."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return fields[0], fields[19]


def _alive(pid: Any, process_start: Any) -> bool:
    found = _process(int(pid or 0)) if int(pid or 0) > 0 else None
    return bool(found and found[0] != "Z" and found[1] == process_start)


def _import_paths() -> str:
    """The parent's own import roots (the venv's src, or the installed source, dependencies and scripts)."""
    import sysconfig
    # Only the standard library is the interpreter's own; the policy-mode supervisor puts the installed
    # source, its dependencies and the scripts root on sys.path (``_runtime_import_paths``), so pass those.
    stdlib = tuple({sysconfig.get_paths()[key] for key in ("stdlib", "platstdlib")})
    paths = [str(Path(__file__).resolve().parents[1])]
    paths += [entry for entry in sys.path if entry and os.path.isabs(entry) and not entry.startswith(stdlib)
              and not entry.endswith(".zip") and entry not in paths]
    return os.pathsep.join(paths)


def child_environment(parent: Mapping[str, str] | None = None) -> dict[str, str]:
    """PATH, a C.UTF-8 locale, PYTHONPATH to the parent's import roots, the host lock and ``PLAN_ROOT_VARIABLES``.

    The plan reads its default roots from those variables, so an override in the
    intake's environment reaches the plan. Nothing else the intake holds (provider
    key files, sync tokens, webhook URLs) is passed.
    """
    parent = os.environ if parent is None else parent
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": _import_paths(),
                   LOCK_ENV: str(host_lock(Path(parent[LOCK_ENV]) if parent.get(LOCK_ENV) else None))}
    environment.update({name: parent[name] for name in PLAN_ROOT_VARIABLES if name in parent})
    return environment


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace ``path``; refuses a symlinked directory and replaces (never follows) a symlinked file."""
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        temporary = f".{path.name}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600,
                             dir_fd=directory)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        os.close(directory)


def read_state(journal: Path) -> dict[str, Any]:
    path = journal / STATE_FILE
    if path.is_symlink() or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def write_state(journal: Path, state: str, **fields: Any) -> dict[str, Any]:
    """Replace this site's state record, keeping the attempt counts and the last launch and failure."""
    previous = read_state(journal)
    value = {"schema_version": STATE_SCHEMA, "failed_attempts": int(previous.get("failed_attempts") or 0),
             "launches": int(previous.get("launches") or 0), "last_launch": previous.get("last_launch"),
             "last_failure": previous.get("last_failure"), **fields, "state": state, "updated_at_iso": _now_iso()}
    _write_json_atomic(journal / STATE_FILE, value)
    return value


def record_failure(journal: Path, reason: str, **details: Any) -> dict[str, Any]:
    """Record a failed attempt and when the next may start, or stop after ``MAX_FAILED_ATTEMPTS``."""
    previous = read_state(journal)
    failed = int(previous.get("failed_attempts") or 0) + 1
    failure = {"reason": reason, "at_iso": _now_iso(), "attempt": failed, **details}
    if failed >= MAX_FAILED_ATTEMPTS:
        return write_state(journal, "stopped_needs_operator", failed_attempts=failed, last_failure=failure,
                           next_attempt_not_before_epoch=None,
                           needs_operator=f"{failed} automatic attempts failed (last: {reason}); run the plan by hand "
                                          "(docs/runbooks/beta-operations.md section 8)")
    return write_state(journal, "failed", failed_attempts=failed, last_failure=failure,
                       next_attempt_not_before_epoch=time.time() + BACKOFF_SECONDS[failed - 1])


def record_recorded(journal: Path, plan_digest: str) -> dict[str, Any]:
    previous = read_state(journal)
    superseded = previous.get("last_failure")
    return write_state(journal, "recorded", plan_digest=plan_digest, last_failure=None,
                       superseded_failure={**superseded, "superseded": True} if superseded else None)


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
    if holder is None or _read_holder(lock) == holder:
        _holder_path(lock).unlink(missing_ok=True)


def _journal_of(holder: Mapping[str, Any]) -> Path | None:
    journal = Path(str(holder.get("journal") or ""))
    if journal.is_absolute() and journal.is_dir() and not journal.is_symlink() and not (journal / PLAN_FILE).is_file():
        return journal
    return None


def reap_stale(lock: Path, *, now: float | None = None, own_pid: int | None = None) -> str:
    """``free``, ``fresh`` (a live holder within its limit, or one that survived the kill), or ``reaped``."""
    holder = _read_holder(lock)
    if holder is None:
        return "free"
    pid, started = int(holder.get("pid") or 0), float(holder.get("started_at_epoch") or 0)
    if own_pid is not None and pid == own_pid:
        return "free"
    now = time.time() if now is None else now
    alive = _alive(pid, holder.get("process_start"))
    if alive and now - started <= WALL_CLOCK_SECONDS + STALE_GRACE_SECONDS:
        return "fresh"
    journal = _journal_of(holder)
    if holder.get("reservation"):
        # The intake's own reservation while it launches: never killed; an overdue or dead one is just cleared.
        _clear_holder(lock, holder)
        return "reaped"
    if alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            return "fresh"  # not ours to kill; it stays the holder
        deadline = time.monotonic() + 5
        while _alive(pid, holder.get("process_start")) and time.monotonic() < deadline:
            time.sleep(0.05)
        if _alive(pid, holder.get("process_start")):
            # Uninterruptible IO: it still holds the lock, so it stays the holder, and its site says so.
            if journal is not None:
                write_state(journal, "stuck", stuck={"pid": pid, "reason": "automatic_plan_survived_sigkill",
                                                     "at_iso": _now_iso()})
            return "fresh"
    if journal is not None:
        # A holder the intake wrote at launch was never replaced by the child's own: it died before the lock.
        record_failure(journal, "automatic_plan_timed_out_killed" if alive
                       else "automatic_plan_child_died_before_recording" if holder.get("launched_by") == "intake"
                       else "automatic_plan_process_died", pid=pid)
    _clear_holder(lock, holder)
    return "reaped"


def _open_lock(lock: Path) -> int:
    lock.parent.mkdir(parents=True, exist_ok=True)
    return os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)


def _try_flock(descriptor: int, *, wait: float = 0.0) -> bool:
    deadline = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)


@contextmanager
def hold_host_lock(lock: Path, journal: Path) -> Iterator[bool]:
    """In the child: yield True while this process is the host's one automatic plan, False if another is."""
    with os.fdopen(_open_lock(lock), "a+") as handle:
        # Another request may hold the flock briefly while it decides, so wait a little for it.
        if not _try_flock(handle.fileno(), wait=_LOCK_WAIT_SECONDS) and (
                reap_stale(lock) != "reaped" or not _try_flock(handle.fileno(), wait=_LOCK_WAIT_SECONDS)):
            holder = _read_holder(lock) or {}
            write_state(journal, "owed", owed_behind=holder.get("journal"))
            yield False
            return
        current = _read_holder(lock) or {}
        launched_for_us = current.get("journal") == str(journal) and (
            current.get("reservation") or current.get("launched_by") == "intake")
        if not launched_for_us and reap_stale(lock, own_pid=os.getpid()) == "fresh":
            holder = _read_holder(lock) or {}
            write_state(journal, "owed", owed_behind=holder.get("journal"))
            yield False
            return
        holder = {"pid": os.getpid(), "process_start": (_process(os.getpid()) or ("", ""))[1],
                  "started_at_epoch": time.time(), "journal": str(journal)}
        _write_json_atomic(_holder_path(lock), holder)
        write_state(journal, "planning", planning={"pid": holder["pid"], "process_start": holder["process_start"],
                                                   "at_iso": _now_iso()})
        try:
            yield True
        finally:
            _clear_holder(lock, holder)


def _account_for_last_launch(journal: Path) -> None:
    """A ``launched`` or ``planning`` state whose process is gone, with no plan, is a failure."""
    state = read_state(journal)
    if state.get("state") not in {"launched", "planning"} or (journal / PLAN_FILE).is_file():
        return
    process = state.get("planning") if state["state"] == "planning" else state.get("last_launch")
    if process and not _alive(process.get("pid"), process.get("process_start")):
        record_failure(journal, "automatic_plan_child_died_before_recording" if state["state"] == "launched"
                       else "automatic_plan_process_died", pid=process.get("pid"))


def request(journal: Path, command: list[str], *, launch: Callable[[list[str]], Any] | None = None,
            lock_path: Path | None = None) -> str:
    """The intake's decision for one acknowledged site; returns what the state record now says.

    ``recorded``, ``refused`` (the plan path is not a regular file; counted as a
    failure), ``running``, ``backoff``, ``stopped``, ``owed`` or ``started``. The
    decision and the spawn happen under the host lock; a spawn that raises is
    recorded as a failure, never only logged.
    """
    plan = journal / PLAN_FILE
    if plan.is_symlink() or (plan.exists() and not plan.is_file()):
        state = read_state(journal)
        if state.get("state") != "stopped_needs_operator":
            record_failure(journal, "website_withdrawal_deletion_automatic_plan_path_unsafe")
        return "refused"
    if plan.is_file():
        return "recorded"
    lock = host_lock(lock_path)
    with os.fdopen(_open_lock(lock), "a+") as handle:
        if not _try_flock(handle.fileno()) and (
                reap_stale(lock) != "reaped" or not _try_flock(handle.fileno(), wait=1.0)):
            _account_for_last_launch(journal)
            return _owed_or(journal, lock)
        reap_stale(lock)  # a holder that died, or a launched child that never took the lock
        _account_for_last_launch(journal)
        state = read_state(journal)
        if state.get("state") == "stopped_needs_operator":
            return "stopped"
        if state.get("state") in {"launched", "planning", "stuck"}:
            return "running"
        if state.get("state") == "failed" and time.time() < float(state.get("next_attempt_not_before_epoch") or 0):
            return "backoff"
        if _read_holder(lock) is not None:
            return _owed_or(journal, lock)
        # Reserve the host for this site before the lock is released, so a burst starts one child.
        reservation = {"pid": os.getpid(), "process_start": (_process(os.getpid()) or ("", ""))[1],
                       "started_at_epoch": time.time(), "journal": str(journal), "reservation": True}
        _write_json_atomic(_holder_path(lock), reservation)
        attempt = int(state.get("launches") or 0) + 1
        write_state(journal, "launching", launches=attempt,
                    last_launch={"attempt": attempt, "at_iso": _now_iso(), "pid": None})
    try:
        pid = (launch or launch_detached)(command)
    except Exception as error:  # noqa: BLE001 - recorded, never only logged
        _clear_holder(lock, reservation)
        code = (errno.errorcode.get(error.errno or 0, "unknown") if isinstance(error, OSError)
                else type(error).__name__)
        record_failure(journal, "automatic_plan_spawn_failed_" + code)
        return "failed"
    if isinstance(pid, int):
        start = (_process(pid) or ("", ""))[1]
        if _read_holder(lock) == reservation:
            _write_json_atomic(_holder_path(lock), {"pid": pid, "process_start": start,
                                                    "started_at_epoch": reservation["started_at_epoch"],
                                                    "journal": str(journal), "launched_by": "intake"})
        if read_state(journal).get("state") == "launching":
            write_state(journal, "launched", last_launch={"attempt": attempt, "at_iso": _now_iso(), "pid": pid,
                                                          "process_start": start})
    else:
        # A launcher that ran the plan in this process (the drill, tests) has left its own record.
        _clear_holder(lock, reservation)
        if read_state(journal).get("state") == "launching":
            write_state(journal, "launched", last_launch={"attempt": attempt, "at_iso": _now_iso(), "pid": None})
    return "started"


def _owed_or(journal: Path, lock: Path) -> str:
    state = read_state(journal)
    if state.get("state") == "stopped_needs_operator":
        return "stopped"
    holder = _read_holder(lock) or {}
    if holder.get("journal") == str(journal):
        return "running"
    write_state(journal, "owed", owed_behind=holder.get("journal"))
    return "owed"


class PlanTimedOut(Exception):
    """The automatic plan passed its wall clock."""


def apply_child_limits() -> dict[str, Any]:
    """Bound this process; each limit is best effort and reported. Stdlib only, before any heavy import."""
    import ctypes
    import resource

    applied: dict[str, Any] = {}

    def attempt(name: str, action: Callable[[], Any]) -> None:
        applied.setdefault("order", []).append(name)
        try:
            action()
            applied[name] = True
        except (OSError, ValueError, AttributeError, KeyError):
            applied[name] = False

    def idle_io() -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        number = {"x86_64": 251, "aarch64": 30}[os.uname().machine]
        if libc.syscall(number, _IOPRIO_WHO_PROCESS, 0, _IOPRIO_CLASS_IDLE << _IOPRIO_CLASS_SHIFT) != 0:
            raise OSError(ctypes.get_errno(), "ioprio_set")

    # The OOM score and priority first: they matter most and cost nothing.
    attempt("oom_score_adj_1000", lambda: Path("/proc/self/oom_score_adj").write_text(str(OOM_SCORE_ADJ)))
    attempt("nice_19", lambda: os.setpriority(os.PRIO_PROCESS, 0, 19))
    attempt("io_class_idle", idle_io)
    attempt("address_space_bytes", lambda: resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES,) * 2))
    attempt("cpu_seconds", lambda: resource.setrlimit(resource.RLIMIT_CPU, (WALL_CLOCK_SECONDS, WALL_CLOCK_SECONDS + 60)))

    def timed_out(_signum: int, _frame: Any) -> None:
        raise PlanTimedOut()

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(WALL_CLOCK_SECONDS)
    applied["wall_clock_seconds"] = WALL_CLOCK_SECONDS
    return applied


def launch_detached(command: list[str]) -> int:
    """Start the plan child with only its environment; reaped by a thread, never awaited by the request."""
    import subprocess
    import threading
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               env=child_environment(), cwd="/", start_new_session=True, close_fds=True)
    threading.Thread(target=process.wait, daemon=True).start()
    return process.pid


def main(argv: list[str] | None = None) -> None:
    """The child's entry: bound this process, then run the deletion module's plan-only ``--record-automatic-plan``."""
    import importlib
    # Run as ``-m`` this file is ``__main__``; the deletion module reads the imported module's names.
    bounds = importlib.import_module(CHILD_MODULE)
    bounds.APPLIED_LIMITS = bounds.apply_child_limits()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--apply" in arguments or "--authorize-complete-deletion" in arguments:
        raise SystemExit("website_withdrawal_deletion_automatic_plan_never_applies")
    # Only this child process, never the intake that imports this module, loads the deletion module,
    # which reaches the screening ledger and the paid launch path (test_live_pipeline_import_isolation).
    importlib.import_module("blueprint_pipeline.website_capture_withdrawal_deletion").main(
        [*arguments, "--record-automatic-plan"])


if __name__ == "__main__":
    main()
