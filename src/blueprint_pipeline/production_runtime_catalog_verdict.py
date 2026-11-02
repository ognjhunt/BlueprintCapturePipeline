"""Reuse the guard's catalog and residency phases while nothing they read moved.

The Pub/Sub handoff listener starts every minute, and its ExecStartPre (the
production runtime env guard) re-projected every published launch profile into
the served catalog and re-resolved every bound input on each start: 551
profiles and 3,881 distinct inputs on 2026-11-02, 12-28 s of CPU per start, to
find the same consistent catalog almost every time. The control plane repeats
it every five minutes. This module stores the ready result of those two phases
and serves it back while nothing they read has moved.

What the fingerprint holds:

* the key: this verdict's and the guard's schema, the running release commit,
  the interpreter and its flags, ``sys.path``, the import hooks, the working
  directory and the whole process environment (systemd's per-invocation
  variables by presence only) -- so the profile directory, the served catalog
  path, the input roots and the owner-store root are all in it;
* the code: this module, the guard and the two phase modules by source digest
  (re-proven by ``validation_verdict_store``), every other loaded
  ``blueprint_pipeline`` source by full identity, and the names in every
  ``sys.path`` directory, so an installed or upgraded package moves it. (Asking
  the import system to re-find a subpackage module imports its parent package,
  which for ``task_evaluation_supervisor`` cost a second of CPU per check; an
  identity whose ctime is kernel-owned proves the same unchanged bytes without
  importing anything.) A pass whose environment changed after the guard started
  -- the import pass loads OpenCV, which edits ``LD_LIBRARY_PATH`` -- is not
  stored, so a stored verdict always ran under the environment it is keyed on;
* the profile directory: every entry's name and full ``lstat`` identity
  (device, inode, mode, owner, size, nanosecond mtime and kernel-owned ctime);
* the served catalog: the same identity, and its bytes by digest;
* everything the pass consulted, recorded at first touch while it ran: every
  file it opened, every path it stat-ed (``exists``, ``is_file``, ``resolve``
  ...), absent ones included, so a ``revoked.json`` that appears later moves
  the verdict, and the names in every directory it listed.

Only a pass that ended in a ready guard, found the catalog already consistent
and wrote nothing is stored. Skipping a later pass therefore skips only a pass
that would have written nothing too: the same inputs give the same projection
and the same bytes. A pass that writes, starts a process, opens a socket or
reads through a descriptor this module cannot name stores nothing; neither does
one whose inputs moved while it ran. A refusal is never stored. (The spend
authority ledger and the launch-lock slots, which do write, are not cached.)

A stored entry is reused only when it is a regular file owned by this account,
not a symlink, not group- or world-writable, readable, well formed, matching
the key and code, and at most ``MAX_VERDICT_AGE_SECONDS`` old. Anything else
runs the full pass. The layer is a cache: any failure in it runs the full pass,
so it can make the guard slower but never stop a unit.

Time is held explicitly. An owner window, a visual-review grant or a
preparation grant lapses with no file changing, and a lapsed visual-review grant
does not merely demote its profile: its blocker fails the whole catalog, so a
full pass stops every guarded unit. Each such check reports the instant its
answer flips (``decision_deadlines``); the pass stores the earliest one still
ahead, and a verdict is not reused from ``DEADLINE_MARGIN_SECONDS`` before it.
Package code that reads ``time.time`` during the pass without being one of
those checks (``_CLOCK_SITES``) makes the pass unstorable, so an unreported time
check cannot hide behind the cache. What remains bounded only by
``MAX_VERDICT_AGE_SECONDS``: a clock read that bypasses ``time.time`` (a C-level
``datetime.now``), and a change that raises no audit event and no Python
``stat`` (a C library's own reads).
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .decision_deadlines import collecting_deadlines
from .production_runtime_import_verdict import _process_started_ns, verdict_key
from .task_evaluation_release_identity import running_release_commit
from .validation_file_digests import sha256_file, touched_files
from .validation_verdict_store import entry_path, lookup, store, verdict_root

VERDICT_NAME = "production_runtime_env_guard_catalog_phases"
SCHEMA_VERSION = "production_runtime_catalog_phase_verdict.v1"
KEY_VERSION = 1
PHASES = ("launch_profile_catalog", "launch_input_residency")
PACKAGE = "blueprint_pipeline"
# Re-proven by digest through the verdict store; all top-level modules, so finding
# them again imports nothing.
_DIGESTED_MODULES = (
    __name__,
    f"{PACKAGE}.production_runtime_env_guard",
    f"{PACKAGE}.task_evaluation_launch_catalog",
    f"{PACKAGE}.host_resident_launch_inputs",
)
#: Owner windows lapse with no file changing, and the dispatcher refuses a lapsed
#: owner at dispatch, so the served catalog may trail a lapse by at most this.
#: It is also how often inputs no audit event reports are re-proven. At the
#: listener's one-minute cadence it means about four full passes an hour per
#: unit environment instead of one a minute.
MAX_VERDICT_AGE_SECONDS = 15 * 60
#: A verdict is not reused from this long before the earliest grant, authority or
#: window expiry its pass compared with the clock. A pass takes 12-40 s on the
#: host, so a start just inside the margin still decides on the same side of the
#: expiry as the stored pass did, and the starts after it run the full pass.
DEADLINE_MARGIN_SECONDS = 120
#: One verdict per guarded unit environment and release; keep two releases' worth.
RETAINED_VERDICTS = 12
#: Package code that may read ``time.time`` during a stored pass: each reports the
#: instant its answer flips through ``decision_deadlines``, or only stamps a record.
_CLOCK_SITES = frozenset(
    {
        f"{PACKAGE}.task_evaluation_scene_execution_authority.scene_execution_authority_blockers",
        f"{PACKAGE}.task_evaluation_visual_review_authority.validate",
        # Stamps ``created_at_epoch`` on an input-digest cache entry; decides nothing.
        f"{PACKAGE}.validation_verdict_store.store",
    }
)
_ABSENT = "absent"
_MISSING = object()
# Kernel pseudo-files are live state, never a published input.
_PSEUDO_ROOTS = ("/proc/", "/sys/", "/dev/")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_MUTATING_EVENTS = frozenset(
    {
        "os.rename", "os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown",
        "os.chflags", "os.lchflags", "os.utime", "os.truncate", "os.symlink", "os.link",
        "os.setxattr", "os.removexattr", "shutil.copyfile", "shutil.copymode",
        "shutil.copystat", "shutil.copytree", "shutil.rmtree", "shutil.move",
        "shutil.chown", "shutil.make_archive", "shutil.unpack_archive",
    }
)
# Inputs a file fingerprint cannot hold: another process, the network, a database.
_OPAQUE_EVENTS = frozenset(
    {
        "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
        "os.fork", "os.forkpty", "pty.spawn", "socket.connect", "socket.sendto",
        "socket.sendmsg", "socket.getaddrinfo", "urllib.Request", "http.client.connect",
        "sqlite3.connect", "ctypes.dlopen",
    }
)
_STAT = os.stat
_LSTAT = os.lstat
_TIME = time.time
_TIME_NS = time.time_ns

_ACTIVE: _Capture | None = None
_HOOK_INSTALLED = False
_BUSY = threading.local()


@dataclass(frozen=True)
class PhaseScope:
    """The two configured paths the phases project, the guard schema, and the
    environment digest when the guard started (see ``scope``)."""

    profile_dir: str
    catalog_path: str
    guard_schema: str
    environment_sha256: str


def environment_sha256() -> str:
    """The process environment as every key here hashes it."""
    return str(verdict_key(())["environment_sha256"])


def scope(profile_dir: str, catalog_path: str, guard_schema: str) -> PhaseScope:
    """Taken before the guard's import pass, which may edit the environment."""
    return PhaseScope(profile_dir, catalog_path, guard_schema, environment_sha256())


@dataclass
class _Capture:
    scope: PhaseScope
    key: dict[str, Any]
    ignored: tuple[str, ...]
    profile_dir: Any = None
    catalog: Any = None
    sys_path: list[list[Any]] = field(default_factory=list)
    paths: dict[str, Any] = field(default_factory=dict)  # path -> identity at first touch
    listings: dict[str, Any] = field(default_factory=dict)  # directory -> names at first listing
    deadlines: list[float] = field(default_factory=list)  # instants reported by time checks
    started_at: float = 0.0
    refusal: str = ""

    def refuse(self, reason: str) -> None:
        self.refusal = self.refusal or reason

    def skipped(self, path: str) -> bool:
        return path.startswith(_PSEUDO_ROOTS) or any(
            path == root or path.startswith(root + os.sep) for root in self.ignored
        )

    def note_path(self, raw: Any) -> None:
        path = _absolute(raw)
        if path is not None and path not in self.paths and not self.skipped(path):
            self.paths[path] = _identity(path)

    def note_listing(self, raw: Any) -> None:
        if isinstance(raw, int):
            self.refuse("unrecordable:listing_by_descriptor")
            return
        path = _absolute("." if raw is None else raw)
        if path is not None and path not in self.listings and not self.skipped(path):
            self.listings[path] = _listing(path)

    def note_write(self, event: str, args: tuple) -> None:
        # The verdict store's own cache entries are the only writes a stored pass may make.
        named = [_absolute(arg) for arg in args if isinstance(arg, (str, bytes, os.PathLike))]
        if not named or any(path is None or not self.skipped(path) for path in named):
            self.refuse(f"wrote:{event}")

    def note_open(self, raw: Any, mode: Any, flags: Any) -> None:
        if raw is None or isinstance(raw, int):
            return  # a descriptor already opened, and noted, by path
        if (isinstance(flags, int) and flags & _WRITE_FLAGS) or (
            isinstance(mode, str) and any(letter in mode for letter in "wax+")
        ):
            self.note_write("open", (raw,))
        else:
            self.note_path(raw)


def _now() -> float:
    return time.time()


def _euid() -> int:
    return os.geteuid()


def _absolute(raw: Any) -> str | None:
    if raw is None or isinstance(raw, int):
        return None
    try:
        text = os.fsdecode(os.fspath(raw))
    except (TypeError, ValueError):
        return None
    # Not normalized: ``a/link/../b`` names what the kernel resolved, not ``a/b``.
    return text if os.path.isabs(text) else os.path.join(os.getcwd(), text)


def _identity(path: str) -> Any:
    """What a reader of ``path`` depends on: its bytes, or for a directory its kind."""
    try:
        observed = _LSTAT(path)
    except (FileNotFoundError, NotADirectoryError):
        return _ABSENT
    except OSError as exc:
        return ["error", exc.errno]
    mode = observed.st_mode
    if stat.S_ISDIR(mode):
        # Entries come and go without touching what a stat of the directory decided;
        # a listing of it is recorded separately, by name.
        return ["dir", observed.st_dev, observed.st_ino, mode]
    if stat.S_ISLNK(mode):
        try:
            target: Any = os.readlink(path)
        except OSError as exc:
            target = ["error", exc.errno]
        try:
            followed = _STAT(path)
        except OSError as exc:
            return ["link", observed.st_dev, observed.st_ino, observed.st_ctime_ns, target,
                    ["error", exc.errno]]
        return ["link", observed.st_dev, observed.st_ino, observed.st_ctime_ns, target,
                _row(followed)]
    return _row(observed)


def _row(observed: os.stat_result) -> list[Any]:
    if stat.S_ISDIR(observed.st_mode):
        return ["dir", observed.st_dev, observed.st_ino, observed.st_mode]
    return ["file", observed.st_dev, observed.st_ino, observed.st_mode, observed.st_uid,
            observed.st_gid, observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns]


def _listing(directory: str) -> Any:
    try:
        # Bytecode caches come and go beside package sources; they decide nothing here.
        names = sorted(name for name in os.listdir(directory) if name != "__pycache__")
    except FileNotFoundError:
        return _ABSENT
    except NotADirectoryError:
        return ["not_a_directory", _identity(directory)]
    except OSError as exc:
        return ["error", exc.errno]
    return "sha256:" + hashlib.sha256("\n".join(names).encode("utf-8", "surrogateescape")).hexdigest()


def _directory_snapshot(directory: str) -> Any:
    """Every entry of the profile directory, published or not, by full identity."""
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return _ABSENT
    except OSError as exc:
        return ["error", exc.errno]
    return {
        "directory": _identity(directory),
        "entries": [[name, _identity(os.path.join(directory, name))] for name in names],
    }


def _sys_path_listings() -> list[list[Any]]:
    entries = dict.fromkeys(os.path.abspath(entry) if entry else os.getcwd() for entry in sys.path)
    return [[entry, _listing(entry)] for entry in entries]


def _expanded(value: str) -> str:
    return os.path.expanduser(value)


def _key(scope: PhaseScope) -> dict[str, Any]:
    return {
        "key_version": KEY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "guard_schema": scope.guard_schema,
        "release_commit": running_release_commit(),
        "profile_dir": scope.profile_dir,
        "catalog_path": scope.catalog_path,
        # Interpreter, flags, sys.path, import hooks, cwd and the environment digest.
        "runtime": verdict_key(PHASES),
    }


def _audit(event: str, args: tuple) -> None:
    capture = _ACTIVE
    if capture is None or getattr(_BUSY, "on", False):
        return
    _BUSY.on = True  # this hook's own listdir/stat must not re-enter it
    try:
        if event == "open":
            capture.note_open(*(tuple(args) + (None, None, None))[:3])
        elif event in {"os.listdir", "os.scandir"}:
            capture.note_listing(args[0] if args else None)
        elif event in _MUTATING_EVENTS:
            capture.note_write(event, tuple(args))
        elif event in _OPAQUE_EVENTS:
            capture.refuse(f"unrecordable:{event}")
    except Exception:  # noqa: BLE001 - an audit hook must never raise
        capture.refuse("capture_error")
    finally:
        _BUSY.on = False


def _recorded_stat(original: Callable[..., Any]) -> Callable[..., Any]:
    """``os.stat``/``os.lstat`` that note the path: probes raise no audit event."""

    def recorded(path: Any, *args: Any, **kwargs: Any) -> Any:
        capture = _ACTIVE
        if capture is not None and not getattr(_BUSY, "on", False):
            _BUSY.on = True
            try:
                if kwargs.get("dir_fd") is not None:
                    capture.refuse("unrecordable:stat_relative_to_descriptor")
                elif not isinstance(path, int):
                    capture.note_path(path)
            except Exception:  # noqa: BLE001 - recording must never change the probe
                capture.refuse("capture_error")
            finally:
                _BUSY.on = False
        return original(path, *args, **kwargs)

    return recorded


def _recorded_open(original: Callable[..., Any]) -> Callable[..., Any]:
    """``os.open``: its audit event names a path but not the directory it is relative to."""

    def opened(path: Any, flags: int, *args: Any, **kwargs: Any) -> Any:
        capture = _ACTIVE
        if capture is not None and kwargs.get("dir_fd") is not None:
            capture.refuse("unrecordable:open_relative_to_descriptor")
        return original(path, flags, *args, **kwargs)

    return opened


def _recorded_clock(original: Callable[..., Any]) -> Callable[..., Any]:
    """``time.time``/``time.time_ns``: package code reading the clock must report its deadline."""

    def clock(*args: Any, **kwargs: Any) -> Any:
        capture = _ACTIVE
        if capture is not None and not getattr(_BUSY, "on", False):
            caller = sys._getframe(1)
            module = str(caller.f_globals.get("__name__") or "")
            if module == PACKAGE or module.startswith(PACKAGE + "."):
                site = f"{module}.{caller.f_code.co_name}"
                if site not in _CLOCK_SITES:
                    capture.refuse(f"unrecordable:clock:{site}")
        return original(*args, **kwargs)

    return clock


def _start(scope: PhaseScope) -> _Capture:
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit)  # permanent by design; inert outside a capture
        _HOOK_INSTALLED = True
    root = str(verdict_root())
    ignored = tuple(dict.fromkeys((os.path.abspath(root), os.path.realpath(root))))
    capture = _Capture(scope=scope, key=_key(scope), ignored=ignored, started_at=_now())
    capture.profile_dir = _directory_snapshot(_expanded(scope.profile_dir))
    capture.catalog = _identity(_expanded(scope.catalog_path))
    capture.sys_path = _sys_path_listings()
    return capture


@contextmanager
def capturing(scope: PhaseScope | None) -> Iterator[_Capture | None]:
    """Run the phases under a recorder of everything they read and do.

    Yields ``None`` (and records nothing) without a scope or when the recorder
    cannot start; the phases run either way.
    """

    global _ACTIVE
    capture = None
    if scope is not None:
        try:
            capture = _start(scope)
        except Exception:  # noqa: BLE001 - never store, but always run the pass
            capture = None
    if capture is None:
        yield None
        return
    saved = (os.stat, os.lstat, os.open, time.time, time.time_ns)
    os.stat, os.lstat, os.open = _recorded_stat(saved[0]), _recorded_stat(saved[1]), _recorded_open(saved[2])
    time.time, time.time_ns = _recorded_clock(saved[3]), _recorded_clock(saved[4])
    _ACTIVE = capture
    try:
        with collecting_deadlines() as capture.deadlines:
            yield capture
    finally:
        _ACTIVE = None
        os.stat, os.lstat, os.open, time.time, time.time_ns = saved


def _entry_unsafe(path: Path) -> str:
    try:
        observed = _LSTAT(path)
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "entry_unreadable"
    if stat.S_ISLNK(observed.st_mode):
        return "entry_symlinked"
    if not stat.S_ISREG(observed.st_mode):
        return "entry_not_a_regular_file"
    if observed.st_uid != _euid():
        return "entry_foreign_owned"
    if stat.S_IMODE(observed.st_mode) & (stat.S_IWGRP | stat.S_IWOTH):
        return "entry_group_or_world_writable"
    return ""


def _entry_row(path: Path) -> tuple[int, ...]:
    observed = _LSTAT(path)
    return (observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns,
            observed.st_ctime_ns, observed.st_mode, observed.st_uid)


def _payload(stored: Any) -> dict[str, Any] | None:
    if not isinstance(stored, dict) or stored.get("schema_version") != SCHEMA_VERSION:
        return None
    phases = stored.get("phases")
    if (
        not isinstance(phases, dict)
        or set(phases) != set(PHASES)
        or not all(isinstance(phases[name], dict) for name in PHASES)
        # Only a pass that found nothing to repair is a pass worth skipping.
        or phases["launch_profile_catalog"].get("status") != "consistent"
    ):
        return None
    stored_at = stored.get("stored_at_epoch")
    if isinstance(stored_at, bool) or not isinstance(stored_at, (int, float)):
        return None
    deadline = stored.get("decision_deadline_epoch", _MISSING)
    if deadline is _MISSING or (deadline is not None and (
        isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)
    )):
        return None
    for rows in (stored.get("paths"), stored.get("listings"), stored.get("sys_path"),
                 stored.get("code_files")):
        if not isinstance(rows, list) or not all(
            isinstance(row, list) and len(row) == 2 and isinstance(row[0], str) for row in rows
        ):
            return None
    if not stored["code_files"] or "profile_dir" not in stored or "catalog" not in stored:
        return None
    return stored


def _moved(payload: Mapping[str, Any], scope: PhaseScope) -> str:
    if _directory_snapshot(_expanded(scope.profile_dir)) != payload["profile_dir"]:
        return "profile_directory_changed"
    if _identity(_expanded(scope.catalog_path)) != payload["catalog"]:
        return "catalog_changed"
    if _sys_path_listings() != payload["sys_path"]:
        return "python_path_changed"
    for path, identity in payload["code_files"]:
        if _identity(path) != identity:
            return f"code_changed:{path}"
    for path, identity in payload["paths"]:
        if _identity(path) != identity:
            return f"path_changed:{path}"
    for directory, names in payload["listings"]:
        if _listing(directory) != names:
            return f"listing_changed:{directory}"
    return ""


def _reuse(scope: PhaseScope) -> tuple[dict[str, Any] | None, str]:
    key = _key(scope)
    path = entry_path(name=VERDICT_NAME, key=key)
    unsafe = _entry_unsafe(path)
    if unsafe:
        return None, unsafe
    before = _entry_row(path)
    stored = lookup(name=VERDICT_NAME, key=key)
    if _entry_row(path) != before:
        return None, "entry_replaced"  # the checks above saw another file
    payload = _payload(stored)
    if payload is None:
        return None, "entry_stale_or_invalid"
    now = _now()
    if not 0 <= now - float(payload["stored_at_epoch"]) <= MAX_VERDICT_AGE_SECONDS:
        return None, "expired"
    deadline = payload["decision_deadline_epoch"]
    if deadline is not None and now >= float(deadline) - DEADLINE_MARGIN_SECONDS:
        return None, "decision_deadline_reached"
    moved = _moved(payload, scope)
    if moved:
        return None, moved
    return {name: dict(payload["phases"][name]) for name in PHASES}, ""


def reused_phases(scope: PhaseScope) -> tuple[dict[str, Any] | None, str]:
    """The stored ready phase details, or ``None`` and why the full pass must run."""
    try:
        return _reuse(scope)
    except Exception as exc:  # noqa: BLE001 - an unusable cache proves from scratch
        return None, f"error:{type(exc).__name__}"


def _package_sources() -> list[str]:
    """Every loaded source file inside the package, ``__main__`` (the guard as a unit runs it) included."""
    package = sys.modules.get(PACKAGE)
    root = os.path.dirname(os.path.abspath(str(getattr(package, "__file__", "") or "")))
    found: set[str] = set()
    for module in list(sys.modules.values()):
        try:
            origin = getattr(module, "__file__", None)
        except Exception:  # noqa: BLE001 - a module answering every attribute names no file
            continue
        if isinstance(origin, str) and origin.endswith(".py"):
            origin = os.path.abspath(origin)
            if origin.startswith(root + os.sep):
                found.add(origin)
    return sorted(found)


def _code_rows() -> list[dict[str, str]] | None:
    """Digest rows the store re-proves; found by spec, so ``python -m`` mains count too."""
    import importlib.util

    rows = []
    for name in _DIGESTED_MODULES:
        spec = importlib.util.find_spec(name)
        origin = getattr(spec, "origin", None)
        if not origin or not str(origin).endswith(".py"):
            return None
        digest = hashlib.sha256(Path(origin).read_bytes()).hexdigest()
        rows.append({"module": name, "sha256": f"sha256:{digest}"})
    return rows


def _code_moved_since_start(sources: list[str]) -> bool:
    """A package source changed after this process began: it may run older bytes."""
    started = _process_started_ns()
    for origin in sources:
        try:
            if _STAT(origin).st_ctime_ns > started:
                return True
        except OSError:
            return True
    return False


def _retain_newest(root: Path) -> None:
    """Drop all but the newest retained verdicts of this name; they are only cache."""
    try:
        entries = []
        for path in root.glob(f"{VERDICT_NAME}-*.json"):
            if not path.is_symlink():
                entries.append((path.lstat().st_mtime_ns, path.name, path))
        for _mtime, _name, stale in sorted(entries, reverse=True)[RETAINED_VERDICTS:]:
            stale.unlink(missing_ok=True)
    except OSError:
        return


def _record(capture: _Capture, phases: Mapping[str, Mapping[str, Any]]) -> str:
    if capture.refusal:
        return capture.refusal
    scope = capture.scope
    if capture.key["runtime"]["environment_sha256"] != scope.environment_sha256:
        # Keyed on an environment the guard did not start with (the import pass
        # edits LD_LIBRARY_PATH); a later start would never look it up.
        return "environment_changed_since_guard_start"
    for path, identity in capture.paths.items():
        if _identity(path) != identity:
            return f"changed_during_pass:{path}"
    for directory, names in capture.listings.items():
        if _listing(directory) != names:
            return f"changed_during_pass:{directory}"
    profile_dir = _directory_snapshot(_expanded(scope.profile_dir))
    catalog = _identity(_expanded(scope.catalog_path))
    if (profile_dir, catalog) != (capture.profile_dir, capture.catalog):
        return "changed_during_pass:profile_directory_or_catalog"
    if _sys_path_listings() != capture.sys_path:
        return "changed_during_pass:python_path"
    sources = _package_sources()
    if not sources or _code_moved_since_start(sources):
        return "code_changed_after_process_start"
    code = _code_rows()
    if not code:
        return "code_unidentified"
    with touched_files() as rows:
        try:
            sha256_file(Path(_expanded(scope.catalog_path)).resolve())
        except (OSError, ValueError):
            pass  # a symlinked parent: the catalog stays bound by its identity alone
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phases": {name: dict(phases[name]) for name in PHASES},
        "profile_dir": profile_dir,
        "catalog": catalog,
        "paths": [[path, identity] for path, identity in sorted(capture.paths.items())],
        "listings": [[path, names] for path, names in sorted(capture.listings.items())],
        "sys_path": capture.sys_path,
        "code_files": [[path, _identity(path)] for path in sources],
        # The earliest instant ahead at which a time check of this pass flips.
        "decision_deadline_epoch": min(
            (instant for instant in capture.deadlines if instant > capture.started_at), default=None
        ),
        "stored_at_epoch": _now(),
    }
    written = store(
        name=VERDICT_NAME, key=capture.key, files=rows, verdict=payload,
        source_commit=str(capture.key.get("release_commit") or ""), code=code,
    )
    if written is None:
        return "store_failed"
    _retain_newest(verdict_root())
    return ""


def record(capture: _Capture | None, phases: Mapping[str, Mapping[str, Any]]) -> tuple[bool, str]:
    """Store a ready, write-free pass; ``(False, reason)`` when it is not stored."""
    if capture is None:
        return False, "capture_unavailable"
    try:
        reason = _record(capture, phases)
    except Exception as exc:  # noqa: BLE001 - a verdict that cannot be stored is not reused
        return False, f"error:{type(exc).__name__}"
    return not reason, reason


__all__ = [
    "DEADLINE_MARGIN_SECONDS",
    "MAX_VERDICT_AGE_SECONDS",
    "PHASES",
    "PhaseScope",
    "RETAINED_VERDICTS",
    "VERDICT_NAME",
    "capturing",
    "environment_sha256",
    "record",
    "reused_phases",
    "scope",
]
