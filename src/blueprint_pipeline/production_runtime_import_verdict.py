"""Reuse the guard's entrypoint-import verdict while nothing that decides it moved.

Every control-plane unit runs the production runtime env guard as its
ExecStartPre, and the oneshot units (the Pub/Sub handoff listener among them)
start every minute. Each run imported all control-plane entrypoints -- about
1,070 modules, compiled from source every time because units cannot write
bytecode into the read-only release tree. That repeated, unchanged proof is
what this module stops repeating; what is proven is unchanged.

Whether a module imports is decided by the interpreter and its flags,
``sys.path`` and the import hooks, the process environment, the bytes the
import system and module bodies read, the native libraries mapped into the
process, and the directory listings that resolved every name. The import pass
runs with the import caches invalidated and under an audit hook that records,
at first touch, the identity of every file opened or imported (or its absence)
and the listing of every directory searched. The listings of every ``sys.path``
entry, loaded package and loaded module's directory are recorded too, since
the guard's own imports filled those caches before the pass. A verdict is
stored only for a fully successful pass whose recorded files and listings did
not change before they were hashed, and reused only while:

* the interpreter, flags, ``sys.path``, import hooks, working directory and
  environment are identical (systemd's per-invocation variables count by
  presence, never value);
* every recorded directory lists the same names (``absent`` stays absent, a
  path that is a file stays that file), so nothing new can shadow a module or
  satisfy an import that once failed;
* every recorded file -- module sources, extension modules and the native
  libraries they map, ``.pth`` files and data read at import -- and this
  verification code re-prove through ``validation_verdict_store``, and every
  path that was absent is still absent.

Nothing is stored when any recorded file or directory changed after this
process started (a pip install racing a unit start could otherwise bind bytes
the process never loaded), and every symlinked path must still resolve where it
did. A few inputs raise no audit event at all (a bare ``stat``, ``which`` or a C
library's own reads); ``MAX_VERDICT_AGE_SECONDS`` bounds how long a change to
one of those can go unnoticed. A failed pass is never stored, so a broken
environment is re-checked every start. A stored verdict is a cache, never
evidence.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import stat
import sys
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .validation_file_digests import sha256_file, touched_files
from .validation_verdict_store import lookup, store, verdict_root

VERDICT_NAME = "production_runtime_env_guard_entrypoint_imports"
KEY_VERSION = 3
# A full import pass still runs at least this often per unit environment.
MAX_VERDICT_AGE_SECONDS = 3600
# Each release and unit environment keys its own verdict (about 1.6 MB of
# file rows). Keep enough for every guarded unit across two releases.
RETAINED_VERDICTS = 12
# systemd sets these per invocation. Their values never decide an import; a
# module that branches on whether one is set still sees the same presence.
PER_INVOCATION_ENVIRONMENT = frozenset(
    {
        "INVOCATION_ID",
        "JOURNAL_STREAM",
        "SYSTEMD_EXEC_PID",
        "MANAGERPID",
        "TRIGGER_UNIT",
        "TRIGGER_PATH",
        "TRIGGER_TIMER_REALTIME_USEC",
        "TRIGGER_TIMER_MONOTONIC_USEC",
        "LISTEN_PID",
        "LISTEN_FDS",
        "LISTEN_FDNAMES",
        "NOTIFY_SOCKET",
        "WATCHDOG_PID",
        "WATCHDOG_USEC",
    }
)
# Kernel pseudo-files are live state, never part of what an import read.
_PSEUDO_ROOTS = ("/proc/", "/sys/", "/dev/")
# Memory that /proc/self/maps names like a path: /dev/zero and /dev/shm shared
# mappings, memfd and SysV segments all read "(deleted)" without being files.
_NOT_FILE_MAPPINGS = (*_PSEUDO_ROOTS, "/memfd:", "/SYSV")
_CODE_MODULES = (__name__, "blueprint_pipeline.production_runtime_env_guard")
_ABSENT = "absent"


class _Capture:
    def __init__(self) -> None:
        self.files: dict[str, Any] = {}  # path -> identity first seen, or _ABSENT
        self.directories: dict[str, str | None] = {}  # path -> listing first seen
        self.unrecordable = False


_ACTIVE: _Capture | None = None
_HOOK_INSTALLED = False
_IN_HOOK = threading.local()
_MODULE_LOADED_NS = time.time_ns()


def _now() -> float:
    return time.time()


def _process_started_ns() -> int:
    """When this process began, never later than the truth (Linux procfs)."""
    try:
        fields = Path("/proc/self/stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
        started_ticks = int(fields[19])  # field 22, counted after "pid (comm)"
        boot = next(
            int(line.split()[1])
            for line in Path("/proc/stat").read_text(encoding="ascii").splitlines()
            if line.startswith("btime ")
        )
        return boot * 1_000_000_000 + started_ticks * 1_000_000_000 // os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return _MODULE_LOADED_NS  # no procfs: when this module loaded


def _path(value: Any) -> str | None:
    if value is None or isinstance(value, int):
        return None
    return os.path.abspath(os.fsdecode(value))


def _identity(path: str) -> Any:
    try:
        observed = os.stat(path)
    except FileNotFoundError:
        return _ABSENT
    except OSError:
        return None
    return [observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns,
            observed.st_ctime_ns, observed.st_mode]


def _listing_sha256(directory: str) -> str | None:
    """Names that resolve imports here; ``absent`` and a plain file are states too."""
    try:
        names = sorted(name for name in os.listdir(directory) if name != "__pycache__")
    except FileNotFoundError:
        return _ABSENT  # e.g. a sys.path zip that does not exist yet
    except NotADirectoryError:
        identity = _identity(directory)
        return None if identity in (None, _ABSENT) else "file:" + hashlib.sha256(
            repr(identity).encode("utf-8")).hexdigest()
    except OSError:
        return None
    return hashlib.sha256("\n".join(names).encode("utf-8", "surrogateescape")).hexdigest()


def _audit(event: str, args: tuple) -> None:
    capture = _ACTIVE
    if capture is None or getattr(_IN_HOOK, "busy", False):
        return
    _IN_HOOK.busy = True  # the hook's own stat/listdir must not re-enter it
    try:
        if event in {"open", "import"}:
            path = _path(args[0] if event == "open" else args[1])
            if path is not None and path not in capture.files:
                capture.files[path] = _identity(path)
        elif event in {"os.listdir", "os.scandir"}:
            path = _path(args[0] if args[0] is not None else ".")
            if path is not None and path not in capture.directories:
                capture.directories[path] = _listing_sha256(path)
    except Exception:  # noqa: BLE001 - an audit hook must never raise
        capture.unrecordable = True
    finally:
        _IN_HOOK.busy = False


def _module_attribute(module: Any, name: str) -> Any:
    # Some modules answer every attribute (torch._classes returns an object for
    # __path__ and __file__) or raise from __getattr__: neither names a file.
    try:
        return getattr(module, name, None)
    except Exception:  # noqa: BLE001 - an attribute that cannot be read is not a path
        return None


def _loaded_module_files() -> set[str]:
    found = set()
    for name in list(sys.modules):
        origin = _module_attribute(sys.modules.get(name), "__file__")
        if isinstance(origin, str):
            found.add(os.path.abspath(origin))
    return found


def _resolution_directories() -> set[str]:
    """Every directory whose listing decides how a loaded or missing name resolves."""
    found = {os.path.abspath(entry) if entry else os.getcwd() for entry in sys.path}
    for name in list(sys.modules):
        entries = _module_attribute(sys.modules.get(name), "__path__")
        if entries is None or isinstance(entries, (str, bytes)):
            continue
        try:
            found.update(os.path.abspath(entry) for entry in list(entries) if isinstance(entry, str))
        except TypeError:
            continue  # not a package search path (e.g. torch._classes' namespace)
    found.update(os.path.dirname(path) for path in _loaded_module_files())
    return found


def _mapped_files(maps_text: str | None = None) -> set[str] | None:
    """Native files mapped into this process; ``None`` when one was replaced under it."""
    if maps_text is None:
        try:
            maps_text = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return set()  # no procfs (not Linux): extension modules are still recorded
    found: set[str] = set()
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6 or not parts[5].startswith("/"):
            continue
        path, deleted = parts[5], parts[5].endswith(" (deleted)")
        if deleted:
            path = path[: -len(" (deleted)")]
        if path.startswith(_NOT_FILE_MAPPINGS):
            continue  # anonymous, shm or memfd memory: never a library on disk
        if deleted:
            return None  # the process runs bytes no longer at that path
        found.add(path)
    return found


@contextmanager
def _capturing() -> Iterator[_Capture]:
    global _ACTIVE, _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit)  # permanent by design; inert outside a capture
        _HOOK_INSTALLED = True
    capture = _Capture()
    try:
        # Cached listings would let the pass resolve names without listing them.
        importlib.invalidate_caches()
        for path in _loaded_module_files():
            capture.files[path] = _identity(path)
        for directory in _resolution_directories():
            capture.directories[directory] = _listing_sha256(directory)
    except Exception:  # noqa: BLE001 - never store, but always run the pass
        capture.unrecordable = True
    previous, _ACTIVE = _ACTIVE, capture
    try:
        yield capture
    finally:
        _ACTIVE = previous


def _environment_sha256() -> str:
    rows = sorted(
        (name, "<set>" if name in PER_INVOCATION_ENVIRONMENT else value)
        for name, value in os.environ.items()
    )
    return hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()


def _hook_names(hooks: Iterable[Any]) -> list[str]:
    names = []
    for hook in hooks:
        kind = hook if isinstance(hook, type) else type(hook)
        names.append(f"{kind.__module__}.{kind.__qualname__}")
    return names


def verdict_key(entrypoints: Iterable[str]) -> dict[str, Any]:
    """Everything outside the recorded files that decides whether these import."""
    return {
        "key_version": KEY_VERSION,
        "entrypoints": list(entrypoints),
        "python": {
            "version": sys.version,
            "cache_tag": sys.implementation.cache_tag,
            "executable": os.path.realpath(sys.executable),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "flags": list(sys.flags),
        },
        "cwd": os.getcwd(),
        "sys_path": [os.path.abspath(entry) if entry else os.getcwd() for entry in sys.path],
        "meta_path": _hook_names(sys.meta_path),
        "path_hooks": _hook_names(sys.path_hooks),
        "environment_sha256": _environment_sha256(),
    }


def _code_rows() -> list[dict[str, str]] | None:
    rows = []
    for name in _CODE_MODULES:
        origin = getattr(sys.modules.get(name), "__file__", None)
        if not origin:
            return None
        rows.append(
            {"module": name, "sha256": "sha256:" + hashlib.sha256(Path(origin).read_bytes()).hexdigest()}
        )
    return rows


def _pth_files() -> set[str]:
    found = set()
    for entry in sys.path:
        try:
            found.update(str(path) for path in Path(entry or ".").glob("*.pth"))
        except OSError:
            continue
    return found


def _links(paths: Iterable[str]) -> list[dict[str, str]]:
    """Where every symlinked path resolved, so a retargeted link invalidates."""
    rows = []
    for path in sorted(set(paths)):
        real = os.path.realpath(path)
        if real != path:
            rows.append({"path": path, "real": real})
    return rows


def _store(key: dict[str, Any], detail: dict[str, Any], capture: _Capture, started_ns: int) -> bool:
    if capture.unrecordable:
        return False
    mapped = _mapped_files()
    if mapped is None:
        return False
    # Modules, packages and libraries loaded during the pass, and .pth files.
    for path in (*_loaded_module_files(), *mapped, *_pth_files()):
        capture.files.setdefault(os.path.abspath(path), _identity(os.path.abspath(path)))
    for directory in _resolution_directories():
        capture.directories.setdefault(directory, _listing_sha256(directory))
    directories = []
    for directory, seen in sorted(capture.directories.items()):
        if os.path.realpath(directory).startswith(_PSEUDO_ROOTS):
            continue
        if seen is None or _listing_sha256(directory) != seen:
            return False  # unlistable, or it changed after the pass searched it
        try:
            if os.stat(directory).st_ctime_ns > started_ns:
                return False  # its entries moved after this process began
        except FileNotFoundError:
            pass
        except OSError:
            return False
        directories.append({"path": directory, "listing_sha256": seen})
    absent: list[str] = []
    with touched_files() as rows:
        for path, seen in sorted(capture.files.items(), key=lambda item: item[0]):
            real = os.path.realpath(path)
            if real.startswith(_PSEUDO_ROOTS):
                continue
            if seen == _ABSENT:
                if os.path.lexists(path):
                    return False  # created after the pass looked for it
                absent.append(path)
                continue
            if seen is None or _identity(path) != seen:
                return False  # unreadable, or changed after it was read
            if int(seen[4]) > started_ns:
                return False  # changed after this process began: it may run older bytes
            if not stat.S_ISREG(int(seen[5])):
                continue  # a directory or device the pass opened
            try:
                sha256_file(Path(real))
            except (OSError, ValueError):
                return False
    code = _code_rows()
    if not code:
        return False
    verdict = {
        "detail": detail,
        "directories": directories,
        "absent_paths": absent,
        "links": _links([*capture.files, *capture.directories]),
        "stored_at_epoch": _now(),
    }
    if store(name=VERDICT_NAME, key=key, files=rows, verdict=verdict, code=code) is None:
        return False
    _retain_newest(verdict_root())
    return True


def _retain_newest(root: Path) -> None:
    """Drop all but the newest retained verdicts of this name; they are only cache."""
    try:
        entries = []
        for path in root.glob(f"{VERDICT_NAME}-*.json"):
            observed = path.lstat()
            if not path.is_symlink():
                entries.append((observed.st_mtime_ns, path.name, path))
        for _mtime, _name, stale in sorted(entries, reverse=True)[RETAINED_VERDICTS:]:
            stale.unlink(missing_ok=True)
    except OSError:
        return


def _reusable(stored: Any) -> dict[str, Any] | None:
    if not isinstance(stored, dict) or not isinstance(stored.get("detail"), dict):
        return None
    directories, absent = stored.get("directories"), stored.get("absent_paths")
    links, stored_at = stored.get("links"), stored.get("stored_at_epoch")
    if (
        not isinstance(directories, list)
        or not isinstance(absent, list)
        or not isinstance(links, list)
        or not isinstance(stored_at, (int, float))
        or not 0 <= _now() - float(stored_at) <= MAX_VERDICT_AGE_SECONDS
    ):
        return None
    for row in links:
        if not isinstance(row, dict) or os.path.realpath(str(row.get("path"))) != row.get("real"):
            return None
    for row in directories:
        if not isinstance(row, dict) or _listing_sha256(str(row.get("path"))) != row.get(
            "listing_sha256"
        ):
            return None
    if any(not isinstance(path, str) or os.path.lexists(path) for path in absent):
        return None
    detail = stored["detail"]
    return detail if detail.get("importable") is True else None


def checked_entrypoint_imports(
    entrypoints: Iterable[str],
    check: Callable[[], tuple[dict[str, Any], list[str]]],
    *,
    started_ns: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Run ``check`` (the import pass), or reuse its unchanged successful verdict.

    The verdict layer is only a cache: any failure in it runs the full pass and
    stores nothing, so it can make the guard slower but never stop a unit.
    """
    key: dict[str, Any] | None = None
    try:
        started = _process_started_ns() if started_ns is None else started_ns
        key = verdict_key(entrypoints)
        reused = _reusable(lookup(name=VERDICT_NAME, key=key))
    except Exception:  # noqa: BLE001 - an unusable cache proves from scratch
        reused = None
    if reused is not None:
        return {**reused, "verdict_reused": True}, []
    with _capturing() as capture:
        detail, blockers = check()
    recorded = False
    if key is not None and not blockers and detail.get("importable") is True:
        try:
            recorded = _store(key, detail, capture, started)
        except Exception:  # noqa: BLE001 - a verdict that cannot be stored is not reused
            recorded = False
    return {**detail, "verdict_reused": False, "verdict_recorded": recorded}, blockers
