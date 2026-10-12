"""Persisted validator verdicts, reusable only when the code and every consulted byte are unchanged.

The 2026-09-12 scene-840938 run measured the cost of re-proving retained GPU evidence
from bytes at every hand-off: one full prefix validation read about 40 GB against a
4.6 GB artifact tree and took 15 minutes, and the chain repeats it at every stage
boundary, every deploy and every reuse candidate. Renting and running the GPU was
the fast part.

A stored verdict is a cache, never evidence. It is reused only when:

* every ``blueprint_pipeline`` module that was loaded when the verdict was computed
  still has byte-identical source (validator code is part of the verdict, resolved
  through the import system without importing anything), and the caller's key -- the
  content digests of its inputs -- matches; a deploy that leaves the validators
  untouched keeps the verdicts, a deploy that changes one recomputes;
* every file the validator consulted through the evidence readers (``sha256_file`` and
  ``read``) is re-proven: files at or above ``MINIMUM_BYTES`` by full stat identity
  (device, inode, size, nanosecond mtime and kernel-owned ctime, ownership, mode, link
  count) and re-hashed when that identity moved; smaller files always re-hashed. A
  re-hash that proves the same bytes under a moved identity is kept in the entry: a
  storage move left 149 of 151 large launch-input verdicts and 240 GB of prefix-adoption
  rows on identities that no longer held, and every lookup re-hashed them from scratch.

Any unreadable, malformed, foreign or stale entry simply recomputes. Writing is best
effort: a read-only or missing root disables persistence and changes nothing else.
"""
from __future__ import annotations

from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import threading
import time
import weakref

from .decision_evidence_contracts import canonical_digest, canonical_json
from .validation_file_digests import MINIMUM_BYTES, _identity, sha256_file, touched_files

SCHEMA_VERSION = "blueprint_validation_verdict.v1"
ROOT_ENV = "BLUEPRINT_VALIDATION_VERDICT_ROOT"
DEFAULT_ROOT = "/var/lib/blueprint/pipeline-control-plane/validation-verdicts"
PACKAGE = "blueprint_pipeline"
_MODULE_DIGESTS: dict[tuple, str] = {}
_DEPENDENCIES = ContextVar("validator_code_dependencies", default=())
MISS = object()
DEPENDENCY_VERSION = 2
_DEFAULT_CODE = object()
# A finished validation's tracer -> the thread hook it replaced. The hook is process-wide and
# validations in other threads end in any order, so a saved hook may belong to one already over.
_ENDED_TRACERS = weakref.WeakKeyDictionary()
_HOOK_LOCK = threading.Lock()  # guards the process hook's bookkeeping; tracers never take it


def _live_hook(hook):
    try:
        while hook in _ENDED_TRACERS:
            hook = _ENDED_TRACERS[hook]
    except TypeError:  # an unhashable hook, such as a callable dataclass, is never a tracer of ours
        return hook
    return hook


def verdict_root() -> Path:
    return Path(os.getenv(ROOT_ENV) or DEFAULT_ROOT)


def _entry_path(root: Path, name: str, key) -> Path:
    token = hashlib.sha256(canonical_json({"name": name, "key": key}).encode("utf-8")).hexdigest()[:40]
    return root / f"{name}-{token}.json"


def _module_digest(path: Path) -> str | None:
    try:
        observed = path.stat()
        cache_key = (str(path), observed.st_mtime_ns, observed.st_ctime_ns, observed.st_size)
        if cache_key not in _MODULE_DIGESTS:
            _MODULE_DIGESTS[cache_key] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        return _MODULE_DIGESTS[cache_key]
    except OSError:
        return None


def _package_root() -> Path:
    import blueprint_pipeline
    return Path(blueprint_pipeline.__file__).resolve().parent


def _module_name(path: Path, root: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(root)
    except (OSError, ValueError):
        return None
    if relative.suffix != ".py":
        return None
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([PACKAGE, *parts]) if parts else PACKAGE


def executed_code_identity(run, *, always=()):
    """Run ``run`` under a call tracer; return ``(result, rows)`` for every package module whose code executed.

    Keying a verdict on every loaded module made every deploy discard every
    verdict: the 2026-09-13 scene-840938 run re-derived its adopted ten-stage
    chain for about eighteen minutes after each of eight deploys that never
    touched a validator. The code a verdict depends on is the code that ran
    while it was computed; ``always`` names modules whose data the validator
    reads without calling into them.
    """
    import blueprint_pipeline
    roots = [_package_root(), *(Path(p).resolve() for p in blueprint_pipeline.__path__)]
    from .validation_code_dependencies import _DATA_CACHE, frame_dependencies
    data_token = _DATA_CACHE.set({})
    names = set()
    seen = set()
    failures = []
    token = _DEPENDENCIES.set(_DEPENDENCIES.get() + (names,))
    collectors = _DEPENDENCIES.get()
    owner_thread = threading.get_ident()
    ended = False
    if threading.active_count() != 1:
        for collector in collectors:
            collector.add("")

    def tracer(frame, event, arg):
        if ended:  # a thread started during the run outlived it: it gets the hook the run replaced
            hook = _live_hook(previous_threads)
            sys.settrace(hook)
            return hook(frame, event, arg) if hook is not None else None
        if threading.get_ident() != owner_thread:
            for collector in collectors:
                collector.add("")
        if event == "call" and frame.f_code not in seen:
            seen.add(frame.f_code)
            module = next((name for root in roots
                           if (name := _module_name(Path(frame.f_code.co_filename), root))), None)
            if module:
                try:
                    dependencies = {module} | frame_dependencies(frame, module)
                    for collector in _DEPENDENCIES.get():
                        collector.update(dependencies)
                except Exception:  # refuses persistence; never raises into the traced code
                    failures.append(True)
                    for collector in collectors:
                        collector.add("")
        return None

    previous = sys.gettrace()
    with _HOOK_LOCK:
        previous_threads = threading.gettrace()
        threading.settrace(tracer)
    sys.settrace(tracer)
    try:
        result = run()
    finally:
        sys.settrace(previous)
        _DEPENDENCIES.reset(token)
        _DATA_CACHE.reset(data_token)
        with _HOOK_LOCK:
            ended = True
            _ENDED_TRACERS[tracer] = previous_threads
            if threading.gettrace() is tracer:  # otherwise a later validation still running restores it
                threading.settrace(_live_hook(previous_threads))
    if failures or "" in names:
        return result, None
    names.update(str(name) for name in always if str(name).startswith(PACKAGE))
    names.update({__name__, "blueprint_pipeline.validation_file_digests",
                  "blueprint_pipeline.validation_code_dependencies"})
    for collector in _DEPENDENCIES.get():
        collector.update(names)
    rows = []
    for name in sorted(names):
        origin = sys.modules[name].__file__ if name in sys.modules else None
        if not origin:
            try:
                import importlib.util
                spec = importlib.util.find_spec(name)
                origin = getattr(spec, "origin", None) if spec else None
            except (ImportError, ValueError):
                origin = None
        if not origin or not str(origin).endswith(".py"):
            continue
        digest = _module_digest(Path(origin))
        if digest is None:
            return result, None
        rows.append({"module": name, "sha256": digest})
    return result, (rows or None)


def code_identity() -> list[dict] | None:
    """Digest of every loaded ``blueprint_pipeline`` module's source, in module order."""
    rows = []
    for name in sorted(sys.modules):
        if name != PACKAGE and not name.startswith(PACKAGE + "."):
            continue
        origin = getattr(sys.modules[name], "__file__", None)
        if not origin or not str(origin).endswith(".py"):
            continue
        digest = _module_digest(Path(origin))
        if digest is None:
            return None
        rows.append({"module": name, "sha256": digest})
    return rows or None


def _code_unchanged(rows) -> bool:
    import importlib.util
    for row in rows:
        try:
            spec = importlib.util.find_spec(row["module"])
        except (ImportError, ValueError):
            return False
        origin = getattr(spec, "origin", None) if spec is not None else None
        if not origin or _module_digest(Path(origin)) != row["sha256"]:
            return False
    return True


def _reproven(row) -> dict | None:
    """``row`` as its file proves now, or ``None`` when the bytes changed.

    A re-hash returns the identity ``sha256_file`` held for the whole read, so an entry
    that keeps it is exactly as strong as one stored fresh at that moment.
    """
    path = Path(row["path"])
    try:
        observed = path.lstat()
    except OSError:
        return None
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        return None
    if observed.st_size >= MINIMUM_BYTES and list(_identity(observed)) == list(row["identity"]):
        return row
    try:
        with touched_files() as hashed:
            digest = sha256_file(path)
    except (OSError, ValueError):
        return None
    if digest != row["sha256"] or not hashed:
        return None
    return {"path": row["path"], "identity": list(hashed[-1]["identity"]), "sha256": digest}


def _write_entry(path: Path, value: dict, *, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _keep_reproven_identities(path: Path, read: bytes, value: dict, rows: list[dict]) -> None:
    """Rewrite a reused entry with the identities its large files re-proved under.

    Best effort, as every write here: only the entry's own owner rewrites it (a root
    deploy step would leave one the units cannot read), and an entry another process
    replaced since it was read is left to that process.
    """
    try:
        if path.stat().st_uid != os.geteuid() or path.read_bytes() != read:
            return
        refreshed = {key: item for key, item in value.items() if key != "entry_digest"}
        refreshed["files"] = rows
        refreshed["entry_digest"] = canonical_digest(refreshed, digest_field="entry_digest")
        _write_entry(path, refreshed, mode=stat.S_IMODE(path.stat().st_mode))
    except (OSError, TypeError, ValueError):
        return


def lookup(*, name: str, key, root: Path | None = None, missing=None):
    """Return the stored verdict when the code and every consulted byte are unchanged, else ``None``."""
    path = _entry_path(root or verdict_root(), name, key)
    try:
        if path.is_symlink() or not path.is_file():
            return missing
        read = path.read_bytes()
        value = json.loads(read.decode("utf-8"))
    except (OSError, ValueError):
        return missing
    if (not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION
            or value.get("dependency_version") != DEPENDENCY_VERSION
            or value.get("name") != name
            or value.get("key") != json.loads(canonical_json({"key": key}))["key"]
            or value.get("entry_digest") != canonical_digest(value, digest_field="entry_digest")
            or not isinstance(value.get("files"), list) or not isinstance(value.get("code"), list)
            or not value["code"]):
        return missing
    if not all(isinstance(row, dict) and isinstance(row.get("module"), str) for row in value["code"]):
        return missing
    if not _code_unchanged(value["code"]):
        return missing
    rows = []
    for row in value["files"]:
        proven = _reproven(row) if isinstance(row, dict) else None
        if proven is None:
            return missing
        rows.append(proven)
    from .validation_file_digests import record_touched
    for collector in _DEPENDENCIES.get():
        collector.update(row["module"] for row in value["code"])
    for row in rows:
        record_touched(row["path"], row["identity"], row["sha256"])
    # A large file re-proven by its bytes keeps the identity it proved under; small
    # files are re-hashed on every lookup, so their identities are not worth a write.
    if any(new is not old and new["identity"][2] >= MINIMUM_BYTES for new, old in zip(rows, value["files"])):
        _keep_reproven_identities(path, read, value, rows)
    return value["verdict"]


def store(*, name: str, key, files, verdict, source_commit: str = "", root: Path | None = None,
          code=_DEFAULT_CODE) -> Path | None:
    """Persist one verdict with the exact files and code it consulted; ``None`` when unavailable."""
    code = code_identity() if code is _DEFAULT_CODE else code
    if not code:
        return None
    consulted: dict[str, dict] = {}
    for row in files:
        consulted[row["path"]] = {"path": row["path"], "identity": list(row["identity"]), "sha256": row["sha256"]}
    value = {"schema_version": SCHEMA_VERSION, "dependency_version": DEPENDENCY_VERSION, "name": name, "key": key, "source_commit": source_commit,
             "code": code, "files": list(consulted.values()), "verdict": verdict,
             "created_at_epoch": time.time()}
    try:
        value = json.loads(canonical_json(value))
        value["entry_digest"] = canonical_digest(value, digest_field="entry_digest")
        path = _entry_path(root or verdict_root(), name, key)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        _write_entry(path, value, mode=0o640)
    except (OSError, TypeError, ValueError):
        return None
    return path


def inherit_cached_dependencies(name, key):
    """An in-operation cache hit must still bind every enclosing verdict."""
    if _DEPENDENCIES.get() and lookup(name=name, key=key, missing=MISS) is MISS:
        for collector in _DEPENDENCIES.get():
            collector.add("")


def lookup_entry(*, name, key, root=None):
    """Compatibility API for controls byte verdicts: null is a successful value."""
    value = lookup(name=name, key=key, root=root, missing=MISS)
    return (False, None) if value is MISS else (True, value)
