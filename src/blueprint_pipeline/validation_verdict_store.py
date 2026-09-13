"""Persisted validator verdicts, reusable only when the code and every consulted byte are unchanged.

The 2026-09-12 scene-840938 run measured the cost of re-proving retained GPU evidence
from bytes at every hand-off: one full prefix validation read about 40 GB against a
4.6 GB artifact tree and took 15 minutes, and the chain repeats it at every stage
boundary, every deploy and every reuse candidate. Renting and running the GPU was
the fast part.

A stored verdict is a cache, never evidence. It is reused only when:

* every ``blueprint_pipeline`` module in the verdict's dependency closure still has
  byte-identical source and the caller's key -- the content digests of its inputs --
  matches. The closure is the code that ran while the verdict was computed, the
  modules that code imports by statement (they supply its thresholds, schemas and
  constants without ever executing a function), the ``always`` modules the caller
  names, and the closure of every verdict reused or computed inside it. A deploy
  that leaves the closure untouched keeps the verdict, a deploy that changes any
  member recomputes, and a verdict whose closure cannot be established is never
  persisted;
* every file the validator consulted through the evidence readers (``sha256_file`` and
  ``read``) is re-proven: files at or above ``MINIMUM_BYTES`` by full stat identity
  (device, inode, size, nanosecond mtime and kernel-owned ctime, ownership, mode, link
  count) and re-hashed when that identity moved; smaller files always re-hashed.

Any unreadable, malformed, foreign or stale entry simply recomputes. Writing is best
effort: a read-only or missing root disables persistence and changes nothing else.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import threading
import time

from .decision_evidence_contracts import canonical_digest, canonical_json
from .validation_file_digests import MINIMUM_BYTES, _identity, sha256_file

SCHEMA_VERSION = "blueprint_validation_verdict.v1"
ROOT_ENV = "BLUEPRINT_VALIDATION_VERDICT_ROOT"
DEFAULT_ROOT = "/var/lib/blueprint/pipeline-control-plane/validation-verdicts"
PACKAGE = "blueprint_pipeline"
_MODULE_DIGESTS: dict[tuple, str] = {}
_MODULE_IMPORTS: dict[tuple, tuple] = {}
#: Open ``executed_code_identity`` collectors, innermost last. Every module that runs
#: and every verdict reused while a collector is open is a dependency of ALL open
#: collectors: an outer verdict depends on everything its nested validators depend on.
_COLLECTORS: list[dict] = []


def verdict_root() -> Path:
    return Path(os.getenv(ROOT_ENV) or DEFAULT_ROOT)


def _entry_path(root: Path, name: str, key) -> Path:
    token = hashlib.sha256(canonical_json({"name": name, "key": key}).encode("utf-8")).hexdigest()[:40]
    return root / f"{name}-{token}.json"


def _module_digest(path: Path) -> str | None:
    try:
        observed = path.stat()
        cache_key = (str(path), observed.st_mtime_ns, observed.st_size)
        if cache_key not in _MODULE_DIGESTS:
            _MODULE_DIGESTS[cache_key] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        return _MODULE_DIGESTS[cache_key]
    except OSError:
        return None


def _package_root() -> Path:
    import blueprint_pipeline
    return Path(blueprint_pipeline.__file__).resolve().parent


def _package_paths() -> list[Path]:
    """Every directory the package resolves modules from, the real package tree first."""
    import blueprint_pipeline
    root = _package_root()
    roots = [root]
    for entry in list(getattr(blueprint_pipeline, "__path__", None) or []):
        try:
            resolved = Path(entry).resolve()
        except OSError:
            continue
        if resolved not in roots:
            roots.append(resolved)
    return roots


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


def _name_for_file(filename: str) -> str | None:
    path = Path(filename)
    for root in _package_paths():
        name = _module_name(path, root)
        if name:
            return name
    return None


def _origin(name: str) -> Path | None:
    """The source file of package module ``name`` without importing anything; ``None`` when it is not a module."""
    module = sys.modules.get(name)
    origin = getattr(module, "__file__", None) if module is not None else None
    if origin and str(origin).endswith(".py"):
        return Path(origin)
    if name != PACKAGE and not name.startswith(PACKAGE + "."):
        return None
    relative = name[len(PACKAGE):].lstrip(".")
    for root in _package_paths():
        if not relative:
            candidates = [root / "__init__.py"]
        else:
            candidates = [root / (relative.replace(".", "/") + ".py"), root / relative.replace(".", "/") / "__init__.py"]
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue
    return None


def _module_imports(path: Path, name: str) -> tuple[str, ...]:
    """Package modules ``name`` imports by statement: the modules that supply its constants and schemas.

    Statement imports are the only way executed code reaches a threshold, a schema
    version or a table it never calls into; tracing calls cannot see those reads.
    Names that resolve to no module file (imported functions and classes) are dropped
    by ``_origin`` later. Nothing is imported here.
    """
    try:
        observed = path.stat()
        cache_key = (str(path), observed.st_mtime_ns, observed.st_size)
        if cache_key in _MODULE_IMPORTS:
            return _MODULE_IMPORTS[cache_key]
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return ()
    package_parts = name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PACKAGE or alias.name.startswith(PACKAGE + "."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: max(len(package_parts) - (node.level - 1), 0)]
                if not base:
                    continue
                module = ".".join([*base, node.module] if node.module else base)
            else:
                module = node.module or ""
            if module != PACKAGE and not module.startswith(PACKAGE + "."):
                continue
            found.add(module)
            for alias in node.names:  # ``from . import x`` names submodules
                found.add(module + "." + alias.name)
    _MODULE_IMPORTS[cache_key] = tuple(sorted(found))
    return _MODULE_IMPORTS[cache_key]


def _dependency_rows(collector: dict) -> list[dict] | None:
    names = {name for name in (_name_for_file(f) for f in collector["files"]) if name}
    names.update(collector["names"])
    names.update({__name__, "blueprint_pipeline.validation_file_digests"})
    closure = set(names)
    for name in sorted(names):
        origin = _origin(name)
        if origin is not None:
            closure.update(_module_imports(origin, name))
    rows: dict[str, str] = {}
    for name in sorted(closure):
        origin = _origin(name)
        if origin is None:
            if name in names:
                return None  # executed or named code that cannot be located: no closure, no reuse
            continue  # an imported function or class, not a module
        digest = _module_digest(origin)
        if digest is None:
            return None
        rows[name] = digest
    for row in collector["reused"]:
        if not isinstance(row, dict) or not isinstance(row.get("module"), str) or not isinstance(row.get("sha256"), str):
            return None
        if rows.get(row["module"], row["sha256"]) != row["sha256"]:
            return None  # two closures disagree about one module: nothing reusable can be established
        rows[row["module"]] = row["sha256"]
    return [{"module": name, "sha256": digest} for name, digest in sorted(rows.items())] or None


def executed_code_identity(run, *, always=()):
    """Run ``run`` under a call tracer; return ``(result, rows)`` for the verdict's dependency closure.

    Keying a verdict on every loaded module made every deploy discard every
    verdict: the 2026-09-13 scene-840938 run re-derived its adopted ten-stage
    chain for about eighteen minutes after each of eight deploys that never
    touched a validator. The code a verdict depends on is the code that ran
    while it was computed plus the modules that code imports by statement (the
    2026-09-13 audit showed a threshold read from an already-imported module
    never produces a call event); ``always`` names modules whose data the
    validator reads without importing them. Nested verdicts, computed or reused,
    add their closure to every enclosing collector, so an outer verdict can never
    outlive a change in a validator it delegated to.
    """
    collector = {"files": set(), "names": {str(name) for name in always if str(name).startswith(PACKAGE)},
                 "reused": []}
    _COLLECTORS.append(collector)

    def tracer(frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename
            for open_collector in _COLLECTORS:
                open_collector["files"].add(filename)
        return None

    previous, previous_threads = sys.gettrace(), threading.gettrace()
    threading.settrace(tracer)
    sys.settrace(tracer)
    try:
        result = run()
    finally:
        sys.settrace(previous)
        threading.settrace(previous_threads)
        if _COLLECTORS and _COLLECTORS[-1] is collector:
            _COLLECTORS.pop()
        else:  # never leave a foreign collector open
            _COLLECTORS[:] = [c for c in _COLLECTORS if c is not collector]
        for outer in _COLLECTORS:
            outer["files"].update(collector["files"])
            outer["names"].update(collector["names"])
            outer["reused"].extend(collector["reused"])
    return result, _dependency_rows(collector)


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


def _unchanged(row) -> bool:
    path = Path(row["path"])
    try:
        observed = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        return False
    if observed.st_size >= MINIMUM_BYTES and list(_identity(observed)) == list(row["identity"]):
        return True
    try:
        return sha256_file(path) == row["sha256"]
    except (OSError, ValueError):
        return False


def lookup_entry(*, name: str, key, root: Path | None = None) -> tuple[bool, object]:
    """``(True, verdict)`` when the closure and every consulted byte are unchanged, else ``(False, None)``.

    A successful validator may legitimately return ``None``; the found flag keeps
    that verdict reusable instead of recomputing it at every operation.
    """
    path = _entry_path(root or verdict_root(), name, key)
    try:
        if path.is_symlink() or not path.is_file():
            return False, None
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return False, None
    if (not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION
            or value.get("name") != name
            or value.get("key") != json.loads(canonical_json({"key": key}))["key"]
            or value.get("entry_digest") != canonical_digest(value, digest_field="entry_digest")
            or not isinstance(value.get("files"), list) or not isinstance(value.get("code"), list)
            or not value["code"]):
        return False, None
    if not all(isinstance(row, dict) and isinstance(row.get("module"), str) for row in value["code"]):
        return False, None
    if not _code_unchanged(value["code"]):
        return False, None
    if not all(isinstance(row, dict) and _unchanged(row) for row in value["files"]):
        return False, None
    for collector in _COLLECTORS:  # a reused verdict is a dependency of every verdict being computed around it
        collector["reused"].extend(value["code"])
    return True, value["verdict"]


def lookup(*, name: str, key, root: Path | None = None):
    """Return the stored verdict when reusable, else ``None``; use ``lookup_entry`` to tell a ``None`` verdict from a miss."""
    found, verdict = lookup_entry(name=name, key=key, root=root)
    return verdict if found else None


def store(*, name: str, key, files, verdict, source_commit: str = "", root: Path | None = None,
          code=None) -> Path | None:
    """Persist one verdict with the exact files and code it consulted; ``None`` when unavailable."""
    code = code if code else code_identity()
    if not code:
        return None
    consulted: dict[str, dict] = {}
    for row in files:
        consulted[row["path"]] = {"path": row["path"], "identity": list(row["identity"]), "sha256": row["sha256"]}
    value = {"schema_version": SCHEMA_VERSION, "name": name, "key": key, "source_commit": source_commit,
             "code": code, "files": list(consulted.values()), "verdict": verdict,
             "created_at_epoch": time.time()}
    try:
        value = json.loads(canonical_json(value))
        value["entry_digest"] = canonical_digest(value, digest_field="entry_digest")
        path = _entry_path(root or verdict_root(), name, key)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o640)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError):
        return None
    return path
