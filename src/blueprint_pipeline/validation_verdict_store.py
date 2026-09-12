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
  count) and re-hashed when that identity moved; smaller files always re-hashed.

Any unreadable, malformed, foreign or stale entry simply recomputes. Writing is best
effort: a read-only or missing root disables persistence and changes nothing else.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import time

from .decision_evidence_contracts import canonical_digest, canonical_json
from .validation_file_digests import MINIMUM_BYTES, _identity, sha256_file

SCHEMA_VERSION = "blueprint_validation_verdict.v1"
ROOT_ENV = "BLUEPRINT_VALIDATION_VERDICT_ROOT"
DEFAULT_ROOT = "/var/lib/blueprint/pipeline-control-plane/validation-verdicts"
PACKAGE = "blueprint_pipeline"
_MODULE_DIGESTS: dict[tuple, str] = {}


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


def code_identity() -> list[dict] | None:
    """Digest of every loaded ``blueprint_pipeline`` module's source, in module order."""
    import sys
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


def lookup(*, name: str, key, root: Path | None = None):
    """Return the stored verdict when the code and every consulted byte are unchanged, else ``None``."""
    path = _entry_path(root or verdict_root(), name, key)
    try:
        if path.is_symlink() or not path.is_file():
            return None
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION
            or value.get("name") != name
            or value.get("key") != json.loads(canonical_json({"key": key}))["key"]
            or value.get("entry_digest") != canonical_digest(value, digest_field="entry_digest")
            or not isinstance(value.get("files"), list) or not isinstance(value.get("code"), list)
            or not value["code"]):
        return None
    if not all(isinstance(row, dict) and isinstance(row.get("module"), str) for row in value["code"]):
        return None
    if not _code_unchanged(value["code"]):
        return None
    if not all(isinstance(row, dict) and _unchanged(row) for row in value["files"]):
        return None
    return value["verdict"]


def store(*, name: str, key, files, verdict, source_commit: str = "", root: Path | None = None) -> Path | None:
    """Persist one verdict with the exact files and code it consulted; ``None`` when unavailable."""
    code = code_identity()
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
