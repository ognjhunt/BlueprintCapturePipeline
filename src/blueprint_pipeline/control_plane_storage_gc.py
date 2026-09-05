"""Conservative reclamation for control-plane caches, on a timer.

Three reclaim steps, each dry-run first and applied only with its typed
acknowledgement:

* **Derived directories** (``cache`` class: prepared references, compiled
  episodes, activation launch sets) are retired when no live storage pin names
  them, no pending or processing queue message mentions them, and they have
  been idle longer than the grace period.
* **Content-store blobs**: only direct children of an explicitly supplied
  ``sha256`` directory are ever eligible.  A blob is reclaimable when its name
  is its SHA-256 digest, it is an ordinary non-symlink file, its link count is
  exactly one, its bytes still match its name, and it is older than the grace
  period.  The link-count rule makes every derived-directory hardlink an
  implicit pin, so retiring directories first is what frees blobs.
* **Evidence offload** (``evidence_cold`` class) migrates sealed run
  directories to the artifact store behind a digest-bound pointer; it stays a
  dry run until the operator enables it.

Evidence-hot roots, release worktrees, and runtime trees are never candidates
here; release trees are retired by the deploy that supersedes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import os
import shutil
import tempfile

from .control_plane_evidence_offload import (
    DEFAULT_HOT_WINDOW_SECONDS,
    EXECUTE_ACK as OFFLOAD_ACK,
    apply_evidence_offload,
    build_evidence_offload_manifest,
)
from .control_plane_storage_pins import PINS_ROOT_ENV, live_pinned_paths
from .control_plane_storage_roots import require_storage_class
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_release_identity import running_release_commit


SCHEMA_VERSION = "control_plane_storage_gc.v1"
EXECUTE_ACK = "reap-unreferenced-content"
DERIVED_SCHEMA_VERSION = "control_plane_derived_directory_manifest.v1"
DERIVED_RECEIPT_SCHEMA_VERSION = "control_plane_derived_directory_receipt.v1"
DERIVED_ACK = "retire-terminal-derived-directories"
RUN_SCHEMA_VERSION = "control_plane_storage_gc_run.v1"
RUN_ACK = "reclaim-control-plane-storage"
DEFAULT_MINIMUM_AGE_SECONDS = 24 * 60 * 60
# Failed and superseded policy-canary builds can create 10+ GiB of fully
# reproducible prepared/compiled caches in a single attempt.  Six hours keeps
# a debugging window while ensuring the six-hourly timer reclaims terminal,
# unpinned, unqueued work before the next operating window.
DEFAULT_DERIVED_MINIMUM_AGE_SECONDS = 6 * 60 * 60
RESERVED_DERIVED_CHILDREN = frozenset({"content-addressed"})
QUEUE_STATES = ("pending", "processing")
CONTENT_STORE_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_CONTENT_STORE_ROOTS"
DERIVED_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_DERIVED_ROOTS"
QUEUE_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS"
EVIDENCE_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_EVIDENCE_ROOTS"
EVIDENCE_OFFLOAD_ENV = "BLUEPRINT_CONTROL_PLANE_EVIDENCE_OFFLOAD"
EVIDENCE_HOT_WINDOW_ENV = "BLUEPRINT_CONTROL_PLANE_EVIDENCE_HOT_WINDOW_SECONDS"
EVIDENCE_ABANDONED_AFTER_ENV = "BLUEPRINT_CONTROL_PLANE_EVIDENCE_ABANDONED_AFTER_SECONDS"
SCRATCH_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_SCRATCH_ROOTS"
SCRATCH_MINIMUM_AGE_ENV = "BLUEPRINT_CONTROL_PLANE_GC_SCRATCH_MINIMUM_AGE_SECONDS"
RUNNING_COMMIT_ENV = "BLUEPRINT_CONTROL_PLANE_GC_RUNNING_COMMIT"
# Stranded rows: a pending queue row bound to a release other than the running
# one.  Every worker honours only same-release rows, so such a row can never
# progress, yet as long as it sits in ``pending`` it is a live reference that
# keeps that release's trees and every derived directory it names on disk.
STRANDED_SCHEMA_VERSION = "control_plane_stranded_queue_manifest.v1"
STRANDED_RECEIPT_SCHEMA_VERSION = "control_plane_stranded_queue_receipt.v1"
STRANDED_ROW_RECEIPT_SCHEMA_VERSION = "control_plane_stranded_queue_row.v1"
STRANDED_ACK = "strand-superseded-release-rows"
STRANDED_STATE = "stranded"
STRANDED_RECEIPT_SUFFIX = ".stranded.v1.json"
# Scratch: diagnostics and engineering trees nothing references.  Reaped by
# idle age alone; three days keeps any investigation's window open.
SCRATCH_SCHEMA_VERSION = "control_plane_scratch_manifest.v1"
SCRATCH_RECEIPT_SCHEMA_VERSION = "control_plane_scratch_receipt.v1"
SCRATCH_ACK = "reap-idle-scratch"
DEFAULT_SCRATCH_MINIMUM_AGE_SECONDS = 72 * 60 * 60
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_ROW_COMMIT_KEYS = ("expected_production_commit", "source_commit")
_DIGEST_NAME = re.compile(r"[0-9a-f]{64}\Z")
_MAX_QUEUE_MESSAGE_BYTES = 16 * 1024 * 1024


class ControlPlaneStorageGCError(RuntimeError):
    """A storage root or requested mutation was unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_gc_manifest(
    *,
    content_store_roots: Sequence[str | Path],
    minimum_age_seconds: int = DEFAULT_MINIMUM_AGE_SECONDS,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    if (
        not content_store_roots
        or not isinstance(minimum_age_seconds, int)
        or isinstance(minimum_age_seconds, bool)
        or minimum_age_seconds < 0
    ):
        raise ControlPlaneStorageGCError("control_plane_storage_gc_input_invalid")
    roots: list[Path] = []
    candidates: list[dict[str, Any]] = []
    retained: dict[str, int] = {
        "linked": 0,
        "young": 0,
        "unsafe_or_unverified": 0,
    }
    observed_at = now()
    for raw_root in content_store_roots:
        raw = Path(raw_root).expanduser()
        if raw.is_symlink():
            raise ControlPlaneStorageGCError(
                "control_plane_storage_gc_root_unsafe"
            )
        root = raw.resolve(strict=True)
        if not root.is_dir() or root.name != "sha256" or root in roots:
            raise ControlPlaneStorageGCError(
                "control_plane_storage_gc_root_unsafe"
            )
        roots.append(root)
        for path in sorted(root.iterdir()):
            try:
                stat = path.lstat()
            except OSError:
                retained["unsafe_or_unverified"] += 1
                continue
            if (
                path.is_symlink()
                or not path.is_file()
                or _DIGEST_NAME.fullmatch(path.name) is None
            ):
                retained["unsafe_or_unverified"] += 1
                continue
            if stat.st_nlink != 1:
                retained["linked"] += 1
                continue
            age = max(0.0, observed_at - stat.st_mtime)
            if age < minimum_age_seconds:
                retained["young"] += 1
                continue
            if _sha256(path) != path.name:
                retained["unsafe_or_unverified"] += 1
                continue
            candidates.append(
                {
                    "root": str(root),
                    "digest": "sha256:" + path.name,
                    "size_bytes": stat.st_size,
                    "age_seconds": int(age),
                    "observed_link_count": stat.st_nlink,
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run",
        "minimum_age_seconds": minimum_age_seconds,
        "root_count": len(roots),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(row["size_bytes"] for row in candidates),
        "candidates": candidates,
        "retained_counts": retained,
        "evidence_roots_scanned": False,
        "release_or_worktree_roots_scanned": False,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    return manifest


def apply_gc_manifest(
    manifest: dict[str, Any], *, ack: str
) -> dict[str, Any]:
    if (
        ack != EXECUTE_ACK
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
    ):
        raise ControlPlaneStorageGCError(
            "control_plane_storage_gc_apply_not_authorized"
        )
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in manifest.get("candidates") or []:
        root = Path(str(row.get("root") or ""))
        digest = str(row.get("digest") or "").removeprefix("sha256:")
        path = root / digest
        try:
            stat = path.lstat()
            safe = (
                root.name == "sha256"
                and path.parent == root
                and _DIGEST_NAME.fullmatch(path.name) is not None
                and not path.is_symlink()
                and path.is_file()
                and stat.st_nlink == 1
                and stat.st_size == row.get("size_bytes")
                and _sha256(path) == digest
            )
            if not safe:
                raise OSError("candidate changed after dry run")
            path.unlink()
        except OSError:
            skipped.append(
                {"digest": "sha256:" + digest, "reason": "candidate_changed"}
            )
        else:
            removed.append(
                {"digest": "sha256:" + digest, "size_bytes": stat.st_size}
            )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "applied",
        "source_manifest_digest": manifest["manifest_digest"],
        "removed_count": len(removed),
        "removed_bytes": sum(row["size_bytes"] for row in removed),
        "removed": removed,
        "skipped": skipped,
        "evidence_removed": False,
        "release_or_worktree_removed": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(
        result, digest_field="result_digest"
    )
    return result


def _queue_reference_text(queue_roots: Sequence[str | Path]) -> str:
    """Concatenate every pending or processing queue message; a name in it is live."""

    chunks: list[str] = []
    for raw_root in queue_roots:
        root = Path(raw_root).expanduser()
        for state in QUEUE_STATES:
            directory = root / state
            if not directory.is_dir() or directory.is_symlink():
                continue
            for path in sorted(directory.glob("*.json")):
                try:
                    if path.is_symlink() or path.stat().st_size > _MAX_QUEUE_MESSAGE_BYTES:
                        continue
                    chunks.append(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError):
                    continue
    return "\n".join(chunks)


def _tree_snapshot(directory: Path) -> tuple[float, int]:
    latest = directory.lstat().st_mtime
    size = 0
    for root, directories, files in os.walk(directory):
        directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
        for name in files:
            try:
                metadata = (Path(root) / name).lstat()
            except OSError:
                continue
            latest = max(latest, metadata.st_mtime)
            size += metadata.st_size
    return latest, size


def _derived_children(root: Path) -> list[Path]:
    return [
        child
        for child in sorted(root.iterdir())
        if not child.name.startswith(".") and child.name not in RESERVED_DERIVED_CHILDREN
    ]


def build_derived_directory_manifest(
    *,
    derived_roots: Sequence[str | Path],
    pins_root: str | Path,
    queue_roots: Sequence[str | Path],
    minimum_age_seconds: int = DEFAULT_DERIVED_MINIMUM_AGE_SECONDS,
    now: Callable[[], float] = time.time,
    classifier: Callable[..., Any] = require_storage_class,
) -> dict[str, Any]:
    """List derived directories no pin, queue message, or recent write still needs."""

    if (
        not derived_roots
        or not isinstance(minimum_age_seconds, int)
        or isinstance(minimum_age_seconds, bool)
        or minimum_age_seconds < 0
    ):
        raise ControlPlaneStorageGCError("control_plane_storage_gc_input_invalid")
    observed_at = float(now())
    pinned = live_pinned_paths(pins_root, now=lambda: observed_at)
    queue_text = _queue_reference_text(queue_roots)
    candidates: list[dict[str, Any]] = []
    retained = {"pinned": 0, "queue_referenced": 0, "young": 0, "unsafe": 0}
    roots: list[str] = []
    for raw_root in derived_roots:
        root = Path(raw_root).expanduser()
        classifier(str(root), expected="cache", code="control_plane_storage_gc_derived_root_class")
        if root.is_symlink() or not root.is_dir():
            raise ControlPlaneStorageGCError("control_plane_storage_gc_root_unsafe")
        roots.append(str(root))
        for child in _derived_children(root):
            if child.is_symlink() or not child.is_dir():
                retained["unsafe"] += 1
                continue
            if str(child) in pinned or str(child.resolve()) in pinned:
                retained["pinned"] += 1
                continue
            if child.name in queue_text:
                retained["queue_referenced"] += 1
                continue
            latest, size = _tree_snapshot(child)
            if observed_at - latest < minimum_age_seconds:
                retained["young"] += 1
                continue
            candidates.append(
                {
                    "root": str(root),
                    "name": child.name,
                    "size_bytes": size,
                    "idle_seconds": int(observed_at - latest),
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "status": "dry_run",
        "minimum_age_seconds": minimum_age_seconds,
        "roots": roots,
        "candidate_count": len(candidates),
        "candidate_bytes": sum(row["size_bytes"] for row in candidates),
        "candidates": candidates,
        "retained_counts": retained,
        "evidence_roots_scanned": False,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    return manifest


def apply_derived_directory_manifest(
    manifest: dict[str, Any],
    *,
    ack: str,
    pins_root: str | Path,
    queue_roots: Sequence[str | Path],
    now: Callable[[], float] = time.time,
    classifier: Callable[..., Any] = require_storage_class,
) -> dict[str, Any]:
    """Retire exactly the listed directories after re-proving each one is unneeded."""

    if (
        ack != DERIVED_ACK
        or manifest.get("schema_version") != DERIVED_SCHEMA_VERSION
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
    ):
        raise ControlPlaneStorageGCError("control_plane_storage_gc_apply_not_authorized")
    observed_at = float(now())
    pinned = live_pinned_paths(pins_root, now=lambda: observed_at)
    queue_text = _queue_reference_text(queue_roots)
    minimum_age = int(manifest.get("minimum_age_seconds") or 0)
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in manifest.get("candidates") or []:
        root = Path(str(row.get("root") or ""))
        name = str(row.get("name") or "")
        child = root / name
        try:
            classifier(str(root), expected="cache", code="control_plane_storage_gc_derived_root_class")
            if (
                not name
                or "/" in name
                or name.startswith(".")
                or name in RESERVED_DERIVED_CHILDREN
                or child.is_symlink()
                or not child.is_dir()
                or str(child) in pinned
                or str(child.resolve()) in pinned
                or name in queue_text
            ):
                raise OSError("candidate changed after dry run")
            latest, size = _tree_snapshot(child)
            if observed_at - latest < minimum_age:
                raise OSError("candidate changed after dry run")
            shutil.rmtree(child)
        except (OSError, ValueError):
            skipped.append({"name": name, "reason": "candidate_changed"})
            continue
        removed.append({"name": name, "size_bytes": size})
    result: dict[str, Any] = {
        "schema_version": DERIVED_RECEIPT_SCHEMA_VERSION,
        "status": "applied",
        "source_manifest_digest": manifest["manifest_digest"],
        "removed_count": len(removed),
        "removed_bytes": sum(row["size_bytes"] for row in removed),
        "removed": removed,
        "skipped": skipped,
        "evidence_removed": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def _existing(paths: Sequence[str | Path]) -> tuple[list[Path], list[str]]:
    present: list[Path] = []
    absent: list[str] = []
    for raw in paths:
        path = Path(raw).expanduser()
        (present if path.is_dir() else absent).append(path if path.is_dir() else str(path))
    return present, absent


def _row_bound_commit(document: Mapping[str, Any]) -> str:
    """The release a queue row is bound to, or "" when the row does not say."""

    for key in _ROW_COMMIT_KEYS:
        value = document.get(key)
        if isinstance(value, str) and _COMMIT.fullmatch(value):
            return value
    release = document.get("release")
    if isinstance(release, Mapping):
        value = release.get("commit")
        if isinstance(value, str) and _COMMIT.fullmatch(value):
            return value
    return ""


def build_stranded_queue_manifest(
    *,
    queue_roots: Sequence[str | Path],
    running_commit: str,
    now: Callable[[], float] = time.time,
    classifier: Callable[..., Any] = require_storage_class,
) -> dict[str, Any]:
    """List pending rows bound to a release other than the running one; mutate nothing.

    Rows in ``processing`` belong to a worker and are never touched.  Rows that
    do not name a release are left alone: the worker that owns them decides.
    """

    if not isinstance(running_commit, str) or not _COMMIT.fullmatch(running_commit):
        raise ControlPlaneStorageGCError("control_plane_storage_gc_running_commit_invalid")
    observed_at = float(now())
    candidates: list[dict[str, Any]] = []
    retained = {"same_release": 0, "unbound": 0, "unsafe": 0}
    roots: list[str] = []
    for raw_root in queue_roots:
        root = Path(raw_root).expanduser()
        classifier(str(root), expected="work", code="control_plane_storage_gc_stranded_root_class")
        roots.append(str(root))
        pending = root / "pending"
        if pending.is_symlink() or not pending.is_dir():
            continue
        for path in sorted(pending.glob("*.json")):
            try:
                if path.is_symlink() or not path.is_file():
                    retained["unsafe"] += 1
                    continue
                metadata = path.stat()
                if metadata.st_size > _MAX_QUEUE_MESSAGE_BYTES:
                    retained["unsafe"] += 1
                    continue
                raw = path.read_bytes()
                document = json.loads(raw.decode("utf-8"))
            except (OSError, ValueError):
                retained["unsafe"] += 1
                continue
            if not isinstance(document, Mapping):
                retained["unsafe"] += 1
                continue
            bound = _row_bound_commit(document)
            if not bound:
                retained["unbound"] += 1
                continue
            if bound == running_commit:
                retained["same_release"] += 1
                continue
            candidates.append(
                {
                    "queue_root": str(root),
                    "name": path.name,
                    "bound_commit": bound,
                    "size_bytes": metadata.st_size,
                    "inode": metadata.st_ino,
                    "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": STRANDED_SCHEMA_VERSION,
        "status": "dry_run",
        "running_commit": running_commit,
        "observed_at_epoch": observed_at,
        "roots": roots,
        "candidate_count": len(candidates),
        "candidate_bytes": sum(row["size_bytes"] for row in candidates),
        "candidates": candidates,
        "retained_counts": retained,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    return manifest


def apply_stranded_queue_manifest(
    manifest: Mapping[str, Any], *, ack: str, now: Callable[[], float] = time.time
) -> dict[str, Any]:
    """Move every unchanged candidate to ``stranded/`` beside a digest-bound row receipt.

    Nothing is deleted.  Restoring a row is moving it back to ``pending`` under
    a release bound to its commit; the receipt records what moved and why.
    """

    if (
        ack != STRANDED_ACK
        or manifest.get("schema_version") != STRANDED_SCHEMA_VERSION
        or manifest.get("manifest_digest")
        != canonical_digest(dict(manifest), digest_field="manifest_digest")
    ):
        raise ControlPlaneStorageGCError("control_plane_storage_gc_stranded_apply_not_authorized")
    moved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in manifest.get("candidates") or []:
        root = Path(str(row.get("queue_root") or ""))
        name = str(row.get("name") or "")
        source = root / "pending" / name
        if not name or "/" in name or name.startswith(".") or source.is_symlink() or not source.is_file():
            skipped.append({"name": name, "reason": "candidate_changed"})
            continue
        try:
            metadata = source.stat()
            raw = source.read_bytes()
        except OSError:
            skipped.append({"name": name, "reason": "candidate_changed"})
            continue
        if (
            metadata.st_ino != row.get("inode")
            or metadata.st_size != row.get("size_bytes")
            or "sha256:" + hashlib.sha256(raw).hexdigest() != row.get("sha256")
        ):
            skipped.append({"name": name, "reason": "candidate_changed"})
            continue
        destination_root = root / STRANDED_STATE
        destination = destination_root / name
        receipt_path = destination_root / f"{name}{STRANDED_RECEIPT_SUFFIX}"
        try:
            destination_root.mkdir(mode=0o750, exist_ok=True)
            if destination.exists() or receipt_path.exists():
                skipped.append({"name": name, "reason": "destination_exists"})
                continue
            receipt: dict[str, Any] = {
                "schema_version": STRANDED_ROW_RECEIPT_SCHEMA_VERSION,
                "name": name,
                "queue_root": str(root),
                "bound_commit": row["bound_commit"],
                "running_commit": manifest["running_commit"],
                "previous_state": "pending",
                "sha256": row["sha256"],
                "size_bytes": metadata.st_size,
                "stranded_at_epoch": float(now()),
                "evidence_deleted": False,
                "receipt_digest": "",
            }
            receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
            with receipt_path.open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(source, destination)
        except OSError as exc:
            skipped.append({"name": name, "reason": f"strand_failed:{type(exc).__name__}"})
            continue
        moved.append(
            {
                "name": name,
                "queue_root": str(root),
                "bound_commit": row["bound_commit"],
                "size_bytes": metadata.st_size,
            }
        )
    result: dict[str, Any] = {
        "schema_version": STRANDED_RECEIPT_SCHEMA_VERSION,
        "status": "applied",
        "source_manifest_digest": manifest["manifest_digest"],
        "stranded_count": len(moved),
        "stranded_bytes": sum(row["size_bytes"] for row in moved),
        "stranded": moved,
        "skipped": skipped,
        "evidence_deleted": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def _make_writable_and_retry(function: Any, target: str, _error: Any) -> None:
    os.chmod(target, 0o700 if os.path.isdir(target) else 0o600)
    function(target)


def build_scratch_manifest(
    *,
    scratch_roots: Sequence[str | Path],
    minimum_age_seconds: int = DEFAULT_SCRATCH_MINIMUM_AGE_SECONDS,
    now: Callable[[], float] = time.time,
    classifier: Callable[..., Any] = require_storage_class,
) -> dict[str, Any]:
    """List idle children of scratch roots, without mutating anything.

    Scratch holds diagnostics and engineering trees that no queue, pin, or
    receipt references, so a child idle longer than the window is reaped by
    age alone.  Anything touched since is kept.
    """

    if (
        not isinstance(minimum_age_seconds, int)
        or isinstance(minimum_age_seconds, bool)
        or minimum_age_seconds < 0
    ):
        raise ControlPlaneStorageGCError("control_plane_storage_gc_scratch_window_invalid")
    observed_at = float(now())
    candidates: list[dict[str, Any]] = []
    retained = {"recent": 0, "unsafe": 0}
    roots: list[str] = []
    for raw_root in scratch_roots:
        root = Path(raw_root).expanduser()
        classifier(str(root), expected="scratch", code="control_plane_storage_gc_scratch_root_class")
        if root.is_symlink() or not root.is_dir():
            continue
        roots.append(str(root))
        for child in sorted(root.iterdir()):
            if child.name.startswith("."):
                continue
            if child.is_symlink():
                retained["unsafe"] += 1
                continue
            try:
                if child.is_dir():
                    latest, size = _tree_snapshot(child)
                    kind = "directory"
                else:
                    metadata = child.lstat()
                    latest, size, kind = metadata.st_mtime, metadata.st_size, "file"
            except OSError:
                retained["unsafe"] += 1
                continue
            idle_seconds = observed_at - latest
            if idle_seconds < minimum_age_seconds:
                retained["recent"] += 1
                continue
            candidates.append(
                {
                    "root": str(root),
                    "name": child.name,
                    "kind": kind,
                    "size_bytes": size,
                    "idle_seconds": int(idle_seconds),
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": SCRATCH_SCHEMA_VERSION,
        "status": "dry_run",
        "minimum_age_seconds": minimum_age_seconds,
        "observed_at_epoch": observed_at,
        "roots": roots,
        "candidate_count": len(candidates),
        "candidate_bytes": sum(row["size_bytes"] for row in candidates),
        "candidates": candidates,
        "retained_counts": retained,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    return manifest


def apply_scratch_manifest(
    manifest: Mapping[str, Any], *, ack: str, now: Callable[[], float] = time.time
) -> dict[str, Any]:
    """Remove every scratch candidate that is still idle; keep anything touched since."""

    if (
        ack != SCRATCH_ACK
        or manifest.get("schema_version") != SCRATCH_SCHEMA_VERSION
        or manifest.get("manifest_digest")
        != canonical_digest(dict(manifest), digest_field="manifest_digest")
    ):
        raise ControlPlaneStorageGCError("control_plane_storage_gc_scratch_apply_not_authorized")
    minimum_age = int(manifest.get("minimum_age_seconds") or 0)
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in manifest.get("candidates") or []:
        root = Path(str(row.get("root") or ""))
        name = str(row.get("name") or "")
        path = root / name
        kind = row.get("kind")
        if (
            not name
            or "/" in name
            or name.startswith(".")
            or path.is_symlink()
            or (kind == "directory") != path.is_dir()
            or (kind == "file") != path.is_file()
        ):
            skipped.append({"name": name, "reason": "candidate_changed"})
            continue
        try:
            latest = _tree_snapshot(path)[0] if kind == "directory" else path.lstat().st_mtime
            if float(now()) - latest < minimum_age:
                skipped.append({"name": name, "reason": "candidate_changed"})
                continue
            if kind == "directory":
                shutil.rmtree(path, onerror=_make_writable_and_retry)
            else:
                path.unlink()
            if os.path.lexists(path):
                raise OSError("scratch_remove_incomplete")
        except OSError as exc:
            skipped.append({"name": name, "reason": f"remove_failed:{type(exc).__name__}"})
            continue
        removed.append({"name": name, "root": str(root), "kind": kind, "size_bytes": row.get("size_bytes")})
    result: dict[str, Any] = {
        "schema_version": SCRATCH_RECEIPT_SCHEMA_VERSION,
        "status": "applied",
        "source_manifest_digest": manifest["manifest_digest"],
        "removed_count": len(removed),
        "removed_bytes": sum(int(row.get("size_bytes") or 0) for row in removed),
        "removed": removed,
        "skipped": skipped,
        "evidence_removed": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def run_storage_gc(
    *,
    content_store_roots: Sequence[str | Path],
    derived_roots: Sequence[str | Path],
    queue_roots: Sequence[str | Path],
    pins_root: str | Path,
    evidence_roots: Sequence[str | Path] = (),
    offload_enabled: bool = False,
    apply: bool = False,
    ack: str = "",
    content_minimum_age_seconds: int = DEFAULT_MINIMUM_AGE_SECONDS,
    derived_minimum_age_seconds: int = DEFAULT_DERIVED_MINIMUM_AGE_SECONDS,
    hot_window_seconds: int = DEFAULT_HOT_WINDOW_SECONDS,
    abandoned_after_seconds: int | None = None,
    running_commit: str = "",
    scratch_roots: Sequence[str | Path] = (),
    scratch_minimum_age_seconds: int = DEFAULT_SCRATCH_MINIMUM_AGE_SECONDS,
    now: Callable[[], float] = time.time,
    publisher: Callable[..., Any] | None = None,
    classifier: Callable[..., Any] = require_storage_class,
) -> dict[str, Any]:
    """One timer tick: stranded rows, derived directories, blobs, offload, scratch.

    Stranded rows go first so the derived-directory step in the same tick no
    longer sees them as live queue references.
    """

    if apply and ack != RUN_ACK:
        raise ControlPlaneStorageGCError("control_plane_storage_gc_apply_not_authorized")
    observed_at = float(now())
    clock = lambda: observed_at  # noqa: E731 - one observation per tick
    report: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "applied" if apply else "dry_run",
        "observed_at_epoch": observed_at,
        "apply": apply,
        "skipped_roots": [],
    }
    queue_present, _absent_queue_roots = _existing(queue_roots)
    if queue_present:
        if running_commit:
            stranded = build_stranded_queue_manifest(
                queue_roots=queue_present,
                running_commit=running_commit,
                now=clock,
                classifier=classifier,
            )
            report["stranded_queue_rows"] = (
                apply_stranded_queue_manifest(stranded, ack=STRANDED_ACK, now=clock)
                if apply
                else stranded
            )
        else:
            report["stranded_queue_rows"] = {
                "status": "skipped",
                "reason": "running_commit_unknown",
            }
    derived_present, absent = _existing(derived_roots)
    report["skipped_roots"].extend(absent)
    if derived_present:
        derived = build_derived_directory_manifest(
            derived_roots=derived_present,
            pins_root=pins_root,
            queue_roots=queue_roots,
            minimum_age_seconds=derived_minimum_age_seconds,
            now=clock,
            classifier=classifier,
        )
        report["derived_directories"] = (
            apply_derived_directory_manifest(
                derived,
                ack=DERIVED_ACK,
                pins_root=pins_root,
                queue_roots=queue_roots,
                now=clock,
                classifier=classifier,
            )
            if apply
            else derived
        )
    content_present, absent = _existing(content_store_roots)
    report["skipped_roots"].extend(absent)
    if content_present:
        blobs = build_gc_manifest(
            content_store_roots=content_present,
            minimum_age_seconds=content_minimum_age_seconds,
            now=clock,
        )
        report["content_store"] = apply_gc_manifest(blobs, ack=EXECUTE_ACK) if apply else blobs
    evidence_present, absent = _existing(evidence_roots)
    report["skipped_roots"].extend(absent)
    if evidence_present:
        offload = build_evidence_offload_manifest(
            evidence_roots=evidence_present,
            hot_window_seconds=hot_window_seconds,
            abandoned_after_seconds=abandoned_after_seconds,
            now=clock,
            classifier=classifier,
        )
        if apply and offload_enabled:
            extra = {"publisher": publisher} if publisher is not None else {}
            report["evidence_offload"] = apply_evidence_offload(
                offload, ack=OFFLOAD_ACK, now=clock, **extra
            )
        else:
            report["evidence_offload"] = offload
        report["evidence_offload_enabled"] = bool(offload_enabled)
    scratch_present, absent = _existing(scratch_roots)
    report["skipped_roots"].extend(absent)
    if scratch_present:
        scratch = build_scratch_manifest(
            scratch_roots=scratch_present,
            minimum_age_seconds=scratch_minimum_age_seconds,
            now=clock,
            classifier=classifier,
        )
        report["scratch_directories"] = (
            apply_scratch_manifest(scratch, ack=SCRATCH_ACK, now=clock) if apply else scratch
        )
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    return report


def _split_env(name: str) -> list[str]:
    return [item for item in str(os.getenv(name) or "").split(":") if item]


def _env_int(name: str, default: int | None) -> int | None:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ControlPlaneStorageGCError(
            f"control_plane_storage_gc_environment_int_invalid:{name}"
        ) from exc


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor, temporary = tempfile.mkstemp(prefix=".gc-report-", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _run_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="control_plane_storage_gc run")
    parser.add_argument("--content-store-root", action="append", default=None)
    parser.add_argument("--derived-root", action="append", default=None)
    parser.add_argument("--queue-root", action="append", default=None)
    parser.add_argument("--evidence-root", action="append", default=None)
    parser.add_argument("--pins-root", default=os.getenv(PINS_ROOT_ENV) or None)
    parser.add_argument("--scratch-root", action="append", default=None)
    parser.add_argument(
        "--scratch-minimum-age-seconds",
        type=int,
        default=_env_int(SCRATCH_MINIMUM_AGE_ENV, DEFAULT_SCRATCH_MINIMUM_AGE_SECONDS),
    )
    parser.add_argument(
        "--hot-window-seconds",
        type=int,
        default=_env_int(EVIDENCE_HOT_WINDOW_ENV, DEFAULT_HOT_WINDOW_SECONDS),
    )
    parser.add_argument(
        "--abandoned-after-seconds",
        type=int,
        default=_env_int(EVIDENCE_ABANDONED_AFTER_ENV, None),
    )
    parser.add_argument(
        "--running-commit",
        default=str(os.getenv(RUNNING_COMMIT_ENV) or "").strip() or running_release_commit(),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack", default="")
    parser.add_argument("--report-out", default=None)
    args = parser.parse_args(argv)
    pins_root = args.pins_root
    if not pins_root:
        raise ControlPlaneStorageGCError("control_plane_storage_gc_pins_root_missing")
    report = run_storage_gc(
        content_store_roots=args.content_store_root or _split_env(CONTENT_STORE_ROOTS_ENV),
        derived_roots=args.derived_root or _split_env(DERIVED_ROOTS_ENV),
        queue_roots=args.queue_root or _split_env(QUEUE_ROOTS_ENV),
        pins_root=pins_root,
        evidence_roots=args.evidence_root or _split_env(EVIDENCE_ROOTS_ENV),
        offload_enabled=str(os.getenv(EVIDENCE_OFFLOAD_ENV) or "").strip().lower()
        in {"1", "true", "yes"},
        apply=args.apply,
        ack=args.ack,
        hot_window_seconds=args.hot_window_seconds,
        abandoned_after_seconds=args.abandoned_after_seconds,
        running_commit=args.running_commit or "",
        scratch_roots=args.scratch_root or _split_env(SCRATCH_ROOTS_ENV),
        scratch_minimum_age_seconds=args.scratch_minimum_age_seconds,
        classifier=require_storage_class,
    )
    if args.report_out:
        _write_report(Path(args.report_out).expanduser(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "run":
        return _run_main(arguments[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-store-root", action="append", required=True)
    parser.add_argument(
        "--minimum-age-seconds",
        type=int,
        default=DEFAULT_MINIMUM_AGE_SECONDS,
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack", default="")
    args = parser.parse_args(argv)
    manifest = build_gc_manifest(
        content_store_roots=args.content_store_root,
        minimum_age_seconds=args.minimum_age_seconds,
    )
    result = (
        apply_gc_manifest(manifest, ack=args.ack) if args.apply else manifest
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ControlPlaneStorageGCError",
    "DERIVED_ACK",
    "EXECUTE_ACK",
    "RUN_ACK",
    "SCHEMA_VERSION",
    "apply_derived_directory_manifest",
    "apply_gc_manifest",
    "build_derived_directory_manifest",
    "build_gc_manifest",
    "main",
    "run_storage_gc",
]


if __name__ == "__main__":
    raise SystemExit(main())
