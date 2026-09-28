"""Conservative reclamation for control-plane caches, on a timer.

One tick (``run_storage_gc``) runs nine phases in this order. Every phase
plans before it mutates, and a tick applies nothing unless it runs with
``--apply`` and its typed acknowledgement. The plan-only phase never applies:

* **Stranded queue rows**: pending rows bound to a release other than the
  running one are moved to ``stranded/`` beside a receipt, so they stop
  counting as live queue references.  Nothing is deleted.
* **Terminal cache pins** whose run is proven closed by archived-run or sealed
  cold-run evidence are released. The extended proofs (a sealed registry run,
  an activation whose mutation window lapsed unlaunched, a stale preparation or
  compilation nothing consumes) only list their candidates until
  ``BLUEPRINT_CONTROL_PLANE_GC_EXTENDED_PIN_PROOFS=1``. This changes only the
  pin ledger.
* **Derived directories** (``cache`` class: prepared references, compiled
  episodes, activation launch sets) are retired when no live storage pin names
  them, no pending or processing queue message mentions them, and they have
  been idle longer than the grace period.
* **Planned derived directories** inventory configured roots such as SAM31
  preparation output. They are reported but never applied by this phase.
* **Content-store blobs**: only direct children of an explicitly supplied
  ``sha256`` directory are ever eligible.  A blob is reclaimable when its name
  is its SHA-256 digest, it is an ordinary non-symlink file, its link count is
  exactly one, its bytes still match its name, and it is older than the grace
  period.  The link-count rule makes every derived-directory hardlink an
  implicit pin, so retiring directories first is what frees blobs.
* **Evidence offload** (``evidence_cold`` class) migrates sealed run
  directories, and first their result artifacts, to the artifact store behind
  a digest-bound pointer; it stays a dry run until the operator enables it. A
  result run's residue follows once its bulk artifacts are remote
  (``task_evaluation_result_residue_offload``), only planned until
  ``BLUEPRINT_CONTROL_PLANE_GC_RESULT_RESIDUE_OFFLOAD=1`` as well.
* **Scratch directories** (``scratch`` class) idle longer than their window
  are reaped by age alone: nothing references them.
* **Workspace bundles**: the reproducible ``bundle/`` copy inside an idle,
  unpinned semantic-pretraining workspace is removed behind a sealed marker.
* **Scene workspaces** (``scene_workspace`` class) are retired by
  ``website_scene_workspace_retention`` once every file verifies in Firebase
  Storage or is archived to the artifact store behind a replayable receipt, and
  nothing can still need them. It only plans until its own explicit opt-in,
  ``BLUEPRINT_CONTROL_PLANE_SCENE_WORKSPACE_RETIREMENT=1``, enables it.
* **Replay caches** (``work`` class: activation lookaheads): the scratch inputs
  left in completed parent replays are removed by ``control_plane_replay_cache_gc``,
  which only plans until ``BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_RETENTION=1``.

Each phase runs isolated: an exception is recorded under its report key and the
remaining phases still run. Evidence-hot roots, release worktrees, and runtime
trees are never candidates here; release trees are retired by the deploy that
supersedes them.

The derived and evidence phases name why they kept each entry, with its bytes
(``retained_by_reason``), and ``--report-out`` also publishes a small summary
of the tick beside the report (``control_plane_storage_gc_reasons``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import os
import secrets
import shutil

from .control_plane_evidence_offload import (
    DEFAULT_HOT_WINDOW_SECONDS,
    EXECUTE_ACK as OFFLOAD_ACK,
    apply_evidence_offload,
    build_evidence_offload_manifest,
)
from .control_plane_replay_cache_gc import (
    REPLAY_PARENT_ROOTS_ENV, _truthy_setting, reclaim_replay_caches, replay_cache_retention_setting,
)
from .control_plane_storage_gc_reasons import (
    SUMMARY_FILENAME, WalkMeter, build_storage_gc_summary, count_retained, entry_bytes,
    evidence_protection_reason, live_pin_kinds, pin_protection, walked_bytes,
)
from .control_plane_storage_pins import PINS_ROOT_ENV, live_pinned_paths
# Kept under their old names for every existing caller.
from .control_plane_storage_references import (  # noqa: F401 - re-exported
    MAX_QUEUE_MESSAGE_BYTES as _MAX_QUEUE_MESSAGE_BYTES,
    QUEUE_STATES,
    SETTLEMENT_RECORD_GLOBS,
    queue_reference_text as _queue_reference_text,
    settlement_reference_text as _settlement_reference_text,
    settlement_reopens_beyond_retained_receipts,
)
from .control_plane_storage_roots import require_storage_class
from .control_plane_pin_proofs import activation_queue_root_of, launch_queue_root_of, preparation_queue_root_of
from .control_plane_terminal_cache_pins import extended_pin_proofs_setting, reconcile_terminal_cache_pins
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_release_identity import running_release_commit
from .task_evaluation_standing_launch_authorization import STANDING_AUTHORIZATION_DIR_ENV


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
CONTENT_STORE_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_CONTENT_STORE_ROOTS"
DERIVED_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_DERIVED_ROOTS"
PLAN_ONLY_DERIVED_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_PLAN_ONLY_DERIVED_ROOTS"
QUEUE_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS"
EVIDENCE_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_EVIDENCE_ROOTS"
SETTLEMENT_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_SETTLEMENT_ROOTS"
EVIDENCE_OFFLOAD_ENV = "BLUEPRINT_CONTROL_PLANE_EVIDENCE_OFFLOAD"
EVIDENCE_HOT_WINDOW_ENV = "BLUEPRINT_CONTROL_PLANE_EVIDENCE_HOT_WINDOW_SECONDS"
EVIDENCE_ABANDONED_AFTER_ENV = "BLUEPRINT_CONTROL_PLANE_EVIDENCE_ABANDONED_AFTER_SECONDS"
SCRATCH_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_SCRATCH_ROOTS"
DERIVED_MINIMUM_AGE_ENV = "BLUEPRINT_CONTROL_PLANE_GC_DERIVED_MINIMUM_AGE_SECONDS"
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
#: Workspace bundles: a semantic-pretraining workspace copies the whole
#: provider runtime into ``<workspace>/bundle`` (1.4 GB on 2026-09-13) and the
#: sealed run receipts already carry the bundle digest and its remote index.
#: The copy is reproducible from the release, so an idle workspace keeps its
#: outputs and receipts while the bundle is reaped; nothing else in the
#: workspace is touched.
WORKSPACE_BUNDLE_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_WORKSPACE_BUNDLE_ROOTS"
WORKSPACE_BUNDLE_MINIMUM_AGE_ENV = "BLUEPRINT_CONTROL_PLANE_GC_WORKSPACE_BUNDLE_MINIMUM_AGE_SECONDS"
WORKSPACE_BUNDLE_SCHEMA_VERSION = "control_plane_workspace_bundle_manifest.v1"
WORKSPACE_BUNDLE_RECEIPT_SCHEMA_VERSION = "control_plane_workspace_bundle_receipt.v1"
WORKSPACE_BUNDLE_MARKER_SCHEMA_VERSION = "control_plane_workspace_bundle_reaped.v1"
WORKSPACE_BUNDLE_ACK = "reap-idle-workspace-bundles"
WORKSPACE_BUNDLE_CHILD = "bundle"
WORKSPACE_BUNDLE_MARKER = "bundle_reaped.json"
DEFAULT_WORKSPACE_BUNDLE_MINIMUM_AGE_SECONDS = 6 * 60 * 60
#: Scene workspaces: pubsub handoff spools whose finished website scene working copies are
#: retired once cloud storage can restore them (website_scene_workspace_retention).
SCENE_WORKSPACE_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_SCENE_WORKSPACE_ROOTS"
SCENE_INTENT_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_GC_SCENE_INTENT_ROOT"
SCENE_BINDING_ROOT_ENV = "BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT"
SCENE_WORKSPACE_RETIREMENT_ENV = "BLUEPRINT_CONTROL_PLANE_SCENE_WORKSPACE_RETIREMENT"
SCENE_WORKSPACE_RETIREMENT_INVALID = "scene_workspace_retirement_setting_invalid"
REPORT_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_GC_REPORT_ROOT"
DEFAULT_MAX_SCENE_RETIREMENTS = 20
_MAX_SCENE_RESULTS = 50
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_ROW_COMMIT_KEYS = ("expected_production_commit", "source_commit")
_DIGEST_NAME = re.compile(r"[0-9a-f]{64}\Z")


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
        "candidate_count": manifest["candidate_count"],
        "candidate_bytes": manifest["candidate_bytes"],
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


def _tree_census(directory: Path) -> tuple[float, int, int]:
    """The newest mtime, the bytes and the number of files under ``directory``, never through a link."""

    latest = directory.lstat().st_mtime
    size = count = 0
    for root, directories, files in os.walk(directory):
        directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
        for name in files:
            try:
                metadata = (Path(root) / name).lstat()
            except OSError:
                continue
            latest = max(latest, metadata.st_mtime)
            size += metadata.st_size
            count += 1
    return latest, size, count


def _tree_snapshot(directory: Path) -> tuple[float, int]:
    latest, size, _files = _tree_census(directory)
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
    """List derived directories no pin, queue message, or recent write still needs.

    ``retained_by_reason`` gives the count and bytes behind each ``retained_counts``
    reason, and for ``pinned`` the kinds of the pins that hold them (``by_kind``);
    ``walked_file_count`` and ``walk_seconds`` say what walking the trees cost.
    """

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
    by_reason: dict[str, dict[str, Any]] = {}
    pin_kinds: dict[str, str] | None = None
    walk = WalkMeter(_tree_census)
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
                count_retained(by_reason, "unsafe", entry_bytes(child))
                continue
            if str(child) in pinned or str(child.resolve()) in pinned:
                retained["pinned"] += 1
                pin_kinds = live_pin_kinds(pins_root, now=lambda: observed_at) if pin_kinds is None else pin_kinds
                kind = pin_kinds.get(str(child)) or pin_kinds.get(str(child.resolve())) or "unknown"
                count_retained(by_reason, "pinned", walked_bytes(walk, child), kind=kind)
                continue
            if child.name in queue_text:
                retained["queue_referenced"] += 1
                count_retained(by_reason, "queue_referenced", walked_bytes(walk, child))
                continue
            latest, size, _files = walk(child)
            if observed_at - latest < minimum_age_seconds:
                retained["young"] += 1
                count_retained(by_reason, "young", size)
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
        "retained_by_reason": by_reason,
        **walk.fields(),
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
    from .control_plane_storage_pins import storage_pin_guard
    minimum_age = int(manifest.get("minimum_age_seconds") or 0)
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in manifest.get("candidates") or []:
        root = Path(str(row.get("root") or ""))
        name = str(row.get("name") or "")
        child = root / name
        try:
            with storage_pin_guard(pins_root, exclusive=True):
                pinned = live_pinned_paths(pins_root, now=lambda: observed_at)
                queue_text = _queue_reference_text(queue_roots)
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
        # What the verified manifest planned, why it kept the rest, and what its walk cost.
        "candidate_count": manifest.get("candidate_count"),
        "candidate_bytes": manifest.get("candidate_bytes"),
        "retained_by_reason": manifest.get("retained_by_reason"),
        "walked_file_count": manifest.get("walked_file_count"),
        "walk_seconds": manifest.get("walk_seconds"),
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
            # The dispatcher may still be delivering a completed older run.
            # It owns release admission and retirement of these queue entries;
            # moving one here can strand a live, resumable download readback.
            if document.get("schema_version") == "task_evaluation_policy_canary_dispatch_envelope.v1":
                retained["owner_managed_delivery"] = retained.get("owner_managed_delivery", 0) + 1
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
        "candidate_count": manifest["candidate_count"],
        "candidate_bytes": manifest["candidate_bytes"],
        "removed_count": len(removed),
        "removed_bytes": sum(int(row.get("size_bytes") or 0) for row in removed),
        "removed": removed,
        "skipped": skipped,
        "evidence_removed": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def workspace_process_active(workspace):
    from .completed_replay_cache_retention import active_reference
    try:
        return active_reference(workspace, ignored_process_ids=(os.getpid(),))
    except (OSError, ValueError):
        return True


def _workspace_busy(workspace, queue_roots):
    if not (workspace.parent / ".workspace-locks" / (workspace.name + ".lock")).is_file():
        return True  # No evidence that this legacy workspace uses the lock protocol.
    return workspace.name in _queue_reference_text(queue_roots) or workspace_process_active(workspace)


def _pinned_workspace(workspace: Path, pinned: set[str]) -> bool:
    """True when a live storage pin names the workspace, anything inside it, or an ancestor."""
    if not pinned:
        return False
    candidates = {workspace}
    try:
        candidates.add(workspace.resolve())
    except OSError:
        pass
    for candidate in candidates:
        for raw in pinned:
            pin = Path(raw)
            if pin == candidate or candidate in pin.parents or pin in candidate.parents:
                return True
    return False


def build_workspace_bundle_manifest(
    *,
    workspace_roots: Sequence[str | Path],
    minimum_age_seconds: int = DEFAULT_WORKSPACE_BUNDLE_MINIMUM_AGE_SECONDS,
    now: Callable[[], float] = time.time,
    classifier: Callable[..., Any] = require_storage_class,
    pins_root: str | Path | None = None,
    queue_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """List ``bundle`` copies inside idle workspaces, without mutating anything.

    A workspace is idle when nothing anywhere in its tree (outputs, receipts,
    or the bundle) changed within the window: the synchronous allocator run
    that uses the bundle finishes within hours, and a later run creates a new
    digest-named workspace rather than reusing this one. Reading a bundle does
    not move its modification time, so age alone cannot prove the absence of a
    live consumer: a workspace named by a live storage pin (2026-09-13 audit) is
    retained regardless of age, like every other reclaim class, and the pin is
    re-checked immediately before removal.
    """

    if (
        not isinstance(minimum_age_seconds, int)
        or isinstance(minimum_age_seconds, bool)
        or minimum_age_seconds < 0
    ):
        raise ControlPlaneStorageGCError("control_plane_storage_gc_workspace_bundle_window_invalid")
    from .control_plane_storage_pins import pins_root_from_environment
    pins_root = pins_root or pins_root_from_environment()
    observed_at = float(now())
    candidates: list[dict[str, Any]] = []
    retained = {"recent": 0, "unsafe": 0, "no_bundle": 0, "pinned": 0, "in_use": 0}
    pinned = live_pinned_paths(pins_root, now=lambda: observed_at) if pins_root is not None else set()
    roots: list[str] = []
    for raw_root in workspace_roots:
        root = Path(raw_root).expanduser()
        classifier(str(root), expected="work", code="control_plane_storage_gc_workspace_bundle_root_class")
        if root.is_symlink() or not root.is_dir():
            continue
        roots.append(str(root))
        for workspace in sorted(root.iterdir()):
            if workspace.name.startswith(".") or workspace.is_symlink() or not workspace.is_dir():
                continue
            if _pinned_workspace(workspace, pinned):
                retained["pinned"] += 1
                continue
            if _workspace_busy(workspace, queue_roots):
                retained["in_use"] += 1
                continue
            bundle = workspace / WORKSPACE_BUNDLE_CHILD
            if bundle.is_symlink():
                retained["unsafe"] += 1
                continue
            if not bundle.is_dir():
                retained["no_bundle"] += 1
                continue
            try:
                latest, _ = _tree_snapshot(workspace)
                _, size = _tree_snapshot(bundle)
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
                    "workspace": workspace.name,
                    "size_bytes": size,
                    "idle_seconds": int(idle_seconds),
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": WORKSPACE_BUNDLE_SCHEMA_VERSION,
        "status": "dry_run",
        "minimum_age_seconds": minimum_age_seconds,
        "observed_at_epoch": observed_at,
        "roots": roots,
        "candidate_count": len(candidates),
        "candidate_bytes": sum(row["size_bytes"] for row in candidates),
        "candidates": candidates,
        "retained_counts": retained,
        "pins_root": str(pins_root),
        "queue_roots": [str(p) for p in queue_roots],
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    return manifest


def apply_workspace_bundle_manifest(
    manifest: Mapping[str, Any], *, ack: str, now: Callable[[], float] = time.time
) -> dict[str, Any]:
    """Remove each candidate bundle whose workspace is still idle and unpinned; leave a sealed marker."""

    if (
        ack != WORKSPACE_BUNDLE_ACK
        or manifest.get("schema_version") != WORKSPACE_BUNDLE_SCHEMA_VERSION
        or manifest.get("manifest_digest")
        != canonical_digest(dict(manifest), digest_field="manifest_digest")
    ):
        raise ControlPlaneStorageGCError("control_plane_storage_gc_workspace_bundle_apply_not_authorized")
    from .control_plane_workspace_lock import workspace_lock
    minimum_age = int(manifest.get("minimum_age_seconds") or 0)
    pins_root = manifest.get("pins_root")
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in manifest.get("candidates") or []:
        root = Path(str(row.get("root") or ""))
        name = str(row.get("workspace") or "")
        workspace = root / name
        bundle = workspace / WORKSPACE_BUNDLE_CHILD
        if not name or "/" in name or name.startswith(".") or workspace.is_symlink() or bundle.is_symlink() or not bundle.is_dir():
            skipped.append({"workspace": name, "reason": "candidate_changed"})
            continue
        if pins_root and _pinned_workspace(workspace, live_pinned_paths(pins_root, now=now)):
            skipped.append({"workspace": name, "reason": "pinned"})
            continue
        try:
            with workspace_lock(workspace, reclaim=True) as acquired:
                if (not acquired or _workspace_busy(workspace, manifest.get("queue_roots", ()))
                        or (pins_root and _pinned_workspace(workspace, live_pinned_paths(pins_root, now=now)))):
                    skipped.append({"workspace": name, "reason": "workspace_in_use"})
                    continue
                if float(now()) - _tree_snapshot(workspace)[0] < minimum_age:
                    skipped.append({"workspace": name, "reason": "candidate_changed"})
                    continue
                shutil.rmtree(bundle, onerror=_make_writable_and_retry)
                if os.path.lexists(bundle):
                    raise OSError("workspace_bundle_remove_incomplete")
                marker = {
                    "schema_version": WORKSPACE_BUNDLE_MARKER_SCHEMA_VERSION,
                    "workspace": name,
                    "reaped_at_epoch": float(now()),
                    "reaped_bytes": int(row.get("size_bytes") or 0),
                    "source_manifest_digest": manifest["manifest_digest"],
                    "outputs_and_receipts_retained": True,
                }
                marker["marker_digest"] = canonical_digest(marker, digest_field="marker_digest")
                (workspace / WORKSPACE_BUNDLE_MARKER).write_text(
                    json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
                )
        except OSError as exc:
            skipped.append({"workspace": name, "reason": f"remove_failed:{type(exc).__name__}"})
            continue
        removed.append({"workspace": name, "root": str(root), "size_bytes": row.get("size_bytes")})
    result: dict[str, Any] = {
        "schema_version": WORKSPACE_BUNDLE_RECEIPT_SCHEMA_VERSION,
        "status": "applied",
        "source_manifest_digest": manifest["manifest_digest"],
        "candidate_count": manifest["candidate_count"],
        "candidate_bytes": manifest["candidate_bytes"],
        "removed_count": len(removed),
        "removed_bytes": sum(int(row.get("size_bytes") or 0) for row in removed),
        "removed": removed,
        "skipped": skipped,
        "evidence_removed": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def scene_workspace_retirement_setting(environ: Mapping[str, str] = os.environ) -> tuple[bool, str | None]:
    """Whether scene workspace retirement applies, and an alert when its setting is invalid.

    Retirement is its own owner decision: only
    ``BLUEPRINT_CONTROL_PLANE_SCENE_WORKSPACE_RETIREMENT=1`` (or ``true``/``yes``)
    enables it, and unset it only plans. It never follows the evidence offload
    opt-in. Any other value disables it and is reported as an alert; it never
    aborts the tick.
    """

    return _truthy_setting(environ, SCENE_WORKSPACE_RETIREMENT_ENV, SCENE_WORKSPACE_RETIREMENT_INVALID)


def retire_scene_workspaces(
    *,
    storage_roots: Sequence[str | Path],
    context_factory: Callable[[Path], Any],
    apply: bool,
    enabled: bool,
    now: float,
    cloud_factory: Callable[[], Any],
    stream_publisher: Callable[..., Any] | None = None,
    process_checker: Callable[[Path], bool] | None = None,
    max_retirements: int = DEFAULT_MAX_SCENE_RETIREMENTS,
    hash_budget_bytes: int | None = None,
    inventory_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Plan every scene workspace; attempt at most ``max_retirements`` retirements per tick when enabled.

    The bound counts attempts, not successes: each one can publish a large archive,
    so a failing publisher costs at most that many uploads per tick. The reference
    index (registrations and intents) is read once per root for the plans; each
    retirement re-reads it under the capture locks. Plans hash at most
    ``hash_budget_bytes`` uncached bytes per tick (twenty GiB by default) and cache
    file digests under ``inventory_cache_root``, dropping a scene's cache once it is
    retired or gone. One scene's failure is recorded on its row and never stops the others.
    """

    from .website_scene_workspace_retention import (
        DEFAULT_HASH_BUDGET_BYTES,
        RETIRE_ACK,
        HashBudget,
        apply_scene_workspace_retirement,
        build_reference_index,
        inventory_cache_path,
        plan_scene_workspace_retirement,
        scene_workspaces,
        sweep_retirement_temporaries,
        sweep_retiring_workspaces,
    )

    applying = bool(apply and enabled)
    observed_at = float(now)
    budget = HashBudget(remaining_bytes=DEFAULT_HASH_BUDGET_BYTES if hash_budget_bytes is None else hash_budget_bytes)
    live_scenes: set[tuple[str, str]] = set()
    report: dict[str, Any] = {
        "status": "applied" if applying else "dry_run",
        "enabled": bool(enabled),
        "candidate_count": 0,
        "candidate_bytes": 0,
        "attempted_count": 0,
        "retired_count": 0,
        "retired_bytes": 0,
        "archive_bytes": 0,
        "retained_counts": {},
        "retiring_removed_count" if applying else "retiring_removable_count": 0,
        "retiring_kept_without_receipt": [],
        "retiring_cleanup_incomplete": [],
        "receipt_temporaries_removed_count": 0,
        "published_archives": [],
    }
    rows: list[dict[str, Any]] = []
    cloud = None
    candidate_bytes_complete = True

    def retained(reasons: Sequence[str]) -> None:
        for prefix in sorted({reason.split(":", 1)[0] for reason in reasons}):
            report["retained_counts"][prefix] = report["retained_counts"].get(prefix, 0) + 1

    for storage_root in storage_roots:
        context = context_factory(Path(storage_root))
        # First finish any removal a crash interrupted: its retirement is already committed.
        swept = sweep_retiring_workspaces(context.storage_root, apply=applying)
        if applying:
            report["receipt_temporaries_removed_count"] += len(
                sweep_retirement_temporaries(context.storage_root, now=observed_at))
        report["retiring_removed_count" if applying else "retiring_removable_count"] += len(
            swept["removed" if applying else "removable"])
        report["retiring_kept_without_receipt"].extend(swept["kept_without_receipt"])
        report["retiring_cleanup_incomplete"].extend(swept.get("cleanup_incomplete", []))
        workspaces = scene_workspaces(context.storage_root)
        if not workspaces:
            continue
        index = build_reference_index(context, now=observed_at)
        cloud = cloud if cloud is not None else cloud_factory()
        for bucket, scene_id, _path in workspaces:
            row: dict[str, Any] = {"bucket": bucket, "scene_id": scene_id}
            candidate_measured = False
            live_scenes.add((bucket, scene_id))
            try:
                plan = plan_scene_workspace_retirement(
                    context=context, bucket=bucket, scene_id=scene_id, now=observed_at, cloud=cloud,
                    index=index, process_checker=process_checker, hash_budget=budget)
                if plan["status"] != "retirable":
                    retained(plan["reasons"])
                    rows.append({**row, "status": "retained", "reasons": plan["reasons"][:5]})
                    continue
                report["candidate_count"] += 1
                report["candidate_bytes"] += int(plan["totals"]["workspace_allocated_bytes"])
                candidate_measured = True
                if not applying or report["attempted_count"] >= max_retirements:
                    rows.append({**row, "status": "retirable",
                                 "workspace_allocated_bytes": plan["totals"]["workspace_allocated_bytes"],
                                 "archive_bytes": plan["totals"]["archive_bytes"]})
                    continue
                report["attempted_count"] += 1
                outcome = apply_scene_workspace_retirement(
                    plan, context=context, ack=RETIRE_ACK, cloud=cloud, now=observed_at,
                    stream_publisher=stream_publisher, process_checker=process_checker)
            except Exception as exc:  # noqa: BLE001 - one scene never costs the others
                if not candidate_measured:
                    candidate_bytes_complete = False
                rows.append({**row, "status": "error", "error": type(exc).__name__})
                continue
            if outcome["status"] == "retired":
                live_scenes.discard((bucket, scene_id))
                report["retired_count"] += 1
                report["retired_bytes"] += int(outcome["freed_allocated_bytes"])
                report["archive_bytes"] += int(outcome["archive_bytes"])
                rows.append({**row, "status": "retired", "receipt": outcome["receipt"],
                             "removal_complete": outcome["removal_complete"]})
            else:
                retained([outcome["reason"]])
                if "published_archive" in outcome:
                    report["published_archives"].append({**row, **outcome["published_archive"]})
                if outcome["reason"].startswith("candidate_changed") and context.inventory_cache_root is not None:
                    inventory_cache_path(context.inventory_cache_root, bucket, scene_id).unlink(missing_ok=True)
                rows.append({**row, "status": "skipped", "reason": outcome["reason"],
                             **({"published_archive": outcome["published_archive"]}
                                if "published_archive" in outcome else {})})
    # A cache is only worth keeping for a workspace that still exists. The
    # default cache follows each spool root onto the work volume.
    for storage_root in storage_roots:
        root = Path(inventory_cache_root) if inventory_cache_root is not None else Path(storage_root) / ".scene-workspace-inventory"
        for cache in sorted(root.glob("*/*.json")):
            if (cache.parent.name, cache.stem) not in live_scenes:
                cache.unlink(missing_ok=True)
    if not candidate_bytes_complete:
        report["candidate_bytes"] = None
    report["hashed_bytes"] = budget.hashed_bytes
    report["result_count"] = len(rows)
    report["error_count"] = sum(row["status"] == "error" for row in rows)
    report["omitted_result_count"] = max(0, len(rows) - _MAX_SCENE_RESULTS)
    report["results"] = rows[:_MAX_SCENE_RESULTS]
    return report


def _isolated(report: dict[str, Any], key: str, phase: Callable[[], Any]) -> None:
    """Run one reclaim phase so that its failure is recorded and later phases still run.

    A phase returns the value to store under ``key``, or None when it wrote its own keys.
    The report records only the exception type; the traceback goes to the journal.
    """

    try:
        result = phase()
    except Exception as exc:  # noqa: BLE001 - one phase never costs the tick
        report[key] = {"status": "error", "error": type(exc).__name__}
        report.setdefault("phase_errors", []).append(key)
        traceback.print_exc(file=sys.stderr)
        return
    if result is not None:
        report[key] = result


def run_storage_gc(
    *,
    content_store_roots: Sequence[str | Path],
    derived_roots: Sequence[str | Path],
    plan_only_derived_roots: Sequence[str | Path] = (),
    queue_roots: Sequence[str | Path],
    pins_root: str | Path,
    evidence_roots: Sequence[str | Path] = (),
    settlement_roots: Sequence[str | Path] = (),
    offload_enabled: bool = False,
    result_residue_offload_enabled: bool = False,
    result_residue_offload_alert: str | None = None,
    result_residue_max_runs_per_tick: int | None = None,
    apply: bool = False,
    ack: str = "",
    content_minimum_age_seconds: int = DEFAULT_MINIMUM_AGE_SECONDS,
    derived_minimum_age_seconds: int = DEFAULT_DERIVED_MINIMUM_AGE_SECONDS,
    hot_window_seconds: int = DEFAULT_HOT_WINDOW_SECONDS,
    abandoned_after_seconds: int | None = None,
    running_commit: str = "",
    scratch_roots: Sequence[str | Path] = (),
    scratch_minimum_age_seconds: int = DEFAULT_SCRATCH_MINIMUM_AGE_SECONDS,
    workspace_bundle_roots: Sequence[str | Path] = (),
    workspace_bundle_minimum_age_seconds: int = DEFAULT_WORKSPACE_BUNDLE_MINIMUM_AGE_SECONDS,
    scene_workspace_roots: Sequence[str | Path] = (),
    scene_intent_root: str | Path | None = None,
    scene_binding_root: str | Path | None = None,
    scene_workspace_retirement_enabled: bool = False,
    scene_workspace_retirement_alert: str | None = None,
    scene_cloud_factory: Callable[[], Any] | None = None,
    scene_stream_publisher: Callable[..., Any] | None = None,
    scene_process_checker: Callable[[Path], bool] | None = None,
    scene_inventory_cache_root: str | Path | None = None,
    scene_hash_budget_bytes: int | None = None,
    replay_parent_roots: Sequence[str | Path] = (),
    replay_cache_retention_enabled: bool = False,
    replay_cache_retention_alert: str | None = None,
    extended_pin_proofs_enabled: bool = False,
    extended_pin_proofs_alert: str | None = None,
    standing_authorization_dir: str | Path | None = None,
    now: Callable[[], float] = time.time,
    publisher: Callable[..., Any] | None = None,
    classifier: Callable[..., Any] = require_storage_class,
) -> dict[str, Any]:
    """One timer tick: stranded rows, derived directories, blobs, offload, scratch, replay caches, scenes.

    Stranded rows go first so the derived-directory step in the same tick no
    longer sees them as live queue references. Every phase is isolated.
    """

    if apply and ack != RUN_ACK:
        raise ControlPlaneStorageGCError("control_plane_storage_gc_apply_not_authorized")
    if {
        str(Path(root).expanduser()) for root in derived_roots
    } & {
        str(Path(root).expanduser()) for root in plan_only_derived_roots
    }:
        raise ControlPlaneStorageGCError("control_plane_storage_gc_plan_only_root_in_apply_roots")
    observed_at = float(now())
    clock = lambda: observed_at  # noqa: E731 - one observation per tick
    report: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "applied" if apply else "dry_run",
        "observed_at_epoch": observed_at,
        "apply": apply,
        "opt_in": {
            "evidence_offload": bool(offload_enabled),
            "result_residue_offload": bool(result_residue_offload_enabled),
            "scene_workspace_retirement": bool(scene_workspace_retirement_enabled),
            "replay_cache_retention": bool(replay_cache_retention_enabled),
            "extended_pin_proofs": bool(extended_pin_proofs_enabled),
        },
        "skipped_roots": [],
    }
    alerts = [alert for alert in (scene_workspace_retirement_alert, replay_cache_retention_alert,
                                  extended_pin_proofs_alert) if alert]
    if alerts:
        report["alerts"] = alerts
    if result_residue_offload_alert:
        report.setdefault("alerts", []).append(result_residue_offload_alert)
    queue_present, _absent_queue_roots = _existing(queue_roots)
    if queue_present:
        def stranded_phase() -> Any:
            if not running_commit:
                return {"status": "skipped", "reason": "running_commit_unknown"}
            stranded = build_stranded_queue_manifest(
                queue_roots=queue_present,
                running_commit=running_commit,
                now=clock,
                classifier=classifier,
            )
            return apply_stranded_queue_manifest(stranded, ack=STRANDED_ACK, now=clock) if apply else stranded

        _isolated(report, "stranded_queue_rows", stranded_phase)

    def terminal_cache_pins_phase() -> Any:
        return reconcile_terminal_cache_pins(
            pins_root=pins_root, queue_roots=queue_roots, evidence_roots=evidence_roots,
            now=observed_at, apply=apply, classifier=classifier, hot_window_seconds=hot_window_seconds,
            extended_proofs_enabled=extended_pin_proofs_enabled, running_commit=running_commit,
            activation_queue_root=activation_queue_root_of(queue_roots),
            preparation_queue_root=preparation_queue_root_of(queue_roots),
            launch_queue_root=launch_queue_root_of(queue_roots), standing_authorization_dir=standing_authorization_dir)

    _isolated(report, "terminal_cache_pins", terminal_cache_pins_phase)
    if extended_pin_proofs_alert and isinstance(report.get("terminal_cache_pins"), dict):
        report["terminal_cache_pins"]["alerts"] = [extended_pin_proofs_alert]
    derived_present, absent = _existing(derived_roots)
    report["skipped_roots"].extend(absent)
    if derived_present:
        def derived_phase() -> Any:
            derived = build_derived_directory_manifest(
                derived_roots=derived_present,
                pins_root=pins_root,
                queue_roots=queue_roots,
                minimum_age_seconds=derived_minimum_age_seconds,
                now=clock,
                classifier=classifier,
            )
            if not apply:
                return derived
            return apply_derived_directory_manifest(
                derived,
                ack=DERIVED_ACK,
                pins_root=pins_root,
                queue_roots=queue_roots,
                now=clock,
                classifier=classifier,
            )

        _isolated(report, "derived_directories", derived_phase)
    planned_present, absent = _existing(plan_only_derived_roots)
    report["skipped_roots"].extend(absent)
    if planned_present:
        _isolated(report, "planned_derived_directories", lambda: build_derived_directory_manifest(
            derived_roots=planned_present,
            pins_root=pins_root,
            queue_roots=queue_roots,
            minimum_age_seconds=derived_minimum_age_seconds,
            now=clock,
            classifier=classifier,
        ))
    content_present, absent = _existing(content_store_roots)
    report["skipped_roots"].extend(absent)
    if content_present:
        def content_phase() -> Any:
            blobs = build_gc_manifest(
                content_store_roots=content_present,
                minimum_age_seconds=content_minimum_age_seconds,
                now=clock,
            )
            return apply_gc_manifest(blobs, ack=EXECUTE_ACK) if apply else blobs

        _isolated(report, "content_store", content_phase)
    evidence_present, absent = _existing(evidence_roots)
    report["skipped_roots"].extend(absent)
    if evidence_present:
        def evidence_phase() -> None:
            observed_text, observed_unreadable = _settlement_reference_text(settlement_roots)
            report["evidence_settlement_reference"] = {
                "roots": [str(Path(root).expanduser()) for root in settlement_roots],
                "unreadable_count": observed_unreadable,
                "protect_all": bool(observed_unreadable),
            }
            del observed_text

            def protection_reason(directory: Path) -> str | None:
                # The one protection hook for the manifest, its apply and the per-artifact
                # offload: a reason keeps the run and names why. Re-reads settlements and
                # queues on every check; see evidence_protection_reason.
                return evidence_protection_reason(
                    directory, settlement_roots=settlement_roots, pins_root=pins_root,
                    queue_roots=queue_roots, now=clock, ignored_process_ids=(os.getpid(),))

            # Keep authenticated downloads usable after cold evidence reclamation.
            from .task_evaluation_result_artifact_store import (
                APPLY_ACK as RESULT_ARTIFACT_ACK, offload_failure, offload_result_artifacts,
            )
            from .task_evaluation_result_residue_offload import ResidueTick
            residue = ResidueTick(
                applying=apply and offload_enabled and result_residue_offload_enabled,
                enabled=result_residue_offload_enabled, max_runs=result_residue_max_runs_per_tick,
                hot_window_seconds=hot_window_seconds, protection_checker=protection_reason, publisher=publisher,
                now=clock, queue_roots=queue_roots)
            report["result_artifact_offload"] = []
            for evidence_root in evidence_present:
                classifier(str(evidence_root), expected="evidence_cold", code="result_artifact_offload_root_class")
                for registry_path in sorted(Path(evidence_root).glob("*/artifacts/result_delivery/artifact_registry.json")):
                    try:
                        result = offload_result_artifacts(
                            run_root=registry_path.parents[2],
                            apply=apply and offload_enabled,
                            ack=RESULT_ARTIFACT_ACK if apply and offload_enabled else "",
                            hot_window_seconds=hot_window_seconds,
                            protection_checker=protection_reason, now=clock,
                            publisher=publisher,
                        )
                    except Exception as exc:
                        # Type, errno and stage only: a message can carry a host path.
                        result = {"status": "retained", "run_directory": registry_path.parents[2].name,
                                  "reason": type(exc).__name__, **offload_failure(exc)}
                    report["result_artifact_offload"].append(result)
                    residue.add(registry_path.parents[2], result)
            report["result_residue_offload"] = residue.phase(alert=result_residue_offload_alert)
            offload = build_evidence_offload_manifest(
                evidence_roots=evidence_present,
                hot_window_seconds=hot_window_seconds,
                abandoned_after_seconds=abandoned_after_seconds,
                now=clock,
                classifier=classifier,
                protection_checker=protection_reason,
                protection_detail=lambda directory: pin_protection(directory, pins_root=pins_root, now=clock),
            )
            if apply and offload_enabled:
                extra = {"publisher": publisher} if publisher is not None else {}
                report["evidence_offload"] = apply_evidence_offload(
                    offload, ack=OFFLOAD_ACK, now=clock, protection_checker=protection_reason, **extra
                )
            else:
                report["evidence_offload"] = offload
            report["evidence_offload_enabled"] = bool(offload_enabled)

        _isolated(report, "evidence_offload", evidence_phase)
    scratch_present, absent = _existing(scratch_roots)
    report["skipped_roots"].extend(absent)
    if scratch_present:
        def scratch_phase() -> Any:
            scratch = build_scratch_manifest(
                scratch_roots=scratch_present,
                minimum_age_seconds=scratch_minimum_age_seconds,
                now=clock,
                classifier=classifier,
            )
            return apply_scratch_manifest(scratch, ack=SCRATCH_ACK, now=clock) if apply else scratch

        _isolated(report, "scratch_directories", scratch_phase)
    bundle_present, absent = _existing(workspace_bundle_roots)
    report["skipped_roots"].extend(absent)
    if bundle_present:
        def bundle_phase() -> Any:
            bundles = build_workspace_bundle_manifest(
                workspace_roots=bundle_present, queue_roots=queue_roots,
                minimum_age_seconds=workspace_bundle_minimum_age_seconds,
                now=clock,
                classifier=classifier,
                pins_root=pins_root,
            )
            return apply_workspace_bundle_manifest(bundles, ack=WORKSPACE_BUNDLE_ACK, now=clock) if apply else bundles

        _isolated(report, "workspace_bundles", bundle_phase)
    replay_present, absent = _existing(replay_parent_roots)
    report["skipped_roots"].extend(absent)
    if replay_present:
        _isolated(report, "replay_caches", lambda: reclaim_replay_caches(
            parent_roots=replay_present, apply=apply, enabled=replay_cache_retention_enabled,
            now=clock, classifier=classifier))
        if replay_cache_retention_alert and isinstance(report.get("replay_caches"), dict):
            report["replay_caches"]["alerts"] = [replay_cache_retention_alert]
    scene_present, absent = _existing(scene_workspace_roots)
    report["skipped_roots"].extend(absent)
    if scene_present:
        def scene_phase() -> Any:
            from .website_scene_workspace_retention import GcsCloudInventory, RetentionContext, scene_queue_roots

            for root in scene_present:
                classifier(str(root), expected="work", code="control_plane_storage_gc_scene_workspace_root_class")
            intent_root = Path(scene_intent_root) if scene_intent_root else None
            if scene_binding_root:
                binding_root: Path | None = Path(scene_binding_root)
            else:
                binding_root = None if intent_root is None else intent_root.parent / "website-source-bindings"
            scene_queues = scene_queue_roots(queue_roots, intent_root)
            cache_root = Path(scene_inventory_cache_root) if scene_inventory_cache_root else None

            def context_factory(storage_root: Path) -> Any:
                # Without an intent root the reference index is unreadable, so nothing retires.
                return RetentionContext(storage_root=storage_root, pins_root=Path(pins_root),
                                        queue_roots=scene_queues, intent_root=intent_root,
                                        binding_root=binding_root,
                                        inventory_cache_root=cache_root or storage_root / ".scene-workspace-inventory")

            return retire_scene_workspaces(
                storage_roots=scene_present, context_factory=context_factory, apply=apply,
                enabled=scene_workspace_retirement_enabled, now=observed_at,
                cloud_factory=scene_cloud_factory or GcsCloudInventory,
                stream_publisher=scene_stream_publisher, process_checker=scene_process_checker,
                hash_budget_bytes=scene_hash_budget_bytes, inventory_cache_root=cache_root,
            )

        _isolated(report, "scene_workspaces", scene_phase)
        if scene_workspace_retirement_alert and isinstance(report.get("scene_workspaces"), dict):
            report["scene_workspaces"]["alerts"] = [scene_workspace_retirement_alert]
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
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    # The door runs without root privileges. Repair directories created by older
    # ticks under the service's 0077 umask before publishing the secret-free report.
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            os.fchmod(parent_fd, 0o755)
        except PermissionError:
            # Earlier GC units ran as blueprint. The current root unit has
            # CAP_CHOWN but not CAP_FOWNER, so take ownership before chmod.
            os.fchown(parent_fd, os.geteuid(), -1)
            os.fchmod(parent_fd, 0o755)
        temporary = f".gc-report-{secrets.token_hex(12)}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fchmod(stream.fileno(), 0o644)
            os.replace(temporary, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        published_parent = os.stat(path.parent, follow_symlinks=False)
        bound_parent = os.fstat(parent_fd)
        if (published_parent.st_dev, published_parent.st_ino) != (bound_parent.st_dev, bound_parent.st_ino):
            raise ControlPlaneStorageGCError("storage_gc_report_directory_retargeted")
    finally:
        os.close(parent_fd)


def _write_summary(report_path: Path, report: Mapping[str, Any]) -> bool:
    """Publish ``summary.json`` beside the report exactly as the report was published.

    It only projects a report already written, so a failure is traced to stderr
    and fails the unit without costing the report, and the previous tick's
    summary is withdrawn: a stale summary beside a newer report would be
    fabricated state. A report itself named ``summary.json`` is never
    overwritten by its summary.
    """

    if report_path.name == SUMMARY_FILENAME:
        return True
    try:
        _write_report(report_path.with_name(SUMMARY_FILENAME), build_storage_gc_summary(report))
    except Exception:  # noqa: BLE001 - the full report is already written
        traceback.print_exc(file=sys.stderr)
        _withdraw_summary(report_path.parent)
        return False
    return True


def _withdraw_summary(directory: Path) -> None:
    """Best effort: unlink ``summary.json`` by name through the report directory's descriptor."""

    try:
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return
    except OSError:
        traceback.print_exc(file=sys.stderr)
        return
    try:
        os.unlink(SUMMARY_FILENAME, dir_fd=directory_fd)
    except FileNotFoundError:
        pass
    except OSError:
        traceback.print_exc(file=sys.stderr)
    finally:
        os.close(directory_fd)


def _run_main(argv: list[str]) -> int:
    from .task_evaluation_result_residue_offload import (
        DEFAULT_MAX_RUNS_PER_TICK, RESIDUE_MAX_RUNS_ENV, result_residue_offload_setting,
    )
    parser = argparse.ArgumentParser(prog="control_plane_storage_gc run")
    parser.add_argument("--content-store-root", action="append", default=None)
    parser.add_argument("--derived-root", action="append", default=None)
    parser.add_argument("--plan-only-derived-root", action="append", default=None)
    parser.add_argument("--queue-root", action="append", default=None)
    parser.add_argument("--evidence-root", action="append", default=None)
    parser.add_argument("--settlement-root", action="append", default=None)
    parser.add_argument(
        "--derived-minimum-age-seconds",
        type=int,
        default=_env_int(DERIVED_MINIMUM_AGE_ENV, DEFAULT_DERIVED_MINIMUM_AGE_SECONDS),
    )
    parser.add_argument("--pins-root", default=os.getenv(PINS_ROOT_ENV) or None)
    parser.add_argument("--scene-workspace-root", action="append", default=None)
    parser.add_argument("--scene-intent-root", default=os.getenv(SCENE_INTENT_ROOT_ENV) or None)
    parser.add_argument("--scratch-root", action="append", default=None)
    parser.add_argument("--replay-parent-root", action="append", default=None)
    parser.add_argument("--workspace-bundle-root", action="append", default=None)
    parser.add_argument(
        "--workspace-bundle-minimum-age-seconds",
        type=int,
        default=_env_int(WORKSPACE_BUNDLE_MINIMUM_AGE_ENV, DEFAULT_WORKSPACE_BUNDLE_MINIMUM_AGE_SECONDS),
    )
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
        "--result-residue-max-runs-per-tick",
        type=int,
        default=_env_int(RESIDUE_MAX_RUNS_ENV, DEFAULT_MAX_RUNS_PER_TICK),
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
    residue_enabled, residue_alert = result_residue_offload_setting()
    if residue_alert:
        print(f"storage_gc_alert:{residue_alert}", file=sys.stderr)
    retirement_enabled, retirement_alert = scene_workspace_retirement_setting()
    replay_enabled, replay_alert = replay_cache_retention_setting()
    extended_enabled, extended_alert = extended_pin_proofs_setting()
    for alert in (retirement_alert, replay_alert, extended_alert):
        if alert:
            print(f"storage_gc_alert:{alert}", file=sys.stderr)
    report_root = str(os.getenv(REPORT_ROOT_ENV) or "").strip()
    if not report_root and args.report_out:
        report_root = str(Path(args.report_out).expanduser().parent)
    report = run_storage_gc(
        content_store_roots=args.content_store_root or _split_env(CONTENT_STORE_ROOTS_ENV),
        derived_roots=args.derived_root or _split_env(DERIVED_ROOTS_ENV),
        plan_only_derived_roots=args.plan_only_derived_root or _split_env(PLAN_ONLY_DERIVED_ROOTS_ENV),
        queue_roots=args.queue_root or _split_env(QUEUE_ROOTS_ENV),
        pins_root=pins_root,
        evidence_roots=args.evidence_root or _split_env(EVIDENCE_ROOTS_ENV),
        settlement_roots=args.settlement_root or _split_env(SETTLEMENT_ROOTS_ENV),
        offload_enabled=str(os.getenv(EVIDENCE_OFFLOAD_ENV) or "").strip().lower()
        in {"1", "true", "yes"},
        result_residue_offload_enabled=residue_enabled,
        result_residue_offload_alert=residue_alert,
        result_residue_max_runs_per_tick=args.result_residue_max_runs_per_tick,
        apply=args.apply,
        ack=args.ack,
        hot_window_seconds=args.hot_window_seconds,
        abandoned_after_seconds=args.abandoned_after_seconds,
        running_commit=args.running_commit or "",
        scratch_roots=args.scratch_root or _split_env(SCRATCH_ROOTS_ENV),
        workspace_bundle_roots=args.workspace_bundle_root or _split_env(WORKSPACE_BUNDLE_ROOTS_ENV),
        workspace_bundle_minimum_age_seconds=args.workspace_bundle_minimum_age_seconds,
        scratch_minimum_age_seconds=args.scratch_minimum_age_seconds,
        # Without this the run always used DEFAULT_DERIVED_MINIMUM_AGE_SECONDS (6h) and
        # both the unit environment and --derived-minimum-age-seconds were silently
        # ignored, unlike every other class. Scene 840938, 2026-09-15: each attempt leaves
        # a ~1.3 GiB activation set and a new attempt lands roughly every 45 minutes, so
        # about eight pile up before the first becomes eligible. Disk starved and blocked
        # the run three separate times.
        derived_minimum_age_seconds=args.derived_minimum_age_seconds,
        scene_workspace_roots=args.scene_workspace_root or _split_env(SCENE_WORKSPACE_ROOTS_ENV),
        scene_intent_root=args.scene_intent_root,
        scene_binding_root=str(os.getenv(SCENE_BINDING_ROOT_ENV) or "").strip() or None,
        scene_workspace_retirement_enabled=retirement_enabled,
        scene_workspace_retirement_alert=retirement_alert,
        replay_parent_roots=args.replay_parent_root or _split_env(REPLAY_PARENT_ROOTS_ENV),
        replay_cache_retention_enabled=replay_enabled,
        replay_cache_retention_alert=replay_alert,
        extended_pin_proofs_enabled=extended_enabled,
        extended_pin_proofs_alert=extended_alert,
        # Where launch admission records consumed standing authorizations; unset, no activation is unlaunched.
        standing_authorization_dir=str(os.getenv(STANDING_AUTHORIZATION_DIR_ENV) or "").strip() or None,
        # Per-file digests, so an hourly plan re-reads only what changed.
        scene_inventory_cache_root=None,
        classifier=require_storage_class,
    )
    summary_written = True
    if args.report_out:
        _write_report(Path(args.report_out).expanduser(), report)
        summary_written = _write_summary(Path(args.report_out).expanduser(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    # The whole report is written first; a failed phase still fails the unit so it is seen.
    return 1 if report.get("phase_errors") or not summary_written else 0


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
    "SCENE_INTENT_ROOT_ENV",
    "SCENE_WORKSPACE_RETIREMENT_ENV",
    "SCENE_WORKSPACE_ROOTS_ENV",
    "SCHEMA_VERSION",
    "apply_derived_directory_manifest",
    "apply_gc_manifest",
    "build_derived_directory_manifest",
    "build_gc_manifest",
    "main",
    "retire_scene_workspaces",
    "run_storage_gc",
    "scene_workspace_retirement_setting",
]


if __name__ == "__main__":
    raise SystemExit(main())
