"""Retire finished website scene working copies once cloud storage can restore them.

ADP-009D, day 28. On 2026-09-26 the control-plane host ran out of disk partly
because a scene's working copy under
``/var/lib/blueprint/pubsub-handoffs/<bucket>/scenes/<scene_id>`` was never
reclaimed. Each holds gigabytes of staged raw capture plus pipeline outputs, and
the website ``pipeline/**`` outputs exist nowhere else: their ``gs://`` names
are local aliases and nothing uploads them.

A workspace is retirable only when every check passes, cheapest first:

1. the scene path and its parents up to the storage root are real directories,
   and nothing inside is a symlink or special file;
2. every capture is a website capture (the listener's own lane test on its
   ``raw/manifest.json``), is terminal (a committed output, or an authority ending
   with its receipt) and holds no live lease;
3. every terminal message was acknowledged (an ack receipt written after the
   terminal state, or a ledger idle past Pub/Sub's message retention);
4. nothing in the tree changed for ``minimum_idle_seconds``;
5. no live storage pin names it, lies inside it or contains it;
6. no pending or processing queue message names the scene;
7. no live process holds it;
8. no scene intent that can still run resolves a website source registered inside it
   (an expired intent stays open for a grace period, since its owner may extend it,
   and a revoked or expired intent is held by any attempt it has not settled until
   that same grace period after it finished: every hold expires);
9. every file is recoverable: it verifies against its Firebase Storage object
   (size, and MD5 or, without MD5, CRC32C) or it is archived. Raw capture bytes
   are never archived, so a raw file that does not verify keeps the workspace.

Retirement (``apply``) re-proves checks 1-8 under the listener's own ledger
locks, streams the local-only files to the private artifact store with a full
readback, re-verifies the cloud objects, writes a digest-bound receipt beside
the workspace and only then removes it. ``restore`` replays the receipt byte for
byte. Nothing is deleted that cannot be restored.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .completed_replay_cache_retention import active_reference
from .control_plane_disk_budget import DEFAULT_RESERVATION_ROOT, reserve_control_plane_disk
from .control_plane_evidence_offload import ControlPlaneEvidenceOffloadError, _HashingSink, _pack_stream
from .control_plane_storage_gc import _pinned_workspace
from .control_plane_storage_pins import DEFAULT_PINS_ROOT, PINS_ROOT_ENV, live_pinned_paths, storage_pin_guard
from .core.security_controls import SecurityValidationError, strict_gcs_bucket, strict_identifier
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_configured_scene_object_store import (
    materialize_configured_scene_artifact,
    publish_configured_scene_stream,
)
from .website_capture_entry import is_website_capture_manifest
from .website_scene_workspace_digests import (
    Digests as _Digests,
    HashBudget,
    HashDeadlineExceeded,
    cached_digests as _cached_digests,
    hash_file as _hash_file,
    save_inventory_cache,
)
from .website_scene_workspace_queue import QueueInventoryUnavailable, queue_reference_text


PLAN_SCHEMA = "website_scene_workspace_retirement_plan.v1"
RETIRED_SCHEMA = "website_scene_workspace_retired.v1"
RESTORE_SCHEMA = "website_scene_workspace_restore_receipt.v1"
RETIRE_ACK = "retire-scene-workspace"
ARTIFACT_KIND = "website-scene-workspace"
ARCHIVE_FILENAME = "workspace.tar"
RETIRED_SUFFIX = ".retired.v1.json"
#: A retired workspace is renamed to ``.retiring-<scene_id>-<16 hex>`` beside it before removal.
RETIRING_PREFIX = ".retiring-"
_RETIRING_NAME = re.compile(r"\.retiring-(?P<scene>[A-Za-z0-9][A-Za-z0-9._-]{0,127})-(?P<token>[0-9a-f]{16})")
_RECEIPT_TEMP = re.compile(r"\.[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.retired\.v1\.json\.[0-9]+\.[0-9a-f]{8}\.tmp")
DEFAULT_MINIMUM_IDLE_SECONDS = 48 * 3600
#: Pub/Sub message retention (deploy/terraform/main.tf): after it, a message can no longer be redelivered.
DEFAULT_ACK_RETENTION_SECONDS = 7 * 24 * 3600
#: A website sponsorship lasts at most 24 hours, so an intent claims its registration well within this.
DEFAULT_ORPHAN_REGISTRATION_SECONDS = 72 * 3600
#: An owner may extend an expired intent's execution window, and scene progression would then resolve
#: its website source again, so an expired intent stays open this long after its effective expiry.
DEFAULT_EXPIRED_GRACE_SECONDS = 7 * 24 * 3600
#: Every hourly plan would otherwise re-read every candidate scene (tens of GB): digests are cached
#: per file identity, and a tick hashes at most this many uncached bytes.
DEFAULT_HASH_BUDGET_BYTES = 20 * 1024**3
INVENTORY_CACHE_SCHEMA = "website_scene_workspace_inventory_cache.v1"

# Production roots. The operator door runs the command line with only the control-plane
# environment file loaded, so every default must already name the production tree.
_CONTROL_PLANE = "/var/lib/blueprint/pipeline-control-plane"
DEFAULT_STORAGE_ROOT = Path("/var/lib/blueprint/pubsub-handoffs")
DEFAULT_INTENT_ROOT = Path(f"{_CONTROL_PLANE}/task-evaluation-scene-intents")
#: The reclaim timer's queue roots (BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS); a test pins them to the unit.
DEFAULT_QUEUE_ROOTS = tuple(Path(f"{_CONTROL_PLANE}/{name}") for name in (
    "task-evaluation-launches",
    "task-evaluation-launch-preparations",
    "task-evaluation-episode-compilations",
    "task-evaluation-launch-activations",
    "task-evaluation-policy-canary-dispatches",
    "task-evaluation-scene-constructions",
))
#: Queues beside the scene intents that also carry scene ids: SAM preparation children,
#: configured-scene activation intents, and capture reconstruction jobs (which read the
#: capture's staged raw bytes).
SCENE_QUEUE_CHILDREN = (
    "sam31-preparation-executions",
    "task-evaluation-scene-configuration-activation-intents",
    "capture-reconstruction-queue",
)
STORAGE_ROOT_ENV = "BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT"
INTENT_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_GC_SCENE_INTENT_ROOT"
INTAKE_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT"
BINDING_ROOT_ENV = "BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT"
QUEUE_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS"

#: What the Pub/Sub handoff listener writes into each capture (pubsub_handoff_listener). Copied rather
#: than imported so the reclaim timer does not load the pipeline; a test pins them to the listener.
LISTENER_FILES = {
    "ledger": "pipeline_job_ledger.json",
    "output_commit": "pipeline_job_output_commit.json",
    "terminal_receipt": "pipeline_job_terminal_receipt.json",
    "ack_receipt": "pipeline_job_ack_receipt.json",
    "staging_manifest": "pipeline_staging_manifest.json",
}
LISTENER_SCHEMAS = {
    "output_commit": "pipeline_job_output_commit.v1",
    "terminal_receipt": "pipeline_job_terminal_receipt.v1",
    "ack_receipt": "pubsub_handoff_ack_receipt.v1",
    "staging_manifest": "pipeline_handoff_staging_manifest.v1",
}
TERMINAL_AUTHORITY_STATUS = "terminal_authority_ended"
LEDGER_LOCK = f".{LISTENER_FILES['ledger']}.lock"
#: A terminal ledger status and the acknowledgement disposition that ends its message.
TERMINAL_DISPOSITIONS = {"completed": "terminal_success", TERMINAL_AUTHORITY_STATUS: TERMINAL_AUTHORITY_STATUS}
REGISTRATION_SCHEMA = "website_scene_source_registration.v1"
#: Where website_scene_handoff records the source registration it wrote for a capture.
WEBSITE_HANDOFF = ("pipeline", "website_scene_preparation", "handoff.json")
#: The most a record reader here (``_load_json``: the listener's receipt lookup, restore) reads.
_MAX_RECORD_BYTES = 16 * 1024 * 1024
#: A receipt must stay readable once its workspace is gone, so it may never outgrow its readers.
RECEIPT_MAX_BYTES = _MAX_RECORD_BYTES
#: A receipt at or under this is a small write that frees gigabytes, and ENOSPC on it fails before
#: anything is removed, so only a larger one reserves disk first.
_RECEIPT_RESERVATION_THRESHOLD = 16 * 1024 * 1024
_MAX_REASONS = 50
_SHA256 = "sha256:"


class WebsiteSceneWorkspaceRetentionError(RuntimeError):
    """A retirement was asked for something unsafe or unauthorized."""


@dataclass(frozen=True)
class CloudObject:
    name: str
    size: int
    generation: str | None
    md5_hash: str | None  # base64, as GCS reports it
    crc32c: str | None  # base64, as GCS reports it


class CloudInventory(Protocol):
    def list_objects(self, bucket: str, prefix: str) -> dict[str, CloudObject]: ...

    def download(self, bucket: str, name: str, destination: Path, *, generation: str) -> None: ...


class GcsCloudInventory:
    """Firebase Storage through ``google-cloud-storage``; the client is created on first use."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def _storage(self) -> Any:
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client()
        return self._client

    def list_objects(self, bucket: str, prefix: str) -> dict[str, CloudObject]:
        objects: dict[str, CloudObject] = {}
        for blob in self._storage().list_blobs(bucket, prefix=prefix):
            name = str(getattr(blob, "name", "") or "")
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            size = getattr(blob, "size", None)
            generation = getattr(blob, "generation", None)
            md5_hash = getattr(blob, "md5_hash", None)
            crc32c = getattr(blob, "crc32c", None)
            objects[name] = CloudObject(
                name=name,
                # An unknown size never equals a local size, so the object cannot verify.
                size=size if isinstance(size, int) and not isinstance(size, bool) else -1,
                generation=str(generation) if generation is not None else None,
                md5_hash=md5_hash if isinstance(md5_hash, str) and md5_hash else None,
                crc32c=crc32c if isinstance(crc32c, str) and crc32c else None,
            )
        return objects

    def download(self, bucket: str, name: str, destination: Path, *, generation: str) -> None:
        self._storage().bucket(bucket).blob(name, generation=int(generation)).download_to_filename(str(destination))


@dataclass(frozen=True)
class RetentionContext:
    storage_root: Path  # /var/lib/blueprint/pubsub-handoffs
    pins_root: Path
    queue_roots: tuple[Path, ...]  # a scene_id in <root>/{pending,processing}/*.json protects it
    intent_root: Path | None  # .../task-evaluation-scene-intents
    binding_root: Path | None  # .../website-source-bindings
    minimum_idle_seconds: int = DEFAULT_MINIMUM_IDLE_SECONDS
    ack_retention_seconds: int = DEFAULT_ACK_RETENTION_SECONDS
    orphan_registration_seconds: int = DEFAULT_ORPHAN_REGISTRATION_SECONDS
    expired_grace_seconds: int = DEFAULT_EXPIRED_GRACE_SECONDS
    #: <root>/<bucket>/<scene_id>.json per-file digests for plans; None re-hashes everything.
    inventory_cache_root: Path | None = None
    # Authority-ended captures can contain revoked local derivatives. No production
    # entrypoint enables their archival until an owner-approved deletion lifecycle exists.
    allow_authority_ended_archive: bool = False


@dataclass(frozen=True)
class ReferenceIndex:
    """Website source registrations and scene intents, read once per tick."""

    readable: bool
    binding_root: Path | None = None
    #: {"path", "request_digest", "reference_paths", "registered_at_epoch"}
    registrations: tuple[Mapping[str, Any], ...] = ()
    #: {"intent_id", "request_digest", "finished": "completed" | "revoked" | "expired" | None,
    #:  "open_attempts": unsettled attempt ids a revoked or expired intent still holds, which it
    #:  does only within the grace period after it finished}
    intents: tuple[Mapping[str, Any], ...] = ()


class _Unreadable(Exception):
    pass


def _identity(bucket: Any, scene_id: Any) -> tuple[str, str]:
    try:
        safe_bucket = strict_gcs_bucket(bucket)
        safe_scene = strict_identifier(scene_id, field="scene_id")
    except SecurityValidationError as exc:
        raise WebsiteSceneWorkspaceRetentionError("website_scene_workspace_identity_invalid") from exc
    if safe_bucket != bucket or safe_scene != scene_id:
        raise WebsiteSceneWorkspaceRetentionError("website_scene_workspace_identity_invalid")
    return safe_bucket, safe_scene


def scene_path(storage_root: Path, bucket: str, scene_id: str) -> Path:
    return Path(storage_root) / bucket / "scenes" / scene_id


def receipt_path(storage_root: Path, bucket: str, scene_id: str) -> Path:
    return Path(storage_root) / bucket / "scenes" / f"{scene_id}{RETIRED_SUFFIX}"


def binding_root_for(intent_root: Path | None, environ: Mapping[str, str] = os.environ) -> Path | None:
    """Where website sources are registered, as ``website_scene_dispatch.binding_root`` finds it."""

    configured = str(environ.get(BINDING_ROOT_ENV) or "").strip()
    if configured:
        return Path(configured)
    return None if intent_root is None else Path(intent_root).parent / "website-source-bindings"


def scene_queue_roots(queue_roots: Sequence[str | Path], intent_root: Path | None) -> tuple[Path, ...]:
    """The reclaim timer's queues plus the scene queues kept beside the intents."""

    roots = [Path(root) for root in queue_roots]
    if intent_root is not None:
        roots.extend(Path(intent_root).parent / name for name in SCENE_QUEUE_CHILDREN)
    return tuple(dict.fromkeys(roots))


def _load_json(path: Path) -> tuple[str, dict[str, Any] | None, os.stat_result | None]:
    """``("ok", object, stat)``, ``("absent", None, None)`` or ``("unreadable", None, stat)``.

    Never follows a symlink and reads only regular files of bounded size.
    """

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (FileNotFoundError, NotADirectoryError):
        return "absent", None, None
    except OSError:
        return "unreadable", None, None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_RECORD_BYTES:
            return "unreadable", None, info
        data = bytearray()
        while chunk := os.read(descriptor, 1 << 20):
            data += chunk
            if len(data) > _MAX_RECORD_BYTES:
                return "unreadable", None, info
    except OSError:
        return "unreadable", None, None
    finally:
        os.close(descriptor)
    try:
        value = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return "unreadable", None, info
    return ("ok", value, info) if isinstance(value, dict) else ("unreadable", None, info)


def _epoch(value: Any) -> float | None:
    from datetime import datetime, timezone

    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _allocated(info: os.stat_result) -> int:
    """Allocated bytes (``st_blocks * 512``), or the apparent size when no blocks are reported."""

    blocks = getattr(info, "st_blocks", None)
    if isinstance(blocks, int) and blocks > 0:
        return blocks * 512
    return int(info.st_size)


def _is_raw(relative: str) -> bool:
    parts = relative.split("/")
    return len(parts) >= 4 and parts[0] == "captures" and parts[2] == "raw"


# --- the workspace --------------------------------------------------------------------------------


@dataclass
class _Walk:
    files: list[tuple[str, os.stat_result]]
    unsafe: list[str]
    newest_mtime: float
    allocated_bytes: int


def _walk(scene: Path) -> _Walk:
    """Every regular file (relative POSIX path, lstat) without following a link.

    Allocated bytes count each inode once, as ``control_plane_disk_usage.tree_usage`` does.
    """

    top = os.lstat(scene)
    files: list[tuple[str, os.stat_result]] = []
    unsafe: list[str] = []
    newest = top.st_mtime
    allocated = _allocated(top)
    seen: set[tuple[int, int]] = set()
    pending: list[tuple[Path, str]] = [(scene, "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            unsafe.append(prefix or ".")
            continue
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                unsafe.append(relative)
                continue
            newest = max(newest, info.st_mtime)
            if stat.S_ISDIR(info.st_mode):
                allocated += _allocated(info)
                pending.append((Path(entry.path), relative))
            elif stat.S_ISREG(info.st_mode):
                key = (info.st_dev, info.st_ino)
                if key not in seen:
                    if info.st_nlink > 1:
                        seen.add(key)
                    allocated += _allocated(info)
                files.append((relative, info))
            else:  # a symlink, FIFO, socket or device
                unsafe.append(relative)
    files.sort(key=lambda row: row[0])
    return _Walk(files=files, unsafe=sorted(unsafe), newest_mtime=newest, allocated_bytes=allocated)


def _snapshot(files: Sequence[tuple[str, os.stat_result]]) -> list[list[Any]]:
    return [[relative, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino]
            for relative, info in files]


def _workspace_path_safe(
    storage_root: Path, bucket: str, scene: Path, *, unreadable_raises: bool = False
) -> bool:
    for path in (Path(storage_root), Path(storage_root) / bucket, Path(storage_root) / bucket / "scenes", scene):
        try:
            if not stat.S_ISDIR(os.lstat(path).st_mode):
                return False
        except FileNotFoundError:
            return False
        except OSError:
            if unreadable_raises:
                raise
            return False
    return True


def _valid_receipt(path: Path, *, bucket: str, scene_id: str) -> bool:
    state, receipt, _ = _load_json(path)
    return (
        state == "ok"
        and receipt is not None
        and receipt.get("schema_version") == RETIRED_SCHEMA
        and receipt.get("receipt_digest") == canonical_digest(receipt, digest_field="receipt_digest")
        and receipt.get("bucket") == bucket
        and receipt.get("scene_id") == scene_id
    )


def _receipt_matches_workspace(receipt: Mapping[str, Any], workspace: Path) -> bool:
    """Prove that a renamed tree contains exactly the receipt's captures and files."""

    try:
        walk = _walk(workspace)
        if walk.unsafe:
            return False
        capture_ids = sorted(row["capture_id"] for row in receipt["captures"])
        if _capture_ids(workspace) != capture_ids:
            return False
        cloud_rows = receipt["cloud_verified"]
        archive = receipt.get("archive")
        archive_rows = archive["members"] if isinstance(archive, Mapping) else []
        expected = [row["relative_path"] for row in cloud_rows] + [row["relative_path"] for row in archive_rows]
        if sorted(expected) != [relative for relative, _ in walk.files] or len(set(expected)) != len(expected):
            return False
        for row in cloud_rows:
            relative = _safe_relative(row["relative_path"])
            remote = CloudObject(name=relative, size=row["size"], generation=row["generation"],
                                 md5_hash=row["md5_hash"], crc32c=row["crc32c"])
            if not _verifies(remote, _hash_file(workspace / relative)):
                return False
        for row in archive_rows:
            relative = _safe_relative(row["relative_path"])
            if _sha256_file(workspace / relative) != row["sha256"]:
                return False
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return True


def sweep_retiring_workspaces(storage_root: Path, *, apply: bool = True) -> dict[str, list[str]]:
    """Finish removing ``.retiring-*`` copies a crash left behind, when their receipt restores them.

    A copy whose scene has no valid receipt is kept and reported: nothing proves it restorable.
    Without ``apply`` nothing is removed and the copies that would be are listed as ``removable``.
    """

    root = Path(storage_root)
    removed: list[str] = []
    kept: list[str] = []
    cleanup_incomplete: list[str] = []
    done_key = "removed" if apply else "removable"
    try:
        buckets = sorted(os.listdir(root))
    except OSError:
        return {done_key: removed, "kept_without_receipt": kept, "cleanup_incomplete": cleanup_incomplete}
    for bucket in buckets:
        scenes = root / bucket / "scenes"
        if bucket.startswith(".") or not _workspace_path_safe(root, bucket, scenes):
            continue
        try:
            names = sorted(os.listdir(scenes))
        except OSError:
            continue
        for name in names:
            match = _RETIRING_NAME.fullmatch(name)
            path = scenes / name
            try:
                if match is None or not stat.S_ISDIR(os.lstat(path).st_mode):
                    continue
            except OSError:
                continue
            record_path = receipt_path(root, bucket, match["scene"])
            state, receipt, _ = _load_json(record_path)
            if (_valid_receipt(record_path, bucket=bucket, scene_id=match["scene"])
                    and state == "ok" and receipt is not None
                    and receipt.get("retiring_token") == match["token"]
                    and _receipt_matches_workspace(receipt, path)):
                if apply:
                    shutil.rmtree(path, ignore_errors=True)
                    if os.path.lexists(path):
                        cleanup_incomplete.append(str(path))
                        continue
                removed.append(str(path))
            else:
                kept.append(str(path))
    return {done_key: removed, "kept_without_receipt": kept, "cleanup_incomplete": cleanup_incomplete}


def sweep_retirement_temporaries(storage_root: Path, *, now: float) -> list[str]:
    """Remove only old, unreferenced receipt temp files from interrupted writes."""

    removed: list[str] = []
    root = Path(storage_root)
    try:
        buckets = sorted(os.listdir(root))
    except OSError:
        return removed
    for bucket in buckets:
        scenes = root / bucket / "scenes"
        if bucket.startswith(".") or not _workspace_path_safe(root, bucket, scenes):
            continue
        try:
            names = sorted(os.listdir(scenes))
        except OSError:
            continue
        for name in names:
            if not _RECEIPT_TEMP.fullmatch(name):
                continue
            path = scenes / name
            try:
                info = os.lstat(path)
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                        or now - info.st_mtime < 24 * 3600 or _process_in_use(path)):
                    continue
                path.unlink()
            except (OSError, ValueError):
                continue
            removed.append(str(path))
    return removed


def scene_workspaces(storage_root: Path) -> list[tuple[str, str, Path]]:
    """``(bucket, scene_id, path)`` of every scene working copy under the spool, sorted."""

    root = Path(storage_root)
    rows: list[tuple[str, str, Path]] = []
    try:
        buckets = sorted(os.listdir(root))
    except FileNotFoundError:
        return rows
    for bucket in buckets:
        scenes = root / bucket / "scenes"
        if bucket.startswith(".") or not _workspace_path_safe(root, bucket, scenes, unreadable_raises=True):
            continue
        try:
            strict_gcs_bucket(bucket)
            names = sorted(os.listdir(scenes))
        except (SecurityValidationError, FileNotFoundError):
            continue
        for name in names:
            path = scenes / name
            if name.startswith(".") or name.endswith(RETIRED_SUFFIX):
                continue
            try:
                if strict_identifier(name, field="scene_id") != name:
                    continue
                mode = os.lstat(path).st_mode
            except (SecurityValidationError, FileNotFoundError):
                continue
            # A symlink is listed so its plan reports it; a stray regular file is not a workspace.
            if stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                rows.append((bucket, name, path))
    return rows


# --- the reference index: website source registrations and the intents that resolve them ----------


def _registrations(binding_root: Path | None) -> tuple[Mapping[str, Any], ...]:
    if binding_root is None:
        raise _Unreadable
    try:
        info = os.lstat(binding_root)
    except FileNotFoundError:
        return ()  # nothing was ever registered
    except OSError as exc:
        raise _Unreadable from exc
    if not stat.S_ISDIR(info.st_mode):
        raise _Unreadable
    try:
        names = sorted(os.listdir(binding_root))
    except OSError as exc:
        raise _Unreadable from exc
    rows: list[Mapping[str, Any]] = []
    for name in names:
        if name.startswith(".") or not name.endswith(".json"):
            continue  # exclusive writers stage hidden temporaries
        state, value, record_info = _load_json(Path(binding_root) / name)
        if state != "ok" or value is None or record_info is None:
            raise _Unreadable
        references = value.get("references")
        request_digest = value.get("request_digest")
        if (
            value.get("schema_version") != REGISTRATION_SCHEMA
            or value.get("registration_digest") != canonical_digest(value, digest_field="registration_digest")
            or not isinstance(request_digest, str)
            or not request_digest.startswith(_SHA256)
            or not isinstance(references, Mapping)
            or not references
            or not all(isinstance(ref, Mapping) and isinstance(ref.get("path"), str) for ref in references.values())
        ):
            raise _Unreadable
        rows.append({
            "path": str(Path(binding_root) / name),
            "request_digest": request_digest,
            "reference_paths": tuple(str(ref["path"]) for ref in references.values()),
            "registered_at_epoch": float(record_info.st_mtime),
        })
    return tuple(rows)


def _revoked_at(path: Path) -> float:
    """When an intent was revoked: the time its receipt records, else the receipt file's time."""

    state, receipt, _ = _load_json(path)
    value = receipt.get("revoked_at_epoch") if state == "ok" and receipt is not None else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return os.stat(path).st_mtime


def _intent_finished(directory: Path, intent: Mapping[str, Any], *, now: float,
                     expired_grace_seconds: int) -> tuple[str | None, float | None]:
    """Why scene progression will never resolve this intent's source again, and since when.

    Mirrors ``task_evaluation_scene_progression._advance_intent``: a completed
    progression, a revocation, or an elapsed (possibly extended) execution window.
    An owner can still extend an elapsed window, so expiry counts only once the
    grace period after it has passed too. ``(None, None)`` while the intent is open.
    """

    from . import task_evaluation_scene_intake as intake

    projection_path = directory / "progression.json"
    if os.path.lexists(projection_path):
        projection = intake._read(projection_path, "progression_digest")
        if projection.get("intent_digest") != intent["intent_digest"] or projection.get("intent_id") != intent["intent_id"]:
            raise _Unreadable
        if projection.get("status") == "completed":
            return "completed", None
    revocation = directory / "revoked.json"
    if revocation.exists():
        return "revoked", _revoked_at(revocation)
    expiry = intake.effective_execution_expiry(directory, intent)
    if now >= expiry + expired_grace_seconds:
        return "expired", expiry
    return None, None


def _open_attempts(directory: Path) -> tuple[str, ...]:
    """Attempt rows scene progression still treats as live: no validated cancellation or settlement.

    Read the way progression reads them (``attempts/*.json`` under the intent, each
    digest-checked, terminal only through ``validated_cancellation``); anything
    unreadable raises and so protects every scene.
    """

    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_retained_controls_evidence import validated_cancellation

    attempts = directory / "attempts"
    try:
        mode = os.lstat(attempts).st_mode
    except FileNotFoundError:
        return ()
    if not stat.S_ISDIR(mode):
        raise _Unreadable
    live = []
    for path in sorted(attempts.glob("*.json")):
        row = intake._read(path, "attempt_digest")
        if validated_cancellation(directory, row) is None:
            live.append(str(row.get("attempt_id") or path.stem))
    return tuple(live)


def _intents(intent_root: Path | None, *, now: float,
             expired_grace_seconds: int = DEFAULT_EXPIRED_GRACE_SECONDS) -> tuple[Mapping[str, Any], ...]:
    from . import task_evaluation_scene_intake as intake

    if intent_root is None:
        raise _Unreadable
    try:
        if not stat.S_ISDIR(os.lstat(intent_root).st_mode):
            raise _Unreadable
        names = sorted(os.listdir(intent_root))
    except OSError as exc:
        raise _Unreadable from exc
    rows: list[Mapping[str, Any]] = []
    for name in names:
        if not name.startswith("scene-"):
            continue
        directory = Path(intent_root) / name
        try:
            mode = os.lstat(directory).st_mode
            if stat.S_ISLNK(mode):
                raise _Unreadable
            if not stat.S_ISDIR(mode):
                continue  # scene progression reads directories only
            intent = intake._read(directory / "intent.json", "intent_digest")
            request = intent.get("request")
            if intent.get("intent_id") != name or not isinstance(request, Mapping):
                raise _Unreadable
            finished, finished_at = _intent_finished(directory, intent, now=now,
                                                     expired_grace_seconds=expired_grace_seconds)
            # A completed intent's attempts ended with it: progression completes only after
            # joining the attempt's terminal result, and only retired predecessors are ever
            # settled, so its own row stays a spend hold forever. A revoked or expired intent
            # holds an attempt it has not cancelled or settled, but only for the grace period
            # after it finished: materialization copies the workspace inputs into the
            # attempt's own staging, and no factory pass runs for days, so every hold expires.
            # (An expired intent's period has passed by the time it counts as finished.)
            held = finished in {"revoked", "expired"} and now < float(finished_at) + expired_grace_seconds
            rows.append({
                "intent_id": name,
                "request_digest": cross_runtime_canonical_digest(request),
                "finished": finished,
                "open_attempts": _open_attempts(directory) if held else (),
            })
        except _Unreadable:
            raise
        except Exception as exc:  # noqa: BLE001 - an intent that cannot be read protects every scene
            raise _Unreadable from exc
    return tuple(rows)


def build_reference_index(context: RetentionContext, *, now: float) -> ReferenceIndex:
    """Read every website source registration and scene intent once; unreadable means protect all."""

    try:
        registrations = _registrations(context.binding_root)
        intents = _intents(context.intent_root, now=float(now), expired_grace_seconds=context.expired_grace_seconds)
    except _Unreadable:
        return ReferenceIndex(readable=False, binding_root=context.binding_root)
    return ReferenceIndex(readable=True, binding_root=context.binding_root,
                          registrations=registrations, intents=intents)


def _under(path_text: str, roots: Sequence[Path]) -> bool:
    candidate = Path(os.path.normpath(path_text))
    return any(candidate == root or root in candidate.parents for root in roots)


def _same_directory(left: Path, right: Path) -> bool:
    if os.path.normpath(left) == os.path.normpath(right):
        return True
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


_UNKNOWN = object()


def _named_registration(capture_root: Path) -> Any:
    """The registration a capture's website handoff says it wrote: a path, None, or ``_UNKNOWN``."""

    state, handoff, _ = _load_json(capture_root.joinpath(*WEBSITE_HANDOFF))
    if state == "absent":
        return None
    if state != "ok" or handoff is None:
        return _UNKNOWN
    registration = handoff.get("source_registration")
    if registration is None:
        return None
    if not isinstance(registration, Mapping) or not isinstance(registration.get("path"), str):
        return _UNKNOWN
    return Path(registration["path"])


def _reference_reasons(
    *, scene: Path, capture_ids: Sequence[str], index: ReferenceIndex, now: float, context: RetentionContext
) -> list[str]:
    if not index.readable or index.binding_root is None:
        return ["reference_index_unreadable"]
    reasons: list[str] = []
    forms = [Path(os.path.normpath(scene))]
    try:
        forms.append(scene.resolve())
    except OSError:
        pass
    for registration in index.registrations:
        if not any(_under(path, forms) for path in registration["reference_paths"]):
            continue
        claimed = [row for row in index.intents if row["request_digest"] == registration["request_digest"]]
        reasons.extend(f"open_scene_intent:{row['intent_id']}" for row in claimed if row["finished"] is None)
        reasons.extend(f"open_scene_attempt:{row['intent_id']}/{attempt}"
                       for row in claimed for attempt in row.get("open_attempts", ()))
        if not claimed and now - float(registration["registered_at_epoch"]) < context.orphan_registration_seconds:
            reasons.append("unclaimed_source_registration")
    for capture_id in capture_ids:
        named = _named_registration(scene / "captures" / capture_id)
        if named is _UNKNOWN or (named is not None and not _same_directory(Path(named).parent, index.binding_root)):
            # A registration this index did not read (another binding root) could still be resolved.
            reasons.append(f"source_registration_unindexed:{capture_id}")
    return reasons


# --- captures -------------------------------------------------------------------------------------


def _terminal_proven(capture_root: Path, *, status: str, ledger: Mapping[str, Any], scene_id: str,
                     capture_id: str) -> bool:
    if status == "completed":
        state, commit, _ = _load_json(capture_root / LISTENER_FILES["output_commit"])
        return (
            state == "ok"
            and commit is not None
            and commit.get("schema_version") == LISTENER_SCHEMAS["output_commit"]
            and commit.get("status") == "committed"
            and commit.get("scene_id") == scene_id
            and commit.get("capture_id") == capture_id
            and isinstance(commit.get("result_sha256"), str)
            and bool(commit["result_sha256"].strip())
        )
    state, receipt, _ = _load_json(capture_root / LISTENER_FILES["terminal_receipt"])
    ended_by = ledger.get("terminal_payload_sha256")
    return (
        state == "ok"
        and receipt is not None
        and receipt.get("schema_version") == LISTENER_SCHEMAS["terminal_receipt"]
        and receipt.get("status") == "authority_ended"
        and receipt.get("receipt_digest") == canonical_digest(receipt, digest_field="receipt_digest")
        and receipt.get("scene_id") == scene_id
        and receipt.get("capture_id") == capture_id
        and isinstance(ended_by, str)
        and bool(ended_by)
        and receipt.get("payload_sha256") == ended_by
    )


def _acknowledgement(capture_root: Path, *, status: str, ledger: Mapping[str, Any], now: float,
                     ack_retention_seconds: int) -> str | None:
    """How the capture's last terminal message is known to be acknowledged, or None."""

    ended = status == TERMINAL_AUTHORITY_STATUS
    terminal_at = _epoch(ledger.get("terminal_at") if ended else ledger.get("completed_at"))
    state, ack, _ = _load_json(capture_root / LISTENER_FILES["ack_receipt"])
    if (
        state == "ok"
        and ack is not None
        and ack.get("schema_version") == LISTENER_SCHEMAS["ack_receipt"]
        and ack.get("disposition") == TERMINAL_DISPOSITIONS[status]
        and (not ended or ack.get("payload_sha256") == ledger.get("terminal_payload_sha256"))
    ):
        acknowledged_at = _epoch(ack.get("acknowledged_at"))
        # An acknowledgement older than the terminal state acknowledged an earlier message.
        if acknowledged_at is not None and terminal_at is not None and acknowledged_at >= terminal_at:
            return "ack_receipt"
    updated_at = _epoch(ledger.get("updated_at"))
    if updated_at is not None and now - updated_at >= ack_retention_seconds:
        return "pubsub_retention_elapsed"  # Pub/Sub can no longer redeliver it
    return None


def _capture(capture_root: Path, *, scene_id: str, capture_id: str, now: float,
             ack_retention_seconds: int) -> tuple[dict[str, Any], list[str]]:
    row: dict[str, Any] = {"capture_id": capture_id, "ledger_status": None, "acknowledgement": None}
    state, ledger, _ = _load_json(capture_root / LISTENER_FILES["ledger"])
    if state == "absent":
        return row, [f"capture_not_terminal:{capture_id}"]
    if (
        state != "ok"
        or ledger is None
        or ledger.get("scene_id") != scene_id
        or ledger.get("capture_id") != capture_id
    ):
        return row, [f"capture_ledger_unreadable:{capture_id}"]
    status = ledger.get("status") if isinstance(ledger.get("status"), str) else ""
    row["ledger_status"] = status
    reasons: list[str] = []
    lease = ledger.get("lease_expires_at")
    if lease is not None:
        expires = _epoch(lease)
        if expires is None or expires > now:
            reasons.append(f"capture_lease_held:{capture_id}")
    try:
        lock_ok = stat.S_ISREG(os.lstat(capture_root / LEDGER_LOCK).st_mode)
    except OSError:
        lock_ok = False
    if not lock_ok:
        # Retirement excludes the listener through this lock; without it nothing would.
        reasons.append(f"capture_lock_missing:{capture_id}")
    if status not in TERMINAL_DISPOSITIONS or not _terminal_proven(
        capture_root, status=status, ledger=ledger, scene_id=scene_id, capture_id=capture_id
    ):
        reasons.append(f"capture_not_terminal:{capture_id}")
        return row, reasons
    if status == "completed":
        stage_state, stage_ledger, _ = _load_json(capture_root / "pipeline" / "run_e2e_stage_ledger.json")
        if stage_state == "unreadable":
            reasons.append(f"capture_result_unreadable:{capture_id}")
        elif stage_ledger is not None:
            stages = stage_ledger.get("stages")
            capture_result = stages.get("capture_pipeline") if isinstance(stages, Mapping) else None
            snapshot = capture_result.get("result_snapshot") if isinstance(capture_result, Mapping) else None
            if isinstance(snapshot, Mapping) and snapshot.get("status") == "completed_with_lane_failures":
                reasons.append(f"capture_not_terminal:{capture_id}")
    acknowledgement = _acknowledgement(capture_root, status=status, ledger=ledger, now=now,
                                       ack_retention_seconds=ack_retention_seconds)
    row["acknowledgement"] = acknowledgement
    if acknowledgement is None:
        reasons.append(f"acknowledgement_unproven:{capture_id}")
    return row, reasons


def _capture_ids(scene: Path) -> list[str] | None:
    captures = scene / "captures"
    try:
        if not stat.S_ISDIR(os.lstat(captures).st_mode):
            return None
        names = sorted(os.listdir(captures))
    except OSError:
        return None
    ids = []
    for name in names:
        try:
            if stat.S_ISDIR(os.lstat(captures / name).st_mode):
                ids.append(name)
        except OSError:
            continue
    return ids


def _is_website_capture(capture_root: Path) -> bool:
    """The lane the listener gave this capture: ``is_website_capture_manifest`` on its raw manifest."""

    state, manifest, _ = _load_json(capture_root / "raw" / "manifest.json")
    return state == "ok" and is_website_capture_manifest(manifest)


def _process_in_use(path: Path) -> bool:
    # This process holds the ledger locks while it re-checks, so it must not count itself.
    return bool(active_reference(path, ignored_process_ids=(os.getpid(),)))


@dataclass
class _Evaluation:
    reasons: list[str]
    files: list[tuple[str, os.stat_result]]
    captures: list[dict[str, Any]]
    capture_ids: list[str]
    idle_seconds: float | None
    allocated_bytes: int


def _evaluate(*, context: RetentionContext, bucket: str, scene_id: str, scene: Path, now: float,
              index: ReferenceIndex | None, process_checker: Callable[[Path], bool]) -> _Evaluation:
    """Checks 1-8. They read local state only and never touch the workspace."""

    if not _workspace_path_safe(context.storage_root, bucket, scene):
        return _Evaluation(["workspace_path_unsafe"], [], [], [], None, 0)
    # A pin or queue message already forbids deletion. Avoid walking and statting
    # a multi-gigabyte tree on every GC tick while either cheap hold remains.
    early: list[str] = []
    if _pinned_workspace(scene, live_pinned_paths(context.pins_root, now=lambda: now)):
        early.append("pinned")
    try:
        if scene_id in queue_reference_text(context.queue_roots):
            early.append("queue_referenced")
    except QueueInventoryUnavailable:
        early.append("queue_inventory_unreadable")
    if early:
        return _Evaluation(sorted(early), [], [], _capture_ids(scene) or [], None, 0)
    walk = _walk(scene)
    reasons = [f"unsafe_entry:{relative}" for relative in walk.unsafe]
    captures: list[dict[str, Any]] = []
    capture_ids = _capture_ids(scene) or []
    if not capture_ids:
        reasons.append("scene_has_no_captures")
    elif not all(_is_website_capture(scene / "captures" / capture_id) for capture_id in capture_ids):
        # Retirement is for finished website scenes; a device capture's staging has other readers.
        reasons.append("not_a_website_scene")
    for capture_id in capture_ids:
        row, capture_reasons = _capture(scene / "captures" / capture_id, scene_id=scene_id, capture_id=capture_id,
                                        now=now, ack_retention_seconds=context.ack_retention_seconds)
        captures.append(row)
        reasons.extend(capture_reasons)
        if (row["ledger_status"] == TERMINAL_AUTHORITY_STATUS and not capture_reasons
                and not context.allow_authority_ended_archive):
            reasons.append("authority_ended_capture_kept_local")
    idle_seconds = now - walk.newest_mtime
    if idle_seconds < context.minimum_idle_seconds:
        reasons.append("recently_active")
    try:
        in_use = bool(process_checker(scene))
    except Exception:  # noqa: BLE001 - a process inventory we cannot read proves nothing is idle
        in_use = True
    if in_use:
        reasons.append("in_use")
    if index is None:
        index = build_reference_index(context, now=now)
    reasons.extend(_reference_reasons(scene=scene, capture_ids=capture_ids, index=index, now=now, context=context))
    return _Evaluation(sorted(set(reasons)), walk.files, captures, capture_ids, idle_seconds, walk.allocated_bytes)


# --- check 9: every file is recoverable -----------------------------------------------------------


def _verifies(cloud: CloudObject, local: _Digests) -> bool:
    if cloud.size != local.size:
        return False
    if cloud.md5_hash:
        return cloud.md5_hash == local.md5
    return bool(cloud.crc32c) and cloud.crc32c == local.crc32c


def inventory_cache_path(cache_root: Path, bucket: str, scene_id: str) -> Path:
    return Path(cache_root) / bucket / f"{scene_id}.json"


def _load_inventory_cache(path: Path, *, bucket: str, scene_id: str) -> dict[str, Any]:
    """Cached digests by relative path, or nothing when the cache is absent or not this scene's."""

    state, cache, _ = _load_json(path)
    files = cache.get("files") if state == "ok" and cache is not None else None
    if (
        not isinstance(files, dict)
        or cache.get("schema_version") != INVENTORY_CACHE_SCHEMA
        or cache.get("bucket") != bucket
        or cache.get("scene_id") != scene_id
    ):
        return {}
    return files


def _inventory(*, scene: Path, bucket: str, scene_id: str, files: Sequence[tuple[str, os.stat_result]],
               cloud: CloudInventory, cache_path: Path | None = None,
               budget: HashBudget | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int]:
    """Match every file to its cloud object, hashing only files the cache does not already know.

    A cached digest counts only for the same (size, mtime_ns, ctime_ns, device, inode). A file the tick's
    budget cannot cover is deferred (``inventory_deferred``); what was hashed is cached, so
    the next tick carries on. Apply never trusts the cache: it re-hashes what it deletes.
    """

    prefix = f"scenes/{scene_id}/"
    try:
        listing = dict(cloud.list_objects(bucket, prefix))
    except Exception:  # noqa: BLE001 - without a listing nothing is proven recoverable
        return [], [], ["cloud_inventory_unavailable"], 0
    cache = _load_inventory_cache(cache_path, bucket=bucket, scene_id=scene_id) if cache_path else {}
    fresh: dict[str, Any] = {}
    verified: list[dict[str, Any]] = []
    archive: list[dict[str, Any]] = []
    reasons: list[str] = []
    hashed = 0
    for relative, info in files:
        identity = [info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino]
        digests = _cached_digests(cache.get(relative), identity)
        if digests is None:
            if budget is not None and not budget.allows(info.st_size):
                reasons.append("inventory_deferred")
                continue
            try:
                digests = _hash_file(scene / relative, deadline_monotonic=(
                    budget.deadline_for(info.st_size) if budget is not None else None))
            except HashDeadlineExceeded:
                reasons.append("oversized_hash_timeout" if budget is not None
                               and info.st_size > budget.oversized_file_threshold else "inventory_deferred")
                continue
            except OSError:
                reasons.append(f"unsafe_entry:{relative}")
                continue
            hashed += digests.size
            if budget is not None:
                budget.spend(digests.size)
        if digests.size != info.st_size:
            reasons.append("recently_active")  # it changed while it was read
            continue
        fresh[relative] = {"identity": identity, "digests": {"size": digests.size, "sha256": digests.sha256,
                                                             "md5": digests.md5, "crc32c": digests.crc32c}}
        remote = listing.get(prefix + relative)
        if (remote is not None and isinstance(remote.generation, str)
                and remote.generation.isdecimal() and _verifies(remote, digests)):
            verified.append({"relative_path": relative, "uri": f"gs://{bucket}/{prefix}{relative}",
                             "generation": remote.generation, "size": remote.size,
                             "md5_hash": remote.md5_hash, "crc32c": remote.crc32c})
        elif _is_raw(relative):
            # Raw capture bytes live only in Firebase Storage; they are never archived here.
            reasons.append(f"raw_not_verified_in_cloud:{relative}")
        else:
            archive.append({"relative_path": relative, "size_bytes": digests.size,
                            "sha256": _SHA256 + digests.sha256})
    if cache_path is not None:
        save_inventory_cache(cache_path, bucket=bucket, scene_id=scene_id, files=fresh,
                             schema=INVENTORY_CACHE_SCHEMA)
    return verified, archive, reasons, hashed


# --- plan -----------------------------------------------------------------------------------------


def plan_scene_workspace_retirement(
    *,
    context: RetentionContext,
    bucket: str,
    scene_id: str,
    now: float,
    cloud: CloudInventory,
    index: ReferenceIndex | None = None,
    process_checker: Callable[[Path], bool] | None = None,
    hash_budget: HashBudget | None = None,
) -> dict[str, Any]:
    """Decide, without touching the workspace, whether one scene workspace may be retired.

    ``process_checker`` defaults to ``active_reference`` over ``/proc``, ignoring this process.
    With ``context.inventory_cache_root`` file digests are cached, and ``hash_budget`` bounds the
    uncached bytes this plan may hash.
    """

    bucket, scene_id = _identity(bucket, scene_id)
    observed_at = float(now)
    scene = scene_path(context.storage_root, bucket, scene_id)
    evaluation = _evaluate(context=context, bucket=bucket, scene_id=scene_id, scene=scene, now=observed_at,
                           index=index, process_checker=process_checker or _process_in_use)
    reasons = list(evaluation.reasons)
    verified: list[dict[str, Any]] = []
    archive: list[dict[str, Any]] = []
    hashed = 0
    if not reasons:  # the cloud inventory runs only when nothing cheaper retained the scene
        cache_path = (inventory_cache_path(context.inventory_cache_root, bucket, scene_id)
                      if context.inventory_cache_root is not None else None)
        verified, archive, inventory_reasons, hashed = _inventory(
            scene=scene, bucket=bucket, scene_id=scene_id, files=evaluation.files, cloud=cloud,
            cache_path=cache_path, budget=hash_budget)
        reasons = sorted(set(inventory_reasons))
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "status": "retained" if reasons else "retirable",
        "bucket": bucket,
        "scene_id": scene_id,
        "workspace": str(scene),
        "observed_at_epoch": observed_at,
        "idle_seconds": None if evaluation.idle_seconds is None else int(evaluation.idle_seconds),
        "captures": evaluation.captures,
        "reasons": reasons[:_MAX_REASONS],
        "reason_count": len(reasons),
        "cloud_verified": [] if reasons else verified,
        "archive": [] if reasons else archive,
        "snapshot": _snapshot(evaluation.files),
        "totals": {
            "file_count": len(evaluation.files),
            "cloud_verified_bytes": 0 if reasons else sum(row["size"] for row in verified),
            "archive_bytes": 0 if reasons else sum(row["size_bytes"] for row in archive),
            "workspace_allocated_bytes": evaluation.allocated_bytes,
            "hashed_bytes": hashed,
        },
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


# --- apply ----------------------------------------------------------------------------------------


class _CaptureLocks:
    """The listener's own per-capture ledger locks, taken in sorted order without waiting.

    The lock files are opened read-only and never created, so taking them changes nothing
    in the workspace; closing the descriptors releases them.
    """

    def __init__(self, scene: Path, capture_ids: Sequence[str]) -> None:
        self._paths = [scene / "captures" / capture_id / LEDGER_LOCK for capture_id in sorted(capture_ids)]
        self._descriptors: list[int] = []
        self.refusal: str | None = None

    def __enter__(self) -> "_CaptureLocks":
        try:
            for path in self._paths:
                try:
                    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                except OSError:
                    self.refusal = "candidate_changed"  # the lock the plan saw is gone or replaced
                    return self
                self._descriptors.append(descriptor)
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    self.refusal = "candidate_changed"
                    return self
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:  # BlockingIOError: the listener holds it right now
                    self.refusal = "candidate_busy"
                    return self
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_exc: object) -> None:
        while self._descriptors:
            os.close(self._descriptors.pop())


def _authorized_plan(plan: Any, *, ack: str, context: RetentionContext) -> tuple[str, str, Path]:
    refused = WebsiteSceneWorkspaceRetentionError("website_scene_workspace_retirement_apply_not_authorized")
    if (
        ack != RETIRE_ACK
        or not isinstance(plan, Mapping)
        or plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "retirable"
        or plan.get("reasons")
        or plan.get("plan_digest") != canonical_digest(dict(plan), digest_field="plan_digest")
    ):
        raise refused
    try:
        bucket, scene_id = _identity(plan.get("bucket"), plan.get("scene_id"))
        scene = scene_path(context.storage_root, bucket, scene_id)
        verified = [row["relative_path"] for row in plan["cloud_verified"]]
        archived = [row["relative_path"] for row in plan["archive"]]
        snapshot = [row[0] for row in plan["snapshot"]]
        well_formed = (
            plan.get("workspace") == str(scene)
            and all(isinstance(row["capture_id"], str) for row in plan["captures"])
            and all(isinstance(path, str) for path in (*verified, *archived, *snapshot))
            # Raw capture bytes are never archived, and every file is either verified or archived, once.
            and not any(_is_raw(path) for path in archived)
            and len(set(verified) | set(archived)) == len(verified) + len(archived)
            and sorted(verified + archived) == sorted(snapshot)
        )
    except (WebsiteSceneWorkspaceRetentionError, KeyError, TypeError, IndexError) as exc:
        raise refused from exc
    if not well_formed:
        raise refused
    return bucket, scene_id, scene


def _capture_records(scene: Path, captures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Parsed copies of each capture's records, so a retired capture stays idempotent."""

    def parsed(path: Path) -> dict[str, Any] | None:
        state, value, _ = _load_json(path)
        return value if state == "ok" else None

    records = []
    for row in captures:
        root = scene / "captures" / row["capture_id"]
        record: dict[str, Any] = {"capture_id": row["capture_id"], "ledger": parsed(root / LISTENER_FILES["ledger"])}
        if row["ledger_status"] == TERMINAL_AUTHORITY_STATUS:
            record["terminal_receipt"] = parsed(root / LISTENER_FILES["terminal_receipt"])
        else:
            record["output_commit"] = parsed(root / LISTENER_FILES["output_commit"])
        record["ack_receipt"] = parsed(root / LISTENER_FILES["ack_receipt"])
        record["staging_manifest"] = parsed(root / LISTENER_FILES["staging_manifest"])
        records.append(record)
    return records


def _archive(scene: Path, rows: Sequence[Mapping[str, Any]],
             publisher: Callable[..., Mapping[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    """Stream exactly the planned local-only files to the artifact store and prove the remote bytes."""

    members = [str(row["relative_path"]) for row in rows]
    sink = _HashingSink()
    try:
        packed = _pack_stream(scene, sink, members=members)
    except (OSError, ControlPlaneEvidenceOffloadError):
        return None, "candidate_changed"
    planned = [(row["relative_path"], row["size_bytes"], row["sha256"]) for row in rows]
    if [(row["relative_path"], row["size_bytes"], row["sha256"]) for row in packed] != planned:
        return None, "candidate_changed"
    digest, size = "sha256:" + sink.digest.hexdigest(), sink.size
    try:
        reference = dict(publisher(
            write_stream=lambda stream: _pack_stream(scene, stream, members=members),
            digest=digest, size_bytes=size, filename=ARCHIVE_FILENAME, artifact_kind=ARTIFACT_KIND))
    except Exception:  # noqa: BLE001 - any publication failure keeps every local byte
        return None, "archive_readback_failed"
    if (
        reference.get("digest") != digest
        or reference.get("size_bytes") != size
        or reference.get("full_byte_service_account_readback_passed") is not True
        or not isinstance(reference.get("uri"), str)
    ):
        return None, "archive_readback_failed"
    return {"uri": reference["uri"], "digest": digest, "size_bytes": size,
            "member_count": len(packed), "members": packed}, None


def _cloud_unchanged(cloud: CloudInventory, *, bucket: str, scene_id: str,
                     rows: Sequence[Mapping[str, Any]]) -> str | None:
    prefix = f"scenes/{scene_id}/"
    try:
        listing = dict(cloud.list_objects(bucket, prefix))
    except Exception:  # noqa: BLE001 - an unproven cloud copy keeps the local one
        return "cloud_inventory_unavailable"
    for row in rows:
        remote = listing.get(prefix + row["relative_path"])
        if remote is None or (remote.generation, remote.size, remote.md5_hash, remote.crc32c) != (
            row["generation"], row["size"], row["md5_hash"], row["crc32c"]
        ):
            return "cloud_changed"
    return None


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _receipt_document(*, bucket: str, scene_id: str, scene: Path, observed_at: float, plan: Mapping[str, Any],
                      records: list[dict[str, Any]], archive: dict[str, Any] | None,
                      retiring_token: str) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": RETIRED_SCHEMA,
        "bucket": bucket,
        "scene_id": scene_id,
        "workspace": str(scene),
        "retired_at_epoch": observed_at,
        "source_plan_digest": plan["plan_digest"],
        "retiring_token": retiring_token,
        "captures": records,
        "cloud_verified": list(plan["cloud_verified"]),
        "archive": archive,
        "totals": dict(plan["totals"]),
        "evidence_deleted": False,
        "receipt_digest": "",
    }
    document["receipt_digest"] = canonical_digest(document, digest_field="receipt_digest")
    return document


def _receipt_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _largest_archive_record(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """An archive record at least as large as publishing these rows can produce, to size the receipt."""

    if not rows:
        return None
    return {"uri": "s3://" + "x" * 1024, "digest": _SHA256 + "0" * 64, "size_bytes": 2**63,
            "member_count": len(rows),
            "members": [{"relative_path": row["relative_path"], "size_bytes": row["size_bytes"],
                         "sha256": row["sha256"]} for row in rows]}


def _publish_receipt(path: Path, data: bytes) -> bool:
    """Create the receipt exclusively and completely, owned like the ``scenes/`` directory it lives in.

    Returns False, writing nothing, when a receipt already exists.
    """

    owner = os.lstat(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o640)
    try:
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(descriptor, view):]
            os.fchmod(descriptor, 0o640)
            # The GC runs as root; the listener that reads receipts runs as the service account.
            os.fchown(descriptor, owner.st_uid, owner.st_gid)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _quarantine_receipt(path: Path, token: str) -> Path:
    """Preserve a receipt that no longer describes the live workspace."""

    quarantine = path.with_name(f".{path.name}.{token}.recovery")
    os.replace(path, quarantine)
    _fsync_directory(path.parent)
    return quarantine


def apply_scene_workspace_retirement(
    plan: Mapping[str, Any],
    *,
    context: RetentionContext,
    ack: str,
    cloud: CloudInventory,
    now: float,
    index: ReferenceIndex | None = None,
    stream_publisher: Callable[..., Mapping[str, Any]] | None = None,
    process_checker: Callable[[Path], bool] | None = None,
) -> dict[str, Any]:
    """Retire one planned workspace, keeping listener locks out of the upload path."""

    bucket, scene_id, scene = _authorized_plan(plan, ack=ack, context=context)
    publisher = stream_publisher or publish_configured_scene_stream
    observed_at = float(now)
    receipt = receipt_path(context.storage_root, bucket, scene_id)
    base = {"bucket": bucket, "scene_id": scene_id, "source_plan_digest": plan["plan_digest"]}
    archive: dict[str, Any] | None = None

    def skipped(reason: str) -> dict[str, Any]:
        result = {**base, "status": "skipped", "reason": reason}
        if archive is not None:
            result["published_archive"] = {key: archive[key] for key in ("uri", "digest", "size_bytes")}
        return result

    planned_ids = [row["capture_id"] for row in plan["captures"]]
    checker = process_checker or _process_in_use
    if os.path.lexists(receipt):
        state, previous, _ = _load_json(receipt)
        if (state != "ok" or previous is None
                or not _valid_receipt(receipt, bucket=bucket, scene_id=scene_id)):
            return skipped("receipt_invalid")
        token = previous.get("retiring_token")
        if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{16}", token) is None:
            return skipped("receipt_invalid")
        if not _receipt_matches_workspace(previous, scene):
            quarantine = _quarantine_receipt(receipt, token)
            return {**skipped("receipt_workspace_mismatch"), "quarantined_receipt": str(quarantine)}
        retiring = scene.parent / f"{RETIRING_PREFIX}{scene_id}-{token}"
        with storage_pin_guard(context.pins_root, exclusive=True):
            with _CaptureLocks(scene, planned_ids) as locks:
                if locks.refusal is not None:
                    return skipped(locks.refusal)
                current = _evaluate(context=context, bucket=bucket, scene_id=scene_id, scene=scene,
                                    now=observed_at, index=None, process_checker=checker)
                if current.reasons or current.capture_ids != planned_ids:
                    return skipped("receipt_workspace_active")
                if not _receipt_matches_workspace(previous, scene):
                    quarantine = _quarantine_receipt(receipt, token)
                    return {**skipped("receipt_workspace_mismatch"), "quarantined_receipt": str(quarantine)}
                try:
                    os.rename(scene, retiring)
                except OSError as exc:
                    return {**skipped("receipt_beside_live_workspace"), "removal_error": type(exc).__name__}
        if (not _valid_receipt(receipt, bucket=bucket, scene_id=scene_id)
                or not _receipt_matches_workspace(previous, retiring)):
            quarantine = _quarantine_receipt(receipt, token)
            if not os.path.lexists(scene):
                try:
                    os.rename(retiring, scene)
                except OSError:
                    pass
            return {**skipped("candidate_changed_after_rename"), "quarantined_receipt": str(quarantine)}
        shutil.rmtree(retiring, ignore_errors=True)
        removal_complete = not os.path.lexists(retiring) and not os.path.lexists(scene)
        return {**base, "status": "retired", "receipt": str(receipt),
                "removal_complete": removal_complete,
                "freed_allocated_bytes": int(current.allocated_bytes) if removal_complete else 0,
                "archive_bytes": int((previous.get("archive") or {}).get("size_bytes") or 0),
                "archive_member_count": int((previous.get("archive") or {}).get("member_count") or 0),
                "cloud_verified_count": len(previous["cloud_verified"]), "recovered_receipt": True}
    # This first proof and all large reads happen without a listener ledger lock. A
    # fresh proof under locks below decides whether the published archive is usable.
    evaluation = _evaluate(context=context, bucket=bucket, scene_id=scene_id, scene=scene, now=observed_at,
                           index=index, process_checker=checker)
    if evaluation.reasons or evaluation.capture_ids != planned_ids or _snapshot(evaluation.files) != plan["snapshot"]:
        return skipped("candidate_changed")
    for row in plan["cloud_verified"]:
        expected = CloudObject(name=row["relative_path"], size=row["size"], generation=row["generation"],
                               md5_hash=row["md5_hash"], crc32c=row["crc32c"])
        try:
            if not _verifies(expected, _hash_file(scene / row["relative_path"])):
                return skipped("candidate_changed")
        except OSError:
            return skipped("candidate_changed")
    token = secrets.token_hex(8)
    identity = {"bucket": bucket, "scene_id": scene_id, "scene": scene, "observed_at": observed_at,
                "plan": plan, "retiring_token": token}
    records = _capture_records(scene, evaluation.captures)
    estimate = len(_receipt_bytes(_receipt_document(
        **identity, records=records, archive=_largest_archive_record(plan["archive"]))))
    if estimate > RECEIPT_MAX_BYTES:
        return skipped("receipt_too_large")
    reservation = None
    if estimate > _RECEIPT_RESERVATION_THRESHOLD:
        try:
            reservation = reserve_control_plane_disk(
                "evidence_offload", target_root=receipt.parent, expected_bytes=2 * estimate,
                reservation_root=DEFAULT_RESERVATION_ROOT)
        except Exception:  # noqa: BLE001 - no room for the receipt means no retirement
            return skipped("disk_reservation_refused")
    try:
        if plan["archive"]:
            archive, refusal = _archive(scene, plan["archive"], publisher)
            if refusal is not None:
                return skipped(refusal)
        refusal = _cloud_unchanged(cloud, bucket=bucket, scene_id=scene_id, rows=plan["cloud_verified"])
        if refusal is not None:
            return skipped(refusal)
        retiring = scene.parent / f"{RETIRING_PREFIX}{scene_id}-{token}"
        with storage_pin_guard(context.pins_root, exclusive=True):
            with _CaptureLocks(scene, planned_ids) as locks:
                if locks.refusal is not None:
                    return skipped(locks.refusal)
                after = _evaluate(context=context, bucket=bucket, scene_id=scene_id, scene=scene, now=observed_at,
                                  index=None, process_checker=checker)
                if after.reasons or after.capture_ids != planned_ids or _snapshot(after.files) != plan["snapshot"]:
                    return skipped("candidate_changed_during_archive")
                records = _capture_records(scene, after.captures)
                data = _receipt_bytes(_receipt_document(**identity, records=records, archive=archive))
                if len(data) > RECEIPT_MAX_BYTES:
                    return skipped("receipt_too_large")
                if not _publish_receipt(receipt, data):
                    return {**skipped("receipt_beside_live_workspace"), "receipt": str(receipt)}
                try:
                    os.rename(scene, retiring)
                except OSError as exc:
                    return {**skipped("receipt_beside_live_workspace"), "receipt": str(receipt),
                            "removal_error": type(exc).__name__}
        state, document, _ = _load_json(receipt)
        if (state != "ok" or document is None
                or not _valid_receipt(receipt, bucket=bucket, scene_id=scene_id)
                or not _receipt_matches_workspace(document, retiring)):
            # A first delivery may create a new capture immediately before rename.
            # Return its whole tree to the listener path and preserve the receipt
            # separately for forensic recovery; never sweep an unproved copy.
            quarantine = _quarantine_receipt(receipt, token)
            if not os.path.lexists(scene):
                try:
                    os.rename(retiring, scene)
                except OSError:
                    pass
            return {**skipped("candidate_changed_after_rename"), "retiring": str(retiring),
                    "quarantined_receipt": str(quarantine)}
        shutil.rmtree(retiring, ignore_errors=True)
    finally:
        if reservation is not None:
            reservation.release()
    return {
        **base,
        "status": "retired",
        "receipt": str(receipt),
        "removal_complete": not os.path.lexists(scene) and not os.path.lexists(retiring),
        "freed_allocated_bytes": evaluation.allocated_bytes if not os.path.lexists(retiring)
                                 and not os.path.lexists(scene) else 0,
        "archive_bytes": archive["size_bytes"] if archive else 0,
        "archive_member_count": archive["member_count"] if archive else 0,
        "cloud_verified_count": len(plan["cloud_verified"]),
    }


# --- restore --------------------------------------------------------------------------------------


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError("relative path invalid")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("relative path invalid")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def restore_scene_workspace(
    *,
    receipt_path: Path,
    destination: Path,
    cloud: CloudInventory,
    materializer: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replay a retirement receipt into ``destination``, verifying every byte before exposing it.

    Cloud-verified files are downloaded and re-checked by size and MD5 (or CRC32C); the
    archive is materialized, its digest re-checked, extracted with the ``data`` filter and
    every member's SHA-256 re-checked. The union must be exactly the retired file set.
    """

    materialize = materializer or materialize_configured_scene_artifact
    invalid = WebsiteSceneWorkspaceRetentionError("website_scene_workspace_restore_receipt_invalid")
    state, receipt, _ = _load_json(Path(receipt_path))
    if (
        state != "ok"
        or receipt is None
        or receipt.get("schema_version") != RETIRED_SCHEMA
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise invalid
    try:
        bucket, scene_id = _identity(receipt.get("bucket"), receipt.get("scene_id"))
        verified = list(receipt["cloud_verified"])
        archive = receipt.get("archive")
        members = list(archive["members"]) if archive else []
        planned = [_safe_relative(row["relative_path"]) for row in (*verified, *members)]
    except (KeyError, TypeError, ValueError, WebsiteSceneWorkspaceRetentionError) as exc:
        raise invalid from exc
    if len(set(planned)) != len(planned):
        raise invalid
    target = Path(destination)
    if os.path.lexists(target):
        raise WebsiteSceneWorkspaceRetentionError("website_scene_workspace_restore_destination_exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".restore-", dir=target.parent))
    try:
        tree = staging / "tree"
        tree.mkdir()
        prefix = f"scenes/{scene_id}/"
        for row in verified:
            path = tree / row["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            generation = row.get("generation")
            if not isinstance(generation, str) or not generation.isdecimal():
                raise invalid
            try:
                cloud.download(bucket, prefix + row["relative_path"], path, generation=generation)
            except Exception as exc:  # noqa: BLE001 - a missing pinned object cannot be restored
                raise WebsiteSceneWorkspaceRetentionError("website_scene_workspace_restore_generation_unavailable") from exc
            expected = CloudObject(name=prefix + row["relative_path"], size=row["size"],
                                   generation=row.get("generation"), md5_hash=row.get("md5_hash"),
                                   crc32c=row.get("crc32c"))
            if not _verifies(expected, _hash_file(path)):
                raise WebsiteSceneWorkspaceRetentionError("website_scene_workspace_restore_cloud_mismatch")
        if archive:
            mismatch = WebsiteSceneWorkspaceRetentionError("website_scene_workspace_restore_archive_mismatch")
            bundle = staging / ARCHIVE_FILENAME
            materialize(
                reference={
                    "schema_version": "task_evaluation_scene_artifact_reference.v1",
                    "status": "remote_verified",
                    "artifact_kind": ARTIFACT_KIND,
                    "uri": archive["uri"],
                    "digest": archive["digest"],
                    "size_bytes": archive["size_bytes"],
                    # A receipt exists only after a full remote readback of this archive.
                    "remote_identity_verified": True,
                    "full_byte_service_account_readback_passed": True,
                    "raw_secret_values_recorded": False,
                },
                destination=bundle,
                maximum_size_bytes=int(archive["size_bytes"]),
            )
            if _sha256_file(bundle) != archive["digest"]:
                raise mismatch
            extracted = staging / "archive"
            extracted.mkdir()
            with tarfile.open(bundle) as source:
                source.extractall(extracted, filter="data")
            expected_members = {row["relative_path"]: row for row in members}
            observed: dict[str, Path] = {}
            for path in extracted.rglob("*"):
                if path.is_symlink():
                    raise mismatch
                if path.is_file():
                    observed[path.relative_to(extracted).as_posix()] = path
            if set(observed) != set(expected_members) or any(
                observed[name].stat().st_size != row["size_bytes"] or _sha256_file(observed[name]) != row["sha256"]
                for name, row in expected_members.items()
            ):
                raise mismatch
            for name, path in observed.items():
                placed = tree / name
                placed.parent.mkdir(parents=True, exist_ok=True)
                if os.path.lexists(placed):
                    raise invalid
                os.replace(path, placed)
        restored = sorted(path.relative_to(tree).as_posix() for path in tree.rglob("*")
                          if path.is_file() or path.is_symlink())
        if restored != sorted(planned):
            raise WebsiteSceneWorkspaceRetentionError("website_scene_workspace_restore_incomplete")
        # Downloads and the archive (which normalizes ownership to root) take the destination's owner.
        owner = os.lstat(target.parent)
        for path in (tree, *tree.rglob("*")):
            os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)
        os.replace(tree, target)
        restored_receipt = None
        if target == Path(receipt["workspace"]):
            # The restored working copy is live again. Keep its historical proof,
            # but remove the sidecar name that makes the listener call it retired.
            restored_receipt = _quarantine_receipt(Path(receipt_path), secrets.token_hex(8))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "schema_version": RESTORE_SCHEMA,
        "status": "restored",
        "bucket": bucket,
        "scene_id": scene_id,
        "destination": str(target),
        "file_count": len(planned),
        "cloud_file_count": len(verified),
        "archive_member_count": len(members),
        "source_receipt_digest": receipt["receipt_digest"],
        "historical_receipt": str(restored_receipt) if restored_receipt else str(receipt_path),
    }


# --- what the listener asks about a retired capture -----------------------------------------------


def ended_payload_digests(ledger: Mapping[str, Any]) -> set[str]:
    """Every payload digest whose run a capture ended for lost authority.

    The listener's own ``_ended_payload_digests``, copied so the reclaim timer does not
    load the pipeline; a test pins the two together.
    """

    def text(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    digests = {text(ledger.get("terminal_payload_sha256"))}
    history = ledger.get("attempt_history")
    for row in history if isinstance(history, list) else ():
        if not isinstance(row, Mapping):
            continue
        if row.get("status") == TERMINAL_AUTHORITY_STATUS:
            digests.add(text(row.get("payload_sha256")))
        elif row.get("status") == "reopened_after_terminal_authority":
            digests.add(text(row.get("terminal_payload_sha256")))
    digests.discard("")
    return digests


def ended_producer_delivery_keys(ledger: Mapping[str, Any]) -> set[str]:
    """Original semantic deliveries ended by this captured job ledger."""
    def key(value: Any) -> str:
        return value if isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) else ""

    keys = {key(ledger.get("terminal_producer_delivery_key"))}
    history = ledger.get("attempt_history")
    for row in history if isinstance(history, list) else ():
        if not isinstance(row, Mapping):
            continue
        if row.get("status") == TERMINAL_AUTHORITY_STATUS:
            keys.add(key(row.get("producer_delivery_key")))
        elif row.get("status") == "reopened_after_terminal_authority":
            keys.add(key(row.get("terminal_producer_delivery_key")))
    if ledger.get("status") == "completed":
        keys.add(key(ledger.get("producer_delivery_key")))
    keys.discard("")
    return keys


def retired_capture_status(*, storage_root: Path, bucket: str, scene_id: str,
                           capture_id: str) -> dict[str, Any] | None:
    """What a retired scene's receipt records about one capture, or None when it records nothing.

    The listener asks when the capture's workspace is absent, and answers exactly as it
    would from the capture's own ledger: a completed capture answers every payload
    (``covers_every_payload``) from its output commit; an authority ending answers the
    payloads it ended (``payload_sha256s``), and any other payload is a new request. A
    Only an absent receipt means "not retired". A present but invalid receipt
    fails closed so the listener retries and alerts instead of re-staging.
    """

    try:
        bucket, scene_id = _identity(bucket, scene_id)
        if strict_identifier(capture_id, field="capture_id") != capture_id:
            return None
    except (WebsiteSceneWorkspaceRetentionError, SecurityValidationError):
        return None
    path = receipt_path(Path(storage_root), bucket, scene_id)
    state, receipt, _ = _load_json(path)
    if state == "absent":
        return None
    if (
        state != "ok"
        or receipt is None
        or receipt.get("schema_version") != RETIRED_SCHEMA
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("bucket") != bucket
        or receipt.get("scene_id") != scene_id
        or not isinstance(receipt.get("captures"), list)
    ):
        raise WebsiteSceneWorkspaceRetentionError("retirement_receipt_invalid")
    record = next((row for row in receipt["captures"]
                   if isinstance(row, Mapping) and row.get("capture_id") == capture_id), None)
    ledger = record.get("ledger") if isinstance(record, Mapping) else None
    status = ledger.get("status") if isinstance(ledger, Mapping) else None
    disposition = TERMINAL_DISPOSITIONS.get(status) if isinstance(status, str) else None
    if disposition is None:
        if record is None:
            return None  # A valid receipt for an earlier capture does not cover a new one.
        raise WebsiteSceneWorkspaceRetentionError("retirement_receipt_invalid")
    digests: set[str] = set()
    ack = record.get("ack_receipt")
    if isinstance(ack, Mapping) and ack.get("disposition") == disposition and isinstance(ack.get("payload_sha256"), str):
        digests.add(ack["payload_sha256"])
    if status == TERMINAL_AUTHORITY_STATUS:
        digests |= ended_payload_digests(ledger)
    delivery_keys = ended_producer_delivery_keys(ledger)
    return {"status": status, "queue_disposition": disposition, "covers_every_payload": status == "completed",
            "payload_sha256s": sorted(digests), "producer_delivery_keys": sorted(delivery_keys),
            "receipt": str(path),
            "retired_at_epoch": receipt.get("retired_at_epoch")}


# --- command line ---------------------------------------------------------------------------------


def _now() -> float:
    return time.time()


def _cloud_inventory() -> CloudInventory:
    return GcsCloudInventory()


def _context_from_arguments(args: argparse.Namespace) -> RetentionContext:
    intent_root = Path(args.intent_root) if args.intent_root else None
    base = args.queue_root or [item for item in str(os.getenv(QUEUE_ROOTS_ENV) or "").split(":") if item]
    return RetentionContext(
        storage_root=Path(args.storage_root),
        pins_root=Path(args.pins_root),
        queue_roots=scene_queue_roots(base or DEFAULT_QUEUE_ROOTS, intent_root),
        intent_root=intent_root,
        binding_root=Path(args.binding_root) if args.binding_root else binding_root_for(intent_root),
    )


def _find_bucket(storage_root: Path, scene_id: str) -> str:
    """The one bucket holding this scene's workspace or retirement receipt."""

    try:
        buckets = sorted(os.listdir(storage_root))
    except OSError:
        buckets = []
    matches = [
        bucket for bucket in buckets
        if not bucket.startswith(".")
        and (os.path.lexists(scene_path(storage_root, bucket, scene_id))
             or os.path.lexists(receipt_path(storage_root, bucket, scene_id)))
    ]
    if len(matches) > 1:
        raise WebsiteSceneWorkspaceRetentionError("scene_workspace_bucket_ambiguous")
    if not matches:
        raise WebsiteSceneWorkspaceRetentionError("scene_workspace_not_found")
    return matches[0]


def _resolve_scene(args: argparse.Namespace, context: RetentionContext) -> tuple[str, str]:
    try:
        scene_id = strict_identifier(args.scene_id, field="scene_id")
    except SecurityValidationError as exc:
        raise WebsiteSceneWorkspaceRetentionError("website_scene_workspace_identity_invalid") from exc
    if scene_id != args.scene_id:
        raise WebsiteSceneWorkspaceRetentionError("website_scene_workspace_identity_invalid")
    return _identity(args.bucket or _find_bucket(context.storage_root, scene_id), scene_id)


def _retire_command(args: argparse.Namespace) -> dict[str, Any]:
    """``planned``, ``retained`` (with reasons), ``retired``; errors become ``failed`` in ``main``."""

    context = _context_from_arguments(args)
    bucket, scene_id = _resolve_scene(args, context)
    identity = {"bucket": bucket, "scene_id": scene_id}
    receipt = receipt_path(context.storage_root, bucket, scene_id)
    if not os.path.lexists(scene_path(context.storage_root, bucket, scene_id)):
        if os.path.lexists(receipt):
            if not _valid_receipt(receipt, bucket=bucket, scene_id=scene_id):
                raise WebsiteSceneWorkspaceRetentionError("retirement_receipt_invalid")
            return {"status": "retired", **identity, "already_retired": True, "receipt": str(receipt)}
        raise WebsiteSceneWorkspaceRetentionError("scene_workspace_not_found")
    now = _now()
    cloud = _cloud_inventory()
    plan = plan_scene_workspace_retirement(context=context, bucket=bucket, scene_id=scene_id, now=now, cloud=cloud)
    if plan["status"] != "retirable":
        return {"status": "retained", **identity, "reasons": plan["reasons"], "plan": plan}
    if not args.apply:
        return {"status": "planned", **identity, "plan": plan}
    outcome = apply_scene_workspace_retirement(plan, context=context, ack=args.ack, cloud=cloud, now=now)
    if outcome["status"] == "retired":
        return {**outcome, "plan_digest": plan["plan_digest"]}
    return {"status": "retained", **identity, "reasons": [outcome["reason"]], "apply": outcome, "plan": plan}


def _write_result(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _add_root_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--storage-root", default=os.getenv(STORAGE_ROOT_ENV) or str(DEFAULT_STORAGE_ROOT))
    parser.add_argument("--pins-root", default=os.getenv(PINS_ROOT_ENV) or str(DEFAULT_PINS_ROOT))
    parser.add_argument("--intent-root", default=os.getenv(INTENT_ROOT_ENV) or os.getenv(INTAKE_ROOT_ENV)
                        or str(DEFAULT_INTENT_ROOT))
    parser.add_argument("--binding-root", default=None,
                        help=f"default: ${BINDING_ROOT_ENV}, else <intent root parent>/website-source-bindings")
    parser.add_argument("--queue-root", action="append", default=None,
                        help=f"default: ${QUEUE_ROOTS_ENV}, else the reclaim timer's queues")


def main(argv: Sequence[str] | None = None) -> int:
    """``plan`` | ``retire [--apply --ack retire-scene-workspace]`` | ``restore``; JSON on stdout.

    ``retire`` reports exactly one of ``planned``, ``retained``, ``retired`` or ``failed``
    (exit 1), the statuses the operator door maps to its outcome.
    """

    parser = argparse.ArgumentParser(prog="python -m blueprint_pipeline.website_scene_workspace_retention")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "retire"):
        command = commands.add_parser(name)
        command.add_argument("--scene-id", required=True)
        command.add_argument("--bucket")
        _add_root_arguments(command)
        command.add_argument("--result-out")
        if name == "retire":
            command.add_argument("--apply", action="store_true")
            command.add_argument("--ack", default="")
    restore = commands.add_parser("restore")
    restore.add_argument("--receipt", required=True)
    restore.add_argument("--destination", required=True)
    restore.add_argument("--result-out")
    args = parser.parse_args(argv)
    try:
        if args.command == "retire":
            result = _retire_command(args)
        elif args.command == "plan":
            context = _context_from_arguments(args)
            bucket, scene_id = _resolve_scene(args, context)
            result = plan_scene_workspace_retirement(context=context, bucket=bucket, scene_id=scene_id,
                                                     now=_now(), cloud=_cloud_inventory())
        else:
            result = restore_scene_workspace(receipt_path=Path(args.receipt), destination=Path(args.destination),
                                             cloud=_cloud_inventory())
    except WebsiteSceneWorkspaceRetentionError as exc:
        result = {"status": "failed", "code": str(exc)}
    except Exception as exc:  # noqa: BLE001 - the door records a typed failure, never a traceback
        result = {"status": "failed", "code": f"website_scene_workspace_error:{type(exc).__name__}"}
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.result_out:
        _write_result(Path(args.result_out), text)
    print(text)
    return 1 if result.get("status") == "failed" else 0


__all__ = [
    "ARCHIVE_FILENAME",
    "ARTIFACT_KIND",
    "CloudInventory",
    "CloudObject",
    "DEFAULT_ACK_RETENTION_SECONDS",
    "DEFAULT_EXPIRED_GRACE_SECONDS",
    "DEFAULT_INTENT_ROOT",
    "DEFAULT_MINIMUM_IDLE_SECONDS",
    "DEFAULT_ORPHAN_REGISTRATION_SECONDS",
    "DEFAULT_QUEUE_ROOTS",
    "DEFAULT_STORAGE_ROOT",
    "DEFAULT_HASH_BUDGET_BYTES",
    "GcsCloudInventory",
    "HashBudget",
    "PLAN_SCHEMA",
    "RESTORE_SCHEMA",
    "RETIRED_SCHEMA",
    "RETIRED_SUFFIX",
    "RETIRE_ACK",
    "ReferenceIndex",
    "RetentionContext",
    "WebsiteSceneWorkspaceRetentionError",
    "apply_scene_workspace_retirement",
    "binding_root_for",
    "build_reference_index",
    "ended_payload_digests",
    "inventory_cache_path",
    "main",
    "plan_scene_workspace_retirement",
    "receipt_path",
    "restore_scene_workspace",
    "retired_capture_status",
    "scene_path",
    "scene_queue_roots",
    "scene_workspaces",
    "sweep_retiring_workspaces",
]


if __name__ == "__main__":
    raise SystemExit(main())
