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
2. every capture is terminal (a committed output, or an authority ending with
   its receipt) and holds no live lease;
3. every terminal message was acknowledged (an ack receipt written after the
   terminal state, or a ledger idle past Pub/Sub's message retention);
4. nothing in the tree changed for ``minimum_idle_seconds``;
5. no live storage pin names it, lies inside it or contains it;
6. no pending or processing queue message names the scene;
7. no live process holds it;
8. no open scene intent can still resolve a website source registered inside it;
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

import base64
import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .completed_replay_cache_retention import active_reference
from .control_plane_storage_gc import _pinned_workspace, _queue_reference_text
from .control_plane_storage_pins import live_pinned_paths
from .core.security_controls import SecurityValidationError, strict_gcs_bucket, strict_identifier
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest


PLAN_SCHEMA = "website_scene_workspace_retirement_plan.v1"
RETIRED_SCHEMA = "website_scene_workspace_retired.v1"
RESTORE_SCHEMA = "website_scene_workspace_restore_receipt.v1"
RETIRE_ACK = "retire-scene-workspace"
ARTIFACT_KIND = "website-scene-workspace"
RETIRED_SUFFIX = ".retired.v1.json"
DEFAULT_MINIMUM_IDLE_SECONDS = 48 * 3600
#: Pub/Sub message retention (deploy/terraform/main.tf): after it, a message can no longer be redelivered.
DEFAULT_ACK_RETENTION_SECONDS = 7 * 24 * 3600
#: A website sponsorship lasts at most 24 hours, so an intent claims its registration well within this.
DEFAULT_ORPHAN_REGISTRATION_SECONDS = 72 * 3600

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
_MAX_RECORD_BYTES = 16 * 1024 * 1024
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

    def download(self, bucket: str, name: str, destination: Path) -> None: ...


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

    def download(self, bucket: str, name: str, destination: Path) -> None:
        self._storage().bucket(bucket).blob(name).download_to_filename(str(destination))


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


@dataclass(frozen=True)
class ReferenceIndex:
    """Website source registrations and scene intents, read once per tick."""

    readable: bool
    binding_root: Path | None = None
    #: {"path", "request_digest", "reference_paths", "registered_at_epoch"}
    registrations: tuple[Mapping[str, Any], ...] = ()
    #: {"intent_id", "request_digest", "finished": "completed" | "revoked" | "expired" | None}
    intents: tuple[Mapping[str, Any], ...] = ()


class _Unreadable(Exception):
    pass


def _b64(digest: bytes) -> str:
    return base64.b64encode(digest).decode("ascii")


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
    return [[relative, info.st_size, info.st_mtime_ns, info.st_ino] for relative, info in files]


def _workspace_path_safe(storage_root: Path, bucket: str, scene: Path) -> bool:
    for path in (Path(storage_root), Path(storage_root) / bucket, Path(storage_root) / bucket / "scenes", scene):
        try:
            if not stat.S_ISDIR(os.lstat(path).st_mode):
                return False
        except OSError:
            return False
    return True


def scene_workspaces(storage_root: Path) -> list[tuple[str, str, Path]]:
    """``(bucket, scene_id, path)`` of every scene working copy under the spool, sorted."""

    root = Path(storage_root)
    rows: list[tuple[str, str, Path]] = []
    try:
        buckets = sorted(os.listdir(root))
    except OSError:
        return rows
    for bucket in buckets:
        scenes = root / bucket / "scenes"
        if bucket.startswith(".") or not _workspace_path_safe(root, bucket, scenes):
            continue
        try:
            strict_gcs_bucket(bucket)
            names = sorted(os.listdir(scenes))
        except (SecurityValidationError, OSError):
            continue
        for name in names:
            path = scenes / name
            if name.startswith(".") or name.endswith(RETIRED_SUFFIX):
                continue
            try:
                if strict_identifier(name, field="scene_id") != name:
                    continue
                mode = os.lstat(path).st_mode
            except (SecurityValidationError, OSError):
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


def _intent_finished(directory: Path, intent: Mapping[str, Any], *, now: float) -> str | None:
    """Why scene progression will never resolve this intent's source again, or None.

    Mirrors ``task_evaluation_scene_progression._advance_intent``: a completed
    progression, a revocation, or an elapsed (possibly extended) execution window.
    """

    from . import task_evaluation_scene_intake as intake

    projection_path = directory / "progression.json"
    if os.path.lexists(projection_path):
        projection = intake._read(projection_path, "progression_digest")
        if projection.get("intent_digest") != intent["intent_digest"] or projection.get("intent_id") != intent["intent_id"]:
            raise _Unreadable
        if projection.get("status") == "completed":
            return "completed"
    if (directory / "revoked.json").exists():
        return "revoked"
    if now >= intake.effective_execution_expiry(directory, intent):
        return "expired"
    return None


def _intents(intent_root: Path | None, *, now: float) -> tuple[Mapping[str, Any], ...]:
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
            rows.append({
                "intent_id": name,
                "request_digest": cross_runtime_canonical_digest(request),
                "finished": _intent_finished(directory, intent, now=now),
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
        intents = _intents(context.intent_root, now=float(now))
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
    walk = _walk(scene)
    reasons = [f"unsafe_entry:{relative}" for relative in walk.unsafe]
    captures: list[dict[str, Any]] = []
    capture_ids = _capture_ids(scene) or []
    if not capture_ids:
        reasons.append("scene_has_no_captures")
    for capture_id in capture_ids:
        row, capture_reasons = _capture(scene / "captures" / capture_id, scene_id=scene_id, capture_id=capture_id,
                                        now=now, ack_retention_seconds=context.ack_retention_seconds)
        captures.append(row)
        reasons.extend(capture_reasons)
    idle_seconds = now - walk.newest_mtime
    if idle_seconds < context.minimum_idle_seconds:
        reasons.append("recently_active")
    if _pinned_workspace(scene, live_pinned_paths(context.pins_root, now=lambda: now)):
        reasons.append("pinned")
    if scene_id in _queue_reference_text(context.queue_roots):
        reasons.append("queue_referenced")
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


@dataclass(frozen=True)
class _Digests:
    size: int
    sha256: str  # hex
    md5: str  # base64
    crc32c: str | None  # base64; None when google_crc32c is unavailable


def _hash_file(path: Path) -> _Digests:
    """SHA-256, MD5 and CRC32C of a regular file in one read."""

    try:
        import google_crc32c

        crc = google_crc32c.Checksum()
    except ImportError:  # pragma: no cover - google-cloud-storage depends on it
        crc = None
    sha, md5, size = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise OSError("not a regular file")
        while chunk := stream.read(1 << 20):
            sha.update(chunk)
            md5.update(chunk)
            if crc is not None:
                crc.update(chunk)
            size += len(chunk)
    return _Digests(size=size, sha256=sha.hexdigest(), md5=_b64(md5.digest()),
                    crc32c=_b64(crc.digest()) if crc is not None else None)


def _verifies(cloud: CloudObject, local: _Digests) -> bool:
    if cloud.size != local.size:
        return False
    if cloud.md5_hash:
        return cloud.md5_hash == local.md5
    return bool(cloud.crc32c) and cloud.crc32c == local.crc32c


def _inventory(*, scene: Path, bucket: str, scene_id: str, files: Sequence[tuple[str, os.stat_result]],
               cloud: CloudInventory) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    prefix = f"scenes/{scene_id}/"
    try:
        listing = dict(cloud.list_objects(bucket, prefix))
    except Exception:  # noqa: BLE001 - without a listing nothing is proven recoverable
        return [], [], ["cloud_inventory_unavailable"]
    verified: list[dict[str, Any]] = []
    archive: list[dict[str, Any]] = []
    reasons: list[str] = []
    for relative, info in files:
        try:
            digests = _hash_file(scene / relative)
        except OSError:
            reasons.append(f"unsafe_entry:{relative}")
            continue
        if digests.size != info.st_size:
            reasons.append("recently_active")  # it changed while it was read
            continue
        remote = listing.get(prefix + relative)
        if remote is not None and _verifies(remote, digests):
            verified.append({"relative_path": relative, "uri": f"gs://{bucket}/{prefix}{relative}",
                             "generation": remote.generation, "size": remote.size,
                             "md5_hash": remote.md5_hash, "crc32c": remote.crc32c})
        elif _is_raw(relative):
            # Raw capture bytes live only in Firebase Storage; they are never archived here.
            reasons.append(f"raw_not_verified_in_cloud:{relative}")
        else:
            archive.append({"relative_path": relative, "size_bytes": digests.size,
                            "sha256": _SHA256 + digests.sha256})
    return verified, archive, reasons


# --- plan -----------------------------------------------------------------------------------------


def plan_scene_workspace_retirement(
    *,
    context: RetentionContext,
    bucket: str,
    scene_id: str,
    now: float,
    cloud: CloudInventory,
    index: ReferenceIndex | None = None,
    process_checker: Callable[[Path], bool] = _process_in_use,
) -> dict[str, Any]:
    """Decide, without touching anything, whether one scene workspace may be retired."""

    bucket, scene_id = _identity(bucket, scene_id)
    observed_at = float(now)
    scene = scene_path(context.storage_root, bucket, scene_id)
    evaluation = _evaluate(context=context, bucket=bucket, scene_id=scene_id, scene=scene, now=observed_at,
                           index=index, process_checker=process_checker)
    reasons = list(evaluation.reasons)
    verified: list[dict[str, Any]] = []
    archive: list[dict[str, Any]] = []
    if not reasons:  # the cloud inventory runs only when nothing cheaper retained the scene
        verified, archive, inventory_reasons = _inventory(scene=scene, bucket=bucket, scene_id=scene_id,
                                                          files=evaluation.files, cloud=cloud)
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
        },
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


__all__ = [
    "ARTIFACT_KIND",
    "CloudInventory",
    "CloudObject",
    "DEFAULT_ACK_RETENTION_SECONDS",
    "DEFAULT_MINIMUM_IDLE_SECONDS",
    "DEFAULT_ORPHAN_REGISTRATION_SECONDS",
    "GcsCloudInventory",
    "PLAN_SCHEMA",
    "RESTORE_SCHEMA",
    "RETIRED_SCHEMA",
    "RETIRED_SUFFIX",
    "RETIRE_ACK",
    "ReferenceIndex",
    "RetentionContext",
    "WebsiteSceneWorkspaceRetentionError",
    "build_reference_index",
    "plan_scene_workspace_retirement",
    "scene_workspaces",
]
