"""Slim a finished website job's workspace: archive and remove its bulky Atlas intermediates, keep the rest.

Why (2026-11-19). The work volume was 97% full. A website job's working copy under
``pubsub-handoffs/<bucket>/scenes/<scene>`` was 4 to 26 GB, and 62 to 80% of it was the Atlas stage's
intermediates under each capture's ``pipeline/atlas/``: World Labs downloads and their unpacked copies, preserved
completions, pose and survey frames, splat and mesh copies. Whole-workspace retirement
(``website_scene_workspace_retention``) is the only reclaim, and it cannot serve a shared job: the company clip and
simulation routes serve only while the capture's folder exists to be checked, and the rights re-check writes
there. So a shared job either kept every byte or stopped being shared. This keeps what sharing and the site's
own view read, and moves only those intermediates to the private artifact store.

**What stays, in place.** Everything outside each capture's ``pipeline/atlas/``: the listener's records (ledger,
output commit, receipts), ``raw/``, the clip and simulation folders with their manifests and rights re-check, the
source video, the website scene preparation (base scene, levelled splats, room package, look sources) and every
other record. Inside ``pipeline/atlas/`` every record (``RECORD_SUFFIXES``: the Atlas job and its receipts), every
file under ``MINIMUM_MEMBER_BYTES``, every file with another name (a hard link), every file whose name the search
cannot prove unnamed (``name_supported``), the scene preview's own sources (``website_scene_preview``: the base
scene's splats, else the Atlas reconstruction's splat), and every file that anything staying names by path, closed
over what those name in turn (``task_evaluation_result_residue_scan``; USD layers are searched whole, binary or
not). The members are what is left: the bulky files nobody reads once the job is done.

**Readers of a finished job's folder** (2026-11-19). The company clip and simulation routes read the manifests,
the rights re-check record, the consent and takedown records and the withdrawal tombstone, and serve the bytes
from the artifact store. Their passes rebuild only on a new revision, from the source video, the compiled half
(outside the workspace) or the room package, and the look's source splat, which the base scene records name.
The site's scene preview rebuilds from the base scene's splat or the reconstruction's (kept, above). The Atlas
timer opens the job file each tick, its provider cleanup reads the folder's records until it is complete, and a
ready job may reopen by itself inside its window, reusing what it bought; after the window and the cleanup it
reads only the job file. The listener answers a redelivery from the ledger and output commit. Nothing lists a
directory for bulky files. An operator operation that re-runs a stage (``selected-handoff``) restores first.

**When.** The scene passes every check retirement makes (``website_scene_workspace_retention``: every capture a
completed website capture with its committed output and no lease, nothing changed for 48 hours, no pin, queue
message, live process or open scene intent or attempt), except two that slimming does not need
(``WAIVED_REASONS``): the acknowledgement proof (the ledger and output commit stay, so a redelivery is answered
"already processed" and never restages), and sharing (what sharing reads stays). An authority-ended capture keeps
everything (privacy: no archive of its derivatives without the owner's deletion lifecycle), and so does a
withdrawn one (withdrawal deletion owns it). Each capture's Atlas job (``pipeline/atlas/preview_job.json``) must
have ended, hold no retry or step, have finished its provider cleanup, and be past its window
(``MAXIMUM_JOB_SECONDS`` from enqueue): until then a job may reopen by itself and reuse what it bought, and nothing
bought may be missing then (money).

**How.** Every capture's Atlas preview lock is held from start to end, the lock withdrawal deletion holds from
its listing of the site's stored copies to their deletion (``_hold_withdrawal_fence``): a withdrawal either landed
before the check below and nothing is uploaded, or deletes after the archive is in the store, where it finds it.
Per capture, a withdrawn site (its tombstone) uploads nothing; otherwise the members are packed as one tar and
published to the private artifact store under a name for the capture (``archive_filename``, as the company clip is
named, so withdrawal deletion finds every version by name) with a full readback; no listener lock is held for the
upload. Then, under the storage-pin lock and the
listener's capture locks, every check is proven again and every member's identity (device, inode, size, mtime) is
checked; the receipt ``pipeline/workspace_slim.v1.json`` (the archive, each member's SHA-256) is written beside it
and only then are the members removed. ``restore`` puts each member back after checking the archive's digest and
each member's bytes (one already there with its bytes is left). A capture is slimmed once; a stop after its
receipt leaves the members not yet removed in place.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import posixpath
import re
import secrets
import shutil
import stat
import sys
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any

from . import website_scene_workspace_retention as retention
from .control_plane_evidence_offload import ControlPlaneEvidenceOffloadError, _HashingSink, _pack_stream
from .control_plane_storage_pins import storage_pin_guard
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_result_residue_scan import name_supported, named_paths, stream_tokens, suffix_index
from .website_scene_workspace_records import _load_json

PLAN_SCHEMA = "website_scene_workspace_slim_plan.v1"
RECEIPT_SCHEMA = "website_scene_workspace_slim.v1"
RESTORE_SCHEMA = "website_scene_workspace_slim_restore.v1"
SLIM_ACK = "slim-scene-workspace"
ARTIFACT_KIND = "website-scene-workspace-slim"
#: Per capture, beside the listener's own ``pipeline/`` records.
RECEIPT_PARTS = ("pipeline", "workspace_slim.v1.json")
_RECEIPT_NAMES = re.compile(r"workspace_(slim\.v1(\.restored-[0-9a-f]{16})?|dedupe\.v1)\.json")
#: The one folder of a capture whose bulky files may go: the Atlas stage's intermediates.
SLIMMED_PARTS = ("pipeline", "atlas")
MINIMUM_MEMBER_BYTES = 1024 * 1024
#: Records stay whatever their size: the Atlas job, its receipts and plans are what its readers open.
RECORD_SUFFIXES = frozenset({".json", ".jsonl", ".txt", ".md", ".yaml", ".yml", ".toml", ".csv", ".log", ".lock",
                             ".sha256", ".sig"})
#: Searched whole for the paths they name, binary or not: a USD layer can name a file by path in a binary crate.
WHOLE_SCAN_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
#: Retention's reasons that do not hold a slim: the ledger and output commit stay, and what sharing reads stays.
WAIVED_REASONS = frozenset({"acknowledgement_unproven", "shared_with_companies"})
DEFAULT_MAX_SLIMS_PER_TICK = 5
_MAX_RESULTS = 50
SLIMMING_ENV = "BLUEPRINT_CONTROL_PLANE_SCENE_WORKSPACE_SLIMMING"
SLIMMING_INVALID = "scene_workspace_slimming_setting_invalid"


class WebsiteSceneWorkspaceSlimError(RuntimeError):
    """A slim or restore was asked for something unsafe, unauthorized or unprovable."""


def archive_filename(capture_id: str, request_id: str) -> str:
    """The archive's name in the store, as the company clip names its file (``website_company_clip.clip_filename``),
    so withdrawal deletion finds every version by it."""

    return f"{capture_id}.tar" if capture_id.startswith("walkthrough-") else f"{request_id}-{capture_id}.tar"


def receipt_path(capture_root: Path) -> Path:
    return Path(capture_root).joinpath(*RECEIPT_PARTS)


# --- what may go ----------------------------------------------------------------------------------


def _atlas_reasons(capture_root: Path, capture_id: str, now: float) -> list[str]:
    """Why the capture's Atlas job may still move by itself (and need what it bought), or nothing."""

    from .atlas_preparation_job_control import STEP_IN_FLIGHT
    from .website_atlas_preview_store import JOB_FILE, MAXIMUM_JOB_SECONDS, TERMINAL

    state, job, _ = _load_json(Path(capture_root).joinpath(*SLIMMED_PARTS, JOB_FILE))
    if state == "absent":
        return []
    if state != "ok" or job is None:
        return [f"atlas_job_unreadable:{capture_id}"]
    if job.get("state") not in TERMINAL or job.get("retry_at_epoch") or STEP_IN_FLIGHT in job:
        return [f"atlas_job_open:{capture_id}"]
    cleanup = job.get("provider_cleanup")
    # A job with no cleanup record is cleaned on its next tick, from records inside the folder.
    if not isinstance(cleanup, Mapping) or cleanup.get("state") in (None, "pending"):
        return [f"atlas_provider_cleanup_pending:{capture_id}"]
    enqueued = job.get("enqueued_at_epoch")
    if (not isinstance(enqueued, (int, float)) or isinstance(enqueued, bool)
            or float(now) - float(enqueued) < MAXIMUM_JOB_SECONDS):
        return [f"atlas_job_window_open:{capture_id}"]
    return []


class _NulBlind:
    """A binary read as text: NUL separates tokens anyway, so the search reads every path a USD crate holds."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size).replace(b"\x00", b" ")


def _tokens(scene: Path, relative: str):
    try:
        descriptor = os.open(scene / relative, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return  # gone since it was listed: it names nothing
    except OSError as exc:
        raise WebsiteSceneWorkspaceSlimError("reference_document_unreadable") from exc
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise WebsiteSceneWorkspaceSlimError("reference_document_changed")
        source = _NulBlind(stream) if PurePosixPath(relative).suffix.lower() in WHOLE_SCAN_SUFFIXES else stream
        try:
            yield from stream_tokens(source)
        except OSError as exc:
            raise WebsiteSceneWorkspaceSlimError("reference_document_unreadable") from exc


def _named(scene: Path, documents: Sequence[str], files: Mapping[str, Any]) -> set[str]:
    """Every file of the scene a document names, closed over what those name in turn.

    A named directory keeps only what is named in it, as for a result run's residue: every record names the
    capture's own folder, and the readers of a finished job's Atlas folder (survey above) open named files and
    records, never a listing of bulky files.
    """

    basenames = {PurePosixPath(path).name for path in files}
    suffixes = suffix_index(files)
    named: set[str] = set()
    queue, searched = list(documents), set()
    while queue:
        document = queue.pop()
        if document in searched:
            continue
        searched.add(document)
        for tokens in _tokens(scene, document):
            for name in named_paths(tokens, document, scene.name, basenames, suffixes):
                path = posixpath.normpath(name)
                if path in files and path not in named:
                    named.add(path)
                    queue.append(path)
    return named


def _preview_sources(capture_root: Path) -> set[Path]:
    """The files the site's scene preview rebuilds from (``website_scene_preview``), asked of its own resolvers:
    the base scene's splat, and the Atlas reconstruction's splat it falls back to. (The base scene records both
    of their splats by path, so the reference search keeps them too.)"""

    from . import website_scene_preview as preview

    found: set[Path] = set()
    for source in (preview._base_scene_source(capture_root), preview._reconstruction_source(capture_root)):
        if isinstance(source, Mapping) and source.get("path") is not None:
            found.add(Path(source["path"]))
    return found


def _scene_relative(path: Path, scene: Path) -> str | None:
    for base in (scene, scene.resolve()):
        try:
            return Path(os.path.normpath(path)).relative_to(base).as_posix()
        except ValueError:
            continue
    return None


def _in_atlas(relative: str, capture_ids: Sequence[str]) -> str | None:
    """The capture whose ``pipeline/atlas/`` holds this scene-relative path, or None."""

    parts = relative.split("/")
    if len(parts) > 4 and parts[0] == "captures" and parts[1] in capture_ids and tuple(parts[2:4]) == SLIMMED_PARTS:
        return parts[1]
    return None


def _members(scene: Path, files: Sequence[tuple[str, os.stat_result]], capture_ids: Sequence[str],
             done: set[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Each not yet slimmed capture's members, and how many bulky Atlas files stay for each reason."""

    listed = {relative: info for relative, info in files}
    if not any((capture := _in_atlas(relative, capture_ids)) is not None and capture not in done
               and stat.S_ISREG(info.st_mode) and info.st_size >= MINIMUM_MEMBER_BYTES
               and PurePosixPath(relative).suffix.lower() not in RECORD_SUFFIXES for relative, info in files):
        return {}, {}  # nothing bulky left to search for: the reference search (USD layers read whole) is skipped
    documents = [relative for relative in listed
                 if _in_atlas(relative, capture_ids) is None
                 and not (relative.count("/") == 3 and relative.split("/")[2] == "pipeline"
                          and _RECEIPT_NAMES.fullmatch(relative.rsplit("/", 1)[-1]))]
    named = _named(scene, documents, listed)
    sources: set[str] = set()
    for capture_id in capture_ids:
        for path in _preview_sources(scene / "captures" / capture_id):
            relative = _scene_relative(path, scene)
            if relative is not None:
                sources.add(relative)
    members: dict[str, list[dict[str, Any]]] = {}
    kept: dict[str, int] = {}
    for relative, info in files:
        capture_id = _in_atlas(relative, capture_ids)
        if capture_id is None or capture_id in done or not stat.S_ISREG(info.st_mode):
            continue
        if info.st_size < MINIMUM_MEMBER_BYTES or PurePosixPath(relative).suffix.lower() in RECORD_SUFFIXES:
            continue
        reason = ("hard_linked" if info.st_nlink != 1 else "name_unsupported" if not name_supported(relative)
                  else "preview_source" if relative in sources else "named_by_kept_record" if relative in named
                  else None)
        if reason is not None:
            kept[reason] = kept.get(reason, 0) + 1
            continue
        members.setdefault(capture_id, []).append({
            "relative_path": relative.split("/", 2)[2], "size_bytes": int(info.st_size),
            "identity": [int(info.st_dev), int(info.st_ino), int(info.st_mtime_ns)]})
    return members, kept


def _scene_reasons(*, context: retention.RetentionContext, bucket: str, scene_id: str, scene: Path, now: float,
                   index: Any, process_checker: Callable[[Path], bool]) -> tuple[list[str], Any]:
    """Retention's checks 1-8 without the two slimming waives, plus each capture's own Atlas and withdrawal."""

    from .website_atlas_preview_store import capture_withdrawn

    evaluation = retention._evaluate(context=context, bucket=bucket, scene_id=scene_id, scene=scene, now=now,
                                     index=index, process_checker=process_checker)
    reasons = [reason for reason in evaluation.reasons if reason.split(":", 1)[0] not in WAIVED_REASONS]
    if not reasons:
        for capture_id in evaluation.capture_ids:
            root = scene / "captures" / capture_id
            if capture_withdrawn(root):
                reasons.append(f"capture_withdrawn:{capture_id}")
            reasons.extend(_atlas_reasons(root, capture_id, now))
    return sorted(set(reasons)), evaluation


# --- plan -----------------------------------------------------------------------------------------


def plan_scene_workspace_slim(*, context: retention.RetentionContext, bucket: str, scene_id: str, now: float,
                              index: Any = None, process_checker: Callable[[Path], bool] | None = None
                              ) -> dict[str, Any]:
    """Decide, without touching the workspace, which bulky Atlas files of a finished job may go."""

    bucket, scene_id = retention._identity(bucket, scene_id)
    scene = retention.scene_path(context.storage_root, bucket, scene_id)
    observed_at = float(now)
    reasons, evaluation = _scene_reasons(context=context, bucket=bucket, scene_id=scene_id, scene=scene,
                                         now=observed_at, index=index,
                                         process_checker=process_checker or retention._process_in_use)
    captures: list[dict[str, Any]] = []
    kept: dict[str, int] = {}
    if not reasons:
        done = {capture_id for capture_id in evaluation.capture_ids
                if os.path.lexists(receipt_path(scene / "captures" / capture_id))}
        try:
            members, kept = _members(scene, evaluation.files, evaluation.capture_ids, done)
        except WebsiteSceneWorkspaceSlimError as exc:
            reasons = [str(exc)]
            members = {}
        captures = [{"capture_id": capture_id, "already_slimmed": capture_id in done,
                     "members": members.get(capture_id, []),
                     "member_bytes": sum(row["size_bytes"] for row in members.get(capture_id, []))}
                    for capture_id in evaluation.capture_ids]
    member_bytes = sum(row["member_bytes"] for row in captures)
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "status": "retained" if reasons else "slimmable" if member_bytes else "nothing_to_slim",
        "bucket": bucket, "scene_id": scene_id, "workspace": str(scene), "observed_at_epoch": observed_at,
        "reasons": reasons[:50], "captures": captures, "kept_bulky_counts": kept,
        "totals": {"member_bytes": member_bytes, "workspace_allocated_bytes": int(evaluation.allocated_bytes)},
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


# --- apply ----------------------------------------------------------------------------------------


def _authorized(plan: Any, ack: str, context: retention.RetentionContext) -> tuple[str, str, Path]:
    refused = WebsiteSceneWorkspaceSlimError("website_scene_workspace_slim_not_authorized")
    if (ack != SLIM_ACK or not isinstance(plan, Mapping) or plan.get("schema_version") != PLAN_SCHEMA
            or plan.get("status") != "slimmable" or plan.get("reasons")
            or plan.get("plan_digest") != canonical_digest(dict(plan), digest_field="plan_digest")):
        raise refused
    try:
        bucket, scene_id = retention._identity(plan.get("bucket"), plan.get("scene_id"))
        scene = retention.scene_path(context.storage_root, bucket, scene_id)
        # Only files inside a capture's Atlas folder can ever be members, whatever a plan says.
        inside = all(tuple(parts[:2]) == SLIMMED_PARTS and len(parts) > 2
                     for capture in plan["captures"] for row in capture["members"]
                     for parts in [retention._safe_relative(row["relative_path"]).split("/")])
    except (retention.WebsiteSceneWorkspaceRetentionError, KeyError, TypeError, ValueError) as exc:
        raise refused from exc
    if plan.get("workspace") != str(scene) or not inside:
        raise refused
    return bucket, scene_id, scene


def _pack(capture_root: Path, rows: Sequence[Mapping[str, Any]], stream: Any) -> list[dict[str, Any]]:
    return _pack_stream(capture_root, stream, members=[row["relative_path"] for row in rows],
                        identities={row["relative_path"]: row["identity"][:2] for row in rows})


def _archive(capture_root: Path, rows: Sequence[Mapping[str, Any]], *, filename: str,
             publisher: Callable[..., Mapping[str, Any]],
             artifact_kind: str = ARTIFACT_KIND) -> tuple[dict[str, Any] | None, str | None]:
    """Stream exactly the planned members to the artifact store and prove the remote bytes."""

    sink = _HashingSink()
    try:
        packed = _pack(capture_root, rows, sink)
    except (OSError, ControlPlaneEvidenceOffloadError):
        return None, "candidate_changed"
    if [(row["relative_path"], row["size_bytes"]) for row in packed] != [
            (row["relative_path"], row["size_bytes"]) for row in rows]:
        return None, "candidate_changed"
    digest, size = "sha256:" + sink.digest.hexdigest(), sink.size
    try:
        reference = dict(publisher(write_stream=lambda stream: _pack(capture_root, rows, stream), digest=digest,
                                   size_bytes=size, filename=filename, artifact_kind=artifact_kind))
    except Exception:  # noqa: BLE001 - any publication failure keeps every local byte
        return None, "archive_readback_failed"
    if (reference.get("digest") != digest or reference.get("size_bytes") != size
            or reference.get("full_byte_service_account_readback_passed") is not True
            or not isinstance(reference.get("uri"), str)):
        return None, "archive_readback_failed"
    return {"uri": reference["uri"], "digest": digest, "size_bytes": size, "filename": filename,
            "member_count": len(packed), "members": packed}, None


def _unchanged(capture_root: Path, row: Mapping[str, Any]) -> bool:
    try:
        info = os.lstat(capture_root / row["relative_path"])
    except OSError:
        return False
    return (stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size == row["size_bytes"]
            and [info.st_dev, info.st_ino, info.st_mtime_ns] == list(row["identity"]))


def _write_receipt(path: Path, document: Mapping[str, Any]) -> None:
    """Create the receipt exclusively, owned like the folder it lives in (the GC runs as root)."""

    data = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    owner = os.lstat(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o640)
    try:
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(descriptor, view):]
            os.fchown(descriptor, owner.st_uid, owner.st_gid)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, path)  # FileExistsError: a receipt is already there
        retention._fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _prune_empty_directories(top: Path) -> None:
    for directory, _subdirectories, _files in sorted(os.walk(top, topdown=False), key=lambda row: -len(row[0])):
        if Path(directory) != top:
            try:
                os.rmdir(directory)
            except OSError:
                pass


def _hold_withdrawal_fence(stack: ExitStack, scene: Path, capture_ids: Sequence[str]) -> str | None:
    """Take each capture's Atlas preview lock without waiting; a refusal code, or None once all are held.

    Withdrawal deletion holds these same locks (``website_capture_withdrawal_deletion.apply_withdrawal_deletion``)
    from its listing of the site's stored copies to their deletion. Holding them across the withdrawal check, the
    upload and the removal means a deletion either finishes before this check (whose tombstone it sees) or starts
    after the archive is visible in the store (its listing finds it by name): it never lists the site's archives
    while one is being uploaded. The lock file is opened, never created: one this unit made would be root's.
    """

    from .website_atlas_preview_store import LOCK_FILE

    for capture_id in sorted(capture_ids):
        path = scene / "captures" / capture_id / "pipeline" / "atlas" / LOCK_FILE
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return "atlas_preview_lock_missing"
        except OSError:
            return "atlas_preview_lock_unsafe"
        lock = stack.enter_context(os.fdopen(descriptor, "rb"))
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            return "atlas_preview_lock_unsafe"
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "atlas_preview_busy"
    return None


def _withdrawn(scene: Path, capture_root: Path) -> bool:
    """The site's withdrawal tombstone is there (valid or not: nothing is uploaded on doubt), or consent is revoked."""

    from .website_atlas_preview_store import capture_withdrawn

    return os.path.lexists(scene / "website_withdrawal" / "tombstone.json") or capture_withdrawn(capture_root)


def apply_scene_workspace_slim(plan: Mapping[str, Any], *, context: retention.RetentionContext, ack: str,
                               now: float, index: Any = None,
                               stream_publisher: Callable[..., Mapping[str, Any]] | None = None,
                               process_checker: Callable[[Path], bool] | None = None) -> dict[str, Any]:
    """Under the withdrawal fence (``_hold_withdrawal_fence``): check each capture is not withdrawn, archive its
    planned members with readback, prove everything again under the listener's locks, write the receipt and only
    then remove them."""

    bucket, scene_id, scene = _authorized(plan, ack, context)
    base = {"bucket": bucket, "scene_id": scene_id, "source_plan_digest": plan["plan_digest"]}
    participating = [capture for capture in plan["captures"] if capture["members"] and not capture["already_slimmed"]]
    with ExitStack() as fence:
        refusal = _hold_withdrawal_fence(fence, scene, [capture["capture_id"] for capture in participating])
        if refusal is not None:
            return {**base, "status": "skipped", "reason": refusal, "freed_bytes": 0, "captures": []}
        return _apply_fenced(plan, participating, context=context, scene=scene, base=base, now=float(now),
                             publisher=stream_publisher, checker=process_checker or retention._process_in_use)


def _apply_fenced(plan: Mapping[str, Any], participating: Sequence[Mapping[str, Any]], *,
                  context: retention.RetentionContext, scene: Path, base: Mapping[str, Any], now: float,
                  publisher: Callable[..., Mapping[str, Any]] | None,
                  checker: Callable[[Path], bool]) -> dict[str, Any]:
    from .company_share_permission import capture_identity
    from .task_evaluation_configured_scene_object_store import publish_configured_scene_stream

    publisher = publisher or publish_configured_scene_stream
    bucket, scene_id = base["bucket"], base["scene_id"]
    archives: dict[str, dict[str, Any]] = {}
    outcomes: list[dict[str, Any]] = []
    for capture in participating:
        capture_root = scene / "captures" / capture["capture_id"]
        if _withdrawn(scene, capture_root):
            # Withdrawal deletion owns the site now: nothing of it is uploaded.
            outcomes.append({"capture_id": capture["capture_id"], "status": "skipped", "reason": "capture_withdrawn"})
            continue
        try:
            identity = capture_identity(capture_root)
        except ValueError:
            outcomes.append({"capture_id": capture["capture_id"], "status": "skipped",
                             "reason": "capture_identity_invalid"})
            continue
        archive, refusal = _archive(capture_root, capture["members"], publisher=publisher,
                                    filename=archive_filename(identity["capture_id"], identity["request_id"]))
        if refusal is not None:
            outcomes.append({"capture_id": capture["capture_id"], "status": "skipped", "reason": refusal})
            continue
        archives[capture["capture_id"]] = archive
    freed = 0
    if archives:
        with storage_pin_guard(context.pins_root, exclusive=True):
            with retention._CaptureLocks(scene, [row["capture_id"] for row in plan["captures"]]) as locks:
                reasons, _evaluation = ((["capture_locks:" + locks.refusal], None) if locks.refusal is not None
                                        else _scene_reasons(context=context, bucket=bucket, scene_id=scene_id,
                                                            scene=scene, now=now, index=None,
                                                            process_checker=checker))
                for capture in participating:
                    archive = archives.get(capture["capture_id"])
                    if archive is None:
                        continue
                    published = {key: archive[key] for key in ("uri", "digest", "size_bytes")}
                    capture_root = scene / "captures" / capture["capture_id"]
                    if reasons:
                        outcomes.append({"capture_id": capture["capture_id"], "status": "skipped",
                                         "reason": "candidate_changed_during_archive", "reasons": reasons[:5],
                                         "published_archive": published})
                        continue
                    if not all(_unchanged(capture_root, row) for row in capture["members"]):
                        outcomes.append({"capture_id": capture["capture_id"], "status": "skipped",
                                         "reason": "candidate_changed", "published_archive": published})
                        continue
                    document = {"schema_version": RECEIPT_SCHEMA, **base, "capture_id": capture["capture_id"],
                                "slimmed_at_epoch": now, "slimmed_parts": list(SLIMMED_PARTS),
                                "archive": archive, "evidence_deleted": False, "receipt_digest": ""}
                    document["receipt_digest"] = canonical_digest(document, digest_field="receipt_digest")
                    try:
                        _write_receipt(receipt_path(capture_root), document)
                    except FileExistsError:
                        outcomes.append({"capture_id": capture["capture_id"], "status": "skipped",
                                         "reason": "receipt_exists", "published_archive": published})
                        continue
                    removed = 0
                    for row in capture["members"]:
                        if _unchanged(capture_root, row):
                            (capture_root / row["relative_path"]).unlink()
                            removed += row["size_bytes"]
                    freed += removed
                    outcomes.append({"capture_id": capture["capture_id"], "status": "slimmed",
                                     "receipt": str(receipt_path(capture_root)), "freed_bytes": removed,
                                     "archive_bytes": archive["size_bytes"],
                                     "member_count": archive["member_count"]})
        for capture_id in archives:
            _prune_empty_directories((scene / "captures" / capture_id).joinpath(*SLIMMED_PARTS))
    slimmed = [row for row in outcomes if row["status"] == "slimmed"]
    return {**base, "status": "slimmed" if slimmed else "skipped", "freed_bytes": freed, "captures": outcomes}


# --- restore --------------------------------------------------------------------------------------


def restore_scene_workspace_slim(*, capture_root: Path, materializer: Callable[..., Any] | None = None
                                 ) -> dict[str, Any]:
    """Put every member back from the receipt's archive, each byte checked, then set the receipt aside."""

    from .task_evaluation_configured_scene_object_store import materialize_configured_scene_artifact

    materialize = materializer or materialize_configured_scene_artifact
    root = Path(capture_root)
    path = receipt_path(root)
    invalid = WebsiteSceneWorkspaceSlimError("website_scene_workspace_slim_receipt_invalid")
    state, receipt, _ = _load_json(path)
    if (state != "ok" or receipt is None or receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
            or receipt.get("capture_id") != root.name):
        raise invalid
    try:
        archive = receipt["archive"]
        members = {retention._safe_relative(row["relative_path"]): row for row in archive["members"]}
    except (KeyError, TypeError, ValueError) as exc:
        raise invalid from exc
    if len(members) != len(archive["members"]):
        raise invalid
    mismatch = WebsiteSceneWorkspaceSlimError("website_scene_workspace_slim_archive_mismatch")
    staging = Path(tempfile.mkdtemp(prefix=".slim-restore-", dir=root / "pipeline"))
    placed = 0
    try:
        bundle = staging / "archive.tar"
        materialize(reference={
            "schema_version": "task_evaluation_scene_artifact_reference.v1", "status": "remote_verified",
            "artifact_kind": ARTIFACT_KIND, "uri": archive["uri"], "digest": archive["digest"],
            "size_bytes": archive["size_bytes"],
            # A receipt exists only after a full remote readback of this archive.
            "remote_identity_verified": True, "full_byte_service_account_readback_passed": True,
            "raw_secret_values_recorded": False,
        }, destination=bundle, maximum_size_bytes=int(archive["size_bytes"]))
        if retention._sha256_file(bundle) != archive["digest"]:
            raise mismatch
        extracted = staging / "tree"
        extracted.mkdir()
        with tarfile.open(bundle) as source:
            source.extractall(extracted, filter="data")
        observed = {item.relative_to(extracted).as_posix(): item for item in extracted.rglob("*")
                    if item.is_file() and not item.is_symlink()}
        if set(observed) != set(members) or any(
                observed[name].stat().st_size != row["size_bytes"]
                or retention._sha256_file(observed[name]) != row["sha256"] for name, row in members.items()):
            raise mismatch
        owner = os.lstat(root / "pipeline")
        for name, row in sorted(members.items()):
            target = root / name
            if os.path.lexists(target):
                if target.is_symlink() or not target.is_file() or retention._sha256_file(target) != row["sha256"]:
                    raise WebsiteSceneWorkspaceSlimError("website_scene_workspace_slim_restore_conflict")
                continue
            parent = target.parent
            missing = [directory for directory in (parent, *parent.parents) if not directory.exists()
                       and root in directory.parents]
            parent.mkdir(parents=True, exist_ok=True)
            for directory in missing:
                os.chown(directory, owner.st_uid, owner.st_gid, follow_symlinks=False)
            os.replace(observed[name], target)
            os.chown(target, owner.st_uid, owner.st_gid, follow_symlinks=False)
            placed += 1
        historical = path.with_name(f"workspace_slim.v1.restored-{secrets.token_hex(8)}.json")
        os.replace(path, historical)
        retention._fsync_directory(path.parent)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"schema_version": RESTORE_SCHEMA, "status": "restored", "capture_root": str(root),
            "member_count": len(members), "placed_count": placed, "source_receipt_digest": receipt["receipt_digest"],
            "historical_receipt": str(historical)}


# --- the reclaim timer's phase --------------------------------------------------------------------


def slimming_setting(environ: Mapping[str, str] = os.environ) -> tuple[bool, str | None]:
    """On by default, like every storage GC switch (owner decision 2026-09-30); ``0``/``false``/``no`` in the
    operator environment file keeps it planning. Any other value keeps it planning and is reported as an alert."""

    from .control_plane_replay_cache_gc import _truthy_setting
    return _truthy_setting(environ, SLIMMING_ENV, SLIMMING_INVALID, default=True)


#: The owner approved on 2026-11-19 (relayed by the coordinating session) "remove duplicate copies inside each job's
#: scene preparation folder". A finished job's ``pipeline/website_scene_preparation/`` keeps byte-identical copies
#: (a room package's half beside the appearance and collision it was assembled from, a renewal beside the
#: preparation it renewed): about 0.4 to 0.55 GB a job on 2026-11-19. Each extra copy becomes a hard link to the
#: first, so every path still reads the same bytes and nothing needs archiving.
PREPARATION_PARTS = ("pipeline", "website_scene_preparation")
DEDUPE_RECEIPT_PARTS = ("pipeline", "workspace_dedupe.v1.json")
DEDUPE_SCHEMA = "website_scene_workspace_dedupe.v1"


def _duplicate_groups(capture_root: Path) -> list[list[tuple[str, os.stat_result, str]]]:
    """Byte-identical copies under the capture's scene preparation folder, each group sorted by path: regular
    files of at least ``MINIMUM_MEMBER_BYTES`` with one name, the same owner, group and mode. Only same-size files
    are read."""

    import hashlib

    top = capture_root.joinpath(*PREPARATION_PARTS)
    by_size: dict[int, list[tuple[str, os.stat_result]]] = {}
    for directory, directories, names in os.walk(top):
        directories[:] = sorted(name for name in directories if not (Path(directory) / name).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            info = os.lstat(path)
            if stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size >= MINIMUM_MEMBER_BYTES:
                by_size.setdefault(info.st_size, []).append((path.relative_to(capture_root).as_posix(), info))
    groups: dict[tuple[Any, ...], list[tuple[str, os.stat_result, str]]] = {}
    for rows in by_size.values():
        if len(rows) < 2:
            continue
        for relative, info in rows:
            digest = hashlib.sha256()
            with (capture_root / relative).open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(chunk)
            key = (digest.hexdigest(), info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode))
            groups.setdefault(key, []).append((relative, info, "sha256:" + digest.hexdigest()))
    return [sorted(rows, key=lambda row: row[0]) for rows in groups.values() if len(rows) > 1]


def _same_file(path: Path, info: os.stat_result, *, sole_name: bool) -> bool:
    """Still the file that was read: device, inode, size and mtime; ``sole_name``, also still its only name (the
    kept file gains a name with each copy linked to it)."""
    try:
        now = os.lstat(path)
    except OSError:
        return False
    return (stat.S_ISREG(now.st_mode) and (not sole_name or now.st_nlink == 1)
            and (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns)
            == (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))


def dedupe_scene_preparation(*, context: retention.RetentionContext, bucket: str, scene_id: str, now: float,
                             process_checker: Callable[[Path], bool] | None = None) -> dict[str, Any]:
    """Link each finished capture's duplicate scene preparation copies to the first, once per capture.

    Under the storage-pin lock and the listener's capture locks the scene must pass every slimming check again
    (``_scene_reasons``: done, idle, unpinned, unqueued, not withdrawn, its Atlas job ended). Each copy is checked
    unchanged (device, inode, size, mtime) right before it is replaced, atomically, by a link to the kept file.
    ``pipeline/workspace_dedupe.v1.json`` records what was linked (an empty list too), so a capture is read once.
    The copies share one inode afterwards: a writer that rewrote one in place would change both, so this runs only
    for a finished job, whose writers replace files rather than rewrite them.
    """

    bucket, scene_id = retention._identity(bucket, scene_id)
    scene = retention.scene_path(context.storage_root, bucket, scene_id)
    captures = retention._capture_ids(scene) or []
    outcome: dict[str, Any] = {"status": "skipped", "freed_bytes": 0, "linked_count": 0}
    pending = [capture_id for capture_id in captures
               if not os.path.lexists((scene / "captures" / capture_id).joinpath(*DEDUPE_RECEIPT_PARTS))]
    if not pending:
        return {**outcome, "reason": "already_deduped"}
    with storage_pin_guard(context.pins_root, exclusive=True):
        with retention._CaptureLocks(scene, captures) as locks:
            if locks.refusal is not None:
                return {**outcome, "reason": locks.refusal}
            reasons, _evaluation = _scene_reasons(context=context, bucket=bucket, scene_id=scene_id, scene=scene,
                                                  now=float(now), index=None,
                                                  process_checker=process_checker or retention._process_in_use)
            if reasons:
                return {**outcome, "reason": "scene_not_finished", "reasons": reasons[:5]}
            for capture_id in pending:
                capture_root = scene / "captures" / capture_id
                linked: list[dict[str, Any]] = []
                for group in _duplicate_groups(capture_root):
                    kept, kept_info, digest = group[0]
                    for relative, info, _digest in group[1:]:
                        path = capture_root / relative
                        if not (_same_file(capture_root / kept, kept_info, sole_name=False)
                                and _same_file(path, info, sole_name=True)):
                            continue
                        temporary = path.with_name(f".{path.name}.dedupe-{secrets.token_hex(8)}")
                        os.link(capture_root / kept, temporary)
                        os.replace(temporary, path)
                        linked.append({"path": relative, "linked_to": kept, "size_bytes": int(info.st_size),
                                       "sha256": digest})
                document = {"schema_version": DEDUPE_SCHEMA, "bucket": bucket, "scene_id": scene_id,
                            "capture_id": capture_id, "deduped_at_epoch": float(now), "linked": linked,
                            "freed_bytes": sum(row["size_bytes"] for row in linked), "receipt_digest": ""}
                document["receipt_digest"] = canonical_digest(document, digest_field="receipt_digest")
                _write_receipt(capture_root.joinpath(*DEDUPE_RECEIPT_PARTS), document)
                outcome["freed_bytes"] += document["freed_bytes"]
                outcome["linked_count"] += len(linked)
    return {**outcome, "status": "deduped"}


def slim_scene_workspaces(*, storage_roots: Sequence[str | Path], context_factory: Callable[[Path], Any],
                          apply: bool, enabled: bool, now: float,
                          stream_publisher: Callable[..., Any] | None = None,
                          process_checker: Callable[[Path], bool] | None = None,
                          max_slims: int = DEFAULT_MAX_SLIMS_PER_TICK, alert: str | None = None) -> dict[str, Any]:
    """Plan every scene workspace; slim at most ``max_slims`` scenes a tick when enabled (attempts, not successes,
    since each publishes gigabytes), and link a finished job's duplicate scene preparation copies
    (``dedupe_scene_preparation``). One scene's failure is recorded on its row and never stops the others."""

    applying = bool(apply and enabled)
    report: dict[str, Any] = {"status": "applied" if applying else "dry_run", "enabled": bool(enabled),
                              "candidate_count": 0, "candidate_bytes": 0, "attempted_count": 0, "slimmed_count": 0,
                              "removed_bytes": 0, "archive_bytes": 0, "retained_counts": {}, "kept_bulky_counts": {},
                              "published_archives": [], "deduped_count": 0, "dedupe_freed_bytes": 0}
    rows: list[dict[str, Any]] = []

    def dedupe(context: Any, row: dict[str, Any], plan: Mapping[str, Any]) -> None:
        if not applying or plan["reasons"]:
            return
        try:
            result = dedupe_scene_preparation(context=context, bucket=row["bucket"], scene_id=row["scene_id"],
                                              now=now, process_checker=process_checker)
        except Exception as exc:  # noqa: BLE001 - linking copies never costs the scene its slim
            row["dedupe"] = {"status": "error", "error": type(exc).__name__}
            return
        if result["status"] == "deduped":
            report["deduped_count"] += 1
            report["dedupe_freed_bytes"] += result["freed_bytes"]
            report["removed_bytes"] += result["freed_bytes"]
            row["dedupe"] = {key: result[key] for key in ("status", "freed_bytes", "linked_count")}

    for storage_root in storage_roots:
        context = context_factory(Path(storage_root))
        workspaces = retention.scene_workspaces(context.storage_root)
        if not workspaces:
            continue
        index = retention.build_reference_index(context, now=float(now))
        for bucket, scene_id, _path in workspaces:
            row: dict[str, Any] = {"bucket": bucket, "scene_id": scene_id}
            try:
                plan = plan_scene_workspace_slim(context=context, bucket=bucket, scene_id=scene_id, now=now,
                                                 index=index, process_checker=process_checker)
                for reason, count in plan["kept_bulky_counts"].items():
                    report["kept_bulky_counts"][reason] = report["kept_bulky_counts"].get(reason, 0) + count
                if plan["status"] != "slimmable":
                    for prefix in sorted({reason.split(":", 1)[0] for reason in plan["reasons"]}):
                        report["retained_counts"][prefix] = report["retained_counts"].get(prefix, 0) + 1
                    row.update(status=plan["status"], reasons=plan["reasons"][:5])
                    dedupe(context, row, plan)
                    rows.append(row)
                    continue
                report["candidate_count"] += 1
                report["candidate_bytes"] += plan["totals"]["member_bytes"]
                if not applying or report["attempted_count"] >= max_slims:
                    rows.append({**row, "status": "slimmable", "member_bytes": plan["totals"]["member_bytes"]})
                    continue
                report["attempted_count"] += 1
                outcome = apply_scene_workspace_slim(plan, context=context, ack=SLIM_ACK, now=now,
                                                     stream_publisher=stream_publisher,
                                                     process_checker=process_checker)
            except Exception as exc:  # noqa: BLE001 - one scene never costs the others
                rows.append({**row, "status": "error", "error": type(exc).__name__})
                continue
            for capture in outcome["captures"]:
                if capture["status"] == "slimmed":
                    report["archive_bytes"] += int(capture["archive_bytes"])
                elif "published_archive" in capture:
                    report["published_archives"].append({**row, "capture_id": capture["capture_id"],
                                                         **capture["published_archive"]})
            if outcome["status"] == "slimmed":
                report["slimmed_count"] += 1
            report["removed_bytes"] += int(outcome["freed_bytes"])
            row.update(status=outcome["status"], freed_bytes=outcome["freed_bytes"],
                       captures=[{key: value for key, value in capture.items() if key != "published_archive"}
                                 for capture in outcome["captures"]])
            dedupe(context, row, plan)
            rows.append(row)
    report["result_count"] = len(rows)
    report["error_count"] = sum(row["status"] == "error" for row in rows)
    report["results"] = rows[:_MAX_RESULTS]
    if alert:
        report["alerts"] = [alert]
    return report


def finished_job_phases(report: dict[str, Any], isolated: Callable[..., None], *, storage_roots: Sequence[Path],
                        pins_root: Path, queue_roots: Sequence[Path], intent_root: Path | None,
                        binding_root: Path | None, apply: bool, now: float, enabled: bool, alert: str | None,
                        stream_publisher: Callable[..., Any] | None = None,
                        process_checker: Callable[[Path], bool] | None = None) -> None:
    """The reclaim timer's phases for finished jobs, each isolated (``isolated`` is the tick's own): slimming
    their workspaces (with the scene preparation's duplicate copies), then archiving their scene intents'
    preparation output (``completed_scene_preparation_retention``)."""

    from . import completed_scene_preparation_retention as preparation
    from .website_scene_workspace_retention import timer_context_factory

    factory = timer_context_factory(pins_root=pins_root, queue_roots=queue_roots, intent_root=intent_root,
                                    binding_root=binding_root)
    isolated(report, "scene_workspace_slimming", lambda: slim_scene_workspaces(
        storage_roots=storage_roots, context_factory=factory, apply=apply, enabled=enabled, now=now,
        stream_publisher=stream_publisher, process_checker=process_checker, alert=alert))
    offload_enabled, offload_alert = preparation.setting()
    isolated(report, "scene_preparation_outputs", lambda: preparation.offload_finished_intent_outputs(
        roots=preparation.configured_roots(), pins_root=Path(pins_root), queue_roots=[Path(root) for root in queue_roots],
        intent_root=Path(intent_root) if intent_root else None, binding_root=Path(binding_root) if binding_root else None,
        apply=apply, enabled=offload_enabled, now=now, stream_publisher=stream_publisher,
        process_checker=process_checker, alert=offload_alert))


# --- command line ---------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    restore = commands.add_parser("restore", help="put a slimmed capture's Atlas files back from its receipt")
    restore.add_argument("--capture-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = restore_scene_workspace_slim(capture_root=args.capture_root)
    except (WebsiteSceneWorkspaceSlimError, OSError) as exc:
        print(json.dumps({"status": "failed", "reason": str(exc) or type(exc).__name__}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ARTIFACT_KIND", "MINIMUM_MEMBER_BYTES", "RECEIPT_SCHEMA", "SLIM_ACK", "SLIMMING_ENV",
           "WebsiteSceneWorkspaceSlimError", "apply_scene_workspace_slim", "archive_filename",
           "plan_scene_workspace_slim", "receipt_path", "restore_scene_workspace_slim", "slim_scene_workspaces",
           "slimming_setting"]
