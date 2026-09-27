"""Offload a sealed result run's residue: every file its registry neither delivers nor keeps.

On 2026-09-27 the first storage GC summary showed evidence offload keeping 27
result-registry runs, 12.94 GB, as ``result_registry``. Whole-run offload
(``control_plane_evidence_offload``) skips every run with
``artifacts/result_delivery/artifact_registry.json``: published downloads keep
their registry and closure metadata. Per-artifact offload
(``task_evaluation_result_artifact_store``) moves only registered ``BULK_ROLES``
payloads of at least 64 KiB behind a private reference under
``artifacts/result_delivery/remote_artifacts/``. Unregistered files (logs,
intermediates, provider zips) and small or non-bulk registered files stayed
forever. Once the eviction-lease fix let the per-artifact offload finish (3.5 GB,
no failures), what remains of those runs is this residue.

**When.** ``offload_result_residue`` acts only on a run that is sealed
(``_sealed_registry``: completed_unqualified, blocked or cancelled, with sealed
closure receipts) by the canary dispatcher (a digest-bound ``dispatch_receipt.json``
for the registry's run), whose registry is older than the hot window, whose bulk
artifacts are all remote (a dry run of ``offload_result_artifacts`` has no
candidates), that nothing protects, that has no residue pointer yet, and whose
root is a real directory. Anything else is a retained row with a typed
``retained_reason``; nothing is published or removed.

**Members** are the run's regular files, except: everything under
``artifacts/result_delivery/`` (registry, delivery, projection, Website sync,
remote references, locks); every path the registry records, bulk or not
(receipts, manifests, scores, reports and the whole artifact inventory stay);
any file named in ``TERMINAL_RECEIPT_NAMES`` or ``RETAINED_RECEIPTS``, at any
depth; the ``READER_REOPENED_NAMES`` and everything under
``READER_REOPENED_DIRECTORIES``, at any depth (``reader_reopened``); and every
file that one of those kept JSON documents names by path, closed over the JSON
documents it names in turn (``receipt_referenced``). A link, a special file, a
file on another filesystem than the run root, one newer than the registry, one
with a hard link outside the residue, or one whose name a tar member cannot
carry is a typed skip and stays. When what the kept documents name cannot be
known (a directory that cannot be listed, a kept directory or JSON document that
is a link or on another filesystem, one too large to search) the whole run stays
(``plan_failed``).

**Reader survey** (2026-09-27): who reopens files inside a sealed run.

* Only ``task-evaluation-policy-canaries`` runs carry a registry. The canary
  dispatcher's runs seal with ``dispatch_receipt.json``; operator runs have
  none, and G1 reviews have no ``delivery.json`` (``_sealed_registry`` refuses
  them). Nothing reads ``episode-interpretation-backfills`` or
  ``policy-canary-preprovider-audits`` runs, and no launch run has a registry.
* Every two minutes the launch reconciler's canary terminal index reopens
  ``dispatch_receipt.json``, the projection and Website sync under
  ``artifacts/result_delivery/`` and ``post_teardown_global_provider_zero.json``;
  it ``rglob``s ``preprovider_blocked.json`` and
  ``no_provider_allocation_blocked.json`` across the whole canary root.
* The WebApp download route (``live_pipeline_result_artifact_resolution``,
  ``resolve_task_evaluation_result_artifact``) reopens the registry, its reader
  lock, remote references and registered artifacts, and an operator run's
  ``website-operator-registration.json``.
* The existing-run continuation and ``finalize_operator_policy_canary`` reopen
  an operator run's ``existing_run_continuation/`` and
  ``operator_terminal_delivery/`` files and every file its intent records.
* After the seal the canary dispatcher reopens only ``dispatch_receipt.json``;
  its resume paths (allocator invocations, session authority, pending and
  progress records, ``status_events.jsonl``) run only before it exists.
* Official-billing re-validation reopens ``allocator_result.json`` (or
  ``allocator-result.json``) and every ``terminal_execution_evidence`` path of
  ``official_billing_reconciliation.json``; same-goal spend ledgers
  (``paid_attempt_authority.validate_same_goal_spend_reconciliation``) reopen
  and rehash every bound ``source_receipts`` path on each scene-progression
  tick. Both bind files by absolute path from the terminal result and its
  records, which is what ``receipt_referenced`` keeps. Scene-intent settlement
  records protect the whole run (``protected_settlement``).
* Rescoring, graded reports and interpretation closeout read
  ``policy_canary_terminal_result.json``, the registered artifact inventory,
  ``episode_interpretation_sources/`` and ``episode_interpretation/``.
* Terminal cache pins, the evidence manifest, disk usage and release leases
  only test existence. Stage replay, city-launch evidence, provider billing,
  terminal resource release, release retention and the audit scripts read no
  canary run; completed-prefix and completed-training reuse read launch runs,
  which have no registry.

**Mechanics.** Apply takes the per-artifact offload's exclusive run lock
(``artifacts/result_delivery/.offload.lock``) without waiting, reserves disk for
the staging the way the offloads do, and packs exactly the members with
``_pack_stream`` through the same content-addressed publisher and full readback
as ``apply_evidence_offload``. Only after a verified upload, with the registry
and the run unchanged and still unprotected, does it write the digest-bound
pointer ``<run>.residue.v1.json`` beside the run (temporary file, fsync,
``os.replace``, directory fsync). Then it unlinks each member through directory
descriptors held from the run root (``completed_replay_cache_retention``'s
``_HeldChild`` and ``_remove_group``): device, inode, links, size and mtime are
rechecked and its bytes hashed once, as the other offloads do, and a member that
changed, moved or became a link is skipped and recorded in the pointer as
``kept``. ``restore_result_residue`` streams the archive back, verifies every
member's digest and size, never overwrites a different file, and records a
receipt beside the pointer.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any

from . import completed_replay_cache_retention as held_files
from . import control_plane_evidence_offload as evidence
from .control_plane_replay_cache_gc import _truthy_setting
from .control_plane_retained_receipt import RETAINED_RECEIPTS
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_result_artifact_store import (
    _remote_path,
    _safe_path,
    _sealed_registry,
    offload_failure,
    offload_result_artifacts,
)

REPORT_SCHEMA_VERSION = "control_plane_result_residue_offload.v1"
PHASE_SCHEMA_VERSION = "control_plane_result_residue_offload_phase.v1"
POINTER_SCHEMA_VERSION = "control_plane_result_residue_pointer.v1"
RESTORE_SCHEMA_VERSION = "control_plane_result_residue_restore_receipt.v1"
POINTER_SUFFIX = evidence.RESIDUE_POINTER_SUFFIX
RESTORE_RECEIPT_SUFFIX = evidence.RESIDUE_RESTORE_SUFFIX
APPLY_ACK = "offload-sealed-result-residue"
RESIDUE_OFFLOAD_ENV = "BLUEPRINT_CONTROL_PLANE_GC_RESULT_RESIDUE_OFFLOAD"
RESIDUE_OFFLOAD_INVALID = "result_residue_offload_setting_invalid"
DEFAULT_HOT_WINDOW_SECONDS = 172800
RESULT_DELIVERY = "artifacts/result_delivery"
DISPATCH_RECEIPT = "dispatch_receipt.json"
#: File names a live reader reopens after the seal, wherever they sit in the run (see the survey).
READER_REOPENED_NAMES = frozenset({
    "policy_canary_terminal_result.json",
    "post_teardown_global_provider_zero.json",
    "official_billing_reconciliation.json",
    "allocator_result.json",
    "allocator-result.json",
    "website-operator-registration.json",
    "preprovider_blocked.json",
    "no_provider_allocation_blocked.json",
})
#: Directories a live reader reopens whole, wherever they sit in the run: nothing under one is residue.
READER_REOPENED_DIRECTORIES = frozenset({
    "operator_terminal_delivery",
    "existing_run_continuation",
    "episode_interpretation",
    "episode_interpretation_sources",
})
_KEPT_NAMES = frozenset({*evidence.TERMINAL_RECEIPT_NAMES, *RETAINED_RECEIPTS})
#: A kept JSON document larger than this cannot be searched for the files it names: the run stays.
MAX_REFERENCE_DOCUMENT_BYTES = 64 * 1024 * 1024
_MAX_LISTED = 50
_MIB = 1024 * 1024
_STRING_LITERAL = re.compile(r'"((?:[^"\\\n]|\\.){1,4096})"')


class ResultResidueOffloadError(RuntimeError):
    """A residue could not be offloaded or restored safely."""


def result_residue_offload_setting(environ: Mapping[str, str] = os.environ) -> tuple[bool, str | None]:
    """Whether residue offload may apply, and an alert when its setting is invalid.

    Its own owner decision, parsed like every other storage GC opt-in: only
    ``1``, ``true`` or ``yes`` enables it; any other value leaves it planning and
    is reported. The evidence offload opt-in never enables it.
    """

    return _truthy_setting(environ, RESIDUE_OFFLOAD_ENV, RESIDUE_OFFLOAD_INVALID)


def _new_row(name: str, *, apply: bool, now: Callable[[], float]) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run": name,
        "status": "applied" if apply else "dry_run",
        "retained_reason": None,
        "registry_digest": None,
        # None until the run's members were listed.
        "candidate_count": None,
        "candidate_bytes": None,
        "offloaded_count": 0,
        "offloaded_bytes": 0,
        "skipped": [],
        "skipped_by_reason": {},
        "omitted_skipped_count": 0,
        "observed_at_epoch": float(now()),
    }


def _finished(row: dict[str, Any]) -> dict[str, Any]:
    row["result_digest"] = canonical_digest(row, digest_field="result_digest")
    return row


def _retained(row: dict[str, Any], reason: str, *, failure: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row["status"], row["retained_reason"] = "retained", str(reason)
    if failure is not None:
        row["failure"] = dict(failure)
    return _finished(row)


def _skip(row: dict[str, Any], relative: str, reason: str, size: int) -> None:
    counted = row["skipped_by_reason"].setdefault(reason, {"count": 0, "bytes": 0})
    counted["count"] += 1
    counted["bytes"] += int(size)
    if len(row["skipped"]) < _MAX_LISTED:
        row["skipped"].append({"relative_path": relative, "reason": reason})
    else:
        row["omitted_skipped_count"] += 1


def _registered_paths(root: Path, registry: Mapping[str, Any]) -> set[str]:
    """Every path the registry records, relative to the run, as the artifact offload resolves it."""

    paths = set()
    for record in registry["artifacts"]:
        evidence_root = Path(str(record.get("evidence_root") or ""))
        relative = (evidence_root.relative_to(root) / str(record.get("relative_path") or "")).as_posix()
        paths.add(relative)
    return paths


def _delivery_directories(root: Path) -> tuple[str, ...]:
    # Remote references live wherever the artifact offload puts them.
    remote = _remote_path(root, "residue").parent.relative_to(root).as_posix()
    return tuple(sorted({RESULT_DELIVERY, remote}))


def _under(relative: str, directories: Sequence[str]) -> bool:
    return any(relative == directory or relative.startswith(directory + "/") for directory in directories)


def _kept_directory(relative: str, delivery: Sequence[str]) -> bool:
    """Whether nothing under the directory ``relative`` can be residue."""

    return _under(relative, delivery) or not READER_REOPENED_DIRECTORIES.isdisjoint(PurePosixPath(relative).parts)


def _kept_by_design(relative: str, delivery: Sequence[str], registered: set[str]) -> str | None:
    """Why the file ``relative`` is never residue: ``kept`` (not reported) or ``reader_reopened``; else None."""

    path = PurePosixPath(relative)
    if _under(relative, delivery) or relative in registered or path.name in _KEPT_NAMES:
        return "kept"
    if path.name in READER_REOPENED_NAMES or _kept_directory(str(path.parent), ()):
        return "reader_reopened"
    return None


def _name_supported(relative: str) -> bool:
    """Whether ``_pack_stream`` can carry the name and the pointer can record it."""

    try:
        relative.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return "\\" not in relative


def _document_value(raw: bytes) -> Any:
    """A kept document's JSON value; for one that does not parse, its string literals."""

    try:
        return json.loads(raw)
    except ValueError:
        strings = []
        for literal in _STRING_LITERAL.findall(raw.decode("utf-8", errors="replace")):
            try:
                strings.append(json.loads(f'"{literal}"'))
            except ValueError:
                strings.append(literal)
        return strings


def _named_paths(value: Any, run_name: str):
    """Every path a JSON value's strings may name inside the run.

    That is what follows each ``/<run>/`` (the run's name may recur deeper in the
    path, so every occurrence counts), what follows a leading ``<run>/`` (a path
    relative to the evidence root), and a relative string as written.
    """

    marker = f"/{run_name}/"
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str):
            text = item if item.startswith("/") else "/" + item
            start = text.find(marker)
            while start != -1:
                yield text[start + len(marker):]
                start = text.find(marker, start + 1)
            if not item.startswith("/"):
                yield item


def _receipt_references(root: Path, documents: Sequence[str], candidates: Mapping[str, Any]) -> set[str]:
    """The candidates a kept JSON document names, closed over the JSON documents those name in turn.

    Billing re-validation, spend ledgers and rescoring reopen the files a sealed
    receipt binds by path (the terminal result, and the adapter result and
    manifests it names), so a file any kept document names is kept too. A document
    that cannot be read, or is too large to search, raises: its references are unknown.
    """

    referenced: set[str] = set()
    queue, seen = list(documents), set()
    while queue:
        relative = queue.pop()
        if relative in seen:
            continue
        seen.add(relative)
        try:
            descriptor = os.open(root / relative, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        except FileNotFoundError:
            continue  # gone since it was listed: it names nothing
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                continue
            raw = stream.read(MAX_REFERENCE_DOCUMENT_BYTES + 1)
        if len(raw) > MAX_REFERENCE_DOCUMENT_BYTES:
            raise ResultResidueOffloadError("result_residue_reference_document_too_large")
        for named in _named_paths(_document_value(raw), root.name):
            path = str(PurePosixPath(named))
            if path in candidates and path not in referenced:
                referenced.add(path)
                if path.endswith(".json"):
                    queue.append(path)
    return referenced


def _plan_members(root: Path, row: dict[str, Any], registered: set[str], registry_mtime_ns: int) -> list[dict]:
    """The run's residue as inode groups, each with every name it has; typed skips go on ``row``.

    The walk never follows a link and never enters another filesystem. Nothing
    kept by design is residue, nor anything a kept JSON document names. A group is
    a candidate only when all of its links are residue names, so removing them
    frees its blocks and nothing that stays shares them. A directory that cannot be
    listed, or a kept directory or JSON document that is a link or on another
    filesystem, raises: what it names is unknown, so the whole run stays.
    """

    device = os.lstat(root).st_dev
    delivery = _delivery_directories(root)
    candidates: dict[str, os.stat_result] = {}
    documents: list[str] = []
    unreadable: list[OSError] = []
    for directory, directories, files in os.walk(root, onerror=unreadable.append):
        base = Path(directory)
        relative_directory = base.relative_to(root)
        entered = []
        for name in sorted(directories):
            relative = (relative_directory / name).as_posix()
            try:
                info = os.lstat(base / name)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode) or info.st_dev != device:
                if _kept_directory(relative, delivery):
                    # A reader follows it; the walk will not, so what it names is unknown.
                    raise ResultResidueOffloadError("result_residue_kept_directory_unsearchable")
                # Never followed or entered.
                _skip(row, relative, "symlink" if stat.S_ISLNK(info.st_mode) else "cross_device",
                      info.st_size if stat.S_ISLNK(info.st_mode) else 0)
                continue
            entered.append(name)
        directories[:] = entered
        for name in sorted(files):
            relative = (relative_directory / name).as_posix()
            try:
                info = os.lstat(base / name)
            except FileNotFoundError:
                continue
            by_design = _kept_by_design(relative, delivery, registered)
            if by_design is not None:
                if name.endswith(".json"):
                    if not stat.S_ISREG(info.st_mode) or info.st_dev != device:
                        raise ResultResidueOffloadError("result_residue_kept_document_unsearchable")
                    documents.append(relative)
                if by_design == "reader_reopened":
                    _skip(row, relative, by_design, info.st_size)
                continue
            reason = None
            if stat.S_ISLNK(info.st_mode):
                reason = "symlink"
            elif not stat.S_ISREG(info.st_mode):
                reason = "special_file"
            elif info.st_dev != device:
                reason = "cross_device"
            elif info.st_mtime_ns > registry_mtime_ns:
                reason = "newer_than_registry"
            elif not _name_supported(relative):
                reason = "name_unsupported"
            if reason is not None:
                _skip(row, relative, reason, info.st_size if reason != "special_file" else 0)
                continue
            candidates[relative] = info
    if unreadable:
        # A directory that could not be listed may hold a kept document naming any candidate.
        raise ResultResidueOffloadError("result_residue_directory_unreadable")
    for relative in sorted(_receipt_references(root, documents, candidates)):
        _skip(row, relative, "receipt_referenced", candidates.pop(relative).st_size)
    groups: dict[tuple[int, int], tuple[os.stat_result, list[str]]] = {}
    for relative, info in candidates.items():
        groups.setdefault((info.st_dev, info.st_ino), (info, []))[1].append(relative)
    members = []
    for (dev, inode), (info, names) in groups.items():
        if len(names) != info.st_nlink:
            for relative in names:
                _skip(row, relative, "linked_outside_residue", info.st_size)
            continue
        members.append({
            "relative_paths": sorted(names), "dev": dev, "inode": inode, "nlink": info.st_nlink,
            "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "mode": stat.S_IMODE(info.st_mode),
        })
    return sorted(members, key=lambda group: group["relative_paths"])


def _dispatch_receipt_reason(root: Path, registry: Mapping[str, Any]) -> str | None:
    """Why the run is not a sealed policy-canary dispatch, or None when it is.

    Only the canary dispatcher seals a result run with ``dispatch_receipt.json``,
    and after it nothing reopens the dispatcher's working files. An operator run
    has a registry but no dispatch receipt, and its continuation, terminal
    delivery and download route keep reopening its files, so it stays whole.
    """

    path = root / DISPATCH_RECEIPT
    if path.is_symlink() or not path.is_file():
        return "dispatch_receipt_missing"
    try:
        if path.stat().st_size > MAX_REFERENCE_DOCUMENT_BYTES:
            return "dispatch_receipt_invalid"
        receipt = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return "dispatch_receipt_invalid"
    if (
        not isinstance(receipt, dict)
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("run_id") != registry.get("run_id")
    ):
        return "dispatch_receipt_invalid"
    return None


def _bulk_pending(root: Path, now: Callable[[], float]) -> tuple[str | None, Mapping[str, Any] | None]:
    """Why the run's registered bulk artifacts are not all remote yet, or None when they are."""

    try:
        dry = offload_result_artifacts(run_root=root, apply=False, hot_window_seconds=0, now=now)
    except Exception as exc:  # noqa: BLE001 - any failure keeps the residue
        return "bulk_check_failed", offload_failure(exc, "registry")
    if dry.get("status") != "dry_run" or dry.get("candidate_count") != 0:
        return "bulk_not_remote", None
    return None, None


def _hold_offload_lock(root: Path, stack: ExitStack) -> bool:
    """Take the per-artifact offload's exclusive run lock without waiting; False when it is held."""

    path = _safe_path(root, f"{RESULT_DELIVERY}/.offload.lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o660)
    stack.callback(os.close, descriptor)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _adopt_owner(descriptor: int, owner: os.stat_result, *, mode: int) -> None:
    """Set ``mode``, then give the file ``owner``'s user and group.

    The storage GC is root with CAP_CHOWN but not CAP_FOWNER, so everything only
    an owner may do happens first and the owner changes last.
    """

    os.fchmod(descriptor, mode)
    current = os.fstat(descriptor)
    if (current.st_uid, current.st_gid) != (owner.st_uid, owner.st_gid):
        os.fchown(descriptor, owner.st_uid, owner.st_gid)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write ``value`` beside the run: temporary file, fsync, replace, directory fsync.

    The file takes the evidence root's owner, like the whole-run pointer, so the
    service user can read what the root GC wrote.
    """

    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    temporary = f".{path.name}.{secrets.token_hex(8)}.tmp"
    try:
        owner = os.fstat(directory)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                _adopt_owner(stream.fileno(), owner, mode=0o440)
                os.fsync(stream.fileno())
            os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
            os.fsync(directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
    finally:
        os.close(directory)


def _publish(root: Path, names: list[str], members: Sequence[Mapping[str, Any]], publisher, stream_publisher):
    """Pack exactly ``names`` and publish them with full readback, as the whole-run offload does.

    Returns the packed member rows, the archive digest and size, the verified
    reference and the disk reservation, which the caller releases.
    """

    reservation = archive = None
    try:
        if publisher is None:
            # Hash the exact tar stream without an archive on disk; only the pointer needs headroom.
            sink = evidence._HashingSink()
            packed = evidence._pack_stream(root, sink, members=names)
            digest, size = "sha256:" + sink.digest.hexdigest(), sink.size
            pointer_bytes = len(json.dumps(packed, indent=2).encode()) + 65536
            reservation = evidence.reserve_control_plane_disk(
                "evidence_offload", target_root=root.parent, expected_bytes=max(_MIB, 2 * pointer_bytes),
                reservation_root=evidence.DEFAULT_RESERVATION_ROOT)
            reference = dict((stream_publisher or evidence.publish_configured_scene_stream)(
                write_stream=lambda stream: evidence._pack_stream(root, stream, members=names),
                digest=digest, size_bytes=size, filename="residue.tar", artifact_kind=evidence.ARTIFACT_KIND))
        else:
            # Explicit file publishers stage the archive beside the run, sized per inode plus headers.
            footprint = _MIB + sum(
                group["size_bytes"] + sum(8192 + 4 * len(name.encode()) for name in group["relative_paths"])
                for group in members)
            reservation = evidence.reserve_control_plane_disk(
                "evidence_offload", target_root=root.parent, expected_bytes=footprint,
                reservation_root=evidence.DEFAULT_RESERVATION_ROOT)
            descriptor, archive_name = tempfile.mkstemp(prefix=f".{root.name}.residue-", suffix=".tar",
                                                        dir=root.parent)
            os.close(descriptor)
            archive = Path(archive_name)
            with archive.open("wb") as stream:
                packed = evidence._pack_stream(root, stream, members=names)
            digest, size = evidence._sha256(archive), archive.stat().st_size
            reference = dict(publisher(path=archive, artifact_kind=evidence.ARTIFACT_KIND))
        if (
            reference.get("digest") != digest
            or reference.get("size_bytes") != size
            or reference.get("full_byte_service_account_readback_passed") is not True
            or not isinstance(reference.get("uri"), str)
        ):
            raise ResultResidueOffloadError("result_residue_publication_mismatch")
        return packed, digest, size, reference, reservation
    except BaseException:
        if reservation is not None:
            reservation.release()
        raise
    finally:
        if archive is not None:
            archive.unlink(missing_ok=True)


def _evict(root: Path, members: Sequence[Mapping[str, Any]], sha_by_name: Mapping[str, str], row) -> list[dict]:
    """Unlink every member group through held directory descriptors; the kept members, typed."""

    kept: list[dict[str, str]] = []

    def keep(group, reason):
        for relative in group["relative_paths"]:
            kept.append({"relative_path": relative, "reason": reason})
            _skip(row, relative, reason, group["size_bytes"])

    try:
        held = held_files._HeldChild(root.parent, root.name)
    except OSError as exc:
        for group in members:
            keep(group, f"root_unavailable:{type(exc).__name__}")
        return kept
    try:
        if not held.named_by(root):
            for group in members:
                keep(group, "path_changed")
            return kept
        for group in members:
            names = [Path(name) for name in group["relative_paths"]]
            remove = functools.partial(held_files._remove_group, changed="member_changed",
                                       sha256=sha_by_name[group["relative_paths"][0]])
            reason = held.item(remove, group, names)
            if reason:
                keep(group, reason)
                continue
            row["offloaded_count"] += len(names)
            row["offloaded_bytes"] += group["size_bytes"]
    finally:
        held.close()
    return kept


def offload_result_residue(
    *,
    run_root: str | Path,
    apply: bool = False,
    ack: str = "",
    hot_window_seconds: int = DEFAULT_HOT_WINDOW_SECONDS,
    protection_checker: Callable[[Path], bool | str | None] | None = None,
    publisher: Callable[..., Mapping[str, Any]] | None = None,
    stream_publisher: Callable[..., Mapping[str, Any]] | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Plan, and with ``apply`` offload, one sealed result run's residue; see the module docstring.

    Returns a digest-bound row: ``dry_run``, ``applied`` or ``retained`` with a
    typed ``retained_reason``; the members it would move or moved
    (``candidate_*``, ``offloaded_*``); and every member it left, as typed
    ``skipped`` rows with ``skipped_by_reason`` counts and bytes. A failure
    records ``failure`` (error type, errno and stage), never a message.
    """

    if apply and ack != APPLY_ACK:
        raise ResultResidueOffloadError("result_residue_offload_not_authorized")
    if not isinstance(hot_window_seconds, int) or isinstance(hot_window_seconds, bool) or hot_window_seconds < 0:
        raise ResultResidueOffloadError("result_residue_offload_window_invalid")
    unresolved = Path(run_root).expanduser()
    row = _new_row(unresolved.name, apply=apply, now=now)
    if unresolved.is_symlink() or not unresolved.is_dir():
        return _retained(row, "run_root_invalid")
    root = unresolved.resolve()
    pointer = root.parent / f"{root.name}{POINTER_SUFFIX}"
    if pointer.exists() or pointer.is_symlink():
        return _retained(row, "already_offloaded")
    try:
        registry, registry_path, registry_bytes = _sealed_registry(root)
    except Exception as exc:  # noqa: BLE001 - an unsealed or unreadable run keeps its residue
        return _retained(row, "registry_unsealed", failure=offload_failure(exc, "registry"))
    row["registry_digest"] = registry["registry_digest"]
    receipt_reason = _dispatch_receipt_reason(root, registry)
    if receipt_reason:
        return _retained(row, receipt_reason)
    with ExitStack() as stack:
        if apply and not _hold_offload_lock(root, stack):
            return _retained(row, "offload_locked")
        registry_stat = registry_path.stat()
        if float(now()) - registry_stat.st_mtime < hot_window_seconds:
            return _retained(row, "hot")
        pending, failure = _bulk_pending(root, now)
        if pending:
            return _retained(row, pending, failure=failure)
        verdict = protection_checker(root) if protection_checker is not None else None
        if verdict:
            return _retained(row, verdict if isinstance(verdict, str) else "protected")
        try:
            members = _plan_members(root, row, _registered_paths(root, registry), registry_stat.st_mtime_ns)
        except Exception as exc:  # noqa: BLE001 - what a run's receipts name must be known
            return _retained(row, "plan_failed", failure=offload_failure(exc, "plan"))
        row["candidate_count"] = sum(len(group["relative_paths"]) for group in members)
        row["candidate_bytes"] = sum(group["size_bytes"] for group in members)
        if not apply or not members:
            return _finished(row)
        return _apply(row, root, pointer, registry, registry_path, registry_bytes, members,
                      protection_checker, publisher, stream_publisher, now)


def _apply(row, root, pointer, registry, registry_path, registry_bytes, members, protection_checker,
           publisher, stream_publisher, now) -> dict[str, Any]:
    identity = os.lstat(root)
    names = [name for group in members for name in group["relative_paths"]]
    modes = {name: group["mode"] for group in members for name in group["relative_paths"]}
    try:
        packed, digest, size, reference, reservation = _publish(root, names, members, publisher, stream_publisher)
    except Exception as exc:  # noqa: BLE001 - nothing was evicted
        return _retained(row, "publication_failed", failure=offload_failure(exc, "publish"))
    try:
        row["archive"] = {"uri": reference["uri"], "sha256": digest, "size_bytes": size}
        current = os.lstat(root)
        if (
            registry_path.read_bytes() != registry_bytes
            or (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino)
            or pointer.exists()
            or pointer.is_symlink()
            or (protection_checker is not None and protection_checker(root))
        ):
            return _retained(row, "run_changed_or_active")
        value: dict[str, Any] = {
            "schema_version": POINTER_SCHEMA_VERSION,
            "status": "offloaded",
            "run": root.name,
            "run_id": registry.get("run_id"),
            "registry_digest": registry["registry_digest"],
            "archive": {**row["archive"], "artifact_kind": evidence.ARTIFACT_KIND},
            "members": [{**member, "mode": modes[member["relative_path"]]} for member in packed],
            "kept": [],
            "offloaded_at_epoch": float(now()),
            "evidence_deleted": False,
            "pointer_digest": "",
        }
        value["pointer_digest"] = canonical_digest(value, digest_field="pointer_digest")
        try:
            _write_json(pointer, value)
        except Exception as exc:  # noqa: BLE001 - without a pointer nothing is evicted
            return _retained(row, "pointer_failed", failure=offload_failure(exc, "pointer"))
        row["pointer"] = pointer.name
        kept = _evict(root, members, {member["relative_path"]: member["sha256"] for member in packed}, row)
        if kept:
            value["kept"] = kept
            value["pointer_digest"] = canonical_digest(value, digest_field="pointer_digest")
            try:
                _write_json(pointer, value)
            except Exception as exc:  # noqa: BLE001 - kept members are still local; restore finds them
                row["failure"] = offload_failure(exc, "pointer")
        return _finished(row)
    finally:
        reservation.release()


def _read_pointer(root: Path) -> dict[str, Any]:
    path = root.parent / f"{root.name}{POINTER_SUFFIX}"
    if path.is_symlink() or not path.is_file():
        raise ResultResidueOffloadError("result_residue_restore_pointer_invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != POINTER_SCHEMA_VERSION
        or value.get("pointer_digest") != canonical_digest(value, digest_field="pointer_digest")
        or value.get("run") != root.name
        or not isinstance(value.get("archive"), dict)
        or not isinstance(value.get("members"), list)
        or not isinstance(value.get("kept"), list)
    ):
        raise ResultResidueOffloadError("result_residue_restore_pointer_invalid")
    for member in value["members"]:
        parts = PurePosixPath(str(member.get("relative_path"))).parts
        if (
            not parts
            or PurePosixPath(member["relative_path"]).is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
            or not _name_supported(member["relative_path"])
        ):
            raise ResultResidueOffloadError("result_residue_restore_pointer_invalid")
    return value


def _same_file(directory: int, name: str, entry: os.stat_result, member: Mapping[str, Any]) -> bool:
    if not stat.S_ISREG(entry.st_mode) or entry.st_size != member["size_bytes"]:
        return False
    try:
        return held_files._held_sha(directory, name, entry) == member["sha256"]
    except held_files._CrossDevice:
        return False


def _restore_member(root_fd: int, archive: tarfile.TarFile, info: tarfile.TarInfo, member, owner) -> str:
    """Put one member back through descriptors from the run root: ``restored``, ``already_present`` or ``conflict``."""

    parts = PurePosixPath(member["relative_path"]).parts
    opened: list[int] = []
    directory = root_fd
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o750, dir_fd=directory)
                created = True
            except FileExistsError:
                created = False
            # O_NOFOLLOW: a directory swapped for a link is never entered.
            directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
            opened.append(directory)
            if created:
                _adopt_owner(directory, owner, mode=0o750)
        name = parts[-1]
        try:
            entry = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            entry = None
        if entry is not None:
            return "already_present" if _same_file(directory, name, entry, member) else "conflict"
        temporary = f".{name}.residue-restore-{secrets.token_hex(6)}"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=directory)
        try:
            digest, size = hashlib.sha256(), 0
            with os.fdopen(descriptor, "wb") as stream:
                source = archive.extractfile(info)
                if source is None:
                    raise ResultResidueOffloadError("result_residue_restore_archive_invalid")
                for chunk in iter(lambda: source.read(_MIB), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    stream.write(chunk)
                stream.flush()
                if ("sha256:" + digest.hexdigest(), size) != (member["sha256"], member["size_bytes"]):
                    raise ResultResidueOffloadError("result_residue_restore_member_mismatch")
                os.utime(stream.fileno(), (info.mtime, info.mtime))
                _adopt_owner(stream.fileno(), owner, mode=int(member["mode"]) & 0o7777)
                os.fsync(stream.fileno())
            try:
                # A link never replaces an entry that appeared meanwhile.
                os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
            except FileExistsError:
                entry = os.stat(name, dir_fd=directory, follow_symlinks=False)
                return "already_present" if _same_file(directory, name, entry, member) else "conflict"
            return "restored"
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def restore_result_residue(
    *,
    run_root: str | Path,
    materializer: Callable[..., Any] | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Bring a run's offloaded residue back, verifying every member; never overwrite a different file.

    Members the pointer lists as ``kept`` never left and are not touched. A
    member already in place with the same bytes is ``already_present``; a
    different file at its path is a ``conflict`` and stays. The receipt is
    written beside the pointer as ``<run>.residue-restore.v1.json`` and returned.
    """

    unresolved = Path(run_root).expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise ResultResidueOffloadError("result_residue_restore_run_invalid")
    root = unresolved.resolve()
    pointer = _read_pointer(root)
    registry, _registry_path, _registry_bytes = _sealed_registry(root)
    if registry["registry_digest"] != pointer["registry_digest"]:
        raise ResultResidueOffloadError("result_residue_restore_registry_changed")
    archive_row = pointer["archive"]
    kept = {str(row.get("relative_path")) for row in pointer["kept"] if isinstance(row, Mapping)}
    expected = {member["relative_path"]: member for member in pointer["members"] if member["relative_path"] not in kept}
    reservation = evidence.reserve_control_plane_disk(
        "evidence_offload", target_root=root.parent,
        expected_bytes=_MIB + int(archive_row["size_bytes"]) + sum(int(m["size_bytes"]) for m in expected.values()),
        reservation_root=evidence.DEFAULT_RESERVATION_ROOT)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.residue-restore-", dir=root.parent))
    outcomes: dict[str, list[str]] = {"restored": [], "already_present": [], "conflict": []}
    try:
        archive_path = staging / "residue.tar"
        (materializer or evidence.materialize_configured_scene_artifact)(
            reference={
                "schema_version": "task_evaluation_scene_artifact_reference.v1",
                "status": "remote_verified",
                "artifact_kind": archive_row.get("artifact_kind", evidence.ARTIFACT_KIND),
                "uri": archive_row["uri"],
                "digest": archive_row["sha256"],
                "size_bytes": archive_row["size_bytes"],
                # A pointer is written only after a full remote readback.
                "remote_identity_verified": True,
                "full_byte_service_account_readback_passed": True,
                "raw_secret_values_recorded": False,
            },
            destination=archive_path,
            maximum_size_bytes=int(archive_row["size_bytes"]),
        )
        if (evidence._sha256(archive_path), archive_path.stat().st_size) != (
                archive_row["sha256"], archive_row["size_bytes"]):
            raise ResultResidueOffloadError("result_residue_restore_digest_mismatch")
        owner = os.stat(root)
        with tarfile.open(archive_path, mode="r:") as archive:
            infos: dict[str, tarfile.TarInfo] = {}
            for info in archive.getmembers():
                if info.name in infos or not (info.isreg() or info.islnk()):
                    raise ResultResidueOffloadError("result_residue_restore_archive_invalid")
                infos[info.name] = info
            if set(infos) != {member["relative_path"] for member in pointer["members"]}:
                raise ResultResidueOffloadError("result_residue_restore_archive_invalid")
            root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                for relative in sorted(expected):
                    outcome = _restore_member(root_fd, archive, infos[relative], expected[relative], owner)
                    outcomes[outcome].append(relative)
            finally:
                os.close(root_fd)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        reservation.release()
    receipt: dict[str, Any] = {
        "schema_version": RESTORE_SCHEMA_VERSION,
        "status": "restored_with_conflicts" if outcomes["conflict"] else "restored",
        "run": root.name,
        "pointer_digest": pointer["pointer_digest"],
        "archive_sha256": archive_row["sha256"],
        "restored_count": len(outcomes["restored"]),
        "restored_bytes": sum(int(expected[relative]["size_bytes"]) for relative in outcomes["restored"]),
        "already_present_count": len(outcomes["already_present"]),
        "kept_in_place_count": len(kept),
        "conflicts": [{"relative_path": relative, "reason": "existing_file_differs"}
                      for relative in outcomes["conflict"]],
        "restored_at_epoch": float(now()),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write_json(root.parent / f"{root.name}{RESTORE_RECEIPT_SUFFIX}", receipt)
    return receipt


def residue_row(
    run_root: str | Path,
    bulk_result: Mapping[str, Any],
    *,
    apply: bool,
    hot_window_seconds: int,
    protection_checker: Callable[[Path], bool | str | None] | None,
    publisher: Callable[..., Mapping[str, Any]] | None,
    now: Callable[[], float],
) -> dict[str, Any]:
    """The storage GC's residue row for one registry run, given its per-artifact offload result.

    Only a run whose bulk offload shows nothing left to move is handed to
    ``offload_result_residue``. A bulk run kept hot or protected keeps its
    residue for the same reason; one whose bulk offload failed or still has
    candidates is ``bulk_offload_failed`` or ``bulk_not_remote``. An exception is
    an ``error`` row with its type, errno and stage only.
    """

    name = Path(run_root).name
    status = bulk_result.get("status")
    skipped = [skip for skip in bulk_result.get("skipped") or () if not (
        isinstance(skip, Mapping) and skip.get("reason") == "already_evicted")]
    if status == "retained_hot_or_active":
        reason = bulk_result.get("retained_reason")
        return _retained(_new_row(name, apply=apply, now=now), reason if isinstance(reason, str) else "protected")
    if status not in ("dry_run", "applied"):
        return _retained(_new_row(name, apply=apply, now=now), "bulk_offload_failed")
    if skipped or (status == "dry_run" and bulk_result.get("candidate_count") != 0):
        return _retained(_new_row(name, apply=apply, now=now), "bulk_not_remote")
    try:
        return offload_result_residue(
            run_root=run_root, apply=apply, ack=APPLY_ACK if apply else "", hot_window_seconds=hot_window_seconds,
            protection_checker=protection_checker, publisher=publisher, now=now)
    except Exception as exc:  # noqa: BLE001 - one run never costs the others
        return {"status": "error", "run": name, **offload_failure(exc, "residue")}


def residue_phase(
    rows: Sequence[Mapping[str, Any]], *, enabled: bool, applying: bool, alert: str | None = None,
) -> dict[str, Any]:
    """The storage GC phase entry: what the residue offload planned, moved and kept, and why.

    ``retained_by_reason`` counts each retained run under its reason (bytes null
    unless its members were listed) and each skipped member under
    ``member_skipped:<reason>`` with its bytes, the reason without the exception
    type a run row adds to it. A run that raised is listed under ``errors`` and
    makes ``candidate_bytes`` unknown in the summary.
    """

    retained: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    def keep(reason: str, count: int, size: int | None) -> None:
        counted = retained.setdefault(reason, {"count": 0, "bytes": 0})
        counted["count"] += count
        counted["bytes"] = None if size is None or counted["bytes"] is None else counted["bytes"] + size

    for row in rows:
        if row.get("status") == "error":
            errors.append({key: row.get(key) for key in ("run", "error_type", "errno", "stage")})
            keep("residue_offload_failed", 1, None)
            continue
        if row.get("status") == "retained":
            keep(str(row.get("retained_reason")), 1, row.get("candidate_bytes"))
        for reason, counted in (row.get("skipped_by_reason") or {}).items():
            # ``recheck_failed:OSError`` counts as ``recheck_failed``: the summary copies only
            # typed lower-case reasons, and the run row keeps the exception type.
            keep(f"member_skipped:{reason.split(':', 1)[0]}", int(counted["count"]), int(counted["bytes"]))
    phase: dict[str, Any] = {
        "schema_version": PHASE_SCHEMA_VERSION,
        "enabled": bool(enabled),
        "status": "applied" if applying else "dry_run",
        "run_count": len(rows),
        "candidate_count": sum(int(row.get("candidate_count") or 0) for row in rows),
        "candidate_bytes": sum(int(row.get("candidate_bytes") or 0) for row in rows),
        "offloaded_count": sum(int(row.get("offloaded_count") or 0) for row in rows),
        "offloaded_bytes": sum(int(row.get("offloaded_bytes") or 0) for row in rows),
        "retained_by_reason": retained,
        "runs": list(rows[:_MAX_LISTED]),
        "omitted_runs_count": max(0, len(rows) - _MAX_LISTED),
        "errors": errors[:_MAX_LISTED],
        "omitted_errors_count": max(0, len(errors) - _MAX_LISTED),
    }
    if alert:
        phase["alerts"] = [alert]
    return phase


__all__ = [
    "APPLY_ACK",
    "POINTER_SCHEMA_VERSION",
    "POINTER_SUFFIX",
    "READER_REOPENED_DIRECTORIES",
    "READER_REOPENED_NAMES",
    "RESIDUE_OFFLOAD_ENV",
    "RESIDUE_OFFLOAD_INVALID",
    "RESTORE_RECEIPT_SUFFIX",
    "ResultResidueOffloadError",
    "offload_result_residue",
    "residue_phase",
    "residue_row",
    "restore_result_residue",
    "result_residue_offload_setting",
]
