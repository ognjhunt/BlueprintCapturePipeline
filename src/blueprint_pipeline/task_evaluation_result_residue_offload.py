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
for the registry's run) that no pending or processing queue row names, whose
registry is older than the hot window, whose bulk
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
``READER_REOPENED_DIRECTORIES``, at any depth (``reader_reopened``); and
anything a reader can reach from what stays. That is the target of every link
that stays inside the run, with everything under it (``symlink_target``), and
every file that a kept or reached text document names by path, in any format
(JSON, JSON lines or free text), absolutely, relative to the run, the evidence
root or any directory above the document; each file so reached is searched in
turn (``receipt_referenced``). The search streams each file a chunk at a time,
whatever its size, reading every run of name characters as a candidate path; a
binary (a NUL in its first 64 KiB) names nothing. A named directory keeps only
what is named in it: the registry names every evidence root, the run root among
them, and the surveyed readers reopen bound files, never a named directory's
listing. A link, a special file, a file on another filesystem than the run root,
one newer than the registry, one with a hard link outside the residue, or one
whose name holds a character the search does not read as part of a path
(``name_unsupported``) is a typed skip and stays. When what stays cannot be
searched (a directory that cannot be listed, a kept link that leaves the run, a
kept directory or file on another filesystem, a file that cannot be read) the
whole run stays (``plan_failed``).

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
  ``operator_terminal_delivery/`` files and every file its intent records,
  through record chains (setup, specs, nested file records, relocations) that
  can pass through documents outside the run. The reference closure below
  reads only documents inside the run, so an operator run, which has no
  dispatch receipt, stays whole.
* The canary dispatcher reopens ``dispatch_receipt.json`` after the seal, and
  its blocked/stranded rescan and resume paths (allocator invocations, session
  authority, pending and progress records, ``status_events.jsonl``) run only
  while the receipt is missing. In queue mode, though, it runs a pending or
  processing envelope whether or not the run is sealed: it reopens the run's
  authority and records and rebuilds the bundle when its receipt is missing
  (``allocator_result.json``, kept by name, stops any new spend). So a run that
  a pending or processing row of a configured queue names stays whole
  (``dispatch_row_pending``), and a row that cannot be read keeps every run
  (``dispatch_queue_unreadable``).
* Official-billing re-validation reopens ``allocator_result.json`` (or
  ``allocator-result.json``) and every ``terminal_execution_evidence`` path of
  ``official_billing_reconciliation.json``; same-goal spend ledgers
  (``paid_attempt_authority.validate_same_goal_spend_reconciliation``) reopen
  and rehash every bound ``source_receipts`` path on each scene-progression
  tick. The ledgers live outside the run, but every in-run file a canary entry
  can bind (terminal result, teardown manifest, provider zero, official billing
  response and source receipt, adapter result) is either kept by name or named
  by absolute path in the run's own ``official_billing_reconciliation.json``
  (``vast_official_billing_extractor`` records each one), which is searched, so
  ``receipt_referenced`` keeps it. Scene-intent settlement records protect the
  whole run (``protected_settlement``).
* Scene-attempt recovery (``task_evaluation_scene_progression_recovery``) scans
  every canary run for ``*.lease.json`` and ``pending_teardowns/*.json`` (or
  ``pending-teardowns``) and counts each open record as an ambiguous-create
  blocker, so those stay wherever they sit (``OWNERSHIP_RECORD_*``).
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
as ``apply_evidence_offload``; the packer opens each member without following a
link or blocking and requires the planned inode, so nothing else can reach the
archive. Only after a verified upload, with the registry
and the run unchanged and still unprotected, does it write the digest-bound
pointer ``<run>.residue.v1.json`` beside the run (temporary file, fsync,
``os.replace``, directory fsync). Then it unlinks each member through directory
descriptors held from the run root (``completed_replay_cache_retention``'s
``_HeldChild`` and ``_remove_group``): device, inode, links, size and mtime are
rechecked and its bytes hashed once, as the other offloads do, and a member that
changed, moved or became a link is skipped and recorded in the pointer as
``kept``; of a hard-linked group cut short, only the names still there are kept,
and a name that went without the offload (its directory moved away) is
``member_vanished``: neither offloaded nor kept, so restore brings it back.
A pointer behind which nothing could be evicted is withdrawn (``nothing_evicted``)
so the next tick tries again. A crash after the pointer is written leaves members
behind it: every later tick reports the listed members still local
(``pointed_remaining_*``), and one that applies and passes every gate under the
lock resumes (``resume``), evicting each listed member the pointer does not keep
whose bytes still hash to the pointer's, keeping the rest and rewriting the
pointer; it publishes nothing and never withdraws that pointer. A pointer that
does not verify, or whose registry digest is not the run's, leaves the run alone
(``pointer_invalid``). ``restore_result_residue``
(``task_evaluation_result_residue_restore``) streams the archive back, verifies
every member's digest and size, never overwrites a different file, and records a
receipt beside the pointer. The reference search itself lives in
``task_evaluation_result_residue_scan``.
"""

from __future__ import annotations

import bisect
import fcntl
import functools
import json
import os
import secrets
import stat
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
from .control_plane_storage_references import MAX_QUEUE_MESSAGE_BYTES, QUEUE_STATES
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_result_residue_scan import (
    ResultResidueOffloadError,
    name_supported as _name_supported,
    receipt_references as _receipt_references,
)
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
#: Like scene workspace retirement, a tick attempts at most this many publications; the rest wait.
RESIDUE_MAX_RUNS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_RESULT_RESIDUE_MAX_RUNS_PER_TICK"
DEFAULT_MAX_RUNS_PER_TICK = 5
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
#: Scene-attempt recovery (``task_evaluation_scene_progression_recovery.reconcile_ownership``)
#: scans every canary run with ``rglob("*.lease.json")`` and ``glob("**/pending_teardowns/*.json")``
#: (or ``pending-teardowns``) and counts an open record as a blocker: these stay wherever they sit.
OWNERSHIP_RECORD_SUFFIX = ".lease.json"
OWNERSHIP_RECORD_DIRECTORIES = frozenset({"pending_teardowns", "pending-teardowns"})
#: Directories a live reader reopens whole, wherever they sit in the run: nothing under one is residue.
READER_REOPENED_DIRECTORIES = frozenset({
    "operator_terminal_delivery",
    "existing_run_continuation",
    "episode_interpretation",
    "episode_interpretation_sources",
})
_KEPT_NAMES = frozenset({*evidence.TERMINAL_RECEIPT_NAMES, *RETAINED_RECEIPTS})
#: A dispatch receipt larger than this is not the dispatcher's.
_MAX_DISPATCH_RECEIPT_BYTES = 64 * 1024 * 1024
_MAX_LISTED = 50
_MIB = 1024 * 1024



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
    if (
        path.name in READER_REOPENED_NAMES
        or path.name.endswith(OWNERSHIP_RECORD_SUFFIX)
        or (path.parent.name in OWNERSHIP_RECORD_DIRECTORIES and path.name.endswith(".json"))
        or _kept_directory(str(path.parent), ())
    ):
        return "reader_reopened"
    return None


def _within(relative: str, paths: Sequence[str]) -> list[str]:
    """``relative`` and everything under it, from sorted run-relative ``paths``."""

    if relative == ".":
        return list(paths)
    index = bisect.bisect_left(paths, relative)
    found = [relative] if index < len(paths) and paths[index] == relative else []
    prefix = relative + "/"
    for index in range(bisect.bisect_left(paths, prefix), len(paths)):
        if not paths[index].startswith(prefix):
            break
        found.append(paths[index])
    return found


def _link_target(root: Path, relative: str) -> str | None:
    """Where a link inside the run leads, relative to the run, or None when it leaves the run."""

    target = Path(os.path.realpath(root / relative))
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return None


def _plan_members(root: Path, row: dict[str, Any], registered: set[str], registry_mtime_ns: int) -> list[dict]:
    """The run's residue as inode groups, each with every name it has; typed skips go on ``row``.

    The walk never follows a link and never enters another filesystem. Nothing
    kept by design is residue; nor is anything a reader reaches from a kept file:
    the target of any link that stays inside the run, with everything under it
    (``symlink_target``), and every file a kept or reached text document names
    (``receipt_referenced``). A group is a candidate only when all of its links
    are residue names, so removing them frees its blocks and nothing that stays
    shares them. A directory that cannot be listed, a kept link that leaves the
    run, or a kept directory or file on another filesystem raises: what a reader
    reaches through it is unknown, so the whole run stays.
    """

    device = os.lstat(root).st_dev
    delivery = _delivery_directories(root)
    remote = _remote_path(root, "residue").parent.relative_to(root).as_posix()
    candidates: dict[str, os.stat_result] = {}
    files: dict[str, os.stat_result] = {}
    documents: list[str] = []
    links: list[tuple[str, bool]] = []
    unreadable: list[OSError] = []
    for directory, directories, names in os.walk(root, onerror=unreadable.append):
        base = Path(directory)
        relative_directory = base.relative_to(root)
        entered = []
        for name in sorted(directories):
            relative = (relative_directory / name).as_posix()
            try:
                info = os.lstat(base / name)
            except FileNotFoundError:
                continue
            kept = _kept_directory(relative, delivery)
            if stat.S_ISLNK(info.st_mode):
                links.append((relative, kept))
                if not kept:
                    _skip(row, relative, "symlink", info.st_size)
                continue
            if info.st_dev != device:
                if kept:
                    raise ResultResidueOffloadError("result_residue_kept_directory_unsearchable")
                _skip(row, relative, "cross_device", 0)
                continue
            entered.append(name)
        directories[:] = entered
        for name in sorted(names):
            relative = (relative_directory / name).as_posix()
            try:
                info = os.lstat(base / name)
            except FileNotFoundError:
                continue
            by_design = _kept_by_design(relative, delivery, registered)
            if stat.S_ISLNK(info.st_mode):
                links.append((relative, by_design is not None))
                if by_design != "kept":
                    _skip(row, relative, by_design or "symlink", info.st_size)
                continue
            if stat.S_ISREG(info.st_mode) and info.st_dev == device:
                files[relative] = info
            if by_design is not None:
                if stat.S_ISREG(info.st_mode):
                    if info.st_dev != device:
                        raise ResultResidueOffloadError("result_residue_kept_document_unsearchable")
                    # Remote references name only registered paths; everything else kept is searched.
                    if not _under(relative, (remote,)):
                        documents.append(relative)
                if by_design == "reader_reopened":
                    _skip(row, relative, by_design, info.st_size)
                continue
            reason = None
            if not stat.S_ISREG(info.st_mode):
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
    listed = sorted(files)
    for link, kept in links:
        target = _link_target(root, link)
        if target is None:
            if kept:
                # A reader follows it out of the run, where nothing here can search what it names.
                raise ResultResidueOffloadError("result_residue_kept_link_leaves_run")
            continue
        for path in _within(target, listed):
            documents.append(path)
            if path in candidates:
                _skip(row, path, "symlink_target", candidates.pop(path).st_size)
    for relative in sorted(_receipt_references(root, documents, files)):
        if relative in candidates:
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

    # The terminal index's own constants; that module is heavy, so it loads only when a run gets here.
    from .task_evaluation_scene_terminal_reconciler import DISPATCH_RECEIPT_SCHEMA, POLICY_CANARY_RUN_KIND

    path = root / DISPATCH_RECEIPT
    if path.is_symlink() or not path.is_file():
        return "dispatch_receipt_missing"
    try:
        if path.stat().st_size > _MAX_DISPATCH_RECEIPT_BYTES:
            return "dispatch_receipt_invalid"
        receipt = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return "dispatch_receipt_invalid"
    # As index_policy_canary_terminal checks it: schema, digest, run kind and a run id, here the registry's.
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != DISPATCH_RECEIPT_SCHEMA
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("run_kind") != POLICY_CANARY_RUN_KIND
        or not isinstance(receipt.get("run_id"), str)
        or not receipt["run_id"]
        or receipt["run_id"] != registry.get("run_id")
    ):
        return "dispatch_receipt_invalid"
    return None


def _queue_row_reason(run_name: str, queue_roots: Sequence[str | Path]) -> str | None:
    """``dispatch_row_pending`` when a pending or processing queue row names the run, else None.

    In queue mode the canary dispatcher runs a pending or processing envelope
    whether or not its run is sealed, reopening the run's authority, bundle and
    allocator records, so such a row keeps the whole run. The read is strict: a
    row or state directory that is a link, not a regular file or directory, too
    large or unreadable might name any run (``dispatch_queue_unreadable``).
    """

    needle = run_name.encode("utf-8")
    for raw_root in queue_roots:
        root = Path(raw_root).expanduser()
        for state in QUEUE_STATES:
            directory = root / state
            if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
                return "dispatch_queue_unreadable"
            if not directory.is_dir():
                continue
            try:
                names = sorted(entry.name for entry in os.scandir(directory) if entry.name.endswith(".json"))
            except OSError:
                return "dispatch_queue_unreadable"
            for name in names:
                try:
                    descriptor = os.open(directory / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
                except FileNotFoundError:
                    continue  # consumed since it was listed
                except OSError:
                    return "dispatch_queue_unreadable"
                with os.fdopen(descriptor, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        return "dispatch_queue_unreadable"
                    raw = stream.read(MAX_QUEUE_MESSAGE_BYTES + 1)
                if len(raw) > MAX_QUEUE_MESSAGE_BYTES:
                    return "dispatch_queue_unreadable"
                if needle in raw:
                    return "dispatch_row_pending"
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
    # Each member must still be the regular file the plan listed when it is packed.
    identities = {name: (group["dev"], group["inode"]) for group in members for name in group["relative_paths"]}
    try:
        if publisher is None:
            # Hash the exact tar stream without an archive on disk; only the pointer needs headroom.
            sink = evidence._HashingSink()
            packed = evidence._pack_stream(root, sink, members=names, identities=identities)
            digest, size = "sha256:" + sink.digest.hexdigest(), sink.size
            pointer_bytes = len(json.dumps(packed, indent=2).encode()) + 65536
            reservation = evidence.reserve_control_plane_disk(
                "evidence_offload", target_root=root.parent, expected_bytes=max(_MIB, 2 * pointer_bytes),
                reservation_root=evidence.DEFAULT_RESERVATION_ROOT)
            reference = dict((stream_publisher or evidence.publish_configured_scene_stream)(
                write_stream=lambda stream: evidence._pack_stream(root, stream, members=names, identities=identities),
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
                packed = evidence._pack_stream(root, stream, members=names, identities=identities)
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


def _still_there(held, name: Path) -> bool:
    """Whether ``name`` is still listed; a name or directory that cannot be looked at counts as there."""

    try:
        os.stat(name.name, dir_fd=held.directory(name.parts[:-1]), follow_symlinks=False)
    except FileNotFoundError:
        return False  # the name, or a directory above it, is gone
    except OSError:
        return True
    return True


def _remove_members(held, group, names, *, sha256):
    """Remove one planned group: the reason it stopped (or None), the names it removed, and the
    names that vanished without it.

    ``_remove_group`` rechecks every name, then unlinks them in order and stops at
    the first unlink that fails. So after ``unlink_failed`` the names before the
    first one still listed were removed here (offloaded: restore brings them
    back). Any other name that is missing, after any failure, went without this
    offload: its directory or itself moved or was removed (``member_vanished``).
    Only the names still listed are kept.
    """

    reason = held_files._remove_group(held, group, names, changed="member_changed", sha256=sha256)
    if reason is None:
        return None, list(names), []
    listed = [_still_there(held, name) for name in names]
    removed = 0
    if reason.startswith("unlink_failed:"):
        removed = next((index for index, there in enumerate(listed) if there), len(names))
    vanished = [name for name, there in zip(names[removed:], listed[removed:]) if not there]
    return reason, list(names[:removed]), vanished


def _evict(root: Path, members: Sequence[Mapping[str, Any]], sha_by_name: Mapping[str, str], row) -> list[dict]:
    """Unlink every member group through held directory descriptors; the kept members, typed."""

    kept: list[dict[str, str]] = []

    def keep(group, reason, names=None):
        for relative in names if names is not None else group["relative_paths"]:
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
            digests = {sha_by_name[name] for name in group["relative_paths"]}
            if len(digests) != 1:
                # Its names were packed with different bytes: the archive holds no one version of it.
                keep(group, "member_changed")
                continue
            names = [Path(name) for name in group["relative_paths"]]
            reason, removed, vanished = held.item(
                functools.partial(_remove_members, sha256=digests.pop()), group, names)
            gone = {name.as_posix() for name in (*removed, *vanished)}
            for name in vanished:
                # Not removed here and not local: the pointer does not keep it, so restore brings it back.
                _skip(row, name.as_posix(), "member_vanished", group["size_bytes"])
            if reason:
                keep(group, reason, [name for name in group["relative_paths"] if name not in gone])
            row["offloaded_count"] += len(removed)
            if len(removed) == len(names):
                row["offloaded_bytes"] += group["size_bytes"]
    finally:
        held.close()
    return kept


def _withdraw(path: Path) -> None:
    """Remove a pointer nothing was evicted behind, so the next tick plans its run again."""

    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.unlink(path.name, dir_fd=directory)
        os.fsync(directory)
    finally:
        os.close(directory)


def offload_result_residue(
    *,
    run_root: str | Path,
    apply: bool = False,
    ack: str = "",
    hot_window_seconds: int = DEFAULT_HOT_WINDOW_SECONDS,
    protection_checker: Callable[[Path], bool | str | None] | None = None,
    publisher: Callable[..., Mapping[str, Any]] | None = None,
    stream_publisher: Callable[..., Mapping[str, Any]] | None = None,
    queue_roots: Sequence[str | Path] = (),
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
    pointed: dict[str, Any] | None = None
    if pointer.exists() or pointer.is_symlink():
        try:
            pointed = _read_pointer(root)
        except Exception as exc:  # noqa: BLE001 - a pointer that does not verify is left alone
            return _retained(row, "pointer_invalid", failure=offload_failure(exc, "pointer"))
        resumable, row["pointed_remaining_count"], row["pointed_remaining_bytes"] = _pointed_state(root, pointed)
        if not resumable:
            return _retained(row, "already_offloaded")
        # A crash after the pointer left members behind it: evict them, through every gate below.
        row["resume"] = True
    try:
        registry, registry_path, registry_bytes = _sealed_registry(root)
    except Exception as exc:  # noqa: BLE001 - an unsealed or unreadable run keeps its residue
        return _retained(row, "registry_unsealed", failure=offload_failure(exc, "registry"))
    row["registry_digest"] = registry["registry_digest"]
    if pointed is not None and pointed["registry_digest"] != registry["registry_digest"]:
        return _retained(row, "pointer_invalid")
    receipt_reason = _dispatch_receipt_reason(root, registry) or _queue_row_reason(root.name, queue_roots)
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
        if pointed is not None:
            resumable, _count, _size = _pointed_state(root, pointed)
            row["candidate_count"] = sum(len(group["relative_paths"]) for group in resumable)
            row["candidate_bytes"] = sum(group["size_bytes"] for group in resumable)
            return _resume(row, root, pointer, pointed) if apply else _finished(row)
        try:
            members = _plan_members(root, row, _registered_paths(root, registry), registry_stat.st_mtime_ns)
        except Exception as exc:  # noqa: BLE001 - what a run's receipts name must be known
            return _retained(row, "plan_failed", failure=offload_failure(exc, "plan"))
        row["candidate_count"] = sum(len(group["relative_paths"]) for group in members)
        row["candidate_bytes"] = sum(group["size_bytes"] for group in members)
        if not apply or not members:
            return _finished(row)
        return _apply(row, root, pointer, registry, registry_path, registry_bytes, members,
                      protection_checker, publisher, stream_publisher, now, queue_roots)


def _apply(row, root, pointer, registry, registry_path, registry_bytes, members, protection_checker,
           publisher, stream_publisher, now, queue_roots) -> dict[str, Any]:
    # Counted against the tick's cap whether or not the publication succeeds.
    row["publication_attempted"] = True
    identity = os.lstat(root)
    names = [name for group in members for name in group["relative_paths"]]
    modes = {name: group["mode"] for group in members for name in group["relative_paths"]}
    # The names of one inode share a group, so restore links them again instead of copying.
    groups = {name: index for index, group in enumerate(members) for name in group["relative_paths"]}
    try:
        packed, digest, size, reference, reservation = _publish(root, names, members, publisher, stream_publisher)
    except Exception as exc:  # noqa: BLE001 - nothing was evicted
        return _retained(row, "publication_failed", failure=offload_failure(exc, "publish"))
    try:
        row["archive"] = {"uri": reference["uri"], "sha256": digest, "size_bytes": size}
        queued = _queue_row_reason(root.name, queue_roots)
        if queued:
            return _retained(row, queued)
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
            "members": [{**member, "mode": modes[member["relative_path"]], "group": groups[member["relative_path"]]}
                        for member in packed],
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
        # The archive is the only record of a vanished member's bytes: its pointer stays.
        if row["offloaded_count"] == 0 and not row["skipped_by_reason"].get("member_vanished"):
            try:
                _withdraw(pointer)
            except OSError as exc:
                row["failure"] = offload_failure(exc, "pointer")
            else:
                del row["pointer"]
                return _retained(row, "nothing_evicted")
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
    """The run's residue pointer, digest-verified and bound to this run, or a refusal."""

    path = root.parent / f"{root.name}{POINTER_SUFFIX}"
    if path.is_symlink() or not path.is_file():
        raise ResultResidueOffloadError("result_residue_pointer_invalid")
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
        raise ResultResidueOffloadError("result_residue_pointer_invalid")
    for member in value["members"]:
        parts = PurePosixPath(str(member.get("relative_path"))).parts
        if (
            not parts
            or PurePosixPath(member["relative_path"]).is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
            or not _name_supported(member["relative_path"])
            or not isinstance(member.get("sha256"), str)
            or not isinstance(member.get("size_bytes"), int)
        ):
            raise ResultResidueOffloadError("result_residue_pointer_invalid")
    return value


def _pointed_state(root: Path, value: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    """What a pointer's members left local: the groups still to evict, and every listed member still here.

    A member still listed through no link, as a regular file on the run's device,
    is local; one the pointer does not keep is to be evicted again, grouped by
    inode as the plan grouped them. The count and bytes cover every listed member
    still local, kept or not.
    """

    kept = {str(row.get("relative_path")) for row in value["kept"] if isinstance(row, Mapping)}
    device = os.lstat(root).st_dev
    groups: dict[tuple[int, int], tuple[os.stat_result, list[str]]] = {}
    count = size = 0
    for member in value["members"]:
        relative = member["relative_path"]
        path = root
        try:
            for part in PurePosixPath(relative).parts:
                path = path / part
                info = os.lstat(path)
                if stat.S_ISLNK(info.st_mode):
                    break
            else:
                if stat.S_ISREG(info.st_mode):
                    count, size = count + 1, size + info.st_size
                    if relative not in kept and info.st_dev == device:
                        groups.setdefault((info.st_dev, info.st_ino), (info, []))[1].append(relative)
        except OSError:
            continue  # gone, or no longer under a directory: not local
    members = [
        {"relative_paths": sorted(names), "dev": dev, "inode": inode, "nlink": info.st_nlink,
         "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
        for (dev, inode), (info, names) in groups.items()
    ]
    return sorted(members, key=lambda group: group["relative_paths"]), count, size


def _resume(row: dict[str, Any], root: Path, pointer: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Finish an eviction a crash cut short, under the run lock and every gate.

    Each listed member the pointer does not keep, still local, goes only when its
    bytes still hash to the pointer's (the archive's); any other is kept, and the
    pointer is rewritten to say so. Nothing is published again, and the pointer is
    never withdrawn: members evicted before the crash live only in its archive.
    """

    members, _count, _size = _pointed_state(root, value)
    kept = _evict(root, members, {member["relative_path"]: member["sha256"] for member in value["members"]}, row)
    if kept:
        known = {str(entry.get("relative_path")) for entry in value["kept"] if isinstance(entry, Mapping)}
        value = {**value, "kept": [*value["kept"], *(entry for entry in kept if entry["relative_path"] not in known)]}
        value["pointer_digest"] = canonical_digest(value, digest_field="pointer_digest")
        try:
            _write_json(pointer, value)
        except Exception as exc:  # noqa: BLE001 - kept members are still local; restore finds them
            row["failure"] = offload_failure(exc, "pointer")
    _members, row["pointed_remaining_count"], row["pointed_remaining_bytes"] = _pointed_state(root, value)
    return _finished(row)


def restore_result_residue(**options: Any) -> dict[str, Any]:
    """Bring a run's residue back: ``task_evaluation_result_residue_restore.restore_result_residue``."""

    from .task_evaluation_result_residue_restore import restore_result_residue as restore

    return restore(**options)


def residue_row(
    run_root: str | Path,
    bulk_result: Mapping[str, Any],
    *,
    apply: bool,
    hot_window_seconds: int,
    protection_checker: Callable[[Path], bool | str | None] | None,
    publisher: Callable[..., Mapping[str, Any]] | None,
    now: Callable[[], float],
    queue_roots: Sequence[str | Path] = (),
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
            protection_checker=protection_checker, publisher=publisher, now=now, queue_roots=queue_roots)
    except Exception as exc:  # noqa: BLE001 - one run never costs the others
        return {"status": "error", "run": name, **offload_failure(exc, "residue")}


class ResidueTick:
    """One storage GC tick's residue phase: a row per registry run, and the phase entry.

    While applying it attempts at most ``max_runs`` publications, counted whether
    they succeed or not, so a failing publisher costs at most that many uploads a
    tick and no run is retried within it. Every later run is only planned and
    reported as ``deferred_tick_cap`` with its candidate bytes; its turn comes in
    a later tick.
    """

    def __init__(self, *, applying: bool, enabled: bool, max_runs: int | None = None, **options):
        max_runs = DEFAULT_MAX_RUNS_PER_TICK if max_runs is None else max_runs
        if not isinstance(max_runs, int) or isinstance(max_runs, bool) or max_runs < 0:
            raise ResultResidueOffloadError("result_residue_max_runs_invalid")
        self.applying, self.enabled, self.max_runs, self.options = bool(applying), bool(enabled), max_runs, options
        self.rows: list[dict[str, Any]] = []
        self.attempted = 0

    def add(self, run_root: str | Path, bulk_result: Mapping[str, Any]) -> dict[str, Any]:
        """Plan or offload one registry run's residue given its per-artifact offload result."""

        applying = self.applying and self.attempted < self.max_runs
        row = residue_row(run_root, bulk_result, apply=applying, **self.options)
        if row.get("publication_attempted"):
            self.attempted += 1
        elif self.applying and not applying and row.get("status") == "dry_run" and row.get("candidate_count"):
            row = _retained(row, "deferred_tick_cap")
        self.rows.append(row)
        return row

    def phase(self, *, alert: str | None = None) -> dict[str, Any]:
        entry = residue_phase(self.rows, enabled=self.enabled, applying=self.applying, alert=alert)
        entry["max_runs_per_tick"], entry["attempted_count"] = self.max_runs, self.attempted
        return entry


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
        # What pointed runs still hold locally: kept members, or ones a crash left to resume.
        "pointed_remaining_bytes": sum(int(row.get("pointed_remaining_bytes") or 0) for row in rows),
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
    "RESIDUE_MAX_RUNS_ENV",
    "RESIDUE_OFFLOAD_ENV",
    "RESIDUE_OFFLOAD_INVALID",
    "RESTORE_RECEIPT_SUFFIX",
    "ResidueTick",
    "ResultResidueOffloadError",
    "offload_result_residue",
    "residue_phase",
    "residue_row",
    "restore_result_residue",
    "result_residue_offload_setting",
]
