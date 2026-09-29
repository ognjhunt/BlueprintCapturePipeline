"""Offload sealed result payloads while preserving authenticated artifact URLs.

The scientific registry and closure metadata remain immutable and local. Only
registered bulk payloads are evicted, after full remote byte readback and a
fsynced, registry-bound private reference. Downloads materialize one artifact
under a disk reservation; the response releases it even on disconnect.

A streamed run's artifacts that never came to the host stay in the promoted
provider archive, named by ``archive_member_references.v1.json``
(``task_evaluation_result_archive_members``): a download reads such a member
with one B2 range request, and the offload counts it as already remote.

Coalescing. A reader that downloads every artifact -- the owner readback --
would pay one B2 range request per member. A download therefore reads, in the
same range, the registered members that follow its member in the archive (at
most ``SPAN_MEMBER_LIMIT`` of them within ``SPAN_BYTES`` of record data, gaps
under ``SPAN_GAP_BYTES``). Each is checked against its CRC-32 and SHA-256, and
the followers' bytes wait in a small in-process cache (``CACHE_BYTES``, for
``CACHE_SECONDS``) for the next download; the readback walks members in
archive order, so a whole run costs a few span reads.
"""

from __future__ import annotations

import argparse
import bisect
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from functools import partial
from pathlib import Path
from typing import Any, Callable

from .control_plane_disk_budget import reserve_control_plane_disk
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_configured_scene_object_store import (
    _artifact_object_store_client,
    materialize_configured_scene_artifact,
    publish_configured_scene_artifact,
)
from .task_evaluation_result_delivery import (
    REGISTRY_SCHEMA_VERSION,
    TaskEvaluationResultDeliveryError,
    _sha256,
)

REFERENCE_SCHEMA = "task_evaluation_result_artifact_remote_reference.v1"
REMOTE_DIRECTORY = "remote_artifacts"
ARTIFACT_KIND = "task-evaluation-result"
APPLY_ACK = "offload-sealed-result-artifacts"
CACHE_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_RESULT_ARTIFACT_CACHE_ROOT"
DEFAULT_CACHE_ROOT = "/var/lib/blueprint/pipeline-control-plane/result-artifact-cache"
# Keep receipts, manifests, scores and reports for closeout/replay discovery.
BULK_ROLES = frozenset(
    {
        "review_video",
        "episode_evidence",
        "retained_lossless_frame",
        "exact_policy_request",
        "runtime_supporting_evidence",
        "retained_interrupted_cell_evidence",
    }
)
#: Where an offload failed: reading and verifying the run's registry, planning
#: what to move from the registered files and remote references, checking its
#: protection, publishing to the artifact store, or evicting local copies.
OFFLOAD_STAGES = ("registry", "plan", "protection", "publish", "evict")
_STAGE_ATTRIBUTE = "result_artifact_offload_stage"


@contextmanager
def _offload_stage(stage: str):
    """Mark an exception leaving this block with ``stage``, unless an inner block already did."""
    try:
        yield
    except Exception as exc:
        if not hasattr(exc, _STAGE_ATTRIBUTE):
            with suppress(AttributeError):
                setattr(exc, _STAGE_ATTRIBUTE, stage)
        raise


def offload_failure(exc: BaseException, stage: str | None = None) -> dict[str, Any]:
    """A failed offload's typed record: its error type, errno for an ``OSError``, and stage.

    Never the message or file name, which can carry a host path. An exception no
    stage marked failed while the run's registry was read and verified.
    """
    return {
        "error_type": type(exc).__name__,
        "errno": exc.errno if isinstance(exc, OSError) else None,
        "stage": stage or getattr(exc, _STAGE_ATTRIBUTE, "registry"),
    }


def _safe_path(root: Path, relative: str) -> Path:
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(p in {".", ".."} for p in parts):
        raise TaskEvaluationResultDeliveryError("result_artifact_path_invalid")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise TaskEvaluationResultDeliveryError("result_artifact_symlink_forbidden")
    if path.exists() and not path.is_file():
        raise TaskEvaluationResultDeliveryError("result_artifact_file_invalid")
    return path


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TaskEvaluationResultDeliveryError("result_artifact_metadata_missing")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TaskEvaluationResultDeliveryError("result_artifact_metadata_invalid")
    return value


def _remote_path(root: Path, relative: str) -> Path:
    name = hashlib.sha256(relative.encode()).hexdigest() + ".json"
    return _safe_path(root, f"artifacts/result_delivery/{REMOTE_DIRECTORY}/{name}")


def _validate_remote(value: dict, *, registry: dict, relative: str, record: dict) -> dict:
    reference = value.get("reference", {})
    if (
        value.get("schema_version") != REFERENCE_SCHEMA
        or not isinstance(reference, dict)
        or value.get("reference_digest") != canonical_digest(value, digest_field="reference_digest")
        or value.get("run_id") != registry.get("run_id")
        or value.get("registry_digest") != registry.get("registry_digest")
        or value.get("relative_path") != relative
        or reference.get("artifact_kind") != ARTIFACT_KIND
        or reference.get("digest") != record.get("sha256")
        or reference.get("size_bytes") != record.get("size_bytes")
        or reference.get("remote_identity_verified") is not True
        or reference.get("full_byte_service_account_readback_passed") is not True
        or reference.get("readback_digest") != record.get("sha256")
        or reference.get("readback_size_bytes") != record.get("size_bytes")
    ):
        raise TaskEvaluationResultDeliveryError("result_artifact_remote_reference_invalid")
    return reference


def _durable_reference(path: Path, value: dict, owner: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    metadata = owner.stat()
    if os.geteuid() == 0:
        os.chown(path.parent, metadata.st_uid, metadata.st_gid)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".reference-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fchmod(stream.fileno(), 0o440)
            if os.geteuid() == 0:
                os.fchown(stream.fileno(), metadata.st_uid, metadata.st_gid)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _reap_abandoned_downloads(cache: Path) -> None:
    """Reclaim only this service owner's downloads from dead processes."""
    for directory in cache.iterdir():
        match = re.fullmatch(r"download-([0-9]+)-[A-Za-z0-9_]+", directory.name)
        if not match or directory.is_symlink() or not directory.is_dir():
            continue
        try:
            if directory.stat().st_uid != os.geteuid():
                continue
        except FileNotFoundError:
            continue
        try:
            os.kill(int(match[1]), 0)
        except ProcessLookupError:
            shutil.rmtree(directory, ignore_errors=True)
        except PermissionError:
            continue


def acquire_artifact_read_lease(root: Path, *, exclusive: bool = False):
    if root.is_symlink() or root.resolve() != root:
        raise TaskEvaluationResultDeliveryError("result_artifact_run_root_invalid")
    registry_path = _safe_path(root, "artifacts/result_delivery/artifact_registry.json")
    path = _safe_path(root, "artifacts/result_delivery/.artifact-readers.lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o660)
    try:
        owner = registry_path.stat()
        # The storage GC is root with CAP_CHOWN but not CAP_FOWNER: once the lock
        # belongs to the registry owner, only that owner may change its mode. So
        # the mode is set only while this process owns the file, and the owner is
        # changed last. A lock someone else owns keeps its mode; the open and the
        # flock are the access check.
        metadata = os.fstat(descriptor)
        if stat.S_IMODE(metadata.st_mode) != 0o660 and metadata.st_uid == os.geteuid():
            os.fchmod(descriptor, 0o660)
        if os.geteuid() == 0 and (metadata.st_uid, metadata.st_gid) != (owner.st_uid, owner.st_gid):
            os.fchown(descriptor, owner.st_uid, owner.st_gid)
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    except BaseException:
        os.close(descriptor)
        raise
    closed = False

    def release():
        nonlocal closed
        if not closed:
            os.close(descriptor)
            closed = True

    return release


@contextmanager
def _artifact_eviction_lease(root: Path):
    release = acquire_artifact_read_lease(root, exclusive=True)
    try:
        yield
    finally:
        release()


def _download_cache() -> Path:
    cache = Path(os.getenv(CACHE_ROOT_ENV, DEFAULT_CACHE_ROOT))
    if not cache.is_absolute() or cache.is_symlink() or cache.resolve() != cache:
        raise TaskEvaluationResultDeliveryError("result_artifact_cache_root_invalid")
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    _reap_abandoned_downloads(cache)
    return cache


SPAN_BYTES = 8 * 1024 * 1024
SPAN_GAP_BYTES = 1024 * 1024
SPAN_MEMBER_LIMIT = 64
CACHE_BYTES = 64 * 1024 * 1024
CACHE_SECONDS = 300.0


class _MemberCache:
    """Verified member bytes a span read fetched ahead, bounded in bytes and time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: OrderedDict[tuple[str, str], tuple[bytes, float]] = OrderedDict()
        self._bytes = 0

    def take(self, key: tuple[str, str]) -> bytes | None:
        with self._lock:
            row = self._rows.pop(key, None)
            if row is None:
                return None
            self._bytes -= len(row[0])
            return row[0] if row[1] > time.monotonic() else None

    def put(self, key: tuple[str, str], data: bytes) -> None:
        with self._lock:
            if key in self._rows or len(data) > CACHE_BYTES:
                return
            self._rows[key] = (data, time.monotonic() + CACHE_SECONDS)
            self._bytes += len(data)
            while self._bytes > CACHE_BYTES:
                _, (dropped, _) = self._rows.popitem(last=False)
                self._bytes -= len(dropped)

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()
            self._bytes = 0


_MEMBER_CACHE = _MemberCache()


def clear_archive_member_cache() -> None:
    """Drop every member fetched ahead (tests; a process that must release memory)."""
    _MEMBER_CACHE.clear()


class _SpanSource:
    """A fetched byte span, served to the member reader like the archive it came from."""

    def __init__(self, data: bytes, start: int) -> None:
        self._data, self._start, self.block_bytes = data, start, 1024 * 1024

    def stream_to(self, sink, *, start: int, end: int) -> int:
        view = memoryview(self._data)[start - self._start:end - self._start]
        for offset in range(0, len(view), self.block_bytes):
            sink(bytes(view[offset:offset + self.block_bytes]))
        return end - start


def _member_row(entry: dict) -> dict:
    return {"method": entry["method"], "size": entry["size_bytes"], "crc32": entry["crc32"],
            "sha256": entry["sha256"], "data_offset": entry["data_offset"],
            "compressed_size": entry["compressed_size"]}


_ARCHIVE_ORDER: OrderedDict[str, tuple[list[dict], list[int]]] = OrderedDict()
_ARCHIVE_ORDER_LOCK = threading.Lock()


def _archive_order(references: dict) -> tuple[list[dict], list[int]]:
    """The references' distinct members in archive order and their offsets, kept for a few runs."""
    key = str(references.get("references_digest"))
    with _ARCHIVE_ORDER_LOCK:
        order = _ARCHIVE_ORDER.get(key)
        if order is not None:
            _ARCHIVE_ORDER.move_to_end(key)
            return order
    rows = sorted({row["archive_path"]: row for row in references["members"].values()}.values(),
                  key=lambda row: row["data_offset"])
    order = (rows, [row["data_offset"] for row in rows])
    with _ARCHIVE_ORDER_LOCK:
        _ARCHIVE_ORDER[key] = order
        while len(_ARCHIVE_ORDER) > 4:
            _ARCHIVE_ORDER.popitem(last=False)
    return order


def _span(references: dict, entry: dict) -> list[dict]:
    """``entry`` and the registered members that follow it closely enough to share its range read."""
    start, end, span = entry["data_offset"], entry["data_offset"] + entry["compressed_size"], [entry]
    followers, offsets = _archive_order(references)
    for row in followers[bisect.bisect_right(offsets, start):]:
        if row["data_offset"] < end:
            continue
        stop = row["data_offset"] + row["compressed_size"]
        if (len(span) >= SPAN_MEMBER_LIMIT or row["data_offset"] - end > SPAN_GAP_BYTES
                or stop - start > SPAN_BYTES or row["size_bytes"] > SPAN_BYTES):
            break
        span.append(row)
        end = stop
    return span


def _read_archive_member(references: dict, entry: dict, sink) -> None:
    """Write one member's checked bytes to ``sink``; fetch its followers ahead in the same range."""
    from .provider_output_member_view import (
        ProviderOutputMemberViewError,
        open_durable_archive,
        stream_indexed_member,
    )
    from .provider_output_range_transport import ProviderOutputTransportError

    archive = references["archive"]
    cached = _MEMBER_CACHE.take((archive["sha256"], entry["archive_path"]))
    if (cached is not None and len(cached) == entry["size_bytes"]
            and "sha256:" + hashlib.sha256(cached).hexdigest() == entry["sha256"]):
        sink(cached)
        return
    reader = open_durable_archive(archive["durable_reference"], archive["size_bytes"])
    span = _span(references, entry)
    if len(span) == 1:
        stream_indexed_member(reader, _member_row(entry), sink)
        return
    start = entry["data_offset"]
    end = span[-1]["data_offset"] + span[-1]["compressed_size"]
    fetched = bytearray()
    try:
        reader.stream_to(fetched.extend, start=start, end=end)
    except ProviderOutputTransportError as exc:
        raise ProviderOutputMemberViewError(str(exc)) from None
    source = _SpanSource(bytes(fetched), start)
    stream_indexed_member(source, _member_row(entry), sink)
    for row in span[1:]:
        chunks: list[bytes] = []
        try:
            stream_indexed_member(source, _member_row(row), chunks.append)
        except ProviderOutputMemberViewError:
            continue  # its own download reads and refuses it; this one's bytes were checked
        _MEMBER_CACHE.put((archive["sha256"], row["archive_path"]), b"".join(chunks))


def materialize_missing_result_artifact(
    *,
    run_root: Path,
    registry: dict,
    record: dict,
    source_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """A registered artifact that is not on disk: offloaded, or an archive member.

    An offload reference takes precedence; without one, a streamed run's
    archive-member references serve the member. A run with neither is exactly
    today's ``materialize_result_artifact``.
    """
    from .task_evaluation_result_archive_members import REFERENCES_RELATIVE_PATH

    relative = source_path.relative_to(run_root).as_posix()
    remote = _remote_path(run_root, relative)
    members = run_root / REFERENCES_RELATIVE_PATH
    if not remote.exists() and not remote.is_symlink() and (members.exists() or members.is_symlink()):
        return materialize_archive_member_artifact(
            run_root=run_root, registry=registry, record=record, source_path=source_path)
    return materialize_result_artifact(
        run_root=run_root, registry=registry, record=record, source_path=source_path)


def materialize_archive_member_artifact(
    *,
    run_root: Path,
    registry: dict,
    record: dict,
    source_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Serve a registered artifact that stays in its streamed attempt's promoted archive.

    The run's archive-member references, bound to this registry, name the
    member's record in the durable archive. One B2 range request (after the
    reader's one-byte ETag probe) inflates it into the result-artifact cache
    under a ``result_artifact_download`` reservation of its exact size plus
    1 MiB, checked against the recorded CRC-32 and SHA-256. The cleanup
    contract is ``materialize_result_artifact``'s.
    """
    from .provider_output_member_view import ProviderOutputMemberViewError
    from .task_evaluation_result_archive_members import (
        ArchiveMemberReferenceError,
        archive_member_entry,
        load_archive_member_references,
    )

    relative = source_path.relative_to(run_root).as_posix()
    _safe_path(run_root, relative)
    try:
        references = load_archive_member_references(run_root, registry)
        entry = archive_member_entry(references, relative, record)
    except ArchiveMemberReferenceError as exc:
        raise TaskEvaluationResultDeliveryError(str(exc)) from None
    if entry is None:
        raise TaskEvaluationResultDeliveryError("result_artifact_metadata_missing")
    cache = _download_cache()
    reservation = reserve_control_plane_disk(
        "result_artifact_download",
        target_root=cache,
        expected_bytes=record["size_bytes"] + 1024 * 1024,
    )
    temporary: Path | None = None

    def cleanup() -> None:
        try:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
        finally:
            reservation.release()

    try:
        temporary = Path(tempfile.mkdtemp(prefix=f"download-{os.getpid()}-", dir=cache))
        destination = temporary / source_path.name
        with destination.open("xb") as sink:
            try:
                _read_archive_member(references, entry, sink.write)
            except ProviderOutputMemberViewError as exc:
                raise TaskEvaluationResultDeliveryError(
                    f"result_artifact_archive_member_unreadable:{exc}") from None
        return destination, {**record, "_artifact_cleanup": cleanup}
    except Exception:
        cleanup()
        raise


def materialize_result_artifact(
    *,
    run_root: Path,
    registry: dict,
    record: dict,
    source_path: Path,
) -> tuple[Path, dict[str, Any]]:
    relative = source_path.relative_to(run_root).as_posix()
    _safe_path(run_root, relative)
    reference = _validate_remote(
        _read(_remote_path(run_root, relative)),
        registry=registry,
        relative=relative,
        record=record,
    )
    cache = _download_cache()
    reservation = reserve_control_plane_disk(
        "result_artifact_download",
        target_root=cache,
        expected_bytes=record["size_bytes"] + 1024 * 1024,
    )
    temporary: Path | None = None

    def cleanup() -> None:
        try:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
        finally:
            reservation.release()

    try:
        temporary = Path(tempfile.mkdtemp(prefix=f"download-{os.getpid()}-", dir=cache))
        destination = temporary / source_path.name
        materialize_configured_scene_artifact(
            reference=reference,
            destination=destination,
            maximum_size_bytes=record["size_bytes"],
        )
        return destination, {**record, "_artifact_cleanup": cleanup}
    except Exception:
        cleanup()
        raise


def _sealed_registry(root: Path) -> tuple[dict, Path, bytes]:
    registry_path = _safe_path(root, "artifacts/result_delivery/artifact_registry.json")
    raw = registry_path.read_bytes()
    registry = json.loads(raw)
    delivery = _read(_safe_path(root, "artifacts/result_delivery/delivery.json"))
    if (
        registry.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or registry.get("registry_digest")
        != canonical_digest(registry, digest_field="registry_digest")
        or not isinstance(registry.get("artifacts"), list)
        or not all(isinstance(row, dict) for row in registry["artifacts"])
        or delivery.get("schema_version") != "task_evaluation_result_delivery.v2"
        or delivery.get("run_id") != registry.get("run_id")
        or delivery.get("delivery_digest") != registry.get("delivery_digest")
        or delivery.get("delivery_digest")
        != cross_runtime_canonical_digest(delivery, digest_field="delivery_digest")
        or delivery.get("result_status") not in {"completed_unqualified", "blocked", "cancelled"}
    ):
        raise TaskEvaluationResultDeliveryError("result_artifact_sealed_delivery_invalid")
    closure = delivery.get("reproducibility", {})
    for name in ("billing", "teardown", "provider_zero"):
        descriptor = closure.get(f"{name}_receipt", {})
        matches = [
            row
            for row in registry["artifacts"]
            if row.get("artifact_id") == descriptor.get("artifact_id")
            and row.get("role") == f"closure_{name}"
        ]
        if len(matches) != 1 or (
            name == "provider_zero" and descriptor.get("provider_zero_verified") is not True
        ):
            raise TaskEvaluationResultDeliveryError("result_artifact_resource_closeout_unsealed")
        record = matches[0]
        evidence = Path(str(record.get("evidence_root") or ""))
        relative = (evidence.relative_to(root) / str(record.get("relative_path") or "")).as_posix()
        path = _safe_path(root, relative)
        if (
            (record.get("sha256"), record.get("size_bytes"))
            != (descriptor.get("digest"), descriptor.get("size_bytes"))
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            raise TaskEvaluationResultDeliveryError("result_artifact_resource_closeout_changed")
    return registry, registry_path, raw


def _archive_member_references(root: Path, registry: dict) -> dict:
    from .task_evaluation_result_archive_members import (
        ArchiveMemberReferenceError,
        load_archive_member_references,
    )

    try:
        return load_archive_member_references(root, registry) or {}
    except ArchiveMemberReferenceError as exc:
        raise TaskEvaluationResultDeliveryError(str(exc)) from None


def _archive_member_entry(references: dict, relative: str, record: dict) -> dict | None:
    from .task_evaluation_result_archive_members import (
        ArchiveMemberReferenceError,
        archive_member_entry,
    )

    try:
        return archive_member_entry(references, relative, record)
    except ArchiveMemberReferenceError as exc:
        raise TaskEvaluationResultDeliveryError(str(exc)) from None


def _offload_candidates(root: Path, registry: dict, minimum_size_bytes: int, report: dict) -> list:
    """The registered bulk files still local, each with its path and record; the remote ones counted.

    A record outside the run, an alias conflict, a remote reference that does
    not verify or a file whose bytes changed raises: the run keeps everything.
    """
    groups: dict[str, list[dict]] = {}
    for record in registry["artifacts"]:
        evidence = Path(str(record.get("evidence_root") or ""))
        # Verify every path component, including directories, before resolving.
        try:
            evidence_relative = evidence.relative_to(root)
        except ValueError as exc:
            raise TaskEvaluationResultDeliveryError("result_artifact_evidence_outside_run") from exc
        relative = (evidence_relative / str(record.get("relative_path") or "")).as_posix()
        _safe_path(root, relative)
        groups.setdefault(relative, []).append(record)
    candidates = []
    members = None
    for relative, records in groups.items():
        record = records[0]
        if any(row.get("role") not in BULK_ROLES for row in records):
            continue
        if (
            relative.startswith("artifacts/result_delivery/")
            or record.get("size_bytes", 0) < minimum_size_bytes
        ):
            continue
        if any(
            (row.get("sha256"), row.get("size_bytes"))
            != (record.get("sha256"), record.get("size_bytes"))
            for row in records
        ):
            raise TaskEvaluationResultDeliveryError("result_artifact_alias_conflict")
        path = _safe_path(root, relative)
        if not path.exists():
            remote = _remote_path(root, relative)
            if not remote.exists() and not remote.is_symlink():
                # A streamed run's member that never came to the host.
                members = members if members is not None else _archive_member_references(root, registry)
                if _archive_member_entry(members, relative, record) is not None:
                    report["already_remote_count"] += 1
                    continue
            _validate_remote(
                _read(remote),
                registry=registry,
                relative=relative,
                record=record,
            )
            report["already_remote_count"] += 1
            continue
        if path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
            raise TaskEvaluationResultDeliveryError("result_artifact_source_changed")
        candidates.append((relative, path, record))
    return candidates


def offload_result_artifacts(
    *,
    run_root: str | Path,
    apply: bool = False,
    ack: str = "",
    minimum_size_bytes: int = 64 * 1024,
    hot_window_seconds: int = 172800,
    protection_checker: Callable[[Path], bool | str | None] | None = None,
    publisher: Callable[..., dict] | None = None,
    now: Callable[[], float] = time.time,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Dry-run first; preserve every source whose publication or seal changes.

    A retained run says why (``retained_reason``): ``hot``, the reason string the
    protection checker returned, or ``protected``. A skipped artifact records
    ``offload_failure``'s fields, and an exception that escapes carries the stage
    it failed in for ``offload_failure`` to read.
    """
    unresolved = Path(run_root).expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise TaskEvaluationResultDeliveryError("result_artifact_run_root_invalid")
    root = unresolved.resolve()
    if apply and ack != APPLY_ACK:
        raise TaskEvaluationResultDeliveryError("result_artifact_offload_not_authorized")
    if minimum_size_bytes < 1 or hot_window_seconds < 0:
        raise TaskEvaluationResultDeliveryError("result_artifact_offload_limit_invalid")
    registry, registry_path, registry_bytes = _sealed_registry(root)
    report: dict[str, Any] = {
        "schema_version": "task_evaluation_result_artifact_offload.v1",
        "status": "applied" if apply else "dry_run",
        "run_id": registry["run_id"],
        "registry_digest": registry["registry_digest"],
        "candidate_count": 0,
        "candidate_bytes": 0,
        "offloaded_count": 0,
        "offloaded_bytes": 0,
        "already_remote_count": 0,
        "skipped": [],
        "observed_at_epoch": now(),
    }

    def completed_report():
        report["result_digest"] = canonical_digest(report, digest_field="result_digest")
        return report

    hot = now() - registry_path.stat().st_mtime < hot_window_seconds
    protected = None
    if not hot and protection_checker is not None:
        with _offload_stage("protection"):
            protected = protection_checker(root)
    if hot or protected:
        report["status"] = "retained_hot_or_active"
        report["retained_reason"] = "hot" if hot else (protected if isinstance(protected, str) else "protected")
        return completed_report()
    with _offload_stage("plan"):
        candidates = _offload_candidates(root, registry, minimum_size_bytes, report)
    report["candidate_count"] = len(candidates)
    report["candidate_bytes"] = sum(row[2]["size_bytes"] for row in candidates)
    if not apply or not candidates:
        return completed_report()
    # One client per run, shared by bounded S3 upload workers.
    if publisher is None:
        with _offload_stage("publish"):
            client, bucket = _artifact_object_store_client()
        publisher = partial(publish_configured_scene_artifact, client=client, bucket=bucket)
    lock_path = _safe_path(root, "artifacts/result_delivery/.offload.lock")
    with (
        # Anything below that no inner stage marks is publication: the disk
        # reservation, the offload lock and the upload workers.
        _offload_stage("publish"),
        reserve_control_plane_disk(
            "evidence_offload",
            target_root=root,
            expected_bytes=max(1024 * 1024, len(candidates) * 8192),
        ),
        lock_path.open("a+b") as lock,
        ThreadPoolExecutor(max_workers=4) as pool,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX)
        for start in range(0, len(candidates), 32):
            batch = candidates[start : start + 32]

            def publish(candidate):
                relative, path, record = candidate
                try:
                    if not path.exists():
                        return candidate, None, {"reason": "already_evicted"}
                    before = path.stat()
                    reference = dict(publisher(path=path, artifact_kind=ARTIFACT_KIND))
                    value = {
                        "schema_version": REFERENCE_SCHEMA,
                        "run_id": registry["run_id"],
                        "registry_digest": registry["registry_digest"],
                        "relative_path": relative,
                        "reference": reference,
                        "reference_digest": "",
                    }
                    value["reference_digest"] = canonical_digest(
                        value, digest_field="reference_digest"
                    )
                    _validate_remote(value, registry=registry, relative=relative, record=record)
                    return candidate, (value, before), None
                except Exception as exc:
                    return candidate, None, {"reason": type(exc).__name__, **offload_failure(exc, "publish")}

            published = list(pool.map(publish, batch))
            with _offload_stage("registry"):
                changed = registry_path.read_bytes() != registry_bytes
            with _offload_stage("protection"):
                changed = changed or (protection_checker and protection_checker(root))
            if changed:
                report["skipped"].append({"reason": "run_changed_or_active"})
                break
            with _offload_stage("evict"), _artifact_eviction_lease(root):
                for (relative, path, record), result, error in published:
                    if error:
                        report["skipped"].append({"relative_path": relative, **error})
                        continue
                    value, before = result
                    try:
                        _safe_path(root, relative)
                        current = path.stat()
                        if (
                            current.st_dev,
                            current.st_ino,
                            current.st_size,
                            current.st_mtime_ns,
                        ) != (
                            before.st_dev,
                            before.st_ino,
                            before.st_size,
                            before.st_mtime_ns,
                        ) or _sha256(path) != record["sha256"]:
                            raise TaskEvaluationResultDeliveryError(
                                "result_artifact_source_changed"
                            )
                        _durable_reference(_remote_path(root, relative), value, registry_path)
                        _safe_path(root, relative)
                        final = path.stat()
                        if (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) != (
                            current.st_dev,
                            current.st_ino,
                            current.st_size,
                            current.st_mtime_ns,
                        ) or _sha256(path) != record["sha256"]:
                            raise TaskEvaluationResultDeliveryError(
                                "result_artifact_source_changed"
                            )
                        path.unlink()
                        report["offloaded_count"] += 1
                        # Logical bytes; hardlinks can keep the same blocks allocated elsewhere.
                        report["offloaded_bytes"] += current.st_size
                    except Exception as exc:
                        report["skipped"].append({
                            "relative_path": relative,
                            "reason": type(exc).__name__,
                            **offload_failure(exc, "evict"),
                        })
            if progress:
                progress(dict(report))
    return completed_report()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack", default="")
    parser.add_argument("--hot-window-seconds", type=int, default=172800)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--pins-root", required=True)
    parser.add_argument("--queue-root", action="append", required=True)
    args = parser.parse_args()
    from .completed_replay_cache_retention import active_reference
    from .control_plane_storage_roots import require_storage_class
    from .control_plane_storage_gc import _queue_reference_text
    from .control_plane_storage_pins import live_pinned_paths

    require_storage_class(
        str(Path(args.run_root).parent),
        expected="evidence_cold",
        code="result_artifact_offload_root_class",
    )

    def protected(root):
        if active_reference(root, ignored_process_ids=(os.getpid(),)):
            return True
        if root.name in _queue_reference_text(args.queue_root):
            return True
        return any(
            Path(p) == root or root in Path(p).parents or Path(p) in root.parents
            for p in live_pinned_paths(args.pins_root)
        )

    report = offload_result_artifacts(
        run_root=args.run_root,
        apply=args.apply,
        ack=args.ack,
        hot_window_seconds=args.hot_window_seconds,
        protection_checker=protected,
        progress=lambda row: print(
            json.dumps({key: row[key] for key in ("offloaded_count", "offloaded_bytes")}),
            flush=True,
        ),
    )
    Path(args.report_out).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return int(bool(report["skipped"]))


if __name__ == "__main__":
    raise SystemExit(main())
