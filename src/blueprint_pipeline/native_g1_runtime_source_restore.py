"""Restore the shared G1 runtime source packet from its retained object-store copy.

The shared G1 registry binds ``runtime_source_receipt_path`` to
``/mnt/blueprint-work/g1-cuda-runtime-source-7c1f528-20260926/native_task_runtime_source_packet.v1.json``,
and that whole directory was deleted from the host, so every G1 registration ends
``g1_shared_source_verification_failed``. Two things survive: the build record beside
it, which is a self-consistent receipt of the same packet, and the content-addressed
copy of the archive that a 2026-09-27 run left in the WAM provider object store for
reuse (``stage_cached_runtime_dependency_object_store``). This puts both back, byte
for byte, so the existing registry binding verifies unchanged.

Every identity is pinned here; the operator door's ``restore-g1-runtime-source``
request carries only ``apply``. A plan reads and writes nothing but its result. An
apply overwrites nothing, starts no compute and moves no money: its only provider
calls are one HEAD and one GET of the retained object.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import signal
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import wam_provider_object_store as object_store
from .control_plane_disk_budget import (
    DEFAULT_RESERVATION_ROOT,
    ControlPlaneDiskBudgetError,
    disk_headroom,
    reserve_control_plane_disk,
)
from .decision_evidence_contracts import canonical_digest
from .native_task_runtime_source_packet import (
    SCHEMA_VERSION as RECEIPT_SCHEMA_VERSION,
    NativeTaskRuntimeSourcePacketError,
    verify_native_task_runtime_source_packet,
)


SCHEMA_VERSION = "native_g1_runtime_source_restore.v1"
RECEIPT_NAME = "native_task_runtime_source_packet.v1.json"
PACKET_NAME = "native_task_runtime_sources.zip"
# The arena runner (adp_isaac_lab_arena_vast.DEFAULT_KEY_PREFIX) cached the packet as a
# runtime dependency; a test pins this prefix to that constant.
OBJECT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/arena-native-control"
ARTIFACT_KIND = "runtime_dependency"
# The G1 lane's cache-fill role: its 4 h ledger TTL outlives the door's 2 h unit limit.
# The hold is caller-exact and binds no workspace, so it adds no footprint sample.
DISK_ROLE = "g1_checkpoint_cache"
DISK_MARGIN_BYTES = 256 * 1024**2
RESERVATION_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT"
CHUNK_BYTES = 8 * 1024**2
DIRECTORY_MODE = 0o750
FILE_MODE = 0o640
OUTCOMES = ("planned", "restored", "already_restored", "refused")
REFUSALS = (
    "g1_runtime_source_build_record_changed",
    "g1_runtime_source_target_conflict",
    "g1_runtime_source_restore_busy",
    "g1_runtime_source_object_key_changed",
    "g1_runtime_source_object_store_unavailable",
    "g1_runtime_source_object_missing",
    "g1_runtime_source_object_size_mismatch",
    "g1_runtime_source_digest_mismatch",
    "g1_runtime_source_disk_insufficient",
    "g1_runtime_source_download_failed",
    "g1_runtime_source_verification_failed",
    "g1_runtime_source_restore_failed",
)


@dataclass(frozen=True)
class Pins:
    """Everything the restore is allowed to touch or accept, fixed in code."""

    work_root: Path
    build_record: Path
    build_record_sha256: str
    build_record_size_bytes: int
    receipt_digest: str
    target_dir: Path
    packet_sha256: str
    packet_size_bytes: int
    runtime_profile: str
    #: sha256 of the object key string, as the 09-27 run's cache record names it.
    object_key_sha256: str

    @property
    def receipt_path(self) -> Path:
        return self.target_dir / RECEIPT_NAME

    @property
    def packet_path(self) -> Path:
        return self.target_dir / PACKET_NAME


PINS = Pins(
    work_root=Path("/mnt/blueprint-work"),
    build_record=Path("/mnt/blueprint-work/g1-cuda-runtime-source-7c1f528-20260926-build.json"),
    build_record_sha256="sha256:b303f3017e799a36a2a467de60f235109b1d810a1ef3a87d8d92feb02e6ca5d2",
    build_record_size_bytes=44_943,
    receipt_digest="sha256:50418d4ed677eeedfd64ed8de7b62384e9cd024581e1fc25277ff0e0aabde6bc",
    target_dir=Path("/mnt/blueprint-work/g1-cuda-runtime-source-7c1f528-20260926"),
    packet_sha256="sha256:8128c868951113b5376b9bf9f79fb7c81d5c19f146f99fef90aae7289395674d",
    packet_size_bytes=4_638_507_022,
    runtime_profile="unitree_g1",
    object_key_sha256="561630a41910cf00dc8f4a0073670516e96befe1f7841a77cb56dad2b4be254b",
)


class RestoreRefused(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _disk_usage(path: str | os.PathLike[str]) -> Any:
    return shutil.disk_usage(path)


def _reservation_root() -> Path:
    return Path(os.environ.get(RESERVATION_ROOT_ENV) or DEFAULT_RESERVATION_ROOT)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise RestoreRefused(code)


def _no_symlink_parents(path: Path) -> bool:
    return path.is_absolute() and not any(parent.is_symlink() for parent in path.parents)


def read_build_record(pins: Pins) -> bytes:
    """The build record's exact bytes, once they match every pinned identity."""

    code = "g1_runtime_source_build_record_changed"
    _require(_no_symlink_parents(pins.build_record), code)
    try:
        fd = os.open(pins.build_record, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as error:
        raise RestoreRefused(code) from error
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_size == pins.build_record_size_bytes, code)
        raw = stream.read(pins.build_record_size_bytes + 1)
    _require(len(raw) == pins.build_record_size_bytes, code)
    _require("sha256:" + hashlib.sha256(raw).hexdigest() == pins.build_record_sha256, code)
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise RestoreRefused(code) from error
    _require(isinstance(record, dict), code)
    _require(record.get("schema_version") == RECEIPT_SCHEMA_VERSION
             and record.get("receipt_digest") == pins.receipt_digest
             and canonical_digest(record, digest_field="receipt_digest") == pins.receipt_digest
             and record.get("packet_path") == str(pins.packet_path)
             and record.get("packet_sha256") == pins.packet_sha256
             and record.get("packet_size_bytes") == pins.packet_size_bytes
             and record.get("runtime_profile") == pins.runtime_profile, code)
    return raw


def object_key(pins: Pins) -> str:
    """The retained copy's key, by the cache's own scheme; never recorded."""

    key = object_store.cached_dependency_object_key(OBJECT_KEY_PREFIX, pins.packet_sha256, ARTIFACT_KIND)
    _require(hashlib.sha256(key.encode("utf-8")).hexdigest() == pins.object_key_sha256,
             "g1_runtime_source_object_key_changed")
    return key


def _open_directory(path: Path, *, dir_fd: int | None = None) -> int:
    name: str | Path = path.name if dir_fd is not None else path
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)


def _open_work_root(pins: Pins) -> int:
    _require(_no_symlink_parents(pins.work_root), "g1_runtime_source_target_conflict")
    try:
        return _open_directory(pins.work_root)
    except OSError as error:
        raise RestoreRefused("g1_runtime_source_target_conflict") from error


def _open_target(work_fd: int, pins: Pins) -> int | None:
    """The target directory's descriptor, or None while it does not exist."""

    try:
        info = os.stat(pins.target_dir.name, dir_fd=work_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    _require(stat.S_ISDIR(info.st_mode), "g1_runtime_source_target_conflict")
    try:
        return _open_directory(pins.target_dir, dir_fd=work_fd)
    except OSError as error:
        raise RestoreRefused("g1_runtime_source_target_conflict") from error


def _file_state(target_fd: int | None, name: str, *, size: int, sha256: str) -> str:
    """``absent``, ``matches`` or ``conflict`` (anything else at that name)."""

    if target_fd is None:
        return "absent"
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=target_fd)
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "conflict"
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            return "conflict"
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return "matches" if "sha256:" + digest.hexdigest() == sha256 else "conflict"


def _target_states(target_fd: int | None, pins: Pins) -> dict[str, str]:
    states = {
        "receipt": _file_state(target_fd, RECEIPT_NAME, size=pins.build_record_size_bytes,
                               sha256=pins.build_record_sha256),
        "packet": _file_state(target_fd, PACKET_NAME, size=pins.packet_size_bytes, sha256=pins.packet_sha256),
    }
    _require("conflict" not in states.values(), "g1_runtime_source_target_conflict")
    return states


def _store() -> tuple[Any, str]:
    """A client for the WAM provider store, configured exactly as its staging is."""

    access, secret, endpoint, bucket, region = object_store._staging_store_values(
        access_key_id_file=None, secret_access_key_file=None, endpoint_url="", endpoint_url_file=None,
        bucket="", bucket_file=None, region="", region_file=None)
    _require(bool(access and secret and bucket), "g1_runtime_source_object_store_unavailable")
    try:
        client = object_store._s3_client(access_key=access, secret_key=secret, endpoint=endpoint, region=region)
    except Exception as error:  # noqa: BLE001 - a typed refusal, never the credential
        raise RestoreRefused("g1_runtime_source_object_store_unavailable") from error
    return client, bucket


def _head(client: Any, bucket: str, key: str, pins: Pins) -> str:
    """Check the retained object's size and recorded sha256; return its ETag for the GET."""

    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as error:  # noqa: BLE001 - provider shape varies
        if object_store._s3_object_missing(error):
            raise RestoreRefused("g1_runtime_source_object_missing") from error
        raise RestoreRefused("g1_runtime_source_object_store_unavailable") from error
    head = dict(head) if isinstance(head, Mapping) else {}
    size = head.get("ContentLength")
    _require(type(size) is int and size == pins.packet_size_bytes, "g1_runtime_source_object_size_mismatch")
    metadata = head.get("Metadata")
    recorded = metadata.get("sha256") if isinstance(metadata, Mapping) else None
    # The cache writes its sha256 as object metadata on upload and checks it on every reuse.
    _require(recorded == pins.packet_sha256.removeprefix("sha256:"), "g1_runtime_source_digest_mismatch")
    etag = head.get("ETag")
    return etag if isinstance(etag, str) else ""


def _unlink_quietly(name: str, dir_fd: int) -> None:
    try:
        os.unlink(name, dir_fd=dir_fd)
    except FileNotFoundError:
        pass


def _publish(target_fd: int, temporary: str, final: str, written: list[str]) -> None:
    """Give the verified temporary file its name; a name that already exists is never replaced."""

    try:
        os.link(temporary, final, src_dir_fd=target_fd, dst_dir_fd=target_fd, follow_symlinks=False)
        written.append(final)
    except FileExistsError as error:
        raise RestoreRefused("g1_runtime_source_target_conflict") from error
    finally:
        _unlink_quietly(temporary, target_fd)
    os.fsync(target_fd)


def _download(client: Any, bucket: str, key: str, etag: str, target_fd: int, pins: Pins) -> str:
    """Stream the object into a private temporary file, hashing as it lands; return its name."""

    temporary = f".{PACKET_NAME}.{secrets.token_hex(8)}.partial"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                 dir_fd=target_fd)
    try:
        digest, size = hashlib.sha256(), 0
        request = {"Bucket": bucket, "Key": key, **({"IfMatch": etag} if etag else {})}
        try:
            response = client.get_object(**request)
        except Exception as error:  # noqa: BLE001 - provider shape varies
            if object_store._s3_object_missing(error):
                raise RestoreRefused("g1_runtime_source_object_missing") from error
            raise RestoreRefused("g1_runtime_source_download_failed") from error
        body = response.get("Body") if isinstance(response, Mapping) else None
        _require(body is not None and callable(getattr(body, "read", None)), "g1_runtime_source_download_failed")
        try:
            for block in iter(lambda: body.read(CHUNK_BYTES), b""):
                size += len(block)
                _require(size <= pins.packet_size_bytes, "g1_runtime_source_object_size_mismatch")
                digest.update(block)
                view = memoryview(block)
                while view:
                    view = view[os.write(fd, view):]
        except RestoreRefused:
            raise
        except Exception as error:  # noqa: BLE001 - a cut or failed transfer
            raise RestoreRefused("g1_runtime_source_download_failed") from error
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        os.fchmod(fd, FILE_MODE)
        os.fsync(fd)
        _require(size == pins.packet_size_bytes, "g1_runtime_source_object_size_mismatch")
        _require("sha256:" + digest.hexdigest() == pins.packet_sha256, "g1_runtime_source_digest_mismatch")
    except BaseException:
        os.close(fd)
        _unlink_quietly(temporary, target_fd)
        raise
    os.close(fd)
    return temporary


def _write_receipt(target_fd: int, raw: bytes, written: list[str]) -> None:
    temporary = f".{RECEIPT_NAME}.{secrets.token_hex(8)}.partial"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                 dir_fd=target_fd)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(fd, view):]
        os.fchmod(fd, FILE_MODE)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        _unlink_quietly(temporary, target_fd)
        raise
    os.close(fd)
    _publish(target_fd, temporary, RECEIPT_NAME, written)


def _verify(pins: Pins, report: dict[str, Any]) -> None:
    try:
        verify_native_task_runtime_source_packet(pins.receipt_path)
    except NativeTaskRuntimeSourcePacketError as error:
        report["verification"] = {"verified": False, "errors": list(error.errors)[:8]}
        raise RestoreRefused("g1_runtime_source_verification_failed") from error
    report["verification"] = {"verified": True}


def _need_bytes(states: Mapping[str, str], pins: Pins) -> int:
    packet = pins.packet_size_bytes if states["packet"] == "absent" else 0
    receipt = pins.build_record_size_bytes if states["receipt"] == "absent" else 0
    return packet + receipt + DISK_MARGIN_BYTES


def _plan_disk(need: int, pins: Pins, report: dict[str, Any]) -> None:
    try:
        available = int(disk_headroom(target_root=pins.work_root, reservation_root=_reservation_root(),
                                      disk_usage=lambda path: _disk_usage(path))["available_bytes"])
    except ControlPlaneDiskBudgetError as error:
        report["disk"] = {"need_bytes": need, "sufficient": False, "detail": str(error).split(":", 1)[0]}
        raise RestoreRefused("g1_runtime_source_disk_insufficient") from error
    report["disk"] = {"need_bytes": need, "available_bytes": available, "sufficient": need <= available}
    _require(need <= available, "g1_runtime_source_disk_insufficient")


def _base_report(pins: Pins, apply: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "mode": "apply" if apply else "plan",
        "outcome": None,
        "code": None,
        "build_record": {"path": str(pins.build_record), "sha256": pins.build_record_sha256,
                         "size_bytes": pins.build_record_size_bytes, "receipt_digest": pins.receipt_digest,
                         "verified": False},
        "targets": {"directory": str(pins.target_dir), "receipt_path": str(pins.receipt_path),
                    "packet_path": str(pins.packet_path), "directory_present": None,
                    "receipt": None, "packet": None},
        # The receipt is the build record's exact bytes, so this is its identity on disk.
        "receipt_sha256": pins.build_record_sha256,
        "receipt_size_bytes": pins.build_record_size_bytes,
        "packet_sha256": pins.packet_sha256,
        "packet_size_bytes": pins.packet_size_bytes,
        "runtime_profile": pins.runtime_profile,
        "object_store": {"artifact_kind": ARTIFACT_KIND, "object_key_sha256": pins.object_key_sha256,
                         "checked": False},
        "restore_needed": None,
        "disk": None,
        "download": {"performed": False},
        "written": [],
        "verification": None,
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
    }


def _check_store(pins: Pins, report: dict[str, Any]) -> tuple[Any, str, str, str]:
    key = object_key(pins)
    client, bucket = _store()
    etag = _head(client, bucket, key, pins)
    report["object_store"].update(checked=True, present=True, size_matches=True, sha256_metadata_matches=True)
    return client, bucket, key, etag


def _plan(pins: Pins, report: dict[str, Any]) -> str:
    work_fd = _open_work_root(pins)
    try:
        target_fd = _open_target(work_fd, pins)
        report["targets"]["directory_present"] = target_fd is not None
        try:
            states = _target_states(target_fd, pins)
        finally:
            if target_fd is not None:
                os.close(target_fd)
    finally:
        os.close(work_fd)
    report["targets"].update(states)
    report["restore_needed"] = states != {"receipt": "matches", "packet": "matches"}
    if report["restore_needed"]:
        _check_store(pins, report)
        _plan_disk(_need_bytes(states, pins), pins, report)
    return "planned"


def _apply(pins: Pins, raw: bytes, report: dict[str, Any]) -> str:
    work_fd = _open_work_root(pins)
    created_dir = False
    target_fd = None
    written: list[str] = []
    try:
        target_fd = _open_target(work_fd, pins)
        report["targets"]["directory_present"] = target_fd is not None
        states = _target_states(target_fd, pins)
        report["targets"].update(states)
        report["restore_needed"] = states != {"receipt": "matches", "packet": "matches"}
        if not report["restore_needed"]:
            _verify(pins, report)
            return "already_restored"
        client, bucket, key, etag = _check_store(pins, report)
        need = _need_bytes(states, pins)
        try:
            reservation = reserve_control_plane_disk(
                DISK_ROLE, target_root=pins.work_root, expected_bytes=need,
                reservation_root=_reservation_root(), disk_usage=lambda path: _disk_usage(path))
        except ControlPlaneDiskBudgetError as error:
            report["disk"] = {"need_bytes": need, "sufficient": False, "detail": str(error).split(":", 1)[0]}
            raise RestoreRefused("g1_runtime_source_disk_insufficient") from error
        report["disk"] = {"need_bytes": need, "available_bytes": reservation.available_bytes,
                          "sufficient": True, "reserved": True}
        disk_outcome = "failed"
        try:
            if target_fd is None:
                try:
                    os.mkdir(pins.target_dir.name, DIRECTORY_MODE, dir_fd=work_fd)
                except FileExistsError as error:
                    raise RestoreRefused("g1_runtime_source_restore_busy") from error
                created_dir = True
                report["targets"]["directory_present"] = True
                target_fd = _open_directory(pins.target_dir, dir_fd=work_fd)
                os.fchmod(target_fd, DIRECTORY_MODE)
                os.fsync(work_fd)
            try:
                fcntl.flock(target_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RestoreRefused("g1_runtime_source_restore_busy") from error
            # Another restore may have finished between the first look and the lock.
            states = _target_states(target_fd, pins)
            report["targets"].update(states)
            if states == {"receipt": "matches", "packet": "matches"}:
                _verify(pins, report)
                disk_outcome = "completed"
                return "already_restored"
            if states["packet"] == "absent":
                temporary = _download(client, bucket, key, etag, target_fd, pins)
                report["download"] = {"performed": True, "size_bytes": pins.packet_size_bytes,
                                      "sha256": pins.packet_sha256}
                _publish(target_fd, temporary, PACKET_NAME, written)
            if states["receipt"] == "absent":
                _write_receipt(target_fd, raw, written)
            report["written"] = [str(pins.target_dir / name) for name in written]
            report["targets"].update(receipt="matches", packet="matches")
            _verify(pins, report)
            disk_outcome = "completed"
        finally:
            reservation.release(outcome=disk_outcome)
        return "restored"
    except BaseException:
        # Remove only what this run created; anything that was already there stays.
        if target_fd is not None:
            for name in reversed(written):
                _unlink_quietly(name, target_fd)
            if written:
                os.fsync(target_fd)
        if created_dir:
            try:
                os.rmdir(pins.target_dir.name, dir_fd=work_fd)
            except OSError:
                pass
        if written or created_dir:
            report["written"] = []
            report["removed_after_refusal"] = [str(pins.target_dir / name) for name in written] + (
                [str(pins.target_dir)] if created_dir else [])
            for name in written:
                report["targets"]["receipt" if name == RECEIPT_NAME else "packet"] = "absent"
            report["targets"]["directory_present"] = not created_dir
        raise
    finally:
        if target_fd is not None:
            os.close(target_fd)
        os.close(work_fd)


def restore(*, apply: bool = False, pins: Pins | None = None) -> dict[str, Any]:
    """Plan (default) or apply the restore; always returns the result document."""

    pins = pins or PINS
    report = _base_report(pins, apply)
    try:
        raw = read_build_record(pins)
        report["build_record"]["verified"] = True
        outcome = _apply(pins, raw, report) if apply else _plan(pins, report)
    except RestoreRefused as refusal:
        report.update(outcome="refused", code=refusal.code)
        return report
    except Exception as error:  # noqa: BLE001 - one fixed code, never the message
        report.update(outcome="refused", code="g1_runtime_source_restore_failed", error_type=type(error).__name__)
        return report
    report["outcome"] = outcome
    if outcome in {"restored", "already_restored"}:
        # Exactly the reference an install-native-g1-policies request names as its runtime_source.
        report["runtime_source"] = {"sha256": pins.build_record_sha256, "size_bytes": pins.build_record_size_bytes}
    return report


def _write_result(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _terminate(_signum: int, _frame: Any) -> None:
    # The door's RuntimeMaxSec sends SIGTERM: unwind so a partial download is removed.
    raise SystemExit(143)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="restore; without this only a plan is made")
    parser.add_argument("--result", type=Path, help="also write the result document here")
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, _terminate)
    document = restore(apply=args.apply)
    if args.result is not None:
        _write_result(args.result, document)
    print(json.dumps(document, sort_keys=True))
    if document["outcome"] != "refused":
        return 0
    return 1 if document["code"] == "g1_runtime_source_restore_failed" else 2


if __name__ == "__main__":
    sys.exit(main())
