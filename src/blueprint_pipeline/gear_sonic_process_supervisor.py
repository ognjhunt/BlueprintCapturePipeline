"""Own and clean the GEAR-SONIC controller process tree.

The official deployment shell starts ``just`` and the compiled controller as
descendants.  Killing only the deployment shell can therefore leave the
controller alive with its ``g1_debug`` ZMQ listener bound to TCP port 5557.
This module gives that tree a dedicated session and only terminates processes
whose inherited launch/qualification identity proves that they belong to the
current evaluation.
"""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


GEAR_SONIC_DEBUG_PORT = 5557
GEAR_SONIC_EXECUTABLE = (
    "/opt/wbc/gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
)
TERM_GRACE_SECONDS = 1.0
KILL_GRACE_SECONDS = 1.0
POLL_SECONDS = 0.05


class GearSonicOwnershipError(RuntimeError):
    """Raised before any unrelated or unverifiable process is signalled."""


@dataclass(frozen=True)
class AttemptIdentity:
    launch_session_id: str
    qualification_sequence: int | None
    qualification_nonce: str
    qualification_nonce_sha256: str
    single_episode_attempt_id: str


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    process_group_id: int
    start_time_ticks: int
    executable: str
    environment: Mapping[str, str]


def _qualification_nonce(launch_session_id: str, sequence: int) -> str:
    return f"{launch_session_id}:attempt_{sequence:04d}"


def _identity_from_environment(environment: Mapping[str, str]) -> AttemptIdentity:
    launch_session_id = str(environment.get("BLUEPRINT_LAUNCH_SESSION_ID") or "").strip()
    if not launch_session_id:
        raise GearSonicOwnershipError("gear_sonic_launch_session_identity_missing")
    sequence_text = str(
        environment.get("BLUEPRINT_QUALIFICATION_ATTEMPT_SEQUENCE") or ""
    ).strip()
    nonce = str(environment.get("BLUEPRINT_QUALIFICATION_ATTEMPT_NONCE") or "").strip()
    nonce_sha256 = str(
        environment.get("BLUEPRINT_QUALIFICATION_ATTEMPT_NONCE_SHA256") or ""
    ).strip()
    qualification_values_present = tuple(bool(value) for value in (sequence_text, nonce, nonce_sha256))
    if any(qualification_values_present) and not all(qualification_values_present):
        raise GearSonicOwnershipError("gear_sonic_qualification_identity_incomplete")
    sequence: int | None = None
    if all(qualification_values_present):
        try:
            sequence = int(sequence_text)
        except ValueError as exc:
            raise GearSonicOwnershipError(
                "gear_sonic_qualification_sequence_invalid"
            ) from exc
        if sequence < 1:
            raise GearSonicOwnershipError("gear_sonic_qualification_sequence_invalid")
        expected_nonce = _qualification_nonce(launch_session_id, sequence)
        expected_sha256 = hashlib.sha256(expected_nonce.encode("utf-8")).hexdigest()
        if nonce != expected_nonce or nonce_sha256 != expected_sha256:
            raise GearSonicOwnershipError("gear_sonic_qualification_nonce_invalid")
    attempt_id = str(
        environment.get("BLUEPRINT_SINGLE_EPISODE_ATTEMPT_ID") or ""
    ).strip()
    if sequence is None and not attempt_id:
        raise GearSonicOwnershipError("gear_sonic_attempt_identity_missing")
    return AttemptIdentity(
        launch_session_id=launch_session_id,
        qualification_sequence=sequence,
        qualification_nonce=nonce,
        qualification_nonce_sha256=nonce_sha256,
        single_episode_attempt_id=attempt_id,
    )


def _read_environment(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        raw_name, raw_value = item.split(b"=", 1)
        result[os.fsdecode(raw_name)] = os.fsdecode(raw_value)
    return result


def _parse_process_stat(raw: str) -> tuple[int, int]:
    closing_parenthesis = raw.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("process_stat_comm_missing")
    fields = raw[closing_parenthesis + 2 :].split()
    # After the parenthesized comm field: state is field 3, pgrp is field 5,
    # and starttime is field 22 in proc_pid_stat(5).
    if len(fields) < 20:
        raise ValueError("process_stat_fields_missing")
    return int(fields[2]), int(fields[19])


def _read_process(proc_root: Path, pid: int) -> ProcessRecord:
    process_root = proc_root / str(pid)
    process_group_id, start_time_ticks = _parse_process_stat(
        (process_root / "stat").read_text(encoding="utf-8")
    )
    executable = os.readlink(process_root / "exe")
    return ProcessRecord(
        pid=pid,
        process_group_id=process_group_id,
        start_time_ticks=start_time_ticks,
        executable=executable.removesuffix(" (deleted)"),
        environment=_read_environment(process_root / "environ"),
    )


def _listener_socket_inodes(proc_root: Path, port: int) -> set[str]:
    port_hex = f"{port:04X}"
    inodes: set[str] = set()
    for relative in ("net/tcp", "net/tcp6"):
        path = proc_root / relative
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[1:]
        except FileNotFoundError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            local_address = fields[1]
            if ":" not in local_address:
                continue
            if local_address.rsplit(":", 1)[1].upper() == port_hex and fields[9] != "0":
                inodes.add(fields[9])
    return inodes


def _socket_owner_pids(proc_root: Path, socket_inodes: set[str]) -> set[int]:
    owners: set[int] = set()
    if not socket_inodes:
        return owners
    expected_links = {f"socket:[{inode}]" for inode in socket_inodes}
    for candidate in proc_root.iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            descriptors = (candidate / "fd").iterdir()
            if any(os.readlink(descriptor) in expected_links for descriptor in descriptors):
                owners.add(int(candidate.name))
        except (FileNotFoundError, NotADirectoryError, PermissionError, ProcessLookupError):
            continue
    return owners


def _listener_owner_records(proc_root: Path, port: int) -> list[ProcessRecord]:
    inodes = _listener_socket_inodes(proc_root, port)
    if not inodes:
        return []
    owner_pids = _socket_owner_pids(proc_root, inodes)
    if not owner_pids:
        raise GearSonicOwnershipError("gear_sonic_debug_listener_owner_unresolved")
    records: list[ProcessRecord] = []
    for pid in sorted(owner_pids):
        try:
            records.append(_read_process(proc_root, pid))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            raise GearSonicOwnershipError(
                "gear_sonic_debug_listener_owner_changed_during_audit"
            ) from None
    return records


def _same_process(proc_root: Path, record: ProcessRecord) -> bool:
    try:
        current = _read_process(proc_root, record.pid)
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return False
    return current.start_time_ticks == record.start_time_ticks


def _record_is_zombie(proc_root: Path, record: ProcessRecord) -> bool:
    try:
        raw = (proc_root / str(record.pid) / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    closing_parenthesis = raw.rfind(")")
    fields = raw[closing_parenthesis + 2 :].split() if closing_parenthesis >= 0 else []
    return bool(fields and fields[0] == "Z")


def _remaining_records(proc_root: Path, records: Sequence[ProcessRecord]) -> list[ProcessRecord]:
    return [
        record
        for record in records
        if _same_process(proc_root, record) and not _record_is_zombie(proc_root, record)
    ]


def _terminate_records(
    records: Sequence[ProcessRecord],
    *,
    proc_root: Path,
    signal_process: Callable[[int, int], None] = os.kill,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if not records:
        return
    if any(not _same_process(proc_root, record) for record in records):
        raise GearSonicOwnershipError("gear_sonic_owned_process_identity_changed")
    for record in records:
        try:
            signal_process(record.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = monotonic() + TERM_GRACE_SECONDS
    remaining = _remaining_records(proc_root, records)
    while remaining and monotonic() < deadline:
        sleep(POLL_SECONDS)
        remaining = _remaining_records(proc_root, records)
    if remaining:
        if any(not _same_process(proc_root, record) for record in remaining):
            raise GearSonicOwnershipError("gear_sonic_owned_process_identity_changed")
        for record in remaining:
            try:
                signal_process(record.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        deadline = monotonic() + KILL_GRACE_SECONDS
        remaining = _remaining_records(proc_root, records)
        while remaining and monotonic() < deadline:
            sleep(POLL_SECONDS)
            remaining = _remaining_records(proc_root, records)
    if remaining:
        raise GearSonicOwnershipError("gear_sonic_owned_process_cleanup_timed_out")


def _require_same_attempt(record: ProcessRecord, current: AttemptIdentity) -> None:
    owner = _identity_from_environment(record.environment)
    if owner.launch_session_id != current.launch_session_id:
        raise GearSonicOwnershipError("gear_sonic_process_launch_session_mismatch")
    if current.qualification_sequence is not None:
        if (
            owner.qualification_sequence != current.qualification_sequence
            or owner.qualification_nonce != current.qualification_nonce
            or owner.qualification_nonce_sha256 != current.qualification_nonce_sha256
        ):
            raise GearSonicOwnershipError("gear_sonic_process_attempt_identity_mismatch")
    elif owner.single_episode_attempt_id != current.single_episode_attempt_id:
        raise GearSonicOwnershipError("gear_sonic_process_attempt_identity_mismatch")


def _require_older_owned_listener(record: ProcessRecord, current: AttemptIdentity) -> None:
    if record.executable != GEAR_SONIC_EXECUTABLE:
        raise GearSonicOwnershipError("gear_sonic_debug_port_owned_by_unrelated_process")
    if current.qualification_sequence is None:
        raise GearSonicOwnershipError("gear_sonic_debug_port_occupied_without_qualification")
    owner = _identity_from_environment(record.environment)
    if (
        owner.launch_session_id != current.launch_session_id
        or owner.qualification_sequence is None
        or owner.qualification_sequence >= current.qualification_sequence
    ):
        raise GearSonicOwnershipError("gear_sonic_debug_port_owned_by_unrelated_process")


def cleanup_stale_debug_listener(
    *,
    environment: Mapping[str, str] = os.environ,
    proc_root: Path = Path("/proc"),
    terminate_records: Callable[..., None] = _terminate_records,
) -> tuple[int, ...]:
    """Terminate only an older, cryptographically attempt-bound port owner."""

    current = _identity_from_environment(environment)
    records = _listener_owner_records(proc_root, GEAR_SONIC_DEBUG_PORT)
    if not records:
        return ()
    for record in records:
        _require_older_owned_listener(record, current)
    terminate_records(records, proc_root=proc_root)
    remaining = _listener_owner_records(proc_root, GEAR_SONIC_DEBUG_PORT)
    if remaining:
        raise GearSonicOwnershipError("gear_sonic_debug_listener_cleanup_failed")
    return tuple(record.pid for record in records)


def _group_records(proc_root: Path, process_group_id: int) -> list[ProcessRecord]:
    records: list[ProcessRecord] = []
    for candidate in proc_root.iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            record = _read_process(proc_root, int(candidate.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        if record.process_group_id == process_group_id:
            records.append(record)
    return records


def cleanup_owned_process_group(
    process_group_id: int,
    *,
    environment: Mapping[str, str] = os.environ,
    proc_root: Path = Path("/proc"),
) -> tuple[int, ...]:
    current = _identity_from_environment(environment)
    records = _group_records(proc_root, process_group_id)
    for record in records:
        _require_same_attempt(record, current)
    _terminate_records(records, proc_root=proc_root)
    return tuple(record.pid for record in records)


def supervise(command: Sequence[str]) -> int:
    """Run the controller in a new session and reap its entire owned tree."""

    if not command:
        raise GearSonicOwnershipError("gear_sonic_supervisor_command_missing")
    current_environment = dict(os.environ)
    stale_pids = cleanup_stale_debug_listener(environment=current_environment)
    if stale_pids:
        print(
            "gear_sonic_stale_listener_cleanup=passed pids="
            + ",".join(str(pid) for pid in stale_pids),
            flush=True,
        )
    requested_signal: int | None = None

    def _request_shutdown(signum: int, _frame: object) -> None:
        nonlocal requested_signal
        requested_signal = signum

    previous_handlers = {
        signum: signal.signal(signum, _request_shutdown)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    child: subprocess.Popen[bytes] | None = None
    try:
        if requested_signal is not None:
            return 128 + requested_signal
        child = subprocess.Popen(list(command), start_new_session=True)
        while True:
            if requested_signal is not None:
                cleanup_owned_process_group(
                    child.pid,
                    environment=current_environment,
                )
                return 128 + requested_signal
            try:
                returncode = child.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                continue
            cleanup_owned_process_group(
                child.pid,
                environment=current_environment,
            )
            return int(returncode)
    finally:
        if child is not None and child.poll() is None:
            cleanup_owned_process_group(
                child.pid,
                environment=current_environment,
            )
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "supervise":
        print("usage: gear_sonic_process_supervisor supervise -- COMMAND...", file=sys.stderr)
        return 64
    command = args[1:]
    if command and command[0] == "--":
        command = command[1:]
    try:
        return supervise(command)
    except GearSonicOwnershipError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 72


if __name__ == "__main__":
    raise SystemExit(main())
