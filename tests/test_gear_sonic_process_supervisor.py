from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from blueprint_pipeline import gear_sonic_process_supervisor as supervisor


def _qualification_environment(*, session: str, sequence: int) -> dict[str, str]:
    nonce = f"{session}:attempt_{sequence:04d}"
    return {
        "BLUEPRINT_LAUNCH_SESSION_ID": session,
        "BLUEPRINT_QUALIFICATION_ATTEMPT_SEQUENCE": str(sequence),
        "BLUEPRINT_QUALIFICATION_ATTEMPT_NONCE": nonce,
        "BLUEPRINT_QUALIFICATION_ATTEMPT_NONCE_SHA256": hashlib.sha256(
            nonce.encode("utf-8")
        ).hexdigest(),
    }


def _record(
    *,
    pid: int,
    session: str,
    sequence: int,
    executable: str = supervisor.GEAR_SONIC_EXECUTABLE,
) -> supervisor.ProcessRecord:
    return supervisor.ProcessRecord(
        pid=pid,
        process_group_id=pid,
        start_time_ticks=1000 + pid,
        executable=executable,
        environment=_qualification_environment(session=session, sequence=sequence),
    )


def test_stale_listener_cleanup_terminates_only_older_same_session_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _qualification_environment(session="launch-123", sequence=4)
    stale = _record(pid=2402, session="launch-123", sequence=2)
    scans = iter(([stale], []))
    terminated: list[supervisor.ProcessRecord] = []

    monkeypatch.setattr(
        supervisor,
        "_listener_owner_records",
        lambda _root, _port: list(next(scans)),
    )

    def terminate(records: list[supervisor.ProcessRecord], **_kwargs: object) -> None:
        terminated.extend(records)

    cleaned = supervisor.cleanup_stale_debug_listener(
        environment=current,
        terminate_records=terminate,
    )

    assert cleaned == (2402,)
    assert terminated == [stale]


@pytest.mark.parametrize(
    "owner",
    [
        _record(pid=3301, session="different-launch", sequence=2),
        _record(pid=3302, session="launch-123", sequence=4),
        _record(
            pid=3303,
            session="launch-123",
            sequence=2,
            executable="/usr/bin/unrelated-server",
        ),
    ],
)
def test_stale_listener_cleanup_rejects_unowned_process_before_signalling(
    monkeypatch: pytest.MonkeyPatch,
    owner: supervisor.ProcessRecord,
) -> None:
    monkeypatch.setattr(
        supervisor,
        "_listener_owner_records",
        lambda _root, _port: [owner],
    )
    signalled = False

    def terminate(_records: object, **_kwargs: object) -> None:
        nonlocal signalled
        signalled = True

    with pytest.raises(
        supervisor.GearSonicOwnershipError,
        match="gear_sonic_debug_port_owned_by_unrelated_process",
    ):
        supervisor.cleanup_stale_debug_listener(
            environment=_qualification_environment(session="launch-123", sequence=4),
            terminate_records=terminate,
        )

    assert signalled is False


def test_listener_owner_scan_binds_socket_inode_to_exact_process(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    (proc_root / "net").mkdir(parents=True)
    header = "  sl  local_address rem_address st tx_queue tr retrnsmt uid timeout inode\n"
    listener = (
        "   0: 0100007F:15B5 00000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000 1000 0 4242 1\n"
    )
    (proc_root / "net" / "tcp").write_text(header + listener, encoding="utf-8")
    (proc_root / "net" / "tcp6").write_text(header, encoding="utf-8")
    process = proc_root / "424"
    (process / "fd").mkdir(parents=True)
    os.symlink("socket:[4242]", process / "fd" / "7")
    os.symlink(supervisor.GEAR_SONIC_EXECUTABLE, process / "exe")
    environment = _qualification_environment(session="launch-123", sequence=2)
    (process / "environ").write_bytes(
        b"\0".join(f"{name}={value}".encode() for name, value in environment.items())
        + b"\0"
    )
    stat_fields = ["S", "1", "424", "424", *(["0"] * 15), "98765", "0"]
    (process / "stat").write_text(
        "424 (g1 deploy) " + " ".join(stat_fields) + "\n",
        encoding="utf-8",
    )

    records = supervisor._listener_owner_records(
        proc_root,
        supervisor.GEAR_SONIC_DEBUG_PORT,
    )

    assert records == [
        supervisor.ProcessRecord(
            pid=424,
            process_group_id=424,
            start_time_ticks=98765,
            executable=supervisor.GEAR_SONIC_EXECUTABLE,
            environment=environment,
        )
    ]


def test_process_group_cleanup_rejects_mixed_attempt_before_signalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _record(pid=5001, session="launch-123", sequence=4),
        _record(pid=5002, session="launch-123", sequence=3),
    ]
    monkeypatch.setattr(supervisor, "_group_records", lambda _root, _pgid: records)
    signalled = False

    def terminate(_records: object, **_kwargs: object) -> None:
        nonlocal signalled
        signalled = True

    monkeypatch.setattr(supervisor, "_terminate_records", terminate)
    with pytest.raises(
        supervisor.GearSonicOwnershipError,
        match="gear_sonic_process_attempt_identity_mismatch",
    ):
        supervisor.cleanup_owned_process_group(
            5001,
            environment=_qualification_environment(session="launch-123", sequence=4),
        )

    assert signalled is False
