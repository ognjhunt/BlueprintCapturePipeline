# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_disk_budget.py
#   src/blueprint_pipeline/control_plane_disk_footprints.py
#   src/blueprint_pipeline/control_plane_disk_ledger.py
from __future__ import annotations

import json
import os
import re
from collections import namedtuple
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_disk_budget as disk_budget
from blueprint_pipeline import control_plane_disk_footprints as footprints
from blueprint_pipeline.control_plane_disk_budget import (
    ControlPlaneDiskBudgetError,
    disk_headroom,
    reserve_control_plane_disk,
)


Usage = namedtuple("Usage", "total used free")
GIB = 1024**3


def test_preinstalled_group_writable_lock_is_not_rechmodded(
    tmp_path, monkeypatch
) -> None:
    ledger = tmp_path / "ledger"
    ledger.mkdir(mode=0o2770)
    lock = ledger / ".lock"
    lock.touch(mode=0o660)
    lock.chmod(0o660)
    original_chmod = disk_budget.os.chmod

    def reject_lock_chmod(path, mode, **kwargs) -> None:
        if path == lock:
            raise PermissionError("root-owned correct lock must not be rechmodded")
        original_chmod(path, mode, **kwargs)

    monkeypatch.setattr(disk_budget.os, "chmod", reject_lock_chmod)

    reservation = reserve_control_plane_disk(
        "launch_activation",
        target_root=tmp_path,
        expected_bytes=GIB,
        reservation_root=ledger,
        disk_usage=lambda _path: Usage(100 * GIB, 60 * GIB, 40 * GIB),
        now=lambda: 100.0,
        pid_alive=lambda _pid: True,
    )

    assert lock.stat().st_mode & 0o777 == 0o660
    reservation.release()


def test_unsafe_lock_mode_fails_closed_when_owner_rejects_repair(
    tmp_path, monkeypatch
) -> None:
    ledger = tmp_path / "ledger"
    ledger.mkdir(mode=0o2770)
    lock = ledger / ".lock"
    lock.touch(mode=0o640)
    lock.chmod(0o640)
    original_chmod = disk_budget.os.chmod

    def reject_lock_chmod(path, mode, **kwargs) -> None:
        if path == lock:
            raise PermissionError("runtime account cannot chmod root-owned lock")
        original_chmod(path, mode, **kwargs)

    monkeypatch.setattr(disk_budget.os, "chmod", reject_lock_chmod)

    with pytest.raises(
        ControlPlaneDiskBudgetError,
        match="control_plane_disk_budget_lock_mode_invalid:0640",
    ):
        reserve_control_plane_disk(
            "launch_activation",
            target_root=tmp_path,
            expected_bytes=GIB,
            reservation_root=ledger,
            disk_usage=lambda _path: Usage(100 * GIB, 60 * GIB, 40 * GIB),
        )


def test_reservation_accounts_for_live_concurrent_reservations(tmp_path) -> None:
    ledger = tmp_path / "ledger"

    def usage(_path):
        return Usage(100 * GIB, 60 * GIB, 40 * GIB)

    first = reserve_control_plane_disk(
        "launch_activation",
        target_root=tmp_path / "future-output",
        expected_bytes=20 * GIB,
        reservation_root=ledger,
        disk_usage=usage,
        now=lambda: 100.0,
        pid_alive=lambda _pid: True,
    )
    assert first.path.is_file()
    assert first.path.stat().st_mode & 0o777 == 0o640
    with pytest.raises(
        ControlPlaneDiskBudgetError,
        match=(
            r"control_plane_disk_budget_exceeded:launch_activation:"
            r"need_bytes=21474836480:available_bytes=12884901888"
        ),
    ):
        reserve_control_plane_disk(
            "launch_activation",
            target_root=tmp_path,
            expected_bytes=20 * GIB,
            reservation_root=ledger,
            disk_usage=usage,
            now=lambda: 101.0,
            pid_alive=lambda _pid: True,
        )
    first.release()
    assert not first.path.exists()


def test_expired_or_dead_reservations_do_not_consume_headroom(tmp_path) -> None:
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    stale = ledger / "stale.json"
    stale.write_text(
        json.dumps(
            {
                "device": tmp_path.stat().st_dev,
                "pid": 999,
                "expected_bytes": 30 * GIB,
                "expires_at_epoch": 50,
            }
        ),
        encoding="utf-8",
    )
    reservation = reserve_control_plane_disk(
        "launch_activation",
        target_root=tmp_path,
        expected_bytes=20 * GIB,
        reservation_root=ledger,
        disk_usage=lambda _path: Usage(100 * GIB, 60 * GIB, 40 * GIB),
        now=lambda: 100.0,
        pid_alive=lambda _pid: False,
    )
    assert reservation.reserved_bytes == 0
    assert not stale.exists()
    reservation.release()


def test_headroom_projects_refused_roles_without_paths(tmp_path) -> None:
    report = disk_headroom(
        target_root=tmp_path / "not-created",
        reservation_root=tmp_path / "ledger",
        disk_usage=lambda _path: Usage(100 * GIB, 93 * GIB, 7 * GIB),
        now=lambda: 100.0,
        pid_alive=lambda _pid: True,
    )
    assert report["status"] == "exhausted"
    assert set(report["refused_roles"]) == {
        "control_plane_deploy",
        "launch_preparation",
        "episode_compilation",
        "launch_activation",
        "launch_dispatch",
        "policy_canary_dispatch",
        "evidence_offload",
        "result_artifact_download",
        "stage_replay",
        "semantic_pretraining",
        "cpu_prestage",
    }
    assert str(tmp_path) not in json.dumps(report)


def test_environment_overrides_floor_and_role_footprint(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES", str(GIB))
    monkeypatch.setenv(
        "BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_LAUNCH_ACTIVATION_BYTES",
        str(3 * GIB),
    )
    reservation = reserve_control_plane_disk(
        "launch_activation",
        target_root=tmp_path,
        reservation_root=tmp_path / "ledger",
        disk_usage=lambda _path: Usage(10 * GIB, 5 * GIB, 5 * GIB),
        now=lambda: 100.0,
        pid_alive=lambda _pid: True,
    )
    assert reservation.floor_bytes == GIB
    assert reservation.expected_bytes == 3 * GIB
    reservation.release()


def test_invalid_environment_override_fails_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES", "unknown")
    with pytest.raises(
        ControlPlaneDiskBudgetError,
        match="control_plane_disk_budget_configuration_invalid",
    ):
        disk_headroom(
            target_root=tmp_path,
            reservation_root=tmp_path / "ledger",
        )


MIB = 1024**2


def _samples(ledger, role, values, outcome="completed"):
    for value in values:
        assert disk_budget.record_footprint_sample(
            reservation_root=ledger, role=role, observed_bytes=value,
            reserved_bytes=2 * GIB, outcome=outcome, now=lambda: 50.0)


def test_short_history_keeps_the_declared_constant(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 9)
    measured = disk_budget.measured_footprint("launch_activation", reservation_root=ledger)
    assert measured == {"role": "launch_activation", "bytes": 2 * GIB, "basis": "declared_default",
                        "sample_count": 9, "declared_bytes": 2 * GIB, "p95_bytes": None}


def test_p95_times_headroom_is_used_once_ten_samples_exist(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 9 + [400 * MIB])
    measured = disk_budget.measured_footprint("launch_activation", reservation_root=ledger)
    assert measured["basis"] == "measured_p95"
    assert measured["p95_bytes"] == 400 * MIB
    assert measured["bytes"] == 500 * MIB


def test_sample_above_the_clamp_cannot_raise_the_reservation(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [10 * GIB] * 12)
    assert disk_budget.effective_footprint_bytes("launch_activation", reservation_root=ledger) == 2 * GIB


def test_tiny_samples_are_floored_and_failed_samples_ignored(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [1] * 10)
    _samples(ledger, "launch_activation", [9 * GIB] * 5, outcome="failed")
    assert disk_budget.effective_footprint_bytes("launch_activation", reservation_root=ledger) == 64 * MIB


def test_unreadable_history_falls_back_to_declared(tmp_path):
    ledger = tmp_path / "ledger"
    (ledger / "history").mkdir(parents=True)
    (ledger / "history" / "launch_activation.jsonl").write_text("{not json\n" * 20)
    assert disk_budget.measured_footprint("launch_activation", reservation_root=ledger)["basis"] == "declared_default"


def test_history_is_compacted(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [MIB] * 400)
    lines = (ledger / "history" / "launch_activation.jsonl").read_text().splitlines()
    assert len(lines) <= 400 and len(lines) >= disk_budget.HISTORY_MAX_LINES


def test_live_reservations_filter_device_and_dead_pids(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    for name, device, pid, expires in (("a", 1, 11, 999.0), ("b", 2, 11, 999.0), ("c", 1, 12, 999.0), ("d", 1, 11, 1.0)):
        (ledger / f"{name}.json").write_text(json.dumps({"device": device, "pid": pid, "expected_bytes": GIB,
                                                         "expires_at_epoch": expires}))
    assert disk_budget.live_reservations(ledger, device=1, now=100.0, pid_alive=lambda pid: pid == 11) == (GIB, 1)
    assert sorted(p.name for p in ledger.glob("*.json")) == ["a.json", "b.json", "c.json", "d.json"]


def test_release_records_the_workspace_peak_in_unique_inodes(tmp_path):
    ledger, work = tmp_path / "ledger", tmp_path / "work" / "job-1"
    work.mkdir(parents=True)
    (work / "preexisting.bin").write_bytes(b"p" * 8192)  # part of the baseline
    reservation = reserve_control_plane_disk(
        "launch_activation", target_root=tmp_path, reservation_root=ledger, workspace=work,
        disk_usage=lambda _p: Usage(100 * GIB, 10 * GIB, 90 * GIB), now=lambda: 100.0,
        pid_alive=lambda _pid: True)
    (work / "new.bin").write_bytes(b"n" * 200_000)
    os.link(work / "new.bin", work / "new-link.bin")        # counted once
    reservation.release()
    rows = [json.loads(line) for line in (ledger / "history" / "launch_activation.jsonl").read_text().splitlines()]
    assert rows[-1]["outcome"] == "completed"
    assert 200_000 <= rows[-1]["observed_bytes"] < 200_000 + 64 * 1024


def test_context_manager_marks_a_failed_run(tmp_path):
    ledger, work = tmp_path / "ledger", tmp_path / "job"
    with pytest.raises(RuntimeError):
        with reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
                                        workspace=work, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB),
                                        now=lambda: 1.0, pid_alive=lambda _pid: True):
            raise RuntimeError("boom")
    rows = (ledger / "history" / "launch_activation.jsonl").read_text().splitlines()
    assert json.loads(rows[-1])["outcome"] == "failed"


def test_receipt_names_the_basis_and_sample_count(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 10)
    reservation = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0, pid_alive=lambda _pid: True)
    receipt = reservation.receipt()
    assert receipt["expected_bytes"] == 125 * MIB
    assert receipt["footprint_basis"] == "measured_p95" and receipt["footprint_sample_count"] == 10
    exact = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        expected_bytes=GIB, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0,
        pid_alive=lambda _pid: True)
    assert exact.receipt()["footprint_basis"] == "caller_exact"


def test_measurement_failure_never_breaks_release(tmp_path, monkeypatch):
    ledger, work = tmp_path / "ledger", tmp_path / "job"
    reservation = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        workspace=work, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0,
        pid_alive=lambda _pid: True)
    monkeypatch.setattr(disk_budget, "record_footprint_sample", lambda **_k: (_ for _ in ()).throw(OSError("full")))
    reservation.release()
    assert not reservation.path.exists()


def test_long_roles_outlive_the_default_ttl(tmp_path):
    ledger = tmp_path / "ledger"
    reservation = reserve_control_plane_disk("cpu_prestage", target_root=tmp_path, reservation_root=ledger,
        expected_bytes=GIB, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 0.0,
        pid_alive=lambda _pid: True)
    entry = json.loads(reservation.path.read_text())
    assert entry["expires_at_epoch"] == 12 * 3600


def test_headroom_refuses_roles_by_their_measured_footprint(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 10)
    def usage(_p):
        return Usage(100 * GIB, 0, 8 * GIB + GIB)  # 1 GiB above the 8 GiB floor

    headroom = disk_headroom(target_root=tmp_path, reservation_root=ledger, disk_usage=usage,
                             now=lambda: 1.0, pid_alive=lambda _pid: True)
    assert "launch_activation" not in headroom["refused_roles"]
    assert "launch_preparation" in headroom["refused_roles"]
    assert headroom["footprints"]["launch_activation"]["basis"] == "measured_p95"


def test_recording_never_waits_forever_on_a_held_ledger_lock(tmp_path, monkeypatch):
    # An evictor runs under the exclusive admission lock; if it released a
    # measured reservation, an unbounded wait here would stall every worker.
    import fcntl

    ledger, work = tmp_path / "ledger", tmp_path / "job"
    reservation = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        workspace=work, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0,
        pid_alive=lambda _pid: True)
    monkeypatch.setattr(footprints, "HISTORY_LOCK_WAIT_SECONDS", 0.2)
    with (ledger / ".lock").open("a+b") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        assert disk_budget.record_footprint_sample(
            reservation_root=ledger, role="launch_activation", observed_bytes=1, reserved_bytes=GIB) is False
        reservation.release()
    assert not reservation.path.exists()
    assert not (ledger / "history" / "launch_activation.jsonl").exists()


@pytest.mark.parametrize("outcome", ["failed", "blocked"])
def test_released_outcome_is_recorded_and_only_completed_shapes_admission(tmp_path, outcome):
    ledger = tmp_path / "ledger"
    for index in range(10):
        reservation = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
            workspace=tmp_path / f"job-{index}", disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB),
            now=lambda: 1.0, pid_alive=lambda _pid: True)
        reservation.release(outcome=outcome)
        reservation.release(outcome="completed")  # idempotent: the first outcome stands
    rows = [json.loads(line) for line in (ledger / "history" / "launch_activation.jsonl").read_text().splitlines()]
    assert [row["outcome"] for row in rows] == [outcome] * 10
    measured = disk_budget.measured_footprint("launch_activation", reservation_root=ledger)
    assert (measured["basis"], measured["bytes"], measured["sample_count"]) == ("declared_default", 2 * GIB, 0)


def test_an_unknown_outcome_never_counts_as_completed(tmp_path):
    ledger = tmp_path / "ledger"
    assert disk_budget.record_footprint_sample(reservation_root=ledger, role="launch_activation",
        observed_bytes=1, reserved_bytes=GIB, outcome="finished") is False
    reservation = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        workspace=tmp_path / "job", disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0,
        pid_alive=lambda _pid: True)
    reservation.release(outcome="finished")
    [row] = [json.loads(line) for line in (ledger / "history" / "launch_activation.jsonl").read_text().splitlines()]
    assert row["outcome"] == "failed"


def _roomy_reservation(tmp_path, ledger, **kwargs):
    return reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0, pid_alive=lambda _pid: True,
        **kwargs)


def _history_rows(ledger, role="launch_activation"):
    return [json.loads(line) for line in (ledger / "history" / f"{role}.jsonl").read_text().splitlines()]


def test_a_pass_over_an_already_full_workspace_records_resumed_not_completed(tmp_path):
    # A resumed pass starts from a workspace that already holds the run, so its
    # growth is near zero; counted as completed it would collapse the p95.
    ledger, work = tmp_path / "ledger", tmp_path / "job"
    work.mkdir()
    (work / "run.bin").write_bytes(b"r" * (footprints.FRESH_WORKSPACE_MAX_BYTES + 64 * 1024))
    reservation = _roomy_reservation(tmp_path, ledger, workspace=work)
    (work / "resume.log").write_bytes(b"l" * 4096)
    reservation.release()
    [row] = _history_rows(ledger)
    assert (row["outcome"], row["fresh"]) == ("resumed", False)
    assert row["baseline_bytes"] > footprints.FRESH_WORKSPACE_MAX_BYTES
    assert disk_budget.measured_footprint("launch_activation", reservation_root=ledger)["sample_count"] == 0


def test_a_fresh_workspace_records_completed_with_its_baseline(tmp_path):
    ledger = tmp_path / "ledger"
    reservation = _roomy_reservation(tmp_path, ledger, workspace=tmp_path / "absent-until-run")
    reservation.release()
    [row] = _history_rows(ledger)
    assert (row["outcome"], row["fresh"], row["baseline_bytes"]) == ("completed", True, 0)


def test_callers_can_assert_or_deny_freshness(tmp_path):
    ledger, work = tmp_path / "ledger", tmp_path / "job"
    work.mkdir()
    (work / "leftover.bin").write_bytes(b"x" * (footprints.FRESH_WORKSPACE_MAX_BYTES + 64 * 1024))
    cleared = _roomy_reservation(tmp_path, ledger)
    cleared.bind_workspace(work, fresh=True)  # e.g. measured right after the caller cleared it
    cleared.release()
    resumed = _roomy_reservation(tmp_path, ledger, workspace=tmp_path / "empty", fresh=False)
    resumed.release()
    assert [(row["outcome"], row["fresh"]) for row in _history_rows(ledger)] == [
        ("completed", True), ("resumed", False)]
    assert disk_budget.record_footprint_sample(reservation_root=ledger, role="launch_activation",
        observed_bytes=1, reserved_bytes=GIB, fresh=False)
    assert _history_rows(ledger)[-1]["outcome"] == "resumed"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-0000 directory anyway")
def test_an_unreadable_subtree_makes_the_measurement_incomplete(tmp_path):
    ledger, work = tmp_path / "ledger", tmp_path / "job"
    reservation = _roomy_reservation(tmp_path, ledger, workspace=work)
    hidden = work / "hidden"
    hidden.mkdir(parents=True)
    (hidden / "payload.bin").write_bytes(b"h" * 8192)
    hidden.chmod(0)
    try:
        reservation.release()
    finally:
        hidden.chmod(0o700)
    [row] = _history_rows(ledger)
    # Bytes under the unreadable directory were not counted; the sample must not count either.
    assert row["outcome"] == "incomplete"
    assert disk_budget.measured_footprint("launch_activation", reservation_root=ledger)["sample_count"] == 0


def test_the_footprint_is_the_largest_workload_p95_not_a_blend(tmp_path):
    # Two scene attempts among 48 small preparations vanish in a blended p95,
    # yet each needs its own 1.2 GiB.
    ledger = tmp_path / "ledger"
    for workload, value, count in (("prepared_references", 100 * MIB, 48),
                                   ("scene_preparation_attempt", 1200 * MIB, 2)):
        for _ in range(count):
            assert disk_budget.record_footprint_sample(reservation_root=ledger, role="launch_preparation",
                workload=workload, observed_bytes=value, reserved_bytes=2 * GIB)
    measured = disk_budget.measured_footprint("launch_preparation", reservation_root=ledger)
    assert (measured["basis"], measured["sample_count"]) == ("measured_p95", 50)
    assert measured["p95_bytes"] == 1200 * MIB and measured["bytes"] == 1500 * MIB


def test_a_job_that_declares_more_than_the_footprint_reserves_what_it_declares(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 10)  # measured: 125 MiB
    declared = _roomy_reservation(tmp_path, ledger, minimum_bytes=900 * MIB)
    assert declared.expected_bytes == 900 * MIB
    assert declared.receipt()["footprint_basis"] == "declared_minimum"
    measured = _roomy_reservation(tmp_path, ledger, minimum_bytes=10 * MIB)
    assert measured.expected_bytes == 125 * MIB
    assert measured.receipt()["footprint_basis"] == "measured_p95"


def test_workload_names_are_always_valid_labels():
    assert footprints.workload_name("activation", "native_task_arena_construction") == (
        "activation_native_task_arena_construction")
    odd = footprints.workload_name("activation", "Lane-With.Odd Chars/" + "x" * 80)
    assert disk_budget._ROLE_RE.fullmatch(odd) and odd.startswith("activation_lane_with_odd_chars")


# The systemd units whose jobs hold each role's reservations.  A job holds its
# reservation for at most its unit's start timeout, so the ledger entry's TTL
# must outlive that timeout or a running job's reservation is deleted as stale.
# control_plane_deploy is run by an operator (no unit), and
# result_artifact_download by the long-running intake service (no start timeout).
ROLE_WORKER_UNITS = {
    "launch_preparation": ("blueprint-task-evaluation-launch-preparation.service",
                           "blueprint-task-evaluation-scene-progression.service"),
    "episode_compilation": ("blueprint-task-evaluation-episode-compilation.service",),
    "launch_activation": ("blueprint-task-evaluation-launch-activation.service",),
    "launch_dispatch": ("blueprint-task-evaluation-launch-dispatcher.service",),
    "policy_canary_dispatch": ("blueprint-task-evaluation-policy-canary-dispatcher.service",),
    "evidence_offload": ("blueprint-control-plane-storage-gc.service",),
    "stage_replay": ("blueprint-agent-stage-replay.service",),
    "semantic_pretraining": ("blueprint-task-evaluation-launch-dispatcher.service",
                             "blueprint-task-evaluation-launch-activation.service"),
    "cpu_prestage": ("blueprint-task-evaluation-launch-dispatcher.service",),
}


def _start_timeout_seconds(unit_text):
    [value] = re.findall(r"^TimeoutStartSec=(\S+)\s*$", unit_text, flags=re.MULTILINE)[-1:] or [None]
    match = re.fullmatch(r"(\d+)(h|min|m|s)?", value or "")
    assert match, f"unparseable TimeoutStartSec={value}"
    return int(match[1]) * {"h": 3600, "min": 60, "m": 60, "s": 1}[match[2] or "s"]


def test_every_role_ttl_outlives_its_worker_units_start_timeout():
    units = Path(__file__).resolve().parents[1] / "deploy" / "systemd"
    assert set(ROLE_WORKER_UNITS) | {"control_plane_deploy", "result_artifact_download"} == set(
        disk_budget.ROLE_FOOTPRINT_BYTES)
    for role, names in ROLE_WORKER_UNITS.items():
        ttl = disk_budget.ROLE_TTL_SECONDS.get(role, disk_budget.DEFAULT_TTL_SECONDS)
        for name in names:
            timeout = _start_timeout_seconds((units / name).read_text(encoding="utf-8"))
            assert ttl >= timeout, f"{role} TTL {ttl}s < {name} TimeoutStartSec {timeout}s"
