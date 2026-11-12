"""Canonical capacity is observed, and only proven unspent races may retry."""
from __future__ import annotations

import fcntl
import json

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline import beta_screening_disk_admission as disk_admission
from blueprint_pipeline import vast_launch_slots as slots


def test_three_real_canonical_slots_wait_then_resume_without_touching_lock_bytes(tmp_path, monkeypatch):
    base = tmp_path / "vast.lock"
    monkeypatch.setenv("VAST_LAUNCH_LOCK_FILE", str(base))
    monkeypatch.setenv(slots.MAX_CONCURRENT_PAID_LAUNCHES_ENV, "3")
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    # The work-disk check is its own test below; here the disk has room.
    monkeypatch.setattr(disk_admission, "work_disk_admission", lambda _attempt: {"admitted": True})
    gate = slots.vast_launch_gate_path(base)
    gate.write_text("existing deployment metadata\n")
    paths = slots.vast_launch_lock_paths(base)
    handles = []
    for index, path in enumerate(paths):
        path.write_text(json.dumps({"owner": index}) + "\n")
        handle = path.open("r+")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(handle)
    before = {path: path.read_bytes() for path in [gate, *paths]}
    kwargs = {"plan": {}, "candidate": {}, "attempt_dir": tmp_path, "prepared": {}}
    try:
        assert launch.preflight_candidate_capacity(**kwargs) == {
            "capacity_available": False, "blockers": ["vast_paid_launch_lock_busy"]}
        handles.pop().close()
        assert launch.preflight_candidate_capacity(**kwargs) == {"capacity_available": True, "blockers": []}
        with gate.open("r+") as deployment:
            fcntl.flock(deployment, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert launch.preflight_candidate_capacity(**kwargs) == {
                "capacity_available": False, "blockers": ["vast_release_deployment_in_progress"]}
    finally:
        for handle in handles:
            handle.close()
    assert {path: path.read_bytes() for path in before} == before
    assert not (tmp_path / "allocator_launch_intent.json").exists()


@pytest.mark.parametrize("change", [{}, {"blockers": ["vast_paid_launch_lock_busy", "another_failure"]},
    {"provider_create_attempted": True}, {"status": "failed"}, {"blockers": ["vast_paid_launch_lock_unusable"]}])
def test_only_exact_proven_unallocated_lock_refusal_carries_retry_authority(tmp_path, monkeypatch, change):
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    adapter = {"status": "blocked", "provider_mutations_performed": 0, "provider_create_attempted": False,
               "blockers": ["vast_paid_launch_lock_busy"], **change}
    path = launch._write(tmp_path / "allocator_result.json", adapter)
    if change.get("provider_create_attempted"):
        # A create may have happened: never turn unrelated provider-zero into
        # authority to reissue it. Stop the test at the live reconciliation seam.
        from blueprint_pipeline import team_candidate_quick10_launch as native
        monkeypatch.setattr(native, "_default_provider_zero", lambda: {
            "schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
            "provider_zero_verified": False, "live_instance_count": None, "blockers": ["unproven"]})
    result = launch.reconcile_candidate(plan={}, candidate={}, attempt_dir=tmp_path, prepared={})
    if not change:
        assert result["retryable_capacity_refusal"] and result["no_policy_api_calls_proven"]
        assert result["capacity_refusal_receipt"] == launch._record(path)
        assert result["no_allocation_proven"] and result["provider_allocation_performed"] is False
    else:
        assert result.get("retryable_capacity_refusal") is not True
        assert result.get("no_policy_api_calls_proven") is not True


GIB = 1024 ** 3


class DiskRefused:
    """The real pre-launch check's disk refusal, as ``preflight_candidate_capacity`` returns it."""

    def __init__(self):
        self.room = False

    def __call__(self, **_kwargs):
        if self.room:
            return {"capacity_available": True, "blockers": []}
        return {"capacity_available": False, "blockers": [disk_admission.WORK_DISK_WAIT_BLOCKER], "work_disk": {}}


def test_a_full_work_disk_is_a_wait_in_line_through_the_coordinator_and_resumes_when_room_returns(tmp_path, monkeypatch):
    """Review 26 probe: the disk refusal used to end the row blocked (``capacity_preflight_failed``)."""
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from tests.test_beta_policy_screening_coordinator import Backend, plan, publication

    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    backend, check = Backend(), DiskRefused()
    backend.preflight_candidate_capacity = check
    monkeypatch.setattr(coordinator, "_retry_after_capacity", lambda *_args: "2000-01-01T00:00:00+00:00")
    for _ in range(3):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    rows = coordinator._ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]
    assert backend.launches == 0
    for row in rows.values():
        if row["status"] == "pending":
            continue
        # In line, never ended: the site reads the capacity wait it already knows ("in line").
        assert row["status"] == "preparing" and row["blockers"] == [coordinator.CAPACITY_WAIT_BLOCKER]
        assert row["attempt"]["reservation_status"] == "reserved" and "error" not in row["attempt"]
        assert row["attempt"][disk_admission.WAIT_SINCE_FIELD]
    published = publication(tmp_path)["candidates"]
    assert all(item["status"] in {"preparing", "pending"} for item in published)
    # One alert while any row waits, however many passes and rows: how many, since when.
    posts = []
    for _ in range(3):
        record = disk_admission.alert_disk_waits(tmp_path, webhook_url="https://alerts.invalid/hook",
                                                 poster=lambda url, payload: posts.append(payload))
    assert len(posts) == 1 and record["posted"] is True and record["waiting"] >= 1
    assert posts[0]["alerts"][0]["code"] == disk_admission.ALERT_CODE and "waiting_since" in posts[0]["alerts"][0]
    # Room returns: the same row launches, and the wait is over (the next wait is a new alert).
    check.room = True
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert backend.launches == 1
    launched = [row for row in coordinator._ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"].values()
                if row["status"] not in {"pending", "preparing"}]
    assert launched and all(disk_admission.WAIT_SINCE_FIELD not in row["attempt"] for row in launched)


def test_admission_holds_the_landing_room_so_the_landing_is_never_refused_for_disk(tmp_path, monkeypatch):
    """A state where admission passes and the landing then fits, though another writer fills the disk."""
    from types import SimpleNamespace

    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk
    from blueprint_pipeline.control_plane_disk_ledger import ControlPlaneDiskBudgetError

    ledger = tmp_path / "ledger"
    monkeypatch.setenv(disk_admission.RESERVATION_ROOT_ENV, str(ledger))
    monkeypatch.setenv(disk_admission.LANDING_HOLD_ENV, str(4 * GIB))
    total = 100 * GIB
    disk = {"free": 0}
    usage = lambda _path: SimpleNamespace(total=total, free=disk["free"], used=total - disk["free"])  # noqa: E731
    from blueprint_pipeline.control_plane_disk_budget import floor_bytes

    floor = floor_bytes(total, role=disk_admission.ROLE)
    disk["free"] = floor + 5 * GIB  # room for the 4 GiB hold and 1 GiB more
    job = tmp_path / "job"
    job.mkdir()
    assert disk_admission.work_disk_admission(job, streamed=False, disk_usage=usage)["admitted"] is True
    hold = disk_admission.reserve_landing_hold(job, disk_usage=usage)
    try:
        # Another writer that uses the ledger now sees only 1 GiB: its 2 GiB is refused, the landing room stays.
        with pytest.raises(ControlPlaneDiskBudgetError):
            with reserve_control_plane_disk("evidence_offload", target_root=job, expected_bytes=2 * GIB,
                                            reservation_root=ledger, workload="other", disk_usage=usage):
                pass
        # The zip lands (1.5 GiB of real bytes), then its 2.5 GiB expansion is reserved inside the hold.
        disk["free"] -= int(1.5 * GIB)
        with disk_admission.enclosing(hold.token):
            with reserve_control_plane_disk(disk_admission.ROLE, target_root=job, expected_bytes=int(2.5 * GIB),
                                            reservation_root=ledger, workload="team_archive_extraction",
                                            disk_usage=usage):
                pass
        # Without the hold the same landing would have been refused once the other writer had its 2 GiB.
    finally:
        hold.release()
    assert disk_admission.work_disk_admission(job, streamed=False, disk_usage=usage)["expected_bytes"] == 4 * GIB


def test_the_hosts_present_headroom_admits_a_streamed_website_row_and_holds_a_download_by_its_own_size(
        tmp_path, monkeypatch):
    """Work disk on 2026-11-12 (door ``status``): 20,758,401,024 bytes free, floor 15,823,241,216."""
    from types import SimpleNamespace

    from blueprint_pipeline import policy_canary_output_members as members

    monkeypatch.setenv(disk_admission.RESERVATION_ROOT_ENV, str(tmp_path / "ledger"))
    monkeypatch.delenv(disk_admission.LANDING_HOLD_ENV, raising=False)
    usage = lambda _path: SimpleNamespace(total=316_464_824_320, free=20_758_401_024, used=0)  # noqa: E731
    job = tmp_path / "attempt"
    job.mkdir()
    streamed = disk_admission.work_disk_admission(job, streamed=True, disk_usage=usage)
    assert streamed["admitted"] is True and streamed["expected_bytes"] == GIB  # the session's own forecast hold
    downloaded = disk_admission.work_disk_admission(job, streamed=False, disk_usage=usage)
    # 2 x min(1.25 x 5,088,693,294, 8 GB): 11.85 GiB, more than the 4.6 GiB above the floor today.
    assert downloaded["admitted"] is False and downloaded["expected_bytes"] == 12_721_733_236
    # The pre-launch check asks for the hold of the path the row will take.
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    seen = []
    monkeypatch.setattr(disk_admission, "work_disk_admission",
                        lambda _attempt, *, streamed: seen.append(streamed) or {"admitted": True})
    import blueprint_pipeline.vast_launch_slots as slots_module
    monkeypatch.setattr(slots_module, "inspect_vast_launch_admission", lambda: {"status": "admitted", "blockers": []})
    robot = {"execution_spec": {"backend": "robolab.v1"}}
    for mode, expected in ((members.STREAM, True), (members.DOWNLOAD, False)):
        monkeypatch.setattr(members, "resolve_output_delivery", lambda *_a, mode=mode, **_k: members.OutputDelivery(mode, "explicit"))
        launch.preflight_candidate_capacity(plan={}, candidate=robot, attempt_dir=job, prepared={})
    assert seen == [True, False]
    # A G1 row keeps its own collection hold: no landing check is added for it.
    launch.preflight_candidate_capacity(plan={}, candidate={"execution_spec": {"backend": "native_g1.v1"}},
                                        attempt_dir=job, prepared={})
    assert seen == [True, False]


def test_a_disk_wait_is_bounded_then_held_by_name_with_its_money_released_and_one_alert(tmp_path, monkeypatch):
    """Review 36 (6): a download that cannot land would wait for ever holding its reservation."""
    from datetime import datetime, timedelta, timezone

    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from tests.test_beta_policy_screening_coordinator import Backend, plan

    coordinator.enqueue_screening_campaign(plan(count=1), queue_root=tmp_path)
    backend = Backend()
    backend.preflight_candidate_capacity = DiskRefused()
    monkeypatch.setattr(coordinator, "_retry_after_capacity", lambda *_args: "2000-01-01T00:00:00+00:00")
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    ledger = coordinator._ledger(tmp_path)
    row = ledger["campaigns"]["campaign-1"]["candidates"]["candidate-000"]
    assert row["status"] == "preparing" and row["attempt"]["reservation_status"] == "reserved"
    # Seven hours into the wait (past the six-hour bound):
    row["attempt"][disk_admission.WAIT_SINCE_FIELD] = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
    coordinator._save(tmp_path, ledger)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    row = coordinator._ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]["candidate-000"]
    assert row["status"] == "blocked" and row["blockers"] == [disk_admission.CANNOT_LAND_BLOCKER]
    assert row["attempt"]["reservation_status"] == "released" and backend.launches == 0
    posts = []
    for _ in range(2):
        disk_admission.alert_disk_waits(tmp_path, webhook_url="https://alerts.invalid/hook",
                                        poster=lambda url, payload: posts.append(payload))
    assert [alert["code"] for post in posts for alert in post["alerts"]] == [disk_admission.CANNOT_LAND_ALERT]
