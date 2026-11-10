"""Automatic consent-withdrawal deletion: timer passes with no operator, applying only under a standing approval.

Beta journey step: site withdrawal. Every test runs on the offline drill's synthetic world under a
temporary root, with in-memory B2 and source-bucket fakes; nothing reads or writes /etc/blueprint
or /var/lib/blueprint.
"""
# Covers: src/blueprint_pipeline/website_capture_withdrawal_automatic.py
# Covers: src/blueprint_pipeline/website_capture_withdrawal_deletion.py
# Covers: src/blueprint_pipeline/website_capture_withdrawal_drill.py
from __future__ import annotations

import fcntl
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import website_capture_withdrawal as withdrawal
from blueprint_pipeline import website_capture_withdrawal_automatic as automatic
from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
from blueprint_pipeline import website_capture_withdrawal_drill as drill

ROOT = Path(__file__).resolve().parents[1]
VECTOR = ROOT / "tests/fixtures/website-withdrawal-vector.json"
EXAMPLE = ROOT / "deploy/website-withdrawal-deletion/standing-approval.example.json"
HOUR = 3600.0


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _command(request_id: str) -> dict:
    return {"schema_version": "website_capture_withdrawal.v1", "request_id": request_id,
            "scene_id": f"site-{request_id}", "capture_id": f"walkthrough-{request_id}",
            "withdrawal_id": f"withdrawal-{request_id}", "requested_at_iso": "2026-10-06T00:00:00Z"}


class Harness:
    def __init__(self, tmp_path: Path, *, sites: tuple[str, ...] = ("req1",)):
        self.world = drill.build_world(tmp_path, sites[0])
        self.partition = self.world["partition"]
        self.captures = {sites[0]: self.world["capture"]}
        for request_id in sites[1:]:
            self.captures[request_id], _ = drill._capture(self.partition, request_id)
        for request_id, capture in self.captures.items():
            withdrawal.acknowledge_withdrawal(capture_root=capture, command=_command(request_id))
        self.uncovered = self.world["resolution"]["uncovered_roots"][0]
        # The drill's entry no record attributes stays for the owner (its own test); most tests start without it.
        import shutil
        shutil.rmtree(self.uncovered / "drill-entry")
        self.index_root = tmp_path / "report/site-index"
        self.provider_calls: list[tuple[str, str]] = []
        self.provider_failures: dict[str, Exception] = {}

        def deleter(provider: str):
            def delete(object_id: str):
                self.provider_calls.append((provider, object_id))
                if object_id in self.provider_failures:
                    raise self.provider_failures[object_id]
                return {"deleted": True}
            return delete

        self.resolution = {**self.world["resolution"],
                           "object_store": deletion.B2ObjectVersions(self.world["store"], drill.BUCKET),
                           "cloud_storage": self.world["gcs"], "site_index_root": self.index_root,
                           "provider_copies": True,
                           "provider_deleters": {name: deleter(name) for name in ("gemini", "meta", "world_labs")}}
        self.approval_path = tmp_path / "etc/website-withdrawal-deletion-approval.json"
        self.report_root = tmp_path / "report"
        self.now = time.time()
        self.busy: list[str] = []
        self.posts: list[dict] = []
        self.plans: list[str] = []
        self.applies: list[str] = []
        self.planner = deletion.plan_withdrawal_deletion

    def approve(self, **overrides) -> None:
        value = {"schema_version": automatic.APPROVAL_SCHEMA, "approved_by": "owner (synthetic test)",
                 "approved_at": _iso(self.now - HOUR), "policy_class": automatic.POLICY_CLASS,
                 "cool_off_hours": 0, **overrides}
        self.approval_path.parent.mkdir(parents=True, exist_ok=True)
        self.approval_path.write_text(json.dumps({key: item for key, item in value.items() if item is not None}))
        self.approval_path.chmod(0o644)

    def run(self, offset: float = 0.0, **kwargs) -> dict:
        def planner(**arguments):
            self.plans.append(arguments["capture_root"].name)
            return self.planner(**arguments)

        def applier(**arguments):
            self.applies.append(arguments["capture_root"].name)
            return deletion.apply_withdrawal_deletion(**arguments)

        return automatic.run_pass(
            site_roots=[self.partition], report_root=self.report_root, resolution=lambda: self.resolution,
            quiet=lambda: list(self.busy), approval_path=self.approval_path, approval_owner_uid=os.getuid(),
            now=self.now + offset, webhook_url="https://alerts.invalid/hook",
            poster=lambda url, report: self.posts.append(report), planner=planner, applier=applier,
            **{"site_index_root": self.index_root, "indexed_roots": [self.uncovered], **kwargs})

    def site(self, request_id: str = "req1") -> Path:
        return self.captures[request_id].parents[1]

    def state(self, request_id: str = "req1") -> dict:
        return json.loads((self.site(request_id) / "website_withdrawal" / automatic.STATE_FILENAME).read_text())

    def site_files(self, request_id: str = "req1") -> int:
        return drill._site_files(self.site(request_id))

    def posted_codes(self) -> list[str]:
        return [alert["code"] for report in self.posts for alert in report["alerts"]]


@pytest.fixture()
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def test_a_timer_pass_plans_a_withdrawal_with_no_operator_and_deletes_nothing_without_approval(harness):
    files = harness.site_files()
    summary = harness.run()
    state = harness.state()
    assert harness.plans == ["walkthrough-req1"] and harness.applies == []
    assert state["status"] == "planned" and state["waiting_on"] == ["standing_approval_absent"]
    assert state["unresolved_count"] == 0 and state["money"]["screening_incomplete"] is False
    assert state["uncovered_pipeline_roots"] == [str(harness.world["work"] / "inputs/sam31-preparations")]
    plan = json.loads((harness.site() / "website_withdrawal" / automatic.PLAN_FILENAME).read_text())
    assert plan["digest"] == state["plan_digest"]
    # Nothing deleted: the scene files, every B2 version and the source bucket object are still there.
    assert harness.site_files() == files and harness.world["store"].deleted == []
    assert len(harness.world["gcs"].objects("scenes/site-req1/")) == 1
    assert not (harness.site() / "website_withdrawal/deletion_receipts").exists()
    # The owner sees it waiting only on the approval; an absent approval is not an alert by itself.
    assert summary["approval"]["status"] == "absent"
    assert summary["withdrawals"]["awaiting_owner_approval"] == 1 and summary["alerts"] == []
    assert harness.posts == []
    assert json.loads((harness.report_root / "summary.json").read_text())["withdrawals"]["waiting"] == 1


def test_a_valid_standing_approval_deletes_and_seals_a_receipt_naming_it(harness):
    harness.approve()
    summary = harness.run()
    state = harness.state()
    assert state["status"] == "applied", state.get("last_error")
    assert harness.applies == ["walkthrough-req1"]
    receipt = json.loads((harness.site() / "website_withdrawal/deletion_receipts"
                          / f"{state['plan_digest'][7:]}.json").read_text())
    assert receipt["digest"] == state["receipt_digest"]
    authorization = receipt["authorization"]
    assert authorization["kind"] == "standing_owner_approval"
    assert authorization["approval_digest"] == summary["approval"]["digest"]
    assert authorization["policy_class"] == automatic.POLICY_CLASS
    assert authorization["class_checks"]["money_settled"] is True
    assert receipt["covered_copies_deleted"] is True and harness.site_files() == 0
    # The other site's run, capture and objects, and the tombstone and audit journal, stay.
    assert harness.world["store"].versions_of(harness.world["other_uri"]) == 1
    assert (harness.world["other_capture"] / "raw/video.mp4").is_file()
    assert (harness.site() / "website_withdrawal/tombstone.json").is_file()
    assert summary["withdrawals"]["completed"] == 1

    # A second pass after success does nothing at all.
    again = harness.run(offset=10 * HOUR)
    assert harness.plans == ["walkthrough-req1"] and again["action"] is None
    assert again["withdrawals"] == {**again["withdrawals"], "completed": 1, "waiting": 0}


@pytest.mark.parametrize("override, reason", [
    ({"expires_at": "EXPIRED"}, "approval_expired"),
    ({"policy_class": "website_withdrawal_deletion.everything.v1"}, "approval_policy_class_mismatch"),
    ({"surprise": True}, "approval_unknown_field"),
    ({"cool_off_hours": "72"}, "approval_cool_off_hours_invalid"),
])
def test_an_expired_or_malformed_approval_only_plans_and_alerts(harness, override, reason):
    if override.get("expires_at") == "EXPIRED":
        override = {"expires_at": _iso(harness.now - 60)}
    harness.approve(**override)
    summary = harness.run()
    state = harness.state()
    assert harness.applies == [] and harness.site_files() > 0
    expected = "standing_approval_expired" if reason == "approval_expired" else "standing_approval_invalid"
    assert state["status"] == "planned" and state["waiting_on"] == [expected]
    assert summary["approval"]["reason"] == reason
    assert harness.posted_codes() == ["website_withdrawal_approval_invalid"]
    # Deduplicated: the same alert again within a day does not post again.
    harness.run(offset=1.0)
    assert len(harness.posts) == 1


def test_approval_files_that_are_not_strictly_valid_never_authorize(tmp_path):
    path = tmp_path / "approval.json"
    now = time.time()
    good = {"schema_version": automatic.APPROVAL_SCHEMA, "approved_by": "owner", "approved_at": _iso(now - 60),
            "policy_class": automatic.POLICY_CLASS, "cool_off_hours": 72, "expires_at": None}
    assert automatic.load_approval(path, now=now, owner_uid=os.getuid())["status"] == "absent"
    path.write_text(json.dumps(good))
    path.chmod(0o644)
    loaded = automatic.load_approval(path, now=now, owner_uid=os.getuid())
    assert loaded["status"] == "valid" and loaded["approval"]["cool_off_hours"] == 72
    assert loaded["digest"].startswith("sha256:")
    assert automatic.load_approval(path, now=now, owner_uid=os.getuid() + 1)["reason"] == "approval_owner_unexpected"
    path.chmod(0o664)
    assert automatic.load_approval(path, now=now, owner_uid=os.getuid())["reason"] == "approval_writable_by_others"
    path.chmod(0o644)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    assert automatic.load_approval(link, now=now, owner_uid=os.getuid())["reason"] == "approval_symlink"
    for bad, reason in (({**good, "approved_at": _iso(now + 3600)}, "approval_approved_at_in_future"),
                        ({**good, "approved_at": "2026-11-10T00:00:00"}, "approval_approved_at_invalid"),
                        ({**good, "cool_off_hours": True}, "approval_cool_off_hours_invalid"),
                        ({**good, "cool_off_hours": 721}, "approval_cool_off_hours_invalid"),
                        ({**good, "expires_at": _iso(now - 120)}, "approval_expires_before_approval"),
                        ({**good, "schema_version": "v0"}, "approval_schema_unknown"),
                        ({key: good[key] for key in good if key != "approved_by"}, "approval_field_missing")):
        path.write_text(json.dumps(bad))
        assert automatic.load_approval(path, now=now, owner_uid=os.getuid())["reason"] == reason, reason
    path.write_text("[1]")
    assert automatic.load_approval(path, now=now, owner_uid=os.getuid())["reason"] == "approval_not_an_object"


def test_the_shipped_example_approval_does_not_validate_until_the_owner_fills_it_in(tmp_path):
    example = json.loads(EXAMPLE.read_text())
    path = tmp_path / "approval.json"
    path.write_text(EXAMPLE.read_text())
    path.chmod(0o644)
    now = time.time()
    assert automatic.load_approval(path, now=now, owner_uid=os.getuid())["status"] == "invalid"
    filled = {**example, "approved_by": "Owner Name", "approved_at": _iso(now - 60), "expires_at": None}
    path.write_text(json.dumps(filled))
    assert automatic.load_approval(path, now=now, owner_uid=os.getuid())["status"] == "valid"
    # The deploy installs nothing from that directory.
    deploy_tool = (ROOT / "scripts/deploy_control_plane_commit.py").read_text()
    installer = (ROOT / "scripts/install_live_pipeline_control_plane.sh").read_text()
    assert "website-withdrawal-deletion/" not in deploy_tool and "website-withdrawal-deletion/" not in installer
    assert "website-withdrawal-deletion-approval.json" not in deploy_tool + installer


def test_a_legal_hold_refuses_even_with_a_valid_approval(harness):
    harness.approve()
    hold = harness.site() / "website_withdrawal/legal_hold.json"
    hold.write_text(json.dumps({"legal_hold": True}))
    harness.run()
    assert harness.state()["status"] == "legal_hold" and harness.applies == [] and harness.site_files() > 0
    # A held withdrawal is not overdue: the hold is the reason it stays.
    later = harness.run(offset=30 * 86400.0)
    assert "website_withdrawal_deletion_overdue" not in [alert["code"] for alert in later["alerts"]]
    assert harness.site_files() > 0


def test_an_unresolved_plan_refuses_is_recorded_and_alerts_after_the_bound(harness):
    harness.approve()
    harness.resolution["object_store"] = None  # every B2 object is then unresolved
    harness.run()
    state = harness.state()
    assert state["status"] == "unresolved" and harness.applies == []
    assert state["unresolved_reasons"] == {"website_withdrawal_deletion_object_store_unconfigured": 4}
    assert harness.site_files() > 0 and harness.posts == []
    # Re-planned within the bound: still recorded, still quiet.
    harness.run(offset=7 * HOUR)
    assert harness.state()["blocked_since_epoch"] == harness.now and harness.posts == []
    harness.run(offset=automatic.BLOCKED_ALERT_SECONDS + 1)
    assert "website_withdrawal_deletion_unresolved" in harness.posted_codes()
    assert harness.applies == [] and harness.site_files() > 0


def test_a_live_paid_stage_makes_the_pass_do_nothing(harness):
    harness.approve()
    harness.busy = ["paid_launch_lock_held:vast_paid_launch.lock"]
    summary = harness.run()
    assert harness.plans == [] and summary["action"] is None
    assert not (harness.site() / "website_withdrawal" / automatic.STATE_FILENAME).exists()
    assert summary["host"] == {"quiet": False, "reasons": ["paid_launch_lock_held:vast_paid_launch.lock"]}
    harness.busy = []
    harness.run(offset=60.0)
    assert harness.state()["status"] == "applied"


def test_a_held_paid_launch_slot_or_live_screening_candidates_count_as_busy(tmp_path):
    locks = tmp_path / "provider-locks"
    locks.mkdir()
    slot = locks / "vast_paid_launch.slot1.lock"
    slot.write_text("")
    (locks / "vast_paid_launch.lock").write_text("")
    status = tmp_path / "worker_status.json"
    assert automatic.paid_stage_reasons(provider_lock_dir=locks, screening_status=status) == []
    with slot.open("r") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        info = os.stat(slot)
        holders = {(os.major(info.st_dev), os.minor(info.st_dev), info.st_ino): [os.getpid()]}
        assert automatic.paid_stage_reasons(provider_lock_dir=locks, screening_status=status,
                                            lock_reader=lambda: holders) == [
            "paid_launch_lock_held:vast_paid_launch.slot1.lock"]
    status.write_text(json.dumps({"candidate_counts": {"launching": 0, "running": 2}}))
    assert automatic.paid_stage_reasons(provider_lock_dir=locks, screening_status=status,
                                        lock_reader=lambda: {}) == ["screening_candidates_live"]
    status.write_text("{not json")

    def unreadable():
        raise OSError("synthetic")

    assert automatic.paid_stage_reasons(provider_lock_dir=locks, screening_status=status,
                                        lock_reader=unreadable) == [
        "paid_launch_locks_unknown", "screening_status_unknown"]


def test_two_waiting_sites_are_handled_one_per_pass(tmp_path):
    harness = Harness(tmp_path, sites=("req1", "req2"))
    harness.approve()
    first = harness.run()
    assert harness.plans == ["walkthrough-req1"] and first["withdrawals"]["completed"] == 1
    assert not (harness.site("req2") / "website_withdrawal" / automatic.STATE_FILENAME).exists()
    assert harness.site_files("req2") > 0
    second = harness.run(offset=60.0)
    assert harness.plans == ["walkthrough-req1", "walkthrough-req2"]
    assert harness.state("req2")["status"] == "applied", harness.state("req2").get("last_error")
    assert second["withdrawals"]["completed"] == 2


def test_a_failing_plan_backs_off_and_is_retried_and_alerts_when_it_keeps_failing(harness):
    calls = []

    def failing(**arguments):
        calls.append(arguments)
        raise OSError("synthetic_disk_error")

    harness.planner = failing
    harness.run()
    state = harness.state()
    assert state["status"] == "failed" and state["failure_count"] == 1
    assert state["last_error"] == "synthetic_disk_error"
    assert state["next_attempt_at_epoch"] == harness.now + HOUR
    harness.run(offset=HOUR - 1)
    assert len(calls) == 1  # backing off
    harness.run(offset=HOUR + 1)
    assert len(calls) == 2 and harness.state()["next_attempt_at_epoch"] == harness.now + HOUR + 1 + 4 * HOUR
    assert harness.posts == []
    harness.run(offset=6 * HOUR)
    assert len(calls) == 3 and harness.state()["failure_count"] == 3
    assert "website_withdrawal_deletion_failing" in harness.posted_codes()
    # Never terminal: once the cause is gone the next due pass plans.
    harness.planner = deletion.plan_withdrawal_deletion
    harness.run(offset=31 * HOUR)
    assert harness.state()["status"] == "planned" and harness.state()["failure_count"] == 0


def test_a_pass_killed_inside_a_step_is_recorded_as_a_failure_with_backoff(harness):
    harness.run()
    state = harness.state()
    state.update(status="planning")
    (harness.site() / "website_withdrawal" / automatic.STATE_FILENAME).write_text(json.dumps(state))
    summary = harness.run(offset=60.0)
    recovered = harness.state()
    assert recovered["status"] == "failed" and recovered["last_error"] == "interrupted_while_planning"
    assert recovered["next_attempt_at_epoch"] == harness.now + 60.0 + HOUR
    assert summary["action"] is None and summary["withdrawals"]["failing"] == 1


def test_cool_off_is_counted_from_the_acknowledgement(harness):
    harness.approve(cool_off_hours=48)
    harness.run()
    state = harness.state()
    assert state["status"] == "planned" and state["waiting_on"] == ["cool_off"] and harness.applies == []
    harness.run(offset=47 * HOUR)
    assert harness.applies == []
    harness.run(offset=48 * HOUR + 60)
    assert harness.state()["status"] == "applied"


def test_an_overdue_withdrawal_alerts_whatever_it_waits_on(harness):
    harness.run()
    summary = harness.run(offset=automatic.OVERDUE_ALERT_SECONDS + 60)
    assert "website_withdrawal_deletion_overdue" in harness.posted_codes()
    assert summary["withdrawals"]["overdue"] == 1 and summary["withdrawals"]["awaiting_owner_approval"] == 1


def test_capacity_summary_carries_the_counts_for_door_status(harness, tmp_path, monkeypatch):
    from collections import namedtuple

    from blueprint_pipeline import control_plane_capacity_controller as cap

    monkeypatch.setattr("blueprint_pipeline.task_evaluation_scene_spend.observe_configured_scene_project_spend",
                        lambda **_kwargs: None)
    usage = namedtuple("Usage", "total used free")(100 * 2**30, 0, 100 * 2**30)
    harness.run()
    report = cap.run_controller(
        mounts=[str(tmp_path)], report_root=tmp_path / "capacity", reservation_root=tmp_path / "ledger",
        webhook_url="", volume=None, ack="", token="", now=harness.now, survey=None,
        disk_usage=lambda _path: usage, website_withdrawal_summary_path=harness.report_root / "summary.json")
    summary = json.loads((tmp_path / "capacity/summary.json").read_text())
    section = summary["website_withdrawal_deletion"]
    assert section == report["website_withdrawal_deletion"]
    assert section["withdrawals"]["awaiting_owner_approval"] == 1 and section["approval_status"] == "absent"
    assert set(section) == {"status", "observed_at_epoch", "approval_status", "host_quiet", "scan_complete",
                            "withdrawals", "alert_codes"}


def test_the_automatic_drill_plans_then_applies_under_a_synthetic_approval(tmp_path):
    result = drill.run_automatic_drill(tmp_path, vector_path=VECTOR)
    assert result["outside_work_root_unchanged"] is True
    assert result["plan_pass"]["action"]["waiting_on"] == ["standing_approval_absent"]
    assert result["apply_pass"]["action"]["status"] == "applied"
    assert result["idle_pass"]["action"] is None
    assert result["receipt"]["authorization"]["kind"] == "standing_owner_approval"


def test_the_preparation_retry_copies_of_the_site_are_deleted_and_unreadable_ones_reported(harness):
    unclaimed = harness.world["resolution"]["unclaimed_handoff_root"]
    # The per-capture retry payload lies inside the site's capture, so it goes with the scene files.
    retry = harness.captures["req1"] / ".pipeline_preparation_retry_payload.json"
    retry.write_text(json.dumps({"bucket": "synthetic", "scene_id": "site-req1"}))
    (unclaimed / "garbled.json").write_text("{not json")
    harness.approve()
    harness.run()
    state = harness.state()
    # The unreadable entry keeps it incomplete: whether it is the site's is unknown.
    assert state["status"] == "incomplete", state.get("last_error")
    assert state["waiting_on"] == ["unclaimed_preparation_handoffs_unreadable"]
    receipt = json.loads((harness.site() / "website_withdrawal/deletion_receipts"
                          / f"{state['plan_digest'][7:]}.json").read_text())
    assert receipt["deletion_status"] == "incomplete"
    assert not retry.exists()
    assert receipt["deleted"]["unclaimed_handoffs"] == [str(unclaimed / "handoff-req1")]
    assert not (unclaimed / "handoff-req1").exists()
    assert (unclaimed / "handoff-drill-other.json").is_file() and (unclaimed / "garbled.json").is_file()
    reported = next(row for row in receipt["remaining"] if row["holder"] == "unclaimed_preparation_handoffs_unreadable")
    assert reported["entries"] == [{"path": str(unclaimed / "garbled.json"), "reason": "no_scene_or_capture_named"}]
    assert receipt["pipeline_controlled_deletion_complete"] is False


def test_an_unclaimed_handoff_naming_two_sites_is_unresolved_and_kept(harness):
    unclaimed = harness.world["resolution"]["unclaimed_handoff_root"]
    shared = unclaimed / "handoff-shared.json"
    shared.write_text(json.dumps({"scene_id": "site-req1", "capture_id": "walkthrough-drill-other"}))
    harness.approve()
    harness.run()
    state = harness.state()
    assert state["status"] == "unresolved" and harness.applies == []
    assert state["unresolved_reasons"] == {"website_withdrawal_deletion_unclaimed_names_another_site": 1}
    assert shared.is_file() and (unclaimed / "handoff-req1").exists()


def _receipt(harness) -> dict:
    state = harness.state()
    return json.loads((harness.site() / "website_withdrawal/deletion_receipts"
                       / f"{state['plan_digest'][7:]}.json").read_text())


def test_a_finished_deletion_names_the_copies_that_remain_at_providers(harness):
    harness.approve()
    summary = harness.run()
    receipt = _receipt(harness)
    assert harness.state()["status"] == "applied"
    # Everything this system can delete is gone; copies at providers with no delete remain, named.
    assert receipt["deletion_status"] == "complete_provider_retention_remains"
    assert receipt["incomplete_reasons"] == []
    assert {row["provider"] for row in receipt["provider_retention_remains"]} == {
        "openai", "gemini", "world_labs", "vast"}
    assert all(row["reason"] for row in receipt["provider_retention_remains"])
    # Every reader shows the value as it is, never plain "complete".
    assert summary["action"]["deletion_status"] == "complete_provider_retention_remains"
    assert summary["withdrawals"]["complete_provider_retention_remains"] == 1
    assert summary["withdrawals"]["complete"] == 0
    assert summary["sites"] == [{"scene_id": "site-req1", "status": "complete_provider_retention_remains",
                                 "provider_retention_remains": ["gemini", "openai", "vast", "world_labs"],
                                 "acknowledged_at_epoch": summary["sites"][0]["acknowledged_at_epoch"]}]
    projected = automatic.capacity_projection(summary)
    assert projected["withdrawals"]["complete_provider_retention_remains"] == 1
    assert projected["withdrawals"]["complete"] == 0
    # The passes end there.
    again = harness.run(offset=30 * 86400.0)
    assert again["action"] is None and again["alerts"] == []
    assert receipt["deleted"]["indexed_root_entries"] == [str(harness.uncovered / "site-entry")]
    # Another site's entry in the same root is never touched.
    assert (harness.uncovered / "other-entry/record.json").is_file()


def test_an_unindexed_root_makes_the_result_incomplete(harness):
    harness.approve()
    harness.run(site_index_root=None)  # no index pass has ever read the root
    state = harness.state()
    receipt = _receipt(harness)
    assert state["status"] == "incomplete" and receipt["deletion_status"] == "incomplete"
    reason = next(row for row in receipt["incomplete_reasons"] if row["holder"] == "pipeline_root_not_indexed")
    assert reason["path"] == str(harness.uncovered) and reason["reason"] == "index_never_completed"
    assert (harness.uncovered / "site-entry").is_dir()  # nothing deleted where the index has not read
    # Incomplete stays waiting: blocked for 72 h it alerts, like an unresolved plan.
    harness.run(offset=automatic.BLOCKED_ALERT_SECONDS + 1, site_index_root=None)
    assert "website_withdrawal_deletion_unresolved" in harness.posted_codes()


def test_a_root_read_only_part_way_reads_unknown_never_nothing_found(harness):
    from blueprint_pipeline import website_withdrawal_site_index as site_index

    for name in ("a-entry", "b-entry", "c-entry"):
        (harness.uncovered / name).mkdir()
        (harness.uncovered / name / "record.json").write_text(json.dumps({"scene_id": "site-drill-other"}))
    report = site_index.advance(harness.index_root, [harness.uncovered], now=harness.now, max_entries=2)
    assert report == {harness.uncovered.name: "in_progress"}
    rows = site_index.lookup(harness.index_root, [harness.uncovered], seeds={"site-req1"}, site_tokens={"site-req1"})
    assert rows[0]["status"] == "unknown" and rows[0]["reason"] == "index_never_completed"
    # Resumed from its cursor, the next bounded run finishes the sweep.
    while site_index.advance(harness.index_root, [harness.uncovered], now=harness.now,
                             max_entries=2)[harness.uncovered.name] != "complete":
        pass
    rows = site_index.lookup(harness.index_root, [harness.uncovered], seeds={"site-req1"}, site_tokens={"site-req1"})
    assert rows[0]["status"] == "indexed" and [row["name"] for row in rows[0]["owned"]] == ["site-entry"]
    # An entry written after the sweep is unknown until it is read.
    (harness.uncovered / "d-entry").mkdir()
    rows = site_index.lookup(harness.index_root, [harness.uncovered], seeds={"site-req1"}, site_tokens={"site-req1"})
    assert rows[0]["status"] == "unknown" and rows[0]["unscanned"] == ["d-entry"]


def test_an_entry_whose_owner_is_unknown_or_shared_is_never_deleted(harness):
    unknown = harness.uncovered / "content-addressed"
    unknown.mkdir()
    (unknown / "blob.bin").write_bytes(b"shared bytes")
    shared = harness.uncovered / "both-entry"
    shared.mkdir()
    (shared / "record.json").write_text(json.dumps({"scenes": ["site-req1", "site-drill-other"]}))
    harness.approve()
    harness.run()
    receipt = _receipt(harness)
    assert receipt["deletion_status"] == "incomplete"
    holders = {row["holder"]: row for row in receipt["incomplete_reasons"]}
    assert holders["pipeline_root_owner_unknown"]["examples"] == [
        {"name": "content-addressed", "reason": "no_site_named"}]
    assert holders["pipeline_root_shared_entries"]["entries"] == ["both-entry"]
    assert (unknown / "blob.bin").is_file() and (shared / "record.json").is_file()
    assert not (harness.uncovered / "site-entry").exists()


def test_old_index_less_receipts_and_plans_stay_readable(harness):
    # A receipt sealed before indexing (no deletion_status) is read, and the site is planned again.
    plan = deletion.plan_withdrawal_deletion(capture_root=harness.captures["req1"], **{
        key: value for key, value in harness.resolution.items()
        if key not in {"site_index_root", "provider_copies", "provider_deleters"}})
    assert plan["indexed_roots"] is None and plan["provider_copies"] is None
    receipt = deletion.apply_withdrawal_deletion(
        capture_root=harness.captures["req1"], expected_plan_digest=plan["digest"], authorize_complete_deletion=True,
        **{key: value for key, value in harness.resolution.items()
           if key not in {"site_index_root", "provider_copies", "provider_deleters"}})
    assert receipt["deletion_status"] == "incomplete"
    assert {row["holder"] for row in receipt["incomplete_reasons"]} >= {"pipeline_root_not_indexed",
                                                                      "provider_copies_not_checked"}
    legacy = {key: value for key, value in receipt.items() if key not in {"deletion_status", "incomplete_reasons"}}
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    legacy["digest"] = canonical_digest(legacy, digest_field="digest")
    (harness.site() / "website_withdrawal/deletion_receipts" / "legacy.json").write_text(json.dumps(legacy))
    harness.approve()
    summary = harness.run()
    assert summary["action"]["scene_id"] == "site-req1"  # not counted complete; planned again
    assert harness.state()["status"] == "applied"


def _provider_records(harness) -> None:
    pipeline = harness.captures["req1"] / "pipeline"
    (pipeline / "review").mkdir(parents=True, exist_ok=True)
    (pipeline / "review/review-media-abc.json").write_text(json.dumps({
        "schema_version": "website_review_media.v1", "images": [{"name": "files/gem-1"}, {"name": "files/gem-2"}],
        "cleanup_errors": [{"name": "files/gem-2", "error_type": "TimeoutError"}], "cleanup_status": "incomplete"}))
    (pipeline / "sam").mkdir(parents=True, exist_ok=True)
    (pipeline / "sam/uploaded-video.json").write_text(json.dumps({"file_id": "file-meta1", "clip_digest": "x"}))


def test_provider_copies_are_deleted_through_their_clients_and_recorded(harness):
    _provider_records(harness)
    harness.approve()
    harness.run()
    receipt = _receipt(harness)
    # The job's own cleanup reported gem-1 deleted and gem-2 failed; meta's file was never cleaned up.
    assert sorted(harness.provider_calls) == [("gemini", "files/gem-2"), ("meta", "file-meta1")]
    statuses = {row["id"]: row["status"] for row in receipt["provider_copies"]["copies"]}
    assert statuses == {"files/gem-1": "deleted_by_job", "files/gem-2": "deleted_at_provider",
                        "file-meta1": "deleted_at_provider"}
    assert {row["provider"] for row in receipt["provider_copies"]["not_deleted_at_provider"]} >= {"openai", "vast"}
    assert receipt["deletion_status"] == "complete_provider_retention_remains"
    # The ids outlive the local records that held them.
    ledger = json.loads((harness.site() / "website_withdrawal/provider_copies.json").read_text())
    assert set(ledger["copies"]) == {"gemini:files/gem-1", "gemini:files/gem-2", "meta:file-meta1"}


def test_a_failed_provider_delete_is_retried_next_pass_and_never_reported_done(harness):
    from blueprint_pipeline.website_capture_withdrawal_provider_copies import ProviderAnswer

    _provider_records(harness)
    harness.provider_failures["file-meta1"] = ProviderAnswer(503, "meta_files_delete_refused")
    harness.approve()
    harness.run()
    state, receipt = harness.state(), _receipt(harness)
    assert state["status"] == "incomplete" and receipt["deletion_status"] == "incomplete"
    meta = next(row for row in receipt["provider_copies"]["copies"] if row["id"] == "file-meta1")
    assert meta["status"] == "delete_failed" and meta["last_error"] == "meta_files_delete_refused:503"
    assert state["next_attempt_at_epoch"] == harness.now + HOUR
    harness.run(offset=HOUR - 1)
    assert harness.provider_calls.count(("meta", "file-meta1")) == 1
    # A 404 next time means the copy is gone at the provider: recorded as absent, not as deleted by us.
    harness.provider_failures["file-meta1"] = ProviderAnswer(404, "meta_files_delete_refused")
    harness.run(offset=HOUR + 1)
    assert harness.provider_calls.count(("meta", "file-meta1")) == 2
    assert harness.provider_calls.count(("gemini", "files/gem-2")) == 1  # done once, never again
    statuses = {row["id"]: row["status"] for row in _receipt(harness)["provider_copies"]["copies"]}
    assert statuses["file-meta1"] == "absent_at_provider" and harness.state()["status"] == "applied"


def test_a_provider_copy_with_no_recorded_id_is_unknown(harness):
    sam = harness.captures["req1"] / "pipeline/sam"
    sam.mkdir(parents=True)
    (sam / "video-upload-intent.json").write_text(json.dumps({"clip_digest": "x"}))
    harness.approve()
    harness.run()
    receipt = _receipt(harness)
    assert receipt["deletion_status"] == "incomplete"
    # Meta offers a delete, so a copy without its id keeps the deletion incomplete, by name.
    unknown = next(row for row in receipt["incomplete_reasons"] if row["holder"] == "provider_copy_id_not_recorded")
    assert unknown["copies"][0]["provider"] == "meta" and harness.provider_calls == []
    assert harness.state()["status"] == "incomplete"


def test_without_the_approval_no_provider_copy_is_deleted(harness):
    _provider_records(harness)
    harness.run()
    assert harness.provider_calls == [] and harness.state()["status"] == "planned"


def test_complete_only_when_nothing_known_remains_anywhere(harness, monkeypatch):
    from blueprint_pipeline import website_capture_withdrawal_provider_copies as provider_copy

    # A host whose jobs left nothing at any provider without a delete.
    monkeypatch.setattr(provider_copy, "NOT_DELETED_AT_PROVIDER", ())
    harness.approve()
    summary = harness.run()
    receipt = _receipt(harness)
    assert receipt["deletion_status"] == "complete" and receipt["provider_retention_remains"] == []
    assert summary["withdrawals"]["complete"] == 1 and summary["sites"][0]["status"] == "complete"


def test_an_unknown_copy_at_a_provider_with_no_delete_is_retention_not_incomplete():
    from blueprint_pipeline import website_capture_withdrawal_provider_copies as provider_copy

    section = {"copies": [], "not_deleted_at_provider": [],
               "unknown": [{"provider": "vast", "reason": "no instance id recorded"}]}
    assert provider_copy.blocking(section) == []
    assert provider_copy.retention_remains(section)[0]["provider"] == "vast"
    assert deletion.deletion_status([], provider_copy.retention_remains(section)) == \
        "complete_provider_retention_remains"
    assert deletion.deletion_status([{"holder": "x"}], []) == "incomplete"
    assert deletion.deletion_status([], []) == "complete"


def test_door_status_shows_the_middle_value_as_it_is(harness, tmp_path, monkeypatch):
    from collections import namedtuple

    from blueprint_pipeline import control_plane_capacity_controller as cap

    monkeypatch.setattr("blueprint_pipeline.task_evaluation_scene_spend.observe_configured_scene_project_spend",
                        lambda **_kwargs: None)
    harness.approve()
    harness.run()
    usage = namedtuple("Usage", "total used free")(100 * 2**30, 0, 100 * 2**30)
    cap.run_controller(mounts=[str(tmp_path)], report_root=tmp_path / "capacity", reservation_root=tmp_path / "ledger",
                       webhook_url="", volume=None, ack="", token="", now=harness.now, survey=None,
                       disk_usage=lambda _path: usage,
                       website_withdrawal_summary_path=harness.report_root / "summary.json")
    section = json.loads((tmp_path / "capacity/summary.json").read_text())["website_withdrawal_deletion"]
    assert section["withdrawals"]["complete_provider_retention_remains"] == 1
    assert section["withdrawals"]["complete"] == 0
