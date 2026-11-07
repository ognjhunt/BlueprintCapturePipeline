"""Withdrawal deletion of a site's rigid robot-free probe copies once each probe run's money is final.

Beta journey step: site withdrawal after the site's compiled half queued its robot-free probe. The
owner's rule (2026-11-06): delete them once settled, as withdrawal deletes screening attempts. The
attempts are laid out as the runner and the allocator leave them on the host (the settlement tests'
Vast evidence), with synthetic bytes only; nothing reaches the WebApp, a provider or the host.
"""
# Covers: src/blueprint_pipeline/website_capture_withdrawal_rigid_probe.py
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import rigid_robot_free_probe_lane as lane
from blueprint_pipeline import rigid_robot_free_probe_queue_root as layout
from blueprint_pipeline import rigid_robot_free_probe_runner as runner
from blueprint_pipeline import rigid_robot_free_probe_settlement as settlement
from blueprint_pipeline import rigid_robot_free_probe_worker as probe_worker
from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
from blueprint_pipeline import website_capture_withdrawal_native_g1 as native_g1
from blueprint_pipeline import website_capture_withdrawal_team_scene as team_scene
from tests import test_website_capture_withdrawal_deletion as base
from tests import test_website_capture_withdrawal_team_scene as scene
from tests.test_website_capture_withdrawal_team_scene import _sealed, probe_attempt

world, team = base.world, scene.team  # the site, its halves, records and probe jobs
REPO = Path(__file__).resolve().parents[1]


def _job(team, suffix: str) -> str:
    return f"{team['key']}-{suffix * 16}"


def test_a_settled_jobs_copies_are_deleted_and_its_money_records_still_replay(world, team):
    settled, unfunded = team["probe_attempts"]
    job = team["job"]
    before = {"counts": runner._counts(job), "funding": settlement.funding_admission(settled["attempt"], settled["work"]),
              "command": settlement.settlement_command(settled["attempt"], settled["work"])[0],
              "settled": runner.settle_attempt_funding(settled["attempt"]),
              "unfunded": runner.settle_attempt_funding(unfunded["attempt"])}
    assert before["settled"]["status"] == "settled" and before["unfunded"]["status"] == "not_funded"
    gcs = base.FakeGcs()
    receipt = base._apply(world, base._plan(world, cloud_storage=gcs), cloud_storage=gcs)
    # Gone: the sealed bundle built from the half, the probe input (the half's scene plan), the run's output
    # archive and its extraction.
    assert not any(path.exists() for attempt in team["probe_attempts"] for path in attempt["bulk"])
    # Kept: what the settlement reads, which replays exactly as before.
    assert all(path.exists() for attempt in team["probe_attempts"] for path in attempt["records"])
    assert runner._counts(job) == before["counts"]
    assert settlement.funding_admission(settled["attempt"], settled["work"]) == before["funding"]
    assert settlement.settlement_command(settled["attempt"], settled["work"])[0] == before["command"]
    assert runner.settle_attempt_funding(settled["attempt"]) == before["settled"]
    assert runner.settle_attempt_funding(unfunded["attempt"]) == before["unfunded"]
    [row, other] = receipt["deleted"]["team_scene"]["probe_bulk"]
    assert (row["attempt"], row["money"]) == (str(settled["attempt"]), "settled")
    assert (other["attempt"], other["money"]) == (str(unfunded["attempt"]), "never_funded")
    # Another site's settled job is never this site's to delete.
    assert all(path.exists() for path in [*team["other_job"]["bulk"], *team["other_job"]["records"]])


@pytest.mark.parametrize("money,reason", [("pending", "settlement_pending"),
                                          ("running", "allocator_running_or_unknown")])
def test_the_deletion_waits_until_the_sites_probe_money_settles(world, team, money, reason):
    # Owner decision (2026-11-06): settlement reads the capture's task context, which the deletion removes.
    waiting = probe_attempt(team["probes"], team["work_root"], _job(team, "f"), 0, capture=team["capture"],
                            money=money)
    job = str(waiting["attempt"].parents[1])
    gcs = base.FakeGcs()
    plan = base._plan(world, cloud_storage=gcs)
    assert {"path": job, "reason": "website_withdrawal_deletion_rigid_probe_money_not_final"} in plan["unresolved"]
    assert plan["team_scene"]["probe_jobs_waiting"] == [{"job": job, "final": False, "attempts": [],
                                                         "waiting": [{"attempt": "00", "reason": reason}]}]
    capture = sorted(Path(team["capture"]).rglob("*"))
    with pytest.raises(deletion.WithdrawalDeletionError, match="unresolved_copies"):
        base._apply(world, plan, cloud_storage=gcs)
    # Nothing went: the settled job, the waiting one, and the capture (whose task context settlement reads).
    assert all(path.exists() for path in [*waiting["bulk"], *waiting["records"], *team["probe_attempts"][0]["bulk"]])
    assert capture and sorted(Path(team["capture"]).rglob("*")) == capture
    # Once the runner settles it, the deletion planned again goes ahead.
    settled = json.loads((team["probe_attempts"][0]["attempt"] / settlement.SETTLEMENT_FILENAME).read_text())
    (waiting["attempt"] / settlement.SETTLEMENT_FILENAME).write_text(
        json.dumps(_sealed({**settled, "attempt": "00"}, "record_digest")))
    plan = base._plan(world, cloud_storage=gcs)
    assert not [row for row in plan["unresolved"] if row["path"] == job]
    receipt = base._apply(world, plan, cloud_storage=gcs)
    assert not any(path.exists() for path in waiting["bulk"])
    assert all(path.exists() for path in waiting["records"])  # Its money records stay, as for any settled job.
    assert "rigid_robot_free_probe_jobs_waiting" not in {row["holder"] for row in receipt["remaining"]}


@pytest.mark.parametrize("fault", ["unreadable", "unsealed", "another_attempt", "linked"])
def test_a_settlement_that_does_not_read_as_this_attempts_is_never_taken_as_settled(world, team, fault):
    settled = team["probe_attempts"][0]
    path = settled["attempt"] / settlement.SETTLEMENT_FILENAME
    record = json.loads(path.read_text())
    if fault == "unreadable":
        path.write_bytes(b'{"status": "settled", ')
    elif fault == "unsealed":
        path.write_text(json.dumps({**record, "settled_cost_usd": 0.01}))
    elif fault == "another_attempt":
        path.write_text(json.dumps(_sealed({**record, "attempt": "01"}, "record_digest")))
    else:
        path.unlink()
        os.symlink(team["other_job"]["attempt"] / settlement.SETTLEMENT_FILENAME, path)
    gcs = base.FakeGcs()
    plan = base._plan(world, cloud_storage=gcs)
    [row] = plan["team_scene"]["probe_jobs_waiting"]
    assert row["waiting"] == [{"attempt": "00", "reason": "settlement_unreadable"}]
    assert plan["team_scene"]["probe_jobs_final"] == []
    # It may still settle, so the deletion waits.
    assert {"path": row["job"], "reason": "website_withdrawal_deletion_rigid_probe_money_not_final"} in plan["unresolved"]
    with pytest.raises(deletion.WithdrawalDeletionError, match="unresolved_copies"):
        base._apply(world, plan, cloud_storage=gcs)
    assert all(path.exists() for attempt in team["probe_attempts"] for path in attempt["bulk"])


@pytest.mark.parametrize("change", ["bundle_gains_a_file", "output_rewritten", "attempt_staged_since"])
def test_a_probe_job_changed_after_review_is_not_deleted(world, team, tmp_path, change):
    plan = base._plan(world)
    native_g1.apply_native_g1(plan["native_g1"])  # as in a deletion: the G1 namespaces go first
    settled = team["probe_attempts"][0]
    if change == "bundle_gains_a_file":
        base._write(settled["work"] / "bundle" / "late.json", {"written": "after review"})
    elif change == "output_rewritten":
        base._write(settled["bulk"][2], b"rewritten after review")
    else:
        probe_attempt(team["probes"], team["work_root"], team["job"].name, 2, capture=team["capture"],
                      money="running")
    before = base._snapshot(tmp_path)
    with pytest.raises(deletion.WithdrawalDeletionError, match="rigid_probe_job_changed"):
        team_scene.apply_team_scene(plan["team_scene"])
    assert base._snapshot(tmp_path) == before  # not one record or half went before the refusal
    # The whole deletion refuses the same change before it deletes anything.
    with pytest.raises(deletion.WithdrawalDeletionError, match="plan_changed"):
        base._apply(world, plan)


def test_a_runner_advancing_the_job_refuses_the_deletion(world, team, tmp_path):
    lock = base._write(team["job"] / layout.JOB_LOCK_FILENAME, b"")
    plan = base._plan(world)
    native_g1.apply_native_g1(plan["native_g1"])
    before = base._snapshot(tmp_path)
    with lock.open("rb") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(deletion.WithdrawalDeletionError, match="rigid_probe_job_busy"):
            team_scene.apply_team_scene(plan["team_scene"])
    assert base._snapshot(tmp_path) == before


@pytest.mark.parametrize("fault", ["link_in_bundle", "fifo_in_allocator"])
def test_a_settled_jobs_copy_it_cannot_judge_fails_closed(world, team, tmp_path, fault):
    settled = team["probe_attempts"][0]
    if fault == "link_in_bundle":
        os.symlink("/etc", settled["work"] / "bundle" / "escape")
    else:
        os.mkfifo(settled["work"] / "allocator" / "pipe")
    plan = base._plan(world)
    assert {"path": str(team["job"]), "reason": "website_withdrawal_deletion_unsafe_entry"} in plan["unresolved"]
    before = base._snapshot(tmp_path)
    with pytest.raises(deletion.WithdrawalDeletionError, match="unresolved_copies"):
        base._apply(world, plan)
    assert base._snapshot(tmp_path) == before


@pytest.mark.parametrize("fault,reason", [("unmounted", "work_dir_unavailable"), ("elsewhere", "work_dir_invalid"),
                                          ("unbound", "work_dir_unbound")])
def test_a_work_directory_not_tied_to_its_attempt_waits(world, team, fault, reason):
    if fault == "unmounted":
        attempt = probe_attempt(team["probes"], team["work_root"], _job(team, "f"), 0, capture=team["capture"],
                                money="settled")
        os.rename(attempt["work"], team["work_root"].parent / "not-mounted")
        job, name = attempt["attempt"].parents[1], "00"
    elif fault == "elsewhere":
        attempt = probe_attempt(team["probes"], team["work_root"], _job(team, "f"), 0, capture=team["capture"],
                                money="settled")
        record = attempt["attempt"] / layout.ATTEMPT_FILENAME
        record.write_text(json.dumps({**json.loads(record.read_text()), "work_dir": "/etc/00"}))
        job, name = attempt["attempt"].parents[1], "00"
    else:
        # Bulk under a job of the capture that the queue no longer has.
        orphan = team["work_root"] / "jobs" / _job(team, "f") / "attempts" / "00"
        base._write(orphan / "bundle" / "native_task_arena_bundle.zip", b"synthetic")
        job, name = orphan.parents[1], None
    plan = base._plan(world)
    assert [row for row in plan["team_scene"]["probe_jobs_waiting"] if row["job"] == str(job)] == [
        {"job": str(job), "final": False, "attempts": [], "waiting": [{"attempt": name, "reason": reason}]}]
    # Bulk no queue attempt names settles nothing; a work directory it cannot read may, so that one waits.
    held = {"path": str(job), "reason": "website_withdrawal_deletion_rigid_probe_money_not_final"} in plan["unresolved"]
    assert held is (reason != "work_dir_unbound")


def test_a_staging_that_failed_before_its_record_is_deleted_with_its_attempt(world, team):
    # stage_attempt had made the work directory and part of the bundle when it refused; no allocator ran.
    attempt = team["probes"] / "jobs" / _job(team, "f") / "attempts" / "00"
    base._write(attempt / runner.INGEST_FILENAME, {"state": "not_staged", "funded": False, "allocated": False})
    partial = base._write(team["work_root"] / "jobs" / _job(team, "f") / "attempts" / "00" / "bundle" / "x.zip",
                          b"synthetic")
    gcs = base.FakeGcs()
    plan = base._plan(world, cloud_storage=gcs)
    [row] = [job for job in plan["team_scene"]["probe_jobs_final"] if job["job"] == str(attempt.parents[1])]
    assert row["attempts"][0]["money"] == "never_launched"
    base._apply(world, plan, cloud_storage=gcs)
    assert not partial.exists() and (attempt / runner.INGEST_FILENAME).is_file()


def test_the_layout_names_are_the_ones_the_runner_and_the_money_code_write():
    assert (runner.QUEUE_ENV, runner.DEFAULT_QUEUE, runner.WORK_ENV) == (
        layout.QUEUE_ENV, layout.DEFAULT_QUEUE, layout.WORK_ENV)
    assert (runner.ATTEMPT_FILENAME, runner.INTENT_FILENAME, runner.PROCESS_FILENAME, runner.SPAWN_FAILED_FILENAME,
            runner.RESULT_FILENAME, runner.WORK_DIR_FIELD, runner.JOB_LOCK_FILENAME) == (
        layout.ATTEMPT_FILENAME, layout.INTENT_FILENAME, layout.PROCESS_FILENAME, layout.SPAWN_FAILED_FILENAME,
        layout.RESULT_FILENAME, layout.WORK_DIR_FIELD, layout.JOB_LOCK_FILENAME)
    # Kept where the money code defines them; the light module's copies must match.
    assert (settlement.SETTLEMENT_FILENAME, settlement.SETTLEMENT_SCHEMA, settlement.FUNDING_FILENAME,
            settlement.RESULT_FILENAME) == (layout.SETTLEMENT_FILENAME, layout.SETTLEMENT_SCHEMA,
                                            layout.FUNDING_FILENAME, layout.RESULT_FILENAME)
    assert lane.FUNDING_FILENAME == layout.FUNDING_FILENAME
    assert probe_worker.INPUTS_FILENAME == layout.INPUTS_FILENAME
    unit = (REPO / "deploy/systemd/blueprint-rigid-robot-free-probe-runner.service").read_text()
    assert f"Environment={layout.WORK_ENV}={layout.DEFAULT_WORK_ROOT}\n" in unit
    assert f"Environment={layout.QUEUE_ENV}={layout.DEFAULT_QUEUE}\n" in unit
