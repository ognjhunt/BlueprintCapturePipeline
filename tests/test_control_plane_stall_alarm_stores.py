"""The four job stores nothing watched (2026-11-10) page on the stall route; their intended holds never do."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.deploy_control_plane_commit as deploy
from blueprint_pipeline import control_plane_capacity_controller as cap
from blueprint_pipeline import control_plane_stall_alarm as stall

REPO = Path(__file__).resolve().parents[1]
NOW = 1_793_500_000.0
MINUTE = 60.0
HOUR = 3600.0
CAPTURE = "walkthrough-req1"
KEY = "ab" * 16
CAMPAIGN = "campaign-site-req1-r1"


def _budget() -> stall._Budget:
    return stall._Budget(deadline=float("inf"), events=stall.MAX_EVENT_READS, clock=lambda: 0.0)


def _config(tmp_path: Path, **overrides) -> stall.StallConfig:
    values = {"scene_intent_root": None, "launch_queue_root": None, "launch_state_root": None, "deploy_tool": None,
              "atlas_storage_root": tmp_path / "storage", "team_scene_queue_root": tmp_path / "team",
              "rigid_probe_queue_root": tmp_path / "rigid", "screening_queue_root": tmp_path / "screening"}
    values.update(overrides)
    return stall.StallConfig(**values)


def _pages(alerts: list[dict]) -> list[dict]:
    return [row for row in alerts if row["code"] in stall.PAGE_CODES]


def _incomplete(alerts: list[dict]) -> list[str]:
    return [row["reason"] for row in alerts if row["code"] == stall.INCOMPLETE_CODE]


def _hold(tmp_path: Path, unit: str) -> Path:
    holds = tmp_path / "holds"
    holds.mkdir(exist_ok=True)
    (holds / f"{unit}.json").write_text("{}")
    return holds


# --------------------------------------------------------------------------- #
# Atlas preview jobs
# --------------------------------------------------------------------------- #


def _atlas_job(tmp_path: Path, *, capture: str = CAPTURE, request: str = "req1", **fields) -> Path:
    path = tmp_path / f"storage/bucket/scenes/site-{request}/captures/{capture}/pipeline/atlas/preview_job.json"
    path.parent.mkdir(parents=True)
    job = {"schema_version": "website_atlas_preview_job.v1", "request_id": request, "scene_id": f"site-{request}",
           "capture_id": capture, "state": "processing", "reason": None, "stage": "source_pose", "revision": 1,
           "enqueued_at_epoch": NOW - 5 * HOUR, "updated_at_epoch": NOW - MINUTE, "last_tick_epoch": NOW - MINUTE,
           "next_attempt_at_epoch": NOW - MINUTE, "retry_at_epoch": None, "transient_failures": 0, **fields}
    path.write_text(json.dumps(job))
    return path


def _atlas(tmp_path: Path, *, now: float = NOW, **kwargs):
    return stall.check_atlas_previews(_config(tmp_path), now=now, budget=_budget(), **kwargs)


def test_a_healthy_finished_or_waiting_atlas_job_is_quiet(tmp_path: Path) -> None:
    _atlas_job(tmp_path)
    # Final, however old.
    for index, state in enumerate(("ready", "failed", "withdrawn")):
        _atlas_job(tmp_path, capture=f"walkthrough-old{index}", request=f"old{index}", state=state,
                   updated_at_epoch=NOW - 6 * 86400, next_attempt_at_epoch=NOW - 6 * 86400)
    # Waiting for a person (a new capture, a background review): no recheck is scheduled.
    _atlas_job(tmp_path, capture="walkthrough-person", request="person", state="needs_input",
               reason="atlas_background_review_required", updated_at_epoch=NOW - 6 * 86400)
    # An outage backing off: the job made no progress for hours, but its next attempt is still ahead.
    _atlas_job(tmp_path, capture="walkthrough-outage", request="outage", transient_failures=9,
               last_transient_reason="website_atlas_api_503", updated_at_epoch=NOW - 5 * HOUR,
               next_attempt_at_epoch=NOW + 40 * MINUTE)
    section, alerts = _atlas(tmp_path)
    assert alerts == []
    assert section["status"] == "observed" and section["examined"] == 6 and section["stalled"] == 0
    assert section["dispositions"] == {"failed": 1, "needs_input": 1, "ready": 1, "scheduled_wait": 1,
                                       "watched": 1, "withdrawn": 1}


def test_a_processing_atlas_job_with_no_progress_and_no_wait_pages(tmp_path: Path) -> None:
    # 2026-11-09: no tick finished a step, so the job file kept its old progress and an old next attempt.
    _atlas_job(tmp_path, updated_at_epoch=NOW - 2 * HOUR, next_attempt_at_epoch=NOW - 2 * HOUR,
               last_tick_epoch=NOW - 2 * HOUR)
    _atlas_job(tmp_path, capture="walkthrough-req2", request="req2", updated_at_epoch=NOW - 89 * MINUTE,
               next_attempt_at_epoch=NOW - 89 * MINUTE)
    section, [alert] = _atlas(tmp_path)
    assert section["stalled"] == 1 and section["dispositions"] == {"watched": 1}
    assert alert == {"code": "atlas_preview_stalled", "count": 1, "targets": [CAPTURE], "target": CAPTURE,
                     "phase": "source_pose", "status": "silent", "age_seconds": 7200, "threshold_seconds": 5400}
    assert stall.describe(alert) == (
        "Atlas preview job processing with no finished step 2.0h at stage source_pose: walkthrough-req1")


def test_an_atlas_refusal_retry_not_yet_due_is_quiet_and_one_overdue_pages(tmp_path: Path) -> None:
    # The fifth timed retry of an unclassified refusal waits 12 h; the job last progressed a day ago.
    fields = {"refusal_retries": 5, "refusal_retries_total": 5, "last_refusal_reason": "atlas_pose_alignment_refused",
              "updated_at_epoch": NOW - 30 * HOUR}
    _atlas_job(tmp_path, next_attempt_at_epoch=NOW + 6 * HOUR, **fields)
    section, alerts = _atlas(tmp_path)
    assert alerts == [] and section["dispositions"] == {"scheduled_wait": 1}
    # Due, and not taken yet within the threshold: still quiet (the worker steps every minute).
    assert _atlas(tmp_path, now=NOW + 6 * HOUR + 89 * MINUTE)[1] == []
    # Overdue past the threshold: the worker is not picking it up.
    _section, [alert] = _atlas(tmp_path, now=NOW + 6 * HOUR + 91 * MINUTE)
    assert (alert["status"], alert["age_seconds"], alert["refusal_retries"]) == ("retry_overdue", 91 * 60, 5)
    assert alert["blockers"] == ["atlas_pose_alignment_refused"]
    assert stall.describe(alert) == ("Atlas preview job scheduled retry overdue 91m [atlas_pose_alignment_refused] "
                                     "at stage source_pose: walkthrough-req1")


def test_an_atlas_outage_backoff_not_yet_due_is_quiet_and_one_overdue_pages(tmp_path: Path) -> None:
    _atlas_job(tmp_path, transient_failures=12, last_transient_reason="website_atlas_api_503",
               updated_at_epoch=NOW - 3 * 86400, next_attempt_at_epoch=NOW + 30 * MINUTE)
    assert _atlas(tmp_path)[1] == []
    _section, [alert] = _atlas(tmp_path, now=NOW + 30 * MINUTE + 2 * HOUR)
    assert (alert["status"], alert["age_seconds"], alert["transient_failures"]) == ("retry_overdue", 7200, 12)
    assert alert["blockers"] == ["website_atlas_api_503"]


def test_an_atlas_step_that_never_returned_pages(tmp_path: Path) -> None:
    # The worker marks a step in flight before it runs; a kill leaves the mark, and the next tick would
    # turn it into a timed retry. A mark this old means no tick came, or one step has run for 90 minutes.
    _atlas_job(tmp_path, step_in_flight={"since_epoch": NOW - 2 * HOUR, "stage": "source_pose"},
               next_attempt_at_epoch=NOW - 2 * HOUR, updated_at_epoch=NOW - 3 * HOUR)
    _section, [alert] = _atlas(tmp_path)
    assert (alert["status"], alert["age_seconds"]) == ("step_unreturned", 7200)
    assert stall.describe(alert) == (
        "Atlas preview job step started 2.0h ago never returned at stage source_pose: walkthrough-req1")
    # A step running for ten minutes is quiet.
    section, alerts = _atlas(tmp_path, now=NOW - 2 * HOUR + 10 * MINUTE)
    assert alerts == [] and section["dispositions"] == {"step_running": 1}


def test_an_atlas_input_recheck_overdue_pages_unless_it_waits_on_money_or_consent(tmp_path: Path) -> None:
    # An operator-liftable refusal the worker re-asks every 30 minutes.
    _atlas_job(tmp_path, state="needs_input", reason="task_brief_missing", updated_at_epoch=NOW - 2 * 86400,
               retry_at_epoch=NOW + 10 * MINUTE)
    section, alerts = _atlas(tmp_path)
    assert alerts == [] and section["dispositions"] == {"scheduled_wait": 1}
    _section, [alert] = _atlas(tmp_path, now=NOW + 10 * MINUTE + 2 * HOUR)
    assert (alert["status"], alert["age_seconds"], alert["blockers"]) == (
        "recheck_overdue", 7200, ["task_brief_missing"])
    # A money refusal waits as needs_input and a consent refusal is re-asked too: neither pages, however late.
    _atlas_job(tmp_path / "held", state="needs_input", reason="atlas_runtime_price_review_expired",
               retry_at_epoch=NOW - 2 * 86400)
    _atlas_job(tmp_path / "held", capture="walkthrough-req2", request="req2", state="needs_input",
               reason="website_atlas_source_consent_required", retry_at_epoch=NOW - 2 * 86400)
    section, alerts = stall.check_atlas_previews(_config(tmp_path / "held"), now=NOW, budget=_budget())
    assert alerts == [] and section["dispositions"] == {"money_approval": 1, "privacy": 1}


def test_the_alarm_reads_what_the_atlas_worker_writes(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import atlas_preparation_job_control as control
    from blueprint_pipeline import website_atlas_preview as preview

    assert stall.ATLAS_STEP_IN_FLIGHT == control.STEP_IN_FLIGHT
    monkeypatch.setenv(preview.ROUTING_ENV, "consented")
    capture = tmp_path / "storage/bucket/scenes/site-req1/captures/walkthrough-req1"
    (capture / "raw").mkdir(parents=True)
    (capture / "raw/walkthrough.mov").write_bytes(b"video")
    (capture / "pipeline").mkdir()
    descriptor = SimpleNamespace(site_submission_id="req1", metadata={}, capture_id=CAPTURE, scene_id="site-req1")
    queued = preview.enqueue_preview(descriptor=descriptor, capture_root=capture, pipeline_dir=capture / "pipeline",
                                     source_video=lambda: capture / "raw/walkthrough.mov")
    start = queued["enqueued_at_epoch"]
    job_path = capture / "pipeline/atlas" / preview.JOB_FILE
    config = _config(tmp_path)

    def check(job, now):
        preview.write_json(job_path, job)
        return stall.check_atlas_previews(config, now=now, budget=_budget())

    # An outage: capped backoff, quiet until its attempt is overdue by the threshold.
    job = queued
    for failure in range(1, 8):
        job = preview._failed_step(job, OSError("connection reset"), start + failure * 600)
    assert job["state"] == "processing" and job["next_attempt_at_epoch"] > start + 4200
    assert check(job, job["next_attempt_at_epoch"] - 1)[1] == []
    _section, [alert] = check(job, job["next_attempt_at_epoch"] + 5400)
    assert (alert["status"], alert["transient_failures"]) == ("retry_overdue", 7)
    assert alert["blockers"] == [job["last_transient_reason"]]
    # An unclassified refusal on the six timed retries: its 15-minute wait is quiet.
    job = preview._failed_step(queued, ValueError("atlas_pose_alignment_refused"), start + 60)
    assert job["refusal_retries"] == 1 and job["next_attempt_at_epoch"] == start + 60 + 900
    assert check(job, start + 60 + 899)[1] == []
    assert check(job, start + 60 + 900 + 5400)[1][0]["status"] == "retry_overdue"
    # A step killed before it returned: the next tick turns the mark into a timed retry.
    marked = {**queued, control.STEP_IN_FLIGHT: {"since_epoch": start + 60, "stage": "queued"}}
    assert check(marked, start + 60 + 5400)[1][0]["status"] == "step_unreturned"
    job = preview._interrupted(dict(marked), start + 120)
    assert job["last_refusal_reason"] == "atlas_preview_step_interrupted"
    assert check(job, start + 120 + 899)[1] == []
    # A money refusal: needs_input, re-asked every 30 minutes; never a page.
    job = preview._failed_step(queued, ValueError("atlas_preview_spend_ceiling_reached"), start + 60)
    assert job["state"] == "needs_input" and job["retry_at_epoch"] == start + 60 + preview.AUTHORITY_RECHECK_SECONDS
    section, alerts = check(job, start + 5 * 86400)
    assert alerts == [] and section["dispositions"] == {"money_approval": 1}


def test_atlas_holds_never_page(tmp_path: Path) -> None:
    # An outage retried on a code that names consent, long overdue: an intended hold.
    _atlas_job(tmp_path, transient_failures=30, last_transient_reason="website_control_consent_recheck_pending",
               updated_at_epoch=NOW - 2 * 86400, next_attempt_at_epoch=NOW - 3 * HOUR)
    section, alerts = _atlas(tmp_path)
    assert alerts == [] and section["dispositions"] == {"privacy": 1}
    # The operator held the worker's timer through the door: every job waits on purpose.
    _atlas_job(tmp_path, capture="walkthrough-req2", request="req2", updated_at_epoch=NOW - 9 * HOUR,
               next_attempt_at_epoch=NOW - 9 * HOUR)
    assert _pages(_atlas(tmp_path)[1])
    section, alerts = _atlas(tmp_path, holds=_hold(tmp_path, stall.ATLAS_TRIGGER))
    assert alerts == [] and section == {"status": "held", "trigger": "blueprint-website-atlas-preview.timer"}


def test_an_unreadable_atlas_store_is_incomplete_not_a_page(tmp_path: Path) -> None:
    assert _atlas(tmp_path) == ({"status": "absent"}, [])  # no directory: nothing to check
    assert stall.check_atlas_previews(_config(tmp_path, atlas_storage_root=None), now=NOW, budget=_budget()) == (
        {"status": "not_configured"}, [])
    path = _atlas_job(tmp_path)
    path.write_text("{not json")
    _atlas_job(tmp_path, capture="walkthrough-req2", request="req2", schema_version="other.v1")
    section, alerts = _atlas(tmp_path)
    assert section["unreadable"] == 2 and _pages(alerts) == []
    assert alerts == [{"code": stall.INCOMPLETE_CODE, "reason": "atlas_preview_job_unreadable", "count": 2}]
    (tmp_path / "file").write_text("not a directory")
    section, alerts = stall.check_atlas_previews(_config(tmp_path, atlas_storage_root=tmp_path / "file"), now=NOW,
                                                 budget=_budget())
    assert section == {"status": "unreadable"} and _incomplete(alerts) == ["atlas_preview_root_unreadable"]


# --------------------------------------------------------------------------- #
# Team scene packet requests
# --------------------------------------------------------------------------- #


def _team_request(tmp_path: Path, state: str, *, key: str = KEY) -> Path:
    """A request moved into a queue directory just now: its age there is its ctime, which no test can set."""
    directory = tmp_path / "team" / state
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.json"
    path.write_text(json.dumps({"schema_version": "blueprint.team_scene_packet_request.v1", "capture_id": CAPTURE}))
    return path


def _team_status(tmp_path: Path, status: str, blockers: list[str], *, key: str = KEY, age: float,
                 now: float = NOW) -> Path:
    directory = tmp_path / "team" / "status"
    directory.mkdir(parents=True, exist_ok=True)
    (tmp_path / "team" / "processed").mkdir(exist_ok=True)
    path = directory / f"{key}.json"
    path.write_text(json.dumps({"schema_version": "blueprint.team_scene_packet_status.v1", "request_key": key,
                                "capture_id": CAPTURE, "status": status, "blockers": blockers}))
    os.utime(path, (now - age, now - age))
    return path


def _team(tmp_path: Path, *, now: float = NOW, **kwargs):
    return stall.check_team_scene_packets(_config(tmp_path), now=now, budget=_budget(), **kwargs)


def test_a_healthy_team_scene_queue_is_quiet(tmp_path: Path) -> None:
    moved = os.stat(_team_request(tmp_path, "pending")).st_ctime
    _team_request(tmp_path, "processing", key="cd" * 16)
    _team_status(tmp_path, "completed", [], key="ef" * 16, age=30 * 86400, now=moved)
    _team_status(tmp_path, "skipped", [], key="01" * 16, age=30 * 86400, now=moved)
    # Blocked an hour ago: the worker's own retries (5 min, 30 min, 2 h) are still running.
    _team_status(tmp_path, "blocked", ["team_scene_packet_interrupted"], key="23" * 16, age=HOUR, now=moved)
    section, alerts = _team(tmp_path, now=moved + MINUTE)
    assert alerts == []
    assert section["dispositions"] == {"blocked": 1, "pending": 1, "processing": 1} and section["stalled"] == 0


def test_a_team_scene_request_left_waiting_or_blocked_pages(tmp_path: Path) -> None:
    moved = os.stat(_team_request(tmp_path, "pending")).st_ctime
    _team_request(tmp_path, "processing", key="cd" * 16)
    _team_status(tmp_path, "blocked", ["team_scene_source_launch_unavailable"], key="ef" * 16, age=23 * HOUR,
                 now=moved)
    # Seven hours on, nothing in the queue directories has waited long enough; the old block has.
    _section, [alert] = _team(tmp_path, now=moved + 7 * HOUR)
    assert alert["targets"] == ["ef" * 16] and alert["status"] == "blocked" and alert["age_seconds"] == 30 * 3600
    assert alert["label"] == CAPTURE and alert["blockers"] == ["team_scene_source_launch_unavailable"]
    assert stall.describe(alert) == (
        "team scene packet request blocked 30.0h [team_scene_source_launch_unavailable]: walkthrough-req1")
    section, [alert] = _team(tmp_path, now=moved + 9 * HOUR)
    assert alert["code"] == "team_scene_packet_stalled" and alert["count"] == 3 and section["stalled"] == 3
    assert set(alert["targets"]) == {KEY, "cd" * 16, "ef" * 16} and alert["threshold_seconds"] == 8 * 3600
    # A blocked request the worker returned to its queue is counted there, not as an old block.
    _team_request(tmp_path, "pending", key="ef" * 16)
    assert _team(tmp_path, now=moved + 7 * HOUR)[1] == []


@pytest.mark.parametrize("blockers,reason", [
    (["team_scene_owner_contract_invalid:task_evaluation_success_contract_owner_unconfirmed"], "owner_confirmation"),
    (["team_scene_capture_withdrawn"], "privacy"),
    (["team_scene_capture_consent_required"], "privacy"),
    (["team_scene_capture_revoked"], "rights"),
    (["team_scene_dataset_license_unverified"], "rights"),
    (["team_scene_spend_approval_required"], "money_approval"),
])
def test_team_scene_holds_never_page(tmp_path: Path, blockers, reason) -> None:
    _team_status(tmp_path, "blocked", blockers, age=40 * 86400)
    section, alerts = _team(tmp_path)
    assert alerts == [] and section["dispositions"] == {reason: 1}


def _retry_status(tmp_path: Path, field: str, record: dict, *, key: str = KEY, age: float = 30 * HOUR) -> None:
    path = _team_status(tmp_path, "blocked", ["team_scene_source_launch_unavailable"], key=key, age=age)
    value = json.loads(path.read_text())
    path.write_text(json.dumps({**value, field: record}))
    os.utime(path, (NOW - age, NOW - age))


@pytest.mark.parametrize("field", ["refusal_retry", "automatic_retry"])
def test_a_team_scene_block_waiting_for_its_timed_retry_is_quiet_until_overdue(tmp_path: Path, field) -> None:
    # Blocked 30 h ago; the shared schedule's last step (24 h) is still ahead.
    _retry_status(tmp_path, field, {"attempts": 5, "limit": 6, "reason": "team_scene_source_launch_unavailable",
                                    "not_before": _iso(NOW + 2 * HOUR)})
    section, alerts = _team(tmp_path)
    assert alerts == [] and section["dispositions"] == {"scheduled_wait": 1}
    # Due seven hours ago and not returned to pending: within the threshold.
    _retry_status(tmp_path, field, {"attempts": 5, "not_before": _iso(NOW - 7 * HOUR)})
    assert _team(tmp_path)[1] == []
    # Overdue by more than the threshold: the worker is not taking it.
    _retry_status(tmp_path, field, {"attempts": 5, "not_before": _iso(NOW - 9 * HOUR)})
    _section, [alert] = _team(tmp_path)
    assert (alert["status"], alert["age_seconds"]) == ("retry_overdue", 9 * 3600)
    assert stall.describe(alert) == ("team scene packet request blocked, timed retry overdue 9.0h "
                                     "[team_scene_source_launch_unavailable]: walkthrough-req1")
    # A spent schedule names no wait: the block's own age counts, as before.
    _retry_status(tmp_path, field, {"attempts": 6, "exhausted": True})
    _section, [alert] = _team(tmp_path)
    assert (alert["status"], alert["age_seconds"]) == ("blocked", 30 * 3600)


def test_a_held_team_scene_trigger_and_an_unreadable_queue(tmp_path: Path) -> None:
    assert _team(tmp_path) == ({"status": "absent"}, [])
    _team_status(tmp_path, "blocked", ["team_scene_source_launch_unavailable"], age=30 * HOUR)
    assert _pages(_team(tmp_path)[1])
    assert _team(tmp_path, holds=_hold(tmp_path, stall.TEAM_SCENE_TRIGGER)) == (
        {"status": "held", "trigger": "blueprint-team-scene-packet.path"}, [])
    # A status that does not read is reported, never guessed at.
    (tmp_path / "team/status" / f"{KEY}.json").write_text("{not json")
    section, alerts = _team(tmp_path)
    assert _pages(alerts) == [] and section["unreadable"] == 1
    assert alerts == [{"code": stall.INCOMPLETE_CODE, "reason": "team_scene_request_unreadable", "count": 1}]
    # A queue directory that cannot be listed (a file where the directory should be).
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken/pending").write_text("not a directory")
    section, alerts = stall.check_team_scene_packets(
        _config(tmp_path, team_scene_queue_root=tmp_path / "broken"), now=NOW, budget=_budget())
    assert section == {"status": "unreadable"} and _incomplete(alerts) == ["team_scene_queue_unreadable"]


# --------------------------------------------------------------------------- #
# Rigid robot-free probe entries
# --------------------------------------------------------------------------- #

REQUEST_DIGEST = "sha256:" + "9" * 64


def _probe_entry(tmp_path: Path, *, key: str = KEY) -> Path:
    directory = tmp_path / "rigid" / "pending"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.json"
    path.write_text(json.dumps({"schema_version": "blueprint.rigid_robot_free_probe_queue_entry.v1",
                                "capture_root": f"/captures/{key}", "probe_request_digest": REQUEST_DIGEST}))
    return path


def _probe_attempt(tmp_path: Path, index: int, record: dict, *, key: str = KEY) -> None:
    attempt = tmp_path / "rigid" / "jobs" / f"{key}-{'9' * 16}" / "attempts" / f"{index:02d}"
    attempt.mkdir(parents=True)
    (attempt / "rigid_robot_free_probe_ingest.json").write_text(json.dumps(record))


def _probes(tmp_path: Path, *, now: float, **kwargs):
    return stall.check_rigid_probes(_config(tmp_path), now=now, budget=_budget(), **kwargs)


def test_a_rigid_probe_pending_past_its_attempts_pages(tmp_path: Path) -> None:
    queued = os.stat(_probe_entry(tmp_path)).st_ctime
    _probe_attempt(tmp_path, 0, {"state": "unfunded_blocked", "allocator_blockers": ["provider_offer_unavailable"]})
    _probe_attempt(tmp_path, 1, {"state": "provider_zero_unproven", "funded": True, "allocated": True})
    # Two funded attempts of at most an hour each: quiet while they can still be running.
    section, alerts = _probes(tmp_path, now=queued + 3 * HOUR)
    assert alerts == [] and section["dispositions"] == {"watched": 1} and section["pending"] == 1
    section, [alert] = _probes(tmp_path, now=queued + 9 * HOUR)
    assert alert["code"] == "rigid_probe_stalled" and alert["target"] == KEY and section["stalled"] == 1
    # The latest attempt's own record says where it stopped.
    assert alert["status"] == "provider_zero_unproven" and alert["threshold_seconds"] == 8 * 3600
    assert stall.describe({**alert, "age_seconds": 9 * 3600}) == (
        f"rigid robot-free probe pending 9.0h (provider_zero_unproven): {KEY}")
    # A new attempt after a long wait is progress: the age counts from the attempt's last record.
    _probe_attempt(tmp_path, 2, {"state": "unfunded_blocked", "allocator_blockers": []})
    latest = tmp_path / "rigid/jobs" / f"{KEY}-{'9' * 16}" / "attempts/02"
    os.utime(latest, (queued + 20 * HOUR, queued + 20 * HOUR))
    assert _probes(tmp_path, now=queued + 21 * HOUR)[1] == []
    _section, [alert] = _probes(tmp_path, now=queued + 29 * HOUR)
    assert (alert["status"], alert["age_seconds"]) == ("unfunded_blocked", 9 * 3600)
    # An attempt that never recorded how it ended (its allocator's fate is unknown) is held for a person.
    (tmp_path / "rigid/jobs" / f"{KEY}-{'9' * 16}" / "attempts/03").mkdir()
    os.utime(tmp_path / "rigid/jobs" / f"{KEY}-{'9' * 16}" / "attempts/03", (queued + 30 * HOUR, queued + 30 * HOUR))
    assert _probes(tmp_path, now=queued + 31 * HOUR)[1] == []
    _section, [alert] = _probes(tmp_path, now=queued + 39 * HOUR)
    assert (alert["status"], alert["age_seconds"]) == ("attempt_unsettled", 9 * 3600)
    # An entry no attempt was ever staged for pages too, and says so.
    other = os.stat(_probe_entry(tmp_path / "other")).st_ctime
    _section, [alert] = stall.check_rigid_probes(_config(tmp_path, rigid_probe_queue_root=tmp_path / "other/rigid"),
                                                 now=other + 9 * HOUR, budget=_budget())
    assert alert["status"] == "no_attempt_recorded"


def test_rigid_probe_money_and_operator_holds_never_page(tmp_path: Path) -> None:
    queued = os.stat(_probe_entry(tmp_path)).st_ctime
    # The WebApp refused the capture's grant: a money stop the owner lifts; the runner asks again every 3 h.
    _probe_attempt(tmp_path, 0, {"state": "website_grant_refused",
                                 "allocator_blockers": ["rigid_robot_free_probe_website_grant_exhausted"]})
    section, alerts = _probes(tmp_path, now=queued + 5 * 86400)
    assert alerts == [] and section["dispositions"] == {"money_approval": 1}
    # A refusal that names a withdrawn capture.
    queued = os.stat(_probe_entry(tmp_path, key="cd" * 16)).st_ctime
    _probe_attempt(tmp_path, 0, {"state": "unfunded_blocked", "allocator_blockers": ["website_capture_withdrawn"]},
                   key="cd" * 16)
    section, alerts = _probes(tmp_path, now=queued + 5 * 86400)
    assert alerts == [] and section["dispositions"] == {"money_approval": 1, "privacy": 1}
    # A lane the operator switched off through the door.
    queued = os.stat(_probe_entry(tmp_path, key="ef" * 16)).st_ctime
    assert _pages(_probes(tmp_path, now=queued + 9 * HOUR)[1])
    assert _probes(tmp_path, now=queued + 9 * HOUR, holds=_hold(tmp_path, stall.RIGID_PROBE_TRIGGER)) == (
        {"status": "held", "trigger": "blueprint-rigid-robot-free-probe-runner.timer"}, [])


def _retired_refusal(tmp_path: Path, *, reason: str, sealed: float, markers: int = 0, key: str = KEY) -> Path:
    """An entry the runner retired with a staging refusal, and the ``refusal_retry`` set-asides it already had."""
    entry = _probe_entry(tmp_path, key=key)
    (tmp_path / "rigid/done").mkdir(exist_ok=True)
    retired = tmp_path / "rigid/done" / entry.name
    entry.rename(retired)
    job = tmp_path / "rigid/jobs" / f"{key}-{'9' * 16}"
    (job / "superseded_terminals").mkdir(parents=True, exist_ok=True)
    (job / "rigid_robot_free_probe_terminal.json").write_text(json.dumps({
        "status": "staging_refused", "detail": {"reason": reason}, "sealed_at_iso": _iso(sealed)}))
    for index in range(markers):
        (job / "superseded_terminals" / f"{index:02d}.supersession.json").write_text(json.dumps(
            {"trigger": "refusal_retry", "refusal_retry": index + 1}))
    return job


def test_a_retired_rigid_probe_waiting_for_its_timed_retry_is_quiet_until_overdue(tmp_path: Path) -> None:
    # Two timed retries used: the third waits 3 h after the refusal was sealed.
    _retired_refusal(tmp_path, reason="rigid_robot_free_probe_bundle_missing", sealed=NOW - 2 * HOUR, markers=2)
    section, alerts = _probes(tmp_path, now=NOW)
    assert alerts == [] and section["dispositions"] == {"scheduled_wait": 1}
    # Due seven hours ago: within the threshold.
    assert _probes(tmp_path, now=NOW + 8 * HOUR)[1] == []
    # Overdue by more than the threshold and still retired: the runner is not taking it.
    _section, [alert] = _probes(tmp_path, now=NOW + 10 * HOUR)
    assert (alert["code"], alert["status"], alert["age_seconds"]) == ("rigid_probe_stalled", "retry_overdue", 9 * 3600)
    assert alert["target"] == KEY and alert["blockers"] == ["rigid_robot_free_probe_bundle_missing"]
    assert stall.describe(alert) == ("rigid robot-free probe timed staging retry overdue 9.0h "
                                     f"[rigid_robot_free_probe_bundle_missing]: {KEY}")


def test_a_retired_rigid_probe_nothing_retries_by_the_clock_never_pages(tmp_path: Path) -> None:
    # A money refusal is never retried by the clock, a spent schedule is final, and a marker that does
    # not read counts as spent, as in the runner.
    _retired_refusal(tmp_path, reason="rigid_robot_free_probe_spend_ceiling_reached", sealed=NOW - 30 * 86400)
    _retired_refusal(tmp_path, reason="rigid_robot_free_probe_bundle_missing", sealed=NOW - 30 * 86400, markers=6,
                     key="cd" * 16)
    job = _retired_refusal(tmp_path, reason="rigid_robot_free_probe_bundle_missing", sealed=NOW - 30 * 86400,
                           key="ef" * 16)
    (job / "superseded_terminals/00.supersession.json").write_text("{not json")
    section, alerts = _probes(tmp_path, now=NOW)
    assert alerts == [] and section["dispositions"] == {}


def test_an_unreadable_rigid_probe_queue_is_incomplete_not_a_page(tmp_path: Path) -> None:
    assert _probes(tmp_path, now=NOW) == ({"status": "absent"}, [])
    path = _probe_entry(tmp_path)
    path.write_text("[]")
    section, alerts = _probes(tmp_path, now=os.stat(path).st_ctime + 9 * HOUR)
    assert section["unreadable"] == 1 and _pages(alerts) == []
    assert alerts == [{"code": stall.INCOMPLETE_CODE, "reason": "rigid_probe_entry_unreadable", "count": 1}]
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken/pending").write_text("not a directory")
    section, alerts = stall.check_rigid_probes(_config(tmp_path, rigid_probe_queue_root=tmp_path / "broken"),
                                               now=NOW, budget=_budget())
    assert section == {"status": "unreadable"} and _incomplete(alerts) == ["rigid_probe_queue_unreadable"]


# --------------------------------------------------------------------------- #
# Screening campaigns and their publications
# --------------------------------------------------------------------------- #


def _iso(epoch: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _campaign(rows: dict[str, dict], *, updated: float, revision: int = 3, published: int | None = None) -> dict:
    return {"plan_digest": "sha256:" + "1" * 64, "site_id": "site-req1", "revision": revision,
            "created_at": _iso(NOW - 2 * 86400), "updated_at": _iso(updated),
            "published_revision": revision if published is None else published, "published_digest": None,
            "candidates": rows}


def _ledger(tmp_path: Path, campaigns: dict[str, dict]) -> Path:
    root = tmp_path / "screening"
    root.mkdir(exist_ok=True)
    path = root / "screening_ledger.v1.json"
    path.write_text(json.dumps({"schema_version": "blueprint.beta_policy_screening_ledger.v1",
                                "campaigns": campaigns, "ledger_digest": "sha256:" + "0" * 64}))
    return path


def _screening(tmp_path: Path, **kwargs):
    return stall.check_screening(_config(tmp_path), now=NOW, budget=_budget(), **kwargs)


def test_a_healthy_screening_ledger_is_quiet(tmp_path: Path) -> None:
    _ledger(tmp_path, {
        # A run holds its row for up to its 4 h hard TTL without a ledger change.
        "running": _campaign({"a": {"status": "running", "blockers": []},
                              "b": {"status": "pending", "blockers": ["beta_screening_waiting_for_capacity"]}},
                             updated=NOW - 5 * HOUR),
        "finished": _campaign({"a": {"status": "completed", "blockers": []},
                               "b": {"status": "blocked", "blockers": ["beta_screening_gpu_capacity_unavailable"]},
                               "c": {"status": "unavailable", "blockers": ["candidate_infeasible_measured_geometry"]},
                               "d": {"status": "deferred", "blockers": []}}, updated=NOW - 30 * 86400),
    })
    section, alerts = _screening(tmp_path)
    assert alerts == []
    assert section["dispositions"] == {"finished": 1, "watched": 1} and section["campaigns"] == 2
    assert section["publications_pending"] == 0


def test_a_screening_campaign_without_ledger_progress_pages(tmp_path: Path) -> None:
    _ledger(tmp_path, {
        CAMPAIGN: _campaign({"a": {"status": "completed", "blockers": []},
                             "b": {"status": "pending", "blockers": ["beta_screening_waiting_for_capacity"]},
                             "c": {"status": "reconciliation_required", "blockers": []},
                             "d": {"status": "waitlisted", "blockers": ["site_screening_budget_insufficient"]}},
                            updated=NOW - 9 * HOUR),
        "fresh": _campaign({"a": {"status": "preparing", "blockers": []}}, updated=NOW - 7 * HOUR),
    })
    section, [alert] = _screening(tmp_path)
    assert alert == {"code": "screening_campaign_stalled", "count": 1, "targets": [CAMPAIGN], "target": CAMPAIGN,
                     "status": "reconciliation_required", "unfinished_rows": 2,
                     "blockers": ["beta_screening_waiting_for_capacity"], "age_seconds": 9 * 3600,
                     "threshold_seconds": 8 * 3600}
    assert section["stalled"] == 1 and section["dispositions"] == {"watched": 2}
    assert stall.describe(alert) == (
        "screening campaign without ledger progress 9.0h, 2 unfinished (reconciliation_required) "
        "[beta_screening_waiting_for_capacity]: campaign-site-req1-r1")


def test_screening_money_and_rights_holds_never_page(tmp_path: Path) -> None:
    _ledger(tmp_path, {
        # Only the site's allowance or the job's ceiling keeps these rows from running.
        "waitlisted": _campaign({"a": {"status": "completed", "blockers": []},
                                 "b": {"status": "waitlisted", "blockers": ["site_screening_budget_insufficient"]}},
                                updated=NOW - 20 * 86400),
        # Every row still to run waits on consent.
        "consent": _campaign({"a": {"status": "pending",
                                    "blockers": ["beta_screening_current_capture_consent_required"]}},
                             updated=NOW - 20 * 86400, published=2),
    })
    section, alerts = _screening(tmp_path)
    assert alerts == []
    assert section["dispositions"] == {"money_approval": 1, "privacy": 1}
    # Nothing is owed to the website for a job it ended itself: the unpublished revision is not tracked.
    assert section["publications_pending"] == 1 and section["publications_stalled"] == 0
    stale = {"consent": NOW - 20 * 86400}
    assert _screening(tmp_path, remembered=stale)[1] == []
    # The operator held the screening timer through the door.
    _ledger(tmp_path, {CAMPAIGN: _campaign({"a": {"status": "launching", "blockers": []}}, updated=NOW - 9 * HOUR)})
    assert _pages(_screening(tmp_path)[1])
    assert _screening(tmp_path, holds=_hold(tmp_path, stall.SCREENING_TRIGGER)) == (
        {"status": "held", "trigger": "blueprint-beta-policy-screening.timer"}, [])


def test_a_screening_row_waiting_for_its_timed_replan_is_quiet(tmp_path: Path) -> None:
    # Refused before allocation for a reason no rule owns: the coordinator plans it again on the shared
    # schedule. A blocked row is not a row still to run, so the wait is quiet. The ledger records no due
    # time for the re-plan, so an overdue one is not seen here.
    _ledger(tmp_path, {CAMPAIGN: _campaign({"a": {"status": "blocked",
                                                  "blockers": ["beta_screening_prelaunch_refused:scene_unbound"]}},
                                           updated=NOW - 30 * HOUR)})
    section, alerts = _screening(tmp_path)
    assert alerts == [] and section["dispositions"] == {"finished": 1}


def test_an_unpublished_screening_revision_pages_once_it_has_been_seen_long_enough(tmp_path: Path) -> None:
    done = {"a": {"status": "completed", "blockers": []}}
    _ledger(tmp_path, {CAMPAIGN: _campaign(done, updated=NOW - 6 * HOUR, published=2),
                       "published": _campaign(done, updated=NOW - 6 * HOUR)})
    # First seen: the ledger does not say since when (a restart resets published_revision on purpose).
    memory: dict = {}
    section, alerts = _screening(tmp_path, memory_out=memory)
    assert alerts == [] and memory == {CAMPAIGN: NOW} and section["publications_pending"] == 1
    # Still unpublished 100 minutes of ticks later.
    later: dict = {}
    section, [alert] = stall.check_screening(_config(tmp_path), now=NOW + 100 * MINUTE, budget=_budget(),
                                             remembered=memory, memory_out=later)
    assert alert == {"code": "screening_publication_stalled", "count": 1, "targets": [CAMPAIGN], "target": CAMPAIGN,
                     "age_seconds": 6000, "age_is_lower_bound": True, "threshold_seconds": 5400}
    assert later == {CAMPAIGN: NOW} and section["publications_stalled"] == 1
    assert stall.describe(alert) == "screening publication undelivered over 100m: campaign-site-req1-r1"
    # One tick that sees it published forgets it.
    _ledger(tmp_path, {CAMPAIGN: _campaign(done, updated=NOW - 6 * HOUR)})
    cleared: dict = {}
    assert stall.check_screening(_config(tmp_path), now=NOW + 110 * MINUTE, budget=_budget(), remembered=later,
                                 memory_out=cleared)[1] == [] and cleared == {}


def test_an_unreadable_screening_ledger_is_incomplete_not_a_page(tmp_path: Path) -> None:
    assert _screening(tmp_path) == ({"status": "absent"}, [])
    (tmp_path / "screening").mkdir()
    assert _screening(tmp_path) == ({"status": "absent"}, [])  # a queue with no ledger has no campaign
    path = _ledger(tmp_path, {})
    path.write_text("{not json")
    assert _screening(tmp_path) == ({"status": "unreadable"}, [
        {"code": stall.INCOMPLETE_CODE, "reason": "screening_ledger_unreadable"}])
    path.write_text(json.dumps({"schema_version": "other.v1", "campaigns": {}}))
    assert _incomplete(_screening(tmp_path)[1]) == ["screening_ledger_unreadable"]
    # One campaign that does not read never hides the others.
    _ledger(tmp_path, {"broken": {"candidates": "?"},
                       CAMPAIGN: _campaign({"a": {"status": "running", "blockers": []}}, updated=NOW - 9 * HOUR)})
    section, alerts = _screening(tmp_path)
    assert section["unreadable"] == 1 and [row["code"] for row in _pages(alerts)] == ["screening_campaign_stalled"]
    assert _incomplete(alerts) == ["screening_campaign_unreadable"]


# --------------------------------------------------------------------------- #
# A capture whose execution window is over is history, not a stall
# --------------------------------------------------------------------------- #


def _scene_intent(tmp_path: Path, name: str, capture: str, *, expires_at: float, revoked: bool = False) -> None:
    directory = tmp_path / "intents" / name
    directory.mkdir(parents=True)
    (directory / "intent.json").write_text(json.dumps({
        "intent_id": name, "accepted_at_epoch": NOW - 86400,
        "request": {"submission_id": capture, "execution": {"expires_at_epoch": expires_at}}}))
    if revoked:
        (directory / "revoked.json").write_text("{}")


def test_open_windows_are_the_unrevoked_unexpired_intents_captures(tmp_path: Path) -> None:
    _scene_intent(tmp_path, "scene-" + "a" * 64, "walkthrough-open", expires_at=NOW + 3600)
    _scene_intent(tmp_path, "scene-" + "b" * 64, "walkthrough-expired", expires_at=NOW - 60)
    _scene_intent(tmp_path, "scene-" + "c" * 64, "walkthrough-revoked", expires_at=NOW + 3600, revoked=True)
    # A renewal's later intent reopens the same capture.
    _scene_intent(tmp_path, "scene-" + "d" * 64, "walkthrough-renewed", expires_at=NOW - 86400)
    _scene_intent(tmp_path, "scene-" + "e" * 64, "walkthrough-renewed", expires_at=NOW + 86400)
    # A request whose re-issue marker does not verify still names its submission; one that
    # does not read names nothing.
    _scene_intent(tmp_path, "scene-" + "f" * 64, "walkthrough-marked", expires_at=NOW + 3600)
    marked = tmp_path / "intents" / ("scene-" + "f" * 64) / "intent.json"
    value = json.loads(marked.read_text())
    value["request"]["task"] = {"website_navigation_reissue": None}
    marked.write_text(json.dumps(value))
    (tmp_path / "intents" / ("scene-" + "0" * 64)).mkdir()
    (tmp_path / "intents" / ("scene-" + "0" * 64) / "intent.json").write_text("{not json")
    config = _config(tmp_path, scene_intent_root=tmp_path / "intents")
    assert stall.open_window_captures(config, now=NOW, budget=_budget()) == {
        "walkthrough-open", "walkthrough-renewed", "walkthrough-marked"}
    # Unknown is never "closed": no intent store, no answer, and every row stays watched.
    assert stall.open_window_captures(_config(tmp_path), now=NOW, budget=_budget()) is None
    assert stall.open_window_captures(_config(tmp_path, scene_intent_root=tmp_path / "absent"), now=NOW,
                                      budget=_budget()) is None
    assert stall._window_closed(lambda: None, "walkthrough-expired") is False
    assert stall._window_closed(lambda: {"walkthrough-open"}, None) is False
    assert stall._window_closed(lambda: {"walkthrough-open"}, "walkthrough-expired") is True


def test_a_capture_the_intent_scan_did_not_reach_is_never_classed_as_closed(tmp_path: Path, monkeypatch) -> None:
    """Review P2: only the newest 256 intents were read, and a capture left out counted as expired."""
    count = stall.MAX_INTENTS + 1
    for index in range(count):
        _scene_intent(tmp_path, f"scene-{index:064d}", f"walkthrough-{index:04d}", expires_at=NOW + 3600)
        os.utime(tmp_path / "intents" / f"scene-{index:064d}", (NOW - index, NOW - index))  # the last is the oldest
    config = _config(tmp_path, scene_intent_root=tmp_path / "intents")
    captures = stall.open_window_captures(config, now=NOW, budget=_budget())
    last = f"walkthrough-{count - 1:04d}"
    assert len(captures) == count and last in captures
    assert stall._window_closed(lambda: captures, last) is False
    # Its blocked request therefore pages.
    _team_status(tmp_path, "blocked", ["team_scene_source_launch_unavailable"], age=30 * HOUR)
    status = tmp_path / "team/status" / f"{KEY}.json"
    status.write_text(status.read_text().replace(CAPTURE, last))
    os.utime(status, (NOW - 30 * HOUR, NOW - 30 * HOUR))
    result = stall.observe_stalls(config, now=NOW, unit_reader=lambda _units: {})
    [page] = [row for row in result["alerts"] if row["code"] == "team_scene_packet_stalled"]
    assert page["label"] == last and "execution_window_scan_incomplete" not in _incomplete(result["alerts"])
    # A listing that was cut, or a scan the deadline ended, is unknown: nothing is hidden, and it is said once.
    spent = stall._Budget(deadline=0.0, events=0, clock=lambda: 1.0)
    assert stall.open_window_captures(config, now=NOW, budget=spent) is None
    monkeypatch.setattr(stall, "MAX_INTENT_DIRECTORY_ENTRIES", 10)
    assert stall.open_window_captures(config, now=NOW, budget=_budget()) is None
    result = stall.observe_stalls(config, now=NOW, unit_reader=lambda _units: {})
    assert [row["code"] for row in _pages(result["alerts"]) if row["code"] == "team_scene_packet_stalled"]
    assert _incomplete(result["alerts"]).count("execution_window_scan_incomplete") == 1


def test_a_store_row_of_a_capture_past_its_window_never_pages(tmp_path: Path) -> None:
    """A refusal from months ago must not page every hour for good."""
    _team_status(tmp_path, "blocked", ["team_scene_source_launch_unavailable"], age=60 * 86400)
    queued = os.stat(_probe_entry(tmp_path)).st_ctime
    (tmp_path / "rigid/pending" / f"{KEY}.json").write_text(json.dumps({
        "capture_root": f"/captures/scenes/site-req1/captures/{CAPTURE}", "probe_request_digest": REQUEST_DIGEST}))
    _ledger(tmp_path, {CAMPAIGN: _campaign({"a": {"status": "pending", "blockers": []}}, updated=NOW - 60 * 86400,
                                           published=2)})
    (tmp_path / "screening/plans").mkdir()
    (tmp_path / "screening/plans" / f"{CAMPAIGN}.json").write_text(json.dumps({"capture_id": CAPTURE}))
    later = max(NOW, queued) + 9 * HOUR
    asked: list[int] = []

    def closed() -> set[str]:
        asked.append(1)
        return {"walkthrough-other"}

    def run(windows):
        return (stall.check_team_scene_packets(_config(tmp_path), now=later, budget=_budget(), windows=windows),
                stall.check_rigid_probes(_config(tmp_path), now=later, budget=_budget(), windows=windows),
                stall.check_screening(_config(tmp_path), now=later, budget=_budget(), windows=windows,
                                      remembered={CAMPAIGN: NOW - 86400}))

    for section, alerts in run(closed):
        assert alerts == [] and section["dispositions"] == {"execution_window": 1}
    assert asked  # asked only because rows were past their thresholds
    # The same rows while the capture's window is open, or when no one can say: they page.
    for windows in (lambda: {CAPTURE}, lambda: None, None):
        codes = [row["code"] for _section, alerts in run(windows) for row in _pages(alerts)]
        assert codes == ["team_scene_packet_stalled", "rigid_probe_stalled", "screening_campaign_stalled",
                         "screening_publication_stalled"]
    # Rows inside their thresholds never ask at all.
    asked.clear()
    fresh = tmp_path / "fresh"
    moved = os.stat(_team_request(fresh, "pending")).st_ctime
    assert stall.check_team_scene_packets(_config(fresh), now=moved + MINUTE, budget=_budget(),
                                          windows=closed)[1] == [] and not asked


def test_one_tick_reads_the_windows_from_the_intent_store(tmp_path: Path) -> None:
    _team_status(tmp_path, "blocked", ["team_scene_source_launch_unavailable"], age=30 * HOUR)
    _team_status(tmp_path, "blocked", ["team_scene_source_launch_unavailable"], key="cd" * 16, age=30 * HOUR)
    status = tmp_path / "team/status" / f"{'cd' * 16}.json"
    status.write_text(status.read_text().replace(CAPTURE, "walkthrough-old"))
    os.utime(status, (NOW - 30 * HOUR, NOW - 30 * HOUR))
    _scene_intent(tmp_path, "scene-" + "a" * 64, CAPTURE, expires_at=NOW + 3600)
    _scene_intent(tmp_path, "scene-" + "b" * 64, "walkthrough-old", expires_at=NOW - 30 * 86400)
    # A status written before its request was read names no capture; the retained request does.
    _team_status(tmp_path, "blocked", ["team_scene_request_invalid"], key="ef" * 16, age=30 * HOUR)
    unnamed = tmp_path / "team/status" / f"{'ef' * 16}.json"
    unnamed.write_text(json.dumps({"request_key": "ef" * 16, "status": "blocked",
                                   "blockers": ["team_scene_request_invalid"]}))
    os.utime(unnamed, (NOW - 30 * HOUR, NOW - 30 * HOUR))
    (tmp_path / "team/processed" / f"{'ef' * 16}.json").write_text(json.dumps({"capture_id": "walkthrough-old"}))
    result = stall.observe_stalls(_config(tmp_path, scene_intent_root=tmp_path / "intents"), now=NOW,
                                  unit_reader=lambda _units: {})
    [page] = [row for row in result["alerts"] if row["code"] == "team_scene_packet_stalled"]
    assert page["targets"] == [KEY] and page["label"] == CAPTURE
    assert result["checks"]["team_scene_packets"]["dispositions"] == {"blocked": 1, "execution_window": 2}


# --------------------------------------------------------------------------- #
# One tick, configuration and the capacity route
# --------------------------------------------------------------------------- #


def test_one_tick_runs_every_store_check_and_carries_the_publication_memory(tmp_path: Path) -> None:
    _atlas_job(tmp_path, updated_at_epoch=NOW - 2 * HOUR, next_attempt_at_epoch=NOW - 2 * HOUR)
    _ledger(tmp_path, {CAMPAIGN: _campaign({"a": {"status": "completed", "blockers": []}}, updated=NOW - HOUR,
                                           published=2)})
    first = stall.observe_stalls(_config(tmp_path), now=NOW, unit_reader=lambda _units: {})
    assert {"atlas_preview", "team_scene_packets", "rigid_probes", "screening"} <= set(first["checks"])
    assert [row["code"] for row in _pages(first["alerts"])] == ["atlas_preview_stalled"]
    assert first["memory"]["screening_publication"] == {CAMPAIGN: NOW}
    assert first["thresholds_seconds"]["atlas"] == 5400 and first["thresholds_seconds"]["publication"] == 5400
    assert {first["thresholds_seconds"][name] for name in ("team_scene", "rigid_probe", "screening")} == {8 * 3600}
    second = stall.observe_stalls(_config(tmp_path), now=NOW + 2 * HOUR, previous=first,
                                  unit_reader=lambda _units: {})
    assert {row["code"] for row in _pages(second["alerts"])} == {"atlas_preview_stalled",
                                                                "screening_publication_stalled"}
    # The door's holds directory comes from the deploy tool, read once for the tick.
    holds = _hold(tmp_path, stall.ATLAS_TRIGGER)
    tool = tmp_path / "deploy_tool.py"
    tool.write_text(f"DEFAULT_DOOR_HOLDS_DIR = {str(holds)!r}\nDEFAULT_ALWAYS_ARM_PATH_UNITS = ()\n"
                    "DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS = ()\nDEFAULT_ALWAYS_ARM_TIMER_UNITS = ()\n")
    held = stall.observe_stalls(_config(tmp_path, deploy_tool=tool), now=NOW, unit_reader=lambda _units: {})
    assert held["checks"]["atlas_preview"]["status"] == "held" and _pages(held["alerts"]) == []
    assert held["checks"]["triggers"]["status"] == "observed"


def test_store_roots_and_thresholds_come_from_the_capacity_units_environment() -> None:
    config = stall.config_from_environment({
        stall.ATLAS_STORAGE_ROOT_ENV: "/srv/handoffs", stall.TEAM_SCENE_QUEUE_ROOT_ENV: "/srv/team",
        stall.RIGID_PROBE_QUEUE_ROOT_ENV: "/srv/rigid", stall.SCREENING_QUEUE_ROOT_ENV: "relative/never",
        stall.ATLAS_MINUTES_ENV: "30", stall.SCREENING_MINUTES_ENV: "0", stall.PUBLICATION_MINUTES_ENV: "soon",
    })
    assert (config.atlas_storage_root, config.team_scene_queue_root, config.rigid_probe_queue_root,
            config.screening_queue_root) == (Path("/srv/handoffs"), Path("/srv/team"), Path("/srv/rigid"), None)
    assert config.atlas_seconds == 30 * MINUTE
    # Defaults are the two existing waits; an unusable value keeps the default and is reported.
    assert (config.team_scene_seconds, config.rigid_probe_seconds, config.screening_seconds) == (8 * HOUR,) * 3
    assert config.publication_seconds == 90 * MINUTE == stall.DEFAULT_PROGRESSION_MINUTES * MINUTE
    assert config.config_errors == (f"threshold_invalid:{stall.SCREENING_MINUTES_ENV}",
                                    f"threshold_invalid:{stall.PUBLICATION_MINUTES_ENV}")
    # The capacity unit names each store exactly as the store's own worker unit does.
    units = REPO / "deploy" / "systemd"
    capacity = (units / "blueprint-control-plane-capacity.service").read_text().splitlines()
    for name, owner in ((stall.ATLAS_STORAGE_ROOT_ENV, "blueprint-website-atlas-preview.service"),
                        (stall.TEAM_SCENE_QUEUE_ROOT_ENV, "blueprint-team-scene-packet.service"),
                        (stall.RIGID_PROBE_QUEUE_ROOT_ENV, "blueprint-rigid-robot-free-probe-runner.service"),
                        (stall.SCREENING_QUEUE_ROOT_ENV, "blueprint-beta-policy-screening.service")):
        line = next(row for row in capacity if row.startswith(f"Environment={name}="))
        assert line in (units / owner).read_text().splitlines(), name


def test_the_mirrored_store_names_are_their_owners() -> None:
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from blueprint_pipeline import beta_policy_screening_result as result
    from blueprint_pipeline import rigid_robot_free_probe_queue_root as probe_queue
    from blueprint_pipeline import rigid_robot_free_probe_runner as probe_runner
    from blueprint_pipeline import team_scene_capture_root as team_queue
    from blueprint_pipeline import team_scene_packet_worker as team_worker
    from blueprint_pipeline import website_atlas_preview as preview
    from blueprint_pipeline import website_atlas_preview_store as store

    assert stall.ATLAS_JOB_SCHEMA == store.SCHEMA and stall.ATLAS_JOB_GLOB.endswith("/pipeline/atlas/" + store.JOB_FILE)
    assert stall.ATLAS_TRIGGER == preview.TRIGGER_UNIT
    assert Path(deploy.DEFAULT_DOOR_HOLDS_DIR) / f"{stall.ATLAS_TRIGGER}.json" == preview.HOLD_RECORD
    assert stall.TEAM_SCENE_QUEUE_ROOT_ENV == team_queue.QUEUE_ROOT_ENV
    assert {*stall.TEAM_SCENE_WAITING_STATES, "status"} < set(team_queue.QUEUE_STATES)
    assert stall.TEAM_SCENE_OWNER_CONTRACT_PREFIX == team_worker.OWNER_CONTRACT_BLOCKER_PREFIX
    assert stall.RIGID_PROBE_QUEUE_ROOT_ENV == probe_queue.QUEUE_ENV and "pending" in probe_queue.ENTRY_DIRECTORIES
    assert stall.RIGID_PROBE_INGEST_FILENAME == probe_runner.INGEST_FILENAME
    assert stall.RIGID_PROBE_GRANT_REFUSED_STATE == probe_runner.GRANT_REFUSED_STATE
    assert stall.RIGID_PROBE_TERMINAL_FILENAME == probe_runner.TERMINAL_FILENAME
    assert stall.RIGID_PROBE_TIMED_STATES == probe_runner.RELEASE_RETRIED_STATES
    assert stall.RIGID_PROBE_SUPERSEDED_DIRNAME == probe_runner.SUPERSEDED_DIRNAME
    assert stall.RIGID_PROBE_REFUSAL_RETRY_TRIGGER == probe_runner.REFUSAL_RETRY_TRIGGER
    assert set(stall.TEAM_SCENE_RETRY_FIELDS) == {"automatic_retry", team_worker.REFUSAL_RETRY_FIELD}
    assert stall.SCREENING_QUEUE_ROOT_ENV == coordinator.QUEUE_ROOT_ENV
    assert stall.SCREENING_LEDGER_FILENAME == coordinator.LEDGER_FILENAME
    assert stall.SCREENING_LEDGER_SCHEMA == coordinator.LEDGER_SCHEMA
    assert stall.SCREENING_LIVE_STATUSES | {stall.SCREENING_WAITLISTED_STATUS} == (
        result.QUEUED_STATUSES | result.ACTIVE_STATUSES)
    # The defaults rest on these: a screening run's and a probe attempt's longest hard TTL, the
    # team scene worker's own retries, and the Atlas job's own lifetime.
    from blueprint_pipeline import beta_policy_screening_capabilities as capabilities
    from blueprint_pipeline import rigid_robot_free_probe_lane as probe_lane
    assert stall.DEFAULT_SCREENING_MINUTES * 60 >= 2 * capabilities.DEFAULT_HARD_TTL_SECONDS
    assert stall.DEFAULT_RIGID_PROBE_MINUTES * 60 >= 2 * probe_lane.HARD_TTL_SECONDS + probe_runner.GRANT_REFUSAL_RETRY_SECONDS
    assert stall.DEFAULT_TEAM_SCENE_MINUTES * 60 > sum(team_worker.AUTOMATIC_RETRY_BACKOFF_SECONDS)
    from blueprint_pipeline import job_retry_policy
    # An outage's longest backoff is shorter than the threshold, so a live outage never looks overdue.
    assert stall.DEFAULT_ATLAS_MINUTES * 60 > job_retry_policy.outage_retry_delay(10**6)
    assert stall.DEFAULT_ATLAS_MINUTES * 60 < store.MAXIMUM_JOB_SECONDS
    assert set(stall._SCREENING_STATUS_ORDER) == stall.SCREENING_LIVE_STATUSES
    # Each store's trigger is one the deploy always arms, so the trigger check watches it too.
    armed, _holds = stall.read_deploy_trigger_contract(REPO / "scripts" / "deploy_control_plane_commit.py")
    assert {stall.ATLAS_TRIGGER, stall.TEAM_SCENE_TRIGGER, stall.RIGID_PROBE_TRIGGER,
            stall.SCREENING_TRIGGER} <= set(armed)


def test_the_alarm_loads_none_of_the_workers_it_watches() -> None:
    script = (
        "import sys\n"
        "from blueprint_pipeline import control_plane_stall_alarm as stall\n"
        "from pathlib import Path\n"
        "stall.open_window_captures(stall.StallConfig(Path(sys.argv[1]), None, None, None), now=0.0, "
        "budget=stall._Budget(deadline=float('inf'), events=1, clock=lambda: 0.0))\n"
        "stall.observe_stalls(stall.StallConfig(None, None, None, None, atlas_storage_root=Path(sys.argv[1]), "
        "team_scene_queue_root=Path(sys.argv[1]), rigid_probe_queue_root=Path(sys.argv[1]), "
        "screening_queue_root=Path(sys.argv[1])), unit_reader=lambda units: {})\n"
        "heavy = sorted(name for name in sys.modules if name.startswith('blueprint_pipeline.') and any(part in name "
        "for part in ('website_atlas', 'atlas_preparation', 'team_scene', 'rigid_robot_free_probe', "
        "'beta_policy_screening', 'paid_resource', 'vast_', 'torch')))\n"
        "print(heavy + sorted(name for name in ('torch', 'numpy', 'cv2') if name in sys.modules))\n")
    done = subprocess.run([sys.executable, "-c", script, str(REPO / "docs")], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=120, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "[]"


def test_a_store_stall_pages_on_the_capacity_route_and_repeats_hourly(tmp_path: Path, monkeypatch) -> None:
    posted: list = []
    page = {"code": "atlas_preview_stalled", "count": 1, "targets": [CAPTURE], "target": CAPTURE,
            "phase": "source_pose", "status": "silent", "age_seconds": 7200, "threshold_seconds": 5400}

    def tick(now: float, alerts: list[dict]) -> dict:
        return cap.run_controller(
            mounts=[str(tmp_path)], report_root=tmp_path / "capacity", reservation_root=tmp_path / "r",
            webhook_url="https://alerts.example/hook", volume=None, ack="", token="", survey=None,
            disk_usage=lambda _path: SimpleNamespace(total=100 * 1024**3, used=10 * 1024**3, free=90 * 1024**3),
            now=now, poster=lambda _url, value: posted.append(value),
            release_retirement_summary_path=tmp_path / "absent-retirement",
            break_glass_notes_root=tmp_path / "absent-notes", storage_gc_summary_path=tmp_path / "absent-gc",
            stall_check=lambda observed, _prior: {"schema_version": stall.SCHEMA_VERSION,
                                                  "observed_at_epoch": observed, "checks": {},
                                                  "alerts": [dict(row) for row in alerts]})

    report = tick(NOW, [page])
    row = next(a for a in report["alerts"] if a["code"] == "atlas_preview_stalled")
    assert row["severity"] == "page" and report["level"] == "ok" and report["alert_posted"] is True
    # The door-readable summary names the job, where it stopped and for how long.
    [shown] = [a for a in cap.capacity_summary(report)["alerts"] if a["code"] == "atlas_preview_stalled"]
    assert (shown["target"], shown["status"], shown["phase"], shown["age_seconds"]) == (
        CAPTURE, "silent", "source_pose", 7200)
    # Persisting: quiet inside the hour, then the hourly repeat; a newly stalled job posts at once.
    assert tick(NOW + 600, [page])["alert_posted"] is False
    assert tick(NOW + 3601, [page])["alert_posted"] is True
    more = {**page, "count": 2, "targets": [CAPTURE, "walkthrough-req2"]}
    assert tick(NOW + 3700, [more])["alert_posted"] is True and len(posted) == 3
    for code in ("team_scene_packet_stalled", "rigid_probe_stalled", "screening_campaign_stalled",
                 "screening_publication_stalled", "atlas_preview_stalled"):
        assert cap._severity(code) == "page"
        assert stall.describe({"code": code, "target": "t", "age_seconds": 7200, "count": 1})
    # The webhook's summary leads with the stall, as for every stall page.
    payloads: list = []

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(cap.urllib.request, "urlopen",
                        lambda request, *, timeout: payloads.append(json.loads(request.data)) or Response())
    cap.post_alert("https://alerts.example/hook", {"level": "ok", "mounts": [],
                                                   "alerts": [{**page, "severity": "page"}]})
    assert payloads[0]["page"] is True and payloads[0]["summary"].startswith(
        "Atlas preview job processing with no finished step 2.0h at stage source_pose: walkthrough-req1")
    assert payloads[0]["stall_runbook"] == "docs/runbooks/control-plane-capacity.md#stalled-jobs"
