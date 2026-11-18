# Covers (for impacted-test selection):
#   src/blueprint_pipeline/atlas_refined_splat_retry.py
#   src/blueprint_pipeline/website_atlas_pipeline.py
#   src/blueprint_pipeline/website_atlas_preview.py
#   src/blueprint_pipeline/preparation_fault_alert.py
"""A refined splat that still fails waits for a fresh real-frames-only attempt instead of settling for the draft.

The draft becomes the look for good only when the job's allowance cannot cover another splat and mesh, or
within two hours of the window's end; a task that may still finish is never bought again.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import atlas_refined_splat_retry as retry
from blueprint_pipeline.job_retry_policy import REFUSAL_RETRY_DELAYS_SECONDS

NOW = 1_800_000_000.0
NEEDED = 5.5


def _submission(base: Path, status: str, *, form_suffix: str = "-minimal") -> None:
    """One operation's receipt under a form root of ``base``: failed (closed) or submitted (may still finish)."""
    state = {"status": status, "operation_id": "op"}
    if status == "failed":
        state["terminal_operation"] = {"id": "op", "done": True, "error": {"code": 13}}
    path = base.with_name(base.name + form_suffix) / "native/atlas_pose/submission.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))


def _step(tmp_path, *, now, run_attempt=None, remaining=23.0, end=None):
    calls = []

    def attempt(base):
        calls.append(base)
        if run_attempt is None:
            pytest.fail("an attempt was run")
        return run_attempt(base)
    value = retry.next_refined_splat(output_root=tmp_path, original_root=tmp_path / "refined_splat",
                                     run_attempt=attempt, remaining_cost_usd=lambda: remaining, needed_usd=NEEDED,
                                     deadline=lambda: end, now=now)
    return value, calls


def test_a_failure_waits_on_the_retry_schedule_then_a_fresh_real_frames_attempt_succeeds(tmp_path):
    _submission(tmp_path / "refined_splat", "failed")
    end = NOW + 5 * 86400
    value, _ = _step(tmp_path, now=NOW, end=end)
    assert value == {"status": "processing", "stage": retry.WAIT_STAGE, "attempt": 1,
                     "next_attempt_at_epoch": NOW + REFUSAL_RETRY_DELAYS_SECONDS[0]}
    # Before it is due nothing is run or bought, and the wait is unchanged.
    assert _step(tmp_path, now=NOW + 60, end=end)[0] == value
    due = value["next_attempt_at_epoch"]

    def submitted(base):
        _submission(base, "submitted")
        return {"status": "processing", "stage": "refined_splat", "operation_id": "op-2"}
    value, calls = _step(tmp_path, now=due, end=end, run_attempt=submitted)
    assert value["status"] == "processing" and calls == [retry.attempt_root(tmp_path, 1)]
    # Once sent, the attempt is resumed from its receipts on every pass until it is done.
    done = {"status": "completed", "splat_root": "x", "splat_manifest": {"digest": "d"}}
    value, calls = _step(tmp_path, now=due + 600, end=end, run_attempt=lambda base: done)
    assert value == done and calls == [retry.attempt_root(tmp_path, 1)]
    record = json.loads((tmp_path / retry.RECORD).read_text())
    assert [row["root"] for row in record["attempts"]] == ["refined_splat-retry-1"] and record["final"] is None


def _fails(base):
    _submission(base, "failed")
    raise ValueError("atlas_task_operation_failed")


def test_repeated_failures_follow_the_schedule_and_then_daily_while_the_window_is_open(tmp_path):
    _submission(tmp_path / "refined_splat", "failed")
    end = NOW + 30 * 86400
    value, _ = _step(tmp_path, now=NOW, end=end)
    delays = [value["next_attempt_at_epoch"] - NOW]
    for attempt in range(1, 9):
        now = value["next_attempt_at_epoch"]
        value, calls = _step(tmp_path, now=now, end=end, run_attempt=_fails)
        assert calls == [retry.attempt_root(tmp_path, attempt)] and value["attempt"] == attempt + 1
        delays.append(value["next_attempt_at_epoch"] - now)
    assert delays[:6] == list(REFUSAL_RETRY_DELAYS_SECONDS) and delays[6:] == [REFUSAL_RETRY_DELAYS_SECONDS[-1]] * 3


def test_an_allowance_too_small_takes_the_draft_at_once(tmp_path):
    _submission(tmp_path / "refined_splat", "failed")
    value, _ = _step(tmp_path, now=NOW, remaining=5.0, end=NOW + 5 * 86400)
    assert value == {"status": retry.DRAFT_FINAL, "retry": {"attempts": 0, "final_reason": retry.UNAFFORDABLE,
                                                            "deadline_epoch": NOW + 5 * 86400,
                                                            "remaining_cost_usd": 5.0}}
    # Final stays final: nothing is asked or run again.
    assert _step(tmp_path, now=NOW + 86400, remaining=100.0)[0] == value


def test_an_allowance_spent_before_a_due_attempt_takes_the_draft_without_buying(tmp_path):
    _submission(tmp_path / "refined_splat", "failed")
    value, _ = _step(tmp_path, now=NOW, end=NOW + 5 * 86400)
    value, _ = _step(tmp_path, now=value["next_attempt_at_epoch"], remaining=1.0, end=NOW + 5 * 86400)
    assert value["status"] == retry.DRAFT_FINAL and value["retry"]["final_reason"] == retry.UNAFFORDABLE


def test_the_window_end_takes_the_draft(tmp_path):
    _submission(tmp_path / "refined_splat", "failed")
    # Within two hours of the window's end: the draft at once.
    value, _ = _step(tmp_path, now=NOW, end=NOW + retry.LAST_CALL_SECONDS - 1)
    assert value["status"] == retry.DRAFT_FINAL and value["retry"]["final_reason"] == retry.DEADLINE_CLOSING


def test_the_last_attempt_is_set_at_the_start_of_the_last_stretch(tmp_path):
    _submission(tmp_path / "refined_splat", "failed")
    end = NOW + retry.LAST_CALL_SECONDS + 600
    value, _ = _step(tmp_path, now=NOW, end=end)
    assert value["next_attempt_at_epoch"] == end - retry.LAST_CALL_SECONDS  # earlier than the 15-minute step

    value, calls = _step(tmp_path, now=value["next_attempt_at_epoch"], end=end, run_attempt=_fails)
    assert len(calls) == 1 and value["retry"]["final_reason"] == retry.DEADLINE_CLOSING


def test_a_window_that_ended_before_a_due_attempt_takes_the_draft(tmp_path):
    _submission(tmp_path / "refined_splat", "failed")
    value, _ = _step(tmp_path, now=NOW, end=NOW + 3 * 86400)
    value, _ = _step(tmp_path, now=value["next_attempt_at_epoch"], end=NOW)
    assert value["retry"]["final_reason"] == retry.DEADLINE_CLOSING


def test_an_unreadable_window_gets_the_six_timed_retries_then_the_draft(tmp_path):
    _submission(tmp_path / "refined_splat", "failed")
    now = NOW
    for _ in REFUSAL_RETRY_DELAYS_SECONDS:
        value, _ = _step(tmp_path, now=now)
        now = value["next_attempt_at_epoch"]

        value, _ = _step(tmp_path, now=now, run_attempt=_fails)
    assert value["retry"] == {"attempts": 6, "final_reason": retry.EXHAUSTED, "deadline_epoch": None,
                              "remaining_cost_usd": 23.0}


def test_a_first_attempt_that_may_still_finish_is_never_bought_again(tmp_path):
    _submission(tmp_path / "refined_splat", "failed", form_suffix="-whole")
    _submission(tmp_path / "refined_splat", "submitted")
    with pytest.raises(ValueError, match="atlas_task_operation_failed"):
        _step(tmp_path, now=NOW, end=NOW + 5 * 86400)
    assert not (tmp_path / retry.RECORD).exists()


def test_a_retry_in_flight_is_resumed_and_never_resubmitted(tmp_path):
    _submission(tmp_path / "refined_splat", "failed")
    value, _ = _step(tmp_path, now=NOW, end=NOW + 5 * 86400)
    _submission(retry.attempt_root(tmp_path, 1), "submitted")
    # Sent already: it is resumed at once, whatever its due time, and nothing new is scheduled.
    value, calls = _step(tmp_path, now=NOW + 1, end=NOW + 5 * 86400,
                         run_attempt=lambda base: {"status": "processing", "stage": "refined_splat"})
    assert value["stage"] == "refined_splat" and calls == [retry.attempt_root(tmp_path, 1)]

    def unreconciled(base):
        raise ValueError("atlas_task_operation_failed")  # its receipt still holds a live operation
    with pytest.raises(ValueError, match="atlas_task_operation_failed"):
        _step(tmp_path, now=NOW + 2, end=NOW + 5 * 86400, run_attempt=unreconciled)
    record = json.loads((tmp_path / retry.RECORD).read_text())
    assert len(record["attempts"]) == 1 and record["attempts"][0].get("failed_at_epoch") is None


def test_a_privacy_or_rights_wall_raised_by_an_attempt_stays_a_stop(tmp_path):
    from blueprint_pipeline.hard_stops import PrivacyRightsStop
    _submission(tmp_path / "refined_splat", "failed")
    value, _ = _step(tmp_path, now=NOW, end=NOW + 5 * 86400)

    def refused(base):
        raise PrivacyRightsStop("website_atlas_consent_expired")
    with pytest.raises(PrivacyRightsStop):
        _step(tmp_path, now=value["next_attempt_at_epoch"], end=NOW + 5 * 86400, run_attempt=refused)
    record = json.loads((tmp_path / retry.RECORD).read_text())
    assert record["final"] is None and record["attempts"][0].get("failed_at_epoch") is None


@pytest.mark.parametrize("wall", ["typed", "admission", "untyped"])
def test_a_money_refusal_on_a_retry_takes_the_usable_draft_instead_of_parking(tmp_path, wall):
    from blueprint_pipeline.hard_stops import MoneyStop
    from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionBlocked
    _submission(tmp_path / "refined_splat", "failed")
    value, _ = _step(tmp_path, now=NOW, end=NOW + 5 * 86400)
    error = {"typed": MoneyStop("website_atlas_spend_reservation_refused"),
             "admission": PaidResourceAdmissionBlocked(["account_monthly_spend_limit_reached"]),
             "untyped": ValueError("website_scene_budget_exhausted")}[wall]

    def refused(base):
        raise error
    value, calls = _step(tmp_path, now=value["next_attempt_at_epoch"], end=NOW + 5 * 86400, run_attempt=refused)
    assert len(calls) == 1 and value["status"] == retry.DRAFT_FINAL
    assert value["retry"]["final_reason"] == retry.MONEY_REFUSED and value["retry"]["refusal"]
    # Final: nothing more is asked or bought for the look.
    assert _step(tmp_path, now=NOW + 86400, end=NOW + 5 * 86400)[0] == value


def test_a_money_refusal_with_an_attempt_in_flight_waits_on_it(tmp_path):
    from blueprint_pipeline.hard_stops import MoneyStop
    _submission(tmp_path / "refined_splat", "failed")
    value, _ = _step(tmp_path, now=NOW, end=NOW + 5 * 86400)
    _submission(retry.attempt_root(tmp_path, 1), "submitted")

    def refused(base):
        raise MoneyStop("website_atlas_spend_reservation_refused")
    with pytest.raises(MoneyStop):
        _step(tmp_path, now=NOW + 1, end=NOW + 5 * 86400, run_attempt=refused)
    assert json.loads((tmp_path / retry.RECORD).read_text())["final"] is None


def _run_until_final(tmp_path, *, now, deadline):
    """Every attempt fails; the moments each is due, and the final outcome."""
    dues = []
    value = retry.next_refined_splat(output_root=tmp_path, original_root=tmp_path / "refined_splat",
        run_attempt=_fails, remaining_cost_usd=lambda: 23.0, needed_usd=NEEDED, deadline=deadline, now=now)
    while value["status"] != retry.DRAFT_FINAL:
        dues.append(value["next_attempt_at_epoch"])
        value = retry.next_refined_splat(output_root=tmp_path, original_root=tmp_path / "refined_splat",
            run_attempt=_fails, remaining_cost_usd=lambda: 23.0, needed_usd=NEEDED, deadline=deadline,
            now=value["next_attempt_at_epoch"])
    return dues, value


def _job_file(root: Path, enqueued: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "preview_job.json").write_text(json.dumps({"enqueued_at_epoch": enqueued, "state": "processing"}))


def test_a_window_past_the_jobs_own_life_ends_on_the_draft_before_the_job_expires(tmp_path, monkeypatch):
    """A grant issued after enqueue (or renewed, or extended) can end after the Atlas job's 7-day life."""
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    from blueprint_pipeline.website_atlas_preview_store import MAXIMUM_JOB_SECONDS
    enqueued = NOW - 86400
    _job_file(tmp_path, enqueued)
    monkeypatch.setattr(pipeline, "_job_window_end", lambda *_: enqueued + 10 * 86400)
    life = enqueued + MAXIMUM_JOB_SECONDS
    assert pipeline._job_deadline(tmp_path, {}) == life
    _submission(tmp_path / "refined_splat", "failed")
    dues, value = _run_until_final(tmp_path, now=NOW, deadline=lambda: pipeline._job_deadline(tmp_path, {}))
    assert value["retry"]["final_reason"] == retry.DEADLINE_CLOSING and value["retry"]["deadline_epoch"] == life
    assert max(dues) == life - retry.LAST_CALL_SECONDS  # the last attempt, and the draft at once after it


def test_an_unreadable_window_late_in_the_job_ends_on_the_jobs_own_deadline(tmp_path, monkeypatch):
    """The first failure after about day 5: six timed retries (about 46 h) would outlive the job."""
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    from blueprint_pipeline.website_atlas_preview_store import MAXIMUM_JOB_SECONDS
    enqueued = NOW - int(5.5 * 86400)
    _job_file(tmp_path, enqueued)
    monkeypatch.setattr(pipeline, "_job_window_end", lambda *_: None)
    _submission(tmp_path / "refined_splat", "failed")
    dues, value = _run_until_final(tmp_path, now=NOW, deadline=lambda: pipeline._job_deadline(tmp_path, {}))
    life = enqueued + MAXIMUM_JOB_SECONDS
    # Not the six retries' own end, about 46 h on: the last attempt at the job's last stretch, then the draft.
    assert value["retry"]["final_reason"] == retry.DEADLINE_CLOSING and value["retry"]["deadline_epoch"] == life
    assert max(dues) == life - retry.LAST_CALL_SECONDS


def test_the_pipeline_waits_then_adopts_the_retried_splat(tmp_path, monkeypatch):
    """``_preserved_appearance``: the first forms fail, the job waits, the fresh attempt's splat is the look."""
    from blueprint_pipeline import atlas_native_splat
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    from blueprint_pipeline.atlas_refined_splat_forms import MINIMAL

    _submission(tmp_path / "refined_splat", "failed")
    clock = [NOW]
    monkeypatch.setattr(pipeline.time, "time", lambda: clock[0])
    monkeypatch.setattr(pipeline, "load_configuration", lambda *args: {"remaining_cost_usd": 23})
    monkeypatch.setattr(pipeline, "_job_window_end", lambda *_: NOW + 5 * 86400)
    calls = []

    def native(**options):
        calls.append((options["output_root"], options.get("forms")))
        if options["output_root"] == tmp_path / "refined_splat":
            raise ValueError("atlas_task_operation_failed")
        return {"status": "completed", "splat_root": str(tmp_path / "refined_splat-retry-1-minimal"),
                "splat_manifest": {"digest": "sha256:" + "e" * 64}}
    monkeypatch.setattr(atlas_native_splat, "run_native_splat", native)
    options = dict(frames=[{"frame_id": "a"}], pose={"digest": "pose"}, draft={"digest": "draft"},
                   output_root=tmp_path, policy={"refined_splat_cost_usd": 5, "mesh_export_cost_usd": .5},
                   task_context={"context_digest": "task"}, source_digest="capture", configuration={},
                   check=lambda _stage: None, experimental=False, max_polls=1)
    value = pipeline._preserved_appearance(**options)
    assert value["stage"] == retry.WAIT_STAGE and value["next_attempt_at_epoch"] == NOW + 900
    clock[0] = NOW + 900
    value = pipeline._preserved_appearance(**options)
    assert value["basis"] == pipeline.REFINED_APPEARANCE
    assert value["splat_root"] == str(tmp_path / "refined_splat-retry-1-minimal")
    assert calls[-1] == (retry.attempt_root(tmp_path, 1), (MINIMAL,))


def test_the_controller_waits_until_the_step_says_it_is_due(tmp_path, monkeypatch):
    from blueprint_pipeline import website_atlas_preview as preview
    job = {"state": "processing", "reason": None, "stage": "refined_splat", "revision": 1,
           "enqueued_at_epoch": NOW - 100, "next_attempt_at_epoch": NOW, "retry_at_epoch": None}
    wait = {"status": "processing", "stage": retry.WAIT_STAGE, "next_attempt_at_epoch": NOW + 900}
    out = preview.advance_preparation_step(job, job_dir=tmp_path, source_video=tmp_path / "v.mov", task_context={},
                                           checkpoint=lambda *_a, **_k: None, now=lambda: NOW,
                                           run_step=lambda: wait)
    assert out["state"] == "processing" and out["next_attempt_at_epoch"] == NOW + 900
    polling = {"status": "processing", "stage": "refined_splat"}
    out = preview.advance_preparation_step(job, job_dir=tmp_path, source_video=tmp_path / "v.mov", task_context={},
                                           checkpoint=lambda *_a, **_k: None, now=lambda: NOW,
                                           run_step=lambda: polling)
    assert out["next_attempt_at_epoch"] == NOW


def test_a_draft_look_for_good_pages_once(tmp_path):
    from blueprint_pipeline import control_plane_stall_alarm as stall
    from blueprint_pipeline import preparation_fault_alert as fault
    from blueprint_pipeline import control_plane_capacity_controller as capacity
    for capture, fields in (("a", {"state": "ready", "look_fallback": "atlas_refined_splat_provider_failed"}),
                            ("b", {"state": "ready"}), ("c", {"state": "processing", "look_fallback": "x"})):
        path = tmp_path / f"bucket/scenes/site-{capture}/captures/walkthrough-{capture}/pipeline/atlas/preview_job.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(fields))
    memory: dict = {}
    section, alerts = fault.check_preparation_faults(tmp_path, now=NOW, memory_out=memory)
    assert section["draft_looks_announced"] == 1 and [row["code"] for row in alerts] == [fault.LOOK_CODE]
    assert alerts[0]["label"] == "walkthrough-a" and alerts[0]["blockers"] == ["atlas_refined_splat_provider_failed"]
    assert capacity._severity(fault.LOOK_CODE) == "page" and "draft" in stall.describe(alerts[0])
    assert fault.check_preparation_faults(tmp_path, now=NOW + fault.ANNOUNCE_SECONDS, remembered=memory)[1] == []
