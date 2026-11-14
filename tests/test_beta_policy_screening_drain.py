"""The early trial is drained: queued rows close, finished work settles, nothing new starts."""
from __future__ import annotations

import json

from blueprint_pipeline import beta_policy_screening_control as control
from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_worker as worker
from tests.test_beta_policy_screening_coordinator import plan
from tests.test_beta_policy_screening_settlement import Measured, WebApp, attempts, ledger


def _drained_tick(tmp_path, monkeypatch, backend, webapp, argv=()):
    seen = []
    process = coordinator.process_screening_queue

    def through_backend(**kwargs):
        seen.append(kwargs["max_launches"])
        return process(**{**kwargs, "executor": backend, "require_publication_ack": False})

    monkeypatch.setattr(coordinator, "process_screening_queue", through_backend)
    monkeypatch.setattr(worker, "publish_screening", webapp.accept)
    monkeypatch.setattr(worker, "enqueue_prepared_scenes", lambda **_: (_ for _ in ()).throw(
        AssertionError("a drained tick adopts no source")))
    assert worker.main(["--queue-root", str(tmp_path), *argv]) == 0
    return seen


def test_a_drained_tick_closes_queued_rows_settles_the_running_one_and_launches_nothing(tmp_path, monkeypatch, capsys):
    value = plan(count=3)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    webapp = WebApp()
    webapp.deliver(tmp_path)
    backend = Measured(value, running_first=True)
    # Before the drain: one attempt was launched and is still running; two rows wait in the queue.
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend,
                                        max_launches=1)
    webapp.deliver(tmp_path)
    rows = ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]
    assert [rows[key]["status"] for key in sorted(rows)] == ["running", "pending", "pending"]
    assert backend.launches == 1

    # A stale unit file still asks for a launch: the worker refuses it in code.
    seen = _drained_tick(tmp_path, monkeypatch, backend, webapp, argv=("--max-launches", "1"))
    result = json.loads(capsys.readouterr().out)
    assert seen == [0]
    assert result["mode"] == "drain" and result["queued_rows_closed"] == 2
    assert result["launch_refusal"] == {"requested_max_launches": 1, "refused": "beta_screening_trial_drained"}
    assert "sources" not in result and "holds" not in result
    assert backend.launches == 1 and backend.preparations == 1 and backend.reconciliations == 1

    rows = ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]
    # The queued rows closed on the existing terminal path, before reserving anything.
    for key in ("candidate-001", "candidate-002"):
        assert rows[key]["status"] == "unavailable" and rows[key]["blockers"] == ["candidate_disabled"]
        assert "attempt" not in rows[key]
    # The attempt in flight was reconciled to its end and settled at its measured cost.
    assert rows["candidate-000"]["status"] == "completed"
    assert attempts(tmp_path)["candidate-000"]["settled_cost_usd"] == "0.45"
    # The result owed was published, and the WebApp's transition rules accepted every change.
    published = webapp.accepted["campaign-1"]
    assert published["status"] == "completed"
    assert [row["status"] for row in published["candidates"]] == ["completed", "unavailable", "unavailable"]

    # Nothing is left queued: a later tick has nothing to close, start or reconcile.
    _drained_tick(tmp_path, monkeypatch, backend, webapp)
    result = json.loads(capsys.readouterr().out)
    assert result["queued_rows_closed"] == 0 and "launch_refusal" not in result
    assert result["execution"]["campaign_ids"] == [] and result["execution"]["launches_started"] == 0
    assert backend.launches == 1


def test_waitlisted_rows_close_too_and_a_row_with_an_attempt_is_left_to_its_end(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    root = coordinator._root(tmp_path)
    state = coordinator._ledger(root)
    rows = state["campaigns"]["campaign-1"]["candidates"]
    rows["candidate-000"].update(status="waitlisted", blockers=["site_screening_budget_insufficient"])
    rows["candidate-001"]["attempt"] = {"attempt_id": "screen-" + "0" * 32, "reservation_status": "released",
                                      "maximum_cost_usd": "4.00"}
    coordinator._save(root, state)
    assert worker.close_queued_rows(queue_root=tmp_path) == 1
    rows = ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]
    assert rows["candidate-000"]["status"] == "unavailable"
    assert rows["candidate-001"]["status"] == "pending" and rows["candidate-001"]["attempt"]["attempt_id"] == "screen-" + "0" * 32
    assert worker.close_queued_rows(queue_root=tmp_path) == 0


def test_the_sites_consent_stop_still_reaches_a_drained_trials_control(tmp_path, monkeypatch):
    # The control reads the source through the worker's name, which the drained worker still provides.
    def revoked(*args, **kwargs):
        raise ValueError("beta_screening_current_capture_consent_required")

    monkeypatch.setattr(worker, "require_current_source", revoked)
    value = {"site_id": "site-current", "source": {"capture_id": "capture"},
             "prepared_digest": "sha256:" + "a" * 64, "initial_observed_at_epoch": 1000,
             "execution_deadline_epoch": None}
    stopped = control.observe(control=value, root=tmp_path, now=1001)
    assert stopped["reason"] == "beta_screening_current_source_revoked"
    assert stopped["new_processing_authorized"] is False
