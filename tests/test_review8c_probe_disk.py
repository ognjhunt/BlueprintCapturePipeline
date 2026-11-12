"""Reviewer probe (finding 6): the 6 h disk wait clock, its alert and the money release."""
from datetime import datetime, timedelta, timezone

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_screening_disk_admission as disk
from tests.test_beta_policy_screening_coordinator import plan as queue_plan
from tests.test_beta_policy_screening_known_good_control import PerRow, run, summaries  # noqa: F401

DISK = {"capacity_available": False, "blockers": [disk.WORK_DISK_WAIT_BLOCKER]}
BUSY = {"capacity_available": False, "blockers": ["vast_paid_launch_lock_busy"]}


class Scripted(PerRow):
    def __init__(self):
        super().__init__({})
        self.next = DISK

    def preflight_candidate_capacity(self, **kwargs):
        return dict(self.next)


def _edit(root, **fields):
    with coordinator._lock(root):
        ledger = coordinator._ledger(root)
        attempt = ledger["campaigns"]["campaign-1"]["candidates"]["candidate-000"]["attempt"]
        attempt["retry_not_before"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        attempt.update(fields)
        coordinator._save(root, ledger)


def _row(root):
    return coordinator._ledger(root)["campaigns"]["campaign-1"]["candidates"]["candidate-000"]


def test_a_busy_lock_poll_restarts_the_six_hour_clock_and_the_alert(tmp_path, summaries):  # noqa: F811
    coordinator.enqueue_screening_campaign(queue_plan(count=1), queue_root=tmp_path)
    backend, posts = Scripted(), []
    alert = lambda: disk.alert_disk_waits(tmp_path, webhook_url="https://alerts.invalid/x",
                                          poster=lambda url, payload: posts.append(payload))
    run(tmp_path, backend); alert()
    first = _row(tmp_path)["attempt"].get(disk.WAIT_SINCE_FIELD)
    assert first and len(posts) == 1
    _edit(tmp_path, **{disk.WAIT_SINCE_FIELD: (datetime.now(timezone.utc) - timedelta(hours=5, minutes=59)).isoformat()})
    backend.next = BUSY
    _edit(tmp_path); run(tmp_path, backend); alert()
    backend.next = DISK
    _edit(tmp_path); run(tmp_path, backend); alert()
    row = _row(tmp_path)
    print("since after busy poll:", row["attempt"].get(disk.WAIT_SINCE_FIELD), "status", row["status"], "alerts", len(posts))
    assert row["attempt"].get(disk.WAIT_SINCE_FIELD) < (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(), \
        "a lock-busy poll reset the 6 h clock"
    assert len(posts) == 1, "a second wait alert for the same wait"


def test_past_six_hours_held_released_once_and_final(tmp_path, summaries):  # noqa: F811
    coordinator.enqueue_screening_campaign(queue_plan(count=1), queue_root=tmp_path)
    backend, posts = Scripted(), []
    run(tmp_path, backend)
    _edit(tmp_path, **{disk.WAIT_SINCE_FIELD: (datetime.now(timezone.utc) - timedelta(hours=6, seconds=1)).isoformat()})
    run(tmp_path, backend)
    row = _row(tmp_path)
    budget = coordinator._budget(coordinator._ledger(tmp_path), "site-job-1")
    print(row["status"], row["blockers"], row["attempt"]["reservation_status"], budget)
    assert row["status"] == "blocked" and row["blockers"] == [disk.CANNOT_LAND_BLOCKER]
    assert float(budget["committed_upper_bound_usd"]) == 0
    for _ in range(3):
        run(tmp_path, backend)
        disk.alert_disk_waits(tmp_path, webhook_url="https://alerts.invalid/x", poster=lambda u, p: posts.append(p))
    print("cannot-land alerts", [a["code"] for p in posts for a in p["alerts"]], "launches", backend.launches)
    assert sum(a["code"] == disk.CANNOT_LAND_ALERT for p in posts for a in p["alerts"]) == 1 and backend.launches == 0
    assert _row(tmp_path)["status"] == "blocked"
