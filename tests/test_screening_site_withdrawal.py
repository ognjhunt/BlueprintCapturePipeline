"""G52: a screening campaign never reserves for a withdrawn site, and releases what it held unlaunched."""
from __future__ import annotations

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import screening_site_withdrawal as withdrawal
from tests.test_beta_policy_screening_coordinator import Backend, plan, publication


def _withdraw(root, site="site-job-1", bucket=None):
    site_dir = (root / bucket if bucket else root) / "scenes" / site / "website_withdrawal"
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "tombstone.json").write_text("{}")


def _roots(monkeypatch, tmp_path):
    roots = tmp_path / "handoffs"
    roots.mkdir()
    monkeypatch.setenv(withdrawal.SITE_ROOTS_ENV, str(roots))
    return roots


def test_withdrawn_site_reserves_nothing_and_never_prepares(tmp_path, monkeypatch):
    roots = _roots(monkeypatch, tmp_path)
    _withdraw(roots, bucket="bucket-a")
    queue = tmp_path / "queue"
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=queue)
    backend = Backend()
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=queue, executor=backend)
    value = publication(queue)
    assert backend.preparations == 0 and backend.launches == 0
    assert value["budget"]["committed_upper_bound_usd"] == "0"
    assert {row["status"] for row in value["candidates"]} == {"blocked"}
    assert all(row["blockers"] == [withdrawal.WITHDRAWN_BLOCKER] for row in value["candidates"])


def test_withdrawal_after_reservation_releases_the_unlaunched_hold(tmp_path, monkeypatch):
    roots = _roots(monkeypatch, tmp_path)
    queue = tmp_path / "queue"
    coordinator.enqueue_screening_campaign(plan(), queue_root=queue)

    class Busy(Backend):
        def preflight_candidate_capacity(self, **kwargs):
            return {"capacity_available": False, "blockers": ["vast_paid_launch_lock_busy"]}

    backend = Busy()
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=queue, executor=backend)
    assert publication(queue)["budget"]["reserved_usd"] == "4"
    _withdraw(roots)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=queue, executor=backend)
    value = publication(queue)
    assert value["budget"]["reserved_usd"] == "0" and value["budget"]["committed_upper_bound_usd"] == "0"
    assert value["candidates"][0]["status"] == "blocked"
    assert value["candidates"][0]["blockers"] == [withdrawal.WITHDRAWN_BLOCKER]
    assert backend.launches == 0


def test_unreadable_withdrawal_state_waits_without_reserving(tmp_path, monkeypatch):
    missing = tmp_path / "not-mounted"
    monkeypatch.setenv(withdrawal.SITE_ROOTS_ENV, str(missing))
    queue = tmp_path / "queue"
    coordinator.enqueue_screening_campaign(plan(), queue_root=queue)
    backend = Backend()
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=queue, executor=backend)
    assert publication(queue)["candidates"][0]["status"] == "pending"
    assert publication(queue)["budget"]["committed_upper_bound_usd"] == "0" and backend.preparations == 0
    missing.mkdir()  # the root comes back: the next pass reserves and launches as before
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=queue, executor=backend)
    assert backend.launches == 1


def test_state_reads_only_configured_roots(tmp_path, monkeypatch):
    monkeypatch.delenv(withdrawal.SITE_ROOTS_ENV, raising=False)
    assert withdrawal.site_withdrawal_state("site-job-1") == "not_configured"
    roots = _roots(monkeypatch, tmp_path)
    assert withdrawal.site_withdrawal_state("site-job-1") == "not_withdrawn"
    _withdraw(roots, site="site-job-2")
    assert withdrawal.site_withdrawal_state("site-job-2") == "withdrawn"
    assert withdrawal.site_withdrawal_state("site-job-1") == "not_withdrawn"
