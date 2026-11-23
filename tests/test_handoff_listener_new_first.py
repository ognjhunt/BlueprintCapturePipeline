"""A listener tick starts a new delivery before it re-runs other jobs, and still re-runs them.

Each re-run pass may run a whole preparation. A fresh job's first step must not
wait behind them, and the passes must still get their turn in the same tick,
including when Pub/Sub cannot be reached.
"""

from __future__ import annotations

import pytest

from blueprint_pipeline import (
    pubsub_handoff_listener as listener,
    website_assessment_resume,
    website_held_handoff_replay,
    website_preparation_retry,
    website_preparation_status,
    website_reconstruction_resume,
    website_robot_fit_hold,
    website_task_target_choice,
)
from tests.test_pubsub_handoff_listener import PAYLOAD_BYTES, FakeSubscriber, _install_fake_pubsub, _received

PASSES = (
    (website_preparation_status, "reconcile_preparation_wakeups"),
    (website_assessment_resume, "reconcile_waiting_assessments"),
    (website_reconstruction_resume, "reconcile_waiting_reconstructions"),
    (website_task_target_choice, "reconcile_target_choices"),
    (website_robot_fit_hold, "reconcile_robot_fit_holds"),
    (website_held_handoff_replay, "reconcile_held_handoffs"),
    (website_preparation_retry, "reconcile_preparation_retries"),
)


def _record_passes(monkeypatch, events: list[str], subscriber: FakeSubscriber) -> None:
    for module, name in PASSES:
        def run(*_a, _name=name, **_k):
            # A due re-run that takes long: nothing pulled may still be waiting on it.
            events.append(f"{_name}:acked={list(subscriber.acknowledged)}")
            return {}
        monkeypatch.setattr(module, name, run)


def test_a_new_delivery_starts_before_the_re_runs_and_they_still_run_in_the_same_tick(tmp_path, monkeypatch):
    subscriber = FakeSubscriber([_received(ack_id="new-job", data=PAYLOAD_BYTES)])
    _install_fake_pubsub(monkeypatch, subscriber)
    events: list[str] = []
    _record_passes(monkeypatch, events, subscriber)
    monkeypatch.setattr(listener, "process_handoff_payload",
                        lambda *_a, **_k: events.append("new_delivery_started") or {"status": "completed"})
    assert listener.pull_and_process(subscription="projects/p/subscriptions/s", storage_root=tmp_path,
                                     provider="openai", max_messages=1) == 1
    assert events[0] == "new_delivery_started"
    # Every re-run pass still ran this tick, after the new delivery was acknowledged.
    assert events[1:] == [f"{name}:acked=['new-job']" for _, name in PASSES]


def test_the_re_runs_still_run_when_pubsub_cannot_be_reached(tmp_path, monkeypatch):
    subscriber = FakeSubscriber([])

    def unreachable(**_k):
        raise ConnectionError("pubsub unreachable")
    subscriber.pull = unreachable
    _install_fake_pubsub(monkeypatch, subscriber)
    events: list[str] = []
    _record_passes(monkeypatch, events, subscriber)
    with pytest.raises(ConnectionError):
        listener.pull_and_process(subscription="projects/p/subscriptions/s", storage_root=tmp_path,
                                  provider="openai", max_messages=1)
    assert events == [f"{name}:acked=[]" for _, name in PASSES]
