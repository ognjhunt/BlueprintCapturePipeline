from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.task_evaluation_run_state import (
    RUN_STATES,
    TaskEvaluationRunStateError,
    TaskEvaluationRunStateStore,
)


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64


def _binding() -> dict:
    return {
        "intake_digest": DIGEST_A,
        "testbed_digest": DIGEST_B,
        "request_digest": DIGEST_C,
    }


def test_run_state_is_append_only_idempotent_and_projection_repairable(tmp_path) -> None:
    store = TaskEvaluationRunStateStore(tmp_path)
    created = store.transition(
        run_id="run-1",
        from_state=None,
        to_state="testbed_ready",
        idempotency_key="run-1-created",
        actor={"role": "pipeline", "identity": "pipeline:testbed-compiler"},
        binding=_binding(),
    )
    assert created["already_exists"] is False
    planning = store.transition(
        run_id="run-1",
        from_state="testbed_ready",
        to_state="planning",
        idempotency_key="run-1-planning",
        actor={"role": "pipeline", "identity": "pipeline:router"},
        binding=_binding(),
        artifacts={"plan_digest": DIGEST_A},
    )
    replay = store.transition(
        run_id="run-1",
        from_state="testbed_ready",
        to_state="planning",
        idempotency_key="run-1-planning",
        actor={"role": "pipeline", "identity": "pipeline:router"},
        binding=_binding(),
        artifacts={"plan_digest": DIGEST_A},
    )
    assert replay["already_exists"] is True
    assert replay["event_digest"] == planning["event_digest"]
    projection = tmp_path / "runs" / "run-1" / "state.json"
    projection.unlink()
    assert store.inspect("run-1")["state"] == "planning"
    assert len(list((tmp_path / "runs" / "run-1" / "events").glob("*.json"))) == 2


def test_run_state_forbids_skips_stale_writes_and_idempotency_reuse(tmp_path) -> None:
    store = TaskEvaluationRunStateStore(tmp_path)
    store.transition(
        run_id="run-1",
        from_state=None,
        to_state="testbed_ready",
        idempotency_key="create-run-1",
        actor={"role": "pipeline"},
        binding=_binding(),
    )
    with pytest.raises(TaskEvaluationRunStateError, match="transition_forbidden"):
        store.transition(
            run_id="run-1",
            from_state="testbed_ready",
            to_state="executing",
            idempotency_key="skip-authorization",
            actor={"role": "pipeline"},
            binding=_binding(),
        )
    with pytest.raises(TaskEvaluationRunStateError, match="stale_transition"):
        store.transition(
            run_id="run-1",
            from_state=None,
            to_state="testbed_ready",
            idempotency_key="second-create",
            actor={"role": "pipeline"},
            binding=_binding(),
        )
    changed = copy.deepcopy(_binding())
    changed["testbed_digest"] = DIGEST_C
    with pytest.raises(TaskEvaluationRunStateError, match="idempotency_conflict"):
        store.transition(
            run_id="run-1",
            from_state=None,
            to_state="testbed_ready",
            idempotency_key="create-run-1",
            actor={"role": "pipeline"},
            binding=changed,
        )


def test_run_state_rejects_secrets_and_preserves_frozen_verdict(tmp_path) -> None:
    store = TaskEvaluationRunStateStore(tmp_path)
    with pytest.raises(TaskEvaluationRunStateError, match="secret_value_forbidden"):
        store.transition(
            run_id="run-secret",
            from_state=None,
            to_state="testbed_ready",
            idempotency_key="run-secret-create",
            actor={"role": "pipeline", "access_token": "must-not-store"},
            binding=_binding(),
        )
    event = store.transition(
        run_id="run-safe",
        from_state=None,
        to_state="testbed_ready",
        idempotency_key="run-safe-create",
        actor={"role": "pipeline"},
        binding=_binding(),
    )
    assert event["proof_boundary"]["comparative_policy_ranking_verdict"] == (
        "thesis_not_supported"
    )
    assert "authorization_required" in RUN_STATES
    assert "partially_decided" in RUN_STATES
    assert "abstained" in RUN_STATES


@pytest.mark.parametrize("publication_completed", [False, True])
def test_interrupted_event_commit_recovers_without_partial_or_duplicate_state(tmp_path, monkeypatch, publication_completed):
    from blueprint_pipeline.core import immutable_file

    original = immutable_file.os.link
    arguments = {
        "run_id": "run-restart", "from_state": None, "to_state": "testbed_ready",
        "idempotency_key": "retained-transition", "actor": {"role": "pipeline"},
        "binding": _binding(),
    }

    def interrupted(source, destination, **kwargs):
        if destination.parent.name == "events":
            if publication_completed:
                original(source, destination, **kwargs)
            raise OSError("worker interrupted during event commit")
        return original(source, destination, **kwargs)

    with monkeypatch.context() as interruption:
        interruption.setattr(immutable_file.os, "link", interrupted)
        with pytest.raises(OSError, match="worker interrupted"):
            TaskEvaluationRunStateStore(tmp_path).transition(**arguments)

    restarted = TaskEvaluationRunStateStore(tmp_path)
    event = restarted.transition(**arguments)
    assert event["already_exists"] is publication_completed
    assert restarted.inspect("run-restart")["state"] == "testbed_ready"
    assert len(list((tmp_path / "runs" / "run-restart" / "events").iterdir())) == 1
    assert restarted.transition(**arguments)["event_digest"] == event["event_digest"]
