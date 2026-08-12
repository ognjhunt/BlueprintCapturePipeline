"""Operator-initiated, evidence-gated re-drive of a blocked terminal release.

On 2026-08-12 a release blocked because the canonical allocator could not be
imported: it exited 1 before any provider call, wrote no receipt, and the
request landed in ``blocked/``. The queue processor only scans ``pending/`` and
re-staging the same request is idempotent into the existing blocked file, so
the stopped Vast record it named became permanently unreleasable -- and that
record is the sole remaining blocker on provider-zero.

Two properties make a retry safe here, and both are asserted below:

* the outcome receipt is retained for every terminal outcome, so a re-drive is
  justified by evidence rather than by assumption; and
* re-arming happens only on an explicit operator re-submit, and only when that
  retained receipt proves the provider was never contacted. Nothing re-arms on
  a timer, so the zero-automatic-retry contract still holds.
"""

import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_terminal_resource_release import (
    process_terminal_resource_release_queue,
)
from blueprint_pipeline.task_evaluation_terminal_resource_release_contract import (
    TerminalResourceReleaseError,
    canonical_digest,
    release_redrive_admission,
    stage_terminal_resource_release_request,
)

RELEASE_ID = "adp009d-launch-vast-47508030"


def _request() -> dict:
    request = {
        "schema_version": "task_evaluation_terminal_resource_release_request.v1",
        "release_id": RELEASE_ID,
        "launch_id": "adp009d-launch",
        "run_id": "adp009d-run",
        "request_digest": "sha256:" + "f7" * 32,
        "provider": "vast",
        "instance_id": "47508030",
        "expected_label": "blueprint-adp009d-1786496624",
        "claim_ceiling": "operational_resource_release_only",
        "provider_mutation_performed_inside_web_request": False,
        "automatic_retry_performed": False,
        "authorization": {
            "action": "terminal_provider_record_release",
            "approved": True,
            "max_additional_spend_usd": 0,
            "retry_cap": 0,
            "actor": {"id": "ops-operator", "role": "ops"},
            "authorized_at": "2026-08-12T14:20:42.620Z",
        },
        "control_plane_terminal_blocker": {
            "schema_version": "task_evaluation_launch_control_plane_blocker.v1",
            "status": "blocked",
            "code": "control_plane_terminal_receipt_missing_after_spend_authority_expiry",
            "launch_id": "adp009d-launch",
            "run_id": "adp009d-run",
            "request_digest": "sha256:" + "f7" * 32,
            "pipeline_terminal_receipt_observed": False,
            "provider_mutation_performed_by_webapp": False,
            "paid_execution_retry_performed": False,
            "execution_result": "not_observed",
            "scripted_positive_controls_result": "not_observed",
            "learned_policy_result": "not_observed",
        },
    }
    request["terminal_resource_release_digest"] = canonical_digest(
        request, digest_field="terminal_resource_release_digest"
    )
    return request


def _receipt(**overrides) -> dict:
    receipt = {
        "schema_version": "task_evaluation_terminal_resource_release_receipt.v1",
        "status": "blocked",
        "release_id": RELEASE_ID,
        "provider_mutation_attempted": None,
        "provider_mutations_performed": None,
        "automatic_retry_performed": False,
        "blockers": ["terminal_resource_release_allocator_receipt_missing"],
        "allocator_exit_code": 1,
        "raw_secret_values_recorded": False,
    }
    receipt.update(overrides)
    return receipt


# --- the retained-evidence half -------------------------------------------

def test_retains_a_receipt_for_a_blocked_outcome(tmp_path: Path) -> None:
    """Without this the blocked release keeps no evidence of why it failed."""
    queue, state = tmp_path / "queue", tmp_path / "state"
    stage_terminal_resource_release_request(value=_request(), queue_root=queue)

    process_terminal_resource_release_queue(
        queue_root=queue, state_root=state,
        dispatcher=lambda **_: _receipt(),
    )

    retained = state / RELEASE_ID / "terminal_resource_release_receipt.json"
    assert retained.is_file(), "a blocked release must retain its outcome receipt"
    assert json.loads(retained.read_text())["blockers"] == [
        "terminal_resource_release_allocator_receipt_missing"
    ]


def test_retains_a_receipt_for_a_completed_outcome(tmp_path: Path) -> None:
    queue, state = tmp_path / "queue", tmp_path / "state"
    stage_terminal_resource_release_request(value=_request(), queue_root=queue)

    process_terminal_resource_release_queue(
        queue_root=queue, state_root=state,
        dispatcher=lambda **_: _receipt(
            status="completed", provider_mutation_attempted=True,
            provider_mutations_performed=["destroy:47508030"],
        ),
    )

    retained = state / RELEASE_ID / "terminal_resource_release_receipt.json"
    assert json.loads(retained.read_text())["status"] == "completed"


# --- the admission half ----------------------------------------------------

def test_admits_a_redrive_when_the_provider_was_never_contacted() -> None:
    assert release_redrive_admission(_receipt())["admitted"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider_mutation_attempted": True},
        {"provider_mutations_performed": ["destroy:47508030"]},
        {"status": "completed"},
    ],
    ids=["mutation_attempted", "mutation_performed", "already_completed"],
)
def test_refuses_a_redrive_once_the_provider_may_have_been_touched(overrides) -> None:
    admission = release_redrive_admission(_receipt(**overrides))
    assert admission["admitted"] is False
    assert admission["blockers"]


def test_refuses_a_redrive_without_a_retained_receipt() -> None:
    """Absent evidence is not evidence of absence."""
    assert release_redrive_admission(None)["admitted"] is False


# --- the seam --------------------------------------------------------------

def test_operator_resubmit_rearms_a_provably_untouched_release(tmp_path: Path) -> None:
    queue, state = tmp_path / "queue", tmp_path / "state"
    stage_terminal_resource_release_request(value=_request(), queue_root=queue)
    process_terminal_resource_release_queue(
        queue_root=queue, state_root=state, dispatcher=lambda **_: _receipt(),
    )
    assert list((queue / "blocked").glob("*.json"))

    receipt = stage_terminal_resource_release_request(
        value=_request(), queue_root=queue, state_root=state,
    )

    assert receipt["status"] == "requeued"
    assert receipt["provider_mutation_performed"] is False
    assert list((queue / "pending").glob("*.json")), "re-armed request must be runnable"
    assert not list((queue / "blocked").glob("*.json"))


def test_operator_resubmit_refuses_to_rearm_after_a_provider_mutation(tmp_path: Path) -> None:
    queue, state = tmp_path / "queue", tmp_path / "state"
    stage_terminal_resource_release_request(value=_request(), queue_root=queue)
    process_terminal_resource_release_queue(
        queue_root=queue, state_root=state,
        dispatcher=lambda **_: _receipt(provider_mutation_attempted=True),
    )

    with pytest.raises(TerminalResourceReleaseError, match="redrive_refused"):
        stage_terminal_resource_release_request(
            value=_request(), queue_root=queue, state_root=state,
        )
    assert list((queue / "blocked").glob("*.json")), "the terminal record must survive"
    assert not list((queue / "pending").glob("*.json"))


def test_processing_the_queue_never_rearms_on_its_own(tmp_path: Path) -> None:
    """Retry stays operator-initiated; nothing re-drives on a timer."""
    queue, state = tmp_path / "queue", tmp_path / "state"
    stage_terminal_resource_release_request(value=_request(), queue_root=queue)
    process_terminal_resource_release_queue(
        queue_root=queue, state_root=state, dispatcher=lambda **_: _receipt(),
    )

    second = process_terminal_resource_release_queue(
        queue_root=queue, state_root=state,
        dispatcher=lambda **_: pytest.fail("must not dispatch a blocked release"),
    )

    assert second["processed_count"] == 0
    assert second["automatic_retry_performed"] is False


def test_a_redrive_is_not_recorded_as_an_automatic_retry(tmp_path: Path) -> None:
    queue, state = tmp_path / "queue", tmp_path / "state"
    stage_terminal_resource_release_request(value=_request(), queue_root=queue)
    process_terminal_resource_release_queue(
        queue_root=queue, state_root=state, dispatcher=lambda **_: _receipt(),
    )
    stage_terminal_resource_release_request(
        value=_request(), queue_root=queue, state_root=state,
    )

    run = process_terminal_resource_release_queue(
        queue_root=queue, state_root=state,
        dispatcher=lambda **_: _receipt(
            status="completed", provider_mutation_attempted=True,
            provider_mutations_performed=["destroy:47508030"],
        ),
    )

    assert run["processed_count"] == 1
    assert run["automatic_retry_performed"] is False
    assert run["status"] == "completed"


def test_staging_without_a_state_root_keeps_the_previous_idempotent_behaviour(
    tmp_path: Path,
) -> None:
    """Callers that cannot prove anything must not silently re-arm paid work."""
    queue, state = tmp_path / "queue", tmp_path / "state"
    stage_terminal_resource_release_request(value=_request(), queue_root=queue)
    process_terminal_resource_release_queue(
        queue_root=queue, state_root=state, dispatcher=lambda **_: _receipt(),
    )

    receipt = stage_terminal_resource_release_request(value=_request(), queue_root=queue)

    assert receipt["status"] == "blocked"
    assert receipt["already_exists"] is True
    assert not list((queue / "pending").glob("*.json"))
