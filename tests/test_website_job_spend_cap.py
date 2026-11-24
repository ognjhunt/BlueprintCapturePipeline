"""One cap at $40 per job (owner decision 2026-11-19): the number, and a scene intent held to it.

The job-wide sum across every provider is the WebApp's ledger (``websiteJobCeiling.ts``); these tests
cover the Pipeline's side: one committed figure, every scene reservation and retry counted against it,
nothing that raises it, and an in-flight intent read correctly across the switch.
"""
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import website_job_spend_cap as cap
from blueprint_pipeline.task_evaluation_scene_execution_budget_evidence import (
    ATTEMPT_GRANT_FIELD, DIRECTORY, SCHEMA, effective_execution_budget,
)
from tests.test_task_evaluation_scene_intake import request, stage

REPO = Path(__file__).resolve().parents[1]


def test_the_committed_figure_is_forty_dollars_and_the_webapp_mirror_matches_it_byte_for_byte():
    assert cap.per_job_cap_usd() == 40
    mirror = REPO.parent / "Blueprint-WebApp" / "contracts" / "pipeline" / "website-job-spend-cap.v1.json"
    if mirror.exists():  # a sibling checkout; the WebApp's own test pins its copy otherwise
        assert mirror.read_bytes() == cap.CONFIG_PATH.read_bytes()


@pytest.mark.parametrize("change", [
    {"per_job_cap_usd": 40}, {"per_job_cap_usd": "0"}, {"per_job_cap_usd": "forty"},
    {"per_job_cap_usd": "40.001"}, {"schema_version": "blueprint.website_job_spend_cap.v2"}, {"extra": True},
])
def test_a_cap_file_that_does_not_validate_fails_closed(tmp_path, change):
    path = tmp_path / "cap.json"
    path.write_text(json.dumps({**json.loads(cap.CONFIG_PATH.read_text()), **change}))
    with pytest.raises(ValueError, match=cap.UNAVAILABLE):
        cap.load(path)
    with pytest.raises(ValueError, match=cap.UNAVAILABLE):
        cap.load(tmp_path / "missing.json")


def _intent(root, total=65, attempts=8):
    value = request()
    value["execution"].update(max_total_spend_usd=total, max_paid_attempts=attempts, max_retries=1,
                              expires_at_epoch=10_000, allowed_providers=["vast", "openai"])
    return stage(root, value)


def _reserve(root, intent, attempt_id, cost, provider="vast", now=101):
    return intake.reserve_scene_attempt(queue_root=root, intent_id=intent["intent_id"], attempt_id=attempt_id,
        source_commit="d" * 40, runtime_digest="sha256:" + "e" * 64, input_digest="sha256:" + "f" * 64,
        provider=provider, maximum_spend_usd=cost, now=now)


def test_a_job_under_the_cap_reserves_and_its_budget_never_passes_forty(tmp_path):
    intent = _intent(tmp_path)  # sealed at $65 by an older grant: the cap still bounds it
    assert intake.scene_spend_budget(json.loads((tmp_path / intent["intent_id"] / "intent.json").read_text())) == {
        "max_total_spend_usd": 40, "max_paid_attempts": 8}
    _reserve(tmp_path, intent, "a1", 18)
    _reserve(tmp_path, intent, "a2", 18, provider="openai")
    assert intake.committed_spend_usd(tmp_path / intent["intent_id"], json.loads(
        (tmp_path / intent["intent_id"] / "intent.json").read_text())) == 36


def test_a_call_that_would_cross_the_cap_is_refused_and_retries_count(tmp_path):
    intent = _intent(tmp_path)
    _reserve(tmp_path, intent, "a1", 18)
    # The retry of the same work is a new reservation: it counts beside the first.
    _reserve(tmp_path, intent, "a1-retry", 18)
    with pytest.raises(intake.SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(tmp_path, intent, "a1-retry-2", 4.01)
    # A quote above the whole cap is refused on its own.
    fresh = _intent(tmp_path / "fresh")
    with pytest.raises(intake.SceneIntakeError, match="attempt_spend_exceeds_authorized_limit"):
        _reserve(tmp_path / "fresh", fresh, "big", 40.01)
    # Exactly at the cap is admitted.
    _reserve(tmp_path, intent, "a1-retry-3", 4)


def test_a_smaller_sealed_grant_stays_the_intents_budget(tmp_path):
    intent = _intent(tmp_path, total=20)
    _reserve(tmp_path, intent, "a1", 18)
    with pytest.raises(intake.SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(tmp_path, intent, "a2", 2.01)


def _owner_extension(directory: Path, intent: dict, *, spend: float, issued: float) -> dict:
    """An owner grant exactly as the release before 2026-11-19 wrote it (``extend-scene-execution-budget``)."""
    execution = intent["request"]["execution"]
    original = {"max_total_spend_usd": execution["max_total_spend_usd"],
                "max_paid_attempts": execution["max_paid_attempts"]}
    row = intake._seal({
        "schema_version": SCHEMA, "scope": "cumulative_budget_and_attempts_only", "intent_id": intent["intent_id"],
        "intent_digest": intent["intent_digest"], "owner": intent["request"]["owner"],
        "authenticated_issuer": intent["authenticated_issuer"], "authorization_reference": "owner_approval:before",
        "original_limits": original, "prior_limits": original,
        "limits": {"max_total_spend_usd": spend, "max_paid_attempts": execution["max_paid_attempts"]},
        "unchanged_execution_bounds": {k: v for k, v in execution.items()
                                       if k not in {"max_total_spend_usd", "max_paid_attempts"}},
        "sequence": 1, "predecessor_digest": intent["intent_digest"], "issued_at_epoch": issued,
        "provider_mutation_performed": False, "historical_reservations_released": False}, "extension_digest")
    store = directory / DIRECTORY
    store.mkdir(exist_ok=True)
    (store / (row["extension_digest"].removeprefix("sha256:") + ".json")).write_text(json.dumps(row))
    return row


def test_an_in_flight_job_across_the_switch_keeps_its_history_and_its_past_spend_counts(tmp_path):
    accepted = _intent(tmp_path, total=20)
    directory = tmp_path / accepted["intent_id"]
    intent = json.loads((directory / "intent.json").read_text())
    _reserve(tmp_path, accepted, "before-1", 18)
    grant = _owner_extension(directory, intent, spend=70, issued=101)
    # An attempt the old release reserved under the owner's grant, bound to it.
    bound = intake._seal({"schema_version": intake.ATTEMPT_SCHEMA, "intent_id": intent["intent_id"],
        "intent_digest": intent["intent_digest"], "attempt_id": "before-2", "source_commit": "d" * 40,
        "runtime_digest": "sha256:" + "e" * 64, "input_digest": "sha256:" + "f" * 64, "provider": "openai",
        "maximum_spend_usd": 30, "status": "reserved", ATTEMPT_GRANT_FIELD: grant["extension_digest"],
        "reserved_at_epoch": 102}, "attempt_digest")
    (directory / "attempts" / "before-2.json").write_text(json.dumps(bound))
    # A standing extension store, as the old release left it, is never read again.
    (directory / "standing-budget-extensions").mkdir()
    # The money history stays readable: the grant chain validates and the status read reports it.
    assert effective_execution_budget(directory, intent)["max_total_spend_usd"] == 70
    status = intake.scene_intent_status(queue_root=tmp_path, intent_id=intent["intent_id"], now=103)
    assert {row["attempt_id"] for row in status["attempts"]} == {"before-1", "before-2"}
    assert status["effective_execution_budget"]["extension_count"] == 1
    # Its $48 of past holds count; the grant raises nothing, so no further paid attempt starts.
    with pytest.raises(intake.SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(tmp_path, accepted, "after-1", 0.5, now=104)
    # The idempotent re-read of an attempt reserved before the switch still validates against its grant.
    assert _reserve(tmp_path, accepted, "before-2", 30, provider="openai", now=104) == bound
    assert json.loads((directory / "attempts" / "before-2.json").read_text()) == bound
    with pytest.raises(intake.SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(tmp_path, accepted, "after-2", 0.01, now=105)
    # New rows never bind a grant.
    assert not any(ATTEMPT_GRANT_FIELD in json.loads(path.read_text())
                   for path in (directory / "attempts").glob("after-*.json"))


def test_the_accounts_monthly_limit_still_applies_under_the_job_cap(monkeypatch):
    """A job far under its $40 is still refused once the account's month is spent, before the website is asked."""
    from blueprint_pipeline import paid_resource_admission as admission
    from blueprint_pipeline import website_task_context as context
    monkeypatch.setattr(admission, "_account_monthly_limit_blockers",
                        lambda **_: ["account_monthly_spend_limit_reached"])
    monkeypatch.setattr(admission, "_monthly_budget_blockers", lambda _resource_class: [])
    monkeypatch.setattr(context, "preparation_webapp_request",
                        lambda **_: pytest.fail("the website was asked for a reservation past the account's limit"))
    task_context = {"context_digest": "sha256:" + "1" * 64, "request_id": "req-1", "scene_id": "site-req-1",
                    "capture_id": "walkthrough-req-1"}
    with pytest.raises(admission.PaidResourceAdmissionBlocked, match="account_monthly_spend_limit_reached"):
        context.reserve_website_preparation_spend(task_context=task_context, binding_digest="sha256:" + "2" * 64,
            maximum_cost_usd=0.25, request_count=1, resource_class="evaluator_api", provider="google")
