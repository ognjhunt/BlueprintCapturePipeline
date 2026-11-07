"""A website navigation intake: its intent contract, its completion without paid configuration, and no spend.

Beta journey step: a job whose robot walks to a goal in a scene kept as filmed gets its scene intent
from the website's scene intake with a navigation task (``website_navigation_intake``). Owner,
2026-11-07: it reserves $0 and grants no construction allowance. The real intake stages the intent and
the real progression completes it; the request is synthetic (its digests stand for a navigation
record's). Nothing reaches the website, a provider, money or the host.
"""
# Covers: src/blueprint_pipeline/website_navigation_intake.py
from __future__ import annotations

import copy

import pytest

from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_scene_progression as engine
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_intent_contracts import SceneIntakeError, validate_request
from blueprint_pipeline.website_navigation_intake import (
    RESULT_PHASE,
    RESULT_SCHEMA,
    navigation_preparation_result,
    refusal_code,
    validate_navigation_preparation_result,
)

NOW = 1_800_000_000.0
OWNER = {"user_id": "blueprint", "organization_id": "blueprint"}
CLIENT = "blueprint-webapp"


def _request(**changes) -> dict:
    digest = lambda name: canonical_digest({"synthetic": name})  # noqa: E731 - stands for a record's digest
    value = {
        "schema_version": "task_evaluation_scene_intake_request.v1", "submission_id": "walkthrough-req1",
        "owner": dict(OWNER),
        "source": {"kind": "gaussian_splat", "binding_id": "website-splat-" + "a" * 32,
                   "content_digest": "sha256:" + "a" * 64},
        "task": {"task_id": "website-" + "0" * 20, "strategy": "goal_region_navigation",
                 "geometry_origin": "kept_as_filmed", "navigation_task_spec_digest": digest("task"),
                 "compiled_task_spec_digest": digest("compiled"), "task_success_contract_digest": digest("success"),
                 "runtime_contract_digest": digest("runtime"), "scene_revision_digest": digest("revision")},
        "execution": {"purpose": "scene_preparation", "max_total_spend_usd": 20, "max_paid_attempts": 2,
                      "max_retries": 0, "expires_at_epoch": NOW + 3600, "allowed_providers": ["vast", "openai"],
                      "policy_candidates": [], "claim_scope": "development_only"},
        "consent": {"accepted_by": "blueprint", "accepted_at_epoch": NOW - 60,
                    "rights_reference": "sha256:" + "b" * 64, "provider_terms_reference": "sha256:" + "c" * 64,
                    "private_processing_authorized": True, "provider_training_authorized": False,
                    "task_confirmed": True, "spend_authorized": True},
    }
    for path, item in changes.items():
        section, _, key = path.partition("__")
        if key:
            value[section][key] = item
        else:
            value[section] = item
    return value


def _staged(tmp_path, request=None) -> tuple[dict, dict]:
    root = tmp_path / "intents"
    receipt = intake.stage_scene_intent(value=request or _request(), queue_root=root, authenticated_client=CLIENT,
                                        trusted_clients={CLIENT}, now=NOW)
    directory = root / receipt["intent_id"]
    intent = intake._read(directory / "intent.json", "intent_digest")
    return {"root": root, "directory": directory, "receipt": receipt}, intent


def _advance(tmp_path, staged, intent, *, now=NOW):
    config = {"intent_root": str(staged["root"]), "factory_output_root": str(tmp_path / "factory-output")}
    release = {"source_commit": "0" * 40, "runtime_digest": "sha256:" + "d" * 64}
    return engine._advance_intent(staged["directory"], intent, config, release, resolver=None, publisher=None,
                                  submitter=None, status_reader=None, activation_provisioner=None, now=now)


def test_the_intent_contract_accepts_a_navigation_task_with_no_subject_support_or_destination():
    request = _request()
    assert validate_request(request, now=NOW) == request


@pytest.mark.parametrize("changes,code", [
    ({"task__subject": {"geometry_origin": "kept_as_filmed"}}, "scene_intake_task_navigation_invalid"),
    ({"task__destination": {"relation": "on"}}, "scene_intake_task_navigation_invalid"),
    ({"task__geometry_origin": "removed_before_reconstruction"}, "scene_intake_task_navigation_invalid"),
    ({"task__runtime_contract_digest": "not-a-digest"}, "scene_intake_task_navigation_invalid"),
    ({"task__scene_revision_digest": None}, "scene_intake_task_navigation_invalid"),
    ({"execution__purpose": None}, "scene_intake_navigation_must_be_scene_preparation"),
    ({"consent__task_confirmed": False}, "scene_intake_task_context_not_confirmed"),
])
def test_the_intent_contract_refuses_any_other_navigation_shape_by_name(changes, code):
    request = _request(**changes)
    if request["task"].get("scene_revision_digest") is None:
        del request["task"]["scene_revision_digest"]
    if request["execution"].get("purpose") is None:
        del request["execution"]["purpose"]
        request["execution"]["policy_candidates"] = [
            {"id": "pi05_droid", "artifact_digest": "sha256:" + "1" * 64},
            {"id": "groot_n17_droid", "artifact_digest": "sha256:" + "2" * 64}]
    with pytest.raises(SceneIntakeError, match=f"^{code}$"):
        validate_request(request, now=NOW)


def test_an_unconfirmed_brief_cannot_be_staged_whatever_a_preparation_grant_would_allow(tmp_path):
    """``task_confirmed`` is the website's reading of the brief; the intake never stages it as anything else."""
    with pytest.raises(SceneIntakeError, match="^scene_intake_task_context_not_confirmed$"):
        _staged(tmp_path, _request(consent__task_confirmed=False))
    assert not any((tmp_path / "intents").glob("scene-*/intent.json"))


def test_the_intent_completes_without_paid_configuration_naming_the_records_revision(tmp_path):
    staged, intent = _staged(tmp_path)
    progress = _advance(tmp_path, staged, intent)
    assert progress["status"] == "completed" and progress["phase"] == RESULT_PHASE
    reference = progress["result_reference"]
    assert reference == navigation_preparation_result(intent["request"])
    assert reference["schema_version"] == RESULT_SCHEMA
    assert reference["digest"] == intent["request"]["task"]["scene_revision_digest"]
    assert reference["paid_construction_performed"] is False and reference["policy_episode_claimed"] is False
    # Nothing was sourced, attempted, reserved or written outside the intent's own records.
    assert not (staged["directory"] / "attempts").exists()
    assert not (tmp_path / "factory-output").exists()
    status = intake.scene_intent_status(queue_root=staged["root"], intent_id=intent["intent_id"], now=NOW)
    assert status["status"] == "completed" and status["phase"] == RESULT_PHASE and status["attempts"] == []
    assert status["result_reference"] == reference and "outcome" not in status
    # A completed intent stays completed.
    assert _advance(tmp_path, staged, intent, now=NOW + 60) == progress


def test_a_navigation_intent_reserves_no_attempt(tmp_path):
    staged, intent = _staged(tmp_path)
    with pytest.raises(SceneIntakeError, match="^scene_intake_navigation_intent_reserves_nothing$"):
        intake.reserve_scene_attempt(queue_root=staged["root"], intent_id=intent["intent_id"], attempt_id="attempt-1",
                                     source_commit="0" * 40, runtime_digest="sha256:" + "d" * 64,
                                     input_digest="sha256:" + "e" * 64, provider="vast", maximum_spend_usd=1.0,
                                     now=NOW)
    assert not (staged["directory"] / "attempts").exists()
    assert intake.committed_spend_usd(staged["directory"], intent) == 0


def test_a_revoked_navigation_intent_does_not_complete(tmp_path):
    staged, intent = _staged(tmp_path)
    intake.revoke_scene_intent(queue_root=staged["root"], intent_id=intent["intent_id"],
                               intent_digest=intent["intent_digest"], owner=intent["request"]["owner"], now=NOW)
    progress = _advance(tmp_path, staged, intent)
    assert progress["status"] == "blocked" and progress["blockers"] == ["scene_intake_authority_revoked"]


@pytest.mark.parametrize("corruption", ["digest", "claim", "phase", "extra"])
def test_the_status_refuses_a_result_reference_that_does_not_match_its_request(tmp_path, corruption):
    from blueprint_pipeline.task_evaluation_scene_progression_state import advance

    staged, intent = _staged(tmp_path)
    reference = copy.deepcopy(navigation_preparation_result(intent["request"]))
    phase = RESULT_PHASE
    if corruption == "digest":
        reference["digest"] = "sha256:" + "9" * 64
    elif corruption == "claim":
        reference["paid_construction_performed"] = True
    elif corruption == "phase":
        phase = "scene_prepared"
    else:
        reference["path"] = "/elsewhere"
    advance(staged["directory"], intent, None, status="completed", phase=phase, state={},
            result_reference=reference, now=NOW)
    with pytest.raises(SceneIntakeError, match="^scene_intake_result_reference_invalid$"):
        intake.scene_intent_status(queue_root=staged["root"], intent_id=intent["intent_id"], now=NOW)
    with pytest.raises(ValueError, match="website_navigation_preparation_result_reference_invalid"):
        validate_navigation_preparation_result(reference, request=intent["request"], status="completed", phase=phase)


def test_only_the_websites_typed_prepared_scene_refusal_is_a_refusal():
    assert refusal_code(ValueError("website_control_prepared-scene_http_409:website_task_context_not_confirmed")) \
        == "website_task_context_not_confirmed"
    for other in ("website_control_prepared-scene_http_503:website_scene_sponsorship_unavailable",
                  "website_control_scene-sponsorship_http_409:consent_expired", "website_scene_outbox_receipt_invalid"):
        assert refusal_code(ValueError(other)) is None
