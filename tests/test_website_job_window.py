"""A website job's grant and intent window is WebApp policy, bounded by one mirrored value.

Synthetic offline vectors only; no WebApp, host or provider is reached.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_job_window.py
#   src/blueprint_pipeline/website_task_context.py
#   src/blueprint_pipeline/website_preparation_contracts.py
#   src/blueprint_pipeline/task_evaluation_scene_intent_contracts.py

from __future__ import annotations

import copy

import pytest

from blueprint_pipeline import task_evaluation_scene_execution_window as window
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import website_task_context as context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_job_window import WEBSITE_JOB_WINDOW_MAX_SECONDS
from tests.test_website_preparation_contract_v2 import NOW, VECTOR, retained_source

DAY = 86400
WEEK = 7 * DAY
RIGHTS = {"consent_revoked": False, "derived_scene_generation_allowed": True}
DOOR_REQUEST = "20261103T120000Z-extend-scene-execution-window-0000abcd"


def test_the_mirrored_bound_is_the_proposed_seven_days():
    # The WebApp's WEBSITE_JOB_WINDOW_MAX_SECONDS (server/utils/websiteJobWindow.ts) must equal this.
    assert WEBSITE_JOB_WINDOW_MAX_SECONDS == WEEK


def _grant(seconds):
    grant = copy.deepcopy(VECTOR["sponsorship_response"])
    grant["expires_at_epoch"] = grant["consent"]["accepted_at_epoch"] + seconds
    grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    return grant


@pytest.mark.parametrize("seconds", [3600, DAY, WEEK])
def test_signed_read_and_preparation_check_accept_any_window_up_to_the_bound(monkeypatch, seconds):
    from blueprint_pipeline.website_preparation_contracts import validate_preparation_authority

    task = copy.deepcopy(VECTOR["task_context_response_preparation"])
    grant = _grant(seconds)
    late = NOW + seconds - 1
    monkeypatch.setattr(context, "website_webapp_request", lambda **_: copy.deepcopy(grant))
    assert context.load_website_scene_sponsorship(task_context=task, now=late, create=False) == grant
    assert validate_preparation_authority(task_context=task, authority=grant, now=late) == grant
    with pytest.raises(ValueError, match="website_preparation_authority_invalid"):
        validate_preparation_authority(task_context=task, authority=grant, now=NOW + seconds)


def test_a_grant_one_second_past_the_bound_is_refused_by_both_checks(monkeypatch):
    from blueprint_pipeline.website_preparation_contracts import validate_preparation_authority

    task = copy.deepcopy(VECTOR["task_context_response_preparation"])
    grant = _grant(WEEK + 1)
    monkeypatch.setattr(context, "website_webapp_request", lambda **_: copy.deepcopy(grant))
    with pytest.raises(ValueError, match="website_scene_sponsorship_binding_invalid"):
        context.load_website_scene_sponsorship(task_context=task, now=NOW, create=False)
    with pytest.raises(ValueError, match="website_preparation_authority_invalid"):
        validate_preparation_authority(task_context=task, authority=grant, now=NOW)


def _staged(tmp_path, monkeypatch, *, seconds, stage_at):
    helper, task, grant, value = retained_source(tmp_path, window_seconds=seconds)
    clock = [stage_at]
    monkeypatch.setattr(helper, "current_time", lambda: clock[0])
    monkeypatch.setattr(context, "website_webapp_request",
                        lambda **kw: copy.deepcopy(task if kw["operation"] == "task-context" else grant))
    root = tmp_path / "intents"
    staged = intake.stage_scene_intent(value=value, queue_root=root, authenticated_client="blueprint-webapp",
                                       trusted_clients={"blueprint-webapp"}, now=stage_at)
    return root, staged, grant, clock


def _reserve(root, staged, *, now, letter="c"):
    return intake.reserve_scene_attempt(
        queue_root=root, intent_id=staged["intent_id"], attempt_id="scene-configuration-" + letter * 24,
        source_commit="a" * 40, runtime_digest="sha256:" + "b" * 64, input_digest="sha256:" + "c" * 64,
        provider="vast", maximum_spend_usd=1, now=now)


def test_a_seven_day_grant_stages_and_reserves_days_after_consent(tmp_path, monkeypatch):
    # Under the former one-day bound this intent was refused when staged three days in.
    root, staged, grant, clock = _staged(tmp_path, monkeypatch, seconds=WEEK, stage_at=NOW + 3 * DAY)
    assert staged["status"] == "accepted"
    clock[0] = NOW + 6 * DAY
    reserved = _reserve(root, staged, now=NOW + 6 * DAY)
    # Only time changed: the sealed caps and the one paid attempt are the grant's.
    assert reserved["status"] == "reserved" and reserved["maximum_spend_usd"] == grant["max_total_spend_usd"]
    with pytest.raises(ValueError):
        _reserve(root, staged, now=NOW + 6 * DAY, letter="d")


def test_a_one_day_grant_issued_before_the_change_still_runs(tmp_path, monkeypatch):
    root, staged, grant, clock = _staged(tmp_path, monkeypatch, seconds=DAY, stage_at=NOW + 23 * 3600)
    assert staged["status"] == "accepted"
    assert _reserve(root, staged, now=NOW + 23 * 3600)["status"] == "reserved"


def test_a_grant_past_the_bound_never_stages(tmp_path, monkeypatch):
    with pytest.raises(intake.SceneIntakeError, match="scene_intake_preparation_authority_invalid"):
        _staged(tmp_path, monkeypatch, seconds=WEEK + 1, stage_at=NOW + DAY)
    assert not list((tmp_path / "intents").glob("scene-*/intent.json"))


def test_the_owner_extension_still_adds_up_to_seven_days_past_the_longer_window(tmp_path, monkeypatch):
    root, staged, grant, clock = _staged(tmp_path, monkeypatch, seconds=WEEK, stage_at=NOW + DAY)
    raw = grant["expires_at_epoch"]
    clock[0] = raw + 60
    with pytest.raises(intake.SceneIntakeError, match="scene_intake_preparation_authority_invalid"):
        _reserve(root, staged, now=raw + 60)
    common = dict(queue_root=root, intent_id=staged["intent_id"], intent_digest=staged["intent_digest"],
                  authorization_reference="owner approval 2026-11-03 (time only)", ack=window.ACK,
                  door_caller="cloud-scene-runs", door_request_id=DOOR_REQUEST,
                  trusted_clients={"blueprint-webapp"}, capture_rights=RIGHTS, now=raw + 30)
    # The extension's bound is relative to the issued expiry, so it is unchanged by the longer window.
    with pytest.raises(ValueError, match="expiry_beyond_original_window"):
        window.extend_scene_execution_window_by_operator(**common, expires_at_epoch=raw + WEEK + 1, apply=False)
    window.extend_scene_execution_window_by_operator(**common, expires_at_epoch=raw + 6 * DAY, apply=True)
    reserved = _reserve(root, staged, now=raw + 60)
    assert reserved["status"] == "reserved" and reserved["maximum_spend_usd"] == grant["max_total_spend_usd"]


def test_an_intent_consent_may_be_as_old_as_the_longest_grant_window():
    from tests.test_task_evaluation_scene_intake import request

    def at_age(age):
        value = request()
        now = value["consent"]["accepted_at_epoch"] + age
        value["execution"]["expires_at_epoch"] = now + 3600
        return value, now

    for age in (0, DAY + 1, 3 * DAY, WEEK):
        value, now = at_age(age)
        assert intake.validate_request(value, now=now)["consent"] == value["consent"]
    value, now = at_age(WEEK + 1)
    with pytest.raises(intake.SceneIntakeError, match="consent_actor_or_time_invalid"):
        intake.validate_request(value, now=now)
