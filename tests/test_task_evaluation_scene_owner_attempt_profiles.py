"""Non-policy profile construction preserves owner scope and live consent."""
import pytest

from blueprint_pipeline import task_evaluation_scene_owner_attempt_profiles as profiles
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.task_evaluation_scene_execution_authority import bind_scene_attempt
from tests.test_task_evaluation_scene_intake import stage, attempt, request


@pytest.fixture
def bound(tmp_path, monkeypatch):
    intent = stage(tmp_path)
    reserved = attempt(tmp_path, intent)
    monkeypatch.setenv(intake.ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")
    record = profiles.make_owner_attempt_record(owner_fields=bind_scene_attempt(reserved),
        phase="controls", team_namespace="org1", scene_id="scene1", task_id="pick-book",
        runtime_source_bundle_digest=reserved["runtime_digest"])
    return tmp_path, intent, record


def validate(record, **kwargs):
    args = dict(phase="controls", source_commit="d" * 40, scene_id="scene1",
        task_id="pick-book", maximum_spend_usd=2, team_namespace="org1", now=102,
        runtime_source_bundle_digest="sha256:" + "e" * 64)
    return profiles.validate_owner_attempt_record(record, **{**args, **kwargs})


def test_owner_record_admits_exact_scope(bound):
    _, _, record = bound
    assert validate(record)["scene_attempt_id"] == "a1"


@pytest.mark.parametrize("overrides", [
    {"phase": "construction"}, {"scene_id": "other"}, {"task_id": "other"},
    {"team_namespace": "other"}, {"runtime_source_bundle_digest": "sha256:" + "a" * 64},
    {"source_commit": "a" * 40}, {"maximum_spend_usd": 2.01}, {"provider": "runpod"},
])
def test_owner_record_rejects_scope_or_budget_substitution(bound, overrides):
    with pytest.raises(ValueError):
        validate(bound[2], **overrides)


def test_owner_record_reopens_revocation(bound):
    root, intent, record = bound
    intake.revoke_scene_intent(queue_root=root, intent_id=intent["intent_id"],
        intent_digest=intent["intent_digest"], owner=request()["owner"], now=103)
    with pytest.raises(ValueError, match="revoked"):
        validate(record, now=104)


def test_owner_reference_cannot_fall_back_to_legacy_profile():
    with pytest.raises(ValueError, match="owner_attempt_path_missing"):
        profiles.profile_owner_fields(path=None, authority={"reference": "scene-intent:abc"},
            phase="controls", source_commit="d" * 40, scene_id="scene1", task_id="pick-book",
            maximum_spend_usd=2)


def test_destination_probe_keeps_exact_owner_scope_through_activation(bound,monkeypatch):
    root,owner,original=bound
    record=profiles.make_owner_attempt_record(owner_fields={k:original[k] for k in profiles.OWNER_FIELDS},
        phase='destination',team_namespace='org1',scene_id='scene1',task_id='pick-book',
        runtime_source_bundle_digest='sha256:'+'e'*64)
    validate_record=profiles.validate_owner_attempt_record
    monkeypatch.setattr(profiles,'validate_owner_attempt_record',lambda value,**kwargs:validate_record(value,**{**kwargs,'now':102}))
    path=profiles.retain_native_owner_attempt(activation_request={'lane':'native_task_arena_destination_qualification',
        'authorization':{'scene_owner_attempt':record},'expected_production_commit':'d'*40,'team_namespace':'org1'},
        preparation_request={'scene':{'identity':{'id':'scene1'}},'task':{'identity':{'id':'pick-book'}},
            'spend':{'hard_cap_usd':2,'selected_provider':'vast'},'scene_intent_digest':owner['intent_digest'],
            'execution_adapter':{'runtime_source_bundle':{'digest':'sha256:'+'e'*64}}},output_path=root/'destination-owner.json')
    assert profiles._read(path)['phase']=='destination'
