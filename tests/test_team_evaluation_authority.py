"""ADP-009D/day-21: evaluations own spend without changing scene preparation."""
import copy
import time

import pytest

from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_scene_policy_binding as policy
from blueprint_pipeline import task_evaluation_team_run_authority as authority
from tests.test_task_evaluation_scene_intake import request


def _case(tmp_path, monkeypatch):
    now = time.time()
    source = request()
    source['execution'].update(purpose='scene_preparation', policy_candidates=[], expires_at_epoch=now+3600)
    source['task'].pop('robot_binding_id', None)
    source['consent']['accepted_at_epoch'] = now
    root = tmp_path/'intents'
    staged = intake.stage_scene_intent(value=source, queue_root=root, authenticated_client='webapp', trusted_clients={'webapp'}, now=now)
    monkeypatch.setenv(intake.ROOT_ENV, str(root))
    monkeypatch.setenv(intake.CLIENTS_ENV, 'webapp')
    monkeypatch.delenv('BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG', raising=False)
    profile = {'profile_id':'prepared-scene', 'scene_intent_digest':staged['intent_digest'],
               'task_evaluation_run':{'task_id':source['task']['task_id']}}
    profile['profile_digest'] = policy.canonical_digest(profile, digest_field='profile_digest')
    return root, source, profile, now


def _evaluation(root, source, profile, now, run_id='eval-one', **changes):
    value = copy.deepcopy(source)
    value['submission_id'] = run_id
    value['execution'] = request()['execution']
    value['execution']['expires_at_epoch'] = now+3600
    value['task']['robot_binding_id'] = 'selected-franka'
    binding = {'evaluation_run_id':run_id, 'source_launch_id':'source-launch',
               'source_profile_digest':profile['profile_digest'],
               'configured_scene_revision_digest':'sha256:'+'a'*64}
    value['task']['evaluation_source'] = binding
    value.update(changes)
    staged = intake.stage_scene_intent(value=value, queue_root=root, authenticated_client='webapp', trusted_clients={'webapp'}, now=now)
    return {**binding, 'scene_intent_digest':staged['intent_digest']}, staged


def _resolve(profile, binding, now):
    return authority.evaluation_owner(source_profile=profile, authority=binding,
        source_launch_id='source-launch', configured_scene_revision_digest='sha256:'+'a'*64,
        evaluation_run_id=binding['evaluation_run_id'], now=now)


def test_two_evaluations_have_independent_authority_and_leave_preparation_unchanged(tmp_path, monkeypatch):
    root, source, profile, now = _case(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in root.glob('scene-*/intent.json')}
    owners=[]
    for run_id in ('eval-one','eval-two'):
        binding, staged = _evaluation(root, source, profile, now, run_id)
        owner = _resolve(profile, binding, now)
        assert owner['intent_id'] == staged['intent_id']
        assert owner['request']['task']['robot_binding_id'] == 'selected-franka'
        owners.append(owner['intent_id'])
    assert len(set(owners)) == 2
    assert all(p.read_bytes()==raw for p,raw in before.items())
    assert not list(root.glob('scene-*/attempts/*.json'))


@pytest.mark.parametrize('field', ['source_launch_id','source_profile_digest','configured_scene_revision_digest','evaluation_run_id'])
def test_changed_source_or_run_refuses(tmp_path, monkeypatch, field):
    root, source, profile, now = _case(tmp_path, monkeypatch)
    binding, _ = _evaluation(root, source, profile, now)
    binding[field] = ('sha256:'+'b'*64) if field.endswith('digest') else 'different'
    with pytest.raises(ValueError, match='team_evaluation_'):
        _resolve(profile, binding, now)


def test_cannot_use_preparation_authority_for_robot_evaluation(tmp_path, monkeypatch):
    root, source, profile, now = _case(tmp_path, monkeypatch)
    binding, _ = _evaluation(root, source, profile, now)
    binding['scene_intent_digest'] = profile['scene_intent_digest']
    with pytest.raises(ValueError, match='team_evaluation_'):
        _resolve(profile, binding, now)


def test_revoked_evaluation_is_refused(tmp_path, monkeypatch):
    root, source, profile, now = _case(tmp_path, monkeypatch)
    binding, staged = _evaluation(root, source, profile, now)
    (root/staged['intent_id']/'revoked.json').write_text('{}')
    with pytest.raises(ValueError, match='owner_revoked'):
        _resolve(profile, binding, now)


def test_private_source_cannot_be_taken_by_another_owner(tmp_path, monkeypatch):
    root, source, profile, now = _case(tmp_path, monkeypatch)
    owner = {**source['owner'], 'user_id':'another-user'}
    consent = {**source['consent'], 'accepted_by':'another-user'}
    binding, _ = _evaluation(root, source, profile, now, owner=owner, consent=consent)
    with pytest.raises(ValueError, match='team_evaluation_source_access'):
        _resolve(profile, binding, now)
