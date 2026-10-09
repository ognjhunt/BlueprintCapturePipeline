"""Installed protected authorization is distinct from a planner's sealed lineage."""
import json
import os
from pathlib import Path

import pytest

from tests.test_scene_retirement_real_participants import access_fixture
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
import hashlib


def protected_consent(tmp_path, *, action='retire', principal='operator', plan=None, journal=None):
    value = dict(schema_version='scene_retirement_consent.v1',consent_id='1'*32,principal_id=principal,
        intent_id='scene-1',intent_raw_ref={'path':str(tmp_path/'intent.json'),'sha256':'sha256:'+'a'*64,'size_bytes':1},
        plan_raw_ref=plan,retired_journal_raw_ref=journal,policy_sha256='sha256:'+'b'*64,
        cohort_sha256='sha256:'+'c'*64,action=action,created_at=99,expires_at=200,
        members=[],private_archive_classes=[],consent_digest=None)
    value['consent_digest'] = canonical_digest(value,digest_field='consent_digest')
    path = tmp_path/'consent.json'
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path,value


@pytest.mark.parametrize('case',['unknown-principal','unknown-action','restore-with-plan','retire-with-journal','writable-consent'])
def test_actual_action_authority_refuses_forged_or_misbound_consent_before_members(tmp_path,monkeypatch,case):
    from blueprint_pipeline.task_evaluation_scene_retirement_authority import load_authority
    access,policy,member = access_fixture(tmp_path,monkeypatch)
    policy['principals'] = [dict(principal_id='operator',actions=['retire','restore'],
                                owner_intent_ids=['scene-1'],private_archive_classes=[])]
    policy['policy_digest'] = canonical_digest(policy,digest_field='policy_digest')
    policy_path = os.environ['BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE']
    with open(policy_path,'w') as stream:
        json.dump(policy,stream)
    plan={'path':str(tmp_path/'plan.json'),'sha256':'sha256:'+'d'*64,'size_bytes':1}
    action = 'restore' if case=='restore-with-plan' else 'retire'
    path,value = protected_consent(tmp_path,action=action,principal='forged' if case=='unknown-principal' else 'operator',
                                  plan=plan,journal=plan if case=='retire-with-journal' else None)
    value['policy_sha256']='sha256:'+hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
    value['cohort_sha256']=canonical_digest({'consumer_cohort':policy['consumer_cohort']})
    value['consent_digest']=canonical_digest(value,digest_field='consent_digest')
    path.write_text(json.dumps(value))
    if case == 'unknown-action':
        value['action']='delete'
        value['consent_digest']=canonical_digest(value,digest_field='consent_digest')
        path.write_text(json.dumps(value))
    if case == 'writable-consent':
        path.chmod(0o666)
    reasons={'unknown-principal':'scene_retirement_principal_untrusted',
             'unknown-action':'scene_retirement_consent_action_invalid',
             'restore-with-plan':'scene_retirement_consent_selector_invalid',
             'retire-with-journal':'scene_retirement_consent_selector_invalid',
             'writable-consent':'scene_retirement_access_unsafe'}
    with pytest.raises(ValueError,match=reasons[case]):
        load_authority(path,action=action,now=lambda:100)
    assert member.exists()


def test_protected_scope_load_keeps_exact_raw_identity_without_cleanup_grant(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_authority import load_authority,cohort_digest
    _,policy,_ = access_fixture(tmp_path,monkeypatch)
    policy['principals']=[dict(principal_id='operator',actions=['retire'],owner_intent_ids=['scene-1'],private_archive_classes=[])]
    policy['policy_digest']=canonical_digest(policy,digest_field='policy_digest')
    policy_path=Path(os.environ['BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE'])
    policy_path.write_text(json.dumps(policy))
    plan={'path':str(tmp_path/'plan.json'),'sha256':'sha256:'+'d'*64,'size_bytes':1}
    path,value=protected_consent(tmp_path,plan=plan)
    value['policy_sha256']='sha256:'+hashlib.sha256(policy_path.read_bytes()).hexdigest()
    value['cohort_sha256']=cohort_digest(policy['consumer_cohort'])
    value['consent_digest']=canonical_digest(value,digest_field='consent_digest')
    path.write_text(json.dumps(value))
    result=load_authority(path,action='retire',now=lambda:100)
    assert result['consent']==value
    assert result['consent_raw_ref']['sha256']=='sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()
    assert 'cleanup_authorized' not in result


def test_capture_owner_scope_requires_exact_user_request_pair_and_shares_owner_cap():
    from blueprint_pipeline.task_evaluation_scene_retirement_authority import _scopes

    principal = dict(principal_id='operator', actions=['retire', 'restore'],
                     owner_intent_ids=['sponsor-intent'], private_archive_classes=[],
                     capture_owner_scopes=[dict(user_id='capture-user', request_id='request-1')])
    scopes = _scopes({'principals': [principal]})
    assert scopes['operator']['owners'] == {'sponsor-intent'}
    assert scopes['operator']['captures'] == {('capture-user', 'request-1')}

    for invalid in (
        [dict(user_id='capture-user', request_id='request-1'),
         dict(user_id='capture-user', request_id='request-1')],
        [dict(user_id='capture-user', request_id='*')],
        [dict(user_id='capture-user', request_id='request-1', owner_intent_id='sponsor-intent')],
    ):
        with pytest.raises(ValueError):
            _scopes({'principals': [dict(principal, capture_owner_scopes=invalid)]})
    with pytest.raises(ValueError):
        _scopes({'principals': [dict(principal, owner_intent_ids=[f'intent-{i}' for i in range(256)])]})


def test_capture_member_requires_original_and_sponsor_scopes(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_authority import load_authority, cohort_digest
    _, policy, target = access_fixture(tmp_path, monkeypatch)
    principal = dict(principal_id='operator', actions=['retire', 'restore'],
                     owner_intent_ids=['scene-1'], private_archive_classes=[],
                     capture_owner_scopes=[dict(user_id='capture-user', request_id='request-1')])
    ref = {'path': str(tmp_path / 'proof.json'), 'sha256': 'sha256:' + 'a' * 64, 'size_bytes': 1}
    info = target.stat()
    capture = dict(canonical_path=str(target), **{'class': 'site_capture'},
                   generation_id='1' * 32, dev=info.st_dev, ino=info.st_ino,
                   mode=info.st_mode, inventory_sha256='sha256:' + 'b' * 64,
                   capture_owner_user_id='capture-user', request_id='request-1',
                   sponsoring_intent_id='scene-1', owner_observation_raw_ref=ref,
                   birth_delivery_raw_ref=ref, source_membership_raw_ref=ref,
                   association_raw_ref=ref, scene_intent_raw_ref=ref)
    policy_path = Path(os.environ['BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE'])
    plan = dict(ref, sha256='sha256:' + 'd' * 64)
    journal = dict(plan, path=str(Path(policy['journal_store']) / 'retired' / ('d' * 64 + '.json')))

    def admit(*, scopes=None, member=None, action='retire'):
        policy['principals'] = [dict(principal, **(scopes or {}))]
        policy['policy_digest'] = canonical_digest(policy, digest_field='policy_digest')
        policy_path.write_text(json.dumps(policy))
        consent_path, consent = protected_consent(tmp_path, action=action,
            plan=plan if action == 'retire' else None, journal=journal if action == 'restore' else None)
        consent['members'] = [dict(capture, **(member or {}))]
        consent['policy_sha256'] = 'sha256:' + hashlib.sha256(policy_path.read_bytes()).hexdigest()
        consent['cohort_sha256'] = cohort_digest(policy['consumer_cohort'])
        consent['consent_digest'] = canonical_digest(consent, digest_field='consent_digest')
        consent_path.write_text(json.dumps(consent))
        return load_authority(consent_path, action=action, now=lambda: 100)

    assert admit()['consent']['members'] == [capture]
    with pytest.raises(ValueError, match='scene_retirement_capture_owner_scope_denied'):
        admit(scopes={'capture_owner_scopes': []})
    with pytest.raises(ValueError, match='scene_retirement_owner_scope_denied'):
        admit(scopes={'owner_intent_ids': []})
    with pytest.raises(ValueError, match='scene_retirement_capture_association_invalid'):
        admit(member={'sponsoring_intent_id': 'other-intent'})
    for action in ('retire', 'restore'):
        with pytest.raises(ValueError, match='scene_retirement_capture_owner_scope_denied'):
            admit(member={'capture_owner_user_id': None}, action=action)
