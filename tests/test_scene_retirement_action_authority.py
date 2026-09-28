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
