"""Actual action entrypoint refuses before archive/mutation without installed proof."""
import hashlib
import json
from pathlib import Path
import pytest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_retirement_real_participants import access_fixture
from tests.test_scene_retirement_action_authority import protected_consent


def action_fixture(tmp_path, monkeypatch):
    access, policy, member=access_fixture(tmp_path,monkeypatch)
    policy['principals']=[dict(principal_id='operator',actions=['retire'],owner_intent_ids=['scene-1'],private_archive_classes=[])]
    policy['policy_digest']=canonical_digest(policy,digest_field='policy_digest')
    policy_path=tmp_path/'policy.json'
    policy_path.write_text(json.dumps(policy))
    plan_path=tmp_path/'plan.json'
    plan_path.write_text(json.dumps({'schema_version':'task_evaluation_scene_lifecycle_plan.v1','intent_id':'scene-1',
        'action':'KEEP','cleanup_authorized':False}))
    plan_path.chmod(0o600)
    raw=plan_path.read_bytes()
    plan=dict(path=str(plan_path),sha256='sha256:'+hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
    consent_path,consent=protected_consent(tmp_path,plan=plan)
    consent['policy_sha256']='sha256:'+hashlib.sha256(policy_path.read_bytes()).hexdigest()
    consent['cohort_sha256']=canonical_digest({'consumer_cohort':[]})
    consent['consent_digest']=canonical_digest(consent,digest_field='consent_digest')
    consent_path.write_text(json.dumps(consent))
    return access,member,plan_path,consent_path


class UntouchedTransport:
    def put_archive(self,*args):
        pytest.fail('unproved action invoked transport')
    def read_archive(self,*args):
        pytest.fail('unproved action invoked readback')


def test_actual_retire_api_keeps_missing_installed_scan_contract_before_transport(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement import retire_scene
    _,member,plan,consent=action_fixture(tmp_path,monkeypatch)
    result=retire_scene(plan,consent,transport=UntouchedTransport(),now=lambda:100,monotonic=lambda:0)
    assert result['status']=='kept' and result['reason']=='scene_retirement_installed_context_unproven'
    assert member.exists() and result['mutations']==0


def test_actual_retire_api_cannot_enter_with_live_real_reader(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement import retire_scene
    access,member,plan,consent=action_fixture(tmp_path,monkeypatch)
    with access.scene_access(member):
        result=retire_scene(plan,consent,transport=UntouchedTransport(),now=lambda:100,monotonic=lambda:0)
    assert result['status']=='kept' and result['reason']=='scene_retirement_reader_active'
    assert member.exists() and result['mutations']==0


def test_actual_retire_api_refuses_changed_raw_plan_before_transport(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement import retire_scene
    _,member,plan,consent=action_fixture(tmp_path,monkeypatch)
    Path(plan).write_text('{"changed":true}')
    result=retire_scene(plan,consent,transport=UntouchedTransport(),now=lambda:100,monotonic=lambda:0)
    assert result['status']=='kept' and result['reason']=='scene_retirement_raw_reference_changed'
    assert member.exists() and result['mutations']==0
