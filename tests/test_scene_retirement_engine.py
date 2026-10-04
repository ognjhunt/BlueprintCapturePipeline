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


def test_installed_transport_binds_the_one_native_allowance_before_any_action_work(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement import retire_scene
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    _,member,plan,consent=action_fixture(tmp_path,monkeypatch)
    class BoundTransport(UntouchedTransport):
        def __init__(self):
            self.allowances=[]
        def bind_allowance(self,allowance):
            assert type(allowance) is ActionAllowance
            allowance.tick()
            self.allowances.append(allowance)
    transport=BoundTransport()
    result=retire_scene(plan,consent,transport=transport,now=lambda:100,monotonic=lambda:0)
    assert result['status']=='kept' and member.exists()
    assert len(transport.allowances)==1


def test_consent_targets_include_exact_coalesced_children_without_a_second_owner_or_clearance(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_retirement import _plan_members
    root=str(tmp_path.resolve()/'member')
    plan={'action':'KEEP','cleanup_authorized':False,'measured_members':[
        {'path':root,'status':'observed_scoped_metadata','keeps':[]},
        {'path':root+'/nested','status':'coalesced_descendant_member','attributed_root':root}]}
    consent={'members':[{'canonical_path':root,'class':'host'}],'private_archive_classes':[]}
    _plan_members(plan,consent)
    assert plan['action']=='KEEP' and plan['cleanup_authorized'] is False
    plan['measured_members'][1]['attributed_root']=str(tmp_path.resolve()/'foreign')
    with pytest.raises(ValueError):
        _plan_members(plan,consent)


def test_independent_sam_owner_consent_can_supply_retention_policy_without_erasing_other_keeps(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_retirement import _plan_members
    root=str(tmp_path.resolve()/'sam-owner-member')
    row={'path':root,'status':'observed_scoped_metadata','storage_class':'evidence',
         'keeps':['sam_evidence_retention_policy_required']}
    plan={'action':'KEEP','cleanup_authorized':False,'measured_members':[row]}
    consent={'members':[{'canonical_path':root,'class':'evidence','owner_intent_id':'original-owner'}],
             'private_archive_classes':['evidence']}
    _plan_members(plan,consent)
    assert row['keeps']==['sam_evidence_retention_policy_required'] and plan['action']=='KEEP'
    consent['private_archive_classes']=[]
    with pytest.raises(ValueError):
        _plan_members(plan,consent)
    consent['private_archive_classes']=['evidence']
    row['keeps'].append('external_hardlink_or_unobserved_alias')
    with pytest.raises(ValueError):
        _plan_members(plan,consent)


@pytest.mark.parametrize('storage_class',['host','cache'])
def test_unselected_shared_cache_outside_delete_roots_remains_in_place(tmp_path,storage_class):
    from blueprint_pipeline.task_evaluation_scene_retirement import _plan_members
    root=str(tmp_path.resolve()/'selected-scene')
    cache=str(tmp_path.resolve()/'content'/'sha256'/('a'*64))
    plan={'measured_members':[
        {'path':root,'status':'observed_scoped_metadata','keeps':[]},
        {'path':cache,'status':'observed_scoped_metadata','storage_class':storage_class,
         'kinds':['prepared_cache_object'],'keeps':['shared_content_object_not_exclusive']}]}
    consent={'members':[{'canonical_path':root,'class':'host'}],
             'private_archive_classes':[]}
    assert _plan_members(plan,consent)==[{'canonical_path':cache,'action':'KEEP',
        'observation_status':'observed_scoped_metadata',
        'reasons':['shared_content_object_not_exclusive']}]
    plan['measured_members'][1]['keeps'].append('external_hardlink_or_unobserved_alias')
    with pytest.raises(ValueError,match='scene_retirement_members_unproven'):
        _plan_members(plan,consent)
    plan['measured_members'][1].update(status='incomplete_scoped_metadata',physical_identity=None,
        keeps=['member_or_child_unavailable_or_changed','shared_content_object_not_exclusive'])
    assert _plan_members(plan,consent)==[{'canonical_path':cache,'action':'KEEP',
        'observation_status':'incomplete_scoped_metadata',
        'reasons':['member_or_child_unavailable_or_changed','shared_content_object_not_exclusive']}]
    plan['measured_members'][1]['physical_identity']=[1,2,3]
    with pytest.raises(ValueError,match='scene_retirement_members_unproven'):
        _plan_members(plan,consent)
