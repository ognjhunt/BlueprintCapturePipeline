"""An exact private owner directive selects diagnostics; strict defaults stay intact."""
import copy
import json
import time

import pytest

from blueprint_pipeline import task_evaluation_scene_control_omission as omission
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from tests.test_task_evaluation_scene_intake import request


def scoped(tmp_path, monkeypatch, *, launch_id='source-launch', task_id=None):
    now=time.time()
    value=request()
    if task_id is not None:
        value['task']['task_id']=task_id
    value['execution']['expires_at_epoch']=now+3600
    value['consent']['accepted_at_epoch']=now-1
    root=tmp_path/'owners'
    owner=intake.stage_scene_intent(value=value,queue_root=root,authenticated_client='webapp',trusted_clients={'webapp'},now=now)
    monkeypatch.setenv(intake.ROOT_ENV,str(root))
    monkeypatch.setenv(intake.CLIENTS_ENV,'webapp')
    directive_root=tmp_path/'directives'
    directive_root.mkdir()
    monkeypatch.setenv(omission.ROOT_ENV,str(directive_root))
    launches=tmp_path/'launch-runs'
    profile={'scene_intent_digest':owner['intent_digest'],'scene_attempt_binding':{'intent_id':owner['intent_id']},
             'task_evaluation_run':{'task_id':value['task']['task_id']}}
    profile['profile_digest']=canonical_digest(profile,digest_field='profile_digest')
    profile_path=launches/launch_id/'launch_profile.json'
    profile_path.parent.mkdir(parents=True,exist_ok=True)
    profile_path.write_text(json.dumps(profile))
    directive={'schema_version':omission.SCHEMA,'intent_id':owner['intent_id'],'intent_digest':owner['intent_digest'],
        'owner':value['owner'],'authenticated_issuer':'webapp','authorized_by':value['owner']['user_id'],
        'authorization_reference':'fixture-explicit-current-user-request','user_request':'skip robot controls for this run',
        'original_task_digest':cross_runtime_canonical_digest(value['task']),
        'policy_candidates':value['execution']['policy_candidates'],'expires_at_epoch':now+3600,
        'omitted_controls':omission.OMITTED,'run_kind':'internal_policy_canary','claim_ceiling':'diagnostic_policy_execution',
        'maximum_policy_episodes':20,'task_scoring_criteria_changed':False,'qualified_comparison_permitted':False}
    directive['directive_digest']=canonical_digest(directive,digest_field='directive_digest')
    path=directive_root/(owner['intent_id']+'.json')
    path.write_text(json.dumps(directive))
    path.chmod(0o640)
    return launches,launch_id,path,directive,root/owner['intent_id'],now


def test_directive_is_exact_and_missing_file_keeps_strict_default(tmp_path, monkeypatch):
    launches,launch,path,directive,directory,now=scoped(tmp_path,monkeypatch)
    before=(directory/'intent.json').read_bytes()
    assert omission.load_for_run(launch_state_root=launches,source_launch_id=launch,now=now)==directive
    path.unlink()
    assert omission.load_for_run(launch_state_root=launches,source_launch_id=launch,now=now) is None
    assert (directory/'intent.json').read_bytes()==before


@pytest.mark.parametrize('field,value', [('owner',{'user_id':'other'}),('intent_digest','sha256:'+'0'*64),
    ('authenticated_issuer','other'),('original_task_digest','sha256:'+'0'*64),('policy_candidates',[]),
    ('maximum_policy_episodes',21),('maximum_policy_episodes',True),('qualified_comparison_permitted',True),
    ('task_scoring_criteria_changed',True),('omitted_controls',[]),('claim_ceiling','qualified_evaluation')])
def test_resealed_expansion_or_wrong_owner_refuses(tmp_path, monkeypatch, field,value):
    launches,launch,path,directive,_,now=scoped(tmp_path,monkeypatch)
    directive[field]=value
    directive['directive_digest']=canonical_digest(directive,digest_field='directive_digest')
    path.write_text(json.dumps(directive))
    with pytest.raises(ValueError,match='directive_invalid'):
        omission.load_for_run(launch_state_root=launches,source_launch_id=launch,now=now)


@pytest.mark.parametrize('fault',['tamper','symlink','public','revoked','expired'])
def test_private_bytes_and_current_consent_remain_required(tmp_path, monkeypatch,fault):
    launches,launch,path,directive,directory,now=scoped(tmp_path,monkeypatch)
    if fault=='tamper':
        directive['authorization_reference']='changed'
        path.write_text(json.dumps(directive))
    elif fault=='symlink':
        target=path.with_name('target.json')
        path.rename(target)
        path.symlink_to(target)
    elif fault=='public':
        path.chmod(0o644)
    elif fault=='revoked':
        (directory/'revoked.json').write_text('{}')
    else:
        now+=3601
    with pytest.raises(ValueError):
        omission.load_for_run(launch_state_root=launches,source_launch_id=launch,now=now)


def test_existing_typed_derivation_preserves_scoring_and_original_bytes(tmp_path):
    from tests.test_native_task_arena_policy_canary_session import _activation
    contract=copy.deepcopy(_activation()['task_success_contract'])
    contract['criteria']['controls']={'mode':'required_per_cell','control_ids':omission.OMITTED}
    contract['contract_digest']=cross_runtime_canonical_digest(contract,digest_field='contract_digest')
    request={'schema_version':'native_task_arena_packet_request.v1','task_spec':{'task_success_contract':contract,
        'configured_success_criteria':{'per_cell_controls_required':True}}}
    request['request_digest']=canonical_digest(request,digest_field='request_digest')
    path=tmp_path/'request.json'
    path.write_text(json.dumps(request))
    before=path.read_bytes()
    derived,authority=omission.derived_contract(packet_request_path=path,
        directive={'authorized_by':'owner','authorization_reference':'user:skip-controls'})
    expected=copy.deepcopy(contract['criteria'])
    expected.pop('controls')
    assert derived['criteria']==expected and path.read_bytes()==before
    assert authority['source_task_success_contract_digest']==contract['contract_digest']
    assert authority['qualified_comparison_permitted'] is False


def test_worker_omits_controls_before_any_standalone_controls_admission(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as worker
    from tests.test_task_evaluation_configured_controls_progression_worker import _plan,_write,_sealed_progression
    plan_path=_plan(tmp_path)
    plan=json.loads(plan_path.read_bytes())
    launches,_,_,directive,_,_=scoped(tmp_path,monkeypatch,launch_id=plan['source_launch_id'])
    state=tmp_path/'progressions'/plan['source_launch_id']/f"franka-controls-{plan['expected_production_commit'][:12]}"
    _write(state/'configured_controls_progression.v1.json',_sealed_progression('episode_preparation_queued',
        episode_preparation_request={'preparation_id':'prep-1'}))
    _write(state/'construction_activation_progression.json',_sealed_progression('construction_activation_queued'))
    _write(state/'construction_launch_progression.json',_sealed_progression('construction_launch_queued',launch_id='construction-1'))
    prep=tmp_path/'preparations'
    _write(prep/'identities/prep-1.json',{'identity':'prep-1'})
    _write(prep/'results/prep-1-a.json',{'status':'prepared'})
    def forbidden(*args,**kwargs):
        pytest.fail('omitted standalone controls reached activation or launch')
    monkeypatch.setattr(worker,'stage_configured_controls_activation',forbidden)
    monkeypatch.setattr(worker,'submit_authorized_progression_launch',forbidden)
    result=worker.advance_configured_controls_plan(plan_path=plan_path,launch_state_root=launches,
        progression_root=tmp_path/'progressions',preparation_queue_root=prep,activation_queue_root=tmp_path/'activations')
    assert result['status']=='controls_omitted_for_diagnostic_policy'
    assert result['control_omission_directive_digest']==directive['directive_digest']
    assert result['controls_qualified'] is result['qualified_comparison_permitted'] is False
    assert not (state/'controls_activation_progression.json').exists()
    assert not (state/'controls_launch_progression.json').exists()
    _write(state/'controls_activation_progression.json',_sealed_progression('controls_activation_queued'))
    with pytest.raises(worker.TaskEvaluationConfiguredControlsProgressionWorkerError,match='omission_after_controls_admission'):
        worker.advance_configured_controls_plan(plan_path=plan_path,launch_state_root=launches,
            progression_root=tmp_path/'progressions',preparation_queue_root=prep,activation_queue_root=tmp_path/'activations')
