"""Real construction closure and exact-owner omission feed the existing canary producers."""
import copy
import json

import pytest

from blueprint_pipeline import task_evaluation_scene_control_omission as omission
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.adp_task_scoring import seal_rigid_task_success_contract
from tests import test_task_evaluation_policy_canary_handoff as rehearsal
from tests.test_task_evaluation_scene_policy_binding import _owner_scene


def prepared(tmp_path,monkeypatch):
    state,directory,_=_owner_scene(tmp_path,monkeypatch,controls_terminal=False)
    (state/'controls_launch_progression.json').unlink()
    owner=intake._read(directory/'intent.json','intent_digest')
    request=owner['request']
    root=tmp_path/'directives'
    root.mkdir()
    monkeypatch.setenv(omission.ROOT_ENV,str(root))
    directive={'schema_version':omission.SCHEMA,'intent_id':owner['intent_id'],'intent_digest':owner['intent_digest'],
        'owner':request['owner'],'authenticated_issuer':owner['authenticated_issuer'],'authorized_by':request['owner']['user_id'],
        'authorization_reference':'fixture-user-explicit-omission','user_request':'skip robot controls for this run',
        'original_task_digest':cross_runtime_canonical_digest(request['task']),
        'policy_candidates':request['execution']['policy_candidates'],'expires_at_epoch':request['execution']['expires_at_epoch'],
        'omitted_controls':omission.OMITTED,'run_kind':'internal_policy_canary','claim_ceiling':'diagnostic_policy_execution',
        'maximum_policy_episodes':20,'task_scoring_criteria_changed':False,'qualified_comparison_permitted':False}
    directive['directive_digest']=canonical_digest(directive,digest_field='directive_digest')
    path=root/(owner['intent_id']+'.json')
    path.write_text(json.dumps(directive))
    path.chmod(0o640)
    packet_root=tmp_path/'compiled-episodes'/rehearsal.PREPARATION_ID/'native-task-packet'
    plan_path=packet_root/'native_task_arena_scene_plan.v1.json'
    plan=json.loads(plan_path.read_bytes())
    contract=seal_rigid_task_success_contract(task_spec=plan['task_spec'],site_id=plan['scene_id'],task_id=plan['task_id'],
        author_source='task_owner',author_id='u1',confirmation_status='confirmed',confirmed_by_team_id='org1')
    contract['criteria']['controls']={'mode':'required_per_cell','control_ids':omission.OMITTED}
    contract['contract_digest']=cross_runtime_canonical_digest(contract,digest_field='contract_digest')
    plan['task_spec']['task_success_contract']=contract
    plan['task_spec']['task_success_contract_digest']=contract['contract_digest']
    plan['plan_digest']=canonical_digest(plan,digest_field='plan_digest')
    plan_path.write_text(json.dumps(plan))
    packet_path=packet_root/'native_task_arena_packet_receipt.v1.json'
    packet=json.loads(packet_path.read_bytes())
    packet['arena_scene_plan_digest']=plan['plan_digest']
    packet_path.write_text(json.dumps(packet))
    source={'schema_version':'native_task_arena_packet_request.v1','task_spec':copy.deepcopy(plan['task_spec'])}
    source['request_digest']=canonical_digest(source,digest_field='request_digest')
    (packet_root/'native_task_arena_packet_request.v1.json').write_text(json.dumps(source))
    return state,directory,path,contract


def test_full_handoff_uses_real_construction_without_a_controls_receipt(tmp_path,monkeypatch):
    state,directory,directive_path,original=prepared(tmp_path,monkeypatch)
    before=(directory/'intent.json').read_bytes()
    webapp=rehearsal._WebApp()
    publisher=rehearsal._Publisher()
    profile_calls=[]
    result=rehearsal._advance(tmp_path,state=state,webapp=webapp,publisher=publisher,profile_calls=profile_calls)
    assert result['status']=='canary_launch_submitted'
    assert result['controls_launch_id'] is None
    assert len(webapp.calls)==1
    parameters=json.loads((state/'policy-canary-inputs/presubmission_parameters.json').read_bytes())
    typed=parameters['diagnostic_control_omission_authority']
    assert typed['source_task_success_contract_digest']==original['contract_digest']
    expected=copy.deepcopy(original['criteria'])
    expected.pop('controls')
    assert parameters['task_success_contract']['criteria']==expected
    assert typed['qualified_comparison_permitted'] is False
    native=json.loads(publisher.published[parameters['activation_lineage']['construction_result']['uri']])
    assert native['schema_version']=='native_task_arena_construction_result.v1'
    assert native['construction_gate_qualified'] is True
    assert 'controls_qualified' not in native
    assert not any(rehearsal.CONTROLS_LAUNCH_ID in uri for uri in publisher.published)
    assert not (state/'controls_launch_progression.json').exists()
    assert (directory/'intent.json').read_bytes()==before
    assert rehearsal._advance(tmp_path,state=state,webapp=webapp,publisher=publisher,profile_calls=profile_calls)['status']=='canary_launch_submitted'
    assert len(webapp.calls)==1
    directive_path.unlink()
    with pytest.raises(rehearsal.handoff.PolicyCanaryHandoffError,match='omission_authority_changed'):
        rehearsal._advance(tmp_path,state=state,webapp=webapp,publisher=publisher,profile_calls=profile_calls)
    assert len(webapp.calls)==1


@pytest.mark.parametrize('missing',['launch_receipt.json','post_teardown_provider_zero_receipt.json','webapp_sync_succeeded.json'])
def test_omission_does_not_bypass_construction_closeout(tmp_path,monkeypatch,missing):
    state,_,_,_=prepared(tmp_path,monkeypatch)
    (tmp_path/'launch-runs'/rehearsal.CONSTRUCTION_LAUNCH_ID/missing).unlink()
    webapp=rehearsal._WebApp()
    result=rehearsal._advance(tmp_path,state=state,webapp=webapp,publisher=rehearsal._Publisher(),profile_calls=[])
    assert result['status']=='awaiting_construction_terminal'
    assert webapp.calls==[]
    assert not (state/rehearsal.handoff.STATE_FILENAME).exists()
