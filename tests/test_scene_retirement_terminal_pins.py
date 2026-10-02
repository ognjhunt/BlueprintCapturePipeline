# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_pins.py
#   src/blueprint_pipeline/task_evaluation_scene_retirement_reference_transfer.py
"""Exact native terminal pin closure; no process or whole-scene clearance."""
import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest


def fixture(tmp_path,monkeypatch):
    from tests.test_scene_retirement_terminal_obligations import fixture as terminal
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    from blueprint_pipeline.control_plane_storage_pins import write_storage_pin
    from blueprint_pipeline.control_plane_storage_pin_observation import observe_storage_pins
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    fresh,member,_=terminal(tmp_path)
    monkeypatch.setattr(access,'_INSTALLED_POLICY',tmp_path/'absent-policy')
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE',raising=False)
    pins=tmp_path/'pins'
    write_storage_pin(pins_root=pins,kind='preparation',owner_id=member.name,paths=[member],now=lambda:0)
    # The observer is real; caller-supplied status/expiry booleans are not proof.
    observed=observe_storage_pins(str(pins),observed_at_epoch=30000,monotonic=lambda:0)
    assert observed.complete and len(observed.rows)==1
    observation=json.loads(json.dumps(asdict(observed.rows[0])))
    fresh['reference_observation']['protections'].extend([
        {'kind':'positive_pin_path','path':str(member),'source':{
            key:observation[key] for key in ('row_path','raw_sha256','raw_size_bytes','row_identity')},
         'binding_status':'historical_positive_only','action':'KEEP'},
        {'kind':'pin_observation','observation':observation,'action':'KEEP'}])
    context={'pins_root':str(pins),'roots':{'preparation_input_root':str(member.parent)}}
    fresh['planner_context']=context
    fresh['finished_observation']={'status':'completed'}
    policy={'reference_context':context}
    intent=fresh['selected_intent_provenance']
    consent={'intent_id':'intent-1','intent_raw_ref':{k:intent[k] for k in ('path','sha256','size_bytes')},
             'members':[{'canonical_path':str(member),'owner_intent_id':'intent-1',
                         'owner_raw_ref':{k:intent[k] for k in ('path','sha256','size_bytes')}}],
             'terminal_pin_refs':[{'path':observation['row_path'],'sha256':observation['raw_sha256'],
                                   'size_bytes':observation['raw_size_bytes']}]}
    allowance=ActionAllowance(expires_at=40000,now=lambda:30000,monotonic=lambda:0)
    return fresh,policy,consent,allowance


def selected(fresh,policy,consent,allowance):
    from blueprint_pipeline.task_evaluation_scene_retirement_reference_transfer import validate_current_reference_transfer
    return validate_current_reference_transfer(fresh,allowance,policy=policy,consent=consent)


def test_real_terminal_preparation_pin_is_selected_before_any_ledger_mutation(tmp_path,monkeypatch):
    fresh,policy,consent,allowance=fixture(tmp_path,monkeypatch)
    path=Path(consent['terminal_pin_refs'][0]['path'])
    original=path.read_bytes()
    before=copy.deepcopy(fresh)
    result=selected(fresh,policy,consent,allowance)
    assert len(result['terminal_pin_release_rows'])==1
    row=result['terminal_pin_release_rows'][0]
    assert row['original_raw_ref']==consent['terminal_pin_refs'][0]
    assert row['original_value']==json.loads(original)
    assert path.read_bytes()==original and fresh==before
    assert result['references_clear'] is False and result['consumer_fence_checked'] is False


@pytest.mark.parametrize('authority_end', ['revoked', 'expired'])
@pytest.mark.parametrize('grace_offset', [-1, 0, 1])
def test_terminal_pin_selection_uses_actual_authority_end_grace_observation(
        tmp_path, monkeypatch, authority_end, grace_offset):
    """Compose the actual finished predicate and pin observer; no action clearance."""
    from tests.test_scene_lifecycle_finished import data, observe
    from tests.test_scene_inventory_history import seal
    from blueprint_pipeline.control_plane_storage_pin_observation import observe_storage_pins
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    fresh, policy, consent, _ = fixture(tmp_path, monkeypatch)
    args, intent, owner, rows = data()
    ended_at = 100
    if authority_end == 'revoked':
        revocation = seal({
            'schema_version': 'task_evaluation_scene_intent_revocation.v1',
            'intent_id': args['intent_id'], 'intent_digest': intent['intent_digest'],
            'owner': intent['request']['owner'], 'status': 'revoked',
            'scope': 'future_execution', 'revoked_at_epoch': ended_at,
            'provider_mutation_performed': False}, 'receipt_digest')
        rows.append({'role': 'revocations', 'path': owner + '/revoked.json',
                     'value': revocation})
    observed_at = ended_at + 7 * 86400 + grace_offset
    fresh['finished_observation'] = observe(rows, args, observed_at)
    assert fresh['finished_observation']['status'] == (
        authority_end + '_grace_elapsed' if grace_offset >= 0 else 'unknown')
    # Reobserve the exact pin at the same time; expiry never erases its protection.
    observation = json.loads(json.dumps(asdict(observe_storage_pins(
        fresh['planner_context']['pins_root'], observed_at_epoch=observed_at,
        monotonic=lambda: 0).rows[0])))
    fresh['reference_observation']['protections'][-1]['observation'] = observation
    allowance = ActionAllowance(expires_at=observed_at + 900,
                                now=lambda: observed_at, monotonic=lambda: 0)
    pin = Path(consent['terminal_pin_refs'][0]['path'])
    before = pin.read_bytes()
    if grace_offset < 0:
        with pytest.raises(ValueError, match='scene_retirement_terminal_pin_unproven'):
            selected(fresh, policy, consent, allowance)
    else:
        result = selected(fresh, policy, consent, allowance)
        assert len(result['terminal_pin_release_rows']) == 1
        assert result['terminal_pin_release_rows'][0]['original_raw_ref'] == consent['terminal_pin_refs'][0]
        assert result['references_clear'] is False
        assert result['consumer_fence_checked'] is False
    assert pin.read_bytes() == before


@pytest.mark.parametrize('status', [
    'cancelled', 'failed', 'blocked', 'needs_input', 'running', 'unknown', 'revoked', 'expired'])
def test_non_cleanup_terminal_status_cannot_select_a_terminal_pin(tmp_path, monkeypatch, status):
    fresh, policy, consent, allowance = fixture(tmp_path, monkeypatch)
    fresh['finished_observation'] = {'status': status}
    pin = Path(consent['terminal_pin_refs'][0]['path'])
    before = pin.read_bytes()
    with pytest.raises(ValueError, match='scene_retirement_terminal_pin_unproven'):
        selected(fresh, policy, consent, allowance)
    assert pin.read_bytes() == before


@pytest.mark.parametrize('change', ['none', 'sponsor', 'intent_raw', 'hybrid'])
def test_capture_generation_and_ordinary_pin_keep_distinct_owner_schemas(
        tmp_path, monkeypatch, change):
    """Real capture birth/generation plus pin-stage scope, never capture action admission."""
    from tests.test_capture_generation_birth import _fixture as capture_fixture
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member
    from blueprint_pipeline.task_evaluation_scene_retirement import _generation
    from blueprint_pipeline.task_evaluation_scene_retirement_authority import CAPTURE_MEMBER_KEYS
    fresh, policy, consent, allowance = fixture(tmp_path, monkeypatch)
    capture_root = tmp_path / 'capture-fixture'
    capture_root.mkdir()
    _, capture_policy, target, owner, selector, membership = capture_fixture(capture_root, monkeypatch)
    born = birth_capture_member(target, observation=owner, membership_selector=selector,
                                membership_raw=membership)
    member = dict(canonical_path=str(target), **{'class': 'site_capture'},
        generation_id=born['generation_id'], dev=born['dev'], ino=born['ino'], mode=born['mode'],
        inventory_sha256='sha256:' + 'b' * 64,
        capture_owner_user_id=born['capture_owner_user_id'], request_id=owner['request_id'],
        sponsoring_intent_id=consent['intent_id'], scene_intent_raw_ref=consent['intent_raw_ref'],
        owner_observation_raw_ref=born['owner_observation_raw_ref'],
        birth_delivery_raw_ref=born['birth_delivery_raw_ref'],
        source_membership_raw_ref=json.loads(Path(born['birth_delivery_raw_ref']['path']).read_bytes())[
            'source_membership_raw_ref'], association_raw_ref=born['birth_delivery_raw_ref'])
    assert set(member) == CAPTURE_MEMBER_KEYS
    assert _generation(capture_policy, member, expected_states={'active'})[0] == born
    original_owner = copy.deepcopy(member)
    if change == 'sponsor':
        member['sponsoring_intent_id'] = 'other-intent'
    elif change == 'intent_raw':
        member['scene_intent_raw_ref'] = dict(consent['intent_raw_ref'], sha256='sha256:' + 'f' * 64)
    elif change == 'hybrid':
        member.update(owner_intent_id=consent['intent_id'], owner_raw_ref=consent['intent_raw_ref'])
    consent['members'].append(member)
    pin = Path(consent['terminal_pin_refs'][0]['path'])
    before = pin.read_bytes()
    if change == 'none':
        result = selected(fresh, policy, consent, allowance)
        assert len(result['terminal_pin_release_rows']) == 1
        assert result['references_clear'] is False and result['consumer_fence_checked'] is False
        assert member == original_owner and 'owner_intent_id' not in member
        assert _generation(capture_policy, member, expected_states={'active'})[0] == born
    else:
        with pytest.raises(ValueError, match='scene_retirement_terminal_pin_unproven'):
            selected(fresh, policy, consent, allowance)
    assert pin.read_bytes() == before


@pytest.mark.parametrize('change',['omitted','foreign_path','changed_raw','young','outside_member','other_dependent','missing_terminal_result'])
def test_terminal_pin_selection_never_clears_unknown_live_or_unrelated_rows(tmp_path,monkeypatch,change):
    fresh,policy,consent,allowance=fixture(tmp_path,monkeypatch)
    pin=fresh['reference_observation']['protections'][-1]['observation']
    if change=='omitted':
        consent['terminal_pin_refs']=[]
    elif change=='foreign_path':
        consent['terminal_pin_refs'][0]['path']=str(tmp_path/'foreign.json')
    elif change=='changed_raw':
        Path(pin['row_path']).write_bytes(b'{}')
    elif change=='young':
        from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
        allowance=ActionAllowance(expires_at=40000,now=lambda:100,monotonic=lambda:0)
    elif change=='outside_member':
        consent['members'][0]['canonical_path']=str(tmp_path/'unrelated')
    elif change=='other_dependent':
        foreign=copy.deepcopy(pin)
        foreign.update(kind='activation',owner_id='other-owner',row_path=str(tmp_path/'pins'/'activation'/'other-owner.json'),
                       depends_on=[{'kind':pin['kind'],'owner_id':pin['owner_id']}])
        fresh['reference_observation']['protections'].append({'kind':'pin_observation','observation':foreign,'action':'KEEP'})
    else:
        fresh['reference_observation']['record_dispositions']=[row for row in fresh['reference_observation']['record_dispositions']
                                                              if row['source']['role']!='result']
        for member in fresh['measured_members']:
            member['source_provenance']=[p for p in member['source_provenance'] if p['role']!='preparation_results']
    with pytest.raises(ValueError,match='scene_retirement_'):
        selected(fresh,policy,consent,allowance)


def test_native_selected_compilation_pin_uses_exact_current_result_not_primary_only_index(tmp_path,monkeypatch):
    from tests.test_scene_retirement_compilation_reference_transfer import fixture as compilation
    from blueprint_pipeline.control_plane_storage_pins import write_storage_pin
    from blueprint_pipeline.control_plane_storage_pin_observation import observe_storage_pins
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    fresh,proofs,_=compilation(tmp_path)
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    monkeypatch.setattr(access,'_INSTALLED_POLICY',tmp_path/'absent-policy')
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE',raising=False)
    value=json.loads(Path(proofs['compilation_results']['path']).read_bytes())
    member=Path(fresh['planner_context']['roots']['compilation_output_root'])/value['compilation_id']
    pins=tmp_path/'pins'
    write_storage_pin(pins_root=pins,kind='compilation',owner_id=value['compilation_id'],paths=[member],now=lambda:0)
    observed=json.loads(json.dumps(asdict(observe_storage_pins(str(pins),observed_at_epoch=30000,monotonic=lambda:0).rows[0])))
    # The selected owner is retained at the real native intent path.
    import hashlib
    intent_path=Path(fresh['planner_context']['roots']['intent_root'])/'intent-1'/'intent.json'
    intent_bytes=intent_path.read_bytes()
    intent={'role':'intent','path':str(intent_path),'sha256':'sha256:'+hashlib.sha256(intent_bytes).hexdigest(),
            'size_bytes':len(intent_bytes),'seal_field':'intent_digest',
            'seal_digest':json.loads(intent_bytes)['intent_digest']}
    raw={key:intent[key] for key in ('path','sha256','size_bytes')}
    fresh.update(selected_intent_provenance=intent,finished_observation={'status':'completed'})
    fresh['planner_context']['pins_root']=str(pins)
    fresh['reference_observation']['protections'].append({'kind':'pin_observation','observation':observed,'action':'KEEP'})
    consent={'intent_id':'intent-1','intent_raw_ref':raw,'members':[{'canonical_path':str(member),
             'owner_intent_id':'intent-1','owner_raw_ref':raw}],
             'terminal_pin_refs':[{'path':observed['row_path'],'sha256':observed['raw_sha256'],
                                  'size_bytes':observed['raw_size_bytes']}]}
    allowance=ActionAllowance(expires_at=40000,now=lambda:30000,monotonic=lambda:0)
    result=selected(fresh,{'reference_context':fresh['planner_context']},consent,allowance)
    assert result['terminal_pin_release_rows'][0]['original_value']['kind']=='compilation'
