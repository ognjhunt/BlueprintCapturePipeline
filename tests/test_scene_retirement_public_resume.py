"""Real public journal recovery; current-lifetime admission isolated explicitly.

These tests do not assert old-worker or reference clearance. Those separate
admission functions are stubbed while actual authority, EX, preservation,
mutation, generations and intent receipts execute.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest


from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_retirement_intent_receipt import operation,raw_ref


def fresh_action(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    policy,scope,_,_,_,_,transport=operation(tmp_path,monkeypatch,member_count=2,with_transport=True)
    policy['principals']=[dict(principal_id='operator',actions=['retire','restore'],owner_intent_ids=[scope['intent_id']],private_archive_classes=['host'])]
    policy['private_archive_allowed_classes']=['host']
    policy['limits']={'elapsed_seconds':20}
    policy['policy_digest']=canonical_digest(policy,digest_field='policy_digest')
    policy_path=Path(os.environ['BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE'])
    policy_path.write_text(json.dumps(policy))
    (Path(policy['journal_store']+'.metadata')).mkdir(mode=0o750)
    members=[]
    for row in scope['members']:
        info=Path(row['canonical_path']).stat()
        member=dict(row,dev=info.st_dev,ino=info.st_ino,mode=info.st_mode)
        members.append(member)
        generation=dict(schema_version='scene_member_generation.v1',canonical_path=member['canonical_path'],
            owner_intent_id=member['owner_intent_id'],owner_raw_ref=member['owner_raw_ref'],generation_id=member['generation_id'],
            dev=info.st_dev,ino=info.st_ino,mode=info.st_mode,state='active',state_sequence=1)
        generation['state_digest']=canonical_digest(generation,digest_field='state_digest')
        path=Path(policy['generation_store'])/(hashlib.sha256(member['canonical_path'].encode()).hexdigest()+'.json')
        path.write_text(json.dumps(generation))
        path.chmod(0o600)
    scope.update(schema_version='scene_retirement_consent.v1',consent_id='a'*32,principal_id='operator',
        retired_journal_raw_ref=None,action='retire',created_at=100,expires_at=999,members=members,
        policy_sha256=raw_ref(policy_path)['sha256'],cohort_sha256=canonical_digest({'consumer_cohort':[]}),
        private_archive_classes=['host'])
    scope['consent_digest']=canonical_digest(scope,digest_field='consent_digest')
    consent=tmp_path/'resume-consent.json'
    consent.write_text(json.dumps(scope))
    consent.chmod(0o600)
    # Replay-only isolation supplies a complete empty reference-stage result;
    # production admission still performs the real scans and native closure.
    monkeypatch.setattr(engine,'_current_plan',lambda *args: {'unselected_shared_content_keeps':[], 'reference_observation':{
        'blockers':[], 'child_scopes':[{'child':name,'complete':True}
            for name in ('pins','primary_queues','auxiliary_queues')],
        'record_dispositions':[], 'protections':[]}})
    monkeypatch.setattr(engine,'_resume_current_references',lambda *args,**kwargs: None,raising=False)
    # No assertion of unknown process absence: this test only exercises replay.
    monkeypatch.setattr(engine,'_installed_cohort',lambda *args: None)
    # Public replay mechanics only; installed native admission is mandatory in
    # production and separately challenged by restore-reader refusal tests.
    monkeypatch.setattr(engine,'_current_readers',lambda *args: None)
    return engine,policy,scope,consent,transport


def pending_action(tmp_path,monkeypatch,*,after_generation=False):
    engine,policy,scope,consent,transport=fresh_action(tmp_path,monkeypatch)
    members=scope['members']
    original=engine.detach_and_remove
    def interrupted(*args,**kwargs):
        outcome=original(*args,**kwargs)
        if kwargs['member_index']==0:
            raise OSError('owned test interruption after durable native removal')
        return outcome
    original_transition=engine._transition
    if after_generation:
        def interrupted_transition(*args,**kwargs):
            updated=original_transition(*args,**kwargs)
            if kwargs['state']=='retired' and args[1]['canonical_path']==members[0]['canonical_path']:
                raise OSError('owned interruption after native removal and generation update')
            return updated
        monkeypatch.setattr(engine,'_transition',interrupted_transition)
    else:
        monkeypatch.setattr(engine,'detach_and_remove',interrupted)
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:200,monotonic=lambda:0)
    assert result['status']=='incomplete',result
    assert not Path(members[0]['canonical_path']).exists()
    assert Path(members[1]['canonical_path']).exists()
    monkeypatch.setattr(engine,'detach_and_remove',original)
    monkeypatch.setattr(engine,'_transition',original_transition)
    # Original source-presence assertion is for INITIAL archive readback only;
    # replay validates the exact already-preserved private archive natively.
    def readback(uri):
        yield transport.objects[uri]
    transport.read_archive=readback
    return engine,policy,scope,consent,result,transport


def test_public_retire_resumes_same_journal_and_finishes_remaining_member(tmp_path,monkeypatch):
    engine,_,scope,consent,first,transport=pending_action(tmp_path,monkeypatch)
    old_objects=set(transport.objects)
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:201,monotonic=lambda:1)
    assert result['status']=='retired',result
    assert set(transport.objects)==old_objects,'replay uploaded under a fresh action'
    assert result['token']==Path(first['journal_initial_raw_ref']['path']).name.split('.')[0]
    assert all(not Path(row['canonical_path']).exists() for row in scope['members'])
    assert len(result['members'])==2 and all(row['removed_file_count']==1 for row in result['members'])
    assert json.loads(Path(result['intent_receipt_path']).read_bytes())['status']=='retired'


def test_durable_member_journal_finishes_without_rechecking_public_progress_per_member(tmp_path,monkeypatch):
    engine,_,scope,consent,transport=fresh_action(tmp_path,monkeypatch)
    monkeypatch.setattr(engine,'publish_progress_receipt',
                        lambda *args,**kwargs: pytest.fail('public progress reread per member'))
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,
                               now=lambda:200,monotonic=lambda:0)
    assert result['status']=='retired',result
    assert len(result['members'])==2
    assert json.loads(Path(result['intent_receipt_path']).read_bytes())['status']=='retired'


def test_public_resume_after_retired_snapshot_before_terminal_receipt(tmp_path,monkeypatch):
    engine,_,scope,consent,transport=fresh_action(tmp_path,monkeypatch)
    original_terminal=engine.publish_terminal_receipt
    monkeypatch.setattr(engine,'publish_terminal_receipt',
                        lambda *args,**kwargs: (_ for _ in ()).throw(OSError('owned terminal projection interruption')))
    first=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,
                              now=lambda:200,monotonic=lambda:0)
    assert first['status']=='incomplete',first
    assert all(not Path(row['canonical_path']).exists() for row in scope['members'])
    objects=dict(transport.objects)
    monkeypatch.setattr(engine,'publish_terminal_receipt',original_terminal)
    transport.read_archive=lambda uri: iter((transport.objects[uri],))
    second=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,
                               now=lambda:201,monotonic=lambda:1)
    assert second['status']=='retired',second
    assert transport.objects==objects
    assert json.loads(Path(second['intent_receipt_path']).read_bytes())['status']=='retired'


def test_public_recovery_cannot_renew_the_original_action_deadline(tmp_path,monkeypatch):
    engine,_,scope,consent,_,transport=pending_action(tmp_path,monkeypatch)
    remaining=Path(scope['members'][1]['canonical_path'])/'proof.bin'
    original=remaining.read_bytes()
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:230,monotonic=lambda:30)
    assert result['reason']=='scene_retirement_deadline',result
    assert remaining.read_bytes()==original


def test_public_recovery_never_adopts_new_capture_at_an_old_removed_name(tmp_path,monkeypatch):
    engine,_,scope,consent,_,transport=pending_action(tmp_path,monkeypatch)
    foreign=Path(scope['members'][0]['canonical_path'])
    foreign.mkdir()
    (foreign/'new-capture.bin').write_bytes(b'foreign-identity')
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:201,monotonic=lambda:1)
    assert result['status']!='retired',result
    assert (foreign/'new-capture.bin').read_bytes()==b'foreign-identity'
    assert Path(scope['members'][1]['canonical_path']).exists()


def test_public_recovery_retains_reserved_counts_before_readback(tmp_path,monkeypatch):
    engine,_,scope,consent,first,transport=pending_action(tmp_path,monkeypatch)
    initial=json.loads(Path(first['journal_initial_raw_ref']['path']).read_bytes())
    events=sorted(Path(first['journal_initial_raw_ref']['path']).parent.glob(initial['token']+'.[0-9]*.json'),
                  key=lambda path:int(path.name.split('.')[1]))
    reserved=[json.loads(path.read_bytes())['evidence']['action_allowance'] for path in events
              if json.loads(path.read_bytes())['event']=='allowance_reserved'][-1]
    assert reserved['counts']['local_bytes']>initial['action_allowance']['counts']['local_bytes']
    original_consume=engine._consume
    observations=[]
    def consume(preserved,transport,allowance):
        observations.append(dict(allowance.counts))
        assert allowance._resume_origin['start_monotonic']==initial['action_allowance']['start_monotonic']
        assert allowance.counts['local_bytes']>=reserved['counts']['local_bytes']
        assert allowance.counts['remote_bytes']>=reserved['counts']['remote_bytes']+preserved['archive']['size_bytes']
        return original_consume(preserved,transport,allowance)
    monkeypatch.setattr(engine,'_consume',consume)
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:201,monotonic=lambda:1)
    assert result['status']=='retired',result
    assert len(observations)==1


def test_public_recovery_handles_durable_retired_generation_before_receipt(tmp_path,monkeypatch):
    engine,policy,scope,consent,_,transport=pending_action(tmp_path,monkeypatch,after_generation=True)
    member=scope['members'][0]
    path=Path(policy['generation_store'])/(hashlib.sha256(member['canonical_path'].encode()).hexdigest()+'.json')
    before=json.loads(path.read_bytes())
    assert before['state']=='retired'
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:201,monotonic=lambda:1)
    assert result['status']=='retired',result
    assert json.loads(path.read_bytes())==before,'replay rewrote an already proved retired generation'


def test_public_recovery_cannot_borrow_a_changed_consent(tmp_path,monkeypatch):
    engine,_,scope,consent,_,transport=pending_action(tmp_path,monkeypatch)
    value=json.loads(consent.read_bytes())
    value['created_at']=101
    value['consent_digest']=canonical_digest(value,digest_field='consent_digest')
    consent.write_text(json.dumps(value))
    entered=[]
    monkeypatch.setattr(engine,'_consume',lambda *args:entered.append('readback'))
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:201,monotonic=lambda:1)
    assert result['status']!='retired',result
    assert entered==[] and Path(scope['members'][1]['canonical_path']).exists()


def test_completed_preparation_before_private_initial_resumes_without_second_upload(tmp_path,monkeypatch):
    engine,policy,scope,consent,transport=fresh_action(tmp_path,monkeypatch)
    create=engine.SceneJournal.create
    def interrupted(*args,**kwargs):
        raise OSError('owned fault after archive before private initializer')
    monkeypatch.setattr(engine.SceneJournal,'create',interrupted)
    first=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:200,monotonic=lambda:0)
    assert first['status']!='retired' and all(Path(row['canonical_path']).exists() for row in scope['members'])
    old_objects=set(transport.objects)
    monkeypatch.setattr(engine.SceneJournal,'create',create)
    second=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:201,monotonic=lambda:1)
    assert set(transport.objects)==old_objects,'unproven preparation retried under a new archive token'
    assert second['status']=='retired',second
    claim=json.loads((Path(policy['journal_store'])/('retirement-attempt.'+scope['consent_id']+'.json')).read_bytes())
    assert second['token']==claim['token']
    assert all(not Path(row['canonical_path']).exists() for row in scope['members'])


def test_private_initial_without_public_projection_resumes_exact_token_and_origin(tmp_path,monkeypatch):
    engine,_,scope,consent,transport=fresh_action(tmp_path,monkeypatch)
    publish=engine.publish_pending_receipt
    monkeypatch.setattr(engine,'publish_pending_receipt',lambda *args:(_ for _ in ()).throw(OSError('owned pre-projection crash')))
    first=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:200,monotonic=lambda:0)
    assert first['status']=='incomplete'
    old_objects=set(transport.objects)
    old_token=Path(first['journal_initial_raw_ref']['path']).name.split('.')[0]
    monkeypatch.setattr(engine,'publish_pending_receipt',publish)
    second=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:201,monotonic=lambda:1)
    assert second['status']=='retired',second
    assert second['token']==old_token and set(transport.objects)==old_objects


def test_unfinished_preparation_claim_cannot_renew_original_deadline(tmp_path,monkeypatch):
    engine,_,scope,consent,transport=fresh_action(tmp_path,monkeypatch)
    create=engine.SceneJournal.create
    monkeypatch.setattr(engine.SceneJournal,'create',lambda *args,**kwargs:(_ for _ in ()).throw(OSError('owned initial crash')))
    engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:200,monotonic=lambda:0)
    old_objects=set(transport.objects)
    monkeypatch.setattr(engine.SceneJournal,'create',create)
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:230,monotonic=lambda:30)
    assert result['reason']=='scene_retirement_deadline',result
    assert set(transport.objects)==old_objects and all(Path(row['canonical_path']).exists() for row in scope['members'])


def interrupt_partial_archive(transport):
    put=transport.put_archive
    entered=[]
    def interrupted(key,chunks):
        entered.append(key)
        stream=iter(chunks)
        raw=next(stream)
        transport.objects['s3://private-fixture/'+key]=raw
        stream.close()
        raise OSError('owned partial upload interruption')
    transport.put_archive=interrupted
    return put,entered


def test_unknown_partial_preparation_retries_same_original_token_without_overwrite(tmp_path,monkeypatch):
    engine,policy,scope,consent,transport=fresh_action(tmp_path,monkeypatch)
    put,entered=interrupt_partial_archive(transport)
    first=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:200,monotonic=lambda:0)
    assert first['status']=='incomplete' and len(entered)==1,first
    assert first['preparation_budget_counts']['remote_bytes']>0
    assert first['preparation_claim_raw_ref']['path'].endswith(scope['consent_id']+'.json')
    assert first['last_preparation_escrow_raw_ref']['path'].endswith('.1.escrow.json')
    old_objects=dict(transport.objects)
    transport.put_archive=put
    second=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:201,monotonic=lambda:1)
    assert second['status']=='retired',second
    assert entered[0]==second['token']+'.1.tar'
    assert second['token']+'.2.tar' in {Path(key).name for key in transport.objects}
    assert all(transport.objects[key]==raw for key,raw in old_objects.items())
    claim=json.loads((Path(policy['journal_store'])/('retirement-attempt.'+scope['consent_id']+'.json')).read_bytes())
    initial=json.loads(Path(second['retired_journal_raw_ref']['path']).read_bytes())
    assert initial['action_allowance']['start_monotonic']==claim['action_allowance']['start_monotonic']==0
    assert initial['action_allowance']['started_wall']==200


def test_preparation_escrow_is_durable_before_first_payload_read(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_preservation as preservation
    engine,policy,scope,consent,transport=fresh_action(tmp_path,monkeypatch)
    original=preservation._payload
    observed=[]
    def payload(*args,**kwargs):
        records=list(Path(policy['journal_store']).glob('preparation.*.escrow.json'))
        assert records,'payload read occurred before durable original-budget escrow'
        escrow=json.loads(records[-1].read_bytes())
        assert escrow['action_allowance']['counts']['local_bytes']>0
        assert escrow['action_allowance']['counts']['archive_bytes']>0
        assert escrow['action_allowance']['counts']['remote_bytes']>0
        observed.append(escrow)
        yield from original(*args,**kwargs)
    monkeypatch.setattr(preservation,'_payload',payload)
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:200,monotonic=lambda:0)
    assert result['status']=='retired',result
    assert observed


def test_unknown_partial_attempt_never_refunds_escrow_on_retry(tmp_path,monkeypatch):
    engine,policy,scope,consent,transport=fresh_action(tmp_path,monkeypatch)
    policy['limits']['remote_bytes']=10000
    policy['policy_digest']=canonical_digest(policy,digest_field='policy_digest')
    policy_path=Path(os.environ['BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE'])
    policy_path.write_text(json.dumps(policy))
    scope['policy_sha256']=raw_ref(policy_path)['sha256']
    scope['consent_digest']=canonical_digest(scope,digest_field='consent_digest')
    consent.write_text(json.dumps(scope))
    _,entered=interrupt_partial_archive(transport)
    engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:200,monotonic=lambda:0)
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:201,monotonic=lambda:1)
    assert result['reason']=='scene_retirement_byte_limit',result
    assert len(entered)==1,'retry reached physical transport after original escrow exhausted remaining cap'
    assert all(Path(row['canonical_path']).exists() for row in scope['members'])


def test_changed_consent_inventory_refuses_before_archive_upload(tmp_path,monkeypatch):
    engine,_,scope,consent,transport=fresh_action(tmp_path,monkeypatch)
    payload=Path(scope['members'][0]['canonical_path'])/'proof.bin'
    payload.write_bytes(b'changed-private-source')
    original_objects=dict(transport.objects)
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:200,monotonic=lambda:0)
    assert result['reason']=='scene_retirement_inventory_changed',result
    assert transport.objects==original_objects,'unknown changed bytes crossed archive transport before consent inventory proof'
    assert payload.read_bytes()==b'changed-private-source'


@pytest.mark.parametrize('interrupted',[False,True])
def test_public_pin_phase_releases_before_removal_and_restores_after_current_generations(tmp_path,monkeypatch,interrupted):
    """Isolate lineage/lifetime admission; actual EX, pin, journal and action run.

    This is integration of previously selected rows, not a current-reader or
    all-family acceptance proof. The selected native lineage is covered by the
    separate real terminal-pin selectors.
    """
    from blueprint_pipeline.control_plane_storage_pins import write_storage_pin
    from blueprint_pipeline.task_evaluation_scene_retirement_pin_mutation import _snapshot
    engine,policy,scope,consent,transport=fresh_action(tmp_path,monkeypatch)
    pins=tmp_path/'pins'
    first=Path(scope['members'][0]['canonical_path'])
    write_storage_pin(pins_root=pins,kind='preparation',owner_id=first.name,paths=[first],now=lambda:0)
    pin=pins/'preparation'/(first.name+'.json')
    original=pin.read_bytes()
    reference=raw_ref(pin)
    selected=dict(original_raw_ref=reference,original_value=json.loads(original),original_raw_hex=original.hex(),
        physical_identity=[pin.stat().st_dev,pin.stat().st_ino,pin.stat().st_mode],snapshot=_snapshot(pin.stat()))
    policy['reference_context']['pins_root']=str(pins)
    policy['policy_digest']=canonical_digest(policy,digest_field='policy_digest')
    policy_path=Path(os.environ['BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE'])
    policy_path.write_text(json.dumps(policy))
    scope.update(terminal_pin_refs=[reference],policy_sha256=raw_ref(policy_path)['sha256'])
    scope['consent_digest']=canonical_digest(scope,digest_field='consent_digest')
    consent.write_text(json.dumps(scope))
    proof={'archive_inventory_verified':True,'terminal_pin_release_rows':[selected]}
    monkeypatch.setattr(engine,'validate_current_reference_transfer',lambda *args,**kwargs:proof)
    current=engine._current_plan
    monkeypatch.setattr(engine,'_current_plan',lambda *args:dict(current(*args),reference_transfer=proof))
    removed=engine.detach_and_remove
    def remove(*args,**kwargs):
        assert json.loads(pin.read_bytes())['released_at_epoch'] is not None
        assert any(event['event']=='pin_released' and
            event['evidence']['released_raw_ref']==raw_ref(pin) for event in kwargs['journal'].events)
        return removed(*args,**kwargs)
    monkeypatch.setattr(engine,'detach_and_remove',remove)
    if interrupted:
        def interrupt(*args,**kwargs):
            assert json.loads(pin.read_bytes())['released_at_epoch'] is not None
            raise OSError('owned interruption after durable pin release before first folder mutation')
        monkeypatch.setattr(engine,'detach_and_remove',interrupt)
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:200,monotonic=lambda:0)
    if interrupted:
        from blueprint_pipeline.task_evaluation_scene_retirement_pin_mutation import pin_history
        assert result['status']=='incomplete',result
        assert all(Path(member['canonical_path']).exists() for member in scope['members'])
        initial=json.loads(Path(result['journal_initial_raw_ref']['path']).read_bytes())
        from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
        from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
        journal=SceneJournal.resume(result['journal_initial_raw_ref'],
            allowance=ActionAllowance(expires_at=999,now=lambda:201,monotonic=lambda:1))
        history=pin_history(policy,scope,journal)
        observed=history[tuple(reference[key] for key in ('path','sha256','size_bytes'))]
        assert observed['original_raw_ref']==reference
        assert observed['observed_raw_ref']==raw_ref(pin)
        original_objects=set(transport.objects)
        monkeypatch.setattr(engine,'detach_and_remove',remove)
        result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:201,monotonic=lambda:1)
        assert result['token']==initial['token']
        assert set(transport.objects)==original_objects,'pin replay created a new preservation origin'
    assert result['status']=='retired',result
    retired=json.loads(Path(result['retired_journal_raw_ref']['path']).read_bytes())
    assert retired['terminal_pin_release_rows']==[selected] and len(retired['terminal_pin_outcomes'])==1
    scope.update(action='restore',plan_raw_ref=None,retired_journal_raw_ref=result['retired_journal_raw_ref'])
    scope['consent_digest']=canonical_digest(scope,digest_field='consent_digest')
    consent.write_text(json.dumps(scope))
    transport.read_archive=lambda uri:iter([transport.objects[uri]])
    restored=engine.restore_scene(result['retired_journal_raw_ref']['path'],consent,transport=transport,now=lambda:202,monotonic=lambda:2)
    assert restored['status']=='restored',restored
    assert pin.read_bytes()==original
    for member in scope['members']:
        assert engine._generation(policy,member,expected_states={'restored-active'})[0]['state']=='restored-active'
