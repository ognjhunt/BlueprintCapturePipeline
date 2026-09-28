"""Real public journal recovery; current-lifetime admission isolated explicitly.

These tests do not assert old-worker or reference clearance. Those separate
admission functions are stubbed while actual authority, EX, preservation,
mutation, generations and intent receipts execute.
"""
import hashlib
import json
import os
from pathlib import Path


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
    monkeypatch.setattr(engine,'_current_plan',lambda *args: {})
    monkeypatch.setattr(engine,'_resume_current_references',lambda *args: None,raising=False)
    # No assertion of unknown process absence: this test only exercises replay.
    monkeypatch.setattr(engine,'_installed_cohort',lambda *args: None)
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
    original_progress=engine.publish_progress_receipt
    if after_generation:
        def interrupted_progress(*args,**kwargs):
            raise OSError('owned interruption after native removal and generation update')
        monkeypatch.setattr(engine,'publish_progress_receipt',interrupted_progress)
    else:
        monkeypatch.setattr(engine,'detach_and_remove',interrupted)
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:200,monotonic=lambda:0)
    assert result['status']=='incomplete',result
    assert not Path(members[0]['canonical_path']).exists()
    assert Path(members[1]['canonical_path']).exists()
    monkeypatch.setattr(engine,'detach_and_remove',original)
    monkeypatch.setattr(engine,'publish_progress_receipt',original_progress)
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
