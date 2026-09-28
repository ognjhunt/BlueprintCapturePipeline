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


def pending_action(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    policy,scope,_,_,_,_,transport=operation(tmp_path,monkeypatch,member_count=2,with_transport=True)
    policy['principals']=[dict(principal_id='operator',actions=['retire'],owner_intent_ids=[scope['intent_id']],private_archive_classes=['host'])]
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
        path.write_text(json.dumps(generation));path.chmod(0o600)
    scope.update(schema_version='scene_retirement_consent.v1',consent_id='a'*32,principal_id='operator',
        retired_journal_raw_ref=None,action='retire',created_at=100,expires_at=999,members=members,
        policy_sha256=raw_ref(policy_path)['sha256'],cohort_sha256=canonical_digest({'consumer_cohort':[]}),
        private_archive_classes=['host'])
    scope['consent_digest']=canonical_digest(scope,digest_field='consent_digest')
    consent=tmp_path/'resume-consent.json';consent.write_text(json.dumps(scope));consent.chmod(0o600)
    monkeypatch.setattr(engine,'_current_plan',lambda *args: {})
    monkeypatch.setattr(engine,'_resume_current_references',lambda *args: None,raising=False)
    # No assertion of unknown process absence: this test only exercises replay.
    monkeypatch.setattr(engine,'_installed_cohort',lambda *args: None)
    original=engine.detach_and_remove
    def interrupted(*args,**kwargs):
        outcome=original(*args,**kwargs)
        if kwargs['member_index']==0:
            raise OSError('owned test interruption after durable native removal')
        return outcome
    monkeypatch.setattr(engine,'detach_and_remove',interrupted)
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:200,monotonic=lambda:0)
    assert result['status']=='incomplete',result
    assert not Path(members[0]['canonical_path']).exists()
    assert Path(members[1]['canonical_path']).exists()
    monkeypatch.setattr(engine,'detach_and_remove',original)
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
    foreign.mkdir();(foreign/'new-capture.bin').write_bytes(b'foreign-identity')
    result=engine.retire_scene(scope['plan_raw_ref']['path'],consent,transport=transport,now=lambda:201,monotonic=lambda:1)
    assert result['status']!='retired',result
    assert (foreign/'new-capture.bin').read_bytes()==b'foreign-identity'
    assert Path(scope['members'][1]['canonical_path']).exists()
