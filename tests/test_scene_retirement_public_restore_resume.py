"""Actual public restore replay; reference/process clearance remains isolated."""
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_retirement_public_resume import pending_action


def interrupted_public_restore(tmp_path,monkeypatch,phase):
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
    engine,policy,scope,retire_consent,_,transport=pending_action(tmp_path,monkeypatch)
    retired=engine.retire_scene(scope['plan_raw_ref']['path'],retire_consent,transport=transport,
                                now=lambda:201,monotonic=lambda:1)
    assert retired['status']=='retired',retired
    value=dict(scope,consent_id='b'*32,action='restore',plan_raw_ref=None,
               retired_journal_raw_ref=retired['retired_journal_raw_ref'])
    value['consent_digest']=canonical_digest(value,digest_field='consent_digest')
    consent=tmp_path/'restore-consent.json'
    consent.write_text(json.dumps(value))
    consent.chmod(0o600)
    original=SceneJournal.append
    original_progress=engine.publish_progress_receipt
    def interrupted(self,event,**kwargs):
        reference=original(self,event,**kwargs)
        if event==phase:
            raise OSError('owned interruption after durable restore event')
        return reference
    monkeypatch.setattr(SceneJournal,'append',interrupted)
    if phase=='activated-prefix':
        def interrupted_progress(*args,**kwargs):
            reference=original_progress(*args,**kwargs)
            if args[3]['status']=='restoring' and len(args[3]['members'])==1:
                raise OSError('owned interruption after activated prefix receipt')
            return reference
        monkeypatch.setattr(engine,'publish_progress_receipt',interrupted_progress)
    result=engine.restore_scene(retired['retired_journal_raw_ref']['path'],consent,transport=transport,
                               now=lambda:202,monotonic=lambda:2)
    assert result['status']=='incomplete',result
    monkeypatch.setattr(SceneJournal,'append',original)
    monkeypatch.setattr(engine,'publish_progress_receipt',original_progress)
    return engine,policy,value,consent,retired,result,transport


@pytest.mark.parametrize('phase',['restore_directory_created','restore_file_created','member_restored','activated-prefix'])
def test_public_restore_resumes_original_restore_token_and_full_member_union(tmp_path,monkeypatch,phase):
    engine,_,scope,consent,retired,first,transport=interrupted_public_restore(tmp_path,monkeypatch,phase)
    result=engine.restore_scene(retired['retired_journal_raw_ref']['path'],consent,transport=transport,
                               now=lambda:203,monotonic=lambda:3)
    assert result['status']=='restored',result
    assert result['token']==Path(first['journal_initial_raw_ref']['path']).name.split('.')[0]
    assert len(result['members'])==2
    assert (Path(scope['members'][0]['canonical_path'])/'proof.bin').read_bytes()==b'actual-preserved-proof'
    assert (Path(scope['members'][1]['canonical_path'])/'proof.bin').read_bytes()==b'second-member-proof'
    receipt=json.loads(Path(result['intent_receipt_path']).read_bytes())
    assert receipt['status']=='restored' and receipt['restore_token']==result['token']
    assert receipt['retiring_token']==retired['token']


def test_public_restore_replay_cannot_restart_the_original_deadline(tmp_path,monkeypatch):
    engine,_,scope,consent,retired,_,transport=interrupted_public_restore(tmp_path,monkeypatch,'restore_directory_created')
    original={row['canonical_path']:Path(row['canonical_path']).exists() for row in scope['members']}
    entered=[]
    transport.read_archive=lambda *args:entered.append('physical-read')
    result=engine.restore_scene(retired['retired_journal_raw_ref']['path'],consent,transport=transport,
                               now=lambda:230,monotonic=lambda:30)
    assert result['reason']=='scene_retirement_deadline',result
    assert entered==[]
    assert {row['canonical_path']:Path(row['canonical_path']).exists() for row in scope['members']}==original


def test_public_restore_recovery_preserves_a_substituted_capture(tmp_path,monkeypatch):
    engine,_,scope,consent,retired,_,transport=interrupted_public_restore(tmp_path,monkeypatch,'restore_directory_created')
    destination=Path(scope['members'][0]['canonical_path'])
    destination.rename(destination.parent/'owned-original-restore')
    destination.mkdir()
    (destination/'new-capture.bin').write_bytes(b'foreign-capture')
    result=engine.restore_scene(retired['retired_journal_raw_ref']['path'],consent,transport=transport,
                               now=lambda:203,monotonic=lambda:3)
    assert result['status']!='restored',result
    assert (destination/'new-capture.bin').read_bytes()==b'foreign-capture'


def test_completed_public_restore_repeat_reverifies_bytes_without_generation_reset(tmp_path,monkeypatch):
    engine,policy,_,consent,retired,_,transport=interrupted_public_restore(tmp_path,monkeypatch,'member_restored')
    first=engine.restore_scene(retired['retired_journal_raw_ref']['path'],consent,transport=transport,
                              now=lambda:203,monotonic=lambda:3)
    assert first['status']=='restored',first
    directory=Path(policy['generation_store'])
    generations={path.name:path.read_bytes() for path in directory.glob('*.json')}
    second=engine.restore_scene(retired['retired_journal_raw_ref']['path'],consent,transport=transport,
                               now=lambda:204,monotonic=lambda:4)
    assert second['status']=='restored',second
    assert second['token']==first['token']
    assert {path.name:path.read_bytes() for path in directory.glob('*.json')}==generations
