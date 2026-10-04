"""Durable exact detachment intent precedes every namespace or leaf mutation."""
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_scene_retirement_real_participants import access_fixture


def test_journal_persists_exact_planned_detach_before_any_payload_mutation(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    access,policy,member=access_fixture(tmp_path,monkeypatch)
    store=tmp_path/'journals'
    store.mkdir(mode=0o700)
    payload=member/'evidence.bin'
    payload.write_bytes(b'durable-original')
    info=member.stat()
    journal=SceneJournal.create(store,token='1'*32,initial={'schema_version':'scene_retirement_journal.v1',
        'status':'pending','members':[{'canonical_path':str(member),'generation_id':'2'*32,
                                     'pre_identity':[info.st_dev,info.st_ino,info.st_mode]}]},
        allowance=ActionAllowance(expires_at=200,now=lambda:100,monotonic=lambda:0))
    destination=str(member.parent/('.scene-retirement-'+'1'*32+'-0'))
    event=journal.append('detach_planned',member_key='0',evidence={'canonical_path':str(member),
        'detached_path':destination,'generation_id':'2'*32,'pre_identity':[info.st_dev,info.st_ino,info.st_mode],
        'parent_identity':[member.parent.stat().st_dev,member.parent.stat().st_ino,member.parent.stat().st_mode],
        'inventory_sha256':'sha256:'+'a'*64})
    value=json.loads(open(event['path']).read())
    assert value['event']=='detach_planned' and value['evidence']['detached_path']==destination
    assert value['prior_event_sha256']==journal.initial_ref['sha256']
    assert event['sha256']=='sha256:'+hashlib.sha256(open(event['path'],'rb').read()).hexdigest()
    assert payload.read_bytes()==b'durable-original'
    assert not (member.parent/('.scene-retirement-'+'1'*32+'-0')).exists()


def test_existing_journal_token_cannot_be_replaced_and_expiry_refuses_new_event(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    access,_,_=access_fixture(tmp_path,monkeypatch)
    store=tmp_path/'journals'
    store.mkdir(mode=0o700)
    now=[100]
    allowance=ActionAllowance(expires_at=200,now=lambda:now[0],monotonic=lambda:0)
    journal=SceneJournal.create(store,token='1'*32,initial={'status':'pending'},allowance=allowance)
    original=open(journal.initial_ref['path'],'rb').read()
    with pytest.raises(FileExistsError):
        SceneJournal.create(store,token='1'*32,initial={'status':'different'},allowance=allowance)
    assert open(journal.initial_ref['path'],'rb').read()==original
    now[0]=200
    with pytest.raises(ValueError,match='scene_retirement_consent_expired'):
        journal.append('detach_planned',member_key='0',evidence={})
    assert len(list(store.glob('*.json')))==1


def test_large_initial_journal_checks_parent_per_bounded_chunk(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_journal as journal_module
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    access_fixture(tmp_path,monkeypatch)
    store=tmp_path/'journals'
    store.mkdir(mode=0o700)
    original_parent=journal_module._parent
    calls=[]
    def counted_parent(*args,**kwargs):
        calls.append(None)
        return original_parent(*args,**kwargs)
    monkeypatch.setattr(journal_module,'_parent',counted_parent)
    payload={'members':[{'index':index,'proof':'x'*96} for index in range(1000)]}
    journal=journal_module.SceneJournal.create(store,token='3'*32,initial=payload,
        allowance=ActionAllowance(expires_at=200,now=lambda:100,monotonic=lambda:0))
    raw=Path(journal.initial_ref['path']).read_bytes()
    assert len(raw)>100_000 and hashlib.sha256(raw).hexdigest()==journal.initial_ref['sha256'][7:]
    assert json.loads(raw)['members']==payload['members']
    assert len(calls)<=32
