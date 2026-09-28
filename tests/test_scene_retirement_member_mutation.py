"""Actual no-replace detach, proved leaf removal and fsynced outcomes."""
import json

import pytest

from tests.test_scene_retirement_real_participants import access_fixture
from tests.test_scene_retirement_preservation import MemoryTransport


def setup_operation(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members,ActionAllowance
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
    access,policy,member=access_fixture(tmp_path,monkeypatch)
    (member/'nested').mkdir()
    (member/'nested'/'evidence.bin').write_bytes(b'preserved-evidence')
    allowance=ActionAllowance(expires_at=200,now=lambda:100,monotonic=lambda:0)
    preserved=preserve_members([member],transport=MemoryTransport([member]),allowance=allowance,token='1'*32)
    store=tmp_path/'journals'
    store.mkdir(mode=0o700)
    journal=SceneJournal.create(store,token='1'*32,initial={'status':'pending','preservation':preserved},allowance=allowance)
    return access,member,preserved,journal


def test_actual_member_detaches_only_after_durable_exact_intent_then_removes_known_union(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    access,member,preserved,journal=setup_operation(tmp_path,monkeypatch)
    from blueprint_pipeline import control_plane_lane_scratch as primitive
    original=primitive._publish_no_replace
    observed=[]
    def rename(parent,source,destination):
        events=[json.loads(path.read_text()) for path in journal.directory.glob('*.json')]
        planned=[event for event in events if event.get('event')=='detach_planned']
        assert len(planned)==1 and planned[0]['evidence']['detached_path']==str(member.parent/destination)
        assert (member/'nested'/'evidence.bin').read_bytes()==b'preserved-evidence'
        observed.append(destination)
        return original(parent,source,destination)
    monkeypatch.setattr(primitive,'_publish_no_replace',rename)
    with access.exclusive_scene_access():
        outcome=detach_and_remove(preserved,member_index=0,generation_id='2'*32,journal=journal)
    assert observed and outcome['outcome']=='removed' and not member.exists()
    assert not (member.parent/observed[0]).exists()
    events=[json.loads(path.read_text()) for path in journal.directory.glob('*.json')]
    assert [event['event'] for event in sorted([v for v in events if 'event' in v],key=lambda v:v['sequence'])] == ['detach_planned','detached','member_removed']


def test_new_detach_destination_is_never_overwritten(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    access,member,preserved,journal=setup_operation(tmp_path,monkeypatch)
    destination=member.parent/('.scene-retirement-'+'1'*32+'-0')
    destination.mkdir()
    (destination/'new.bin').write_bytes(b'new-writer')
    with access.exclusive_scene_access(),pytest.raises(FileExistsError):
        detach_and_remove(preserved,member_index=0,generation_id='2'*32,journal=journal)
    assert (destination/'new.bin').read_bytes()==b'new-writer'
    assert (member/'nested'/'evidence.bin').read_bytes()==b'preserved-evidence'


def test_proved_internal_cross_member_hardlink_union_removes_completely(tmp_path,monkeypatch):
    import os
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members
    from tests.test_scene_retirement_preservation import MemoryTransport
    access,first,_,journal=setup_operation(tmp_path,monkeypatch)
    second=first.parent/'second'
    second.mkdir()
    os.link(first/'nested'/'evidence.bin',second/'same-evidence.bin')
    preserved=preserve_members([first,second],transport=MemoryTransport([first,second]),
                              allowance=journal.allowance,token='3'*32)
    removed={}
    with access.exclusive_scene_access():
        detach_and_remove(preserved,member_index=0,generation_id='2'*32,journal=journal,removed_inodes=removed)
        outcome=detach_and_remove(preserved,member_index=1,generation_id='4'*32,journal=journal,removed_inodes=removed)
    assert outcome['outcome']=='removed' and not first.exists() and not second.exists()
