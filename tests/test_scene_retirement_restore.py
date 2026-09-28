"""Restoration preserves exact bytes and never overwrites a replacement writer."""
import os

import pytest

from tests.test_scene_retirement_member_mutation import setup_operation


def test_full_preserved_member_restores_bytes_hardlinks_and_mode_without_overwrite(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_preserved_members
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members
    from tests.test_scene_retirement_preservation import MemoryTransport
    access,member,_,journal=setup_operation(tmp_path,monkeypatch)
    os.link(member/'nested'/'evidence.bin',member/'linked.bin')
    transport=MemoryTransport([member])
    preserved=preserve_members([member],transport=transport,allowance=journal.allowance,token='3'*32)
    with access.exclusive_scene_access():
        detach_and_remove(preserved,member_index=0,generation_id='2'*32,journal=journal)
        transport.members=[]  # Fake's upload/readback pre-removal assertion has finished.
        result=restore_preserved_members(preserved,transport=transport,journal=journal)
    assert result[0]['outcome']=='restored'
    assert (member/'nested'/'evidence.bin').read_bytes()==b'preserved-evidence'
    assert (member/'linked.bin').stat().st_ino==(member/'nested'/'evidence.bin').stat().st_ino
    assert (member.stat().st_mode & 0o777)==preserved['members'][0]['mode']
    assert (member.stat().st_uid,member.stat().st_gid)==(preserved['members'][0]['uid'],preserved['members'][0]['gid'])
    expected=next(row for row in preserved['files'] if row['relative_path']=='nested/evidence.bin')
    info=(member/'nested'/'evidence.bin').stat()
    assert (info.st_uid,info.st_gid,info.st_mode & 0o777)==(expected['uid'],expected['gid'],expected['mode'])


def test_restore_conflict_keeps_new_writer_and_corrupt_remote_creates_nothing(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_preserved_members
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members
    from tests.test_scene_retirement_preservation import MemoryTransport
    access,member,_,journal=setup_operation(tmp_path,monkeypatch)
    transport=MemoryTransport([member])
    preserved=preserve_members([member],transport=transport,allowance=journal.allowance,token='3'*32)
    with access.exclusive_scene_access():
        detach_and_remove(preserved,member_index=0,generation_id='2'*32,journal=journal)
        transport.members=[]
        transport.corrupt=True
        with pytest.raises(ValueError,match='scene_retirement_readback_unproven'):
            restore_preserved_members(preserved,transport=transport,journal=journal)
        assert not member.exists()
        transport.corrupt=False
        member.mkdir()
        (member/'new.bin').write_bytes(b'new-writer')
        with pytest.raises(FileExistsError):
            restore_preserved_members(preserved,transport=transport,journal=journal)
    assert list(member.iterdir())==[member/'new.bin']
    assert (member/'new.bin').read_bytes()==b'new-writer'


@pytest.mark.parametrize('drift',['payload','extra-entry','external-hardlink'])
def test_restored_member_requires_full_current_byte_and_exclusive_inode_proof_before_completion(tmp_path,monkeypatch,drift):
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_preserved_members
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    from tests.test_scene_retirement_preservation import MemoryTransport
    access,member,preserved,journal=setup_operation(tmp_path,monkeypatch)
    transport=MemoryTransport([member])
    # Recover the exact archive created by setup, never manufacture preservation.
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members
    preserved=preserve_members([member],transport=transport,allowance=journal.allowance,token='3'*32)
    with access.exclusive_scene_access():
        detach_and_remove(preserved,member_index=0,generation_id='2'*32,journal=journal)
        transport.members=[]
        append=journal.append
        def inject_after_published_file(event,**kwargs):
            reference=append(event,**kwargs)
            if event=='restore_file_created':
                path=member/kwargs['evidence']['relative_path']
                if drift=='payload':
                    path.write_bytes(b'contradictory-payload')
                elif drift=='extra-entry':
                    (member/'foreign.bin').write_bytes(b'new-writer')
                else:
                    os.link(path,member.parent/'foreign-linked.bin')
            return reference
        monkeypatch.setattr(journal,'append',inject_after_published_file)
        with pytest.raises(ValueError):
            restore_preserved_members(preserved,transport=transport,journal=journal)
    assert not any(row['event']=='member_restored' for row in journal.events)


@pytest.mark.parametrize('limit',['events','bytes'])
def test_complete_restore_journal_capacity_refuses_before_creating_first_directory(tmp_path,monkeypatch,limit):
    from blueprint_pipeline import task_evaluation_scene_retirement_journal as journal_module
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_preserved_members
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members
    from tests.test_scene_retirement_preservation import MemoryTransport
    access,member,_,journal=setup_operation(tmp_path,monkeypatch)
    transport=MemoryTransport([member])
    preserved=preserve_members([member],transport=transport,allowance=journal.allowance,token='3'*32)
    with access.exclusive_scene_access():
        detach_and_remove(preserved,member_index=0,generation_id='2'*32,journal=journal)
        transport.members=[]
        if limit=='events':
            monkeypatch.setattr(journal_module,'MAX_EVENTS',journal.sequence+4)
        else:
            monkeypatch.setattr(journal_module,'MAX_JOURNAL_BYTES',journal.bytes+1)
        with pytest.raises(ValueError,match='scene_retirement_journal_limit'):
            restore_preserved_members(preserved,transport=transport,journal=journal)
    assert not member.exists()
