"""Resume exact native-created destinations without adopting a new writer."""
import os

import pytest

from tests.test_scene_retirement_member_mutation import setup_operation


def interrupted_restore(tmp_path, monkeypatch, phase):
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_preserved_members
    from tests.test_scene_retirement_preservation import MemoryTransport
    access, member, _, journal = setup_operation(tmp_path, monkeypatch)
    os.link(member / 'nested' / 'evidence.bin', member / 'linked.bin')
    transport = MemoryTransport([member])
    preserved = preserve_members([member], transport=transport, allowance=journal.allowance, token='3' * 32)
    original = journal.append
    def interrupt(event, **kwargs):
        reference = original(event, **kwargs)
        if event == phase:
            raise OSError('owned interruption after durable native creation')
        return reference
    with access.exclusive_scene_access():
        detach_and_remove(preserved, member_index=0, generation_id='2' * 32, journal=journal)
        transport.members = []
        monkeypatch.setattr(journal, 'append', interrupt)
        with pytest.raises(OSError):
            restore_preserved_members(preserved, transport=transport, journal=journal)
    monkeypatch.setattr(journal, 'append', original)
    return access, member, preserved, transport, journal


@pytest.mark.parametrize('phase', ['restore_directory_created', 'restore_file_created', 'member_restored'])
def test_exact_native_creation_journal_resumes_full_bytes_and_hardlink_union(tmp_path, monkeypatch, phase):
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_preserved_members
    access, member, preserved, transport, journal = interrupted_restore(tmp_path, monkeypatch, phase)
    with access.exclusive_scene_access():
        result = restore_preserved_members(preserved, transport=transport, journal=journal)
    assert result[0]['outcome'] == 'restored'
    assert (member / 'nested' / 'evidence.bin').read_bytes() == b'preserved-evidence'
    assert (member / 'linked.bin').stat().st_ino == (member / 'nested' / 'evidence.bin').stat().st_ino
    assert len([event for event in journal.events if event['event'] == 'member_restored']) == 1


def test_resume_does_not_adopt_substituted_destination_inode(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_preserved_members
    access, member, preserved, transport, journal = interrupted_restore(tmp_path, monkeypatch, 'restore_directory_created')
    member.rename(member.parent / 'owned-old-restore')
    member.mkdir()
    (member / 'new-capture.bin').write_bytes(b'foreign')
    with access.exclusive_scene_access(), pytest.raises(ValueError):
        restore_preserved_members(preserved, transport=transport, journal=journal)
    assert (member / 'new-capture.bin').read_bytes() == b'foreign'


def test_resume_cannot_hide_unjournaled_entry_in_proved_root(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_preserved_members
    access, member, preserved, transport, journal = interrupted_restore(tmp_path, monkeypatch, 'restore_directory_created')
    (member / 'unknown.bin').write_bytes(b'unjournaled')
    with access.exclusive_scene_access(), pytest.raises(ValueError):
        restore_preserved_members(preserved, transport=transport, journal=journal)
    assert (member / 'unknown.bin').read_bytes() == b'unjournaled'
