"""ADP-009D/day28: delayed final inode guards cannot authorize expired effects.

Actual tiny member/journal syscalls, with explicit CPU clock models. This does
not establish installed Linux reader absence or a completed website producer.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_mutation.py
#   src/blueprint_pipeline/task_evaluation_scene_retirement_generations.py
#   src/blueprint_pipeline/task_evaluation_scene_retirement_restore.py
#   src/blueprint_pipeline/task_evaluation_scene_retirement_journal.py
#   src/blueprint_pipeline/task_evaluation_scene_retirement_metadata.py
#   src/blueprint_pipeline/task_evaluation_scene_retirement_pin_mutation.py
#   src/blueprint_pipeline/task_evaluation_scene_retirement_intent_receipt.py

import os
import sys
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from tests.test_scene_retirement_member_mutation import setup_operation


@pytest.mark.parametrize('effect', ['detach', 'unlink', 'rmdir'])
@pytest.mark.parametrize('clock_kind', ['wall', 'monotonic'])
def test_final_named_guard_expiry_preserves_next_actual_member_effect(
        tmp_path, monkeypatch, effect, clock_kind):
    from blueprint_pipeline import control_plane_lane_scratch as primitive
    from blueprint_pipeline import task_evaluation_scene_retirement_mutation as mutation
    access, member, preserved, journal = setup_operation(tmp_path, monkeypatch)
    wall, mono = [100], [0]
    journal.allowance.now = lambda: wall[0]
    journal.allowance.monotonic = lambda: mono[0]
    selected_name = {'detach': member.name, 'unlink': 'evidence.bin', 'rmdir': 'nested'}[effect]
    planned = {'detach': 'detach_planned', 'unlink': 'leaf_unlink_planned',
               'rmdir': 'directory_unlink_planned'}[effect]
    caller_name = '_remove_directory' if effect == 'rmdir' else 'detach_and_remove'
    expired = []
    actual_stat = os.stat
    def delayed_stat(path, *args, **kwargs):
        observed = actual_stat(path, *args, **kwargs)
        if (str(path) == selected_name and sys._getframe(1).f_code.co_name == caller_name
                and any(event['event'] == planned for event in journal.events)):
            (wall if clock_kind == 'wall' else mono)[0] = 201 if clock_kind == 'wall' else 1801
        return observed
    monkeypatch.setattr(os, 'stat', delayed_stat)
    def after_expiry():
        return wall[0] >= 200 or mono[0] > journal.allowance.elapsed_seconds
    actual_detach, actual_unlink, actual_rmdir = primitive._publish_no_replace, os.unlink, os.rmdir
    def detach(*args, **kwargs):
        if effect == 'detach' and after_expiry():
            expired.append('detach')
        return actual_detach(*args, **kwargs)
    def unlink(path, *args, **kwargs):
        if effect == 'unlink' and str(path) == selected_name and after_expiry():
            expired.append('unlink')
        return actual_unlink(path, *args, **kwargs)
    def rmdir(path, *args, **kwargs):
        if effect == 'rmdir' and str(path) == selected_name and after_expiry():
            expired.append('rmdir')
        return actual_rmdir(path, *args, **kwargs)
    monkeypatch.setattr(primitive, '_publish_no_replace', detach)
    monkeypatch.setattr(os, 'unlink', unlink)
    monkeypatch.setattr(os, 'rmdir', rmdir)
    reason = 'scene_retirement_consent_expired' if clock_kind == 'wall' else 'scene_retirement_deadline'
    with access.exclusive_scene_access(), pytest.raises(ValueError, match=reason):
        mutation.detach_and_remove(preserved, member_index=0, generation_id='2'*32, journal=journal)
    assert after_expiry() and expired == []
    detached = member.parent / ('.scene-retirement-' + journal.token + '-0')
    if effect == 'detach':
        assert (member / 'nested/evidence.bin').read_bytes() == b'preserved-evidence'
        assert not detached.exists()
    elif effect == 'unlink':
        assert (detached / 'nested/evidence.bin').read_bytes() == b'preserved-evidence'
    else:
        assert (detached / 'nested').is_dir()


@pytest.mark.parametrize('preparation', ['library', 'arguments'])
@pytest.mark.parametrize('clock_kind', ['wall', 'monotonic'])
def test_native_detach_preparation_cannot_cross_original_expiry(
        tmp_path, monkeypatch, preparation, clock_kind):
    from blueprint_pipeline import control_plane_lane_scratch as primitive
    from blueprint_pipeline import task_evaluation_scene_retirement_mutation as mutation
    access, member, preserved, journal = setup_operation(tmp_path, monkeypatch)
    wall, mono = [100], [0]
    journal.allowance.now = lambda: wall[0]
    journal.allowance.monotonic = lambda: mono[0]
    reached = []
    def expire():
        reached.append(preparation)
        (wall if clock_kind == 'wall' else mono)[0] = 201 if clock_kind == 'wall' else 1801
    actual_library, actual_encode = primitive.ctypes.CDLL, os.fsencode
    def library(*args, **kwargs):
        result = actual_library(*args, **kwargs)
        if preparation == 'library':
            expire()
        return result
    def encode(value):
        result = actual_encode(value)
        if preparation == 'arguments' and str(value).startswith('.scene-retirement-'):
            expire()
        return result
    monkeypatch.setattr(primitive.ctypes, 'CDLL', library)
    monkeypatch.setattr(os, 'fsencode', encode)
    reason = 'scene_retirement_consent_expired' if clock_kind == 'wall' else 'scene_retirement_deadline'
    with access.exclusive_scene_access(), pytest.raises(ValueError, match=reason):
        mutation.detach_and_remove(preserved, member_index=0, generation_id='2'*32, journal=journal)
    assert reached == [preparation]
    assert (member / 'nested/evidence.bin').read_bytes() == b'preserved-evidence'
    assert not (member.parent / ('.scene-retirement-' + journal.token + '-0')).exists()


@pytest.mark.parametrize('replacement', [False, True])
@pytest.mark.parametrize('clock_kind', ['wall', 'monotonic'])
def test_generation_transition_cannot_publish_after_its_final_named_guard(
        tmp_path, monkeypatch, replacement, clock_kind):
    from tests.test_scene_retirement_public_resume import fresh_action
    from blueprint_pipeline import task_evaluation_scene_retirement_generations as generations
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    engine, policy, scope, _, _ = fresh_action(tmp_path, monkeypatch)
    path = Path(policy['generation_store']) / (hashlib.sha256(
        scope['members'][0]['canonical_path'].encode()).hexdigest() + '.json')
    original = path.read_bytes()
    prior = json.loads(original)
    wall, mono = [100], [0]
    allowance = ActionAllowance(expires_at=200, now=lambda: wall[0], monotonic=lambda: mono[0])
    actual_named, actual_fsync = generations._named, os.fsync
    synced = set()
    def fsync(fd):
        result = actual_fsync(fd)
        info = os.fstat(fd)
        synced.add((info.st_dev, info.st_ino))
        return result
    def named(*args, **kwargs):
        result = actual_named(*args, **kwargs)
        caller = sys._getframe(1)
        publisher = caller.f_back
        if (caller.f_code.co_name == 'named' and publisher.f_code.co_name == '_write'
                and publisher.f_locals['replace'] == replacement and tuple(args[4][:2]) in synced):
            (wall if clock_kind == 'wall' else mono)[0] = 201 if clock_kind == 'wall' else 1801
        return result
    monkeypatch.setattr(os, 'fsync', fsync)
    monkeypatch.setattr(generations, '_named', named)
    kwargs = {'allowance': allowance} if 'allowance' in inspect.signature(engine._transition).parameters else {}
    reason = 'scene_retirement_consent_expired' if clock_kind == 'wall' else 'scene_retirement_deadline'
    with pytest.raises(ValueError, match=reason):
        engine._transition(policy, prior, state='retiring', token='3'*32,
                           journal_ref={'sha256': 'sha256:'+'a'*64}, **kwargs)
    assert (wall[0] >= 200 or mono[0] > allowance.elapsed_seconds)
    assert path.read_bytes() == original
    if not replacement:
        assert list(path.parent.glob(path.stem + '.*.retiring.json')) == []


@pytest.mark.parametrize('effect', ['root_mkdir', 'child_mkdir', 'write', 'file_chown',
                                  'file_chmod', 'file_fsync', 'publish', 'temp_unlink',
                                  'directory_chown', 'directory_chmod', 'directory_fsync'])
@pytest.mark.parametrize('clock_kind', ['wall', 'monotonic'])
def test_restore_final_guard_cannot_mutate_after_original_expiry(
        tmp_path, monkeypatch, effect, clock_kind):
    from blueprint_pipeline import task_evaluation_scene_retirement_restore as restoration
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members
    from tests.test_scene_retirement_preservation import MemoryTransport
    access, member, _, journal = setup_operation(tmp_path, monkeypatch)
    transport = MemoryTransport([member])
    preserved = preserve_members([member], transport=transport, allowance=journal.allowance, token='3'*32)
    with access.exclusive_scene_access():
        detach_and_remove(preserved, member_index=0, generation_id='2'*32, journal=journal)
    transport.members = []
    wall, mono = [100], [0]
    journal.allowance.now = lambda: wall[0]
    journal.allowance.monotonic = lambda: mono[0]
    reached, expired_effects, phase = [], [], ['empty']
    def expire():
        reached.append(effect)
        (wall if clock_kind == 'wall' else mono)[0] = 201 if clock_kind == 'wall' else 1801
    actual_parent, actual_named = restoration._current_parent, restoration._named
    def parent(path, *args, **kwargs):
        result = actual_parent(path, *args, **kwargs)
        if sys._getframe(1).f_code.co_name == 'restore_preserved_members':
            if effect == 'root_mkdir' and path == member.parent and not member.exists():
                expire()
            elif effect == 'child_mkdir' and path == member and not (member / 'nested').exists():
                expire()
        return result
    selected_phase = {'write': 'empty', 'file_chown': 'written', 'file_chmod': 'chowned',
                      'file_fsync': 'chmodded', 'publish': 'synced', 'temp_unlink': 'linked',
                      'directory_chown': 'linked', 'directory_chmod': 'directory_chowned',
                      'directory_fsync': 'directory_chmodded'}.get(effect)
    def named(parent, parent_identity, name, fd, identity):
        result = actual_named(parent, parent_identity, name, fd, identity)
        if phase[0] == selected_phase:
            if effect.startswith('directory_') and name == 'nested':
                expire()
            elif not effect.startswith('directory_') and str(name).endswith('.restore'):
                expire()
        return result
    monkeypatch.setattr(restoration, '_current_parent', parent)
    monkeypatch.setattr(restoration, '_named', named)
    def crossed():
        return wall[0] >= 200 or mono[0] > journal.allowance.elapsed_seconds
    transitions = {'write': 'written', 'fchown': 'chowned', 'fchmod': 'chmodded', 'fsync': 'synced'}
    for name in ('mkdir', 'write', 'fchown', 'fchmod', 'fsync', 'link', 'unlink'):
        original = getattr(os, name)
        def syscall(*args, _name=name, _original=original, **kwargs):
            if crossed() and _name != 'unlink':  # Exact owned temporary cleanup may still run.
                expired_effects.append(_name)
            elif crossed() and _name == 'unlink' and phase[0] == 'linked':
                # Ordinary post-publication unlink is an effect, not failure cleanup.
                if sys._getframe(1).f_code.co_name == 'file_sink' and sys.exc_info()[0] is None:
                    expired_effects.append(_name)
            result = _original(*args, **kwargs)
            caller = sys._getframe(1).f_code.co_name
            if _name in transitions:
                info = os.fstat(args[0])
                if info.st_mode & 0o170000 == 0o100000 and caller in {'file_sink', 'write'}:
                    phase[0] = transitions[_name]
                elif (caller == 'restore_preserved_members' and _name in ('fchown', 'fchmod')
                      and (member / 'nested').exists() and info.st_ino == (member / 'nested').stat().st_ino):
                    phase[0] = 'directory_chowned' if _name == 'fchown' else 'directory_chmodded'
            elif _name == 'link' and caller == 'file_sink':
                phase[0] = 'linked'
            return result
        monkeypatch.setattr(os, name, syscall)
    reason = 'scene_retirement_consent_expired' if clock_kind == 'wall' else 'scene_retirement_deadline'
    with access.exclusive_scene_access(), pytest.raises(ValueError, match=reason):
        restoration.restore_preserved_members(preserved, transport=transport, journal=journal)
    assert reached and expired_effects == []


@pytest.mark.parametrize('publisher', ['journal', 'metadata', 'pin', 'intent'])
@pytest.mark.parametrize('clock_kind', ['wall', 'monotonic'])
def test_private_publisher_last_guard_expiry_prevents_its_next_effect(
        tmp_path, monkeypatch, publisher, clock_kind):
    import importlib
    names = {'journal': 'journal', 'metadata': 'metadata', 'pin': 'pin_mutation',
             'intent': 'intent_receipt'}
    module = importlib.import_module('blueprint_pipeline.task_evaluation_scene_retirement_' + names[publisher])
    _, _, _, journal = setup_operation(tmp_path, monkeypatch)
    store = tmp_path / 'protected-publisher'
    store.mkdir(mode=0o700 if publisher == 'journal' else 0o750)
    raw = b'{"development_only":true}'
    pin = store / 'original.json'
    pin.write_bytes(raw)
    pin.chmod(0o640)
    original_bytes = pin.read_bytes()
    wall, mono = [100], [0]
    journal.allowance.now = lambda: wall[0]
    journal.allowance.monotonic = lambda: mono[0]
    actual_named = module._named
    reached, expired = [], []
    def named(*args, **kwargs):
        result = actual_named(*args, **kwargs)
        reached.append(args[2])
        (wall if clock_kind == 'wall' else mono)[0] = 201 if clock_kind == 'wall' else 1801
        return result
    monkeypatch.setattr(module, '_named', named)
    for name in ('write', 'fchown', 'fchmod', 'fsync', 'link', 'replace'):
        original = getattr(os, name)
        def effect(*args, _name=name, _original=original, **kwargs):
            if wall[0] >= 200 or mono[0] > journal.allowance.elapsed_seconds:
                expired.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(os, name, effect)
    reason = 'scene_retirement_consent_expired' if clock_kind == 'wall' else 'scene_retirement_deadline'
    with pytest.raises(ValueError, match=reason):
        if publisher == 'journal':
            module.publish_record(store, 'new.json', {'development_only': True},
                                  maximum=65536, allowance=journal.allowance)
        elif publisher == 'metadata':
            module._publish_raw(store, 'new.json', raw, journal.allowance)
        elif publisher == 'pin':
            module._prepare(pin, b'{"development_only":true,"restored":true}',
                            module._snapshot(pin.stat()), journal.allowance)
        else:
            module._publish(store, 'new.json', raw, journal.allowance)
    assert reached and expired == []
    assert pin.read_bytes() == original_bytes and not (store / 'new.json').exists()
