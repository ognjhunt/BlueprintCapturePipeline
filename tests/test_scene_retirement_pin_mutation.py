# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_pin_mutation.py
#   src/blueprint_pipeline/task_evaluation_scene_retirement_pins.py
"""Real native pin bytes, private journal, exact CAS and replay; no action grant."""
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.test_scene_retirement_terminal_pins import fixture,selected


def operation(tmp_path,monkeypatch,*,include_fresh=False):
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
    fresh,policy,consent,allowance=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(access,'_POLICY_UID',os.geteuid())
    rows=selected(fresh,policy,consent,allowance)['terminal_pin_release_rows']
    directory=tmp_path/'journal'
    directory.mkdir(mode=0o700)
    initial={'schema_version':'scene_retirement_journal.v1','status':'pending',
             'intent_id':consent['intent_id'],'intent_raw_ref':consent['intent_raw_ref'],
             'terminal_pin_release_rows':rows}
    journal=SceneJournal.create(directory,token='a'*32,initial=initial,allowance=allowance)
    value={'schema_version':'scene_lifecycle_retirement_receipt.v1','status':'pending',
           'intent_id':consent['intent_id'],'journal_initial_raw_ref':journal.initial_ref}
    raw=json.dumps(value,sort_keys=True).encode()
    pending=tmp_path/'pending.json'
    pending.write_bytes(raw)
    reference={'path':str(pending),'sha256':'sha256:'+hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}
    result=(policy,consent,rows,journal,reference)
    return result+(fresh,) if include_fresh else result


def api():
    from blueprint_pipeline import task_evaluation_scene_retirement_pin_mutation as module
    return module


def test_real_pin_release_and_restore_preserve_original_bytes_owner_mode_and_chain(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
    policy,consent,rows,journal,pending=operation(tmp_path,monkeypatch)
    path=Path(rows[0]['original_raw_ref']['path'])
    original=path.read_bytes()
    before=path.stat()
    outcomes=api().release_terminal_pins(policy,consent,rows,journal=journal,pending_raw_ref=pending)
    assert json.loads(path.read_bytes())==dict(json.loads(original),released_at_epoch=30000)
    assert [e['event'] for e in journal.events]==['pin_release_planned','pin_released']
    assert outcomes[0]['released_raw_ref']['sha256']=='sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.stat().st_mode&0o777==0o640 and (path.stat().st_uid,path.stat().st_gid)==(before.st_uid,before.st_gid)
    # Exact repeat and native resumed chain must not mint another pin version.
    resumed=SceneJournal.resume(journal.initial_ref,allowance=journal.allowance)
    assert api().release_terminal_pins(policy,consent,rows,journal=resumed,pending_raw_ref=pending)==outcomes
    restore=SceneJournal.create(journal.directory,token='b'*32,
        initial={'schema_version':'scene_restore_journal.v1','original_retirement_token':journal.token,
                 'terminal_pin_release_rows':rows,'terminal_pin_outcomes':outcomes},allowance=journal.allowance)
    restored=api().restore_terminal_pins(policy,rows,outcomes,journal=restore)
    assert path.read_bytes()==original
    assert restored[0]['restored_raw_ref']==rows[0]['original_raw_ref']
    assert [e['event'] for e in restore.events]==['pin_restore_planned','pin_restored']
    assert api().restore_terminal_pins(policy,rows,outcomes,journal=restore)==restored


@pytest.mark.parametrize('phase',['after_plan','after_replace'])
def test_release_crash_reselects_only_exact_prejournaled_temporary_inode(tmp_path,monkeypatch,phase):
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
    policy,consent,rows,journal,pending=operation(tmp_path,monkeypatch)
    native=journal.append
    def interrupted(event,**kwargs):
        if phase=='after_replace' and event=='pin_released':
            raise OSError('test crash')
        result=native(event,**kwargs)
        if phase=='after_plan' and event=='pin_release_planned':
            raise OSError('test crash')
        return result
    monkeypatch.setattr(journal,'append',interrupted)
    with pytest.raises(OSError,match='test crash'):
        api().release_terminal_pins(policy,consent,rows,journal=journal,pending_raw_ref=pending)
    resumed=SceneJournal.resume(journal.initial_ref,allowance=journal.allowance)
    result=api().release_terminal_pins(policy,consent,rows,journal=resumed,pending_raw_ref=pending)
    assert len(result)==1 and json.loads(Path(rows[0]['original_raw_ref']['path']).read_bytes())['released_at_epoch']==30000
    assert [e['event'] for e in resumed.events]==['pin_release_planned','pin_released']


@pytest.mark.parametrize('change',['pending_missing','initial_changed','pin_changed','pin_symlink','restore_foreign'])
def test_unproven_pin_versions_never_replace_current_bytes(tmp_path,monkeypatch,change):
    policy,consent,rows,journal,pending=operation(tmp_path,monkeypatch)
    path=Path(rows[0]['original_raw_ref']['path'])
    if change=='pending_missing':
        Path(pending['path']).unlink()
    elif change=='initial_changed':
        rows=copy.deepcopy(rows)
        rows[0]['original_value']['owner_id']='foreign'
    elif change=='pin_changed':
        path.write_bytes(b'foreign')
    elif change=='pin_symlink':
        other=tmp_path/'foreign'
        other.write_bytes(b'foreign')
        path.unlink()
        path.symlink_to(other)
    else:
        from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
        outcomes=api().release_terminal_pins(policy,consent,rows,journal=journal,pending_raw_ref=pending)
        path.write_bytes(b'foreign')
        restore=SceneJournal.create(journal.directory,token='b'*32,
            initial={'schema_version':'scene_restore_journal.v1','original_retirement_token':journal.token,
                     'terminal_pin_release_rows':rows,'terminal_pin_outcomes':outcomes},allowance=journal.allowance)
        with pytest.raises((ValueError,OSError),match='scene_retirement_'):
            api().restore_terminal_pins(policy,rows,outcomes,journal=restore)
        assert path.read_bytes()==b'foreign'
        return
    before=path.read_bytes()
    with pytest.raises((ValueError,OSError)):
        api().release_terminal_pins(policy,consent,rows,journal=journal,pending_raw_ref=pending)
    assert path.read_bytes()==before and not journal.events


def test_real_partial_release_reobserves_exact_journal_version_without_ignoring_other_pin(tmp_path,monkeypatch):
    from dataclasses import asdict
    from blueprint_pipeline.control_plane_storage_pin_observation import observe_storage_pins
    from blueprint_pipeline.task_evaluation_scene_retirement_pins import covers
    from blueprint_pipeline.task_evaluation_scene_retirement_reference_transfer import validate_current_reference_transfer
    policy,consent,rows,journal,pending,fresh=operation(tmp_path,monkeypatch,include_fresh=True)
    api().release_terminal_pins(policy,consent,rows,journal=journal,pending_raw_ref=pending)
    history=api().pin_history(policy,consent,journal)
    observed=observe_storage_pins(policy['reference_context']['pins_root'],observed_at_epoch=30000,monotonic=lambda:0)
    protection={'kind':'pin_observation','observation':json.loads(json.dumps(asdict(observed.rows[0]))),'action':'KEEP'}
    assert covers(protection,history.values())
    fresh['reference_observation']['protections']=[row for row in fresh['reference_observation']['protections']
        if row.get('kind') not in {'pin_observation','positive_pin_path'}]+[protection]
    replay=validate_current_reference_transfer(fresh,journal.allowance,policy=policy,consent=consent,pin_journal=journal)
    assert replay['terminal_pin_release_rows'][0]['original_raw_ref']==rows[0]['original_raw_ref']
    assert replay['terminal_pin_release_rows'][0]['observed_raw_ref']==history[tuple(
        rows[0]['original_raw_ref'][key] for key in ('path','sha256','size_bytes'))]['observed_raw_ref']
    # This exact row mapping does not cover a later foreign owner/version.
    foreign=copy.deepcopy(protection)
    foreign['observation']['raw_sha256']='sha256:'+'f'*64
    assert not covers(foreign,history.values())
    assert len(history)==1 and journal.sequence==2
