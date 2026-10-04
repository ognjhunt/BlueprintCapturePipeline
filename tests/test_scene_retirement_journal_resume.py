"""Replay only an exact protected operation chain, never an orphan basename."""
import json
from pathlib import Path
import pytest
from tests.test_scene_retirement_member_mutation import setup_operation


def test_protected_chain_reopens_exact_cursor_and_continues_without_reset(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
    _,_,_,journal=setup_operation(tmp_path,monkeypatch)
    first=journal.append('retiring',member_key='0',evidence={'generation_id':'2'*32})
    second=journal.append('detach_planned',member_key='0',evidence={'detached_path':'/exact/proved-token'})
    resumed=SceneJournal.resume(journal.initial_ref,allowance=journal.allowance)
    assert resumed.token==journal.token and resumed.sequence==2 and resumed.prior_ref==second
    assert [row['event'] for row in resumed.events]==['retiring','detach_planned']
    assert resumed.events[0]['raw_ref']==first
    following=resumed.append('detached',member_key='0',evidence={'detach_plan_raw_ref':second})
    value=json.loads(Path(following['path']).read_bytes())
    assert value['sequence']==3 and value['prior_event_sha256']==second['sha256']
    assert resumed.bytes==journal.bytes+following['size_bytes']
    assert resumed.allowance is journal.allowance


@pytest.mark.parametrize('fault',['missing-middle','changed-event','unknown-same-token-entry'])
def test_chain_refuses_gaps_substitution_or_unknown_operation_entries(tmp_path,monkeypatch,fault):
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
    _,member,_,journal=setup_operation(tmp_path,monkeypatch)
    first=journal.append('retiring',member_key='0',evidence={'generation_id':'2'*32})
    journal.append('detach_planned',member_key='0',evidence={'canonical_path':str(member)})
    if fault=='missing-middle':
        Path(first['path']).unlink()
    elif fault=='changed-event':
        Path(first['path']).write_text('{}')
    else:
        (journal.directory/(journal.token+'.orphan.json')).write_text('{}')
    with pytest.raises(ValueError):
        SceneJournal.resume(journal.initial_ref,allowance=journal.allowance)
    assert member.exists()
