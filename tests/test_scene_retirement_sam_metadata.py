"""Exact private SAM metadata remains readable without widening its audience."""
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_retirement_real_participants import access_fixture


CASES = [
    ('completed/sam31-'+'a'*64+'.json', 'task_evaluation_sam31_preparation_execution_job.v1', 'job_digest'),
    ('results/sam31-'+'a'*64+'.json', 'task_evaluation_sam31_preparation_execution_result.v1', 'result_digest'),
    ('a'*64+'/sam31-'+'b'*64+'/phase_execution_receipt.v1.json', 'task_evaluation_sam31_phase_execution_receipt.v1', 'receipt_digest'),
    ('sam31-prefix-'+'a'*64+'.json', 'task_evaluation_sam31_completed_prefix_adoption.v1', 'adoption_digest'),
]


def publish(tmp_path,monkeypatch,records):
    from blueprint_pipeline.task_evaluation_scene_retirement_metadata import retain_metadata_closure
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance,_scan,_payload
    access,policy,member=access_fixture(tmp_path,monkeypatch)
    member.chmod(0o700)
    store=Path(policy['journal_store']+'.metadata')
    store.mkdir(mode=0o750)
    raw_rows={}
    for relative,value in records:
        source=member/relative
        source.parent.mkdir(parents=True,exist_ok=True)
        for parent in (source.parent,*source.parent.parents):
            if parent.is_relative_to(member):
                parent.chmod(0o700)
        raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
        source.write_bytes(raw)
        source.chmod(0o600)
        raw_rows[source]=raw
    allowance=ActionAllowance(expires_at=1000,now=lambda:200)
    files,directories=[],[]
    root=_scan(member,0,allowance,files,directories)
    for row in files:
        row['sha256']='sha256:'+hashlib.sha256(b''.join(_payload(member/row['relative_path'],row,allowance))).hexdigest()
    token='2'*32
    closure=retain_metadata_closure(dict(files=files,directories=directories,members=[root]),policy,
        [dict(generation_id='1'*32)],token,allowance)
    info=member.stat()
    generation=dict(schema_version='scene_member_generation.v1',canonical_path=str(member),
        state='retired',generation_id='1'*32,retirement_token=token,dev=info.st_dev,ino=info.st_ino,mode=info.st_mode)
    generation['state_digest']=canonical_digest(generation,digest_field='state_digest')
    entry=Path(policy['generation_store'])/(hashlib.sha256(str(member).encode()).hexdigest()+'.json')
    entry.write_text(json.dumps(generation))
    entry.chmod(0o600)
    shutil.rmtree(member)  # Only this owned tiny fixture, after preservation.
    return raw_rows,json.loads(Path(closure['path']).read_bytes()),member


@pytest.mark.parametrize('relative,schema,seal',CASES)
def test_real_sam_readers_reopen_private_metadata_with_original_permissions(tmp_path,monkeypatch,relative,schema,seal):
    value=dict(schema_version=schema,status='completed')
    value[seal]=canonical_digest(value,digest_field=seal)
    raws,closure,_=publish(tmp_path,monkeypatch,[(relative,value)])
    logical,raw=next(iter(raws.items()))
    assert len(closure['rows'])==1
    row=closure['rows'][0]
    clone=Path(row['retained_raw_ref']['path'])
    assert clone.read_bytes()==raw
    assert (clone.stat().st_uid,clone.stat().st_gid,clone.stat().st_mode & 0o777)==(os.getuid(),os.getgid(),0o600)
    if 'execution_' in schema:
        from blueprint_pipeline.task_evaluation_sam31_phase_queue import _read
        assert _read(logical)==value
    else:
        from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import read
        assert read(logical,digest_field=seal)==value
    assert not logical.exists()


def test_real_progress_reader_uses_complete_retained_chain_not_absent_directory(tmp_path,monkeypatch):
    digest='sha256:'+'a'*64
    records=[]
    prior=None
    for sequence in (1,2):
        value=dict(schema_version='task_evaluation_sam31_preparation_progress.v1',request_digest=digest,
            sequence=sequence,previous_progress_digest=prior,advancement={'status':'ready' if sequence==2 else 'waiting'})
        value['progress_digest']=canonical_digest(value,digest_field='progress_digest')
        records.append(('source-progress/prep-'+digest[7:]+f'/{sequence:06d}-{value["progress_digest"][7:]}.json',value))
        prior=value['progress_digest']
    _,_,member=publish(tmp_path,monkeypatch,records)
    from blueprint_pipeline.task_evaluation_sam31_preparation_queue import load_progress
    assert load_progress(member,'prep-'+digest[7:]+'.json',digest)==records[-1][1]


def test_changed_clone_permissions_refuse_instead_of_widening_private_history(tmp_path,monkeypatch):
    relative,schema,seal=CASES[0]
    value=dict(schema_version=schema)
    value[seal]=canonical_digest(value,digest_field=seal)
    raws,closure,_=publish(tmp_path,monkeypatch,[(relative,value)])
    Path(closure['rows'][0]['retained_raw_ref']['path']).chmod(0o640)
    from blueprint_pipeline.task_evaluation_sam31_phase_queue import _read
    with pytest.raises(ValueError):
        _read(next(iter(raws)))


def test_unknown_progress_does_not_publish_a_false_empty_history(tmp_path,monkeypatch):
    with pytest.raises(ValueError):
        publish(tmp_path,monkeypatch,[('source-progress/prep-'+('a'*64)+'/000001-'+('b'*64)+'.json',
            {'schema_version':'future_schema.v9'})])


@pytest.mark.parametrize('mutation',['wrong_seal','future_schema','unexpected_filename'])
def test_progress_listing_cannot_hide_unsupported_original_json(tmp_path,monkeypatch,mutation):
    value=dict(schema_version='task_evaluation_sam31_preparation_progress.v1')
    value['progress_digest']=canonical_digest(value,digest_field='progress_digest')
    relative='source-progress/prep-'+('a'*64)+'/000001-'+('b'*64)+'.json'
    if mutation=='wrong_seal':
        value['progress_digest']='sha256:'+'f'*64
    elif mutation=='future_schema':
        value['schema_version']='future_schema.v9'
    else:
        relative='source-progress/prep-'+('a'*64)+'/unknown.json'
    with pytest.raises(ValueError):
        publish(tmp_path,monkeypatch,[(relative,value)])


def test_metadata_framing_quota_refuses_before_payload_read(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_metadata as metadata
    monkeypatch.setattr(metadata,'MAX_METADATA_BYTES',1)
    def unexpected(*args):
        pytest.fail('closure payload read preceded framing allowance')
    monkeypatch.setattr(metadata,'_payload',unexpected)
    relative,schema,seal=CASES[0]
    value=dict(schema_version=schema)
    value[seal]=canonical_digest(value,digest_field=seal)
    with pytest.raises(ValueError,match='scene_retirement_metadata_limit'):
        publish(tmp_path,monkeypatch,[(relative,value)])
