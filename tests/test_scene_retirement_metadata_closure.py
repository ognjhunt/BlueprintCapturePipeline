"""Real retained accounting reopens exact metadata after member removal."""
import hashlib
import json
from pathlib import Path
import pytest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_retirement_real_participants import access_fixture


def closure_fixture(tmp_path,monkeypatch):
    access,policy,member=access_fixture(tmp_path,monkeypatch)
    journal=Path(policy['journal_store']+'.metadata');journal.mkdir(mode=0o750)
    token='2'*32
    blob=journal/'original-launch.json'
    raw=b'{"schema_version":"task_evaluation_launch_receipt.v1","status":"completed","paid_execution_requested":true}'
    blob.write_bytes(raw);blob.chmod(0o640)
    digest='sha256:'+hashlib.sha256(raw).hexdigest()
    logical=member/'launch_receipt.json'
    identity=member.stat()
    generation=dict(schema_version='scene_member_generation.v1',canonical_path=str(member),
        state='retired',generation_id='1'*32,retirement_token=token,dev=identity.st_dev,ino=identity.st_ino,mode=identity.st_mode)
    generation['state_digest']=canonical_digest(generation,digest_field='state_digest')
    path=Path(policy['generation_store'])/(hashlib.sha256(str(member).encode()).hexdigest()+'.json')
    path.write_text(json.dumps(generation));path.chmod(0o600)
    index=dict(schema_version='scene_retirement_metadata_closure.v1',token=token,members=[dict(
        canonical_path=str(member),generation_id=generation['generation_id'])],rows=[dict(
            logical_path=str(logical),sha256=digest,size_bytes=len(raw),retained_raw_ref=dict(
                path=str(blob),sha256=digest,size_bytes=len(raw)))])
    index['closure_digest']=canonical_digest(index,digest_field='closure_digest')
    path=journal/(token+'.metadata.json');path.write_text(json.dumps(index));path.chmod(0o640)
    member.rmdir()
    return access,policy,logical,blob,raw


def test_real_retained_accounting_reader_reopens_exact_original_metadata_under_lock(tmp_path,monkeypatch):
    from blueprint_pipeline.control_plane_retained_receipt import read_receipt_bytes
    access,_,logical,_,raw=closure_fixture(tmp_path,monkeypatch)
    assert read_receipt_bytes(logical)==raw
    with access.exclusive_scene_access():
        with pytest.raises(ValueError):
            read_receipt_bytes(logical)


def test_missing_or_changed_metadata_clone_never_falls_back_to_other_owner_or_remote(tmp_path,monkeypatch):
    from blueprint_pipeline.control_plane_retained_receipt import read_receipt_bytes
    _,_,logical,blob,_=closure_fixture(tmp_path,monkeypatch)
    blob.write_bytes(b'foreign-owner')
    with pytest.raises(ValueError):
        read_receipt_bytes(logical)
    assert not logical.exists()


def test_real_factory_request_reader_keeps_original_logical_raw_selector(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import checked_file, read
    _,_,logical,_,raw=closure_fixture(tmp_path,monkeypatch)
    reference=dict(path=str(logical),sha256='sha256:'+hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
    selected=checked_file(logical,reference)
    assert selected==logical
    assert read(selected)==json.loads(raw)
    assert not logical.exists()


@pytest.mark.parametrize('field,value',[('sha256','sha256:'+'f'*64),('sha256',None),('size_bytes',False),('size_bytes',1)])
def test_real_factory_request_reader_refuses_wrong_raw_selector(tmp_path,monkeypatch,field,value):
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import checked_file
    _,_,logical,_,raw=closure_fixture(tmp_path,monkeypatch)
    reference=dict(path=str(logical),sha256='sha256:'+hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
    reference[field]=value
    with pytest.raises(ValueError):
        checked_file(logical,reference)


def test_metadata_publisher_never_widens_private_or_secret_records(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_metadata import retain_metadata_closure
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance, _scan, _payload
    access,policy,member=access_fixture(tmp_path,monkeypatch)
    store=Path(policy['journal_store']+'.metadata');store.mkdir(mode=0o750)
    raw=b'{"schema_version":"task_evaluation_launch_receipt.v1"}'
    source=member/'launch_receipt.json';source.write_bytes(raw);source.chmod(0o600)
    allowance=ActionAllowance(expires_at=1000,now=lambda:200)
    files,directories=[],[];root=_scan(member,0,allowance,files,directories)
    files[0]['sha256']='sha256:'+hashlib.sha256(b''.join(_payload(source,files[0],allowance))).hexdigest()
    with pytest.raises(ValueError):
        retain_metadata_closure(dict(files=files,members=[root]),policy,[dict(generation_id='1'*32)],'2'*32,allowance)
    assert not list(store.iterdir())
