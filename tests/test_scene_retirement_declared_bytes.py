"""Promised raw identities and actual published bytes precede scene removal."""
import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance


@pytest.mark.parametrize('field,value',[('sha256','sha256:'+'f'*64),('size_bytes',999)])
def test_actual_preserved_local_payload_must_match_declared_reference(tmp_path,monkeypatch,field,value):
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    from tests.test_scene_retirement_member_mutation import setup_operation
    _,member,preserved,journal=setup_operation(tmp_path,monkeypatch)
    row=preserved['files'][0]
    path=member/row['relative_path']
    reference=dict(path=str(path),sha256=row['sha256'],size_bytes=row['size_bytes'])
    reference[field]=value
    fresh={'historical_lineage':{'raw_reference_obligations':[reference]}}
    with pytest.raises(ValueError,match='scene_retirement_declared_payload_changed'):
        engine._verify_declared_bytes(fresh,preserved,object(),journal.allowance)
    assert path.exists()


def published_scene(tmp_path):
    from tests.scene_lifecycle_fixture_support import rebase_graph
    from tests.test_scene_source_family_website import publication_fixture,api
    payload=b'example'
    expected='sha256:'+hashlib.sha256(payload).hexdigest()
    args=rebase_graph(publication_fixture(host_only=True),tmp_path.resolve(),
                      remote_digest_replacements={'sha256:'+'d'*64:expected})
    for group in ('seed_records','downstream_records','source_records'):
        for role,rows in args[group].items():
            rows=([rows] if rows is not None else []) if role in {'intent','projection'} else rows
            for path,raw in rows:
                path=Path(path)
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(raw)
    history=api().join_retained_scene_source_family_inventory(**args)
    assert history['publication_observations'][0]['historical_publication_binding_verified'] is True
    import json
    publication=json.loads(args['source_records']['submission_publications'][0][1])
    manifest_path,manifest_raw=args['seed_records']['source_submissions'][1]
    objects={row['uri']:(payload if row['relative_path'].startswith('derived/') else manifest_raw)
             for row in publication['published_objects']}
    assert Path(manifest_path).read_bytes()==manifest_raw
    return history,publication,objects


@pytest.mark.parametrize('corrupt',[False,True])
def test_owner_bound_publication_gets_fresh_full_byte_readback_without_host_source_fallback(tmp_path,corrupt):
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    history,publication,objects=published_scene(tmp_path)
    allowance=ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)
    seen=[]
    class Transport:
        def read_published_object_charged(self,uri,selected,*,expected_size_bytes):
            assert selected is allowance
            assert expected_size_bytes==len(objects[uri])
            seen.append(uri)
            raw=objects[uri]+(b'wrong' if corrupt else b'')
            selected.charge('remote_bytes',len(raw))
            yield raw
    fresh={'historical_lineage':{'source_family_inventory':history}}
    if corrupt:
        with pytest.raises(ValueError,match='scene_retirement_published_readback_unproven'):
            engine._verify_declared_bytes(fresh,{'files':[]},Transport(),allowance)
    else:
        proof=engine._verify_declared_bytes(fresh,{'files':[]},Transport(),allowance)
        assert proof['published_objects_verified']==len(objects)
        assert set(seen)==set(objects)
        assert allowance.counts['remote_bytes']==sum(map(len,objects.values()))
    assert not any(row['uri'] in seen for row in publication['host_only_source_objects'])


def test_declared_dictionary_bound_precedes_exposing_its_semantics():
    from blueprint_pipeline.task_evaluation_scene_retirement_declared_bytes import _objects
    allowance=ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)
    source=_objects(dict.fromkeys(range(10001)),allowance)
    with pytest.raises(ValueError,match='scene_retirement_declared_reference_limit'):
        next(source)


@pytest.mark.parametrize('bad_digest',[False,True])
def test_published_iterator_cleanup_is_typed_and_preserves_incoming_refusal(bad_digest):
    from blueprint_pipeline.task_evaluation_scene_retirement_declared_bytes import verify_publication_rows
    allowance=ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)
    class Stream:
        def __iter__(self):
            return self
        def __next__(self):
            if getattr(self,'done',False):
                raise StopIteration
            self.done=True
            allowance.charge('remote_bytes',1)
            return b'x'
        def close(self):
            raise RuntimeError('private provider token must not escape')
    class Transport:
        def read_published_object_charged(self,uri,selected,*,expected_size_bytes):
            return Stream()
    digest='sha256:'+hashlib.sha256(b'x').hexdigest()
    if bad_digest:
        digest='sha256:'+'a'*64
    reason='scene_retirement_published_readback_unproven' if bad_digest else 'scene_retirement_remote_cleanup_unproven'
    with pytest.raises(ValueError,match=reason) as caught:
        verify_publication_rows([dict(uri='s3://blueprint/task-evaluation/production-inputs/fixture/a',
                                     digest=digest,size_bytes=1)],Transport(),allowance)
    assert 'private provider' not in str(caught.value)


def test_resume_reserves_publication_bytes_before_persisting_or_entering_readback():
    from blueprint_pipeline.task_evaluation_scene_retirement_recovery import reserve_phase
    allowance=ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)
    observed=[]
    class Journal:
        def __init__(self):
            self.allowance=allowance
        def preflight(self,events):
            observed.append(('preflight',dict(allowance.counts)))
        def append(self,*args,**kwargs):
            observed.append(('append',dict(allowance.counts)))
            assert kwargs['evidence']['action_allowance']['counts']['remote_bytes']==10
    reserve_phase(Journal(),{'archive':{'size_bytes':2}},readback=True,
                  published_objects=[dict(size_bytes=2),dict(size_bytes=3)])
    assert observed==[(kind,dict(local_bytes=0,archive_bytes=0,remote_bytes=10))
                      for kind in ('preflight','append')]


def test_restoration_reserves_both_remote_eof_probes_before_each_read():
    from blueprint_pipeline.task_evaluation_scene_retirement_recovery import reserve_phase
    allowance=ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)
    observed=[]
    class Journal:
        def __init__(self):
            self.allowance=allowance
        def preflight(self,events):
            observed.append(dict(allowance.counts))
        def append(self,*args,**kwargs):
            observed.append(dict(allowance.counts))
    reserve_phase(Journal(),{'archive':{'size_bytes':2},'files':[{'size_bytes':1}]},restoring=True)
    assert observed==[dict(local_bytes=2,archive_bytes=0,remote_bytes=6)]*2


@pytest.mark.parametrize('field,value',[('sha256','sha256:'+'f'*64),('size_bytes',999)])
def test_native_measured_rows_do_not_hide_declared_payload_contradictions(tmp_path,monkeypatch,field,value):
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    from tests.test_scene_retirement_member_mutation import setup_operation
    _,member,preserved,journal=setup_operation(tmp_path,monkeypatch)
    row=preserved['files'][0]
    reference=dict(path=str(member/row['relative_path']),sha256=row['sha256'],size_bytes=row['size_bytes'])
    reference[field]=value
    sink=RetainedEmissionBudget(max_bytes=10000,max_rows=10,max_references=10)
    fresh={'historical_lineage':{'raw_reference_obligations':sink.rows([reference])}}
    with pytest.raises(ValueError,match='scene_retirement_declared_payload_changed'):
        engine._verify_declared_bytes(fresh,preserved,object(),journal.allowance)


def test_unknown_collection_subclass_cannot_hide_promised_bytes_or_run_callbacks():
    from blueprint_pipeline.task_evaluation_scene_retirement_declared_bytes import _objects
    allowance=ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)
    class Foreign(list):
        def __iter__(self):
            raise AssertionError('foreign callback must not run')
    with pytest.raises(ValueError,match='scene_retirement_declared_reference_invalid'):
        list(_objects({'raw_reference_obligations':Foreign([{'path':'/untrusted'}])},allowance))
