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
        def read_published_object_charged(self,uri,selected):
            assert selected is allowance
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
