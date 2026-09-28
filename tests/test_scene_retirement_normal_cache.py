# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_cache.py
#   src/blueprint_pipeline/task_evaluation_launch_preparation_worker.py
#   src/blueprint_pipeline/task_evaluation_scene_progression_transport.py
"""ADP-009D/day28: normal CAS publication must retain authenticated storage birth."""
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_scene_retirement_real_participants import access_fixture, authenticated_birth_refs
from tests.test_scene_retirement_connected_acceptance import _raw, _sealed_file
from tests.test_task_evaluation_launch_preparation_contract import test_configuration_request as configuration_request
from tests.test_task_evaluation_launch_preparation_worker import request_with_fetchable_bytes, fetcher, SERVICE_ACCOUNT


def owner_submission(tmp_path,monkeypatch):
    access,policy,_=access_fixture(tmp_path,monkeypatch)
    _,owner,attempt=authenticated_birth_refs(tmp_path,monkeypatch)
    intent=json.loads(Path(owner['path']).read_bytes())
    birth=json.loads(Path(attempt['path']).read_bytes())
    value,payloads=request_with_fetchable_bytes(configuration_request())
    value['expected_production_commit']=birth['source_commit']
    value['scene_intent_digest']=intent['intent_digest']
    value['task']['identity']['id']=intent['request']['task']['task_id']
    output=tmp_path/'factory'
    output.mkdir()
    request=output/'submission_request.json'
    request.write_text(json.dumps(value))
    factory=output/'factory.json'
    _sealed_file(factory,dict(schema_version='website_scene_attempt_factory.v1',status='publication_ready',
        intent_digest=intent['intent_digest'],attempt_digest=birth['attempt_digest'],
        source_commit=birth['source_commit'],submission_request=_raw(request),
        provider_mutation_performed=False), 'factory_digest')
    inputs=tmp_path/'inputs'
    inputs.mkdir()
    queue=tmp_path/'queue'
    queue.mkdir()
    policy['roots']=[dict(root=str(inputs),storage_class='cache',device=inputs.stat().st_dev),
        dict(root=str(output),storage_class='host',device=output.stat().st_dev)]
    _sealed_file(tmp_path/'policy.json',policy,'policy_digest',mode=0o644)
    return access,policy,value,payloads,queue,inputs,dict(intent_raw_ref=owner,attempt_raw_ref=attempt,
        factory_raw_ref=_raw(factory),submission_request_raw_ref=_raw(request))


def test_normal_owned_queue_authority_exists_before_actual_queue_publication(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request
    _,_,value,_,queue,_,proofs=owner_submission(tmp_path,monkeypatch)
    selected=cache.publish_preparation_storage_authority(queue_root=queue,request=value,now=101,**proofs)
    assert not (queue/'pending').exists()
    receipt=stage_launch_preparation_request(value=value,queue_root=queue,submitted_by='scene-progression')
    path=Path(selected['path'])
    assert path.parent==queue/'scene-authorities'
    assert path.name==Path(receipt['queue_path']).name
    assert _raw(path)==selected
    assert json.loads(path.read_bytes())['request_digest']==receipt['request_digest']


def test_actual_normal_materializer_births_projection_and_exact_regular_cache_generations(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request
    _,policy,value,payloads,queue,inputs,proofs=owner_submission(tmp_path,monkeypatch)
    cache.publish_preparation_storage_authority(queue_root=queue,request=value,now=101,**proofs)
    queued=stage_launch_preparation_request(value=value,queue_root=queue,submitted_by='scene-progression')
    target=inputs/value['preparation_id']
    cache.enroll_preparation_storage(queue_path=queued['queue_path'],input_root=inputs,now=101)
    result=worker.materialize_preparation_references(request=value,input_root=target,
        content_store_root=inputs/'content-addressed'/'sha256',
        allowed_uri_prefixes=['s3://blueprint-production-inputs/'],service_account=SERVICE_ACCOUNT,
        source_commit=value['expected_production_commit'],fetcher=fetcher(payloads))
    generations=Path(policy['generation_store'])
    directory=json.loads((generations/(hashlib.sha256(str(target).encode()).hexdigest()+'.json')).read_bytes())
    assert directory['state']=='active'
    for row in result['references']:
        leaf=inputs/'content-addressed'/'sha256'/row['digest'][7:]
        generation=json.loads((generations/(hashlib.sha256(str(leaf).encode()).hexdigest()+'.json')).read_bytes())
        assert generation['schema_version']=='scene_content_generation.v1'
        assert generation['state']=='active' and generation['digest']==row['digest']
        assert generation['dev']==leaf.stat().st_dev and generation['ino']==leaf.stat().st_ino
        assert leaf.stat().st_ino==Path(row['materialized_path']).stat().st_ino
        assert leaf.stat().st_nlink==2


@pytest.mark.parametrize('field',['intent_raw_ref','attempt_raw_ref','factory_raw_ref','submission_request_raw_ref'])
def test_owner_sidecar_cannot_borrow_resealed_foreign_or_changed_raw_authority(tmp_path,monkeypatch,field):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    _,_,value,_,queue,_,proofs=owner_submission(tmp_path,monkeypatch)
    changed=Path(proofs[field]['path'])
    changed.unlink()  # Substitute the immutable named entry; never weaken its mode.
    changed.write_text('{"foreign":"owner"}')
    with pytest.raises(ValueError):
        cache.publish_preparation_storage_authority(queue_root=queue,request=value,now=101,**proofs)
    assert not (queue/'scene-authorities').exists()


def test_missing_normal_owner_sidecar_does_not_adopt_legacy_preparation_directory(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    _,policy,value,_,queue,inputs,_=owner_submission(tmp_path,monkeypatch)
    target=inputs/value['preparation_id']
    target.mkdir()
    assert cache.enroll_preparation_storage(queue_path=queue/'pending'/'legacy.json',input_root=inputs,now=101) is None
    assert target.is_dir()
    assert list(Path(policy['generation_store']).glob('*.json'))==[]
