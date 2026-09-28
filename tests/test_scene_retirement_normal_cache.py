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


def owner_submission(tmp_path,monkeypatch,*,production=False):
    access,policy,_=access_fixture(tmp_path,monkeypatch)
    _,owner,attempt=authenticated_birth_refs(tmp_path,monkeypatch)
    intent=json.loads(Path(owner['path']).read_bytes())
    birth=json.loads(Path(attempt['path']).read_bytes())
    if production:
        from tests.test_task_evaluation_launch_preparation_worker import production_request_with_fetchable_bytes
        value,payloads=production_request_with_fetchable_bytes()
    else:
        value,payloads=request_with_fetchable_bytes(configuration_request())
    value['expected_production_commit']=birth['source_commit']
    value['scene_intent_digest']=intent['intent_digest']
    value['task']['identity']['id']=intent['request']['task']['task_id']
    if production:
        from blueprint_pipeline.decision_evidence_contracts import canonical_digest
        reference=value['construction']['recipe']
        recipe=json.loads(payloads[reference['uri']])
        recipe['task_identity']=value['task']['identity']
        recipe['recipe_digest']=canonical_digest(recipe,digest_field='recipe_digest')
        raw=json.dumps(recipe,sort_keys=True).encode()
        payloads[reference['uri']]=raw
        reference.update(digest='sha256:'+hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
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


def test_actual_owned_submitter_publishes_authority_before_queue(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline.task_evaluation_scene_progression_transport import submit_owned_preparation
    _,_,value,_,queue,_,proofs=owner_submission(tmp_path,monkeypatch)
    monkeypatch.setattr(cache.time,'time',lambda:101)
    result=submit_owned_preparation(request_path=Path(proofs['submission_request_raw_ref']['path']),
        config={'preparation_queue_root':str(queue)},intent_reference=proofs['intent_raw_ref'],
        attempt_reference=proofs['attempt_raw_ref'],factory_reference=proofs['factory_raw_ref'])
    pending=Path(result['queue_receipt']['queue_path'])
    sidecar=json.loads((queue/'scene-authorities'/pending.name).read_bytes())
    assert sidecar['factory_raw_ref']==proofs['factory_raw_ref']
    assert sidecar['attempt_raw_ref']==proofs['attempt_raw_ref']


def test_actual_normal_queue_births_before_global_cas_materialization(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request
    from tests.test_task_evaluation_launch_preparation_worker import fake_scene_render_inputs
    _,policy,value,payloads,queue,inputs,proofs=owner_submission(tmp_path,monkeypatch,production=True)
    monkeypatch.setattr(cache.time,'time',lambda:101)
    cache.publish_preparation_storage_authority(queue_root=queue,request=value,now=101,**proofs)
    stage_launch_preparation_request(value=value,queue_root=queue,submitted_by='scene-progression')
    run=worker.process_launch_preparation_queue(queue_root=queue,input_root=inputs,
        allowed_uri_prefixes=['s3://blueprint-production-inputs/'],service_account=SERVICE_ACCOUNT,
        source_commit=value['expected_production_commit'],fetcher=fetcher(payloads),
        scene_render_input_materializer=fake_scene_render_inputs,construction_queue_root=tmp_path/'construction')
    assert run['results'][0]['status']=='queued_for_production_scene_configuration',run
    store=Path(policy['generation_store'])
    target=inputs/value['preparation_id']
    generation=json.loads((store/(hashlib.sha256(str(target).encode()).hexdigest()+'.json')).read_bytes())
    assert generation['state']=='active' and generation['owner_raw_ref']==proofs['intent_raw_ref']
    for leaf in (inputs/'content-addressed'/'sha256').iterdir():
        if len(leaf.name)!=64:  # Existing materializer lock directory is not a digest object.
            continue
        record=json.loads((store/(hashlib.sha256(str(leaf).encode()).hexdigest()+'.json')).read_bytes())
        assert record['state']=='active' and record['source_publication_raw_ref']


def test_actual_preparation_only_authority_can_birth_without_paid_reservation(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_preparation_attempts import create_preparation_attempt
    access,_,value,_,_,inputs,proofs=owner_submission(tmp_path,monkeypatch)
    owner=json.loads(Path(proofs['intent_raw_ref']['path']).read_bytes())
    directory=Path(proofs['intent_raw_ref']['path']).parent
    create_preparation_attempt(directory=directory,attempt_id='preparing-1',
        source_commit=value['expected_production_commit'],runtime_digest='sha256:'+'a'*64,
        input_digest='sha256:'+'b'*64,now=101)
    target=inputs/'new-preparation'
    born=access.birth_scene_member(target,owner_intent_id=owner['intent_id'],
        owner_raw_ref=proofs['intent_raw_ref'],birth_request_raw_ref=_raw(directory/'preparation-attempts'/'preparing-1.json'),now=101)
    assert born['state']=='active'
    assert born['birth_request_raw_ref']['path'].endswith('/preparation-attempts/preparing-1.json')
    assert json.loads(Path(born['birth_request_raw_ref']['path']).read_bytes())['paid_authority_granted'] is False


def test_preservation_includes_exact_consent_cache_alias_without_adopting_store_parent(tmp_path):
    import os
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members,ActionAllowance
    from tests.test_scene_retirement_preservation import MemoryTransport
    member=tmp_path/'owned-projection'
    member.mkdir()
    (member/'payload').write_bytes(b'normal-object')
    store=tmp_path/'global-store'
    store.mkdir()
    digest='sha256:'+hashlib.sha256(b'normal-object').hexdigest()
    alias=store/digest[7:]
    os.link(member/'payload',alias)
    result=preserve_members([member],transport=MemoryTransport([member]),token='a'*32,
        allowance=ActionAllowance(expires_at=200,now=lambda:100,monotonic=lambda:0),
        cache_aliases=[{'canonical_path':str(alias),'digest':digest,'size_bytes':13}])
    assert result['cache_aliases'][0]['path']==str(alias)
    assert result['cache_aliases'][0]['member_index']==0
    assert result['cache_aliases'][0]['relative_path']=='payload'
    assert result['members'][0]['path']==str(member)
    assert str(store) not in [row['path'] for row in result['members']]
    assert alias.exists() and member.exists()


def test_cache_union_refuses_other_alias_before_upload(tmp_path):
    import os
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members,ActionAllowance
    from tests.test_scene_retirement_preservation import MemoryTransport
    member=tmp_path/'projection'
    member.mkdir()
    (member/'payload').write_bytes(b'normal-object')
    alias=tmp_path/hashlib.sha256(b'normal-object').hexdigest()
    os.link(member/'payload',alias)
    other=tmp_path/'other-owner'
    os.link(alias,other)
    transport=MemoryTransport([member])
    with pytest.raises(ValueError,match='scene_retirement_shared_inode'):
        preserve_members([member],transport=transport,token='a'*32,
            allowance=ActionAllowance(expires_at=200,now=lambda:100,monotonic=lambda:0),
            cache_aliases=[{'canonical_path':str(alias),'digest':'sha256:'+alias.name,'size_bytes':13}])
    assert not transport.objects and other.read_bytes()==b'normal-object'


def test_real_cache_alias_removal_and_restore_keep_full_bytes_mode_and_inode_union(tmp_path,monkeypatch):
    import os
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_preserved_members
    from tests.test_scene_retirement_member_mutation import setup_operation
    from tests.test_scene_retirement_preservation import MemoryTransport
    access,member,_,journal=setup_operation(tmp_path,monkeypatch)
    payload=member/'nested'/'evidence.bin'
    digest='sha256:'+hashlib.sha256(payload.read_bytes()).hexdigest()
    store=tmp_path/'store'
    store.mkdir()
    alias=store/digest[7:]
    os.link(payload,alias)
    transport=MemoryTransport([member])
    preserved=preserve_members([member],transport=transport,allowance=journal.allowance,token='3'*32,
        cache_aliases=[dict(canonical_path=str(alias),digest=digest,size_bytes=payload.stat().st_size)])
    removed={}
    with access.exclusive_scene_access():
        detach_and_remove(preserved,member_index=0,generation_id='2'*32,journal=journal,removed_inodes=removed)
        outcomes=cache.remove_preserved_cache_aliases(preserved,journal=journal,removed_inodes=removed)
        assert outcomes[0]['outcome']=='removed' and not alias.exists()
        transport.members=[]
        restored=restore_preserved_members(preserved,transport=transport,journal=journal)
    assert restored[0]['outcome']=='restored'
    assert payload.read_bytes()==alias.read_bytes()==b'preserved-evidence'
    assert payload.stat().st_ino==alias.stat().st_ino and payload.stat().st_nlink==2
    assert (alias.stat().st_uid,alias.stat().st_gid,alias.stat().st_mode & 0o777)==(
        preserved['cache_aliases'][0]['uid'],preserved['cache_aliases'][0]['gid'],preserved['cache_aliases'][0]['mode'])


def test_restore_cache_alias_never_overwrites_even_same_bytes(tmp_path,monkeypatch):
    import os
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import detach_and_remove
    from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_preserved_members
    from tests.test_scene_retirement_member_mutation import setup_operation
    from tests.test_scene_retirement_preservation import MemoryTransport
    access,member,_,journal=setup_operation(tmp_path,monkeypatch)
    payload=member/'nested'/'evidence.bin'
    data=payload.read_bytes()
    digest='sha256:'+hashlib.sha256(data).hexdigest()
    alias=tmp_path/digest[7:]
    os.link(payload,alias)
    transport=MemoryTransport([member])
    preserved=preserve_members([member],transport=transport,allowance=journal.allowance,token='3'*32,
        cache_aliases=[dict(canonical_path=str(alias),digest=digest,size_bytes=len(data))])
    removed={}
    with access.exclusive_scene_access():
        detach_and_remove(preserved,member_index=0,generation_id='2'*32,journal=journal,removed_inodes=removed)
        cache.remove_preserved_cache_aliases(preserved,journal=journal,removed_inodes=removed)
        alias.write_bytes(data)
        original=alias.stat().st_ino
        transport.members=[]
        with pytest.raises(ValueError,match='scene_retirement_cache_restore_conflict'):
            restore_preserved_members(preserved,transport=transport,journal=journal)
    assert alias.read_bytes()==data and alias.stat().st_ino==original


def cache_action_consent(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request
    _,policy,value,payloads,queue,inputs,proofs=owner_submission(tmp_path,monkeypatch)
    source=cache.publish_preparation_storage_authority(queue_root=queue,request=value,now=101,**proofs)
    queued=stage_launch_preparation_request(value=value,queue_root=queue,submitted_by='scene-progression')
    cache.enroll_preparation_storage(queue_path=queued['queue_path'],input_root=inputs,now=101)
    worker.materialize_preparation_references(request=value,input_root=inputs/value['preparation_id'],
        content_store_root=inputs/'content-addressed'/'sha256',allowed_uri_prefixes=['s3://blueprint-production-inputs/'],
        service_account=SERVICE_ACCOUNT,source_commit=value['expected_production_commit'],fetcher=fetcher(payloads))
    generation_path=next(path for path in Path(policy['generation_store']).glob('*.json')
        if json.loads(path.read_bytes()).get('schema_version')=='scene_content_generation.v1')
    generation=json.loads(generation_path.read_bytes())
    owner=json.loads(Path(proofs['intent_raw_ref']['path']).read_bytes())
    policy['principals']=[dict(principal_id='operator',actions=['retire','restore'],
        owner_intent_ids=[owner['intent_id']],private_archive_classes=[])]
    _sealed_file(tmp_path/'policy.json',policy,'policy_digest',mode=0o644)
    plan=tmp_path/'plan.json'
    plan.write_text('{}')
    consent=dict(schema_version='scene_retirement_consent.v1',consent_id='a'*32,principal_id='operator',
        intent_id=owner['intent_id'],intent_raw_ref=proofs['intent_raw_ref'],plan_raw_ref=_raw(plan),
        retired_journal_raw_ref=None,policy_sha256=_raw(tmp_path/'policy.json')['sha256'],
        cohort_sha256=__import__('blueprint_pipeline.task_evaluation_scene_retirement_authority',fromlist=['cohort_digest']).cohort_digest(policy['consumer_cohort']),
        action='retire',created_at=100,expires_at=200,members=[],private_archive_classes=[],
        cache_objects=[dict(canonical_path=generation['canonical_path'],digest=generation['digest'],
            size_bytes=generation['size_bytes'],generation_id=generation['generation_id'],
            generation_raw_ref=_raw(generation_path),source_raw_ref=source)],terminal_pin_refs=[])
    path=tmp_path/'consent.json'
    _sealed_file(path,consent,'consent_digest',mode=0o600)
    return path,consent


def test_cache_consent_is_bound_to_actual_generation_and_same_authenticated_owner(tmp_path,monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_authority import load_authority
    path,consent=cache_action_consent(tmp_path,monkeypatch)
    result=load_authority(path,action='retire',now=lambda:101)
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    selected=cache.validate_cache_objects(result['policy'],result['consent'],
        ActionAllowance(expires_at=200,now=lambda:101,monotonic=lambda:0))
    assert selected[0]['canonical_path']==consent['cache_objects'][0]['canonical_path']


@pytest.mark.parametrize('field',['canonical_path','digest','generation_id','source_raw_ref'])
def test_cache_consent_cannot_select_free_or_foreign_cache_authority(tmp_path,monkeypatch,field):
    from blueprint_pipeline.task_evaluation_scene_retirement_authority import load_authority
    path,consent=cache_action_consent(tmp_path,monkeypatch)
    row=consent['cache_objects'][0]
    if field=='source_raw_ref':
        row[field]=consent['intent_raw_ref']
    elif field=='canonical_path':
        row[field]=str(tmp_path/'foreign-store'/('a'*64))
    elif field=='digest':
        row[field]='sha256:'+'a'*64
    else:
        row[field]='f'*32
    _sealed_file(path,consent,'consent_digest',mode=0o600)
    with pytest.raises(ValueError):
        authority=load_authority(path,action='retire',now=lambda:101)
        from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
        from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
        cache.validate_cache_objects(authority['policy'],authority['consent'],
            ActionAllowance(expires_at=200,now=lambda:101,monotonic=lambda:0))


def test_fresh_retirement_consent_preserves_expired_original_authority_as_history_only(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline.task_evaluation_scene_retirement_authority import load_authority
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    from blueprint_pipeline.task_evaluation_scene_owner_authority import reopen_scene_intent
    path,consent=cache_action_consent(tmp_path,monkeypatch)
    consent.update(created_at=2000,expires_at=3000)
    _sealed_file(path,consent,'consent_digest',mode=0o600)
    with pytest.raises(ValueError,match='authority_expired'):
        reopen_scene_intent(consent['intent_raw_ref'],now=2001)
    authority=load_authority(path,action='retire',now=lambda:2001)
    selected=cache.validate_cache_objects(authority['policy'],consent,
        ActionAllowance(expires_at=3000,now=lambda:2001,monotonic=lambda:0))
    assert selected==consent['cache_objects']
    source=json.loads(Path(selected[0]['source_raw_ref']['path']).read_bytes())
    request=json.loads(Path(source['submission_request_raw_ref']['path']).read_bytes())
    # Cleanup proof cannot reopen publication or a fresh producer invocation.
    with pytest.raises(ValueError,match='authority_expired'):
        cache._validate(source,request,now=2001)


def test_native_action_targets_exact_normal_cache_union_without_claiming_store_parent(tmp_path,monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_preservation as preservation
    from blueprint_pipeline.task_evaluation_scene_retirement import _plan_members
    path,consent=cache_action_consent(tmp_path,monkeypatch)
    source=json.loads(Path(consent['cache_objects'][0]['source_raw_ref']['path']).read_bytes())
    request=json.loads(Path(source['submission_request_raw_ref']['path']).read_bytes())
    root=Path(consent['cache_objects'][0]['canonical_path']).parents[2]/request['preparation_id']
    # Include every actual worker cache object in the requested projection union.
    aliases=[]
    for leaf in Path(consent['cache_objects'][0]['canonical_path']).parent.iterdir():
        if len(leaf.name)==64:
            aliases.append(dict(canonical_path=str(leaf),digest='sha256:'+leaf.name,size_bytes=leaf.stat().st_size))
    allowance=preservation.ActionAllowance(expires_at=200,now=lambda:101,monotonic=lambda:0)
    monkeypatch.setattr(preservation,'_payload',lambda *a:pytest.fail('target metadata opened payload'))
    inventory=preservation._inventory_members([root],allowance,cache_aliases=aliases)
    consent['members']=[{'canonical_path':str(root),'class':'host'}]
    consent['cache_objects']=aliases
    plan={'action':'KEEP','cleanup_authorized':False,'measured_members':[
        {'path':str(root),'status':'observed_scoped_metadata','keeps':['external_hardlink_or_unobserved_alias']},
        *[{'path':row['canonical_path'],'status':'observed_scoped_metadata','storage_class':'cache',
            'keeps':['shared_content_object_not_exclusive','external_hardlink_or_unobserved_alias']} for row in aliases]]}
    _plan_members(plan,consent,allowance,cache_inventory=inventory)
    assert plan['action']=='KEEP' and plan['cleanup_authorized'] is False
    assert str(root.parent/'content-addressed') not in [row['path'] for row in plan['measured_members']]
    with pytest.raises(ValueError):
        _plan_members(plan,consent,allowance)  # Consent alone cannot waive sharing.


def test_native_cache_union_metadata_refuses_a_second_owner_alias_before_payload(tmp_path,monkeypatch):
    import os
    from blueprint_pipeline import task_evaluation_scene_retirement_preservation as preservation
    member=tmp_path/'projection';member.mkdir()
    (member/'file').write_bytes(b'object')
    alias=tmp_path/hashlib.sha256(b'object').hexdigest()
    os.link(member/'file',alias)
    os.link(alias,tmp_path/'other-owner')
    monkeypatch.setattr(preservation,'_payload',lambda *a:pytest.fail('unproven union opened payload'))
    with pytest.raises(ValueError,match='scene_retirement_shared_inode'):
        preservation._inventory_members([member],
            preservation.ActionAllowance(expires_at=200,now=lambda:100,monotonic=lambda:0),
            cache_aliases=[dict(canonical_path=str(alias),digest='sha256:'+alias.name,size_bytes=6)])
