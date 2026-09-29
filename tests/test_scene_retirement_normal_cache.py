# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_cache.py
#   src/blueprint_pipeline/task_evaluation_launch_preparation_worker.py
#   src/blueprint_pipeline/task_evaluation_scene_progression_transport.py
"""ADP-009D/day28: normal CAS publication must retain authenticated storage birth."""
import hashlib
import json
import os
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


@pytest.mark.parametrize('mode',['single','two','two-resume'])
def test_real_cache_alias_removal_and_restore_keep_full_bytes_mode_and_inode_union(tmp_path,monkeypatch,mode):
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
    aliases=[alias]
    if mode!='single':
        alternate=tmp_path/'adapter-store'
        alternate.mkdir()
        aliases.append(alternate/digest[7:])
        os.link(payload,aliases[-1])
    allocated=payload.stat().st_blocks*512
    transport=MemoryTransport([member])
    preserved=preserve_members([member],transport=transport,allowance=journal.allowance,token='3'*32,
        cache_aliases=[dict(canonical_path=str(path),digest=digest,size_bytes=payload.stat().st_size) for path in aliases])
    removed={}
    with access.exclusive_scene_access():
        detach_and_remove(preserved,member_index=0,generation_id='2'*32,journal=journal,removed_inodes=removed)
        if mode=='two-resume':
            append=journal.append
            def interrupted(event,**kwargs):
                if event=='cache_unlink_planned' and kwargs.get('member_key')=='cache-1':
                    raise OSError('test interruption after first cache unlink')
                return append(event,**kwargs)
            monkeypatch.setattr(journal,'append',interrupted)
            with pytest.raises(OSError):
                cache.remove_preserved_cache_aliases(preserved,journal=journal,removed_inodes=removed)
            monkeypatch.setattr(journal,'append',append)
            from blueprint_pipeline.task_evaluation_scene_retirement_recovery import removed_inode_counts
            removed=removed_inode_counts(journal)
        outcomes=cache.remove_preserved_cache_aliases(preserved,journal=journal,removed_inodes=removed)
        assert all(row['outcome']=='removed' for row in outcomes) and all(not path.exists() for path in aliases)
        assert sum(row['removed_allocated_bytes'] for row in outcomes)==allocated
        transport.members=[]
        restored=restore_preserved_members(preserved,transport=transport,journal=journal)
    assert restored[0]['outcome']=='restored'
    assert payload.read_bytes()==alias.read_bytes()==b'preserved-evidence'
    assert all(payload.stat().st_ino==path.stat().st_ino for path in aliases) and payload.stat().st_nlink==len(aliases)+1
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


def cache_action_consent(tmp_path,monkeypatch,*,cache_age=24*60*60):
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
    # Actual native idle age, independent of logical ownership or consent.
    os.utime(generation['canonical_path'],(101-cache_age,101-cache_age))
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


@pytest.mark.parametrize('age',[24*60*60-1,24*60*60,24*60*60+1])
def test_native_cache_action_preserves_original_idle_grace(tmp_path,monkeypatch,age):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline.task_evaluation_scene_retirement_authority import load_authority
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    path,consent=cache_action_consent(tmp_path,monkeypatch,cache_age=age)
    authority=load_authority(path,action='retire',now=lambda:101)
    allowance=ActionAllowance(expires_at=200,now=lambda:101,monotonic=lambda:0)
    original=Path(consent['cache_objects'][0]['canonical_path']).stat()
    if age<24*60*60:
        with pytest.raises(ValueError,match='scene_retirement_cache_idle_grace_unproven'):
            cache.validate_cache_objects(authority['policy'],consent,allowance)
    else:
        assert cache.validate_cache_objects(authority['policy'],consent,allowance)==consent['cache_objects']
    current=Path(consent['cache_objects'][0]['canonical_path']).stat()
    assert (current.st_dev,current.st_ino,current.st_mtime_ns)==(original.st_dev,original.st_ino,original.st_mtime_ns)


@pytest.mark.parametrize('closed',['exclusive_action','retired_generation','active_generation'])
def test_existing_blob_gc_cannot_race_scene_exclusion_or_closed_generation(tmp_path,monkeypatch,closed):
    from blueprint_pipeline import control_plane_storage_gc as gc
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    _,consent=cache_action_consent(tmp_path,monkeypatch)
    row=consent['cache_objects'][0]
    leaf=Path(row['canonical_path'])
    inode=(leaf.stat().st_dev,leaf.stat().st_ino)
    for candidate in (tmp_path/'inputs').rglob('*'):
        if candidate.is_file() and candidate!=leaf and (candidate.stat().st_dev,candidate.stat().st_ino)==inode:
            candidate.unlink()
    assert leaf.stat().st_nlink==1
    original=leaf.read_bytes()
    manifest=gc.build_gc_manifest(content_store_roots=[leaf.parent],minimum_age_seconds=0,now=lambda:101)
    assert manifest['candidate_count']==1
    if closed=='retired_generation':
        ledger=Path(row['generation_raw_ref']['path'])
        value=json.loads(ledger.read_bytes())
        value['state']='retired'
        value['state_digest']=canonical_digest(value,digest_field='state_digest')
        ledger.write_text(json.dumps(value))
        result=gc.apply_gc_manifest(manifest,ack=gc.EXECUTE_ACK)
    elif closed=='exclusive_action':
        with access.exclusive_scene_access():
            result=gc.apply_gc_manifest(manifest,ack=gc.EXECUTE_ACK)
    else:
        from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
        native_read=cache.os.read
        observed=[]
        def read(fd,count):
            info=os.fstat(fd)
            if (info.st_dev,info.st_ino)==inode:
                with pytest.raises(ValueError,match='scene_retirement_reader_active'):
                    with access.exclusive_scene_access():
                        pytest.fail('blob GC payload read escaped its native shared lifetime')
                observed.append(count)
            return native_read(fd,count)
        monkeypatch.setattr(cache.os,'read',read)
        result=gc.apply_gc_manifest(manifest,ack=gc.EXECUTE_ACK)
        assert result['removed_count']==1 and not leaf.exists() and observed
        return
    assert result['removed_count']==0 and len(result['skipped'])==1
    assert leaf.read_bytes()==original


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
    member=tmp_path/'projection'
    member.mkdir()
    (member/'file').write_bytes(b'object')
    alias=tmp_path/hashlib.sha256(b'object').hexdigest()
    os.link(member/'file',alias)
    os.link(alias,tmp_path/'other-owner')
    monkeypatch.setattr(preservation,'_payload',lambda *a:pytest.fail('unproven union opened payload'))
    with pytest.raises(ValueError,match='scene_retirement_shared_inode'):
        preservation._inventory_members([member],
            preservation.ActionAllowance(expires_at=200,now=lambda:100,monotonic=lambda:0),
            cache_aliases=[dict(canonical_path=str(alias),digest='sha256:'+alias.name,size_bytes=6)])


@pytest.mark.parametrize('mode',['complete','combined_journal_cap','restore','restore_generation_changed','restore_combined_cap'])
def test_real_engine_finish_removes_last_normal_cache_links_and_retires_generations(tmp_path,monkeypatch,mode):
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    from blueprint_pipeline.task_evaluation_scene_retirement_mutation import inventory_digest
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import preserve_members,ActionAllowance
    from blueprint_pipeline.task_evaluation_scene_retirement_journal import SceneJournal
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt
    from tests.test_scene_retirement_preservation import MemoryTransport
    path,consent=cache_action_consent(tmp_path,monkeypatch)
    policy=access._policy()
    policy['reference_context']={'roots':{'intent_root':str(Path(consent['intent_raw_ref']['path']).parent.parent)}}
    _sealed_file(tmp_path/'policy.json',policy,'policy_digest',mode=0o644)
    policy=access._policy()
    source=json.loads(Path(consent['cache_objects'][0]['source_raw_ref']['path']).read_bytes())
    request=json.loads(Path(source['submission_request_raw_ref']['path']).read_bytes())
    root=Path(consent['cache_objects'][0]['canonical_path']).parents[2]/request['preparation_id']
    records=[(file,json.loads(file.read_bytes())) for file in Path(policy['generation_store']).glob('*.json')
             if len(file.name)==69]
    generation=next(value for _,value in records if value.get('canonical_path')==str(root))
    caches=[(file,value) for file,value in records if value.get('schema_version')=='scene_content_generation.v1']
    consent['cache_objects']=[dict(canonical_path=value['canonical_path'],digest=value['digest'],
        size_bytes=value['size_bytes'],generation_id=value['generation_id'],generation_raw_ref=_raw(file),
        source_raw_ref=value['source_publication_raw_ref']) for file,value in caches]
    allowance=ActionAllowance(expires_at=200,now=lambda:101,monotonic=lambda:0)
    transport=MemoryTransport([root])
    preserved=preserve_members([root],transport=transport,allowance=allowance,token='4'*32,
        cache_aliases=[{key:row[key] for key in ('canonical_path','digest','size_bytes')}
                       for row in consent['cache_objects']])
    consent['members']=[dict(canonical_path=str(root),**{key:generation[key] for key in
        ('owner_intent_id','owner_raw_ref','generation_id','dev','ino','mode')},
        inventory_sha256=inventory_digest(preserved,0),**{'class':'host'})]
    store=Path(policy['journal_store'])
    store.mkdir(mode=0o700)
    (store/'retired').mkdir(mode=0o700)
    initial=dict(schema_version='scene_retirement_journal.v1',status='pending',
        intent_id=consent['intent_id'],intent_raw_ref=consent['intent_raw_ref'],plan_raw_ref=consent['plan_raw_ref'],
        members=consent['members'],generations=[generation],cache_objects=consent['cache_objects'],
        cache_generations=[value for _,value in caches],preserved=preserved,metadata_closure_raw_ref=None)
    journal=SceneJournal.create(store,token='4'*32,initial=initial,allowance=allowance)
    pending=publish_pending_receipt(policy,consent,journal,preserved,allowance)
    with access.exclusive_scene_access():
        event=journal.append('retiring',member_key='0',evidence={
            'generation_id':generation['generation_id'],'inventory_sha256':consent['members'][0]['inventory_sha256']})
        generation=engine._transition(policy,generation,state='retiring',token=journal.token,journal_ref=event,
            inventory_sha256=consent['members'][0]['inventory_sha256'])
        if mode=='combined_journal_cap':
            from blueprint_pipeline import task_evaluation_scene_retirement_journal as journal_module
            from blueprint_pipeline.task_evaluation_scene_retirement_mutation import removal_records
            folder_rows=list(removal_records(preserved,0,generation['generation_id'],journal,
                (root.parent.stat().st_dev,root.parent.stat().st_ino,root.parent.stat().st_mode)))
            cache_rows=list(engine._cache_remove_records(initial))
            monkeypatch.setattr(journal_module,'MAX_EVENTS',journal.sequence+max(len(folder_rows),len(cache_rows)))
            with pytest.raises(ValueError,match='scene_retirement_journal_limit'):
                engine._finish_retirement(policy,consent,initial,journal,pending,[generation],[],allowance)
            assert root.is_dir() and all(Path(row['canonical_path']).is_file() for row in consent['cache_objects'])
            assert all(json.loads(file.read_bytes())['state']=='active' for file,_ in caches)
            return
        receipt=engine._finish_retirement(policy,consent,initial,journal,pending,[generation],[],allowance)
    assert receipt['status']=='retired' and not root.exists()
    assert all(not Path(row['canonical_path']).exists() for row in consent['cache_objects'])
    assert receipt['removed_allocated_bytes']==preserved['unique_allocated_bytes']
    assert len(receipt['cache_outcomes'])==len(caches)
    snapshot=json.loads(Path(receipt['retired_journal_raw_ref']['path']).read_bytes())
    assert snapshot['cache_outcomes']==receipt['cache_outcomes']
    assert all(json.loads(file.read_bytes())['state']=='retired' for file,_ in caches)
    if not mode.startswith('restore'):
        return
    from blueprint_pipeline.task_evaluation_scene_retirement_intent_receipt import publish_progress_receipt
    retired_generation=engine._generation(policy,consent['members'][0],expected_states={'retired'},
        retired_token=journal.token)[0]
    restore_consent=dict(consent,action='restore',plan_raw_ref=None,
        retired_journal_raw_ref=receipt['retired_journal_raw_ref'])
    restore_path=tmp_path/'restore-consent.json'
    _sealed_file(restore_path,restore_consent,'consent_digest',mode=0o600)
    restore_consent=json.loads(restore_path.read_bytes())
    restore=SceneJournal.create(store,token='8'*32,allowance=allowance,initial=dict(
        schema_version='scene_restore_journal.v1',status='restoring',intent_id=consent['intent_id'],
        intent_raw_ref=consent['intent_raw_ref'],members=consent['members'],generations=[retired_generation],
        original_retirement_token=journal.token,retired_journal_raw_ref=receipt['retired_journal_raw_ref'],
        consent_raw_ref=_raw(restore_path),cache_objects=consent['cache_objects'],
        cache_generations=snapshot['cache_generations']))
    context=dict(original_retirement_token=journal.token,restore_journal_initial_raw_ref=restore.initial_ref)
    pending=publish_progress_receipt(policy,restore_consent,receipt['intent_receipt_raw_ref'],dict(
        status='restoring',token=restore.token,intent_id=consent['intent_id'],members=[],
        last_event_raw_ref=restore.prior_ref,**context),allowance)
    transport.members=[]
    with access.exclusive_scene_access():
        event=restore.append('restoring',member_key='0',evidence={'generation_id':retired_generation['generation_id']})
        restoring=engine._transition(policy,retired_generation,state='restoring',token=journal.token,journal_ref=event)
        if mode=='restore_generation_changed':
            target,foreign=caches[0]
            foreign=json.loads(target.read_bytes())
            foreign['source_publication_raw_ref']=consent['intent_raw_ref']
            _sealed_file(target,foreign,'state_digest',mode=0o600)
            monkeypatch.setattr(transport,'read_archive',lambda *args:pytest.fail('unproved generation entered restore read'))
            with pytest.raises(ValueError,match='scene_retirement_generation_changed'):
                engine._finish_restore(policy,restore_consent,snapshot,receipt['retired_journal_raw_ref'],
                    restore,pending,context,[restoring],[],allowance,transport)
            assert not root.exists() and all(not Path(row['canonical_path']).exists() for row in consent['cache_objects'])
            return
        if mode=='restore_combined_cap':
            from blueprint_pipeline import task_evaluation_scene_retirement_journal as journal_module
            from blueprint_pipeline.task_evaluation_scene_retirement_restore import restore_records
            folder_rows=list(restore_records(preserved,restore))
            cache_rows=list(engine._cache_restore_records(snapshot))
            monkeypatch.setattr(journal_module,'MAX_EVENTS',restore.sequence+max(len(folder_rows),len(cache_rows)))
            monkeypatch.setattr(transport,'read_archive',lambda *args:pytest.fail('underreserved restore entered payload read'))
            with pytest.raises(ValueError,match='scene_retirement_journal_limit'):
                engine._finish_restore(policy,restore_consent,snapshot,receipt['retired_journal_raw_ref'],
                    restore,pending,context,[restoring],[],allowance,transport)
            assert not root.exists()
            assert all(json.loads(file.read_bytes())['state']=='retired' for file,_ in caches)
            return
        result=engine._finish_restore(policy,restore_consent,snapshot,receipt['retired_journal_raw_ref'],
            restore,pending,context,[restoring],[],allowance,transport)
    assert result['status']=='restored' and root.is_dir()
    for file,original in caches:
        value=json.loads(file.read_bytes())
        alias=Path(original['canonical_path'])
        assert value['state']=='restored-active', 'engine restored cache bytes without activating exact generation'
        assert value['generation_id']==original['generation_id']
        assert (value['dev'],value['ino'],value['mode'])==(alias.stat().st_dev,alias.stat().st_ino,alias.stat().st_mode)
        source=next(row for row in preserved['cache_aliases'] if row['path']==str(alias))
        projection=root/source['relative_path']
        assert alias.stat().st_ino==projection.stat().st_ino
        assert alias.read_bytes()==projection.read_bytes()
        assert (alias.stat().st_uid,alias.stat().st_gid)==(source['uid'],source['gid'])



def test_cache_action_selects_exact_native_generated_publication_without_request_digest_waiver(tmp_path,monkeypatch):
    from tests.test_scene_retirement_generated_publication import generated_fixture,extract,generation
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    fixture=generated_fixture(tmp_path,monkeypatch)
    manifest,_=extract(fixture)
    leaf,current=generation(fixture,manifest['entries'][0]['sha256'])
    os.utime(leaf,(101-24*60*60,101-24*60*60))
    ledger=Path(fixture['policy']['generation_store'])/(hashlib.sha256(str(leaf).encode()).hexdigest()+'.json')
    row=dict(canonical_path=str(leaf),digest=current['digest'],size_bytes=current['size_bytes'],
        generation_id=current['generation_id'],generation_raw_ref=_raw(ledger),
        source_raw_ref=current['source_publication_raw_ref'])
    consent=dict(intent_raw_ref=fixture['proofs']['intent_raw_ref'],cache_objects=[row])
    selected=cache.validate_cache_objects(fixture['policy'],consent,
        ActionAllowance(expires_at=1000,now=lambda:101,monotonic=lambda:0))
    assert selected==[row]
    # The generated digest was never an original direct request reference.
    source=json.loads(Path(row['source_raw_ref']['path']).read_bytes())
    original=json.loads(Path(source['storage_authority_raw_ref']['path']).read_bytes())
    request=json.loads(Path(original['submission_request_raw_ref']['path']).read_bytes())
    from blueprint_pipeline.task_evaluation_launch_preparation_worker import collect_preparation_references
    assert all(ref['digest']!=row['digest'] for ref in collect_preparation_references(request))
    fixture['bundle'].unlink()
    fixture['bundle'].write_bytes(b'foreign native source cannot borrow generated digest')
    with pytest.raises(ValueError):
        cache.validate_cache_objects(fixture['policy'],consent,
            ActionAllowance(expires_at=1000,now=lambda:101,monotonic=lambda:0))


def test_normal_runtime_layer_cas_has_actual_wrapper_proof_before_derived_action(tmp_path,monkeypatch):
    from tests.test_scene_retirement_generated_publication import generated_fixture
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline import task_evaluation_scene_retirement_generated as generated
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    fixture=generated_fixture(tmp_path,monkeypatch,external=True)
    projection=next(iter(fixture['external_layers'].values()))
    source_root=projection.parent.parent/'content-addressed'/'sha256'
    leaf=source_root/projection.name
    os.utime(leaf,(101-24*60*60,101-24*60*60))
    key=hashlib.sha256(str(leaf).encode()).hexdigest()+'.json'
    ledger=Path(fixture['policy']['generation_store'])/key
    current=json.loads(ledger.read_bytes())
    publication=json.loads(Path(current['source_publication_raw_ref']['path']).read_bytes())
    assert publication['schema_version']==generated.SCHEMA, 'normal dynamic layer lacks native wrapper publication'
    assert publication['role']=='runtime_source' and publication['entry']['sha256']==current['digest']
    row=dict(canonical_path=str(leaf),digest=current['digest'],size_bytes=current['size_bytes'],
        generation_id=current['generation_id'],generation_raw_ref=_raw(ledger),
        source_raw_ref=current['source_publication_raw_ref'])
    assert cache.validate_cache_objects(fixture['policy'],dict(
        intent_raw_ref=fixture['proofs']['intent_raw_ref'],cache_objects=[row]),
        ActionAllowance(expires_at=1000,now=lambda:101,monotonic=lambda:0))==[row]
