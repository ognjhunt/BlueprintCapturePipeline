# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_generated.py
#   src/blueprint_pipeline/task_evaluation_native_arena_preparation_adapter.py
#   src/blueprint_pipeline/task_evaluation_native_arena_episode_compiler.py
"""ADP-009D/day28: actual generated members retain authenticated native provenance."""
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_scene_retirement_connected_acceptance import _raw, _sealed_file
from tests.test_scene_retirement_normal_cache import owner_submission
from tests.test_task_evaluation_launch_preparation_contract import request
from tests.test_task_evaluation_launch_preparation_worker import request_with_fetchable_bytes


def generated_fixture(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_preparation_child
    from blueprint_pipeline import task_evaluation_native_arena_preparation_adapter as adapter
    access, policy, _, _, queue, inputs, proofs = owner_submission(tmp_path, monkeypatch)
    value, _ = request_with_fetchable_bytes(request())
    intent = json.loads(Path(proofs['intent_raw_ref']['path']).read_bytes())
    attempt = json.loads(Path(proofs['attempt_raw_ref']['path']).read_bytes())
    value['expected_production_commit'] = attempt['source_commit']
    value['scene_intent_digest'] = intent['intent_digest']
    value['task']['identity']['id'] = intent['request']['task']['task_id']
    selected_request = inputs.parent / 'factory' / 'generated_request.json'
    selected_request.write_text(json.dumps(value))
    factory = selected_request.with_name('generated_factory.json')
    _sealed_file(factory, dict(schema_version='website_scene_attempt_factory.v1',
        status='publication_ready', intent_digest=intent['intent_digest'],
        attempt_digest=attempt['attempt_digest'], source_commit=attempt['source_commit'],
        submission_request=_raw(selected_request), provider_mutation_performed=False), 'factory_digest')
    proofs = dict(proofs, factory_raw_ref=_raw(factory), submission_request_raw_ref=_raw(selected_request))
    monkeypatch.setattr(cache.time, 'time', lambda: 101)
    selected = cache.publish_preparation_storage_authority(queue_root=queue, request=value, now=101, **proofs)
    queued = stage_launch_preparation_request(value=value, queue_root=queue, submitted_by='scene-progression')
    cache.enroll_preparation_storage(queue_path=queued['queue_path'], input_root=inputs, now=101)
    producer = inputs / 'compiled-owned-episode'
    born = enroll_preparation_child(producer, preparation_root=inputs/value['preparation_id'],
        request=value, now=101)
    assert born['source_storage_authority_raw_ref'] == selected
    payload = producer / 'native-packet-source'
    payload.mkdir()
    (payload/'member.json').write_bytes(b'{"native":"tiny generated packet"}')
    bundle = producer/'compiled-packet.zip'
    built = adapter.build_task_evaluation_adapter_bundle(source_root=payload, output_path=bundle,
        request=value, role='construction_packet')
    reference = dict(uri='production-internal://compiled-episode-packet', digest=built['sha256'],
                     size_bytes=built['size_bytes'])
    store = inputs/'content-addressed'/'adapter-members'/'sha256'
    return dict(access=access, policy=policy, request=value, authority=selected, producer=producer,
        bundle=bundle, reference=reference, store=store, adapter=adapter, proofs=proofs)


def extract(fixture, name='first'):
    return fixture['adapter']._extract_verified_bundle(bundle_path=fixture['bundle'],
        request=fixture['request'], expected_reference=fixture['reference'], role='construction_packet',
        destination=fixture['producer']/name, content_store_root=fixture['store'])


def generation(fixture, digest):
    leaf = fixture['store']/digest[7:]
    name = hashlib.sha256(str(leaf).encode()).hexdigest()+'.json'
    path = Path(fixture['policy']['generation_store'])/name
    assert path.is_file(), 'actual generated CAS publication lacks authenticated regular generation'
    return leaf, json.loads(path.read_bytes())


def test_actual_native_generated_member_is_registered_before_final_cas_name(tmp_path, monkeypatch):
    fixture = generated_fixture(tmp_path, monkeypatch)
    import os
    original = os.link
    publications = []
    def observe(source, target, *args, **kwargs):
        target = Path(target)
        directory = kwargs.get('dst_dir_fd')
        if directory is not None and fixture['store'].exists():
            observed, expected = os.fstat(directory), fixture['store'].stat()
            if (observed.st_dev, observed.st_ino) == (expected.st_dev, expected.st_ino):
                target = fixture['store']/target.name
        if target.parent == fixture['store'] and not target.name.startswith('.'):
            key = hashlib.sha256(str(target).encode()).hexdigest()+'.json'
            assert (Path(fixture['policy']['generation_store'])/key).is_file(), \
                'final CAS name appeared before protected birth publication'
            publications.append(str(target))
        return original(source, target, *args, **kwargs)
    monkeypatch.setattr(os, 'link', observe)
    manifest, destination = extract(fixture)
    row = manifest['entries'][0]
    leaf, current = generation(fixture, row['sha256'])
    assert publications == [str(leaf)]
    assert current['state'] == 'active' and current['digest'] == row['sha256']
    publication = json.loads(Path(current['source_publication_raw_ref']['path']).read_bytes())
    assert publication['schema_version'] == 'scene_generated_content_publication.v1'
    assert publication['storage_authority_raw_ref'] == fixture['authority']
    assert publication['bundle_raw_ref'] == _raw(fixture['bundle'])
    assert publication['producer_root'] == str(fixture['producer'])
    assert publication['entry'] == row
    assert leaf.stat().st_ino == (destination/'member.json').stat().st_ino


def test_actual_generated_cache_reuse_preserves_exact_birth_and_owner(tmp_path, monkeypatch):
    fixture = generated_fixture(tmp_path, monkeypatch)
    manifest, first = extract(fixture)
    leaf, before = generation(fixture, manifest['entries'][0]['sha256'])
    _, second = extract(fixture, 'second')
    _, after = generation(fixture, manifest['entries'][0]['sha256'])
    assert before == after
    assert leaf.stat().st_ino == (first/'member.json').stat().st_ino == (second/'member.json').stat().st_ino


def test_generated_validator_reselects_native_bundle_and_original_owner(tmp_path, monkeypatch):
    fixture = generated_fixture(tmp_path, monkeypatch)
    manifest, _ = extract(fixture)
    _, current = generation(fixture, manifest['entries'][0]['sha256'])
    from blueprint_pipeline import task_evaluation_scene_retirement_generated as generated
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    publication = json.loads(Path(current['source_publication_raw_ref']['path']).read_bytes())
    allowance = ActionAllowance(expires_at=1000, now=lambda: 101)
    verified = generated.validate_publication(publication, policy=fixture['policy'],
        consent={'intent_raw_ref': fixture['proofs']['intent_raw_ref']}, allowance=allowance)
    assert verified['digest'] == manifest['entries'][0]['sha256']
    assert verified['size_bytes'] == manifest['entries'][0]['size_bytes']
    assert verified['intent_raw_ref'] == fixture['proofs']['intent_raw_ref']
    fixture['bundle'].unlink()
    fixture['bundle'].write_bytes(b'foreign archive replaces protected native source')
    allowance = ActionAllowance(expires_at=1000, now=lambda: 101)
    with pytest.raises(ValueError):
        generated.validate_publication(publication, policy=fixture['policy'],
            consent={'intent_raw_ref': fixture['proofs']['intent_raw_ref']}, allowance=allowance)


def test_existing_unregistered_generated_cas_bytes_are_never_adopted(tmp_path, monkeypatch):
    fixture = generated_fixture(tmp_path, monkeypatch)
    fixture['store'].mkdir(parents=True)
    payload = b'{"native":"tiny generated packet"}'
    digest = 'sha256:'+hashlib.sha256(payload).hexdigest()
    leaf = fixture['store']/digest[7:]
    leaf.write_bytes(payload)
    leaf.chmod(0o440)
    extract(fixture)
    path = Path(fixture['policy']['generation_store'])/(hashlib.sha256(str(leaf).encode()).hexdigest()+'.json')
    assert not path.exists(), 'old bytes are not a newly authenticated creation'


@pytest.mark.parametrize('changed', ['bundle', 'expected_reference', 'request'])
def test_native_source_or_identity_refusal_precedes_any_generated_publication(tmp_path, monkeypatch, changed):
    fixture = generated_fixture(tmp_path, monkeypatch)
    if changed == 'bundle':
        fixture['bundle'].unlink()
        fixture['bundle'].write_bytes(b'bad zip')
    elif changed == 'expected_reference':
        fixture['reference']['digest'] = 'sha256:'+'f'*64
    else:
        fixture['request']['runtime']['identity']['version'] = 'foreign-version'
    with pytest.raises((fixture['adapter'].TaskEvaluationNativeArenaAdapterError,
                        fixture['access'].SceneRetirementAccessError)):
        extract(fixture)
    assert not fixture['store'].exists()
