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


def generated_fixture(tmp_path, monkeypatch, *, external=False):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_preparation_child
    from blueprint_pipeline import task_evaluation_native_arena_preparation_adapter as adapter
    access, policy, _, _, queue, inputs, proofs = owner_submission(tmp_path, monkeypatch)
    value, payloads = request_with_fetchable_bytes(request())
    intent = json.loads(Path(proofs['intent_raw_ref']['path']).read_bytes())
    attempt = json.loads(Path(proofs['attempt_raw_ref']['path']).read_bytes())
    value['expected_production_commit'] = attempt['source_commit']
    value['scene_intent_digest'] = intent['intent_digest']
    value['task']['identity']['id'] = intent['request']['task']['task_id']
    runtime_build = None
    if external:
        runtime_source = tmp_path/'factory'/'runtime-source'
        runtime_source.mkdir()
        (runtime_source/'runtime.py').write_bytes(b'# actual tiny native release payload\n')
        runtime_build = adapter.build_task_evaluation_runtime_source_bundle(
            source_root=runtime_source, output_path=tmp_path/'factory'/'runtime.zip',
            expected_production_commit=value['expected_production_commit'],
            runtime_identity=value['runtime']['identity'],
            external_layer_store_root=tmp_path/'factory'/'layers',
            external_layer_bucket='blueprint-production-inputs', external_layer_min_bytes=1)
        uri = 's3://blueprint-production-inputs/runtime.zip'
        value['execution_adapter']['runtime_source_bundle'] = dict(uri=uri,
            digest=runtime_build['sha256'], size_bytes=runtime_build['size_bytes'])
        payloads[uri] = Path(runtime_build['path']).read_bytes()
        for row in runtime_build['external_layers']:
            payloads[row['uri']] = Path(row['store_path']).read_bytes()
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
    materialized = None
    if external:
        from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
        from tests.test_task_evaluation_launch_preparation_worker import fetcher, SERVICE_ACCOUNT
        prep = inputs/value['preparation_id']
        native_store = inputs/'content-addressed'/'sha256'
        materialized = worker.materialize_preparation_references(request=value, input_root=prep,
            content_store_root=native_store, allowed_uri_prefixes=['s3://blueprint-production-inputs/'],
            service_account=SERVICE_ACCOUNT, source_commit=value['expected_production_commit'], fetcher=fetcher(payloads))
        runtime_row = next(row for row in materialized['references']
                           if row['contract_path'] == 'execution_adapter.runtime_source_bundle')
        layers = worker._materialize_runtime_source_external_layers(request=value, runtime_source=runtime_row,
            input_root=prep, content_store_root=native_store,
            allowed_uri_prefixes=['s3://blueprint-production-inputs/'], fetcher=fetcher(payloads),
            disk_reservation_root=None, disk_reservations=[])
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
    if external:
        bundle = Path(runtime_row['materialized_path'])
        reference = value['execution_adapter']['runtime_source_bundle']
    return dict(access=access, policy=policy, request=value, authority=selected, producer=producer,
        bundle=bundle, reference=reference, store=store, adapter=adapter, proofs=proofs,
        external_layers={} if not external else {row['digest']: Path(row['materialized_path']) for row in layers},
        role='runtime_source' if external else 'construction_packet')


def extract(fixture, name='first'):
    return fixture['adapter']._extract_verified_bundle(bundle_path=fixture['bundle'],
        request=fixture['request'], expected_reference=fixture['reference'], role=fixture['role'],
        destination=fixture['producer']/name, content_store_root=fixture['store'],
        external_layers=fixture['external_layers'])


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


def test_actual_registered_runtime_external_member_keeps_native_zero_copy(tmp_path, monkeypatch):
    fixture = generated_fixture(tmp_path, monkeypatch, external=True)
    original = next(iter(fixture['external_layers'].values()))
    manifest, destination = extract(fixture)
    row = manifest['entries'][0]
    leaf, current = generation(fixture, row['sha256'])
    assert leaf.stat().st_ino == original.stat().st_ino == (destination/'runtime.py').stat().st_ino
    publication = json.loads(Path(current['source_publication_raw_ref']['path']).read_bytes())
    assert publication['external_source_raw_ref'] == _raw(original)
    assert publication['external_generation_raw_ref'] is not None
    assert json.loads(Path(publication['external_generation_raw_ref']['path']).read_bytes())['ino'] == original.stat().st_ino


@pytest.mark.parametrize('operation', ['whole_source_hash', 'native_zip_read'])
def test_action_source_byte_refusal_precedes_actual_native_read(tmp_path, monkeypatch, operation):
    fixture = generated_fixture(tmp_path, monkeypatch)
    from blueprint_pipeline import task_evaluation_scene_retirement_generated as generated
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    import os
    calls = []
    native = os.read
    with pytest.raises(ValueError):
        with generated.bundle_lifetime(bundle_path=fixture['bundle'], request=fixture['request'],
                expected_reference=fixture['reference'], role=fixture['role'],
                destination=fixture['producer']/'budget-test', content_store_root=fixture['store']) as use:
            use.allowance = ActionAllowance(expires_at=1000, now=lambda: 101, local_bytes=0)
            def observed(fd, size):
                if fd == use.fd:
                    calls.append(size)
                return native(fd, size)
            monkeypatch.setattr(os, 'read', observed)
            if operation == 'whole_source_hash':
                use.verify_reference()
            else:
                generated._ArchiveReader(use).read(1)
    assert calls == [], 'physical native source read occurred after the original byte budget was exhausted'


def test_generated_manifest_publication_is_immutable_only_after_full_write(tmp_path, monkeypatch):
    fixture = generated_fixture(tmp_path, monkeypatch)
    import os
    native = os.write
    ledger = Path(fixture['policy']['generation_store'])
    seen = []
    def observed(fd, raw):
        for path in ledger.glob('generated-manifest-*.json'):
            current, expected = os.fstat(fd), path.stat()
            assert (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino), \
                'partial manifest is exposed under its final immutable selector'
        seen.append(len(raw))
        return native(fd, raw)
    monkeypatch.setattr(os, 'write', observed)
    extract(fixture)
    assert seen and list(ledger.glob('generated-manifest-*.json'))


def test_actual_producer_metadata_uses_service_store_not_root_authority_mode(tmp_path, monkeypatch):
    fixture = generated_fixture(tmp_path, monkeypatch)
    from blueprint_pipeline import task_evaluation_scene_retirement_generated as generated
    from contextlib import contextmanager
    original = generated._opened
    store = Path(fixture['policy']['generation_store'])
    @contextmanager
    def observed(path, *, directory=False, protected=False):
        if Path(path) == store or Path(path).parent == store:
            assert protected is False, 'blueprint-owned producer evidence cannot require root ownership'
        with original(path, directory=directory, protected=protected) as selected:
            yield selected
    monkeypatch.setattr(generated, '_opened', observed)
    extract(fixture)


def test_actual_generated_temporary_proves_original_fd_before_first_write(tmp_path, monkeypatch):
    fixture = generated_fixture(tmp_path, monkeypatch)
    import os
    native_open, native_stat, native_write = os.open, os.fstat, os.write
    created, proved = set(), set()
    def opened(path, flags, *args, **kwargs):
        fd = native_open(path, flags, *args, **kwargs)
        if flags & os.O_CREAT and flags & os.O_WRONLY and '.partial-' in str(path):
            created.add(fd)
        return fd
    def observed(fd):
        value = native_stat(fd)
        if fd in created:
            proved.add(fd)
        return value
    def written(fd, raw):
        assert fd not in created or fd in proved, 'generated payload write used an unadopted numeric descriptor'
        return native_write(fd, raw)
    monkeypatch.setattr(os, 'open', opened)
    monkeypatch.setattr(os, 'fstat', observed)
    monkeypatch.setattr(os, 'write', written)
    extract(fixture)
    assert created and created <= proved


def test_generated_first_fd_named_mismatch_never_writes_or_closes_unknown_token(tmp_path, monkeypatch):
    fixture = generated_fixture(tmp_path, monkeypatch)
    import os
    native_open, native_stat, native_write, native_close = os.open, os.fstat, os.write, os.close
    selected = {}
    writes, closes = [], []
    def opened(path, flags, *args, **kwargs):
        fd = native_open(path, flags, *args, **kwargs)
        if flags & os.O_CREAT and flags & os.O_WRONLY and '.partial-' in str(path):
            target = Path(path)
            if not target.is_absolute():
                target = fixture['store']/target.name
            selected.update(fd=fd, identity=(native_stat(fd).st_dev, native_stat(fd).st_ino), path=target)
            target.unlink()
            target.write_bytes(b'foreign named replacement')
        return fd
    def written(fd, raw):
        if fd == selected.get('fd'):
            writes.append(len(raw))
        return native_write(fd, raw)
    def closed(fd):
        if fd == selected.get('fd'):
            closes.append(fd)
        return native_close(fd)
    monkeypatch.setattr(os, 'open', opened)
    monkeypatch.setattr(os, 'write', written)
    monkeypatch.setattr(os, 'close', closed)
    try:
        with pytest.raises(ValueError):
            extract(fixture)
        assert writes == [] and closes == [], 'first unproven numeric token was used or closed'
        assert selected['path'].read_bytes() == b'foreign named replacement'
    finally:
        # Only this test owns the independently recorded real acquisition.
        # Production has no adoption proof and must preserve that token.
        if 'fd' in selected:
            try:
                current = native_stat(selected['fd'])
            except OSError:
                pass
            else:
                if (current.st_dev, current.st_ino) == selected['identity']:
                    native_close(selected['fd'])
