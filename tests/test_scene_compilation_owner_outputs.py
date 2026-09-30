# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_compilation_owner_outputs.py
#   src/blueprint_pipeline/task_evaluation_scene_compilation_native_owner_inventory.py
"""ADP-009D/day28: raw packets and canonical adapter selectors stay distinct."""
import copy
import json

import pytest

from tests.test_scene_compilation_owner_preparations import api, change, fixture as prep_fixture, refuses
from tests.test_scene_source_family_website import pair, seal


def fixture(*, output=True, adapter=True, mode='episode_evaluation'):
    args = prep_fixture(mode=mode)
    roots, bridge = args['roots'], args['bridge_records']
    envelope = json.loads(args['downstream_records']['compilation_envelopes'][0][1])
    root = roots['compilation_output_root']+'/'+envelope['compilation_id']
    value = seal({'schema_version': 'task_evaluation_native_arena_adapter_result.v1',
        'status': 'native_arena_adapter_materialized', 'preparation_id': envelope['compilation_id'],
        'source_commit': 'a'*40, 'adapter_kind': 'native_task_arena', 'adapter_version': 'v1',
        'configured_scene_revision_digest': envelope['configured_scene_revision_digest'],
        **{k: 'sha256:'+'e'*64 for k in ('construction_manifest_digest', 'runtime_source_manifest_digest',
            'packet_receipt_digest', 'runtime_source_receipt_digest')},
        'packet_root': root+'/native-arena-adapter/packet',
        'runtime_source_receipt': root+'/native-arena-adapter/runtime/native_task_runtime_source_packet.v1.json',
        'provider_mutation_performed': False, 'catalog_mutation_performed': False, 'paid_execution_requested': False}, 'result_digest')
    adapter_path = root+'/native-arena-adapter/task_evaluation_native_arena_adapter_result.v1.json'
    bridge['compilation_adapter_results'] = [pair(adapter_path, value)] if adapter else []
    packet = {'format': 'native_task_arena_bundle_zip', 'path': root+'/native-task-arena-bundle.zip',
        'digest': 'sha256:'+'f'*64, 'size_bytes': 5}
    output_value = seal({'schema_version': 'task_evaluation_episode_compiler_output.v1', 'status': 'completed',
        'run_id': envelope['run_id'], 'configured_scene_revision_digest': envelope['configured_scene_revision_digest'],
        'configured_task_template_adapter': {'schema_version': 'retained_adapter.v1', 'adapter_digest': 'sha256:'+'1'*64,
            'source_documents_digest': 'sha256:'+'2'*64, 'manipulation_strategy': 'retained'},
        'compiled_episode_packet': packet, 'adapter_result': {'path': adapter_path, 'digest': value['result_digest'],
            'packet_receipt_digest': value['packet_receipt_digest'], 'runtime_source_receipt_digest': value['runtime_source_receipt_digest']},
        'native_scene_appearance': {}, 'compiled_by_production': True, 'customer_supplied_prebuilt_episode_packet': False,
        'provider_mutation_performed': False, 'paid_execution_requested': False, 'raw_secret_values_recorded': False}, 'compiler_output_digest')
    if mode == 'destination_qualification':
        output_value = seal(dict(output_value, destination_native_probe_request={'path': root+'/probe.json',
            'digest': 'sha256:'+'3'*64, 'size_bytes': 6, 'request_digest': 'sha256:'+'4'*64}), 'compiler_output_digest')
    bridge['compiler_outputs'] = [pair('/retained/metadata/compiler.json', output_value)] if output else []
    result = seal({'schema_version': 'task_evaluation_episode_compilation_result.v1', 'status': 'compiled_for_production_launch',
        'compilation_id': envelope['compilation_id'], 'run_id': envelope['run_id'], 'team_namespace': envelope['team_namespace'],
        'source_commit': 'a'*40, 'configured_scene_revision_digest': envelope['configured_scene_revision_digest'],
        'compiled_episode_packet_path': packet['path'], 'compiled_episode_packet_digest': packet['digest'],
        'compiled_episode_packet_size_bytes': packet['size_bytes'], 'adapter_result_path': adapter_path,
        'adapter_result_digest': value['result_digest'], 'compiler_output_digest': output_value['compiler_output_digest'],
        'customer_supplied_prebuilt_episode_packet': False, 'compiled_by_production': True,
        'provider_mutation_performed': False, 'paid_execution_requested': False, 'automatic_progression_required': True,
        'blockers': []}, 'result_digest')
    if mode == 'destination_qualification':
        probe = output_value['destination_native_probe_request']
        result = seal(dict(result, destination_native_probe_request_path=probe['path'],
            destination_native_probe_request_digest=probe['digest'], destination_native_probe_request_document_digest=probe['request_digest']), 'result_digest')
    filename = envelope['compilation_id']+'-'+envelope['envelope_digest'][7:]+'.json'
    args['downstream_records']['compilation_results'] = [pair(roots['compilation_queue_root']+'/results/'+filename, result)]
    return args


@pytest.mark.parametrize('output,adapter', [(True, True), (False, True), (True, False), (False, False)])
def test_compilation_metadata_edges_preserve_raw_and_canonical_proof_distinction(output, adapter):
    args = fixture(output=output, adapter=adapter)
    before = copy.deepcopy(args)
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    observation = next(r for r in result['compilation_native_owner_observations'] if r.get('kind') == 'compilation_output')
    assert observation['adapter_metadata_binding_verified'] is adapter
    assert observation['compiler_output_metadata_binding_verified'] is output
    packet = next(r for r in result['declared_lexical_members'] if r['kind'] == 'compiled_episode_packet')
    assert packet['binding']['size_bytes'] == 5 and not packet['presence_checked']
    assert packet['measured_bytes'] is None and args == before
    if not output:
        assert any(r['role'] == 'compiler_output' for r in result['structural_join_obligations'])
    assert not any(r.get('sha256') == json.loads(args['downstream_records']['compilation_results'][0][1])['adapter_result_digest']
        for r in result['raw_reference_obligations'])


@pytest.mark.parametrize('role,field,edit', [
    ('compilation_adapter_results', 'result_digest', {'source_commit': 'b'*40}),
    ('compilation_adapter_results', 'result_digest', {'packet_root': '/foreign/packet'}),
    ('compilation_adapter_results', 'result_digest', {'paid_execution_requested': True}),
    ('compiler_outputs', 'compiler_output_digest', lambda v: v['compiled_episode_packet'].update(size_bytes=True)),
    ('compiler_outputs', 'compiler_output_digest', lambda v: v['adapter_result'].update(path='/foreign/adapter.json')),
])
def test_known_output_self_contradictions_refuse_even_without_other_outputs(role, field, edit):
    args = fixture()
    change(args, role, edit, field)
    if role == 'compilation_adapter_results':
        args['bridge_records']['compiler_outputs'] = []
        updated = json.loads(args['bridge_records'][role][0][1])
        change(args, 'compilation_results', {'adapter_result_digest': updated['result_digest']}, 'result_digest', family='downstream_records')
    refuses(args)


def test_future_output_schema_is_retained_without_current_metadata_promotion():
    args = fixture()
    change(args, 'compiler_outputs', {'schema_version': 'future.v2'}, 'compiler_output_digest')
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert any(r['role'] == 'compiler_outputs' for r in result['raw_versions'])
    assert not next(r for r in result['compilation_native_owner_observations'] if r.get('kind') == 'compilation_output')['compiler_output_metadata_binding_verified']


@pytest.mark.parametrize('role', ['compiler_outputs', 'compilation_adapter_results'])
def test_selected_unknown_extension_keeps_metadata_unproven_with_all_raw_proofs(role):
    args = fixture()
    field = 'compiler_output_digest' if role == 'compiler_outputs' else 'result_digest'
    change(args, role, {'future_extension': {'protected': True}}, field)
    changed = json.loads(args['bridge_records'][role][0][1])
    if role == 'compiler_outputs':
        change(args, 'compilation_results', {'compiler_output_digest': changed[field]}, 'result_digest', family='downstream_records')
    else:
        change(args, 'compilation_results', {'adapter_result_digest': changed[field]}, 'result_digest', family='downstream_records')
        change(args, 'compiler_outputs', lambda v: v['adapter_result'].update(digest=changed[field]), 'compiler_output_digest')
        output = json.loads(args['bridge_records']['compiler_outputs'][0][1])
        change(args, 'compilation_results', {'compiler_output_digest': output['compiler_output_digest']}, 'result_digest', family='downstream_records')
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    observation = next(r for r in result['compilation_native_owner_observations'] if r.get('kind') == 'compilation_output')
    key = 'compiler_output_metadata_binding_verified' if role == 'compiler_outputs' else 'adapter_metadata_binding_verified'
    assert not observation[key]
    assert any(r['reason'] == 'unsupported_retained_field_set' for r in result['structural_join_obligations'])
    assert any(r['role'] == role for r in result['raw_versions'])


@pytest.mark.parametrize('member', ['compiled_episode_packet', 'adapter_result', 'configured_task_template_adapter'])
def test_selected_nested_artifact_extension_is_protected(member):
    args = fixture()
    change(args, 'compiler_outputs', lambda v: v[member].update(future_extension='protected'), 'compiler_output_digest')
    updated = json.loads(args['bridge_records']['compiler_outputs'][0][1])
    change(args, 'compilation_results', {'compiler_output_digest': updated['compiler_output_digest']}, 'result_digest', family='downstream_records')
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert not next(r for r in result['compilation_native_owner_observations'] if r.get('kind') == 'compilation_output')['compiler_output_metadata_binding_verified']


@pytest.mark.parametrize('output', [False, True])
def test_destination_probe_raw_size_and_canonical_request_are_distinct(output):
    args = fixture(mode='destination_qualification', output=output)
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    if output:
        assert any(r['path'].endswith('/probe.json') and r['size_bytes'] == 6 for r in result['raw_reference_obligations'])
    else:
        assert any(r['role'] == 'destination_probe' and r['reason'] == 'raw_size_unavailable' for r in result['structural_join_obligations'])


def test_available_episode_mode_contradicts_destination_probe_even_when_adapter_absent():
    args = fixture(adapter=False)
    probe = {'path': args['roots']['compilation_output_root']+'/native-prep/probe.json',
        'digest': 'sha256:'+'3'*64, 'size_bytes': 6, 'request_digest': 'sha256:'+'4'*64}
    change(args, 'compiler_outputs', {'destination_native_probe_request': probe}, 'compiler_output_digest')
    output = json.loads(args['bridge_records']['compiler_outputs'][0][1])
    change(args, 'compilation_results', {'compiler_output_digest': output['compiler_output_digest'],
        'destination_native_probe_request_path': probe['path'], 'destination_native_probe_request_digest': probe['digest'],
        'destination_native_probe_request_document_digest': probe['request_digest']}, 'result_digest', family='downstream_records')
    refuses(args)


def test_reused_compilation_result_extension_keeps_new_output_bindings_unproven():
    args = fixture()
    change(args, 'compilation_results', {'future_writer_extension': True}, 'result_digest', family='downstream_records')
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    row = next(r for r in result['compilation_native_owner_observations'] if r.get('kind') == 'compilation_output')
    assert not row['adapter_metadata_binding_verified'] and not row['compiler_output_metadata_binding_verified']
    assert any(r['role'] == 'compilation_results' and r['reason'] == 'unsupported_retained_field_set'
        for r in result['structural_join_obligations'])


POINTERS = 'compilation_remote_output_pointers'


def _pointer(args, **changes):
    """The paid unit's ``<output root>/<id>.remote-output.v1.json`` for the fixture's compile (plan 14 §16)."""
    result_path, raw = args['downstream_records']['compilation_results'][0]
    result, root = json.loads(raw), args['roots']['compilation_output_root']
    compilation_id, name = result['compilation_id'], result_path.rsplit('/', 1)[1]
    cas = 's3://blueprint-artifacts/remote-cpu-output/sha256/'
    value = {'schema_version': 'remote_cpu_output_pointer.v1', 'stage': 'episode_compilation',
        'compilation_id': compilation_id, 'queue_row': {'queue': 'task-evaluation-episode-compilations', 'name': name,
            'envelope_digest': 'sha256:'+name[len(compilation_id)+1:-5]},
        'attempt_id': 'rcj-ec-'+'0'*24+'-a1-'+'1'*32, 'descriptor_digest': 'sha256:'+'5'*64, 'receipt_digest': 'sha256:'+'6'*64,
        'execution': {'provider': 'gcp_cloud_run_job', 'job': 'blueprint-remote-cpu-episode-compilation',
            'worker_identity': 'gcp-cloud-run:project/us-central1/blueprint-remote-cpu-episode-compilation/executions/run-1',
            'allocation_binding_digest': 'sha256:'+'7'*64, 'spend_consumption': 'sha256:'+'8'*64},
        'code': {'source_commit': 'a'*40, 'source_archive_digest': 'sha256:'+'9'*64,
            'image': 'us-central1-docker.pkg.dev/project/workers/remote-cpu@sha256:'+'d'*64, 'environment_digest': 'sha256:'+'c'*64},
        'archive': {'uri': cas+'0a'*32+'/blobs.tar', 'digest': 'sha256:'+'0a'*32, 'size_bytes': 10240},
        'index': {'uri': cas+'0b'*32+'/index.json', 'digest': 'sha256:'+'0b'*32, 'size_bytes': 900},
        'output_root': root+'/'+compilation_id, 'paths_total': 12, 'bytes_total': 50000,
        'host_known': {'count': 1, 'bytes': 10}, 'landed': {'subset': 'episode_compilation_consumer.v1', 'paths': 3, 'bytes': 900},
        'raw_references': [{'path': result['compiled_episode_packet_path'], 'digest': result['compiled_episode_packet_digest'],
            'size_bytes': result['compiled_episode_packet_size_bytes']}],
        'state': 'landed', 'teardown_receipt_digest': None, 'provider_zero_proven': False}
    value.update(changes)
    return pair(root+'/'+compilation_id+'.remote-output.v1.json', seal(value, 'pointer_digest'))


def _packet(result, packet_path):
    return [row for row in result['raw_reference_obligations'] if row['path'] == packet_path]


def test_census_accepts_a_remote_output_pointer_for_the_packet_raw_reference():
    """Plan 14 task 4.8: a remote compile lands only what consumers read, so its packet's bytes stay remote; the
    pointer that lists the packet's path, digest and size stands for them."""
    from blueprint_pipeline import task_evaluation_scene_compilation_owner_outputs as outputs

    assert 'remote_cpu_output_pointer.v1' in outputs.REMOTE_OUTPUT_POINTER_SCHEMAS
    args = fixture()
    packet = json.loads(args['downstream_records']['compilation_results'][0][1])['compiled_episode_packet_path']
    unpointed = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert {(r['status'], r['reason']) for r in _packet(unpointed, packet)} == {('kept_unresolved', 'reference_bytes_unavailable')}

    args['bridge_records'][POINTERS] = [_pointer(args)]
    before = copy.deepcopy(args)
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert args == before
    # Both edges that name the packet, the compile result and the compiler's output metadata, resolve to the pointer.
    rows = _packet(result, packet)
    assert len(rows) == 2 and {(r['status'], r['reason']) for r in rows} == {('matched_remote_output_pointer', None)}
    assert {r['matched_provenance']['path'] for r in rows} == {args['bridge_records'][POINTERS][0][0]}
    assert any(r['role'] == POINTERS for r in result['raw_versions'])
    # The pointer's archive and index are remote references like any other: the census never checks availability.
    assert {r['reason'] for r in result['remote_reference_obligations']
            if r['uri'].startswith('s3://blueprint-artifacts/remote-cpu-output/')} == {'remote_availability_unverified'}

    # A pointer that lists other bytes stands for nothing: the packet stays unresolved.
    other = fixture()
    other['bridge_records'][POINTERS] = [_pointer(other, raw_references=[
        {'path': packet, 'digest': 'sha256:'+'0'*64, 'size_bytes': 5}])]
    assert {r['status'] for r in _packet(api().join_retained_scene_compilation_native_owner_inventory(**other), packet)} == {
        'kept_unresolved'}
    # A pointer bound to another output, listing bytes outside its own, or from another stage contradicts itself.
    for changes in ({'output_root': args['roots']['compilation_output_root']+'/another-prep'},
                    {'raw_references': [{'path': '/retained/elsewhere/packet.zip', 'digest': 'sha256:'+'f'*64, 'size_bytes': 5}]},
                    {'stage': 'environment_probe'}):
        bad = fixture()
        bad['bridge_records'][POINTERS] = [_pointer(bad, **changes)]
        refuses(bad)
