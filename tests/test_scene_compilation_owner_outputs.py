# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_compilation_owner_outputs.py
#   src/blueprint_pipeline/task_evaluation_scene_compilation_native_owner_inventory.py
"""ADP-009D/day28: raw packets and canonical adapter selectors stay distinct."""
import copy
import json

import pytest

from tests.test_scene_compilation_owner_preparations import api, change, fixture as prep_fixture, refuses
from tests.test_scene_source_family_website import pair, seal


def fixture(*, output=True, adapter=True):
    args = prep_fixture()
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
