# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_compilation_native_owners.py
#   src/blueprint_pipeline/task_evaluation_scene_compilation_native_owner_inventory.py
"""ADP-009D/day28: native owner joins original inputs, never runtime authority."""
import copy
import hashlib
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_compilation_owner_preparations import api, change, refuses
from tests.test_scene_compilation_owner_outputs import fixture as output_fixture
from tests.test_scene_inventory_preparations import fixture as seed_fixture
from tests.test_scene_source_family_website import pair, seal


def filename(activation, digest):
    readable = activation+'-'+digest[7:]+'.json'
    return readable if len(readable.encode()) <= 255 else 'activation-'+hashlib.sha256(activation.encode()).hexdigest()+'-'+digest[7:]+'.json'


def fixture(*, standalone=True, profile=True, activation_id='native-activation'):
    args = output_fixture()
    args['seed_records'] = seed_fixture()['records']
    records, bridge, roots = args['seed_records'], args['bridge_records'], args['roots']
    intent = json.loads(records['intent'][1])
    original = json.loads(records['preparation_envelopes'][0][1])
    native = json.loads(bridge['native_preparation_envelopes'][0][1])
    final = json.loads(bridge['native_preparation_results'][0][1])
    attempt = seal({'schema_version': 'task_evaluation_scene_attempt.v1', 'intent_id': args['intent_id'],
        'intent_digest': intent['intent_digest'], 'attempt_id': 'controls-1', 'source_commit': 'a'*40,
        'input_digest': original['request_digest'], 'runtime_digest': 'sha256:'+'b'*64,
        'provider': 'vast', 'maximum_spend_usd': 1.0}, 'attempt_digest', cross=True)
    records['attempts'] = [pair(roots['intent_root']+'/'+args['intent_id']+'/attempts/controls-1.json', attempt)]
    binding = {k: attempt[k] for k in ('intent_id', 'intent_digest', 'attempt_id', 'source_commit', 'input_digest', 'runtime_digest')}
    binding['schema_version'] = 'task_evaluation_scene_attempt_binding.v1'
    owner = seal({'schema_version': 'task_evaluation_scene_owner_attempt.v1', 'phase': 'controls',
        'team_namespace': 'team-1', 'scene_id': 'scene-1', 'task_id': 'task-1',
        'runtime_source_bundle_digest': native['request']['execution_adapter']['runtime_source_bundle']['digest'],
        'scene_intent_digest': intent['intent_digest'], 'scene_attempt_id': attempt['attempt_id'],
        'scene_attempt_binding': binding}, 'owner_attempt_digest')
    request = {'schema_version': 'task_evaluation_launch_activation_request.v1', 'activation_id': activation_id,
        'lane': 'native_task_arena_controls', 'team_namespace': 'team-1', 'expected_production_commit': 'a'*40,
        'preparation': {'preparation_id': final['preparation_id'], 'request_digest': native['request_digest'], 'result_digest': final['result_digest']},
        'authorization': {'scene_owner_attempt': owner},
        'release_window': {'uri': 's3://test/window.json', 'digest': 'sha256:'+'9'*64, 'size_bytes': 1}}
    digest = canonical_digest(request)
    envelope = seal({'schema_version': 'task_evaluation_launch_activation_envelope.v1', 'request': request,
        'request_digest': digest, 'provider_mutation_performed_inside_intake': False, 'catalog_mutation_performed_inside_intake': False,
        'standing_authorization_published_inside_intake': False, 'paid_execution_requested': False}, 'envelope_digest')
    name = filename(activation_id, digest)
    bridge['native_activation_envelopes'] = [pair(roots['activation_queue_root']+'/prepared/'+name, envelope)]
    if standalone:
        bridge['native_owner_records'] = [pair(roots['activation_output_root']+'/'+activation_id+'/scene_owner_attempt.json', owner)]
    profile_value = seal({'schema_version': 'task_evaluation_launch_profile.v1', 'profile_id': 'native-profile',
        'source_commit': 'a'*40, **{k: owner[k] for k in ('scene_intent_digest', 'scene_attempt_id', 'scene_attempt_binding')}}, 'profile_digest')
    result = seal({'schema_version': 'task_evaluation_launch_activation_result.v1', 'status': 'profile_authority_materialized_no_execution',
        'activation_id': activation_id, 'preparation_id': final['preparation_id'], 'team_namespace': 'team-1',
        'lane': request['lane'], 'source_commit': 'a'*40, 'preparation_result_digest': final['result_digest'],
        'release_window_digest': 'sha256:'+'8'*64, 'profile_id': profile_value['profile_id'], 'profile_digest': profile_value['profile_digest'],
        'profile_publication_receipt_digest': 'sha256:'+'7'*64, 'standing_authorization_digest': 'sha256:'+'6'*64,
        'full_byte_activation_reference_readback_passed': True, 'profile_publication_performed': True,
        'catalog_mutation_performed': True, 'standing_authorization_published': True,
        'provider_mutation_performed': False, 'paid_execution_requested': False, 'blockers': []}, 'result_digest')
    bridge['native_activation_results'] = [pair(roots['activation_queue_root']+'/results/'+name, result)]
    if profile:
        args['downstream_records']['launch_profiles'] = [pair(roots['launch_execution_root']+'/launch-native/launch_profile.json', profile_value)]
    return args


def owner_change(args, edit, *, standalone=False):
    if standalone:
        change(args, 'native_owner_records', edit, 'owner_attempt_digest')
        return
    path, raw = args['bridge_records']['native_activation_envelopes'][0]
    envelope = json.loads(raw)
    owner = envelope['request']['authorization']['scene_owner_attempt']
    edit(owner)
    envelope['request']['authorization']['scene_owner_attempt'] = seal(owner, 'owner_attempt_digest')
    envelope['request_digest'] = canonical_digest(envelope['request'])
    new_name = filename(envelope['request']['activation_id'], envelope['request_digest'])
    args['bridge_records']['native_activation_envelopes'] = [pair(path.rsplit('/', 1)[0]+'/'+new_name, seal(envelope, 'envelope_digest'))]
    result_path, result_raw = args['bridge_records']['native_activation_results'][0]
    args['bridge_records']['native_activation_results'] = [(result_path.rsplit('/', 1)[0]+'/'+new_name, result_raw)]


@pytest.mark.parametrize('standalone,profile', [(True, True), (False, True), (True, False)])
def test_native_owner_bridge_keeps_original_input_and_distinct_payload_bundle_meanings(standalone, profile):
    args = fixture(standalone=standalone, profile=profile)
    before = copy.deepcopy(args)
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    row = next(r for r in result['compilation_native_owner_observations'] if r.get('kind') == 'native_owner')
    assert row['owner_metadata_binding_verified'] and row['profile_metadata_binding_verified'] is profile
    assert any(p.get('json_pointer') == '/request/authorization/scene_owner_attempt' for p in row['source_provenance'])
    assert not any(r['role'] == 'embedded_owner_raw' for r in result['raw_versions'])
    assert result['cleanup_authorized'] is False and args == before


@pytest.mark.parametrize('edit', [
    lambda v: v.update(task_id='foreign'), lambda v: v.update(scene_id='foreign'),
    lambda v: v.update(team_namespace='foreign'), lambda v: v.update(phase='construction'),
    lambda v: v.update(runtime_source_bundle_digest='sha256:'+'1'*64),
    lambda v: v['scene_attempt_binding'].update(intent_id='foreign'),
    lambda v: v['scene_attempt_binding'].update(input_digest='sha256:'+'1'*64),
    lambda v: v['scene_attempt_binding'].update(runtime_digest='sha256:'+'1'*64),
])
def test_available_owner_contradictions_refuse_with_missing_compiler_output(edit):
    args = fixture(standalone=False, profile=False)
    args['bridge_records']['compiler_outputs'] = []
    owner_change(args, edit)
    refuses(args)


def test_standalone_owner_semantic_disagreement_refuses():
    args = fixture()
    owner_change(args, lambda v: v.update(phase='construction'), standalone=True)
    refuses(args)


def test_existing_downstream_activation_result_reused_once_and_long_activation_uses_hashed_filename():
    args = fixture(activation_id='x'*192)
    args['downstream_records']['activation_results'] = args['bridge_records']['native_activation_results']
    args['bridge_records']['native_activation_results'] = []
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert next(r for r in result['compilation_native_owner_observations'] if r.get('kind') == 'native_owner')['owner_metadata_binding_verified']
    assert result['source_family_inventory']['downstream_inventory']['activation_observations'][0]['status'] == 'kept_unresolved'
