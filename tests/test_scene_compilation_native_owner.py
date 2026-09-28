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


def test_exact_historical_format_copies_join_once_with_all_provenance():
    args = fixture()
    row = args['bridge_records']['native_activation_envelopes'][0]
    args['bridge_records']['native_activation_envelopes'].append((row[0], json.dumps(json.loads(row[1]), indent=1).encode()))
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    observations = [r for r in result['compilation_native_owner_observations'] if r.get('kind') == 'native_owner']
    assert len(observations) == 1
    assert len([p for p in observations[0]['source_provenance'] if p['role'] == 'native_activation_envelopes' and 'json_pointer' not in p]) == 2


@pytest.mark.parametrize('paid', [False, True])
def test_blocked_native_activation_self_scope_is_checked_without_other_proofs(paid):
    args = fixture()
    path = args['bridge_records']['native_activation_results'][0][0]
    value = seal({'schema_version': 'task_evaluation_launch_activation_result.v1', 'status': 'blocked',
        'activation_id': 'native-activation', 'blockers': ['retained_failure'],
        'provider_mutation_performed': False, 'paid_execution_requested': paid}, 'result_digest')
    args['bridge_records']['native_activation_results'] = [pair(path, value)]
    if paid:
        refuses(args)
    else:
        result = api().join_retained_scene_compilation_native_owner_inventory(**args)
        assert not next(r for r in result['compilation_native_owner_observations'] if r.get('kind') == 'native_owner')['profile_metadata_binding_verified']


def test_known_activation_result_own_filename_refuses_without_envelope():
    args = fixture()
    args['bridge_records']['native_activation_envelopes'] = []
    path, raw = args['bridge_records']['native_activation_results'][0]
    args['bridge_records']['native_activation_results'] = [(path.rsplit('/', 1)[0]+'/foreign-'+path.rsplit('-', 1)[1], raw)]
    refuses(args)


@pytest.mark.parametrize('missing', ['compiler_outputs', 'native_owner_records'])
def test_available_profile_owner_mismatch_cannot_hide_behind_unrelated_missing_bytes(missing):
    args = fixture()
    args['bridge_records'][missing] = []
    change(args, 'launch_profiles', lambda v: v['scene_attempt_binding'].update(intent_id='foreign'), 'profile_digest', family='downstream_records')
    profile = json.loads(args['downstream_records']['launch_profiles'][0][1])
    change(args, 'native_activation_results', {'profile_digest': profile['profile_digest']}, 'result_digest')
    refuses(args)


def test_native_input_selector_cannot_replace_original_configuration_input_even_with_matching_attempt():
    args = fixture(standalone=False, profile=False)
    digest = json.loads(args['bridge_records']['native_preparation_envelopes'][0][1])['request_digest']
    path, raw = args['seed_records']['attempts'][0]
    attempt = json.loads(raw)
    args['seed_records']['attempts'][0] = pair(path, seal(dict(attempt, input_digest=digest), 'attempt_digest', cross=True))
    owner_change(args, lambda v: v['scene_attempt_binding'].update(input_digest=digest))
    refuses(args)


def test_standalone_available_attempt_contradiction_refuses_without_activation_envelope():
    args = fixture(profile=False)
    args['bridge_records']['native_activation_envelopes'] = []
    owner_change(args, lambda v: v['scene_attempt_binding'].update(runtime_digest='sha256:'+'1'*64), standalone=True)
    refuses(args)


def rebind_activation_final(args):
    final = json.loads(args['bridge_records']['native_preparation_results'][0][1])
    path, raw = args['bridge_records']['native_activation_envelopes'][0]
    value = json.loads(raw)
    value['request']['preparation']['result_digest'] = final['result_digest']
    value['request_digest'] = canonical_digest(value['request'])
    name = filename(value['request']['activation_id'], value['request_digest'])
    args['bridge_records']['native_activation_envelopes'] = [pair(path.rsplit('/', 1)[0]+'/'+name, seal(value, 'envelope_digest'))]
    path, raw = args['bridge_records']['native_activation_results'][0]
    value = json.loads(raw)
    value['preparation_result_digest'] = final['result_digest']
    args['bridge_records']['native_activation_results'] = [pair(path.rsplit('/', 1)[0]+'/'+name, seal(value, 'result_digest'))]


@pytest.mark.parametrize('role,field', [('native_activation_envelopes', 'envelope_digest'),
    ('native_preparation_envelopes', 'envelope_digest'), ('native_preparation_results', 'result_digest'),
    ('native_activation_results', 'result_digest'), ('native_owner_records', 'owner_attempt_digest'),
    ('compilation_intake_receipts', 'receipt_digest')])
def test_finite_new_wrapper_extensions_keep_proofs_and_affected_bindings_unproven(role, field):
    args = fixture()
    change(args, role, {'future_writer_extension': {'keeps_until_owner_approval': True}}, field)
    if role == 'compilation_intake_receipts':
        receipt = json.loads(args['bridge_records'][role][0][1])
        change(args, 'native_preparation_results', {'episode_compilation_queue_receipt_digest': receipt['receipt_digest']}, 'result_digest')
        rebind_activation_final(args)
    elif role == 'native_preparation_results':
        rebind_activation_final(args)
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert any(r['role'] == role and r['reason'] == 'unsupported_retained_field_set' for r in result['structural_join_obligations'])
    assert any(r['role'] == role for r in result['raw_versions'])
    owner = next(r for r in result['compilation_native_owner_observations'] if r.get('kind') == 'native_owner')
    if role == 'native_activation_results':
        assert not owner['profile_metadata_binding_verified']
    elif role == 'compilation_intake_receipts':
        assert not result['preparation_handoff_observations'][0]['pre_handoff_binding_verified']
    else:
        assert not owner['owner_metadata_binding_verified']


def test_unsupported_native_envelope_still_checks_available_known_owner_contradictions():
    args = fixture(standalone=False, profile=False)
    change(args, 'native_activation_envelopes', {'future_writer_extension': True}, 'envelope_digest')
    owner_change(args, lambda v: v['scene_attempt_binding'].update(runtime_digest='sha256:'+'1'*64))
    refuses(args)


def test_finite_native_activation_request_extension_is_protected_without_deep_validation():
    args = fixture()
    path, raw = args['bridge_records']['native_activation_envelopes'][0]
    value = json.loads(raw)
    value['request']['future_writer_extension'] = {'protected': True}
    value['request_digest'] = canonical_digest(value['request'])
    name = filename(value['request']['activation_id'], value['request_digest'])
    args['bridge_records']['native_activation_envelopes'] = [pair(path.rsplit('/', 1)[0]+'/'+name, seal(value, 'envelope_digest'))]
    path, raw = args['bridge_records']['native_activation_results'][0]
    args['bridge_records']['native_activation_results'] = [(path.rsplit('/', 1)[0]+'/'+name, raw)]
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert any(r['reason'] == 'unsupported_retained_field_set' and r['source_provenance'][0].get('json_pointer') == '/request'
        for r in result['structural_join_obligations'])
    assert not next(r for r in result['compilation_native_owner_observations'] if r.get('kind') == 'native_owner')['owner_metadata_binding_verified']


def test_finite_embedded_owner_extension_is_protected_with_known_profile_fields_still_checked():
    args = fixture()
    owner_change(args, lambda v: v.update(future_writer_extension={'protected': True}))
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert not next(r for r in result['compilation_native_owner_observations'] if r.get('kind') == 'native_owner')['owner_metadata_binding_verified']
    assert any(r['reason'] == 'unsupported_retained_field_set' for r in result['structural_join_obligations'])


def test_reused_compilation_envelope_extension_protects_new_handoff_interpretation_only():
    args = fixture()
    path, raw = args['downstream_records']['compilation_envelopes'][0]
    value = seal(dict(json.loads(raw), future_writer_extension={'protected': True}), 'envelope_digest')
    name = value['compilation_id']+'-'+value['envelope_digest'][7:]+'.json'
    new_path = path.rsplit('/', 1)[0]+'/'+name
    args['downstream_records']['compilation_envelopes'] = [pair(new_path, value)]
    change(args, 'compilation_intake_receipts', {'envelope_digest': value['envelope_digest'], 'queue_path': new_path}, 'receipt_digest')
    intake = json.loads(args['bridge_records']['compilation_intake_receipts'][0][1])
    change(args, 'native_preparation_results', {'episode_compilation_queue_envelope_digest': value['envelope_digest'],
        'episode_compilation_queue_receipt_digest': intake['receipt_digest']}, 'result_digest')
    rebind_activation_final(args)
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert not result['preparation_handoff_observations'][0]['pre_handoff_binding_verified']
    assert any(r['role'] == 'compilation_envelopes' and r['reason'] == 'unsupported_retained_field_set'
        for r in result['structural_join_obligations'])


def test_reused_native_activation_result_extension_cannot_promote_profile():
    args = fixture()
    change(args, 'native_activation_results', {'future_writer_extension': True}, 'result_digest')
    args['downstream_records']['activation_results'] = args['bridge_records']['native_activation_results']
    args['bridge_records']['native_activation_results'] = []
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert not next(r for r in result['compilation_native_owner_observations'] if r.get('kind') == 'native_owner')['profile_metadata_binding_verified']
