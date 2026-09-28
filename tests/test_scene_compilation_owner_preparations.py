# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_compilation_native_owner_inventory.py
#   src/blueprint_pipeline/task_evaluation_scene_compilation_owner_contracts.py
#   src/blueprint_pipeline/task_evaluation_scene_compilation_owner_preparations.py
"""ADP-009D/day28: native handoff inverse is metadata, never historical bytes."""
import copy
import importlib
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_source_family_website import fixture as base_fixture, pair, ref, seal

ROLES = ('native_preparation_envelopes', 'native_preparation_results', 'native_activation_envelopes',
    'native_activation_results', 'configured_revisions', 'compilation_intake_receipts', 'compiler_outputs',
    'compilation_adapter_results', 'native_owner_records')


def api():
    return importlib.import_module('blueprint_pipeline.task_evaluation_scene_compilation_native_owner_inventory')


def fixture(*, retained_pre=False, policy=False, prep_id='native-prep'):
    args = base_fixture()
    args['bridge_records'] = {role: [] for role in ROLES}
    roots, rows = args['roots'], args['bridge_records']
    revision = seal({'schema_version': 'task_evaluation_configured_scene_revision.v1', 'status': 'configured',
        'source_commit': 'b'*40, 'scene_identity': {'id': 'scene-1'}, 'task_template': {'identity': {'id': 'task-1'}},
        'configured_scene_bundle': {'uri': 's3://test/bundle.zip', 'digest': 'sha256:'+'c'*64, 'size_bytes': 2}}, 'revision_digest')
    revision_pair = pair('/retained/metadata/revision.json', revision)
    rows['configured_revisions'] = [revision_pair]
    request = {'schema_version': 'task_evaluation_launch_preparation_request.v1', 'run_mode': 'episode_evaluation',
        'preparation_id': prep_id, 'run_id': 'native-run', 'team_namespace': 'team-1', 'expected_production_commit': 'a'*40,
        'scene': {'identity': revision['scene_identity'], 'configured_revision': ref(revision_pair)},
        'task': {'identity': revision['task_template']['identity'], 'binding_mode': 'reuse_configured_template',
            'configured_scene_revision_digest': revision['revision_digest']}, 'construction': {'mode': 'reuse_configured_scene'},
        'execution_adapter': {'runtime_source_bundle': {'uri': 's3://test/runtime.zip', 'digest': 'sha256:'+'d'*64, 'size_bytes': 3}}}
    digest = canonical_digest(request)
    envelope = seal({'schema_version': 'task_evaluation_launch_preparation_envelope.v1', 'request': request,
        'request_digest': digest}, 'envelope_digest')
    rows['native_preparation_envelopes'] = [pair(roots['preparation_queue_root']+'/materialized/'+prep_id+'-'+digest[7:]+'.json', envelope)]
    references = [{'contract_path': 'scene.configured_revision.configured_scene_bundle', **revision['configured_scene_bundle'],
        'materialized_path': roots['preparation_input_root']+'/'+prep_id+'/bundle.zip', 'content_addressed_reuse': False,
        'full_byte_service_account_readback_passed': True}]
    pre = {'schema_version': 'task_evaluation_launch_preparation_result.v1', 'status': 'inputs_materialized_awaiting_construction_adapter',
        'preparation_id': prep_id, 'run_id': request['run_id'], 'team_namespace': request['team_namespace'], 'source_commit': 'a'*40,
        'reference_count': 1, 'unique_object_count': 1, 'content_addressed_reuse_count': 0, 'references': references,
        'full_byte_service_account_readback_passed': True, 'service_account': 'blueprint', 'service_account_uid': 1000,
        'provider_mutation_performed': False, 'catalog_mutation_performed': False, 'paid_execution_requested': False,
        'observed_at_iso': '2026-09-28T07:00:00+00:00'}
    if policy:
        pre['policy_run_plan'] = {'schema_version': 'retained_policy.v1', 'policy': 'protected'}
    pre = seal(pre, 'result_digest')
    compilation = seal({'schema_version': 'task_evaluation_episode_compilation_envelope.v1', 'compilation_id': prep_id,
        'preparation_id': prep_id, 'run_id': request['run_id'], 'team_namespace': request['team_namespace'],
        'expected_production_commit': 'a'*40, 'configured_scene_revision_digest': revision['revision_digest'],
        'configured_scene_bundle': references[0], 'materialized_references': references, 'request': request,
        'preparation_result_digest': pre['result_digest'], 'automatic_progression_required': True,
        'robot_specific_episode_packet_compiled_in_production': True, 'customer_supplied_prebuilt_episode_packet': False,
        'production_compiler_owns_episode_packet': True, 'provider_mutation_performed': False, 'paid_execution_requested': False}, 'envelope_digest')
    filename = prep_id+'-'+compilation['envelope_digest'][7:]+'.json'
    compilation_pair = pair(roots['compilation_queue_root']+'/completed/'+filename, compilation)
    args['downstream_records']['compilation_envelopes'] = [compilation_pair]
    intake = seal({'schema_version': 'task_evaluation_episode_compilation_intake_receipt.v1',
        'status': 'queued_for_production_episode_compilation', 'compilation_id': prep_id, 'run_id': request['run_id'],
        'configured_scene_revision_digest': revision['revision_digest'], 'envelope_digest': compilation['envelope_digest'],
        'queue_path': compilation_pair[0], 'created': True, 'automatic_progression_required': True,
        'provider_mutation_performed': False, 'paid_execution_requested': False}, 'receipt_digest')
    rows['compilation_intake_receipts'] = [pair('/retained/metadata/intake.json', intake)]
    final = seal(dict(pre, status='queued_for_production_episode_compilation', run_mode=request['run_mode'],
        configured_scene_revision_digest=revision['revision_digest'], configured_scene_bundle_digest=references[0]['digest'],
        episode_compilation_id=prep_id, episode_compilation_queue_envelope_digest=compilation['envelope_digest'],
        episode_compilation_queue_receipt_digest=intake['receipt_digest'], customer_supplied_prebuilt_episode_packet=False,
        construction_packet_materialized=False, automatic_progression_required=True), 'result_digest')
    path = roots['preparation_queue_root']+'/results/'+prep_id+'-'+digest[7:]+'.json'
    rows['native_preparation_results'] = [pair(path, final)] + ([pair(path, pre)] if retained_pre else [])
    return args


def change(args, role, edit, field, *, family='bridge_records', position=0):
    path, raw = args[family][role][position]
    value = json.loads(raw)
    edit(value) if callable(edit) else value.update(edit)
    args[family][role][position] = pair(path, seal(value, field))


def refuses(args):
    with pytest.raises(api().SceneCompilationOwnerInventoryError) as exc:
        api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert str(exc.value).startswith('scene_compilation_owner_') and len(str(exc.value)) < 120


@pytest.mark.parametrize('retained_pre,policy', [(False, False), (True, False), (False, True)])
def test_exact_native_handoff_inverse_has_final_provenance_without_fake_raw_pre_bytes(retained_pre, policy):
    args = fixture(retained_pre=retained_pre, policy=policy)
    before = copy.deepcopy(args)
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    observation = result['preparation_handoff_observations'][0]
    assert observation['pre_handoff_binding_verified']
    assert observation['pre_handoff_proof_strength'] == ('retained_raw_and_derived_metadata' if retained_pre else 'derived_metadata_inverse')
    assert result['mutations'] == 0 and result['complete_scene_inventory'] is False
    assert args == before
    assert not any(p.get('role') == 'derived_pre_handoff_raw' for p in result['raw_versions'])


@pytest.mark.parametrize('edit', [
    {'reference_count': True}, {'unique_object_count': 2}, {'content_addressed_reuse_count': 1},
    {'source_commit': 'c'*40}, {'run_id': 'foreign-run'}, {'team_namespace': 'foreign'},
    {'configured_scene_revision_digest': 'sha256:'+'e'*64}, {'configured_scene_bundle_digest': 'sha256:'+'e'*64},
    {'episode_compilation_id': 'foreign'}, {'construction_packet_materialized': True},
])
def test_available_native_handoff_contradictions_refuse(edit):
    args = fixture()
    change(args, 'native_preparation_results', edit, 'result_digest')
    refuses(args)


def test_unknown_final_field_protects_without_guessing_inverse():
    args = fixture()
    change(args, 'native_preparation_results', {'future_field': 'preserved'}, 'result_digest')
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert not result['preparation_handoff_observations'][0]['pre_handoff_binding_verified']


@pytest.mark.parametrize('field', ['episode_compilation_queue_envelope_digest', 'episode_compilation_queue_receipt_digest'])
def test_absent_exact_canonical_selector_does_not_borrow_other_historical_bytes(field):
    args = fixture()
    change(args, 'native_preparation_results', {field: 'sha256:'+'e'*64}, 'result_digest')
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert any(r['selector'].get('canonical_digest') == 'sha256:'+'e'*64 for r in result['structural_join_obligations'])


def test_known_result_own_layout_refuses_when_compilation_and_parent_are_absent():
    args = fixture()
    args['downstream_records']['compilation_envelopes'] = []
    args['bridge_records']['native_preparation_envelopes'] = []
    path, raw = args['bridge_records']['native_preparation_results'][0]
    args['bridge_records']['native_preparation_results'] = [(path.replace('/results/', '/impossible/'), raw)]
    refuses(args)


def test_native_parent_existing_role_reuse_is_once_and_permutation_stable():
    args = fixture(retained_pre=True, prep_id='p'*192)
    args['source_records']['sam_parent_envelopes'] = args['bridge_records']['native_preparation_envelopes']
    args['bridge_records']['native_preparation_envelopes'] = []
    first = api().join_retained_scene_compilation_native_owner_inventory(**args)
    args['bridge_records']['native_preparation_results'].reverse()
    assert first == api().join_retained_scene_compilation_native_owner_inventory(**args)


def test_cross_role_identical_raw_copy_refuses_before_child():
    args = fixture()
    args['source_records']['sam_parent_envelopes'] = args['bridge_records']['native_preparation_envelopes'][:]
    refuses(args)


def test_intake_original_pending_path_survives_retained_completed_envelope_copy():
    args = fixture()
    change(args, 'compilation_intake_receipts', lambda value: value.update(queue_path=value['queue_path'].replace('/completed/', '/pending/')), 'receipt_digest')
    receipt = json.loads(args['bridge_records']['compilation_intake_receipts'][0][1])
    change(args, 'native_preparation_results', {'episode_compilation_queue_receipt_digest': receipt['receipt_digest']}, 'result_digest')
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert result['preparation_handoff_observations'][0]['pre_handoff_binding_verified']
