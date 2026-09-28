# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_downstream_terminal.py
#   src/blueprint_pipeline/task_evaluation_scene_downstream_inventory.py
"""Retained terminal copies and archives are not current proof or restore authority."""
from __future__ import annotations

import copy
import json

import pytest

from tests.test_scene_downstream_execution import D, api, edit, fixture as execution_fixture, put, refuses
from tests.test_scene_inventory_history import pair, ref, seal

NONEXECUTION = ('prepared_no_execution', 'blocked_before_paid_dispatch', 'blocked_awaiting_website_notification',
                'blocked_without_provider_allocation', 'blocked_without_provider_allocation_awaiting_notification')


def fixture(*, nonexecution=None, original='dispatch_receipt.json', pointer=False):
    args = execution_fixture()
    rows, roots = args['downstream_records'], args['roots']
    index = roots['terminal_result_root'] + '/' + args['intent_id']
    canary = roots['policy_canary_root'] + '/actual-canary-directory'
    for role, name in (('launch_requests', 'launch_request.json'), ('launch_profiles', 'launch_profile.json')):
        rows[role].append((index + '/' + name, rows[role][0][1]))
    if nonexecution:
        allocator = pair(canary + '/allocator_result.json', {'opaque_provider_record': True})
        rows['allocator_results'] = [allocator]
        field = 'blocked_result_digest' if original == 'preprovider_blocked.json' else 'receipt_digest'
        schema = 'task_evaluation_policy_canary_preprovider_blocked.v1' if field == 'blocked_result_digest' else 'task_evaluation_policy_canary_dispatch.v1'
        dispatch = {'schema_version': schema, 'run_id': 'run-2', 'run_kind': 'internal_policy_canary',
                    'claim_ceiling': 'diagnostic_policy_execution', 'status': nonexecution,
                    'provider_mutation_performed': False, 'provider_allocation_performed': False,
                    'provider_zero_required': False, 'provider_zero_not_applicable': True,
                    'paid_execution_requested': False, 'automatic_retry_authorized': False,
                    'automatic_retry_performed': False, 'retry_cap': 0, 'provider_call_reached': False,
                    'allocator_result': ref(allocator)}
        put(rows, 'canary_dispatches', canary + '/' + original, dispatch, field)
        rows['canary_dispatches'].append((index + '/policy_canary_nonexecution.json', rows['canary_dispatches'][0][1]))
        state = {'schema_version': 'task_evaluation_scene_nonexecution_terminal_state.v1', 'run_id': 'run-2',
                 'canary_run_root': canary, 'record_digest': json.loads(rows['canary_dispatches'][0][1])[field], 'status': nonexecution}
        put(rows, 'terminal_states', index + '/nonexecution_terminal_state.json', state, 'state_digest')
        return args
    projection = {'schema_version': 'task_evaluation_policy_canary_result_projection.v1', 'run_id': 'run-2',
                  'request_digest': D, 'configuration_digest': D, 'result_delivery_digest': D, 'result_status': 'completed_unqualified'}
    put(rows, 'canary_projections', canary + '/artifacts/result_delivery/policy_canary_result_projection.json', projection, 'projection_digest', cross=True)
    rows['canary_projections'].append((index + '/policy_canary_result_projection.json', rows['canary_projections'][0][1]))
    projection = json.loads(rows['canary_projections'][0][1])
    sync = {'schema_version': 'task_evaluation_policy_canary_webapp_sync_result.v1', 'status': 'succeeded', 'run_id': 'run-2',
            'request_digest': D, 'configuration_digest': D, 'result_status': 'completed_unqualified',
            'policy_canary_projection_digest': projection['projection_digest'], 'notification_delivery': {'status': 'delivered'}}
    rows['canary_syncs'] = [pair(canary + '/artifacts/result_delivery/policy_canary_webapp_sync.json', sync),
                            pair(index + '/policy_canary_webapp_sync.json', sync)]
    zero = {'schema_version': 'task_evaluation_policy_canary_vast_provider_zero.v1', 'status': 'provider_zero_confirmed',
            'api_confirmed': True, 'provider_zero_verified': True, 'live_instance_count': 0, 'blockers': []}
    put(rows, 'provider_zero_receipts', canary + '/post_teardown_global_provider_zero.json', zero, 'receipt_digest')
    rows['provider_zero_receipts'].append((index + '/provider_zero_closure.json', rows['provider_zero_receipts'][0][1]))
    dispatch = {'schema_version': 'task_evaluation_policy_canary_dispatch.v1', 'run_kind': 'internal_policy_canary',
                'run_id': 'run-2', 'status': 'completed_unqualified', 'policy_canary_projection_digest': projection['projection_digest'],
                'result_delivery_digest': D, 'policy_canary_result_projection': ref(rows['canary_projections'][0]),
                'policy_canary_webapp_sync': ref(rows['canary_syncs'][0]),
                'provider_zero': {**ref(rows['provider_zero_receipts'][0]), 'provider_zero_verified': True},
                'notification_delivery': sync['notification_delivery']}
    put(rows, 'canary_dispatches', canary + '/dispatch_receipt.json', dispatch, 'receipt_digest')
    rows['canary_dispatches'].append((index + '/dispatch_receipt.json', rows['canary_dispatches'][0][1]))
    dispatch = json.loads(rows['canary_dispatches'][0][1])
    state = {'schema_version': 'task_evaluation_scene_terminal_index_state.v1', 'run_id': 'run-2',
             'canary_run_root': canary, 'dispatch_receipt_digest': dispatch['receipt_digest'],
             'projection_digest': projection['projection_digest']}
    put(rows, 'terminal_states', index + '/terminal_index_state.json', state, 'state_digest')
    if pointer:
        members = [{ 'relative_path': 'artifacts/result_delivery/policy_canary_result_projection.json',
                     **{k: ref(rows['canary_projections'][0])[k] for k in ('sha256', 'size_bytes')}},
                   {'relative_path': 'dispatch_receipt.json', **{k: ref(rows['canary_dispatches'][0])[k] for k in ('sha256', 'size_bytes')}}]
        archive = {'schema_version': 'control_plane_evidence_offload_pointer.v1', 'status': 'offloaded',
                   'directory': 'actual-canary-directory', 'terminal_receipt': 'dispatch_receipt.json',
                   'uri': 's3://evidence/archive.tar.gz', 'digest': 'sha256:' + 'd' * 64,
                   'size_bytes': 100, 'member_count': len(members), 'members': members}
        put(rows, 'canary_offload_pointers', canary + '.offloaded.v1.json', archive, 'pointer_digest')
        publication = {'schema_version': 'task_evaluation_scene_terminal_result_publication.v1', 'run_id': 'run-2',
                       'uri': archive['uri'], 'digest': projection['projection_digest'], 'archive_digest': archive['digest'],
                       'size_bytes': 100, 'archive_member_count': 2,
                       'pointer_digest': json.loads(rows['canary_offload_pointers'][0][1])['pointer_digest'], 'provider_allocated': False}
        put(rows, 'terminal_publications', index + '/terminal_result_publication.json', publication, 'publication_digest')
    return args


def test_executed_original_and_indexed_copies_bind_actual_root_without_live_proof():
    args = fixture(pointer=True)
    result = api().join_retained_scene_downstream_inventory(**args)
    assert any(r['kind'] == 'canary_evidence_workspace' and r['path'].endswith('/actual-canary-directory') for r in result['lexical_members'])
    assert result['terminal_observations'][0]['status'] == 'matched_retained_bytes'
    assert result['terminal_observations'][0]['archive_binding_verified'] is True
    for flag in api().FALSE_FLAGS:
        assert result[flag] is False
    assert len([p for p in result['raw_versions'] if p['role'] == 'canary_projections']) == 2


@pytest.mark.parametrize('status', NONEXECUTION)
@pytest.mark.parametrize('original', ['dispatch_receipt.json', 'preprovider_blocked.json', 'no_provider_allocation_blocked.json'])
def test_nonexecution_producer_paths_and_seals_keep_original_evidence(status, original):
    args = fixture(nonexecution=status, original=original)
    result = api().join_retained_scene_downstream_inventory(**args)
    assert result['terminal_observations'][0]['status'] == 'matched_retained_bytes'
    assert any(r['kind'] == 'canary_evidence_workspace' for r in result['lexical_members'])
    assert not result['current_provider_zero_verified']


@pytest.mark.parametrize('mode', ['missing', 'wrongbytes', 'outside'])
def test_nonexecution_allocator_exact_raw_proof(mode):
    args = fixture(nonexecution='prepared_no_execution')
    if mode == 'missing':
        args['downstream_records']['allocator_results'] = []
        result = api().join_retained_scene_downstream_inventory(**args)
        assert result['terminal_observations'][0]['status'] == 'kept_unresolved'
        assert any(r['reason'] == 'reference_bytes_unavailable' for r in result['raw_reference_obligations'])
    elif mode == 'wrongbytes':
        path, raw = args['downstream_records']['allocator_results'][0]
        args['downstream_records']['allocator_results'][0] = (path, raw + b' ')
        result = api().join_retained_scene_downstream_inventory(**args)
        assert result['terminal_observations'][0]['status'] == 'kept_unresolved'
    else:
        for i in (0, 1):
            edit(args, 'canary_dispatches', lambda r: r['allocator_result'].update(path='/foreign/allocator_result.json'), 'receipt_digest', position=i)
        refuses(args)


def test_known_provider_null_variant_is_kept_unresolved_not_malformed_nonexecution():
    args = fixture(nonexecution='blocked_without_provider_allocation', original='no_provider_allocation_blocked.json')
    args['downstream_records']['terminal_states'] = []
    args['downstream_records']['canary_dispatches'].pop()
    edit(args, 'canary_dispatches', {'terminal_result_kind': 'definite_provider_create_refusal',
         'provider_call_reached': True, 'provider_zero_required': True, 'paid_execution_requested': True,
         'provider_zero_not_applicable': False, 'provider_null_evidence': {'provider_zero': ref(args['downstream_records']['allocator_results'][0])}}, 'receipt_digest')
    result = api().join_retained_scene_downstream_inventory(**args)
    assert any(r['reason'] == 'kept_unresolved_provider_refusal' for r in result['terminal_observations'])
    assert not any(r['kind'] == 'canary_evidence_workspace' for r in result['lexical_members'])


@pytest.mark.parametrize('role,field,changes,cross', [
    ('terminal_states', 'state_digest', {'canary_run_root': '/foreign/root'}, False),
    ('canary_projections', 'projection_digest', {'run_id': 'wrong'}, True),
    ('provider_zero_receipts', 'receipt_digest', {'live_instance_count': True}, False),
    ('canary_dispatches', 'receipt_digest', {'result_delivery_digest': 'sha256:' + 'f' * 64}, False),
])
def test_available_bad_terminal_edge_refuses_even_if_other_inputs_missing(role, field, changes, cross):
    args = fixture()
    edit(args, role, changes, field, cross=cross)
    args['downstream_records']['canary_syncs'] = []
    refuses(args)


@pytest.mark.parametrize('changes', [
    lambda p: p['members'][0].update(relative_path='../escape'),
    lambda p: p['members'].append(copy.deepcopy(p['members'][0])),
    lambda p: p['members'][0].update(size_bytes=True),
    lambda p: p['members'][0].update(size_bytes=p['members'][0]['size_bytes'] + 1),
    lambda p: p.update(member_count=True),
    lambda p: p.update(member_count=99),
])
def test_resealed_pointer_members_refuse_traversal_alias_or_size_mismatch(changes):
    args = fixture(pointer=True)
    edit(args, 'canary_offload_pointers', changes, 'pointer_digest')
    args['downstream_records']['terminal_publications'] = []
    refuses(args)


def test_publication_projection_digest_is_not_archive_digest():
    args = fixture(pointer=True)
    edit(args, 'terminal_publications', {'digest': 'sha256:' + 'd' * 64}, 'publication_digest')
    refuses(args)


def test_missing_pointer_or_original_record_bytes_never_claims_restore():
    args = fixture()
    result = api().join_retained_scene_downstream_inventory(**args)
    assert result['terminal_observations'][0]['archive_binding_verified'] is False
    assert any(r['reason'] == 'canary_offload_pointer_unavailable' for r in result['structural_join_obligations'])
    # Byte-identical indexed copies may verify retained bytes, never original presence.
    for role in ('canary_dispatches', 'canary_projections', 'canary_syncs', 'provider_zero_receipts'):
        args['downstream_records'][role].pop(0)
    result = api().join_retained_scene_downstream_inventory(**args)
    assert result['terminal_observations'][0]['status'] == 'matched_retained_bytes'
    assert any(r['reason'] == 'reference_bytes_unavailable' for r in result['raw_reference_obligations'])
    assert all(not m['presence_checked'] and not m['restore_verified'] for m in result['lexical_members'])


def test_compilation_non_scene_configuration_contract_keeps_owner_bridge_unproven():
    args = execution_fixture()
    rows, roots = args['downstream_records'], args['roots']
    envelope = {'schema_version': 'task_evaluation_episode_compilation_envelope.v1',
                'compilation_id': 'episode-prep', 'preparation_id': 'episode-prep', 'run_id': 'episode-run',
                'team_namespace': 'team-1', 'expected_production_commit': 'a' * 40,
                'configured_scene_revision_digest': D, 'configured_scene_bundle': {'uri': 's3://configured/scene', 'digest': D, 'size_bytes': 1},
                'preparation_result_digest': D,
                'request': {'run_mode': 'episode_evaluation', 'preparation_id': 'episode-prep',
                            'task': {'binding_mode': 'reuse_configured_template', 'configured_scene_revision_digest': D},
                            'construction': {'mode': 'reuse_configured_scene'}},
                'materialized_references': [], 'provider_mutation_performed': False, 'paid_execution_requested': False}
    value = seal(envelope, 'envelope_digest')
    filename = 'episode-prep-' + value['envelope_digest'][7:] + '.json'
    rows['compilation_envelopes'] = [pair(roots['compilation_queue_root'] + '/pending/' + filename, value)]
    output = {'schema_version': 'task_evaluation_episode_compilation_result.v1', 'status': 'compiled_for_production_launch',
              'compilation_id': 'episode-prep', 'run_id': 'episode-run', 'team_namespace': 'team-1', 'source_commit': 'a' * 40,
              'configured_scene_revision_digest': D, 'compiled_episode_packet_path': roots['compilation_output_root'] + '/episode-prep/packet.json',
              'compiled_episode_packet_digest': D, 'compiled_episode_packet_size_bytes': 1,
              'adapter_result_path': roots['compilation_output_root'] + '/episode-prep/adapter.json', 'adapter_result_digest': D,
              'compiler_output_digest': D, 'customer_supplied_prebuilt_episode_packet': False,
              'compiled_by_production': True, 'provider_mutation_performed': False, 'paid_execution_requested': False,
              'automatic_progression_required': True, 'blockers': []}
    put(rows, 'compilation_results', roots['compilation_queue_root'] + '/results/' + filename, output, 'result_digest')
    result = api().join_retained_scene_downstream_inventory(**args)
    assert result['compilation_observations'][0]['reason'] == 'compilation_owner_join_unproven'
    assert not any(r['kind'] == 'compilation_workspace' for r in result['lexical_members'])
    assert any(r['role'] == 'compilation_owner_bridge' for r in result['structural_join_obligations'])
