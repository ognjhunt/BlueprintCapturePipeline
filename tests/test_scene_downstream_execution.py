# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_downstream_inventory.py
#   src/blueprint_pipeline/task_evaluation_scene_downstream_contracts.py
#   src/blueprint_pipeline/task_evaluation_scene_downstream_execution.py
"""ADP-009D/day-28: retained execution identity never authorizes work/cleanup."""
from __future__ import annotations

import copy
import importlib
import json

import pytest

from tests.test_scene_inventory_history import pair, seal
from tests.test_scene_inventory_preparations import fixture as seed_fixture

D = 'sha256:' + 'b' * 64
ROLES = ('activation_results', 'launch_progressions', 'launch_profiles', 'launch_requests',
         'launch_receipts', 'terminal_states', 'canary_dispatches', 'canary_projections',
         'canary_syncs', 'provider_zero_receipts', 'allocator_results', 'canary_offload_pointers',
         'terminal_publications', 'compilation_envelopes', 'compilation_results')


def api():
    return importlib.import_module('blueprint_pipeline.task_evaluation_scene_downstream_inventory')


def fixture(*, activation='activation-1', launch='launch-1', run='run-2'):
    base = seed_fixture(activation=True, configuration=True, activation_edit=lambda r: r.update(
        activation_id=activation, release_window={'uri': 's3://window/file', 'digest': D, 'size_bytes': 1}))
    records, roots = base['records'], dict(base['roots'])
    roots.update(activation_output_root='/retained/activation-output', launch_execution_root='/retained/launches',
                 terminal_result_root='/retained/terminal', policy_canary_root='/retained/canaries',
                 compilation_queue_root='/retained/compilation-queue', compilation_output_root='/retained/compiled')
    rows = {role: [] for role in ROLES}
    envelope = json.loads(records['activation_envelopes'][0][1])
    request = envelope['request']
    filename = records['activation_envelopes'][0][0].rsplit('/', 1)[1]
    result = {'schema_version': 'task_evaluation_launch_activation_result.v1',
              'status': 'profile_authority_materialized_no_execution', 'activation_id': activation,
              'preparation_id': 'prep-1', 'team_namespace': 'team-1', 'lane': request['lane'],
              'source_commit': 'a' * 40, 'preparation_result_digest': request['preparation']['result_digest'],
              'release_window_digest': 'sha256:' + 'c' * 64,
              'profile_id': 'profile-1', 'profile_digest': D,
              'profile_publication_receipt_digest': D, 'standing_authorization_digest': D,
              'full_byte_activation_reference_readback_passed': True, 'profile_publication_performed': True,
              'catalog_mutation_performed': True, 'standing_authorization_published': True,
              'provider_mutation_performed': False, 'paid_execution_requested': False, 'blockers': []}
    attempt = json.loads(records['attempts'][0][1])
    binding = {key: attempt[key] for key in ('intent_id', 'intent_digest', 'attempt_id', 'source_commit', 'runtime_digest', 'input_digest')}
    binding['schema_version'] = 'task_evaluation_scene_attempt_binding.v1'
    profile = seal({'schema_version': 'task_evaluation_launch_profile.v1', 'profile_id': 'profile-1',
                    'source_commit': 'a' * 40, 'scene_intent_digest': attempt['intent_digest'],
                    'scene_attempt_id': attempt['attempt_id'], 'scene_attempt_binding': binding}, 'profile_digest')
    result['profile_digest'] = profile['profile_digest']
    put(rows, 'activation_results', roots['activation_queue_root'] + '/results/' + filename, result, 'result_digest')
    progression = {'schema_version': 'task_evaluation_scene_configuration_activation_progression.v1',
        'status': 'scene_configuration_launch_queued', 'preparation_id': 'prep-1', 'activation_id': activation,
        'expected_production_commit': 'a' * 40, 'launch_id': launch, 'run_id': run,
        'profile_id': 'profile-1', 'profile_digest': profile['profile_digest'],
        'activation_result_digest': json.loads(rows['activation_results'][0][1])['result_digest'],
        'standing_authorization_digest': D, 'submitted_through_webapp': True,
        'provider_mutation_performed_inside_progression': False, 'paid_execution_requested': True}
    put(rows, 'launch_progressions', roots['configuration_progression_root'] + '/scene-configuration-activations/prep-1/launch_progression.json', progression, 'progression_digest')
    rows['launch_profiles'] = [pair(roots['launch_execution_root'] + '/' + launch + '/launch_profile.json', profile)]
    launch_request = {'schema_version': 'task_evaluation_launch_request.v1', 'launch_id': launch, 'run_id': run,
                      'launch_profile_id': 'profile-1', 'launch_profile_digest': profile['profile_digest'], 'source_commit': 'a' * 40}
    put(rows, 'launch_requests', roots['launch_execution_root'] + '/' + launch + '/launch_request.json', launch_request, 'request_digest')
    receipt = {'schema_version': 'task_evaluation_launch_receipt.v1', 'status': 'completed', 'launch_id': launch,
               'run_id': run, 'launch_profile_digest': profile['profile_digest'],
               'request_digest': json.loads(rows['launch_requests'][0][1])['request_digest'], 'source_commit': 'a' * 40,
               'receipt_digest_canonicalization': 'rfc8785', 'provider_mutation_attempted': True}
    put(rows, 'launch_receipts', roots['launch_execution_root'] + '/' + launch + '/launch_receipt.json', receipt, 'receipt_digest', cross=True)
    return {'intent_id': base['intent_id'], 'seed_records': records, 'downstream_records': rows, 'roots': roots}


def put(rows, role, path, value, field, *, cross=False):
    rows[role].append(pair(path, seal(value, field, cross=cross)))


def edit(args, role, changes, field, *, cross=False, position=0):
    path, raw = args['downstream_records'][role][position]
    value = json.loads(raw)
    if callable(changes):
        changes(value)
    else:
        value.update(changes)
    args['downstream_records'][role][position] = pair(path, seal(value, field, cross=cross))


def refuses(args, code=None):
    with pytest.raises(api().SceneDownstreamInventoryError) as exc:
        api().join_retained_scene_downstream_inventory(**args)
    assert str(exc.value).startswith('scene_downstream_') and len(str(exc.value)) < 100
    if code:
        assert str(exc.value) == 'scene_downstream_' + code


def test_full_activation_launch_chain_keeps_seed_verbatim_and_historical_paid_scope():
    args = fixture()
    result = api().join_retained_scene_downstream_inventory(**args)
    from blueprint_pipeline.task_evaluation_scene_inventory_seed import join_retained_scene_inventory_seed
    assert result['seed'] == join_retained_scene_inventory_seed(intent_id=args['intent_id'], records=args['seed_records'],
                                                               roots={k: v for k, v in args['roots'].items() if k not in api().EXTRA_ROOTS})
    assert result['seed']['source_attempt_obligations'][0]['child_status'] == 'kept_out_of_scope'
    assert {r['kind'] for r in result['lexical_members']} == {'activation_workspace', 'launch_workspace'}
    assert next(r for r in result['lexical_members'] if r['kind'] == 'launch_workspace')['path'].endswith('/launch-1')
    assert result['activation_observations'][0]['window_binding_verified'] is False
    assert result['remote_reference_obligations'][0]['digest'] == D
    assert result['activation_observations'][0]['release_window_digest'] != D
    assert result['mutations'] == 0 and result['paid_scope_verified'] is False


@pytest.mark.parametrize('field,value', [('preparation_id', 'wrong'), ('team_namespace', 'wrong'), ('lane', 'wrong'),
                                        ('source_commit', 'f' * 40), ('preparation_result_digest', D),
                                        ('provider_mutation_performed', True), ('paid_execution_requested', True)])
def test_resealed_activation_identity_mismatch_refuses(field, value):
    args = fixture()
    edit(args, 'activation_results', {field: value}, 'result_digest')
    refuses(args)


def test_long_activation_id_binds_hashed_result_filename():
    args = fixture(activation='a' * 192)
    result = api().join_retained_scene_downstream_inventory(**args)
    assert result['activation_observations'][0]['status'] == 'matched_retained_bytes'
    assert args['downstream_records']['activation_results'][0][0].split('/')[-1].startswith('activation-')


@pytest.mark.parametrize('role,field', [('activation_results', 'result_digest'), ('launch_progressions', 'progression_digest'),
                                      ('launch_profiles', 'profile_digest'), ('launch_requests', 'request_digest'),
                                      ('launch_receipts', 'receipt_digest')])
def test_wrong_owned_path_refuses_even_when_related_evidence_absent(role, field):
    args = fixture()
    path, raw = args['downstream_records'][role][0]
    args['downstream_records'][role][0] = ('/foreign/' + path.rsplit('/', 1)[1], raw)
    args['downstream_records']['activation_results'] = [] if role != 'activation_results' else args['downstream_records'][role]
    refuses(args)


@pytest.mark.parametrize('role,field,changes', [
    ('launch_progressions', 'progression_digest', {'paid_execution_requested': False}),
    ('launch_requests', 'request_digest', {'launch_profile_digest': D}),
    ('launch_profiles', 'profile_digest', {'source_commit': 'f' * 40}),
    ('launch_receipts', 'receipt_digest', {'run_id': 'wrong'}),
])
def test_available_bad_launch_edge_cannot_hide_behind_missing_activation(role, field, changes):
    args = fixture()
    edit(args, role, changes, field, cross=role == 'launch_receipts')
    args['downstream_records']['activation_results'] = []
    refuses(args)


def test_missing_and_foreign_owner_keep_exact_raw_evidence_without_promoting_launch():
    for foreign in (False, True):
        args = fixture()
        def change(profile):
            if foreign:
                profile['scene_intent_digest'] = D
                profile['scene_attempt_binding']['intent_digest'] = D
            else:
                for key in ('scene_intent_digest', 'scene_attempt_id', 'scene_attempt_binding'):
                    profile.pop(key)
        edit(args, 'launch_profiles', change, 'profile_digest')
        profile = json.loads(args['downstream_records']['launch_profiles'][0][1])
        edit(args, 'launch_requests', {'launch_profile_digest': profile['profile_digest']}, 'request_digest')
        args['downstream_records']['launch_progressions'] = []
        args['downstream_records']['launch_receipts'] = []
        result = api().join_retained_scene_downstream_inventory(**args)
        assert not any(r['kind'] == 'launch_workspace' for r in result['lexical_members'])
        assert result['launch_observations'][0]['status'] == 'kept_unresolved'
        assert any(p['role'] == 'launch_profiles' for p in result['raw_versions'])


def test_missing_attempt_bytes_produces_structural_obligation_not_fake_raw_identity():
    args = fixture()
    args['seed_records']['attempts'] = []
    result = api().join_retained_scene_downstream_inventory(**args)
    row = next(r for r in result['structural_join_obligations'] if r['reason'] == 'owner_attempt_bytes_unavailable')
    assert row['expected_path'].endswith('.json') and 'sha256' not in row and 'size_bytes' not in row
    assert not any(r['kind'] == 'launch_workspace' for r in result['lexical_members'])


def test_blocked_activation_minimal_record_is_retained_without_workspace():
    args = fixture()
    path = args['downstream_records']['activation_results'][0][0]
    args['downstream_records']['activation_results'] = [pair(path, seal({
        'schema_version': 'task_evaluation_launch_activation_result.v1', 'activation_id': 'activation-1',
        'status': 'blocked', 'blockers': ['held'], 'catalog_mutation_state': 'unknown_if_preparation_started',
        'provider_mutation_performed': False, 'paid_execution_requested': False}, 'result_digest'))]
    args['downstream_records']['launch_progressions'] = []
    result = api().join_retained_scene_downstream_inventory(**args)
    assert not any(r['kind'] == 'activation_workspace' for r in result['lexical_members'])
    assert result['activation_observations'][0]['reason'] == 'activation_output_scope_unproven'


def test_input_order_and_nonmutation():
    args = fixture()
    before = copy.deepcopy(args)
    result = api().join_retained_scene_downstream_inventory(**args)
    assert args == before
    for rows in args['downstream_records'].values():
        rows.reverse()
    assert api().join_retained_scene_downstream_inventory(**args) == result


def policy_fixture():
    from tests.test_scene_inventory_history import fixture as empty_seed
    args = fixture()
    args['seed_records'] = empty_seed()['records']
    rows = args['downstream_records']
    for role in ('activation_results', 'launch_progressions', 'launch_receipts'):
        rows[role] = []
    intent_path, raw = args['seed_records']['intent']
    intent = json.loads(raw)
    candidates = [{'id': 'candidate-1', 'artifact_digest': D}, {'id': 'candidate-2', 'artifact_digest': 'sha256:' + 'e' * 64}]
    intent['request']['execution'] = {'policy_candidates': candidates}
    intent = seal(intent, 'intent_digest', cross=True)
    args['seed_records']['intent'] = pair(intent_path, intent)
    attempt = {'schema_version': 'task_evaluation_scene_attempt.v1', 'intent_id': args['intent_id'],
               'intent_digest': intent['intent_digest'], 'attempt_id': 'paid-1', 'source_commit': 'a' * 40,
               'input_digest': D, 'runtime_digest': D}
    args['seed_records']['attempts'] = [pair(args['roots']['intent_root'] + '/' + args['intent_id'] + '/attempts/paid-1.json', seal(attempt, 'attempt_digest', cross=True))]
    binding = seal({'schema_version': 'task_evaluation_scene_policy_binding.v1', 'scene_intent_digest': intent['intent_digest'],
                    'attempt_id': 'paid-1', 'policy_candidates': candidates, 'runtime_digest': D, 'input_digest': D}, 'binding_digest')
    path, raw = rows['launch_profiles'][0]
    profile = json.loads(raw)
    for key in ('scene_attempt_binding', 'scene_attempt_id'):
        profile.pop(key)
    profile.update(scene_intent_digest=intent['intent_digest'], internal_policy_canary_execution_plan={'scene_policy_binding': binding})
    profile = seal(profile, 'profile_digest')
    rows['launch_profiles'] = [pair(path, profile)]
    edit(args, 'launch_requests', {'launch_profile_digest': profile['profile_digest']}, 'request_digest')
    return args


def test_policy_owner_form_binds_exact_historical_attempt_and_frozen_pair():
    result = api().join_retained_scene_downstream_inventory(**policy_fixture())
    assert any(row['kind'] == 'launch_workspace' for row in result['lexical_members'])
    assert result['seed']['source_attempt_obligations'][0]['child_status'] == 'kept_out_of_scope'


@pytest.mark.parametrize('change', ['candidate', 'runtime', 'commit', 'attempt'])
def test_resealed_policy_attempt_contradictions_refuse(change):
    args = policy_fixture()
    path, raw = args['downstream_records']['launch_profiles'][0]
    profile = json.loads(raw)
    binding = profile['internal_policy_canary_execution_plan']['scene_policy_binding']
    if change == 'candidate':
        binding['policy_candidates'][0]['artifact_digest'] = 'sha256:' + 'f' * 64
    elif change == 'runtime':
        binding['runtime_digest'] = 'sha256:' + 'f' * 64
    elif change == 'commit':
        profile['source_commit'] = 'f' * 40
    else:
        binding['attempt_id'] = 'paid-2'
        # Missing proof remains unresolved; supplied contradiction uses the
        # selected attempt's canonical path with mismatched actual attempt ID.
        original_path, attempt_raw = args['seed_records']['attempts'][0]
        args['seed_records']['attempts'][0] = (original_path.replace('paid-1.json', 'paid-2.json'), attempt_raw)
    profile['internal_policy_canary_execution_plan']['scene_policy_binding'] = seal(binding, 'binding_digest')
    profile = seal(profile, 'profile_digest')
    args['downstream_records']['launch_profiles'][0] = pair(path, profile)
    edit(args, 'launch_requests', {'launch_profile_digest': profile['profile_digest'], 'source_commit': profile['source_commit']}, 'request_digest')
    refuses(args)


def test_launch_and_profile_contract_ids_support_192_character_producer_limit():
    args = fixture(launch='l' * 192, run='r' * 192)
    rows = args['downstream_records']
    edit(args, 'launch_profiles', {'profile_id': 'p' * 192}, 'profile_digest')
    digest = json.loads(rows['launch_profiles'][0][1])['profile_digest']
    edit(args, 'launch_requests', {'launch_profile_id': 'p' * 192, 'launch_profile_digest': digest}, 'request_digest')
    edit(args, 'activation_results', {'profile_id': 'p' * 192, 'profile_digest': digest}, 'result_digest')
    edit(args, 'launch_progressions', {'profile_id': 'p' * 192, 'profile_digest': digest,
         'activation_result_digest': json.loads(rows['activation_results'][0][1])['result_digest']}, 'progression_digest')
    edit(args, 'launch_receipts', {'launch_profile_digest': digest,
         'request_digest': json.loads(rows['launch_requests'][0][1])['request_digest']}, 'receipt_digest', cross=True)
    result = api().join_retained_scene_downstream_inventory(**args)
    assert any(row['path'].endswith('/' + 'l' * 192) for row in result['lexical_members'])
