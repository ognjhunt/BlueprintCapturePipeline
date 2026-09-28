# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_source_family_inventory.py
#   src/blueprint_pipeline/task_evaluation_scene_source_family_sam.py
#   src/blueprint_pipeline/task_evaluation_scene_source_family_adoption.py
"""ADP-009D/day-28: partial and original SAM evidence stays protected."""
from __future__ import annotations

import copy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_source_family_website import api, fixture as base_fixture, pair, ref, seal, change, refuses

PHASES = ('source_selections', 'standard_splat_conversion', 'calibrated_views', 'sam31_inputs', 'sam31_tracking',
          'sam31_review', 'calibrated_masks', 'removal_freezes', 'contribution_sweep', 'segment_cutout')
HOST_NAMES = ('task_request', 'installation_receipt', 'publisher_intake', 'source_preparation_receipt', 'interiorgs_terms')
COMMIT = 'a'*40


def fixture(*, phase='source_selections', result_status='completed', parent_id='parent-1'):
    args = base_fixture()
    rows, roots = args['source_records'], args['roots']
    task = {'schema_version': 'task_evaluation_minimal_task_request.v1', 'expected_production_commit': COMMIT,
            'scene_identity': {'id': 'scene-1'}, 'task_identity': {'id': 'task-1'}, 'publisher_scene_id': 'scene-1'}
    task_pair = pair(roots['host_input_root'] + '/task.json', task)
    rows['sam_host_tasks'] = [task_pair]
    host = {'task_request': ref(task_pair)}
    for name in HOST_NAMES[1:]:
        item = (roots['host_input_root'] + '/' + name + '.txt', name.encode())
        rows['opaque_evidence'].append(item)
        host[name] = ref(item)
    profile = seal({'schema_version': 'task_evaluation_sam31_preparation_profile.v1', 'source_commit': COMMIT,
                    'artifact_references': {}}, 'profile_digest')
    profile_pair = pair('/retained/metadata/profile.json', profile)
    rows['sam_profiles'] = [profile_pair]
    plan = seal({'schema_version': 'task_evaluation_sam31_preparation_plan.v1', 'source_commit': COMMIT,
        'scene_identity': task['scene_identity'], 'task_identity': task['task_identity'], 'publisher_scene_id': 'scene-1',
        'phase_sequence': list(PHASES), 'host_inputs': host, 'server_profile_sha256': ref(profile_pair)['sha256']}, 'plan_digest')
    plan_pair = pair('/retained/metadata/plan.json', plan)
    rows['sam_plans'] = [plan_pair]
    request = {'schema_version': 'task_evaluation_launch_preparation_request.v1', 'run_mode': 'scene_configuration',
        'preparation_id': parent_id, 'run_id': 'run-1', 'team_namespace': 'team-1', 'expected_production_commit': COMMIT,
        'scene': {'identity': task['scene_identity']}, 'task': {'identity': task['task_identity']},
        'runtime': {'mounts': [{'source': {'uri': 's3://test/plan.json', 'digest': ref(plan_pair)['sha256'],
                                         'size_bytes': ref(plan_pair)['size_bytes']}}]}}
    request_digest = canonical_digest(request)
    parent = seal({'schema_version': 'task_evaluation_launch_preparation_envelope.v1', 'request': request,
                   'request_digest': request_digest}, 'envelope_digest')
    rows['sam_parent_envelopes'] = [pair(roots['preparation_queue_root'] + '/completed/' + parent_id + '-' + request_digest[7:] + '.json', parent)]
    identities = {name: {k: row[k] for k in ('sha256', 'size_bytes')} for name, row in host.items()}
    key = {'parent_request_digest': request_digest, 'plan_digest': ref(plan_pair)['sha256'],
           'phase': phase, 'inputs_digest': canonical_digest(identities)}
    child_id = 'sam31-' + canonical_digest(key)[7:]
    job = seal({'schema_version': 'task_evaluation_sam31_preparation_execution_job.v1', 'child_id': child_id,
        'parent_preparation_id': parent_id, **key, 'expected_source_commit': COMMIT,
        'plan_ref': ref(plan_pair), 'inputs': host}, 'job_digest')
    rows['sam_jobs'] = [pair(roots['sam_queue_root'] + '/completed/' + child_id + '.json', job)]
    payload = (roots['sam_execution_root'] + '/' + request_digest[7:] + '/' + child_id + '/artifact.bin', b'tiny-evidence')
    rows['opaque_evidence'].append(payload)
    artifacts = {'phase_artifact': ref(payload)} if result_status == 'completed' else {}
    outcome = {'status': result_status, 'artifacts': artifacts}
    result = seal({'schema_version': 'task_evaluation_sam31_preparation_execution_result.v1', 'child_id': child_id,
        'job_digest': job['job_digest'], 'parent_request_digest': request_digest, 'plan_digest': key['plan_digest'],
        'phase': phase, 'source_commit': COMMIT, 'status': result_status, 'artifacts': artifacts,
        'executor_result': outcome}, 'result_digest')
    rows['sam_results'] = [pair(roots['sam_queue_root'] + '/results/' + child_id + '.json', result)]
    receipt = seal({'schema_version': 'task_evaluation_sam31_phase_execution_receipt.v1', 'job_digest': job['job_digest'],
        'phase': phase, 'source_commit': COMMIT, 'outcome': outcome}, 'receipt_digest')
    rows['sam_execution_receipts'] = [pair(roots['sam_execution_root'] + '/' + request_digest[7:] + '/' + child_id + '/phase_execution_receipt.v1.json', receipt)]
    return args


@pytest.mark.parametrize('phase', PHASES)
def test_every_supported_phase_retains_exact_job_result_receipt_without_live_claim(phase):
    result = api().join_retained_scene_source_family_inventory(**fixture(phase=phase))
    job = next(r for r in result['sam_observations'] if r['role'] == 'sam_job')
    assert job['phase'] == phase and job['result_binding_verified']
    assert result['current_queue_ownership_clear'] is False and result['scientific_validity_checked'] is False
    assert not any(m['kind'] == 'sam_execution_dependency' for m in result['lexical_members'])  # No current owner anchor.


@pytest.mark.parametrize('parent_id', ['parent-1', 'p'*192])
def test_parent_actual_192_bound_preserves_original_parent_identity(parent_id):
    result = api().join_retained_scene_source_family_inventory(**fixture(parent_id=parent_id))
    assert any(r.get('preparation_id') == parent_id for r in result['sam_observations'])


@pytest.mark.parametrize('role,seal_field,edits', [
    ('sam_jobs', 'job_digest', lambda v: v.update(inputs_digest='sha256:' + 'f'*64)),
    ('sam_jobs', 'job_digest', lambda v: v.update(plan_digest='sha256:' + 'f'*64)),
    ('sam_results', 'result_digest', lambda v: v.update(phase='sam31_tracking')),
    ('sam_execution_receipts', 'receipt_digest', lambda v: v['outcome'].update(status='failed')),
    ('sam_plans', 'plan_digest', lambda v: v.update(source_commit='b'*40)),
    ('sam_profiles', 'profile_digest', lambda v: v.update(source_commit='b'*40)),
])
def test_available_sam_contradictions_refuse_without_unrelated_parent(role, seal_field, edits):
    args = fixture()
    change(args, role, edits, seal_field)
    if role == 'sam_profiles':
        # Preserve the exact available selection; changing unselected bytes
        # alone truthfully makes the former raw profile unavailable.
        change(args, 'sam_plans', {'server_profile_sha256': ref(args['source_records']['sam_profiles'][0])['sha256']}, 'plan_digest')
    args['source_records']['sam_parent_envelopes'] = []
    refuses(args)


def test_exception_failure_null_optional_identities_protects_without_blocker_leak():
    args = fixture(result_status='failed')
    change(args, 'sam_results', {'job_digest': None, 'parent_request_digest': None, 'plan_digest': None,
        'phase': None, 'blocker': 'PRIVATE-TEXT-MUST-NOT-ESCAPE'}, 'result_digest')
    result = api().join_retained_scene_source_family_inventory(**args)
    assert 'PRIVATE-TEXT' not in json.dumps(result)
    assert any(r.get('result_status') == 'failed' for r in result['sam_observations'])


def test_copied_job_paths_keep_semantic_identity_and_all_raw_versions():
    args = fixture()
    path, raw = args['source_records']['sam_jobs'][0]
    copied = json.loads(raw)
    copied['inputs']['interiorgs_terms']['path'] = '/retained/metadata/copied-terms.txt'
    args['source_records']['sam_jobs'].append(pair(path.replace('/completed/', '/processing/'), seal(copied, 'job_digest')))
    result = api().join_retained_scene_source_family_inventory(**args)
    jobs = [r for r in result['sam_observations'] if r['role'] == 'sam_job']
    assert len(jobs) == 2 and len({r['child_id'] for r in jobs}) == 1
    assert result['current_queue_ownership_clear'] is False


def test_durable_final_is_nested_source_progress_and_preserves_json_pointer():
    from tests.test_scene_source_family_adoption import fixture as completed_fixture
    args = completed_fixture(through='segment_cutout')
    rows = args['source_records']
    parent = json.loads(rows['sam_parent_envelopes'][0][1])
    request = parent['request']
    artifacts = {}
    for _, raw in rows['sam_results']:
        artifacts.update(json.loads(raw)['artifacts'])
    artifacts['standard_splat_conversion'] = artifacts['standard_splat_conversion_receipt']
    evidence = {name: artifacts[name] for name in ('calibrated_mask_set', 'segment_cutout_set',
        'track_selection_review', 'selection_inputs', 'standard_splat_conversion')}
    final = seal({'schema_version': 'task_evaluation_sam31_preparation_result.v1', 'status': 'exact_mask_inputs_ready',
        'source_commit': COMMIT, 'plan_digest': ref(rows['sam_plans'][0])['sha256'], 'evidence': evidence,
        'stage_result_receipts': [ref(row) for row in rows['sam_results']]}, 'result_digest')
    progress = seal({'schema_version': 'task_evaluation_sam31_preparation_progress.v1',
        'preparation_id': request['preparation_id'], 'request_digest': parent['request_digest'], 'run_id': request['run_id'],
        'source_commit': COMMIT, 'status': 'ready', 'sequence': 1, 'previous_progress_digest': None,
        'provider_mutation_performed': False, 'paid_execution_requested': False,
        'advancement': {'status': 'ready', 'sam31_preparation_result': final, 'sam31_exact_mask_inputs': evidence,
                        'evidence_refs': list(evidence.values())}}, 'progress_digest')
    stem = request['preparation_id'] + '-' + parent['request_digest'][7:]
    rows['source_progress'] = [pair(args['roots']['preparation_queue_root'] + '/source-progress/' + stem +
        '/000001-' + progress['progress_digest'][7:] + '.json', progress)]
    before = copy.deepcopy(args)
    result = api().join_retained_scene_source_family_inventory(**args)
    final_row = next(r for r in result['sam_observations'] if r['role'] == 'sam_final')
    assert final_row['source_provenance'][0]['json_pointer'] == '/advancement/sam31_preparation_result'
    assert args == before
    assert 'sam31_preparation_result' not in json.loads(rows['sam_parent_envelopes'][0][1])


@pytest.mark.parametrize('bad_path', ['not-a-child', 'wrong-receipt.json'])
def test_production_receipt_own_layout_refuses_even_without_job(bad_path):
    args = fixture()
    args['source_records']['sam_jobs'] = []
    path, raw = args['source_records']['sam_execution_receipts'][0]
    parts = path.split('/')
    parts[-2 if bad_path == 'not-a-child' else -1] = bad_path
    args['source_records']['sam_execution_receipts'] = [('/'.join(parts), raw)]
    refuses(args)


@pytest.mark.parametrize('field,replacement', [('destination_root', '/retained/host/wrong'), ('paid_resource_used', True)])
def test_known_installation_own_metadata_refuses_without_plan(field, replacement):
    args = base_fixture()
    root = args['roots']['host_input_root']
    installed = seal({'schema_version': 'public_scene_host_input_installation_receipt.v1', 'status': 'installed',
        'scene_id': 'scene-1', 'packet_id': 'packet-1', 'source_commit_sha': COMMIT,
        'packet_digest': 'sha256:' + '1'*64, 'authoritative_request_digest': 'sha256:' + '1'*64,
        'destination_root': root, 'service_readable': True, 'provider_mutation_performed': False,
        'paid_resource_used': False, field: replacement}, 'receipt_digest')
    args['source_records']['sam_host_evidence'] = [pair(root + '/public_scene_host_input_installation_receipt.v1.json', installed)]
    refuses(args)


def test_selected_source_installation_canonical_identity_refuses_even_without_jobs():
    args = fixture()
    rows, root = args['source_records'], args['roots']['host_input_root']
    installation = seal({'schema_version': 'public_scene_host_input_installation_receipt.v1', 'status': 'installed',
        'scene_id': 'scene-1', 'packet_id': 'packet-1', 'source_commit_sha': 'b'*40,
        'packet_digest': 'sha256:' + '1'*64, 'authoritative_request_digest': 'sha256:' + '1'*64,
        'destination_root': root, 'service_readable': True, 'provider_mutation_performed': False,
        'paid_resource_used': False}, 'receipt_digest')
    source = seal({'schema_version': 'public_scene_source_preparation.v1',
        'status': 'source_context_prepared_pending_calibrated_views', 'source_commit': 'b'*40,
        'scene_id': 'scene-1', 'source_installation_digest': 'sha256:' + 'f'*64,
        'provider_mutation_performed': False, 'paid_resource_used': False, 'candidate_policy_queried': False}, 'receipt_digest')
    install_pair = pair(root + '/public_scene_host_input_installation_receipt.v1.json', installation)
    source_pair = pair(root + '/source/receipt.json', source)
    rows['sam_host_evidence'] = [install_pair, source_pair]
    change(args, 'sam_plans', lambda v: v['host_inputs'].update(installation_receipt=ref(install_pair),
        source_preparation_receipt=ref(source_pair)), 'plan_digest')
    for role in ('sam_jobs', 'sam_results', 'sam_execution_receipts', 'sam_parent_envelopes'):
        rows[role] = []
    refuses(args)
