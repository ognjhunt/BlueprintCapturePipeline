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


def packet_fixture():
    import hashlib
    args = base_fixture()
    root = args['roots']['sam_execution_root'] + '/packet'
    rows = args['source_records']
    dependency = (root + '/dependency.json', b'tiny')
    rows['opaque_evidence'] = [dependency]
    request = {'schema_version': 'semantic_sam31_source_track_run_request.v1',
               'provider_profile': {'checkpoint': 'historical-only'}, 'prompts': ['tiny ☃']}
    request_pair = pair(root + '/semantic_sam31_source_track_run_request.v1.json', request)
    packet = seal({'schema_version': 'public_scene_sam31_task_input_packet.v1', 'status': 'prepared_no_upload_no_execution',
        'task_freeze': dict(ref(dependency), task_freeze_digest='sha256:'+'1'*64),
        'calibrated_view_receipt': dict(ref(dependency), receipt_digest='sha256:'+'2'*64),
        'provider_profile': dict(ref(dependency), profile_digest='sha256:'+'3'*64),
        'run_request': {'relative_path': request_pair[0].rsplit('/', 1)[1], 'sha256': ref(request_pair)['sha256'],
            'size_bytes': ref(request_pair)['size_bytes'], 'request_digest': 'sha256:' + hashlib.sha256(
                json.dumps(request, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()},
        'paid_execution_started': False, 'provider_mutations_performed': 0}, 'receipt_digest')
    rows['sam_artifact_metadata'] = [pair(root + '/public_scene_sam31_task_input_packet.v1.json', packet), request_pair]
    return args


@pytest.mark.parametrize('edit', [lambda v: v['run_request'].update(relative_path='../escape.json'),
    lambda v: v['run_request'].update(size_bytes=False), lambda v: v.update(provider_mutations_performed=False),
    lambda v: v['run_request'].update(request_digest='sha256:'+'f'*64)])
def test_known_packet_own_and_available_request_metadata_refuse_without_adoption(edit):
    args = packet_fixture()
    change(args, 'sam_artifact_metadata', edit, 'receipt_digest')
    refuses(args)


def test_packet_unicode_uses_actual_ascii_canonical_selector_and_keeps_model_unchecked():
    result = api().join_retained_scene_source_family_inventory(**packet_fixture())
    assert result['scientific_validity_checked'] is False
    assert any(r['role'] == 'sam_artifact_metadata' for r in result['raw_versions'])


def test_exact_selected_future_recipe_keeps_raw_bytes_without_known_stage_interpretation():
    args = fixture()
    rows = args['source_records']
    recipe_pair = pair('/retained/metadata/recipe.json', {'schema_version': 'future_recipe', 'stage_sequence': None})
    rows['sam_recipes'] = [recipe_pair]
    parent = json.loads(rows['sam_parent_envelopes'][0][1])
    parent['request']['construction'] = {'recipe': {'uri': 's3://test/recipe', 'digest': ref(recipe_pair)['sha256'],
                                                   'size_bytes': ref(recipe_pair)['size_bytes']}}
    parent['request_digest'] = canonical_digest(parent['request'])
    parent = seal(parent, 'envelope_digest')
    rows['sam_parent_envelopes'] = [pair(args['roots']['preparation_queue_root'] + '/completed/parent-1-' +
        parent['request_digest'][7:] + '.json', parent)]
    job = json.loads(rows['sam_jobs'][0][1])
    job['parent_request_digest'] = parent['request_digest']
    job['child_id'] = 'sam31-' + canonical_digest({k: job[k] for k in
        ('parent_request_digest', 'plan_digest', 'phase', 'inputs_digest')})[7:]
    rows['sam_jobs'] = [pair(args['roots']['sam_queue_root'] + '/completed/' + job['child_id'] + '.json', seal(job, 'job_digest'))]
    rows['sam_results'], rows['sam_execution_receipts'] = [], []
    result = api().join_retained_scene_source_family_inventory(**args)
    assert any(r['path'] == recipe_pair[0] for r in result['raw_versions'])
    assert any(r['reason'] == 'unsupported_retained_schema' for r in result['structural_join_obligations'])


def test_unknown_nested_final_status_and_unknown_result_status_remain_discoverable():
    args = fixture()
    change(args, 'sam_results', {'status': 'future_status', 'artifacts': None}, 'result_digest')
    args['source_records']['sam_execution_receipts'] = []
    parent = json.loads(args['source_records']['sam_parent_envelopes'][0][1])
    final = seal({'schema_version': 'task_evaluation_sam31_preparation_result.v1', 'status': 'future_status'}, 'result_digest')
    progress = seal({'schema_version': 'task_evaluation_sam31_preparation_progress.v1',
        'preparation_id': 'parent-1', 'request_digest': parent['request_digest'], 'run_id': 'run-1', 'source_commit': COMMIT,
        'status': 'future_status', 'advancement': {'status': 'future_status', 'sam31_preparation_result': final},
        'sequence': 1, 'previous_progress_digest': None, 'provider_mutation_performed': False, 'paid_execution_requested': False}, 'progress_digest')
    path = args['roots']['preparation_queue_root'] + '/source-progress/parent-1-' + parent['request_digest'][7:] + '/000001-' + progress['progress_digest'][7:] + '.json'
    args['source_records']['source_progress'] = [pair(path, progress)]
    result = api().join_retained_scene_source_family_inventory(**args)
    assert any(r['path'] == path for r in result['raw_versions'])
    assert not any(r['role'] == 'sam_final' for r in result['sam_observations'])


def test_known_replay_receipt_cannot_claim_production_execution():
    args = fixture()
    change(args, 'sam_execution_receipts', {'schema_version': 'task_evaluation_sam31_phase_replay_receipt.v1',
        'production_execution_authorized': True, 'diagnostic_replay_code_root': '/retained/metadata/replay'}, 'receipt_digest')
    refuses(args)


def test_available_receipt_result_contradiction_refuses_when_job_is_absent():
    args = fixture()
    args['source_records']['sam_jobs'] = []
    change(args, 'sam_execution_receipts', lambda v: v['outcome'].update(status='failed'), 'receipt_digest')
    refuses(args)


def test_conflicting_historical_result_variants_stay_raw_without_cross_joining_one_receipt():
    args = fixture()
    rows = args['source_records']
    path, raw = rows['sam_results'][0]
    older = json.loads(raw)
    older.update(status='failed', artifacts={}, executor_result={'status': 'failed', 'artifacts': {}})
    older = seal(older, 'result_digest')
    rows['sam_results'].append(pair(path[:-5] + '.conflict-' + older['result_digest'][7:] + '.json', older))
    result = api().join_retained_scene_source_family_inventory(**args)
    assert len([r for r in result['raw_versions'] if r['role'] == 'sam_results']) == 2
    assert not next(r for r in result['sam_observations'] if r['role'] == 'sam_job')['result_binding_verified']


def test_bare_source_conversion_and_renderer_selectors_remain_structural_without_raw_size():
    args = fixture()
    rows, root = args['source_records'], args['roots']['host_input_root']
    source = seal({'schema_version': 'public_scene_source_preparation.v1', 'status': 'blocked',
        'source_commit': COMMIT, 'scene_id': 'scene-1', 'source_installation_digest': 'sha256:'+'c'*64,
        'provider_mutation_performed': False, 'paid_resource_used': False, 'candidate_policy_queried': False}, 'receipt_digest')
    conversion = seal({'schema_version': 'standard_splat_conversion_receipt.v1',
        'output': {'relative_path': 'derived/model.ply', 'sha256': 'sha256:'+'d'*64, 'size_bytes': 1},
        'rights': {'terms_digest': 'sha256:'+'e'*64}}, 'receipt_digest')
    rows['sam_host_evidence'] = [pair(root+'/source.json', source), pair(root+'/conversion.json', conversion)]
    rows['sam_artifact_metadata'] = [pair('/retained/metadata/renderer.json',
        {'schema_version': 'public_scene_interiorgs_edit_input_request.v2', 'scene': {'standard_splat_path': root+'/original.ply'}})]
    result = api().join_retained_scene_source_family_inventory(**args)
    obligations = result['structural_join_obligations']
    assert {'sam_source_installation', 'sam_conversion_terms', 'sam_renderer_input'} <= {r['role'] for r in obligations}
    assert next(r for r in obligations if r['role'] == 'sam_renderer_input')['expected_path'] == root+'/original.ply'


def adopted_final_fixture():
    from tests.test_scene_source_family_adoption import fixture as adoption_fixture
    args = adoption_fixture(through='segment_cutout')
    rows = args['source_records']
    adoption = json.loads(rows['sam_adoptions'][0][1])
    artifacts = {}
    for _, raw in rows['sam_results']:
        artifacts.update(json.loads(raw)['artifacts'])
    artifacts.update({name: item['successor'] for name, item in adoption['administrative_rebindings'].items()})
    evidence = {name: artifacts[name] for name in ('calibrated_mask_set', 'segment_cutout_set',
        'track_selection_review', 'selection_inputs', 'standard_splat_conversion')}
    final = seal({'schema_version': 'task_evaluation_sam31_preparation_result.v1', 'status': 'exact_mask_inputs_ready',
        'source_commit': adoption['source_commit'], 'plan_digest': ref(rows['sam_plans'][0])['sha256'], 'evidence': evidence,
        'stage_result_receipts': [], 'completed_prefix_adoption': {'receipt': ref(rows['sam_adoptions'][0]),
            'original_execution_commit': adoption['original_execution_commit'], 'through_phase': adoption['through_phase'],
            'original_phase_result_receipts': [r['result'] for r in adoption['phase_records']]}}, 'result_digest')
    request_digest = 'sha256:'+'f'*64  # Current parent proof intentionally unavailable.
    progress = seal({'schema_version': 'task_evaluation_sam31_preparation_progress.v1',
        'preparation_id': 'current-parent', 'request_digest': request_digest, 'run_id': 'current-run',
        'source_commit': adoption['source_commit'], 'status': 'ready', 'sequence': 1, 'previous_progress_digest': None,
        'provider_mutation_performed': False, 'paid_execution_requested': False,
        'advancement': {'status': 'ready', 'sam31_preparation_result': final, 'sam31_exact_mask_inputs': evidence,
                        'evidence_refs': list(evidence.values())}}, 'progress_digest')
    path = args['roots']['preparation_queue_root']+'/source-progress/current-parent-'+request_digest[7:]+'/000001-'+progress['progress_digest'][7:]+'.json'
    rows['source_progress'] = [pair(path, progress)]
    return args


def edit_adopted_final(args, edit):
    path, raw = args['source_records']['source_progress'][0]
    progress = json.loads(raw)
    final = progress['advancement']['sam31_preparation_result']
    edit(final)
    progress['advancement']['sam31_preparation_result'] = seal(final, 'result_digest')
    progress['advancement']['sam31_exact_mask_inputs'] = final['evidence']
    progress['advancement']['evidence_refs'] = [final['evidence'][name] for name in
        ('calibrated_mask_set', 'segment_cutout_set', 'track_selection_review', 'selection_inputs', 'standard_splat_conversion')]
    progress = seal(progress, 'progress_digest')
    args['source_records']['source_progress'] = [pair(path.rsplit('/', 1)[0]+'/000001-'+progress['progress_digest'][7:]+'.json', progress)]


@pytest.mark.parametrize('edit', [
    lambda f: f['completed_prefix_adoption'].update(original_execution_commit='c'*40),
    lambda f: f['completed_prefix_adoption'].update(original_phase_result_receipts=[]),
    lambda f: f['completed_prefix_adoption']['original_phase_result_receipts'].reverse(),
    lambda f: f['evidence'].update(track_selection_review=f['evidence']['selection_inputs']),
])
def test_available_adoption_final_contradictions_refuse_without_current_parent(edit):
    args = adopted_final_fixture()
    edit_adopted_final(args, edit)
    refuses(args)


def test_matching_adoption_final_retains_original_and_successor_evidence_without_current_parent():
    result = api().join_retained_scene_source_family_inventory(**adopted_final_fixture())
    assert any(r['role'] == 'sam_final' and not r['parent_binding_verified'] for r in result['sam_observations'])


def test_final_selected_adoption_through_phase_disagreement_refuses_with_results_absent():
    args = adopted_final_fixture()
    references = [ref(r) for r in args['source_records']['sam_results']]
    args['source_records']['sam_results'] = []
    def edit(final):
        final['completed_prefix_adoption']['through_phase'] = 'calibrated_views'
        final['stage_result_receipts'] = references[3:]
    edit_adopted_final(args, edit)
    refuses(args)


def test_final_selected_adoption_source_commit_disagreement_refuses_without_current_parent():
    args = adopted_final_fixture()
    edit_adopted_final(args, lambda f: f.update(source_commit='c'*40))
    path, raw = args['source_records']['source_progress'][0]
    progress = seal(dict(json.loads(raw), source_commit='c'*40), 'progress_digest')
    args['source_records']['source_progress'] = [pair(path.rsplit('/', 1)[0]+'/000001-'+progress['progress_digest'][7:]+'.json', progress)]
    refuses(args)


@pytest.mark.parametrize('mode', ['missing_adoption', 'missing_results', 'future_schema', 'future_status', 'permutation'])
def test_adoption_final_missing_unknown_and_permuted_proofs_remain_protected(mode):
    args = adopted_final_fixture()
    rows = args['source_records']
    if mode == 'missing_adoption':
        rows['sam_adoptions'] = []
    elif mode == 'missing_results':
        rows['sam_results'] = []
    elif mode in {'future_schema', 'future_status'}:
        path, raw = rows['sam_adoptions'][0]
        adoption = json.loads(raw)
        adoption.update({'schema_version': 'task_evaluation_sam31_completed_prefix_adoption.v2'} if mode == 'future_schema'
                        else {'status': 'future_status'})
        rows['sam_adoptions'] = [pair(path, seal(adoption, 'adoption_digest'))]
        edit_adopted_final(args, lambda f: f['completed_prefix_adoption'].update(receipt=ref(rows['sam_adoptions'][0])))
    else:
        for values in rows.values():
            values.reverse()
    result = api().join_retained_scene_source_family_inventory(**args)
    assert any(r['role'] == 'sam_final' for r in result['sam_observations'])
    assert result['scientific_validity_checked'] is False and result['mutations'] == 0
