# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_source_family_adoption.py
#   src/blueprint_pipeline/task_evaluation_scene_source_family_inventory.py
"""ADP-009D/day-28: original prefix dependencies never transfer ownership."""
from __future__ import annotations

import copy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_source_family_website import api, pair, ref, seal, change, refuses
from tests.test_scene_source_family_sam import fixture as sam_fixture, PHASES


def fixture(*, through='calibrated_views'):
    args = sam_fixture()
    rows = args['source_records']
    original_job = json.loads(rows['sam_jobs'][0][1])
    plan = rows['sam_plans'][0]
    inputs = copy.deepcopy(original_job['inputs'])
    rows['sam_jobs'], rows['sam_results'], rows['sam_execution_receipts'] = [], [], []
    phase_records, artifacts = [], {}
    for phase in PHASES[:PHASES.index(through) + 1]:
        key = {'parent_request_digest': original_job['parent_request_digest'], 'plan_digest': ref(plan)['sha256'],
               'phase': phase, 'inputs_digest': canonical_digest({name: {k: r[k] for k in ('sha256', 'size_bytes')} for name, r in inputs.items()})}
        child = 'sam31-' + canonical_digest(key)[7:]
        job = seal({'schema_version': original_job['schema_version'], 'child_id': child,
            'parent_preparation_id': original_job['parent_preparation_id'], **key,
            'expected_source_commit': 'a'*40, 'plan_ref': ref(plan), 'inputs': copy.deepcopy(inputs)}, 'job_digest')
        names = {'source_selections': ['selection_inputs'], 'standard_splat_conversion': ['standard_splat', 'standard_splat_conversion_receipt'],
                 'calibrated_views': ['calibrated_view_request'], 'sam31_review': ['track_selection_review'],
                 'calibrated_masks': ['calibrated_mask_set'], 'segment_cutout': ['segment_cutout_set']}.get(phase, [phase + '_artifact'])
        generated = {}
        for name in names:
            item = (args['roots']['sam_execution_root'] + '/' + key['parent_request_digest'][7:] + '/' + child + '/' + name + '.bin', name.encode())
            rows['opaque_evidence'].append(item)
            generated[name] = ref(item)
        outcome = {'status': 'completed', 'artifacts': generated}
        result = seal({'schema_version': 'task_evaluation_sam31_preparation_execution_result.v1', 'child_id': child,
            'job_digest': job['job_digest'], 'parent_request_digest': key['parent_request_digest'], 'plan_digest': key['plan_digest'],
            'phase': phase, 'source_commit': 'a'*40, 'status': 'completed', 'artifacts': generated, 'executor_result': outcome}, 'result_digest')
        receipt = seal({'schema_version': 'task_evaluation_sam31_phase_execution_receipt.v1', 'job_digest': job['job_digest'],
            'source_commit': 'a'*40, 'phase': phase, 'outcome': outcome}, 'receipt_digest')
        job_pair = pair(args['roots']['sam_queue_root'] + '/completed/' + child + '.json', job)
        result_pair = pair(args['roots']['sam_queue_root'] + '/results/' + child + '.json', result)
        receipt_pair = pair(args['roots']['sam_execution_root'] + '/' + key['parent_request_digest'][7:] + '/' + child + '/phase_execution_receipt.v1.json', receipt)
        rows['sam_jobs'].append(job_pair)
        rows['sam_results'].append(result_pair)
        rows['sam_execution_receipts'].append(receipt_pair)
        phase_records.append({'phase': phase, 'job': ref(job_pair), 'result': ref(result_pair), 'execution_receipt': ref(receipt_pair)})
        inputs.update(generated)
        artifacts.update(generated)
        if phase == 'standard_splat_conversion':
            inputs['standard_splat_conversion'] = artifacts['standard_splat_conversion'] = generated['standard_splat_conversion_receipt']
    successor_standard = ('/retained/host-inputs/successor.glb', b'successor')
    rows['opaque_evidence'].append(successor_standard)
    conversion = pair('/retained/host-inputs/conversion.json', seal({'schema_version': 'standard_splat_conversion_receipt.v1',
        'output': {'relative_path': 'successor.glb', 'sha256': ref(successor_standard)['sha256'], 'size_bytes': len(successor_standard[1])}}, 'receipt_digest'))
    rows['sam_host_evidence'].append(conversion)
    current_task = json.loads(rows['sam_host_tasks'][0][1])
    current_task.update(expected_production_commit='b'*40, source_input_references={'standard_splat_conversion_receipt': ref(conversion)})
    current_task_pair = pair('/retained/host-inputs/current_task.json', current_task)
    rows['sam_host_tasks'].append(current_task_pair)
    current_host = dict(inputs)
    current_host = {name: r for name, r in current_host.items() if name in {'task_request', 'installation_receipt', 'publisher_intake', 'source_preparation_receipt', 'interiorgs_terms'}}
    current_host['task_request'] = ref(current_task_pair)
    adoption = {'schema_version': 'task_evaluation_sam31_completed_prefix_adoption.v1', 'status': 'verified_completed_prefix',
        'source_commit': 'b'*40, 'original_execution_commit': 'a'*40,
        'original_parent_request_digest': original_job['parent_request_digest'], 'original_parent_envelope': ref(rows['sam_parent_envelopes'][0]),
        'source_plan': ref(plan), 'source_profile': ref(rows['sam_profiles'][0]), 'through_phase': through,
        'phase_records': phase_records, 'current_host_inputs': current_host,
        'current_sam31_provider_profile': ref(rows['opaque_evidence'][0]), 'provider_zero_at_adoption': ref(rows['opaque_evidence'][0]),
        'current_release_root': '/retained/release', 'historical_receipts_modified': False,
        'paid_execution_performed': False, 'candidate_policy_queried': False,
        'administrative_rebindings': {name: {'original': artifacts[name], 'successor': ref(successor_standard) if name == 'standard_splat' else ref(conversion)}
            for name in ('standard_splat', 'standard_splat_conversion_receipt', 'standard_splat_conversion')}}
    rows['sam_adoptions'] = [pair('/retained/metadata/adoption.json', seal(adoption, 'adoption_digest'))]
    return args


def change_phase_input_path(args, position):
    rows = args['source_records']
    job_path, raw = rows['sam_jobs'][position]
    job = json.loads(raw)
    job['inputs']['interiorgs_terms']['path'] = '/retained/metadata/copied-terms.txt'
    rows['sam_jobs'][position] = pair(job_path, seal(job, 'job_digest'))
    job = json.loads(rows['sam_jobs'][position][1])
    result_path, raw = rows['sam_results'][position]
    rows['sam_results'][position] = pair(result_path, seal(dict(json.loads(raw), job_digest=job['job_digest']), 'result_digest'))
    receipt_path, raw = rows['sam_execution_receipts'][position]
    rows['sam_execution_receipts'][position] = pair(receipt_path, seal(dict(json.loads(raw), job_digest=job['job_digest']), 'receipt_digest'))
    for index, (adoption_path, raw) in enumerate(rows['sam_adoptions']):
        adoption = json.loads(raw)
        for phase in adoption['phase_records']:
            if phase['job']['path'] == job_path:
                phase.update(job=ref(rows['sam_jobs'][position]), result=ref(rows['sam_results'][position]),
                             execution_receipt=ref(rows['sam_execution_receipts'][position]))
        rows['sam_adoptions'][index] = pair(adoption_path, seal(adoption, 'adoption_digest'))


@pytest.mark.parametrize('missing', ['parent', 'first_receipt'])
def test_available_original_phase_inputs_refuse_despite_unrelated_missing_prefix_proof(missing):
    args = fixture()
    change_phase_input_path(args, 0 if missing == 'parent' else 1)
    if missing == 'parent':
        args['source_records']['sam_parent_envelopes'] = []
    else:
        args['source_records']['sam_execution_receipts'].pop(0)
    refuses(args)


@pytest.mark.parametrize('missing', ['plan', 'profile', 'first_result'])
def test_unreconstructible_original_input_map_stays_unresolved_without_empty_map_inference(missing):
    args = fixture()
    if missing == 'plan':
        change_phase_input_path(args, 1)  # This host key is genuinely unavailable.
    rows = args['source_records']
    if missing == 'plan':
        rows['sam_plans'] = []
    elif missing == 'profile':
        rows['sam_profiles'] = []
    else:
        rows['sam_results'].pop(0)
    result = api().join_retained_scene_source_family_inventory(**args)
    assert not result['adoption_observations'][0]['prefix_binding_verified']
    assert not next(r for r in result['original_phase_observations'] if r['phase'] == 'standard_splat_conversion')['phase_binding_verified']


def test_original_prefix_is_reconstructed_without_owner_transfer_or_scientific_proof():
    result = api().join_retained_scene_source_family_inventory(**fixture())
    adoption = result['adoption_observations'][0]
    assert adoption['phase_count'] == 3 and adoption['prefix_binding_verified']
    assert len(result['original_phase_observations']) == 3
    assert adoption['selection_origin']['source_commit'] == 'a'*40
    assert adoption['tracking_origin']['source_commit'] == 'a'*40
    assert result['original_owner_transfer_authorized'] is False
    assert result['scientific_validity_checked'] is False


@pytest.mark.parametrize('edit', [
    lambda a: a.update(original_execution_commit='c'*40),
    lambda a: a.update(original_parent_request_digest='sha256:' + 'f'*64),
    lambda a: a['phase_records'].reverse(),
    lambda a: a['phase_records'].append(dict(a['phase_records'][0])),
    lambda a: a.update(historical_receipts_modified=0),
    lambda a: a['administrative_rebindings']['standard_splat_conversion'].update(successor=a['administrative_rebindings']['standard_splat']['successor']),
])
def test_known_adoption_contradictions_refuse_even_without_current_owner(edit):
    args = fixture()
    change(args, 'sam_adoptions', edit, 'adoption_digest')
    refuses(args)


def test_missing_original_result_stays_unresolved_preserving_all_phase_references():
    args = fixture()
    removed = args['source_records']['sam_results'].pop(1)
    result = api().join_retained_scene_source_family_inventory(**args)
    assert result['adoption_observations'][0]['prefix_binding_verified'] is False
    assert any(r['path'] == removed[0] and r['status'] == 'kept_unresolved' for r in result['raw_reference_obligations'])


@pytest.mark.parametrize('field,wrong', [('schema_sha256', 'sha256:' + 'f'*64),
    ('policy_source_revision', 'a'*40), ('policy_source_path', 'wrong.py')])
def test_known_optional_frozen_contract_identity_refuses_own_contradictions(field, wrong):
    args = fixture()
    identity = {'schema_version': 'task_evaluation_retained_preparation_contract_identity.v1',
        'request_schema_version': 'task_evaluation_launch_preparation_request.v1', 'request_source_commit': 'a'*40,
        'contract_id': 'scene_preparation_27000_repair.v1',
        'schema_sha256': 'sha256:8d1f6826901b7e4fdbc486ec83ee4de7fa1dffcda4b3060c82b464577866da8c',
        'policy_source_revision': 'ac689e03ab6a7c6fb598f855d4f5bc37b4f87d43',
        'policy_source_path': 'src/blueprint_pipeline/task_evaluation_scene_configuration_runtime_budget.py', field: wrong}
    change(args, 'sam_adoptions', {'historical_parent_contract': seal(identity, 'contract_digest')}, 'adoption_digest')
    args['source_records']['sam_parent_envelopes'] = []
    refuses(args)


def test_future_adoption_status_is_retained_unresolved_without_interpreting_prefix():
    args = fixture()
    change(args, 'sam_adoptions', {'status': 'future_status', 'phase_records': None}, 'adoption_digest')
    result = api().join_retained_scene_source_family_inventory(**args)
    assert not result['adoption_observations']
    assert any(r['path'] == args['source_records']['sam_adoptions'][0][0] for r in result['raw_versions'])


@pytest.mark.parametrize('change_kind', ['missing_result_wrong_job', 'wrong_parent_path', 'wrong_child_path'])
def test_selected_original_receipt_binds_every_available_edge(change_kind):
    args = fixture()
    rows = args['source_records']
    path, raw = rows['sam_execution_receipts'][0]
    receipt = json.loads(raw)
    if change_kind == 'missing_result_wrong_job':
        rows['sam_results'].pop(0)
        receipt['job_digest'] = 'sha256:' + 'f'*64
    else:
        parts = path.split('/')
        parts[-3 if change_kind == 'wrong_parent_path' else -2] = 'f'*64 if change_kind == 'wrong_parent_path' else 'sam31-' + 'f'*64
        path = '/'.join(parts)
    updated = pair(path, seal(receipt, 'receipt_digest'))
    rows['sam_execution_receipts'][0] = updated
    change(args, 'sam_adoptions', lambda v: v['phase_records'][0].update(execution_receipt=ref(updated)), 'adoption_digest')
    refuses(args)


def test_retained_release_and_tracking_bare_selectors_are_explicit_unverified_obligations():
    args = fixture(through='sam31_tracking')
    dependency = ref(args['source_records']['opaque_evidence'][0])
    change(args, 'sam_adoptions', {'retained_release_pin': {'path': '/retained/original-release',
        'source_commit': 'a'*40, 'tree': 'c'*40}, 'tracking_identity': {'raw_runtime_result': dependency,
        'provider_instance_id': 12345, 'checkpoint_digest': 'sha256:'+'4'*64,
        'official_charge': {'provider_billing_source_receipt': dependency}}}, 'adoption_digest')
    result = api().join_retained_scene_source_family_inventory(**args)
    roles = {r['role']: r for r in result['structural_join_obligations']}
    assert roles['sam_current_release']['expected_path'] == '/retained/release'
    assert roles['sam_retained_release']['expected_path'] == '/retained/original-release'
    assert roles['sam_tracking_identity']['selector'] == {'provider_instance_id': 12345,
        'checkpoint_digest': 'sha256:'+'4'*64}
    assert roles['sam_tracking_identity']['expected_path'] is None
    assert result['current_billing_settled'] is False


def test_adoption_input_permutations_keep_original_successor_provenance():
    args = fixture()
    result = api().join_retained_scene_source_family_inventory(**args)
    for rows in args['source_records'].values():
        rows.reverse()
    assert result == api().join_retained_scene_source_family_inventory(**args)


@pytest.mark.parametrize('through', PHASES[2:])
def test_each_completed_prefix_length_has_source_backed_order_and_distinct_raw_versions(through):
    result = api().join_retained_scene_source_family_inventory(**fixture(through=through))
    assert result['adoption_observations'][0]['phase_count'] == PHASES.index(through) + 1
    assert result['adoption_observations'][0]['prefix_binding_verified']


def inherited_fixture(*, through='calibrated_views'):
    args = fixture(through=through)
    rows, roots = args['source_records'], args['roots']
    previous = rows['sam_adoptions'][0]
    prior = json.loads(previous[1])
    old_task = next(p for p in rows['sam_host_tasks'] if p[0] == prior['current_host_inputs']['task_request']['path'])
    task = json.loads(old_task[1])
    profile = pair('/retained/metadata/second_profile.json', seal({
        'schema_version': 'task_evaluation_sam31_preparation_profile.v1', 'source_commit': 'b'*40,
        'artifact_references': {}, 'completed_prefix_adoption': ref(previous)}, 'profile_digest'))
    plan = pair('/retained/metadata/second_plan.json', seal({
        'schema_version': 'task_evaluation_sam31_preparation_plan.v1', 'source_commit': 'b'*40,
        'scene_identity': task['scene_identity'], 'task_identity': task['task_identity'], 'publisher_scene_id': task['publisher_scene_id'],
        'phase_sequence': list(PHASES), 'host_inputs': prior['current_host_inputs'],
        'server_profile_sha256': ref(profile)['sha256']}, 'plan_digest'))
    rows['sam_profiles'].append(profile)
    rows['sam_plans'].append(plan)
    request = {'schema_version': 'task_evaluation_launch_preparation_request.v1', 'run_mode': 'scene_configuration',
        'preparation_id': 'second-parent', 'run_id': 'second-run', 'team_namespace': 'separate-original-team',
        'expected_production_commit': 'b'*40, 'scene': {'identity': task['scene_identity']},
        'task': {'identity': task['task_identity']}, 'runtime': {'mounts': [{'source': {
            'uri': 's3://test/second_plan.json', 'digest': ref(plan)['sha256'], 'size_bytes': ref(plan)['size_bytes']}}]}}
    digest = canonical_digest(request)
    parent = pair(roots['preparation_queue_root'] + '/completed/second-parent-' + digest[7:] + '.json', seal({
        'schema_version': 'task_evaluation_launch_preparation_envelope.v1', 'request': request, 'request_digest': digest}, 'envelope_digest'))
    rows['sam_parent_envelopes'].append(parent)
    inputs = copy.deepcopy(prior['current_host_inputs'])
    original_artifacts = {}
    for selected in prior['phase_records']:
        result = next(json.loads(raw) for path, raw in rows['sam_results'] if path == selected['result']['path'])
        original_artifacts.update(result['artifacts'])
    original_artifacts['standard_splat_conversion'] = original_artifacts['standard_splat_conversion_receipt']
    original_artifacts.update({name: row['successor'] for name, row in prior['administrative_rebindings'].items()})
    inputs.update(original_artifacts)
    selected_rows = []
    for phase in PHASES[PHASES.index(through) + 1:6]:
        key = {'parent_request_digest': digest, 'plan_digest': ref(plan)['sha256'], 'phase': phase,
            'inputs_digest': canonical_digest({name: {k: r[k] for k in ('sha256', 'size_bytes')} for name, r in inputs.items()})}
        child = 'sam31-' + canonical_digest(key)[7:]
        job = pair(roots['sam_queue_root'] + '/completed/' + child + '.json', seal({
            'schema_version': 'task_evaluation_sam31_preparation_execution_job.v1', 'child_id': child,
            'parent_preparation_id': 'second-parent', **key, 'expected_source_commit': 'b'*40,
            'plan_ref': ref(plan), 'inputs': copy.deepcopy(inputs)}, 'job_digest'))
        payload = (roots['sam_execution_root'] + '/' + digest[7:] + '/' + child + '/' + phase + '.bin', phase.encode())
        rows['opaque_evidence'].append(payload)
        generated = {phase + '_artifact': ref(payload)}
        outcome = {'status': 'completed', 'artifacts': generated}
        result = pair(roots['sam_queue_root'] + '/results/' + child + '.json', seal({
            'schema_version': 'task_evaluation_sam31_preparation_execution_result.v1', 'child_id': child,
            'job_digest': json.loads(job[1])['job_digest'], 'parent_request_digest': digest,
            'plan_digest': ref(plan)['sha256'], 'phase': phase, 'source_commit': 'b'*40,
            'status': 'completed', 'artifacts': generated, 'executor_result': outcome}, 'result_digest'))
        receipt = pair(roots['sam_execution_root'] + '/' + digest[7:] + '/' + child + '/phase_execution_receipt.v1.json', seal({
            'schema_version': 'task_evaluation_sam31_phase_execution_receipt.v1', 'phase': phase, 'source_commit': 'b'*40,
            'job_digest': json.loads(job[1])['job_digest'], 'outcome': outcome}, 'receipt_digest'))
        rows['sam_jobs'].append(job)
        rows['sam_results'].append(result)
        rows['sam_execution_receipts'].append(receipt)
        inputs.update(generated)
        original_artifacts.update(generated)
        selected_rows.append({'phase': phase, 'job': ref(job), 'result': ref(result), 'execution_receipt': ref(receipt)})
    current_task = pair('/retained/host-inputs/third_task.json', dict(task, expected_production_commit='c'*40))
    rows['sam_host_tasks'].append(current_task)
    current_host = dict(prior['current_host_inputs'], task_request=ref(current_task))
    current = dict(prior, source_commit='c'*40, original_execution_commit='b'*40,
        original_parent_request_digest=digest, original_parent_envelope=ref(parent), source_plan=ref(plan), source_profile=ref(profile),
        through_phase='sam31_review', phase_records=selected_rows, current_host_inputs=current_host,
        administrative_rebindings={name: {'original': original_artifacts[name], 'successor': original_artifacts[name]}
                                   for name in ('standard_splat', 'standard_splat_conversion_receipt', 'standard_splat_conversion')})
    rows['sam_adoptions'].append(pair('/retained/metadata/second_adoption.json', seal(current, 'adoption_digest')))
    return args


@pytest.mark.parametrize('missing', ['original_parent', 'original_result'])
def test_inherited_input_availability_is_independent_of_broader_prefix_proof(missing):
    args = inherited_fixture()
    rows = args['source_records']
    if missing == 'original_parent':
        change_phase_input_path(args, 3)  # First phase extending the inherited prefix.
        rows['sam_parent_envelopes'].pop(0)
        refuses(args)
    else:
        rows['sam_results'].pop(0)
        result = api().join_retained_scene_source_family_inventory(**args)
        phase = next(r for r in result['original_phase_observations'] if r['phase'] == 'sam31_inputs')
        assert not phase['phase_binding_verified']


@pytest.mark.parametrize('missing', ['profile', 'first_result'])
def test_known_host_input_subset_cannot_be_hidden_by_other_unavailable_inputs(missing):
    args = fixture()
    change_phase_input_path(args, 0 if missing == 'profile' else 1)
    if missing == 'profile':
        args['source_records']['sam_profiles'] = []
    else:
        args['source_records']['sam_results'].pop(0)
    refuses(args)


def test_known_profile_input_subset_cannot_be_hidden_by_unavailable_plan():
    args = fixture()
    rows = args['source_records']
    plan = json.loads(rows['sam_plans'][0][1])
    profile_path, raw = rows['sam_profiles'][0]
    profile = json.loads(raw)
    profile['artifact_references'] = {'interiorgs_terms': dict(plan['host_inputs']['interiorgs_terms'],
        path='/retained/metadata/copied-terms.txt')}
    rows['sam_profiles'] = [pair(profile_path, seal(profile, 'profile_digest'))]
    rows['sam_plans'] = []
    change(args, 'sam_adoptions', {'source_profile': ref(rows['sam_profiles'][0])}, 'adoption_digest')
    refuses(args)


def test_selected_original_future_result_stays_raw_without_phase_promotion():
    args = fixture()
    rows = args['source_records']
    change(args, 'sam_results', {'status': 'future_status', 'executor_result': {'status': 'future_status'}}, 'result_digest')
    change(args, 'sam_adoptions', lambda a: a['phase_records'][0].update(result=ref(rows['sam_results'][0])), 'adoption_digest')
    result = api().join_retained_scene_source_family_inventory(**args)
    assert any(r['sha256'] == ref(rows['sam_results'][0])['sha256'] for r in result['raw_versions'])
    phase = next(r for r in result['original_phase_observations'] if r['phase'] == 'source_selections')
    assert not phase['phase_binding_verified']


def test_selected_original_known_failed_result_still_refuses_without_execution_receipt():
    args = fixture()
    rows = args['source_records']
    rows['sam_execution_receipts'] = []
    change(args, 'sam_results', {'status': 'failed', 'artifacts': {}, 'executor_result': {'status': 'failed', 'artifacts': {}}}, 'result_digest')
    change(args, 'sam_adoptions', lambda a: a['phase_records'][0].update(result=ref(rows['sam_results'][0])), 'adoption_digest')
    refuses(args)


@pytest.mark.parametrize('through,tracking_commit', [('calibrated_views', 'b'*40), ('sam31_tracking', 'a'*40)])
def test_recursive_prefix_keeps_earliest_selection_and_actual_tracking_producer(through, tracking_commit):
    result = api().join_retained_scene_source_family_inventory(**inherited_fixture(through=through))
    latest = next(r for r in result['adoption_observations'] if r['source_provenance'][0]['path'].endswith('/second_adoption.json'))
    assert latest['prefix_binding_verified'] and latest['phase_count'] == 6
    assert latest['selection_origin']['source_commit'] == 'a'*40
    assert latest['tracking_origin']['source_commit'] == tracking_commit
    assert result['original_owner_transfer_authorized'] is False


def selection_fixture():
    args = fixture()
    adoption = json.loads(args['source_records']['sam_adoptions'][0][1])
    selection = seal({'schema_version': 'task_evaluation_sam31_prefix_selection.v1', 'status': 'reusable_prefix_selected',
        'through_phase': 'calibrated_views', 'adoption': adoption,
        'rejected_candidates': [{'blocker': 'PRIVATE-REJECTED-TEXT', 'through_phase': 'segment_cutout', 'error_type': 'ValueError'}],
        'paid_execution_performed': False}, 'selection_digest')
    args['source_records']['sam_prefix_selections'] = [pair('/retained/metadata/selection.json', selection)]
    return args


def test_nested_prefix_selection_retains_enclosing_raw_provenance_not_invented_bytes():
    result = api().join_retained_scene_source_family_inventory(**selection_fixture())
    selection = next(r for r in result['sam_observations'] if r['role'] == 'sam_prefix_selection')
    assert selection['nested_adoption_provenance']['json_pointer'] == '/adoption'
    assert selection['factory_binding_verified'] is False
    assert 'PRIVATE-REJECTED' not in json.dumps(result)


@pytest.mark.parametrize('edit', [
    lambda s: s.update(paid_execution_performed=0),
    lambda s: s.update(through_phase='sam31_tracking'),
    lambda s: s['adoption'].update(original_execution_commit='c'*40),
])
def test_prefix_selection_own_known_contradictions_refuse_without_factory(edit):
    args = selection_fixture()
    change(args, 'sam_prefix_selections', edit, 'selection_digest')
    refuses(args)
