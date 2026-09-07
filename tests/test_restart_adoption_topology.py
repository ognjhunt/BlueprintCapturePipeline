"""Real adoption/queue topology with explicit synthetic scientific closure seams.

The graph, digests, historical contract, immutable publication, and scheduling
are real. Renderer/tracking closure and source-rights evidence are separate
contracts and are replaced here, as in test_sam31_prefix_adoption's driver test.
These fixtures establish no scientific, provider, billing, or host qualification.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_sam31_prefix_adoption as adoption
from blueprint_pipeline import task_evaluation_scene_configuration_sam31_preparation_driver as driver
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_sam31_phase_queue import enqueue_sam31_phase
from tests.test_sam31_prefix_adoption import prefix as prefix, write

A, B, C, D = (letter * 40 for letter in 'abcd')


@pytest.fixture
def topology(prefix, tmp_path, monkeypatch):
    value, original_plan, original_profile, _ = prefix
    calls = dict(render_validation=0, tracking_validation=0, provider=0, model=0, queue=[])
    conversions = {}
    standard = write(tmp_path / 'retained-source/standard.json', {'synthetic': 'source'})
    for commit in (A, B, C, D):
        root = tmp_path / ('release-' + commit[0])
        root.mkdir()
        for relative in (
            'scripts/adp_gaussian_excision_provider_runner.py',
            'src/blueprint_pipeline/public_scene_gaussian_excision_audit.py',
            'src/blueprint_pipeline/public_scene_calibrated_object_masks.py',
            'src/blueprint_pipeline/public_scene_segment_contribution_cutout.py',
            'src/blueprint_pipeline/task_evaluation_sam31_preparation_review_stages.py',
            'src/blueprint_pipeline/task_evaluation_sam31_preparation_profile.py',
        ):
            file = root / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('# synthetic producer presence; never executed\n')
        conversions[commit] = {'standard': standard,
            'conversion': write(root / 'conversion.json', {'synthetic_release': commit})}
    def source(host, commit):
        return json.loads(Path(host['task_request']['path']).read_text()), {}, {'synthetic_source': True}
    def render(*args):
        calls['render_validation'] += 1
        return {'path': str(tmp_path / 'release-a'), 'source_commit': A, 'tree': 'e' * 40}
    def tracking(outcome, artifacts, profile, current_profile, commit, billing):
        calls['tracking_validation'] += 1
        # This seam checks the provenance supplied by the real nesting logic.
        assert outcome['synthetic_tracking_producer'] == commit
        assert profile['source_commit'] == commit
        return {'synthetic_tracking_producer': commit}
    monkeypatch.setattr(adoption, 'source_science', source)
    monkeypatch.setattr(adoption, 'validate_current_rights', lambda task, context, host, commit, roots: conversions[commit])
    monkeypatch.setattr(adoption, 'validate_render', render)
    monkeypatch.setattr(adoption, 'validate_tracking', tracking)
    monkeypatch.setattr('blueprint_pipeline.public_scene_inpainting_inputs._git_identity',
        lambda root: {'commit': root.name.removeprefix('release-') * 40, 'tree': 'e' * 40})
    bindings = tmp_path / 'release-bindings'
    bindings.mkdir()
    billing = write(tmp_path / 'external-billing/receipt.json', {'synthetic_only': True})
    provider = original_profile['artifact_references']['sam31_provider_profile']

    def build(commit, through, inherited=None):
        host = deepcopy(original_plan['host_inputs'])
        host['task_request'] = write(tmp_path / commit[0] / 'task.json',
            {'expected_production_commit': commit, 'subject': {'source_instance_id': '115'}})
        profile = {**original_profile, 'source_commit': commit, 'repo_root': str(tmp_path / ('release-' + commit[0]))}
        if inherited is not None:
            profile['completed_prefix_adoption'] = inherited
        profile_ref = write(tmp_path / commit[0] / 'profile.json', profile, 'profile_digest')
        plan = {**original_plan, 'source_commit': commit, 'host_inputs': host,
            'server_profile_sha256': profile_ref['sha256']}
        plan_ref = write(tmp_path / commit[0] / 'plan.json', plan, 'plan_digest')
        request = json.loads(Path(value['original_parent_envelope']['path']).read_text())['request']
        request['expected_production_commit'] = commit
        request['runtime']['mounts'][-1]['source'].update(digest=plan_ref['sha256'], size_bytes=plan_ref['size_bytes'])
        request['spend']['external_service_caps']['openai']['stage_max_cost_usd']['artifixer_visual_review'] = .64
        digest = canonical_digest(request)
        write(tmp_path / 'parents/blocked' / (request['preparation_id'] + '-' + digest[7:] + '.json'),
            {'request': request, 'request_digest': digest}, 'envelope_digest')
        inputs, inherited_value = adoption._seed(plan, profile, (tmp_path,))
        start = inherited_value['phase_count'] if inherited_value else 0
        for phase in adoption.PHASES[start:adoption.PREFIX_LENGTHS[through]]:
            intake = enqueue_sam31_phase(queue_root=tmp_path / ('queue-' + commit[0]),
                parent_preparation_id=request['preparation_id'], parent_request_digest=digest,
                expected_source_commit=commit, plan_ref=plan_ref, phase=phase, inputs=inputs)
            job_path = Path(intake['job_path'])
            job = json.loads(job_path.read_text())
            job_path.rename(job_path.parent.parent / 'completed' / job_path.name)
            names = {'source_selections': 'task_selection', 'standard_splat_conversion': 'standard_splat_conversion_receipt',
                'calibrated_views': 'calibrated_view_receipt', 'sam31_inputs': 'sam31_run_request', 'sam31_tracking': 'sam31_source_tracks',
                'sam31_review': 'track_selection_review', 'calibrated_masks': 'calibrated_mask_set',
                'removal_freezes': 'selection_inputs', 'contribution_sweep': 'synthetic_contribution',
                'segment_cutout': 'segment_cutout_set'}
            artifacts = {names[phase]: write(tmp_path / commit[0] / (phase + '.json'), {'synthetic_phase': phase})}
            if phase == 'standard_splat_conversion':
                artifacts.update(standard_splat=standard, standard_splat_conversion_receipt=conversions[commit]['conversion'])
            if phase == 'calibrated_views':
                artifacts['calibrated_view_request'] = write(tmp_path / commit[0] / 'render-request.json',
                    {'scene': {'standard_splat_path': standard['path']}})
            if phase == 'contribution_sweep':
                paid_root = tmp_path / commit[0] / 'paid-closure'
                artifacts = {name: write(paid_root / (name + '.json'), {'synthetic': name})
                    for name in ('gaussian_allocator_result', 'gaussian_provider_execution_result', 'gaussian_contribution_evidence')}
                artifacts['gaussian_teardown'] = write(paid_root / 'teardown.json',
                    {'schema_version': 'vast_teardown_manifest.v1', 'continuing_spend_from_this_run': False})
                rows = [{'relative_path': Path(ref['path']).name, 'sha256': ref['sha256'], 'size_bytes': ref['size_bytes']}
                        for ref in artifacts.values()]
                artifacts['gaussian_artifact_manifest'] = write(paid_root / 'manifest.json',
                    {'schema_version': 'task_evaluation_artifact_manifest.v1', 'status': 'completed', 'blockers': [],
                     'binding': {'allocator_lane': 'adp_gaussian_excision', 'retry_cap': 0},
                     'files': rows, 'file_count': len(rows), 'total_size_bytes': sum(row['size_bytes'] for row in rows)},
                    'manifest_digest')
            outcome = {'status': 'completed', 'stage_id': phase, 'artifacts': artifacts}
            if phase == 'sam31_tracking':
                outcome['synthetic_tracking_producer'] = commit
            result = {'schema_version': 'task_evaluation_sam31_preparation_execution_result.v1',
                'source_commit': commit, 'status': 'completed', 'artifacts': artifacts,
                **{key: job[key] for key in ('phase', 'job_digest', 'child_id', 'parent_request_digest', 'plan_digest')}}
            write(Path(intake['result_path']), result, 'result_digest')
            write(tmp_path / 'executions' / digest[7:] / job['child_id'] / 'phase_execution_receipt.v1.json',
                {'schema_version': 'task_evaluation_sam31_phase_execution_receipt.v1',
                 'source_commit': commit, 'job_digest': job['job_digest'], 'phase': phase, 'outcome': outcome}, 'receipt_digest')
            inputs.update(artifacts)
            if phase == 'standard_splat_conversion':
                inputs['standard_splat_conversion'] = inputs['standard_splat_conversion_receipt']
        return dict(plan=plan, profile=profile, plan_ref=plan_ref, profile_ref=profile_ref, digest=digest)

    def materialize(built, successor, through, output_name):
        host = deepcopy(original_plan['host_inputs'])
        host['task_request'] = write(tmp_path / successor[0] / 'task.json',
            {'expected_production_commit': successor, 'subject': {'source_instance_id': '115'}})
        zero = write(tmp_path / (output_name + '-zero.json'), dict(provider='vast', status='observed',
            api_confirmed=True, name_prefix='', live_resource_count=0, resources=[], http=200, observed_at_epoch=1000.))
        args = dict(source_plan_path=built['plan_ref']['path'], source_profile_path=built['profile_ref']['path'],
            parent_request_digest=built['digest'], through_phase=through, current_host_inputs=host,
            current_provider_profile_path=provider['path'], current_repo_root=tmp_path / ('release-' + successor[0]),
            expected_source_commit=successor, provider_zero_path=zero['path'],
            output_path=tmp_path / (output_name + '.json'), approved_roots=(tmp_path,),
            queue_root=tmp_path / ('queue-' + built['plan']['source_commit'][0]), parent_queue_root=tmp_path / 'parents',
            execution_root=tmp_path / 'executions', now_epoch=1001., release_binding_root=bindings,
            sam31_billing_source_path=billing['path'] if adoption.PREFIX_LENGTHS[through] >= 5 else None)
        result = adoption.materialize_completed_prefix_adoption(**args)
        return result, adoption.record(args['output_path']), args
    return build, materialize, calls


def test_nested_render_adoption_then_new_tracking_keeps_distinct_producers(topology, tmp_path):
    build, materialize, calls = topology
    original = build(A, 'calibrated_views')
    first, first_ref, _ = materialize(original, B, 'calibrated_views', 'first')
    before = Path(first_ref['path']).read_bytes()
    successor = build(B, 'sam31_tracking', first_ref)
    second, second_ref, _ = materialize(successor, C, 'sam31_tracking', 'second')
    observed = adoption.validate_completed_prefix_adoption(second_ref['path'], expected_source_commit=C, approved_roots=(tmp_path,))
    assert second['original_execution_commit'] == B
    assert observed['tracking_origin']['commit'] == B
    assert second['retained_release_pin']['source_commit'] == A
    assert second['tracking_identity']['synthetic_tracking_producer'] == B
    assert [row['phase'] for row in second['phase_records']] == ['sam31_inputs', 'sam31_tracking']
    assert Path(first_ref['path']).read_bytes() == before
    assert first['source_commit'] == B and second['source_commit'] == C
    assert calls['provider'] == calls['model'] == 0


def test_existing_adoption_retry_and_successor_schedule_are_byte_stable(topology, tmp_path, monkeypatch):
    build, materialize, calls = topology
    original = build(A, 'sam31_tracking')
    result, ref, args = materialize(original, B, 'sam31_tracking', 'first')
    before = Path(ref['path']).read_bytes()
    args.update(provider_zero_path=write(tmp_path / 'fresh-zero.json', dict(provider='vast', status='observed',
        api_confirmed=True, name_prefix='', live_resource_count=0, resources=[], http=200, observed_at_epoch=2000.))['path'], now_epoch=2001.)
    assert adoption.materialize_completed_prefix_adoption(**args) == result
    assert Path(ref['path']).read_bytes() == before
    current = build(B, 'sam31_tracking', ref)  # no phases newly executed
    monkeypatch.setenv(driver.PROFILE_ENV, current['profile_ref']['path'])
    monkeypatch.setenv(driver.CHILD_QUEUE_ENV, str(tmp_path / 'next-queue'))
    expected = dict(uri='s3://synthetic/plan.json', digest=current['plan_ref']['sha256'], size_bytes=current['plan_ref']['size_bytes'])
    context = dict(expected_source_commit=B, request_digest=current['digest'],
        request=dict(preparation_id='successor', scene={'identity': current['plan']['scene_identity']},
                     task={'identity': current['plan']['task_identity']}),
        stage_one_configuration={'sam31_preparation_plan': expected},
        materialized_references=[{**expected, 'materialized_path': current['plan_ref']['path']}])
    def enqueue(**kwargs):
        calls['queue'].append(kwargs['phase'])
        return enqueue_sam31_phase(**kwargs)
    for _ in range(2):
        observed = driver.advance_sam31_preparation(context, approved_roots=(tmp_path,), enqueue_phase=enqueue)
        assert observed['phase'] == 'sam31_review'
    assert calls['queue'] == ['sam31_review', 'sam31_review']
    assert len(list((tmp_path / 'next-queue/pending').glob('*.json'))) == 1
    assert not list((tmp_path / 'next-queue/results').glob('*.json'))
    assert Path(ref['path']).read_bytes() == before
    assert calls['provider'] == calls['model'] == 0


def test_nested_tracking_adoption_keeps_original_producer_through_cutout(topology, tmp_path, monkeypatch):
    build, materialize, calls = topology
    original = build(A, 'calibrated_views')
    _, render_ref, _ = materialize(original, B, 'calibrated_views', 'render')
    tracker = build(B, 'sam31_tracking', render_ref)
    _, tracking_ref, _ = materialize(tracker, C, 'sam31_tracking', 'tracking')
    retained = {Path(ref['path']): Path(ref['path']).read_bytes() for ref in (render_ref, tracking_ref)}
    cutter = build(C, 'segment_cutout', tracking_ref)
    result, cutout_ref, _ = materialize(cutter, D, 'segment_cutout', 'cutout')
    observed = adoption.validate_completed_prefix_adoption(cutout_ref['path'], expected_source_commit=D, approved_roots=(tmp_path,))
    assert result['original_execution_commit'] == C
    assert result['source_commit'] == D
    assert observed['tracking_origin']['commit'] == B
    assert result['tracking_identity']['synthetic_tracking_producer'] == B
    assert result['retained_release_pin']['source_commit'] == A
    assert observed['phase_count'] == 10
    successor = build(D, 'segment_cutout', cutout_ref)
    monkeypatch.setenv(driver.PROFILE_ENV, successor['profile_ref']['path'])
    monkeypatch.setenv(driver.CHILD_QUEUE_ENV, str(tmp_path / 'unused-successor-queue'))
    expected = dict(uri='s3://synthetic/complete-plan.json', digest=successor['plan_ref']['sha256'],
                    size_bytes=successor['plan_ref']['size_bytes'])
    context = dict(expected_source_commit=D, request_digest=successor['digest'],
        request=dict(preparation_id='complete-successor', scene={'identity': successor['plan']['scene_identity']},
                     task={'identity': successor['plan']['task_identity']}),
        stage_one_configuration={'sam31_preparation_plan': expected},
        materialized_references=[{**expected, 'materialized_path': successor['plan_ref']['path']}])
    def unexpected_enqueue(**kwargs):
        pytest.fail('a complete ten-stage adoption must enqueue no SAM work')
    ready = driver.advance_sam31_preparation(context, approved_roots=(tmp_path,), enqueue_phase=unexpected_enqueue)
    assert ready['status'] == 'ready'
    assert not (tmp_path / 'unused-successor-queue').exists()
    assert [row['phase'] for row in result['phase_records']] == list(adoption.PHASES[5:])
    assert all(path.read_bytes() == payload for path, payload in retained.items())
    assert calls['provider'] == calls['model'] == 0


@pytest.mark.parametrize('field', ['subject', 'success', 'human_authority'])
def test_resealed_successor_cannot_change_task_science(topology, tmp_path, field):
    build, materialize, _ = topology
    original = build(A, 'sam31_tracking')
    result, ref, _ = materialize(original, B, 'sam31_tracking', 'first')
    before = Path(ref['path']).read_bytes()
    changed = deepcopy(result)
    task = json.loads(Path(result['current_host_inputs']['task_request']['path']).read_text())
    task[field] = {'subject': {'source_instance_id': '116'},
                   'success': {'minimum_lift_m': .2},
                   'human_authority': {'private_derived_frame_disclosure_authorized': False}}[field]
    changed['current_host_inputs']['task_request'] = write(tmp_path / 'changed-task.json', task)
    changed['adoption_digest'] = canonical_digest(changed, digest_field='adoption_digest')
    with pytest.raises(ValueError, match='sam31_adoption_task_or_source_changed'):
        adoption.validate_completed_prefix_adoption(changed, expected_source_commit=B, approved_roots=(tmp_path,))
    assert Path(ref['path']).read_bytes() == before


def test_failed_cutout_exposes_current_prefix_granularity(topology, tmp_path):
    """Current contract supports 3/5/10, so nine completed stages retain only five.

    This is a coverage lead about scheduling, not proof of a repeated paid call.
    Changing the admitted prefix lengths needs independent scientific validation.
    """
    build, materialize, calls = topology
    original = build(A, 'segment_cutout')
    completed, _, args = materialize(original, B, 'segment_cutout', 'completed')
    final = completed['phase_records'][-1]
    path = Path(final['result']['path'])
    failed = json.loads(path.read_text())
    failed['status'] = 'failed'
    write(path, failed, 'result_digest')
    before = path.read_bytes()
    args.pop('through_phase')
    args['output_path'] = None
    selected = adoption.select_completed_prefix_adoption(**args)
    assert selected['through_phase'] == 'sam31_tracking'
    assert 'sam31_adoption_prefix_not_terminal' in selected['rejected_candidates'][0]['blocker']
    assert adoption.PHASES[adoption.PREFIX_LENGTHS[selected['through_phase']]] == 'sam31_review'
    assert path.read_bytes() == before
    assert calls['provider'] == calls['model'] == 0
