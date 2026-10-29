"""Uncertain task destinations cannot erase independently usable room observations."""
from copy import deepcopy
import json

import pytest

from blueprint_pipeline import atlas_observed_room as room
from blueprint_pipeline import website_atlas_pipeline as pipeline
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.common import write_json
from tests.test_atlas_scene_planner import fixture


def observed_case(tmp_path):
    path, survey, proposal, policy = fixture(tmp_path)
    proposal.update(status='abstained', summary='No observed empty destination large enough for an observed box.')
    # The task region is unresolved; the separately reported room anchor is usable.
    proposal['coverage_requirements'][0]['required_frame_ids'] = []
    context = {'capture_id': 'source', 'context_digest': 'sha256:' + 'b' * 64, 'description': 'Move a box on the same shelf'}
    binding = {'source_video_digest': survey['source_video_digest'], 'survey_digest': survey['digest'],
               'task_context_digest': context['context_digest']}
    receipt = {'status': 'completed', 'binding': binding, 'binding_digest': canonical_digest(binding),
               'proposal': proposal, 'model': policy['model'], 'cost_usd': .14}
    config = {'authority': {'authorization_id': 'original', 'provider_retention': 'same-owner-policy'},
              'policy': policy, 'remaining_cost_usd': 15}
    return path, survey, proposal, context, receipt, config


def test_abstention_projects_only_observed_room_and_preserves_original_task(tmp_path):
    _, survey, proposal, _, _, config = observed_case(tmp_path)
    original = deepcopy(proposal)
    review = room.observation_review(proposal=proposal, survey=survey, policy=config['policy'], pose_budget=15)
    assert proposal == original and proposal['status'] == 'abstained'
    assert review['subject_origin'] == 'captured'
    assert review['review_scope'] == 'observed_room_only'
    assert review['coverage_requirements'] == [proposal['coverage_requirements'][1]]
    assert review['anchor_frame_id'] == '0' and review['original_task_anchor_frame_id'] == '1'
    assert not review['automation']['objects'] and not review['automation']['removal_instruction']
    assert review['task_and_surrounding_scene_covered'] is False and review['simulation_qualified'] is False


@pytest.mark.parametrize('change', ['unsafe_anchor', 'transient_anchor', 'unknown_frame', 'no_static_interval', 'no_room'])
def test_no_destination_guess_can_replace_source_or_privacy_requirements(tmp_path, change):
    _, survey, proposal, _, _, config = observed_case(tmp_path)
    if change == 'unsafe_anchor':
        proposal['unsafe_frame_ids'].append('0')
    elif change == 'transient_anchor':
        proposal['transient_frame_ids'] = ['0']
    elif change == 'unknown_frame':
        proposal['unsafe_frame_ids'].append('unobserved')
    elif change == 'no_static_interval':
        proposal['static_intervals_seconds'] = []
    else:
        proposal['coverage_requirements'] = proposal['coverage_requirements'][:1]
    with pytest.raises(ValueError, match='atlas_observed_room_'):
        room.observation_review(proposal=proposal, survey=survey, policy=config['policy'], pose_budget=15)


def test_all_49_reviewed_safe_observations_remain_candidates_without_an_arbitrary_count_cap(tmp_path):
    _, survey, proposal, _, _, config = observed_case(tmp_path)
    survey['frames'] = [{**survey['frames'][0], 'frame_id': str(i), 'timestamp_seconds': i} for i in range(49)]
    proposal.update(static_intervals_seconds=[[0, 48.5]], unsafe_frame_ids=[])
    proposal['coverage_requirements'][1]['required_frame_ids'] = [str(i) for i in range(49)]
    review = room.observation_review(proposal=proposal, survey=survey, policy=config['policy'], pose_budget=15)
    assert len(review['coverage_requirements'][0]['required_frame_ids']) == 49
    assert not review['excluded_frame_ids']


def test_room_continuation_uses_real_image_selection_privacy_screen_and_retained_inputs(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import numpy as np
    from PIL import Image
    from blueprint_pipeline import atlas_video_inputs, website_atlas_pose, paid_resource_allocator
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    _, survey, proposal, context, receipt, config = observed_case(tmp_path)
    root = tmp_path / 'atlas'
    survey_root = root / 'survey'
    survey_root.mkdir(parents=True)
    pixels = np.random.default_rng(31).integers(20, 230, (384, 512), dtype=np.uint8)
    frames = []
    for i in range(3):
        path = survey_root / f'{i}.png'
        Image.fromarray(np.roll(pixels, i * 32, axis=1)).convert('RGB').save(path)
        frames.append({'frame_id': str(i), 'timestamp_seconds': i, 'source_image_path': str(path),
            'source_image_digest': _sha256_file(path), 'sharpness': 50, 'clipped_fraction': 0})
    survey.update(schema_version=atlas_video_inputs.SURVEY_SCHEMA, frames=frames,
                  display_rotation_degrees=0, sample_count=3, heldout_frame_count=0)
    survey['digest'] = canonical_digest(survey, digest_field='digest')
    survey_path = survey_root / 'atlas_video_survey.json'
    write_json(survey_path, survey)
    proposal.update(unsafe_frame_ids=[], static_intervals_seconds=[[0, 2.5]])
    receipt['binding']['survey_digest'] = survey['digest']
    receipt['binding_digest'] = canonical_digest(receipt['binding'])
    config['policy']['reviewed_on'] = datetime.now(timezone.utc).date().isoformat()
    write_json(root / 'configuration.json', config)
    room.retain_selection(output_root=root, survey_path=survey_path, survey=survey,
        receipt=receipt, task_context=context, configuration=config)
    monkeypatch.setattr(website_atlas_pose, 'atlas_api_config', lambda: ('configured', 'test-non-secret'))
    monkeypatch.setattr(pipeline, 'load_configuration', lambda *_, **kw: deepcopy(config))
    posed = []
    def pose_boundary(**kwargs):
        posed.append(json.loads((kwargs['output_root'] / atlas_video_inputs.GEOMETRY_INPUTS_FILE).read_text()))
        return {'status': 'processing', 'operation_id': 'test-only-no-provider-call'}
    monkeypatch.setattr(paid_resource_allocator, 'run_sponsored_website_atlas_pose', pose_boundary)
    run = dict(output_root=root, source_digest=survey['source_video_digest'], task_context=context,
               max_polls=1, checkpoint=lambda _: None)
    expected = {'status': 'processing', 'stage': 'source_pose', 'operation_id': 'test-only-no-provider-call'}
    assert room.run_observed_room(**run) == expected
    selection_path = root / 'observed_room/pose/atlas_view_selection.json'
    selected = json.loads(selection_path.read_text())
    assert selected['selected_frame_count'] == 3
    assert selected['task_and_surrounding_scene_covered'] is False
    assert selected['observed_room_only'] is True
    assert {row['role'] for row in selected['coverage_requirements']} == {'surrounding_scene'}
    retained_bytes = selection_path.read_bytes()
    assert room.run_observed_room(**run) == expected
    assert posed[0] == posed[1] and selection_path.read_bytes() == retained_bytes
    assert len(posed[0]['frames']) == 3
    assert receipt['proposal']['status'] == 'abstained'


def test_same_source_resume_uses_existing_allowance_and_never_reasks_task_planner(tmp_path, monkeypatch):
    path, survey, proposal, context, receipt, config = observed_case(tmp_path)
    root = tmp_path / 'job'
    root.mkdir()
    survey_path = root / 'survey/atlas_video_survey.json'
    write_json(survey_path, survey)
    write_json(root / 'configuration.json', config)
    before = deepcopy(receipt)
    selection = room.retain_selection(output_root=root, survey_path=survey_path, survey=survey, receipt=receipt,
        task_context=context, configuration=config)
    from blueprint_pipeline import atlas_video_inputs, paid_resource_allocator, website_atlas_pose
    monkeypatch.setattr(website_atlas_pose, 'atlas_api_config', lambda: ('configured', 'test-non-secret'))
    def configuration(*args, **kwargs):
        assert kwargs == {'create': False}
        return deepcopy(config)
    monkeypatch.setattr(pipeline, 'load_configuration', configuration)
    monkeypatch.setattr(atlas_video_inputs, 'prepare_video_inputs', lambda **kw: {'frames': []})
    monkeypatch.setattr(paid_resource_allocator, 'run_sponsored_website_atlas_plan',
        lambda **_: pytest.fail('the cached task question must not be purchased again'))
    received = []
    def observed(**kwargs):
        received.append(kwargs)
        assert kwargs['reconstruction_only'] is True
        assert kwargs['task_context'] == context
        assert kwargs['review']['subject_origin'] == 'captured'
        if len(received) == 1:
            return {'status': 'processing', 'stage': 'source_pose'}
        return {'status': 'needs_input', 'source_pose': {'status': 'completed', 'pose_relative_path': 'pose'},
            'download': {'zip_path': 'retained-observations.zip'}, 'subject_origin': 'captured',
            'simulation_qualified': False, 'physical_evidence': False, 'blockers': ['metric_scene_registration_unproven']}
    monkeypatch.setattr(pipeline, '_preserved_scene', observed)
    call = dict(output_root=root, source_digest=survey['source_video_digest'], task_context=context,
                max_polls=1, checkpoint=lambda _: None)
    assert room.run_observed_room(**call)['status'] == 'processing'
    result = room.run_observed_room(**call)
    assert result['observed_scene_only'] is True and result['task_and_surrounding_scene_covered'] is False
    assert result['native_preparation_eligible'] is False and result['execution_authority_granted'] is False
    assert result['task_evidence_limitation']['summary'] == proposal['summary']
    assert result['source_pose']['pose_relative_path'] == 'observed_room/pose'
    assert receipt == before and json.loads((root / room.SELECTION).read_text()) == selection
    assert not (root / 'selection_binding.json').exists()


def test_stopped_job_adopts_actual_cached_detail_abstention_without_another_model_purchase(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from blueprint_pipeline import atlas_scene_planner, atlas_video_inputs, website_atlas_pose
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    from tests.test_atlas_scene_planner import _abstained_source_review

    options, calls, survey, policy = _abstained_source_review(tmp_path, monkeypatch, still_missing=True)
    root = tmp_path / 'retained-atlas'
    video = tmp_path / 'original.mp4'
    video.write_bytes(b'synthetic retained capture; no live provider')
    survey['source_video_digest'] = _sha256_file(video)
    survey['digest'] = canonical_digest(survey, digest_field='digest')
    survey_path = root / 'survey/atlas_video_survey.json'
    write_json(survey_path, survey)
    options.update(survey_path=survey_path, output_root=root / 'scene_planner')
    policy['reviewed_on'] = datetime.now(timezone.utc).date().isoformat()
    config = {**options['configuration'], 'authority': {'provider_retention': 'original-policy'},
              'remaining_cost_usd': 15}
    options['configuration'] = config
    cached = atlas_scene_planner.review_survey(**options)
    assert cached['binding']['source_preview_long_side'] == 1280
    assert cached['proposal']['status'] == 'abstained'
    assert len(calls['inputs']) == len(calls['reserved']) == len(calls['settled']) == 2
    write_json(root / 'configuration.json', config)
    historical_job = {'state': 'needs_input', 'reason': 'atlas_scene_planner_abstained',
                      'revision': 4, 'enqueued_at_epoch': 1234,
                      'retries': [{'reason': 'original_failure', 'at_epoch': 1250}]}
    write_json(root / 'preview_job.json', historical_job)
    before = {path: path.read_bytes() for path in (root / 'scene_planner').glob('*.json')}
    def current(*_, **kwargs):
        assert kwargs == {'create': False}
        return {**deepcopy(config), 'remaining_cost_usd': 3}
    monkeypatch.setattr(pipeline, 'load_configuration', current)
    monkeypatch.setattr(website_atlas_pose, 'atlas_api_config', lambda: ('configured', 'test-non-secret'))
    monkeypatch.setattr(atlas_video_inputs, 'prepare_video_inputs', lambda **_: {'frames': []})
    monkeypatch.setattr(pipeline, '_preserved_scene', lambda **_: {'status': 'processing', 'stage': 'source_pose'})
    options['invoker'].invoke = lambda *_: pytest.fail('cached task abstention must not be bought again')
    result = pipeline.run_atlas_pipeline(output_root=root, source_video=video,
        task_context=options['task_context'], invoker=options['invoker'])
    assert result == {'status': 'processing', 'stage': 'source_pose'}
    assert len(calls['inputs']) == len(calls['reserved']) == len(calls['settled']) == 2
    assert {path: path.read_bytes() for path in before} == before
    assert json.loads((root / 'preview_job.json').read_text()) == historical_job
    selection = json.loads((root / room.SELECTION).read_text())
    assert selection['planner_receipt'] == cached
    assert selection['review']['selection_budget']['maximum_pose_cost_usd'] == 3
    assert not (root / 'selection_binding.json').exists()


@pytest.mark.parametrize('origin', ['captured', 'synthetic_design', 'no_manipulated_subject'])
def test_visual_only_ready_can_never_enter_native_scene_or_scoring(tmp_path, origin):
    from blueprint_pipeline.website_atlas_scene_source import atlas_scene_source
    root = tmp_path / 'pipeline/atlas'
    result = {'schema_version': 'website_atlas_pipeline_result.v1', 'subject_origin': origin,
              'observed_scene_only': True, 'mesh_available': True, 'simulation_qualified': False}
    result['digest'] = canonical_digest(result, digest_field='digest')
    write_json(root / 'result.json', result)
    write_json(root / 'preview_job.json', {'state': 'ready', 'result_digest': result['digest']})
    value = atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / 'native')
    assert value == {'status': 'needs_input', 'reason': 'website_atlas_task_evidence_incomplete',
                     'observed_scene_available': True, 'simulation_qualified': False}
    assert not (tmp_path / 'native').exists()


def test_actual_preserved_adapter_exports_raw_room_without_task_review_generation_or_mesh(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_depth_splat, atlas_result_bundle, atlas_source_pose_inputs, paid_resource_allocator
    pose_root = tmp_path / 'pose'
    pose_root.mkdir()
    write_json(pose_root / 'atlas_posed_rgbd_result.json', {'digest': 'retained-raw-pose'})
    write_json(pose_root / 'scene_measurements.json', {})
    write_json(tmp_path / 'atlas_scene_review.json', {})
    original_context = {'context_digest': 'same-original-task', 'description': 'Move an observed box on the same shelf'}
    calls = []
    def source_pose(**kwargs):
        assert kwargs['allow_scale_review'] is False
        assert kwargs['task_context'] == original_context
        calls.append('observed_pose')
        return {'status': 'completed', 'pose_relative_path': 'pose', 'scale_status': 'estimated'}
    monkeypatch.setattr(pipeline, '_source_pose', source_pose)
    monkeypatch.setattr(atlas_source_pose_inputs, 'prepare_source_pose', lambda root: (root, {}))
    observed = [{'rgb': 'original box remains in its observed place', 'depth': 'provider depth estimate'}]
    monkeypatch.setattr(atlas_depth_splat, 'load_posed_rgbd', lambda _: ({'digest': 'retained-raw-pose'}, observed))
    def fuse(**kwargs):
        assert kwargs['frames'] is observed
        calls.append('source_splat')
        return {'digest': 'source-splat', 'splat_kind': 'depth_fused_draft'}
    monkeypatch.setattr(atlas_depth_splat, 'write_depth_splat', fuse)
    def no_task(**_):
        pytest.fail('an unresolved task must not buy task review, image completion or mesh execution')
    for name in ('run_sponsored_website_atlas_task', 'run_sponsored_website_atlas_automation', 'run_sponsored_website_atlas_plan'):
        monkeypatch.setattr(paid_resource_allocator, name, no_task)
    monkeypatch.setattr(atlas_result_bundle, 'export_pose_bundle', lambda **kwargs: {'zip_path': 'observed-only.zip'})
    result = pipeline._preserved_scene(pose_root=pose_root, output_root=tmp_path, inputs={},
        review={'subject_origin': 'captured', 'automation': {'objects': [], 'removal_instruction': ''},
                'uncaptured_regions': ['the task destination is unresolved']},
        policy={'mesh_export_cost_usd': 5, 'rgbd_view_cost_usd': .2}, task_context=original_context,
        source_digest='same-source', configuration={'authority': {'id': 'original'}}, check=lambda _: None,
        experimental=False, max_polls=1, model='retained-model', reconstruction_only=True,
        source_survey_path=tmp_path / 'original-survey.json')
    assert calls == ['observed_pose', 'source_splat']
    assert result['subject_origin'] == 'captured' and result['captured_scene_preserved'] is True
    assert result['object_removal_performed'] is False and result['image_inpainting_performed'] is False
    assert result['generated_completion_performed'] is False and result['mesh_available'] is False
    assert result['observed_scene_only'] is True and result['task_and_surrounding_scene_covered'] is False
    assert result['simulation_qualified'] is False and result['native_preparation_eligible'] is False


def test_automatic_controller_continues_room_work_and_resumes_without_reasking_task(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from blueprint_pipeline import atlas_video_inputs, paid_resource_allocator, website_atlas_pose
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    _, survey, proposal, context, receipt, config = observed_case(tmp_path)
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'synthetic source; no provider calls')
    survey['source_video_digest'] = _sha256_file(video)
    survey['schema_version'] = 'atlas_video_survey.v1'
    survey['digest'] = canonical_digest(survey, digest_field='digest')
    receipt['binding'].update(survey_digest=survey['digest'], source_video_digest=survey['source_video_digest'])
    receipt['binding_digest'] = canonical_digest(receipt['binding'])
    config['policy']['reviewed_on'] = datetime.now(timezone.utc).date().isoformat()
    root = tmp_path / 'atlas'
    calls, effects = [], []
    def configuration(*args, **kwargs):
        calls.append(kwargs.get('create', True))
        return deepcopy(config)
    monkeypatch.setattr(pipeline, 'load_configuration', configuration)
    monkeypatch.setattr(website_atlas_pose, 'atlas_api_config', lambda: ('configured', 'test-non-secret'))
    monkeypatch.setattr(atlas_video_inputs, 'survey_video', lambda **kw: write_json(kw['output_root'] / 'atlas_video_survey.json', survey))
    monkeypatch.setattr(atlas_video_inputs, 'prepare_video_inputs', lambda **_: {'frames': []})
    planner = []
    monkeypatch.setattr(paid_resource_allocator, 'run_sponsored_website_atlas_plan', lambda **kw: planner.append(kw) or receipt)
    def preserve(**kwargs):
        assert kwargs['reconstruction_only'] is True
        assert kwargs['task_context'] == context and kwargs['review']['subject_origin'] == 'captured'
        effects.append(kwargs)
        if len(effects) == 1:
            return {'status': 'processing', 'stage': 'source_pose'}
        return {'status': 'needs_input', 'source_pose': {'status': 'completed', 'pose_relative_path': 'pose'},
                'download': {'zip_path': 'observed-source.zip'}, 'blockers': [], 'mesh_available': False,
                'simulation_qualified': False, 'physical_evidence': False}
    monkeypatch.setattr(pipeline, '_preserved_scene', preserve)
    run = dict(output_root=root, source_video=video, task_context=context, max_polls=1)
    assert pipeline.run_atlas_pipeline(**run)['status'] == 'processing'
    result = pipeline.run_atlas_pipeline(**run)
    assert len(planner) == 1 and len(effects) == 2
    assert calls[0] is True and all(value is False for value in calls[1:])
    assert result['task_evidence_limitation']['status'] == proposal['status'] == 'abstained'
    assert result['task_and_surrounding_scene_covered'] is False and result['observed_scene_only'] is True
    assert not (root / 'selection_binding.json').exists()
