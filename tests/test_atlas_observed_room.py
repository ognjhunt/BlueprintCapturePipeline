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
    # A final abstention under the current instructions: no frame shows the task object.
    proposal.update(status='abstained', summary='No frame shows the box the task names.',
                    abstention_reasons=['task_object_or_region_not_visible'])
    # The task region is unresolved; the separately reported room anchor is usable.
    proposal['coverage_requirements'][0]['required_frame_ids'] = []
    context = {'capture_id': 'source', 'context_digest': 'sha256:' + 'b' * 64, 'description': 'Move a box on the same shelf'}
    from blueprint_pipeline.atlas_scene_planner import INSTRUCTIONS_REVISION
    binding = {'source_video_digest': survey['source_video_digest'], 'survey_digest': survey['digest'],
               'task_context_digest': context['context_digest'], 'instructions_revision': INSTRUCTIONS_REVISION}
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
    monkeypatch.setattr(pipeline, '_preserved_scene', lambda **_: pytest.fail('a final abstention buys no room'))
    options['invoker'].invoke = lambda *_: pytest.fail('cached task abstention must not be bought again')
    # A final abstention (part 6, 2026-11-12): the ask for a new video, with no room bought for it.
    with pytest.raises(ValueError, match='website_capture_frames_unusable|website_task_object_not_observed'):
        pipeline.run_atlas_pipeline(output_root=root, source_video=video,
            task_context=options['task_context'], invoker=options['invoker'])
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
    (tmp_path / 'case').mkdir()
    *_, receipt, _ = observed_case(tmp_path / 'case')
    write_json(root / room.SELECTION, {'planner_receipt': receipt})
    value = atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / 'native')
    assert value == {'status': 'needs_input', 'reason': 'website_atlas_task_evidence_incomplete',
                     'observed_scene_available': True, 'simulation_qualified': False}
    assert not (tmp_path / 'native').exists()
    # An abstention that is not final is the job's usual wait while its task is reviewed once more.
    older = {**receipt, 'binding': {key: value for key, value in receipt['binding'].items()
                                    if key != 'instructions_revision'}}
    write_json(root / room.SELECTION, {'planner_receipt': older})
    assert atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / 'native') == {
        'status': 'pending', 'reason': 'website_atlas_job_processing', 'stage': 'observed_room_review'}
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


def _controller(tmp_path, monkeypatch):
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
    calls, effects, planner = [], [], []
    def configuration(*args, **kwargs):
        calls.append(kwargs.get('create', True))
        return deepcopy(config)
    monkeypatch.setattr(pipeline, 'load_configuration', configuration)
    monkeypatch.setattr(website_atlas_pose, 'atlas_api_config', lambda: ('configured', 'test-non-secret'))
    monkeypatch.setattr(atlas_video_inputs, 'survey_video', lambda **kw: write_json(kw['output_root'] / 'atlas_video_survey.json', survey))
    monkeypatch.setattr(atlas_video_inputs, 'prepare_video_inputs', lambda **_: {'frames': []})
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
    return root, run, proposal, receipt, calls, effects, planner


@pytest.mark.parametrize('website_lists_reason', [False, True])
def test_a_final_abstention_buys_no_room_and_asks_for_a_new_video(tmp_path, monkeypatch, website_lists_reason):
    """Part 6 (2026-11-12): a final abstention bought the observed room's pose although the site is asked
    to film again either way. Now its selection is kept at $0 and no room is bought."""
    from blueprint_pipeline.website_preparation_outcome import NEW_REASONS_ENV
    monkeypatch.setenv(NEW_REASONS_ENV, 'true' if website_lists_reason else 'false')
    root, run, _, _, _, effects, planner = _controller(tmp_path, monkeypatch)
    for _ in range(2):  # The retained selection decides the second run too, without the review.
        with pytest.raises(ValueError) as stop:
            pipeline.run_atlas_pipeline(**run)
        assert str(stop.value) == ('website_task_object_not_observed' if website_lists_reason
                                   else 'atlas_scene_planner_abstained:final')
    assert len(planner) == 1 and effects == [] and (root / room.SELECTION).is_file()
    assert not (root / 'observed_room/pose').exists()
    from blueprint_pipeline import website_atlas_preview as preview
    job = {'state': 'processing', 'reason': None, 'revision': 1, 'stage': 'plan'}
    assert preview._failed_step(job, stop.value, 10)['state'] == 'needs_input'


def test_a_room_already_bought_for_a_final_abstention_is_still_collected(tmp_path, monkeypatch):
    """A purchase made before this change is finished and collected, never abandoned."""
    root, run, proposal, _, calls, effects, planner = _controller(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        pipeline.run_atlas_pipeline(**run)
    (root / 'observed_room/pose').mkdir(parents=True)  # bought under the earlier rule, in flight
    assert pipeline.run_atlas_pipeline(**run)['status'] == 'processing'
    result = pipeline.run_atlas_pipeline(**run)
    assert len(planner) == 1 and len(effects) == 2
    assert result['task_evidence_limitation']['status'] == proposal['status'] == 'abstained'
    assert result['task_and_surrounding_scene_covered'] is False and result['observed_scene_only'] is True
    assert not (root / 'selection_binding.json').exists()


def test_an_abstention_that_is_not_final_reviews_the_task_once_more(tmp_path, monkeypatch):
    """capture-f88f1bd0: an abstention under older instructions was every later run's answer."""
    from datetime import datetime, timezone
    from blueprint_pipeline import atlas_video_inputs, paid_resource_allocator, website_atlas_pose
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    _, survey, proposal, context, receipt, config = observed_case(tmp_path)
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'synthetic source; no provider calls')
    survey['source_video_digest'] = _sha256_file(video)
    survey['schema_version'] = 'atlas_video_survey.v1'
    survey['digest'] = canonical_digest(survey, digest_field='digest')
    config['policy']['reviewed_on'] = datetime.now(timezone.utc).date().isoformat()
    root = tmp_path / 'atlas'
    older = {**receipt, 'binding': {key: value for key, value in receipt['binding'].items()
                                    if key != 'instructions_revision'}}
    older['proposal'] = {**proposal, 'abstention_reasons': None}
    selection = {'planner_receipt': older, 'digest': 'sha256:' + 'a' * 64}
    write_json(root / room.SELECTION, selection)
    write_json(root / 'configuration.json', config)
    (root / 'scene_planner').mkdir()
    monkeypatch.setattr(pipeline, 'load_configuration', lambda *a, **k: deepcopy(config))
    monkeypatch.setattr(website_atlas_pose, 'atlas_api_config', lambda: ('configured', 'test-non-secret'))
    monkeypatch.setattr(atlas_video_inputs, 'survey_video', lambda **kw: write_json(kw['output_root'] / 'atlas_video_survey.json', survey))
    monkeypatch.setattr(room, 'run_observed_room', lambda **_: pytest.fail('a non-final room is not the answer'))

    class Reviewed(Exception):
        pass

    def review(**kwargs):
        raise Reviewed()
    monkeypatch.setattr(paid_resource_allocator, 'run_sponsored_website_atlas_plan', review)
    with pytest.raises(Reviewed):
        pipeline.run_atlas_pipeline(output_root=root, source_video=video, task_context=context, max_polls=1)
    # The retained room is kept beside the job, never deleted, and the task is reviewed again.
    assert not (root / room.SELECTION).exists()
    assert json.loads((root / 'observed_room/superseded' / ('a' * 64 + '.selection.json')).read_text()) == selection


def _ready_observed_job(tmp_path, *, age_days, grant_seconds_left=86_400 * 3, ledger=None):
    import time
    *_, receipt, _ = observed_case(tmp_path)
    capture = tmp_path / 'scenes/site-x/captures/walkthrough-x'
    root = capture / 'pipeline/atlas'
    older = {**receipt, 'binding': {}}  # Abstained under older instructions: not final.
    write_json(root / room.SELECTION, {'planner_receipt': older, 'digest': 'sha256:' + 'f' * 64})
    now = int(time.time())
    if grant_seconds_left is not None:
        write_json(capture / 'pipeline/website_scene_sponsorship.json', {'expires_at_epoch': now + grant_seconds_left})
    write_json(capture / 'pipeline_job_ledger.json', ledger or {'status': 'retryable_blocked'})
    job = {'schema_version': 'website_atlas_preview_job.v1', 'state': 'ready', 'stage': 'archived', 'reason': None,
           'revision': 4, 'enqueued_at_epoch': now - int(age_days * 86_400), 'next_attempt_at_epoch': 0,
           'retry_at_epoch': None, 'transient_failures': 0, 'request_id': 'x', 'scene_id': 'site-x',
           'capture_id': 'walkthrough-x', 'source_video': 'raw/v.mp4'}
    write_json(root / 'preview_job.json', job)
    (root / 'preview.lock').touch()
    return root / 'preview_job.json', job, now


def _advance(monkeypatch, path, step):
    from blueprint_pipeline import website_atlas_preview as preview
    monkeypatch.setattr(preview, '_withdrawn', lambda root: False)
    monkeypatch.setattr(preview, '_step', step)
    monkeypatch.setattr(preview, '_deliver', lambda job, now: job)
    monkeypatch.setattr(preview, '_clean_provider', lambda job, **_: job)
    monkeypatch.setattr(preview, '_retract_archive', lambda job: job)
    preview.advance_job(path)
    return json.loads(path.read_text())


@pytest.mark.parametrize('case', ['expired_window', 'no_grant', 'ended', 'closed_by_owner'])
def test_an_observed_room_job_that_cannot_finish_is_never_touched(tmp_path, monkeypatch, case):
    """Review 2026-11-12: a reopened job past its window failed at once and lost its ready preview."""
    options = {'expired_window': dict(age_days=15), 'no_grant': dict(age_days=1, grant_seconds_left=None),
               'ended': dict(age_days=1, ledger={'status': 'terminal_authority_ended', 'terminal_code': 'consent_expired'}),
               'closed_by_owner': dict(age_days=1, ledger={'status': 'failed_retryable',
                                                          'last_error': 'website_control_task-context_http_409:website_run_closed_by_owner'})}[case]
    path, job, _ = _ready_observed_job(tmp_path, **options)
    after = _advance(monkeypatch, path, lambda job, **_: pytest.fail('a ready job is never stepped'))
    assert after == job


def test_an_observed_room_job_inside_its_window_is_reviewed_once_and_never_ended_by_it(tmp_path, monkeypatch):
    from blueprint_pipeline import website_atlas_preview as preview
    path, job, now = _ready_observed_job(tmp_path, age_days=1)
    stepped = []

    def refused(job, **_):
        stepped.append(job['stage'])
        return preview._transition(job, now, 'failed', 'atlas_runtime_budget_insufficient')
    after = _advance(monkeypatch, path, refused)
    # Reviewed once; the review could not run, so the job is the ready job it was, marked as reopened.
    assert stepped == ['observed_room_review']
    assert after['state'] == 'ready' and after['stage'] == 'archived' and after['revision'] == job['revision']
    assert after['observed_room_reopened'] == 'sha256:' + 'f' * 64
    assert after['observed_room_review_outcome'] == {'state': 'failed', 'reason': 'atlas_runtime_budget_insufficient'}
    again = _advance(monkeypatch, path, lambda job, **_: pytest.fail('reopened only once'))
    assert again == after


def test_a_room_purchase_in_flight_is_collected_before_its_selection_is_set_aside(tmp_path, monkeypatch):
    """Review 2026-11-12: a release landing while the room is being built must not abandon that purchase."""
    _, _, proposal, context, receipt, config = observed_case(tmp_path)
    root = tmp_path / 'atlas'
    older = {**receipt, 'binding': {}}
    write_json(root / room.SELECTION, {'planner_receipt': older, 'digest': 'sha256:' + 'a' * 64})
    (root / 'observed_room/pose').mkdir(parents=True)
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'synthetic source; no provider calls')
    collected = []
    monkeypatch.setattr(room, 'run_observed_room', lambda **kw: collected.append(kw) or {'status': 'processing'})
    assert pipeline.run_atlas_pipeline(output_root=root, source_video=video, task_context=context,
                                       max_polls=1)['status'] == 'processing'
    assert len(collected) == 1 and (root / room.SELECTION).is_file()
