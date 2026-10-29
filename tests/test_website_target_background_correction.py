"""A new selected instance must never inherit the old instance's edited background."""
from copy import deepcopy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_target_background_correction import derive_review, correction_root
from blueprint_pipeline.website_task_target_choice import _candidate_rows


def case():
    observation = {'source_frame_id': 'view-one', 'width': 10, 'height': 10,
                   'runs': [{'start': 26, 'length': 3}, {'start': 36, 'length': 3}]}
    track_b = {'track_id': 'box-right', 'observations': [observation]}
    review = {'subject_origin': 'captured', 'automation': {'scale_references': [],
        'objects': [{'target_id': 'atlas-box', 'label': 'cardboard box', 'segmentation_prompt': 'cardboard box at left',
                     'disposition': 'remove', 'regions': [{'frame_id': 'view-one', 'box_xywh_normalized': [0.1, 0.2, 0.2, 0.2]}]}],
        'removal_instruction': 'Remove the box at left.', 'mask_margin_fraction': 0.01}}
    target = {'target_id': 'native-box', 'task_effect': 'manipulated', 'disposition': 'remove',
              'source_track': track_b, 'track': deepcopy(track_b)}
    correction = {'target_id': target['target_id'], 'question_digest': 'sha256:' + 'a' * 64,
                  'candidate_id': _candidate_rows([track_b])[0]['candidate_id']}
    return {'review': review, 'targets': [target], 'corrections': [correction],
        'task_context': {'capture_id': 'walkthrough-job', 'context_digest': 'sha256:' + 'b' * 64},
        'parent_result': {'digest': 'sha256:' + 'c' * 64, 'source_video_digest': 'sha256:' + 'd' * 64,
                          'source_pose': {'status': 'completed', 'result_digest': 'sha256:' + 'e' * 64}},
        'task_masks_digest': 'sha256:' + 'f' * 64}


def test_b_selection_derives_only_b_pixels_and_retains_original_a_review():
    inputs = case()
    original = deepcopy(inputs)
    request = derive_review(**inputs)
    assert inputs == original
    objects = request['review']['automation']['objects']
    assert objects[0]['regions'] == [{'frame_id': 'view-one', 'box_xywh_normalized': [0.6, 0.2, 0.3, 0.2]}]
    assert 'at left' not in objects[0]['segmentation_prompt']
    assert 'at left' not in request['review']['automation']['removal_instruction']
    assert request['original_review_digest'] == canonical_digest(original['review'])
    assert request['selected_tracks']['atlas-box'] == inputs['targets'][0]['track']
    assert request['source_pose'] == inputs['parent_result']['source_pose']
    assert request['authority_issued'] is False


def test_a_track_cannot_be_used_as_the_owner_selected_b_track():
    inputs = case()
    inputs['targets'][0]['source_track']['track_id'] = 'box-left'
    with pytest.raises(ValueError, match='website_target_background_correction_candidate_changed'):
        derive_review(**inputs)


def test_unobserved_b_is_not_invented():
    inputs = case()
    inputs['targets'][0]['track']['observations'][0]['runs'] = []
    with pytest.raises(ValueError, match='website_target_background_correction_mask_binding_changed'):
        derive_review(**inputs)


def test_correction_output_is_immutable_and_scoped_to_the_atlas_job(tmp_path):
    request = derive_review(**case())
    assert correction_root(tmp_path, request['digest']).parent == tmp_path / 'target_corrections'
    with pytest.raises(ValueError, match='website_target_background_correction_digest_invalid'):
        correction_root(tmp_path, '../other')


def source_frames():
    import numpy as np
    return [{'frame_id': 'view-one', 'rgb': np.zeros((10, 10, 3), dtype='uint8'),
             'image_path': 'not-read.png', 'image_digest': 'sha256:' + '1' * 64,
             'intrinsics': {'width': 10, 'height': 10},
             'pixel_alignment': {'source_grid': [10, 10], 'matrix': np.eye(3).tolist()}}]


def test_background_uses_the_exact_selected_b_mask_without_reasking_sam(tmp_path, monkeypatch):
    from blueprint_pipeline import meta_sam31
    from blueprint_pipeline.atlas_background_automation import segment_objects
    monkeypatch.setattr(meta_sam31, 'run_meta_sam31', lambda **_: pytest.fail('selected B must not be guessed again'))
    request = derive_review(**case())
    selected = segment_objects(frames=source_frames(), objects=request['review']['automation']['objects'],
        inputs={'frames': [{'frame_id': 'view-one', 'timestamp_seconds': 4.0}]},
        task_context={}, source_digest=request['source_video_digest'], root=tmp_path,
        selected_tracks=request['selected_tracks'])
    assert selected[0]['track'] == case()['targets'][0]['track']
    assert selected[0]['track']['observations'][0]['runs'][0]['start'] == 26


@pytest.mark.parametrize('change', ['missing_frame', 'wrong_grid', 'guide_only'])
def test_source_mask_substitution_is_refused_before_background_edit(tmp_path, change):
    from blueprint_pipeline.atlas_background_automation import segment_objects
    request = derive_review(**case())
    selected = request['selected_tracks']
    if change == 'missing_frame':
        selected['atlas-box']['observations'] = []
    elif change == 'wrong_grid':
        selected['atlas-box']['observations'][0]['width'] = 11
    else:
        selected['atlas-box']['guide_only'] = True
    with pytest.raises(ValueError, match='atlas_selected_source_track_'):
        segment_objects(frames=source_frames(), objects=request['review']['automation']['objects'],
            inputs={'frames': [{'frame_id': 'view-one', 'timestamp_seconds': 4.0}]},
            task_context={}, source_digest=request['source_video_digest'], root=tmp_path,
            selected_tracks=selected)


def rejected_job(tmp_path):
    import json
    from blueprint_pipeline import website_task_target_choice as choice
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    from tests.test_website_task_target_choice import _question, _answer
    capture = tmp_path / 'scenes/site-req/captures/walkthrough-req'
    pipeline = capture / 'pipeline'
    pipeline.mkdir(parents=True)
    video = capture / 'saved-video.mp4'
    video.write_bytes(b'synthetic retained footage; never uploaded')
    context = {'request_id': 'req', 'scene_id': 'site-req', 'capture_id': 'walkthrough-req',
               'description': 'Move the tote', 'task_family': 'rigid_pick_place'}
    context['context_digest'] = canonical_digest(context, digest_field='context_digest')
    question = _question(scene_id=context['scene_id'], capture_id=context['capture_id'])
    question['source_video_digest'] = _sha256_file(video)
    question['question_digest'] = canonical_digest(question, digest_field='question_digest')
    selectors = {key: context[key] for key in ('request_id', 'scene_id', 'capture_id')}
    selectors.update(completion_marker_generation='1', producer_delivery_key='sha256:' + '1' * 64,
                     source_payload_sha256='2' * 64, task_context_digest=context['context_digest'])
    answer = _answer(question, selectors, {'kind': 'none_of_these'})
    for path, value in [
        (choice._path(pipeline, question['target_id'], 'question'), question),
        (choice._path(pipeline, question['target_id'], 'answer'), answer),
        (pipeline / choice.QUESTION_DIR / choice.INDEX_FILE,
         {'schema_version': choice.INDEX_SCHEMA, 'questions': {question['target_id']: question['question_digest']}}),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return capture, context, video


def test_negative_answer_runs_real_preparation_once_without_model_native_or_upload(tmp_path):
    from types import SimpleNamespace
    from blueprint_pipeline import website_task_target_choice as choice
    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    from blueprint_pipeline.website_preparation_outcome import read_preparation_outcome
    from blueprint_pipeline.website_target_background_correction import retains_rejected_selection
    capture, context, video = rejected_job(tmp_path)
    digest = choice.correction_binding_digest(capture / 'pipeline')
    before = {'status': 'completed', 'attempt_history': []}
    assert choice.correction_pending(before, capture / 'pipeline') == digest
    descriptor = SimpleNamespace(metadata={'site_task_context': context, 'original_grant': {'unchanged': True}})
    def forbidden(**_kwargs):
        pytest.fail('negative answer must not replay a guessed target or buy another model answer')
    result = run_website_capture_preparation(descriptor=descriptor, capture_root=capture, pipeline_dir=capture / 'pipeline',
        source_video=video, derived_scene_generation_allowed=True, gates=[], prepare_media=forbidden,
        check_privacy=forbidden, make_gate=forbidden)
    assert result['status'] == 'completed' and result['native_execution_complete'] is False
    handoff = result['website_scene_preparation']
    assert handoff['status'] == 'needs_input' and handoff['simulator_ready'] is False
    assert retains_rejected_selection(handoff, context)
    assert handoff['selection_rejection']['additional_footage_required'] is False
    assert descriptor.metadata['original_grant'] == {'unchanged': True}
    after = {'status': 'completed', 'attempt_history': [{choice.CORRECTION_MARK: digest}]}
    assert choice.correction_pending(after, capture / 'pipeline') is None
    assert read_preparation_outcome(capture_root=capture, task_context_digest=context['context_digest'],
                                    state='awaiting_inputs')['category'] == 'insufficient_evidence'


def test_negative_answer_cannot_apply_to_a_different_current_task(tmp_path):
    from blueprint_pipeline.website_target_background_correction import retain_rejected_selection
    capture, context, video = rejected_job(tmp_path)
    context = {**context, 'context_digest': 'sha256:' + '9' * 64}
    assert retain_rejected_selection(capture_root=capture, task_context=context, source_video=video) is None
    assert not (capture / 'pipeline/website_scene_preparation/handoff.json').exists()
