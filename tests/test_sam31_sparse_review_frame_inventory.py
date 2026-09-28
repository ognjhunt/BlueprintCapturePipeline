"""Sparse real SAM normalization must retain every exact calibrated review frame."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import public_scene_sam31_track_selection_review as review
from blueprint_pipeline import public_scene_sam31_task_inputs as task_inputs
from blueprint_pipeline.public_scene_calibrated_object_masks import materialize_calibrated_object_mask_set
from blueprint_pipeline.scene_placement.sam31_source_track_provider import execute_sam31_source_track_request
from blueprint_pipeline.scene_placement.semantic_source_track_import import import_semantic_source_tracks
from blueprint_pipeline.task_evaluation_sam31_preparation_review_stages import _failure_blocker
from tests.test_public_scene_calibrated_object_masks import _fixture, _task_packets_for_fixture, _run_production_ai_review
from tests.test_sam31_source_track_provider import _request, FakePredictor


@pytest.fixture
def sparse(tmp_path: Path, monkeypatch):
    fixture = _fixture(tmp_path, camera_count=16)
    fixture['tasks'] = fixture['tasks'][:1]
    fixture['task_inputs'] = {'task_a': fixture['task_inputs']['task_a']}
    seed_packet = json.loads(_task_packets_for_fixture(tmp_path, fixture)[0].read_text())
    profile = tmp_path/'profile.json'
    profile.write_text(json.dumps(_request()['provider_profile']))
    prompts = tmp_path/'prompts.json'
    prompts.write_text(json.dumps([{'prompt_id': 'target', 'text': 'washer', 'output_label': 'washer'}]))
    ffmpeg = tmp_path/'ffmpeg'
    ffmpeg.write_text('fixture executable')
    def encode(*, output_path, **kwargs):
        output_path.write_bytes(b'hermetic retained sequence')
        return ['fixture-ffmpeg']
    monkeypatch.setattr(task_inputs, '_encode_lossless_sequence', encode)
    root = tmp_path/'actual-task-inputs'
    packet = task_inputs.materialize_public_scene_sam31_task_inputs(
        calibrated_view_receipt_path=seed_packet['calibrated_view_receipt']['path'],
        task_freeze_path=fixture['tasks'][0], provider_profile_path=profile,
        prompts_path=prompts, output_root=root, ffmpeg_executable=ffmpeg)
    packet_path = root/'public_scene_sam31_task_input_packet.v1.json'
    request_path = root/packet['run_request']['relative_path']
    request = json.loads(request_path.read_text())
    outputs = {}
    for index in range(16):
        masks = np.zeros((0 if index < 7 else 1, 3, 4), dtype=bool)
        if index >= 7:
            masks[0, 1, 1:3] = True
        outputs[index] = {'out_obj_ids': np.array([] if index < 7 else [0], dtype=np.int64),
                          'out_probs': np.array([] if index < 7 else [.9]), 'out_binary_masks': masks}
    result = execute_sam31_source_track_request(request, predictor_factory=lambda _: FakePredictor(outputs),
        materialized_frame_directory=Path(request['frame_artifacts'][0]['path']).parent)
    assert result['status'] == 'completed', result['blockers']
    tracks = import_semantic_source_tracks(result['source_track_import_request'], result['provider_result'])
    assert tracks['status'] == 'completed' and len(tracks['frame_masks']) == 16
    # The retained02f worker image uses the older sparse serialization: positive observations only.
    tracks['frame_masks'] = [row for row in tracks['frame_masks'] if row['track_masks']]
    tracks['bindings']['frame_masks_digest'] = review.canonical_json_digest(tracks['frame_masks'])
    tracks['result_digest'] = review.canonical_json_digest({k:v for k,v in tracks.items() if k!='result_digest'})
    assert len(tracks['frame_masks']) == 9
    source = tmp_path/'actual-sparse-source-tracks.json'
    source.write_text(json.dumps(tracks))
    prepared = tmp_path/'prepared-review'
    review.materialize_sam31_track_selection_inputs(task_input_packet_paths=[packet_path],
        source_track_result_paths_by_task={'task_a': source},
        selected_track_ids_by_task={'task_a': ['sam31-target-0']}, output_root=prepared)
    prepared_path = prepared/'public_scene_sam31_track_selection_inputs.v1.json'
    freezes, inputs, selected = review.load_validated_sam31_track_selection_inputs(prepared_path)
    return locals()


def _candidate(sparse, tmp_path):
    root = tmp_path/'candidate'
    candidate = review.materialize_sam31_track_selection_review_candidate(
        task_freeze_paths=sparse['freezes'], task_inputs=sparse['inputs'],
        selected_track_ids_by_task=sparse['selected'], output_root=root,
        prepared_inputs_path=sparse['prepared_path'])
    return candidate, root/'public_scene_sam31_track_selection_review_candidate.v1.json'


def test_sparse_import_yields_all16_sdk_review_views_and_preserves_mask_scope(sparse, tmp_path, monkeypatch):
    original = sparse['source'].read_bytes()
    candidate, candidate_path = _candidate(sparse, tmp_path)
    frames = candidate['review_media'][0]['frames']
    assert len(frames) == 16
    assert sum(frame['foreground_pixel_count'] == 0 for frame in frames) == 7
    assert sum(frame['foreground_pixel_count'] == 2 for frame in frames) == 9
    assert candidate['claim_boundary']['per_view_segmentation_completeness_qualified'] is False
    result, captured = _run_production_ai_review(monkeypatch=monkeypatch, candidate_path=candidate_path,
        output_root=tmp_path/'sdk-review', decision='accepted')
    assert len(captured['input'][0]['content'][2::2]) == 16
    assert result['decision'] == 'accepted'
    masks = materialize_calibrated_object_mask_set(task_freeze_paths=sparse['freezes'], task_inputs=sparse['inputs'],
        selected_track_ids_by_task=sparse['selected'], reviewed_track_selection_receipt_path=result['review_receipt']['path'],
        output_root=tmp_path/'calibrated-masks')
    assert masks['camera_count_total'] == 16
    assert masks['claim_boundary']['per_view_segmentation_completeness_qualified'] is False
    assert masks['claim_boundary']['removal_qualified'] is False
    assert sum(row['foreground_pixel_count'] == 0 for row in masks['tasks'][0]['masks']) == 7
    assert sparse['source'].read_bytes() == original


def test_all16_view_review_can_reject_the_candidate(sparse, tmp_path, monkeypatch):
    _, path = _candidate(sparse, tmp_path)
    result, captured = _run_production_ai_review(monkeypatch=monkeypatch, candidate_path=path,
        output_root=tmp_path/'rejected', decision='rejected')
    assert len(captured['input'][0]['content'][2::2]) == 16
    assert result['decision'] == 'rejected'


@pytest.mark.parametrize('defect', ['request_bytes_changed', 'request_missing', 'source_registry_binding_changed'])
def test_sparse_completion_requires_the_exact_retained_registry(sparse, tmp_path, defect):
    if defect == 'request_missing':
        sparse['request_path'].unlink()
    elif defect == 'request_bytes_changed':
        sparse['request_path'].write_text('{}')
    else:
        tracks = json.loads(sparse['source'].read_text())
        tracks['bindings']['frame_registry_digest'] = 'sha256:'+'f'*64
        tracks['result_digest'] = review.canonical_json_digest({k:v for k,v in tracks.items() if k!='result_digest'})
        sparse['source'].write_text(json.dumps(tracks))
    with pytest.raises(ValueError, match='retained_frame_registry_unproven'):
        _candidate(sparse, tmp_path)


def test_candidate_readback_reopens_the_registry_used_for_empty_frames(sparse, tmp_path):
    _, path = _candidate(sparse, tmp_path)
    sparse['request_path'].write_text('{}')
    with pytest.raises(ValueError, match='retained_frame_registry_unproven'):
        review.load_validated_sam31_track_selection_review_candidate(path)


def test_review_failure_retains_only_known_safe_codes():
    assert _failure_blocker(review.Sam31TrackSelectionReviewError('sam31_review_camera_frame_set_invalid')).endswith(
        ':sam31_review_camera_frame_set_invalid')
    assert _failure_blocker(ValueError('sam31_review_retained_frame_registry_unproven')).endswith(
        ':sam31_review_retained_frame_registry_unproven')
    assert _failure_blocker(review.Sam31TrackSelectionReviewError('secret-key-do-not-record')).endswith(
        ':Sam31TrackSelectionReviewError')
    # R16 (2026-09-05): the seal's code reached the operator as a bare type name. Lane-typed
    # errors carry codes, which are safe by construction; free text still becomes the type.
    assert _failure_blocker(review.Sam31TrackSelectionReviewError('sam31_review_execution_receipt_invalid')).endswith(
        ':sam31_review_execution_receipt_invalid')
    assert _failure_blocker(review.Sam31TrackSelectionReviewError('Bearer abc.def')).endswith(
        ':Sam31TrackSelectionReviewError')
    assert _failure_blocker(RuntimeError('sam31_review_execution_receipt_invalid')).endswith(':RuntimeError')
