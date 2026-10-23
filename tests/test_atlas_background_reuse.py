"""Expanding a source selection must not buy the accepted edits again."""
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline import atlas_background_automation as background
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def image(path, value):
    Image.fromarray(np.full((16, 20, 3), value, np.uint8)).save(path)
    return str(path), _sha256_file(path)


def review(frames):
    binding = {'frames': [{k: row[k] for k in ('frame_id', 'image_digest')} for row in frames]}
    return {'status': 'passed', 'binding': binding, 'request_digest': canonical_digest(binding),
        'review': dict(consistent_background=True, task_objects_removed=True, unrelated_objects_preserved=True)}


def fixture(root):
    original, digest = image(root / 'source.png', 40)
    edited, edited_digest = image(root / 'accepted.png', 80)
    old = {'frame_id': 'old', 'image_path': edited, 'image_digest': edited_digest,
        'original_image_path': original, 'original_image_digest': digest,
        'remaining_pixel_count': 0, 'removed_task_object_ids': ['dishwasher']}
    context = {'request_id': 'same-capture'}
    binding = {'source_video_digest': 'video', 'task_context_digest': canonical_digest(context)}
    prior = root / 'prepared.json'
    background._prepared_background(prior, binding, {'status': 'fallback_prepared', 'frames': [old],
        'review': review([old]), 'ready_for_reconstruction': True})
    return prior, old, context, binding


def test_reuse_requires_same_capture_source_pixels_and_unchanged_retained_bytes(tmp_path):
    prior, old, context, _ = fixture(tmp_path)
    options = dict(source_digest='video', context=context,
        frames=[{'frame_id': 'old', 'rgb': np.full((16, 20, 3), 40, np.uint8)}],
        objects=[{'target_id': 'dishwasher', 'disposition': 'remove'}])
    reused, digest = background.reusable_background(prior, **options)
    assert reused == [old] and digest
    with pytest.raises(ValueError, match='source_or_context_changed'):
        background.reusable_background(prior, **{**options, 'context': {'request_id': 'another-capture'}})
    options['frames'][0]['rgb'][0, 0] = 41
    with pytest.raises(ValueError, match='source_pixels_changed'):
        background.reusable_background(prior, **options)
    Path(old['image_path']).write_bytes(b'changed accepted edit')
    with pytest.raises(ValueError, match='pixels_changed'):
        background.reusable_background(prior, **options)


def test_expanded_set_edits_only_new_view_and_reviews_both_without_rewriting_prior(tmp_path, monkeypatch):
    from blueprint_pipeline import website_image_completion as editor, website_image_repair_agent

    prior, old, context, binding = fixture(tmp_path)
    prior_bytes, accepted_bytes = prior.read_bytes(), Path(old['image_path']).read_bytes()
    path, digest = image(tmp_path / 'new-source.png', 60)
    prepared = [{**old, 'image_path': old['original_image_path'], 'image_digest': old['original_image_digest'],
                 'remaining_pixel_count': 10},
        {'frame_id': 'new', 'image_path': path, 'image_digest': digest, 'original_image_path': path,
         'original_image_digest': digest, 'remaining_pixel_count': 20, 'removed_task_object_ids': ['dishwasher']}]
    edited, edited_digest = image(tmp_path / 'new-edited.png', 90)
    calls, reviewed = [], []
    def complete(**kw):
        if kw.get('require_cached'):
            raise ValueError('website_image_completion_cached_batch_missing')
        calls.append([r['frame_id'] for r in kw['frames']])
        return [{**kw['frames'][0], 'image_path': edited, 'image_digest': edited_digest, 'remaining_pixel_count': 0}]
    def verify(**kw):
        reviewed.append(kw['frames'])
        return review(kw['frames'])
    monkeypatch.setattr(editor, 'complete_background_images', complete)
    monkeypatch.setattr(editor, 'verify_completed_background', verify)
    monkeypatch.setattr(website_image_repair_agent, 'image_repair_enabled', lambda: False)
    monkeypatch.setattr(background, '_provider_preflight', lambda: None)
    monkeypatch.setattr(background, 'object_evidence', lambda **_: {})
    segmented = []
    def segment(**kw):
        segmented.append([f['frame_id'] for f in kw['frames']])
        return [{'disposition': 'remove'}]
    monkeypatch.setattr(background, 'segment_objects', segment)
    monkeypatch.setattr(background, 'prepare_masked_frames', lambda **_: [prepared[1]])
    monkeypatch.setattr(editor, '_validated_backend', lambda *_a, **_kw:
                        ({}, {'pricing_binding': {'max_cost_per_request_usd': 1}}, 'backend'))
    root = tmp_path / 'expanded'
    root.mkdir()
    frames = [{'frame_id': key, 'rgb': np.full((16, 20, 3), value, np.uint8),
               'intrinsics': {'width': 20, 'height': 16}} for key, value in [('old', 40), ('new', 60)]]
    actual = background.run_background(frames=frames,
        objects=[{'target_id': 'dishwasher', 'label': 'Dishwasher', 'disposition': 'remove',
                  'regions': [{'frame_id': 'new', 'box_xywh_normalized': [.1, .1, .8, .8]}]}],
        automation={'removal_instruction': 'Remove dishwasher', 'mask_margin_fraction': .01, 'anchor_frame_id': 'old'},
        inputs={'digest': 'inputs'}, pose={'digest': 'pose'}, root=root, context=context,
        policy={'rgbd_view_cost_usd': .05}, source_digest='video', task_runner=lambda **_: pytest.fail('Atlas upload before review'),
        release_blockers=lambda: ('commit', []), selected_frame_ids=['old', 'new'], prior_background_path=prior)
    assert calls == [['new']]
    assert segmented == [['new']]
    assert [r['frame_id'] for r in reviewed[0]] == ['old', 'new']
    assert actual['reused_prepared_frame_ids'] == ['old'] and actual['frames'][0] == old
    assert prior.read_bytes() == prior_bytes and Path(old['image_path']).read_bytes() == accepted_bytes
    import json
    new_binding = json.loads((root / 'prepared_background.json').read_text())['binding']
    assert background._prepared_background(root / 'prepared_background.json', new_binding) == actual
