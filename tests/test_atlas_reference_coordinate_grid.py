"""Retained review coordinates belong to their original image grid, even after widening pose."""
import json
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline.atlas_scene_measurements import source_to_posed, mapped_pixels
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_atlas_context_fov import texture
from tests.test_website_atlas_scene_source import job_fixture, digest, write, MPU, BACKGROUND_SCALE


def test_a_reference_on_an_earlier_crop_is_registered_to_the_full_image(tmp_path):
    full = texture()
    crop = cv2.resize(full[112:400, 63:225], (288, 512))
    path = tmp_path / 'old-grid.png'
    Image.fromarray(crop).save(path)
    alignment = source_to_posed(path, full)
    pixels = mapped_pixels({'pixel_alignment': alignment}, [[.1, .1], [.4, .7]])
    expected = np.array([[63 + .1 * 162, 112 + .1 * 288], [63 + .4 * 162, 112 + .7 * 288]])
    assert np.allclose(pixels, expected, atol=1.)
    assert not np.allclose(pixels, np.array([[.1, .1], [.4, .7]]) * [288, 512], atol=1.)


@pytest.mark.parametrize('additional_task_views', [False, True])
@pytest.mark.parametrize('repair_legacy_scale', [False, True])
def test_retained_reference_inputs_select_a_distinct_measurement_receipt(tmp_path, monkeypatch, additional_task_views, repair_legacy_scale):
    from blueprint_pipeline import website_atlas_pipeline as pipeline, atlas_source_pose_inputs as preparation
    from blueprint_pipeline import atlas_scene_measurements as measurements, atlas_context_fov as fov
    from blueprint_pipeline import atlas_scene_automation as automation, paid_resource_allocator as allocator

    root = tmp_path / 'pose'
    child = root / 'full-view-source-v1'
    child.mkdir(parents=True)
    (child / 'atlas_posed_rgbd_result.json').write_text('{}')
    native = {'binding': {'source_video_digest': 'video'}, 'frames': [{'frame_id': 'a'}, {'frame_id': 'b'}], 'digest': 'native'}
    references = {**native, 'heldout_pixels_included': False}
    references['digest'] = canonical_digest(references, digest_field='digest')
    expected_references = references
    if additional_task_views:
        extra = {'frame_id': 'closed', 'image_path': 'full-closed.png'}
        native = {**native, 'frames': [*native['frames'], {**extra, 'image_path': 'padded-closed.png'}]}
        full_sources = [{**row, 'image_path': 'full-' + row['frame_id'] + '.png'} for row in native['frames']]
        monkeypatch.setattr(fov, 'full_view_sources', lambda inputs, _: full_sources)
        expected_references = {**references, 'frames': [*references['frames'], extra]}
    monkeypatch.setattr(preparation, 'prepare_source_pose', lambda _: (child, native))
    calls = []
    monkeypatch.setattr(allocator, 'run_sponsored_website_atlas_pose', lambda **_: {'status': 'completed'})
    def load(pose_root, inputs):
        calls.append(inputs)
        return {'digest': 'pose', 'operation': {'id': 'source'}}, ['mapped' if inputs == expected_references else 'full']
    monkeypatch.setattr(measurements, 'load_views', load)
    monkeypatch.setattr(fov, 'verify_pose_context_fov', lambda **kw: calls.append(kw['frames']))
    monkeypatch.setattr(automation, 'measurement_plan', lambda _: {})
    outputs = []
    monkeypatch.setattr(measurements, 'estimate_sizes', lambda **kw: outputs.append(kw) or {
        'digest': 'sizes', 'meters_per_unit_estimate': 1.45, 'scale_status': 'estimated_from_named_assumptions'})
    repaired = []
    if repair_legacy_scale:
        from blueprint_pipeline import atlas_scale_review
        def repair(**kwargs):
            repaired.append(kwargs)
            return ({'status': 'estimated', 'digest': 'revised-sizes', 'meters_per_unit_estimate': 1.6,
                     'scale_status': 'estimated_from_named_assumptions'}, child / 'scale_review/measurements/revised.json')
        monkeypatch.setattr(atlas_scale_review, 'repair_source_scale', repair)
    policy = {'pose_input_cost_usd': .01, 'rate_card_reference': 'rates', 'reviewed_by': 'test',
              'reviewed_on': datetime.now(timezone.utc).date().isoformat()}
    result = pipeline._source_pose(pose_root=root, inputs=native, review={}, policy=policy,
        task_context={'capture_id': 'capture'}, check=lambda _: pytest.fail('Completed pose must be reused'),
        experimental=False, max_polls=1, reference_inputs=references,
        configuration={} if repair_legacy_scale else None)
    assert result['status'] == 'completed'
    assert calls == [native, ['full'], expected_references]
    if not additional_task_views:
        assert calls[-1] is references
    assert outputs[0]['frames'] == ['mapped']
    if repair_legacy_scale:
        assert result['measurements_relative_path'] == 'scale_review/measurements/revised.json'
        assert result['meters_per_unit_estimate'] == 1.6
        assert repaired[0]['inputs'] == native and repaired[0]['previous']['digest'] == 'sizes'
    else:
        assert outputs[0]['output'].name == result['measurements_relative_path']
    assert outputs[0]['output'].name != 'scene_measurements.json'
    assert result['measurement_reference_inputs_digest'] == (native['digest'] if repair_legacy_scale else references['digest'])
    changed = {**references, 'binding': {'source_video_digest': 'different'}}
    changed['digest'] = canonical_digest(changed, digest_field='digest')
    refused = pipeline._source_pose(pose_root=root, inputs=native, review={}, policy=policy,
        task_context={'capture_id': 'capture'}, check=lambda _: None, experimental=False,
        max_polls=1, reference_inputs=changed)
    assert refused == {'status': 'needs_input', 'reason': 'atlas_source_reference_inputs_changed'}


def test_scene_reader_uses_the_reference_bound_measurement_file(tmp_path):
    from blueprint_pipeline.website_atlas_scene_source import atlas_scene_source
    job = job_fixture(tmp_path)
    original = job / 'pose/scene_measurements.json'
    target = original.with_name('scene_measurements_reference-grid.json')
    original.rename(target)
    result = json.loads((job / 'result.json').read_text())
    result['source_pose']['measurements_relative_path'] = target.name
    write(job / 'result.json', digest(result))
    write(job / 'preview_job.json', {'state': 'ready', 'result_digest': result['digest']})
    ready = atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / 'prep')
    assert ready['status'] == 'ready'
    assert ready['base_scene']['meters_per_unit'] == pytest.approx(MPU / BACKGROUND_SCALE)
