"""Source padding retains coordinates; task geometry still lifts a known plane."""
import json

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.atlas_source_pose_inputs import prepare_source_pose
from blueprint_pipeline.atlas_context_fov import full_view_sources
from blueprint_pipeline.atlas_scene_measurements import load_views
from blueprint_pipeline import website_atlas_scene_source as scene_source
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_website_atlas_scene_source import pose_root_fixture, job_fixture, UPRIGHT, MPU, BACKGROUND_SCALE


def test_preparation_retains_every_source_pixel_and_reuses_the_exact_inputs(tmp_path):
    pose = tmp_path / "pose"
    original, _ = pose_root_fixture(pose, "survey")
    root, inputs = prepare_source_pose(pose)
    assert root == pose / "full-view-source-v1"
    assert inputs["binding"]["geometry_input_digest"] != original["digest"]
    sources = full_view_sources(inputs, root)
    for before, after in zip(original["frames"], sources, strict=True):
        assert np.array_equal(np.asarray(Image.open(before["image_path"])), np.asarray(Image.open(after["image_path"])))
    assert prepare_source_pose(pose)[1] == inputs
    original["binding"]["survey_digest"] = "changed"
    original["digest"] = canonical_digest(original, digest_field="digest")
    (pose / "atlas_geometry_inputs.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="parent_changed"):
        prepare_source_pose(pose)


def test_padded_pose_keeps_reference_coordinates_and_known_plane_geometry(tmp_path, monkeypatch):
    job = job_fixture(tmp_path, padded=True)
    root = job / "pose"
    inputs = json.loads((root / "atlas_geometry_inputs.json").read_text())
    # This tiny fixture is below ORB's feature support. Verify the reference
    # mapper receives the unpadded source; geometry below uses its real mapper.
    from blueprint_pipeline import atlas_scene_measurements
    sources = []
    def map_source(path, _rgb):
        sources.append(path)
        return {"source_grid": list(Image.open(path).size)}
    monkeypatch.setattr(atlas_scene_measurements, "source_to_posed", map_source)
    _, views = load_views(root, inputs)
    assert len(sources) == 3
    assert all(view["pixel_alignment"]["source_grid"] == list(UPRIGHT) for view in views)
    result = scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert result["status"] == "ready" and result["alignment"] == "anchored"
    assert result["base_scene"]["meters_per_unit"] == pytest.approx(MPU / BACKGROUND_SCALE)


def test_source_pose_subdirectory_is_resolved_inside_the_capture_job(tmp_path):
    job = job_fixture(tmp_path)
    result = json.loads((job / "result.json").read_text())
    result["source_pose"]["pose_relative_path"] = "../../outside"
    result["digest"] = canonical_digest(result, digest_field="digest")
    (job / "result.json").write_text(json.dumps(result))
    (job / "preview_job.json").write_text(json.dumps({"state": "ready", "result_digest": result["digest"]}))
    with pytest.raises(ValueError, match="artifact_missing"):
        scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")


def task_state_fixture(tmp_path):
    pose = tmp_path / "pipeline/atlas/pose"
    original, _ = pose_root_fixture(pose, "survey")
    worker = tmp_path / "pipeline/clean_plate/source_geometry/worker_inputs"
    candidate, _ = pose_root_fixture(worker, "survey")
    for row, frame_id in zip(candidate['frames'], ['decoded-000000010', 'decoded-000000601', 'decoded-000000901']):
        row.update(frame_id=frame_id, display_rotation_degrees=90)
    candidate['digest'] = canonical_digest(candidate, digest_field='digest')
    (worker / 'geometry_inputs.json').write_text(json.dumps(candidate))
    sources = {'schema_version': 'website_source_frames.v1', 'binding': candidate['binding'],
               'geometry_input_digest': candidate['digest'], 'frames': candidate['frames']}
    sources['digest'] = canonical_digest(sources, digest_field='digest')
    manifest = tmp_path / 'pipeline/clean_plate/clean_plate_stage_manifest.json'
    manifest.write_text(json.dumps({'source_frames': sources}))
    return pose, original, worker, candidate, manifest


def test_object_pose_adds_full_video_states_without_changing_background_inputs(tmp_path):
    pose, original, worker, candidate, manifest = task_state_fixture(tmp_path)
    parent_bytes = (pose / 'atlas_geometry_inputs.json').read_bytes()
    root, inputs = prepare_source_pose(pose)
    assert root.parent.name.startswith('task-coverage-')
    assert [r['frame_id'] for r in inputs['frames']] == [r['frame_id'] for r in original['frames']] + [
        'decoded-000000601', 'decoded-000000901']
    assert (pose / 'atlas_geometry_inputs.json').read_bytes() == parent_bytes
    prepared = json.loads((root / 'atlas_geometry_inputs.json').read_text())
    assert prepared['heldout_pixels_included'] is False
    assert prepared['binding']['source_image_rotation_degrees'] == 0
    assert prepared['binding']['object_geometry_only'] is True
    assert prepared['binding']['maximum_frames'] == 5
    sources = full_view_sources(inputs, root)
    # Native, upright pixels; never the 518px geometry model's derivative.
    expected = Image.open(candidate['frames'][-1]['source_image_path']).convert('RGB').rotate(90, expand=True)
    assert np.array_equal(np.asarray(Image.open(sources[-1]['image_path'])), np.asarray(expected))
    assert Image.open(prepared['frames'][-1]['source_image_path']).size == UPRIGHT
    assert prepare_source_pose(pose) == (root, inputs)
    with open(candidate['frames'][-1]['source_image_path'], 'ab') as output:
        output.write(b'changed')
    with pytest.raises(ValueError, match='artifact_changed'):
        prepare_source_pose(pose)


@pytest.mark.parametrize('broken', ['video', 'candidate_video', 'holdout'])
def test_extra_task_views_require_the_same_video_and_admitted_candidate_split(tmp_path, broken):
    pose, _, worker, candidate, manifest = task_state_fixture(tmp_path)
    record = json.loads(manifest.read_text())
    if broken == 'video':
        record['source_frames']['binding']['source_video_digest'] = 'sha256:' + 'b' * 64
        record['source_frames']['digest'] = canonical_digest(record['source_frames'], digest_field='digest')
        manifest.write_text(json.dumps(record))
    elif broken == 'candidate_video':
        candidate['binding']['source_video_digest'] = 'sha256:' + 'b' * 64
        candidate['digest'] = canonical_digest(candidate, digest_field='digest')
        record['source_frames']['geometry_input_digest'] = candidate['digest']
        record['source_frames']['digest'] = canonical_digest(record['source_frames'], digest_field='digest')
        manifest.write_text(json.dumps(record))
        (worker / 'geometry_inputs.json').write_text(json.dumps(candidate))
    else:
        candidate['heldout_pixels_included'] = True
        (worker / 'geometry_inputs.json').write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match='atlas_task_source_'):
        prepare_source_pose(pose)
