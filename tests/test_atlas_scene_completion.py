"""Generated views must contribute bound geometry, not just downloaded pictures."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline import atlas_scene_completion as completion
from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def frame(key="a", x=0):
    return posed_frame(frame_id=key, rgb=np.full((8, 6, 3), 100, np.uint8),
        depth=np.full((8, 6), 2, np.float32), confidence=np.ones((8, 6), np.float32),
        camera={"intrinsics": {"width": 6, "height": 8, "fx": 5, "fy": 5, "cx": 3, "cy": 4},
                "extrinsics": {"position": [x, 0, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"}})


def test_completion_camera_pool_preserves_vertical_rays_and_observed_camera_centers():
    first = frame()
    candidates = completion.camera_candidates([first])
    assert len({r["camera_id"] for r in candidates}) == len(candidates)
    for row in candidates:
        k = row["camera"]["intrinsics"]
        assert (k["width"], k["height"]) == (1280, 720)
        assert row['purpose'] in {'wide', 'left', 'right', 'lower', 'higher'}
        assert k['fx'] == k['fy'] == pytest.approx(600.)
    scaled = deepcopy(first)
    scaled["depth"] *= 10
    scaled["position"] *= 10
    scaled["extrinsics"]["position"] = scaled["position"].tolist()
    for before, after in zip(candidates, completion.camera_candidates([scaled]), strict=True):
        assert np.allclose(np.array(before["camera"]["extrinsics"]["position"]) * 10,
                           after["camera"]["extrinsics"]["position"])
    first["depth"][:] = np.nan
    with pytest.raises(ValueError, match="no_context_depth"):
        completion.camera_candidates([first])


def test_room_center_is_an_actual_capture_position_instead_of_an_unobserved_coordinate_median():
    frames = [frame(str(i)) for i in range(4)]
    for f, position in zip(frames, ([0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]), strict=True):
        f['position'] = np.array(position, dtype=float)
        f['extrinsics']['position'] = list(position)
    allowed = {tuple(f['position']) for f in frames}
    candidates = completion.camera_candidates(frames)
    assert all(tuple(row['camera']['extrinsics']['position']) in allowed for row in candidates)
    assert all(tuple(row['camera']['extrinsics']['position']) != (1., 1., 0.) for row in candidates)
    by_id = {f['frame_id']: f for f in frames}
    assert all(np.array_equal(row['camera']['extrinsics']['position'], by_id[row['source_frame_id']]['position'])
               for row in candidates)


def test_portrait_target_preserves_original_rays_without_expanding_into_new_room_directions():
    from blueprint_pipeline.atlas_scene_preservation import _world, _project
    source = frame()
    source['rgb'] = np.full((1280, 720, 3), 100, np.uint8)
    source['depth'] = np.full((1280, 720), 2., np.float32)
    source['confidence'] = np.ones((1280, 720), np.float32)
    source['intrinsics'] = {'width': 720, 'height': 1280, 'fx': 600., 'fy': 600., 'cx': 360., 'cy': 640.}
    rows = completion.camera_candidates([source])
    assert len(rows) == 5 and not any(row['camera_id'].startswith('room-') for row in rows)
    points, y, x = _world(source, np.ones((1280, 720), bool))
    _, u, v, inside = _project(points, rows[0]['camera'])
    assert inside.all()
    assert np.allclose(u, 1280 - (y + .5)) and np.allclose(v, x + .5)


def test_generated_loader_binds_every_pixel_and_keeps_generated_camera(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_depth_splat
    image, depth = tmp_path / "image.png", tmp_path / "depth.exr"
    Image.fromarray(frame()["rgb"]).save(image)
    depth.write_bytes(b"retained-exr")
    monkeypatch.setattr(atlas_depth_splat, "read_exr_depth", lambda _: np.full((8, 6), 2, np.float32))
    cam = {"intrinsics": frame()["intrinsics"], "extrinsics": frame(x=.2)["extrinsics"]}
    result = {"status": "completed", "operation": {"response": {"frames": [{"camera": cam}]}},
              "outputs": [{"role": role, "path": p.name, "sha256": _sha256_file(p)} for role, p in
                          (("frames-0-imageAsset", image), ("frames-0-depth-depthAsset", depth))]}
    result["digest"] = canonical_digest(result, digest_field="digest")
    (tmp_path / "result.json").write_text(json.dumps(result))
    _, frames, images = completion.load_generated_views(tmp_path, ["novel-a"])
    assert np.allclose(frames[0]["position"], [.2, 0, 0])
    assert frames[0]["frame_id"] == "novel-a" and images[0]["novel_view"] is True
    assert np.all(frames[0]["confidence"] == 1)
    depth.write_bytes(b"different-exr")
    with pytest.raises(ValueError, match="output_changed"):
        completion.load_generated_views(tmp_path, ["novel-a"])


def test_completion_routes_all_planned_views_through_preservation_before_review(tmp_path, monkeypatch):
    from blueprint_pipeline import website_image_completion, atlas_scene_chisel
    frames = [frame(), frame("b", .2)]
    usage, returned = [], []
    for index, f in enumerate(frames):
        image, depth = tmp_path / f"{index}-image.bin", tmp_path / f"{index}-depth.bin"
        Image.fromarray(f["rgb"]).save(image, format="PNG")
        depth.write_bytes(b"\x76\x2f\x31\x01retained-exr")
        usage.append({"frame_id": f["frame_id"], "outputs": [
            {"role": role, "path": p.name, "sha256": _sha256_file(p)} for role, p in (("image", image), ("depth", depth))]})
        returned.append({"imageAsset": {"assetId": "expired-image"}, "depth": {"depthAsset": {"assetId": "expired-depth"}},
                         "camera": {"intrinsics": f["intrinsics"], "extrinsics": f["extrinsics"]}})
    sources = []
    for i, f in enumerate(frames):
        source = tmp_path / f'{i}-source.png'
        Image.fromarray(f['rgb']).save(source)
        sources.append({'frame_id': f['frame_id'], 'image_path': str(source), 'image_digest': _sha256_file(source)})
    inputs = {'schema_version': 'website_atlas_pose_inputs.v1', 'status': 'pending',
              'endpoint': '/api/v2/tasks:images2PosedRGBD', 'heldout_pixels_included': False,
              'claim_ceiling': 'development_only', 'metric_measurement_proven': False, 'frames': sources}
    inputs['digest'] = canonical_digest(inputs, digest_field='digest')
    (tmp_path / 'atlas_pose_inputs.json').write_text(json.dumps(inputs))
    baseline = {'pose_result_digest': 'earlier-cropped-pose'}
    baseline['digest'] = canonical_digest(baseline, digest_field='digest')
    (tmp_path / 'background_splat').mkdir()
    (tmp_path / 'background_splat/atlas_depth_splat_manifest.json').write_text(json.dumps(baseline))
    pose = {"digest": "pose", "binding": {"source_video_digest": "source"},
            "operation": {"id": "pose", "done": True, "response": {"frames": returned}},
            "operation_receipt": {"input_usage": {"frames": usage}}}
    proposal = {"status": "proposed", "selected_camera_ids": ["a-wide", "b-lower"],
                "gap_assessment": ["Complete floor and empty bay"], "remaining_unknowns": ["Generated rear wall"],
                'scene_caption': 'The photographed room with bare continuous floor in cleared space.',
                'view_gap_reasons': [{'camera_id': 'a-wide', 'reason': 'Retain the observed task-region anchor'},
                                     {'camera_id': 'b-lower', 'reason': 'Fill its nearby floor gap'}]}
    calls = []
    def generate(**kw):
        calls.append(kw)
        generated = [frame('completion-' + r['camera_id']) for r in kw['selected']]
        images = [{"frame_id": f['frame_id'], "source_frame_id": r['source_frame_id'],
                   "image_path": str(tmp_path / "0-image.bin"),
                   "image_digest": _sha256_file(tmp_path / "0-image.bin"), "novel_view": True}
                  for f, r in zip(generated, kw['selected'], strict=True)]
        return (generated, images), {"digest": "preserved-generation", "generated_roots": [str(tmp_path / 'native')],
            "preservation_paths": [], "known_pixels_changed": 0}
    monkeypatch.setattr(atlas_scene_chisel, 'generate_chisel_completion', generate)
    reviews = []
    monkeypatch.setattr(website_image_completion, "verify_completed_background",
                        lambda **kw: reviews.append(kw) or {"status": "passed"})
    args = dict(pose_root=tmp_path, pose=pose, frames=frames,
        review={"automation": {"objects": [], "removal_instruction": "The dishwasher bay is empty"}},
        root=tmp_path / "complete", context={"description": "test"},
        configuration={"remaining_cost_usd": 5, "policy": {"rgbd_view_cost_usd": .05,
            "rate_card_reference": "rates", "reviewed_on": "2026-10-22", "reviewed_by": "test"}},
        task_runner=lambda **kw: pytest.fail('Unconstrained generation bypassed preservation'),
        planner_runner=lambda **_: {"proposal": proposal},
        release_blockers=lambda: ("commit", []), max_polls=1)
    output, report = completion.complete_scene(**args)
    feedback = json.loads((tmp_path / 'complete/plan-inputs.json').read_text())
    assert feedback['baseline_geometry_comparable'] is False and 'baseline_geometry_gaps' not in feedback
    assert len(output) == len(proposal["selected_camera_ids"]) and report["generated_geometry_ready"] is True
    assert len(calls) == 1 and [row['camera_id'] for row in calls[0]['selected']] == proposal['selected_camera_ids']
    assert report['generation']['known_pixels_changed'] == 0
    assert report['generation_result_digest'] == 'preserved-generation'
    assert report['generated_roots'] == [str(tmp_path / 'native')]
    assert reviews[0]["frames"][1]["original_image_path"] == str(tmp_path / "1-image.bin")
    assert report['completion_extent'] == 'bounded_task_relevant_gaps'
    assert report['generation_extent']['additional_space_beyond_observed_bounds'] is False
    assert calls[0]['scene_caption'] == proposal['scene_caption']
    proposal['view_gap_reasons'].pop()
    with pytest.raises(ValueError, match="task_gap_plan_missing"):
        completion.complete_scene(**args)
    assert len(calls) == 1
    proposal["selected_camera_ids"] = ["invented-camera"]
    with pytest.raises(ValueError, match="camera_selection_invalid"):
        completion.complete_scene(**args)
    assert len(calls) == 1


@pytest.mark.parametrize("drop_generated", [False, True])
def test_automation_consumes_generated_depth_and_refuses_a_stale_baseline(tmp_path, monkeypatch, drop_generated):
    from blueprint_pipeline import atlas_scene_automation as worker, atlas_background_automation
    from blueprint_pipeline import atlas_scene_measurements, atlas_depth_splat, atlas_result_bundle
    source, novel = frame(), frame("novel", .1)
    pose = {"digest": "pose", "binding": {"source_video_digest": "source"}}
    monkeypatch.setattr(atlas_scene_measurements, "load_views", lambda *a: (pose, [source]))
    monkeypatch.setattr(atlas_scene_measurements, "estimate_sizes", lambda **_: {})
    monkeypatch.setattr(atlas_background_automation, "run_background", lambda **_: {
        "status": "fallback_prepared", "frames": [], "review": {"status": "passed"}})
    monkeypatch.setattr(worker, "_fallback_pose", lambda *a: ((pose, [source], tmp_path), {"status": "completed"}))
    monkeypatch.setattr(completion, "complete_scene", lambda **_: ([novel], {
        "status": "passed", "generation_result_digest": "generated", "review": {"status": "passed"},
        "generated_root": str(tmp_path / "generated")}))
    def build(**kw):
        assert [f["frame_id"] for f in kw["frames"]] == ["a", "novel"]
        assert kw["generated_result_digests"][-1] == "generated"
        assert kw["output_root"].name == "background_completed_splat-" + completion.COMPLETION_REVISION
        return {"digest": "splat", "generated_result_digests": kw["generated_result_digests"],
                "frames": [{"frame_id": f["frame_id"]} for f in (kw["frames"][:1] if drop_generated else kw["frames"])]}
    monkeypatch.setattr(atlas_depth_splat, "write_depth_splat", build)
    exports = []
    monkeypatch.setattr(atlas_result_bundle, "export_pose_bundle", lambda **kw: exports.append(kw) or {"zip_path": "scene.zip"})
    result = worker.run_automation(pose_root=tmp_path, inputs={},
        review={"automation": {"objects": [], "scale_references": [], "removal_instruction": "Empty bay", "mask_margin_fraction": .01}},
        output_root=tmp_path / "out", task_context={}, configuration={"policy": {"rgbd_view_cost_usd": .05}},
        task_runner=None, pose_runner=None, release_blockers=lambda: ("commit", []))
    if drop_generated:
        assert result["status"] == "needs_input" and "did_not_consume_generated_depth" in result["background"]["reason"]
        assert not exports
    else:
        assert result["status"] == "completed" and result["background"]["reconstruction_frame_count"] == 2
        assert exports[0]["generated_roots"] == [tmp_path / "generated"]
