"""Previously selected image masks must not hide other recoverable 3D gaps."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline import atlas_scene_recovery as recovery


def fixture():
    camera = {"intrinsics": {"width": 64, "height": 48, "fx": 60., "fy": 60., "cx": 32., "cy": 24.},
              "extrinsics": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"}}
    source = posed_frame(frame_id="source", rgb=np.full((48, 64, 3), 50, np.uint8),
                         depth=np.full((48, 64), 2., np.float32),
                         confidence=np.ones((48, 64), np.float32), camera=camera)
    native = deepcopy(source)
    native["frame_id"] = "recovered-native-0"
    native["rgb"][:] = 240
    source["confidence"][12:36, 20:44] = 0
    prior = deepcopy(native)
    prior["frame_id"] = "prior-small-fill"
    prior["rgb"][:] = 50
    prior["confidence"][:] = 0
    prior["confidence"][12:36, 20:24] = 1
    reviewed = [{"frame": native, "native_result_digest": "sha256:native",
                 "native_frame_index": 0, "review_digest": "sha256:review"}]
    return source, prior, reviewed


def test_recovery_fills_remaining_photo_hole_without_overwriting_the_scene_or_native_output(tmp_path):
    source, prior, reviewed = fixture()
    before = deepcopy([source, prior, reviewed[0]["frame"]])
    frames, report = recovery.recover_reviewed_geometry(
        frames=[source], geometry_frames=[source, prior], reviewed_views=reviewed, root=tmp_path)
    assert len(frames) == 1 and report["exportable_samples"] > 300
    actual = frames[0]
    assert actual["confidence"][14:34, 26:42].all()
    assert not actual["confidence"][:, :24].any(), "Existing photo and previous completion must stay fixed"
    assert np.all(actual["rgb"] == 50), "Generated replacement colours cannot overwrite photographed RGB"
    assert report["new_provider_calls"] == 0 and report["whole_scene_complete"] is False
    for current, original in zip([source, prior, reviewed[0]["frame"]], before, strict=True):
        for key in ("rgb", "depth", "confidence"):
            assert np.array_equal(current[key], original[key])


@pytest.mark.parametrize("depth", [0., 10.])
def test_recovery_preserves_scene_extent_instead_of_adding_an_entire_generated_room(tmp_path, depth):
    source, prior, reviewed = fixture()
    reviewed[0]["frame"]["depth"][:] = depth
    frames, report = recovery.recover_reviewed_geometry(
        frames=[source], geometry_frames=[source, prior], reviewed_views=reviewed, root=tmp_path)
    assert not frames and report["added_geometry_pixels"] == 0


def test_recovery_reuses_verified_masks_and_refuses_changed_output(tmp_path, monkeypatch):
    source, prior, reviewed = fixture()
    args = dict(frames=[source], geometry_frames=[source, prior], reviewed_views=reviewed, root=tmp_path)
    frames, report = recovery.recover_reviewed_geometry(**args)
    from blueprint_pipeline import atlas_scene_chisel
    monkeypatch.setattr(atlas_scene_chisel, "project_source_depth",
                        lambda *_: pytest.fail("Cached recovery was recomputed"))
    cached, same = recovery.recover_reviewed_geometry(**args)
    assert same == report and np.array_equal(cached[0]["confidence"], frames[0]["confidence"])
    Image.fromarray(np.zeros((48, 64), np.uint8)).save(report["views"][0]["mask_path"])
    with pytest.raises(ValueError, match="output_changed"):
        recovery.recover_reviewed_geometry(**args)


def test_unreviewed_outputs_are_never_recovered(tmp_path):
    with pytest.raises(ValueError, match="review_not_bound"):
        recovery.reviewed_completion_views([{"status": "passed", "review": {"status": "needs_input"}}])
    source, prior, reviewed = fixture()
    reviewed[0].pop("review_digest")
    with pytest.raises(ValueError, match="review_not_bound"):
        recovery.recover_reviewed_geometry(frames=[source], geometry_frames=[source, prior],
                                           reviewed_views=reviewed, root=tmp_path)


def test_bound_review_cannot_admit_a_different_native_result(tmp_path):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    (tmp_path / "result.json").write_text(json.dumps({"digest": "another-result"}))
    report = {"status": "passed", "review": {"status": "passed"}, "generated_roots": [str(tmp_path)],
              "generation": {"native_result_digests": ["reviewed-result"]}}
    report["digest"] = canonical_digest(report, digest_field="digest")
    with pytest.raises(ValueError, match="not_reviewed"):
        recovery.reviewed_completion_views([report])


def test_no_gap_completion_has_no_native_outputs_to_recover():
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    report = {"status": "passed", "review": {"status": "not_required"}, "generated_roots": []}
    report["digest"] = canonical_digest(report, digest_field="digest")
    assert recovery.reviewed_completion_views([report]) == []
