"""Scale repairs reuse retained views, check endpoints, and leave prior evidence intact."""
from copy import deepcopy
import json

import pytest

from blueprint_pipeline import atlas_scale_review as repair
from blueprint_pipeline import atlas_scene_measurements as sizes
from tests.test_atlas_scene_automation import view, plan, track


def proposal(automation):
    refs = deepcopy(automation["scale_references"])
    for row in refs:
        row["segmentation_prompt"] = "visible reference object"
    return {"status": "proposed", "scale_references": refs, "summary": "Corrected visible object edges"}


def test_review_remeasures_new_endpoints_without_overwriting_retained_scale(tmp_path):
    frames = [view(tmp_path)]
    original = plan()
    original["scale_references"][0]["endpoints_normalized"][0][0] = .05
    old_path = tmp_path / "old.json"
    old = sizes.estimate_sizes(frames=frames, automation=original, pose_digest="pose", output=old_path)
    before = old_path.read_bytes()
    assert repair.needs_scale_review(old)
    masks = [{"reference_index": 0, "frame_id": "a", "instances": track(frames)["track"]["observations"]}]
    fixed, path = repair.measure_review(frames=frames, pose_digest="pose", automation=original,
        proposal=proposal(plan()), reference_sources=[], reference_masks=masks, output_root=tmp_path / "revised")
    assert fixed["meters_per_unit_estimate"] == pytest.approx(1.)
    assert old["meters_per_unit_estimate"] < fixed["meters_per_unit_estimate"]
    assert fixed["binding"]["automation"]["objects"] == original["objects"]
    assert not repair.needs_scale_review(fixed)
    assert old_path.read_bytes() == before and path != old_path
    assert repair.measure_review(frames=frames, pose_digest="pose", automation=original,
        proposal=proposal(plan()), reference_sources=[], reference_masks=masks,
        output_root=tmp_path / "revised") == (fixed, path)


def test_repair_cannot_accept_foreground_endpoint_or_missing_object_masks(tmp_path):
    frames = [view(tmp_path)]
    p = proposal(plan())
    p["scale_references"][0]["endpoints_normalized"][0][0] = .05
    masks = [{"reference_index": 0, "frame_id": "a", "instances": track(frames)["track"]["observations"]}]
    options = dict(frames=frames, pose_digest="pose", automation=plan(), proposal=p,
                   reference_sources=[], output_root=tmp_path / "revised")
    result, _ = repair.measure_review(**options, reference_masks=masks)
    assert result["status"] == "relative_sizes_only"
    assert result["reference_gaps"][0]["reason"] == "atlas_reference_endpoint_off_object"
    with pytest.raises(ValueError, match="object_masks_required"):
        repair.measure_review(**options, reference_masks=None)


@pytest.mark.parametrize("change", ["unknown_frame", "unnamed", "abstained"])
def test_invalid_reference_proposals_cannot_be_measured(change):
    p = proposal(plan())
    if change == "unknown_frame":
        p["scale_references"][0]["frame_id"] = "not_seen"
    elif change == "unnamed":
        p["scale_references"][0].pop("segmentation_prompt")
    else:
        p["status"] = "abstained"
    with pytest.raises(ValueError, match="atlas_scale_review"):
        repair.revised_plan(automation=plan(), proposal=p, frame_ids={"a"}, reference_sources=[])


def test_automatic_repair_reviews_full_views_and_keeps_one_revision(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_context_fov as fov, atlas_reference_masks as masking
    frames = [view(tmp_path)]
    source = {"frame_id": "a", "image_path": frames[0]["image_path"], "image_digest": frames[0]["image_digest"]}
    monkeypatch.setattr(fov, "full_view_sources", lambda *_: [source])
    monkeypatch.setattr(sizes, "load_views", lambda *_: ({}, frames))
    checked = []
    def segment(**kw):
        assert kw["references"][0]["segmentation_prompt"] == "visible reference object"
        return [{"reference_index": 0, "frame_id": "a", "instances": track(frames)["track"]["observations"]}]
    monkeypatch.setattr(masking, "segment_scale_references", segment)
    def planner(**kw):
        survey = json.loads(kw["survey_path"].read_text())
        assert survey["frames"][0]["source_image_path"] == source["image_path"]
        assert survey["display_rotation_degrees"] == 0
        assert not kw["scaling"]["named_object_masks_checked"]
        assert kw["scaling"]["previous_measurement_digest"] == "prior"
        return {"proposal": proposal(plan()), "reference_sources": []}
    result, path = repair.repair_source_scale(pose_root=tmp_path, pose={"digest": "pose"},
        inputs={"binding": {"source_video_digest": "video"}}, automation=plan(),
        previous={"digest": "prior"}, task_context={}, configuration={}, check=checked.append, planner_runner=planner)
    assert result["status"] == "estimated" and checked == ["scale_review"]
    assert path.is_relative_to(tmp_path / "scale_review/measurements")
