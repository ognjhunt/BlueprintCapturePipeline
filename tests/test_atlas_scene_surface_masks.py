"""Whole SAM surfaces, optional accurate context, correction and cached reuse."""
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.atlas_scene_surface_masks import SurfaceMaskReview
from blueprint_pipeline.atlas_scene_task_regions import plan_task_regions, task_region_masks
from tests.test_atlas_scene_recovery import fixture
from tests.test_atlas_scene_task_regions import make_plan, region


def observation(frame, mask):
    edges = np.diff(np.pad(mask.reshape(-1).astype(np.int8), (1, 1)))
    return {**frame, "runs": [{"start": int(a), "length": int(b - a)}
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)]}


def test_full_surface_and_accurate_context_survive_a_tight_agent_polygon_and_cache(tmp_path):
    source, _, reviewed = fixture()
    fid = reviewed[0]["frame"]["frame_id"]
    task = region("assembly bench", polygon=[[.4, .4], [.6, .4], [.6, .6], [.4, .6]])
    context = {**region("wall", purpose="visual_context"), "priority": "context"}
    proposal = make_plan(tmp_path, reviewed, [task, context])["proposal"]
    calls = []
    def sam(**kw):
        calls.append(kw)
        f = kw["frame_registry"][0]
        assert f["width"] == 64 and f["height"] == 48
        from PIL import Image
        assert Image.open(kw["frame_artifacts"][0]["path"]).size == (64, 48)
        mask = np.zeros((48, 64), bool)
        if kw["prompts"][0]["text"] == "assembly bench":
            mask[12:, 8:56] = True  # Rear/side extend far outside the rough seed.
        else:
            mask[:10] = True
        return {"binding_digest": "binding", "profile": {"model": "sam-3.1"}, "responses": [],
                "tracks": [{"observations": [observation(f, mask)]}]}
    def planner(**kw):
        feedback = kw["recovery"]
        if feedback["scope"] != "task_surface_masks":
            return {"proposal": proposal}
        rows = feedback["candidates"]
        return {"proposal": {"status": "passed", "views": [{"frame_id": fid,
            "task_mask_ids": [r["mask_id"] for r in rows if r["concept"] == "assembly bench"],
            "context_mask_ids": [r["mask_id"] for r in rows if r["concept"] == "wall"],
            "corrections": [], "reason": "Entire bench and accurate wall context"}],
            "remaining_required_surfaces": []}}
    args = dict(frames=[source], reviewed_views=reviewed,
        pose={"digest": "pose", "binding": {"source_video_digest": "video"}},
        context={"description": "Assemble parts at a bench", "context_digest": "context"},
        configuration={"policy": {}}, root=tmp_path / "plans")
    plan = plan_task_regions(**args, planner_runner=planner, segmenter=sam)
    mask = task_region_masks(plan, reviewed)[fid]
    assert mask[12:, 8:56].all(), "The coarse polygon must not cut off the complete SAM surface"
    assert mask[:10].all(), "Useful background is not discarded because it is optional"
    assert not mask[10:12].any()
    assert len(calls) == 2
    cached = plan_task_regions(**args, planner_runner=lambda **_: pytest.fail("Repeated paid review"),
                              segmenter=lambda **_: pytest.fail("Repeated SAM call"))
    assert cached == plan
    Path(plan["segmentation"]["candidates"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="surface_mask_changed"):
        plan_task_regions(**args, planner_runner=planner, segmenter=sam)


def test_missing_visible_part_is_corrected_with_a_new_concept_before_admission(tmp_path):
    from blueprint_pipeline.atlas_scene_surface_masks import segment_and_review_surfaces
    _, _, reviewed = fixture()
    fid = reviewed[0]["frame"]["frame_id"]
    prompts = []
    def sam(**kw):
        prompt = kw["prompts"][0]["text"]
        prompts.append(prompt)
        frame = kw["frame_registry"][0]
        mask = np.zeros((48, 64), bool)
        mask[:24 if prompt == "worktop" else 48] = True
        return {"binding_digest": prompt, "profile": {"model": "sam-3.1"}, "responses": [],
                "tracks": [{"observations": [observation(frame, mask)]}]}
    def reviewer(**kw):
        corrected = len(prompts) == 2
        return {"proposal": {"status": "passed" if corrected else "revise", "views": [{
            "frame_id": fid, "task_mask_ids": [r["mask_id"] for r in kw["recovery"]["candidates"]],
            "context_mask_ids": [], "reason": "A visible support was missed",
            "corrections": [] if corrected else [region("support cabinet")]}],
            "remaining_required_surfaces": []}}
    proposal = make_plan(tmp_path, reviewed, [region("worktop")])["proposal"]
    result = segment_and_review_surfaces(proposal=proposal, reviewed_views=reviewed,
        survey_path=tmp_path / "survey.json", context={}, configuration={"policy": {}},
        root=tmp_path / "masks", source_digest="video", planner_runner=reviewer, segmenter=sam)
    assert prompts == ["worktop", "support cabinet"]
    assert result["review_rounds"] == 2 and result["review"]["proposal"]["status"] == "passed"


def test_actual_sdk_accepts_surface_review_schema():
    from agents import AgentOutputSchema
    assert AgentOutputSchema(SurfaceMaskReview).is_strict_json_schema()
