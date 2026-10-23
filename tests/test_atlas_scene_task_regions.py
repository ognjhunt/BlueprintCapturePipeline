"""Task scope selects regions, not entire images or fixed site categories."""
import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import atlas_scene_recovery as recovery
from blueprint_pipeline.atlas_scene_task_regions import (
    TaskSurfaceProposal, task_region_masks, plan_task_regions,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_atlas_scene_recovery import fixture


def make_plan(root, reviewed, regions):
    frame = reviewed[0]["frame"]
    plan = {"proposal": {"status": "proposed", "task_extent_summary": "Local work area",
        "views": [{"frame_id": frame["frame_id"], "regions": regions, "reason": "Task and context"}],
        "remaining_required_surfaces": [], "excluded_scene_regions": ["Distant unrelated area"]},
        "native_bindings": {frame["frame_id"]: canonical_digest(recovery._frame_binding(frame))},
        "plan_path": str(root / "task-regions.json")}
    plan["digest"] = canonical_digest(plan, digest_field="digest")
    (root / "task-regions.json").write_text(json.dumps(plan))
    return plan


def region(label="work area", decision="include", purpose="work_surface", polygon=None):
    return {"label": label, "decision": decision, "purpose": purpose,
            "polygon_normalized": polygon or [[0, 0], [1, 0], [1, 1], [0, 1]],
            "reason": "Supports the stated task" if decision == "include" else "Outside the task area",
            "priority": "task", "segmentation_prompt": label}


@pytest.mark.parametrize("label,purpose", [("packing bench", "work_surface"), ("laboratory rack", "support")])
def test_one_view_can_include_task_surfaces_and_exclude_unrelated_area(tmp_path, label, purpose):
    _, _, reviewed = fixture()
    plan = make_plan(tmp_path, reviewed, [region(label, purpose=purpose),
        region("unrelated adjoining area", "exclude", "unrelated",
               [[0, 0], [.5, 0], [.5, 1], [0, 1]])])
    mask = task_region_masks(plan, reviewed)[reviewed[0]["frame"]["frame_id"]]
    assert not mask[:, :33].any() and mask[:, 33:].all()
    # Native identity includes depth and cameras, not just a reused frame ID.
    reviewed[0]["frame"]["depth"][0, 0] += 1
    with pytest.raises(ValueError, match="native_inputs_changed"):
        task_region_masks(plan, reviewed)


def test_schema_is_accepted_by_the_actual_strict_agents_sdk():
    from agents import AgentOutputSchema
    assert AgentOutputSchema(TaskSurfaceProposal).is_strict_json_schema()


@pytest.mark.parametrize("mutation", ["missing_view", "duplicate_view", "invalid_polygon", "unrelated_include"])
def test_semantic_selection_cannot_silently_expand_or_skip_an_unassessed_view(tmp_path, mutation):
    _, _, reviewed = fixture()
    plan = make_plan(tmp_path, reviewed, [region()])
    if mutation == "missing_view":
        plan["proposal"]["views"][0]["frame_id"] = "not-a-native-view"
    elif mutation == "duplicate_view":
        plan["proposal"]["views"] *= 2
    elif mutation == "invalid_polygon":
        plan["proposal"]["views"][0]["regions"][0]["polygon_normalized"][0] = [-.1, 0]
    else:
        plan["proposal"]["views"][0]["regions"][0]["purpose"] = "unrelated"
    plan["digest"] = canonical_digest(plan, digest_field="digest")
    with pytest.raises(ValueError, match="atlas_task_region_"):
        task_region_masks(plan, reviewed)


def test_recovery_honors_partial_task_scope_and_keeps_other_geometry_unchanged(tmp_path):
    source, prior, reviewed = fixture()
    plan = make_plan(tmp_path, reviewed, [region(),
        region("unrelated side", "exclude", "unrelated", [[.5, 0], [1, 0], [1, 1], [.5, 1]])])
    frames, report = recovery.recover_reviewed_geometry(
        frames=[source], geometry_frames=[source, prior], reviewed_views=reviewed,
        root=tmp_path / "recovery", region_plan=plan)
    assert frames and frames[0]["confidence"][14:34, 26:31].all()
    assert not frames[0]["confidence"][:, 32:].any()
    assert report["task_region_plan"]["digest"] == plan["digest"]
    assert report["whole_scene_complete"] is False and report["source_geometry_modified"] is False


def test_large_task_surface_is_not_discarded_by_pre_generation_context_ratio(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_preservation as preservation
    from PIL import Image
    source, prior, reviewed = fixture()
    # A new camera may have only a small supported RGB patch. This does not
    # invalidate an already reviewed missing surface with consistent depth.
    real_prepare = preservation.prepare_completion_views
    def small_context(**kwargs):
        rows = real_prepare(**kwargs)
        for row in rows:
            keep = np.zeros((48, 64), np.uint8)
            keep[:4, :4] = 255
            Image.fromarray(keep).save(row["keep_mask_path"])
            row["keep_mask_digest"] = preservation._sha256_file(Path(row["keep_mask_path"]))
        return rows
    monkeypatch.setattr(preservation, "prepare_completion_views", small_context)
    plan = make_plan(tmp_path, reviewed, [region()])
    frames, report = recovery.recover_reviewed_geometry(
        frames=[source], geometry_frames=[source, prior], reviewed_views=reviewed,
        root=tmp_path / "recovery", region_plan=plan)
    assert frames and report["views"][0]["eligible_gap_pixels"] > 16
    assert report["exportable_samples"] > 300


def test_production_plan_receives_all_exact_views_and_the_current_task(tmp_path):
    source, _, reviewed = fixture()
    calls = []
    def planner(**kwargs):
        calls.append(kwargs)
        survey = json.loads(kwargs["survey_path"].read_text())
        assert [r["frame_id"] for r in survey["frames"]] == [source["frame_id"], reviewed[0]["frame"]["frame_id"]]
        assert kwargs["task_context"]["description"] == "Move packages from conveyor to packing bench"
        assert kwargs["recovery"]["native_frame_ids"] == [reviewed[0]["frame"]["frame_id"]]
        if kwargs["recovery"]["scope"] == "task_surface_masks":
            proposal = {"status": "passed", "views": [{"frame_id": reviewed[0]["frame"]["frame_id"],
                "task_mask_ids": [r["mask_id"] for r in kwargs["recovery"]["candidates"]],
                "context_mask_ids": [], "corrections": [], "reason": "Whole bench included"}],
                "remaining_required_surfaces": []}
        else:
            proposal = make_plan(tmp_path, reviewed, [region("packing bench")])["proposal"]
        return {"proposal": proposal, "cost_usd": 0}
    def segmenter(**kwargs):
        assert len(kwargs["frame_registry"]) == 1
        frame = kwargs["frame_registry"][0]
        return {"binding_digest": "sam-binding", "profile": {"model": "sam-3.1"}, "responses": [],
            "tracks": [{"observations": [{**frame, "runs": [{"start": 0, "length": 48 * 64}]}]}]}
    plan = plan_task_regions(frames=[source], reviewed_views=reviewed,
        pose={"digest": "pose", "binding": {"source_video_digest": "video"}},
        context={"description": "Move packages from conveyor to packing bench", "context_digest": "context"},
        configuration={"policy": {}}, root=tmp_path / "plans", planner_runner=planner, segmenter=segmenter)
    assert len(calls) == 2 and task_region_masks(plan, reviewed)
    assert json.loads(Path(plan["plan_path"]).read_text()) == plan


@pytest.mark.parametrize("depth", [1., 10.])
def test_task_relevance_never_waives_free_space_or_extent_checks(tmp_path, depth):
    source, prior, reviewed = fixture()
    prior["confidence"][:] = 1
    reviewed[0]["frame"]["depth"][:] = depth
    plan = make_plan(tmp_path, reviewed, [region()])
    frames, report = recovery.recover_reviewed_geometry(
        frames=[source], geometry_frames=[source, prior], reviewed_views=reviewed,
        root=tmp_path / "recovery", region_plan=plan)
    assert not frames and report["added_geometry_pixels"] == 0

