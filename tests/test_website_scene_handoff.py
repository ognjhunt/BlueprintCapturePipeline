"""The capture's Atlas job is the only website scene source.

No Marble world is bought or read (owner decision 2026-10-22). Preparation waits
while the Atlas job runs and reports why it cannot proceed otherwise.
"""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline.common import write_json
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def inputs(tmp_path):
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir(exist_ok=True)
    removal = pipeline / "removal.json"
    write_json(removal, {"entries": []})
    return dict(
        descriptor={"capture_id": "cap-1", "scene_id": "scene-1", "metadata": {
            "site_task_context": {"capture_id": "cap-1", "scene_id": "scene-1"}}},
        clean_plate={"privacy_verified": True, "status": "objects_removed", "source_geometry": None,
                     "task_masks": {}, "removal_manifest_path": str(removal)},
        capture_root=tmp_path, now=1800000000,
    )


ATLAS = "blueprint_pipeline.website_atlas_scene_source.atlas_scene_source"


def atlas_ready(tmp_path):
    collider = tmp_path / "pipeline/atlas/background.glb"
    collider.parent.mkdir(parents=True, exist_ok=True)
    collider.write_bytes(b"atlas-collider")
    base = {"splat_path": str(collider), "splat_digest": _sha256_file(collider), "collision_mesh_path": str(collider),
            "collision_mesh_digest": _sha256_file(collider), "up_axis": "Y", "meters_per_unit": 0.8,
            "provider": "world_labs_atlas", "scale_authority": "atlas_reference_estimate",
            "ground_plane_offset_m": None, "anchor": {"kind": "source_world_alignment"}, "operation_id": "atlas-op"}
    return {"status": "ready", "base_scene": base, "alignment": "anchored",
            "source_geometry": {"digest": "atlas-geometry", "geometry_backend": "world_labs_atlas"},
            "source_pose_digest": "sha256:" + "1" * 64, "background_pose_digest": "sha256:" + "2" * 64}


@pytest.fixture
def ready_atlas(tmp_path, monkeypatch):
    ready = atlas_ready(tmp_path)
    monkeypatch.setattr(ATLAS, lambda **_: ready)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry",
                        lambda **kw: {**kw["task_masks"], "source_geometry_digest": "atlas-geometry"})
    return ready


def test_a_ready_atlas_scene_drives_preparation_but_never_invents_execution(tmp_path, monkeypatch, ready_atlas):
    kwargs = inputs(tmp_path)
    calls = []
    def compile(**value):
        calls.append(value)
        assert value["spend"]["max_total_spend_usd"] == 0
        assert value["base_scene"] == ready_atlas["base_scene"]
        return {"status": "needs_input", "blockers": ["website_scene_execution_authority_required"],
                "digest": "sha256:" + "a" * 64, "thumbnail": {"path": "proposed-thumbnail.png"}}
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", compile)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "needs_input" and len(calls) == 1
    assert result["simulator_ready"] is False and result["visual_reconstruction_ready"] is True
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/handoff.json").read_text()) == result


@pytest.mark.parametrize("deferred_masks", [False, True])
@pytest.mark.parametrize("atlas,status", [
    ({"status": "needs_input", "reason": "website_atlas_job_required"}, "needs_input"),
    ({"status": "needs_input", "reason": "website_atlas_background_unavailable"}, "needs_input"),
    ({"status": "pending", "reason": "website_atlas_job_processing", "stage": "pose"}, "awaiting_reconstruction"),
])
def test_preparation_waits_for_the_capture_atlas_job_with_no_other_scene(
        tmp_path, monkeypatch, deferred_masks, atlas, status):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    kwargs = inputs(tmp_path)
    kwargs["clean_plate"]["task_masks"] = {
        "deferred_target_ids": ["support"] if deferred_masks else []}
    def unexpected(**_):
        pytest.fail("a pending Atlas job must not invoke masks or the compiler")
    monkeypatch.setattr(ATLAS, lambda **_: atlas)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.complete_retained_static_task_masks", unexpected)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", unexpected)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == status and result["blockers"] == [atlas["reason"]]
    assert result["visual_reconstruction_ready"] is False and "base_scene_path" not in result
    assert result["provider_mutation_performed"] is False
    assert result["simulator_ready"] is False and result["geometry_backend"] is None
    assert result["atlas_scene_source"]["status"] == atlas["status"]
    assert result["native_geometry_requirements"] == {
        "source_aligned_depth_and_camera_poses": "required_for_mask_lifting_and_scene_registration",
        "measured_scale": "unknown", "atlas_required": True}
    assert not (tmp_path / "pipeline/website_scene_preparation/base_scene.json").exists()
    assert result["digest"] == canonical_digest(result, digest_field="digest")
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/handoff.json").read_text()) == result


@pytest.mark.parametrize("change,reason", [
    ("privacy", "website_scene_preparation_pending"),
    ("identity", "website_scene_task_identity_mismatch"),
])
def test_held_or_changed_inputs_do_not_enter_scene_construction(tmp_path, monkeypatch, change, reason):
    kwargs = inputs(tmp_path)
    if change == "privacy":
        kwargs["clean_plate"]["privacy_verified"] = False
    else:
        kwargs["descriptor"]["metadata"]["site_task_context"]["capture_id"] = "other"
    monkeypatch.setattr(ATLAS, lambda **_: pytest.fail("held inputs must not read the Atlas job"))
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", lambda **_: pytest.fail("must not compile"))
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"] == [reason]
    assert result["simulator_ready"] is False


def test_a_ready_atlas_job_is_the_scene_and_its_pose_the_geometry_without_marble(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    masks = {"targets": [{"target_id": "object"}], "digest": "sha256:" + "b" * 64}
    kwargs["clean_plate"].update(task_masks=masks)
    ready = atlas_ready(tmp_path)
    monkeypatch.setattr(ATLAS, lambda **kw: ready)
    bound = {**masks, "source_geometry_digest": "atlas-geometry"}
    seen = {}
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry",
                        lambda **kw: seen.update(bind=kw) or bound)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **value: seen.update(compile=value) or (_ for _ in ()).throw(ValueError("stop")))
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"] == ["stop"] and result["geometry_backend"] == "world_labs_atlas"
    assert seen["bind"] == {"task_masks": masks, "source_geometry": ready["source_geometry"]}
    assert seen["compile"]["base_scene"] == ready["base_scene"]
    assert seen["compile"]["source_geometry"] == ready["source_geometry"] and seen["compile"]["task_masks"] == bound
    assert result["atlas_scene_source"]["alignment"] == "anchored"
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/base_scene.json").read_text()) == ready["base_scene"]
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/task_masks.geometry.json").read_text()) == bound


@pytest.mark.parametrize("support_ready", [False, True])
def test_full_masks_are_required_before_they_lift_onto_atlas_geometry(tmp_path, monkeypatch, support_ready):
    kwargs = inputs(tmp_path)
    plan = tmp_path / "pipeline/plan.json"
    write_json(plan, {"targets": [{"target_id": "support"}]})
    kwargs["clean_plate"].update(source_frames={"digest": "source"},
        removal_plan_path=str(plan), task_masks={"deferred_target_ids": ["support"]})
    monkeypatch.setattr(ATLAS, lambda **kw: atlas_ready(tmp_path))
    calls = []
    full_masks = {"targets": [{"target_id": "object"}, {"target_id": "support"}]}
    def resolve(**kw):
        calls.append("masks")
        assert kw["source_geometry"] == {"digest": "source"}
        if not support_ready:
            raise ValueError("support_unresolved")
        return full_masks
    def bind(**kw):
        calls.append("bind")
        assert kw["task_masks"] == full_masks
        raise ValueError("binding_pending")
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.complete_retained_static_task_masks", resolve)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", bind)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["visual_reconstruction_ready"] is True and result["simulator_ready"] is False
    assert result["blockers"] == ["binding_pending" if support_ready else "support_unresolved"]
    assert calls == (["masks", "bind"] if support_ready else ["masks"])


def test_articulated_targets_gain_coverage_from_the_retained_video_before_compile(tmp_path, monkeypatch,
                                                                                   ready_atlas):
    kwargs = inputs(tmp_path)
    video = tmp_path / "walkthrough.mov"
    video.write_bytes(b"source")
    masks = {"targets": [{"target_id": "dishwasher-1"}], "digest": "sha256:" + "b" * 64}
    kwargs["clean_plate"].update(task_masks=masks, input_video_path=str(video))
    seen, attached = {}, {**masks, "targets": [{"target_id": "dishwasher-1", "authoring_coverage": {"status": "complete"}}]}

    def attach(**value):
        seen["attach"] = value
        return attached
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage", attach)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **value: seen.update(compile=value) or (_ for _ in ()).throw(ValueError("stop")))
    handoff.prepare_website_scene_handoff(**kwargs)
    bound = {**masks, "source_geometry_digest": "atlas-geometry"}
    assert seen["attach"]["task_masks"] == bound and seen["attach"]["source_video"] == video
    assert seen["attach"]["source_geometry"] == ready_atlas["source_geometry"]
    assert seen["attach"]["task_context"] == kwargs["descriptor"]["metadata"]["site_task_context"]
    assert seen["attach"]["output_root"] == tmp_path / "pipeline/website_scene_preparation/assembly_coverage"
    assert seen["compile"]["task_masks"] == attached
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/task_masks.coverage.json").read_text()) == attached
    # A video outside the capture is never read.
    outside = tmp_path.parent / (tmp_path.name + "-outside.mov")
    outside.write_bytes(b"other")
    kwargs["clean_plate"]["input_video_path"] = str(outside)
    handoff.prepare_website_scene_handoff(**kwargs)
    assert seen["attach"]["source_video"] is None


def test_identified_targets_gain_published_specs_after_coverage_and_before_compile(tmp_path, monkeypatch,
                                                                                   ready_atlas):
    kwargs = inputs(tmp_path)
    masks = {"targets": [{"target_id": "dishwasher-1"}], "digest": "sha256:" + "b" * 64}
    kwargs["clean_plate"].update(task_masks=masks)
    covered = {**masks, "targets": [{"target_id": "dishwasher-1", "authoring_coverage": {"status": "complete"}}]}
    specified = {**covered, "targets": [{**covered["targets"][0], "object_spec": {"status": "researched"}}]}
    seen = {}
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage", lambda **_: covered)

    def attach(**value):
        seen["attach"] = value
        return specified
    monkeypatch.setattr("blueprint_pipeline.website_object_spec_research.attach_object_specs", attach)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **value: seen.update(compile=value) or (_ for _ in ()).throw(ValueError("stop")))
    handoff.prepare_website_scene_handoff(**kwargs)
    assert seen["attach"]["task_masks"] == covered
    assert seen["attach"]["task_context"] == kwargs["descriptor"]["metadata"]["site_task_context"]
    assert seen["attach"]["output_root"] == tmp_path / "pipeline/website_scene_preparation/object_spec"
    assert seen["compile"]["task_masks"] == specified
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/task_masks.coverage.json").read_text()) == specified
