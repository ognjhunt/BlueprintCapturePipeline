from __future__ import annotations

import json

import pytest

from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline.common import write_json
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def inputs(tmp_path):
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    rows = []
    for name, kind in (("world.ply", "splat_ply"), ("collider.glb", "collider_mesh_glb")):
        path = pipeline / name
        path.write_bytes(name.encode())
        rows.append({"kind": kind, "local_path": str(path), "sha256": _sha256_file(path)[7:]})
    manifest = pipeline / "assets.json"
    write_json(manifest, {"world_id": "world-1", "downloads": rows})
    removal = pipeline / "removal.json"
    write_json(removal, {"entries": []})
    return dict(
        descriptor={"capture_id": "cap-1", "scene_id": "scene-1", "metadata": {
            "site_task_context": {"capture_id": "cap-1", "scene_id": "scene-1"}}},
        clean_plate={"privacy_verified": True, "status": "objects_removed", "source_geometry": {},
                     "task_masks": {}, "removal_manifest_path": str(removal)},
        provider_run={"status": "ready", "world_id": "world-1", "provider_run_id": "op-1",
                      "worldlabs_asset_materialization": {"manifest_path": str(manifest)}},
        capture_root=tmp_path, now=1800000000,
    )


def test_collected_world_drives_preparation_but_never_invents_execution(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    calls = []
    def compile(**value):
        calls.append(value)
        assert value["spend"]["max_total_spend_usd"] == 0
        assert value["base_scene"]["meters_per_unit"] is None
        assert value["base_scene"]["up_axis"] == "-Y"
        return {"status": "needs_input", "blockers": ["website_scene_execution_authority_required"],
                "digest": "sha256:" + "a" * 64, "thumbnail": {"path": "proposed-thumbnail.png"}}
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", compile)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "needs_input" and len(calls) == 1
    assert result["simulator_ready"] is False
    assert result["provider_mutation_performed"] is False
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/handoff.json").read_text()) == result


@pytest.mark.parametrize("change,reason", [
    ("privacy", "website_scene_preparation_pending"),
    ("pending", "website_reconstruction_pending"),
    ("world", "website_reconstruction_world_mismatch"),
    ("bytes", "website_reconstruction_asset_changed"),
    ("identity", "website_scene_task_identity_mismatch"),
])
def test_held_or_changed_inputs_do_not_enter_scene_construction(tmp_path, monkeypatch, change, reason):
    kwargs = inputs(tmp_path)
    if change == "privacy":
        kwargs["clean_plate"]["privacy_verified"] = False
    elif change == "pending":
        kwargs["provider_run"]["status"] = "processing"
    elif change == "world":
        kwargs["provider_run"]["world_id"] = "world-other"
    elif change == "bytes":
        (tmp_path / "pipeline/world.ply").write_bytes(b"changed")
    else:
        kwargs["descriptor"]["metadata"]["site_task_context"]["capture_id"] = "other"
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", lambda **_: pytest.fail("must not compile"))
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"] == [reason]
    assert result["simulator_ready"] is False


def test_geometry_controller_runs_only_after_visual_world_and_assets_are_ready(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    video = tmp_path / "walkthrough.mov"
    video.write_bytes(b"source")
    kwargs["clean_plate"].update(source_geometry=None, input_video_path=str(video),
        stage_manifest_path=str(tmp_path / "pipeline/clean_plate/stage.json"))
    calls = []
    def dispatch(**kw):
        calls.append(kw)
        raise ValueError("geometry_capacity_pending")
    monkeypatch.setattr("blueprint_pipeline.website_scene_geometry.run_website_scene_geometry", dispatch)
    kwargs["provider_run"]["status"] = "processing"
    assert handoff.prepare_website_scene_handoff(**kwargs)["blockers"] == ["website_reconstruction_pending"]
    assert calls == []
    kwargs["provider_run"]["status"] = "ready"
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"] == ["geometry_capacity_pending"]
    assert len(calls) == 1 and calls[0]["task_context"] == kwargs["descriptor"]["metadata"]["site_task_context"]
    assert kwargs["provider_run"]["status"] == "ready"
    assert result["geometry_controller_invoked"] is True


def test_provider_declared_scale_ground_and_anchor_view_enter_the_base_scene(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    pipeline = tmp_path / "pipeline"
    world = pipeline / "world.json"
    write_json(world, {"world_id": "world-1", "assets": {"splats": {
        "semantics_metadata": {"metric_scale_factor": 1.4771584, "ground_plane_offset": 1.6066047}}}})
    manifest = pipeline / "assets.json"
    write_json(manifest, {**json.loads(manifest.read_text()), "source_world_manifest": str(world)})
    kwargs["clean_plate"]["prepared_views"] = {"frames": [{"frame_id": "decoded-000000000"}, {"frame_id": "decoded-000000108"}]}
    seen = {}
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **value: seen.update(value) or (_ for _ in ()).throw(ValueError("stop")))
    handoff.prepare_website_scene_handoff(**kwargs)
    base = seen["base_scene"]
    assert base["meters_per_unit"] == 1.4771584 and base["ground_plane_offset_m"] == 1.6066047
    assert base["scale_authority"] == "provider_declared_estimate"
    assert base["anchor"] == {"kind": "first_input_view_camera", "frame_id": "decoded-000000000"}
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/base_scene.json").read_text()) == base


@pytest.mark.parametrize("support_ready", [False, True])
def test_visual_world_survives_deferred_support_and_full_masks_are_required_before_geometry(tmp_path, monkeypatch, support_ready):
    kwargs = inputs(tmp_path)
    video = tmp_path / "walkthrough.mov"
    video.write_bytes(b"source")
    plan = tmp_path / "pipeline/plan.json"
    write_json(plan, {"targets": [{"target_id": "support"}]})
    kwargs["clean_plate"].update(source_geometry=None, source_frames={"digest": "source"},
        input_video_path=str(video), removal_plan_path=str(plan), task_masks={"deferred_target_ids": ["support"]},
        stage_manifest_path=str(tmp_path / "pipeline/stage.json"))
    calls = []
    full_masks = {"targets": [{"target_id": "object"}, {"target_id": "support"}]}
    def resolve(**kw):
        calls.append("masks")
        assert not kw.get("defer_kept_static")
        assert kw["source_geometry"] == {"digest": "source"}
        if not support_ready:
            raise ValueError("support_unresolved")
        return full_masks
    def geometry(**kw):
        calls.append("geometry")
        assert kw["task_masks"] == full_masks
        raise ValueError("geometry_pending")
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.run_website_task_masks", resolve)
    monkeypatch.setattr("blueprint_pipeline.website_scene_geometry.run_website_scene_geometry", geometry)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["visual_reconstruction_ready"] is True and result["simulator_ready"] is False
    assert result["blockers"] == ["geometry_pending" if support_ready else "support_unresolved"]
    assert calls == (["masks", "geometry"] if support_ready else ["masks"])
    assert kwargs["provider_run"]["status"] == "ready"


def test_world_without_declared_scale_leaves_registration_to_estimate(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    kwargs["clean_plate"]["prepared_views"] = {"frames": [{"frame_id": "decoded-000000000"}]}
    seen = {}
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **value: seen.update(value) or (_ for _ in ()).throw(ValueError("stop")))
    handoff.prepare_website_scene_handoff(**kwargs)
    base = seen["base_scene"]
    assert base["meters_per_unit"] is None and base["anchor"] is None
    assert base["scale_authority"] == "registration_estimate"
