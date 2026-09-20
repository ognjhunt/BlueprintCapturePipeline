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
