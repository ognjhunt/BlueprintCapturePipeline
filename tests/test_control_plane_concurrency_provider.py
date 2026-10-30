"""ADP-009D: the fixture provider reads the collision layer in either USD format."""

from __future__ import annotations

import json
import shutil

import pytest

from scripts.control_plane_concurrency_provider import file_record, fixture_scene_artifacts


@pytest.mark.parametrize("layer_suffix", [".usdc", ".usda"])
def test_provider_reads_extensionless_collision_bytes_of_either_format(tmp_path, layer_suffix):
    # 2026-10-30, PR #3129 CI: a crate normalized layer copied to a .usda name was
    # "not a valid usda file". Earlier receipts still bind text layers.
    from pxr import Usd, UsdGeom, UsdPhysics

    layer = tmp_path / ("normalized_scene" + layer_suffix)
    stage = Usd.Stage.CreateNew(str(layer))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Root").GetPrim())
    for name in ("Subject", "Support"):
        UsdGeom.Mesh.Define(stage, "/Root/" + name).CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    stage.GetRootLayer().Save()
    stage = None
    assert layer.read_bytes()[:8] == (b"PXR-USDC" if layer_suffix == ".usdc" else b"#usda 1.")
    # Launch preparation materializes references under their digest, with no suffix.
    cas = tmp_path / "cas" / file_record(layer)["digest"][7:]
    cas.parent.mkdir()
    shutil.copyfile(layer, cas)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"source_object_id": "/Subject",
                                     "aabb_min_xyz_m": [0, 0, 0], "aabb_max_xyz_m": [0.2, 0.2, 0.3]}))
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"fixture frame")
    envelope = {
        "envelope_digest": "sha256:" + "e" * 64,
        "request": {"task": {"subject": {"identity": {"id": "fixture-subject", "version": "v1"}}}},
        "materialized_references": [
            {"contract_path": "task.subject.source_object", "materialized_path": str(selection)},
            {"contract_path": "scene.geometry.collision", "materialized_path": str(cas)},
        ],
        "render_inputs_result": {"derived_frames": [{"path": str(frame)}] * 4},
    }
    output = tmp_path / "provider"
    rows = fixture_scene_artifacts(envelope=envelope, output_root=output)

    assert rows[0]["actual_provider_calls"] == 0
    assert (output / "source-collision.usd").read_bytes() == cas.read_bytes()
    # The fixture's own authored outputs keep their text names.
    assert (output / "collision.usda").read_bytes().startswith(b"#usda")
    collision = Usd.Stage.Open(str(output / "collision.usda"))
    assert not collision.GetPrimAtPath("/Root/Subject").IsValid()
    assert collision.GetPrimAtPath("/Root/Support").HasAPI(UsdPhysics.CollisionAPI)
    receipt = json.loads((output / "collision-receipt.json").read_text())
    assert receipt["source_geometry"] == file_record(cas)
    assert receipt["removed_source_prim_path"] == "/Root/Subject"
