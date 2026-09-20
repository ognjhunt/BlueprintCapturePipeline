from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import write_standard_3dgs_ply
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_native_appearance import prepare_native_appearance
from blueprint_pipeline.website_scene_runtime_inputs import prepare_website_runtime_inputs
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from tests.test_particlefield_usd import _splat
from tests.test_website_task_preparation import _arguments


def inputs(tmp_path):
    args = _arguments(tmp_path)
    data, _ = _splat(32)
    data.scales[:] = -5  # Small finite Gaussians in a room-sized fixture.
    source = Path(args["base_scene"]["splat_path"])
    write_standard_3dgs_ply(data, source)
    args["base_scene"]["splat_digest"] = _sha256_file(source)
    preparation = compile_website_scene_preparation(**args)
    return args, preparation, data


@pytest.mark.parametrize("up_axis", ["Y", "-Y"])
def test_native_appearance_uses_same_world_frame_without_editing_gaussians(tmp_path, monkeypatch, up_axis):
    args, preparation, data = inputs(tmp_path)
    if up_axis == "-Y":
        preparation["coordinate_frame"]["declared_up_axis"] = "-Y"
        transform = np.array(preparation["coordinate_frame"]["runtime_to_simulator"])
        transform[:3, 1:3] *= -1
        preparation["coordinate_frame"]["runtime_to_simulator"] = transform.tolist()
        preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    source = Path(args["base_scene"]["splat_path"])
    original = source.read_bytes()
    value = prepare_native_appearance(preparation=preparation, base_scene=args["base_scene"], output_root=tmp_path / "out")
    stage = Usd.Stage.Open(value["artifact"]["path"])
    field = stage.GetPrimAtPath(value["authoring"]["prim_path"])
    positions = np.asarray(field.GetAttribute("positions").Get())
    np.testing.assert_allclose(positions, data.xyz)
    actual = UsdGeom.Xformable(field).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world = np.asarray([actual.Transform(Gf.Vec3d(*p.astype(float))) for p in positions])
    transform = np.asarray(preparation["coordinate_frame"]["runtime_to_simulator"])
    np.testing.assert_allclose(world, data.xyz @ transform[:3, :3].T)
    assert source.read_bytes() == original
    assert value["physical_measurement_proven"] is False
    assert value["renderer_qualified"] is False
    assert value["appearance_removal_performed"] is False
    assert value["reconstruction_performed"] is False
    monkeypatch.setattr("blueprint_pipeline.particlefield_usd.write_particlefield_usd",
                        lambda *a, **kw: pytest.fail("unchanged background converted twice"))
    assert prepare_native_appearance(preparation=preparation, base_scene=args["base_scene"], output_root=tmp_path / "out") == value


def test_website_runtime_handoff_automatically_authors_ready_splat(tmp_path):
    args, preparation, _ = inputs(tmp_path)
    runtime = prepare_website_runtime_inputs(preparation=preparation, base_scene=args["base_scene"],
        source_geometry=args["source_geometry"], task_masks=args["task_masks"], output_root=tmp_path / "runtime")
    assert runtime["appearance"]["status"] == "native_appearance_authored"
    assert runtime["appearance"]["renderer_qualified"] is False
    assert runtime["simulator_ready"] is False


@pytest.mark.parametrize("changed", ["source", "coordinate", "cached_asset"])
def test_native_appearance_rejects_changed_source_frame_or_output(tmp_path, changed):
    args, preparation, _ = inputs(tmp_path)
    kwargs = dict(preparation=preparation, base_scene=args["base_scene"], output_root=tmp_path / "out")
    if changed == "source":
        Path(args["base_scene"]["splat_path"]).write_bytes(b"changed")
    elif changed == "coordinate":
        preparation["coordinate_frame"]["runtime_to_simulator"][0][0] *= 2
        preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    else:
        value = prepare_native_appearance(**kwargs)
        Path(value["artifact"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed|frame_invalid"):
        prepare_native_appearance(**kwargs)
