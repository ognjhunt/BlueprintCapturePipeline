"""Ordinary USD backgrounds preserve geometry without claiming splat fidelity."""

import pytest
from pxr import Gf, Usd, UsdGeom

from blueprint_pipeline.native_task_appearance_frame_alignment import (
    NativeTaskAppearanceFrameAlignmentError, require_native_task_appearance_frame_alignment,
)
from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import (
    _materialize_native_particlefield_appearance, TaskEvaluationNativeArenaEpisodeCompilerError,
)
from blueprint_pipeline.native_task_nurec_render_setup import (
    appearance_render_path_from_plan, prepare_site_appearance_renderer,
)
from blueprint_pipeline.native_task_arena_policy_canary_worker import appearance_render_backend_from_plan
from blueprint_pipeline.native_task_arena_runtime_preflight_worker import _plain_nurec_volume_contract
from blueprint_pipeline.appearance_render_backend import validate_appearance_render_backend


def surface(path):
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Root").GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    cube = UsdGeom.Cube.Define(stage, "/Root/Table")
    cube.CreateSizeAttr(1.)
    cube.AddScaleOp().Set(Gf.Vec3d(1.2, .8, .05))
    stage.GetRootLayer().Save()
    return stage


def test_geometry_compiler_alignment_and_renderer_agree_without_conversion(tmp_path):
    source = tmp_path / "surface.usdc"
    stage = surface(source)
    original = source.read_bytes()
    prepared = _materialize_native_particlefield_appearance(source_path=source, output_root=tmp_path / "unused")
    assert prepared["representation"] == "usd_geometry"
    assert prepared["source_bytes_preserved"] is True
    assert prepared["representation_conversion_performed"] is False
    assert prepared["captured_scene_fidelity_proven"] is False
    assert "exact_learned_arrays_preserved" not in prepared
    alignment = require_native_task_appearance_frame_alignment(source,
        required_world_positions_m={"robot": [0., 0., .1]}, containment_margin_m=.15)
    assert alignment["measurement_authority"] == "usd_composed_geometry_bounds"
    assert "gaussian_count" not in alignment
    plan = {"appearance_frame_alignment": alignment, "objects": [{"semantic_role": "scene_appearance",
        "usd_path": source.name, "sha256": prepared["source_configured_appearance_digest"]}]}
    assert appearance_render_path_from_plan(plan) == "usd_geometry"
    preflight = _plain_nurec_volume_contract(tmp_path, plan)
    assert preflight["passed"] and preflight["render_path"] == "usd_geometry"
    backend = appearance_render_backend_from_plan(plan)
    assert validate_appearance_render_backend(backend) == backend
    assert backend["development_only"] and backend["kind"] == "isaac_usd_geometry"
    assert prepare_site_appearance_renderer(simulation_app=None, plan=plan)["status"] == "not_required"
    assert source.read_bytes() == original
    # A valid USD file is not enough: actual occupied bounds must fit the task.
    with pytest.raises(NativeTaskAppearanceFrameAlignmentError, match="excludes_scene_position"):
        require_native_task_appearance_frame_alignment(source, required_world_positions_m={"robot": [10., 0., 0.]})
    del stage


@pytest.mark.parametrize("bad", ["empty", "points", "volume", "units"])
def test_geometry_cannot_hide_unsupported_or_invalid_appearance(tmp_path, bad):
    source = tmp_path / "surface.usdc"
    stage = surface(source)
    if bad == "empty":
        stage.RemovePrim("/Root/Table")
    elif bad == "points":
        UsdGeom.Points.Define(stage, "/Root/UnqualifiedPointCloud")
    elif bad == "volume":
        stage.DefinePrim("/Root/UnqualifiedVolume", "Volume")
    else:
        UsdGeom.SetStageMetersPerUnit(stage, .01)
    stage.GetRootLayer().Save()
    with pytest.raises((TaskEvaluationNativeArenaEpisodeCompilerError, NativeTaskAppearanceFrameAlignmentError)):
        _materialize_native_particlefield_appearance(source_path=source, output_root=tmp_path / "unused")
