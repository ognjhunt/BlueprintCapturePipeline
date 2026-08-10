from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from blueprint_pipeline.adp009d_sage_franka_placement import (
    SageFrankaPlacementError,
    materialize_registered_sage_franka_placement_packet,
    materialize_sage_collision_analysis_glb,
    materialize_sage_franka_placement_packet,
)


def _write_stage(path: Path, *, meters_per_unit: float = 1.0) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    root = UsdGeom.Xform.Define(stage, "/Root")
    root.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Collider")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateExtentAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 1, 0)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.SetDefaultPrim(root.GetPrim())
    assert stage.GetRootLayer().Save()
    assert stage.GetRootLayer().realPath == str(path)
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/Root")


def test_sage_conversion_preserves_world_coordinates_and_source_bytes(tmp_path) -> None:
    source = tmp_path / "scene.usda"
    _write_stage(source)
    before = source.read_bytes()

    receipt = materialize_sage_collision_analysis_glb(
        sage_usd_path=source,
        output_dir=tmp_path / "conversion",
    )

    assert source.read_bytes() == before
    assert receipt["source_usd_mutated"] is False
    assert receipt["simulation_asset_replacement"] is False
    assert receipt["mesh_count"] == 1
    assert receipt["vertex_count"] == 3
    assert receipt["triangle_count"] == 1
    glb = trimesh.load(receipt["analysis_glb"]["path"], force="mesh", process=False)
    stage_points = np.column_stack(
        (glb.vertices[:, 0], -glb.vertices[:, 2], glb.vertices[:, 1])
    )
    assert np.allclose(stage_points.min(axis=0), [1.0, 2.0, 3.0])
    assert np.allclose(stage_points.max(axis=0), [2.0, 3.0, 3.0])


def test_sage_conversion_rejects_nonmetric_stage(tmp_path) -> None:
    source = tmp_path / "centimeter_scene.usda"
    _write_stage(source, meters_per_unit=0.01)

    with pytest.raises(
        SageFrankaPlacementError, match="sage_collision_stage_not_meter_units"
    ):
        materialize_sage_collision_analysis_glb(
            sage_usd_path=source,
            output_dir=tmp_path / "conversion",
        )


def test_registered_sage_placement_receives_scene_and_task_identity_as_data(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "scene.usda"
    _write_stage(source)
    conversion = materialize_sage_collision_analysis_glb(
        sage_usd_path=source,
        output_dir=tmp_path / "conversion",
    )
    captured = {}

    def _propose(*, collision_glb_path, request, target_analysis):
        captured.update(
            collision_glb_path=collision_glb_path,
            request=request,
            target_analysis=target_analysis,
        )
        return {
            "placement": {"status": "runtime_visualization_candidate_only"},
            "render_options": {"robot_id": "franka_panda"},
        }

    monkeypatch.setattr(
        "blueprint_pipeline.adp009d_sage_franka_placement."
        "propose_external_scene_robot_placement",
        _propose,
    )
    digest_a = "sha256:" + "a" * 64
    digest_b = "sha256:" + "b" * 64
    digest_c = "sha256:" + "c" * 64

    packet = materialize_registered_sage_franka_placement_packet(
        conversion_receipt=conversion,
        output_dir=tmp_path / "placement",
        source_scene_digest=digest_a,
        scene_frame_binding_digest=digest_b,
        target_id="observed_object_17",
        target_label="observed rigid object",
        task_family="rigid_object_relocation",
        target_position_m=(1.0, 2.0, 0.5),
        target_binding_digest=digest_c,
        target_spatial_uncertainty_m=0.02,
        visual_confidence=0.91,
        metric_scale_status="provider_declared_not_independently_validated",
        collision_status="candidate_compiled",
    )

    assert packet["status"] == "placement_candidate_materialized"
    assert captured["request"]["source_scene_digest"] == digest_a
    assert captured["request"]["scene_frame_binding_digest"] == digest_b
    assert captured["request"]["target_binding_digest"] == digest_c
    assert captured["request"]["target_label"] == "observed rigid object"
    assert (
        captured["request"]["metric_scale_status"]
        == "provider_declared_not_independently_validated"
    )
    assert captured["target_analysis"]["selected_target"] == {
        "target_id": "observed_object_17",
        "target_label": "observed rigid object",
        "task_family": "rigid_object_relocation",
        "position_m": [1.0, 2.0, 0.5],
    }
    assert packet["policy_execution_authorized"] is False
    assert packet["native_contact_reachability_qualified"] is False


def test_legacy_placement_adapter_translates_through_registered_api(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    def _registered(**kwargs):
        captured.update(kwargs)
        return {"status": "blocked"}

    monkeypatch.setattr(
        "blueprint_pipeline.adp009d_sage_franka_placement."
        "materialize_registered_sage_franka_placement_packet",
        _registered,
    )

    result = materialize_sage_franka_placement_packet(
        conversion_receipt={"fixture": True},
        output_dir=tmp_path / "placement",
    )

    assert result == {"status": "blocked"}
    assert captured["task_family"] == "rigid_opaque_pick_place"
    assert captured["target_spatial_uncertainty_m"] == 0.0311


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"source_scene_digest": "bad"}, "sage_franka_source_scene_digest_invalid"),
        ({"target_id": ""}, "sage_franka_target_id_invalid"),
        (
            {"target_spatial_uncertainty_m": -1.0},
            "sage_franka_target_spatial_uncertainty_invalid",
        ),
        ({"visual_confidence": 1.1}, "sage_franka_visual_confidence_invalid"),
    ],
)
def test_registered_sage_placement_rejects_unbound_inputs(
    tmp_path, override, error
) -> None:
    source = tmp_path / "scene.usda"
    _write_stage(source)
    conversion = materialize_sage_collision_analysis_glb(
        sage_usd_path=source,
        output_dir=tmp_path / "conversion",
    )
    arguments = {
        "conversion_receipt": conversion,
        "output_dir": tmp_path / "placement",
        "source_scene_digest": "sha256:" + "a" * 64,
        "scene_frame_binding_digest": "sha256:" + "b" * 64,
        "target_id": "observed_object_17",
        "target_label": "observed rigid object",
        "task_family": "rigid_object_relocation",
        "target_position_m": (1.0, 2.0, 0.5),
        "target_binding_digest": "sha256:" + "c" * 64,
        "target_spatial_uncertainty_m": 0.02,
        "visual_confidence": 0.91,
        "metric_scale_status": "provider_declared_not_independently_validated",
        "collision_status": "candidate_compiled",
    }
    arguments.update(override)

    with pytest.raises(SageFrankaPlacementError, match=error):
        materialize_registered_sage_franka_placement_packet(**arguments)
