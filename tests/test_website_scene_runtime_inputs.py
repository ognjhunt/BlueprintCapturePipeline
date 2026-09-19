import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.website_scene_runtime_inputs import prepare_website_runtime_inputs
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_website_task_preparation import _arguments


def test_real_native_collision_conversion_preserves_background_and_separate_subject(tmp_path):
    args = _arguments(tmp_path)
    preparation = compile_website_scene_preparation(**args)
    kwargs = {"preparation": preparation, "base_scene": args["base_scene"], "source_geometry": args["source_geometry"], "task_masks": args["task_masks"], "output_root": tmp_path / "native"}
    value = prepare_website_runtime_inputs(**kwargs)
    assert value["status"] == "background_collision_prepared"
    assert value["appearance_removal_required"] is False
    assert value["collision_excision_required"] is False
    assert value["simulator_ready"] is False
    assert value["appearance"]["status"] == "awaiting_splat_frame_binding"
    assert value["object_authoring"]["source_frames"] == preparation["authoring_inputs"]["source_frames"]
    stage = Usd.Stage.Open(value["collision"]["path"])
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 1  # No manufactured subject prim in the background.
    assert meshes[0].HasAPI(UsdPhysics.CollisionAPI)
    assert not meshes[0].HasAPI(UsdPhysics.RigidBodyAPI)
    points = np.asarray(UsdGeom.Mesh(meshes[0]).GetPointsAttr().Get())
    # Original registered tabletop runtime Y=1.75 -> estimated simulator Z=3.5.
    assert np.any(np.isclose(points[:, 2], 3.5))
    assert prepare_website_runtime_inputs(**kwargs) == value
    assert json.loads((tmp_path / "native/runtime_inputs.json").read_text()) == value


@pytest.mark.parametrize("changed", ["preparation", "background", "source_frame"])
def test_native_handoff_refuses_changed_task_or_bytes(tmp_path, changed):
    args = _arguments(tmp_path)
    preparation = compile_website_scene_preparation(**args)
    if changed == "preparation":
        preparation["subject"]["description"] = "another object"
    elif changed == "background":
        Path(args["base_scene"]["collision_mesh_path"]).write_bytes(b"changed")
    else:
        Path(preparation["authoring_inputs"]["source_frames"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="website_native_.*changed"):
        prepare_website_runtime_inputs(preparation=preparation, base_scene=args["base_scene"],
                                       source_geometry=args["source_geometry"], task_masks=args["task_masks"],
                                       output_root=tmp_path / "native")


def test_revised_scale_cannot_reuse_previous_collision_cache(tmp_path):
    args = _arguments(tmp_path)
    preparation = compile_website_scene_preparation(**args)
    old = prepare_website_runtime_inputs(preparation=preparation, base_scene=args["base_scene"],
                                       source_geometry=args["source_geometry"], task_masks=args["task_masks"],
                                         output_root=tmp_path / "native")
    args["base_scene"]["meters_per_unit"] = 3.0
    preparation = compile_website_scene_preparation(**args)
    new = prepare_website_runtime_inputs(preparation=preparation, base_scene=args["base_scene"],
                                       source_geometry=args["source_geometry"], task_masks=args["task_masks"],
                                         output_root=tmp_path / "native")
    assert new["collision"]["path"] != old["collision"]["path"]
    assert new["collision"]["digest"] != old["collision"]["digest"]


def test_native_collision_cannot_disagree_with_subject_coordinate_conversion(tmp_path):
    args = _arguments(tmp_path)
    preparation = compile_website_scene_preparation(**args)
    preparation["coordinate_frame"]["declared_meters_per_unit"] = 3.0
    preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    with pytest.raises(ValueError, match="website_native_coordinate_frame_mismatch"):
        prepare_website_runtime_inputs(preparation=preparation, base_scene=args["base_scene"],
                                       source_geometry=args["source_geometry"], task_masks=args["task_masks"],
                                       output_root=tmp_path / "native")


def test_capture_processing_rights_are_required_even_without_paid_execution(tmp_path):
    args = _arguments(tmp_path)
    args["task_context"]["capture_rights"]["derived_scene_generation_allowed"] = False
    args["task_context"]["context_digest"] = canonical_digest(args["task_context"], digest_field="context_digest")
    preparation = compile_website_scene_preparation(**args)
    with pytest.raises(ValueError, match="website_scene_processing_rights_required"):
        prepare_website_runtime_inputs(preparation=preparation, base_scene=args["base_scene"],
                                       source_geometry=args["source_geometry"], task_masks=args["task_masks"],
                                       output_root=tmp_path / "native")
    assert not (tmp_path / "native").exists()


def test_original_observations_reach_existing_authoring_without_background_excision(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_configuration_content_agents_driver import (
        _dependency_candidate, _normalize_candidate, _reference_frames,
    )
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    args = _arguments(tmp_path)
    preparation = compile_website_scene_preparation(**args)
    value = prepare_website_runtime_inputs(preparation=preparation, base_scene=args["base_scene"],
        source_geometry=args["source_geometry"], task_masks=args["task_masks"], output_root=tmp_path / "native")
    dependencies = [{"status": "completed", "output_artifacts": value["authoring_dependency_artifacts"]}]
    config = value["object_authoring"]["configuration"]
    frames = _reference_frames({"configuration": config}, dependencies)
    assert len(frames) == 1  # Only the selected subject's tracked observation.
    assert _sha256_file(frames[0]) == args["source_geometry"]["frames"][0]["image_digest"]
    record, candidate = _dependency_candidate(dependencies)
    assert record["digest"] == value["object_authoring"]["source_candidate"]["digest"]
    stage = Usd.Stage.Open(str(candidate))
    meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 1 and len(meshes[0].GetFaceVertexCountsAttr().Get()) == 24
    points = np.asarray(meshes[0].GetPointsAttr().Get())
    assert np.allclose(points[:, 2], 3.5)  # Registered support-frame source patch.
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())
    _normalize_candidate(candidate, tmp_path / "authoring_candidate.usda")
    observed = json.loads(Path(value["object_authoring"]["observation_manifest"]["path"]).read_text())
    assert observed["complete_object_geometry"] is False
    assert observed["physical_measurement_proven"] is False
    assert observed["background_modified"] is False
    # A forged or stale reference must fail before any authoring call.
    frames[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact_changed"):
        _reference_frames({"configuration": config}, dependencies)


def test_original_observations_cannot_be_reused_for_another_subject(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_configuration_content_agents_driver import _reference_frames
    args = _arguments(tmp_path)
    preparation = compile_website_scene_preparation(**args)
    value = prepare_website_runtime_inputs(preparation=preparation, base_scene=args["base_scene"],
        source_geometry=args["source_geometry"], task_masks=args["task_masks"], output_root=tmp_path / "native")
    config = {**value["object_authoring"]["configuration"], "source_object_identity": "unrelated_movable_object"}
    with pytest.raises(ValueError, match="binding_invalid"):
        _reference_frames({"configuration": config}, [{"output_artifacts": value["authoring_dependency_artifacts"]}])
