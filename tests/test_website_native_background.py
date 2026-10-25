import json
from pathlib import Path
import shutil

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_native_background import prepare_collision_stage, prepare_appearance_stage
from blueprint_pipeline.website_scene_runtime_inputs import prepare_website_runtime_inputs
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from blueprint_pipeline.task_evaluation_scene_configuration_adapters import SceneConfigurationAdapterRegistry
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_adapters import builtin_scene_configuration_adapter_handlers
from blueprint_pipeline.task_evaluation_scene_configuration_content_agents_driver import _reference_frames, _dependency_candidate
from blueprint_pipeline.task_evaluation_scene_configuration_stage_configuration import validate_immutable_stage_configurations
from tests.test_website_task_preparation import _arguments


def test_void_clipping_preserves_floor_boundary_and_all_outside_area():
    import numpy as np
    from blueprint_pipeline.website_native_background import exclude_box_interior
    triangles = np.array([[[-1., -1., 0.], [2., -1., 0.], [0., 2., 0.]],
                          [[.2, .2, .5], [.4, .2, .5], [.2, .4, .5]],
                          [[-1., .5, .5], [.5, -1., .5], [.5, .5, .5]]])
    before = triangles.copy()
    kept, pieces, report = exclude_box_interior(triangles, minimum=[0, 0, 0], maximum=[1, 1, 1])
    assert kept.tolist() == [0]
    assert report["fully_interior_faces"] == 1 and report["clipped_faces"] == 1
    assert not np.any(np.all((pieces.mean(axis=1) > 0) & (pieces.mean(axis=1) < 1), axis=1))
    def area(rows):
        return np.linalg.norm(np.cross(rows[:, 1] - rows[:, 0], rows[:, 2] - rows[:, 0]), axis=1).sum() / 2
    assert area(triangles) == pytest.approx(area(triangles[kept]) + area(pieces) + report["excluded_area_m2"])
    np.testing.assert_array_equal(triangles, before)


def _enclosed_fixture_preparation(mode="fixed_in_scene", excludes=True):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    review = {"mode": mode, "source_frame_ids": ["capture-view"], "reason": "Capture-reviewed candidate enclosure.",
              "body_envelope_excludes_environment": excludes}
    value = {"claim_ceiling": "development_only", "authoring_inputs": {
        "source_frames": [{"frame_id": "capture-view"}], "configuration": {
            "construction_constraints": {"assembly_mounting": review},
            "body_extent_m": {"depth": 1., "width": 1., "height": 1.},
            "mechanism": {"estimated_front_normal_world": [1., 0., 0.]}}},
        "subject": {"geometry_origin": "removed_before_reconstruction",
                    "aabb_min_xyz": [-.5, -.5, 0.], "aabb_max_xyz": [.5, .5, 1.]}}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


@pytest.mark.parametrize("mode,excludes", [("fixed_in_scene", None), ("fixed_in_scene", False), ("free", False), ("unknown", None)])
def test_no_void_is_assumed_for_open_movable_or_unknown_objects(tmp_path, mode, excludes):
    from blueprint_pipeline.website_native_background import reconcile_enclosed_fixture_collision
    assert reconcile_enclosed_fixture_collision(source=tmp_path / "not-read.usdc",
        preparation=_enclosed_fixture_preparation(mode, excludes), output_root=tmp_path / "unused") is None
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("yaw", [0, 35])
def test_usd_void_derivative_preserves_source_and_remaps_subsets(tmp_path, yaw):
    import math
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from blueprint_pipeline.website_native_background import reconcile_enclosed_fixture_collision
    source = tmp_path / "background.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, "/Room")
    if yaw:
        root.AddTranslateOp().Set((3., -2., .1))
        root.AddRotateZOp().Set(yaw)
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Room/collision")
    points = np.array([[-2, -2, 0], [2, -2, 0], [0, 2, 0],
                       [-.2, -.2, .5], [.2, -.2, .5], [-.2, .2, .5],
                       [-1, -.2, .5], [1, -.2, .5], [0, 1, .5]], dtype=np.float32)
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr([3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr(list(range(9)))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    subset = UsdGeom.Subset.Define(stage, "/Room/collision/selected")
    subset.CreateElementTypeAttr("face")
    subset.CreateIndicesAttr([1, 2])
    stage.GetRootLayer().Save()
    digest = _sha256_file(source)
    preparation = _enclosed_fixture_preparation()
    if yaw:
        from blueprint_pipeline.decision_evidence_contracts import canonical_digest
        angle = math.radians(yaw)
        half = (math.sin(angle) + math.cos(angle)) / 2
        preparation["subject"].update(aabb_min_xyz=[3 - half, -2 - half, .1],
                                      aabb_max_xyz=[3 + half, -2 + half, 1.1])
        preparation["authoring_inputs"]["configuration"]["mechanism"]["estimated_front_normal_world"] = [math.cos(angle), math.sin(angle), 0.]
        preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    arguments = dict(source=source, preparation=preparation, output_root=tmp_path / "derived")
    receipt = reconcile_enclosed_fixture_collision(**arguments)
    assert receipt["source_bytes_unchanged"] and receipt["outside_volume_geometry_preserved"]
    assert receipt["appearance_modified"] is False and receipt["native_physics_qualified"] is False
    assert _sha256_file(source) == digest
    assert len(mesh.GetFaceVertexCountsAttr().Get()) == 3  # No in-memory source-layer mutation either.
    derived = Usd.Stage.Open(receipt["output"]["path"])
    result = UsdGeom.Mesh(derived.GetPrimAtPath("/Room/collision"))
    np.testing.assert_array_equal(np.asarray(result.GetPointsAttr().Get())[:len(points)], points)
    assert list(result.GetFaceVertexIndicesAttr().Get())[:3] == [0, 1, 2]
    ids = UsdGeom.Subset(derived.GetPrimAtPath("/Room/collision/selected")).GetIndicesAttr().Get()
    assert ids and max(ids) < len(result.GetFaceVertexCountsAttr().Get()) and 0 not in ids
    assert reconcile_enclosed_fixture_collision(**arguments) == receipt
    Path(receipt["output"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="cache_changed"):
        reconcile_enclosed_fixture_collision(**arguments)


def test_fixture_void_uses_same_bounded_scene_pose_without_rewriting_capture_bounds(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.website_native_background import reconcile_enclosed_fixture_collision
    source = tmp_path / "background.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, "/Room")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Room/collision")
    mesh.CreatePointsAttr([(-.2, -.2, .5), (.2, -.2, .5), (-.2, .2, .5)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    preparation = _enclosed_fixture_preparation()
    subject = json.loads(json.dumps(preparation["subject"]))
    preparation["coordinate_frame"] = {"placement_uncertainty_m": .1}
    preparation["scene_instance_pose_world"] = {"position_world_m": [.03, .04, 0.], "orientation_xyzw": [0., 0., 0., 1.]}
    preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    receipt = reconcile_enclosed_fixture_collision(source=source, preparation=preparation, output_root=tmp_path / "derived")
    assert receipt["binding"]["center_world_m"] == [.03, .04, 0.]
    assert receipt["binding"]["scene_instance_pose_refinement"]["maximum_corner_displacement_m"] == pytest.approx(.05)
    assert preparation["subject"] == subject
    preparation["scene_instance_pose_world"]["position_world_m"][0] = .2
    preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    with pytest.raises(ValueError, match="outside_registration_uncertainty"):
        reconcile_enclosed_fixture_collision(source=source, preparation=preparation, output_root=tmp_path / "other")


def staged(tmp_path, *, appearance=False):
    control = tmp_path / "control"
    control.mkdir()
    if appearance:
        from tests.test_website_native_appearance import inputs as appearance_inputs
        args, preparation, _ = appearance_inputs(control)
    else:
        args = _arguments(control)
        preparation = compile_website_scene_preparation(**args)
    runtime = prepare_website_runtime_inputs(preparation=preparation, base_scene=args["base_scene"],
        source_geometry=args["source_geometry"], task_masks=args["task_masks"], output_root=control / "native")
    inputs = prepare_collision_stage(control / "native/runtime_inputs.json")
    if appearance:
        inputs["appearance"] = prepare_appearance_stage(control / "native/runtime_inputs.json")
        seen = {row["contract_path"] for row in inputs["references"]}
        inputs["references"] += [row for row in inputs["appearance"]["references"] if row["contract_path"] not in seen]
    worker = tmp_path / "worker"
    worker.mkdir()
    rows = []
    for i, row in enumerate(inputs["references"]):
        path = worker / f"transport-{i}"
        shutil.copyfile(row["path"], path)
        rows.append({"contract_path": row["contract_path"], "materialized_path": str(path),
                     "digest": row["digest"], "size_bytes": row["size_bytes"],
                     "full_byte_service_account_readback_passed": True})
    envelope = {"materialized_references": rows, "recipe": {"stage_sequence": [inputs["stage"]],
        "subject_identity": runtime["object_authoring"]["configuration"]["replacement_identity"]}}
    config_path = worker / "configuration.json"
    write_json(config_path, inputs["configuration"])
    # The worker must consume only transported bytes, with no accidental access
    # to the website/control-plane machine's original paths.
    shutil.rmtree(control)
    return runtime, inputs, envelope, config_path


def execute(tmp_path, inputs, envelope, config_path):
    registry = SceneConfigurationAdapterRegistry(builtin_scene_configuration_adapter_handlers())
    validate_immutable_stage_configurations(envelope=envelope, configurations={"stage-2": inputs["configuration"]})
    return registry.execute(stage=inputs["stage"], envelope=envelope, configuration=inputs["configuration"],
        configuration_path=config_path, dependency_results=(), output_root=tmp_path / "result")


def test_native_collision_stage_preserves_background_and_transports_original_references(tmp_path):
    runtime, inputs, envelope, config_path = staged(tmp_path)
    result = execute(tmp_path, inputs, envelope, config_path)
    assert result["status"] == "completed"
    assert result["provider_mutations_performed"] == 0
    artifacts = {row["role"]: row for row in result["output_artifacts"]}
    collision = artifacts["configured_collision_without_source_object"]
    assert _sha256_file(Path(collision["path"])) == runtime["collision"]["digest"]
    config = runtime["object_authoring"]["configuration"]
    frames = _reference_frames({"configuration": config}, [result])
    assert frames and all(frame.is_relative_to(tmp_path / "result") for frame in frames)
    record, candidate = _dependency_candidate([result])
    assert record["digest"] == runtime["object_authoring"]["source_candidate"]["digest"]
    assert candidate.is_file()
    receipt = json.loads(Path(artifacts["website_background_reuse_receipt"]["path"]).read_text())
    assert receipt["source_prim_excision_performed"] is False
    assert receipt["background_bytes_unchanged"] is True
    assert receipt["physical_measurement_proven"] is False


def test_both_native_background_stages_execute_without_removing_the_subject_again(tmp_path):
    runtime, inputs, envelope, config_path = staged(tmp_path, appearance=True)
    appearance = inputs["appearance"]
    stages = [appearance["stage"], inputs["stage"]]
    envelope["recipe"]["stage_sequence"] = stages
    configurations = {"stage-1": appearance["configuration"], "stage-2": inputs["configuration"]}
    validate_immutable_stage_configurations(envelope=envelope, configurations=configurations)
    appearance_config = tmp_path / "appearance-config.json"
    write_json(appearance_config, appearance["configuration"])
    registry = SceneConfigurationAdapterRegistry(builtin_scene_configuration_adapter_handlers())
    first = registry.execute(stage=stages[0], envelope=envelope, configuration=appearance["configuration"],
        configuration_path=appearance_config, dependency_results=(), output_root=tmp_path / "stage-1")
    second = registry.execute(stage=stages[1], envelope=envelope, configuration=inputs["configuration"],
        configuration_path=config_path, dependency_results=(first,), output_root=tmp_path / "stage-2")
    artifacts = {row["role"]: row for result in (first, second) for row in result["output_artifacts"]}
    assert artifacts["configured_appearance_without_source_object"]["digest"] == runtime["appearance"]["digest"]
    assert artifacts["configured_collision_without_source_object"]["digest"] == runtime["collision"]["digest"]
    frames = _reference_frames({"configuration": runtime["object_authoring"]["configuration"]}, [first, second])
    assert frames and all(frame.is_relative_to(tmp_path / "stage-2") for frame in frames)
    assert _dependency_candidate([first, second])[1].is_file()
    assert all(result["provider_mutations_performed"] == 0 for result in (first, second))


def test_prepared_appearance_rejects_another_scene_asset(tmp_path):
    _, inputs, envelope, _ = staged(tmp_path, appearance=True)
    appearance = inputs["appearance"]
    row = next(r for r in envelope["materialized_references"] if r["contract_path"].endswith(".appearance"))
    Path(row["materialized_path"]).write_bytes(b"different-scene")
    config = tmp_path / "config.json"
    write_json(config, appearance["configuration"])
    registry = SceneConfigurationAdapterRegistry(builtin_scene_configuration_adapter_handlers())
    with pytest.raises(RuntimeError, match="reference_invalid"):
        registry.execute(stage=appearance["stage"], envelope=envelope, configuration=appearance["configuration"],
            configuration_path=config, dependency_results=(), output_root=tmp_path / "stage-1")


@pytest.mark.parametrize("changed", ["collision", "frame", "subject", "runtime"])
def test_native_collision_stage_refuses_changed_transport_or_task(tmp_path, changed):
    runtime, inputs, envelope, config_path = staged(tmp_path)
    if changed == "subject":
        envelope["recipe"]["subject_identity"] = {"id": "different-task-object", "version": "v1"}
    else:
        suffix = {"collision": "scene.geometry.collision", "frame": ".frames.0", "runtime": ".runtime_inputs"}[changed]
        row = next(r for r in envelope["materialized_references"] if r["contract_path"].endswith(suffix))
        Path(row["materialized_path"]).write_bytes(b"changed")
    with pytest.raises((ValueError, RuntimeError), match="invalid|mismatch"):
        execute(tmp_path, inputs, envelope, config_path)
