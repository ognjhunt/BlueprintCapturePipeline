"""Static qualification reads the real assembly bytes: one passive task joint, tagged handle, per-link physics."""
import hashlib
import json
from pathlib import Path

import pytest
from pxr import Usd, UsdPhysics

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_articulated_static_qualification import (
    SCHEMA_VERSION, qualify_scene_configuration_articulated_asset_static,
)
from blueprint_pipeline.task_evaluation_scene_configuration_static_qualification import (
    TaskEvaluationSceneConfigurationStaticQualificationError,
)
from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import ARTICULATED_STATIC_CHECKS
from blueprint_pipeline.task_object_articulated_packaging import (
    articulation_graph_from_plan, package_astra_articulated_candidate,
)

IDENTITY = {"id": "website-subject-cab", "version": "v1"}


def _sealed(tmp_path):
    from tests.test_task_object_articulated_packaging import fixture
    plan, requests, results, bounds = fixture(tmp_path)
    receipt = package_astra_articulated_candidate(requests=requests, authoring_results=results, plan=plan,
                                                  output_root=tmp_path / "packaged", physics_bounds=bounds)
    completion = dict(receipt["physics_completion"])
    completion["metric_envelope_validation"] = {"status": "within_preregistered_metric_envelope"}
    completion["completion_digest"] = canonical_digest(completion, digest_field="completion_digest")
    graph = {"schema_version": "task_evaluation_articulated_replacement_graph.v1", "asset_id": IDENTITY["id"],
             "asset_version": IDENTITY["version"], "articulation_graph": articulation_graph_from_plan(plan),
             "task_joint_prim_path": completion["task_joint_prim_path"],
             "task_link_prim_path": completion["task_link_prim_path"],
             "fixed_base_body_prim_path": completion["fixed_base_body_prim_path"],
             "handle_prim_paths": completion["handle_prim_paths"],
             "handle_grasp_point_link_m": completion["handle_grasp_point_link_m"],
             "link_prim_paths": {row["link_id"]: row["prim_path"] for row in completion["links"]},
             "assembly_plan": dict(plan), "physics_bounds": bounds, "physics_authority_granted": False,
             "authoring_backend": "astra_cad_blender_v1"}
    asset = Path(receipt["asset"]["path"])
    authoring = {"schema_version": "task_evaluation_articulated_replacement_authoring_result.v1",
                 "status": "authored_candidate_pending_qualification", "asset_kind": "articulated_assembly",
                 "replacement_identity": dict(IDENTITY),
                 "output_usd": {"sha256": "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest(),
                                "size_bytes": asset.stat().st_size},
                 "candidate_physics_completion": completion, "physics_authority_granted": False, "result_digest": ""}
    authoring["result_digest"] = canonical_digest(authoring, digest_field="result_digest")
    return asset, graph, authoring


def _reseal(value, field="result_digest"):
    value[field] = ""
    value[field] = canonical_digest(value, digest_field=field)
    return value


def test_composed_assembly_passes_every_articulated_static_check(tmp_path):
    asset, graph, authoring = _sealed(tmp_path)
    result = qualify_scene_configuration_articulated_asset_static(
        asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
        replacement_identity=IDENTITY, output_path=tmp_path / "static.json")
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["status"] == "authored_structure_statically_qualified"
    assert result["asset_kind"] == "articulated_assembly" and result["structural_findings"] == []
    assert result["checks"] == dict(ARTICULATED_STATIC_CHECKS)
    assert result["task_joint"]["joint_id"] == "task_part_joint"
    assert result["task_joint"]["limits"][0] == 0.0 and result["task_joint"]["reset_position"] == 0.0
    assert result["claim_boundary"] == {"native_simulator_import_qualified": False, "physical_equivalence_proven": False,
                                        "generated_geometry_is_observed_truth": False, "joint_travel_is_measured": False}
    observed = result["observed_structure"]
    assert set(observed["links"]) == {"carcass", "drawer_0", "drawer_1", "drawer_2"}
    assert observed["links"]["drawer_1"]["collision_prim_paths"] == [
        "/Asset/links/drawer_1/collision/FinalVisualShape", "/Asset/links/drawer_1/collision/handle"]
    written = json.loads((tmp_path / "static.json").read_text())
    assert written["result_digest"] == canonical_digest(written, digest_field="result_digest")
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError, match="output_exists"):
        qualify_scene_configuration_articulated_asset_static(
            asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "static.json")


def test_a_position_servo_on_the_task_joint_is_refused(tmp_path):
    """A drive that can open the drawer by itself would let the asset do the robot's work."""
    asset, graph, authoring = _sealed(tmp_path)
    unpacked = tmp_path / "servo.usdc"
    stage = Usd.Stage.Open(str(asset))
    stage.Flatten().Export(str(unpacked))
    edited = Usd.Stage.Open(str(unpacked))
    joint = edited.GetPrimAtPath("/Asset/joints/task_part_joint")
    UsdPhysics.DriveAPI(joint, "linear").CreateStiffnessAttr().Set(400.0)
    edited.GetRootLayer().Save()
    from pxr import Sdf, UsdUtils
    servo_asset = tmp_path / "servo.usdz"
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(unpacked)), str(servo_asset))
    authoring["output_usd"] = {"sha256": "sha256:" + hashlib.sha256(servo_asset.read_bytes()).hexdigest(),
                               "size_bytes": servo_asset.stat().st_size}
    _reseal(authoring)
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=servo_asset, graph_spec=graph, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "servo.json")
    assert "replacement_target_joint_position_servo_forbidden" in error.value.codes


def test_graph_spec_and_completion_must_agree_with_the_actual_bytes(tmp_path):
    asset, graph, authoring = _sealed(tmp_path)
    moved = dict(graph, task_joint_prim_path="/Asset/joints/other_joint")
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=asset, graph_spec=moved, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "a.json")
    assert "replacement_physics_completion_invalid" in error.value.codes
    heavier = json.loads(json.dumps(authoring))
    heavier["candidate_physics_completion"]["links"][0]["mass_kg"] = 99.0
    _reseal(heavier["candidate_physics_completion"], "completion_digest")
    _reseal(heavier)
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=asset, graph_spec=graph, authoring_receipt=authoring | heavier,
            replacement_identity=IDENTITY, output_path=tmp_path / "b.json")
    assert "replacement_physics_completion_invalid" in error.value.codes
    granted = dict(graph, physics_authority_granted=True)
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=asset, graph_spec=granted, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "c.json")
    assert "replacement_graph_spec_invalid" in error.value.codes


def test_a_rigid_single_solid_cannot_pass_the_articulated_gate(tmp_path):
    from tests.test_task_object_articulated_packaging import _part
    from blueprint_pipeline.task_object_simready_packaging import package_astra_candidate
    request, result, _ = _part(tmp_path / "solid", object_id="solid", dimensions=[0.42, 0.55, 0.62],
                               mass_kg=12.0, density=(60.0, 140.0),
                               bounds={"mass_kg": [4.0, 40.0], "static_friction": [0.3, 0.8],
                                       "dynamic_friction": [0.2, 0.6], "restitution": [0.0, 0.2]})
    rigid = package_astra_candidate(request=request, authoring_result=result, output_root=tmp_path / "rigid",
                                    physics_bounds={"mass_kg": [4.0, 40.0], "static_friction": [0.3, 0.8],
                                                    "dynamic_friction": [0.2, 0.6], "restitution": [0.0, 0.2]})
    _asset, graph, authoring = _sealed(tmp_path / "reference")
    rigid_asset = Path(rigid["asset"]["path"])
    authoring["output_usd"] = {"sha256": "sha256:" + hashlib.sha256(rigid_asset.read_bytes()).hexdigest(),
                               "size_bytes": rigid_asset.stat().st_size}
    _reseal(authoring)
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=rigid_asset, graph_spec=graph, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "rigid.json")
    assert "replacement_single_target_joint_required" in error.value.codes
    assert "replacement_link_set_disagrees_with_graph" in error.value.codes
