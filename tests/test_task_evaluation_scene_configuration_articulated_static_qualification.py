"""Static qualification reads the real assembly bytes: one passive task joint, tagged handle, per-link physics."""
import hashlib
import json
from pathlib import Path

import pytest
from pxr import Usd, UsdPhysics

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_articulated_static_qualification import (
    SCHEMA_VERSION, _development_hypothesis_findings,
    qualify_scene_configuration_articulated_asset_static,
)
from blueprint_pipeline.task_evaluation_scene_configuration_static_qualification import (
    TaskEvaluationSceneConfigurationStaticQualificationError,
)
from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import ARTICULATED_STATIC_CHECKS
from blueprint_pipeline.task_object_articulated_packaging import (
    articulation_graph_from_plan, package_astra_articulated_candidate,
)

IDENTITY = {"id": "website-subject-cab", "version": "v1"}


def _sealed(tmp_path, *, development_hypothesis=False):
    from tests.test_task_object_articulated_packaging import fixture
    plan, requests, results, bounds = fixture(tmp_path)
    if development_hypothesis:
        plan = json.loads(json.dumps(plan))
        dimensions = plan["assembly_dimensions_m"]
        depth = dimensions["depth_x"]
        source_depth = depth - 0.287
        source_min = [-source_depth / 2, -dimensions["width_y"] / 2, 0.0]
        source_max = [source_depth / 2, dimensions["width_y"] / 2, dimensions["height_z"]]
        plan["source_geometry"] = {
            "aabb_min_xyz_m": source_min, "aabb_max_xyz_m": source_max,
            "projected_depth_m": source_depth, "projected_width_m": dimensions["width_y"],
            "height_m": dimensions["height_z"],
            "authority": "retained_source_envelope_not_physical_measurement"}
        plan["development_geometry_hypothesis"] = {
            "schema_version": "articulated_cabinet_depth_hypothesis.v1",
            "claim_ceiling": "development_only", "basis": "bounded_cabinet_prior",
            "status": "development_only_estimate_disagrees_with_source",
            "source_aabb_min_xyz_m": source_min, "source_aabb_max_xyz_m": source_max,
            "estimated_depth_m": depth, "depth_interval_m": [depth - 0.1, depth + 0.1],
            "source_projected_depth_m": source_depth, "depth_disagreement_m": 0.287,
            "whole_assembly_mass_interval_kg": [15.0, 50.0],
            "revised_part_mass_bounds_kg": {part: row["mass_kg"] for part, row in bounds.items()},
            "estimated_usable_stroke_m": plan["task_joint"]["limits_m"][1],
            "minimum_opening_fraction": 0.6,
            "estimated_minimum_opening_m": round(0.6 * plan["task_joint"]["limits_m"][1], 5),
            "prior_record_schema_version": "website_drawer_depth_prior.v1",
            "reference_retrieved_date": "2026-09-23",
            "manufacturer_examples": [{"source": "fixture"}],
            "prior_comparison": {"reference_models_are_exact_match": False},
            "physical_measurement_proven": False,
        }
    receipt = package_astra_articulated_candidate(requests=requests, authoring_results=results, plan=plan,
                                                  output_root=tmp_path / "packaged", physics_bounds=bounds)
    completion = dict(receipt["physics_completion"])
    if development_hypothesis:
        from blueprint_pipeline.task_object_articulated_packaging import HANDLE_PROTRUSION_M
        expected = [depth + HANDLE_PROTRUSION_M, dimensions["width_y"], dimensions["height_z"]]
        actual = completion["collision_dimensions_m"]
        completion["metric_envelope_validation"] = {
            "status": "within_development_geometry_hypothesis",
            "frame": "assembly_frame_closed_plus_handle_protrusion",
            "expected_dimensions_m": expected, "observed_collision_dimensions_m": actual,
            "dimension_relative_errors": [abs(actual[i] - expected[i]) / expected[i] for i in range(3)],
            "maximum_dimension_relative_error": 0.2,
            "source_aabb_disagreement": {
                "source_projected_depth_m": source_depth,
                "estimated_depth_m": depth, "depth_disagreement_m": 0.287,
                "physical_measurement_proven": False}}
    else:
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


def test_a_grasp_point_outside_the_handle_is_refused(tmp_path):
    """The runtime reaches for this point; it has to be on the handle the bytes actually carry."""
    asset, graph, authoring = _sealed(tmp_path)
    ok = qualify_scene_configuration_articulated_asset_static(
        asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
        replacement_identity=IDENTITY, output_path=tmp_path / "ok.json")
    point = ok["task_contact"]["contact_point_link_m"]
    assert ok["task_contact"]["handle_prim_paths"] == ["/Asset/links/drawer_1/collision/handle"]
    bounds = ok["observed_structure"]["task_contact"]["handle_bounds_link_frame_m"]
    assert all(bounds["minimum"][i] - 1e-6 <= point[i] <= bounds["maximum"][i] + 1e-6 for i in range(3))
    assert [row["joint_id"] for row in ok["articulation_graph"]["joints"] if row["role"] == "target"] == ["task_part_joint"]
    moved = dict(graph, handle_grasp_point_link_m=[point[0] + 0.5, point[1], point[2]])
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=asset, graph_spec=moved, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "moved.json")
    assert "replacement_handle_grasp_point_outside_handle" in error.value.codes


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


@pytest.mark.parametrize(("change", "expected_code"), [
    ("target_parent", "replacement_target_joint_bodies_invalid"),
    ("target_axis", "replacement_target_joint_axis_mismatch"),
    ("fixed_joint_body", "replacement_joint_graph_topology_mismatch:drawer_0_fixed"),
    ("link_path", "replacement_graph_link_paths_disagree_with_usd"),
    ("collision_filter", "replacement_collision_filter_disagrees_with_graph"),
    ("assembly_plan", "replacement_assembly_plan_disagrees_with_usd"),
])
def test_resealed_graph_cannot_relabel_the_exact_jointed_asset(tmp_path, change, expected_code):
    asset, graph, authoring = _sealed(tmp_path)
    graph = json.loads(json.dumps(graph))
    joints = graph["articulation_graph"]["joints"]
    if change == "target_parent":
        next(row for row in joints if row["role"] == "target")["parent_link_id"] = "drawer_0"
    elif change == "target_axis":
        next(row for row in joints if row["role"] == "target")["axis"] = [0.0, 1.0, 0.0]
    elif change == "fixed_joint_body":
        next(row for row in joints if row["joint_id"] == "drawer_0_fixed")["parent_link_id"] = "drawer_2"
    elif change == "link_path":
        graph["link_prim_paths"]["drawer_1"] = "/Asset/links/other"
    elif change == "collision_filter":
        graph["articulation_graph"]["collision_pairs"][0]["collision_enabled"] = True
    else:
        graph["assembly_plan"]["links"][0]["rest_translation_m"][0] += 0.01
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "tampered.json")
    assert expected_code in error.value.codes


@pytest.mark.parametrize("field", ["physics_material", "collision_bounds_link_frame_m", "part_id"])
def test_resealed_completion_cannot_change_per_link_physics_or_identity(tmp_path, field):
    asset, graph, authoring = _sealed(tmp_path)
    authoring = json.loads(json.dumps(authoring))
    link = next(row for row in authoring["candidate_physics_completion"]["links"]
                if row["link_id"] == "drawer_1")
    if field == "physics_material":
        link[field]["static_friction"] = 0.5
    elif field == "collision_bounds_link_frame_m":
        link[field]["maximum"][0] += 0.02
    else:
        link[field] = "other_part"
    _reseal(authoring["candidate_physics_completion"], "completion_digest")
    _reseal(authoring)
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "tampered.json")
    assert "replacement_physics_completion_invalid" in error.value.codes


def test_resealed_completion_cannot_relabel_the_task_joint(tmp_path):
    asset, graph, authoring = _sealed(tmp_path)
    authoring = json.loads(json.dumps(authoring))
    target = next(row for row in authoring["candidate_physics_completion"]["joints"]
                  if row["role"] == "target")
    target["limits"][1] -= 0.01
    _reseal(authoring["candidate_physics_completion"], "completion_digest")
    _reseal(authoring)
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "tampered.json")
    assert "replacement_physics_completion_invalid" in error.value.codes


def test_malformed_resealed_completion_fails_with_a_typed_finding(tmp_path):
    asset, graph, authoring = _sealed(tmp_path)
    authoring["candidate_physics_completion"]["links"] = None
    _reseal(authoring["candidate_physics_completion"], "completion_digest")
    _reseal(authoring)
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "tampered.json")
    assert "replacement_physics_completion_invalid" in error.value.codes


def test_resealed_completion_cannot_hide_an_assembly_depth_change(tmp_path):
    asset, graph, authoring = _sealed(tmp_path)
    authoring["candidate_physics_completion"]["collision_dimensions_m"][0] += 0.1
    _reseal(authoring["candidate_physics_completion"], "completion_digest")
    _reseal(authoring)
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "tampered.json")
    assert "replacement_physics_completion_invalid" in error.value.codes


@pytest.mark.parametrize(("change", "expected_code"), [
    ("joint_axis", "replacement_target_joint_axis_mismatch"),
    ("fixed_joint_body", "replacement_joint_graph_topology_mismatch:drawer_0_fixed"),
    ("link_part", "replacement_link_identity_or_part_mismatch:drawer_1"),
])
def test_resealed_usdz_still_has_to_match_the_declared_assembly(tmp_path, change, expected_code):
    from pxr import Sdf, UsdUtils

    asset, graph, authoring = _sealed(tmp_path)
    flattened = tmp_path / "edited.usdc"
    assert Usd.Stage.Open(str(asset)).Flatten().Export(str(flattened))
    edited = Usd.Stage.Open(str(flattened))
    if change == "joint_axis":
        UsdPhysics.PrismaticJoint(edited.GetPrimAtPath(
            "/Asset/joints/task_part_joint")).GetAxisAttr().Set("Y")
    elif change == "fixed_joint_body":
        UsdPhysics.Joint(edited.GetPrimAtPath(
            "/Asset/joints/drawer_0_fixed")).GetBody0Rel().SetTargets(
                [Sdf.Path("/Asset/links/drawer_2")])
    else:
        edited.GetPrimAtPath("/Asset/links/drawer_1").SetCustomDataByKey(
            "blueprint:partId", "different_part")
    edited.GetRootLayer().Save()
    modified = tmp_path / "edited.usdz"
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(flattened)), str(modified))
    authoring["output_usd"] = {"sha256": "sha256:" + hashlib.sha256(modified.read_bytes()).hexdigest(),
                               "size_bytes": modified.stat().st_size}
    _reseal(authoring)
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=modified, graph_spec=graph, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "tampered.json")
    assert expected_code in error.value.codes


def test_development_depth_hypothesis_binds_usd_dimensions_mass_and_source_disagreement(tmp_path):
    asset, graph, authoring = _sealed(tmp_path, development_hypothesis=True)
    result = qualify_scene_configuration_articulated_asset_static(
        asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
        replacement_identity=IDENTITY, output_path=tmp_path / "qualified.json")
    observed = result["observed_structure"]
    assert observed["whole_assembly_mass_kg"] > 15.0
    assert observed["collision_dimensions_m"][0] > graph["assembly_plan"]["source_geometry"]["projected_depth_m"]
    assert authoring["candidate_physics_completion"]["metric_envelope_validation"]["source_aabb_disagreement"][
        "physical_measurement_proven"] is False

    graph["assembly_plan"]["development_geometry_hypothesis"]["whole_assembly_mass_interval_kg"] = [1.0, 2.0]
    with pytest.raises(TaskEvaluationSceneConfigurationStaticQualificationError) as error:
        qualify_scene_configuration_articulated_asset_static(
            asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
            replacement_identity=IDENTITY, output_path=tmp_path / "tampered.json")
    assert "replacement_development_geometry_hypothesis_invalid" in error.value.codes


def test_final_website_depth_prior_plan_matches_stage_four_schema():
    from tests.test_task_object_articulated_packaging import _depth_prior, _thin_website_cabinet
    from blueprint_pipeline.task_object_articulated_packaging import plan_articulated_assembly

    configuration = _thin_website_cabinet()
    hypothesis = _depth_prior(configuration)
    configuration["development_geometry_hypothesis"] = hypothesis
    configuration["mechanism"]["estimated_usable_stroke_m"] = hypothesis["estimated_usable_stroke_m"]
    configuration["mechanism"]["joint_limits"] = [0.0, hypothesis["estimated_usable_stroke_m"]]
    configuration["required_output"]["mass_kg_bounds"] = [4.0, 30.0]
    configuration["required_output"]["task_part_mass_kg_bounds"] = [0.5, 9.0]
    plan = plan_articulated_assembly(configuration)
    graph = articulation_graph_from_plan(plan)
    dimensions = [0.58, 0.42, 0.62]
    completion = {
        "collision_dimensions_m": dimensions,
        "metric_envelope_validation": {
            "status": "within_development_geometry_hypothesis",
            "frame": "assembly_frame_closed_plus_handle_protrusion",
            "expected_dimensions_m": dimensions,
            "observed_collision_dimensions_m": dimensions,
            "dimension_relative_errors": [0.0, 0.0, 0.0],
            "maximum_dimension_relative_error": 0.2,
            "source_aabb_disagreement": {
                "source_projected_depth_m": plan["source_geometry"]["projected_depth_m"],
                "estimated_depth_m": plan["assembly_dimensions_m"]["depth_x"],
                "depth_disagreement_m": plan["development_geometry_hypothesis"]["depth_disagreement_m"],
                "physical_measurement_proven": False}}}
    observed = {"collision_dimensions_m": dimensions,
                "links": {"carcass": {"mass_kg": 12.0},
                          **{f"drawer_{index}": {"mass_kg": 4.0} for index in range(3)}}}
    physics_bounds = {part: {"mass_kg": interval}
                      for part, interval in hypothesis["revised_part_mass_bounds_kg"].items()}
    assert plan["assembly_dimensions_m"]["depth_x"] == 0.55
    assert plan["source_geometry"]["projected_depth_m"] == 0.163
    assert _development_hypothesis_findings(
        plan=plan, completion=completion, observed=observed,
        graph=graph, physics_bounds=physics_bounds) == []
