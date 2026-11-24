"""Free parametric shapes (owner direction 2026-11-23): a carton, tote, open bin, tray or can made from the video's
estimated dimensions, at no cost, as a rigid body with collision, mass and bounds; and simulated variation of it."""
from __future__ import annotations

import hashlib
import json

import pytest

from blueprint_pipeline import website_parametric_object as parametric

BOUNDS = {"mass_kg": [3.5, 4.5], "static_friction": [.5, .8], "dynamic_friction": [.4, .5], "restitution": [.1, .3]}


@pytest.mark.parametrize("shape, parts, open_top", [("carton", 1, False), ("cylinder", 1, False),
                                                    ("open_bin", 5, True), ("tray", 5, True), ("tote", 9, True)])
def test_each_shape_fills_exactly_its_outer_dimensions(shape, parts, open_top):
    chosen = parametric.design(shape, [0.6, 0.4, 0.3 if shape != "tray" else 0.05])
    rows = parametric.part_boxes(chosen)
    assert len(rows) == parts
    shape_mesh = parametric.mesh(chosen)
    assert shape_mesh.extents.tolist() == pytest.approx(chosen["dimensions_m"], abs=1e-6)
    assert shape_mesh.bounds.mean(axis=0).tolist() == pytest.approx([0, 0, 0], abs=1e-6)
    # An open container is open: its middle, above the floor, holds no part.
    inside = [0.0, 0.0, 0.0]
    hit = any(all(abs(p - c) <= e / 2 for p, c, e in zip(inside, row["center_m"], row["extents_m"], strict=True))
              for row in rows)
    assert hit is not open_top


def test_a_can_takes_the_footprints_mean_side_and_bad_dimensions_are_refused():
    assert parametric.design("cylinder", [0.08, 0.06, 0.12])["dimensions_m"] == [0.07, 0.07, 0.12]
    for shape, dims in (("crate", [.3, .3, .3]), ("carton", [.3, .3]), ("carton", [.3, .3, 9]),
                        ("carton", [.3, float("nan"), .3]), ("carton", [True, .3, .3])):
        with pytest.raises(ValueError, match="website_parametric_object_design_invalid"):
            parametric.design(shape, dims)


def test_the_candidate_sits_where_the_object_stood_and_costs_nothing(tmp_path):
    from pxr import Usd, UsdGeom
    chosen = parametric.design("tote", [0.6, 0.4, 0.3])
    record = parametric.write_candidate(chosen, center_xyz_m=[1.0, 2.0, 0.15], path=tmp_path / "candidate.usda")
    assert (record["source"], record["authoring_cost_usd"], record["part_count"]) == ("parametric", 0.0, 9)
    stage = Usd.Stage.Open(str(tmp_path / "candidate.usda"))
    assert stage.GetPrimAtPath("/Root/SourceObjectCandidate").IsValid()
    points = UsdGeom.Mesh(stage.GetPrimAtPath("/Root/SourceObjectCandidate/Parametric")).GetPointsAttr().Get()
    xs, ys, zs = zip(*points)
    assert (min(xs), max(xs), min(zs), max(zs)) == pytest.approx((0.7, 1.3, 0.0, 0.3), abs=1e-5)
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z and UsdGeom.GetStageMetersPerUnit(stage) == 1.0


def test_the_asset_is_a_rigid_body_with_a_collider_per_part_its_mass_and_its_bounds(tmp_path):
    from pxr import Usd, UsdPhysics
    chosen = parametric.design("open_bin", [0.5, 0.3, 0.25])
    receipt = parametric.author_asset(chosen, physics_bounds=BOUNDS, output_root=tmp_path / "a")
    stage = Usd.Stage.Open(str(tmp_path / "a" / "parametric_object.usda"))
    asset = stage.GetPrimAtPath("/Asset")
    assert asset.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.MassAPI(asset).GetMassAttr().Get() == pytest.approx(4)
    colliders = [prim for prim in stage.GetPrimAtPath("/Asset/Geometry/Collision").GetChildren()
                 if prim.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 5 == len(receipt["collision_parts"])
    assert receipt["bounds_m"] == {"minimum": [-.25, -.15, -.125], "maximum": [.25, .15, .125]}
    assert (receipt["source"], receipt["authoring_cost_usd"], receipt["model_called"]) == ("parametric", 0.0, False)
    assert receipt["physical_measurement"] is False and "not a measurement" in receipt["claim"]
    # Deterministic: the same design writes the same bytes.
    again = parametric.author_asset(chosen, physics_bounds=BOUNDS, output_root=tmp_path / "b")
    assert again["asset"]["sha256"] == receipt["asset"]["sha256"] and again["digest"] == receipt["digest"]
    with pytest.raises(ValueError, match="physics_invalid"):
        parametric.author_asset(chosen, physics_bounds={**BOUNDS, "mass_kg": [0, 1]}, output_root=tmp_path / "c")


def test_variants_stay_inside_the_plans_ranges_repeat_for_the_job_and_vary_nothing_unstated():
    chosen = parametric.design("carton", [0.4, 0.3, 0.2])
    row = {"size_m": {"low": 0.35, "high": 0.45, "basis": "estimated", "from": "x"},
           "mass_kg": {"low": 1.0, "high": 2.0, "basis": "site_words", "quote": "1 to 2 kg"}}
    value = parametric.variants(chosen, row, seed="sha256:job", count=8)
    assert value == parametric.variants(chosen, row, seed="sha256:job", count=8)
    assert value != parametric.variants(chosen, row, seed="sha256:other", count=8)
    assert value["varied"] == ["size_m", "mass_kg"] and value["label"] == "simulated_variation"
    assert value["physical_measurement"] is False
    for variant in value["variants"]:
        assert variant["label"] == "simulated_variation"
        dims = variant["design"]["dimensions_m"]
        assert 0.35 - 1e-6 <= max(dims) <= 0.45 + 1e-6 and 1.0 <= variant["mass_kg"] <= 2.0
        assert dims[0] / dims[2] == pytest.approx(2.0, rel=1e-4)  # every side by the same factor
    plain = parametric.variants(chosen, {"size_m": None, "mass_kg": None}, seed="sha256:job", count=3)
    assert plain["varied"] == [] and all(v["design"] == chosen and "mass_kg" not in v for v in plain["variants"])
    for count in (0, 33, True):
        with pytest.raises(ValueError, match="variants_invalid"):
            parametric.variants(chosen, row, seed="sha256:job", count=count)


def _authoring_stage(tmp_path, chosen, *, center):
    from blueprint_pipeline import task_evaluation_scene_configuration_submission_records as records
    from blueprint_pipeline.website_synthetic_subject import AUTHORING_ADAPTER, AUTHORING_METHOD
    x, y, z = chosen["dimensions_m"]
    configuration = records.stage_three_configuration(scene_id="site-1",
        replacement_identity={"id": "website-subject-tote", "version": "v1"}, source_instance_id="tote",
        authoring_target="tote", source_min=[center[0] - x / 2, center[1] - y / 2, center[2] - z / 2],
        source_max=[center[0] + x / 2, center[1] + y / 2, center[2] + z / 2], dimension_tolerance=0.1,
        physics_bounds={key + "_bounds": value for key, value in BOUNDS.items()})
    configuration.pop("authoring_backend")
    configuration.update(authoring_method=AUTHORING_METHOD, parametric_design=dict(chosen), authoring_source="parametric",
                         authoring_cost_usd=0.0)
    stage = {**records.stage_sequence()[2], "adapter": {"id": AUTHORING_ADAPTER, "version": "v1"},
             "execution_class": "no_spend"}
    return configuration, stage


def test_construction_authors_the_shape_as_supplied_geometry_within_its_estimated_bounds(tmp_path):
    from blueprint_pipeline.task_evaluation_completed_scene_adapters import execute_provided_mesh_authoring
    chosen = parametric.design("tote", [0.6, 0.4, 0.3])
    center = [1.0, 2.0, 0.15]
    configuration, stage = _authoring_stage(tmp_path, chosen, center=center)
    candidate = tmp_path / "source_object_candidate.parametric.usda"
    parametric.write_candidate(chosen, center_xyz_m=center, path=candidate)
    data = candidate.read_bytes()
    dependency = {"output_artifacts": [{"role": "source_object_candidate_mesh", "path": str(candidate),
                                        "digest": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}]}
    configuration_path = tmp_path / "stage-3.json"
    configuration_path.write_text(json.dumps(configuration))
    result = execute_provided_mesh_authoring(envelope={}, stage=stage, configuration=configuration,
        configuration_path=configuration_path, dependency_results=(dependency,), output_root=tmp_path / "out")
    receipt = json.loads((tmp_path / "out" / "replacement_authoring_receipt.v1.json").read_text())
    assert result["status"] == "completed" and receipt["model_called"] is False
    assert receipt["source_candidate_claim"] == "parametric_shape_candidate_geometry"
    completion = receipt["candidate_physics_completion"]
    assert completion["metric_envelope_validation"]["status"] == "within_preregistered_metric_envelope"
    assert completion["collision_dimensions_m"] == pytest.approx([0.6, 0.4, 0.3], abs=0.002)


def test_the_parametric_stage_has_no_provider_request_and_only_its_own_form_passes(tmp_path):
    from blueprint_pipeline.website_native_background import PREFIX
    from blueprint_pipeline.website_native_inputs import preflight_website_authoring_request
    from blueprint_pipeline.website_synthetic_subject import AUTHORING_ADAPTER, AUTHORING_METHOD
    candidate = tmp_path / "source_object_candidate.parametric.usda"
    parametric.write_candidate(parametric.design("carton", [.3, .2, .2]), center_xyz_m=[0, 0, .1], path=candidate)
    data = candidate.read_bytes()
    row = {"contract_path": PREFIX + ".candidate", "materialized_path": str(candidate), "size_bytes": len(data),
           "digest": "sha256:" + hashlib.sha256(data).hexdigest(), "full_byte_service_account_readback_passed": True}
    third = {"stage_id": "stage-3", "adapter": {"id": AUTHORING_ADAPTER, "version": "v1"}, "execution_class": "no_spend"}
    envelope = {"recipe": {"stage_sequence": [{}, {}, third]}, "materialized_references": [row]}
    shaped = {"authoring_method": AUTHORING_METHOD, "source_observation_kind": "website_capture_frames",
              "parametric_design": parametric.design("carton", [.3, .2, .2])}
    assert preflight_website_authoring_request(envelope=envelope, configurations={"stage-3": shaped}) is None
    # Captured with no shape, a shape on a designed object, or a backend: refused.
    for value in ({key: v for key, v in shaped.items() if key != "parametric_design"},
                  {**shaped, "source_observation_kind": "synthetic_design"}, {**shaped, "authoring_backend": "artisan_v1"}):
        with pytest.raises(ValueError, match="designed_subject_authoring_invalid"):
            preflight_website_authoring_request(envelope=envelope, configurations={"stage-3": value})

