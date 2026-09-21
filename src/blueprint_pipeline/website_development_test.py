"""Explicit object/component rehearsal on an authored surface, ADP-040/050 day 28.

This is never a reconstruction repair. The original preparation stays blocked;
only the separate, owner-authorized development environment enters construction.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .website_object_local_frame import REGISTRATION_REFUSALS
from .website_object_observations import _record, materialize_object_observations, REFERENCE_ROLE

KIND = "authored_surface_component_test"
ENV = "BLUEPRINT_WEBSITE_DEVELOPMENT_TEST_TASK_DIGESTS"
LABEL = "Development test on an authored surface; captured scene integration pending."


def enabled(context_digest: str) -> bool:
    """An exact task allowlist; never enable as an automatic error fallback."""
    values = json.loads(os.environ.get(ENV, "[]"))
    if not isinstance(values, list) or any(not isinstance(v, str) or not v.startswith("sha256:")
                                          or len(v) != 71 for v in values):
        raise ValueError("website_development_test_authorization_invalid")
    return context_digest in values


def environment(preparation):
    value = preparation.get("development_test")
    if value is None:
        if (preparation.get("intake_request", {}).get("task", {}).get("subject", {}).get("test_environment")
                or preparation.get("intake_request", {}).get("source", {}).get("binding_id", "").startswith("website-development-")):
            raise ValueError("website_development_test_binding_invalid")
        return None
    if (value.get("kind") != KIND or value.get("claim_scope") != "development_only"
            or value.get("captured_scene_evaluation_allowed") is not False
            or value.get("captured_scene_integration") != "pending"
            or value.get("label") != LABEL
            or value.get("source_task_context_digest") != preparation["binding"]["task_context_digest"]
            or preparation["intake_request"]["task"]["subject"].get("test_environment") != value):
        raise ValueError("website_development_test_binding_invalid")
    return value


def prepare_development_test(*, preparation, source_geometry, task_masks, output_root: Path):
    """No provider calls. Preserve source dimensions and use a separate test frame."""
    from pxr import Gf, Usd, UsdGeom
    import trimesh
    from .task_evaluation_completed_scene_geometry import normalize_completed_mesh
    from .task_evaluation_scene_intake import validate_request
    from .website_support_geometry import support_under

    if preparation.get("digest") != canonical_digest(preparation, digest_field="digest"):
        raise ValueError("website_development_test_source_changed")
    if not enabled(preparation["binding"]["task_context_digest"]):
        raise ValueError("website_development_test_not_authorized")
    # Other refusals (rights, budget, missing target evidence) cannot be bypassed.
    allowed = {"support_surface_not_found_under_subject", "task_destination_surface_contact_required",
               "task_distinct_destination_surface_binding_required"} | REGISTRATION_REFUSALS
    if preparation.get("registration", {}).get("schema_version") == "website_object_local_frame.v1":
        allowed.add("task_destination_pose_required")
    if (preparation.get("claim_ceiling") != "development_only"
            or set(preparation.get("blockers", [])) - allowed):
        raise ValueError("website_development_test_source_not_admitted")
    value = copy.deepcopy(preparation)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    low = np.asarray(preparation["subject"]["aabb_min_xyz"], dtype=float)
    high = np.asarray(preparation["subject"]["aabb_max_xyz"], dtype=float)
    dims = high - low
    if not np.isfinite(dims).all() or not (dims > 0).all():
        raise ValueError("website_development_test_dimensions_invalid")
    # Generous test surface, without shrinking the source object to fit a gripper.
    top = 0.75
    width, depth = max(1.2, 6 * dims[0]), max(0.8, 4 * dims[1])
    lower = np.array([-width / 2, -depth / 2, top - 0.05])
    upper = np.array([width / 2, depth / 2, top])
    start = np.array([-max(0.15, dims[0]), 0.0, top + dims[2] / 2])
    finish = np.array([max(0.15, dims[0]), 0.0, top])
    new_low, new_high = start - dims / 2, start + dims / 2
    translation = new_low - low
    surface = trimesh.creation.box(extents=upper - lower)
    surface.apply_translation((upper + lower) / 2)
    mesh = output_root / "authored_surface.glb"
    mesh.write_bytes(surface.export(file_type="glb"))
    normalized = normalize_completed_mesh(source=mesh, original_filename=mesh.name,
        coordinate_frame={"up_axis": "Z", "meters_per_unit": 1.0}, output_root=output_root / "collision")
    collision = output_root / "collision" / normalized["output"]["relative_path"]
    support = support_under(surface, new_low, new_high, up=2, meters_per_unit=1.0)
    if support is None:
        raise ValueError("website_development_test_surface_contact_invalid")
    appearance_path = output_root / "authored_surface.usdc"
    stage = Usd.Stage.CreateNew(str(appearance_path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Root").GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    cube = UsdGeom.Cube.Define(stage, "/Root/DevelopmentSurface")
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*((lower + upper) / 2)))
    cube.AddScaleOp().Set(Gf.Vec3d(*(upper - lower)))
    cube.CreateDisplayColorAttr([(0.55, 0.55, 0.55)])
    stage.GetRootLayer().Save()
    test = {"kind": KIND, "label": LABEL, "claim_scope": "development_only",
            "source_task_context_digest": preparation["binding"]["task_context_digest"],
            "source_preparation_digest": preparation["digest"],
            "captured_scene_integration": "pending", "captured_scene_evaluation_allowed": False,
            "source_scene_blockers": preparation["blockers"]}
    value.update(status="intake_ready", blockers=[], development_test=test, simulator_ready=False)
    value["binding"].update(appearance_digest=_record(appearance_path)["digest"],
        collision_mesh_digest=_record(mesh)["digest"], provider="blueprint_authored_test_surface", operation_id=None)
    value["binding"].pop("splat_digest", None)
    # Transform only the observed object patches into the fixture frame. This is
    # an authored placement, not a successful registration to the captured room.
    original_snap = np.eye(4)
    original_up = {"Y": 1, "-Y": 1, "Z": 2}[preparation["coordinate_frame"]["declared_up_axis"]]
    original_snap[original_up, 3] = preparation["compose_back"]["pose_world"]["support_snap_runtime_units"]
    transform = (np.asarray(preparation["coordinate_frame"]["runtime_to_simulator"]) @ original_snap
                 @ np.asarray(preparation["registration"]["source_to_runtime"]))
    transform[:3, 3] += translation
    value["registration"] = {"source_to_runtime": transform.tolist(),
        "basis": "authored_development_placement", "physical_registration_proven": False}
    value["coordinate_frame"] = {"declared_up_axis": "Z", "declared_meters_per_unit": 1.0,
        "physical_scale_measured": False, "task_coordinates": "Z_up_estimated_meters",
        "runtime_to_simulator": np.eye(4).tolist(), "scale_authority": "model_estimated",
        "placement_uncertainty_m": None}
    value["compose_back"] = {"pose_world": {"support_snap_runtime_units": 0.0},
                             "captured_scene_integration": "pending"}
    subject = value["subject"]
    subject.update(aabb_min_xyz=new_low.tolist(), aabb_max_xyz=new_high.tolist(), test_environment=test)
    value["support"] = support
    destination = {"relation": "on", "visible_label": "development test target on authored surface",
        "mode": "existing_support_surface", "position_world_m": finish.tolist(),
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    value["destination"] = destination
    request = value["intake_request"]
    request["source"] = {"kind": "mesh", "binding_id": "website-development-" + _record(mesh)["digest"][7:39],
                         "content_digest": _record(mesh)["digest"]}
    request["task"].update(task_id=request["task"]["task_id"] + "-development", subject=subject,
        support={"description": "authored development surface, not captured support",
                 "aabb_min_xyz": lower.tolist(), "aabb_max_xyz": upper.tolist()}, destination=destination)
    # Keep submission_id, owner and execution authority unchanged: one native
    # reservation, never a second budget for a diagnostic sibling.
    validate_request(request, now=preparation["intake_request"]["consent"]["accepted_at_epoch"])
    authoring = value["authoring_inputs"]
    config = authoring["configuration"]
    config["scene_id"] += "-development"
    for envelope in (config["metric_envelope"], authoring["metric_envelope"]):
        envelope.update(minimum_xyz_m=new_low.tolist(), maximum_xyz_m=new_high.tolist())
    config["construction_constraints"].update(destination=destination, test_environment=test)
    value["digest"] = canonical_digest(value, digest_field="digest")
    environment(value)
    write_json(output_root / "preparation.json", value)
    observations = materialize_object_observations(preparation=value, source_geometry=source_geometry,
        task_masks=task_masks, output_root=output_root / "object_observations")
    appearance = {"schema_version": "website_native_appearance.v1", "status": "native_appearance_authored",
        "binding": {"preparation_digest": value["digest"], "source_digest": _record(appearance_path)["digest"]},
        "artifact": _record(appearance_path), "appearance_removal_performed": False,
        "reconstruction_performed": False, "physical_measurement_proven": False,
        "renderer_qualified": False, "claim_ceiling": "development_only", "development_test": test}
    appearance["digest"] = canonical_digest(appearance, digest_field="digest")
    runtime = {"schema_version": "website_scene_runtime_inputs.v1", "status": "background_collision_prepared",
        "preparation_digest": value["digest"], "claim_ceiling": "development_only", "development_test": test,
        "collision": {**_record(collision), "normalization_path": str(output_root / "collision/mesh_normalization.v1.json"),
            "normalization_digest": normalized["normalization_digest"], "object_mapping": normalized["object_mapping"]},
        "appearance": {**appearance["artifact"], "status": appearance["status"], "receipt": appearance,
                       "renderer_qualified": False},
        "object_authoring": {**authoring, "configuration": observations["configuration"],
            "source_candidate": observations["candidate"], "observation_manifest": observations["manifest"]},
        "authoring_dependency_artifacts": [{"role": "source_object_candidate_mesh", **observations["candidate"]},
            {"role": REFERENCE_ROLE, **observations["manifest"]}],
        "subject": subject, "destination": destination,
        "coordinate_frame": {"up_axis": "Z", "unit": "estimated_meters", "physical_scale_measured": False},
        "appearance_removal_required": False, "collision_excision_required": False,
        "reconstruction_performed": False, "provider_mutation_performed": False,
        "simulator_ready": False, "physics_qualified": False}
    runtime["digest"] = canonical_digest(runtime, digest_field="digest")
    write_json(output_root / "runtime_inputs.json", runtime)
    return value, runtime
