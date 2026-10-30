"""Reuse the website's prepared background in native construction, ADP-030/day 28.

The subject was removed before reconstruction. Preparation may reconcile
collision inside a capture-reviewed enclosed fixture; the stage transports
those prepared bytes and supplies separate object observations to authoring.
It never claims to have excised an observed prim from the background.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .website_object_observations import REFERENCE_ROLE, _record, validate_observation_handoff

SCHEMA = "website_prepared_collision.v1"
ADAPTER_ID = "website_prepared_collision"
PREFIX = "scene.website_native_inputs"
APPEARANCE_ADAPTER_ID = "website_prepared_appearance"


def exclude_box_interior(triangles, *, minimum, maximum):
    """Subtract an open box, splitting crossing faces without losing outside area."""
    import numpy as np
    from .adp009d_native_microcheck_bundle import _clip_polygon_to_axis_plane, _clip_triangle_to_aabb

    triangles, low, high = (np.asarray(value, dtype=float) for value in (triangles, minimum, maximum))
    if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or low.shape != (3,)
            or high.shape != (3,) or not np.isfinite(triangles).all()
            or not np.isfinite(low).all() or not np.isfinite(high).all() or not np.all(low < high)):
        raise ValueError("website_fixture_void_geometry_invalid")
    outside = np.any(np.all(triangles <= low, axis=1) | np.all(triangles >= high, axis=1), axis=1)
    inside = np.all((triangles > low) & (triangles < high), axis=(1, 2))
    untouched = list(np.flatnonzero(outside))
    pieces, source_faces = [], []
    removed_area = float(np.linalg.norm(np.cross(triangles[inside, 1] - triangles[inside, 0],
        triangles[inside, 2] - triangles[inside, 0]), axis=1).sum() / 2)
    clipped = 0
    def area(rows):
        value = np.asarray(rows, dtype=float).reshape(-1, 3, 3)
        return float(np.linalg.norm(np.cross(value[:, 1] - value[:, 0], value[:, 2] - value[:, 0]), axis=1).sum() / 2)
    for index in np.flatnonzero(~outside & ~inside):
        triangle = tuple(map(tuple, triangles[index]))
        intersection = _clip_triangle_to_aabb(triangle, roi_min=tuple(low), roi_max=tuple(high))
        if not intersection:
            untouched.append(index)
            continue
        polygon, exterior = list(triangle), []
        for axis in range(3):
            for boundary, keep_greater in ((low[axis], True), (high[axis], False)):
                part = _clip_polygon_to_axis_plane(polygon, axis=axis,
                    boundary=float(boundary), keep_greater=not keep_greater)
                exterior.extend((part[0], part[i], part[i + 1]) for i in range(1, len(part) - 1))
                polygon = _clip_polygon_to_axis_plane(polygon, axis=axis,
                    boundary=float(boundary), keep_greater=keep_greater)
        exterior = [row for row in exterior if area([row]) > 1e-12]
        before, removed, after = area([triangle]), area(intersection), area(exterior)
        if not np.isclose(before, removed + after, rtol=1e-8, atol=1e-12):
            raise ValueError("website_fixture_void_outside_area_changed")
        removed_area += removed
        clipped += 1
        pieces.extend(exterior)
        source_faces.extend([int(index)] * len(exterior))
    return (np.asarray(sorted(untouched), dtype=np.int64),
            np.asarray(pieces, dtype=float).reshape(-1, 3, 3),
            {"input_faces": len(triangles), "untouched_faces": len(untouched),
             "fully_interior_faces": int(inside.sum()), "clipped_faces": clipped,
             "replacement_faces": len(pieces), "replacement_source_faces": source_faces,
             "excluded_area_m2": removed_area, "boundary_surfaces_preserved": True})


def reconcile_enclosed_fixture_collision(*, source: Path, preparation: Mapping[str, Any],
                                         output_root: Path) -> dict[str, Any] | None:
    """Remove generated collision intrusions only from a capture-reviewed enclosure.

    An open frame, movable object or unknown enclosure keeps the original
    background. The source volume and all results remain estimated candidates.
    """
    import math
    import numpy as np
    from .local_reconstruction_adapters import _sha256_file
    from .task_object_required_motions import assembly_mounting, bounded_assembly_pose

    configuration = preparation["authoring_inputs"]["configuration"]
    declaration = (configuration.get("construction_constraints") or {}).get("assembly_mounting")
    if declaration is None:
        return None
    frames = preparation["authoring_inputs"]["source_frames"]
    review = assembly_mounting(declaration,
        source_frame_ids=[row["frame_id"] for row in frames])
    if review is None or review.get("body_envelope_excludes_environment") is not True:
        return None
    if (preparation.get("digest") != canonical_digest(preparation, digest_field="digest")
            or preparation.get("claim_ceiling") != "development_only"
            or preparation["subject"].get("geometry_origin") != "removed_before_reconstruction"):
        raise ValueError("website_fixture_void_preparation_invalid")
    body = configuration["body_extent_m"]
    size = np.asarray([body[key] for key in ("depth", "width", "height")], dtype=float)
    normal = np.asarray(configuration["mechanism"]["estimated_front_normal_world"], dtype=float)
    if (not np.isfinite(size).all() or np.any(size <= 0) or normal.shape != (3,)
            or not np.isfinite(normal).all() or abs(normal[2]) > 1e-8
            or not np.isclose(np.linalg.norm(normal), 1, atol=1e-6)):
        raise ValueError("website_fixture_void_frame_invalid")
    low = np.asarray(preparation["subject"]["aabb_min_xyz"], dtype=float)
    high = np.asarray(preparation["subject"]["aabb_max_xyz"], dtype=float)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all():
        raise ValueError("website_fixture_void_body_bounds_inconsistent")
    center = (low + high) / 2
    center[2] = low[2]
    yaw = math.atan2(normal[1], normal[0])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.]])
    if not np.allclose(high - low, np.abs(rotation) @ size, rtol=0, atol=1e-5):
        raise ValueError("website_fixture_void_body_bounds_inconsistent")
    minimum, maximum = [-size[0] / 2, -size[1] / 2, 0.], [size[0] / 2, size[1] / 2, size[2]]
    refinement = None
    if "scene_instance_pose_world" in preparation:
        pose, refinement = bounded_assembly_pose(preparation["scene_instance_pose_world"],
            reference_pose={"position_world_m": center.tolist(),
                            "orientation_xyzw": [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]},
            minimum=minimum, maximum=maximum,
            uncertainty_m=preparation["coordinate_frame"].get("placement_uncertainty_m"))
        from scipy.spatial.transform import Rotation
        center = np.asarray(pose["position_world_m"])
        rotation = Rotation.from_quat(pose["orientation_xyzw"]).as_matrix()
    binding = {"source_sha256": _sha256_file(source), "preparation_digest": preparation["digest"],
               "review": review, "center_world_m": center.tolist(), "rotation_world_from_volume": rotation.tolist(),
               "minimum_volume_m": list(map(float, minimum)), "maximum_volume_m": list(map(float, maximum))}
    if refinement is not None:
        binding["scene_instance_pose_refinement"] = refinement
    target = output_root / canonical_digest(binding)[7:]
    receipt_path, output = target / "reconciliation.json", target / "collision.usdc"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("digest") != canonical_digest(receipt, digest_field="digest")
                or receipt.get("binding") != binding or receipt["output"]["digest"] != _sha256_file(output)):
            raise ValueError("website_fixture_void_cache_changed")
        return receipt
    from pxr import Usd, UsdGeom, UsdPhysics, Vt
    stage = Usd.Stage.Open(Usd.Stage.Open(str(source)).Flatten())
    if UsdGeom.GetStageUpAxis(stage) != "Z" or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1., atol=1e-10):
        raise ValueError("website_fixture_void_background_frame_invalid")
    if any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()):
        raise ValueError("website_fixture_void_background_not_static")
    cache, reports = UsdGeom.XformCache(), []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh = UsdGeom.Mesh(prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        if not np.all(counts == 3):
            raise ValueError("website_fixture_void_triangles_required")
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
        world = points @ matrix[:3, :3].T + matrix[:3, 3]
        local = (world - center) @ rotation
        kept, replacements, report = exclude_box_interior(local[faces], minimum=minimum, maximum=maximum)
        reports.append({"prim_path": str(prim.GetPath()), **report})
        if report["fully_interior_faces"] == 0 and report["clipped_faces"] == 0:
            continue
        extra_world = replacements.reshape(-1, 3) @ rotation.T + center
        inverse = np.linalg.inv(matrix)
        extra = extra_world @ inverse[:3, :3].T + inverse[:3, 3]
        new_points = np.concatenate([points, extra.astype(np.float32)])
        new_faces = np.concatenate([faces[kept], np.arange(len(extra)).reshape(-1, 3) + len(points)])
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(new_points))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(new_faces.reshape(-1).astype(np.int32)))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(new_faces), 3, dtype=np.int32)))
        source_faces = np.concatenate([kept, np.asarray(report["replacement_source_faces"], dtype=int)])
        for subset in UsdGeom.Subset.GetGeomSubsets(mesh):
            if subset.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
                raise ValueError("website_fixture_void_subset_type_unsupported")
            remapped = np.flatnonzero(np.isin(source_faces, np.asarray(subset.GetIndicesAttr().Get(), dtype=int)))
            subset.GetIndicesAttr().Set(Vt.IntArray.FromNumpy(remapped.astype(np.int32)))
        # Appearance uses the separate source splat. Per-face/vertex arrays
        # cannot silently refer to the old topology in a collision derivative.
        mesh.GetNormalsAttr().Clear()
        for var in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            if var.GetInterpolation() != UsdGeom.Tokens.constant:
                name = var.GetAttr().GetName()
                prim.RemoveProperty(name)
                prim.RemoveProperty(name + ":indices")
    if not reports:
        raise ValueError("website_fixture_void_collision_mesh_missing")
    target.mkdir(parents=True, exist_ok=False)
    stage.GetRootLayer().Export(str(output))
    if _sha256_file(source) != binding["source_sha256"]:
        raise ValueError("website_fixture_void_source_changed")
    receipt = {"schema_version": "website_enclosed_fixture_collision_reconciliation.v1", "binding": binding,
               "meshes": reports, "output": _record(output), "source_bytes_unchanged": _sha256_file(source) == binding["source_sha256"],
               "outside_volume_geometry_preserved": True, "appearance_modified": False,
               "native_physics_qualified": False, "physical_measurement_proven": False,
               "claim_ceiling": "development_only", "provider_mutation_performed": False}
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    write_json(receipt_path, receipt)
    return receipt


def derived_stage_three_configuration(*, runtime: Mapping[str, Any],
                                      preparation: Mapping[str, Any],
                                      observation_manifest_path: Path | None = None) -> dict[str, Any]:
    """Recompile only stage 3 from the retained exact-scene evidence on retry."""
    from .task_object_articulated_packaging import derived_website_cabinet_depth_hypothesis
    from .website_drawer_depth_prior import prior_for

    original = runtime["object_authoring"]["configuration"]
    config = dict(original)
    observation_record = runtime["object_authoring"]["observation_manifest"]
    prior = prior_for(scene_id=original.get("scene_id"),
                      subject_identity=original.get("replacement_identity"))
    if (prior is None or preparation.get("digest") != prior["preparation_digest"]
            or (preparation.get("development_test") or {}).get("kind") != "development_drawer_fixture"
            or (preparation.get("development_test") or {}).get("captured_scene_evaluation_allowed") is not False
            or observation_record.get("digest") != prior["observation_manifest_digest"]):
        return config
    manifest_path = observation_manifest_path or Path(observation_record["path"])
    if not manifest_path.is_file():
        return config
    checked = _record(manifest_path)
    if (checked["digest"] != observation_record["digest"]
            or checked["size_bytes"] != observation_record["size_bytes"]):
        return config
    observed = json.loads(manifest_path.read_text())
    if (observed.get("digest") != canonical_digest(observed, digest_field="digest")
            or observed.get("preparation_digest") != preparation["digest"]):
        return config
    original_frames = observed.get("frames") or []
    hashes = {row.get("image", {}).get("digest") for row in original_frames
              if row.get("image_basis") == "original_capture"}
    if len(original_frames) != len(hashes) or hashes != set(prior["original_frame_sha256s"]):
        return config
    frames = [{"role": "observed_source", "sha256": digest} for digest in sorted(hashes)]
    fraction = (preparation.get("intake_request", {}).get("task", {}).get("success") or {}).get(
        "minimum_opening_fraction_of_estimated_stroke")
    hypothesis = derived_website_cabinet_depth_hypothesis(original, frames, fraction)
    if hypothesis is None:
        return config  # The stage-3 no-spend plausibility gate still refuses the thin box.
    stroke = hypothesis["estimated_usable_stroke_m"]
    config["mechanism"] = {**original["mechanism"], "estimated_usable_stroke_m": stroke,
                           "joint_limits": [0.0, stroke],
                           "travel_authority": "development_only_bounded_cabinet_depth_prior"}
    mass = hypothesis["revised_part_mass_bounds_kg"]
    config["required_output"] = {**original["required_output"],
                                  "mass_kg_bounds": mass["carcass"],
                                  "task_part_mass_kg_bounds": mass["drawer"]}
    config["development_geometry_hypothesis"] = hypothesis
    return config


def collision_configuration_refusal(configuration: Mapping[str, Any], envelope: Mapping[str, Any]) -> str | None:
    if (configuration.get("schema_version") != SCHEMA
            or configuration.get("operation") != "reuse_background_without_excision"
            or configuration.get("claim_ceiling") != "development_only"):
        return "website_background_operation"
    for name, contract in (("runtime_inputs_digest", PREFIX + ".runtime_inputs"),
                           ("collision_source_digest", "scene.geometry.collision")):
        rows = [r for r in envelope.get("materialized_references", []) if r.get("contract_path") == contract]
        if len(rows) != 1 or rows[0].get("digest") != configuration.get(name):
            return name
    return None


def appearance_configuration_refusal(configuration: Mapping[str, Any], envelope: Mapping[str, Any]) -> str | None:
    if (configuration.get("schema_version") != "website_prepared_appearance.v1"
            or configuration.get("operation") != "reuse_background_without_removal"
            or configuration.get("claim_ceiling") != "development_only"):
        return "website_background_operation"
    for name, contract in (("runtime_inputs_digest", PREFIX + ".runtime_inputs"),
                           ("appearance_source_digest", PREFIX + ".appearance")):
        rows = [r for r in envelope.get("materialized_references", []) if r.get("contract_path") == contract]
        if len(rows) != 1 or rows[0].get("digest") != configuration.get(name):
            return name
    return None


def _runtime(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (value.get("schema_version") != "website_scene_runtime_inputs.v1"
            or value.get("digest") != canonical_digest(value, digest_field="digest")
            or value.get("claim_ceiling") != "development_only"
            or value.get("collision_excision_required") is not False
            or value.get("appearance_removal_required") is not False
            or value.get("subject", {}).get("geometry_origin") not in {"removed_before_reconstruction",
                                                                       "synthetic_design"}
            or value.get("coordinate_frame") != {"up_axis": "Z", "unit": "estimated_meters",
                                                 "physical_scale_measured": False}):
        raise ValueError("website_native_background_inputs_invalid")
    return value


def prepare_collision_stage(runtime_inputs_path: Path) -> dict[str, Any]:
    """List the immutable files the existing object-store stager must transport."""
    value = _runtime(runtime_inputs_path)
    authoring = value["object_authoring"]
    manifest = Path(authoring["observation_manifest"]["path"])
    observed, frames = validate_observation_handoff(manifest, configuration=authoring["configuration"])
    records = {PREFIX + ".runtime_inputs": _record(runtime_inputs_path),
               "scene.geometry.collision": value["collision"],
               PREFIX + ".observations": authoring["observation_manifest"],
               PREFIX + ".candidate": authoring["source_candidate"],
               **{PREFIX + f".frames.{i}": _record(frame) for i, frame in enumerate(frames)}}
    cad = (observed.get("candidate_geometry") or {}).get("cad_authoring")
    if cad:
        records[PREFIX + ".cad_evidence"] = _record(manifest.parent / cad["evidence"]["path"])
    references = []
    for contract, row in records.items():
        checked = _record(Path(row["path"]))
        if any(row[key] != checked[key] for key in ("digest", "size_bytes")):
            raise ValueError("website_native_background_artifact_changed")
        references.append({"contract_path": contract, **checked})
    return {"stage": {"stage_id": "stage-2", "capability": "collision_object_excision",
                       "execution_class": "no_spend", "adapter": {"id": ADAPTER_ID, "version": "v1"}},
            "configuration": {"schema_version": SCHEMA, "operation": "reuse_background_without_excision",
                              "claim_ceiling": "development_only",
                              "runtime_inputs_digest": _record(runtime_inputs_path)["digest"],
                              "collision_source_digest": value["collision"]["digest"]},
            "references": references}


def prepare_appearance_stage(runtime_inputs_path: Path) -> dict[str, Any]:
    value = _runtime(runtime_inputs_path)
    appearance = value["appearance"]
    if appearance.get("status") != "native_appearance_authored":
        raise ValueError("website_native_appearance_pending")
    source = _record(Path(appearance["path"]))
    if any(source[key] != appearance[key] for key in ("digest", "size_bytes")):
        raise ValueError("website_native_appearance_changed")
    runtime = _record(runtime_inputs_path)
    return {"stage": {"stage_id": "stage-1", "capability": "observed_appearance_object_removal",
                       "execution_class": "no_spend", "adapter": {"id": APPEARANCE_ADAPTER_ID, "version": "v1"}},
            "configuration": {"schema_version": "website_prepared_appearance.v1",
                              "operation": "reuse_background_without_removal", "claim_ceiling": "development_only",
                              "runtime_inputs_digest": runtime["digest"], "appearance_source_digest": source["digest"]},
            "references": [{"contract_path": PREFIX + ".runtime_inputs", **runtime},
                           {"contract_path": PREFIX + ".appearance", **source}]}


def prepare_construction_stages(*, runtime_inputs_path: Path, preparation_path: Path) -> dict[str, Any]:
    """Compile all six existing stages for this task, with no provider calls."""
    from . import task_evaluation_scene_configuration_submission_records as records
    runtime = _runtime(runtime_inputs_path)
    preparation = json.loads(preparation_path.read_text())
    if (preparation.get("digest") != canonical_digest(preparation, digest_field="digest")
            or preparation["digest"] != runtime["preparation_digest"]):
        raise ValueError("website_native_construction_preparation_changed")
    if not isinstance(preparation.get("support"), Mapping):
        # Compile flagged support_surface_not_found_under_subject: nothing to assemble the subject on.
        raise ValueError("website_native_construction_support_missing")
    first, second = prepare_appearance_stage(runtime_inputs_path), prepare_collision_stage(runtime_inputs_path)
    third = derived_stage_three_configuration(runtime=runtime, preparation=preparation)
    identity = third["replacement_identity"]
    scene_identity = {"id": third["scene_id"], "version": "v1"}
    subject = runtime["subject"]
    lower, upper = subject["aabb_min_xyz"], subject["aabb_max_xyz"]
    support = preparation["intake_request"]["task"]["support"]
    assembly = records.stage_six_configuration(scene_identity=scene_identity,
        support_plane={"sage_prim_path": "/Root", "publisher_instance_id": "website-support",
                       "top_z_m": support["aabb_max_xyz"][2], "bounds_min_xyz_m": support["aabb_min_xyz"],
                       "bounds_max_xyz_m": support["aabb_max_xyz"]},
        start_center=[(a + b) / 2 for a, b in zip(lower, upper, strict=True)], bottom_z=lower[2])
    assembly["appearance"] = {"source": "website_prepared_background", "reuse_from_stage": "stage-1",
                               "generated_region_label_required": True}
    assembly["collision"] = {"source": "website_prepared_background", "reuse_from_stage": "stage-2",
                              "support_prim": "/Root"}
    from .website_development_test import environment
    test = environment(preparation)
    assembly["support_plane"].update(authority="authored_development_surface" if test else "registered_estimated_capture_and_reconstruction",
        physical_scale_measured=False, source_face_indices=preparation["support"]["face_indices"])
    articulated = third.get("schema_version") == records.ARTICULATED_AUTHORING_SCHEMA_VERSION
    configurations = [first["configuration"], second["configuration"], third,
        records.stage_four_configuration(replacement_identity=identity,
            authoring_backend=third.get("authoring_backend"),
            dimension_tolerance=third["metric_envelope"]["maximum_dimension_relative_error"], articulated=articulated),
        records.stage_five_configuration(replacement_identity=identity, articulated=articulated), assembly]
    if articulated:
        assembly["replacement"].update(asset_kind="articulated_assembly",
            front_normal_world=third["mechanism"]["estimated_front_normal_world"],
            task_joint_reset="closed", swept_volume_must_stay_clear_of_scene_collision=True)
    stages = records.stage_sequence()
    for old, new in zip(stages[:2], (first["stage"], second["stage"]), strict=True):
        old.update(new)
    from .website_synthetic_subject import AUTHORING_ADAPTER, AUTHORING_METHOD
    if third.get("authoring_method") == AUTHORING_METHOD:
        # A designed subject's mesh is supplied: authored as given, no provider called.
        stages[2].update(adapter={"id": AUTHORING_ADAPTER, "version": "v1"}, execution_class="no_spend")
    references = {row["contract_path"]: row for row in first["references"] + second["references"]}
    return {"schema_version": "website_native_construction_inputs.v1", "scene_identity": scene_identity,
            "subject_identity": identity, "stage_sequence": stages, "configurations": configurations,
            "references": list(references.values()), "preparation": _record(preparation_path),
            "provider_mutation_performed": False, "claim_ceiling": "development_only"}


def construction_rights_admission(*, preparation: Mapping[str, Any], task_context: Mapping[str, Any],
                                 now: float) -> dict[str, Any]:
    """Carry recorded website consent into the existing worker disclosure gate."""
    from .task_evaluation_scene_intake import validate_request

    if (preparation.get("digest") != canonical_digest(preparation, digest_field="digest")
            or task_context.get("context_digest") != canonical_digest(task_context, digest_field="context_digest")
            or preparation["binding"]["task_context_digest"] != task_context["context_digest"]
            or task_context.get("capture_rights", {}).get("derived_scene_generation_allowed") is not True):
        raise ValueError("website_construction_rights_context_invalid")
    from .website_development_test import environment, enabled
    test = environment(preparation)
    if test and not enabled(task_context["context_digest"]):
        raise ValueError("website_development_test_not_authorized")
    if preparation.get("status") != "intake_ready" or preparation.get("blockers"):
        raise ValueError("website_construction_preparation_not_ready")
    request = validate_request(preparation["intake_request"], now=now)
    if task_context.get("confirmed") is not True:
        from .website_preparation_contracts import validate_preparation_request
        validate_preparation_request(request=request, task_context=task_context,
            authority=preparation.get("website_preparation_authority") or {}, now=now)
    consent = request["consent"]
    value = {"schema_version": "website_native_rights_admission.v1",
             # The authoring drivers read these four fields at the top level
             # (astra_driver.build_authoring_request); the website consent
             # admits exactly this: private provider processing of derived
             # views for internal development, no training, no redistribution.
             "status": "admitted_for_internal_development",
             "private_provider_processing_allowed": True,
             "provider_training_allowed": False,
             "public_redistribution_allowed": False,
             "preparation_digest": preparation["digest"],
             "task_context_digest": task_context["context_digest"],
             "capture_id": task_context["capture_id"], "scene_id": task_context["scene_id"],
             "owner": request["owner"], "consent": consent,
             **({"anthropic_provider_terms_reference": preparation["authoring_provider_terms_reference"]}
                if request["execution"]["allowed_providers"][-1:] == ["anthropic"] else {}),
             "execution_authority": request["execution"],
             "provider_disclosure": {"captured_frame_derivatives_allowed": True,
                 "prepared_background_allowed": True, "raw_capture_video_allowed": False,
                 "provider_training_allowed": False, "public_redistribution_allowed": False},
             "physical_measurement_proven": False, "provider_mutation_performed": False,
             "claim_ceiling": "development_only"}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


def execute_prepared_appearance(*, envelope, stage, configuration, configuration_path,
                                dependency_results, output_root, provider_runtime_artifacts=()):
    from .task_evaluation_scene_configuration_builtin_adapters import (
        _copy_artifact, _materialized_reference, _stage_result,
    )
    if (appearance_configuration_refusal(configuration, envelope) or dependency_results
            or stage.get("adapter", {}).get("id") != APPEARANCE_ADAPTER_ID
            or stage.get("execution_class") != "no_spend"):
        raise ValueError("website_prepared_appearance_configuration_invalid")
    _, runtime_path = _materialized_reference(envelope, contract_path=PREFIX + ".runtime_inputs")
    value = _runtime(runtime_path)
    appearance = value["appearance"]
    row, source = _materialized_reference(envelope, contract_path=PREFIX + ".appearance")
    receipt = appearance.get("receipt", {})
    if (appearance.get("status") != "native_appearance_authored"
            or receipt.get("digest") != canonical_digest(receipt, digest_field="digest")
            or receipt.get("binding", {}).get("preparation_digest") != value["preparation_digest"]
            or receipt.get("appearance_removal_performed") is not False
            or receipt.get("renderer_qualified") is not False
            or receipt.get("physical_measurement_proven") is not False
            or envelope.get("recipe", {}).get("subject_identity") != value["object_authoring"]["configuration"]["replacement_identity"]
            or any(row[key] != appearance[key] or row[key] != receipt["artifact"][key]
                   for key in ("digest", "size_bytes"))):
        raise ValueError("website_prepared_appearance_source_mismatch")
    output_root.mkdir(parents=True, exist_ok=True)
    appearance_artifact = _copy_artifact(source, output_root / "background_appearance.usdc")
    receipt_path = output_root / "appearance_authoring.json"
    write_json(receipt_path, receipt)
    return _stage_result(stage=stage, configuration_path=configuration_path, output_artifacts=[
        {"role": "configured_appearance_without_source_object", **appearance_artifact},
        {"role": "website_background_appearance_receipt", **_record(receipt_path)}])


def execute_prepared_collision(*, envelope, stage, configuration, configuration_path,
                               dependency_results, output_root, provider_runtime_artifacts=()):
    from .task_evaluation_scene_configuration_builtin_adapters import (
        _copy_artifact, _materialized_reference, _stage_result,
    )

    if (collision_configuration_refusal(configuration, envelope)
            or stage.get("adapter", {}).get("id") != ADAPTER_ID
            or stage.get("execution_class") != "no_spend"):
        raise ValueError("website_prepared_collision_configuration_invalid")
    _, inputs_path = _materialized_reference(envelope, contract_path=PREFIX + ".runtime_inputs")
    value = _runtime(inputs_path)
    authoring = value["object_authoring"]
    identity = envelope.get("recipe", {}).get("subject_identity")
    if identity != authoring["configuration"]["replacement_identity"]:
        raise ValueError("website_prepared_collision_subject_mismatch")

    def source(contract, expected):
        row, path = _materialized_reference(envelope, contract_path=contract)
        if any(row[key] != expected[key] for key in ("digest", "size_bytes")):
            raise ValueError("website_prepared_collision_source_mismatch")
        return path

    collision = source("scene.geometry.collision", value["collision"])
    manifest = source(PREFIX + ".observations", authoring["observation_manifest"])
    observed = json.loads(manifest.read_text())
    output_root.mkdir(parents=True, exist_ok=True)
    observed_root = output_root / "observations"
    observed_root.mkdir(exist_ok=True)
    # Retain portable manifest bytes. Resolve files through the admitted envelope,
    # never through the control-plane machine's original absolute paths.
    copy_rows = [(PREFIX + ".candidate", observed["candidate"])] + [
        (PREFIX + f".frames.{i}", row["image"]) for i, row in enumerate(observed["frames"])]
    cad = (observed.get("candidate_geometry") or {}).get("cad_authoring")
    if cad:
        copy_rows.append((PREFIX + ".cad_evidence", cad["evidence"]))
    for contract, record in copy_rows:
        name = Path(record["path"])
        if name.is_absolute() or len(name.parts) != 1 or name.name in {"", ".", ".."}:
            raise ValueError("website_prepared_collision_observation_path_invalid")
        _copy_artifact(source(contract, record), observed_root / name)
    observation_artifact = _copy_artifact(manifest, observed_root / "observations.json")
    validate_observation_handoff(Path(observation_artifact["path"]), configuration=authoring["configuration"])
    candidate = _record(observed_root / observed["candidate"]["path"])
    if any(candidate[key] != authoring["source_candidate"][key] for key in ("digest", "size_bytes")):
        raise ValueError("website_prepared_collision_candidate_mismatch")
    # The transported path has no extension. Keep the recorded layer's own
    # format name (text .usda before, crate .usdc now and when reconciled):
    # OpenUSD refuses crate bytes under a .usda name.
    background = _copy_artifact(
        collision, output_root / ("background_collision" + Path(value["collision"]["path"]).suffix))
    receipt = {"schema_version": "website_prepared_background_result.v1", "status": "prepared_background_reused",
               "runtime_inputs_digest": configuration["runtime_inputs_digest"],
               "preparation_digest": value["preparation_digest"], "collision_digest": background["digest"],
               "background_bytes_unchanged": True, "source_prim_excision_performed": False,
               "source_object_candidate_separate": True, "claim_ceiling": "development_only",
               "physical_measurement_proven": False}
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    receipt_path = output_root / "background_reuse.json"
    write_json(receipt_path, receipt)
    return _stage_result(stage=stage, configuration_path=configuration_path, output_artifacts=[
        {"role": "configured_collision_without_source_object", **background},
        {"role": "source_object_candidate_mesh", **candidate},
        {"role": REFERENCE_ROLE, **observation_artifact},
        *([{"role": "source_cad_authoring_evidence", **_record(observed_root / cad["evidence"]["path"])}]
          if cad else []),
        {"role": "website_background_reuse_receipt", **_record(receipt_path)}])
