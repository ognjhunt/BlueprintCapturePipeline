"""Installed source-mesh construction adapters; no reconstruction or model call."""
from __future__ import annotations

import json

from .decision_evidence_contracts import canonical_digest, canonical_json


def mesh_appearance_configuration_refusal(configuration, envelope):
    rows = [row for row in envelope.get("materialized_references", [])
            if row.get("contract_path") == "scene.geometry.collision"]
    first = ((envelope.get("recipe") or {}).get("stage_sequence") or [{}])[0]
    if (configuration.get("schema_version") != "task_evaluation_provided_mesh_appearance_excision.v1"
            or first.get("adapter", {}).get("id") != "provided_mesh_appearance_excision"
            or first.get("execution_class") != "no_spend"
            or envelope.get("request", {}).get("scene", {}).get("appearance", {}).get("kind") != "other_observed"
            or configuration.get("source_origin") != "owner_provided_completed_asset"
            or configuration.get("source_bytes_unchanged_required") is not True
            or configuration.get("unobserved_surfaces_recovered") is not False
            or configuration.get("physical_truth_claimed") is not False
            or configuration.get("generated_appearance") is not False
            or len(rows) != 1
            or configuration.get("collision_source_digest") != rows[0].get("digest")
            or not str(configuration.get("exact_target_prim", "")).startswith("/Root/")):
        return "provided_mesh_source_binding"
    return None


def _visual_inventory(stage, excluded):
    """Compare authored visual data, including points/materials, not only bounds."""
    return {str(prim.GetPath()): {
        "type": prim.GetTypeName(),
        "attributes": {str(attr.GetName()): (str(attr.Get()), [str(p) for p in attr.GetConnections()])
                       for attr in prim.GetAttributes()
                       if not str(attr.GetName()).startswith(("physics:", "physx"))},
        "relationships": {str(rel.GetName()): [str(p) for p in rel.GetTargets()]
                          for rel in prim.GetRelationships()
                          if not str(rel.GetName()).startswith(("physics:", "physx"))},
    } for prim in stage.Traverse()
        if str(prim.GetPath()) != excluded and not str(prim.GetPath()).startswith(excluded + "/")}


def execute_provided_mesh_appearance(*, envelope, stage, configuration, configuration_path,
                                    dependency_results, output_root, provider_runtime_artifacts=()):
    from pxr import Usd, UsdPhysics
    from .task_evaluation_scene_configuration_builtin_adapters import _materialized_reference, _stage_result, _sha256_and_size
    from .source_collider_subtree_removal import remove_source_collider_subtree

    if mesh_appearance_configuration_refusal(configuration, envelope) is not None or dependency_results:
        raise ValueError("provided_mesh_appearance_configuration_invalid")
    row, source = _materialized_reference(envelope, contract_path="scene.geometry.collision")
    target = configuration["exact_target_prim"]
    original = Usd.Stage.Open(str(source))
    visual_before = _visual_inventory(original, target)
    appearance = output_root / "appearance_without_source_object.usda"
    excision = remove_source_collider_subtree(source_usd_path=source, target_prim_path=target,
        output_usda_path=appearance, expected_source_sha256=row["digest"],
        removal_id=envelope["recipe"]["subject_identity"]["id"])
    reopened = Usd.Stage.Open(str(appearance))
    for prim in reopened.Traverse():
        for api in (UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
        for name in prim.GetPropertyNames():
            if str(name).startswith(("physics:", "physx")):
                prim.RemoveProperty(name)
    reopened.GetRootLayer().Save()
    if _visual_inventory(reopened, target) != visual_before or _sha256_and_size(source)[0] != row["digest"]:
        raise ValueError("provided_mesh_appearance_visual_data_changed")
    receipt = {"schema_version": "task_evaluation_provided_mesh_appearance_excision_result.v1",
        "status": "exact_source_visual_subtree_removed", "source_digest": row["digest"],
        "removed_prim_path": target, "non_target_visual_inventory_unchanged": True,
        "source_bytes_unchanged": True, "generated_appearance": False,
        "unobserved_surfaces_recovered": False, "physical_truth_claimed": False,
        "source_excision": excision}
    receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
    path = output_root / "appearance_excision.v1.json"
    path.write_text(canonical_json(receipt) + "\n")
    return _stage_result(stage=stage, configuration_path=configuration_path, output_artifacts=[
        {"role": role, "path": str(artifact), "digest": _sha256_and_size(artifact)[0],
         "size_bytes": artifact.stat().st_size}
        for role, artifact in [("configured_appearance_without_source_object", appearance),
                               ("appearance_removal_receipt", path)]])


def execute_provided_mesh_authoring(*, envelope, stage, configuration, configuration_path,
                                   dependency_results, output_root, provider_runtime_artifacts=()):
    """Preserve the exact source shape; author bounded simulation priors only."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    from .task_evaluation_scene_configuration_builtin_adapters import _dependency_artifact, _stage_result, _sha256_and_size
    from .task_evaluation_scene_configuration_content_agents_driver import (
        _normalize_candidate, _complete_candidate_physics, _physics_bounds,
        _metric_envelope_spec, _validate_metric_envelope_dimensions, _package_replacement_asset,
    )

    if (configuration.get("authoring_method") != "exact_provided_mesh_with_bounded_simulation_physics"
            or stage.get("adapter", {}).get("id") != "provided_mesh_rigid_authoring"
            or stage.get("execution_class") != "no_spend"):
        raise ValueError("provided_mesh_authoring_configuration_invalid")
    source_record, source = _dependency_artifact(dependency_results, role="source_object_candidate_mesh")
    usd = output_root / "provided_mesh_candidate.usda"
    normalization = _normalize_candidate(source, usd)
    authored = Usd.Stage.Open(str(usd))
    layer = authored.GetRootLayer()
    if not Sdf.CopySpec(layer, "/Asset/Geometry/Visual", layer, "/Asset/Geometry/Collision"):
        raise ValueError("provided_mesh_collider_copy_failed")
    UsdPhysics.RigidBodyAPI.Apply(authored.GetPrimAtPath("/Asset"))
    for prim in Usd.PrimRange(authored.GetPrimAtPath("/Asset/Geometry/Collision")):
        if prim.IsA(UsdGeom.Mesh):
            UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    layer.Save()
    bounds = _physics_bounds(configuration)
    completion = _complete_candidate_physics(usd, bounds=bounds)
    completion["metric_envelope_validation"] = _validate_metric_envelope_dimensions(
        envelope=_metric_envelope_spec(configuration), observed_dimensions=completion["collision_dimensions_m"])
    completion["completion_digest"] = canonical_digest(completion, digest_field="completion_digest")
    asset = output_root / "provided_mesh_candidate.usdz"
    _package_replacement_asset(usd, asset)
    identity = configuration["replacement_identity"]
    graph = {"schema_version": "task_evaluation_rigid_replacement_graph.v1",
        "asset_id": identity["id"], "asset_version": identity["version"],
        "articulation_graph": {"joints": []}, "single_rigid_candidate": True,
        "physics_bounds": bounds, "physics_authority_granted": False}
    receipt = {"schema_version": "task_evaluation_rigid_replacement_authoring_result.v1",
        "status": "authored_candidate_pending_qualification", "replacement_identity": identity,
        "source_candidate_digest": source_record["digest"], "source_candidate_claim": "owner_provided_geometry_candidate",
        "source_normalization": normalization, "authoring_method": configuration["authoring_method"],
        "model_called": False, "physics_authority_granted": False,
        "output_usd": {"sha256": _sha256_and_size(asset)[0], "size_bytes": asset.stat().st_size},
        "candidate_physics_completion": completion}
    receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
    artifacts = [("replacement_asset", asset)]
    for role, value in [("replacement_graph_spec", graph), ("replacement_authoring_receipt", receipt)]:
        path = output_root / (role + ".v1.json")
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        artifacts.append((role, path))
    return _stage_result(stage=stage, configuration_path=configuration_path, output_artifacts=[
        {"role": role, "path": str(path), "digest": _sha256_and_size(path)[0], "size_bytes": path.stat().st_size}
        for role, path in artifacts])
