"""Explicit result adaptation at existing static/native qualification boundaries."""
from __future__ import annotations

from pathlib import Path
import re
from .decision_evidence_contracts import canonical_digest
from .task_object_authoring_contracts import require, verified_file, read_json, file_record, write_json, verify_construction_request
from .task_object_artisan_exports import adapt_usd, validate_final_usd, validated_sdk_exports

SDK_CANDIDATE_SCHEMA = "task_object_artisan_sdk_candidate.v1"


def package_validated_sdk_candidate(*, request_path, manifest_path, model_path, blueprint_path,
                                    output_root, sdk_source_commit, controller, program_dependencies=(),
                                    annotations_path=None):
    """Beta scene preparation / ADP-040 day 28: reuse a reviewed SDK candidate.

    This only packages retained bytes. It cannot launch generation, register an
    upstream asset, or grant native physics and scene-placement qualification.
    Its ordinary stage-3 candidate can be adopted by the existing stage driver.
    """
    require(isinstance(sdk_source_commit, str) and re.fullmatch(r"[0-9a-f]{40}", sdk_source_commit)
            and isinstance(controller, str) and controller.strip(), "artisan_sdk_controller_identity_required")
    request_path, output_root = Path(request_path), Path(output_root)
    request = read_json(request_path)
    verify_construction_request(request)
    exports = validated_sdk_exports(manifest_path=Path(manifest_path), model_path=Path(model_path),
                                    blueprint_path=Path(blueprint_path))
    dependencies = [file_record(Path(path)) for path in program_dependencies]
    require(not output_root.exists(), "artisan_candidate_output_exists")
    output_root.mkdir(parents=True)
    asset = output_root / "replacement_asset.usdz"
    adaptation = adapt_usd(Path(exports["usd"]["path"]), asset, request,
                          annotations_path=Path(annotations_path) if annotations_path is not None else None)
    observed = adaptation["final_readback"]
    articulated = bool(observed["articulation_graph"]["joints"])
    graph = {"schema_version": "task_evaluation_articulated_replacement_graph.v1" if articulated
             else "task_evaluation_rigid_replacement_graph.v1", "backend": "artisan_v1",
             "asset_id": request["replacement_identity"]["id"],
             "asset_version": request["replacement_identity"]["version"],
             "articulation_graph": observed["articulation_graph"], "final_readback": observed,
             "construction_request": file_record(request_path), "physics_authority_granted": False,
             "single_rigid_candidate": not articulated}
    graph_path = output_root / "replacement_graph_spec.v1.json"
    write_json(graph_path, graph)
    receipt = {"schema_version": SDK_CANDIDATE_SCHEMA, "backend": "artisan_v1",
        "status": "authored_candidate_pending_qualification",
        "asset_kind": "articulated_assembly" if articulated else "rigid_body",
        "replacement_identity": request["replacement_identity"],
        "source_candidate_digest": request["source_candidate"]["digest"],
        "source_candidate_claim": "source_geometry_not_observed_truth_or_physics_authority",
        "source_object_identity": request["source_object_identity"], "request_digest": request["request_digest"],
        "construction_request": file_record(request_path),
        "upstream_runtime": {"controller": controller, "sdk_source_commit": sdk_source_commit,
                             "execution": "retained_validated_sdk_export", "packaging_provider_calls": 0},
        "program_dependencies": dependencies, "upstream_artifacts": exports, "export_adaptation": adaptation,
        "output_usd": file_record(asset), "graph_sha256": file_record(graph_path)["sha256"],
        "physics_authority_granted": False, "claim_ceiling": "development_only", "native_import_qualified": False,
        "scene_placement_qualified": False, "physical_equivalence_proven": False}
    receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
    receipt_path = output_root / "replacement_authoring_receipt.v1.json"
    write_json(receipt_path, receipt)
    verify_artisan_candidate(receipt=receipt, graph=graph, asset=asset)
    return {"asset": file_record(asset), "graph": file_record(graph_path), "receipt": file_record(receipt_path)}


def verify_artisan_candidate(*, receipt, graph, asset, configuration=None, source_candidate_digest=None):
    require(receipt.get("schema_version") in {"task_object_artisan_candidate.v1", SDK_CANDIDATE_SCHEMA}
            and receipt.get("backend") == graph.get("backend") == "artisan_v1"
            and receipt.get("status") == "authored_candidate_pending_qualification"
            and receipt.get("physics_authority_granted") is False
            and receipt.get("result_digest") == canonical_digest(receipt, digest_field="result_digest"), "artisan_candidate_receipt_invalid")
    require(file_record(asset)["sha256"] == receipt["output_usd"]["sha256"]
            and file_record(asset)["size_bytes"] == receipt["output_usd"]["size_bytes"], "artisan_final_asset_changed")
    request_path = verified_file(receipt["construction_request"])
    request = read_json(request_path)
    verify_construction_request(request)
    artifacts = receipt["upstream_artifacts"]
    for name in ("export_manifest", "candidate_manifest", "program", "blueprint", "physics_report"):
        verified_file(artifacts[name])
    for record in artifacts["exports"].values():
        verified_file(record)
    if receipt["schema_version"] == SDK_CANDIDATE_SCHEMA:
        runtime = receipt.get("upstream_runtime") or {}
        require("upstream_terminal" not in receipt
                and runtime.get("execution") == "retained_validated_sdk_export"
                and runtime.get("packaging_provider_calls") == 0
                and isinstance(runtime.get("controller"), str) and runtime["controller"].strip()
                and re.fullmatch(r"[0-9a-f]{40}", str(runtime.get("sdk_source_commit", ""))),
                "artisan_sdk_controller_identity_required")
        checked = validated_sdk_exports(manifest_path=verified_file(artifacts["candidate_manifest"]),
            model_path=verified_file(artifacts["program"]), blueprint_path=verified_file(artifacts["blueprint"]))
        require(checked == artifacts, "artisan_sdk_export_binding_changed")
        for record in receipt["program_dependencies"]:
            verified_file(record)
    require(request["request_digest"] == receipt["request_digest"]
            and graph["construction_request"] == receipt["construction_request"]
            and request["replacement_identity"] == receipt["replacement_identity"]
            and graph["asset_id"] == receipt["replacement_identity"]["id"]
            and graph["asset_version"] == receipt["replacement_identity"]["version"], "artisan_candidate_source_identity_changed")
    if configuration is not None:
        require(request["configuration"] == configuration, "artisan_candidate_configuration_changed")
    if source_candidate_digest is not None:
        require(request["source_candidate"]["digest"] == receipt["source_candidate_digest"] == source_candidate_digest,
                "artisan_candidate_source_geometry_changed")
    observed = validate_final_usd(asset, request)
    require(observed == graph["final_readback"] == receipt["export_adaptation"]["final_readback"]
            and observed["articulation_graph"] == graph["articulation_graph"], "artisan_candidate_graph_or_final_readback_changed")
    return request, observed


def qualify_artisan_asset_static(*, asset_path, graph_spec, authoring_receipt, replacement_identity, output_path):
    asset = Path(asset_path)
    request, observed = verify_artisan_candidate(receipt=authoring_receipt, graph=graph_spec, asset=asset)
    require(replacement_identity == request["replacement_identity"], "artisan_static_identity_changed")
    articulated = authoring_receipt["asset_kind"] == "articulated_assembly"
    from .task_evaluation_scene_configuration_submission_records import ARTISAN_ARTICULATED_STATIC_CHECKS, STATIC_CHECKS
    if articulated:
        structure = {**observed, "collision_bounds_asset_frame_closed_m": observed["bounds_m"],
                     "collision_dimensions_m": observed["dimensions_m"],
                     "joint_prim_paths": sorted(observed["joint_prim_paths"].values())}
        # Preserve the graph and exact USD coupling bytes; the current native
        # settling adapter cannot certify mimic motion. Refuse before stage 5.
        require(not any(j["role"] == "dependent" for j in observed["articulation_graph"]["joints"]),
                "artisan_coupled_joint_requires_native_adapter_support")
        handles = observed["handle_prim_paths"]
        require(handles, "artisan_static_task_handle_missing")
        from pxr import Usd, UsdGeom
        from .task_object_artisan_exports import _bounds
        stage = Usd.Stage.Open(str(asset))
        child_id = observed["task_joint"]["child_link_id"]
        child = stage.GetPrimAtPath(observed["links"][child_id]["prim_path"])
        handle_bounds = _bounds([stage.GetPrimAtPath(path) for path in handles],
            cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]),
            frame=UsdGeom.XformCache().GetLocalToWorldTransform(child).GetInverse())
        contact = {"handle_prim_paths": handles,
                   "contact_link_id": child_id, "handle_bounds_link_frame_m": handle_bounds,
                   "provenance": "generated_geometry_matched_to_retained_task_handle_observations",
                   "contact_point_link_m": request["configuration"].get("task_contact_point_link_m")
                        or observed["handle_grasp_point_link_m"],
                   "contact_point_authority": "retained_source_constraint" if request["configuration"].get("task_contact_point_link_m")
                        else "generated_candidate_handle_bounds_center"}
        require(contact["contact_point_link_m"] is not None, "artisan_task_contact_point_required_for_native_import")
        extra = {"asset_kind": "articulated_assembly", "articulation_graph": observed["articulation_graph"],
                 "link_prim_paths": {name: row["prim_path"] for name, row in observed["links"].items()},
                 "task_joint": observed["task_joint"], "task_contact": contact,
                 "backend": "artisan_v1"}
        readback = observed.get("required_motion_readback")
        if readback and readback.get("motions"):
            from .articulated_required_motion_probe import build_required_motion_probe_plan
            extra["required_motion_probe_plan"] = build_required_motion_probe_plan(
                readback=readback, graph=observed["articulation_graph"],
                requirements=request["construction_constraints"].get("required_motions", []),
                source_frames=request["source_frame_manifest"].get("frames", []))
    else:
        from .task_evaluation_scene_configuration_static_qualification import _usd_findings
        required = request["configuration"]["required_output"]
        bounds = {name: required[name + "_bounds"] for name in ("mass_kg", "static_friction", "dynamic_friction", "restitution")}
        findings, structure = _usd_findings(asset, physics_bounds=bounds)
        require(not findings, "artisan_existing_rigid_static_validator_refused:" + ";".join(findings))
        extra = {"backend": "artisan_v1"}
    result = {"schema_version": "task_evaluation_articulated_replacement_static_qualification.v1" if articulated
              else "task_evaluation_rigid_replacement_static_qualification.v1",
        "status": "authored_structure_statically_qualified", "replacement_identity": replacement_identity,
        "replacement_usd": file_record(asset), "checks": dict(ARTISAN_ARTICULATED_STATIC_CHECKS if articulated else STATIC_CHECKS),
        "observed_structure": structure, **extra, "authored_structure_statically_qualified": True,
        "structural_findings": [], "source_constraints": observed,
        "upstream_candidate_id": authoring_receipt["upstream_artifacts"]["manifest"]["candidate_id"],
        "authoring_execution": (authoring_receipt.get("upstream_runtime") or {}).get("execution", "artisan_cli"),
        "claim_boundary": {"native_simulator_import_qualified": False, "physical_equivalence_proven": False,
                           "generated_geometry_is_observed_truth": False, "joint_travel_is_measured": False}}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    require(not Path(output_path).exists(), "artisan_static_qualification_output_exists")
    write_json(Path(output_path), result)
    return result
