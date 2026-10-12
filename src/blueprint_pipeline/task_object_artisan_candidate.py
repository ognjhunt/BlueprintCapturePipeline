"""Explicit result adaptation at existing static/native qualification boundaries."""
from __future__ import annotations

from pathlib import Path
from .decision_evidence_contracts import canonical_digest
from .task_object_authoring_contracts import require, verified_file, read_json, file_record, write_json, verify_construction_request
from .task_object_artisan_exports import validate_final_usd


def verify_artisan_candidate(*, receipt, graph, asset, configuration=None, source_candidate_digest=None):
    require(receipt.get("schema_version") == "task_object_artisan_candidate.v1"
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
        contact = {"handle_prim_paths": handles,
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
        "upstream_candidate_id": authoring_receipt["upstream_terminal"]["candidate_id"],
        "claim_boundary": {"native_simulator_import_qualified": False, "physical_equivalence_proven": False,
                           "generated_geometry_is_observed_truth": False, "joint_travel_is_measured": False}}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    require(not Path(output_path).exists(), "artisan_static_qualification_output_exists")
    write_json(Path(output_path), result)
    return result
