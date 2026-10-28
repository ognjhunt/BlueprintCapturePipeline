"""Retain source-bound family authoring requirements without inventing facts.

Beta: capture-to-native preparation. This is the resumable precursor to a
family producer, not that producer and not an execution-readiness decision.
It reuses existing task corrections and retained source artifacts; no model,
provider, asset authoring, approval, or runtime effect is performed here.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .launch_immutable_input_writer import write_exclusive_private_bytes
from .task_family_scoring import FAMILY_CRITERIA, FAMILY_OBSERVABLES, canonical_task_family

SCHEMA = "website_task_family_authoring_requirements.v1"
PRODUCER = "task_family_authoring_requirements.v1"

# Quantity labels are questions, never defaults or inferred measurements.
_QUESTIONS = {
    "minimum_transferred_volume_ml": ("ml", "What minimum volume must reach the receiving container?"),
    "maximum_spilled_volume_ml": ("ml", "What maximum spilled volume is acceptable?"),
    "maximum_volume_balance_error_ml": ("ml", "What volume-balance error is acceptable?"),
    "maximum_final_in_flight_volume_ml": ("ml", "How much liquid may remain in flight at completion?"),
    "minimum_cleaned_fraction": ("fraction", "What fraction of the observed dirty surface must be cleaned?"),
    "maximum_remaining_dirty_area_m2": ("m2", "What remaining dirty surface area is acceptable?"),
    "minimum_insertion_depth_m": ("m", "What insertion depth counts as complete?"),
    "maximum_lateral_error_m": ("m", "What lateral insertion error is acceptable?"),
    "maximum_alignment_error_rad": ("rad", "What insertion alignment error is acceptable?"),
    "maximum_contact_force_n": ("N", "What maximum contact force is acceptable for this operation?"),
    "minimum_support_margin_m": ("m", "How far inside the support boundary must the object's centre of mass remain?"),
    "maximum_linear_speed_m_s": ("m/s", "What maximum residual linear speed counts as settled?"),
    "maximum_angular_speed_rad_s": ("rad/s", "What maximum residual angular speed counts as settled?"),
    "minimum_button_travel_m": ("m", "What minimum button travel counts as pressed?"),
    "maximum_fold_keypoint_error_m": ("m", "What fold-position error is acceptable?"),
    "minimum_fold_overlap_fraction": ("fraction", "What minimum overlap is required for the fold?"),
    "require_release": ("boolean", "Must the robot release the cloth before completion?"),
    "minimum_hold_seconds": ("s", "For how long must the completed state remain stable?"),
}
_GEOMETRY = {
    "pouring": ("container_interiors", "liquid_material_parameters", "transfer_workspace"),
    "wiping": ("observed_residue_cells", "surface_material_parameters"),
    "insertion": ("peg_and_socket_geometry", "contact_material_parameters"),
    "stacking": ("object_geometry_and_mass", "support_polygon", "contact_material_parameters"),
    "button_press": ("button_joint_axis_and_rest", "activation_sensor_and_force_model"),
    "cloth_folding": ("cloth_mesh_topology", "cloth_material_parameters", "fold_plane_and_target_keypoints"),
}


def _operations(*, family: str, task_context: Mapping[str, Any],
                task_semantics: Mapping[str, Any] | None, source_video_digest: str | None,
                contract: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if contract is not None:
        return [{"stage_id": row["stage_id"], "task_family": row["task_family"],
                 "stage_identity_basis": "retained_success_contract", "order": index}
                for index, row in enumerate(contract["stages"], start=1)]
    if task_semantics is not None:
        from .website_task_semantics import validate_task_semantics
        value = validate_task_semantics(task_semantics, task_context=task_context,
                                        source_video_digest=source_video_digest)
        reading = ((value.get("operation_coverage_clarification") or {}).get("operation_coverage")
                   or value["semantics"])
        complete = reading.get("operation_coverage_complete", reading.get("covers_entire_job"))
        unresolved = reading.get("unresolved_operations", reading.get("unresolved_aspects"))
        if complete and not unresolved:
            steps = reading["steps"]
            kinds = [canonical_task_family(step["operation"]) for step in steps]
            if all(kind in FAMILY_CRITERIA for kind in kinds):
                return [{"stage_id": "operation-" + str(index).zfill(3) + "-" + canonical_digest(step)[7:15],
                         "task_family": kind, "order": index,
                         "stage_identity_basis": "derived_source_bound_operation", "source_operation": dict(step)}
                        for index, (step, kind) in enumerate(zip(steps, kinds), start=1)]
    if family != "multi_step":
        return [{"stage_id": "operation-001", "task_family": family, "order": 1,
                 "stage_identity_basis": "single_declared_task_family"}]
    return []


def retain_family_authoring_requirements(
    *, family: str, task_context: Mapping[str, Any], task_masks: Mapping[str, Any],
    source_geometry: Mapping[str, Any], source_bindings: Mapping[str, Any],
    output_root: Path, task_semantics: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None, native: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write/adopt an immutable requirement generation; never advance a stage.

    Supplied contracts are identified as retained internal inputs. Their
    presence does not assert new owner confirmation or measurement authority.
    No generic prose success definition is converted to numeric thresholds.
    """
    for value, field in ((task_context, "context_digest"), (task_masks, "digest"), (source_geometry, "digest")):
        if value.get(field) != canonical_digest(value, digest_field=field):
            raise ValueError("website_task_family_authoring_source_changed")
    expected = {"task_context_digest": task_context["context_digest"],
                "source_geometry_digest": source_geometry["digest"], "task_masks_digest": task_masks["digest"]}
    if any(source_bindings.get(key) != digest for key, digest in expected.items()):
        raise ValueError("website_task_family_authoring_binding_mismatch")
    source_video = (source_geometry.get("binding") or {}).get("source_video_digest")
    operations = _operations(family=family, task_context=task_context, task_semantics=task_semantics,
                             source_video_digest=source_video, contract=contract)
    requirements: list[dict[str, Any]] = []

    def missing(key: str, category: str, owner: str, question: str, *, stage_id: str | None = None,
                quantity: str | None = None, unit: str | None = None) -> None:
        requirements.append({"requirement_id": key, "category": category, "owner": owner,
            "status": "required", "question": question, "stage_id": stage_id,
            "quantity": quantity, "unit": unit, "authority_created": False})

    if not operations:
        missing("ordered_operations", "task_fact", "task_owner",
                "What are the ordered robot operations and the target of each operation?")
    for operation in operations:
        stage_id, kind = operation["stage_id"], operation["task_family"]
        if contract is None:
            for quantity in (*FAMILY_CRITERIA[kind], "minimum_hold_seconds"):
                unit, question = _QUESTIONS[quantity]
                missing(stage_id + ":" + quantity, "task_fact", "task_owner", question,
                        stage_id=stage_id, quantity=quantity, unit=unit)
        if native is None:
            for fact in _GEOMETRY[kind]:
                missing(stage_id + ":" + fact, "source_measurement", "scene_preparation",
                        "Retain source evidence for " + fact.replace("_", " ") + " for this operation.",
                        stage_id=stage_id, quantity=fact)
            missing(stage_id + ":native_reset", "source_measurement", "scene_preparation",
                    "Retain the exact reset and independently measured starting state for this operation.",
                    stage_id=stage_id, quantity="native_reset")
    if native is None:
        missing("native_runtime_inputs", "runtime_integration", "runtime_operator",
                "Bind a supported engine, robot, calibrated cameras, assets and native state interface to this scene.")
    binding = {**dict(source_bindings), "source_video_digest": source_video,
        "task_semantics_digest": task_semantics.get("digest") if task_semantics is not None else None,
        "task_success_contract_digest": contract.get("contract_digest") if contract is not None else None,
        "native_runtime_contract_digest": native.get("contract_digest") if native is not None else None}
    value = {"schema_version": SCHEMA, "producer": PRODUCER,
        "status": "needs_authoring" if requirements else "external_inputs_retained",
        "precursor_only": True, "native_inputs_authored": False,
        "binding": binding, "task_family": family, "operations": operations,
        "requirements": requirements, "next_action": next((row for row in requirements
            if row["category"] == "task_fact"), requirements[0] if requirements else None),
        "retained_stage_inputs": {"task_success_contract": contract is not None,
                                  "native_runtime_contract": native is not None},
        "source_frame_bindings": [{key: frame.get(key) for key in
            ("frame_id", "image_digest", "geometry_digest")} for frame in source_geometry.get("frames", [])],
        "source_targets": [{key: target.get(key) for key in
            ("target_id", "semantic_label", "task_effect", "target_role")} for target in task_masks.get("targets", [])],
        "native_observables": {row["stage_id"]: list(FAMILY_OBSERVABLES[row["task_family"]]) for row in operations},
        "owner_criteria": task_context.get("success_criteria"), "owner_facts_confirmed": False,
        "claim_ceiling": "development_only", "provider_mutation_performed": False,
        "automatic_paid_execution_authorized": False}
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    path = output_root / "family_authoring" / value["request_digest"][7:] / "requirements.json"
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    write_exclusive_private_bytes(path, payload)
    return {"request": value, "reference": {"path": str(path),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}}
