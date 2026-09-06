"""Normalize completed scene assets and exact object identities without reconstructing them."""
from __future__ import annotations

import os
import re

from . import task_evaluation_scene_intake as intake
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import read
from .task_evaluation_scene_progression_state import safe_path, require
from .task_evaluation_public_scene_attempt_factory import record


def owned_upload(binding, owner, config):
    from .reconstruction_control_plane import _source_binding
    store = safe_path(config.get("capture_store_root") or os.getenv("PIPELINE_CAPTURE_INTAKE_STORE_ROOT", "/var/lib/blueprint/capture-intake"))
    matches = []
    for path in (store / "transfer_receipts").glob("*.json"):
        receipt = read(path)
        if receipt.get("capture_session_id") == binding["binding_id"]:
            matches.append((path, receipt))
    if not matches:
        return None
    require(len(matches) == 1, "completed_scene_asset_identity_ambiguous")
    path, receipt = matches[0]
    verified = _source_binding(capture_store_root=store, capture_session_id=binding["binding_id"], intake_id=receipt["intake_id"])
    envelope = verified["envelope"]
    require(receipt.get("capture_digest") == binding["content_digest"]
            and envelope.get("customer_id") == owner["user_id"]
            and envelope.get("organization_id") == owner["organization_id"], "completed_scene_asset_owner_or_digest_mismatch")
    if binding.get("rights_reference") is not None:
        require(envelope["envelope_digest"] == binding["rights_reference"], "completed_scene_asset_rights_changed")
    return path, verified


def _normalized(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def select_exact_object(objects, requested):
    identifier = requested.get("source_object_id") or requested.get("id")
    if identifier:
        matches = [row for row in objects if row["source_object_id"] == identifier]
        basis = "exact_source_object_id"
    else:
        description = _normalized(requested.get("description", ""))
        matches = [row for row in objects if description and _normalized(row["source_object_id"].split("/")[-1]) == description]
        basis = "unique_normalized_source_name"
    if len(matches) != 1:
        return None
    return {**matches[0], "selection_basis": basis, "owner_request": requested,
            "geometry_origin": "provided_asset", "physical_object_identity_proven": False}


def bind_completed_scene_source(*, intent, config):
    from .provided_scene_mesh import inspect_mesh
    source, owner = intent["request"]["source"], intent["request"]["owner"]
    primary = owned_upload({**source, "rights_reference": intent["request"]["consent"]["rights_reference"]}, owner, config)
    if primary is None:
        return {"status": "awaiting_source", "blockers": ["completed_scene_asset_bytes_pending"]}
    primary_receipt, primary_asset = primary
    frame = primary_asset["envelope"]["coordinate_frame_declaration"]
    reports, references = {}, {"primary": record(primary_asset["object_path"]), "primary_receipt": record(primary_receipt)}
    if source["kind"] == "gaussian_splat":
        from .provided_scene_splat import inspect_splat
        require(primary_asset["envelope"]["capture_authority_profile"] == "provided_scene_splat", "completed_scene_splat_kind_mismatch")
        reports["appearance"] = inspect_splat(primary_asset["object_path"], coordinate_frame_declaration=frame)
        if "collision_mesh" not in source:
            return {"status": "needs_input", "blockers": ["registered_collision_mesh_required"], "inspection": reports}
        companion = owned_upload(source["collision_mesh"], owner, config)
        if companion is None:
            return {"status": "awaiting_source", "blockers": ["collision_mesh_bytes_pending"], "inspection": reports}
        collision_receipt, collision = companion
        require(source["collision_mesh"]["frame_relation"] == "owner_declared_common_frame", "completed_scene_frame_relation_missing")
        collision_frame = collision["envelope"]["coordinate_frame_declaration"]
        require(all(frame.get(key) == collision_frame.get(key) for key in ("meters_per_unit", "up_axis")),
                "completed_scene_declared_frames_disagree")
        references.update(collision=record(collision["object_path"]), collision_receipt=record(collision_receipt))
    else:
        require(source["kind"] == "mesh", "completed_scene_kind_invalid")
        collision, collision_frame = primary_asset, frame
        references["collision"] = references["primary"]
    require(collision["envelope"]["capture_authority_profile"] == "provided_scene_mesh", "completed_scene_collision_kind_mismatch")
    reports["geometry"] = inspect_mesh(collision["object_path"], original_filename=collision["source_original_filename"],
                                       coordinate_frame_declaration=collision_frame)
    selected = {role: select_exact_object(reports["geometry"]["objects"], intent["request"]["task"][role])
                for role in ("subject", "support")}
    blockers = [f"source_{role}_object_identity_required" for role, value in selected.items() if value is None]
    if not blockers and selected["subject"]["source_object_id"] == selected["support"]["source_object_id"]:
        blockers.append("source_subject_and_support_must_differ")
    # The declared metric frame is a required typed input: `source_task_objects_bound`
    # never fires on an undeclared or non-physical scale, so no downstream stage
    # can silently treat unit-less geometry as metric.
    scale = frame.get("meters_per_unit")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not (0 < scale <= 1000) \
            or frame.get("up_axis") not in {"Y", "Z"}:
        blockers.append("source_metric_scale_declaration_required")
    value = {"schema_version": "task_evaluation_completed_scene_source.v1",
        "binding_id": source["binding_id"], "source_content_digest": source["content_digest"],
        "intent_digest": intent["intent_digest"], "owner": owner, "source_kind": source["kind"],
        "task_digest": intent["task_content_digest"], "references": references, "inspection": reports,
        "object_bindings": selected, "coordinate_frame": frame,
        "status": "needs_input" if blockers else "source_task_objects_bound", "blockers": blockers,
        "physical_scale_measured": False, "physical_registration_proven": False,
        "provider_reconstruction_started": False, "provider_mutation_performed": False}
    value["binding_digest"] = canonical_digest(value, digest_field="binding_digest")
    root = safe_path(config["factory_output_root"]) / intent["intent_id"] / "completed-source"
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    path = root / (value["binding_digest"][7:] + ".json")
    if not path.exists():
        intake.write_exclusive(path, value)
    require(read(path, digest_field="binding_digest") == value, "completed_scene_binding_changed")
    return {**value, "binding_reference": record(path)}
