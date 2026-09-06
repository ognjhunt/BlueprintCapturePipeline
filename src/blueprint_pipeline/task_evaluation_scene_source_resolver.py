"""Resolve owned capture/mesh bytes and retain evidence-bounded source decisions.

No supplied mesh is relabeled as InteriorGS or an observed capture. A source
without the measurements required by the admitted construction method returns
specific input requirements instead of manufacturing a qualified scene.
"""
from __future__ import annotations

import os
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_intake as intake
from .task_evaluation_scene_configuration_submission_inputs import read, checked_file, sha


def _record(path):
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def _retain(intent, config, value):
    root = Path(config["factory_output_root"]) / intent["intent_id"] / "source-analysis"
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("scene_source_analysis_path_unsafe")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    value = {**value, "intent_digest": intent["intent_digest"], "provider_mutation_performed": False}
    value["analysis_digest"] = canonical_digest(value, digest_field="analysis_digest")
    path = root / (value["analysis_digest"][7:] + ".json")
    if not path.exists():
        intake.write_exclusive(path, value)
    elif read(path) != value:
        raise ValueError("scene_source_analysis_conflict")
    return _record(path)


def _compiled(intent, config, release):
    """Consume a real compiler's signed-input manifest, never caller readiness flags."""
    from .task_evaluation_scene_progression import SourceResolution
    configured = config.get("compiled_source_binding_root")
    if not configured:
        return None
    path = Path(configured) / (intent["request"]["source"]["content_digest"][7:] + ".json")
    if not path.exists():
        return None
    binding = read(path, digest_field="binding_digest")
    if (binding.get("schema_version") != "task_evaluation_compiled_scene_source.v1"
            or binding.get("source_content_digest") != intent["request"]["source"]["content_digest"]
            or binding.get("source_kind") != intent["request"]["source"]["kind"]
            or binding.get("owner") != intent["request"]["owner"]
            or binding.get("task_digest") != intent["task_content_digest"]):
        raise ValueError("scene_source_compilation_binding_invalid")
    return SourceResolution("resolved", path, Path(config["machinery_path"]), materialize_compiled_source)


def materialize_compiled_source(*, intent_path, source_binding_path, machinery_path,
                               release_binding_path, output_root, attempt_id):
    from .task_evaluation_launch_preparation_contract import validate_launch_preparation_request
    intent = intake._read(Path(intent_path), "intent_digest")
    attempt = intake._read(Path(intent_path).parent / "attempts" / (attempt_id + ".json"), "attempt_digest")
    binding = read(source_binding_path, digest_field="binding_digest")
    release = read(release_binding_path, digest_field="release_digest")
    refs = binding["references"]
    paths = {key: checked_file(row["path"], row) for key, row in refs.items()}
    request = validate_launch_preparation_request(read(paths["submission_request"]))
    if (request.get("scene_intent_digest") != intent["intent_digest"]
            or request["expected_production_commit"] != release["source_commit"]
            or attempt["source_commit"] != release["source_commit"]
            or attempt["input_digest"] != binding["binding_digest"]
            or request["scene"].get("appearance", {}).get("kind") == "interiorgs"):
        raise ValueError("scene_source_compilation_release_or_origin_invalid")
    manifest = read(paths["submission_manifest"], digest_field="manifest_digest")
    if manifest.get("source_commit") != release["source_commit"] or manifest.get("raw_source_upload_allowed") is not False:
        raise ValueError("scene_source_compilation_manifest_invalid")
    result = {"schema_version": "task_evaluation_compiled_scene_factory.v1", "status": "publication_ready",
              "source_commit": release["source_commit"], "attempt_digest": attempt["attempt_digest"],
              "submission_request": refs["submission_request"], "submission_manifest": refs["submission_manifest"],
              "intent_digest": intent["intent_digest"], "provider_mutation_performed": False}
    result["factory_digest"] = canonical_digest(result, digest_field="factory_digest")
    return result


def resolve_scene_source(*, intent, config, release):
    from .task_evaluation_scene_progression import SourceResolution
    source = intent["request"]["source"]
    if source["binding_id"].startswith("native-"):
        # This workflow starts after reconstruction. A raw-capture record may
        # still exist in the shared intake API, but does not start a trainer.
        return SourceResolution("needs_input", blockers=("completed_3d_scene_result_required",))
    store = Path(config.get("capture_store_root") or os.getenv("PIPELINE_CAPTURE_INTAKE_STORE_ROOT", "/var/lib/blueprint/capture-intake"))
    matches = []
    for path in (store / "transfer_receipts").glob("*.json"):
        receipt = read(path)
        if receipt.get("capture_session_id") == source["binding_id"]:
            matches.append((path, receipt))
    if not matches:
        return SourceResolution("awaiting_source", blockers=("capture_upload_byte_verification_pending",))
    if len(matches) != 1:
        raise ValueError("capture_upload_source_ambiguous")
    path, receipt = matches[0]
    from .reconstruction_control_plane import _source_binding
    verified = _source_binding(capture_store_root=store, capture_session_id=source["binding_id"], intake_id=receipt["intake_id"])
    envelope = verified["envelope"]
    if (receipt.get("capture_digest") != source["content_digest"]
            or envelope.get("customer_id") != intent["request"]["owner"]["user_id"]
            or envelope.get("organization_id") != intent["request"]["owner"]["organization_id"]):
        raise ValueError("capture_upload_owner_or_digest_mismatch")
    compiled = _compiled(intent, config, release)
    if compiled is not None:
        return compiled
    if source["kind"] == "gaussian_splat":
        if envelope["capture_authority_profile"] != "provided_scene_splat":
            raise ValueError("provided_splat_source_kind_mismatch")
        from .provided_scene_splat import inspect_splat
        report = inspect_splat(verified["object_path"], coordinate_frame_declaration=envelope["coordinate_frame_declaration"])
        evidence = _retain(intent, config, {"source_kind": "gaussian_splat", "source_receipt": _record(path),
                                           "splat_inspection": report, "captured_observations_supplied": False})
        return SourceResolution("needs_input", blockers=("registered_collision_mesh_and_task_object_identity_required",),
                                analysis_reference=evidence)
    if source["kind"] == "mesh":
        if envelope["capture_authority_profile"] != "provided_scene_mesh":
            raise ValueError("provided_mesh_source_kind_mismatch")
        from .provided_scene_mesh import inspect_mesh
        report = inspect_mesh(verified["object_path"], original_filename=verified["source_original_filename"],
                              coordinate_frame_declaration=envelope["coordinate_frame_declaration"])
        evidence = _retain(intent, config, {"source_kind": "mesh", "source_receipt": _record(path),
                                           "mesh_inspection": report, "capture_observations_supplied": False})
        return SourceResolution("needs_input", blockers=("native_task_object_physics_and_scene_evidence_required",),
                                analysis_reference=evidence)
    evidence = _retain(intent, config, {"source_kind": "capture_bundle", "source_receipt": _record(path),
                                       "capture_admission": verified["receipt"]["claim_ceiling"]})
    return SourceResolution("needs_input", blockers=("metric_workcell_capture_and_object_geometry_required",),
                            analysis_reference=evidence)
