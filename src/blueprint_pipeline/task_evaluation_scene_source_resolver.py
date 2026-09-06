"""Resolve owned capture/mesh bytes and retain evidence-bounded source decisions.

No supplied mesh is relabeled as InteriorGS or an observed capture. A source
without the measurements required by the admitted construction method returns
specific input requirements instead of manufacturing a qualified scene.

A completed 3DGS/mesh source (with a companion collision mesh, declared frame,
named subject/support, and owner consent) is bound by
``bind_completed_scene_source`` and, once every typed input is genuinely
present, drives ``materialize_completed_scene_attempt`` to a publication-ready
scene-configuration submission.  Capture bundles still require metric workcell
capture and object geometry before they can start a trainer.
"""
from __future__ import annotations

import os
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_intake as intake
from .task_evaluation_scene_configuration_submission_inputs import read, sha


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


def _completed_resolution(intent, config):
    """Bind owner-provided completed bytes; only a full typed binding resolves."""
    from .task_evaluation_scene_progression import SourceResolution
    from .task_evaluation_completed_scene_source import bind_completed_scene_source
    bound = bind_completed_scene_source(intent=intent, config=config)
    status = bound.get("status")
    analysis = bound.get("binding_reference")
    if status == "source_task_objects_bound":
        if analysis is None:
            raise ValueError("completed_source_binding_reference_missing")
        if not config.get("completed_source_machinery_path"):
            raise ValueError("completed_source_machinery_path_missing")
        from .task_evaluation_completed_scene_attempt_factory import materialize_completed_scene_attempt
        return SourceResolution("resolved", binding_path=Path(analysis["path"]),
                                machinery_path=Path(config["completed_source_machinery_path"]),
                                materializer=materialize_completed_scene_attempt)
    if status not in {"awaiting_source", "needs_input"}:
        raise ValueError("completed_source_status_invalid")
    return SourceResolution(status, blockers=tuple(bound.get("blockers", ())), analysis_reference=analysis)


def resolve_scene_source(*, intent, config, release):
    from .task_evaluation_scene_progression import SourceResolution
    source = intent["request"]["source"]
    if source["binding_id"].startswith("native-"):
        # This workflow starts after reconstruction. A raw-capture record may
        # still exist in the shared intake API, but does not start a trainer.
        return SourceResolution("needs_input", blockers=("completed_3d_scene_result_required",))
    if source["kind"] in {"gaussian_splat", "mesh"}:
        return _completed_resolution(intent, config)
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
    evidence = _retain(intent, config, {"source_kind": "capture_bundle", "source_receipt": _record(path),
                                       "capture_admission": verified["receipt"]["claim_ceiling"]})
    return SourceResolution("needs_input", blockers=("metric_workcell_capture_and_object_geometry_required",),
                            analysis_reference=evidence)
