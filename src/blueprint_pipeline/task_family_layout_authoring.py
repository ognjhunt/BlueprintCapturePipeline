"""Emit source-measured family layouts using the shared registration verifier.

Estimated provider geometry never supplies measured calibration, orientation or
robot-base registration. This CPU adapter grants no runtime or spend authority.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_family_authoring_evidence import validate_source_frame_identities
from .task_family_metric_registration import (
    SCHEMA, _KINDS, _QUANTITIES, _derive, _records,
    verify_metric_layout_registration as verify_metric_layout_registration,
)


def emit_registered_task_layout(*, task_context, task_masks, source_geometry, base_scene, output_root):
    """Use existing family authoring stages; missing measurements stay required."""
    if task_masks.get("registered_layout") is not None:
        return task_masks
    source_frames = source_geometry.get("frames")
    validate_source_frame_identities(source_frames,
        maximum_frames=(source_geometry.get("binding") or {}).get("maximum_frames"))
    from .task_family_button_authoring import _retain
    references = task_masks.get("layout_source_records")
    binding = {"capture_id": task_context["capture_id"], "scene_id": task_context["scene_id"],
        "source_video_digest": (source_geometry.get("binding") or {}).get("source_video_digest"),
        "source_geometry_digest": source_geometry["digest"]}
    missing = [kind for kind in _KINDS if not isinstance(references, Mapping) or references.get(kind) is None]
    if (source_geometry.get("metric_measurement_proven") is not True or source_geometry.get("unit") not in {"m", "meters"}
            or source_geometry.get("geometry_backend") not in {"arkit", "native_arkit_rgbd.v1"}):
        missing = list(_KINDS)
    requirements = [{"requirement_id": "job:" + _QUANTITIES[kind], "stage_id": "job", "quantity": _QUANTITIES[kind],
        "category": "source_measurement", "owner": "scene_preparation", "status": "required",
        "question": "Retain source-bound " + _QUANTITIES[kind].replace("_", " ") + "; estimated geometry cannot supply it.",
        "unit": None, "authority_created": False} for kind in missing]
    receipt = {"binding": binding, "source_records": dict(references) if isinstance(references, Mapping) else {},
               "requirements": requirements, "status": "needs_measurement" if missing else "layout_emitted"}
    result = dict(task_masks)
    if not missing:
        frame_rows = [{key: row.get(key) for key in ("frame_id", "intrinsics", "world_from_camera",
            "image_digest", "geometry_digest", "width", "height")} for row in source_frames]
        scene = {key: base_scene[key] for key in ("collision_mesh_digest", "meters_per_unit", "up_axis")}
        target_ids = [row["target_id"] for row in task_masks["targets"] if row.get("task_effect") in {"manipulated", "static_contact"}]
        layout = _derive(records=_records(references), binding=binding, scene=scene, frame_rows=frame_rows, target_ids=target_ids)
        registration = {"schema_version": SCHEMA, "source_records": dict(references), "binding": binding,
            "scene": scene, "source_frames": frame_rows, "target_ids": target_ids, "layout_projection": layout,
            "source_video_digest": binding["source_video_digest"], "frame_registration": layout["frame_registration"],
            "claim_ceiling": "development_only", "automatic_paid_execution_authorized": False}
        registration["registration_digest"] = canonical_digest(registration, digest_field="registration_digest")
        stage = _retain(Path(output_root), "metric-layout-registration", registration)
        # _layout consumes a flat, immutable registration record. Reuse the
        # same exclusive authoring writer rather than rewriting a current file.
        from .launch_immutable_input_writer import write_exclusive_private_bytes
        from .task_object_authoring_contracts import file_record
        path = Path(output_root) / "family_authoring" / "registrations" / registration["registration_digest"][7:] / "registration.json"
        write_exclusive_private_bytes(path, (json.dumps(registration, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
        layout = {**layout, "registration_evidence": file_record(path)}
        layout["layout_digest"] = canonical_digest(layout, digest_field="layout_digest")
        result["registered_layout"] = layout
        receipt.update(layout_digest=layout["layout_digest"], registration_stage=stage)
    result["layout_authoring"] = {"reference": _retain(Path(output_root), "registered-layout", receipt), "requirements": requirements}
    result["digest"] = canonical_digest(result, digest_field="digest")
    return result
