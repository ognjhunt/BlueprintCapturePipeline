"""Compile capture-bound family tasks without coercing them into relocation.

Beta: job-to-evaluation. The family scorer and runtime adapter exist; this
compiler asks for the exact missing task, assets or runtime bindings. A plan
does not establish native physics application or authorize paid execution.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .common import sha256_file, write_json
from .decision_evidence_contracts import canonical_digest
from .task_family_scoring import (
    FAMILY_CRITERIA,
    FAMILY_OBSERVABLES,
    canonical_task_family,
    validate_task_family_contract,
)


def family_source_blockers(*, native: Mapping[str, Any], source_bindings: Mapping[str, Any],
                           source_scene: Mapping[str, Any]) -> list[str]:
    """Apply the same source/frame requirements at compile and retained read."""
    blockers: list[str] = []
    unit = source_scene.get("meters_per_unit")
    if (isinstance(unit, bool) or not isinstance(unit, (int, float)) or not math.isfinite(unit) or unit <= 0
            or source_scene.get("up_axis") not in {"Y", "-Y", "Z"}):
        raise ValueError("website_task_family_source_frame_invalid")
    binding = native["task_state_binding"].get("source_scene_binding")
    if not isinstance(binding, Mapping):
        blockers.append("website_task_family_source_scene_binding_required")
    elif any(binding.get(key) != source_bindings.get(key) for key in (
            "task_context_digest", "source_geometry_digest", "task_masks_digest", "splat_digest", "collision_mesh_digest")):
        raise ValueError("website_task_family_source_scene_binding_mismatch")
    else:
        frame = binding.get("frame_registration")
        if not isinstance(frame, Mapping):
            blockers.append("website_task_family_frame_registration_required")
        else:
            transform = np.asarray(frame.get("source_to_runtime"), dtype=float)
            linear = transform[:3, :3] if transform.shape == (4, 4) else np.zeros((3, 3))
            gram = linear.T @ linear
            if (frame.get("source_geometry_digest") != source_bindings.get("source_geometry_digest")
                    or frame.get("collision_mesh_digest") != source_scene["collision_mesh_digest"]
                    or frame.get("runtime_up_axis") != source_scene["up_axis"]
                    or frame.get("runtime_meters_per_unit") != source_scene.get("meters_per_unit")
                    or frame.get("registration_digest") != canonical_digest(frame, digest_field="registration_digest")
                    or transform.shape != (4, 4) or not np.all(np.isfinite(transform))
                    or not np.allclose(transform[3], [0, 0, 0, 1])
                    or np.linalg.det(linear) <= 1e-12
                    or not np.allclose(gram, np.eye(3) * np.trace(gram) / 3, rtol=1e-8, atol=1e-10)):
                raise ValueError("website_task_family_frame_registration_invalid")
    for role, digest in (("scene_appearance", source_scene["splat_digest"]),
                         ("scene_collision", source_scene["collision_mesh_digest"])):
        if not any(row.get("semantic_role") == role and row.get("content_digest") == digest for row in native["assets"]):
            blockers.append("website_task_family_" + role + "_asset_required")
    return blockers


def prepare_task_family(*, task_context: Mapping[str, Any], task_masks: Mapping[str, Any],
                        removal_manifest: Mapping[str, Any], source_geometry: Mapping[str, Any],
                        base_scene: Mapping[str, Any], output_root: Path,
                        evaluation_inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Retain the whole job, including multiple subjects and deformable state."""
    family = canonical_task_family(task_context.get("task_family"))
    if family is None:
        raise ValueError("website_task_family_unknown")
    if task_masks.get("source_geometry_digest") != source_geometry.get("digest"):
        raise ValueError("website_preparation_geometry_binding_mismatch")
    for path_key, digest_key in (("splat_path", "splat_digest"), ("collision_mesh_path", "collision_mesh_digest")):
        if "sha256:" + sha256_file(Path(base_scene[path_key])) != base_scene[digest_key]:
            raise ValueError("website_base_scene_changed")
    inputs = dict(evaluation_inputs or {})
    blockers: list[str] = []
    if (task_context.get("capture_rights") or {}).get("derived_scene_generation_allowed") is not True:
        blockers.append("website_scene_processing_rights_required")
    task_id = task_context.get("task_id") or "website-" + task_context["context_digest"][7:27]
    scope = {"scene_id": task_context["scene_id"], "task_id": task_id,
             "task_context_digest": task_context["context_digest"]}
    contract = inputs.get("task_success_contract")
    if contract is None:
        blockers.append("website_task_family_success_contract_required")
    else:
        contract = validate_task_family_contract(contract)
        if contract["scope"] != scope or contract["task_family"] != family:
            raise ValueError("website_task_family_contract_scope_mismatch")
    # Runtime inputs are data from the configured native scene, not model
    # executable code. The existing native contract entrypoint validates them.
    native = None
    source_bindings = {"task_context_digest": task_context["context_digest"],
        "source_geometry_digest": source_geometry["digest"], "task_masks_digest": task_masks["digest"],
        "splat_digest": base_scene["splat_digest"], "collision_mesh_digest": base_scene["collision_mesh_digest"]}
    runtime_inputs = inputs.get("native_runtime_inputs")
    if runtime_inputs is None:
        blockers.append("website_task_family_native_runtime_inputs_required")
    elif contract is not None:
        if not isinstance(runtime_inputs, Mapping):
            raise ValueError("website_task_family_native_runtime_inputs_invalid")
        from .native_task_runtime_contract import materialize_native_task_runtime_contract
        kwargs = dict(runtime_inputs)
        if set(kwargs) & {"scene_id", "task_id", "task_spec", "destination"}:
            raise ValueError("website_task_family_native_runtime_scope_override")
        native = materialize_native_task_runtime_contract(scene_id=scope["scene_id"], task_id=task_id,
            task_spec={"task_kind": family, "task_success_contract": contract}, **kwargs)
        blockers.extend(family_source_blockers(native=native, source_bindings=source_bindings, source_scene=base_scene))
    stages = contract["stages"] if contract else []
    requirements = ({stage["stage_id"]: {"task_family": stage["task_family"],
        "criteria": list(FAMILY_CRITERIA[stage["task_family"]]),
        "native_observables": list(FAMILY_OBSERVABLES[stage["task_family"]])} for stage in stages}
        if stages else {family: {"criteria": list(FAMILY_CRITERIA.get(family, ())),
            "native_observables": list(FAMILY_OBSERVABLES.get(family, ())),
            **({"ordered_stages_required": True} if family == "multi_step" else {})}})
    value = {
        "schema_version": "website_scene_preparation.v1",
        "status": "needs_input" if blockers else "task_evaluation_ready",
        "blockers": blockers, "job_disposition": "accepted", "claim_ceiling": "development_only",
        "binding": {"task_context_digest": task_context["context_digest"], "task_masks_digest": task_masks["digest"],
            "source_geometry_digest": source_geometry["digest"], "removal_manifest_digest": canonical_digest(removal_manifest),
            "splat_digest": base_scene["splat_digest"], "collision_mesh_digest": base_scene["collision_mesh_digest"]},
        "task_evaluation_inputs": {"task_family": family, "scope": scope,
            "runtime_adapter": "task_evaluation_task_family_native_adapter",
            "task_success_contract": contract, "native_runtime_contract": native,
            "source_bindings": source_bindings,
            "requirements": requirements,
            "source_targets": [dict(row) for row in task_masks.get("targets", ())],
            "source_scene": {key: base_scene.get(key) for key in ("splat_path", "splat_digest", "collision_mesh_path",
                "collision_mesh_digest", "up_axis", "meters_per_unit")},
            "reset_required": True, "independent_native_readback_required": True,
            "automatic_paid_execution_authorized": False},
        "intake_request": None, "thumbnail": None, "provider_mutation_performed": False,
    }
    value["digest"] = canonical_digest(value, digest_field="digest")
    detached = json.loads(json.dumps(value, allow_nan=False))
    write_json(output_root / "preparation.json", detached)
    return detached
