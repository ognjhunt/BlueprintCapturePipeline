"""Compile capture-bound family tasks without coercing them into relocation.

Beta: job-to-evaluation. The family scorer and runtime adapter exist; this
compiler asks for the exact missing task, assets or runtime bindings. A plan
does not establish native physics application or authorize paid execution.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core.handoff_lease import guarded_handoff_write_json as write_json, handoff_lease_checkpoint
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .task_family_native_contracts import family_source_blockers
from .task_family_scoring import (
    FAMILY_CRITERIA,
    FAMILY_OBSERVABLES,
    canonical_task_family,
    validate_task_family_contract,
)


def prepare_task_family(*, task_context: Mapping[str, Any], task_masks: Mapping[str, Any],
                        removal_manifest: Mapping[str, Any], source_geometry: Mapping[str, Any],
                        base_scene: Mapping[str, Any], output_root: Path,
                        evaluation_inputs: Mapping[str, Any] | None = None,
                        spend: Mapping[str, Any] | None = None, now: float | None = None,
                        task_semantics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Retain the whole job, including multiple subjects and deformable state."""
    handoff_lease_checkpoint("website_task_family_preparation_start")
    from .website_task_semantics import effective_task_profile
    source_video_digest = (source_geometry.get("binding") or {}).get("source_video_digest")
    family = canonical_task_family(effective_task_profile(task_context=task_context,
        task_semantics=task_semantics, source_video_digest=source_video_digest))
    if family is None:
        raise ValueError("website_task_family_unknown")
    if task_masks.get("source_geometry_digest") != source_geometry.get("digest"):
        raise ValueError("website_preparation_geometry_binding_mismatch")
    for path_key, digest_key in (("splat_path", "splat_digest"), ("collision_mesh_path", "collision_mesh_digest")):
        if _sha256_file(Path(base_scene[path_key])) != base_scene[digest_key]:
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
        from .website_task_semantics import require_semantic_stage_order
        require_semantic_stage_order(task_context=task_context, task_semantics=task_semantics,
                                    source_video_digest=source_video_digest, contract=contract)
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
        handoff_lease_checkpoint("website_task_family_native_materialization")
        native = materialize_native_task_runtime_contract(scene_id=scope["scene_id"], task_id=task_id,
            task_spec={"task_kind": family, "task_success_contract": contract}, **kwargs)
        blockers.extend(family_source_blockers(native=native, source_bindings=source_bindings, source_scene=base_scene))
    # Source/task readiness and dispatch authority are separate facts. Missing
    # dispatch inputs retain the prepared job and cannot mint a paid grant.
    dispatch_blockers = []
    request = None
    selection = inputs.get("task_family_runtime_selection")
    policy_selection = inputs.get("policy_selection")
    if not blockers:
        from .task_family_dispatch_contracts import validate_runtime_selection, validate_policy_selection
        try:
            selection = validate_runtime_selection(selection)
            if policy_selection is not None:
                policy_selection = validate_policy_selection(policy_selection)
        except ValueError as exc:
            dispatch_blockers.append(str(exc))
        authority = dict(spend or {})
        if not authority.get("owner") or not authority.get("consent"):
            dispatch_blockers.append("website_scene_execution_authority_required")
        anthropic_terms = authority.get("anthropic_provider_terms_reference")
        if authority.get("authoring_provider") == "anthropic" and (
                not isinstance(anthropic_terms, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", anthropic_terms) is None):
            dispatch_blockers.append("website_anthropic_provider_terms_authority_required")
        if not dispatch_blockers:
            from .task_evaluation_scene_intent_contracts import REQUEST_SCHEMA, validate_request, SceneIntakeError
            request = {"schema_version": REQUEST_SCHEMA, "submission_id": task_context["capture_id"],
                "owner": authority["owner"], "source": {"kind": "gaussian_splat",
                    "binding_id": base_scene["splat_binding_id"], "content_digest": base_scene["splat_digest"]},
                "task": {"task_id": task_id, "strategy": family, "task_success_contract_digest": contract["contract_digest"],
                    "native_runtime_contract_digest": native["contract_digest"],
                    "task_family_runtime_selection": selection,
                    **({"policy_selection": policy_selection} if policy_selection is not None else {})},
                "execution": {"purpose": "scene_preparation", "max_total_spend_usd": authority.get("max_total_spend_usd"),
                    "max_paid_attempts": authority.get("max_paid_attempts"), "max_retries": 0,
                    "expires_at_epoch": authority.get("expires_at_epoch"),
                    "allowed_providers": authority.get("allowed_providers", ["vast", "openai"] +
                        (["anthropic"] if authority.get("authoring_provider") == "anthropic" else [])),
                    "policy_candidates": [], "claim_scope": "development_only"}, "consent": authority["consent"]}
            try:
                request = validate_request(request, now=now)
            except SceneIntakeError as exc:
                dispatch_blockers.append(str(exc))
                request = None
    stages = contract["stages"] if contract else []
    requirements = ({stage["stage_id"]: {"task_family": stage["task_family"],
        "criteria": list(FAMILY_CRITERIA[stage["task_family"]]),
        "native_observables": list(FAMILY_OBSERVABLES[stage["task_family"]])} for stage in stages}
        if stages else {family: {"criteria": list(FAMILY_CRITERIA.get(family, ())),
            "native_observables": list(FAMILY_OBSERVABLES.get(family, ())),
            **({"ordered_stages_required": True} if family == "multi_step" else {})}})
    value = {
        "schema_version": "website_scene_preparation.v1",
        "status": "needs_input" if blockers else "intake_ready" if request else "task_evaluation_ready",
        "dispatch_blockers": dispatch_blockers,
        "blockers": blockers, "job_disposition": "accepted", "claim_ceiling": "development_only",
        "binding": {"task_context_digest": task_context["context_digest"], "task_masks_digest": task_masks["digest"],
            "source_geometry_digest": source_geometry["digest"], "removal_manifest_digest": canonical_digest(removal_manifest),
            "splat_digest": base_scene["splat_digest"], "collision_mesh_digest": base_scene["collision_mesh_digest"]},
        "task_evaluation_inputs": {"task_family": family, "scope": scope,
            **({"task_semantics": dict(task_semantics), "source_video_digest": source_video_digest}
               if task_semantics is not None else {}),
            "runtime_adapter": "task_evaluation_task_family_native_adapter",
            "task_success_contract": contract, "native_runtime_contract": native,
            "source_bindings": source_bindings,
            "task_family_runtime_selection": selection, "policy_selection": policy_selection,
            "asset_sources": inputs.get("asset_sources", {}),
            "runtime_source": inputs.get("runtime_source"),
            "requirements": requirements,
            "source_targets": [dict(row) for row in task_masks.get("targets", ())],
            "source_scene": {key: base_scene.get(key) for key in ("splat_path", "splat_digest", "collision_mesh_path",
                "collision_mesh_digest", "up_axis", "meters_per_unit")},
            "reset_required": True, "independent_native_readback_required": True,
            "automatic_paid_execution_authorized": False},
        **({"website_preparation_authority": dict(spend or {})} if task_context.get("confirmed") is False else {}),
        **({"authoring_provider_terms_reference": spend["anthropic_provider_terms_reference"]}
           if (spend or {}).get("authoring_provider") == "anthropic"
               and isinstance((spend or {}).get("anthropic_provider_terms_reference"), str) else {}),
        "intake_request": request, "thumbnail": None, "provider_mutation_performed": False,
    }
    value["digest"] = canonical_digest(value, digest_field="digest")
    detached = json.loads(json.dumps(value, allow_nan=False))
    write_json(output_root / "preparation.json", detached)
    return detached
