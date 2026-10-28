"""Pure validation of website task context; no signed reads or execution.

ADP-010/day14 compatibility preserves the exact default digest/confirmation and
explicit preparation projection, including false/null/unknown source facts.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest


def admits_scene_preparation(value: Mapping[str, Any] | None) -> bool:
    """A site-confirmed task, or the assessment-proposed task of a preparation run.

    This only stops a site click from gating scene preparation. It grants no
    spend: paid calls still need the loaded sponsorship (issued for an admitted
    assessment proposal) and a reservation bound to this task context. Site
    confirmation still gates what commits the site: success thresholds,
    sharing and physical trials.
    """
    return bool(value) and (value.get("confirmed") is True or (
        value.get("purpose") == "scene_preparation" and value.get("confirmed") is False))


def sponsors_scene_preparation(metadata: Mapping[str, Any] | None) -> bool:
    """A loaded scene_preparation sponsorship requests the scene preview and geometry it funds."""
    return ((metadata or {}).get("website_scene_execution_authority") or {}).get("purpose") == "scene_preparation"


def holds_native_task_preparation(handoff: Mapping[str, Any], task_context: Mapping[str, Any]) -> bool:
    """A completed scene-preparation stage may retain an exact native-profile gap.

    This does not admit native execution or turn an arbitrary failed preparation
    into a completed stage. Current owner/source checks and the output commit
    remain the callers' responsibility.
    """
    if not isinstance(handoff, Mapping) or not isinstance(task_context, Mapping):
        return False
    gap = handoff.get("native_task_preparation_gap")
    family, description, digest = (task_context.get(key) for key in
                                   ("task_family", "description", "context_digest"))
    if (not isinstance(gap, Mapping) or not isinstance(family, str) or not family.strip()
            or not isinstance(description, str) or not description.strip()
            or not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            or any(not isinstance(task_context.get(key), str) or not task_context[key]
                   for key in ("scene_id", "capture_id"))):
        return False
    from .task_profile_contracts import task_profile_for

    code = gap.get("code")
    declared = task_profile_for(family)
    if code == "website_native_task_profile_preparation_required":
        if declared is not None or handoff.get("provider_mutation_performed") is not False:
            return False
    elif code == "website_task_operation_coverage_unresolved":
        from .website_operation_coverage import operation_coverage_gap
        try:
            if (family != "general_task" or not isinstance(gap.get("task_semantics"), Mapping)
                    or gap != operation_coverage_gap(gap["task_semantics"], task_context=task_context,
                        source_video_digest=gap.get("source_video_digest"))
                    or ("provider_mutation_performed" in handoff
                        and type(handoff["provider_mutation_performed"]) is not bool)):
                return False
        except (ValueError, TypeError, KeyError):
            return False
    elif code == "website_task_profile_mechanism_mismatch":
        if declared is None and family == "general_task" and isinstance(gap.get("task_semantics"), Mapping):
            from .website_task_semantics import resolved_task_profile
            try:
                resolved = resolved_task_profile(gap["task_semantics"], task_context=task_context,
                    source_video_digest=gap.get("source_video_digest"))
                declared = task_profile_for(resolved)
            except (ValueError, TypeError, KeyError):
                return False
        inferred_id = gap.get("inferred_profile_id")
        inferred = task_profile_for(inferred_id) if isinstance(inferred_id, str) else None
        if (declared is None or inferred is None or inferred.profile_id != inferred_id
                or declared.profile_id == inferred.profile_id
                or ("provider_mutation_performed" in handoff and
                    type(handoff["provider_mutation_performed"]) is not bool)):
            return False
    else:
        return False
    return (
        task_context.get("schema_version") == "website_site_task_context.v1"
        and digest == canonical_digest(task_context, digest_field="context_digest")
        and handoff.get("schema_version") == "website_scene_handoff.v1"
        and handoff.get("digest") == canonical_digest(handoff, digest_field="digest")
        and all(handoff.get(key) == task_context[key] for key in ("scene_id", "capture_id"))
        and handoff.get("status") == "needs_input"
        and handoff.get("blockers") == [code]
        and handoff.get("simulator_ready") is False
        and handoff.get("visual_reconstruction_ready") is True
        and not handoff.get("website_intake_outbox")
        and "runtime_inputs" not in handoff
        and gap.get("code") == code
        and gap.get("task_context_digest") == digest
        and gap.get("task_family") == family
        and gap.get("description") == description
        and gap.get("job_disposition") == "held_for_preparation"
    )


def retains_task_family_preparation(handoff: Mapping[str, Any], task_context: Mapping[str, Any]) -> bool:
    """Recognize the compiled family route without claiming runtime execution."""
    if not isinstance(handoff, Mapping) or not isinstance(task_context, Mapping):
        return False
    from .task_family_scoring import canonical_task_family, validate_task_family_contract
    inputs = handoff.get("task_evaluation_inputs")
    try:
        if not isinstance(inputs, Mapping):
            return False
        from .website_task_semantics import effective_task_profile
        family = canonical_task_family(effective_task_profile(task_context=task_context,
            task_semantics=inputs.get("task_semantics"), source_video_digest=inputs.get("source_video_digest")))
        if (family is None or not isinstance(inputs, Mapping)
                or task_context.get("context_digest") != canonical_digest(task_context, digest_field="context_digest")
                or handoff.get("schema_version") != "website_scene_handoff.v1"
                or handoff.get("digest") != canonical_digest(handoff, digest_field="digest")
                or any(handoff.get(key) != task_context.get(key) for key in ("scene_id", "capture_id"))
                or handoff.get("visual_reconstruction_ready") is not True
                or handoff.get("simulator_ready") is not False
                or handoff.get("job_disposition") != "accepted"
                or inputs.get("task_family") != family
                or (inputs.get("scope") or {}).get("task_context_digest") != task_context["context_digest"]
                or (task_context.get("task_family") == "general_task" and (
                    (inputs.get("source_bindings") or {}).get("task_context_digest") != task_context["context_digest"]
                    or (handoff.get("source_geometry_digest") is not None and
                        (inputs.get("source_bindings") or {}).get("source_geometry_digest") != handoff["source_geometry_digest"])))
                or inputs.get("runtime_adapter") != "task_evaluation_task_family_native_adapter"
                or inputs.get("automatic_paid_execution_authorized") is not False
                or "runtime_inputs" in handoff
                or handoff.get("status") not in {"needs_input", "task_evaluation_ready", "intake_ready"}):
            return False
        contract = inputs.get("task_success_contract")
        if contract is not None:
            contract = validate_task_family_contract(contract)
            if contract["scope"] != inputs["scope"] or contract["task_family"] != family:
                return False
            from .website_task_semantics import require_semantic_stage_order
            require_semantic_stage_order(task_context=task_context, task_semantics=inputs.get("task_semantics"),
                source_video_digest=inputs.get("source_video_digest"), contract=contract)
        blockers = handoff.get("blockers")
        if handoff["status"] == "needs_input":
            return (isinstance(blockers, list) and bool(blockers)
                and all(isinstance(code, str) and code in {
                    "website_scene_processing_rights_required", "website_task_family_success_contract_required",
                    "website_task_family_native_runtime_inputs_required", "website_task_family_source_scene_binding_required",
                    "website_task_family_frame_registration_required", "website_task_family_scene_appearance_asset_required",
                    "website_task_family_scene_collision_asset_required"} for code in blockers))
        if blockers != [] or contract is None:
            return False
        from .task_family_native_contracts import family_source_blockers, validate_task_family_native_runtime_contract
        native = validate_task_family_native_runtime_contract(inputs["native_runtime_contract"])
        source = inputs["source_bindings"]
        return (native["task_spec"]["task_success_contract"] == contract
            and source.get("task_context_digest") == task_context["context_digest"]
            and source.get("source_geometry_digest") == handoff.get("source_geometry_digest")
            and not family_source_blockers(native=native, source_bindings=source, source_scene=inputs["source_scene"]))
    except (ValueError, KeyError, TypeError):
        return False


def validate_website_task_context(
    value: Mapping[str, Any], *, request_id: str, scene_id: str, capture_id: str,
    purpose: str | None = None,
) -> dict[str, Any]:
    if value.get("schema_version") != "website_site_task_context.v1":
        raise ValueError("website_task_context_schema_invalid")
    if any(value.get(key) != expected for key, expected in (
        ("request_id", request_id), ("scene_id", scene_id), ("capture_id", capture_id),
    )):
        raise ValueError("website_task_context_identity_mismatch")
    if value.get("context_digest") != canonical_digest(value, digest_field="context_digest"):
        raise ValueError("website_task_context_digest_mismatch")
    if purpose not in {None, "scene_preparation"}:
        raise ValueError("website_task_context_purpose_invalid")
    if purpose == "scene_preparation" and value.get("purpose") != purpose and not (
            "purpose" not in value and value.get("confirmed") is True and value.get("confirmed_at")):
        raise ValueError("website_task_context_purpose_mismatch")
    if purpose == "scene_preparation" and (type(value.get("confirmed")) is not bool
            or (value["confirmed"] is False and value.get("confirmed_at") is not None)
            or (value["confirmed"] is True and not value.get("confirmed_at"))):
        raise ValueError("website_task_context_confirmation_invalid")
    if purpose is None and (value.get("confirmed") is not True or not value.get("confirmed_at")):
        raise ValueError("website_task_context_not_confirmed")
    if not isinstance(value.get("description"), str) or not value["description"].strip():
        raise ValueError("website_task_context_description_missing")
    # A source-bound analysis may name any family. The runnable inventory is
    # an execution concern, never an intake allowlist.
    if "task_family" in value and (not isinstance(value["task_family"], str)
                                   or not value["task_family"].strip()):
        raise ValueError("website_task_context_family_invalid")
    details = value.get("operator_task_details")
    if details is not None and (not isinstance(details, Mapping) or any(key not in {"item_weight", "item_make_model"}
            or not isinstance(item, str) for key, item in details.items())):
        raise ValueError("website_task_context_item_details_invalid")
    criteria = value.get("success_criteria")
    if criteria is not None:
        if not isinstance(criteria, Mapping) or set(criteria) != {"successDefinition", "successRate", "cycleTimeSeconds", "unknown"}:
            raise ValueError("website_task_context_success_criteria_invalid")
        if type(criteria["unknown"]) is not bool or (criteria["successDefinition"] is not None and not isinstance(criteria["successDefinition"], str)):
            raise ValueError("website_task_context_success_criteria_invalid")
        for key in ("successRate", "cycleTimeSeconds"):
            number = criteria[key]
            if number is not None and (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(number) or number < 0 or (key == "successRate" and number > 100)):
                raise ValueError("website_task_context_success_criteria_invalid")
    items = value.get("task_items")
    if items is not None:
        if not isinstance(items, list) or len(items) > 100:
            raise ValueError("website_task_context_items_invalid")
        identifiers = []
        for item in items:
            if not isinstance(item, Mapping) or not isinstance(item.get("item_id"), str) or not isinstance(item.get("label"), str):
                raise ValueError("website_task_context_items_invalid")
            identifiers.append(item["item_id"])
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", item["item_id"]) or not isinstance(item.get("images"), list):
                raise ValueError("website_task_context_items_invalid")
            for image in item["images"]:
                prefix = f"scenes/{scene_id}/items/{item['item_id']}/"
                name = image.get("storage_path", "") if isinstance(image, Mapping) else ""
                if not isinstance(name, str) or not name.startswith(prefix) or any(part in {"", ".", ".."} for part in name.split("/")):
                    raise ValueError("website_item_source_outside_site")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("website_task_context_items_invalid")
    continuation = value.get("capture_binding")
    if capture_id.startswith("supplement-") and continuation is None:
        raise ValueError("website_task_context_continuation_invalid")
    if continuation is not None:
        if (not isinstance(continuation, Mapping) or continuation.get("schema_version") not in {
                "website_capture_continuation.v1", "website_capture_continuation.v2"}
                or continuation.get("capture_id") != capture_id or continuation.get("original_capture_id") != f"walkthrough-{request_id}"
                or continuation.get("coordinate_frames_independent") is not True
                or not isinstance(continuation.get("lineage"), list) or not 1 <= len(continuation["lineage"]) <= 8):
            raise ValueError("website_task_context_continuation_invalid")
        expected = capture_id
        seen = set()
        for entry in continuation["lineage"]:
            if (not isinstance(entry, Mapping) or any(not isinstance(entry.get(key), Mapping) for key in ("child", "parent", "supplement"))
                    or entry["child"].get("capture_id") != expected or expected in seen
                    or entry.get("supplement", {}).get("parent_capture_id") != entry.get("parent", {}).get("capture_id")
                    or entry.get("supplement", {}).get("parent_bundle_digest") != entry.get("parent", {}).get("raw_bundle_digest")
                    or entry.get("supplement", {}).get("parent_manifest_uri") != entry.get("parent", {}).get("raw_manifest_uri")):
                raise ValueError("website_task_context_continuation_invalid")
            seen.add(expected)
            expected = entry["parent"]["capture_id"]
        if expected != f"walkthrough-{request_id}":
            raise ValueError("website_task_context_continuation_invalid")
        if continuation["schema_version"] == "website_capture_continuation.v1" and any(
                entry.get("recording_scope") == "additional_views"
                or entry["supplement"].get("recording_scope") == "additional_views"
                for entry in continuation["lineage"]):
            raise ValueError("website_task_context_continuation_invalid")
        if continuation["schema_version"] == "website_capture_continuation.v2":
            from .website_recording_sources import recording_sources
            recording_sources(continuation, request_id=request_id, scene_id=scene_id, capture_id=capture_id)
    return dict(value)


