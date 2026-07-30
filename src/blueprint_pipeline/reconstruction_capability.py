"""Replaceable reconstruction planning and bounded spatial asset decisions.

These artifacts are derived evidence. They never upgrade raw-capture authority,
and visual/generated outputs never become collision or physical truth.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


METHOD_PROFILE_SCHEMA_VERSION = "reconstruction_method_profile.v1"
PLAN_SCHEMA_VERSION = "reconstruction_plan.v1"
RESULT_SCHEMA_VERSION = "reconstruction_result.v1"
SIMREADY_DECISION_SCHEMA_VERSION = "simready_asset_decision.v1"
ROBOT_PLACEMENT_SCHEMA_VERSION = "robot_placement_result.v1"

METHOD_KINDS = {
    "decoded_observation_index",
    "pose_sfm_estimation",
    "metric_scaffold",
    "lidar_depth_fusion",
    "photogrammetric_mesh",
    "gaussian_splat_3d",
    "semantic_scene_graph",
    "object_segmentation",
    "room_structural_prior",
    "collision_proxy",
    "articulated_object_asset",
    "simready_usd_composition",
    "generated_visual_completion",
    "manual_owner_attested_correction",
}
REPRESENTATIONS = {
    "decoded_observation_frames",
    "calibrated_frames",
    "appearance_layer",
    "metric_reference_layer",
    "semantic_layer",
    "physics_layer",
    "collision_geometry",
    "articulated_object_asset",
}
PHYSICS_DEPENDENT_CLAIMS = {
    "collision_contact",
    "grasp_contact",
    "articulation",
    "containment",
    "mass_inertia",
    "friction_compliance",
    "object_state_transition",
}
GENERATED_FORBIDDEN_CLAIM_OUTPUTS = {
    "metric_reference_layer",
    "physics_layer",
    "collision_geometry",
    "articulated_object_asset",
}


class ReconstructionContractError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__("; ".join(self.errors))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ReconstructionContractError(["artifact:not_json_serializable"]) from exc


def _text(value: Any) -> str:
    return str(value or "").strip()


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({_text(item) for item in value if _text(item)})


def _is_digest(value: Any) -> bool:
    text = _text(value)
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _number(value: Any, *, minimum: float = 0.0, maximum: float | None = None) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < minimum or (maximum is not None and number > maximum):
        return None
    return number


def _reject_secrets(value: Any, *, prefix: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            lowered = key.lower()
            if any(word in lowered for word in ("password", "secret", "credential", "api_key")):
                if nested not in (None, "", [], {}):
                    errors.append(f"secret_value_forbidden:{path}")
            errors.extend(_reject_secrets(nested, prefix=path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            errors.extend(_reject_secrets(nested, prefix=f"{prefix}[{index}]"))
    return errors


def build_reconstruction_method_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    profile = _clone(dict(value))
    errors: list[str] = []
    supplied_schema = profile.get("schema_version")
    supplied_digest = profile.get("method_profile_digest")
    if supplied_schema not in (None, METHOD_PROFILE_SCHEMA_VERSION):
        errors.append("schema_version:mismatch")
    profile["schema_version"] = METHOD_PROFILE_SCHEMA_VERSION
    for key in ("method_id", "version", "provider_identity", "execution_mode"):
        if not _text(profile.get(key)):
            errors.append(f"{key}:missing")
    if not _is_digest(profile.get("implementation_digest")):
        errors.append("implementation_digest:invalid")
    if _text(profile.get("method_kind")) not in METHOD_KINDS:
        errors.append("method_kind:unsupported")
    outputs = _strings(profile.get("outputs"))
    if not outputs or any(item not in REPRESENTATIONS for item in outputs):
        errors.append("outputs:missing_or_unsupported")
    profile["outputs"] = outputs
    profile["required_capture_authority_profiles"] = _strings(
        profile.get("required_capture_authority_profiles")
    )
    profile["required_claim_ceiling_flags"] = _strings(
        profile.get("required_claim_ceiling_flags")
    )
    profile["qualified_claim_types"] = _strings(profile.get("qualified_claim_types"))
    if not isinstance(profile.get("execution_authorized"), bool):
        errors.append("execution_authorized:must_be_boolean")
    if _text(profile.get("qualification_status")) not in {
        "qualified",
        "debug_only",
        "not_qualified",
    }:
        errors.append("qualification_status:unsupported")
    cost = _number(profile.get("expected_cost_usd"))
    if cost is None:
        errors.append("expected_cost_usd:invalid")
    else:
        profile["expected_cost_usd"] = cost
    profile.setdefault("provider_constraints", {})
    profile.setdefault("rights_constraints", {})
    profile.setdefault("failure_modes", [])
    errors.extend(_reject_secrets(profile))
    expected_digest = canonical_digest(profile, digest_field="method_profile_digest")
    if supplied_digest is not None and supplied_digest != expected_digest:
        errors.append("method_profile_digest:mismatch")
    if errors:
        raise ReconstructionContractError(errors)
    profile["method_profile_digest"] = expected_digest
    return profile


def _required_representations(claim_types: Sequence[str]) -> list[str]:
    claims = {_text(claim) for claim in claim_types if _text(claim)}
    required: set[str] = set()
    if claims.intersection({"perception_visibility", "task_discovery", "appearance_review"}):
        required.add("decoded_observation_frames")
    if claims.intersection({"reachability", "robot_placement", "navigation_clearance"}):
        required.add("metric_reference_layer")
    if claims.intersection(PHYSICS_DEPENDENT_CLAIMS):
        required.update({"metric_reference_layer", "physics_layer", "collision_geometry"})
    if "articulation" in claims:
        required.add("articulated_object_asset")
    if "appearance_review" in claims:
        required.add("appearance_layer")
    return sorted(required)


def plan_reconstruction_methods(
    *,
    intake_id: str,
    capture_digest: str,
    capture_authority_profile: str,
    claim_ceiling: Mapping[str, Any],
    requested_claim_types: Sequence[str],
    permitted_provider_identities: Sequence[str],
    method_profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose the cheapest authorized method set that covers required outputs."""

    if not _text(intake_id) or not _is_digest(capture_digest):
        raise ReconstructionContractError(["source_capture_binding:invalid"])
    profiles = [build_reconstruction_method_profile(row) for row in method_profiles]
    permitted = {_text(item) for item in permitted_provider_identities if _text(item)}
    required = _required_representations(requested_claim_types)
    rejected_by_representation: dict[str, list[dict[str, str]]] = {
        representation: [] for representation in required
    }
    applicable: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    for profile in profiles:
        eligible_outputs: list[str] = []
        for representation in required:
            if representation not in profile["outputs"]:
                continue
            reason = ""
            if not profile["execution_authorized"]:
                reason = "execution_not_authorized"
            elif permitted and profile["provider_identity"] not in permitted:
                reason = "provider_not_permitted"
            elif (
                profile["required_capture_authority_profiles"]
                and capture_authority_profile
                not in profile["required_capture_authority_profiles"]
            ):
                reason = "capture_authority_profile_not_supported"
            elif any(
                claim_ceiling.get(flag) is not True
                for flag in profile["required_claim_ceiling_flags"]
            ):
                reason = "required_capture_evidence_missing"
            elif representation in {
                "physics_layer",
                "collision_geometry",
                "articulated_object_asset",
            } and profile["qualification_status"] != "qualified":
                reason = "method_not_qualified_for_physics_output"
            elif profile["method_kind"] == "generated_visual_completion" and (
                representation in GENERATED_FORBIDDEN_CLAIM_OUTPUTS
            ):
                reason = "generated_completion_cannot_supply_physics_or_metric_output"
            if reason:
                rejected_by_representation[representation].append(
                    {"method_id": profile["method_id"], "reason": reason}
                )
            else:
                eligible_outputs.append(representation)
        if eligible_outputs:
            applicable.append((profile, tuple(sorted(eligible_outputs))))
    applicable.sort(
        key=lambda item: (
            item[0]["method_id"],
            item[0]["version"],
            item[0]["method_profile_digest"],
        )
    )

    representation_bits = {
        representation: 1 << index for index, representation in enumerate(required)
    }
    # mask -> (cost, profile indexes). The deterministic tuple tie-break avoids
    # provider ordering or input ordering changing the plan.
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for index, (profile, outputs) in enumerate(applicable):
        output_mask = 0
        for output in outputs:
            output_mask |= representation_bits[output]
        for mask, (cost, indexes) in list(states.items()):
            next_mask = mask | output_mask
            candidate = (round(cost + profile["expected_cost_usd"], 9), indexes + (index,))
            existing = states.get(next_mask)
            if existing is None or (candidate[0], len(candidate[1]), candidate[1]) < (
                existing[0],
                len(existing[1]),
                existing[1],
            ):
                states[next_mask] = candidate
    full_mask = (1 << len(required)) - 1
    if full_mask in states:
        chosen_mask = full_mask
    else:
        chosen_mask = min(
            states,
            key=lambda mask: (
                -mask.bit_count(),
                states[mask][0],
                len(states[mask][1]),
                states[mask][1],
            ),
        )
    chosen_indexes = states[chosen_mask][1]
    selected = [
        {
            "representations": list(applicable[index][1]),
            "method_id": applicable[index][0]["method_id"],
            "method_version": applicable[index][0]["version"],
            "method_profile_digest": applicable[index][0]["method_profile_digest"],
            "provider_identity": applicable[index][0]["provider_identity"],
            **(
                {"adapter_reference": applicable[index][0]["adapter_reference"]}
                if applicable[index][0].get("adapter_reference")
                else {}
            ),
            "expected_cost_usd": applicable[index][0]["expected_cost_usd"],
        }
        for index in chosen_indexes
    ]
    missing: list[dict[str, Any]] = []
    for representation in required:
        if chosen_mask & representation_bits[representation]:
            continue
        missing.append(
            {
                "representation": representation,
                "rejected_candidates": sorted(
                    rejected_by_representation[representation],
                    key=lambda item: (item["method_id"], item["reason"]),
                ),
                "next_cheapest_experiment": (
                    "obtain a verified collision/physics asset or targeted object capture"
                    if representation in GENERATED_FORBIDDEN_CLAIM_OUTPUTS
                    else "capture or measure the missing representation with a permitted method"
                ),
            }
        )
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_capture": {
            "intake_id": intake_id,
            "capture_digest": capture_digest,
            "capture_authority_profile": capture_authority_profile,
        },
        "requested_claim_types": sorted({_text(item) for item in requested_claim_types if _text(item)}),
        "required_representations": required,
        "selected_methods": selected,
        "missing_representations": missing,
        "estimated_cost_usd": round(sum(item["expected_cost_usd"] for item in selected), 6),
        "status": "planned" if not missing else "partial_plan",
        "proof_boundary": {
            "provider_availability_is_qualification": False,
            "generated_completion_upgrades_metric_or_physics_claims": False,
            "physical_task_success_established": False,
        },
    }
    plan["reconstruction_plan_digest"] = canonical_digest(
        plan, digest_field="reconstruction_plan_digest"
    )
    return plan


def normalize_reconstruction_result(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _clone(dict(value))
    errors: list[str] = []
    supplied_schema = result.get("schema_version")
    supplied_digest = result.get("reconstruction_result_digest")
    if supplied_schema not in (None, RESULT_SCHEMA_VERSION):
        errors.append("schema_version:mismatch")
    result["schema_version"] = RESULT_SCHEMA_VERSION
    for key in ("result_id", "method_id", "method_version", "provider_identity", "runtime_identity"):
        if not _text(result.get(key)):
            errors.append(f"{key}:missing")
    for key in (
        "capture_digest",
        "method_profile_digest",
        "implementation_digest",
        "runtime_digest",
    ):
        if not _is_digest(result.get(key)):
            errors.append(f"{key}:invalid")
    if not _text(result.get("intake_id")):
        errors.append("intake_id:missing")
    outputs = _strings(result.get("outputs"))
    if not outputs or any(item not in REPRESENTATIONS for item in outputs):
        errors.append("outputs:missing_or_unsupported")
    result["outputs"] = outputs
    for key in (
        "source_frames",
        "camera_solution",
        "coordinate_system",
        "asset_references",
        "coverage_map",
        "uncertainty_map",
        "validation_metrics",
        "rights_and_retention",
        "claim_ceiling",
    ):
        if not isinstance(result.get(key), Mapping):
            errors.append(f"{key}:missing_or_invalid")
    asset_references = result.get("asset_references")
    if isinstance(asset_references, Mapping):
        for key, reference in asset_references.items():
            if not isinstance(reference, Mapping) or not _text(reference.get("uri")) or not _is_digest(
                reference.get("digest")
            ):
                errors.append(f"asset_references.{key}:binding_invalid")
    for key in ("observed_regions", "generated_regions", "invalid_regions"):
        if not isinstance(result.get(key), list):
            errors.append(f"{key}:must_be_list")
    generated = _rows(result.get("generated_regions"))
    if generated and not all(_text(row.get("mask_reference")) for row in generated):
        errors.append("generated_regions:mask_reference_required")
    if generated and any(item in outputs for item in GENERATED_FORBIDDEN_CLAIM_OUTPUTS):
        if result.get("claim_ceiling", {}).get("generated_regions_excluded_from_physics") is not True:
            errors.append("generated_regions:physics_exclusion_required")
    trajectory_intersections = [
        row for row in generated if row.get("intersects_planned_trajectory") is True
    ]
    if trajectory_intersections:
        if any(item in outputs for item in GENERATED_FORBIDDEN_CLAIM_OUTPUTS):
            errors.append("generated_regions:trajectory_intersection_forbids_physics_output")
        claim_ceiling = result.get("claim_ceiling")
        if isinstance(claim_ceiling, dict):
            if claim_ceiling.get("trajectory_clearance_established") is True:
                errors.append("generated_regions:trajectory_clearance_claim_forbidden")
            claim_ceiling["trajectory_clearance_established"] = False
            claim_ceiling["generated_trajectory_intersection_physics_use"] = False
        next_experiment = result.get("next_cheapest_experiment")
        allowed_experiments = {
            "targeted_recapture",
            "owner_measurement",
            "verified_asset",
            "targeted_physical_evidence",
            "abstention",
            "targeted_recapture_or_verified_asset",
        }
        if (
            not isinstance(next_experiment, Mapping)
            or _text(next_experiment.get("kind")) not in allowed_experiments
        ):
            errors.append("generated_regions:trajectory_intersection_experiment_required")
        result["generated_trajectory_intersection"] = {
            "intersects_planned_trajectory": True,
            "region_ids": sorted(
                _text(row.get("region_id"))
                for row in trajectory_intersections
                if _text(row.get("region_id"))
            ),
            "physics_use_allowed": False,
        }
    cost = _number(result.get("cost_usd"))
    duration = _number(result.get("duration_seconds"))
    if cost is None:
        errors.append("cost_usd:invalid")
    else:
        result["cost_usd"] = cost
    if duration is None:
        errors.append("duration_seconds:invalid")
    else:
        result["duration_seconds"] = duration
    result.setdefault("provider_receipt", None)
    result.setdefault("deletion_evidence", None)
    errors.extend(_reject_secrets(result))
    expected_digest = canonical_digest(result, digest_field="reconstruction_result_digest")
    if supplied_digest is not None and supplied_digest != expected_digest:
        errors.append("reconstruction_result_digest:mismatch")
    if errors:
        raise ReconstructionContractError(errors)
    result["reconstruction_result_digest"] = expected_digest
    return result


def decide_simready_assets(
    *,
    approved_task_digest: str,
    capture_digest: str,
    requested_claim_types: Sequence[str],
    task_objects: Sequence[Mapping[str, Any]],
    asset_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not _is_digest(approved_task_digest) or not _is_digest(capture_digest):
        raise ReconstructionContractError(["simready_source_binding:invalid"])
    claims = sorted({_text(item) for item in requested_claim_types if _text(item)})
    required = bool(set(claims).intersection(PHYSICS_DEPENDENT_CLAIMS))
    assets = _rows(asset_candidates)
    decisions: list[dict[str, Any]] = []
    for task_object in sorted(_rows(list(task_objects)), key=lambda item: _text(item.get("object_id"))):
        object_id = _text(task_object.get("object_id"))
        if not object_id:
            raise ReconstructionContractError(["task_objects.object_id:missing"])
        matching = [asset for asset in assets if _text(asset.get("object_id")) == object_id]
        verified: list[dict[str, Any]] = []
        for asset in matching:
            validation = asset.get("independent_validation")
            validator = _text(asset.get("validator_identity"))
            provider = _text(asset.get("provider_identity"))
            validation_passed = isinstance(validation, Mapping) and all(
                validation.get(key) is True
                for key in (
                    "scale",
                    "site_to_object_transform",
                    "support_surface",
                    "orientation",
                    "penetration",
                    "reprojection",
                    "physics_properties",
                )
            )
            if (
                asset.get("validation_status") == "passed"
                and asset.get("source_capture_digest") == capture_digest
                and _is_digest(asset.get("asset_digest"))
                and asset.get("generated_only") is not True
                and validator
                and validator != provider
                and validation_passed
                and _text(asset.get("asset_uri"))
                and isinstance(asset.get("site_to_object_transform"), list)
                and len(asset.get("site_to_object_transform", [])) == 16
                and all(
                    not isinstance(item, bool)
                    and isinstance(item, (int, float))
                    and math.isfinite(float(item))
                    for item in asset.get("site_to_object_transform", [])
                )
            ):
                verified.append(asset)
        verified.sort(key=lambda item: (_text(item.get("asset_digest")), _text(item.get("asset_uri"))))
        selected = verified[0] if verified else None
        decisions.append(
            {
                "object_id": object_id,
                "required": required,
                "status": (
                    "verified_asset_selected"
                    if selected
                    else ("required_missing" if required else "not_required")
                ),
                "selected_asset": _clone(selected) if selected else None,
                "rejected_asset_digests": sorted(
                    _text(asset.get("asset_digest"))
                    for asset in matching
                    if asset not in verified and _text(asset.get("asset_digest"))
                ),
                "required_validation": [
                    "scale",
                    "site_to_object_transform",
                    "support_surface",
                    "orientation",
                    "penetration",
                    "reprojection",
                    "independent_physics_properties",
                ] if required else [],
            }
        )
    artifact = {
        "schema_version": SIMREADY_DECISION_SCHEMA_VERSION,
        "approved_task_digest": approved_task_digest,
        "capture_digest": capture_digest,
        "requested_claim_types": claims,
        "object_decisions": decisions,
        "status": "blocked_missing_asset" if any(
            row["status"] == "required_missing" for row in decisions
        ) else "complete",
        "proof_boundary": {
            "visual_realism_proves_physics": False,
            "provider_output_self_qualifies": False,
            "physical_task_success_established": False,
        },
    }
    artifact["simready_decision_digest"] = canonical_digest(
        artifact, digest_field="simready_decision_digest"
    )
    return artifact


def score_robot_placements(
    *,
    robot_binding: Mapping[str, Any],
    approved_task_digest: str,
    capture_digest: str,
    task_object_id: str,
    target_region_id: str,
    candidates: Sequence[Mapping[str, Any]],
    minimum_coverage: float = 0.8,
) -> dict[str, Any]:
    binding = _clone(dict(robot_binding))
    errors: list[str] = []
    if not _is_digest(approved_task_digest) or not _is_digest(capture_digest):
        errors.append("robot_placement_source_binding:invalid")
    for key in ("robot_id", "embodiment_version", "controller_id", "end_effector_id"):
        if not _text(binding.get(key)):
            errors.append(f"robot_binding.{key}:missing")
    if not isinstance(binding.get("base_footprint"), Mapping) or not binding.get("base_footprint"):
        errors.append("robot_binding.base_footprint:missing")
    if not isinstance(binding.get("sensors"), Mapping) or not binding.get("sensors"):
        errors.append("robot_binding.sensors:missing")
    if not _text(task_object_id) or not _text(target_region_id):
        errors.append("task_object_or_target_region:missing")
    coverage_threshold = _number(minimum_coverage, maximum=1.0)
    if coverage_threshold is None:
        errors.append("minimum_coverage:invalid")
        coverage_threshold = 0.8
    errors.extend(_reject_secrets(binding))
    if errors:
        raise ReconstructionContractError(errors)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    required_bools = (
        "floor_support_valid",
        "footprint_clear",
        "access_path_clear",
        "collision_free",
        "reset_feasible",
        "human_clearance_valid",
    )
    score_keys = (
        "reachability_score",
        "manipulability_score",
        "sensor_visibility_score",
        "approach_direction_score",
        "cable_controller_score",
        "stability_score",
    )
    for raw in sorted(_rows(list(candidates)), key=lambda item: _text(item.get("candidate_id"))):
        candidate = _clone(raw)
        candidate_id = _text(candidate.get("candidate_id"))
        reasons: list[str] = []
        if not candidate_id:
            reasons.append("candidate_id_missing")
        transform = candidate.get("site_from_robot_base")
        if (
            not isinstance(transform, list)
            or len(transform) != 16
            or not all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                for item in transform
            )
        ):
            reasons.append("site_from_robot_base_invalid")
        evidence_digests = _strings(candidate.get("evidence_digests"))
        if not evidence_digests or any(not _is_digest(item) for item in evidence_digests):
            reasons.append("evidence_digests_missing_or_invalid")
        candidate["evidence_digests"] = evidence_digests
        if candidate.get("method_qualification_status") not in {
            "qualified",
            "analytic_only",
        }:
            reasons.append("placement_method_not_qualified")
        for key in required_bools:
            if candidate.get(key) is not True:
                reasons.append(f"{key}_failed")
        coverage = _number(candidate.get("captured_coverage"), maximum=1.0)
        if coverage is None or coverage < coverage_threshold:
            reasons.append("captured_coverage_insufficient")
        scores = [_number(candidate.get(key), maximum=1.0) for key in score_keys]
        if any(score is None for score in scores):
            reasons.append("candidate_score_invalid")
        uncertainty = _number(candidate.get("calibration_uncertainty_m"))
        if uncertainty is None:
            reasons.append("calibration_uncertainty_invalid")
        if reasons:
            rejected.append({"candidate_id": candidate_id or "invalid", "reasons": sorted(reasons)})
            continue
        candidate["score"] = round(
            sum(float(score) for score in scores if score is not None) / len(score_keys)
            - min(float(uncertainty or 0.0), 1.0) * 0.1,
            9,
        )
        accepted.append(candidate)
    accepted.sort(key=lambda item: (-item["score"], item["candidate_id"]))
    selected = accepted[0] if accepted else None
    artifact = {
        "schema_version": ROBOT_PLACEMENT_SCHEMA_VERSION,
        "robot_binding": binding,
        "robot_binding_digest": canonical_digest(binding),
        "approved_task_digest": approved_task_digest,
        "capture_digest": capture_digest,
        "task_object_id": task_object_id,
        "target_region_id": target_region_id,
        "minimum_coverage": coverage_threshold,
        "accepted_candidates": accepted,
        "rejected_candidates": rejected,
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "status": "candidate_selected" if selected else "abstained",
        "next_cheapest_experiment": None if selected else {
            "kind": "targeted_recapture_or_measurement",
            "instruction": "Capture the complete proposed robot placement area, access path, support surface, approach direction, and human clearance, then re-score exact candidates.",
        },
        "proof_boundary": {
            "placement_is_physical_deployment_approval": False,
            "coverage_gap_becomes_pass": False,
            "physical_task_success_established": False,
        },
    }
    artifact["robot_placement_digest"] = canonical_digest(
        artifact, digest_field="robot_placement_digest"
    )
    return artifact


__all__ = [
    "ReconstructionContractError",
    "build_reconstruction_method_profile",
    "plan_reconstruction_methods",
    "normalize_reconstruction_result",
    "decide_simready_assets",
    "score_robot_placements",
]
