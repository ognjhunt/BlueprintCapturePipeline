"""Provider-neutral new-site Task Evaluation Run compiler.

This module is the deterministic seam between capture/reconstruction support,
3D target authorization, robot placement, task/site measurement routing, and a
five-candidate learned-policy comparison.  Semantic agents and providers may
propose inputs, but only digest-bound qualifications and execution receipts can
advance the run.  Missing evidence produces an ordered, explicit abstention.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_site_measurement_routing import (
    MeasurementRoutingError,
    TASK_CAPABILITIES,
    route_task_site_measurement,
)


REQUEST_SCHEMA_VERSION = "new_site_task_evaluation_request.v1"
RESULT_SCHEMA_VERSION = "new_site_task_evaluation_run.v1"
POLICY_IDENTITY_SCHEMA_VERSION = "learned_policy_candidate_identity.v1"
ATTEMPT_SCHEMA_VERSION = "learned_policy_attempt_receipt.v1"
METRIC_SCHEMA_VERSION = "task_outcome_metric_spec.v1"
AUTHORIZATION_SCHEMA_VERSION = "new_site_policy_execution_authorization.v1"
EXPECTED_POLICY_CANDIDATES = 5

_HUMANOID_TASK_CLASSES = {
    "humanoid_locomotion",
    "humanoid_mobile_manipulation",
    "whole_body_humanoid",
}
_LEARNED_ACTION_SOURCES = {"learned_policy", "policy_endpoint", "vla_policy"}
_RENDERED_TARGET_SCHEMA_VERSION = "rendered_scene_task_target_orchestration.v1"
_RENDERED_TARGET_ANALYSIS_SCHEMA_VERSION = "scene_task_target_analysis_result.v1"
_SPLAT_TARGET_BINDING_SCHEMA_VERSION = "splat_bbox_target_binding_result.v1"
_TASK_ZONE_REQUIREMENT_SCHEMA_VERSION = "task_zone_asset_requirement_candidate.v1"
_EXTERNAL_PLACEMENT_SCHEMA_VERSION = "external_scene_robot_placement_candidate.v1"
_ANALYZER_TASK_FAMILY_BINDINGS = {
    "franka_small_object_pick": ("rigid_pick_place", "franka_panda"),
    "franka_utensil_pick": ("rigid_pick_place", "franka_panda"),
    "franka_tool_pick": ("rigid_pick_place", "franka_panda"),
    "franka_work_surface_inspection": ("visual_perception", "franka_panda"),
    "franka_surface_inspection": ("visual_perception", "franka_panda"),
    "franka_sink_inspection": ("visual_perception", "franka_panda"),
    "franka_appliance_interaction": ("doors_drawers_handles", "franka_panda"),
    "g1_object_retrieval": ("mobile_manipulation_clutter", "unitree_g1"),
    "g1_obstacle_aware_navigation": ("locomotion", "unitree_g1"),
    "g1_work_surface_inspection": ("visual_navigation_active_perception", "unitree_g1"),
    "g1_sink_inspection": ("visual_navigation_active_perception", "unitree_g1"),
    "g1_appliance_interaction": ("mobile_manipulation_clutter", "unitree_g1"),
}
_HUMANOID_TASK_FAMILIES = {
    family
    for family, (_, robot_id) in _ANALYZER_TASK_FAMILY_BINDINGS.items()
    if robot_id == "unitree_g1"
}


class NewSiteTaskEvaluationError(ValueError):
    """Stable validation failure for malformed or replayed run inputs."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NewSiteTaskEvaluationError(["new_site_value_not_json_serializable"]) from exc


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _parse_timestamp(value: Any, *, code: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise NewSiteTaskEvaluationError([code]) from exc
    if parsed.tzinfo is None:
        raise NewSiteTaskEvaluationError([code])
    return parsed


def _validate_canonical_artifact(
    value: Any,
    *,
    label: str,
    digest_field: str,
    accepted_schemas: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NewSiteTaskEvaluationError([f"{label}_invalid"])
    artifact = _clone(dict(value))
    if accepted_schemas is not None and artifact.get("schema_version") not in accepted_schemas:
        raise NewSiteTaskEvaluationError([f"{label}_schema_invalid"])
    if artifact.get(digest_field) != canonical_digest(artifact, digest_field=digest_field):
        raise NewSiteTaskEvaluationError([f"{label}_digest_mismatch"])
    return artifact


def _abstention(
    *,
    run_id: str,
    source_profile_digest: str | None,
    code: str,
    instruction: str,
    stage: str,
    blockers: Sequence[str] | None = None,
    routing_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "abstained",
        "source_profile_digest": source_profile_digest,
        "terminal_stage": stage,
        "blockers": sorted(set(blockers or [code])),
        "smallest_missing_measurement": {
            "code": code,
            "instruction": instruction,
            "stage": stage,
        },
        "routing_decision": _clone(routing_decision) if routing_decision else None,
        "robot_id": None,
        "policy_attempt_count": 0,
        "supported_ranking_candidate_count": 0,
        "ranking": [],
        "claim_boundary": {
            "task_evaluation_run_completed": False,
            "learned_policy_attempts_observed": False,
            "learned_policy_ranking_proven": False,
            "physical_success_proven": False,
            "deployment_readiness_proven": False,
        },
    }
    result["run_digest"] = canonical_digest(result, digest_field="run_digest")
    return result


def _source_gate(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    source = request.get("source_profile")
    if not isinstance(source, Mapping):
        return {}, {
            "code": "source_profile_missing",
            "instruction": "Compile and supply a digest-bound capture source profile.",
        }
    source = _validate_canonical_artifact(
        source,
        label="source_profile",
        digest_field="source_profile_digest",
    )
    status = str(source.get("status") or "")
    if status not in {"admitted_provider_derived_support", "admitted_blueprint_raw_contract"}:
        smallest = source.get("smallest_missing_measurement")
        smallest = dict(smallest) if isinstance(smallest, Mapping) else {}
        return source, {
            "code": str(smallest.get("code") or "source_profile_not_admitted"),
            "instruction": str(
                smallest.get("instruction")
                or "Resolve the source-profile blockers against the preserved capture."
            ),
        }
    boundary = source.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        raise NewSiteTaskEvaluationError(["source_profile_claim_boundary_missing"])
    if status == "admitted_provider_derived_support":
        if (
            boundary.get("provider_derived_support") is not True
            or boundary.get("blueprint_raw_contract_truth") is not False
        ):
            raise NewSiteTaskEvaluationError(["provider_source_truth_boundary_invalid"])
    elif (
        boundary.get("provider_derived_support") is not False
        or boundary.get("blueprint_raw_contract_truth") is not True
    ):
        raise NewSiteTaskEvaluationError(["blueprint_raw_source_truth_boundary_invalid"])
    return source, None


def _reconstruction_gate(value: Any) -> tuple[dict[str, Any], dict[str, str] | None]:
    if not isinstance(value, Mapping):
        return {}, {
            "code": "registered_reconstruction_missing",
            "instruction": "Generate a native 3DGS and register it to the capture geometry frame.",
        }
    reconstruction = _clone(dict(value))
    for field in (
        "appearance_asset_digest",
        "geometry_asset_digest",
        "scene_registration_digest",
        "reconstruction_digest",
    ):
        if not _is_digest(reconstruction.get(field)):
            raise NewSiteTaskEvaluationError([f"reconstruction_{field}_invalid"])
    if reconstruction.get("source_scene_digest") is not None and not _is_digest(
        reconstruction.get("source_scene_digest")
    ):
        raise NewSiteTaskEvaluationError(["reconstruction_source_scene_digest_invalid"])
    if reconstruction.get("reconstruction_digest") != canonical_digest(
        reconstruction, digest_field="reconstruction_digest"
    ):
        raise NewSiteTaskEvaluationError(["reconstruction_digest_mismatch"])
    if reconstruction.get("appearance_format") != "native_3dgs":
        return reconstruction, {
            "code": "native_3dgs_appearance_missing",
            "instruction": "Export the same capture's full-resolution native 3DGS PLY.",
        }
    if reconstruction.get("full_resolution_appearance_preserved") is not True:
        return reconstruction, {
            "code": "full_resolution_appearance_truth_missing",
            "instruction": "Preserve and bind the full-resolution native 3DGS appearance artifact.",
        }
    if reconstruction.get("registration_status") != "qualified":
        return reconstruction, {
            "code": "splat_metric_frame_registration_missing",
            "instruction": "Measure and qualify the splat-to-metric-geometry frame transform.",
        }
    if reconstruction.get("presentation_output_used_as_evaluation_evidence") is not False:
        raise NewSiteTaskEvaluationError(["presentation_output_used_as_evaluation_evidence"])
    return reconstruction, None


def _target_gate(
    value: Any,
    *,
    reconstruction_digest: str,
    source_scene_digest: str,
    appearance_asset_digest: str,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    if not isinstance(value, Mapping):
        return {}, {
            "code": "automatic_task_target_proposal_missing",
            "instruction": "Run the registered-view analyzer and deterministic 3D target binder.",
        }
    target_schema = str(value.get("schema_version") or "")
    if target_schema == _RENDERED_TARGET_SCHEMA_VERSION:
        rendered = _validate_canonical_artifact(
            value,
            label="target_orchestration",
            digest_field="orchestration_digest",
            accepted_schemas={_RENDERED_TARGET_SCHEMA_VERSION},
        )
        if (
            rendered.get("source_scene_digest") != source_scene_digest
            or rendered.get("analysis_splat_digest") != appearance_asset_digest
        ):
            raise NewSiteTaskEvaluationError(["target_orchestration_reconstruction_mismatch"])
        if rendered.get("candidate_may_self_authorize") is not False:
            raise NewSiteTaskEvaluationError(["selected_target_self_authorization_forbidden"])
        analysis = _validate_canonical_artifact(
            rendered.get("target_analysis"),
            label="target_analysis",
            digest_field="target_analysis_digest",
            accepted_schemas={_RENDERED_TARGET_ANALYSIS_SCHEMA_VERSION},
        )
        if analysis.get("source_scene_digest") != source_scene_digest:
            raise NewSiteTaskEvaluationError(["target_orchestration_reconstruction_mismatch"])
        selected = analysis.get("selected_target")
        selected = dict(selected) if isinstance(selected, Mapping) else None
        binding_rows = rendered.get("binding_results")
        if not isinstance(binding_rows, list):
            raise NewSiteTaskEvaluationError(["rendered_target_binding_results_invalid"])
        matching_bindings = [
            dict(row)
            for row in binding_rows
            if isinstance(row, Mapping)
            and selected is not None
            and row.get("proposal_id") == selected.get("proposal_id")
        ]
        if selected is not None and len(matching_bindings) != 1:
            raise NewSiteTaskEvaluationError(["selected_target_binding_result_ambiguous"])
        binding_result = matching_bindings[0] if matching_bindings else None
        if binding_result is not None and (
            binding_result.get("status") != "candidate_bound"
            or binding_result.get("blockers") not in ([], ())
        ):
            raise NewSiteTaskEvaluationError(["selected_target_binding_result_not_admitted"])
        binding = (
            _validate_canonical_artifact(
                binding_result.get("binding"),
                label="target_binding",
                digest_field="binding_evidence_digest",
                accepted_schemas={_SPLAT_TARGET_BINDING_SCHEMA_VERSION},
            )
            if isinstance(binding_result, Mapping)
            else {}
        )
        if binding and (
            binding.get("source_scene_digest") != source_scene_digest
            or binding.get("analysis_splat_digest") != appearance_asset_digest
        ):
            raise NewSiteTaskEvaluationError(["selected_target_binding_reconstruction_mismatch"])
        task_zone_requirement = _validate_canonical_artifact(
            rendered.get("task_zone_asset_requirement"),
            label="task_zone_asset_requirement",
            digest_field="requirement_digest",
            accepted_schemas={_TASK_ZONE_REQUIREMENT_SCHEMA_VERSION},
        )
        if task_zone_requirement.get("authoritative_asset_selection_performed") is not False:
            raise NewSiteTaskEvaluationError(["task_zone_candidate_self_authorization_forbidden"])
        task_family = str((selected or {}).get("task_family") or "").strip()
        supplied_task_class = str((selected or {}).get("task_class") or "").strip()
        family_binding = _ANALYZER_TASK_FAMILY_BINDINGS.get(task_family)
        if supplied_task_class:
            if supplied_task_class not in TASK_CAPABILITIES:
                raise NewSiteTaskEvaluationError(["rendered_target_task_class_unknown"])
            task_class = supplied_task_class
        elif family_binding is not None:
            task_class = family_binding[0]
        else:
            raise NewSiteTaskEvaluationError(["rendered_target_task_family_unmapped"])
        analysis_robot = str(analysis.get("robot_id") or "").strip()
        if analysis_robot not in {"franka_panda", "unitree_g1"}:
            raise NewSiteTaskEvaluationError(["rendered_target_robot_binding_invalid"])
        if family_binding is not None and analysis_robot != family_binding[1]:
            raise NewSiteTaskEvaluationError(["rendered_target_robot_family_mismatch"])
        normalized_selected = (
            {
                **selected,
                "task_family": task_family,
                "task_class": task_class,
                "required_embodiment": analysis_robot,
                "target_binding_digest": binding.get("binding_evidence_digest"),
                "candidate_self_authorized": False,
            }
            if selected is not None
            else None
        )
        target = {
            "schema_version": target_schema,
            "status": rendered.get("status"),
            "reconstruction_digest": reconstruction_digest,
            "analysis_appearance_digest": appearance_asset_digest,
            "selected_target": normalized_selected,
            "task_zone_asset_requirement": task_zone_requirement,
            "target_orchestration_digest": rendered["orchestration_digest"],
        }
        if task_zone_requirement.get("status") in {
            "abstained_interaction_mode_ambiguous",
            "abstained_no_selected_target",
        }:
            return target, {
                "code": "task_zone_interaction_mode_unresolved",
                "instruction": (
                    "Bind an explicit inspection, contact, articulation, or object-state "
                    "interaction mode before scene composition."
                ),
            }
        requirement_pair = (
            task_zone_requirement.get("status"),
            task_zone_requirement.get("verified_simready_asset_required"),
        )
        if requirement_pair not in {
            ("not_required_for_inspection_only", False),
            ("verified_task_zone_asset_required", True),
        }:
            raise NewSiteTaskEvaluationError(["task_zone_requirement_status_invalid"])
    else:
        target = _validate_canonical_artifact(
            value,
            label="target_orchestration",
            digest_field="target_orchestration_digest",
        )
    if (
        target.get("reconstruction_digest") != reconstruction_digest
        or target.get("analysis_appearance_digest") != appearance_asset_digest
    ):
        raise NewSiteTaskEvaluationError(["target_orchestration_reconstruction_mismatch"])
    if target.get("status") not in {
        "target_ready_for_bounded_sim",
        "selected_target_ready",
    }:
        smallest = target.get("smallest_missing_measurement")
        smallest = dict(smallest) if isinstance(smallest, Mapping) else {}
        return target, {
            "code": str(smallest.get("code") or "no_qualified_3d_task_target"),
            "instruction": str(
                smallest.get("instruction")
                or "Capture or reconstruct enough supporting views to bind one task target in 3D."
            ),
        }
    selected = target.get("selected_target")
    if not isinstance(selected, Mapping):
        raise NewSiteTaskEvaluationError(["selected_target_missing"])
    for field in ("proposal_id", "task_family", "task_class"):
        if not str(selected.get(field) or "").strip():
            raise NewSiteTaskEvaluationError([f"selected_target_{field}_missing"])
    if not _is_digest(selected.get("target_binding_digest")):
        raise NewSiteTaskEvaluationError(["selected_target_binding_digest_invalid"])
    if selected.get("candidate_self_authorized") is not False:
        raise NewSiteTaskEvaluationError(["selected_target_self_authorization_forbidden"])
    return target, None


def _expected_robot(selected_target: Mapping[str, Any]) -> str:
    task_class = str(selected_target.get("task_class") or "")
    task_family = str(selected_target.get("task_family") or "")
    embodiment = str(selected_target.get("required_embodiment") or "").strip()
    if embodiment and embodiment not in {"franka_panda", "unitree_g1"}:
        raise NewSiteTaskEvaluationError(["selected_target_embodiment_unsupported"])
    humanoid = task_class in _HUMANOID_TASK_CLASSES or task_family in _HUMANOID_TASK_FAMILIES
    if embodiment == "unitree_g1" and not humanoid:
        raise NewSiteTaskEvaluationError(["g1_requires_humanoid_task"])
    if embodiment == "franka_panda" and humanoid:
        raise NewSiteTaskEvaluationError(["franka_for_humanoid_task_forbidden"])
    if humanoid:
        return "unitree_g1"
    return "franka_panda"


def _placement_gate(
    value: Any, *, expected_robot: str, target_binding_digest: str
) -> tuple[dict[str, Any], dict[str, str] | None]:
    if not isinstance(value, Mapping):
        return {}, {
            "code": "qualified_robot_placement_missing",
            "instruction": f"Qualify a collision-aware {expected_robot} placement for the selected target.",
        }
    placement_schema = str(value.get("schema_version") or "")
    if placement_schema == _EXTERNAL_PLACEMENT_SCHEMA_VERSION:
        external = _validate_canonical_artifact(
            value,
            label="robot_placement",
            digest_field="placement_proposal_digest",
            accepted_schemas={_EXTERNAL_PLACEMENT_SCHEMA_VERSION},
        )
        if external.get("status") not in {"runtime_visualization_candidate_only", "abstained"}:
            raise NewSiteTaskEvaluationError(["external_robot_placement_status_invalid"])
        placement = {
            **external,
            "status": "unqualified",
            "placement_digest": external["placement_proposal_digest"],
        }
    else:
        placement = _validate_canonical_artifact(
            value,
            label="robot_placement",
            digest_field="placement_digest",
        )
    if placement.get("robot_id") != expected_robot:
        raise NewSiteTaskEvaluationError(["robot_default_binding_mismatch"])
    if placement.get("target_binding_digest") != target_binding_digest:
        raise NewSiteTaskEvaluationError(["robot_placement_target_binding_mismatch"])
    if placement.get("status") != "qualified":
        return placement, {
            "code": "qualified_robot_placement_missing",
            "instruction": f"Measure reach and footprint clearance for the proposed {expected_robot} pose.",
        }
    return placement, None


def _scene_composition_gate(
    value: Any, *, task_zone_required: bool
) -> tuple[dict[str, Any], dict[str, str] | None]:
    if not isinstance(value, Mapping):
        return {}, {
            "code": "scene_composition_qualification_missing",
            "instruction": "Declare whether qualified floor/support or task-zone composition is required.",
        }
    composition = _clone(dict(value))
    floor = composition.get("floor_support_mount")
    zone = composition.get("task_zone_replacement")
    if not isinstance(floor, Mapping) or not isinstance(zone, Mapping):
        raise NewSiteTaskEvaluationError(["scene_composition_records_invalid"])
    floor_status = floor.get("status")
    if floor_status not in {"not_required", "qualified"}:
        return composition, {
            "code": "qualified_floor_support_mount_missing",
            "instruction": "Qualify a bounded floor/support mount or demonstrate that the source collider is sufficient.",
        }
    if floor_status == "qualified" and not _is_digest(floor.get("qualification_digest")):
        raise NewSiteTaskEvaluationError(["floor_support_qualification_digest_invalid"])
    zone_status = zone.get("status")
    if task_zone_required and zone_status != "qualified":
        return composition, {
            "code": "qualified_simready_task_zone_missing",
            "instruction": "Insert and qualify a digest-bound SimReady task-zone replacement for the selected interaction.",
        }
    if not task_zone_required and zone_status != "not_required":
        raise NewSiteTaskEvaluationError(["unrequired_task_zone_replacement_forbidden"])
    if zone_status == "qualified" and not _is_digest(zone.get("qualification_digest")):
        raise NewSiteTaskEvaluationError(["task_zone_qualification_digest_invalid"])
    composition_digest = composition.get("scene_composition_digest")
    if composition_digest != canonical_digest(composition, digest_field="scene_composition_digest"):
        raise NewSiteTaskEvaluationError(["scene_composition_digest_mismatch"])
    return composition, None


def _route_gate(
    value: Any,
    *,
    source_profile_digest: str,
    target_binding_digest: str,
    placement_digest: str,
    robot_id: str,
    task_class: str,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    if not isinstance(value, Mapping):
        return {}, {
            "code": "task_site_engine_route_inputs_missing",
            "instruction": "Supply the task requirements, site evidence, method catalog, and qualifications.",
        }
    route_inputs = _clone(dict(value))
    required = {
        "requirements",
        "site_evidence_profile",
        "method_capability_profiles",
        "measurement_qualifications",
        "catalog_snapshot_hash",
        "routing_as_of",
        "source_profile_digest",
        "target_binding_digest",
        "placement_digest",
        "robot_id",
        "task_class",
    }
    if not required.issubset(route_inputs):
        return {}, {
            "code": "task_site_engine_route_inputs_missing",
            "instruction": "Complete every deterministic measurement-routing input.",
        }
    expected_bindings = {
        "source_profile_digest": source_profile_digest,
        "target_binding_digest": target_binding_digest,
        "placement_digest": placement_digest,
        "robot_id": robot_id,
        "task_class": task_class,
    }
    if any(route_inputs.get(key) != expected for key, expected in expected_bindings.items()):
        raise NewSiteTaskEvaluationError(["task_site_engine_route_scope_mismatch"])
    requirements = route_inputs.get("requirements")
    if not isinstance(requirements, Mapping):
        raise NewSiteTaskEvaluationError(["task_site_engine_requirements_invalid"])
    robot_scope = requirements.get("robot_scope")
    robot_scope = dict(robot_scope) if isinstance(robot_scope, Mapping) else {}
    if requirements.get("task_class") != task_class or robot_scope.get("robot_id") != robot_id:
        raise NewSiteTaskEvaluationError(["task_site_engine_requirements_scope_mismatch"])
    try:
        routing_as_of = date.fromisoformat(str(route_inputs["routing_as_of"]))
        decision = route_task_site_measurement(
            route_inputs["requirements"],
            route_inputs["site_evidence_profile"],
            route_inputs["method_capability_profiles"],
            route_inputs["measurement_qualifications"],
            catalog_snapshot_hash=str(route_inputs["catalog_snapshot_hash"]),
            as_of=routing_as_of,
        )
    except (MeasurementRoutingError, TypeError, ValueError) as exc:
        raise NewSiteTaskEvaluationError(["task_site_engine_route_inputs_invalid"]) from exc
    if decision.get("status") != "route_selected":
        abstention = decision.get("abstention")
        abstention = dict(abstention) if isinstance(abstention, Mapping) else {}
        action = abstention.get("smallest_next_action")
        action = dict(action) if isinstance(action, Mapping) else {}
        exact_scope = action.get("exact_scope")
        exact_scope = list(exact_scope) if isinstance(exact_scope, list) else []
        code = str(
            exact_scope[0]
            if exact_scope
            else action.get("action_type") or "no_exact_qualified_measurement_route"
        )
        return decision, {
            "code": code,
            "instruction": "Collect the router's smallest missing site/task measurement and replay qualification.",
        }
    if (
        decision.get("agent_selected_route") is not False
        or decision.get("agent_qualified_method") is not False
    ):
        raise NewSiteTaskEvaluationError(["agent_authorized_engine_route_forbidden"])
    return decision, None


def _metric_spec(value: Any) -> tuple[dict[str, Any], datetime]:
    metric = _validate_canonical_artifact(
        value,
        label="task_metric",
        digest_field="metric_spec_digest",
        accepted_schemas={METRIC_SCHEMA_VERSION},
    )
    if metric.get("direction") not in {"maximize", "minimize"}:
        raise NewSiteTaskEvaluationError(["task_metric_direction_invalid"])
    if not str(metric.get("metric_id") or "").strip() or not str(metric.get("units") or "").strip():
        raise NewSiteTaskEvaluationError(["task_metric_identity_invalid"])
    if metric.get("fixed_before_execution") is not True:
        raise NewSiteTaskEvaluationError(["task_metric_not_fixed_before_execution"])
    frozen_at = _parse_timestamp(metric.get("frozen_at"), code="task_metric_frozen_at_invalid")
    return metric, frozen_at


def _policy_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != EXPECTED_POLICY_CANDIDATES:
        raise NewSiteTaskEvaluationError(["exactly_five_learned_policy_candidates_required"])
    candidates: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    identity_digests: set[str] = set()
    immutable_policy_references: set[tuple[str, str]] = set()
    contract_triples: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(value):
        candidate = _validate_canonical_artifact(
            raw,
            label=f"policy_candidate_{index}",
            digest_field="policy_identity_digest",
            accepted_schemas={POLICY_IDENTITY_SCHEMA_VERSION},
        )
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in identifiers:
            raise NewSiteTaskEvaluationError(["policy_candidate_identity_duplicate_or_missing"])
        if candidate.get("candidate_kind") != "learned_policy":
            raise NewSiteTaskEvaluationError(["scripted_controller_candidate_forbidden"])
        immutable = [
            candidate.get("checkpoint_digest"),
            candidate.get("endpoint_identity_digest"),
        ]
        if sum(_is_digest(row) for row in immutable) != 1:
            raise NewSiteTaskEvaluationError(["policy_candidate_immutable_identity_invalid"])
        immutable_reference = (
            ("checkpoint", str(candidate["checkpoint_digest"]))
            if _is_digest(candidate.get("checkpoint_digest"))
            else ("endpoint", str(candidate["endpoint_identity_digest"]))
        )
        if immutable_reference in immutable_policy_references:
            raise NewSiteTaskEvaluationError(["policy_candidate_immutable_identity_duplicate"])
        immutable_policy_references.add(immutable_reference)
        for field in (
            "runtime_digest",
            "observation_schema_digest",
            "action_schema_digest",
            "observation_sequence_spec_digest",
        ):
            if not _is_digest(candidate.get(field)):
                raise NewSiteTaskEvaluationError([f"policy_candidate_{field}_invalid"])
        contract_triples.add(
            (
                candidate["observation_schema_digest"],
                candidate["action_schema_digest"],
                candidate["observation_sequence_spec_digest"],
            )
        )
        identifiers.add(candidate_id)
        identity_digests.add(candidate["policy_identity_digest"])
        candidates.append(candidate)
    if len(identity_digests) != EXPECTED_POLICY_CANDIDATES:
        raise NewSiteTaskEvaluationError(["policy_candidate_identity_digest_duplicate"])
    if len(contract_triples) != 1:
        raise NewSiteTaskEvaluationError(["policy_candidate_observation_action_contract_mismatch"])
    return candidates


def _execution_authorization(
    value: Any,
    *,
    route: Mapping[str, Any],
    placement: Mapping[str, Any],
    metric: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    authorization = _validate_canonical_artifact(
        value,
        label="policy_execution_authorization",
        digest_field="authorization_digest",
        accepted_schemas={AUTHORIZATION_SCHEMA_VERSION},
    )
    expected_candidate_set_digest = canonical_digest(
        {
            "policy_identity_digests": sorted(
                str(row["policy_identity_digest"]) for row in candidates
            )
        }
    )
    errors: list[str] = []
    if authorization.get("policy_execution_authorized") is not True:
        errors.append("policy_execution_not_authorized")
    if authorization.get("routing_decision_digest") != route.get("routing_decision_digest"):
        errors.append("policy_authorization_route_mismatch")
    if authorization.get("placement_digest") != placement.get("placement_digest"):
        errors.append("policy_authorization_placement_mismatch")
    if authorization.get("metric_spec_digest") != metric.get("metric_spec_digest"):
        errors.append("policy_authorization_metric_mismatch")
    if authorization.get("candidate_set_digest") != expected_candidate_set_digest:
        errors.append("policy_authorization_candidate_set_mismatch")
    if authorization.get("physical_robot_execution_authorized") is not False:
        errors.append("policy_authorization_physical_scope_invalid")
    if errors:
        raise NewSiteTaskEvaluationError(errors)
    return authorization


def _attempts(
    value: Any,
    *,
    candidates: Sequence[Mapping[str, Any]],
    metric: Mapping[str, Any],
    metric_frozen_at: datetime,
    route: Mapping[str, Any],
    placement: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if not isinstance(value, list) or len(value) != EXPECTED_POLICY_CANDIDATES:
        raise NewSiteTaskEvaluationError(["exactly_five_learned_policy_attempts_required"])
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates}
    attempts: list[dict[str, Any]] = []
    seen: set[str] = set()
    reset_bindings: set[tuple[str, str]] = set()
    unsupported: list[str] = []
    route_digest = route.get("routing_decision_digest")
    for index, raw in enumerate(value):
        attempt = _validate_canonical_artifact(
            raw,
            label=f"policy_attempt_{index}",
            digest_field="attempt_digest",
            accepted_schemas={ATTEMPT_SCHEMA_VERSION},
        )
        candidate_id = str(attempt.get("candidate_id") or "")
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None or candidate_id in seen:
            raise NewSiteTaskEvaluationError(["policy_attempt_candidate_binding_invalid"])
        seen.add(candidate_id)
        for field in (
            "execution_receipt_digest",
            "matched_reset_digest",
            "initial_state_observation_digest",
            "observation_trace_digest",
            "action_trace_digest",
            "contact_evidence_digest",
            "collision_evidence_digest",
        ):
            if not _is_digest(attempt.get(field)):
                raise NewSiteTaskEvaluationError([f"policy_attempt_{field}_invalid"])
        if attempt.get("policy_identity_digest") != candidate.get("policy_identity_digest"):
            raise NewSiteTaskEvaluationError(["policy_attempt_identity_mismatch"])
        if attempt.get("routing_decision_digest") != route_digest:
            raise NewSiteTaskEvaluationError(["policy_attempt_engine_route_mismatch"])
        if attempt.get("placement_digest") != placement.get("placement_digest"):
            raise NewSiteTaskEvaluationError(["policy_attempt_placement_mismatch"])
        if attempt.get("action_source") not in _LEARNED_ACTION_SOURCES:
            raise NewSiteTaskEvaluationError(["policy_attempt_not_learned_policy"])
        started = _parse_timestamp(
            attempt.get("started_at"), code="policy_attempt_started_at_invalid"
        )
        ended = _parse_timestamp(attempt.get("ended_at"), code="policy_attempt_ended_at_invalid")
        if ended < started or started <= metric_frozen_at:
            raise NewSiteTaskEvaluationError(["task_metric_not_frozen_before_attempt"])
        reset_bindings.add(
            (attempt["matched_reset_digest"], attempt["initial_state_observation_digest"])
        )
        fresh_query_count = attempt.get("fresh_policy_query_count")
        learned_action_count = attempt.get("learned_policy_action_count")
        if (
            isinstance(fresh_query_count, bool)
            or not isinstance(fresh_query_count, int)
            or fresh_query_count < 1
            or isinstance(learned_action_count, bool)
            or not isinstance(learned_action_count, int)
            or learned_action_count < 1
            or attempt.get("learned_policy_action_proven") is not True
            or attempt.get("reset_observed") is not True
        ):
            raise NewSiteTaskEvaluationError(["policy_attempt_real_execution_not_proven"])
        outcome = attempt.get("task_metric_result")
        if not isinstance(outcome, Mapping):
            raise NewSiteTaskEvaluationError(["policy_attempt_metric_result_missing"])
        if outcome.get("metric_spec_digest") != metric.get("metric_spec_digest"):
            raise NewSiteTaskEvaluationError(["policy_attempt_metric_spec_mismatch"])
        value_number = outcome.get("value")
        if (
            isinstance(value_number, bool)
            or not isinstance(value_number, (int, float))
            or not math.isfinite(float(value_number))
        ):
            raise NewSiteTaskEvaluationError(["policy_attempt_metric_value_invalid"])
        blockers = outcome.get("blockers")
        if not isinstance(blockers, list) or not all(isinstance(row, str) for row in blockers):
            raise NewSiteTaskEvaluationError(["policy_attempt_metric_blockers_invalid"])
        supported = bool(
            attempt.get("status") == "completed"
            and outcome.get("supported_for_ranking") is True
            and not blockers
        )
        if not supported:
            unsupported.append(candidate_id)
        attempt["supported_for_ranking"] = supported
        attempts.append(attempt)
    if seen != set(candidate_by_id):
        raise NewSiteTaskEvaluationError(["policy_attempt_candidate_set_mismatch"])
    if len(reset_bindings) != 1:
        raise NewSiteTaskEvaluationError(["policy_attempt_matched_reset_mismatch"])
    supported_attempts = [row for row in attempts if row["supported_for_ranking"]]
    return attempts, supported_attempts, sorted(unsupported)


def _rank(attempts: Sequence[Mapping[str, Any]], *, direction: str) -> list[dict[str, Any]]:
    reverse = direction == "maximize"
    ordered = sorted(
        attempts,
        key=lambda row: (
            -float(dict(row["task_metric_result"])["value"])
            if reverse
            else float(dict(row["task_metric_result"])["value"]),
            str(row["candidate_id"]),
        ),
    )
    ranking: list[dict[str, Any]] = []
    last_value: float | None = None
    rank = 0
    for index, row in enumerate(ordered, start=1):
        value = float(dict(row["task_metric_result"])["value"])
        if last_value is None or value != last_value:
            rank = index
            last_value = value
        ranking.append(
            {
                "rank": rank,
                "candidate_id": row["candidate_id"],
                "policy_identity_digest": row["policy_identity_digest"],
                "attempt_digest": row["attempt_digest"],
                "metric_value": value,
            }
        )
    return ranking


def compile_new_site_task_evaluation_run(value: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one complete new-site run or the smallest fail-closed abstention."""

    if not isinstance(value, Mapping):
        raise NewSiteTaskEvaluationError(["new_site_request_invalid"])
    request = _clone(dict(value))
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise NewSiteTaskEvaluationError(["new_site_request_schema_invalid"])
    supplied_digest = request.pop("request_digest", None)
    if supplied_digest is not None and supplied_digest != canonical_digest(request):
        raise NewSiteTaskEvaluationError(["new_site_request_digest_mismatch"])
    request["request_digest"] = canonical_digest(request)
    run_id = str(request.get("run_id") or "").strip()
    if not run_id:
        raise NewSiteTaskEvaluationError(["new_site_run_id_missing"])

    source, missing = _source_gate(request)
    source_digest = source.get("source_profile_digest") if source else None
    if missing:
        return _abstention(
            run_id=run_id,
            source_profile_digest=source_digest,
            code=missing["code"],
            instruction=missing["instruction"],
            stage="capture_source_admission",
        )
    reconstruction, missing = _reconstruction_gate(request.get("reconstruction"))
    if missing:
        return _abstention(
            run_id=run_id,
            source_profile_digest=source_digest,
            code=missing["code"],
            instruction=missing["instruction"],
            stage="reconstruction_registration",
        )
    if reconstruction.get("source_profile_digest") != source_digest:
        raise NewSiteTaskEvaluationError(["reconstruction_source_profile_mismatch"])
    target, missing = _target_gate(
        request.get("target_orchestration"),
        reconstruction_digest=reconstruction["reconstruction_digest"],
        source_scene_digest=str(
            reconstruction.get("source_scene_digest") or reconstruction["reconstruction_digest"]
        ),
        appearance_asset_digest=reconstruction["appearance_asset_digest"],
    )
    if missing:
        return _abstention(
            run_id=run_id,
            source_profile_digest=source_digest,
            code=missing["code"],
            instruction=missing["instruction"],
            stage="automatic_task_target_binding",
        )
    selected_target = dict(target["selected_target"])
    expected_robot = _expected_robot(selected_target)
    placement, missing = _placement_gate(
        request.get("robot_placement"),
        expected_robot=expected_robot,
        target_binding_digest=selected_target["target_binding_digest"],
    )
    if missing:
        result = _abstention(
            run_id=run_id,
            source_profile_digest=source_digest,
            code=missing["code"],
            instruction=missing["instruction"],
            stage="robot_placement",
        )
        result["robot_id"] = expected_robot
        result["run_digest"] = canonical_digest(result, digest_field="run_digest")
        return result
    requirement = target.get("task_zone_asset_requirement")
    requirement = dict(requirement) if isinstance(requirement, Mapping) else {}
    task_zone_required = requirement.get("verified_simready_asset_required") is True
    composition, missing = _scene_composition_gate(
        request.get("scene_composition"), task_zone_required=task_zone_required
    )
    if missing:
        result = _abstention(
            run_id=run_id,
            source_profile_digest=source_digest,
            code=missing["code"],
            instruction=missing["instruction"],
            stage="qualified_scene_composition",
        )
        result["robot_id"] = expected_robot
        result["run_digest"] = canonical_digest(result, digest_field="run_digest")
        return result
    route, missing = _route_gate(
        request.get("routing_inputs"),
        source_profile_digest=str(source_digest),
        target_binding_digest=selected_target["target_binding_digest"],
        placement_digest=placement["placement_digest"],
        robot_id=expected_robot,
        task_class=str(selected_target["task_class"]),
    )
    if missing:
        result = _abstention(
            run_id=run_id,
            source_profile_digest=source_digest,
            code=missing["code"],
            instruction=missing["instruction"],
            stage="task_site_engine_routing",
            blockers=(
                dict(dict(route.get("abstention") or {}).get("smallest_next_action") or {}).get(
                    "blocking_codes"
                )
                if route
                else None
            ),
            routing_decision=route or None,
        )
        result["robot_id"] = expected_robot
        result["run_digest"] = canonical_digest(result, digest_field="run_digest")
        return result

    evaluation = request.get("policy_evaluation")
    if not isinstance(evaluation, Mapping):
        result = _abstention(
            run_id=run_id,
            source_profile_digest=source_digest,
            code="five_learned_policy_attempts_missing",
            instruction="Bind exactly five immutable learned-policy candidates and execute one matched-reset attempt per candidate.",
            stage="learned_policy_evaluation",
            routing_decision=route,
        )
        result["robot_id"] = expected_robot
        result["run_digest"] = canonical_digest(result, digest_field="run_digest")
        return result
    authorization = request.get("execution_authorization")
    if not isinstance(authorization, Mapping):
        result = _abstention(
            run_id=run_id,
            source_profile_digest=source_digest,
            code="policy_execution_authorization_missing",
            instruction="Provide a digest-bound authorization for this exact route and candidate set.",
            stage="learned_policy_evaluation",
            routing_decision=route,
        )
        result["robot_id"] = expected_robot
        result["run_digest"] = canonical_digest(result, digest_field="run_digest")
        return result
    execution_bundle = evaluation.get("learned_policy_execution_bundle")
    if execution_bundle is not None:
        if not isinstance(execution_bundle, Mapping):
            raise NewSiteTaskEvaluationError(["learned_policy_execution_bundle_invalid"])
        if any(
            key in evaluation
            for key in ("task_metric", "policy_candidates", "attempts")
        ):
            raise NewSiteTaskEvaluationError(
                ["caller_policy_evidence_forbidden_with_execution_bundle"]
            )
        try:
            from .franka_inspection_learned_policy_lane import (
                LearnedPolicyLaneError,
                unpack_execution_bundle,
            )

            derived = unpack_execution_bundle(execution_bundle)
        except LearnedPolicyLaneError as exc:
            raise NewSiteTaskEvaluationError(
                [f"learned_policy_execution_bundle:{code}" for code in exc.codes]
            ) from exc
        bundle_authorization = dict(derived["execution_authorization"])
        if authorization != bundle_authorization:
            raise NewSiteTaskEvaluationError(
                ["execution_authorization_bundle_binding_mismatch"]
            )
        candidate_values = derived["policy_candidates"]
        attempt_values = derived["attempts"]
        metric_value = derived["task_metric"]
        authorization = bundle_authorization
    else:
        candidate_values = evaluation.get("policy_candidates")
        attempt_values = evaluation.get("attempts")
        metric_value = evaluation.get("task_metric")
    candidate_count = len(candidate_values) if isinstance(candidate_values, list) else 0
    attempt_count = len(attempt_values) if isinstance(attempt_values, list) else 0
    if candidate_count != EXPECTED_POLICY_CANDIDATES or attempt_count != EXPECTED_POLICY_CANDIDATES:
        result = _abstention(
            run_id=run_id,
            source_profile_digest=source_digest,
            code="exactly_five_learned_policy_attempts_required",
            instruction="Bind exactly five immutable learned-policy candidates and one real matched-reset attempt receipt for each candidate.",
            stage="learned_policy_evaluation",
            blockers=[
                f"learned_policy_candidate_count:{candidate_count}",
                f"learned_policy_attempt_count:{attempt_count}",
            ],
            routing_decision=route,
        )
        result["robot_id"] = expected_robot
        result["policy_attempt_count"] = min(attempt_count, EXPECTED_POLICY_CANDIDATES)
        result["run_digest"] = canonical_digest(result, digest_field="run_digest")
        return result
    metric, metric_frozen_at = _metric_spec(metric_value)
    candidates = _policy_candidates(candidate_values)
    _execution_authorization(
        authorization,
        route=route,
        placement=placement,
        metric=metric,
        candidates=candidates,
    )
    attempts, supported, unsupported = _attempts(
        attempt_values,
        candidates=candidates,
        metric=metric,
        metric_frozen_at=metric_frozen_at,
        route=route,
        placement=placement,
    )
    if len(supported) < 2:
        result = _abstention(
            run_id=run_id,
            source_profile_digest=source_digest,
            code="insufficient_supported_policy_candidates_for_ranking",
            instruction="Repair the task-metric or execution evidence for at least two of the five completed learned-policy attempts.",
            stage="task_metric_ranking",
            blockers=[f"unsupported_policy_candidate:{row}" for row in unsupported],
            routing_decision=route,
        )
        result["robot_id"] = expected_robot
        result["policy_attempt_count"] = len(attempts)
        result["supported_ranking_candidate_count"] = len(supported)
        result["claim_boundary"]["learned_policy_attempts_observed"] = True
        result["run_digest"] = canonical_digest(result, digest_field="run_digest")
        return result
    ranking = _rank(supported, direction=str(metric["direction"]))
    winner_candidate_ids = sorted(str(row["candidate_id"]) for row in ranking if row["rank"] == 1)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "completed",
        "request_digest": request["request_digest"],
        "source_profile_digest": source_digest,
        "source_truth_class": (
            "provider_derived_support"
            if source.get("status") == "admitted_provider_derived_support"
            else "blueprint_raw_contract"
        ),
        "reconstruction_digest": reconstruction["reconstruction_digest"],
        "appearance_evidence_digest": reconstruction["appearance_asset_digest"],
        "dynamics_geometry_digest": reconstruction["geometry_asset_digest"],
        "target_orchestration_digest": target["target_orchestration_digest"],
        "target_binding_digest": selected_target["target_binding_digest"],
        "task_family": selected_target["task_family"],
        "task_class": selected_target["task_class"],
        "robot_id": expected_robot,
        "placement_digest": placement["placement_digest"],
        "scene_composition_digest": composition["scene_composition_digest"],
        "routing_decision": route,
        "selected_engine_stack": [
            {
                "method_id": row["method_id"],
                "method_family": row["method_family"],
                "qualification_digest": row["qualification_digest"],
            }
            for row in route["selected_route"]["stages"]
        ],
        "task_metric": metric,
        "matched_reset_digest": attempts[0]["matched_reset_digest"],
        "initial_state_observation_digest": attempts[0]["initial_state_observation_digest"],
        "policy_attempt_count": len(attempts),
        "supported_ranking_candidate_count": len(supported),
        "unsupported_policy_candidate_ids": unsupported,
        "policy_attempt_digests": sorted(row["attempt_digest"] for row in attempts),
        "ranking": ranking,
        "winner_candidate_ids": winner_candidate_ids,
        "winner_candidate_id": (
            winner_candidate_ids[0] if len(winner_candidate_ids) == 1 else None
        ),
        "terminal_stage": "task_metric_ranking",
        "blockers": [],
        "smallest_missing_measurement": None,
        "claim_boundary": {
            "task_evaluation_run_completed": True,
            "learned_policy_attempts_observed": True,
            "learned_policy_ranking_proven": True,
            "controller_ranking_is_learned_policy_ranking": False,
            "provider_source_is_blueprint_raw_truth": source.get("status")
            == "admitted_blueprint_raw_contract",
            "appearance_used_as_dynamics_authority": False,
            "presentation_processing_used_as_evaluation_evidence": False,
            "physical_success_proven": False,
            "deployment_readiness_proven": False,
        },
    }
    result["run_digest"] = canonical_digest(result, digest_field="run_digest")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NewSiteTaskEvaluationError(["new_site_request_file_invalid"]) from exc
    if not isinstance(value, dict):
        raise NewSiteTaskEvaluationError(["new_site_request_file_invalid"])
    return value


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise NewSiteTaskEvaluationError(["new_site_result_output_conflict"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile a provider-neutral new-site Task Evaluation Run or abstention."
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = compile_new_site_task_evaluation_run(_load_json(args.request))
    _write_immutable(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "run_digest": result["run_digest"],
                "smallest_missing_measurement": result["smallest_missing_measurement"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "completed" else 2


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "AUTHORIZATION_SCHEMA_VERSION",
    "EXPECTED_POLICY_CANDIDATES",
    "METRIC_SCHEMA_VERSION",
    "NewSiteTaskEvaluationError",
    "POLICY_IDENTITY_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "compile_new_site_task_evaluation_run",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
