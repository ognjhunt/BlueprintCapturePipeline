"""Deterministic, fail-closed task/site measurement authorization kernel."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from typing import Any, Mapping, Sequence

from .measurement_site_evidence_contracts import site_geometry_bridge_validation_errors
from .task_site_measurement_taxonomy import (
    CAPABILITY_SITE_EVIDENCE,
    EVIDENCE_SMALLEST_ACTION,
)


REQUIREMENTS_SCHEMA_VERSION = "task_measurement_requirements.v1"
SITE_EVIDENCE_SCHEMA_VERSION = "site_evidence_profile.v1"
METHOD_CAPABILITY_SCHEMA_VERSION = "method_capability_profile.v1"
QUALIFICATION_SCHEMA_VERSION = "measurement_qualification_record.v1"
ROUTING_DECISION_SCHEMA_VERSION = "task_site_measurement_routing_decision.v1"
ABSTENTION_SCHEMA_VERSION = "abstention_or_next_action.v1"
SITE_EVIDENCE_AUDIT_SCHEMA_VERSION = "site_evidence_audit.v1"
POLICY_VERSION = "task_site_measurement_policy.2026-08-02"
TAXONOMY_VERSION = "2026-08-01"

CLAIM_LEVELS = {f"C{level}": level for level in range(9)}

MEASUREMENT_METHOD_FAMILY_CLAIM_CEILING: dict[str, str] = {
    "analytic_geometry_kinematics": "C2",
    "captured_real_observation": "C4",
    "traditional_simulation": "C4",
    "multiphysics_engineering_solver": "C4",
    "calibrated_renderer_sensor_simulation": "C4",
    "neural_reconstruction_appearance": "C4",
    "learned_world_model": "C4",
    "task_benchmark_framework": "C4",
    "external_provider_tool": "C4",
    "owner_attested_operational_input": "C7",
    "physical_evidence": "C8",
}

# Families that may never be qualified as collision/contact/material authorities. A
# splat, a rendered frame, a generated rollout, or a passive capture cannot
# measure force truth.
PHYSICS_AUTHORITY_FORBIDDEN_FAMILIES = frozenset(
    {
        "captured_real_observation",
        "calibrated_renderer_sensor_simulation",
        "neural_reconstruction_appearance",
        "learned_world_model",
        "task_benchmark_framework",
    }
)

WORLD_MODEL_ALLOWED_ROLES = frozenset(
    {
        "proposal_generation",
        "qualitative_rollout",
        "synthetic_training_data",
        "counterfactual_hypothesis",
        "evaluator_support",
        "comparative_policy_ranking",
    }
)
WORLD_MODEL_FORBIDDEN_ROLES = frozenset(
    {
        "collision_authority",
        "force_authority",
        "safety_authority",
        "physical_success_proof",
    }
)

CAPABILITY_FIELDS_BY_GROUP: dict[str, tuple[str, ...]] = {
    "identity_reproducibility": (
        "method_id",
        "method_family",
        "version",
        "release_date",
        "commit_hash",
        "container_digest",
        "plugin_versions",
        "solver_backend",
        "numeric_precision",
        "deterministic_mode",
        "operating_system",
        "gpu_model",
        "driver_version",
        "random_seed_policy",
    ),
    "geometry_scene": (
        "metric_scale_supported",
        "mesh_supported",
        "convex_collider_supported",
        "triangle_mesh_collider_supported",
        "sdf_collider_supported",
        "esdf_supported",
        "continuous_collision_supported",
        "thin_shell_collision_supported",
        "two_sided_collision_supported",
        "self_collision_supported",
        "dynamic_collision_supported",
        "openusd_supported",
        "mjcf_supported",
        "urdf_supported",
        "gaussian_splat_supported_for_rendering",
        "nerf_supported_for_rendering",
        "uncertainty_map_supported",
    ),
    "contact": (
        "contact_formulation",
        "penalty_contact_supported",
        "complementarity_contact_supported",
        "hydroelastic_contact_supported",
        "ipc_contact_supported",
        "adhesion_supported",
        "static_friction_supported",
        "dynamic_friction_supported",
        "anisotropic_friction_supported",
        "rolling_friction_supported",
        "torsional_friction_supported",
        "restitution_supported",
        "contact_compliance_supported",
        "contact_force_output_supported",
        "impulse_output_supported",
        "penetration_metric_supported",
    ),
    "articulation": (
        "revolute_joint_supported",
        "prismatic_joint_supported",
        "closed_chain_supported",
        "mimic_joint_supported",
        "joint_limits_supported",
        "joint_friction_supported",
        "joint_damping_supported",
        "backlash_supported",
        "detent_supported",
        "joint_compliance_supported",
        "articulation_force_output_supported",
    ),
    "deformation_materials": (
        "cloth_shell_supported",
        "anisotropic_cloth_supported",
        "seams_supported",
        "tearing_supported",
        "rod_cable_supported",
        "torsional_rod_supported",
        "plastic_bending_supported",
        "hyperelastic_fem_supported",
        "nearly_incompressible_supported",
        "viscoelastic_supported",
        "foam_compression_supported",
        "mpm_supported",
        "dem_supported",
        "sph_supported",
        "cfd_supported",
        "granular_cohesion_supported",
        "particle_breakage_supported",
        "fluid_surface_tension_supported",
        "fluid_wetting_supported",
        "fracture_supported",
        "cutting_supported",
        "tissue_model_supported",
        "food_specific_model_supported",
    ),
    "robot_controller_action": (
        "robot_model_formats",
        "supported_embodiments",
        "supported_end_effectors",
        "position_control_supported",
        "velocity_control_supported",
        "torque_control_supported",
        "impedance_control_supported",
        "compliance_control_supported",
        "maximum_control_rate_hz",
        "action_representation_types",
        "controller_latency_model_supported",
        "actuator_dynamics_supported",
        "whole_body_control_supported",
    ),
    "sensors_rendering": (
        "rgb_supported",
        "depth_supported",
        "structured_light_supported",
        "lidar_supported",
        "radar_supported",
        "event_camera_supported",
        "imu_supported",
        "force_torque_supported",
        "contact_sensor_supported",
        "optical_tactile_supported",
        "taxel_tactile_supported",
        "intrinsics_import_supported",
        "extrinsics_import_supported",
        "distortion_model_supported",
        "rolling_shutter_supported",
        "motion_blur_supported",
        "isp_model_supported",
        "temporal_noise_supported",
        "material_brdf_supported",
        "transmission_refraction_supported",
        "spectral_model_supported",
        "sensor_timing_supported",
    ),
    "qualification_claim": (
        "qualification_record_ids",
        "qualified_task_classes",
        "qualified_material_regimes",
        "qualified_robot_ids",
        "qualified_end_effector_ids",
        "qualified_controller_ids",
        "qualified_sensor_ids",
        "qualified_site_classes",
        "qualified_parameter_ranges",
        "qualified_metric_ids",
        "qualified_claim_ceiling",
        "qualification_expiration",
        "harmful_false_negative_bound",
        "known_failure_modes",
        "prohibited_extrapolations",
    ),
    "rights_privacy_operations": (
        "source_available",
        "local_offline_supported",
        "api_only",
        "commercial_use_allowed",
        "redistribution_allowed",
        "asset_license_ids",
        "model_license_ids",
        "provider_training_use_allowed",
        "data_retention_days",
        "deletion_right_supported",
        "subprocessor_regions",
        "output_export_supported",
        "output_formats",
        "maximum_latency_class",
        "maximum_compute_class",
        "estimated_cost_class",
    ),
}

ALL_CAPABILITY_FIELDS = frozenset(
    field for fields in CAPABILITY_FIELDS_BY_GROUP.values() for field in fields
)

# Capability fields whose qualified use asserts collision, contact,
# articulation, or material-dynamics truth.  Families in
# PHYSICS_AUTHORITY_FORBIDDEN_FAMILIES may never be qualified for these.
PHYSICS_AUTHORITY_CAPABILITIES = frozenset(
    field
    for group in ("contact", "articulation", "deformation_materials")
    for field in CAPABILITY_FIELDS_BY_GROUP[group]
    if field != "contact_formulation"
) | frozenset(
    {
        "continuous_collision_supported",
        "dynamic_collision_supported",
        "self_collision_supported",
        "thin_shell_collision_supported",
        "two_sided_collision_supported",
    }
)

TASK_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "static_reachability": (
        "metric_scale_supported",
        "self_collision_supported",
        "joint_limits_supported",
    ),
    "collision_free_motion": (
        "metric_scale_supported",
        "continuous_collision_supported",
        "self_collision_supported",
        "uncertainty_map_supported",
    ),
    "rigid_pick_place": (
        "metric_scale_supported",
        "dynamic_collision_supported",
        "static_friction_supported",
        "dynamic_friction_supported",
        "contact_compliance_supported",
    ),
    "insertion_assembly": (
        "continuous_collision_supported",
        "contact_compliance_supported",
        "static_friction_supported",
        "force_torque_supported",
        "impedance_control_supported",
    ),
    "doors_drawers_handles": (
        "metric_scale_supported",
        "revolute_joint_supported",
        "prismatic_joint_supported",
        "joint_limits_supported",
        "joint_friction_supported",
        "joint_damping_supported",
        "detent_supported",
        "contact_force_output_supported",
    ),
    "valves_switches_buttons": (
        "metric_scale_supported",
        "revolute_joint_supported",
        "joint_limits_supported",
        "detent_supported",
        "joint_compliance_supported",
        "contact_force_output_supported",
    ),
    "contact_rich_dexterous_manipulation": (
        "metric_scale_supported",
        "self_collision_supported",
        "static_friction_supported",
        "dynamic_friction_supported",
        "rolling_friction_supported",
        "torsional_friction_supported",
        "contact_compliance_supported",
    ),
    "visual_perception": (
        "rgb_supported",
        "intrinsics_import_supported",
        "extrinsics_import_supported",
        "distortion_model_supported",
        "sensor_timing_supported",
    ),
    "visual_navigation_active_perception": (
        "metric_scale_supported",
        "rgb_supported",
        "depth_supported",
        "intrinsics_import_supported",
        "extrinsics_import_supported",
        "sensor_timing_supported",
        "uncertainty_map_supported",
    ),
    "transparent_reflective_objects": (
        "rgb_supported",
        "intrinsics_import_supported",
        "material_brdf_supported",
        "transmission_refraction_supported",
    ),
    "small_thin_occluded_objects": (
        "metric_scale_supported",
        "thin_shell_collision_supported",
        "two_sided_collision_supported",
        "uncertainty_map_supported",
    ),
    "locomotion": (
        "dynamic_collision_supported",
        "static_friction_supported",
        "dynamic_friction_supported",
        "imu_supported",
        "actuator_dynamics_supported",
        "whole_body_control_supported",
    ),
    "mobile_manipulation_clutter": (
        "metric_scale_supported",
        "continuous_collision_supported",
        "self_collision_supported",
        "dynamic_collision_supported",
        "static_friction_supported",
        "whole_body_control_supported",
    ),
    "human_robot_interaction": (
        "metric_scale_supported",
        "continuous_collision_supported",
        "sensor_timing_supported",
    ),
    "long_horizon_task_execution": (
        "metric_scale_supported",
        "dynamic_collision_supported",
        "self_collision_supported",
        "static_friction_supported",
    ),
    "garment_manipulation": (
        "cloth_shell_supported",
        "anisotropic_cloth_supported",
        "seams_supported",
        "self_collision_supported",
        "static_friction_supported",
    ),
    "cable_hose_routing": (
        "rod_cable_supported",
        "torsional_rod_supported",
        "self_collision_supported",
        "static_friction_supported",
    ),
    "granular_manipulation": (
        "dem_supported",
        "granular_cohesion_supported",
        "rolling_friction_supported",
    ),
    "fluid_manipulation": (
        "fluid_surface_tension_supported",
        "fluid_wetting_supported",
    ),
    "food_manipulation": (
        "food_specific_model_supported",
        "viscoelastic_supported",
        "fracture_supported",
    ),
    "tactile_manipulation": (
        "force_torque_supported",
        "contact_sensor_supported",
        "optical_tactile_supported",
    ),
}

# Alternative capability groups: at least one member of each group must be
# covered by the composite route.  This encodes solver alternatives such as
# "SPH, CFD, or validated MPM" for free-surface fluids without demanding all.
TASK_CAPABILITY_ALTERNATIVES: dict[str, tuple[tuple[str, ...], ...]] = {
    "fluid_manipulation": (("sph_supported", "cfd_supported", "mpm_supported"),),
}

# Task-class evidence that is not implied by any single capability field.
TASK_SITE_EVIDENCE: dict[str, tuple[str, ...]] = {
    "human_robot_interaction": ("dynamic_object_tracks",),
    "visual_navigation_active_perception": ("coverage_uncertainty",),
    "long_horizon_task_execution": ("real_demonstrations",),
}

# Controlled material regimes.  "deformable" is deliberately NOT a value: the
# router refuses it because cloth, cable, foam, dough, powder, fluid, bag,
# carton, food, and tissue need different state representations and evidence.
MATERIAL_REGIME_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "none": (),
    "rigid": (),
    "garment_cloth": (
        "cloth_shell_supported",
        "anisotropic_cloth_supported",
        "seams_supported",
        "self_collision_supported",
        "static_friction_supported",
    ),
    "towel_sheet": (
        "cloth_shell_supported",
        "self_collision_supported",
        "static_friction_supported",
    ),
    "rope_cable_hose": (
        "rod_cable_supported",
        "torsional_rod_supported",
        "self_collision_supported",
        "static_friction_supported",
    ),
    "paper_cardboard_thin_sheet": (
        "thin_shell_collision_supported",
        "two_sided_collision_supported",
        "plastic_bending_supported",
    ),
    "elastomer_rubber": (
        "hyperelastic_fem_supported",
        "nearly_incompressible_supported",
        "viscoelastic_supported",
    ),
    "foam": ("foam_compression_supported", "viscoelastic_supported"),
    "elastoplastic_dough_clay": ("mpm_supported", "viscoelastic_supported"),
    "granular_media": (
        "dem_supported",
        "granular_cohesion_supported",
        "rolling_friction_supported",
    ),
    "fluid_viscous_free_surface": (
        "fluid_surface_tension_supported",
        "fluid_wetting_supported",
    ),
    "plastic_fabric_bag": (
        "cloth_shell_supported",
        "thin_shell_collision_supported",
        "self_collision_supported",
        "static_friction_supported",
    ),
    "carton_box_packaging": (
        "thin_shell_collision_supported",
        "two_sided_collision_supported",
        "plastic_bending_supported",
    ),
    "food_cuttable_multiphase": (
        "food_specific_model_supported",
        "cutting_supported",
        "fracture_supported",
        "viscoelastic_supported",
    ),
    "tissue_surgical_soft_body": (
        "tissue_model_supported",
        "hyperelastic_fem_supported",
        "viscoelastic_supported",
    ),
}

MATERIAL_REGIME_ALTERNATIVES: dict[str, tuple[tuple[str, ...], ...]] = {
    "fluid_viscous_free_surface": (("sph_supported", "cfd_supported", "mpm_supported"),),
}

FORBIDDEN_GENERIC_MATERIAL_REGIMES = frozenset(
    {"deformable", "soft", "soft_body", "cloth", "flexible", "generic"}
)

# Structured-interaction vocabularies from the task_measurement_requirements
# research schema.
CONTACT_REGIME_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "none": (),
    "hard": (),
    "compliant": ("contact_compliance_supported",),
    "hydroelastic": ("hydroelastic_contact_supported",),
    "adhesive": ("adhesion_supported",),
    "cutting": ("cutting_supported",),
    "puncture": ("cutting_supported", "fracture_supported"),
}

DEFORMATION_FAMILY_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "none": (),
    "cloth_shell": ("cloth_shell_supported", "self_collision_supported"),
    "rod_cable": ("rod_cable_supported", "torsional_rod_supported"),
    "thin_sheet": ("thin_shell_collision_supported", "two_sided_collision_supported"),
    "hyperelastic_solid": ("hyperelastic_fem_supported",),
    "foam": ("foam_compression_supported",),
    "mpm_elastoplastic": ("mpm_supported",),
    "dem_granular": ("dem_supported",),
    "sph_fluid": ("sph_supported",),
    "cfd_fluid": ("cfd_supported",),
    "tissue": ("tissue_model_supported",),
    "food_specific": ("food_specific_model_supported",),
}

SENSOR_MODALITY_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "rgb": (
        "rgb_supported",
        "intrinsics_import_supported",
        "extrinsics_import_supported",
        "distortion_model_supported",
    ),
    "depth": ("depth_supported",),
    "structured_light": ("structured_light_supported",),
    "lidar": ("lidar_supported",),
    "radar": ("radar_supported",),
    "event_camera": ("event_camera_supported",),
    "imu": ("imu_supported",),
    "force_torque": ("force_torque_supported",),
    "contact": ("contact_sensor_supported",),
    "optical_tactile": ("optical_tactile_supported",),
    "taxel_tactile": ("taxel_tactile_supported",),
}

CLAIM_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "capture_provenance": (),
    "reachability": TASK_CAPABILITIES["static_reachability"],
    "kinematic_feasibility": TASK_CAPABILITIES["static_reachability"],
    "perception_visibility": TASK_CAPABILITIES["visual_perception"],
    "collision_contact": ("metric_scale_supported", "continuous_collision_supported"),
    "comparative_policy_ranking": (
        "sensor_timing_supported",
        "controller_latency_model_supported",
    ),
    "physical_task_success": (),
    "deployment_readiness": (),
    "safety_certification": (),
}

CLAIM_TYPE_LEVEL = {
    "capture_provenance": "C0",
    "reachability": "C1",
    "kinematic_feasibility": "C1",
    "perception_visibility": "C2",
    "collision_contact": "C3",
    "comparative_policy_ranking": "C4",
    "sim_to_real_transfer": "C5",
    "physical_task_success": "C6",
    "deployment_readiness": "C7",
    "safety_certification": "C8",
}

# Controlled site-evidence vocabulary (the minimum site-evidence taxonomy).
# Unknown evidence identifiers are rejected: an uncontrolled vocabulary would
# let an appearance artifact masquerade as a collision or material record.
SITE_EVIDENCE_TAXONOMY = (
    frozenset(evidence_id for values in CAPABILITY_SITE_EVIDENCE.values() for evidence_id in values)
    | frozenset(evidence_id for values in TASK_SITE_EVIDENCE.values() for evidence_id in values)
    | frozenset(
        {
            "camera_poses",
            "multiview_coverage",
            "object_segmentation",
            "semantic_labels",
            "gaussian_splat_appearance",
            "appearance_mesh",
            "physical_outcomes",
        }
    )
)

# Appearance-only evidence can never stand in for physical evidence.  The
# router treats these identifiers as insufficient for every physical evidence
# requirement; they are listed so audits and documentation can say so.
APPEARANCE_ONLY_SITE_EVIDENCE = frozenset({"gaussian_splat_appearance", "appearance_mesh"})

# Smallest-next-action taxonomy from the routing research, mapped from the
# blocking site-evidence identifier.
SMALLEST_NEXT_ACTION_TYPES = frozenset(EVIDENCE_SMALLEST_ACTION.values()) | frozenset(
    {
        "adapter_work",
        "qualification_benchmark",
        "rights_approval",
        "physical_execution",
        "request_contract_clarification",
    }
)

# Deterministic-replay is an allowlist: an unverified or free-form mode string
# never satisfies a replay requirement.
DETERMINISTIC_REPLAY_MODES = frozenset(
    {"strict", "deterministic", "replay", "seeded", "single_threaded_deterministic"}
)


class MeasurementRoutingError(ValueError):
    """Stable fail-closed validation or routing error."""

    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise MeasurementRoutingError("artifact_not_json") from exc
    if not isinstance(result, dict):
        raise MeasurementRoutingError("artifact_not_object")
    return result


def _canonical_digest(value: Mapping[str, Any], digest_field: str | None = None) -> str:
    normalized = dict(value)
    if digest_field:
        normalized.pop(digest_field, None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _strings(value: Any) -> list[str]:
    return (
        sorted({_string(item) for item in value if _string(item)})
        if isinstance(value, list)
        else []
    )


def _number(value: Any, default: float = math.inf) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _finalize(value: Mapping[str, Any], *, schema: str, digest_field: str) -> dict[str, Any]:
    result = _clone(value)
    if result.get("schema_version") != schema:
        raise MeasurementRoutingError(f"schema_version_must_be:{schema}")
    expected = _canonical_digest(result, digest_field)
    supplied = result.get(digest_field)
    if supplied is not None and supplied != expected:
        raise MeasurementRoutingError(f"{digest_field}_mismatch")
    result[digest_field] = expected
    return result


def _normalized_alternative_groups(
    groups: Sequence[Sequence[str]], required: set[str]
) -> list[list[str]]:
    normalized: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        members = tuple(sorted({_string(item) for item in group if _string(item)}))
        if not members or set(members) & required or members in seen:
            continue
        seen.add(members)
        normalized.append(list(members))
    return sorted(normalized)


def _interaction_capability_requirements(
    interaction: Mapping[str, Any],
) -> tuple[set[str], list[str]]:
    """Map a structured interaction section to capability requirements."""

    errors: list[str] = []
    capabilities: set[str] = set()
    contact = interaction.get("contact_regime")
    contact = dict(contact) if isinstance(contact, Mapping) else {}
    for regime in _strings(contact.get("type")):
        if regime not in CONTACT_REGIME_CAPABILITIES:
            errors.append(f"contact_regime_unknown:{regime}")
            continue
        capabilities.update(CONTACT_REGIME_CAPABILITIES[regime])
    if contact.get("continuous_collision_required") is True:
        capabilities.add("continuous_collision_supported")
    if contact.get("self_contact_required") is True:
        capabilities.add("self_collision_supported")
    friction = contact.get("friction")
    friction = dict(friction) if isinstance(friction, Mapping) else {}
    if friction.get("required") is True:
        capabilities.update({"static_friction_supported", "dynamic_friction_supported"})
    if friction.get("anisotropic") is True:
        capabilities.add("anisotropic_friction_supported")
    if friction.get("rolling") is True:
        capabilities.add("rolling_friction_supported")
    if friction.get("torsional") is True:
        capabilities.add("torsional_friction_supported")
    if contact.get("force_or_impulse_output_required") is True:
        capabilities.add("contact_force_output_supported")
    deformation = interaction.get("deformation")
    deformation = dict(deformation) if isinstance(deformation, Mapping) else {}
    for family in _strings(deformation.get("family")):
        if family in FORBIDDEN_GENERIC_MATERIAL_REGIMES:
            errors.append(f"deformation_family_forbidden_generic:{family}")
            continue
        if family not in DEFORMATION_FAMILY_CAPABILITIES:
            errors.append(f"deformation_family_unknown:{family}")
            continue
        capabilities.update(DEFORMATION_FAMILY_CAPABILITIES[family])
    return capabilities, errors


def minimum_capability_requirements(
    *,
    task_class: str,
    claim_type: str,
    material_regimes: Sequence[str],
    sensor_modalities: Sequence[str],
    interaction: Mapping[str, Any] | None = None,
) -> tuple[set[str], list[list[str]], list[str]]:
    """Deterministic capability floor for a controlled task/claim/scope tuple.

    Returns ``(required_capabilities, alternative_groups, errors)``.  Agents
    may extend the floor with narrower requirements; they may never go below
    it (enforced by ``validate_task_measurement_requirements``).
    """

    errors: list[str] = []
    if task_class not in TASK_CAPABILITIES:
        errors.append(f"task_class_unknown:{task_class or 'missing'}")
    capabilities: set[str] = set()
    alternatives: list[Sequence[str]] = []
    if claim_type != "capture_provenance":
        capabilities.update(TASK_CAPABILITIES.get(task_class, ()))
        capabilities.update(CLAIM_CAPABILITIES.get(claim_type, ()))
        alternatives.extend(TASK_CAPABILITY_ALTERNATIVES.get(task_class, ()))
        for regime in material_regimes:
            regime_id = _string(regime)
            if regime_id in FORBIDDEN_GENERIC_MATERIAL_REGIMES:
                errors.append(f"material_regime_forbidden_generic:{regime_id}")
                continue
            if regime_id not in MATERIAL_REGIME_CAPABILITIES:
                errors.append(f"material_regime_unknown:{regime_id}")
                continue
            capabilities.update(MATERIAL_REGIME_CAPABILITIES[regime_id])
            alternatives.extend(MATERIAL_REGIME_ALTERNATIVES.get(regime_id, ()))
        for modality in sensor_modalities:
            modality_id = _string(modality)
            if modality_id not in SENSOR_MODALITY_CAPABILITIES:
                errors.append(f"sensor_modality_unknown:{modality_id}")
                continue
            capabilities.update(SENSOR_MODALITY_CAPABILITIES[modality_id])
        if isinstance(interaction, Mapping) and interaction:
            interaction_capabilities, interaction_errors = _interaction_capability_requirements(
                interaction
            )
            capabilities.update(interaction_capabilities)
            errors.extend(interaction_errors)
    groups = _normalized_alternative_groups(alternatives, capabilities)
    return capabilities, groups, errors


def _required_site_evidence_for(
    capabilities: Sequence[str],
    alternative_groups: Sequence[Sequence[str]],
    task_class: str,
) -> list[str]:
    fields = set(capabilities) | {field for group in alternative_groups for field in group}
    evidence = {
        evidence_id
        for capability in fields
        for evidence_id in CAPABILITY_SITE_EVIDENCE.get(capability, ())
    }
    evidence.update(TASK_SITE_EVIDENCE.get(task_class, ()))
    return sorted(evidence)


def derive_task_measurement_requirements(
    claim: Mapping[str, Any], testbed: Mapping[str, Any]
) -> dict[str, Any]:
    """Compile controlled requirements from a claim and exact testbed binding."""

    explicit = claim.get("task_measurement_requirements")
    if isinstance(explicit, Mapping):
        return validate_task_measurement_requirements(explicit)
    task_family = _string(claim.get("measurement_task_class"))
    if not task_family:
        distribution = testbed.get("task_distribution")
        if isinstance(distribution, Mapping):
            task_family = _string(distribution.get("measurement_task_class"))
            if not task_family:
                legacy = _string(distribution.get("task_family"))
                task_family = {
                    "rigid_object_pick_place": "rigid_pick_place",
                    "drawer_opening": "doors_drawers_handles",
                    "garment_folding": "garment_manipulation",
                    "cable_routing": "cable_hose_routing",
                }.get(legacy, legacy)
    claim_type = _string(claim.get("claim_type"))
    material_regimes = [_string(item) for item in (claim.get("material_regimes") or ["none"])]
    interaction = claim.get("interaction")
    interaction = dict(interaction) if isinstance(interaction, Mapping) else {}
    sensor_scope = dict(claim.get("sensor_scope") or {})
    bindings = testbed.get("robot_sensor_controller_bindings")
    bindings = dict(bindings) if isinstance(bindings, Mapping) else {}
    robot_scope = dict(claim.get("robot_scope") or {})
    embodiment = bindings.get("embodiment")
    if not robot_scope and isinstance(embodiment, Mapping):
        robot_scope = {
            key: embodiment.get(source)
            for key, source in (("robot_id", "robot_id"), ("end_effector_id", "end_effector_id"))
            if _string(embodiment.get(source))
        }
    controller = bindings.get("controller_action_representation")
    if isinstance(controller, Mapping) and _string(controller.get("controller_id")):
        robot_scope.setdefault("controller_id", controller.get("controller_id"))
    bound_sensors = bindings.get("sensors")
    if not sensor_scope and isinstance(bound_sensors, Mapping):
        sensor_scope = {
            "sensor_ids": sorted(
                {_string(item) for item in bound_sensors.values() if _string(item)}
            )
        }
    capabilities, alternative_groups, errors = minimum_capability_requirements(
        task_class=task_family,
        claim_type=claim_type,
        material_regimes=material_regimes,
        sensor_modalities=_strings(sensor_scope.get("required_modalities")),
        interaction=interaction,
    )
    if errors:
        raise MeasurementRoutingError(*errors)
    validation_envelope = testbed.get("validation_envelope")
    validation_envelope = (
        dict(validation_envelope) if isinstance(validation_envelope, Mapping) else {}
    )
    value = {
        "schema_version": REQUIREMENTS_SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "request_id": _string(claim.get("claim_id")) or "measurement-request",
        "task_class": task_family,
        "claim_type": claim_type,
        "requested_claim_level": CLAIM_TYPE_LEVEL.get(claim_type, "C3"),
        "required_capabilities": sorted(capabilities),
        "required_capability_alternatives": alternative_groups,
        "required_site_evidence": _required_site_evidence_for(
            sorted(capabilities), alternative_groups, task_family
        ),
        "interaction": interaction,
        "task_scope": {
            "object_ids": list(claim.get("object_ids") or []),
            "material_regimes": material_regimes,
            "metric_ids": list(claim.get("metric_ids") or []),
            "parameter_ranges": dict(claim.get("parameter_ranges") or {}),
        },
        "site_scope": {
            key: validation_envelope.get(key)
            for key in ("site_id", "site_class")
            if _string(validation_envelope.get(key))
        },
        "robot_scope": robot_scope,
        "sensor_scope": sensor_scope,
        "constraints": dict(claim.get("measurement_constraints") or {}),
        "agent_interpretation_authoritative": False,
    }
    return validate_task_measurement_requirements(value)


def validate_task_measurement_requirements(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _finalize(
        value, schema=REQUIREMENTS_SCHEMA_VERSION, digest_field="requirements_digest"
    )
    required = ("request_id", "task_class", "claim_type", "requested_claim_level")
    errors = [f"{key}_missing" for key in required if not _string(result.get(key))]
    if result.get("taxonomy_version") != TAXONOMY_VERSION:
        errors.append(f"taxonomy_version_must_be:{TAXONOMY_VERSION}")
    if result.get("requested_claim_level") not in CLAIM_LEVELS:
        errors.append("requested_claim_level_invalid")
    capabilities = _strings(result.get("required_capabilities"))
    if any(item not in ALL_CAPABILITY_FIELDS for item in capabilities):
        errors.append("required_capability_unknown")
    result.setdefault("required_capability_alternatives", [])
    alternatives = result.get("required_capability_alternatives")
    if not isinstance(alternatives, list) or any(
        not isinstance(group, list)
        or not group
        or any(_string(field) not in ALL_CAPABILITY_FIELDS for field in group)
        for group in alternatives
    ):
        errors.append("required_capability_alternatives_invalid")
    if not isinstance(result.get("required_site_evidence"), list):
        errors.append("required_site_evidence_invalid")
    else:
        for evidence_id in _strings(result.get("required_site_evidence")):
            if evidence_id not in SITE_EVIDENCE_TAXONOMY:
                errors.append(f"required_site_evidence_unknown:{evidence_id}")
    result.setdefault("interaction", {})
    if not isinstance(result.get("interaction"), Mapping):
        errors.append("interaction_invalid")
    for key in ("task_scope", "site_scope", "robot_scope", "sensor_scope", "constraints"):
        result.setdefault(key, {})
        if not isinstance(result.get(key), Mapping):
            errors.append(f"{key}_invalid")
    if result.get("agent_interpretation_authoritative") is not False:
        errors.append("agent_interpretation_must_be_non_authoritative")
    task_scope = dict(result.get("task_scope") or {})
    sensor_scope = dict(result.get("sensor_scope") or {})
    if not errors:
        minimum, minimum_groups, minimum_errors = minimum_capability_requirements(
            task_class=_string(result.get("task_class")),
            claim_type=_string(result.get("claim_type")),
            material_regimes=[_string(item) for item in (task_scope.get("material_regimes") or [])],
            sensor_modalities=_strings(sensor_scope.get("required_modalities")),
            interaction=dict(result.get("interaction") or {}),
        )
        errors.extend(minimum_errors)
        if not minimum_errors:
            below_floor = sorted(minimum - set(capabilities))
            if below_floor:
                errors.append(
                    "required_capabilities_below_deterministic_minimum:" + ",".join(below_floor)
                )
            supplied_groups = {
                tuple(sorted(_string(field) for field in group))
                for group in (alternatives or [])
                if isinstance(group, list)
            }
            for group in minimum_groups:
                members = tuple(sorted(group))
                if members not in supplied_groups and not (set(members) & set(capabilities)):
                    errors.append(
                        "required_capability_alternatives_below_minimum:" + ",".join(members)
                    )
    if errors:
        raise MeasurementRoutingError(*errors)
    result["required_capabilities"] = capabilities
    result["required_capability_alternatives"] = _normalized_alternative_groups(
        [list(group) for group in alternatives], set(capabilities)
    )
    result["required_site_evidence"] = _strings(result.get("required_site_evidence"))
    # These are policy defaults, not agent suggestions.  An omitted field may
    # never silently authorize provider training, non-portable output, or
    # non-commercial terms for customer-site evidence.
    constraints = dict(result.get("constraints") or {})
    constraints.setdefault("commercial_use_required", True)
    constraints.setdefault("output_portability_required", True)
    constraints.setdefault("provider_training_use_allowed", False)
    result["constraints"] = constraints
    result["requirements_digest"] = _canonical_digest(result, "requirements_digest")
    return result


def validate_site_evidence_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _finalize(
        value, schema=SITE_EVIDENCE_SCHEMA_VERSION, digest_field="site_evidence_digest"
    )
    errors: list[str] = []
    for key in ("profile_id", "bundle_id", "bundle_hash", "provenance_record_id"):
        if not _string(result.get(key)):
            errors.append(f"{key}_missing")
    if not _string(result.get("bundle_hash")).startswith("sha256:"):
        errors.append("bundle_hash_invalid")
    for key in ("rights", "privacy", "coordinate_system", "evidence", "limitations"):
        if not isinstance(result.get(key), Mapping):
            errors.append(f"{key}_invalid")
    evidence = result.get("evidence")
    if isinstance(evidence, Mapping):
        for key, record in evidence.items():
            evidence_id = _string(key)
            if evidence_id not in SITE_EVIDENCE_TAXONOMY:
                errors.append(f"evidence_id_unknown:{evidence_id}")
            if (
                not isinstance(record, Mapping)
                or not isinstance(record.get("available"), bool)
                or not isinstance(record.get("validated"), bool)
            ):
                errors.append(f"evidence_record_invalid:{evidence_id}")
            elif record.get("available") is True and not _string(record.get("record_id")):
                errors.append(f"evidence_record_id_missing:{evidence_id}")
    limitations = result.get("limitations")
    if isinstance(limitations, Mapping):
        forbidden = limitations.get("forbidden_claims")
        if forbidden is not None and not isinstance(forbidden, list):
            errors.append("limitations_forbidden_claims_invalid")
        else:
            for claim in _strings(forbidden):
                if claim not in CLAIM_LEVELS and claim not in CLAIM_TYPE_LEVEL:
                    errors.append(f"limitations_forbidden_claim_unknown:{claim}")
    errors.extend(site_geometry_bridge_validation_errors(result))
    if errors:
        raise MeasurementRoutingError(*errors)
    return result


def audit_site_evidence_profile(
    site_value: Mapping[str, Any],
    requirements_value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic capture-evidence audit.

    Identifies missing metric scale, unobserved regions, unvalidated colliders,
    missing articulation, uncharacterized materials, and sensor-calibration
    gaps, with the smallest next action per gap.  The audit reads the profile;
    it never rewrites or repairs raw capture truth.
    """

    site = validate_site_evidence_profile(site_value)
    core_evidence = (
        "metric_scale",
        "robot_site_registration",
        "validated_mesh",
        "validated_collider",
        "articulation_model",
        "material_parameters",
        "sensor_calibration",
    )
    required: set[str] = set(core_evidence)
    required_by_request: set[str] = set()
    if requirements_value is not None:
        requirements = validate_task_measurement_requirements(requirements_value)
        required_by_request = set(_strings(requirements.get("required_site_evidence")))
        required |= required_by_request
    evidence = dict(site.get("evidence") or {})
    gaps: list[dict[str, Any]] = []
    for evidence_id in sorted(required):
        record = evidence.get(evidence_id)
        record = dict(record) if isinstance(record, Mapping) else {}
        if record.get("available") is True and record.get("validated") is True:
            continue
        gaps.append(
            {
                "evidence_id": evidence_id,
                "gap": ("unvalidated" if record.get("available") is True else "missing"),
                "required_by_request": evidence_id in required_by_request,
                "smallest_next_action": EVIDENCE_SMALLEST_ACTION.get(
                    evidence_id, "targeted_recapture"
                ),
            }
        )
    coordinate_system = dict(site.get("coordinate_system") or {})
    if coordinate_system.get("metric_scale_verified") is not True:
        gaps.insert(
            0,
            {
                "evidence_id": "metric_scale",
                "gap": "metric_scale_unverified",
                "required_by_request": "metric_scale" in required_by_request,
                "smallest_next_action": "metric_scale_check",
            },
        )
    limitations = dict(site.get("limitations") or {})
    audit = {
        "schema_version": SITE_EVIDENCE_AUDIT_SCHEMA_VERSION,
        "profile_id": site["profile_id"],
        "site_evidence_digest": site["site_evidence_digest"],
        "gaps": gaps,
        "gap_count": len(gaps),
        "metric_scale_verified": coordinate_system.get("metric_scale_verified") is True,
        "known_missing_regions": list(limitations.get("known_missing_regions") or []),
        "transparent_reflective_objects": list(
            limitations.get("transparent_reflective_objects") or []
        ),
        "forbidden_claims": _strings(limitations.get("forbidden_claims")),
        "appearance_evidence_is_not_physical_evidence": True,
        "raw_capture_truth_rewritten": False,
    }
    return _finalize(
        audit,
        schema=SITE_EVIDENCE_AUDIT_SCHEMA_VERSION,
        digest_field="site_evidence_audit_digest",
    )


def validate_method_capability_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _finalize(
        value, schema=METHOD_CAPABILITY_SCHEMA_VERSION, digest_field="capability_profile_digest"
    )
    capabilities = result.get("capabilities")
    errors: list[str] = []
    if not isinstance(capabilities, Mapping):
        errors.append("capabilities_invalid")
    else:
        missing = sorted(ALL_CAPABILITY_FIELDS - set(capabilities))
        if missing:
            errors.append("capability_fields_missing:" + ",".join(missing))
        identity = capabilities.get("method_id")
        if _string(identity) != _string(result.get("method_id")):
            errors.append("method_identity_mismatch")
        family = _string(capabilities.get("method_family"))
        family_ceiling = MEASUREMENT_METHOD_FAMILY_CLAIM_CEILING.get(family)
        if family_ceiling is None:
            errors.append(f"method_family_unknown:{family or 'missing'}")
        ceiling = capabilities.get("qualified_claim_ceiling")
        if ceiling not in CLAIM_LEVELS:
            errors.append("qualified_claim_ceiling_invalid")
        elif family_ceiling is not None and (CLAIM_LEVELS[ceiling] > CLAIM_LEVELS[family_ceiling]):
            errors.append(f"method_claim_ceiling_exceeds_family_cap:{family}:{family_ceiling}")
        boolean_fields = {
            key for key in ALL_CAPABILITY_FIELDS if key.endswith(("_supported", "_allowed"))
        } | {"source_available", "api_only"}
        for key in sorted(boolean_fields):
            if key in capabilities and not isinstance(capabilities.get(key), bool):
                errors.append(f"capability_boolean_invalid:{key}")
        if (
            not math.isfinite(_number(capabilities.get("data_retention_days")))
            or _number(capabilities.get("data_retention_days")) < 0
        ):
            errors.append("data_retention_days_invalid")
        roles = result.get("world_model_roles")
        if family == "learned_world_model":
            role_values = _strings(roles)
            if not role_values:
                errors.append("world_model_roles_missing")
            for role in role_values:
                if role in WORLD_MODEL_FORBIDDEN_ROLES:
                    errors.append(f"world_model_role_forbidden:{role}")
                elif role not in WORLD_MODEL_ALLOWED_ROLES:
                    errors.append(f"world_model_role_unknown:{role}")
        elif roles is not None and not isinstance(roles, list):
            errors.append("world_model_roles_invalid")
    if not _string(result.get("method_id")):
        errors.append("method_id_missing")
    for key in ("expected_cost_usd", "expected_latency_seconds"):
        if not math.isfinite(_number(result.get(key))) or _number(result.get(key)) < 0:
            errors.append(f"{key}_invalid")
    if not isinstance(result.get("evidence_quality"), Mapping):
        errors.append("evidence_quality_invalid")
    if errors:
        raise MeasurementRoutingError(*errors)
    return result


def validate_measurement_qualification(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _finalize(
        value, schema=QUALIFICATION_SCHEMA_VERSION, digest_field="measurement_qualification_digest"
    )
    errors: list[str] = []
    for key in ("qualification_id", "method_id", "method_version", "capability_profile_digest"):
        if not _string(result.get(key)):
            errors.append(f"{key}_missing")
    if not _string(result.get("admission_record_digest")).startswith("sha256:"):
        errors.append("admission_record_digest_invalid")
    if result.get("admission_stage") not in {"R7", "R8"}:
        errors.append("qualification_not_catalog_admitted")
    if result.get("status") != "approved":
        errors.append("qualification_not_approved")
    if not isinstance(result.get("scope"), Mapping):
        errors.append("qualification_scope_invalid")
    if not _strings(result.get("qualified_capabilities")):
        errors.append("qualified_capabilities_missing")
    if result.get("claim_ceiling") not in CLAIM_LEVELS:
        errors.append("claim_ceiling_invalid")
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append("metrics_invalid")
    else:
        for key in (
            "physical_accuracy_error",
            "uncertainty",
            "scope_distance",
            "harmful_false_negative_rate",
            "reproducibility_score",
            "privacy_preference",
        ):
            if not math.isfinite(_number(metrics.get(key))):
                errors.append(f"qualification_metric_missing:{key}")
    approval = result.get("approval")
    if not isinstance(approval, Mapping):
        errors.append("approval_invalid")
    else:
        if approval.get("signature_status") != "verified":
            errors.append("qualification_signature_not_verified")
        approvers = _strings(approval.get("approved_by"))
        if not _string(approval.get("signature_id")) or len(approvers) < 2:
            errors.append("qualification_approval_incomplete")
        if approval.get("agent_approved") is not False:
            errors.append("agent_qualification_approval_forbidden")
    if result.get("self_grading") is not False:
        errors.append("self_grading_forbidden")
    if any(
        item not in ALL_CAPABILITY_FIELDS for item in _strings(result.get("qualified_capabilities"))
    ):
        errors.append("qualified_capability_unknown")
    expiration = _string(result.get("expiration_date"))
    try:
        date.fromisoformat(expiration)
    except ValueError:
        errors.append("expiration_date_invalid")
    if errors:
        raise MeasurementRoutingError(*errors)
    result["qualified_capabilities"] = _strings(result.get("qualified_capabilities"))
    result["measurement_qualification_digest"] = _canonical_digest(
        result, "measurement_qualification_digest"
    )
    return result


def _scope_contains(
    scope: Mapping[str, Any], requirements: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    task_scope = dict(requirements.get("task_scope") or {})
    robot_scope = dict(requirements.get("robot_scope") or {})
    checks = {
        "task_classes": [requirements.get("task_class")],
        "material_regimes": task_scope.get("material_regimes") or [],
        "robot_ids": [robot_scope.get("robot_id")],
        "end_effector_ids": [robot_scope.get("end_effector_id")],
        "controller_ids": [robot_scope.get("controller_id")],
        "sensor_ids": list(dict(requirements.get("sensor_scope") or {}).get("sensor_ids") or []),
        "site_classes": [dict(requirements.get("site_scope") or {}).get("site_class")],
        "metric_ids": task_scope.get("metric_ids") or [],
    }
    for key, raw_expected in checks.items():
        expected = {_string(item) for item in raw_expected if _string(item)}
        if not expected:
            continue
        qualified = set(_strings(scope.get(key)))
        if not qualified or not expected.issubset(qualified):
            reasons.append(f"qualification_scope_mismatch:{key}")
    requested_rate = robot_scope.get("action_rate_hz")
    if requested_rate is not None:
        allowed_rates = scope.get("action_rates_hz")
        if not (
            isinstance(allowed_rates, list)
            and len(allowed_rates) == 2
            and _number(allowed_rates[0]) <= _number(requested_rate, -math.inf)
            and _number(requested_rate, math.inf) <= _number(allowed_rates[1], -math.inf)
        ):
            reasons.append("qualification_scope_mismatch:action_rates_hz")
    requested_ranges = task_scope.get("parameter_ranges") or {}
    qualified_ranges = scope.get("parameter_ranges") or {}
    if isinstance(requested_ranges, Mapping):
        for key, requested in requested_ranges.items():
            allowed = qualified_ranges.get(key) if isinstance(qualified_ranges, Mapping) else None
            if not (
                isinstance(requested, list)
                and len(requested) == 2
                and isinstance(allowed, list)
                and len(allowed) == 2
                and _number(allowed[0]) <= _number(requested[0])
                and _number(requested[1]) <= _number(allowed[1])
            ):
                reasons.append(f"qualification_parameter_range_mismatch:{key}")
    return not reasons, reasons


def _site_evidence_available(site: Mapping[str, Any], evidence_id: str) -> bool:
    evidence = site.get("evidence")
    if not isinstance(evidence, Mapping):
        return False
    record = evidence.get(evidence_id)
    return bool(
        isinstance(record, Mapping)
        and record.get("available") is True
        and record.get("validated") is True
    )


def _candidate_assessment(
    *,
    requirements: Mapping[str, Any],
    site: Mapping[str, Any],
    profile: Mapping[str, Any],
    qualifications: Sequence[Mapping[str, Any]],
    as_of: date,
) -> dict[str, Any]:
    capabilities = dict(profile.get("capabilities") or {})
    method_id = _string(profile.get("method_id"))
    method_family = _string(capabilities.get("method_family"))
    family_ceiling = MEASUREMENT_METHOD_FAMILY_CLAIM_CEILING.get(method_family)
    reasons: list[str] = []
    rights = dict(site.get("rights") or {})
    privacy = dict(site.get("privacy") or {})
    constraints = dict(requirements.get("constraints") or {})
    robot_scope = dict(requirements.get("robot_scope") or {})
    required = set(_strings(requirements.get("required_capabilities")))
    alternative_groups = [
        tuple(_strings(group))
        for group in (requirements.get("required_capability_alternatives") or [])
        if isinstance(group, list)
    ]
    alternative_fields = {field for group in alternative_groups for field in group}
    measurable = required | alternative_fields
    supported = {key for key in measurable if capabilities.get(key) is True}
    if not supported and measurable:
        reasons.append("required_solver_capability_absent")
    if (
        constraints.get("local_only") is True
        and capabilities.get("local_offline_supported") is not True
    ):
        reasons.append("local_only_not_supported")
    if (
        constraints.get("commercial_use_required", True)
        and capabilities.get("commercial_use_allowed") is not True
    ):
        reasons.append("commercial_use_not_allowed")
    if (
        constraints.get("output_portability_required", True)
        and capabilities.get("output_export_supported") is not True
    ):
        reasons.append("output_not_portable")
    if (
        constraints.get("provider_training_use_allowed") is False
        and capabilities.get("provider_training_use_allowed") is not False
    ):
        reasons.append("provider_training_use_not_prohibited")
    maximum_retention = constraints.get("maximum_data_retention_days")
    if maximum_retention is not None and _number(capabilities.get("data_retention_days")) > _number(
        maximum_retention
    ):
        reasons.append("data_retention_exceeds_limit")
    if (
        constraints.get("data_retention_allowed") is False
        and _number(capabilities.get("data_retention_days"), math.inf) != 0.0
    ):
        reasons.append("data_retention_not_allowed")
    if (
        constraints.get("deterministic_replay_required") is True
        and _string(capabilities.get("deterministic_mode")).lower()
        not in DETERMINISTIC_REPLAY_MODES
    ):
        reasons.append("deterministic_replay_not_supported")
    allowed_regions = set(_strings(constraints.get("allowed_regions")))
    if allowed_regions:
        subprocessors = set(_strings(capabilities.get("subprocessor_regions")))
        if capabilities.get("api_only") is True and not subprocessors:
            reasons.append("subprocessor_regions_unknown")
        elif subprocessors - allowed_regions:
            reasons.append("subprocessor_region_not_allowed")
    if privacy.get("external_processing_allowed") is False and (
        capabilities.get("local_offline_supported") is not True
        or capabilities.get("api_only") is True
    ):
        reasons.append("site_privacy_requires_local_processing")
    if rights.get("commercial_evaluation_allowed") is not True:
        reasons.append("site_rights_not_cleared")
    if family_ceiling is None:
        reasons.append(f"method_family_unknown:{method_family or 'missing'}")
    requested_robot = _string(robot_scope.get("robot_id"))
    supported_embodiments = _strings(capabilities.get("supported_embodiments"))
    if requested_robot and supported_embodiments and requested_robot not in supported_embodiments:
        reasons.append("robot_interface_unsupported:robot_id")
    requested_effector = _string(robot_scope.get("end_effector_id"))
    supported_effectors = _strings(capabilities.get("supported_end_effectors"))
    if requested_effector and supported_effectors and requested_effector not in supported_effectors:
        reasons.append("robot_interface_unsupported:end_effector_id")
    requested_rate = robot_scope.get("action_rate_hz")
    if requested_rate is not None and _number(requested_rate, 0.0) > _number(
        capabilities.get("maximum_control_rate_hz"), 0.0
    ):
        reasons.append("control_rate_unsupported")
    if _number(profile.get("expected_cost_usd"), 0.0) > _number(
        constraints.get("maximum_compute_cost_usd"), math.inf
    ):
        reasons.append("cost_exceeds_hard_limit")
    if _number(profile.get("expected_latency_seconds"), 0.0) > _number(
        constraints.get("maximum_latency_seconds"), math.inf
    ):
        reasons.append("latency_exceeds_hard_limit")

    matching: list[dict[str, Any]] = []
    qualification_mismatch_reasons: list[str] = []
    for raw in qualifications:
        try:
            qualification = validate_measurement_qualification(raw)
        except MeasurementRoutingError:
            continue
        if qualification.get("method_id") != method_id:
            continue
        if qualification.get("method_version") != capabilities.get("version"):
            qualification_mismatch_reasons.append("qualification_method_version_mismatch")
            continue
        if qualification.get("capability_profile_digest") != profile.get(
            "capability_profile_digest"
        ):
            qualification_mismatch_reasons.append("qualification_capability_profile_mismatch")
            continue
        expiration = _string(qualification.get("expiration_date"))
        if expiration:
            try:
                if date.fromisoformat(expiration) < as_of:
                    qualification_mismatch_reasons.append("qualification_expired")
                    continue
            except ValueError:
                continue
        if family_ceiling is not None and (
            CLAIM_LEVELS[qualification["claim_ceiling"]] > CLAIM_LEVELS[family_ceiling]
        ):
            qualification_mismatch_reasons.append("claim_ceiling_exceeds_method_family_cap")
            continue
        if method_family in PHYSICS_AUTHORITY_FORBIDDEN_FAMILIES and (
            set(qualification["qualified_capabilities"]) & PHYSICS_AUTHORITY_CAPABILITIES
        ):
            qualification_mismatch_reasons.append("physics_authority_forbidden_for_method_family")
            continue
        contained, scope_reasons = _scope_contains(
            dict(qualification.get("scope") or {}), requirements
        )
        if not contained:
            qualification_mismatch_reasons.extend(scope_reasons)
            continue
        if (
            CLAIM_LEVELS[qualification["claim_ceiling"]]
            < CLAIM_LEVELS[requirements["requested_claim_level"]]
        ):
            qualification_mismatch_reasons.append("requested_claim_exceeds_qualification_ceiling")
            continue
        qualified_supported = supported & set(qualification["qualified_capabilities"])
        if measurable and not qualified_supported:
            qualification_mismatch_reasons.append("required_capability_not_qualified")
            continue
        matching.append(qualification)
    if not matching:
        reasons.extend(qualification_mismatch_reasons)
        reasons.append("no_exact_verified_qualification")

    qualification = min(
        matching,
        key=lambda item: (
            _number(dict(item.get("metrics") or {}).get("physical_accuracy_error")),
            _number(dict(item.get("metrics") or {}).get("uncertainty")),
            _number(dict(item.get("metrics") or {}).get("scope_distance")),
        ),
        default=None,
    )
    covered = supported & set((qualification or {}).get("qualified_capabilities") or [])
    # Explicit request-level evidence is a global hard gate.  It must not
    # disappear merely because a composite stage covers a different subset of
    # solver capabilities.
    missing_evidence = sorted(
        {
            evidence_id
            for capability in covered
            for evidence_id in CAPABILITY_SITE_EVIDENCE.get(capability, ())
            if not _site_evidence_available(site, evidence_id)
        }
        | {
            evidence_id
            for evidence_id in _strings(requirements.get("required_site_evidence"))
            if not _site_evidence_available(site, evidence_id)
        }
    )
    reasons.extend(f"required_site_evidence_missing:{item}" for item in missing_evidence)
    metrics = dict((qualification or {}).get("metrics") or {})
    rank = (
        _number(metrics.get("physical_accuracy_error")),
        _number(metrics.get("uncertainty")),
        _number(metrics.get("scope_distance")),
        _number(metrics.get("harmful_false_negative_rate")),
        -_number(metrics.get("reproducibility_score"), -math.inf),
        -_number(metrics.get("privacy_preference"), -math.inf),
        _number(profile.get("expected_latency_seconds")),
        _number(profile.get("expected_cost_usd")),
        method_id,
    )
    return {
        "method_id": method_id,
        "method_family": method_family,
        "capability_profile_digest": profile.get("capability_profile_digest"),
        "qualification_id": (qualification or {}).get("qualification_id"),
        "qualification_digest": (qualification or {}).get("measurement_qualification_digest"),
        "eligible": not reasons,
        "covered_capabilities": sorted(covered) if not reasons else [],
        "rejection_codes": sorted(set(reasons)),
        "rank": list(rank[:-1]),
        "expected_cost_usd": profile.get("expected_cost_usd", 0),
        "expected_latency_seconds": profile.get("expected_latency_seconds", 0),
        "claim_ceiling": (qualification or {}).get("claim_ceiling"),
        "residual_uncertainty": metrics.get("uncertainty"),
    }


def _smallest_next_action(
    candidates: Sequence[Mapping[str, Any]], *, requested_claim_level: str = "C3"
) -> dict[str, Any]:
    # The smallest next action is judged from the candidates closest to
    # eligibility: when some method already supports the required solver
    # capabilities, an unrelated method's capability gap must not divert the
    # answer from qualification/evidence work to adapter work.
    supporting = [
        row
        for row in candidates
        if "required_solver_capability_absent" not in (row.get("rejection_codes") or [])
    ]
    scored = supporting or candidates
    codes = sorted({code for row in scored for code in row.get("rejection_codes") or []})
    evidence_prefix = "required_site_evidence_missing:"
    missing_evidence = sorted(
        {code[len(evidence_prefix) :] for code in codes if code.startswith(evidence_prefix)}
    )
    if missing_evidence:
        actions = sorted(
            {
                EVIDENCE_SMALLEST_ACTION.get(evidence_id, "targeted_recapture")
                for evidence_id in missing_evidence
            }
        )
        return {
            "action_type": actions[0],
            "action_types": actions,
            "blocking_codes": [
                f"{evidence_prefix}{evidence_id}" for evidence_id in missing_evidence
            ],
            "exact_scope": missing_evidence,
        }
    if CLAIM_LEVELS.get(requested_claim_level, 0) >= CLAIM_LEVELS["C6"]:
        # Physical success, deployment readiness, and safety process claims can
        # never be unblocked by more simulation or a simulator benchmark.
        return {
            "action_type": "physical_execution",
            "action_types": ["physical_execution"],
            "blocking_codes": codes or ["authoritative_physical_evidence_missing"],
            "exact_scope": codes or ["authoritative_physical_evidence_missing"],
        }
    priority = (
        (("site_rights_not_cleared",), "rights_approval"),
        (
            (
                "required_solver_capability_absent",
                "robot_interface_unsupported",
                "control_rate_unsupported",
                "deterministic_replay_not_supported",
                "subprocessor_region_not_allowed",
                "subprocessor_regions_unknown",
            ),
            "adapter_work",
        ),
        (
            (
                "no_exact_verified_qualification",
                "qualification_scope_mismatch:",
                "qualification_parameter_range_mismatch:",
                "qualification_method_version_mismatch",
                "qualification_capability_profile_mismatch",
                "qualification_expired",
                "required_capability_not_qualified",
                "claim_ceiling_exceeds_method_family_cap",
                "physics_authority_forbidden_for_method_family",
            ),
            "qualification_benchmark",
        ),
        (("requested_claim_exceeds_qualification_ceiling",), "physical_execution"),
    )
    for prefixes, action in priority:
        matches = [code for code in codes if any(code.startswith(prefix) for prefix in prefixes)]
        if matches:
            return {
                "action_type": action,
                "action_types": [action],
                "blocking_codes": matches,
                "exact_scope": matches,
            }
    return {
        "action_type": "request_contract_clarification",
        "action_types": ["request_contract_clarification"],
        "blocking_codes": codes,
        "exact_scope": codes,
    }


def build_measurement_input_abstention(
    *, request_id: str, blocker: str, catalog_snapshot_hash: str
) -> dict[str, Any]:
    """Build a fully digest-bound abstention when a routing input is absent."""

    abstention = _finalize(
        {
            "schema_version": ABSTENTION_SCHEMA_VERSION,
            "abstention_code": blocker,
            "blocking_requirements": [blocker],
            "smallest_next_action": {
                "action_type": f"resolve_{blocker}",
                "blocking_codes": [blocker],
                "exact_scope": [request_id],
            },
            "prohibited_fallbacks": [
                "unqualified_method",
                "silent_provider_default",
                "agent_invented_input",
            ],
        },
        schema=ABSTENTION_SCHEMA_VERSION,
        digest_field="abstention_digest",
    )
    decision = {
        "schema_version": ROUTING_DECISION_SCHEMA_VERSION,
        "routing_id": f"{request_id}-measurement-route",
        "request_id": request_id,
        "policy_version": POLICY_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "catalog_snapshot_hash": catalog_snapshot_hash,
        "requirements_digest": _canonical_digest({"missing": blocker, "request_id": request_id}),
        "site_evidence_profile_id": "missing",
        "site_evidence_digest": _canonical_digest({"missing": blocker}),
        "status": "abstention",
        "candidates_considered": [],
        "selected_route": {"type": "abstention", "stages": []},
        "claim_boundary": {
            "permitted_claim": "none",
            "prohibited_claims": list(CLAIM_LEVELS),
            "physical_success_established": False,
            "deployment_readiness_established": False,
            "safety_certification_established": False,
        },
        "route_claim_ceiling": "none",
        "evidence_package": {"evidence_record_ids": []},
        "decision_explanation": f"routing input missing: {blocker}",
        "projected_cost_usd": 0.0,
        "projected_latency_seconds": 0.0,
        "abstention": abstention,
        "agent_selected_route": False,
        "agent_qualified_method": False,
        "execution_authorized": False,
    }
    decision["deterministic_policy_signature"] = _canonical_digest(
        decision, "deterministic_policy_signature"
    )
    return _finalize(
        decision,
        schema=ROUTING_DECISION_SCHEMA_VERSION,
        digest_field="routing_decision_digest",
    )


def _route_type(selected: Sequence[Mapping[str, Any]]) -> str:
    if not selected:
        return "abstention"
    if len(selected) > 1:
        return "composite"
    family = _string(selected[0].get("method_family"))
    if family == "captured_real_observation":
        return "direct_observation"
    if family == "physical_evidence":
        return "physical_test"
    return "single_method"


def route_task_site_measurement(
    requirements_value: Mapping[str, Any],
    site_value: Mapping[str, Any],
    profile_values: Sequence[Mapping[str, Any]],
    qualification_values: Sequence[Mapping[str, Any]],
    *,
    catalog_snapshot_hash: str,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Return a qualified single/composite measurement route or abstention."""

    requirements = validate_task_measurement_requirements(requirements_value)
    site = validate_site_evidence_profile(site_value)
    profiles = [validate_method_capability_profile(value) for value in profile_values]
    if not catalog_snapshot_hash.startswith("sha256:"):
        raise MeasurementRoutingError("catalog_snapshot_hash_invalid")
    today = as_of or date.today()
    candidates = [
        _candidate_assessment(
            requirements=requirements,
            site=site,
            profile=profile,
            qualifications=qualification_values,
            as_of=today,
        )
        for profile in profiles
    ]
    limitations = dict(site.get("limitations") or {})
    forbidden_claims = set(_strings(limitations.get("forbidden_claims")))
    claim_forbidden = bool(
        forbidden_claims & {requirements["claim_type"], requirements["requested_claim_level"]}
    )

    required = set(requirements["required_capabilities"])
    alternative_groups = [
        tuple(_strings(group))
        for group in (requirements.get("required_capability_alternatives") or [])
        if isinstance(group, list)
    ]
    alternative_tokens = {
        f"alternative:{index}:{'|'.join(group)}": set(group)
        for index, group in enumerate(alternative_groups)
    }

    def _coverage(row: Mapping[str, Any]) -> set[str]:
        covered = set(row.get("covered_capabilities") or [])
        tokens = set(covered & required)
        for token, group in alternative_tokens.items():
            if covered & group:
                tokens.add(token)
        return tokens

    # Task-level site evidence (for example real demonstrations for
    # long-horizon claims or dynamic-object tracks for HRI) is not implied by
    # any solver capability, so no candidate ever reports it missing.  It is
    # still a hard requirement of the request and blocks the route directly.
    capability_tied_evidence = frozenset(
        evidence_id for values in CAPABILITY_SITE_EVIDENCE.values() for evidence_id in values
    )
    task_level_missing_evidence = sorted(
        evidence_id
        for evidence_id in _strings(requirements.get("required_site_evidence"))
        if evidence_id not in capability_tied_evidence
        and not _site_evidence_available(site, evidence_id)
    )

    selected: list[dict[str, Any]] = []
    uncovered: set[str] = set(required) | set(alternative_tokens)
    eligible = [row for row in candidates if row["eligible"]]
    if claim_forbidden:
        eligible = []
        uncovered = {"claim_forbidden_by_site_limitations"}
    while uncovered:
        useful = [row for row in eligible if uncovered & _coverage(row)]
        if not useful:
            break
        useful.sort(
            key=lambda row: (
                tuple(row["rank"]),
                -len(uncovered & _coverage(row)),
                row["method_id"],
            )
        )
        chosen = useful[0]
        selected.append(chosen)
        uncovered -= _coverage(chosen)
        eligible = [row for row in eligible if row["method_id"] != chosen["method_id"]]

    # A claim with no solver-token requirements still needs an explicitly
    # qualified measurement method. An empty capability set is never a pass.
    if not required and not alternative_tokens and not claim_forbidden:
        qualified_claim_methods = [
            row
            for row in eligible
            if row.get("claim_ceiling") in CLAIM_LEVELS
            and CLAIM_LEVELS[row["claim_ceiling"]]
            >= CLAIM_LEVELS[requirements["requested_claim_level"]]
        ]
        if qualified_claim_methods:
            qualified_claim_methods.sort(key=lambda row: (tuple(row["rank"]), row["method_id"]))
            selected = [qualified_claim_methods[0]]
        else:
            uncovered.add(
                "authoritative_physical_evidence"
                if requirements["requested_claim_level"] in {"C6", "C7", "C8"}
                else "qualified_claim_measurement_method"
            )

    for evidence_id in task_level_missing_evidence:
        uncovered.add(f"required_site_evidence:{evidence_id}")

    total_cost = sum(_number(row.get("expected_cost_usd"), 0.0) for row in selected)
    total_latency = sum(_number(row.get("expected_latency_seconds"), 0.0) for row in selected)
    constraints = dict(requirements.get("constraints") or {})
    if total_cost > _number(constraints.get("maximum_compute_cost_usd"), math.inf):
        uncovered.add("budget_limit")
    if total_latency > _number(constraints.get("maximum_latency_seconds"), math.inf):
        uncovered.add("latency_limit")

    status = "route_selected" if not uncovered else "abstention"
    evidence_records = dict(site.get("evidence") or {})

    def _stage_evidence_ids(row: Mapping[str, Any]) -> list[str]:
        identifiers = {
            evidence_id
            for capability in row.get("covered_capabilities") or []
            for evidence_id in CAPABILITY_SITE_EVIDENCE.get(capability, ())
            if _site_evidence_available(site, evidence_id)
        }
        return sorted(
            _string(dict(evidence_records.get(evidence_id) or {}).get("record_id"))
            for evidence_id in identifiers
            if _string(dict(evidence_records.get(evidence_id) or {}).get("record_id"))
        )

    stages = (
        [
            {
                "stage_index": index,
                "purpose": "measure:" + ",".join(row["covered_capabilities"]),
                "method_id": row["method_id"],
                "method_family": row["method_family"],
                "capability_profile_digest": row["capability_profile_digest"],
                "qualification_id": row["qualification_id"],
                "qualification_digest": row["qualification_digest"],
                "covered_capabilities": list(row["covered_capabilities"]),
                "input_site_evidence_record_ids": _stage_evidence_ids(row),
                "residual_uncertainty": row.get("residual_uncertainty"),
                "permitted_metrics": list(
                    dict(requirements.get("task_scope") or {}).get("metric_ids") or []
                ),
            }
            for index, row in enumerate(selected)
        ]
        if status == "route_selected"
        else []
    )
    route_claim_ceiling = "none"
    if status == "route_selected" and selected:
        route_claim_ceiling = min(
            (row.get("claim_ceiling") for row in selected),
            key=lambda level: CLAIM_LEVELS.get(_string(level), -1),
        )
    abstention = None
    if status == "abstention":
        if claim_forbidden:
            action: dict[str, Any] = {
                "action_type": "request_contract_clarification",
                "action_types": ["request_contract_clarification"],
                "blocking_codes": ["claim_forbidden_by_site_limitations"],
                "exact_scope": sorted(
                    forbidden_claims
                    & {requirements["claim_type"], requirements["requested_claim_level"]}
                ),
            }
            abstention_code = "claim_forbidden_by_site_limitations"
        else:
            action_inputs: list[Mapping[str, Any]] = list(candidates)
            if task_level_missing_evidence:
                action_inputs.append(
                    {
                        "rejection_codes": [
                            f"required_site_evidence_missing:{evidence_id}"
                            for evidence_id in task_level_missing_evidence
                        ]
                    }
                )
            action = _smallest_next_action(
                action_inputs,
                requested_claim_level=requirements["requested_claim_level"],
            )
            abstention_code = "no_exact_qualified_measurement_route"
        if uncovered and not claim_forbidden:
            action["blocking_codes"] = sorted(
                set(action["blocking_codes"])
                | {f"uncovered_capability:{item}" for item in uncovered}
            )
        abstention_value: dict[str, Any] = {
            "schema_version": ABSTENTION_SCHEMA_VERSION,
            "abstention_code": abstention_code,
            "blocking_requirements": action["blocking_codes"],
            "smallest_next_action": action,
            "prohibited_fallbacks": [
                "unqualified_method",
                "silent_provider_default",
                "visual_similarity_as_physics",
            ],
        }
        if any(
            code.startswith("required_site_evidence_missing:") for code in action["blocking_codes"]
        ):
            abstention_value["site_evidence_audit"] = audit_site_evidence_profile(
                site, requirements
            )
        abstention = _finalize(
            abstention_value,
            schema=ABSTENTION_SCHEMA_VERSION,
            digest_field="abstention_digest",
        )

    if status == "route_selected":
        explanation = (
            f"deterministic policy {POLICY_VERSION} selected "
            f"{len(stages)} qualified stage(s) covering "
            f"{len(required)} required capability field(s) and "
            f"{len(alternative_tokens)} alternative group(s) for claim "
            f"{requirements['requested_claim_level']} at site "
            f"{site['profile_id']}; execution remains unauthorized"
        )
    else:
        explanation = f"deterministic policy {POLICY_VERSION} abstained: " + ",".join(
            sorted(uncovered)
        )
    decision = {
        "schema_version": ROUTING_DECISION_SCHEMA_VERSION,
        "routing_id": f"{requirements['request_id']}-measurement-route",
        "request_id": requirements["request_id"],
        "policy_version": POLICY_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "catalog_snapshot_hash": catalog_snapshot_hash,
        "requirements_digest": requirements["requirements_digest"],
        "site_evidence_profile_id": site["profile_id"],
        "site_evidence_digest": site["site_evidence_digest"],
        "status": status,
        "candidates_considered": sorted(candidates, key=lambda row: row["method_id"]),
        "selected_route": {
            "type": _route_type(selected) if status == "route_selected" else "abstention",
            "stages": stages,
        },
        "claim_boundary": {
            "permitted_claim": requirements["requested_claim_level"]
            if status == "route_selected"
            else "none",
            "prohibited_claims": [
                level
                for level, ordinal in CLAIM_LEVELS.items()
                if status != "route_selected"
                or ordinal > CLAIM_LEVELS[requirements["requested_claim_level"]]
            ],
            # A route permits a later measurement. Planning is never the
            # measurement itself, including for an accepted physical method.
            "physical_success_established": False,
            "deployment_readiness_established": False,
            "safety_certification_established": False,
        },
        "route_claim_ceiling": route_claim_ceiling,
        "evidence_package": {
            "site_evidence_profile_id": site["profile_id"],
            "bundle_id": site["bundle_id"],
            "bundle_hash": site["bundle_hash"],
            "evidence_record_ids": sorted(
                {
                    record_id
                    for stage in stages
                    for record_id in stage["input_site_evidence_record_ids"]
                }
            ),
        },
        "decision_explanation": explanation,
        "projected_cost_usd": total_cost if status == "route_selected" else 0.0,
        "projected_latency_seconds": total_latency if status == "route_selected" else 0.0,
        "abstention": abstention,
        "agent_selected_route": False,
        "agent_qualified_method": False,
        "execution_authorized": False,
    }
    decision["deterministic_policy_signature"] = _canonical_digest(
        decision, "deterministic_policy_signature"
    )
    return _finalize(
        decision,
        schema=ROUTING_DECISION_SCHEMA_VERSION,
        digest_field="routing_decision_digest",
    )


__all__ = [
    "ABSTENTION_SCHEMA_VERSION",
    "ALL_CAPABILITY_FIELDS",
    "APPEARANCE_ONLY_SITE_EVIDENCE",
    "CAPABILITY_FIELDS_BY_GROUP",
    "CAPABILITY_SITE_EVIDENCE",
    "CLAIM_CAPABILITIES",
    "DETERMINISTIC_REPLAY_MODES",
    "CLAIM_LEVELS",
    "CLAIM_TYPE_LEVEL",
    "CONTACT_REGIME_CAPABILITIES",
    "DEFORMATION_FAMILY_CAPABILITIES",
    "EVIDENCE_SMALLEST_ACTION",
    "FORBIDDEN_GENERIC_MATERIAL_REGIMES",
    "MATERIAL_REGIME_ALTERNATIVES",
    "MATERIAL_REGIME_CAPABILITIES",
    "MEASUREMENT_METHOD_FAMILY_CLAIM_CEILING",
    "METHOD_CAPABILITY_SCHEMA_VERSION",
    "MeasurementRoutingError",
    "PHYSICS_AUTHORITY_CAPABILITIES",
    "PHYSICS_AUTHORITY_FORBIDDEN_FAMILIES",
    "POLICY_VERSION",
    "QUALIFICATION_SCHEMA_VERSION",
    "REQUIREMENTS_SCHEMA_VERSION",
    "ROUTING_DECISION_SCHEMA_VERSION",
    "SENSOR_MODALITY_CAPABILITIES",
    "SITE_EVIDENCE_AUDIT_SCHEMA_VERSION",
    "SITE_EVIDENCE_SCHEMA_VERSION",
    "SITE_EVIDENCE_TAXONOMY",
    "SMALLEST_NEXT_ACTION_TYPES",
    "TASK_CAPABILITIES",
    "TASK_CAPABILITY_ALTERNATIVES",
    "TASK_SITE_EVIDENCE",
    "TAXONOMY_VERSION",
    "WORLD_MODEL_ALLOWED_ROLES",
    "WORLD_MODEL_FORBIDDEN_ROLES",
    "audit_site_evidence_profile",
    "build_measurement_input_abstention",
    "derive_task_measurement_requirements",
    "minimum_capability_requirements",
    "route_task_site_measurement",
    "validate_measurement_qualification",
    "validate_method_capability_profile",
    "validate_site_evidence_profile",
    "validate_task_measurement_requirements",
]
