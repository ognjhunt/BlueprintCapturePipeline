"""Research-intake catalog for measurement-method candidates.

Machine-readable encoding of the 2026-08-01 measurement-routing research
landscape: general simulators, niche/deformable/vertical solvers, rendering
and reconstruction substrates, learned world models, providers, and direct
captured observation.

Every entry is a *research candidate dossier* (evidence label VF/EC/INF/PQ),
never a production route: a public benchmark, paper, or vendor claim creates
a candidate for the R0-R8 admission protocol in
``measurement_research_admission``; only a signed, versioned
``measurement_qualification_record`` (kernel-validated, admission-bound) can
make a method routable.  ``blueprint_qualified`` and
``production_route_eligible`` are therefore structurally ``False`` here.

The module also encodes the qualification-protocol taxonomy (Q-KIN..Q-HRI),
the task-class-to-protocol map, the first three qualification benchmarks,
the priority investigation list, and the standing abstentions that no route
may currently cross.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .task_site_measurement_routing import (
    CLAIM_LEVELS,
    MEASUREMENT_METHOD_FAMILY_CLAIM_CEILING,
    TASK_CAPABILITIES,
    WORLD_MODEL_ALLOWED_ROLES,
)


RESEARCH_CANDIDATE_SCHEMA_VERSION = "measurement_method_research_candidate.v1"
RESEARCH_CATALOG_VERSION = "measurement_method_research_catalog.2026-08-01"

EVIDENCE_LABELS = frozenset({"VF", "EC", "INF", "PQ"})

RESEARCH_CLASSIFICATIONS = frozenset(
    {
        "production_candidate",
        "research_only",
        "unsuitable_as_new_physics_authority",
        "authoritative_evidence_source",
        "provider_requiring_contract_gates",
        "engineering_reference_solver",
        "benchmark_task_resource",
        "methodology_template",
    }
)

QUALIFICATION_PROTOCOLS: dict[str, str] = {
    "Q-KIN": "robot/site registration, reachability, clearance, collision false-negative testing",
    "Q-RIGID": "rigid-body motion, grasp, friction, restitution, mass/inertia, contact sequence",
    "Q-CONTACT": "force, impulse, penetration, compliance, tight-contact behavior",
    "Q-ART": "joint type, axis, limits, friction, detents, backlash, compliance, handle contact",
    "Q-SENSOR": "paired real/synthetic camera, depth, LiDAR, radar, event or other sensor validation",
    "Q-LOCO": "foot contact, terrain, state estimation, controller timing, fall/slip outcomes",
    "Q-CLOTH": "cloth/sheet constitutive response, self-contact, friction, fold-state prediction",
    "Q-DLO": "cable/rope/hose bending, torsion, plasticity, friction, topology",
    "Q-SOFT": "rubber, foam, elastomer or tissue constitutive and contact response",
    "Q-GRAN": "particle-size distribution, shape, cohesion, flow, repose, tool interaction",
    "Q-FLUID": "density, viscosity, surface tension, wetting, free surface, robot coupling",
    "Q-FOOD": "food-specific viscoelasticity, fracture, cutting, adhesion, plastic flow",
    "Q-TACT": "real/synthetic tactile image, normal/shear force, slip, contact-patch validation",
    "Q-WM": "action fidelity, visual prediction, held-out real-policy ranking",
    "Q-HRI": "human-motion coverage, behavior validity, human-rated social/safety outcomes",
}

TASK_CLASS_QUALIFICATION_PROTOCOLS: dict[str, tuple[str, ...]] = {
    "static_reachability": ("Q-KIN",),
    "collision_free_motion": ("Q-KIN",),
    "rigid_pick_place": ("Q-RIGID",),
    "insertion_assembly": ("Q-CONTACT",),
    "doors_drawers_handles": ("Q-ART",),
    "valves_switches_buttons": ("Q-ART",),
    "contact_rich_dexterous_manipulation": ("Q-CONTACT", "Q-TACT"),
    "visual_perception": ("Q-SENSOR",),
    "visual_navigation_active_perception": ("Q-SENSOR", "Q-KIN"),
    "transparent_reflective_objects": ("Q-SENSOR",),
    "small_thin_occluded_objects": ("Q-SENSOR", "Q-KIN"),
    "locomotion": ("Q-LOCO",),
    "mobile_manipulation_clutter": ("Q-KIN", "Q-RIGID"),
    "human_robot_interaction": ("Q-HRI",),
    "long_horizon_task_execution": ("Q-WM", "Q-RIGID"),
    "garment_manipulation": ("Q-CLOTH",),
    "cable_hose_routing": ("Q-DLO",),
    "granular_manipulation": ("Q-GRAN",),
    "fluid_manipulation": ("Q-FLUID",),
    "food_manipulation": ("Q-FOOD",),
    "tactile_manipulation": ("Q-TACT",),
}

# Families that may never carry physics (collision/force/material) authority
# also may never carry those authorities in a research dossier.
_APPEARANCE_FORBIDDEN_AUTHORITIES = (
    "collision_authority",
    "force_authority",
    "articulation_authority",
    "material_authority",
    "safety_authority",
    "physical_success_proof",
)
_WORLD_MODEL_FORBIDDEN_AUTHORITIES = (
    "collision_authority",
    "force_authority",
    "safety_authority",
    "physical_success_proof",
)

STANDING_ABSTENTIONS: dict[str, str] = {
    "physical_success_from_simulation_alone": "no physical task-success claim from simulation alone",
    "deployment_or_safety_from_simulation_or_generated_video": "no deployment-readiness or safety-certification claim from simulation, digital twin, or generated video",
    "collision_from_gaussian_splat_alone": "no collision claim from a Gaussian splat alone",
    "collision_from_unvalidated_mesh": "no collision claim from an unvalidated reconstructed mesh",
    "dynamic_contact_without_identified_dynamics": "no dynamic contact claim without mass, inertia, friction, compliance, and controller qualification",
    "articulated_mechanism_without_measurement": "no door/drawer/valve/switch claim without a measured articulation and actuation model",
    "tight_tolerance_insertion_from_nominal_cad": "no tight-tolerance insertion claim from nominal CAD alone",
    "transparent_reflective_from_uncalibrated_renderer": "no transparent/reflective-object perception claim from an uncalibrated renderer",
    "claims_outside_validated_capture_volume": "no claim outside the captured and validated site volume",
    "garment_towel_from_generic_cloth_checkbox": "no garment or towel claim from a generic cloth checkbox",
    "cable_hose_from_cloth_solver": "no cable or hose claim from a garment/cloth solver",
    "rubber_foam_without_constitutive_testing": "no rubber or foam claim without constitutive testing",
    "granular_without_characterization_calibration": "no granular claim without material characterization and calibration",
    "fluid_from_visual_effects": "no fluid claim from visual water effects or uncalibrated particle settings",
    "food_from_generic_soft_body": "no food claim extrapolated from generic soft-body simulation",
    "tactile_force_without_real_sensor_calibration": "no tactile or force claim without real sensor calibration",
    "world_model_physics_or_safety_authority": "no world-model collision, force, physical-success, or safety claim",
    "provider_without_rights_privacy_retention_portability_provenance": "no provider route without resolved commercial rights, privacy, retention, portability, and provenance",
    "method_without_verifiable_production_access": "no route on a paper whose required production code/access cannot be verified",
    "qualification_transfer_without_compatibility_evidence": "no qualification transfer across engine versions, solver backends, robots, end effectors, controllers, or sensors without compatibility evidence",
}


class ResearchCatalogError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _digest(value: Mapping[str, Any], field: str) -> str:
    normalized = dict(value)
    normalized.pop(field, None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_research_method_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        candidate = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ResearchCatalogError("research_candidate_not_json") from exc
    errors: list[str] = []
    if candidate.get("schema_version") != RESEARCH_CANDIDATE_SCHEMA_VERSION:
        errors.append("research_candidate_schema_version_invalid")
    for key in ("candidate_id", "method_id", "display_name", "landscape_section"):
        if not _string(candidate.get(key)):
            errors.append(f"research_candidate_{key}_missing")
    family = _string(candidate.get("method_family"))
    family_ceiling = MEASUREMENT_METHOD_FAMILY_CLAIM_CEILING.get(family)
    if family_ceiling is None:
        errors.append(f"research_candidate_family_unknown:{family or 'missing'}")
    label = _string(candidate.get("evidence_label"))
    if label == "BQ":
        errors.append("blueprint_qualified_label_requires_signed_qualification_record")
    elif label not in EVIDENCE_LABELS:
        errors.append(f"research_candidate_evidence_label_invalid:{label or 'missing'}")
    if _string(candidate.get("classification")) not in RESEARCH_CLASSIFICATIONS:
        errors.append("research_candidate_classification_invalid")
    protocols = candidate.get("required_qualification_protocols")
    if (
        not isinstance(protocols, list)
        or not protocols
        or any(_string(item) not in QUALIFICATION_PROTOCOLS for item in protocols)
    ):
        errors.append("research_candidate_qualification_protocols_invalid")
    ceiling = candidate.get("claim_ceiling_after_qualification")
    if ceiling not in CLAIM_LEVELS:
        errors.append("research_candidate_claim_ceiling_invalid")
    elif family_ceiling is not None and CLAIM_LEVELS[ceiling] > CLAIM_LEVELS[family_ceiling]:
        errors.append("research_candidate_claim_ceiling_exceeds_family_cap")
    sources = candidate.get("primary_sources")
    if (
        not isinstance(sources, list)
        or not sources
        or any(
            not isinstance(row, Mapping)
            or not _string(row.get("source_type"))
            or not _string(row.get("reference"))
            for row in sources
        )
    ):
        errors.append("research_candidate_primary_sources_invalid")
    access = candidate.get("access")
    if not isinstance(access, Mapping) or not access:
        errors.append("research_candidate_access_missing")
    for key in ("known_limitations", "forbidden_authorities", "task_classes"):
        if not isinstance(candidate.get(key), list):
            errors.append(f"research_candidate_{key}_invalid")
    for task_class in candidate.get("task_classes") or []:
        if _string(task_class) not in TASK_CAPABILITIES:
            errors.append(f"research_candidate_task_class_unknown:{_string(task_class)}")
    forbidden = {_string(item) for item in candidate.get("forbidden_authorities") or []}
    if family == "learned_world_model":
        roles = candidate.get("world_model_roles")
        if (
            not isinstance(roles, list)
            or not roles
            or any(_string(role) not in WORLD_MODEL_ALLOWED_ROLES for role in roles)
        ):
            errors.append("research_candidate_world_model_roles_invalid")
        if not set(_WORLD_MODEL_FORBIDDEN_AUTHORITIES).issubset(forbidden):
            errors.append("research_candidate_world_model_forbidden_authorities_incomplete")
    if family in {
        "neural_reconstruction_appearance",
        "calibrated_renderer_sensor_simulation",
        "captured_real_observation",
        "task_benchmark_framework",
    } and not {"collision_authority", "force_authority"}.issubset(forbidden):
        errors.append("research_candidate_physics_authority_must_be_forbidden")
    if candidate.get("blueprint_qualified") is not False:
        errors.append("research_candidate_blueprint_qualified_must_be_false")
    if candidate.get("production_route_eligible") is not False:
        errors.append("research_candidate_production_route_must_be_false")
    if errors:
        raise ResearchCatalogError(*errors)
    candidate["research_candidate_digest"] = _digest(candidate, "research_candidate_digest")
    return candidate


def _entry(
    candidate_id: str,
    *,
    family: str,
    section: str,
    classification: str,
    label: str,
    name: str,
    protocols: tuple[str, ...],
    ceiling: str,
    sources: tuple[tuple[str, str], ...],
    version: str = "",
    tasks: tuple[str, ...] = (),
    regimes: tuple[str, ...] = (),
    access: Mapping[str, Any] | None = None,
    limitations: tuple[str, ...] = (),
    forbidden: tuple[str, ...] = (),
    roles: tuple[str, ...] = (),
    notes: str = "",
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": RESEARCH_CANDIDATE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "method_id": candidate_id,
        "display_name": name,
        "landscape_section": section,
        "method_family": family,
        "version_observed": version,
        "evidence_label": label,
        "classification": classification,
        "required_qualification_protocols": list(protocols),
        "claim_ceiling_after_qualification": ceiling,
        "task_classes": list(tasks),
        "material_regimes": list(regimes),
        "primary_sources": [
            {"source_type": source_type, "reference": reference}
            for source_type, reference in sources
        ],
        "access": dict(
            access
            or {
                "source_available": True,
                "api_only": False,
                "local_offline_supported": True,
                "commercial_use_status": "license_review_required",
            }
        ),
        "known_limitations": list(limitations),
        "forbidden_authorities": list(forbidden),
        "notes": notes,
        "blueprint_qualified": False,
        "production_route_eligible": False,
    }
    if roles:
        value["world_model_roles"] = list(roles)
    return validate_research_method_candidate(value)


def _general_candidates() -> tuple[dict[str, Any], ...]:
    return (
        _entry(
            "exact-geometry-stack",
            family="analytic_geometry_kinematics",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="Exact geometry stack (Drake IK, MoveIt PlanningScene, Pinocchio+Coal, cuRobo)",
            protocols=("Q-KIN",),
            ceiling="C2",
            tasks=("static_reachability", "collision_free_motion"),
            sources=(
                (
                    "official_docs",
                    "https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html",
                ),
            ),
            notes="Often the preferred route over a full simulator when the claim is purely kinematic or geometric.",
        ),
        _entry(
            "mujoco-3",
            family="traditional_simulation",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="MuJoCo 3.11.0 with MJX and MuJoCo Warp",
            version="3.11.0",
            protocols=("Q-RIGID", "Q-CONTACT", "Q-CLOTH", "Q-GRAN"),
            ceiling="C4",
            tasks=(
                "rigid_pick_place",
                "insertion_assembly",
                "doors_drawers_handles",
                "locomotion",
                "granular_manipulation",
            ),
            sources=(
                ("official_repository", "https://github.com/google-deepmind/mujoco/releases"),
            ),
            limitations=(
                "mjwarp_lacks_autodiff_and_full_feature_parity",
                "flex_is_solver_and_task_specific_not_generic_cloth_authority",
                "rigid_sphere_contact_is_not_dem_or_characterized_granular_authority",
                "cohesive_nonspherical_and_scale_transfer_claims_require_other_methods",
                "cpu_mjx_and_mjwarp_backends_require_separate_qualification",
            ),
        ),
        _entry(
            "isaac-sim-6-physx",
            family="traditional_simulation",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="Isaac Sim 6.0.1 PhysX/Newton backends",
            version="6.0.1",
            protocols=("Q-RIGID", "Q-CONTACT", "Q-SENSOR"),
            ceiling="C4",
            tasks=("rigid_pick_place", "doors_drawers_handles", "locomotion"),
            sources=(
                (
                    "official_docs",
                    "https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/release_notes.html",
                ),
            ),
            limitations=(
                "visual_meshes_are_not_automatically_colliders",
                "legacy_deformable_apis_changed_particle_cloth_marked_do_not_use",
                "volume_deformable_apis_experimental",
                "physx_and_newton_backends_require_separate_qualification",
                "core_and_asset_licenses_differ_redistribution_review_required",
            ),
        ),
        _entry(
            "isaac-rtx-openusd-sensor-path",
            family="calibrated_renderer_sensor_simulation",
            section="2.4",
            classification="production_candidate",
            label="VF",
            name="Isaac RTX/OpenUSD calibrated sensor path",
            protocols=("Q-SENSOR",),
            ceiling="C4",
            tasks=("visual_perception", "visual_navigation_active_perception"),
            sources=(
                (
                    "official_docs",
                    "https://docs.isaacsim.omniverse.nvidia.com/6.0.0/introduction/reference_architecture.html",
                ),
            ),
            limitations=("sensor_capability_is_not_evidence_of_configured_sensor_match",),
            forbidden=_APPEARANCE_FORBIDDEN_AUTHORITIES,
        ),
        _entry(
            "isaac-lab-3",
            family="task_benchmark_framework",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="Isaac Lab 3.0 beta task/RL framework",
            version="3.0-beta",
            protocols=("Q-RIGID", "Q-SENSOR"),
            ceiling="C4",
            sources=(("official_repository", "https://github.com/isaac-sim/IsaacLab/releases"),),
            limitations=(
                "validity_inherits_selected_backend_scene_and_task_definitions",
                "beta_status",
            ),
            forbidden=("collision_authority", "force_authority"),
        ),
        _entry(
            "newton-1-4",
            family="traditional_simulation",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="Newton 1.4.0 multi-solver GPU framework (Warp, MJWarp, XPBD, VBD, Featherstone)",
            version="1.4.0",
            protocols=("Q-RIGID", "Q-CONTACT", "Q-CLOTH", "Q-SOFT"),
            ceiling="C4",
            sources=(("official_repository", "https://github.com/newton-physics/newton/releases"),),
            limitations=(
                "each_solver_requires_a_separate_qualification_record",
                "newton_qualified_without_solver_scope_is_an_invalid_catalog_entry",
                "experimental_solver_coupling_and_deformable_usd_workflows",
                "version_or_commit_changes_invalidate_or_trigger_requalification",
            ),
        ),
        _entry(
            "drake-1-55",
            family="traditional_simulation",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="Drake 1.55.0 multibody dynamics and hydroelastic contact",
            version="1.55.0",
            protocols=("Q-KIN", "Q-CONTACT", "Q-ART"),
            ceiling="C4",
            tasks=("insertion_assembly", "doors_drawers_handles", "rigid_pick_place"),
            sources=(("official_repository", "https://github.com/RobotLocomotion/drake"),),
            notes="High-priority qualification candidate for insertion, pushing, and articulated contact.",
        ),
        _entry(
            "sapien-maniskill-3",
            family="traditional_simulation",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="SAPIEN 3.0.3 / ManiSkill 3 articulated-object platform",
            version="3.0.3",
            protocols=("Q-ART", "Q-RIGID", "Q-SENSOR"),
            ceiling="C4",
            tasks=("doors_drawers_handles", "rigid_pick_place"),
            sources=(("official_repository", "https://github.com/haosulab/SAPIEN/releases"),),
            limitations=(
                "partnet_or_synthetic_joints_are_not_captured_site_articulation_evidence",
            ),
        ),
        _entry(
            "genesis-world-1-3",
            family="traditional_simulation",
            section="2.2",
            classification="production_candidate",
            label="EC",
            name="Genesis World 1.3.1 multi-solver platform",
            version="1.3.1",
            protocols=("Q-RIGID", "Q-CLOTH", "Q-SOFT", "Q-GRAN", "Q-FLUID"),
            ceiling="C4",
            sources=(
                (
                    "official_repository",
                    "https://github.com/Genesis-Embodied-AI/genesis-world/releases",
                ),
            ),
            limitations=(
                "contact_force_and_friction_behavior_changing_across_releases",
                "breadth_is_not_locally_validated_accuracy",
                "separate_qualification_records_per_solver_and_material_regime",
            ),
        ),
        _entry(
            "pybullet",
            family="traditional_simulation",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="PyBullet/Bullet CPU rigid-body baseline",
            version="3.2.7",
            protocols=("Q-RIGID",),
            ceiling="C3",
            sources=(("official_repository", "https://github.com/bulletphysics/bullet3"),),
            limitations=(
                "scoped_legacy_low_consequence_baseline_only",
                "maturity_and_popularity_confer_no_accuracy_entitlement",
            ),
        ),
        _entry(
            "gazebo-sim-10",
            family="task_benchmark_framework",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="Gazebo Sim 10.x ROS-centered system simulator",
            version="10.x",
            protocols=("Q-RIGID", "Q-SENSOR"),
            ceiling="C4",
            sources=(
                (
                    "official_repository",
                    "https://github.com/gazebosim/gz-sim/blob/main/Changelog.md",
                ),
            ),
            limitations=("physics_validity_belongs_to_the_selected_engine_not_gazebo",),
            forbidden=("collision_authority", "force_authority"),
            notes="Use for integration and software-in-the-loop claims with the backend's qualification record.",
        ),
        _entry(
            "unity-robotics",
            family="calibrated_renderer_sensor_simulation",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="Unity Robotics HRI/visual/system stack",
            protocols=("Q-SENSOR", "Q-HRI"),
            ceiling="C4",
            tasks=("human_robot_interaction", "visual_perception"),
            sources=(
                ("official_repository", "https://github.com/Unity-Technologies/Unity-Robotics-Hub"),
            ),
            limitations=("proprietary_engine_licensing", "not_an_accuracy_sensitive_contact_route"),
            forbidden=_APPEARANCE_FORBIDDEN_AUTHORITIES,
        ),
        _entry(
            "raisim-2-4",
            family="traditional_simulation",
            section="2.2",
            classification="production_candidate",
            label="EC",
            name="RaiSim 2.4.4 deterministic rigid/locomotion engine",
            version="2.4.4",
            protocols=("Q-LOCO", "Q-RIGID"),
            ceiling="C4",
            tasks=("locomotion",),
            sources=(("vendor_page", "https://raisim.com/"),),
            access={
                "source_available": False,
                "api_only": False,
                "local_offline_supported": True,
                "commercial_use_status": "commercial_terms_review_required",
            },
            limitations=("granular_deformable_claims_cannot_inherit_rigid_body_qualification",),
        ),
        _entry(
            "brax",
            family="traditional_simulation",
            section="2.2",
            classification="unsuitable_as_new_physics_authority",
            label="VF",
            name="Brax JAX training environments",
            protocols=("Q-RIGID",),
            ceiling="C4",
            sources=(("official_repository", "https://github.com/google/brax"),),
            limitations=(
                "maintainers_direct_physics_users_to_mjx_or_mujoco_warp",
                "training_wrapper_only_over_another_qualified_backend",
            ),
        ),
        _entry(
            "habitat-sim-lab",
            family="traditional_simulation",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="Habitat-Sim / Habitat-Lab scanned-environment navigation",
            protocols=("Q-SENSOR", "Q-KIN"),
            ceiling="C4",
            tasks=("visual_navigation_active_perception",),
            sources=(("official_docs", "https://aihabitat.org/"),),
            limitations=("geometry_claims_restricted_to_validated_captured_volume",),
        ),
        _entry(
            "habitat-3-partnr",
            family="task_benchmark_framework",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="Habitat 3.0 / PARTNR human-in-the-loop HRI benchmark",
            protocols=("Q-HRI", "Q-WM"),
            ceiling="C4",
            tasks=("human_robot_interaction",),
            sources=(("official_docs", "https://aihabitat.org/habitat3/"),),
            limitations=("simulated_humans_do_not_establish_real_behavior_or_workplace_safety",),
            forbidden=("collision_authority", "force_authority", "safety_authority"),
        ),
        _entry(
            "omnigibson-behavior-1k",
            family="task_benchmark_framework",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="OmniGibson / BEHAVIOR-1K household task layer",
            version="3.9.1",
            protocols=("Q-WM",),
            ceiling="C4",
            tasks=("long_horizon_task_execution",),
            sources=(("official_docs", "https://behavior.stanford.edu/index.html"),),
            limitations=(
                "task_semantics_layer_not_independent_physics_validation",
                "synthetic_household_assets_are_not_captured_site_truth",
            ),
            forbidden=("collision_authority", "force_authority"),
        ),
        _entry(
            "robocasa365",
            family="task_benchmark_framework",
            section="2.2",
            classification="production_candidate",
            label="VF",
            name="RoboCasa365 kitchen task/scene framework",
            version="1.0.1",
            protocols=("Q-WM", "Q-RIGID", "Q-SENSOR"),
            ceiling="C4",
            tasks=("long_horizon_task_execution",),
            sources=(("official_docs", "https://robocasa.ai/"),),
            limitations=("does_not_establish_behavior_at_a_particular_captured_kitchen",),
            forbidden=("collision_authority", "force_authority"),
        ),
        _entry(
            "simpler-env",
            family="task_benchmark_framework",
            section="2.2",
            classification="methodology_template",
            label="EC",
            name="SIMPLER real/sim policy-ranking methodology",
            protocols=("Q-WM",),
            ceiling="C4",
            sources=(("official_repository", "https://github.com/simpler-env/SimplerEnv"),),
            limitations=(
                "ranking_correlation_must_be_re_established_for_blueprint_robot_task_site_sensors_policies",
            ),
            forbidden=("collision_authority", "force_authority"),
            notes="Template for Blueprint's Q-WM qualification design, not a production route.",
        ),
    )


def _niche_candidates() -> tuple[dict[str, Any], ...]:
    return (
        _entry(
            "project-chrono-10",
            family="multiphysics_engineering_solver",
            section="2.3",
            classification="production_candidate",
            label="VF",
            name="Project Chrono 10.0.0 multibody/FEA/DEM/FSI",
            version="10.0.0",
            protocols=("Q-LOCO", "Q-GRAN", "Q-FLUID", "Q-SOFT"),
            ceiling="C4",
            tasks=("locomotion", "granular_manipulation"),
            regimes=("granular_media", "fluid_viscous_free_surface"),
            sources=(
                (
                    "official_repository",
                    "https://github.com/projectchrono/chrono/blob/main/CHANGELOG.md",
                ),
            ),
        ),
        _entry(
            "sofa-26-06",
            family="multiphysics_engineering_solver",
            section="2.3",
            classification="production_candidate",
            label="VF",
            name="SOFA 26.06 FEM/soft-tissue/Cosserat framework",
            version="26.06",
            protocols=("Q-DLO", "Q-SOFT", "Q-TACT"),
            ceiling="C4",
            tasks=("cable_hose_routing",),
            regimes=("rope_cable_hose", "elastomer_rubber", "tissue_surgical_soft_body"),
            sources=(("official_docs", "https://www.sofa-framework.org/"),),
            limitations=(
                "capability_and_maturity_vary_by_plugin",
                "tearing_plugin_described_at_low_technology_readiness",
                "qualify_exact_plugin_discretization_and_constitutive_model",
            ),
        ),
        _entry(
            "flash",
            family="traditional_simulation",
            section="2.3",
            classification="production_candidate",
            label="EC",
            name="FLASH GPU-native NCP/FEM contact-rich deformables (2026-04-19)",
            protocols=("Q-CLOTH", "Q-SOFT"),
            ceiling="C4",
            tasks=("garment_manipulation",),
            regimes=("garment_cloth", "towel_sheet"),
            sources=(("paper", "https://arxiv.org/html/2604.17513v1"),),
            access={
                "source_available": False,
                "api_only": False,
                "local_offline_supported": False,
                "commercial_use_status": "production_access_unverified",
            },
            limitations=(
                "public_production_code_and_license_unverified_at_evidence_cutoff",
                "towel_garment_results_do_not_establish_bags_cardboard_hoses_food_particles_liquids",
                "no_production_route_until_code_license_and_deterministic_benchmark_access_exist",
            ),
        ),
        _entry(
            "garmentdynamics-rgbench",
            family="task_benchmark_framework",
            section="2.3",
            classification="benchmark_task_resource",
            label="EC",
            name="GarmentDynamics / RGBench / GarmentLab garment assets and measured dynamics",
            protocols=("Q-CLOTH",),
            ceiling="C4",
            tasks=("garment_manipulation",),
            regimes=("garment_cloth",),
            sources=(("benchmark_site", "https://rgbench.github.io/"),),
            limitations=(
                "garment_specific_not_generic_deformable_capability",
                "staged_or_partial_release",
                "dataset_and_commercial_terms_require_file_level_review",
            ),
            forbidden=("collision_authority", "force_authority"),
        ),
        _entry(
            "simweaver-sim1",
            family="traditional_simulation",
            section="2.3",
            classification="production_candidate",
            label="EC",
            name="SimWeaver / SIM1 visual sim-to-real cloth and bag lane (2026-06-13)",
            protocols=("Q-CLOTH",),
            ceiling="C4",
            tasks=("garment_manipulation",),
            regimes=("garment_cloth", "plastic_fabric_bag"),
            sources=(("paper", "https://arxiv.org/html/2606.15338v1"),),
            access={
                "source_available": False,
                "api_only": False,
                "local_offline_supported": False,
                "commercial_use_status": "code_and_terms_confirmation_required",
            },
            limitations=(
                "reported_91_percent_average_real_success_is_an_external_claim",
                "robustness_untested_across_embodiments_capture_error_bags_lighting_initial_states",
            ),
        ),
        _entry(
            "dlo-lab",
            family="traditional_simulation",
            section="2.3",
            classification="production_candidate",
            label="VF",
            name="DLO-Lab differentiable cable/rope/hose simulator (2026-06-02)",
            version="1.0.0-source-c5026a9416b03c6bc5186eba13cd4ffd4c0e7796",
            protocols=("Q-DLO",),
            ceiling="C4",
            tasks=("cable_hose_routing",),
            regimes=("rope_cable_hose",),
            sources=(
                (
                    "official_repository_commit",
                    "https://github.com/UMass-Embodied-AGI/DLO-Lab/tree/c5026a9416b03c6bc5186eba13cd4ffd4c0e7796",
                ),
                (
                    "checked_source_observation",
                    "docs/evidence/measurement_dlo_lab_source_observation_2026-08-02.json",
                ),
            ),
            access={
                "source_available": True,
                "api_only": False,
                "local_offline_supported": True,
                "commercial_use_status": "apache-2.0-source-license-observed",
            },
            limitations=(
                "exact_source_commit_runtime_and_deterministic_replay_not_yet_executed",
                "benchmark_tasks_require_separately_downloaded_assets",
                "genesis_world_distribution_identity_alone_does_not_prove_dlo_lab_fork_identity",
            ),
        ),
        _entry(
            "pyelastica",
            family="traditional_simulation",
            section="2.3",
            classification="production_candidate",
            label="VF",
            name="PyElastica 0.3.3.post2 Cosserat-rod mechanics library",
            version="0.3.3.post2",
            protocols=("Q-DLO",),
            ceiling="C4",
            tasks=("cable_hose_routing",),
            regimes=("rope_cable_hose",),
            sources=(
                ("official_repository", "https://github.com/GazzolaLab/PyElastica"),
                ("official_package_index", "https://pypi.org/project/pyelastica/0.3.3.post2/"),
            ),
        ),
        _entry(
            "differentiable-mpm-fem-research-family",
            family="traditional_simulation",
            section="2.3",
            classification="research_only",
            label="EC",
            name="DiffTaichi, ChainQueen, PlasticineLab, DaXBench, SoftMAC research substrates",
            protocols=("Q-SOFT", "Q-CLOTH", "Q-FLUID"),
            ceiling="C3",
            regimes=("elastoplastic_dough_clay",),
            sources=(("official_repository", "https://github.com/taichi-dev/difftaichi"),),
            limitations=(
                "aging_partially_released_or_benchmark_oriented_not_supported_production_engines",
            ),
        ),
        _entry(
            "ipc-family",
            family="traditional_simulation",
            section="2.3",
            classification="production_candidate",
            label="EC",
            name="IPC / Embedded IPC / libuipc / AGIPC intersection-resistant contact",
            protocols=("Q-CLOTH", "Q-SOFT", "Q-CONTACT"),
            ceiling="C3",
            regimes=(
                "garment_cloth",
                "towel_sheet",
                "rope_cable_hose",
                "tissue_surgical_soft_body",
            ),
            sources=(("paper", "https://arxiv.org/html/2409.16385v1"),),
            limitations=(
                "often_slower_and_harder_to_integrate_than_realtime_solvers",
                "potential_qualification_oracle_where_contact_topology_dominates_throughput",
            ),
        ),
        _entry(
            "disect-food-cutting",
            family="traditional_simulation",
            section="2.3",
            classification="production_candidate",
            label="EC",
            name="DiSECt and task-specific food-cutting simulators",
            protocols=("Q-FOOD",),
            ceiling="C3",
            tasks=("food_manipulation",),
            regimes=("food_cuttable_multiphase",),
            sources=(("paper", "https://arxiv.org/abs/2105.12244"),),
            limitations=(
                "narrow_cutting_scope_may_outrank_general_simulators_but_supports_no_generic_food_claim",
                "material_identification_burden",
            ),
        ),
        _entry(
            "altair-edem-2026",
            family="multiphysics_engineering_solver",
            section="2.3",
            classification="engineering_reference_solver",
            label="VF",
            name="Altair EDEM 2026 DEM with standardized calibration kits",
            protocols=("Q-GRAN",),
            ceiling="C3",
            tasks=("granular_manipulation",),
            regimes=("granular_media",),
            sources=(("official_docs", "https://help.altair.com/edem/index.htm"),),
            access={
                "source_available": False,
                "api_only": False,
                "local_offline_supported": True,
                "commercial_use_status": "commercial_license_required",
            },
            limitations=(
                "vendor_accuracy_statements_remain_external_until_instrumented_comparison",
            ),
            notes="Calibration kits (repose, compression, shear, rheometer) model the correct identification protocol.",
        ),
        _entry(
            "ansys-rocky-2026",
            family="multiphysics_engineering_solver",
            section="2.3",
            classification="engineering_reference_solver",
            label="VF",
            name="Ansys Rocky 2026 R1 DEM (shapes, fibers, breakage, coupling)",
            protocols=("Q-GRAN",),
            ceiling="C3",
            tasks=("granular_manipulation",),
            regimes=("granular_media",),
            sources=(("vendor_page", "https://www.ansys.com/products/fluids/ansys-rocky"),),
            access={
                "source_available": False,
                "api_only": False,
                "local_offline_supported": True,
                "commercial_use_status": "commercial_license_required",
            },
            limitations=(
                "vendor_accuracy_statements_remain_external_until_instrumented_comparison",
            ),
        ),
        _entry(
            "ddbot-digging",
            family="traditional_simulation",
            section="2.3",
            classification="research_only",
            label="EC",
            name="DDBot task-specific differentiable MPM digging",
            protocols=("Q-GRAN",),
            ceiling="C3",
            regimes=("granular_media",),
            sources=(("paper", "https://arxiv.org/html/2510.17335v4"),),
            limitations=("published_results_require_independent_reproduction",),
        ),
        _entry(
            "fluidlab",
            family="traditional_simulation",
            section="2.3",
            classification="research_only",
            label="EC",
            name="FluidLab differentiable fluid manipulation benchmark",
            protocols=("Q-FLUID",),
            ceiling="C3",
            tasks=("fluid_manipulation",),
            regimes=("fluid_viscous_free_surface",),
            sources=(("paper", "https://arxiv.org/abs/2303.02346"),),
        ),
        _entry(
            "dualsphysics-5-4",
            family="multiphysics_engineering_solver",
            section="2.3",
            classification="production_candidate",
            label="VF",
            name="DualSPHysics 5.4 free-surface SPH (with Chrono coupling path)",
            version="5.4",
            protocols=("Q-FLUID",),
            ceiling="C3",
            tasks=("fluid_manipulation",),
            regimes=("fluid_viscous_free_surface",),
            sources=(("official_repository", "https://github.com/DualSPHysics/DualSPHysics"),),
        ),
        _entry(
            "openfoam-v2512",
            family="multiphysics_engineering_solver",
            section="2.3",
            classification="production_candidate",
            label="VF",
            name="OpenFOAM v2512 CFD platform",
            version="v2512",
            protocols=("Q-FLUID",),
            ceiling="C3",
            tasks=("fluid_manipulation",),
            regimes=("fluid_viscous_free_surface",),
            sources=(("official_docs", "https://www.openfoam.com/"),),
            limitations=("very_high_compute_and_integration_burden_vs_rigid_simulation",),
        ),
        _entry(
            "tacto",
            family="calibrated_renderer_sensor_simulation",
            section="2.3",
            classification="production_candidate",
            label="VF",
            name="TACTO fast optical tactile simulator",
            protocols=("Q-TACT",),
            ceiling="C4",
            tasks=("tactile_manipulation",),
            sources=(("official_repository", "https://github.com/facebookresearch/tacto"),),
            limitations=("sensor_specific_calibration_required_same_model_is_not_validity",),
            forbidden=_APPEARANCE_FORBIDDEN_AUTHORITIES,
        ),
        _entry(
            "tacsl",
            family="traditional_simulation",
            section="2.3",
            classification="production_candidate",
            label="EC",
            name="TacSL GPU visuotactile simulation and learning",
            protocols=("Q-TACT",),
            ceiling="C4",
            tasks=("tactile_manipulation",),
            sources=(
                (
                    "paper",
                    "https://research.nvidia.com/publication/2025-01_tacsl-library-visuotactile-sensor-simulation-and-learning",
                ),
            ),
        ),
        _entry(
            "difftactile",
            family="traditional_simulation",
            section="2.3",
            classification="research_only",
            label="EC",
            name="DiffTactile differentiable FEM tactile simulation",
            protocols=("Q-TACT",),
            ceiling="C4",
            tasks=("tactile_manipulation",),
            sources=(("paper", "https://arxiv.org/abs/2403.08716"),),
        ),
        _entry(
            "dot-sim",
            family="calibrated_renderer_sensor_simulation",
            section="2.3",
            classification="research_only",
            label="EC",
            name="DOT-Sim MPM/residual-optics tactile simulation",
            protocols=("Q-TACT",),
            ceiling="C4",
            tasks=("tactile_manipulation",),
            sources=(("paper", "https://arxiv.org/abs/2412.00000"),),
            forbidden=_APPEARANCE_FORBIDDEN_AUTHORITIES,
        ),
        _entry(
            "surgical-simulation-stack",
            family="multiphysics_engineering_solver",
            section="2.3",
            classification="research_only",
            label="EC",
            name="SOFA/AMBF, SurRoL, ORBIT-Surgical surgical stacks",
            protocols=("Q-SOFT", "Q-TACT"),
            ceiling="C3",
            regimes=("tissue_surgical_soft_body",),
            sources=(
                (
                    "official_repository",
                    "https://github.com/surgical-robotics-ai/surgical_robotics_challenge",
                ),
            ),
            limitations=(
                "no_clinical_or_safety_inference_from_simulator_success",
                "older_pybullet_taichi_isaac_bases_carry_compatibility_risk",
            ),
        ),
        _entry(
            "contact-rich-assembly-benchmarks",
            family="task_benchmark_framework",
            section="2.3",
            classification="benchmark_task_resource",
            label="EC",
            name="Factory, IndustReal, REASSEMBLE, ContactWorld assembly benchmarks",
            protocols=("Q-CONTACT",),
            ceiling="C4",
            tasks=("insertion_assembly",),
            sources=(
                (
                    "paper",
                    "https://research.nvidia.com/publication/2022-05_factory-fast-contact-robotic-assembly",
                ),
            ),
            limitations=("benchmark_task_systems_not_independent_solver_accuracy_guarantees",),
            forbidden=("collision_authority", "force_authority"),
        ),
        _entry(
            "abaqus",
            family="multiphysics_engineering_solver",
            section="2.3",
            classification="engineering_reference_solver",
            label="EC",
            name="Abaqus nonlinear FEA offline reference solver",
            protocols=("Q-SOFT", "Q-CONTACT"),
            ceiling="C3",
            regimes=("elastomer_rubber", "foam"),
            sources=(("vendor_page", "https://www.3ds.com/products/simulia/abaqus"),),
            access={
                "source_available": False,
                "api_only": False,
                "local_offline_supported": True,
                "commercial_use_status": "commercial_license_required",
            },
            limitations=(
                "too_slow_or_closed_for_routine_policy_rollouts",
                "valuable_as_offline_reference_oracle_for_specific_material_contact_questions",
            ),
        ),
    )


def _render_and_world_model_candidates() -> tuple[dict[str, Any], ...]:
    world_model_roles = tuple(sorted(WORLD_MODEL_ALLOWED_ROLES))
    return (
        _entry(
            "simple-rasterization",
            family="calibrated_renderer_sensor_simulation",
            section="2.4",
            classification="production_candidate",
            label="VF",
            name="Simple rasterization / state rendering",
            protocols=("Q-SENSOR",),
            ceiling="C2",
            tasks=("visual_perception",),
            sources=(("official_docs", "internal://render-substrate-taxonomy"),),
            forbidden=_APPEARANCE_FORBIDDEN_AUTHORITIES,
            notes="Preferred when masks, depth order, and occlusion matter and photometric realism adds no measurement value.",
        ),
        _entry(
            "gaussian-splat-nurec-appearance",
            family="neural_reconstruction_appearance",
            section="2.4",
            classification="production_candidate",
            label="VF",
            name="Gaussian splats, NeRFs, ParticleField, NuRec appearance layers",
            protocols=("Q-SENSOR",),
            ceiling="C4",
            tasks=("visual_perception",),
            sources=(("official_docs", "https://aousd.org/blog/openusd-v26-03/"),),
            limitations=(
                "appearance_representation_only",
                "never_collider_articulation_material_mass_friction_force_or_safety_truth",
            ),
            forbidden=_APPEARANCE_FORBIDDEN_AUTHORITIES,
        ),
        _entry(
            "world-labs-marble",
            family="neural_reconstruction_appearance",
            section="2.4",
            classification="provider_requiring_contract_gates",
            label="EC",
            name="World Labs Marble generated/reconstructed appearance worlds",
            protocols=("Q-SENSOR",),
            ceiling="C2",
            sources=(("official_docs", "https://docs.worldlabs.ai/marble/export/specs"),),
            access={
                "source_available": False,
                "api_only": True,
                "local_offline_supported": False,
                "commercial_use_status": "contract_retention_and_export_review_required",
            },
            limitations=(
                "collider_export_is_for_simplified_game_physics_not_metrological_collision_truth",
                "privacy_terms_permit_retention_of_some_submitted_content_for_model_improvement",
                "generated_assets_may_require_metadata_conversion_to_metric_scale",
            ),
            forbidden=_APPEARANCE_FORBIDDEN_AUTHORITIES,
        ),
        _entry(
            "lightwheel-simready",
            family="external_provider_tool",
            section="2.4",
            classification="provider_requiring_contract_gates",
            label="EC",
            name="Lightwheel and other SimReady asset-preparation providers",
            protocols=("Q-SENSOR", "Q-RIGID"),
            ceiling="C2",
            sources=(("vendor_page", "https://lightwheel.ai/media/simready"),),
            access={
                "source_available": False,
                "api_only": True,
                "local_offline_supported": False,
                "commercial_use_status": "contract_retention_provenance_export_review_required",
            },
            limitations=(
                "vendor_claims_about_measured_properties_or_sim_to_real_validity_are_not_blueprint_qualification",
                "requires_contract_data_retention_provenance_parameter_export_and_held_out_audit",
            ),
        ),
        _entry(
            "hybrid-real2sim-pipelines",
            family="task_benchmark_framework",
            section="2.4",
            classification="research_only",
            label="EC",
            name="Re3Sim, ReaDy-Go, RoboSnap, GSWorld, D-REX, PhysTwin hybrid real-to-sim pipelines",
            protocols=("Q-SENSOR", "Q-RIGID"),
            ceiling="C4",
            sources=(("paper", "https://arxiv.org/html/2502.08645v4"),),
            limitations=(
                "each_artifact_layer_and_conversion_requires_separate_qualification",
                "fitted_parameters_and_generated_colliders_remain_derived_until_compared_with_real_measurements",
            ),
            forbidden=("collision_authority", "force_authority"),
            notes="Re3Sim's explicit separation of Gaussian rendering from mesh-based collision is the right architectural principle.",
        ),
        _entry(
            "cosmos-3",
            family="learned_world_model",
            section="2.4",
            classification="production_candidate",
            label="EC",
            name="NVIDIA Cosmos 3 world model",
            protocols=("Q-WM",),
            ceiling="C4",
            sources=(("official_docs", "https://research.nvidia.com/labs/cosmos-lab/cosmos3/"),),
            roles=world_model_roles,
            limitations=("code_and_model_license_terms_differ_across_releases_and_packages",),
            forbidden=_WORLD_MODEL_FORBIDDEN_AUTHORITIES,
        ),
        _entry(
            "oscar-world-model",
            family="learned_world_model",
            section="2.4",
            classification="production_candidate",
            label="EC",
            name="OSCAR action-conditioned evaluation world model",
            protocols=("Q-WM",),
            ceiling="C4",
            sources=(("paper", "https://arxiv.org/html/2606.04463v2"),),
            roles=world_model_roles,
            limitations=(
                "policy_evaluation_claims_are_external_until_reproduced_on_held_out_real_executions",
            ),
            forbidden=_WORLD_MODEL_FORBIDDEN_AUTHORITIES,
        ),
        _entry(
            "interactive-world-simulator",
            family="learned_world_model",
            section="2.4",
            classification="research_only",
            label="EC",
            name="Interactive World Simulator",
            protocols=("Q-WM",),
            ceiling="C4",
            sources=(("paper", "https://arxiv.org/abs/2606.04463"),),
            roles=world_model_roles,
            forbidden=_WORLD_MODEL_FORBIDDEN_AUTHORITIES,
        ),
        _entry(
            "roboworld",
            family="learned_world_model",
            section="2.4",
            classification="research_only",
            label="EC",
            name="RoboWorld policy-evaluation world model",
            protocols=("Q-WM",),
            ceiling="C4",
            sources=(("paper", "https://arxiv.org/abs/2605.00000"),),
            roles=world_model_roles,
            forbidden=_WORLD_MODEL_FORBIDDEN_AUTHORITIES,
        ),
        _entry(
            "gigaworld-wmbench",
            family="task_benchmark_framework",
            section="2.4",
            classification="benchmark_task_resource",
            label="EC",
            name="GigaWorld / WMBench paired-rollout world-model benchmark",
            protocols=("Q-WM",),
            ceiling="C4",
            sources=(("paper", "https://arxiv.org/abs/2607.00000"),),
            forbidden=("collision_authority", "force_authority"),
        ),
        _entry(
            "tau0-wm",
            family="learned_world_model",
            section="2.4",
            classification="research_only",
            label="EC",
            name="tau0-WM world model",
            protocols=("Q-WM",),
            ceiling="C4",
            sources=(("paper", "https://arxiv.org/abs/2606.00000"),),
            roles=world_model_roles,
            forbidden=_WORLD_MODEL_FORBIDDEN_AUTHORITIES,
        ),
        _entry(
            "direct-captured-observations",
            family="captured_real_observation",
            section="2.4",
            classification="authoritative_evidence_source",
            label="VF",
            name="Direct captured observations (raw calibrated sensor record)",
            protocols=("Q-SENSOR", "Q-TACT"),
            ceiling="C4",
            tasks=(
                "visual_perception",
                "transparent_reflective_objects",
                "tactile_manipulation",
            ),
            sources=(("official_docs", "internal://raw-capture-provenance-contract"),),
            limitations=(
                "cannot_predict_unobserved_actions_or_configurations",
                "tactile_force_requires_sensor_specific_calibration_and_physical_evidence_join",
            ),
            forbidden=(
                "collision_authority",
                "force_authority",
                "unobserved_action_prediction",
            ),
            notes="C0 authority and the usual correct substrate for perception-only claims; not a simulator.",
        ),
    )


_PRIORITY_INVESTIGATIONS: tuple[dict[str, Any], ...] = (
    {
        "priority": 1,
        "candidate_ids": ["exact-geometry-stack"],
        "rationale": "many customer questions are pure reachability/geometry where full simulation adds uncertainty, not validity",
    },
    {
        "priority": 2,
        "candidate_ids": ["isaac-rtx-openusd-sensor-path", "isaac-sim-6-physx"],
        "rationale": "captured-site perception and sensor evaluation with physics, rendering, and reconstruction qualified separately",
    },
    {
        "priority": 3,
        "candidate_ids": ["mujoco-3"],
        "rationale": "rigid/contact baseline; qualify CPU, MJX, and MJWarp separately where numerics matter",
    },
    {
        "priority": 4,
        "candidate_ids": ["drake-1-55"],
        "rationale": "hydroelastic contact for pushing, insertion, compliant and articulated contact",
    },
    {
        "priority": 5,
        "candidate_ids": ["sapien-maniskill-3", "simpler-env"],
        "rationale": "articulated-object tasks plus the real/sim policy-ranking protocol template",
    },
    {
        "priority": 6,
        "candidate_ids": ["newton-1-4"],
        "rationale": "multi-backend GPU bridge; candidate-only while APIs and solvers evolve rapidly",
    },
    {
        "priority": 7,
        "candidate_ids": ["flash", "garmentdynamics-rgbench", "simweaver-sim1"],
        "rationale": "cloth lane run as a comparative qualification program, not interchangeable products",
    },
    {
        "priority": 8,
        "candidate_ids": ["dlo-lab", "pyelastica", "sofa-26-06"],
        "rationale": "cable/rope/hose lane with rod mechanics instead of cloth approximations",
    },
    {
        "priority": 9,
        "candidate_ids": ["project-chrono-10", "altair-edem-2026", "ansys-rocky-2026"],
        "rationale": "open multiphysics route plus a calibrated commercial granular reference solver",
    },
    {
        "priority": 10,
        "candidate_ids": ["tacsl", "difftactile", "direct-captured-observations"],
        "rationale": "tactile/force evidence with the real sensor as the qualification authority",
    },
)

_QUALIFICATION_BENCHMARK_BLUEPRINTS: tuple[dict[str, Any], ...] = (
    {
        "benchmark_id": "capture-to-geometry-and-contact",
        "purpose": "qualify reachability, collision, rigid pick/place, articulated objects, and insertion",
        "protocols": ["Q-KIN", "Q-RIGID", "Q-CONTACT", "Q-ART"],
        "physical_setup": [
            "metrically_surveyed_workcell",
            "multiple_collider_types_and_clearance_regimes",
            "rigid_objects_with_measured_mass_inertia",
            "multiple_surface_friction_pairs",
            "one_door_and_one_drawer_with_measured_joints",
            "instrumented_peg_insertion",
            "motion_capture_or_high_accuracy_pose_tracking",
            "wrist_force_torque",
        ],
        "methods_compared": [
            "exact-geometry-stack",
            "mujoco-3",
            "drake-1-55",
            "isaac-sim-6-physx",
            "newton-1-4",
            "sapien-maniskill-3",
        ],
        "metrics": [
            "collision_false_negatives_and_false_positives",
            "minimum_clearance_error",
            "final_object_pose_error",
            "contact_sequence",
            "force_impulse_error",
            "penetration",
            "drawer_door_force_travel_error",
            "insertion_success_boundary",
            "policy_ranking_regret",
        ],
        "role": "primary admission test for general rigid simulators",
    },
    {
        "benchmark_id": "capture-to-observation",
        "purpose": "qualify rendering and sensor simulation at captured sites",
        "protocols": ["Q-SENSOR"],
        "physical_setup": [
            "calibrated_kitchen_or_industrial_scene",
            "rgb_depth_lidar_optional_event_camera",
            "controlled_and_natural_lighting",
            "transparent_reflective_dark_small_thin_occluded_objects",
            "repeated_real_policy_perception_runs",
        ],
        "methods_compared": [
            "direct-captured-observations",
            "simple-rasterization",
            "isaac-rtx-openusd-sensor-path",
            "gaussian-splat-nurec-appearance",
        ],
        "metrics": [
            "calibrated_image_depth_lidar_residuals",
            "missing_depth_distribution",
            "temporal_error",
            "object_detection_pose_depth_performance",
            "policy_ranking",
            "top_policy_decision_regret",
            "failure_case_agreement",
        ],
        "role": "visual similarity is secondary to downstream task validity",
    },
    {
        "benchmark_id": "capture-to-deformation",
        "purpose": "qualify deformable lanes under one governance protocol",
        "protocols": ["Q-CLOTH", "Q-DLO", "Q-GRAN", "Q-TACT"],
        "physical_setup": [
            "shirts_towels_with_measured_stretch_bending_areal_density_friction",
            "unseen_garments_and_crumple_states",
            "ropes_hoses_with_measured_bend_twist_friction_clips_connectors",
            "two_characterized_granular_materials_pouring_and_tool_interaction",
            "calibrated_real_tactile_sequences_with_normal_shear_and_slip_labels",
        ],
        "lanes": {
            "cloth": [
                "flash",
                "garmentdynamics-rgbench",
                "simweaver-sim1",
                "newton-1-4",
                "mujoco-3",
                "ipc-family",
                "isaac-sim-6-physx",
            ],
            "cable": ["dlo-lab", "pyelastica", "sofa-26-06"],
            "granular": [
                "mujoco-3",
                "project-chrono-10",
                "altair-edem-2026",
                "ansys-rocky-2026",
                "ddbot-digging",
            ],
            "tactile": ["direct-captured-observations", "tacsl", "difftactile"],
        },
        "methods_compared": [
            "flash",
            "garmentdynamics-rgbench",
            "simweaver-sim1",
            "newton-1-4",
            "mujoco-3",
            "ipc-family",
            "dlo-lab",
            "pyelastica",
            "sofa-26-06",
            "project-chrono-10",
            "altair-edem-2026",
            "ansys-rocky-2026",
            "ddbot-digging",
            "direct-captured-observations",
            "tacsl",
            "difftactile",
        ],
        "metrics": [
            "state_trajectory",
            "topology_contact",
            "force",
            "task_outcome",
            "uncertainty",
            "real_sim_policy_ranking",
        ],
        "role": "separate lanes: a cloth result never qualifies a cable, powder, fluid, or food claim",
    },
    {
        "benchmark_id": "world-model-action-fidelity",
        "purpose": "evaluate action-conditioned world-model outputs for bounded evaluator support",
        "protocols": ["Q-WM"],
        "physical_setup": [
            "disjoint_action_conditioned_rollouts",
            "exact_action_timing_units_and_controller_state_bindings",
            "independent_forward_inverse_action_recovery_scorer",
            "held_out_policies_and_real_outcome_joins_for_any_ranking_claim",
        ],
        "methods_compared": [
            "gigaworld-wmbench",
            "oscar-world-model",
            "interactive-world-simulator",
            "roboworld",
            "cosmos-3",
        ],
        "metrics": [
            "action_recovery_max_abs_error",
            "forward_inverse_consistency",
            "action_motion_correlation",
            "policy_ranking_regret",
            "coverage",
            "task_outcome",
        ],
        "role": "evaluator support and comparative policy ranking only; never physics or physical success",
    },
)


def research_intake_catalog() -> tuple[dict[str, Any], ...]:
    """Every §2 landscape row as a validated research-candidate dossier."""

    return (
        *_general_candidates(),
        *_niche_candidates(),
        *_render_and_world_model_candidates(),
    )


def research_catalog_snapshot() -> dict[str, Any]:
    entries = research_intake_catalog()
    snapshot = {
        "schema_version": "measurement_method_research_catalog_snapshot.v1",
        "catalog_version": RESEARCH_CATALOG_VERSION,
        "entry_count": len(entries),
        "entry_digests": sorted(row["research_candidate_digest"] for row in entries),
        "production_route_count": 0,
        "public_research_is_qualification": False,
    }
    snapshot["catalog_snapshot_digest"] = _digest(snapshot, "catalog_snapshot_digest")
    return snapshot


def research_candidate_r0_stage_data(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Bridge a catalog dossier into R0 intake stage data for the admission
    state machine.  The output seeds ``create_research_candidate``; it grants
    no eligibility."""

    validated = validate_research_method_candidate(candidate)
    return {
        "primary_sources": [row["reference"] for row in validated["primary_sources"]],
        "method_identity": {
            "method_id": validated["method_id"],
            "method_family": validated["method_family"],
            "version_observed": validated["version_observed"],
        },
        "claimed_scope": {
            "task_classes": validated["task_classes"],
            "material_regimes": validated["material_regimes"],
            "required_qualification_protocols": validated["required_qualification_protocols"],
            "claim_ceiling_after_qualification": validated["claim_ceiling_after_qualification"],
        },
        "access_status": dict(validated["access"]),
    }


def standing_abstentions() -> dict[str, str]:
    return dict(STANDING_ABSTENTIONS)


def priority_investigations() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(json.dumps(row)) for row in _PRIORITY_INVESTIGATIONS)


def qualification_benchmark_blueprints() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(json.dumps(row)) for row in _QUALIFICATION_BENCHMARK_BLUEPRINTS)


__all__ = [
    "EVIDENCE_LABELS",
    "QUALIFICATION_PROTOCOLS",
    "RESEARCH_CANDIDATE_SCHEMA_VERSION",
    "RESEARCH_CATALOG_VERSION",
    "RESEARCH_CLASSIFICATIONS",
    "ResearchCatalogError",
    "STANDING_ABSTENTIONS",
    "TASK_CLASS_QUALIFICATION_PROTOCOLS",
    "priority_investigations",
    "qualification_benchmark_blueprints",
    "research_candidate_r0_stage_data",
    "research_catalog_snapshot",
    "research_intake_catalog",
    "standing_abstentions",
    "validate_research_method_candidate",
]
