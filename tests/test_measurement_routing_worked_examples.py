"""Worked routing examples from the 2026-08-01 measurement-routing research.

Each test pins one report scenario end to end: exact-geometry reachability,
appearance-only drawer abstention, transparent-object direct observation,
uncharacterized granular abstention, cloth-checkbox and cable-solver
guardrails, world-model ranking bounds, physical-test routing, mandatory
safety abstention, and the three proposal-only supervisor agent roles.
"""

from __future__ import annotations

from datetime import date

from blueprint_pipeline.task_evaluation_supervisor.capabilities import (
    DeterministicCaptureTestbedSupervisor,
    DeterministicClaimTaskInterpreter,
    DeterministicScenarioAdversarialProposer,
    SupervisorContext,
)
from blueprint_pipeline.task_site_measurement_routing import (
    ALL_CAPABILITY_FIELDS,
    derive_task_measurement_requirements,
    route_task_site_measurement,
    validate_measurement_qualification,
    validate_method_capability_profile,
    validate_site_evidence_profile,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64

_LIST_FIELDS = {
    "plugin_versions",
    "robot_model_formats",
    "supported_embodiments",
    "supported_end_effectors",
    "action_representation_types",
    "qualification_record_ids",
    "qualified_task_classes",
    "qualified_material_regimes",
    "qualified_robot_ids",
    "qualified_end_effector_ids",
    "qualified_controller_ids",
    "qualified_sensor_ids",
    "qualified_site_classes",
    "qualified_metric_ids",
    "known_failure_modes",
    "prohibited_extrapolations",
    "asset_license_ids",
    "model_license_ids",
    "subprocessor_regions",
    "output_formats",
}


def _profile(
    method_id: str,
    enabled: set[str],
    *,
    family: str = "traditional_simulation",
    ceiling: str = "C3",
    world_model_roles: tuple[str, ...] = (),
    cost: float = 1.0,
) -> dict:
    values: dict = {field: False for field in ALL_CAPABILITY_FIELDS}
    for field in _LIST_FIELDS:
        values[field] = []
    values.update(
        {
            "method_id": method_id,
            "method_family": family,
            "version": "1",
            "release_date": "2026-08-01",
            "commit_hash": "fixture-commit",
            "container_digest": SHA_A,
            "solver_backend": "fixture",
            "numeric_precision": "float64",
            "deterministic_mode": "strict",
            "operating_system": "linux",
            "gpu_model": "none",
            "driver_version": "none",
            "random_seed_policy": "frozen",
            "contact_formulation": "fixture",
            "maximum_control_rate_hz": 1000,
            "qualified_parameter_ranges": {},
            "qualified_claim_ceiling": ceiling,
            "qualification_expiration": "2027-08-01",
            "harmful_false_negative_bound": 0.01,
            "maximum_latency_class": "interactive",
            "maximum_compute_class": "cpu",
            "estimated_cost_class": "low",
            "data_retention_days": 0,
            "source_available": True,
            "local_offline_supported": True,
            "commercial_use_allowed": True,
            "provider_training_use_allowed": False,
            "deletion_right_supported": True,
            "output_export_supported": True,
        }
    )
    for field in enabled:
        values[field] = True
    profile_value: dict = {
        "schema_version": "method_capability_profile.v1",
        "method_id": method_id,
        "capabilities": values,
        "evidence_quality": {"source": "independent_fixture"},
        "expected_cost_usd": cost,
        "expected_latency_seconds": 1.0,
    }
    if world_model_roles:
        profile_value["world_model_roles"] = list(world_model_roles)
    return validate_method_capability_profile(profile_value)


def _qualification(
    profile: dict,
    capabilities: set[str],
    *,
    ceiling: str = "C3",
    task_classes: tuple[str, ...] = ("rigid_pick_place",),
    material_regimes: tuple[str, ...] = ("none",),
    accuracy: float = 0.01,
) -> dict:
    return validate_measurement_qualification(
        {
            "schema_version": "measurement_qualification_record.v1",
            "qualification_id": f"q-{profile['method_id']}",
            "method_id": profile["method_id"],
            "method_version": "1",
            "capability_profile_digest": profile["capability_profile_digest"],
            "admission_record_digest": SHA_A,
            "admission_stage": "R7",
            "status": "approved",
            "qualified_capabilities": sorted(capabilities),
            "claim_ceiling": ceiling,
            "scope": {
                "task_classes": list(task_classes),
                "material_regimes": list(material_regimes),
                "robot_ids": [],
                "end_effector_ids": [],
                "controller_ids": [],
                "sensor_ids": [],
                "metric_ids": [],
                "parameter_ranges": {},
            },
            "metrics": {
                "physical_accuracy_error": accuracy,
                "uncertainty": accuracy * 2,
                "scope_distance": 0.0,
                "harmful_false_negative_rate": 0.001,
                "reproducibility_score": 1.0,
                "privacy_preference": 1.0,
            },
            "approval": {
                "signature_status": "verified",
                "signature_id": f"fixture-signature-{profile['method_id']}",
                "approved_by": ["benchmark-owner", "independent-reviewer"],
                "agent_approved": False,
            },
            "expiration_date": "2027-08-01",
            "self_grading": False,
        }
    )


def _site(present: set[str], *, known_missing: tuple[str, ...] = ()) -> dict:
    return validate_site_evidence_profile(
        {
            "schema_version": "site_evidence_profile.v1",
            "profile_id": "worked-example-site",
            "bundle_id": "capture-1",
            "bundle_hash": SHA_A,
            "provenance_record_id": "provenance-fixture",
            "rights": {"commercial_evaluation_allowed": True},
            "privacy": {"external_processing_allowed": False},
            "coordinate_system": {"metric_scale_verified": True},
            "evidence": {
                evidence_id: {
                    "available": True,
                    "validated": True,
                    "record_id": f"record-{evidence_id}",
                }
                for evidence_id in present
            },
            "limitations": {
                "known_missing_regions": list(known_missing),
                "forbidden_claims": [],
            },
        }
    )


def _route(requirements, site, profiles, qualifications):
    return route_task_site_measurement(
        requirements,
        site,
        profiles,
        qualifications,
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 1),
    )


def test_example_1_reachability_routes_to_exact_geometry_over_full_dynamics() -> None:
    requirements = derive_task_measurement_requirements(
        {"claim_id": "reach-corners", "claim_type": "reachability"},
        {"task_distribution": {"measurement_task_class": "static_reachability"}},
    )
    caps = {"metric_scale_supported", "self_collision_supported", "joint_limits_supported"}
    exact = _profile(
        "exact-geometry", caps, family="analytic_geometry_kinematics", ceiling="C2", cost=5.0
    )
    dynamics = _profile("full-dynamics-engine", caps, cost=0.1)
    decision = _route(
        requirements,
        _site({"metric_scale", "robot_site_registration", "articulation_model"}),
        [dynamics, exact],
        [
            _qualification(
                exact, caps, ceiling="C2", task_classes=("static_reachability",), accuracy=0.001
            ),
            _qualification(
                dynamics, caps, ceiling="C3", task_classes=("static_reachability",), accuracy=0.02
            ),
        ],
    )
    assert decision["status"] == "route_selected"
    assert decision["selected_route"]["type"] == "single_method"
    # Held-out accuracy dominates price: the cheap dynamics engine loses.
    assert decision["selected_route"]["stages"][0]["method_id"] == "exact-geometry"
    assert decision["claim_boundary"]["permitted_claim"] == "C1"
    assert decision["route_claim_ceiling"] == "C2"


def test_example_3_drawer_with_appearance_only_evidence_abstains_to_articulation_measurement() -> (
    None
):
    requirements = derive_task_measurement_requirements(
        {"claim_id": "open-drawer", "claim_type": "collision_contact"},
        {"task_distribution": {"measurement_task_class": "doors_drawers_handles"}},
    )
    caps = {
        "metric_scale_supported",
        "revolute_joint_supported",
        "prismatic_joint_supported",
        "joint_limits_supported",
        "joint_friction_supported",
        "joint_damping_supported",
        "detent_supported",
        "contact_force_output_supported",
        "continuous_collision_supported",
    }
    simulator = _profile("articulated-sim", caps)
    decision = _route(
        requirements,
        _site(
            {
                "metric_scale",
                "robot_site_registration",
                "validated_collider",
                "validated_mesh",
                "gaussian_splat_appearance",
            }
        ),
        [simulator],
        [_qualification(simulator, caps, task_classes=("doors_drawers_handles",))],
    )
    assert decision["status"] == "abstention"
    action = decision["abstention"]["smallest_next_action"]
    assert action["action_type"] == "articulation_measurement"
    assert "articulation_model" in action["exact_scope"]
    # The splat and mesh in evidence never stood in for the joint model.
    assert any(
        "required_site_evidence_missing:articulation_model" in code
        for code in decision["abstention"]["blocking_requirements"]
    )


def test_example_9_transparent_objects_prefer_direct_captured_observation() -> None:
    requirements = derive_task_measurement_requirements(
        {"claim_id": "glassware", "claim_type": "perception_visibility"},
        {"task_distribution": {"measurement_task_class": "transparent_reflective_objects"}},
    )
    caps = set(requirements["required_capabilities"])
    capture = _profile("direct-capture", caps, family="captured_real_observation", ceiling="C4")
    renderer = _profile(
        "rtx-renderer", caps, family="calibrated_renderer_sensor_simulation", ceiling="C4"
    )
    site = _site({"calibrated_rgb", "sensor_calibration", "sensor_timing", "lighting_coverage"})
    decision = _route(
        requirements,
        site,
        [renderer, capture],
        [
            _qualification(
                capture,
                caps,
                ceiling="C2",
                task_classes=("transparent_reflective_objects",),
                accuracy=0.005,
            ),
            _qualification(
                renderer,
                caps,
                ceiling="C2",
                task_classes=("transparent_reflective_objects",),
                accuracy=0.2,
            ),
        ],
    )
    assert decision["status"] == "route_selected"
    assert decision["selected_route"]["type"] == "direct_observation"
    assert decision["selected_route"]["stages"][0]["method_id"] == "direct-capture"


def test_example_11_uncharacterized_granular_material_abstains_to_identification() -> None:
    requirements = derive_task_measurement_requirements(
        {
            "claim_id": "pellet-pouring",
            "claim_type": "collision_contact",
            "material_regimes": ["granular_media"],
        },
        {"task_distribution": {"measurement_task_class": "granular_manipulation"}},
    )
    caps = {
        "dem_supported",
        "granular_cohesion_supported",
        "rolling_friction_supported",
        "metric_scale_supported",
        "continuous_collision_supported",
    }
    dem = _profile("dem-solver", caps)
    decision = _route(
        requirements,
        _site({"metric_scale", "robot_site_registration", "validated_collider"}),
        [dem],
        [
            _qualification(
                dem,
                caps,
                task_classes=("granular_manipulation",),
                material_regimes=("granular_media",),
            )
        ],
    )
    assert decision["status"] == "abstention"
    assert decision["abstention"]["smallest_next_action"]["action_type"] == (
        "material_identification"
    )
    audit = decision["abstention"]["site_evidence_audit"]
    assert any(gap["evidence_id"] == "material_parameters" for gap in audit["gaps"])


def test_examples_6_and_7_cloth_checkbox_never_becomes_towel_or_garment_authority() -> None:
    requirements = derive_task_measurement_requirements(
        {
            "claim_id": "fold-shirts",
            "claim_type": "collision_contact",
            "material_regimes": ["garment_cloth"],
        },
        {"task_distribution": {"measurement_task_class": "garment_manipulation"}},
    )
    caps = set(requirements["required_capabilities"])
    cloth_checkbox = _profile("generic-cloth-engine", caps)
    decision = _route(
        requirements,
        _site(
            {
                "metric_scale",
                "robot_site_registration",
                "validated_collider",
                "material_parameters",
                "initial_material_state",
                "friction_contact",
            }
        ),
        [cloth_checkbox],
        [
            # Rigid-scope qualification: a cloth feature flag plus a rigid-task
            # record is exactly the "supports cloth" fallacy.
            _qualification(cloth_checkbox, caps, task_classes=("rigid_pick_place",))
        ],
    )
    assert decision["status"] == "abstention"
    codes = decision["candidates_considered"][0]["rejection_codes"]
    assert "qualification_scope_mismatch:task_classes" in codes
    assert "qualification_scope_mismatch:material_regimes" in codes
    assert decision["abstention"]["smallest_next_action"]["action_type"] == (
        "qualification_benchmark"
    )


def test_example_8_cable_routing_requires_rod_mechanics_not_a_cloth_solver() -> None:
    requirements = derive_task_measurement_requirements(
        {
            "claim_id": "route-hose",
            "claim_type": "collision_contact",
            "material_regimes": ["rope_cable_hose"],
        },
        {"task_distribution": {"measurement_task_class": "cable_hose_routing"}},
    )
    cloth_only = _profile(
        "cloth-solver",
        {"cloth_shell_supported", "anisotropic_cloth_supported", "seams_supported"},
    )
    rod_caps = set(requirements["required_capabilities"])
    rod = _profile("dlo-rod-solver", rod_caps)
    decision = _route(
        requirements,
        _site(
            {
                "metric_scale",
                "robot_site_registration",
                "validated_collider",
                "material_parameters",
                "initial_material_state",
                "friction_contact",
            }
        ),
        [cloth_only, rod],
        [
            _qualification(
                cloth_only,
                {"cloth_shell_supported"},
                task_classes=("garment_manipulation",),
                material_regimes=("garment_cloth",),
            ),
            _qualification(
                rod,
                rod_caps,
                task_classes=("cable_hose_routing",),
                material_regimes=("rope_cable_hose",),
            ),
        ],
    )
    assert decision["status"] == "route_selected"
    assert decision["selected_route"]["stages"][0]["method_id"] == "dlo-rod-solver"
    cloth_row = next(
        row for row in decision["candidates_considered"] if row["method_id"] == "cloth-solver"
    )
    assert cloth_row["eligible"] is False


def test_example_13_world_model_may_rank_policies_but_never_measure_physics() -> None:
    requirements = derive_task_measurement_requirements(
        {"claim_id": "rank-rgb-policies", "claim_type": "comparative_policy_ranking"},
        {"task_distribution": {"measurement_task_class": "visual_perception"}},
    )
    caps = set(requirements["required_capabilities"])
    world_model = _profile(
        "action-video-world-model",
        caps,
        family="learned_world_model",
        ceiling="C4",
        world_model_roles=("comparative_policy_ranking", "evaluator_support"),
    )
    decision = _route(
        requirements,
        _site(
            {
                "calibrated_rgb",
                "sensor_calibration",
                "sensor_timing",
                "robot_controller_calibration",
            }
        ),
        [world_model],
        [
            _qualification(
                world_model,
                caps,
                ceiling="C4",
                task_classes=("visual_perception",),
            )
        ],
    )
    assert decision["status"] == "route_selected"
    assert decision["claim_boundary"]["permitted_claim"] == "C4"
    boundary = decision["claim_boundary"]
    assert boundary["physical_success_established"] is False
    assert boundary["deployment_readiness_established"] is False
    assert boundary["safety_certification_established"] is False
    assert set(boundary["prohibited_claims"]) == {"C5", "C6", "C7", "C8"}


def test_physical_task_success_routes_only_through_physical_evidence() -> None:
    requirements = derive_task_measurement_requirements(
        {"claim_id": "really-pick", "claim_type": "physical_task_success"},
        {"task_distribution": {"measurement_task_class": "rigid_pick_place"}},
    )
    caps = set(requirements["required_capabilities"])
    simulator = _profile("rigid-sim", caps)
    physical = _profile("instrumented-trials", caps, family="physical_evidence", ceiling="C8")
    site = _site(
        {
            "metric_scale",
            "robot_site_registration",
            "validated_collider",
            "mass_inertia",
            "friction_contact",
            "material_parameters",
        }
    )
    decision = _route(
        requirements,
        site,
        [simulator, physical],
        [
            _qualification(simulator, caps, ceiling="C4"),
            _qualification(physical, caps, ceiling="C6"),
        ],
    )
    assert decision["status"] == "route_selected"
    assert decision["selected_route"]["type"] == "physical_test"
    assert decision["selected_route"]["stages"][0]["method_id"] == "instrumented-trials"
    # Planning a physical test never establishes the physical claim.
    assert decision["claim_boundary"]["physical_success_established"] is False
    simulator_row = next(
        row for row in decision["candidates_considered"] if row["method_id"] == "rigid-sim"
    )
    assert "requested_claim_exceeds_qualification_ceiling" in simulator_row["rejection_codes"]


def test_example_16_safety_certification_is_a_mandatory_abstention_from_simulation() -> None:
    requirements = derive_task_measurement_requirements(
        {"claim_id": "workcell-safety", "claim_type": "safety_certification"},
        {"task_distribution": {"measurement_task_class": "static_reachability"}},
    )
    caps = set(requirements["required_capabilities"])
    simulator = _profile("workcell-sim", caps)
    over_claiming = _qualification(simulator, caps, ceiling="C8")
    decision = _route(
        requirements,
        _site({"metric_scale", "robot_site_registration", "articulation_model"}),
        [simulator],
        [over_claiming],
    )
    assert decision["status"] == "abstention"
    assert decision["abstention"]["smallest_next_action"]["action_type"] == ("physical_execution")
    codes = decision["candidates_considered"][0]["rejection_codes"]
    # Even a signed C8 record cannot lift a simulation family past its cap.
    assert "claim_ceiling_exceeds_method_family_cap" in codes
    assert decision["claim_boundary"]["safety_certification_established"] is False


def test_example_2_controller_ranking_routes_only_through_the_ranked_qualified_engine() -> None:
    requirements = derive_task_measurement_requirements(
        {"claim_id": "rank-controllers", "claim_type": "comparative_policy_ranking"},
        {"task_distribution": {"measurement_task_class": "rigid_pick_place"}},
    )
    caps = set(requirements["required_capabilities"])
    ranked = _profile("ranked-rigid-engine", caps, ceiling="C4")
    unranked = _profile("other-rigid-engine", caps, ceiling="C4")
    site = _site(
        {
            "metric_scale",
            "robot_site_registration",
            "validated_collider",
            "mass_inertia",
            "friction_contact",
            "material_parameters",
            "sensor_timing",
            "robot_controller_calibration",
        }
    )
    decision = _route(
        requirements,
        site,
        [ranked, unranked],
        # Only one engine holds paired real/sim ranking qualification.
        [_qualification(ranked, caps, ceiling="C4")],
    )
    assert decision["status"] == "route_selected"
    assert decision["selected_route"]["stages"][0]["method_id"] == "ranked-rigid-engine"
    assert decision["claim_boundary"]["permitted_claim"] == "C4"
    other = next(
        row for row in decision["candidates_considered"] if row["method_id"] == "other-rigid-engine"
    )
    assert "no_exact_verified_qualification" in other["rejection_codes"]


def test_example_4_tight_tolerance_insertion_abstains_on_nominal_cad_alone() -> None:
    requirements = derive_task_measurement_requirements(
        {"claim_id": "peg-socket", "claim_type": "collision_contact"},
        {"task_distribution": {"measurement_task_class": "insertion_assembly"}},
    )
    caps = set(requirements["required_capabilities"])
    simulator = _profile("contact-sim", caps)
    decision = _route(
        requirements,
        # Nominal CAD gives geometry, but no measured compliance, force
        # calibration, or controller identification.
        _site({"metric_scale", "robot_site_registration", "validated_collider"}),
        [simulator],
        [_qualification(simulator, caps, task_classes=("insertion_assembly",))],
    )
    assert decision["status"] == "abstention"
    action = decision["abstention"]["smallest_next_action"]
    assert action["action_type"] == "force_tactile_collection"
    assert {"force_tactile", "material_parameters"} <= set(action["exact_scope"])


def test_example_5_photoreal_kitchen_perception_ranking_is_a_composite_route() -> None:
    requirements = derive_task_measurement_requirements(
        {"claim_id": "kitchen-rgb-ranking", "claim_type": "comparative_policy_ranking"},
        {"task_distribution": {"measurement_task_class": "visual_perception"}},
    )
    appearance_caps = {
        "rgb_supported",
        "intrinsics_import_supported",
        "extrinsics_import_supported",
        "distortion_model_supported",
    }
    timing_caps = {
        "sensor_timing_supported",
        "controller_latency_model_supported",
    }
    capture = _profile(
        "calibrated-capture",
        appearance_caps,
        family="captured_real_observation",
        ceiling="C4",
    )
    renderer = _profile(
        "calibrated-rtx-path",
        appearance_caps | timing_caps,
        family="calibrated_renderer_sensor_simulation",
        ceiling="C4",
    )
    site = _site(
        {
            "calibrated_rgb",
            "sensor_calibration",
            "sensor_timing",
            "robot_controller_calibration",
        }
    )
    decision = _route(
        requirements,
        site,
        [capture, renderer],
        [
            _qualification(
                capture,
                appearance_caps,
                ceiling="C4",
                task_classes=("visual_perception",),
                accuracy=0.005,
            ),
            _qualification(
                renderer,
                timing_caps,
                ceiling="C4",
                task_classes=("visual_perception",),
                accuracy=0.05,
            ),
        ],
    )
    assert decision["status"] == "route_selected"
    assert decision["selected_route"]["type"] == "composite"
    assert {row["method_id"] for row in decision["selected_route"]["stages"]} == {
        "calibrated-capture",
        "calibrated-rtx-path",
    }
    assert decision["claim_boundary"]["permitted_claim"] == "C4"


def test_example_10_partially_captured_warehouse_demands_targeted_recapture() -> None:
    requirements = derive_task_measurement_requirements(
        {"claim_id": "warehouse-navigation", "claim_type": "perception_visibility"},
        {"task_distribution": {"measurement_task_class": "visual_navigation_active_perception"}},
    )
    caps = set(requirements["required_capabilities"])
    navigator = _profile("navigation-stack", caps, ceiling="C2")
    decision = _route(
        requirements,
        _site(
            {
                "metric_scale",
                "robot_site_registration",
                "calibrated_rgb",
                "calibrated_depth",
                "sensor_calibration",
                "sensor_timing",
            },
            known_missing=("aisle-7", "aisle-9"),
        ),
        [navigator],
        [
            _qualification(
                navigator,
                caps,
                ceiling="C2",
                task_classes=("visual_navigation_active_perception",),
            )
        ],
    )
    assert decision["status"] == "abstention"
    action = decision["abstention"]["smallest_next_action"]
    assert action["action_type"] == "targeted_recapture"
    assert "coverage_uncertainty" in action["exact_scope"]
    audit = decision["abstention"]["site_evidence_audit"]
    assert audit["known_missing_regions"] == ["aisle-7", "aisle-9"]


def test_example_12_food_cutting_abstains_to_specimen_identification() -> None:
    requirements = derive_task_measurement_requirements(
        {
            "claim_id": "cut-food",
            "claim_type": "collision_contact",
            "material_regimes": ["food_cuttable_multiphase"],
        },
        {"task_distribution": {"measurement_task_class": "food_manipulation"}},
    )
    caps = set(requirements["required_capabilities"])
    cutter = _profile("food-cutting-sim", caps)
    decision = _route(
        requirements,
        _site({"metric_scale", "robot_site_registration", "validated_collider"}),
        [cutter],
        [
            _qualification(
                cutter,
                caps,
                task_classes=("food_manipulation",),
                material_regimes=("food_cuttable_multiphase",),
            )
        ],
    )
    assert decision["status"] == "abstention"
    action = decision["abstention"]["smallest_next_action"]
    assert action["action_type"] == "material_identification"
    assert "physical_specimens" in action["exact_scope"]
    assert decision["claim_boundary"]["physical_success_established"] is False


def test_example_14_locomotion_slip_ranking_needs_friction_but_geometry_still_plans() -> None:
    slip_requirements = derive_task_measurement_requirements(
        {"claim_id": "floor-slip-ranking", "claim_type": "collision_contact"},
        {"task_distribution": {"measurement_task_class": "locomotion"}},
    )
    slip_caps = set(slip_requirements["required_capabilities"])
    locomotion_engine = _profile("locomotion-engine", slip_caps)
    site = _site(
        {
            "metric_scale",
            "robot_site_registration",
            "validated_collider",
            "mass_inertia",
            "calibrated_imu",
            "robot_controller_calibration",
            "articulation_model",
        }
    )
    slip_decision = _route(
        slip_requirements,
        site,
        [locomotion_engine],
        [_qualification(locomotion_engine, slip_caps, task_classes=("locomotion",))],
    )
    assert slip_decision["status"] == "abstention"
    assert slip_decision["abstention"]["smallest_next_action"]["action_type"] == (
        "material_identification"
    )

    reach_requirements = derive_task_measurement_requirements(
        {"claim_id": "route-planning", "claim_type": "reachability"},
        {"task_distribution": {"measurement_task_class": "static_reachability"}},
    )
    reach_caps = set(reach_requirements["required_capabilities"])
    geometry = _profile(
        "exact-geometry",
        reach_caps,
        family="analytic_geometry_kinematics",
        ceiling="C2",
    )
    reach_decision = _route(
        reach_requirements,
        site,
        [geometry],
        [
            _qualification(
                geometry,
                reach_caps,
                ceiling="C2",
                task_classes=("static_reachability",),
            )
        ],
    )
    assert reach_decision["status"] == "route_selected"
    assert reach_decision["claim_boundary"]["permitted_claim"] == "C1"


def test_example_15_long_horizon_composite_ceiling_is_the_minimum_stage_ceiling() -> None:
    requirements = derive_task_measurement_requirements(
        {"claim_id": "kitchen-long-horizon", "claim_type": "collision_contact"},
        {"task_distribution": {"measurement_task_class": "long_horizon_task_execution"}},
    )
    assert "real_demonstrations" in requirements["required_site_evidence"]
    geometry_caps = {
        "metric_scale_supported",
        "continuous_collision_supported",
        "self_collision_supported",
    }
    contact_caps = {"dynamic_collision_supported", "static_friction_supported"}
    geometry = _profile("navigation-geometry", geometry_caps, ceiling="C4")
    contact = _profile("manipulation-contact", contact_caps)
    evidence = {
        "metric_scale",
        "robot_site_registration",
        "validated_collider",
        "mass_inertia",
        "friction_contact",
    }
    qualifications = [
        _qualification(
            geometry,
            geometry_caps,
            ceiling="C4",
            task_classes=("long_horizon_task_execution",),
        ),
        _qualification(
            contact,
            contact_caps,
            ceiling="C3",
            task_classes=("long_horizon_task_execution",),
        ),
    ]

    # Without accepted real demonstrations the task-level evidence gate blocks
    # the route even though every solver capability is covered.
    missing = _route(requirements, _site(evidence), [geometry, contact], qualifications)
    assert missing["status"] == "abstention"
    assert missing["abstention"]["smallest_next_action"]["action_type"] == ("physical_execution")
    assert any(
        "real_demonstrations" in code for code in missing["abstention"]["blocking_requirements"]
    )

    complete = _route(
        requirements,
        _site(evidence | {"real_demonstrations"}),
        [geometry, contact],
        qualifications,
    )
    assert complete["status"] == "route_selected"
    assert complete["selected_route"]["type"] == "composite"
    # The composite ceiling is the lowest ceiling of its critical components.
    assert complete["route_claim_ceiling"] == "C3"


def _supervisor_fixtures() -> tuple[dict, dict, dict]:
    from blueprint_pipeline.decision_evidence_contracts import (
        DecisionEvidenceRequest,
        MaintainedSiteTaskTestbed,
    )

    site_profile = _site(
        {
            "metric_scale",
            "robot_site_registration",
            "validated_collider",
            "validated_mesh",
            "articulation_model",
            "material_parameters",
            "sensor_calibration",
            "mass_inertia",
            "friction_contact",
        }
    )
    testbed = MaintainedSiteTaskTestbed.from_mapping(
        {
            "schema_version": "maintained_site_task_testbed.v1",
            "testbed_id": "worked-example-testbed",
            "version": "1",
            "predecessor_testbed_digest": None,
            "supersedes": [],
            "source_capture_bundles": [{"bundle_id": "capture-1", "version": "1", "digest": SHA_A}],
            "artifact_references": {
                key: {"uri": f"fixture://{key}", "digest": SHA_B}
                for key in (
                    "site_card",
                    "task_cards",
                    "scenario_cards",
                    "eval_cards",
                    "evaluator",
                    "reset",
                )
            },
            "task_distribution": {"task_family": "rigid_object_pick_place"},
            "supported_condition_ranges": {"lighting_lux": [300, 600]},
            "robot_sensor_controller_bindings": {
                "embodiment": {"robot_id": "fixture-arm"},
                "sensors": {"camera": "fixture-rgb-v1"},
                "controller_action_representation": {"type": "joint_position"},
            },
            "governance": {
                "rights": "accepted",
                "consent": "accepted",
                "privacy": "cleared",
                "revocation": "version_invalidates_on_revocation",
                "allowed_uses": ["evaluation"],
            },
            "evidence_inventory": [
                {"evidence_id": "metric_geometry"},
                {"evidence_id": "captured_rgb_frames"},
            ],
            "validation_envelope": {"site_id": "worked-example-site"},
            "known_unsupported_conditions": [],
            "invalidation_triggers": [],
            "physical_outcome_history_refs": [],
            "lifecycle_state": "active",
            "site_evidence_profile": site_profile,
        }
    ).to_mapping()

    def claim(claim_id: str, claim_type: str, **extra) -> dict:
        return {
            "claim_id": claim_id,
            "claim_type": claim_type,
            "subject": f"fixture:{claim_id}",
            "measurable_threshold": {"operator": ">=", "value": 0.8, "units": "ratio"},
            "false_safe_consequence": "moderate",
            "acceptable_false_safe_risk": 0.05,
            "desired_confidence_or_coverage": {"minimum_coverage": 0.5},
            "permitted_abstention_behavior": {"allowed": True},
            **extra,
        }

    request = DecisionEvidenceRequest.from_mapping(
        {
            "schema_version": "decision_evidence_request.v1",
            "request_id": "worked-example-request",
            "decision_id": "worked-example-decision",
            "testbed_id": testbed["testbed_id"],
            "testbed_version": testbed["version"],
            "testbed_digest": testbed["testbed_digest"],
            "decision_question": "Can the arm pick the part, and can we rank policies?",
            "candidates": [],
            "claims": [
                claim("collision", "collision_contact"),
                claim("fold", "collision_contact", material_regimes=["deformable"]),
            ],
            "budget": {"max_cost_usd": 20.0},
            "deadline": "2026-08-30T00:00:00Z",
            "available_physical_evidence": [],
            "permitted_evidence_methods": ["traditional_simulation"],
            "restrictions": {"external_processing_allowed": False},
            "requested_result_audience": "robot_team_buyer",
            "provenance": {"caller_identity": "fixture-test"},
            "idempotency_key": "worked-example-request",
        }
    ).to_mapping()

    measurement_profile = _profile(
        "fixture-rigid-engine",
        {
            "metric_scale_supported",
            "continuous_collision_supported",
            "dynamic_collision_supported",
            "static_friction_supported",
            "dynamic_friction_supported",
            "contact_compliance_supported",
        },
    )
    method_profile = {"measurement_capability_profile": measurement_profile}
    return request, testbed, method_profile


def test_interpreter_proposes_requirements_and_flags_generic_deformable() -> None:
    request, testbed, _ = _supervisor_fixtures()
    result = (
        DeterministicClaimTaskInterpreter()
        .propose(
            SupervisorContext(
                run_id="run-interpreter",
                customer_question="Fold the towels and pick the part",
                decision_request=request,
                testbed=testbed,
            )
        )
        .to_mapping()
    )
    artifact = result["artifact"]
    assert artifact["measurement_interpretation_authoritative"] is False
    assert artifact["measurement_requirements_proposed"] is False
    claims = {row["claim_id"]: row for row in artifact["claims"]}
    proposed = claims["collision"]["proposed_task_measurement_requirements"]
    assert proposed["task_class"] == "rigid_pick_place"
    assert proposed["agent_interpretation_authoritative"] is False
    assert any(
        "material_regime_forbidden_generic:deformable" in code
        for code in claims["fold"]["measurement_interpretation_blockers"]
    )
    assert any(
        row["action_type"] == "request_measurement_scope_clarification"
        for row in result["proposals"]
    )
    assert result["authoritative"] is False


def test_capture_supervisor_surfaces_site_evidence_audit_gaps() -> None:
    request, testbed, _ = _supervisor_fixtures()
    result = (
        DeterministicCaptureTestbedSupervisor()
        .propose(
            SupervisorContext(
                run_id="run-capture",
                customer_question="Audit the kitchen capture",
                decision_request=request,
                testbed=testbed,
            )
        )
        .to_mapping()
    )
    artifact = result["artifact"]
    audit = artifact["site_evidence_audit"]
    assert artifact["appearance_evidence_is_not_physical_evidence"] is True
    assert audit["raw_capture_truth_rewritten"] is False
    assert artifact["site_evidence_gap_count"] == audit["gap_count"]
    gap_proposals = [
        row
        for row in result["proposals"]
        if row["action_type"] == "request_targeted_site_measurement"
    ]
    if audit["gaps"]:
        assert gap_proposals
        assert gap_proposals[0]["parameters"]["gaps"] == audit["gaps"]


def test_scenario_proposer_drafts_qualification_benchmarks_it_cannot_approve() -> None:
    request, testbed, method_profile = _supervisor_fixtures()
    result = (
        DeterministicScenarioAdversarialProposer()
        .propose(
            SupervisorContext(
                run_id="run-benchmark-design",
                customer_question="Qualify a rigid engine for this site",
                decision_request=request,
                testbed=testbed,
                method_profiles=[method_profile],
            )
        )
        .to_mapping()
    )
    artifact = result["artifact"]
    drafts = artifact["qualification_benchmark_drafts"]
    assert artifact["qualification_design_authoritative"] is False
    assert artifact["vendor_self_grading_prohibited"] is True
    assert drafts
    for draft in drafts:
        assert draft["agent_may_approve"] is False
        assert draft["heldout_labels_exposed"] is False
        assert draft["qualification_protocols"] == ["Q-RIGID"]
        assert "capture-to-geometry-and-contact" in draft["matching_benchmark_blueprints"]
        assert "development_split_hash" in draft["frozen_fields_required"]
    assert any(
        row["action_type"] == "draft_qualification_benchmark_preregistration"
        for row in result["proposals"]
    )
