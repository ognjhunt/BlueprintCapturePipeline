from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_franka_evaluation_harness import (
    Adp009dHarnessError,
    _canonical_asset_binding,
    _check_instance_constraints,
    admit_task_construction,
    admit_cousin_static_validation,
    materialize_cousin_package,
    validate_task_construction_admission,
)
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.evaluation_run_contract import (
    default_evaluation_run_adapter_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = REPO_ROOT / "docs/arm_decision_proof_v1/manifests"
FEATURE_VERSIONS = {
    "FET000_CORE": "0.1.0",
    "FET001_BASE_NEUTRAL": "1.0.0",
    "FET003_BASE_NEUTRAL": "0.1.0",
    "FET003_BASE_PHYSX": "0.1.0",
    "FET004_BASE_NEUTRAL": "0.1.0",
    "FET004_BASE_PHYSX": "0.1.0",
    "FET005_BASE_NEUTRAL": "0.1.0",
    "FET006_BASE_MDL": "0.1.0",
}


def test_reusable_harness_resolves_through_existing_evaluation_run_seams() -> None:
    registry = default_evaluation_run_adapter_registry()
    expected = {
        "scene_bundle": "sealed_aura_sage_simready_scene",
        "robot_adapter": "isaac_lab_franka_robotiq",
        "task_scenario_pack": "adp009d_deterministic_scenario_pack",
        "policy_adapter": "adp009d_frozen_droid_policy",
        "runtime_provider_profile": "adp009d_isaac_lab_arena_runtime",
        "proof_contract": "adp009d_simulator_state_proof",
    }

    for component, adapter_id in expected.items():
        descriptor = registry.resolve(
            component=component,
            adapter_id=adapter_id,
            adapter_version="1",
        )
        assert descriptor is not None


def _load(name: str) -> dict:
    return json.loads((MANIFEST_ROOT / name).read_text(encoding="utf-8"))


def _factor(
    parameter_id: str,
    *,
    nominal: float,
    minimum: float,
    maximum: float,
    unit: str,
) -> dict:
    return {
        "parameter_id": parameter_id,
        "semantic_meaning": f"bounded test factor for {parameter_id}",
        "unit": unit,
        "nominal_value": nominal,
        "allowed": {"minimum": minimum, "maximum": maximum},
        "sampling": {"kind": "uniform", "decimals": 8},
        "source": "preregistered unit-test fixture",
        "reason": "exercise deterministic materialization",
        "runtime_target": f"EventManager.reset.{parameter_id}",
        "affects": [parameter_id],
        "validity": {
            "invalid_behavior": "reject_instance_fail_closed",
            "native_probe_required": True,
        },
    }


def _valid_suite(harness: dict) -> dict:
    seeds = list(range(16))
    factors = [
        _factor(
            "object_start_x_m",
            nominal=3.4681748,
            minimum=3.46,
            maximum=3.48,
            unit="m",
        ),
        _factor(
            "light_intensity_scale",
            nominal=1.0,
            minimum=0.8,
            maximum=1.2,
            unit="ratio",
        ),
        _factor(
            "external_camera_extrinsic_dx_m",
            nominal=0.0,
            minimum=-0.005,
            maximum=0.005,
            unit="m",
        ),
        _factor(
            "object_mass_kg",
            nominal=0.355,
            minimum=0.32,
            maximum=0.39,
            unit="kg",
        ),
    ]
    templates = [
        {
            "template_id": "canonical_anchor",
            "family": "canonical",
            "partition": "qualification",
            "scored": True,
            "factor_ids": [],
            "seeds": seeds,
            "cousin_id": "approved_can",
        },
        {
            "template_id": "placement_x",
            "family": "placement_approach",
            "partition": "qualification",
            "scored": True,
            "factor_ids": ["object_start_x_m"],
            "seeds": seeds,
            "cousin_id": "approved_can",
        },
        {
            "template_id": "illumination_intensity",
            "family": "illumination",
            "partition": "qualification",
            "scored": True,
            "factor_ids": ["light_intensity_scale"],
            "seeds": seeds,
            "cousin_id": "approved_can",
        },
        {
            "template_id": "camera_dx",
            "family": "camera_sensor",
            "partition": "qualification",
            "scored": True,
            "factor_ids": ["external_camera_extrinsic_dx_m"],
            "seeds": seeds,
            "cousin_id": "approved_can",
        },
        {
            "template_id": "physics_mass",
            "family": "physics",
            "partition": "qualification",
            "scored": True,
            "factor_ids": ["object_mass_kg"],
            "seeds": seeds,
            "cousin_id": "approved_can",
        },
        {
            "template_id": "visual_material_cousin",
            "family": "visual_material_cousin",
            "partition": "qualification",
            "scored": True,
            "factor_ids": [],
            "seeds": seeds,
            "cousin_id": "adp009d_visual_material_cousin",
        },
        {
            "template_id": "geometric_cousin",
            "family": "geometric_cousin",
            "partition": "qualification",
            "scored": True,
            "factor_ids": [],
            "seeds": seeds,
            "cousin_id": "adp009d_geometric_cousin",
        },
        {
            "template_id": "held_out_light_mass",
            "family": "held_out_composed",
            "partition": "held_out",
            "scored": True,
            "factor_ids": ["light_intensity_scale", "object_mass_kg"],
            "seeds": seeds,
            "cousin_id": "approved_can",
        },
    ]
    suite = {
        "schema_version": "adp009d_scenario_suite.v1",
        "program_id": "arm-decision-proof-v1",
        "freeze_status": "frozen_after_canonical_canary_before_scenario_evaluation",
        "harness_digest": harness["harness_digest"],
        "prior_canary_disclosure": {
            "scope": "canonical_smoke_canaries_only",
            "retained_receipt_digests": [
                "sha256:fc35d64d3ba255bfb086d74ea8cab327e11664b6d0bfeb6fd468bee686c0b253",
                "sha256:ddac03ff73648f61fee28bc0093f5d4a02a83a9f51916ba50a1265d388a8a7f9",
                "sha256:15385d341dbedf49f75e1b2bd52e52290b1ad841f93ebf9723b2aea14b8e24fc",
            ],
            "prior_outcomes_used_to_select_parameters": False,
            "scenario_family_results_observed": False,
            "next_learned_run_requires_frozen_suite_digest": True,
            "claim_ceiling": (
                "post_canary_preregistered_scenario_evaluation_not_prospective_"
                "from_first_learned_contact"
            ),
        },
        "required_controls": [
            "zero_action_negative",
            "deterministic_scripted_positive",
        ],
        "cartesian_product_allowed": False,
        "factors": factors,
        "cell_templates": templates,
        "invalid_combinations": [
            {
                "constraint_id": "placement_and_camera_not_composed",
                "when_all_non_nominal": [
                    "object_start_x_m",
                    "external_camera_extrinsic_dx_m",
                ],
                "behavior": "reject_instance_fail_closed",
            }
        ],
        "power_cost_analysis": {
            "two_sided_alpha": 0.05,
            "target_power": 0.8,
            "minimum_paired_difference": 0.2,
            "anticipated_discordance": 0.3,
            "computed_required_paired_cells": 57,
            "planned_paired_cells": 128,
            "canonical_worst_case_wilson_half_width": 0.22000436389673977,
            "canonical_half_width_max": 0.221,
            "per_family_half_width_max": 0.221,
            "estimated_episode_wall_seconds": 60.0,
            "planned_episode_count": 512,
            "estimated_total_gpu_hours": 8.533333333333333,
            "maximum_total_gpu_hours": 9.0,
            "analysis_frozen_before_learned_outcomes": False,
            "analysis_frozen_before_scenario_evaluation_outcomes": True,
        },
        "suite_digest": "",
    }
    suite["suite_digest"] = canonical_digest(suite, digest_field="suite_digest")
    return suite


@pytest.fixture(scope="module")
def cousin_evidence(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[dict], list[dict]]:
    root = tmp_path_factory.mktemp("adp009d-cousins")
    manifests = [
        _load("adp009d_visual_material_cousin_manifest.v1.json"),
        _load("adp009d_geometric_cousin_manifest.v1.json"),
    ]
    receipts = []
    for manifest in manifests:
        cousin_id = manifest["cousin_id"]
        package_root = root / cousin_id / "simready_usd"
        package = materialize_cousin_package(
            cousin_manifest=manifest,
            repo_root=REPO_ROOT,
            output_dir=package_root,
        )
        report_dir = root / cousin_id / "validation"
        report_dir.mkdir()
        report_path = report_dir / "prop_robotics_physx_2_0_0.json"
        write_json(
            report_path,
            {
                str((package_root / package["root_layer"]).resolve()): {
                    "profile_id": "Prop-Robotics-Physx",
                    "profile_version": "2.0.0",
                    "features_summary": {
                        feature_id: {"passed": True, "version": version}
                        for feature_id, version in FEATURE_VERSIONS.items()
                    },
                }
            },
        )
        receipt = admit_cousin_static_validation(
            cousin_manifest=manifest,
            package_receipt=package,
            repo_root=REPO_ROOT,
            validator_report_path=report_path,
            output_path=report_dir
            / "adp009d_cousin_static_validation_receipt.v1.json",
        )
        receipts.append(receipt)
    return manifests, receipts


def test_task_neutral_constraints_admit_articulated_fridge_without_rigid_positions() -> None:
    fridge_digest = "sha256:" + "f" * 64
    harness = {
        "task_contract": {
            "schema_version": "adp_task_spec.v1",
            "task_kind": "articulated_open_close",
            "target_joint_id": "refrigerator_upper_door_hinge",
            "joint_reset_positions_rad": {
                "refrigerator_upper_door_hinge": 0.0,
                "refrigerator_lower_door_hinge": 0.0,
            },
            "target_success_interval_rad": [0.7853981633974483, 0.9599310885968813],
            "joint_hard_limits_rad": {
                "refrigerator_upper_door_hinge": [0.0, 1.5707963267948966],
                "refrigerator_lower_door_hinge": [0.0, 1.5707963267948966],
            },
        },
        "canonical_condition": {
            "object_id": "840796_articulated_refrigerator",
            "asset_binding": {
                "asset_id": "840796_articulated_refrigerator",
                "sha256": fridge_digest,
            },
        },
    }
    parameters = {
        "target_joint_reset_rad": 0.0,
        "scripted_positive_target_rad": 0.8726646259971648,
        "robot_base_x_m": 1.75,
        "robot_base_y_m": 1.99,
    }

    assert _canonical_asset_binding(harness) == (
        "840796_articulated_refrigerator",
        fridge_digest,
    )
    _check_instance_constraints(
        harness=harness,
        suite={"invalid_combinations": []},
        factor_records=[],
        parameters=parameters,
    )

    parameters["scripted_positive_target_rad"] = 1.2
    with pytest.raises(
        Adp009dHarnessError, match="scenario_articulated_scripted_target_invalid"
    ):
        _check_instance_constraints(
            harness=harness,
            suite={"invalid_combinations": []},
            factor_records=[],
            parameters=parameters,
        )


def _clear_sweep_receipt() -> dict:
    receipt = {
        "schema_version": "articulated_sage_mesh_sweep.v1",
        "status": "exact_sage_mesh_clearance_candidate_only",
        "first_collision": None,
        "collision_prim_paths": [],
        "claim_boundary": {
            "triangle_prism_intersection_tested": True,
            "full_stage_inventory_is_bound_broadphase": True,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def _door_state_receipt(
    *, classes: list[str], blocked: bool = False, source: str = "/Root/chair"
) -> dict:
    angles = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
    receipt = {
        "schema_version": "articulated_door_state_clearance.v1",
        "status": (
            "blocked_by_door_state_contact"
            if blocked
            else "door_state_matrix_clearance_candidate_only"
        ),
        "door_state_rows": [
            {
                "angle_degrees": angle,
                "sage_contact_prim_paths": [],
                "static_box_contacts": [],
                "clear": not blocked,
            }
            for angle in angles
        ],
        "static_obstacle_classes_bound": sorted(classes),
        "first_contact": (
            {
                "angle_degrees": 50.0,
                "source": source,
                "obstacle_class": "sage_static_scene",
            }
            if blocked
            else None
        ),
        "claim_boundary": {
            "triangle_prism_intersection_tested": True,
            "full_stage_inventory_is_bound_broadphase": True,
            "clear_result_is_not_native_dynamic_qualification": True,
            "replacement_self_geometry_bound": "replacement_body" in classes
            or "replacement_lower_door" in classes,
            "franka_base_bound": "franka_base" in classes,
            "ik_or_contact_qualified": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


_ARTICULATED_CONTRACT = {
    "schema_version": "adp_task_spec.v1",
    "task_kind": "articulated_open_close",
    "target_joint_id": "refrigerator_upper_door_hinge",
}


def test_clear_sweep_without_door_state_matrix_stays_unmaterializable() -> None:
    admission = admit_task_construction(
        task_contract=_ARTICULATED_CONTRACT,
        member_sweep_clearance=_clear_sweep_receipt(),
    )

    assert admission["scenario_materialization_authorized"] is False
    assert admission["placement_search_authorized"] is True
    assert "articulated_door_state_matrix_missing" in admission["blockers"]
    assert admission["door_state_clearance_receipt_digest"] is None


def test_clear_matrix_with_unbound_classes_blocks_materialization() -> None:
    admission = admit_task_construction(
        task_contract=_ARTICULATED_CONTRACT,
        member_sweep_clearance=_clear_sweep_receipt(),
        door_state_clearance=_door_state_receipt(classes=[]),
    )

    assert admission["scenario_materialization_authorized"] is False
    assert any(
        blocker.startswith("articulated_door_state_obstacle_classes_unbound:")
        for blocker in admission["blockers"]
    )


def test_blocked_door_state_matrix_blocks_materialization() -> None:
    admission = admit_task_construction(
        task_contract=_ARTICULATED_CONTRACT,
        member_sweep_clearance=_clear_sweep_receipt(),
        door_state_clearance=_door_state_receipt(
            classes=["replacement_body", "replacement_lower_door", "franka_base"],
            blocked=True,
        ),
    )

    assert admission["scenario_materialization_authorized"] is False
    assert any(
        blocker.startswith("articulated_door_state_contact:") for blocker in admission["blockers"]
    )


def test_fully_bound_clear_matrix_with_gates_admits_materialization() -> None:
    gates = [
        {"gate_id": gate_id, "status": "passed", "receipt_digest": "sha256:" + "9" * 64}
        for gate_id in sorted(
            {
                "source_link_partition",
                "source_visual_removal",
                "replacement_asset",
                "native_articulation_readback",
                "interior_exposure",
                "native_robot_placement",
                "native_phase_ik",
                "policy_camera_observability",
                "review_camera_observability",
            }
        )
    ]
    admission = admit_task_construction(
        task_contract=_ARTICULATED_CONTRACT,
        member_sweep_clearance=_clear_sweep_receipt(),
        door_state_clearance=_door_state_receipt(
            classes=["replacement_body", "replacement_lower_door", "franka_base"]
        ),
        construction_gate_receipts=gates,
    )

    assert admission["blockers"] == []
    assert admission["scenario_materialization_authorized"] is True
    assert admission["door_state_clearance_receipt_digest"].startswith("sha256:")
    assert validate_task_construction_admission(
        admission, task_contract=_ARTICULATED_CONTRACT
    ) == admission


def _native_probe_gate(status: str = "passed") -> dict:
    return {
        "gate_id": "native_articulation_readback",
        "status": status,
        "receipt_digest": "sha256:" + "7" * 64,
    }


def test_scenario_admission_requires_the_native_articulation_readback() -> None:
    """Static admission plus a clear sweep cannot stand in for native readback."""

    gates = [
        {"gate_id": gate_id, "status": "passed", "receipt_digest": "sha256:" + "9" * 64}
        for gate_id in sorted(
            {
                "source_link_partition",
                "source_visual_removal",
                "replacement_asset",
                "native_robot_placement",
                "native_phase_ik",
                "policy_camera_observability",
                "review_camera_observability",
            }
        )
    ]
    door_states = _door_state_receipt(
        classes=["replacement_body", "replacement_lower_door", "franka_base"]
    )

    without_probe = admit_task_construction(
        task_contract=_ARTICULATED_CONTRACT,
        member_sweep_clearance=_clear_sweep_receipt(),
        door_state_clearance=door_states,
        construction_gate_receipts=gates,
    )
    assert without_probe["scenario_materialization_authorized"] is False
    assert (
        "articulated_construction_gate_missing:native_articulation_readback"
        in without_probe["blockers"]
    )

    with_probe = admit_task_construction(
        task_contract=_ARTICULATED_CONTRACT,
        member_sweep_clearance=_clear_sweep_receipt(),
        door_state_clearance=door_states,
        construction_gate_receipts=[
            *gates,
            _native_probe_gate(),
            {
                "gate_id": "interior_exposure",
                "status": "passed",
                "receipt_digest": "sha256:" + "4" * 64,
            },
        ],
    )
    assert with_probe["blockers"] == []
    assert with_probe["scenario_materialization_authorized"] is True


def test_scenario_admission_requires_the_interior_to_be_reachable() -> None:
    """A door that opens onto a sealed carcass is not a usable articulated task.

    The 840796 twin passed articulation, limits, colliders and the
    required-interior check while its own shell walled the cavity off. Nothing
    in the gate set noticed, because every gate asked about parts rather than
    about what opening the door actually reveals.
    """

    base = {
        "source_link_partition",
        "source_visual_removal",
        "replacement_asset",
        "native_articulation_readback",
        "native_robot_placement",
        "native_phase_ik",
        "policy_camera_observability",
        "review_camera_observability",
    }
    gates = [
        {"gate_id": gate_id, "status": "passed", "receipt_digest": "sha256:" + "9" * 64}
        for gate_id in sorted(base)
    ]
    door_states = _door_state_receipt(
        classes=["replacement_body", "replacement_lower_door", "franka_base"]
    )

    without = admit_task_construction(
        task_contract=_ARTICULATED_CONTRACT,
        member_sweep_clearance=_clear_sweep_receipt(),
        door_state_clearance=door_states,
        construction_gate_receipts=gates,
    )
    assert without["scenario_materialization_authorized"] is False
    assert (
        "articulated_construction_gate_missing:interior_exposure"
        in without["blockers"]
    )

    with_gate = admit_task_construction(
        task_contract=_ARTICULATED_CONTRACT,
        member_sweep_clearance=_clear_sweep_receipt(),
        door_state_clearance=door_states,
        construction_gate_receipts=[
            *gates,
            {
                "gate_id": "interior_exposure",
                "status": "passed",
                "receipt_digest": "sha256:" + "4" * 64,
            },
        ],
    )
    assert with_gate["blockers"] == []
    assert with_gate["scenario_materialization_authorized"] is True
