from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_franka_evaluation_harness import (
    Adp009dHarnessError,
    admit_task_construction,
    admit_cousin_static_validation,
    materialize_cousin_package,
    materialize_scenario_suite,
    validate_cousin_manifest,
    validate_cousin_static_validation_receipt,
    validate_harness_manifest,
    validate_scenario_suite,
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


def test_checked_in_harness_binds_static_sage_triangle_override() -> None:
    harness = _load("adp009d_franka_eval_harness_manifest.v1.json")

    validated = validate_harness_manifest(
        harness,
        repo_root=REPO_ROOT,
        evidence_root=REPO_ROOT,
        verify_files=False,
    )

    sage = validated["physics"]["entity_overrides"]["sealed_sage_static_collision"]
    assert sage["source_mesh_count"] == 165
    assert sage["source_convex_decomposition_count"] == 164
    assert sage["source_rigid_body_count"] == 0
    assert sage["runtime_active_triangle_mesh_count"] == 15
    assert sage["runtime_source_face_count"] == 47359
    assert sage["runtime_clipped_source_face_count"] == 24248
    assert sage["runtime_derived_face_count"] == 26828
    assert sage["runtime_derived_point_count"] == 80484
    assert sage["runtime_maximum_edge_m"] == 0.5
    assert sage["runtime_approximation"] == "none"
    assert sage["geometry_mutation_allowed"] is False
    assert sage["runtime_surface_preserving_derivative_allowed"] is True
    assert sage["out_of_envelope_source_colliders_active"] is False
    assert sage["physx_triangle_stability_warning_allowed"] is False
    assert sage["cold_cooking_is_startup_evidence_only"] is True
    settings = validated["physics"]["settings"]
    assert settings["collision_cooking_profile"] == (
        "legacy_cooker_after_ujitso_stall.v1"
    )
    assert settings["collision_cooking_backend"] == "legacy"
    assert settings["ujitso_collision_cooking"] is False
    assert settings["collision_geometry_or_parameters_changed"] is False
    assert validated["runtime_timing_receipt"]["fields_seconds"][0] == (
        "environment_build"
    )
    assert validated["canonical_condition"]["target_position_m"] == [
        3.750152333333333,
        -3.4074919,
        0.5264650138348479,
    ]
    assert validated["canonical_condition"]["target_selection"]["receipt_digest"] == (
        "sha256:d9f4a32dbc58adfb5e8e1112e30b8b490f7a928b5ade2c3fd1ae8b84bc7aaf79"
    )
    assert validated["candidate_pair"]["candidate_ids"] == [
        "pi05_droid",
        "groot_n17_droid",
    ]


def test_harness_rejects_forged_native_timing_or_provider_zero() -> None:
    harness = _load("adp009d_franka_eval_harness_manifest.v1.json")
    measurement = harness["runtime_timing_receipt"]["latest_measurement"]
    measurement["result_sha256"] = "sha256:" + "0" * 64
    measurement["provider_zero_observed"] = False
    harness["harness_digest"] = canonical_digest(
        harness, digest_field="harness_digest"
    )

    with pytest.raises(Adp009dHarnessError) as exc_info:
        validate_harness_manifest(
            harness,
            repo_root=REPO_ROOT,
            evidence_root=REPO_ROOT,
            verify_files=False,
        )

    assert "harness_runtime_timing_measurement_invalid" in exc_info.value.errors


def test_harness_rejects_sage_geometry_or_approximation_mutation() -> None:
    harness = _load("adp009d_franka_eval_harness_manifest.v1.json")
    sage = harness["physics"]["entity_overrides"]["sealed_sage_static_collision"]
    sage["source_face_count"] -= 1
    sage["runtime_approximation"] = "convexHull"
    harness["harness_digest"] = canonical_digest(harness, digest_field="harness_digest")

    with pytest.raises(
        Adp009dHarnessError,
        match="harness_sage_static_triangle_override_invalid",
    ):
        validate_harness_manifest(
            harness,
            repo_root=REPO_ROOT,
            evidence_root=REPO_ROOT,
            verify_files=False,
        )


def test_articulated_sweep_rejection_is_a_harness_materialization_decision(
    tmp_path: Path,
) -> None:
    rejection = _load(
        "second_scene_candidate_840411_right_door_exact_sage_mesh_sweep.v1.json"
    )
    task_contract = {
        "schema_version": "adp_task_spec.v1",
        "task_kind": "articulated_open_close",
        "target_joint_id": "refrigerator_right_door_hinge",
    }
    admission = admit_task_construction(
        task_contract=task_contract,
        member_sweep_clearance=rejection,
    )

    assert admission["scenario_materialization_authorized"] is False
    assert admission["placement_search_authorized"] is False
    exact_blocker = (
        "articulated_member_sweep_obstructed:"
        "/Root/ZTMAS3RVAUVREPTUKQ888888_005"
    )
    assert admission["blockers"] == [exact_blocker]
    assert validate_task_construction_admission(
        admission, task_contract=task_contract
    ) == admission

    harness = _load("adp009d_franka_eval_harness_manifest.v1.json")
    harness["task_contract"] = task_contract
    with pytest.raises(Adp009dHarnessError) as caught:
        materialize_scenario_suite(
            harness_manifest=harness,
            scenario_suite={},
            cousin_manifests=[],
            cousin_static_validation_receipts=[],
            output_dir=tmp_path / "blocked",
            task_construction_admission=admission,
        )

    assert caught.value.errors == (
        f"scenario_task_construction_not_admitted:{exact_blocker}",
    )
    assert not (tmp_path / "blocked").exists()


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


def test_checked_in_cousin_manifests_and_materialized_identity(
    cousin_evidence: tuple[list[dict], list[dict]],
) -> None:
    manifests, receipts = cousin_evidence
    for manifest, receipt in zip(manifests, receipts, strict=True):
        validate_cousin_manifest(manifest, repo_root=REPO_ROOT)
        validate_cousin_static_validation_receipt(
            receipt, cousin_manifest=manifest, verify_files=True
        )
        assert receipt["authored_usd"]["sha256"] == manifest["materialization"][
            "expected_self_contained_usd"
        ]["sha256"]


def test_cousin_manifest_rejects_fake_materialized_identity(tmp_path: Path) -> None:
    manifest = _load("adp009d_visual_material_cousin_manifest.v1.json")
    manifest["materialization"]["expected_self_contained_usd"]["sha256"] = (
        "sha256:" + "0" * 64
    )
    manifest["cousin_digest"] = canonical_digest(
        manifest, digest_field="cousin_digest"
    )
    with pytest.raises(
        Adp009dHarnessError, match="cousin_materialized_usd_identity_mismatch"
    ):
        materialize_cousin_package(
            cousin_manifest=manifest,
            repo_root=REPO_ROOT,
            output_dir=tmp_path / "invalid-identity",
        )


def test_static_receipt_rejects_caller_authored_profile_pass(
    cousin_evidence: tuple[list[dict], list[dict]],
) -> None:
    manifests, receipts = cousin_evidence
    receipt = copy.deepcopy(receipts[0])
    receipt["validator"]["profile_version"] = "invented"
    receipt["validation_receipt_digest"] = canonical_digest(
        receipt, digest_field="validation_receipt_digest"
    )
    with pytest.raises(
        Adp009dHarnessError,
        match="cousin_static_validation_receipt_validator_identity_invalid",
    ):
        validate_cousin_static_validation_receipt(
            receipt, cousin_manifest=manifests[0], verify_files=True
        )


def test_scenario_materialization_is_stable_policy_neutral_and_canonical(
    tmp_path: Path,
    cousin_evidence: tuple[list[dict], list[dict]],
) -> None:
    harness = _load("adp009d_franka_eval_harness_manifest.v1.json")
    manifests, receipts = cousin_evidence
    suite = _valid_suite(harness)
    validate_scenario_suite(
        suite,
        harness_manifest=harness,
        cousin_manifests=manifests,
        cousin_static_validation_receipts=receipts,
    )
    first = materialize_scenario_suite(
        harness_manifest=harness,
        scenario_suite=suite,
        cousin_manifests=manifests,
        cousin_static_validation_receipts=receipts,
        output_dir=tmp_path / "first",
    )
    second = materialize_scenario_suite(
        harness_manifest=harness,
        scenario_suite=suite,
        cousin_manifests=manifests,
        cousin_static_validation_receipts=receipts,
        output_dir=tmp_path / "second",
    )
    assert first.receipt["materialization_digest"] == second.receipt[
        "materialization_digest"
    ]
    assert [row["instance_digest"] for row in first.instances] == [
        row["instance_digest"] for row in second.instances
    ]
    assert len(first.instances) == 128
    canonical = [row for row in first.instances if row["family"] == "canonical"]
    assert len(canonical) == 16
    assert all(row["factor_records"] == [] for row in canonical)
    assert all(
        row["resolved_parameters"] == harness["canonical_condition"]["parameters"]
        for row in canonical
    )
    assert all(row["policy_neutral"] is True for row in first.instances)
    assert all(
        row["required_controls"]
        == ["deterministic_scripted_positive", "zero_action_negative"]
        for row in first.instances
    )


def test_checked_in_scenario_suite_is_frozen_bounded_and_materializable(
    tmp_path: Path,
    cousin_evidence: tuple[list[dict], list[dict]],
) -> None:
    harness = _load("adp009d_franka_eval_harness_manifest.v1.json")
    suite = _load("adp009d_scenario_suite.v1.json")
    manifests, receipts = cousin_evidence

    validated = validate_scenario_suite(
        suite,
        harness_manifest=harness,
        cousin_manifests=manifests,
        cousin_static_validation_receipts=receipts,
    )
    materialized = materialize_scenario_suite(
        harness_manifest=harness,
        scenario_suite=validated,
        cousin_manifests=manifests,
        cousin_static_validation_receipts=receipts,
        output_dir=tmp_path / "checked-in-suite",
    )

    assert validated["freeze_status"] == (
        "frozen_after_canonical_canary_before_scenario_evaluation"
    )
    assert validated["prior_canary_disclosure"][
        "prior_outcomes_used_to_select_parameters"
    ] is False
    assert validated["power_cost_analysis"]["planned_episode_count"] == 512
    assert validated["power_cost_analysis"]["estimated_total_gpu_hours"] == pytest.approx(
        18.488888888888887
    )
    assert Counter(row["family"] for row in materialized.instances) == {
        "canonical": 16,
        "placement_approach": 16,
        "illumination": 16,
        "camera_sensor": 16,
        "physics": 16,
        "visual_material_cousin": 16,
        "geometric_cousin": 16,
        "held_out_composed": 16,
    }
    assert all(
        factor["resolved_value"] != factor["nominal_value"]
        for instance in materialized.instances
        for factor in instance["factor_records"]
    )
    assert all(
        instance["resolved_parameters"]["target_x_m"] == 3.750152333333333
        and instance["resolved_parameters"]["target_y_m"] == -3.4074919
        for instance in materialized.instances
    )
    checked_canonical = _load("adp009d_canonical_scenario_instance.v1.json")
    canonical = next(
        instance
        for instance in materialized.instances
        if instance["cell_id"] == "canonical_anchor.seed_2026080600"
    )
    assert canonical == checked_canonical


def test_scenario_suite_rejects_missing_cousin_receipt(
    cousin_evidence: tuple[list[dict], list[dict]],
) -> None:
    harness = _load("adp009d_franka_eval_harness_manifest.v1.json")
    manifests, receipts = cousin_evidence
    suite = _valid_suite(harness)
    with pytest.raises(
        Adp009dHarnessError,
        match="scenario_suite_cousin_static_receipt_set_invalid",
    ):
        validate_scenario_suite(
            suite,
            harness_manifest=harness,
            cousin_manifests=manifests,
            cousin_static_validation_receipts=receipts[:1],
        )


def test_scenario_suite_rejects_caller_asserted_outcome(
    cousin_evidence: tuple[list[dict], list[dict]],
) -> None:
    harness = _load("adp009d_franka_eval_harness_manifest.v1.json")
    manifests, receipts = cousin_evidence
    suite = _valid_suite(harness)
    suite["task_success"] = True
    suite["suite_digest"] = canonical_digest(suite, digest_field="suite_digest")
    with pytest.raises(
        Adp009dHarnessError, match="scenario_suite_caller_asserted_outcome_forbidden"
    ):
        validate_scenario_suite(
            suite,
            harness_manifest=harness,
            cousin_manifests=manifests,
            cousin_static_validation_receipts=receipts,
        )


def test_scenario_materialization_rejects_invalid_composed_cell(
    tmp_path: Path,
    cousin_evidence: tuple[list[dict], list[dict]],
) -> None:
    harness = _load("adp009d_franka_eval_harness_manifest.v1.json")
    manifests, receipts = cousin_evidence
    suite = _valid_suite(harness)
    suite["invalid_combinations"] = [
        {
            "constraint_id": "held_out_light_mass_forbidden",
            "when_all_non_nominal": ["light_intensity_scale", "object_mass_kg"],
            "behavior": "reject_instance_fail_closed",
        }
    ]
    suite["suite_digest"] = canonical_digest(suite, digest_field="suite_digest")
    with pytest.raises(
        Adp009dHarnessError,
        match="scenario_invalid_combination:held_out_light_mass_forbidden",
    ):
        materialize_scenario_suite(
            harness_manifest=harness,
            scenario_suite=suite,
            cousin_manifests=manifests,
            cousin_static_validation_receipts=receipts,
            output_dir=tmp_path / "invalid",
        )


def test_cousin_authoring_is_path_independent(tmp_path: Path) -> None:
    """The flattened cousin must not embed the checkout path.

    USD's flatten writes the composed root layer's absolute path into the
    output layer's doc string, which made the pinned materialization digest
    verify only in the directory it was authored in -- every other checkout,
    worktree, or CI runner failed ``cousin_materialized_usd_identity_mismatch``
    at fixture setup.  Identity must be a property of the sealed inputs, not
    of where the repository happens to sit on disk.
    """

    from blueprint_pipeline.adp009d_franka_evaluation_harness import (
        _author_flattened_cousin,
        _mapping,
        _resolve_file_record,
    )

    manifest = _load("adp009d_visual_material_cousin_manifest.v1.json")
    base = _resolve_file_record(
        _mapping(manifest["base_asset"]),
        repo_root=REPO_ROOT,
        evidence_root=REPO_ROOT,
        error_prefix="cousin_base",
    )
    overlay = _resolve_file_record(
        _mapping(manifest["overlay_asset"]),
        repo_root=REPO_ROOT,
        evidence_root=REPO_ROOT,
        error_prefix="cousin_overlay",
    )

    authored: list[bytes] = []
    for name in ("root-a", "root-b"):
        output = tmp_path / name / f"{manifest['cousin_id']}.usda"
        output.parent.mkdir(parents=True)
        _author_flattened_cousin(
            base_path=base,
            overlay_path=overlay,
            output_path=output,
            cousin_type=str(manifest["cousin_type"]),
            dimensions=_mapping(manifest.get("dimensions_m")),
        )
        content = output.read_bytes()
        assert str(REPO_ROOT).encode() not in content
        assert str(tmp_path).encode() not in content
        authored.append(content)

    assert authored[0] == authored[1]


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
