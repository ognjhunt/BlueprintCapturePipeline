from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_franka_evaluation_harness import (
    Adp009dHarnessError,
    admit_cousin_static_validation,
    materialize_cousin_package,
    materialize_scenario_suite,
    validate_cousin_manifest,
    validate_cousin_static_validation_receipt,
    validate_harness_manifest,
    validate_scenario_suite,
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
    assert sage["runtime_active_triangle_mesh_count"] == 164
    assert sage["runtime_approximation"] == "none"
    assert sage["geometry_mutation_allowed"] is False


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
        "freeze_status": "frozen_pre_learned_outcomes",
        "harness_digest": harness["harness_digest"],
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
            "analysis_frozen_before_learned_outcomes": True,
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
