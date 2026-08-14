from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from blueprint_pipeline.adp009d_physics_backend_comparison import (
    CANARY_ADMISSION_SCHEMA_VERSION,
    COMPARABILITY_BINDINGS,
    FRANKA_CORRECTED_DIAGONAL_INERTIA_KG_M2,
    FRANKA_INERTIA_UNIT_CORRECTION_FACTOR,
    FRANKA_SOURCE_DIAGONAL_INERTIA_KG_M2,
    FRANKA_SOURCE_MESH_SCALE,
    GRAVITY_REAL_ARM_ACTUATOR_GROUPS,
    MEASUREMENT_FIELDS,
    NEWTON_MAPPED_PHYSX_PROPERTY_NAMES,
    NEWTON_MAPPED_PHYSX_PROPERTY_PREFIXES,
    NEWTON_UNREPRESENTABLE_PHYSX_PROPERTY_NAMES,
    PROBE_SCHEMA_VERSION,
    PhysicsBackendContractError,
    build_backend_control_run_receipt,
    build_backend_profile,
    build_comparison_design_contract,
    build_gravity_real_actuation_contract,
    build_comparison_receipt,
    build_newton_canary_admission,
    build_newton_canary_terminal_receipt,
    build_newton_actuator_limit_mapping_contract,
    build_newton_robot_inertial_overlay_contract,
    normalize_physics_backend,
    validate_backend_probe,
    validate_backend_profile,
    validate_comparison_receipt,
    validate_comparison_design_contract,
    validate_newton_canary_admission,
    validate_gravity_real_actuation,
    validate_newton_dynamics_representable,
    validate_newton_explicit_pd_feasibility,
)
from blueprint_pipeline.adp009d_approach_capture import (
    next_episode_start_restore_command,
)
from blueprint_pipeline.adp009d_newton_gripper_drive import (
    build_newton_gripper_drive_candidate,
)
from blueprint_pipeline.adp009d_provider_zero import build_provider_zero_receipt
from blueprint_pipeline.adp009d_control_episode import materialize_control_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.spend_admission_lock import build_spend_admission_lock


ROOT = Path(__file__).resolve().parents[1]
COMMITTED_DESIGN = (
    ROOT
    / "docs/arm_decision_proof_v1/manifests/"
    "adp009d_physics_backend_comparison.v1.json"
)


def _newton_terminal_inputs() -> dict[str, dict]:
    profile = build_backend_profile("newton")
    provider_inventory = build_provider_zero_receipt(
        {
            provider: {
                "provider": provider,
                "status": "observed",
                "api_confirmed": True,
                "resources": [],
                "blockers": [],
            }
            for provider in ("runpod", "vast", "digitalocean")
        },
        now=datetime(2026, 8, 11, 18, 0, tzinfo=timezone.utc),
    )
    inputs = {
        "admission": {
            "schema_version": CANARY_ADMISSION_SCHEMA_VERSION,
            "status": "passed",
            "backend_profile_digest": profile["profile_digest"],
            "controls_only": True,
            "policy_query_allowed": False,
            "retry_cap": 0,
            "max_spend_usd": 2.0,
            "admission_digest": "sha256:admission",
        },
        "bundle_receipt": {
            "status": "ready",
            "physics_backend": "newton",
            "physics_backend_profile_digest": profile["profile_digest"],
            "controls_requested": True,
            "policy_candidate_id": None,
            "candidate_policy_queried": False,
            "candidate_outcomes_accessed": False,
            "retry_cap": 0,
            "implementation_commit": "a" * 40,
            "bundle_sha256": "sha256:bundle",
            "input_digest": "sha256:input",
            "scenario_instance_digest": "sha256:scenario",
            "control_plan_digest": "sha256:plan",
            "control_plan_semantic_digest": "sha256:semantic",
        },
        "allocator_result": {
            "status": "blocked",
            "blockers": ["adp009d_backend_runtime_version_mismatch"],
            "retry_cap": 0,
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
        },
        "native_result": {
            "schema_version": "adp009d_native_microcheck.v1",
            "status": "blocked",
            "blockers": ["adp009d_backend_runtime_version_mismatch"],
            "candidate_policy_queried": False,
            "candidate_outcomes_accessed": False,
        },
        "artifact_manifest": {
            "status": "completed",
            "blockers": [],
            "file_count": 20,
            "total_size_bytes": 234318,
            "manifest_digest": "sha256:artifacts",
            "required_roles": ["provider_runtime_evidence", "teardown_manifest"],
            "observed_roles": ["provider_runtime_evidence", "teardown_manifest"],
        },
        "teardown_manifest": {
            "generated_at": "2026-08-11T17:59:00+00:00",
            "status": "completed",
            "vast_instance_ids": [123],
            "runner_gpu_teardown_completed": True,
            "continuing_spend_from_this_run": False,
            "teardown_actions_performed": [
                {
                    "instance_id": 123,
                    "action": "destroy_instance",
                    "status": "completed",
                }
            ],
        },
        "provider_inventory": provider_inventory,
        "vast_charge": {
            "type": "instance",
            "source": "instance-123",
            "amount": 0.321,
            "items": [
                {"type": "gpu", "description": "gpu", "amount": 0.25},
                {"type": "disk", "description": "disk", "amount": 0.058},
                {"type": "bwd", "description": "network", "amount": 0.013},
            ],
        },
    }
    inputs["admission"]["admission_digest"] = canonical_digest(
        inputs["admission"], digest_field="admission_digest"
    )
    return inputs


def test_newton_blocked_canary_retains_terminal_comparison_evidence() -> None:
    receipt = build_newton_canary_terminal_receipt(**_newton_terminal_inputs())

    assert receipt["status"] == "blocked"
    assert receipt["scientific_phase"] == "pre_controls_blocked"
    assert receipt["media_gap"]["status"] == "typed_gap"
    assert receipt["spend"]["actual_provider_charge_usd"] == 0.321
    assert receipt["provider_zero"]["api_confirmed"] is True
    assert receipt["provider_zero"]["live_instance_count"] == 0
    assert receipt["policy_verdict"] is None
    assert receipt["engine_promotion_performed"] is False
    assert receipt["claim_ceiling"] == "controls_comparison_evidence_only"
    assert set(receipt["evidence_input_digests"]) == {
        "admission",
        "allocator_result",
        "artifact_manifest",
        "bundle_receipt",
        "native_result",
        "provider_inventory",
        "teardown_manifest",
        "vast_charge",
    }
    assert receipt["terminal_receipt_digest"] == canonical_digest(
        receipt, digest_field="terminal_receipt_digest"
    )


def test_newton_pre_runtime_failure_retains_terminal_receipt_without_fake_native_result() -> None:
    inputs = _newton_terminal_inputs()
    inputs["native_result"] = None
    inputs["allocator_result"].update(
        {
            "native_control_result_path": None,
            "blockers": [
                "adp009d_candidate_policy_query_status_missing",
                "adp009d_provider_output_zip_missing",
                "adp009d_runtime_not_completed",
                "task_evaluation_artifact_role_missing:provider_runtime_evidence",
                "vast_heartbeat_instance_exited",
            ],
        }
    )
    inputs["artifact_manifest"].update(
        {
            "status": "blocked",
            "blockers": [
                "task_evaluation_artifact_role_missing:provider_runtime_evidence"
            ],
            "file_count": 2,
            "required_roles": [
                "allocator_adapter_result",
                "provider_runtime_evidence",
                "teardown_manifest",
            ],
            "observed_roles": [
                "allocator_adapter_result",
                "teardown_manifest",
            ],
        }
    )

    receipt = build_newton_canary_terminal_receipt(**inputs)

    assert receipt["status"] == "blocked"
    assert receipt["scientific_phase"] == "pre_runtime_blocked"
    assert receipt["native_runtime_evidence_observed"] is False
    assert receipt["media_gap"]["reason"] == "adp009d_runtime_not_completed"
    assert "native_result" not in receipt["evidence_input_digests"]
    assert receipt["policy_query_count"] == 0
    assert receipt["policy_verdict"] is None


def test_newton_terminal_receipt_preserves_a_prior_immutable_backend_profile() -> None:
    inputs = _newton_terminal_inputs()
    executed_profile = deepcopy(build_backend_profile("newton"))
    executed_profile["required_capabilities"].pop("newton_actuator_limit_mapping")
    executed_profile.pop("actuator_limit_mapping")
    executed_profile["profile_digest"] = canonical_digest(
        executed_profile, digest_field="profile_digest"
    )
    inputs["admission"]["backend_profile_digest"] = executed_profile["profile_digest"]
    inputs["admission"]["admission_digest"] = canonical_digest(
        inputs["admission"], digest_field="admission_digest"
    )
    inputs["bundle_receipt"]["physics_backend_profile_digest"] = executed_profile[
        "profile_digest"
    ]

    receipt = build_newton_canary_terminal_receipt(
        **inputs, backend_profile=executed_profile
    )

    assert receipt["backend_profile_digest"] == executed_profile["profile_digest"]
    assert receipt["evidence_input_digests"]["backend_profile"] == canonical_digest(
        executed_profile
    )


def test_newton_pre_runtime_terminal_receipt_rejects_ambiguous_missing_runtime() -> None:
    inputs = _newton_terminal_inputs()
    inputs["native_result"] = None
    inputs["allocator_result"].update(
        {
            "native_control_result_path": None,
            "blockers": ["adp009d_runtime_not_completed"],
        }
    )

    with pytest.raises(
        PhysicsBackendContractError,
        match="adp009d_newton_terminal_pre_runtime_invalid",
    ):
        build_newton_canary_terminal_receipt(**inputs)


def test_newton_terminal_receipt_rejects_nonzero_provider_inventory() -> None:
    inputs = _newton_terminal_inputs()
    inputs["provider_inventory"]["live_instance_count"] = 1

    with pytest.raises(
        PhysicsBackendContractError,
        match="adp009d_newton_terminal_provider_zero_invalid",
    ):
        build_newton_canary_terminal_receipt(**inputs)


def test_newton_terminal_receipt_rejects_provider_zero_from_before_teardown() -> None:
    inputs = _newton_terminal_inputs()
    inputs["provider_inventory"]["generated_at"] = "2026-08-11T17:58:59+00:00"
    inputs["provider_inventory"]["receipt_digest"] = canonical_digest(
        inputs["provider_inventory"], digest_field="receipt_digest"
    )

    with pytest.raises(
        PhysicsBackendContractError,
        match="adp009d_newton_terminal_provider_zero_invalid",
    ):
        build_newton_canary_terminal_receipt(**inputs)


def test_newton_terminal_receipt_rejects_unsubstantiated_completed_status() -> None:
    inputs = _newton_terminal_inputs()
    inputs["allocator_result"]["status"] = "completed"
    inputs["allocator_result"]["blockers"] = []
    inputs["native_result"]["status"] = "completed"
    inputs["native_result"]["blockers"] = []

    with pytest.raises(
        PhysicsBackendContractError,
        match="adp009d_newton_terminal_completed_controls_invalid",
    ):
        build_newton_canary_terminal_receipt(**inputs)


def test_newton_terminal_receipt_cli_compiles_retained_evidence(
    tmp_path: Path,
) -> None:
    inputs = _newton_terminal_inputs()
    vast_charge = inputs.pop("vast_charge")
    paths: dict[str, Path] = {}
    for name, value in inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    billing_path = tmp_path / "vast_billing_response.json"
    billing_path.write_text(
        json.dumps({"results": [vast_charge]}), encoding="utf-8"
    )
    output_path = tmp_path / "terminal_receipt.json"
    script = (
        ROOT / "scripts/build_adp009d_newton_canary_terminal_receipt.py"
    )
    command = [sys.executable, str(script)]
    for name, path in paths.items():
        command.extend((f"--{name.replace('_', '-')}", str(path)))
    command.extend(
        (
            "--vast-billing-response",
            str(billing_path),
            "--output",
            str(output_path),
        )
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(output_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "blocked"
    assert receipt["scientific_phase"] == "pre_controls_blocked"
    assert receipt["spend"]["actual_provider_charge_usd"] == 0.321
    assert receipt["terminal_receipt_digest"] == canonical_digest(
        receipt, digest_field="terminal_receipt_digest"
    )


def _probe(profile: dict) -> dict:
    overlay = profile["asset_conversion"].get("robot_inertial_overlay") or {}
    value = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "passed",
        "physics_backend": profile["physics_backend"],
        "backend_profile_digest": profile["profile_digest"],
        "backend_active_at_simulation_construction": True,
        "backend_switch_attempted": False,
        "backend_switch_observed": False,
        "runtime_identity": deepcopy(profile["runtime_identity"]),
        "source_bindings": deepcopy(profile["source_bindings"]),
        "capabilities": {
            name: True for name in profile["required_capabilities"]
        },
        "solver_configuration": deepcopy(profile["solver_configuration"]),
        "contact_readback": {
            "force_vectors_world_n": [[1.0, 2.0, 3.0]],
            "partner_prim_paths": (
                ["/World/envs/env_0/approved_can/colliders/body_collider"]
                if profile["physics_backend"] == "newton"
                else ["/World/envs/env_0/approved_can"]
            ),
            "partner_native_identifier_kind": (
                "newton_shape_label"
                if profile["physics_backend"] == "newton"
                else "usd_rigid_body_prim"
            ),
            "partner_native_identifiers": (
                ["body_collider"]
                if profile["physics_backend"] == "newton"
                else ["/World/envs/env_0/approved_can"]
            ),
        },
        "asset_conversion": {
            "source_asset_digest": profile["source_bindings"]["approved_can_digest"],
            "converted_model_digest": "sha256:" + "a" * 64,
            "silently_ignored_settings": [],
            "physx_sdf_overlay_loaded": False,
            "physx_only_fields_observed": [],
            "robot_source_asset_digest": profile["source_bindings"].get(
                "droid_franka_robotiq_usd_digest"
            ),
            "robot_inertial_overlay_contract_digest": overlay.get(
                "overlay_digest"
            ),
            "robot_inertial_overlay_status": (
                "applied_and_verified"
                if profile["physics_backend"] == "newton"
                else None
            ),
            "robot_inertial_overlay_receipt_digest": (
                "sha256:" + "b" * 64
                if profile["physics_backend"] == "newton"
                else None
            ),
            "robot_source_mutated": (
                False if profile["physics_backend"] == "newton" else None
            ),
            "newton_actuator_limit_mapping_contract_digest": (
                profile["actuator_limit_mapping"]["mapping_digest"]
                if profile["physics_backend"] == "newton"
                else None
            ),
            "newton_actuator_limit_mapping_status": (
                "applied_and_verified"
                if profile["physics_backend"] == "newton"
                else None
            ),
            "newton_actuator_limit_mapping_receipt_digest": (
                "sha256:" + "c" * 64
                if profile["physics_backend"] == "newton"
                else None
            ),
            "newton_gripper_drive_contract_digest": (
                profile["gripper_drive_candidate"]["contract_digest"]
                if profile["physics_backend"] == "newton"
                else None
            ),
            "newton_gripper_drive_status": (
                "applied_for_native_identification"
                if profile["physics_backend"] == "newton"
                else None
            ),
            "newton_gripper_drive_receipt_digest": (
                "sha256:" + "d" * 64
                if profile["physics_backend"] == "newton"
                else None
            ),
        },
        "gripper_drive_trace": (
            {"status": "passed", "blockers": [], "commands": {}}
            if profile["physics_backend"] == "newton"
            else None
        ),
        "contact_buffer": {"nconmax": 1024, "overflow_observed": False},
        "policy_query_count": 0,
        "candidate_outcomes_accessed": False,
        "task_success_claimed": False,
        "physical_claimed": False,
    }
    value["probe_digest"] = canonical_digest(value, digest_field="probe_digest")
    return value


def test_newton_profile_binds_comparison_ineligible_gripper_drive_candidate() -> None:
    profile = build_backend_profile("newton")

    assert profile["gripper_drive_candidate"] == build_newton_gripper_drive_candidate()
    assert profile["gripper_drive_candidate"]["comparison_eligible"] is False
    assert "gripper_drive_candidate" not in build_backend_profile("physx")


def _admission(profile: dict, now: datetime) -> dict:
    value = {
        "schema_version": CANARY_ADMISSION_SCHEMA_VERSION,
        "status": "passed",
        "backend_profile_digest": profile["profile_digest"],
        "controls_only": True,
        "policy_query_allowed": False,
        "candidate_outcome_access_allowed": False,
        "canonical_allocator": (
            "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
        ),
        "authorization_evidence_ref": "codex-thread:test-explicit-authorization",
        "issued_at": now.isoformat(),
        "explicit_paid_run_authorization": True,
        "canonical_spend_admission": True,
        "canonical_spend_admission_digest": "sha256:" + "f" * 64,
        "watchdog_required": True,
        "artifact_storage_required": True,
        "teardown_required": True,
        "provider_inventory_precheck_mode": "provider_zero",
        "allowed_active_vast_instance_ids": [],
        "provider_zero_precheck_passed": True,
        "exact_concurrency_precheck_passed": False,
        "unapproved_live_instance_count": 0,
        "provider_zero_precheck_digest": "sha256:" + "0" * 64,
        "retry_cap": 0,
        "max_spend_usd": 2.0,
        "hard_ttl_seconds": 3600,
        "expires_at": (now + timedelta(minutes=15)).isoformat(),
        "provider_mutation_performed": False,
    }
    value["admission_digest"] = canonical_digest(
        value, digest_field="admission_digest"
    )
    return value


def _run(backend: str) -> dict:
    bindings = {
        name: (
            20260806
            if name == "seed"
            else f"{name}-same-value"
        )
        for name in COMPARABILITY_BINDINGS
    }
    measurements = {
        "initialization_reset": {
            "initialization_completed": True,
            "reset_completed": True,
        },
        "target_robot_pose": {
            "target_pose_world": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "robot_pose_world": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        "contacts_and_force_vectors": {
            "force_vectors_world_n": [[1.0, 0.0, 0.0]],
            "partner_prim_paths": ["/World/approved_can"],
        },
        "torque_utilization_and_clipping": {
            "maximum_utilization": 0.5,
            "clipping_observed": False,
        },
        "closest_geometric_clearance": {"minimum_m": 0.008},
        "action_delivery": {
            "requested_count": 10,
            "delivered_count": 10,
            "nontrivial_action_delivered": True,
        },
        "phase_completion": {"rows": [{"phase_id": "pregrasp"}]},
        "lossless_frames": {
            "frame_manifest_digest": "sha256:" + "b" * 64,
            "frame_count": 2,
            "lossless": True,
        },
        "review_media": {
            "media_digest": "sha256:" + "c" * 64,
            "derived_from_lossless_frames": True,
        },
        "teardown": {"completed": True, "continuing_spend": False},
        "spend": {"total_usd": 0.5},
        "provider_zero": {"api_confirmed": True, "live_instance_count": 0},
    }
    assert set(measurements) == set(MEASUREMENT_FIELDS)
    return build_backend_control_run_receipt(
        physics_backend=backend,
        comparability_bindings=bindings,
        backend_control_plan_digest=(
            "sha256:" + ("d" if backend == "physx" else "e") * 64
        ),
        measurements=measurements,
    )


def _fidelity() -> dict:
    return {
        "status": "observed",
        "metric_id": "closest_geometric_clearance_error_m",
        "metric_authority": "deterministic_geometry",
        "direction": "lower_is_better",
        "physx_value": 0.01,
        "newton_value": 0.008,
        "delta": -0.002,
        "meaningful_threshold": -0.001,
        "meaningful_improvement_observed": True,
    }


def test_backend_contract_is_strict_and_physx_remains_default_profile() -> None:
    physx = build_backend_profile("physx")
    newton = build_backend_profile("newton")

    assert validate_backend_profile(physx) == []
    assert validate_backend_profile(newton) == []
    assert physx["maturity"] == "production_baseline"
    assert newton["maturity"] == "experimental_comparison_candidate"
    assert newton["solver_configuration"]["nconmax"] == 1024
    assert newton["contact_model"]["physx_sdf_semantics_inherited"] is False
    with pytest.raises(PhysicsBackendContractError):
        normalize_physics_backend("PhysX")
    with pytest.raises(PhysicsBackendContractError):
        normalize_physics_backend(None)


def test_newton_robot_inertial_overlay_is_exact_and_digest_bound() -> None:
    overlay = build_newton_robot_inertial_overlay_contract()

    assert overlay == build_backend_profile("newton")["asset_conversion"][
        "robot_inertial_overlay"
    ]
    assert overlay["expected_source_body_count"] == 9
    assert len(overlay["body_masses_kg"]) == 9
    assert all(value > 0.0 for value in overlay["body_masses_kg"].values())
    for side in ("left", "right"):
        assert overlay["body_masses_kg"][f"{side}_inner_knuckle"] > 0.0
        assert overlay["body_masses_kg"][f"{side}_outer_knuckle"] > 0.0
    assert overlay["authored_center_of_mass_allowed"] is False
    assert overlay["robotiq_authored_diagonal_inertia_allowed"] is False
    assert overlay["franka_exact_unit_corrected_diagonal_inertia_required"] is True
    conversion = overlay["franka_inertia_unit_conversion"]
    assert conversion["expected_stage_meters_per_unit"] == 1.0
    assert conversion["source_mesh_scale"] == FRANKA_SOURCE_MESH_SCALE
    assert conversion["correction_factor"] == FRANKA_INERTIA_UNIT_CORRECTION_FACTOR
    assert conversion["source_diagonal_inertia_kg_m2"] == {
        name: list(value)
        for name, value in FRANKA_SOURCE_DIAGONAL_INERTIA_KG_M2.items()
    }
    assert conversion["corrected_diagonal_inertia_kg_m2"] == {
        name: list(value)
        for name, value in FRANKA_CORRECTED_DIAGONAL_INERTIA_KG_M2.items()
    }
    assert conversion["arbitrary_minimum_inertia_clamp_allowed"] is False
    assert overlay["authored_principal_axes_allowed"] is False
    assert overlay["arbitrary_minimum_mass_or_inertia_clamp_allowed"] is False
    assert overlay["usd_float32_mass_roundtrip_tolerance_kg"] == 2.0e-8
    admission = overlay["physx_property_admission"]
    assert admission["unmapped_authored_property_policy"] == (
        "block_value_before_newton_model_import"
    )
    assert admission["physx_contact_report_api_activation"] is False
    assert admission["arena_solver_iteration_overrides_authored"] is False
    assert (
        admission["arena_max_depenetration_velocity_override_authored"] is False
    )
    assert not any(
        "solverPositionIterationCount" in name
        or "solverVelocityIterationCount" in name
        for name in admission["mapped_property_names"]
    )
    assert overlay["source_robot_asset"]["source_mutated"] is False
    assert overlay["overlay_digest"] == canonical_digest(
        overlay, digest_field="overlay_digest"
    )


def test_committed_design_is_canonical_and_provider_free() -> None:
    committed = json.loads(COMMITTED_DESIGN.read_text(encoding="utf-8"))

    assert committed == build_comparison_design_contract()
    assert validate_comparison_design_contract(committed) == []
    assert committed["default_physics_backend"] == "physx"
    assert committed["provider_mutation_performed"] is False
    assert committed["status"] == "validated_without_provider_launch"


def test_newton_actuator_limit_mapping_is_exact_and_non_retuning() -> None:
    mapping = build_newton_actuator_limit_mapping_contract()

    assert mapping["physics_backend"] == "newton"
    assert mapping["legacy_fields_must_be_cleared"] is True
    assert mapping["retune_or_fidelity_claimed"] is False
    assert mapping["actuators"]["panda_shoulder"]["effort_limit_sim"] == 87.0
    assert mapping["actuators"]["panda_forearm"]["velocity_limit_sim"] == 2.61
    assert mapping["mapping_digest"] == canonical_digest(
        mapping, digest_field="mapping_digest"
    )


def test_backend_plans_share_semantics_but_not_contact_configuration() -> None:
    instance_path = (
        ROOT
        / "docs/arm_decision_proof_v1/manifests/"
        "adp009d_canonical_scenario_instance.v1.json"
    )
    instance = json.loads(instance_path.read_text(encoding="utf-8"))

    physx = materialize_control_plan(instance, physics_backend="physx")
    newton = materialize_control_plan(instance, physics_backend="newton")

    assert physx["semantic_plan_digest"] == newton["semantic_plan_digest"]
    assert physx["plan_digest"] != newton["plan_digest"]
    assert physx["contact_envelope"] is not None
    assert newton["contact_envelope"] is None
    assert newton["backend_contact_configuration"]["contact_model"][
        "physx_sdf_semantics_inherited"
    ] is False


def test_profile_and_digest_drift_fail_closed() -> None:
    profile = build_backend_profile("newton")
    profile["runtime_identity"]["isaac_sim_version"] = "latest"

    assert validate_backend_profile(profile) == [
        "adp009d_backend_profile_digest_invalid",
        "adp009d_backend_profile_drifted",
    ]


def test_newton_profile_rejects_physx_only_fields_even_with_recomputed_digest() -> None:
    profile = build_backend_profile("newton")
    profile["solver_configuration"]["sdf_margin_m"] = 0.001
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )

    blockers = validate_backend_profile(profile)

    assert "adp009d_newton_profile_contains_physx_only_fields" in blockers
    assert "adp009d_backend_profile_drifted" in blockers


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("franka_import", "adp009d_backend_probe_capability_missing"),
        ("contact_partner_readback", "adp009d_backend_probe_capability_missing"),
    ],
)
def test_unsupported_franka_or_contact_readback_blocks_probe(
    field: str, expected: str
) -> None:
    profile = build_backend_profile("newton")
    probe = _probe(profile)
    probe["capabilities"][field] = False
    probe["probe_digest"] = canonical_digest(probe, digest_field="probe_digest")

    assert expected in validate_backend_probe(probe, profile=profile)


def test_newton_probe_rejects_requested_filter_without_resolved_native_partner() -> None:
    profile = build_backend_profile("newton")
    probe = _probe(profile)
    probe["contact_readback"]["partner_native_identifiers"] = [
        "*body_collider"
    ]
    probe["probe_digest"] = canonical_digest(probe, digest_field="probe_digest")

    assert "adp009d_backend_probe_contact_readback_invalid" in (
        validate_backend_probe(probe, profile=profile)
    )


def test_probe_rejects_silent_physx_field_handling_and_contact_overflow() -> None:
    profile = build_backend_profile("newton")
    probe = _probe(profile)
    probe["asset_conversion"]["silently_ignored_settings"] = [
        "physxSDFMeshCollision:sdfMargin"
    ]
    probe["contact_buffer"]["overflow_observed"] = True
    probe["probe_digest"] = canonical_digest(probe, digest_field="probe_digest")

    blockers = validate_backend_probe(probe, profile=profile)

    assert "adp009d_backend_probe_asset_conversion_invalid" in blockers
    assert "adp009d_newton_probe_contact_model_invalid" in blockers


def test_provider_free_profiles_do_not_authorize_newton_mutation() -> None:
    profile = build_backend_profile("newton")
    now = datetime(2026, 8, 11, tzinfo=timezone.utc)
    admission = _admission(profile, now)
    admission["explicit_paid_run_authorization"] = False
    admission["admission_digest"] = canonical_digest(
        admission, digest_field="admission_digest"
    )

    assert validate_newton_canary_admission(
        admission, profile=profile, now=now
    ) == ["adp009d_newton_canary_admission_invalid"]
    assert profile["controls_only"] is True
    assert profile["policy_query_allowed"] is False


def test_complete_newton_admission_is_time_bounded_and_non_mutating() -> None:
    profile = build_backend_profile("newton")
    now = datetime(2026, 8, 11, tzinfo=timezone.utc)
    admission = _admission(profile, now)

    assert validate_newton_canary_admission(
        admission, profile=profile, now=now
    ) == []
    assert admission["provider_mutation_performed"] is False


@pytest.mark.parametrize(
    "issued_at",
    (
        "2026-08-11T15:59:59+00:00",
        "2026-08-11T16:30:06+00:00",
    ),
)
def test_newton_admission_rejects_stale_or_future_issuance(issued_at: str) -> None:
    now = datetime(2026, 8, 11, 16, 30, tzinfo=timezone.utc)
    admission = _admission(build_backend_profile("newton"), now)
    admission["issued_at"] = issued_at
    admission["admission_digest"] = canonical_digest(
        admission, digest_field="admission_digest"
    )

    assert validate_newton_canary_admission(
        admission, profile=build_backend_profile("newton"), now=now
    ) == ["adp009d_newton_canary_admission_invalid"]


def test_newton_admission_builder_binds_authority_spend_and_provider_zero() -> None:
    now = datetime(2026, 8, 11, 16, 40, tzinfo=timezone.utc)
    inventory = [
        {
            "provider": provider,
            "status": "succeeded",
            "required": True,
            "credential_present": True,
            "row_count": 0,
            "blockers": [],
        }
        for provider in ("runpod", "vast", "digitalocean")
    ]
    spend_lock = build_spend_admission_lock(
        fleet_budget={"status": "passed", "total_spend_usd": 0.0, "blockers": []},
        billing_reconciliation={
            "status": "reconciled",
            "required": True,
            "billing_export_schema_version": "blueprint.provider_billing_export.v1",
            "billing_export_sha256": "sha256:" + "a" * 64,
            "billing_export_mode_octal": "0600",
            "generated_at": now.isoformat(),
            "currency": "USD",
            "scope": "blueprint_beta_100_user_cohort",
            "provider_totals_usd": {
                "runpod": 98.0,
                "vast": 253.0,
                "digitalocean": 142.0,
            },
            "blockers": [],
        },
        instances=[],
        reap_results=[],
        inventory_results=inventory,
        override_path=None,
        now=now,
    )
    provider_zero = {
        "schema_version": "gpu_spend_guard.v1",
        "generated_at": now.isoformat(),
        "live_instance_count": 0,
        "blockers": [],
        "inventory_results": inventory,
        "instances": [],
    }

    admission = build_newton_canary_admission(
        authorization_evidence_ref="codex-thread:user-authorized-newton",
        spend_admission_lock=spend_lock,
        provider_zero_precheck=provider_zero,
        max_spend_usd=2.0,
        hard_ttl_seconds=7200,
        issued_at=now,
    )

    assert validate_newton_canary_admission(
        admission, profile=build_backend_profile("newton"), now=now
    ) == []
    assert admission["authorization_evidence_ref"] == (
        "codex-thread:user-authorized-newton"
    )
    assert admission["canonical_spend_admission_digest"].startswith("sha256:")
    assert admission["provider_zero_precheck_digest"].startswith("sha256:")


def test_newton_admission_builder_binds_exact_allowed_concurrency() -> None:
    now = datetime(2026, 8, 11, 17, 30, tzinfo=timezone.utc)
    allowed_id = 47_482_504
    inventory = [
        {
            "provider": provider,
            "status": "succeeded",
            "required": True,
            "credential_present": True,
            "row_count": 1 if provider == "vast" else 0,
            "blockers": [],
        }
        for provider in ("runpod", "vast", "digitalocean")
    ]
    spend_lock = build_spend_admission_lock(
        fleet_budget={"status": "passed", "total_spend_usd": 0.0, "blockers": []},
        billing_reconciliation={
            "status": "reconciled",
            "required": True,
            "billing_export_schema_version": "blueprint.provider_billing_export.v1",
            "billing_export_sha256": "sha256:" + "a" * 64,
            "billing_export_mode_octal": "0600",
            "generated_at": now.isoformat(),
            "currency": "USD",
            "scope": "blueprint_beta_100_user_cohort",
            "provider_totals_usd": {
                "runpod": 98.0,
                "vast": 253.0,
                "digitalocean": 142.0,
            },
            "blockers": [],
        },
        instances=[],
        reap_results=[],
        inventory_results=inventory,
        override_path=None,
        now=now,
    )
    provider_inventory = {
        "schema_version": "gpu_spend_guard.v1",
        "generated_at": now.isoformat(),
        "live_instance_count": 1,
        "blockers": [],
        "inventory_results": inventory,
        "instances": [
            {
                "provider": "vast",
                "id": str(allowed_id),
                "live": True,
            }
        ],
    }

    admission = build_newton_canary_admission(
        authorization_evidence_ref="codex-thread:user-authorized-second-newton-gpu",
        spend_admission_lock=spend_lock,
        provider_zero_precheck=provider_inventory,
        max_spend_usd=2.0,
        hard_ttl_seconds=7200,
        allowed_active_vast_instance_ids=[allowed_id],
        issued_at=now,
    )

    assert validate_newton_canary_admission(
        admission, profile=build_backend_profile("newton"), now=now
    ) == []
    assert admission["provider_inventory_precheck_mode"] == (
        "exact_allowed_concurrency"
    )
    assert admission["allowed_active_vast_instance_ids"] == [allowed_id]
    assert admission["provider_zero_precheck_passed"] is False
    assert admission["exact_concurrency_precheck_passed"] is True
    assert admission["unapproved_live_instance_count"] == 0


def test_comparison_receipt_blocks_identification_candidate_and_binding_drift() -> None:
    physx = _run("physx")
    newton = _run("newton")
    receipt = build_comparison_receipt(
        physx_run=physx,
        newton_run=newton,
        fidelity_result=_fidelity(),
    )

    assert receipt["schema_version"] == "adp009d_physics_backend_comparison.v1"
    assert receipt["status"] == "blocked"
    assert receipt["evidence_parity_observed"] is False
    assert receipt["promotion_review_eligible"] is False
    assert "adp009d_newton_gripper_drive_comparison_ineligible" in receipt["blockers"]
    assert receipt["engine_promotion_performed"] is False
    assert receipt["default_backend_after_comparison"] == "physx"
    assert receipt["policy_verdict"] is None
    assert receipt["backend_comparability_bindings"] == {
        "physx": physx["comparability_bindings"],
        "newton": newton["comparability_bindings"],
    }
    assert validate_comparison_receipt(receipt) == []

    newton["comparability_bindings"]["seed"] = 7
    newton["run_digest"] = canonical_digest(newton, digest_field="run_digest")
    blocked = build_comparison_receipt(
        physx_run=physx,
        newton_run=newton,
        fidelity_result=_fidelity(),
    )
    assert blocked["status"] == "blocked"
    assert blocked["promotion_review_eligible"] is False
    assert "adp009d_backend_comparison_bindings_differ" in blocked["blockers"]
    assert blocked["backend_comparability_bindings"] == {
        "physx": physx["comparability_bindings"],
        "newton": newton["comparability_bindings"],
    }
    assert validate_comparison_receipt(blocked) == []


@pytest.mark.parametrize("invalid_spend", [float("nan"), float("inf"), True])
def test_comparison_rejects_nonfinite_terminal_spend(
    invalid_spend: float | bool,
) -> None:
    newton = _run("newton")
    measurements = deepcopy(newton["measurements"])
    measurements["spend"]["total_usd"] = invalid_spend
    newton = build_backend_control_run_receipt(
        physics_backend="newton",
        comparability_bindings=newton["comparability_bindings"],
        backend_control_plan_digest=newton["backend_control_plan_digest"],
        measurements=measurements,
    )

    receipt = build_comparison_receipt(
        physx_run=_run("physx"),
        newton_run=newton,
        fidelity_result=_fidelity(),
    )

    assert receipt["status"] == "blocked"
    assert "adp009d_backend_control_run_terminal_evidence_invalid" in receipt[
        "blockers"
    ]
    assert validate_comparison_receipt(receipt) == []


def test_comparison_recomputes_fidelity_meaning_and_nested_receipts() -> None:
    fidelity = _fidelity()
    fidelity["meaningful_improvement_observed"] = False
    receipt = build_comparison_receipt(
        physx_run=_run("physx"),
        newton_run=_run("newton"),
        fidelity_result=fidelity,
    )

    assert receipt["status"] == "blocked"
    assert "adp009d_backend_comparison_fidelity_result_invalid" in receipt[
        "blockers"
    ]
    assert validate_comparison_receipt(receipt) == []

    receipt["backend_runs"]["newton"]["measurements"]["spend"][
        "total_usd"
    ] = 999.0
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    assert validate_comparison_receipt(receipt) == [
        "adp009d_backend_comparison_receipt_invalid"
    ]


def test_comparison_retains_typed_pre_controls_measurement_gaps() -> None:
    blocker = "canonical_hold_arm_pose_drift:maximum_error_rad=0.853033662"
    baseline_newton = _run("newton")
    measurements = deepcopy(baseline_newton["measurements"])
    gap_fields = (
        "contacts_and_force_vectors",
        "torque_utilization_and_clipping",
        "closest_geometric_clearance",
        "action_delivery",
        "phase_completion",
    )
    gaps = [
        {
            "field": field,
            "status": "not_reached",
            "typed_blocker": blocker,
        }
        for field in gap_fields
    ]
    for gap in gaps:
        measurements[gap["field"]] = {
            "status": "not_reached",
            "typed_blocker": blocker,
        }
    newton = build_backend_control_run_receipt(
        physics_backend="newton",
        comparability_bindings=baseline_newton["comparability_bindings"],
        backend_control_plan_digest=baseline_newton[
            "backend_control_plan_digest"
        ],
        measurements=measurements,
        blockers=[blocker],
        measurement_gaps=gaps,
    )

    receipt = build_comparison_receipt(
        physx_run=_run("physx"),
        newton_run=newton,
        fidelity_result={"status": "not_measured", "typed_blocker": blocker},
    )

    assert receipt["status"] == "blocked"
    assert receipt["evidence_parity_observed"] is False
    assert receipt["meaningful_improvement_observed"] is False
    assert receipt["promotion_review_eligible"] is False
    assert receipt["engine_promotion_performed"] is False
    assert receipt["policy_verdict"] is None
    assert receipt["backend_runs"]["newton"]["measurement_gaps"] == sorted(
        gaps, key=lambda item: item["field"]
    )
    assert (
        "adp009d_backend_comparison_backend_run_not_completed:newton"
        in receipt["blockers"]
    )
    assert "adp009d_backend_comparison_fidelity_not_measured" in receipt[
        "blockers"
    ]
    assert validate_comparison_receipt(receipt) == []


def test_comparison_rejects_untyped_or_terminal_measurement_gaps() -> None:
    blocker = "pre_controls_native_measurement_missing"
    newton = _run("newton")
    newton["status"] = "blocked"
    newton["blockers"] = [blocker]
    newton["measurement_gaps"] = [
        {
            "field": "spend",
            "status": "not_reached",
            "typed_blocker": blocker,
        }
    ]
    newton["run_digest"] = canonical_digest(newton, digest_field="run_digest")

    receipt = build_comparison_receipt(
        physx_run=_run("physx"),
        newton_run=newton,
        fidelity_result={"status": "not_measured", "typed_blocker": blocker},
    )

    assert "adp009d_backend_control_run_measurement_gaps_invalid" in receipt[
        "blockers"
    ]
    assert validate_comparison_receipt(receipt) == []


def test_disable_gravity_is_not_claimed_as_a_newton_mapped_property() -> None:
    """PhysX honours per-body gravity disable; the Newton MJCF has no equivalent.

    Measured on the sealed asset: PhysX held the canonical pose to 4.649e-06 rad
    because ``disable_gravity=True`` makes the arm weightless, while the Newton
    model mjwarp actually simulated carried the full 18.28 kg and drifted
    0.853033662 rad.  Listing the property as *mapped* asserted a semantic Newton
    does not provide, so the divergence surfaced as a hold failure instead of a
    typed non-comparability blocker.
    """

    assert (
        "physxRigidBody:disableGravity"
        not in NEWTON_MAPPED_PHYSX_PROPERTY_NAMES
    )
    assert (
        "physxRigidBody:disableGravity"
        in NEWTON_UNREPRESENTABLE_PHYSX_PROPERTY_NAMES
    )
    for property_name in NEWTON_UNREPRESENTABLE_PHYSX_PROPERTY_NAMES:
        assert property_name not in NEWTON_MAPPED_PHYSX_PROPERTY_NAMES
        assert not any(
            property_name.startswith(prefix)
            for prefix in NEWTON_MAPPED_PHYSX_PROPERTY_PREFIXES
        )


def test_unrepresentable_physx_property_fails_closed_before_a_paid_newton_run() -> None:
    """An authored dynamics-changing property Newton cannot express must block."""

    admission = validate_newton_dynamics_representable(
        [
            {
                "prim_path": "/World/template/Robot/proto_asset_0/panda_link4",
                "property_name": "physxRigidBody:disableGravity",
            }
        ]
    )
    assert admission["status"] == "blocked"
    assert admission["comparable_across_backends"] is False
    assert admission["typed_blocker"] == (
        "adp009d_newton_unrepresentable_physx_property:physxRigidBody:disableGravity"
    )
    assert admission["affected_prim_paths"] == [
        "/World/template/Robot/proto_asset_0/panda_link4"
    ]


def test_representable_physx_properties_admit_cleanly() -> None:
    """Properties Newton does express must not be turned into false blockers."""

    admission = validate_newton_dynamics_representable(
        [
            {
                "prim_path": "/World/template/Robot/proto_asset_0/panda_link4",
                "property_name": "physxJoint:maxJointVelocity",
            }
        ]
    )
    assert admission["status"] == "admitted"
    assert admission["comparable_across_backends"] is True
    assert admission["typed_blocker"] is None
    assert admission["affected_prim_paths"] == []


def _feasible_drive(**overrides: object) -> dict[str, object]:
    row = {
        "joint_name": "panda_joint4",
        "stiffness_nm_per_rad": 2400.0,
        "damping_nm_s_per_rad": 196.0,
        "effective_inertia_kg_m2": 1.0,
        "gravity_torque_nm": 20.070,
        "effort_limit_nm": 87.0,
    }
    row.update(overrides)
    return row


def test_explicit_pd_feasibility_admits_a_gain_that_can_actually_hold() -> None:
    """kp=2400 against 20.07 N*m droops 0.00836 rad, inside the 1.0e-2 gate."""

    receipt = validate_newton_explicit_pd_feasibility(
        joint_drives=[_feasible_drive()],
        timestep_seconds=1.0 / 120.0,
        hold_tolerance_rad=1.0e-2,
    )
    assert receipt["status"] == "admitted"
    assert receipt["typed_blockers"] == []
    joint = receipt["joints"][0]
    assert joint["steady_state_droop_rad"] == pytest.approx(0.0083625, abs=1e-7)
    assert joint["explicit_stability_ratio"] < 2.0


def test_explicit_pd_feasibility_rejects_a_gate_unreachable_by_arithmetic() -> None:
    """The shipped kp=400 droops 0.0502 rad -- five times its own hold gate.

    This is the exact configuration the ninth paid Newton canary launched with.
    It could not have passed the canonical hold gate at any solver setting, so it
    must fail closed before a provider allocation rather than after.
    """

    receipt = validate_newton_explicit_pd_feasibility(
        joint_drives=[_feasible_drive(stiffness_nm_per_rad=400.0, damping_nm_s_per_rad=80.0)],
        timestep_seconds=1.0 / 120.0,
        hold_tolerance_rad=1.0e-2,
    )
    assert receipt["status"] == "blocked"
    assert (
        "adp009d_newton_hold_gate_unreachable_by_droop:panda_joint4"
        in receipt["typed_blockers"]
    )
    assert receipt["joints"][0]["steady_state_droop_rad"] == pytest.approx(
        0.050175, abs=1e-6
    )


def test_explicit_pd_feasibility_rejects_an_explicitly_unstable_drive() -> None:
    """The Robotiq drive: kp=5729.58 on a 3.8e-07 kg*m^2 knuckle.

    omega = sqrt(kp/M) ~ 1.2e+05 rad/s, so omega*dt at 1/120 s is ~1.0e+03 against
    an explicit-integration limit of 2.  PhysX solves this drive implicitly and is
    unconditionally stable; Newton realises it as an explicit PD force and
    diverges, which is the NaN the earlier canary recorded.
    """

    receipt = validate_newton_explicit_pd_feasibility(
        joint_drives=[
            _feasible_drive(
                joint_name="finger_joint",
                stiffness_nm_per_rad=5729.58,
                damping_nm_s_per_rad=0.0114592,
                effective_inertia_kg_m2=3.80173e-07,
                gravity_torque_nm=0.0,
                effort_limit_nm=16.5,
            )
        ],
        timestep_seconds=1.0 / 120.0,
        hold_tolerance_rad=1.0e-2,
    )
    assert receipt["status"] == "blocked"
    assert (
        "adp009d_newton_explicit_pd_unstable:finger_joint"
        in receipt["typed_blockers"]
    )
    assert receipt["joints"][0]["explicit_stability_ratio"] > 100.0


def test_explicit_pd_feasibility_rejects_a_drive_that_cannot_reach_its_load() -> None:
    """A hold torque above the effort limit can never be delivered."""

    receipt = validate_newton_explicit_pd_feasibility(
        joint_drives=[_feasible_drive(gravity_torque_nm=120.0)],
        timestep_seconds=1.0 / 120.0,
        hold_tolerance_rad=1.0e-2,
    )
    assert receipt["status"] == "blocked"
    assert (
        "adp009d_newton_hold_torque_exceeds_effort_limit:panda_joint4"
        in receipt["typed_blockers"]
    )


def _gravity_real_receipt(**overrides: object) -> dict[str, object]:
    contract = build_gravity_real_actuation_contract()
    receipt = {
        "contract_digest": contract["contract_digest"],
        "robot_disable_gravity": False,
        "source_asset_mutated": False,
        "observed_arm_gains": {
            group: {"stiffness": 2400.0, "damping": 196.0}
            for group in GRAVITY_REAL_ARM_ACTUATOR_GROUPS
        },
    }
    receipt.update(overrides)
    return receipt


def test_gravity_real_contract_applies_to_both_backends_and_voids_prior_evidence() -> None:
    """The arm carries its weight in both lanes; weightless evidence does not carry."""

    contract = build_gravity_real_actuation_contract()
    assert contract["applies_to_backends"] == ["physx", "newton"]
    assert contract["robot_disable_gravity"] is False
    assert contract["source_asset_disable_gravity"] is True
    assert contract["source_asset_mutated"] is False
    assert contract["prior_weightless_evidence_carries_over"] is False
    assert contract["independent_fidelity_claimed"] is False


def test_gravity_real_stiffness_makes_the_hold_gate_reachable() -> None:
    """kp=2400 must put worst-joint droop inside the gate it is judged against."""

    contract = build_gravity_real_actuation_contract()
    assert contract["arm_stiffness_nm_per_rad"] == 2400.0
    assert contract["predicted_worst_joint"] == "panda_joint4"
    assert contract["predicted_worst_droop_rad"] == pytest.approx(0.0083625, abs=1e-7)
    assert contract["predicted_worst_droop_rad"] < contract["hold_tolerance_rad"]
    # the superseded gain could not, which is why it is recorded alongside
    superseded = (
        abs(contract["hold_torque_nm"]["panda_joint4"])
        / contract["superseded_arm_stiffness_nm_per_rad"]
    )
    assert superseded > contract["hold_tolerance_rad"]


def test_gravity_real_gains_are_feasible_under_the_explicit_pd_gate() -> None:
    """The chosen gain must survive the same feasibility gate Newton is held to."""

    contract = build_gravity_real_actuation_contract()
    receipt = validate_newton_explicit_pd_feasibility(
        joint_drives=[
            {
                "joint_name": "panda_joint4",
                "stiffness_nm_per_rad": contract["arm_stiffness_nm_per_rad"],
                "damping_nm_s_per_rad": contract["arm_damping_nm_s_per_rad"],
                "effective_inertia_kg_m2": 1.0,
                "gravity_torque_nm": contract["hold_torque_nm"]["panda_joint4"],
                "effort_limit_nm": 87.0,
            }
        ],
        timestep_seconds=1.0 / 120.0,
        hold_tolerance_rad=contract["hold_tolerance_rad"],
    )
    assert receipt["status"] == "admitted"


def test_gravity_real_validation_rejects_a_still_weightless_run() -> None:
    validation = validate_gravity_real_actuation(
        _gravity_real_receipt(robot_disable_gravity=True)
    )
    assert validation["status"] == "blocked"
    assert "adp009d_gravity_real_robot_still_weightless" in validation["typed_blockers"]


def test_gravity_real_validation_rejects_the_superseded_stiffness() -> None:
    validation = validate_gravity_real_actuation(
        _gravity_real_receipt(
            observed_arm_gains={
                group: {"stiffness": 400.0, "damping": 80.0}
                for group in GRAVITY_REAL_ARM_ACTUATOR_GROUPS
            }
        )
    )
    assert validation["status"] == "blocked"
    assert any(
        blocker.startswith("adp009d_gravity_real_stiffness_invalid")
        for blocker in validation["typed_blockers"]
    )


def test_gravity_real_validation_accepts_a_correctly_applied_run() -> None:
    validation = validate_gravity_real_actuation(_gravity_real_receipt())
    assert validation["status"] == "validated"
    assert validation["typed_blockers"] == []


def test_disabled_gravity_authored_false_is_representable() -> None:
    """After the gravity-real change the property is authored as False.

    ``disableGravity=False`` is the backend-neutral default: it means normal
    gravity, which Newton does express. Blocking on the property being *authored*
    rather than on it being *active* would make every gravity-real Newton run
    impossible.
    """

    admission = validate_newton_dynamics_representable(
        [
            {
                "prim_path": "/World/template/Robot/proto_asset_0/panda_link4",
                "property_name": "physxRigidBody:disableGravity",
                "value": False,
            }
        ]
    )
    assert admission["status"] == "admitted"
    assert admission["comparable_across_backends"] is True
    assert admission["affected_prim_paths"] == []


def test_disabled_gravity_authored_true_still_fails_closed() -> None:
    admission = validate_newton_dynamics_representable(
        [
            {
                "prim_path": "/World/template/Robot/proto_asset_0/panda_link4",
                "property_name": "physxRigidBody:disableGravity",
                "value": True,
            }
        ]
    )
    assert admission["status"] == "blocked"
    assert admission["typed_blocker"] == (
        "adp009d_newton_unrepresentable_physx_property:physxRigidBody:disableGravity"
    )


def test_unrepresentable_property_without_a_value_fails_closed() -> None:
    """An unreadable value must not be treated as harmless."""

    admission = validate_newton_dynamics_representable(
        [
            {
                "prim_path": "/World/template/Robot/proto_asset_0/panda_link4",
                "property_name": "physxRigidBody:disableGravity",
            }
        ]
    )
    assert admission["status"] == "blocked"


def test_restore_command_integrates_so_the_achieved_pose_reaches_the_target() -> None:
    """The episode-start replay must converge the ACHIEVED pose, not the command.

    Deriving the command from the achieved pose each step gives the fixed point
    command==target, achieved==target-droop: under gravity the replay lands a
    full droop short of the pose it is replaying, and the early exit at
    tolerance/3 can never trigger. Integrating the command instead makes
    achieved==target the fixed point, with the command carrying the droop.
    """

    target, droop, max_step = 1.0, 0.0083, 0.01
    commanded = 0.0
    for _ in range(400):
        achieved = commanded - droop
        commanded = next_episode_start_restore_command(
            commanded_joint_position_rad=commanded,
            achieved_joint_position_rad=achieved,
            target_joint_position_rad=target,
            max_joint_step_rad=max_step,
        )
    achieved = commanded - droop
    assert abs(achieved - target) <= 1.0e-9
    assert commanded == pytest.approx(target + droop, abs=1.0e-9)


def test_restore_command_respects_the_step_clamp() -> None:
    assert next_episode_start_restore_command(
        commanded_joint_position_rad=0.0,
        achieved_joint_position_rad=0.0,
        target_joint_position_rad=5.0,
        max_joint_step_rad=0.01,
    ) == pytest.approx(0.01)
    assert next_episode_start_restore_command(
        commanded_joint_position_rad=0.0,
        achieved_joint_position_rad=0.0,
        target_joint_position_rad=-5.0,
        max_joint_step_rad=0.01,
    ) == pytest.approx(-0.01)


def test_restore_command_is_a_fixed_point_once_achieved() -> None:
    """No creep once the achieved pose is already on target."""

    assert next_episode_start_restore_command(
        commanded_joint_position_rad=1.0083,
        achieved_joint_position_rad=1.0,
        target_joint_position_rad=1.0,
        max_joint_step_rad=0.01,
    ) == pytest.approx(1.0083)
