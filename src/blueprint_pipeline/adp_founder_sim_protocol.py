"""Founder-approved simulation-only precursor for Arm Decision Proof v1.

The protocol deliberately replaces partner admission and physical holdout gates
with one Blueprint founder approval while the claim remains development-only
simulation.  It does not weaken the two-candidate identity, power, denominator,
visual evidence, or independent simulator-state grading contracts.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_prospective_design import (
    INVALID_TRIAL_RULE,
    MULTIPLICITY_RULE,
    POWER_METHOD,
    STOP_RULE,
    UNCERTAINTY_METHOD,
    compile_trial_schedule,
    validate_schedule_for_execution,
)
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .episode_visual_evidence import (
    FRAME_MANIFEST_SCHEMA_VERSION,
    VISUAL_EVIDENCE_SCHEMA_VERSION,
)
from .franka_droid_closed_loop import SCHEMA_VERSION as FRANKA_RUNNER_SCHEMA_VERSION
from .franka_droid_control_preflight import SCHEMA_VERSION as CONTROL_PREFLIGHT_SCHEMA_VERSION
from .groot_n17_droid_policy_runtime import GrootN17DroidPolicySpec


PROTOCOL_SCHEMA_VERSION = "adp_founder_sim_only_protocol.v1"
APPROVAL_SCHEMA_VERSION = "adp_founder_sim_only_approval.v1"
EXECUTION_ADMISSION_SCHEMA_VERSION = "adp_founder_sim_execution_admission.v1"
PROTOCOL_ID = "adp-founder-sim-arena-droid-pi05-vs-groot-n17-v1"
BASELINE_ID = "pi05_droid_jointpos_polaris"
ALTERNATIVE_ID = "nvidia/GR00T-N1.7-DROID"


class FounderSimProtocolError(ValueError):
    def __init__(self, blockers: Sequence[str]):
        self.blockers = tuple(sorted(set(str(item) for item in blockers if str(item))))
        super().__init__(";".join(self.blockers))


def _reset_specs() -> list[dict[str, Any]]:
    rows = [
        {
            "condition_id": "arena_maple_table_baseline",
            "environment": "pick_and_place_maple_table",
            "embodiment": "droid_abs_joint_pos",
            "pick_up_object": "rubiks_cube_hot3d_robolab",
            "destination_location": "bowl_ycb_robolab",
            "hdr": "home_office_robolab",
            "light_intensity": 500.0,
            "placement": "pinned Arena On(table) relation solver controlled by row seed",
            "additional_table_objects": [],
            "variations_enabled": [],
        }
    ]
    for row in rows:
        row["reset_digest"] = canonical_digest(row, digest_field="reset_digest")
    return rows


def _statistical_design() -> dict[str, Any]:
    return {
        "method": POWER_METHOD,
        "planning_variance_rate": 0.5,
        "minimum_decision_relevant_difference": 0.30,
        "alpha": 0.05,
        "power": 0.80,
        "uncertainty_method": UNCERTAINTY_METHOD,
        "invalid_trial_handling": INVALID_TRIAL_RULE,
        "stop_rule": STOP_RULE,
        "multiplicity": MULTIPLICITY_RULE,
    }


def build_founder_sim_protocol() -> dict[str, Any]:
    resets = _reset_specs()
    schedule = compile_trial_schedule(
        candidate_pair={
            "baseline_candidate_id": BASELINE_ID,
            "alternative_candidate_id": ALTERNATIVE_ID,
        },
        conditions=[
            {
                "condition_id": row["condition_id"],
                "reset_digest": row["reset_digest"],
            }
            for row in resets
        ],
        statistical_design=_statistical_design(),
        randomization_seed=20260804,
        seed_start=41000,
    )
    schedule_admission = validate_schedule_for_execution(schedule)
    groot_identity = GrootN17DroidPolicySpec().identity()
    protocol: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_pending_founder_digest_approval",
        "program": "arm-decision-proof-v1",
        "phase": "development_only_simulation_precursor",
        "authority_model": {
            "required_approvers": ["blueprint_founder_sim_owner"],
            "partner_task_owner_required": False,
            "physical_holdout_custodian_required": False,
            "reason": "founder-scoped simulation only; no partner or physical outcome exists",
        },
        "decision": {
            "question": (
                "Which of two frozen DROID-compatible Franka policies earns the next "
                "simulation engineering budget on the Arena Rubik's-cube-to-bowl task?"
            ),
            "baseline_candidate_id": BASELINE_ID,
            "alternative_candidate_id": ALTERNATIVE_ID,
            "real_or_physical_allocation_claimed": False,
        },
        "task": {
            "task_id": "isaac_lab_arena_droid_rubiks_cube_to_bowl_v1",
            "instruction": "Pick up the Rubik's cube and place it in the bowl.",
            "family": "fixed_arm_rigid_object_pick_and_place",
            "robot": "Franka Panda",
            "gripper": "Robotiq 2F-85 in the DROID embodiment",
            "success_evaluator": {
                "owner": "simulator_not_candidate_policy",
                "grader_type": "deterministic_simulator_state",
                "primary_metric": "binary_task_success",
                "predicates": [
                    "Arena PickAndPlaceTask object_reaches_destination_region",
                    "Arena task success termination fires",
                ],
                "arena_metric": "SuccessRateMetric",
                "policy_self_report_used": False,
            },
            "termination": {
                "maximum_action_steps": 600,
                "outer_control_hz": 15,
                "open_loop_actions_per_query": 8,
                "episode_length_seconds": 20.0,
            },
        },
        "scene": {
            "scene_id": "isaac_lab_arena_pick_and_place_maple_table_v1",
            "production_execution_backend": "native_isaac_lab_arena_on_isaac_sim_physx",
            "local_control_backend": "mujoco_franka_proxy_not_candidate_evidence",
            "simulator_stack": {
                "isaac_lab_arena": {
                    "repository": "isaac-sim/IsaacLab-Arena",
                    "revision": "3c19a3a9e45fc2cc1b64ab8a43047ecac9c0ad4d",
                    "version_at_revision": "0.3.0-alpha-development",
                    "uv_lock_sha256": (
                        "35001404fa10d3f591d326d7a36b15c0b35cf307b754edea87310f719ec439da"
                    ),
                    "stability": "alpha_exact_revision_requires_blueprint_canary",
                },
                "isaac_lab_revision": "af1bab4dc173ba69b08fab779c14ead61d13fd33",
                "isaac_sim_version": "6.0.0.1",
                "physics_backend": "PhysX",
                "renderer": "Isaac RTX",
                "environment_type": "ManagerBasedRLEnv",
                "arena_embodiment": "droid_abs_joint_pos",
                "arena_droid_contract_git_blobs": {
                    "actions.py": "c9f68b7b1b91077e44f72feb2a78e39b9945ecfb",
                    "droid.py": "42085976e5b4004eeadae7f18973cc21a8452332",
                    "observations.py": "7de4229abdce5a3aebb752d34680ed17fd4e560d",
                },
                "arena_environment": "pick_and_place_maple_table",
                "blueprint_integration": (
                    "thin receipt and schedule adapter around Arena's existing environment "
                    "and remote policies; no Arena fork or custom environment"
                ),
            },
            "arena_registry_assets": {
                "background": "maple_table_robolab",
                "pick_up_object": "rubiks_cube_hot3d_robolab",
                "destination": "bowl_ycb_robolab",
                "hdr": "home_office_robolab",
                "asset_byte_manifest": "required_from_materialized_worker_receipt",
            },
            "interaction_assets": {
                "production_robot_source": {
                    "source": "Isaac Lab-Arena droid_abs_joint_pos embodiment",
                    "robot": "Franka Panda",
                    "gripper": "Robotiq 2F-85",
                    "action_space": "absolute_joint_position",
                },
                "local_control_robot_source": {
                    "repository": "google-deepmind/mujoco_menagerie",
                    "revision": "71f066ad0be9cd271f7ed58c030243ef157af9f4",
                    "asset_path": "franka_emika_panda",
                    "claim_ceiling": "local_control_oracle_only",
                },
                "task_object": "Arena registered Rubik's cube rigid object",
                "destination": "Arena registered YCB bowl",
                "isaac_physics_required_for_candidate_trials": True,
            },
            "scenario_variation_policy": {
                "scene_cousins_in_this_protocol": False,
                "frozen_factors": [
                    "Arena relation-solver object placement sampled only from row seed"
                ],
                "unfrozen_factors_forbidden": [
                    "object_identity",
                    "background_hdr",
                    "lighting",
                    "camera_intrinsics",
                    "camera_extrinsics",
                    "friction",
                    "mass",
                ],
                "future_cousin_backend": "Isaac Lab-Arena variation system",
                "future_cousin_requires_new_protocol_digest": True,
                "agentic_generation_role": "proposal_only_before_protocol_freeze",
                "post_approval_scenario_mutation_forbidden": True,
            },
            "simready_asset_generation_required": False,
            "asset_gap_disposition": (
                "Do not invoke SimReady generation unless materialization fails or a "
                "future task requires an object absent from the pinned Arena registry."
            ),
        },
        "candidates": [
            {
                "role": "baseline",
                "candidate_id": BASELINE_ID,
                "family": "Physical Intelligence pi0.5",
                "runtime": ("identity-bound OpenPI worker through the pinned Arena DROID adapter"),
                "checkpoint_uri": (
                    "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris"
                ),
                "checkpoint_generation_manifest_sha256": (
                    "385e9e318d92062e60e2d055296bb90b1ee8db0e0ab5233442382a7a861f81d6"
                ),
                "checkpoint_inventory_sha256": (
                    "492ef95fa2e0ea8c026fda4bf6a2662758e7958ab5223ecb270cde5bc3797063"
                ),
                "openpi_revision": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
                "action_chunk_shape": [15, 8],
            },
            {
                "role": "alternative",
                "candidate_id": ALTERNATIVE_ID,
                "family": "NVIDIA Isaac GR00T N1.7",
                "runtime": (
                    "NVIDIA PolicyClient ZMQ through Blueprint's identity-bound N1.7 "
                    "adapter and the pinned Arena DROID environment"
                ),
                "identity": groot_identity,
                "checkpoint_files_sha256": "required_from_materialized_worker_receipt",
                "environment_lock_sha256": "required_from_materialized_worker_receipt",
                "action_chunk_shape": ["checkpoint_reported_horizon_at_least_8", 8],
            },
        ],
        "shared_interface": {
            "observations": [
                "external_rgb",
                "wrist_rgb",
                "franka_joint_position_rad_7",
                "gripper_position_0_open_1_closed",
                "droid_eef_xyz_corrected_rotation6d",
                "language_instruction",
            ],
            "environment_actions": "absolute_joint_position_rad_7_plus_gripper_position",
            "candidate_specific_translation": {
                BASELINE_ID: "pinned Arena OpenPI DROID adapter plus Blueprint identity receipt",
                ALTERNATIVE_ID: "Blueprint NVIDIA N1.7 DROID ZMQ adapter",
            },
            "same_controller_reset_cameras_termination_and_scorer": True,
        },
        "conditions": resets,
        "secondary_metrics": [],
        "statistical_design": _statistical_design(),
        "schedule": schedule,
        "schedule_admission": schedule_admission,
        "visual_evidence": {
            "lossless_policy_input_pngs_required": True,
            "terminal_png_required": True,
            "digest_bound_frame_manifest_required": True,
            "human_review_mp4_required": True,
            "independent_grader_provenance_required": True,
            "missing_media_disposition": "trial_invalid_and_retained_as_failure",
        },
        "harness_contracts": {
            "franka_runner_schema_version": FRANKA_RUNNER_SCHEMA_VERSION,
            "control_preflight_schema_version": CONTROL_PREFLIGHT_SCHEMA_VERSION,
            "frame_manifest_schema_version": FRAME_MANIFEST_SCHEMA_VERSION,
            "visual_evidence_schema_version": VISUAL_EVIDENCE_SCHEMA_VERSION,
            "arena_worker_request_schema_version": ("adp_isaac_lab_arena_worker_request.v1"),
            "execution_schedule_admission_required": True,
            "episode_evidence_admission_required": True,
        },
        "execution_preconditions": [
            "founder_approval_receipt_quotes_exact_protocol_digest",
            "both_checkpoint_byte_manifests_verified_on_worker",
            "exact_arena_registry_asset_bytes_and_native_runtime_lock_verified",
            "local_mujoco_scripted_positive_and_stationary_negative_preflight_passes",
            "native_arena_zero_action_negative_control_fails_task_success",
            "native_arena_replay_or_scripted_positive_control_passes_task_success",
            "native_arena_droid_camera_action_and_reset_parity_canary_passes",
            "every_episode_media_writer_passes pre-production dry-run",
            "canonical_paid_resource_allocator_price_cap_ttl_watchdog_and_provider_zero",
        ],
        "amendment_policy": (
            "Any protocol, candidate, scene, reset, metric, schedule, or evidence change "
            "creates a new digest and requires new founder approval before execution."
        ),
        "claim_boundary": {
            "development_only": True,
            "simulator_only": True,
            "partner_admission": False,
            "physical_holdout": False,
            "sim_to_real_accuracy": False,
            "deployment_or_safety_claim": False,
            "purpose": "exercise Blueprint two-candidate decision harness before IRL",
        },
        "execution_state": {
            "capture_started": False,
            "production_simulation_started": False,
            "physical_trial_started": False,
            "paid_compute_authorized_by_this_artifact": False,
        },
    }
    protocol["protocol_digest"] = canonical_digest(protocol, digest_field="protocol_digest")
    return protocol


def admit_founder_sim_execution(
    protocol: Mapping[str, Any], approval: Mapping[str, Any]
) -> dict[str, Any]:
    blockers: list[str] = []
    expected = build_founder_sim_protocol()
    if dict(protocol) != expected:
        blockers.append("founder_sim_protocol_not_canonical")
    if approval.get("schema_version") != APPROVAL_SCHEMA_VERSION:
        blockers.append("founder_sim_approval_schema_invalid")
    if approval.get("approved") is not True:
        blockers.append("founder_sim_approval_missing")
    if approval.get("approver_role") != "blueprint_founder_sim_owner":
        blockers.append("founder_sim_approver_role_invalid")
    if approval.get("protocol_id") != PROTOCOL_ID:
        blockers.append("founder_sim_approval_protocol_id_mismatch")
    if approval.get("protocol_digest") != expected["protocol_digest"]:
        blockers.append("founder_sim_approval_protocol_digest_mismatch")
    if blockers:
        raise FounderSimProtocolError(blockers)
    result = {
        "schema_version": EXECUTION_ADMISSION_SCHEMA_VERSION,
        "status": "founder_approved_pending_runtime_and_paid_resource_preconditions",
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": expected["protocol_digest"],
        "schedule_digest": expected["schedule"]["schedule_digest"],
        "approval": dict(approval),
        "simulation_only": True,
        "physical_execution_authorized": False,
        "paid_compute_authorized": False,
    }
    result["execution_admission_digest"] = canonical_digest(
        result, digest_field="execution_admission_digest"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    protocol = build_founder_sim_protocol()
    if args.output:
        write_json(Path(args.output).expanduser().resolve(), protocol)
    else:
        print(json.dumps(protocol, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
