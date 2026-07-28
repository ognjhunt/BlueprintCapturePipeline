"""Fail-closed contracts for the RoboArena whole-stack calibration campaign.

This module deliberately separates a benchmark reproduction from a prospective
captured-site claim.  OSCAR and Cosmos are parallel WAM arms; neither backend
may consume the other's generated output.  Closed-loop ``back and forth`` is
between a candidate policy and one WAM arm through a short executed prefix.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_ID = "policy_ranking_roboarena_full_stack_calibration_20260728"
SCHEMA_VERSION = "policy_ranking_roboarena_full_stack_calibration.v2"
ACTION_CONTROL_SCHEMA_VERSION = "droid_action_controls.v2"

DROID_ACTION_DIM = 10
DROID_ACTION_CHUNK = 16
DROID_FREQUENCY_HZ = 15.0
IDENTITY_ROT6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
SUPERSEDED_PROTOCOL_SHA256 = "eab9e7868bcc7cbd774c940c781e8c3a8faac3270cbc942f1248966ba037f683"
PREFIX_PILOT_CANDIDATE_STEPS = (4, 8, 16)
EXECUTED_PREFIX_STEPS = 16
EXECUTED_PREFIX_SECONDS_DERIVED = EXECUTED_PREFIX_STEPS / DROID_FREQUENCY_HZ
ABSTENTION_ADJACENT_RISK_TOLERANCE = 0.02
PHASE_A_MODEL = "gpt-5-mini-2025-08-07"
PHASE_A_FRAME_COUNT = 32


class CalibrationContractError(ValueError):
    """The full-stack calibration protocol or an action control is invalid."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_matrix(value: Any) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CalibrationContractError("actions_missing_or_not_sequence")
    if len(value) != DROID_ACTION_CHUNK:
        raise CalibrationContractError("action_chunk_must_have_16_rows")
    matrix: list[list[float]] = []
    for index, raw_row in enumerate(value):
        if not isinstance(raw_row, Sequence) or isinstance(
            raw_row, (str, bytes, bytearray)
        ):
            raise CalibrationContractError(f"action_row_not_sequence:{index}")
        if len(raw_row) != DROID_ACTION_DIM:
            raise CalibrationContractError(f"action_row_must_have_10_values:{index}")
        try:
            row = [float(item) for item in raw_row]
        except (TypeError, ValueError) as exc:
            raise CalibrationContractError(f"action_row_not_numeric:{index}") from exc
        if not all(math.isfinite(item) for item in row):
            raise CalibrationContractError(f"action_row_not_finite:{index}")
        if not 0.0 <= row[-1] <= 1.0:
            raise CalibrationContractError(f"gripper_outside_unit_interval:{index}")
        first = row[3:6]
        second = row[6:9]
        first_norm = math.sqrt(sum(item * item for item in first))
        second_norm = math.sqrt(sum(item * item for item in second))
        dot = sum(left * right for left, right in zip(first, second, strict=True))
        if (
            not math.isclose(first_norm, 1.0, rel_tol=0.0, abs_tol=2e-3)
            or not math.isclose(second_norm, 1.0, rel_tol=0.0, abs_tol=2e-3)
            or not math.isclose(dot, 0.0, rel_tol=0.0, abs_tol=2e-3)
        ):
            raise CalibrationContractError(f"rot6d_columns_not_orthonormal:{index}")
        matrix.append(row)
    return matrix


def no_motion_action_chunk(*, gripper_hold_value: float) -> list[list[float]]:
    """Return a valid delta-pose no-motion control.

    Six literal zero rotation values are not a valid rot6d identity.  A neutral
    gripper command must also be explicit; zero cannot be assumed to mean hold.
    """

    hold = float(gripper_hold_value)
    if not math.isfinite(hold) or not 0.0 <= hold <= 1.0:
        raise CalibrationContractError("gripper_hold_outside_unit_interval")
    row = [0.0, 0.0, 0.0, *IDENTITY_ROT6D, hold]
    return [list(row) for _ in range(DROID_ACTION_CHUNK)]


def _real_trace(value: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    trace_kind = str(value.get("trace_kind") or "").strip()
    if trace_kind not in {"recorded_real_candidate_policy", "live_frozen_policy_endpoint"}:
        raise CalibrationContractError(f"{role}_must_be_real_candidate_policy_trace")
    trace_id = str(value.get("source_trace_id") or "").strip()
    if not trace_id:
        raise CalibrationContractError(f"{role}_source_trace_id_missing")
    return {
        "trace_kind": trace_kind,
        "source_trace_id": trace_id,
        "actions": _action_matrix(value.get("actions")),
    }


def build_action_controls_v2(
    recorded: Mapping[str, Any],
    policy_swapped: Mapping[str, Any],
    *,
    gripper_hold_value: float,
    shuffle_seed: int,
    temporal_shift_steps: int = 1,
) -> dict[str, Any]:
    """Build valid controls from two independently sourced candidate traces."""

    source = _real_trace(recorded, role="recorded")
    swapped = _real_trace(policy_swapped, role="policy_swapped")
    if source["source_trace_id"] == swapped["source_trace_id"]:
        raise CalibrationContractError("policy_swapped_trace_must_be_distinct")
    if canonical_sha256(source["actions"]) == canonical_sha256(swapped["actions"]):
        raise CalibrationContractError("policy_swapped_actions_must_be_distinct")

    order = list(range(DROID_ACTION_CHUNK))
    random.Random(int(shuffle_seed)).shuffle(order)
    if order in (list(range(DROID_ACTION_CHUNK)), list(reversed(range(DROID_ACTION_CHUNK)))):
        order = order[1:] + order[:1]
    if isinstance(temporal_shift_steps, bool) or not isinstance(temporal_shift_steps, int):
        raise CalibrationContractError("temporal_shift_steps_must_be_integer")
    if not 0 < temporal_shift_steps < DROID_ACTION_CHUNK:
        raise CalibrationContractError("temporal_shift_steps_outside_chunk")

    conditions = {
        "recorded": source["actions"],
        "no_motion": no_motion_action_chunk(gripper_hold_value=gripper_hold_value),
        "shuffled": [source["actions"][index] for index in order],
        "reversed": list(reversed(source["actions"])),
        "temporally_shifted": (
            source["actions"][temporal_shift_steps:]
            + source["actions"][:temporal_shift_steps]
        ),
        "policy_swapped": swapped["actions"],
    }
    hashes = {name: canonical_sha256(actions) for name, actions in conditions.items()}
    if len(set(hashes.values())) != len(hashes):
        raise CalibrationContractError("action_controls_not_pairwise_distinct")
    return {
        "schema_version": ACTION_CONTROL_SCHEMA_VERSION,
        "conditions": conditions,
        "action_sha256_by_condition": hashes,
        "recorded_source_trace_id": source["source_trace_id"],
        "policy_swapped_source_trace_id": swapped["source_trace_id"],
        "no_motion_rotation": "rot6d_identity",
        "no_motion_gripper": "explicit_hold_value",
        "temporal_shift_steps": temporal_shift_steps,
        "synthetic_policy_swapped_forbidden": True,
    }


def preregistered_protocol() -> dict[str, Any]:
    """Return the frozen scientific design for the next calibration campaign."""

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "question": (
            "Can Blueprint reproduce useful RoboArena/DROID candidate-policy ordering "
            "with its complete evaluation, ranking, uncertainty, and abstention service "
            "before any captured-site transfer claim?"
        ),
        "phases": [
            {
                "phase": "A_public_known_answer_reproduction",
                "purpose": "positive-control reproduction, not independent confirmation",
                "inputs": "published full OSCAR and real-robot RoboArena/DROID episodes",
                "wam_arms": ["oscar_wam"],
                "full_episode_required": True,
                "short_chunk_only_result_forbidden": True,
                "candidate_policy_identity_hidden_from_evaluator": True,
                "benchmark_outcomes_hidden_until_predictions_frozen": True,
            },
            {
                "phase": "B_independent_closed_loop_confirmation",
                "purpose": "independent full-service qualification",
                "admission": "phase_A_endpoint_gates_passed_and_new_disjoint_snapshot_available",
                "inputs": "new disjoint sessions plus runnable frozen candidate-policy endpoints",
                "wam_arms": [
                    "oscar_skeleton_only",
                    "oscar_wam",
                    "cosmos3_nano_native_forward_dynamics",
                    "cosmos3_oscar_skeleton_hybrid_optional_registered",
                ],
                "full_episode_required": True,
                "policy_wam_receding_horizon": True,
                "executed_prefix_steps": EXECUTED_PREFIX_STEPS,
                "executed_prefix_seconds_derived": EXECUTED_PREFIX_SECONDS_DERIVED,
                "prefix_governing_variable": "executed_prefix_steps",
                "stop_conditions": [
                    "task_terminal",
                    "safety_abstention",
                    "collapse_abstention",
                    "maximum_horizon",
                ],
            },
            {
                "phase": "C_captured_site_transfer",
                "purpose": "prospective captured-site policy ranking",
                "admission": "phase_B_all_gates_passed",
                "physical_outcomes_required_for_accuracy_claim": True,
            },
        ],
        "backend_graph": {
            "shared_inputs": [
                "initial_observation",
                "task_instruction",
                "frozen_candidate_policy_action_or_endpoint",
            ],
            "oscar_wam": {
                "consumes": ["shared_inputs"],
                "produces": ["oscar_rollout", "oscar_uncertainty"],
            },
            "cosmos3_wam_parallel_diagnostic": {
                "consumes": ["shared_inputs"],
                "produces": ["cosmos3_rollout", "cosmos3_uncertainty"],
            },
            "independent_evaluator": {
                "model": "gpt-5-mini-2025-08-07",
                "consumes": ["one_attributable_wam_rollout", "task_rubric"],
                "cannot_consume": ["policy_identity", "benchmark_outcome", "other_wam_grade"],
            },
            "forbidden_edges": [
                "oscar_wam->cosmos3_wam_parallel_diagnostic",
                "cosmos3_wam_parallel_diagnostic->oscar_wam",
            ],
        },
        "methodology": {
            "roboworld_retained": [
                "task_progress_rubric",
                "full_episode_scoring",
                "aggregate_policy_ranking",
            ],
            "sc3_retained": [
                "counterfactual_controls",
                "temporal_consistency",
                "uncertainty",
                "calibration",
                "abstention",
            ],
            "oscar_retained": [
                "skeleton_conditioning",
                "public_roboarena_baseline",
                "compatibility_backend",
            ],
            "scene_masked_and_visible_skeleton_scores_reported_separately": True,
            "unpublished_methodology_labels": {
                "roboworld": "inspired_by",
                "sc3_eval": "inspired_by",
            },
        },
        "prefix_selection": {
            "candidate_steps": list(PREFIX_PILOT_CANDIDATE_STEPS),
            "selected_steps": EXECUTED_PREFIX_STEPS,
            "selected_seconds_derived": EXECUTED_PREFIX_SECONDS_DERIVED,
            "selection_mode": "upstream_contract_fallback_empirical_pilot_unavailable_pre_provider",
            "deterministic_rule": (
                "Select the smallest candidate that exactly matches the published native "
                "Cosmos DROID autoregressive chunk-advance contract; break ties toward fewer WAM calls."
            ),
            "cost_multipliers_per_16_executed_steps": {"4": 4.0, "8": 2.0, "16": 1.0},
            "outcome_labels_used": False,
        },
        "endpoint_gates": {
            "policy_count_minimum": 7,
            "spearman_rho_minimum": 0.70,
            "kendall_tau_b_minimum": 0.50,
            "pairwise_accuracy_minimum": 0.70,
            "pairwise_accuracy_clustered_ci95_lower_minimum": 0.50,
            "true_top_policy_in_predicted_top_two": True,
            "selective_coverage_minimum": 0.50,
            "selective_pairwise_accuracy_minimum": 0.75,
            "abstention_risk_rule": {
                "full_empirical_curve_required": True,
                "isotonic_smoothed_diagnostic_required": True,
                "session_clustered_bootstrap_ci95_required": True,
                "adjacent_risk_increase_tolerance": ABSTENTION_ADJACENT_RISK_TOLERANCE,
                "ties_or_ci_level_noise_fail": False,
                "material_supported_increase_fails": True,
                "material_supported_increase_definition": (
                    "adjacent empirical risk increase exceeds tolerance and its session-clustered "
                    "bootstrap 95% confidence interval lower bound exceeds zero"
                ),
            },
            "all_gates_required": True,
        },
        "action_controls": {
            "schema_version": ACTION_CONTROL_SCHEMA_VERSION,
            "no_motion_translation": [0.0, 0.0, 0.0],
            "no_motion_rotation_rot6d": list(IDENTITY_ROT6D),
            "no_motion_gripper": "explicit_observation_bound_hold_value",
            "policy_swapped": "real_distinct_candidate_policy_trace",
            "temporally_shifted": "recorded_trace_cyclic_shift_where_valid",
            "literal_zero_rot6d_forbidden": True,
            "synthetic_constant_policy_swapped_forbidden": True,
            "external_action_normalization": "none_for_pinned_droid_forward_dynamics_contract",
            "model_width_padding": "zero_pad_after_raw_width_10_without_rescaling",
        },
        "collapse_handling": {
            "retained_and_counted_against_reliability": True,
            "may_trigger_safety_abstention_or_early_terminal": True,
            "categories": [
                "static_or_frozen_future_frames",
                "first_future_frame_collapse",
                "repeated_frame_loop",
                "sudden_visual_discontinuity",
                "robot_or_skeleton_divergence",
                "object_disappearance_or_scene_corruption",
                "out_of_view_robot_trajectory",
                "uncertainty_increase_across_horizon",
                "action_following_degradation_with_rollout_depth",
            ],
        },
        "normalization_resolution": {
            "selected_transform": "raw_droid_midtrain_actions_no_external_normalization",
            "translation_units": "meters",
            "rotation": "backward_framewise_rot6d_after_droid_to_opencv",
            "gripper": "explicit_open_close_state_after_dataset_flip_when_configured",
            "dose_response_required": False,
            "basis": "pinned_upstream_droid_dataset_config_and_vllm_request_path",
        },
        "cost_caps_usd": {
            "openai_evaluator_api": 25.0,
            "gpu_compute": 50.0,
            "storage_and_transfer": 10.0,
            "total_campaign": 100.0,
            "maximum_concurrent_gpus": 1,
        },
        "claim_boundaries": {
            "phase_A_can_prove": "Blueprint can reproduce a published known-answer result",
            "phase_A_cannot_prove": "independent generalization to new sessions or sites",
            "phase_B_can_prove": "disjoint DROID/RoboArena full-stack rank calibration",
            "phase_B_cannot_prove": "captured-site transfer or physical deployment",
            "phase_C_requires": "site-specific independently published physical outcomes",
        },
        "terminal_components": [
            "cosmos_wam_qualification",
            "frozen_benchmark_calibration",
            "captured_site_transfer",
            "economics_and_speed",
        ],
        "overall_verdicts": ["thesis_supported", "thesis_not_supported", "inconclusive"],
        "supersedes_protocol_sha256": SUPERSEDED_PROTOCOL_SHA256,
        "paid_execution_admitted": False,
        "provider_called": False,
        "outcome_labels_accessed": False,
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    return protocol


def build_phase_a_inventory(
    *,
    rollout_root: str | Path,
    roboarena_root: str | Path,
    expected_session_count: int = 63,
    expected_policy_count: int = 7,
) -> dict[str, Any]:
    """Build the credential-free full-episode known-answer request matrix.

    The inventory contains no preference, success, or long-form feedback field.
    The provider payload may use the task instruction and generated left half,
    but never the policy identifier or the real-robot right half.
    """

    rollouts = Path(rollout_root).resolve()
    outcomes = Path(roboarena_root).resolve()
    if not rollouts.is_dir():
        raise CalibrationContractError("phase_a_rollout_root_missing")
    sessions_root = outcomes / "evaluation_sessions"
    if not sessions_root.is_dir():
        raise CalibrationContractError("phase_a_roboarena_sessions_missing")

    protocol = preregistered_protocol()
    evaluator_identity = {
        "model": PHASE_A_MODEL,
        "method": "roboworld_inspired_full_episode_progress_v1",
        "frame_count": PHASE_A_FRAME_COUNT,
        "frame_sampling_rule": "32_even_positions_with_replacement_for_sources_under_32_frames",
        "generated_crop": "candidate_left_half_requires_layout_audit_before_transport",
        "policy_identity_in_provider_prompt": False,
        "benchmark_outcomes_in_provider_prompt": False,
        "third_party_physical_pixels_in_provider_prompt": False,
    }
    evaluator_digest = canonical_sha256(evaluator_identity)
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    policy_ids: set[str] = set()
    session_ids: list[str] = []

    for session_dir in sorted(
        path for path in rollouts.iterdir() if path.is_dir() and not path.name.startswith(".")
    ):
        metadata_path = sessions_root / session_dir.name / "metadata.yaml"
        if not metadata_path.is_file():
            blockers.append(f"session_metadata_missing:{session_dir.name}")
            continue
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            blockers.append(f"session_metadata_invalid:{session_dir.name}")
            continue
        instruction = str(metadata.get("language_instruction") or "").strip()
        if not instruction:
            blockers.append(f"task_instruction_missing:{session_dir.name}")
            continue
        session_policy_dirs = sorted(path for path in session_dir.iterdir() if path.is_dir())
        if len(session_policy_dirs) != expected_policy_count:
            blockers.append(
                f"session_policy_count_expected_{expected_policy_count}_got_"
                f"{len(session_policy_dirs)}:{session_dir.name}"
            )
        session_ids.append(session_dir.name)
        for policy_dir in session_policy_dirs:
            video = policy_dir / "left" / "compare_overlay_vs_gt.mp4"
            if not video.is_file():
                blockers.append(f"full_episode_missing:{session_dir.name}:{policy_dir.name}")
                continue
            with video.open("rb") as handle:
                header = handle.read(42)
            if header.startswith(b"version https://git-lfs.github.com"):
                blockers.append(f"full_episode_not_materialized:{session_dir.name}:{policy_dir.name}")
                continue
            policy_ids.add(policy_dir.name)
            relative_path = video.relative_to(rollouts).as_posix()
            digest = file_sha256(video)
            request_identity = {
                "session_id": session_dir.name,
                "policy_id_internal_only": policy_dir.name,
                "task_instruction": instruction,
                "relative_path": relative_path,
                "video_sha256": digest,
                "evaluator_digest": evaluator_digest,
            }
            rows.append(
                {
                    **request_identity,
                    "request_id": canonical_sha256(request_identity),
                    "frame_count": PHASE_A_FRAME_COUNT,
                    "full_episode_source": True,
                    "policy_identity_in_provider_prompt": False,
                    "benchmark_outcomes_in_provider_prompt": False,
                    "third_party_physical_pixels_in_provider_prompt": False,
                }
            )

    if len(session_ids) != expected_session_count:
        blockers.append(
            f"session_count_expected_{expected_session_count}_got_{len(session_ids)}"
        )
    if len(policy_ids) != expected_policy_count:
        blockers.append(f"policy_count_expected_{expected_policy_count}_got_{len(policy_ids)}")
    expected_rows = expected_session_count * expected_policy_count
    if len(rows) != expected_rows:
        blockers.append(f"request_count_expected_{expected_rows}_got_{len(rows)}")

    result: dict[str, Any] = {
        "schema_version": "policy_ranking_roboarena_phase_a_inventory.v2",
        "experiment_id": EXPERIMENT_ID,
        "phase": "A_public_known_answer_reproduction",
        "claim_class": "reproduction_only_not_independent_confirmation",
        "status": "ready" if not blockers else "blocked",
        "protocol_sha256": protocol["protocol_sha256"],
        "evaluator": evaluator_identity,
        "evaluator_digest": evaluator_digest,
        "session_count": len(session_ids),
        "policy_count": len(policy_ids),
        "request_count": len(rows),
        "session_ids": session_ids,
        "policy_ids_internal_only": sorted(policy_ids),
        "requests": rows,
        "blockers": sorted(set(blockers)),
        "provider_called": False,
        "data_uploaded": False,
        "outcome_fields_loaded_into_inventory": False,
        "actual_rollout_root_persisted": False,
    }
    result["inventory_sha256"] = canonical_sha256(result)
    return result


def validate_preregistered_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Check the attribution and phase-order constraints before any paid run."""

    candidate = dict(protocol)
    supplied_digest = str(candidate.pop("protocol_sha256", ""))
    if candidate.get("schema_version") != SCHEMA_VERSION:
        raise CalibrationContractError("protocol_schema_invalid")
    if supplied_digest != canonical_sha256(candidate):
        raise CalibrationContractError("protocol_sha256_mismatch")
    graph = candidate.get("backend_graph")
    if not isinstance(graph, Mapping):
        raise CalibrationContractError("backend_graph_missing")
    forbidden = set(graph.get("forbidden_edges") or [])
    required_forbidden = {
        "oscar_wam->cosmos3_wam_parallel_diagnostic",
        "cosmos3_wam_parallel_diagnostic->oscar_wam",
    }
    if not required_forbidden.issubset(forbidden):
        raise CalibrationContractError("serial_wam_chain_not_forbidden")
    phases = candidate.get("phases")
    if not isinstance(phases, list) or [row.get("phase") for row in phases] != [
        "A_public_known_answer_reproduction",
        "B_independent_closed_loop_confirmation",
        "C_captured_site_transfer",
    ]:
        raise CalibrationContractError("phase_order_invalid")
    if phases[0].get("full_episode_required") is not True:
        raise CalibrationContractError("known_answer_full_episode_not_required")
    prefix_steps = phases[1].get("executed_prefix_steps")
    if isinstance(prefix_steps, bool) or not isinstance(prefix_steps, int) or prefix_steps <= 0:
        raise CalibrationContractError("closed_loop_prefix_steps_must_be_positive_integer")
    if prefix_steps > DROID_ACTION_CHUNK:
        raise CalibrationContractError("closed_loop_prefix_steps_exceed_action_chunk")
    if "executed_prefix_seconds" in phases[1]:
        raise CalibrationContractError("closed_loop_prefix_seconds_cannot_govern_v2")
    derived = phases[1].get("executed_prefix_seconds_derived")
    if not isinstance(derived, (int, float)) or not math.isclose(
        float(derived), prefix_steps / DROID_FREQUENCY_HZ, rel_tol=0.0, abs_tol=1e-12
    ):
        raise CalibrationContractError("closed_loop_prefix_derived_duration_mismatch")
    risk_rule = (candidate.get("endpoint_gates") or {}).get("abstention_risk_rule")
    if not isinstance(risk_rule, Mapping):
        raise CalibrationContractError("abstention_risk_rule_missing")
    if risk_rule.get("adjacent_risk_increase_tolerance") != ABSTENTION_ADJACENT_RISK_TOLERANCE:
        raise CalibrationContractError("abstention_risk_tolerance_mismatch")
    collapse = candidate.get("collapse_handling")
    if not isinstance(collapse, Mapping) or collapse.get("retained_and_counted_against_reliability") is not True:
        raise CalibrationContractError("collapse_retention_not_required")
    return {
        "status": "passed",
        "protocol_sha256": supplied_digest,
        "serial_wam_chain_forbidden": True,
        "full_episode_positive_control_required": True,
        "captured_site_transfer_fail_closed": True,
        "executed_prefix_steps": prefix_steps,
        "executed_prefix_seconds_derived": float(derived),
        "uncertainty_aware_risk_rule_required": True,
        "collapse_retention_required": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args == ["protocol"]:
        print(json.dumps(preregistered_protocol(), indent=2, sort_keys=True))
        return 0
    if len(args) == 3 and args[0] == "phase-a-inventory":
        inventory = build_phase_a_inventory(rollout_root=args[1], roboarena_root=args[2])
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return 0 if inventory["status"] == "ready" else 2
    raise SystemExit(
        "usage: policy_ranking_roboarena_calibration "
        "[protocol | phase-a-inventory <rollout_root> <roboarena_root>]"
    )


if __name__ == "__main__":
    raise SystemExit(main())
