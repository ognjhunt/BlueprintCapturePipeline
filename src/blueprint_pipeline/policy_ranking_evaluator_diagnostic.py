"""Post-unseal evaluator comparison for the public RoboArena episodes.

This module intentionally produces diagnostic evidence only.  The source
episodes and their physical outcomes have already been unsealed, so no result
from this campaign can be promoted to independent confirmation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .policy_ranking_roboarena_calibration import canonical_sha256, file_sha256


SCHEMA_VERSION = "policy_ranking_evaluator_diagnostic.v3"
PAIR_INVENTORY_SCHEMA_VERSION = "policy_ranking_pair_inventory.v1"
PAIR_RESULT_SCHEMA_VERSION = "policy_ranking_pair_result.v1"
PAIR_ANALYSIS_SCHEMA_VERSION = "policy_ranking_pair_analysis.v1"
NATIVE_VIDEO_MANIFEST_SCHEMA_VERSION = "policy_ranking_native_video_manifest.v1"
SESSION_COUNT = 63
POLICY_COUNT = 7
PAIR_COUNT_PER_SESSION = 7
TOTAL_PAIR_COUNT = SESSION_COUNT * PAIR_COUNT_PER_SESSION

GPT5_MODEL = "gpt-5-2025-08-07"
GPT54_MINI_MODEL = "gpt-5.4-mini-2026-03-17"
GEMINI_MODEL = "gemini-3.6-flash"
COSMOS_MODEL = "nvidia/Cosmos3-Nano"
COSMOS_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"

PAIR_PROMPT = (
    "You are Blueprint's independent conservative robot-episode comparison judge. "
    "Two complete generated episodes, A and B, show the same task and starting session. "
    "Policy identities, published physical outcomes, and physical ground-truth pixels are "
    "withheld. Compare visible full-episode task progress directly. Success requires task "
    "completion that remains stable across adjacent moments. A visible skeleton is intended-"
    "motion evidence, not proof of an object or scene consequence. Retain and penalize frozen "
    "futures, loops, discontinuities, robot-skeleton divergence, disappearance, corruption, "
    "and out-of-view motion. Choose A, B, tie, or abstain. Abstain when neither ordering is "
    "trustworthy. Return only the registered JSON object."
)

PAIR_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "preferred_episode": {
            "type": "string",
            "enum": ["A", "B", "tie", "abstain"],
        },
        "episode_a_progress_0_to_5": {"type": "integer", "minimum": 0, "maximum": 5},
        "episode_b_progress_0_to_5": {"type": "integer", "minimum": 0, "maximum": 5},
        "stable_success_a": {"type": "boolean"},
        "stable_success_b": {"type": "boolean"},
        "comparison_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
        "decisive_evidence": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string"},
        },
        "artifact_flags_a": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string"},
        },
        "artifact_flags_b": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string"},
        },
        "abstention_factors": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string"},
        },
    },
    "required": [
        "preferred_episode",
        "episode_a_progress_0_to_5",
        "episode_b_progress_0_to_5",
        "stable_success_a",
        "stable_success_b",
        "comparison_confidence",
        "uncertainty",
        "decisive_evidence",
        "artifact_flags_a",
        "artifact_flags_b",
        "abstention_factors",
    ],
}


class DiagnosticContractError(ValueError):
    """The post-unseal diagnostic contract is invalid."""


def diagnostic_protocol() -> dict[str, Any]:
    """Return the prospectively frozen four-arm diagnostic design."""

    protocol: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "post_unseal_evaluator_failure_diagnosis",
        "claim_class": "diagnostic_only",
        "independent_confirmation_credit": False,
        "phase_b_admission_credit": False,
        "source_episode_count": 441,
        "source_and_model_freeze": [
            {
                "name": "OSCAR_public_policy_rollouts",
                "revision": "db5edfaef285c15d0a41d5115177a983c08b4f5f",
                "license": "upstream_repository_license_and_dataset_terms",
                "use": "already_unsealed_generated_episode_pixels",
            },
            {
                "name": "RoboArena_public_outcomes",
                "revision": "7931db81f3f6a48a3245427f7213a4c461f92ccc",
                "license": "upstream_repository_license_and_dataset_terms",
                "use": "already_unsealed_diagnostic_answer_key",
            },
            {
                "name": GPT5_MODEL,
                "url": "https://developers.openai.com/api/docs/models/gpt-5",
                "identity_kind": "immutable_API_snapshot",
                "terms": "OpenAI_API_service_terms",
            },
            {
                "name": GPT54_MINI_MODEL,
                "url": "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
                "identity_kind": "immutable_API_snapshot",
                "terms": "OpenAI_API_service_terms",
            },
            {
                "name": GEMINI_MODEL,
                "url": "https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash",
                "identity_kind": "stable_GA_API_model_observed_2026_07_28",
                "terms": "Google_Gemini_API_service_terms",
            },
            {
                "name": COSMOS_MODEL,
                "url": "https://huggingface.co/nvidia/Cosmos3-Nano",
                "revision": COSMOS_REVISION,
                "last_modified_observed": "2026-07-09T16:28:47Z",
                "license": "NVIDIA_Open_Model_License",
            },
            {
                "name": "NVIDIA/cosmos",
                "url": "https://github.com/NVIDIA/cosmos",
                "revision": "bebca76311266941d06c5f5572fb601184ba24fa",
                "license": "Apache-2.0",
                "license_sha256": (
                    "2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df"
                ),
            },
        ],
        "session_count": SESSION_COUNT,
        "policy_count": POLICY_COUNT,
        "comparison_graph": {
            "kind": "sorted_policy_cycle_per_session",
            "edges_per_session": PAIR_COUNT_PER_SESSION,
            "total_edges": TOTAL_PAIR_COUNT,
            "complete_graph_edges_per_session": 21,
            "complete_graph_total_edges": 1323,
            "side_assignment": "sha256_session_edge_parity",
            "label_fields_used_to_select_edges": False,
            "all_episodes_included": True,
            "each_episode_appearances": 2,
            "connected_for_bradley_terry": True,
            "claim_label": "OSCAR_inspired_reduced_graph_not_exact_OSCAR_reproduction",
        },
        "shared_judging_contract": {
            "prompt": PAIR_PROMPT,
            "prompt_sha256": canonical_sha256(PAIR_PROMPT),
            "output_schema": PAIR_OUTPUT_SCHEMA,
            "output_schema_sha256": canonical_sha256(PAIR_OUTPUT_SCHEMA),
            "policy_identity_in_provider_payload": False,
            "physical_outcome_in_provider_payload": False,
            "physical_ground_truth_pixels_in_provider_payload": False,
            "ties": "half_win_each_in_bradley_terry",
            "abstentions": "retained_and_excluded_from_fit",
            "default_max_output_tokens_including_reasoning": 3000,
        },
        "arms": [
            {
                "arm_id": "gpt5_oscar_comparability",
                "provider": "openai",
                "model": GPT5_MODEL,
                "media": "32_generated_only_frames_per_episode_64_per_pair",
                "reasoning_effort": "high",
                "max_output_tokens_including_reasoning": 4000,
                "transport": "batch_api_sequential_shards",
                "diagnostic_role": "paper_comparability_anchor",
                "full_matrix_cap_usd": 8.75,
            },
            {
                "arm_id": "gpt54_mini_challenger",
                "provider": "openai",
                "model": GPT54_MINI_MODEL,
                "media": "32_generated_only_frames_per_episode_64_per_pair",
                "reasoning_effort": "medium",
                "max_output_tokens_including_reasoning": 3000,
                "transport": "batch_api_sequential_shards",
                "diagnostic_role": "newer_lower_cost_openai_challenger",
                "full_matrix_cap_usd": 4.75,
            },
            {
                "arm_id": "gemini36_flash_native_video",
                "provider": "google",
                "model": GEMINI_MODEL,
                "media": "two_generated_only_native_mp4_videos",
                "thinking_level": "medium",
                "transport": "batch_api",
                "diagnostic_role": "cross_family_native_video_challenger",
                "full_matrix_cap_usd": 8.75,
            },
            {
                "arm_id": "cosmos3_nano_reasoner",
                "provider": "self_hosted_gpu",
                "model": COSMOS_MODEL,
                "revision": COSMOS_REVISION,
                "surface": "reasoner_only_vllm",
                "media": "two_generated_only_native_mp4_videos",
                "diagnostic_role": "open_physical_ai_native_video_challenger",
                "cannot_be_sole_judge_of_native_cosmos_generated_rollouts": True,
                "full_matrix_cap_usd": 15.0,
            },
        ],
        "admission_and_stopping": {
            "one_schema_transport_canary_per_arm": True,
            "small_label_free_transport_batch_after_canary": True,
            "full_matrix_requires_measured_projected_cost_within_arm_cap": True,
            "seven_pair_cost_projection": {
                "sample_size": 7,
                "per_request_upper_estimate": (
                    "max(single_canary_batch_equivalent_cost, pilot_max_cost, "
                    "pilot_mean_cost_plus_1.943_times_sample_standard_error)"
                ),
                "matrix_projection": "per_request_upper_estimate_times_441",
                "arm_admission": "matrix_projection_lte_frozen_arm_cap",
                "campaign_admission": (
                    "prior_phase_a_plus_all_realized_diagnostic_api_costs_plus_all_"
                    "remaining_admitted_matrix_projections_lte_25_usd"
                ),
                "ambiguous_duplicate_requests": "count_conservatively_until_billing_resolved",
            },
            "partial_matrix_ranking_credit": False,
            "scientifically_unfavorable_outputs_are_not_a_stop_reason": True,
            "stop_reasons": [
                "schema_or_transport_invalid",
                "redaction_or_media_integrity_failure",
                "measured_projection_exceeds_frozen_arm_cap",
                "campaign_category_or_total_cap_would_be_exceeded",
            ],
        },
        "cost_caps_usd": {
            "combined_evaluator_api_including_prior_phase_a": 25.0,
            "prior_phase_a_measured": 2.53707425,
            "gpu_compute_campaign": 50.0,
            "storage_and_transfer_campaign": 10.0,
            "total_campaign": 100.0,
            "maximum_concurrent_gpus": 1,
            "api_arm_caps_plus_prior_total": 24.78707425,
            "api_contingency_remaining": 0.21292575,
        },
        "prospective_api_pricing_usd_per_million_tokens": {
            "gpt5_standard": {"input": 1.25, "output": 10.0},
            "gpt54_mini_standard": {"input": 0.75, "output": 4.5},
            "openai_batch_discount_fraction": 0.5,
            "gemini36_flash_standard": {"input": 1.5, "output": 7.5},
            "gemini36_flash_batch": {"input": 0.75, "output": 3.75},
            "pricing_is_prospective_not_measured_campaign_economics": True,
        },
        "analysis": {
            "ranking": "Bradley_Terry_maximum_likelihood",
            "ties": "half_win_each",
            "abstentions": "retained_excluded_and_reported_as_coverage",
            "metrics": [
                "full_seven_policy_vector",
                "spearman_rho",
                "kendall_tau_b",
                "pairwise_ordering_accuracy",
                "session_clustered_bootstrap_ci95",
                "exact_or_permutation_small_n_uncertainty",
                "top_policy_rank",
                "abstention_coverage",
                "cost_and_latency",
            ],
            "outcomes_already_unsealed": True,
            "threshold_tuning_forbidden": True,
        },
        "paid_execution_admitted": False,
        "evaluator_provider_called": False,
        "provider_metadata_lookup_called": True,
        "provider_metadata_lookup_scope": "Gemini_model_get_no_generation_no_media",
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    return protocol


def _validate_source_inventory(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if source.get("status") != "ready" or source.get("request_count") != 441:
        raise DiagnosticContractError("source_inventory_not_complete_ready_441")
    requests = source.get("requests")
    if not isinstance(requests, list) or len(requests) != 441:
        raise DiagnosticContractError("source_requests_not_complete_441")
    payload = {key: value for key, value in source.items() if key != "inventory_sha256"}
    if canonical_sha256(payload) != source.get("inventory_sha256"):
        raise DiagnosticContractError("source_inventory_digest_invalid")
    return requests


def build_pair_inventory(source: Mapping[str, Any]) -> dict[str, Any]:
    """Build a label-free seven-edge cycle for every public session."""

    rows = _validate_source_inventory(source)
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row["session_id"])].append(row)
    if len(by_session) != SESSION_COUNT:
        raise DiagnosticContractError("session_count_not_63")

    pairs: list[dict[str, Any]] = []
    for session_id in sorted(by_session):
        session_rows = sorted(
            by_session[session_id], key=lambda value: str(value["policy_id_internal_only"])
        )
        if len(session_rows) != POLICY_COUNT:
            raise DiagnosticContractError(f"policy_count_not_7:{session_id}")
        if len({str(row["task_instruction"]) for row in session_rows}) != 1:
            raise DiagnosticContractError(f"task_instruction_differs_within_session:{session_id}")
        for edge_index in range(PAIR_COUNT_PER_SESSION):
            left = session_rows[edge_index]
            right = session_rows[(edge_index + 1) % POLICY_COUNT]
            side_digest = canonical_sha256(
                {"session_id": session_id, "edge_index": edge_index, "side_rule": "v1"}
            )
            if int(side_digest[-1], 16) % 2:
                left, right = right, left
            pair_core = {
                "session_id": session_id,
                "edge_index": edge_index,
                "task_instruction": left["task_instruction"],
                "episode_a": {
                    "source_request_id": left["source_request_id"],
                    "policy_id_internal_only": left["policy_id_internal_only"],
                    "frames": left["frames"],
                    "cropped_output_sha256": left["cropped_output_sha256"],
                    "deterministic_collapse_flags": left.get(
                        "deterministic_collapse_flags", []
                    ),
                },
                "episode_b": {
                    "source_request_id": right["source_request_id"],
                    "policy_id_internal_only": right["policy_id_internal_only"],
                    "frames": right["frames"],
                    "cropped_output_sha256": right["cropped_output_sha256"],
                    "deterministic_collapse_flags": right.get(
                        "deterministic_collapse_flags", []
                    ),
                },
                "policy_identity_in_provider_payload": False,
                "physical_outcome_in_provider_payload": False,
                "physical_ground_truth_pixels_in_provider_payload": False,
            }
            pair_core["pair_id"] = canonical_sha256(pair_core)
            pairs.append(pair_core)

    if len(pairs) != TOTAL_PAIR_COUNT:
        raise DiagnosticContractError("pair_count_not_441")
    appearances: dict[tuple[str, str], int] = defaultdict(int)
    for pair in pairs:
        for side in ("episode_a", "episode_b"):
            appearances[(pair["session_id"], pair[side]["policy_id_internal_only"])] += 1
    if set(appearances.values()) != {2}:
        raise DiagnosticContractError("each_episode_must_appear_twice")

    protocol = diagnostic_protocol()
    inventory: dict[str, Any] = {
        "schema_version": PAIR_INVENTORY_SCHEMA_VERSION,
        "status": "ready",
        "protocol_sha256": protocol["protocol_sha256"],
        "source_inventory_sha256": source["inventory_sha256"],
        "session_count": len(by_session),
        "policy_count": POLICY_COUNT,
        "pair_count": len(pairs),
        "pairs": pairs,
        "provider_called": False,
        "outcome_labels_accessed_to_build_pairs": False,
        "blockers": [],
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    return inventory


def validate_pair_result(result: Mapping[str, Any], pair: Mapping[str, Any]) -> None:
    if result.get("schema_version") != PAIR_RESULT_SCHEMA_VERSION:
        raise DiagnosticContractError("pair_result_schema_invalid")
    if result.get("pair_id") != pair.get("pair_id"):
        raise DiagnosticContractError("pair_result_id_mismatch")
    response = result.get("structured_response")
    if not isinstance(response, Mapping):
        raise DiagnosticContractError("pair_structured_response_missing")
    preference = response.get("preferred_episode")
    if preference not in {"A", "B", "tie", "abstain"}:
        raise DiagnosticContractError("pair_preference_invalid")
    if preference == "abstain" and not response.get("abstention_factors"):
        raise DiagnosticContractError("pair_abstention_factor_missing")
    recorded = result.get("result_sha256")
    payload = {key: value for key, value in result.items() if key != "result_sha256"}
    if canonical_sha256(payload) != recorded:
        raise DiagnosticContractError("pair_result_digest_invalid")


def bradley_terry_scores(
    pairs: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    """Fit Bradley-Terry strengths by the standard MM update."""

    pair_by_id = {str(pair["pair_id"]): pair for pair in pairs}
    if len(pair_by_id) != len(pairs):
        raise DiagnosticContractError("duplicate_pair_ids")
    wins: dict[str, float] = defaultdict(float)
    games: dict[tuple[str, str], float] = defaultdict(float)
    policies: set[str] = set()
    usable = 0
    for result in results:
        pair = pair_by_id.get(str(result.get("pair_id") or ""))
        if pair is None:
            raise DiagnosticContractError("result_not_in_inventory")
        validate_pair_result(result, pair)
        a = str(pair["episode_a"]["policy_id_internal_only"])
        b = str(pair["episode_b"]["policy_id_internal_only"])
        policies.update((a, b))
        preference = result["structured_response"]["preferred_episode"]
        if preference == "abstain":
            continue
        usable += 1
        games[tuple(sorted((a, b)))] += 1.0
        if preference == "A":
            wins[a] += 1.0
        elif preference == "B":
            wins[b] += 1.0
        else:
            wins[a] += 0.5
            wins[b] += 0.5
    if usable == 0 or len(policies) != POLICY_COUNT:
        raise DiagnosticContractError("insufficient_usable_connected_results")

    strength = {policy: 1.0 for policy in policies}
    for _ in range(10_000):
        updated: dict[str, float] = {}
        for policy in policies:
            denominator = 0.0
            for opponent in policies:
                if opponent == policy:
                    continue
                count = games.get(tuple(sorted((policy, opponent))), 0.0)
                if count:
                    denominator += count / (strength[policy] + strength[opponent])
            updated[policy] = max(wins[policy], 1e-12) / max(denominator, 1e-12)
        mean = sum(updated.values()) / len(updated)
        updated = {policy: value / mean for policy, value in updated.items()}
        delta = max(abs(updated[policy] - strength[policy]) for policy in policies)
        strength = updated
        if delta < 1e-12:
            break
    return dict(sorted(strength.items()))


def analyze_pair_results(
    inventory: Mapping[str, Any], results: Sequence[Mapping[str, Any]], *, arm_id: str
) -> dict[str, Any]:
    pairs = inventory.get("pairs")
    if not isinstance(pairs, list):
        raise DiagnosticContractError("pair_inventory_missing_pairs")
    if len(results) != len(pairs):
        raise DiagnosticContractError("partial_matrix_has_no_ranking_credit")
    scores = bradley_terry_scores(pairs, results)
    abstentions = sum(
        result["structured_response"]["preferred_episode"] == "abstain"
        for result in results
    )
    report: dict[str, Any] = {
        "schema_version": PAIR_ANALYSIS_SCHEMA_VERSION,
        "arm_id": arm_id,
        "claim_class": "post_unseal_diagnostic_only",
        "independent_confirmation_credit": False,
        "pair_count": len(results),
        "usable_pair_count": len(results) - abstentions,
        "abstention_count": abstentions,
        "coverage": (len(results) - abstentions) / len(results),
        "bradley_terry_strength_by_policy": scores,
        "predicted_policy_order": sorted(scores, key=lambda key: (-scores[key], key)),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def verify_pair_media(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Verify all previously audited frame files before any new transport."""

    failures: list[str] = []
    checked: set[str] = set()
    for pair in inventory.get("pairs", []):
        for side in ("episode_a", "episode_b"):
            for frame in pair[side]["frames"]:
                path = str(frame["path"])
                if path in checked:
                    continue
                checked.add(path)
                resolved = Path(path)
                if not resolved.is_file() or file_sha256(resolved) != frame["sha256"]:
                    failures.append(path)
    report: dict[str, Any] = {
        "status": "passed" if not failures else "blocked",
        "unique_frame_count": len(checked),
        "expected_unique_frame_count": 441 * 32,
        "failures": failures,
        "provider_called": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _ffprobe(path: Path) -> dict[str, Any]:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(process.stdout).get("streams", [])
    if len(streams) != 1:
        raise DiagnosticContractError(f"native_video_stream_count_invalid:{path}")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": str(stream["avg_frame_rate"]),
        "frame_count": int(stream["nb_read_frames"]),
    }


def _materialize_one_native_video(
    audit_row: Mapping[str, Any], *, source_root: Path, output_root: Path
) -> dict[str, Any]:
    request_id = str(audit_row["request_id"])
    source = source_root / str(audit_row["source_relative_path"])
    if not source.is_file() or file_sha256(source) != audit_row.get("source_video_sha256"):
        raise DiagnosticContractError(f"native_video_source_changed:{request_id}")
    if [audit_row.get("source_width"), audit_row.get("source_height")] != [1280, 480]:
        raise DiagnosticContractError(f"native_video_source_geometry_invalid:{request_id}")
    if audit_row.get("generated_crop_xyxy") != [0, 0, 640, 480]:
        raise DiagnosticContractError(f"native_video_crop_contract_invalid:{request_id}")
    if audit_row.get("physical_right_half_pixels_encoded") is not False:
        raise DiagnosticContractError(f"native_video_prior_crop_audit_invalid:{request_id}")

    destination = output_root / f"{request_id}.mp4"
    temporary = output_root / f".{request_id}.{os.getpid()}.tmp.mp4"
    if destination.is_file():
        probe = _ffprobe(destination)
        if probe["width"] == 640 and probe["height"] == 480:
            return {
                "request_id": request_id,
                "source_relative_path": audit_row["source_relative_path"],
                "source_video_sha256": audit_row["source_video_sha256"],
                "generated_crop_xyxy": [0, 0, 640, 480],
                "physical_right_half_x_range": [640, 1280],
                "physical_right_half_pixels_encoded": False,
                "output_path": str(destination),
                "output_sha256": file_sha256(destination),
                "output_size_bytes": destination.stat().st_size,
                "output_probe": probe,
                "resumed_existing_output": True,
            }
    command = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        "crop=640:480:0:0",
        "-an",
        "-map_metadata",
        "-1",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-threads",
        "1",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        probe = _ffprobe(temporary)
        if [probe["width"], probe["height"]] != [640, 480]:
            raise DiagnosticContractError(f"native_video_output_geometry_invalid:{request_id}")
        if probe["frame_count"] != int(audit_row["source_frame_count"]):
            raise DiagnosticContractError(f"native_video_output_frame_count_invalid:{request_id}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "request_id": request_id,
        "source_relative_path": audit_row["source_relative_path"],
        "source_video_sha256": audit_row["source_video_sha256"],
        "generated_crop_xyxy": [0, 0, 640, 480],
        "physical_right_half_x_range": [640, 1280],
        "physical_right_half_pixels_encoded": False,
        "output_path": str(destination),
        "output_sha256": file_sha256(destination),
        "output_size_bytes": destination.stat().st_size,
        "output_probe": probe,
        "resumed_existing_output": False,
    }


def materialize_native_videos(
    crop_audit: Mapping[str, Any],
    *,
    visual_review: Mapping[str, Any],
    source_root: str | Path,
    output_root: str | Path,
    workers: int = 4,
) -> dict[str, Any]:
    """Crop all 441 full episodes to generated-only native MP4 inputs."""

    payload = {key: value for key, value in crop_audit.items() if key != "audit_sha256"}
    if canonical_sha256(payload) != crop_audit.get("audit_sha256"):
        raise DiagnosticContractError("crop_audit_digest_invalid")
    rows = crop_audit.get("requests")
    if (
        crop_audit.get("status") != "ready_for_manual_visual_review"
        or crop_audit.get("all_physical_right_half_pixels_excluded") is not True
        or not isinstance(rows, list)
        or len(rows) != 441
    ):
        raise DiagnosticContractError("crop_audit_not_complete_passed_441")
    if (
        visual_review.get("status") != "passed"
        or visual_review.get("crop_audit_sha256") != crop_audit.get("audit_sha256")
        or visual_review.get("all_441_geometry_and_hash_audited") is not True
        or visual_review.get("representative_policy_count") != 7
    ):
        raise DiagnosticContractError("crop_visual_review_not_passed")
    root = Path(source_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        future_by_id = {
            pool.submit(
                _materialize_one_native_video,
                row,
                source_root=root,
                output_root=output,
            ): str(row["request_id"])
            for row in rows
        }
        for future in as_completed(future_by_id):
            request_id = future_by_id[future]
            try:
                receipts.append(future.result())
            except Exception as exc:  # fail closed while preserving all failures
                failures.append({"request_id": request_id, "error_type": type(exc).__name__})
    receipts.sort(key=lambda value: value["request_id"])
    total_bytes = sum(int(value["output_size_bytes"]) for value in receipts)
    manifest: dict[str, Any] = {
        "schema_version": NATIVE_VIDEO_MANIFEST_SCHEMA_VERSION,
        "status": "passed" if len(receipts) == 441 and not failures else "blocked",
        "source_crop_audit_sha256": crop_audit["audit_sha256"],
        "source_visual_review_sha256": canonical_sha256(visual_review),
        "source_root": str(root),
        "output_root": str(output),
        "ffmpeg_transform": (
            "crop=640:480:0:0;h264_libx264_preset_fast_crf18_yuv420p;no_audio;no_metadata"
        ),
        "video_count": len(receipts),
        "total_bytes": total_bytes,
        "paid_storage_cost_usd": 0.0,
        "provider_called": False,
        "data_uploaded": False,
        "all_physical_right_half_pixels_excluded": (
            len(receipts) == 441
            and all(row["physical_right_half_pixels_encoded"] is False for row in receipts)
        ),
        "receipts": receipts,
        "failures": failures,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    protocol = commands.add_parser("protocol")
    protocol.add_argument("--output", required=True)
    inventory = commands.add_parser("build-inventory")
    inventory.add_argument("--source-inventory", required=True)
    inventory.add_argument("--output", required=True)
    verify = commands.add_parser("verify-media")
    verify.add_argument("--inventory", required=True)
    verify.add_argument("--output", required=True)
    video = commands.add_parser("materialize-videos")
    video.add_argument("--crop-audit", required=True)
    video.add_argument("--visual-review", required=True)
    video.add_argument("--source-root", required=True)
    video.add_argument("--output-root", required=True)
    video.add_argument("--output", required=True)
    video.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "protocol":
        result = diagnostic_protocol()
    elif args.command == "build-inventory":
        source = json.loads(Path(args.source_inventory).read_text(encoding="utf-8"))
        result = build_pair_inventory(source)
    elif args.command == "verify-media":
        value = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
        result = verify_pair_media(value)
    else:
        value = json.loads(Path(args.crop_audit).read_text(encoding="utf-8"))
        visual_review = json.loads(Path(args.visual_review).read_text(encoding="utf-8"))
        result = materialize_native_videos(
            value,
            visual_review=visual_review,
            source_root=args.source_root,
            output_root=args.output_root,
            workers=args.workers,
        )
    write_json(Path(args.output), result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"pairs", "receipts"}
            }
        )
    )
    return 0 if result.get("status", "ready") in {"ready", "passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
