"""Label-blind GPT-5 mini evaluator for audited Phase-A episode crops."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import write_json
from .policy_ranking_roboarena_calibration import canonical_sha256, file_sha256


MODEL = "gpt-5-mini-2025-08-07"
GATE_ENV = "BLUEPRINT_ALLOW_ROBOARENA_PHASE_A_OPENAI"
REASONING_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 3000
MAX_ESTIMATED_REQUEST_USD = 0.05
MAX_ATTEMPTS_PER_REQUEST = 2
INPUT_USD_PER_MILLION_TOKENS = 0.25
CACHED_INPUT_USD_PER_MILLION_TOKENS = 0.025
OUTPUT_USD_PER_MILLION_TOKENS = 2.0
PROMPT = (
    "You are Blueprint's independent, conservative evaluator of one complete generated "
    "robot episode. The 32 images are chronological audited crops containing only the "
    "generated OSCAR side; policy identity, benchmark outcomes, and physical ground-truth "
    "pixels are withheld. Build a task-specific ordered progress rubric, then score the full "
    "episode. Success requires visible task completion that remains stable across multiple "
    "adjacent sampled positions. A wrist or close view may reveal artifacts but cannot by "
    "itself establish success. Separate intended skeleton motion from scene consequences. "
    "Flag frozen futures, first-future-frame collapse, repeated loops, discontinuity, "
    "robot-skeleton divergence, object disappearance, scene corruption, out-of-view robot "
    "motion, or increasing uncertainty. Abstain when the pixels do not support a trustworthy "
    "progress judgment. Do not infer success from attractive motion, the task instruction, "
    "or provider completion. Return only the registered JSON schema."
)
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_specific_rubric": {
            "type": "array",
            "minItems": 3,
            "maxItems": 6,
            "items": {"type": "string"},
        },
        "progress_score_0_to_5": {"type": "integer", "minimum": 0, "maximum": 5},
        "success_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "stable_success_confirmed": {"type": "boolean"},
        "stable_success_frame_positions": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "integer", "minimum": 0, "maximum": 31},
        },
        "success_evidence": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string"},
        },
        "failure_evidence": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string"},
        },
        "artifact_flags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "static_or_frozen_future",
                    "first_future_frame_collapse",
                    "repeated_frame_loop",
                    "sudden_visual_discontinuity",
                    "robot_skeleton_divergence",
                    "object_disappearance",
                    "scene_corruption",
                    "robot_out_of_view",
                    "uncertainty_increases_with_depth",
                    "action_following_degrades_with_depth",
                    "none",
                ],
            },
        },
        "temporal_consistency": {"type": "number", "minimum": 0, "maximum": 1},
        "action_following_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
        "abstain": {"type": "boolean"},
        "abstention_factors": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string"},
        },
    },
    "required": [
        "task_specific_rubric",
        "progress_score_0_to_5",
        "success_probability",
        "stable_success_confirmed",
        "stable_success_frame_positions",
        "success_evidence",
        "failure_evidence",
        "artifact_flags",
        "temporal_consistency",
        "action_following_confidence",
        "uncertainty",
        "abstain",
        "abstention_factors",
    ],
}
OUTPUT_SCHEMA_V3 = copy.deepcopy(OUTPUT_SCHEMA)
OUTPUT_SCHEMA_V3["properties"]["artifact_flags"]["uniqueItems"] = True


def _evaluator_contract(*, schema_version: str, output_schema: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "schema_version": schema_version,
        "model": MODEL,
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "output_schema": dict(output_schema),
        "frame_count": 32,
        "image_detail": "low",
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "idempotency_header": "request_id",
        "method": "roboworld_inspired_full_episode_progress_v1",
        "policy_identity_in_provider_payload": False,
        "benchmark_outcomes_in_provider_payload": False,
        "physical_ground_truth_pixels_in_provider_payload": False,
    }
    contract["evaluator_digest"] = canonical_sha256(contract)
    return contract


def evaluator_contract() -> dict[str, Any]:
    return _evaluator_contract(
        schema_version="policy_ranking_roboarena_evaluator_contract.v4",
        output_schema=OUTPUT_SCHEMA,
    )


def evaluator_contract_v3() -> dict[str, Any]:
    """Return the immutable failed-canary transport contract."""

    return _evaluator_contract(
        schema_version="policy_ranking_roboarena_evaluator_contract.v3",
        output_schema=OUTPUT_SCHEMA_V3,
    )


def _secure_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    mode = candidate.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode) or mode & 0o077:
        raise ValueError("secure_file_must_be_regular_0600")
    return candidate.resolve()


def validate_rotation_attestation(path: str | Path) -> dict[str, Any]:
    value = json.loads(_secure_file(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("rotation_attestation_invalid")
    required = {
        "schema_version": "blueprint_openai_key_rotation_attestation.v1",
        "previous_chat_exposed_key_revoked": True,
        "replacement_key_created_after_revocation": True,
        "attested_by": "user",
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("rotation_attestation_requirements_not_met")
    if not str(value.get("recorded_at") or ""):
        raise ValueError("rotation_attestation_timestamp_missing")
    return {
        "schema_version": value["schema_version"],
        "recorded_at": value["recorded_at"],
        "previous_chat_exposed_key_revoked": True,
        "replacement_key_created_after_revocation": True,
        "attested_by": "user",
    }


def build_evaluator_inventory(
    phase_a_inventory: Mapping[str, Any],
    crop_manifest: Mapping[str, Any],
    review_attestation: Mapping[str, Any],
    collapse_report: Mapping[str, Any],
) -> dict[str, Any]:
    contract = evaluator_contract()
    blockers: list[str] = []
    if phase_a_inventory.get("status") != "ready":
        blockers.append("phase_a_inventory_not_ready")
    if crop_manifest.get("status") != "ready_for_manual_visual_review":
        blockers.append("crop_manifest_not_complete")
    if crop_manifest.get("all_physical_right_half_pixels_excluded") is not True:
        blockers.append("physical_pixel_exclusion_not_proven")
    if (
        review_attestation.get("status") != "passed"
        or review_attestation.get("crop_audit_sha256") != crop_manifest.get("audit_sha256")
        or review_attestation.get("contact_sheet_sha256")
        != (crop_manifest.get("contact_sheet") or {}).get("sha256")
    ):
        blockers.append("representative_visual_review_not_bound")
    if collapse_report.get("status") != "completed" or collapse_report.get(
        "crop_audit_sha256"
    ) != crop_manifest.get("audit_sha256"):
        blockers.append("collapse_report_not_bound")

    phase_rows = {str(row["request_id"]): row for row in phase_a_inventory.get("requests") or []}
    collapse_rows = {str(row["request_id"]): row for row in collapse_report.get("episodes") or []}
    rows: list[dict[str, Any]] = []
    output_root = Path(str(crop_manifest.get("output_root") or ""))
    for crop in crop_manifest.get("requests") or []:
        source_request_id = str(crop.get("request_id") or "")
        phase_row = phase_rows.get(source_request_id)
        if phase_row is None:
            blockers.append(f"crop_request_not_in_phase_inventory:{source_request_id}")
            continue
        collapse = collapse_rows.get(source_request_id)
        if collapse is None:
            blockers.append(f"collapse_request_not_found:{source_request_id}")
            continue
        frames: list[dict[str, Any]] = []
        for frame in crop.get("sampled_frames") or []:
            path = output_root / str(frame["relative_output_path"])
            if not path.is_file() or file_sha256(path) != frame.get("encoded_jpeg_sha256"):
                blockers.append(f"audited_frame_missing_or_changed:{source_request_id}")
                break
            frames.append(
                {
                    "sample_position": frame["sample_position"],
                    "source_frame_index": frame["frame_index"],
                    "path": str(path.resolve()),
                    "sha256": frame["encoded_jpeg_sha256"],
                }
            )
        if len(frames) != 32:
            blockers.append(f"audited_frame_count_not_32:{source_request_id}")
            continue
        identity = {
            "source_request_id": source_request_id,
            "cropped_output_sha256": crop["cropped_output_sha256"],
            "task_instruction": phase_row["task_instruction"],
            "evaluator_digest": contract["evaluator_digest"],
        }
        rows.append(
            {
                "request_id": canonical_sha256(identity),
                "source_request_id": source_request_id,
                "session_id": phase_row["session_id"],
                "policy_id_internal_only": phase_row["policy_id_internal_only"],
                "task_instruction": phase_row["task_instruction"],
                "frames": frames,
                "short_episode_source": crop["short_episode_source"],
                "unique_sampled_frame_count": crop["unique_sampled_frame_count"],
                "repeated_sample_count": crop["repeated_sample_count"],
                "cropped_output_sha256": crop["cropped_output_sha256"],
                "evaluator_digest": contract["evaluator_digest"],
                "deterministic_collapse_flags": list(
                    collapse.get("deterministic_collapse_flags") or []
                ),
                "deterministic_safety_abstention_recommended": bool(
                    collapse.get("safety_abstention_recommended")
                ),
                "provider_payload_fields": [
                    "task_instruction",
                    "chronological_sample_positions",
                    "source_frame_indices",
                    "short_episode_sampling_metadata",
                    "audited_generated_images",
                ],
                "policy_identity_in_provider_payload": False,
                "benchmark_outcomes_in_provider_payload": False,
                "physical_ground_truth_pixels_in_provider_payload": False,
            }
        )
    if len(rows) != 441:
        blockers.append(f"evaluator_request_count_expected_441_got_{len(rows)}")
    result: dict[str, Any] = {
        "schema_version": "policy_ranking_roboarena_evaluator_inventory.v4",
        "status": "ready" if not blockers else "blocked",
        "experiment_id": phase_a_inventory.get("experiment_id"),
        "protocol_sha256": phase_a_inventory.get("protocol_sha256"),
        "source_inventory_sha256": phase_a_inventory.get("inventory_sha256"),
        "crop_audit_sha256": crop_manifest.get("audit_sha256"),
        "collapse_report_sha256": collapse_report.get("report_sha256"),
        "review_attestation_sha256": canonical_sha256(dict(review_attestation)),
        "evaluator": contract,
        "request_count": len(rows),
        "requests": rows,
        "blockers": sorted(set(blockers)),
        "provider_called": False,
        "data_uploaded": False,
        "outcome_labels_accessed": False,
        "precall_cost_bound": {
            "per_request_usd": MAX_ESTIMATED_REQUEST_USD,
            "complete_matrix_usd": round(len(rows) * MAX_ESTIMATED_REQUEST_USD, 6),
            "campaign_api_cap_usd": 25.0,
        },
    }
    result["inventory_sha256"] = canonical_sha256(result)
    return result


def supersede_transport_inventory_v3(
    prior_inventory: Mapping[str, Any],
    *,
    expected_request_count: int = 441,
) -> dict[str, Any]:
    """Rebind an unpaid v2 inventory to provider idempotency transport v3.

    The audited frames and scientific prompt stay unchanged.  This amendment is
    allowed only while the prior matrix is label sealed and provider untouched.
    """

    blockers: list[str] = []
    prior = dict(prior_inventory)
    prior_sha = str(prior.get("inventory_sha256") or "")
    prior_without_sha = {key: value for key, value in prior.items() if key != "inventory_sha256"}
    if canonical_sha256(prior_without_sha) != prior_sha:
        blockers.append("prior_inventory_digest_invalid")
    if prior.get("status") != "ready":
        blockers.append("prior_inventory_not_ready")
    if prior.get("provider_called") is not False:
        blockers.append("prior_inventory_provider_already_called")
    if prior.get("data_uploaded") is not False:
        blockers.append("prior_inventory_data_already_uploaded")
    if prior.get("outcome_labels_accessed") is not False:
        blockers.append("prior_inventory_outcomes_already_accessed")

    contract = evaluator_contract_v3()
    former_evaluator_digest = str(prior.get("evaluator", {}).get("evaluator_digest") or "")
    if former_evaluator_digest == contract["evaluator_digest"]:
        blockers.append("prior_inventory_already_uses_transport_v3")
    rows: list[dict[str, Any]] = []
    for raw in prior.get("requests") or []:
        row = dict(raw)
        for frame in row.get("frames") or []:
            path = Path(str(frame.get("path") or ""))
            if not path.is_file() or file_sha256(path) != frame.get("sha256"):
                blockers.append(f"audited_frame_missing_or_changed:{row.get('request_id')}")
                break
        row["evaluator_digest"] = contract["evaluator_digest"]
        identity = {
            "source_request_id": row["source_request_id"],
            "cropped_output_sha256": row["cropped_output_sha256"],
            "task_instruction": row["task_instruction"],
            "evaluator_digest": contract["evaluator_digest"],
        }
        row["request_id"] = canonical_sha256(identity)
        rows.append(row)
    if len(rows) != expected_request_count:
        blockers.append(
            f"evaluator_request_count_expected_{expected_request_count}_got_{len(rows)}"
        )
    if len({str(row["request_id"]) for row in rows}) != len(rows):
        blockers.append("transport_v3_request_ids_not_unique")

    result = {
        key: value
        for key, value in prior.items()
        if key
        not in {"schema_version", "status", "evaluator", "requests", "blockers", "inventory_sha256"}
    }
    result.update(
        {
            "schema_version": "policy_ranking_roboarena_evaluator_inventory.v3",
            "status": "ready" if not blockers else "blocked",
            "evaluator": contract,
            "request_count": len(rows),
            "requests": rows,
            "blockers": sorted(set(blockers)),
            "transport_amendment": {
                "schema_version": "policy_ranking_roboarena_transport_amendment.v3",
                "former_inventory_sha256": prior_sha,
                "former_evaluator_digest": former_evaluator_digest,
                "reason": (
                    "Bind every provider request to its frozen request_id through the "
                    "Idempotency-Key header so a crash between provider acceptance and "
                    "local result persistence cannot duplicate a scientific row."
                ),
                "changed_fields": [
                    "evaluator.schema_version",
                    "evaluator.idempotency_header",
                    "evaluator.evaluator_digest",
                    "requests[*].evaluator_digest",
                    "requests[*].request_id",
                    "schema_version",
                ],
                "prompt_changed": False,
                "schema_changed": False,
                "sampling_changed": False,
                "paid_execution_admitted": False,
                "provider_called": False,
                "outcome_labels_accessed": False,
            },
        }
    )
    result["inventory_sha256"] = canonical_sha256(result)
    return result


def supersede_schema_inventory_v4(
    prior_inventory: Mapping[str, Any],
    failed_run: Mapping[str, Any],
    schema_diagnostic: Mapping[str, Any],
    *,
    schema_diagnostic_file_sha256: str,
    expected_request_count: int = 441,
) -> dict[str, Any]:
    """Replace only the provider-invalid uniqueness keyword after a zero-row canary."""

    blockers: list[str] = []
    prior = dict(prior_inventory)
    prior_sha = str(prior.get("inventory_sha256") or "")
    if (
        canonical_sha256({key: value for key, value in prior.items() if key != "inventory_sha256"})
        != prior_sha
    ):
        blockers.append("prior_inventory_digest_invalid")
    if (
        prior.get("schema_version") != "policy_ranking_roboarena_evaluator_inventory.v3"
        or prior.get("evaluator") != evaluator_contract_v3()
    ):
        blockers.append("prior_inventory_not_exact_v3_contract")

    failed_run_sha = str(failed_run.get("run_sha256") or "")
    if (
        canonical_sha256({key: value for key, value in failed_run.items() if key != "run_sha256"})
        != failed_run_sha
    ):
        blockers.append("failed_run_digest_invalid")
    if (
        failed_run.get("inventory_sha256") != prior_sha
        or failed_run.get("status") != "blocked"
        or failed_run.get("completed_request_count") != 0
        or failed_run.get("provider_called") is not True
        or failed_run.get("outcome_labels_accessed") is not False
    ):
        blockers.append("failed_run_not_zero_row_label_sealed_canary")
    failures = list(failed_run.get("failures") or [])
    if len(failures) != 1 or failures[0].get("error_type") != "BadRequestError":
        blockers.append("failed_run_not_single_bad_request")

    diagnostic_error = schema_diagnostic.get("error") or {}
    if (
        schema_diagnostic.get("status") != "failed"
        or schema_diagnostic.get("diagnostic") != "text_only_exact_model_and_structured_schema"
        or schema_diagnostic.get("experiment_pixels_uploaded") is not False
        or schema_diagnostic.get("outcome_labels_accessed") is not False
        or diagnostic_error.get("provider_error_code") != "invalid_json_schema"
        or diagnostic_error.get("provider_error_param") != "text.format.schema"
        or len(schema_diagnostic_file_sha256) != 64
    ):
        blockers.append("schema_diagnostic_not_exact_invalid_json_schema_evidence")

    contract = evaluator_contract()
    rows: list[dict[str, Any]] = []
    for raw in prior.get("requests") or []:
        row = dict(raw)
        for frame in row.get("frames") or []:
            path = Path(str(frame.get("path") or ""))
            if not path.is_file() or file_sha256(path) != frame.get("sha256"):
                blockers.append(f"audited_frame_missing_or_changed:{row.get('request_id')}")
                break
        row["evaluator_digest"] = contract["evaluator_digest"]
        row["request_id"] = canonical_sha256(
            {
                "source_request_id": row["source_request_id"],
                "cropped_output_sha256": row["cropped_output_sha256"],
                "task_instruction": row["task_instruction"],
                "evaluator_digest": contract["evaluator_digest"],
            }
        )
        rows.append(row)
    if len(rows) != expected_request_count:
        blockers.append(
            f"evaluator_request_count_expected_{expected_request_count}_got_{len(rows)}"
        )
    if len({str(row["request_id"]) for row in rows}) != len(rows):
        blockers.append("schema_v4_request_ids_not_unique")

    result = {
        key: value
        for key, value in prior.items()
        if key
        not in {"schema_version", "status", "evaluator", "requests", "blockers", "inventory_sha256"}
    }
    result.update(
        {
            "schema_version": "policy_ranking_roboarena_evaluator_inventory.v4",
            "status": "ready" if not blockers else "blocked",
            "evaluator": contract,
            "request_count": len(rows),
            "requests": rows,
            "blockers": sorted(set(blockers)),
            "provider_called": False,
            "data_uploaded": False,
            "outcome_labels_accessed": False,
            "schema_amendment": {
                "schema_version": "policy_ranking_roboarena_schema_amendment.v4",
                "former_inventory_sha256": prior_sha,
                "former_evaluator_digest": evaluator_contract_v3()["evaluator_digest"],
                "failed_run_sha256": failed_run_sha,
                "schema_diagnostic_file_sha256": schema_diagnostic_file_sha256,
                "reason": (
                    "The live strict-schema endpoint rejected uniqueItems at artifact_flags. "
                    "Remove only that provider-invalid keyword and enforce the same uniqueness "
                    "in the local response validator."
                ),
                "changed_fields": [
                    "evaluator.schema_version",
                    "evaluator.output_schema.properties.artifact_flags.uniqueItems",
                    "evaluator.evaluator_digest",
                    "requests[*].evaluator_digest",
                    "requests[*].request_id",
                    "schema_version",
                ],
                "prompt_changed": False,
                "semantic_output_contract_changed": False,
                "sampling_changed": False,
                "scientific_thresholds_changed": False,
                "local_duplicate_artifact_flag_rejection_required": True,
                "former_completed_scientific_rows": 0,
                "former_outcome_labels_accessed": False,
                "provider_called": False,
                "outcome_labels_accessed": False,
            },
        }
    )
    result["inventory_sha256"] = canonical_sha256(result)
    return result


def _usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cached = int(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0)
    cost = (
        (input_tokens - cached) * INPUT_USD_PER_MILLION_TOKENS
        + cached * CACHED_INPUT_USD_PER_MILLION_TOKENS
        + output_tokens * OUTPUT_USD_PER_MILLION_TOKENS
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
    }


def _openai_sdk_version() -> str:
    try:
        return importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable_in_test_environment"


def _score_one(client: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "task_instruction": request["task_instruction"],
        "chronological_sample_positions": list(range(32)),
        "source_frame_indices": [frame["source_frame_index"] for frame in request["frames"]],
        "short_episode_source": request["short_episode_source"],
        "unique_sampled_frame_count": request["unique_sampled_frame_count"],
        "repeated_sample_count": request["repeated_sample_count"],
        "claim_boundary": "Generated-episode progress judgment only; not physical success.",
    }
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": PROMPT + "\n" + json.dumps(metadata, sort_keys=True)}
    ]
    for frame in request["frames"]:
        path = Path(str(frame["path"]))
        payload = path.read_bytes()
        if file_sha256(path) != frame["sha256"]:
            raise ValueError("audited_frame_changed_before_transport")
        content.append(
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii"),
                "detail": "low",
            }
        )
    started = time.monotonic()
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": "blueprint_roboarena_episode_evaluation",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=False,
        extra_headers={"Idempotency-Key": str(request["request_id"])},
    )
    if str(getattr(response, "status", "") or "") != "completed":
        raise ValueError("provider_response_not_completed")
    if not str(getattr(response, "id", "") or ""):
        raise ValueError("provider_response_id_missing")
    payload = json.loads(str(getattr(response, "output_text", "")))
    stable_positions = sorted(
        set(int(value) for value in payload["stable_success_frame_positions"])
    )
    if payload["stable_success_confirmed"] and not any(
        right == left + 1 for left, right in zip(stable_positions, stable_positions[1:])
    ):
        raise ValueError("stable_success_missing_adjacent_sample_positions")
    artifact_flags = set(payload["artifact_flags"])
    if len(artifact_flags) != len(payload["artifact_flags"]):
        raise ValueError("artifact_flags_must_be_unique")
    if "none" in artifact_flags and len(artifact_flags) > 1:
        raise ValueError("artifact_none_cannot_coexist_with_other_flags")
    if payload["abstain"] and not payload["abstention_factors"]:
        raise ValueError("evaluator_abstention_requires_factor")
    deterministic_abstain = bool(request.get("deterministic_safety_abstention_recommended"))
    result = {
        "schema_version": "policy_ranking_roboarena_evaluator_result.v4",
        "request_id": request["request_id"],
        "source_request_id": request["source_request_id"],
        "session_id": request["session_id"],
        "policy_id_internal_only": request["policy_id_internal_only"],
        "evaluator_digest": request["evaluator_digest"],
        "provider": "openai",
        "model": MODEL,
        "response_id": str(getattr(response, "id", "") or ""),
        "response_status": str(getattr(response, "status", "") or ""),
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "structured_response": payload,
        "deterministic_collapse_flags": list(request.get("deterministic_collapse_flags") or []),
        "evaluator_abstain": bool(payload["abstain"]),
        "blueprint_safety_abstain": bool(payload["abstain"] or deterministic_abstain),
        "abstention_sources": [
            source
            for source, active in (
                ("gpt5_mini_evaluator", bool(payload["abstain"])),
                ("deterministic_collapse_audit", deterministic_abstain),
            )
            if active
        ],
        "usage": _usage(response),
        "latency_seconds": time.monotonic() - started,
        "policy_identity_sent_to_provider": False,
        "benchmark_outcomes_sent_to_provider": False,
        "physical_ground_truth_pixels_sent_to_provider": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def _valid_persisted_result(result: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    recorded_sha = str(result.get("result_sha256") or "")
    payload = {key: value for key, value in result.items() if key != "result_sha256"}
    if canonical_sha256(payload) != recorded_sha:
        return False
    if result.get("schema_version") != "policy_ranking_roboarena_evaluator_result.v4":
        return False
    for field in (
        "request_id",
        "source_request_id",
        "session_id",
        "policy_id_internal_only",
        "evaluator_digest",
    ):
        if result.get(field) != request.get(field):
            return False
    return bool(
        result.get("model") == MODEL
        and result.get("response_id")
        and result.get("response_status") == "completed"
        and result.get("policy_identity_sent_to_provider") is False
        and result.get("benchmark_outcomes_sent_to_provider") is False
        and result.get("physical_ground_truth_pixels_sent_to_provider") is False
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--rotation-attestation", required=True)
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    result = run_evaluator_inventory(
        inventory,
        output_root=args.output_root,
        api_key_file=args.api_key_file,
        rotation_attestation_file=args.rotation_attestation,
        max_requests=args.max_requests,
        max_cost_usd=args.max_cost_usd,
        max_workers=args.max_workers,
        source_commit=args.source_commit,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
    return 0 if result["status"] == "completed" else 2


def run_evaluator_inventory(
    inventory: Mapping[str, Any],
    *,
    output_root: str | Path,
    api_key_file: str | Path,
    rotation_attestation_file: str | Path,
    max_requests: int,
    max_cost_usd: float,
    max_workers: int = 4,
    source_commit: str,
) -> dict[str, Any]:
    invocation_started = time.monotonic()
    invocation_started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if os.getenv(GATE_ENV, "").lower() not in {"1", "true", "yes"}:
        return {
            "status": "blocked",
            "blockers": [f"missing_env_{GATE_ENV}"],
            "provider_called": False,
        }
    if inventory.get("status") != "ready":
        return {
            "status": "blocked",
            "blockers": ["evaluator_inventory_not_ready"],
            "provider_called": False,
        }
    inventory_payload = {
        key: value for key, value in inventory.items() if key != "inventory_sha256"
    }
    if canonical_sha256(inventory_payload) != inventory.get("inventory_sha256"):
        return {
            "status": "blocked",
            "blockers": ["evaluator_inventory_digest_invalid"],
            "provider_called": False,
        }
    if (
        inventory.get("schema_version") != "policy_ranking_roboarena_evaluator_inventory.v4"
        or inventory.get("evaluator", {}).get("evaluator_digest")
        != evaluator_contract()["evaluator_digest"]
    ):
        return {
            "status": "blocked",
            "blockers": ["evaluator_inventory_transport_contract_not_v4"],
            "provider_called": False,
        }
    rotation = validate_rotation_attestation(rotation_attestation_file)
    key_path = _secure_file(api_key_file)
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        return {
            "status": "blocked",
            "blockers": ["openai_api_key_file_empty"],
            "provider_called": False,
        }
    if not 1 <= max_workers <= 4:
        raise ValueError("max_workers_must_be_between_1_and_4")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source_commit_must_be_full_lowercase_sha1")
    if not 0.0 < max_cost_usd <= 25.0:
        raise ValueError("max_cost_usd_must_be_positive_and_within_campaign_api_cap")
    inventory_requests = list(inventory.get("requests") or [])
    if not 1 <= max_requests <= len(inventory_requests):
        raise ValueError("max_requests_must_be_within_inventory")
    requests = inventory_requests[:max_requests]
    from openai import OpenAI  # type: ignore[import-not-found]

    client = OpenAI(api_key=key)
    previous_run: dict[str, Any] = {}
    run_path = output / "run.json"
    invalid_previous_run = False
    if run_path.is_file():
        candidate = json.loads(run_path.read_text(encoding="utf-8"))
        candidate_sha = str(candidate.get("run_sha256") or "")
        candidate_payload = {key: value for key, value in candidate.items() if key != "run_sha256"}
        if (
            candidate.get("inventory_sha256") == inventory.get("inventory_sha256")
            and candidate.get("source_commit") == source_commit
            and canonical_sha256(candidate_payload) == candidate_sha
        ):
            previous_run = candidate
        else:
            invalid_previous_run = True
    if invalid_previous_run:
        return {
            "schema_version": "policy_ranking_roboarena_evaluator_run.v4",
            "status": "blocked",
            "inventory_sha256": inventory.get("inventory_sha256"),
            "blockers": ["existing_run_digest_or_inventory_binding_invalid"],
            "provider_called": False,
            "provider_called_status": "not_inferred_from_invalid_state",
            "source_commit": source_commit,
            "outcome_labels_accessed": False,
        }
    completed: list[dict[str, Any]] = []
    resume_blockers: list[str] = []
    request_lookup = {str(row["request_id"]): row for row in requests}
    for request in requests:
        result_path = output / "requests" / str(request["request_id"]) / "result.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if _valid_persisted_result(existing, request):
                completed.append(existing)
            else:
                resume_blockers.append(f"persisted_result_invalid:{request['request_id']}")
    if resume_blockers:
        result = {
            "schema_version": "policy_ranking_roboarena_evaluator_run.v4",
            "status": "blocked",
            "inventory_sha256": inventory.get("inventory_sha256"),
            "selected_request_count": len(requests),
            "completed_request_count": len(completed),
            "results": completed,
            "failures": list(previous_run.get("failures") or []),
            "blockers": resume_blockers,
            "provider_called": bool(previous_run.get("provider_called")),
            "data_uploaded": bool(previous_run.get("data_uploaded")),
            "estimated_cost_usd": float(previous_run.get("estimated_cost_usd") or 0.0),
            "max_cost_usd": max_cost_usd,
            "rotation_attestation": rotation,
            "source_commit": source_commit,
            "credential_path_or_value_persisted": False,
            "outcome_labels_accessed": False,
        }
        result["run_sha256"] = canonical_sha256(result)
        write_json(run_path, result)
        return result
    completed_ids = {row["request_id"] for row in completed}
    selected_ids = set(request_lookup)
    failures = [
        dict(row)
        for row in previous_run.get("failures") or []
        if str(row.get("request_id")) in selected_ids
        and str(row.get("request_id")) not in completed_ids
    ]
    attempt_counts = Counter(str(row.get("request_id")) for row in failures)
    spent = sum(float(row.get("usage", {}).get("estimated_cost_usd") or 0.0) for row in completed)
    spent += sum(float(row.get("conservative_cost_usd") or 0.0) for row in failures)
    pending = [
        row
        for row in requests
        if row["request_id"] not in completed_ids
        and attempt_counts[str(row["request_id"])] < MAX_ATTEMPTS_PER_REQUEST
    ]
    admitted_count = min(
        len(pending),
        int((max(0.0, max_cost_usd - spent) + 1e-12) // MAX_ESTIMATED_REQUEST_USD),
    )
    admitted = pending[:admitted_count]
    if admitted:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_score_one, client, request): request for request in admitted
            }
            for future in as_completed(futures):
                request = futures[future]
                result_path = output / "requests" / str(request["request_id"]) / "result.json"
                try:
                    result = future.result()
                    write_json(result_path, result)
                    completed.append(result)
                    spent += float(result["usage"]["estimated_cost_usd"])
                except Exception as exc:  # pragma: no cover - live provider behavior
                    body = getattr(exc, "body", {}) or {}
                    if not isinstance(body, Mapping):
                        body = {}
                    nested = body.get("error") if isinstance(body.get("error"), Mapping) else {}
                    failures.append(
                        {
                            "request_id": request["request_id"],
                            "attempt_number": attempt_counts[str(request["request_id"])] + 1,
                            "error_type": type(exc).__name__,
                            "http_status": getattr(exc, "status_code", None),
                            "provider_request_id": str(getattr(exc, "request_id", "") or ""),
                            "provider_error_type": body.get("type") or nested.get("type"),
                            "provider_error_code": body.get("code") or nested.get("code"),
                            "provider_error_param": body.get("param") or nested.get("param"),
                            "credential_or_exception_text_persisted": False,
                            "conservative_cost_usd": MAX_ESTIMATED_REQUEST_USD,
                        }
                    )
                    attempt_counts[str(request["request_id"])] += 1
                    spent += MAX_ESTIMATED_REQUEST_USD
    completed.sort(key=lambda row: str(row["request_id"]))
    result: dict[str, Any] = {
        "schema_version": "policy_ranking_roboarena_evaluator_run.v4",
        "status": "completed" if len(completed) == len(requests) else "blocked",
        "inventory_sha256": inventory.get("inventory_sha256"),
        "selected_request_count": len(requests),
        "completed_request_count": len(completed),
        "results": completed,
        "failures": failures,
        "provider_called": bool(previous_run.get("provider_called") or admitted),
        "data_uploaded": bool(previous_run.get("data_uploaded") or admitted),
        "estimated_cost_usd": spent,
        "max_cost_usd": max_cost_usd,
        "rotation_attestation": rotation,
        "source_commit": source_commit,
        "openai_sdk_version": _openai_sdk_version(),
        "invocation_started_at_utc": invocation_started_at,
        "invocation_wall_seconds": time.monotonic() - invocation_started,
        "credential_path_or_value_persisted": False,
        "outcome_labels_accessed": False,
        "max_attempts_per_request": MAX_ATTEMPTS_PER_REQUEST,
        "retry_exhausted_request_ids": sorted(
            request_id
            for request_id, count in attempt_counts.items()
            if count >= MAX_ATTEMPTS_PER_REQUEST and request_id not in completed_ids
        ),
        "blockers": ([] if len(completed) == len(requests) else ["provider_results_incomplete"]),
    }
    result["run_sha256"] = canonical_sha256(result)
    write_json(run_path, result)
    return result


__all__ = [
    "GATE_ENV",
    "MODEL",
    "OUTPUT_SCHEMA",
    "PROMPT",
    "build_evaluator_inventory",
    "evaluator_contract",
    "evaluator_contract_v3",
    "run_evaluator_inventory",
    "supersede_schema_inventory_v4",
    "supersede_transport_inventory_v3",
    "validate_rotation_attestation",
]


if __name__ == "__main__":
    raise SystemExit(main())
