"""OpenAI transport for the post-unseal pairwise evaluator diagnostic."""

from __future__ import annotations

import argparse
import base64
import importlib.metadata
import json
import os
import stat
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import write_json
from .policy_ranking_evaluator_diagnostic import (
    GPT5_MODEL,
    GPT54_MINI_MODEL,
    PAIR_OUTPUT_SCHEMA,
    PAIR_PROMPT,
    PAIR_RESULT_SCHEMA_VERSION,
    complete_graph_diagnostic_protocol,
    diagnostic_protocol,
)
from .policy_ranking_roboarena_calibration import canonical_sha256, file_sha256
from .policy_ranking_roboarena_evaluator_openai import validate_rotation_attestation


GATE_ENV = "BLUEPRINT_ALLOW_ROBOARENA_DIAGNOSTIC_OPENAI"
MODEL_PRICES_STANDARD = {
    GPT5_MODEL: {"input": 1.25, "cached_input": 0.125, "output": 10.0},
    GPT54_MINI_MODEL: {"input": 0.75, "cached_input": 0.075, "output": 4.5},
}
MODEL_ARM_IDS = {
    GPT5_MODEL: "gpt5_oscar_comparability",
    GPT54_MINI_MODEL: "gpt54_mini_challenger",
}
MODEL_MAX_OUTPUT_TOKENS = {GPT5_MODEL: 4000, GPT54_MINI_MODEL: 3000}
MODEL_REASONING_EFFORT = {GPT5_MODEL: "high", GPT54_MINI_MODEL: "medium"}
SUPPORTED_REASONING_EFFORTS = {"minimal", "low", "medium", "high"}


class OpenAIDiagnosticError(ValueError):
    """The OpenAI diagnostic request or evidence is invalid."""


def _secure_file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or stat.S_IMODE(resolved.stat().st_mode) != 0o600:
        raise OpenAIDiagnosticError("credential_file_missing_or_mode_not_0600")
    return resolved


def _image_content(frame: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(frame["path"]))
    if not path.is_file() or file_sha256(path) != frame.get("sha256"):
        raise OpenAIDiagnosticError("audited_frame_changed_before_transport")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "input_image",
        "image_url": "data:image/jpeg;base64," + encoded,
        "detail": "low",
    }


def _request_config(
    *,
    model: str,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    arm_id: str | None = None,
) -> tuple[str, int, str]:
    if model not in MODEL_PRICES_STANDARD:
        raise OpenAIDiagnosticError("unregistered_openai_model")
    effort = reasoning_effort or MODEL_REASONING_EFFORT[model]
    tokens = max_output_tokens or MODEL_MAX_OUTPUT_TOKENS[model]
    resolved_arm = arm_id or MODEL_ARM_IDS[model]
    if effort not in SUPPORTED_REASONING_EFFORTS:
        raise OpenAIDiagnosticError("reasoning_effort_not_supported")
    if isinstance(tokens, bool) or not 1 <= int(tokens) <= 128_000:
        raise OpenAIDiagnosticError("max_output_tokens_invalid")
    if not resolved_arm:
        raise OpenAIDiagnosticError("arm_id_missing")
    return effort, int(tokens), resolved_arm


def build_response_body(
    pair: Mapping[str, Any],
    *,
    model: str,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Build a policy-identity-free Responses API body for one pair."""

    effort, tokens, _ = _request_config(
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )
    metadata = {
        "task_instruction": pair["task_instruction"],
        "episode_a_frame_count": len(pair["episode_a"]["frames"]),
        "episode_b_frame_count": len(pair["episode_b"]["frames"]),
        "claim_boundary": "generated_episode_pair_diagnostic_not_physical_success",
    }
    if metadata["episode_a_frame_count"] != 32 or metadata["episode_b_frame_count"] != 32:
        raise OpenAIDiagnosticError("each_episode_must_have_32_frames")
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": PAIR_PROMPT + "\n" + json.dumps(metadata, sort_keys=True),
        },
        {"type": "input_text", "text": "EPISODE A — frames in chronological order"},
    ]
    content.extend(_image_content(frame) for frame in pair["episode_a"]["frames"])
    content.append(
        {"type": "input_text", "text": "EPISODE B — frames in chronological order"}
    )
    content.extend(_image_content(frame) for frame in pair["episode_b"]["frames"])
    return {
        "model": model,
        "reasoning": {"effort": effort},
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "blueprint_roboarena_pair_evaluation",
                "strict": True,
                "schema": PAIR_OUTPUT_SCHEMA,
            }
        },
        "max_output_tokens": tokens,
        "store": False,
    }


def _usage(response: Any, *, model: str) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    details = getattr(usage, "input_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0)
    prices = MODEL_PRICES_STANDARD[model]
    standard_cost = (
        (input_tokens - cached) * prices["input"]
        + cached * prices["cached_input"]
        + output_tokens * prices["output"]
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output_tokens,
        "standard_cost_usd": standard_cost,
        "projected_batch_cost_same_usage_usd": standard_cost * 0.5,
    }


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("preferred_episode") not in {"A", "B", "tie", "abstain"}:
        raise OpenAIDiagnosticError("structured_preference_invalid")
    if payload.get("preferred_episode") == "abstain" and not payload.get(
        "abstention_factors"
    ):
        raise OpenAIDiagnosticError("structured_abstention_factor_missing")
    for field in PAIR_OUTPUT_SCHEMA["required"]:
        if field not in payload:
            raise OpenAIDiagnosticError(f"structured_field_missing:{field}")


def score_canary(
    client: Any,
    pair: Mapping[str, Any],
    *,
    model: str,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    arm_id: str | None = None,
) -> dict[str, Any]:
    effort, tokens, resolved_arm = _request_config(
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        arm_id=arm_id,
    )
    started = time.monotonic()
    body = build_response_body(
        pair, model=model, reasoning_effort=effort, max_output_tokens=tokens
    )
    request_identity = canonical_sha256(
        {
            "model": model,
            "pair_id": pair["pair_id"],
            "reasoning_effort": effort,
            "max_output_tokens": tokens,
            "prompt_sha256": canonical_sha256(PAIR_PROMPT),
            "schema_sha256": canonical_sha256(PAIR_OUTPUT_SCHEMA),
        }
    )
    response = client.responses.create(
        **body,
        extra_headers={"Idempotency-Key": f"diag-canary-{request_identity}"},
    )
    if getattr(response, "status", None) != "completed" or not getattr(response, "id", None):
        details = getattr(response, "incomplete_details", None)
        if hasattr(details, "model_dump"):
            details = details.model_dump(mode="json")
        result = {
            "schema_version": "policy_ranking_openai_pair_incomplete_canary.v1",
            "pair_id": pair["pair_id"],
            "arm_id": resolved_arm,
            "provider": "openai",
            "model": model,
            "reasoning_effort": effort,
            "max_output_tokens": tokens,
            "response_id": str(getattr(response, "id", "") or ""),
            "response_status": str(getattr(response, "status", "") or ""),
            "incomplete_details": details,
            "usage": _usage(response, model=model),
            "latency_seconds": time.monotonic() - started,
            "transport": "synchronous_schema_canary_only",
            "scientific_valid": False,
            "policy_identity_sent_to_provider": False,
            "physical_outcome_sent_to_provider": False,
            "physical_ground_truth_pixels_sent_to_provider": False,
        }
        result["result_sha256"] = canonical_sha256(result)
        return result
    payload = json.loads(str(getattr(response, "output_text", "")))
    _validate_payload(payload)
    usage = _usage(response, model=model)
    result: dict[str, Any] = {
        "schema_version": PAIR_RESULT_SCHEMA_VERSION,
        "pair_id": pair["pair_id"],
        "arm_id": resolved_arm,
        "provider": "openai",
        "model": model,
        "reasoning_effort": effort,
        "max_output_tokens": tokens,
        "response_id": response.id,
        "response_status": response.status,
        "structured_response": payload,
        "usage": usage,
        "latency_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "transport": "synchronous_schema_canary_only",
        "policy_identity_sent_to_provider": False,
        "physical_outcome_sent_to_provider": False,
        "physical_ground_truth_pixels_sent_to_provider": False,
        "claim_class": "post_unseal_transport_diagnostic_only",
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def run_canary(
    inventory: Mapping[str, Any],
    *,
    model: str,
    api_key_file: str | Path,
    rotation_attestation_file: str | Path,
    output: str | Path,
    source_commit: str,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    arm_id: str | None = None,
) -> dict[str, Any]:
    if os.getenv(GATE_ENV, "").lower() not in {"1", "true", "yes"}:
        return {"status": "blocked", "blockers": [f"missing_env_{GATE_ENV}"]}
    pair_count = inventory.get("pair_count")
    if inventory.get("status") != "ready" or pair_count not in {441, 1323}:
        return {"status": "blocked", "blockers": ["pair_inventory_not_ready_supported_matrix"]}
    protocol = (
        diagnostic_protocol() if pair_count == 441 else complete_graph_diagnostic_protocol()
    )
    if inventory.get("protocol_sha256") != protocol["protocol_sha256"]:
        return {"status": "blocked", "blockers": ["protocol_digest_mismatch"]}
    if pair_count == 1323 and (
        model != GPT5_MODEL
        or reasoning_effort != "medium"
        or max_output_tokens != 4000
        or arm_id != "gpt5_complete_graph"
    ):
        return {"status": "blocked", "blockers": ["complete_graph_gpt5_config_mismatch"]}
    key = _secure_file(api_key_file)
    attestation = validate_rotation_attestation(rotation_attestation_file)
    from openai import OpenAI

    client = OpenAI(api_key=key.read_text(encoding="utf-8").strip())
    result = score_canary(
        client,
        inventory["pairs"][0],
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        arm_id=arm_id,
    )
    projected_batch = (
        result["usage"]["projected_batch_cost_same_usage_usd"] * pair_count
    )
    if pair_count == 441:
        arm_cap = next(
            arm["full_matrix_cap_usd"]
            for arm in protocol["arms"]
            if arm["model"] == model
        )
    else:
        arm_cap = protocol["cost_boundary"]["successor_evaluator_api_cap_usd"]
    report: dict[str, Any] = {
        "schema_version": "policy_ranking_openai_pair_canary.v1",
        "status": (
            "passed"
            if result.get("response_status") == "completed" and projected_batch <= arm_cap
            else "blocked"
        ),
        "model": model,
        "arm_id": result["arm_id"],
        "reasoning_effort": result["reasoning_effort"],
        "max_output_tokens": result["max_output_tokens"],
        "matrix_request_count": pair_count,
        "result": result,
        "projected_full_batch_cost_usd_from_one_canary": projected_batch,
        "frozen_arm_cap_usd": arm_cap,
        "full_matrix_admitted_from_canary_only": False,
        "next_gate": "seven_pair_batch_transport_and_cost_pilot",
        "rotation_attestation": attestation,
        "openai_sdk_version": importlib.metadata.version("openai"),
        "source_commit": source_commit,
        "credential_path_or_value_persisted": False,
        "experiment_media_uploaded": True,
        "physical_ground_truth_pixels_uploaded": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    write_json(Path(output), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_PRICES_STANDARD), required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--rotation-attestation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--reasoning-effort", choices=sorted(SUPPORTED_REASONING_EFFORTS))
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--arm-id")
    args = parser.parse_args(argv)
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    result = run_canary(
        inventory,
        model=args.model,
        api_key_file=args.api_key_file,
        rotation_attestation_file=args.rotation_attestation,
        output=args.output,
        source_commit=args.source_commit,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        arm_id=args.arm_id,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "result"}))
    return 0 if result.get("status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
