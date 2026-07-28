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
MAX_OUTPUT_TOKENS = 3000
MODEL_REASONING_EFFORT = {GPT5_MODEL: "high", GPT54_MINI_MODEL: "medium"}


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


def build_response_body(pair: Mapping[str, Any], *, model: str) -> dict[str, Any]:
    """Build a policy-identity-free Responses API body for one pair."""

    if model not in MODEL_PRICES_STANDARD:
        raise OpenAIDiagnosticError("unregistered_openai_model")
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
        "reasoning": {"effort": MODEL_REASONING_EFFORT[model]},
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "blueprint_roboarena_pair_evaluation",
                "strict": True,
                "schema": PAIR_OUTPUT_SCHEMA,
            }
        },
        "max_output_tokens": MAX_OUTPUT_TOKENS,
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


def score_canary(client: Any, pair: Mapping[str, Any], *, model: str) -> dict[str, Any]:
    started = time.monotonic()
    body = build_response_body(pair, model=model)
    response = client.responses.create(
        **body,
        extra_headers={"Idempotency-Key": f"diag-canary-{model}-{pair['pair_id']}"},
    )
    if getattr(response, "status", None) != "completed" or not getattr(
        response, "id", None
    ):
        raise OpenAIDiagnosticError("canary_response_not_completed")
    payload = json.loads(str(getattr(response, "output_text", "")))
    _validate_payload(payload)
    usage = _usage(response, model=model)
    result: dict[str, Any] = {
        "schema_version": PAIR_RESULT_SCHEMA_VERSION,
        "pair_id": pair["pair_id"],
        "arm_id": MODEL_ARM_IDS[model],
        "provider": "openai",
        "model": model,
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
) -> dict[str, Any]:
    if os.getenv(GATE_ENV, "").lower() not in {"1", "true", "yes"}:
        return {"status": "blocked", "blockers": [f"missing_env_{GATE_ENV}"]}
    if inventory.get("status") != "ready" or inventory.get("pair_count") != 441:
        return {"status": "blocked", "blockers": ["pair_inventory_not_ready_441"]}
    protocol = diagnostic_protocol()
    if inventory.get("protocol_sha256") != protocol["protocol_sha256"]:
        return {"status": "blocked", "blockers": ["protocol_digest_mismatch"]}
    key = _secure_file(api_key_file)
    attestation = validate_rotation_attestation(rotation_attestation_file)
    from openai import OpenAI

    client = OpenAI(api_key=key.read_text(encoding="utf-8").strip())
    result = score_canary(client, inventory["pairs"][0], model=model)
    projected_batch = result["usage"]["projected_batch_cost_same_usage_usd"] * 441
    arm_cap = next(
        arm["full_matrix_cap_usd"] for arm in protocol["arms"] if arm["model"] == model
    )
    report: dict[str, Any] = {
        "schema_version": "policy_ranking_openai_pair_canary.v1",
        "status": "passed" if projected_batch <= arm_cap else "blocked",
        "model": model,
        "arm_id": MODEL_ARM_IDS[model],
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
    args = parser.parse_args(argv)
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    result = run_canary(
        inventory,
        model=args.model,
        api_key_file=args.api_key_file,
        rotation_attestation_file=args.rotation_attestation,
        output=args.output,
        source_commit=args.source_commit,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "result"}))
    return 0 if result.get("status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
