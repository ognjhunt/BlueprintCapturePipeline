"""Bounded native Anthropic replacement for Luna's direct vision judgments.

Callers retain their opt-in gates, local media sampling and review-only outputs.
No hosted agent session, server tool, retry, cache, or remote image fetch is used.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .claude_native_transport import (
    ClaudeAuthoringBlocked,
    _admission_lock,
    _post_message,
    _scoped_key,
)
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .inference_reservations import (
    INFERENCE_COMPLETION_SCHEMA_VERSION,
    INFERENCE_RESERVATION_SCHEMA_VERSION,
    InferenceReservationAudit,
)

MODEL = "claude-haiku-5-5"
MAX_OUTPUT_TOKENS = 4096
MAX_INPUT_TOKENS = 100_000
MAXIMUM_COST_USD = 5.0


def replacement_model(value: str) -> str:
    return MODEL if re.fullmatch(r"gpt-6-luna(?:-\d{4}-\d{2}-\d{2})?", value) else value


def anthropic_key() -> tuple[str, str | None]:
    base = os.getenv("ANTHROPIC_BASE_URL", "").strip().rstrip("/")
    if base and base != "https://api.anthropic.com":
        return "", None
    value = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if value:
        return value, "environment"
    if os.getenv("ANTHROPIC_API_KEY_FILE"):
        try:
            return _scoped_key(), "scoped_file"
        except ClaudeAuthoringBlocked:
            return "", None
    return "", None


def _content(blocks: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    result: list[dict[str, Any]] = []
    ceiling = 4096
    for block in blocks:
        if block.get("type") == "input_text" and isinstance(block.get("text"), str):
            result.append({"type": "text", "text": block["text"]})
            ceiling += len(block["text"].encode())
        elif block.get("type") == "input_image":
            match = re.fullmatch(r"data:(image/(?:png|jpeg));base64,(.+)", str(block.get("image_url", "")))
            if not match:
                raise RuntimeError("haiku_local_image_required")
            data = base64.b64decode(match[2], validate=True)
            if len(data) > 5_000_000:
                raise RuntimeError("haiku_image_bytes_exceeded")
            with Image.open(io.BytesIO(data)) as image:
                if image.format != {"image/png": "PNG", "image/jpeg": "JPEG"}[match[1]] or not 0 < image.width <= 8000 or not 0 < image.height <= 8000:
                    raise RuntimeError("haiku_image_invalid")
                image.verify()
            result.append({"type": "image", "source": {"type": "base64", "media_type": match[1], "data": match[2]}})
            ceiling += 4784
        else:
            raise RuntimeError("haiku_content_invalid")
    return result, ceiling


def _context_ceiling(value: Any) -> int:
    """Count text/tool framing conservatively and validate every local image."""
    if isinstance(value, Mapping):
        if "cache_control" in value:
            raise RuntimeError("haiku_cache_not_admitted")
        if value.get("type") == "image":
            source = value.get("source", {})
            if not isinstance(source, Mapping) or source.get("type") != "base64":
                raise RuntimeError("haiku_local_image_required")
            _, ceiling = _content([{"type": "input_image", "image_url":
                f"data:{source.get('media_type')};base64,{source.get('data')}"}])
            return ceiling - 4096
        return sum(len(str(key).encode()) + _context_ceiling(item) + 8 for key, item in value.items())
    if isinstance(value, list):
        return sum(_context_ceiling(item) + 8 for item in value)
    return len(json.dumps(value, ensure_ascii=True, allow_nan=False).encode())


def bounded_message(*, payload: Mapping[str, Any], api_key: str, audit_root: Path,
                    capability: str, input_limit: int = MAX_INPUT_TOKENS,
                    maximum_cost_usd: float = MAXIMUM_COST_USD) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Reserve one native call durably; preserve unknown outcomes and prohibit replay."""
    output_limit = payload.get("max_tokens")
    if (not api_key or payload.get("model") != MODEL or type(output_limit) is not int
            or not 1 <= output_limit <= 16000 or not 1 <= input_limit <= MAX_INPUT_TOKENS
            or not 0 < maximum_cost_usd <= MAXIMUM_COST_USD):
        raise RuntimeError("haiku_bounded_request_required")
    if _context_ceiling(payload) + 4096 > input_limit:
        raise RuntimeError("haiku_input_or_cache_budget_exhausted")
    audit = InferenceReservationAudit(run_root=audit_root / "haiku_vision_budget", run_id="haiku-vision-judges")
    policy = {"status": "disabled", "policy_digest": canonical_digest({"provider": "anthropic", "model": MODEL, "cache_control": "absent"})}
    identity = {"run_id": audit.run_id, "capability": capability, "model": MODEL,
                "provider": "anthropic", "input_digest": canonical_digest(payload),
                "max_turns": 1, "max_output_tokens": output_limit,
                "cache_policy_digest": policy["policy_digest"]}
    # Reserve the expensive tier even though admission caps input at its threshold.
    projected = (input_limit * 0.5 + output_limit * 2.5) * 1.1 / 1_000_000
    reservation = {"schema_version": INFERENCE_RESERVATION_SCHEMA_VERSION, **identity,
                   "reservation_id": canonical_digest(identity), "projected_max_cost_usd": projected,
                   "cache_policy": policy, "input_token_ceiling": input_limit,
                   "breakpoint_digests": []}
    reservation["inference_reservation_digest"] = canonical_digest(reservation)
    with _admission_lock(audit):
        manifest = audit.manifest()
        if float(manifest.get("reserved_max_cost_usd", 0)) + projected > maximum_cost_usd:
            raise RuntimeError("haiku_cost_budget_exhausted")
        audit.record_reservation(reservation)
    # Fixed Anthropic endpoint, no redirects/retries. A transport failure keeps the reservation.
    response = _post_message(payload, api_key)
    write_json(audit.root / "responses" / f"{reservation['reservation_id'].removeprefix('sha256:')}.json", response)
    if response.get("model") != MODEL:
        raise RuntimeError("haiku_response_wrong_model")
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise RuntimeError("haiku_usage_missing")  # noqa: TRY004 - invalid provider receipt
    inp, out = usage.get("input_tokens"), usage.get("output_tokens")
    if (type(inp) is not int or type(out) is not int or not 0 <= inp <= input_limit
            or not 0 <= out <= output_limit
            or any(usage.get(key, 0) not in (0, None) for key in ("cache_creation_input_tokens", "cache_read_input_tokens"))):
        raise RuntimeError("haiku_usage_invalid")
    cost = (inp * 0.1 + out * 0.5) * (1 if usage.get("inference_geo") == "global" else 1.1) / 1_000_000
    completion = {"schema_version": INFERENCE_COMPLETION_SCHEMA_VERSION,
                  "reservation_id": reservation["reservation_id"], "run_id": audit.run_id,
                  "capability": capability, "model": MODEL, "provider": "anthropic",
                  "cache_policy": policy, "breakpoint_digests": [],
                  "projected_max_cost_usd": projected, "reconciled_actual_cost_usd": cost,
                  "released_reservation_usd": projected - cost,
                  "usage": dict(usage), "provider_response_id": response.get("id"),
                  "stop_reason": response.get("stop_reason")}
    completion["inference_completion_digest"] = canonical_digest(completion)
    with _admission_lock(audit):
        audit.record_completion(completion)
    receipt = {"provider": "anthropic", "model": MODEL, "response_id": response.get("id"),
               "cache_policy": policy, "reservation_id": reservation["reservation_id"],
               "usage": {**usage, "estimated_total_cost_usd": cost,
                         "cost_status": "model_pricing_estimate_not_official_billing"}}
    return response, receipt


def judge_json(*, system: str, content: Sequence[Mapping[str, Any]],
               api_key: str, audit_root: Path, capability: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not api_key:
        raise RuntimeError("missing_anthropic_api_key")
    native_content, ceiling = _content(content)
    ceiling += len(system.encode())
    if ceiling > MAX_INPUT_TOKENS:
        raise RuntimeError("haiku_input_budget_exhausted")
    payload = {"model": MODEL, "max_tokens": MAX_OUTPUT_TOKENS, "system": system,
               "messages": [{"role": "user", "content": native_content}],
               "output_config": {"effort": "medium"}}
    response, receipt = bounded_message(payload=payload, api_key=api_key,
        audit_root=audit_root, capability=capability)
    if response.get("stop_reason") != "end_turn":
        raise RuntimeError("haiku_response_incomplete_or_wrong_model")
    blocks = response.get("content")
    if not isinstance(blocks, list) or any(not isinstance(block, Mapping) or block.get("type") not in {"text", "thinking", "redacted_thinking"} for block in blocks):
        raise RuntimeError("haiku_response_content_invalid")
    text = "\n".join(str(block.get("text", "")) for block in blocks if block.get("type") == "text")
    result = json.loads(text)
    if not isinstance(result, dict):
        raise RuntimeError("haiku_json_object_required")  # noqa: TRY004 - invalid provider output
    return result, receipt
