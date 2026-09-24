"""Opt-in, one-request Claude Opus authoring calls under Blueprint's local budget.

This is a local model adapter, not Claude Managed Agents. Managed Agents does
not expose a per-request output ceiling, so its session budget cannot prove the
hard authoring limit. The CAD/Blender executor and validators stay local.
"""
from __future__ import annotations

import base64
import binascii
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Callable, Mapping
from urllib import request as urllib_request

from PIL import Image
from pydantic import ValidationError

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_supervisor.agents_sdk import (
    AgentsSDKAgentSpec, AgentsSDKInvocationResult,
)
from .task_evaluation_supervisor.inference_reservations import (
    INFERENCE_COMPLETION_SCHEMA_VERSION, INFERENCE_RESERVATION_SCHEMA_VERSION,
    InferenceReservationAudit,
)

MODEL = "claude-opus-5-5"
PROVIDER = "anthropic"
KEY_FILE_ENV = "ANTHROPIC_API_KEY_FILE"
_API_URL = "https://api.anthropic.com/v1/messages"
_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")
# Official Opus 5.5 prices (USD per million tokens, 2026-09-22). No cache,
# server tools, fast mode, batch, or provider-side agent loop is requested.
_INPUT_RATE = 4.0
_OUTPUT_RATE = 20.0
_US_GEO_MULTIPLIER = 1.1
# Claude 4.7+ high-resolution images are capped at 4,784 visual tokens each.
_MAX_IMAGE_TOKENS = 4_784
_FIXED_INPUT_MARGIN_TOKENS = 4_096
_MAX_REQUEST_BYTES = 30_000_000  # under the provider's 32 MB request ceiling
_MODEL_INPUT_CONTEXT_TOKENS = 1_000_000
_UNSUPPORTED_OUTPUT_CONSTRAINTS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "maxItems", "uniqueItems",
})


class ClaudeAuthoringBlocked(RuntimeError):
    """The provider, rights, spend, or output boundary failed closed."""


@dataclass(frozen=True)
class ClaudeAuthoringConfig:
    run_id: str
    maximum_cost_usd: float
    maximum_calls: int
    allow_live_invocation: bool = False
    inference_geo: str = "us"

    def __post_init__(self) -> None:
        if (not self.run_id or not math.isfinite(self.maximum_cost_usd)
                or not 0 < self.maximum_cost_usd <= 7
                or not 1 <= self.maximum_calls <= 32
                or self.inference_geo != "us"):
            raise ClaudeAuthoringBlocked("claude_authoring_configuration_invalid")


def _scoped_key() -> str:
    named = os.environ.get(KEY_FILE_ENV, "")
    path = Path(named)
    if not named or not path.is_absolute():
        raise ClaudeAuthoringBlocked("claude_key_file_missing")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ClaudeAuthoringBlocked("claude_key_file_missing") from exc
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        mode = os.fstat(stream.fileno()).st_mode
        if not stat.S_ISREG(mode) or mode & 0o077:
            raise ClaudeAuthoringBlocked("claude_key_file_permissions_invalid")
        value = stream.read(16_384).strip()
        too_long = bool(stream.read(1))
    if too_long:
        raise ClaudeAuthoringBlocked("claude_key_file_invalid")
    if not value:
        raise ClaudeAuthoringBlocked("claude_key_file_empty")
    return value


def _post_message(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    if len(body) > _MAX_REQUEST_BYTES:
        raise ClaudeAuthoringBlocked("claude_request_bytes_exceeded")
    req = urllib_request.Request(_API_URL, data=body, method="POST", headers={
        "content-type": "application/json", "anthropic-version": "2023-06-01",
        "x-api-key": key,
    })
    # urllib has no automatic model retry. Unknown outcomes retain the full
    # reservation and must be reconciled before any same-identity replay.
    with urllib_request.urlopen(req, timeout=600) as response:
        return json.load(response)


@contextmanager
def _admission_lock(audit: InferenceReservationAudit):
    """Serialize Claude reservations sharing one attempt ledger."""
    audit.run_root.mkdir(parents=True, exist_ok=True)
    with (audit.run_root / ".claude_authoring_admission.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _source_image(value: str) -> tuple[dict[str, Any], int]:
    prefix = "data:image/png;base64,"
    if not value.startswith(prefix):
        raise ClaudeAuthoringBlocked("claude_image_format_unsupported")
    encoded = value[len(prefix):]
    try:
        data = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(data)) as picture:
            if picture.format != "PNG" or not 0 < picture.width <= 8000 or not 0 < picture.height <= 8000:
                raise ClaudeAuthoringBlocked("claude_image_invalid")
            picture.verify()
    except (ValueError, OSError, binascii.Error) as exc:
        raise ClaudeAuthoringBlocked("claude_image_invalid") from exc
    if len(data) > 7_000_000:
        raise ClaudeAuthoringBlocked("claude_image_too_large")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                       "data": encoded}}, _MAX_IMAGE_TOKENS


def _provider_output_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only constraints unsupported by Claude's JSON-output grammar.

    The complete Pydantic schema stays in the prompt and is enforced again on
    the returned value. Nested object schemas must prohibit extra properties.
    """
    result = {key: value for key, value in schema.items()
              if key not in _UNSUPPORTED_OUTPUT_CONSTRAINTS}
    if "properties" in result:
        if result.get("additionalProperties", False) is not False:
            raise ClaudeAuthoringBlocked("claude_output_schema_open_object")
        result["additionalProperties"] = False
        result["properties"] = {key: _provider_output_schema(value)
                                for key, value in result["properties"].items()}
    if "$defs" in result:
        result["$defs"] = {key: _provider_output_schema(value)
                           for key, value in result["$defs"].items()}
    if "items" in result:
        result["items"] = _provider_output_schema(result["items"])
    for name in ("anyOf", "allOf"):
        if name in result:
            result[name] = [_provider_output_schema(value) for value in result[name]]
    if result.get("minItems") not in (None, 0, 1):
        result.pop("minItems")
    return result


def _payload(spec: AgentsSDKAgentSpec, input_value: str | list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    if (spec.model != MODEL or spec.max_turns != 1 or spec.tool_bindings
            or not 1 <= spec.max_output_tokens <= 20_000
            or spec.max_input_tokens is None or not 1 <= spec.max_input_tokens <= 80_000):
        raise ClaudeAuthoringBlocked("claude_authoring_spec_invalid")
    schema = json.dumps(spec.output_type.model_json_schema(), sort_keys=True, ensure_ascii=True)
    system = (spec.instructions + "\nReturn exactly one JSON object matching this schema, with no markdown: "
              + schema)
    if spec.stable_developer_prefix:
        system += "\n" + spec.stable_developer_prefix
    content: list[dict[str, Any]] = []
    image_tokens = 0
    text_bytes = len(system.encode())
    if isinstance(input_value, str):
        content.append({"type": "text", "text": input_value})
        text_bytes += len(input_value.encode())
    elif isinstance(input_value, list) and len(input_value) == 1 and input_value[0].get("role") == "user":
        for block in input_value[0].get("content", []):
            if block.get("type") == "input_text" and isinstance(block.get("text"), str):
                content.append({"type": "text", "text": block["text"]})
                text_bytes += len(block["text"].encode())
            elif block.get("type") == "input_image" and isinstance(block.get("image_url"), str):
                image, tokens = _source_image(block["image_url"])
                content.append(image)
                image_tokens += tokens
            else:
                raise ClaudeAuthoringBlocked("claude_authoring_content_invalid")
    else:
        raise ClaudeAuthoringBlocked("claude_authoring_input_invalid")
    if not content:
        raise ClaudeAuthoringBlocked("claude_authoring_input_empty")
    # A UTF-8 byte is a conservative text-token ceiling. Image reservation uses
    # the provider's published visual-token maximum plus fixed prompt margin.
    input_ceiling = text_bytes + image_tokens + _FIXED_INPUT_MARGIN_TOKENS
    if input_ceiling > spec.max_input_tokens:
        raise ClaudeAuthoringBlocked("claude_input_token_ceiling_exceeded")
    payload = {"model": MODEL, "max_tokens": spec.max_output_tokens,
               "inference_geo": "us", "system": system,
               "messages": [{"role": "user", "content": content}],
               "output_config": {"effort": spec.reasoning_effort or "medium",
                                 "format": {"type": "json_schema",
                                            "schema": _provider_output_schema(
                                                spec.output_type.model_json_schema())}}}
    if len(json.dumps(payload, ensure_ascii=True).encode()) > _MAX_REQUEST_BYTES:
        raise ClaudeAuthoringBlocked("claude_request_bytes_exceeded")
    return payload, input_ceiling


def _cost(usage: Mapping[str, Any]) -> float:
    if any(usage.get(name, 0) not in (0, None) for name in (
            "cache_creation_input_tokens", "cache_read_input_tokens")):
        raise ClaudeAuthoringBlocked("claude_unrequested_cache_usage")
    inp, out = usage.get("input_tokens"), usage.get("output_tokens")
    if (type(inp) is not int or type(out) is not int or inp < 0 or out < 0):
        raise ClaudeAuthoringBlocked("claude_usage_missing")
    return round((inp * _INPUT_RATE + out * _OUTPUT_RATE)
                 * _US_GEO_MULTIPLIER / 1_000_000, 9)


def _contains_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        return any(key in forbidden or _contains_key(part, forbidden)
                   for key, part in value.items())
    if isinstance(value, list):
        return any(_contains_key(part, forbidden) for part in value)
    return False


class ClaudeOpusAuthoringInvoker:
    """Implements the existing authoring invoker shape without changing defaults.

    ``verify_authority`` must validate the signed scene intent and provider
    terms, then return its digest-bound projection. No callback means no call.
    """

    model = MODEL

    def __init__(self, config: ClaudeAuthoringConfig, *, audit: InferenceReservationAudit,
                 verify_authority: Callable[[str, str], Mapping[str, Any]] | None = None,
                 send: Callable[[Mapping[str, Any], str], Mapping[str, Any]] = _post_message):
        self.config, self.audit = config, audit
        self.verify_authority, self.send = verify_authority, send

    def invoke_tool_turn(self, *, capability: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """One native Messages tool turn with the same hard admission ledger.

        The caller owns a durable transcript and must retain the entire raw
        content array, including signed thinking blocks, before tool execution.
        """
        if not self.config.allow_live_invocation or self.verify_authority is None:
            raise ClaudeAuthoringBlocked("claude_live_authority_missing")
        output_config = payload.get("output_config")
        output_format = output_config.get("format") if isinstance(output_config, dict) else None
        if (not capability or payload.get("model") != MODEL
                or payload.get("inference_geo") != "us"
                or type(payload.get("max_tokens")) is not int
                or not 1 <= payload["max_tokens"] <= 12_000
                or not isinstance(payload.get("messages"), list)
                or not isinstance(payload.get("tools"), list)
                or not isinstance(output_config, dict)
                or output_config.get("effort") != "medium"
                or set(output_config) - {"effort", "format"}
                or (output_format is not None and (not isinstance(output_format, dict)
                    or set(output_format) != {"type", "schema"}
                    or output_format.get("type") != "json_schema"
                    or not isinstance(output_format.get("schema"), dict)))
                or payload.get("thinking", {"type": "adaptive", "display": "omitted"})
                    != {"type": "adaptive", "display": "omitted"}
                or payload.get("tool_choice", {"type": "auto", "disable_parallel_tool_use": True})
                    != {"type": "auto", "disable_parallel_tool_use": True}
                or _contains_key(payload, {"cache_control", "mcp_servers",
                                           "context_management", "speed"})
                or any(not isinstance(tool, dict)
                       or set(tool) != {"name", "description", "input_schema", "strict"}
                       or tool["strict"] is not True
                       or not isinstance(tool["name"], str)
                       or not isinstance(tool["description"], str)
                       or not isinstance(tool["input_schema"], dict)
                       for tool in payload["tools"])):
            raise ClaudeAuthoringBlocked("claude_tool_payload_invalid")
        encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode()
        if len(encoded) > _MAX_REQUEST_BYTES:
            raise ClaudeAuthoringBlocked("claude_request_bytes_exceeded")
        input_digest = canonical_digest({"request": payload})
        authority = dict(self.verify_authority(self.config.run_id, input_digest))
        if (authority.get("run_id") != self.config.run_id
                or PROVIDER not in authority.get("allowed_providers", [])
                or authority.get("private_provider_processing_allowed") is not True
                or authority.get("provider_training_allowed") is not False
                or not _SHA.fullmatch(str(authority.get("authority_digest") or ""))
                or not _SHA.fullmatch(str(authority.get("provider_terms_digest") or ""))):
            raise ClaudeAuthoringBlocked("claude_provider_authority_invalid")
        projected = round((_MODEL_INPUT_CONTEXT_TOKENS * _INPUT_RATE
                           + payload["max_tokens"] * _OUTPUT_RATE)
                          * _US_GEO_MULTIPLIER / 1_000_000, 9)
        key = _scoped_key()
        identity = {"run_id": self.config.run_id, "capability": capability,
                    "model": MODEL, "provider": PROVIDER,
                    "input_digest": input_digest, "max_turns": 1,
                    "max_output_tokens": payload["max_tokens"]}
        cache_policy = {"status": "disabled", "policy_digest": canonical_digest({
            "provider": PROVIDER, "model": MODEL, "cache_control": "absent"})}
        reservation_id = canonical_digest({**identity,
            "cache_policy_digest": cache_policy["policy_digest"]})
        reservation = {"schema_version": INFERENCE_RESERVATION_SCHEMA_VERSION,
            "reservation_id": reservation_id, **identity,
            "input_token_ceiling": _MODEL_INPUT_CONTEXT_TOKENS,
            "projected_max_cost_usd": projected,
            "cache_policy": cache_policy, "cache_policy_digest": cache_policy["policy_digest"],
            "breakpoint_digests": {}, "authority_digest": authority["authority_digest"],
            "provider_terms_digest": authority["provider_terms_digest"],
            "billing_status": "worst_case_reserved_before_provider_call", "proof_effect": "none"}
        reservation["inference_reservation_digest"] = canonical_digest(
            reservation, digest_field="inference_reservation_digest")
        with _admission_lock(self.audit):
            manifest = self.audit.manifest()
            if manifest["reservation_count"] >= self.config.maximum_calls:
                raise ClaudeAuthoringBlocked("claude_call_cap_exhausted")
            if manifest["reserved_max_cost_usd"] + projected > self.config.maximum_cost_usd:
                raise ClaudeAuthoringBlocked("claude_spend_cap_exhausted")
            self.audit.record_reservation(reservation)
        try:
            response = dict(self.send(payload, key))
        except Exception as exc:
            raise ClaudeAuthoringBlocked("claude_provider_outcome_unknown") from exc
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            raise ClaudeAuthoringBlocked("claude_usage_missing")
        actual = _cost(usage)
        if (actual > projected + 1e-9 or usage["input_tokens"] > _MODEL_INPUT_CONTEXT_TOKENS
                or usage["output_tokens"] > payload["max_tokens"]):
            raise ClaudeAuthoringBlocked("claude_provider_usage_exceeds_reserved_maximum")
        valid = (response.get("model") == MODEL and isinstance(response.get("id"), str)
                 and response.get("stop_reason") in {"tool_use", "end_turn"}
                 and isinstance(response.get("content"), list)
                 and all(isinstance(block, dict) and block.get("type") in
                         {"thinking", "redacted_thinking", "text", "tool_use"}
                         for block in response["content"]))
        if valid:
            allowed_names = {tool["name"] for tool in payload["tools"]}
            tool_ids = [block.get("id") for block in response["content"]
                        if block["type"] == "tool_use"]
            valid = (all(isinstance(block.get("signature"), str) and block["signature"]
                         for block in response["content"] if block["type"] == "thinking")
                     and all(isinstance(block.get("text"), str)
                             for block in response["content"] if block["type"] == "text")
                     and all(isinstance(block.get("data"), str) and block["data"]
                             for block in response["content"] if block["type"] == "redacted_thinking")
                     and all(block.get("name") in allowed_names
                             and isinstance(block.get("input"), dict)
                             for block in response["content"] if block["type"] == "tool_use")
                     and all(isinstance(call_id, str) and call_id for call_id in tool_ids)
                     and len(set(tool_ids)) == len(tool_ids))
        completion = {"schema_version": INFERENCE_COMPLETION_SCHEMA_VERSION,
            "reservation_id": reservation_id, "run_id": self.config.run_id,
            "capability": capability, "model": MODEL, "provider": PROVIDER,
            "status": "completed" if valid else "invalid_model_output",
            "provider_outcome": response.get("stop_reason"),
            "provider_response_id": response.get("id"), "usage": dict(usage),
            "cost_basis": "provider_reported_usage_list_price",
            "reconciled_actual_cost_usd": actual, "observed_actual_cost_usd": actual,
            "projected_max_cost_usd": projected,
            "released_reservation_usd": max(0.0, projected - actual),
            "cache_policy": cache_policy, "breakpoint_digests": {},
            "authority_digest": authority["authority_digest"],
            "provider_terms_digest": authority["provider_terms_digest"],
            "response_output_digest": canonical_digest({"content": response.get("content")}),
            "proof_effect": "none"}
        completion["inference_completion_digest"] = canonical_digest(
            completion, digest_field="inference_completion_digest")
        self.audit.record_completion(completion)
        if not valid:
            raise ClaudeAuthoringBlocked("claude_model_output_invalid")
        return response

    def invoke(self, spec: AgentsSDKAgentSpec,
               input_value: str | list[dict[str, Any]]) -> AgentsSDKInvocationResult:
        if not self.config.allow_live_invocation or self.verify_authority is None:
            raise ClaudeAuthoringBlocked("claude_live_authority_missing")
        if spec.run_id != self.config.run_id:
            raise ClaudeAuthoringBlocked("claude_run_identity_mismatch")
        payload, input_ceiling = _payload(spec, input_value)
        input_digest = canonical_digest({"request": payload})
        authority = dict(self.verify_authority(spec.run_id, input_digest))
        if (authority.get("run_id") != spec.run_id
                or PROVIDER not in authority.get("allowed_providers", [])
                or authority.get("private_provider_processing_allowed") is not True
                or authority.get("provider_training_allowed") is not False
                or not _SHA.fullmatch(str(authority.get("authority_digest") or ""))
                or not _SHA.fullmatch(str(authority.get("provider_terms_digest") or ""))):
            raise ClaudeAuthoringBlocked("claude_provider_authority_invalid")
        # The byte/image estimate is an admission check, not a billing proof.
        # Reserve the model's entire published 1M input window so that even a
        # provider token-count difference cannot push this call past the cap.
        projected = round((_MODEL_INPUT_CONTEXT_TOKENS * _INPUT_RATE
                           + spec.max_output_tokens * _OUTPUT_RATE)
                          * _US_GEO_MULTIPLIER / 1_000_000, 9)
        key = _scoped_key()
        identity = {"run_id": spec.run_id, "capability": str(spec.capability), "model": MODEL,
                    "input_digest": input_digest, "max_turns": 1,
                    "max_output_tokens": spec.max_output_tokens, "provider": PROVIDER}
        if spec.reasoning_effort is not None:
            identity["reasoning_effort"] = spec.reasoning_effort
        cache_policy = {"status": "disabled", "policy_digest": canonical_digest({
            "provider": PROVIDER, "model": MODEL, "cache_control": "absent"})}
        reservation_id = canonical_digest({**identity,
            "cache_policy_digest": cache_policy["policy_digest"]})
        reservation = {"schema_version": INFERENCE_RESERVATION_SCHEMA_VERSION,
            "reservation_id": reservation_id, **identity,
            "input_token_ceiling": _MODEL_INPUT_CONTEXT_TOKENS,
            "submitted_input_token_estimate_ceiling": input_ceiling,
            "projected_max_cost_usd": projected,
            "caller_input_digest": canonical_digest(
                {"input_text": input_value} if isinstance(input_value, str)
                else {"input": input_value}),
            "cache_policy": cache_policy, "cache_policy_digest": cache_policy["policy_digest"],
            "breakpoint_digests": {}, "authority_digest": authority["authority_digest"],
            "provider_terms_digest": authority["provider_terms_digest"],
            "billing_status": "worst_case_reserved_before_provider_call", "proof_effect": "none"}
        reservation["inference_reservation_digest"] = canonical_digest(
            reservation, digest_field="inference_reservation_digest")
        with _admission_lock(self.audit):
            manifest = self.audit.manifest()
            if manifest["reservation_count"] >= self.config.maximum_calls:
                raise ClaudeAuthoringBlocked("claude_call_cap_exhausted")
            if manifest["reserved_max_cost_usd"] + projected > self.config.maximum_cost_usd:
                raise ClaudeAuthoringBlocked("claude_spend_cap_exhausted")
            self.audit.record_reservation(reservation)
        started = time.monotonic()
        try:
            response = dict(self.send(payload, key))
        except Exception as exc:
            # A timeout or transport error can follow a provider dispatch.
            # Keep the full reservation as unknown and never replay its ID.
            raise ClaudeAuthoringBlocked("claude_provider_outcome_unknown") from exc
        latency = time.monotonic() - started
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            raise ClaudeAuthoringBlocked("claude_usage_missing")
        actual = _cost(usage)
        if (actual > projected + 1e-9 or usage["input_tokens"] > _MODEL_INPUT_CONTEXT_TOKENS
                or usage["output_tokens"] > spec.max_output_tokens):
            raise ClaudeAuthoringBlocked("claude_provider_usage_exceeds_reserved_maximum")
        blocks = response.get("content")
        text_blocks = [row.get("text") for row in blocks or []
                       if isinstance(row, Mapping) and row.get("type") == "text"]
        valid_response = (response.get("model") == MODEL
                          and response.get("stop_reason") == "end_turn"
                          and isinstance(response.get("id"), str)
                          and len(text_blocks) == 1 and isinstance(text_blocks[0], str))
        output = None
        invalid_output_reason = None if valid_response else "response_envelope"
        validation_error_paths: list[str] = []
        if valid_response:
            try:
                output = spec.output_type.model_validate(json.loads(text_blocks[0]))
            except json.JSONDecodeError:
                invalid_output_reason = "json_syntax"
                valid_response = False
            except ValidationError as exc:
                invalid_output_reason = "schema_validation"
                validation_error_paths = [".".join(map(str, row["loc"]))
                                          for row in exc.errors()[:8]]
                valid_response = False
            except ValueError:
                invalid_output_reason = "schema_validation"
                valid_response = False
        completion = {"schema_version": INFERENCE_COMPLETION_SCHEMA_VERSION,
            "reservation_id": reservation["reservation_id"], "run_id": spec.run_id,
            "capability": str(spec.capability), "model": MODEL, "provider": PROVIDER,
            "status": "completed" if valid_response else "invalid_structured_output",
            "provider_outcome": response.get("stop_reason"),
            "provider_response_id": response.get("id"), "usage": dict(usage),
            "cost_basis": "provider_reported_usage_list_price",
            "reconciled_actual_cost_usd": actual, "observed_actual_cost_usd": actual,
            "projected_max_cost_usd": projected,
            "released_reservation_usd": max(0.0, projected - actual),
            "cache_policy": cache_policy, "breakpoint_digests": {},
            "authority_digest": authority["authority_digest"],
            "provider_terms_digest": authority["provider_terms_digest"],
            "structured_output_digest": (canonical_digest(output.model_dump(mode="json"))
                                         if output is not None else None),
            "proof_effect": "none"}
        if not valid_response:
            completion["invalid_output_reason"] = invalid_output_reason
            completion["validation_error_paths"] = validation_error_paths
        completion["inference_completion_digest"] = canonical_digest(
            completion, digest_field="inference_completion_digest")
        self.audit.record_completion(completion)
        if not valid_response or output is None:
            raise ClaudeAuthoringBlocked("claude_output_invalid")
        return AgentsSDKInvocationResult(output=output, provider=PROVIDER, model=MODEL,
            sdk_version="anthropic-messages-rest-v1", latency_seconds=latency,
            usage=dict(usage), cost_usd=actual,
            cost_status="provider_reported_usage_list_price", trace_id=response["id"])


def budgeted_claude_invoker(*, root: Path, run_id: str, maximum_cost_usd: float,
                            maximum_calls: int,
                            verify_authority: Callable[[str, str], Mapping[str, Any]]):
    """Build the opt-in model adapter on the same durable authoring ledger."""
    audit = InferenceReservationAudit(run_root=root, run_id=run_id)
    return ClaudeOpusAuthoringInvoker(ClaudeAuthoringConfig(
        run_id=run_id, maximum_cost_usd=maximum_cost_usd,
        maximum_calls=maximum_calls, allow_live_invocation=True),
        audit=audit, verify_authority=verify_authority), audit
