"""Secret-clean Pipeline projection and signed WebApp sync for OpenAI call usage."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_webapp_sync import (
    PipelineSyncTokenError,
    load_pipeline_sync_token,
)
from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url


PACKET_SCHEMA_VERSION = "blueprint_openai_inference_usage_packet.v1"
DEFAULT_WEBAPP_ENDPOINT = (
    "https://tryblueprint.io/api/internal/pipeline/openai-inference-usage"
)
WEBAPP_URL_ENV = "PIPELINE_OPENAI_INFERENCE_USAGE_WEBAPP_URL"


class OpenAIInferenceUsageError(ValueError):
    """The retained usage or cache-policy evidence is incomplete."""


def _cache_key_digest(value: object) -> str | None:
    key = str(value or "").strip()
    if not key:
        return None
    return "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest()


def _number(value: object, *, field: str, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenAIInferenceUsageError(f"openai_inference_usage_{field}_invalid")
    result = float(value)
    if result < 0:
        raise OpenAIInferenceUsageError(f"openai_inference_usage_{field}_invalid")
    return result


def _call_usage(
    *,
    packet_run_id: str,
    capability: str,
    round_index: int,
    usage: Mapping[str, Any],
) -> dict[str, Any]:
    policy = usage.get("cache_policy")
    if not isinstance(policy, Mapping):
        raise OpenAIInferenceUsageError("openai_inference_usage_cache_policy_missing")
    economics = policy.get("economics")
    economics = economics if isinstance(economics, Mapping) else {}
    input_tokens = int(_number(usage.get("input_tokens"), field="input_tokens") or 0)
    cached_tokens = int(_number(usage.get("cached_tokens"), field="cached_tokens") or 0)
    cache_write_tokens = int(
        _number(usage.get("cache_write_tokens"), field="cache_write_tokens") or 0
    )
    uncached_tokens = int(
        _number(usage.get("uncached_input_tokens"), field="uncached_input_tokens") or 0
    )
    if cached_tokens + cache_write_tokens + uncached_tokens != input_tokens:
        raise OpenAIInferenceUsageError("openai_inference_usage_partition_invalid")
    status = str(policy.get("status") or "")
    raw_cache_key = policy.get("cache_key")
    retained_cache_key_digest = str(policy.get("cache_key_digest") or "").strip()
    cache_key_digest = (
        retained_cache_key_digest
        if retained_cache_key_digest
        else _cache_key_digest(raw_cache_key)
    )
    if (status == "enabled") != (cache_key_digest is not None):
        raise OpenAIInferenceUsageError("openai_inference_usage_cache_key_invalid")
    response_id = str(usage.get("provider_response_id") or "").strip() or None
    request_id = str(usage.get("provider_request_id") or "").strip() or None
    call_identity = {
        "run_id": packet_run_id,
        "round_index": round_index,
        "capability": capability,
        "provider_response_id": response_id,
        "usage_receipt_digest": usage.get("usage_receipt_digest"),
        "policy_digest": policy.get("policy_digest"),
    }
    call: dict[str, Any] = {
        "schema_version": "openai_prompt_cache_usage.v1",
        "call_id": canonical_digest(call_identity),
        "capability": capability,
        "provider": "openai",
        "model": str(usage.get("model") or policy.get("model_family") or ""),
        "cache_family": str(policy.get("family") or ""),
        "cache_policy_status": status,
        "cache_decision_reason": str(policy.get("decision_reason") or ""),
        "cache_key_digest": cache_key_digest,
        "prompt_contract_version": str(policy.get("contract_version") or ""),
        "stable_prefix_digest": str(policy.get("stable_prefix_digest") or ""),
        "breakpoint_digests": list(usage.get("breakpoint_digests") or []),
        "policy_digest": str(policy.get("policy_digest") or ""),
        "privacy_scope": str(policy.get("privacy_scope") or ""),
        "processing_region": str(policy.get("processing_region") or ""),
        "reusable_prefix_tokens": int(
            _number(
                economics.get("stable_prefix_tokens", 0),
                field="reusable_prefix_tokens",
            )
            or 0
        ),
        "dynamic_suffix_tokens": uncached_tokens,
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "uncached_input_tokens": uncached_tokens,
        "output_tokens": int(
            _number(usage.get("output_tokens"), field="output_tokens") or 0
        ),
        "reasoning_tokens": int(
            _number(usage.get("reasoning_tokens"), field="reasoning_tokens") or 0
        ),
        "cache_hit_ratio": float(usage.get("cache_hit_ratio") or 0.0),
        "uncached_input_cost_usd": _number(
            usage.get("uncached_input_cost_usd"), field="uncached_input_cost", nullable=True
        ),
        "cache_write_cost_usd": _number(
            usage.get("cache_write_cost_usd"), field="cache_write_cost", nullable=True
        ),
        "cached_read_cost_usd": _number(
            usage.get("cached_read_cost_usd"), field="cached_read_cost", nullable=True
        ),
        "output_cost_usd": _number(
            usage.get("output_cost_usd"), field="output_cost", nullable=True
        ),
        "estimated_total_cost_usd": _number(
            usage.get("estimated_total_cost_usd"), field="estimated_total_cost", nullable=True
        ),
        "estimated_cost_without_caching_usd": _number(
            usage.get("estimated_cost_without_caching_usd"),
            field="estimated_cost_without_caching",
            nullable=True,
        ),
        "estimated_savings_usd": (
            float(usage["estimated_savings_usd"])
            if isinstance(usage.get("estimated_savings_usd"), (int, float))
            and not isinstance(usage.get("estimated_savings_usd"), bool)
            else None
        ),
        "cost_status": str(usage.get("cost_status") or "model_pricing_unknown"),
        "provider_response_id": response_id,
        "provider_request_id": request_id,
        "usage_detail_status": "complete",
        "dynamic_content_before_breakpoint": False,
        "raw_prompt_recorded": False,
        "raw_secret_values_recorded": False,
        "usage_receipt_digest": "",
    }
    call["usage_receipt_digest"] = canonical_digest(
        call, digest_field="usage_receipt_digest"
    )
    return call


def build_placement_inference_usage_packet(
    *,
    placement_receipt: Mapping[str, Any],
    packet_run_id: str,
    launch_id: str | None,
    source_commit: str,
) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for index, round_record in enumerate(placement_receipt.get("rounds") or []):
        if not isinstance(round_record, Mapping):
            continue
        for usage_field, capability in (
            ("proposal_usage", "task_aware_robot_placement_proposal"),
            ("visual_review_usage", "robot_placement_visual_review"),
        ):
            usage = round_record.get(usage_field)
            if isinstance(usage, Mapping):
                calls.append(
                    _call_usage(
                        packet_run_id=packet_run_id,
                        capability=capability,
                        round_index=index,
                        usage=usage,
                    )
                )
    if not calls:
        raise OpenAIInferenceUsageError("openai_inference_usage_calls_missing")
    packet: dict[str, Any] = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "run_id": packet_run_id,
        "launch_id": launch_id,
        "source_commit": source_commit,
        "source_receipt_digest": str(placement_receipt.get("receipt_digest") or ""),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "calls": calls,
        "raw_prompts_recorded": False,
        "raw_secret_values_recorded": False,
        "packet_digest": "",
    }
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    return packet


def sync_inference_usage_to_webapp(
    *,
    packet: Mapping[str, Any],
    endpoint_url: str | None = None,
    token: str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    common = {
        "schema_version": "blueprint_openai_inference_usage_sync_result.v1",
        "run_id": packet.get("run_id"),
        "launch_id": packet.get("launch_id"),
        "source_commit": packet.get("source_commit"),
        "packet_digest": packet.get("packet_digest"),
        "call_count": len(packet.get("calls") or []),
    }
    if packet.get("packet_digest") != canonical_digest(
        packet, digest_field="packet_digest"
    ):
        return {**common, "status": "failed", "reason": "packet_digest_invalid"}
    try:
        resolved_token = load_pipeline_sync_token(token=token)
    except PipelineSyncTokenError:
        return {**common, "status": "skipped", "reason": "sync_not_configured"}
    resolved_url = str(
        endpoint_url or os.getenv(WEBAPP_URL_ENV) or DEFAULT_WEBAPP_ENDPOINT
    ).strip()
    try:
        url = validated_https_sync_url(resolved_url)
    except ValueError:
        return {**common, "status": "failed", "reason": "sync_url_invalid"}
    body = json.dumps(dict(packet), separators=(",", ":")).encode("utf-8")
    outbound = urllib_request.Request(
        url,
        data=body,
        headers=_pipeline_sync_headers(resolved_token, body),
        method="POST",
    )
    try:
        with urllib_request.urlopen(  # nosec B310 - URL is validated HTTPS
            outbound, timeout=max(0.1, timeout_seconds)
        ) as response:
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        return {**common, "status": "failed", "reason": f"http_error:{exc.code}"}
    except urllib_error.URLError as exc:
        return {**common, "status": "failed", "reason": f"url_error:{exc.reason}"}
    except (TimeoutError, ValueError) as exc:
        return {**common, "status": "failed", "reason": type(exc).__name__.lower()}
    try:
        response = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {**common, "status": "failed", "reason": "invalid_json"}
    if not isinstance(response, Mapping) or any(
        response.get(field) != common[field]
        for field in ("run_id", "launch_id", "source_commit", "packet_digest", "call_count")
    ):
        return {**common, "status": "failed", "reason": "response_binding_mismatch"}
    if response.get("schema_version") != "blueprint_openai_inference_usage_ingest_receipt.v1":
        return {**common, "status": "failed", "reason": "response_schema_mismatch"}
    if response.get("status") not in {"created", "replayed"}:
        return {**common, "status": "failed", "reason": "response_status_invalid"}
    return {**common, "status": "succeeded", "response": dict(response)}


__all__ = [
    "DEFAULT_WEBAPP_ENDPOINT",
    "OpenAIInferenceUsageError",
    "PACKET_SCHEMA_VERSION",
    "build_placement_inference_usage_packet",
    "sync_inference_usage_to_webapp",
]
