"""No-inference Opus 5.5 availability and published-price preflight.

This GET request does not perform authoring, reserve a scene grant, or prove
that a later Messages call will fit the scene's $7 model cap. It checks the
configured credential against the specific model before any CPU rental.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Mapping
from urllib import request as urllib_request

from .claude_opus_authoring_invoker import (
    ClaudeAuthoringBlocked, MODEL, _INPUT_RATE, _MODEL_INPUT_CONTEXT_TOKENS,
    _OUTPUT_RATE, _US_GEO_MULTIPLIER, _scoped_key,
)
from .decision_evidence_contracts import canonical_digest

_MODEL_URL = f"https://api.anthropic.com/v1/models/{MODEL}"
_PUBLISHED_PRICE_URL = "https://platform.claude.com/docs/en/models/opus-5-5/overview"
_AUTHOR_OUTPUT_CEILING = 12_000


def _get_model(key: str) -> Mapping[str, Any]:
    request = urllib_request.Request(_MODEL_URL, method="GET", headers={
        "anthropic-version": "2023-06-01", "x-api-key": key,
    })
    with urllib_request.urlopen(request, timeout=15) as response:
        return json.load(response)


def preflight(*, fetch: Callable[[str], Mapping[str, Any]] = _get_model) -> dict[str, Any]:
    """Return metadata-only evidence or fail closed without leaking a key."""
    key = _scoped_key()
    try:
        value = dict(fetch(key))
    except Exception as exc:
        raise ClaudeAuthoringBlocked("claude_model_metadata_unavailable") from exc
    capabilities = value.get("capabilities")
    if (value.get("id") != MODEL
            or type(value.get("max_input_tokens")) is not int
            or value["max_input_tokens"] < _MODEL_INPUT_CONTEXT_TOKENS
            or type(value.get("max_tokens")) is not int
            or value["max_tokens"] < _AUTHOR_OUTPUT_CEILING
            or not isinstance(capabilities, dict)
            or not all((capabilities.get(name) or {}).get("supported") is True
                       for name in ("image_input", "structured_outputs"))
            or ((capabilities.get("thinking") or {}).get("types") or {}).get(
                "adaptive", {}).get("supported") is not True
            or ((capabilities.get("effort") or {}).get("medium") or {}).get(
                "supported") is not True):
        raise ClaudeAuthoringBlocked("claude_model_capability_unavailable")
    quote = round((_MODEL_INPUT_CONTEXT_TOKENS * _INPUT_RATE
                   + _AUTHOR_OUTPUT_CEILING * _OUTPUT_RATE)
                  * _US_GEO_MULTIPLIER / 1_000_000, 9)
    if quote > 7:
        raise ClaudeAuthoringBlocked("claude_published_maximum_quote_exceeds_stage_cap")
    return {
        "schema_version": "claude_opus_model_preflight.v1",
        "status": "model_metadata_available_no_inference",
        "model": MODEL,
        "model_metadata_digest": canonical_digest(value),
        "max_input_tokens": value["max_input_tokens"],
        "max_output_tokens": value["max_tokens"],
        "maximum_one_call_reservation_usd": quote,
        "authoring_model_cap_usd": 7,
        "price_basis": "published_list_price_2026-09-22_us_inference",
        "price_source": _PUBLISHED_PRICE_URL,
        "billing_observed": False,
        "provider_calls_performed": 0,
    }


def main() -> int:
    try:
        print(json.dumps(preflight(), sort_keys=True))
        return 0
    except ClaudeAuthoringBlocked as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
