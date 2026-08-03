"""One narrow transport layer for provider REST/GraphQL calls (C4 step a).

Consolidates the hand-rolled urllib call paths (Vast ``_api_json`` /
``_fetch_text``, Lambda ``_http_json``) onto the existing
``safe_outbound_http`` boundary without changing provider response or error
semantics: JSON parsing and the ``(status, dict)`` return shape are
preserved, and ``urllib.error.HTTPError`` / ``URLError`` propagate exactly
as before so every provider-side ambiguity classifier
(``allocation_outcome_ambiguous``, ``*_side_effects_may_have_occurred``)
keeps working unmodified.

What the boundary adds on every call: https-only scheme enforcement, no
credentials in URLs, per-request host pinning to the resolved endpoint,
cross-origin redirect blocking (same-origin allowed, matching prior
follow-redirect behavior), response-size caps, and timeout clamping.

Retry policy is operation-classified and OFF by default (bespoke provider
retry loops keep their current behavior until call sites adopt
``transport_retry_policy`` explicitly): reads may opt into a bounded,
jittered, allowlisted retry; mutations are refused a retry at this layer —
after an ambiguous mutation the only sanctioned path is
``transport_retry_policy.reconcile_then_retry_mutation``.

This module performs no paid-resource mutation decisions of its own: it is
transport only, behind the grant-gated adapters, which stay behind
``paid_resource_allocator``.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping

from . import safe_outbound_http
from .transport_retry_policy import MutationRetryForbidden

DEFAULT_PROVIDER_MAX_RESPONSE_BYTES = 16 * 1024 * 1024

_READ_METHODS = frozenset({"GET", "HEAD"})


def classify_operation(method: str) -> str:
    """READ (safe to retry with bounds) vs MUTATION (never auto-retried)."""

    return "read" if str(method or "").strip().upper() in _READ_METHODS else "mutation"


def _endpoint_policy(url: str, max_response_bytes: int) -> safe_outbound_http.OutboundHttpPolicy:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return safe_outbound_http.OutboundHttpPolicy(
        allowed_hosts=frozenset({host}) if host else None,
        follow_same_origin_redirects=True,
        max_response_bytes=max_response_bytes,
    )


def provider_json_request(
    *,
    url: str,
    method: str,
    headers: Mapping[str, str],
    body_json: Mapping[str, Any] | None = None,
    timeout_seconds: float,
    max_response_bytes: int = DEFAULT_PROVIDER_MAX_RESPONSE_BYTES,
    read_retry: Callable[[Callable[[], Any]], Callable[[], Any]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """JSON request through the outbound boundary; ``(status, dict)`` shape."""

    operation_kind = classify_operation(method)
    if read_retry is not None and operation_kind != "read":
        raise MutationRetryForbidden(
            f"transport_read_retry_refused_for_mutation:{method}"
        )

    def _once() -> tuple[int, dict[str, Any]]:
        data = (
            json.dumps(body_json).encode("utf-8") if body_json is not None else None
        )
        request = urllib.request.Request(
            url,
            data=data,
            method=str(method).upper(),
            headers=dict(headers),
        )
        response = safe_outbound_http.open_request(
            request,
            policy=_endpoint_policy(url, max_response_bytes),
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
        )
        response_text = response.body.decode("utf-8", errors="replace")
        if not response_text.strip():
            return int(response.status), {}
        parsed = json.loads(response_text)
        return (
            int(response.status),
            dict(parsed) if isinstance(parsed, Mapping) else {"response": parsed},
        )

    if read_retry is not None:
        return read_retry(_once)()
    return _once()


def provider_text_request(
    *,
    url: str,
    timeout_seconds: float,
    max_response_bytes: int = DEFAULT_PROVIDER_MAX_RESPONSE_BYTES,
) -> str:
    """Bounded text fetch through the outbound boundary (read-only)."""

    request = urllib.request.Request(url, method="GET")
    response = safe_outbound_http.open_request(
        request,
        policy=_endpoint_policy(url, max_response_bytes),
        timeout_seconds=timeout_seconds,
        max_response_bytes=max_response_bytes,
    )
    return response.body.decode("utf-8", errors="replace")


__all__ = [
    "DEFAULT_PROVIDER_MAX_RESPONSE_BYTES",
    "classify_operation",
    "provider_json_request",
    "provider_text_request",
]
