"""Optional FindAll mutations behind the existing paid-admission capability.

No grant issuer, credential discovery, CLI launch, retry, or runtime hook lives
here. The owning controller supplies its durable one-dispatch journal and an
exactly bound grant after checking spend, disclosure and action-time authority.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from . import safe_outbound_http
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .parallel_findall import (
    _MAX_RESPONSE_BYTES,
    RUNS_URL,
    FindAllClient,
    FindAllError,
    _validate_run,
    _validate_run_id,
    prepare_run,
)

PAID_RESOURCE_CLASS = "parallel_findall"
PRICING_VERSION = "parallel-findall-2026-10-03"
PRICING_SOURCE = "https://docs.parallel.ai/getting-started/pricing"
# Versioned provider data, not observed billing or a provider-enforced cap.
_RATES = {
    "preview": ("0.10", "0.00"),
    "base": ("0.25", "0.03"),
    "core": ("2.00", "0.15"),
    "pro": ("10.00", "1.00"),
}


class FindAllSubmissionJournal(Protocol):
    """Adapt the owner's existing durable one-start control, not a new store.

    claim_submission must atomically validate current authority and reserve the
    operation_id with state submission_unresolved BEFORE returning literal True.
    It must reject any previously claimed operation_id, even for a changed body,
    including after crashes, timeouts and HTTP failures. It must not release an
    ambiguous claim. record_created durably binds the exact provider run receipt;
    write failure raises. Neither method receives an API key.
    """

    def claim_submission(self, prepared: Mapping[str, Any]) -> bool: ...

    def record_created(
        self, operation_id: str, allocation_binding_digest: str, run: Mapping[str, Any]
    ) -> None: ...


class FindAllSubmissionUnresolved(FindAllError):
    """Submission may have happened: retain the claim and never retry it."""

    def __init__(self, *, findall_id: str | None = None) -> None:
        self.findall_id = findall_id
        suffix = f":{findall_id}" if findall_id else ""
        super().__init__(f"findall_submission_unresolved{suffix}")


def prepare_submission(
    spec: Mapping[str, Any], *, operation_id: str, maximum_cost_usd: str
) -> dict[str, Any]:
    """Offline review/binding facts. These facts never issue spend authority.

    Admission must recheck this pricing snapshot before issuing the exact grant.
    The owner also checks organization, expiry, disclosure and shared remaining
    allowance, and retains these facts with its existing durable claim receipt.
    """
    body = prepare_run(spec)["body_json"]
    if not isinstance(operation_id, str) or not operation_id.strip():
        raise FindAllError("findall_operation_id_required")
    try:
        if not isinstance(maximum_cost_usd, str):
            raise InvalidOperation
        budget = Decimal(maximum_cost_usd)
    except InvalidOperation:
        raise FindAllError("findall_spend_ceiling_invalid") from None
    fixed, per_match = _RATES[body["generator"]]
    estimate = Decimal(fixed) + Decimal(per_match) * body["match_limit"]
    if not budget.is_finite() or budget < estimate:
        raise FindAllError("findall_spend_ceiling_below_estimated_cost")
    binding = {
        "schema_version": "parallel_findall_submission.v1",
        "resource_class": PAID_RESOURCE_CLASS,
        "operation_id": operation_id,
        "method": "POST",
        "url": RUNS_URL,
        "body_json": body,
        "maximum_cost_usd": format(budget.normalize(), "f"),
        "pricing_version": PRICING_VERSION,
        "pricing_source": PRICING_SOURCE,
        "fixed_cost_usd": fixed,
        "per_match_cost_usd": per_match,
        "estimated_maximum_cost_usd": format(estimate, "f"),
        "provider_enforced_dollar_cap": False,
    }
    raw = json.dumps(binding, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {
        **binding,
        "allocation_binding_digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "execution_authorized": False,
        "network_called": False,
    }


class AdmittedFindAllClient(FindAllClient):
    """Future controller seam. A runtime key alone cannot authorize creation."""

    def _post(self, url: str, body: Mapping[str, Any] | None = None) -> bytes:
        headers = {"x-api-key": self._api_key, "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, method="POST", headers=headers, data=data)
        try:
            response = safe_outbound_http.open_request(
                request,
                policy=safe_outbound_http.pinned_api_policy(
                    RUNS_URL, max_response_bytes=_MAX_RESPONSE_BYTES
                ),
                timeout_seconds=self._timeout_seconds,
                max_response_bytes=_MAX_RESPONSE_BYTES,
            )
            if not 200 <= response.status < 300:
                raise FindAllError(f"findall_http_error:{response.status}")
            if body is None and response.status != 204:
                raise FindAllError("findall_cancel_response_status_invalid")
            return response.body
        except urllib.error.HTTPError as exc:
            raise FindAllError(f"findall_http_error:{exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
            raise FindAllError("findall_transport_failed") from None
        except safe_outbound_http.SafeOutboundHttpError:
            raise FindAllError("findall_outbound_policy_refused") from None

    def create(
        self,
        spec: Mapping[str, Any],
        *,
        operation_id: str,
        maximum_cost_usd: str,
        paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
        journal: FindAllSubmissionJournal,
    ) -> dict[str, Any]:
        """One request after grant and durable claim; ambiguous outcomes stay held.

        There is no documented FindAll create idempotency key/run-list endpoint.
        The journal must never turn a restart into another create. An unresolved
        receipt requires owner/provider reconciliation, not a guessed absence.
        """
        prepared = prepare_submission(
            spec, operation_id=operation_id, maximum_cost_usd=maximum_cost_usd
        )
        binding = prepared["allocation_binding_digest"]
        require_paid_resource_admission_grant(
            paid_resource_admission_grant,
            resource_class=PAID_RESOURCE_CLASS,
            allocation_binding_digest=binding,
            require_allocation_binding=True,
        )
        try:
            claimed = journal.claim_submission(json.loads(json.dumps(prepared)))
        except Exception:  # noqa: BLE001 - expose a stable error without secret-bearing store prose
            raise FindAllError("findall_submission_claim_failed") from None
        if claimed is not True:
            raise FindAllError("findall_submission_already_claimed_or_not_authorized")
        run_id = None
        try:
            payload = json.loads(self._post(RUNS_URL, prepared["body_json"]).decode("utf-8"))
            if not isinstance(payload, dict):
                raise FindAllError("findall_response_must_be_object")
            _validate_run_id(payload.get("findall_id"))
            run_id = payload["findall_id"]
            # Retain even an unexpected/malformed run before validating status:
            # once its ID is known, the owner can read/cancel it without replay.
            journal.record_created(operation_id, binding, json.loads(json.dumps(payload)))
            _validate_run(payload, run_id)
            if payload.get("generator") != prepared["body_json"]["generator"]:
                raise FindAllError("findall_response_generator_mismatch")
            return payload
        except Exception:  # noqa: BLE001 - preserve uncertain submission and known ID without leaking upstream prose
            raise FindAllSubmissionUnresolved(findall_id=run_id) from None

    def cancel(self, findall_id: str) -> None:
        """One explicitly selected cancel POST; 204 means accepted, not a refund.

        Use status(findall_id) afterward to reconcile is_active and terminal
        state. A 409 is an error requiring that read, not a cancellation receipt.
        """
        _validate_run_id(findall_id)
        self._post(f"{RUNS_URL}/{findall_id}/cancel")
