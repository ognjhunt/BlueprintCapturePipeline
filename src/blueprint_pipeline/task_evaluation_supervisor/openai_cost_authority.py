"""Independent OpenAI organization-cost metering for paid candidates.

The candidate process never supplies authoritative usage.  This adapter queries
OpenAI's organization Costs endpoint with an exact project and API-key filter.
Because provider cost data can lag execution, immediate settlement remains
``reconciliation_required`` until a conservative reporting window has closed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import stat
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from ..decision_evidence_contracts import canonical_digest
from .candidate_policy import (
    CANDIDATE_COST_RESERVATION_SCHEMA_VERSION,
    CANDIDATE_COST_SETTLEMENT_SCHEMA_VERSION,
    CandidatePolicyError,
)


OPENAI_COST_SNAPSHOT_SCHEMA_VERSION = "openai_organization_cost_snapshot.v1"
OPENAI_COST_SCOPE_ATTESTATION_SCHEMA_VERSION = "openai_candidate_cost_scope_attestation.v1"
OPENAI_COST_AUTHORITY_ID = "openai_organization_costs_api@1"
OPENAI_COSTS_SOURCE_ID = "openai_organization_costs_api"
OPENAI_COSTS_ENDPOINT = "https://api.openai.com/v1/organization/costs"
OPENAI_COSTS_DOCUMENTATION_URL = (
    "https://developers.openai.com/api/reference/resources/admin/subresources/"
    "organization/subresources/usage/methods/costs"
)
_SHA256_PREFIX = "sha256:"
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_PAGES = 200


class OpenAICostAuthorityError(CandidatePolicyError):
    """Raised when provider cost evidence is missing, malformed, or ambiguous."""


def _digest(value: Any, *, field: str) -> str:
    text = str(value or "")
    if len(text) != 71 or not text.startswith(_SHA256_PREFIX):
        raise OpenAICostAuthorityError(f"{field}:invalid_digest")
    try:
        int(text.removeprefix(_SHA256_PREFIX), 16)
    except ValueError as exc:
        raise OpenAICostAuthorityError(f"{field}:invalid_digest") from exc
    return text


def _identifier(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 255 or any(character.isspace() for character in text):
        raise OpenAICostAuthorityError(f"{field}:invalid")
    return text


def _nonnegative_number(value: Any, *, field: str) -> float:
    if value is None or isinstance(value, bool):
        raise OpenAICostAuthorityError(f"{field}:invalid")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OpenAICostAuthorityError(f"{field}:invalid") from exc
    if not math.isfinite(number) or number < 0:
        raise OpenAICostAuthorityError(f"{field}:invalid")
    return number


def _aware_datetime(value: Any, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise OpenAICostAuthorityError(f"{field}:invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OpenAICostAuthorityError(f"{field}:timezone_required")
    return parsed.astimezone(timezone.utc)


def _clock_value(clock: Callable[[], datetime], *, field: str) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise OpenAICostAuthorityError(f"{field}:timezone_required")
    return value.astimezone(timezone.utc)


def openai_cost_authority_binding_digest(
    *,
    provider_id: str,
    paid_resource_class: str,
    project_id: str,
    api_key_id: str,
    scope_attestation_digest: str,
) -> str:
    """Bind the runtime's credential scope to its independent cost authority."""

    return canonical_digest(
        {
            "schema_version": "openai_candidate_cost_authority_binding.v1",
            "metering_source": OPENAI_COSTS_SOURCE_ID,
            "provider_id": _identifier(provider_id, field="provider_id"),
            "paid_resource_class": _identifier(
                paid_resource_class,
                field="paid_resource_class",
            ),
            "project_id": _identifier(project_id, field="openai_project_id"),
            "api_key_id": _identifier(api_key_id, field="openai_api_key_id"),
            "exclusive_scope_attestation_digest": _digest(
                scope_attestation_digest,
                field="openai_scope_attestation_digest",
            ),
        }
    )


def validate_openai_cost_scope_attestation(
    value: Mapping[str, Any],
    *,
    provider_id: str,
    paid_resource_class: str,
    project_id: str,
    api_key_id: str,
) -> dict[str, Any]:
    """Validate the operator-owned exclusive key/project attribution receipt."""

    attestation = dict(value)
    expected_digest = canonical_digest(
        attestation,
        digest_field="scope_attestation_digest",
    )
    if (
        attestation.get("schema_version") != OPENAI_COST_SCOPE_ATTESTATION_SCHEMA_VERSION
        or attestation.get("scope_attestation_digest") != expected_digest
        or attestation.get("status") != "approved"
        or attestation.get("issued_by_agent") is not False
        or not str(attestation.get("operator_id") or "").strip()
        or attestation.get("provider_id") != provider_id
        or attestation.get("paid_resource_class") != paid_resource_class
        or attestation.get("project_id") != project_id
        or attestation.get("api_key_id") != api_key_id
        or attestation.get("exclusive_use") is not True
        or attestation.get("candidate_reported_usage_is_authoritative") is not False
        or attestation.get("proof_effect") != "none"
    ):
        raise OpenAICostAuthorityError("openai_cost_scope_attestation_invalid")
    exclusive_from = _aware_datetime(
        attestation.get("exclusive_from"),
        field="openai_scope_exclusive_from",
    )
    exclusive_until = _aware_datetime(
        attestation.get("exclusive_until"),
        field="openai_scope_exclusive_until",
    )
    if exclusive_until <= exclusive_from:
        raise OpenAICostAuthorityError("openai_cost_scope_attestation_window_invalid")
    return attestation


CostsTransport = Callable[[str, Mapping[str, str], float], Mapping[str, Any]]


@dataclass
class OpenAIOrganizationCostsClient:
    """Minimal, read-only client for the official organization Costs endpoint."""

    project_id: str
    api_key_id: str
    admin_api_key_file: Path
    timeout_seconds: float = 30.0
    transport: CostsTransport | None = None
    wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    def __post_init__(self) -> None:
        self.project_id = _identifier(self.project_id, field="openai_project_id")
        self.api_key_id = _identifier(self.api_key_id, field="openai_api_key_id")
        self.admin_api_key_file = self.admin_api_key_file.expanduser().absolute()
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise OpenAICostAuthorityError("openai_cost_timeout_invalid")

    def _admin_key(self) -> str:
        path = self.admin_api_key_file
        if path.is_symlink() or not path.is_file():
            raise OpenAICostAuthorityError("openai_admin_key_file_missing_or_symlink")
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise OpenAICostAuthorityError("openai_admin_key_file_permissions_too_open")
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise OpenAICostAuthorityError("openai_admin_key_file_unreadable") from exc
        if len(value) < 16 or any(character.isspace() for character in value):
            raise OpenAICostAuthorityError("openai_admin_key_invalid")
        return value

    @staticmethod
    def _default_transport(
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "api.openai.com"
            or parsed.path != "/v1/organization/costs"
        ):
            raise OpenAICostAuthorityError("openai_cost_endpoint_invalid")
        request = Request(url, headers=dict(headers), method="GET")
        try:
            # URL origin, scheme, and exact path are pinned immediately above.
            with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
                payload = response.read(_MAX_RESPONSE_BYTES + 1)
                status_code = int(getattr(response, "status", 200))
        except OSError as exc:
            raise OpenAICostAuthorityError("openai_cost_request_failed") from exc
        if status_code != 200 or len(payload) > _MAX_RESPONSE_BYTES:
            raise OpenAICostAuthorityError("openai_cost_response_invalid")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenAICostAuthorityError("openai_cost_response_invalid") from exc
        if not isinstance(decoded, Mapping):
            raise OpenAICostAuthorityError("openai_cost_response_invalid")
        return decoded

    def snapshot(self, *, start_time: int, end_time: int) -> dict[str, Any]:
        if (
            isinstance(start_time, bool)
            or isinstance(end_time, bool)
            or start_time < 0
            or end_time <= start_time
            or end_time - start_time > 180 * 86_400
        ):
            raise OpenAICostAuthorityError("openai_cost_query_window_invalid")
        headers = {
            "Authorization": f"Bearer {self._admin_key()}",
            "Content-Type": "application/json",
        }
        transport = self.transport or self._default_transport
        base_params: list[tuple[str, str]] = [
            ("start_time", str(start_time)),
            ("end_time", str(end_time)),
            ("bucket_width", "1d"),
            ("limit", str(max(1, min(180, math.ceil((end_time - start_time) / 86_400))))),
            ("project_ids", self.project_id),
            ("api_key_ids", self.api_key_id),
            ("group_by", "project_id"),
            ("group_by", "api_key_id"),
        ]
        page: str | None = None
        seen_pages: set[str] = set()
        page_digests: list[str] = []
        bucket_count = 0
        result_count = 0
        total_cost_usd = 0.0
        for _page_index in range(_MAX_PAGES):
            params = list(base_params)
            if page is not None:
                params.append(("page", page))
            url = f"{OPENAI_COSTS_ENDPOINT}?{urlencode(params)}"
            response = dict(transport(url, headers, self.timeout_seconds))
            page_digests.append(canonical_digest(response))
            if response.get("object") != "page":
                raise OpenAICostAuthorityError("openai_cost_page_schema_invalid")
            buckets = response.get("data")
            if not isinstance(buckets, Sequence) or isinstance(
                buckets,
                (str, bytes, bytearray),
            ):
                raise OpenAICostAuthorityError("openai_cost_page_schema_invalid")
            for bucket in buckets:
                if not isinstance(bucket, Mapping) or bucket.get("object") != "bucket":
                    raise OpenAICostAuthorityError("openai_cost_bucket_schema_invalid")
                bucket_start = bucket.get("start_time")
                bucket_end = bucket.get("end_time")
                if (
                    isinstance(bucket_start, bool)
                    or isinstance(bucket_end, bool)
                    or not isinstance(bucket_start, int)
                    or not isinstance(bucket_end, int)
                    or bucket_end <= bucket_start
                    or bucket_start < start_time
                    or bucket_end > end_time
                ):
                    raise OpenAICostAuthorityError("openai_cost_bucket_window_invalid")
                results = bucket.get("results")
                if not isinstance(results, Sequence) or isinstance(
                    results,
                    (str, bytes, bytearray),
                ):
                    raise OpenAICostAuthorityError("openai_cost_result_schema_invalid")
                bucket_count += 1
                for result in results:
                    if (
                        not isinstance(result, Mapping)
                        or result.get("object") != "organization.costs.result"
                        or result.get("project_id") != self.project_id
                        or result.get("api_key_id") != self.api_key_id
                    ):
                        raise OpenAICostAuthorityError("openai_cost_scope_mismatch")
                    amount = result.get("amount")
                    if (
                        not isinstance(amount, Mapping)
                        or str(amount.get("currency") or "").lower() != "usd"
                    ):
                        raise OpenAICostAuthorityError("openai_cost_currency_invalid")
                    total_cost_usd += _nonnegative_number(
                        amount.get("value"),
                        field="openai_cost_amount",
                    )
                    result_count += 1
            has_more = response.get("has_more")
            if not isinstance(has_more, bool):
                raise OpenAICostAuthorityError("openai_cost_pagination_invalid")
            if not has_more:
                break
            next_page = str(response.get("next_page") or "").strip()
            if not next_page or next_page in seen_pages:
                raise OpenAICostAuthorityError("openai_cost_pagination_invalid")
            seen_pages.add(next_page)
            page = next_page
        else:
            raise OpenAICostAuthorityError("openai_cost_pagination_exhausted")

        observed_at = _clock_value(
            self.wall_clock,
            field="openai_cost_observed_at",
        ).isoformat()
        value: dict[str, Any] = {
            "schema_version": OPENAI_COST_SNAPSHOT_SCHEMA_VERSION,
            "status": "complete",
            "source": OPENAI_COSTS_SOURCE_ID,
            "documentation_url": OPENAI_COSTS_DOCUMENTATION_URL,
            "project_id": self.project_id,
            "api_key_id": self.api_key_id,
            "start_time": start_time,
            "end_time": end_time,
            "observed_at": observed_at,
            "bucket_width": "1d",
            "page_count": len(page_digests),
            "bucket_count": bucket_count,
            "result_count": result_count,
            "total_cost_usd": round(total_cost_usd, 8),
            "source_page_digests": page_digests,
            "raw_admin_key_recorded": False,
            "candidate_reported_usage_used": False,
            "proof_effect": "none",
        }
        value["openai_cost_snapshot_digest"] = canonical_digest(
            value,
            digest_field="openai_cost_snapshot_digest",
        )
        return value


@dataclass
class OpenAIProjectCandidateCostAuthority:
    """Candidate cost authority backed by a dedicated OpenAI key/project scope."""

    client: OpenAIOrganizationCostsClient
    scope_attestation: Mapping[str, Any]
    provider_id: str = "pigey_external_candidate"
    paid_resource_class: str = "openai_api_candidate"
    authority_id: str = OPENAI_COST_AUTHORITY_ID
    attribution_window_seconds: int = 3_600
    reconciliation_delay_seconds: int = 86_400
    wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    def __post_init__(self) -> None:
        self.provider_id = _identifier(self.provider_id, field="provider_id")
        self.paid_resource_class = _identifier(
            self.paid_resource_class,
            field="paid_resource_class",
        )
        self.scope_attestation = validate_openai_cost_scope_attestation(
            self.scope_attestation,
            provider_id=self.provider_id,
            paid_resource_class=self.paid_resource_class,
            project_id=self.client.project_id,
            api_key_id=self.client.api_key_id,
        )
        if (
            isinstance(self.attribution_window_seconds, bool)
            or isinstance(self.reconciliation_delay_seconds, bool)
            or self.attribution_window_seconds < 1
            or self.reconciliation_delay_seconds < 3_600
        ):
            raise OpenAICostAuthorityError("openai_cost_reconciliation_window_invalid")

    @property
    def scope_attestation_digest(self) -> str:
        return str(self.scope_attestation["scope_attestation_digest"])

    @property
    def cost_authority_binding_digest(self) -> str:
        return openai_cost_authority_binding_digest(
            provider_id=self.provider_id,
            paid_resource_class=self.paid_resource_class,
            project_id=self.client.project_id,
            api_key_id=self.client.api_key_id,
            scope_attestation_digest=self.scope_attestation_digest,
        )

    def reserve(
        self,
        *,
        candidate_id: str,
        candidate_evaluation_suite_digest: str,
        authorization_receipt_digest: str,
        max_cost_usd: float,
    ) -> Mapping[str, Any]:
        reserved_cost = _nonnegative_number(max_cost_usd, field="reserved_max_cost_usd")
        now = _clock_value(self.wall_clock, field="openai_cost_reservation_time")
        day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        candidate_runtime_window_end = now + timedelta(seconds=self.attribution_window_seconds)
        attribution_end_day = datetime(
            candidate_runtime_window_end.year,
            candidate_runtime_window_end.month,
            candidate_runtime_window_end.day,
            tzinfo=timezone.utc,
        ) + timedelta(days=1)
        exclusive_from = _aware_datetime(
            self.scope_attestation.get("exclusive_from"),
            field="openai_scope_exclusive_from",
        )
        exclusive_until = _aware_datetime(
            self.scope_attestation.get("exclusive_until"),
            field="openai_scope_exclusive_until",
        )
        if exclusive_from > now or exclusive_until < attribution_end_day:
            raise OpenAICostAuthorityError("openai_cost_scope_attestation_window_insufficient")
        baseline = self.client.snapshot(
            start_time=int(day_start.timestamp()),
            end_time=int(attribution_end_day.timestamp()),
        )
        if float(baseline["total_cost_usd"]) != 0.0:
            raise OpenAICostAuthorityError("openai_cost_scope_baseline_not_zero")
        reconciliation_not_before = attribution_end_day + timedelta(
            seconds=self.reconciliation_delay_seconds
        )
        value: dict[str, Any] = {
            "schema_version": CANDIDATE_COST_RESERVATION_SCHEMA_VERSION,
            "status": "reserved",
            "authority_id": self.authority_id,
            "provider_id": self.provider_id,
            "paid_resource_class": self.paid_resource_class,
            "cost_authority_binding_digest": self.cost_authority_binding_digest,
            "candidate_id": _identifier(candidate_id, field="candidate_id"),
            "candidate_evaluation_suite_digest": _digest(
                candidate_evaluation_suite_digest,
                field="candidate_evaluation_suite_digest",
            ),
            "authorization_receipt_digest": _digest(
                authorization_receipt_digest,
                field="authorization_receipt_digest",
            ),
            "reserved_max_cost_usd": reserved_cost,
            "reserved_at": now.isoformat(),
            "attribution_start_time": int(day_start.timestamp()),
            "candidate_runtime_window_end": candidate_runtime_window_end.isoformat(),
            "attribution_end_time": int(attribution_end_day.timestamp()),
            "reconciliation_not_before": reconciliation_not_before.isoformat(),
            "project_id": self.client.project_id,
            "api_key_id": self.client.api_key_id,
            "exclusive_scope_attestation_digest": self.scope_attestation_digest,
            "baseline_cost_snapshot": baseline,
            "candidate_reported_usage_is_authoritative": False,
            "proof_effect": "none",
        }
        value["cost_reservation_digest"] = canonical_digest(
            value,
            digest_field="cost_reservation_digest",
        )
        return value

    def settle(
        self,
        *,
        reservation: Mapping[str, Any],
        runtime_result: Mapping[str, Any] | None,
        runtime_exception_type: str | None,
    ) -> Mapping[str, Any]:
        value = dict(reservation)
        if (
            value.get("schema_version") != CANDIDATE_COST_RESERVATION_SCHEMA_VERSION
            or value.get("cost_reservation_digest")
            != canonical_digest(value, digest_field="cost_reservation_digest")
            or value.get("authority_id") != self.authority_id
            or value.get("provider_id") != self.provider_id
            or value.get("paid_resource_class") != self.paid_resource_class
            or value.get("cost_authority_binding_digest") != self.cost_authority_binding_digest
            or value.get("project_id") != self.client.project_id
            or value.get("api_key_id") != self.client.api_key_id
            or value.get("exclusive_scope_attestation_digest") != self.scope_attestation_digest
        ):
            raise OpenAICostAuthorityError("openai_cost_reservation_binding_invalid")
        now = _clock_value(self.wall_clock, field="openai_cost_settlement_time")
        not_before = _aware_datetime(
            value.get("reconciliation_not_before"),
            field="reconciliation_not_before",
        )
        status = "reconciliation_required"
        actual_cost_usd: float | None = None
        final = False
        blocker: str | None = "openai_cost_reporting_window_open"
        provider_observed_cost_usd: float | None = None
        final_snapshot: Mapping[str, Any] | None = None
        if now >= not_before:
            final_snapshot = self.client.snapshot(
                start_time=int(value["attribution_start_time"]),
                end_time=int(value["attribution_end_time"]),
            )
            provider_observed_cost_usd = float(final_snapshot["total_cost_usd"])
            reserved_max = float(value["reserved_max_cost_usd"])
            if provider_observed_cost_usd <= reserved_max:
                status = "reconciled"
                actual_cost_usd = provider_observed_cost_usd
                final = True
                blocker = None
            else:
                blocker = "openai_provider_cost_exceeds_reservation"
        settlement: dict[str, Any] = {
            "schema_version": CANDIDATE_COST_SETTLEMENT_SCHEMA_VERSION,
            "status": status,
            "authority_id": self.authority_id,
            "provider_id": self.provider_id,
            "paid_resource_class": self.paid_resource_class,
            "cost_authority_binding_digest": self.cost_authority_binding_digest,
            "candidate_id": value["candidate_id"],
            "cost_reservation_digest": value["cost_reservation_digest"],
            "actual_cost_usd": actual_cost_usd,
            "provider_observed_cost_usd": provider_observed_cost_usd,
            "cost_is_final": final,
            "candidate_reported_cost_accepted": False,
            "runtime_result_observed": runtime_result is not None,
            "runtime_exception_type": runtime_exception_type,
            "reconciled_at": now.isoformat(),
            "reconciliation_blocker": blocker,
            "provider_cost_snapshot": final_snapshot,
            "raw_admin_key_recorded": False,
            "proof_effect": "none",
        }
        settlement["cost_settlement_digest"] = canonical_digest(
            settlement,
            digest_field="cost_settlement_digest",
        )
        return settlement


__all__ = [
    "OPENAI_COST_AUTHORITY_ID",
    "OPENAI_COST_SNAPSHOT_SCHEMA_VERSION",
    "OPENAI_COST_SCOPE_ATTESTATION_SCHEMA_VERSION",
    "OPENAI_COSTS_DOCUMENTATION_URL",
    "OpenAICostAuthorityError",
    "OpenAIOrganizationCostsClient",
    "OpenAIProjectCandidateCostAuthority",
    "openai_cost_authority_binding_digest",
    "validate_openai_cost_scope_attestation",
]
