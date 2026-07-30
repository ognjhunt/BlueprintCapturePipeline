"""Fail-closed Phase 3 pre-authorized recovery execution controller."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..decision_evidence_contracts import canonical_digest
from ..paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)


RECOVERY_ACTION_SCHEMA_VERSION = "task_evaluation_recovery_action.v1"
RECOVERY_RESULT_SCHEMA_VERSION = "task_evaluation_recovery_result.v1"
MAX_RECOVERY_COST_USD = 100.0
MAX_RECOVERY_RETRIES = 3
MAX_RECOVERY_TTL_SECONDS = 3_600.0
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_NON_RETRYABLE_FAILURES = {
    "budget_exhaustion",
    "invalid_scientific_output",
    "permanent_incompatibility",
    "rights_or_authority_missing",
}


class RecoveryControlError(ValueError):
    """Raised before execution when recovery authority is invalid."""


class RecoveryAdapter(Protocol):
    provider_id: str
    paid_resource_class: str
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None

    def execute(
        self,
        *,
        action_id: str,
        immutable_commit_sha: str,
        immutable_input_digests: Sequence[str],
        timeout_seconds: float,
        max_cost_usd: float,
    ) -> Mapping[str, Any]: ...

    def teardown(self) -> Mapping[str, Any]: ...


def _parse_time(value: Any, *, field: str) -> datetime:
    text = str(value or "").replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RecoveryControlError(f"{field}:invalid") from exc
    if parsed.tzinfo is None:
        raise RecoveryControlError(f"{field}:timezone_required")
    return parsed.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _digest(value: Any, *, field: str) -> str:
    text = str(value or "")
    if not _SHA256_DIGEST.fullmatch(text):
        raise RecoveryControlError(f"{field}:invalid")
    return text


@dataclass(frozen=True)
class PreauthorizedRecoveryPolicy:
    run_id: str
    authorization_receipt: Mapping[str, Any]
    immutable_commit_sha: str
    immutable_input_digests: tuple[str, ...]
    allowed_provider_ids: tuple[str, ...]
    allowed_action_ids: tuple[str, ...]
    watchdog_seconds: float
    teardown_required: bool = True

    def __post_init__(self) -> None:
        receipt = dict(self.authorization_receipt)
        expected = canonical_digest(receipt, digest_field="authorization_receipt_digest")
        if receipt.get("authorization_receipt_digest") != expected:
            raise RecoveryControlError("recovery_authorization_receipt_digest_mismatch")
        if receipt.get("approved") is not True or receipt.get("issued_by_agent") is not False:
            raise RecoveryControlError("recovery_not_operator_authorized")
        if receipt.get("granted_tool_id") != "execute_preauthorized_recovery":
            raise RecoveryControlError("recovery_wrong_tool_authority")
        if receipt.get("run_id") != self.run_id:
            raise RecoveryControlError("recovery_authorization_run_mismatch")
        if not _COMMIT_SHA.fullmatch(self.immutable_commit_sha):
            raise RecoveryControlError("recovery_commit_sha_invalid")
        normalized_inputs = tuple(sorted({_digest(row, field="input_digest") for row in self.immutable_input_digests}))
        receipt_inputs = tuple(sorted(str(row) for row in receipt.get("immutable_input_digests") or []))
        if (
            not normalized_inputs
            or self.immutable_input_digests != normalized_inputs
            or normalized_inputs != receipt_inputs
        ):
            raise RecoveryControlError("recovery_immutable_inputs_not_authorized")
        if not self.allowed_provider_ids or not self.allowed_action_ids:
            raise RecoveryControlError("recovery_allowlist_missing")
        issued = _parse_time(receipt.get("issued_at"), field="recovery_issued_at")
        expires = _parse_time(receipt.get("expires_at"), field="recovery_expires_at")
        granted_ttl = float(receipt.get("granted_ttl_seconds") or 0)
        granted_cost = float(receipt.get("granted_max_cost_usd") or 0)
        granted_retries = int(receipt.get("granted_retry_count") or 0)
        if (
            not math.isfinite(granted_ttl)
            or granted_ttl <= 0
            or not math.isfinite(granted_cost)
            or granted_cost < 0
            or granted_cost > MAX_RECOVERY_COST_USD
            or granted_retries < 0
            or granted_retries > MAX_RECOVERY_RETRIES
            or granted_ttl > MAX_RECOVERY_TTL_SECONDS
            or expires <= issued
            or (expires - issued).total_seconds() > granted_ttl
        ):
            raise RecoveryControlError("recovery_receipt_expiry_exceeds_ttl")
        receipt_providers = tuple(sorted(str(row) for row in receipt.get("granted_provider_ids") or []))
        receipt_actions = tuple(sorted(str(row) for row in receipt.get("granted_action_ids") or []))
        if tuple(sorted(set(self.allowed_provider_ids))) != receipt_providers:
            raise RecoveryControlError("recovery_provider_allowlist_not_authorized")
        if tuple(sorted(set(self.allowed_action_ids))) != receipt_actions:
            raise RecoveryControlError("recovery_action_allowlist_not_authorized")
        if not math.isfinite(self.watchdog_seconds) or self.watchdog_seconds <= 0:
            raise RecoveryControlError("recovery_watchdog_invalid")
        if self.watchdog_seconds > float(receipt.get("granted_ttl_seconds") or 0):
            raise RecoveryControlError("recovery_watchdog_exceeds_ttl")
        if not self.teardown_required:
            raise RecoveryControlError("recovery_teardown_required")

    @property
    def receipt(self) -> dict[str, Any]:
        return dict(self.authorization_receipt)


class PreauthorizedRecoveryController:
    """Execute only allowlisted recovery while preserving every attempt."""

    def __init__(
        self,
        policy: PreauthorizedRecoveryPolicy,
        adapters: Sequence[RecoveryAdapter],
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.policy = policy
        self._adapters = {adapter.provider_id: adapter for adapter in adapters}
        if len(self._adapters) != len(adapters):
            raise RecoveryControlError("recovery_provider_duplicate")
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._attempts: list[dict[str, Any]] = []
        self._spent_usd = 0.0
        self._cost_reconciliation_required = False
        self._terminal_failure = False

    @property
    def max_cost_usd(self) -> float:
        return float(self.policy.receipt.get("granted_max_cost_usd") or 0.0)

    @property
    def attempt_ledger(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(row) for row in self._attempts)

    def execute(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self._cost_reconciliation_required:
            raise RecoveryControlError("recovery_cost_reconciliation_required")
        if self._terminal_failure:
            raise RecoveryControlError("recovery_terminal_failure_requires_stop")
        if "now" in arguments:
            raise RecoveryControlError("recovery_agent_clock_forbidden")
        action_id = str(arguments.get("action_id") or "")
        provider_id = str(arguments.get("provider_id") or "")
        commit_sha = str(arguments.get("immutable_commit_sha") or "")
        input_digests = tuple(sorted(str(row) for row in arguments.get("input_digests") or []))
        try:
            projected_cost = float(arguments.get("projected_cost_usd") or 0.0)
        except (TypeError, ValueError) as exc:
            raise RecoveryControlError("recovery_projected_cost_invalid") from exc
        failure_type = str(arguments.get("failure_type") or "")
        now_value = self._wall_clock()
        if now_value.tzinfo is None:
            raise RecoveryControlError("recovery_controller_clock_timezone_required")
        now = now_value.astimezone(timezone.utc)
        receipt = self.policy.receipt
        expires = _parse_time(receipt.get("expires_at"), field="recovery_expires_at")
        issued = _parse_time(receipt.get("issued_at"), field="recovery_issued_at")
        if now < issued:
            raise RecoveryControlError("recovery_authority_not_yet_valid")
        if now >= expires:
            raise RecoveryControlError("recovery_authority_expired")
        effective_timeout_seconds = min(
            self.policy.watchdog_seconds,
            max(0.0, (expires - now).total_seconds()),
        )
        if effective_timeout_seconds <= 0:
            raise RecoveryControlError("recovery_authority_expired")
        if action_id not in self.policy.allowed_action_ids:
            raise RecoveryControlError("recovery_action_not_allowed")
        if provider_id not in self.policy.allowed_provider_ids:
            raise RecoveryControlError("recovery_provider_not_allowed")
        if commit_sha != self.policy.immutable_commit_sha:
            raise RecoveryControlError("recovery_commit_sha_mismatch")
        if input_digests != self.policy.immutable_input_digests:
            raise RecoveryControlError("recovery_input_digest_mismatch")
        if (
            not math.isfinite(projected_cost)
            or projected_cost < 0
            or self._spent_usd + projected_cost > self.max_cost_usd
        ):
            raise RecoveryControlError("recovery_spend_ceiling_exceeded")
        max_attempts = int(receipt.get("granted_retry_count") or 0) + 1
        if len(self._attempts) >= max_attempts:
            raise RecoveryControlError("recovery_retry_ceiling_exceeded")
        if failure_type in _NON_RETRYABLE_FAILURES:
            raise RecoveryControlError("recovery_failure_non_retryable")
        adapter = self._adapters.get(provider_id)
        if adapter is None:
            raise RecoveryControlError("recovery_provider_adapter_unavailable")
        try:
            require_paid_resource_admission_grant(
                adapter.paid_resource_admission_grant,
                resource_class=adapter.paid_resource_class,
            )
        except (AttributeError, PaidResourceAdmissionBlocked) as exc:
            raise RecoveryControlError("recovery_shared_paid_admission_missing_or_invalid") from exc

        started = self._monotonic()
        raw: Mapping[str, Any] = {}
        teardown: Mapping[str, Any] = {"status": "not_attempted"}
        status = "failed"
        typed_failure: dict[str, Any] | None = None
        actual_cost: float | None = 0.0
        cost_reconciliation_required = False
        try:
            raw = adapter.execute(
                action_id=action_id,
                immutable_commit_sha=commit_sha,
                immutable_input_digests=input_digests,
                timeout_seconds=effective_timeout_seconds,
                max_cost_usd=min(projected_cost, self.max_cost_usd - self._spent_usd),
            )
            elapsed = max(0.0, self._monotonic() - started)
            finished_at = self._wall_clock()
            if finished_at.tzinfo is None:
                raise RecoveryControlError("recovery_controller_clock_timezone_required")
            authority_expired_during_execution = finished_at.astimezone(timezone.utc) >= expires
            if elapsed > effective_timeout_seconds or authority_expired_during_execution:
                status = "timed_out"
                typed_failure = {
                    "failure_type": (
                        "authority_expired_during_execution"
                        if authority_expired_during_execution
                        else "watchdog_timeout"
                    ),
                    "retryable": False,
                }
            else:
                raw_status = str(raw.get("status") or "")
                status = "completed" if raw_status == "completed" else "failed"
                if "cost_usd" not in raw:
                    actual_cost = None
                    status = "failed"
                    cost_reconciliation_required = True
                    typed_failure = {
                        "failure_type": "provider_cost_missing",
                        "retryable": False,
                    }
                else:
                    try:
                        actual_cost = float(raw["cost_usd"])
                    except (TypeError, ValueError):
                        actual_cost = None
                        status = "failed"
                        cost_reconciliation_required = True
                        typed_failure = {
                            "failure_type": "provider_cost_invalid",
                            "retryable": False,
                        }
                if typed_failure is None and actual_cost is not None and (
                    not math.isfinite(actual_cost)
                    or actual_cost < 0
                    or actual_cost > projected_cost
                ):
                    if not math.isfinite(actual_cost):
                        actual_cost = None
                    status = "failed"
                    cost_reconciliation_required = True
                    typed_failure = {
                        "failure_type": "provider_cost_exceeded_projection",
                        "retryable": False,
                    }
                elif typed_failure is None and status != "completed":
                    adapter_retryable = raw.get("retryable")
                    if adapter_retryable not in {None, True, False}:
                        status = "failed"
                        typed_failure = {
                            "failure_type": "provider_retryability_invalid",
                            "retryable": False,
                        }
                    else:
                        retryable = (
                            len(self._attempts) + 1 < max_attempts
                            and adapter_retryable is not False
                        )
                        typed_failure = {
                            "failure_type": str(
                                raw.get("failure_type") or "provider_failure"
                            ),
                            "retryable": retryable,
                        }
        except Exception as exc:  # noqa: BLE001 - normalized and preserved below
            elapsed = max(0.0, self._monotonic() - started)
            actual_cost = None
            cost_reconciliation_required = True
            typed_failure = {
                "failure_type": "provider_adapter_exception",
                "exception_type": type(exc).__name__,
                "raw_message_persisted": False,
                "retryable": False,
            }
        finally:
            try:
                teardown = dict(adapter.teardown())
            except Exception as exc:  # noqa: BLE001 - fail closed on teardown
                teardown = {
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "raw_message_persisted": False,
                }
            if teardown.get("status") != "completed" or teardown.get("provider_zero") is not True:
                status = "failed_teardown"
                typed_failure = {
                    "failure_type": "provider_zero_not_proven",
                    "retryable": False,
                }

        duration = max(0.0, self._monotonic() - started)
        if actual_cost is not None:
            self._spent_usd += actual_cost
        if cost_reconciliation_required:
            self._cost_reconciliation_required = True
        if typed_failure is not None and typed_failure.get("retryable") is not True:
            self._terminal_failure = True
        result: dict[str, Any] = {
            "schema_version": RECOVERY_RESULT_SCHEMA_VERSION,
            "run_id": self.policy.run_id,
            "attempt_id": f"{self.policy.run_id}-{action_id}-{len(self._attempts)}",
            "action_id": action_id,
            "provider_id": provider_id,
            "paid_resource_class": adapter.paid_resource_class,
            "shared_paid_resource_admission_validated": True,
            "status": status,
            "typed_result": dict(raw) if status == "completed" else {},
            "typed_failure": typed_failure,
            "failed_evidence_preserved": status != "completed",
            "immutable_commit_sha": commit_sha,
            "immutable_input_digests": list(input_digests),
            "projected_cost_usd": projected_cost,
            "actual_cost_usd": actual_cost,
            "cumulative_cost_usd": self._spent_usd,
            "max_cost_usd": self.max_cost_usd,
            "duration_seconds": duration,
            "watchdog_seconds": self.policy.watchdog_seconds,
            "effective_timeout_seconds": effective_timeout_seconds,
            "attempt_number": len(self._attempts) + 1,
            "max_attempts": max_attempts,
            "authorization_receipt_digest": receipt["authorization_receipt_digest"],
            "teardown": dict(teardown),
            "provider_zero_proven": teardown.get("provider_zero") is True,
            "cost_reconciliation_required": cost_reconciliation_required,
            "proof_effect": "none",
            "scientific_validity_inferred": False,
            "suggested_next_legal_actions": (
                ["bounded_retry"]
                if typed_failure and typed_failure.get("retryable") is True
                else ["stop_and_preserve_evidence"]
                if status != "completed"
                else []
            ),
        }
        result["recovery_result_digest"] = canonical_digest(
            result,
            digest_field="recovery_result_digest",
        )
        self._attempts.append(result)
        return result


__all__ = [
    "PreauthorizedRecoveryController",
    "PreauthorizedRecoveryPolicy",
    "RECOVERY_ACTION_SCHEMA_VERSION",
    "RECOVERY_RESULT_SCHEMA_VERSION",
    "MAX_RECOVERY_COST_USD",
    "MAX_RECOVERY_RETRIES",
    "MAX_RECOVERY_TTL_SECONDS",
    "RecoveryAdapter",
    "RecoveryControlError",
]
