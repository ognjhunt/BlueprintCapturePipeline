"""Operation-classified retry policy for transport calls (tenacity-backed).

Rules (shared by provider transport and job transport):

- **Reads** (list/status/inspect) may retry, but only with an explicit
  exception allowlist, a bounded stop (attempts AND total delay), jittered
  exponential backoff, and an evidence hook — tenacity's unbounded defaults
  are never used.
- **Mutations** (create/update/start/stop/delete) get exactly one attempt.
  After an ambiguous outcome the only sanctioned path is reconcile-then-
  retry: prove by provider inventory (exact name/id) that nothing was
  created before a single retry is allowed. Unproven absence refuses the
  retry (`MutationRetryForbidden`) — this preserves the paid-lane
  ``allocation_created is False`` vs ``allocation_outcome_ambiguous``
  discipline.
- The optional circuit breaker (pybreaker) is process-local advisory state
  at provider integration points. It never replaces `provider_race`'s
  circuit breaker, provider reconciliation, or paid-resource admission —
  those own the money truth.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Mapping, Sequence

import tenacity

RETRY_EVIDENCE_SCHEMA_VERSION = "transport_retry_evidence.v1"

EvidenceHook = Callable[[Mapping[str, Any]], None]


class TransportRetryConfigError(ValueError):
    """A retry policy was requested without its mandatory bounds/config."""


class MutationRetryForbidden(RuntimeError):
    """A mutation retry was attempted without proven non-allocation."""


def _evidence_row(
    *,
    operation: str,
    attempt: int,
    outcome: str,
    exception: BaseException | None = None,
    delay_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RETRY_EVIDENCE_SCHEMA_VERSION,
        "operation": operation,
        "attempt": attempt,
        "outcome": outcome,
        "exception_type": type(exception).__name__ if exception else None,
        "delay_seconds": delay_seconds,
    }


def _validate_common(
    *, operation: str, evidence_hook: EvidenceHook | None
) -> None:
    if not str(operation or "").strip():
        raise TransportRetryConfigError("transport_retry_operation_required")
    if not callable(evidence_hook):
        raise TransportRetryConfigError("transport_retry_evidence_hook_required")


def bounded_read_retry(
    *,
    operation: str,
    exception_allowlist: Sequence[type[BaseException]],
    max_attempts: int,
    max_delay_seconds: float,
    evidence_hook: EvidenceHook,
    jitter_initial_seconds: float = 0.5,
    jitter_max_seconds: float = 8.0,
    sleep: Callable[[float], None] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Bounded, jittered retry decorator for idempotent READ operations only."""

    _validate_common(operation=operation, evidence_hook=evidence_hook)
    allowlist = tuple(exception_allowlist)
    if not allowlist or not all(
        isinstance(item, type) and issubclass(item, BaseException) for item in allowlist
    ):
        raise TransportRetryConfigError("transport_retry_exception_allowlist_required")
    if int(max_attempts) < 1:
        raise TransportRetryConfigError("transport_retry_max_attempts_invalid")
    if float(max_delay_seconds) <= 0:
        raise TransportRetryConfigError("transport_retry_max_delay_invalid")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            def _before_sleep(retry_state: "tenacity.RetryCallState") -> None:
                outcome = retry_state.outcome
                exception = outcome.exception() if outcome else None
                next_action = retry_state.next_action
                evidence_hook(
                    _evidence_row(
                        operation=operation,
                        attempt=retry_state.attempt_number,
                        outcome="retrying",
                        exception=exception,
                        delay_seconds=getattr(next_action, "sleep", None),
                    )
                )

            retry_kwargs: dict[str, Any] = {
                "retry": tenacity.retry_if_exception_type(allowlist),
                "stop": (
                    tenacity.stop_after_attempt(int(max_attempts))
                    | tenacity.stop_after_delay(float(max_delay_seconds))
                ),
                "wait": tenacity.wait_exponential_jitter(
                    initial=jitter_initial_seconds, max=jitter_max_seconds
                ),
                "before_sleep": _before_sleep,
                "reraise": True,
            }
            if sleep is not None:
                retry_kwargs["sleep"] = sleep
            return tenacity.Retrying(**retry_kwargs)(func, *args, **kwargs)

        return wrapper

    return decorator


def mutation_single_attempt(
    *, operation: str, evidence_hook: EvidenceHook
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Exactly one attempt for a MUTATION; failures propagate with evidence."""

    _validate_common(operation=operation, evidence_hook=evidence_hook)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)
            except BaseException as exc:
                evidence_hook(
                    _evidence_row(
                        operation=operation,
                        attempt=1,
                        outcome="failed_no_retry",
                        exception=exc,
                    )
                )
                raise
            evidence_hook(
                _evidence_row(operation=operation, attempt=1, outcome="succeeded")
            )
            return result

        return wrapper

    return decorator


def reconcile_then_retry_mutation(
    *,
    operation: str,
    mutate: Callable[[], Any],
    reconcile: Callable[[], Mapping[str, Any]],
    evidence_hook: EvidenceHook,
) -> Any:
    """Mutation with the only sanctioned retry: reconcile-proven absence.

    ``reconcile`` must consult provider inventory by exact name/id and return
    a mapping with ``exists``: True (resource found — return it, no retry),
    False (proven absent — one retry allowed), or anything else (unproven —
    retry refused).
    """

    _validate_common(operation=operation, evidence_hook=evidence_hook)
    try:
        result = mutate()
    except BaseException as exc:
        evidence_hook(
            _evidence_row(
                operation=operation, attempt=1, outcome="ambiguous", exception=exc
            )
        )
        reconciled = dict(reconcile())
        exists = reconciled.get("exists")
        if exists is True:
            evidence_hook(
                _evidence_row(
                    operation=operation, attempt=1, outcome="reconciled_existing"
                )
            )
            outcome = {
                key: value for key, value in reconciled.items() if key != "exists"
            }
            return {"status": "reconciled_existing", **outcome}
        if exists is False:
            evidence_hook(
                _evidence_row(
                    operation=operation, attempt=2, outcome="retry_after_proven_absence"
                )
            )
            return mutate()
        raise MutationRetryForbidden(
            f"mutation_retry_without_reconciled_absence:{operation}"
        ) from exc
    evidence_hook(_evidence_row(operation=operation, attempt=1, outcome="succeeded"))
    return result


def optional_circuit_breaker(
    *, name: str, fail_max: int = 5, reset_timeout_seconds: int = 60
) -> Any:
    """Optional pybreaker circuit breaker for provider integration points.

    Process-local by default (multi-worker protection would need pybreaker's
    durable storage); advisory only — never a replacement for
    ``provider_race.ProviderCircuitBreaker``, provider reconciliation, or
    paid-resource circuit state. Raises if pybreaker is not installed rather
    than silently degrading.
    """

    if not str(name or "").strip():
        raise TransportRetryConfigError("transport_breaker_name_required")
    try:
        import pybreaker
    except ImportError as exc:
        raise TransportRetryConfigError("pybreaker_not_installed") from exc
    return pybreaker.CircuitBreaker(
        fail_max=fail_max, reset_timeout=reset_timeout_seconds, name=name
    )


__all__ = [
    "RETRY_EVIDENCE_SCHEMA_VERSION",
    "TransportRetryConfigError",
    "MutationRetryForbidden",
    "bounded_read_retry",
    "mutation_single_attempt",
    "reconcile_then_retry_mutation",
    "optional_circuit_breaker",
]
