"""Decide whether a provider's "no funds" answer may move a role to the next provider.

Pure: no I/O, clock, environment or SDK import. It reads an SDK error by its
class and typed fields, never by log text, and it is not wired into any money
path yet: nothing calls it.

One trigger table, the same in the website and here (``classify_fields``):

- OpenAI, out of funds: HTTP 429 AND (error code in ``OPENAI_NO_FUNDS_CODES``
  OR error type ``insufficient_quota``). The codes are the 429 billing codes on
  the OpenAI error-codes page; ``insufficient_quota`` is also the recorded stop
  (``task_evaluation_unentered_authoring_budget._quota_refusal_termination``).
- Anthropic, out of funds: HTTP 402 OR error type ``billing_error``; HTTP 429
  with error code ``enforced_spend_limit_reached`` (the usage tier's monthly
  cap); HTTP 400 ``invalid_request_error`` whose message starts with one of the
  two documented "You have reached your specified ... API usage limits" texts
  (a spend limit the organization set; only the message identifies it, so a
  wording change can only fail to switch).
- Never out of funds: any other 429 (rate limit, ramp limit, ``slow_down``), 529
  and 5xx, timeouts, connection errors, other 400s, 401/403/404/413, refusals,
  an error from another SDK, and every shape in ``UNSETTLED_SHAPES``.

A wrapped error is classified through its ``__cause__`` chain, and only a link
that is a typed error of the provider's own SDK can match. A switch never happens
silently: a ``switch`` decision always carries its ``provider_switch.v1`` record,
and a refused one carries the typed reason for every missing precondition,
including the privacy gate, which is re-run here on the target entry.

When a switch may take effect: a single stateless call switches at the next
call; an agent run switches only between attempts, as a fresh run, because its
history holds provider-bound items (reasoning, signed thinking, tool-call ids)
that do not carry across providers.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .model_routing import ResolvedRoute, RouteEntry, load_config, privacy_refusal, validate_config

SWITCH_SCHEMA_VERSION = "provider_switch.v1"
NO_FUNDS = "no_funds"
NOT_NO_FUNDS = "not_no_funds"

OPENAI_NO_FUNDS_CODES = frozenset({"insufficient_quota", "credit_balance_exhausted",
                                   "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
                                   "organization_usage_limit_exceeded"})
ANTHROPIC_OWN_LIMIT_PREFIXES = ("You have reached your specified API usage limits",
                                "You have reached your specified workspace API usage limits")
_CAUSE_DEPTH = 8
#: Shapes that may mean "no funds" but that neither the SDK types, the live
#: documentation nor a recorded stop settle. Classified NOT no-funds until one is.
UNSETTLED_SHAPES = (
    "anthropic: the text 'Your credit balance is too low' (no status or type in the documentation read 2026-11-13);"
    " the first paid canary must record what it returns",
    "anthropic: whether 402 billing_error also covers an expired card rather than a lack of funds",
    "google: HTTP 402 (the troubleshooting page lists depleted Prepay credits among 400/402/403 causes, no body shape)",
    "google: HTTP 403 PERMISSION_DENIED citing an outstanding balance (message only)",
    "google: HTTP 429 RESOURCE_EXHAUSTED with 'prepayment credits are depleted' (repository string match, no recorded stop)",
    "openai: a billing code that is not on the error-codes page",
)


@dataclass(frozen=True)
class ErrorClassification:
    kind: str
    reason: str
    provider: str
    http_status: int | None = None
    error_type: str | None = None
    error_code: str | None = None
    request_id: str | None = None

    @property
    def no_funds(self) -> bool:
        return self.kind == NO_FUNDS

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "reason": self.reason, "provider": self.provider, "http_status": self.http_status,
                "error_type": self.error_type, "error_code": self.error_code, "request_id": self.request_id}


def _sdk(exc: BaseException) -> str:
    return type(exc).__module__.split(".", 1)[0]


def _mro_names(exc: BaseException) -> frozenset[str]:
    return frozenset(cls.__name__ for cls in type(exc).__mro__)


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _error_object(body: Any) -> Mapping[str, Any]:
    """The ``error`` object of an SDK error body, whichever shape the SDK version keeps."""
    if isinstance(body, Mapping) and isinstance(body.get("error"), Mapping):
        return body["error"]
    return body if isinstance(body, Mapping) else {}


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _request_id(exc: BaseException) -> str | None:
    value = _str(getattr(exc, "request_id", None))
    if value is None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None and hasattr(headers, "get"):
            value = _str(headers.get("x-request-id")) or _str(headers.get("request-id"))
    return value


def classify_fields(provider: str, *, http_status: int | None, error_type: str | None, error_code: str | None,
                    message: str | None = None, request_id: str | None = None) -> ErrorClassification:
    """The trigger table over typed fields of one provider answer.

    The fields must come from a typed record of the provider's answer (an SDK
    error, or a record built from one), never from log text.
    """
    common = {"provider": provider, "http_status": http_status, "error_type": error_type, "error_code": error_code,
              "request_id": request_id}
    if provider == "openai":
        if http_status == 429 and (error_code in OPENAI_NO_FUNDS_CODES or error_type == "insufficient_quota"):
            return ErrorClassification(NO_FUNDS, "openai_out_of_funds", **common)
    elif provider == "anthropic":
        if http_status == 402 or error_type == "billing_error":
            return ErrorClassification(NO_FUNDS, "anthropic_billing_error", **common)
        if http_status == 429 and error_code == "enforced_spend_limit_reached":
            return ErrorClassification(NO_FUNDS, "anthropic_enforced_spend_limit_reached", **common)
        if (http_status == 400 and error_type == "invalid_request_error" and isinstance(message, str)
                and message.startswith(ANTHROPIC_OWN_LIMIT_PREFIXES)):
            return ErrorClassification(NO_FUNDS, "anthropic_own_spend_limit_reached", **common)
    status = http_status if http_status is not None else "no_status"
    return ErrorClassification(NOT_NO_FUNDS, f"{provider}_error_not_no_funds:{status}", **common)


def _sdk_fields(provider: str, exc: BaseException) -> dict[str, Any] | None:
    """Typed fields of ``exc`` when it is an error of ``provider``'s own SDK, else ``None``."""
    sdk = {"openai": "openai", "anthropic": "anthropic", "google": "google"}.get(provider)
    if sdk is None or _sdk(exc) != sdk:
        return None
    if provider == "google":
        return {"http_status": _int(getattr(exc, "code", None)), "error_type": _str(getattr(exc, "status", None)),
                "error_code": None, "message": _str(getattr(exc, "message", None))}
    if "APIStatusError" not in _mro_names(exc):
        return {"http_status": None, "error_type": None, "error_code": None, "message": None,
                "request_id": None}
    error = _error_object(getattr(exc, "body", None))
    details = error.get("details") if isinstance(error.get("details"), Mapping) else {}
    if provider == "openai":
        error_type = _str(getattr(exc, "type", None)) or _str(error.get("type"))
        code = _str(getattr(exc, "code", None)) or _str(error.get("code"))
    else:
        error_type = _str(error.get("type")) or _str(getattr(exc, "type", None))
        code = _str(details.get("error_code")) or _str(getattr(exc, "error_code", None))
    message = _str(error.get("message")) or _str(getattr(exc, "api_message", None))
    return {"http_status": _int(getattr(exc, "status_code", None)), "error_type": error_type, "error_code": code,
            "message": message, "request_id": _request_id(exc)}


def classify_provider_error(provider: str, exc: BaseException) -> ErrorClassification:
    """Is ``exc`` (or an error in its ``__cause__`` chain) a typed "no funds" answer of ``provider``'s SDK?

    The first link that is an error of the provider's own SDK and matches the
    table decides. With no such link the answer is not no-funds.
    """
    first: ErrorClassification | None = None
    link: BaseException | None = exc
    for _ in range(_CAUSE_DEPTH):
        if link is None:
            break
        fields = _sdk_fields(provider, link)
        if fields is not None:
            result = classify_fields(provider, **fields)
            if result.no_funds:
                return result
            first = first or result
        link = link.__cause__
    if first is not None:
        return first
    return ErrorClassification(NOT_NO_FUNDS, f"{provider}_error_not_from_provider_sdk:{type(exc).__name__}",
                               provider=provider)


@dataclass(frozen=True)
class SwitchFacts:
    """What the caller knows about the job and the machine, for each provider.

    Every field names providers. Nothing defaults to admitted: a provider absent
    from a field fails that precondition. ``official_cost_providers`` and
    ``account_counted_providers`` are read from the official-cost authorities
    and ``account_monthly_spend.ACCOUNT_PROVIDERS`` by the caller.
    """

    allowed_providers: frozenset[str]
    terms_reference_providers: frozenset[str]
    official_cost_providers: frozenset[str]
    key_present_providers: frozenset[str]
    network_allowed_providers: frozenset[str]
    account_counted_providers: frozenset[str]


def precondition_failures(entry: RouteEntry, *, list_price: Mapping[str, Any] | None,
                          facts: SwitchFacts) -> tuple[str, ...]:
    """The typed reason for each precondition ``entry`` misses; empty when all hold.

    The privacy gate is checked separately by ``decide_failover``.
    The codes' words let ``job_retry_policy.hard_stop_kind`` name the hard stop:
    ``unauthorized`` and ``consent`` are privacy or rights, ``price``, ``cost``
    and ``spend`` are money; a missing key or host is neither.
    """
    provider = entry.provider
    failures = []
    if provider not in facts.allowed_providers:
        failures.append(f"provider_failover_unauthorized:{provider}")
    if provider not in facts.terms_reference_providers:
        failures.append(f"provider_failover_terms_consent_missing:{provider}")
    if list_price is None:
        failures.append(f"provider_failover_list_price_missing:{provider}:{entry.model}")
    if provider not in facts.official_cost_providers:
        failures.append(f"provider_failover_official_cost_missing:{provider}")
    if provider not in facts.key_present_providers:
        failures.append(f"provider_failover_runtime_key_missing:{provider}")
    if provider not in facts.network_allowed_providers:
        failures.append(f"provider_failover_runtime_network_missing:{provider}")
    if provider not in facts.account_counted_providers:
        failures.append(f"provider_failover_account_spend_uncounted:{provider}")
    return tuple(failures)


@dataclass(frozen=True)
class FailoverDecision:
    #: ``no_switch``: not a no-funds error, handle it as today. ``stop``: no funds and
    #: no admitted next provider. ``switch``: move to ``to_index`` as ``takes_effect`` says.
    action: str
    code: str
    classification: ErrorClassification
    from_index: int
    to_index: int | None = None
    takes_effect: str | None = None
    refusals: tuple[tuple[int, tuple[str, ...]], ...] = ()
    record: dict[str, Any] | None = None


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def decide_failover(route: ResolvedRoute, *, failed_index: int, error: BaseException, facts: SwitchFacts,
                    attempt_id: str, occurred_at: str,
                    refused_call_meter_row: Mapping[str, Any] | None = None,
                    config: Mapping[str, Any] | None = None) -> FailoverDecision:
    """After ``route.entries[failed_index]`` raised ``error``: no switch, stop, or switch.

    A stop's ``code`` is ``provider_funds_exhausted:<provider>``, which the retry
    policy already classifies as a money stop. Every later entry is considered in
    order; the first one meeting every precondition is the target, and each one
    passed over is listed with its reasons in ``refusals``.
    """
    if not 0 <= failed_index < len(route.entries):
        raise ValueError("provider_failover_index_invalid")
    current = route.entries[failed_index]
    classification = classify_provider_error(current.provider, error)
    if not classification.no_funds:
        return FailoverDecision("no_switch", classification.reason, classification, failed_index)
    stop_code = f"provider_funds_exhausted:{current.provider}"
    config = load_config() if config is None else validate_config(config)
    refusals: list[tuple[int, tuple[str, ...]]] = []
    for index in range(failed_index + 1, len(route.entries)):
        entry = route.entries[index]
        if entry.provider == current.provider:
            refusals.append((index, (f"provider_failover_same_provider:{entry.provider}",)))
            continue
        # The gate again, from the configuration's data class for the role: a
        # hand-built or restored route cannot carry a forbidden provider this far.
        privacy = privacy_refusal(config, role=route.role, entry=entry,
                                  privacy_policy_version=route.privacy_policy_version,
                                  site_consents=route.site_consents)
        failures = ((f"provider_failover_privacy_not_permitted:{entry.provider}",) if privacy is not None else ()) \
            + precondition_failures(entry, list_price=route.list_prices[index], facts=facts)
        if failures:
            refusals.append((index, failures))
            continue
        takes_effect = "next_call" if route.call_kind == "single_call" else "next_attempt_fresh_run"
        record = switch_record(route, from_index=failed_index, to_index=index, classification=classification,
                               takes_effect=takes_effect, attempt_id=attempt_id, occurred_at=occurred_at,
                               refused_call_meter_row=refused_call_meter_row)
        return FailoverDecision("switch", f"provider_switch:{current.provider}:{entry.provider}", classification,
                                failed_index, to_index=index, takes_effect=takes_effect,
                                refusals=tuple(refusals), record=record)
    return FailoverDecision("stop", stop_code, classification, failed_index, refusals=tuple(refusals))


def switch_record(route: ResolvedRoute, *, from_index: int, to_index: int, classification: ErrorClassification,
                  takes_effect: str, attempt_id: str, occurred_at: str,
                  refused_call_meter_row: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``provider_switch.v1`` receipt; ``record_digest`` seals every other field."""
    if not isinstance(attempt_id, str) or not attempt_id or not isinstance(occurred_at, str) or not occurred_at:
        raise ValueError("provider_switch_identity_required")
    body = {
        "schema_version": SWITCH_SCHEMA_VERSION, "role": route.role, "data_class": route.data_class,
        "call_kind": route.call_kind, "privacy_policy_version": route.privacy_policy_version,
        "privacy_policy": route.privacy_policy, "takes_effect": takes_effect,
        "fresh_run": takes_effect == "next_attempt_fresh_run",
        "from": {**route.entries[from_index].as_dict(), "index": from_index},
        "to": {**route.entries[to_index].as_dict(), "index": to_index,
               "list_price": route.plain_list_price(to_index)},
        "error": classification.as_dict(), "attempt_id": attempt_id, "occurred_at": occurred_at,
        "refused_call_meter_row": None if refused_call_meter_row is None else dict(refused_call_meter_row),
        "route_digest": route.record()["route_digest"],
    }
    return {**body, "record_digest": _digest(body)}
