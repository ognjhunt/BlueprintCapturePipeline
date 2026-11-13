from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from datetime import date

import httpx
import openai
import pytest

from blueprint_pipeline import model_routing as routing
from blueprint_pipeline.account_monthly_spend import ACCOUNT_PROVIDERS
from blueprint_pipeline.job_retry_policy import MONEY, PRIVACY_RIGHTS, hard_stop_kind
from blueprint_pipeline.model_failover import (
    ANTHROPIC_OWN_LIMIT_PREFIXES, NO_FUNDS, NOT_NO_FUNDS, OPENAI_NO_FUNDS_CODES, SwitchFacts, classify_fields,
    classify_provider_error, decide_failover,
)

TODAY = date(2026, 11, 13)


# --- fake SDK errors ---------------------------------------------------------
# The Anthropic and Google SDKs are not installed in the Pipeline environment.
# These stand-ins carry the class names, module and fields of anthropic 2.8.0
# (anthropic/_exceptions.py, _error_classes.py) and google-genai 2.24.0
# (google/genai/errors.py), which is all the classifier reads.


class _AnthropicAPIError(Exception):
    pass


class _AnthropicAPIStatusError(_AnthropicAPIError):
    def __init__(self, status_code: int, body: object, request_id: str = "req_test") -> None:
        super().__init__(f"Error code: {status_code} - {body}")
        self.status_code = status_code
        self.body = body
        self.request_id = request_id


def _anthropic_class(name: str, base: type) -> type:
    return type(name, (base,), {"__module__": "anthropic._exceptions"})


_AnthropicAPIError.__module__ = "anthropic._exceptions"
_AnthropicAPIError.__name__ = "APIError"
_AnthropicAPIStatusError.__module__ = "anthropic._exceptions"
_AnthropicAPIStatusError.__name__ = "APIStatusError"
A_STATUS = _AnthropicAPIStatusError
A_BAD_REQUEST = _anthropic_class("BadRequestError", A_STATUS)
A_AUTH = _anthropic_class("AuthenticationError", A_STATUS)
A_PERMISSION = _anthropic_class("PermissionDeniedError", A_STATUS)
A_RATE = _anthropic_class("RateLimitError", A_STATUS)
A_SPEND_CAP = type("EnforcedSpendLimitReachedError", (A_RATE,), {"__module__": "anthropic.types.shared."
                                                                  "enforced_spend_limit_reached_error"})
A_RAMP = type("RampLimitExceededError", (A_RATE,), {"__module__": "anthropic.types.shared.ramp_limit_exceeded_error"})
A_ON_HOLD = type("OrganizationOnHoldError", (A_BAD_REQUEST,), {"__module__": "anthropic.types.shared."
                                                               "organization_on_hold_error"})
A_SERVER = _anthropic_class("InternalServerError", A_STATUS)
A_OVERLOADED = _anthropic_class("OverloadedError", A_STATUS)
A_TIMEOUT = _anthropic_class("APITimeoutError", _AnthropicAPIError)


def _anthropic_body(error_type: str, message: str, error_code: str | None = None) -> dict:
    error: dict = {"type": error_type, "message": message}
    if error_code is not None:
        error["details"] = {"error_code": error_code}
    return {"type": "error", "error": error, "request_id": "req_test"}


class _GoogleAPIError(Exception):
    def __init__(self, code: int, status: str, message: str) -> None:
        super().__init__(f"{code} {status}. {message}")
        self.code, self.status, self.message = code, status, message


_GoogleAPIError.__module__ = "google.genai.errors"
_GoogleAPIError.__name__ = "APIError"
G_CLIENT = type("ClientError", (_GoogleAPIError,), {"__module__": "google.genai.errors"})


def _openai(cls: type, status: int, body: dict | None):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status, request=request, headers={"x-request-id": "req_openai"})
    return cls(f"Error code: {status} - {body}", response=response, body=body)


SPEND_CAP = ("You have reached your API usage limits: your organization has crossed its monthly API usage "
             "threshold, set based on your organization's API tier. You will regain access on 2026-12-01 at 00:00 UTC.")

def _oai(status: int, error_type: str | None, code: str | None, cls=None):
    cls = cls or {429: openai.RateLimitError, 400: openai.BadRequestError, 401: openai.AuthenticationError,
                  403: openai.PermissionDeniedError, 404: openai.NotFoundError}.get(status, openai.APIStatusError)
    return _openai(cls, status, {"type": error_type, "code": code, "message": "m"})


OPENAI_FUNDS_CODES = ["insufficient_quota", "credit_balance_exhausted", "organization_spend_limit_exceeded",
                      "project_spend_limit_exceeded", "organization_usage_limit_exceeded"]

TRIGGER_TABLE = [
    # OpenAI, out of funds: 429 and (a billing code, or type insufficient_quota).
    *[("openai", (lambda code=code, typ=typ: _oai(429, typ, code)), NO_FUNDS, "openai_out_of_funds")
      for code in OPENAI_FUNDS_CODES for typ in (None, "insufficient_quota", "requests")],
    ("openai", lambda: _oai(429, "insufficient_quota", None), NO_FUNDS, "openai_out_of_funds"),
    ("openai", lambda: _oai(429, "insufficient_quota", "billing_hard_limit"), NO_FUNDS, "openai_out_of_funds"),
    # OpenAI, never.
    ("openai", lambda: _oai(429, "requests", "rate_limit_exceeded"), NOT_NO_FUNDS, None),
    ("openai", lambda: _oai(429, "rate_limit_error", "slow_down"), NOT_NO_FUNDS, None),
    ("openai", lambda: _oai(429, None, None), NOT_NO_FUNDS, None),
    ("openai", lambda: _openai(openai.RateLimitError, 429, None), NOT_NO_FUNDS, None),
    ("openai", lambda: _oai(500, "server_error", None, openai.InternalServerError), NOT_NO_FUNDS, None),
    ("openai", lambda: _oai(400, "insufficient_quota", "insufficient_quota"), NOT_NO_FUNDS, None),
    ("openai", lambda: _oai(401, "insufficient_quota", "insufficient_quota"), NOT_NO_FUNDS, None),
    ("openai", lambda: _oai(403, None, "organization_spend_limit_exceeded"), NOT_NO_FUNDS, None),
    ("openai", lambda: _oai(404, None, "model_not_found"), NOT_NO_FUNDS, None),
    ("openai", lambda: openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/x")),
     NOT_NO_FUNDS, None),
    ("openai", lambda: RuntimeError("openai.RateLimitError: Error code: 429 - insufficient_quota"),
     NOT_NO_FUNDS, "openai_error_not_from_provider_sdk:RuntimeError"),
    # Anthropic, out of funds: 402 or billing_error; 429 enforced spend cap; 400 own-limit prefixes.
    ("anthropic", lambda: A_STATUS(402, _anthropic_body("billing_error", "There is an issue with your billing.")),
     NO_FUNDS, "anthropic_billing_error"),
    ("anthropic", lambda: A_STATUS(402, _anthropic_body("invalid_request_error", "x")), NO_FUNDS,
     "anthropic_billing_error"),
    ("anthropic", lambda: A_BAD_REQUEST(400, _anthropic_body("billing_error", "x")), NO_FUNDS,
     "anthropic_billing_error"),
    ("anthropic", lambda: A_SPEND_CAP(429, _anthropic_body("rate_limit_error", SPEND_CAP,
                                                           "enforced_spend_limit_reached")),
     NO_FUNDS, "anthropic_enforced_spend_limit_reached"),
    ("anthropic", lambda: A_RATE(429, _anthropic_body("rate_limit_error", SPEND_CAP,
                                                      "enforced_spend_limit_reached")["error"]),
     NO_FUNDS, "anthropic_enforced_spend_limit_reached"),
    ("anthropic", lambda: A_BAD_REQUEST(400, _anthropic_body(
        "invalid_request_error", "You have reached your specified API usage limits. You will regain access on "
        "2026-12-01 at 00:00 UTC.")), NO_FUNDS, "anthropic_own_spend_limit_reached"),
    ("anthropic", lambda: A_BAD_REQUEST(400, _anthropic_body(
        "invalid_request_error", "You have reached your specified workspace API usage limits. You will regain "
        "access on 2026-12-01 at 00:00 UTC.")), NO_FUNDS, "anthropic_own_spend_limit_reached"),
    # Anthropic, never: everything else, including the unsettled credit-balance text.
    ("anthropic", lambda: A_RAMP(429, _anthropic_body("rate_limit_error", "ramp", "ramp_limit_exceeded")),
     NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_RATE(429, _anthropic_body("rate_limit_error", "Number of request tokens has exceeded")),
     NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_RATE(429, _anthropic_body("rate_limit_error", SPEND_CAP)), NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_OVERLOADED(529, _anthropic_body("overloaded_error", "Overloaded")), NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_SERVER(500, _anthropic_body("api_error", "Internal server error")), NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_STATUS(504, _anthropic_body("timeout_error", "timeout")), NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_TIMEOUT("Request timed out"), NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_BAD_REQUEST(400, _anthropic_body("invalid_request_error", "max_tokens: too large")),
     NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_BAD_REQUEST(400, _anthropic_body(
        "invalid_request_error", "messages.0.content: You have reached your specified API usage limits")),
     NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_BAD_REQUEST(400, _anthropic_body(
        "invalid_request_error", "Your credit balance is too low to access the Anthropic API.")), NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_ON_HOLD(400, _anthropic_body("invalid_request_error", "on hold", "organization_on_hold")),
     NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_AUTH(401, _anthropic_body("authentication_error", "invalid x-api-key")),
     NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_PERMISSION(403, _anthropic_body("permission_error", "billing")), NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_STATUS(404, _anthropic_body("not_found_error", "model: x")), NOT_NO_FUNDS, None),
    ("anthropic", lambda: A_STATUS(413, _anthropic_body("request_too_large", "too large")), NOT_NO_FUNDS, None),
    ("anthropic", lambda: _oai(429, "insufficient_quota", "insufficient_quota"),
     NOT_NO_FUNDS, "anthropic_error_not_from_provider_sdk:RateLimitError"),
    # Google: no shape is settled, so none switches.
    ("google", lambda: G_CLIENT(429, "RESOURCE_EXHAUSTED", "Your prepayment credits are depleted."),
     NOT_NO_FUNDS, None),
    ("google", lambda: G_CLIENT(402, "FAILED_PRECONDITION", "Prepay credits depleted"), NOT_NO_FUNDS, None),
    ("google", lambda: G_CLIENT(403, "PERMISSION_DENIED", "outstanding balance"), NOT_NO_FUNDS, None),
]


@pytest.mark.parametrize("provider, make, kind, reason", TRIGGER_TABLE)
def test_trigger_table(provider: str, make, kind: str, reason: str | None) -> None:
    result = classify_provider_error(provider, make())
    assert result.kind == kind
    if reason is not None:
        assert result.reason == reason
    assert result.provider == provider


def test_the_table_matches_its_published_text() -> None:
    assert OPENAI_NO_FUNDS_CODES == frozenset(OPENAI_FUNDS_CODES)
    assert ANTHROPIC_OWN_LIMIT_PREFIXES == ("You have reached your specified API usage limits",
                                            "You have reached your specified workspace API usage limits")


def _wrapped(inner: BaseException, depth: int = 1) -> BaseException:
    error = inner
    for level in range(depth):
        try:
            raise RuntimeError(f"wrapped {level}") from error
        except RuntimeError as outer:
            error = outer
    return error


def test_a_wrapped_sdk_error_is_classified_through_its_cause_chain() -> None:
    assert classify_provider_error("openai", _wrapped(_oai(429, "insufficient_quota", "insufficient_quota"))).no_funds
    assert classify_provider_error("anthropic", _wrapped(A_SPEND_CAP(429, _anthropic_body(
        "rate_limit_error", SPEND_CAP, "enforced_spend_limit_reached")), depth=3)).no_funds
    # A plain rate limit stays a rate limit, wrapped or not.
    wrapped = classify_provider_error("openai", _wrapped(_oai(429, "requests", "rate_limit_exceeded")))
    assert not wrapped.no_funds and wrapped.reason == "openai_error_not_no_funds:429"
    # Only the provider's own SDK error can match; log text in a wrapper never does.
    assert not classify_provider_error("openai", RuntimeError("insufficient_quota")).no_funds
    assert not classify_provider_error("anthropic", _wrapped(_oai(429, "insufficient_quota",
                                                                  "insufficient_quota"))).no_funds
    # ``__context__`` (an error raised while handling another) is not a cause.
    try:
        try:
            raise _oai(429, "insufficient_quota", "insufficient_quota")
        except openai.RateLimitError:
            raise RuntimeError("cleanup failed")
    except RuntimeError as unrelated:
        assert not classify_provider_error("openai", unrelated).no_funds


def test_the_table_classifies_typed_fields_alone() -> None:
    assert classify_fields("openai", http_status=429, error_type=None,
                           error_code="project_spend_limit_exceeded").no_funds
    assert not classify_fields("openai", http_status=429, error_type="RateLimitError", error_code=None).no_funds
    assert classify_fields("anthropic", http_status=402, error_type=None, error_code=None).no_funds
    assert not classify_fields("google", http_status=402, error_type=None, error_code=None).no_funds


def test_classification_keeps_the_typed_fields() -> None:
    result = classify_provider_error("anthropic", A_SPEND_CAP(429, _anthropic_body(
        "rate_limit_error", SPEND_CAP, "enforced_spend_limit_reached")))
    assert result.as_dict() == {"kind": NO_FUNDS, "reason": "anthropic_enforced_spend_limit_reached",
                                "provider": "anthropic", "http_status": 429, "error_type": "rate_limit_error",
                                "error_code": "enforced_spend_limit_reached", "request_id": "req_test"}
    quota = classify_provider_error("openai", _oai(429, "insufficient_quota", "insufficient_quota"))
    assert (quota.http_status, quota.error_type, quota.error_code, quota.request_id) == (
        429, "insufficient_quota", "insufficient_quota", "req_openai")


# --- decisions ---------------------------------------------------------------


CONSENT = routing.SiteConsent("2026-09-24.v1", "anthropic", "claude-opus-5-5")


@pytest.fixture()
def config() -> dict:
    config = copy.deepcopy(routing.load_config(today=TODAY))
    # A test-only order: OpenAI first, then Claude, then Gemini, for an agent-run role and a single-call role.
    table = copy.deepcopy(config["profiles"]["current"])
    table["artisan_text"] = [{"provider": "openai", "model": "gpt-6.1-sol", "effort": "max"},
                             {"provider": "anthropic", "model": "claude-opus-5-5", "effort": "max"},
                             {"provider": "google", "model": "gemini-3.8-flash", "effort": None}]
    table["website_task_grounding"] = [{"provider": "openai", "model": "gpt-6.1-sol", "effort": None},
                                       {"provider": "google", "model": "gemini-3.8-flash", "effort": None}]
    config["profiles"]["openai_then_claude"] = table
    return routing.validate_config(config, today=TODAY)


ALL = frozenset({"openai", "anthropic", "google"})
FACTS = SwitchFacts(allowed_providers=ALL, terms_reference_providers=ALL, official_cost_providers=ALL,
                    key_present_providers=ALL, network_allowed_providers=ALL, account_counted_providers=ALL)
QUOTA = {"type": "insufficient_quota", "code": "insufficient_quota", "message": "You exceeded your current quota"}


def _route(config: dict, role: str = "artisan_text", **kwargs):
    return routing.resolve_route(role, config=config, profile="openai_then_claude", site_consents=(CONSENT,),
                                 **kwargs)


def _decide(route, error, config, facts=FACTS, failed_index=0):
    return decide_failover(route, failed_index=failed_index, error=error, facts=facts, attempt_id="attempt-1",
                           occurred_at="2026-11-13T04:00:00Z",
                           refused_call_meter_row={"id": "call-7", "cost_usd": 0}, config=config)


def test_an_agent_run_switches_only_at_the_next_attempt_as_a_fresh_run(config: dict) -> None:
    route = _route(config)
    decision = _decide(route, _openai(openai.RateLimitError, 429, QUOTA), config)
    assert (decision.action, decision.to_index, decision.takes_effect) == ("switch", 1, "next_attempt_fresh_run")
    assert decision.code == "provider_switch:openai:anthropic"
    record = decision.record
    assert record["schema_version"] == "provider_switch.v1" and record["fresh_run"] is True
    assert record["from"] == {"provider": "openai", "model": "gpt-6.1-sol", "effort": "max", "index": 0}
    assert record["to"]["model"] == "claude-opus-5-5" and record["to"]["list_price"]["usd_per_million_tokens"][
        "output"] == 20.0
    assert record["error"]["reason"] == "openai_out_of_funds" and record["error"]["request_id"] == "req_openai"
    assert record["refused_call_meter_row"] == {"id": "call-7", "cost_usd": 0}
    assert record["route_digest"] == route.record()["route_digest"]
    assert record["data_class"] == "site_frames" and record["privacy_policy"] == "baseline"
    assert record["privacy_policy_version"] is None
    body = {key: value for key, value in record.items() if key != "record_digest"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert record["record_digest"] == "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def test_a_single_call_switches_at_the_next_call(config: dict) -> None:
    route = _route(config, "website_task_grounding")
    assert [entry.provider for entry in route.entries] == ["openai", "google"]
    decision = _decide(route, _openai(openai.RateLimitError, 429, QUOTA), config)
    assert (decision.action, decision.to_index, decision.takes_effect) == ("switch", 1, "next_call")
    assert decision.record["fresh_run"] is False


def test_without_consent_claude_is_never_a_target(config: dict) -> None:
    route = routing.resolve_route("artisan_text", config=config, profile="openai_then_claude")
    assert [entry.provider for entry in route.entries] == ["openai", "google"]
    decision = _decide(route, _openai(openai.RateLimitError, 429, QUOTA), config)
    assert decision.action == "switch" and route.entries[decision.to_index].provider == "google"


def test_a_hand_built_route_cannot_switch_to_a_forbidden_provider(config: dict, monkeypatch) -> None:
    """The reviewer's probe, inverted: a route built with Claude for frames is refused, and so is the switch."""
    base = routing.resolve_route("website_spec_research", config=config)
    opus = routing.RouteEntry("anthropic", "claude-opus-5-5", None)
    opus_price = config["prices"]["anthropic"]["claude-opus-5-5"]
    with pytest.raises(routing.ModelRoutingError, match="model_route_entry_provider_privacy_not_permitted"):
        dataclasses.replace(base, entries=(base.entries[0], opus), list_prices=(base.list_prices[0], opus_price))
    # A route that was admitted under another table (here: one that let Claude see frames) is
    # gated again at the decision, under the shipped tables.
    loose = copy.deepcopy(config)
    loose["privacy_policies"]["baseline"]["data_classes"]["site_frames"]["providers"].append("anthropic")
    loose["privacy_policies"]["baseline"]["data_classes"]["site_frames"]["providers_by_site_choice"] = []
    loose["site_consents"] = {}
    monkeypatch.setattr(routing, "_load_shipped", lambda: copy.deepcopy(loose))
    admitted = dataclasses.replace(base, entries=(base.entries[0], opus), list_prices=(base.list_prices[0],
                                                                                         opus_price))
    monkeypatch.undo()
    decision = _decide(admitted, _openai(openai.RateLimitError, 429, QUOTA), config)
    assert decision.action == "stop" and decision.record is None
    assert decision.refusals == ((1, ("provider_failover_privacy_not_permitted:anthropic",)),)
    assert hard_stop_kind(decision.refusals[0][1][0]) == PRIVACY_RIGHTS
    assert decision.config_digest_matches_current is True


def test_a_loosened_configuration_cannot_drive_a_switch(config: dict) -> None:
    loose = copy.deepcopy(config)
    loose["privacy_policies"]["baseline"]["data_classes"]["site_frames"]["providers"].append("anthropic")
    loose["privacy_policies"]["baseline"]["data_classes"]["site_frames"]["providers_by_site_choice"] = []
    loose["site_consents"] = {}
    route = _route(config)
    with pytest.raises(routing.ModelRoutingError, match="model_routing_privacy_not_shipped:privacy_policies"):
        _decide(route, _openai(openai.RateLimitError, 429, QUOTA), loose)


def test_a_switch_record_says_which_configuration_it_was_resolved_under(config: dict) -> None:
    route = _route(config)
    changed = copy.deepcopy(config)
    changed["prices"]["anthropic"]["claude-opus-5-5"]["usd_per_million_tokens"]["output"] = 21.0
    decision = _decide(route, _openai(openai.RateLimitError, 429, QUOTA), changed)
    assert decision.action == "switch" and decision.config_digest_matches_current is False
    assert decision.record["config_digest"] == route.config_digest
    assert decision.record["config_digest_matches_current"] is False
    # The frozen price is the one the route was resolved with.
    assert decision.record["to"]["list_price"]["usd_per_million_tokens"]["output"] == 20.0


@pytest.mark.parametrize("make", [
    lambda: _oai(429, "requests", "rate_limit_exceeded"),
    lambda: _oai(429, "rate_limit_error", "slow_down"),
    lambda: _oai(503, "server_error", None, openai.InternalServerError),
    lambda: RuntimeError("insufficient_quota"),
])
def test_anything_but_no_funds_does_not_switch(config: dict, make) -> None:
    decision = _decide(_route(config), make(), config)
    assert decision.action == "no_switch" and decision.record is None and decision.to_index is None


@pytest.mark.parametrize("field, reason, stop_kind", [
    ("allowed_providers", "provider_failover_unauthorized:anthropic", PRIVACY_RIGHTS),
    ("terms_reference_providers", "provider_failover_terms_consent_missing:anthropic", PRIVACY_RIGHTS),
    ("official_cost_providers", "provider_failover_official_cost_missing:anthropic", MONEY),
    ("account_counted_providers", "provider_failover_account_spend_uncounted:anthropic", MONEY),
    ("key_present_providers", "provider_failover_runtime_key_missing:anthropic", None),
    ("network_allowed_providers", "provider_failover_runtime_network_missing:anthropic", None),
])
def test_each_missing_precondition_refuses_with_its_typed_reason(config: dict, field: str, reason: str,
                                                                 stop_kind: str | None) -> None:
    facts = SwitchFacts(**{**FACTS.__dict__, field: frozenset({"openai"})})
    decision = _decide(_route(config), _openai(openai.RateLimitError, 429, QUOTA), config, facts=facts)
    assert decision.action == "stop" and decision.record is None
    assert decision.code == "provider_funds_exhausted:openai" and hard_stop_kind(decision.code) == MONEY
    assert decision.refusals == ((1, (reason,)), (2, (reason.replace("anthropic", "google"),)))
    assert hard_stop_kind(reason) == stop_kind


def test_a_missing_list_price_refuses_as_money(config: dict) -> None:
    route = _route(config)
    unpriced = dataclasses.replace(route, list_prices=(route.list_prices[0], None, route.list_prices[2]))
    decision = _decide(unpriced, _openai(openai.RateLimitError, 429, QUOTA), config,
                       facts=SwitchFacts(**{**FACTS.__dict__, "allowed_providers": frozenset({"openai",
                                                                                              "anthropic"})}))
    assert decision.refusals[0] == (1, ("provider_failover_list_price_missing:anthropic:claude-opus-5-5",))
    assert hard_stop_kind(decision.refusals[0][1][0]) == MONEY


def test_a_refused_target_is_passed_over_for_the_next_one(config: dict) -> None:
    facts = SwitchFacts(**{**FACTS.__dict__, "official_cost_providers": frozenset({"openai", "google"})})
    decision = _decide(_route(config), _openai(openai.RateLimitError, 429, QUOTA), config, facts=facts)
    assert decision.action == "switch" and decision.to_index == 2
    assert decision.refusals == ((1, ("provider_failover_official_cost_missing:anthropic",)),)


def test_todays_account_limit_does_not_count_anthropic(config: dict) -> None:
    facts = SwitchFacts(**{**FACTS.__dict__, "account_counted_providers": frozenset(ACCOUNT_PROVIDERS)})
    decision = _decide(_route(config), _openai(openai.RateLimitError, 429, QUOTA), config, facts=facts)
    assert decision.refusals[0] == (1, ("provider_failover_account_spend_uncounted:anthropic",))
    assert decision.refusals[1] == (2, ("provider_failover_account_spend_uncounted:google",))
    assert decision.action == "stop"


def test_no_funds_on_the_last_provider_stops_as_money(config: dict) -> None:
    route = routing.resolve_route("artisan_text", config=config, profile="same_level_claude",
                                  site_consents=(CONSENT,))
    decision = _decide(route, _oai(429, None, "credit_balance_exhausted"), config, failed_index=1)
    assert (decision.action, decision.code, decision.refusals) == ("stop", "provider_funds_exhausted:openai", ())
    assert hard_stop_kind(decision.code) == MONEY


def test_a_later_entry_on_the_same_provider_is_not_a_switch(config: dict) -> None:
    route = _route(config)
    same = routing.RouteEntry("openai", "gpt-6.1-sol", "high")
    doubled = dataclasses.replace(route, entries=(route.entries[0], same),
                                  list_prices=(route.list_prices[0], route.list_prices[0]))
    decision = _decide(doubled, _openai(openai.RateLimitError, 429, QUOTA), config)
    assert decision.action == "stop"
    assert decision.refusals == ((1, ("provider_failover_same_provider:openai",)),)


def test_the_failed_index_must_name_an_entry(config: dict) -> None:
    with pytest.raises(ValueError, match="provider_failover_index_invalid"):
        _decide(_route(config), RuntimeError("x"), config, failed_index=5)


def test_a_switch_is_never_silent(config: dict) -> None:
    route = _route(config, "website_task_grounding")
    for make in [row[1] for row in TRIGGER_TABLE if row[0] == "openai"]:
        decision = _decide(route, make(), config)
        assert (decision.action == "switch") == (decision.record is not None)
        assert decision.action in {"switch", "stop", "no_switch"} and decision.code
