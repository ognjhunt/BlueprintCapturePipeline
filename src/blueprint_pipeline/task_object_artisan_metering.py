"""Thin metering around upstream API transports; no second agent/session loop.

Unknown charges retain the whole remaining reservation and refuse another call
until official reconciliation. Resume retains this file and its request count.

The SDK's own retries stay off so no request escapes metering. A call whose
transport raises is settled here instead: a 4xx is a rejected request and is
priced $0; anything else is priced at that request's upper bound when it has
one, else it stays unknown. Retryable failures are then retried with the SDK's
own policy, each attempt metered as its own call.

With ``provider="anthropic"`` the upstream Messages call is metered the same
way: its output is bounded by the SDK's token count and the remaining
authority, it is pinned to US-only inference, and it is priced from usage at the served
model's upstream price (with the frozen US-region uplift) only when the response carries its stop reason and
whole usage at the standard speed and tier, and otherwise counted at its bound
for official reconciliation, and
upstream's own retries each pass through the meter. A refusal for lack of
funds is settled at $0 and is not retried. The bound carries the same US-region
uplift as the price, so a call can never be charged above what it was bounded at.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid


class Meter:
    def __init__(self, path: Path, *, maximum_cost_usd, maximum_requests, deadline_epoch):
        self.path = path
        self.cap = float(maximum_cost_usd)
        self.maximum_requests = int(maximum_requests)
        self.deadline = float(deadline_epoch)
        if (not math.isfinite(self.cap) or self.cap <= 0 or self.maximum_requests <= 0
                or not math.isfinite(self.deadline) or self.path.is_symlink()):
            raise ValueError("artisan_meter_authority_invalid")
        self.binding = {"maximum_cost_usd": self.cap, "maximum_requests": self.maximum_requests,
                        "deadline_epoch": self.deadline}
        if not path.exists():
            self._write({"schema_version": "artisan_upstream_usage.v1", **self.binding, "calls": []})
        if any(self.read().get(k) != v for k, v in self.binding.items()):
            raise ValueError("artisan_meter_authority_changed_on_resume")

    def read(self):
        if self.path.is_symlink():
            raise ValueError("artisan_meter_symlink_forbidden")
        return json.loads(self.path.read_text())

    def _write(self, value):
        descriptor, raw = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        temporary = Path(raw)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(value, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def remaining(self):
        data = self.read()
        if any(row["status"] != "priced" for row in data["calls"]):
            raise RuntimeError("artisan_unknown_charge_requires_reconciliation")
        return self.cap - sum(row["cost_usd"] for row in data["calls"])

    def mark_no_funds(self, identity):
        """Name a settled $0 rejection as the provider's refusal for lack of funds (the attempt ends here)."""
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self.read()
            row = next(c for c in data["calls"] if c["id"] == identity)
            if row["status"] == "priced" and row["cost_usd"] == 0:
                row["cost_origin"] = NO_FUNDS_ORIGIN
                self._write(data)

    def before(self, kind, model):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self.read()
            calls = data["calls"]
            if any(c["status"] != "priced" for c in calls):
                raise RuntimeError("artisan_unknown_charge_requires_reconciliation")
            spent = sum(c["cost_usd"] for c in calls)
            if spent >= self.cap or len(calls) >= self.maximum_requests or time.time() >= self.deadline:
                raise RuntimeError("artisan_shared_stage_authority_exhausted")
            identity = uuid.uuid4().hex
            calls.append({"id": identity, "kind": kind, "model": model, "status": "in_flight",
                          "reserved_usd": self.cap - spent, "cost_usd": None})
            self._write(data)
            return identity

    def after(self, identity, cost, *, usage=None, response_id=None, cost_origin="upstream_priced_usage",
              failure=None):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self.read()
            row = next(c for c in data["calls"] if c["id"] == identity)
            known = isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0
            row.update(status="priced" if known else "reconciliation_required", cost_usd=cost if known else None,
                       usage=usage, response_id=response_id, cost_origin=cost_origin)
            if failure is not None:
                row["failure"] = failure
            self._write(data)
            if not known:
                raise RuntimeError("artisan_unknown_charge_requires_reconciliation")
            if sum(c.get("cost_usd") or 0 for c in data["calls"]) > self.cap:
                raise RuntimeError("artisan_reported_cost_exceeded_stage_authority")


    def failed(self, identity, error, *, bound_usd=None):
        """Settle a call whose transport raised before returning usage.

        Returns whether the row is now priced, so the caller may retry.
        """
        status = getattr(error, "status_code", None)
        status = status if isinstance(status, int) and not isinstance(status, bool) else None
        bounded = (isinstance(bound_usd, (int, float)) and not isinstance(bound_usd, bool)
                   and math.isfinite(bound_usd) and bound_usd >= 0)
        request_id = getattr(error, "request_id", None)
        failure = {"error_type": type(error).__name__, "http_status": status,
                   "request_id": request_id if isinstance(request_id, str) else None}
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self.read()
            row = next(c for c in data["calls"] if c["id"] == identity)
            if row["status"] != "in_flight":
                raise RuntimeError("artisan_meter_call_already_settled")
            if status is not None and 400 <= status < 500:
                # Rejected before any model work (invalid, unauthorized, rate limited).
                row.update(status="priced", cost_usd=0.0, cost_origin="upstream_rejected_request", failure=failure)
            elif bounded:
                # May have run: count the most this request could have cost.
                row.update(status="priced", cost_usd=float(bound_usd), cost_origin="request_upper_bound_no_usage",
                           failure=failure)
            else:
                row.update(status="reconciliation_required", cost_usd=None, cost_origin="no_usage_returned",
                           failure=failure)
            self._write(data)
            return row["status"] == "priced"


def _retryable(error) -> bool:
    """The SDK's retry rule: timeouts, conflicts, rate limits, 5xx, lost connections."""
    import openai

    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status in (408, 409, 429) or status >= 500
    return isinstance(error, openai.APIConnectionError)


def _retry_delay(error, attempt: int) -> float:
    """The SDK's schedule: a Retry-After of at most 60 s, else 0.5 s doubling to 8 s."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        seconds = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        seconds = math.nan
    return seconds if math.isfinite(seconds) and 0 < seconds <= 60 else min(0.5 * 2 ** attempt, 8.0)


def _next_attempt(meter: Meter, identity, error, bound, attempt: int) -> float:
    """Settle a raised call; the delay before retrying it, or re-raise."""
    import openai

    settled = identity is None or meter.failed(identity, error, bound_usd=bound)
    delay = _retry_delay(error, attempt)
    if (not settled or not _retryable(error) or attempt >= openai.DEFAULT_MAX_RETRIES
            or time.time() + delay >= meter.deadline):
        raise error
    return delay


ANTHROPIC_OWN_LIMIT_PREFIXES = ("You have reached your specified API usage limits",
                                "You have reached your specified workspace API usage limits")
NO_FUNDS_ORIGIN = "upstream_rejected_request_no_funds"


def anthropic_no_funds(http_status, error_type, error_code, message) -> bool:
    return (http_status == 402 or error_type == "billing_error"
            or (http_status == 429 and error_code == "enforced_spend_limit_reached")
            or (http_status == 400 and error_type == "invalid_request_error" and isinstance(message, str)
                and message.startswith(ANTHROPIC_OWN_LIMIT_PREFIXES)))


class ProviderOutOfFunds(RuntimeError):
    """A metered call the provider refused for lack of funds; upstream does not retry it.

    The provider's own error is ``__cause__``.
    """

    def __str__(self) -> str:
        return "provider refused the request for lack of funds"


def _anthropic_error_fields(error):
    """Typed status, type, code and message of an Anthropic SDK error, or of upstream's wrapper of one."""
    for candidate in (error, getattr(error, "__cause__", None)):
        body = getattr(candidate, "body", None)
        status = getattr(candidate, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool) and isinstance(body, dict):
            detail = body.get("error") if isinstance(body.get("error"), dict) else {}
            fields = [detail.get(key) for key in ("type", "code", "message")]
            return (status, *(value if isinstance(value, str) else None for value in fields))
    return None


def _anthropic_bound(meter: Meter, rates, tokens: int, requested: int, us_multiplier=None):
    """The output tokens this request may use, and the most it can cost (with the uplift pricing may apply)."""
    if rates is None or type(tokens) is not int or tokens < 0:
        raise RuntimeError("artisan_request_price_or_token_count_unavailable")
    uplift = max(1.0, us_multiplier if isinstance(us_multiplier, (int, float)) and not isinstance(us_multiplier, bool)
                 else 1.0)
    # Every cache write is the 5-minute kind: upstream writes a 1-hour cache only on a setting the controller never passes.
    input_rate = uplift * max(rates["input_uncached"], rates.get("input_cache_write", 0))
    output_rate = uplift * rates["output"]
    remaining = meter.remaining() - tokens * input_rate / 1_000_000
    output = math.floor(remaining * 1_000_000 / output_rate)
    if output < 1:
        raise RuntimeError("artisan_shared_stage_authority_exhausted")
    output = min(output, requested)
    return output, (tokens * input_rate + output * output_rate) / 1_000_000


ANTHROPIC_COUNT_KEYS = frozenset({"model", "messages", "system", "tools", "thinking", "output_config", "tool_choice",
                                  "cache_control"})


#: Every Messages request is pinned to US-only inference; only this pin brings the US uplift into its bound and price.
INFERENCE_GEO = "us"
STANDARD_GEOS = (None, "global")
STANDARD_SPEEDS = (None, "standard")
STANDARD_TIERS = (None, "standard")
UNPRICEABLE_RESPONSE_ORIGIN = "request_upper_bound_unpriceable_response"


def _priceable(response, asked, uplift):
    """The token multiplier when the response can be priced from its own usage, else ``None``.

    Only a response that carries its stop reason (so its final usage arrived), whole
    token counts, the standard speed and tier, and a region that agrees with what was asked:
    the region asked decides the price, and the response can only confirm or contradict it.
    """
    usage = getattr(response, "usage", None)
    if getattr(response, "stop_reason", None) is None or not all(
            type(getattr(usage, name, None)) is int for name in ("input_tokens", "output_tokens")):
        return None
    if getattr(usage, "speed", None) not in STANDARD_SPEEDS or getattr(usage, "service_tier", None) not in STANDARD_TIERS:
        return None
    geo = getattr(usage, "inference_geo", None)
    if asked == "us":
        return uplift if geo in (None, "us") else None
    return 1.0 if geo in STANDARD_GEOS else None


def _install_anthropic_metering(meter: Meter, us_multiplier=None):
    from pipeline.cost import calculate_cost, pricing_for_provider_model
    from pipeline.llm import anthropic as upstream

    original = upstream.AnthropicLLM._create_message
    asked = INFERENCE_GEO
    uplift = None
    if asked == "us":  # pinned: without its frozen multiplier no call can be bounded or priced
        if (isinstance(us_multiplier, bool) or not isinstance(us_multiplier, (int, float))
                or not math.isfinite(us_multiplier) or us_multiplier < 1):
            raise RuntimeError("artisan_anthropic_us_multiplier_unavailable")
        uplift = float(us_multiplier)

    async def create_message(self, payload):
        if "context_management" in payload:
            # A server compaction's tokens are billed outside the request's bound.
            raise RuntimeError("artisan_anthropic_compaction_unmetered")
        betas = self._request_betas()
        count_payload = {key: value for key, value in payload.items() if key in ANTHROPIC_COUNT_KEYS}
        if betas:
            count = await self._client.beta.messages.count_tokens(**count_payload, betas=betas)
        else:
            count = await self._client.messages.count_tokens(**count_payload)
        rates = pricing_for_provider_model("anthropic", payload.get("model"))
        geo = {"inference_geo": asked} if asked else {}
        output, bound = _anthropic_bound(meter, rates, count.input_tokens, payload["max_tokens"], uplift)
        identity = meter.before("messages", payload.get("model"))
        try:
            # Standard capacity only: a Priority Tier answer would be priced above the bound.
            response = await original(self, {**payload, **geo, "max_tokens": output, "service_tier": "standard_only"})
        except Exception as error:
            meter.failed(identity, error, bound_usd=bound)
            fields = _anthropic_error_fields(error)
            if fields is not None and anthropic_no_funds(*fields):
                meter.mark_no_funds(identity)
                raise ProviderOutOfFunds() from error
            raise
        usage = upstream._extract_usage(response)
        multiplier = _priceable(response, asked, uplift)
        if multiplier is None:
            # A cut stream (no stop reason, so no final usage), an unknown region, speed or tier:
            # counted at the most the request could have cost, and left for the official figure.
            raw = getattr(response, "usage", None)
            meter.after(identity, bound, usage=usage, response_id=getattr(response, "id", None),
                        cost_origin=UNPRICEABLE_RESPONSE_ORIGIN,
                        failure={"reconciliation": "official_cost_required",
                                 "stop_reason": getattr(response, "stop_reason", None),
                                 **{name: getattr(raw, name, None) for name in ("inference_geo", "speed", "service_tier")}})
            return response
        served = getattr(response, "model", None)
        served_rates = pricing_for_provider_model("anthropic", served) if isinstance(served, str) else None
        iterations = getattr(getattr(response, "usage", None), "iterations", None) or []
        # Only plain sampling iterations are inside the top-level counts.
        itemized = all(getattr(item, "type", None) == "message" for item in iterations)
        cost = (calculate_cost(usage, served_rates).total_cost * multiplier
                if usage and served_rates and itemized else None)
        meter.after(identity, cost, usage=usage, response_id=getattr(response, "id", None))
        return response

    upstream.AnthropicLLM._create_message = create_message


def install_metering(meter: Meter, provider: str = "openai", *, us_inference_geo_multiplier=None):
    if provider not in ("openai", "anthropic"):
        raise ValueError("artisan_meter_provider_unknown")
    from pipeline.cost import calculate_cost, pricing_for_provider_model
    from pipeline.llm.openai_codec import extract_usage
    import openai
    from openai.resources.responses.responses import AsyncResponses, Responses
    # Both normal text calls and maintenance/compaction use these official
    # resources. Force upstream HTTP transport so no request escapes metering.
    os.environ["OPENAI_TRANSPORT"] = "http"
    count_keys = {"model", "input", "instructions", "tools", "tool_choice", "previous_response_id",
                  "parallel_tool_calls", "reasoning", "text", "truncation"}

    def bounded_output(kwargs, count):
        rates = pricing_for_provider_model("openai", kwargs.get("model"))
        if rates is None or type(count.input_tokens) is not int or count.input_tokens < 0:
            raise RuntimeError("artisan_request_price_or_token_count_unavailable")
        tokens = count.input_tokens
        high = tokens > rates.get("prompt_tier_threshold_tokens", math.inf)
        input_rate = max(rates["input_uncached"], rates.get("input_cache_write", 0),
                         rates.get("input_uncached_above_threshold", 0) if high else 0,
                         rates.get("input_cache_write_above_threshold", 0) if high else 0)
        output_rate = rates.get("output_above_threshold", rates["output"]) if high else rates["output"]
        remaining = meter.remaining() - tokens * input_rate / 1_000_000
        output = math.floor(remaining * 1_000_000 / output_rate)
        if output < 1:
            raise RuntimeError("artisan_shared_stage_authority_exhausted")
        output = min(output, kwargs.get("max_output_tokens") or 32768)
        bound = (tokens * input_rate + output * output_rate) / 1_000_000
        return {**kwargs, "max_output_tokens": output}, bound

    for cls in (AsyncResponses, Responses):
        for name in ("create", "compact"):
            original = getattr(cls, name)
            if cls is AsyncResponses:
                def wrap(original, name):
                    async def call(self, *args, **kwargs):
                        import asyncio
                        model = kwargs.get("model")
                        self._client.max_retries = 0
                        for attempt in range(1 + openai.DEFAULT_MAX_RETRIES):
                            identity = bound = None
                            try:
                                request = kwargs
                                if name == "create":
                                    count = await self.input_tokens.count(**{k: v for k, v in kwargs.items() if k in count_keys})
                                    request, bound = bounded_output(kwargs, count)
                                identity = meter.before(name, model)
                                response = await original(self, *args, **request)
                                break
                            except Exception as error:
                                await asyncio.sleep(_next_attempt(meter, identity, error, bound, attempt))
                        usage = extract_usage(response)
                        rates = pricing_for_provider_model("openai", model)
                        cost = calculate_cost(usage, rates).total_cost if usage and rates else None
                        meter.after(identity, cost, usage=usage, response_id=getattr(response, "id", None))
                        return response
                    return call
            else:
                def wrap(original, name):
                    def call(self, *args, **kwargs):
                        model = kwargs.get("model")
                        self._client.max_retries = 0
                        for attempt in range(1 + openai.DEFAULT_MAX_RETRIES):
                            identity = bound = None
                            try:
                                request = kwargs
                                if name == "create":
                                    count = self.input_tokens.count(**{k: v for k, v in kwargs.items() if k in count_keys})
                                    request, bound = bounded_output(kwargs, count)
                                identity = meter.before(name, model)
                                response = original(self, *args, **request)
                                break
                            except Exception as error:
                                time.sleep(_next_attempt(meter, identity, error, bound, attempt))
                        usage = extract_usage(response)
                        rates = pricing_for_provider_model("openai", model)
                        cost = calculate_cost(usage, rates).total_cost if usage and rates else None
                        meter.after(identity, cost, usage=usage, response_id=getattr(response, "id", None))
                        return response
                    return call
            setattr(cls, name, wrap(original, name))
    from pipeline.agents import design
    original_image = design._generate_image
    def image(prompt, out_png, *, image_paths=None, retries=1):
        identity = meter.before("generated_design_board", "configured_upstream_image_model")
        ok, detail = original_image(prompt, out_png, image_paths=image_paths, retries=1)
        accounting = detail.get("accounting") or {}
        meter.after(identity, accounting.get("budget_cost_usd"), usage=accounting,
                    cost_origin="upstream_image_budget_estimate")
        return ok, detail
    design._generate_image = image
    if provider == "anthropic":
        _install_anthropic_metering(meter, us_inference_geo_multiplier)
