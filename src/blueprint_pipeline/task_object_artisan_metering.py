"""Thin metering around upstream API transports; no second agent/session loop.

Unknown charges retain the whole remaining reservation and refuse another call
until official reconciliation. Resume retains this file and its request count.
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

    def after(self, identity, cost, *, usage=None, response_id=None, cost_origin="upstream_priced_usage"):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self.read()
            row = next(c for c in data["calls"] if c["id"] == identity)
            known = isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0
            row.update(status="priced" if known else "reconciliation_required", cost_usd=cost if known else None,
                       usage=usage, response_id=response_id, cost_origin=cost_origin)
            self._write(data)
            if not known:
                raise RuntimeError("artisan_unknown_charge_requires_reconciliation")
            if sum(c.get("cost_usd") or 0 for c in data["calls"]) > self.cap:
                raise RuntimeError("artisan_reported_cost_exceeded_stage_authority")


def install_metering(meter: Meter):
    from pipeline.cost import calculate_cost, pricing_for_provider_model
    from pipeline.llm.openai_codec import extract_usage
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
        return {**kwargs, "max_output_tokens": min(output, kwargs.get("max_output_tokens") or 32768)}

    for cls in (AsyncResponses, Responses):
        for name in ("create", "compact"):
            original = getattr(cls, name)
            if cls is AsyncResponses:
                def wrap(original, name):
                    async def call(self, *args, **kwargs):
                        model = kwargs.get("model")
                        self._client.max_retries = 0
                        if name == "create":
                            count = await self.input_tokens.count(**{k: v for k, v in kwargs.items() if k in count_keys})
                            kwargs = bounded_output(kwargs, count)
                        identity = meter.before(name, model)
                        response = await original(self, *args, **kwargs)
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
                        if name == "create":
                            count = self.input_tokens.count(**{k: v for k, v in kwargs.items() if k in count_keys})
                            kwargs = bounded_output(kwargs, count)
                        identity = meter.before(name, model)
                        response = original(self, *args, **kwargs)
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
