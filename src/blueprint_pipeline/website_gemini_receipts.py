"""ADP-009B/day 14: one admitted Gemini dispatch per exact website input."""
from __future__ import annotations

import fcntl
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .website_task_context import reserve_website_preparation_spend, validate_website_task_context


def gemini_quote(*, model: str, input_tokens: int) -> float:
    # Standard Gemini 3.8 pricing, checked 2026-09-19. Reserve the entire
    # 65,536-token output window (including thinking), not just final JSON.
    # https://ai.google.dev/gemini-api/docs/pricing
    if model != "gemini-3.8-flash" or datetime.now(timezone.utc).year != 2026:
        raise ValueError("website_gemini_pricing_refresh_required")
    if not 0 < input_tokens <= 1_048_576:
        raise ValueError("website_gemini_input_budget_invalid")
    return math.ceil((input_tokens * 0.75 + 65_536 * 3.75) / 10_000) / 100


def retained_gemini_call(*, output_root: Path, binding: Mapping[str, Any],
                        task_context: Mapping[str, Any], maximum_cost_usd: float,
                        preflight: Callable[[], None], invoke: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Reuse completed results; an uncertain call never authorizes another call.

    The existing WebApp transaction is the cross-worker spending authority.
    Local flock and exclusive durable intent cover restarts on this worker.
    """
    validate_website_task_context(task_context, request_id=task_context["request_id"],
        scene_id=task_context["scene_id"], capture_id=task_context["capture_id"])
    request = {"binding": dict(binding), "task_context_digest": task_context["context_digest"],
               "maximum_cost_usd": maximum_cost_usd}
    digest = canonical_digest(request)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{digest[7:]}.json"
    with (output_root / f"{digest[7:]}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("website_gemini_request_in_progress") from exc
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt.get("status") != "completed":
                raise ValueError("website_gemini_requires_reconciliation")
            result = receipt.get("result")
            if (receipt.get("request_digest") != digest or not isinstance(result, dict)
                    or receipt.get("result_digest") != canonical_digest(result)):
                raise ValueError("website_gemini_receipt_invalid")
            return result
        preflight()
        admission, _grant = reserve_website_preparation_spend(task_context=task_context,
            binding_digest=digest, maximum_cost_usd=maximum_cost_usd, request_count=1,
            resource_class="evaluator_api", provider="google")
        with path.open("x") as stream:
            json.dump({"status": "submitting", "request_digest": digest,
                       "request": request, "admission": admission}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        result = invoke()
        receipt = {"status": "completed", "request_digest": digest, "request": request,
                   "admission": admission, "result": result, "result_digest": canonical_digest(result)}
        temporary = path.with_suffix(".tmp")
        write_json(temporary, receipt)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return result
