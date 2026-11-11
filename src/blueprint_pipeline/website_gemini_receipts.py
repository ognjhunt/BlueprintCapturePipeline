"""ADP-009B/day 14: one admitted Gemini dispatch per exact website input, and one resend if it got no answer."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import require_paid_resource_admission_grant
from .website_task_context import reserve_website_preparation_spend, validate_website_task_context


def gemini_quote(*, model: str, input_tokens: int, max_output_tokens: int = 65_536) -> float:
    # Standard Gemini 3.8 pricing, checked 2026-09-19. The requested output
    # ceiling includes thinking and final output; default covers the full window.
    # https://ai.google.dev/gemini-api/docs/thinking#token-limits-and-max_output_tokens
    # https://ai.google.dev/gemini-api/docs/pricing
    # The rate is pinned to the model, not to a calendar year: a new model
    # needs a reviewed rate, and the job's admitted allowance bounds spend.
    if model != "gemini-3.8-flash":
        raise ValueError("website_gemini_pricing_refresh_required")
    if not 0 < input_tokens <= 1_048_576:
        raise ValueError("website_gemini_input_budget_invalid")
    if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or not 0 < max_output_tokens <= 65_536:
        raise ValueError("website_gemini_output_budget_invalid")
    return math.ceil((input_tokens * 0.75 + max_output_tokens * 3.75) / 10_000) / 100


def _request(*, binding: Mapping[str, Any], task_context: Mapping[str, Any], maximum_cost_usd: float,
             source_video_digest: str | None) -> tuple[dict[str, Any], str]:
    validate_website_task_context(task_context, request_id=task_context["request_id"],
        scene_id=task_context["scene_id"], capture_id=task_context["capture_id"], purpose="scene_preparation")
    request = {"binding": dict(binding), "task_context_digest": task_context["context_digest"],
               "maximum_cost_usd": maximum_cost_usd}
    if source_video_digest is not None:
        request["source_video_digest"] = source_video_digest
    return request, canonical_digest(request)


#: Owner decision 2026-11-10 ("same rule for Gemini"), after the SAM decision of 2026-11-07
#: ("Yes, resend once"): a Gemini call left without a completed result (its receipt still says
#: ``submitting`` because the process died, the call timed out or ``invoke`` raised) is counted as
#: spent, once, and sent once more. Its receipt is never deleted, rewritten or released: its
#: reservation stays held at its maximum. The resend is the same request marked with the digest it
#: resends, so the website reserves it afresh inside the job's existing caps; it has its own
#: receipt. A resend that is itself left without a result stops for review. A request that already
#: supersedes an incomplete one (``supersedes_incomplete_request_digest``) is its decision's resend
#: and gets none of its own.
UNANSWERED = "unanswered"
UNANSWERED_SCHEMA = "website_gemini_unanswered_dispatch.v1"
_HELD = "held_at_maximum_never_released_or_settled_as_zero"


def _receipt(path: Path) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    try:
        return raw, json.loads(raw)
    except ValueError:
        # A receipt cut short by a crash proves nothing, so it buys nothing.
        raise ValueError("website_gemini_requires_reconciliation") from None


def _completed_result(path: Path, digest: str) -> dict[str, Any]:
    receipt = _receipt(path)[1]
    if not isinstance(receipt, dict) or receipt.get("status") != "completed":
        raise ValueError("website_gemini_requires_reconciliation")
    result = receipt.get("result")
    if (receipt.get("request_digest") != digest or not isinstance(result, dict)
            or receipt.get("result_digest") != canonical_digest(result)):
        raise ValueError("website_gemini_receipt_invalid")
    return result


def _counted_as_spent(output_root: Path, digest: str, raw: bytes, receipt: Any) -> dict[str, Any] | None:
    """The unanswered receipt's counted-as-spent record, checked against it, or None before it is counted."""
    request, admission = receipt.get("request"), receipt.get("admission")
    if (receipt.get("status") != "submitting" or receipt.get("request_digest") != digest
            or not isinstance(request, dict) or canonical_digest(request) != digest
            or not isinstance(admission, dict) or admission.get("allocation_binding_digest") != digest
            or "resend_of" in request or (request.get("binding") or {}).get("supersedes_incomplete_request_digest")):
        # Not provably this request's own first call, or already its decision's resend.
        raise ValueError("website_gemini_requires_reconciliation")
    path = output_root / UNANSWERED / f"{digest[7:]}.json"
    if not path.exists() and not path.is_symlink():
        return None
    try:
        record = None if path.is_symlink() else json.loads(path.read_text())
    except ValueError:
        record = None
    resend = record.get("resend_request_digest") if isinstance(record, dict) else None
    if (not isinstance(record, dict) or record.get("schema_version") != UNANSWERED_SCHEMA
            or record.get("status") != "counted_as_spent" or record.get("outcome") != "unknown"
            or record.get("request_digest") != digest
            or record.get("receipt_sha256") != "sha256:" + hashlib.sha256(raw).hexdigest()
            or record.get("admission") != admission or record.get("reservation") != _HELD
            or record.get("resends_allowed") != 1 or not isinstance(resend, str) or len(resend) != 71
            or not resend.startswith("sha256:")):
        raise ValueError("website_gemini_unanswered_record_invalid")
    return record


def _count_once(output_root: Path, record: Mapping[str, Any]) -> None:
    path = output_root / UNANSWERED / f"{record['request_digest'][7:]}.json"
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix(".tmp")
    write_json(temporary, record)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.link(temporary, path)  # exclusive: a record is written once and never replaced
    temporary.unlink()


def answering_receipt(*, output_root: Path, binding: Mapping[str, Any], task_context: Mapping[str, Any],
                      maximum_cost_usd: float, source_video_digest: str | None = None,
                      earlier_requests: Sequence[tuple[Mapping[str, Any], float]] = ()) -> tuple[str, Path]:
    """The request digest and path of the completed receipt ``retained_gemini_call`` answers from
    for the same arguments: the decision's own, else its counted resend's. Reads only."""
    forms = [_request(binding=form, task_context=task_context, maximum_cost_usd=cost,
                      source_video_digest=source_video_digest)[1]
             for form, cost in ((binding, maximum_cost_usd), *earlier_requests)]
    bought = [form for form in forms if (output_root / f"{form[7:]}.json").exists()]
    if not bought:
        raise ValueError("website_gemini_requires_reconciliation")
    digest = bought[0]
    raw, receipt = _receipt(output_root / f"{digest[7:]}.json")
    if isinstance(receipt, dict) and receipt.get("status") != "completed" and len(bought) == 1:
        record = _counted_as_spent(output_root, digest, raw, receipt)
        if record is not None:
            digest = record["resend_request_digest"]
    _completed_result(output_root / f"{digest[7:]}.json", digest)
    return digest, output_root / f"{digest[7:]}.json"


def retained_gemini_call(*, output_root: Path, binding: Mapping[str, Any],
                        task_context: Mapping[str, Any], maximum_cost_usd: float,
                        preflight: Callable[[], None], invoke: Callable[[], dict[str, Any]],
                        source_video_digest: str | None = None,
                        earlier_requests: Sequence[tuple[Mapping[str, Any], float]] = ()) -> dict[str, Any]:
    """Reuse completed results; a call left without one is counted as spent and sent once more.

    ``earlier_requests`` are superseded ``(binding, maximum_cost_usd)`` forms of
    this same decision: one already bought is reused as it stands. One left
    without a result is counted as spent and this form is sent as the decision's
    one resend. The existing WebApp transaction is the cross-worker spending
    authority. Local flock and exclusive durable intent cover restarts on this worker.
    """
    request, digest = _request(binding=binding, task_context=task_context, maximum_cost_usd=maximum_cost_usd,
                               source_video_digest=source_video_digest)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / f"{digest[7:]}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("website_gemini_request_in_progress") from exc
        forms = [digest, *(_request(binding=earlier_binding, task_context=task_context, maximum_cost_usd=earlier_cost,
                                    source_video_digest=source_video_digest)[1]
                           for earlier_binding, earlier_cost in earlier_requests)]
        bought = [form for form in forms if (output_root / f"{form[7:]}.json").exists()]
        mark: dict[str, Any] = {}
        if bought:
            first = output_root / f"{bought[0][7:]}.json"
            raw, receipt = _receipt(first)
            if not isinstance(receipt, dict) or receipt.get("status") == "completed" or len(bought) > 1:
                # Reused as it stands; more than one form bought is already more than one call.
                return _completed_result(first, bought[0])
            record = _counted_as_spent(output_root, bought[0], raw, receipt)
            mark = {"resend_of": bought[0], "resend_index": 1}
            resend_digest = record["resend_request_digest"] if record else canonical_digest({**request, **mark})
            if (output_root / f"{resend_digest[7:]}.json").exists():
                # The decision's one resend: its result, or review when it too was left unanswered.
                return _completed_result(output_root / f"{resend_digest[7:]}.json", resend_digest)
            if resend_digest != canonical_digest({**request, **mark}):
                raise ValueError("website_gemini_requires_reconciliation")  # counted for another form's resend
        preflight()
        if mark:
            if record is None:
                _count_once(output_root, {
                    "schema_version": UNANSWERED_SCHEMA, "status": "counted_as_spent", "outcome": "unknown",
                    "request_digest": bought[0], "receipt_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                    "admission": receipt["admission"], "maximum_cost_usd": receipt["request"].get("maximum_cost_usd"),
                    "reservation": _HELD, "resends_allowed": 1, "resend_request_digest": resend_digest,
                    "recorded_at": datetime.now(timezone.utc).isoformat()})
            request, digest = {**request, **mark}, resend_digest
        path = output_root / f"{digest[7:]}.json"
        admission, grant = reserve_website_preparation_spend(task_context=task_context,
            binding_digest=digest, maximum_cost_usd=maximum_cost_usd, request_count=1,
            resource_class="evaluator_api", provider="google", source_video_digest=source_video_digest,
            atlas_auxiliary=source_video_digest is not None)
        with path.open("x") as stream:
            json.dump({"status": "submitting", "request_digest": digest,
                       "request": request, "admission": admission, **mark}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        require_paid_resource_admission_grant(grant, resource_class="evaluator_api",
            allocation_binding_digest=digest, require_allocation_binding=True)
        result = invoke()
        receipt = {"status": "completed", "request_digest": digest, "request": request,
                   "admission": admission, "result": result, "result_digest": canonical_digest(result), **mark}
        temporary = path.with_suffix(".tmp")
        write_json(temporary, receipt)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return result
