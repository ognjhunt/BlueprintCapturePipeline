"""ADP-009B/day 14: one admitted Gemini dispatch per exact website input."""
from __future__ import annotations

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
from .website_task_context import (
    RetainedAdmissionUnusable, reserve_website_preparation_spend, reuse_website_preparation_spend,
    validate_website_task_context,
)


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


#: An intent whose call got no recorded answer (killed mid-flight, timed out, or errored after
#: dispatch). The provider may have billed it, so it is counted as SPENT: its reservation stays
#: held at its maximum, which the WebApp ledger already debits against the job's ceiling (an
#: unsettled hold counts at its reserved amount), and nothing here releases or settles it as
#: zero. The call is then sent again once under one NEW reservation, which the WebApp admits
#: only inside the job's ceiling and the account limit; a refusal there is a money wait.
UNANSWERED_SCHEMA = "website_gemini_unanswered.v1"
RESEND_KIND = "website_gemini_resend_v1"
REPLACEMENT_KIND = "website_gemini_replacement_reservation_v1"
#: The call had no reservation of its own on record: no proof of what it may have spent.
UNRESERVED = "website_gemini_unanswered_call_spend_unreserved"


def _completed_result(path: Path, digest: str) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "completed":
        raise ValueError("website_gemini_requires_reconciliation")
    result = receipt.get("result")
    if (receipt.get("request_digest") != digest or not isinstance(result, dict)
            or receipt.get("result_digest") != canonical_digest(result)):
        raise ValueError("website_gemini_receipt_invalid")
    return result


def _fsynced(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


#: At most this many dispatches of one request may end with no recorded answer (the first and
#: two resends); then nothing more is sent and the job's timed schedule ends it by itself.
UNKNOWN_DISPATCH_LIMIT = 3
EXHAUSTED = "website_gemini_unanswered_dispatches_exhausted"
#: An answered refusal the provider gives before any work, as for SAM (``meta_sam31.REFUSAL_STATUSES``):
#: proof that nothing was billed, so the same reservation pays the next dispatch.
NO_WORK_STATUSES = frozenset({429, 503})


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status the provider answered with, from the error or its causes; None when there was no answer."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(seen) < 16:
        seen.add(id(current))
        for name in ("code", "status_code"):
            value = getattr(current, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return None


def _dispatch_chain(output_root: Path, digest: str):
    """Walk one request's dispatches: its first intent, then each resend.

    Returns ``(completed result or None, next dispatch digest, next intent path, resend marks,
    reservation to reuse or None)``. An intent the provider refused before any work, or one never
    dispatched, is proof of $0: its own reservation pays the next dispatch under the same digest,
    and it is neither counted nor a resend. Each unanswered intent on the way is counted as spent
    once (``unanswered-<k>.json``), so the next dispatch is the single resend one new reservation
    pays for; after ``UNKNOWN_DISPATCH_LIMIT`` unanswered dispatches nothing more is sent.
    """
    stem = digest[7:]
    dispatch, path, marks, index = digest, output_root / f"{stem}.json", {}, 0
    while path.exists():
        receipt = json.loads(path.read_text())
        if receipt.get("status") == "completed":
            return _completed_result(path, dispatch), dispatch, path, marks, None
        admission = receipt.get("admission")
        if (receipt.get("status") not in {"submitting", "refused", "not_dispatched", "superseded"}
                or receipt.get("request_digest") != dispatch
                or not isinstance(admission, Mapping) or admission.get("allocation_binding_digest") != dispatch
                or admission.get("status") != "admitted"):
            raise ValueError(UNRESERVED)
        if receipt.get("status") == "superseded":
            # Its hold, which bought nothing, can no longer pay a dispatch (expired or invalid):
            # one new reservation under a new digest replaces it; not counted, not a resend.
            marks = {**marks, "replaces": dispatch}
            dispatch = canonical_digest({"kind": REPLACEMENT_KIND, "request_digest": digest, "replaces": dispatch})
            path = output_root / f"{stem}.replacement-{dispatch[7:23]}.json"
            continue
        if receipt.get("status") != "submitting":
            return None, dispatch, path, marks, dict(admission)
        intent_sha256 = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        index += 1
        counted = output_root / f"{stem}.unanswered-{index}.json"
        if not counted.exists():
            _fsynced(counted, {"schema_version": UNANSWERED_SCHEMA, "status": "counted_as_spent", "outcome": "unknown",
                               "request_digest": digest, "dispatch_digest": dispatch, "intent_sha256": intent_sha256,
                               "reservation": "held_at_maximum_never_released_or_settled_as_zero",
                               "resends_allowed": 1})
        if index >= UNKNOWN_DISPATCH_LIMIT:
            raise ValueError(EXHAUSTED)
        marks = {"original_request_digest": digest, "resend_index": index, "resend_of": intent_sha256}
        dispatch = canonical_digest({"kind": RESEND_KIND, "request_digest": digest, "resend_index": index,
                                     "resend_of": intent_sha256})
        path = output_root / f"{stem}.resend-{index}.json"
    return None, dispatch, path, marks, None


def retained_gemini_call(*, output_root: Path, binding: Mapping[str, Any],
                        task_context: Mapping[str, Any], maximum_cost_usd: float,
                        preflight: Callable[[], None], invoke: Callable[[], dict[str, Any]],
                        source_video_digest: str | None = None,
                        earlier_requests: Sequence[tuple[Mapping[str, Any], float]] = ()) -> dict[str, Any]:
    """Reuse completed results; each reservation pays for exactly one dispatch of unknown outcome.

    Every Gemini call of website preparation goes through here. ``earlier_requests`` are
    superseded ``(binding, maximum_cost_usd)`` forms of this same decision: one already bought
    is reused as it stands. A dispatch with no recorded answer is counted as spent and sent
    again under one new reservation, at most twice (``_dispatch_chain``); a 429 or 503 answer,
    or a run stopped before its dispatch, reuses its own reservation. An earlier form's
    unanswered dispatch is counted the same way and this form is bought under its own
    reservation. The existing WebApp transaction is the cross-worker spending authority. Local
    flock and exclusive durable intent cover restarts on this worker.
    """
    request, digest = _request(binding=binding, task_context=task_context, maximum_cost_usd=maximum_cost_usd,
                               source_video_digest=source_video_digest)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / f"{digest[7:]}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("website_gemini_request_in_progress") from exc
        result, dispatch, path, marks, reuse = _dispatch_chain(output_root, digest)
        if result is not None:
            return result
        if not marks and reuse is None:
            for earlier_binding, earlier_cost in earlier_requests:
                _, earlier = _request(binding=earlier_binding, task_context=task_context,
                                      maximum_cost_usd=earlier_cost, source_video_digest=source_video_digest)
                retained = _dispatch_chain(output_root, earlier)[0]
                if retained is not None:
                    return retained
        preflight()
        if reuse is not None:
            try:
                # The same hold, never released, pays this dispatch: the last one provably bought nothing.
                grant = reuse_website_preparation_spend(admission=reuse, binding_digest=dispatch)
                admission = reuse
            except RetainedAdmissionUnusable:
                # Expired or invalid: keep its hold (never released here) and reserve once more.
                _fsynced(path, {**json.loads(path.read_text()), "status": "superseded"})
                result, dispatch, path, marks, reuse = _dispatch_chain(output_root, digest)
                if result is not None or reuse is not None:
                    raise ValueError(UNRESERVED) from None
        if reuse is None:
            admission, grant = reserve_website_preparation_spend(task_context=task_context,
                binding_digest=dispatch, maximum_cost_usd=maximum_cost_usd, request_count=1,
                resource_class="evaluator_api", provider="google", source_video_digest=source_video_digest,
                atlas_auxiliary=source_video_digest is not None)
            with path.open("x") as stream:
                json.dump({"status": "submitting", "request_digest": dispatch,
                           "request": request, "admission": admission, **marks}, stream)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            _fsynced(path, {"status": "submitting", "request_digest": dispatch, "request": request,
                            "admission": admission, **marks})
        try:
            require_paid_resource_admission_grant(grant, resource_class="evaluator_api",
                allocation_binding_digest=dispatch, require_allocation_binding=True)
        except BaseException:
            # Stopped before the dispatch started (a refused grant, a lost lease): proof of $0.
            _fsynced(path, {"status": "not_dispatched", "request_digest": dispatch, "request": request,
                            "admission": admission, **marks})
            raise
        try:
            result = invoke()
        except Exception as exc:
            status = _http_status(exc)
            if status in NO_WORK_STATUSES:
                _fsynced(path, {"status": "refused", "http_status": status, "request_digest": dispatch,
                                "request": request, "admission": admission, **marks})
            raise
        _fsynced(path, {"status": "completed", "request_digest": dispatch, "request": request,
                        "admission": admission, "result": result, "result_digest": canonical_digest(result), **marks})
        return result
