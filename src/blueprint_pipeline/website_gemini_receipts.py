"""ADP-009B/day 14: one admitted Gemini dispatch per exact website input.

Every Gemini call a website job makes is reserved on the job's ledger at its quote before it is sent
(``reserve_website_preparation_spend``), so the job's $40 cap counts it first. Once answered, the call is
settled at what its reported usage costs (``note_gemini_usage``, ``gemini_usage_cost``): the job's ledger
row releases the rest of its quote (``preparation-settlement``) and the job's meter records the cost
(``paid_spend_meter``, provider ``gcp``, where Gemini bills). A call whose usage is not fully reported
keeps its quote on the ledger and is metered at it; a dispatch with no answer is metered at its quote.
"""
from __future__ import annotations

import contextvars
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .hard_stops import MoneyStop
from .paid_resource_admission import require_paid_resource_admission_grant
from .website_task_context import (
    RetainedAdmissionUnusable, require_current_website_source, reserve_website_preparation_spend,
    reuse_website_preparation_spend, validate_website_task_context,
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


#: The provider name a Gemini call is metered under: Gemini bills to the Google Cloud project, so its meter rows
#: sit beside that project's bill in the account's reconciliation.
METER_PROVIDER = "gcp"
_USAGE: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar("website_gemini_usage",
                                                                                      default=None)


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def gemini_response_usage(response: Any) -> dict[str, Any]:
    """Billed input and output tokens a Gemini answer reports, or ``{"status": "unknown"}``.

    ``generate_content`` reports ``usage_metadata``; an Interactions answer reports ``usage``. Thinking is billed
    as output. A total above the parts is read as output (an upper bound), never dropped.
    """

    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        prompt, tool = _count(getattr(meta, "prompt_token_count", None)), _count(
            getattr(meta, "tool_use_prompt_token_count", None))
        visible, thoughts = _count(getattr(meta, "candidates_token_count", None)), _count(
            getattr(meta, "thoughts_token_count", None))
        total = _count(getattr(meta, "total_token_count", None))
    else:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"status": "unknown"}
        prompt, tool = _count(getattr(usage, "total_input_tokens", None)), 0
        visible, thoughts = _count(getattr(usage, "total_output_tokens", None)), _count(
            getattr(usage, "total_thought_tokens", None))
        total = _count(getattr(usage, "total_tokens", None))
    if prompt is None:
        return {"status": "unknown"}
    input_tokens = prompt + (tool or 0)
    if total is not None and total >= input_tokens:
        output_tokens = total - input_tokens
    elif visible is not None and thoughts is not None and total is None:
        # With neither a total nor a thinking count, the thinking billed as output is unknown.
        output_tokens = visible + thoughts
    else:
        return {"status": "unknown"}
    return {"status": "reported", "input_tokens": input_tokens, "output_tokens": output_tokens}


def note_gemini_usage(response: Any) -> None:
    """Record one provider answer's usage for the call ``retained_gemini_call`` is making; elsewhere a no-op.

    Every ``generate_content`` or Interactions answer inside a retained call's ``invoke`` is noted, a repair turn
    included, so the call settles at what all of its turns cost.
    """

    calls = _USAGE.get()
    if calls is not None:
        calls.append(gemini_response_usage(response))


def gemini_usage_cost(*, model: Any, calls: Sequence[Mapping[str, Any]]) -> float | None:
    """What the noted answers cost at ``gemini_quote``'s rates, rounded up to the micro-dollar; None if unknown."""

    if model != "gemini-3.8-flash" or not calls or any(call.get("status") != "reported" for call in calls):
        return None
    micro = sum(call["input_tokens"] * 0.75 + call["output_tokens"] * 3.75 for call in calls)
    return math.ceil(micro) / 1_000_000


def _meter_settlement(*, job_id: Any, meter_key: str, settled_usd: float, basis: str) -> None:
    """Record a Gemini call's cost in the job's meter (``paid_spend_meter``); best effort, never raises."""

    try:
        from .paid_spend_meter import meter_root, record_settlement
        root = meter_root()
        if root is not None and job_id:
            record_settlement(job_id=job_id, meter_key=meter_key, settled_usd=settled_usd, basis=basis,
                              provider=METER_PROVIDER, root=root)
    except Exception:  # noqa: BLE001 - metering is visibility; the job's ledger holds the quote
        return


def _settle_answered(*, path: Path, task_context: Mapping[str, Any], dispatch: str,
                     admission: Mapping[str, Any], model: Any, calls: Sequence[Mapping[str, Any]]) -> None:
    """Settle an answered call on the job's ledger at its usage cost, and meter it; never raises.

    The ledger row keeps its quote unless the WebApp accepts the settlement (an older release refuses a Gemini
    settlement): the cap then over-counts, never under-counts. The meter records the usage cost when every turn
    reported it, else the quote.
    """

    try:
        maximum = float(admission.get("maximum_cost_usd"))
        cost = gemini_usage_cost(model=model, calls=calls)
        if cost is None or not 0 <= cost <= maximum:
            _meter_settlement(job_id=task_context.get("request_id"), meter_key=dispatch, settled_usd=maximum,
                              basis="quote_upper_bound")
            return
        _meter_settlement(job_id=task_context.get("request_id"), meter_key=dispatch, settled_usd=cost,
                          basis="reported_usage")
        from .website_task_context import preparation_webapp_request
        command = {"task_context_digest": task_context["context_digest"], "allocation_binding_digest": dispatch,
                   "provider": "google", "completed_request_count": 1, "provider_charge_amount_usd": cost,
                   "usage_receipt_digest": canonical_digest({"model": model, "calls": list(calls)})}
        receipt = preparation_webapp_request(task_context=task_context, operation="preparation-settlement",
            payload={"request_id": task_context["request_id"], "scene_id": task_context["scene_id"],
                     "settlement": command})
        if receipt.get("status") == "settled" and all(receipt.get(key) == value for key, value in command.items()):
            _fsynced(path.with_suffix(".settlement.json"), receipt)
    except Exception:  # noqa: BLE001 - the answer is bought; an unsettled row keeps its quote
        return


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
#: Gemini's own answer when the account's prepaid balance is spent: 402 RESOURCE_EXHAUSTED, "Your prepayment
#: credits are depleted" (test job 3, 2026-11-17). It refuses before any work, so, like a 429 or 503, its
#: reservation pays the next dispatch and it is never counted as spent. It is a money wait (``CreditDepleted``):
#: the code reads as money (``job_retry_policy.hard_stop_kind``), so the job is asked again on the money schedule
#: and goes on by itself once the account has funds. No other 402 is read as this.
CREDIT_DEPLETED = "website_gemini_provider_credit_depleted"
_DEPLETED_STATUS, _DEPLETED_REASON = "RESOURCE_EXHAUSTED", "prepayment credits are depleted"


class CreditDepleted(MoneyStop):
    """Gemini's credits-depleted answer, a money wall under ``CREDIT_DEPLETED``."""

    def __init__(self) -> None:
        super().__init__(CREDIT_DEPLETED)


#: For a job with a grant, the longest wait between two money rechecks (``job_retry_policy.outage_retry_delay``'s
#: cap; ``website_preparation_retry.retry_delay``).
MONEY_RECHECK_SECONDS = 3600


def last_money_recheck(error: BaseException, now: float, window_ends_at_epoch: float | None) -> bool:
    """Whether ``error`` is Gemini's credits-depleted answer met so near the end of the job's window, as read when
    asked (``website_preparation_terminal.retained_window_end``), that the money wait may not ask again before it
    ends (the longest wait, and as long again for the listener's tick and the run): then, and only then, the job
    goes on without that piece. A window that cannot be read is never near its end."""
    return (isinstance(error, CreditDepleted) and window_ends_at_epoch is not None
            and window_ends_at_epoch - now <= 2 * MONEY_RECHECK_SECONDS)


class LaterTurnFailed(Exception):
    """Raised by a multi-turn ``invoke`` (a repair turn after the first answer) when a turn fails after an earlier
    turn of the same dispatch was answered: the provider did work for this dispatch, so it is never read as
    refused, whatever the later turn's status; it is counted as spent and the next send takes a new reservation.
    It chains the turn's own error and keeps its text, so a credits-depleted answer still waits on money and the
    retry lane reads the failure as before."""

    def __init__(self, error: BaseException):
        super().__init__(str(error))
        self.__cause__ = error


def answered_before(exc: BaseException) -> bool:
    """Whether ``LaterTurnFailed`` marks ``exc`` or one of its causes: it wins over the status found below it."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(seen) < 16:
        seen.add(id(current))
        if isinstance(current, LaterTurnFailed):
            return True
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return False


def credit_depleted(exc: BaseException) -> bool:
    """Whether Gemini answered 402 with its own reason that the prepaid credits are depleted (the client's
    ``code``, ``status`` and ``message``, or the same in its ``details``), on the error or one of its causes."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(seen) < 16:
        seen.add(id(current))
        details = getattr(current, "details", None)
        body = details.get("error") if isinstance(details, Mapping) else None
        body = body if isinstance(body, Mapping) else {}
        code = getattr(current, "code", body.get("code"))
        status, message = getattr(current, "status", None) or body.get("status"), \
            getattr(current, "message", None) or body.get("message")
        if (code == 402 and not isinstance(code, bool) and status == _DEPLETED_STATUS
                and isinstance(message, str) and _DEPLETED_REASON in message.lower()):
            return True
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return False


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


def _dispatch_chain(output_root: Path, digest: str, job_id: str | None = None):
    """Walk one request's dispatches: its first intent, then each resend.

    Returns ``(completed result or None, next dispatch digest, next intent path, resend marks,
    reservation to reuse or None)``. An intent the provider refused before any work (a 429, a 503, or
    Gemini's credits-depleted 402, each kept on its own receipt), or one never dispatched, is proof of $0:
    its own reservation pays the next dispatch under the same digest, and it is neither counted nor a
    resend. Such an intent is always its chain's latest record, so no resend follows it. Each intent whose
    receipt holds no answer is counted as spent once (``unanswered-<k>.json``), whatever else is recorded
    about its run, so the next dispatch is the single resend one new reservation pays for; after
    ``UNKNOWN_DISPATCH_LIMIT`` unanswered dispatches nothing more is sent. ``job_id``, when given, meters each
    unanswered dispatch at its quote in the job's meter as it is first counted.
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
            _meter_settlement(job_id=job_id, meter_key=dispatch, settled_usd=admission.get("maximum_cost_usd"),
                              basis="unanswered_counted_as_spent")
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
                        earlier_requests: Sequence[tuple[Mapping[str, Any], float]] = (),
                        accept_earlier: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None) -> dict[str, Any]:
    """Reuse completed results; each reservation pays for exactly one dispatch of unknown outcome.

    Every Gemini call of website preparation goes through here. ``earlier_requests`` are
    superseded ``(binding, maximum_cost_usd)`` forms of this same decision: one already bought
    is reused as it stands. A dispatch with no recorded answer is counted as spent and sent
    again under one new reservation, at most twice (``_dispatch_chain``), per form: an earlier form whose bound is
    spent leaves this form to be bought under its own; a 429 or 503 answer, Gemini's
    credits-depleted 402 (raised as ``CreditDepleted``, a money wait), or a run stopped before its
    dispatch, reuses its own reservation, unless an earlier turn of the same dispatch was answered
    (``LaterTurnFailed``, raised by a multi-turn ``invoke``): then it is counted as spent. An earlier form's
    unanswered dispatch is counted the same way and this form is bought under its own
    reservation. ``accept_earlier(earlier_binding, result)``, when given, says which earlier answer still stands; one it
    refuses is kept as it is and this form is bought once. The existing WebApp transaction is the
    cross-worker spending authority. Local
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
        job_id = task_context.get("request_id")
        result, dispatch, path, marks, reuse = _dispatch_chain(output_root, digest, job_id)
        if result is not None:
            return result
        if not marks and reuse is None:
            for earlier_binding, earlier_cost in earlier_requests:
                _, earlier = _request(binding=earlier_binding, task_context=task_context,
                                      maximum_cost_usd=earlier_cost, source_video_digest=source_video_digest)
                try:
                    retained = _dispatch_chain(output_root, earlier, job_id)[0]
                except ValueError as exc:
                    if str(exc) != EXHAUSTED:
                        raise
                    # The earlier form spent its own bound of unanswered sends and holds no answer. Nothing more
                    # is sent in that form; this form is another request, bought under its own reservation and
                    # its own bound, as for an earlier form with fewer unanswered sends.
                    retained = None
                if retained is not None and (accept_earlier is None or accept_earlier(earlier_binding, retained)):
                    return retained
        preflight()
        if reuse is not None:
            # A held reservation skips the website's spend transaction, so the capture's current
            # source is read first: nothing is sent under it once the site has withdrawn.
            require_current_website_source(task_context)
            try:
                # The same hold, never released, pays this dispatch: the last one provably bought nothing.
                grant = reuse_website_preparation_spend(admission=reuse, binding_digest=dispatch, job_id=job_id)
                admission = reuse
            except RetainedAdmissionUnusable:
                # Expired or invalid: keep its hold (never released here) and reserve once more.
                _fsynced(path, {**json.loads(path.read_text()), "status": "superseded"})
                result, dispatch, path, marks, reuse = _dispatch_chain(output_root, digest, job_id)
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
        noted = _USAGE.set([])
        try:
            result = invoke()
        except Exception as exc:
            status, depleted = _http_status(exc), credit_depleted(exc)
            # Refused before any work only when no turn of this dispatch was answered (``LaterTurnFailed``):
            # otherwise it keeps no answer, is counted as spent, and the next send takes a new reservation.
            if (status in NO_WORK_STATUSES or depleted) and not answered_before(exc):
                _fsynced(path, {"status": "refused", "http_status": status,
                                **({"reason": CREDIT_DEPLETED} if depleted else {}), "request_digest": dispatch,
                                "request": request, "admission": admission, **marks})
            if depleted:
                raise CreditDepleted() from exc
            raise
        finally:
            calls = _USAGE.get() or []
            _USAGE.reset(noted)
        usage = {"model": binding.get("model"), "calls": calls,
                 "cost_usd": gemini_usage_cost(model=binding.get("model"), calls=calls)}
        _fsynced(path, {"status": "completed", "request_digest": dispatch, "request": request,
                        "admission": admission, "result": result, "result_digest": canonical_digest(result),
                        "usage": usage, **marks})
        _settle_answered(path=path, task_context=task_context, dispatch=dispatch, admission=admission,
                         model=binding.get("model"), calls=calls)
        return result
