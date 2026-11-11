"""One retry policy for every stage of a job: what happens after a step fails.

Only privacy, rights and money stop a job (AGENTS.md, "The Only Hard Stops").
A refusal that protects one of them ends or holds the job at once; nothing in
this module retries past it.

An outage (provider 429 or 5xx, transport, storage, lock contention) is
retried with capped exponential backoff until the job's own deadline, which
the caller enforces. No attempt count ends a job during an outage.

A refusal nobody has classified gets a bounded automatic schedule, about 46
hours across two daily deploy windows, and then the job ends by itself with
the refusal's own code. Every retry is timed: a deploy or an operator command
is never required to trigger one, it only changes what the next retry finds.

Pure: standard library only, no I/O, clock, randomness or environment.
"""

from __future__ import annotations

import re

__all__ = [
    "MONEY",
    "PRIVACY_RIGHTS",
    "REFUSAL_RETRY_DELAYS_SECONDS",
    "hard_stop_kind",
    "outage_retry_delay",
    "RETRY_LATER_STATUSES",
    "provider_answer",
    "provider_says_retry_later",
    "refusal_retry_delay",
    "request_cannot_succeed",
]

PRIVACY_RIGHTS = "privacy_rights"
MONEY = "money"

# 15 min, 1 h, 3 h, 6 h, 12 h, 24 h: 166,500 s in all.
REFUSAL_RETRY_DELAYS_SECONDS: tuple[int, ...] = (900, 3600, 10800, 21600, 43200, 86400)

# fmt: off
_PRIVACY_RIGHTS_WORDS = frozenset({
    "consent", "revoked", "revocation", "revoke", "withdrawn", "withdrawal",
    "rights", "license", "licence", "privacy", "takedown", "authority",
    "authorization", "authorisation", "unauthorized", "unauthorised",
})
_MONEY_WORDS = frozenset({
    "spend", "spent", "budget", "ceiling", "credit", "billing", "paid",
    "price", "cost", "money", "funding", "funds", "payment", "usd",
})
# fmt: on
_WORD_SPLIT = re.compile(r"[^0-9a-z]+")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def refusal_retry_delay(retries_done: int) -> int | None:
    """Seconds before the next automatic retry of an unclassified refusal.

    ``retries_done`` counts the automatic retries this failure streak already
    had. Returns None once the schedule is spent, and for a counter that is
    not a non-negative int, so a corrupt counter never retries forever.
    """
    if not _is_int(retries_done) or retries_done < 0:
        return None
    if retries_done >= len(REFUSAL_RETRY_DELAYS_SECONDS):
        return None
    return REFUSAL_RETRY_DELAYS_SECONDS[retries_done]


def outage_retry_delay(failures: int, *, first: int = 120, cap: int = 3600) -> int:
    """Seconds to wait after the ``failures``-th consecutive outage failure.

    ``min(cap, first * 2 ** (failures - 1))``; never None, because the job's
    deadline, not an attempt count, ends an outage. A ``failures`` that is not
    a positive int counts as 1.
    """
    if not (_is_int(first) and _is_int(cap)) or first <= 0 or cap <= 0 or first > cap:
        raise ValueError("job_retry_policy_bounds_invalid")
    if not _is_int(failures) or failures < 1:
        failures = 1
    # first * 2 ** cap.bit_length() already exceeds cap, so larger exponents
    # need not be computed.
    exponent = min(failures - 1, cap.bit_length())
    return min(cap, first * 2**exponent)


def hard_stop_kind(code: object) -> str | None:
    """Guess from a refusal code's words whether it protects privacy/rights or money.

    Returns PRIVACY_RIGHTS, MONEY or None; privacy/rights wins when both match.
    Whole lower-case words only, split on every non-alphanumeric character.
    The guess errs toward naming a hard stop: a controller uses it only to stop
    or wait sooner, never to skip a check. Retrying is safe at all because a
    retry never bypasses a gate, it re-asks it: consent, withdrawal and every
    money admission are checked again by the step itself on each attempt, so
    the gates stay fail-closed while the job keeps going.
    """
    if not isinstance(code, str) or not code:
        return None
    words = set(_WORD_SPLIT.split(code.lower()))
    if words & _PRIVACY_RIGHTS_WORDS:
        return PRIVACY_RIGHTS
    if words & _MONEY_WORDS:
        return MONEY
    return None


# An outside service's HTTP answer, as typed refusal codes name it ("meta_sam_http_413").
_PROVIDER_ANSWER = re.compile(r"[a-z0-9_-]+_http_([0-9]{3})")
_OWN_CONTROL_PREFIXES = ("website_control_", "owned_capture_atlas_control_")


def provider_answer(code: object) -> int | None:
    """The HTTP status an outside service answered, when ``code`` names one; else None.

    Only the code before any ``:`` detail counts. Such an answer is the service's
    outcome for one request, never a fact only a person can supply: a controller
    retries it on this module's schedules rather than waiting for input. Blueprint's
    own WebApp control answers (``website_control_...``) are not outside services:
    the typed reason after their ``:`` decides them.
    """
    if not isinstance(code, str):
        return None
    head = code.partition(":")[0].strip().lower()
    if head.startswith(_OWN_CONTROL_PREFIXES):
        return None
    match = _PROVIDER_ANSWER.fullmatch(head)
    return int(match.group(1)) if match else None


#: Answers that mean "try again later": a timeout, a conflict, too early, a rate limit, and every 5xx.
RETRY_LATER_STATUSES = frozenset({408, 409, 425, 429})

#: Requests this release can never send successfully as they stand: an inline clip whose last
#: encoding still does not fit (``meta_sam31``), and a prompt whose one allowed resend is used up
#: (owner decision 2026-11-07: never a second resend; the step ends rather than wait for a person).
_UNSENDABLE_REQUEST_CODES = frozenset({"meta_sam_inline_clip_exceeds_request_ceiling",
                                       "meta_sam_submission_requires_reconciliation",
                                       "meta_sam_caller_grant_covers_another_clip"})


def provider_says_retry_later(code: object) -> bool:
    """Whether an outside service's answer means "try again later" (``RETRY_LATER_STATUSES`` or any 5xx)."""
    status = provider_answer(code)
    return status is not None and (status in RETRY_LATER_STATUSES or 500 <= status < 600)


def request_cannot_succeed(code: object) -> bool:
    """Whether sending this request again unchanged cannot succeed: an outside service refused the
    request itself (a 4xx that does not mean "try again later"), or the request cannot be sent at all."""
    if not isinstance(code, str):
        return False
    status = provider_answer(code)
    if status is not None:
        return 400 <= status < 500 and status not in RETRY_LATER_STATUSES
    return code.partition(":")[0].strip().lower() in _UNSENDABLE_REQUEST_CODES
