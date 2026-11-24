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
    "NOT_WALL_CODES",
    "OWNER_CLOSED_CODE",
    "PROVIDER_ACCOUNT_STATUSES",
    "PROVIDER_OUTAGE_STATUSES",
    "PRIVACY_RIGHTS",
    "REFUSAL_RETRY_DELAYS_SECONDS",
    "RELEASE_SWITCHING_BLOCKERS",
    "hard_stop_kind",
    "owner_closed",
    "owner_closed_refusal",
    "outage_retry_delay",
    "RETRY_LATER_STATUSES",
    "provider_account_refused",
    "error_chain",
    "provider_answer",
    "provider_error_status",
    "provider_error_waits",
    "provider_funding_refused",
    "provider_refusal_unbilled",
    "provider_says_retry_later",
    "refusal_retry_delay",
    "release_refusal",
    "release_switching",
    "request_cannot_succeed",
]

PRIVACY_RIGHTS = "privacy_rights"
MONEY = "money"

#: The WebApp's typed answer once the site's owner closed this capture's run
#: (``websiteRunClosure.ts``: ``new_execution_authorized: false``,
#: ``explicit_resume_required: true``). It is not a delay and not a guess: the
#: run is closed for good, and only a new instruction from the site (a new
#: delivery) starts work again. Every stage ends the job on this exact code and
#: on nothing else; a 409 with any other code is handled as before.
OWNER_CLOSED_CODE = "website_run_closed_by_owner"
# The typed 409 as a whole token, as the listener reads authority endings (it may sit inside a longer
# message that joins several blockers, but nothing may touch it on either side).
_OWNER_CLOSED_REFUSAL = re.compile(r"(?<![A-Za-z0-9_-])website_control_([a-z0-9-]+)_http_409:"
                                   + OWNER_CLOSED_CODE + r"(?![A-Za-z0-9_-])")

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
#: Codes whose words read as a wall but that guard none (stress round 2, 2026-11-24). Each is taken out of the
#: text, as a whole token, before its words are read, so the rest of a longer message still counts:
#: - ``atlas_coverage_exceeds_frame_budget``: "budget" counts the views a rebuild may take, never money (as
#:   ``website_scene_handoff.REQUEST_SIZE_CODES`` count a request's tokens or bytes). It was read as a money wait
#:   and rechecked every 30 minutes until the job's deadline, for nothing.
#: - the WebApp's own configuration states, which it once sent as the consent and brief codes: Blueprint repairs
#:   them, the site never can, so "consent" in the name is no withdrawal and never ends the job.
#: - ``atlas_scene_planner_coverage_or_privacy_invalid``: the scene review's answer in the wrong form, no longer
#:   raised (``atlas_scene_planner.proposal_review`` names each part since its r2). Read as a privacy wall, it ended
#:   an Atlas job failed at once and no release could reopen it (the Austin test job, 2026-11-24). A reopened job
#:   runs the review again, whose unsafe-frame rule still refuses before anything is sent.
NOT_WALL_CODES = ("atlas_coverage_exceeds_frame_budget", "website_atlas_consent_unconfigured",
                  "website_scene_grant_policy_unconfigured", "atlas_scene_planner_coverage_or_privacy_invalid")
_NOT_WALL = re.compile(r"(?<![0-9a-z_])(?:" + "|".join(NOT_WALL_CODES) + r")(?![0-9a-z_])")


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


def owner_closed(code: object) -> bool:
    """Whether ``code`` is the WebApp's owner-closure answer: the bare code as the whole value, or
    its typed ``website_control_<operation>_http_409:`` refusal as a whole token.

    A longer token, another HTTP status or another reason is not a closure.
    """
    if not isinstance(code, str):
        return False
    return code.strip() == OWNER_CLOSED_CODE or owner_closed_refusal(code) is not None


def owner_closed_refusal(text: object) -> str | None:
    """The refusing WebApp operation when ``text`` carries the typed closure refusal as a whole token.

    The one match every caller uses: the retry classifier (through ``owner_closed``), the sweep that
    ends a closed job (``end_owner_closed_preparation``) and the Atlas preview step. The token may
    sit inside a longer message (``run_e2e failed: ...``), as the listener reads authority endings.
    """
    match = _OWNER_CLOSED_REFUSAL.search(text) if isinstance(text, str) else None
    return match.group(1) if match else None


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
    words = set(_WORD_SPLIT.split(_NOT_WALL.sub(" ", code.lower())))
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


#: An outside service refusing Blueprint's own key or account: unauthenticated (401), unpaid (402) or not permitted
#: (403). Nothing about the job's request is wrong, so it is an outage until the key or account is fixed: the
#: controller retries it on the outage schedule up to the job's deadline, and Blueprint is paged. A refused request
#: does no work at the provider; every retry still asks the job's own money admissions again.
PROVIDER_ACCOUNT_STATUSES = frozenset({401, 402, 403})


def provider_account_refused(code: object) -> bool:
    """Whether an outside service refused Blueprint's own key or account (``PROVIDER_ACCOUNT_STATUSES``)."""
    return provider_answer(code) in PROVIDER_ACCOUNT_STATUSES


#: A provider's own answer that means "try again later" when its SDK error carries it as a status: 429 or any 5xx.
PROVIDER_OUTAGE_STATUSES = frozenset({429}) | frozenset(range(500, 600))
#: A provider's words for an account with no money left on it (OpenAI's ``insufficient_quota``, "You have no
#: credits remaining"; Anthropic's "credit balance is too low"). Only the owner can fix it; the job waits as for
#: an outage and Blueprint is paged (``preparation_fault_alert``).
_FUNDING_REFUSAL = re.compile(r"insufficient_quota|no credits remaining|credit balance is too low"
                              r"|exceeded your current quota", re.IGNORECASE)
_CHAIN_LIMIT = 16


def error_chain(exc: BaseException):
    """``exc`` and the errors it was raised from or while handling, at most ``_CHAIN_LIMIT``, each once."""
    seen, current = set(), exc
    while current is not None and id(current) not in seen and len(seen) < _CHAIN_LIMIT:
        seen.add(id(current))
        yield current
        current = current.__cause__ if current.__cause__ is not None else current.__context__


def provider_error_status(exc: BaseException) -> int | None:
    """The HTTP status a provider answered with, read from the error's own ``code``, ``status_code`` or ``status``
    along its chain (as ``website_gemini_receipts._http_status`` reads it); None when no link carries one."""
    for link in error_chain(exc):
        for name in ("code", "status_code", "status"):
            value = getattr(link, name, None)
            if _is_int(value) and 100 <= value < 600:
                return value
    return None


def provider_funding_refused(text: object) -> bool:
    """Whether a provider's answer says Blueprint's account has no credit left (``_FUNDING_REFUSAL``)."""
    return isinstance(text, str) and _FUNDING_REFUSAL.search(text) is not None


def provider_error_waits(exc: BaseException) -> bool:
    """The one rule both website retry classifiers use for a provider SDK's error: it waits as an outage (2 min
    doubling to 1 h, never past the job's deadline, no refusal counted) when the provider answered 429 or a 5xx,
    refused Blueprint's own key or account (``PROVIDER_ACCOUNT_STATUSES``, from an error that carries the
    provider's response), or said the account has no credit left. A retry still asks every admission again."""
    status = provider_error_status(exc)
    if status in PROVIDER_OUTAGE_STATUSES:
        return True
    for link in error_chain(exc):
        if status in PROVIDER_ACCOUNT_STATUSES and getattr(link, "response", None) is not None:
            return True
        try:
            if provider_funding_refused(str(link)) or provider_funding_refused(getattr(link, "code", None)):
                return True
        except Exception:  # noqa: BLE001 - an unprintable error says nothing
            continue
    return False


#: A provider's code for an account past the spend limit set on it (Anthropic's 429, as
#: ``task_object_artisan_metering.anthropic_no_funds`` reads it).
_SPEND_LIMIT_CODES = frozenset({"enforced_spend_limit_reached"})


def provider_refusal_unbilled(exc: BaseException) -> bool:
    """Whether a provider SDK's error is a refusal known to have billed nothing: the account is unpaid (402), has
    no credit left (``provider_funding_refused``: OpenAI's 429 ``insufficient_quota``) or is past its spend limit,
    or the provider refused Blueprint's own key or account (401, 403, from an error that carries the provider's
    response). Each is also an outage (``provider_error_waits``). A plain 429 rate limit, a 5xx or a timeout is not
    known to be unbilled: a caller holds what it reserved for it."""
    status = provider_error_status(exc)
    if status == 402:
        return True
    for link in error_chain(exc):
        if status in (401, 403) and getattr(link, "response", None) is not None:
            return True
        code = getattr(link, "code", None)
        if isinstance(code, str) and code in _SPEND_LIMIT_CODES:
            return True
        try:
            if provider_funding_refused(str(link)) or provider_funding_refused(code):
                return True
        except Exception:  # noqa: BLE001 - an unprintable error says nothing
            continue
    return False


def provider_says_retry_later(code: object) -> bool:
    """Whether an outside service's answer means "try again later" (``RETRY_LATER_STATUSES`` or any 5xx)."""
    status = provider_answer(code)
    return status is not None and (status in RETRY_LATER_STATUSES or 500 <= status < 600)


def request_cannot_succeed(code: object) -> bool:
    """Whether sending this request again unchanged cannot succeed: an outside service refused the
    request itself (a 4xx that does not mean "try again later" and is no refusal of Blueprint's own key or
    account, ``provider_account_refused``), or the request cannot be sent at all."""
    if not isinstance(code, str):
        return False
    status = provider_answer(code)
    if status is not None:
        return 400 <= status < 500 and status not in RETRY_LATER_STATUSES | PROVIDER_ACCOUNT_STATUSES
    return code.partition(":")[0].strip().lower() in _UNSENDABLE_REQUEST_CODES


#: The active-release admission's answers while a release is switching: a step that started on the old
#: release and reached a paid stage after the link moved, or one that ran before the new release's receipt
#: existed (2026-11-16, 19:13). The stage still refuses to buy every time; the job waits as for an outage,
#: since the next attempt runs on the new release.
RELEASE_SWITCHING_BLOCKERS = frozenset({"gpu_canary_deployed_release_inactive_or_untrusted",
                                        "gpu_canary_deployed_release_receipt_unverified"})


def release_refusal(code: object) -> bool:
    """Whether ``code`` is a stage's ``<stage>_release_not_admitted:<blockers>`` refusal.

    It answers which release may buy, never anything a person can give the job: the step that met it
    raises it to its controller's retry schedule instead of holding the job for input.
    """
    if not isinstance(code, str):
        return False
    return code.partition(":")[0].strip().lower().endswith("_release_not_admitted")


def release_switching(code: object) -> bool:
    """A release refusal whose every blocker is a release switching (``RELEASE_SWITCHING_BLOCKERS``)."""
    if not release_refusal(code):
        return False
    blockers = {item for item in str(code).partition(":")[2].split(",") if item}
    return bool(blockers) and blockers <= RELEASE_SWITCHING_BLOCKERS
