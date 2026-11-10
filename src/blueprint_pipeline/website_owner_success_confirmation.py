"""The site owner's decision on a website job's success thresholds, made on the job page.

A website job's thresholds are Blueprint's fixed development controls, marked
``proposal_only``. The team packet compiles and screening scores the job on
them at once, marked as assumed and not confirmed by the site
(``website_owner_success_assumption``); nothing waits for the owner. The owner
can still confirm them (``task_evaluation_owner_success_confirmation``). The
operator door records that confirmation by hand. This records the same
confirmation from the website, with no operator.

On the team packet queue's timer pass (``team_scene_packet_worker.
requeue_transient_blocked``, run by the screening timer), each completed
request whose half was compiled on assumed thresholds (and each request an
older release left blocked on the owner contract) sends its intent's published
thresholds to the WebApp over the existing signed Pipeline sync transport
(``POST /api/internal/pipeline/task-evaluation-scene-intents/<intent>/owner-success-proposal``).
The WebApp keeps the proposal, shows it to the job's owner (the owner link
only) in plain words with Confirm and "this is wrong", and answers with the
owner's decision on exactly this proposal, if there is one:

- ``confirmed``: the decision must name this proposal's digest, revision and
  binding. The record is written by ``confirm_owner_success_contract``, the
  door's own writer, under the same validators, with ``issuer_channel``
  ``website``. The same pass then returns the request to pending, as it does
  for a door confirmation, and the team packet compiles the half once more,
  with the owner contract as the owner's. Results already scored stay bound to
  the assumed contract they were scored under.
- ``held``: the owner said the thresholds are wrong, with a typed reason.
  Nothing is written under the intent and no number changes. The request's
  status keeps the answer and the reason, and the WebApp alerts Blueprint's
  team; the half and its screening go on as they were, still marked as
  assumed. (A request an older release left blocked stays blocked.)
- ``pending``: no decision yet.

Each request keeps its next check time in its status (``STATUS_FIELD``) and is
asked again only when that time comes, or sooner when its intent's
progression changes and the proposal it then builds is a new one. The wait
doubles while the answer stays the same, separately for ``pending``, ``held``,
a WebApp refusal (4xx) and a transport failure, up to a cap (``BACKOFF``), so
a job whose owner never answers, or a WebApp that refuses, costs a few signed
calls a day rather than one a minute. Any error in one job's exchange is
recorded as ``failed`` for that job and never stops the pass.

A republished job (another revision, success document, threshold or proposal)
has another proposal digest, so a decision on the old proposal is never
applied; the WebApp asks the owner again. A revoked intent sends nothing. The
door path is unchanged. Nothing here starts compute, calls a provider or moves
money.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit

from .decision_evidence_contracts import cross_runtime_canonical_digest

NOTICE_SCHEMA = "website_owner_success_proposal.v1"
RECEIPT_SCHEMA = "website_owner_success_proposal_receipt.v1"
DECISION_SCHEMA = "website_owner_success_decision.v1"
# Why an owner says the thresholds are wrong; the WebApp offers exactly these.
HOLD_REASONS = frozenset({"wrong_goal", "too_strict", "too_loose", "wrong_task", "other"})
DECISION_FIELDS = frozenset({"schema_version", "intent_id", "intent_digest", "request_id", "capture_id",
                             "proposal_digest", "configured_scene_revision_digest", "binding_digest", "decision",
                             "reason", "decided_via", "decided_at", "decision_digest"})
# The team packet status field that keeps the latest exchange for its request.
STATUS_FIELD = "website_owner_success"
# Each pass asks about at most this many blocked jobs, so an unreachable WebApp costs a bounded wait.
MAX_EXCHANGES_PER_PASS = 4
TIMEOUT_SECONDS = 5
# (first wait, longest wait) in seconds before a request is asked again, by its last outcome; the
# wait doubles each time the same outcome repeats for the same proposal.
BACKOFF = {"pending": (120, 3600), "held": (1800, 86400), "refused": (900, 86400), "failed": (120, 21600),
           "confirmed": (3600, 86400), "already_confirmed": (3600, 86400), "intent_unbound": (3600, 86400),
           "intent_revoked": (3600, 86400)}
# What each forbidden contact class counts, and the contact it leaves out, as the native scorer defines
# them (``task_evaluation_task_parameter_proposal`` ``contact_class_semantics``; one entry per class in
# ``native_rigid_episode_telemetry.CONTACT_CHANNELS``). The WebApp words the owner's page from these.
CONTACT_CLASS_SCOPE = {
    "robot_background": {"parts": ["robot_protected_bodies", "scene_including_destination_support"], "excluding": []},
    "robot_object": {"parts": ["robot_protected_bodies", "task_object"],
                     "excluding": ["admitted_gripper_task_contact_bodies"]},
    "object_background": {"parts": ["task_object", "scene"], "excluding": ["destination_support"]},
    "destination_background": {"parts": ["destination", "scene"], "excluding": ["qualified_placement_support"]},
}
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,131}\Z")
_INSTANT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z\Z")
_CODE = re.compile(r"[a-z][a-z0-9_]{0,99}\Z")
Transport = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("website_owner_success_" + code)


def proposal_notice(bound: Mapping[str, Any]) -> dict[str, Any]:
    """What the WebApp shows the owner: the job's thresholds as built, bound to its whole binding.

    ``bound`` is ``task_evaluation_owner_success_confirmation.binding`` for the
    published job. ``binding_digest`` seals all of it (owner, revision, success
    document, thresholds, proposal); ``proposal_digest`` seals this notice, so an
    owner's decision on it binds every one of those.
    """

    from .task_evaluation_scene_intent_contracts import canonical_digest

    notice = {"schema_version": NOTICE_SCHEMA, "intent_id": bound["intent_id"],
              "intent_digest": bound["intent_digest"], "task_id": bound["task_id"],
              "configured_scene_revision_digest": bound["configured_scene_revision_digest"],
              "configured_success_criteria": json.loads(json.dumps(bound["configured_success_criteria"])),
              "configured_success_criteria_digest": bound["configured_success_criteria_digest"],
              "proposal_authority_digest": bound["proposal_authority_digest"],
              "forbidden_contacts": forbidden_contacts(bound["configured_success_criteria"]),
              "binding_digest": canonical_digest(dict(bound))}
    notice["proposal_digest"] = cross_runtime_canonical_digest(notice)
    return notice


def forbidden_contacts(criteria: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each forbidden contact class the thresholds name, with what it counts and what it leaves out.

    A class the scorer does not define here is sent with no parts, so the page
    shows its name rather than a guess.
    """

    classes = criteria.get("forbidden_contact_classes")
    if not isinstance(classes, list) or not all(isinstance(name, str) for name in classes):
        return []
    empty: dict[str, list[str]] = {"parts": [], "excluding": []}
    return [{"contact_class": name, **json.loads(json.dumps(CONTACT_CLASS_SCOPE.get(name, empty)))}
            for name in classes]


def send_proposal(notice: Mapping[str, Any]) -> dict[str, Any]:
    """POST the proposal with the existing signed Pipeline -> WebApp transport; return the WebApp's answer."""

    from .safe_outbound_http import pinned_api_policy, request
    from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
    from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url

    configured = os.getenv("PIPELINE_SYNC_WEBAPP_URL", "").strip()
    _require(bool(configured), "webapp_url_missing")
    parsed = urlsplit(validated_https_sync_url(configured))
    origin = f"{parsed.scheme}://{parsed.netloc}"
    body = json.dumps(dict(notice), separators=(",", ":"), allow_nan=False).encode()
    url = (f"{origin}/api/internal/pipeline/task-evaluation-scene-intents/"
           f"{quote(str(notice['intent_id']), safe='')}/owner-success-proposal")
    try:
        response = request(url, method="POST", data=body,
                           headers=_pipeline_sync_headers(load_pipeline_sync_token(), body),
                           timeout_seconds=TIMEOUT_SECONDS,
                           policy=pinned_api_policy(origin, max_response_bytes=20_000), max_response_bytes=20_000)
    except HTTPError as exc:
        code = None
        try:
            failure = json.loads(exc.read(4096))
            code = failure.get("code") if isinstance(failure, dict) else None
        except (ValueError, OSError):
            pass
        finally:
            exc.close()
        suffix = f":{code}" if isinstance(code, str) and _CODE.fullmatch(code) else ""
        raise ValueError(f"website_owner_success_http_{exc.code}{suffix}") from None
    value = json.loads(response.body)
    _require(isinstance(value, dict), "receipt_invalid")
    return value


def validated_receipt(value: Any, notice: Mapping[str, Any]) -> dict[str, Any]:
    """The WebApp's answer for exactly this proposal; anything else refuses (and so applies nothing)."""

    _require(isinstance(value, Mapping) and set(value) == {"schema_version", "status", "intent_id",
                                                             "proposal_digest", "decision"}
             and value["schema_version"] == RECEIPT_SCHEMA and value["intent_id"] == notice["intent_id"]
             and value["proposal_digest"] == notice["proposal_digest"]
             and value["status"] in {"pending", "confirmed", "held"}, "receipt_invalid")
    decision = value["decision"]
    if value["status"] == "pending":
        _require(decision is None, "receipt_invalid")
        return dict(value)
    _require(isinstance(decision, Mapping) and set(decision) == DECISION_FIELDS
             and decision["schema_version"] == DECISION_SCHEMA and decision["decision"] == value["status"]
             and all(decision[field] == notice[field] for field in (
                 "intent_id", "intent_digest", "proposal_digest", "configured_scene_revision_digest",
                 "binding_digest"))
             and decision["decided_via"] == "owner_job_link"
             and all(isinstance(decision[field], str) and _ID.fullmatch(decision[field])
                     for field in ("request_id", "capture_id"))
             and isinstance(decision["decided_at"], str) and _INSTANT.fullmatch(decision["decided_at"])
             and (decision["reason"] is None if decision["decision"] == "confirmed"
                  else decision["reason"] in HOLD_REASONS)
             and isinstance(decision["decision_digest"], str) and _DIGEST.fullmatch(decision["decision_digest"])
             and decision["decision_digest"] == cross_runtime_canonical_digest(
                 dict(decision), digest_field="decision_digest"), "decision_invalid")
    return {**value, "decision": dict(decision)}


def record_website_confirmation(*, intent_root: str | Path, intent: Mapping[str, Any],
                                decision: Mapping[str, Any]) -> dict[str, Any]:
    """Write the owner's website confirmation with the door's own writer, channel ``website``.

    The preview is recomputed from the retained published job and must still be
    the proposal the owner confirmed; the writer recomputes it again under the
    intake lock. The record names the intent's owner, as the door's does, and
    its evidence names the WebApp decision.
    """

    from .task_evaluation_owner_success_confirmation import (
        _BINDING_FIELDS,
        ACK,
        WEBSITE_CHANNEL,
        confirm_owner_success_contract,
        preview_owner_success_confirmation,
    )

    _require(decision.get("decision") == "confirmed", "decision_not_a_confirmation")
    owner = dict(intent["request"]["owner"])
    evidence = (f"site owner confirmed these thresholds on the website job page (owner link): "
                f"webapp decision {decision['decision_digest']} at {decision['decided_at']}, "
                f"proposal {decision['proposal_digest']}")
    planned = preview_owner_success_confirmation(queue_root=intent_root, intent_id=intent["intent_id"], owner=owner,
                                                 approved_by=owner["user_id"], approval_evidence=evidence)
    bound = {field: planned["preview"][field] for field in _BINDING_FIELDS}
    _require(proposal_notice(bound)["proposal_digest"] == decision["proposal_digest"], "proposal_changed")
    reference = (f"owner_approval:approved_by={owner['user_id']};evidence=webapp_decision:"
                 f"{decision['decision_digest']};channel={WEBSITE_CHANNEL};request_id={decision['request_id']};"
                 f"proposal_digest={decision['proposal_digest']}")
    return confirm_owner_success_contract(
        queue_root=intent_root, intent_id=intent["intent_id"], owner=owner, approved_by=owner["user_id"],
        approval_evidence=evidence, apply_ack=planned["apply_ack"], authorization_reference=reference, ack=ACK,
        channel=WEBSITE_CHANNEL)


def _owner_contract_blocked(status: Mapping[str, Any]) -> bool:
    from .team_scene_packet_worker import OWNER_CONTRACT_BLOCKER_PREFIX

    blockers = status.get("blockers")
    return (status.get("status") == "blocked" and isinstance(blockers, list) and len(blockers) == 1
            and str(blockers[0]).startswith(OWNER_CONTRACT_BLOCKER_PREFIX))


def _unconfirmed(status: Mapping[str, Any]) -> bool:
    """A request whose owner may still decide on its thresholds: its completed half assumed them, or an
    older release's refusal waits on the owner contract."""

    from .website_owner_success_assumption import status_assumed

    return status_assumed(status) or _owner_contract_blocked(status)


def _error_code(exc: BaseException) -> str:
    text = str(exc).split(":", 2)
    code = ":".join(text[:2]) if text[0].startswith("website_owner_success_http_") else text[0]
    return code[:160] if re.fullmatch(r"[a-z][a-z0-9_.:-]{0,159}", code) else type(exc).__name__


def _now() -> float:
    return time.time()


def _instant(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _next_check(previous: Mapping[str, Any] | None) -> float | None:
    """When the request may be asked again, from its kept exchange; None asks now."""

    value = (previous or {}).get("next_check_at")
    try:
        return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _kind(outcome: Mapping[str, Any]) -> str:
    if outcome.get("status") == "failed" and str(outcome.get("error", "")).startswith("website_owner_success_http_4"):
        return "refused"  # a typed WebApp refusal (or its rate limit) will not change in a minute
    return str(outcome.get("status"))


def _scheduled(outcome: dict[str, Any], previous: Mapping[str, Any] | None, progression_digest: str | None,
               now: float) -> dict[str, Any]:
    """The outcome with its check time and the next one: the wait doubles while the same answer repeats."""

    previous = previous or {}
    attempt = previous.get("attempt")
    repeated = (_kind(previous) == _kind(outcome) and previous.get("proposal_digest") == outcome.get("proposal_digest")
                and previous.get("progression_digest") == progression_digest and type(attempt) is int and attempt > 0)
    attempt = attempt + 1 if repeated else 1
    first, longest = BACKOFF.get(_kind(outcome), BACKOFF["failed"])
    wait = min(longest, first * 2 ** min(attempt - 1, 20))
    return {**outcome, "progression_digest": progression_digest, "attempt": attempt,
            "checked_at": _instant(now), "next_check_at": _instant(now + wait)}


def exchange_owner_decision(packet_root: Path, request_key: str, *, transport: Transport | None = None,
                            now: float | None = None) -> dict[str, Any] | None:
    """Ask the WebApp about one request whose thresholds its owner has not confirmed, and record a confirmation.

    None when the request's thresholds are not open to its owner (``_unconfirmed``), or it is not due:
    its kept check time has not come and its intent's progression is unchanged.
    The outcome is what the request's status keeps under ``STATUS_FIELD``. Any
    error is this request's ``failed`` outcome; it never leaves this function.
    """

    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_owner_success_confirmation import binding, find_confirmation, published_job
    from .team_scene_packet_worker import _read_json, owner_success_confirmation_scope, validated_request

    now = _now() if now is None else now
    try:
        status = _read_json(packet_root / "status" / f"{request_key}.json") or {}
    except Exception:  # noqa: BLE001 - an unreadable status is not this exchange's to decide
        return None
    if not _unconfirmed(status) or (packet_root / "pending" / f"{request_key}.json").exists():
        return None
    previous = status.get(STATUS_FIELD) if isinstance(status.get(STATUS_FIELD), dict) else None
    # Cheap reads only, before anything is due: the request, its intent and its progression's digest.
    scope = unbound = progression_digest = None
    try:
        request = validated_request(_read_json(packet_root / "processed" / f"{request_key}.json"))
        scope, unbound = owner_success_confirmation_scope(request)
        if scope is not None:
            progression_digest = intake._read(Path(scope["directory"]) / "progression.json",
                                              "progression_digest")["progression_digest"]
    except Exception as exc:  # noqa: BLE001 - one job's exchange never stops the pass
        prelude_error: BaseException | None = exc
    else:
        prelude_error = None
    next_check = _next_check(previous)
    backoff_expired = next_check is None or now >= next_check
    if not backoff_expired and (previous or {}).get("progression_digest") == progression_digest:
        return None
    if prelude_error is not None:
        return _scheduled({"status": "failed", "error": _error_code(prelude_error)}, previous, progression_digest, now)
    if scope is None:
        return _scheduled({"status": "intent_unbound", "reason": unbound}, previous, progression_digest, now)
    proposal_digest: str | None = None
    try:
        intent, directory = scope["intent"], Path(scope["directory"])
        if (directory / "revoked.json").exists():
            return _scheduled({"status": "intent_revoked"}, previous, progression_digest, now)
        job = published_job(directory, intent)
        bound = binding(intent=intent, revision=job["revision"], criteria=job["criteria"],
                        proposal=job["proposal"] or {})
        notice = proposal_notice(bound)
        proposal_digest = notice["proposal_digest"]
        if not backoff_expired and previous is not None and previous.get("proposal_digest") == notice["proposal_digest"]:
            # The progression moved but the proposal is the same: keep the schedule, send nothing.
            return {**previous, "progression_digest": progression_digest}
        existing = find_confirmation(directory, intent, bound)
        if existing is not None:
            # Already confirmed (door or website): the requeue pass takes it; nothing is sent.
            outcome: dict[str, Any] = {"status": "already_confirmed", "proposal_digest": notice["proposal_digest"],
                                       "confirmation_digest": existing["confirmation_digest"],
                                       "issuer_channel": existing["issuer_channel"]}
            return _scheduled(outcome, previous, progression_digest, now)
        outcome = {"status": "failed", "proposal_digest": notice["proposal_digest"]}
        receipt = validated_receipt((transport or send_proposal)(notice), notice)
        outcome["status"] = receipt["status"]
        decision = receipt["decision"]
        if decision is not None:
            outcome["decision_digest"] = decision["decision_digest"]
        if receipt["status"] == "held":
            outcome["reason"] = decision["reason"]
        elif receipt["status"] == "confirmed":
            written = record_website_confirmation(intent_root=directory.parent, intent=intent, decision=decision)
            outcome.update(record=written["status"], confirmation_digest=written["confirmation_digest"])
        return _scheduled(outcome, previous, progression_digest, now)
    except Exception as exc:  # noqa: BLE001 - one job's exchange never stops the pass (http.client errors too)
        failed = {"status": "failed", "error": _error_code(exc),
                  **({"proposal_digest": proposal_digest} if proposal_digest else {})}
        return _scheduled(failed, previous, progression_digest, now)


def exchange_owner_decisions(packet_root: str | Path, *, transport: Transport | None = None,
                             limit: int = MAX_EXCHANGES_PER_PASS, now: float | None = None
                             ) -> dict[str, dict[str, Any]]:
    """This pass's exchange for up to ``limit`` due requests with unconfirmed thresholds, by request key.

    A request that is not due is skipped and does not count. The starting
    request rotates every minute, so a job whose owner never answers cannot
    starve a newer one. Runs without the packet queue lock: it reads statuses
    only, and a confirmation is written under the intake lock by the door's
    writer.
    """

    from .team_scene_packet_worker import _read_json

    now = _now() if now is None else now
    root = Path(packet_root)
    blocked = []
    for path in sorted((root / "processed").glob("*.json")):
        try:
            if (_unconfirmed(_read_json(root / "status" / path.name) or {})
                    and not (root / "pending" / path.name).exists()):
                blocked.append(path.stem)
        except Exception:  # noqa: BLE001 - one unreadable status never stops the pass
            continue
    if not blocked:
        return {}
    start = int(now // 60) % len(blocked)
    outcomes: dict[str, dict[str, Any]] = {}
    for key in blocked[start:] + blocked[:start]:
        if len(outcomes) >= max(0, limit):
            break
        outcome = exchange_owner_decision(root, key, transport=transport, now=now)
        if outcome is not None:
            outcomes[key] = outcome
    return outcomes


__all__ = [
    "BACKOFF",
    "CONTACT_CLASS_SCOPE",
    "DECISION_SCHEMA",
    "HOLD_REASONS",
    "NOTICE_SCHEMA",
    "RECEIPT_SCHEMA",
    "STATUS_FIELD",
    "exchange_owner_decision",
    "exchange_owner_decisions",
    "forbidden_contacts",
    "proposal_notice",
    "record_website_confirmation",
    "send_proposal",
    "validated_receipt",
]
