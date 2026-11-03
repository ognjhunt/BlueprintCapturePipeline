"""The site owner's decision on a website job's success thresholds, made on the job page.

A website job's thresholds are Blueprint's fixed development controls, marked
``proposal_only``, and every owner-scored step (the team packet, screening)
refuses them until the owner confirms them
(``task_evaluation_owner_success_confirmation``). The operator door records
that confirmation by hand. This records the same confirmation from the
website, with no operator.

On the team packet queue's timer pass (``team_scene_packet_worker.
requeue_transient_blocked``, run by the screening timer), each request blocked
only on that refusal sends its intent's published thresholds to the WebApp
over the existing signed Pipeline sync transport
(``POST /api/internal/pipeline/task-evaluation-scene-intents/<intent>/owner-success-proposal``).
The WebApp keeps the proposal, shows it to the job's owner (the owner link
only) in plain words with Confirm and "this is wrong", and answers with the
owner's decision on exactly this proposal, if there is one:

- ``confirmed``: the decision must name this proposal's digest, revision and
  binding. The record is written by ``confirm_owner_success_contract``, the
  door's own writer, under the same validators, with ``issuer_channel``
  ``website``. The same pass then returns the request to pending, as it does
  for a door confirmation, and the team packet compiles the owner contract as
  the owner's.
- ``held``: the owner said the thresholds are wrong, with a typed reason.
  Nothing is written under the intent, no number changes, and the request
  stays blocked; its status keeps the hold and the reason.
- ``pending``: no decision yet.

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
              "binding_digest": canonical_digest(dict(bound))}
    notice["proposal_digest"] = cross_runtime_canonical_digest(notice)
    return notice


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


def _error_code(exc: BaseException) -> str:
    text = str(exc).split(":", 2)
    code = ":".join(text[:2]) if text[0].startswith("website_owner_success_http_") else text[0]
    return code[:160] if re.fullmatch(r"[a-z][a-z0-9_.:-]{0,159}", code) else type(exc).__name__


def exchange_owner_decision(packet_root: Path, request_key: str, *,
                            transport: Transport | None = None) -> dict[str, Any] | None:
    """Ask the WebApp about one owner-contract-blocked request, and record a confirmation; None if not blocked.

    The outcome is what the request's status keeps under ``STATUS_FIELD``.
    """

    from .task_evaluation_owner_success_confirmation import binding, find_confirmation, published_job
    from .team_scene_packet_worker import _read_json, owner_success_confirmation_scope, validated_request

    status = _read_json(packet_root / "status" / f"{request_key}.json") or {}
    if not _owner_contract_blocked(status) or (packet_root / "pending" / f"{request_key}.json").exists():
        return None
    try:
        request = validated_request(_read_json(packet_root / "processed" / f"{request_key}.json"))
        scope, unbound = owner_success_confirmation_scope(request)
        if scope is None:
            return {"status": "intent_unbound", "reason": unbound}
        intent, directory = scope["intent"], Path(scope["directory"])
        if (directory / "revoked.json").exists():
            return {"status": "intent_revoked"}
        job = published_job(directory, intent)
        bound = binding(intent=intent, revision=job["revision"], criteria=job["criteria"],
                        proposal=job["proposal"] or {})
        notice = proposal_notice(bound)
        existing = find_confirmation(directory, intent, bound)
        if existing is not None:
            # Already confirmed (door or website): the requeue pass takes it; nothing is sent.
            return {"status": "already_confirmed", "proposal_digest": notice["proposal_digest"],
                    "confirmation_digest": existing["confirmation_digest"],
                    "issuer_channel": existing["issuer_channel"]}
        receipt = validated_receipt((transport or send_proposal)(notice), notice)
        outcome: dict[str, Any] = {"status": receipt["status"], "proposal_digest": notice["proposal_digest"]}
        decision = receipt["decision"]
        if decision is not None:
            outcome["decision_digest"] = decision["decision_digest"]
        if receipt["status"] == "held":
            outcome["reason"] = decision["reason"]
        elif receipt["status"] == "confirmed":
            written = record_website_confirmation(intent_root=directory.parent, intent=intent, decision=decision)
            outcome.update(record=written["status"], confirmation_digest=written["confirmation_digest"])
        return outcome
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"status": "failed", "error": _error_code(exc)}


def exchange_owner_decisions(packet_root: str | Path, *, transport: Transport | None = None,
                             limit: int = MAX_EXCHANGES_PER_PASS, now: float | None = None
                             ) -> dict[str, dict[str, Any]]:
    """This pass's exchange for up to ``limit`` owner-contract-blocked requests, by request key.

    The starting request rotates every minute, so a job whose owner never
    answers cannot starve a newer one. Runs without the packet queue lock: it
    reads statuses only, and a confirmation is written under the intake lock
    by the door's writer.
    """

    from .team_scene_packet_worker import _read_json

    root = Path(packet_root)
    blocked = [path.stem for path in sorted((root / "processed").glob("*.json"))
               if _owner_contract_blocked(_read_json(root / "status" / path.name) or {})
               and not (root / "pending" / path.name).exists()]
    if not blocked:
        return {}
    start = int((time.time() if now is None else now) // 60) % len(blocked)
    outcomes: dict[str, dict[str, Any]] = {}
    for key in (blocked[start:] + blocked[:start])[:max(0, limit)]:
        outcome = exchange_owner_decision(root, key, transport=transport)
        if outcome is not None:
            outcomes[key] = outcome
    return outcomes


__all__ = [
    "DECISION_SCHEMA",
    "HOLD_REASONS",
    "NOTICE_SCHEMA",
    "RECEIPT_SCHEMA",
    "STATUS_FIELD",
    "exchange_owner_decision",
    "exchange_owner_decisions",
    "proposal_notice",
    "record_website_confirmation",
    "send_proposal",
    "validated_receipt",
]
