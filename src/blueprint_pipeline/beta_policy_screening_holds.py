"""Tell a website job, by name, that its evaluation is held before any campaign could start.

Beta journey: a prepared site's automatic evaluation, after its scene is
published. Only a campaign publication reached the website, so a job whose
evaluation ended before the screening queue could plan a campaign kept reading
"in progress" for good. These ends are named here, each with a typed reason:

``robot_free_probe_failed``
    The scene's robot-free probe decided that it cannot tell success from
    failure (a decided receipt is never replaced), whether scene preparation
    retained it (the team half is then never compiled) or the rigid probe lane
    recorded it.
``robot_free_probe_not_completed``
    The probe could not be completed and nothing retries it under the running
    release: a staging refusal this release already tried
    (``rigid_robot_free_probe_runner``, retried once per new release), a spent
    attempt budget, a trace conflict with the request's own trace (one beside a
    superseded request's trace is re-ingested once per new release), an unbound
    revision, or a decided receipt the gate refuses. A receipt bound to another
    scene revision or task is a superseded request's, never the current one's
    (``current_probe_state``), so the lane probes the current request.
``robot_free_probe_unavailable_for_task``
    The task has no robot-free probe, so no robot can be scored on it.
``team_scene_packet_blocked``
    The scene's team half could not be compiled, and nothing returns it to the
    queue under the running release (``team_scene_packet_worker.final_block``).

The screening worker's source pass decides one hold per scene intent
(``probe_hold``, ``packet_hold``) and keeps the latest under
``<queue_root>/holds/``. ``deliver_holds`` posts it over the same signed
Pipeline transport as campaign publications, with a revision and a self-digest,
until the website acknowledges those exact bytes. A hold is never final here:
when a later pass finds the evaluation moving again (a new release retried the
probe or the team half, the probe passed, a campaign was planned), the same hold
is published again as ``released``, and the website reads the campaign. Privacy
and rights refusals are never published as holds; the website ends those itself.
Nothing here spends, launches or writes outside the hold store.

Three more ends used to say nothing for good, and are held the same way:

- a source whose planning keeps failing with the same refusal for
  ``STALL_HOLD_AFTER_SECONDS`` (``stalled_source_hold``: the scene profile is
  unavailable, the job's dollar ceiling cannot be read). Planning is still tried
  on every pass, and the hold is released the pass it succeeds;
- a planned campaign with a row whose provider outcome stays unproven
  (``reconciliation_required``) past its bound;
- a planned campaign that ended with nothing scored on a reached account
  monthly limit, a money stop that is never retried here.

The last two come from the coordinator (``campaign_hold_detail``) as a detail
code (``campaign_hold``). The WebApp accepts a closed set of reasons
(``REASONS``, its ``betaPolicyScreeningHold.ts``) and none names these ends, so
they are sent under ``STALL_HOLD_REASON``, the accepted reason the site is told
as a problem on our side, with the real cause in ``detail_code``. Publishing a
hold changes no ledger row: an unproven reservation stays open and is still
reconciled, and nothing is released or settled.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

HOLD_SCHEMA = "beta_policy_screening_hold.v1"
HOLDS_DIRNAME = "holds"
ROUTE = "/api/internal/pipeline/beta-policy-screening-holds"
REASONS = frozenset({"robot_free_probe_failed", "robot_free_probe_not_completed",
                     "robot_free_probe_unavailable_for_task", "team_scene_packet_blocked"})
PROBE_GATE_PREFIX = "team_evaluation_robot_free_probe_"
#: A positive sign the evaluation moves again: a campaign was planned, the team half went back to its queue,
#: or a probe refusal is no longer final. Only this releases a hold; an error or an unreadable state never does.
RELEASE = "release"
_CODE = re.compile(r"[a-z][a-z0-9_:.=-]{0,199}")
# The website ends a withdrawn or revoked job itself; a hold must never stand in for that.
_RIGHTS = re.compile(r"withdrawn|revoked|consent|rights")
_SOURCE_FIELDS = ("scene_intent_id", "scene_intent_digest", "capture_id", "task_id", "task_context_digest",
                  "scene_revision_digest")
#: How long a source's planning may keep failing the same way before its job is told it is held.
STALL_HOLD_AFTER_SECONDS = 6 * 3600
#: The accepted reason a stalled source, a stalled campaign or a money stop is sent under: the website
#: words it as a problem on our side (``held_setup``). ``detail_code`` carries the real cause.
STALL_HOLD_REASON = "team_scene_packet_blocked"
#: A re-plan refused because the job's window closed: the job's own end, never a hold.
WINDOW_CLOSED = "beta_screening_job_window_closed"


def hold_id(scene_intent_id: str) -> str:
    """One hold per scene intent, stable across revisions and passes."""
    return "hold-" + hashlib.sha256(scene_intent_id.encode("utf-8")).hexdigest()[:32]


def _code(value: Any) -> str | None:
    text = str(value or "")
    return text if _CODE.fullmatch(text) else None


def packet_hold(status: Mapping[str, Any], *, release: str | None) -> tuple[str, str | None] | None:
    """A team-half block nothing returns to the queue under ``release``, as a hold; else None."""
    from .team_scene_packet_worker import final_block

    blockers = [str(code) for code in status.get("blockers") or []]
    if not final_block(status, release=release) or any(_RIGHTS.search(code) for code in blockers):
        return None
    return "team_scene_packet_blocked", _code(blockers[0] if blockers else None)


def probe_hold(code: str, capture_root: str | Path, *, queue_root: str | Path | None = None,
               release: str | None = None) -> tuple[str, str | None] | None:
    """The hold a robot-free probe gate refusal stands for once nothing will decide the probe; else None.

    ``code`` is the planning refusal (``robot_free_probe_gate``). A refusal the
    probe lane can still clear (a run queued, waiting or due a retry by a new
    release) is no hold: the job is still being evaluated.
    """
    if not code.startswith(PROBE_GATE_PREFIX):
        return None
    reason = code[len(PROBE_GATE_PREFIX):]
    if reason.startswith("unavailable_for_task"):
        return "robot_free_probe_unavailable_for_task", _code(reason)
    from .rigid_robot_free_probe_bundle import current_probe_state
    from .rigid_robot_free_probe_runner import final_terminal

    state = current_probe_state(capture_root)
    status = state.get("status")
    if status == "receipt_decided":
        # A decided receipt of the current request is never replaced: a failed one is final.
        failed = state.get("receipt_status") == "failed"
        return ("robot_free_probe_failed" if failed else "robot_free_probe_not_completed"), _code(reason)
    if status == "unavailable_for_task":
        # A rigid task no probe plan can carry is a capability limit; any other kind kept no receipt to read.
        rigid = state.get("task_kind") == "rigid_pick_place"
        return ("robot_free_probe_unavailable_for_task" if rigid else "robot_free_probe_not_completed"), _code(reason)
    if status == "scene_revision_unbound":
        return "robot_free_probe_not_completed", "scene_revision_unbound"
    if status == "probe_required":
        terminal = final_terminal(state, queue_root=queue_root, release=release)
        if terminal is not None:
            return "robot_free_probe_not_completed", _code(terminal.get("status"))
    return None


def campaign_hold(detail: str) -> tuple[str, str | None]:
    """The hold a planned campaign's own end stands for (``coordinator.campaign_hold_detail``)."""
    return STALL_HOLD_REASON, _code(detail)


def blocked_since(previous: Mapping[str, Any] | None, blockers: list[str], *, now: datetime | None = None) -> str:
    """Since when a source has been refused with exactly ``blockers``, carried from its previous outcome."""
    kept = (previous or {}).get("blocked_since") if (previous or {}).get("blockers") == blockers else None
    try:
        valid = isinstance(kept, str) and datetime.fromisoformat(kept).tzinfo is not None
    except ValueError:
        valid = False
    return kept if valid else (now or datetime.now(timezone.utc)).isoformat()


def stalled_source_hold(code: str, since: str, *, now: datetime | None = None) -> tuple[str, str | None] | None:
    """The hold a source refusal stands for once it has failed the same way long enough; else None.

    Never for a privacy or rights refusal (the website ends those itself) or a
    closed job window. An unreadable start reads as not yet.
    """
    if _RIGHTS.search(code) or code == WINDOW_CLOSED:
        return None
    try:
        elapsed = ((now or datetime.now(timezone.utc)) - datetime.fromisoformat(since)).total_seconds()
    except (TypeError, ValueError):
        return None
    return (STALL_HOLD_REASON, _code(code)) if elapsed >= STALL_HOLD_AFTER_SECONDS else None


def scene_preparation_probe_failed(request: Mapping[str, Any]) -> bool:
    """Scene preparation retained a decided failed receipt beside the request's launch (re-read, not trusted)."""
    from .robot_free_probe_gate import retained_probe_receipt

    receipt = retained_probe_receipt(Path(str(request["source_launch_root"])))
    return (isinstance(receipt, Mapping) and receipt.get("status") == "failed"
            and receipt.get("receipt_digest") == canonical_digest(dict(receipt), digest_field="receipt_digest"))


def _source(request: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    from .team_scene_packet_worker import _read_json, resolve_capture_root

    capture = resolve_capture_root(request)
    retained = _read_json(capture / "pipeline" / "website_task_context.json") or {}
    site_id = str(retained.get("scene_id") or "")
    if not re.fullmatch(r"site-[A-Za-z0-9][A-Za-z0-9._:-]{0,191}", site_id) or not retained.get("context_digest"):
        raise ValueError("beta_screening_hold_source_unavailable")
    return {"scene_intent_id": request["intent_id"], "scene_intent_digest": request["intent_digest"],
            "capture_id": request["capture_id"], "task_id": request["task_id"],
            "task_context_digest": retained["context_digest"],
            "scene_revision_digest": request["scene_revision_digest"]}, site_id


def _store(queue_root: str | Path) -> Path:
    from .beta_policy_screening_coordinator import _root

    path = _root(queue_root) / HOLDS_DIRNAME
    if path.is_symlink():
        raise ValueError("beta_screening_hold_store_unsafe")
    path.mkdir(exist_ok=True, mode=0o700)
    return path


def _publication(*, previous: Mapping[str, Any] | None, source: Mapping[str, Any], site_id: str,
                 decision: tuple[str, str | None] | None) -> dict[str, Any]:
    value = {"schema_version": HOLD_SCHEMA, "hold_id": hold_id(str(source["scene_intent_id"])), "site_id": site_id,
             "source": {field: source[field] for field in _SOURCE_FIELDS},
             "status": "held" if decision else "released",
             "reason_code": decision[0] if decision else None, "detail_code": decision[1] if decision else None,
             "revision": int((previous or {}).get("revision", 0)) + 1,
             "generated_at": datetime.now(timezone.utc).isoformat(), "hold_digest": ""}
    value["hold_digest"] = cross_runtime_canonical_digest(value, digest_field="hold_digest")
    return value


def settle_hold(queue_root: str | Path, request: Mapping[str, Any],
                decision: tuple[str, str | None] | str | None) -> dict[str, Any] | None:
    """Record this pass's decision for the request's scene intent; never raises, never spends.

    ``decision`` is a ``(reason, detail)`` hold, ``RELEASE`` (positive progress)
    or None: nothing was decided (an error, an unreadable state), so a hold is
    kept as it is, unless the request is a newer scene revision of the intent
    than the one held, which supersedes it. A new or changed hold, or a
    release, is a new revision; the same decision again changes nothing.
    Nothing to record returns None. ``request`` must be the intent's newest.
    """
    from .beta_policy_screening_coordinator import _lock, _read, _write

    try:
        if isinstance(decision, tuple) and decision[0] not in REASONS:
            raise ValueError("beta_screening_hold_reason_invalid")
        store = _store(queue_root)
        path = store / f"{hold_id(str(request['intent_id']))}.json"
        with _lock(store.parent, ".holds.lock"):
            record = _read(path) if path.exists() else None
            previous = (record or {}).get("publication")
            held = previous is not None and previous.get("status") == "held"
            if decision is None:
                superseded = held and previous["source"].get("scene_revision_digest") != request["scene_revision_digest"]
                if not superseded:
                    return None
                decision = RELEASE
            if decision == RELEASE and not held:
                return None
            decision = None if decision == RELEASE else decision
            source, site_id = _source(request)
            same = previous is not None and (
                previous.get("status"), previous.get("reason_code"), previous.get("detail_code"),
                previous.get("source"), previous.get("site_id")) == (
                "held" if decision else "released", decision[0] if decision else None,
                decision[1] if decision else None, source, site_id)
            if same:
                return {"status": previous["status"], "reason_code": previous["reason_code"],
                        "revision": previous["revision"], "change": "unchanged",
                        "acknowledged": record.get("acknowledged_digest") == previous["hold_digest"]}
            publication = _publication(previous=previous, source=source, site_id=site_id, decision=decision)
            _write(path, {"publication": publication, "acknowledged_digest": (record or {}).get("acknowledged_digest")})
            return {"status": publication["status"], "reason_code": publication["reason_code"],
                    "revision": publication["revision"], "change": "recorded"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        from .bounded_error import error_description

        return {"status": "unrecorded", "error": error_description(exc)}


def pending_holds(*, queue_root: str | Path) -> list[dict[str, Any]]:
    """Every hold whose latest bytes the website has not acknowledged."""
    from .beta_policy_screening_coordinator import _lock, _read

    store = _store(queue_root)
    with _lock(store.parent, ".holds.lock"):
        rows = [_read(path) for path in sorted(store.glob("hold-*.json")) if not path.is_symlink()]
    return [row["publication"] for row in rows
            if row["publication"]["hold_digest"] != row.get("acknowledged_digest")]


def acknowledge_hold(publication: Mapping[str, Any], *, queue_root: str | Path) -> bool:
    """Acknowledge only the current bytes; a late answer never swallows a newer decision."""
    from .beta_policy_screening_coordinator import _lock, _read, _write

    store = _store(queue_root)
    path = store / f"{publication['hold_id']}.json"
    with _lock(store.parent, ".holds.lock"):
        record = _read(path)
        if record["publication"]["hold_digest"] != publication["hold_digest"]:
            return False
        _write(path, {**record, "acknowledged_digest": publication["hold_digest"]})
    return True


def publish_hold(publication: Mapping[str, Any]) -> dict[str, Any]:
    """Send one hold over the existing signed Pipeline transport; require the exact acknowledgment."""
    from .safe_outbound_http import pinned_api_policy, request
    from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
    from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url

    configured = os.getenv("PIPELINE_SYNC_WEBAPP_URL", "").strip()
    if not configured:
        raise ValueError("beta_screening_webapp_url_missing")
    parsed = urlsplit(validated_https_sync_url(configured))
    origin = f"{parsed.scheme}://{parsed.netloc}"
    body = json.dumps(dict(publication), separators=(",", ":"), allow_nan=False).encode()
    try:
        response = request(f"{origin}{ROUTE}", method="POST", data=body,
            headers=_pipeline_sync_headers(load_pipeline_sync_token(), body), timeout_seconds=15,
            policy=pinned_api_policy(origin, max_response_bytes=100_000), max_response_bytes=100_000)
    except HTTPError as exc:
        code = exc.code
        exc.close()
        raise ValueError(f"beta_screening_hold_http_{code}") from None
    value = json.loads(response.body)
    if not isinstance(value, dict) or value.get("hold_id") != publication["hold_id"]:
        raise ValueError("beta_screening_hold_ack_invalid")
    exact = (value.get("status") in {"created", "updated", "replayed"} and value.get("revision") == publication["revision"]
             and value.get("hold_digest") == publication["hold_digest"])
    # The website already keeps this revision or a newer one of ours (it never lets an older one replace it).
    newer = (value.get("status") == "replayed" and type(value.get("revision")) is int
             and value["revision"] >= publication["revision"])
    if not (exact or newer):
        raise ValueError("beta_screening_hold_ack_invalid")
    return value


def deliver_holds(*, queue_root: str | Path, transport=None) -> dict[str, Any]:
    """Post every unacknowledged hold; a failure stays pending for the next pass."""
    from .bounded_error import error_description

    sent = failed = 0
    failure: dict[str, Any] = {}
    try:
        pending = pending_holds(queue_root=queue_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"acknowledged": 0, "pending": 0, "last_error": error_description(exc)}
    for publication in pending:
        try:
            (transport or publish_hold)(publication)
            sent += int(acknowledge_hold(publication, queue_root=queue_root))
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            failed += 1
            failure = {"last_error": error_description(exc)}
    return {"acknowledged": sent, "pending": failed, **failure}


__all__ = ["HOLD_SCHEMA", "REASONS", "RELEASE", "ROUTE", "STALL_HOLD_AFTER_SECONDS", "STALL_HOLD_REASON",
           "WINDOW_CLOSED", "acknowledge_hold", "blocked_since", "campaign_hold", "deliver_holds", "hold_id",
           "packet_hold", "pending_holds", "probe_hold", "publish_hold", "scene_preparation_probe_failed",
           "settle_hold", "stalled_source_hold"]
