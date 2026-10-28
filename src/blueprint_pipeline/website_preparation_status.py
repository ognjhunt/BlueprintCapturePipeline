"""ADP-010/day14: current source-bound preparation status and durable wake-ups.

A wake-up is never failure authority. The WebApp re-reads the signed status
route, which verifies the current source/context and the latest ledger revision.
No provider is dispatched, and handed_off is not native or assessment completion.

Beta journey L3/L6: v2 is additive. A WebApp that asks for it (request header
below) also learns that preparation ended for good (failed_terminal, with a
reason) and that an attempt only waits for control-plane capacity
(capacity_delayed) rather than for anything the site provides. Every other
caller, including a WebApp that predates v2, keeps the exact v1 receipt.

A preparation-only task stays unconfirmed, so its read also asks the WebApp
where the grant stands. A grant that expired still names the context it was
issued for, so the ledger is reported as for a current one. A preparation with
no grant yet is not read. The read never creates, renews or spends a grant.

The default-off v4 adds optional source-bound family authoring to this same
receipt/outbox. A missing optional proof keeps its delivery pending with a
fixed private diagnostic; the original preparation state/outcome is preserved.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

SELECTORS = frozenset({"request_id", "scene_id", "capture_id", "completion_marker_generation",
                      "producer_delivery_key", "source_payload_sha256", "task_context_digest"})
DELIVERY_FILE = "website_preparation_status_delivery.json"
_UNAVAILABLE = "website_preparation_status_unavailable"
STATUS_V1 = "website_preparation_status.v1"
STATUS_V2 = "website_preparation_status.v2"
STATUS_V3 = "website_preparation_status.v3"
STATUS_V4 = "website_preparation_status.v4"
FAMILY_STATUS_ENV = "BLUEPRINT_WEBSITE_FAMILY_TASK_FACTS_V4_ENABLED"
# The WebApp opts in per read; v4 additionally requires FAMILY_STATUS_ENV.
VERSION_HEADER = "x-blueprint-preparation-status-version"
_STATES_V1 = frozenset({"preparing", "awaiting_inputs", "failed_retryable", "authority_ended", "handed_off"})
_STATES_V2 = _STATES_V1 | {"capacity_delayed", "failed_terminal"}
_ACCEPTANCE = {"website_preparation_status_acceptance.v1": (STATUS_V1, _STATES_V1),
               "website_preparation_status_acceptance.v2": (STATUS_V2, _STATES_V2),
               "website_preparation_status_acceptance.v3": (STATUS_V3, _STATES_V2),
               "website_preparation_status_acceptance.v4": (STATUS_V4, _STATES_V2)}
_CODES = {"preparing": "preparation_in_progress", "awaiting_inputs": "preparation_inputs_pending",
          "capacity_delayed": "preparation_capacity_delayed", "failed_retryable": "preparation_retryable_failure",
          "failed_terminal": "preparation_failed_terminal", "authority_ended": "preparation_authority_ended",
          "handed_off": "preparation_handed_off"}
_LEDGER_STATES = {"processing": "preparing", "failed_retryable": "failed_retryable",
                  "retryable_blocked": "awaiting_inputs", "terminal_authority_ended": "authority_ended"}
# Supplement ids are WebApp-minted digests (captureSupplement.ts); the
# supplement's own birth and owner records still bind it to this request.
_SUPPLEMENT_CAPTURE = re.compile(r"supplement-[0-9a-f]{40}")


def requested_status_version(value: Any) -> str:
    """The newest receipt schema this Pipeline has at or below what the reader asked for."""
    text = value.strip() if type(value) is str else ""
    if re.fullmatch(r"[1-9][0-9]{0,2}", text):
        if int(text) >= 4 and os.getenv(FAMILY_STATUS_ENV) == "true":
            return STATUS_V4
        return STATUS_V3 if int(text) >= 3 else STATUS_V2 if int(text) == 2 else STATUS_V1
    return STATUS_V1


class PreparationStatusUnavailable(ValueError):
    """Fixed public refusal; internal paths/errors never cross this boundary."""


def _require(condition: bool) -> None:
    if not condition:
        raise PreparationStatusUnavailable(_UNAVAILABLE)


def validate_selectors(value: Any) -> dict[str, str]:
    _require(type(value) is dict and set(value) == SELECTORS)
    _require(all(type(item) is str for item in value.values()))
    request = value["request_id"]
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", request) is not None)
    _require(value["scene_id"] == f"site-{request}" and (value["capture_id"] == f"walkthrough-{request}"
             or _SUPPLEMENT_CAPTURE.fullmatch(value["capture_id"]) is not None))
    _require(re.fullmatch(r"[1-9][0-9]{0,19}", value["completion_marker_generation"]) is not None)
    _require(all(re.fullmatch(r"sha256:[0-9a-f]{64}", value[key]) is not None
                 for key in ("producer_delivery_key", "task_context_digest")))
    _require(re.fullmatch(r"[0-9a-f]{64}", value["source_payload_sha256"]) is not None)
    return dict(value)


def _read(path: Path) -> dict[str, Any]:
    _require(not path.is_symlink())
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(type(value) is dict)
    return value


def _read_delivery(path: Path) -> dict[str, Any]:
    """Only a valid scheduling record may receive canonical retry bookkeeping."""
    value = _read(path)
    _require(value.get("schema_version") == "website_preparation_status_delivery.v1"
             and value.get("state") in {"pending", "delivered"}
             and type(value.get("revision")) is int and value["revision"] > 0)
    validate_selectors(value.get("selectors"))
    for field in ("delivery_attempt_count", "delivery_failure_count"):
        counter = value.get(field, 0)
        _require(type(counter) is int and counter >= 0)
    for field in ("pending_since_epoch", "next_attempt_at_epoch"):
        timestamp = value.get(field, 0.0)
        _require(type(timestamp) in (int, float) and math.isfinite(timestamp) and timestamp >= 0)
    return value


def _root(root: Path) -> Path:
    root = Path(root).absolute()
    _require(not any(item.is_symlink() for item in (root, *root.parents)))
    _require(root.is_dir())
    return root


def _retained_context(root: Path) -> dict[str, Any]:
    # This is written by the actual website task descriptor producer, before
    # sponsorship/provider work. Never replace it with a newly fetched context.
    return _read(root / "pipeline" / "website_task_context.json")


def _retained_context_or_none(root: Path) -> dict[str, Any] | None:
    """None only when no attempt has retained a context yet (it failed earlier).

    Such an attempt prepared nothing under any context, so its status is bound
    to the site's current context instead; nothing retained is replaced.
    """
    path = root / "pipeline" / "website_task_context.json"
    if path.exists() or path.is_symlink():
        return _retained_context(root)
    return None


def _current_context(*, request_id: str, scene_id: str, capture_id: str,
                     purpose: str | None, allow_unconfirmed_preparation: bool = False) -> dict[str, Any]:
    from .website_task_context import load_current_website_task_context
    arguments = {"request_id": request_id, "scene_id": scene_id, "capture_id": capture_id}
    try:
        return load_current_website_task_context(**arguments, **({"purpose": purpose} if purpose else {}))
    except ValueError as exc:
        # Before a task snapshot exists, an ordinary capture may already have
        # an unconfirmed preparation proposal. Read that explicit scope only;
        # transport, identity and changed-context refusals remain refusals.
        if (not allow_unconfirmed_preparation or purpose is not None
                or str(exc) != "website_task_context_not_confirmed"):
            raise
        return load_current_website_task_context(**arguments, purpose="scene_preparation")


def _v2_state(ledger: Mapping[str, Any]) -> tuple[str, str | None]:
    """Typed v2 meaning of a ledger that is not handed off."""
    from .website_preparation_terminal import capacity_only, current_terminal, current_overdue
    status = ledger.get("status")
    _require(status in _LEDGER_STATES)
    terminal = current_terminal(ledger) if status != "terminal_authority_ended" else None
    if terminal is not None:
        return "failed_terminal", terminal["reason"]
    if status in {"failed_retryable", "retryable_blocked"} and capacity_only(ledger):
        return "capacity_delayed", None
    if status == "processing" and current_overdue(ledger) is not None:
        return "failed_retryable", None
    return _LEDGER_STATES[status], None


def read_preparation_status(*, capture_root: Path, selectors: Mapping[str, Any],
                            schema_version: str = STATUS_V1,
                            grantless_assessment_wait: bool = False,
                            _projection_diagnostics: list[str] | None = None) -> dict[str, Any]:
    """Read an as-of snapshot; refuse missing, changed or unverified authority."""
    from .capture_original_owner_observer import load_original_owner_observation
    from .consent_takedown import read_consent_state
    from .handoff_job_state import _read_job_ledger, _output_commit
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    from .website_task_context import validate_website_task_context

    try:
        _require(schema_version in {STATUS_V1, STATUS_V2, STATUS_V3, STATUS_V4}
                 and (schema_version != STATUS_V4 or os.getenv(FAMILY_STATUS_ENV) == "true"))
        selected = validate_selectors(dict(selectors))
        root = _root(capture_root)
        _require(root.name == selected["capture_id"] and root.parent.name == "captures"
                 and root.parent.parent.name == selected["scene_id"])
        _require(not (root / "pipeline_job_ledger.json").is_symlink())
        before = _read_job_ledger(root)
        _require(before.get("schema_version") == "pipeline_job_ledger.v1")
        _require(type(before.get("revision")) is int and before["revision"] > 0
                 and type(before.get("attempt_count")) is int and before["attempt_count"] > 0)
        _require(all(before.get(key) == selected[key] for key in
                     ("scene_id", "capture_id", "producer_delivery_key", "source_payload_sha256")))
        birth = capture_birth_source_projection(root, expected_purpose="scene_preparation")
        _require(type(birth) is dict)
        _require(all(birth.get(key) == selected[key] for key in ("request_id", "scene_id", "capture_id")))
        _require(birth.get("delivery_key") == selected["producer_delivery_key"])
        delivery = _read(Path(birth["birth_delivery_raw_ref"]["path"]))
        purpose = birth.get("capture_observation_purpose")
        if purpose not in {None, "scene_preparation"}:
            raise ValueError("website_preparation_purpose_invalid")
        _require(delivery["source_finalize"]["generation"] == selected["completion_marker_generation"])
        owner = load_original_owner_observation(bucket=delivery["source_finalize"]["bucket"], scene_id=selected["scene_id"],
            capture_id=selected["capture_id"], marker_generation=selected["completion_marker_generation"],
            **({"expected_purpose": purpose} if purpose else {}))
        _require((owner["capture_owner"] or {}).get("user_id") == birth["capture_owner_user_id"])
        _require(owner["producer_delivery"] == delivery["producer_delivery"])
        _require(owner["producer_delivery"]["delivery_key"] == selected["producer_delivery_key"])
        _require(owner["producer_delivery"]["raw_video"]["generation"] == birth["raw_video"]["generation"])
        _require(owner["producer_delivery"]["raw_video"]["object_name"] == birth["raw_video"]["object_name"])
        # Current fresh owner transport validates rights and complete generation
        # authority; historical birth alone grants no current consent.
        _require(read_consent_state(root)["state"] != "revoked")
        raw_retained = _retained_context_or_none(root)
        # Birth scope authenticates the original owner. Task preparation has
        # its own scope and may remain unconfirmed under that same capture.
        context_purpose = raw_retained.get("purpose", purpose) if raw_retained is not None else purpose
        retained = None if raw_retained is None else validate_website_task_context(raw_retained,
                         request_id=selected["request_id"], scene_id=selected["scene_id"],
                         capture_id=selected["capture_id"], purpose=context_purpose)
        current = _current_context(request_id=selected["request_id"],
                        scene_id=selected["scene_id"], capture_id=selected["capture_id"],
                        purpose=context_purpose, allow_unconfirmed_preparation=raw_retained is None)
        _require(current["context_digest"] == selected["task_context_digest"]
                 and (retained is None or retained["context_digest"] == current["context_digest"]))
        # A completed job always retained its context; never hand off without it.
        _require(retained is not None or before.get("status") != "completed")
        if current.get("confirmed") is False:
            from .website_assessment_resume import PENDING_CODE
            from .website_task_context import website_scene_sponsorship_standing
            # A status grants nothing, so a grant that expired still binds the
            # context it was issued for. It never changes the reported state.
            # A capture that waited on its assessment never had a grant: only
            # its resume may read it, because that resume requests the first one.
            waits_on_assessment = (grantless_assessment_wait and before.get("status") == "failed_retryable"
                                   and before.get("last_error") == PENDING_CODE
                                   and isinstance(before.get("assessment_resume"), dict))
            website_scene_sponsorship_standing(task_context=current, now=time.time(),
                                               allow_missing=waits_on_assessment)
        terminal_reason = None
        handoff, family_prepared = None, False
        if before.get("status") == "completed":
            from .website_task_context_contracts import holds_native_task_preparation, retains_task_family_preparation
            handoff = _read(root / "pipeline" / "website_scene_preparation" / "handoff.json")
            native_profile_held = holds_native_task_preparation(handoff, current)
            family_prepared = retains_task_family_preparation(handoff, current)
            _require(handoff.get("schema_version") == "website_scene_handoff.v1"
                     and handoff.get("scene_id") == selected["scene_id"]
                     and handoff.get("capture_id") == selected["capture_id"]
                     and handoff.get("digest") == canonical_digest(handoff, digest_field="digest")
                     and (native_profile_held or family_prepared or
                          handoff.get("website_intake_outbox", {}).get("state") == "forward_pending"))
            _require(bool(_output_commit(root, scene_id=selected["scene_id"], capture_id=selected["capture_id"])))
            state = ("awaiting_inputs" if native_profile_held or (family_prepared and handoff["status"] == "needs_input")
                     else "handed_off")
        elif schema_version in {STATUS_V2, STATUS_V3, STATUS_V4}:
            state, terminal_reason = _v2_state(before)
        else:
            _require(before.get("status") in _LEDGER_STATES)
            state = _LEDGER_STATES[before["status"]]
        outcome = None
        if schema_version in {STATUS_V3, STATUS_V4}:
            from .website_preparation_outcome import read_preparation_outcome
            outcome = read_preparation_outcome(capture_root=root,
                task_context_digest=selected["task_context_digest"], state=state)
        family_authoring = None
        if schema_version == STATUS_V4:
            from .website_preparation_family_status import read_family_authoring_projection
            family_authoring, diagnostic = read_family_authoring_projection(capture_root=root,
                task_context=current, birth=birth, ledger=before, handoff=handoff, family_prepared=family_prepared)
            if diagnostic is not None and _projection_diagnostics is not None:
                _projection_diagnostics.append(diagnostic)
            current_after = _current_context(request_id=selected["request_id"], scene_id=selected["scene_id"],
                capture_id=selected["capture_id"], purpose=context_purpose,
                allow_unconfirmed_preparation=raw_retained is None)
            owner_after = load_original_owner_observation(bucket=delivery["source_finalize"]["bucket"],
                scene_id=selected["scene_id"], capture_id=selected["capture_id"],
                marker_generation=selected["completion_marker_generation"],
                **({"expected_purpose": purpose} if purpose else {}))
            _require(current_after == current and capture_birth_source_projection(root,
                expected_purpose="scene_preparation") == birth)
            _require(all(owner_after.get(key) == owner.get(key) for key in
                ("capture_owner", "capture_rights", "producer_delivery", "completion_marker", "purpose")))
        after = _read_job_ledger(root)
        _require(before == after)
        _require(read_consent_state(root)["state"] != "revoked")
        result = {"schema_version": schema_version, **selected,
                  "attempt_count": before["attempt_count"], "revision": before["revision"],
                  "state": state, "code": _CODES[state],
                  **({"terminal_reason": terminal_reason} if schema_version in {STATUS_V2, STATUS_V3, STATUS_V4} else {}),
                  **({"outcome": outcome} if schema_version in {STATUS_V3, STATUS_V4} else {}),
                  **({"family_authoring": family_authoring} if schema_version == STATUS_V4 else {}),
                  "correlation_id": cross_runtime_canonical_digest({**selected,
                      "attempt_count": before["attempt_count"], "revision": before["revision"]})}
        result["status_digest"] = cross_runtime_canonical_digest(result)
        if schema_version == STATUS_V4 and len(json.dumps(result, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False).encode()) > 65536:
            result["family_authoring"] = None
            result["status_digest"] = cross_runtime_canonical_digest(result, digest_field="status_digest")
            if _projection_diagnostics is not None:
                _projection_diagnostics.append("website_family_authoring_payload_limit")
        return result
    except PreparationStatusUnavailable:
        raise
    except Exception as exc:
        raise PreparationStatusUnavailable(_UNAVAILABLE) from exc


def retain_preparation_wakeup(capture_root: Path) -> None:
    """Persist only selectors after a committed ledger; no network under lock.

    Reconciliation scans actual ledgers too, closing the crash gap between a
    ledger commit and this separate durable delivery record.
    """
    from .handoff_job_state import _existing_job_ledger_lock, _read_job_ledger
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    from .website_preparation_delivery_index import update as update_index
    root = _root(capture_root)
    birth = capture_birth_source_projection(root, expected_purpose="scene_preparation")
    if not birth:
        return
    context = _retained_context_or_none(root)
    if context is None:
        if not (root / "pipeline_job_ledger.json").is_file():
            return  # Nothing claimed yet; the lock below re-checks before any write.
        # An attempt that failed before retaining a context is still reported,
        # bound to the site's current context (one signed read, before the lock).
        purpose = birth.get("capture_observation_purpose")
        _require(purpose in {None, "scene_preparation"})
        context = _current_context(request_id=birth["request_id"], scene_id=birth["scene_id"],
            capture_id=birth["capture_id"], purpose=purpose, allow_unconfirmed_preparation=True)
    with _existing_job_ledger_lock(root) as lock_state:
        if lock_state != "ledger_present":
            return
        ledger = _read_job_ledger(root)
        marker = _read(Path(birth["birth_delivery_raw_ref"]["path"]))["source_finalize"]["generation"]
        selected = validate_selectors({"request_id": birth["request_id"], "scene_id": birth["scene_id"],
            "capture_id": birth["capture_id"], "completion_marker_generation": marker,
            "producer_delivery_key": ledger.get("producer_delivery_key"),
            "source_payload_sha256": ledger.get("source_payload_sha256"),
            "task_context_digest": context.get("context_digest")})
        _require(birth["delivery_key"] == selected["producer_delivery_key"])
        revision = ledger.get("revision")
        _require(type(revision) is int and revision > 0)
        path = root / DELIVERY_FILE
        old = _read_delivery(path) if path.is_file() else {}
        if old.get("selectors") == selected and old.get("revision") == revision:
            if old.get("state") == "pending" and "pending_since_epoch" not in old:
                old = {**old, "pending_since_epoch": time.time(), "next_attempt_at_epoch": 0.0}
                write_json(path, old)
            update_index(root, old)
            return
        delivery = {"schema_version": "website_preparation_status_delivery.v1",
                          "state": "pending", "selectors": selected, "revision": revision,
                          "delivery_attempt_count": 0, "delivery_failure_count": 0,
                          "pending_since_epoch": time.time(), "next_attempt_at_epoch": 0.0}
        write_json(path, delivery)
        update_index(root, delivery)


def validate_acceptance(value: Any, selected: Mapping[str, str]) -> dict[str, Any]:
    """Require the source-bound receipt issued only after the WebApp commit."""
    fields = SELECTORS | {"schema_version", "accepted", "status_digest", "state",
                          "native_execution_complete", "correlation_id"}
    _require(type(value) is dict and set(value) == fields)
    # The acceptance names the receipt version the WebApp read (v1 from a WebApp
    # that predates v2), so postflight re-reads that same version.
    _require(value["schema_version"] in _ACCEPTANCE
             and value["accepted"] is True and value["native_execution_complete"] is False)
    _require(all(value[key] == selected[key] for key in SELECTORS))
    digest = value["status_digest"]
    _require(type(digest) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None)
    _require(value["state"] in _ACCEPTANCE[value["schema_version"]][1])
    _require(value["correlation_id"] == "bp-prep-" + digest.removeprefix("sha256:")[:16])
    return dict(value)


def accepted_status_version(acceptance: Mapping[str, Any]) -> str:
    return _ACCEPTANCE[acceptance["schema_version"]][0]


def reconcile_preparation_wakeups(storage_root: Path, *, limit: int = 50,
                                 now_epoch: float | None = None) -> dict[str, int]:
    """Existing listener tick retries delivery independently of provider work."""
    from .handoff_job_state import _existing_job_ledger_lock, _read_job_ledger
    from .website_preparation_terminal import note_sweep_liveness, record_unretried_preparation
    from .website_preparation_delivery_index import due, defer_unavailable, update as update_index, METRICS_FILE
    from .website_task_context import website_webapp_request
    counts = {"attempted": 0, "delivered": 0, "unavailable": 0}
    storage_root = Path(storage_root)
    try:
        # Liveness qualifies overdue observations, never per-message exhaustion.
        alive_since = note_sweep_liveness(storage_root)
    except OSError:
        alive_since = None
    roots = sorted(path.parent for path in storage_root.glob("*/scenes/site-*/captures/*/pipeline_job_ledger.json")
                   if path.parent.name.startswith(("walkthrough-", "supplement-")))
    cursor_path = storage_root / ".website_preparation_delivery_cursor.json"
    try:
        cursor = _read(cursor_path).get("last_capture_digest") if cursor_path.is_file() else None
    except (OSError, ValueError):
        cursor = None
    keys = [hashlib.sha256(str(root).encode()).hexdigest() for root in roots]
    if cursor in keys:
        pivot = keys.index(cursor) + 1
        roots = roots[pivot:] + roots[:pivot]
    maximum = max(0, min(limit, 50))
    # Historical/index repair has its own budget. Settled history cannot consume
    # the callback budget; newly retained revisions enter the due index directly.
    bounded = roots[:maximum]
    for root in bounded:
        try:
            if alive_since is not None:
                record_unretried_preparation(root, alive_since=alive_since)
            retain_preparation_wakeup(root)
        except Exception:
            counts["unavailable"] += 1
    now = time.time() if now_epoch is None else now_epoch
    try:
        pending_roots, metrics = due(storage_root, now=now, limit=maximum)
    except Exception:
        pending_roots, metrics = [], {"index_unavailable": True}
        counts["unavailable"] += 1
    success_lags = []
    for pending in pending_roots:
        root = pending.capture_root
        delivery = None
        path = root / DELIVERY_FILE
        attempted = False
        try:
            # Retain the canonical pending record before a current-source
            # lookup can fail. Its failure backoff must not monopolize a small
            # dispatch budget or leave unrelated healthy callbacks unvisited.
            if path.is_file():
                delivery = _read_delivery(path)
            retain_preparation_wakeup(root)
            _require(path.is_file())
            delivery = _read_delivery(path)
            if delivery.get("state") != "pending":
                update_index(root, delivery)
                continue
            if delivery.get("next_attempt_at_epoch", 0.0) > now:
                continue
            selected = validate_selectors(delivery["selectors"])
            read_preparation_status(capture_root=root, selectors=selected)
            counts["attempted"] += 1
            attempted = True
            error = None
            acceptance = None
            current_status = None
            projection_diagnostics = []
            try:
                acceptance = validate_acceptance(website_webapp_request(capture_id=selected["capture_id"],
                    operation="preparation-status", payload=selected), selected)
                # The callback may legitimately commit a newer status than our
                # preflight. Verify against fresh authority, never an old receipt.
                accepted_version = accepted_status_version(acceptance)
                current_status = read_preparation_status(capture_root=root, selectors=selected,
                    schema_version=accepted_version,
                    **({"_projection_diagnostics": projection_diagnostics} if accepted_version == STATUS_V4 else {}))
                _require(acceptance["status_digest"] == current_status["status_digest"]
                         and acceptance["state"] == current_status["state"])
                if projection_diagnostics:
                    error = "FamilyAuthoringProjectionPending"
            except Exception as exc:
                error = type(exc).__name__  # Private retry bookkeeping only, never raw response/error.
            # Network is over. Under lock update only the still-current delivery;
            # a concurrent newer producer retains its own pending revision.
            with _existing_job_ledger_lock(root) as lock_state:
                if lock_state != "ledger_present" or _read(path) != delivery:
                    continue
                if error is None or current_status is not None:
                    ledger = _read_job_ledger(root)
                    _require(current_status is not None and ledger.get("revision") == current_status["revision"]
                             and ledger.get("attempt_count") == current_status["attempt_count"])
                failures = delivery.get("delivery_failure_count", 0) + int(error is not None)
                updated = {**delivery, "state": "pending" if error else "delivered",
                                  "delivery_attempt_count": delivery.get("delivery_attempt_count", 0) + 1,
                                  "delivery_failure_count": failures,
                                  "next_attempt_at_epoch": now + min(1800, 60 * 2 ** min(failures - 1, 5)) if error else 0.0,
                                  "last_attempt_at_epoch": now,
                                  "delivered_at_epoch": now if error is None else None,
                                  "last_error_type": error, "acceptance": acceptance}
                if projection_diagnostics:
                    updated["last_optional_projection_failure"] = {"code": projection_diagnostics[-1],
                                                                  "observed_at_epoch": now}
                if projection_diagnostics:
                    updated["optional_projection_state"] = "pending"
                elif (delivery.get("last_optional_projection_failure") and error is None
                        and current_status is not None and current_status.get("schema_version") == STATUS_V4
                        and current_status.get("family_authoring") is not None):
                    updated["optional_projection_state"] = "current"
                write_json(path, updated)
                update_index(root, updated)
            counts["delivered"] += int(error is None)
            if error is None:
                success_lags.append(max(0.0, now - delivery.get("pending_since_epoch", now)))
        except Exception as exc:
            counts["unavailable"] += 1
            # Source/control outages also back off. This changes no authority,
            # and cannot acknowledge or overwrite a concurrent newer revision.
            if delivery is not None and delivery.get("state") == "pending":
                try:
                    with _existing_job_ledger_lock(root) as lock_state:
                        if lock_state == "ledger_present" and _read(path) == delivery:
                            failures = delivery.get("delivery_failure_count", 0) + 1
                            updated = {**delivery, "state": "pending",
                                "delivery_attempt_count": delivery.get("delivery_attempt_count", 0) + int(attempted),
                                "delivery_failure_count": failures,
                                "next_attempt_at_epoch": now + min(1800, 60 * 2 ** min(failures - 1, 5)),
                                "last_error_type": type(exc).__name__}
                            write_json(path, updated)
                            update_index(root, updated)
                except Exception:
                    pass
            # Unreadable canonical bytes cannot receive canonical bookkeeping.
            # Preserve them and defer their exact recoverable hint instead. CAS
            # refuses a concurrent new revision or a changed scheduling hint.
            try:
                defer_unavailable(pending, now=now)
            except Exception:
                pass
    if bounded:
        # Fair bounded scan across restarts; this cursor selects only members of
        # the server-owned glob, never an arbitrary path or a source authority.
        try:
            write_json(cursor_path, {"schema_version": "website_preparation_delivery_cursor.v1",
                       "last_capture_digest": hashlib.sha256(str(bounded[-1]).encode()).hexdigest()})
        except OSError:
            counts["unavailable"] += 1
    try:
        write_json(storage_root / METRICS_FILE, {"schema_version": "website_preparation_delivery_metrics.v1",
                   "observed_at_epoch": now, **metrics, **counts, "repair_scanned": len(bounded),
                   "successful_delivery_lag_seconds": success_lags})
    except OSError:
        counts["unavailable"] += 1
    return counts
