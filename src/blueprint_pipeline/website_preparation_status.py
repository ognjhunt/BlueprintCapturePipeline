"""ADP-010/day14: current source-bound preparation status and durable wake-ups.

A wake-up is never failure authority. The WebApp re-reads the signed status
route, which verifies the current source/context and the latest ledger revision.
No provider is dispatched, and handed_off is not native or assessment completion.

Beta journey L3/L6: v2 is additive. A WebApp that asks for it (request header
below) also learns that preparation ended for good (failed_terminal, with a
reason) and that an attempt only waits for control-plane capacity
(capacity_delayed) rather than for anything the site provides. Every other
caller, including a WebApp that predates v2, keeps the exact v1 receipt.
"""
from __future__ import annotations

import hashlib
import json
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
# The WebApp opts in per read; an absent or unknown value is answered with v1.
VERSION_HEADER = "x-blueprint-preparation-status-version"
_STATES_V1 = frozenset({"preparing", "awaiting_inputs", "failed_retryable", "authority_ended", "handed_off"})
_STATES_V2 = _STATES_V1 | {"capacity_delayed", "failed_terminal"}
_ACCEPTANCE = {"website_preparation_status_acceptance.v1": (STATUS_V1, _STATES_V1),
               "website_preparation_status_acceptance.v2": (STATUS_V2, _STATES_V2)}
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
    return STATUS_V2 if re.fullmatch(r"[1-9][0-9]{0,2}", text) and int(text) >= 2 else STATUS_V1


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


def _v2_state(ledger: Mapping[str, Any]) -> tuple[str, str | None]:
    """Typed v2 meaning of a ledger that is not handed off."""
    from .website_preparation_terminal import capacity_only, current_terminal
    status = ledger.get("status")
    _require(status in _LEDGER_STATES)
    terminal = current_terminal(ledger) if status != "terminal_authority_ended" else None
    if terminal is not None:
        return "failed_terminal", terminal["reason"]
    if status in {"failed_retryable", "retryable_blocked"} and capacity_only(ledger):
        return "capacity_delayed", None
    return _LEDGER_STATES[status], None


def read_preparation_status(*, capture_root: Path, selectors: Mapping[str, Any],
                            schema_version: str = STATUS_V1) -> dict[str, Any]:
    """Read an as-of snapshot; refuse missing, changed or unverified authority."""
    from .capture_original_owner_observer import load_original_owner_observation
    from .consent_takedown import read_consent_state
    from .handoff_job_state import _read_job_ledger, _output_commit
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    from .website_task_context import load_current_website_task_context, validate_website_task_context

    try:
        _require(schema_version in {STATUS_V1, STATUS_V2})
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
        retained = None if raw_retained is None else validate_website_task_context(raw_retained,
                         request_id=selected["request_id"], scene_id=selected["scene_id"],
                         capture_id=selected["capture_id"], purpose=purpose)
        current = load_current_website_task_context(request_id=selected["request_id"],
                        scene_id=selected["scene_id"], capture_id=selected["capture_id"],
                        **({"purpose": purpose} if purpose else {}))
        _require(current["context_digest"] == selected["task_context_digest"]
                 and (retained is None or retained["context_digest"] == current["context_digest"]))
        # A completed job always retained its context; never hand off without it.
        _require(retained is not None or before.get("status") != "completed")
        if current.get("confirmed") is False:
            from .website_task_context import load_website_scene_sponsorship
            load_website_scene_sponsorship(task_context=current, now=time.time(), create=False)
        terminal_reason = None
        if before.get("status") == "completed":
            handoff = _read(root / "pipeline" / "website_scene_preparation" / "handoff.json")
            _require(handoff.get("schema_version") == "website_scene_handoff.v1"
                     and handoff.get("scene_id") == selected["scene_id"]
                     and handoff.get("capture_id") == selected["capture_id"]
                     and handoff.get("digest") == canonical_digest(handoff, digest_field="digest")
                     and handoff.get("website_intake_outbox", {}).get("state") == "forward_pending")
            _require(bool(_output_commit(root, scene_id=selected["scene_id"], capture_id=selected["capture_id"])))
            state = "handed_off"
        elif schema_version == STATUS_V2:
            state, terminal_reason = _v2_state(before)
        else:
            _require(before.get("status") in _LEDGER_STATES)
            state = _LEDGER_STATES[before["status"]]
        after = _read_job_ledger(root)
        _require(before == after)
        _require(read_consent_state(root)["state"] != "revoked")
        result = {"schema_version": schema_version, **selected,
                  "attempt_count": before["attempt_count"], "revision": before["revision"],
                  "state": state, "code": _CODES[state],
                  **({"terminal_reason": terminal_reason} if schema_version == STATUS_V2 else {}),
                  "correlation_id": cross_runtime_canonical_digest({**selected,
                      "attempt_count": before["attempt_count"], "revision": before["revision"]})}
        result["status_digest"] = cross_runtime_canonical_digest(result)
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
        from .website_task_context import load_current_website_task_context
        purpose = birth.get("capture_observation_purpose")
        _require(purpose in {None, "scene_preparation"})
        context = load_current_website_task_context(request_id=birth["request_id"], scene_id=birth["scene_id"],
            capture_id=birth["capture_id"], **({"purpose": purpose} if purpose else {}))
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
        old = _read(path) if path.is_file() else {}
        if old.get("selectors") == selected and old.get("revision") == revision:
            return
        write_json(path, {"schema_version": "website_preparation_status_delivery.v1",
                          "state": "pending", "selectors": selected, "revision": revision,
                          "delivery_attempt_count": 0})


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


def reconcile_preparation_wakeups(storage_root: Path, *, limit: int = 50) -> dict[str, int]:
    """Existing listener tick retries delivery independently of provider work."""
    from .handoff_job_state import _existing_job_ledger_lock, _read_job_ledger
    from .website_preparation_terminal import record_abandoned_final_delivery
    from .website_task_context import website_webapp_request
    counts = {"attempted": 0, "delivered": 0, "unavailable": 0}
    storage_root = Path(storage_root)
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
    bounded = roots[:max(0, min(limit, 50))]
    for root in bounded:
        try:
            # A final attempt whose worker died is dead-lettered with nothing
            # left to retry it; record that before reporting the ledger.
            record_abandoned_final_delivery(root)
            retain_preparation_wakeup(root)
            path = root / DELIVERY_FILE
            if not path.is_file():
                continue
            delivery = _read(path)
            if delivery.get("state") != "pending":
                continue
            selected = validate_selectors(delivery["selectors"])
            read_preparation_status(capture_root=root, selectors=selected)
            counts["attempted"] += 1
            error = None
            acceptance = None
            current_status = None
            try:
                acceptance = validate_acceptance(website_webapp_request(capture_id=selected["capture_id"],
                    operation="preparation-status", payload=selected), selected)
                # The callback may legitimately commit a newer status than our
                # preflight. Verify against fresh authority, never an old receipt.
                current_status = read_preparation_status(capture_root=root, selectors=selected,
                                                         schema_version=accepted_status_version(acceptance))
                _require(acceptance["status_digest"] == current_status["status_digest"]
                         and acceptance["state"] == current_status["state"])
            except Exception as exc:
                error = type(exc).__name__  # Private retry bookkeeping only, never raw response/error.
            # Network is over. Under lock update only the still-current delivery;
            # a concurrent newer producer retains its own pending revision.
            with _existing_job_ledger_lock(root) as lock_state:
                if lock_state != "ledger_present" or _read(path) != delivery:
                    continue
                if error is None:
                    ledger = _read_job_ledger(root)
                    _require(current_status is not None and ledger.get("revision") == current_status["revision"]
                             and ledger.get("attempt_count") == current_status["attempt_count"])
                write_json(path, {**delivery, "state": "pending" if error else "delivered",
                                  "delivery_attempt_count": delivery.get("delivery_attempt_count", 0) + 1,
                                  "last_error_type": error, "acceptance": acceptance})
            counts["delivered"] += int(error is None)
        except Exception:
            counts["unavailable"] += 1
    if bounded:
        # Fair bounded scan across restarts; this cursor selects only members of
        # the server-owned glob, never an arbitrary path or a source authority.
        try:
            write_json(cursor_path, {"schema_version": "website_preparation_delivery_cursor.v1",
                       "last_capture_digest": hashlib.sha256(str(bounded[-1]).encode()).hexdigest()})
        except OSError:
            counts["unavailable"] += 1
    return counts
