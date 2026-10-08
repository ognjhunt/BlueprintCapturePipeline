"""ADP-010/day14: current source-bound preparation status and durable wake-ups.

A wake-up is never failure authority. The WebApp re-reads the signed status
route, which verifies the current source/context and the latest ledger revision.
No provider is dispatched, and handed_off is not native or assessment completion.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

SELECTORS = frozenset({"request_id", "scene_id", "capture_id", "completion_marker_generation",
                      "producer_delivery_key", "source_payload_sha256", "task_context_digest"})
DELIVERY_FILE = "website_preparation_status_delivery.json"
_UNAVAILABLE = "website_preparation_status_unavailable"


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
    _require(value["scene_id"] == f"site-{request}" and value["capture_id"] == f"walkthrough-{request}")
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


def read_preparation_status(*, capture_root: Path, selectors: Mapping[str, Any]) -> dict[str, Any]:
    """Read an as-of snapshot; refuse missing, changed or unverified authority."""
    from .capture_original_owner_observer import load_original_owner_observation
    from .consent_takedown import read_consent_state
    from .pubsub_handoff_listener import _read_job_ledger, _output_commit
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    from .website_task_context import load_current_website_task_context, validate_website_task_context

    try:
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
        birth = capture_birth_source_projection(root)
        _require(type(birth) is dict)
        _require(all(birth.get(key) == selected[key] for key in ("request_id", "scene_id", "capture_id")))
        _require(birth.get("delivery_key") == selected["producer_delivery_key"])
        delivery = _read(Path(birth["birth_delivery_raw_ref"]["path"]))
        _require(delivery["source_finalize"]["generation"] == selected["completion_marker_generation"])
        owner = load_original_owner_observation(bucket=delivery["source_finalize"]["bucket"], scene_id=selected["scene_id"],
            capture_id=selected["capture_id"], marker_generation=selected["completion_marker_generation"])
        _require(owner["capture_owner"]["user_id"] == birth["capture_owner_user_id"])
        _require(owner["producer_delivery"] == delivery["producer_delivery"])
        _require(owner["producer_delivery"]["delivery_key"] == selected["producer_delivery_key"])
        _require(owner["producer_delivery"]["raw_video"]["generation"] == birth["raw_video"]["generation"])
        _require(owner["producer_delivery"]["raw_video"]["object_name"] == birth["raw_video"]["object_name"])
        # Current fresh owner transport validates rights and complete generation
        # authority; historical birth alone grants no current consent.
        _require(read_consent_state(root)["state"] != "revoked")
        retained = validate_website_task_context(_retained_context(root), request_id=selected["request_id"],
                         scene_id=selected["scene_id"], capture_id=selected["capture_id"])
        current = load_current_website_task_context(request_id=selected["request_id"],
                        scene_id=selected["scene_id"], capture_id=selected["capture_id"])
        _require(retained["context_digest"] == current["context_digest"] == selected["task_context_digest"])
        state_code = {"processing": ("preparing", "preparation_in_progress"),
                      "failed_retryable": ("failed_retryable", "preparation_retryable_failure"),
                      "retryable_blocked": ("awaiting_inputs", "preparation_inputs_pending"),
                      "terminal_authority_ended": ("authority_ended", "preparation_authority_ended")}
        if before.get("status") == "completed":
            handoff = _read(root / "pipeline" / "website_scene_preparation" / "handoff.json")
            _require(handoff.get("schema_version") == "website_scene_handoff.v1"
                     and handoff.get("scene_id") == selected["scene_id"]
                     and handoff.get("capture_id") == selected["capture_id"]
                     and handoff.get("digest") == canonical_digest(handoff, digest_field="digest")
                     and handoff.get("website_intake_outbox", {}).get("state") == "forward_pending")
            _require(bool(_output_commit(root, scene_id=selected["scene_id"], capture_id=selected["capture_id"])))
            state, code = "handed_off", "preparation_handed_off"
        else:
            _require(before.get("status") in state_code)
            state, code = state_code[before["status"]]
        after = _read_job_ledger(root)
        _require(before == after)
        _require(read_consent_state(root)["state"] != "revoked")
        result = {"schema_version": "website_preparation_status.v1", **selected,
                  "attempt_count": before["attempt_count"], "revision": before["revision"],
                  "state": state, "code": code,
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
    from .pubsub_handoff_listener import _existing_job_ledger_lock, _read_job_ledger
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    root = _root(capture_root)
    birth = capture_birth_source_projection(root)
    if not birth:
        return
    context = _retained_context(root)
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


def reconcile_preparation_wakeups(storage_root: Path, *, limit: int = 50) -> dict[str, int]:
    """Existing listener tick retries delivery independently of provider work."""
    from .pubsub_handoff_listener import _existing_job_ledger_lock
    from .website_task_context import website_webapp_request
    counts = {"attempted": 0, "delivered": 0, "unavailable": 0}
    storage_root = Path(storage_root)
    roots = sorted(path.parent for path in storage_root.glob("*/scenes/site-*/captures/walkthrough-*/pipeline_job_ledger.json"))
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
            try:
                website_webapp_request(capture_id=selected["capture_id"], operation="preparation-status", payload=selected)
            except Exception as exc:
                error = type(exc).__name__  # Private retry bookkeeping only, never raw response/error.
            # Network is over. Under lock update only the still-current delivery;
            # a concurrent newer producer retains its own pending revision.
            with _existing_job_ledger_lock(root) as lock_state:
                if lock_state != "ledger_present" or _read(path) != delivery:
                    continue
                write_json(path, {**delivery, "state": "pending" if error else "delivered",
                                  "delivery_attempt_count": delivery.get("delivery_attempt_count", 0) + 1,
                                  "last_error_type": error})
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
