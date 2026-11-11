"""The drain tick's half of the timed retry of website preparation (``website_preparation_timed_retry``).

Kept apart from that module because it reads the robot fit catalog (``website_robot_fit_hold``), which the
terminal sweep and the status reader, both of which reach that module, must not import.
"""
from __future__ import annotations

import base64
import hashlib
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .website_preparation_timed_retry import iso_epoch, stamp, timed_plan


def robot_fit_spent(root: Path, ledger: Mapping[str, Any]) -> bool:
    """Whether a stopped-short robot fit recheck has used its replays for the current catalog and registry.

    Until then ``reconcile_robot_fit_holds`` owes it; afterwards it waits only for the next change of
    inputs, and the timer may drive it.
    """
    from . import website_robot_fit_hold as fit
    from .decision_evidence_contracts import canonical_digest

    record = ledger.get(fit.FIELD)
    if (type(record) is not dict or not fit._rerun_stopped_short(ledger) or set(record) != fit._RECORD_FIELDS
            or record.get("schema_version") != fit.SCHEMA):
        return False
    digest = canonical_digest({"recheck_inputs": fit.recheck_inputs()})
    return not fit._due(Path(root) / "pipeline", digest, math.inf)


def reconcile_timed_retries(listener, *, storage_root: Path, process_args, limit=1,
                            now: float | None = None) -> dict[str, int]:
    """Run each preparation whose timed retry is due again with its own bytes; at most ``limit`` per tick."""
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    from .website_assessment_resume import MAX_BYTES, _payload_bytes, _require
    from .website_preparation_status import DELIVERY_FILE, _read, _root, read_preparation_status, \
        validate_selectors
    current = time.time() if now is None else now
    counts = {"retried": 0, "waiting": 0}
    roots = sorted(path.parent for path in Path(storage_root).glob(
        f"*/scenes/site-*/captures/*/{listener.JOB_LEDGER_FILENAME}")
        if path.parent.name.startswith(("walkthrough-", "supplement-")))
    # Start after the capture the status reconciliation last visited, as the assessment wait does, so
    # one capture that keeps coming due cannot keep the others waiting.
    cursor_path = Path(storage_root) / ".website_preparation_delivery_cursor.json"
    try:
        cursor = _read(cursor_path).get("last_capture_digest") if cursor_path.is_file() else None
    except (OSError, ValueError):
        cursor = None
    keys = [hashlib.sha256(str(root).encode()).hexdigest() for root in roots]
    if cursor in keys:
        pivot = keys.index(cursor) + 1
        roots = roots[pivot:] + roots[:pivot]
    for root in roots:
        if counts["retried"] >= max(0, min(limit, 50)):
            break
        try:
            _root(root)
            _require(not (root / listener.JOB_LEDGER_FILENAME).is_symlink())
            before = listener._read_job_ledger(root)
            plan = timed_plan(root, before, current, robot_fit_spent=robot_fit_spent)
            if plan is None:
                continue
            if plan["due_at_epoch"] > current:
                counts["waiting"] += 1
                continue
            record = plan["record"]
            _require(len(record["payload_base64"]) <= 4 * ((MAX_BYTES + 2) // 3))
            raw = _payload_bytes(listener, base64.b64decode(record["payload_base64"], validate=True),
                                 record["source_payload_sha256"])
            handoff = listener.parse_handoff_payload(raw)
            _require(root == listener._handoff_capture_root(handoff, storage_root=storage_root))
            _require(handoff.source_finalize is not None and handoff.source_membership_selector is not None)
            # Durable before any network read, so an outage backs off instead of re-driving every tick.
            _require(stamp(root, before, plan, current))
            counts["retried"] += 1
            # The status reader re-checks the active birth and membership, fresh owner, local consent,
            # current context and unchanged ledger; a withdrawn or ended job is never run again.
            selectors = validate_selectors((_read(root / DELIVERY_FILE) or {}).get("selectors"))
            status = read_preparation_status(capture_root=root, selectors=selectors)
            _require(status["revision"] == before["revision"] and status["state"] != "authority_ended")
            birth = capture_birth_source_projection(root, expected_purpose="scene_preparation")
            _require(birth is not None and birth["source_membership_selector"] == handoff.source_membership_selector)
            _require(listener._read_job_ledger(root) == before)
            listener.process_handoff_payload(raw, storage_root=storage_root,
                payload_digest=record["source_payload_sha256"],
                expected_timed_retry={"revision": before["revision"], "record": record, "kind": plan["kind"],
                                      "retry": plan["retry"], "due_at": iso_epoch(plan["due_at_epoch"])},
                **({"expected_preparation_purpose": "scene_preparation"}
                   if birth.get("capture_observation_purpose") == "scene_preparation" else {}), **process_args)
        except Exception:  # noqa: BLE001 - one capture never stops the others; a stopped attempt is retried
            listener.logger.debug("pubsub_handoff.timed_retry_pending")
    return counts



__all__ = ["reconcile_timed_retries", "robot_fit_spent"]
