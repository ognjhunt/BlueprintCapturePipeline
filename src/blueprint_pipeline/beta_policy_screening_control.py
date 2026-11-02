"""Bounded current-source stops for an exact Blueprint screening invocation.

This reuses the signed task-context reader and the existing team's monitor
grace/identity-bound stopper. It grants no execution and sends no global sync
credential to a provider. Run-specific WebApp cancellation and GPU per-cell
leases remain separate integration requirements.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import signal
import os
import threading
import time
from typing import Any

from .common import write_json, utc_now_iso
from .decision_evidence_contracts import canonical_digest

POLL_SECONDS = 60
MONITOR_GRACE_SECONDS = 600


def _deadline(current: Mapping[str, Any]) -> float | None:
    rights = current.get("capture_rights") or {}
    candidates = [current.get("execution_deadline_at_ms"), rights.get("consent_expires_at_epoch")]
    values = []
    for index, value in enumerate(candidates):
        if value is None:
            continue
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError("beta_screening_control_deadline_invalid")
        values.append(float(value) / 1000 if index == 0 else float(value))
    return min(values) if values else None


def observe(*, control: Mapping[str, Any], root: Path, now: float | None = None) -> dict[str, Any]:
    """Persist a current observation or an immutable stop; never reauthorize."""
    from .allocator_invocation import _read, _once
    from .beta_policy_screening_worker import require_current_source
    moment = time.time() if now is None else now
    binding = canonical_digest(control)
    stop_path, state_path = root / "stop_requested.json", root / "execution_control.json"
    if stop_path.exists():
        stopped = _read(stop_path)
        if stopped.get("control_digest") != binding:
            raise ValueError("beta_screening_control_binding_changed")
        return stopped
    previous = _read(state_path) if state_path.exists() else {}
    if previous and previous.get("control_digest") != binding:
        raise ValueError("beta_screening_control_binding_changed")
    last_success = previous.get("last_success_at_epoch", control["initial_observed_at_epoch"])
    old_deadline = previous.get("execution_deadline_epoch", control.get("execution_deadline_epoch"))
    if (type(last_success) not in (int, float) or not math.isfinite(last_success)
            or last_success > moment + 60 or last_success <= 0
            or old_deadline is not None and (type(old_deadline) not in (float, int)
                or not math.isfinite(old_deadline) or old_deadline <= 0)):
        raise ValueError("beta_screening_control_record_unproven")
    reason = None
    state = {"schema_version": "blueprint.beta_policy_screening_execution_control.v1",
             "control_digest": binding, "observed_at_epoch": moment,
             "last_success_at_epoch": last_success, "next_check_at_epoch": moment + POLL_SECONDS,
             "execution_deadline_epoch": old_deadline, "status": "current"}
    if old_deadline is not None and moment >= old_deadline:
        reason = "beta_screening_authority_expired"
    else:
        try:
            current = require_current_source(control["source"], site_id=control["site_id"])
            deadline = _deadline(current)
            if deadline is not None:
                state["execution_deadline_epoch"] = min(deadline, old_deadline) if old_deadline else deadline
            state["last_success_at_epoch"] = moment
            if state["execution_deadline_epoch"] is not None and moment >= state["execution_deadline_epoch"]:
                reason = "beta_screening_authority_expired"
        except (OSError, ValueError, RuntimeError) as exc:
            code = str(exc)
            if code in {"beta_screening_current_capture_consent_required",
                        "beta_screening_current_task_context_changed", "beta_screening_source_context_mismatch",
                        "website_task_context_not_confirmed"} or code.startswith("website_control_task-context_http_409:"):
                reason = "beta_screening_current_source_revoked"
            else:
                state["status"] = "monitor_unavailable"
                if moment - last_success >= MONITOR_GRACE_SECONDS:
                    reason = "beta_screening_control_unavailable_timeout"
    if reason:
        stopped = {"schema_version": "blueprint.beta_policy_screening_stop.v1",
                   "control_digest": binding, "prepared_digest": control["prepared_digest"],
                   "reason": reason, "observed_at_iso": utc_now_iso(),
                   "applicability": "historical_only", "new_processing_authorized": False}
        _once(stop_path, stopped)
        return stopped
    write_json(state_path, state)
    return state


class CurrentSourceMonitor:
    """A monitor owned by the allocator process, independent of its coordinator."""

    def __init__(self, value: Mapping[str, Any], *, stop=None):
        self.control, self.root = value["control"], Path(value["root"])
        self.stop = stop or (lambda: os.kill(os.getpid(), signal.SIGTERM))
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.done.wait(POLL_SECONDS):
            try:
                state = observe(control=self.control, root=self.root)
            except Exception:
                # An unprovable local control record cannot extend processing.
                self.stop()
                return
            if state.get("new_processing_authorized") is False:
                self.stop()
                return

    def __enter__(self):
        state = observe(control=self.control, root=self.root)
        if state.get("new_processing_authorized") is False or state.get("status") == "monitor_unavailable":
            raise ValueError("beta_screening_current_source_unproven_before_start")
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join(timeout=1)
