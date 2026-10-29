"""Bounded current-source stops for an exact Blueprint screening invocation.

This reuses the signed task-context reader and the existing team's monitor
grace/identity-bound stopper. It grants no execution and sends no global sync
credential to a provider. Run-specific WebApp cancellation and GPU per-cell
leases remain separate integration requirements.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import fcntl
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


@contextmanager
def _control_lock(root: Path) -> Iterator[None]:
    """Serialize only local control commits/admissions, never remote reads."""
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(root / ".execution_control.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _stopped(control: Mapping[str, Any], root: Path) -> dict[str, Any] | None:
    from .allocator_invocation import _read
    path = root / "stop_requested.json"
    if not path.exists():
        return None
    value = _read(path)
    if (value.get("control_digest") != canonical_digest(control)
            or value.get("prepared_digest") != control["prepared_digest"]):
        raise ValueError("beta_screening_control_binding_changed")
    if (value.get("schema_version") != "blueprint.beta_policy_screening_stop.v1"
            or value.get("applicability") != "historical_only"
            or value.get("new_processing_authorized") is not False):
        raise ValueError("beta_screening_control_record_unproven")
    return value


def _state(control: Mapping[str, Any], root: Path, moment: float) -> dict[str, Any]:
    from .allocator_invocation import _read
    path = root / "execution_control.json"
    previous = _read(path) if path.exists() else {}
    binding = canonical_digest(control)
    if previous and previous.get("control_digest") != binding:
        raise ValueError("beta_screening_control_binding_changed")
    last_success = previous.get("last_success_at_epoch", control["initial_observed_at_epoch"])
    deadlines = [value for value in (previous.get("execution_deadline_epoch"),
                                     control.get("execution_deadline_epoch")) if value is not None]
    issued, committed = previous.get("issued_generation", 0), previous.get("committed_generation", 0)
    if (type(last_success) not in (int, float) or not math.isfinite(last_success)
            or last_success > moment + POLL_SECONDS or last_success <= 0
            or any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0
                   for value in deadlines)
            or type(issued) is not int or type(committed) is not int or not 0 <= committed <= issued
            or previous.get("status", "monitor_unavailable") not in {"current", "monitor_unavailable"}):
        raise ValueError("beta_screening_control_record_unproven")
    return {"schema_version": "blueprint.beta_policy_screening_execution_control.v1",
            "control_digest": binding, "observed_at_epoch": previous.get("observed_at_epoch", moment),
            "last_success_at_epoch": last_success,
            "next_check_at_epoch": previous.get("next_check_at_epoch", moment),
            "execution_deadline_epoch": min(deadlines) if deadlines else None,
            "status": previous.get("status", "monitor_unavailable"),
            "issued_generation": issued, "committed_generation": committed,
            **({"source_control": previous["source_control"]} if previous.get("source_control") else {})}


def _stop(control: Mapping[str, Any], root: Path, reason: str, *, source_control=None) -> dict[str, Any]:
    from .allocator_invocation import _once
    existing = _stopped(control, root)
    if existing is not None:
        return existing
    value = {"schema_version": "blueprint.beta_policy_screening_stop.v1",
             "control_digest": canonical_digest(control), "prepared_digest": control["prepared_digest"],
             "reason": reason, "observed_at_iso": utc_now_iso(),
             "applicability": "historical_only", "new_processing_authorized": False,
             **({"source_control": source_control} if source_control is not None else {})}
    _once(root / "stop_requested.json", value)
    return value


def observe(*, control: Mapping[str, Any], root: Path, now: float | None = None) -> dict[str, Any]:
    """Commit monotonic observations; a stop is absorbing across delayed reads."""
    from .beta_policy_screening_worker import require_current_source
    moment = time.time() if now is None else now
    with _control_lock(root):
        stopped = _stopped(control, root)
        if stopped is not None:
            return stopped
        state = _state(control, root, moment)
        if state["execution_deadline_epoch"] is not None and moment >= state["execution_deadline_epoch"]:
            return _stop(control, root, "beta_screening_authority_expired")
        generation = state["issued_generation"] + 1
        state["issued_generation"] = generation
        write_json(root / "execution_control.json", state)
    # A slow signed read cannot block a newer revocation or provider closeout.
    outcome, deadline, source_control = "current", None, None
    try:
        if control.get("source_control_binding") is not None:
            from .beta_policy_screening_control_transport import request_control
            source_control = request_control(binding=control["source_control_binding"],
                execution_deadline_epoch=state["execution_deadline_epoch"])
            deadline = _deadline(source_control)
            if source_control["status"] == "stopped":
                outcome = "revoked"
        else:
            deadline = _deadline(require_current_source(control["source"], site_id=control["site_id"]))
    except (OSError, ValueError, RuntimeError) as exc:
        code = str(exc)
        outcome = ("revoked" if code in {"beta_screening_current_capture_consent_required",
            "beta_screening_current_task_context_changed", "beta_screening_source_context_mismatch",
            "website_task_context_not_confirmed"} or code.startswith("website_control_task-context_http_409:")
            else "monitor_unavailable")
    completed = time.time() if now is None else now
    with _control_lock(root):
        stopped = _stopped(control, root)
        if stopped is not None:
            return stopped
        # Reload after the remote read: never overwrite a newer stop/deadline.
        state = _state(control, root, completed)
        if deadline is not None:
            old = state["execution_deadline_epoch"]
            state["execution_deadline_epoch"] = min(old, deadline) if old is not None else deadline
        if outcome == "revoked":
            return _stop(control, root, "beta_screening_current_source_revoked", source_control=source_control)
        if state["execution_deadline_epoch"] is not None and completed >= state["execution_deadline_epoch"]:
            return _stop(control, root, "beta_screening_authority_expired")
        if generation >= state["committed_generation"]:
            state["status"] = outcome
            state["observed_at_epoch"] = max(completed, state["observed_at_epoch"])
            state["next_check_at_epoch"] = state["observed_at_epoch"] + POLL_SECONDS
            state["committed_generation"] = generation
            if outcome == "current":
                state["last_success_at_epoch"] = max(moment, state["last_success_at_epoch"])
                if source_control is not None:
                    state["source_control"] = source_control
        if completed - state["last_success_at_epoch"] >= MONITOR_GRACE_SECONDS:
            return _stop(control, root, "beta_screening_control_unavailable_timeout")
        write_json(root / "execution_control.json", state)
        return state


@contextmanager
def current_control_admission(*, control: Mapping[str, Any], root: Path) -> Iterator[None]:
    """Atomically check current control and retain a bounded local create intent.

    Lock order is invocation -> control. Observers never acquire the invocation
    lock; neither lock is held by this scope for a remote call or teardown.
    """
    observed = observe(control=control, root=root)
    with _control_lock(root):
        stopped = _stopped(control, root)
        state = _state(control, root, time.time())
        deadline = state["execution_deadline_epoch"]
        if deadline is not None and time.time() >= deadline:
            stopped = _stop(control, root, "beta_screening_authority_expired")
        if (stopped is not None or observed.get("new_processing_authorized") is False
                or state["status"] != "current"
                or time.time() - state["last_success_at_epoch"] >= MONITOR_GRACE_SECONDS):
            raise ValueError("beta_screening_current_source_unproven_before_create")
        yield


_monitors_lock = threading.Lock()
_monitors: dict[tuple[str, str], "CurrentSourceMonitor"] = {}


def _active_monitor(*, authority_digest: str, resource_name: str) -> "CurrentSourceMonitor":
    with _monitors_lock:
        monitor = _monitors.get((authority_digest, resource_name))
    if monitor is None or monitor.done.is_set() or not monitor.thread.is_alive():
        raise ValueError("beta_screening_runtime_control_monitor_unavailable")
    monitor.current_snapshot()
    return monitor


def require_runtime_control_monitor(*, authority_digest: str, resource_name: str) -> None:
    """Before provider mutation, an exact beta run must have its live monitor."""
    _active_monitor(authority_digest=authority_digest, resource_name=resource_name)


def subscribe_runtime_control(*, authority_digest: str, resource_name: str,
                              sink: Callable[[Mapping[str, Any], Mapping[str, Any]], Any]) -> Callable[[], None]:
    """Deliver verified observations through an already-owned provider transport.

    No service credential is sent to the sink. It receives only the exact
    immutable binding and validated historical/current control response.
    """
    monitor = _active_monitor(authority_digest=authority_digest, resource_name=resource_name)
    return monitor.subscribe(sink)


class CurrentSourceMonitor:
    """A monitor owned by the allocator process, independent of its coordinator."""

    def __init__(self, value: Mapping[str, Any], *, stop=None):
        self.control, self.root = value["control"], Path(value["root"])
        self.stop = stop or (lambda: os.kill(os.getpid(), signal.SIGTERM))
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._snapshot_lock = threading.Lock()
        self._delivery_lock = threading.Lock()
        self._latest: dict[str, Any] = {}
        self._sinks: dict[object, Callable] = {}
        self.binding = self.control.get("source_control_binding")
        if self.binding is not None:
            from .beta_policy_screening_control_contracts import validate_binding
            validate_binding(self.binding)
            if any(self.control.get(key) != self.binding[key] for key in ("source", "site_id", "prepared_digest")):
                raise ValueError("beta_screening_control_binding_changed")
        self.identity = ((self.binding["authority_digest"], self.binding["resource_name"])
                         if self.binding is not None else None)

    def current_snapshot(self) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        from .beta_policy_screening_control_contracts import validate_control
        with _control_lock(self.root):
            stopped = _stopped(self.control, self.root)
            state = _state(self.control, self.root, time.time())
        if (self.binding is None or state.get("status") != "current"
                or stopped is not None or state.get("new_processing_authorized") is False):
            raise ValueError("beta_screening_runtime_control_monitor_unavailable")
        remote = validate_control(state.get("source_control"), binding=self.binding)
        return self.binding, remote

    def subscribe(self, sink: Callable) -> Callable[[], None]:
        token = object()
        # Serialize initial snapshot delivery with monitor updates so an older
        # healthy initial push cannot follow a newer stopped push.
        with self._delivery_lock:
            binding, remote = self.current_snapshot()
            with self._snapshot_lock:
                self._sinks[token] = sink
            try:
                sink(binding, remote)
            except BaseException:
                with self._snapshot_lock:
                    self._sinks.pop(token, None)
                raise
        def unsubscribe():
            with self._delivery_lock:
                with self._snapshot_lock:
                    self._sinks.pop(token, None)
        return unsubscribe

    def allocator_argv(self, argv) -> list[str]:
        """Pass the latest smaller actual expiry into provider/watchdog caps."""
        effective = list(argv)
        if self.binding is None:
            return effective
        self.current_snapshot()
        with _control_lock(self.root):
            state = _state(self.control, self.root, time.time())
        deadline = state["execution_deadline_epoch"]
        flag = "--team-candidate-execution-deadline-epoch"
        if effective.count(flag) > 1 or deadline is None:
            raise ValueError("beta_screening_execution_control_deadline_unproven")
        if flag in effective:
            index = effective.index(flag) + 1
            prior = float(effective[index])
            if not math.isfinite(prior) or prior <= 0:
                raise ValueError("beta_screening_execution_control_deadline_unproven")
            deadline = min(prior, deadline)
            effective[index] = str(deadline)
        else:
            effective += [flag, str(deadline)]
        write_json(self.root / "effective_execution_deadline.json", {
            "schema_version": "blueprint.beta_screening_effective_deadline.v1",
            "binding": self.binding, "execution_deadline_epoch": deadline})
        return effective

    def _publish(self, state: Mapping[str, Any]) -> None:
        # Only a successful signed observation can refresh the GPU file. A
        # service outage leaves its original freshness running down locally.
        with self._delivery_lock:
            with self._snapshot_lock:
                self._latest = dict(state)
                sinks = list(self._sinks.values())
            remote = state.get("source_control")
            if remote is None or self.binding is None or state.get("status") == "monitor_unavailable":
                return
            from .beta_policy_screening_control_contracts import validate_control
            verified = validate_control(remote, binding=self.binding)
            for sink in sinks:
                sink(self.binding, verified)

    def _run(self):
        while not self.done.wait(POLL_SECONDS):
            try:
                state = observe(control=self.control, root=self.root)
                self._publish(state)
            except Exception:
                # An unprovable local record or failed own-resource control
                # delivery cannot extend processing. Existing finally remains.
                self.done.set()
                self.stop()
                return
            if state.get("new_processing_authorized") is False:
                self.done.set()
                self.stop()
                return

    def __enter__(self):
        state = observe(control=self.control, root=self.root)
        if state.get("new_processing_authorized") is False or state.get("status") == "monitor_unavailable":
            raise ValueError("beta_screening_current_source_unproven_before_start")
        self._publish(state)
        if self.identity is not None:
            with _monitors_lock:
                if self.identity in _monitors:
                    raise ValueError("beta_screening_runtime_control_monitor_conflict")
                _monitors[self.identity] = self
        try:
            self.thread.start()
        except BaseException:
            with _monitors_lock:
                if self.identity is not None:
                    _monitors.pop(self.identity, None)
            raise
        return self

    def __exit__(self, *args):
        self.done.set()
        with _monitors_lock:
            if self.identity is not None and _monitors.get(self.identity) is self:
                _monitors.pop(self.identity, None)
        self.thread.join(timeout=1)
