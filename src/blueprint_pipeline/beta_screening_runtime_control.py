"""GPU-local screening stop checks; no network client or service credentials.

The existing controller delivers observations over its pinned, own-instance SSH
channel. These files confer no launch or spending authority. Missing, stale,
rebound or stopped observations prohibit new processing while closeout remains
permitted. A local stop is absorbing across process and cell restarts.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .beta_policy_screening_control_contracts import validate_binding, validate_control as _validate_control

CONTROL_FILENAME = "blueprint_beta_execution_control.v1.json"
STOP_FILENAME = "blueprint_beta_execution_stop.v1.json"
STATE_FILENAME = "blueprint_beta_execution_guard_state.v1.json"
POLL_SECONDS = 60
MONITOR_GRACE_SECONDS = 600
MAX_BYTES = 32 * 1024
BETA_RUN_KIND = "blueprint_beta_screening"


def _epoch_ms(value: Any) -> bool:
    return type(value) is int and 0 < value <= 8_640_000_000_000_000


def _digest(value: Mapping[str, Any], field: str) -> str:
    return cross_runtime_canonical_digest({key: item for key, item in value.items() if key != field})


def validate_control(value: Mapping[str, Any], *, binding: Mapping[str, Any]) -> dict[str, Any]:
    """Share exact wire validation; the GPU applies its retained outage grace below."""
    try:
        observed = datetime.fromisoformat(value["observed_at_iso"].replace("Z", "+00:00")).timestamp()
        return _validate_control(value, binding=binding, now=observed)
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError, OSError):
        raise ValueError("beta_screening_runtime_observation_invalid") from None


def _session(authority: Mapping[str, Any]) -> dict[str, Any]:
    if (authority.get("run_kind") != BETA_RUN_KIND
            or authority.get("authority_digest") != canonical_digest(authority, digest_field="authority_digest")):
        raise ValueError("beta_screening_runtime_authority_invalid")
    return {key: authority[key] for key in ("run_id", "authority_digest", "runtime_inputs_digest", "resource_name")}


def make_delivery(*, authority: Mapping[str, Any], binding: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    validate_binding(binding)
    control = validate_control(control, binding=binding)
    session = _session(authority)
    if (binding["authority_digest"] != session["authority_digest"] or binding["resource_name"] != session["resource_name"]
            or binding["plan_digest"] != authority.get("plan_digest")
            or binding["configuration_digest"] != authority.get("configuration_digest")
            or [binding["candidate_id"]] != authority.get("candidate_ids")
            or binding["source"]["scene_revision_digest"] != authority.get("scene_revision_digest")):
        raise ValueError("beta_screening_runtime_session_mismatch")
    value = {"schema_version": "blueprint.beta_screening_runtime_control_delivery.v1", "session": session,
             "control": control, "maximum_execution_deadline_at_ms": control["execution_deadline_at_ms"]}
    return {**value, "delivery_digest": _digest(value, "delivery_digest")}


def _read(path: Path) -> dict[str, Any]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        attributes = os.fstat(fd)
        if (not stat.S_ISREG(attributes.st_mode) or attributes.st_size > MAX_BYTES
                or attributes.st_mode & 0o022 or attributes.st_uid != os.geteuid()):
            raise ValueError("beta_screening_runtime_file_unsafe")
        with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("beta_screening_runtime_file_invalid")
        return value
    finally:
        os.close(fd)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    if len(raw) > MAX_BYTES or path.is_symlink():
        raise ValueError("beta_screening_runtime_file_unsafe")
    fd, temporary = tempfile.mkstemp(prefix=".beta-control-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _lock(root: Path):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("beta_screening_runtime_root_unsafe")
    fd = os.open(root / ".blueprint_beta_execution_control.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _validate_delivery(value: Mapping[str, Any], *, authority: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {"schema_version", "session", "control", "maximum_execution_deadline_at_ms", "delivery_digest"}:
        raise ValueError("beta_screening_runtime_delivery_invalid")
    control = value.get("control")
    if (value.get("schema_version") != "blueprint.beta_screening_runtime_control_delivery.v1"
            or value.get("delivery_digest") != _digest(value, "delivery_digest") or value.get("session") != _session(authority)
            or not isinstance(control, Mapping) or not isinstance(control.get("binding"), Mapping)
            or not _epoch_ms(value.get("maximum_execution_deadline_at_ms"))):
        raise ValueError("beta_screening_runtime_delivery_invalid")
    make_delivery(authority=authority, binding=control["binding"], control=control)
    if control["execution_deadline_at_ms"] > value["maximum_execution_deadline_at_ms"]:
        raise ValueError("beta_screening_runtime_deadline_extended")
    return dict(value)


def install_runtime_control(*, runtime_root: Path, delivery: Mapping[str, Any]) -> dict[str, Any]:
    """Fixed-file receiver invoked by the controller's fixed pinned SSH command."""
    runtime = Path(runtime_root)
    root = runtime.parent
    authority = _read(runtime / "runtime_inputs" / "policy_canary_session_authority.json")
    value = _validate_delivery(delivery, authority=authority)
    path = root / CONTROL_FILENAME
    with _lock(root):
        if path.exists():
            previous = _validate_delivery(_read(path), authority=authority)
            old, new = previous["control"], value["control"]
            if old["binding_digest"] != new["binding_digest"]:
                raise ValueError("beta_screening_runtime_binding_changed")
            if old["status"] == "stopped" or new["status"] == "current" and new["revision"] < old["revision"]:
                return previous
            if new["revision"] == old["revision"]:
                if old != new:
                    raise ValueError("beta_screening_runtime_revision_conflict")
                return previous
            if new["execution_deadline_at_ms"] > old["execution_deadline_at_ms"]:
                raise ValueError("beta_screening_runtime_deadline_extended")
            value["maximum_execution_deadline_at_ms"] = previous["maximum_execution_deadline_at_ms"]
            value["delivery_digest"] = _digest(value, "delivery_digest")
        _write(path, value)
    return value


class RuntimeControlStopped(RuntimeError):
    def __init__(self, receipt: Mapping[str, Any]):
        self.receipt = dict(receipt)
        super().__init__(str(receipt["reason"]))


class RuntimeControlGuard:
    def __init__(self, *, runtime_root: Path, inputs: Mapping[str, Any], authority: Mapping[str, Any]):
        self.runtime, self.inputs, self.authority = Path(runtime_root), inputs, authority
        self.root = self.runtime.parent
        self.session = _session(authority)
        # The existing provider entrypoint supplies the watchdog's absolute
        # deadline and forbids runtime-environment overrides. Freeze it once;
        # later cells also inherit the smaller retained guard-state deadline.
        self.maximum_deadline_ms = None
        for limit in (authority.get("hard_deadline_epoch"), os.environ.get("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH")):
            if limit is None:
                continue
            try:
                if isinstance(limit, bool):
                    raise ValueError
                numeric_limit = float(limit)
                if not math.isfinite(numeric_limit) or not _epoch_ms(int(numeric_limit * 1000)):
                    raise ValueError
                candidate = int(numeric_limit * 1000)
                self.maximum_deadline_ms = min(candidate, self.maximum_deadline_ms) if self.maximum_deadline_ms is not None else candidate
            except (ValueError, TypeError, OverflowError):
                raise ValueError("beta_screening_runtime_deadline_invalid") from None
        if (inputs.get("run_kind") != BETA_RUN_KIND or inputs.get("run_id") != self.session["run_id"]
                or inputs.get("runtime_inputs_digest") != self.session["runtime_inputs_digest"]
                or inputs.get("plan_digest") != authority.get("plan_digest")
                or inputs.get("configuration_digest") != authority.get("configuration_digest")
                or inputs.get("candidate_ids") != authority.get("candidate_ids")
                or inputs.get("scene_revision_digest") != authority.get("scene_revision_digest")):
            raise ValueError("beta_screening_runtime_session_mismatch")

    def stop_receipt(self) -> dict[str, Any] | None:
        path = self.root / STOP_FILENAME
        if not path.exists():
            return None
        value = _read(path)
        if (value.get("session") != self.session or value.get("schema_version") != "blueprint.beta_screening_runtime_stop.v1"
                or value.get("receipt_digest") != _digest(value, "receipt_digest")
                or value.get("applicability") != "historical_only" or value.get("new_processing_authorized") is not False
                or value.get("read_only_closeout_allowed") is not True):
            raise ValueError("beta_screening_runtime_stop_invalid")
        return value

    def _stop(self, reason: str, boundary: str, binding_digest: str | None = None,
              source_control: Mapping[str, Any] | None = None) -> RuntimeControlStopped:
        value = {"schema_version": "blueprint.beta_screening_runtime_stop.v1", "session": self.session,
                 "binding_digest": binding_digest, "reason": reason, "boundary": boundary,
                 "observed_at_epoch": int(time.time()), "applicability": "historical_only",
                 "new_processing_authorized": False, "read_only_closeout_allowed": True,
                 "source_control": dict(source_control) if source_control is not None else None}
        receipt = {**value, "receipt_digest": _digest(value, "receipt_digest")}
        _write(self.root / STOP_FILENAME, receipt)
        return RuntimeControlStopped(receipt)

    def check(self, boundary: str) -> dict[str, Any]:
        """Check before a new processing boundary; never needed for teardown."""
        if re.fullmatch("[a-z_]+", boundary) is None:
            raise ValueError("beta_screening_runtime_boundary_invalid")
        with _lock(self.root):
            stopped = self.stop_receipt()
            if stopped is not None:
                raise RuntimeControlStopped(stopped)
            binding_digest = None
            control = None
            try:
                delivery = _validate_delivery(_read(self.root / CONTROL_FILENAME), authority=self.authority)
                control = delivery["control"]
                binding_digest = control["binding_digest"]
                state_path = self.root / STATE_FILENAME
                previous = _read(state_path) if state_path.exists() else None
                deadline = delivery["maximum_execution_deadline_at_ms"]
                if previous:
                    if (previous.get("state_digest") != _digest(previous, "state_digest")
                            or previous.get("session") != self.session or previous.get("binding_digest") != binding_digest
                            or not _epoch_ms(previous.get("execution_deadline_at_ms"))
                            or type(previous.get("revision")) is not int or control["revision"] < previous["revision"]):
                        raise ValueError("beta_screening_runtime_binding_changed")
                    deadline = min(deadline, previous["execution_deadline_at_ms"])
                if self.maximum_deadline_ms is not None:
                    deadline = min(deadline, self.maximum_deadline_ms)
                deadline = min(deadline, control["execution_deadline_at_ms"])
                now = time.time() * 1000
                observed = datetime.fromisoformat(control["observed_at_iso"].replace("Z", "+00:00")).timestamp() * 1000
                if control["status"] == "stopped":
                    raise ValueError("beta_screening_runtime_authority_stopped")
                if now >= deadline:
                    raise ValueError("beta_screening_runtime_authority_expired")
                if observed > now + POLL_SECONDS * 1000 or now - observed >= MONITOR_GRACE_SECONDS * 1000:
                    raise ValueError("beta_screening_runtime_controller_stale")
                state = {"schema_version": "blueprint.beta_screening_runtime_guard_state.v1", "session": self.session,
                         "binding_digest": binding_digest, "execution_deadline_at_ms": deadline, "revision": control["revision"]}
                _write(state_path, {**state, "state_digest": _digest(state, "state_digest")})
                return control
            except (OSError, ValueError, KeyError, TypeError) as exc:
                reason = str(exc) if re.fullmatch("beta_screening_runtime_[a-z_]+", str(exc)) else "beta_screening_runtime_control_unproven"
                raise self._stop(reason, boundary, binding_digest, control) from exc


def begin_runtime_processing(*, runtime_root: Path, inputs: Mapping[str, Any], authority: Mapping[str, Any],
                             boundary: str, on_stop: Callable[[Mapping[str, Any]], None]) -> tuple[RuntimeControlGuard | None, bool]:
    """Share the two worker entry checks while retaining every refusal before return."""
    if inputs.get("run_kind") != BETA_RUN_KIND:
        return None, True
    guard = RuntimeControlGuard(runtime_root=runtime_root, inputs=inputs, authority=authority)
    try:
        guard.check(boundary)
    except RuntimeControlStopped as exc:
        on_stop(exc.receipt)
        return guard, False
    return guard, True


def await_initial_control(*, runtime_root: Path, timeout_seconds: float = MONITOR_GRACE_SECONDS) -> dict[str, Any]:
    """Wait without model/engine work for the existing controller's first delivery."""
    if not 0 <= timeout_seconds <= MONITOR_GRACE_SECONDS:
        raise ValueError("beta_screening_runtime_initial_wait_invalid")
    runtime = Path(runtime_root)
    guard = RuntimeControlGuard(runtime_root=runtime,
        inputs=_read(runtime / "runtime_inputs" / "policy_canary_runtime_inputs.json"),
        authority=_read(runtime / "runtime_inputs" / "policy_canary_session_authority.json"))
    end = time.monotonic() + timeout_seconds
    if guard.maximum_deadline_ms is not None:
        end = min(end, time.monotonic() + max(0, guard.maximum_deadline_ms / 1000 - time.time()))
    while not (guard.root / CONTROL_FILENAME).exists() and not (guard.root / STOP_FILENAME).exists() and time.monotonic() < end:
        time.sleep(min(0.5, max(0, end - time.monotonic())))
    return guard.check("before_policy_provisioning")


if __name__ == "__main__":  # pragma: no cover - exercised through its pure entry point
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    await_initial_control(runtime_root=args.runtime_root)
