"""Crash-safe handoff checkpoints and request-bound Website acknowledgements."""
from __future__ import annotations

import fcntl
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, Mapping

from .decision_evidence_contracts import canonical_digest


class PolicyCanaryHandoffError(ValueError):
    """A handoff cannot safely advance from its retained authority or evidence."""


def _payload(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_write(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise PolicyCanaryHandoffError("policy_canary_handoff_unsafe_state_path")
    payload = _payload(value)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".handoff-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o440)
            if replace:
                os.replace(temporary, path)
            else:
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    if path.is_symlink() or path.read_bytes() != payload:
                        raise PolicyCanaryHandoffError("policy_canary_handoff_immutable_conflict") from None
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


def write_immutable(path: Path, value: Mapping[str, Any]) -> Path:
    _atomic_write(path, value, replace=False)
    return path


def seal_state(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    record = {"schema_version": "task_evaluation_policy_canary_handoff_progression.v1",
              **value, "provider_mutation_performed": False, "progression_digest": ""}
    record["progression_digest"] = canonical_digest(record, digest_field="progression_digest")
    _atomic_write(path, record, replace=True)
    return record


def serialized_handoff(function):
    @wraps(function)
    def run(*, state_root, **kwargs):
        root = Path(state_root).expanduser()
        if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
            raise PolicyCanaryHandoffError("policy_canary_handoff_unsafe_state_path")
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        descriptor = os.open(root / ".policy-canary-handoff.lock", os.O_CREAT | os.O_RDONLY | os.O_NOFOLLOW, 0o440)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise PolicyCanaryHandoffError("policy_canary_handoff_lock_invalid")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            return function(state_root=state_root, **kwargs)
        finally:
            os.close(descriptor)
    return run


def _read(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink():
            raise PolicyCanaryHandoffError("policy_canary_handoff_ack_invalid")
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyCanaryHandoffError("policy_canary_handoff_ack_invalid") from exc
    if not isinstance(value, dict):
        raise PolicyCanaryHandoffError("policy_canary_handoff_ack_invalid")
    return value


def _validate_ack(receipt: Mapping[str, Any], *, run_id: str) -> None:
    run, forward = receipt.get("run"), receipt.get("forward")
    if (receipt.get("schema_version") != "task_evaluation_policy_canary_web_receipt.v1"
            or receipt.get("submission_channel") != "production_webapp_service_api"
            or not isinstance(forward, Mapping) or forward.get("status") != "forwarded"
            or not isinstance(run, Mapping) or run.get("run_id") != run_id
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(run.get("request_digest") or ""))):
        raise PolicyCanaryHandoffError("policy_canary_handoff_webapp_ack_mismatch")


def submit_or_adopt(*, root: Path, endpoint: str, selection: Mapping[str, Any],
        source_commit: str, headers: Callable[[], Mapping[str, str]], poster: Callable) -> tuple[dict[str, Any], str]:
    run_id = str(selection["run_id"])
    binding = {"schema_version": "policy_canary_handoff_web_request_binding.v1",
               "endpoint": endpoint, "run_id": run_id, "source_commit": source_commit,
               "selection_digest": canonical_digest(selection)}
    binding_path = root / "policy-canary-webapp-request-binding.json"
    receipt_path = root / "policy-canary-webapp-receipt.json"
    if receipt_path.exists() and not binding_path.is_file():
        raise PolicyCanaryHandoffError("policy_canary_handoff_ack_binding_missing")
    write_immutable(binding_path, binding)
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _read(receipt_path)
        _validate_ack(receipt, run_id=run_id)
    else:
        body = json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
        status, payload = poster(endpoint=endpoint, headers=headers(), body=body)
        try:
            receipt = json.loads(payload.decode("utf-8")) if payload else {}
        except (UnicodeError, json.JSONDecodeError):
            receipt = {}
        if status != 202 or not isinstance(receipt, Mapping):
            raise PolicyCanaryHandoffError(f"policy_canary_handoff_webapp_rejected:{status}")
        _validate_ack(receipt, run_id=run_id)
        # This also retains a legitimate already_exists response when the first
        # request succeeded remotely but its response never reached this host.
        write_immutable(receipt_path, receipt)
    return dict(receipt), "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()


def verify_completed_ack(root: Path, record: Mapping[str, Any]) -> None:
    selection = _read(root / "policy-canary-selection.json")
    binding = _read(root / "policy-canary-webapp-request-binding.json")
    receipt_path = root / "policy-canary-webapp-receipt.json"
    receipt = _read(receipt_path)
    if (selection.get("run_id") != record.get("run_id")
            or binding.get("run_id") != record.get("run_id")
            or binding.get("source_commit") != record.get("expected_production_commit")
            or binding.get("selection_digest") != canonical_digest(selection)
            or record.get("webapp_receipt_digest") != "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()):
        raise PolicyCanaryHandoffError("policy_canary_handoff_completed_ack_mismatch")
    _validate_ack(receipt, run_id=str(record["run_id"]))
