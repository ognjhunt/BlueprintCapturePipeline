"""Append-only, digest-bound state for one customer-facing Task Evaluation Run."""

from __future__ import annotations

import fcntl
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest, canonical_json


RUN_STATES = {
    "upload_pending",
    "uploaded",
    "validating",
    "rejected_or_recapture_required",
    "capture_accepted",
    "analyzing",
    "task_candidates_ready",
    "task_approval_required",
    "testbed_compiling",
    "testbed_ready",
    "planning",
    "authorization_required",
    "executing",
    "aggregating",
    "decided",
    "partially_decided",
    "abstained",
    "physical_evidence_requested",
    "outcome_joined",
    "failed",
}
TERMINAL_DECISION_STATES = {"decided", "partially_decided", "abstained"}
_TRANSITIONS = {
    "upload_pending": {"uploaded", "failed"},
    "uploaded": {"validating", "failed"},
    "validating": {"rejected_or_recapture_required", "capture_accepted", "failed"},
    "rejected_or_recapture_required": {"uploaded", "failed"},
    "capture_accepted": {"analyzing", "testbed_compiling", "failed"},
    "analyzing": {"task_candidates_ready", "task_approval_required", "testbed_compiling", "failed"},
    "task_candidates_ready": {"task_approval_required", "testbed_compiling", "failed"},
    "task_approval_required": {"testbed_compiling", "rejected_or_recapture_required", "failed"},
    "testbed_compiling": {"testbed_ready", "rejected_or_recapture_required", "failed"},
    "testbed_ready": {"planning", "failed"},
    "planning": {"authorization_required", "abstained", "failed"},
    "authorization_required": {"executing", "abstained", "failed"},
    "executing": {"aggregating", "failed"},
    "aggregating": TERMINAL_DECISION_STATES | {"failed"},
    "decided": {"physical_evidence_requested", "outcome_joined"},
    "partially_decided": {"physical_evidence_requested", "outcome_joined"},
    "abstained": {"physical_evidence_requested", "outcome_joined"},
    "physical_evidence_requested": {"outcome_joined", "failed"},
    "outcome_joined": set(),
    "failed": set(),
}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class TaskEvaluationRunStateError(ValueError):
    pass


def _identifier(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(text):
        raise TaskEvaluationRunStateError(f"{field}:invalid")
    return text


def _digest(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not _DIGEST.fullmatch(text):
        raise TaskEvaluationRunStateError(f"{field}:invalid")
    return text


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationRunStateError("artifact:not_json_serializable") from exc


def _secret_paths(value: Any, prefix: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if (
                lowered in {"authorization", "credential", "credentials", "password", "secret", "token"}
                or lowered.endswith("_token")
                or lowered.endswith("_secret")
                or lowered.endswith("_password")
            ) and child not in (None, "", [], {}):
                errors.append(path)
            errors.extend(_secret_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_secret_paths(child, f"{prefix}[{index}]"))
    return errors


class TaskEvaluationRunStateStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def _run_root(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def _events(self, run_id: str) -> list[dict[str, Any]]:
        event_root = self._run_root(run_id) / "events"
        if not event_root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(event_root.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def _write_projection(self, run_id: str, event: Mapping[str, Any]) -> None:
        root = self._run_root(run_id)
        projection = {
            "schema_version": "task_evaluation_run_state.v1",
            "run_id": run_id,
            "state": event["to_state"],
            "sequence": event["sequence"],
            "latest_event_digest": event["event_digest"],
            "binding": event["binding"],
            "artifacts": event["artifacts"],
            "proof_boundary": event["proof_boundary"],
        }
        payload = (canonical_json(projection) + "\n").encode("utf-8")
        temporary = root / f".state-{os.getpid()}.tmp"
        temporary.write_bytes(payload)
        os.replace(temporary, root / "state.json")

    def transition(
        self,
        *,
        run_id: str,
        from_state: str | None,
        to_state: str,
        idempotency_key: str,
        actor: Mapping[str, Any],
        binding: Mapping[str, Any],
        artifacts: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = _identifier(run_id, field="run_id")
        key = _identifier(idempotency_key, field="idempotency_key")
        if to_state not in RUN_STATES or (from_state is not None and from_state not in RUN_STATES):
            raise TaskEvaluationRunStateError("run_state:unsupported")
        bound = _clone(dict(binding))
        for field in ("intake_digest", "capture_digest", "testbed_digest", "request_digest"):
            if field in bound and bound[field] is not None:
                _digest(bound[field], field=f"binding.{field}")
        payload_artifacts = _clone(dict(artifacts or {}))
        if _secret_paths({"actor": actor, "binding": bound, "artifacts": payload_artifacts}):
            raise TaskEvaluationRunStateError("run_state:secret_value_forbidden")
        root = self._run_root(run)
        (root / "events").mkdir(parents=True, exist_ok=True)
        with (root / ".lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            events = self._events(run)
            fingerprint = canonical_digest({
                "run_id": run,
                "from_state": from_state,
                "to_state": to_state,
                "actor": dict(actor),
                "binding": bound,
                "artifacts": payload_artifacts,
            })
            replay = next((row for row in events if row.get("idempotency_key") == key), None)
            if replay is not None:
                if replay.get("transition_fingerprint") != fingerprint:
                    raise TaskEvaluationRunStateError("run_state:idempotency_conflict")
                self._write_projection(run, replay)
                return {**replay, "already_exists": True}
            current = events[-1]["to_state"] if events else None
            if current != from_state:
                raise TaskEvaluationRunStateError("run_state:stale_transition")
            if from_state is not None and to_state not in _TRANSITIONS[from_state]:
                raise TaskEvaluationRunStateError("run_state:transition_forbidden")
            sequence = len(events) + 1
            event = {
                "schema_version": "task_evaluation_run_state_event.v1",
                "run_id": run,
                "sequence": sequence,
                "from_state": from_state,
                "to_state": to_state,
                "idempotency_key": key,
                "transition_fingerprint": fingerprint,
                "actor": _clone(dict(actor)),
                "binding": bound,
                "artifacts": payload_artifacts,
                "proof_boundary": {
                    "state_is_scientific_verdict": False,
                    "simulation_is_physical_success": False,
                    "deployment_or_safety_approved": False,
                    "comparative_policy_ranking_verdict": "thesis_not_supported",
                },
            }
            event["event_digest"] = canonical_digest(event, digest_field="event_digest")
            path = root / "events" / f"{sequence:08d}-{event['event_digest'][7:]}.json"
            with path.open("x", encoding="utf-8") as stream:
                stream.write(canonical_json(event) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._write_projection(run, event)
            return {**event, "already_exists": False}

    def inspect(self, run_id: str) -> dict[str, Any]:
        run = _identifier(run_id, field="run_id")
        events = self._events(run)
        if not events:
            raise TaskEvaluationRunStateError("run_state:not_found")
        self._write_projection(run, events[-1])
        return json.loads((self._run_root(run) / "state.json").read_text(encoding="utf-8"))


__all__ = [
    "RUN_STATES",
    "TERMINAL_DECISION_STATES",
    "TaskEvaluationRunStateError",
    "TaskEvaluationRunStateStore",
]
