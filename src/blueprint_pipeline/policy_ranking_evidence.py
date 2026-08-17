"""Crash-safe, append-only evidence storage for policy-ranking provider runs.

The journal is a directory of immutable, hash-chained JSON events.  A directory
journal avoids the torn-final-line ambiguity of JSONL while retaining strict
append-only semantics.  Human-friendly aggregates and manifests are derived
from the journal and may always be rebuilt.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .policy_ranking_thesis import canonical_sha256, file_sha256


SCHEMA = "policy_ranking_evidence_store.v2"
EVENT_SCHEMA = "policy_ranking_evidence_event.v2"
AGGREGATE_SCHEMA = "policy_ranking_evidence_aggregate.v2"
MANIFEST_SCHEMA = "policy_ranking_evidence_manifest.v2"
_SECRET_KEY = re.compile(
    r"(^|_)(authorization|api_?key|credential|password|private_?key|"
    r"access_?token|refresh_?token|client_?secret|signed_?url)($|_)",
    re.I,
)
_SECRET_VALUE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{20,}"
    # Google's newer API-key shape (`AQ.` + ~49 chars).  The leading boundary
    # keeps prose such as "FAQ.Something" out; 30 is far below real key length.
    r"|\bAQ\.[A-Za-z0-9_-]{30,}"
    r"|Bearer\s+[A-Za-z0-9._~+/=-]{12,})"
)


class EvidenceError(RuntimeError):
    """Base evidence-integrity failure."""


class InventoryMismatchError(EvidenceError):
    """Raised when a journal is opened with a different frozen inventory."""


class JournalIntegrityError(EvidenceError):
    """Raised when an immutable event or its hash chain is invalid."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe(value: Any, *, key: str = "") -> Any:
    if key and _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): _safe(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED]", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, value: Mapping[str, Any]) -> bool:
    """Atomically materialize an immutable file; never replace an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_replace(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


class EvidenceStore:
    """Immutable provider-event journal bound to one experiment and inventory."""

    def __init__(
        self,
        root: str | Path,
        *,
        experiment_id: str,
        inventory_sha256: str,
        configuration_sha256: str,
    ) -> None:
        self.root = Path(root).resolve()
        self.journal_dir = self.root / "journal"
        self.requests_dir = self.root / "requests"
        self.lock_path = self.root / ".journal.lock"
        self.metadata_path = self.root / "store_identity.json"
        self.aggregate_path = self.root / "derived_aggregate.json"
        self.manifest_path = self.root / "evidence_manifest.json"
        self.experiment_id = experiment_id
        self.inventory_sha256 = inventory_sha256
        self.configuration_sha256 = configuration_sha256
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal_dir.mkdir(exist_ok=True)
        self.requests_dir.mkdir(exist_ok=True)
        identity = {
            "schema_version": SCHEMA,
            "experiment_id": experiment_id,
            "inventory_sha256": inventory_sha256,
            "configuration_sha256": configuration_sha256,
        }
        if not _write_once(self.metadata_path, identity):
            existing = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if existing != identity:
                raise InventoryMismatchError("evidence_store_identity_mismatch")
        self.verify()
        if not self.events():
            self.append(
                "experiment_initialized",
                {
                    "provider_called": False,
                    "consumed_infrastructure_retry": False,
                    "consumed_scientific_response": False,
                },
            )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _event_paths(self) -> list[Path]:
        return sorted(self.journal_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-*.json"))

    def events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        previous = "0" * 64
        for expected_sequence, path in enumerate(self._event_paths(), start=1):
            try:
                event = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise JournalIntegrityError(f"event_unreadable:{path.name}") from exc
            event_hash = str(event.get("event_sha256") or "")
            body = dict(event)
            body.pop("event_sha256", None)
            if event.get("sequence") != expected_sequence:
                raise JournalIntegrityError(f"event_sequence_invalid:{path.name}")
            if event.get("previous_event_sha256") != previous:
                raise JournalIntegrityError(f"event_chain_invalid:{path.name}")
            if canonical_sha256(body) != event_hash:
                raise JournalIntegrityError(f"event_digest_invalid:{path.name}")
            expected_name = f"{expected_sequence:08d}-{event_hash}.json"
            if path.name != expected_name:
                raise JournalIntegrityError(f"event_filename_invalid:{path.name}")
            events.append(event)
            previous = event_hash
        return events

    def verify(self) -> None:
        self.events()

    def _append_locked(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        current = self.events()
        sequence = len(current) + 1
        event = {
            "schema_version": EVENT_SCHEMA,
            "experiment_id": self.experiment_id,
            "inventory_sha256": self.inventory_sha256,
            "configuration_sha256": self.configuration_sha256,
            "sequence": sequence,
            "previous_event_sha256": current[-1]["event_sha256"] if current else "0" * 64,
            "event_type": event_type,
            "recorded_at": utc_now(),
            "payload": _safe(payload),
        }
        event["event_sha256"] = canonical_sha256(event)
        path = self.journal_dir / f"{sequence:08d}-{event['event_sha256']}.json"
        if not _write_once(path, event):
            raise JournalIntegrityError(f"event_path_collision:{path.name}")
        return event

    def append(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._locked():
            event = self._append_locked(event_type, payload)
        self.rebuild()
        return event

    def state(self) -> dict[str, Any]:
        accepted: dict[str, dict[str, Any]] = {}
        claims: dict[str, dict[str, Any]] = {}
        attempts: dict[str, list[dict[str, Any]]] = {}
        preflight: list[dict[str, Any]] = []
        for event in self.events():
            payload = event["payload"]
            request_id = str(payload.get("request_id") or "")
            if event["event_type"] == "request_claimed":
                claims[request_id] = event
            elif event["event_type"] in {
                "response_accepted",
                "attempt_failed",
                "duplicate_completion_ignored",
            }:
                attempts.setdefault(request_id, []).append(event)
                claims.pop(request_id, None)
                if event["event_type"] == "response_accepted":
                    accepted.setdefault(request_id, event)
            elif event["event_type"] == "preflight_failed":
                preflight.append(event)
        return {
            "accepted": accepted,
            "claims": claims,
            "attempts": attempts,
            "preflight": preflight,
        }

    def claim(
        self,
        request: Mapping[str, Any],
        *,
        arm_id: str,
        provider: str,
        model_snapshot: str,
        attempt_type: str,
        lease_seconds: float = 300.0,
    ) -> str | None:
        request_id = str(request["request_id"])
        now = time.time()
        with self._locked():
            state = self.state()
            if request_id in state["accepted"]:
                return None
            prior = state["claims"].get(request_id)
            if prior and float(prior["payload"].get("lease_expires_unix") or 0.0) > now:
                return None
            claim_id = uuid.uuid4().hex
            self._append_locked(
                "request_claimed",
                {
                    "request_id": request_id,
                    "deterministic_input_hash": request.get("deterministic_input_hash")
                    or canonical_sha256(dict(request)),
                    "session_id": request.get("session_id"),
                    "policy_id": request.get("policy_id"),
                    "task_id": request.get("task_id") or request.get("task_instruction"),
                    "arm_id": arm_id,
                    "attempt_type": attempt_type,
                    "provider": provider,
                    "provider_called": False,
                    "model_snapshot": model_snapshot,
                    "configuration_hash": self.configuration_sha256,
                    "claim_id": claim_id,
                    "request_start_timestamp": utc_now(),
                    "lease_expires_unix": now + lease_seconds,
                    "consumed_infrastructure_retry": False,
                    "consumed_scientific_response": False,
                    "accepted_first_valid": False,
                },
            )
        self.rebuild()
        return claim_id

    def complete(
        self,
        *,
        request: Mapping[str, Any],
        claim_id: str,
        arm_id: str,
        attempt_type: str,
        provider: str,
        model_snapshot: str,
        started_at: str,
        elapsed_seconds: float,
        structured_response: Mapping[str, Any] | None,
        validation_result: str,
        usage: Mapping[str, Any] | None,
        estimated_cost_usd: float,
        actual_cost_usd: float | None,
        response_id: str = "",
        provider_error_category: str = "",
        retry_after_seconds: float | None = None,
        reset_metadata: Mapping[str, Any] | None = None,
        consumed_infrastructure_retry: bool = False,
        consumed_scientific_response: bool = False,
    ) -> dict[str, Any]:
        request_id = str(request["request_id"])
        valid = validation_result == "valid" and structured_response is not None
        with self._locked():
            state = self.state()
            if request_id in state["accepted"]:
                event_type = "duplicate_completion_ignored"
            else:
                claim = state["claims"].get(request_id)
                if not claim or claim["payload"].get("claim_id") != claim_id:
                    raise EvidenceError(f"claim_mismatch:{request_id}")
                event_type = "response_accepted" if valid else "attempt_failed"
            payload = {
                "request_id": request_id,
                "deterministic_input_hash": request.get("deterministic_input_hash")
                or canonical_sha256(dict(request)),
                "session_id": request.get("session_id"),
                "policy_id": request.get("policy_id"),
                "task_id": request.get("task_id") or request.get("task_instruction"),
                "arm_id": arm_id,
                "attempt_type": attempt_type,
                "provider": provider,
                "provider_called": True,
                "model_snapshot": model_snapshot,
                "configuration_hash": self.configuration_sha256,
                "claim_id": claim_id,
                "request_start_timestamp": started_at,
                "request_end_timestamp": utc_now(),
                "elapsed_seconds": elapsed_seconds,
                "response_id": response_id,
                "structured_response": structured_response,
                "validation_result": validation_result,
                "usage": usage or {},
                "estimated_cost_usd": estimated_cost_usd,
                "actual_cost_usd": actual_cost_usd,
                "provider_error_category": provider_error_category,
                "retry_after_seconds": retry_after_seconds,
                "reset_metadata": reset_metadata or {},
                "consumed_infrastructure_retry": consumed_infrastructure_retry,
                "consumed_scientific_response": consumed_scientific_response,
                "accepted_first_valid": event_type == "response_accepted",
            }
            event = self._append_locked(event_type, payload)
        request_path = (
            self.requests_dir
            / request_id
            / (f"{event['sequence']:08d}-{event['event_sha256']}.json")
        )
        _write_once(request_path, event)
        self.rebuild()
        return event

    def mark_provider_call_started(
        self,
        *,
        request: Mapping[str, Any],
        claim_id: str,
        arm_id: str,
        attempt_type: str,
        provider: str,
        model_snapshot: str,
        started_at: str,
    ) -> dict[str, Any]:
        request_id = str(request["request_id"])
        with self._locked():
            state = self.state()
            if request_id in state["accepted"]:
                raise EvidenceError(f"provider_call_after_acceptance:{request_id}")
            claim = state["claims"].get(request_id)
            if not claim or claim["payload"].get("claim_id") != claim_id:
                raise EvidenceError(f"claim_mismatch:{request_id}")
            event = self._append_locked(
                "provider_call_started",
                {
                    "request_id": request_id,
                    "deterministic_input_hash": request.get("deterministic_input_hash")
                    or canonical_sha256(dict(request)),
                    "session_id": request.get("session_id"),
                    "policy_id": request.get("policy_id"),
                    "task_id": request.get("task_id") or request.get("task_instruction"),
                    "arm_id": arm_id,
                    "attempt_type": attempt_type,
                    "provider": provider,
                    "provider_called": True,
                    "model_snapshot": model_snapshot,
                    "configuration_hash": self.configuration_sha256,
                    "claim_id": claim_id,
                    "request_start_timestamp": started_at,
                    "consumed_infrastructure_retry": False,
                    "consumed_scientific_response": False,
                    "accepted_first_valid": False,
                },
            )
        self.rebuild()
        return event

    def record_preflight_failure(self, blocker: str, *, provider: str) -> dict[str, Any]:
        return self.append(
            "preflight_failed",
            {
                "blocker": blocker,
                "provider": provider,
                "provider_called": False,
                "consumed_infrastructure_retry": False,
                "consumed_scientific_response": False,
                "accepted_first_valid": False,
            },
        )

    def rebuild(self) -> dict[str, Any]:
        events = self.events()
        state = self.state()
        for event in events:
            request_id = str(event["payload"].get("request_id") or "")
            if request_id:
                _write_once(
                    self.requests_dir
                    / request_id
                    / f"{event['sequence']:08d}-{event['event_sha256']}.json",
                    event,
                )
        cost_estimated = sum(
            float(event["payload"].get("estimated_cost_usd") or 0.0)
            for event in events
            if event["event_type"] in {"response_accepted", "attempt_failed"}
        )
        actual_values = [
            float(event["payload"]["actual_cost_usd"])
            for event in events
            if event["event_type"] in {"response_accepted", "attempt_failed"}
            and event["payload"].get("actual_cost_usd") is not None
        ]
        aggregate = {
            "schema_version": AGGREGATE_SCHEMA,
            "experiment_id": self.experiment_id,
            "inventory_sha256": self.inventory_sha256,
            "configuration_sha256": self.configuration_sha256,
            "event_count": len(events),
            "accepted_request_count": len(state["accepted"]),
            "accepted_request_ids": sorted(state["accepted"]),
            "failed_attempt_count": sum(
                event["event_type"] == "attempt_failed" for event in events
            ),
            "preflight_failure_count": len(state["preflight"]),
            "provider_called": any(
                bool(event["payload"].get("provider_called")) for event in events
            ),
            "estimated_cost_usd_recomputed": cost_estimated,
            "actual_cost_usd_recomputed": sum(actual_values) if actual_values else None,
            "last_event_sha256": events[-1]["event_sha256"] if events else "0" * 64,
            "events": events,
        }
        aggregate["aggregate_sha256"] = canonical_sha256(aggregate)
        _atomic_replace(self.aggregate_path, aggregate)
        manifest_files = [self.metadata_path, *self._event_paths()]
        manifest_files.extend(sorted(self.requests_dir.glob("*/*.json")))
        manifest_files.append(self.aggregate_path)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "experiment_id": self.experiment_id,
            "files": [
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in manifest_files
            ],
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _atomic_replace(self.manifest_path, manifest)
        return aggregate

    def verify_manifest(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        for row in manifest["files"]:
            path = self.root / row["path"]
            if not path.is_file() or file_sha256(path) != row["sha256"]:
                raise JournalIntegrityError(f"manifest_file_mismatch:{row['path']}")
        body = dict(manifest)
        digest = body.pop("manifest_sha256")
        if canonical_sha256(body) != digest:
            raise JournalIntegrityError("manifest_digest_mismatch")


def scan_for_secrets(root: str | Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(Path(root).rglob("*.json")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if _SECRET_VALUE.search(text):
            findings.append(path.as_posix())
    return findings
