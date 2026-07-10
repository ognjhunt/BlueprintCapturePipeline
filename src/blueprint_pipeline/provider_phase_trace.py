"""Automatic provider phase and heartbeat trace (startup reliability P1-2).

Long image pulls used to look opaque because only manual 30-45 second
spend-guard snapshots existed. This module persists an atomically rewritten
phase trace on every provider state change and at least every 60 seconds while
a phase is in flight, so postmortems never depend on operator polling.

Every row carries run ID, attempt ID, launch nonce, provider allocation ID,
UTC timestamp, elapsed seconds, and a monotonic sequence number. Externally
supplied callback rows are validated: stale, duplicate, or out-of-order rows
are rejected, never merged. Signed URL query strings and raw provider API
responses are refused at write time.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import utc_now_iso, write_json

SCHEMA_VERSION = "provider_phase_trace.v1"
TRACE_FILENAME = "provider_phase_trace.json"
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60

PHASE_PRE_SPEND_INVENTORY = "pre_spend_inventory"
PHASE_CAPACITY_PROBE = "capacity_probe"
PHASE_ALLOCATION_REQUESTED = "allocation_requested"
PHASE_ALLOCATION_CREATED = "allocation_created"
PHASE_MACHINE_IDENTITY_OBSERVED = "machine_identity_observed"
PHASE_IMAGE_PULL_NO_RUNTIME = "image_pull_no_runtime"
PHASE_RUNTIME_PRESENT = "runtime_present"
PHASE_EARLY_MARKER = "early_marker"
PHASE_BUNDLE_DOWNLOAD = "bundle_download"
PHASE_CUDA_DRIVER_CHECK = "cuda_driver_check"
PHASE_ISAAC_START = "isaac_start"
PHASE_RTX_FRAME = "rtx_frame"
PHASE_RESULT_UPLOAD = "result_upload"
PHASE_STOP = "stop"
PHASE_PROMOTE = "promote"
PHASE_DELETE = "delete"
PHASE_TEARDOWN_VERIFICATION = "teardown_verification"
PHASE_FINAL_INVENTORY = "final_inventory"

KNOWN_PHASES = (
    PHASE_PRE_SPEND_INVENTORY,
    PHASE_CAPACITY_PROBE,
    PHASE_ALLOCATION_REQUESTED,
    PHASE_ALLOCATION_CREATED,
    PHASE_MACHINE_IDENTITY_OBSERVED,
    PHASE_IMAGE_PULL_NO_RUNTIME,
    PHASE_RUNTIME_PRESENT,
    PHASE_EARLY_MARKER,
    PHASE_BUNDLE_DOWNLOAD,
    PHASE_CUDA_DRIVER_CHECK,
    PHASE_ISAAC_START,
    PHASE_RTX_FRAME,
    PHASE_RESULT_UPLOAD,
    PHASE_STOP,
    PHASE_PROMOTE,
    PHASE_DELETE,
    PHASE_TEARDOWN_VERIFICATION,
    PHASE_FINAL_INVENTORY,
)

_FORBIDDEN_DETAIL_KEY_MARKERS = (
    "raw_response",
    "raw_provider",
    "signed_url",
    "api_key",
    "token",
    "secret",
    "authorization",
    "password",
)
_SIGNED_QUERY_MARKERS = (
    "x-amz-",
    "signature=",
    "awsaccesskeyid=",
    "token=",
    "expires=",
)


class PhaseTraceRejected(ValueError):
    """Raised when a row would corrupt the trace or leak sensitive material."""


def _scrub_url_query(text: str) -> str:
    """Drop any query string from URL-ish text: signed URLs must not persist."""
    if "://" in text and "?" in text:
        return text.split("?", 1)[0] + "?<query-stripped>"
    return text


def _sanitize_detail(detail: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if detail is None:
        return None
    sanitized: dict[str, Any] = {}
    for key, value in detail.items():
        key_text = str(key)
        lowered_key = key_text.lower()
        for marker in _FORBIDDEN_DETAIL_KEY_MARKERS:
            if marker in lowered_key:
                raise PhaseTraceRejected(f"phase_trace_forbidden_detail_key:{key_text}")
        if isinstance(value, str):
            lowered = value.lower()
            for marker in _SIGNED_QUERY_MARKERS:
                if marker in lowered:
                    raise PhaseTraceRejected(
                        f"phase_trace_signed_or_secret_value:{key_text}"
                    )
            sanitized[key_text] = _scrub_url_query(value)[:512]
        elif isinstance(value, (int, float, bool)) or value is None:
            sanitized[key_text] = value
        else:
            # Nested payloads are where raw API responses hide; refuse them.
            raise PhaseTraceRejected(f"phase_trace_non_scalar_detail:{key_text}")
    return sanitized


class PhaseTraceRecorder:
    """Append-only in memory, atomically rewritten on disk on every row."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        attempt_id: str,
        launch_nonce: str,
        provider: str | None = None,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not str(run_id or "").strip():
            raise PhaseTraceRejected("phase_trace_run_id_missing")
        if not str(launch_nonce or "").strip():
            raise PhaseTraceRejected("phase_trace_launch_nonce_missing")
        self.path = Path(path)
        self.run_id = str(run_id).strip()
        self.attempt_id = str(attempt_id or "").strip()
        self.launch_nonce = str(launch_nonce).strip()
        self.provider = str(provider or "").strip() or None
        self.heartbeat_interval_seconds = max(1.0, float(heartbeat_interval_seconds))
        self._clock = clock
        self._started = clock()
        self._rows: list[dict[str, Any]] = []
        self._sequence = 0
        self._last_persist_elapsed = 0.0
        self._current_phase: str | None = None
        self._allocation_id: str | None = None

    # -- writing ------------------------------------------------------------

    def record(
        self,
        phase: str,
        *,
        allocation_id: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a state-change row. Unknown phase names are refused."""
        phase_name = str(phase or "").strip()
        if phase_name not in KNOWN_PHASES:
            raise PhaseTraceRejected(f"phase_trace_unknown_phase:{phase_name}")
        if allocation_id is not None:
            self._allocation_id = str(allocation_id).strip() or None
        self._current_phase = phase_name
        return self._append(phase_name, kind="state_change", detail=detail)

    def heartbeat(self, *, detail: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        """Persist a heartbeat row when the interval elapsed; else a no-op.

        Call this from any wait/poll loop; combined with ``record`` on each
        state change it guarantees a persisted row at least every interval.
        """
        if self._current_phase is None:
            return None
        elapsed = self._clock() - self._started
        if elapsed - self._last_persist_elapsed < self.heartbeat_interval_seconds:
            return None
        return self._append(self._current_phase, kind="heartbeat", detail=detail)

    def _append(
        self, phase: str, *, kind: str, detail: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        elapsed = self._clock() - self._started
        self._sequence += 1
        row: dict[str, Any] = {
            "sequence": self._sequence,
            "phase": phase,
            "kind": kind,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "launch_nonce": self.launch_nonce,
            "provider": self.provider,
            "allocation_id": self._allocation_id,
            "utc_timestamp": utc_now_iso(),
            "elapsed_seconds": round(elapsed, 3),
            "detail": _sanitize_detail(detail),
        }
        self._rows.append(row)
        self._last_persist_elapsed = elapsed
        self._persist()
        return row

    def _persist(self) -> None:
        write_json(self.path, self.payload())

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "launch_nonce": self.launch_nonce,
            "provider": self.provider,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "row_count": len(self._rows),
            "rows": list(self._rows),
            "raw_provider_responses_recorded": False,
            "signed_url_query_strings_recorded": False,
            "claim_boundary": (
                "This trace shows provider phase progression and liveness only. "
                "It proves neither render success nor task success, and a "
                "complete trace is not teardown proof."
            ),
        }

    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    # -- external callback ingestion -----------------------------------------

    def ingest_callback_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Accept an externally reported row only if fresh and in order.

        Rejects (never merges) rows whose launch nonce mismatches, whose
        sequence duplicates or precedes what is already recorded, or whose
        phase is unknown. This is the stale/duplicate/out-of-order guard for
        worker-side callbacks.
        """
        nonce = str(row.get("launch_nonce") or "").strip()
        if nonce != self.launch_nonce:
            raise PhaseTraceRejected("phase_trace_callback_stale_nonce")
        phase = str(row.get("phase") or "").strip()
        if phase not in KNOWN_PHASES:
            raise PhaseTraceRejected(f"phase_trace_unknown_phase:{phase}")
        try:
            sequence = int(row.get("sequence"))
        except (TypeError, ValueError):
            raise PhaseTraceRejected("phase_trace_callback_sequence_invalid") from None
        if sequence <= self._sequence:
            raise PhaseTraceRejected("phase_trace_callback_out_of_order_or_duplicate")
        detail = row.get("detail")
        sanitized = _sanitize_detail(detail if isinstance(detail, Mapping) else None)
        self._sequence = sequence - 1
        recorded = self._append(phase, kind="callback", detail=sanitized)
        self._current_phase = phase
        return recorded


def validate_phase_trace(payload: Mapping[str, Any]) -> list[str]:
    """Consumer-side integrity check. Returns blocker slugs (empty = valid)."""
    blockers: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        blockers.append("phase_trace_schema_version_invalid")
    nonce = str(payload.get("launch_nonce") or "").strip()
    if not nonce:
        blockers.append("phase_trace_launch_nonce_missing")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        blockers.append("phase_trace_rows_missing")
        return blockers
    seen_sequences: set[int] = set()
    previous_sequence = 0
    previous_elapsed = -1.0
    for row in rows:
        if not isinstance(row, Mapping):
            blockers.append("phase_trace_row_not_mapping")
            continue
        if str(row.get("launch_nonce") or "") != nonce:
            blockers.append("phase_trace_row_nonce_mismatch")
        phase = str(row.get("phase") or "")
        if phase not in KNOWN_PHASES:
            blockers.append(f"phase_trace_unknown_phase:{phase}")
        try:
            sequence = int(row.get("sequence"))
        except (TypeError, ValueError):
            blockers.append("phase_trace_row_sequence_invalid")
            continue
        if sequence in seen_sequences:
            blockers.append("phase_trace_duplicate_sequence")
        if sequence <= previous_sequence:
            blockers.append("phase_trace_out_of_order_sequence")
        seen_sequences.add(sequence)
        previous_sequence = max(previous_sequence, sequence)
        try:
            elapsed = float(row.get("elapsed_seconds"))
        except (TypeError, ValueError):
            blockers.append("phase_trace_row_elapsed_invalid")
            continue
        if elapsed < previous_elapsed:
            blockers.append("phase_trace_elapsed_regressed")
        previous_elapsed = max(previous_elapsed, elapsed)
    return sorted(set(blockers))


def phase_durations(payload: Mapping[str, Any]) -> dict[str, float]:
    """Per-phase wall-clock durations from consecutive state-change rows."""
    rows = [
        row
        for row in payload.get("rows", [])
        if isinstance(row, Mapping) and row.get("kind") == "state_change"
    ]
    durations: dict[str, float] = {}
    for index, row in enumerate(rows):
        try:
            start = float(row.get("elapsed_seconds"))
        except (TypeError, ValueError):
            continue
        if index + 1 < len(rows):
            try:
                end = float(rows[index + 1].get("elapsed_seconds"))
            except (TypeError, ValueError):
                continue
        else:
            end = start
        phase = str(row.get("phase") or "")
        durations[phase] = round(durations.get(phase, 0.0) + max(0.0, end - start), 3)
    return durations
