"""Append-only, hash-chained event storage for supervisor runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..decision_evidence_contracts import canonical_json
from .contracts import SupervisorContractError, SupervisorEvent


class SupervisorLedgerError(RuntimeError):
    pass


class AppendOnlyEventLedger:
    """Durably append validated events and fail closed on partial/corrupt chains."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> tuple[SupervisorEvent, ...]:
        if not self.path.exists():
            return ()
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise SupervisorLedgerError("supervisor_event_ledger_partial_record")
        events: list[SupervisorEvent] = []
        for index, line in enumerate(raw.decode("utf-8").splitlines()):
            if not line.strip():
                raise SupervisorLedgerError(f"supervisor_event_ledger_blank_record:{index}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SupervisorLedgerError(
                    f"supervisor_event_ledger_invalid_json:{index}"
                ) from exc
            if not isinstance(value, Mapping):
                raise SupervisorLedgerError(f"supervisor_event_ledger_non_mapping:{index}")
            try:
                event = SupervisorEvent.from_mapping(value)
            except SupervisorContractError as exc:
                raise SupervisorLedgerError(
                    f"supervisor_event_ledger_invalid_event:{index}:{exc}"
                ) from exc
            events.append(event)
        self._validate_chain(events)
        return tuple(events)

    @staticmethod
    def _validate_chain(events: Sequence[SupervisorEvent]) -> None:
        previous_digest: str | None = None
        run_id: str | None = None
        for index, event in enumerate(events):
            value = event.to_mapping()
            if value["sequence"] != index:
                raise SupervisorLedgerError(f"supervisor_event_ledger_sequence_mismatch:{index}")
            if value.get("previous_event_digest") != previous_digest:
                raise SupervisorLedgerError(f"supervisor_event_ledger_chain_mismatch:{index}")
            if run_id is None:
                run_id = str(value["run_id"])
            elif value["run_id"] != run_id:
                raise SupervisorLedgerError(f"supervisor_event_ledger_run_mismatch:{index}")
            previous_digest = event.digest

    def append(self, event_value: Mapping[str, Any]) -> SupervisorEvent:
        event = SupervisorEvent.from_mapping(event_value)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # The repository targets POSIX/macOS. Lock when available so two
        # supervisor processes cannot both append the same sequence number.
        with self.path.open("a+", encoding="utf-8", newline="") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - non-POSIX compatibility
                fcntl = None  # type: ignore[assignment]
            try:
                handle.seek(0)
                raw = handle.read()
                if raw and not raw.endswith("\n"):
                    raise SupervisorLedgerError("supervisor_event_ledger_partial_record")
                existing: list[SupervisorEvent] = []
                for index, line in enumerate(raw.splitlines()):
                    try:
                        parsed = json.loads(line)
                        existing.append(SupervisorEvent.from_mapping(parsed))
                    except (json.JSONDecodeError, SupervisorContractError) as exc:
                        raise SupervisorLedgerError(
                            f"supervisor_event_ledger_invalid_event:{index}"
                        ) from exc
                self._validate_chain(existing)
                value = event.to_mapping()
                expected_sequence = len(existing)
                expected_previous = existing[-1].digest if existing else None
                if value["sequence"] != expected_sequence:
                    raise SupervisorLedgerError("supervisor_event_append_sequence_mismatch")
                if value.get("previous_event_digest") != expected_previous:
                    raise SupervisorLedgerError("supervisor_event_append_chain_mismatch")
                if existing and value["run_id"] != existing[0].to_mapping()["run_id"]:
                    raise SupervisorLedgerError("supervisor_event_append_run_mismatch")
                handle.seek(0, os.SEEK_END)
                handle.write(canonical_json(value) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                if fcntl is not None:  # type: ignore[possibly-undefined]
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return event


__all__ = ["AppendOnlyEventLedger", "SupervisorLedgerError"]
