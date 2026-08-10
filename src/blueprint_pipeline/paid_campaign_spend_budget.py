"""Atomic, evidence-bound USD budgets shared by paid workflow lanes.

Per-attempt spend caps do not prevent two independent controllers from each
spending the same remaining campaign budget.  This ledger is deliberately
provider- and task-neutral: every paid lane reserves its worst-case USD charge
under one campaign identity before any provider mutation, then settles to the
retained cost after teardown.  An open reservation keeps its full charge so a
crashed controller fails closed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, cast

from .common import utc_now_iso


SCHEMA_VERSION = "paid_campaign_spend_budget.v1"
_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}


def _money(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"paid_campaign_budget_{field}_invalid")
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        raise ValueError(f"paid_campaign_budget_{field}_invalid") from None
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        raise ValueError(f"paid_campaign_budget_{field}_invalid")
    return round(number, 6)


def _identifier(value: object, *, field: str) -> str:
    identifier = str(value or "").strip()
    if not 3 <= len(identifier) <= 200 or any(ord(char) < 32 for char in identifier):
        raise ValueError(f"paid_campaign_budget_{field}_invalid")
    return identifier


def _path_lock(path: Path) -> threading.RLock:
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(path, threading.RLock())


class PaidCampaignBudgetExceeded(RuntimeError):
    """A worst-case reservation would cross the campaign's authorized cap."""

    def __init__(self, admission: Mapping[str, Any]):
        self.admission = dict(admission)
        super().__init__(
            str(self.admission.get("blocker") or "paid_campaign_budget_exceeded")
        )


class PaidCampaignSpendBudget:
    """Durable, process-safe money-only campaign budget.

    ``initial_spent_usd`` may already exceed the bound cap.  That state is
    retained as an explicit overrun and every new reservation is rejected;
    refusing to materialize the ledger would hide the most important fact.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        campaign_id: str,
        authority_id: str,
        initial_spent_usd: float,
        total_spend_cap_usd: float,
        initial_spend_basis: str,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._thread_lock = _path_lock(self.path)
        self.campaign_id = _identifier(campaign_id, field="campaign_id")
        self.authority_id = _identifier(authority_id, field="authority_id")
        self.initial_spend_basis = _identifier(
            initial_spend_basis, field="initial_spend_basis"
        )
        self.initial_spent_usd = _money(
            initial_spent_usd, field="initial_spent_usd"
        )
        self.total_spend_cap_usd = _money(
            total_spend_cap_usd, field="total_spend_cap_usd", positive=True
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if self.path.exists():
                self._validate_identity(self._read())
            else:
                self._write(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "created_at": utc_now_iso(),
                        "campaign_id": self.campaign_id,
                        "authority_id": self.authority_id,
                        "initial_spent_usd": self.initial_spent_usd,
                        "initial_spend_basis": self.initial_spend_basis,
                        "total_spend_cap_usd": self.total_spend_cap_usd,
                        "reservations": [],
                    }
                )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "r+") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    yield
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _read(self) -> dict[str, Any]:
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("paid_campaign_budget_ledger_missing_or_symlink")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError("paid_campaign_budget_ledger_unreadable") from None
        if not isinstance(value, dict):
            raise ValueError("paid_campaign_budget_ledger_not_object")
        return value

    def _write(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        payload["updated_at"] = utc_now_iso()
        payload.update(self._totals(payload))
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _validate_identity(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "authority_id": self.authority_id,
            "initial_spent_usd": self.initial_spent_usd,
            "initial_spend_basis": self.initial_spend_basis,
            "total_spend_cap_usd": self.total_spend_cap_usd,
        }
        for field, value in expected.items():
            if state.get(field) != value:
                raise ValueError(f"paid_campaign_budget_ledger_identity_mismatch:{field}")
        reservations = state.get("reservations")
        if not isinstance(reservations, list):
            raise ValueError("paid_campaign_budget_reservations_invalid")
        seen_ids: set[str] = set()
        for row in reservations:
            if not isinstance(row, dict) or row.get("status") not in {
                "open",
                "settled",
            }:
                raise ValueError("paid_campaign_budget_reservations_invalid")
            try:
                reservation_id = _identifier(
                    row.get("reservation_id"), field="reservation_id"
                )
                _identifier(
                    row.get("reservation_owner_id"),
                    field="reservation_owner_id",
                )
                reserved = _money(
                    row.get("reserved_usd"),
                    field="reservation_max_spend_usd",
                    positive=True,
                )
                if row.get("status") == "settled":
                    charged = _money(row.get("charged_usd"), field="charged_usd")
                    _identifier(row.get("cost_basis"), field="cost_basis")
                    if charged > reserved:
                        raise ValueError(
                            "paid_campaign_budget_settlement_exceeds_reservation"
                        )
            except ValueError:
                raise ValueError("paid_campaign_budget_reservations_invalid") from None
            if reservation_id in seen_ids:
                raise ValueError("paid_campaign_budget_reservations_invalid")
            seen_ids.add(reservation_id)

    def _totals(self, state: Mapping[str, Any]) -> dict[str, Any]:
        committed = float(state.get("initial_spent_usd") or 0.0)
        open_count = 0
        for raw in state.get("reservations") or []:
            row = dict(raw)
            if row.get("status") == "settled":
                committed += float(row.get("charged_usd") or 0.0)
            else:
                committed += float(row.get("reserved_usd") or 0.0)
                open_count += 1
        cap = float(state.get("total_spend_cap_usd") or 0.0)
        committed = round(committed, 6)
        return {
            "committed_usd": committed,
            "remaining_usd": round(max(0.0, cap - committed), 6),
            "cap_overrun_usd": round(max(0.0, committed - cap), 6),
            "open_reservation_count": open_count,
            "budget_status": "exhausted" if committed >= cap else "available",
        }

    def _admission(
        self,
        state: Mapping[str, Any],
        *,
        reservation_id: str,
        max_spend_usd: float,
    ) -> dict[str, Any]:
        key = _identifier(reservation_id, field="reservation_id")
        requested = _money(
            max_spend_usd, field="reservation_max_spend_usd", positive=True
        )
        totals = self._totals(state)
        admission = {
            "schema_version": "paid_campaign_spend_admission.v1",
            "campaign_id": self.campaign_id,
            "authority_id": self.authority_id,
            "reservation_id": key,
            "requested_max_spend_usd": requested,
            "committed_usd_before": totals["committed_usd"],
            "remaining_usd_before": totals["remaining_usd"],
            "cap_overrun_usd_before": totals["cap_overrun_usd"],
            "total_spend_cap_usd": self.total_spend_cap_usd,
            "admitted": True,
            "blocker": None,
        }
        if totals["committed_usd"] + requested > self.total_spend_cap_usd:
            admission.update(
                admitted=False,
                blocker="paid_campaign_total_spend_cap_exceeded",
            )
        return admission

    def preview(
        self, *, reservation_id: str, max_spend_usd: float
    ) -> dict[str, Any]:
        """Read-only admission preview; execute must still reserve atomically."""

        with self._locked():
            state = self._read()
            self._validate_identity(state)
            return self._admission(
                state,
                reservation_id=reservation_id,
                max_spend_usd=max_spend_usd,
            )

    def reserve(
        self,
        *,
        reservation_id: str,
        reservation_owner_id: str,
        max_spend_usd: float,
    ) -> dict[str, Any]:
        key = _identifier(reservation_id, field="reservation_id")
        owner = _identifier(reservation_owner_id, field="reservation_owner_id")
        requested = _money(
            max_spend_usd, field="reservation_max_spend_usd", positive=True
        )
        with self._locked():
            state = self._read()
            self._validate_identity(state)
            for row in state["reservations"]:
                if row.get("reservation_id") != key:
                    continue
                if row.get("status") != "open":
                    raise ValueError("paid_campaign_budget_reservation_already_settled")
                if row.get("reservation_owner_id") != owner:
                    raise ValueError(
                        "paid_campaign_budget_reservation_owned_by_another_controller"
                    )
                if row.get("reserved_usd") != requested:
                    raise ValueError("paid_campaign_budget_reservation_conflict")
                return dict(row)
            admission = self._admission(
                state, reservation_id=key, max_spend_usd=requested
            )
            if admission["admitted"] is not True:
                raise PaidCampaignBudgetExceeded(admission)
            row = {
                "reservation_id": key,
                "reservation_owner_id": owner,
                "status": "open",
                "reserved_usd": requested,
                "admitted_at": utc_now_iso(),
            }
            state["reservations"].append(row)
            self._write(state)
            return dict(row)

    def settle(
        self,
        *,
        reservation_id: str,
        reservation_owner_id: str,
        charged_usd: float,
        cost_basis: str,
        outcome: str,
    ) -> dict[str, Any]:
        key = _identifier(reservation_id, field="reservation_id")
        owner = _identifier(reservation_owner_id, field="reservation_owner_id")
        charged = _money(charged_usd, field="charged_usd")
        basis = _identifier(cost_basis, field="cost_basis")
        with self._locked():
            state = self._read()
            self._validate_identity(state)
            for row in state["reservations"]:
                if row.get("reservation_id") != key:
                    continue
                if row.get("reservation_owner_id") != owner:
                    raise ValueError(
                        "paid_campaign_budget_reservation_owned_by_another_controller"
                    )
                if row.get("status") == "settled":
                    if row.get("charged_usd") != charged:
                        raise ValueError("paid_campaign_budget_settlement_conflict")
                    return dict(row)
                if charged > float(row.get("reserved_usd") or 0.0):
                    raise ValueError("paid_campaign_budget_settlement_exceeds_reservation")
                row.update(
                    status="settled",
                    charged_usd=charged,
                    cost_basis=basis,
                    outcome=str(outcome or "unknown")[:200],
                    settled_at=utc_now_iso(),
                )
                self._write(state)
                return dict(row)
        raise ValueError(f"paid_campaign_budget_reservation_not_found:{key}")

    def snapshot(self) -> dict[str, Any]:
        with self._locked():
            state = self._read()
            self._validate_identity(state)
            snapshot = {**state, **self._totals(state), "snapshot_digest": ""}
            encoded = json.dumps(
                snapshot, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            snapshot["snapshot_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
            return snapshot


__all__ = [
    "PaidCampaignBudgetExceeded",
    "PaidCampaignSpendBudget",
    "SCHEMA_VERSION",
]
