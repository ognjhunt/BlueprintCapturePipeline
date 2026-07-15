"""Atomic budget reservations for the bounded production GPU campaign.

The campaign has two independent hard ceilings: estimated USD spend and combined
GPU wall time.  Every paid allocation path reserves both before it may call a
provider.  Open reservations retain their full worst-case charge so a crashed
controller cannot silently return budget to the pool.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, cast

from .common import utc_now_iso


SCHEMA_VERSION = "production_gpu_campaign_budget.v1"
AUTHORIZED_SPEND_CAP_USD = 20.0
AUTHORIZED_GPU_WALL_CAP_SECONDS = 16_800
MAX_HOURLY_RATE_USD = 1.99


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"campaign_budget_{field}_invalid")
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        raise ValueError(f"campaign_budget_{field}_invalid") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"campaign_budget_{field}_invalid")
    return number


def _seconds(value: object, *, field: str) -> int:
    number = _number(value, field=field)
    if not number.is_integer():
        raise ValueError(f"campaign_budget_{field}_invalid")
    return int(number)


class CampaignBudgetExceeded(RuntimeError):
    def __init__(self, admission: Mapping[str, Any]):
        self.admission = dict(admission)
        super().__init__(str(self.admission.get("blocker") or "campaign_budget_exceeded"))


class ProductionGpuCampaignBudget:
    """Durable, process-safe dual-cap reservation ledger."""

    def __init__(
        self,
        path: str | Path,
        *,
        initial_spent_usd: float,
        initial_used_gpu_seconds: int,
        total_spend_cap_usd: float = AUTHORIZED_SPEND_CAP_USD,
        combined_gpu_wall_cap_seconds: int = AUTHORIZED_GPU_WALL_CAP_SECONDS,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        # flock provides cross-process exclusion, but its same-process/thread
        # semantics vary across supported Unix kernels. Pair it with an
        # instance lock so concurrent controller threads cannot observe an
        # atomic-replace boundary while another thread owns the ledger.
        self._thread_lock = threading.RLock()
        self.initial_spent_usd = _number(initial_spent_usd, field="initial_spent_usd")
        self.initial_used_gpu_seconds = _seconds(
            initial_used_gpu_seconds, field="initial_used_gpu_seconds"
        )
        self.total_spend_cap_usd = _number(
            total_spend_cap_usd, field="total_spend_cap_usd"
        )
        self.combined_gpu_wall_cap_seconds = _seconds(
            combined_gpu_wall_cap_seconds, field="combined_gpu_wall_cap_seconds"
        )
        if self.total_spend_cap_usd > AUTHORIZED_SPEND_CAP_USD:
            raise ValueError("campaign_budget_spend_cap_exceeds_authorization")
        if self.combined_gpu_wall_cap_seconds > AUTHORIZED_GPU_WALL_CAP_SECONDS:
            raise ValueError("campaign_budget_wall_cap_exceeds_authorization")
        if self.initial_spent_usd > self.total_spend_cap_usd:
            raise ValueError("campaign_budget_initial_spend_exceeds_cap")
        if self.initial_used_gpu_seconds > self.combined_gpu_wall_cap_seconds:
            raise ValueError("campaign_budget_initial_wall_time_exceeds_cap")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if self.path.exists():
                self._validate_identity(self._read())
            else:
                self._write(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "created_at": utc_now_iso(),
                        "total_spend_cap_usd": self.total_spend_cap_usd,
                        "combined_gpu_wall_cap_seconds": self.combined_gpu_wall_cap_seconds,
                        "initial_spent_usd": self.initial_spent_usd,
                        "initial_used_gpu_seconds": self.initial_used_gpu_seconds,
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
                # fdopen owns and closes fd on the normal path.
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _read(self) -> dict[str, Any]:
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("campaign_budget_ledger_missing_or_symlink")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError("campaign_budget_ledger_unreadable") from None
        if not isinstance(value, dict):
            raise ValueError("campaign_budget_ledger_not_object")
        return value

    def _write(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        payload["updated_at"] = utc_now_iso()
        payload.update(self._totals(payload))
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
            os.chmod(self.path, 0o600)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    def _validate_identity(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("campaign_budget_ledger_schema_invalid")
        expected = {
            "total_spend_cap_usd": self.total_spend_cap_usd,
            "combined_gpu_wall_cap_seconds": self.combined_gpu_wall_cap_seconds,
            "initial_spent_usd": self.initial_spent_usd,
            "initial_used_gpu_seconds": self.initial_used_gpu_seconds,
        }
        for field, value in expected.items():
            if state.get(field) != value:
                raise ValueError(f"campaign_budget_ledger_identity_mismatch:{field}")
        reservations = state.get("reservations")
        if not isinstance(reservations, list) or not all(
            isinstance(row, dict) for row in reservations
        ):
            raise ValueError("campaign_budget_reservations_invalid")

    def _totals(self, state: Mapping[str, Any]) -> dict[str, Any]:
        committed_usd = float(state.get("initial_spent_usd") or 0.0)
        committed_seconds = int(state.get("initial_used_gpu_seconds") or 0)
        open_count = 0
        for item in state.get("reservations") or []:
            row = dict(item)
            if row.get("status") == "settled":
                committed_usd += float(row.get("charged_usd") or 0.0)
                committed_seconds += int(row.get("charged_gpu_seconds") or 0)
            else:
                committed_usd += float(row.get("reserved_usd") or 0.0)
                committed_seconds += int(row.get("reserved_gpu_seconds") or 0)
                open_count += 1
        return {
            "committed_usd": round(committed_usd, 6),
            "committed_gpu_seconds": committed_seconds,
            "remaining_usd": round(
                max(0.0, float(state.get("total_spend_cap_usd") or 0.0) - committed_usd),
                6,
            ),
            "remaining_gpu_seconds": max(
                0,
                int(state.get("combined_gpu_wall_cap_seconds") or 0) - committed_seconds,
            ),
            "open_reservation_count": open_count,
        }

    def reserve(
        self,
        *,
        reservation_id: str,
        gpu_seconds: int,
        max_hourly_rate_usd: float,
    ) -> dict[str, Any]:
        key = str(reservation_id or "").strip()
        if len(key) < 8 or len(key) > 160:
            raise ValueError("campaign_budget_reservation_id_invalid")
        seconds = _seconds(gpu_seconds, field="reservation_gpu_seconds")
        rate = _number(max_hourly_rate_usd, field="max_hourly_rate_usd")
        if seconds <= 0:
            raise ValueError("campaign_budget_reservation_seconds_invalid")
        if rate <= 0 or rate > MAX_HOURLY_RATE_USD:
            raise ValueError("campaign_budget_hourly_rate_exceeds_authorization")
        reserved_usd = round(rate * seconds / 3600.0, 6)
        with self._locked():
            state = self._read()
            self._validate_identity(state)
            for row in state["reservations"]:
                if row.get("reservation_id") == key:
                    if row.get("status") != "open":
                        raise ValueError("campaign_budget_reservation_id_already_settled")
                    if (
                        row.get("reserved_gpu_seconds") != seconds
                        or row.get("max_hourly_rate_usd") != rate
                    ):
                        raise ValueError("campaign_budget_reservation_id_conflict")
                    return dict(row)
            totals = self._totals(state)
            admission = {
                "reservation_id": key,
                "requested_gpu_seconds": seconds,
                "requested_reservation_usd": reserved_usd,
                "committed_gpu_seconds_before": totals["committed_gpu_seconds"],
                "committed_usd_before": totals["committed_usd"],
                "gpu_wall_cap_seconds": self.combined_gpu_wall_cap_seconds,
                "total_spend_cap_usd": self.total_spend_cap_usd,
                "admitted": True,
            }
            if totals["committed_gpu_seconds"] + seconds > self.combined_gpu_wall_cap_seconds:
                admission.update(admitted=False, blocker="campaign_gpu_wall_time_cap_exceeded")
                raise CampaignBudgetExceeded(admission)
            if totals["committed_usd"] + reserved_usd > self.total_spend_cap_usd:
                admission.update(admitted=False, blocker="campaign_total_spend_cap_exceeded")
                raise CampaignBudgetExceeded(admission)
            row = {
                "reservation_id": key,
                "status": "open",
                "reserved_gpu_seconds": seconds,
                "max_hourly_rate_usd": rate,
                "reserved_usd": reserved_usd,
                "admitted_at": utc_now_iso(),
            }
            state["reservations"].append(row)
            self._write(state)
            return dict(row)

    def settle(
        self,
        *,
        reservation_id: str,
        charged_gpu_seconds: int,
        charged_usd: float,
        outcome: str,
    ) -> dict[str, Any]:
        key = str(reservation_id or "").strip()
        seconds = _seconds(charged_gpu_seconds, field="charged_gpu_seconds")
        usd = _number(charged_usd, field="charged_usd")
        with self._locked():
            state = self._read()
            self._validate_identity(state)
            for row in state["reservations"]:
                if row.get("reservation_id") != key:
                    continue
                if row.get("status") == "settled":
                    if row.get("charged_gpu_seconds") != seconds or row.get("charged_usd") != usd:
                        raise ValueError("campaign_budget_settlement_conflict")
                    return dict(row)
                if seconds > int(row.get("reserved_gpu_seconds") or 0):
                    raise ValueError("campaign_budget_settlement_exceeds_reserved_seconds")
                if usd > float(row.get("reserved_usd") or 0.0):
                    raise ValueError("campaign_budget_settlement_exceeds_reserved_usd")
                row.update(
                    status="settled",
                    charged_gpu_seconds=seconds,
                    charged_usd=round(usd, 6),
                    outcome=str(outcome or "unknown")[:160],
                    settled_at=utc_now_iso(),
                )
                self._write(state)
                return dict(row)
        raise ValueError(f"campaign_budget_reservation_not_found:{key}")

    def snapshot(self) -> dict[str, Any]:
        with self._locked():
            state = self._read()
            self._validate_identity(state)
            return {**state, **self._totals(state)}
