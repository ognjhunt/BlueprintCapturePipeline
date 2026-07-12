"""Startup telemetry and spend reconciliation (startup reliability P1-3).

Spend admission was already bounded and teardown proven, but final inventories
reported billing reconciliation as ``not_configured`` and nothing separated a
conservative estimate from an invoice. This module makes the distinction
explicit and non-fudgeable:

- ``reserved_worst_case_usd``: what admission set aside before launch;
- ``elapsed_rate_upper_bound_usd``: provider rate x allocation age — an upper
  bound, never an invoice;
- ``provider_reported_actual_usd``: only ever populated from an authoritative
  provider billing API (source ``provider_billing_api``);
- ``standing_stopped_disk_usd_per_hour``: residual stopped-disk/volume burn;
- ``billing_reconciliation``: ``provider_api`` or ``not_configured`` — an
  estimate is never labeled actual.

The goal-level :class:`CumulativeSpendLedger` includes failed and successful
attempts and enforces the user's total cap before each new allocation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .common import utc_now_iso, write_json

SCHEMA_VERSION = "startup_spend_reconciliation.v1"
LEDGER_SCHEMA_VERSION = "startup_cumulative_spend_ledger.v1"
PROVIDER_BILLING_API_SOURCE = "provider_billing_api"
BILLING_RECONCILIATION_PROVIDER_API = "provider_api"
BILLING_RECONCILIATION_NOT_CONFIGURED = "not_configured"


def _non_negative(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"spend_reconciliation_{field}_invalid") from None
    if not math.isfinite(number):
        raise ValueError(f"spend_reconciliation_{field}_nonfinite")
    if number < 0:
        raise ValueError(f"spend_reconciliation_{field}_negative")
    return number


def build_spend_reconciliation(
    *,
    provider: str,
    hourly_rate_usd: float,
    reserved_seconds: float,
    elapsed_seconds: float,
    provider_reported_actual_usd: float | None = None,
    provider_reported_source: str | None = None,
    stopped_disk_usd_per_hour: float | None = None,
    stopped_disk_seconds: float = 0.0,
    container_disk_usd_per_hour: float = 0.0,
    persistent_volume_usd_per_hour: float = 0.0,
    network_volume_usd_per_hour: float = 0.0,
    container_disk_seconds: float | None = None,
    persistent_volume_seconds: float | None = None,
    network_volume_seconds: float | None = None,
    phase_durations_seconds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """One attempt's spend picture with estimates and actuals kept apart.

    ``provider_reported_actual_usd`` is dropped (with an explicit refusal note)
    unless ``provider_reported_source`` is the authoritative
    ``provider_billing_api`` — a rate-times-age computation passed in as
    "actual" is a labeling violation, not a reconciliation.
    """
    rate = _non_negative(hourly_rate_usd, field="hourly_rate_usd")
    reserved = _non_negative(reserved_seconds, field="reserved_seconds")
    elapsed = _non_negative(elapsed_seconds, field="elapsed_seconds")
    disk_rate = (
        _non_negative(stopped_disk_usd_per_hour, field="stopped_disk_usd_per_hour")
        if stopped_disk_usd_per_hour is not None
        else None
    )
    disk_seconds = _non_negative(stopped_disk_seconds, field="stopped_disk_seconds")
    storage_rates = {
        "container_disk_usd_per_hour": _non_negative(
            container_disk_usd_per_hour, field="container_disk_usd_per_hour"
        ),
        "persistent_volume_usd_per_hour": _non_negative(
            persistent_volume_usd_per_hour, field="persistent_volume_usd_per_hour"
        ),
        "network_volume_usd_per_hour": _non_negative(
            network_volume_usd_per_hour, field="network_volume_usd_per_hour"
        ),
    }
    storage_seconds = {
        "container_disk_seconds": _non_negative(
            elapsed if container_disk_seconds is None else container_disk_seconds,
            field="container_disk_seconds",
        ),
        "persistent_volume_seconds": _non_negative(
            elapsed + disk_seconds
            if persistent_volume_seconds is None
            else persistent_volume_seconds,
            field="persistent_volume_seconds",
        ),
        "network_volume_seconds": _non_negative(
            elapsed + disk_seconds
            if network_volume_seconds is None
            else network_volume_seconds,
            field="network_volume_seconds",
        ),
    }

    source = str(provider_reported_source or "").strip().lower()
    actual: float | None = None
    actual_refused_reason: str | None = None
    if provider_reported_actual_usd is not None:
        if source == PROVIDER_BILLING_API_SOURCE:
            actual = _non_negative(
                provider_reported_actual_usd, field="provider_reported_actual_usd"
            )
        else:
            actual_refused_reason = (
                "provider_reported_actual_requires_provider_billing_api_source"
            )

    component_costs = {
        "compute_usd": rate * elapsed / 3600.0,
        "container_disk_usd": storage_rates["container_disk_usd_per_hour"]
        * storage_seconds["container_disk_seconds"]
        / 3600.0,
        "persistent_volume_usd": storage_rates["persistent_volume_usd_per_hour"]
        * storage_seconds["persistent_volume_seconds"]
        / 3600.0,
        "network_volume_usd": storage_rates["network_volume_usd_per_hour"]
        * storage_seconds["network_volume_seconds"]
        / 3600.0,
    }
    elapsed_upper_bound = sum(component_costs.values())
    if disk_rate is not None:
        elapsed_upper_bound += disk_rate * disk_seconds / 3600.0
    durations = {
        str(name): round(float(value), 3)
        for name, value in (phase_durations_seconds or {}).items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "provider": str(provider or "").strip().lower() or None,
        "hourly_rate_usd": rate,
        "reserved_worst_case_usd": round(rate * reserved / 3600.0, 6),
        "elapsed_rate_upper_bound_usd": round(elapsed_upper_bound, 6),
        "provider_reported_actual_usd": actual,
        "provider_reported_actual_refused_reason": actual_refused_reason,
        "standing_stopped_disk_usd_per_hour": disk_rate,
        "storage_rate_breakdown_usd_per_hour": storage_rates,
        "storage_duration_breakdown_seconds": storage_seconds,
        "cost_component_upper_bounds_usd": {
            name: round(value, 6) for name, value in component_costs.items()
        },
        "stopped_disk_seconds": disk_seconds,
        "billing_reconciliation": (
            BILLING_RECONCILIATION_PROVIDER_API
            if actual is not None
            else BILLING_RECONCILIATION_NOT_CONFIGURED
        ),
        "phase_durations_seconds": durations,
        "estimate_labeled_actual": False,
        "claim_boundary": (
            "reserved_worst_case_usd and elapsed_rate_upper_bound_usd are "
            "conservative estimates from provider rate and allocation age, not "
            "invoice amounts. Only provider_reported_actual_usd sourced from an "
            "authoritative provider billing API is actual spend."
        ),
    }


class SpendCapExceeded(RuntimeError):
    """Raised when admitting a reservation would cross the total cap."""

    def __init__(self, admission: Mapping[str, Any]):
        self.admission = dict(admission)
        super().__init__(
            "startup_spend_cap_exceeded:"
            f"{self.admission.get('requested_reservation_usd')}"
        )


class CumulativeSpendLedger:
    """Goal-level cumulative ledger across every attempt, failed or not.

    ``admit`` must be called before each allocation; it fails closed when the
    open reservations plus settled upper bounds plus the new reservation would
    exceed the total cap. ``settle`` replaces an attempt's reservation with its
    elapsed-rate upper bound once the attempt terminates.
    """

    def __init__(self, path: str | Path, *, total_cap_usd: float) -> None:
        self.path = Path(path)
        self.total_cap_usd = _non_negative(total_cap_usd, field="total_cap_usd")
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.is_file():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = None
            if (
                isinstance(payload, Mapping)
                and payload.get("schema_version") == LEDGER_SCHEMA_VERSION
            ):
                state = dict(payload)
                state["attempts"] = [
                    dict(item)
                    for item in state.get("attempts", [])
                    if isinstance(item, Mapping)
                ]
                return state
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "total_cap_usd": self.total_cap_usd,
            "attempts": [],
        }

    def _persist(self) -> None:
        self._state["total_cap_usd"] = self.total_cap_usd
        self._state["updated_at"] = utc_now_iso()
        self._state["committed_usd"] = self.committed_usd()
        self._state["remaining_usd"] = round(
            max(0.0, self.total_cap_usd - self.committed_usd()), 6
        )
        write_json(self.path, self._state)

    def attempts(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._state["attempts"]]

    def committed_usd(self) -> float:
        """Open reservations plus settled elapsed upper bounds, all attempts."""
        total = 0.0
        for attempt in self._state["attempts"]:
            if attempt.get("settled"):
                total += float(attempt.get("elapsed_upper_bound_usd") or 0.0)
            else:
                total += float(attempt.get("reserved_usd") or 0.0)
        return round(total, 6)

    def admit(self, *, attempt_id: str, reserved_usd: float) -> dict[str, Any]:
        reservation = _non_negative(reserved_usd, field="reserved_usd")
        attempt_key = str(attempt_id or "").strip()
        if not attempt_key:
            raise ValueError("spend_ledger_attempt_id_missing")
        committed = self.committed_usd()
        admission = {
            "attempt_id": attempt_key,
            "requested_reservation_usd": round(reservation, 6),
            "committed_before_usd": committed,
            "total_cap_usd": self.total_cap_usd,
            "admitted": committed + reservation <= self.total_cap_usd,
            "admitted_at": utc_now_iso(),
        }
        if not admission["admitted"]:
            admission["blocker"] = "startup_cumulative_spend_cap_exceeded"
            raise SpendCapExceeded(admission)
        self._state["attempts"].append(
            {
                "attempt_id": attempt_key,
                "reserved_usd": round(reservation, 6),
                "elapsed_upper_bound_usd": None,
                "outcome": None,
                "settled": False,
                "admitted_at": admission["admitted_at"],
            }
        )
        self._persist()
        return admission

    def settle(
        self, *, attempt_id: str, elapsed_upper_bound_usd: float, outcome: str
    ) -> dict[str, Any]:
        upper = _non_negative(elapsed_upper_bound_usd, field="elapsed_upper_bound_usd")
        attempt_key = str(attempt_id or "").strip()
        for attempt in self._state["attempts"]:
            if attempt.get("attempt_id") == attempt_key and not attempt.get("settled"):
                attempt["elapsed_upper_bound_usd"] = round(upper, 6)
                attempt["outcome"] = str(outcome or "").strip() or None
                attempt["settled"] = True
                attempt["settled_at"] = utc_now_iso()
                self._persist()
                return dict(attempt)
        raise ValueError(f"spend_ledger_attempt_not_open:{attempt_key}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "total_cap_usd": self.total_cap_usd,
            "committed_usd": self.committed_usd(),
            "remaining_usd": round(max(0.0, self.total_cap_usd - self.committed_usd()), 6),
            "attempt_count": len(self._state["attempts"]),
            "attempts": self.attempts(),
            "includes_failed_attempts": True,
            "claim_boundary": (
                "Committed totals are reservations and elapsed-rate upper "
                "bounds, not provider invoices."
            ),
        }
