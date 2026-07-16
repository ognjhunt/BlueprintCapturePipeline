"""Platform-level rolling fleet spend ledger and aggregate GPU-cost ceiling.

Audit finding R041 (P1): the robot-eval orchestrator's GPU cost guardrails were
strictly per-job. ``_provider_prelaunch_spend_guard`` validated a single request
(one positive ``requested_budget_usd`` and ``max_active_workers == 1``) and
``_gpu_cost_control_ledger`` accounted for one job at a time. Nothing tracked
aggregate/fleet spend, so N concurrent jobs could each pass the per-job gate
while the platform bill grew without any ceiling or kill switch.

This module is a small, self-contained, deterministic (CPU-only, no network)
platform ledger. It persists a rolling record of external-provider launches to a
JSON file and exposes an aggregate ceiling that is consulted BEFORE any launch:

* ``check_budget`` fails CLOSED when a configured daily cap, monthly/total cap,
  or max-concurrent-GPU ceiling *would be* exceeded by the pending launch, or
  when a global kill switch is engaged, or (when any guardrail is active) when
  the ledger itself cannot be read.
* When no caps and no kill switch are configured, the check is DEFAULT-SAFE: it
  allows the launch and does not touch disk, so pre-existing per-job behaviour
  and tests are unchanged. The ledger fields are still surfaced so callers can
  expose remaining budget.

Determinism: the clock and the ledger path are injectable. ``clock`` defaults to
real UTC wall time and ``path`` defaults to the ``BLUEPRINT_FLEET_SPEND_LEDGER_PATH``
env var, but tests pass explicit values so results are reproducible.

Environment variables (all optional; absence == guardrail disabled):

* ``BLUEPRINT_FLEET_DAILY_SPEND_USD``    - rolling 24h spend ceiling (USD).
* ``BLUEPRINT_FLEET_MONTHLY_SPEND_USD``  - rolling 30d / total spend ceiling (USD).
* ``BLUEPRINT_FLEET_MAX_CONCURRENT_GPU`` - max concurrent active GPU pods.
* ``BLUEPRINT_FLEET_SPEND_KILL_SWITCH``  - truthy => block ALL launches.
* ``BLUEPRINT_FLEET_SPEND_LEDGER_PATH``  - JSON ledger file path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional


FLEET_SPEND_LEDGER_SCHEMA_VERSION = "blueprint_fleet_spend_ledger.v1"
FLEET_SPEND_GUARD_DECISION_SCHEMA_VERSION = "blueprint_fleet_spend_guard_decision.v1"

DEFAULT_LEDGER_PATH = "output/fleet_spend_ledger.json"

DAILY_SPEND_ENV = "BLUEPRINT_FLEET_DAILY_SPEND_USD"
MONTHLY_SPEND_ENV = "BLUEPRINT_FLEET_MONTHLY_SPEND_USD"
MAX_CONCURRENT_GPU_ENV = "BLUEPRINT_FLEET_MAX_CONCURRENT_GPU"
KILL_SWITCH_ENV = "BLUEPRINT_FLEET_SPEND_KILL_SWITCH"
LEDGER_PATH_ENV = "BLUEPRINT_FLEET_SPEND_LEDGER_PATH"

_ROLLING_DAILY_WINDOW = timedelta(hours=24)
_ROLLING_MONTHLY_WINDOW = timedelta(days=30)

_TRUTHY = {"1", "true", "yes", "on", "engaged", "enabled"}


# Blocker identifiers surfaced to callers (stable strings for assertions/wiring).
BLOCKER_KILL_SWITCH = "fleet_spend_kill_switch_engaged"
BLOCKER_DAILY_CAP = "fleet_daily_spend_cap_exceeded"
BLOCKER_MONTHLY_CAP = "fleet_monthly_spend_cap_exceeded"
BLOCKER_MAX_CONCURRENT_GPU = "fleet_max_concurrent_gpu_exceeded"
BLOCKER_LEDGER_UNREADABLE = "fleet_spend_ledger_unreadable_fail_closed"


def _env_truthy(value: Optional[str]) -> bool:
    return bool(value) and str(value).strip().lower() in _TRUTHY


def _positive_number(value: Any) -> Optional[float]:
    """Parse a strictly-positive number, else ``None`` (guardrail disabled)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _positive_int(value: Any) -> Optional[int]:
    number = _positive_number(value)
    if number is None:
        return None
    return int(number)


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class FleetSpendCaps:
    """Aggregate ceiling configuration. ``None`` fields mean "not enforced"."""

    daily_spend_usd: Optional[float] = None
    monthly_spend_usd: Optional[float] = None
    max_concurrent_gpu: Optional[int] = None
    kill_switch: bool = False

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "FleetSpendCaps":
        source: Mapping[str, str] = env if env is not None else os.environ
        return cls(
            daily_spend_usd=_positive_number(source.get(DAILY_SPEND_ENV)),
            monthly_spend_usd=_positive_number(source.get(MONTHLY_SPEND_ENV)),
            max_concurrent_gpu=_positive_int(source.get(MAX_CONCURRENT_GPU_ENV)),
            kill_switch=_env_truthy(source.get(KILL_SWITCH_ENV)),
        )

    @property
    def any_enforced(self) -> bool:
        return bool(
            self.kill_switch
            or self.daily_spend_usd is not None
            or self.monthly_spend_usd is not None
            or self.max_concurrent_gpu is not None
        )


@dataclass(frozen=True)
class FleetSpendTotals:
    rolling_daily_spend_usd: float
    rolling_monthly_spend_usd: float
    active_gpu_pods: int
    record_count: int


class FleetSpendLedger:
    """A rolling ledger of external-provider launches persisted to JSON.

    Parameters
    ----------
    path:
        Ledger file path. May not exist yet; reads of a missing file yield an
        empty ledger and never create it.
    clock:
        Zero-arg callable returning an aware UTC ``datetime``. Injectable for
        deterministic tests; defaults to real UTC wall time.
    """

    def __init__(
        self,
        path: Any,
        *,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.path = Path(path)
        self._clock = clock or _default_clock

    # -- persistence ----------------------------------------------------------

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)

    def load_records(self) -> List[Dict[str, Any]]:
        """Return the persisted launch records.

        Raises on a corrupt/unreadable ledger so callers can fail closed.
        A missing file is treated as an empty ledger (not an error).
        """
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            records = raw.get("records", [])
        else:
            records = raw
        if not isinstance(records, list):
            raise ValueError("fleet spend ledger 'records' must be a list")
        out: List[Dict[str, Any]] = []
        for record in records:
            if isinstance(record, Mapping):
                out.append(dict(record))
        return out

    def _write_records(self, records: List[Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": FLEET_SPEND_LEDGER_SCHEMA_VERSION,
            "updated_at": self._now().isoformat(),
            "records": records,
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, self.path)

    # -- rolling aggregates ---------------------------------------------------

    @staticmethod
    def _record_spend(record: Mapping[str, Any]) -> float:
        """Spend attributed to a record: actual if known, else estimated."""
        actual = record.get("actual_usd")
        if actual is not None:
            value = _positive_number(actual)
            if value is not None:
                return value
            # actual explicitly set to 0 (or non-positive) -> zero spend.
            try:
                return max(0.0, float(actual))
            except (TypeError, ValueError):
                return 0.0
        estimated = record.get("estimated_usd")
        try:
            return max(0.0, float(estimated))
        except (TypeError, ValueError):
            return 0.0

    def totals(
        self,
        records: Optional[List[Dict[str, Any]]] = None,
        *,
        now: Optional[datetime] = None,
    ) -> FleetSpendTotals:
        if records is None:
            records = self.load_records()
        if now is None:
            now = self._now()
        daily_cutoff = now - _ROLLING_DAILY_WINDOW
        monthly_cutoff = now - _ROLLING_MONTHLY_WINDOW
        daily = 0.0
        monthly = 0.0
        active_gpu = 0
        for record in records:
            spend = self._record_spend(record)
            timestamp = _parse_timestamp(record.get("timestamp"))
            # An unparseable timestamp is counted conservatively (fail-closed):
            # attributed to both rolling windows so it cannot hide spend.
            if timestamp is None or timestamp >= monthly_cutoff:
                monthly += spend
            if timestamp is None or timestamp >= daily_cutoff:
                daily += spend
            if bool(record.get("active", False)):
                try:
                    active_gpu += max(0, int(record.get("gpu_count", 0) or 0))
                except (TypeError, ValueError):
                    active_gpu += 0
        return FleetSpendTotals(
            rolling_daily_spend_usd=round(daily, 6),
            rolling_monthly_spend_usd=round(monthly, 6),
            active_gpu_pods=active_gpu,
            record_count=len(records),
        )

    # -- mutations ------------------------------------------------------------

    def record_launch(
        self,
        *,
        job_id: str,
        estimated_usd: float,
        gpu_count: int = 1,
        provider: str = "",
        active: bool = True,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Append a launch record and persist the ledger. Returns the record."""
        records = self.load_records()
        timestamp = (now or self._now())
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        record = {
            "timestamp": timestamp.astimezone(timezone.utc).isoformat(),
            "job_id": str(job_id),
            "provider": str(provider or ""),
            "estimated_usd": max(0.0, float(estimated_usd or 0.0)),
            "actual_usd": None,
            "gpu_count": max(0, int(gpu_count or 0)),
            "active": bool(active),
        }
        records.append(record)
        self._write_records(records)
        return record

    def record_actual(self, *, job_id: str, actual_usd: float, active: bool = False) -> bool:
        """Reconcile the most recent launch of ``job_id`` with actual spend."""
        records = self.load_records()
        for record in reversed(records):
            if str(record.get("job_id")) == str(job_id):
                record["actual_usd"] = max(0.0, float(actual_usd or 0.0))
                record["active"] = bool(active)
                self._write_records(records)
                return True
        return False

    def mark_inactive(self, *, job_id: str) -> bool:
        """Mark all launches of ``job_id`` inactive (pod torn down)."""
        records = self.load_records()
        changed = False
        for record in records:
            if str(record.get("job_id")) == str(job_id) and record.get("active"):
                record["active"] = False
                changed = True
        if changed:
            self._write_records(records)
        return changed

    # -- the aggregate ceiling ------------------------------------------------

    def check_budget(
        self,
        *,
        estimated_usd: float,
        gpu_count: int,
        caps: FleetSpendCaps,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Evaluate the aggregate ceiling for a pending launch.

        Returns a decision mapping. ``allowed`` is ``False`` (fail closed) when a
        guardrail would be breached, the kill switch is engaged, or (when any
        guardrail is active) the ledger cannot be read. When no guardrails are
        configured the decision is allowed and the ledger is not read.
        """
        now = now or self._now()
        estimated_usd = max(0.0, float(estimated_usd or 0.0))
        gpu_count = max(0, int(gpu_count or 0))

        base: Dict[str, Any] = {
            "schema_version": FLEET_SPEND_GUARD_DECISION_SCHEMA_VERSION,
            "generated_at": now.isoformat(),
            "ledger_path": str(self.path),
            "aggregate_ceiling_enforced": caps.any_enforced,
            "kill_switch_engaged": bool(caps.kill_switch),
            "pending_launch": {
                "estimated_usd": estimated_usd,
                "gpu_count": gpu_count,
            },
            "caps": {
                "daily_spend_usd": caps.daily_spend_usd,
                "monthly_spend_usd": caps.monthly_spend_usd,
                "max_concurrent_gpu": caps.max_concurrent_gpu,
                "kill_switch": bool(caps.kill_switch),
            },
        }

        # DEFAULT-SAFE: nothing configured -> allow without touching disk.
        if not caps.any_enforced:
            base.update(
                {
                    "allowed": True,
                    "fail_closed": False,
                    "ledger_read": False,
                    "current": {
                        "rolling_daily_spend_usd": None,
                        "rolling_monthly_spend_usd": None,
                        "active_gpu_pods": None,
                        "record_count": None,
                    },
                    "projected": {
                        "rolling_daily_spend_usd": None,
                        "rolling_monthly_spend_usd": None,
                        "active_gpu_pods": None,
                    },
                    "remaining": {
                        "daily_spend_usd": None,
                        "monthly_spend_usd": None,
                        "concurrent_gpu": None,
                    },
                    "blockers": [],
                }
            )
            return base

        # A guardrail is active: read the ledger, failing closed on any error.
        try:
            totals = self.totals(now=now)
            ledger_read = True
        except Exception as exc:  # noqa: BLE001 - fail closed on any ledger error.
            base.update(
                {
                    "allowed": False,
                    "fail_closed": True,
                    "ledger_read": False,
                    "ledger_error": type(exc).__name__,
                    "current": None,
                    "projected": None,
                    "remaining": None,
                    "blockers": [BLOCKER_LEDGER_UNREADABLE],
                }
            )
            return base

        projected_daily = round(totals.rolling_daily_spend_usd + estimated_usd, 6)
        projected_monthly = round(totals.rolling_monthly_spend_usd + estimated_usd, 6)
        projected_gpu = totals.active_gpu_pods + gpu_count

        blockers: List[str] = []
        if caps.kill_switch:
            blockers.append(BLOCKER_KILL_SWITCH)
        if caps.daily_spend_usd is not None and projected_daily > caps.daily_spend_usd:
            blockers.append(BLOCKER_DAILY_CAP)
        if caps.monthly_spend_usd is not None and projected_monthly > caps.monthly_spend_usd:
            blockers.append(BLOCKER_MONTHLY_CAP)
        if caps.max_concurrent_gpu is not None and projected_gpu > caps.max_concurrent_gpu:
            blockers.append(BLOCKER_MAX_CONCURRENT_GPU)

        def _remaining(cap: Optional[float], current: float) -> Optional[float]:
            if cap is None:
                return None
            return round(cap - current, 6)

        def _remaining_int(cap: Optional[int], current: int) -> Optional[int]:
            if cap is None:
                return None
            return cap - current

        base.update(
            {
                "allowed": not blockers,
                # A breach of an active ceiling IS the fail-closed stop.
                "fail_closed": bool(blockers),
                "ledger_read": ledger_read,
                "current": {
                    "rolling_daily_spend_usd": totals.rolling_daily_spend_usd,
                    "rolling_monthly_spend_usd": totals.rolling_monthly_spend_usd,
                    "active_gpu_pods": totals.active_gpu_pods,
                    "record_count": totals.record_count,
                },
                "projected": {
                    "rolling_daily_spend_usd": projected_daily,
                    "rolling_monthly_spend_usd": projected_monthly,
                    "active_gpu_pods": projected_gpu,
                },
                "remaining": {
                    "daily_spend_usd": _remaining(caps.daily_spend_usd, totals.rolling_daily_spend_usd),
                    "monthly_spend_usd": _remaining(
                        caps.monthly_spend_usd, totals.rolling_monthly_spend_usd
                    ),
                    "concurrent_gpu": _remaining_int(
                        caps.max_concurrent_gpu, totals.active_gpu_pods
                    ),
                },
                "blockers": blockers,
            }
        )
        return base


def resolve_ledger_path(
    path: Any = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> Path:
    if path is not None:
        return Path(path)
    source: Mapping[str, str] = env if env is not None else os.environ
    configured = source.get(LEDGER_PATH_ENV)
    if configured:
        return Path(configured)
    return Path(DEFAULT_LEDGER_PATH)


def evaluate_fleet_spend_guard(
    *,
    estimated_usd: float,
    gpu_count: int,
    now: Optional[datetime] = None,
    clock: Optional[Callable[[], datetime]] = None,
    env: Optional[Mapping[str, str]] = None,
    ledger_path: Any = None,
    caps: Optional[FleetSpendCaps] = None,
) -> Dict[str, Any]:
    """Convenience entry point used by the pre-launch guard.

    Builds caps from the environment (unless supplied), resolves the ledger path
    (env-overridable), and returns the aggregate-ceiling decision. Default-safe:
    with no caps configured it allows and does not read the ledger, so existing
    per-job behaviour and tests are unchanged.
    """
    caps = caps if caps is not None else FleetSpendCaps.from_env(env)
    ledger = FleetSpendLedger(
        resolve_ledger_path(ledger_path, env=env),
        clock=clock,
    )
    return ledger.check_budget(
        estimated_usd=estimated_usd,
        gpu_count=gpu_count,
        caps=caps,
        now=now,
    )
