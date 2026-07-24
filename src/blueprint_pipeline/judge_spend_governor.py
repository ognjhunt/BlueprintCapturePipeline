"""Spend governance for VLM/LLM judge inference.

Rented GPUs in this repository are governed carefully: an hourly rate ceiling, a
target spend, a hard cap, a live-minutes TTL, a detached watchdog armed before
allocation, and a budget ledger written per attempt. Judge inference -- which is
also metered, also billable, and also launched from automation -- had none of
that. A single boolean environment flag authorised an unbounded number of
requests against a paid API.

That asymmetry gets worse precisely as graded task-progress scoring is adopted:
a binary success label reads a handful of frames per rollout, while a 0-5
progress rubric over a 300-frame episode reads an order of magnitude more, and a
qualification cohort multiplies that by policies times sites times trials.

This module gives judge spend the same shape the GPU path already has:

* a **policy** with a target spend, a hard cap, a request ceiling, a frame
  ceiling and a TTL;
* a **ledger** that records every reservation and settlement; and
* a **cohort hard stop** -- once the hard cap is reached every later reservation
  is denied, so an overspend is bounded by one in-flight request rather than by
  whenever someone notices.

Prices are never invented here. A policy that does not carry operator-supplied
rates cannot price a request, and an unpriceable request is denied rather than
waved through: governing spend you cannot measure is not governance.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable


POLICY_SCHEMA_VERSION = "judge_spend_policy.v1"
LEDGER_SCHEMA_VERSION = "judge_spend_ledger.v1"
DECISION_SCHEMA_VERSION = "judge_spend_decision.v1"

# Mirrors the GPU envelope vocabulary so an operator reads one mental model.
DEFAULT_TARGET_SPEND_USD = 5.00
DEFAULT_HARD_CAP_USD = 20.00
DEFAULT_MAX_REQUESTS = 2_000
DEFAULT_MAX_FRAMES = 40_000
DEFAULT_TTL_SECONDS = 6 * 3600

POLICY_ENV = "BLUEPRINT_JUDGE_SPEND_POLICY_FILE"
LEDGER_ENV = "BLUEPRINT_JUDGE_SPEND_LEDGER_FILE"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def build_judge_spend_policy(
    *,
    campaign_id: str,
    usd_per_1k_input_tokens: Any = None,
    usd_per_1k_output_tokens: Any = None,
    usd_per_image: Any = None,
    estimated_tokens_per_frame: Any = None,
    target_spend_usd: float = DEFAULT_TARGET_SPEND_USD,
    hard_cap_usd: float = DEFAULT_HARD_CAP_USD,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> dict[str, Any]:
    """Assemble a judge spend policy.

    At least one operator-supplied rate is required. Without a rate the governor
    can count requests and frames but cannot price them, and it says so rather
    than reporting a spend of zero.
    """

    rates = {
        "usd_per_1k_input_tokens": _number(usd_per_1k_input_tokens),
        "usd_per_1k_output_tokens": _number(usd_per_1k_output_tokens),
        "usd_per_image": _number(usd_per_image),
    }
    priceable = any(value is not None and value >= 0 for value in rates.values())
    blockers: list[str] = []
    if not _string(campaign_id):
        blockers.append("judge_spend_campaign_id_missing")
    if not priceable:
        blockers.append("judge_spend_rates_not_operator_supplied")
    if _number(hard_cap_usd) is None or float(hard_cap_usd) <= 0:
        blockers.append("judge_spend_hard_cap_invalid")
    if _number(target_spend_usd) is None or float(target_spend_usd) <= 0:
        blockers.append("judge_spend_target_invalid")
    elif _number(hard_cap_usd) is not None and float(target_spend_usd) > float(hard_cap_usd):
        blockers.append("judge_spend_target_above_hard_cap")

    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "campaign_id": _string(campaign_id),
        "rates": rates,
        "priceable": priceable,
        "estimated_tokens_per_frame": _number(estimated_tokens_per_frame),
        "target_spend_usd": _number(target_spend_usd),
        "hard_cap_usd": _number(hard_cap_usd),
        "max_requests": _positive_int(max_requests),
        "max_frames": _positive_int(max_frames),
        "ttl_seconds": _positive_int(ttl_seconds),
        "blockers": sorted(set(blockers)),
        "status": "ready" if not blockers else "blocked",
        "claim_boundary": {
            "policy_bounds_spend_not_judge_quality": True,
            "rates_are_operator_supplied_not_blueprint_measurements": True,
        },
    }


def estimate_request_cost_usd(
    *,
    policy: Mapping[str, Any],
    frame_count: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float | None:
    """Estimate one judge request's cost from operator-supplied rates."""

    rates = _mapping(policy.get("rates"))
    per_image = _number(rates.get("usd_per_image"))
    per_input = _number(rates.get("usd_per_1k_input_tokens"))
    per_output = _number(rates.get("usd_per_1k_output_tokens"))
    tokens_per_frame = _number(policy.get("estimated_tokens_per_frame"))

    if per_image is None and per_input is None and per_output is None:
        return None

    total = 0.0
    frames = max(0, int(frame_count or 0))
    if per_image is not None:
        total += per_image * frames
    effective_input = max(0, int(input_tokens or 0))
    if per_image is None and tokens_per_frame is not None:
        # Frames priced as tokens when the provider bills that way.
        effective_input += int(round(tokens_per_frame * frames))
    if per_input is not None:
        total += per_input * effective_input / 1000.0
    if per_output is not None:
        total += per_output * max(0, int(output_tokens or 0)) / 1000.0
    return round(total, 6)


class JudgeSpendGovernor:
    """Bounded, ledgered authorisation for judge inference."""

    def __init__(
        self,
        *,
        policy: Mapping[str, Any],
        ledger_path: str | Path | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        now_iso: Callable[[], str] | None = None,
    ) -> None:
        self._policy = dict(policy)
        self._ledger_path = Path(ledger_path).expanduser() if ledger_path else None
        self._monotonic = monotonic
        self._now_iso = now_iso
        self._started = monotonic()
        self.spent_usd = 0.0
        self.request_count = 0
        self.frame_count = 0
        self.denied_count = 0
        self.entries: list[dict[str, Any]] = []
        self._stopped_reason: str | None = None

    # -- authorisation -----------------------------------------------------

    @property
    def stopped(self) -> bool:
        return self._stopped_reason is not None

    def _deny(self, reasons: Sequence[str], *, estimate: float | None) -> dict[str, Any]:
        self.denied_count += 1
        decision = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "authorized": False,
            "blockers": sorted(set(reasons)),
            "estimated_cost_usd": estimate,
            "spent_usd": round(self.spent_usd, 6),
            "request_count": self.request_count,
            "frame_count": self.frame_count,
        }
        self._append(decision)
        return decision

    def authorize(
        self, *, frame_count: int = 0, input_tokens: int = 0, output_tokens: int = 0
    ) -> dict[str, Any]:
        """Decide whether one judge request may proceed."""

        policy = self._policy
        reasons: list[str] = []
        if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
            reasons.append("judge_spend_policy_schema_invalid")
        if policy.get("status") != "ready":
            reasons.append("judge_spend_policy_not_ready")
        if self._stopped_reason:
            reasons.append(self._stopped_reason)

        estimate = estimate_request_cost_usd(
            policy=policy,
            frame_count=frame_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if estimate is None:
            reasons.append("judge_spend_request_not_priceable")

        ttl = policy.get("ttl_seconds")
        if isinstance(ttl, int) and self._monotonic() - self._started > ttl:
            self._stopped_reason = "judge_spend_campaign_ttl_expired"
            reasons.append(self._stopped_reason)

        max_requests = policy.get("max_requests")
        if isinstance(max_requests, int) and self.request_count >= max_requests:
            self._stopped_reason = "judge_spend_request_ceiling_reached"
            reasons.append(self._stopped_reason)

        max_frames = policy.get("max_frames")
        if isinstance(max_frames, int) and self.frame_count + max(0, frame_count) > max_frames:
            self._stopped_reason = "judge_spend_frame_ceiling_reached"
            reasons.append(self._stopped_reason)

        hard_cap = _number(policy.get("hard_cap_usd"))
        if hard_cap is not None and estimate is not None:
            # Checked against the projected post-request total, so the cap
            # bounds total spend rather than being noticed one request late.
            if self.spent_usd + estimate > hard_cap:
                self._stopped_reason = "judge_spend_hard_cap_reached"
                reasons.append(self._stopped_reason)

        if reasons:
            return self._deny(reasons, estimate=estimate)

        target = _number(policy.get("target_spend_usd"))
        decision = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "authorized": True,
            "blockers": [],
            "estimated_cost_usd": estimate,
            "spent_usd": round(self.spent_usd, 6),
            "request_count": self.request_count,
            "frame_count": self.frame_count,
            "over_target_spend": bool(
                target is not None and estimate is not None and self.spent_usd + estimate > target
            ),
        }
        self._append(decision)
        return decision

    def settle(
        self, *, frame_count: int = 0, actual_cost_usd: Any = None, estimated_cost_usd: Any = None
    ) -> dict[str, Any]:
        """Record what a completed request actually consumed."""

        cost = _number(actual_cost_usd)
        if cost is None:
            cost = _number(estimated_cost_usd) or 0.0
        self.spent_usd += float(cost)
        self.request_count += 1
        self.frame_count += max(0, int(frame_count or 0))
        entry = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "event": "settled",
            "cost_usd": round(float(cost), 6),
            "cost_is_actual": actual_cost_usd is not None,
            "frame_count": max(0, int(frame_count or 0)),
            "spent_usd": round(self.spent_usd, 6),
            "request_count": self.request_count,
        }
        hard_cap = _number(self._policy.get("hard_cap_usd"))
        if hard_cap is not None and self.spent_usd >= hard_cap:
            self._stopped_reason = "judge_spend_hard_cap_reached"
            entry["cohort_stopped"] = True
        self._append(entry)
        return entry

    # -- reporting ---------------------------------------------------------

    def _append(self, entry: Mapping[str, Any]) -> None:
        row = dict(entry)
        if self._now_iso is not None:
            row.setdefault("recorded_at", self._now_iso())
        self.entries.append(row)
        if self._ledger_path is not None:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self._ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def ledger(self) -> dict[str, Any]:
        policy = self._policy
        hard_cap = _number(policy.get("hard_cap_usd"))
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "campaign_id": policy.get("campaign_id"),
            "status": "stopped" if self._stopped_reason else "open",
            "stopped_reason": self._stopped_reason,
            "spent_usd": round(self.spent_usd, 6),
            "target_spend_usd": _number(policy.get("target_spend_usd")),
            "hard_cap_usd": hard_cap,
            "remaining_usd": (
                round(max(0.0, hard_cap - self.spent_usd), 6) if hard_cap is not None else None
            ),
            "request_count": self.request_count,
            "frame_count": self.frame_count,
            "denied_count": self.denied_count,
            "entries": list(self.entries),
            "claim_boundary": {
                "ledger_bounds_spend_not_judge_quality": True,
                "estimated_costs_are_not_provider_invoices": True,
            },
        }


def load_policy_from_env() -> dict[str, Any] | None:
    """Load a judge spend policy from the operator-configured file, if any."""

    path = _string(os.getenv(POLICY_ENV))
    if not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return None
    try:
        return dict(json.loads(candidate.read_text(encoding="utf-8")))
    except (ValueError, OSError):
        return None


def governor_from_env(*, campaign_id: str) -> JudgeSpendGovernor | None:
    """Build a governor from environment configuration.

    Returns ``None`` when no policy is configured so callers can decide whether
    ungoverned judge inference is acceptable for their lane; the graded-progress
    lane treats ``None`` as a refusal.
    """

    policy = load_policy_from_env()
    if policy is None:
        return None
    policy.setdefault("campaign_id", campaign_id)
    ledger_path = _string(os.getenv(LEDGER_ENV)) or None
    return JudgeSpendGovernor(policy=policy, ledger_path=ledger_path)
