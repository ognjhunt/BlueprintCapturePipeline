#!/usr/bin/env python3
"""Measure the appearance campaign's prior spend and seal it as an anchor.

Every link of the appearance campaign is authorized against a shared $12 cap
via `prior_spend + hard_cap_usd > aggregate_cap`, so something has to say what
the campaign has already spent. A retired compatibility design expected a
completed paid predecessor carrying both authority and terminal evidence, but
no qualifying predecessor exists for the active campaign. ArtiFixer3D instead
starts from the exact control-plane ledgers it can reopen and measure.

This produces the replacement, and the distinction that matters is that it
**measures** rather than asserts. It reads the consumed-authority directory and
the spend ledger, sums what it finds, and binds each file it read by digest
into `measured_from`. A receipt claiming no prior spend while naming prior paid
attempts is refused by the validator rather than believed, so a zero here has
to be a zero someone can re-derive.

Run it on the control plane, where those ledgers actually live. Pointing it at
an empty directory would measure an empty directory, which is why the paths are
required rather than defaulted.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_repair_spend_chain import (
    AGGREGATE_GOAL_SPEND_CAP_USD,
    CAMPAIGN_START_SCHEMA_VERSION,
    _record,
)

#: Cost fields a consumed authority or ledger row may carry.
COST_FIELDS = (
    "terminal_cost_usd",
    "estimated_cost_usd",
    "actual_cost_usd",
    "cost_usd",
)


def _cost_of(payload: Any) -> float:
    if not isinstance(payload, dict):
        return 0.0
    for field in COST_FIELDS:
        value = payload.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0


def measure(
    *, consumed_root: Path, spend_ledger: Path, campaign_marker: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Sum this campaign's paid attempts, and bind everything that was read."""

    evidence: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    total = 0.0

    if not consumed_root.is_dir():
        raise ValueError("appearance_campaign_consumed_root_missing")
    if not spend_ledger.is_file():
        raise ValueError("appearance_campaign_spend_ledger_missing")

    # The ledger is read and bound even when it names nothing, because "we
    # looked and it was empty" is the claim being sealed.
    evidence.append(_record(spend_ledger))
    ledger = json.loads(spend_ledger.read_text(encoding="utf-8"))
    for row in ledger.get("instances") or []:
        if campaign_marker in json.dumps(row).lower():
            cost = _cost_of(row)
            total += cost
            attempts.append({"source": "spend_ledger", "cost_usd": cost})

    for path in sorted(consumed_root.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        # Match the filename as well as the body: a consumed authority is named
        # for its lane, and its body need not repeat the lane name anywhere.
        if campaign_marker not in f"{path.name}\n{text}".lower():
            continue
        evidence.append(_record(path))
        payload = json.loads(text)
        cost = _cost_of(payload)
        total += cost
        attempts.append(
            {
                "source": "consumed_authority",
                "authority_digest": payload.get("authorization_digest"),
                "cost_usd": cost,
            }
        )

    return evidence, attempts, round(total, 6)


def seal(
    *,
    consumed_root: str | Path,
    spend_ledger: str | Path,
    campaign_marker: str,
    output_path: str | Path,
) -> dict[str, Any]:
    evidence, attempts, total = measure(
        consumed_root=Path(consumed_root).expanduser().resolve(),
        spend_ledger=Path(spend_ledger).expanduser().resolve(),
        campaign_marker=campaign_marker.lower(),
    )
    receipt: dict[str, Any] = {
        "schema_version": CAMPAIGN_START_SCHEMA_VERSION,
        "campaign_marker": campaign_marker,
        "prior_goal_spend_usd": total,
        "aggregate_goal_spend_cap_usd": AGGREGATE_GOAL_SPEND_CAP_USD,
        "measured_paid_attempts": attempts,
        "measured_from": evidence,
        "provider_mutation_performed": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consumed-authority-root",
        required=True,
        help="Where spent standing authorizations are recorded.",
    )
    parser.add_argument("--spend-ledger", required=True)
    parser.add_argument(
        "--campaign-marker",
        default="artifixer",
        help="Substring identifying this campaign's attempts.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        receipt = seal(
            consumed_root=args.consumed_authority_root,
            spend_ledger=args.spend_ledger,
            campaign_marker=args.campaign_marker,
            output_path=args.output,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                indent=1,
                sort_keys=True,
            )
        )
        return 2

    print(
        json.dumps(
            {
                "status": "sealed",
                "prior_goal_spend_usd": receipt["prior_goal_spend_usd"],
                "measured_paid_attempts": len(receipt["measured_paid_attempts"]),
                "measured_from": len(receipt["measured_from"]),
                "receipt_digest": receipt["receipt_digest"],
                "provider_mutation_performed": False,
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
