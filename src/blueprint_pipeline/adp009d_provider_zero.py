"""Read-only, digest-bound provider-zero evidence for ADP-009D canaries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .common import write_json


REQUIRED_PROVIDERS = ("runpod", "vast", "digitalocean")
SCHEMA_VERSION = "gpu_spend_guard.v1"


def build_provider_zero_receipt(
    inventories: Mapping[str, Mapping[str, Any]], *, now: datetime | None = None
) -> dict[str, Any]:
    """Normalize three read-only provider inventories without creating resources.

    ``GpuRenderProvider.billable_inventory`` deliberately returns only sanitized
    resource rows.  Passing an empty name prefix makes each provider query its
    whole billable inventory, which is the scope required before a canary can
    claim global provider-zero.
    """

    generated_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    inventory_results: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    blockers: list[str] = []
    for provider in REQUIRED_PROVIDERS:
        observed = dict(inventories.get(provider) or {})
        resources = observed.get("resources")
        api_confirmed = observed.get("api_confirmed") is True
        source_blockers = [
            str(value)
            for value in observed.get("blockers") or []
            if isinstance(value, str) and value
        ]
        valid_resources = (
            isinstance(resources, list)
            and all(isinstance(row, Mapping) for row in resources)
        )
        succeeded = (
            observed.get("provider") == provider
            and observed.get("status") == "observed"
            and api_confirmed
            and valid_resources
            and not source_blockers
        )
        row_blockers = list(source_blockers)
        if not succeeded and not row_blockers:
            row_blockers.append(f"{provider}_provider_inventory_invalid")
        inventory_results.append(
            {
                "provider": provider,
                "required": True,
                "credential_present": api_confirmed,
                "status": "succeeded" if succeeded else "blocked",
                "row_count": len(resources) if valid_resources else 0,
                "blockers": row_blockers,
                "api_confirmed": api_confirmed,
                "raw_provider_response_recorded": False,
            }
        )
        if not succeeded:
            blockers.extend(f"provider_inventory:{provider}:{item}" for item in row_blockers)
            continue
        for resource in resources:
            resource_id = str(
                resource.get("instance_id") or resource.get("id") or ""
            ).strip()
            if not resource_id:
                blockers.append(f"provider_inventory:{provider}:resource_id_missing")
                continue
            instances.append(
                {
                    "provider": provider,
                    "id": resource_id,
                    "live": True,
                }
            )
    if instances:
        blockers.append("provider_zero_live_resources_detected")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "generated_at": generated_at.isoformat(),
        "inventory_scope": "global_billable_resources",
        "inventory_results": inventory_results,
        "instances": instances,
        "live_instance_count": len(instances),
        "total_burn_per_hour_usd": 0.0 if not instances else None,
        "provider_zero_verified": not blockers,
        "provider_zero": {
            "status": "verified" if not blockers else "blocked",
            "required_provider_ids": list(REQUIRED_PROVIDERS),
            "global_live_instance_count": len(instances),
            "global_total_burn_per_hour_usd": 0.0 if not instances else None,
            "blockers": sorted(set(blockers)),
            "claim_boundary": (
                "Provider-zero is verified only when every required provider "
                "inventory query succeeds with zero returned resources."
            ),
        },
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def collect_provider_zero_receipt() -> dict[str, Any]:
    """Query the supported provider APIs without performing any mutation."""

    from .gpu_render_providers import get_render_provider

    inventories = {
        provider: get_render_provider(provider).billable_inventory(name_prefix="")
        for provider in REQUIRED_PROVIDERS
    }
    return build_provider_zero_receipt(inventories)


def main(argv: list[str] | None = None) -> int:
    """Write one compact, read-only global provider-zero observation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = collect_provider_zero_receipt()
    write_json(Path(args.output), receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "generated_at": receipt["generated_at"],
                "live_instance_count": receipt["live_instance_count"],
                "provider_zero_verified": receipt["provider_zero_verified"],
                "blockers": receipt["blockers"],
                "receipt_digest": receipt["receipt_digest"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "passed" else 2


__all__ = [
    "REQUIRED_PROVIDERS",
    "SCHEMA_VERSION",
    "build_provider_zero_receipt",
    "collect_provider_zero_receipt",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
