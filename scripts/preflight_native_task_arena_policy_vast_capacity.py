#!/usr/bin/env python3
"""Run a transfer-aware, read-only Vast capacity preflight for one policy."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from blueprint_pipeline.gpu_render_providers import VastRenderProvider
from blueprint_pipeline.native_task_arena_vast import (
    MINIMUM_DRIVER_VERSION,
    POLICY_MIN_COMPUTE_CAP,
    _policy_provider_transfer_byte_budget,
)


def build_native_task_arena_policy_vast_capacity_request(
    *,
    candidate: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    container_disk_gb: int = 200,
    required_provider_disk_gb: int = 200,
    allowed_machine_ids: Sequence[int] = (),
    excluded_machine_ids: Sequence[int] = (),
) -> dict:
    """Bind the no-spend request to the live transport's byte derivation."""

    if candidate not in {"pi05_droid", "groot_n17_droid"}:
        raise ValueError("native_task_arena_policy_capacity_candidate_invalid")
    if (
        type(max_hourly_rate_usd) not in {int, float}
        or not 0 < float(max_hourly_rate_usd) <= 0.80
        or type(hard_cap_usd) not in {int, float}
        or not 0 < float(hard_cap_usd)
        or type(hard_ttl_seconds) is not int
        or hard_ttl_seconds <= 0
        or float(max_hourly_rate_usd) * hard_ttl_seconds / 3600.0
        > float(hard_cap_usd)
    ):
        raise ValueError("native_task_arena_policy_capacity_budget_invalid")
    if (
        type(container_disk_gb) is not int
        or container_disk_gb <= 0
        or type(required_provider_disk_gb) is not int
        or required_provider_disk_gb < container_disk_gb
    ):
        raise ValueError("native_task_arena_policy_capacity_disk_invalid")
    allowed = tuple(sorted({int(value) for value in allowed_machine_ids}))
    excluded = tuple(sorted({int(value) for value in excluded_machine_ids}))
    if any(value <= 0 for value in (*allowed, *excluded)) or set(allowed) & set(
        excluded
    ):
        raise ValueError("native_task_arena_policy_capacity_machine_scope_invalid")
    download_bytes, upload_bytes = _policy_provider_transfer_byte_budget(candidate)
    return {
        "container_disk_gb": container_disk_gb,
        "required_provider_disk_gb": required_provider_disk_gb,
        "max_hourly_rate_usd": float(max_hourly_rate_usd),
        "hard_cap_usd": float(hard_cap_usd),
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "min_gpu_ram_mb": 46_000,
        "min_compute_cap": POLICY_MIN_COMPUTE_CAP,
        "max_compute_cap": 900,
        "minimum_driver_version": MINIMUM_DRIVER_VERSION,
        "require_known_supported_isaac_driver": True,
        "require_direct_port": True,
        "require_global_inventory_zero": True,
        "prefer_isaac_rt": True,
        "preferred_gpu_keywords": ["L40S", "RTX 6000 Ada", "RTX A6000"],
        "allowed_machine_ids": list(allowed),
        "excluded_machine_ids": list(excluded),
        "expected_provider_download_bytes": download_bytes,
        "expected_provider_upload_bytes": upload_bytes,
    }


def _write_exclusive_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != encoded:
        raise OSError("native_task_arena_capacity_receipt_readback_mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", choices=("pi05_droid", "groot_n17_droid"), required=True
    )
    parser.add_argument("--max-hourly-rate-usd", type=float, required=True)
    parser.add_argument("--hard-cap-usd", type=float, required=True)
    parser.add_argument("--hard-ttl-seconds", type=int, required=True)
    parser.add_argument("--container-disk-gb", type=int, default=200)
    parser.add_argument("--required-provider-disk-gb", type=int, default=200)
    parser.add_argument("--allow-machine", action="append", type=int, default=[])
    parser.add_argument("--exclude-machine", action="append", type=int, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        request = build_native_task_arena_policy_vast_capacity_request(
            candidate=args.candidate,
            max_hourly_rate_usd=args.max_hourly_rate_usd,
            hard_cap_usd=args.hard_cap_usd,
            hard_ttl_seconds=args.hard_ttl_seconds,
            container_disk_gb=args.container_disk_gb,
            required_provider_disk_gb=args.required_provider_disk_gb,
            allowed_machine_ids=args.allow_machine,
            excluded_machine_ids=args.exclude_machine,
        )
        result = VastRenderProvider().capacity_preflight(request)
        receipt = {
            "schema_version": "blueprint.native-task-arena-vast-capacity-preflight.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": result.get("status"),
            "candidate": args.candidate,
            "request": request,
            "result": result,
            "provider_mutation_performed": False,
            "raw_secret_values_recorded": False,
        }
        _write_exclusive_json(args.output.expanduser().resolve(), receipt)
    except (OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2

    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(args.output),
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "available" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
