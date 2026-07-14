"""Independent name-bound hard-TTL watchdog for the GR00T + OSCAR canary."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import utc_now_iso, write_json
from .gpu_render_providers import get_render_provider

SCHEMA_VERSION = "groot_oscar_runpod_canary_watchdog.v1"
EVIDENCE_NAME = "groot_oscar_runpod_canary_watchdog.json"


def arm_watchdog(
    *, out_dir: str | Path, pod_name_prefix: str, deadline_epoch: float, pid: int | None = None
) -> dict[str, Any]:
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not pod_name_prefix.startswith("blueprint-groot-oscar-canary-"):
        raise ValueError("watchdog_pod_name_prefix_not_canary_scoped")
    if float(deadline_epoch) <= time.time() + 60:
        raise ValueError("watchdog_deadline_must_be_more_than_60_seconds_future")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "armed",
        "independent_process": True,
        "pid": int(pid if pid is not None else os.getpid()),
        "armed_at": utc_now_iso(),
        "deadline_epoch": float(deadline_epoch),
        "pod_name_prefix": pod_name_prefix,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    write_json(root / EVIDENCE_NAME, payload)
    return payload


def terminate_canary_resources(
    *, provider: Any, pod_name_prefix: str, armed: Mapping[str, Any]
) -> dict[str, Any]:
    inventory = provider.billable_inventory(name_prefix=pod_name_prefix)
    resources = inventory.get("resources")
    resources = resources if isinstance(resources, list) else []
    terminations: list[dict[str, Any]] = []
    for row in resources:
        row = row if isinstance(row, Mapping) else {}
        instance_id = str(row.get("instance_id") or row.get("id") or "").strip()
        if not instance_id:
            terminations.append({"status": "blocked", "reason": "resource_id_missing"})
            continue
        result = provider.terminate(instance_id)
        terminations.append({"instance_id": instance_id, **dict(result)})
    final_inventory = provider.billable_inventory(name_prefix=pod_name_prefix)
    absent = bool(
        final_inventory.get("api_confirmed") is True
        and final_inventory.get("live_resource_count") == 0
    )
    return {
        **dict(armed),
        "status": "provider_terminal" if absent else "teardown_unverified",
        "completed_at": utc_now_iso(),
        "initial_inventory": inventory,
        "terminations": terminations,
        "final_inventory": final_inventory,
        "provider_absence_confirmed": absent,
        "provider_mutations_performed": len(terminations),
    }


def run_watchdog(
    *,
    out_dir: str | Path,
    pod_name_prefix: str,
    deadline_epoch: float,
    provider_factory: Callable[[str], Any] = get_render_provider,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    root = Path(out_dir).expanduser().resolve()
    armed = arm_watchdog(
        out_dir=root,
        pod_name_prefix=pod_name_prefix,
        deadline_epoch=deadline_epoch,
    )
    while clock() < deadline_epoch:
        sleeper(min(10.0, max(0.0, deadline_epoch - clock())))
    result = terminate_canary_resources(
        provider=provider_factory("runpod"),
        pod_name_prefix=pod_name_prefix,
        armed=armed,
    )
    write_json(root / EVIDENCE_NAME, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pod-name-prefix", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args(argv)
    result = run_watchdog(
        out_dir=args.out_dir,
        pod_name_prefix=args.pod_name_prefix,
        deadline_epoch=args.deadline_epoch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "provider_terminal" else 2


if __name__ == "__main__":
    raise SystemExit(main())
