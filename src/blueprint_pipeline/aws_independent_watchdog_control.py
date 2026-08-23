"""Independent hard-TTL teardown watchdog for one AWS reconstruction lane."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .common import utc_now_iso, write_json
from .gpu_render_providers import get_render_provider

HANDOFF_SCHEMA = "aws_independent_watchdog_handoff.v1"


def arm_aws_watchdog(
    *, job_dir: str | Path, name_prefix: str, hard_ttl_seconds: int
) -> tuple[dict[str, Any], subprocess.Popen[bytes] | None]:
    root = Path(job_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not name_prefix.startswith("blueprint-postshot-") or hard_ttl_seconds < 300:
        return ({
            "schema_version": HANDOFF_SCHEMA,
            "status": "blocked",
            "blockers": ["aws_independent_watchdog_scope_or_ttl_invalid"],
            "independent_process": False,
        }, None)
    config = {
        "schema_version": HANDOFF_SCHEMA,
        "name_prefix": name_prefix,
        "deadline_epoch": time.time() + hard_ttl_seconds,
        "poll_seconds": 15,
    }
    config_path = root / "watchdog_config.json"
    write_json(config_path, config)
    log = (root / "watchdog.log").open("ab")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "blueprint_pipeline.aws_independent_watchdog_control",
                "run",
                "--job-dir",
                str(root),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log.close()
    time.sleep(0.05)
    live = process.poll() is None
    handoff = {
        "schema_version": HANDOFF_SCHEMA,
        "status": "armed" if live else "blocked",
        "name_prefix": name_prefix,
        "hard_ttl_seconds": hard_ttl_seconds,
        "pid": process.pid,
        "independent_process": live,
        "watchdog_armed_before_allocation": live,
        "raw_secret_values_recorded": False,
        "blockers": [] if live else ["aws_independent_watchdog_process_not_live"],
    }
    write_json(root / "watchdog_handoff.json", handoff)
    return handoff, process if live else None


def close_aws_watchdog(
    *, job_dir: str | Path, process: subprocess.Popen[bytes] | None
) -> dict[str, Any]:
    root = Path(job_dir).expanduser().resolve()
    (root / "cancel").touch(exist_ok=True)
    if process is not None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=15)
    config = json.loads((root / "watchdog_config.json").read_text(encoding="utf-8"))
    inventory = get_render_provider("aws").billable_inventory(
        name_prefix=str(config["name_prefix"])
    )
    closed = bool(
        inventory.get("api_confirmed") is True
        and inventory.get("live_resource_count") == 0
    )
    receipt = {
        "schema_version": HANDOFF_SCHEMA,
        "status": "provider_terminal" if closed else "blocked",
        "provider_absence_confirmed": closed,
        "inventory": inventory,
        "closed_at": utc_now_iso(),
        "raw_secret_values_recorded": False,
        "blockers": [] if closed else ["aws_independent_watchdog_provider_not_zero"],
    }
    write_json(root / "watchdog_close.json", receipt)
    return receipt


def run_watchdog(job_dir: str | Path) -> int:
    root = Path(job_dir).expanduser().resolve()
    config = json.loads((root / "watchdog_config.json").read_text(encoding="utf-8"))
    provider = get_render_provider("aws")
    while time.time() < float(config["deadline_epoch"]):
        if (root / "cancel").exists():
            write_json(root / "watchdog_terminal.json", {
                "schema_version": HANDOFF_SCHEMA,
                "status": "cancelled_after_controller_teardown",
                "provider_mutations_performed": 0,
                "observed_at": utc_now_iso(),
            })
            return 0
        time.sleep(min(30, max(1, int(config.get("poll_seconds") or 15))))
    inventory = provider.billable_inventory(name_prefix=str(config["name_prefix"]))
    terminated: list[str] = []
    for row in inventory.get("resources") or []:
        instance_id = str(row.get("instance_id") or "")
        result = provider.terminate(instance_id)
        if result.get("status") == "terminated":
            terminated.append(instance_id)
    write_json(root / "watchdog_terminal.json", {
        "schema_version": HANDOFF_SCHEMA,
        "status": "ttl_teardown_requested",
        "terminated_instance_ids": terminated,
        "provider_mutations_performed": len(terminated),
        "observed_at": utc_now_iso(),
    })
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    return run_watchdog(args.job_dir)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["HANDOFF_SCHEMA", "arm_aws_watchdog", "close_aws_watchdog"]
