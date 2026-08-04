"""Print an allowlisted Vast instance summary without provider-secret fields."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Mapping, Sequence


SAFE_INSTANCE_FIELDS = (
    "id",
    "actual_status",
    "cur_state",
    "intended_status",
    "gpu_name",
    "dph_total",
    "start_date",
    "uptime_mins",
    "label",
)


def summarize_instances(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("vast_instance_payload_not_list")
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("vast_instance_payload_row_not_object")
        rows.append({name: item.get(name) for name in SAFE_INSTANCE_FIELDS})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("vast_safe_status_accepts_no_arguments")
    result = subprocess.run(
        ["vastai", "show", "instances", "--raw"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(json.dumps({"status": "blocked", "error": "vast_status_command_failed"}))
        return 2
    try:
        summary = summarize_instances(json.loads(result.stdout))
    except (json.JSONDecodeError, ValueError):
        print(json.dumps({"status": "blocked", "error": "vast_status_payload_invalid"}))
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
