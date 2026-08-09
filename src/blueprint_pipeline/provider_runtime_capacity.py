"""Native provider-runtime capacity admission before large dependency installs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA_VERSION = "provider_runtime_disk_headroom.v1"


def _mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("provider_runtime_capacity_manifest_invalid")
    return dict(value)


def measure_runtime_disk_headroom(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path,
    measurement_path: str | Path,
    disk_usage: Callable[[str | Path], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    """Measure real free bytes against an immutable manifest requirement."""

    manifest = _mapping(Path(manifest_path).expanduser().resolve())
    requirements = manifest.get("runtime_resource_requirements") or {}
    if not isinstance(requirements, Mapping):
        raise ValueError("provider_runtime_capacity_requirements_invalid")
    minimum = requirements.get("minimum_free_bytes_before_dependency_install")
    requested = requirements.get("requested_disk_gb")
    blocker = str(
        requirements.get("failure_blocker")
        or "provider_runtime_disk_headroom_insufficient"
    )
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum <= 0
        or isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested <= 0
    ):
        raise ValueError("provider_runtime_capacity_requirements_invalid")
    measured_path = Path(measurement_path).expanduser().resolve()
    usage = disk_usage(measured_path)
    passed = int(usage.free) >= minimum
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "blocked",
        "measurement_path": str(measured_path),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "minimum_free_bytes": minimum,
        "requested_disk_gb": requested,
        "native_readback": True,
        "blockers": [] if passed else [blocker],
    }
    destination = Path(receipt_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path")
    parser.add_argument("receipt_path")
    parser.add_argument("measurement_path")
    args = parser.parse_args()
    try:
        receipt = measure_runtime_disk_headroom(
            manifest_path=args.manifest_path,
            receipt_path=args.receipt_path,
            measurement_path=args.measurement_path,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return 3
    return 0 if receipt["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "measure_runtime_disk_headroom"]
