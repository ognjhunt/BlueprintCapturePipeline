"""Record the worker facts the policy-server design could not verify off-worker.

Three questions were left open because answering them needs the container:
what Isaac's own interpreter is and where it lives, which torch it ships, and
how much of the GPU is already committed before a policy is co-resident.  Each
one decides something concrete -- whether a policy venv can be built beside
Isaac, whether a policy's torch pin collides with Isaac's, and whether a
co-resident server has room at all.

Runs under ``/isaac-sim/python.sh`` before anything else, so a later failure
cannot erase the answers.  Every probe is individually guarded: a missing torch
or an unavailable device must still produce the other facts.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = "adp009d_worker_environment_facts.v1"


def collect_facts() -> dict[str, object]:
    facts: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "isaac_python_executable": sys.executable,
        "isaac_python_version": platform.python_version(),
        "isaac_sys_prefix": sys.prefix,
        "isaac_sys_path_entry_count": len(sys.path),
        # Only the prefixes that decide interpreter isolation; never the values,
        # which can carry credentials.
        "isaac_environment_keys": sorted(
            key
            for key in os.environ
            if key.startswith(("ISAAC", "OMNI", "CARB", "LD_LIBRARY", "PYTHONPATH"))
        ),
    }

    try:
        import torch

        facts["torch_version"] = torch.__version__
        facts["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        facts["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            facts["gpu_name"] = torch.cuda.get_device_name(0)
            facts["gpu_total_bytes"] = int(total_bytes)
            facts["gpu_free_bytes_before_policy"] = int(free_bytes)
            facts["gpu_used_fraction_before_policy"] = round(
                1.0 - (free_bytes / total_bytes), 6
            )
    except Exception as exc:  # noqa: BLE001 - a missing torch is itself a fact
        facts["torch_error"] = f"{type(exc).__name__}: {exc}"

    # Whether a separate interpreter exists to build the policy venv from.
    try:
        completed = subprocess.run(
            [
                "python3",
                "-c",
                "import sys; print(sys.executable); print(sys.version.split()[0])",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        facts["system_python3_executable"] = lines[0] if lines else None
        facts["system_python3_version"] = lines[1] if len(lines) > 1 else None
        facts["system_python3_distinct_from_isaac"] = bool(
            lines and lines[0] != sys.executable
        )
    except Exception as exc:  # noqa: BLE001
        facts["system_python3_error"] = f"{type(exc).__name__}: {exc}"

    return facts


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise SystemExit("usage: adp009d_worker_environment_facts.py OUTPUT_DIR")
    output = Path(arguments[0])
    output.mkdir(parents=True, exist_ok=True)
    (output / "adp009d_worker_environment_facts.json").write_text(
        json.dumps(collect_facts(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
