"""Read-only Linux host admission before a G1 container episode.

The staged container plan checks immutable inputs. This check covers the host
resources that are only knowable on the machine which will run Docker. It does
not allocate a provider, start the image, or claim a policy episode occurred.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


SCHEMA = "native_g1_container_host_readiness.v1"
MIN_DRIVER = (580, 95, 5)
# nvidia-smi reports usable MiB, slightly less than nominal 16 GB devices.
MIN_GPU_MEMORY_MIB = 16_000
MIN_FREE_DISK_BYTES = 16 * 1024**3


def _read_command(argv: list[str]) -> str:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("g1_container_host_probe_unavailable") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("g1_container_host_probe_failed")
    return result.stdout.strip()


def verify_g1_container_host(*, output_dir: Path) -> dict[str, Any]:
    """Require GPU 0 and the exact local image before Docker execution."""

    if sys.platform != "linux" or not output_dir.is_absolute() or not output_dir.is_dir():
        raise ValueError("g1_container_execution_host_invalid")
    gpu_text = _read_command([
        "nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])
    rows = list(csv.reader(gpu_text.splitlines()))
    gpu_zero = [row for row in rows if len(row) == 4 and row[0].strip() == "0"]
    if len(gpu_zero) != 1:
        raise ValueError("g1_container_gpu_zero_missing")
    _, name, driver, memory = [field.strip() for field in gpu_zero[0]]
    if not name or re.fullmatch(r"\d+\.\d+\.\d+", driver) is None:
        raise ValueError("g1_container_gpu_identity_invalid")
    # Isaac Sim 6 explicitly excludes A100/H100 because they lack RT cores.
    # Other GPU names remain only a probe; the container must still prove it
    # can initialize RTX rendering on this exact device.
    if re.search(r"\b(?:A100|H100)\b", name, flags=re.IGNORECASE):
        raise ValueError("g1_container_gpu_without_rt_cores")
    if tuple(int(part) for part in driver.split(".")) < MIN_DRIVER:
        raise ValueError("g1_container_driver_below_tested_isaac_floor")
    if not memory.isdecimal() or int(memory) < MIN_GPU_MEMORY_MIB:
        raise ValueError("g1_container_gpu_memory_insufficient")

    try:
        image_id = _read_command([
            "docker", "image", "inspect", "--format", "{{.Id}}", NATIVE_TASK_ARENA_IMAGE,
        ])
    except ValueError as exc:
        raise ValueError("g1_container_pinned_image_unavailable") from exc
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise ValueError("g1_container_local_image_identity_invalid")
    free_disk = shutil.disk_usage(output_dir).free
    if free_disk < MIN_FREE_DISK_BYTES:
        raise ValueError("g1_container_free_disk_insufficient")

    receipt = {
        "schema_version": SCHEMA,
        "status": "host_ready_for_container_attempt",
        "gpu_index": 0,
        "gpu_name": name,
        "driver_version": driver,
        "gpu_memory_mib": int(memory),
        "image_reference": NATIVE_TASK_ARENA_IMAGE,
        "local_image_id": image_id,
        "free_disk_bytes": free_disk,
        "rt_core_compatibility_verified": False,
        "container_started": False,
        "episode_executed": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def record_g1_container_host(*, output_dir: Path) -> dict[str, Any]:
    """Persist the pre-execution host observation beside the staged plan."""

    receipt = verify_g1_container_host(output_dir=output_dir)
    (output_dir / (SCHEMA + ".json")).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt
