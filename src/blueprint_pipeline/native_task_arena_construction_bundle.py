"""Build the complete task-neutral runtime bundle for a Panda construction gate.

The lower-level bundle builder intentionally accepts an explicit module list so
other native task workers can remain small.  This module freezes the dependency
closure for the articulated Panda construction worker in one reusable place.
Scene packets remain data: no scene id, object class, or task coordinate appears
in this module, and the exact sealed packet is copied without reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .native_task_arena_bundle import build_native_task_arena_bundle


PROBE_KIND = "native-task-arena-construction"
PROVIDER_BUNDLE_KIND = "native_task_arena"
RESULT_SCHEMA_VERSION = "native_task_arena_construction_result.v1"

# Import-time closure of native_task_arena_construction_worker.py.  Keep this
# explicit and hermetically import-tested: provider startup may not discover
# missing internal modules one at a time.
CONSTRUCTION_RUNTIME_MODULE_NAMES = (
    "articulated_control_planner.py",
    "decision_evidence_contracts.py",
    "native_articulated_construction_plan.py",
    "native_articulated_motion_geometry.py",
    "native_articulated_task_state.py",
    "native_franka_pose_servo.py",
    "native_franka_action_math.py",
    "native_pose_transforms.py",
    "native_task_arena_readback.py",
    "native_task_arena_runtime.py",
    "native_task_camera_observability.py",
)


def construction_runtime_sources() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    return tuple(package / name for name in CONSTRUCTION_RUNTIME_MODULE_NAMES)


def build_native_task_arena_construction_bundle(
    *,
    job_dir: str | Path,
    packet_dir: str | Path,
    implementation_commit: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Package one sealed task packet for the native Panda construction worker."""

    package = Path(__file__).resolve().parent
    return build_native_task_arena_bundle(
        job_dir=job_dir,
        packet_dir=packet_dir,
        worker_source=package / "native_task_arena_construction_worker.py",
        runtime_module_sources=construction_runtime_sources(),
        implementation_commit=implementation_commit,
        execution_mode="construction_canary",
        generated_at=generated_at,
    )


__all__ = [
    "CONSTRUCTION_RUNTIME_MODULE_NAMES",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "RESULT_SCHEMA_VERSION",
    "build_native_task_arena_construction_bundle",
    "construction_runtime_sources",
]
