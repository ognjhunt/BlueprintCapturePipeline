"""Build the complete task-neutral runtime bundle for a Panda construction gate.

The lower-level bundle builder intentionally accepts an explicit module list so
other native task workers can remain small.  This module freezes the dependency
closure for the articulated Panda construction worker in one reusable place.
Scene packets remain data: no scene id, object class, or task coordinate appears
in this module, and the exact sealed packet is copied without reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from .adp009d_native_microcheck_bundle import DEFAULT_IMAGE as QUALIFIED_ADP_IMAGE
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_bundle import build_native_task_arena_bundle
from .native_task_construction_plan import (
    materialize_native_task_construction_phase_plan,
)


PROBE_KIND = "native-task-arena-construction"
PROVIDER_BUNDLE_KIND = "native_task_arena"
RESULT_SCHEMA_VERSION = "native_task_arena_construction_result.v1"

# Import-time closure of native_task_arena_construction_worker.py.  Keep this
# explicit and hermetically import-tested: provider startup may not discover
# missing internal modules one at a time.
CONSTRUCTION_RUNTIME_MODULE_NAMES = (
    "articulation_graph_contract.py",
    "articulated_control_planner.py",
    "decision_evidence_contracts.py",
    "native_articulated_construction_plan.py",
    "native_articulated_motion_geometry.py",
    "native_articulated_task_state.py",
    "native_task_construction_plan.py",
    "native_franka_pose_servo.py",
    "native_franka_action_math.py",
    "native_pose_transforms.py",
    "native_task_arena_readback.py",
    "native_task_arena_device_readback.py",
    "native_task_arena_import_scope.py",
    "native_task_arena_preconstruction.py",
    "native_task_arena_runtime.py",
    "native_task_isaaclab_launch.py",
    "native_task_camera_observability.py",
    "native_task_runtime_source_packet.py",
    "native_task_runtime_source_provision.py",
)


def construction_runtime_sources() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    return tuple(package / name for name in CONSTRUCTION_RUNTIME_MODULE_NAMES)


def build_native_task_arena_construction_bundle(
    *,
    job_dir: str | Path,
    packet_dir: str | Path,
    runtime_source_packet_receipt: str | Path,
    implementation_commit: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Package one sealed task packet for the native Panda construction worker."""

    package = Path(__file__).resolve().parent
    packet = Path(packet_dir).expanduser().resolve()
    scene_plan = json.loads(
        (packet / "native_task_arena_scene_plan.v1.json").read_text(
            encoding="utf-8"
        )
    )
    # Historical transport-only fixtures intentionally carry no executable
    # scene-plan schema. Every real native packet must freeze this local plan
    # before a provider can be allocated.
    if scene_plan.get("schema_version") != "native_task_arena_scene_plan.v1":
        return build_native_task_arena_bundle(
            job_dir=job_dir,
            packet_dir=packet,
            runtime_source_packet_receipt=runtime_source_packet_receipt,
            worker_source=package / "native_task_arena_construction_worker.py",
            runtime_module_sources=construction_runtime_sources(),
            implementation_commit=implementation_commit,
            execution_mode="construction_canary",
            container_image=QUALIFIED_ADP_IMAGE,
            generated_at=generated_at,
        )
    frozen = materialize_native_task_construction_phase_plan(scene_plan)
    with tempfile.TemporaryDirectory(prefix="blueprint-native-construction-plan-") as raw:
        phase_path = Path(raw) / "native_task_construction_phase_plan.v1.json"
        phase_path.write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return build_native_task_arena_bundle(
            job_dir=job_dir,
            packet_dir=packet,
            runtime_source_packet_receipt=runtime_source_packet_receipt,
            worker_source=package / "native_task_arena_construction_worker.py",
            runtime_module_sources=construction_runtime_sources(),
            implementation_commit=implementation_commit,
            execution_mode="construction_canary",
            container_image=QUALIFIED_ADP_IMAGE,
            bound_runtime_inputs={phase_path.name: phase_path},
            generated_at=generated_at,
        )


def load_verified_native_task_arena_construction_bundle(
    receipt_path: str | Path,
    *,
    expected_implementation_commit: str,
    expected_packet_receipt_digest: str | None = None,
    expected_runtime_source_packet_digest: str | None = None,
) -> dict[str, Any]:
    """Reverify an already dry-run bundle without rebuilding or changing its SHA."""

    path = Path(receipt_path).expanduser().resolve()
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("native_task_arena_bundle_receipt_invalid") from exc
    if not isinstance(receipt, dict):
        raise ValueError("native_task_arena_bundle_receipt_invalid")
    bundle_path = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    manifest = {
        key: value
        for key, value in receipt.items()
        if key not in {"bundle_path", "bundle_size_bytes", "bundle_sha256"}
    }
    digest = hashlib.sha256()
    try:
        with bundle_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("native_task_arena_bundle_bytes_missing") from exc
    errors = []
    if (
        receipt.get("schema_version") != "native_task_arena_provider_bundle.v1"
        or receipt.get("status") != "ready"
        or receipt.get("execution_mode") != "construction_canary"
        or receipt.get("policy_candidate_id") is not None
        or receipt.get("candidate_policy_queried") is not False
    ):
        errors.append("native_task_arena_bundle_receipt_contract_invalid")
    if receipt.get("implementation_commit") != expected_implementation_commit:
        errors.append("native_task_arena_bundle_implementation_commit_mismatch")
    if receipt.get("container_image") != QUALIFIED_ADP_IMAGE:
        errors.append("native_task_arena_bundle_container_image_mismatch")
    if expected_packet_receipt_digest and (
        receipt.get("packet_receipt_digest") != expected_packet_receipt_digest
    ):
        errors.append("native_task_arena_bundle_packet_receipt_mismatch")
    source_packet = receipt.get("runtime_source_packet") or {}
    if not source_packet or source_packet.get("redistribution_permitted") is not True:
        errors.append("native_task_arena_bundle_runtime_source_packet_missing")
    if expected_runtime_source_packet_digest and (
        source_packet.get("receipt_digest") != expected_runtime_source_packet_digest
    ):
        errors.append("native_task_arena_bundle_runtime_source_packet_mismatch")
    if receipt.get("input_digest") != canonical_digest(
        manifest, digest_field="input_digest"
    ):
        errors.append("native_task_arena_bundle_input_digest_invalid")
    if (
        receipt.get("bundle_size_bytes") != bundle_path.stat().st_size
        or receipt.get("bundle_sha256") != "sha256:" + digest.hexdigest()
    ):
        errors.append("native_task_arena_bundle_bytes_identity_mismatch")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    return receipt


__all__ = [
    "CONSTRUCTION_RUNTIME_MODULE_NAMES",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "RESULT_SCHEMA_VERSION",
    "build_native_task_arena_construction_bundle",
    "construction_runtime_sources",
    "load_verified_native_task_arena_construction_bundle",
]
