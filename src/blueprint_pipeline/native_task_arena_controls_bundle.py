"""Freeze the task-neutral native Arena controls provider bundle."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from .adp009d_native_microcheck_bundle import DEFAULT_IMAGE as QUALIFIED_ADP_IMAGE
from .decision_evidence_contracts import canonical_digest
from .native_articulated_control_plan import (
    materialize_native_articulated_control_plan,
)
from .native_task_arena_bundle import build_native_task_arena_bundle


PROBE_KIND = "native-task-arena-controls"
PROVIDER_BUNDLE_KIND = "native_task_arena"
RESULT_SCHEMA_VERSION = "native_task_arena_control_result.v1"
RESULT_FILENAME = "native_task_arena_control_result.v1.json"

CONTROLS_RUNTIME_MODULE_NAMES = (
    "adp009d_control_episode.py",
    "adp009d_droid_observation.py",
    "adp009d_isaac_episode_adapter.py",
    "adp009d_task_scoring.py",
    "adp_task_scoring.py",
    "decision_evidence_contracts.py",
    "episode_visual_evidence.py",
    "groot_n17_droid_policy_runtime.py",
    "native_articulated_motion_geometry.py",
    "native_articulated_task_state.py",
    "native_franka_action_math.py",
    "native_franka_pose_servo.py",
    "native_pose_transforms.py",
    "native_task_arena_construction_worker.py",
    "native_task_arena_readback.py",
    "native_task_arena_runtime.py",
    "native_task_camera_observability.py",
    "native_task_episode_environment.py",
    "native_task_runtime_source_packet.py",
    "native_task_runtime_source_provision.py",
)


def controls_runtime_sources() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    return tuple(package / name for name in CONTROLS_RUNTIME_MODULE_NAMES)


def _read_mapping(path: Path, *, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(error) from exc
    if not isinstance(value, dict):
        raise ValueError(error)
    return value


def build_native_task_arena_controls_bundle(
    *,
    job_dir: str | Path,
    packet_dir: str | Path,
    construction_result_path: str | Path,
    runtime_source_packet_receipt: str | Path,
    implementation_commit: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Bind a qualified construction receipt to zero and positive controls."""

    packet = Path(packet_dir).expanduser().resolve()
    scene_plan = _read_mapping(
        packet / "native_task_arena_scene_plan.v1.json",
        error="native_task_controls_scene_plan_invalid",
    )
    construction_path = Path(construction_result_path).expanduser().resolve()
    construction = _read_mapping(
        construction_path,
        error="native_task_controls_construction_result_invalid",
    )
    control_plan = materialize_native_articulated_control_plan(
        scene_plan=scene_plan,
        construction_result=construction,
    )
    with tempfile.TemporaryDirectory(prefix="blueprint-native-task-controls-") as raw:
        frozen_plan = Path(raw) / "adp_task_control_plan.v1.json"
        frozen_plan.write_text(
            json.dumps(control_plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return build_native_task_arena_bundle(
            job_dir=job_dir,
            packet_dir=packet,
            runtime_source_packet_receipt=runtime_source_packet_receipt,
            worker_source=(
                Path(__file__).resolve().parent
                / "native_task_arena_controls_worker.py"
            ),
            runtime_module_sources=controls_runtime_sources(),
            implementation_commit=implementation_commit,
            execution_mode="controls",
            expected_output_filename=RESULT_FILENAME,
            container_image=QUALIFIED_ADP_IMAGE,
            bound_runtime_inputs={
                "native_task_arena_construction_result.v1.json": construction_path,
                "adp_task_control_plan.v1.json": frozen_plan,
            },
            generated_at=generated_at,
        )


def load_verified_native_task_arena_controls_bundle(
    receipt_path: str | Path,
    *,
    expected_implementation_commit: str,
    expected_packet_receipt_digest: str | None = None,
    expected_runtime_source_packet_digest: str | None = None,
) -> dict[str, Any]:
    """Reverify the immutable controls bundle without rebuilding its bytes."""

    path = Path(receipt_path).expanduser().resolve()
    receipt = _read_mapping(
        path, error="native_task_arena_controls_bundle_receipt_invalid"
    )
    bundle_path = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    digest = hashlib.sha256()
    try:
        with bundle_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("native_task_arena_controls_bundle_bytes_missing") from exc
    manifest = {
        key: value
        for key, value in receipt.items()
        if key not in {"bundle_path", "bundle_size_bytes", "bundle_sha256"}
    }
    errors: list[str] = []
    input_names = {
        Path(str(row.get("relative_path") or "")).name
        for row in receipt.get("bound_runtime_inputs") or []
        if isinstance(row, dict)
    }
    if (
        receipt.get("schema_version") != "native_task_arena_provider_bundle.v1"
        or receipt.get("status") != "ready"
        or receipt.get("execution_mode") != "controls"
        or receipt.get("policy_candidate_id") is not None
        or receipt.get("candidate_policy_queried") is not False
        or receipt.get("expected_output_filename") != RESULT_FILENAME
        or input_names
        != {
            "native_task_arena_construction_result.v1.json",
            "adp_task_control_plan.v1.json",
        }
    ):
        errors.append("native_task_arena_controls_bundle_contract_invalid")
    if receipt.get("implementation_commit") != expected_implementation_commit:
        errors.append("native_task_arena_controls_bundle_commit_mismatch")
    if receipt.get("container_image") != QUALIFIED_ADP_IMAGE:
        errors.append("native_task_arena_controls_bundle_image_mismatch")
    if expected_packet_receipt_digest and (
        receipt.get("packet_receipt_digest") != expected_packet_receipt_digest
    ):
        errors.append("native_task_arena_controls_bundle_packet_mismatch")
    source_packet = receipt.get("runtime_source_packet") or {}
    if expected_runtime_source_packet_digest and (
        source_packet.get("receipt_digest")
        != expected_runtime_source_packet_digest
    ):
        errors.append("native_task_arena_controls_bundle_sources_mismatch")
    if receipt.get("input_digest") != canonical_digest(
        manifest, digest_field="input_digest"
    ):
        errors.append("native_task_arena_controls_bundle_input_digest_invalid")
    if (
        receipt.get("bundle_size_bytes") != bundle_path.stat().st_size
        or receipt.get("bundle_sha256") != "sha256:" + digest.hexdigest()
    ):
        errors.append("native_task_arena_controls_bundle_bytes_identity_mismatch")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    return receipt


__all__ = [
    "CONTROLS_RUNTIME_MODULE_NAMES",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "RESULT_FILENAME",
    "RESULT_SCHEMA_VERSION",
    "build_native_task_arena_controls_bundle",
    "controls_runtime_sources",
    "load_verified_native_task_arena_controls_bundle",
]
