"""Strict adapter from UNITREE_G1_SONIC actions to official GEAR-SONIC WBC/FK.

The adapter does not implement a substitute controller. It invokes the
protocol-v4 client for the official ``/opt/wbc`` deployment stack and turns its numeric result
into the attested contract consumed by ``make_controller_fk_skeleton_projector``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso
from .gear_sonic_joint_order_contract import (
    JOINT_ORDER_SCHEMA_VERSION,
    PROTOCOL_V4_FULL_JOINT_ORDER,
    PROTOCOL_V4_MAPPING_DIGEST,
    controller_frame_sequence_start,
    validate_full_joint_order,
)
from .oscar_isaac_closed_loop_eval import build_sc3_runtime_attestation


ROOT_ENV = "BLUEPRINT_GEAR_SONIC_ROOT"
EXECUTOR_COMMAND_ENV = "BLUEPRINT_GEAR_SONIC_EXECUTOR_COMMAND"
ROBOT_MODEL_ENV = "BLUEPRINT_GEAR_SONIC_ROBOT_MODEL"
SIGNING_KEY_ENV = "BLUEPRINT_SC3_FK_EXECUTOR_PRIVATE_KEY_FILE"
DEFAULT_ROOT = "/opt/wbc"
DEFAULT_ROBOT_MODEL = "/opt/wbc/gear_sonic_deploy/g1/g1_29dof_with_hand.xml"
SCHEMA_VERSION = "gear_sonic_controller_fk_execution.v1"
CONTROLLER_ACTION_SEQUENCE_SCHEMA_VERSION = "gear_sonic_controller_action_sequence.v1"
CONTROLLER_EXECUTION_SCHEMA_VERSION = "gear_sonic_controller_horizon_execution.v1"
EXECUTOR_TIMEOUT_SECONDS = 180
EXECUTOR_LOG_MAX_CHARS = 32 * 1024
CONTROLLER_RUNTIME_ARTIFACT_RELATIVE_PATHS: tuple[str, ...] = (
    "gear_sonic_deploy/target/release/g1_deploy_onnx_ref",
    "gear_sonic_deploy/policy/release/model_decoder.onnx",
    "gear_sonic_deploy/policy/release/model_encoder.onnx",
    "gear_sonic_deploy/policy/release/observation_config.yaml",
    "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
    "gear_sonic_deploy/g1/g1_29dof_with_hand.xml",
)
_URL_QUERY_RE = re.compile(r"(https?://[^\s?#]+)\?[^\s]+", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(
    r"(?i)((?:[a-z0-9_-]*(?:authorization|api[_-]?key|access[_-]?key|"
    r"secret(?:_access)?[_-]?key|token|password))\s*[:=]\s*"
    r"(?:bearer\s+)?)[^\s,;]+"
)
_SENSITIVE_ENV_NAME_MARKERS = (
    "API_KEY",
    "ACCESS_KEY",
    "AUTHORIZATION",
    "PASSWORD",
    "SECRET",
    "SIGNED_URL",
    "TOKEN",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _numeric_vector(value: Any, *, name: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name}_missing")
    result = [float(item) for item in value]
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name}_nonfinite_or_empty")
    return result


def _action_execution_frames(
    action: Mapping[str, Any], selected_vector: Sequence[float]
) -> tuple[list[list[float]], dict[str, Any], bool]:
    raw_contract = action.get("controller_action")
    selected = [float(item) for item in selected_vector]
    if not isinstance(raw_contract, Mapping):
        frames = [selected]
        return (
            frames,
            {
                "execution_mode": "single_frame_receding_horizon",
                "execution_frame_count": 1,
                "source_horizon_frame_count": 1,
                "frame_dimension": len(selected),
                "control_hz": 50.0,
                "frames_sha256": _canonical(frames),
                "source_frames_sha256": _canonical(frames),
            },
            False,
        )
    contract = dict(raw_contract)
    if contract.get("schema_version") != CONTROLLER_ACTION_SEQUENCE_SCHEMA_VERSION:
        raise ValueError("unitree_g1_sonic_controller_action_sequence_schema_mismatch")
    raw_frames = contract.get("frames")
    if isinstance(raw_frames, (str, bytes, bytearray)) or not isinstance(
        raw_frames, Sequence
    ):
        raise ValueError("unitree_g1_sonic_controller_action_sequence_missing")
    frames = [
        _numeric_vector(
            frame, name=f"unitree_g1_sonic_controller_action_frame_{index}"
        )
        for index, frame in enumerate(raw_frames)
    ]
    if (
        not frames
        or any(len(frame) != len(selected) for frame in frames)
        or int(contract.get("execution_frame_count") or 0) != len(frames)
        or int(contract.get("source_horizon_frame_count") or 0) < len(frames)
        or int(contract.get("frame_dimension") or 0) != len(selected)
        or frames[0] != selected
    ):
        raise ValueError("unitree_g1_sonic_controller_action_sequence_shape_invalid")
    if str(contract.get("frames_sha256") or "") != _canonical(frames):
        raise ValueError("unitree_g1_sonic_controller_action_sequence_sha256_mismatch")
    control_hz = float(contract.get("control_hz") or 0.0)
    if not math.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("unitree_g1_sonic_controller_action_sequence_timing_invalid")
    return frames, contract, True


def _bounded_redacted_executor_log(value: str | bytes | None) -> tuple[str, bool]:
    """Return a secret-scrubbed diagnostic that retains both failure ends."""

    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value or ""
    sensitive_values = sorted(
        {
            str(environment_value)
            for environment_name, environment_value in os.environ.items()
            if environment_value
            and len(str(environment_value)) >= 6
            and any(marker in environment_name.upper() for marker in _SENSITIVE_ENV_NAME_MARKERS)
        },
        key=len,
        reverse=True,
    )
    for sensitive_value in sensitive_values:
        text = text.replace(sensitive_value, "[REDACTED]")
    text = _URL_QUERY_RE.sub(r"\1?[REDACTED_QUERY]", text)
    text = _SECRET_VALUE_RE.sub(r"\1[REDACTED]", text)
    if len(text) <= EXECUTOR_LOG_MAX_CHARS:
        return text, False
    marker = "\n...[executor diagnostic truncated]...\n"
    remaining = EXECUTOR_LOG_MAX_CHARS - len(marker)
    head_chars = remaining // 3
    tail_chars = remaining - head_chars
    return text[:head_chars] + marker + text[-tail_chars:], True


def _executor_failure_summary(stderr: str, stdout: str) -> str | None:
    for diagnostic in (stderr, stdout):
        lines = [line.strip() for line in diagnostic.splitlines() if line.strip()]
        if lines:
            return lines[-1][:512]
    return None


def _write_executor_diagnostics(
    *,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    returncode: int | None,
    timed_out: bool,
    raw_output: Path,
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
) -> str | None:
    safe_stdout, stdout_truncated = _bounded_redacted_executor_log(stdout)
    safe_stderr, stderr_truncated = _bounded_redacted_executor_log(stderr)
    stdout_path.write_text(safe_stdout, encoding="utf-8")
    stderr_path.write_text(safe_stderr, encoding="utf-8")
    failure_summary = _executor_failure_summary(safe_stderr, safe_stdout)
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "gear_sonic_executor_command_result.v1",
                "status": "timed_out" if timed_out else (
                    "completed" if returncode == 0 else "failed"
                ),
                "returncode": returncode,
                "timed_out": timed_out,
                "timeout_seconds": EXECUTOR_TIMEOUT_SECONDS,
                "failure_summary": failure_summary,
                "stdout_log": str(stdout_path.resolve()),
                "stderr_log": str(stderr_path.resolve()),
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "output_path": str(raw_output.resolve()),
                "output_present": raw_output.is_file(),
                "diagnostics_redacted": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return failure_summary


def _controller_tree_manifest(root: Path, output: Path) -> tuple[str, dict[str, Any]]:
    """Hash the exact controller runtime closure, never the multi-GB checkout."""

    rows: list[dict[str, Any]] = []
    resolved_root = root.resolve()
    for relative_path in CONTROLLER_RUNTIME_ARTIFACT_RELATIVE_PATHS:
        candidate = resolved_root / relative_path
        path = candidate.resolve()
        if (
            path == resolved_root
            or resolved_root not in path.parents
            or candidate.is_symlink()
            or not path.is_file()
        ):
            raise ValueError(
                f"official_gear_sonic_controller_runtime_artifact_missing:{relative_path}"
            )
        rows.append(
            {
                "relative_path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": "gear_sonic_controller_runtime_manifest.v1",
        "root_name": resolved_root.name,
        "artifact_scope": "exact_runtime_binary_models_config_and_robot_xml",
        "files": rows,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(output), {"path": str(output.resolve()), "sha256": _sha256(output)}


def run_adapter(*, input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    request_path = Path(input_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, Mapping) or request.get("schema_version") != "controller_fk_skeleton_request.v1":
        raise ValueError("controller_fk_request_schema_mismatch")
    action = dict(request.get("action") or {})
    action_sha = _canonical(action)
    if action_sha != str(request.get("source_action_sha256") or ""):
        raise ValueError("controller_fk_request_action_sha256_mismatch")
    vector = _numeric_vector(
        action.get("sonic_action_chunk") or action.get("action_chunk"),
        name="unitree_g1_sonic_action",
    )
    units = action.get("action_units")
    if (
        isinstance(units, (str, bytes, bytearray))
        or not isinstance(units, Sequence)
        or len(units) != len(vector)
        or any(not str(item).strip() for item in units)
    ):
        raise ValueError("unitree_g1_sonic_action_units_missing_or_dimension_mismatch")
    timing = dict(action.get("action_timing") or {})
    control_hz = float(timing.get("control_hz") or 0.0)
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("unitree_g1_sonic_action_timing_missing_or_invalid")
    execution_frames, requested_execution, explicit_execution_sequence = (
        _action_execution_frames(action, vector)
    )

    root = Path(os.environ.get(ROOT_ENV, DEFAULT_ROOT)).expanduser().resolve()
    if root.name != "wbc" or not (root / "gear_sonic_deploy").is_dir():
        raise ValueError("official_gear_sonic_root_missing_or_invalid")
    model = Path(os.environ.get(ROBOT_MODEL_ENV, DEFAULT_ROBOT_MODEL)).expanduser().resolve()
    if not model.is_file() or root not in model.parents:
        raise ValueError("official_gear_sonic_robot_model_missing_or_outside_root")
    argv = shlex.split(os.environ.get(EXECUTOR_COMMAND_ENV, ""))
    if not argv:
        raise ValueError("official_gear_sonic_executor_command_missing")
    raw_output = target.with_name("gear_sonic_raw_output.json")
    executor_stdout = target.with_name("gear_sonic_executor_stdout.log")
    executor_stderr = target.with_name("gear_sonic_executor_stderr.log")
    executor_result = target.with_name("gear_sonic_executor_command_result.json")
    for stale_path in (raw_output, executor_stdout, executor_stderr, executor_result):
        stale_path.unlink(missing_ok=True)
    try:
        completed = subprocess.run(
            argv,
            cwd=str(root),
            env={
                **os.environ,
                "BLUEPRINT_GEAR_SONIC_INPUT": str(request_path),
                "BLUEPRINT_GEAR_SONIC_OUTPUT": str(raw_output),
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=EXECUTOR_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        failure_summary = _write_executor_diagnostics(
            stdout=error.stdout,
            stderr=error.stderr,
            returncode=None,
            timed_out=True,
            raw_output=raw_output,
            stdout_path=executor_stdout,
            stderr_path=executor_stderr,
            result_path=executor_result,
        )
        detail = f":{failure_summary}" if failure_summary else ""
        raise RuntimeError(
            f"official_gear_sonic_executor_timeout_{EXECUTOR_TIMEOUT_SECONDS}{detail}"
        ) from error
    failure_summary = _write_executor_diagnostics(
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=int(completed.returncode),
        timed_out=False,
        raw_output=raw_output,
        stdout_path=executor_stdout,
        stderr_path=executor_stderr,
        result_path=executor_result,
    )
    if completed.returncode != 0:
        detail = f":{failure_summary}" if failure_summary else ""
        raise RuntimeError(
            f"official_gear_sonic_executor_returncode_{completed.returncode}{detail}"
        )
    raw = json.loads(raw_output.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("status") != "completed":
        raise RuntimeError("official_gear_sonic_executor_result_not_completed")
    if str(raw.get("source_action_sha256") or "") != action_sha:
        raise RuntimeError("official_gear_sonic_executor_action_sha256_mismatch")
    proprioceptive_state = dict(raw.get("proprioceptive_state") or {})
    if proprioceptive_state.get("official_controller_protocol") != 4:
        raise RuntimeError("official_gear_sonic_protocol_v4_evidence_missing")
    landmarks = list(raw.get("landmarks") or [])
    if not landmarks or any(not isinstance(row, Mapping) for row in landmarks):
        raise RuntimeError("official_gear_sonic_fk_landmarks_missing")
    projection_context_sha256 = str(
        raw.get("camera_projection_context_sha256") or ""
    ).lower()
    source_frame_sha256 = str(raw.get("camera_source_frame_sha256") or "").lower()
    registration = dict(raw.get("cross_simulator_registration") or {})
    if not _is_sha256(projection_context_sha256):
        raise RuntimeError("official_gear_sonic_projection_context_sha256_invalid")
    if not _is_sha256(source_frame_sha256):
        raise RuntimeError("official_gear_sonic_projection_source_frame_sha256_invalid")
    if (
        registration.get("status") != "passed"
        or registration.get("surrogate") is not False
    ):
        raise RuntimeError("official_gear_sonic_cross_simulator_registration_not_proven")
    for index, landmark in enumerate(landmarks):
        projection = dict(landmark.get("image_projection") or {})
        if (
            str(projection.get("projection_context_sha256") or "").lower()
            != projection_context_sha256
            or str(projection.get("source_frame_sha256") or "").lower()
            != source_frame_sha256
        ):
            raise RuntimeError(
                f"official_gear_sonic_landmark_projection_binding_invalid:{index}"
            )
    joint_positions = _numeric_vector(raw.get("joint_positions"), name="gear_sonic_joint_positions")
    joint_names = [str(item) for item in raw.get("joint_names") or []]
    if len(joint_names) != len(joint_positions) or any(not item for item in joint_names):
        raise RuntimeError("official_gear_sonic_joint_names_missing_or_dimension_mismatch")
    if str(raw.get("joint_order_schema_version") or "") != JOINT_ORDER_SCHEMA_VERSION:
        raise RuntimeError(
            "official_gear_sonic_executor_joint_order_schema_missing_or_unsupported"
        )
    if str(raw.get("mapping_digest") or "") != PROTOCOL_V4_MAPPING_DIGEST:
        raise RuntimeError("official_gear_sonic_executor_mapping_digest_missing_or_mismatch")
    validate_full_joint_order(joint_names, source="executor")
    if len(joint_positions) != len(PROTOCOL_V4_FULL_JOINT_ORDER):
        raise RuntimeError("official_gear_sonic_executor_joint_positions_dimension_invalid")
    applied_dof_mapping = list(raw.get("applied_dof_mapping") or [])
    if len(applied_dof_mapping) != len(PROTOCOL_V4_FULL_JOINT_ORDER) or any(
        not isinstance(row, Mapping) or str(row.get("joint_name") or "") != joint_names[index]
        for index, row in enumerate(applied_dof_mapping)
    ):
        raise RuntimeError("official_gear_sonic_executor_applied_dof_mapping_missing_or_invalid")
    runtime_result_id = str(raw.get("runtime_result_id") or "").strip()
    if not runtime_result_id:
        raise RuntimeError("official_gear_sonic_runtime_result_id_missing")

    raw_sequence = raw.get("controller_fk_sequence")
    if isinstance(raw_sequence, (str, bytes, bytearray, Mapping)):
        raise RuntimeError("official_gear_sonic_controller_fk_sequence_invalid")
    if isinstance(raw_sequence, Sequence):
        controller_fk_sequence = [
            dict(row) for row in raw_sequence if isinstance(row, Mapping)
        ]
        if len(controller_fk_sequence) != len(execution_frames):
            raise RuntimeError(
                "official_gear_sonic_controller_fk_sequence_count_mismatch"
            )
        controller_fk_sequence_sha256 = _canonical(controller_fk_sequence)
        if (
            str(raw.get("controller_fk_sequence_sha256") or "")
            != controller_fk_sequence_sha256
        ):
            raise RuntimeError(
                "official_gear_sonic_controller_fk_sequence_sha256_mismatch"
            )
        raw_execution = dict(raw.get("execution_contract") or {})
        if (
            raw_execution.get("schema_version")
            != CONTROLLER_EXECUTION_SCHEMA_VERSION
            or int(raw_execution.get("controller_session_count") or 0) != 1
            or int(raw_execution.get("execution_frame_count") or 0)
            != len(execution_frames)
            or int(raw_execution.get("source_horizon_frame_count") or 0)
            != int(requested_execution["source_horizon_frame_count"])
            or int(raw_execution.get("frame_dimension") or 0) != len(vector)
            or abs(
                float(raw_execution.get("control_hz") or 0.0)
                - float(requested_execution["control_hz"])
            )
            > 1e-9
            or str(raw_execution.get("input_action_frames_sha256") or "")
            != str(requested_execution["frames_sha256"])
            or str(raw_execution.get("source_action_frames_sha256") or "")
            != str(requested_execution["source_frames_sha256"])
            or str(raw_execution.get("controller_fk_sequence_sha256") or "")
            != controller_fk_sequence_sha256
            or not _is_sha256(
                raw_execution.get("controller_state_sequence_sha256")
            )
            or str(raw_execution.get("final_controller_fk_frame_sha256") or "")
            != _canonical(controller_fk_sequence[-1])
        ):
            raise RuntimeError(
                "official_gear_sonic_controller_execution_contract_mismatch"
            )
        execution_contract = raw_execution
    else:
        if explicit_execution_sequence or len(execution_frames) != 1:
            raise RuntimeError("official_gear_sonic_controller_fk_sequence_missing")
        controller_fk_sequence = [
            {
                "horizon_frame_index": 0,
                "controller_frame_index": int(request.get("step_index") or 0),
                "source_action_frame_sha256": _canonical(execution_frames[0]),
                "controller_state_sha256": _canonical(
                    {
                        "proprioceptive_state": proprioceptive_state,
                        "state_timestamp": raw.get("state_timestamp"),
                    }
                ),
                "command_send_offset_seconds": None,
                "joint_positions": joint_positions,
                "joint_names": joint_names,
                "applied_dof_mapping": applied_dof_mapping,
                "landmarks": landmarks,
                "proprioceptive_state": proprioceptive_state,
                "state_timestamp": raw.get("state_timestamp"),
            }
        ]
        controller_fk_sequence_sha256 = _canonical(controller_fk_sequence)
        execution_contract = {
            "schema_version": CONTROLLER_EXECUTION_SCHEMA_VERSION,
            "execution_mode": "single_frame_receding_horizon",
            "controller_session_count": 1,
            "execution_frame_count": 1,
            "source_horizon_frame_count": 1,
            "frame_dimension": len(vector),
            "control_hz": control_hz,
            "sample_period_seconds": 1.0 / control_hz,
            "declared_execution_duration_seconds": 1.0 / control_hz,
            "input_action_frames_sha256": _canonical(execution_frames),
            "source_action_frames_sha256": _canonical(execution_frames),
            "controller_fk_sequence_sha256": controller_fk_sequence_sha256,
            "legacy_single_frame_executor_sequence_synthesized": True,
        }

    controller_frame_start = controller_frame_sequence_start(
        outer_step_index=int(request.get("step_index") or 0),
        source_horizon_frame_count=int(
            requested_execution["source_horizon_frame_count"]
        ),
        explicit_horizon=explicit_execution_sequence,
    )
    for index, (frame, row) in enumerate(
        zip(execution_frames, controller_fk_sequence)
    ):
        if (
            "horizon_frame_index" not in row
            or int(row.get("horizon_frame_index") or 0) != index
            or int(row.get("controller_frame_index") or -1)
            != controller_frame_start + index
            or str(row.get("source_action_frame_sha256") or "")
            != _canonical(frame)
            or not _is_sha256(row.get("controller_state_sha256"))
        ):
            raise RuntimeError(
                f"official_gear_sonic_controller_fk_sequence_binding_invalid:{index}"
            )
        row_positions = _numeric_vector(
            row.get("joint_positions"),
            name=f"gear_sonic_joint_positions_frame_{index}",
        )
        row_names = [str(item) for item in row.get("joint_names") or []]
        if len(row_positions) != len(PROTOCOL_V4_FULL_JOINT_ORDER):
            raise RuntimeError(
                f"official_gear_sonic_controller_fk_sequence_dimension_invalid:{index}"
            )
        validate_full_joint_order(row_names, source=f"executor_frame_{index}")
        row_mapping = list(row.get("applied_dof_mapping") or [])
        if len(row_mapping) != len(row_names) or any(
            not isinstance(mapping, Mapping)
            or str(mapping.get("joint_name") or "") != row_names[mapping_index]
            for mapping_index, mapping in enumerate(row_mapping)
        ):
            raise RuntimeError(
                f"official_gear_sonic_controller_fk_sequence_mapping_invalid:{index}"
            )
        row_proprio = dict(row.get("proprioceptive_state") or {})
        if row_proprio.get("official_controller_protocol") != 4:
            raise RuntimeError(
                f"official_gear_sonic_controller_fk_sequence_protocol_invalid:{index}"
            )
        row_landmarks = list(row.get("landmarks") or [])
        if not row_landmarks or any(
            not isinstance(landmark, Mapping) for landmark in row_landmarks
        ):
            raise RuntimeError(
                f"official_gear_sonic_controller_fk_sequence_landmarks_invalid:{index}"
            )
        for landmark_index, landmark in enumerate(row_landmarks):
            projection = dict(landmark.get("image_projection") or {})
            if (
                str(projection.get("projection_context_sha256") or "").lower()
                != projection_context_sha256
                or str(projection.get("source_frame_sha256") or "").lower()
                != source_frame_sha256
            ):
                raise RuntimeError(
                    "official_gear_sonic_controller_fk_sequence_projection_invalid:"
                    f"{index}:{landmark_index}"
                )
    final_sequence_frame = controller_fk_sequence[-1]
    if (
        final_sequence_frame["joint_positions"] != joint_positions
        or final_sequence_frame["joint_names"] != joint_names
        or final_sequence_frame["landmarks"] != landmarks
    ):
        raise RuntimeError("official_gear_sonic_final_state_sequence_mismatch")

    controller_sha, controller_ref = _controller_tree_manifest(
        root, target.with_name("gear_sonic_controller_tree_manifest.json")
    )
    model_sha = _sha256(model)
    state = {
        "source_action_sha256": action_sha,
        "proxy_or_surrogate": False,
        "joint_positions": joint_positions,
        "joint_names": joint_names,
        "joint_order_schema_version": JOINT_ORDER_SCHEMA_VERSION,
        "mapping_digest": PROTOCOL_V4_MAPPING_DIGEST,
        "applied_dof_mapping": applied_dof_mapping,
        "controller_revision": str(raw.get("controller_revision") or ""),
        "proprioceptive_state": proprioceptive_state,
        "state_timestamp": raw.get("state_timestamp"),
        "executed_control_frame_count": len(controller_fk_sequence),
        "controller_fk_sequence": controller_fk_sequence,
        "controller_fk_sequence_sha256": controller_fk_sequence_sha256,
        "controller_execution_contract": execution_contract,
    }
    action_contract = {
        "command": "UNITREE_G1_SONIC",
        "dimension": len(vector),
        "values_sha256": _canonical(vector),
        "timing": timing,
        "units": [str(item) for item in units],
    }
    if explicit_execution_sequence:
        action_contract.update(
            {
                "execution_frame_count": len(execution_frames),
                "execution_frames_sha256": _canonical(execution_frames),
                "source_horizon_frame_count": int(
                    requested_execution["source_horizon_frame_count"]
                ),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "runtime_result_id": runtime_result_id,
        "source_action_sha256": action_sha,
        "derived_via_controller_fk": True,
        "controller_id": "nvidia/GEAR-SONIC:/opt/wbc@protocol-v4",
        "controller_sha256": controller_sha,
        "robot_model_sha256": model_sha,
        "joint_order_schema_version": JOINT_ORDER_SCHEMA_VERSION,
        "mapping_digest": PROTOCOL_V4_MAPPING_DIGEST,
        "controller_code_artifact": controller_ref,
        "robot_model_artifact": {"path": str(model), "sha256": model_sha},
        "landmarks": landmarks,
        "camera_projection_context_sha256": projection_context_sha256,
        "camera_source_frame_sha256": source_frame_sha256,
        "cross_simulator_registration": registration,
        "controller_fk_sequence": controller_fk_sequence,
        "controller_fk_sequence_sha256": controller_fk_sequence_sha256,
        "controller_execution_contract": execution_contract,
        "generated_robot_state": state,
        "action_contract": action_contract,
        "executed_at": utc_now_iso(),
    }
    signed = {
        "schema_version": "sc3_controller_fk_runtime_result.v1",
        "request_sha256": _canonical(request),
        "step_index": int(request.get("step_index") or 0),
        "source_action_sha256": action_sha,
        "runtime_result_id": runtime_result_id,
        "controller_id": payload["controller_id"],
        "controller_sha256": controller_sha,
        "robot_model_sha256": model_sha,
        "controller_code_artifact": controller_ref,
        "robot_model_artifact": payload["robot_model_artifact"],
        "derived_via_controller_fk": True,
        "landmarks": landmarks,
        "camera_projection_context_sha256": payload[
            "camera_projection_context_sha256"
        ],
        "camera_source_frame_sha256": payload["camera_source_frame_sha256"],
        "cross_simulator_registration": payload[
            "cross_simulator_registration"
        ],
        "generated_robot_state": state,
    }
    key = os.environ.get(SIGNING_KEY_ENV, "")
    if not key:
        raise ValueError("gear_sonic_executor_attestation_signing_key_missing")
    payload["executor_attestation"] = build_sc3_runtime_attestation(
        signed,
        private_key_file=key,
        report_path=target.with_name("gear_sonic_executor_signature_report.json"),
        signer_key_id="gear-sonic-controller-fk-runtime",
        verifier_id="blueprint-sc3-controller-fk-verifier",
    )
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    input_path = os.environ.get("BLUEPRINT_CONTROLLER_FK_INPUT", "")
    output_path = os.environ.get("BLUEPRINT_CONTROLLER_FK_OUTPUT", "")
    if not input_path or not output_path:
        raise SystemExit("BLUEPRINT_CONTROLLER_FK_INPUT and OUTPUT are required")
    run_adapter(input_path=input_path, output_path=output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
