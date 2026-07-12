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


def _numeric_vector(value: Any, *, name: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name}_missing")
    result = [float(item) for item in value]
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name}_nonfinite_or_empty")
    return result


def _controller_tree_manifest(root: Path, output: Path) -> tuple[str, dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not rows:
        raise ValueError("official_gear_sonic_controller_tree_empty")
    payload = {
        "schema_version": "gear_sonic_controller_tree_manifest.v1",
        "root_name": root.name,
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
    raw_output.unlink(missing_ok=True)
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
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"official_gear_sonic_executor_returncode_{completed.returncode}")
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
    }
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
        "generated_robot_state": state,
        "action_contract": {
            "command": "UNITREE_G1_SONIC",
            "dimension": len(vector),
            "values_sha256": _canonical(vector),
            "timing": timing,
            "units": [str(item) for item in units],
        },
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
