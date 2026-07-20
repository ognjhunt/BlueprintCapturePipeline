#!/usr/bin/env python3
"""Attempt-local runtime repair for the sealed G1 kitchen episode image.

The immutable image contains the pinned GEAR-SONIC controller and the official
MuJoCo simulator source, but not the simulator's Unitree SDK environment.  This
script prepares that public, commit-pinned sidecar under ``/workspace`` and
patches the two small Python composition seams discovered by the first live
action in attempt 011.  It does not replace GR00T, GEAR-SONIC, OSCAR, Isaac, or
their checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


WBC_SOURCE_URL = "https://github.com/NVlabs/GR00T-WholeBodyControl.git"
WBC_SOURCE_REF = "6d8e931b9b10a4db2d8e7aba3ad6d5da3529ff3b"
WORKSPACE = Path("/workspace")
SIM_ROOT = WORKSPACE / "gear_sonic_sim_runtime"
SIM_SITE = SIM_ROOT / "site-packages"
SIM_CMEEL_SITE = SIM_SITE / "cmeel.prefix/lib/python3.10/site-packages"
SIM_CMEEL_LIB = SIM_SITE / "cmeel.prefix/lib"
SDK_SOURCE = SIM_ROOT / "wbc-source"
PROVENANCE = WORKSPACE / "closed_loop_out/gear_sonic_sim/runtime_provenance.json"
READINESS = WORKSPACE / "closed_loop_out/gear_sonic_sim/controller_readiness.json"
SIM_SCRIPT_SHA256 = "6e79b61d43e94b81997a417f175971c3cb1d35bc2f361fb6d5f29ece27c3a40b"
UNITREE_SDK_TREE_SHA = "de46dc81e2ea59272d9fc788a6f127c6ac7a9f91"
PREPARE_SPECS = (
    "numpy==1.26.4",
    "cyclonedds==0.10.2",
    "scipy==1.15.3",
    "mujoco==3.5.0",
    "tyro==1.0.8",
    "pin==2.7.0",
    "pyyaml==6.0.3",
    "pyzmq==27.1.0",
    "msgpack==1.1.2",
    "msgpack-numpy==0.4.8",
    "opencv-python==4.11.0.86",
    "easydict==1.13",
    "loguru==0.7.3",
    "joblib==1.5.2",
    "tqdm==4.67.1",
)
MOTION_DIM = 64
HAND_DIM = 7
FRAME_DIM = MOTION_DIM + 2 * HAND_DIM


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in argv],
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", **dict(env or {})},
    )


def _prepare() -> int:
    marker = Path("/opt/wbc/.blueprint-source-revision")
    if marker.read_text(encoding="utf-8").strip() != WBC_SOURCE_REF:
        raise RuntimeError("gear_sonic_sim_sealed_source_revision_mismatch")
    sim_script = Path("/opt/wbc/gear_sonic/scripts/run_sim_loop.py")
    if not sim_script.is_file():
        raise RuntimeError("gear_sonic_sim_script_missing_from_sealed_image")
    if hashlib.sha256(sim_script.read_bytes()).hexdigest() != SIM_SCRIPT_SHA256:
        raise RuntimeError("gear_sonic_sim_script_sha256_mismatch")

    SIM_ROOT.mkdir(parents=True, exist_ok=True)
    if not (SDK_SOURCE / ".git").is_dir():
        SDK_SOURCE.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", str(SDK_SOURCE)])
        _run(["git", "remote", "add", "origin", WBC_SOURCE_URL], cwd=SDK_SOURCE)
        _run(["git", "sparse-checkout", "init", "--cone"], cwd=SDK_SOURCE)
        _run(
            [
                "git",
                "sparse-checkout",
                "set",
                "external_dependencies/unitree_sdk2_python",
            ],
            cwd=SDK_SOURCE,
        )
        _run(
            [
                "git",
                "fetch",
                "--depth",
                "1",
                "--filter=blob:none",
                "origin",
                WBC_SOURCE_REF,
            ],
            cwd=SDK_SOURCE,
        )
        _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=SDK_SOURCE)
    observed_ref = _run(["git", "rev-parse", "HEAD"], cwd=SDK_SOURCE).stdout.strip()
    if observed_ref != WBC_SOURCE_REF:
        raise RuntimeError("gear_sonic_sim_sdk_source_revision_mismatch")
    observed_sdk_tree = _run(
        ["git", "rev-parse", "HEAD:external_dependencies/unitree_sdk2_python"],
        cwd=SDK_SOURCE,
    ).stdout.strip()
    if observed_sdk_tree != UNITREE_SDK_TREE_SHA:
        raise RuntimeError("gear_sonic_sim_unitree_sdk_tree_mismatch")
    sdk_package = SDK_SOURCE / "external_dependencies/unitree_sdk2_python"
    if not (sdk_package / "setup.py").is_file():
        raise RuntimeError("gear_sonic_sim_unitree_sdk_source_missing")

    SIM_SITE.mkdir(parents=True, exist_ok=True)
    uv = [
        "/usr/local/bin/uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--target",
        str(SIM_SITE),
    ]
    _run([*uv, "--only-binary=:all:", *PREPARE_SPECS])
    _run([*uv, "--no-deps", str(sdk_package)])

    probe = _run(
        [
            sys.executable,
            "-c",
            (
                "import cv2, cyclonedds, mujoco, pinocchio, tyro, yaml; "
                "import unitree_sdk2py; "
                "from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory; "
                "assert SimulatorFactory"
            ),
        ],
        env={
            "PYTHONPATH": ":".join(
                item
                for item in (
                    str(SIM_SITE),
                    str(SIM_CMEEL_SITE),
                    "/opt/wbc",
                    "/opt/OSCAR",
                    os.environ.get("PYTHONPATH", ""),
                )
                if item
            ),
            "LD_LIBRARY_PATH": ":".join(
                item
                for item in (
                    str(SIM_CMEEL_LIB),
                    "/opt/wbc/gear_sonic_deploy/thirdparty_runtime/lib",
                    "/opt/onnxruntime/lib",
                    "/usr/local/lib",
                    os.environ.get("LD_LIBRARY_PATH", ""),
                )
                if item
            ),
            "MUJOCO_GL": "osmesa",
        },
    )
    distributions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(path=[str(SIM_SITE)]):
        name = str(distribution.metadata.get("Name") or "").strip().lower()
        if name:
            distributions[name] = str(distribution.version)
    payload = {
        "schema_version": "single_g1_kitchen_gear_sonic_sim_runtime.v1",
        "status": "prepared_and_import_verified",
        "official_source_url": WBC_SOURCE_URL,
        "official_source_commit": observed_ref,
        "unitree_sdk_source_tree": observed_sdk_tree,
        "sealed_controller_source_commit": marker.read_text(encoding="utf-8").strip(),
        "sim_script": str(sim_script),
        "sim_script_sha256": SIM_SCRIPT_SHA256,
        "unitree_sdk_source": str(sdk_package),
        "site_packages": str(SIM_SITE),
        "resolved_distributions": dict(sorted(distributions.items())),
        "probe_returncode": probe.returncode,
        "headless_mujoco_required": True,
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "dependency_readiness_is_not_controller_execution": True,
            "dependency_readiness_is_not_task_success": True,
        },
    }
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _runtime_env() -> dict[str, str]:
    if not PROVENANCE.is_file():
        raise RuntimeError("gear_sonic_sim_runtime_not_prepared")
    return {
        **os.environ,
        "PYTHONPATH": ":".join(
            item
            for item in (
                str(SIM_SITE),
                str(SIM_CMEEL_SITE),
                "/opt/wbc",
                "/opt/OSCAR",
                os.environ.get("PYTHONPATH", ""),
            )
            if item
        ),
        "LD_LIBRARY_PATH": ":".join(
            item
            for item in (
                str(SIM_CMEEL_LIB),
                "/opt/wbc/gear_sonic_deploy/thirdparty_runtime/lib",
                "/opt/onnxruntime/lib",
                "/usr/local/lib",
                os.environ.get("LD_LIBRARY_PATH", ""),
            )
            if item
        ),
        "MUJOCO_GL": "osmesa",
    }


def _run_sim() -> int:
    argv = [
        sys.executable,
        "/opt/wbc/gear_sonic/scripts/run_sim_loop.py",
        "--no-enable-onscreen",
        "--no-enable-offscreen",
    ]
    os.execve(sys.executable, argv, _runtime_env())
    return 127


def _finite_vector(value: Any, *, size: int, name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name}_missing")
    result = [float(item) for item in value]
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name}_dimension_or_value_invalid")
    return result


def _wait_controller_ready() -> int:
    """Start protocol-v4 control and require one matching live state reply.

    The old bootstrap accepted ``robot_config``, which is published while the
    controller is still waiting for Unitree low-state.  This probe publishes a
    tiny attempt-local protocol-v4 pose, then accepts only the corresponding
    ``g1_debug`` state.  The sockets close before the episode executor binds.
    """

    runtime_env = _runtime_env()
    os.environ.update(runtime_env)
    for path in (str(SIM_CMEEL_SITE), str(SIM_SITE), "/opt/wbc", "/opt/OSCAR"):
        if path not in sys.path:
            sys.path.insert(0, path)

    from blueprint_pipeline.gear_sonic_official_zmq_executor import _zmq_roundtrip

    motion = [((index % 7) - 3) * 0.0001 for index in range(MOTION_DIM)]
    left = [(index + 1) * 0.0001 for index in range(HAND_DIM)]
    right = [-(index + 1) * 0.0001 for index in range(HAND_DIM)]
    deadline = time.monotonic() + 900.0
    accepted: dict[str, Any] | None = None
    last_error = ""
    attempts = 0
    while time.monotonic() < deadline:
        for env_name, failure in (
            ("GEAR_SONIC_SIM_PID", "official_gear_sonic_sim_exited_before_ready"),
            ("GEAR_SONIC_PID", "official_gear_sonic_controller_exited_before_ready"),
        ):
            try:
                os.kill(int(os.environ[env_name]), 0)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise RuntimeError(failure) from exc
        attempts += 1
        try:
            accepted = dict(
                _zmq_roundtrip(
                    motion_token=motion,
                    left_hand=left,
                    right_hand=right,
                    frame_index=-1,
                    timeout_seconds=min(20.0, max(1.0, deadline - time.monotonic())),
                )
            )
            break
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            time.sleep(0.5)
    if accepted is None:
        raise RuntimeError(
            "official_gear_sonic_controller_state_not_ready"
            + (f":{last_error}" if last_error else "")
        )

    body_target = _finite_vector(
        accepted.get("body_q_target"), size=29, name="official_body_q_target"
    )
    body_measured = _finite_vector(
        accepted.get("body_q_measured"), size=29, name="official_body_q_measured"
    )
    base_quat = _finite_vector(
        accepted.get("base_quat_measured"),
        size=4,
        name="official_base_quat_measured",
    )
    echoed_left = _finite_vector(
        accepted.get("last_left_hand_action"),
        size=HAND_DIM,
        name="official_left_hand_echo",
    )
    echoed_right = _finite_vector(
        accepted.get("last_right_hand_action"),
        size=HAND_DIM,
        name="official_right_hand_echo",
    )
    for sent, echoed, side in (
        (left, echoed_left, "left"),
        (right, echoed_right, "right"),
    ):
        if any(abs(a - b) > 1e-6 for a, b in zip(sent, echoed)):
            raise RuntimeError(f"official_gear_sonic_readiness_hand_echo_mismatch:{side}")

    canary = {"motion_token": motion, "left_hand": left, "right_hand": right}
    payload = {
        "schema_version": "single_g1_kitchen_gear_sonic_controller_readiness.v1",
        "status": "ready",
        "topic": "g1_debug",
        "protocol_version": 4,
        "readiness_attempts": attempts,
        "readiness_probe_sha256": hashlib.sha256(
            json.dumps(canary, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "validated_fields": {
            "token_state_dimension": MOTION_DIM,
            "body_q_target_dimension": len(body_target),
            "body_q_measured_dimension": len(body_measured),
            "base_quat_measured_dimension": len(base_quat),
            "left_hand_echo_dimension": len(echoed_left),
            "right_hand_echo_dimension": len(echoed_right),
        },
        "sim_process_alive": True,
        "controller_process_alive": True,
        "robot_config_only_is_not_readiness": True,
        "active_matching_state_roundtrip_required": True,
        "raw_probe_values_recorded": False,
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "readiness_probe_only": True,
            "readiness_probe_is_not_episode_policy_action": True,
            "controller_state_ready_is_not_task_success": True,
        },
    }
    READINESS.parent.mkdir(parents=True, exist_ok=True)
    READINESS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _finite_frames(value: Any, *, width: int, name: str) -> Any:
    import numpy as np

    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"blocked_{name}_not_numeric") from exc
    if array.ndim < 1 or array.size == 0 or array.shape[-1] != width:
        raise ValueError(f"blocked_{name}_shape_invalid")
    frames = array.reshape(-1, width)
    if not np.isfinite(frames).all():
        raise ValueError(f"blocked_{name}_nonfinite")
    return frames


def _patched_normalize(server_module: Any, action: Mapping[str, Any]) -> dict[str, Any] | None:
    import numpy as np

    motion = server_module._action_field(action, "motion_token")
    left = server_module._action_field(action, "left_hand_joints")
    right = server_module._action_field(action, "right_hand_joints")
    if motion is None and left is None and right is None:
        return None
    if motion is None or left is None or right is None:
        raise ValueError("blocked_incomplete_unitree_g1_sonic_control_fields")
    motion_frames = _finite_frames(motion, width=MOTION_DIM, name="unitree_g1_sonic_motion_token")
    left_frames = _finite_frames(left, width=HAND_DIM, name="unitree_g1_sonic_left_hand")
    right_frames = _finite_frames(right, width=HAND_DIM, name="unitree_g1_sonic_right_hand")
    frame_count = int(motion_frames.shape[0])
    if int(left_frames.shape[0]) != frame_count or int(right_frames.shape[0]) != frame_count:
        raise ValueError("blocked_unitree_g1_sonic_horizon_frame_count_mismatch")
    selected = np.concatenate((motion_frames[0], left_frames[0], right_frames[0])).astype(
        np.float32
    )
    chunk = selected.tolist()
    fields = {
        "motion_token": motion_frames.tolist(),
        "left_hand_joints": left_frames.tolist(),
        "right_hand_joints": right_frames.tolist(),
    }
    full_sha = hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    selected_sha = hashlib.sha256(
        json.dumps(chunk, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    horizon = {
        "schema_version": "unitree_g1_sonic_action_horizon.v1",
        "frame_count": frame_count,
        "frame_dimension": FRAME_DIM,
        "full_dimension": frame_count * FRAME_DIM,
        "source_field_shapes": {
            "motion_token": list(np.asarray(motion).shape),
            "left_hand_joints": list(np.asarray(left).shape),
            "right_hand_joints": list(np.asarray(right).shape),
        },
        "source_fieldwise_horizon_sha256": full_sha,
        "selected_frame_index": 0,
        "selected_frame_sha256": selected_sha,
        "selection_mode": "fresh_receding_horizon_first_frame",
    }
    return {
        "action_type": "unitree_g1_sonic_latent_action_chunk",
        "action_chunk": chunk,
        "action_dimension": FRAME_DIM,
        "action_units": ["latent"] * MOTION_DIM + ["rad"] * (2 * HAND_DIM),
        "action_timing": {
            "control_hz": 50.0,
            "sample_period_seconds": 0.02,
            "selected_horizon_frame_index": 0,
            "source_horizon_frame_count": frame_count,
            "_blueprint_action_horizon": horizon,
        },
        "action_horizon": horizon,
        "unitree_groot_n17_sonic_action_payload_present": True,
        "unitree_groot_n17_sonic_action_chunk_present": True,
        "unitree_g1_sonic_control_fields": [
            "left_hand_joints",
            "motion_token",
            "right_hand_joints",
        ],
        "sonic_latent_action": server_module._jsonable(motion),
        "hand_targets": {
            "left_hand_joints": server_module._jsonable(left),
            "right_hand_joints": server_module._jsonable(right),
        },
        "action_values_sha256": selected_sha,
    }


def _run_closed_loop() -> int:
    from blueprint_pipeline import groot_sonic_policy_endpoint as endpoint_module
    from blueprint_pipeline import oscar_isaac_closed_loop_eval as eval_module
    from blueprint_pipeline import unitree_groot_n17_sonic_policy_server_command as server_module

    server_module._normalize_policy_server_action = lambda action: _patched_normalize(
        server_module, action
    )
    original_make_endpoint = endpoint_module.make_groot_sonic_zmq_policy_endpoint

    def make_endpoint(*args: Any, **kwargs: Any) -> Any:
        endpoint = original_make_endpoint(*args, **kwargs)

        def wrapped(*call_args: Any, **call_kwargs: Any) -> dict[str, Any]:
            action = dict(endpoint(*call_args, **call_kwargs))
            timing = dict(action.get("action_timing") or {})
            horizon = timing.pop("_blueprint_action_horizon", None)
            action["action_timing"] = timing
            if isinstance(horizon, Mapping):
                action["action_horizon"] = dict(horizon)
            return action

        return wrapped

    endpoint_module.make_groot_sonic_zmq_policy_endpoint = make_endpoint
    original_action_record = eval_module._action_record_from_policy_endpoint

    def action_record(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = dict(original_action_record(*args, **kwargs))
        endpoint_action = kwargs.get("endpoint_action")
        if not isinstance(endpoint_action, Mapping) and len(args) >= 2:
            endpoint_action = args[1]
        if isinstance(endpoint_action, Mapping):
            for key in ("action_units", "action_timing", "action_horizon"):
                if key in endpoint_action:
                    result[key] = endpoint_action[key]
        return result

    eval_module._action_record_from_policy_endpoint = action_record
    return int(eval_module.main())


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run-sim", action="store_true")
    group.add_argument("--wait-controller-ready", action="store_true")
    group.add_argument("--run-closed-loop", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.prepare:
        if remaining:
            parser.error("--prepare accepts no remaining arguments")
        return _prepare()
    if args.run_sim:
        if remaining:
            parser.error("--run-sim accepts no remaining arguments")
        return _run_sim()
    if args.wait_controller_ready:
        if remaining:
            parser.error("--wait-controller-ready accepts no remaining arguments")
        return _wait_controller_ready()
    sys.argv = [sys.argv[0], *remaining]
    return _run_closed_loop()


if __name__ == "__main__":
    raise SystemExit(main())
