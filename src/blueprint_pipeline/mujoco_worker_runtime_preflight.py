"""MuJoCo runtime preflight command for robot-eval workers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .common import ensure_dir, utc_now_iso, write_json


MUJOCO_WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION = "mujoco_worker_runtime_preflight.v1"
WORKER_PREFLIGHT_DETAIL_OUTPUT_ENV = "BLUEPRINT_RUNTIME_PREFLIGHT_DETAIL_OUTPUT"
WORKER_PREFLIGHT_OUTPUT_ENV = "BLUEPRINT_RUNTIME_PREFLIGHT_OUTPUT"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _check(name: str, status: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, **details}


def _nvidia_smi_check(*, env: Mapping[str, str], required: bool) -> tuple[dict[str, Any], list[str]]:
    executable = shutil.which("nvidia-smi", path=env.get("PATH"))
    if not executable:
        status = "blocked" if required else "skipped_not_required"
        blockers = ["nvidia_smi_unavailable"] if required else []
        return (
            _check(
                "nvidia_smi_gpu_inventory",
                status,
                required=required,
                executable_found=False,
            ),
            blockers,
        )
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        env=dict(env),
    )
    success = completed.returncode == 0
    blockers = ["nvidia_smi_failed"] if required and not success else []
    return (
        _check(
            "nvidia_smi_gpu_inventory",
            "passed" if success else "blocked" if required else "skipped_unavailable",
            required=required,
            executable_found=True,
            exit_code=completed.returncode,
            stdout=completed.stdout.strip()[:2000],
            stderr=completed.stderr.strip()[:2000],
        ),
        blockers,
    )


def _mujoco_smoke_checks(
    *,
    smoke_steps: int,
    require_egl_render: bool,
    env: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        import mujoco  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on runtime install
        checks.append(
            _check(
                "python_import_mujoco",
                "blocked",
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
            )
        )
        return checks, ["python_import_mujoco_failed"]

    checks.append(
        _check(
            "python_import_mujoco",
            "passed",
            mujoco_version=_string(getattr(mujoco, "__version__", "")),
        )
    )
    selected_gl = _string(env.get("MUJOCO_GL")) or ("egl" if require_egl_render else "platform_default")
    checks.append(
        _check(
            "headless_context_selection",
            "passed",
            selected_context=selected_gl,
            require_egl_render=require_egl_render,
        )
    )

    model = None
    data = None
    try:
        xml = """
<mujoco model="blueprint_worker_preflight_blank">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.01"/>
    <body name="probe" pos="0 0 0.2">
      <freejoint/>
      <geom name="probe_geom" type="sphere" size="0.03" rgba="0.2 0.6 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        checks.append(
            _check(
                "blank_model_or_scene_load",
                "passed",
                nq=int(model.nq),
                nv=int(model.nv),
                ngeom=int(model.ngeom),
            )
        )
    except Exception as exc:
        checks.append(
            _check(
                "blank_model_or_scene_load",
                "blocked",
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
            )
        )
        blockers.append("blank_model_or_scene_load_failed")

    if model is not None and data is not None:
        try:
            for _ in range(max(1, int(smoke_steps))):
                mujoco.mj_step(model, data)
            checks.append(
                _check(
                    "short_rollout_smoke",
                    "passed",
                    smoke_steps=max(1, int(smoke_steps)),
                    sim_time=float(data.time),
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    "short_rollout_smoke",
                    "blocked",
                    smoke_steps=max(1, int(smoke_steps)),
                    error_type=type(exc).__name__,
                    error=str(exc)[:2000],
                )
            )
            blockers.append("short_rollout_smoke_failed")

    if require_egl_render and model is not None and data is not None:
        previous_gl = os.environ.get("MUJOCO_GL")
        os.environ["MUJOCO_GL"] = "egl"
        try:
            renderer = mujoco.Renderer(model, height=16, width=16)
            renderer.update_scene(data)
            pixels = renderer.render()
            renderer.close()
            checks.append(
                _check(
                    "egl_context_when_rendering",
                    "passed",
                    width=int(pixels.shape[1]),
                    height=int(pixels.shape[0]),
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    "egl_context_when_rendering",
                    "blocked",
                    error_type=type(exc).__name__,
                    error=str(exc)[:2000],
                )
            )
            blockers.append("egl_context_when_rendering_failed")
        finally:
            if previous_gl is None:
                os.environ.pop("MUJOCO_GL", None)
            else:
                os.environ["MUJOCO_GL"] = previous_gl
    elif require_egl_render:
        checks.append(
            _check(
                "egl_context_when_rendering",
                "blocked",
                reason="blank_model_or_scene_load_failed",
            )
        )
        blockers.append("egl_context_when_rendering_not_attempted")
    else:
        checks.append(
            _check(
                "egl_context_when_rendering",
                "skipped_not_required_for_local_rehearsal",
                require_egl_render=False,
            )
        )
    return checks, blockers


def run_mujoco_worker_runtime_preflight(
    *,
    output_path: str | Path,
    require_nvidia_smi: bool = False,
    require_egl_render: bool = False,
    smoke_steps: int = 5,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    runtime_env = dict(env or os.environ)
    generated_at = utc_now_iso()
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    nvidia_check, nvidia_blockers = _nvidia_smi_check(
        env=runtime_env,
        required=require_nvidia_smi,
    )
    checks.append(nvidia_check)
    blockers.extend(nvidia_blockers)

    mujoco_checks, mujoco_blockers = _mujoco_smoke_checks(
        smoke_steps=smoke_steps,
        require_egl_render=require_egl_render,
        env=runtime_env,
    )
    checks.extend(mujoco_checks)
    blockers.extend(mujoco_blockers)

    payload = {
        "schema_version": MUJOCO_WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed" if not blockers else "blocked",
        "simulator": "mujoco",
        "worker_image_family": "mujoco-eval-worker",
        "checks": checks,
        "blockers": blockers,
        "requirements": {
            "require_nvidia_smi": require_nvidia_smi,
            "require_egl_render": require_egl_render,
            "smoke_steps": max(1, int(smoke_steps)),
        },
        "proof_boundary": {
            "runtime_preflight_executed": True,
            "runtime_preflight_is_not_simulator_proof": True,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "safety_validated": False,
            "public_claim_upgrade_allowed": False,
        },
        "secret_values_in_artifact": False,
    }
    write_json(Path(output_path), payload)
    return payload


def _default_output_path() -> Path:
    value = (
        os.environ.get(WORKER_PREFLIGHT_DETAIL_OUTPUT_ENV)
        or os.environ.get(WORKER_PREFLIGHT_OUTPUT_ENV)
        or "mujoco_worker_runtime_preflight.json"
    )
    path = Path(value)
    ensure_dir(path.parent if path.parent != Path("") else Path("."))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-nvidia-smi", action="store_true")
    parser.add_argument("--require-egl-render", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=5)
    args = parser.parse_args(argv)
    payload = run_mujoco_worker_runtime_preflight(
        output_path=args.output or _default_output_path(),
        require_nvidia_smi=args.require_nvidia_smi,
        require_egl_render=args.require_egl_render,
        smoke_steps=args.smoke_steps,
    )
    print(json.dumps({"status": payload["status"], "blockers": payload["blockers"]}))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
