"""Isaac runtime preflight command for robot-eval workers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .common import ensure_dir, utc_now_iso, write_json


ISAAC_WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION = "isaac_worker_runtime_preflight.v1"
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


def _isaac_smoke_checks(
    *,
    smoke_steps: int,
    require_rtx_render: bool,
    env: Mapping[str, str],  # noqa: ARG001 - retained for parity with MuJoCo preflight
) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        from isaacsim import SimulationApp  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on worker image
        checks.append(
            _check(
                "python_import_isaacsim",
                "blocked",
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
            )
        )
        return checks, ["python_import_isaacsim_failed"]

    checks.append(_check("python_import_isaacsim", "passed"))
    renderer = "RayTracedLighting"
    checks.append(
        _check(
            "headless_rtx_context_selection",
            "passed",
            renderer=renderer,
            headless=True,
        )
    )
    if not require_rtx_render:
        checks.append(
            _check(
                "rtx_smoke_frame_render",
                "skipped_not_required_for_local_rehearsal",
                require_rtx_render=False,
            )
        )
        return checks, blockers

    simulation_app = None
    try:
        simulation_app = SimulationApp({"headless": True, "renderer": renderer})
        import omni.replicator.core as rep  # type: ignore[import-not-found]

        camera = rep.create.camera(position=(0, 0, 2), look_at=(0, 0, 0))
        render_product = rep.create.render_product(camera, (16, 16))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([render_product])
        for _ in range(max(1, int(smoke_steps))):
            rep.orchestrator.step()
        pixels = annot.get_data()
        pixel_count = int(getattr(pixels, "size", 0) or 0)
        if pixel_count <= 0:
            raise RuntimeError("empty_rtx_smoke_frame")
        shape = list(getattr(pixels, "shape", []) or [])
        checks.append(
            _check(
                "rtx_smoke_frame_render",
                "passed",
                width=16,
                height=16,
                pixel_count=pixel_count,
                pixel_shape=shape,
            )
        )
    except Exception as exc:
        checks.append(
            _check(
                "rtx_smoke_frame_render",
                "blocked",
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
            )
        )
        blockers.append("rtx_smoke_frame_render_failed")
    finally:
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass
    return checks, blockers


def run_isaac_worker_runtime_preflight(
    *,
    output_path: str | Path,
    require_nvidia_smi: bool = False,
    require_rtx_render: bool = False,
    smoke_steps: int = 1,
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

    isaac_checks, isaac_blockers = _isaac_smoke_checks(
        smoke_steps=smoke_steps,
        require_rtx_render=require_rtx_render,
        env=runtime_env,
    )
    checks.extend(isaac_checks)
    blockers.extend(isaac_blockers)

    payload = {
        "schema_version": ISAAC_WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed" if not blockers else "blocked",
        "simulator": "isaac",
        "worker_image_family": "isaac-eval-worker",
        "checks": checks,
        "blockers": blockers,
        "requirements": {
            "require_nvidia_smi": require_nvidia_smi,
            "require_rtx_render": require_rtx_render,
            "smoke_steps": max(1, int(smoke_steps)),
        },
        "proof_boundary": {
            "runtime_preflight_executed": True,
            "runtime_preflight_is_not_simulator_proof": True,
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "non_ranking_operational_claim_validated": False,
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
        or "isaac_worker_runtime_preflight.json"
    )
    path = Path(value)
    ensure_dir(path.parent if path.parent != Path("") else Path("."))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-nvidia-smi", action="store_true")
    parser.add_argument("--require-rtx-render", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=1)
    args = parser.parse_args(argv)
    payload = run_isaac_worker_runtime_preflight(
        output_path=args.output or _default_output_path(),
        require_nvidia_smi=args.require_nvidia_smi,
        require_rtx_render=args.require_rtx_render,
        smoke_steps=args.smoke_steps,
    )
    print(json.dumps({"status": payload["status"], "blockers": payload["blockers"]}))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
