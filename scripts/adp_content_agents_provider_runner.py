#!/usr/bin/env python3
"""Run the bounded NVIDIA Content Agents can comparison once on a GPU host."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "adp_content_agents_vast_result.v1"
TIMEOUT_SECONDS = 2400


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _redact(value: str, env: dict[str, str]) -> str:
    redacted = value
    for name, secret in env.items():
        if secret and any(marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET")):
            redacted = redacted.replace(secret, "REDACTED_SECRET")
    return redacted


def _run(
    command: Sequence[str], *, log_path: Path, env: dict[str, str], timeout: int = TIMEOUT_SECONDS
) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            check=False,
        )
        returncode = completed.returncode
        output = completed.stdout + completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        output = (exc.stdout or "") + (exc.stderr or "")
        timed_out = True
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(_redact(output, env), encoding="utf-8")
    finished = dt.datetime.now(dt.timezone.utc)
    return {
        "command": [str(item) for item in command],
        "returncode": returncode,
        "timed_out": timed_out,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "runtime_seconds": (finished - started).total_seconds(),
        "log": log_path.name,
    }


def _files(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.stat().st_size <= 100_000_000:
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return rows


def _copy_evidence(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if not path.is_file() or path.stat().st_size > 100_000_000:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _physics_output(work: Path) -> Path | None:
    candidates = sorted(
        path
        for path in work.rglob("*")
        if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc"}
    )
    preferred = [path for path in candidates if "physics" in path.name.lower()]
    return (preferred or candidates or [None])[0]


def _validation_stage(source_usd: Path, destination: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    source_stage = Usd.Stage.Open(str(source_usd))
    if source_stage is None or not source_stage.GetDefaultPrim().IsValid():
        raise ValueError("physics_agent_output_default_prim_missing")
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdPhysics.Scene.Define(stage, "/World/physics_scene")
    control = stage.DefinePrim("/World/control", "Xform")
    control.GetReferences().AddReference(str(source_usd.resolve()))
    stage.GetRootLayer().Save()


def main() -> int:
    runtime_root = Path(__file__).resolve().parent
    output_root = Path(
        os.environ.get(
            "BLUEPRINT_ADP_CONTENT_AGENTS_OUTPUT_DIR",
            runtime_root.parent / "runtime_output",
        )
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "adp_content_agents_vast_result.json"
    source_root = runtime_root / "content_agents_source"
    python = Path(
        os.environ.get(
            "BLUEPRINT_ADP_CONTENT_AGENTS_PYTHON",
            source_root / ".venv" / "bin" / "python",
        )
    )
    bin_dir = python.parent
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["WU_OVRTX_AUTO_PROVISION"] = "0"
    env["WU_OVRTX_VENV_DIR"] = str(source_root / ".ovrtx_venv")
    env["NVIDIA_DRIVER_CAPABILITIES"] = "all"

    gpu = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        log_path=output_root / "nvidia-smi.log",
        env=env,
        timeout=60,
    )
    configs = runtime_root / "configs"
    agent_specs = {
        "material": ("material-agent", configs / "material_agent.yaml", configs / ".material"),
        "texture": ("texture-agent", configs / "texture_agent.yaml", configs / ".texture"),
        "physics": ("physics-agent", configs / "physics_agent.yaml", configs / ".physics"),
    }
    agents: dict[str, Any] = {}
    blockers: list[str] = []
    for name in ("material", "texture", "physics"):
        executable, config, work = agent_specs[name]
        command = [str(bin_dir / executable), "run", str(config)]
        if name in {"material", "physics"}:
            command.append("--clean")
        execution = _run(
            command,
            log_path=output_root / f"{name}-agent.log",
            env=env,
        )
        produced = _files(work)
        success = execution["returncode"] == 0 and bool(produced)
        if not success:
            blockers.append(f"{name}_agent_full_execution_failed")
        _copy_evidence(work, output_root / f"{name}_workdir")
        agents[name] = {
            f"{name}_agent_attempted": True,
            f"{name}_agent_executed": success,
            "execution": execution,
            "produced_artifacts": produced,
            "retry_count": 0,
        }

    validation: dict[str, Any] = {
        "validation_agent_attempted": False,
        "validation_agent_executed": False,
    }
    physics_output = _physics_output(agent_specs["physics"][2])
    if physics_output is None:
        blockers.append("physics_agent_output_missing_for_validation")
    else:
        try:
            stage_path = output_root / "physics_agent_validation_stage.usda"
            _validation_stage(physics_output, stage_path)
            validation_dir = output_root / "validation_agent"
            command = [
                str(bin_dir / "validation-agent"),
                "validate",
                str(stage_path),
                "--task",
                "Validate static physics authoring only; do not infer dynamic or physical truth.",
                "--template",
                "physics_sane",
                "--output-dir",
                str(validation_dir),
                "--format",
                "json",
            ]
            execution = _run(
                command,
                log_path=output_root / "validation-agent.log",
                env=env,
                timeout=600,
            )
            validation_result_path = validation_dir / "validation_result.json"
            payload = (
                json.loads(validation_result_path.read_text(encoding="utf-8"))
                if validation_result_path.is_file()
                else {}
            )
            success = execution["returncode"] == 0 and payload.get("verdict") == "pass"
            validation = {
                "validation_agent_attempted": True,
                "validation_agent_executed": success,
                "execution": execution,
                "verdict": payload.get("verdict"),
                "result_sha256": (
                    _sha256(validation_result_path)
                    if validation_result_path.is_file()
                    else None
                ),
            }
            if not success:
                blockers.append("validation_agent_static_check_failed")
        except Exception as exc:  # preserve a typed terminal result
            validation = {
                "validation_agent_attempted": True,
                "validation_agent_executed": False,
                "error_type": type(exc).__name__,
            }
            blockers.append("validation_agent_static_check_failed")
    agents["validation"] = validation
    agents["joint"] = {
        "joint_agent_inapplicable_single_rigid_body": True,
        "joint_agent_executed": False,
        "reason": "selected target has one rigid body and zero articulated joints",
    }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "completed" if not blockers else "blocked",
        "source_commit": "36dbf3f274f8e256637230a05a085853f65cc175",
        "source_tree": "d36ddaed4c3ea44ab81c9f8178ab40d2eb0f8fe3",
        "source_version": "0.5.2",
        "input_usd_sha256": _sha256(
            runtime_root / "input" / "adp009a_840313_canned_beverage_control.usda"
        ),
        "gpu_probe": gpu,
        "agents": agents,
        "material_agent_executed": agents["material"]["material_agent_executed"],
        "texture_agent_executed": agents["texture"]["texture_agent_executed"],
        "physics_agent_executed": agents["physics"]["physics_agent_executed"],
        "validation_agent_executed": validation["validation_agent_executed"],
        "joint_agent_inapplicable_single_rigid_body": True,
        "model_backend_call_authorized": True,
        "paid_gpu_execution": True,
        "retry_cap": 0,
        "blockers": sorted(set(blockers)),
        "claim_boundaries": {
            "agent_outputs_are_authored_candidates_not_measurements": True,
            "static_validation_is_not_dynamic_simulation": True,
            "inpainting_result": False,
            "physical_evidence": False,
        },
        "raw_secret_values_recorded": False,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
