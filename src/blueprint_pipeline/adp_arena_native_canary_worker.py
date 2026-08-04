"""Self-contained native Isaac Lab-Arena zero-action canary worker.

This file is copied into an immutable paid-provider bundle.  It intentionally
runs only the frozen negative control; candidate policy servers and checkpoints
are neither downloaded nor queried here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ARENA_REPOSITORY = "https://github.com/isaac-sim/IsaacLab-Arena.git"
ARENA_REVISION = "3c19a3a9e45fc2cc1b64ab8a43047ecac9c0ad4d"
ISAAC_LAB_REVISION = "af1bab4dc173ba69b08fab779c14ead61d13fd33"
GROOT_REVISION = "e29d8fc50b0e4745120ae3fb72447986fe638aa6"
OPENPI_REVISION = "c23745b5ad24e98f66967ea795a07b2588ed6c79"
ARENA_UV_LOCK_SHA256 = "35001404fa10d3f591d326d7a36b15c0b35cf307b754edea87310f719ec439da"
RESULT_NAME = "adp_arena_native_canary.json"
ISAAC_PYTHON = Path("/isaac-sim/python.sh")
# Contract marker: the native runtime is also checked with isaacsim.SimulationApp.
SIMULATION_APP_CONTRACT = "SimulationApp"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _run(command: list[str], *, log_path: Path, cwd: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "log_path": log_path.name,
        "log_sha256": _sha256(log_path),
    }


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clone_and_verify(source: Path, output: Path) -> dict[str, Any]:
    if source.exists():
        shutil.rmtree(source)
    commands: list[dict[str, Any]] = []
    commands.append(
        _run(
            ["git", "config", "--global", "url.https://github.com/.insteadOf", "git@github.com:"],
            log_path=output / "git_transport_config.log",
        )
    )
    commands.append(
        _run(
            ["git", "clone", "--no-checkout", ARENA_REPOSITORY, str(source)],
            log_path=output / "arena_clone.log",
        )
    )
    if commands[-1]["returncode"] != 0:
        raise RuntimeError("arena_source_clone_failed")
    commands.append(
        _run(
            ["git", "-C", str(source), "checkout", "--detach", ARENA_REVISION],
            log_path=output / "arena_checkout.log",
        )
    )
    commands.append(
        _run(
            ["git", "-C", str(source), "submodule", "update", "--init", "--recursive"],
            log_path=output / "arena_submodules.log",
        )
    )
    if any(row["returncode"] != 0 for row in commands):
        raise RuntimeError("arena_source_materialization_failed")
    if _git(source, "rev-parse", "HEAD") != ARENA_REVISION:
        raise RuntimeError("arena_source_revision_mismatch")
    expected_submodules = {
        "submodules/IsaacLab": ISAAC_LAB_REVISION,
        "submodules/Isaac-GR00T": GROOT_REVISION,
    }
    observed_submodules = {
        path: _git(source / path, "rev-parse", "HEAD") for path in expected_submodules
    }
    if observed_submodules != expected_submodules:
        raise RuntimeError("arena_submodule_revision_mismatch")
    openpi_commit = (source / "isaaclab_arena_openpi/docker/OPENPI_COMMIT").read_text().strip()
    if openpi_commit != OPENPI_REVISION:
        raise RuntimeError("arena_openpi_revision_mismatch")
    if _sha256(source / "uv.lock") != "sha256:" + ARENA_UV_LOCK_SHA256:
        raise RuntimeError("arena_uv_lock_mismatch")
    return {
        "commands": commands,
        "arena_revision": ARENA_REVISION,
        "submodule_revisions": observed_submodules,
        "openpi_revision": openpi_commit,
        "uv_lock_sha256": "sha256:" + ARENA_UV_LOCK_SHA256,
    }


def _install(source: Path, output: Path) -> list[dict[str, Any]]:
    commands = [
        ["apt-get", "update"],
        [
            "apt-get",
            "install",
            "-y",
            "git",
            "git-lfs",
            "cmake",
            "ffmpeg",
            "jq",
            "python3-pip",
        ],
        ["bash", "-lc", f"ln -sfn /isaac-sim {source}/submodules/IsaacLab/_isaac_sim"],
        [
            "bash",
            "-lc",
            (
                f"for d in {source}/submodules/IsaacLab/source/isaaclab*/; do "
                f'{ISAAC_PYTHON} -m pip install --no-deps -e "$d" || exit $?; done'
            ),
        ],
        [str(source / "submodules/IsaacLab/isaaclab.sh"), "-i"],
        [str(ISAAC_PYTHON), "-m", "pip", "install", "--force-reinstall", "daqp==0.8.5"],
        [
            str(ISAAC_PYTHON),
            "-m",
            "pip",
            "install",
            "-e",
            str(source / "submodules/IsaacLab/source/isaaclab_newton[all]"),
        ],
        [
            str(ISAAC_PYTHON),
            "-m",
            "pip",
            "install",
            "msgpack==1.1.0",
            "msgpack-numpy==0.4.8",
            "pyzmq==27.0.1",
            "av",
        ],
        [
            str(ISAAC_PYTHON),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--ignore-requires-python",
            "-e",
            str(source / "submodules/Isaac-GR00T"),
        ],
        [
            str(ISAAC_PYTHON),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            (
                "openpi-client @ git+https://github.com/Physical-Intelligence/openpi@"
                f"{OPENPI_REVISION}#subdirectory=packages/openpi-client"
            ),
        ],
        [str(ISAAC_PYTHON), "-m", "pip", "install", "-e", f"{source}[dev]"],
        [
            str(ISAAC_PYTHON),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "boto3==1.40.61",
            "botocore==1.40.61",
            "s3transfer==0.14.0",
            "requests==2.32.3",
        ],
    ]
    results: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        row = _run(command, log_path=output / f"install_{index:02d}.log", cwd=source)
        results.append(row)
        if row["returncode"] != 0:
            raise RuntimeError(f"arena_install_step_failed:{index:02d}")
    return results


def _episode_rows(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output.rglob("episode_results*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _artifact_manifest(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != RESULT_NAME:
            rows.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return rows


def main() -> int:
    output = Path(os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR", "runtime_output")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path("/workspace/blueprint_adp_arena_source")
    result: dict[str, Any] = {
        "schema_version": "adp_arena_native_canary.v1",
        "status": "blocked",
        "control_id": "arena_zero_action_negative",
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
        "blockers": [],
    }
    try:
        if not ISAAC_PYTHON.is_file():
            raise RuntimeError("isaac_python_missing")
        result["source_materialization"] = _clone_and_verify(source, output)
        result["install_steps"] = _install(source, output)
        smoke = _run(
            [
                str(ISAAC_PYTHON),
                "-c",
                (
                    "from isaacsim import SimulationApp; "
                    "app=SimulationApp({'headless': True}); print('BLUEPRINT_ADP_ARENA_SIMULATION_APP_OK'); app.close()"
                ),
            ],
            log_path=output / "simulation_app_smoke.log",
            cwd=source,
        )
        result["simulation_app_smoke"] = smoke
        if smoke["returncode"] != 0:
            raise RuntimeError("arena_simulation_app_smoke_failed")
        canary_output = output / "zero_action_control"
        command = [
            str(ISAAC_PYTHON),
            "isaaclab_arena/evaluation/policy_runner.py",
            "--headless",
            "--policy_type",
            "zero_action",
            "--num_episodes",
            "1",
            "--seed",
            "41000",
            "--enable_cameras",
            "--record_camera_video",
            "--output_base_dir",
            str(canary_output),
            "pick_and_place_maple_table",
            "--embodiment",
            "droid_abs_joint_pos",
            "--pick_up_object",
            "rubiks_cube_hot3d_robolab",
            "--destination_location",
            "bowl_ycb_robolab",
            "--hdr",
            "home_office_robolab",
            "--light_intensity",
            "500.0",
            "--placement_seed",
            "41000",
        ]
        rollout = _run(command, log_path=output / "zero_action_control.log", cwd=source)
        result["rollout"] = rollout
        rows = _episode_rows(canary_output)
        videos = sorted(canary_output.rglob("*.mp4"))
        result["episode_rows"] = rows
        result["episode_count"] = len(rows)
        result["video_count"] = len(videos)
        if rollout["returncode"] != 0:
            raise RuntimeError("arena_zero_action_rollout_failed")
        if len(rows) != 1:
            raise RuntimeError("arena_zero_action_episode_receipt_count_invalid")
        if rows[0].get("success") is not False:
            raise RuntimeError("arena_zero_action_negative_control_unexpected_success")
        if not videos:
            raise RuntimeError("arena_zero_action_review_video_missing")
        result["status"] = "completed"
    except Exception as exc:
        result["blockers"].append(str(exc))
    finally:
        freeze = (
            _run(
                [str(ISAAC_PYTHON), "-m", "pip", "freeze"],
                log_path=output / "isaac_runtime_lock.txt",
            )
            if ISAAC_PYTHON.is_file()
            else None
        )
        result["runtime_lock"] = freeze
        result["artifacts"] = _artifact_manifest(output)
        result_path = output / RESULT_NAME
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        "BLUEPRINT_ADP_ARENA_NATIVE_CANARY_"
        + ("OK" if result["status"] == "completed" else "BLOCKED")
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
