"""Build the pinned HumanoidArena LeRobot inference environment in Isaac 6.0.1.

Planning is local and read-only. Execution can use a Linux Docker host with
the pinned Isaac image already present, or run inside that image when a paid
provider owns the container. Both paths install the same hashed Python lock.
This command never starts a policy or downloads a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_policy_server_supervisor import (
    PINNED_SOURCE_REVISION,
    _source_revision,
)
from .native_g1_run_preflight import PINNED_POLICY_SERVER_SHA256
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


SCHEMA = "native_g1_policy_runtime_build.v1"
LOCK = Path(__file__).resolve().parents[2] / (
    "configs/g1_humanoidarena_lerobot_pi_py312_linux_x86_64.requirements.txt"
)
LOCK_SHA256 = "584a48b4c42f7911780d5cde20c2c216e35a4ba2d0680a8805d8327e53536e2a"
PYPROJECT_SHA256 = "034d8e821601a1cf514c5e7c61b998b7b5fce129be16752c3328b16ab074092d"
EXECUTION_MODES = frozenset({"docker_host", "inside_isaac_container"})
PINNED_IMAGE_ENV = "BLUEPRINT_G1_PINNED_ISAAC_IMAGE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _absolute(path: Path, *, kind: str) -> Path:
    if (
        not path.is_absolute() or path.is_symlink() or path.resolve() != path
        or any(char in str(path) for char in ",\n\r")
        or (kind == "dir" and not path.is_dir())
        or (kind == "file" and not path.is_file())
    ):
        raise ValueError("g1_policy_runtime_path_invalid")
    return path


def prepare_g1_policy_runtime_build(
    *, checkout: Path, output_dir: Path, execution_mode: str = "docker_host"
) -> dict[str, Any]:
    """Seal a build command against source and dependency identities."""

    if execution_mode not in EXECUTION_MODES:
        raise ValueError("g1_policy_runtime_execution_mode_invalid")

    source = _absolute(checkout, kind="dir")
    server = _absolute(source / "lerobot/scripts/serve_lerobot_vla_http.py", kind="file")
    pyproject = _absolute(source / "lerobot/pyproject.toml", kind="file")
    lock = _absolute(LOCK, kind="file")
    if (
        _source_revision(server) != PINNED_SOURCE_REVISION
        or _sha256(server) != "sha256:" + PINNED_POLICY_SERVER_SHA256
        or _sha256(pyproject) != "sha256:" + PYPROJECT_SHA256
        or _sha256(lock) != "sha256:" + LOCK_SHA256
    ):
        raise ValueError("g1_policy_runtime_source_or_lock_identity_mismatch")
    output = Path(output_dir).expanduser()
    if (
        not output.is_absolute() or output.exists() or output.is_symlink()
        or output.resolve() != output or not output.parent.is_dir()
        or any(char in str(output) for char in ",\n\r")
        or output.is_relative_to(source) or source.is_relative_to(output)
        or output.is_relative_to(lock) or lock.is_relative_to(output)
    ):
        raise ValueError("g1_policy_runtime_output_invalid")

    runtime = output / "policy-runtime"
    python = runtime / "bin/python"
    freeze = output / "installed.freeze"
    source_package = source / "lerobot/src"
    probe = (
        "import pathlib,torch,lerobot;"
        "from lerobot.configs.policies import PreTrainedConfig;"
        "from lerobot.policies.factory import get_policy_class;"
        "from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy;"
        "from lerobot.policies.pi05.modeling_pi05 import PI05Policy;"
        f"expected=pathlib.Path({str(source_package)!r}).resolve();"
        "actual=pathlib.Path(lerobot.__file__).resolve();"
        "assert actual.is_relative_to(expected), 'g1_policy_import_origin_mismatch';"
        "assert torch.version.cuda is not None, 'g1_policy_torch_not_cuda_build'"
    )
    native_prerequisites = (
        "test \"$(id -u)\" = 0",
        "apt-get update -qq",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "linux-libc-dev build-essential pkg-config python3-dev python3.12-dev",
        "test -f /usr/include/linux/input.h",
        "test -f /usr/include/linux/input-event-codes.h",
        "test -f /usr/include/python3.12/Python.h",
        "command -v cc >/dev/null",
    ) if execution_mode == "inside_isaac_container" else ()
    shell = " && ".join((
        *native_prerequisites,
        f"/isaac-sim/python.sh -m venv --copies {shlex.quote(str(runtime))}",
        f"{shlex.quote(str(python))} -m pip install --no-cache-dir --require-hashes -r {shlex.quote(str(lock))}",
        f"{shlex.quote(str(python))} -m pip check",
        f"{shlex.quote(str(python))} -c {shlex.quote(probe)}",
        f"{shlex.quote(str(python))} -m pip freeze > {shlex.quote(str(freeze))}",
    ))
    docker_command = [
        "docker", "run", "--rm", "--pull", "never", "--network", "bridge",
        "--env", "ACCEPT_EULA=Y", "--env", "PRIVACY_CONSENT=Y",
        "--env", "PIP_DISABLE_PIP_VERSION_CHECK=1",
        "--env", f"PYTHONPATH={source_package}",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--mount", f"type=bind,src={source},dst={source},readonly",
        "--mount", f"type=bind,src={lock},dst={lock},readonly",
        "--mount", f"type=bind,src={output},dst={output}",
        "--entrypoint", "/bin/bash", NATIVE_TASK_ARENA_IMAGE,
        "-euo", "pipefail", "-c", shell,
    ]
    command = (
        docker_command if execution_mode == "docker_host"
        else ["/bin/bash", "-euo", "pipefail", "-c", shell]
    )
    plan = {
        "schema_version": SCHEMA,
        "status": "planned_not_built",
        "source_revision": PINNED_SOURCE_REVISION,
        "source_pyproject_sha256": "sha256:" + PYPROJECT_SHA256,
        "policy_server_sha256": "sha256:" + PINNED_POLICY_SERVER_SHA256,
        "checkout_path": str(source),
        "dependency_lock_path": str(lock),
        "dependency_lock_sha256": "sha256:" + LOCK_SHA256,
        "python_target": "CPython 3.12 Linux x86_64",
        "image": NATIVE_TASK_ARENA_IMAGE,
        "execution_mode": execution_mode,
        "runtime_root": str(runtime),
        "command": command,
        "model_bytes_downloaded": False,
        "episode_executed": False,
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def execute_g1_policy_runtime_build(*, plan: dict[str, Any]) -> dict[str, Any]:
    """Run the prepared command and retain terminal success or failure."""

    if sys.platform != "linux" or plan.get("plan_digest") != canonical_digest(
        plan, digest_field="plan_digest"
    ):
        raise ValueError("g1_policy_runtime_execution_plan_invalid")
    if plan.get("execution_mode") == "inside_isaac_container" and (
        os.environ.get(PINNED_IMAGE_ENV) != NATIVE_TASK_ARENA_IMAGE
        or not Path("/isaac-sim/python.sh").is_file()
    ):
        raise ValueError("g1_policy_runtime_pinned_container_unverified")
    output = Path(plan["runtime_root"]).parent
    if prepare_g1_policy_runtime_build(
        checkout=Path(plan["checkout_path"]), output_dir=output,
        execution_mode=plan.get("execution_mode", ""),
    ) != plan:
        raise ValueError("g1_policy_runtime_execution_inputs_changed")
    output.mkdir()
    (output / (SCHEMA + ".json")).write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    log = output / "build.log"
    try:
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.run(plan["command"], stdout=stream, stderr=subprocess.STDOUT, check=False)
        freeze = output / "installed.freeze"
        success = (
            process.returncode == 0 and freeze.is_file()
            and not freeze.is_symlink() and freeze.stat().st_size > 0
            and (output / "policy-runtime/bin/python").is_file()
        )
        result = {
            "schema_version": SCHEMA + ".result",
            "status": "built_import_probe_passed_no_cuda_probe" if success else "blocked",
            "plan_digest": plan["plan_digest"],
            "container_exit_code": process.returncode,
            "execution_mode": plan["execution_mode"],
            "build_log_sha256": _sha256(log),
            "installed_freeze_sha256": _sha256(freeze) if success else None,
            "runtime_root": plan["runtime_root"],
            "cuda_device_probed": False,
            "model_bytes_downloaded": False,
            "episode_executed": False,
        }
    except OSError as exc:
        result = {
            "schema_version": SCHEMA + ".result",
            "status": "blocked",
            "plan_digest": plan["plan_digest"],
            "container_exit_code": None,
            "execution_mode": plan["execution_mode"],
            "blocker": {"type": type(exc).__name__, "message": str(exc)},
            "cuda_device_probed": False,
            "model_bytes_downloaded": False,
            "episode_executed": False,
        }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (output / (SCHEMA + ".result.json")).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--execution-mode", choices=sorted(EXECUTION_MODES), default="docker_host"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = prepare_g1_policy_runtime_build(
        checkout=args.checkout, output_dir=args.output_dir,
        execution_mode=args.execution_mode,
    )
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    result = execute_g1_policy_runtime_build(plan=plan)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "built_import_probe_passed_no_cuda_probe" else 1


if __name__ == "__main__":
    sys.exit(main())
