"""Self-contained installer/launcher for the ADP-009D native Isaac micro-check."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ARENA_REPOSITORY = "https://github.com/isaac-sim/IsaacLab-Arena.git"
ARENA_REVISION = "8b4a3a47fc53de23e8205089d71109a2e2348acd"
ARENA_TREE = "03f31f3dd56c56d00f24dbfb09711ec0ab345de8"
ISAAC_LAB_REVISION = "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
ISAAC_LAB_TREE = "454115265327a80acabd07cbd36e10071fc0c065"
RESULT_NAME = "adp009d_native_microcheck.json"
ISAAC_PYTHON = Path("/isaac-sim/python.sh")
INSTALL_PROFILE_ID = "isaaclab_arena_physx_task_runtime.v3"
# Arena's asset decorators execute its policy-package registration closure even
# for a no-policy environment. That closure imports isaaclab_rl.rsl_rl and
# rsl_rl, so include the official rsl-rl extra explicitly.
ISAAC_LAB_INSTALL_TARGETS = (
    "assets",
    "ov",
    "physx",
    "rl[rsl-rl]",
    "tasks",
    "teleop",
)
H5PY_VERSION = "3.16.0"
H5PY_LINUX_CP312_WHEEL_SHA256 = (
    "dfc21898ff025f1e8e67e194965a95a8d4754f452f83454538f98f8a3fcb207e"
)
H5PY_LINUX_CP312_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/9e/e9/"
    "1a19e42cd43cc1365e127db6aae85e1c671da1d9a5d746f4d34a50edb577/"
    f"h5py-{H5PY_VERSION}-cp312-cp312-manylinux_2_28_x86_64.whl"
    f"#sha256={H5PY_LINUX_CP312_WHEEL_SHA256}"
)
HYDRA_RUNTIME_URLS = (
    "https://files.pythonhosted.org/packages/3e/38/"
    "7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/"
    "antlr4-python3-runtime-4.9.3.tar.gz"
    "#sha256=f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b",
    "https://files.pythonhosted.org/packages/a4/0e/"
    "152509871bf30df6fc38569f52a2db9b55dd41aae957adae50a053ac7778/"
    "omegaconf-2.3.1-py3-none-any.whl"
    "#sha256=3d701d14e9a8828f1edd28bb70b725908b34277cdd72cf7d6a83f94dadc6b6a0",
    "https://files.pythonhosted.org/packages/9c/97/"
    "f9d463a6f3c7d0955753eca5cbbf35b596ac471dd13fe357211a53fd37be/"
    "hydra_core-1.3.5-py3-none-any.whl"
    "#sha256=a3ff35b4ea6794e4c83d993016f4bde4ac35797ebe7a08f30e83ed9341880331",
)
ARENA_REMOTE_TRANSPORT_URLS = (
    "https://files.pythonhosted.org/packages/f1/54/"
    "65af8de681fa8255402c80eda2a501ba467921d5a7a028c9c22a2c2eedb5/"
    "msgpack-1.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    "#sha256=17fb65dd0bec285907f68b15734a993ad3fc94332b5bb21b0435846228de1f39",
    "https://files.pythonhosted.org/packages/7e/0a/"
    "2356305c423a975000867de56888b79e44ec2192c690ff93c3109fd78081/"
    "pyzmq-27.0.1-cp312-abi3-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl"
    "#sha256=f5b6133c8d313bde8bd0d123c169d22525300ff164c2189f849de495e1344577",
)
EXPECTED_RUNTIME_DISTRIBUTIONS = (
    "antlr4-python3-runtime",
    "h5py",
    "hydra-core",
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_newton",
    "isaaclab_ov",
    "isaaclab_physx",
    "isaaclab_rl",
    "isaaclab_tasks",
    "isaaclab_teleop",
    "msgpack",
    "omegaconf",
    "pyzmq",
    "rsl-rl-lib",
    "isaaclab_arena",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _run(command: list[str], *, log_path: Path, cwd: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    print(f"BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:{log_path.stem}:started", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        returncode = process.wait()
    row = {
        "command": command,
        "returncode": returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "log_path": log_path.name,
        "log_sha256": _sha256(log_path),
    }
    print(
        f"BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:{log_path.stem}:completed:rc={returncode}",
        flush=True,
    )
    return row


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _materialize_source(source: Path, output: Path) -> dict[str, Any]:
    if source.exists():
        shutil.rmtree(source)
    rows = [
        _run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", ARENA_REPOSITORY, str(source)],
            log_path=output / "arena_clone.log",
        )
    ]
    if rows[-1]["returncode"] != 0:
        raise RuntimeError("arena_source_clone_failed")
    rows.append(
        _run(
            ["git", "-C", str(source), "checkout", "--detach", ARENA_REVISION],
            log_path=output / "arena_checkout.log",
        )
    )
    rows.append(
        _run(
            [
                "git",
                "-c",
                "url.https://github.com/.insteadOf=git@github.com:",
                "-C",
                str(source),
                "submodule",
                "update",
                "--init",
                "submodules/IsaacLab",
            ],
            log_path=output / "isaac_lab_submodule.log",
        )
    )
    if any(row["returncode"] != 0 for row in rows):
        raise RuntimeError("arena_source_materialization_failed")
    observed = {
        "arena_revision": _git(source, "rev-parse", "HEAD"),
        "arena_tree": _git(source, "rev-parse", "HEAD^{tree}"),
        "isaac_lab_revision": _git(source / "submodules/IsaacLab", "rev-parse", "HEAD"),
        "isaac_lab_tree": _git(source / "submodules/IsaacLab", "rev-parse", "HEAD^{tree}"),
    }
    expected = {
        "arena_revision": ARENA_REVISION,
        "arena_tree": ARENA_TREE,
        "isaac_lab_revision": ISAAC_LAB_REVISION,
        "isaac_lab_tree": ISAAC_LAB_TREE,
    }
    if observed != expected:
        raise RuntimeError("official_source_tree_identity_mismatch")
    return {"commands": rows, **observed}


def _install_commands(source: Path) -> list[list[str]]:
    """Return the pinned minimum official Lab/Arena runtime installation plan."""

    lab = source / "submodules/IsaacLab"
    distribution_probe = (
        "import antlr4, h5py, hydra, importlib.metadata as m, json, msgpack, omegaconf, zmq; "
        "from isaaclab_ov.renderers import OVRTXRendererCfg; "
        "from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper; "
        "from rsl_rl.runners import DistillationRunner, OnPolicyRunner; "
        f"names={EXPECTED_RUNTIME_DISTRIBUTIONS!r}; "
        "print(json.dumps({name: m.version(name) for name in names}, sort_keys=True))"
    )
    return [
        ["ln", "-sfn", "/isaac-sim", str(lab / "_isaac_sim")],
        [str(lab / "isaaclab.sh"), "-i", ",".join(ISAAC_LAB_INSTALL_TARGETS)],
        [str(ISAAC_PYTHON), "-m", "pip", "install", H5PY_LINUX_CP312_WHEEL_URL],
        [
            str(ISAAC_PYTHON),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            *HYDRA_RUNTIME_URLS,
        ],
        [
            str(ISAAC_PYTHON),
            "-m",
            "pip",
            "install",
            *ARENA_REMOTE_TRANSPORT_URLS,
        ],
        [str(ISAAC_PYTHON), "-m", "pip", "install", "--editable", str(source)],
        [str(ISAAC_PYTHON), "-c", distribution_probe],
    ]


def _validate_install_commands(commands: list[list[str]]) -> None:
    """Fail closed if the paid worker expands beyond the admitted runtime profile."""

    flattened = "\n".join(" ".join(command) for command in commands)
    expected_selector = f"-i {','.join(ISAAC_LAB_INSTALL_TARGETS)}"
    forbidden = (
        "[dev]",
        "source/isaaclab*/",
        "isaaclab_mimic",
        "isaaclab_visualizers",
        "isaaclab_contrib",
        "isaaclab_tasks_experimental",
    )
    if expected_selector not in flattened:
        raise RuntimeError("adp009d_runtime_install_profile_selector_mismatch")
    if H5PY_LINUX_CP312_WHEEL_URL not in flattened:
        raise RuntimeError("adp009d_runtime_h5py_pin_missing")
    if any(url not in flattened for url in HYDRA_RUNTIME_URLS):
        raise RuntimeError("adp009d_runtime_hydra_pin_missing")
    if any(url not in flattened for url in ARENA_REMOTE_TRANSPORT_URLS):
        raise RuntimeError("adp009d_runtime_arena_transport_pin_missing")
    if any(marker in flattened for marker in forbidden):
        raise RuntimeError("adp009d_runtime_install_profile_expanded")


def _install(source: Path, output: Path) -> list[dict[str, Any]]:
    commands = _install_commands(source)
    _validate_install_commands(commands)
    rows: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        row = _run(command, log_path=output / f"install_{index:02d}.log", cwd=source)
        rows.append(row)
        if row["returncode"] != 0:
            raise RuntimeError(f"arena_install_step_failed:{index:02d}")
    return rows


def _artifact_manifest(output: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(output).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != RESULT_NAME
    ]


def main() -> int:
    runtime = Path(__file__).resolve().parent
    output = Path(os.environ.get("BLUEPRINT_ADP009D_OUTPUT_DIR", runtime.parent / "runtime_output")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path("/workspace/blueprint_adp009d_arena_source")
    outer: dict[str, Any] = {
        "schema_version": "adp009d_native_microcheck_worker.v1",
        "status": "blocked",
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
        "install_profile": {
            "profile_id": INSTALL_PROFILE_ID,
            "isaac_lab_targets": list(ISAAC_LAB_INSTALL_TARGETS),
            "expected_runtime_distributions": list(EXPECTED_RUNTIME_DISTRIBUTIONS),
            "arena_development_extras_installed": False,
            "all_isaac_lab_extensions_installed": False,
            "fail_closed_plan_validation": True,
        },
        "blockers": [],
    }
    try:
        if not ISAAC_PYTHON.is_file():
            raise RuntimeError("isaac_python_missing")
        outer["source_materialization"] = _materialize_source(source, output)
        outer["install_steps"] = _install(source, output)
        runtime_row = _run(
            [
                str(ISAAC_PYTHON),
                str(runtime / "adp009d_isaac_runtime.py"),
                "--runtime-dir",
                str(runtime),
                "--output-dir",
                str(output),
                "--headless",
                "--enable_cameras",
                "--device",
                "cuda:0",
            ],
            log_path=output / "native_microcheck.log",
            cwd=source,
        )
        outer["native_runtime"] = runtime_row
        inner_path = output / RESULT_NAME
        if not inner_path.is_file():
            raise RuntimeError("native_microcheck_result_missing")
        inner = json.loads(inner_path.read_text(encoding="utf-8"))
        if runtime_row["returncode"] != 0 or inner.get("status") != "completed":
            outer["blockers"].extend(inner.get("blockers") or ["native_microcheck_not_completed"])
        else:
            outer["status"] = "completed"
    except Exception as exc:
        outer["blockers"].append(str(exc))
    finally:
        lock = None
        if ISAAC_PYTHON.is_file():
            lock = _run(
                [str(ISAAC_PYTHON), "-m", "pip", "freeze"],
                log_path=output / "isaac_runtime_lock.txt",
            )
        outer["runtime_lock"] = lock
        outer["artifacts"] = _artifact_manifest(output)
        (output / "adp009d_native_microcheck_worker.json").write_text(
            json.dumps(outer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print("BLUEPRINT_ADP009D_WORKER_" + ("OK" if outer["status"] == "completed" else "BLOCKED"))
    return 0 if outer["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
