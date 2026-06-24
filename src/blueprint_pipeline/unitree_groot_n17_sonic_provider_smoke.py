"""Build or import a bounded GR00T N1.7 + UNITREE_G1_SONIC provider smoke.

This provider bundle proves only packaging and, after a real provider run,
whether a GR00T/SONIC action command returned a Blueprint-compatible action. A
dry run, replay, or imported provider output is not a fresh local policy proof.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .unitree_groot_n17_sonic_policy_runtime import (
    GROOT_ROOT_ENV,
    N17_CHECKPOINT_ENV,
    POLICY_COMMAND_ENV,
    POLICY_ID,
    POLICY_SERVER_URL_ENV,
    SIM2SIM_COMMAND_ENV,
    SONIC_CHECKPOINT_ENV,
    WBC_ROOT_ENV,
)


SCHEMA_VERSION = "unitree_groot_n17_sonic_policy_provider_smoke.v1"
OUTPUT_SCHEMA_VERSION = "unitree_groot_n17_sonic_policy_provider_output.v1"
DEFAULT_TASK_ID = "contact_or_push_light_object"
DEFAULT_TASK_PROMPT = "move the Unitree G1 SONIC hand toward the light object"
DEFAULT_BUNDLE_FILENAME = "unitree_groot_n17_sonic_policy_provider_runtime_bundle.zip"
PROVIDER_OUTPUT_FILENAME = "unitree_groot_n17_sonic_policy_provider_output.json"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_executable(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _copy_frame(frame_path: Path, output_path: Path) -> None:
    if not frame_path.is_file():
        raise FileNotFoundError(f"unitree_groot_n17_sonic_input_frame_missing:{frame_path}")
    ensure_dir(output_path.parent)
    shutil.copy2(frame_path, output_path)


def _load_policy_observation(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("policy_observation_json_must_be_object")
    raw_observation = value.get("observation") if isinstance(value.get("observation"), Mapping) else value
    return dict(raw_observation) if isinstance(raw_observation, Mapping) else None


PROVIDER_RUNNER = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

OUTPUT_SCHEMA_VERSION = "unitree_groot_n17_sonic_policy_provider_output.v1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase(name: str, **fields: Any) -> None:
    print(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_PHASE:"
        + json.dumps(
            {
                "phase": name,
                "observed_at_epoch": round(time.time(), 3),
                "raw_secret_values_recorded": False,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def _run_logged(
    command: list[str],
    *,
    cwd: Path | None,
    log_path: Path,
    timeout_seconds: float | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("ab") as handle:
        handle.write(("BLUEPRINT_COMMAND_STARTED:" + json.dumps(command) + "\n").encode())
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            return {
                "status": "completed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "duration_seconds": round(time.time() - started, 3),
                "log_path": str(log_path),
                "log_tail": _tail(log_path),
            }
        except subprocess.TimeoutExpired:
            handle.write(b"\nBLUEPRINT_COMMAND_TIMED_OUT\n")
            return {
                "status": "timed_out",
                "returncode": None,
                "duration_seconds": round(time.time() - started, 3),
                "log_path": str(log_path),
                "log_tail": _tail(log_path),
            }


def _parse_tcp_url(value: str) -> tuple[str, int] | None:
    text = value.strip()
    if not text:
        return None
    parsed = urlparse(text if "://" in text else f"tcp://{text}")
    if parsed.hostname and parsed.port:
        return parsed.hostname, int(parsed.port)
    return None


def _tcp_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=1.0):
            return True
    except OSError:
        return False


def _install_uv(output_dir: Path) -> dict[str, Any]:
    if subprocess.run(["bash", "-lc", "command -v uv"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return {"status": "completed", "uv_already_available": True}
    return _run_logged(
        [
            "bash",
            "-lc",
            "curl -LsSf https://astral.sh/uv/0.8.14/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh",
        ],
        cwd=None,
        log_path=output_dir / "groot_policy_server_bootstrap_uv_install.log",
        timeout_seconds=180,
    )


def _checkout_groot_repo(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    repo_url = os.environ.get(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_REPO_URL",
        "https://github.com/NVIDIA/Isaac-GR00T.git",
    ).strip()
    repo_ref = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_REPO_REF", "").strip()
    root = Path(
        os.environ.get(
            "BLUEPRINT_UNITREE_GROOT_N17_SONIC_REMOTE_ROOT",
            output_dir / "groot_runtime" / "Isaac-GR00T",
        )
    )
    if (root / "gr00t" / "eval" / "run_gr00t_server.py").is_file():
        return root, {"status": "completed", "repo_already_available": True, "repo_root": str(root)}
    root.parent.mkdir(parents=True, exist_ok=True)
    if repo_ref:
        result = _run_logged(
            [
                "bash",
                "-lc",
                (
                    f"rm -rf {root!s} && git init {root!s} && cd {root!s} && "
                    f"git remote add origin {repo_url} && "
                    f"git fetch --depth 1 origin {repo_ref} && git checkout --detach FETCH_HEAD"
                ),
            ],
            cwd=None,
            log_path=output_dir / "groot_policy_server_bootstrap_git_checkout.log",
            timeout_seconds=600,
        )
    else:
        result = _run_logged(
            ["git", "clone", "--depth", "1", repo_url, str(root)],
            cwd=None,
            log_path=output_dir / "groot_policy_server_bootstrap_git_checkout.log",
            timeout_seconds=600,
        )
    result["repo_root"] = str(root)
    result["repo_url"] = repo_url
    result["repo_ref_configured"] = bool(repo_ref)
    return root, result


def _looks_like_hf_repo_id(value: str) -> bool:
    text = value.strip()
    if not text or text.startswith(("/", "./", "../")):
        return False
    if "://" in text:
        return False
    return text.count("/") == 1 and all(part for part in text.split("/"))


def _materialize_groot_model_path(
    *,
    output_dir: Path,
    model_path: str,
    venv_python: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    raw_model_path = model_path.strip()
    if not raw_model_path:
        return {
            "status": "blocked",
            "blockers": ["blocked_missing_gr00t_model_path"],
            "raw_model_path": raw_model_path,
        }
    candidate = Path(raw_model_path).expanduser()
    if candidate.exists():
        return {
            "status": "completed",
            "source": "local_path",
            "raw_model_path": raw_model_path,
            "resolved_model_path": str(candidate),
            "snapshot_download_ran": False,
        }
    if not _looks_like_hf_repo_id(raw_model_path):
        return {
            "status": "completed",
            "source": "passthrough",
            "raw_model_path": raw_model_path,
            "resolved_model_path": raw_model_path,
            "snapshot_download_ran": False,
        }

    safe_name = raw_model_path.replace("/", "__")
    local_dir = output_dir / "groot_runtime" / "model_snapshots" / safe_name
    script = r"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = Path(sys.argv[2])
local_dir.mkdir(parents=True, exist_ok=True)
snapshot_path = snapshot_download(
    repo_id=repo_id,
    local_dir=str(local_dir),
    allow_patterns=[
        "config.json",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "processor/*",
    ],
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
)
processor_config = Path(snapshot_path) / "processor" / "processor_config.json"
if not processor_config.is_file():
    print("BLUEPRINT_GROOT_MODEL_SNAPSHOT_MISSING_PROCESSOR_CONFIG")
    raise SystemExit(42)
print("BLUEPRINT_GROOT_MODEL_SNAPSHOT_READY:" + snapshot_path)
"""
    result = _run_logged(
        [str(venv_python), "-c", script, raw_model_path, str(local_dir)],
        cwd=None,
        log_path=output_dir / "groot_policy_server_model_snapshot_download.log",
        timeout_seconds=float(
            os.environ.get(
                "BLUEPRINT_UNITREE_GROOT_N17_SONIC_MODEL_SNAPSHOT_TIMEOUT_SECONDS",
                "900",
            )
        ),
        env=env,
    )
    result.update(
        {
            "source": "huggingface_snapshot_download",
            "raw_model_path": raw_model_path,
            "resolved_model_path": str(local_dir),
            "snapshot_download_ran": True,
            "allow_patterns": [
                "config.json",
                "model.safetensors.index.json",
                "model-*.safetensors",
                "processor/*",
            ],
            "processor_config_present": (local_dir / "processor" / "processor_config.json").is_file(),
        }
    )
    if result.get("status") != "completed" or not result["processor_config_present"]:
        result.setdefault("blockers", [])
        result["blockers"] = sorted(
            set(
                [*result.get("blockers", []), "blocked_gr00t_model_snapshot_download_failed"]
            )
        )
        result["status"] = "blocked"
    return result


def _bootstrap_gr00t_policy_server(
    *,
    output_dir: Path,
    policy_server_url: str,
    model_path: str,
) -> tuple[dict[str, Any], subprocess.Popen[Any] | None]:
    if not _truthy_env("BLUEPRINT_UNITREE_GROOT_N17_SONIC_AUTO_START_POLICY_SERVER"):
        return {"requested": False, "status": "not_requested"}, None
    parsed = _parse_tcp_url(policy_server_url)
    if parsed is None:
        return {
            "requested": True,
            "status": "blocked",
            "blockers": ["blocked_missing_policy_server_url_for_auto_start"],
        }, None
    host, port = parsed
    if host not in {"127.0.0.1", "localhost", "0.0.0.0"}:
        return {
            "requested": True,
            "status": "blocked",
            "blockers": ["blocked_auto_start_only_supports_localhost_policy_server_url"],
        }, None
    if _tcp_ready("127.0.0.1" if host == "0.0.0.0" else host, port):
        return {"requested": True, "status": "completed", "server_already_listening": True}, None

    output_dir.mkdir(parents=True, exist_ok=True)
    uv_result = _install_uv(output_dir)
    if uv_result.get("status") != "completed":
        return {
            "requested": True,
            "status": "blocked",
            "blockers": ["blocked_gr00t_uv_install_failed"],
            "uv_install": uv_result,
        }, None
    repo_root, checkout = _checkout_groot_repo(output_dir)
    if checkout.get("status") != "completed":
        return {
            "requested": True,
            "status": "blocked",
            "blockers": ["blocked_isaac_groot_repo_checkout_failed"],
            "uv_install": uv_result,
            "checkout": checkout,
        }, None

    env = dict(os.environ)
    env.setdefault("HF_HOME", str(output_dir / "hf_cache"))
    env.setdefault("HF_HUB_CACHE", str(output_dir / "hf_cache" / "hub"))
    env.setdefault("UV_PROJECT_ENVIRONMENT", str(output_dir / "groot_runtime" / "venv"))
    install_timeout = float(
        os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_UV_SYNC_TIMEOUT_SECONDS", "1800")
    )
    sync = _run_logged(
        ["uv", "sync", "--frozen", "--no-install-project", "--no-cache"],
        cwd=repo_root,
        log_path=output_dir / "groot_policy_server_bootstrap_uv_sync.log",
        timeout_seconds=install_timeout,
        env=env,
    )
    if sync.get("status") != "completed":
        return {
            "requested": True,
            "status": "blocked",
            "blockers": ["blocked_isaac_groot_uv_sync_failed"],
            "uv_install": uv_result,
            "checkout": checkout,
            "uv_sync": sync,
        }, None

    venv_python = Path(env["UV_PROJECT_ENVIRONMENT"]) / "bin" / "python"
    if not venv_python.is_file():
        return {
            "requested": True,
            "status": "blocked",
            "blockers": ["blocked_isaac_groot_uv_sync_did_not_create_python"],
            "uv_install": uv_result,
            "checkout": checkout,
            "uv_sync": sync,
            "venv_python": str(venv_python),
        }, None

    model_resolution = _materialize_groot_model_path(
        output_dir=output_dir,
        model_path=model_path,
        venv_python=venv_python,
        env=env,
    )
    if model_resolution.get("status") != "completed":
        return {
            "requested": True,
            "status": "blocked",
            "blockers": ["blocked_gr00t_model_snapshot_download_failed"],
            "uv_install": uv_result,
            "checkout": checkout,
            "uv_sync": sync,
            "venv_python": str(venv_python),
            "model_resolution": model_resolution,
        }, None
    resolved_model_path = str(model_resolution.get("resolved_model_path") or model_path)

    log_path = output_dir / "groot_policy_server.log"
    log = log_path.open("ab")
    server_env = dict(env)
    server_env["PYTHONPATH"] = (
        str(repo_root)
        + os.pathsep
        + server_env.get("PYTHONPATH", "")
    )
    server_cmd = [
        str(venv_python),
        "gr00t/eval/run_gr00t_server.py",
        "--model-path",
        resolved_model_path,
        "--embodiment-tag",
        "UNITREE_G1_SONIC",
        "--device",
        "cuda:0",
        "--host",
        "127.0.0.1" if host == "0.0.0.0" else host,
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        server_cmd,
        cwd=str(repo_root),
        env=server_env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    startup_timeout = float(
        os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SERVER_STARTUP_TIMEOUT_SECONDS", "900")
    )
    started = time.time()
    while time.time() - started < startup_timeout:
        if proc.poll() is not None:
            return {
                "requested": True,
                "status": "blocked",
                "blockers": ["blocked_gr00t_policy_server_exited_before_listening"],
                "server_returncode": proc.returncode,
                "venv_python": str(venv_python),
                "uv_install": uv_result,
                "checkout": checkout,
                "uv_sync": sync,
                "model_resolution": model_resolution,
                "server_log_path": str(log_path),
                "server_log_tail": _tail(log_path),
            }, proc
        if _tcp_ready("127.0.0.1" if host == "0.0.0.0" else host, port):
            return {
                "requested": True,
                "status": "completed",
                "server_started": True,
                "server_pid": proc.pid,
                "policy_server_host": host,
                "policy_server_port": port,
                "model_path": model_path,
                "resolved_model_path": resolved_model_path,
                "venv_python": str(venv_python),
                "uv_install": uv_result,
                "checkout": checkout,
                "uv_sync": sync,
                "model_resolution": model_resolution,
                "server_log_path": str(log_path),
            }, proc
        time.sleep(5)
    return {
        "requested": True,
        "status": "blocked",
        "blockers": ["blocked_gr00t_policy_server_startup_timeout"],
        "server_pid": proc.pid,
        "venv_python": str(venv_python),
        "uv_install": uv_result,
        "checkout": checkout,
        "uv_sync": sync,
        "model_resolution": model_resolution,
        "server_log_path": str(log_path),
        "server_log_tail": _tail(log_path),
    }, proc


def main() -> int:
    runtime_dir = Path(__file__).resolve().parent
    payload_path = Path(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_INPUT", runtime_dir / "policy_input.json"))
    output_path = Path(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT", runtime_dir / "unitree_groot_n17_sonic_policy_provider_output.json"))
    command = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND", "")
    n17_checkpoint = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT", "")
    sonic_checkpoint = os.environ.get("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT", "")
    groot_root = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT", "")
    wbc_root = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT", "")
    policy_server_url = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL", "")
    sim2sim_command = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND", "")
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        _phase("bootstrap_gr00t_policy_server_if_requested", requested=_truthy_env("BLUEPRINT_UNITREE_GROOT_N17_SONIC_AUTO_START_POLICY_SERVER"))
        policy_server_bootstrap, policy_server_process = _bootstrap_gr00t_policy_server(
            output_dir=output_path.parent,
            policy_server_url=policy_server_url,
            model_path=n17_checkpoint or "LucaFrat/groot-bs16",
        )
        if (
            policy_server_bootstrap.get("status") == "completed"
            and "unitree_groot_n17_sonic_policy_server_command" in command
        ):
            repo_root = _mapping(policy_server_bootstrap.get("checkout")).get("repo_root")
            venv_python = policy_server_bootstrap.get("venv_python")
            if repo_root and venv_python:
                command = (
                    f"{shlex.quote(str(venv_python))} "
                    "-m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
                )
                os.environ["PYTHONPATH"] = (
                    str(repo_root)
                    + os.pathsep
                    + os.environ.get("PYTHONPATH", "")
                )
        _phase(
            "invoke_unitree_groot_n17_sonic_adapter",
            command_configured=bool(command),
            sonic_checkpoint_configured=bool(sonic_checkpoint),
            policy_server_bootstrap_status=policy_server_bootstrap.get("status"),
        )
        from blueprint_pipeline.unitree_groot_n17_sonic_policy_command_adapter import run_unitree_groot_n17_sonic_policy

        response, exit_code = run_unitree_groot_n17_sonic_policy(
            payload=payload,
            command=command,
            n17_checkpoint=n17_checkpoint,
            sonic_checkpoint=sonic_checkpoint,
            groot_root=groot_root,
            wbc_root=wbc_root,
            policy_server_url=policy_server_url,
            sim2sim_command=sim2sim_command,
            timeout_seconds=float(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_TIMEOUT_SECONDS", "240")),
        )
        action = _mapping(response.get("action"))
        completed = exit_code == 0 and response.get("status") == "completed" and bool(action)
        blockers = [] if completed else list(response.get("blockers", []) or ["unitree_groot_n17_sonic_provider_smoke_blocked"])
        if policy_server_bootstrap.get("status") == "blocked":
            blockers.extend(policy_server_bootstrap.get("blockers", []))
        output = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "completed" if completed else "blocked",
            "policy_id": "unitree_groot_n17_sonic_policy",
            "unitree_groot_n17_sonic_model_executed": bool(response.get("model_ran")),
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(response.get("unitree_groot_n17_sonic_policy_action_command_ran")),
            "policy_action_model_command_ran": bool(response.get("unitree_groot_n17_sonic_policy_action_command_ran")),
            "action": action or None,
            "adapter_response": response,
            "policy_server_bootstrap": policy_server_bootstrap,
            "endpoint_closed_loop_policy_proven": False,
            "unitree_g1_dexterous_manipulation_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "blockers": sorted(set(blockers)),
        }
        _write_json(output_path, output)
        if policy_server_process is not None and policy_server_process.poll() is None:
            policy_server_process.terminate()
        return 0 if completed else 2
    except Exception as exc:
        _write_json(
            output_path,
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "status": "failed",
                "policy_id": "unitree_groot_n17_sonic_policy",
                "unitree_groot_n17_sonic_model_executed": False,
                "unitree_groot_n17_sonic_policy_action_command_ran": False,
                "policy_action_model_command_ran": False,
                "action": None,
                "traceback_tail": traceback.format_exc()[-4000:],
                "blockers": [f"unitree_groot_n17_sonic_provider_runner_failed:{type(exc).__name__}"],
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


RUN_SCRIPT = """#!/usr/bin/env bash
set +e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python3 unitree_groot_n17_sonic_provider_runner.py
runner_rc=$?
if [ $runner_rc -ne 0 ] && [ ! -f unitree_groot_n17_sonic_policy_provider_output.json ]; then
python3 - <<'PY'
import json
from pathlib import Path
Path("unitree_groot_n17_sonic_policy_provider_output.json").write_text(json.dumps({
    "schema_version": "unitree_groot_n17_sonic_policy_provider_output.v1",
    "status": "failed",
    "policy_id": "unitree_groot_n17_sonic_policy",
    "unitree_groot_n17_sonic_model_executed": False,
    "unitree_groot_n17_sonic_policy_action_command_ran": False,
    "policy_action_model_command_ran": False,
    "action": None,
    "blockers": [
        "unitree_groot_n17_sonic_provider_runner_failed_without_runtime_result",
        "blocked_unitree_groot_n17_sonic_process_exited_without_result"
    ],
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False
}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
fi
exit $runner_rc
"""


def _write_minimal_blueprint_runtime(runtime_dir: Path) -> list[str]:
    package_dir = runtime_dir / "blueprint_pipeline"
    ensure_dir(package_dir)
    copied: list[str] = []
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    copied.append("provider_runtime/blueprint_pipeline/__init__.py")
    source_dir = Path(__file__).resolve().parent
    for filename in (
        "unitree_groot_n17_sonic_policy_command_adapter.py",
        "unitree_groot_n17_sonic_policy_runtime.py",
        "unitree_groot_n17_sonic_policy_server_command.py",
    ):
        destination = package_dir / filename
        shutil.copy2(source_dir / filename, destination)
        copied.append(f"provider_runtime/blueprint_pipeline/{filename}")
    common_path = package_dir / "common.py"
    shutil.copy2(source_dir / "common.py", common_path)
    copied.append("provider_runtime/blueprint_pipeline/common.py")
    return copied


def _policy_input(
    *,
    frame_path: Path,
    task_id: str,
    task_prompt: str,
    policy_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    frame_reference = frame_path.name
    if policy_observation:
        observation = dict(policy_observation)
        observation["task_id"] = observation.get("task_id") or task_id
        if task_prompt and not any(
            observation.get(key) for key in ("task_prompt", "prompt", "task_description")
        ):
            observation["task_prompt"] = task_prompt
        visual = _mapping(observation.get("visual_observation"))
        visual["camera_frame_path"] = frame_reference
        observation["visual_observation"] = visual
        observation["camera_frame_path"] = frame_reference
        return {"observation": observation}
    return {
        "observation": {
            "schema_version": "blueprint_policy_observation.v1",
            "task_id": task_id,
            "task_prompt": task_prompt,
            "visual_observation": {"camera_frame_path": frame_reference},
            "object_state": {
                "object_id": "blueprint_light_object",
                "position": [0.36, -0.65, 0.27],
            },
            "route_task_state": {
                "target_pose": [0.54, -0.65, 0.79],
                "target_error_m": 0.8,
            },
        }
    }


def build_unitree_groot_n17_sonic_policy_provider_bundle(
    *,
    job_dir: str | Path,
    frame_path: str | Path,
    task_id: str = DEFAULT_TASK_ID,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    policy_command: str | None = None,
    n17_checkpoint: str | None = None,
    sonic_checkpoint: str | None = None,
    groot_root: str | None = None,
    wbc_root: str | None = None,
    policy_server_url: str | None = None,
    sim2sim_command: str | None = None,
    policy_observation_path: str | Path | None = None,
) -> dict[str, Any]:
    job = Path(job_dir)
    ensure_dir(job)
    runtime_dir = job / "provider_runtime"
    ensure_dir(runtime_dir)
    frame_copy = runtime_dir / "input_frame.png"
    _copy_frame(Path(frame_path).expanduser(), frame_copy)
    policy_observation = _load_policy_observation(policy_observation_path)
    payload = _policy_input(
        frame_path=frame_copy,
        task_id=task_id,
        task_prompt=task_prompt,
        policy_observation=policy_observation,
    )
    policy_input_path = runtime_dir / "policy_input.json"
    write_json(policy_input_path, payload)
    runner_path = runtime_dir / "unitree_groot_n17_sonic_provider_runner.py"
    _write_executable(runner_path, PROVIDER_RUNNER)
    run_script_path = runtime_dir / "run_unitree_groot_n17_sonic_provider_runtime.sh"
    _write_executable(run_script_path, RUN_SCRIPT)
    bundled_blueprint_modules = _write_minimal_blueprint_runtime(runtime_dir)
    runtime_execution_blockers: list[str] = []
    if not policy_command:
        runtime_execution_blockers.append("blocked_missing_unitree_groot_n17_sonic_policy_command")
    if not n17_checkpoint:
        runtime_execution_blockers.append("blocked_missing_unitree_groot_n17_checkpoint")
    if not sonic_checkpoint:
        runtime_execution_blockers.append("blocked_missing_unitree_g1_sonic_checkpoint")
    manifest = {
        "schema_version": "unitree_groot_n17_sonic_policy_provider_bundle.v1",
        "generated_at": utc_now_iso(),
        "status": "bundle_ready",
        "policy_id": POLICY_ID,
        "runtime_entrypoint": "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh",
        "runner_path": str(runner_path),
        "policy_input_path": str(policy_input_path),
        "input_frame_path": str(frame_copy),
        "policy_observation_path": str(Path(policy_observation_path).expanduser())
        if policy_observation_path
        else None,
        "policy_observation_preserved": policy_observation is not None,
        "bundled_blueprint_modules": bundled_blueprint_modules,
        "policy_command_configured": bool(policy_command),
        "n17_checkpoint_configured": bool(n17_checkpoint),
        "g1_sonic_checkpoint_configured": bool(sonic_checkpoint),
        "groot_root_configured": bool(groot_root),
        "wbc_root_configured": bool(wbc_root),
        "policy_server_url_configured": bool(policy_server_url),
        "sim2sim_command_configured": bool(sim2sim_command),
        "local_bundle_ready_for_remote_staging": not runtime_execution_blockers,
        "ready_for_fresh_model_execution": not runtime_execution_blockers,
        "runtime_execution_blockers": runtime_execution_blockers,
        "expected_output_filename": PROVIDER_OUTPUT_FILENAME,
        "env_contract": {
            POLICY_COMMAND_ENV: "<configured>" if policy_command else None,
            N17_CHECKPOINT_ENV: "<configured>" if n17_checkpoint else None,
            SONIC_CHECKPOINT_ENV: "<configured>" if sonic_checkpoint else None,
            GROOT_ROOT_ENV: "<configured>" if groot_root else None,
            WBC_ROOT_ENV: "<configured>" if wbc_root else None,
            POLICY_SERVER_URL_ENV: "<configured>" if policy_server_url else None,
            SIM2SIM_COMMAND_ENV: "<configured>" if sim2sim_command else None,
        },
        "truth_boundary": {
            "provider_bundle_is_not_model_execution": True,
            "unitree_groot_n17_sonic_policy_action_command_ran": False,
            "unitree_g1_dexterous_manipulation_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    manifest_path = runtime_dir / "unitree_groot_n17_sonic_policy_provider_manifest.json"
    write_json(manifest_path, manifest)
    bundle_path = job / DEFAULT_BUNDLE_FILENAME
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(runtime_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(job).as_posix())
    with zipfile.ZipFile(bundle_path) as archive:
        bundle_file_count = len(archive.namelist())
    manifest.update(
        {
            "bundle_path": str(bundle_path),
            "manifest_path": str(manifest_path),
            "bundle_file_count": bundle_file_count,
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def import_unitree_groot_n17_sonic_provider_output(
    *,
    provider_output_zip: str | Path,
    extraction_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    source_zip = Path(provider_output_zip).expanduser()
    extract_dir = Path(extraction_dir)
    ensure_dir(extract_dir)
    with zipfile.ZipFile(source_zip) as archive:
        archive.extractall(extract_dir)
    candidates = [
        extract_dir / PROVIDER_OUTPUT_FILENAME,
        extract_dir / "provider_runtime" / PROVIDER_OUTPUT_FILENAME,
    ]
    payload_path = next((path for path in candidates if path.is_file()), None)
    if payload_path is None:
        payload = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "blocked",
            "unitree_groot_n17_sonic_model_executed": False,
            "unitree_groot_n17_sonic_policy_action_command_ran": False,
            "policy_action_model_command_ran": False,
            "action": None,
            "blockers": ["unitree_groot_n17_sonic_provider_output_json_missing"],
        }
    else:
        value = json.loads(payload_path.read_text(encoding="utf-8"))
        payload = dict(value) if isinstance(value, Mapping) else {}
    completed = bool(
        payload.get("status") == "completed"
        and payload.get("unitree_groot_n17_sonic_model_executed") is True
        and payload.get("unitree_groot_n17_sonic_policy_action_command_ran") is True
        and isinstance(payload.get("action"), Mapping)
    )
    imported = {
        "schema_version": "unitree_groot_n17_sonic_policy_provider_import.v1",
        "status": "completed" if completed else "blocked",
        "provider_output_zip": str(source_zip),
        "provider_output_path": str(payload_path) if payload_path else None,
        "unitree_groot_n17_sonic_model_executed": bool(
            payload.get("unitree_groot_n17_sonic_model_executed")
        ),
        "unitree_groot_n17_sonic_policy_action_command_ran": bool(
            payload.get("unitree_groot_n17_sonic_policy_action_command_ran")
        ),
        "policy_action_model_command_ran": bool(payload.get("policy_action_model_command_ran")),
        "action": dict(payload["action"]) if isinstance(payload.get("action"), Mapping) else None,
        "truth_boundary": {
            "provider_output_import_is_not_fresh_local_policy_execution": True,
            "provider_output_import_is_not_closed_loop_endpoint_control": True,
            "unitree_g1_dexterous_manipulation_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "blockers": list(
            payload.get("blockers", [])
            or ([] if completed else ["unitree_groot_n17_sonic_provider_output_not_completed"])
        ),
    }
    write_json(Path(output_path), imported)
    return imported


def run_unitree_groot_n17_sonic_policy_provider_smoke(
    *,
    job_dir: str | Path,
    frame_path: str | Path,
    provider_output_zip: str | Path | None = None,
    task_id: str = DEFAULT_TASK_ID,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    dry_run: bool = True,
    policy_command: str | None = None,
    n17_checkpoint: str | None = None,
    sonic_checkpoint: str | None = None,
    groot_root: str | None = None,
    wbc_root: str | None = None,
    policy_server_url: str | None = None,
    sim2sim_command: str | None = None,
    policy_observation_path: str | Path | None = None,
) -> dict[str, Any]:
    job = Path(job_dir)
    ensure_dir(job)
    if provider_output_zip:
        imported = import_unitree_groot_n17_sonic_provider_output(
            provider_output_zip=provider_output_zip,
            extraction_dir=job / "provider_output",
            output_path=job / "unitree_groot_n17_sonic_policy_provider_import.json",
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": imported["status"],
            "job_dir": str(job),
            "policy_id": POLICY_ID,
            "unitree_groot_n17_sonic_model_executed": bool(
                imported.get("unitree_groot_n17_sonic_model_executed")
            ),
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                imported.get("unitree_groot_n17_sonic_policy_action_command_ran")
            ),
            "policy_action_model_command_ran": bool(imported.get("policy_action_model_command_ran")),
            "action": imported.get("action"),
            "blockers": imported.get("blockers", []),
            "truth_boundary": imported.get("truth_boundary", {}),
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        write_json(job / "unitree_groot_n17_sonic_policy_provider_smoke_summary.json", summary)
        return summary

    bundle = build_unitree_groot_n17_sonic_policy_provider_bundle(
        job_dir=job,
        frame_path=frame_path,
        task_id=task_id,
        task_prompt=task_prompt,
        policy_command=policy_command,
        n17_checkpoint=n17_checkpoint,
        sonic_checkpoint=sonic_checkpoint,
        groot_root=groot_root,
        wbc_root=wbc_root,
        policy_server_url=policy_server_url,
        sim2sim_command=sim2sim_command,
        policy_observation_path=policy_observation_path,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "dry_run_ready" if dry_run else "blocked_provider_launch_not_implemented",
        "job_dir": str(job),
        "policy_id": POLICY_ID,
        "bundle_manifest_path": bundle["manifest_path"],
        "bundle_path": bundle["bundle_path"],
        "ready_for_fresh_model_execution": bool(bundle.get("ready_for_fresh_model_execution")),
        "runtime_execution_blockers": list(bundle.get("runtime_execution_blockers", [])),
        "unitree_groot_n17_sonic_model_executed": False,
        "unitree_groot_n17_sonic_policy_action_command_ran": False,
        "policy_action_model_command_ran": False,
        "action": None,
        "blockers": [] if dry_run else ["blocked_provider_launch_not_implemented"],
        "truth_boundary": {
            "dry_run_is_not_model_execution": True,
            "unitree_g1_dexterous_manipulation_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job / "unitree_groot_n17_sonic_policy_provider_smoke_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--frame-path", type=Path, required=True)
    parser.add_argument("--provider-output-zip", type=Path)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--task-prompt", default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--policy-command")
    parser.add_argument("--n17-checkpoint")
    parser.add_argument("--sonic-checkpoint")
    parser.add_argument("--groot-root")
    parser.add_argument("--wbc-root")
    parser.add_argument("--policy-server-url")
    parser.add_argument("--sim2sim-command")
    parser.add_argument("--policy-observation-path", type=Path)
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args(argv)
    summary = run_unitree_groot_n17_sonic_policy_provider_smoke(
        job_dir=args.job_dir,
        frame_path=args.frame_path,
        provider_output_zip=args.provider_output_zip,
        task_id=args.task_id,
        task_prompt=args.task_prompt,
        dry_run=args.dry_run,
        policy_command=args.policy_command or os.getenv(POLICY_COMMAND_ENV),
        n17_checkpoint=args.n17_checkpoint or os.getenv(N17_CHECKPOINT_ENV),
        sonic_checkpoint=args.sonic_checkpoint or os.getenv(SONIC_CHECKPOINT_ENV),
        groot_root=args.groot_root or os.getenv(GROOT_ROOT_ENV),
        wbc_root=args.wbc_root or os.getenv(WBC_ROOT_ENV),
        policy_server_url=args.policy_server_url or os.getenv(POLICY_SERVER_URL_ENV),
        sim2sim_command=args.sim2sim_command or os.getenv(SIM2SIM_COMMAND_ENV),
        policy_observation_path=args.policy_observation_path,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary.get("status") in {"completed", "dry_run_ready"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
