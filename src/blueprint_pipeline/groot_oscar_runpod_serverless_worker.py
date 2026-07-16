"""RunPod Serverless worker for the sealed GR00T + OSCAR + Isaac release.

The worker deliberately exposes a tiny allow-listed job surface.  RunPod's
queue supplies only operation metadata; large inputs and outputs live on the
attached network volume.  This keeps signed URLs and provider credentials out
of job payloads and lets one active FlashBoot worker serve the strict policy
probe, kitchen smoke, and dynamic episodes without another image pull.

This module owns execution only.  Endpoint/template allocation and teardown
remain behind :mod:`blueprint_pipeline.paid_resource_allocator`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


SCHEMA_VERSION = "groot_oscar_runpod_serverless_worker.v1"
NETWORK_VOLUME_ROOT = Path("/runpod-volume")
MODEL_CACHE_RELATIVE = Path(".blueprint-model-cache/blueprint-groot-oscar-v1")
RUNTIME_CACHE_LINK = Path("/workspace/.blueprint-model-cache")
ALLOWED_OPERATIONS = frozenset(
    {"startup", "strict-policy-smoke", "robot-eval-worker", "kitchen-campaign"}
)
MAX_STRICT_SECONDS = 300
MAX_EVAL_SECONDS = 900
MAX_CAMPAIGN_SECONDS = 3_600
OSCAR_EXECUTION_PYTHON = Path("/opt/oscar-venv/bin/python")
RUNPOD_SDK_VERSION = "1.10.1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_volume_path(relative: Any, *, must_exist: bool = True) -> Path:
    text = str(relative or "").strip().replace("\\", "/")
    candidate = Path(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("network_volume_relative_path_invalid")
    root = NETWORK_VOLUME_ROOT.resolve()
    resolved = (root / candidate).resolve(strict=must_exist)
    if resolved != root and root not in resolved.parents:
        raise ValueError("network_volume_path_escape")
    return resolved


def _base_result(operation: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "operation": operation,
        "status": "blocked",
        "blockers": [],
        "source_commit": os.getenv("BLUEPRINT_SOURCE_COMMIT", "").strip(),
        "worker_image_digest": os.getenv("BLUEPRINT_WORKER_IMAGE_DIGEST", "").strip(),
        "model_manifest_digest": os.getenv(
            "BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST", ""
        ).strip(),
        "provider": "runpod_serverless",
        "runtime_worker_identity_sha256": hashlib.sha256(
            (
                "runpod-worker:"
                + str(
                    os.getenv("RUNPOD_POD_ID")
                    or os.getenv("RUNPOD_WORKER_ID")
                    or socket.gethostname()
                )
            ).encode()
        ).hexdigest(),
        "network_volume_payload_used": True,
        "raw_secret_values_recorded": False,
        "physical_robot_control_performed": False,
    }


def _ensure_runtime_cache_link() -> bool:
    source = (NETWORK_VOLUME_ROOT / MODEL_CACHE_RELATIVE.parent).resolve()
    if not source.is_dir():
        return False
    if RUNTIME_CACHE_LINK.is_symlink():
        return RUNTIME_CACHE_LINK.resolve() == source
    if RUNTIME_CACHE_LINK.exists():
        return False
    RUNTIME_CACHE_LINK.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_CACHE_LINK.symlink_to(source, target_is_directory=True)
    return RUNTIME_CACHE_LINK.resolve() == source


def _startup() -> dict[str, Any]:
    from .groot_oscar_model_cache import activate_model_cache

    result = _base_result("startup")
    cache = NETWORK_VOLUME_ROOT / MODEL_CACHE_RELATIVE
    runtime_cache = RUNTIME_CACHE_LINK / MODEL_CACHE_RELATIVE.name
    link_ready = _ensure_runtime_cache_link()
    activation = (
        activate_model_cache(
            cache,
            expected_manifest_digest=result["model_manifest_digest"],
            runtime_cache_root=runtime_cache,
        )
        if link_ready and cache.is_dir()
        else {"status": "blocked", "runtime_links_activated": False}
    )
    checks = {
        "network_volume_mounted": NETWORK_VOLUME_ROOT.is_dir(),
        "model_cache_present": cache.is_dir(),
        "canonical_runtime_cache_link_ready": link_ready,
        "model_cache_runtime_verified": activation.get("status") == "passed",
        "gear_sonic_model_links_activated": (
            activation.get("runtime_links_activated") is True
        ),
        "oscar_python_present": Path("/opt/oscar-venv/bin/python").is_file(),
        "groot_python_present": Path("/opt/gr00t-venv/bin/python").is_file(),
        "isaac_python_present": Path("/isaac-sim/python.sh").is_file(),
        "groot_source_present": Path("/opt/gr00t").is_dir(),
        "oscar_source_present": Path("/opt/OSCAR").is_dir(),
    }
    try:
        import torch

        checks["torch_cuda_available"] = bool(torch.cuda.is_available())
        result["torch_version"] = str(torch.__version__)
        if checks["torch_cuda_available"]:
            result["gpu_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:  # pragma: no cover - image runtime only
        checks["torch_cuda_available"] = False
        result["torch_error_type"] = type(exc).__name__
    blockers = [name for name, passed in checks.items() if not passed]
    result.update(
        {
            "status": "completed" if not blockers else "blocked",
            "blockers": blockers,
            "checks": checks,
            "runtime_present": not blockers,
            "model_execution_proven": False,
            "simulator_execution_proven": False,
            "semantic_task_success": None,
            "model_cache_verification": {
                "status": activation.get("status"),
                "model_manifest_digest": activation.get("model_manifest_digest"),
                "verified_file_count": activation.get("verified_file_count"),
                "verified_size_bytes": activation.get("verified_size_bytes"),
                "runtime_links_activated": activation.get("runtime_links_activated"),
                "blockers": list(activation.get("blockers") or []),
            },
        }
    )
    return result


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            # The process group exited after the initial poll.
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # SIGKILL has been sent; the kernel may still be reaping the child.
            return


def _strict_policy_smoke() -> dict[str, Any]:
    from PIL import Image

    from .unitree_groot_n17_sonic_policy_server_command import (
        REQUIRED_STATE_DIMS,
        run_policy_server_command,
    )

    result = _base_result("strict-policy-smoke")
    work = Path(tempfile.mkdtemp(prefix="blueprint-serverless-strict-policy-"))
    frame = work / "strict_policy_smoke_frame.png"
    Image.new("RGB", (640, 480), color=(96, 112, 128)).save(frame)
    state = {key: [0.0] * size for key, size in REQUIRED_STATE_DIMS.items()}
    payload = {
        "observation": {
            "task_id": "groot_oscar_strict_learned_action_smoke",
            "task_prompt": "Return one safe simulated Unitree G1 SONIC action chunk.",
            "visual_observation": {"camera_frame_path": str(frame)},
            "unitree_g1_sonic_state": state,
            "unitree_g1_sonic_state_source": "canonical_strict_smoke_contract_probe",
            "unitree_g1_sonic_state_metadata": {"complete": True},
        }
    }
    cache = NETWORK_VOLUME_ROOT / MODEL_CACHE_RELATIVE
    log_path = work / "gr00t_policy_server.log"
    started = time.monotonic()
    with log_path.open("wb") as log:
        server = subprocess.Popen(
            [
                "/opt/gr00t-venv/bin/python",
                "-m",
                "gr00t.eval.run_gr00t_server",
                "--model-path",
                str(cache / "sonic"),
                "--embodiment-tag",
                "UNITREE_G1_SONIC",
                "--device",
                "cuda:0",
                "--port",
                "5550",
            ],
            cwd="/opt/gr00t",
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        actions: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        blockers: list[str] = []
        deadline = started + 240.0
        try:
            while len(actions) < 3 and time.monotonic() < deadline:
                action_result, exit_code = run_policy_server_command(
                    payload=payload,
                    policy_server_url="tcp://127.0.0.1:5550",
                    groot_root="/opt/gr00t",
                    timeout_ms=15_000,
                )
                attempts.append(
                    {
                        "attempt": len(attempts) + 1,
                        "status": action_result.get("status"),
                        "exit_code": exit_code,
                        "blockers": list(action_result.get("blockers") or []),
                        "model_ran": action_result.get("model_ran") is True,
                        "action_present": isinstance(action_result.get("action"), Mapping),
                    }
                )
                if (
                    exit_code == 0
                    and action_result.get("status") == "completed"
                    and action_result.get("model_ran") is True
                    and action_result.get(
                        "unitree_groot_n17_sonic_policy_action_command_ran"
                    )
                    is True
                    and isinstance(action_result.get("action"), Mapping)
                ):
                    actions.append(dict(action_result["action"]))
                elif server.poll() is not None:
                    blockers.append("strict_policy_smoke_server_exited_before_three_actions")
                    break
                else:
                    time.sleep(5)
            if len(actions) != 3 and not blockers:
                blockers.append("strict_policy_smoke_three_actions_not_completed_before_deadline")
        finally:
            _terminate_process(server)
    completed = len(actions) == 3 and not blockers
    result.update(
        {
            "status": "completed" if completed else "blocked",
            "blockers": blockers,
            "requested_action_count": 3,
            "completed_action_count": len(actions),
            "fresh_learned_action_trace": actions,
            "attempts": attempts,
            "model_execution_proven": completed,
            "policy_action_model_command_ran": completed,
            "simulator_execution_proven": False,
            "semantic_task_success": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "log_sha256": _sha256_file(log_path) if log_path.is_file() else None,
        }
    )
    shutil.rmtree(work, ignore_errors=True)
    return result


def _robot_eval(job_input: Mapping[str, Any]) -> dict[str, Any]:
    from .robot_eval_worker import run_robot_eval_worker

    result = _base_result("robot-eval-worker")
    manifest = _safe_volume_path(job_input.get("manifest_relative_path"))
    expected = str(job_input.get("manifest_sha256") or "").strip().lower()
    if len(expected) != 64 or _sha256_file(manifest) != expected:
        result["blockers"] = ["worker_manifest_sha256_mismatch"]
        return result
    output_relative = str(job_input.get("output_relative_path") or "").strip()
    output = _safe_volume_path(output_relative, must_exist=False)
    output.mkdir(parents=True, exist_ok=True)
    timeout_seconds = int(job_input.get("timeout_seconds") or 0)
    if not 30 <= timeout_seconds <= MAX_EVAL_SECONDS:
        result["blockers"] = ["worker_timeout_out_of_range"]
        return result
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f"blueprint-serverless-eval-{expected[:16]}-"
    ) as raw_work_dir:
        worker = run_robot_eval_worker(
            manifest_uri=str(manifest),
            work_dir=Path(raw_work_dir),
            provisioner="runpod",
            allow_gpu_provisioning=False,
            allow_simulator_execution=True,
            allowed_simulators=("isaac_sim",),
            timeout_seconds=timeout_seconds,
            artifact_output_uri=str(output),
            artifact_output_uri_required=True,
        )
    worker_status = str(worker.get("status") or "blocked")
    public_worker = {
        "schema_version": worker.get("schema_version"),
        "status": worker_status,
        "job_id": worker.get("job_id"),
        "simulator": worker.get("simulator"),
        "job_status": worker.get("job_status"),
        "evaluation_status": worker.get("evaluation_status"),
        "simulator_execution_proven": worker.get("simulator_execution_proven") is True,
        "policy_execution_proven": worker.get("policy_execution_proven") is True,
        "semantic_task_success": worker.get("semantic_task_success"),
        "blockers": list(worker.get("blockers") or worker.get("job_blockers") or []),
    }
    result.update(
        {
            "status": "completed" if worker_status == "completed" else "blocked",
            "blockers": public_worker["blockers"],
            "worker_result": public_worker,
            "output_relative_path": output_relative,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "simulator_execution_proven": public_worker["simulator_execution_proven"],
            "policy_execution_proven": public_worker["policy_execution_proven"],
            "semantic_task_success": public_worker["semantic_task_success"],
        }
    )
    return result


def _kitchen_campaign(job_input: Mapping[str, Any]) -> dict[str, Any]:
    from .groot_oscar_runpod_serverless_campaign_worker import run_kitchen_campaign

    result = _base_result("kitchen-campaign")
    expected_worker = str(job_input.get("expected_runtime_worker_identity_sha256") or "")
    if expected_worker != result["runtime_worker_identity_sha256"]:
        result["blockers"] = ["campaign_runtime_worker_identity_mismatch"]
        return result
    startup = _startup()
    if startup.get("status") != "completed" or startup.get("runtime_present") is not True:
        result["blockers"] = ["campaign_runtime_or_model_cache_not_ready"]
        result["startup"] = startup
        return result
    campaign = run_kitchen_campaign(
        network_volume_root=NETWORK_VOLUME_ROOT,
        campaign_manifest_relative_path=str(
            job_input.get("campaign_manifest_relative_path") or ""
        ),
        campaign_manifest_sha256=str(job_input.get("campaign_manifest_sha256") or ""),
        output_relative_path=str(job_input.get("output_relative_path") or ""),
        source_commit=str(result["source_commit"]),
        image_ref=str(result["worker_image_digest"]),
        model_manifest_digest=str(result["model_manifest_digest"]),
    )
    return {
        **result,
        **campaign,
        "operation": "kitchen-campaign",
        "startup": startup,
    }


def _execute_job(job_input: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(job_input.get("operation") or "").strip()
    if operation not in ALLOWED_OPERATIONS:
        result = _base_result(operation or "missing")
        result["blockers"] = ["serverless_operation_not_allowed"]
        return result
    try:
        if operation == "startup":
            return _startup()
        if operation == "strict-policy-smoke":
            return _strict_policy_smoke()
        if operation == "kitchen-campaign":
            return _kitchen_campaign(job_input)
        return _robot_eval(job_input)
    except Exception as exc:  # fail closed without echoing job inputs or secrets
        result = _base_result(operation)
        result["blockers"] = [f"serverless_worker_exception:{type(exc).__name__}"]
        return result


def _isolated_timeout(job_input: Mapping[str, Any]) -> int:
    operation = str(job_input.get("operation") or "").strip()
    if operation == "startup":
        return 180
    if operation == "strict-policy-smoke":
        return MAX_STRICT_SECONDS + 15
    if operation == "kitchen-campaign":
        return MAX_CAMPAIGN_SECONDS + 30
    try:
        requested = min(
            MAX_EVAL_SECONDS, max(30, int(job_input.get("timeout_seconds") or 0))
        )
        return requested + 15
    except (TypeError, ValueError):
        return 30


def _run_in_oscar_runtime(job_input: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(job_input.get("operation") or "").strip()
    if operation not in ALLOWED_OPERATIONS:
        return _execute_job(job_input)
    with tempfile.TemporaryDirectory(prefix="blueprint-serverless-") as raw_job_dir:
        job_dir = Path(raw_job_dir)
        input_path = job_dir / "input.json"
        result_path = job_dir / "result.json"
        input_path.write_text(
            json.dumps(dict(job_input), sort_keys=True), encoding="utf-8"
        )
        input_path.chmod(0o600)
        process = subprocess.Popen(
            [
                str(OSCAR_EXECUTION_PYTHON),
                "-m",
                "blueprint_pipeline.groot_oscar_runpod_serverless_worker",
                "--execute-job-input",
                str(input_path),
                "--execute-job-result",
                str(result_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            process.wait(timeout=_isolated_timeout(job_input))
        except subprocess.TimeoutExpired:
            _terminate_process(process)
            result = _base_result(operation)
            result["blockers"] = ["serverless_isolated_execution_timeout"]
            return result
        if process.returncode != 0 or not result_path.is_file():
            result = _base_result(operation)
            result["blockers"] = ["serverless_isolated_execution_failed"]
            return result
        try:
            value = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            value = None
        if not isinstance(value, Mapping):
            result = _base_result(operation)
            result["blockers"] = ["serverless_isolated_result_invalid"]
            return result
        return dict(value)


def handler(event: Mapping[str, Any]) -> dict[str, Any]:
    job_input = _mapping(_mapping(event).get("input"))
    return _run_in_oscar_runtime(job_input)


def _execute_job_files(input_path: Path, result_path: Path) -> int:
    try:
        value = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        value = None
    job_input = dict(value) if isinstance(value, Mapping) else {}
    result = _execute_job(job_input)
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    result_path.chmod(0o600)
    return 0 if result.get("status") in {"completed", "blocked"} else 2


def _namespace_package(name: str, path: Path) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    module.__spec__ = spec
    return module


def _load_runpod_serverless() -> Any:
    """Load only the queue-worker portion of the exact pinned RunPod SDK."""

    distribution = importlib.metadata.distribution("runpod")
    if distribution.version != RUNPOD_SDK_VERSION:
        raise RuntimeError("runpod_serverless_sdk_version_mismatch")
    package_root = Path(distribution.locate_file("runpod")).resolve()
    if not (package_root / "serverless" / "__init__.py").is_file():
        raise RuntimeError("runpod_serverless_sdk_package_missing")

    # runpod.__init__ eagerly imports CLI, SSH, and local FastAPI tooling that a
    # private queue worker never calls. Likewise, serverless.utils eagerly
    # imports optional upload helpers. Namespace loading preserves the exact
    # SDK serverless implementation without pulling those unrelated packages
    # into the independently deployable thin release.
    sys.modules.pop("runpod", None)
    sys.modules.pop("runpod.serverless.utils", None)
    sys.modules["runpod"] = _namespace_package("runpod", package_root)
    sys.modules["runpod.cli"] = _namespace_package(
        "runpod.cli", package_root / "cli"
    )
    sys.modules["runpod.cli.groups"] = _namespace_package(
        "runpod.cli.groups", package_root / "cli" / "groups"
    )
    sys.modules["runpod.cli.groups.config"] = _namespace_package(
        "runpod.cli.groups.config", package_root / "cli" / "groups" / "config"
    )
    sys.modules["runpod.serverless.utils"] = _namespace_package(
        "runpod.serverless.utils", package_root / "serverless" / "utils"
    )
    return importlib.import_module("runpod.serverless")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-job-input")
    parser.add_argument("--execute-job-result")
    parser.add_argument("--verify-serverless-runtime", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    if args.execute_job_input or args.execute_job_result:
        if not args.execute_job_input or not args.execute_job_result:
            return 2
        return _execute_job_files(
            Path(args.execute_job_input), Path(args.execute_job_result)
        )
    serverless = _load_runpod_serverless()
    if args.verify_serverless_runtime:
        return 0
    sys.argv = [sys.argv[0], *remaining]

    serverless.start({"handler": handler})
    return 0


if __name__ == "__main__":  # pragma: no cover - provider entrypoint
    raise SystemExit(main())
