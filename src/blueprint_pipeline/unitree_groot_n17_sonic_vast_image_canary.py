"""Run a strict Vast startup canary for the Unitree GR00T/SONIC image.

The canary proves only that the selected container image starts on Vast, sees a
GPU, downloads a Blueprint provider bundle, runs the canary entrypoint, and
uploads provider output. It does not load GR00T, run policy inference, or prove
robot readiness.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .unitree_groot_n17_sonic_provider_smoke import (
    DEFAULT_BUNDLE_FILENAME,
    OUTPUT_SCHEMA_VERSION as PROVIDER_OUTPUT_SCHEMA_VERSION,
    PROVIDER_OUTPUT_FILENAME,
    _write_executable,
    _write_minimal_blueprint_runtime,
)
from .unitree_groot_n17_sonic_vast_policy_command import (
    EXCLUDED_MACHINE_ID_ENVS,
    OBJECT_STORE_KEY_PREFIX_ENV,
    PUBLIC_IMAGE_ENV,
    VAST_LAUNCH_MODE_ENV,
    _excluded_machine_ids_from_env,
    _float_env,
    _int_env,
    _string,
    _truthy,
)
from .vast_provider_adapter import (
    DEFAULT_PUBLIC_CUDA_IMAGE,
    VAST_IMAGE_LOGIN_MODE_ENV,
    run_vast_provider_adapter,
)
from .wam_provider_object_store import stage_wam_provider_bundle_object_store


SCHEMA_VERSION = "unitree_groot_n17_sonic_vast_image_canary.v1"
OUTPUT_SCHEMA_VERSION = PROVIDER_OUTPUT_SCHEMA_VERSION
CANARY_MARKER = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_VAST_IMAGE_CANARY_OK"
DEFAULT_OBJECT_STORE_KEY_PREFIX = "blueprint/unitree-groot-sonic-canary"
DEFAULT_MIN_GPU_RAM_MB = 48000

_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c4944415408d763f8ffff3f0005fe02fea73581e80000000049454e44ae426082"
)


CANARY_RUNNER = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_SCHEMA_VERSION = "unitree_groot_n17_sonic_policy_provider_output.v1"
CANARY_MARKER = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_VAST_IMAGE_CANARY_OK"


def run(cmd: list[str]) -> dict[str, object]:
    started = time.time()
    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=45,
            check=False,
        )
        return {
            "command": cmd,
            "returncode": process.returncode,
            "duration_seconds": round(time.time() - started, 3),
            "output_tail": (process.stdout or "")[-4000:],
        }
    except Exception as exc:
        return {
            "command": cmd,
            "returncode": None,
            "duration_seconds": round(time.time() - started, 3),
            "error_type": type(exc).__name__,
        }


out_dir = Path(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR", "runtime_output"))
out_path = Path(
    os.environ.get(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT",
        out_dir / "unitree_groot_n17_sonic_policy_provider_output.json",
    )
)
out_dir.mkdir(parents=True, exist_ok=True)
checks = {
    "python": run([sys.executable, "--version"]),
    "nvidia_smi": run([
        "bash",
        "-lc",
        "nvidia-smi -L && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
    ]),
}
blockers = []
if checks["python"].get("returncode") != 0:
    blockers.append("canary_python_check_failed")
if checks["nvidia_smi"].get("returncode") != 0:
    blockers.append("canary_nvidia_smi_failed")
completed = not blockers
payload = {
    "schema_version": OUTPUT_SCHEMA_VERSION,
    "status": "completed" if completed else "blocked",
    "canary_only": True,
    "canary_marker": CANARY_MARKER if completed else None,
    "blockers": blockers,
    "unitree_groot_n17_sonic_model_executed": False,
    "unitree_groot_n17_sonic_policy_action_command_ran": False,
    "policy_action_model_command_ran": False,
    "action": None,
    "checks": checks,
    "truth_boundary": {
        "canary_is_not_policy_inference": True,
        "canary_is_not_model_execution": True,
        "unitree_g1_dexterous_manipulation_proven": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "safety_validation_proven": False,
    },
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if completed:
    print(CANARY_MARKER, flush=True)
raise SystemExit(0 if completed else 2)
'''


CANARY_RUN_SCRIPT = """#!/usr/bin/env bash
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _write_canary_frame(path: Path) -> None:
    ensure_dir(path.parent)
    path.write_bytes(_TINY_PNG)


def build_unitree_groot_n17_sonic_vast_image_canary_bundle(
    *,
    job_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    job = Path(job_dir).expanduser().resolve()
    if job.exists():
        shutil.rmtree(job)
    runtime_dir = job / "provider_runtime"
    ensure_dir(runtime_dir)
    generated = generated_at or utc_now_iso()
    frame_path = runtime_dir / "input_frame.png"
    _write_canary_frame(frame_path)
    write_json(
        runtime_dir / "policy_input.json",
        {
            "schema_version": "unitree_groot_n17_sonic_vast_image_canary_input.v1",
            "generated_at": generated,
            "canary_only": True,
            "task_id": "vast_image_canary",
            "truth_boundary": {
                "canary_is_not_policy_inference": True,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
                "safety_validation_proven": False,
            },
        },
    )
    _write_executable(runtime_dir / "unitree_groot_n17_sonic_provider_runner.py", CANARY_RUNNER)
    _write_executable(
        runtime_dir / "run_unitree_groot_n17_sonic_provider_runtime.sh",
        CANARY_RUN_SCRIPT,
    )
    bundled_modules = _write_minimal_blueprint_runtime(runtime_dir)
    manifest = {
        "schema_version": "unitree_groot_n17_sonic_policy_provider_bundle.v1",
        "generated_at": generated,
        "status": "canary_bundle_ready",
        "canary_only": True,
        "canary_marker": CANARY_MARKER,
        "policy_id": "unitree_groot_n17_sonic_policy",
        "runtime_entrypoint": "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh",
        "runner_path": str(runtime_dir / "unitree_groot_n17_sonic_provider_runner.py"),
        "policy_input_path": str(runtime_dir / "policy_input.json"),
        "input_frame_path": str(frame_path),
        "bundled_blueprint_modules": bundled_modules,
        "local_bundle_ready_for_remote_staging": True,
        "ready_for_fresh_model_execution": False,
        "runtime_execution_blockers": [],
        "expected_output_filename": PROVIDER_OUTPUT_FILENAME,
        "truth_boundary": {
            "canary_bundle_is_not_model_execution": True,
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
    write_json(
        job / "vast_unitree_groot_sonic_image_canary_bundle_summary.json",
        {
            "schema_version": "vast_unitree_groot_sonic_image_canary_bundle.v1",
            "generated_at": generated,
            "status": "completed",
            "bundle_path": str(bundle_path),
            "manifest_path": str(manifest_path),
            "bundle_file_count": bundle_file_count,
            "raw_secret_values_recorded": False,
        },
    )
    return manifest


def _provider_output_from_zip(path: Path, extraction_dir: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "blocked",
            "blockers": ["canary_provider_output_zip_missing"],
        }
    if extraction_dir.exists():
        shutil.rmtree(extraction_dir)
    ensure_dir(extraction_dir)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(extraction_dir)
    candidates = [
        extraction_dir / PROVIDER_OUTPUT_FILENAME,
        extraction_dir / "provider_runtime" / PROVIDER_OUTPUT_FILENAME,
    ]
    output_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if output_path is None:
        return {
            "status": "blocked",
            "blockers": ["canary_provider_output_json_missing"],
        }
    payload = _read_json(output_path)
    payload["provider_output_path"] = str(output_path)
    return payload


def _safe_summary(
    *,
    generated_at: str,
    job_dir: Path,
    canary_result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "unitree_groot_n17_sonic_vast_image_canary_safe_summary.v1",
        "generated_at": generated_at,
        "status": canary_result.get("status"),
        "job_dir": str(job_dir),
        "public_image": canary_result.get("public_image"),
        "vast_run_dir": canary_result.get("vast_run_dir"),
        "min_gpu_ram_mb": canary_result.get("min_gpu_ram_mb"),
        "selected_offer": canary_result.get("selected_offer"),
        "vast_instance_ids": canary_result.get("vast_instance_ids"),
        "estimated_cost_usd": canary_result.get("estimated_cost_usd"),
        "continuing_spend_from_this_run": canary_result.get(
            "continuing_spend_from_this_run"
        ),
        "heartbeat_completed": canary_result.get("heartbeat_completed"),
        "gpu_sanity_completed": canary_result.get("gpu_sanity_completed"),
        "provider_bundle_downloaded_and_ran": canary_result.get(
            "provider_bundle_downloaded_and_ran"
        ),
        "provider_output_upload_ok": canary_result.get("provider_output_upload_ok"),
        "canary_marker_observed": canary_result.get("canary_marker_observed"),
        "blockers": canary_result.get("blockers"),
        "raw_secret_values_recorded": False,
    }


def run_unitree_groot_n17_sonic_vast_image_canary(
    *,
    job_dir: str | Path,
    public_image: str | None = None,
    min_gpu_ram_mb: int = DEFAULT_MIN_GPU_RAM_MB,
    launch_mode: str | None = None,
    max_hourly_rate: float | None = None,
    target_spend_usd: float | None = None,
    hard_cap_usd: float | None = None,
    max_live_minutes: int | None = None,
    session_max_live_minutes: int | None = None,
    disk_gb: int | None = None,
    startup_timeout_seconds: int | None = None,
    poll_interval_seconds: int | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle_manifest = build_unitree_groot_n17_sonic_vast_image_canary_bundle(
        job_dir=job / "bundle",
        generated_at=generated,
    )
    bundle_path = Path(str(bundle_manifest["bundle_path"]))
    staging = stage_wam_provider_bundle_object_store(
        job_dir=job / "object_store_staging",
        bundle_path=bundle_path,
        key_prefix=_string(os.getenv(OBJECT_STORE_KEY_PREFIX_ENV))
        or DEFAULT_OBJECT_STORE_KEY_PREFIX,
        expiration_seconds=_int_env(
            "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIGNED_URL_SECONDS",
            21600,
        ),
        generated_at=generated,
    )
    blockers: list[str] = []
    if staging.get("status") != "completed":
        blockers.extend(
            str(item)
            for item in (
                staging.get("blockers")
                or ["unitree_groot_n17_sonic_canary_object_store_staging_blocked"]
            )
        )
    if blockers:
        result = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "job_dir": str(job),
            "bundle_manifest_path": bundle_manifest.get("manifest_path"),
            "object_store_staging_manifest_path": str(
                job / "object_store_staging" / "wam_provider_object_store_staging_manifest.json"
            ),
            "blockers": sorted(set(blockers)),
            "raw_secret_values_recorded": False,
        }
        write_json(job / "vast_unitree_groot_sonic_image_canary_result.json", result)
        write_json(
            job / "vast_unitree_groot_sonic_image_canary_safe_summary.json",
            _safe_summary(generated_at=generated, job_dir=job, canary_result=result),
        )
        return result

    staging_dir = job / "object_store_staging"
    bundle_url = (staging_dir / "provider_bundle_url.txt").read_text(encoding="utf-8").strip()
    output_put_url = (
        staging_dir / "provider_output_put_url.txt"
    ).read_text(encoding="utf-8").strip()
    output_get_url = (
        staging_dir / "provider_output_get_url.txt"
    ).read_text(encoding="utf-8").strip()
    vast_run_dir = job / "vast_provider_run"
    output_zip = vast_run_dir / "vast_provider_runtime_output.zip"
    avoidlist_path = job / "vast_canary_machine_avoidlist.json"
    excluded_machine_ids = _excluded_machine_ids_from_env()
    if excluded_machine_ids:
        write_json(
            avoidlist_path,
            {
                "schema_version": "vast_machine_avoidlist.v1",
                "generated_at": generated,
                "status": "loaded_from_env",
                "machine_ids": sorted(excluded_machine_ids),
                "source_envs": list(EXCLUDED_MACHINE_ID_ENVS),
                "raw_secret_values_recorded": False,
            },
        )
    resolved_public_image = _string(public_image or os.getenv(PUBLIC_IMAGE_ENV)) or DEFAULT_PUBLIC_CUDA_IMAGE
    adapter_result = run_vast_provider_adapter(
        job_dir=vast_run_dir,
        mode="live-startup-probe",
        allow_vast_api_call=_truthy(os.getenv("BLUEPRINT_ALLOW_VAST_API_CALLS")),
        allow_instance_launch=_truthy(os.getenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH")),
        max_hourly_rate=max_hourly_rate
        if max_hourly_rate is not None
        else _float_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_MAX_HOURLY_RATE", 0.60),
        target_spend_usd=target_spend_usd
        if target_spend_usd is not None
        else _float_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_TARGET_SPEND_USD", 3.0),
        hard_cap_usd=hard_cap_usd
        if hard_cap_usd is not None
        else _float_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_HARD_CAP_USD", 3.0),
        max_live_minutes=max_live_minutes
        if max_live_minutes is not None
        else _int_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_MAX_LIVE_MINUTES", 55),
        session_max_live_minutes=session_max_live_minutes
        if session_max_live_minutes is not None
        else _int_env(
            "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_SESSION_MAX_LIVE_MINUTES",
            300,
        ),
        public_image=resolved_public_image,
        provider_bundle=bundle_path,
        provider_bundle_url=bundle_url,
        provider_output_put_url=output_put_url,
        provider_output_get_url=output_get_url,
        provider_runtime_output_zip=output_zip,
        enable_blueprint_bundle=True,
        provider_bundle_kind="unitree_groot_n17_sonic",
        vast_launch_mode=_string(launch_mode or os.getenv(VAST_LAUNCH_MODE_ENV))
        or "ssh_direct",
        ngc_image_login_mode=os.getenv(VAST_IMAGE_LOGIN_MODE_ENV),
        disk_gb=disk_gb
        if disk_gb is not None
        else _int_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_DISK_GB", 80),
        min_gpu_ram_mb=min_gpu_ram_mb,
        poll_interval_seconds=poll_interval_seconds
        if poll_interval_seconds is not None
        else _int_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_POLL_SECONDS", 15),
        startup_timeout_seconds=startup_timeout_seconds
        if startup_timeout_seconds is not None
        else _int_env(
            "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_STARTUP_TIMEOUT_SECONDS",
            1800,
        ),
        machine_avoidlist_path=avoidlist_path,
        verify_staging_urls=True,
    )
    startup = _read_json(vast_run_dir / "vast_startup_probe_manifest.json")
    gpu = _read_json(vast_run_dir / "vast_gpu_sanity_report.json")
    command = _read_json(vast_run_dir / "vast_provider_command_result.json")
    teardown = _read_json(vast_run_dir / "vast_teardown_manifest.json")
    offer = _read_json(vast_run_dir / "vast_offer_selection_manifest.json")
    provider_output = _provider_output_from_zip(output_zip, job / "provider_output")
    selected_offer = offer.get("selected_offer") if isinstance(offer.get("selected_offer"), Mapping) else {}
    result_blockers: list[str] = []
    if adapter_result.get("status") != "completed":
        result_blockers.extend(
            str(item) for item in adapter_result.get("blockers", []) or ["vast_canary_adapter_blocked"]
        )
    if startup.get("heartbeat_completed") is not True:
        result_blockers.append("vast_canary_heartbeat_not_completed")
    if gpu.get("status") != "completed":
        result_blockers.append("vast_canary_nvidia_smi_not_completed")
    if command.get("provider_output_upload_ok") is not True:
        result_blockers.append("vast_canary_provider_output_upload_missing")
    if command.get("provider_runtime_output_zip_produced") is not True:
        result_blockers.append("vast_canary_provider_output_zip_missing")
    if provider_output.get("status") != "completed":
        result_blockers.extend(
            str(item)
            for item in (
                provider_output.get("blockers")
                or ["vast_canary_provider_output_not_completed"]
            )
        )
    if provider_output.get("canary_marker") != CANARY_MARKER:
        result_blockers.append("vast_canary_marker_missing")
    checks = provider_output.get("checks") if isinstance(provider_output.get("checks"), Mapping) else {}
    nvidia_smi = checks.get("nvidia_smi") if isinstance(checks.get("nvidia_smi"), Mapping) else {}
    if nvidia_smi.get("returncode") != 0:
        result_blockers.append("vast_canary_provider_nvidia_smi_check_failed")
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if not result_blockers else "blocked",
        "job_dir": str(job),
        "public_image": resolved_public_image,
        "bundle_manifest_path": bundle_manifest.get("manifest_path"),
        "object_store_staging_manifest_path": str(
            job / "object_store_staging" / "wam_provider_object_store_staging_manifest.json"
        ),
        "vast_run_dir": str(vast_run_dir),
        "vast_provider_adapter_result_path": str(vast_run_dir / "vast_provider_adapter_result.json"),
        "vast_startup_probe_manifest_path": str(vast_run_dir / "vast_startup_probe_manifest.json"),
        "vast_gpu_sanity_report_path": str(vast_run_dir / "vast_gpu_sanity_report.json"),
        "vast_provider_command_result_path": str(vast_run_dir / "vast_provider_command_result.json"),
        "vast_teardown_manifest_path": str(vast_run_dir / "vast_teardown_manifest.json"),
        "provider_output_zip_path": str(output_zip),
        "min_gpu_ram_mb": min_gpu_ram_mb,
        "selected_offer": selected_offer,
        "vast_instance_ids": adapter_result.get("vast_instance_ids", []),
        "estimated_cost_usd": adapter_result.get("estimated_cost_usd"),
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "heartbeat_completed": startup.get("heartbeat_completed") is True,
        "gpu_sanity_completed": gpu.get("status") == "completed",
        "provider_bundle_downloaded_and_ran": bool(
            command.get("provider_command_path_remote_proven")
            and command.get("blueprint_provider_bundle_execution_proven")
        ),
        "provider_output_upload_ok": command.get("provider_output_upload_ok") is True,
        "provider_runtime_output_zip_produced": command.get(
            "provider_runtime_output_zip_produced"
        )
        is True,
        "canary_marker_observed": provider_output.get("canary_marker") == CANARY_MARKER,
        "provider_output": {
            "status": provider_output.get("status"),
            "canary_only": provider_output.get("canary_only"),
            "provider_output_path": provider_output.get("provider_output_path"),
            "checks": provider_output.get("checks"),
        },
        "blockers": sorted(set(result_blockers)),
        "claim_boundary": {
            "canary_is_not_policy_inference": True,
            "canary_is_not_model_execution": True,
            "custom_image_startup_proof_only": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "real_world_manipulation_success_proven": False,
        },
        "raw_secret_values_recorded": False,
    }
    write_json(job / "vast_unitree_groot_sonic_image_canary_result.json", result)
    write_json(
        job / "vast_unitree_groot_sonic_image_canary_safe_summary.json",
        _safe_summary(generated_at=generated, job_dir=job, canary_result=result),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--public-image")
    parser.add_argument("--min-gpu-ram-mb", type=int, default=DEFAULT_MIN_GPU_RAM_MB)
    parser.add_argument("--launch-mode")
    parser.add_argument("--max-hourly-rate", type=float)
    parser.add_argument("--target-spend-usd", type=float)
    parser.add_argument("--hard-cap-usd", type=float)
    parser.add_argument("--max-live-minutes", type=int)
    parser.add_argument("--session-max-live-minutes", type=int)
    parser.add_argument("--disk-gb", type=int)
    parser.add_argument("--startup-timeout-seconds", type=int)
    parser.add_argument("--poll-interval-seconds", type=int)
    args = parser.parse_args(argv)
    result = run_unitree_groot_n17_sonic_vast_image_canary(
        job_dir=args.job_dir,
        public_image=args.public_image,
        min_gpu_ram_mb=args.min_gpu_ram_mb,
        launch_mode=args.launch_mode,
        max_hourly_rate=args.max_hourly_rate,
        target_spend_usd=args.target_spend_usd,
        hard_cap_usd=args.hard_cap_usd,
        max_live_minutes=args.max_live_minutes,
        session_max_live_minutes=args.session_max_live_minutes,
        disk_gb=args.disk_gb,
        startup_timeout_seconds=args.startup_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
