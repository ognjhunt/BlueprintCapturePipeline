"""Run one GR00T N1.7 + UNITREE_G1_SONIC action through the Vast provider path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .unitree_groot_n17_sonic_policy_runtime import (
    GROOT_ROOT_ENV,
    N17_CHECKPOINT_ENV,
    POLICY_ID,
    POLICY_SERVER_URL_ENV,
    SIM2SIM_COMMAND_ENV,
    SONIC_CHECKPOINT_ENV,
    WBC_ROOT_ENV,
)
from .unitree_groot_n17_sonic_provider_smoke import (
    build_unitree_groot_n17_sonic_policy_provider_bundle,
    import_unitree_groot_n17_sonic_provider_output,
)
from .vast_provider_adapter import (
    DEFAULT_PUBLIC_CUDA_IMAGE,
    VAST_IMAGE_LOGIN_MODE_ENV,
    run_vast_provider_adapter,
)
from .wam_provider_object_store import stage_wam_provider_bundle_object_store


SCHEMA_VERSION = "unitree_groot_n17_sonic_vast_policy_command.v1"
JOB_ROOT_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_VAST_POLICY_JOB_ROOT"
PUBLIC_IMAGE_ENV = "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_PUBLIC_IMAGE"
VAST_LAUNCH_MODE_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_VAST_LAUNCH_MODE"
OBJECT_STORE_KEY_PREFIX_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_OBJECT_STORE_KEY_PREFIX"
INNER_POLICY_COMMAND_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_VAST_INNER_POLICY_COMMAND"
STANDARD_POLICY_COMMAND_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"
ALLOW_UNPINNED_FALLBACK_ENV = (
    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_ALLOW_UNPINNED_FALLBACK"
)
DEFAULT_INNER_POLICY_COMMAND = (
    "python -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
)
EXCLUDED_MACHINE_ID_ENVS = (
    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_EXCLUDED_MACHINE_ID",
    "BLUEPRINT_VAST_WAM_EXCLUDED_MACHINE_ID",
)
ALLOWED_MACHINE_ID_ENVS = (
    "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_ALLOWED_MACHINE_ID",
)
DEFAULT_OBJECT_STORE_KEY_PREFIX = "blueprint/unitree-groot-sonic"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(_string(os.getenv(name)) or default)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(_string(os.getenv(name)) or default)
    except ValueError:
        return default


def _excluded_machine_ids_from_env() -> list[int]:
    return _machine_ids_from_env(EXCLUDED_MACHINE_ID_ENVS)


def _allowed_machine_ids_from_env() -> list[int]:
    return _machine_ids_from_env(ALLOWED_MACHINE_ID_ENVS)


def _machine_ids_from_env(env_names: Sequence[str]) -> list[int]:
    values: list[int] = []
    for env_name in env_names:
        for chunk in _string(os.getenv(env_name)).replace(",", " ").split():
            try:
                machine_id = int(chunk)
            except ValueError:
                continue
            if machine_id > 0 and machine_id not in values:
                values.append(machine_id)
    return values


def _read_payload() -> dict[str, Any]:
    input_path = _string(os.getenv("BLUEPRINT_POLICY_ACTION_INPUT"))
    if input_path:
        value = json.loads(Path(input_path).expanduser().read_text(encoding="utf-8"))
    else:
        import sys

        raw = sys.stdin.read().strip()
        value = json.loads(raw) if raw else {}
    if not isinstance(value, Mapping):
        raise ValueError("policy input must be a JSON object")
    return dict(value)


def _write_payload(payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), sort_keys=True)
    output_path = _string(os.getenv("BLUEPRINT_POLICY_ACTION_OUTPUT"))
    if output_path:
        path = Path(output_path).expanduser()
        ensure_dir(path.parent)
        path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def _observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("observation"), Mapping):
        return dict(payload["observation"])  # type: ignore[index]
    return dict(payload)


def _camera_frame_path(observation: Mapping[str, Any]) -> Path | None:
    visual = _mapping(observation.get("visual_observation"))
    candidates = [
        visual.get("camera_frame_path"),
        _mapping(observation.get("sensor_surrogates")).get("camera_frame_path"),
        observation.get("camera_frame_path"),
    ]
    for candidate in candidates:
        if candidate:
            path = Path(str(candidate)).expanduser()
            if path.is_file():
                return path.resolve()
    return None


def _job_dir() -> Path:
    root_text = _string(os.getenv(JOB_ROOT_ENV))
    output_path = _string(os.getenv("BLUEPRINT_POLICY_ACTION_OUTPUT"))
    if root_text:
        root = Path(root_text).expanduser()
    elif output_path:
        root = Path(output_path).expanduser().parent / "unitree_groot_n17_sonic_vast_policy_command"
    else:
        root = Path.cwd() / "unitree_groot_n17_sonic_vast_policy_command"
    job = root / utc_now_iso().replace(":", "").replace("+", "_").replace("-", "")
    ensure_dir(job)
    return job.resolve()


def _blocked_payload(
    *,
    generated_at: str,
    job_dir: Path,
    blockers: Sequence[str],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "policy_id": POLICY_ID,
        "selected_candidate_id": POLICY_ID,
        "job_dir": str(job_dir),
        "action": None,
        "blockers": sorted({str(item) for item in blockers if str(item)}),
        "details": dict(details or {}),
        "unitree_groot_n17_sonic_model_executed": False,
        "unitree_groot_n17_sonic_policy_action_command_ran": False,
        "unitree_policy_action_command_ran": False,
        "policy_action_model_command_ran": False,
        "fresh_unitree_groot_n17_sonic_model_executed_this_invocation": False,
        "provider_output_replay_used": False,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "non_ranking_operational_claim_proven": False,
        "accepted_anchor_manipulation_success_proven": False,
    }


def run_vast_policy_command(payload: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    generated_at = utc_now_iso()
    job_dir = _job_dir()
    observation = _observation(payload)
    frame_path = _camera_frame_path(observation)
    input_path = job_dir / "policy_action_input.json"
    observation_path = job_dir / "policy_observation.json"
    write_json(input_path, dict(payload))
    write_json(observation_path, dict(payload))
    if frame_path is None:
        output = _blocked_payload(
            generated_at=generated_at,
            job_dir=job_dir,
            blockers=["blocked_missing_policy_visual_observation_frame"],
        )
        write_json(job_dir / "unitree_groot_n17_sonic_vast_policy_command_output.json", output)
        return output, 2

    inner_policy_command = _string(os.getenv(INNER_POLICY_COMMAND_ENV)) or DEFAULT_INNER_POLICY_COMMAND
    bundle = build_unitree_groot_n17_sonic_policy_provider_bundle(
        job_dir=job_dir / "provider_bundle",
        frame_path=frame_path,
        task_id=_string(payload.get("task_id") or observation.get("task_id")) or "unitree_g1_sonic",
        task_prompt=_string(payload.get("task_prompt"))
        or "Return one safe Unitree G1 / SONIC action for this observation.",
        policy_command=inner_policy_command,
        n17_checkpoint=os.getenv(N17_CHECKPOINT_ENV),
        sonic_checkpoint=os.getenv(SONIC_CHECKPOINT_ENV),
        groot_root=os.getenv(GROOT_ROOT_ENV),
        wbc_root=os.getenv(WBC_ROOT_ENV),
        policy_server_url=os.getenv(POLICY_SERVER_URL_ENV),
        sim2sim_command=os.getenv(SIM2SIM_COMMAND_ENV),
        policy_observation_path=observation_path,
    )
    if bundle.get("status") != "bundle_ready":
        output = _blocked_payload(
            generated_at=generated_at,
            job_dir=job_dir,
            blockers=bundle.get("blockers") or ["unitree_groot_n17_sonic_provider_bundle_blocked"],
            details={"bundle_manifest_path": bundle.get("manifest_path")},
        )
        write_json(job_dir / "unitree_groot_n17_sonic_vast_policy_command_output.json", output)
        return output, 2

    bundle_path = Path(str(bundle.get("bundle_path"))).expanduser().resolve()
    staging = stage_wam_provider_bundle_object_store(
        job_dir=job_dir / "object_store_staging",
        bundle_path=bundle_path,
        key_prefix=_string(os.getenv(OBJECT_STORE_KEY_PREFIX_ENV))
        or DEFAULT_OBJECT_STORE_KEY_PREFIX,
        expiration_seconds=_int_env("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIGNED_URL_SECONDS", 21600),
        generated_at=generated_at,
    )
    if staging.get("status") != "completed":
        output = _blocked_payload(
            generated_at=generated_at,
            job_dir=job_dir,
            blockers=staging.get("blockers") or ["unitree_groot_n17_sonic_object_store_staging_blocked"],
            details={"object_store_staging_manifest_path": str(job_dir / "object_store_staging" / "wam_provider_object_store_staging_manifest.json")},
        )
        write_json(job_dir / "unitree_groot_n17_sonic_vast_policy_command_output.json", output)
        return output, 2

    staging_dir = job_dir / "object_store_staging"
    bundle_url = (staging_dir / "provider_bundle_url.txt").read_text(encoding="utf-8").strip()
    output_put_url = (staging_dir / "provider_output_put_url.txt").read_text(encoding="utf-8").strip()
    output_get_url = (staging_dir / "provider_output_get_url.txt").read_text(encoding="utf-8").strip()
    machine_avoidlist_path = job_dir / "vast_machine_avoidlist.json"
    excluded_machine_ids = _excluded_machine_ids_from_env()
    allowed_machine_ids = _allowed_machine_ids_from_env()
    if excluded_machine_ids:
        write_json(
            machine_avoidlist_path,
            {
                "schema_version": "vast_machine_avoidlist.v1",
                "generated_at": generated_at,
                "status": "loaded_from_env",
                "machine_ids": sorted(excluded_machine_ids),
                "raw_secret_values_recorded": False,
            },
        )
    previous_policy_command = os.environ.get(STANDARD_POLICY_COMMAND_ENV)
    os.environ[STANDARD_POLICY_COMMAND_ENV] = inner_policy_command

    def run_remote_policy_attempt(
        *,
        run_dir: Path,
        attempt_allowed_machine_ids: Sequence[int],
    ) -> tuple[dict[str, Any], Path]:
        output_zip = run_dir / "vast_provider_runtime_output.zip"
        vast_result = run_vast_provider_adapter(
            job_dir=run_dir,
            mode="live-startup-probe",
            allow_vast_api_call=_truthy(os.getenv("BLUEPRINT_ALLOW_VAST_API_CALLS")),
            allow_instance_launch=_truthy(os.getenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH")),
            max_hourly_rate=_float_env(
                "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_MAX_HOURLY_RATE", 0.60
            ),
            target_spend_usd=_float_env(
                "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_TARGET_SPEND_USD", 3.0
            ),
            hard_cap_usd=_float_env(
                "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_HARD_CAP_USD", 3.0
            ),
            max_live_minutes=_int_env(
                "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_MAX_LIVE_MINUTES", 55
            ),
            session_max_live_minutes=_int_env(
                "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_SESSION_MAX_LIVE_MINUTES",
                300,
            ),
            public_image=_string(os.getenv(PUBLIC_IMAGE_ENV)) or DEFAULT_PUBLIC_CUDA_IMAGE,
            provider_bundle=bundle_path,
            provider_bundle_url=bundle_url,
            provider_output_put_url=output_put_url,
            provider_output_get_url=output_get_url,
            provider_runtime_output_zip=output_zip,
            enable_blueprint_bundle=True,
            provider_bundle_kind="unitree_groot_n17_sonic",
            vast_launch_mode=_string(os.getenv(VAST_LAUNCH_MODE_ENV)) or "ssh_direct",
            ngc_image_login_mode=os.getenv(VAST_IMAGE_LOGIN_MODE_ENV),
            disk_gb=_int_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_DISK_GB", 80),
            min_gpu_ram_mb=_int_env(
                "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_MIN_GPU_RAM_MB",
                48000,
            ),
            poll_interval_seconds=_int_env(
                "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_POLL_SECONDS", 15
            ),
            startup_timeout_seconds=_int_env(
                "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_STARTUP_TIMEOUT_SECONDS",
                1800,
            ),
            machine_avoidlist_path=machine_avoidlist_path,
            allowed_machine_ids=attempt_allowed_machine_ids,
            verify_staging_urls=True,
        )
        return vast_result, output_zip

    vast_run_dir = job_dir / "vast_provider_run"
    effective_vast_run_dir = vast_run_dir
    fallback_vast_result: dict[str, Any] | None = None
    fallback_output_zip: Path | None = None
    try:
        vast_result, output_zip = run_remote_policy_attempt(
            run_dir=vast_run_dir,
            attempt_allowed_machine_ids=allowed_machine_ids,
        )
        if (
            allowed_machine_ids
            and _truthy(os.getenv(ALLOW_UNPINNED_FALLBACK_ENV))
            and vast_result.get("status") != "completed"
            and "no_vast_offer_matching_allowed_machine_ids"
            in {str(item) for item in (vast_result.get("blockers") or [])}
        ):
            effective_vast_run_dir = job_dir / "vast_provider_run_unpinned_fallback"
            fallback_vast_result, fallback_output_zip = run_remote_policy_attempt(
                run_dir=effective_vast_run_dir,
                attempt_allowed_machine_ids=[],
            )
            vast_result = fallback_vast_result
            output_zip = fallback_output_zip
    finally:
        if previous_policy_command is None:
            os.environ.pop(STANDARD_POLICY_COMMAND_ENV, None)
        else:
            os.environ[STANDARD_POLICY_COMMAND_ENV] = previous_policy_command
    if vast_result.get("status") != "completed" or not output_zip.is_file():
        output = _blocked_payload(
            generated_at=generated_at,
            job_dir=job_dir,
            blockers=vast_result.get("blockers") or ["unitree_groot_n17_sonic_vast_provider_blocked"],
            details={
                "vast_provider_adapter_result_path": str(
                    effective_vast_run_dir / "vast_provider_adapter_result.json"
                ),
                "vast_teardown_manifest_path": str(
                    effective_vast_run_dir / "vast_teardown_manifest.json"
                ),
                "fallback_vast_provider_adapter_result_path": str(
                    job_dir
                    / "vast_provider_run_unpinned_fallback"
                    / "vast_provider_adapter_result.json"
                )
                if fallback_vast_result is not None
                else None,
            },
        )
        write_json(job_dir / "unitree_groot_n17_sonic_vast_policy_command_output.json", output)
        return output, 2

    imported = import_unitree_groot_n17_sonic_provider_output(
        provider_output_zip=output_zip,
        extraction_dir=job_dir / "imported_provider_output",
        output_path=job_dir / "unitree_groot_n17_sonic_policy_provider_import.json",
    )
    action = imported.get("action") if isinstance(imported.get("action"), Mapping) else None
    completed = imported.get("status") == "completed" and bool(action)
    output = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if completed else "blocked",
        "policy_id": POLICY_ID,
        "selected_candidate_id": POLICY_ID,
        "job_dir": str(job_dir),
        "action": action,
        "blockers": [] if completed else imported.get("blockers") or ["unitree_groot_n17_sonic_provider_import_blocked"],
        "unitree_groot_n17_sonic_model_executed": bool(
            imported.get("unitree_groot_n17_sonic_model_executed")
        ),
        "unitree_groot_n17_sonic_policy_action_command_ran": bool(
            imported.get("unitree_groot_n17_sonic_policy_action_command_ran")
        ),
        "unitree_policy_action_command_ran": bool(
            imported.get("unitree_groot_n17_sonic_policy_action_command_ran")
        ),
        "policy_action_model_command_ran": bool(
            imported.get("unitree_groot_n17_sonic_policy_action_command_ran")
        ),
        "fresh_unitree_groot_n17_sonic_model_executed_this_invocation": bool(
            imported.get("unitree_groot_n17_sonic_model_executed")
            and imported.get("unitree_groot_n17_sonic_policy_action_command_ran")
        ),
        "provider_output_replay_used": False,
        "vast_provider_adapter_result_path": str(
            effective_vast_run_dir / "vast_provider_adapter_result.json"
        ),
        "provider_import_path": str(job_dir / "unitree_groot_n17_sonic_policy_provider_import.json"),
        "estimated_cost_usd": vast_result.get("estimated_cost_usd"),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "non_ranking_operational_claim_proven": False,
        "accepted_anchor_manipulation_success_proven": False,
    }
    write_json(job_dir / "unitree_groot_n17_sonic_vast_policy_command_output.json", output)
    return output, 0 if completed else 2


def main() -> int:
    payload = _read_payload()
    output, exit_code = run_vast_policy_command(payload)
    _write_payload(output)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
