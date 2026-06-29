"""Provider-backed OSCAR WAM command adapter.

This adapter implements the same command contract as the local OSCAR adapter:
the WAM evaluator sets ``BLUEPRINT_WAM_ROLLOUT_INPUT`` and
``BLUEPRINT_WAM_ROLLOUT_OUTPUT``. The adapter either imports a completed
provider output zip or, with explicit paid-provider gates, launches the selected
WAM compute provider runner and writes Blueprint-compatible rollout JSON.
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
from .oscar_wam_command_adapter import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_WIDTH,
)
from .oscar_wam_provider_bundle import build_oscar_wam_provider_bundle
from .oscar_wam_gpu_image import IMAGE_REF_ENV as OSCAR_WAM_GPU_IMAGE_REF_ENV
from .runpod_provider_adapter import RUNPOD_API_GATE_ENV
from .runpod_wam_async_runner import RUNPOD_POD_LAUNCH_GATE_ENV
from .vast_wam_authorized_runner import DEFAULT_WAM_PUBLIC_IMAGE
from .wam_generated_video_review import (
    validate_generated_mp4_for_review,
    visual_smoke_generated_rollouts_for_review,
)
from .wam_compute_providers import (
    PROVIDER_ORDER_ENV as WAM_COMPUTE_PROVIDER_ORDER_ENV,
    VAST_WAM_PAID_LAUNCH_GATE_ENV,
    WamComputeLaunchSpec,
    run_wam_compute_job,
)
from .wam_provider_object_store import stage_wam_provider_bundle_object_store


SCHEMA_VERSION = "oscar_wam_provider_command_adapter.v1"
ADAPTER_ID = "blueprint_oscar_wam_provider_command_adapter"
ALLOW_VAST_PROVIDER_LAUNCH_ENV = VAST_WAM_PAID_LAUNCH_GATE_ENV
OSCAR_WAM_COMPUTE_PROVIDER_ENV = "BLUEPRINT_OSCAR_WAM_COMPUTE_PROVIDER"
USE_OBJECT_STORE_ENV = "BLUEPRINT_OSCAR_WAM_PROVIDER_USE_OBJECT_STORE"
COMPLETED_PROVIDER_JOB_ENV = "BLUEPRINT_OSCAR_WAM_PROVIDER_COMPLETED_JOB_DIR"
PROVIDER_JOB_DIR_ENV = "BLUEPRINT_OSCAR_WAM_PROVIDER_JOB_DIR"
VAST_WAM_PUBLIC_IMAGE_ENV = "BLUEPRINT_VAST_WAM_PUBLIC_IMAGE"
RUNPOD_WAM_PUBLIC_IMAGE_ENV = "BLUEPRINT_RUNPOD_WAM_PUBLIC_IMAGE"
VAST_WAM_MIN_GPU_RAM_MB_ENV = "BLUEPRINT_VAST_WAM_MIN_GPU_RAM_MB"
VAST_WAM_EXCLUDED_MACHINE_ID_ENV = "BLUEPRINT_VAST_WAM_EXCLUDED_MACHINE_ID"
VAST_WAM_ALLOWED_MACHINE_ID_ENV = "BLUEPRINT_VAST_WAM_ALLOWED_MACHINE_ID"
VAST_WAM_POLL_MAX_WAIT_SECONDS_ENV = "BLUEPRINT_VAST_WAM_POLL_MAX_WAIT_SECONDS"
RUNPOD_WAM_CONTAINER_DISK_GB_ENV = "BLUEPRINT_RUNPOD_WAM_CONTAINER_DISK_GB"
RUNPOD_WAM_VOLUME_GB_ENV = "BLUEPRINT_RUNPOD_WAM_VOLUME_GB"
RUNPOD_WAM_MIN_VCPU_PER_GPU_ENV = "BLUEPRINT_RUNPOD_WAM_MIN_VCPU_PER_GPU"
RUNPOD_WAM_MIN_RAM_PER_GPU_ENV = "BLUEPRINT_RUNPOD_WAM_MIN_RAM_PER_GPU"
REDACTED_PROVIDER_TRANSPORT_URL = "REDACTED_COMPLETED_PROVIDER_TRANSPORT_URL"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _artifact_name(value: str | Path | None) -> str | None:
    text = _string(value)
    if not text:
        return None
    return Path(text).name or text


def _env_truthy(name: str) -> bool:
    return _string(os.getenv(name)).lower() in {"1", "true", "yes", "y"}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _provider_runtime_result_proves_model_output(
    runtime_result_payload: Mapping[str, Any],
) -> bool:
    return bool(
        runtime_result_payload
        and runtime_result_payload.get("status") == "completed"
        and runtime_result_payload.get("learned_wam_model_ran") is True
        and _load_json_value(
            runtime_result_payload,
            "truth_boundary",
            "generated_video_is_model_output",
        )
        is True
    )


def _write_output(output_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output.setdefault("schema_version", SCHEMA_VERSION)
    output.setdefault("adapter_id", ADAPTER_ID)
    output.setdefault("raw_credentials_written_to_artifacts", False)
    output.setdefault("secret_hashes_written_to_artifacts", False)
    write_json(output_path, output)
    return output


def _blocked_payload(
    *,
    blockers: Sequence[str],
    mode: str,
    output_path: Path,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "adapter_id": ADAPTER_ID,
        "mode": mode,
        "blockers": sorted(set(str(item) for item in blockers)),
        "details": dict(details or {}),
        "rollouts": [],
        "fresh_model_run_claimed": False,
        "fresh_provider_model_run_claimed": False,
        "fresh_model_command_executed_this_invocation": False,
        "fresh_provider_launch_attempted": False,
        "provider_output_replayed": False,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    return _write_output(output_path, payload)


def _find_provider_output_zip(provider_job_dir: Path) -> Path | None:
    candidates = [
        provider_job_dir / "vast_provider_runtime_output.zip",
        provider_job_dir / "runpod_provider_runtime_output.zip",
        provider_job_dir / "provider_runtime_output.zip",
    ]
    for path in candidates:
        if path.is_file():
            return path
    for path in sorted(provider_job_dir.glob("*provider*runtime*output*.zip")):
        if path.is_file():
            return path
    return None


def _provider_rollout_context_rows_from_input_manifest() -> list[dict[str, Any]]:
    manifest_path = Path(os.getenv("BLUEPRINT_WAM_ROLLOUT_INPUT", "")).expanduser()
    if not manifest_path.is_file():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, Mapping):
        return []
    task_prompts = [
        dict(row) for row in payload.get("task_prompts", []) or [] if isinstance(row, Mapping)
    ]
    videos = [
        dict(row) for row in payload.get("wam_input_videos", []) or [] if isinstance(row, Mapping)
    ]
    rows: list[dict[str, Any]] = []
    max_len = max(len(task_prompts), len(videos))
    for index in range(max_len):
        prompt = task_prompts[index] if index < len(task_prompts) else {}
        video = videos[index] if index < len(videos) else {}
        row = {
            "scenario_eval_run_id": prompt.get("scenario_eval_run_id")
            or video.get("scenario_eval_run_id"),
            "task_id": prompt.get("task_id") or video.get("task_id"),
            "spawn_id": prompt.get("spawn_id") or video.get("spawn_id"),
            "task_prompt": prompt.get("task_prompt") or video.get("task_prompt"),
            "source_wam_input_video_path": video.get("path"),
            "source_wam_input_camera": video.get("camera"),
        }
        rows.append({key: value for key, value in row.items() if _string(value)})
    return rows


def _backfill_provider_rollout_context(
    row: dict[str, Any],
    *,
    context_rows: Sequence[Mapping[str, Any]],
    index: int,
) -> dict[str, Any]:
    if not context_rows:
        return row
    context = context_rows[index] if index < len(context_rows) else context_rows[0]
    backfilled: list[str] = []
    for key in (
        "scenario_eval_run_id",
        "task_id",
        "spawn_id",
        "task_prompt",
        "source_wam_input_video_path",
        "source_wam_input_camera",
    ):
        if not _string(row.get(key)) and _string(context.get(key)):
            row[key] = context.get(key)
            backfilled.append(key)
    if backfilled:
        row["provider_rollout_context_backfilled_from_input_manifest"] = True
        row["provider_rollout_context_backfilled_fields"] = backfilled
    return row


def _extract_provider_payload(
    *,
    provider_output_zip: Path,
    output_path: Path,
    extraction_dir: Path,
    mode: str,
    source_provider_job_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_dir(extraction_dir)
    with zipfile.ZipFile(provider_output_zip) as archive:
        names = set(archive.namelist())
        if "wam_provider_output.json" not in names:
            return _blocked_payload(
                blockers=["provider_output_zip_missing_wam_provider_output_json"],
                mode=mode,
                output_path=output_path,
                details={
                    "provider_output_zip_name": provider_output_zip.name,
                    "provider_output_zip_path_omitted": True,
                },
            )
        provider_payload = json.loads(archive.read("wam_provider_output.json").decode("utf-8"))
        payload = dict(provider_payload) if isinstance(provider_payload, Mapping) else {}
        runtime_result_payload: dict[str, Any] = {}
        if "wam_runtime_result.json" in names:
            try:
                runtime_result = json.loads(
                    archive.read("wam_runtime_result.json").decode("utf-8")
                )
                if isinstance(runtime_result, Mapping):
                    runtime_result_payload = dict(runtime_result)
            except (json.JSONDecodeError, UnicodeDecodeError):
                runtime_result_payload = {}
        imported_truth_claims = {
            key: payload.get(key)
            for key in (
                "fresh_model_run_claimed",
                "fresh_provider_model_run_claimed",
                "fresh_model_command_executed_this_invocation",
                "fresh_provider_launch_attempted",
                "provider_output_replayed",
            )
            if key in payload
        }
        extracted_videos: dict[str, str] = {}
        for member in names:
            if not member.lower().endswith(".mp4"):
                continue
            safe_name = Path(member).name
            target = extraction_dir / safe_name
            with archive.open(member) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            extracted_videos[member] = str(target.resolve())
            extracted_videos[safe_name] = str(target.resolve())
        rewritten_rollouts: list[dict[str, Any]] = []
        video_validations: list[dict[str, Any]] = []
        invalid_video_count = 0
        missing_extracted_video_count = 0
        context_rows = _provider_rollout_context_rows_from_input_manifest()
        for rollout_index, item in enumerate(payload.get("rollouts", []) or []):
            if not isinstance(item, Mapping):
                continue
            row = _backfill_provider_rollout_context(
                dict(item),
                context_rows=context_rows,
                index=rollout_index,
            )
            original_video = _string(row.get("generated_video_path"))
            original_name = Path(original_video).name if original_video else ""
            extracted = extracted_videos.get(original_video) or extracted_videos.get(original_name)
            if extracted:
                row["provider_original_generated_video_name"] = original_name or None
                row["provider_original_generated_video_path_omitted"] = bool(original_video)
                row["generated_video_path"] = extracted
                validation = validate_generated_mp4_for_review(extracted)
                video_validations.append(
                    {
                        "rollout_id": row.get("rollout_id"),
                        "provider_original_generated_video_name": original_name or None,
                        "provider_original_generated_video_path_omitted": bool(original_video),
                        **validation,
                    }
                )
                if validation.get("status") == "completed":
                    row["generated_video_review_validation"] = validation
                    rewritten_rollouts.append(row)
                else:
                    invalid_video_count += 1
            else:
                missing_extracted_video_count += 1
        payload["rollouts"] = rewritten_rollouts
        payload["generated_video_review_validations"] = video_validations
        payload["schema_version"] = SCHEMA_VERSION
        payload["adapter_id"] = ADAPTER_ID
        payload["mode"] = mode
        payload["provider_output_zip_name"] = provider_output_zip.name
        payload["provider_output_zip_path_omitted"] = True
        payload["provider_output_zip_imported"] = True
        payload["provider_runtime_result_present"] = bool(runtime_result_payload)
        provider_runtime_proves_model_output = _provider_runtime_result_proves_model_output(
            runtime_result_payload
        )
        if runtime_result_payload:
            payload["provider_runtime_result_status"] = runtime_result_payload.get("status")
            payload["provider_learned_wam_model_ran"] = bool(
                runtime_result_payload.get("learned_wam_model_ran")
            )
            payload["provider_generated_video_is_model_output"] = bool(
                _load_json_value(
                    runtime_result_payload,
                    "truth_boundary",
                    "generated_video_is_model_output",
                )
            )
            payload[
                "provider_runtime_result_proves_model_output"
            ] = provider_runtime_proves_model_output
        payload["provider_video_extraction_dir_name"] = extraction_dir.name
        payload["provider_video_extraction_dir_path_omitted"] = True
        payload["raw_credentials_written_to_artifacts"] = False
        payload["secret_hashes_written_to_artifacts"] = False
        blockers = [str(item) for item in payload.get("blockers", []) or [] if str(item)]
        if payload.get("status") == "completed" and not rewritten_rollouts:
            blockers.append("provider_generated_video_not_reviewable")
        if invalid_video_count:
            blockers.append("provider_generated_video_decode_validation_failed")
        if missing_extracted_video_count:
            blockers.append("provider_generated_video_missing_from_output_zip")
        if payload.get("status") != "completed":
            blockers.append("provider_payload_not_completed")
        if blockers:
            payload["status"] = "blocked"
            payload["blockers"] = sorted(set(blockers))
        else:
            payload["blockers"] = []
        visual_smoke = visual_smoke_generated_rollouts_for_review(
            rollouts=rewritten_rollouts,
            output_dir=extraction_dir.parent / "provider_generated_rollout_visual_smoke",
            generated_at=utc_now_iso(),
            require_review_quality_profile=False,
        )
        payload["generated_rollout_visual_smoke"] = visual_smoke
        payload["generated_rollout_visual_smoke_status"] = visual_smoke.get("status")
        payload["generated_rollout_visually_useful_for_success_review"] = bool(
            _load_json_value(
                visual_smoke,
                "claim_boundary",
                "visual_rollout_useful_for_task_success_review",
            )
        )
        payload["generated_rollout_visual_quality_blockers"] = [
            str(item) for item in visual_smoke.get("blockers", []) or []
        ]
        payload["generated_rollout_review_usefulness_status"] = visual_smoke.get(
            "review_usefulness_status"
        )
        payload["generated_rollout_review_usefulness_blockers"] = [
            str(item) for item in visual_smoke.get("review_usefulness_blockers", []) or []
        ]
        if payload.get("status") == "completed" and visual_smoke.get(
            "status"
        ) == "failed_visual_quality_smoke":
            payload["status"] = "blocked"
            payload["blockers"] = sorted(
                set(
                    [
                        *[str(item) for item in payload.get("blockers", []) or [] if str(item)],
                        "provider_generated_rollout_visual_smoke_failed",
                        *[
                            str(item)
                            for item in visual_smoke.get("blockers", []) or []
                            if str(item)
                        ],
                    ]
                )
            )
        is_replay = mode == "replay_existing_provider_output"
        is_current_provider = mode in {"vast_provider", "runpod_provider", "wam_compute_provider"}
        if imported_truth_claims:
            payload["imported_provider_payload_truth_claims"] = imported_truth_claims
        payload["provider_output_replayed"] = bool(is_replay)
        payload["provider_output_imported_from_current_provider_run"] = bool(
            is_current_provider
        )
        payload["fresh_provider_launch_attempted"] = bool(is_current_provider)
        fresh_completed_model_output = bool(
            is_current_provider
            and payload.get("status") == "completed"
            and provider_runtime_proves_model_output
        )
        payload["fresh_model_run_claimed"] = fresh_completed_model_output
        payload["fresh_provider_model_run_claimed"] = fresh_completed_model_output
        payload["fresh_model_command_executed_this_invocation"] = fresh_completed_model_output
        if source_provider_job_dir is not None:
            key = (
                "replay_source_completed_provider_job_dir"
                if is_replay
                else "current_provider_job_dir"
            )
            payload[f"{key}_name"] = _artifact_name(source_provider_job_dir)
            payload[f"{key}_path_omitted"] = True
        return _write_output(output_path, payload)


def import_completed_provider_job(
    *,
    provider_job_dir: str | Path,
    output_path: str | Path,
    work_dir: str | Path,
    mode: str = "replay_existing_provider_output",
) -> dict[str, Any]:
    provider_dir = Path(provider_job_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    workspace = Path(work_dir).expanduser().resolve()
    provider_zip = _find_provider_output_zip(provider_dir)
    if provider_zip is None:
        return _blocked_payload(
            blockers=["completed_provider_job_missing_output_zip"],
            mode=mode,
            output_path=output,
            details={"provider_job_dir": str(provider_dir)},
        )
    return _extract_provider_payload(
        provider_output_zip=provider_zip,
        output_path=output,
        extraction_dir=workspace / "imported_provider_videos",
        mode=mode,
        source_provider_job_dir=provider_dir,
    )


def _provider_url_file_from_env(name: str) -> str | None:
    value = _string(os.getenv(name))
    return value or None


def _scrub_object_store_provider_url_files(
    object_store_manifest: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not object_store_manifest:
        return []
    statuses: list[dict[str, Any]] = []
    for key in (
        "provider_bundle_url_file",
        "provider_output_put_url_file",
        "provider_output_get_url_file",
    ):
        path_text = _string(_load_json_value(object_store_manifest, key, "path"))
        if not path_text:
            statuses.append({"key": key, "configured": False, "scrubbed": False})
            continue
        path = Path(path_text).expanduser()
        status: dict[str, Any] = {
            "key": key,
            "path": str(path),
            "configured": True,
            "present_before_scrub": path.is_file(),
            "raw_secret_values_recorded": False,
        }
        if path.is_file():
            try:
                path.write_text(REDACTED_PROVIDER_TRANSPORT_URL + "\n", encoding="utf-8")
                path.chmod(0o600)
                status.update({"scrubbed": True, "mode": oct(path.stat().st_mode & 0o777)})
            except OSError as exc:
                status.update(
                    {"scrubbed": False, "error_type": type(exc).__name__}
                )
        else:
            status["scrubbed"] = False
        statuses.append(status)
    return statuses


def _load_json_value(mapping: Mapping[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _vast_public_image_from_env() -> str:
    return (
        _string(os.getenv(VAST_WAM_PUBLIC_IMAGE_ENV))
        or _string(os.getenv(OSCAR_WAM_GPU_IMAGE_REF_ENV))
        or DEFAULT_WAM_PUBLIC_IMAGE
    )


def _public_image_for_provider(provider: str) -> str:
    if provider == "runpod":
        return (
            _string(os.getenv(RUNPOD_WAM_PUBLIC_IMAGE_ENV))
            or _string(os.getenv(OSCAR_WAM_GPU_IMAGE_REF_ENV))
            or _string(os.getenv(VAST_WAM_PUBLIC_IMAGE_ENV))
            or DEFAULT_WAM_PUBLIC_IMAGE
        )
    return _vast_public_image_from_env()


def _machine_ids_from_env(name: str) -> list[int]:
    values: list[int] = []
    for chunk in _string(os.getenv(name)).replace(",", " ").split():
        try:
            machine_id = int(chunk)
        except ValueError:
            continue
        if machine_id > 0 and machine_id not in values:
            values.append(machine_id)
    return values


def _env_int(name: str, default: int) -> int:
    try:
        return int(_string(os.getenv(name)) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_string(os.getenv(name)) or default)
    except ValueError:
        return default


def _poll_max_wait_seconds(timeout_seconds: float) -> int:
    return max(1, _env_int(VAST_WAM_POLL_MAX_WAIT_SECONDS_ENV, int(timeout_seconds)))


def _provider_order_from_cli(provider: str) -> list[str]:
    key = _string(provider).lower() or "vast"
    if key == "auto":
        configured = _string(os.getenv(WAM_COMPUTE_PROVIDER_ORDER_ENV))
        values = [
            item.strip().lower()
            for item in configured.replace(";", ",").split(",")
            if item.strip()
        ]
        return values or ["runpod", "vast"]
    return [key]


def run_compute_provider(
    *,
    rollout_input_path: Path,
    output_path: Path,
    work_dir: Path,
    provider: str,
    allow_paid_launch: bool,
    timeout_seconds: float,
    generated_at: str,
) -> dict[str, Any]:
    provider_order = _provider_order_from_cli(provider)
    primary_provider = provider_order[0]
    bundle_job_dir = work_dir / "bundle"
    provider_job_dir = work_dir / f"{primary_provider}_provider_run"
    bundle = build_oscar_wam_provider_bundle(
        job_dir=bundle_job_dir,
        wam_rollout_input_manifest=rollout_input_path,
        timeout_seconds=int(max(1, timeout_seconds)),
        num_steps=_env_int("BLUEPRINT_OSCAR_WAM_NUM_STEPS", 35),
        guidance=_env_float("BLUEPRINT_OSCAR_WAM_GUIDANCE", 6.0),
        seed=_env_int("BLUEPRINT_OSCAR_WAM_SEED", 42),
        num_frames=_env_int("BLUEPRINT_OSCAR_WAM_NUM_FRAMES", DEFAULT_NUM_FRAMES),
        height=_env_int("BLUEPRINT_OSCAR_WAM_HEIGHT", DEFAULT_HEIGHT),
        width=_env_int("BLUEPRINT_OSCAR_WAM_WIDTH", DEFAULT_WIDTH),
        fps=_env_float("BLUEPRINT_OSCAR_WAM_FPS", DEFAULT_FPS),
        generated_at=generated_at,
    )
    if bundle.get("status") != "completed":
        return _blocked_payload(
            blockers=bundle.get("blockers") or ["oscar_wam_provider_bundle_blocked"],
            mode=f"{primary_provider}_provider",
            output_path=output_path,
            details={"bundle_manifest": bundle},
        )
    bundle_path = Path(str(bundle.get("bundle_path"))).expanduser().resolve()
    provider_bundle_url_file = _provider_url_file_from_env("BLUEPRINT_WAM_PROVIDER_BUNDLE_URL_FILE")
    provider_output_put_url_file = _provider_url_file_from_env("BLUEPRINT_WAM_PROVIDER_OUTPUT_PUT_URL_FILE")
    provider_output_get_url_file = _provider_url_file_from_env("BLUEPRINT_WAM_PROVIDER_OUTPUT_GET_URL_FILE")
    object_store_manifest: dict[str, Any] | None = None
    if (
        not provider_bundle_url_file
        and not provider_output_put_url_file
        and _env_truthy(USE_OBJECT_STORE_ENV)
    ):
        object_store_manifest = stage_wam_provider_bundle_object_store(
            job_dir=provider_job_dir,
            bundle_path=bundle_path,
            generated_at=generated_at,
        )
        if object_store_manifest.get("status") != "completed":
            return _blocked_payload(
                blockers=object_store_manifest.get("blockers")
                or ["wam_provider_object_store_staging_blocked"],
                mode=f"{primary_provider}_provider",
                output_path=output_path,
                details={"object_store_staging_manifest": object_store_manifest},
            )
        provider_bundle_url_file = _string(
            object_store_manifest.get("provider_bundle_url_file", {}).get("path")
        )
        provider_output_put_url_file = _string(
            object_store_manifest.get("provider_output_put_url_file", {}).get("path")
        )
        provider_output_get_url_file = _string(
            object_store_manifest.get("provider_output_get_url_file", {}).get("path")
        )
    public_base_url = _string(os.getenv("BLUEPRINT_WAM_PROVIDER_PUBLIC_BASE_URL"))
    spec = WamComputeLaunchSpec(
        name="blueprint-oscar-wam-provider",
        bundle_path=bundle_path,
        provider_bundle_kind="wam",
        image=_public_image_for_provider(primary_provider),
        public_base_url=public_base_url,
        provider_bundle_url_file=provider_bundle_url_file,
        provider_output_put_url_file=provider_output_put_url_file,
        provider_output_get_url_file=provider_output_get_url_file,
        expected_video_count=1,
        max_wait_seconds=_poll_max_wait_seconds(timeout_seconds),
        retry_interval_seconds=_env_int("BLUEPRINT_VAST_WAM_POLL_INTERVAL_SECONDS", 15),
        max_hourly_rate_usd=float(os.getenv("BLUEPRINT_VAST_WAM_MAX_HOURLY_RATE", "0.35")),
        target_spend_usd=float(os.getenv("BLUEPRINT_VAST_WAM_TARGET_SPEND_USD", "3.0")),
        hard_cap_usd=float(os.getenv("BLUEPRINT_VAST_WAM_HARD_CAP_USD", "3.0")),
        max_live_minutes=int(os.getenv("BLUEPRINT_VAST_WAM_MAX_LIVE_MINUTES", "30")),
        session_max_live_minutes=int(os.getenv("BLUEPRINT_VAST_WAM_SESSION_MAX_LIVE_MINUTES", "35")),
        startup_poll_seconds=int(os.getenv("BLUEPRINT_VAST_WAM_STARTUP_POLL_SECONDS", "120")),
        min_gpu_ram_mb=int(os.getenv(VAST_WAM_MIN_GPU_RAM_MB_ENV, "0")),
        excluded_machine_ids=_machine_ids_from_env(VAST_WAM_EXCLUDED_MACHINE_ID_ENV),
        allowed_machine_ids=_machine_ids_from_env(VAST_WAM_ALLOWED_MACHINE_ID_ENV),
        container_disk_gb=_env_int(RUNPOD_WAM_CONTAINER_DISK_GB_ENV, 100),
        volume_gb=_env_int(RUNPOD_WAM_VOLUME_GB_ENV, 30),
        min_vcpu_per_gpu=_env_int(RUNPOD_WAM_MIN_VCPU_PER_GPU_ENV, 8),
        min_ram_per_gpu=_env_int(RUNPOD_WAM_MIN_RAM_PER_GPU_ENV, 40),
        skip_public_staging_verification=True,
    )

    compute_result = run_wam_compute_job(
        spec=spec,
        job_dir=work_dir,
        provider_order=provider_order,
        allow_paid_launch=allow_paid_launch,
        failover_on_blockers=(
            "no_vast_offer",
            "provider_runtime_output_zip_missing_or_empty",
            "runpod_provider_runtime_output_zip_not_received_locally",
            "provider_completed_without_valid_output_zip",
        ),
        teardown=True,
    )
    provider_url_file_scrub = _scrub_object_store_provider_url_files(object_store_manifest)
    provider_job_dir = work_dir / f"{compute_result.provider}_provider_run"
    if compute_result.status != "completed":
        return _blocked_payload(
            blockers=compute_result.blockers or [f"{compute_result.provider}_wam_provider_blocked"],
            mode=f"{compute_result.provider}_provider",
            output_path=output_path,
            details={
                "bundle_manifest": bundle,
                "object_store_staging_manifest": object_store_manifest,
                "wam_compute_result": compute_result.to_dict(),
                "provider_url_file_scrub": provider_url_file_scrub,
            },
        )
    provider_zip = _find_provider_output_zip(provider_job_dir)
    if provider_zip is None:
        return _blocked_payload(
            blockers=[f"{compute_result.provider}_provider_completed_without_output_zip"],
            mode=f"{compute_result.provider}_provider",
            output_path=output_path,
            details={"wam_compute_result": compute_result.to_dict()},
        )
    payload = _extract_provider_payload(
        provider_output_zip=provider_zip,
        output_path=output_path,
        extraction_dir=work_dir / f"{compute_result.provider}_provider_output_videos",
        mode=f"{compute_result.provider}_provider",
        source_provider_job_dir=provider_job_dir,
    )
    payload["details"] = {
        "bundle_manifest_path": str(bundle_job_dir / "oscar_wam_provider_bundle_manifest.json"),
        "wam_compute_provider": compute_result.provider,
        "wam_compute_result_path": str(work_dir / "wam_compute_run_result.json"),
        "provider_job_dir": str(provider_job_dir),
        "vast_provider_job_dir": str(provider_job_dir)
        if compute_result.provider == "vast"
        else None,
        "runpod_provider_job_dir": str(provider_job_dir)
        if compute_result.provider == "runpod"
        else None,
        "vast_create_manifest_path": str(provider_job_dir / "vast_wam_async_create_manifest.json")
        if compute_result.provider == "vast"
        else None,
        "vast_poll_manifest_path": str(provider_job_dir / "vast_wam_async_poll_manifest.json")
        if compute_result.provider == "vast"
        else None,
        "runpod_create_manifest_path": str(
            provider_job_dir / "runpod_wam_async_create_manifest.json"
        )
        if compute_result.provider == "runpod"
        else None,
        "runpod_poll_manifest_path": str(
            provider_job_dir / "runpod_wam_async_poll_manifest.json"
        )
        if compute_result.provider == "runpod"
        else None,
        "wam_compute_result": compute_result.to_dict(),
        "provider_url_file_scrub": provider_url_file_scrub,
    }
    write_json(output_path, payload)
    return payload


def run_vast_provider(
    *,
    rollout_input_path: Path,
    output_path: Path,
    work_dir: Path,
    allow_paid_vast_launch: bool,
    timeout_seconds: float,
    generated_at: str,
) -> dict[str, Any]:
    return run_compute_provider(
        rollout_input_path=rollout_input_path,
        output_path=output_path,
        work_dir=work_dir,
        provider="vast",
        allow_paid_launch=allow_paid_vast_launch,
        timeout_seconds=timeout_seconds,
        generated_at=generated_at,
    )


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "replay-existing-provider-output", "vast-provider"), default=os.getenv("BLUEPRINT_OSCAR_WAM_PROVIDER_MODE", "auto"))
    parser.add_argument(
        "--provider",
        choices=("auto", "vast", "runpod"),
        default=os.getenv(OSCAR_WAM_COMPUTE_PROVIDER_ENV, "vast"),
        help="Fresh provider-launch backend. --mode vast-provider forces Vast for compatibility.",
    )
    parser.add_argument("--completed-provider-job-dir")
    parser.add_argument("--work-dir")
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("BLUEPRINT_OSCAR_WAM_PROVIDER_TIMEOUT_SECONDS", "3600")))
    parser.add_argument("--allow-paid-provider-launch", action="store_true")
    parser.add_argument("--allow-paid-vast-launch", action="store_true")
    parser.add_argument("--allow-paid-runpod-launch", action="store_true")
    args = parser.parse_args(argv)
    output_path = Path(os.getenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", "wam_provider_output.json")).expanduser().resolve()
    rollout_input = Path(os.getenv("BLUEPRINT_WAM_ROLLOUT_INPUT", "")).expanduser()
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else Path(os.getenv(PROVIDER_JOB_DIR_ENV, output_path.parent / "oscar_wam_provider_command_workspace")).expanduser().resolve()
    )
    ensure_dir(work_dir)
    generated = utc_now_iso()
    blockers: list[str] = []
    effective_provider = "vast" if args.mode == "vast-provider" else args.provider
    provider_order = _provider_order_from_cli(effective_provider)
    paid_cli_authorized = bool(
        args.allow_paid_provider_launch
        or ("vast" in provider_order and args.allow_paid_vast_launch)
        or ("runpod" in provider_order and args.allow_paid_runpod_launch)
    )
    paid_gate_blockers: list[str] = []
    if not paid_cli_authorized:
        paid_gate_blockers.append("missing_cli_paid_wam_compute_provider_launch_flag")
    if "vast" in provider_order and not _env_truthy(ALLOW_VAST_PROVIDER_LAUNCH_ENV):
        paid_gate_blockers.append(f"missing_env_{ALLOW_VAST_PROVIDER_LAUNCH_ENV}")
    provider_remote_checkpoint_allowed = bool(
        not paid_gate_blockers and args.mode in {"auto", "vast-provider"}
    )
    completed_provider_job_dir = _string(
        args.completed_provider_job_dir or os.getenv(COMPLETED_PROVIDER_JOB_ENV)
    )
    replay_completed_provider_output_available = bool(
        args.mode in {"auto", "replay-existing-provider-output"}
        and completed_provider_job_dir
    )
    if not _string(os.getenv("BLUEPRINT_WAM_ROLLOUT_INPUT")):
        blockers.append("blocked_missing_BLUEPRINT_WAM_ROLLOUT_INPUT")
    elif not rollout_input.exists():
        blockers.append("blocked_wam_rollout_input_manifest_missing")
    checkpoint = _string(
        os.getenv("BLUEPRINT_WAM_MODEL_CHECKPOINT")
        or os.getenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT")
    )
    if (
        not checkpoint
        and not provider_remote_checkpoint_allowed
        and not replay_completed_provider_output_available
    ):
        blockers.append("blocked_missing_oscar_checkpoint_contract")
    elif checkpoint and not Path(checkpoint).expanduser().exists():
        blockers.append("blocked_configured_oscar_checkpoint_path_missing")
    if blockers:
        return _blocked_payload(
            blockers=blockers,
            mode=args.mode,
            output_path=output_path,
            details={"work_dir": str(work_dir)},
        )
    if args.mode in {"auto", "replay-existing-provider-output"} and completed_provider_job_dir:
        return import_completed_provider_job(
            provider_job_dir=completed_provider_job_dir,
            output_path=output_path,
            work_dir=work_dir,
            mode="replay_existing_provider_output",
        )
    if args.mode == "replay-existing-provider-output":
        return _blocked_payload(
            blockers=["missing_completed_provider_job_dir_for_replay"],
            mode="replay_existing_provider_output",
            output_path=output_path,
            details={"env": COMPLETED_PROVIDER_JOB_ENV},
        )
    if args.mode in {"auto", "vast-provider"}:
        if paid_gate_blockers:
            return _blocked_payload(
                blockers=paid_gate_blockers,
                mode=f"{provider_order[0]}_provider",
                output_path=output_path,
                details={
                    "provider": effective_provider,
                    "provider_order": provider_order,
                    "required_for_paid_launch": [
                        "one of --allow-paid-provider-launch, --allow-paid-vast-launch, --allow-paid-runpod-launch",
                        f"{ALLOW_VAST_PROVIDER_LAUNCH_ENV} when Vast is in provider order",
                        f"{RUNPOD_API_GATE_ENV} when RunPod is selected",
                        f"{RUNPOD_POD_LAUNCH_GATE_ENV} when RunPod is selected",
                        "VAST_API_KEY_FILE for Vast or RUNPOD_API_KEY/RUNPOD_API_KEY_FILE for RunPod",
                        "BLUEPRINT_WAM_PROVIDER_*_URL_FILE or object-store staging envs",
                    ]
                },
            )
        return run_compute_provider(
            rollout_input_path=rollout_input.resolve(),
            output_path=output_path,
            work_dir=work_dir,
            provider=effective_provider,
            allow_paid_launch=True,
            timeout_seconds=args.timeout_seconds,
            generated_at=generated,
        )
    return _blocked_payload(
        blockers=["unsupported_oscar_wam_provider_command_mode"],
        mode=args.mode,
        output_path=output_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run(argv)
    except Exception as exc:
        output_path = Path(os.getenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", "wam_provider_output.json")).expanduser().resolve()
        payload = _blocked_payload(
            blockers=[f"oscar_wam_provider_command_adapter_exception:{type(exc).__name__}"],
            mode=os.getenv("BLUEPRINT_OSCAR_WAM_PROVIDER_MODE", "auto"),
            output_path=output_path,
        )
    print(json.dumps({"adapter_id": ADAPTER_ID, "status": payload.get("status")}, sort_keys=True))
    return 0 if payload.get("status") == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
