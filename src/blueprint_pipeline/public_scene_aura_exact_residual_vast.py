"""Run one rights-admitted, exact-mask Aura residual packet on Vast.

This adapter is intentionally narrower than the historical Aura InteriorGS
lane: it accepts a sealed 1--5 replacement packet, uploads only its
private-derived ZIP, arms an independent watchdog before create, and makes a
raw Aura result available only after provider-zero and object-store absence
are independently retained.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import PaidResourceAdmissionGrant
from .public_scene_aura_exact_residual_bundle import DEFAULT_IMAGE, SCHEMA_VERSION as BUNDLE_SCHEMA
from .vast_independent_watchdog_control import (
    EVIDENCE_NAME as WATCHDOG_EVIDENCE_NAME,
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
)
from .vast_provider_adapter import run_vast_provider_adapter
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


PROBE_KIND = "adp-aurafusion360-exact-residual"
PROVIDER_BUNDLE_KIND = "adp_aura_exact_residual"
RESULT_SCHEMA_VERSION = "public_scene_aura_exact_residual_vast_run.v1"
RAW_RESULT_SCHEMA_VERSION = "public_scene_aura_exact_residual_raw_result.v1"
RUNTIME_ABSTENTION_SCHEMA_VERSION = "public_scene_aura_exact_residual_runtime_abstention.v1"
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/aura-exact-residual"
MAX_TTL_SECONDS = 14_400
MIN_TTL_SECONDS = 7_200
MAX_HARD_CAP_USD = 12.0
MIN_RASTERIZER_COMPUTE_CAP = 890
GPU_SELECTION_POLICY = {
    "policy_id": "aura_exact_residual_l40s_observed_control",
    "allowed_gpu_keywords": ("L40S",),
    "denied_gpu_keywords": (),
    "reason": "same released Aura rasterizer class as the qualified author control",
}
_MUTATION_ENV = ("BLUEPRINT_ALLOW_VAST_API_CALLS", "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH")
_RETRY_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("aura_exact_residual_receipt_unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError("aura_exact_residual_receipt_unreadable")
    return value


def _bound(record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(code)
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ValueError(code)
    return path


def _zip_member_bytes(
    archive: zipfile.ZipFile, *, record: Any, root: str, code: str
) -> bytes:
    """Read one digest-bound ZIP member without trusting a caller path."""

    if not isinstance(record, Mapping):
        raise ValueError(code)
    relative = str(record.get("relative_path") or "")
    member = f"{root}/{relative}" if relative else ""
    if (
        not relative
        or relative.startswith("/")
        or ".." in Path(relative).parts
        or member not in archive.namelist()
    ):
        raise ValueError(code)
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise ValueError(code) from exc
    if (
        len(payload) != record.get("size_bytes")
        or "sha256:" + hashlib.sha256(payload).hexdigest() != record.get("sha256")
    ):
        raise ValueError(code)
    return payload


def _zip_member_json(
    archive: zipfile.ZipFile, *, record: Any, root: str, code: str
) -> dict[str, Any]:
    try:
        value = json.loads(
            _zip_member_bytes(archive, record=record, root=root, code=code).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _authority(
    preflight: Mapping[str, Any], *, backend_value: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    if (
        backend_value.get("schema_version") != "public_scene_released_code_inpainting_admission.v1"
        or backend_value.get("status") != "rights_admitted_for_private_derived_inpainting"
        or backend_value.get("backend_id") != "aurafusion360_exact_residual_multiview"
        or backend_value.get("strict_exact_residual_masks_required") is not True
        or backend_value.get("outside_mask_pixel_delta_required") != 0
        or backend_value.get("private_derived_upload_policy", {}).get("private_derived_upload")
        is not True
        or backend_value.get("private_derived_upload_policy", {}).get("raw_dataset_bytes_upload")
        is not False
        or backend_value.get("private_derived_upload_policy", {}).get("provider_training")
        is not False
        or backend_value.get("receipt_digest")
        != canonical_digest(backend_value, digest_field="receipt_digest")
    ):
        raise ValueError("aura_exact_residual_backend_admission_invalid")
    authority_record = backend_value.get("execution_authority")
    authority_path = _bound(authority_record, code="aura_exact_residual_execution_authority_unbound")
    authority = _read(authority_path)
    paid = authority.get("paid_compute")
    if (
        authority.get("schema_version") != "third_scene_dual_task_execution_authority.v1"
        or authority.get("authority_kind") != "explicit_user_direction_in_current_goal"
        or authority.get("publisher_scene_id") != "840920"
        or authority.get("private_rights_admitted_scene_derived_uploads_authorized") is not True
        or authority.get("raw_interiorgs_upload_authorized") is not False
        or authority.get("training_authorized") is not False
        or authority.get("retention") != "bounded_to_goal_then_provider_zero"
        or not isinstance(paid, Mapping)
        or paid.get("provider") != "vast"
        or paid.get("hard_total_spend_cap_usd") != MAX_HARD_CAP_USD
        or paid.get("zero_retry") is not True
        or paid.get("provider_zero_required_for_lane") is not True
        or paid.get("external_instance_allowlist") != [47373597]
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
        or authority_record.get("authority_digest") != authority.get("authority_digest")
    ):
        raise ValueError("aura_exact_residual_execution_authority_invalid")
    return authority, authority_path


def validate_aura_exact_residual_bundle(receipt_path: str | Path) -> dict[str, Any]:
    """Load file-backed receipts; never accept digest-shaped caller assertions."""

    receipt_file = Path(receipt_path).expanduser().resolve()
    receipt = _read(receipt_file)
    bundle_path = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    if (
        receipt.get("schema_version") != BUNDLE_SCHEMA
        or receipt.get("status") != "ready"
        or not bundle_path.is_file()
        or bundle_path.is_symlink()
        or _sha256(bundle_path) != receipt.get("bundle_sha256")
        or receipt.get("container_image") != DEFAULT_IMAGE
        or not 1 <= receipt.get("replacement_object_count", 0) <= 5
        or receipt.get("shared_camera_count", 0) < receipt.get("replacement_object_count", 0)
        or receipt.get("task_count", 0) < 1
        or receipt.get("private_derived_upload_only") is not True
        or receipt.get("raw_interiorgs_bytes_included") is not False
        or receipt.get("stock_inpaint360gs_code_or_author_data_included") is not False
        or receipt.get("automatic_paid_retry_allowed") is not False
        or receipt.get("provider_zero_required_after_return") is not True
    ):
        raise ValueError("aura_exact_residual_bundle_receipt_invalid")
    rehearsal = receipt.get("exact_bundle_entrypoint_rehearsal")
    if (
        not isinstance(rehearsal, Mapping)
        or rehearsal.get("status") != "passed"
        or rehearsal.get("provider_mutations_performed") != 0
        or rehearsal.get("gpu_runtime_started") is not False
    ):
        raise ValueError("aura_exact_residual_bundle_rehearsal_invalid")
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("aura_exact_residual_bundle_integrity_invalid")
            runtime_request_path = _bound(
                receipt.get("runtime_request"), code="aura_exact_residual_runtime_request_unbound"
            )
            request_bytes = archive.read("provider_runtime/aura_exact_residual_runtime_request.json")
            if request_bytes != runtime_request_path.read_bytes():
                raise ValueError("aura_exact_residual_runtime_request_bundle_mismatch")
            request = json.loads(request_bytes.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("aura_exact_residual_runtime_request_unbound")
            preflight = _zip_member_json(
                archive,
                record=request.get("preflight"),
                root="provider_runtime",
                code="aura_exact_residual_preflight_unbound",
            )
            backend = _zip_member_json(
                archive,
                record=request.get("backend_admission"),
                root="provider_runtime",
                code="aura_exact_residual_backend_receipt_unbound",
            )
            _zip_member_bytes(
                archive,
                record=request.get("shared_retained_scene"),
                root="provider_runtime",
                code="aura_exact_residual_shared_ply_unbound",
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("aura_exact_residual_bundle_integrity_invalid") from exc
    if (
        request.get("schema_version") != "public_scene_aura_exact_residual_runtime_request.v1"
        or request.get("request_digest") != canonical_digest(request, digest_field="request_digest")
        or request.get("private_derived_upload_only") is not True
        or request.get("raw_dataset_bytes_included") is not False
        or request.get("provider_training_authorized") is not False
        or request.get("automatic_paid_retry_allowed") is not False
        or request.get("provider_zero_required_after_return") is not True
        or request.get("learned_policy_outcomes_accessed") is not False
    ):
        raise ValueError("aura_exact_residual_runtime_request_invalid")
    if (
        preflight.get("schema_version") != "public_scene_aura_exact_residual_preflight.v1"
        or preflight.get("status") != "prepared_no_upload_no_execution"
        or preflight.get("preflight_digest") != receipt.get("preflight_digest")
        or preflight.get("preflight_digest") != canonical_digest(preflight, digest_field="preflight_digest")
        or preflight.get("replacement_object_count") != receipt.get("replacement_object_count")
        or preflight.get("execution", {}).get("provider_mutations_performed") != 0
        or preflight.get("execution", {}).get("aura_inpainting_executed") is not False
        or preflight.get("backend_admission", {}).get("sha256")
        != request.get("backend_admission", {}).get("sha256")
        or preflight.get("backend_admission", {}).get("size_bytes")
        != request.get("backend_admission", {}).get("size_bytes")
        or preflight.get("shared_retained_scene", {}).get("sha256")
        != request.get("shared_retained_scene", {}).get("sha256")
        or preflight.get("shared_retained_scene", {}).get("retained_gaussian_count")
        != request.get("shared_retained_scene", {}).get("retained_gaussian_count")
        or len(request.get("camera_inputs") or []) != receipt.get("shared_camera_count")
        or len(request.get("task_plans") or []) != receipt.get("task_count")
    ):
        raise ValueError("aura_exact_residual_preflight_invalid")
    authority, authority_path = _authority(preflight, backend_value=backend)
    return {
        "receipt_path": str(receipt_file),
        "receipt_sha256": _sha256(receipt_file),
        "bundle_path": str(bundle_path),
        "bundle_sha256": receipt["bundle_sha256"],
        "container_image": DEFAULT_IMAGE,
        "preflight_path": None,
        "preflight_digest": preflight["preflight_digest"],
        "replacement_object_count": receipt["replacement_object_count"],
        "shared_camera_count": receipt["shared_camera_count"],
        "task_count": receipt["task_count"],
        "execution_authority_path": str(authority_path),
        "execution_authority_digest": authority["authority_digest"],
        "allowed_active_instance_ids": list(authority["paid_compute"]["external_instance_allowlist"]),
    }


@contextmanager
def _authority_environment():
    names = (*_MUTATION_ENV, _RETRY_ENV)
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in _MUTATION_ENV:
            os.environ[name] = "1"
        os.environ[_RETRY_ENV] = "0"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _extract(path: Path, destination: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, ["aura_exact_residual_provider_output_zip_missing"]
    ensure_dir(destination)
    root = destination.resolve()
    blockers: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (root / member.filename).resolve()
                if root not in target.parents and target != root:
                    blockers.append("aura_exact_residual_provider_output_zip_path_traversal")
            if not blockers:
                archive.extractall(root)
    except (OSError, zipfile.BadZipFile):
        blockers.append("aura_exact_residual_provider_output_zip_invalid")
    result_path = root / "public_scene_aura_exact_residual_runtime_result.json"
    if not result_path.is_file() or result_path.is_symlink():
        blockers.append("aura_exact_residual_runtime_result_missing")
        return {}, blockers
    try:
        return _read(result_path), blockers
    except ValueError:
        return {}, [*blockers, "aura_exact_residual_runtime_result_unreadable"]


def _absolute_runtime_rows(execution: Mapping[str, Any], root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def local(record: Any) -> Path:
        if not isinstance(record, Mapping):
            raise ValueError("aura_exact_residual_runtime_output_record_invalid")
        relative = str(record.get("relative_path") or "")
        path = (root / relative).resolve()
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or root not in path.parents
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise ValueError("aura_exact_residual_runtime_output_record_invalid")
        return path

    frames: list[dict[str, Any]] = []
    for row in execution.get("frames") or []:
        if not isinstance(row, Mapping):
            raise ValueError("aura_exact_residual_runtime_frame_invalid")
        path = local(row.get("native_aura_frame"))
        frames.append({
            "task_id": row.get("task_id"), "camera_id": row.get("camera_id"),
            "native_aura_frame": _record(path),
            "native_aura_point_cloud_sha256": row.get("native_aura_point_cloud_sha256"),
        })
    outputs: list[dict[str, Any]] = []
    for row in execution.get("task_outputs") or []:
        if not isinstance(row, Mapping):
            raise ValueError("aura_exact_residual_runtime_task_output_invalid")
        path = local(row.get("native_aura_point_cloud"))
        outputs.append({
            "task_id": row.get("task_id"), "native_aura_point_cloud": _record(path),
            "native_aura_point_cloud_sha256": row.get("native_aura_point_cloud_sha256"),
            "native_aura_gaussian_count": row.get("native_aura_gaussian_count"),
            "native_aura_representation": row.get("native_aura_representation"),
            "render_camera_ids": row.get("render_camera_ids"),
        })
    if not frames or not outputs:
        raise ValueError("aura_exact_residual_runtime_outputs_missing")
    return frames, outputs


def run_aura_exact_residual_vast(
    *, job_dir: str | Path, paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool, prepared_bundle: Mapping[str, Any], max_hourly_rate_usd: float = 1.5,
    hard_cap_usd: float = 6.0, hard_ttl_seconds: int = MAX_TTL_SECONDS,
    machine_avoidlist_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute once. The only paid path is through the canonical allocator."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = dict(prepared_bundle)
    if (
        not 0 < max_hourly_rate_usd <= hard_cap_usd <= MAX_HARD_CAP_USD
        or not MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS
        or hard_ttl_seconds * max_hourly_rate_usd / 3600 > hard_cap_usd
    ):
        raise ValueError("aura_exact_residual_budget_invalid")
    if not execute:
        result = {"schema_version": RESULT_SCHEMA_VERSION, "generated_at": utc_now_iso(),
                  "status": "dry_run_ready", "prepared_bundle": bundle,
                  "provider_mutations_performed": 0, "retry_cap": 0, "blockers": []}
        write_json(job / "public_scene_aura_exact_residual_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("aura_exact_residual_paid_resource_admission_grant_missing")
    bundle_path = Path(str(bundle["bundle_path"])).resolve()
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir, bundle_path=bundle_path, key_prefix=DEFAULT_KEY_PREFIX,
        expiration_seconds=hard_ttl_seconds + 1800,
    )
    if staging.get("status") != "completed":
        result = {"schema_version": RESULT_SCHEMA_VERSION, "generated_at": utc_now_iso(), "status": "blocked",
                  "provider_mutations_performed": 0, "retry_cap": 0,
                  "blockers": staging.get("blockers") or ["aura_exact_residual_object_store_staging_blocked"]}
        write_json(job / "public_scene_aura_exact_residual_vast_result.json", result)
        return result
    allowed = tuple(int(value) for value in bundle["allowed_active_instance_ids"])
    handoff, handle = arm_independent_vast_watchdog(
        job_dir=job, max_live_minutes=hard_ttl_seconds // 60, generated_at=utc_now_iso(),
        allowed_active_instance_ids=allowed, pod_name_prefix="blueprint-adp-aura-exact-residual-",
    )
    if handle is None:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        result = {"schema_version": RESULT_SCHEMA_VERSION, "generated_at": utc_now_iso(), "status": "blocked",
                  "provider_mutations_performed": 0, "retry_cap": 0,
                  "all_staged_objects_absent": cleanup.get("all_objects_absent"),
                  "independent_watchdog": handoff,
                  "blockers": ["aura_exact_residual_independent_watchdog_not_armed"]}
        write_json(job / "public_scene_aura_exact_residual_vast_result.json", result)
        return result
    provider_run = job / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    adapter: dict[str, Any] = {}
    try:
        with _authority_environment():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run, mode="live-startup-probe", allow_vast_api_call=True,
                allow_instance_launch=True, max_hourly_rate=max_hourly_rate_usd,
                target_spend_usd=hard_cap_usd, hard_cap_usd=hard_cap_usd,
                max_live_minutes=hard_ttl_seconds // 60, session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=bundle["container_image"], isaac_image=bundle["container_image"],
                ngc_image_login_mode="never", provider_bundle=bundle_path,
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
                provider_output_put_url=(staging_dir / "provider_output_put_url.txt").read_text().strip(),
                provider_output_get_url=(staging_dir / "provider_output_get_url.txt").read_text().strip(),
                provider_runtime_output_zip=output_zip, enable_isaac_smoke=False,
                enable_blueprint_bundle=True, provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                vast_launch_mode="ssh_direct", allow_cold_isaac_image_pull=False, disk_gb=192,
                min_gpu_ram_mb=24_000, min_compute_cap=MIN_RASTERIZER_COMPUTE_CAP,
                poll_interval_seconds=15, startup_timeout_seconds=hard_ttl_seconds,
                heartbeat_no_progress_seconds=1800,
                session_budget_ledger_path=job / "aura_exact_residual_vast_session_budget.json",
                verify_staging_urls=True, require_known_supported_isaac_driver=False,
                preferred_gpu_keywords=("L40S",), prefer_isaac_rt=False,
                gpu_selection_policy=GPU_SELECTION_POLICY, machine_avoidlist_path=machine_avoidlist_path,
                allowed_active_instance_ids=allowed,
                vast_launch_lock_file=job.parent / "aura_exact_residual_paid_launch.lock",
                instance_label_prefix="blueprint-adp-aura-exact-residual-",
                started_instance_id_path=handle.started_instance_id_path, forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {"status": "blocked", "blockers": [f"aura_exact_residual_adapter_failed:{type(exc).__name__}"],
                   "raw_secret_values_recorded": False}
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    teardown_path = provider_run / "vast_teardown_manifest.json"
    teardown = _read(teardown_path) if teardown_path.is_file() else {}
    instance_ids = [value for value in teardown.get("vast_instance_ids") or [] if isinstance(value, int) and value > 0]
    watchdog = close_independent_vast_watchdog(
        job_dir=job, handle=handle, instance_ids=instance_ids,
        provider_teardown_completed=teardown.get("continuing_spend_from_this_run") is False,
        provider_allocation_impossible=not instance_ids and adapter.get("provider_create_attempted") is not True,
    )
    execution_root = job / "immutable_execution"
    execution, blockers = _extract(output_zip, execution_root)
    adapter_path = provider_run / "vast_provider_adapter_result.json"
    final_path = provider_run / "vast_final_validation.json"
    # Bind the watchdog's terminal provider-inventory observation, not the
    # owner-to-watchdog cancellation request.  The compositor requires its
    # independent exact-id and global-zero evidence.
    watchdog_path = job / "independent_vast_watchdog" / WATCHDOG_EVIDENCE_NAME
    closeout_adapter = {
        "schema_version": "public_scene_aura_exact_residual_adapter_closeout.v1",
        "source_adapter_result": _record(adapter_path) if adapter_path.is_file() else None,
        "api_call_performed": adapter.get("api_call_performed"),
        "provider_create_attempted": adapter.get("provider_create_attempted"),
        "final_validation_status": adapter.get("final_validation_status"),
        "continuing_spend_from_this_run": adapter.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_ttl_seconds": hard_ttl_seconds,
    }
    closeout_adapter_path = job / "aura_exact_residual_adapter_closeout.json"
    write_json(closeout_adapter_path, closeout_adapter)
    if adapter.get("status") != "completed":
        blockers.append("aura_exact_residual_provider_adapter_not_completed")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("aura_exact_residual_object_store_zero_not_proven")
    if watchdog.get("status") != "provider_terminal":
        blockers.append("aura_exact_residual_watchdog_not_terminal")
    if (
        execution.get("schema_version") != "public_scene_aura_exact_residual_runtime_result.v1"
        or execution.get("status") != "completed"
    ):
        blockers.append("aura_exact_residual_runtime_not_completed")
    if (
        execution.get("aura_inpainting_executed") is not True
        or execution.get("provider_mutations_performed") != 0
    ):
        blockers.append("aura_exact_residual_runtime_claim_invalid")
    raw_path: Path | None = None
    if not blockers:
        try:
            frames, task_outputs = _absolute_runtime_rows(execution, execution_root)
            raw: dict[str, Any] = {
                "schema_version": RAW_RESULT_SCHEMA_VERSION, "status": "aura_native_residual_frames_rendered",
                "preflight_digest": bundle["preflight_digest"], "aura_inpainting_executed": True,
                "provider_mutations_performed": 1, "learned_policy_outcomes_accessed": False,
                "provider_closeout": {"adapter_result": _record(closeout_adapter_path),
                    "teardown_manifest": _record(teardown_path), "final_validation": _record(final_path),
                    "watchdog_receipt": _record(watchdog_path)},
                "task_outputs": task_outputs, "frames": frames, "result_digest": "",
            }
            raw["result_digest"] = canonical_digest(raw, digest_field="result_digest")
            raw_path = job / "public_scene_aura_exact_residual_raw_result.json"
            write_json(raw_path, raw)
        except (OSError, ValueError, KeyError) as exc:
            blockers.append(f"aura_exact_residual_raw_result_materialization_failed:{type(exc).__name__}")
    result = {"schema_version": RESULT_SCHEMA_VERSION, "generated_at": utc_now_iso(),
              "status": "completed" if not blockers else "blocked", "bundle_sha256": bundle["bundle_sha256"],
              "preflight_digest": bundle["preflight_digest"], "execution_result_path": str(execution_root / "public_scene_aura_exact_residual_runtime_result.json"),
              "raw_result_path": str(raw_path) if raw_path else None,
              "adapter_result_path": str(adapter_path), "teardown_manifest_path": str(teardown_path),
              "final_validation_path": str(final_path), "watchdog_receipt_path": str(watchdog_path),
              "estimated_cost_usd": adapter.get("estimated_cost_usd"), "hard_cap_usd": hard_cap_usd,
              "hard_ttl_seconds": hard_ttl_seconds, "retry_cap": 0,
              "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
              "all_staged_objects_absent": cleanup.get("all_objects_absent"),
              "independent_watchdog": watchdog, "blockers": sorted(set(str(item) for item in blockers if str(item))),
              "raw_secret_values_recorded": False}
    write_json(job / "public_scene_aura_exact_residual_vast_result.json", result)
    return result


def materialize_aura_exact_residual_runtime_abstention(
    *,
    execution_result_path: str | Path,
    paid_admission_path: str | Path,
    bundle_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal a pre-entrypoint provider null without treating it as Aura execution.

    This is intentionally narrower than a general run failure.  It applies only
    when a rights-admitted, exact-mask packet created one provider instance but
    the provider failed before the sealed Aura bundle or entrypoint could run,
    and when both the owner and independent watchdog prove resource zero.
    """

    result_path = Path(execution_result_path).expanduser().resolve()
    admission_path = Path(paid_admission_path).expanduser().resolve()
    bundle = validate_aura_exact_residual_bundle(bundle_receipt_path)
    result = _read(result_path)
    admission = _read(admission_path)
    if (
        result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") != "blocked"
        or result.get("retry_cap") != 0
        or result.get("raw_result_path") is not None
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or admission.get("schema_version") != "paid_lane_admission.v1"
        or admission.get("status") != "admitted"
        or admission.get("retry_cap") != 0
        or admission.get("private_derived_upload_only") is not True
        or admission.get("raw_interiorgs_upload_authorized") is not False
        or admission.get("provider_training_authorized") is not False
        or admission.get("exact_mask_only_edits_required") is not True
        or (admission.get("allocation_binding") or {}).get("bundle_receipt_sha256")
        != bundle["receipt_sha256"]
        or result.get("bundle_sha256") != bundle["bundle_sha256"]
        or result.get("preflight_digest") != bundle["preflight_digest"]
    ):
        raise ValueError("aura_exact_residual_runtime_abstention_result_invalid")

    root = result_path.parent.resolve()

    def result_member(field: str, relative: str, code: str) -> Path:
        path = Path(str(result.get(field) or "")).expanduser().resolve()
        expected = (root / relative).resolve()
        if path != expected or not path.is_file() or path.is_symlink():
            raise ValueError(code)
        return path

    adapter_path = result_member(
        "adapter_result_path",
        "vast_provider_run/vast_provider_adapter_result.json",
        "aura_exact_residual_runtime_abstention_adapter_missing",
    )
    teardown_path = result_member(
        "teardown_manifest_path",
        "vast_provider_run/vast_teardown_manifest.json",
        "aura_exact_residual_runtime_abstention_teardown_missing",
    )
    watchdog_path = result_member(
        "watchdog_receipt_path",
        f"independent_vast_watchdog/{WATCHDOG_EVIDENCE_NAME}",
        "aura_exact_residual_runtime_abstention_watchdog_missing",
    )
    adapter = _read(adapter_path)
    teardown = _read(teardown_path)
    watchdog = _read(watchdog_path)
    classification = adapter.get("provider_attempt_classification")
    instance_ids = adapter.get("vast_instance_ids")
    if (
        adapter.get("schema_version") != "vast_provider_adapter_result.v1"
        or adapter.get("status") != "failed"
        or adapter.get("reason") != "vast_probe_failed"
        or adapter.get("provider_bundle_kind") != PROVIDER_BUNDLE_KIND
        or adapter.get("api_call_performed") is not True
        or adapter.get("provider_create_attempted") is not True
        or adapter.get("continuing_spend_from_this_run") is not False
        or not isinstance(instance_ids, list)
        or len(instance_ids) != 1
        or not isinstance(instance_ids[0], int)
        or instance_ids[0] <= 0
        or not isinstance(classification, Mapping)
        or classification.get("classification") != "pre_execution_provider_null"
        or classification.get("provider_bundle_started") is not False
        or classification.get("provider_entrypoint_started") is not False
        or classification.get("provider_output_returned") is not False
        or classification.get("automatic_requeue_authorized") is not False
        or classification.get("automatic_requeue_executed") is not False
        or classification.get("maximum_automatic_requeues") != 0
        or "vast_heartbeat_instance_exited" not in (adapter.get("blockers") or [])
        or teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("runner_gpu_teardown_completed") is not True
        or teardown.get("vast_instance_ids") != instance_ids
        or watchdog.get("schema_version") != "groot_oscar_runpod_canary_watchdog.v1"
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or (watchdog.get("recorded_vast_instance") or {}).get("instance_id")
        != str(instance_ids[0])
        or (watchdog.get("recorded_vast_instance_teardown") or {}).get("status") != "absent"
        or (watchdog.get("final_inventory") or {}).get("live_resource_count") != 0
    ):
        raise ValueError("aura_exact_residual_runtime_abstention_provider_evidence_invalid")

    cleanup_path = root / "object_store_staging" / "wam_provider_object_store_cleanup.json"
    avoidlist_path = root / "vast_machine_avoidlist.json"
    if not cleanup_path.is_file() or cleanup_path.is_symlink():
        raise ValueError("aura_exact_residual_runtime_abstention_object_store_cleanup_missing")
    if not avoidlist_path.is_file() or avoidlist_path.is_symlink():
        raise ValueError("aura_exact_residual_runtime_abstention_machine_avoidlist_missing")
    cleanup = _read(cleanup_path)
    avoidlist = _read(avoidlist_path)
    entries = avoidlist.get("entries")
    if (
        cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("status") != "completed"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or avoidlist.get("schema_version") != "vast_machine_avoidlist.v1"
        or avoidlist.get("status") != "completed"
        or not isinstance(entries, list)
        or not any(
            isinstance(entry, Mapping)
            and entry.get("instance_id") == instance_ids[0]
            and entry.get("reason") == "vast_startup_control_plane_did_not_reach_onstart_heartbeat"
            and entry.get("retry_policy")
            == "exclude_persistently_across_sibling_jobs_until_manual_review"
            for entry in entries
        )
    ):
        raise ValueError("aura_exact_residual_runtime_abstention_closeout_invalid")

    receipt: dict[str, Any] = {
        "schema_version": RUNTIME_ABSTENTION_SCHEMA_VERSION,
        "status": "abstained_provider_runtime_before_aura_entrypoint",
        "bundle_sha256": result.get("bundle_sha256"),
        "preflight_digest": result.get("preflight_digest"),
        "replacement_object_count": bundle["replacement_object_count"],
        "shared_camera_count": bundle["shared_camera_count"],
        "task_count": bundle["task_count"],
        "bundle_receipt": _record(Path(bundle["receipt_path"])),
        "paid_admission": _record(admission_path),
        "execution_result": _record(result_path),
        "provider_adapter": _record(adapter_path),
        "teardown": _record(teardown_path),
        "independent_watchdog": _record(watchdog_path),
        "object_store_cleanup": _record(cleanup_path),
        "machine_avoidlist": _record(avoidlist_path),
        "provider_instance_id": instance_ids[0],
        "aura_inpainting_executed": False,
        "provider_bundle_started": False,
        "provider_entrypoint_started": False,
        "provider_output_returned": False,
        "automatic_paid_retry_allowed": False,
        "automatic_paid_retry_executed": False,
        "provider_mutations_performed": 1,
        "continuing_spend_from_this_run": False,
        "provider_zero_confirmed": True,
        "smallest_missing_capability": (
            "rights_admitted_gpu_provider_runtime_that_reaches_the_sealed_Aura_exact_"
            "residual_container_entrypoint"
        ),
        "blockers": ["aura_exact_residual_provider_runtime_pre_entrypoint_null"],
        "claim_boundary": {
            "rights_admitted_backend_is_not_executed_backend": True,
            "inpainting_output_exists": False,
            "native_aura_frames_exist": False,
            "outside_mask_locality_measured": False,
            "multi_view_consistency_measured": False,
            "simready_or_policy_gate_unlocked": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, receipt)
    return receipt


__all__ = [
    "DEFAULT_IMAGE",
    "MAX_HARD_CAP_USD",
    "MAX_TTL_SECONDS",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "RUNTIME_ABSTENTION_SCHEMA_VERSION",
    "materialize_aura_exact_residual_runtime_abstention",
    "run_aura_exact_residual_vast",
    "validate_aura_exact_residual_bundle",
]
