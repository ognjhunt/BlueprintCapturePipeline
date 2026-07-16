"""Small RunPod launch-contract helpers for the WAM runner.

This keeps volume binding and watchdog handoff validation out of the already
grandfathered asynchronous runner while preserving its public behavior.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import parse_bool
from .groot_oscar_runpod_carrier_volume import (
    RUNTIME_ARCHIVE_ROOTS,
    runtime_bootstrap_shell_prefix,
    verify_carrier_volume_admission,
)

RUNPOD_WAM_LANE = "runpod_wam_async"


def _string(value: Any) -> str:
    return str(value or "").strip()


def redacted_payload_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload) if isinstance(payload, Mapping) else {}
    return {
        "cloudType": body.get("cloudType"),
        "computeType": body.get("computeType"),
        "gpuCount": body.get("gpuCount"),
        "gpuTypeIds": body.get("gpuTypeIds"),
        "gpuTypePriority": body.get("gpuTypePriority"),
        "volumeInGb": body.get("volumeInGb"),
        "networkVolumeId": body.get("networkVolumeId"),
        "dataCenterIds": body.get("dataCenterIds"),
        "containerDiskInGb": body.get("containerDiskInGb"),
        "minVCPUPerGPU": body.get("minVCPUPerGPU"),
        "minRAMPerGPU": body.get("minRAMPerGPU"),
        "name": body.get("name"),
        "imageName": body.get("imageName"),
        "dockerEntrypoint": body.get("dockerEntrypoint"),
        "dockerStartCmd_present": bool(body.get("dockerStartCmd")),
        "env_keys": sorted(
            (dict(body.get("env")) if isinstance(body.get("env"), Mapping) else {}).keys()
        ),
        "raw_secret_values_recorded": False,
    }


def build_pod_payload(
    *,
    job_name: str,
    image_name: str,
    gpu_type_ids: Sequence[str],
    provider_bundle_url: str,
    provider_output_put_url: str,
    provider_bundle_kind: str,
    model_secret_env: Mapping[str, str],
    provider_runtime_config_env: Mapping[str, str],
    container_disk_gb: int,
    volume_gb: int,
    cloud_type: str,
    allowed_cuda_versions: Sequence[str],
    min_vcpu_per_gpu: int,
    min_ram_per_gpu: int,
    provider_script: str,
    keep_on_success: bool,
    carrier_volume_admission: Mapping[str, Any] | None,
) -> dict[str, Any]:
    env = {
        "BLUEPRINT_EVAL_MANIFEST_URI": provider_bundle_url,
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": provider_output_put_url,
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "BLUEPRINT_RUNPOD_PROVIDER_BUNDLE_KIND": provider_bundle_kind,
        "WORK_DIR": "/workspace/blueprint_wam_provider",
    }
    if provider_bundle_kind == "unitree_unifolm":
        env["WORK_DIR"] = "/workspace/blueprint_unitree_unifolm_provider"
    elif provider_bundle_kind == "unitree_groot_n17_sonic":
        env["WORK_DIR"] = "/workspace/blueprint_unitree_groot_sonic_persistent_provider"
    if keep_on_success:
        env["BLUEPRINT_RUNPOD_KEEPALIVE_AFTER_SUCCESS"] = "1"
    env.update({key: value for key, value in provider_runtime_config_env.items() if _string(value)})
    env.update({key: value for key, value in model_secret_env.items() if _string(value)})
    carrier_volume = None
    if carrier_volume_admission is not None:
        carrier_volume = verify_carrier_volume_admission(
            carrier_volume_admission,
            expected_carrier_image_ref=image_name,
        )
        if carrier_volume["status"] != "verified":
            raise ValueError(
                "carrier_volume_admission_invalid:"
                + ",".join(carrier_volume["blockers"])
            )
        if any("H100" in gpu_type_id.upper() for gpu_type_id in gpu_type_ids):
            raise ValueError("carrier_volume_h100_disallowed")
        env.update(
            {
                "BLUEPRINT_RUNTIME_ARCHIVE_PATH": carrier_volume["runtime_archive_path"],
                "BLUEPRINT_RUNTIME_MANIFEST_PATH": carrier_volume["runtime_manifest_path"],
                "BLUEPRINT_RUNTIME_ARCHIVE_SHA256": carrier_volume[
                    "runtime_archive_sha256"
                ],
                "BLUEPRINT_RUNTIME_MANIFEST_SHA256": carrier_volume[
                    "runtime_manifest_sha256"
                ],
                "BLUEPRINT_RUNTIME_ARCHIVE_ROOTS": ":".join(RUNTIME_ARCHIVE_ROOTS),
                "BLUEPRINT_MODEL_CACHE_ROOT": carrier_volume["model_cache_root"],
                "BLUEPRINT_MODEL_CACHE_MANIFEST_PATH": carrier_volume[
                    "model_cache_manifest_path"
                ],
                "BLUEPRINT_MODEL_CACHE_MANIFEST_SHA256": carrier_volume[
                    "model_manifest_sha256"
                ],
                "BLUEPRINT_RUNPOD_CARRIER_VOLUME_ID": carrier_volume[
                    "network_volume_id"
                ],
            }
        )
        provider_script = runtime_bootstrap_shell_prefix() + "\n" + provider_script
    return {
        "cloudType": cloud_type,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": list(gpu_type_ids),
        "gpuTypePriority": "availability",
        "volumeInGb": volume_gb,
        "containerDiskInGb": container_disk_gb,
        "minVCPUPerGPU": min_vcpu_per_gpu,
        "minRAMPerGPU": min_ram_per_gpu,
        "name": job_name,
        "imageName": image_name,
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [provider_script],
        "ports": [],
        "volumeMountPath": "/workspace",
        "env": env,
        **(
            {
                "networkVolumeId": carrier_volume["network_volume_id"],
                "dataCenterIds": [carrier_volume["data_center_id"]],
            }
            if carrier_volume is not None
            else {}
        ),
        **(
            {"allowedCudaVersions": list(allowed_cuda_versions)}
            if allowed_cuda_versions
            else {}
        ),
    }


def extract_pod_id(response: Mapping[str, Any]) -> str:
    for key in ("id", "podId", "pod_id"):
        value = _string(response.get(key))
        if value:
            return value
    for key in ("pod", "data"):
        nested = response.get(key)
        nested = dict(nested) if isinstance(nested, Mapping) else {}
        for nested_key in ("id", "podId", "pod_id"):
            value = _string(nested.get(nested_key))
            if value:
                return value
    return ""


def selected_existing_pod_id(
    explicit: str,
    *,
    wam_existing_pod_id_env: str,
    provider_existing_pod_id_env: str,
) -> str:
    return (
        _string(explicit)
        or _string(os.getenv(wam_existing_pod_id_env))
        or _string(os.getenv(provider_existing_pod_id_env))
    )


def read_compatible_warm_candidate(
    *,
    candidate_path: Path,
    disabled: bool,
    disable_env: str,
    provider_bundle_kind: str,
    image_name: str,
    cloud_type: str,
) -> dict[str, Any]:
    if disabled:
        return {
            "status": "disabled",
            "path": str(candidate_path),
            "disable_env": disable_env,
            "raw_secret_values_recorded": False,
        }
    if not candidate_path.is_file():
        return {
            "status": "missing",
            "path": str(candidate_path),
            "raw_secret_values_recorded": False,
        }
    try:
        value = json.loads(candidate_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("warm_candidate_not_object")
        payload = dict(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "unreadable",
            "path": str(candidate_path),
            "error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    if _string(payload.get("status")) == "retired":
        return {
            "status": "retired",
            "path": str(candidate_path),
            "retired_pod_id": _string(payload.get("retired_pod_id")),
            "reason": _string(payload.get("reason")) or "warm_candidate_retired",
            "raw_secret_values_recorded": False,
        }
    pod_id = _string(payload.get("pod_id"))
    mismatches: dict[str, dict[str, str]] = {}
    expected = {
        "provider_bundle_kind": provider_bundle_kind,
        "image_name": image_name,
        "cloud_type": cloud_type,
    }
    for key, expected_value in expected.items():
        actual_value = _string(payload.get(key))
        if actual_value != expected_value:
            mismatches[key] = {"candidate": actual_value, "requested": expected_value}
    if not pod_id:
        return {
            "status": "incompatible",
            "reason": "warm_candidate_missing_pod_id",
            "path": str(candidate_path),
            "raw_secret_values_recorded": False,
        }
    if mismatches:
        return {
            "status": "incompatible",
            "reason": "warm_candidate_request_mismatch",
            "path": str(candidate_path),
            "mismatches": mismatches,
            "raw_secret_values_recorded": False,
        }
    running_preserved = parse_bool(
        payload.get("running_pod_preserved_for_hot_reuse"), default=False
    )
    stopped_preserved = parse_bool(
        payload.get("stopped_pod_preserved_for_warm_reuse"), default=False
    )
    reuse_kind = "existing_pod_candidate"
    if running_preserved:
        reuse_kind = "running_hot_candidate"
    elif stopped_preserved:
        reuse_kind = "stopped_warm_candidate"
    return {
        "status": "selected",
        "path": str(candidate_path),
        "pod_id": pod_id,
        "provider_bundle_kind": provider_bundle_kind,
        "image_name": image_name,
        "cloud_type": cloud_type,
        "reuse_kind": reuse_kind,
        "running_pod_preserved_for_hot_reuse": running_preserved,
        "stopped_pod_preserved_for_warm_reuse": stopped_preserved,
        "recorded_at": payload.get("generated_at"),
        "source_stop_manifest_path": payload.get("source_stop_manifest_path"),
        "source_keepalive_poll_manifest_path": payload.get(
            "source_keepalive_poll_manifest_path"
        ),
        "source_job_dir": payload.get("source_job_dir"),
        "claim_boundary": {
            "warm_candidate_reuses_provider_pod_id": True,
            "running_hot_candidate_still_uses_update_start_path": running_preserved,
            "resident_in_pod_job_queue_not_proven": running_preserved,
        },
        "raw_secret_values_recorded": False,
    }


def update_provider_lane_handoff_receipt(
    receipt_path: str | Path,
    *,
    pod_name: str,
    pending_teardown_record: str,
    pod_id: str = "",
) -> dict[str, Any]:
    """Bind watchdog control to the WAM runner before and after Pod creation."""

    path = Path(receipt_path).expanduser()
    if not path.is_absolute():
        raise ValueError("provider_lane_handoff_receipt_path_not_absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError("provider_lane_handoff_receipt_missing") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError("provider_lane_handoff_receipt_not_regular_file")
    if metadata.st_mode & 0o077:
        raise ValueError("provider_lane_handoff_receipt_permissions_unsafe")
    if metadata.st_uid not in {0, os.geteuid()}:
        raise ValueError("provider_lane_handoff_receipt_owner_untrusted")
    if metadata.st_size > 1024 * 1024:
        raise ValueError("provider_lane_handoff_receipt_oversized")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("provider_lane_handoff_receipt_unreadable") from exc
    receipt = dict(value) if isinstance(value, Mapping) else {}
    prefix = _string(receipt.get("pod_name_prefix"))
    if (
        receipt.get("status") != "accepted"
        or receipt.get("campaign_kind") != "persistent_policy_wam_loop"
        or not prefix
        or not pod_name.startswith(prefix)
    ):
        raise ValueError("provider_lane_handoff_receipt_binding_invalid")
    pending_path = Path(pending_teardown_record).expanduser()
    if not pending_path.is_absolute():
        raise ValueError("provider_lane_handoff_pending_teardown_path_not_absolute")
    try:
        pending_value = json.loads(pending_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("provider_lane_handoff_pending_teardown_unreadable") from exc
    pending = dict(pending_value) if isinstance(pending_value, Mapping) else {}
    if (
        pending.get("status") != "open"
        or pending.get("provider") != "runpod"
        or pending.get("lane") != RUNPOD_WAM_LANE
        or pending.get("resource_kind") != "compute_instance"
        or pending.get("resource_name") != pod_name
    ):
        raise ValueError("provider_lane_handoff_pending_teardown_binding_invalid")
    bound_pending_pod_id = _string(pending.get("instance_id"))
    if pod_id and bound_pending_pod_id != pod_id:
        raise ValueError("provider_lane_handoff_pending_teardown_pod_id_mismatch")
    receipt.update(
        {
            "pod_id": pod_id or None,
            "pod_name": pod_name,
            "pod_pending_teardown_record": str(pending_path),
            "pre_provider_mutation_confirmed_absent": False,
            "provider_mutation_state": "pod_id_bound" if pod_id else "pending_create",
            "raw_secret_values_recorded": False,
        }
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return {
        "status": "pod_id_bound" if pod_id else "pending_create_bound",
        "path": str(path),
        "pod_id_present": bool(pod_id),
        "raw_secret_values_recorded": False,
    }


def confirm_provider_lane_handoff_no_allocation(
    receipt_path: str | Path,
    *,
    pod_name: str,
    pending_teardown_record: str | Path,
) -> dict[str, Any]:
    """Return a pre-create handoff to provider-absent state after cancellation."""

    path = Path(receipt_path).expanduser()
    pending_path = Path(pending_teardown_record).expanduser()
    if not path.is_absolute() or not pending_path.is_absolute():
        raise ValueError("provider_lane_handoff_no_allocation_path_not_absolute")
    try:
        metadata = path.lstat()
        receipt_value = json.loads(path.read_text(encoding="utf-8"))
        pending_value = json.loads(pending_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "provider_lane_handoff_no_allocation_evidence_unreadable"
        ) from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or metadata.st_mode & 0o077
        or metadata.st_uid not in {0, os.geteuid()}
        or metadata.st_size > 1024 * 1024
    ):
        raise ValueError("provider_lane_handoff_no_allocation_receipt_unsafe")
    receipt = dict(receipt_value) if isinstance(receipt_value, Mapping) else {}
    pending = dict(pending_value) if isinstance(pending_value, Mapping) else {}
    if (
        receipt.get("status") != "accepted"
        or receipt.get("campaign_kind") != "persistent_policy_wam_loop"
        or receipt.get("pod_name") != pod_name
        or receipt.get("pod_pending_teardown_record") != str(pending_path)
        or _string(receipt.get("pod_id"))
        or receipt.get("provider_mutation_state") != "pending_create"
        or pending.get("status") != "cancelled_no_allocation"
        or pending.get("provider") != "runpod"
        or pending.get("lane") != RUNPOD_WAM_LANE
        or pending.get("resource_kind") != "compute_instance"
        or pending.get("resource_name") != pod_name
        or _string(pending.get("instance_id"))
    ):
        raise ValueError("provider_lane_handoff_no_allocation_binding_invalid")
    receipt.update(
        {
            "pod_id": None,
            "pod_pending_teardown_record": None,
            "pre_provider_mutation_confirmed_absent": True,
            "provider_mutation_state": "no_allocation_confirmed",
            "no_allocation_pending_teardown_record": str(pending_path),
            "raw_secret_values_recorded": False,
        }
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return {
        "status": "no_allocation_confirmed",
        "path": str(path),
        "cancelled_pending_teardown_record": str(pending_path),
        "raw_secret_values_recorded": False,
    }
