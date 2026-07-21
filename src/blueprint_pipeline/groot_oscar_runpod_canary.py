"""Fail-closed GR00T + OSCAR RunPod GPU canary launcher.

The provider adapter is unreachable until the exact release, model cache,
existing network volume, GPU/CUDA constraints, budget, and already-armed
watchdog produce an admitted record.  This launcher is intentionally scoped to
the startup canary and the fixed three-action policy smoke; it is not an image
builder, a general robot-evaluation launcher, or a customer cold-start path.
"""

from __future__ import annotations

import json
import math
import os
import stat
import time
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from .common import write_json
from .gpu_render_providers import _runpod_call, get_render_provider
from .groot_oscar_infrastructure_admission import (
    SERVE_SCHEMA_VERSION,
    build_runpod_serve_plane_admission,
)
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
)
from .paid_provider_lane_lease import (
    accept_paid_provider_lane_lease_handoff,
    restore_paid_provider_lane_lease_to_retained_watchdog,
)
from .production_gpu_campaign_budget import (
    CampaignBudgetExceeded,
    ProductionGpuCampaignBudget,
)
from .groot_oscar_runpod_preflight import collect_runpod_preflight
from .groot_oscar_runpod_watchdog import terminate_canary_resources
from .runpod_provider_adapter import (
    RUNPOD_IMAGE_STARTUP_CANARY_MODE,
    RUNPOD_STRICT_POLICY_SMOKE_MODE,
    run_runpod_provider_adapter,
)

STARTUP_PROBE_KIND = "startup"
STRICT_POLICY_SMOKE_PROBE_KIND = "strict-policy-smoke"
CANONICAL_PROBE_KINDS = (STARTUP_PROBE_KIND, STRICT_POLICY_SMOKE_PROBE_KIND)
STRICT_POLICY_SMOKE_RESULT_NAME = "groot_oscar_runpod_strict_policy_smoke.json"
STRICT_POLICY_SMOKE_LOG_NAME = "gr00t_policy_server.log"
STRICT_POLICY_SMOKE_MAX_ZIP_BYTES = 16 * 1024 * 1024
STRICT_POLICY_SMOKE_HARD_TIMEOUT_SECONDS = 420
STRICT_POLICY_SMOKE_STARTUP_ARTIFACT_TIMEOUT_SECONDS = 420
STRICT_POLICY_SMOKE_WATCHDOG_TTL_SECONDS = 480
PREFLIGHT_MAX_AGE_SECONDS = 300
PREFLIGHT_FUTURE_TOLERANCE_SECONDS = 30
RUNTIME_MANIFEST_SIGNED_PUT_URL_ENV = (
    "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"
)


def refresh_runpod_preflight(
    *,
    preflight: Mapping[str, Any],
    volume_getter: Callable[[str], tuple[int, Mapping[str, Any]]],
    capacity_probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    inventory_probe: Callable[[str | None], Mapping[str, Any]],
    clock: Callable[[], float] = time.time,
    process_argv_probe: Callable[[int], Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Recheck every mutable provider fact immediately before allocation."""

    volume = preflight.get("volume")
    volume = volume if isinstance(volume, Mapping) else {}
    runtime = preflight.get("runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    spend = preflight.get("spend")
    spend = spend if isinstance(spend, Mapping) else {}
    watchdog = {
        "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
        "status": "armed",
        "independent_process": spend.get("independent_teardown_watchdog") is True,
        "pid": spend.get("watchdog_pid"),
        "deadline_epoch": spend.get("watchdog_deadline_epoch"),
        "pod_name_prefix": spend.get("watchdog_pod_name_prefix"),
        "watchdog_out_dir": spend.get("watchdog_out_dir"),
    }
    kwargs: dict[str, Any] = {}
    if process_argv_probe is not None:
        kwargs["process_argv_probe"] = process_argv_probe
    return collect_runpod_preflight(
        network_volume_id=str(volume.get("id") or ""),
        model_cache_path=str(volume.get("model_cache_path") or ""),
        gpu_type_id=str(runtime.get("gpu_type_id") or ""),
        required_cuda_version=str(runtime.get("required_cuda_version") or ""),
        name_prefix=str(spend.get("watchdog_pod_name_prefix") or ""),
        watchdog=watchdog,
        model_volume_watchdog_handoff=(
            preflight.get("model_volume_watchdog_handoff")
            if isinstance(preflight.get("model_volume_watchdog_handoff"), Mapping)
            else {}
        ),
        max_spend_usd=float(spend.get("max_spend_usd") or 0),
        paid_mutation_authorized=spend.get("paid_mutation_authorized") is True,
        volume_getter=volume_getter,
        capacity_probe=capacity_probe,
        inventory_probe=inventory_probe,
        clock=clock,
        **kwargs,
    )


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(value)


def _preflight_freshness_blockers(
    preflight: Mapping[str, Any], *, observed_now_epoch: float
) -> list[str]:
    observed_at = preflight.get("observed_at_epoch")
    if observed_at is None:
        return ["runpod_preflight_observed_at_missing"]
    if type(observed_at) not in {int, float} or not math.isfinite(float(observed_at)):
        return ["runpod_preflight_observed_at_invalid"]
    age_seconds = float(observed_now_epoch) - float(observed_at)
    blockers: list[str] = []
    if age_seconds > PREFLIGHT_MAX_AGE_SECONDS:
        blockers.append("runpod_preflight_stale")
    if age_seconds < -PREFLIGHT_FUTURE_TOLERANCE_SECONDS:
        blockers.append("runpod_preflight_observed_at_in_future")
    return blockers


def _read_private_signed_put_url(
    path_value: str | Path | None,
) -> tuple[str, dict[str, Any]]:
    blockers: list[str] = []
    path = Path(path_value).expanduser() if path_value else None
    url = ""
    mode_octal: str | None = None
    regular_private_file = False
    if path is None:
        blockers.append("runtime_manifest_signed_put_url_file_missing")
    else:
        descriptor: int | None = None
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            return "", {
                "status": "blocked",
                "blockers": [
                    "runtime_manifest_signed_put_url_no_follow_unavailable"
                ],
                "private_file_present": False,
                "private_file_mode": None,
                "https_signed_put_url_present": False,
                "signed_put_url_value_recorded": False,
            }
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | no_follow,
            )
            metadata = os.fstat(descriptor)
            mode_octal = oct(stat.S_IMODE(metadata.st_mode))
            if not stat.S_ISREG(metadata.st_mode):
                blockers.append("runtime_manifest_signed_put_url_file_not_regular")
            elif stat.S_IMODE(metadata.st_mode) != 0o600:
                blockers.append("runtime_manifest_signed_put_url_file_mode_not_0600")
            elif metadata.st_size > 8192:
                blockers.append("runtime_manifest_signed_put_url_file_oversized")
            else:
                regular_private_file = True
                with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                    descriptor = None
                    url = handle.read(8193).strip()
        except (OSError, UnicodeError):
            blockers.append("runtime_manifest_signed_put_url_file_unreadable")
        finally:
            if descriptor is not None:
                os.close(descriptor)
    parsed = urlparse(url)
    if url and (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or any(character.isspace() for character in url)
    ):
        blockers.append("runtime_manifest_signed_put_url_invalid")
        url = ""
    if not url and not blockers:
        blockers.append("runtime_manifest_signed_put_url_empty")
    return url, {
        "status": "ready" if url and not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "private_file_present": regular_private_file,
        "private_file_mode": mode_octal,
        "https_signed_put_url_present": bool(url),
        "signed_put_url_value_recorded": False,
    }


def validate_strict_policy_smoke_output(
    *, output_zip: str | Path, evidence_out: str | Path
) -> dict[str, Any]:
    """Validate the fixed three-action smoke artifact without executing it."""

    source = Path(output_zip).expanduser().resolve()
    blockers: list[str] = []
    payload: dict[str, Any] = {}
    inventory: list[str] = []
    try:
        if not source.is_file() or source.stat().st_size > STRICT_POLICY_SMOKE_MAX_ZIP_BYTES:
            raise ValueError("strict_policy_smoke_zip_missing_or_oversized")
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            inventory = [info.filename for info in infos]
            if len(inventory) != len(set(inventory)):
                blockers.append("strict_policy_smoke_zip_duplicate_member")
            allowed = {STRICT_POLICY_SMOKE_RESULT_NAME, STRICT_POLICY_SMOKE_LOG_NAME}
            if not set(inventory).issubset(allowed):
                blockers.append("strict_policy_smoke_zip_inventory_invalid")
            for info in infos:
                mode = info.external_attr >> 16
                if info.is_dir() or stat.S_ISLNK(mode) or info.filename.startswith(("/", "../")):
                    blockers.append("strict_policy_smoke_zip_member_type_invalid")
            if STRICT_POLICY_SMOKE_RESULT_NAME not in inventory:
                blockers.append("strict_policy_smoke_result_missing")
            elif not blockers:
                value = json.loads(archive.read(STRICT_POLICY_SMOKE_RESULT_NAME))
                payload = dict(value) if isinstance(value, Mapping) else {}
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        blockers.append(f"strict_policy_smoke_zip_invalid:{type(exc).__name__}")
    actions = payload.get("fresh_learned_action_trace")
    actions = actions if isinstance(actions, list) else []
    if payload:
        if payload.get("schema_version") != "groot_oscar_runpod_strict_policy_smoke.v1":
            blockers.append("strict_policy_smoke_schema_invalid")
        if payload.get("status") != "completed":
            blockers.append("strict_policy_smoke_not_completed")
        if payload.get("requested_action_count") != 3 or payload.get(
            "completed_action_count"
        ) != 3:
            blockers.append("strict_policy_smoke_action_count_invalid")
        if len(actions) != 3 or not all(
            isinstance(action, Mapping)
            and isinstance(action.get("action_chunk"), list)
            and bool(action.get("action_chunk"))
            for action in actions
        ):
            blockers.append("strict_policy_smoke_learned_action_trace_invalid")
        if payload.get("model_execution_proven") is not True or payload.get(
            "policy_action_model_command_ran"
        ) is not True:
            blockers.append("strict_policy_smoke_model_execution_not_proven")
        if payload.get("physical_robot_control_performed") is not False:
            blockers.append("strict_policy_smoke_physical_control_boundary_invalid")
        if payload.get("raw_secret_values_recorded") is not False:
            blockers.append("strict_policy_smoke_secret_policy_invalid")
    result = {
        "schema_version": "groot_oscar_runpod_strict_policy_smoke_validation.v1",
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "output_zip": str(source),
        "zip_inventory": inventory,
        "completed_action_count": len(actions),
        "model_execution_proven": payload.get("model_execution_proven") is True,
        "task_success_proven": False,
        "physical_robot_control_performed": False,
        "raw_secret_values_recorded": False,
    }
    write_json(Path(evidence_out), result)
    return result


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reserve_campaign_budget(
    config: Mapping[str, Any], *, reservation_id: str
) -> dict[str, Any]:
    try:
        reservation_seconds = int(config["reservation_gpu_seconds"])
        future_allowance_seconds = int(
            config["future_campaign_allowance_gpu_seconds"]
        )
        combined_plan_seconds = reservation_seconds + future_allowance_seconds
        if config.get("reduced_canary_timeout_acknowledged") is not True:
            raise ValueError("gpu_canary_reduced_timeout_authorization_missing")
        campaign_stage = str(config.get("campaign_stage") or "")
        if campaign_stage not in {"gpu_canary", "persistent_carrier_campaign"}:
            raise ValueError("gpu_canary_campaign_stage_invalid")
        if reservation_seconds > int(
            config["maximum_canary_reservation_gpu_seconds"]
        ):
            raise ValueError("gpu_canary_stage_reservation_exceeds_plan")
        if future_allowance_seconds > int(
            config["maximum_future_campaign_allowance_gpu_seconds"]
        ):
            raise ValueError("future_campaign_allowance_exceeds_plan")
        if combined_plan_seconds > int(config["maximum_combined_plan_gpu_seconds"]):
            raise ValueError("combined_gpu_plan_exceeds_reduced_ceiling")
        if (
            float(config.get("initial_spent_usd") or 0)
            < float(config.get("minimum_reconciled_spend_usd") or 0)
            or int(config.get("initial_used_gpu_seconds") or 0)
            < int(config.get("minimum_reconciled_gpu_seconds") or 0)
        ):
            raise ValueError("gpu_canary_cumulative_baseline_understated")
        if (
            int(config["initial_used_gpu_seconds"]) + combined_plan_seconds
            > int(config["combined_gpu_wall_cap_seconds"])
        ):
            raise ValueError("combined_gpu_plan_exceeds_campaign_wall_cap")
        budget = ProductionGpuCampaignBudget(
            str(config.get("ledger_path") or ""),
            initial_spent_usd=float(config["initial_spent_usd"]),
            initial_used_gpu_seconds=int(config["initial_used_gpu_seconds"]),
            total_spend_cap_usd=float(config["total_spend_cap_usd"]),
            combined_gpu_wall_cap_seconds=int(config["combined_gpu_wall_cap_seconds"]),
        )
        reservation = budget.reserve(
            reservation_id=reservation_id,
            gpu_seconds=reservation_seconds,
            max_hourly_rate_usd=float(config["max_hourly_rate_usd"]),
        )
    except (CampaignBudgetExceeded, KeyError, TypeError, ValueError) as exc:
        admission = getattr(exc, "admission", {})
        return {"status": "blocked", "blockers": [str(admission.get("blocker") or exc)]}
    if reservation.get("status") != "open":
        return {"status": "blocked", "blockers": ["campaign_budget_reservation_not_open"]}
    return {
        "status": "reserved",
        "ledger_path": str(Path(str(config["ledger_path"])).expanduser().resolve()),
        "reservation_id": reservation_id,
        "reserved_at_epoch": time.time(),
        "reservation": reservation,
        "plan": {
            "campaign_stage": campaign_stage,
            "canary_reservation_gpu_seconds": reservation_seconds,
            "future_campaign_allowance_gpu_seconds": future_allowance_seconds,
            "combined_plan_gpu_seconds": combined_plan_seconds,
        },
        "identity": {
            key: config[key]
            for key in (
                "initial_spent_usd",
                "initial_used_gpu_seconds",
                "total_spend_cap_usd",
                "combined_gpu_wall_cap_seconds",
            )
        },
        "raw_secret_values_recorded": False,
    }


def _settle_zero_budget(context: Mapping[str, Any], *, outcome: str) -> dict[str, Any]:
    if context.get("status") != "reserved":
        return {"status": "not_reserved"}
    identity = context["identity"]
    budget = ProductionGpuCampaignBudget(
        context["ledger_path"],
        initial_spent_usd=identity["initial_spent_usd"],
        initial_used_gpu_seconds=identity["initial_used_gpu_seconds"],
        total_spend_cap_usd=identity["total_spend_cap_usd"],
        combined_gpu_wall_cap_seconds=identity["combined_gpu_wall_cap_seconds"],
    )
    return budget.settle(
        reservation_id=str(context["reservation_id"]),
        charged_gpu_seconds=0,
        charged_usd=0,
        outcome=outcome,
    )


def _recover_accepted_handoff_before_provider_mutation(
    *,
    acceptance: Mapping[str, Any],
    budget_context: Mapping[str, Any],
    watchdog_out_dir: str,
    pod_name_prefix: str,
    outcome: str,
    pending_path: str | None = None,
) -> dict[str, Any]:
    """Restore every accepted control-plane obligation after a local failure."""

    try:
        pending_result = (
            cancel_pending_teardown(
                pending_path,
                reason=outcome,
                evidence={"provider_mutations_performed": 0},
            )
            if pending_path
            else {"status": "not_opened"}
        )
    except Exception as exc:  # noqa: BLE001 - continue recovery steps
        pending_result = {"status": "error", "error_type": type(exc).__name__}
    try:
        restore_result = restore_paid_provider_lane_lease_to_retained_watchdog(
            acceptance
        )
    except Exception as exc:  # noqa: BLE001 - continue recovery steps
        restore_result = {"status": "error", "error_type": type(exc).__name__}
    try:
        settlement = _settle_zero_budget(budget_context, outcome=outcome)
    except Exception as exc:  # noqa: BLE001 - preserve the open reservation
        settlement = {"status": "error", "error_type": type(exc).__name__}
    terminal = bool(
        pending_result.get("status") in {"not_opened", "cancelled_no_allocation"}
        and restore_result.get("status") == "restored"
        and settlement.get("status") == "settled"
    )
    recovery = {
        "status": "terminal_no_allocation" if terminal else "control_plane_open",
        "provider_mutations_performed": 0,
        "pending_teardown": pending_result,
        "provider_lane_owner_return": restore_result,
        "campaign_budget_settlement": settlement,
    }
    if not terminal:
        try:
            _write_private_json(
                Path(watchdog_out_dir) / "provider_lane_handoff_receipt.json",
                {
                    **dict(acceptance),
                    "pod_pending_teardown_record": pending_path,
                    "pod_id": None,
                    "pod_name_prefix": pod_name_prefix,
                    "campaign_budget": dict(budget_context),
                    "pre_provider_mutation_confirmed_absent": True,
                    "pre_provider_recovery": recovery,
                },
            )
            recovery["recovery_receipt_written"] = True
        except Exception as exc:  # noqa: BLE001 - keep cleanup evidence fail closed
            recovery["recovery_receipt_written"] = False
            recovery["recovery_receipt_error_type"] = type(exc).__name__
    return recovery


def bind_canary_request(
    *,
    request: Mapping[str, Any],
    admission: Mapping[str, Any],
    probe_kind: str = STARTUP_PROBE_KIND,
) -> dict[str, Any]:
    """Bind the adapter request to the already-admitted immutable tuple."""

    result = deepcopy(dict(request))
    shape = result.get("provider_request_shape")
    shape = shape if isinstance(shape, dict) else {}
    image = shape.get("image")
    image = image if isinstance(image, dict) else {}
    configured = str(image.get("configured_image_ref") or "").strip()
    admitted_image = str(admission.get("release_image_ref") or "").strip()
    blockers: list[str] = []
    if probe_kind not in CANONICAL_PROBE_KINDS:
        blockers.append("runpod_canary_probe_kind_unsupported")
    if configured != admitted_image:
        blockers.append("runpod_request_release_image_differs_from_admission")
    gpu = shape.get("gpu")
    gpu = gpu if isinstance(gpu, dict) else {}
    admitted_gpu = str(admission.get("gpu_type_id") or "").strip()
    configured_gpu = str(
        gpu.get("preferred_gpu_type_id") or gpu.get("preferred_gpu_class") or ""
    ).strip()
    if configured_gpu and configured_gpu != admitted_gpu:
        blockers.append("runpod_request_gpu_differs_from_admission")
    admitted_cache_path = str(admission.get("model_cache_path") or "").strip()
    cache = shape.get("cache")
    cache = cache if isinstance(cache, dict) else {}
    cache_paths = cache.get("paths")
    cache_paths = cache_paths if isinstance(cache_paths, dict) else {}
    configured_cache_path = str(cache_paths.get("groot_oscar_models") or "").strip()
    if configured_cache_path and configured_cache_path != admitted_cache_path:
        blockers.append("runpod_request_model_cache_path_differs_from_admission")
    cache_paths["groot_oscar_models"] = admitted_cache_path
    cache["paths"] = cache_paths
    shape["cache"] = cache
    environment = shape.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    plaintext_names = environment.get("plaintext_env_var_names")
    plaintext_names = plaintext_names if isinstance(plaintext_names, list) else []
    digest_env = "BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST"
    if digest_env not in plaintext_names:
        plaintext_names.append(digest_env)
    plaintext_values = environment.get("plaintext_env_values")
    plaintext_values = plaintext_values if isinstance(plaintext_values, dict) else {}
    plaintext_values[digest_env] = admission.get("model_manifest_digest")
    environment["plaintext_env_var_names"] = plaintext_names
    environment["plaintext_env_values"] = plaintext_values
    shape["environment"] = environment
    shape["network_volume_id"] = admission.get("network_volume_id")
    shape["data_center_id"] = admission.get("data_center_id")
    shape["allowed_cuda_versions"] = [admission.get("required_cuda_version")]
    shape["docker_entrypoint"] = ["/opt/blueprint/thin_release_entrypoint.sh"]
    gpu["preferred_gpu_type_id"] = admitted_gpu
    gpu["provider_gpu_priority"] = [admitted_gpu]
    shape["gpu"] = gpu
    shape["image"] = image
    if probe_kind == STRICT_POLICY_SMOKE_PROBE_KIND:
        result["operation"] = "enqueue_runpod_strict_policy_smoke"
        shape["operation"] = "enqueue_runpod_strict_policy_smoke"
        shape.pop("command", None)
        limits = shape.get("limits")
        limits = limits if isinstance(limits, dict) else {}
        limits["hard_timeout_seconds"] = STRICT_POLICY_SMOKE_HARD_TIMEOUT_SECONDS
        limits["startup_artifact_timeout_seconds"] = (
            STRICT_POLICY_SMOKE_STARTUP_ARTIFACT_TIMEOUT_SECONDS
        )
        limits["external_watchdog_ttl_seconds"] = (
            STRICT_POLICY_SMOKE_WATCHDOG_TTL_SECONDS
        )
        shape["limits"] = limits
        claim_boundary = shape.get("claim_boundary")
        claim_boundary = claim_boundary if isinstance(claim_boundary, dict) else {}
        claim_boundary.update(
            {
                "startup_canary_only": False,
                "strict_policy_smoke_only": True,
                "fresh_three_action_policy_smoke_required": True,
                "does_not_prove_task_success": True,
                "does_not_prove_physical_robot_control": True,
            }
        )
        shape["claim_boundary"] = claim_boundary
    result["provider_request_shape"] = shape
    return {
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "request": result,
    }


def prepare_canary_launch(
    *,
    request: Mapping[str, Any],
    release: Mapping[str, Any],
    model_cache: Mapping[str, Any],
    preflight: Mapping[str, Any],
    expected_source_commit: str,
    probe_kind: str = STARTUP_PROBE_KIND,
) -> dict[str, Any]:
    volume = preflight.get("volume")
    runtime = preflight.get("runtime")
    spend = preflight.get("spend")
    admission = build_runpod_serve_plane_admission(
        release=release,
        model_cache=model_cache,
        volume=volume if isinstance(volume, Mapping) else {},
        runtime=runtime if isinstance(runtime, Mapping) else {},
        spend=spend if isinstance(spend, Mapping) else {},
        expected_source_commit=expected_source_commit,
    )
    if preflight.get("status") != "verified":
        admission = {
            **admission,
            "status": "blocked",
            "blockers": sorted(
                set([*admission.get("blockers", []), "runpod_preflight_bundle_not_verified"])
            ),
        }
    bound = bind_canary_request(
        request=request,
        admission=admission,
        probe_kind=probe_kind,
    )
    blockers = list(admission.get("blockers") or []) + list(bound["blockers"])
    return {
        "status": "admitted" if not blockers and admission["status"] == "admitted" else "blocked",
        "blockers": sorted(set(blockers)),
        "admission": admission,
        "bound_request": bound["request"],
        "provider_mutations_performed": 0,
    }


def _finalize_adapter_allocation(
    *,
    adapter: Mapping[str, Any],
    adapter_output: str | Path,
    pod_name: str,
    release_image_ref: str,
) -> dict[str, Any]:
    """Require an authoritative pod id before the paid canary can succeed."""

    result = dict(adapter)
    response = adapter.get("runpod_response")
    response = response if isinstance(response, Mapping) else {}
    pod_id = str(response.get("id") or "").strip()
    write_json(
        Path(adapter_output).resolve().parent / "warm_serve_pod.json",
        {
            "schema_version": "groot_oscar_runpod_canary_allocation.v1",
            "status": "allocated" if pod_id else "allocation_ambiguous",
            "pod_id": pod_id or None,
            "pod_name": pod_name,
            "release_image_ref": release_image_ref,
        },
    )
    if not pod_id:
        result["status"] = "failed"
        result["blockers"] = sorted(
            set([*(result.get("blockers") or []), "runpod_canary_pod_id_missing"])
        )
        result["provider_allocation_ambiguous"] = True
        write_json(Path(adapter_output), result)
    return result


def run_canary(
    *,
    provider_launch_request: str | Path,
    release_evidence: str | Path,
    model_cache_evidence: str | Path,
    preflight_bundle: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    pod_name: str,
    execute: bool,
    expected_source_commit: str,
    provider_output_put_url_file: str | Path | None,
    campaign_budget: Mapping[str, Any] | None = None,
    probe_kind: str = STARTUP_PROBE_KIND,
) -> dict[str, Any]:
    """Run the adapter only through the canonical GPU-canary allocator."""

    preflight = _read(preflight_bundle)
    provider = None
    budget_context: dict[str, Any] = {}
    acceptance: dict[str, Any] = {}
    watchdog_out_dir = ""
    refresh_path = Path(adapter_output).resolve().parent / "runpod_preflight_launch_refresh.json"
    if execute:
        provider = get_render_provider("runpod")
        key = provider._key()  # type: ignore[attr-defined]
        if key:
            def volume_getter(volume_id: str) -> tuple[int, Mapping[str, Any]]:
                status, payload = _runpod_call(
                    "GET", f"/networkvolumes/{volume_id}", None, key=key, timeout=30
                )
                return status, payload if isinstance(payload, Mapping) else {}

            preflight = refresh_runpod_preflight(
                preflight=preflight,
                volume_getter=volume_getter,
                capacity_probe=provider.capacity_preflight,
                inventory_probe=lambda prefix: provider.billable_inventory(
                    name_prefix=prefix
                ),
            )
        else:
            preflight = {
                "schema_version": "groot_oscar_runpod_preflight_bundle.v1",
                "status": "blocked",
                "blockers": ["runpod_api_key_missing_at_launch_refresh"],
                "provider_mutations_performed": 0,
            }
        write_json(refresh_path, preflight)
    signed_put_url, output_sink = _read_private_signed_put_url(
        provider_output_put_url_file
    )
    prepared = prepare_canary_launch(
        request=_read(provider_launch_request),
        release=_read(release_evidence),
        model_cache=_read(model_cache_evidence),
        preflight=preflight,
        expected_source_commit=expected_source_commit,
        probe_kind=probe_kind,
    )
    pre_provider_blockers = [
        *_preflight_freshness_blockers(preflight, observed_now_epoch=time.time()),
        *output_sink["blockers"],
    ]
    if pre_provider_blockers:
        prepared = {
            **prepared,
            "status": "blocked",
            "blockers": sorted(
                set([*prepared.get("blockers", []), *pre_provider_blockers])
            ),
        }
    prepared["preflight_observed_at_epoch"] = preflight.get("observed_at_epoch")
    prepared["runtime_manifest_output_sink"] = output_sink
    spend = preflight.get("spend")
    spend = spend if isinstance(spend, Mapping) else {}
    watchdog_prefix = str(spend.get("watchdog_pod_name_prefix") or "").strip()
    if watchdog_prefix and not pod_name.startswith(watchdog_prefix):
        prepared = {
            **prepared,
            "status": "blocked",
            "blockers": sorted(
                set(
                    [
                        *prepared.get("blockers", []),
                        "runpod_canary_pod_name_outside_watchdog_scope",
                    ]
                )
            ),
        }
    write_json(Path(admission_out), prepared)
    if prepared["status"] != "admitted":
        return prepared
    if execute:
        watchdog_out_dir = str(spend.get("watchdog_out_dir") or "").strip()
        if not watchdog_out_dir or not Path(watchdog_out_dir).is_absolute():
            prepared = {
                **prepared,
                "status": "blocked",
                "blockers": ["runpod_canary_watchdog_out_dir_unverified"],
                "provider_mutations_performed": 0,
            }
            write_json(Path(admission_out), prepared)
            return prepared
        budget_context = _reserve_campaign_budget(
            campaign_budget or {}, reservation_id=pod_name
        )
        if budget_context.get("status") == "reserved":
            reservation = budget_context.get("reservation")
            reservation = reservation if isinstance(reservation, Mapping) else {}
            reserved_seconds = int(reservation.get("reserved_gpu_seconds") or 0)
            hard_ttl_seconds = int(spend.get("hard_ttl_seconds") or 0)
            watchdog_deadline_epoch = float(
                spend.get("watchdog_deadline_epoch") or 0
            )
            watchdog_remaining_seconds = max(
                0,
                math.ceil(
                    watchdog_deadline_epoch
                    - float(budget_context.get("reserved_at_epoch") or 0)
                ),
            )
            budget_context["watchdog_contract"] = {
                "hard_ttl_seconds": hard_ttl_seconds,
                "watchdog_deadline_epoch": watchdog_deadline_epoch,
                "watchdog_remaining_seconds_at_reservation": (
                    watchdog_remaining_seconds
                ),
                "reserved_gpu_seconds": reserved_seconds,
            }
            if (
                hard_ttl_seconds <= 0
                or hard_ttl_seconds > reserved_seconds
                or watchdog_remaining_seconds > reserved_seconds
            ):
                settlement = _settle_zero_budget(
                    budget_context,
                    outcome="watchdog_ttl_exceeds_reservation_no_mutation",
                )
                budget_context = {
                    **budget_context,
                    "status": "blocked",
                    "blockers": ["gpu_canary_watchdog_exceeds_budget_reservation"],
                    "zero_settlement": settlement,
                }
        try:
            write_json(
                Path(adapter_output).resolve().parent
                / "campaign_budget_reservation.json",
                budget_context,
            )
        except Exception as exc:  # noqa: BLE001 - no provider mutation has occurred
            _settle_zero_budget(
                budget_context, outcome="budget_evidence_write_failed_no_mutation"
            )
            return {
                "status": "blocked",
                "blockers": ["campaign_budget_reservation_evidence_write_failed"],
                "provider_mutations_performed": 0,
                "error_type": type(exc).__name__,
            }
        if budget_context.get("status") != "reserved":
            prepared = {
                **prepared,
                "status": "blocked",
                "blockers": list(budget_context.get("blockers") or []),
                "provider_mutations_performed": 0,
            }
            write_json(Path(admission_out), prepared)
            return prepared
        volume_handoff = preflight.get("model_volume_watchdog_handoff")
        volume_handoff = (
            volume_handoff if isinstance(volume_handoff, Mapping) else {}
        )
        lane_handoff = volume_handoff.get("provider_lane_handoff")
        lane_handoff = lane_handoff if isinstance(lane_handoff, Mapping) else {}
        binding = lane_handoff.get("binding")
        binding = binding if isinstance(binding, Mapping) else {}
        volume = preflight.get("volume")
        volume = volume if isinstance(volume, Mapping) else {}
        if str(binding.get("volume_id") or "") != str(volume.get("id") or ""):
            acceptance = {
                "status": "blocked",
                "blockers": ["paid_provider_lane_handoff_volume_mismatch"],
            }
        else:
            acceptance = accept_paid_provider_lane_lease_handoff(
                lane_handoff,
                canary_watchdog=spend,
                expected_binding=binding,
            )
        write_json(
            Path(adapter_output).resolve().parent
            / "provider_lane_handoff_acceptance.json",
            acceptance,
        )
        if acceptance.get("status") != "accepted":
            _settle_zero_budget(
                budget_context, outcome="handoff_rejected_before_provider_mutation"
            )
            prepared = {
                **prepared,
                "status": "blocked",
                "blockers": sorted(
                    set(
                        [
                            *prepared.get("blockers", []),
                            *acceptance.get("blockers", []),
                            "paid_provider_lane_handoff_not_accepted",
                        ]
                    )
                ),
                "provider_mutations_performed": 0,
            }
            write_json(Path(admission_out), prepared)
            return prepared
    try:
        require_paid_resource_admission(
            prepared["admission"],
            resource_class="gpu_canary",
            expected_schema_version=SERVE_SCHEMA_VERSION,
        )
        adapter_admission = build_paid_lane_admission(
            resource_class="runpod_provider_adapter",
            blockers=list(prepared.get("blockers") or []),
        )
        adapter_grant = require_paid_resource_admission(
            adapter_admission,
            resource_class="runpod_provider_adapter",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
        write_json(Path(bound_request_out), prepared["bound_request"])
    except Exception as exc:  # noqa: BLE001 - provider call has not occurred
        recovery: dict[str, Any] = {"status": "not_required_dry_run"}
        if execute:
            recovery = _recover_accepted_handoff_before_provider_mutation(
                acceptance=acceptance,
                budget_context=budget_context,
                watchdog_out_dir=watchdog_out_dir,
                pod_name_prefix=str(spend.get("watchdog_pod_name_prefix") or ""),
                outcome="local_pre_provider_failure_no_mutation",
            )
        blocked = {
            **prepared,
            "status": "blocked",
            "blockers": ["runpod_canary_local_pre_provider_failure"],
            "provider_mutations_performed": 0,
            "error_type": type(exc).__name__,
            "pre_provider_recovery": recovery,
        }
        write_json(Path(admission_out), blocked)
        return blocked
    pod_pending = None
    no_create_terminal = False
    no_create_control_failure = False
    if execute:
        try:
            pod_pending = open_pending_teardown(
                provider="runpod",
                lane="groot_oscar_gpu_canary",
                run_id=pod_name,
                resource_kind="compute_instance",
                resource_name=pod_name,
                job_dir=Path(adapter_output).resolve().parent,
                max_age_seconds=max(
                    300, int(spend.get("hard_ttl_seconds") or 0) + 600
                ),
            )
        except Exception as exc:  # noqa: BLE001 - no provider mutation has occurred
            recovery = _recover_accepted_handoff_before_provider_mutation(
                acceptance=acceptance,
                budget_context=budget_context,
                watchdog_out_dir=watchdog_out_dir,
                pod_name_prefix=str(spend.get("watchdog_pod_name_prefix") or ""),
                outcome="pending_teardown_open_failed_no_mutation",
            )
            blocked = {
                **prepared,
                "status": "blocked",
                "blockers": ["runpod_canary_pending_teardown_open_failed"],
                "provider_mutations_performed": 0,
                "error_type": type(exc).__name__,
                "pre_provider_recovery": recovery,
            }
            write_json(Path(admission_out), blocked)
            return blocked
        if watchdog_out_dir:
            receipt_path = Path(watchdog_out_dir) / "provider_lane_handoff_receipt.json"
            try:
                _write_private_json(
                    receipt_path,
                    {
                        **acceptance,
                        "pod_pending_teardown_record": pod_pending["path"],
                        "pod_id": None,
                        "pod_name_prefix": spend.get("watchdog_pod_name_prefix"),
                        "campaign_budget": budget_context,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - still before provider call
                recovery = _recover_accepted_handoff_before_provider_mutation(
                    acceptance=acceptance,
                    budget_context=budget_context,
                    watchdog_out_dir=watchdog_out_dir,
                    pod_name_prefix=str(
                        spend.get("watchdog_pod_name_prefix") or ""
                    ),
                    outcome="receipt_write_failed_no_mutation",
                    pending_path=str(pod_pending["path"]),
                )
                blocked = {
                    **prepared,
                    "status": "blocked",
                    "blockers": ["runpod_canary_handoff_receipt_write_failed"],
                    "provider_mutations_performed": 0,
                    "error_type": type(exc).__name__,
                    "pre_provider_recovery": recovery,
                }
                write_json(Path(admission_out), blocked)
                return blocked
    previous_signed_put_url = os.environ.get(RUNTIME_MANIFEST_SIGNED_PUT_URL_ENV)
    os.environ[RUNTIME_MANIFEST_SIGNED_PUT_URL_ENV] = signed_put_url
    try:
        adapter = run_runpod_provider_adapter(
            provider_launch_request_path=bound_request_out,
            output_path=adapter_output,
            mode=(
                RUNPOD_STRICT_POLICY_SMOKE_MODE
                if probe_kind == STRICT_POLICY_SMOKE_PROBE_KIND
                else RUNPOD_IMAGE_STARTUP_CANARY_MODE
            ),
            allow_runpod_api_call=execute,
            pod_name=pod_name,
            gpu_type_id=prepared["admission"]["gpu_type_id"],
            paid_resource_admission_grant=adapter_grant,
        )
    except Exception as exc:  # noqa: BLE001 - create outcome can be ambiguous
        if pod_pending is not None:
            mark_pending_teardown_ambiguous(
                pod_pending["path"],
                reason="runpod_canary_adapter_raised_after_create_boundary",
                evidence={"error_type": type(exc).__name__},
            )
        immediate_cleanup = (
            terminate_canary_resources(
                provider=provider,
                pod_name_prefix=pod_name,
                armed={"status": "armed", "pod_name_prefix": pod_name},
            )
            if execute and provider is not None
            else {
                "status": "not_attempted_dry_run",
                "provider_absence_confirmed": False,
                "provider_mutations_performed": 0,
            }
        )
        failed = {
            "status": "failed",
            "blockers": ["runpod_canary_adapter_failed_or_ambiguous"],
            "provider_allocation_ambiguous": True,
            "provider_mutations_performed": immediate_cleanup.get(
                "provider_mutations_performed", 0
            ),
            "error_type": type(exc).__name__,
            "immediate_cleanup": immediate_cleanup,
        }
        write_json(Path(adapter_output), failed)
        return failed
    finally:
        if previous_signed_put_url is None:
            os.environ.pop(RUNTIME_MANIFEST_SIGNED_PUT_URL_ENV, None)
        else:
            os.environ[RUNTIME_MANIFEST_SIGNED_PUT_URL_ENV] = previous_signed_put_url
    if execute and pod_pending is not None:
        response = adapter.get("runpod_response")
        response = response if isinstance(response, Mapping) else {}
        pod_id = str(response.get("id") or "").strip()
        if pod_id:
            bind_pending_teardown_instance(pod_pending["path"], pod_id)
        elif adapter.get("status") in {"blocked", "dry_run"}:
            cancel_result = cancel_pending_teardown(
                pod_pending["path"],
                reason="runpod_canary_adapter_confirmed_no_create",
                evidence={"adapter_status": adapter.get("status")},
            )
            restore_result = restore_paid_provider_lane_lease_to_retained_watchdog(
                acceptance
            )
            settlement = _settle_zero_budget(
                budget_context, outcome="adapter_confirmed_no_create"
            )
            no_create_terminal = bool(
                cancel_result.get("status") == "cancelled_no_allocation"
                and restore_result.get("status") == "restored"
                and settlement.get("status") == "settled"
            )
            no_create_control_failure = not no_create_terminal
        else:
            mark_pending_teardown_ambiguous(
                pod_pending["path"],
                reason="runpod_canary_create_result_missing_pod_id",
                evidence={"adapter_status": adapter.get("status")},
            )
        if watchdog_out_dir:
            receipt_path = Path(watchdog_out_dir) / "provider_lane_handoff_receipt.json"
            if no_create_terminal:
                receipt_path.unlink(missing_ok=True)
                return dict(adapter)
            try:
                _write_private_json(
                    receipt_path,
                    {
                        **acceptance,
                        "pod_pending_teardown_record": pod_pending["path"],
                        "pod_id": pod_id or None,
                        "pod_name_prefix": spend.get("watchdog_pod_name_prefix"),
                        "campaign_budget": budget_context,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - pre-receipt remains authoritative
                failed = {
                    "status": "failed",
                    "blockers": ["runpod_canary_post_create_receipt_update_failed"],
                    "provider_allocation_ambiguous": True,
                    "provider_mutations_performed": 1,
                    "error_type": type(exc).__name__,
                }
                write_json(Path(adapter_output), failed)
                return failed
        if no_create_control_failure:
            failed = {
                "status": "failed",
                "blockers": ["runpod_canary_no_create_control_plane_open"],
                "provider_mutations_performed": 0,
            }
            write_json(Path(adapter_output), failed)
            return failed
    if execute and adapter.get("status") == "submitted":
        return _finalize_adapter_allocation(
            adapter=adapter,
            adapter_output=adapter_output,
            pod_name=pod_name,
            release_image_ref=prepared["admission"]["release_image_ref"],
        )
    return dict(adapter)


def main(argv: Sequence[str] | None = None) -> int:
    """Hard-disable the legacy mutation entrypoint.

    Imports remain for compatibility and tests, but allocation is only exposed
    by ``blueprint-allocate-gpu-canary``.
    """

    del argv
    print("legacy_gpu_canary_launcher_disabled:use_blueprint-allocate-gpu-canary")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
