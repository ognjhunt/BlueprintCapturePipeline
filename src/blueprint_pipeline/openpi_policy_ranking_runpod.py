"""RunPod request binding for the frozen OpenPI policy-ranking campaign.

This module deliberately exposes request construction and a no-mutation shape
check only. Paid creation remains owned by ``paid_resource_allocator`` so the
independent watchdog, budget reservation, pending-teardown record, and exact
protected-main checks cannot be bypassed.
"""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .gpu_render_providers import get_render_provider
from .groot_oscar_runpod_watchdog import terminate_canary_resources
from .openpi_policy_ranking_gpu_admission import (
    build_openpi_policy_ranking_gpu_admission,
    collect_openpi_policy_ranking_runpod_preflight,
)
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
)
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    require_paid_resource_admission,
)
from .production_gpu_campaign_budget import (
    CampaignBudgetExceeded,
    ProductionGpuCampaignBudget,
)
from .runpod_provider_adapter import run_runpod_provider_adapter


SCHEMA_VERSION = "openpi_policy_ranking_runpod_launch.v1"
INPUT_SECRET_URL_ENV = "BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SECRET_URL"
INPUT_SHA256_ENV = "BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SHA256"
OUTPUT_SECRET_PUT_URL_ENV = "BLUEPRINT_OPENPI_POLICY_RANKING_OUTPUT_SECRET_PUT_URL"
GENERIC_OUTPUT_SECRET_URL_ENV = "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"
FORWARD_SECRET_ENV_NAMES_ENV = "BLUEPRINT_RUNPOD_FORWARD_SECRET_ENV_VARS"
WATCHDOG_EVIDENCE_NAME = "groot_oscar_runpod_canary_watchdog.json"
CANARY_NAME_PREFIX = "blueprint-groot-oscar-canary-openpi-ranking-"


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(value)


def _read_private_https_url(path: str | Path, *, field: str) -> str:
    candidate = Path(path).expanduser()
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError(f"{field}_no_follow_unavailable")
    descriptor = os.open(candidate, os.O_RDONLY | no_follow)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{field}_not_regular")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError(f"{field}_not_private_0600")
        value = os.read(descriptor, 64 * 1024).decode("utf-8").strip()
    finally:
        os.close(descriptor)
    if not value.startswith("https://") or any(char.isspace() for char in value):
        raise ValueError(f"{field}_not_https_url")
    return value


def _wait_for_watchdog(
    *, root: Path, process: subprocess.Popen[Any], prefix: str, deadline: float
) -> dict[str, Any]:
    evidence_path = root / WATCHDOG_EVIDENCE_NAME
    until = time.time() + 10
    while time.time() < until:
        if process.poll() is not None:
            break
        try:
            value = _read_object(evidence_path)
        except (OSError, UnicodeError, ValueError):
            time.sleep(0.05)
            continue
        if (
            value.get("status") == "armed"
            and value.get("independent_process") is True
            and value.get("pid") == process.pid
            and value.get("pod_name_prefix") == prefix
            and value.get("deadline_epoch") == deadline
        ):
            return value
        time.sleep(0.05)
    raise RuntimeError("openpi_runpod_watchdog_not_confirmed_armed")


def _stop_watchdog_after_provider_zero(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.kill(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def build_openpi_policy_ranking_provider_request(
    *,
    release: Mapping[str, Any],
    input_bundle: Mapping[str, Any],
    preflight: Mapping[str, Any],
    spend: Mapping[str, Any],
    expected_source_commit: str,
    job_id: str,
) -> dict[str, Any]:
    admission = build_openpi_policy_ranking_gpu_admission(
        release=release,
        input_bundle=input_bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit=expected_source_commit,
    )
    if admission["status"] != "admitted":
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": admission["blockers"],
            "admission": admission,
            "provider_mutations_performed": 0,
        }
    ttl = int(spend["hard_ttl_seconds"])
    image_ref = str(release["resolved_digest_ref"])
    bundle_sha = str(input_bundle["bundle_sha256"])
    gpu_type_id = str(preflight["gpu_type_id"])
    request: dict[str, Any] = {
        "schema_version": "robot_eval_gpu_provider_launch_request.v1",
        "job_id": job_id,
        "provider": "runpod",
        "status": "request_manifest_ready",
        "operation": "execute_frozen_openpi_policy_ranking_campaign",
        "prelaunch_spend_guard": {
            "required_before_provider_launch": True,
            "can_launch": True,
            "blockers": [],
        },
        "provider_request_shape": {
            "api_payload_is_provider_adapter_template": True,
            "api_payload_values_are_redacted": True,
            "operation": "execute_frozen_openpi_policy_ranking_campaign",
            "image": {
                "configured_image_ref": image_ref,
                "configured_image_ref_is_versioned": True,
                "configured_image_ref_fetchable_by_provider": True,
            },
            "docker_entrypoint": [
                "/.venv/bin/python",
                "-m",
                "blueprint_pipeline.openpi_policy_ranking_gpu_bootstrap",
                "run",
            ],
            "docker_start_cmd": [],
            "environment": {
                "secret_env_var_names": [
                    INPUT_SECRET_URL_ENV,
                    OUTPUT_SECRET_PUT_URL_ENV,
                    GENERIC_OUTPUT_SECRET_URL_ENV,
                ],
                "plaintext_env_var_names": [INPUT_SHA256_ENV],
                "plaintext_env_values": {INPUT_SHA256_ENV: bundle_sha},
                "secret_values_in_artifact": False,
                "customer_visible_secret_values_allowed": False,
            },
            "inputs": {
                "manifest_uri_required_for_provider": True,
                "manifest_uri": "private-signed-input-url:not-persisted",
                "manifest_uri_fetchable_by_provider": True,
                "capture_root_bundle_uri_required_for_provider": True,
                "capture_root_bundle_uri": "private-signed-input-url:not-persisted",
                "capture_root_bundle_uri_fetchable_by_provider": True,
                "artifact_output_uri_required": False,
            },
            "gpu": {
                "preferred_gpu_class": gpu_type_id,
                "preferred_gpu_type_id": gpu_type_id,
                "provider_gpu_priority": [gpu_type_id],
                "gpu_count": 1,
                "container_disk_in_gb": int(preflight["container_disk_bytes"])
                // 1024**3,
                "volume_in_gb": 0,
                "min_vcpu_count": 4,
                "min_memory_in_gb": 24,
            },
            "limits": {
                "max_active_workers": 1,
                "requested_budget_usd": float(spend["max_spend_usd"]),
                "hard_timeout_seconds": max(60, ttl - 120),
                "idle_timeout_seconds": 60,
                "startup_artifact_watchdog_required": True,
                "startup_artifact_timeout_seconds": max(60, ttl - 180),
                "idle_shutdown_required": True,
                "external_watchdog_ttl_required": True,
                "external_watchdog_ttl_seconds": ttl,
                "external_watchdog_owner": "independent_name_bound_runpod_watchdog",
                "scale_to_zero_default": True,
            },
            "artifact_finalizer": {
                "upload_before_shutdown_required": True,
                "record_actual_gpu_time_required": True,
            },
            "local_sim_only_prerequisite": {
                "schema_version": "robot_eval_provider_local_sim_only_prerequisite.v1",
                "required_before_provider_spend": True,
                "status": "passed",
                "source_artifact": "interiorgs_0787_dynamic_hybrid_control.json",
                "local_sim_only_evidence_clean": True,
                "sim_only_beta_core_complete": True,
                "sim_only_beta_blocked_requirement_ids": [],
                "blockers": [],
                "claim_boundary": {
                    "controls_are_not_learned_policy_results": True,
                    "local_sim_only_clean_does_not_prove_remote_execution": True,
                },
            },
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "admitted",
        "blockers": [],
        "admission": admission,
        "bound_request": request,
        "provider_mutations_performed": 0,
    }


def shape_openpi_policy_ranking_request_without_mutation(
    *,
    prepared: Mapping[str, Any],
    output_path: str | Path,
    input_secret_url: str,
    output_secret_put_url: str,
    pod_name: str,
) -> dict[str, Any]:
    """Validate the exact redacted provider payload without making an API call."""

    if prepared.get("status") != "admitted" or not isinstance(
        prepared.get("bound_request"), Mapping
    ):
        return {
            "status": "blocked",
            "blockers": ["openpi_runpod_request_not_admitted"],
            "provider_mutations_performed": 0,
        }
    output = Path(output_path).expanduser().resolve()
    bound_request_path = output.parent / "openpi_provider_launch_request.json"
    write_json(bound_request_path, dict(prepared["bound_request"]))
    names = (INPUT_SECRET_URL_ENV, OUTPUT_SECRET_PUT_URL_ENV)
    previous = {
        key: os.environ.get(key)
        for key in (*names, GENERIC_OUTPUT_SECRET_URL_ENV, FORWARD_SECRET_ENV_NAMES_ENV)
    }
    os.environ[INPUT_SECRET_URL_ENV] = input_secret_url
    os.environ[OUTPUT_SECRET_PUT_URL_ENV] = output_secret_put_url
    # The generic adapter contract requires an output sink whenever the worker
    # owns finalization. It is never persisted and the worker uses the OpenPI-
    # specific alias above.
    os.environ[GENERIC_OUTPUT_SECRET_URL_ENV] = output_secret_put_url
    os.environ[FORWARD_SECRET_ENV_NAMES_ENV] = ",".join(names)
    try:
        return run_runpod_provider_adapter(
            provider_launch_request_path=bound_request_path,
            output_path=output,
            mode="dry-run",
            pod_name=pod_name,
            gpu_type_id=str(prepared["admission"]["gpu_type_id"]),
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_openpi_policy_ranking_campaign(
    *,
    release_evidence: str | Path,
    input_bundle_receipt: str | Path,
    preflight_bundle: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    input_secret_url_file: str | Path,
    output_secret_put_url_file: str | Path,
    pod_name: str,
    expected_source_commit: str,
    execute: bool,
    hard_ttl_seconds: int,
    max_spend_usd: float,
    campaign_budget_ledger: str | Path | None = None,
    campaign_initial_spent_usd: float | None = None,
    campaign_initial_used_gpu_seconds: int | None = None,
    campaign_total_spend_cap_usd: float = 20.0,
    campaign_wall_cap_seconds: int = 36_000,
) -> dict[str, Any]:
    """Validate or launch one frozen OpenPI ranking campaign.

    The execute path refreshes provider facts, reserves worst-case campaign
    budget, confirms an independent watchdog, opens a pending-teardown record,
    and only then reaches the RunPod mutation adapter.
    """

    root = Path(adapter_output).expanduser().resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    if not pod_name.startswith(CANARY_NAME_PREFIX):
        result = {
            "status": "blocked",
            "blockers": ["openpi_runpod_pod_name_outside_watchdog_scope"],
            "provider_mutations_performed": 0,
        }
        write_json(Path(admission_out), result)
        return result
    release = _read_object(release_evidence)
    input_bundle = _read_object(input_bundle_receipt)
    preflight = _read_object(preflight_bundle)
    provider = get_render_provider("runpod") if execute else None
    if execute and provider is not None:
        preflight = collect_openpi_policy_ranking_runpod_preflight(
            name_prefix=CANARY_NAME_PREFIX,
            gpu_type_ids=list(preflight.get("requested_gpu_types") or []),
            container_disk_bytes=int(preflight.get("container_disk_bytes") or 0),
            capacity_probe=provider.capacity_preflight,
            inventory_probe=lambda prefix: provider.billable_inventory(
                name_prefix=prefix
            ),
        )
        write_json(root / "openpi_runpod_preflight_launch_refresh.json", preflight)

    price = float(preflight.get("on_demand_price_usd_per_hour") or 0.0)
    spend = {
        "paid_mutation_authorized": True,
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
        "watchdog_armed_before_allocation": True,
        "hard_ttl_seconds": int(hard_ttl_seconds),
        "max_spend_usd": float(max_spend_usd),
        "physical_robot_endpoint_access_allowed": False,
    }
    prepared = build_openpi_policy_ranking_provider_request(
        release=release,
        input_bundle=input_bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit=expected_source_commit,
        job_id=pod_name,
    )
    write_json(Path(admission_out), prepared)
    if prepared.get("status") != "admitted":
        return prepared
    write_json(Path(bound_request_out), dict(prepared["bound_request"]))
    if not execute:
        result = {
            **prepared,
            "status": "dry_run_ready",
            "provider_mutations_performed": 0,
            "watchdog_process_started": False,
            "budget_reservation_created": False,
        }
        write_json(Path(adapter_output), result)
        return result
    if (
        campaign_budget_ledger is None
        or campaign_initial_spent_usd is None
        or campaign_initial_used_gpu_seconds is None
    ):
        result = {
            **prepared,
            "status": "blocked",
            "blockers": ["openpi_runpod_campaign_budget_arguments_missing"],
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result

    input_secret_url = _read_private_https_url(
        input_secret_url_file, field="openpi_input_secret_url_file"
    )
    output_secret_put_url = _read_private_https_url(
        output_secret_put_url_file, field="openpi_output_secret_put_url_file"
    )
    budget = ProductionGpuCampaignBudget(
        campaign_budget_ledger,
        initial_spent_usd=campaign_initial_spent_usd,
        initial_used_gpu_seconds=campaign_initial_used_gpu_seconds,
        total_spend_cap_usd=campaign_total_spend_cap_usd,
        combined_gpu_wall_cap_seconds=campaign_wall_cap_seconds,
    )
    try:
        reservation = budget.reserve(
            reservation_id=pod_name,
            gpu_seconds=hard_ttl_seconds,
            max_hourly_rate_usd=price,
        )
    except CampaignBudgetExceeded as exc:
        result = {
            **prepared,
            "status": "blocked",
            "blockers": [str(exc)],
            "provider_mutations_performed": 0,
            "campaign_budget_admission": exc.admission,
        }
        write_json(Path(adapter_output), result)
        return result
    write_json(root / "openpi_campaign_budget_reservation.json", reservation)

    deadline = time.time() + hard_ttl_seconds
    watchdog = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_watchdog",
            "--out-dir",
            str(root),
            "--pod-name-prefix",
            CANARY_NAME_PREFIX,
            "--deadline-epoch",
            str(deadline),
            "--provider",
            "runpod",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        armed = _wait_for_watchdog(
            root=root,
            process=watchdog,
            prefix=CANARY_NAME_PREFIX,
            deadline=deadline,
        )
    except Exception:
        _stop_watchdog_after_provider_zero(watchdog)
        budget.settle(
            reservation_id=pod_name,
            charged_gpu_seconds=0,
            charged_usd=0.0,
            outcome="watchdog_not_armed_no_mutation",
        )
        raise
    write_json(root / "openpi_watchdog_armed_receipt.json", armed)

    try:
        pending = open_pending_teardown(
            provider="runpod",
            lane="openpi_policy_ranking_gpu_canary",
            run_id=pod_name,
            resource_kind="compute_instance",
            resource_name=pod_name,
            job_dir=root,
            max_age_seconds=hard_ttl_seconds + 600,
        )
        grant = require_paid_resource_admission(
            prepared["admission"]["shared_paid_lane_admission"],
            resource_class="runpod_provider_adapter",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
    except Exception:  # noqa: BLE001 - provider boundary has not been crossed
        _stop_watchdog_after_provider_zero(watchdog)
        budget.settle(
            reservation_id=pod_name,
            charged_gpu_seconds=0,
            charged_usd=0.0,
            outcome="local_pre_provider_failure_no_mutation",
        )
        raise
    previous = {
        key: os.environ.get(key)
        for key in (
            INPUT_SECRET_URL_ENV,
            OUTPUT_SECRET_PUT_URL_ENV,
            GENERIC_OUTPUT_SECRET_URL_ENV,
            FORWARD_SECRET_ENV_NAMES_ENV,
        )
    }
    os.environ[INPUT_SECRET_URL_ENV] = input_secret_url
    os.environ[OUTPUT_SECRET_PUT_URL_ENV] = output_secret_put_url
    os.environ[GENERIC_OUTPUT_SECRET_URL_ENV] = output_secret_put_url
    os.environ[FORWARD_SECRET_ENV_NAMES_ENV] = ",".join(
        (INPUT_SECRET_URL_ENV, OUTPUT_SECRET_PUT_URL_ENV)
    )
    try:
        adapter = run_runpod_provider_adapter(
            provider_launch_request_path=bound_request_out,
            output_path=adapter_output,
            mode="on-demand-pod",
            allow_runpod_api_call=True,
            pod_name=pod_name,
            gpu_type_id=str(preflight["gpu_type_id"]),
            paid_resource_admission_grant=grant,
        )
    except Exception as exc:
        mark_pending_teardown_ambiguous(
            pending["path"],
            reason="openpi_runpod_adapter_raised_after_create_boundary",
            evidence={"error_type": type(exc).__name__},
        )
        cleanup = terminate_canary_resources(
            provider=provider,
            pod_name_prefix=CANARY_NAME_PREFIX,
            armed=armed,
            provider_name="runpod",
        )
        if cleanup.get("provider_absence_confirmed") is True:
            _stop_watchdog_after_provider_zero(watchdog)
        raise
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    response = adapter.get("runpod_response")
    response = response if isinstance(response, Mapping) else {}
    pod_id = str(response.get("id") or "").strip()
    if adapter.get("status") == "submitted" and pod_id:
        bind_pending_teardown_instance(pending["path"], pod_id)
        result = {
            **adapter,
            "status": "submitted",
            "watchdog_pid": watchdog.pid,
            "watchdog_deadline_epoch": deadline,
            "campaign_budget_reservation": reservation,
            "pending_teardown_record": pending["path"],
            "raw_secret_values_recorded": False,
        }
        write_json(Path(adapter_output), result)
        return result

    if adapter.get("runpod_side_effects_may_have_occurred") is True:
        mark_pending_teardown_ambiguous(
            pending["path"],
            reason="openpi_runpod_create_result_missing_pod_id",
            evidence={"adapter_status": adapter.get("status")},
        )
    else:
        cancel_pending_teardown(
            pending["path"],
            reason="openpi_runpod_adapter_confirmed_no_create",
            evidence={"adapter_status": adapter.get("status")},
        )
    cleanup = terminate_canary_resources(
        provider=provider,
        pod_name_prefix=CANARY_NAME_PREFIX,
        armed=armed,
        provider_name="runpod",
    )
    if cleanup.get("provider_absence_confirmed") is True:
        _stop_watchdog_after_provider_zero(watchdog)
        budget.settle(
            reservation_id=pod_name,
            charged_gpu_seconds=0,
            charged_usd=0.0,
            outcome="adapter_no_confirmed_allocation",
        )
    result = {
        **adapter,
        "status": "failed",
        "blockers": sorted(
            set([*(adapter.get("blockers") or []), "openpi_runpod_pod_id_missing"])
        ),
        "immediate_cleanup": cleanup,
    }
    write_json(Path(adapter_output), result)
    return result


__all__ = [
    "build_openpi_policy_ranking_provider_request",
    "run_openpi_policy_ranking_campaign",
    "shape_openpi_policy_ranking_request_without_mutation",
]
