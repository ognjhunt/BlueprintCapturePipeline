"""DigitalOcean launcher for the sealed GR00T+OSCAR closed-loop image.

Provider startup, semantic success, consistency, and teardown remain distinct
proof rows; a structurally completed worker result closes none by itself.
"""
from __future__ import annotations

import json
import shlex
import time
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from .common import ensure_dir, utc_now_iso, write_json
from .groot_oscar_digitalocean_job_inputs import (
    _argv_value,
    _b64_bytes,
    _b64_text,
    _episode_length_contract,
    _int_or_none,
    _json_b64,
    _mapping,
    _read_json_mapping,
    _read_json_mapping_if_present,
    _string,
    _write_input_bundle,
    _write_job_manifest,
    build_digitalocean_job_parser,
    runtime_contract_for_pre_spend,
)
from .gpu_render_providers import (
    DEFAULT_DO_GPU_SIZE,
    DO_GPU_SIZE_VRAM_MB,
    RenderLaunchSpec,
    _do_size_candidates,
    _filter_do_size_candidates_by_gpu_ram,
    get_render_provider,
)
from .g1_kitchen_pre_allocation_identity import enforce_current_checkout_pre_allocation_identity
from .g1_kitchen_pre_allocation_identity import revalidate_attempt_artifact_bytes
from .g1_kitchen_digitalocean_closure import (
    finalize_digitalocean_attempt_closure,
    teardown_proof_from_digitalocean_watch,
)
from .groot_oscar_closed_loop_image import (
    DEFAULT_MIN_TASK_ADAPTIVE_STEPS,
    DEFAULT_WAM_SUCCESS_LABEL_TIMEOUT_SECONDS,
    IMAGE_REF_ENV,
    SEALED_CONFIRMED_ENV,
    build_sealed_launch_plan,
    sealed_image_contract,
)
from .groot_oscar_worker_startup_script import (
    GEAR_SONIC_READY_SCRIPT,
    STARTUP_GATES_SCRIPT,
)
from .isaac_particlefield_render_job import stage_bundle, watch_and_collect
from .lane_hardware_requirements import build_lane_hardware_contract
from .paid_lane_guard import (
    PreSpendPreflightBlocked,
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    image_contract_from_ref,
    open_pending_teardown,
    require_pre_spend_preflight,
)
SCHEMA_VERSION = "groot_oscar_digitalocean_closed_loop_job.v1"
LANE = "kitchen_g1_groot_oscar_closed_loop"
JOB_MANIFEST_FILENAME = "groot_oscar_digitalocean_closed_loop_job_manifest.json"
PAID_RESUME_COMMAND_FILENAME = "paid_launch_resume_command.json"
MATERIALIZED_PAID_COMMAND_FILENAME = "paid_launch_command_materialized.json"
PREPARED_READINESS_AUDIT_FILENAME = "prepared_readiness_audit.json"
OBJECTIVE_READINESS_AUDIT_FILENAME = "kitchen_dishwasher_full_pipeline_readiness_audit.json"
DIGITALOCEAN_CAPACITY_PROBE_FILENAME = "digitalocean_capacity_probe.json"
DIGITALOCEAN_CAPACITY_WAIT_FILENAME = "digitalocean_capacity_wait.json"
DEFAULT_PROVIDER = "digitalocean"
DEFAULT_CONFIGURED_WAM_CONSISTENCY_COMMAND: str | None = "python -m blueprint_pipeline.wam_strict_action_consistency_scorer_client"
DEFAULT_MAX_HOURLY_RATE_USD = 3.5
DEFAULT_CONTAINER_DISK_GB = 220
DEFAULT_VOLUME_GB = 120
DEFAULT_MAX_SECONDS = 7200
DEFAULT_EPISODE_MAX_STEPS = 48
DEFAULT_MIN_COHERENT_HORIZON_FRAMES = 2
DEFAULT_MIN_TASK_COMPLETION_STEPS = DEFAULT_MIN_TASK_ADAPTIVE_STEPS
DEFAULT_MIN_GPU_RAM_MB = 48000
WORKER_PROGRESS_STALL_PHASES = ("container_bash_started", "inputs_ready", "healthcheck_passed", "groot_server_ready", "isaac_task_executor_ready")

def _paid_resume_command_payload(
    *,
    start_frame: Path,
    route_file: Path,
    task_prompt: str,
    out_dir: Path,
    image_ref: str,
    steps: int,
    oscar_height: int,
    oscar_width: int,
    min_coherent_horizon_frames: int,
    min_task_completion_steps: int,
    max_spend_usd: float | None,
    max_seconds: int,
    max_hourly_rate_usd: float,
    container_disk_gb: int,
    volume_gb: int,
    seed_provenance_file: str | Path | None,
    key_prefix: str,
    wam_consistency_command: str | None = None,
    require_generated_video_success_label: bool = False,
    wam_success_label_command: str | None = None,
    allow_wam_success_labeling: bool = False,
    wam_success_label_timeout_seconds: float
    | None = DEFAULT_WAM_SUCCESS_LABEL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    budget_arg = (
        str(float(max_spend_usd))
        if max_spend_usd is not None
        else "<MAX_SPEND_USD_REQUIRED>"
    )
    argv = [
        "python",
        "-m",
        "blueprint_pipeline.groot_oscar_digitalocean_closed_loop_job",
        "--start-frame",
        str(start_frame),
        "--route-file",
        str(route_file),
        "--task-prompt",
        task_prompt,
        "--out-dir",
        str(out_dir),
        "--steps",
        str(int(steps)),
        "--oscar-height",
        str(int(oscar_height)),
        "--oscar-width",
        str(int(oscar_width)),
        "--min-coherent-horizon-frames",
        str(int(min_coherent_horizon_frames)),
        "--min-steps",
        str(int(min_task_completion_steps)),
        "--allow-paid",
        "--max-spend-usd",
        budget_arg,
        "--max-seconds",
        str(int(max_seconds)),
        "--max-hourly-rate-usd",
        str(float(max_hourly_rate_usd)),
        "--container-disk-gb",
        str(int(container_disk_gb)),
        "--volume-gb",
        str(int(volume_gb)),
        "--key-prefix",
        key_prefix,
        "--image-ref",
        image_ref,
    ]
    if seed_provenance_file:
        argv.extend(["--seed-provenance-file", str(seed_provenance_file)])
    if wam_consistency_command:
        argv.extend(["--wam-consistency-command", str(wam_consistency_command)])
    if require_generated_video_success_label:
        argv.append("--require-generated-video-success-label")
    if wam_success_label_command:
        argv.extend(["--wam-success-label-command", str(wam_success_label_command)])
    if allow_wam_success_labeling:
        argv.append("--allow-wam-success-labeling")
    if wam_success_label_timeout_seconds is not None:
        argv.extend(
            [
                "--wam-success-label-timeout-seconds",
                str(float(wam_success_label_timeout_seconds)),
            ]
        )
    return {
        "schema_version": "groot_oscar_digitalocean_closed_loop_resume_command.v1",
        "generated_at": utc_now_iso(),
        "mode": "paid_launch_template",
        "provider": DEFAULT_PROVIDER,
        "argv": argv,
        "shell_command": shlex.join(argv),
        "required_before_running": [
            "explicit_user_approval_to_query_digitalocean",
            "explicit_max_spend_usd_budget",
            "digitalocean_capacity_preflight_allowed",
        ],
        "budget_placeholder": None if max_spend_usd is not None else "<MAX_SPEND_USD_REQUIRED>",
        "will_query_digitalocean": True,
        "capacity_preflight_before_staging": True,
        "staging_after_capacity_preflight": True,
        "pending_teardown_record_required_before_launch": True,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "This is a paid-run resume template. It is not proof of provider "
            "capacity, droplet launch, task success, generated-video success, "
            "forward/inverse consistency, or teardown."
        ),
    }

def _write_paid_resume_command(
    out_dir: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    command = dict(payload)
    write_json(out_dir / PAID_RESUME_COMMAND_FILENAME, command)
    return command

def materialize_paid_resume_command(
    prepared_dir: str | Path,
    *,
    max_spend_usd: float | None,
    acknowledge_digitalocean_query_approval: bool = False,
) -> dict[str, Any]:
    """Fill the paid resume template without executing or contacting providers."""
    root = Path(prepared_dir)
    template_path = root / PAID_RESUME_COMMAND_FILENAME
    materialized_path = root / MATERIALIZED_PAID_COMMAND_FILENAME
    template = _read_json_mapping_if_present(template_path)
    blockers: list[str] = []
    if not template:
        blockers.append("paid_resume_command_missing_or_unreadable")
        argv: list[str] = []
    else:
        argv = [str(item) for item in template.get("argv") or []]
    if "--allow-paid" not in argv:
        blockers.append("template_missing_allow_paid")
    if template.get("will_query_digitalocean") is not True:
        blockers.append("template_does_not_mark_digitalocean_query")
    if template.get("capacity_preflight_before_staging") is not True:
        blockers.append("template_missing_capacity_preflight_contract")
    if not acknowledge_digitalocean_query_approval:
        blockers.append("digitalocean_query_approval_not_acknowledged")
    try:
        budget = float(max_spend_usd) if max_spend_usd is not None else None
    except (TypeError, ValueError):
        budget = None
    if budget is None:
        blockers.append("max_spend_usd_missing")
    elif budget <= 0:
        blockers.append("max_spend_usd_must_be_positive")

    max_spend_idx: int | None = None
    try:
        max_spend_idx = argv.index("--max-spend-usd") + 1
    except ValueError:
        blockers.append("template_missing_max_spend_usd_flag")
    if max_spend_idx is not None and max_spend_idx >= len(argv):
        blockers.append("template_missing_max_spend_usd_value")
    if max_spend_idx is not None and max_spend_idx < len(argv) and budget is not None:
        argv[max_spend_idx] = str(float(budget))

    status = "ready" if not blockers else "blocked"
    command = {
        "schema_version": "groot_oscar_digitalocean_materialized_paid_command.v1",
        "status": status,
        "generated_at": utc_now_iso(),
        "blockers": sorted(set(blockers)),
        "prepared_dir": str(root),
        "template_path": str(template_path),
        "max_spend_usd": budget,
        "acknowledge_digitalocean_query_approval": bool(
            acknowledge_digitalocean_query_approval
        ),
        "argv": argv,
        "shell_command": shlex.join(argv) if argv else "",
        "will_query_digitalocean_if_executed": status == "ready",
        "executes_now": False,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "This materialized command is still only a local artifact. It does "
            "not query DigitalOcean, stage object storage, launch a droplet, "
            "or prove task success until the command is explicitly executed."
        ),
    }
    write_json(materialized_path, command)
    return command


def probe_digitalocean_capacity_for_prepared_dir(
    prepared_dir: str | Path,
) -> dict[str, Any]:
    """Run and persist the read-only DigitalOcean size/region capacity probe."""
    root = Path(prepared_dir)
    manifest = _read_json_mapping_if_present(root / JOB_MANIFEST_FILENAME)
    probe_path = root / DIGITALOCEAN_CAPACITY_PROBE_FILENAME
    provider = get_render_provider(DEFAULT_PROVIDER)
    request = {
        "provider": DEFAULT_PROVIDER,
        "min_gpu_ram_mb": DEFAULT_MIN_GPU_RAM_MB,
        "max_hourly_rate_usd": DEFAULT_MAX_HOURLY_RATE_USD,
        "capacity_preflight_before_staging": True,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "Read-only DigitalOcean size/region capacity check for the prepared "
            "sealed GR00T+OSCAR run. This does not stage object storage, create "
            "a droplet, reserve capacity, or prove launch success."
        ),
    }
    capacity = provider.capacity_preflight(request)
    probe = {
        "schema_version": "groot_oscar_digitalocean_capacity_probe.v1",
        "generated_at": utc_now_iso(),
        "prepared_dir": str(root),
        "manifest_path": str(root / JOB_MANIFEST_FILENAME),
        "manifest_status": manifest.get("status"),
        "provider_available": provider.available(),
        "capacity_preflight_request": request,
        "capacity_preflight": capacity,
        "billable_provider_call": False,
        "droplet_created": False,
        "object_store_staged": False,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "This artifact proves only the outcome of a read-only capacity "
            "probe at generated_at. It is not a paid launch or a capacity "
            "reservation."
        ),
    }
    ensure_dir(probe_path.parent)
    write_json(probe_path, probe)
    return probe


def _resolve_resume_path(value: Any, prepared_dir: Path) -> Path:
    raw = _string(value)
    path = Path(raw)
    if path.is_absolute() or path.exists():
        return path
    return prepared_dir / path


def _float_or_none(value: Any) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _seed_provenance_from_resume(
    *,
    prepared_dir: Path,
    resume_argv: Sequence[Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    seed_provenance_file = _argv_value(resume_argv, "--seed-provenance-file")
    if seed_provenance_file:
        path = _resolve_resume_path(seed_provenance_file, prepared_dir)
        if path.is_file():
            return _read_json_mapping(path)
    bundle_path = _string(manifest.get("bundle_zip"))
    if bundle_path:
        bundle = _resolve_resume_path(bundle_path, prepared_dir)
        if bundle.is_file():
            try:
                with zipfile.ZipFile(bundle) as zf:
                    payload = json.loads(zf.read("seed_provenance.json"))
                return dict(payload) if isinstance(payload, Mapping) else {}
            except Exception:  # noqa: BLE001
                return {}
    return {}


def _launch_prepared_from_materialized_command(
    prepared_dir: Path,
    materialized: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute a materialized paid resume command in-process.

    The materialized command remains the operator-facing audit artifact, but the
    wait loop uses the same argv values to call the launcher directly so tests
    can prove capacity/staging/teardown ordering without shelling out.
    """
    argv = [str(item) for item in materialized.get("argv") or []]
    manifest = _read_json_mapping_if_present(prepared_dir / JOB_MANIFEST_FILENAME)
    blockers: list[str] = []
    if materialized.get("status") != "ready":
        blockers.append("materialized_paid_command_not_ready")
    if argv[:3] != [
        "python",
        "-m",
        "blueprint_pipeline.groot_oscar_digitalocean_closed_loop_job",
    ]:
        blockers.append("materialized_paid_command_module_mismatch")
    required_flags = [
        "--start-frame",
        "--route-file",
        "--task-prompt",
        "--out-dir",
        "--steps",
        "--oscar-height",
        "--oscar-width",
        "--min-coherent-horizon-frames",
        "--min-steps",
        "--max-spend-usd",
        "--max-seconds",
        "--max-hourly-rate-usd",
        "--container-disk-gb",
        "--volume-gb",
        "--key-prefix",
        "--image-ref",
    ]
    missing = [flag for flag in required_flags if _argv_value(argv, flag) is None]
    if missing:
        blockers.extend(
            f"materialized_command_missing_{flag[2:].replace('-', '_')}"
            for flag in missing
        )
    if "--allow-paid" not in argv:
        blockers.append("materialized_command_missing_allow_paid")
    if blockers:
        blocked = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": sorted(set(blockers)),
            "prepared_dir": str(prepared_dir),
            "materialized_command_path": str(
                prepared_dir / MATERIALIZED_PAID_COMMAND_FILENAME
            ),
            "raw_secret_values_recorded": False,
        }
        return _write_job_manifest(prepared_dir, blocked)
    seed_provenance = _seed_provenance_from_resume(
        prepared_dir=prepared_dir,
        resume_argv=argv,
        manifest=manifest,
    )
    seed_provenance_file = _argv_value(argv, "--seed-provenance-file")
    timeout = _float_or_none(_argv_value(argv, "--wam-success-label-timeout-seconds"))
    return run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=_resolve_resume_path(_argv_value(argv, "--start-frame"), prepared_dir),
        route_file=_resolve_resume_path(_argv_value(argv, "--route-file"), prepared_dir),
        task_prompt=str(_argv_value(argv, "--task-prompt")),
        out_dir=_resolve_resume_path(_argv_value(argv, "--out-dir"), prepared_dir),
        steps=int(_argv_value(argv, "--steps") or DEFAULT_EPISODE_MAX_STEPS),
        oscar_height=int(_argv_value(argv, "--oscar-height") or 480),
        oscar_width=int(_argv_value(argv, "--oscar-width") or 640),
        min_coherent_horizon_frames=int(
            _argv_value(argv, "--min-coherent-horizon-frames")
            or DEFAULT_MIN_COHERENT_HORIZON_FRAMES
        ),
        min_task_completion_steps=int(
            _argv_value(argv, "--min-steps") or DEFAULT_MIN_TASK_COMPLETION_STEPS
        ),
        allow_paid=True,
        max_spend_usd=float(_argv_value(argv, "--max-spend-usd") or 0.0),
        max_seconds=int(_argv_value(argv, "--max-seconds") or DEFAULT_MAX_SECONDS),
        max_hourly_rate_usd=float(
            _argv_value(argv, "--max-hourly-rate-usd") or DEFAULT_MAX_HOURLY_RATE_USD
        ),
        container_disk_gb=int(
            _argv_value(argv, "--container-disk-gb") or DEFAULT_CONTAINER_DISK_GB
        ),
        volume_gb=int(_argv_value(argv, "--volume-gb") or DEFAULT_VOLUME_GB),
        seed_provenance=seed_provenance,
        seed_provenance_file=seed_provenance_file,
        key_prefix=str(_argv_value(argv, "--key-prefix")),
        image_ref=str(_argv_value(argv, "--image-ref")),
        require_generated_video_success_label=(
            "--require-generated-video-success-label" in argv
        ),
        wam_success_label_command=_argv_value(argv, "--wam-success-label-command"),
        allow_wam_success_labeling="--allow-wam-success-labeling" in argv,
        wam_success_label_timeout_seconds=(
            timeout
            if timeout is not None
            else DEFAULT_WAM_SUCCESS_LABEL_TIMEOUT_SECONDS
        ),
    )


def wait_for_digitalocean_capacity_then_launch_prepared_dir(
    prepared_dir: str | Path,
    *,
    max_attempts: int = 1,
    poll_interval_seconds: float = 60.0,
    launch_when_available: bool = False,
    allow_paid: bool = False,
    max_spend_usd: float | None = None,
    acknowledge_digitalocean_query_approval: bool = False,
) -> dict[str, Any]:
    """Poll DigitalOcean capacity and optionally launch the prepared sealed run.

    This is the operational bridge between a prepared local bundle and the paid
    launcher. It never stages object storage or creates a droplet while capacity
    is unavailable; when capacity appears it hands off to
    ``run_groot_oscar_digitalocean_closed_loop_job``, which repeats the capacity
    preflight immediately before staging and owns teardown tracking.
    """
    root = Path(prepared_dir)
    wait_path = root / DIGITALOCEAN_CAPACITY_WAIT_FILENAME
    ensure_dir(wait_path.parent)
    attempts_count = max(0, int(max_attempts or 0))
    interval = float(poll_interval_seconds)
    blockers: list[str] = []
    if not acknowledge_digitalocean_query_approval:
        blockers.append("digitalocean_query_approval_not_acknowledged")
    if attempts_count < 1:
        blockers.append("digitalocean_capacity_wait_max_attempts_must_be_positive")
    if interval < 0:
        blockers.append("digitalocean_capacity_wait_interval_must_be_non_negative")
    materialized: dict[str, Any] = {}
    if launch_when_available:
        if not allow_paid:
            blockers.append("paid_launch_not_requested")
        materialized = materialize_paid_resume_command(
            root,
            max_spend_usd=max_spend_usd,
            acknowledge_digitalocean_query_approval=(
                acknowledge_digitalocean_query_approval
            ),
        )
        if materialized.get("status") != "ready":
            blockers.extend(materialized.get("blockers") or [])
    wait: dict[str, Any] = {
        "schema_version": "groot_oscar_digitalocean_capacity_wait.v1",
        "status": "blocked" if blockers else "waiting",
        "generated_at": utc_now_iso(),
        "prepared_dir": str(root),
        "max_attempts": attempts_count,
        "poll_interval_seconds": interval,
        "launch_when_available": bool(launch_when_available),
        "allow_paid": bool(allow_paid),
        "max_spend_usd": max_spend_usd,
        "acknowledge_digitalocean_query_approval": bool(
            acknowledge_digitalocean_query_approval
        ),
        "blockers": sorted(set(blockers)),
        "attempts": [],
        "latest_capacity_probe_path": None,
        "materialized_command_path": str(root / MATERIALIZED_PAID_COMMAND_FILENAME)
        if materialized
        else None,
        "launch_started": False,
        "object_store_staged": False,
        "droplet_created": False,
        "billable_provider_call": False,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "This wait artifact proves only DigitalOcean capacity polling and, "
            "when requested, handoff to the paid launcher. It is not a capacity "
            "reservation or task-success proof."
        ),
    }
    if blockers:
        write_json(wait_path, wait)
        return wait

    last_status = ""
    for attempt_idx in range(1, attempts_count + 1):
        probe = probe_digitalocean_capacity_for_prepared_dir(root)
        capacity = _mapping(probe.get("capacity_preflight"))
        last_status = _string(capacity.get("status"))
        attempt = {
            "attempt": attempt_idx,
            "generated_at": probe.get("generated_at"),
            "capacity_status": capacity.get("status"),
            "capacity_blockers": capacity.get("blockers"),
            "probe_path": str(root / DIGITALOCEAN_CAPACITY_PROBE_FILENAME),
            "object_store_staged": False,
            "droplet_created": False,
            "billable_provider_call": False,
        }
        wait["attempts"].append(attempt)
        wait["latest_capacity_probe_path"] = attempt["probe_path"]
        if last_status == "available":
            if not launch_when_available:
                wait["status"] = "capacity_available"
                write_json(wait_path, wait)
                return wait
            wait["status"] = "launching"
            wait["launch_started"] = True
            write_json(wait_path, wait)
            launch_manifest = _launch_prepared_from_materialized_command(
                root,
                materialized,
            )
            launch = _mapping(launch_manifest.get("launch"))
            wait["status"] = (
                "completed"
                if launch_manifest.get("status") == "completed"
                else "launch_blocked"
            )
            wait["launch_manifest_path"] = str(root / JOB_MANIFEST_FILENAME)
            wait["launch_manifest_status"] = launch_manifest.get("status")
            wait["launch_blockers"] = launch_manifest.get("blockers") or []
            wait["object_store_staged"] = bool(launch_manifest.get("staging"))
            wait["droplet_created"] = bool(launch.get("instance_id"))
            wait["billable_provider_call"] = bool(launch)
            wait["paid_launcher_repeated_capacity_preflight_before_staging"] = True
            wait["paid_launcher_uses_pending_teardown_record"] = True
            wait["paid_launcher_owns_teardown_proof"] = True
            write_json(wait_path, wait)
            return wait
        wait["status"] = "capacity_blocked"
        write_json(wait_path, wait)
        if attempt_idx < attempts_count and interval > 0:
            time.sleep(interval)
    wait["status"] = (
        "capacity_blocked"
        if last_status in {"blocked", "unknown", ""}
        else f"capacity_{last_status}"
    )
    write_json(wait_path, wait)
    return wait


def audit_prepared_closed_loop_job(prepared_dir: str | Path) -> dict[str, Any]:
    """Local-only readiness audit for a prepared sealed DigitalOcean run.

    This does not call DigitalOcean, object storage, Docker, or any model runtime.
    It proves only that the prepared directory is structurally resumable for the
    paid path and still carries the quality/dynamic-episode contracts.
    """
    root = Path(prepared_dir)
    manifest_path = root / JOB_MANIFEST_FILENAME
    resume_path = root / PAID_RESUME_COMMAND_FILENAME
    audit_path = root / PREPARED_READINESS_AUDIT_FILENAME
    manifest = _read_json_mapping_if_present(manifest_path)
    resume = _read_json_mapping_if_present(resume_path)
    blockers: list[str] = []

    if not manifest:
        blockers.append("prepared_manifest_missing_or_unreadable")
    if not resume:
        blockers.append("paid_resume_command_missing_or_unreadable")

    closure_only_blocked_live_result = bool(
        manifest.get("status") == "blocked"
        and _mapping(manifest.get("closed_loop_result_contract")).get("status") == "PASS"
        and _mapping(manifest.get("g1_kitchen_attempt_closure")).get("status") == "blocked"
    )
    if manifest and manifest.get("status") not in {"prepared", "completed"} and not closure_only_blocked_live_result:
        blockers.append("prepared_manifest_status_not_prepared_or_completed")
    plan = _mapping(manifest.get("sealed_launch_plan")) if manifest else {}
    contract = _mapping(manifest.get("sealed_image_contract")) if manifest else {}
    if contract.get("sealed_active") is not True:
        blockers.append("sealed_image_contract_not_active")
    image_ref = _string(contract.get("image_ref") or plan.get("image_ref"))
    if "@sha256:" not in image_ref:
        blockers.append("sealed_image_ref_not_digest_pinned")
    if plan.get("sealed_active") is not True:
        blockers.append("sealed_launch_plan_not_active")
    cmd = [str(item) for item in plan.get("closed_loop_command") or []]
    if "--require-fresh-learned-policy-requery" not in cmd:
        blockers.append("closed_loop_missing_fresh_policy_requery")
    if "--stop-on-task-completion" not in cmd:
        blockers.append("closed_loop_missing_stop_on_task_completion")
    oscar_height = _argv_value(cmd, "--oscar-height")
    oscar_width = _argv_value(cmd, "--oscar-width")
    min_coherent_horizon = _int_or_none(
        _argv_value(cmd, "--min-coherent-horizon-frames")
    )
    if oscar_height != "480" or oscar_width != "640":
        blockers.append("closed_loop_not_native_oscar_resolution_480x640")
    if min_coherent_horizon is None or min_coherent_horizon < 2:
        blockers.append("closed_loop_missing_generated_clip_coherence_gate")
    quality_contract = _mapping(plan.get("quality_gate_contract"))
    consistency_required = bool(
        quality_contract.get("forward_inverse_consistency_required")
    )
    consistency_command = _string(
        quality_contract.get("forward_inverse_consistency_command")
    )
    if not consistency_required:
        blockers.append("closed_loop_missing_forward_inverse_consistency_requirement")
    if not consistency_command:
        blockers.append("closed_loop_missing_wam_consistency_command")
    if "--require-forward-inverse-consistency" not in cmd:
        blockers.append("closed_loop_missing_require_forward_inverse_consistency_flag")
    if "--allow-wam-consistency-scoring" not in cmd:
        blockers.append("closed_loop_missing_allow_wam_consistency_scoring_flag")
    if not _argv_value(cmd, "--wam-consistency-command"):
        blockers.append("closed_loop_missing_wam_consistency_command_arg")
    success_label_required = bool(
        quality_contract.get("generated_video_success_label_required")
    )
    success_label_command = _string(
        quality_contract.get("generated_video_success_label_command")
    )
    plan_success_label_command_arg = _argv_value(cmd, "--wam-success-label-command")
    steps_cap = _int_or_none(_argv_value(cmd, "--steps") or manifest.get("steps"))
    if steps_cap is None or steps_cap < 12:
        blockers.append("closed_loop_steps_cap_too_short_for_task_adaptive_run")
    min_steps_before_task_completion = _int_or_none(_argv_value(cmd, "--min-steps"))
    if (
        min_steps_before_task_completion is None
        or min_steps_before_task_completion < DEFAULT_MIN_TASK_COMPLETION_STEPS
    ):
        blockers.append("closed_loop_min_steps_before_task_completion_too_low")
    oscar_num_frames_arg = _int_or_none(_argv_value(cmd, "--num-frames"))
    manifest_episode_contract = _mapping(manifest.get("episode_length_contract"))
    plan_episode_contract = _mapping(plan.get("episode_length_contract"))
    if manifest_episode_contract.get("episode_not_bound_to_oscar_clip_frames") is not True:
        blockers.append("prepared_manifest_missing_episode_length_contract")
    if plan_episode_contract.get("episode_not_bound_to_oscar_clip_frames") is not True:
        blockers.append("sealed_launch_plan_missing_episode_length_contract")
    if (
        _int_or_none(manifest_episode_contract.get("min_steps_before_task_completion"))
        or 0
    ) < DEFAULT_MIN_TASK_COMPLETION_STEPS:
        blockers.append("prepared_manifest_min_steps_before_task_completion_too_low")
    if (
        _int_or_none(plan_episode_contract.get("min_steps_before_task_completion"))
        or 0
    ) < DEFAULT_MIN_TASK_COMPLETION_STEPS:
        blockers.append("sealed_launch_plan_min_steps_before_task_completion_too_low")
    hardware = _mapping(plan.get("lane_hardware_requirements"))
    if float(hardware.get("min_vram_gb") or 0.0) < 40.0:
        blockers.append("lane_hardware_min_vram_below_40gb")
    if int(hardware.get("min_disk_gb") or 0) < 175:
        blockers.append("lane_hardware_min_disk_below_175gb")
    provider_raw_size_candidates = _do_size_candidates(DEFAULT_DO_GPU_SIZE)
    provider_allowed_size_candidates, provider_gpu_ram_policy = (
        _filter_do_size_candidates_by_gpu_ram(
            provider_raw_size_candidates,
            {"min_gpu_ram_mb": DEFAULT_MIN_GPU_RAM_MB},
        )
    )
    if not provider_allowed_size_candidates:
        blockers.append("digitalocean_provider_no_candidate_meets_min_gpu_ram")

    bundle_zip = Path(_string(manifest.get("bundle_zip"))) if manifest else Path("")
    if not bundle_zip.is_file():
        blockers.append("input_bundle_zip_missing")
        bundle_names: set[str] = set()
        bundle_seed_provenance: dict[str, Any] = {}
    else:
        try:
            with zipfile.ZipFile(bundle_zip) as zf:
                bundle_names = set(zf.namelist())
                bundle_seed_provenance = json.loads(zf.read("seed_provenance.json"))
        except Exception:  # noqa: BLE001
            bundle_names = set()
            bundle_seed_provenance = {}
            blockers.append("input_bundle_zip_unreadable")
    required_bundle_files = {
        "initial_policy_frame.png",
        "route.json",
        "task_prompt.txt",
        "sealed_launch_plan.json",
        "seed_provenance.json",
        "bundle_manifest.json",
        "task_success_contract.json",
        "attempt_input_manifest.json",
        "kitchen_asset_inventory_checksums.json",
        "kitchen/KitchenRoom.usd",
    }
    missing_bundle_files = sorted(required_bundle_files - bundle_names)
    if missing_bundle_files:
        blockers.append("input_bundle_missing_required_files")
    if not bundle_seed_provenance:
        blockers.append("seed_provenance_missing_from_bundle")
    if manifest.get("seed_provenance_present") is not True:
        blockers.append("prepared_manifest_missing_seed_provenance_flag")

    resume_argv = [str(item) for item in resume.get("argv") or []]
    if "--allow-paid" not in resume_argv:
        blockers.append("resume_command_missing_allow_paid")
    if resume.get("will_query_digitalocean") is not True:
        blockers.append("resume_command_does_not_mark_digitalocean_query")
    if resume.get("capacity_preflight_before_staging") is not True:
        blockers.append("resume_command_missing_capacity_before_staging_contract")
    if resume.get("pending_teardown_record_required_before_launch") is not True:
        blockers.append("resume_command_missing_pending_teardown_contract")
    if image_ref and _argv_value(resume_argv, "--image-ref") != image_ref:
        blockers.append("resume_command_image_ref_mismatch")
    resume_height = _argv_value(resume_argv, "--oscar-height")
    resume_width = _argv_value(resume_argv, "--oscar-width")
    resume_min_coherent_horizon = _int_or_none(
        _argv_value(resume_argv, "--min-coherent-horizon-frames")
    )
    resume_min_steps_before_task_completion = _int_or_none(
        _argv_value(resume_argv, "--min-steps")
    )
    resume_oscar_num_frames_arg = _int_or_none(_argv_value(resume_argv, "--num-frames"))
    if resume_height != "480" or resume_width != "640":
        blockers.append("resume_command_not_native_oscar_resolution_480x640")
    if resume_min_coherent_horizon is None or resume_min_coherent_horizon < 2:
        blockers.append("resume_command_missing_generated_clip_coherence_gate")
    if (
        resume_min_steps_before_task_completion is None
        or resume_min_steps_before_task_completion < DEFAULT_MIN_TASK_COMPLETION_STEPS
    ):
        blockers.append("resume_command_min_steps_before_task_completion_too_low")
    resume_success_label_command_arg = _argv_value(
        resume_argv, "--wam-success-label-command"
    )
    if success_label_required:
        if "--require-generated-video-success-label" not in cmd:
            blockers.append(
                "closed_loop_missing_require_generated_video_success_label_flag"
            )
        if not success_label_command:
            blockers.append("closed_loop_missing_wam_success_label_command")
        if not plan_success_label_command_arg:
            blockers.append("closed_loop_missing_wam_success_label_command_arg")
        if "--allow-wam-success-labeling" not in cmd:
            blockers.append("closed_loop_missing_allow_wam_success_labeling_flag")
        if "--require-generated-video-success-label" not in resume_argv:
            blockers.append(
                "resume_command_missing_require_generated_video_success_label_flag"
            )
        if resume_success_label_command_arg != success_label_command:
            blockers.append("resume_command_wam_success_label_command_mismatch")
        if "--allow-wam-success-labeling" not in resume_argv:
            blockers.append("resume_command_missing_allow_wam_success_labeling_flag")
    if resume.get("budget_placeholder") == "<MAX_SPEND_USD_REQUIRED>":
        budget_ready = False
    else:
        budget_ready = _argv_value(resume_argv, "--max-spend-usd") not in {
            None,
            "",
            "<MAX_SPEND_USD_REQUIRED>",
        }

    audit = {
        "schema_version": "groot_oscar_digitalocean_prepared_readiness_audit.v1",
        "status": "PASS" if not blockers else "FAIL",
        "prepared_dir": str(root),
        "generated_at": utc_now_iso(),
        "blockers": sorted(set(blockers)),
        "manifest_path": str(manifest_path),
        "bundle_zip": str(bundle_zip) if manifest else None,
        "paid_resume_command_path": str(resume_path),
        "sealed_image_ref": image_ref or None,
        "steps_cap": steps_cap,
        "oscar_resolution": {
            "height": oscar_height,
            "width": oscar_width,
            "native_required": True,
        },
        "generated_clip_coherence_gate": {
            "min_coherent_horizon_frames": min_coherent_horizon,
            "required_minimum": 2,
        },
        "forward_inverse_consistency_gate": {
            "required": consistency_required,
            "command": consistency_command or None,
            "require_flag_present": "--require-forward-inverse-consistency" in cmd,
            "allow_scoring_flag_present": "--allow-wam-consistency-scoring" in cmd,
            "command_arg": _argv_value(cmd, "--wam-consistency-command"),
        },
        "generated_video_success_label_gate": {
            "required": success_label_required,
            "command": success_label_command or None,
            "require_flag_present": "--require-generated-video-success-label" in cmd,
            "allow_labeling_flag_present": "--allow-wam-success-labeling" in cmd,
            "command_arg": plan_success_label_command_arg,
            "resume_require_flag_present": (
                "--require-generated-video-success-label" in resume_argv
            ),
            "resume_allow_labeling_flag_present": (
                "--allow-wam-success-labeling" in resume_argv
            ),
            "resume_command_arg": resume_success_label_command_arg,
            "claim_boundary": (
                "Generated-video semantic success is a separate review gate. "
                "Prepared readiness does not prove manipulation success."
            ),
        },
        "task_adaptive_termination": {
            "stop_on_task_completion": "--stop-on-task-completion" in cmd,
            "steps_is_safety_cap": True,
        },
        "episode_length_contract": {
            **_episode_length_contract(
                steps_cap=steps_cap,
                stop_on_task_completion="--stop-on-task-completion" in cmd,
                min_steps_before_task_completion=(
                    min_steps_before_task_completion
                    or DEFAULT_MIN_TASK_COMPLETION_STEPS
                ),
                oscar_num_frames_arg=oscar_num_frames_arg,
            ),
            "manifest_contract_present": bool(manifest_episode_contract),
            "sealed_launch_plan_contract_present": bool(plan_episode_contract),
            "resume_min_steps_before_task_completion": (
                resume_min_steps_before_task_completion
            ),
            "resume_oscar_num_frames_arg": resume_oscar_num_frames_arg,
        },
        "seed_provenance_present": bool(bundle_seed_provenance),
        "digitalocean_gpu_candidate_floor": {
            "min_gpu_ram_mb": DEFAULT_MIN_GPU_RAM_MB,
            "raw_size_candidates": provider_raw_size_candidates,
            "allowed_size_candidates": provider_allowed_size_candidates,
            "rejected_size_candidates": provider_gpu_ram_policy.get(
                "rejected_size_candidates"
            ),
            "request_scoped_filter_required": True,
            "provider_filtering_proven_locally": bool(provider_allowed_size_candidates),
        },
        "budget_ready": budget_ready,
        "digitalocean_not_queried_by_audit": True,
        "claim_boundary": (
            "Prepared readiness proves only local resumability and launch-contract "
            "shape. It is not DigitalOcean capacity, droplet launch, WAM quality, "
            "task success, generated-video success, forward/inverse consistency, "
            "or teardown proof."
        ),
    }
    write_json(audit_path, audit)
    return audit


def _requirement(
    requirement_id: str,
    status: str,
    evidence: Mapping[str, Any] | None = None,
    *,
    remaining: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": requirement_id,
        "status": status,
        "evidence": dict(evidence or {}),
    }
    if remaining:
        item["remaining"] = remaining
    return item


def _requirement_by_id(requirements: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("id")): dict(item) for item in requirements}


def _closed_loop_success_proof(manifest: Mapping[str, Any]) -> dict[str, Any]:
    run = _mapping(manifest.get("closed_loop_run"))
    runner_result = _mapping(run.get("runner_result"))
    closed_loop = _mapping(runner_result.get("closed_loop_manifest"))
    success_proof = _mapping(closed_loop.get("success_proof"))
    if success_proof:
        return success_proof
    result_contract = _mapping(manifest.get("closed_loop_result_contract"))
    return _mapping(result_contract.get("task_success_summary"))


def _semantic_task_success_passed(success_proof: Mapping[str, Any]) -> bool:
    return bool(
        success_proof.get("manipulation_success_proven") is True
        or success_proof.get("simulated_manipulation_success_shown") is True
        or success_proof.get("generated_video_success_label_passed") is True
    )


def audit_kitchen_dishwasher_objective_readiness(
    prepared_dir: str | Path,
    *,
    digitalocean_queries_allowed: bool = False,
) -> dict[str, Any]:
    """Requirement audit for the kitchen dishwasher DigitalOcean objective.

    This is intentionally local-only. It verifies that the prepared run packet
    satisfies the contracts needed for the requested full run, while keeping the
    unexecuted live DigitalOcean capacity/run/success proof clearly pending.
    """
    root = Path(prepared_dir)
    audit_path = root / OBJECTIVE_READINESS_AUDIT_FILENAME
    prepared_audit = audit_prepared_closed_loop_job(root)
    manifest_path = root / JOB_MANIFEST_FILENAME
    resume_path = root / PAID_RESUME_COMMAND_FILENAME
    materialized_path = root / MATERIALIZED_PAID_COMMAND_FILENAME
    capacity_probe_path = root / DIGITALOCEAN_CAPACITY_PROBE_FILENAME
    manifest = _read_json_mapping_if_present(manifest_path)
    resume = _read_json_mapping_if_present(resume_path)
    materialized = _read_json_mapping_if_present(materialized_path)
    capacity_probe = _read_json_mapping_if_present(capacity_probe_path)
    plan = _mapping(manifest.get("sealed_launch_plan"))
    contract = _mapping(manifest.get("sealed_image_contract"))
    cmd = [str(item) for item in plan.get("closed_loop_command") or []]
    resume_argv = [str(item) for item in resume.get("argv") or []]
    task_prompt = _string(manifest.get("task_prompt") or _argv_value(cmd, "--task-prompt"))
    prompt_lower = task_prompt.lower()
    episode_contract = _mapping(prepared_audit.get("episode_length_contract"))
    hardware = _mapping(plan.get("lane_hardware_requirements"))
    image_ref = _string(prepared_audit.get("sealed_image_ref") or contract.get("image_ref"))
    capacity = _mapping(manifest.get("provider_capacity_preflight"))
    capacity_source = "manifest" if capacity else None
    if not capacity:
        capacity = _mapping(capacity_probe.get("capacity_preflight"))
        capacity_source = "capacity_probe_artifact" if capacity else None
    launch = _mapping(manifest.get("launch"))
    closed_loop_run = _mapping(manifest.get("closed_loop_run"))
    result_contract = _mapping(manifest.get("closed_loop_result_contract"))
    success_proof = _closed_loop_success_proof(manifest)
    semantic_success_passed = _semantic_task_success_passed(success_proof)

    requirements: list[dict[str, Any]] = []
    dishwasher_task_ok = (
        "dishwasher" in prompt_lower
        and "open" in prompt_lower
        and "close" in prompt_lower
    )
    requirements.append(
        _requirement(
            "kitchen_dishwasher_open_or_close_task",
            "PASS" if dishwasher_task_ok else "FAIL",
            {
                "task_prompt": task_prompt,
                "expects_open_or_close_dishwasher": True,
            },
        )
    )
    requirements.append(
        _requirement(
            "provider_is_digitalocean",
            "PASS" if manifest.get("provider") == DEFAULT_PROVIDER else "FAIL",
            {"provider": manifest.get("provider"), "expected_provider": DEFAULT_PROVIDER},
        )
    )
    image_ok = "blueprint-groot-oscar-eval" in image_ref and "@sha256:" in image_ref
    requirements.append(
        _requirement(
            "sealed_groot_oscar_image_digest_pinned",
            "PASS" if image_ok else "FAIL",
            {"image_ref": image_ref, "digest_pinned": "@sha256:" in image_ref},
        )
    )
    requirements.append(
        _requirement(
            "native_oscar_resolution",
            "PASS"
            if _mapping(prepared_audit.get("oscar_resolution")) == {
                "height": "480",
                "width": "640",
                "native_required": True,
            }
            else "FAIL",
            _mapping(prepared_audit.get("oscar_resolution")),
        )
    )
    coherence = _mapping(prepared_audit.get("generated_clip_coherence_gate"))
    requirements.append(
        _requirement(
            "generated_clip_coherence_gate",
            "PASS"
            if int(coherence.get("min_coherent_horizon_frames") or 0) >= 2
            else "FAIL",
            coherence,
        )
    )
    forward_inverse_gate = _mapping(
        prepared_audit.get("forward_inverse_consistency_gate")
    )
    requirements.append(
        _requirement(
            "forward_inverse_consistency_gate_configured",
            "PASS"
            if (
                forward_inverse_gate.get("required") is True
                and forward_inverse_gate.get("require_flag_present") is True
                and forward_inverse_gate.get("allow_scoring_flag_present") is True
                and bool(forward_inverse_gate.get("command_arg"))
            )
            else "FAIL",
            {
                "required": forward_inverse_gate.get("required"),
                "command": forward_inverse_gate.get("command"),
                "require_flag_present": forward_inverse_gate.get(
                    "require_flag_present"
                ),
                "allow_scoring_flag_present": forward_inverse_gate.get(
                    "allow_scoring_flag_present"
                ),
                "command_arg": forward_inverse_gate.get("command_arg"),
            },
        )
    )
    requirements.append(
        _requirement(
            "episode_not_bound_to_81_frame_oscar_clip",
            "PASS"
            if (
                episode_contract.get("episode_not_bound_to_oscar_clip_frames") is True
                and episode_contract.get("oscar_num_frames_arg") is None
                and episode_contract.get("resume_oscar_num_frames_arg") is None
            )
            else "FAIL",
            {
                "episode_length_unit": episode_contract.get("episode_length_unit"),
                "steps_cap": episode_contract.get("steps_cap"),
                "oscar_num_frames_arg": episode_contract.get("oscar_num_frames_arg"),
                "resume_oscar_num_frames_arg": episode_contract.get(
                    "resume_oscar_num_frames_arg"
                ),
                "oscar_num_frames_scope": episode_contract.get("oscar_num_frames_scope"),
                "episode_not_bound_to_oscar_clip_frames": episode_contract.get(
                    "episode_not_bound_to_oscar_clip_frames"
                ),
            },
        )
    )
    requirements.append(
        _requirement(
            "task_adaptive_termination",
            "PASS"
            if (
                episode_contract.get("stop_on_task_completion") is True
                and episode_contract.get("steps_is_safety_cap") is True
                and int(episode_contract.get("min_steps_before_task_completion") or 0)
                >= DEFAULT_MIN_TASK_COMPLETION_STEPS
            )
            else "FAIL",
            {
                "stop_on_task_completion": episode_contract.get("stop_on_task_completion"),
                "steps_is_safety_cap": episode_contract.get("steps_is_safety_cap"),
                "steps_cap": episode_contract.get("steps_cap"),
                "min_steps_before_task_completion": episode_contract.get(
                    "min_steps_before_task_completion"
                ),
            },
        )
    )
    requirements.append(
        _requirement(
            "gpu_and_disk_floor",
            "PASS"
            if float(hardware.get("min_vram_gb") or 0.0) >= 40.0
            and int(hardware.get("min_disk_gb") or 0) >= 175
            else "FAIL",
            {
                "min_vram_gb": hardware.get("min_vram_gb"),
                "min_disk_gb": hardware.get("min_disk_gb"),
                "recommended_gpu_type_ids": hardware.get("recommended_gpu_type_ids"),
            },
        )
    )
    gpu_candidate_floor = _mapping(prepared_audit.get("digitalocean_gpu_candidate_floor"))
    requirements.append(
        _requirement(
            "digitalocean_request_scoped_gpu_floor",
            "PASS"
            if (
                int(gpu_candidate_floor.get("min_gpu_ram_mb") or 0)
                >= DEFAULT_MIN_GPU_RAM_MB
                and gpu_candidate_floor.get("provider_filtering_proven_locally") is True
            )
            else "FAIL",
            {
                "min_gpu_ram_mb": gpu_candidate_floor.get("min_gpu_ram_mb"),
                "raw_size_candidates": gpu_candidate_floor.get("raw_size_candidates"),
                "allowed_size_candidates": gpu_candidate_floor.get(
                    "allowed_size_candidates"
                ),
                "rejected_size_candidates": gpu_candidate_floor.get(
                    "rejected_size_candidates"
                ),
                "digitalocean_not_queried_by_local_audit": True,
            },
        )
    )
    requirements.append(
        _requirement(
            "capacity_preflight_before_staging_and_teardown_contract",
            "PASS"
            if (
                resume.get("capacity_preflight_before_staging") is True
                and resume.get("staging_after_capacity_preflight") is True
                and resume.get("pending_teardown_record_required_before_launch") is True
            )
            else "FAIL",
            {
                "capacity_preflight_before_staging": resume.get(
                    "capacity_preflight_before_staging"
                ),
                "staging_after_capacity_preflight": resume.get(
                    "staging_after_capacity_preflight"
                ),
                "pending_teardown_record_required_before_launch": resume.get(
                    "pending_teardown_record_required_before_launch"
                ),
            },
        )
    )
    requirements.append(
        _requirement(
            "prepared_bundle_and_seed_provenance",
            "PASS"
            if prepared_audit.get("status") == "PASS"
            and prepared_audit.get("seed_provenance_present") is True
            else "FAIL",
            {
                "prepared_audit_status": prepared_audit.get("status"),
                "seed_provenance_present": prepared_audit.get("seed_provenance_present"),
                "bundle_zip": prepared_audit.get("bundle_zip"),
            },
        )
    )
    requirements.append(
        _requirement(
            "paid_resume_requires_explicit_budget_and_do_approval",
            "PASS"
            if (
                "--allow-paid" in resume_argv
                and resume.get("will_query_digitalocean") is True
                and (
                    resume.get("budget_placeholder") == "<MAX_SPEND_USD_REQUIRED>"
                    or _argv_value(resume_argv, "--max-spend-usd")
                    not in {None, "", "<MAX_SPEND_USD_REQUIRED>"}
                )
            )
            else "FAIL",
            {
                "will_query_digitalocean": resume.get("will_query_digitalocean"),
                "budget_placeholder": resume.get("budget_placeholder"),
                "required_before_running": resume.get("required_before_running"),
                "materialized_command_status": materialized.get("status"),
                "materialized_command_path": str(materialized_path)
                if materialized_path.is_file()
                else None,
                "materialized_max_spend_usd": materialized.get("max_spend_usd"),
            },
        )
    )
    capacity_status = _string(capacity.get("status"))
    if capacity:
        capacity_requirement_status = "PASS" if capacity_status == "available" else "FAIL"
        capacity_remaining = None
    else:
        capacity_requirement_status = "PENDING"
        capacity_remaining = (
            "Explicitly allow DigitalOcean capacity querying and run the paid resume command."
        )
    requirements.append(
        _requirement(
            "digitalocean_capacity_checked",
            capacity_requirement_status,
            {
                "digitalocean_queries_allowed": bool(digitalocean_queries_allowed),
                "digitalocean_not_queried_by_local_audit": True,
                "capacity_source": capacity_source,
                "capacity_probe_path": str(capacity_probe_path)
                if capacity_probe_path.is_file()
                else None,
                "capacity_probe_generated_at": capacity_probe.get("generated_at"),
                "capacity_status": capacity.get("status"),
                "capacity_blockers": capacity.get("blockers"),
            },
            remaining=capacity_remaining,
        )
    )
    live_run_present = bool(launch or closed_loop_run)
    live_run_completed = (
        launch.get("status") == "launched" and closed_loop_run.get("status") == "completed"
    )
    if live_run_completed:
        live_run_status = "PASS"
        live_run_remaining = None
    elif live_run_present:
        live_run_status = "FAIL"
        live_run_remaining = "Inspect launch and worker collection blockers."
    else:
        live_run_status = "PENDING"
        live_run_remaining = "Run on a DigitalOcean GPU droplet and collect worker output."
    requirements.append(
        _requirement(
            "live_digitalocean_droplet_run_completed",
            live_run_status,
            {
                "provider_capacity_preflight_present": bool(capacity),
                "launch_present": bool(launch),
                "launch_status": launch.get("status"),
                "launch_instance_id_present": bool(launch.get("instance_id")),
                "closed_loop_run_present": bool(closed_loop_run),
                "closed_loop_run_status": closed_loop_run.get("status"),
            },
            remaining=live_run_remaining,
        )
    )
    if result_contract:
        result_contract_status = (
            "PASS" if result_contract.get("status") == "PASS" else "FAIL"
        )
        result_contract_remaining = None
    else:
        result_contract_status = "PENDING"
        result_contract_remaining = (
            "Collect a live worker result whose closed-loop result contract is PASS."
        )
    requirements.append(
        _requirement(
            "live_closed_loop_result_contract_passed",
            result_contract_status,
            {"closed_loop_result_contract": result_contract or None},
            remaining=result_contract_remaining,
        )
    )
    semantic_evaluated = bool(success_proof)
    if semantic_evaluated:
        semantic_status = "PASS"
        semantic_remaining = None
    elif closed_loop_run or result_contract:
        semantic_status = "FAIL"
        semantic_remaining = "Collected run is missing explicit success_proof fields."
    else:
        semantic_status = "PENDING"
        semantic_remaining = (
            "Inspect the live manipulation/generated-video success outputs; "
            "do not infer semantic success from local preparedness."
        )
    requirements.append(
        _requirement(
            "semantic_task_success_evaluated",
            semantic_status,
            {
                "success_proof": success_proof or None,
                "semantic_task_success_passed": semantic_success_passed,
                "generated_video_success_label_is_not_real_world_task_success": True,
            },
            remaining=semantic_remaining,
        )
    )
    attempt_closure = _mapping(manifest.get("g1_kitchen_attempt_closure"))
    live_attempt_observed = bool(closed_loop_run or result_contract or attempt_closure)
    closure_status = (
        "PASS"
        if attempt_closure.get("schema_version") == "g1_kitchen_attempt_closure.v1"
        and attempt_closure.get("status") == "completed"
        else "FAIL"
        if live_attempt_observed
        else "PENDING"
    )
    requirements.append(
        _requirement(
            "attempt_bound_g1_kitchen_closure_completed",
            closure_status,
            {
                "closure_status": attempt_closure.get("status"),
                "closure_blockers": attempt_closure.get("blockers") or [],
                "buyer_readout_projection": manifest.get("buyer_readout_projection"),
            },
            remaining=(
                None
                if closure_status == "PASS"
                else "Complete every required proof row plus API teardown and zero inventory."
            ),
        )
    )
    by_id = _requirement_by_id(requirements)
    local_requirement_ids = [
        "kitchen_dishwasher_open_or_close_task",
        "provider_is_digitalocean",
        "sealed_groot_oscar_image_digest_pinned",
        "native_oscar_resolution",
        "generated_clip_coherence_gate",
        "forward_inverse_consistency_gate_configured",
        "episode_not_bound_to_81_frame_oscar_clip",
        "task_adaptive_termination",
        "gpu_and_disk_floor",
        "digitalocean_request_scoped_gpu_floor",
        "capacity_preflight_before_staging_and_teardown_contract",
        "prepared_bundle_and_seed_provenance",
        "paid_resume_requires_explicit_budget_and_do_approval",
    ]
    failed_local = [
        req_id
        for req_id in local_requirement_ids
        if by_id.get(req_id, {}).get("status") != "PASS"
    ]
    failed_live = [
        item["id"] for item in requirements if item.get("status") == "FAIL"
    ]
    pending_live = [
        item["id"] for item in requirements if item.get("status") == "PENDING"
    ]
    if failed_local or failed_live:
        objective_status = "FAILED"
    elif pending_live:
        objective_status = "INCOMPLETE"
    elif semantic_success_passed:
        objective_status = "COMPLETE"
    else:
        objective_status = "COMPLETED_TASK_SUCCESS_NOT_PROVEN"
    audit = {
        "schema_version": "kitchen_dishwasher_full_pipeline_objective_readiness.v1",
        "objective_status": objective_status,
        "local_status": "PASS" if not failed_local else "FAIL",
        "prepared_dir": str(root),
        "generated_at": utc_now_iso(),
        "requirements": requirements,
        "failed_local_requirements": failed_local,
        "failed_live_requirements": failed_live,
        "pending_live_requirements": pending_live,
        "semantic_task_success_passed": semantic_success_passed,
        "prepared_readiness_audit_path": str(root / PREPARED_READINESS_AUDIT_FILENAME),
        "paid_resume_command_path": str(root / PAID_RESUME_COMMAND_FILENAME),
        "digitalocean_capacity_probe_path": str(capacity_probe_path)
        if capacity_probe_path.is_file()
        else None,
        "digitalocean_not_queried_by_objective_audit": True,
        "claim_boundary": (
            "This objective audit proves local readiness only. Full completion "
            "still requires DigitalOcean capacity, droplet execution, collected "
            "closed-loop artifacts, teardown proof, and semantic success review."
        ),
    }
    write_json(audit_path, audit)
    return audit


def _shell_join(argv: Sequence[Any]) -> str:
    return shlex.join([str(item) for item in argv])


def build_worker_bootstrap_script(plan: Mapping[str, Any]) -> str:
    """Container-side script run inside the sealed image.

    The DigitalOcean cloud-init wrapper starts this script as the container's
    bash command. Inputs ride through environment variables so the host does not
    need SSH, and progress/results are uploaded as a zip to the signed PUT URL
    used by the existing watcher.
    """
    env_exports = "\n".join(
        f"export {shlex.quote(str(key))}={shlex.quote(str(value))}"
        for key, value in sorted(_mapping(plan.get("env")).items())
    )
    groot_cmd = _shell_join(plan.get("groot_server_command") or [])
    isaac_task_executor_cmd = _shell_join(
        plan.get("isaac_task_executor_command") or []
    )
    gear_sonic_controller_cmd = _shell_join(
        plan.get("gear_sonic_controller_command") or []
    )
    closed_loop_cmd = _shell_join(plan.get("closed_loop_command") or [])
    return f"""set -euo pipefail
mkdir -p /workspace/closed_loop_out /workspace/out
{env_exports}
cat > /workspace/upload_progress.py <<'PY'
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

workspace = Path("/workspace")
phase = os.environ.get("BLUEPRINT_BOOTSTRAP_PHASE", "unknown")
bootstrap = {{
    "schema_version": "groot_oscar_closed_loop_bootstrap.v1",
    "phase": phase,
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "launch_session_id": os.environ.get("BLUEPRINT_LAUNCH_SESSION_ID"),
    "raw_secret_values_recorded": False,
}}
(workspace / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
zip_path = workspace / "out" / "groot_oscar_closed_loop_worker_output.zip"
include = [
    workspace / "bootstrap.json",
    workspace / "sealed_launch_plan.json",
    workspace / "seed_provenance.json",
    workspace / "initial_policy_frame.png",
    workspace / "route.json",
    workspace / "task_prompt.txt",
    workspace / "groot_oscar_image_healthcheck.json",
    workspace / "groot_oscar_image_healthcheck.stderr.log",
    workspace / "groot_server.log",
    workspace / "gear_sonic_controller.log",
    workspace / "isaac_task_executor.log",
    workspace / "initial_g1_sonic_state.json",
    workspace / "runtime_ephemeral_trust.json",
    workspace / "closed_loop_stdout.log",
    workspace / "closed_loop_stderr.log",
    workspace / "isaac_runtime_result.json",
]
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in include:
        if path.is_file():
            zf.write(path, path.relative_to(workspace).as_posix())
    out_dir = workspace / "closed_loop_out"
    if out_dir.is_dir():
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(workspace).as_posix())
put_url = os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", "")
if put_url:
    subprocess.run(["curl", "-fsS", "-X", "PUT", "--upload-file", str(zip_path), put_url], check=False)
PY

cat > /workspace/write_result.py <<'PY'
import json
import os
from pathlib import Path

rc = int(os.environ.get("BLUEPRINT_CLOSED_LOOP_RC", "0"))
failure = os.environ.get("BLUEPRINT_WORKER_FAILURE", "").strip()
manifest_path = Path("/workspace/closed_loop_out/oscar_isaac_closed_loop_manifest.json")
manifest = {{}}
if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        manifest = {{"status": "blocked", "blockers": [f"closed_loop_manifest_unreadable:{{type(exc).__name__}}"]}}
blockers = list(manifest.get("blockers") or [])
if failure:
    blockers.append(failure)
if rc != 0:
    blockers.append(f"closed_loop_command_exit_{{rc}}")
status = "completed" if rc == 0 and manifest.get("status") == "completed" and not blockers else "blocked"
result = {{
    "schema_version": "groot_oscar_closed_loop_worker_result.v1",
    "status": status,
    "blockers": blockers,
    "closed_loop_return_code": rc,
    "closed_loop_manifest_path": str(manifest_path),
    "closed_loop_manifest": manifest,
    "raw_secret_values_recorded": False,
    "claim_boundary": {{
        "worker_completed_is_not_generated_video_success": True,
        "worker_completed_is_not_forward_inverse_consistency": True,
        "worker_completed_is_not_real_world_task_success": True,
        "worker_completed_is_not_physical_robot_readiness": True,
    }},
}}
Path("/workspace/isaac_runtime_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
PY

upload_phase() {{
  BLUEPRINT_BOOTSTRAP_PHASE="$1" python /workspace/upload_progress.py || true
}}

upload_phase container_bash_started

curl -fsSL "$BLUEPRINT_EVAL_MANIFEST_URI" -o /workspace/input_bundle.zip
python - <<'PY'
import os, pathlib, shutil, stat, zipfile
root = pathlib.Path('/workspace').resolve()
with zipfile.ZipFile('/workspace/input_bundle.zip') as archive:
    for member in archive.infolist():
        rel = pathlib.PurePosixPath(member.filename)
        mode = (member.external_attr >> 16) & 0o170000
        if rel.is_absolute() or '..' in rel.parts or stat.S_ISLNK(mode):
            raise RuntimeError('unsafe_input_bundle_member:' + member.filename)
        target = (root / pathlib.Path(*rel.parts)).resolve()
        if os.path.commonpath([str(root), str(target)]) != str(root):
            raise RuntimeError('input_bundle_member_escapes_workspace')
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, open(target, 'wb') as sink:
            shutil.copyfileobj(source, sink)
PY

python - <<'PY'
import base64
import os
from pathlib import Path

Path("/workspace/initial_policy_frame.png").write_bytes(
    base64.b64decode(os.environ["BLUEPRINT_INITIAL_POLICY_FRAME_B64"])
)
Path("/workspace/route.json").write_text(
    base64.b64decode(os.environ["BLUEPRINT_ROUTE_JSON_B64"]).decode("utf-8"),
    encoding="utf-8",
)
Path("/workspace/task_prompt.txt").write_text(
    os.environ["BLUEPRINT_TASK_PROMPT"],
    encoding="utf-8",
)
Path("/workspace/sealed_launch_plan.json").write_text(
    base64.b64decode(os.environ["BLUEPRINT_SEALED_LAUNCH_PLAN_B64"]).decode("utf-8"),
    encoding="utf-8",
)
Path("/workspace/seed_provenance.json").write_text(
    base64.b64decode(os.environ.get("BLUEPRINT_SEED_PROVENANCE_B64", "e30=")).decode("utf-8"),
    encoding="utf-8",
)
PY
upload_phase inputs_ready
python -m blueprint_pipeline.g1_kitchen_bundle_compatibility \
  --manifest /workspace/bundle_manifest.json

mkdir -p /run/blueprint-secrets
chmod 700 /run/blueprint-secrets
python -m blueprint_pipeline.runtime_ephemeral_trust \
  --secret-root /run/blueprint-secrets \
  --environment-file /run/blueprint-secrets/trust_env.sh \
  --public-manifest /workspace/runtime_ephemeral_trust.json
source /run/blueprint-secrets/trust_env.sh

{STARTUP_GATES_SCRIPT}

set +e
python3 /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py --require-cuda \
  > /workspace/groot_oscar_image_healthcheck.json \
  2> /workspace/groot_oscar_image_healthcheck.stderr.log
HEALTHCHECK_RC=$?
set -e
if [ "$HEALTHCHECK_RC" -ne 0 ]; then
  BLUEPRINT_CLOSED_LOOP_RC="$HEALTHCHECK_RC" \
    BLUEPRINT_WORKER_FAILURE="groot_oscar_image_healthcheck_failed" \
    python /workspace/write_result.py
  upload_phase runner_done
  exit "$HEALTHCHECK_RC"
fi
upload_phase healthcheck_passed

{groot_cmd} > /workspace/groot_server.log 2>&1 &
GROOT_PID=$!
{gear_sonic_controller_cmd} > /workspace/gear_sonic_controller.log 2>&1 &
GEAR_SONIC_PID=$!
{isaac_task_executor_cmd} > /workspace/isaac_task_executor.log 2>&1 &
ISAAC_TASK_PID=$!
cleanup() {{
  kill "$GROOT_PID" >/dev/null 2>&1 || true
  kill "$GEAR_SONIC_PID" >/dev/null 2>&1 || true
  kill "$ISAAC_TASK_PID" >/dev/null 2>&1 || true
}}
trap cleanup EXIT

set +e
python - <<'PY'
import socket
import time

deadline = time.time() + 900
while time.time() < deadline:
    sock = socket.socket()
    sock.settimeout(2)
    try:
        sock.connect(("127.0.0.1", 5550))
    except OSError:
        time.sleep(5)
    else:
        sock.close()
        break
else:
    raise SystemExit("groot_policy_server_not_ready")
PY
GROOT_READY_RC=$?
set -e
if [ "$GROOT_READY_RC" -ne 0 ]; then
  BLUEPRINT_CLOSED_LOOP_RC="$GROOT_READY_RC" \
    BLUEPRINT_WORKER_FAILURE="groot_policy_server_not_ready" \
    python /workspace/write_result.py
  upload_phase runner_done
  exit "$GROOT_READY_RC"
fi
upload_phase groot_server_ready

{GEAR_SONIC_READY_SCRIPT}

set +e
python - <<'PY'
import json
import time
import urllib.request
from pathlib import Path

deadline = time.time() + 900
while time.time() < deadline:
    if Path('/workspace/initial_g1_sonic_state.json').is_file():
        try:
            payload = json.loads(Path('/workspace/initial_g1_sonic_state.json').read_text())
            if payload.get('measurement', {{}}).get('surrogate') is False:
                break
        except Exception:
            pass
    time.sleep(5)
else:
    raise SystemExit('persistent_isaac_task_executor_not_ready')
PY
ISAAC_TASK_READY_RC=$?
set -e
if [ "$ISAAC_TASK_READY_RC" -ne 0 ]; then
  BLUEPRINT_CLOSED_LOOP_RC="$ISAAC_TASK_READY_RC" \
    BLUEPRINT_WORKER_FAILURE="persistent_isaac_task_executor_not_ready" \
    python /workspace/write_result.py
  upload_phase runner_done
  exit "$ISAAC_TASK_READY_RC"
fi
upload_phase isaac_task_executor_ready

set +e
{closed_loop_cmd} > /workspace/closed_loop_stdout.log 2> /workspace/closed_loop_stderr.log
RC=$?
set -e

BLUEPRINT_CLOSED_LOOP_RC="$RC" python /workspace/write_result.py
upload_phase runner_done
exit "$RC"
"""


def build_launch_spec(
    *,
    job_dir: Path,
    image_ref: str,
    start_frame: Path,
    route_payload: Mapping[str, Any],
    task_prompt: str,
    plan: Mapping[str, Any],
    launch_nonce: str,
    seed_provenance: Mapping[str, Any] | None = None,
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
    volume_gb: int = DEFAULT_VOLUME_GB,
    max_hourly_rate_usd: float = DEFAULT_MAX_HOURLY_RATE_USD,
) -> RenderLaunchSpec:
    put_url = (job_dir / "provider_output_put_url.txt").read_text(encoding="utf-8").strip()
    bundle_url = (job_dir / "provider_bundle_url.txt").read_text(encoding="utf-8").strip()
    env = {
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "CUDA_VISIBLE_DEVICES": "0",
        "BLUEPRINT_EVAL_MANIFEST_URI": bundle_url,
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": put_url,
        "BLUEPRINT_INITIAL_POLICY_FRAME_B64": _b64_bytes(start_frame.read_bytes()),
        "BLUEPRINT_ROUTE_JSON_B64": _b64_text(json.dumps(dict(route_payload), indent=2)),
        "BLUEPRINT_TASK_PROMPT": task_prompt,
        "BLUEPRINT_SEALED_LAUNCH_PLAN_B64": _json_b64(plan),
        "BLUEPRINT_SEED_PROVENANCE_B64": _json_b64(seed_provenance or {}),
        "BLUEPRINT_LAUNCH_SESSION_ID": launch_nonce,
        "BLUEPRINT_WORKER_IMAGE_DIGEST": image_ref,
        SEALED_CONFIRMED_ENV: "true",
    }
    return RenderLaunchSpec(
        name="blueprint-groot-oscar-closed-loop",
        image=image_ref,
        env=env,
        bootstrap_argv=["-lc", build_worker_bootstrap_script(plan)],
        entrypoint=["bash"],
        container_disk_gb=container_disk_gb,
        volume_gb=volume_gb,
        max_hourly_rate_usd=max_hourly_rate_usd,
        min_gpu_ram_mb=DEFAULT_MIN_GPU_RAM_MB,
    )


def _prelaunch_spend_guard(
    *,
    allow_paid: bool,
    max_spend_usd: float | None,
    max_seconds: int,
    max_hourly_rate_usd: float,
) -> dict[str, Any]:
    seconds = max(0, int(max_seconds or 0))
    hourly = float(max_hourly_rate_usd)
    estimated = round(hourly * (seconds / 3600.0), 4)
    blockers: list[str] = []
    if not allow_paid:
        blockers.append("paid_launch_not_requested")
    if max_spend_usd is None:
        blockers.append("groot_oscar_closed_loop_max_spend_usd_missing")
    elif float(max_spend_usd) <= 0:
        blockers.append("groot_oscar_closed_loop_max_spend_usd_must_be_positive")
    elif estimated > float(max_spend_usd):
        blockers.append("groot_oscar_closed_loop_estimated_spend_exceeds_budget")
    can_launch = bool(allow_paid and not blockers)
    return {
        "schema_version": "groot_oscar_closed_loop_prelaunch_spend_guard.v1",
        "status": "passed" if can_launch else "blocked",
        "provider": DEFAULT_PROVIDER,
        "allow_paid": bool(allow_paid),
        "required_before_provider_launch": True,
        "can_launch": can_launch,
        "requested_budget_usd": max_spend_usd,
        "estimated_max_spend_usd": estimated,
        "max_hourly_rate_usd": hourly,
        "max_seconds": seconds,
        "blockers": blockers,
        "claim_boundary": {
            "spend_guard_only": True,
            "can_launch_is_not_provider_success": True,
            "can_launch_is_not_task_success": True,
            "no_provider_api_call_before_can_launch": True,
        },
    }


def _capacity_row_for_pre_spend(capacity: Mapping[str, Any]) -> dict[str, Any]:
    viable = capacity.get("viable_size_regions")
    if isinstance(viable, Sequence) and not isinstance(viable, (str, bytes)):
        for item in viable:
            if isinstance(item, Mapping):
                return dict(item)
    return {}


def _gpu_vram_gb_from_do_capacity(row: Mapping[str, Any]) -> float | None:
    raw = row.get("gpu_ram_mb")
    try:
        return float(raw) / 1000.0
    except (TypeError, ValueError):
        pass
    size = _string(row.get("size"))
    if not size:
        return None
    gpu_ram_mb = DO_GPU_SIZE_VRAM_MB.get(size)
    return (float(gpu_ram_mb) / 1000.0) if gpu_ram_mb is not None else None


def _hardware_contract_for_capacity_row(
    *,
    plan: Mapping[str, Any],
    capacity_row: Mapping[str, Any],
    container_disk_gb: int,
) -> dict[str, Any]:
    return build_lane_hardware_contract(
        lane=_string(plan.get("lane")),
        gpu_type_id=_string(capacity_row.get("size")) or None,
        vram_gb=_gpu_vram_gb_from_do_capacity(capacity_row),
        disk_gb=float(container_disk_gb),
    )


def _pre_spend_capacity_evidence(
    *,
    capacity: Mapping[str, Any],
    capacity_row: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(capacity_row)
    return {
        "available": _string(capacity.get("status")) == "available" and bool(row),
        "detail": row.get("size") or "digitalocean_viable_size_region_missing",
        "capacity_preflight_status": capacity.get("status"),
        "selected_size": row.get("size"),
        "selected_region": (row.get("matching_regions") or [None])[0]
        if isinstance(row.get("matching_regions"), list)
        else None,
        "selected_gpu_ram_mb": row.get("gpu_ram_mb"),
    }


def _run_pre_spend_preflight(
    *,
    out: Path,
    plan: Mapping[str, Any],
    capacity: Mapping[str, Any],
    capacity_row: Mapping[str, Any],
    image_ref: str,
    provider_available: Mapping[str, Any],
    prelaunch: Mapping[str, Any],
    container_disk_gb: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hardware_contract = _hardware_contract_for_capacity_row(
        plan=plan,
        capacity_row=capacity_row,
        container_disk_gb=container_disk_gb,
    )
    try:
        preflight = require_pre_spend_preflight(
            lane=_string(plan.get("lane")) or LANE,
            provider=DEFAULT_PROVIDER,
            credential_present=provider_available.get("available") is True,
            capacity_evidence=_pre_spend_capacity_evidence(
                capacity=capacity,
                capacity_row=capacity_row,
            ),
            image_contract=image_contract_from_ref(image_ref),
            runtime_contract=runtime_contract_for_pre_spend(
                WORKER_PROGRESS_STALL_PHASES
            ),
            spend_gate_open=prelaunch.get("can_launch") is True,
            hardware_contract=hardware_contract,
            record_dir=out,
        )
    except PreSpendPreflightBlocked as blocked_preflight:
        preflight = blocked_preflight.preflight
    return hardware_contract, preflight


def _closed_loop_result_contract(
    *,
    watch: Mapping[str, Any],
    expected_steps_cap: int,
    expected_min_task_completion_steps: int,
    expected_min_coherent_horizon_frames: int,
    expected_forward_inverse_consistency_required: bool = True,
    expected_generated_video_success_label_required: bool = False,
) -> dict[str, Any]:
    """Validate the collected closed-loop manifest from the worker output.

    This is post-run contract validation, not semantic task success. It ensures
    the paid worker did not silently regress to fixed-step, sub-native,
    incoherent, or non-fresh-policy behavior after provider execution.
    """
    blockers: list[str] = []
    runner_result = _mapping(watch.get("runner_result"))
    closed_loop = _mapping(runner_result.get("closed_loop_manifest"))
    if watch.get("status") != "completed":
        blockers.append("worker_watch_not_completed")
    if runner_result.get("status") != "completed":
        blockers.append("worker_result_not_completed")
    if not closed_loop:
        blockers.append("closed_loop_manifest_missing_from_worker_result")
    elif closed_loop.get("status") != "completed":
        blockers.append("closed_loop_manifest_not_completed")

    episode = _mapping(closed_loop.get("episode_termination"))
    if episode.get("stop_on_task_completion") is not True:
        blockers.append("closed_loop_result_missing_task_adaptive_termination")
    if int(episode.get("steps_cap") or 0) != int(expected_steps_cap):
        blockers.append("closed_loop_result_steps_cap_mismatch")
    if int(episode.get("min_steps") or 0) < int(expected_min_task_completion_steps):
        blockers.append("closed_loop_result_min_steps_before_task_completion_too_low")
    if int(closed_loop.get("steps_executed") or 0) < 1:
        blockers.append("closed_loop_result_no_steps_executed")

    coherence = _mapping(closed_loop.get("generated_clip_coherence"))
    required = int(coherence.get("min_coherent_horizon_frames_required") or 0)
    measured = coherence.get("min_measured_coherent_horizon_frames")
    if required < int(expected_min_coherent_horizon_frames):
        blockers.append("closed_loop_result_coherence_gate_below_expected")
    if measured is None:
        blockers.append("closed_loop_result_coherence_not_measured")
    elif int(measured) < int(expected_min_coherent_horizon_frames):
        blockers.append("closed_loop_result_coherence_below_expected")

    proof = _mapping(closed_loop.get("proof"))
    if proof.get("feed_forward_verified") is not True:
        blockers.append("closed_loop_result_feed_forward_not_verified")
    if proof.get("policy_observes_wam_generated_next_observation") is not True:
        blockers.append("closed_loop_result_policy_did_not_observe_wam_output")
    if proof.get("fresh_learned_policy_requery_steps") in (None, 0):
        blockers.append("closed_loop_result_fresh_policy_requery_not_proven")
    if expected_forward_inverse_consistency_required:
        if closed_loop.get("forward_inverse_consistency_proven") is not True:
            blockers.append("closed_loop_result_forward_inverse_consistency_not_proven")
        if int(proof.get("external_episode_consistency_scorer_ran_steps") or 0) < int(
            closed_loop.get("steps_executed") or 0
        ):
            blockers.append("closed_loop_result_consistency_scorer_missing_steps")

    success_proof = _mapping(closed_loop.get("success_proof"))
    generated_video_success_label_passed = (
        success_proof.get("generated_video_success_label_passed") is True
    )
    if (
        expected_generated_video_success_label_required
        and not generated_video_success_label_passed
    ):
        blockers.append("closed_loop_result_generated_video_success_label_not_proven")
    return {
        "schema_version": "groot_oscar_closed_loop_result_contract.v1",
        "status": "PASS" if not blockers else "FAIL",
        "blockers": sorted(set(blockers)),
        "runner_result_source": watch.get("runner_result_source"),
        "closed_loop_status": closed_loop.get("status"),
        "steps_executed": closed_loop.get("steps_executed"),
        "steps_cap": episode.get("steps_cap"),
        "min_steps_before_task_completion": episode.get("min_steps"),
        "task_adaptive_termination": {
            "stop_on_task_completion": bool(episode.get("stop_on_task_completion")),
            "reason": episode.get("reason"),
            "task_completed_early": bool(episode.get("task_completed_early")),
        },
        "generated_clip_coherence_gate": {
            "required": required,
            "expected_minimum": int(expected_min_coherent_horizon_frames),
            "min_measured": measured,
        },
        "fresh_policy_requery_steps": proof.get("fresh_learned_policy_requery_steps"),
        "feed_forward_verified": proof.get("feed_forward_verified"),
        "policy_observes_wam_generated_next_observation": proof.get(
            "policy_observes_wam_generated_next_observation"
        ),
        "forward_inverse_consistency": {
            "required": bool(expected_forward_inverse_consistency_required),
            "proven": bool(closed_loop.get("forward_inverse_consistency_proven")),
            "external_episode_consistency_scorer_ran_steps": proof.get(
                "external_episode_consistency_scorer_ran_steps"
            ),
            "forward_inverse_consistency_proven_steps": proof.get(
                "forward_inverse_consistency_proven_steps"
            ),
        },
        "generated_video_success_label": {
            "required": bool(expected_generated_video_success_label_required),
            "passed": generated_video_success_label_passed,
        },
        "task_success_summary": {
            "manipulation_success_proven": bool(
                success_proof.get("manipulation_success_proven")
            ),
            "simulated_manipulation_success_shown": bool(
                success_proof.get("simulated_manipulation_success_shown")
            ),
            "generated_video_success_label_passed": bool(
                generated_video_success_label_passed
            ),
            "real_world_task_success_proven": bool(
                success_proof.get("real_world_task_success_proven")
            ),
            "success_proof_separate_from_structural_loop_proof": bool(
                success_proof.get("success_proof_separate_from_structural_loop_proof")
            ),
        },
        "claim_boundary": (
            "This contract validates the collected closed-loop run shape and "
            "quality gates. It does not convert route completion, generated "
            "video labels, or structural loop completion into real-world task "
            "success or physical robot readiness."
        ),
    }


def run_groot_oscar_digitalocean_closed_loop_job(
    *,
    start_frame: str | Path,
    route_file: str | Path,
    task_prompt: str,
    out_dir: str | Path,
    steps: int = DEFAULT_EPISODE_MAX_STEPS,
    oscar_height: int = 480,
    oscar_width: int = 640,
    min_coherent_horizon_frames: int = DEFAULT_MIN_COHERENT_HORIZON_FRAMES,
    min_task_completion_steps: int = DEFAULT_MIN_TASK_COMPLETION_STEPS,
    allow_paid: bool = False,
    max_spend_usd: float | None = None,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    max_hourly_rate_usd: float = DEFAULT_MAX_HOURLY_RATE_USD,
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
    volume_gb: int = DEFAULT_VOLUME_GB,
    seed_provenance: Mapping[str, Any] | None = None,
    seed_provenance_file: str | Path | None = None,
    key_prefix: str = "blueprint/groot-oscar-closed-loop",
    image_ref: str | None = None,
    wam_consistency_command: str | None = None,
    require_generated_video_success_label: bool = False,
    wam_success_label_command: str | None = None,
    allow_wam_success_labeling: bool = False,
    wam_success_label_timeout_seconds: float
    | None = DEFAULT_WAM_SUCCESS_LABEL_TIMEOUT_SECONDS,
    task_success_contract_file: str | Path | None = None,
    attempt_input_manifest_file: str | Path | None = None,
    kitchen_asset_archive_file: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    ensure_dir(out)
    seed = Path(start_frame)
    route_path = Path(route_file)
    if task_success_contract_file is None:
        candidate = route_path.parent / "task_success_contract.json"
        task_success_contract_file = candidate if candidate.is_file() else None
    if attempt_input_manifest_file is None:
        candidate = route_path.parent / "attempt_input_manifest.json"
        attempt_input_manifest_file = candidate if candidate.is_file() else None
    if kitchen_asset_archive_file is None:
        candidate = route_path.parent / "kitchen_assets.zip"
        kitchen_asset_archive_file = candidate if candidate.is_file() else None
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [],
        "provider": DEFAULT_PROVIDER,
        "start_frame": str(seed),
        "route_file": str(route_path),
        "steps": int(steps),
        "oscar_height": int(oscar_height),
        "oscar_width": int(oscar_width),
        "min_coherent_horizon_frames": int(min_coherent_horizon_frames),
        "episode_termination": "task_completion_or_step_cap",
        "episode_length_contract": _episode_length_contract(
            steps_cap=int(steps),
            stop_on_task_completion=True,
            min_steps_before_task_completion=int(min_task_completion_steps),
            oscar_num_frames_arg=None,
        ),
        "container_disk_gb": int(container_disk_gb),
        "volume_gb": int(volume_gb),
        "seed_provenance_present": bool(seed_provenance),
        "seed_provenance_file": str(seed_provenance_file) if seed_provenance_file else None,
        "raw_secret_values_recorded": False,
    }
    if not seed.is_file():
        manifest["blockers"].append("start_frame_missing")
        return _write_job_manifest(out, manifest)
    if not route_path.is_file():
        manifest["blockers"].append("route_file_missing")
        return _write_job_manifest(out, manifest)
    required_runtime_inputs = {
        "task_success_contract_file": task_success_contract_file,
        "attempt_input_manifest_file": attempt_input_manifest_file,
        "kitchen_asset_archive_file": kitchen_asset_archive_file,
    }
    missing_runtime_inputs = [
        name
        for name, value in required_runtime_inputs.items()
        if value is None or not Path(value).is_file()
    ]
    manifest["persistent_isaac_runtime_inputs"] = {
        "status": "ready" if not missing_runtime_inputs else "blocked",
        "missing": missing_runtime_inputs,
        "required_for_paid_launch": True,
    }
    if allow_paid and missing_runtime_inputs:
        manifest["blockers"].extend(
            f"persistent_isaac_runtime_input_missing:{name}"
            for name in missing_runtime_inputs
        )
        return _write_job_manifest(out, manifest)
    route_payload = json.loads(route_path.read_text(encoding="utf-8"))
    if not isinstance(route_payload, Mapping):
        manifest["blockers"].append("route_file_must_contain_json_object")
        return _write_job_manifest(out, manifest)
    attempt_input = (
        _read_json_mapping(attempt_input_manifest_file)
        if attempt_input_manifest_file is not None
        else {}
    )
    launch_nonce = _string(attempt_input.get("launch_nonce"))
    if allow_paid and not launch_nonce:
        manifest["blockers"].append("attempt_input_manifest_launch_nonce_missing")
        return _write_job_manifest(out, manifest)
    env = {SEALED_CONFIRMED_ENV: "true"}
    if image_ref:
        env[IMAGE_REF_ENV] = image_ref
    contract = sealed_image_contract(env=env)
    manifest["sealed_image_contract"] = contract
    if contract.get("sealed_active") is not True:
        manifest["blockers"].extend(contract.get("blockers") or ["sealed_image_not_active"])
        return _write_job_manifest(out, manifest)
    resolved_wam_consistency_command = (
        str(wam_consistency_command).strip()
        if wam_consistency_command is not None
        else str(DEFAULT_CONFIGURED_WAM_CONSISTENCY_COMMAND or "").strip()
    )
    plan = build_sealed_launch_plan(
        start_frame="/workspace/initial_policy_frame.png",
        route_file="/workspace/route.json",
        steps=int(steps),
        task_prompt=task_prompt,
        output_dir="/workspace/closed_loop_out",
        oscar_height=int(oscar_height),
        oscar_width=int(oscar_width),
        min_coherent_horizon_frames=int(min_coherent_horizon_frames),
        min_task_adaptive_steps=int(min_task_completion_steps),
        wam_consistency_command=resolved_wam_consistency_command or None,
        require_generated_video_success_label=bool(
            require_generated_video_success_label
        ),
        wam_success_label_command=wam_success_label_command,
        allow_wam_success_labeling=bool(allow_wam_success_labeling),
        wam_success_label_timeout_seconds=wam_success_label_timeout_seconds,
        env=env,
    )
    manifest["sealed_launch_plan"] = plan
    if plan.get("sealed_active") is not True or plan.get("blockers"):
        manifest["blockers"].extend(plan.get("blockers") or ["sealed_launch_plan_blocked"])
        return _write_job_manifest(out, manifest)
    resume_command = _write_paid_resume_command(
        out,
        _paid_resume_command_payload(
            start_frame=seed,
            route_file=route_path,
            task_prompt=task_prompt,
            out_dir=out,
            image_ref=str(contract["image_ref"]),
            steps=int(steps),
            oscar_height=int(oscar_height),
            oscar_width=int(oscar_width),
            min_coherent_horizon_frames=int(min_coherent_horizon_frames),
            min_task_completion_steps=int(min_task_completion_steps),
            max_spend_usd=max_spend_usd,
            max_seconds=int(max_seconds),
            max_hourly_rate_usd=float(max_hourly_rate_usd),
            container_disk_gb=int(container_disk_gb),
            volume_gb=int(volume_gb),
            seed_provenance_file=seed_provenance_file,
            key_prefix=key_prefix,
            wam_consistency_command=resolved_wam_consistency_command or None,
            require_generated_video_success_label=bool(
                require_generated_video_success_label
            ),
            wam_success_label_command=wam_success_label_command,
            allow_wam_success_labeling=bool(allow_wam_success_labeling),
            wam_success_label_timeout_seconds=wam_success_label_timeout_seconds,
        ),
    )
    manifest["paid_launch_resume_command"] = {
        "path": str(out / PAID_RESUME_COMMAND_FILENAME),
        "mode": resume_command["mode"],
        "budget_placeholder": resume_command["budget_placeholder"],
        "will_query_digitalocean": resume_command["will_query_digitalocean"],
        "capacity_preflight_before_staging": resume_command[
            "capacity_preflight_before_staging"
        ],
    }
    provider = get_render_provider(DEFAULT_PROVIDER)
    manifest["provider_available"] = provider.available()
    prelaunch = _prelaunch_spend_guard(
        allow_paid=allow_paid,
        max_spend_usd=max_spend_usd,
        max_seconds=max_seconds,
        max_hourly_rate_usd=max_hourly_rate_usd,
    )
    manifest["prelaunch_spend_guard"] = prelaunch
    if not allow_paid:
        bundle_zip = _write_input_bundle(
            bundle_zip=out / "groot_oscar_closed_loop_input_bundle.zip",
            plan=plan,
            route_payload=route_payload,
            seed_path=seed,
            task_prompt=task_prompt,
            seed_provenance=seed_provenance,
            task_success_contract_path=task_success_contract_file,
            attempt_input_manifest_path=attempt_input_manifest_file,
            kitchen_asset_archive_path=kitchen_asset_archive_file,
        )
        manifest["bundle_zip"] = str(bundle_zip)
        manifest["status"] = "prepared"
        manifest["note"] = "local bundle and sealed launch plan prepared; re-run with --allow-paid to launch DigitalOcean"
        return _write_job_manifest(out, manifest)
    if prelaunch.get("can_launch") is not True:
        manifest["blockers"].append("groot_oscar_closed_loop_prelaunch_spend_guard_not_passed")
        manifest["blockers"].extend(prelaunch.get("blockers") or [])
        manifest["blockers"] = sorted(set(manifest["blockers"]))
        return _write_job_manifest(out, manifest)
    if attempt_input_manifest_file is None:
        manifest["blockers"].append("attempt_input_manifest_required_for_paid_launch")
        return _write_job_manifest(out, manifest)
    identity_gate = enforce_current_checkout_pre_allocation_identity(
        attempt_input_manifest_file=attempt_input_manifest_file,
        launch_image_ref=str(contract["image_ref"]),
        repo_root=Path(__file__).resolve().parents[2],
    )
    manifest["pre_allocation_identity_gate"] = identity_gate
    if identity_gate.get("status") != "PASS":
        manifest["blockers"].append("pre_allocation_identity_gate_not_passed")
        manifest["blockers"].extend(identity_gate.get("blockers") or [])
        manifest["blockers"] = sorted(set(manifest["blockers"]))
        return _write_job_manifest(out, manifest)
    capacity_request = {
        "provider": DEFAULT_PROVIDER,
        "min_gpu_ram_mb": DEFAULT_MIN_GPU_RAM_MB,
        "max_hourly_rate_usd": float(max_hourly_rate_usd),
        "capacity_preflight_before_staging": True,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "This read-only capacity request is scoped to the sealed GR00T+OSCAR "
            "lane. It must filter DigitalOcean candidates to the same GPU memory "
            "floor that launch will require before staging large bundles."
        ),
    }
    manifest["capacity_preflight_request_shape"] = {
        "provider": DEFAULT_PROVIDER,
        "min_gpu_ram_mb": capacity_request["min_gpu_ram_mb"],
        "max_hourly_rate_usd": capacity_request["max_hourly_rate_usd"],
        "capacity_preflight_before_staging": True,
    }
    capacity = provider.capacity_preflight(capacity_request)
    manifest["provider_capacity_preflight"] = capacity
    capacity_status = _string(capacity.get("status"))
    if capacity_status != "available":
        manifest["blockers"].extend(capacity.get("blockers") or [])
        manifest["blockers"].append("provider_capacity_unavailable_before_staging")
        manifest["blockers"].append(
            f"provider_capacity_preflight_status_{capacity_status or 'missing'}"
        )
        manifest["blockers"] = sorted(set(manifest["blockers"]))
        return _write_job_manifest(out, manifest)
    capacity_row = _capacity_row_for_pre_spend(capacity)
    manifest["selected_digitalocean_capacity"] = capacity_row or None
    hardware_contract, pre_spend_preflight = _run_pre_spend_preflight(
        out=out,
        plan=plan,
        capacity=capacity,
        capacity_row=capacity_row,
        image_ref=str(contract["image_ref"]),
        provider_available=_mapping(manifest.get("provider_available")),
        prelaunch=prelaunch,
        container_disk_gb=int(container_disk_gb),
    )
    manifest["lane_hardware_contract"] = hardware_contract
    manifest["pre_spend_preflight"] = pre_spend_preflight
    if pre_spend_preflight.get("status") != "PASS":
        manifest["blockers"].append(
            "groot_oscar_closed_loop_pre_spend_preflight_not_passed"
        )
        manifest["blockers"].extend(pre_spend_preflight.get("blockers") or [])
        manifest["blockers"] = sorted(set(manifest["blockers"]))
        return _write_job_manifest(out, manifest)
    if blockers := revalidate_attempt_artifact_bytes(attempt_input_manifest_file):
        manifest["blockers"].extend(blockers)
        return _write_job_manifest(out, manifest)
    bundle_zip = _write_input_bundle(
        bundle_zip=out / "groot_oscar_closed_loop_input_bundle.zip",
        plan=plan,
        route_payload=route_payload,
        seed_path=seed,
        task_prompt=task_prompt,
        seed_provenance=seed_provenance,
        task_success_contract_path=task_success_contract_file,
        attempt_input_manifest_path=attempt_input_manifest_file,
        kitchen_asset_archive_path=kitchen_asset_archive_file,
    )
    manifest["bundle_zip"] = str(bundle_zip)
    job_dir = out / "object_store_real_run"
    staged = stage_bundle(bundle_zip, job_dir, key_prefix=key_prefix)
    manifest["staging"] = {"status": staged.get("status")}
    if staged.get("status") != "completed":
        manifest["blockers"].append("staging_failed")
        manifest["staging"]["stderr_tail"] = staged.get("stderr_tail")
        return _write_job_manifest(out, manifest)
    spec = build_launch_spec(
        job_dir=job_dir,
        image_ref=str(contract["image_ref"]),
        start_frame=seed,
        route_payload=route_payload,
        task_prompt=task_prompt,
        plan=plan,
        launch_nonce=launch_nonce,
        seed_provenance=seed_provenance,
        container_disk_gb=int(container_disk_gb),
        volume_gb=int(volume_gb),
        max_hourly_rate_usd=max_hourly_rate_usd,
    )
    request = provider.build_request(spec, job_dir)
    if launch_nonce:
        (job_dir / "launch_session_nonce.txt").write_text(
            launch_nonce, encoding="utf-8"
        )
    request["max_hourly_rate_usd"] = float(max_hourly_rate_usd)
    request["prelaunch_spend_guard"] = prelaunch
    manifest["launch_request_shape"] = {
        "provider": DEFAULT_PROVIDER,
        "image": spec.image,
        "container_disk_gb": spec.container_disk_gb,
        "volume_gb": spec.volume_gb,
        "max_hourly_rate_usd": spec.max_hourly_rate_usd,
        "min_gpu_ram_mb": spec.min_gpu_ram_mb,
        "has_start_frame_payload": bool(spec.env.get("BLUEPRINT_INITIAL_POLICY_FRAME_B64")),
        "has_route_payload": bool(spec.env.get("BLUEPRINT_ROUTE_JSON_B64")),
    }
    if blockers := revalidate_attempt_artifact_bytes(attempt_input_manifest_file):
        manifest["blockers"].extend(blockers)
        return _write_job_manifest(out, manifest)
    pending = open_pending_teardown(
        provider=DEFAULT_PROVIDER,
        lane=LANE,
        run_id=f"groot-oscar-do-{uuid.uuid4().hex}",
        job_dir=str(job_dir),
        max_age_seconds=int(max_seconds) + 1800,
    )
    manifest["pending_teardown_record"] = pending["path"]
    launch = provider.launch(job_dir, request, cold=True)
    manifest["launch"] = launch
    if launch.get("instance_id"):
        bind_pending_teardown_instance(pending["path"], str(launch["instance_id"]))
    if launch.get("status") != "launched":
        if not launch.get("instance_id"):
            cancel_pending_teardown(
                pending["path"],
                reason="launch_returned_no_allocation",
                evidence=launch,
            )
        manifest["blockers"].append("launch_failed")
        return _write_job_manifest(out, manifest)
    watch = watch_and_collect(
        job_dir,
        out / "closed_loop_output",
        str(launch["instance_id"]),
        provider=provider,
        max_seconds=int(max_seconds),
        progress_timeout_seconds=900,
        progress_stall_phases=WORKER_PROGRESS_STALL_PHASES,
    )
    manifest["closed_loop_run"] = watch
    result_contract = _closed_loop_result_contract(
        watch=watch,
        expected_steps_cap=int(steps),
        expected_min_task_completion_steps=int(min_task_completion_steps),
        expected_min_coherent_horizon_frames=int(min_coherent_horizon_frames),
        expected_forward_inverse_consistency_required=True,
        expected_generated_video_success_label_required=bool(
            require_generated_video_success_label
        ),
    )
    manifest["closed_loop_result_contract"] = result_contract
    teardown_proof = teardown_proof_from_digitalocean_watch(
        instance_id=str(launch["instance_id"]),
        watch=watch,
    )
    manifest["teardown_proof"] = teardown_proof
    manifest["pending_teardown_close"] = close_pending_teardown(pending["path"], teardown_proof)
    finalized = finalize_digitalocean_attempt_closure(
        provider=provider,
        output_dir=out,
        image_ref=_string(contract.get("image_ref")),
        attempt_input_manifest_file=attempt_input_manifest_file,
        task_success_contract_file=task_success_contract_file,
        kitchen_asset_archive_file=kitchen_asset_archive_file,
        launch=launch,
        watch=watch,
        teardown_proof=teardown_proof,
        expected_episode_steps=int(steps),
        expected_min_episode_steps=int(min_task_completion_steps),
        expected_scenario_count=1,
    )
    closure = finalized["closure"]
    manifest["final_inventory"] = finalized["final_inventory"]
    manifest["g1_kitchen_attempt_closure"] = closure
    manifest["buyer_readout_projection"] = finalized["buyer_readout_projection"]
    manifest["status"] = "completed" if closure.get("status") == "completed" else "blocked"
    if manifest["status"] != "completed":
        manifest["blockers"].extend(closure.get("blockers") or [])
        if watch.get("status") != "completed":
            manifest["blockers"].append("closed_loop_run_not_completed")
        if result_contract.get("status") != "PASS":
            manifest["blockers"].append("closed_loop_result_contract_failed")
            manifest["blockers"].extend(result_contract.get("blockers") or [])
        manifest["blockers"] = sorted(set(manifest["blockers"]))
    return _write_job_manifest(out, manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_digitalocean_job_parser(
        description=__doc__ or "",
        defaults={
            "steps": DEFAULT_EPISODE_MAX_STEPS,
            "min_coherent_horizon_frames": DEFAULT_MIN_COHERENT_HORIZON_FRAMES,
            "min_steps": DEFAULT_MIN_TASK_COMPLETION_STEPS,
            "max_seconds": DEFAULT_MAX_SECONDS,
            "max_hourly_rate_usd": DEFAULT_MAX_HOURLY_RATE_USD,
            "container_disk_gb": DEFAULT_CONTAINER_DISK_GB,
            "volume_gb": DEFAULT_VOLUME_GB,
            "wam_success_label_timeout_seconds": DEFAULT_WAM_SUCCESS_LABEL_TIMEOUT_SECONDS,
        },
    )
    args = parser.parse_args(argv)
    if args.audit_prepared_dir:
        audit = audit_prepared_closed_loop_job(args.audit_prepared_dir)
        print(json.dumps(audit, indent=2, default=str))
        return 0 if audit.get("status") == "PASS" else 1
    if args.audit_objective_dir:
        audit = audit_kitchen_dishwasher_objective_readiness(args.audit_objective_dir)
        print(json.dumps(audit, indent=2, default=str))
        return 0 if audit.get("local_status") == "PASS" else 1
    if args.probe_digitalocean_capacity_dir:
        probe = probe_digitalocean_capacity_for_prepared_dir(
            args.probe_digitalocean_capacity_dir
        )
        print(json.dumps(probe, indent=2, default=str))
        return 0 if _mapping(probe.get("capacity_preflight")).get("status") == "available" else 1
    if args.wait_digitalocean_capacity_dir:
        wait = wait_for_digitalocean_capacity_then_launch_prepared_dir(
            args.wait_digitalocean_capacity_dir,
            max_attempts=args.wait_max_attempts,
            poll_interval_seconds=args.wait_poll_interval_seconds,
            launch_when_available=bool(args.launch_when_capacity_available),
            allow_paid=bool(args.allow_paid),
            max_spend_usd=args.max_spend_usd,
            acknowledge_digitalocean_query_approval=bool(
                args.acknowledge_digitalocean_query_approval
            ),
        )
        print(json.dumps(wait, indent=2, default=str))
        return 0 if wait.get("status") in {"capacity_available", "completed"} else 1
    if args.materialize_paid_resume_dir:
        materialized = materialize_paid_resume_command(
            args.materialize_paid_resume_dir,
            max_spend_usd=args.materialize_max_spend_usd,
            acknowledge_digitalocean_query_approval=bool(
                args.acknowledge_digitalocean_query_approval
            ),
        )
        print(json.dumps(materialized, indent=2, default=str))
        return 0 if materialized.get("status") == "ready" else 1
    missing = [
        flag
        for flag, value in (
            ("--start-frame", args.start_frame),
            ("--route-file", args.route_file),
            ("--task-prompt", args.task_prompt),
            ("--out-dir", args.out_dir),
        )
        if not value
    ]
    if missing:
        parser.error("missing required arguments outside --audit-prepared-dir: " + ", ".join(missing))
    result = run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=args.start_frame,
        route_file=args.route_file,
        task_prompt=args.task_prompt,
        out_dir=args.out_dir,
        steps=args.steps,
        oscar_height=args.oscar_height,
        oscar_width=args.oscar_width,
        min_coherent_horizon_frames=args.min_coherent_horizon_frames,
        min_task_completion_steps=args.min_steps,
        allow_paid=args.allow_paid,
        max_spend_usd=args.max_spend_usd,
        max_seconds=args.max_seconds,
        max_hourly_rate_usd=args.max_hourly_rate_usd,
        container_disk_gb=args.container_disk_gb,
        volume_gb=args.volume_gb,
        seed_provenance=_read_json_mapping(args.seed_provenance_file),
        seed_provenance_file=args.seed_provenance_file,
        key_prefix=args.key_prefix,
        image_ref=args.image_ref,
        wam_consistency_command=args.wam_consistency_command,
        require_generated_video_success_label=bool(
            args.require_generated_video_success_label
        ),
        wam_success_label_command=args.wam_success_label_command,
        allow_wam_success_labeling=bool(args.allow_wam_success_labeling),
        wam_success_label_timeout_seconds=args.wam_success_label_timeout_seconds,
        task_success_contract_file=args.task_success_contract_file,
        attempt_input_manifest_file=args.attempt_input_manifest_file,
        kitchen_asset_archive_file=args.kitchen_asset_archive_file,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") in {"completed", "prepared"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
