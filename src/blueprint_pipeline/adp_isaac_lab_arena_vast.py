"""Canonical capped Vast launcher for the founder-approved Arena protocol."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import zipfile
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping

from .adp_founder_sim_protocol import admit_founder_sim_execution, build_founder_sim_protocol
from .adp_isaac_lab_arena_request import build_arena_worker_request
from .common import ensure_dir, redacted_failure_detail, utc_now_iso, write_json
from .paid_lane_guard import (
    SPEND_ADMISSION_LOCK_PATH_ENV,
    PreSpendPreflightBlocked,
    image_contract_from_ref,
    require_pre_spend_preflight,
)
from .paid_resource_admission import PaidResourceAdmissionGrant
from .task_evaluation_artifact_manifest import (
    TaskEvaluationArtifactManifestError,
    build_task_evaluation_artifact_manifest,
    seal_unallocated_provider_teardown,
)
from .vast_independent_watchdog_control import (
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
)
from .vast_provider_adapter import (
    DEFAULT_VAST_API_KEY_FILE,
    VAST_API_KEY_FILE_ENV,
    run_vast_provider_adapter,
)
from .vast_session_budget_contract import attempt_estimated_cost, attempt_runtime_seconds
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


PROBE_KIND = "adp-isaac-lab-arena-native-control"
RESULT_SCHEMA_VERSION = "adp_isaac_lab_arena_vast_run.v1"
DEFAULT_IMAGE = (
    "nvcr.io/nvidia/isaac-sim:6.0.1@"
    "sha256:b1c542b2ecc549b3d1ebb78c25664aa3bacba1709e6ad8e0a68e09426d57dedb"
)
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/arena-native-control"
_VAST_MUTATION_ENV = (
    "BLUEPRINT_ALLOW_VAST_API_CALLS",
    "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
)
_ADP009D_GATED_BACKBONE_AUTH_ENV = (
    "BLUEPRINT_ADP009D_GATED_BACKBONE_AUTHORIZED"
)


ENTRYPOINT = r"""#!/usr/bin/env bash
set +e
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}"
mkdir -p "$OUT_DIR"
/isaac-sim/python.sh "$RUNTIME_DIR/adp_arena_provider_runner.py"
runner_rc=$?
if [ $runner_rc -ne 0 ] && [ ! -f "$OUT_DIR/adp_arena_native_canary.json" ]; then
/isaac-sim/python.sh - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "adp_arena_native_canary.json").write_text(json.dumps({
    "schema_version": "adp_arena_native_canary.v1",
    "status": "blocked",
    "blockers": [
        "adp_arena_runner_failed_without_runtime_result",
        "blocked_adp_arena_process_exited_without_result"
    ],
    "candidate_policy_queried": False,
    "candidate_outcomes_accessed": False,
    "provider_zero_required_after_return": True
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi
exit $runner_rc
"""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _vast_credential_file_present() -> bool:
    path = Path(
        os.environ.get(VAST_API_KEY_FILE_ENV, DEFAULT_VAST_API_KEY_FILE)
    ).expanduser()
    try:
        metadata = path.stat()
    except OSError:
        return False
    return bool(
        path.is_file()
        and not path.is_symlink()
        and metadata.st_size > 0
        and not (stat.S_IMODE(metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH))
    )


def _stage_machine_avoidlist(job: Path, machine_avoidlist_path: str | Path | None) -> Path:
    """Stage the machine avoidlist into the job directory, idempotently.

    Callers reasonably keep the avoidlist inside the job directory, which is
    exactly where it is staged to.  Copying a file onto itself raises
    SameFileError and aborts the launch after the admission receipt is already
    written, so an already-staged avoidlist is treated as satisfied.
    """

    local_avoidlist = job / "adp_arena_vast_machine_avoidlist.json"
    if machine_avoidlist_path is None:
        return local_avoidlist
    source = Path(machine_avoidlist_path).expanduser().resolve()
    if source != local_avoidlist.resolve():
        shutil.copy2(source, local_avoidlist)
    return local_avoidlist


def _next_attempt_root(job: Path) -> tuple[int, Path]:
    """Return a fresh evidence root without ever reusing paid-run staging state."""

    ledger = _read_json(job / "adp_arena_vast_session_budget.json")
    observed = int(ledger.get("attempt_count") or 0)
    attempts_dir = job / "attempts"
    if attempts_dir.is_dir():
        for path in attempts_dir.glob("attempt_*"):
            try:
                observed = max(observed, int(path.name.removeprefix("attempt_")))
            except ValueError:
                continue
    number = observed + 1
    return number, attempts_dir / f"attempt_{number:03d}"


def _write_run_result(job: Path, attempt_root: Path, result: Mapping[str, Any]) -> None:
    """Persist an immutable per-attempt result plus the root latest-result pointer."""

    ensure_dir(attempt_root)
    write_json(attempt_root / "adp_arena_vast_result.json", dict(result))
    write_json(job / "adp_arena_vast_result.json", dict(result))


def _remaining_session_live_minutes(
    *, job: Path, hard_cap_usd: float, hard_ttl_seconds: int, max_hourly_rate_usd: float
) -> int:
    """Compile cumulative spend/runtime evidence into a safe successor TTL."""

    ledger = _read_json(job / "adp_arena_vast_session_budget.json")
    attempts = [item for item in ledger.get("attempts") or [] if isinstance(item, Mapping)]
    prior_seconds = sum(attempt_runtime_seconds(item) for item in attempts)
    prior_cost = sum(attempt_estimated_cost(item) for item in attempts)
    runtime_minutes = math.floor(max(0.0, hard_ttl_seconds - prior_seconds) / 60.0)
    spend_minutes = math.floor(
        max(0.0, hard_cap_usd - prior_cost) * 60.0 / max_hourly_rate_usd
    )
    return max(0, min(runtime_minutes, spend_minutes))


@contextmanager
def _vast_authority_environment(*, gated_backbone_authorized: bool = False):
    managed = (*_VAST_MUTATION_ENV, _ADP009D_GATED_BACKBONE_AUTH_ENV)
    previous = {name: os.environ.get(name) for name in managed}
    try:
        for name in _VAST_MUTATION_ENV:
            os.environ[name] = "1"
        if gated_backbone_authorized:
            os.environ[_ADP009D_GATED_BACKBONE_AUTH_ENV] = "true"
        else:
            os.environ.pop(_ADP009D_GATED_BACKBONE_AUTH_ENV, None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def build_arena_native_control_bundle(
    *, approval_path: str | Path, job_dir: str | Path, generated_at: str | None = None
) -> dict[str, Any]:
    """Build a deterministic public-source bundle after exact digest approval."""

    approval_source = Path(approval_path).expanduser().resolve()
    approval = _read_json(approval_source)
    protocol = build_founder_sim_protocol()
    execution_admission = admit_founder_sim_execution(protocol, approval)
    worker_request = build_arena_worker_request(protocol)
    job = Path(job_dir).expanduser().resolve()
    if job.exists():
        shutil.rmtree(job)
    runtime = job / "provider_runtime"
    ensure_dir(runtime)
    generated = generated_at or utc_now_iso()
    worker_source = Path(__file__).with_name("adp_arena_native_canary_worker.py")
    shutil.copy2(worker_source, runtime / "adp_arena_provider_runner.py")
    _write_executable(runtime / "run_adp_arena_provider_runtime.sh", ENTRYPOINT)
    write_json(runtime / "founder_sim_protocol.json", protocol)
    write_json(runtime / "founder_sim_approval_receipt.json", approval)
    write_json(runtime / "founder_sim_execution_admission.json", execution_admission)
    write_json(runtime / "arena_worker_request.json", worker_request)
    readiness = {
        "schema_version": "adp_arena_provider_bundle.v1",
        "generated_at": generated,
        "status": "ready",
        "protocol_id": protocol["protocol_id"],
        "protocol_digest": protocol["protocol_digest"],
        "schedule_digest": protocol["schedule"]["schedule_digest"],
        "worker_request_digest": worker_request["worker_request_digest"],
        "approval_receipt_digest": approval.get("approval_receipt_digest"),
        "container_image": DEFAULT_IMAGE,
        "control_id": "arena_zero_action_negative",
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "runtime_entrypoint": "provider_runtime/run_adp_arena_provider_runtime.sh",
        "expected_output_filename": "adp_arena_native_canary.json",
        "local_bundle_ready_for_remote_staging": True,
        "provider_zero_required_after_return": True,
        "retry_cap": 0,
        "blockers": [],
        "raw_secret_values_recorded": False,
    }
    write_json(runtime / "adp_arena_provider_manifest.json", readiness)
    bundle_path = job / "adp_arena_provider_runtime_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as archive:
        for path in sorted(runtime.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(
                    path.relative_to(job).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.create_system = 3
                info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
                archive.writestr(
                    info,
                    path.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
    receipt = {
        **readiness,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _file_sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
    }
    write_json(job / "adp_arena_bundle_receipt.json", receipt)
    return receipt


def _extract_provider_output(
    path: Path,
    destination: Path,
    *,
    result_name: str = "adp_arena_native_canary.json",
    blocker_prefix: str = "adp_arena",
) -> dict[str, Any]:
    blockers: list[str] = []
    if not path.is_file():
        return {
            "status": "blocked",
            "blockers": [f"{blocker_prefix}_provider_output_zip_missing"],
        }
    if destination.exists():
        shutil.rmtree(destination)
    ensure_dir(destination)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    blockers.append(f"{blocker_prefix}_provider_output_zip_path_traversal")
            if not blockers:
                archive.extractall(destination)
    except (OSError, zipfile.BadZipFile):
        blockers.append(f"{blocker_prefix}_provider_output_zip_invalid")
    result_path = destination / result_name
    execution = _read_json(result_path)
    if not execution:
        blockers.append(f"{blocker_prefix}_runtime_result_missing")
    return {
        "status": "completed" if not blockers else "blocked",
        "result_path": str(result_path),
        "execution": execution,
        "blockers": sorted(set(blockers)),
    }


def _candidate_policy_query_blocker(
    execution: Mapping[str, Any],
    *,
    blocker_prefix: str,
    query_expected: bool = False,
) -> str | None:
    """Match observed query evidence to the transport's declared run mode."""

    queried = execution.get("candidate_policy_queried")
    if queried is query_expected:
        return None
    if queried is True:
        return f"{blocker_prefix}_candidate_policy_queried"
    if queried is False:
        return f"{blocker_prefix}_candidate_policy_query_required"
    return f"{blocker_prefix}_candidate_policy_query_status_missing"


def run_arena_native_control_vast(
    *,
    approval_path: str | Path,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    prepared_bundle: Mapping[str, Any] | None = None,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 1.00,
    hard_cap_usd: float = 4.00,
    hard_ttl_seconds: int = 14_400,
    expected_output_filename: str = "adp_arena_native_canary.json",
    container_image: str = DEFAULT_IMAGE,
    provider_bundle_kind: str = "adp_arena",
    result_schema_version: str = RESULT_SCHEMA_VERSION,
    object_store_key_prefix: str = DEFAULT_KEY_PREFIX,
    instance_label_prefix: str = "blueprint-adp-arena-",
    blocker_prefix: str = "adp_arena",
    min_gpu_ram_mb: int = 24_000,
    min_compute_cap: int = 0,
    minimum_driver_version: str = "",
    require_known_supported_isaac_driver: bool = True,
    enable_isaac_smoke: bool = True,
    forward_hf_token: bool = False,
    allowed_active_instance_ids: Sequence[int] = (),
    vast_launch_lock_file: str | Path | None = None,
    candidate_policy_query_expected: bool = False,
    preferred_gpu_keywords: tuple[str, ...] = (
        "RTX 4090",
        "RTX A6000",
        "RTX A5000",
        "RTX 3090",
    ),
    require_independent_watchdog: bool = False,
    authorization_consumption: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one zero-retry Arena acquisition behind an independent hard-TTL watchdog."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    generated = utc_now_iso()
    bundle = (
        dict(prepared_bundle)
        if prepared_bundle is not None
        else build_arena_native_control_bundle(
            approval_path=approval_path, job_dir=job / "bundle", generated_at=generated
        )
    )
    bundle_path = Path(str(bundle.get("bundle_path") or "")).expanduser().resolve()
    if (
        bundle.get("status") != "ready"
        or not bundle_path.is_file()
        or _file_sha256(bundle_path) != bundle.get("bundle_sha256")
    ):
        raise ValueError("adp_arena_prepared_bundle_binding_invalid")
    if not execute:
        result = {
            "schema_version": result_schema_version,
            "generated_at": generated,
            "status": "dry_run_ready",
            "bundle": bundle,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(job / "adp_arena_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("adp_arena_paid_resource_admission_grant_missing")

    attempt_number, attempt_root = _next_attempt_root(job)
    ensure_dir(attempt_root)
    remaining_live_minutes = _remaining_session_live_minutes(
        job=job,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        max_hourly_rate_usd=max_hourly_rate_usd,
    )
    if remaining_live_minutes < 30:
        result = {
            "schema_version": result_schema_version,
            "generated_at": generated,
            "status": "blocked",
            "attempt_number": attempt_number,
            "attempt_root": str(attempt_root),
            "remaining_live_minutes": remaining_live_minutes,
            "provider_mutations_performed": 0,
            "blockers": ["adp_arena_cumulative_budget_below_minimum_live_window"],
        }
        _write_run_result(job, attempt_root, result)
        return result
    provider_run = attempt_root / "vast_provider_run"
    ensure_dir(provider_run)
    vast_credential_present = _vast_credential_file_present()
    try:
        pre_spend_preflight = require_pre_spend_preflight(
            lane=provider_bundle_kind,
            provider="vast",
            credential_present=vast_credential_present,
            capacity_evidence={
                "available": vast_credential_present,
                "detail": (
                    "credential_bound; provider adapter rechecks global inventory and "
                    "offer capacity before allocation"
                ),
            },
            image_contract=image_contract_from_ref(container_image),
            runtime_contract={
                "startup_marker": "vast_instance_started_or_blocked",
                "progress_marker": "vast_provider_bundle_progress",
                "startup_timeout_seconds": remaining_live_minutes * 60,
                "no_progress_timeout_seconds": 1800,
            },
            spend_gate_open=(
                0.0 < max_hourly_rate_usd <= hard_cap_usd
                and remaining_live_minutes >= 30
            ),
            record_dir=provider_run,
            spend_admission_lock=(
                None
                if str(os.getenv(SPEND_ADMISSION_LOCK_PATH_ENV) or "").strip()
                else {}
            ),
        )
    except PreSpendPreflightBlocked as exc:
        result = {
            "schema_version": result_schema_version,
            "generated_at": generated,
            "status": "blocked",
            "attempt_number": attempt_number,
            "attempt_root": str(attempt_root),
            "provider_mutations_performed": 0,
            "pre_spend_preflight": exc.preflight,
            "blockers": sorted(
                {
                    f"{blocker_prefix}_pre_spend_preflight_not_passed",
                    *(
                        str(item)
                        for item in exc.preflight.get("blockers") or []
                        if str(item)
                    ),
                }
            ),
        }
        _write_run_result(job, attempt_root, result)
        return result
    staging_dir = attempt_root / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=str(bundle_path),
        key_prefix=os.getenv("BLUEPRINT_ADP_ARENA_OBJECT_STORE_PREFIX", object_store_key_prefix),
        expiration_seconds=max(hard_ttl_seconds + 1800, 18_000),
        generated_at=generated,
    )
    if staging.get("status") != "completed":
        result = {
            "schema_version": result_schema_version,
            "generated_at": generated,
            "status": "blocked",
            "attempt_number": attempt_number,
            "attempt_root": str(attempt_root),
            "provider_mutations_performed": 0,
            "blockers": staging.get("blockers") or ["adp_arena_object_store_staging_blocked"],
        }
        _write_run_result(job, attempt_root, result)
        return result

    bundle_url = (staging_dir / "provider_bundle_url.txt").read_text().strip()
    output_put_url = (staging_dir / "provider_output_put_url.txt").read_text().strip()
    output_get_url = (staging_dir / "provider_output_get_url.txt").read_text().strip()
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    local_avoidlist = _stage_machine_avoidlist(job, machine_avoidlist_path)
    adapter: dict[str, Any] = {}
    watchdog_handoff: dict[str, Any] = {"status": "not_required"}
    watchdog_close: dict[str, Any] = {"status": "not_required"}
    watchdog_handle = None
    if require_independent_watchdog:
        watchdog_handoff, watchdog_handle = arm_independent_vast_watchdog(
            job_dir=attempt_root,
            max_live_minutes=remaining_live_minutes,
            generated_at=generated,
            allowed_active_instance_ids=allowed_active_instance_ids,
            pod_name_prefix=instance_label_prefix,
        )
        if watchdog_handle is None:
            cleanup = cleanup_staged_wam_provider_objects(staging_dir)
            result = {
                "schema_version": result_schema_version,
                "generated_at": generated,
                "status": "blocked",
                "attempt_number": attempt_number,
                "attempt_root": str(attempt_root),
                "provider_mutations_performed": 0,
                "retry_cap": 0,
                "authorization_consumption": authorization_consumption,
                "independent_watchdog": watchdog_handoff,
                "all_staged_objects_absent": cleanup.get("all_objects_absent"),
                "blockers": [f"{blocker_prefix}_independent_watchdog_not_armed"],
            }
            _write_run_result(job, attempt_root, result)
            return result
    watchdog_close = {"status": "not_closed"}
    try:
        with _vast_authority_environment(
            gated_backbone_authorized=forward_hf_token
        ):
            adapter = run_vast_provider_adapter(
                job_dir=provider_run,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=max_hourly_rate_usd,
                target_spend_usd=hard_cap_usd,
                hard_cap_usd=hard_cap_usd,
                max_live_minutes=remaining_live_minutes,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=container_image,
                isaac_image=container_image,
                ngc_image_login_mode="always",
                provider_bundle=str(bundle_path),
                provider_bundle_url=bundle_url,
                provider_output_put_url=output_put_url,
                provider_output_get_url=output_get_url,
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=enable_isaac_smoke,
                enable_blueprint_bundle=True,
                provider_bundle_kind=provider_bundle_kind,
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=True,
                min_cold_isaac_pull_live_minutes=30,
                disk_gb=200,
                min_gpu_ram_mb=min_gpu_ram_mb,
                min_compute_cap=min_compute_cap,
                poll_interval_seconds=15,
                startup_timeout_seconds=remaining_live_minutes * 60,
                heartbeat_no_progress_seconds=1800,
                session_budget_ledger_path=job / "adp_arena_vast_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=(
                    require_known_supported_isaac_driver
                ),
                minimum_driver_version=minimum_driver_version,
                preferred_gpu_keywords=preferred_gpu_keywords,
                prefer_isaac_rt=True,
                machine_avoidlist_path=local_avoidlist,
                instance_label_prefix=(
                    watchdog_handle.pod_name_prefix
                    if watchdog_handle is not None
                    else instance_label_prefix
                ),
                started_instance_id_path=(
                    watchdog_handle.started_instance_id_path
                    if watchdog_handle is not None
                    else None
                ),
                retention_watchdog_handoff=watchdog_handoff,
                forward_hf_token=forward_hf_token,
                allowed_active_instance_ids=allowed_active_instance_ids,
                vast_launch_lock_file=vast_launch_lock_file,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [
                f"{blocker_prefix}_vast_adapter_failed:{redacted_failure_detail(exc)}"
            ],
            "provider_create_attempted": False,
            "continuing_spend_from_this_run": False,
            "raw_secret_values_recorded": False,
        }
        write_json(provider_run / "vast_provider_adapter_result.json", adapter)
        seal_unallocated_provider_teardown(
            provider_run,
            reason=f"{blocker_prefix}_vast_adapter_failed",
            generated_at=generated,
        )
    finally:
        try:
            watchdog_close = close_independent_vast_watchdog(
                job_dir=provider_run,
                handle=watchdog_handle,
                instance_ids=[int(value) for value in adapter.get("vast_instance_ids") or []],
                provider_teardown_completed=(
                    adapter.get("continuing_spend_from_this_run") is False
                ),
                provider_allocation_impossible=(
                    adapter.get("provider_create_attempted") is False
                ),
            )
        finally:
            cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    extracted = _extract_provider_output(
        output_zip,
        attempt_root / "immutable_execution",
        result_name=expected_output_filename,
        blocker_prefix=blocker_prefix,
    )
    execution = dict(extracted.get("execution") or {})
    teardown = _read_json(provider_run / "vast_teardown_manifest.json")
    blockers = list(adapter.get("blockers") or []) + list(extracted.get("blockers") or [])
    if execution.get("status") != "completed":
        blockers.extend(execution.get("blockers") or [f"{blocker_prefix}_runtime_not_completed"])
    policy_query_blocker = _candidate_policy_query_blocker(
        execution,
        blocker_prefix=blocker_prefix,
        query_expected=candidate_policy_query_expected,
    )
    if policy_query_blocker:
        blockers.append(policy_query_blocker)
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append(f"{blocker_prefix}_vast_provider_zero_not_proven")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append(f"{blocker_prefix}_object_store_provider_zero_not_proven")
    if require_independent_watchdog and watchdog_close.get("status") not in {
        "provider_terminal",
    }:
        blockers.append(f"{blocker_prefix}_independent_watchdog_not_terminal")
    artifact_manifest_path = attempt_root / "artifact_manifest.json"
    try:
        artifact_manifest = build_task_evaluation_artifact_manifest(
            attempt_root=attempt_root,
            artifact_roots={
                "provider_runtime_evidence": attempt_root / "immutable_execution",
                "allocator_adapter_result": (
                    provider_run / "vast_provider_adapter_result.json"
                ),
                "teardown_manifest": provider_run / "vast_teardown_manifest.json",
            },
            required_roles=(
                "provider_runtime_evidence",
                "allocator_adapter_result",
                "teardown_manifest",
            ),
            binding={
                "allocator_lane": provider_bundle_kind,
                "attempt_number": attempt_number,
                "bundle_sha256": bundle.get("bundle_sha256"),
                "protocol_digest": bundle.get("protocol_digest"),
                "provider": "vast",
                "result_schema_version": result_schema_version,
                "retry_cap": 0,
            },
            output_path=artifact_manifest_path,
        )
        blockers.extend(artifact_manifest.get("blockers") or [])
    except (OSError, TaskEvaluationArtifactManifestError) as exc:
        blockers.append(f"{blocker_prefix}_artifact_manifest_failed:{redacted_failure_detail(exc)}")
    result = {
        "schema_version": result_schema_version,
        "generated_at": generated,
        "status": "completed" if not blockers else "blocked",
        "attempt_number": attempt_number,
        "attempt_root": str(attempt_root),
        "protocol_digest": bundle.get("protocol_digest"),
        "bundle_sha256": bundle.get("bundle_sha256"),
        "native_control_result_path": extracted.get("result_path"),
        "adapter_result_path": str(provider_run / "vast_provider_adapter_result.json"),
        "teardown_manifest_path": str(provider_run / "vast_teardown_manifest.json"),
        "artifact_manifest_path": str(artifact_manifest_path),
        "pre_spend_preflight": pre_spend_preflight,
        "independent_watchdog_handoff": watchdog_handoff,
        "independent_watchdog_close": watchdog_close,
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "attempt_max_live_minutes": remaining_live_minutes,
        "retry_cap": 0,
        "authorization_consumption": authorization_consumption,
        "independent_watchdog": watchdog_close,
        "watchdog_receipt_path": (
            str(
                attempt_root
                / "independent_vast_watchdog"
                / "groot_oscar_runpod_canary_watchdog.json"
            )
            if require_independent_watchdog
            else None
        ),
        "object_store_cleanup_path": str(
            staging_dir / "wam_provider_object_store_cleanup.json"
        ),
        "candidate_policy_query_expected": bool(candidate_policy_query_expected),
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    _write_run_result(job, attempt_root, result)
    return result


__all__ = [
    "DEFAULT_IMAGE",
    "PROBE_KIND",
    "build_arena_native_control_bundle",
    "run_arena_native_control_vast",
]

def main(argv: list[str] | None = None) -> int:
    """Build the immutable Isaac Lab Arena native-control bundle.

    Added because the allocator refuses a bundle whose `blueprint_commit` is not
    the commit the control plane is running, and every deploy moves that commit.
    A lane whose bundle can only be produced by calling a Python function is
    launchable exactly once and then never again -- which is the state this lane
    was in, reading as `ready` everywhere while being unbuildable.

    Performs no provider mutation and rents nothing.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Build the immutable Isaac Lab Arena native-control bundle.")
    parser.add_argument("--approval", required=True)
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = build_arena_native_control_bundle(
            approval_path=args.approval,
            job_dir=args.job_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") in {"ready", "sealed"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
