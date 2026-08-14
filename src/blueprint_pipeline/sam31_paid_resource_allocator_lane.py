"""Canonical allocator sub-lane for the SAM 3.1 Vast source-track canary."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any, Callable

from .common import ensure_dir, redacted_failure_detail, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .gpu_render_providers import get_render_provider
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .sam31_gpu_admission import collect_sam31_vast_preflight, prepare_sam31_gpu_canary
from .sam31_paid_attempt_authority import (
    consume_sam31_paid_attempt_authority_once,
    validate_sam31_paid_attempt_authority,
)
from .sam31_vast_source_track_canary import run_sam31_vast_source_track_canary
from .task_evaluation_artifact_manifest import (
    seal_lane_terminal_artifacts,
    seal_unallocated_provider_teardown,
)
from .vast_independent_watchdog_control import (
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
    close_independent_vast_watchdog_without_allocation,
)
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _read_private_secret(path_value: str | Path | None) -> tuple[str, list[str]]:
    path = Path(str(path_value or "")).expanduser()
    blockers: list[str] = []
    if not str(path_value or "").strip() or path.is_symlink() or not path.is_file():
        return "", ["sam31_hf_token_file_missing_or_unsafe"]
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return "", ["sam31_hf_token_file_unreadable"]
    # The canonical service secret is ``0640 root:blueprint``.  Owner-only
    # ``0600`` remains valid for hermetic/local runs; group write/execute and
    # every world permission remain forbidden.
    if mode & 0o027:
        blockers.append("sam31_hf_token_file_permissions_not_0600")
    if not value or len(value) > 4096 or "\n" in value or "\r" in value:
        blockers.append("sam31_hf_token_invalid")
    return value, blockers


def _write_terminal_result(
    adapter_path: Path,
    result: dict[str, Any],
    *,
    extra_artifact_roots: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Seal one execute-path result using the shared Task Evaluation contract."""

    attempt_root = adapter_path.parent
    result.setdefault("retry_cap", 0)
    if result.get("provider_mutations_performed") == 0:
        result.setdefault("continuing_spend_from_this_run", False)
    provider_run = attempt_root / "vast_provider_run"
    provider_run.mkdir(parents=True, exist_ok=True)
    if result.get("provider_mutations_performed") == 0:
        seal_unallocated_provider_teardown(
            provider_run,
            reason="semantic_sam31_source_tracks_no_provider_allocation",
        )
    else:
        source_teardown = Path(
            str(result.get("source_teardown_receipt_path") or "")
        ).expanduser()
        source = _load_object(source_teardown) if source_teardown.is_file() else {}
        watchdog = result.get("independent_watchdog")
        watchdog = watchdog if isinstance(watchdog, dict) else {}
        if source or watchdog.get("status") == "provider_terminal":
            instance_id = result.get("instance_id") or source.get("instance_id")
            write_json(
                provider_run / "vast_teardown_manifest.json",
                {
                    "schema_version": "vast_teardown_manifest.v1",
                    "status": (
                        "completed"
                        if result.get("continuing_spend_from_this_run") is False
                        else "blocked"
                    ),
                    "vast_instance_ids": (
                        [int(instance_id)] if str(instance_id or "").isdigit() else []
                    ),
                    "continuing_spend_from_this_run": result.get(
                        "continuing_spend_from_this_run"
                    ),
                    "source_teardown_receipt_path": (
                        str(source_teardown.resolve()) if source else None
                    ),
                    "source_teardown_receipt_digest": source.get(
                        "teardown_receipt_digest"
                    ),
                    "independent_watchdog_status": watchdog.get("status"),
                },
            )
    write_json(
        provider_run / "vast_provider_adapter_result.json",
        {
            "schema_version": "semantic_sam31_allocator_adapter_result.v1",
            "status": result.get("status"),
            "provider_mutations_performed": result.get(
                "provider_mutations_performed", 0
            ),
            "instance_id": result.get("instance_id"),
            "provider_zero_verified": result.get("provider_zero_verified"),
            "blockers": list(result.get("blockers") or []),
            "raw_secret_values_recorded": False,
        },
    )
    sealed = seal_lane_terminal_artifacts(
        result,
        attempt_root=attempt_root,
        lane="semantic_sam31_source_tracks",
        extra_artifact_roots=extra_artifact_roots,
    )
    sealed["execution_result_digest"] = canonical_digest(
        sealed, digest_field="execution_result_digest"
    )
    write_json(adapter_path, sealed)
    return sealed


def run_sam31_paid_resource_allocator_lane(
    args: Any,
    *,
    checkout_commit: str,
    prepare: Callable[..., dict[str, Any]] = prepare_sam31_gpu_canary,
    provider_factory: Callable[[str], Any] = get_render_provider,
    execute_canary: Callable[..., dict[str, Any]] = run_sam31_vast_source_track_canary,
    stage_bundle: Callable[..., dict[str, Any]] = stage_wam_provider_bundle_object_store,
    cleanup_bundle: Callable[..., dict[str, Any]] = cleanup_staged_wam_provider_objects,
    arm_watchdog: Callable[..., tuple[dict[str, Any], Any]] = arm_independent_vast_watchdog,
    close_watchdog: Callable[..., dict[str, Any]] = close_independent_vast_watchdog,
    close_watchdog_without_allocation: Callable[..., dict[str, Any]] = (
        close_independent_vast_watchdog_without_allocation
    ),
) -> dict[str, Any]:
    """Admit and optionally execute one Vast-first semantic track canary."""

    request_path = Path(args.provider_launch_request).expanduser()
    request = _load_object(request_path) if request_path.is_file() else {}
    authority: dict[str, Any] | None = None
    bundle_receipt: dict[str, Any] | None = None
    bundle_path = Path(str(getattr(args, "sam31_input_bundle", None) or "")).expanduser().resolve()
    authority_path = getattr(args, "sam31_attempt_authority", None)
    if authority_path:
        authority = _load_object(authority_path)
        bundle_receipt = _load_object(args.sam31_input_bundle_receipt)
        validate_sam31_paid_attempt_authority(
            authority,
            request=request,
            bundle_path=bundle_path,
            bundle_receipt=bundle_receipt,
            blueprint_commit=checkout_commit,
            max_hourly_rate_usd=args.sam31_max_hourly_rate_usd,
            hard_cap_usd=args.sam31_max_spend_usd,
            hard_ttl_seconds=args.sam31_hard_ttl_seconds,
            allowed_active_instance_ids=args.sam31_allowed_active_vast_instance_id,
        )
    authority_id = authority.get("request_authority_id") if authority else args.sam31_authority_id
    preflight_path = args.preflight_bundle
    handoff: dict[str, Any] | None = None
    handle = None
    if args.execute:
        adapter_path = Path(args.adapter_output).expanduser().resolve()
        handoff, handle = arm_watchdog(
            job_dir=adapter_path.parent,
            max_live_minutes=max(2, (args.sam31_hard_ttl_seconds + 59) // 60),
            generated_at=utc_now_iso(),
            allowed_active_instance_ids=getattr(
                args, "sam31_allowed_active_vast_instance_id", []
            ),
            pod_name_prefix="blueprint-sam31-source-tracks-",
        )
        if handle is None:
            result = {
                "schema_version": "semantic_sam31_gpu_canary_adapter_result.v1",
                "status": "blocked",
                "blockers": ["sam31_independent_watchdog_not_armed"],
                "provider_mutations_performed": 0,
                "paid_execution_started": False,
            }
            return _write_terminal_result(Path(args.adapter_output), result)
        watchdog_snapshot = {
            "status": "armed",
            "independent_process": True,
            "pid": handoff["watchdog_pid"],
            "started_epoch": handoff["watchdog_started_epoch"],
            "deadline_epoch": handoff["watchdog_deadline_epoch"],
            "name_prefix": handoff["pod_name_prefix"],
            "started_instance_id_path": str(handle.started_instance_id_path),
        }
        try:
            provider = provider_factory(args.provider)
            preflight = collect_sam31_vast_preflight(
                name_prefix="blueprint-sam31-source-tracks-",
                container_disk_bytes=80 * 1024**3,
                watchdog=watchdog_snapshot,
                conflicting_owner_present=False,
                capacity_probe=provider.capacity_preflight,
                inventory_probe=lambda prefix: provider.billable_inventory(name_prefix=prefix),
                max_hourly_rate_usd=args.sam31_max_hourly_rate_usd,
            )
        except (OSError, RuntimeError, ValueError):
            close_watchdog_without_allocation(job_dir=adapter_path.parent, handle=handle)
            result = {
                "schema_version": "semantic_sam31_gpu_canary_adapter_result.v1",
                "status": "blocked",
                "blockers": ["sam31_live_preflight_collection_failed"],
                "provider_mutations_performed": 0,
                "paid_execution_started": False,
            }
            return _write_terminal_result(adapter_path, result)
        execution_preflight = (
            Path(args.adapter_output).expanduser().resolve().parent
            / "sam31_execution_preflight.json"
        )
        write_json(execution_preflight, preflight)
        preflight_path = str(execution_preflight)

    admission = prepare(
        request_path=args.provider_launch_request,
        preflight_path=preflight_path,
        admission_out=args.admission_out,
        bound_request_out=args.bound_request_out,
        adapter_output=args.adapter_output,
        provider=args.provider,
        expected_source_commit=args.expected_source_commit or "",
        checkout_source_commit=checkout_commit,
        checkout_clean=True,
        max_spend_usd=args.sam31_max_spend_usd,
        hard_ttl_seconds=args.sam31_hard_ttl_seconds,
        retry_cap=args.sam31_retry_cap,
        authority_id=authority_id,
        execute=args.execute,
        execution_adapter_qualified=args.execute,
    )
    if not args.execute or admission.get("status") != "execute_ready":
        if handle is not None:
            close_watchdog_without_allocation(
                job_dir=Path(args.adapter_output).expanduser().resolve().parent,
                handle=handle,
            )
        if not args.execute:
            return admission
        return _write_terminal_result(
            Path(args.adapter_output).expanduser().resolve(), admission
        )

    blockers: list[str] = []
    hf_token, token_blockers = _read_private_secret(args.sam31_hf_token_file)
    blockers.extend(token_blockers)
    if authority is None:
        blockers.append("sam31_file_backed_paid_authority_missing")
    consumption = (
        consume_sam31_paid_attempt_authority_once(authority, blueprint_commit=checkout_commit)
        if authority is not None and not blockers
        else {"status": "not_consumed"}
    )
    if consumption.get("status") != "consumed":
        blockers.extend(consumption.get("blockers") or ["sam31_paid_authority_not_consumed"])

    adapter_path = Path(args.adapter_output).expanduser().resolve()
    staging_dir = adapter_path.parent / "object_store_staging"
    staging: dict[str, Any] = {"status": "not_started"}
    if not blockers:
        try:
            staging = stage_bundle(
                job_dir=staging_dir,
                bundle_path=bundle_path,
                key_prefix="blueprint/adp/sam31-source-tracks",
                output_content_type="application/json",
                expiration_seconds=args.sam31_hard_ttl_seconds + 1_800,
            )
        except (OSError, RuntimeError, ValueError):
            staging = {
                "status": "blocked",
                "blockers": ["sam31_object_store_staging_failed"],
            }
        if staging.get("status") != "completed":
            blockers.extend(staging.get("blockers") or ["sam31_object_store_staging_blocked"])

    paid_admission = build_paid_lane_admission(
        resource_class="gpu_render",
        blockers=[*list(admission.get("blockers") or []), *blockers],
    )
    ensure_dir(adapter_path.parent)
    write_json(adapter_path.parent / "sam31_paid_lane_admission.json", paid_admission)
    try:
        grant = require_paid_resource_admission(
            paid_admission,
            resource_class="gpu_render",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
    except PaidResourceAdmissionBlocked as exc:
        result = {
            "schema_version": "semantic_sam31_gpu_canary_adapter_result.v1",
            "status": "blocked",
            "blockers": sorted(set(exc.blockers + blockers)),
            "provider_mutations_performed": 0,
            "cost_usd": 0.0,
            "raw_secret_values_recorded": False,
            "scientific_qualification_inferred": False,
            "proof_effect": "none",
            "claim_ceiling": "no_execution_evidence",
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        }
        try:
            cleanup = cleanup_bundle(staging_dir) if staging.get("status") != "not_started" else {}
        except (OSError, RuntimeError, ValueError):
            cleanup = {"all_objects_absent": False, "blockers": ["sam31_cleanup_failed"]}
        result["authorization_consumption"] = consumption
        result["all_staged_objects_absent"] = cleanup.get("all_objects_absent", True)
        if handle is not None:
            result["independent_watchdog"] = close_watchdog_without_allocation(
                job_dir=adapter_path.parent,
                handle=handle,
            )
        return _write_terminal_result(adapter_path, result)

    try:
        result = execute_canary(
            bound_request=_load_object(args.bound_request_out),
            preflight=_load_object(preflight_path),
            job_dir=adapter_path.parent / "sam31_vast_source_track_canary",
            input_bundle_get_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
            output_put_url=(staging_dir / "provider_output_put_url.txt").read_text().strip(),
            output_get_url=(staging_dir / "provider_output_get_url.txt").read_text().strip(),
            hf_token=hf_token,
            provider=provider,
            paid_resource_admission_grant=grant,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        result = {
            "schema_version": "semantic_sam31_vast_source_track_execution.v1",
            "status": "failed",
            "instance_id": None,
            "provider_mutations_performed": 0,
            "provider_zero_verified": False,
            "blockers": [f"sam31_canary_failed:{redacted_failure_detail(exc)}"],
            "raw_secret_values_recorded": False,
            "scientific_qualification_inferred": False,
            "proof_effect": "none",
        }
    finally:
        try:
            cleanup = cleanup_bundle(staging_dir)
        except (OSError, RuntimeError, ValueError):
            cleanup = {"all_objects_absent": False, "blockers": ["sam31_cleanup_failed"]}
    instance_id = result.get("instance_id")
    if instance_id is None and handle.started_instance_id_path.is_file():
        candidate = handle.started_instance_id_path.read_text(encoding="utf-8").strip()
        instance_id = candidate or None
    if instance_id is not None:
        result["instance_id"] = instance_id
        result["provider_mutations_performed"] = max(
            1, int(result.get("provider_mutations_performed") or 0)
        )
    teardown_path = (
        adapter_path.parent
        / "sam31_vast_source_track_canary"
        / "teardown_receipt.json"
    )
    teardown = _load_object(teardown_path) if teardown_path.is_file() else {}
    watchdog = close_watchdog(
        job_dir=adapter_path.parent,
        handle=handle,
        instance_ids=[int(instance_id)] if str(instance_id or "").isdigit() else [],
        provider_teardown_completed=result.get("provider_zero_verified") is True
        or teardown.get("status") == "PASS",
        provider_allocation_impossible=instance_id is None
        and result.get("provider_mutations_performed") == 0,
    )
    result["authorization_consumption"] = consumption
    result["object_store_cleanup_path"] = str(
        staging_dir / "wam_provider_object_store_cleanup.json"
    )
    result["source_teardown_receipt_path"] = str(teardown_path)
    runtime_artifact_path = (
        adapter_path.parent
        / "sam31_vast_source_track_canary"
        / "provider_runtime_result.json"
    )
    normalized_tracks_path = (
        adapter_path.parent
        / "sam31_vast_source_track_canary"
        / "semantic_source_track_import_result.v1.json"
    )
    result["source_track_import_result_path"] = (
        str(normalized_tracks_path) if normalized_tracks_path.is_file() else None
    )
    result["retry_cap"] = 0
    result["all_staged_objects_absent"] = cleanup.get("all_objects_absent")
    result["watchdog_receipt_path"] = str(
        adapter_path.parent
        / "independent_vast_watchdog"
        / "groot_oscar_runpod_canary_watchdog.json"
    )
    result["independent_watchdog"] = watchdog
    result["continuing_spend_from_this_run"] = not (
        result.get("provider_zero_verified") is True
        and watchdog.get("status") == "provider_terminal"
        and cleanup.get("all_objects_absent") is True
    )
    if cleanup.get("all_objects_absent") is not True:
        result["status"] = "failed"
        result.setdefault("blockers", []).append("sam31_object_store_zero_not_proven")
    if watchdog.get("status") != "provider_terminal":
        result["status"] = "failed"
        result.setdefault("blockers", []).append("sam31_watchdog_not_terminal")
    result["blockers"] = sorted(set(result.get("blockers") or []))
    extra_artifact_roots = {
        "sam31_runtime_result": runtime_artifact_path,
        "sam31_normalized_source_tracks": normalized_tracks_path,
        "sam31_source_teardown_receipt": teardown_path,
        "sam31_watchdog_receipt": Path(result["watchdog_receipt_path"]),
        "sam31_object_store_cleanup": Path(result["object_store_cleanup_path"]),
    }
    return _write_terminal_result(
        adapter_path,
        result,
        extra_artifact_roots=extra_artifact_roots,
    )


__all__ = ["run_sam31_paid_resource_allocator_lane"]
