"""Execute the persistent carrier campaign only after canonical GPU admission."""

from __future__ import annotations

import json
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import write_json
from .groot_oscar_infrastructure_admission import SERVE_SCHEMA_VERSION
from .groot_oscar_runpod_canary import (
    _reserve_campaign_budget,
    _settle_zero_budget,
    _write_private_json,
    refresh_runpod_preflight,
)
from .gpu_render_providers import _runpod_call, get_render_provider
from .groot_oscar_runpod_persistent_carrier import (
    PERSISTENT_LOOP_MAX_WAIT_SECONDS,
    prepare_persistent_carrier_launch,
)
from .paid_provider_lane_lease import (
    accept_paid_provider_lane_lease_handoff,
    restore_paid_provider_lane_lease_to_retained_watchdog,
)
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .production_gpu_campaign_budget import ProductionGpuCampaignBudget
from .unitree_groot_n17_sonic_vast_persistent_session import (
    run_persistent_session_runpod,
)


EXPECTED_MEDIA_FILE_COUNT = 21
_MEDIA_SUFFIXES = frozenset({".gif", ".jpeg", ".jpg", ".mp4", ".png", ".webm"})


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(value)


def _settle_reserved_maximum(
    context: Mapping[str, Any], *, outcome: str
) -> dict[str, Any]:
    """Settle conservatively at the reserved maximum when provider billing is absent."""

    if context.get("status") != "reserved":
        return {"status": "not_reserved"}
    identity = context["identity"]
    reservation = context["reservation"]
    budget = ProductionGpuCampaignBudget(
        context["ledger_path"],
        initial_spent_usd=identity["initial_spent_usd"],
        initial_used_gpu_seconds=identity["initial_used_gpu_seconds"],
        total_spend_cap_usd=identity["total_spend_cap_usd"],
        combined_gpu_wall_cap_seconds=identity["combined_gpu_wall_cap_seconds"],
    )
    return budget.settle(
        reservation_id=str(context["reservation_id"]),
        charged_gpu_seconds=int(reservation["reserved_gpu_seconds"]),
        charged_usd=float(reservation["reserved_usd"]),
        outcome=outcome,
    )


def _receipt_mutation_started(receipt_path: Path) -> bool:
    """Treat unreadable receipt state as ambiguous provider mutation."""

    if not receipt_path.is_file() or receipt_path.is_symlink():
        return True
    try:
        receipt = _read(receipt_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return True
    return bool(
        receipt.get("pre_provider_mutation_confirmed_absent") is not True
        or receipt.get("pod_pending_teardown_record")
        or receipt.get("pod_id")
    )


def _retain_watchdog_receipt(receipt_path: Path, result: Mapping[str, Any]) -> None:
    try:
        receipt = _read(receipt_path)
    except (OSError, ValueError, json.JSONDecodeError):
        receipt = {}
    _write_private_json(receipt_path, {**receipt, "campaign_result": dict(result)})


def _remove_terminal_receipt(
    receipt_path: Path,
    *,
    owner_return: Mapping[str, Any],
    settlement: Mapping[str, Any],
) -> bool:
    if owner_return.get("status") != "restored" or settlement.get("status") != "settled":
        return False
    receipt_path.unlink(missing_ok=True)
    return True


def audit_persistent_carrier_output(
    session_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the exact June-30 technical topology without upgrading task claims."""

    blockers: list[str] = []
    expected_counts = {
        "repeated_policy_calls_count": 5,
        "generated_next_observation_count": 4,
        "live_wam_generation_success_count": 4,
        "learned_wam_model_success_count": 4,
    }
    observed_counts: dict[str, int] = {}
    for field, expected in expected_counts.items():
        try:
            observed = int(session_result.get(field) or 0)
        except (TypeError, ValueError):
            observed = 0
        observed_counts[field] = observed
        if observed != expected:
            blockers.append(f"persistent_carrier_exact_{field}_not_proven")
    for field in (
        "persistent_provider_session_used",
        "provider_instance_reused_for_policy_and_wam_loop",
    ):
        if session_result.get(field) is not True:
            blockers.append(f"persistent_carrier_{field}_not_proven")
    if session_result.get("provider_output_replay_used") is not False:
        blockers.append("persistent_carrier_provider_output_replay_disallowed")
    if session_result.get("provider_output_resume_used") is not False:
        blockers.append("persistent_carrier_provider_output_resume_disallowed")

    imported_text = str(session_result.get("imported_provider_output_dir") or "")
    imported = Path(imported_text).expanduser() if imported_text else Path()
    policy_paths = (
        sorted((imported / "policy_calls").glob("policy_call_*.json"))
        if imported_text and imported.is_dir()
        else []
    )
    wam_paths = (
        sorted((imported / "wam_calls").glob("wam_call_*.json"))
        if imported_text and imported.is_dir()
        else []
    )
    action_digests: list[str] = []
    for path in policy_paths:
        try:
            row = _read(path)
        except (OSError, ValueError, json.JSONDecodeError):
            blockers.append("persistent_carrier_policy_artifact_invalid")
            continue
        action = row.get("action")
        if row.get("status") != "completed" or not isinstance(action, Mapping):
            blockers.append("persistent_carrier_policy_artifact_not_completed")
            continue
        if row.get("provider_output_replay_used") is not False:
            blockers.append("persistent_carrier_policy_artifact_replay_disallowed")
        action_bytes = json.dumps(
            dict(action), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        action_digests.append(hashlib.sha256(action_bytes).hexdigest())
    if len(policy_paths) != 5:
        blockers.append("persistent_carrier_requires_exactly_five_policy_artifacts")
    if len(action_digests) != 5 or len(set(action_digests)) != 5:
        blockers.append("persistent_carrier_requires_five_distinct_actions")
    if len(wam_paths) != 4:
        blockers.append("persistent_carrier_requires_exactly_four_wam_artifacts")
    else:
        for path in wam_paths:
            try:
                row = _read(path)
            except (OSError, ValueError, json.JSONDecodeError):
                blockers.append("persistent_carrier_wam_artifact_invalid")
                continue
            if row.get("status") != "completed":
                blockers.append("persistent_carrier_wam_artifact_not_completed")
            if row.get("structural_fallback_used") is True:
                blockers.append("persistent_carrier_structural_wam_fallback_disallowed")
    media_paths = (
        sorted(
            path
            for path in imported.rglob("*")
            if path.is_file() and path.suffix.lower() in _MEDIA_SUFFIXES
        )
        if imported_text and imported.is_dir()
        else []
    )
    if len(media_paths) != EXPECTED_MEDIA_FILE_COUNT:
        blockers.append("persistent_carrier_requires_exactly_21_media_files")
    unique_blockers = sorted(set(blockers))
    return {
        "schema_version": "groot_oscar_persistent_carrier_output_audit.v1",
        "status": "passed" if not unique_blockers else "blocked",
        "blockers": unique_blockers,
        "observed_counts": {
            **observed_counts,
            "policy_artifact_count": len(policy_paths),
            "distinct_action_count": len(set(action_digests)),
            "wam_artifact_count": len(wam_paths),
            "media_file_count": len(media_paths),
        },
        "expected_counts": {
            **expected_counts,
            "policy_artifact_count": 5,
            "distinct_action_count": 5,
            "wam_artifact_count": 4,
            "media_file_count": EXPECTED_MEDIA_FILE_COUNT,
        },
        "action_sha256": action_digests,
        "media_paths": [str(path) for path in media_paths],
        "provider_output_replay_used": session_result.get(
            "provider_output_replay_used"
        ),
        "provider_output_resume_used": session_result.get(
            "provider_output_resume_used"
        ),
        "semantic_task_success_proven": False,
        "raw_secret_values_recorded": False,
    }


def run_persistent_carrier_campaign(
    *,
    provider_launch_request: str | Path,
    release_evidence: str | Path,
    model_cache_evidence: str | Path,
    preflight_bundle: str | Path,
    carrier_volume_admission: str | Path,
    policy_observation_path: str | Path,
    persistent_job_dir: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    pod_name: str,
    execute: bool,
    campaign_budget: Mapping[str, Any] | None = None,
    task_prompt: str | None = None,
    session_runner: Callable[..., tuple[dict[str, Any], int]] = (
        run_persistent_session_runpod
    ),
) -> dict[str, Any]:
    """Validate, reserve, accept one-time ownership, run, and reconcile one Pod."""

    preflight = _read(preflight_bundle)
    refresh_path = (
        Path(adapter_output).resolve().parent / "runpod_preflight_launch_refresh.json"
    )
    if execute:
        provider = get_render_provider("runpod")
        key = provider._key()  # type: ignore[attr-defined]
        if key:

            def volume_getter(volume_id: str) -> tuple[int, Mapping[str, Any]]:
                status, payload = _runpod_call(
                    "GET",
                    f"/networkvolumes/{volume_id}",
                    None,
                    key=key,
                    timeout=30,
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
    carrier = _read(carrier_volume_admission)
    prepared = prepare_persistent_carrier_launch(
        request=_read(provider_launch_request),
        release=_read(release_evidence),
        model_cache=_read(model_cache_evidence),
        preflight=preflight,
        carrier_volume_admission=carrier,
        loop_step_count=5,
        max_wait_seconds=PERSISTENT_LOOP_MAX_WAIT_SECONDS,
    )
    write_json(Path(admission_out), prepared)
    write_json(Path(bound_request_out), prepared["bound_request"])
    if prepared["status"] != "admitted":
        write_json(Path(adapter_output), prepared)
        return prepared
    if not execute:
        result = {
            **prepared,
            "status": "dry_run_ready",
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result

    spend = preflight.get("spend")
    spend = spend if isinstance(spend, Mapping) else {}
    watchdog_prefix = str(spend.get("watchdog_pod_name_prefix") or "").strip()
    watchdog_out_dir = str(spend.get("watchdog_out_dir") or "").strip()
    if not watchdog_out_dir or not Path(watchdog_out_dir).is_absolute():
        result = {
            **prepared,
            "status": "blocked",
            "blockers": ["persistent_carrier_watchdog_out_dir_unverified"],
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    if not watchdog_prefix or not pod_name.startswith(watchdog_prefix):
        result = {
            **prepared,
            "status": "blocked",
            "blockers": ["persistent_carrier_pod_name_outside_watchdog_scope"],
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    budget = _reserve_campaign_budget(
        campaign_budget or {}, reservation_id=pod_name
    )
    write_json(
        Path(adapter_output).resolve().parent / "campaign_budget_reservation.json",
        budget,
    )
    if budget.get("status") != "reserved":
        result = {
            **prepared,
            "status": "blocked",
            "blockers": list(budget.get("blockers") or []),
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    reservation = budget.get("reservation")
    reservation = reservation if isinstance(reservation, Mapping) else {}
    reserved_seconds = int(reservation.get("reserved_gpu_seconds") or 0)
    hard_ttl_seconds = int(spend.get("hard_ttl_seconds") or 0)
    watchdog_remaining_seconds = max(
        0,
        math.ceil(
            float(spend.get("watchdog_deadline_epoch") or 0)
            - float(budget.get("reserved_at_epoch") or time.time())
        ),
    )
    budget["watchdog_contract"] = {
        "hard_ttl_seconds": hard_ttl_seconds,
        "watchdog_deadline_epoch": spend.get("watchdog_deadline_epoch"),
        "watchdog_remaining_seconds_at_reservation": watchdog_remaining_seconds,
        "reserved_gpu_seconds": reserved_seconds,
    }
    if (
        hard_ttl_seconds <= 0
        or hard_ttl_seconds > reserved_seconds
        or watchdog_remaining_seconds > reserved_seconds
    ):
        settlement = _settle_zero_budget(
            budget, outcome="persistent_watchdog_exceeds_reservation_no_mutation"
        )
        result = {
            **prepared,
            "status": "blocked",
            "blockers": ["persistent_carrier_watchdog_exceeds_budget_reservation"],
            "campaign_budget_settlement": settlement,
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    volume_handoff = preflight.get("model_volume_watchdog_handoff")
    volume_handoff = volume_handoff if isinstance(volume_handoff, Mapping) else {}
    lane_handoff = volume_handoff.get("provider_lane_handoff")
    lane_handoff = lane_handoff if isinstance(lane_handoff, Mapping) else {}
    binding = lane_handoff.get("binding")
    binding = binding if isinstance(binding, Mapping) else {}
    admitted_volume_id = prepared["admission"]["network_volume_id"]
    if binding.get("volume_id") != admitted_volume_id:
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
        Path(adapter_output).resolve().parent / "provider_lane_handoff_acceptance.json",
        acceptance,
    )
    if acceptance.get("status") != "accepted":
        settlement = _settle_zero_budget(
            budget, outcome="persistent_carrier_handoff_rejected_no_gpu_create"
        )
        result = {
            **prepared,
            "status": "blocked",
            "blockers": [
                *acceptance.get("blockers", []),
                "paid_provider_lane_handoff_not_accepted",
            ],
            "campaign_budget_settlement": settlement,
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    receipt_path = Path(watchdog_out_dir) / "provider_lane_handoff_receipt.json"
    try:
        require_paid_resource_admission(
            prepared["admission"],
            resource_class="gpu_canary",
            expected_schema_version=SERVE_SCHEMA_VERSION,
        )
        runner_admission = build_paid_lane_admission(
            resource_class="runpod_wam_async", blockers=[]
        )
        runner_grant = require_paid_resource_admission(
            runner_admission,
            resource_class="runpod_wam_async",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
        _write_private_json(
            receipt_path,
            {
                **acceptance,
                "pod_id": None,
                "pod_name_prefix": watchdog_prefix,
                "campaign_budget": budget,
                "campaign_kind": "persistent_policy_wam_loop",
                "pre_provider_mutation_confirmed_absent": True,
                "raw_secret_values_recorded": False,
            },
        )
    except Exception as exc:  # noqa: BLE001 - provider call has not occurred
        owner_return = restore_paid_provider_lane_lease_to_retained_watchdog(
            acceptance
        )
        settlement = _settle_zero_budget(
            budget, outcome="persistent_carrier_pre_provider_failure"
        )
        result = {
            **prepared,
            "status": "blocked",
            "blockers": ["persistent_carrier_pre_provider_admission_failed"],
            "error_type": type(exc).__name__,
            "provider_lane_owner_return": owner_return,
            "campaign_budget_settlement": settlement,
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    try:
        campaign_contract = prepared["admission"]["campaign_contract"]
        request_shape = prepared["bound_request"]["provider_request_shape"]
        session_result, exit_code = session_runner(
            policy_observation_path=policy_observation_path,
            job_dir=persistent_job_dir,
            loop_step_count=5,
            task_prompt=task_prompt,
            timeout_seconds=3600.0,
            use_live_wam=True,
            allow_structural_wam_fallback=False,
            max_wait_seconds=PERSISTENT_LOOP_MAX_WAIT_SECONDS,
            paid_resource_admission_grant=runner_grant,
            carrier_volume_admission=carrier,
            pod_name=pod_name,
            provider_lane_handoff_receipt_path=receipt_path,
            gpu_type_ids=(prepared["admission"]["gpu_type_id"],),
            container_disk_gb=campaign_contract["container_disk_gib"],
            volume_gb=campaign_contract["network_volume_gib"],
            allowed_cuda_versions=tuple(request_shape["allowed_cuda_versions"]),
        )
    except Exception as exc:  # noqa: BLE001 - watchdog retains provider control
        mutation_started = _receipt_mutation_started(receipt_path)
        if mutation_started:
            owner_return = {
                "status": "watchdog_retains_control",
                "restored": False,
            }
            settlement = {"status": "watchdog_retains_open_reservation"}
        else:
            owner_return = restore_paid_provider_lane_lease_to_retained_watchdog(
                acceptance
            )
            settlement = _settle_zero_budget(
                budget, outcome="persistent_carrier_session_exception_no_mutation"
            )
        result = {
            **prepared,
            "status": "blocked",
            "blockers": ["persistent_carrier_session_runner_exception"],
            "error_type": type(exc).__name__,
            "provider_lane_owner_return": owner_return,
            "campaign_budget_settlement": settlement,
            "provider_mutations_performed": int(mutation_started),
        }
        if not _remove_terminal_receipt(
            receipt_path, owner_return=owner_return, settlement=settlement
        ):
            _retain_watchdog_receipt(receipt_path, result)
        write_json(Path(adapter_output), result)
        return result
    session_job_dir = Path(
        str(session_result.get("job_dir") or persistent_job_dir)
    ).expanduser().resolve()
    poll_path = (
        session_job_dir
        / "runpod_persistent_session_run/runpod_wam_async_poll_manifest.json"
    )
    poll = _read(poll_path) if poll_path.is_file() else {}
    output_audit = audit_persistent_carrier_output(session_result)
    output_audit_path = (
        Path(adapter_output).resolve().parent / "persistent_carrier_output_audit.json"
    )
    write_json(output_audit_path, output_audit)
    gpu_terminal = bool(
        poll.get("teardown_performed") is True
        and poll.get("continuing_spend_from_this_run") is False
    )
    mutation_started = _receipt_mutation_started(receipt_path)
    if gpu_terminal:
        owner_return = restore_paid_provider_lane_lease_to_retained_watchdog(acceptance)
        settlement = _settle_reserved_maximum(
            budget, outcome="persistent_carrier_campaign_gpu_terminal"
        )
    elif not mutation_started:
        owner_return = restore_paid_provider_lane_lease_to_retained_watchdog(acceptance)
        settlement = _settle_zero_budget(
            budget, outcome="persistent_carrier_campaign_blocked_no_mutation"
        )
    else:
        owner_return = {"status": "watchdog_retains_control", "restored": False}
        settlement = {"status": "watchdog_retains_open_reservation"}
    completed = bool(
        exit_code == 0
        and session_result.get("status") == "completed"
        and output_audit.get("status") == "passed"
        and gpu_terminal
        and owner_return.get("status") == "restored"
    )
    result = {
        "schema_version": "groot_oscar_runpod_persistent_carrier_campaign.v1",
        "status": "completed" if completed else "blocked",
        "blockers": []
        if completed
        else sorted(
            set(
                [
                    "persistent_carrier_campaign_not_completed",
                    *list(output_audit.get("blockers") or []),
                ]
            )
        ),
        "pod_name": pod_name,
        "session_result_path": str(
            session_job_dir
            / "unitree_groot_n17_sonic_vast_persistent_session_result.json"
        ),
        "poll_manifest_path": str(poll_path),
        "technical_loop_completed": output_audit.get("status") == "passed",
        "persistent_carrier_output_audit_path": str(output_audit_path),
        "persistent_carrier_output_audit": output_audit,
        "gpu_teardown_verified": gpu_terminal,
        "continuing_gpu_spend": poll.get("continuing_spend_from_this_run"),
        "provider_lane_owner_return": owner_return,
        "campaign_budget_settlement": settlement,
        "semantic_task_success_proven": False,
        "provider_mutations_performed": int(mutation_started or gpu_terminal),
        "raw_secret_values_recorded": False,
    }
    if not _remove_terminal_receipt(
        receipt_path, owner_return=owner_return, settlement=settlement
    ):
        _retain_watchdog_receipt(receipt_path, result)
    write_json(Path(adapter_output), result)
    return result
