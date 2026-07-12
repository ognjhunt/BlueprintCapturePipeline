"""Shared fail-closed settlement for known paid-provider allocations."""
from __future__ import annotations

from typing import Any, Callable, Mapping

from .common import utc_now_iso
from .paid_lane_guard import provider_state_from_inspect
from .provider_reliability_manifest import (
    TEARDOWN_STATUS_SOURCE_PROVIDER_API,
    build_teardown_proof,
)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def teardown_proof_from_attempt(
    *,
    provider: Any,
    instance_id: str,
    teardown: Mapping[str, Any],
    action: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    provider_name = _text(getattr(provider, "name", "")) or "unknown"
    status = _text(teardown.get("status")).lower()
    if _text(action).lower() == "stop":
        return build_teardown_proof(
            provider=provider_name,
            allocation_id=instance_id,
            terminate_requested=False,
            provider_terminal_status=None,
            keep_alive_requested=True,
            keep_alive_reason=status or "stopped_for_warm_reuse",
        )
    verification: dict[str, Any] = {}
    if hasattr(provider, "inspect"):
        try:
            verification = provider_state_from_inspect(provider.inspect(instance_id))
        except Exception as exc:  # noqa: BLE001 - failed proof stays fail-closed
            verification = {
                "api_confirmed": False,
                "provider_status": "",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    observed_status = _text(verification.get("provider_status")).lower()
    if verification.get("api_confirmed") is True and observed_status:
        return build_teardown_proof(
            provider=provider_name,
            allocation_id=instance_id,
            terminate_requested=True,
            provider_terminal_status=observed_status,
            verified_at=generated_at or utc_now_iso(),
            status_source=TEARDOWN_STATUS_SOURCE_PROVIDER_API,
        )
    return build_teardown_proof(
        provider=provider_name,
        allocation_id=instance_id,
        terminate_requested=True,
        provider_terminal_status="terminated" if status == "terminated" else status or None,
        verified_at=generated_at or utc_now_iso() if status == "terminated" else None,
    )


def teardown_proof_from_watch_result(
    *, provider_name: str, instance_id: str, watch: Mapping[str, Any]
) -> dict[str, Any]:
    teardown = dict(watch.get("teardown")) if isinstance(watch.get("teardown"), Mapping) else {}
    reason = _text(watch.get("teardown_reason")).lower()
    status = _text(teardown.get("status")).lower()
    if status in {"stopped", "preserved", "skipped"} or reason in {
        "left_running_by_request",
        "runner_done_preserved_for_warm_reuse",
    }:
        return build_teardown_proof(
            provider=provider_name,
            allocation_id=instance_id,
            terminate_requested=False,
            provider_terminal_status=None,
            keep_alive_requested=True,
            keep_alive_reason=reason or status or "kept_alive",
        )
    verification = (
        dict(teardown.get("verification"))
        if isinstance(teardown.get("verification"), Mapping)
        else {}
    )
    observed_status = _text(verification.get("provider_status")).lower()
    if verification.get("api_confirmed") is True and observed_status:
        return build_teardown_proof(
            provider=provider_name,
            allocation_id=instance_id,
            terminate_requested=True,
            provider_terminal_status=observed_status,
            verified_at=utc_now_iso(),
            status_source=TEARDOWN_STATUS_SOURCE_PROVIDER_API,
        )
    return build_teardown_proof(
        provider=provider_name,
        allocation_id=instance_id,
        terminate_requested=True,
        provider_terminal_status="terminated" if status == "terminated" else status or None,
        verified_at=utc_now_iso() if status == "terminated" else None,
    )


def lane_release_passed(release: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(release, Mapping)
        and release.get("all_providers_terminal") is True
        and release.get("results")
        and all(
            item.get("status") in {"released", "already_released"}
            for item in release.get("results") or []
        )
    )


def finalize_known_allocation(
    *,
    provider_obj: Any,
    instance_id: str,
    pending_path: str,
    reason: str,
    teardown_proof_builder: Callable[..., dict[str, Any]],
    close_pending: Callable[..., dict[str, Any]],
    release_lane: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    try:
        teardown = provider_obj.terminate(instance_id)
    except Exception as exc:  # noqa: BLE001 - inspect proof remains authoritative
        teardown = {"status": "terminate_failed", "error_type": type(exc).__name__}
    proof = teardown_proof_builder(
        provider=provider_obj,
        instance_id=instance_id,
        teardown=teardown if isinstance(teardown, Mapping) else {},
        action="terminate",
    )
    closure: dict[str, Any] = {"status": "not_applicable"}
    if pending_path:
        try:
            closure = close_pending(pending_path, proof)
        except Exception as exc:  # noqa: BLE001 - release remains fail-closed
            closure = {"status": "close_failed", "error_type": type(exc).__name__}
    release = release_lane(reason, provider_mutation_started=True)
    return {
        "teardown": teardown,
        "teardown_proof": proof,
        "pending_teardown_close": closure,
        "lane_release": release,
        "terminal": bool(
            str(proof.get("status") or "").upper() == "PASS"
            and closure.get("status") in {"closed", "not_applicable"}
            and lane_release_passed(release)
        ),
    }


def settle_watch_lifecycle(
    *,
    provider_name: str,
    instance_id: str,
    watch: Mapping[str, Any],
    pending_path: str,
    supervised_startup: bool,
    watch_proof_builder: Callable[..., dict[str, Any]],
    close_pending: Callable[..., dict[str, Any]],
    release_lane: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    proof: dict[str, Any] | None = None
    closure: dict[str, Any] = {"status": "not_applicable"}
    if pending_path and not supervised_startup:
        proof = watch_proof_builder(
            provider_name=provider_name,
            instance_id=instance_id,
            watch=watch,
        )
        try:
            closure = close_pending(pending_path, proof)
        except Exception as exc:  # noqa: BLE001 - lifecycle remains blocked
            closure = {"status": "close_failed", "error_type": type(exc).__name__}
    release = release_lane("watch_and_collect_finished", provider_mutation_started=True)
    terminal = bool(
        lane_release_passed(release)
        and (
            not pending_path
            or supervised_startup
            or (
                proof is not None
                and str(proof.get("status") or "").upper() == "PASS"
                and closure.get("status") == "closed"
            )
        )
    )
    return {
        "status": "passed" if terminal else "blocked",
        "terminal_teardown_proven": terminal,
        "teardown_proof": proof,
        "pending_teardown_close": closure,
        "lane_release": release,
    }


def apply_lifecycle_gate(
    manifest: dict[str, Any], lifecycle: Mapping[str, Any]
) -> dict[str, Any]:
    if lifecycle.get("terminal_teardown_proven") is not True:
        manifest["runtime_evidence_status_before_lifecycle_gate"] = manifest.get(
            "status"
        )
        manifest["status"] = "blocked"
        blockers = manifest.setdefault("blockers", [])
        if "paid_provider_lifecycle_not_terminal" not in blockers:
            blockers.append("paid_provider_lifecycle_not_terminal")
    return manifest


def block_unbounded_paid_serve(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest.setdefault("blockers", []).append(
        "paid_warm_serve_requires_bounded_provider_teardown_supervisor"
    )
    manifest["warm_serve_spend_policy"] = {
        "status": "blocked",
        "reason": (
            "serve_idle_timeout ends the runner but does not API-delete the "
            "provider allocation; a finite max-spend claim is unavailable"
        ),
        "standing_spend_authorization_supported": False,
    }
    return manifest


def adopt_launch_result(
    *,
    provider_obj: Any,
    launch: Mapping[str, Any],
    pending_path: str,
    bind_pending: Callable[[str, str], dict[str, Any]],
    close_pending: Callable[..., dict[str, Any]],
    teardown_proof_builder: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Bind a returned id first; clean it if status/binding is not trustworthy."""
    instance_id = str(launch.get("instance_id") or "").strip()
    if not instance_id:
        return {"instance_id": None, "ready": False, "cleanup": None}
    bind_error = None
    if pending_path:
        try:
            bind_pending(pending_path, instance_id)
        except Exception as exc:  # noqa: BLE001 - known id still gets cleanup
            bind_error = type(exc).__name__
    ready = launch.get("status") == "launched" and bind_error is None
    if ready:
        return {"instance_id": instance_id, "ready": True, "cleanup": None}
    cleanup = finalize_known_allocation(
        provider_obj=provider_obj,
        instance_id=instance_id,
        pending_path=pending_path,
        reason="partial_launch_result",
        teardown_proof_builder=teardown_proof_builder,
        close_pending=close_pending,
        release_lane=lambda *_args, **_kwargs: {
            "all_providers_terminal": True,
            "results": [{"status": "released"}],
        },
    )
    cleanup.pop("lane_release", None)
    return {
        "instance_id": instance_id,
        "ready": False,
        "bind_error_type": bind_error,
        "cleanup": cleanup,
    }
