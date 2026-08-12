"""Fail-closed environment guard for production runtime units."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .common import parse_bool
from .launch_proof_policy import PRODUCTION_MODE, launch_proof_mode
from .spend_authority_ledger_migration import (
    SpendAuthorityLedgerError,
    reconcile_spend_authority_ledger,
)

SCHEMA_VERSION = "blueprint.production_runtime_env_guard.v1"

REQUIRED_TRUE_FLAGS = (
    "PRIVACY_PIPELINE_ENABLED",
    "PRIVACY_FAIL_CLOSED",
    "PIPELINE_SYNC_REQUIRED",
    "RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO",
)

# Every module a control-plane host executes. An entrypoint that cannot be
# imported cannot allocate, release, reconcile, or guard spend, and a base
# install without the `runtime` extra is enough to break one: the allocator
# reaches `cv2` transitively through the excision audit.
#
# `paid_resource_allocator` is here even though no unit runs it directly -- the
# dispatcher and release worker invoke it as a subprocess, which is exactly how
# its missing dependency stayed invisible until a stranded provider record
# needed releasing. A test pins this tuple against `deploy/systemd/*.service`
# so a new unit cannot ship an unchecked entrypoint.
CONTROL_PLANE_ENTRYPOINTS = (
    "blueprint_pipeline.paid_resource_allocator",
    "blueprint_pipeline.live_pipeline_control_plane",
    "blueprint_pipeline.live_pipeline_intake_service",
    "blueprint_pipeline.live_pipeline_manifest_alert",
    "blueprint_pipeline.production_gpu_campaign_control_plane",
    "blueprint_pipeline.production_gpu_worker_agent",
    "blueprint_pipeline.production_gpu_worker_pool",
    "blueprint_pipeline.production_runtime_env_guard",
    "blueprint_pipeline.provider_billing_reconciler",
    "blueprint_pipeline.pubsub_handoff_listener",
    "blueprint_pipeline.task_evaluation_launch_dispatcher",
    "blueprint_pipeline.task_evaluation_launch_reconciler",
    "blueprint_pipeline.task_evaluation_launch_supervisor",
    "blueprint_pipeline.task_evaluation_terminal_resource_release",
)

ENTRYPOINT_REMEDIATION = (
    "Install the control-plane dependency set from the deployed checkout: "
    'pip install -e ".[runtime]"'
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_control_plane_entrypoints(
    import_module: Callable[[str], Any],
) -> tuple[dict[str, Any], list[str]]:
    """Import every control-plane entrypoint and report all failures.

    Every module is attempted even after one fails, so a single missing
    dependency does not hide the rest behind it.
    """
    failed: list[dict[str, Any]] = []
    for module in CONTROL_PLANE_ENTRYPOINTS:
        try:
            import_module(module)
        except BaseException as exc:  # noqa: BLE001 - any import failure blocks
            failed.append({
                "module": module,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    detail = {
        "checked": list(CONTROL_PLANE_ENTRYPOINTS),
        "importable": not failed,
        "failed": failed,
        "remediation": ENTRYPOINT_REMEDIATION,
    }
    blockers = [f"control_plane_entrypoint_not_importable:{item['module']}" for item in failed]
    return detail, blockers


def _check_spend_authority_ledger(
    reconcile: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Adopt a ledger left at a previous root, or block the unit from starting.

    Binding ``BLUEPRINT_SPEND_AUTHORITY_ROOT`` on a host that was already
    running moves the single-use consumption ledger and leaves its records
    behind, so the new root reads empty and every authorization spent under the
    old root looks unspent. Observed in production after PR #453 deployed.

    Reconciling here rather than only in the installer means it holds for every
    start of every host, including one restored from an image or rebuilt by a
    path that never ran the installer.
    """
    try:
        receipt = reconcile()
    except SpendAuthorityLedgerError as exc:
        return (
            {"status": "blocked", "error": str(exc)},
            [f"spend_authority_ledger_not_reconciled:{exc}"],
        )
    except OSError as exc:
        return (
            {"status": "blocked", "error": str(exc)},
            ["spend_authority_ledger_not_reconciled:unwritable_root"],
        )
    return (receipt, [])


def build_production_runtime_env_guard(
    env: Mapping[str, str] | None = None,
    import_module: Callable[[str], Any] | None = None,
    reconcile_spend_authority: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source = os.environ if env is None else env
    blockers: list[str] = []
    mode = launch_proof_mode(source)
    if mode != PRODUCTION_MODE:
        blockers.append("missing_BLUEPRINT_LAUNCH_PROOF_MODE_production")

    flag_status: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_TRUE_FLAGS:
        raw = source.get(name)
        enabled = parse_bool(raw, default=False)
        flag_status[name] = {
            "configured": raw is not None and str(raw).strip() != "",
            "enabled": enabled,
        }
        if not enabled:
            blockers.append(f"missing_or_false_{name}")

    entrypoints, entrypoint_blockers = _check_control_plane_entrypoints(
        importlib.import_module if import_module is None else import_module
    )
    blockers.extend(entrypoint_blockers)

    ledger, ledger_blockers = _check_spend_authority_ledger(
        reconcile_spend_authority_ledger
        if reconcile_spend_authority is None
        else reconcile_spend_authority
    )
    blockers.extend(ledger_blockers)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "launch_proof_mode": mode or None,
        "required_true_flags": flag_status,
        "control_plane_entrypoints": entrypoints,
        "spend_authority_ledger": ledger,
        "claim_boundary": (
            "This guard verifies production fail-closed runtime posture, that "
            "every control-plane entrypoint imports, and that no spend-authority "
            "ledger is stranded at a previous root. It is not proof of "
            "deployed health, Pub/Sub message consumption, WebApp forwarding, "
            "buyer delivery, simulator execution, or live provider success."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the full guard report.")
    args = parser.parse_args(argv)
    report = build_production_runtime_env_guard()
    if args.json or report["status"] != "ready":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("[production-runtime-env-guard] status=ready")
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
