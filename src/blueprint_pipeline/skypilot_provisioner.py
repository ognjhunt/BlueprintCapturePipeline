"""SkyPilot pilot adapter — out-of-process, grant-gated, one lane only.

Runs a pinned SkyPilot CLI (installed in a separate venv by
``scripts/setup_skypilot_venv.sh``; the package is never imported into or
installed in the Blueprint runtime — a dry-run install would downgrade
Click/Uvicorn/Pillow) for exactly one lane: cold, disposable, on-demand Vast
smoke work. RunPod, warm reuse, model volumes, retained sessions, spot
recovery, and the parallel readiness race stay on the existing custom path
until `skypilot_promotion_gate` evidence proves parity.

Money-truth boundaries kept custom, per the C4 audit:

- Mutations require a ``PaidResourceAdmissionGrant`` for resource class
  ``skypilot_vast_pilot`` issued by the canonical allocator; without it the
  call is blocked before any subprocess runs
  (marker: skypilot_pilot_launch_disabled).
- Every launch opens ``pending_teardown.v1`` before the mutation; SkyPilot's
  ``sky down`` is only an attempt — teardown is proven exclusively by a
  provider-API inventory (``status_source="provider_api"``, provider-zero),
  because SkyPilot documents that provisioning failures may leave clusters
  and ``--purge`` merely forgets local state.
- An unproven teardown keeps the registry record open as open billing risk
  for ``paid_lane_guard reap-orphans``; SkyPilot state loss never orphans a
  paid instance silently.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import os

import yaml

from .paid_lane_guard import (
    bind_pending_teardown_instance,
    close_pending_teardown,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
)
from .paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .provider_reliability_manifest import (
    TEARDOWN_STATUS_SOURCE_PROVIDER_API,
    build_teardown_proof,
)

SCHEMA_VERSION = "skypilot_pilot_provision.v1"
SKYPILOT_RESOURCE_CLASS = "skypilot_vast_pilot"
SKYPILOT_PILOT_LANES = frozenset({"vast_disposable_smoke"})
SKYPILOT_BIN_ENV = "BLUEPRINT_SKYPILOT_BIN"

_SUBPROCESS_TIMEOUT_SECONDS = 3600

Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]
VastInventory = Callable[[], Mapping[str, Any]]


def _default_runner(command: list[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(  # noqa: S603 - pinned binary from explicit env config
        command,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _result(status: str, **fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "lane": "vast_disposable_smoke",
        "allocation_created": False,
        "allocation_outcome_ambiguous": False,
        "spend_occurred": False,
        "blockers": [],
        "pending_teardown_path": None,
        "teardown_proof": None,
    }
    base.update(fields)
    return base


def _validate_task_yaml(task_yaml_path: Path) -> list[str]:
    blockers: list[str] = []
    try:
        loaded = yaml.safe_load(task_yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return ["skypilot_task_yaml_unreadable"]
    resources = loaded.get("resources") if isinstance(loaded, Mapping) else None
    if not isinstance(resources, Mapping):
        return ["skypilot_task_resources_missing"]
    if str(resources.get("cloud") or "").strip().lower() != "vast":
        blockers.append("skypilot_task_cloud_not_vast")
    try:
        cap = float(resources.get("max_hourly_cost"))
    except (TypeError, ValueError):
        cap = 0.0
    if cap <= 0:
        blockers.append("skypilot_task_max_hourly_cost_missing")
    return blockers


def _provider_zero_proof(
    *,
    cluster_name: str,
    vast_inventory: VastInventory | None,
) -> dict[str, Any]:
    provider_terminal_status: str | None = None
    status_source: str | None = None
    verified_at: str | None = None
    if vast_inventory is not None:
        try:
            inventory = dict(vast_inventory())
        except Exception:  # noqa: BLE001 - unverifiable inventory == unproven teardown
            inventory = {}
        if inventory.get("status") == "passed":
            rows = [
                row
                for row in inventory.get("instances") or []
                if isinstance(row, Mapping)
                and str(row.get("label") or row.get("resource_name") or "")
                == cluster_name
            ]
            status_source = TEARDOWN_STATUS_SOURCE_PROVIDER_API
            verified_at = datetime.now(timezone.utc).isoformat()
            if not rows:
                provider_terminal_status = "not_found"
            else:
                provider_terminal_status = str(
                    rows[0].get("actual_status") or "unknown"
                )
    return build_teardown_proof(
        provider="vast",
        allocation_id=cluster_name,
        terminate_requested=True,
        provider_terminal_status=provider_terminal_status,
        verified_at=verified_at,
        status_source=status_source,
    )


def launch_disposable_vast_smoke(
    *,
    run_id: str,
    task_yaml_path: str | Path,
    cluster_name: str,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None = None,
    env: Mapping[str, str] | None = None,
    runner: Runner | None = None,
    vast_inventory: VastInventory | None = None,
    registry_dir: Path | None = None,
) -> dict[str, Any]:
    """Launch → run → ``sky down`` → provider-API provider-zero proof."""

    try:
        require_paid_resource_admission_grant(
            paid_resource_admission_grant,
            resource_class=SKYPILOT_RESOURCE_CLASS,
        )
    except PaidResourceAdmissionBlocked as exc:
        return _result(
            "blocked",
            blockers=["skypilot_pilot_launch_disabled", *exc.blockers],
        )

    mapping = _env(env)
    sky_bin = str(mapping.get(SKYPILOT_BIN_ENV) or "").strip()
    if not sky_bin or not Path(sky_bin).is_file():
        return _result("unavailable", blockers=["skypilot_bin_missing"])

    task_yaml = Path(task_yaml_path)
    constraint_blockers = _validate_task_yaml(task_yaml)
    if constraint_blockers:
        return _result("blocked_constraints", blockers=constraint_blockers)

    active_runner = runner or _default_runner

    record = open_pending_teardown(
        provider="vast",
        lane="skypilot_vast_pilot",
        run_id=run_id,
        resource_kind="compute_instance",
        resource_name=cluster_name,
        registry_dir=registry_dir,
    )
    record_path = Path(record["path"])

    launch = active_runner(
        [
            sky_bin,
            "launch",
            "-y",
            "--down",
            "-c",
            cluster_name,
            str(task_yaml),
        ]
    )
    launch_ok = int(getattr(launch, "returncode", 1)) == 0
    if launch_ok:
        allocation_created: bool | None = True
        allocation_ambiguous = False
        spend_occurred: bool | None = True
        bind_pending_teardown_instance(record_path, cluster_name)
    else:
        # SkyPilot documents that provisioning/setup failures may leave the
        # cluster for debugging — a nonzero exit is ambiguous, not "nothing
        # was created".
        allocation_created = None
        allocation_ambiguous = True
        spend_occurred = None
        mark_pending_teardown_ambiguous(
            record_path,
            reason="sky_launch_nonzero_exit_cluster_may_exist",
            evidence={"returncode": int(getattr(launch, "returncode", 1))},
        )

    down = active_runner([sky_bin, "down", "-y", cluster_name])

    proof = _provider_zero_proof(
        cluster_name=cluster_name, vast_inventory=vast_inventory
    )
    close_pending_teardown(record_path, proof)

    proof_passed = proof.get("status") == "PASS"
    if launch_ok:
        status = "completed_provider_zero" if proof_passed else "launched_teardown_unproven"
    else:
        status = (
            "launch_ambiguous_provider_zero"
            if proof_passed
            else "launch_ambiguous_teardown_unproven"
        )
    blockers = list(proof.get("blockers") or [])
    if int(getattr(down, "returncode", 1)) != 0:
        blockers.append("sky_down_nonzero_exit")
    return _result(
        status,
        allocation_created=allocation_created,
        allocation_outcome_ambiguous=allocation_ambiguous,
        spend_occurred=spend_occurred,
        blockers=blockers,
        pending_teardown_path=str(record_path),
        teardown_proof=proof,
        sky_launch_returncode=int(getattr(launch, "returncode", 1)),
        sky_down_returncode=int(getattr(down, "returncode", 1)),
        cluster_name=cluster_name,
    )


__all__ = [
    "SCHEMA_VERSION",
    "SKYPILOT_RESOURCE_CLASS",
    "SKYPILOT_PILOT_LANES",
    "SKYPILOT_BIN_ENV",
    "launch_disposable_vast_smoke",
]
