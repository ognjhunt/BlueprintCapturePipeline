"""Pre-spend admission helpers for the persistent G1 kitchen qualification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import write_json
from .groot_oscar_digitalocean_job_inputs import runtime_contract_for_pre_spend
from .lane_hardware_requirements import build_lane_hardware_contract
from .paid_lane_guard import (
    PreSpendPreflightBlocked,
    image_contract_from_ref,
    require_pre_spend_preflight,
)


PRE_SPEND_HARDWARE_LANE = "kitchen_g1_groot_sonic_eval"
PRE_SPEND_PROGRESS_STALL_PHASES = (
    "container_bash_started",
    "inputs_ready",
    "healthcheck_passed",
    "groot_server_ready",
    "isaac_task_executor_ready",
)


def qualification_pre_spend_preflight(
    *,
    root: Path,
    capacity: Mapping[str, Any],
    pre_inventory: Mapping[str, Any],
    image_ref: str,
    execute: bool,
    provider: str = "vast",
) -> tuple[dict[str, Any], list[str]]:
    """Bind an execute request to capacity, hardware, inventory, and spend lock."""

    selected = dict(capacity.get("selected_offer") or {})
    if not selected:
        viable = capacity.get("viable_gpu_types")
        if isinstance(viable, list) and viable and isinstance(viable[0], Mapping):
            selected = dict(viable[0])
    gpu_name = str(
        selected.get("gpu_name")
        or selected.get("gpu_type_id")
        or selected.get("display_name")
        or ""
    ).strip()
    gpu_type_id = gpu_name if gpu_name.startswith("NVIDIA ") else f"NVIDIA {gpu_name}"
    try:
        vram_gb = float(selected.get("gpu_ram_mb")) / 1000.0
    except (TypeError, ValueError):
        vram_gb = None
    hardware_contract = build_lane_hardware_contract(
        lane=PRE_SPEND_HARDWARE_LANE,
        gpu_type_id=gpu_type_id or None,
        vram_gb=vram_gb,
        disk_gb=220.0,
    )
    capacity_available = capacity.get("status") == "available" and bool(selected)
    inventory_zero = (
        pre_inventory.get("api_confirmed") is True
        and pre_inventory.get("live_resource_count") == 0
    )
    # An explicit empty mapping makes the production lock mandatory on execute.
    # The shared chokepoint loads the configured file or fails before launch.
    spend_lock_requirement = {} if execute else None
    try:
        preflight = require_pre_spend_preflight(
            lane=PRE_SPEND_HARDWARE_LANE,
            provider=provider,
            credential_present=pre_inventory.get("api_confirmed") is True,
            capacity_evidence={
                "available": capacity_available,
                "offer_count": capacity.get("offer_count"),
                "detail": gpu_name or "qualification_48gb_rtx_capacity_unavailable",
                "selected_offer": selected or None,
            },
            image_contract=image_contract_from_ref(image_ref),
            runtime_contract=runtime_contract_for_pre_spend(
                PRE_SPEND_PROGRESS_STALL_PHASES
            ),
            spend_gate_open=capacity_available and inventory_zero,
            record_dir=root,
            hardware_contract=hardware_contract,
            spend_admission_lock=spend_lock_requirement,
        )
    except PreSpendPreflightBlocked as exc:
        preflight = exc.preflight
        blockers = [
            f"qualification_pre_spend:{item}"
            for item in preflight.get("blockers") or []
        ]
        return preflight, blockers
    return preflight, []


def write_standard_artifacts(
    *,
    provider_launch_request: str | Path,
    preflight_bundle: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    bound: Mapping[str, Any],
    preflight: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> None:
    """Persist the canonical allocator outputs without provider secrets."""

    write_json(Path(provider_launch_request), dict(bound))
    write_json(Path(preflight_bundle), dict(preflight))
    write_json(Path(admission_out), dict(admission))
    write_json(Path(bound_request_out), dict(bound))
