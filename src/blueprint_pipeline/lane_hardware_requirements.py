"""Fail-closed hardware floors for paid GPU eval lanes.

Born from the 2026-07-06 T4 run: OSCAR-2B holds ~22.3 GB of fp32 weights, so a
24 GB RTX 4090 OOMs before the first denoising step — and the mitigation
someone reaches for under OOM pressure (halving generation resolution) buys no
memory (weights dominate) while silently degrading output quality. The lesson:
GPU sizing is a *contract*, decided before spend, never a knob turned during a
run.

``build_lane_hardware_contract`` is consumed by the pre-spend preflight
chokepoint (``paid_lane_guard.require_pre_spend_preflight``); a FAIL there
means the provider launch call must never happen. Unknown lanes fail closed —
registering a floor here is the price of admission for a new paid lane.
"""

from __future__ import annotations

from typing import Any

LANE_HARDWARE_REQUIREMENTS_SCHEMA_VERSION = "lane_hardware_requirements.v1"

# Measured VRAM residency, not vendor marketing numbers.
OSCAR_2B_FP32_WEIGHTS_GB = 22.3  # UMT5-XXL + DiT + VAE, observed 2026-07-06
GROOT_N17_SONIC_SERVER_GB = 6.5  # observed resident footprint, ZMQ server

# Known RunPod GPU type ids -> usable VRAM (GB). Extend as new types are used.
KNOWN_GPU_VRAM_GB: dict[str, float] = {
    "NVIDIA GeForce RTX 4090": 24.0,
    "NVIDIA GeForce RTX 4080": 16.0,
    "NVIDIA RTX A5000": 24.0,
    "NVIDIA RTX A6000": 48.0,
    "NVIDIA L40S": 48.0,
    "NVIDIA A100 80GB PCIe": 80.0,
    "NVIDIA A100-SXM4-80GB": 80.0,
    "NVIDIA H100 PCIe": 80.0,
    "NVIDIA H100 80GB HBM3": 80.0,
}
KNOWN_NO_RT_CORE_GPU_TYPES = frozenset(
    {
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA H100 PCIe",
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H200",
    }
)

# Lane floors. min_vram_gb includes explicit headroom for activations,
# framework overhead, and renderer contexts — a floor equal to the sum of
# weight residencies is how the 4090 OOM happened.
LANE_HARDWARE_REQUIREMENTS: dict[str, dict[str, Any]] = {
    # OSCAR-2B WAM + GR00T N1.7 SONIC server co-resident (T4 closed loop).
    "kitchen_g1_groot_sonic_eval": {
        "min_vram_gb": 40.0,
        "min_disk_gb": 175,
        "requires_rtx": True,
        "recommended_gpu_type_ids": (
            "NVIDIA RTX A6000",
            "NVIDIA L40S",
        ),
        "resident_models": {
            "oscar_2b_fp32_weights_gb": OSCAR_2B_FP32_WEIGHTS_GB,
            "groot_n17_sonic_server_gb": GROOT_N17_SONIC_SERVER_GB,
        },
        "notes": (
            "OSCAR-2B alone OOMs a 24GB card; resolution reduction does NOT "
            "reduce weight residency and must never be used as an OOM "
            "mitigation on this lane."
        ),
    },
    # Torch/LeRobot policy families through the MuJoCo harness (ACT-class).
    "real_policy_family_gpu_eval": {
        "min_vram_gb": 16.0,
        "min_disk_gb": 80,
        "recommended_gpu_type_ids": (
            "NVIDIA GeForce RTX 4090",
            "NVIDIA RTX A5000",
        ),
        "resident_models": {},
        "notes": "Single policy checkpoint + MuJoCo OSMesa rendering.",
    },
    # GR00T N1.7 + SONIC sim2sim / policy-endpoint lane without OSCAR.
    "groot_sonic_mujoco_eval": {
        "min_vram_gb": 16.0,
        "min_disk_gb": 100,
        "recommended_gpu_type_ids": (
            "NVIDIA GeForce RTX 4090",
            "NVIDIA RTX A5000",
        ),
        "resident_models": {
            "groot_n17_sonic_server_gb": GROOT_N17_SONIC_SERVER_GB,
        },
        "notes": "Policy server + MuJoCo; no diffusion WAM resident.",
    },
}


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_gpu_vram_gb(gpu_type_id: str | None) -> float | None:
    if not gpu_type_id:
        return None
    return KNOWN_GPU_VRAM_GB.get(str(gpu_type_id).strip())


def build_lane_hardware_contract(
    *,
    lane: str,
    gpu_type_id: str | None,
    vram_gb: float | None = None,
    disk_gb: float | None = None,
) -> dict[str, Any]:
    """Evaluate a proposed pod against the lane's registered floor.

    Fail-closed on every unknown: unregistered lane, unknown GPU type with no
    explicit ``vram_gb``, missing disk size. ``status`` is PASS only when every
    floor is provably met.
    """
    lane_name = str(lane or "").strip()
    blockers: list[str] = []
    requirements = LANE_HARDWARE_REQUIREMENTS.get(lane_name)
    if not lane_name:
        blockers.append("lane_hardware_contract_lane_missing")
    elif requirements is None:
        blockers.append(
            f"lane_hardware_requirements_unregistered:{lane_name}"
        )

    resolved_vram = _float(vram_gb)
    if resolved_vram is None:
        resolved_vram = resolve_gpu_vram_gb(gpu_type_id)
    if resolved_vram is None:
        blockers.append(
            "gpu_vram_unknown:"
            + (str(gpu_type_id).strip() if gpu_type_id else "gpu_type_missing")
        )

    resolved_disk = _float(disk_gb)

    if requirements is not None:
        min_vram = float(requirements["min_vram_gb"])
        min_disk = float(requirements["min_disk_gb"])
        if resolved_vram is not None and resolved_vram < min_vram:
            blockers.append(
                f"gpu_vram_below_lane_floor:{resolved_vram:g}gb_lt_{min_vram:g}gb"
            )
        if resolved_disk is None:
            blockers.append("container_disk_size_missing")
        elif resolved_disk < min_disk:
            blockers.append(
                f"container_disk_below_lane_floor:{resolved_disk:g}gb_lt_{min_disk:g}gb"
            )
        if (
            requirements.get("requires_rtx") is True
            and str(gpu_type_id or "").strip() in KNOWN_NO_RT_CORE_GPU_TYPES
        ):
            blockers.append(
                f"gpu_lacks_rt_cores_for_isaac_rtx:{str(gpu_type_id).strip()}"
            )

    return {
        "schema_version": LANE_HARDWARE_REQUIREMENTS_SCHEMA_VERSION,
        "lane": lane_name or None,
        "gpu_type_id": str(gpu_type_id).strip() if gpu_type_id else None,
        "vram_gb": resolved_vram,
        "disk_gb": resolved_disk,
        "requirements": dict(requirements) if requirements else None,
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "claim_boundary": (
            "A passing hardware contract proves the pod meets the lane's "
            "registered memory/disk floor. It does not prove the run will "
            "succeed or that outputs will be high quality."
        ),
    }


def hardware_contract_or_raise(
    *,
    lane: str,
    gpu_type_id: str | None,
    vram_gb: float | None = None,
    disk_gb: float | None = None,
) -> dict[str, Any]:
    """Launch-driver helper: refuse under-provisioned pods before any spend."""
    contract = build_lane_hardware_contract(
        lane=lane, gpu_type_id=gpu_type_id, vram_gb=vram_gb, disk_gb=disk_gb
    )
    if contract["status"] != "PASS":
        raise RuntimeError(
            "lane_hardware_contract_failed:" + ",".join(contract["blockers"])
        )
    return contract
