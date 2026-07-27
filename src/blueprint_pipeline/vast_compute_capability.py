"""CUDA compute-capability admission for Vast GPU offers.

The pinned TensorRT build cannot generate kernels for every GPU the provider
sells.  TensorRT 10.4 (``10.4.0.26-1+cuda12.6``, pinned by the thin-release live
prerequisite gate) tops out at sm_90.  Blackwell parts report compute capability
1200, and the GEAR-SONIC controller builds its policy engine from
``policy/release/model_decoder.onnx`` at startup rather than shipping a prebuilt
``.trt``.  On an sm_120 host that build fails with::

    IBuilder::buildSerializedNetwork: Error Code 10: Could not find any implementation
    ✗ Failed to convert policy ONNX to TRT
    terminate called: Failed to initialize control policy

which surfaces as ``official_gear_sonic_controller_not_ready`` and exits the
episode with code 1.  Observed live on Vast instance 45771989 (RTX PRO 6000 WS).

Rate and VRAM filters cannot separate these parts: Blackwell offers span
$0.077-$1.00/hour and include high-VRAM cards, so an explicit architecture
ceiling is the only filter that works.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

#: Highest compute capability the pinned TensorRT build can target.
#: Applied as a *default* wherever offers are selected so every current and
#: future Vast path inherits it; lanes that never build a TensorRT engine may
#: pass ``max_compute_cap=0`` to opt out.
TENSORRT_MAX_SUPPORTED_COMPUTE_CAP = 900

#: Lifts or lowers the ceiling without a code change.
MAX_COMPUTE_CAP_ENV = "BLUEPRINT_VAST_MAX_COMPUTE_CAP"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def normalized_compute_cap(value: Any) -> int | None:
    """Normalize a compute capability to integer hundreds (8.9 -> 890)."""

    number = _number(value)
    if number is None:
        return None
    if 0 < number < 100:
        return int(round(float(number) * 100))
    return int(number)


def _offer_compute_cap(offer: Mapping[str, Any]) -> int | None:
    return normalized_compute_cap(
        offer.get("compute_cap_normalized")
        if offer.get("compute_cap_normalized") is not None
        else offer.get("compute_cap")
    )


def meets_min_compute_cap(offer: Mapping[str, Any], min_compute_cap: int) -> bool:
    """Whether an offer clears a minimum architecture floor.

    Fail-closed on an unreported capability: a floor exists to guarantee a
    capability, and an offer that cannot prove it does not qualify.
    """

    if not min_compute_cap:
        return True
    compute_cap = _offer_compute_cap(offer)
    return compute_cap is not None and compute_cap >= int(min_compute_cap)


def meets_max_compute_cap(offer: Mapping[str, Any], max_compute_cap: int) -> bool:
    """Whether an offer is at or below the TensorRT-buildable ceiling.

    Deliberately permissive about an unreported capability.  Live Vast offers
    always carry ``compute_cap``, so rejecting unknowns would not catch the
    incompatibility this ceiling exists for -- it would only convert an upstream
    schema change into a total selection outage.  Strict about architectures we
    can prove unusable; permissive about ones we cannot observe.
    """

    if not max_compute_cap:
        return True
    compute_cap = _offer_compute_cap(offer)
    if compute_cap is None:
        return True
    return compute_cap <= int(max_compute_cap)


def resolve_max_compute_cap(explicit: Any = None) -> int:
    """Resolve the effective ceiling for a run.

    Defaults to the TensorRT-buildable ceiling so a lane that builds the pinned
    engine inherits protection without opting in.  A workload that never builds
    it passes ``0`` (or sets the env var) to lift the ceiling.
    """

    env_value = os.getenv(MAX_COMPUTE_CAP_ENV)
    if explicit is not None:
        resolved = _number(explicit)
    elif env_value:
        resolved = _number(env_value)
    else:
        resolved = float(TENSORRT_MAX_SUPPORTED_COMPUTE_CAP)
    return max(0, int(resolved if resolved is not None else 0))


def capacity_selection_overrides(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return optional architecture and policy controls for offer selection."""

    raw_minimum = request.get("min_compute_cap")
    minimum = int(raw_minimum) if type(raw_minimum) is int and raw_minimum > 0 else 0
    raw_maximum = request.get("max_compute_cap")
    maximum = (
        int(raw_maximum)
        if type(raw_maximum) is int and raw_maximum >= 0
        else resolve_max_compute_cap()
    )
    return {
        "min_compute_cap": minimum,
        "max_compute_cap": maximum,
        "prefer_isaac_rt": request.get("prefer_isaac_rt") is True,
        "gpu_selection_policy": request.get("gpu_selection_policy"),
    }


def capacity_selection_policy(
    request: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the artifact-safe policy projection for a capacity preflight."""

    policy = {
        "max_hourly_rate_usd": selection["max_hourly_rate"],
        "min_gpu_ram_mb": selection["min_gpu_ram_mb"],
        "min_reliability": selection["min_reliability"],
        "require_avx": selection["require_avx"],
        "require_known_supported_isaac_driver": selection[
            "require_known_supported_isaac_driver"
        ],
        "require_direct_port": selection["require_direct_port"],
        "preferred_gpu_keywords": selection["preferred_gpu_keywords"],
    }
    optional_keys = {
        "min_compute_cap",
        "max_compute_cap",
        "prefer_isaac_rt",
        "gpu_selection_policy",
    }
    if optional_keys.intersection(request):
        policy.update({key: selection[key] for key in optional_keys})
    return policy


def any_offer_exceeds_ceiling(summaries: "list[Mapping[str, Any]]", max_compute_cap: int) -> bool:
    """Whether any offer was removed specifically for its architecture."""

    if not max_compute_cap:
        return False
    return any(not meets_max_compute_cap(item, max_compute_cap) for item in summaries)


def architecture_excluded_count(summaries: "list[Mapping[str, Any]]", max_compute_cap: int) -> int:
    """How many offers the architecture ceiling removed."""

    if not max_compute_cap:
        return 0
    return sum(1 for item in summaries if not meets_max_compute_cap(item, max_compute_cap))
