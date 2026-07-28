"""Per-workload GPU selection policies for provider offer filtering.

A single global exclusion list is the wrong shape for hardware choice. The
Isaac RT-core denylist was applied to every provider offer selection, so a pure
generation or training campaign was barred from exactly the accelerators it
needs -- and a hardcoded list also ages badly, silently excluding parts nobody
had thought about when it was written.

Hardware eligibility is a property of the *workload*: Isaac's RTX rendering path
needs RT cores, while world-model generation and training are compute and VRAM
bound and have no such requirement. Policies are therefore named, explicit and
overridable, and an unknown policy name fails closed to the most restrictive one
rather than silently opening selection.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


GPU_SELECTION_POLICY_SCHEMA_VERSION = "gpu_selection_policy.v2"

ISAAC_RENDERING_GPU_POLICY = "isaac_rendering"
GENERATION_GPU_POLICY = "generation"
TRAINING_GPU_POLICY = "training"
OPEN_GPU_POLICY = "open"

# Datacenter parts without RT cores: unusable for Isaac's RTX rendering path,
# but among the best available hardware for generation and training.
DISALLOWED_ISAAC_GPU_KEYWORDS = ("A100", "H100", "H200", "B200", "GB200")

GPU_SELECTION_POLICIES: dict[str, dict[str, Any]] = {
    ISAAC_RENDERING_GPU_POLICY: {
        "policy_id": ISAAC_RENDERING_GPU_POLICY,
        "denied_gpu_keywords": DISALLOWED_ISAAC_GPU_KEYWORDS,
        "allowed_gpu_keywords": (),
        "reason": "Isaac RTX rendering requires RT cores",
        # Envelope tuned for short Isaac smoke attempts.
        "recommended_max_hourly_rate": 0.60,
        "recommended_hard_cap_usd": 0.75,
        "recommended_min_gpu_ram_mb": 0,
    },
    GENERATION_GPU_POLICY: {
        "policy_id": GENERATION_GPU_POLICY,
        # No denylist: large-VRAM parts (H100, H200, B200, RTX PRO 6000
        # Blackwell 96GB) are the point, not a hazard. Constrain with
        # min_gpu_ram_mb instead.
        "denied_gpu_keywords": (),
        "allowed_gpu_keywords": (),
        "reason": "generation is compute/VRAM bound and has no RT-core requirement",
        # A generation campaign inheriting the Isaac smoke envelope would match
        # no offers at all, so the envelope travels with the policy.
        "recommended_max_hourly_rate": 3.50,
        "recommended_hard_cap_usd": 25.00,
        "recommended_min_gpu_ram_mb": 48 * 1024,
    },
    TRAINING_GPU_POLICY: {
        "policy_id": TRAINING_GPU_POLICY,
        "denied_gpu_keywords": (),
        "allowed_gpu_keywords": (),
        "reason": "training is compute/VRAM bound and has no RT-core requirement",
        "recommended_max_hourly_rate": 6.00,
        "recommended_hard_cap_usd": 250.00,
        "recommended_min_gpu_ram_mb": 80 * 1024,
    },
    OPEN_GPU_POLICY: {
        "policy_id": OPEN_GPU_POLICY,
        "denied_gpu_keywords": (),
        "allowed_gpu_keywords": (),
        "reason": "no workload-specific hardware constraint",
        "recommended_max_hourly_rate": None,
        "recommended_hard_cap_usd": None,
        "recommended_min_gpu_ram_mb": 0,
    },
}


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def resolve_gpu_selection_policy(
    policy: str | Mapping[str, Any] | None,
    *,
    prefer_isaac_rt: bool = True,
) -> dict[str, Any]:
    """Resolve a named or inline GPU selection policy.

    When no policy is supplied the workload is inferred from ``prefer_isaac_rt``
    so existing Isaac callers keep the RT-core denylist while everything else
    gets an open field. An unknown policy name fails closed to the Isaac policy
    rather than silently opening selection.
    """

    if isinstance(policy, Mapping):
        minimum_cuda_max_good = policy.get("minimum_cuda_max_good")
        try:
            minimum_cuda_max_good = (
                float(minimum_cuda_max_good)
                if minimum_cuda_max_good is not None
                else 0.0
            )
        except (TypeError, ValueError):
            minimum_cuda_max_good = "invalid"
        return {
            "policy_id": _text(policy.get("policy_id")) or "inline",
            "denied_gpu_keywords": tuple(
                _text(item).upper()
                for item in policy.get("denied_gpu_keywords", ()) or ()
                if _text(item)
            ),
            "allowed_gpu_keywords": tuple(
                _text(item).upper()
                for item in policy.get("allowed_gpu_keywords", ()) or ()
                if _text(item)
            ),
            "reason": _text(policy.get("reason")) or "caller supplied",
            "recommended_max_hourly_rate": policy.get("recommended_max_hourly_rate"),
            "recommended_hard_cap_usd": policy.get("recommended_hard_cap_usd"),
            "recommended_min_gpu_ram_mb": policy.get("recommended_min_gpu_ram_mb") or 0,
            "minimum_cuda_max_good": minimum_cuda_max_good,
        }
    name = _text(policy)
    if name:
        selected = GPU_SELECTION_POLICIES.get(name)
        if selected is None:
            selected = GPU_SELECTION_POLICIES[ISAAC_RENDERING_GPU_POLICY]
        return dict(selected)
    default_name = ISAAC_RENDERING_GPU_POLICY if prefer_isaac_rt else OPEN_GPU_POLICY
    return dict(GPU_SELECTION_POLICIES[default_name])


def gpu_allowed_by_policy(
    gpu_name: str,
    policy: Mapping[str, Any],
    *,
    cuda_max_good: Any = None,
) -> bool:
    """Whether one offer's GPU is eligible under a resolved policy."""

    upper = _text(gpu_name).upper()
    denied = tuple(policy.get("denied_gpu_keywords") or ())
    allowed = tuple(policy.get("allowed_gpu_keywords") or ())
    if any(item in upper for item in denied):
        return False
    if allowed and not any(item in upper for item in allowed):
        return False
    try:
        minimum_cuda_max_good = float(policy.get("minimum_cuda_max_good") or 0.0)
    except (TypeError, ValueError):
        return False
    if minimum_cuda_max_good > 0:
        try:
            observed_cuda_max_good = float(cuda_max_good)
        except (TypeError, ValueError):
            return False
        if observed_cuda_max_good < minimum_cuda_max_good:
            return False
    return True


def policy_manifest(policy: Mapping[str, Any]) -> dict[str, Any]:
    """The policy as recorded in an offer-selection manifest."""

    return {
        "schema_version": GPU_SELECTION_POLICY_SCHEMA_VERSION,
        "policy_id": policy.get("policy_id"),
        "denied_gpu_keywords": list(policy.get("denied_gpu_keywords") or ()),
        "allowed_gpu_keywords": list(policy.get("allowed_gpu_keywords") or ()),
        "reason": policy.get("reason"),
        "minimum_cuda_max_good": policy.get("minimum_cuda_max_good") or 0.0,
    }


ISAAC_RT_GPU_KEYWORDS = (
    # Blackwell workstation parts keep RT cores, so they remain Isaac-eligible.
    "RTX PRO 6000",
    "RTX 5090",
    "RTX 4090",
    "RTX 3090 TI",
    "RTX 3090",
    "RTX A6000",
    "RTX 6000 ADA",
    "RTX 6000",
    "L40S",
    "L40",
)


def _is_disallowed_for_isaac(gpu_name: str) -> bool:
    """Isaac-rendering ineligibility, kept as a named predicate for manifests."""

    return any(item in _text(gpu_name).upper() for item in DISALLOWED_ISAAC_GPU_KEYWORDS)


def _is_isaac_rt_candidate(gpu_name: str) -> bool:
    """Whether an offer's GPU carries RT cores Isaac rendering can use."""

    upper = _text(gpu_name).upper()
    return not _is_disallowed_for_isaac(gpu_name) and any(
        item in upper for item in ISAAC_RT_GPU_KEYWORDS
    )
