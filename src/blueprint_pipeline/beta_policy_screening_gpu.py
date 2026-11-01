"""Whether the screening GPU lane can rent any part that meets a candidate's memory floor.

Beta journey: planning one website job's screening, before any candidate is
funded. Every robolab and task-family candidate runs on the policy canary
session lane, which rents one Vast GPU under that lane's own filters
(``native_task_arena_vast.policy_canary_session_offer_hardware``): RT-core
parts only, the Isaac rendering denylist, and a compute-capability window.
A candidate whose declared memory floor no part in that window can meet would
search for offers until its slot closes, holding a reservation from the site's
budget the whole time. It is listed ``infeasible`` with this module's reason
instead, and the budget goes to the next candidate in plan order.

The verdict reads the allocator's own predicates and lane values, applied to
public part specifications. It is a refusal only when every part the lane's RT
keywords match is described here; a keyword with no described part leaves the
candidate feasible, so an unknown market is never ruled out. Offer selection
itself still reads live offers; nothing here changes what the allocator rents.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

GPU_MEMORY_FLOOR_UNRENTABLE = "screening_gpu_memory_floor_unrentable"

# Public NVIDIA specifications for every part the Isaac RT keywords match, by
# the names Vast lists: nominal memory in GB (an upper bound on what an offer
# reports) and CUDA compute capability in hundredths (8.9 -> 890).
RT_GPU_PARTS: dict[str, tuple[int, int]] = {
    "RTX PRO 6000 WS": (96, 1200),
    "RTX PRO 6000 S": (96, 1200),
    "RTX PRO 6000 Max-Q": (96, 1200),
    "RTX 5090": (32, 1200),
    "RTX 4090": (24, 890),
    "RTX 4090D": (24, 890),
    "RTX 3090 Ti": (24, 860),
    "RTX 3090": (24, 860),
    "RTX A6000": (48, 860),
    "RTX 6000 Ada": (48, 890),
    "RTX 6000Ada": (48, 890),
    "Q RTX 6000": (24, 750),
    "L40S": (48, 890),
    "L40": (48, 890),
}


def rentable_parts(minimum_gpu_memory_gb: Any) -> list[str] | None:
    """Described parts the screening lane would rent for this floor; ``None`` when its market is not fully described."""
    from . import vast_compute_capability as vcc
    from .gpu_selection_policy import (
        ISAAC_RT_GPU_KEYWORDS,
        _is_isaac_rt_candidate,
        gpu_allowed_by_policy,
        resolve_gpu_selection_policy,
    )
    from .native_task_arena_vast import policy_canary_session_offer_hardware
    from .policy_canary_session_bundle_identity import minimum_gpu_ram_mb
    from .vast_provider_adapter import _is_isaac_provider_bundle

    declared = {} if minimum_gpu_memory_gb is None else {"minimum_gpu_memory_gb": minimum_gpu_memory_gb}
    lane = policy_canary_session_offer_hardware(minimum_gpu_ram_mb(declared))
    if not _is_isaac_provider_bundle(lane["provider_bundle_kind"]):
        return None  # Without the RT requirement the lane's market is open, and unknown stays unknown.
    names = [name.upper() for name in RT_GPU_PARTS]
    if any(not any(keyword in name for name in names) for keyword in ISAAC_RT_GPU_KEYWORDS):
        return None
    # ``run_arena_native_control_vast`` always prefers RT parts, which resolves an unnamed policy to Isaac's.
    policy = resolve_gpu_selection_policy(lane["gpu_selection_policy"], prefer_isaac_rt=True)
    ceiling = vcc.resolve_max_compute_cap(lane["max_compute_cap"])
    return sorted(
        name for name, (memory_gb, capability) in RT_GPU_PARTS.items()
        if _is_isaac_rt_candidate(name)
        # A driver's CUDA version belongs to the host, not the part, so it never rules a part out here.
        and gpu_allowed_by_policy(name, policy, cuda_max_good=math.inf)
        and vcc.meets_min_compute_cap({"compute_cap": capability}, lane["min_compute_cap"])
        and vcc.meets_max_compute_cap({"compute_cap": capability}, ceiling)
        and memory_gb * 1024 >= lane["min_gpu_ram_mb"]
    )


def gpu_infeasible_blockers(runtime: Any) -> list[str]:
    """The planning refusal for a catalog runtime whose memory floor the screening lane cannot rent."""
    if not isinstance(runtime, Mapping):
        return []
    try:
        parts = rentable_parts(runtime.get("minimum_gpu_memory_gb"))
    except (ValueError, TypeError):
        return []  # An invalid floor keeps its own refusal where the candidate is bound.
    return [GPU_MEMORY_FLOOR_UNRENTABLE] if parts == [] else []


__all__ = ["GPU_MEMORY_FLOOR_UNRENTABLE", "RT_GPU_PARTS", "gpu_infeasible_blockers", "rentable_parts"]
