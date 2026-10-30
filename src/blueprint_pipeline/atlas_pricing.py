"""Reviewed per-request Atlas pricing for ADP-040 / day-28 appearance.

The rate card lives in the signed runtime policy. Historical policies without
an additional-frame rate retain their original quotes and receipt identities.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def render_cost(policy, frame_count: int, *, depth=True, legacy_multiplier=1) -> float:
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("atlas_render_frame_count_invalid")
    field = "rgbd_view_cost_usd" if depth else "rgb_view_cost_usd"
    first = policy[field]
    additional = policy.get("additional_" + field)
    rates = [first] if additional is None else [first, additional]
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
           for v in rates):
        raise ValueError("atlas_render_rate_invalid")
    return round(frame_count * first * legacy_multiplier if additional is None
                 else first + (frame_count - 1) * additional, 6)


def splat_cost(policy) -> float:
    value = policy.get("refined_splat_cost_usd")
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("atlas_refined_splat_price_unreviewed")
    return float(value)


def final_output_cost(policy, admitted=frozenset()) -> float:
    """The final-output prices to hold back, except tasks whose purchase is already admitted."""
    return (float(policy.get("mesh_export_cost_usd", 0)) * ("splats2Mesh" not in admitted)
            + (splat_cost(policy) if "refined_splat_cost_usd" in policy
               and "posedImages2Splats" not in admitted else 0))


def admitted_final_outputs(root: Path) -> frozenset:
    """Final-output tasks with an admitted purchase that is live or bought.

    Its reservation is already deducted from the remaining allowance, so holding
    its price back again counts it twice. capture-4b60431e 2026-10-30: a live
    refined-splat retry plus that double count left $0.91 short and failed the job.
    """
    found = set()
    for path in root.rglob("atlas_pose/submission.json"):
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        task = (state.get("binding") or {}).get("task")
        # A failed operation (free per the provider) or a refused send bought nothing.
        if task in ("posedImages2Splats", "splats2Mesh") and state.get("status") not in ("failed", "rejected"):
            found.add(task)
    return frozenset(found)
