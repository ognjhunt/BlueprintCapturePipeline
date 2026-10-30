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


# Known calls after optional novel views: their independent review, then the
# task-surface plan and its mask review. Unbought final outputs are added per call.
AFTER_NOVEL_VIEWS = {'independent_image_review': 1, 'task_surface_plan': 1, 'task_surface_review': 1}
# The dollar share of those review/planning calls, held back as one margin.
REVIEW_MARGIN_USD = 1


def submitted(stage: Path) -> bool:
    """A batch that is bought or in flight; it only polls or adopts, never re-buys."""
    return (stage / 'atlas_pose/submission.json').exists() or (stage / 'result.json').exists()


def novel_view_headroom(configuration, *, batches, final_output_root, context, source_digest,
                        extra_cost_usd=0.):
    """The finish forecast before buying optional novel views; ``batches`` are (stage, cost).

    Returns None when every batch is already bought, so retained submissions
    always reach their adapter for polling/reconciliation with zero headroom.
    """
    pending = [cost for stage, cost in batches if not submitted(stage)]
    if not pending:
        return None
    return headroom(configuration, stages={'novel_view_batches': len(pending), **AFTER_NOVEL_VIEWS},
                    cost_usd=sum(pending) + extra_cost_usd, final_output_root=final_output_root,
                    context=context, source_digest=source_digest)


def headroom(configuration, *, stages, final_output_root, context, source_digest, cost_usd=0.):
    """Whether the allowance covers this optional step plus what finishing still needs.

    The one forecast every optional paid step uses, for dollars and requests
    alike. It is a lower bound, not a grant: later mask corrections and provider
    retries are not known yet, and canonical admission still checks every call.
    An allowance the configuration does not carry is not forecast.
    """
    if configuration.get('schema_version') in {
            'website_atlas_configuration.v1', 'owned_capture_atlas_configuration.v1'}:
        from .website_task_context import load_atlas_configuration
        from .website_atlas_pipeline import resumed_policy

        # Earlier stages have spent since the caller loaded its snapshot.
        # A noncreating read cannot issue more authority.
        current = load_atlas_configuration(context, source_digest, create=False)
        resumed_policy(configuration, current)
    else:
        current = configuration
    remaining = current.get('remaining_requests')
    if 'remaining_requests' in current and (type(remaining) is not int or remaining < 0):
        raise ValueError('atlas_completion_remaining_requests_invalid')
    dollars = current.get('remaining_cost_usd')
    if remaining is None and dollars is None:
        return None
    reserve = finish_reserve(configuration['policy'], stages, final_output_root)
    required, required_usd = reserve['requests'], round(cost_usd + reserve['cost_usd'], 6)
    return {'schema_version': 'atlas_finish_forecast.v1',
            'remaining_requests': remaining, 'minimum_remaining_requests': required,
            'remaining_cost_usd': dollars, 'minimum_remaining_cost_usd': required_usd,
            'unfinished_stages': reserve['stages'],
            'sufficient_for_known_calls': ((remaining is None or remaining >= required)
                                           and (dollars is None or dollars >= required_usd)),
            'additional_mask_or_retry_calls': 'not_yet_known',
            'reservation_created': False}


def finish_reserve(policy, stages, final_output_root=None):
    """Known calls and dollars still needed after ``stages``, including unbought final outputs."""
    admitted = admitted_final_outputs(final_output_root) if final_output_root else frozenset()
    stages = dict(stages)
    if 'refined_splat_cost_usd' in policy and 'posedImages2Splats' not in admitted:
        stages['refined_splat'] = 1
    if 'mesh_export_cost_usd' in policy and 'splats2Mesh' not in admitted:
        stages['mesh_export'] = 1
    return {'stages': stages, 'requests': sum(stages.values()),
            'cost_usd': round(final_output_cost(policy, admitted) + REVIEW_MARGIN_USD, 6)}


def require_headroom(forecast, path: Path):
    """Retain the forecast and stop before spending what the scene cannot finish with."""
    if forecast is None:
        return
    from .common import write_json

    write_json(path, forecast)
    if forecast['remaining_cost_usd'] is not None and (
            forecast['remaining_cost_usd'] < forecast['minimum_remaining_cost_usd']):
        raise ValueError('atlas_completion_remaining_budget_insufficient')
    if not forecast['sufficient_for_known_calls']:
        raise ValueError('atlas_completion_request_budget_insufficient:'
                         f"required={forecast['minimum_remaining_requests']},"
                         f"remaining={forecast['remaining_requests']}")
