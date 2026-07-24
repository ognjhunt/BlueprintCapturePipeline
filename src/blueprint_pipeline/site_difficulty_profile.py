"""Per-site difficulty profile, so cross-site policy numbers are comparable.

A policy that succeeds 80% of the time at one site and 55% at another has not
necessarily regressed. The second site may have tighter clearances, more
reflective floors, more dynamic traffic, or objects that are simply harder to
grasp. Without a published description of that difference, every cross-site
comparison silently attributes site variance to the policy -- and a buyer
reading two site reports has no way to tell which effect they are looking at.

This publishes the difference. Every axis is derived from measurements the
pipeline already computes and already gates on, so the profile adds no new
sensing and asserts nothing that was not already established:

``spatial_constraint``   clearance corridors and standoff margins
``geometric_complexity`` scene extent, obstacle density, traversable fraction
``visual_conditions``    lighting variance and reflective/low-texture surface share
``object_difficulty``    graspable-object size and affordance reachability
``dynamic_environment``  shared traffic and non-routine operations findings
``task_horizon``         task step count and route length

A difficulty score is a *covariate for interpretation*, never a correction. It
does not normalise success rates, does not adjust a ranking, and does not excuse
a poor result. Dividing a success rate by a difficulty number would invent
precision nobody measured; reporting them side by side does not.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import read_json_any, utc_now_iso, write_json


PROFILE_SCHEMA_VERSION = "site_difficulty_profile.v1"
COMPARISON_SCHEMA_VERSION = "cross_site_difficulty_comparison.v1"

AXES = (
    "spatial_constraint",
    "geometric_complexity",
    "visual_conditions",
    "object_difficulty",
    "dynamic_environment",
    "task_horizon",
)

# Bands are deliberately coarse. The underlying measurements do not support
# distinguishing a 0.61 from a 0.64, and a decimal score would imply they do.
BANDS = ("low", "moderate", "high", "very_high")
_BAND_EDGES = (0.25, 0.5, 0.75)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _normalize(value: Any, *, easy: float, hard: float) -> float | None:
    """Map a raw measurement onto [0, 1] where 1 is harder.

    ``easy``/``hard`` are the anchor values, and the mapping is clamped rather
    than extrapolated so an outlier site cannot produce a score above 1.
    """

    number = _number(value)
    if number is None or easy == hard:
        return None
    fraction = (number - easy) / (hard - easy)
    return round(min(1.0, max(0.0, fraction)), 4)


def band_for(score: Any) -> str | None:
    value = _number(score)
    if value is None:
        return None
    for edge, name in zip(_BAND_EDGES, BANDS):
        if value < edge:
            return name
    return BANDS[-1]


def _axis(name: str, score: float | None, inputs: Mapping[str, Any]) -> dict[str, Any]:
    present = {key: value for key, value in inputs.items() if value is not None}
    return {
        "axis": name,
        "score": score,
        "band": band_for(score),
        "inputs_used": sorted(present),
        "inputs_missing": sorted(set(inputs) - set(present)),
        # An axis computed from no inputs is unmeasured, not easy.
        "measured": score is not None,
    }


def build_site_difficulty_profile(
    *,
    scene_id: str,
    capture_id: str,
    scene_placement: Mapping[str, Any] | None = None,
    geometry_evidence: Mapping[str, Any] | None = None,
    visual_conditions: Mapping[str, Any] | None = None,
    object_inventory: Mapping[str, Any] | None = None,
    shared_traffic_review: Mapping[str, Any] | None = None,
    non_routine_review: Mapping[str, Any] | None = None,
    task_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive a per-site difficulty profile from existing measurements."""

    blockers: list[str] = []
    if not _string(scene_id) or not _string(capture_id):
        blockers.append("site_difficulty_identity_missing")

    placement = _mapping(scene_placement)
    geometry = _mapping(geometry_evidence)
    visual = _mapping(visual_conditions)
    objects = _mapping(object_inventory)
    traffic = _mapping(shared_traffic_review)
    non_routine = _mapping(non_routine_review)
    task = _mapping(task_scope)

    # Tighter clearance and smaller standoff margin are harder.
    spatial_inputs = {
        "min_clearance_m": _normalize(
            placement.get("min_obstacle_clearance_m"), easy=1.20, hard=0.15
        ),
        "standoff_margin_m": _normalize(
            placement.get("standoff_margin_m"), easy=0.80, hard=0.05
        ),
    }
    spatial = _mean(spatial_inputs.values())

    complexity_inputs = {
        "obstacle_density_per_m2": _normalize(
            geometry.get("obstacle_density_per_m2"), easy=0.02, hard=0.60
        ),
        "traversable_fraction": _normalize(
            geometry.get("traversable_fraction"), easy=0.85, hard=0.25
        ),
        "scene_extent_m2": _normalize(geometry.get("scene_extent_m2"), easy=20.0, hard=2000.0),
    }
    complexity = _mean(complexity_inputs.values())

    visual_inputs = {
        "lighting_variance": _normalize(visual.get("lighting_variance"), easy=0.05, hard=0.60),
        "reflective_surface_fraction": _normalize(
            visual.get("reflective_surface_fraction"), easy=0.02, hard=0.45
        ),
        "low_texture_fraction": _normalize(
            visual.get("low_texture_fraction"), easy=0.05, hard=0.50
        ),
    }
    visual_score = _mean(visual_inputs.values())

    object_inputs = {
        "min_graspable_dimension_m": _normalize(
            objects.get("min_graspable_dimension_m"), easy=0.25, hard=0.02
        ),
        "affordance_reach_margin_m": _normalize(
            objects.get("affordance_reach_margin_m"), easy=0.30, hard=0.01
        ),
    }
    object_score = _mean(object_inputs.values())

    dynamic_inputs = {
        "shared_traffic_findings": _normalize(
            _finding_count(traffic), easy=0.0, hard=12.0
        ),
        "non_routine_findings": _normalize(
            _finding_count(non_routine), easy=0.0, hard=8.0
        ),
    }
    dynamic = _mean(dynamic_inputs.values())

    horizon_inputs = {
        "task_step_count": _normalize(task.get("step_count"), easy=2.0, hard=40.0),
        "route_length_m": _normalize(task.get("route_length_m"), easy=2.0, hard=120.0),
    }
    horizon = _mean(horizon_inputs.values())

    axes = [
        _axis("spatial_constraint", spatial, spatial_inputs),
        _axis("geometric_complexity", complexity, complexity_inputs),
        _axis("visual_conditions", visual_score, visual_inputs),
        _axis("object_difficulty", object_score, object_inputs),
        _axis("dynamic_environment", dynamic, dynamic_inputs),
        _axis("task_horizon", horizon, horizon_inputs),
    ]
    measured = [row for row in axes if row["measured"]]
    if not measured:
        blockers.append("site_difficulty_no_axis_measured")

    overall = _mean([row["score"] for row in measured]) if measured else None
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "profiled" if not blockers else "blocked",
        "scene_id": _string(scene_id) or None,
        "capture_id": _string(capture_id) or None,
        "axes": axes,
        "measured_axis_count": len(measured),
        "total_axis_count": len(axes),
        "overall_difficulty": overall,
        "overall_band": band_for(overall),
        # States how complete the profile is, so a site scored on two of six
        # axes is not read as equivalent to one scored on all six.
        "coverage_fraction": round(len(measured) / len(axes), 4) if axes else None,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "difficulty_is_a_covariate_not_a_correction": True,
            "success_rates_must_not_be_normalized_by_difficulty": True,
            "difficulty_does_not_excuse_or_explain_a_specific_failure": True,
            "unmeasured_axes_are_not_easy_axes": True,
            "profile_is_derived_from_existing_measurements_only": True,
        },
    }


def _finding_count(review: Mapping[str, Any]) -> float | None:
    findings = review.get("findings")
    if isinstance(findings, Sequence) and not isinstance(findings, (str, bytes)):
        return float(len(findings))
    count = _number(review.get("finding_count"))
    return count


def _mean(values) -> float | None:
    present = [value for value in values if value is not None]
    return round(sum(present) / len(present), 4) if present else None


def compare_sites(profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Put site profiles side by side so a cross-site number can be read.

    Deliberately reports the spread rather than a correction factor: the honest
    output is "these sites differ by this much on these axes", not an adjusted
    success rate.
    """

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for profile in profiles:
        if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
            blockers.append("site_difficulty_profile_schema_invalid")
            continue
        rows.append(
            {
                "scene_id": profile.get("scene_id"),
                "capture_id": profile.get("capture_id"),
                "overall_difficulty": profile.get("overall_difficulty"),
                "overall_band": profile.get("overall_band"),
                "coverage_fraction": profile.get("coverage_fraction"),
                "axis_scores": {
                    row.get("axis"): row.get("score")
                    for row in profile.get("axes") or []
                    if isinstance(row, Mapping)
                },
            }
        )
    if len(rows) < 2:
        blockers.append("cross_site_comparison_requires_two_profiles")

    per_axis: dict[str, Any] = {}
    for axis in AXES:
        values = [
            row["axis_scores"].get(axis)
            for row in rows
            if row["axis_scores"].get(axis) is not None
        ]
        per_axis[axis] = {
            "measured_site_count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "spread": round(max(values) - min(values), 4) if len(values) > 1 else None,
        }

    overalls = [row["overall_difficulty"] for row in rows if row["overall_difficulty"] is not None]
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "compared" if not blockers else "blocked",
        "sites": rows,
        "per_axis": per_axis,
        "overall_spread": round(max(overalls) - min(overalls), 4) if len(overalls) > 1 else None,
        "blockers": sorted(set(blockers)),
        "interpretation_note": (
            "Report difficulty beside a success rate, never divided into it. A "
            "difference in difficulty explains why two sites are not directly "
            "comparable; it does not quantify how much of a gap it accounts for."
        ),
        "claim_boundary": {
            "comparison_does_not_normalize_policy_results": True,
            "spread_is_descriptive_not_a_correction_factor": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a per-site difficulty profile")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    payload = _mapping(read_json_any(Path(args.input)))
    profile = build_site_difficulty_profile(
        scene_id=_string(payload.get("scene_id")),
        capture_id=_string(payload.get("capture_id")),
        scene_placement=_mapping(payload.get("scene_placement")),
        geometry_evidence=_mapping(payload.get("geometry_evidence")),
        visual_conditions=_mapping(payload.get("visual_conditions")),
        object_inventory=_mapping(payload.get("object_inventory")),
        shared_traffic_review=_mapping(payload.get("shared_traffic_review")),
        non_routine_review=_mapping(payload.get("non_routine_review")),
        task_scope=_mapping(payload.get("task_scope")),
    )
    write_json(Path(args.output), profile)
    print(json.dumps({"path": args.output, "status": profile["status"]}, sort_keys=True))
    return 0 if profile["status"] == "profiled" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
