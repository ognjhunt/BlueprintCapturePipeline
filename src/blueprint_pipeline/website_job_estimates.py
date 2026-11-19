"""Blueprint's best estimates of what a deployment company needs and the site never stated.

Owner decision 2026-11-19: where a site never states a fact a deployment company needs, the job packet gives
Blueprint's best honest estimate, not a useless range or a blank. It is worked out from first principles, labelled
an estimate with its basis and its uncertainty, and never presented as measured. No new paid call: every input is
something the pipeline already reads or buys.

The object's weight (``weight_estimate``: the preparation writes it, the robot fit and the job sheet read it):

1. ``published_exact_model``: the net weight the maker publishes for the exact model, where the pipeline already
   researched the object (``website_object_spec_research``), ±10 % (``website_articulated_mass``);
2. ``designed_object``: a job that adds an object the video does not show weighs what its design says;
3. ``physics_from_size_and_material``: the object's box from the video, across the scale band (its volume goes with
   the cube of the scale) × the share of that box its main material fills for a typical object of its kind × that
   material's density. Blueprint's video analysis reads the material and the share from the object's label and the
   frames (``clean_plate_removal_analysis_gemini``, revision r4: the same call, a few more words of answer); the
   densities are handbook ranges. The best figure is the box at the middle of the band × the geometric middle of the
   share and of the density; the range takes every factor at its lowest, then at its highest.
4. Otherwise it is not estimated, and the sheet says why in plain words.

How often the job runs (``operations_estimate``): the video analysis's own estimate from the job's words and the
place filmed (how many an hour while it runs, hours a day, days a week, each a range, and its basis).

The site's own quick answers on its job page (``website_site_job_facts``) come first: where the site answered how
heavy the object is, how big, how many an hour or how many hours a day, the job sheet prints its answer and not
Blueprint's estimate of the same thing (``job_packet_export.UNSTATED_FACTS`` is the seam an estimate fills).
"""
from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from typing import Any

PUBLISHED = "published_exact_model"
DESIGNED = "designed_object"
PHYSICS = "physics_from_size_and_material"
CLAIM = "estimate_not_weighed"
#: Typical densities of solid materials, kg/m³, as engineering handbooks give them: metal from aluminium (2,700)
#: to steel (7,850); wood from softwood to hardwood and fibreboard; plastic from polypropylene to PVC; soda-lime
#: glass; ceramic from earthenware to porcelain; stone from concrete to granite; paper and card; textile fibre from
#: nylon to cotton; foam; rubber; food from bread to dense produce; liquids from oils to syrups.
MATERIAL_DENSITY_KG_M3 = {
    "metal": (2700.0, 7900.0), "wood": (400.0, 900.0), "plastic": (900.0, 1400.0), "glass": (2400.0, 2600.0),
    "ceramic": (2000.0, 2500.0), "stone": (2200.0, 2800.0), "paper": (700.0, 1200.0), "fabric": (1100.0, 1550.0),
    "foam": (20.0, 200.0), "rubber": (900.0, 1500.0), "food": (200.0, 1200.0), "liquid": (850.0, 1300.0)}
#: Why a weight was not estimated, in the sheet's words.
NOT_ESTIMATED_WORDS = {
    "no_material_reading": ("no weight is published for its exact model, and Blueprint's video analysis did not "
                            "read what it is made of (a job analysed before Blueprint asked has no reading)"),
    "material_not_seen": ("no weight is published for its exact model, and the video does not show what it is "
                          "made of"),
    "no_object_box": "no weight is published for its exact model, and its size was not estimated",
}
OPERATIONS = ("per_hour", "hours_per_day", "days_per_week")
OPERATION_BOUNDS = {"per_hour": math.inf, "hours_per_day": 24.0, "days_per_week": 7.0}
#: The site's quick question (``website_task_context_contracts.SITE_JOB_FACT_KEYS``) each estimate answers, and each
#: line's words. A question the site answered keeps its answer; nothing here replaces it.
SITE_QUESTION = {"per_hour": "items_per_hour", "hours_per_day": "hours_per_day"}
OPERATION_LABELS = {"per_hour": "How many an hour", "hours_per_day": "How many hours a day the job runs",
                    "days_per_week": "How many days a week it runs"}
ESTIMATE_LABEL = "Blueprint's estimate; the site did not say"


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _pair(value: Any, *, low: float, high: float) -> list[float] | None:
    """Two finite numbers, the first above ``low``, at most the second, the second at most ``high``; else None."""
    if (isinstance(value, (list, tuple)) and len(value) == 2 and all(_finite(v) for v in value)
            and low < value[0] <= value[1] <= high):
        return [float(value[0]), float(value[1])]
    return None


def _r(value: float) -> float:
    return float(f"{value:.3g}")


def _published(target: Mapping[str, Any], object_spec: Mapping[str, Any] | None) -> dict[str, Any] | None:
    from .website_articulated_mass import _checked_spec, _published as published_weight
    try:
        spec = _checked_spec(target, object_spec)
        found = published_weight(spec, "net_weight") if spec is not None else None
    except ValueError:  # an unreadable record publishes nothing; the estimate below stands
        return None
    if found is None or found[1]["match"] != "exact_model":
        return None
    (low, high), source = found
    return {"status": "estimated", "basis": PUBLISHED, "best_kg": _r((low + high) / 2),
            "range_kg": [_r(low), _r(high)], "spec": source["spec"], "source_urls": list(source["source_urls"]),
            "claim": CLAIM}


def weight_estimate(*, dimensions_m: Sequence[float], relative_error: float, entry: Mapping[str, Any],
                    target: Mapping[str, Any] | None = None, object_spec: Mapping[str, Any] | None = None,
                    designed_kg: float | None = None) -> dict[str, Any]:
    """The object's weight: published, designed, or from its size and material; else why not (module docstring).

    ``dimensions_m`` is the object's box in simulator metres, ``relative_error`` the scale's (the band is
    ``[1 - f, 1 / (1 - f)]``, as the robot fit takes it), ``entry`` its removal-manifest entry.
    """
    published = _published(target, object_spec) if target is not None else None
    if published is not None:
        return published
    if designed_kg is not None and _finite(designed_kg) and designed_kg > 0:
        return {"status": "estimated", "basis": DESIGNED, "best_kg": _r(designed_kg),
                "range_kg": [_r(designed_kg), _r(designed_kg)], "claim": CLAIM}
    material, share = entry.get("material"), _pair(entry.get("solid_fraction"), low=0.0, high=1.0)
    if material == "unknown":
        return {"status": "not_estimated", "reason": "material_not_seen"}
    if material not in MATERIAL_DENSITY_KG_M3 or share is None:
        return {"status": "not_estimated", "reason": "no_material_reading"}
    if (len(dimensions_m) != 3 or not all(_finite(v) and v > 0 for v in dimensions_m)
            or not _finite(relative_error) or not 0 <= relative_error < 1):
        return {"status": "not_estimated", "reason": "no_object_box"}
    volume = math.prod(float(v) for v in dimensions_m)
    band = (1.0 - relative_error, 1.0 / (1.0 - relative_error))
    density = MATERIAL_DENSITY_KG_M3[material]
    best = volume * math.sqrt(share[0] * share[1]) * math.sqrt(density[0] * density[1])
    low = volume * band[0] ** 3 * share[0] * density[0]
    high = volume * band[1] ** 3 * share[1] * density[1]
    return {"status": "estimated", "basis": PHYSICS, "best_kg": _r(best), "range_kg": [_r(low), _r(high)],
            "material": material, "solid_fraction": share, "density_kg_m3": list(density),
            "box_m": [_r(float(v)) for v in dimensions_m], "scale_band": [_r(v) for v in band],
            "source": "video_analysis_material_and_fill", "claim": CLAIM}


def operations_estimate(value: Any) -> dict[str, Any] | None:
    """The video analysis's estimate of how often the job runs, checked again; None when there is none."""
    if not isinstance(value, Mapping) or not isinstance(value.get("basis"), str) or not value["basis"].strip():
        return None
    ranges = {key: _pair(value.get(key), low=-1e-9, high=OPERATION_BOUNDS[key]) for key in OPERATIONS}
    if any(row is None for row in ranges.values()):
        return None
    return {**ranges, "basis": value["basis"].strip()[:300], "source": "video_analysis_estimate"}


def preparation_estimates(*, subject: Mapping[str, Any], entry: Mapping[str, Any],
                          target: Mapping[str, Any] | None, relative_error: float,
                          removal_manifest: Mapping[str, Any], designed: Mapping[str, Any] | None) -> dict[str, Any]:
    """What the preparation records: ``weight_estimate`` always, ``operations_estimate`` when the analysis gave one."""
    low, high = subject.get("aabb_min_xyz") or [], subject.get("aabb_max_xyz") or []
    dimensions = ([float(b) - float(a) for a, b in zip(low, high)]
                  if len(low) == len(high) == 3 and all(_finite(v) for v in (*low, *high)) else [])
    weight = weight_estimate(dimensions_m=dimensions, relative_error=relative_error, entry=entry, target=target,
                             object_spec=(target or {}).get("object_spec"),
                             designed_kg=(designed or {}).get("mass_kg"))
    operations = operations_estimate(removal_manifest.get("operations_estimate"))
    return {"weight_estimate": weight, **({"operations_estimate": operations} if operations else {})}


# --- the job sheet's words ------------------------------------------------------------------------------------

def _figure(value: float) -> str:
    """Two significant figures, never in exponent form."""
    if value >= 10:
        return f"{value:,.0f}"
    return f"{float(f'{value:.2g}'):g}"


def _span(low: float, high: float, unit: str = "") -> str:
    words = _figure(low) if _figure(low) == _figure(high) else f"{_figure(low)} to {_figure(high)}"
    return words + (f" {unit}" if unit else "")


def weight_line(preparation: Mapping[str, Any], *, site_stated: bool) -> str | None:
    """The sheet's weight line; None for a preparation that weighs nothing (a job no object is built for)."""
    estimate = preparation.get("weight_estimate")
    prior = ((preparation.get("physics") or {}).get("bounds") or {}).get("mass_kg")
    prior = _pair(prior, low=-1e-9, high=math.inf)
    if not isinstance(estimate, Mapping) and prior is None:
        return None
    simulation = f" The simulation uses a {{}} prior from its size alone: {_span(*prior, 'kg')}." if prior else ""
    if site_stated:
        return "Weight: the site's own answer, above (not checked); Blueprint made no estimate of its own."
    if not isinstance(estimate, Mapping):
        return "Weight: not estimated: this job was prepared before Blueprint estimated weights." + simulation.format("wide")
    if estimate.get("status") != "estimated":
        why = NOT_ESTIMATED_WORDS.get(str(estimate.get("reason")), "nothing it was made from gives one")
        return f"Weight: not estimated: {why}." + simulation.format("wide")
    best, (low, high) = estimate["best_kg"], estimate["range_kg"]
    if estimate["basis"] == PUBLISHED:
        return (f"Weight: about {_figure(best)} kg ({_span(low, high, 'kg')}), the weight its maker publishes for the "
                "exact model, give or take 10 %. An estimate, not weighed.")
    if estimate["basis"] == DESIGNED:
        return (f"Weight: {_figure(best)} kg, as Blueprint designed the object the job adds (the video does not show "
                "it). An estimate, not weighed.")
    share = estimate["solid_fraction"]
    return (f"Weight: about {_figure(best)} kg ({_span(low, high, 'kg')}). Blueprint's estimate from its size in the "
            f"video, across the scale band, and its likely material: mostly {estimate['material']}, filling "
            f"{_span(share[0] * 100, share[1] * 100)} % of its box for a typical object of its kind, as Blueprint's "
            "video analysis read it. An estimate, not weighed." + simulation.format("wider"))


def operations_lines(preparation: Mapping[str, Any] | None,
                     answered: Collection[str]) -> tuple[list[str], set[str]]:
    """Blueprint's estimate of how often the job runs, for each figure the site did not answer (``answered``: the
    site's quick-question keys it answered), with its basis; and the site's questions those lines answer. Nothing
    when the analysis gave no estimate: the sheet then says the site did not state it."""
    estimate = operations_estimate((preparation or {}).get("operations_estimate"))
    keys = [key for key in OPERATIONS if SITE_QUESTION.get(key) not in answered]
    if estimate is None or not keys:
        return [], set()
    lines = [f"{OPERATION_LABELS[key]}: {_span(*estimate[key])} ({ESTIMATE_LABEL})." for key in keys]
    lines.append(f"What that estimate rests on: {estimate['basis'].rstrip('.')}. Blueprint's video analysis made it "
                 "from the job's words and the place filmed; nobody at the site checked it.")
    return lines, {SITE_QUESTION[key] for key in keys if key in SITE_QUESTION}


__all__ = ["CLAIM", "DESIGNED", "MATERIAL_DENSITY_KG_M3", "PHYSICS", "PUBLISHED",
           "operations_estimate", "operations_lines", "preparation_estimates", "weight_estimate", "weight_line"]
