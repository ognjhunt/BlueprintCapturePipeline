"""ADP-009: per-link mass bounds for a rebuilt articulated assembly.

The builder admits one reviewed mass per link inside a preregistered interval:
the body (``mass_kg_bounds``), the task part (``task_part_mass_kg_bounds``) and
every fixed interior part such as a rack or basket
(``fixed_part_mass_kg_bounds``). A manufacturer-published weight for the
identified model narrows an interval; otherwise each is a wide role prior
scaled by the estimated geometry. Nothing here is a measurement.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest

SPEC_SCHEMA_VERSION = "website_object_spec.v1"
# Relative half-width around a published weight, by how closely it matches.
PUBLISHED_RELATIVE_SPREAD = {"exact_model": 0.1, "model_family": 0.25, "brand_category": 0.5}
_TO_KG = {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "lbs": 0.45359237}
# Estimated priors: body mass per envelope volume (kg/m^3), a hinged door per
# front area (kg/m^2), a fixed rack or basket per tub footprint (kg/m^2).
BODY_DENSITY_KG_M3 = (60.0, 250.0)
BODY_FLOOR_KG = (4.0, 12.0)
DRAWER_TASK_PART_KG = (0.5, 6.0)
DOOR_AREAL_KG_M2 = (4.0, 25.0)
RACK_AREAL_KG_M2 = (3.0, 15.0)
MINIMUM_PART_KG = 0.2


def _published(spec: Mapping[str, Any] | None, name: str) -> tuple[list[float], dict[str, Any]] | None:
    row = ((spec or {}).get("specs") or {}).get(name)
    if row is None:
        return None
    value, unit, match, urls = row.get("value"), row.get("unit"), row.get("match"), row.get("source_urls")
    # Disagreeing sources are published as a range; the interval spans it.
    span = value if isinstance(value, list) and len(value) == 2 else [value, value]
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in span)
            or span[0] > span[1] or unit not in _TO_KG or match not in PUBLISHED_RELATIVE_SPREAD
            or not isinstance(urls, list) or not urls or any(not isinstance(v, str) or not v.startswith("https://")
                                                             for v in urls)):
        raise ValueError("website_object_spec_invalid:" + name)
    low, high, spread = float(span[0]) * _TO_KG[unit], float(span[1]) * _TO_KG[unit], PUBLISHED_RELATIVE_SPREAD[match]
    return ([round(low * (1 - spread), 3), round(high * (1 + spread), 3)],
            {"mass_authority": "manufacturer_published", "spec": name, "match": match, "source_urls": list(urls)})


def _checked_spec(target: Mapping[str, Any], spec: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if spec is None:
        return None
    identity = spec.get("identity") or {}
    if (spec.get("schema_version") != SPEC_SCHEMA_VERSION
            or spec.get("digest") != canonical_digest(spec, digest_field="digest")
            or spec.get("target_id", target["target_id"]) != target["target_id"]
            or identity.get("basis") not in {"label_read", "owner_stated", "unknown"}
            or not isinstance(spec.get("specs"), Mapping)):
        raise ValueError("website_object_spec_invalid")
    # Specs of an unidentified object cannot be tied to this object.
    return None if identity["basis"] == "unknown" else spec


def articulated_mass_bounds(target: Mapping[str, Any], *, joint_type: str, body_extent_m: Sequence[float],
                            fixed_part_ids: Sequence[str] = (),
                            object_spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Body, task-part and fixed-part mass intervals with each one's authority and sources.

    ``body_extent_m`` is ``[depth, width, height]`` of the body in simulator
    metres. Published specs are read by the names research requests: ``net_weight`` (whole assembly),
    ``door_weight`` / ``drawer_weight`` (task part), and ``<part id>_weight``
    or ``rack_weight`` for fixed parts. ``mass_authority`` is ``mixed`` when
    only some intervals are published; each interval records its own.
    """
    spec = _checked_spec(target, object_spec)
    depth, width, height = (float(v) for v in body_extent_m)
    estimated = {"mass_authority": "estimated", "source_urls": []}
    if joint_type == "revolute":
        area = width * height
        task = ([round(max(MINIMUM_PART_KG, area * DOOR_AREAL_KG_M2[0]), 3),
                 round(max(1.0, area * DOOR_AREAL_KG_M2[1]), 3)],
                {**estimated, "basis": "door_front_area_times_areal_density_prior"})
        task = _published(spec, "door_weight") or task
    elif joint_type == "prismatic":
        task = (list(DRAWER_TASK_PART_KG), {**estimated, "basis": "drawer_role_prior"})
        task = _published(spec, "drawer_weight") or task
    else:
        raise ValueError("website_articulation_kind_invalid")
    fixed = {}
    for part_id in fixed_part_ids:
        footprint = depth * width
        fixed[part_id] = (_published(spec, part_id + "_weight") or _published(spec, "rack_weight")
                          or ([round(max(MINIMUM_PART_KG, footprint * RACK_AREAL_KG_M2[0]), 3),
                               round(max(1.0, footprint * RACK_AREAL_KG_M2[1]), 3)],
                              {**estimated, "basis": "tub_footprint_times_areal_density_prior"}))
    volume = max(depth * width * height, 1e-6)
    body = ([round(max(BODY_FLOOR_KG[0], BODY_DENSITY_KG_M3[0] * volume), 3),
             round(max(BODY_FLOOR_KG[1], BODY_DENSITY_KG_M3[1] * volume), 3)],
            {**estimated, "basis": "body_envelope_volume_times_density_prior"})
    whole = _published(spec, "net_weight")
    if whole is not None:
        # The body is what remains of the published whole after every other link.
        others = [task[0], *(value[0] for value in fixed.values())]
        low = whole[0][0] - sum(interval[1] for interval in others)
        high = whole[0][1] - sum(interval[0] for interval in others)
        if high <= 0:
            raise ValueError("website_object_spec_weight_inconsistent_with_parts")
        body = ([round(max(MINIMUM_PART_KG, low), 3), round(high, 3)],
                {**whole[1], "basis": "published_weight_minus_other_link_bounds"})
    value = {"mass_kg_bounds": body[0], "task_part_mass_kg_bounds": task[0],
             "provenance": {"body": body[1], "task_part": task[1],
                            **{f"fixed_part:{part_id}": row[1] for part_id, row in fixed.items()}}}
    if fixed:
        # The builder admits one interval for every fixed interior part.
        value["fixed_part_mass_kg_bounds"] = [min(row[0][0] for row in fixed.values()),
                                              max(row[0][1] for row in fixed.values())]
    authorities = {row["mass_authority"] for row in value["provenance"].values()}
    value["mass_authority"] = ("manufacturer_published" if authorities == {"manufacturer_published"}
                               else "estimated" if authorities == {"estimated"} else "mixed")
    value["source_urls"] = sorted({url for row in value["provenance"].values() for url in row["source_urls"]})
    return value
