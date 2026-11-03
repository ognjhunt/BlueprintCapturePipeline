"""Which robot models fit a website job by their public spec sheets, before any scene-build spend.

The first and cheapest tier of matching a posted job to robots that can do it. At the website
handoff, after the capability verdict (``website_rigid_feasibility``) and on the CPU only, each model
in the spec catalog (``robot_spec_catalog``) is checked against the one site video's reconstructed
geometry: can it stand at the job, reach the pick and the place, carry the object with the default
tool and keep the owner's cycle time? Policy simulation of a shortlist is a later tier.

Every number from the video is an estimate, so each check is decided across its whole band: it
fails only if it fails even at the optimistic end, passes only if it passes at the pessimistic end,
and is otherwise ``unknown``. The scene's scale may be off by the preparation's dimension prior f
(25%), taken as the band [1 - f, 1 / (1 - f)]: 0.75-1.33x, the band within which preparation admits
the registration scale. The object weighs the owner's stated weight +-10% when the text parses
cleanly, else the preparation's estimated range widened by the band cubed (mass scales with volume).

- ``stance``: a clear floor station exists: the verdict's own stations, or one ``ground_stance``
  search made the same way (``carry_station_search``) when the verdict stopped early or never
  searched. Every model stands at those stations (a fixed arm on the station's standard pedestal)
  and passes only if its published base footprint fits the cleared one; a larger or unpublished
  footprint is ``unknown``. An arm the verdict path-checks passes: the check stood it on that
  pedestal. No clear station is ``unknown`` (reconstructed floors have holes), never ``fail``.
- ``reach`` (fixed arms and mobile bases): from some station, the grasp point and the place point,
  each the object's top plus the tool's length (top-down) where it starts and where it ends, both
  lie within the published reach of the shoulder, taken as the station's base at mount height.
  Scene offsets scale across the band; the tool's length does not.
  A model on an arm the verdict path-checks takes the verdict's own carry for that arm: clear
  passes, blocked fails with its blockers, not tried is ``unknown``. A humanoid's is ``unknown``.
- ``payload``: against the rated payload, it passes only if the heaviest mass plus the default tool
  (on a fixed arm's flange; a mobile base or humanoid lifts with its own hand) is within it, and
  fails only if even the lightest mass alone exceeds it, because a lighter tool may be fitted.
- ``cycle_time``: only against the owner's target. Two straight-line trips at the published TCP
  speed bound it from below; two carry paths at a tenth of that speed plus 4 s bound it from above,
  because a published TCP speed is a peak that short, acceleration-limited moves never hold.
- ``access`` (mobile bases and humanoids): the route from the site's entrance is not checked.

A row ``fits`` when every check passed and its carry was path-checked, ``fits_spec`` when every
check passed on figures alone, ``doesnt_fit`` when any check failed, and is ``unknown`` otherwise. A
job no model fits is held, never refused. The table is informational: it gates nothing and claims
no physical validation.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .droid_policy_canary_embodiment import DROID_EMBODIMENT_ID, DROID_POLICY_CANARY_PRESET_ID
from .robot_spec_catalog import figure_value

SCHEMA = "website_robot_fit_table.v1"
FILENAME = "robot_fit_table.json"
EVIDENCE_LABEL = "spec_bounded_estimate"
CLAIM_CEILING = "development_only"
VERDICTS = ("fits", "fits_spec", "doesnt_fit", "unknown")
OWNER_WEIGHT_RELATIVE_BAND = 0.10
#: The pessimistic cycle: each carry at this fraction of the published (peak) TCP speed, plus this
#: allowance for grasping and releasing.
PESSIMISTIC_SPEED_FRACTION = 0.1
PESSIMISTIC_ALLOWANCE_S = 4.0
#: The recipe (embodiment) id of each robot profile ``robot_arm_path_clearance.ROBOT_PROFILE_ARMS``
#: names: a catalog model on that recipe takes the capability verdict's carry for the profile's arm.
PROFILE_RECIPES = {DROID_POLICY_CANARY_PRESET_ID: DROID_EMBODIMENT_ID}
_KG_PER_UNIT = {**dict.fromkeys(("kg", "kgs", "kilo", "kilos", "kilogram", "kilograms"), 1.0),
                **dict.fromkeys(("g", "gram", "grams"), 0.001),
                **dict.fromkeys(("lb", "lbs", "pound", "pounds"), 0.45359237),
                **dict.fromkeys(("oz", "ounce", "ounces"), 0.028349523125)}
# One number and one unit, optionally "about" it: "5 kg", "12 lb", "about 3kg". Ranges, bounds
# ("under 5 kg"), two figures or a missing unit do not match.
_OWNER_WEIGHT = re.compile(r"(?:(?:about|around|approximately|approx\.?|roughly|circa|~|≈)\s*)?"
                           r"([0-9]+(?:\.[0-9]+)?|\.[0-9]+)\s*([a-z]+)\.?", re.IGNORECASE)


def owner_stated_weight_kg(text: Any) -> float | None:
    """The owner's free-text item weight in kg when it states exactly one weight; ``None`` otherwise."""
    match = _OWNER_WEIGHT.fullmatch(text.strip()) if isinstance(text, str) else None
    factor = _KG_PER_UNIT.get(match.group(2).lower()) if match else None
    value = float(match.group(1)) * factor if match and factor else 0.0
    return value if math.isfinite(value) and value > 0 else None


def _r(value: float) -> float:
    return round(float(value), 6)


def _check(name: str, result: str, reason: str, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"check": name, "result": result, "reason": reason, "values": dict(values or {})}


def _within(name: str, need: Sequence[float], limit: float, values: Mapping[str, Any],
            reasons: Mapping[str, str]) -> dict[str, Any]:
    """Pass when even the pessimistic ``need[1]`` is within ``limit``; fail when even the optimistic
    ``need[0]`` exceeds it; ``unknown`` in between."""
    result = "pass" if need[1] <= limit else "fail" if need[0] > limit else "unknown"
    return _check(name, result, reasons[result], values)


def _distance_band(offset: Sequence[float], lift: float, band: Sequence[float]) -> tuple[float, float]:
    """The least and most of |s * offset + lift * z| for s across ``band``: the scene scales, a tool does not."""
    a = sum(v * v for v in offset)
    b = float(offset[2]) * lift
    scales = [*band] + ([-b / a] if a > 0 and band[0] < -b / a < band[1] else [])
    values = [math.sqrt(max(0.0, s * s * a + 2.0 * s * b + lift * lift)) for s in scales]
    return min(values), max(values)


def _carry_path_m(scale: float, *, rise: float, across: float, clearance: float,
                  inside_rise: float | None) -> float:
    """The held carry's length at scene scale ``scale``: the verdict's lift, carry and lower legs, up from
    the grasp, across above the higher end (and the container's top, for ``inside``), down onto the place."""
    top = max(0.0, scale * rise) + clearance
    if inside_rise is not None:
        top = max(top, scale * inside_rise + clearance)
    return top + scale * across + top - scale * rise


def _stations(preparation: Mapping[str, Any], verdict: Mapping[str, Any], collision: Path,
              digest: str) -> tuple[list[list[float]], str]:
    """The clear floor stations (base at mount height) and where they came from."""
    results = verdict.get("station_results") or []
    if results and verdict.get("status") != "clear":
        # Every grasp was tried from every station the search returned: these are all of them.
        unique = {tuple(row["position_world_m"]): None for row in results}
        return [list(map(float, position)) for position in unique], "capability_verdict_stations"
    blockers = [code for code in verdict.get("blockers") or [] if str(code).startswith("robot_station_")]
    if blockers:
        return [], "capability_verdict_station_search:" + ",".join(blockers)
    # A clear verdict stopped at its first carrying station; with no arm model it never searched.
    from .team_embodiment import TeamEmbodimentError
    from .team_embodiment_stance import ground_stance
    from .website_rigid_feasibility import carry_station_search

    try:
        found = ground_stance(collision_path=collision, collision_sha256=digest, **carry_station_search(preparation))
    except TeamEmbodimentError as exc:
        return [], "ground_stance_search:" + ",".join(exc.blockers)
    return [[float(v) for v in row["position_world_m"]] for row in found], "ground_stance_search"


def _stance(model: Mapping[str, Any], stations: Sequence[Sequence[float]], source: str,
            footprint_half: Sequence[float], *, path_checked: bool) -> dict[str, Any]:
    if not stations:
        return _check("stance", "unknown", "no clear floor station was found; reconstructed floors have holes",
                      {"station_source": source})
    values: dict[str, Any] = {"stations": len(stations), "station_source": source,
                              "station_footprint_half_extent_m": [_r(v) for v in footprint_half]}
    if model["body_type"] == "fixed_arm":
        values.update(mount="station_pedestal_level_with_task_support",
                      assumption="the same standard mount as the existing station")
    cleared = sorted(2.0 * float(v) for v in footprint_half)
    diameter = figure_value(model, "base_footprint_diameter_m")
    pair = figure_value(model, "base_footprint_l_w_m")
    sizes = ([[diameter, diameter]] if diameter is not None else []) + ([pair] if pair is not None else [])
    values["base_footprints_m"] = sizes
    if path_checked:
        return _check("stance", "pass", "the path check stood this arm on the station's pedestal", values)
    if not sizes:
        return _check("stance", "unknown", "no base footprint figure to compare with the cleared station", values)
    if all(all(a <= b for a, b in zip(sorted(size), cleared)) for size in sizes):
        return _check("stance", "pass", "its base fits the cleared station footprint", values)
    return _check("stance", "unknown", "its base is larger than the cleared station; its own footprint was not "
                  "searched", values)


def _verdict_reach(verdict: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """The capability verdict's own carry for ``arm`` decides reach and clearance."""
    entry = next((row for row in verdict.get("embodiments") or [] if row.get("robot_id") == arm), None)
    values: dict[str, Any] = {"decided_by": "capability_verdict_path_check", "robot_id": arm,
                              "verdict_status": verdict.get("status")}
    if entry is None:
        values["verdict_blockers"] = list(verdict.get("blockers") or [])
        return _check("reach", "unknown", "the capability verdict did not try a carry with this arm", values)
    values.update(carry_status=entry.get("status"), grasp=entry.get("grasp"), blockers=list(entry.get("blockers") or []))
    if entry.get("status") == "clear":
        return _check("reach", "pass", "the capability verdict's carry clears the reconstructed scene", values)
    if entry.get("status") == "blocked" and values["blockers"]:
        return _check("reach", "fail", "the capability verdict's carry was refused from every station tried: "
                      + ", ".join(values["blockers"]), values)
    return _check("reach", "unknown", "the capability verdict's carry for this arm is undecided", values)


def _spec_reach(model: Mapping[str, Any], stations: Sequence[Sequence[float]], points: Sequence[Sequence[float]],
                tool_length: float | None, band: Sequence[float]) -> dict[str, Any]:
    reach = figure_value(model, "max_reach_m")
    values: dict[str, Any] = {"max_reach_m": reach, "shoulder": "station base at mount height",
                              "scale_band": [_r(v) for v in band]}
    if reach is None:
        return _check("reach", "unknown", "no published reach figure", values)
    if tool_length is None:
        return _check("reach", "unknown", "the default tool's length is unknown", values)
    if not stations:
        return _check("reach", "unknown", "no clear floor station to measure from", values)
    measured = []
    for station in stations:
        bands = [_distance_band([p - s for p, s in zip(point, station)], tool_length, band) for point in points]
        measured.append((max(lo for lo, _ in bands), max(hi for _, hi in bands), station))
    optimistic = min(row[0] for row in measured)
    best = min(measured, key=lambda row: row[1])
    values.update(stations_measured=len(measured), station_world_m=[_r(v) for v in best[2]],
                  required_m={"optimistic": _r(optimistic), "pessimistic": _r(best[1])})
    return _within("reach", (optimistic, best[1]), float(reach), values, {
        "pass": "the grasp and place points are within reach from a station even at the larger scale",
        "fail": "the grasp or place point is beyond reach from every station even at the smaller scale",
        "unknown": "reach depends on the scene's scale"})


def _payload(model: Mapping[str, Any], mass_range: Sequence[float] | None, tool_mass: float) -> dict[str, Any]:
    payload = figure_value(model, "rated_payload_kg")
    # The default tool goes on a fixed arm's flange; a mobile base or humanoid lifts with its own hand.
    tool = tool_mass if model["body_type"] == "fixed_arm" else 0.0
    values: dict[str, Any] = {"rated_payload_kg": payload, "end_effector": "default_tool" if tool else "own_hand",
                              "tool_mass_kg": _r(tool)}
    if payload is None:
        return _check("payload", "unknown", "no rated payload figure", values)
    if mass_range is None:
        return _check("payload", "unknown", "no mass for the object", values)
    # A lighter tool may replace the default, so only the object's own mass can rule a model out.
    need = (mass_range[0], mass_range[1] + tool)
    plus = " plus the tool" if tool else ""
    values["required_kg"] = {"optimistic": _r(need[0]), "pessimistic": _r(need[1])}
    return _within("payload", need, float(payload), values, {
        "pass": f"the heaviest mass{plus} is within the rated payload",
        "fail": "even the lightest mass, before any tool, exceeds the rated payload",
        "unknown": f"the rated payload falls between the lightest mass and the heaviest{plus}"})


def _cycle_time(model: Mapping[str, Any], target: float | None, straight_m: float, path_m: float) -> dict[str, Any]:
    """``straight_m`` is the shortest grasp-to-place line across the band, ``path_m`` the longest carry."""
    if target is None:
        return _check("cycle_time", "not_requested", "the owner gave no cycle-time target")
    speed = figure_value(model, "max_tcp_speed_m_s")
    values: dict[str, Any] = {"target_s": target, "max_tcp_speed_m_s": speed}
    if speed is None:
        return _check("cycle_time", "unknown", "no published TCP speed figure", values)
    lower = 2.0 * straight_m / float(speed)
    upper = 2.0 * path_m / (PESSIMISTIC_SPEED_FRACTION * float(speed)) + PESSIMISTIC_ALLOWANCE_S
    values.update(lower_bound_s=_r(lower), pessimistic_s=_r(upper))
    return _within("cycle_time", (lower, upper), float(target), values, {
        "pass": "even the pessimistic cycle meets the target",
        "fail": "even two straight trips at full speed exceed the target",
        "unknown": "the target lies between the lower bound and the pessimistic cycle"})


def _mass(preparation: Mapping[str, Any], band: Sequence[float]) -> tuple[list[float] | None, str, dict[str, Any]]:
    """The object's mass band, its basis and what it was read from.

    An estimate comes from the object's measured volume, so it widens by the scale band cubed.
    """
    constraints = (((preparation.get("authoring_inputs") or {}).get("configuration") or {})
                   .get("construction_constraints") or {})
    text = (constraints.get("operator_task_details") or {}).get("item_weight")
    stated = owner_stated_weight_kg(text)
    record: dict[str, Any] = {}
    if isinstance(text, str) and text.strip():
        record["owner_item_weight"] = {"text": text, "parsed_kg": None if stated is None else _r(stated),
                                       "used": stated is not None}
    if stated is not None:
        band = OWNER_WEIGHT_RELATIVE_BAND
        return [stated * (1 - band), stated * (1 + band)], "owner_stated", record
    physics = preparation.get("physics") or {}
    estimate = (physics.get("bounds") or {}).get("mass_kg")
    if (isinstance(estimate, list) and len(estimate) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in estimate)
            and 0 <= estimate[0] <= estimate[1]):
        record.update(mass_estimate_basis=physics.get("basis"), mass_estimate_kg=[_r(v) for v in estimate],
                      mass_widened_by="scale_band_cubed")
        return [float(estimate[0]) * band[0] ** 3, float(estimate[1]) * band[1] ** 3], "estimated", record
    return None, "unavailable", record


def _cycle_target(preparation: Mapping[str, Any]) -> float | None:
    targets = (preparation.get("owner_success_criteria") or {}).get("targets") or {}
    value = targets.get("cycleTimeSeconds") if isinstance(targets, Mapping) else None
    return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0 else None)


def robot_fit_table(*, preparation: Mapping[str, Any], collision_path: Path, verdict: Mapping[str, Any],
                    catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Per catalog model, whether it fits this prepared rigid pick-and-place by spec, from CPU geometry only.

    ``verdict`` is the capability verdict computed for the same preparation and collision;
    ``catalog`` a validated spec catalog (``robot_spec_catalog.load_robot_spec_catalog``).
    """
    from .robot_arm_path_clearance import RIGID_CARRY_CLEARANCE_M, ROBOT_PROFILE_ARMS
    from .team_scene_packet import STATION_FOOTPRINT_HALF_EXTENT_M

    collision = Path(collision_path)
    digest = "sha256:" + hashlib.sha256(collision.read_bytes()).hexdigest()
    if verdict.get("preparation_digest") != preparation["digest"] or verdict.get("collision_sha256") != digest:
        raise ValueError("website_robot_fit_verdict_binding_mismatch")
    if catalog.get("catalog_digest") != canonical_digest(catalog, digest_field="catalog_digest"):
        raise ValueError("website_robot_fit_catalog_unsealed")
    fraction = preparation["authoring_inputs"]["metric_envelope"]["maximum_dimension_relative_error"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction < 1:
        raise ValueError("website_robot_fit_scale_uncertainty_invalid")
    band = (1.0 - float(fraction), 1.0 / (1.0 - float(fraction)))

    lower = [float(v) for v in preparation["subject"]["aabb_min_xyz"]]
    upper = [float(v) for v in preparation["subject"]["aabb_max_xyz"]]
    half_height = (upper[2] - lower[2]) / 2.0
    destination = preparation["destination"]
    # Scene points, before the tool: the object's top where it starts and where it is placed.
    grasp = [(lower[0] + upper[0]) / 2.0, (lower[1] + upper[1]) / 2.0, upper[2]]
    place = [float(v) for v in destination["position_world_m"]]
    place[2] += 2.0 * half_height
    tool = catalog["default_tool"]
    tool_length = figure_value(tool, "length_m")
    lift = 0.0 if tool_length is None else float(tool_length)
    inside_rise = (float(destination["interior_bounds_world_m"]["maximum"][2]) + 2.0 * half_height - grasp[2]
                   if destination.get("relation") == "inside" else None)
    rise, across = place[2] - grasp[2], math.dist(place[:2], grasp[:2])
    straight = [s * math.dist(place, grasp) for s in band]
    path = [_carry_path_m(s, rise=rise, across=across, clearance=RIGID_CARRY_CLEARANCE_M, inside_rise=inside_rise)
            for s in band]
    mass_range, mass_basis, mass_record = _mass(preparation, band)
    target = _cycle_target(preparation)
    stations, source = _stations(preparation, verdict, collision, digest)
    recipe_arms = {PROFILE_RECIPES[p]: arm for p, arm in ROBOT_PROFILE_ARMS.items() if p in PROFILE_RECIPES}

    rows = []
    for model in catalog["models"]:
        body, arm = model["body_type"], recipe_arms.get(model.get("recipe_id"))
        if body == "humanoid":
            reach = _check("reach", "unknown", "a humanoid's reach from a station is not modeled")
        elif arm is not None:
            reach = _verdict_reach(verdict, arm)
        else:
            reach = _spec_reach(model, stations, (grasp, place), tool_length, band)
        stance = _stance(model, stations, source, STATION_FOOTPRINT_HALF_EXTENT_M, path_checked=arm is not None)
        checks = [stance, reach, _payload(model, mass_range, float(figure_value(tool, "mass_kg"))),
                  _cycle_time(model, target, min(straight), max(path))]
        if body != "fixed_arm":
            checks.append(_check("access", "unknown", "path from the entrance not checked"))
        results = {row["result"] for row in checks}
        if "fail" in results:
            row_verdict = "doesnt_fit"
        elif results <= {"pass", "not_requested"}:
            row_verdict = "fits" if arm is not None and reach["result"] == "pass" else "fits_spec"
        else:
            row_verdict = "unknown"
        shown = {"doesnt_fit": "fail", "unknown": "unknown"}.get(row_verdict)
        rows.append({"model_id": model["model_id"], "recipe_id": model.get("recipe_id"), "name": model["name"],
                     "manufacturer": model["manufacturer"], "body_type": body, "verdict": row_verdict,
                     "reasons": [f"{row['check']}: {row['reason']}" for row in checks if row["result"] == shown],
                     "checks": checks})
    table: dict[str, Any] = {
        "schema_version": SCHEMA, "preparation_digest": preparation["digest"], "collision_sha256": digest,
        "catalog_digest": catalog["catalog_digest"], "verdict_digest": verdict.get("verdict_digest"),
        "evidence_label": EVIDENCE_LABEL, "claim_ceiling": CLAIM_CEILING, "physical_world_truth_claimed": False,
        "inputs": {
            "grasp_point_world_m": [_r(grasp[0]), _r(grasp[1]), _r(grasp[2] + lift)],
            "place_point_world_m": [_r(place[0]), _r(place[1]), _r(place[2] + lift)],
            "point_rule": {"grasp": "object_top_plus_tool_length_top_down",
                           "place": "destination_plus_object_height_plus_tool_length"},
            "scale_uncertainty_fraction": float(fraction), "scale_band": [_r(v) for v in band],
            "scale_basis": "preparation.authoring_inputs.metric_envelope.maximum_dimension_relative_error",
            "mass_kg_range": None if mass_range is None else [_r(v) for v in mass_range], "mass_basis": mass_basis,
            **mass_record,
            "cycle_time_target_s": target,
            "straight_line_m": [_r(v) for v in straight], "carry_path_m": [_r(v) for v in path],
            "tool": {"tool_id": tool["tool_id"], "name": tool["name"],
                     **{name: figure_value(tool, name) for name in ("stroke_m", "mass_kg", "length_m")}},
            "stations": {"source": source, "positions_world_m": [[_r(v) for v in row] for row in stations]},
        },
        "rows": rows, "counts": {name: sum(row["verdict"] == name for row in rows) for name in VERDICTS},
    }
    table["table_digest"] = canonical_digest(table, digest_field="table_digest")
    return table


__all__ = ["FILENAME", "PROFILE_RECIPES", "SCHEMA", "owner_stated_weight_kg", "robot_fit_table"]
