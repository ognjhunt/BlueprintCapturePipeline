"""Fidelity-profile templates for the two scored site-task families.

A template proposes, per family, which site readings feed which qualification
check, how the simulator side reads the same quantity, and explicit thresholds
with their basis. It is not a profile. ``profile_from_template`` refuses until a
named person has marked the thresholds reviewed, so no template acts as a
production default. The qualifier is unchanged: it still consumes only the
frozen, operator-pinned ``blueprint.site_task_fidelity_profile.v1`` that an
authorized reviewer accepted for one request.

The catalog projection (readings and checks, no thresholds) is the contract
WebApp uses to collect readings and export the source measurement document. It
is rendered to ``docs/schemas/site_task_measurement_catalog.v1.json`` and
mirrored byte for byte in WebApp ``contracts/pipeline/``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .site_task_simulation_qualification import DIMENSIONS

TEMPLATE_SCHEMA = "blueprint.site_task_fidelity_profile_template.v1"
PROFILE_SCHEMA = "blueprint.site_task_fidelity_profile.v1"
CATALOG_SCHEMA = "blueprint.site_task_measurement_catalog.v1"
SAMPLES_SCHEMA = "blueprint.site_task_measurement_samples.v1"
FAMILIES = ("rigid_pick_place", "articulated_open_close")
FAMILY_CHECK_PREFIX = {"rigid_pick_place": "pick_place.", "articulated_open_close": "open_close."}
QUANTITY_UNITS = {"length": "m", "mass": "kg", "force": "N"}
ARTICULATIONS = ("hinged", "sliding")
PROFILE_CHECK_FIELDS = ("id", "dimension", "unit", "minimum_sample_count", "maximum_absolute_error",
                        "maximum_rms_error")
TEMPLATE_CHECK_FIELDS = frozenset(PROFILE_CHECK_FIELDS) | {"sample_ids", "site_reading", "simulator_reading",
                                                           "threshold_basis"}
CATALOG_CHECK_FIELDS = ("id", "dimension", "unit", "minimum_sample_count", "sample_ids")
# Must equal the qualifier's label pattern (pinned by a test).
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DOCS = {family: Path("docs/site_task_fidelity_profiles") / (family + ".template.v1.json") for family in FAMILIES}
CATALOG_DOC = Path("docs/schemas/site_task_measurement_catalog.v1.json")

_PENDING_REVIEW = {
    "status": "pending_review",
    "reviewed_by": None,
    "reviewed_at_iso": None,
    "proposal": ("AI-assisted engineering proposal of 2026-10-19. These are not reviewed values: the founder "
                 "reviews them, and records the review here, before any profile is produced."),
}

_SCALE_BASIS = ("A careful tape reading is good to about 5 mm over 0.3 to 2 m. 2.5 cm is 1.3 to 2.5 percent of a "
                "1 to 2 m span, inside typical reach tolerance. Website scenes currently carry estimated metres, "
                "so this is the check an unmeasured scale fails.")
_CLEARANCE_BASIS = ("Long spans with soft end points. 5 cm is small against approach clearances of 0.3 m or more "
                    "and still catches a missing or phantom obstruction.")
_OBSERVATION_SIM = ("The same lengths recovered from a rendered RGB-D view at a capture or robot camera pose, using "
                    "the scene's camera intrinsics: the appearance the policy sees, not the collision geometry.")
_OBSERVATION_BASIS = ("About 7 percent of a sheet edge. Catches a render whose appearance is out of scale or "
                      "misaligned with the collision geometry, which the scale check cannot see.")


def _reading(sample_id: str, quantity: str = "length", articulation: str | None = None) -> dict[str, Any]:
    return {"sample_id": sample_id, "quantity": quantity, "articulation": articulation}


_TEMPLATES: dict[str, dict[str, Any]] = {
    "rigid_pick_place": {
        "schema_version": TEMPLATE_SCHEMA,
        "family": "rigid_pick_place",
        "readings": [
            _reading("pick_surface_height"), _reading("place_surface_height"), _reading("pick_to_place_distance"),
            _reading("object_length"), _reading("object_width"), _reading("object_height"),
            _reading("pick_front_clearance"), _reading("pick_overhead_clearance"), _reading("object_mass", "mass"),
            _reading("reference_sheet_long_edge"),
        ],
        "threshold_review": dict(_PENDING_REVIEW),
        "checks": [
            {"id": "pick_place.scale.lengths", "dimension": "scale", "unit": "m", "minimum_sample_count": 1,
             "maximum_absolute_error": 0.025, "maximum_rms_error": 0.02,
             "sample_ids": ["pick_to_place_distance", "reference_sheet_long_edge"],
             "site_reading": ("Horizontal distance from the middle of the start spot to the middle of the end spot, "
                              "and the long edge of a sheet of printer paper lying flat on the work surface while "
                              "filming."),
             "simulator_reading": ("The same lengths in the prepared scene geometry: start pose to destination "
                                   "position in the horizontal plane, and the sheet's long edge."),
             "threshold_basis": _SCALE_BASIS},
            {"id": "pick_place.registration.surface_heights", "dimension": "registration", "unit": "m",
             "minimum_sample_count": 1, "maximum_absolute_error": 0.02, "maximum_rms_error": 0.015,
             "sample_ids": ["pick_surface_height", "place_surface_height"],
             "site_reading": "Floor to the top of the surface where the object starts, and of the one where it ends.",
             "simulator_reading": ("Support-surface height above the registered floor plane at the start "
                                   "(support_plane_z_m) and at the destination."),
             "threshold_basis": ("Surface height sets the approach and contact height. About 2 cm separates a clean "
                                 "top grasp from a fingertip striking the surface for common parallel grippers.")},
            {"id": "pick_place.collision.clearances", "dimension": "collision", "unit": "m",
             "minimum_sample_count": 1, "maximum_absolute_error": 0.05, "maximum_rms_error": 0.05,
             "sample_ids": ["pick_front_clearance", "pick_overhead_clearance"],
             "site_reading": ("Open floor from the front edge of the start surface straight out to the nearest "
                              "obstruction, and open height above that surface when something is above it."),
             "simulator_reading": ("The same free distances in the scene's collision geometry, along the approach "
                                   "direction and vertically above the support surface."),
             "threshold_basis": _CLEARANCE_BASIS},
            {"id": "pick_place.task_state.object_size", "dimension": "task_state", "unit": "m",
             "minimum_sample_count": 3, "maximum_absolute_error": 0.01, "maximum_rms_error": 0.008,
             "sample_ids": ["object_length", "object_width", "object_height"],
             "site_reading": "Length, width and height of one typical object, outside edge to outside edge.",
             "simulator_reading": ("Extents of the task object's collision box in the scoring frame "
                                   "(subject_collision_bounds_scoring_frame_m)."),
             "threshold_basis": ("Object size decides grasp width against gripper stroke. 1 cm is roughly 10 to 15 "
                                 "percent of a common parallel-gripper stroke; website estimates assume 25 "
                                 "percent dimension error.")},
            {"id": "pick_place.observation.visible_lengths", "dimension": "observation", "unit": "m",
             "minimum_sample_count": 1, "maximum_absolute_error": 0.02, "maximum_rms_error": 0.02,
             "sample_ids": ["reference_sheet_long_edge", "object_length"],
             "site_reading": "The sheet's long edge and the object's length, from the same readings.",
             "simulator_reading": _OBSERVATION_SIM,
             "threshold_basis": _OBSERVATION_BASIS},
            {"id": "pick_place.task_dynamics.object_mass", "dimension": "task_dynamics", "unit": "kg",
             "minimum_sample_count": 1, "maximum_absolute_error": 0.1, "maximum_rms_error": 0.1,
             "sample_ids": ["object_mass"],
             "site_reading": "Weight of one typical object on a kitchen, postal or bathroom scale.",
             "simulator_reading": "The task object's configured mass in the prepared scene.",
             "threshold_basis": ("Covers kitchen-scale reading error and is small against typical payload margins. "
                                 "A heavy object may need a relative limit instead; an open question for the "
                                 "reviewer.")},
        ],
    },
    "articulated_open_close": {
        "schema_version": TEMPLATE_SCHEMA,
        "family": "articulated_open_close",
        "readings": [
            _reading("handle_height"), _reading("front_width"), _reading("opening_travel", articulation="sliding"),
            _reading("opening_chord", articulation="hinged"),
            _reading("handle_offset_from_hinge", articulation="hinged"), _reading("front_clearance_open"),
            _reading("opening_force", "force"), _reading("reference_sheet_long_edge"),
        ],
        "threshold_review": dict(_PENDING_REVIEW),
        "checks": [
            {"id": "open_close.scale.lengths", "dimension": "scale", "unit": "m", "minimum_sample_count": 1,
             "maximum_absolute_error": 0.025, "maximum_rms_error": 0.02,
             "sample_ids": ["front_width", "reference_sheet_long_edge"],
             "site_reading": ("Width of the door, drawer or lid front, edge to edge, and the long edge of a sheet "
                              "of printer paper lying flat in view while filming."),
             "simulator_reading": "The front's width and the sheet's long edge in the prepared scene geometry.",
             "threshold_basis": _SCALE_BASIS},
            {"id": "open_close.registration.handle_height", "dimension": "registration", "unit": "m",
             "minimum_sample_count": 1, "maximum_absolute_error": 0.02, "maximum_rms_error": 0.02,
             "sample_ids": ["handle_height"],
             "site_reading": "Floor to the middle of the handle, with the front closed.",
             "simulator_reading": "Height of handle_grasp_point_closed_world_m above the registered floor plane.",
             "threshold_basis": ("The handle height sets the grasp point. About 2 cm separates a centred grasp "
                                 "from an edge or missed grasp on common handles.")},
            {"id": "open_close.collision.open_clearance", "dimension": "collision", "unit": "m",
             "minimum_sample_count": 1, "maximum_absolute_error": 0.05, "maximum_rms_error": 0.05,
             "sample_ids": ["front_clearance_open"],
             "site_reading": "Open floor from the fully open front straight out to the nearest obstruction.",
             "simulator_reading": "The same free distance in the collision geometry with the joint at its upper limit.",
             "threshold_basis": _CLEARANCE_BASIS},
            {"id": "open_close.task_state.opening", "dimension": "task_state", "unit": "m",
             "minimum_sample_count": 1, "maximum_absolute_error": 0.03, "maximum_rms_error": 0.025,
             "sample_ids": ["opening_travel", "opening_chord", "handle_offset_from_hinge"],
             "site_reading": ("Drawer: how far the front moves when pulled out fully. Door or lid: with it fully "
                              "open, straight across from its outer edge to the frame edge it closes against, and "
                              "the distance from the hinge edge to the middle of the handle."),
             "simulator_reading": ("Prismatic travel at the joint's upper limit. For a hinge, the chord swept by the "
                                   "outer edge at the upper limit (2 r sin(theta / 2) from hinge_point_world_m and "
                                   "authored_limits_degrees) and the handle's horizontal distance from the hinge "
                                   "axis."),
             "threshold_basis": ("Opening extent fixes the joint limit the success test reads. 3 cm on a 0.4 to 1 m "
                                 "chord or travel is roughly 2 to 4 degrees, or 5 percent of range.")},
            {"id": "open_close.observation.visible_lengths", "dimension": "observation", "unit": "m",
             "minimum_sample_count": 1, "maximum_absolute_error": 0.02, "maximum_rms_error": 0.02,
             "sample_ids": ["reference_sheet_long_edge", "front_width"],
             "site_reading": "The sheet's long edge and the front's width, from the same readings.",
             "simulator_reading": _OBSERVATION_SIM,
             "threshold_basis": _OBSERVATION_BASIS},
            {"id": "open_close.task_dynamics.opening_force", "dimension": "task_dynamics", "unit": "N",
             "minimum_sample_count": 1, "maximum_absolute_error": 10.0, "maximum_rms_error": 10.0,
             "sample_ids": ["opening_force"],
             "site_reading": ("Peak reading of a luggage or spring scale hooked on the handle and pulled slowly "
                              "until the front starts to move."),
             "simulator_reading": ("Force at the handle needed to start the joint moving in the prepared scene "
                                   "(joint friction, detent and spring), from a quasi-static pull probe."),
             "threshold_basis": ("Untrained luggage-scale peaks vary by roughly 3 to 5 N. 10 N separates a light "
                                 "cabinet front (5 to 15 N) from a stiff door (30 N or more), the difference that "
                                 "changes the outcome.")},
        ],
    },
}


class FidelityTemplateError(ValueError):
    """A template is not explicit, consistent and reviewed enough for its use."""


def _require(ok: Any, code: str) -> None:
    if not ok:
        raise FidelityTemplateError("site_task_fidelity_template_" + code)


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _time(value: Any) -> bool:
    if not (isinstance(value, str) and value.endswith("Z")):
        return False
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").tzinfo is not None
    except ValueError:
        return False


def template(family: str) -> dict[str, Any]:
    """A deep copy of the source-of-truth template for one family."""
    _require(family in _TEMPLATES, "family_unknown")
    return copy.deepcopy(_TEMPLATES[family])


def _validate_review(review: Any) -> None:
    _require(isinstance(review, Mapping)
             and set(review) == {"status", "reviewed_by", "reviewed_at_iso", "proposal"}
             and isinstance(review.get("proposal"), str) and review["proposal"].strip(), "review_invalid")
    if review["status"] == "reviewed":
        _require(isinstance(review["reviewed_by"], str) and _LABEL.fullmatch(review["reviewed_by"])
                 and _time(review["reviewed_at_iso"]), "review_invalid")
    else:
        _require(review["status"] == "pending_review" and review["reviewed_by"] is None
                 and review["reviewed_at_iso"] is None, "review_invalid")


def validate_template(value: Any) -> dict[str, Any]:
    """Return a copy of a well-formed template; every threshold must be explicit and positive."""
    _require(isinstance(value, Mapping)
             and set(value) == {"schema_version", "family", "readings", "threshold_review", "checks"}
             and value.get("schema_version") == TEMPLATE_SCHEMA and value.get("family") in FAMILIES,
             "template_invalid")
    row = copy.deepcopy(dict(value))
    family = row["family"]
    _validate_review(row["threshold_review"])
    readings: dict[str, dict[str, Any]] = {}
    _require(isinstance(row["readings"], list) and row["readings"], "reading_invalid")
    for reading in row["readings"]:
        allowed = (None,) if family == "rigid_pick_place" else (None, *ARTICULATIONS)
        _require(isinstance(reading, Mapping) and set(reading) == {"sample_id", "quantity", "articulation"}
                 and isinstance(reading["sample_id"], str) and _LABEL.fullmatch(reading["sample_id"])
                 and reading["sample_id"] not in readings and isinstance(reading["quantity"], str)
                 and reading["quantity"] in QUANTITY_UNITS and reading["articulation"] in allowed, "reading_invalid")
        readings[reading["sample_id"]] = dict(reading)
    _require(isinstance(row["checks"], list) and row["checks"], "check_invalid")
    check_ids: set[str] = set()
    used: set[str] = set()
    for check in row["checks"]:
        _require(isinstance(check, Mapping) and set(check) == TEMPLATE_CHECK_FIELDS, "check_invalid")
        samples = check["sample_ids"]
        _require(isinstance(check["id"], str) and _LABEL.fullmatch(check["id"]) and check["id"] not in check_ids
                 and check["id"].startswith(FAMILY_CHECK_PREFIX[family])
                 and isinstance(check["dimension"], str) and check["dimension"] in DIMENSIONS
                 and isinstance(samples, list) and samples and all(isinstance(sample, str) for sample in samples)
                 and len(set(samples)) == len(samples)
                 and all(sample in readings for sample in samples)
                 and all(QUANTITY_UNITS[readings[sample]["quantity"]] == check["unit"] for sample in samples)
                 and type(check["minimum_sample_count"]) is int and 1 <= check["minimum_sample_count"] <= len(samples)
                 and all(_number(check[key]) and check[key] > 0
                         for key in ("maximum_absolute_error", "maximum_rms_error"))
                 and check["maximum_rms_error"] <= check["maximum_absolute_error"]
                 and all(isinstance(check[key], str) and check[key].strip()
                         for key in ("site_reading", "simulator_reading", "threshold_basis")), "check_invalid")
        check_ids.add(check["id"])
        used.update(samples)
    _require(set(readings) == used, "reading_unused")
    _require({check["dimension"] for check in row["checks"]} == DIMENSIONS, "coverage_incomplete")
    return row


def catalog() -> dict[str, Any]:
    """The threshold-free contract WebApp uses to collect and export readings."""
    families = []
    for family in FAMILIES:
        value = validate_template(template(family))
        families.append({
            "family": family,
            "readings": [{**reading, "unit": QUANTITY_UNITS[reading["quantity"]]} for reading in value["readings"]],
            "checks": [{key: check[key] for key in CATALOG_CHECK_FIELDS} for check in value["checks"]],
        })
    return {"schema_version": CATALOG_SCHEMA, "measurement_samples_schema": SAMPLES_SCHEMA, "families": families}


def render_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def profile_from_template(value: Any, *, task_id: str) -> dict[str, Any]:
    """Emit a task-bound qualifier profile; refused until a named person reviewed the thresholds."""
    row = validate_template(value)
    review = row["threshold_review"]
    _require(review["status"] == "reviewed", "thresholds_not_reviewed")
    _require(isinstance(task_id, str) and _LABEL.fullmatch(task_id), "task_id_invalid")
    return {
        "schema_version": PROFILE_SCHEMA,
        "family": row["family"],
        "task_id": task_id,
        "template_digest": digest(row),
        "thresholds_reviewed_by": review["reviewed_by"],
        "thresholds_reviewed_at_iso": review["reviewed_at_iso"],
        "checks": [{key: check[key] for key in PROFILE_CHECK_FIELDS} for check in row["checks"]],
    }


def measurement_coverage(value: Any, document: Any) -> dict[str, Any]:
    """Report, before any simulator work, whether a source document can satisfy every template check."""
    row = validate_template(value)
    _require(isinstance(document, Mapping) and document.get("schema_version") == SAMPLES_SCHEMA
             and isinstance(document.get("checks"), list), "measurement_invalid")
    articulation_of = {reading["sample_id"]: reading["articulation"] for reading in row["readings"]}
    rows: dict[str, list[Mapping[str, Any]]] = {}
    for entry in document["checks"]:
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str):
            rows.setdefault(entry["id"], []).append(entry)
    seen_articulations: set[str] = set()
    report_checks = []
    for check in row["checks"]:
        entries = rows.get(check["id"], [])
        entry = entries[0] if len(entries) == 1 else None
        samples = entry.get("samples") if entry is not None else None
        samples = samples if isinstance(samples, list) else []
        present, invalid = [], []
        for sample in samples:
            sample_id = sample.get("sample_id") if isinstance(sample, Mapping) else None
            if isinstance(sample, Mapping) and set(sample) == {"sample_id", "value"} and isinstance(sample_id, str) \
                    and _number(sample["value"]) and sample_id not in present:
                present.append(sample_id)
            else:
                invalid.append(sample_id if isinstance(sample_id, str) else repr(sample))
        unexpected = [sample_id for sample_id in present if sample_id not in check["sample_ids"]]
        allowed = [sample_id for sample_id in present if sample_id in check["sample_ids"]]
        seen_articulations.update(articulation_of[sample_id] for sample_id in allowed
                                  if articulation_of[sample_id] is not None)
        unit_mismatch = entry is not None and entry.get("unit") != check["unit"]
        dimension_mismatch = entry is not None and entry.get("dimension") != check["dimension"]
        missing = max(0, check["minimum_sample_count"] - len(allowed))
        report_checks.append({
            "id": check["id"], "dimension": check["dimension"], "present": allowed, "unexpected": unexpected,
            "invalid": invalid, "duplicate_rows": len(entries) > 1, "unit_mismatch": unit_mismatch,
            "dimension_mismatch": dimension_mismatch, "missing_required": missing,
            "satisfied": bool(entry is not None and not unexpected and not invalid and not unit_mismatch
                              and not dimension_mismatch and missing == 0),
        })
    conflict = len(seen_articulations) > 1
    missing_dimensions = sorted({check["dimension"] for check in report_checks if not check["satisfied"]})
    return {"family": row["family"], "complete": not missing_dimensions and not conflict,
            "articulation_conflict": conflict, "missing_dimensions": missing_dimensions, "checks": report_checks}


def render_documents(root: Path = _ROOT) -> list[Path]:
    """Write the reviewable template copies and the shared catalog."""
    written = []
    for family, relative in TEMPLATE_DOCS.items():
        written.append(_write(root / relative, render_json(validate_template(template(family)))))
    written.append(_write(root / CATALOG_DOC, render_json(catalog())))
    return written


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render", help="Rewrite the tracked template and catalog documents.")
    render.add_argument("--root", type=Path, default=_ROOT)
    profile = commands.add_parser("profile", help="Write a task-bound profile from a reviewed template.")
    profile.add_argument("--family", choices=FAMILIES, required=True)
    profile.add_argument("--task-id", required=True)
    profile.add_argument("--out", type=Path, required=True)
    coverage = commands.add_parser("coverage", help="Report whether a source document covers every check.")
    coverage.add_argument("--family", choices=FAMILIES, required=True)
    coverage.add_argument("--measurement", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "render":
        for path in render_documents(args.root):
            print(path)
        return 0
    if args.command == "profile":
        text = render_json(profile_from_template(template(args.family), task_id=args.task_id))
        if args.out.exists() or args.out.is_symlink():
            _require(args.out.read_text(encoding="utf-8") == text, "output_conflict")
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(text)
        print(json.dumps({"profile_digest": digest(json.loads(text)), "path": str(args.out.resolve())}))
        return 0
    report = measurement_coverage(template(args.family),
                                  json.loads(args.measurement.read_text(encoding="utf-8")))
    print(render_json(report), end="")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
