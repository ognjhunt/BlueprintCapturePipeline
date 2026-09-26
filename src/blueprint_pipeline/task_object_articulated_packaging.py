"""Compose reviewed per-part Astra candidates into one articulated SimReady assembly.

A drawer or door never leaves its cabinet, so the replacement is an assembly:
an open-front carcass plus one moving part on a single passive task joint, with
every other moving part fixed closed. Each part is authored, reviewed and
measured exactly like a rigid candidate; this module only places those exact
solids in one assembly frame (+X out of the front face, Z up, origin at the
carcass centre-XY / bottom-Z), authors the articulation, filters the
intra-assembly contacts that the joints already constrain, and seals a
development-only candidate. Interiors the footage never showed are candidate
geometry and are tagged as such; nothing here grants physics authority.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .task_object_astra_authoring import (
    AssetAuthoringError, AuthoringRequest, file_record, save_json, validate_geometry_readback,
)
from .task_object_physical_property_review import (
    PhysicalPropertyReviewInput, PhysicalPropertyReviewResult, review_physical_properties,
)
from .task_object_simready_packaging import _final_mass_consistency, _final_visual_mesh, _verified

PLAN_SCHEMA_VERSION = "articulated_assembly_plan.v1"
COMPLETION_SCHEMA_VERSION = "task_evaluation_articulated_candidate_physics_completion.v1"
AUTHORING_RESULT_SCHEMA_VERSION = "task_object_astra_articulated_authoring_result.v1"
PROVENANCE_ATTRIBUTE = "blueprint:articulatedReplacement:provenance"
OBSERVED_PROVENANCE = "observed_source_derived"
GENERATED_PROVENANCE = "generated_candidate_geometry"
TASK_CONTACT_ROLE_ATTRIBUTE = "blueprint:articulatedReplacement:taskContactRole"
REQUIRED_PARTS_ATTRIBUTE = "blueprint:articulatedReplacement:requiredParts"
HANDLE_ROLE = "handle"
TARGET_JOINT_ID = "task_part_joint"
CARCASS_LINK_ID = "carcass"
ASSET_ROOT = "/Asset"

# Object-prior construction assumptions for an unobserved interior. Every value
# is recorded in the plan; none is a measurement.
PANEL_THICKNESS_M = 0.018
FRONT_PANEL_THICKNESS_M = 0.018
HANDLE_PROTRUSION_M = 0.03
HANDLE_SECTION_M = 0.012
HANDLE_LENGTH_FRACTION_OF_FRONT = 0.6
BAY_CLEARANCE_M = 0.004
SLIDE_CLEARANCE_M = 0.013
BACK_CLEARANCE_M = 0.02
MINIMUM_RETAINED_DEPTH_M = 0.05
PASSIVE_JOINT_DAMPING_N_S_PER_M = 5.0
DEPTH_HYPOTHESIS_SCHEMA_VERSION = "articulated_cabinet_depth_hypothesis.v1"
DRAWER_FAMILY = "stacked_drawer_cabinet"
HINGED_FAMILY = "hinged_door_appliance"
FAMILY_JOINT_TYPES = {DRAWER_FAMILY: "prismatic", HINGED_FAMILY: "revolute"}
BODY_LINK_ID = "body"
DOOR_LINK_ID = "door"
NOT_CAPTURED_KIND = "not_captured_created_from_description"
REQUIRED_PART_ROLES = frozenset({"body", "task_part", "fixed_interior", "door_feature", "body_feature"})
PART_STATES = frozenset({"closed", "partially_open", "open", "not_visible"})
HINGE_EDGES = frozenset({"bottom", "left", "right", "top"})
# Built-in appliance construction priors (recorded in the plan; none is measured).
APPLIANCE_WALL_M = 0.02
APPLIANCE_DOOR_THICKNESS_M = 0.05
APPLIANCE_BACK_CLEARANCE_M = 0.05
APPLIANCE_KICKPLATE_HEIGHT_M = 0.10
APPLIANCE_CONTROL_PANEL_HEIGHT_M = 0.08
APPLIANCE_HANDLE_OFFSET_M = 0.04
RACK_SIDE_CLEARANCE_M = 0.02
RACK_FRONT_CLEARANCE_M = 0.03
RACK_BACK_CLEARANCE_M = 0.02
RACK_HEIGHT_FRACTION_OF_TUB = 0.22
LOWER_RACK_FLOOR_CLEARANCE_M = 0.06
UPPER_RACK_FLOOR_FRACTION_OF_TUB = 0.55
MINIMUM_APPLIANCE_DEPTH_TO_WIDTH = 0.4
PASSIVE_HINGE_DAMPING_N_M_S_PER_RAD = 0.5
HINGE_RESET_TOLERANCE_RAD = 0.02
# Cavity probes: rays start this far in front of the planned opening and must
# travel at least this fraction of the planned inner depth before any hit.
CAVITY_PROBE_STANDOFF_M = 0.01
CAVITY_MINIMUM_CLEAR_FRACTION = 0.6
CAVITY_PROBE_OFFSETS = (-0.3, 0.0, 0.3)
# Tub hardware the plan fixes inside a cavity is declared on it with its box;
# the probe ignores hits within that box (plus this margin) and nothing else.
# A declaration must sit inside the cavity and stay a small part of it.
CAVITY_FIXTURE_MARGIN_M = 0.01
CAVITY_FIXTURE_MAXIMUM_VOLUME_FRACTION = 0.1
# A hollow body's collider must stay hollow in PhysX, not only in its render
# mesh. convexHull fills the cavity and convexDecomposition's pieces are only
# known after cooking; a triangle mesh cooks only for a static or kinematic
# body, and the root link is dynamic. The hinged body is therefore authored as
# exact analytic boxes: its collision envelope minus the planned cavity.
CAVITY_COLLISION_WALL_BOXES = "analytic_wall_boxes"

_ORDINALS = {
    0: ("top", "upper", "uppermost", "first", "1st", "highest"),
    1: ("middle", "center", "centre", "second", "2nd", "mid"),
    2: ("bottom", "lower", "lowest", "third", "3rd", "last"),
}
_COUNT_WORDS = {"two": 2, "2": 2, "three": 3, "3": 3, "four": 4, "4": 4}


def derived_website_cabinet_depth_hypothesis(configuration: Mapping[str, Any],
                                             source_frames: Sequence[Mapping[str, Any]],
                                             opening_fraction: float) -> dict[str, Any] | None:
    """Deterministic no-spend successor from signed visible bounds and retained frames."""
    from .website_drawer_depth_prior import prior_for
    prior = prior_for(scene_id=configuration.get("scene_id"),
                      subject_identity=configuration.get("replacement_identity"))
    if (configuration.get("schema_version") != "articulated_replacement_authoring_configuration.v1"
            or configuration.get("source_observation_kind") != "website_capture_frames"
            or (configuration.get("mechanism") or {}).get("joint_type") != "prismatic"
            or prior is None
            or configuration.get("development_geometry_hypothesis") is not None):
        return None
    envelope = configuration.get("metric_envelope") or {}
    try:
        lower, upper = envelope["minimum_xyz_m"], envelope["maximum_xyz_m"]
        extents = [float(upper[i]) - float(lower[i]) for i in range(3)]
        normal = configuration["mechanism"]["estimated_front_normal_world"]
        horizontal = math.hypot(float(normal[0]), float(normal[1]))
        nx, ny = float(normal[0]) / horizontal, float(normal[1]) / horizontal
        depth = abs(nx) * extents[0] + abs(ny) * extents[1]
        width = abs(ny) * extents[0] + abs(nx) * extents[1]
        height = extents[2]
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not (height >= 0.35 and width >= 0.30 and depth < 0.25 and depth / width < 0.5):
        return None
    nominal = prior["nominal_depth_m"]
    if nominal / width > 1.5:
        return None  # A wider prior is not defensible for this object; request review.
    hashes = sorted({str(frame.get("sha256") or "") for frame in source_frames
                     if frame.get("role") == "observed_source"})
    if not hashes or any(re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None for digest in hashes):
        return None
    source_stroke = configuration["mechanism"].get("estimated_usable_stroke_m")
    if (not isinstance(source_stroke, (int, float)) or isinstance(source_stroke, bool)
            or not math.isfinite(source_stroke) or source_stroke <= 0
            or not isinstance(opening_fraction, (int, float)) or isinstance(opening_fraction, bool)
            or not math.isfinite(opening_fraction) or not 0 < opening_fraction <= 1):
        return None
    stroke = round(0.75 * nominal, 4)
    return {"schema_version": DEPTH_HYPOTHESIS_SCHEMA_VERSION,
            "claim_ceiling": "development_only", "basis": "bounded_cabinet_prior",
            "source_aabb_min_xyz_m": list(lower), "source_aabb_max_xyz_m": list(upper),
            "estimated_depth_m": nominal,
            "depth_interval_m": list(prior["depth_interval_m"]),
            "source_estimated_usable_stroke_m": source_stroke,
            "estimated_usable_stroke_m": stroke,
            "minimum_opening_fraction": opening_fraction,
            "estimated_minimum_opening_m": round(stroke * opening_fraction, 5),
            "manufacturer_examples": list(prior["manufacturer_examples"]),
            "reference_retrieved_date": prior["reference_retrieved_date"],
            "prior_record_schema_version": prior["schema_version"],
            "prior_comparison": {
                "example_depth_range_m": prior["example_depth_range_m"],
                "example_width_range_m": prior["example_width_range_m"],
                "example_height_range_m": prior["example_height_range_m"],
                "example_weight_range_kg_approx": prior["example_weight_range_kg_approx"],
                "source_width_m": round(width, 5), "source_height_m": round(height, 5),
                "width_status": "outside_examples_review_needed" if width > prior["example_width_range_m"][1] else "within_example_range",
                "height_status": "outside_examples_review_needed" if height > prior["example_height_range_m"][1] else "within_example_range",
                "reference_models_are_exact_match": False,
                **({"owner_reported_dimensions_m": dict(prior["owner_reported_dimensions_m"]),
                    "owner_reported_source": "cabinet_owner_chat_2026-09-24"}
                   if "owner_reported_dimensions_m" in prior else {}),
            },
            "whole_assembly_mass_interval_kg": list(prior["whole_assembly_mass_interval_kg"]),
            "revised_part_mass_bounds_kg": dict(prior["revised_part_mass_bounds_kg"]),
            "rationale": ("Original closed-drawer frames identify an office cabinet but do not show its back. "
                          "A broad cabinet construction prior supplies candidate depth; physical depth is unmeasured."),
            "evidence_frame_sha256s": hashes}


def _depth_hypothesis(configuration: Mapping[str, Any], *, source_depth: float,
                      width: float, height: float) -> dict[str, Any] | None:
    """Admit an explicit development estimate while retaining the source box intact."""
    from .website_drawer_depth_prior import prior_for
    prior = prior_for(scene_id=configuration.get("scene_id"),
                      subject_identity=configuration.get("replacement_identity"))
    raw = configuration.get("development_geometry_hypothesis")
    implausibly_thin = height >= 0.35 and width >= 0.30 and source_depth < 0.25 and source_depth / width < 0.5
    if raw is None:
        if configuration.get("source_observation_kind") == "website_capture_frames" and implausibly_thin:
            raise AssetAuthoringError("articulated_cabinet_depth_implausible_hypothesis_required")
        return None
    if prior is None:
        raise AssetAuthoringError("articulated_cabinet_depth_hypothesis_invalid")
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version", "claim_ceiling", "basis", "source_aabb_min_xyz_m", "source_aabb_max_xyz_m",
        "estimated_depth_m", "depth_interval_m", "rationale", "evidence_frame_sha256s",
        "source_estimated_usable_stroke_m", "estimated_usable_stroke_m",
        "minimum_opening_fraction", "estimated_minimum_opening_m",
        "manufacturer_examples", "reference_retrieved_date", "prior_comparison",
        "whole_assembly_mass_interval_kg", "revised_part_mass_bounds_kg", "prior_record_schema_version",
    } or raw.get("schema_version") != DEPTH_HYPOTHESIS_SCHEMA_VERSION or raw.get("claim_ceiling") != "development_only":
        raise AssetAuthoringError("articulated_cabinet_depth_hypothesis_invalid")
    envelope = configuration["metric_envelope"]
    if (raw["source_aabb_min_xyz_m"] != envelope["minimum_xyz_m"]
            or raw["source_aabb_max_xyz_m"] != envelope["maximum_xyz_m"]):
        raise AssetAuthoringError("articulated_cabinet_depth_hypothesis_source_mismatch")
    basis = raw.get("basis")
    frames = raw.get("evidence_frame_sha256s")
    interval = raw.get("depth_interval_m")
    nominal = raw.get("estimated_depth_m")
    stroke = raw.get("estimated_usable_stroke_m")
    source_stroke = raw.get("source_estimated_usable_stroke_m")
    fraction = raw.get("minimum_opening_fraction")
    opening = raw.get("estimated_minimum_opening_m")
    mechanism = configuration["mechanism"]
    if (configuration.get("source_observation_kind") != "website_capture_frames"
            or basis not in {"original_capture_frames", "bounded_cabinet_prior"}
            or not isinstance(frames, list) or len(frames) != len(set(map(str, frames)))
            or not frames
            or any(not isinstance(v, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", v) is None for v in frames)
            or not isinstance(raw.get("rationale"), str) or len(raw["rationale"].strip()) < 20
            or not isinstance(interval, list) or len(interval) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in [nominal, *interval, stroke, source_stroke, fraction, opening])
            or not 0.25 <= interval[0] < nominal < interval[1] <= 0.80
            or not 0.4 <= nominal / width <= 1.5
            or not 0 < fraction <= 1
            or not math.isclose(stroke, round(0.75 * nominal, 4), abs_tol=1e-8)
            or not math.isclose(opening, round(stroke * fraction, 5), abs_tol=1e-8)
            or mechanism.get("estimated_usable_stroke_m") != stroke
            or mechanism.get("joint_limits") != [0.0, stroke]
            or raw.get("whole_assembly_mass_interval_kg") != prior["whole_assembly_mass_interval_kg"]
            or raw.get("revised_part_mass_bounds_kg") != prior["revised_part_mass_bounds_kg"]
            or configuration.get("required_output", {}).get("mass_kg_bounds") != prior["revised_part_mass_bounds_kg"]["carcass"]
            or configuration.get("required_output", {}).get("task_part_mass_kg_bounds") != prior["revised_part_mass_bounds_kg"]["drawer"]
            or raw.get("reference_retrieved_date") != prior["reference_retrieved_date"]
            or raw.get("prior_record_schema_version") != prior["schema_version"]
            or not isinstance(raw.get("manufacturer_examples"), list)
            or len(raw["manufacturer_examples"]) != 4
            or not isinstance(raw.get("prior_comparison"), Mapping)
            or raw["prior_comparison"].get("source_width_m") != round(width, 5)
            or raw["prior_comparison"].get("source_height_m") != round(height, 5)
            or raw["prior_comparison"].get("reference_models_are_exact_match") is not False
            or ("owner_reported_dimensions_m" in prior and
                (raw["prior_comparison"].get("owner_reported_dimensions_m") != prior["owner_reported_dimensions_m"]
                 or raw["prior_comparison"].get("owner_reported_source") != "cabinet_owner_chat_2026-09-24"))):
        raise AssetAuthoringError("articulated_cabinet_depth_hypothesis_invalid")
    return {**dict(raw), "source_projected_depth_m": round(source_depth, 5),
            "depth_disagreement_m": round(nominal - source_depth, 5),
            "status": "development_only_estimate_disagrees_with_source" if abs(nominal - source_depth) > 1e-5
            else "development_only_estimate_agrees_with_source", "physical_measurement_proven": False}


def _resolve_bay_layout(assembly_label: str, part_label: str) -> tuple[int, int, list[str]]:
    """Resolve how many stacked bays exist and which one the task names (recorded assumptions)."""
    assumptions: list[str] = []
    label = assembly_label.lower()
    match = re.search(r"\b(two|three|four|2|3|4)[\s-]*drawer", label)
    count = _COUNT_WORDS[match.group(1)] if match else 0
    words = re.findall(r"[a-z0-9]+", part_label.lower())
    index = next((k for k, names in _ORDINALS.items() if any(w in names for w in words)), None)
    if index is None:
        raise AssetAuthoringError("articulated_task_part_position_unresolved:" + part_label)
    if count == 0:
        count = 3 if index == 1 else max(index + 1, 2)
        assumptions.append(f"bay_count_assumed_{count}_from_part_label")
    if index == 2 and count > 3:
        index = count - 1  # "bottom" of a taller stack
    if index >= count:
        raise AssetAuthoringError("articulated_task_part_outside_assembly:" + part_label)
    return count, index, assumptions


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def assembly_family(configuration: Mapping[str, Any]) -> str:
    """The declared assembly family; a legacy prismatic configuration is a drawer cabinet."""
    joint_type = (configuration.get("mechanism") or {}).get("joint_type")
    family = configuration.get("assembly_family")
    if family is None and joint_type == "prismatic":
        return DRAWER_FAMILY
    if family not in FAMILY_JOINT_TYPES:
        raise AssetAuthoringError("articulated_assembly_family_unsupported:" + str(family or joint_type))
    if FAMILY_JOINT_TYPES[family] != joint_type:
        raise AssetAuthoringError(f"articulated_assembly_family_joint_mismatch:{family}:{joint_type}")
    return family


def assembly_contract(configuration: Mapping[str, Any], family: str) -> dict[str, Any]:
    """Validate one object's whole-assembly contract: optional for legacy drawers, required for doors.

    A pure function of this object's configuration, so several task objects
    plan and check independently. An object created from a description (not
    captured) carries no frames and may claim no observation.
    """
    captured = configuration.get("source_observation_kind") != NOT_CAPTURED_KIND
    required = family == HINGED_FAMILY
    frames = configuration.get("reference_frames")
    if frames is None and not (required and captured):
        frames = []
    if (not isinstance(frames, list) or len(frames) > 16 or (not captured and frames)
            or (required and captured and not frames)):
        raise AssetAuthoringError("articulated_reference_frames_invalid")
    frame_ids: list[str] = []
    for row in frames:
        if (not isinstance(row, Mapping)
                or not isinstance(row.get("frame_id"), str) or not row["frame_id"]
                or not isinstance(row.get("sha256"), str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", row["sha256"]) is None
                or not isinstance(row.get("visible_parts"), list)
                or any(not isinstance(v, str) or not v for v in row["visible_parts"])
                or row.get("part_state") not in PART_STATES
                or not isinstance(row.get("view"), str) or not row["view"].strip()
                or not isinstance(row.get("reason"), str) or not row["reason"].strip()
                or not _finite_number(row.get("timestamp_seconds"))):
            raise AssetAuthoringError("articulated_reference_frames_invalid")
        frame_ids.append(row["frame_id"])
    if len(frame_ids) != len(set(frame_ids)) or len({row["sha256"] for row in frames}) != len(frames):
        raise AssetAuthoringError("articulated_reference_frames_invalid")
    parts = configuration.get("required_parts")
    if parts is None and not required:
        parts = []
    if not isinstance(parts, list) or (required and not parts):
        raise AssetAuthoringError("articulated_required_parts_missing")
    normalized = []
    for row in parts:
        observed = row.get("observed_frame_ids") if isinstance(row, Mapping) else None
        if (not isinstance(row, Mapping)
                or not isinstance(row.get("part_id"), str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", row["part_id"]) is None
                or not isinstance(row.get("label"), str) or not row["label"].strip()
                or row.get("role") not in REQUIRED_PART_ROLES
                or not isinstance(observed, list) or any(v not in frame_ids for v in observed)
                or (not captured and observed)):
            raise AssetAuthoringError("articulated_required_parts_invalid")
        normalized.append({"part_id": row["part_id"], "label": row["label"].strip(), "role": row["role"],
                           "observed_frame_ids": list(observed)})
    if len({row["part_id"] for row in normalized}) != len(normalized) or (
            normalized and not any(row["role"] == "task_part" for row in normalized)):
        raise AssetAuthoringError("articulated_required_parts_invalid")
    depth = configuration.get("body_depth")
    body_depth = None
    if depth is not None or required:
        basis = "interior_observed_open_state" if captured else "owner_or_catalog_specified"
        ids = depth.get("frame_ids") if isinstance(depth, Mapping) else None
        if (not isinstance(depth, Mapping) or not _finite_number(depth.get("value_m")) or depth["value_m"] <= 0
                or depth.get("basis") != basis or not isinstance(ids, list)
                or any(v not in frame_ids for v in ids) or (captured and not ids)):
            raise AssetAuthoringError("articulated_body_depth_missing_or_invalid")
        body_depth = {"value_m": float(depth["value_m"]), "basis": basis, "frame_ids": list(ids)}
    hinge = configuration.get("hinge_edge")
    if required and hinge not in HINGE_EDGES:
        raise AssetAuthoringError("articulated_hinge_edge_missing_or_invalid")
    return {"captured": captured, "reference_frames": [dict(row) for row in frames],
            "required_parts": normalized, "body_depth": body_depth, "hinge_edge": hinge if required else None}


def _projected_envelope(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Body depth, width and height in the assembly frame.

    An axis-aligned world envelope of a yawed body is wider and deeper than
    the body. When the contract carries the oriented ``body_extent_m``, that
    is used and the envelope must be its axis-aligned bound on the front normal.
    """
    envelope = configuration["metric_envelope"]
    lower, upper = envelope["minimum_xyz_m"], envelope["maximum_xyz_m"]
    extents = [float(upper[i]) - float(lower[i]) for i in range(3)]
    normal = [float(v) for v in configuration["mechanism"]["estimated_front_normal_world"]]
    horizontal = math.hypot(normal[0], normal[1])
    if horizontal < 1e-9:
        raise AssetAuthoringError("articulated_front_normal_invalid")
    nx, ny = abs(normal[0] / horizontal), abs(normal[1] / horizontal)
    value = {"lower": list(lower), "upper": list(upper), "normal": normal,
             "yaw": math.atan2(normal[1] / horizontal, normal[0] / horizontal),
             "depth": nx * extents[0] + ny * extents[1], "width": ny * extents[0] + nx * extents[1],
             "height": extents[2], "extent_authority": "envelope_projected_on_front_normal"}
    body = configuration.get("body_extent_m")
    if body is None:
        return value
    if (not isinstance(body, Mapping)
            or any(not _finite_number(body.get(key)) or body[key] <= 0 for key in ("depth", "width", "height"))):
        raise AssetAuthoringError("articulated_body_extent_invalid")
    depth, width, height = (float(body[key]) for key in ("depth", "width", "height"))
    bound = [depth * nx + width * ny, depth * ny + width * nx, height]
    tolerance = float(envelope["maximum_dimension_relative_error"])
    # Structural only on the coverage path: that envelope is levelled from this same body.
    if any(abs(bound[i] - extents[i]) > tolerance * max(bound[i], extents[i]) for i in range(3)):
        raise AssetAuthoringError("articulated_body_extent_disagrees_with_envelope")
    return {**value, "depth": depth, "width": width, "height": height, "extent_authority": "oriented_body_extent"}


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _match_vocabulary(row: Mapping[str, Any], vocabulary: Sequence[tuple[tuple[str, ...], str]]) -> str | None:
    """Match the part id first, then its label; the first vocabulary entry that hits wins."""
    for words in (_tokens(row["part_id"]), _tokens(row["label"])):
        for keys, name in vocabulary:
            if any(word in keys for word in words):
                return name
    return None


_BODY_FEATURES = (
    (("tub", "interior", "cavity", "inside", "liner"), "tub_cavity"),
    (("kickplate", "kick", "toe", "plinth", "base"), "kickplate"),
    (("left",), "left_side_wall"),
    (("right",), "right_side_wall"),
    (("top", "worktop", "countertop"), "top_panel"),
    (("back", "rear"), "back_wall"),
    (("front", "frame", "face", "opening", "trim"), "front_frame"),
    (("body", "shell", "housing", "cabinet", "carcass", "chassis", "appliance"), "link"),
)
_DOOR_FEATURES = (
    (("handle", "pull", "grip"), "handle"),
    (("control", "controls", "button", "buttons", "display", "dial", "keypad", "console"), "control_panel"),
    (("brand", "logo", "label", "badge", "nameplate"), "brand_label"),
    (("inner", "liner", "inside", "interior"), "inner_liner"),
    (("outer", "skin", "front", "face", "panel"), "outer_skin"),
    (("door",), "link"),
)
_CARCASS_FEATURES = (
    (("left",), "left_side_panel"), (("right",), "right_side_panel"), (("top",), "top_panel"),
    (("back", "rear"), "back_panel"), (("front", "frame", "face", "opening", "bay", "bays"), "open_front_bays"),
    (("body", "carcass", "cabinet", "case", "frame", "shell"), "link"),
)
_DRAWER_FEATURES = (
    (("handle", "pull", "grip"), "handle"), (("front", "face", "panel"), "drawer_front"),
    (("box", "interior", "inside", "tray"), "drawer_box"), (("drawer",), "link"),
)


def _tub_fixture(row: Mapping[str, Any]) -> str | None:
    """Fixed tub hardware that belongs to the body (not a separate rack link)."""
    words = set(_tokens(row["part_id"])) | set(_tokens(row["label"]))
    if words & {"spray", "sprayer", "sprayers", "wash"} and words & {"arm", "arms", "sprayer", "sprayers"}:
        return "upper_spray_arm" if words & {"upper", "middle", "top", "second", "2nd"} else "lower_spray_arm"
    if words & {"filter", "filters", "sump", "drain"}:
        return "floor_filter"
    if words & {"heating", "heater"} or {"heat", "element"} <= words:
        return "heating_element"
    return None


def _interior_slot(row: Mapping[str, Any], *, sole_rack: bool = False) -> str | None:
    words = set(_tokens(row["part_id"])) | set(_tokens(row["label"]))
    if words & {"cutlery", "silverware", "utensil", "utensils", "flatware"}:
        return "third_tray" if words & {"tray", "third", "3rd"} else "cutlery_basket"
    if not words & {"rack", "racks", "basket", "tray", "shelf"}:
        return None
    if words & {"third", "3rd"}:
        return "third_tray"
    if words & {"upper", "top", "second", "2nd"}:
        return "upper_rack"
    if words & {"lower", "bottom", "first", "1st"} or (sole_rack and words & {"rack", "racks"}):
        return "lower_rack"
    return None


def _box(center: Sequence[float], size: Sequence[float]) -> dict[str, list[float]]:
    return {"minimum": [round(c - s / 2, 5) for c, s in zip(center, size)],
            "maximum": [round(c + s / 2, 5) for c, s in zip(center, size)]}


def _place_role_rows(rows: Sequence[Mapping[str, Any]], *, link_id: str, vocabulary, whole_role: str,
                     features: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    placed = {}
    for row in rows:
        feature = _match_vocabulary(row, vocabulary) or ("link" if row["role"] == whole_role else None)
        if feature is None or features.get(feature) is None:
            raise AssetAuthoringError("articulated_required_part_unplanned:" + row["part_id"])
        placed[row["part_id"]] = {"link_id": link_id, "feature": feature}
    return placed


def _plan_hinged_door_appliance(configuration: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    """A built-in appliance: open-front hollow body, one hinged door, fixed interior parts.

    Assembly frame: +X out of the closed door face, Z up, origin at the closed
    envelope centre-XY / bottom-Z. The body occupies the rear of the envelope
    and the closed door its front ``APPLIANCE_DOOR_THICKNESS_M``.
    """
    mechanism = configuration["mechanism"]
    source = _projected_envelope(configuration)
    width, height = source["width"], source["height"]
    tolerance = float(configuration["metric_envelope"]["maximum_dimension_relative_error"])
    body_depth = contract["body_depth"]
    depth = body_depth["value_m"]
    if min(depth, source["depth"]) / width < MINIMUM_APPLIANCE_DEPTH_TO_WIDTH:
        raise AssetAuthoringError("articulated_body_depth_implausibly_thin")
    # Structural only when body_extent_m is present (coverage path): both depths are the same observed body;
    # a published size that contradicts it is held upstream as an object_spec blocker.
    if abs(depth - source["depth"]) > tolerance * max(depth, source["depth"]):
        raise AssetAuthoringError("articulated_body_depth_disagrees_with_envelope")
    swing = mechanism.get("estimated_usable_swing_rad")
    hinge = contract["hinge_edge"]
    if (not _finite_number(swing) or not 0.1 <= swing <= (math.pi / 2 if hinge == "bottom" else math.pi)
            or list(mechanism.get("joint_limits", [0.0, swing])) != [0.0, swing]):
        raise AssetAuthoringError("articulated_hinge_swing_infeasible")
    rows = {role: [row for row in contract["required_parts"] if row["role"] in roles] for role, roles in (
        ("body", {"body", "body_feature"}), ("door", {"task_part", "door_feature"}), ("interior", {"fixed_interior"}))}
    t, door_t, hp, s = APPLIANCE_WALL_M, APPLIANCE_DOOR_THICKNESS_M, HANDLE_PROTRUSION_M, HANDLE_SECTION_M
    base = (APPLIANCE_KICKPLATE_HEIGHT_M if any(_match_vocabulary(row, _BODY_FEATURES) == "kickplate"
                                                for row in rows["body"]) else 0.0)
    panel_h = (APPLIANCE_CONTROL_PANEL_HEIGHT_M if any(_match_vocabulary(row, _DOOR_FEATURES) == "control_panel"
                                                       for row in rows["door"]) else 0.0)
    body_x, door_h = depth - door_t, height - base
    tub_depth, tub_w = body_x - APPLIANCE_BACK_CLEARANCE_M, width - 2 * t
    tub_floor, tub_ceiling = base + t, height - t
    tub_h = tub_ceiling - tub_floor
    if min(tub_depth, tub_w, tub_h) <= 0.15 or door_h <= 0.2:
        raise AssetAuthoringError("articulated_appliance_geometry_infeasible")
    front_x = depth / 2 - door_t  # body front plane, assembly frame
    door_x = door_t + hp
    outer_face_x = door_x / 2 - hp
    if hinge in {"bottom", "top"}:
        handle_z = door_h - panel_h - APPLIANCE_HANDLE_OFFSET_M if hinge == "bottom" else APPLIANCE_HANDLE_OFFSET_M
        handle_center = [round(door_x / 2 - s / 2, 5), 0.0, round(handle_z, 5)]
        handle = {"center_m": handle_center, "length_m": round(HANDLE_LENGTH_FRACTION_OF_FRONT * width, 4),
                  "section_m": s, "axis": "Y", "grasp_point_link_m": handle_center}
    else:  # Seen from the front the viewer's left is -Y; the handle sits on the free edge.
        handle_y = (width / 2 - APPLIANCE_HANDLE_OFFSET_M) * (1 if hinge == "left" else -1)
        handle_center = [round(door_x / 2 - s / 2, 5), round(handle_y, 5), round(door_h / 2, 5)]
        handle = {"center_m": handle_center, "length_m": round(HANDLE_LENGTH_FRACTION_OF_FRONT * door_h, 4),
                  "section_m": s, "axis": "Z", "grasp_point_link_m": handle_center}
    door_features = {
        "link": _box([0.0, 0.0, door_h / 2], [door_x, width, door_h]),
        "outer_skin": _box([outer_face_x - 0.001, 0.0, door_h / 2], [0.002, width, door_h]),
        "inner_liner": _box([-door_x / 2 + 0.001, 0.0, door_h / 2], [0.002, width - 2 * t, door_h - 2 * t]),
        "handle": _box(handle_center, [s, handle["length_m"], s] if handle["axis"] == "Y" else [s, s, handle["length_m"]]),
        "control_panel": _box([outer_face_x - 0.001, 0.0, door_h - panel_h / 2], [0.002, width, panel_h]) if panel_h else None,
        "brand_label": _box([outer_face_x - 0.0005, 0.0, door_h - (panel_h or 0.1) / 2], [0.001, 0.12, 0.03]),
    }
    body_features = {
        "link": _box([0.0, 0.0, height / 2], [body_x, width, height]),
        "tub_cavity": _box([body_x / 2 - tub_depth / 2, 0.0, (tub_floor + tub_ceiling) / 2], [tub_depth, tub_w, tub_h]),
        "front_frame": _box([body_x / 2 - t / 2, 0.0, (base + height) / 2], [t, width, height - base]),
        "left_side_wall": _box([0.0, -width / 2 + t / 2, height / 2], [body_x, t, height]),
        "right_side_wall": _box([0.0, width / 2 - t / 2, height / 2], [body_x, t, height]),
        "top_panel": _box([0.0, 0.0, height - t / 2], [body_x, width, t]),
        "back_wall": _box([-body_x / 2 + APPLIANCE_BACK_CLEARANCE_M / 2, 0.0, height / 2],
                          [APPLIANCE_BACK_CLEARANCE_M, width, height]),
        "kickplate": _box([body_x / 2 - t / 2, 0.0, base / 2], [t, width, base]) if base else None,
    }
    placed = {**_place_role_rows(rows["body"], link_id=BODY_LINK_ID, vocabulary=_BODY_FEATURES,
                                 whole_role="body", features=body_features),
              **_place_role_rows(rows["door"], link_id=DOOR_LINK_ID, vocabulary=_DOOR_FEATURES,
                                 whole_role="task_part", features=door_features)}
    slots: dict[str, str] = {}
    fixtures: dict[str, str] = {}
    # One rack the footage never qualifies as upper or lower is the lower rack.
    sole_rack = sum(1 for row in rows["interior"]
                    if _tub_fixture(row) is None and _interior_slot(row, sole_rack=True) is not None) == 1
    for row in rows["interior"]:
        slot = _interior_slot(row, sole_rack=sole_rack)
        fixture = _tub_fixture(row) if slot is None else None
        if row["part_id"] in {BODY_LINK_ID, DOOR_LINK_ID} or (slot is None and fixture is None):
            raise AssetAuthoringError("articulated_required_part_unplanned:" + row["part_id"])
        if fixture is not None:
            if fixture in fixtures.values():
                raise AssetAuthoringError("articulated_required_part_unplanned:" + row["part_id"])
            fixtures[row["part_id"]] = fixture
            placed[row["part_id"]] = {"link_id": BODY_LINK_ID, "feature": fixture}
            continue
        if slot in slots.values():
            raise AssetAuthoringError("articulated_required_part_unplanned:" + row["part_id"])
        slots[row["part_id"]] = slot
        placed[row["part_id"]] = {"link_id": row["part_id"], "feature": "link"}
    # Fixed interior parts sit inside the tub, clear of the closed door's liner.
    rack_w = tub_w - 2 * RACK_SIDE_CLEARANCE_M
    rack_d = tub_depth - RACK_FRONT_CLEARANCE_M - RACK_BACK_CLEARANCE_M
    rack_h = RACK_HEIGHT_FRACTION_OF_TUB * tub_h
    rack_x = front_x - RACK_FRONT_CLEARANCE_M - rack_d / 2
    lower_z = tub_floor + LOWER_RACK_FLOOR_CLEARANCE_M
    upper_z = tub_floor + UPPER_RACK_FLOOR_FRACTION_OF_TUB * tub_h
    tray_h = 0.06
    tray_z = tub_ceiling - 0.01 - tray_h
    basket = [min(0.25, 0.45 * rack_d), min(0.15, 0.35 * rack_w),
              min(0.2, 0.8 * (upper_z - lower_z) if "upper_rack" in slots.values() else 0.35 * tub_h)]
    slot_geometry = {
        "lower_rack": ([rack_d, rack_w, rack_h], [rack_x, 0.0, lower_z],
                       "Lower rack: open-top wire basket resting near the tub floor"),
        "upper_rack": ([rack_d, rack_w, rack_h], [rack_x, 0.0, upper_z],
                       "Upper rack: open-top wire basket in the upper half of the tub"),
        "third_tray": ([rack_d, rack_w, tray_h], [rack_x, 0.0, tray_z], "Shallow tray just below the tub ceiling"),
        "cutlery_basket": (basket, [front_x - RACK_FRONT_CLEARANCE_M - basket[0] / 2 - 0.01,
                                    rack_w / 2 - basket[1] / 2 - 0.01, lower_z + 0.01],
                           "Open-top basket seated inside the lower rack footprint at the front right"),
    }
    # Tub hardware stays on the body, low and thin so the tub remains open.
    tub_x = body_x / 2 - tub_depth / 2
    fixture_geometry = {
        "lower_spray_arm": ([0.05, 0.8 * tub_w, 0.03], [tub_x, 0.0, tub_floor + 0.035],
                            "lower spray arm: a flat bar across the tub just above the floor"),
        "upper_spray_arm": ([0.05, 0.7 * tub_w, 0.03], [tub_x, 0.0, upper_z - 0.03],
                            "upper spray arm: a flat bar across the tub just below the upper rack"),
        "floor_filter": ([0.12, 0.12, 0.04], [tub_x - 0.15 * tub_depth, 0.0, tub_floor + 0.02],
                         "filter: a round cup set into the tub floor behind its centre"),
        "heating_element": ([0.7 * tub_depth, 0.7 * tub_w, 0.012], [tub_x, 0.0, tub_floor + 0.008],
                            "heating element: a thin loop lying on the tub floor"),
    }
    fixture_text = []
    for part_id, fixture in fixtures.items():
        size, center, text = fixture_geometry[fixture]
        if fixture == "upper_spray_arm" and "upper_rack" not in slots.values():
            raise AssetAuthoringError("articulated_interior_geometry_infeasible:" + part_id)
        body_features[fixture] = _box(center, size)
        label = next(row["label"] for row in rows["interior"] if row["part_id"] == part_id)
        fixture_text.append(f"{label} ({text}, {round(size[0], 3)} x {round(size[1], 3)} x {round(size[2], 3)} m)")
    interior_parts: dict[str, dict[str, Any]] = {}
    interior_links = []
    for part_id, slot in slots.items():
        size, rest, text = slot_geometry[slot]
        if (min(size) <= 0.03 or rest[2] + size[2] > tub_ceiling + 1e-9
                or (slot == "upper_rack" and "third_tray" in slots.values() and upper_z + rack_h > tray_z)):
            raise AssetAuthoringError("articulated_interior_geometry_infeasible:" + part_id)
        label = next(row["label"] for row in rows["interior"] if row["part_id"] == part_id)
        interior_parts[part_id] = {
            "link_role": "fixed_interior", "dimensions_m": [round(v, 5) for v in size],
            "features": {"link": _box([0.0, 0.0, size[2] / 2], size)},
            "description": (f"{label}. {text}: {round(size[0], 4)} m deep x {round(size[1], 4)} m wide x "
                            f"{round(size[2], 4)} m tall overall, open top with thin slotted or wire walls and floor; "
                            "held fixed inside the tub, not a task part.")}
        interior_links.append({"link_id": part_id, "part_id": part_id, "is_root": False,
                               "semantic_role": "fixed_interior", "rest_translation_m": [round(v, 5) for v in rest]})
    # Positive rotation opens the door outward (+X); a negative axis uses a
    # 180 degree joint frame about X so the USD axis token stays Y or Z.
    axis, token, rotation = {"bottom": ([0.0, 1.0, 0.0], "Y", [1.0, 0.0, 0.0, 0.0]),
                             "top": ([0.0, -1.0, 0.0], "Y", [0.0, 1.0, 0.0, 0.0]),
                             "right": ([0.0, 0.0, 1.0], "Z", [1.0, 0.0, 0.0, 0.0]),
                             "left": ([0.0, 0.0, -1.0], "Z", [0.0, 1.0, 0.0, 0.0])}[hinge]
    pivot = {"bottom": [depth / 2, 0.0, base], "top": [depth / 2, 0.0, height],
             "left": [depth / 2, -width / 2, base + door_h / 2], "right": [depth / 2, width / 2, base + door_h / 2]}[hinge]
    label = str(mechanism["task_part_label"])
    features = {row["feature"] for row in placed.values()}
    tub = {"cavity_id": "tub", "link_id": BODY_LINK_ID,
           "opening_center_link_m": [round(body_x / 2, 5), 0.0, round((tub_floor + tub_ceiling) / 2, 5)],
           "opening_width_m": round(tub_w, 5), "opening_height_m": round(tub_h, 5),
           "inner_depth_m": round(tub_depth, 5)}
    if fixtures:
        tub["declared_fixtures"] = [{"fixture_id": fixture, "part_id": part_id, "box_link_m": body_features[fixture]}
                                    for part_id, fixture in sorted(fixtures.items(), key=lambda item: item[1])]
    # The plan must pass its own probe before any part is bought.
    refused = planned_cavity_findings(body_features, [tub])
    if refused:
        raise AssetAuthoringError("articulated_plan_" + refused[0])
    return {
        "schema_version": PLAN_SCHEMA_VERSION, "family": HINGED_FAMILY, "root_link_id": BODY_LINK_ID,
        "assembly_frame": {"front_axis": "+X", "up_axis": "Z", "origin": "closed_envelope_center_xy_bottom_z",
                           "world_yaw_rad_from_estimated_front_normal": source["yaw"],
                           "estimated_front_normal_world": source["normal"]},
        "assembly_dimensions_m": {"depth_x": round(depth, 5), "width_y": round(width, 5), "height_z": round(height, 5),
                                  "authority": "body_depth_" + body_depth["basis"]},
        "closed_collision_dimensions_m": [round(depth + hp, 5), round(width, 5), round(height, 5)],
        "source_geometry": {"aabb_min_xyz_m": source["lower"], "aabb_max_xyz_m": source["upper"],
                            "projected_depth_m": round(source["depth"], 5), "projected_width_m": round(width, 5),
                            "height_m": round(height, 5), "authority": "retained_source_envelope_not_physical_measurement"},
        "body_depth": {**body_depth, "envelope_projected_depth_m": round(source["depth"], 5),
                       "relative_disagreement": round(abs(depth - source["depth"]) / depth, 5)},
        "hinge_edge": hinge,
        "construction_assumptions": [
            f"wall_thickness_m={t}", f"door_thickness_m={door_t}", f"back_clearance_m={APPLIANCE_BACK_CLEARANCE_M}",
            f"kickplate_height_m={base}", f"control_panel_height_m={panel_h}", f"handle_protrusion_m={hp}",
            f"handle_section_m={s}", f"rack_height_fraction_of_tub={RACK_HEIGHT_FRACTION_OF_TUB}",
            "interior_parts_fixed_closed_not_sliding", "handle_modelled_as_bar_grasp_feature",
            "door_counterbalance_spring_not_modelled_only_viscous_damping", "body_treated_as_grounded_base",
            "unobserved_surfaces_are_generated_candidate_geometry",
        ],
        "parts": {
            BODY_LINK_ID: {
                "link_role": "body", "dimensions_m": [round(body_x, 5), round(width, 5), round(height, 5)],
                "features": body_features,
                "description": (
                    f"Body shell of the appliance behind the {label}: the front (+X face) is open onto a hollow tub "
                    f"cavity {round(tub_depth, 4)} m deep x {round(tub_w, 4)} m wide x {round(tub_h, 4)} m tall "
                    f"(part-frame z {round(tub_floor, 4)} to {round(tub_ceiling, 4)}); left and right side walls and "
                    f"top panel {t} m thick, tub floor {t} m thick, closed back wall and service space "
                    f"{APPLIANCE_BACK_CLEARANCE_M} m thick"
                    + (f", and a solid base {round(base, 4)} m tall whose front face is the kickplate" if base else "")
                    + (". Fixed to the tub as part of this body: " + "; ".join(fixture_text) if fixture_text else "")
                    + ". The tub must otherwise stay empty and open to the front; the door, racks and baskets are "
                    "separate parts."),
            },
            DOOR_LINK_ID: {
                "link_role": "task_part", "dimensions_m": [round(door_x, 5), round(width, 5), round(door_h, 5)],
                "features": door_features, "handle": handle,
                "description": (
                    f"The {label}: one solid panel {door_t} m thick spanning the full {round(width, 4)} m width and "
                    f"{round(door_h, 4)} m height at the -X side of this part, hinged along its {hinge} edge; outer "
                    "skin on its +X face and inner liner on its -X face"
                    + (f", a flush control panel strip {panel_h} m tall along the top edge of the outer face"
                       if panel_h else "")
                    + f", and a bar handle {handle['length_m']} m long along {handle['axis']} with a {s} m square "
                    f"section protruding {hp} m in +X (bar centre at part-frame {handle_center})"
                    + (", with the brand label as a flush visual region on the outer skin"
                       if "brand_label" in features else "") + "."),
            },
            **interior_parts,
        },
        "links": [
            {"link_id": BODY_LINK_ID, "part_id": BODY_LINK_ID, "is_root": True, "semantic_role": "appliance_body",
             "rest_translation_m": [round(-door_t / 2, 5), 0.0, 0.0]},
            {"link_id": DOOR_LINK_ID, "part_id": DOOR_LINK_ID, "is_root": False, "semantic_role": "task_door",
             "rest_translation_m": [round(depth / 2 + (hp - door_t) / 2, 5), 0.0, round(base, 5)]},
            *interior_links,
        ],
        "task_joint": {"joint_id": TARGET_JOINT_ID, "joint_type": "revolute", "parent_link_id": BODY_LINK_ID,
                       "child_link_id": DOOR_LINK_ID, "axis_asset_frame": axis, "usd_axis": token,
                       "joint_frame_rotation_wxyz": rotation, "anchor_asset_frame_m": [round(v, 5) for v in pivot],
                       "limits_rad": [0.0, float(swing)], "reset_position_rad": 0.0,
                       "travel_authority": mechanism.get("travel_authority", "object_prior_estimate"),
                       "drive": {"drive_type": "none", "stiffness": 0.0, "damping": PASSIVE_HINGE_DAMPING_N_M_S_PER_RAD,
                                 "maximum_force": 0.0, "implementation": "passive_torque_damper",
                                 "damping_units": "N_m_s_per_rad"}},
        "interior_cavities": [tub],
        "cavity_collision_approximation": CAVITY_COLLISION_WALL_BOXES,
        "required_parts": [{**row, **placed[row["part_id"]]} for row in contract["required_parts"]],
        "source_observation": "captured" if contract["captured"] else NOT_CAPTURED_KIND,
        "intra_assembly_collision": "filtered_joints_constrain_mechanism",
        "lock_status": str(mechanism.get("lock_status") or "unknown"),
        "physical_measurement_proven": False,
    }


_CONTRACT_KEYS = ("required_parts", "reference_frames", "body_depth", "body_extent_m", "hinge_edge")


def legacy_drawer_plan(plan: Mapping[str, Any]) -> bool:
    """origin/main's drawer plan shape: no contract parts, no planned cavities (none were ever checked)."""
    return plan.get("family") == DRAWER_FAMILY and not {"interior_cavities", "required_parts",
                                                        "root_link_id"} & set(plan)


def plan_articulated_assembly(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Derive exact part envelopes, rest poses and the task joint from one object's stage-3 configuration.

    The assembly frame has +X out of the front face and Z up. Width/depth come
    from the estimated world envelope projected onto the estimated front normal;
    they inherit that estimate's uncertainty and are labelled so. Every
    contract required part must land on a planned link or named feature.
    """
    family = assembly_family(configuration)
    contract = assembly_contract(configuration, family)
    # A legacy (familyless) drawer plan is origin/main's and would drop them.
    if "assembly_family" not in configuration and any(configuration.get(key) for key in _CONTRACT_KEYS):
        raise AssetAuthoringError("articulated_assembly_family_undeclared")
    plan = (_plan_hinged_door_appliance(configuration, contract) if family == HINGED_FAMILY
            else _plan_stacked_drawer_cabinet(configuration, contract))
    # Joint damping resists velocity but does nothing at rest. Use the admitted
    # *joint* effort interval for a provisional breakaway model on every new
    # articulated task. This is an estimate, never a measured property of the
    # captured unit or a direct transfer of a comparable product's pull force.
    bounds = (configuration.get("mechanism") or {}).get("passive_dynamics", {}).get("joint_friction_bounds")
    if "assembly_family" in configuration and isinstance(bounds, list) and len(bounds) == 2:
        low, high = (float(value) for value in bounds)
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
            raise AssetAuthoringError("articulated_joint_friction_bounds_invalid")
        plan["task_joint"]["passive_friction"] = {
            "static_effort": round((low + high) / 2, 6),
            "dynamic_effort": round(low, 6),
            "admitted_interval": [low, high],
            "units": "N" if plan["task_joint"]["joint_type"] == "prismatic" else "N_m",
            "basis": "estimated_unobserved_joint_resistance_prior",
            "physical_measurement_proven": False,
        }
        plan["construction_assumptions"] = [*plan["construction_assumptions"],
            "joint_static_and_dynamic_friction_estimated_from_admitted_prior"]
    reference = (configuration.get("mechanism") or {}).get("opening_effort_reference")
    if reference is not None:
        plan["task_joint"]["opening_effort_reference"] = dict(reference)
        plan["construction_assumptions"] = [*plan["construction_assumptions"],
            "published_opening_effort_is_comparison_not_direct_joint_friction_or_unit_measurement"]
    return plan


def _plan_stacked_drawer_cabinet(configuration: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = configuration["mechanism"]
    source = _projected_envelope(configuration)
    lower, upper, normal = source["lower"], source["upper"], source["normal"]
    source_depth, width, height = source["depth"], source["width"], source["height"]
    hypothesis = _depth_hypothesis(configuration, source_depth=source_depth, width=width, height=height)
    depth = float(hypothesis["estimated_depth_m"]) if hypothesis else source_depth
    owner_dimensions = (hypothesis.get("prior_comparison") or {}).get("owner_reported_dimensions_m") if hypothesis else None
    if owner_dimensions:
        width, height = float(owner_dimensions["width"]), float(owner_dimensions["height"])
    if min(depth, width, height) <= 6 * PANEL_THICKNESS_M:
        raise AssetAuthoringError("articulated_assembly_envelope_too_small")
    count, task_index, assumptions = _resolve_bay_layout(
        str(configuration.get("authoring_target") or ""), str(mechanism["task_part_label"]))
    t = PANEL_THICKNESS_M
    bay_height = (height - (count + 1) * t) / count
    bay_width = width - 2 * t
    box_depth = depth - t - BACK_CLEARANCE_M
    if bay_height <= 0.04 or bay_width <= 0.05 or box_depth <= MINIMUM_RETAINED_DEPTH_M + 0.02:
        raise AssetAuthoringError("articulated_bay_geometry_infeasible")
    drawer_x = HANDLE_PROTRUSION_M + FRONT_PANEL_THICKNESS_M + box_depth
    drawer_y = bay_width - BAY_CLEARANCE_M
    drawer_z = bay_height - BAY_CLEARANCE_M
    stroke = min(float(mechanism["estimated_usable_stroke_m"]), box_depth - MINIMUM_RETAINED_DEPTH_M)
    if hypothesis is not None and not math.isclose(stroke, hypothesis["estimated_usable_stroke_m"], abs_tol=1e-8):
        raise AssetAuthoringError("articulated_depth_hypothesis_stroke_clamped")
    if stroke <= 0.02:
        raise AssetAuthoringError("articulated_usable_stroke_infeasible")
    handle_length = round(HANDLE_LENGTH_FRACTION_OF_FRONT * drawer_y, 4)
    handle_center = [round(drawer_x / 2 - HANDLE_SECTION_M / 2, 5), 0.0, round(drawer_z / 2, 5)]
    bays = []
    for k in range(count):
        floor_z = t + (count - 1 - k) * (bay_height + t)
        bays.append({"bay_index": k, "link_id": f"drawer_{k}", "is_task_part": k == task_index,
                     "rest_translation_m": [round(depth / 2 - drawer_x / 2 + HANDLE_PROTRUSION_M, 5), 0.0,
                                            round(floor_z + BAY_CLEARANCE_M / 2, 5)]})
    yaw = source["yaw"]
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "family": DRAWER_FAMILY,
        "assembly_frame": {"front_axis": "+X", "up_axis": "Z", "origin": "carcass_center_xy_bottom_z",
                           "world_yaw_rad_from_estimated_front_normal": yaw,
                           "estimated_front_normal_world": normal},
        "assembly_dimensions_m": {"depth_x": round(depth, 5), "width_y": round(width, 5), "height_z": round(height, 5),
                                  "authority": ("development_only_owner_reported_dimensions" if owner_dimensions else
                                                "development_only_depth_hypothesis") if hypothesis else
                                               "estimated_envelope_projected_on_estimated_front_normal"},
        "source_geometry": {"aabb_min_xyz_m": list(lower), "aabb_max_xyz_m": list(upper),
                            "projected_depth_m": round(source_depth, 5),
                            "projected_width_m": round(source["width"], 5), "height_m": round(source["height"], 5),
                            "authority": "retained_source_envelope_not_physical_measurement"},
        **({"development_geometry_hypothesis": hypothesis} if hypothesis else {}),
        "bay_count": count, "task_bay_index": task_index,
        "construction_assumptions": [
            f"panel_thickness_m={t}", f"front_panel_thickness_m={FRONT_PANEL_THICKNESS_M}",
            f"handle_protrusion_m={HANDLE_PROTRUSION_M}", f"handle_section_m={HANDLE_SECTION_M}",
            f"equal_bay_heights={round(bay_height, 5)}", f"slide_clearance_m={SLIDE_CLEARANCE_M}",
            "interior_and_drawer_boxes_unobserved_generated_candidate_geometry",
            "carcass_treated_as_grounded_base_casters_not_modelled",
            "static_breakaway_friction_not_modelled_only_viscous_damping",
            *assumptions,
        ],
        "parts": {
            CARCASS_LINK_ID: {
                "link_role": "carcass", "dimensions_m": [round(depth, 5), round(width, 5), round(height, 5)],
                "description": (f"Open-front cabinet carcass: top, bottom, back, left and right panels {t} m thick "
                                f"plus {count - 1} horizontal dividers forming {count} equal drawer bays "
                                f"({round(bay_width, 4)} m wide x {round(bay_height, 4)} m tall x {round(depth - t, 4)} m deep). "
                                "The front (+X face) is fully open; no drawer fronts, handles or feet belong to this part."),
            },
            "drawer": {
                "link_role": "task_part", "dimensions_m": [round(drawer_x, 5), round(drawer_y, 5), round(drawer_z, 5)],
                "description": (f"One drawer: a front panel {FRONT_PANEL_THICKNESS_M} m thick spanning the full Y width and Z height "
                                f"at the +X end, a centred horizontal bar handle {handle_length} m long with a {HANDLE_SECTION_M} m "
                                f"square section protruding {HANDLE_PROTRUSION_M} m in +X from the front panel (its bar centre at "
                                f"part-frame {handle_center}), and behind the front an open-top drawer box "
                                f"{round(box_depth, 4)} m deep, {round(drawer_y - 2 * SLIDE_CLEARANCE_M, 4)} m wide, "
                                f"{round(drawer_z - 0.03, 4)} m tall with 0.012 m walls. The same solid is instanced "
                                f"for all {count} bays."),
                "handle": {"center_m": handle_center, "length_m": handle_length, "section_m": HANDLE_SECTION_M,
                           "axis": "Y", "grasp_point_link_m": handle_center},
            },
        },
        "links": [{"link_id": CARCASS_LINK_ID, "part_id": CARCASS_LINK_ID, "is_root": True, "semantic_role": "cabinet_carcass",
                   "rest_translation_m": [0.0, 0.0, 0.0]},
                  *[{"link_id": bay["link_id"], "part_id": "drawer", "is_root": False,
                     "semantic_role": "task_drawer" if bay["is_task_part"] else "fixed_drawer",
                     "rest_translation_m": bay["rest_translation_m"], "bay_index": bay["bay_index"]} for bay in bays]],
        "task_joint": {"joint_id": TARGET_JOINT_ID, "joint_type": "prismatic", "parent_link_id": CARCASS_LINK_ID,
                       "child_link_id": f"drawer_{task_index}", "axis_asset_frame": [1.0, 0.0, 0.0],
                       "limits_m": [0.0, round(stroke, 5)], "reset_position_m": 0.0,
                       "stroke_authority": mechanism.get("travel_authority", "object_prior_estimate"),
                       "drive": {"drive_type": "none", "stiffness": 0.0, "damping": PASSIVE_JOINT_DAMPING_N_S_PER_M,
                                 "maximum_force": 0.0, "implementation": "passive_force_damper"}},
        "intra_assembly_collision": "filtered_joints_constrain_mechanism",
        "lock_status": str(mechanism.get("lock_status") or "unknown"),
        "physical_measurement_proven": False,
    }
    if "assembly_family" not in configuration:
        # A legacy configuration keeps origin/main's exact plan, so its part
        # requests (and any carcass already bought for them) are unchanged.
        return plan
    carcass_features = dict.fromkeys(("link", "left_side_panel", "right_side_panel", "top_panel", "back_panel",
                                      "open_front_bays"), "carcass_panel")
    drawer_features = dict.fromkeys(("link", "handle", "drawer_front", "drawer_box"), "drawer_solid")
    placed = {**_place_role_rows([r for r in contract["required_parts"] if r["role"] in {"body", "body_feature"}],
                                 link_id=CARCASS_LINK_ID, vocabulary=_CARCASS_FEATURES, whole_role="body",
                                 features=carcass_features),
              **_place_role_rows([r for r in contract["required_parts"] if r["role"] in {"task_part", "door_feature"}],
                                 link_id=f"drawer_{task_index}", vocabulary=_DRAWER_FEATURES, whole_role="task_part",
                                 features=drawer_features)}
    for row in contract["required_parts"]:
        if row["role"] != "fixed_interior":
            continue
        words = _tokens(row["part_id"]) + _tokens(row["label"])
        index = next((k for k, names in _ORDINALS.items() if any(w in names for w in words)), None)
        index = count - 1 if index == 2 and count > 3 else index
        if index is None or index == task_index or index >= count or "drawer" not in words:
            raise AssetAuthoringError("articulated_required_part_unplanned:" + row["part_id"])
        placed[row["part_id"]] = {"link_id": f"drawer_{index}", "feature": "link"}
    plan["parts"][CARCASS_LINK_ID]["features"] = carcass_features
    plan["parts"]["drawer"]["features"] = drawer_features
    return {
        **plan, "root_link_id": CARCASS_LINK_ID,
        "closed_collision_dimensions_m": [round(depth + HANDLE_PROTRUSION_M, 5), round(width, 5), round(height, 5)],
        "interior_cavities": [{"cavity_id": f"bay_{bay['bay_index']}", "link_id": CARCASS_LINK_ID,
                               "opening_center_link_m": [round(depth / 2, 5), 0.0,
                                                         round(t + (count - 1 - bay["bay_index"]) * (bay_height + t)
                                                               + bay_height / 2, 5)],
                               "opening_width_m": round(bay_width, 5), "opening_height_m": round(bay_height, 5),
                               "inner_depth_m": round(depth - t, 5)} for bay in bays],
        "required_parts": [{**row, **placed[row["part_id"]]} for row in contract["required_parts"]],
    }


def root_link_id(plan: Mapping[str, Any]) -> str:
    return str(plan.get("root_link_id") or CARCASS_LINK_ID)


def task_joint_limits(task_joint: Mapping[str, Any]) -> list[float]:
    """SI limits of the task joint: metres for prismatic, radians for revolute."""
    key = "limits_rad" if task_joint["joint_type"] == "revolute" else "limits_m"
    return [float(v) for v in task_joint[key]]


def required_parts_by_link(plan: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """``{link_id: {required_part_id: feature}}``; refuses a part no planned link carries."""
    links = {row["link_id"]: row["part_id"] for row in plan["links"]}
    placed: dict[str, dict[str, str]] = {}
    for row in plan.get("required_parts") or []:
        part = plan["parts"].get(links.get(row.get("link_id")), {})
        if row.get("feature") not in (part.get("features") or {}) and row.get("feature") != "link":
            raise AssetAuthoringError("articulated_required_part_unplanned:" + str(row.get("part_id")))
        placed.setdefault(row["link_id"], {})[row["part_id"]] = row["feature"]
    return placed


def required_part_findings(plan: Mapping[str, Any], observed: Mapping[str, Mapping[str, str]]) -> list[str]:
    """Every contract required part must be carried by the link and feature the plan assigned it."""
    try:
        expected = required_parts_by_link(plan)
    except (AssetAuthoringError, KeyError, TypeError):
        return ["required_parts_plan_invalid"]
    return sorted(f"required_part_missing:{part_id}" for link_id, rows in expected.items()
                  for part_id, feature in rows.items() if (observed.get(link_id) or {}).get(part_id) != feature)


def _declared_fixture_boxes(cavity: Mapping[str, Any]) -> list[tuple[list[float], list[float]]] | None:
    """A cavity's declared fixture boxes, or None when any is malformed, outside it or too large."""
    rows = cavity.get("declared_fixtures", [])
    if not isinstance(rows, list):
        return None
    cx, cy, cz = (float(v) for v in cavity["opening_center_link_m"])
    depth, half_w, half_h = (float(cavity[k]) for k in ("inner_depth_m", "opening_width_m", "opening_height_m"))
    half_w, half_h = half_w / 2, half_h / 2
    c_lo, c_hi = [cx - depth, cy - half_w, cz - half_h], [cx, cy + half_w, cz + half_h]
    boxes = []
    for row in rows:
        box = row.get("box_link_m") if isinstance(row, Mapping) else None
        try:
            lo, hi = [float(v) for v in box["minimum"]], [float(v) for v in box["maximum"]]
        except (KeyError, TypeError, ValueError):
            return None
        if (len(lo) != 3 or len(hi) != 3 or not all(map(math.isfinite, lo + hi))
                or any(not c_lo[i] - 1e-4 <= lo[i] < hi[i] <= c_hi[i] + 1e-4 for i in range(3))  # 5 dp rounding
                or math.prod(hi[i] - lo[i] for i in range(3))
                > CAVITY_FIXTURE_MAXIMUM_VOLUME_FRACTION * depth * 4 * half_w * half_h):
            return None
        boxes.append((lo, hi))
    return boxes


def _box_mesh(boxes: Sequence[tuple[Sequence[float], Sequence[float]]]) -> tuple[list[list[float]], list[int]]:
    """Closed axis-aligned boxes as one triangle soup (12 outward triangles each)."""
    corners = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    vertices: list[list[float]] = []
    faces: list[int] = []
    for lo, hi in boxes:
        base = len(vertices)
        vertices.extend([[float((lo, hi)[s][axis]) for axis, s in enumerate(corner)] for corner in corners])
        for q in quads:
            faces.extend(base + v for v in (q[0], q[1], q[2], q[0], q[2], q[3]))
    return vertices, faces


# Planned features that are not solids for the self-probe: the link envelope,
# the void itself, and the front frame (a ring around the opening, not a slab).
_PLANNED_NON_SOLID_FEATURES = frozenset({"link", "tub_cavity", "front_frame"})


def planned_cavity_findings(features: Mapping[str, Any], cavities: Sequence[Mapping[str, Any]]) -> list[str]:
    """The cavity probe on a link's own plan: its wall boxes plus every planned solid feature box.

    Run at planning time so a plan whose own geometry would fail the probe is
    refused before any part is bought.
    """
    try:
        walls = cavity_wall_boxes(features["link"], cavities)
    except AssetAuthoringError as exc:
        return [str(exc).removeprefix("articulated_")]
    boxes = [([c - s / 2 for c, s in zip(w["center_m"], w["size_m"])],
              [c + s / 2 for c, s in zip(w["center_m"], w["size_m"])]) for w in walls]
    boxes += [(box["minimum"], box["maximum"]) for name, box in features.items()
              if isinstance(box, Mapping) and name not in _PLANNED_NON_SOLID_FEATURES]
    vertices, faces = _box_mesh(boxes)
    return interior_cavity_findings(vertices, faces, cavities)


def interior_cavity_findings(vertices: Any, faces: Any, cavities: Sequence[Mapping[str, Any]]) -> list[str]:
    """Rays from just outside each planned opening, along -X, must travel into a hollow and hit a closed back.

    Mesh coordinates are in the cavity's link frame. The first hit of every
    probe ray must lie at least ``CAVITY_MINIMUM_CLEAR_FRACTION`` of the planned
    inner depth past the opening; a solid block or a thin slab stops the ray
    early, a missing back lets it escape. Hits inside a box the cavity declares
    under ``declared_fixtures`` (planned tub hardware) are not obstructions;
    an undeclared, oversized or out-of-cavity box is.
    """
    import numpy as np

    triangles = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int).reshape(-1, 3)]
    if not len(cavities):
        return ["body_interior_cavity_unplanned"]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    # Rays run along -X, so intersections reduce to 2-D point-in-triangle tests in YZ.
    e1, e2 = b[:, 1:] - a[:, 1:], c[:, 1:] - a[:, 1:]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    usable = np.abs(det) > 1e-14
    m = CAVITY_FIXTURE_MARGIN_M

    def clear_depth(origin_x: float, y: float, z: float, fixtures) -> float | None:
        rel = np.array([y, z]) - a[:, 1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (rel[:, 0] * e2[:, 1] - rel[:, 1] * e2[:, 0]) / det
            v = (e1[:, 0] * rel[:, 1] - e1[:, 1] * rel[:, 0]) / det
            inside = usable & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        hit_x = a[inside, 0] + u[inside] * (b[inside, 0] - a[inside, 0]) + v[inside] * (c[inside, 0] - a[inside, 0])
        keep = hit_x < origin_x
        for lo, hi in fixtures:
            if lo[1] - m <= y <= hi[1] + m and lo[2] - m <= z <= hi[2] + m:
                keep &= (hit_x < lo[0] - m) | (hit_x > hi[0] + m)
        ahead = hit_x[keep]
        return float(origin_x - ahead.max()) - CAVITY_PROBE_STANDOFF_M if len(ahead) else None

    findings = []
    for cavity in cavities:
        cx, cy, cz = (float(v) for v in cavity["opening_center_link_m"])
        half_w, half_h = float(cavity["opening_width_m"]) / 2, float(cavity["opening_height_m"]) / 2
        fixtures = _declared_fixture_boxes(cavity)
        if fixtures is None:
            findings.append("body_interior_cavity_fixture_invalid:" + str(cavity["cavity_id"]))
            continue
        depths = [clear_depth(cx + CAVITY_PROBE_STANDOFF_M, cy + dy * half_w, cz + dz * half_h, fixtures)
                  for dy in CAVITY_PROBE_OFFSETS for dz in CAVITY_PROBE_OFFSETS]
        if any(depth is None for depth in depths):
            findings.append("body_interior_cavity_back_open:" + str(cavity["cavity_id"]))
        elif min(depths) < CAVITY_MINIMUM_CLEAR_FRACTION * float(cavity["inner_depth_m"]):
            findings.append("body_interior_cavity_closed:" + str(cavity["cavity_id"]))
    return findings


def cavity_wall_boxes(bounds: Mapping[str, Sequence[float]],
                      cavities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Axis-aligned boxes filling a link's collision envelope except its one open-front cavity.

    ``bounds`` is the link-frame collision envelope; the cavity runs from its
    planned back to the envelope's +X face. Every wall must have positive
    thickness, so the union is exactly the envelope minus the cavity.
    """
    if len(cavities) != 1:
        raise AssetAuthoringError("articulated_body_cavity_collision_infeasible")
    lo = [float(v) for v in bounds["minimum"]]
    hi = [float(v) for v in bounds["maximum"]]
    cavity = cavities[0]
    cx, cy, cz = (float(v) for v in cavity["opening_center_link_m"])
    half_w, half_h = float(cavity["opening_width_m"]) / 2, float(cavity["opening_height_m"]) / 2
    c_lo = [cx - float(cavity["inner_depth_m"]), cy - half_w, cz - half_h]
    c_hi = [hi[0], cy + half_w, cz + half_h]
    if not (lo[0] < c_lo[0] < c_hi[0] and all(lo[i] < c_lo[i] < c_hi[i] < hi[i] for i in (1, 2))):
        raise AssetAuthoringError("articulated_body_cavity_collision_infeasible")
    walls = {"wall_back": (lo, [c_lo[0], hi[1], hi[2]]),
             "wall_left": ([c_lo[0], lo[1], lo[2]], [hi[0], c_lo[1], hi[2]]),
             "wall_right": ([c_lo[0], c_hi[1], lo[2]], hi),
             "wall_floor": ([c_lo[0], c_lo[1], lo[2]], [hi[0], c_hi[1], c_lo[2]]),
             "wall_ceiling": ([c_lo[0], c_lo[1], c_hi[2]], [hi[0], c_hi[1], hi[2]])}
    return [{"name": name, "center_m": [(a + b) / 2 for a, b in zip(low, high)],
             "size_m": [b - a for a, b in zip(low, high)]} for name, (low, high) in walls.items()]


def box_collision_piece(center: Sequence[float], size: Sequence[float]) -> dict[str, Any]:
    """An analytic box collider as link-frame triangles, for the cavity probe."""
    import trimesh

    box = trimesh.creation.box(extents=[float(v) for v in size])
    box.apply_translation([float(v) for v in center])
    return {"approximation": "analytic_box", "vertices": box.vertices.tolist(), "faces": box.faces.tolist()}


def collision_cavity_findings(pieces: Sequence[Mapping[str, Any]],
                              cavities: Sequence[Mapping[str, Any]]) -> list[str]:
    """Probe each planned cavity on the geometry PhysX will collide with, per declared approximation.

    Pieces are link-frame triangles. An ``analytic_box`` is exact, a
    ``convexHull`` mesh is probed as its hull, and any other approximation
    (``convexDecomposition``, a triangle mesh on a dynamic body) cannot be
    proven hollow before cooking, so it fails closed.
    """
    import numpy as np
    import trimesh

    vertices: list[Any] = []
    faces: list[Any] = []
    unproven = []
    for piece in pieces:
        approximation = str(piece.get("approximation") or "")
        points = np.asarray(piece.get("vertices") or [], dtype=float).reshape(-1, 3)
        triangles = np.asarray(piece.get("faces") or [], dtype=int).reshape(-1, 3)
        if approximation == "convexHull" and len(points) >= 4:
            hull = trimesh.Trimesh(points, triangles, process=False).convex_hull
            points, triangles = np.asarray(hull.vertices), np.asarray(hull.faces)
        elif approximation != "analytic_box":
            unproven.append("body_cavity_collision_approximation_unproven:" + (approximation or "unspecified"))
            continue
        faces.append(triangles + sum(len(v) for v in vertices))
        vertices.append(points)
    if unproven:
        return sorted(set(unproven))
    if not vertices:
        return ["body_cavity_collision_missing"]
    codes = interior_cavity_findings(np.concatenate(vertices), np.concatenate(faces), cavities)
    return [code.replace("body_interior_cavity_", "body_cavity_collision_", 1) for code in codes]


def _part_physics(*, request: AuthoringRequest, authoring_result: Mapping[str, Any],
                  physics_bounds: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Re-verify one part exactly as the rigid packager does, without writing USD."""
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    if authoring_result.get("request_digest") != request.request_digest:
        raise AssetAuthoringError("authoring_packaging_request_mismatch")
    if authoring_result.get("status") != "candidate_authored_pending_native_qualification":
        raise AssetAuthoringError("authoring_packaging_candidate_not_reviewed")
    if authoring_result.get("result_digest") != canonical_digest(authoring_result, digest_field="result_digest"):
        raise AssetAuthoringError("authoring_packaging_result_digest_mismatch")
    review_path = _verified(authoring_result["physical_review"])
    review = PhysicalPropertyReviewResult.model_validate_json(review_path.read_text())
    review_input = PhysicalPropertyReviewInput.model_validate_json(
        _verified(authoring_result["physical_review_input"]).read_text())
    if review_physical_properties(review_input, review.proposed).model_dump(mode="json") != review.model_dump(mode="json"):
        raise AssetAuthoringError("authoring_packaging_physics_review_not_reproducible")
    if review.accepted is None or review.blockers or review.claim_ceiling != "development_only":
        raise AssetAuthoringError("authoring_packaging_physics_not_accepted")
    if (review_input.object_id != request.object_id
            or any(abs(getattr(review_input.dimensions, axis).value - expected) > 1e-12
                   for axis, expected in zip(("x_m", "y_m", "z_m"), request.dimensions_m, strict=True))):
        raise AssetAuthoringError("authoring_packaging_physics_identity_mismatch")
    properties = review.accepted.properties
    for name in ("mass_kg", "static_friction", "dynamic_friction", "restitution"):
        value = getattr(properties, name)
        lower, upper = physics_bounds[name]
        # USD receives one mass value. The uncertainty interval describes the
        # unknown photographed object's possible mass; it is not a set of
        # masses the simulator will silently sample. Keep that interval in the
        # completion receipt, including any portion outside the admitted
        # simulation value range. Contact parameters retain their stricter
        # full-interval admission because they affect the policy interaction.
        inside = (lower <= value.value <= upper if name == "mass_kg" and value.basis == "estimated"
                  else lower <= value.interval.lower <= value.value <= value.interval.upper <= upper)
        if not inside:
            raise AssetAuthoringError("authoring_packaging_estimate_outside_admitted_bounds:" + name)
    measurement = json.loads(_verified(authoring_result["geometry_readback"]).read_text())
    validate_geometry_readback(request, measurement, review_input.appearance)
    source_path = _verified(authoring_result["asset"])
    source = Usd.Stage.Open(str(source_path))
    if source is None or source.GetDefaultPrim().GetPath() != Sdf.Path(ASSET_ROOT):
        raise AssetAuthoringError("authoring_packaging_visual_root_invalid")
    if abs(UsdGeom.GetStageMetersPerUnit(source) - 1.0) > 1e-12 or UsdGeom.GetStageUpAxis(source) != "Z":
        raise AssetAuthoringError("authoring_packaging_visual_frame_invalid")
    for prim in source.Traverse():
        if (any(schema.startswith(("Physics", "Physx")) for schema in prim.GetAppliedSchemas())
                or prim.IsA(UsdPhysics.Joint)
                or any(prop.GetName().startswith(("physics:", "physx")) for prop in prim.GetProperties())):
            raise AssetAuthoringError("authoring_packaging_unreviewed_physics_in_visual")
    mesh, mesh_receipt, geometry_sources = _final_visual_mesh(
        request=request, authoring_result=authoring_result, source=source, allow_compound_solid=True)
    consistency = _final_mass_consistency(review, mesh)
    mass_kg = float(properties.mass_kg.value)
    tensor = np.asarray(mesh.moment_inertia, dtype=float) * (mass_kg / mesh.mass)
    principal, rotation = np.linalg.eigh(tensor)
    if np.any(principal <= 0) or not np.isfinite(principal).all():
        raise AssetAuthoringError("authoring_packaging_inertia_invalid")
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return {"mesh": mesh, "source_path": source_path, "review_path": review_path, "mesh_receipt": mesh_receipt,
            "geometry_sources": geometry_sources, "consistency": consistency, "mass_kg": mass_kg,
            "mass_interval_kg": [float(properties.mass_kg.interval.lower), float(properties.mass_kg.interval.upper)],
            "mass_basis": properties.mass_kg.basis,
            "center_of_mass_m": [float(v) for v in mesh.center_mass], "principal_inertia": [float(v) for v in principal],
            "principal_rotation": rotation, "static_friction": float(properties.static_friction.value),
            "dynamic_friction": float(properties.dynamic_friction.value), "restitution": float(properties.restitution.value),
            "collision_bounds_part_frame_m": {"minimum": [float(v) for v in mesh.bounds[0]],
                                              "maximum": [float(v) for v in mesh.bounds[1]]},
            "cad_readback": authoring_result["cad"].get("readback")}


def _quat_from_matrix(rotation):
    from pxr import Gf
    return Gf.Matrix4d(Gf.Matrix3d(*rotation.T.reshape(-1).tolist()), Gf.Vec3d(0)).ExtractRotationQuat()


def package_astra_articulated_candidate(*, requests: Mapping[str, AuthoringRequest],
                                        authoring_results: Mapping[str, Mapping[str, Any]],
                                        plan: Mapping[str, Any], output_root: Path,
                                        physics_bounds: Mapping[str, Mapping[str, Sequence[float]]]) -> dict[str, Any]:
    """Compose the reviewed parts into one articulated USDZ and seal its candidate physics.

    ``physics_bounds`` maps part id to the admitted bounds for that part's
    reviewed properties. The root body (carcass or appliance body) is the
    dynamic root; the runtime grounds it with its anchor joint at staging (no
    link is kinematic, which PhysX refuses). Before any USD is written, every
    contract required part must be carried by a planned link and every planned
    body cavity must be hollow and open-fronted in the reviewed mesh.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise AssetAuthoringError("articulated_plan_schema_invalid")
    if set(requests) != set(plan["parts"]) or set(authoring_results) != set(plan["parts"]):
        raise AssetAuthoringError("articulated_part_set_mismatch")
    required_by_link = required_parts_by_link(plan)
    link_part = {row["link_id"]: row["part_id"] for row in plan["links"]}
    if set(required_by_link) - set(link_part):
        raise AssetAuthoringError("articulated_required_part_unplanned:" + sorted(
            part for link in set(required_by_link) - set(link_part) for part in required_by_link[link])[0])
    root_id = root_link_id(plan)
    legacy = legacy_drawer_plan(plan)  # origin/main's drawer plan never planned or checked a cavity
    cavities = plan.get("interior_cavities") or []
    if (not cavities and not legacy) or any(row.get("link_id") not in link_part for row in cavities):
        raise AssetAuthoringError("articulated_body_interior_cavity_unplanned")
    parts = {part_id: _part_physics(request=requests[part_id], authoring_result=authoring_results[part_id],
                                    physics_bounds=physics_bounds[part_id]) for part_id in plan["parts"]}
    for part_id, spec in plan["parts"].items():
        if [round(v, 5) for v in requests[part_id].dimensions_m] != [round(v, 5) for v in spec["dimensions_m"]]:
            raise AssetAuthoringError("articulated_part_dimensions_changed:" + part_id)
    for link_id in sorted({row["link_id"] for row in cavities}):
        mesh = parts[link_part[link_id]]["mesh"]
        findings = interior_cavity_findings(mesh.vertices, mesh.faces,
                                            [row for row in cavities if row["link_id"] == link_id])
        if findings:
            raise AssetAuthoringError("articulated_" + findings[0])
    walls: dict[str, list[dict[str, Any]]] = {}
    cavity_collision = plan.get("cavity_collision_approximation")
    if cavity_collision not in {None, CAVITY_COLLISION_WALL_BOXES}:
        raise AssetAuthoringError("articulated_body_cavity_collision_unsupported")
    for link_id in sorted({row["link_id"] for row in cavities}) if cavity_collision else []:
        rows = [row for row in cavities if row["link_id"] == link_id]
        walls[link_id] = cavity_wall_boxes(parts[link_part[link_id]]["collision_bounds_part_frame_m"], rows)
        findings = collision_cavity_findings(
            [box_collision_piece(box["center_m"], box["size_m"]) for box in walls[link_id]], rows)
        if findings:
            raise AssetAuthoringError("articulated_" + findings[0])
    task_joint = plan["task_joint"]
    limits = task_joint_limits(task_joint)
    output_root.mkdir(parents=True, exist_ok=True)
    authored = output_root / "astra_articulated_candidate.usdc"
    if authored.exists():
        raise AssetAuthoringError("authoring_packaging_output_exists")
    stage = Usd.Stage.CreateNew(str(authored))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, ASSET_ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("blueprint:articulatedAssemblyPlanDigest", canonical_digest(dict(plan)))
    link_paths: dict[str, str] = {}
    link_rows: dict[str, dict[str, Any]] = {}
    materials: dict[str, Any] = {}
    for link in plan["links"]:
        link_id, part_id = link["link_id"], link["part_id"]
        physics = parts[part_id]
        path = f"{ASSET_ROOT}/links/{link_id}"
        link_paths[link_id] = path
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in link["rest_translation_m"]]))
        body = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(False)
        mass = UsdPhysics.MassAPI.Apply(xform.GetPrim())
        mass.CreateMassAttr(physics["mass_kg"])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*physics["center_of_mass_m"]))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*physics["principal_inertia"]))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(_quat_from_matrix(physics["principal_rotation"])))
        xform.GetPrim().SetCustomDataByKey("blueprint:semanticRole", link["semantic_role"])
        xform.GetPrim().SetCustomDataByKey("blueprint:partId", part_id)
        xform.GetPrim().SetCustomDataByKey(REQUIRED_PARTS_ATTRIBUTE,
                                           json.dumps(required_by_link.get(link_id, {}), sort_keys=True))
        is_task_link = link_id == task_joint["child_link_id"]
        provenance = OBSERVED_PROVENANCE if is_task_link else GENERATED_PROVENANCE
        # Visual appearance: the reviewed part candidate, referenced then flattened.
        visual = stage.DefinePrim(f"{path}/visual", "Xform")
        visual.GetReferences().AddReference(str(physics["source_path"]), ASSET_ROOT)
        visual.SetCustomDataByKey(PROVENANCE_ATTRIBUTE, provenance)
        if part_id not in materials:
            material = UsdShade.Material.Define(stage, f"{ASSET_ROOT}/Looks/ReviewedPhysics_{part_id}")
            contact = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            contact.CreateStaticFrictionAttr(physics["static_friction"])
            contact.CreateDynamicFrictionAttr(physics["dynamic_friction"])
            contact.CreateRestitutionAttr(physics["restitution"])
            materials[part_id] = material
        mesh = physics["mesh"]
        collision_paths = []
        for box in walls.get(link_id, []):
            wall = UsdGeom.Cube.Define(stage, f"{path}/collision/{box['name']}")
            wall.CreateSizeAttr(1.0)
            wall.AddTranslateOp().Set(Gf.Vec3d(*box["center_m"]))
            wall.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*box["size_m"]))
            collision_paths.append(str(wall.GetPath()))
        if not collision_paths:
            shape = UsdGeom.Mesh.Define(stage, f"{path}/collision/FinalVisualShape")
            shape.CreatePointsAttr([Gf.Vec3f(*[float(c) for c in v]) for v in mesh.vertices.tolist()])
            shape.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
            shape.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
            shape.CreateSubdivisionSchemeAttr("none")
            UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("convexDecomposition")
            collision_paths.append(str(shape.GetPath()))
        for collider_path in collision_paths:
            collision = UsdGeom.Gprim(stage.GetPrimAtPath(collider_path))
            collision.CreatePurposeAttr("guide")
            collision.CreateVisibilityAttr("invisible")
            UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdShade.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(
                materials[part_id], UsdShade.Tokens.weakerThanDescendants, "physics")
            collision.GetPrim().SetCustomDataByKey(PROVENANCE_ATTRIBUTE, GENERATED_PROVENANCE)
            collision.GetPrim().SetCustomDataByKey("blueprint:collisionGeometryOnly", True)
        handle_grasp_point = None
        if is_task_link:
            handle = plan["parts"][part_id]["handle"]
            section, length = float(handle["section_m"]), float(handle["length_m"])
            bar = UsdGeom.Cube.Define(stage, f"{path}/collision/handle")
            bar.CreateSizeAttr(1.0)
            bar.AddTranslateOp().Set(Gf.Vec3f(*[float(v) for v in handle["center_m"]]))
            bar.AddScaleOp().Set(Gf.Vec3f(section, length, section) if handle.get("axis", "Y") == "Y"
                                 else Gf.Vec3f(section, section, length))
            bar.CreatePurposeAttr("guide")
            bar.CreateVisibilityAttr("invisible")
            UsdPhysics.CollisionAPI.Apply(bar.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdShade.MaterialBindingAPI.Apply(bar.GetPrim()).Bind(materials[part_id], UsdShade.Tokens.weakerThanDescendants, "physics")
            bar.GetPrim().SetCustomDataByKey(PROVENANCE_ATTRIBUTE, OBSERVED_PROVENANCE)
            bar.GetPrim().SetCustomDataByKey(TASK_CONTACT_ROLE_ATTRIBUTE, HANDLE_ROLE)
            bar.GetPrim().SetCustomDataByKey("blueprint:collisionGeometryOnly", True)
            collision_paths.append(str(bar.GetPath()))
            handle_grasp_point = [float(v) for v in handle["grasp_point_link_m"]]
        link_rows[link_id] = {"link_id": link_id, "part_id": part_id, "prim_path": path, "semantic_role": link["semantic_role"],
                              "rest_translation_m": [float(v) for v in link["rest_translation_m"]],
                              "mass_kg": physics["mass_kg"], "mass_basis": physics["mass_basis"],
                              "mass_uncertainty_interval_kg": physics["mass_interval_kg"],
                              "mass_interval_exceeds_admitted_simulation_bounds": (
                                  physics["mass_interval_kg"][0] < physics_bounds[part_id]["mass_kg"][0]
                                  or physics["mass_interval_kg"][1] > physics_bounds[part_id]["mass_kg"][1]),
                              "center_of_mass_m": physics["center_of_mass_m"],
                              "diagonal_inertia_kg_m2": physics["principal_inertia"],
                              "collision_bounds_link_frame_m": physics["collision_bounds_part_frame_m"],
                              "collision_prim_paths": collision_paths,
                              "physics_material": {"static_friction": physics["static_friction"],
                                                   "dynamic_friction": physics["dynamic_friction"],
                                                   "restitution": physics["restitution"]},
                              **({"handle_grasp_point_link_m": handle_grasp_point} if handle_grasp_point else {})}
    joint_rows = []
    revolute = task_joint["joint_type"] == "revolute"
    root_rest = [float(v) for v in next(row["rest_translation_m"] for row in plan["links"] if row["link_id"] == root_id)]
    for link in plan["links"]:
        if link["is_root"]:
            continue
        is_task = link["link_id"] == task_joint["child_link_id"]
        joint_id = task_joint["joint_id"] if is_task else f"{link['link_id']}_fixed"
        path = f"{ASSET_ROOT}/joints/{joint_id}"
        rest = [float(v) for v in link["rest_translation_m"]]
        # Joint frames sit at the hinge anchor (a drawer's is its own origin),
        # expressed in each body's frame; closed, both frames coincide.
        anchor = [float(v) for v in task_joint.get("anchor_asset_frame_m", rest)] if is_task else rest
        rotation = Gf.Quatf(*[float(v) for v in (task_joint.get("joint_frame_rotation_wxyz") if is_task else None)
                              or [1.0, 0.0, 0.0, 0.0]])
        if is_task and revolute:
            joint = UsdPhysics.RevoluteJoint.Define(stage, path)
            joint.CreateAxisAttr(str(task_joint["usd_axis"]))
            joint.CreateLowerLimitAttr(math.degrees(limits[0]))
            joint.CreateUpperLimitAttr(math.degrees(limits[1]))
        elif is_task:
            joint = UsdPhysics.PrismaticJoint.Define(stage, path)
            joint.CreateAxisAttr("X")
            joint.CreateLowerLimitAttr(limits[0])
            joint.CreateUpperLimitAttr(limits[1])
        else:
            joint = UsdPhysics.FixedJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(link_paths[root_id])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(link_paths[link["link_id"]])])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*[anchor[i] - root_rest[i] for i in range(3)]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*[anchor[i] - rest[i] for i in range(3)]))
        joint.CreateLocalRot0Attr(rotation)
        joint.CreateLocalRot1Attr(rotation)
        prim = joint.GetPrim()
        prim.SetCustomDataByKey("blueprint:jointRole", "target" if is_task else "locked")
        prim.SetCustomDataByKey("blueprint:resetPosition", 0.0)
        drive_row: dict[str, Any] = {"declared_drive_type": "none", "usd_drive_authored": False, "implementation": "none"}
        if is_task:
            prim.SetCustomDataByKey("blueprint:graphAxis", Gf.Vec3d(*task_joint["axis_asset_frame"]))
            prim.SetCustomDataByKey("blueprint:declaredDriveType", "none")
            passive_friction = task_joint.get("passive_friction")
            if passive_friction is not None:
                axis = "angular" if revolute else "linear"
                prim.AddAppliedSchema(f"PhysxJointAxisAPI:{axis}")
                prim.CreateAttribute(f"physxJointAxis:{axis}:staticFrictionEffort", Sdf.ValueTypeNames.Float,
                                     custom=False).Set(
                    float(passive_friction["static_effort"]))
                prim.CreateAttribute(f"physxJointAxis:{axis}:dynamicFrictionEffort", Sdf.ValueTypeNames.Float,
                                     custom=False).Set(
                    float(passive_friction["dynamic_effort"]))
                prim.SetCustomDataByKey("blueprint:passiveFrictionBasis", passive_friction["basis"])
            damping = float(task_joint["drive"]["damping"])
            if damping > 0.0:
                # USD angular drive damping is per degree; the plan's is per radian.
                name, implementation = (("angular", "passive_torque_damper") if revolute
                                        else ("linear", "passive_force_damper"))
                drive = UsdPhysics.DriveAPI.Apply(prim, name)
                drive.CreateTypeAttr().Set("force")
                drive.CreateStiffnessAttr().Set(0.0)
                drive.CreateDampingAttr().Set(math.radians(damping) if revolute else damping)
                drive.CreateTargetPositionAttr().Set(0.0)
                prim.SetCustomDataByKey("blueprint:driveImplementation", implementation)
                drive_row = {"declared_drive_type": "none", "usd_drive_authored": True, "usd_drive_type": "force",
                             "implementation": implementation, "stiffness": 0.0, "damping": damping}
        joint_rows.append({"joint_id": joint_id, "prim_path": path,
                           "joint_type": task_joint["joint_type"] if is_task else "fixed",
                           "parent_link_id": root_id, "child_link_id": link["link_id"],
                           "role": "target" if is_task else "locked",
                           "limits": limits if is_task else [0.0, 0.0],
                           "reset_position": 0.0, "drive": drive_row})
    # The joints and limits constrain the mechanism; approximate colliders of
    # nested parts must not fight them. Robot contacts are unaffected.
    for link_id, path in link_paths.items():
        if link_id == root_id:
            continue
        filtered = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(link_paths[root_id]))
        filtered.CreateFilteredPairsRel().AddTarget(Sdf.Path(path))
        for other_id, other_path in link_paths.items():
            if other_id not in {root_id, link_id} and other_id > link_id:
                UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path)).CreateFilteredPairsRel().AddTarget(Sdf.Path(other_path))
    stage.GetRootLayer().documentation = ("Blueprint articulated SimReady candidate composed from reviewed parts; "
                                          "native behaviour and physical equivalence are not qualified by authoring")
    stage.GetRootLayer().Save()
    flattened = output_root / "astra_articulated_candidate.flat.usdc"
    Usd.Stage.Open(str(authored)).Flatten().Export(str(flattened))
    asset = output_root / "astra_articulated_replacement_candidate.usdz"
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(flattened)), str(asset)):
        raise AssetAuthoringError("authoring_usdz_packaging_failed")
    reopened = Usd.Stage.Open(str(asset))
    if reopened is None or not reopened.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI):
        raise AssetAuthoringError("authoring_usdz_readback_failed")
    moving = [p for p in reopened.Traverse() if p.IsA(UsdPhysics.Joint) and not p.IsA(UsdPhysics.FixedJoint)]
    expected_type = UsdPhysics.RevoluteJoint if revolute else UsdPhysics.PrismaticJoint
    if (len(moving) != 1 or not moving[0].IsA(expected_type)
            or len([p for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]) != len(link_rows)):
        raise AssetAuthoringError("authoring_usdz_readback_failed")
    observed_parts = {prim.GetName(): json.loads(str(prim.GetCustomDataByKey(REQUIRED_PARTS_ATTRIBUTE) or "{}"))
                      for prim in reopened.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    missing = required_part_findings(plan, observed_parts)
    if missing:
        raise AssetAuthoringError("articulated_" + missing[0])
    task_link = link_rows[task_joint["child_link_id"]]
    lows = [min(row["collision_bounds_link_frame_m"]["minimum"][i] + row["rest_translation_m"][i] for row in link_rows.values()) for i in range(3)]
    highs = [max(row["collision_bounds_link_frame_m"]["maximum"][i] + row["rest_translation_m"][i] for row in link_rows.values()) for i in range(3)]
    completion = {
        "schema_version": COMPLETION_SCHEMA_VERSION, "status": "bounded_candidate_completed",
        "asset_kind": "articulated_assembly", "candidate_prior_only": True, "physical_truth_claimed": False,
        "physics_bounds": {part_id: {k: [float(a), float(b)] for k, (a, b) in bounds.items()} for part_id, bounds in physics_bounds.items()},
        "links": [link_rows[link["link_id"]] for link in plan["links"]],
        "joints": joint_rows,
        "task_joint_prim_path": f"{ASSET_ROOT}/joints/{task_joint['joint_id']}",
        "task_link_prim_path": task_link["prim_path"],
        "fixed_base_body_prim_path": link_paths[root_id],
        "handle_prim_paths": [p for p in task_link["collision_prim_paths"] if p.endswith("/handle")],
        "handle_grasp_point_link_m": task_link["handle_grasp_point_link_m"],
        "required_parts_by_link": required_by_link,
        "interior_cavity_check": ({"status": "not_planned_legacy_drawer_configuration"} if legacy else
                                  {"status": "hollow_open_front_verified_on_reviewed_mesh",
                                   "cavity_ids": [row["cavity_id"] for row in cavities],
                                   "minimum_clear_fraction": CAVITY_MINIMUM_CLEAR_FRACTION}),
        "collision_bounds_asset_frame_closed_m": {"minimum": lows, "maximum": highs},
        "collision_dimensions_m": [highs[i] - lows[i] for i in range(3)],
        "intra_assembly_collision_filtered": True,
        "center_of_mass_authority": "constant_density_final_visual_candidate_per_link",
        "inertia_authority": "constant_density_final_visual_candidate_scaled_to_reviewed_mass_per_link",
        "collision_approximation": ("analytic_wall_boxes_hollow_body_convexDecomposition_other_links_plus_tagged_handle_box"
                                    if walls else "convexDecomposition_per_link_plus_tagged_handle_box"),
        **({"cavity_collision": {
            "approximation": CAVITY_COLLISION_WALL_BOXES,
            "status": "cavity_preserved_on_declared_collision_geometry",
            "collider_prim_paths": {link_id: [f"{link_paths[link_id]}/collision/{box['name']}" for box in rows]
                                    for link_id, rows in walls.items()},
            "reason": ("convexHull fills a hollow, convexDecomposition is unknown before cooking, and a "
                       "triangle mesh cannot collide on a dynamic root link"),
        }} if walls else {}),
        "native_collision_cooking_qualified": False,
        "part_reviews": {part_id: {"astra_review": file_record(parts[part_id]["review_path"]),
                                   "final_visual_solid_volume_m3": float(parts[part_id]["mesh"].volume),
                                   "final_visual_mesh_sources": parts[part_id]["geometry_sources"],
                                   "final_visual_mesh_receipt_digest": parts[part_id]["mesh_receipt"]["receipt_digest"],
                                   "mass_model_final_geometry_consistency": parts[part_id]["consistency"],
                                   "original_cad_readback": parts[part_id]["cad_readback"]} for part_id in plan["parts"]},
        "illumination_authority": "site_scene_only",
    }
    completion["completion_digest"] = canonical_digest(completion, digest_field="completion_digest")
    receipt = {"asset": file_record(asset), "physics_completion": completion, "plan": dict(plan),
               "request_digests": {part_id: requests[part_id].request_digest for part_id in plan["parts"]},
               "authoring_result_digests": {part_id: authoring_results[part_id].get("result_digest") for part_id in plan["parts"]},
               "asset_origin": "captured_assembly_reconstruction", "claim_ceiling": "development_only", "native_qualified": False}
    save_json(output_root / "astra_articulated_packaging_receipt.json", receipt)
    return receipt


def articulation_graph_from_plan(plan: Mapping[str, Any], *, opening_fraction: float = 0.6) -> dict[str, Any]:
    """The task-neutral graph contract for the composed assembly (adp_articulation_graph.v1)."""
    task_joint = plan["task_joint"]
    stroke = task_joint_limits(task_joint)[1]
    revolute = task_joint["joint_type"] == "revolute"
    links = [{"link_id": row["link_id"], "is_root": row["is_root"], "semantic_role": row["semantic_role"]} for row in plan["links"]]
    joints = []
    for row in plan["links"]:
        if row["is_root"]:
            continue
        is_task = row["link_id"] == task_joint["child_link_id"]
        joints.append({"joint_id": task_joint["joint_id"] if is_task else f"{row['link_id']}_fixed",
                       "parent_link_id": root_link_id(plan), "child_link_id": row["link_id"],
                       "joint_type": task_joint["joint_type"] if is_task else "fixed",
                       "role": "target" if is_task else "locked",
                       "axis": [float(v) for v in task_joint["axis_asset_frame"]] if is_task else [0.0, 0.0, 0.0],
                       "limits": [0.0, stroke] if is_task else [0.0, 0.0], "reset_position": 0.0,
                       "reset_tolerance": HINGE_RESET_TOLERANCE_RAD if revolute else 0.005,
                       "drive": {"drive_type": "none", "stiffness": 0.0,
                                 "damping": float(task_joint["drive"]["damping"]) if is_task else 0.0, "maximum_force": 0.0}})
    link_ids = [row["link_id"] for row in links]
    pairs = [{"link_a": a, "link_b": b, "collision_enabled": False}
             for i, a in enumerate(sorted(link_ids)) for b in sorted(link_ids)[i + 1:]]
    return {"schema_version": "adp_articulation_graph.v1", "links": links, "joints": joints, "collision_pairs": pairs,
            "success_predicate": {"combination": "all",
                                  "joint_intervals": {task_joint["joint_id"]: [round(opening_fraction * stroke, 5), stroke]}}}


__all__ = [
    "AUTHORING_RESULT_SCHEMA_VERSION", "CAVITY_COLLISION_WALL_BOXES", "COMPLETION_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "PROVENANCE_ATTRIBUTE", "REQUIRED_PARTS_ATTRIBUTE", "TASK_CONTACT_ROLE_ATTRIBUTE", "TARGET_JOINT_ID",
    "articulation_graph_from_plan", "assembly_contract", "assembly_family", "box_collision_piece",
    "cavity_wall_boxes", "collision_cavity_findings", "interior_cavity_findings", "legacy_drawer_plan",
    "package_astra_articulated_candidate", "plan_articulated_assembly", "planned_cavity_findings",
    "required_part_findings",
    "required_parts_by_link", "root_link_id", "task_joint_limits",
]
