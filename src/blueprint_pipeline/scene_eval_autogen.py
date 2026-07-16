"""Auto-generate robot-eval tasks and scenario variations from a single scene file.

Captures frequently arrive as ONE bare geometry file — a 3DGS/point PLY or a USD
stage — with no capture bundle, no object index, and no task anchors. The rest of
the eval lane (`robot_eval_dataset`, `scenario_variation_instantiator`,
`episode_spec`) assumes those sidecars exist. This module closes that gap: given
just the scene file, it analyzes the geometry and any recoverable semantics, then
deterministically synthesizes a full review-input eval scope:

1. **Ingest** the single file on CPU with no network:
   PLY via :func:`scene_asset_preflight.inspect_ply_asset` (plus a stdlib
   binary-vertex sampler for standard 3DGS vertex PLYs, which the inspector
   only header-scans), USD via :func:`scene_asset_preflight.inspect_usd_asset`
   (pxr-backed when available, string inspection otherwise). When pxr is
   importable, per-prim AABBs come from
   :class:`scene_placement.usd_index.UsdSceneSpatialIndex`.
2. **Classify** the environment (kitchen / warehouse / office / ...) from the
   recovered object labels and the filename — deterministic keyword scoring,
   no model in the loop.
3. **Derive zones** (center / quadrant staging points) from the scene bounds so
   every task gets finite, in-bounds start/goal poses in the scene frame.
4. **Synthesize at least :data:`MIN_TASK_COUNT` tasks**: a geometry-grounded
   core set that exists for ANY scene (navigate, inspect, transfer, blocked-path
   recovery, human-crossing response) plus object-grounded tasks (open/close,
   pick-and-place, fixture inspection) for every usable semantic hint. Each task
   is keyed to the shared task ontology via
   :func:`robot_eval_dataset._canonical_task_id_for_task`.
5. **Synthesize many scenarios per task**: a baseline plus every canonical
   scenario variation (:data:`robot_eval_dataset.SCENARIO_VARIATION_DEFINITIONS`)
   at ``seeds_per_variation`` deterministic difficulty seeds, with concrete
   scene-grounded parameters (obstacles at the actual route midpoint, distractors
   near the actual target pose, ...).
6. **Emit** task cards, scenario cards, per-task eval cards, a
   ``scenario_family_library.json`` shaped for
   :func:`scenario_variation_instantiator.build_scenario_variation_instances`,
   and a top-level manifest — all deterministic, fail-closed, and carrying the
   repo's claim boundary (generated artifacts are review inputs, never execution
   or readiness proof).

Truth boundary: everything here is heuristic geometry plus name-token semantics
recovered from one file. Tasks/scenarios are *candidate eval scope for review*;
they claim no navigability, physics, rights clearance, or robot execution.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .industrial_ontology import classify_industrial_entity
from .robot_eval_dataset import (
    DEFAULT_TASK_THRESHOLD_TEMPLATES,
    SCENARIO_FAMILY_LIBRARY_SCHEMA_VERSION,
    SCENARIO_VARIATION_DEFINITIONS,
    SCORING_METRIC_DEFINITIONS,
    TASK_ONTOLOGY_DEFINITIONS,
    _canonical_task_id_for_task,
)
from .scene_asset_preflight import (
    _parse_ply_header,
    _percentile,
    _ply_header,
    inspect_ply_asset,
    inspect_usd_asset,
)
from .scene_placement.target_resolver import (
    _OPENABLE_TARGET_GROUPS,
    _canonical_group_for_token,
)

SCENE_EVAL_AUTOGEN_SCHEMA_VERSION = "scene_eval_autogen.v1"
SCENE_ANALYSIS_SCHEMA_VERSION = "scene_eval_autogen_scene_analysis.v1"
AUTO_TASK_CARDS_SCHEMA_VERSION = "scene_eval_autogen_task_cards.v1"
AUTO_SCENARIO_CARDS_SCHEMA_VERSION = "scene_eval_autogen_scenario_cards.v1"
AUTO_EVAL_CARDS_SCHEMA_VERSION = "scene_eval_autogen_eval_cards.v1"

MIN_TASK_COUNT = 5
DEFAULT_MAX_TASK_COUNT = 12
DEFAULT_SEEDS_PER_VARIATION = 3
DEFAULT_ATTEMPTS_PER_SCENARIO = 3

SUPPORTED_SCENE_SUFFIXES = {".ply", ".usd", ".usda", ".usdc", ".usdz"}

DIFFICULTY_TIERS = ("nominal", "moderate", "hard")

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "auto_generated_eval_scope_from_single_scene_file_for_review",
    "single_file_scene_input": True,
    "tasks_and_scenarios_auto_generated": True,
    "generated_artifacts_are_review_inputs": True,
    "scene_semantics_source": "geometry_and_name_token_heuristics_only",
    "simulator_execution_proven": False,
    "robot_policy_execution_proven": False,
    "navigability_proven": False,
    "physics_contact_validated": False,
    "rights_and_privacy_cleared": False,
    "rank_fidelity_result_proven": False,
    "non_ranking_operational_claim_validated": False,
    "public_claim_upgrade_allowed": False,
    "disallowed_claims": [
        "robot_ready",
        "deployment_ready",
        "simulator_execution_completed",
        "task_acceptance_validated",
        "guaranteed_success_rate",
    ],
}

# Environment keyword banks for deterministic classification. Scores are simple
# hit counts over recovered labels + filename tokens; ties resolve in this order.
_ENVIRONMENT_KEYWORDS: Sequence[tuple[str, frozenset]] = (
    ("kitchen", frozenset({
        "kitchen", "sink", "faucet", "stove", "oven", "fridge", "refrigerator",
        "microwave", "dishwasher", "kettle", "counter", "countertop", "pantry",
        "cooktop", "range",
    })),
    ("warehouse", frozenset({
        "warehouse", "rack", "pallet", "forklift", "tote", "conveyor", "aisle",
        "dock", "carton", "crate", "stockroom",
    })),
    ("manufacturing", frozenset({
        "factory", "assembly", "cnc", "lathe", "workbench", "machine", "fixture",
        "line", "station", "cell", "press",
    })),
    ("office", frozenset({
        "office", "desk", "monitor", "keyboard", "whiteboard", "cubicle",
        "printer", "conference",
    })),
    ("bathroom", frozenset({
        "bathroom", "toilet", "shower", "bathtub", "basin", "washbasin", "vanity",
    })),
    ("bedroom", frozenset({
        "bedroom", "bed", "nightstand", "wardrobe", "dresser", "mattress",
    })),
    ("retail", frozenset({
        "store", "retail", "checkout", "register", "display", "mannequin",
    })),
)
_DEFAULT_ENVIRONMENT = "generic_indoor"

# Prim/label tokens that name scene structure rather than actionable objects.
_STRUCTURAL_TOKENS = frozenset({
    "root", "world", "scene", "stage", "geom", "geometry", "mesh", "meshes",
    "material", "materials", "looks", "shader", "shaders", "xform", "prototypes",
    "instance", "instances", "env", "environment", "render", "rendering",
    "light", "lights", "dome", "camera", "cameras", "physics", "collision",
    "collider", "colliders", "ground", "floor", "floors", "wall", "walls",
    "ceiling", "ceilings", "sky", "skybox", "grid", "default", "prim", "group",
    "layer", "payload", "ref", "proto",
})

# Labels that read as hand-manipulable payloads regardless of localization.
_PICKABLE_LABELS = frozenset({
    "kettle", "cup", "mug", "bowl", "pot", "pan", "plate", "bottle", "vase",
    "book", "box", "carton", "tote", "package", "part", "tool", "can", "jar",
    "lamp", "basket", "tray", "item",
})

# Labels that read as inspectable fixtures / storage the robot surveys.
_FIXTURE_LABELS = frozenset({
    "shelf", "rack", "cabinet", "counter", "table", "desk", "machine",
    "conveyor", "workbench", "station", "pallet", "locker", "bench", "stand",
    "wardrobe", "dresser", "sofa", "couch", "bed",
})

# Objects bigger than this on every axis are structure, not a graspable payload.
_PICKUP_MAX_EXTENT_M = 0.6


# ----------------------------- small helpers -----------------------------

def _slug(value: Any, *, fallback: str = "item") -> str:
    text = str(value or "").strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    collapsed = "_".join(part for part in "".join(chars).split("_") if part)
    return collapsed or fallback


def _hash01(*parts: Any) -> float:
    """Deterministic pseudo-random float in [0, 1) from the given parts."""
    digest = sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def _sha_payload(payload: Any) -> str:
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _finite_triplet(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        floats = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in floats):
        return None
    return floats


def _ontology_entry(ontology_task_id: str) -> Dict[str, Any]:
    for item in TASK_ONTOLOGY_DEFINITIONS:
        if item.get("task_id") == ontology_task_id:
            return dict(item)
    return {}


def _threshold_profile_for_family(task_family: str) -> Dict[str, Any]:
    if task_family in {"manipulation", "pick_place"}:
        return dict(DEFAULT_TASK_THRESHOLD_TEMPLATES["pick_place"])
    if task_family in {"navigation", "articulation_navigation", "recovery", "safety_response"}:
        return dict(DEFAULT_TASK_THRESHOLD_TEMPLATES["navigation"])
    return dict(DEFAULT_TASK_THRESHOLD_TEMPLATES["general"])


# ----------------------------- ingestion -----------------------------

def _sample_binary_vertex_ply(path: Path, *, max_samples: int = 100_000) -> Optional[Dict[str, Any]]:
    """Bounds/centroid/floor from a standard binary vertex PLY via stdlib struct.

    ``inspect_ply_asset`` decodes ascii vertices and compressed chunk bounds but
    only header-scans standard binary vertex PLYs — exactly the format bare 3DGS
    exports use. Stride-sample the vertex records so huge splats stay cheap.
    """
    try:
        lines, header_end = _ply_header(path)
    except (OSError, ValueError):
        return None
    parsed = _parse_ply_header(lines)
    fmt = str(parsed.get("format") or "")
    if fmt not in {"binary_little_endian", "binary_big_endian"}:
        return None
    elements = [dict(item) for item in parsed.get("elements", []) if isinstance(item, Mapping)]
    vertex = next((item for item in elements if item.get("name") == "vertex"), None)
    if vertex is None or elements[0].get("name") != "vertex":
        return None
    properties = [dict(item) for item in vertex.get("properties", []) if isinstance(item, Mapping)]
    scalar_sizes = {"char": 1, "uchar": 1, "int8": 1, "uint8": 1,
                    "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
                    "int": 4, "uint": 4, "int32": 4, "uint32": 4, "float": 4, "float32": 4,
                    "double": 8, "float64": 8}
    offset = 0
    xyz_spec: Dict[str, tuple] = {}
    for prop in properties:
        if prop.get("kind") != "scalar":
            return None
        type_name = str(prop.get("type"))
        size = scalar_sizes.get(type_name, 0)
        if size <= 0:
            return None
        name = str(prop.get("name"))
        if name in {"x", "y", "z"}:
            if type_name in {"float", "float32"}:
                code = "f"
            elif type_name in {"double", "float64"}:
                code = "d"
            else:
                return None
            xyz_spec[name] = (offset, code)
        offset += size
    record_size = offset
    if len(xyz_spec) != 3 or record_size <= 0:
        return None
    endian = "<" if fmt == "binary_little_endian" else ">"
    count = int(vertex.get("count") or 0)
    if count <= 0:
        return None
    stride = max(1, count // max_samples)
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    totals = [0.0, 0.0, 0.0]
    z_values: List[float] = []
    sampled = 0
    with path.open("rb") as handle:
        for index in range(0, count, stride):
            handle.seek(header_end + index * record_size)
            raw = handle.read(record_size)
            if len(raw) != record_size:
                break
            point = []
            for axis in ("x", "y", "z"):
                axis_offset, code = xyz_spec[axis]
                point.append(float(struct.unpack_from(endian + code, raw, axis_offset)[0]))
            if not all(math.isfinite(value) for value in point):
                continue
            for axis_index, value in enumerate(point):
                mins[axis_index] = min(mins[axis_index], value)
                maxs[axis_index] = max(maxs[axis_index], value)
                totals[axis_index] += value
            z_values.append(point[2])
            sampled += 1
    if sampled == 0:
        return None
    return {
        "bounds": {"min": mins, "max": maxs},
        "centroid": [total / sampled for total in totals],
        "floor_z_estimate": _percentile(z_values, 0.02),
        "sampled_point_count": sampled,
        "estimate_method": "binary_vertex_xyz_stride_sample",
        "confidence": "medium",
    }


def _labels_from_semantic_hints(hints: Sequence[Mapping[str, Any]]) -> List[str]:
    """Prim-name hints -> canonical object labels, structure filtered out."""
    labels: List[str] = []
    seen: set = set()
    for hint in hints:
        raw = str((hint or {}).get("label") or "").strip()
        if not raw:
            continue
        tokens = [token for token in _slug(raw).split("_") if token and not token.isdigit()]
        if not tokens or all(token in _STRUCTURAL_TOKENS for token in tokens):
            continue
        label = None
        # Direct domain-label hits (pickable/fixture/openable nouns) beat synonym-group
        # canonicalization, so "tote_bin" stays a pickable tote instead of collapsing
        # into the trash group via "bin".
        for token in tokens:
            if token in _PICKABLE_LABELS or token in _FIXTURE_LABELS or token in _OPENABLE_TARGET_GROUPS:
                label = token
                break
        if label is None:
            for token in tokens:
                group = _canonical_group_for_token(token)
                if group:
                    label = group
                    break
        if label is None:
            candidates = [token for token in tokens if token not in _STRUCTURAL_TOKENS]
            if not candidates:
                continue
            noun = candidates[-1]
            if noun in _PICKABLE_LABELS or noun in _FIXTURE_LABELS or noun in _OPENABLE_TARGET_GROUPS:
                label = noun
            elif len(candidates) == len(tokens):
                # A fully non-structural name is still a usable object hint.
                label = noun
            else:
                continue
        if len(label) < 3:
            continue
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _usd_scene_objects(path: Path) -> List[Dict[str, Any]]:
    """Per-prim world AABBs when pxr is importable; empty list otherwise."""
    try:
        from .scene_placement.usd_index import UsdSceneSpatialIndex

        index = UsdSceneSpatialIndex(usd_path=str(path))
        objects = index.objects()
    except Exception:  # noqa: BLE001 - optional backend; ingest stays single-file safe
        return []
    out: List[Dict[str, Any]] = []
    for obj in objects:
        out.append(
            {
                "object_id": obj.id,
                "label": obj.label,
                "bbox_min": list(obj.bbox_min),
                "bbox_max": list(obj.bbox_max),
                "centroid": list(obj.centroid),
                "source": "usd_stage_prim_aabb",
            }
        )
    return out


def ingest_scene_file(scene_path: Path) -> Dict[str, Any]:
    """Normalize a single PLY/USD file into one scene model dict.

    Fail-closed: unreadable/unsupported inputs return ``status: blocked`` with
    blockers instead of raising. Bounds may be ``None`` (header-only inspection);
    downstream stages degrade rather than fabricate poses.
    """
    suffix = scene_path.suffix.lower()
    scene: Dict[str, Any] = {
        "status": "completed",
        "scene_path": str(scene_path.resolve()),
        "scene_file_name": scene_path.name,
        "asset_type": "ply" if suffix == ".ply" else "usd",
        "up_axis": "Z",
        "bounds": None,
        "centroid": None,
        "floor": None,
        "object_labels": [],
        "objects": [],
        "blockers": [],
        "limitations": [],
    }
    if not scene_path.is_file():
        scene.update(status="blocked", blockers=["scene_file_missing"])
        return scene
    if suffix not in SUPPORTED_SCENE_SUFFIXES:
        scene.update(status="blocked", blockers=[f"unsupported_scene_suffix:{suffix or 'none'}"])
        return scene

    try:
        return _ingest_supported_scene_file(scene_path, suffix, scene)
    except Exception as exc:  # noqa: BLE001 - fail closed on corrupt/malformed inputs
        scene.update(
            status="blocked",
            blockers=[f"scene_file_unreadable_or_malformed:{type(exc).__name__}"],
        )
        scene["limitations"].append(str(exc)[:500])
        return scene


def _ingest_supported_scene_file(
    scene_path: Path, suffix: str, scene: Dict[str, Any]
) -> Dict[str, Any]:
    if suffix == ".ply":
        inspection = inspect_ply_asset(scene_path)
        estimate: Dict[str, Any] = {
            "bounds": inspection.get("bounds"),
            "centroid": inspection.get("centroid"),
            "floor_z_estimate": inspection.get("floor_z_estimate"),
            "estimate_method": inspection.get("estimate_method"),
        }
        if not estimate.get("bounds"):
            sampled = _sample_binary_vertex_ply(scene_path)
            if sampled:
                estimate = sampled
        scene["inspection"] = {
            key: inspection.get(key)
            for key in ("asset_type", "format", "vertex_count", "estimate_method", "confidence")
        }
        scene["estimate_method"] = estimate.get("estimate_method")
    else:
        inspection = inspect_usd_asset(scene_path)
        estimate = {
            "bounds": inspection.get("bounds"),
            "centroid": inspection.get("centroid"),
            "floor_z_estimate": inspection.get("floor_z_estimate"),
            "estimate_method": inspection.get("estimate_method"),
        }
        scene["up_axis"] = str(inspection.get("up_axis") or "Z").upper() or "Z"
        scene["object_labels"] = _labels_from_semantic_hints(inspection.get("semantic_hints") or [])
        scene["objects"] = _usd_scene_objects(scene_path)
        if scene["objects"] and not scene["object_labels"]:
            scene["object_labels"] = _labels_from_semantic_hints(
                [{"label": obj["label"]} for obj in scene["objects"]]
            )
        scene["inspection"] = {
            key: inspection.get(key)
            for key in ("asset_type", "status", "estimate_method", "confidence", "prim_counts")
        }
        scene["estimate_method"] = estimate.get("estimate_method")

    bounds = estimate.get("bounds") if isinstance(estimate.get("bounds"), Mapping) else None
    low = _finite_triplet(bounds.get("min")) if bounds else None
    high = _finite_triplet(bounds.get("max")) if bounds else None
    if low and high:
        scene["bounds"] = {"min": low, "max": high}
        centroid = _finite_triplet(estimate.get("centroid"))
        scene["centroid"] = centroid or [(low[i] + high[i]) * 0.5 for i in range(3)]
        up_index = 1 if scene["up_axis"] == "Y" else 2
        floor = estimate.get("floor_z_estimate") if up_index == 2 else None
        scene["floor"] = float(floor) if isinstance(floor, (int, float)) and math.isfinite(float(floor)) else low[up_index]
    else:
        scene["limitations"].append(
            "scene_bounds_unavailable_header_only_inspection_zone_poses_degraded"
        )
    return scene


# ----------------------------- environment classification -----------------------------

def classify_environment(scene: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic keyword-scored environment guess from labels + filename."""
    tokens: List[str] = []
    for label in scene.get("object_labels") or []:
        tokens.extend(_slug(label).split("_"))
    tokens.extend(_slug(Path(str(scene.get("scene_file_name") or "")).stem).split("_"))
    token_set = {token for token in tokens if token}
    scores = {
        environment: len(token_set & keywords)
        for environment, keywords in _ENVIRONMENT_KEYWORDS
    }
    best = max(scores.values() or [0])
    environment = _DEFAULT_ENVIRONMENT
    if best > 0:
        environment = next(
            env for env, keywords in _ENVIRONMENT_KEYWORDS if scores[env] == best
        )
    return {
        "environment": environment,
        "confidence": "keyword_match" if best > 0 else "default_no_semantic_signal",
        "scores": {env: score for env, score in scores.items() if score > 0},
        "matched_tokens": sorted(token_set & set().union(*(kw for _, kw in _ENVIRONMENT_KEYWORDS))),
        "claim_boundary": "keyword_heuristic_environment_guess_not_verified_site_type",
    }


# ----------------------------- hazard grounding (R060) -----------------------------

# Base scenario variations that model a physical site hazard. When the industrial
# hazard ontology grounds a real hazard-relevant entity in the scene, these
# variations carry that grounding so hazards inform the eval scenarios regardless
# of whether the optional qualification trust layer ever runs.
_HAZARD_SCENARIO_VARIATIONS = frozenset({"blocked_path", "human_crossing", "forklift_nearby"})


def derive_hazard_grounding(scene: Mapping[str, Any]) -> Dict[str, Any]:
    """Classify recovered scene labels through the industrial hazard ontology.

    R060: the structured hazard classification in ``industrial_ontology`` was only
    consumed by the OPTIONAL qualification trust layer, so hazards (forklift lanes,
    shared traffic, barriers, human-interaction zones, thresholds, floor hazards)
    never reached the always-on scenario grounding lane. This surfaces the SAME
    ``classify_industrial_entity`` output here so hazard-relevant entities inform
    eval scenarios even when qualification is not run. Heuristic name-token
    grounding only — not a verified site hazard assessment.
    """
    labels = [str(label) for label in (scene.get("object_labels") or [])]
    filename_tokens = _slug(
        Path(str(scene.get("scene_file_name") or "")).stem
    ).split("_")
    hazard_labels: Dict[str, List[str]] = {}
    for value in [*labels, *filename_tokens]:
        text = str(value or "").strip()
        if not text:
            continue
        entity = classify_industrial_entity(text)
        if not entity.hazard_relevant:
            continue
        bucket = hazard_labels.setdefault(entity.entity_type, [])
        if text not in bucket:
            bucket.append(text)
    hazard_entity_types = sorted(hazard_labels)
    return {
        "schema_version": "scene_eval_autogen_hazard_grounding.v1",
        "ontology_source": "industrial_ontology.classify_industrial_entity",
        "hazard_entity_types": hazard_entity_types,
        "hazard_relevant_labels": {key: hazard_labels[key] for key in hazard_entity_types},
        "any_grounded_hazard": bool(hazard_entity_types),
        "claim_boundary": "name_token_hazard_grounding_not_verified_site_hazard_assessment",
    }


# ----------------------------- zones -----------------------------

def derive_scene_zones(scene: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Named staging zones from the scene bounds: center + inset quadrant midpoints.

    Poses are finite floor-level points in the scene coordinate frame, up-axis
    aware. Without bounds, zones still exist (so tasks stay well-formed) but carry
    ``pose_xyz: None`` and a blocked validation status.
    """
    bounds = scene.get("bounds")
    up_axis = str(scene.get("up_axis") or "Z").upper()
    up = 1 if up_axis == "Y" else 2
    h0, h1 = [axis for axis in range(3) if axis != up]
    zone_specs = (
        ("zone_center", 0.5, 0.5, "central staging zone"),
        ("zone_west", 0.25, 0.5, "west staging zone"),
        ("zone_east", 0.75, 0.5, "east goal station"),
        ("zone_south", 0.5, 0.25, "south approach zone"),
        ("zone_north", 0.5, 0.75, "north handoff zone"),
    )
    zones: List[Dict[str, Any]] = []
    for zone_id, f0, f1, label in zone_specs:
        pose: Optional[List[float]] = None
        if isinstance(bounds, Mapping):
            low = _finite_triplet(bounds.get("min"))
            high = _finite_triplet(bounds.get("max"))
            if low and high:
                floor = scene.get("floor")
                floor_value = float(floor) if isinstance(floor, (int, float)) else low[up]
                pose = [0.0, 0.0, 0.0]
                pose[h0] = low[h0] + f0 * (high[h0] - low[h0])
                pose[h1] = low[h1] + f1 * (high[h1] - low[h1])
                pose[up] = floor_value
        zones.append(
            {
                "zone_id": zone_id,
                "label": label,
                "pose_xyz": pose,
                "coordinate_frame": "scene_file_frame",
                "validation_status": "validated_in_bounds_floor_point"
                if pose is not None
                else "blocked_missing_scene_bounds",
                "claim_boundary": "geometric_staging_point_not_navigability_proof",
            }
        )
    return zones


def _zone_by_id(zones: Sequence[Mapping[str, Any]], zone_id: str) -> Dict[str, Any]:
    for zone in zones:
        if zone.get("zone_id") == zone_id:
            return dict(zone)
    return {"zone_id": zone_id, "pose_xyz": None, "validation_status": "blocked_unknown_zone"}


# ----------------------------- task synthesis -----------------------------

def _object_pose(scene: Mapping[str, Any], label: str) -> Optional[List[float]]:
    for obj in scene.get("objects") or []:
        if _slug(obj.get("label")) == _slug(label):
            return _finite_triplet(obj.get("centroid"))
    return None


def _object_is_pickable_size(scene: Mapping[str, Any], label: str) -> Optional[bool]:
    """True/False when the label is localized with an AABB, None when unknown."""
    for obj in scene.get("objects") or []:
        if _slug(obj.get("label")) != _slug(label):
            continue
        low = _finite_triplet(obj.get("bbox_min"))
        high = _finite_triplet(obj.get("bbox_max"))
        if not low or not high:
            return None
        return max(high[i] - low[i] for i in range(3)) <= _PICKUP_MAX_EXTENT_M
    return None


def _task_card(
    *,
    task_id: str,
    task_text: str,
    task_category: str,
    grounding_source: str,
    start_zone: Mapping[str, Any],
    goal_zone: Mapping[str, Any],
    target_object_label: Optional[str] = None,
    target_object_pose: Optional[List[float]] = None,
) -> Dict[str, Any]:
    ontology_task_id = _canonical_task_id_for_task(
        task_id=task_id, task_text=task_text, task_category=task_category
    )
    ontology = _ontology_entry(ontology_task_id)
    task_family = str(ontology.get("task_family") or "general")
    pair_valid = bool(start_zone.get("pose_xyz")) and bool(
        goal_zone.get("pose_xyz") or target_object_pose
    )
    return {
        "task_id": task_id,
        "task_text": task_text,
        "task_category": task_category,
        "task_family": task_family,
        "ontology_task_id": ontology_task_id,
        "ontology_version": "1.0",
        "grounding_source": grounding_source,
        "target_object_label": target_object_label,
        "target_object_pose": target_object_pose,
        "start_zone_id": start_zone.get("zone_id"),
        "goal_zone_id": goal_zone.get("zone_id"),
        "start_zone": start_zone.get("pose_xyz"),
        "goal_zone": target_object_pose or goal_zone.get("pose_xyz"),
        "zone_pair_status": "validated_zone_pair" if pair_valid else "blocked_missing_zone_poses",
        "success_criteria": list(ontology.get("success_criteria") or []),
        "threshold_profile": _threshold_profile_for_family(task_family),
        "required_evidence": [
            "robot_pov_or_recorded_trace",
            "action_log_or_teleop_demo",
            "actual_outcome_manifest",
        ],
        "claim_boundary": "auto_generated_task_definition_only_no_robot_execution_claim",
    }


def synthesize_tasks(
    scene: Mapping[str, Any],
    environment: Mapping[str, Any],
    zones: Sequence[Mapping[str, Any]],
    *,
    min_tasks: int = MIN_TASK_COUNT,
    max_tasks: int = DEFAULT_MAX_TASK_COUNT,
) -> List[Dict[str, Any]]:
    """At least ``min_tasks`` task cards for ANY scene: geometry core + object tasks."""
    center = _zone_by_id(zones, "zone_center")
    west = _zone_by_id(zones, "zone_west")
    east = _zone_by_id(zones, "zone_east")
    south = _zone_by_id(zones, "zone_south")
    north = _zone_by_id(zones, "zone_north")
    env_name = str(environment.get("environment") or _DEFAULT_ENVIRONMENT).replace("_", " ")

    tasks: List[Dict[str, Any]] = [
        _task_card(
            task_id="navigate_west_to_east_station",
            task_text=f"Navigate from the west staging zone to the east goal station across the {env_name} scene",
            task_category="navigation",
            grounding_source="scene_geometry",
            start_zone=west,
            goal_zone=east,
        ),
        _task_card(
            task_id="inspect_scene_fixtures_sweep",
            task_text=f"Inspect the shelf and fixture zones of the {env_name} scene and record the required viewpoints",
            task_category="inspection",
            grounding_source="scene_geometry",
            start_zone=center,
            goal_zone=north,
        ),
        _task_card(
            task_id="move_payload_center_to_north",
            task_text="Move the staged tote from the central staging zone to the north handoff zone",
            task_category="material_handling",
            grounding_source="scene_geometry",
            start_zone=center,
            goal_zone=north,
        ),
        _task_card(
            task_id="blocked_route_recovery_west_east",
            task_text="Recover safely when the primary west-to-east route is blocked by an obstacle mid-crossing",
            task_category="recovery",
            grounding_source="scene_geometry",
            start_zone=west,
            goal_zone=east,
        ),
        _task_card(
            task_id="human_crossing_response_center",
            task_text="Yield safely when a human crosses the central zone while the robot transits south to north",
            task_category="safety_response",
            grounding_source="scene_geometry",
            start_zone=south,
            goal_zone=north,
        ),
    ]

    # Object-grounded tasks from recovered semantics, strongest affordance first.
    seen_ids = {task["task_id"] for task in tasks}
    labels = [str(label) for label in (scene.get("object_labels") or [])]
    openable = [label for label in labels if label in _OPENABLE_TARGET_GROUPS]
    pickable = [
        label
        for label in labels
        if label not in _OPENABLE_TARGET_GROUPS
        and (
            label in _PICKABLE_LABELS
            if _object_is_pickable_size(scene, label) is None
            else _object_is_pickable_size(scene, label)
        )
    ]
    fixtures = [
        label
        for label in labels
        if label in _FIXTURE_LABELS and label not in openable and label not in pickable
    ]
    object_task_specs = [
        *(
            (f"open_close_{_slug(label)}", f"Open and close the {label}", "open_close", label)
            for label in openable
        ),
        *(
            (
                f"pick_place_{_slug(label)}",
                f"Pick up the {label} and place it in the drop zone",
                "pick_place",
                label,
            )
            for label in pickable
        ),
        *(
            (
                f"inspect_{_slug(label)}",
                f"Inspect the {label} and record the required viewpoints",
                "inspection",
                label,
            )
            for label in fixtures
        ),
    ]
    for task_id, task_text, category, label in object_task_specs:
        if len(tasks) >= max_tasks:
            break
        if task_id in seen_ids:
            continue
        seen_ids.add(task_id)
        tasks.append(
            _task_card(
                task_id=task_id,
                task_text=task_text,
                task_category=category,
                grounding_source="scene_object_hint",
                start_zone=center,
                goal_zone=center,
                target_object_label=label,
                target_object_pose=_object_pose(scene, label),
            )
        )

    if len(tasks) < min_tasks:  # defensive: the core set already covers min_tasks
        for index in range(min_tasks - len(tasks)):
            tasks.append(
                _task_card(
                    task_id=f"navigate_supplemental_route_{index + 1}",
                    task_text=f"Navigate supplemental route {index + 1} between staging zones",
                    task_category="navigation",
                    grounding_source="environment_default",
                    start_zone=south,
                    goal_zone=north,
                )
            )
    return tasks


# ----------------------------- scenario synthesis -----------------------------

def _route_midpoint(task: Mapping[str, Any]) -> Optional[List[float]]:
    start = _finite_triplet(task.get("start_zone"))
    goal = _finite_triplet(task.get("goal_zone"))
    if not start or not goal:
        return None
    return [(start[i] + goal[i]) * 0.5 for i in range(3)]


def _scenario_parameters(
    variation_id: str,
    task: Mapping[str, Any],
    *,
    tier: str,
    unit: float,
) -> Dict[str, Any]:
    """Concrete scene-grounded parameters for one scenario instance.

    ``unit`` is the deterministic per-instance draw in [0, 1); ``tier`` scales
    severity so seeds sweep nominal -> hard rather than resampling one setting.
    """
    severity = {"nominal": 0.35, "moderate": 0.65, "hard": 1.0}.get(tier, 0.5)
    midpoint = _route_midpoint(task)
    target = _finite_triplet(task.get("goal_zone"))
    if variation_id == "lighting_variation":
        return {
            "lighting": {
                "ambient_lux_delta": round(-80.0 - (220.0 * severity) - 40.0 * unit, 1),
                "key_light_intensity_scale": round(1.0 - 0.6 * severity, 3),
                "color_temperature_kelvin": int(3400 + 1400 * unit),
            }
        }
    if variation_id == "object_rotation":
        return {
            "object_pose_delta": {
                "target_selector": task.get("target_object_label")
                or "task_target_object_or_nearest_pick_object",
                "yaw_degrees": round(10.0 + 80.0 * severity * unit + 10.0 * severity, 1),
            }
        }
    if variation_id == "cart_shifted":
        return {
            "cart_pose_delta": {
                "target_selector": "cart_or_mobile_shelf_near_task",
                "translation_m": {
                    "x": round(0.15 + 0.4 * severity * unit, 3),
                    "y": round(-0.3 * severity + 0.2 * unit, 3),
                    "z": 0.0,
                },
                "yaw_degrees": round(15.0 * severity, 1),
            }
        }
    if variation_id == "blocked_path":
        return {
            "path_obstacle": {
                "obstacle_type": "rolling_cart",
                "placement_point": midpoint,
                "placement": "route_midpoint" if midpoint else "approach_midpoint_unlocalized",
                "width_m": round(0.4 + 0.5 * severity, 2),
                "clearance_policy": "must_replan_without_contact",
            }
        }
    if variation_id == "human_crossing":
        return {
            "dynamic_actor": {
                "actor_type": "human",
                "crossing_point": midpoint,
                "speed_mps": round(0.8 + 0.8 * severity, 2),
                "trigger": "robot_within_2m_of_crossing",
            }
        }
    if variation_id == "forklift_nearby":
        return {
            "forklift_actor": {
                "actor_type": "forklift",
                "anchor_point": midpoint or target,
                "standoff_m": round(3.0 - 1.5 * severity, 2),
                "motion_profile": "stationary_or_slow_roll",
            }
        }
    if variation_id == "occlusion":
        return {
            "occluder": {
                "object_type": "carton_stack",
                "target": task.get("target_object_label") or "label_or_grasp_point",
                "coverage_fraction": round(0.2 + 0.55 * severity, 2),
                "pose_hint": "between_robot_camera_and_target",
            }
        }
    if variation_id == "glare":
        return {
            "glare_source": {
                "light_type": "specular_strip",
                "incidence_angle_degrees": round(10.0 + 30.0 * unit, 1),
                "intensity_lux": int(400 + 900 * severity),
                "material_target": "floor_or_label",
            }
        }
    if variation_id == "missing_label":
        return {
            "label_visibility": {
                "target_label": task.get("target_object_label") or "task_target_label",
                "visible": False,
                "replacement_state": "blank_or_torn_label_surface",
            }
        }
    if variation_id == "wrong_object_nearby":
        return {
            "distractor_object": {
                "object_role": "lookalike_wrong_object",
                "anchor_point": target,
                "distance_to_target_m": round(0.6 - 0.4 * severity, 2),
                "label_similarity": "high" if severity > 0.5 else "medium",
            }
        }
    if variation_id == "narrow_approach_angle":
        return {
            "approach_constraint": {
                "allowed_width_m": round(0.9 - 0.4 * severity, 2),
                "approach_yaw_degrees": round(30.0 + 45.0 * severity * unit, 1),
                "requires_precise_base_alignment": severity > 0.5,
            }
        }
    return {"review_mutation": {"variation_id": variation_id, "requires_owner_mapping": True}}


def synthesize_scenarios(
    tasks: Sequence[Mapping[str, Any]],
    *,
    seeds_per_variation: int = DEFAULT_SEEDS_PER_VARIATION,
    hazard_grounding: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Baseline + every canonical variation x difficulty seeds, per task.

    When ``hazard_grounding`` (from :func:`derive_hazard_grounding`) surfaces
    ontology-classified site hazards, hazard-family scenarios (blocked path, human
    crossing, forklift nearby) carry that grounding so the eval scope reflects real
    site hazards independently of the optional qualification layer (R060).
    """
    grounded_hazard_types = (
        list(hazard_grounding.get("hazard_entity_types") or [])
        if isinstance(hazard_grounding, Mapping)
        else []
    )
    scenarios: List[Dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("task_id"))
        scenarios.append(
            {
                "scenario_id": f"scn_{task_id}_baseline",
                "task_id": task_id,
                "variation_id": "baseline",
                "label": "Baseline (as-captured scene)",
                "seed": 0,
                "difficulty_tier": "nominal",
                "parameters": {},
                "scenario_status": "generated_baseline",
                "claim_boundary": "auto_generated_scenario_is_engine_input_not_sim_or_robot_result",
            }
        )
        for definition in SCENARIO_VARIATION_DEFINITIONS:
            variation_id = str(definition.get("variation_id"))
            for seed in range(1, max(1, seeds_per_variation) + 1):
                tier = DIFFICULTY_TIERS[(seed - 1) % len(DIFFICULTY_TIERS)]
                unit = _hash01(task_id, variation_id, seed)
                scenario: Dict[str, Any] = {
                    "scenario_id": f"scn_{task_id}_{variation_id}_s{seed}",
                    "task_id": task_id,
                    "variation_id": variation_id,
                    "label": str(definition.get("label") or variation_id),
                    "seed": seed,
                    "difficulty_tier": tier,
                    "parameters": _scenario_parameters(
                        variation_id, task, tier=tier, unit=unit
                    ),
                    "scenario_status": str(
                        definition.get("default_status") or "agent-inferred-needs-review"
                    ),
                    "claim_boundary": "auto_generated_scenario_is_engine_input_not_sim_or_robot_result",
                }
                if variation_id in _HAZARD_SCENARIO_VARIATIONS and grounded_hazard_types:
                    scenario["grounded_site_hazards"] = list(grounded_hazard_types)
                    scenario["hazard_grounding_source"] = (
                        "industrial_ontology.classify_industrial_entity"
                    )
                scenarios.append(scenario)
    return scenarios


# ----------------------------- eval cards + family library -----------------------------

def _eval_cards(
    tasks: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
    *,
    attempts_per_scenario: int = DEFAULT_ATTEMPTS_PER_SCENARIO,
) -> List[Dict[str, Any]]:
    by_task: Dict[str, List[str]] = {}
    for scenario in scenarios:
        by_task.setdefault(str(scenario.get("task_id")), []).append(
            str(scenario.get("scenario_id"))
        )
    cards: List[Dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("task_id"))
        scenario_ids = by_task.get(task_id, [])
        cards.append(
            {
                "eval_id": f"eval_{task_id}",
                "task_id": task_id,
                "scenario_ids": scenario_ids,
                "scenario_count": len(scenario_ids),
                "attempts_per_scenario": attempts_per_scenario,
                "metrics": [str(item.get("metric_id")) for item in SCORING_METRIC_DEFINITIONS],
                "threshold_profile": dict(task.get("threshold_profile") or {}),
                "claim_boundary": "auto_generated_eval_scope_definition_not_an_executed_eval",
            }
        )
    return cards


def _scenario_family_library(
    tasks: Sequence[Mapping[str, Any]],
    *,
    generated_at: str,
) -> Dict[str, Any]:
    """Family library shaped for ``build_scenario_variation_instances``."""
    families: List[Dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("task_id"))
        families.append(
            {
                "family_id": f"family_{task_id}",
                "scenario_id": f"scn_{task_id}",
                "task_id": task_id,
                "robot_profile_id": None,
                "variations": [
                    {
                        "variation_id": str(definition.get("variation_id")),
                        "label": str(definition.get("label") or definition.get("variation_id")),
                        "scenario_status": str(
                            definition.get("default_status") or "agent-inferred-needs-review"
                        ),
                    }
                    for definition in SCENARIO_VARIATION_DEFINITIONS
                ],
                "source": "scene_eval_autogen",
            }
        )
    return {
        "schema_version": SCENARIO_FAMILY_LIBRARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "family_count": len(families),
        "families": families,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


# ----------------------------- orchestration -----------------------------

def generate_scene_eval_tasks(
    scene_path: str | Path,
    output_dir: str | Path,
    *,
    site_id: Optional[str] = None,
    environment_override: Optional[str] = None,
    min_tasks: int = MIN_TASK_COUNT,
    max_tasks: int = DEFAULT_MAX_TASK_COUNT,
    seeds_per_variation: int = DEFAULT_SEEDS_PER_VARIATION,
    attempts_per_scenario: int = DEFAULT_ATTEMPTS_PER_SCENARIO,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Single scene file -> auto-generated task/scenario/eval review artifacts.

    Returns the top-level manifest (also written to
    ``<output_dir>/scene_eval_autogen_manifest.json``). ``status`` is
    ``completed`` when the scene ingested and the minimum task count was met;
    ``blocked`` (with ``blockers``) when the input is missing or unsupported.
    """
    scene_file = Path(scene_path)
    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    resolved_generated_at = generated_at or utc_now_iso()
    resolved_site_id = site_id or f"site_{_slug(scene_file.stem, fallback='scene')}"

    manifest: Dict[str, Any] = {
        "schema_version": SCENE_EVAL_AUTOGEN_SCHEMA_VERSION,
        "generated_at": resolved_generated_at,
        "site_id": resolved_site_id,
        "scene_file": str(scene_file),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }

    scene = ingest_scene_file(scene_file)
    if scene.get("status") != "completed":
        manifest.update(status="blocked", blockers=list(scene.get("blockers") or []))
        write_json(out_dir / "scene_eval_autogen_manifest.json", manifest)
        return manifest

    environment = classify_environment(scene)
    if environment_override:
        environment = {
            "environment": _slug(environment_override),
            "confidence": "caller_override",
            "scores": {},
            "matched_tokens": [],
            "claim_boundary": "caller_supplied_environment_override",
        }
    zones = derive_scene_zones(scene)
    hazard_grounding = derive_hazard_grounding(scene)
    tasks = synthesize_tasks(
        scene, environment, zones, min_tasks=min_tasks, max_tasks=max_tasks
    )
    scenarios = synthesize_scenarios(
        tasks,
        seeds_per_variation=seeds_per_variation,
        hazard_grounding=hazard_grounding,
    )
    evals = _eval_cards(tasks, scenarios, attempts_per_scenario=attempts_per_scenario)

    scene_analysis = {
        "schema_version": SCENE_ANALYSIS_SCHEMA_VERSION,
        "generated_at": resolved_generated_at,
        "site_id": resolved_site_id,
        "scene": scene,
        "environment": environment,
        "zones": zones,
        "hazard_grounding": hazard_grounding,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    task_cards = {
        "schema_version": AUTO_TASK_CARDS_SCHEMA_VERSION,
        "generated_at": resolved_generated_at,
        "site_id": resolved_site_id,
        "card_count": len(tasks),
        "cards": tasks,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    scenario_cards = {
        "schema_version": AUTO_SCENARIO_CARDS_SCHEMA_VERSION,
        "generated_at": resolved_generated_at,
        "site_id": resolved_site_id,
        "card_count": len(scenarios),
        "cards": scenarios,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    eval_cards = {
        "schema_version": AUTO_EVAL_CARDS_SCHEMA_VERSION,
        "generated_at": resolved_generated_at,
        "site_id": resolved_site_id,
        "card_count": len(evals),
        "cards": evals,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    family_library = _scenario_family_library(tasks, generated_at=resolved_generated_at)

    write_json(out_dir / "scene_analysis.json", scene_analysis)
    write_json(out_dir / "auto_task_cards.json", task_cards)
    write_json(out_dir / "auto_scenario_cards.json", scenario_cards)
    write_json(out_dir / "auto_eval_cards.json", eval_cards)
    write_json(out_dir / "scenario_family_library.json", family_library)

    scenarios_per_task = {
        str(task.get("task_id")): sum(
            1 for scenario in scenarios if scenario.get("task_id") == task.get("task_id")
        )
        for task in tasks
    }
    manifest.update(
        {
            "status": "completed",
            "environment": environment.get("environment"),
            "task_count": len(tasks),
            "scenario_count": len(scenarios),
            "eval_count": len(evals),
            "scenarios_per_task": scenarios_per_task,
            "min_task_count_required": min_tasks,
            "meets_minimum_task_count": len(tasks) >= min_tasks,
            "min_scenarios_per_task": min(scenarios_per_task.values()) if scenarios_per_task else 0,
            "geometry_grounding": "bounds_recovered"
            if scene.get("bounds")
            else "bounds_missing_header_only",
            "grounded_site_hazards": list(hazard_grounding.get("hazard_entity_types") or []),
            "any_grounded_hazard": bool(hazard_grounding.get("any_grounded_hazard")),
            "limitations": list(scene.get("limitations") or []),
            "artifacts": {
                "scene_analysis": "scene_analysis.json",
                "task_cards": "auto_task_cards.json",
                "scenario_cards": "auto_scenario_cards.json",
                "eval_cards": "auto_eval_cards.json",
                "scenario_family_library": "scenario_family_library.json",
            },
            "deterministic_fingerprint": _sha_payload(
                {
                    "site_id": resolved_site_id,
                    "environment": environment.get("environment"),
                    "task_ids": [task["task_id"] for task in tasks],
                    "scenario_ids": [scenario["scenario_id"] for scenario in scenarios],
                    "parameters": [scenario["parameters"] for scenario in scenarios],
                }
            ),
        }
    )
    write_json(out_dir / "scene_eval_autogen_manifest.json", manifest)
    return manifest


# ----------------------------- CLI -----------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-generate 5+ eval tasks and per-task scenario variations from a "
            "single PLY or USD scene file (no other capture inputs required)."
        )
    )
    parser.add_argument("scene", help="Path to the scene file (.ply, .usd, .usda, .usdc, .usdz)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory (default: <scene stem>_eval_autogen next to the scene file)",
    )
    parser.add_argument("--site-id", default=None, help="Explicit site id (default: derived from the file name)")
    parser.add_argument(
        "--environment",
        default=None,
        help="Override the environment guess (e.g. kitchen, warehouse)",
    )
    parser.add_argument("--min-tasks", type=int, default=MIN_TASK_COUNT)
    parser.add_argument("--max-tasks", type=int, default=DEFAULT_MAX_TASK_COUNT)
    parser.add_argument("--seeds-per-variation", type=int, default=DEFAULT_SEEDS_PER_VARIATION)
    args = parser.parse_args(argv)

    scene_file = Path(args.scene)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else scene_file.parent / f"{scene_file.stem}_eval_autogen"
    )
    manifest = generate_scene_eval_tasks(
        scene_file,
        output_dir,
        site_id=args.site_id,
        environment_override=args.environment,
        min_tasks=args.min_tasks,
        max_tasks=args.max_tasks,
        seeds_per_variation=args.seeds_per_variation,
    )
    print(
        json.dumps(
            {
                "status": manifest.get("status"),
                "output_dir": str(output_dir),
                "environment": manifest.get("environment"),
                "task_count": manifest.get("task_count"),
                "scenario_count": manifest.get("scenario_count"),
                "blockers": manifest.get("blockers", []),
            },
            indent=2,
        )
    )
    return 0 if manifest.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
