"""Concrete scenario-family variation instantiation for simulator adapters."""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context
from .robot_eval_dataset import SCENARIO_VARIATION_DEFINITIONS


SCENARIO_VARIATION_INSTANCES_SCHEMA_VERSION = "scenario_variation_instances.v1"
SCENARIO_VARIATION_ENGINE_MUTATION_SCHEMA_VERSION = "scenario_variation_engine_mutation.v1"
SCENARIO_VARIATION_ENGINE_TARGETS = (
    "isaac_sim",
    "isaac_lab_arena",
    "mujoco",
    "pybullet",
    "newton",
)
SCENARIO_VARIATION_NAMES = tuple(
    str(definition["variation_id"]) for definition in SCENARIO_VARIATION_DEFINITIONS
)

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "scenario_variation_instantiation_for_engine_adapter_inputs",
    "scenario_variations_generated": True,
    "generated_variations_are_review_inputs": True,
    "engine_adapter_payloads_written": True,
    "simulator_execution_proven": False,
    "robot_policy_execution_proven": False,
    "robot_readiness_proven": False,
    "physics_contact_validated": False,
    "safety_validated": False,
    "public_claim_upgrade_allowed": False,
}


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _cards_from_payload(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cards = payload.get("cards")
    if isinstance(cards, list):
        return [dict(item) for item in cards if isinstance(item, Mapping)]
    if isinstance(payload, list):  # pragma: no cover - kept for legacy callers that bypass typing.
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return []


def _string(value: Any) -> str:
    return str(value or "").strip()


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = _string(value).lower()
    chars = [char if char.isalnum() else "_" for char in text]
    collapsed = "_".join(part for part in "".join(chars).split("_") if part)
    return collapsed or fallback


def _sha_payload(payload: Mapping[str, Any]) -> str:
    encoded = repr(sorted(payload.items())).encode("utf-8")
    return sha256(encoded).hexdigest()


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _canonical_variation_name(value: Any) -> str:
    text = _safe_id(value)
    aliases = {
        "lighting": "lighting_variation",
        "lighting_variation": "lighting_variation",
        "object_rotation": "object_rotation",
        "object_rotated": "object_rotation",
        "cart_shift": "cart_shifted",
        "cart_shifted": "cart_shifted",
        "blocked_path": "blocked_path",
        "human_crossing": "human_crossing",
        "forklift_nearby": "forklift_nearby",
        "occlusion": "occlusion",
        "glare": "glare",
        "missing_label": "missing_label",
        "wrong_object": "wrong_object_nearby",
        "wrong_object_nearby": "wrong_object_nearby",
        "narrow_approach": "narrow_approach_angle",
        "narrow_approach_angle": "narrow_approach_angle",
    }
    return aliases.get(text, text)


def _normal_variations(variations: Any) -> List[Dict[str, Any]]:
    if not isinstance(variations, list):
        return []
    out: List[Dict[str, Any]] = []
    for index, variation in enumerate(variations, start=1):
        if not isinstance(variation, Mapping):
            continue
        variation_id = _canonical_variation_name(
            variation.get("variation_id")
            or variation.get("variation_name")
            or variation.get("label")
            or f"variation_{index}"
        )
        if variation_id not in SCENARIO_VARIATION_NAMES:
            continue
        out.append(
            {
                "variation_id": variation_id,
                "variation_name": variation_id,
                "label": _string(variation.get("label") or variation.get("variation_name"))
                or variation_id.replace("_", " ").title(),
                "scenario_status": _string(variation.get("scenario_status"))
                or _string(variation.get("status"))
                or "review_required",
                "source_variation": dict(variation),
            }
        )
    return out


def _scenario_family_rows(
    *,
    pipeline_dir: Path,
    generated_at: str,
) -> List[Dict[str, Any]]:
    family_library = _read_optional_mapping(
        pipeline_dir / "robot_eval_dataset" / "scenario_family_library.json"
    )
    families = family_library.get("families")
    if isinstance(families, list) and families:
        rows: List[Dict[str, Any]] = []
        for family_index, family in enumerate(families, start=1):
            if not isinstance(family, Mapping):
                continue
            family_id = _string(family.get("family_id")) or f"family_{family_index}"
            scenario_id = _string(family.get("scenario_id")) or f"scenario_{family_index}"
            task_id = _string(family.get("task_id")) or None
            normal = _normal_variations(family.get("variations"))
            seen = {item["variation_name"] for item in normal}
            for required in SCENARIO_VARIATION_NAMES:
                if required not in seen:
                    normal.append(
                        {
                            "variation_id": required,
                            "variation_name": required,
                            "label": required.replace("_", " ").title(),
                            "scenario_status": "generated_missing_from_family_library",
                            "source_variation": {
                                "variation_id": required,
                                "generated_at": generated_at,
                            },
                        }
                    )
            rows.append(
                {
                    "family_id": family_id,
                    "scenario_id": scenario_id,
                    "task_id": task_id,
                    "robot_profile_id": _string(family.get("robot_profile_id")) or None,
                    "variations": normal,
                    "source": "robot_eval_dataset/scenario_family_library.json",
                }
            )
        return rows

    scenario_cards = _cards_from_payload(
        _read_optional_mapping(pipeline_dir / "robot_eval_dataset" / "scenario_cards.json")
    )
    rows = []
    for index, scenario in enumerate(scenario_cards, start=1):
        scenario_id = _string(scenario.get("scenario_id") or scenario.get("id")) or f"scenario_{index}"
        task_id = _string(scenario.get("task_id")) or None
        rows.append(
            {
                "family_id": f"family_{_safe_id(scenario_id, fallback=str(index))}",
                "scenario_id": scenario_id,
                "task_id": task_id,
                "robot_profile_id": _string(scenario.get("robot_profile_id")) or None,
                "variations": [
                    {
                        "variation_id": name,
                        "variation_name": name,
                        "label": name.replace("_", " ").title(),
                        "scenario_status": "generated_from_scenario_card",
                        "source_variation": {"variation_id": name},
                    }
                    for name in SCENARIO_VARIATION_NAMES
                ],
                "source": "robot_eval_dataset/scenario_cards.json",
            }
        )
    return rows


def _concrete_mutation(variation_name: str, *, ordinal: int) -> Dict[str, Any]:
    seed = ordinal + 1
    if variation_name == "lighting_variation":
        return {
            "lighting": {
                "ambient_lux_delta": -180 - seed * 10,
                "key_light_intensity_scale": 0.45,
                "color_temperature_kelvin": 4100,
                "camera_exposure_bias_ev": -0.7,
            }
        }
    if variation_name == "object_rotation":
        return {
            "object_pose_delta": {
                "target_selector": "task_target_object_or_nearest_pick_object",
                "yaw_degrees": 30.0 + seed,
                "pitch_degrees": 0.0,
                "roll_degrees": 0.0,
            }
        }
    if variation_name == "cart_shifted":
        return {
            "cart_pose_delta": {
                "target_selector": "cart_or_mobile_shelf_near_task",
                "translation_m": {"x": 0.35, "y": -0.18, "z": 0.0},
                "yaw_degrees": 5.0,
            }
        }
    if variation_name == "blocked_path":
        return {
            "path_obstacle": {
                "obstacle_type": "rolling_cart",
                "placement": "approach_midpoint",
                "centerline_offset_m": 0.0,
                "width_m": 0.65,
                "clearance_policy": "must_replan_without_contact",
            }
        }
    if variation_name == "human_crossing":
        return {
            "dynamic_actor": {
                "actor_type": "human",
                "route": [
                    {"x": -0.6, "y": 0.35, "z": 0.0},
                    {"x": 0.8, "y": 0.35, "z": 0.0},
                ],
                "speed_mps": 1.1,
                "trigger": "robot_within_2m_of_crossing",
            }
        }
    if variation_name == "forklift_nearby":
        return {
            "forklift_actor": {
                "actor_type": "forklift",
                "pose": {"x": 1.4, "y": -0.8, "z": 0.0, "yaw_degrees": 90.0},
                "motion_profile": "stationary_or_slow_roll",
                "safety_zone_m": 2.0,
            }
        }
    if variation_name == "occlusion":
        return {
            "occluder": {
                "object_type": "carton_stack",
                "target": "label_or_grasp_point",
                "coverage_fraction": 0.45,
                "pose_hint": "between_robot_camera_and_target",
            }
        }
    if variation_name == "glare":
        return {
            "glare_source": {
                "light_type": "specular_strip",
                "incidence_angle_degrees": 18.0,
                "intensity_lux": 850,
                "material_target": "floor_or_label",
            }
        }
    if variation_name == "missing_label":
        return {
            "label_visibility": {
                "target_label": "task_target_label",
                "visible": False,
                "replacement_state": "blank_or_torn_label_surface",
            }
        }
    if variation_name == "wrong_object_nearby":
        return {
            "distractor_object": {
                "object_role": "lookalike_wrong_object",
                "distance_to_target_m": 0.25,
                "label_similarity": "high",
                "placement": "within_candidate_grasp_or_drop_region",
            }
        }
    if variation_name == "narrow_approach_angle":
        return {
            "approach_constraint": {
                "allowed_width_m": 0.55,
                "approach_yaw_degrees": 65.0,
                "side_clearance_m": 0.1,
                "requires_precise_base_alignment": True,
            }
        }
    return {"review_mutation": {"variation_name": variation_name, "requires_owner_mapping": True}}


def _operation_kind(variation_name: str) -> str:
    if variation_name in {"lighting_variation", "glare"}:
        return "lighting_or_sensor_mutation"
    if variation_name in {"object_rotation", "cart_shifted"}:
        return "rigid_body_pose_mutation"
    if variation_name in {"human_crossing", "forklift_nearby"}:
        return "dynamic_actor_mutation"
    if variation_name in {"blocked_path", "occlusion", "wrong_object_nearby"}:
        return "spawn_static_obstacle_or_distractor"
    if variation_name == "missing_label":
        return "material_or_semantic_visibility_mutation"
    if variation_name == "narrow_approach_angle":
        return "navigation_constraint_mutation"
    return "review_mutation"


def _engine_operation_name(framework: str, variation_name: str) -> str:
    kind = _operation_kind(variation_name)
    prefixes = {
        "isaac_sim": "usd_stage",
        "isaac_lab_arena": "arena_cfg",
        "mujoco": "mjcf_patch",
        "pybullet": "pybullet_api",
        "newton": "newton_scene_graph",
    }
    return f"{prefixes.get(framework, framework)}.{kind}"


def _engine_mutation(
    *,
    framework: str,
    instance_id: str,
    variation_name: str,
    concrete_mutation: Mapping[str, Any],
) -> Dict[str, Any]:
    operation = {
        "operation": _engine_operation_name(framework, variation_name),
        "variation_name": variation_name,
        "parameters": dict(concrete_mutation),
        "requires_owner_engine_adapter": True,
    }
    return {
        "schema_version": SCENARIO_VARIATION_ENGINE_MUTATION_SCHEMA_VERSION,
        "framework": framework,
        "instance_id": instance_id,
        "status": "ready_for_owner_engine_adapter",
        "operation_count": 1,
        "operations": [operation],
        "simulator_execution_proven": False,
        "claim_boundary": "engine_mutation_payload_not_simulator_execution_proof",
    }


def _engine_mutation_plan(instances: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    plan: Dict[str, Dict[str, Any]] = {}
    for framework in SCENARIO_VARIATION_ENGINE_TARGETS:
        mutations = [
            {
                "instance_id": instance.get("instance_id"),
                "variation_name": instance.get("variation_name"),
                "operation_count": _mapping(
                    _mapping(instance.get("engine_mutations")).get(framework)
                ).get("operation_count", 0),
                "operations": _mapping(
                    _mapping(instance.get("engine_mutations")).get(framework)
                ).get("operations", []),
            }
            for instance in instances
        ]
        plan[framework] = {
            "framework": framework,
            "status": "ready_for_owner_engine_adapter" if mutations else "blocked_missing_instances",
            "mutation_count": len(mutations),
            "mutations": mutations,
            "requires_owner_runtime_or_plugin": True,
            "simulator_execution_proven": False,
        }
    return plan


def _unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def build_scenario_variation_instances(
    *,
    capture_root: str | Path,
    output_dir: str | Path | None = None,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Write concrete variation instances and engine mutation payloads.

    The output is deterministic, capture/task/scenario grounded, and explicitly
    not a claim that any simulator or robot run has happened.
    """

    context = resolve_local_capture_context(capture_root)
    automation_dir = Path(output_dir).resolve() if output_dir else context.pipeline_root / "simulation_automation"
    ensure_dir(automation_dir)
    resolved_generated_at = generated_at or utc_now_iso()

    families = _scenario_family_rows(
        pipeline_dir=context.pipeline_root,
        generated_at=resolved_generated_at,
    )
    instances: List[Dict[str, Any]] = []
    for family_index, family in enumerate(families, start=1):
        for variation_index, variation in enumerate(family.get("variations") or [], start=1):
            if not isinstance(variation, Mapping):
                continue
            variation_name = _canonical_variation_name(
                variation.get("variation_name") or variation.get("variation_id")
            )
            if variation_name not in SCENARIO_VARIATION_NAMES:
                continue
            scenario_id = _string(family.get("scenario_id")) or f"scenario_{family_index}"
            task_id = _string(family.get("task_id")) or "task_unknown"
            family_id = _string(family.get("family_id")) or f"family_{family_index}"
            instance_id = (
                f"variation_{_safe_id(task_id)}_{_safe_id(scenario_id)}_{variation_name}"
            )
            mutation = _concrete_mutation(variation_name, ordinal=len(instances))
            engine_mutations = {
                framework: _engine_mutation(
                    framework=framework,
                    instance_id=instance_id,
                    variation_name=variation_name,
                    concrete_mutation=mutation,
                )
                for framework in SCENARIO_VARIATION_ENGINE_TARGETS
            }
            instances.append(
                {
                    "instance_id": instance_id,
                    "family_id": family_id,
                    "scenario_id": scenario_id,
                    "task_id": task_id,
                    "robot_profile_id": family.get("robot_profile_id"),
                    "variation_id": _string(variation.get("variation_id")) or variation_name,
                    "variation_name": variation_name,
                    "label": _string(variation.get("label")) or variation_name.replace("_", " ").title(),
                    "scenario_status": _string(variation.get("scenario_status"))
                    or "review_required",
                    "source": family.get("source"),
                    "concrete_mutation": mutation,
                    "engine_mutations": engine_mutations,
                    "review_required": True,
                    "simulator_execution_proven": False,
                    "claim_boundary": "concrete_variation_is_engine_input_not_robot_or_sim_result",
                }
            )

    manifest = {
        "schema_version": SCENARIO_VARIATION_INSTANCES_SCHEMA_VERSION,
        "generated_at": resolved_generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "completed" if instances else "blocked_missing_scenario_families",
        "required_variation_names": list(SCENARIO_VARIATION_NAMES),
        "variation_names_instantiated": _unique(
            _string(instance.get("variation_name")) for instance in instances
        ),
        "engine_targets": list(SCENARIO_VARIATION_ENGINE_TARGETS),
        "family_count": len(families),
        "instance_count": len(instances),
        "instances": instances,
        "engine_mutation_plan": _engine_mutation_plan(instances),
        "source_artifacts": {
            "scenario_family_library": _relative_to(
                automation_dir,
                context.pipeline_root / "robot_eval_dataset" / "scenario_family_library.json",
            )
            if (context.pipeline_root / "robot_eval_dataset" / "scenario_family_library.json").is_file()
            else None,
            "scenario_cards": _relative_to(
                automation_dir,
                context.pipeline_root / "robot_eval_dataset" / "scenario_cards.json",
            )
            if (context.pipeline_root / "robot_eval_dataset" / "scenario_cards.json").is_file()
            else None,
        },
        "deterministic_fingerprint": _sha_payload(
            {
                "scene_id": context.scene_id,
                "capture_id": context.capture_id,
                "instance_ids": [instance["instance_id"] for instance in instances],
                "variation_names": [instance["variation_name"] for instance in instances],
            }
        ),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(automation_dir / "scenario_variation_instances.json", manifest)
    return manifest
