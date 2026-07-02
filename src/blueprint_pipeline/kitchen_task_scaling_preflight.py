"""Local task-scaling preflight for the Lightwheel kitchen manipulation lane.

The preflight is deliberately no-spend: it runs the existing dry-render path and
normalizes its sidecars into task-by-task gates. A task that passes here is
eligible for the paid WAM/SAM3/DA3 pipeline; it is not itself a generated-video
success claim.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import utc_now_iso
from .g1_render_noise_audit import normalize_legacy_robot_material_mode
from .wam_auxiliary_observation import build_wam_auxiliary_observation_manifest


SCHEMA_VERSION = "kitchen_task_scaling_preflight.v1"
LOCAL_PREFLIGHT_GATE_SET_VERSION = "kitchen_task_scaling_local_gates.v1"
POLICY_OBSERVATION_EXPORT_SCHEMA_VERSION = "kitchen_task_policy_observation_export.v1"
POLICY_OBSERVATION_EXPORT_INDEX_SCHEMA_VERSION = (
    "kitchen_task_policy_observation_export_index.v1"
)
MIN_FULL_KITCHEN_OBJECT_COUNT = 20
RUNNER_RELATIVE = "scripts/run_isaac_g1_kitchen_parity_eval.py"
UNITREE_G1_SONIC_STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "projected_gravity": 3,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_kitchen_usd_candidates(repo_root: Path | None = None) -> list[Path]:
    root = repo_root or _repo_root()
    return [
        root
        / "output/first-gpu-walkthrough2-storage/local-blueprint/scenes/"
        "first-gpu-walkthrough-2/captures/downloads-walkthrough2-20260611/"
        "pipeline/lightwheel_kitchen_isaac_scenarios/assets/Collected_KitchenRoom/"
        "KitchenRoom.usd",
        root / "assets/Collected_KitchenRoom/KitchenRoom.usd",
        root / "output/isaac_g1_dynamic_standing_contact_floor_asset/Collected_KitchenRoom/KitchenRoom.usd",
    ]


def resolve_kitchen_usd(value: str | None = None) -> Path | None:
    raw = (value or os.getenv("BLUEPRINT_KITCHEN_USD") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        return path.resolve() if path.is_file() else path
    for candidate in default_kitchen_usd_candidates():
        if candidate.is_file():
            return candidate.resolve()
    return None


def materialize_kitchen_usd_from_source(
    *,
    out_dir: Path,
    source_zip: str | Path | None = None,
    source_repo_root: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    if not source_zip and not source_repo_root:
        return None, {"status": "skipped", "reason": "no_source_zip_or_source_repo_root"}
    try:
        from blueprint_pipeline.lightwheel_kitchen_isaac_scenarios import (
            build_lightwheel_kitchen_isaac_scenarios,
        )

        result = build_lightwheel_kitchen_isaac_scenarios(
            source_zip=source_zip,
            source_repo_root=source_repo_root,
            output_dir=out_dir / "lightwheel_kitchen_isaac_scenarios",
        )
    except Exception as exc:  # noqa: BLE001 - convert asset setup failures to manifest blockers
        return None, {
            "status": "blocked",
            "blockers": ["lightwheel_kitchen_asset_materialization_failed"],
            "error": repr(exc),
        }
    scene_usd = Path(str(result.get("scene_usd_path") or "")).expanduser()
    if scene_usd.is_file():
        return scene_usd.resolve(), {
            "status": "complete",
            "scene_usd_path": str(scene_usd.resolve()),
            "handoff_manifest": result.get("manifest_path"),
        }
    return None, {
        "status": "blocked",
        "blockers": ["lightwheel_kitchen_materialized_scene_usd_missing"],
        "result": result,
    }


def default_task_specs() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "sink_faucet",
            "scenario_id": "lightwheel_kitchen_task_01_sink_faucet",
            "description": "Stand at the kitchen sink and turn on the faucet.",
            "required_target_terms": ["faucet", "sink"],
            "zone": "lower_front_manipulation",
            "preferred_stance_distance_m": 0.24,
            "stance_distance_candidates_m": [0.16, 0.18, 0.22, 0.24, 0.27, 0.30, 0.32, 0.38, 0.45, 0.55],
        },
        {
            "task_id": "stovetop_knob",
            "scenario_id": "lightwheel_kitchen_task_02_stovetop_knob",
            "description": "Stand at the stovetop and turn a front burner knob.",
            "required_target_terms": ["knob", "stovetop", "stove", "burner"],
            "zone": "side_appliance_manipulation",
            "preferred_stance_distance_m": 0.24,
            "stance_distance_candidates_m": [0.16, 0.18, 0.22, 0.24, 0.27, 0.30, 0.32, 0.38, 0.45, 0.55],
        },
        {
            "task_id": "top_cabinet",
            "scenario_id": "lightwheel_kitchen_task_03_top_cabinet",
            "description": "Stand at the upper kitchen cabinet and reach for the cabinet handle.",
            "required_target_terms": ["cabinet", "handle", "knob"],
            "zone": "high_reach_manipulation",
            "preferred_stance_distance_m": 0.55,
            "stance_distance_candidates_m": [0.38, 0.45, 0.55, 0.65, 0.75, 0.9],
        },
    ]


def perception_target_prompts_for_task(task_spec: Mapping[str, Any]) -> list[str]:
    """Build concise object prompts for SAM3/DA3/object-index support backends."""
    prompts: list[str] = []
    description = str(task_spec.get("description") or "").strip()
    if description:
        prompts.append(description)
    for term in task_spec.get("required_target_terms") or []:
        cleaned = str(term).strip()
        if cleaned:
            prompts.append(cleaned)
    task_id = str(task_spec.get("task_id") or "")
    if task_id == "sink_faucet":
        prompts.extend(["faucet lever", "sink faucet handle"])
    elif task_id == "stovetop_knob":
        prompts.extend(["front stove knob", "burner control knob"])
    elif task_id == "top_cabinet":
        prompts.extend(["upper cabinet handle", "cabinet pull"])
    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        key = prompt.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(prompt)
    return deduped


def target_object_id_candidates_for_task(task_spec: Mapping[str, Any]) -> list[str]:
    """Ordered USD name/label aliases for stance planning: coarse fixtures first."""
    task_id = str(task_spec.get("task_id") or "")
    if task_id == "sink_faucet":
        return ["sink", "basin"]
    if task_id == "stovetop_knob":
        return ["stovetop", "cooktop", "stove", "range", "burner"]
    if task_id == "top_cabinet":
        return ["topcabinet", "top_cabinet", "upper cabinet", "cabinet", "cupboard"]
    return [str(term).strip() for term in task_spec.get("required_target_terms") or [] if str(term).strip()]


def affordance_object_id_candidates_for_task(task_spec: Mapping[str, Any]) -> list[str]:
    """Ordered USD name/label aliases for manipulation focus: fine affordances first."""
    task_id = str(task_spec.get("task_id") or "")
    if task_id == "sink_faucet":
        return ["handle", "lever", "mixer", "faucet", "tap", "spout"]
    if task_id == "stovetop_knob":
        return ["knob", "dial", "control", "burner"]
    if task_id == "top_cabinet":
        return ["handle", "pull", "knob"]
    return []


def build_request(*, kitchen_usd: Path, task_specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for spec in task_specs:
        description = str(spec["description"])
        scenarios.append(
            {
                "scenario_id": spec["scenario_id"],
                "task_id": spec["task_id"],
                "description": description,
                "task": description,
                "task_description": description,
                "task_instruction": description,
                "task_target_deferred": True,
                "floor_z_hint": 0.05,
                "preferred_stance_distance_m": spec.get("preferred_stance_distance_m"),
                "stance_distance_candidates_m": list(spec.get("stance_distance_candidates_m") or []),
                "perception_target_prompts": perception_target_prompts_for_task(spec),
                "target_object_ids": target_object_id_candidates_for_task(spec),
                "affordance_object_ids": affordance_object_id_candidates_for_task(spec),
            }
        )
    return {
        "schema_version": "kitchen_task_scaling_preflight_request.v1",
        "kitchen_usd": str(kitchen_usd),
        "scenarios": scenarios,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _gate(name: str, passed: bool, *, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "evidence": dict(evidence or {}),
    }


def _pending_gate(name: str, *, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": "PENDING", "evidence": dict(evidence or {})}


def _roles_by_arm(geometry: Mapping[str, Any]) -> dict[str, set[str]]:
    raw = geometry.get("arm_roles_in_frame_by_arm")
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, set[str]] = {}
    for arm, roles in raw.items():
        if isinstance(roles, Sequence) and not isinstance(roles, (str, bytes)):
            result[str(arm)] = {str(role) for role in roles}
    return result


def _both_hands_wrists_visible(geometry: Mapping[str, Any]) -> bool:
    required = geometry.get("required_arms") or ["left", "right"]
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        required = ["left", "right"]
    roles_by_arm = _roles_by_arm(geometry)
    for arm in required:
        roles = roles_by_arm.get(str(arm), set())
        if not {"hand", "wrist"}.issubset(roles):
            return False
    return bool(required)


GEOMETRY_REACH_BLOCKERS = {
    "manipulation_pov_affordance_outside_g1_reach_envelope",
    "manipulation_pov_effector_too_far_from_affordance",
    "manipulation_pov_reach_feasibility_unverified",
}


def _non_reach_geometry_blockers(geometry: Mapping[str, Any]) -> list[str]:
    return [
        str(blocker)
        for blocker in (geometry.get("blockers") or [])
        if str(blocker) not in GEOMETRY_REACH_BLOCKERS
    ]


def _selected_candidate_reachability(
    stance_plan: Mapping[str, Any],
) -> dict[str, Any]:
    if not stance_plan.get("reachability_selection_enabled"):
        return {
            "status": "not_applicable",
            "reason": "stance_plan_has_no_affordance_reachability_estimates",
        }
    candidates = stance_plan.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return {"status": "FAIL", "blockers": ["task_stance_candidates_missing"]}
    try:
        selected_index = int(stance_plan.get("selected_candidate_index") or 0)
    except Exception:  # noqa: BLE001
        selected_index = -1
    if selected_index < 0 or selected_index >= len(candidates):
        return {"status": "FAIL", "blockers": ["task_stance_selected_candidate_missing"]}
    selected = candidates[selected_index]
    if not isinstance(selected, Mapping):
        return {"status": "FAIL", "blockers": ["task_stance_selected_candidate_invalid"]}
    estimate = selected.get("reachability_estimate")
    if not isinstance(estimate, Mapping):
        return {"status": "FAIL", "blockers": ["task_stance_reachability_estimate_missing"]}
    passing_candidates = [
        idx
        for idx, candidate in enumerate(candidates)
        if isinstance(candidate, Mapping)
        and isinstance(candidate.get("reachability_estimate"), Mapping)
        and candidate["reachability_estimate"].get("status") == "PASS"
    ]
    reach_clearance_conflict = _reach_clearance_conflict(candidates)
    return {
        "status": estimate.get("status") or "FAIL",
        "blockers": list(estimate.get("blockers") or []),
        "selected_candidate_index": selected_index,
        "passing_candidate_count": len(passing_candidates),
        "passing_candidate_indices": passing_candidates[:20],
        "reach_clearance_conflict": reach_clearance_conflict,
        "selected_candidate_reachability": estimate,
    }


def _candidate_placement_passed(candidate: Mapping[str, Any]) -> bool:
    placement = candidate.get("placement_validation")
    if not isinstance(placement, Mapping):
        return False
    return str(placement.get("status") or "").lower() in {"pass", "passed", "accepted"}


def _reach_clearance_conflict(candidates: Sequence[Any]) -> dict[str, Any]:
    reachable_but_placement_blocked: list[dict[str, Any]] = []
    placement_clean_but_reach_blocked: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        reach = candidate.get("reachability_estimate")
        if not isinstance(reach, Mapping):
            continue
        placement = candidate.get("placement_validation")
        placement = placement if isinstance(placement, Mapping) else {}
        row = {
            "candidate_index": idx,
            "pose": candidate.get("pose"),
            "yaw": candidate.get("yaw"),
            "standoff_from_target_surface_m": candidate.get("standoff_from_target_surface_m"),
            "angle_offset_deg": candidate.get("angle_offset_deg"),
            "placement_status": placement.get("status"),
            "placement_blockers": list(placement.get("blockers") or []),
            "reach_status": reach.get("status"),
            "reach_blockers": list(reach.get("blockers") or []),
            "nearest_shoulder_to_affordance_m": reach.get("nearest_shoulder_to_affordance_m"),
            "nearest_seed_effector_to_affordance_m": reach.get(
                "nearest_seed_effector_to_affordance_m"
            ),
        }
        if reach.get("status") == "PASS" and not _candidate_placement_passed(candidate):
            reachable_but_placement_blocked.append(row)
        elif reach.get("status") != "PASS" and _candidate_placement_passed(candidate):
            placement_clean_but_reach_blocked.append(row)
    status = (
        "detected"
        if reachable_but_placement_blocked and placement_clean_but_reach_blocked
        else "not_detected"
    )
    return {
        "status": status,
        "reachable_but_placement_blocked_count": len(reachable_but_placement_blocked),
        "placement_clean_but_reach_blocked_count": len(placement_clean_but_reach_blocked),
        "reachable_but_placement_blocked_examples": reachable_but_placement_blocked[:5],
        "placement_clean_but_reach_blocked_examples": placement_clean_but_reach_blocked[:5],
        "next_step": (
            "improve_initial_arm_or_torso_seed_before_wam"
            if status == "detected"
            else None
        ),
        "claim_boundary": (
            "This conflict is a local planning diagnostic. It does not prove physical reach, "
            "contact, task completion, WAM visual success, or deployment readiness."
        ),
    }


def _semantic_resolution_passed(
    stance_plan: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> bool:
    resolution = stance_plan.get("target_resolution")
    if not isinstance(resolution, Mapping):
        return False
    source = str(resolution.get("source") or summary.get("target", {}).get("resolution_source") or "")
    if resolution.get("status") != "resolved":
        return False
    if source.startswith("scene_placement"):
        return True
    selected = resolution.get("selected")
    return (
        source == "usd_prim_bounds"
        and isinstance(selected, Mapping)
        and selected.get("prim_path")
        and isinstance(selected.get("target_object_priority"), int)
    )


def evaluate_local_task_gates(
    *,
    task_spec: Mapping[str, Any],
    task_dir: Path,
    min_scene_objects: int = MIN_FULL_KITCHEN_OBJECT_COUNT,
) -> dict[str, Any]:
    summary_path = task_dir / "dry_render_summary.json"
    stance_path = task_dir / "task_stance_plan.json"
    placement_path = task_dir / "placement_validation.json"
    geometry_path = task_dir / "manipulation_pov_geometry.json"
    summary = _read_json(summary_path)
    stance_plan = _read_json(stance_path)
    placement = _read_json(placement_path)
    geometry = _read_json(geometry_path)
    scene = summary.get("scene") if isinstance(summary.get("scene"), Mapping) else {}
    object_count = int(scene.get("object_count") or 0)
    geometry_extension = geometry.get("arm_extension") if isinstance(geometry.get("arm_extension"), Mapping) else {}
    reach_feasibility = geometry.get("reach_feasibility") if isinstance(geometry.get("reach_feasibility"), Mapping) else {}
    non_reach_geometry_blockers = _non_reach_geometry_blockers(geometry)
    selected_reachability = _selected_candidate_reachability(stance_plan)
    stance_reach_required = bool(
        stance_plan.get("reachability_selection_enabled")
        or stance_plan.get("task_affordance_xyz") is not None
        or isinstance(stance_plan.get("affordance_resolution"), Mapping)
    )
    selected_stance_reach_passed = (
        selected_reachability.get("status") == "PASS"
        if stance_reach_required
        else True
    )
    rendered_seed_reach_passed = reach_feasibility.get("status") == "PASS"
    gates = [
        _gate(
            "task_stance_plan.status = accepted",
            stance_plan.get("status") == "accepted",
            evidence={"path": str(stance_path), "status": stance_plan.get("status")},
        ),
        _gate(
            "placement_validation.status = PASS",
            placement.get("status") == "PASS",
            evidence={"path": str(placement_path), "status": placement.get("status")},
        ),
        _gate(
            "full kitchen scene loaded",
            object_count >= int(min_scene_objects),
            evidence={
                "path": str(summary_path),
                "object_count": object_count,
                "min_scene_objects": int(min_scene_objects),
            },
        ),
        _gate(
            "target resolves semantically",
            _semantic_resolution_passed(stance_plan, summary),
            evidence={
                "path": str(stance_path),
                "target_resolution": stance_plan.get("target_resolution"),
            },
        ),
        _gate(
            "manipulation POV has no non-reach framing blockers",
            not non_reach_geometry_blockers,
            evidence={
                "path": str(geometry_path),
                "status": geometry.get("status"),
                "blockers": geometry.get("blockers"),
                "non_reach_blockers": non_reach_geometry_blockers,
                "reach_blockers_are_handled_by_static_reach_gate": True,
            },
        ),
        _gate(
            "task stance selected candidate passes static G1 reach envelope",
            selected_stance_reach_passed,
            evidence={
                "path": str(stance_path),
                "reachability_selection_enabled": stance_reach_required,
                "selected_candidate_reachability": selected_reachability,
            },
        ),
        _gate(
            "target visible in manipulation POV",
            bool(geometry.get("target_in_frame")),
            evidence={"path": str(geometry_path), "target_in_frame": geometry.get("target_in_frame")},
        ),
        _gate(
            "both hands/wrists visible",
            _both_hands_wrists_visible(geometry),
            evidence={
                "path": str(geometry_path),
                "required_arms": geometry.get("required_arms"),
                "arm_roles_in_frame_by_arm": geometry.get("arm_roles_in_frame_by_arm"),
            },
        ),
        _gate(
            "straight arms-out seed passes framing",
            geometry_extension.get("status") == "PASS",
            evidence={"path": str(geometry_path), "arm_extension": geometry_extension},
        ),
        _gate(
            "rendered seed arm can plausibly reach affordance",
            rendered_seed_reach_passed,
            evidence={"path": str(geometry_path), "reach_feasibility": reach_feasibility},
        ),
    ]
    local_passed = all(gate["status"] == "PASS" for gate in gates)
    target_resolution = stance_plan.get("target_resolution") if isinstance(
        stance_plan.get("target_resolution"), Mapping
    ) else {}
    affordance_resolution = stance_plan.get("affordance_resolution") if isinstance(
        stance_plan.get("affordance_resolution"), Mapping
    ) else {}
    return {
        "schema_version": LOCAL_PREFLIGHT_GATE_SET_VERSION,
        "task_id": task_spec.get("task_id"),
        "scenario_id": task_spec.get("scenario_id"),
        "description": task_spec.get("description"),
        "zone": task_spec.get("zone"),
        "status": "passed" if local_passed else "blocked",
        "task_dir": str(task_dir),
        "artifacts": {
            "dry_render_summary": str(summary_path),
            "task_stance_plan": str(stance_path),
            "placement_validation": str(placement_path),
            "manipulation_pov_geometry": str(geometry_path),
            "dry_render_preview": str(task_dir / "dry_render_preview.png"),
        },
        "reachability_evidence": {
            "authority": "g1_static_reach_envelope_from_sim_geometry",
            "status": reach_feasibility.get("status") or "unverified",
            "target_resolution": target_resolution,
            "affordance_resolution": affordance_resolution,
            "perception_target_prompts": perception_target_prompts_for_task(task_spec),
            "selected_candidate_reachability": selected_reachability,
            "reach_clearance_conflict": selected_reachability.get(
                "reach_clearance_conflict"
            ),
            "selected_candidate_reachability_required_for_local_preflight": stance_reach_required,
            "static_reach_required_for_local_preflight": True,
            "sam3_da3_role": (
                "optional_affordance_mask_depth_refinement; not final reach authority"
            ),
        },
        "local_gates": gates,
        "downstream_gates": [
            _pending_gate(
                "WAM completes without edge/entropy collapse",
                evidence={"reason": "not_run_by_local_no_spend_preflight"},
            ),
            _pending_gate(
                "SAM3/DA3 post-harness completes if configured",
                evidence={"reason": "not_run_by_local_no_spend_preflight"},
            ),
        ],
        "claim_boundary": (
            "Local preflight gates placement, target resolution, and manipulation seed framing. "
            "Static reach feasibility is USD/sim geometry; SAM3/DA3 can refine target/mask/depth "
            "evidence when configured, but it does not prove contact, WAM visual success, "
            "manipulation success, deployment readiness, or physical robot safety."
        ),
    }


def _selected_specs(task_ids: Sequence[str]) -> list[dict[str, Any]]:
    specs = default_task_specs()
    if not task_ids:
        return specs
    wanted = set(task_ids)
    selected = [spec for spec in specs if str(spec["task_id"]) in wanted]
    missing = sorted(wanted - {str(spec["task_id"]) for spec in selected})
    if missing:
        raise ValueError(f"unknown_kitchen_task_ids:{','.join(missing)}")
    return selected


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _blocked_manifest(
    *,
    out_dir: Path,
    kitchen_usd: Path | None,
    task_specs: Sequence[Mapping[str, Any]],
    blockers: Sequence[str],
    kitchen_asset_materialization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tasks = []
    for spec in task_specs:
        tasks.append(
            {
                "task_id": spec.get("task_id"),
                "scenario_id": spec.get("scenario_id"),
                "description": spec.get("description"),
                "status": "blocked",
                "local_gates": [
                    _gate(
                        "full kitchen scene loaded",
                        False,
                        evidence={
                            "kitchen_usd": str(kitchen_usd) if kitchen_usd else None,
                            "blockers": list(blockers),
                        },
                    )
                ],
                "downstream_gates": [
                    _pending_gate(
                        "WAM completes without edge/entropy collapse",
                        evidence={"reason": "local_preflight_blocked"},
                    ),
                    _pending_gate(
                        "SAM3/DA3 post-harness completes if configured",
                        evidence={"reason": "local_preflight_blocked"},
                    ),
                ],
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "local_preflight_status": "blocked",
        "wam_pipeline_status": "not_run",
        "kitchen_usd": str(kitchen_usd) if kitchen_usd else None,
        "kitchen_asset_materialization": dict(kitchen_asset_materialization or {}),
        "blockers": list(blockers),
        "tasks": tasks,
    }
    _write_json(out_dir / "kitchen_task_scaling_preflight_manifest.json", manifest)
    return manifest


def run_preflight(
    *,
    out_dir: Path,
    kitchen_usd: Path | None = None,
    g1_usd: str | None = None,
    source_zip: str | Path | None = None,
    source_repo_root: str | Path | None = None,
    robot_review_material_override: bool = False,
    task_ids: Sequence[str] = (),
    min_scene_objects: int = MIN_FULL_KITCHEN_OBJECT_COUNT,
    python_executable: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _selected_specs(task_ids)
    resolved_kitchen = kitchen_usd or resolve_kitchen_usd()
    materialization: dict[str, Any] = {"status": "not_needed"}
    if resolved_kitchen is None or not Path(resolved_kitchen).is_file():
        materialized, materialization = materialize_kitchen_usd_from_source(
            out_dir=out_dir,
            source_zip=source_zip,
            source_repo_root=source_repo_root,
        )
        if materialized is not None:
            resolved_kitchen = materialized
    if resolved_kitchen is None or not Path(resolved_kitchen).is_file():
        return _blocked_manifest(
            out_dir=out_dir,
            kitchen_usd=resolved_kitchen,
            task_specs=specs,
            blockers=[
                "missing_full_kitchen_usd",
                *[str(item) for item in materialization.get("blockers", [])],
            ],
            kitchen_asset_materialization=materialization,
        )

    request = build_request(kitchen_usd=Path(resolved_kitchen), task_specs=specs)
    request_path = out_dir / "kitchen_task_scaling_request.json"
    _write_json(request_path, request)

    runner = _repo_root() / RUNNER_RELATIVE
    cmd = [
        python_executable or sys.executable,
        str(runner),
        "--request",
        str(request_path),
        "--kitchen-usd",
        str(resolved_kitchen),
        "--out-dir",
        str(out_dir),
        "--dry-render",
        "--manipulation-reach-arm",
        "both",
        "--width",
        "1280",
        "--height",
        "960",
        "--camera-vfov",
        "90",
    ]
    if robot_review_material_override:
        cmd.append("--robot-review-material-override")
    if g1_usd:
        cmd.extend(["--g1-usd", g1_usd])
    completed = subprocess.run(cmd, cwd=str(_repo_root()), text=True, capture_output=True, check=False)
    run_record = {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    _write_json(out_dir / "dry_render_command_result.json", run_record)
    if completed.returncode != 0:
        return _blocked_manifest(
            out_dir=out_dir,
            kitchen_usd=Path(resolved_kitchen),
            task_specs=specs,
            blockers=["dry_render_command_failed"],
        )

    task_reports = []
    for spec in specs:
        task_dir = out_dir / "dry_render" / str(spec["scenario_id"])
        task_reports.append(
            evaluate_local_task_gates(
                task_spec=spec,
                task_dir=task_dir,
                min_scene_objects=min_scene_objects,
            )
        )
    local_passed = all(report.get("status") == "passed" for report in task_reports)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_local_preflight" if local_passed else "blocked",
        "local_preflight_status": "passed" if local_passed else "blocked",
        "wam_pipeline_status": "not_run",
        "kitchen_usd": str(resolved_kitchen),
        "g1_usd": g1_usd,
        "kitchen_asset_materialization": materialization,
        "seed_media_preferences": {
            "robot_review_material_override": bool(robot_review_material_override),
            "robot_material_mode": (
                "neutral_matte_untextured_g1"
                if robot_review_material_override
                else "preserve_authored_g1_materials_when_available"
            ),
            # Spec-normalized label: the matte override is an explicit white proxy; preserved
            # authored materials stay textured_unverified until a render-noise-audit material
            # resolution manifest proves texture refs resolved (never silently verified_textured).
            "robot_material_mode_normalized": normalize_legacy_robot_material_mode(
                "neutral_matte_untextured_g1"
                if robot_review_material_override
                else "preserve_authored_g1_materials_when_available"
            ),
        },
        "min_scene_objects": int(min_scene_objects),
        "request_path": str(request_path),
        "dry_render_command_result": str(out_dir / "dry_render_command_result.json"),
        "tasks": task_reports,
        "next_step": (
            "eligible_for_sink_faucet_wam_run"
            if local_passed
            else "fix_local_preflight_blockers_before_paid_wam"
        ),
        "claim_boundary": (
            "A passed local preflight means the task is eligible for the paid WAM pipeline. "
            "It is not generated-video success, SAM3/DA3 completion, manipulation success, "
            "deployment approval, or physical robot readiness."
        ),
    }
    _write_json(out_dir / "kitchen_task_scaling_preflight_manifest.json", manifest)
    return manifest


def _neutral_unitree_g1_sonic_state() -> dict[str, list[float]]:
    state = {
        key: [0.0] * int(dim)
        for key, dim in UNITREE_G1_SONIC_STATE_DIMS.items()
    }
    state["projected_gravity"] = [0.0, 0.0, -1.0]
    return state


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _selected_resolution_id(resolution: Mapping[str, Any]) -> str | None:
    selected = resolution.get("selected")
    if not isinstance(selected, Mapping):
        return None
    for key in ("target_object_id", "object_id", "target_object_label", "prim_path"):
        value = str(selected.get(key) or "").strip()
        if value:
            return value
    return None


def _target_object_id_for_export(
    *,
    task_report: Mapping[str, Any],
    stance_plan: Mapping[str, Any],
) -> str:
    reachability = task_report.get("reachability_evidence")
    reachability = reachability if isinstance(reachability, Mapping) else {}
    for resolution in (
        reachability.get("affordance_resolution"),
        stance_plan.get("affordance_resolution"),
        reachability.get("target_resolution"),
        stance_plan.get("target_resolution"),
    ):
        if isinstance(resolution, Mapping):
            selected = _selected_resolution_id(resolution)
            if selected:
                return selected
    return str(task_report.get("task_id") or "kitchen_task_target")


def _target_bbox_for_export(
    *,
    stance_plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    bounds = _first_mapping(stance_plan.get("task_affordance_bounds"), stance_plan.get("task_target_bounds"))
    if not bounds:
        return None
    bbox_min = bounds.get("bbox_min_xyz")
    bbox_max = bounds.get("bbox_max_xyz")
    if not isinstance(bbox_min, Sequence) or isinstance(bbox_min, (str, bytes)):
        return None
    if not isinstance(bbox_max, Sequence) or isinstance(bbox_max, (str, bytes)):
        return None
    return {"bbox_min_xyz": list(bbox_min), "bbox_max_xyz": list(bbox_max)}


def _bridge_readiness_from_geometry(geometry: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    camera_meta = geometry.get("camera_meta")
    camera_meta = camera_meta if isinstance(camera_meta, Mapping) else {}
    if geometry.get("status") != "PASS":
        blockers.append("manipulation_pov_geometry_not_passed")
    if not camera_meta.get("camera_eye_xyz") or not camera_meta.get("camera_target_xyz"):
        blockers.append("manipulation_pov_camera_meta_missing")
    arm_points_by_arm = camera_meta.get("arm_link_points_by_arm_xyz")
    if not isinstance(arm_points_by_arm, Mapping) or not arm_points_by_arm:
        blockers.append("manipulation_pov_arm_link_points_missing")
    else:
        for arm in ("left", "right"):
            roles = arm_points_by_arm.get(arm)
            if not isinstance(roles, Mapping) or not {"shoulder", "elbow", "wrist", "hand"}.issubset(roles):
                blockers.append(f"manipulation_pov_{arm}_arm_link_chain_incomplete")
    return {
        "schema_version": "kitchen_task_action_projection_bridge_readiness.v1",
        "status": "ready" if not blockers else "blocked",
        "bridge": "isaac_geometry_policy_action_projection_bridge",
        "blockers": sorted(set(blockers)),
        "camera_meta_available": bool(camera_meta),
        "arm_link_points_by_arm_available": bool(arm_points_by_arm),
        "policy_ranking_claim_safe": False,
        "claim_boundary": (
            "Readiness means the WAM input has enough seed-geometry metadata for the "
            "existing action-projection bridge to attempt geometry-anchored conditioning. "
            "It is not proof of task success, contact, official whole-body control, physical "
            "robot safety, or ranking-safe policy evaluation."
        ),
    }


def _wam_seed_eligibility_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    render_source = str(summary.get("render_source") or "").strip()
    render_note = str(
        _first_mapping(summary.get("render_provenance"), summary.get("render_source_headers")).get(
            "render_source_note"
        )
        or _first_mapping(summary.get("render_source_headers")).get("X-Blueprint-Render-Note")
        or ""
    )
    blockers: list[str] = []
    if render_source == "dry_render_preview":
        blockers.append("source_policy_observation_is_dry_render_debug_preview")
    return {
        "schema_version": "kitchen_task_wam_seed_eligibility.v1",
        "status": "eligible" if not blockers else "blocked",
        "blockers": blockers,
        "render_source": render_source or None,
        "render_source_note": render_note or None,
        "source_policy_observation_visual_qa_required_before_wam": True,
        "claim_boundary": (
            "Geometry sidecars can be exported from dry-render preflight, but WAM should "
            "consume a review-quality RGB robot POV frame. Schematic/debug previews are "
            "not visually useful world-model seed frames."
        ),
    }


def _task_report_from_manifest(
    manifest: Mapping[str, Any],
    task_id: str,
) -> dict[str, Any] | None:
    for task in manifest.get("tasks") or []:
        if isinstance(task, Mapping) and str(task.get("task_id") or "") == str(task_id):
            return dict(task)
    return None


def _artifact_path(artifacts: Mapping[str, Any], key: str) -> Path:
    return Path(str(artifacts.get(key) or "")).expanduser()


def export_policy_observation_from_preflight(
    *,
    preflight_manifest_path: str | Path,
    task_id: str,
    out_dir: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Export one passed kitchen preflight task as a WAM policy-observation seed.

    This is a fail-closed adapter: a task with blocked local gates does not get promoted into
    WAM. The observation is explicitly marked as a local simulated/dry-render seed, not raw
    capture truth or manipulation success.
    """
    generated = generated_at or utc_now_iso()
    manifest_path = Path(preflight_manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_path)
    task_report = _task_report_from_manifest(manifest, task_id)
    default_out = manifest_path.parent / "wam_seed" / str(task_id)
    export_dir = Path(out_dir).expanduser().resolve() if out_dir else default_out.resolve()
    blockers: list[str] = []
    if task_report is None:
        blockers.append("kitchen_task_preflight_report_missing")
        task_report = {"task_id": task_id, "artifacts": {}}
    if task_report.get("status") != "passed":
        blockers.append("kitchen_task_local_preflight_not_passed")
    failed_gates = [
        str(gate.get("name"))
        for gate in task_report.get("local_gates") or []
        if isinstance(gate, Mapping) and gate.get("status") != "PASS"
    ]
    if failed_gates:
        blockers.append("kitchen_task_local_gate_failed")
    artifacts = task_report.get("artifacts") if isinstance(task_report.get("artifacts"), Mapping) else {}
    preview_path = _artifact_path(artifacts, "dry_render_preview")
    summary_path = _artifact_path(artifacts, "dry_render_summary")
    stance_path = _artifact_path(artifacts, "task_stance_plan")
    placement_path = _artifact_path(artifacts, "placement_validation")
    geometry_path = _artifact_path(artifacts, "manipulation_pov_geometry")
    required_paths = {
        "dry_render_preview": preview_path,
        "dry_render_summary": summary_path,
        "task_stance_plan": stance_path,
        "placement_validation": placement_path,
        "manipulation_pov_geometry": geometry_path,
    }
    for label, path in required_paths.items():
        if not str(path) or not path.is_file():
            blockers.append(f"{label}_missing")
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    stance_plan = _read_json(stance_path) if stance_path.is_file() else {}
    geometry = _read_json(geometry_path) if geometry_path.is_file() else {}
    wam_seed_eligibility = _wam_seed_eligibility_from_summary(summary)
    bridge_readiness = _bridge_readiness_from_geometry(geometry) if geometry else {
        "schema_version": "kitchen_task_action_projection_bridge_readiness.v1",
        "status": "blocked",
        "bridge": "isaac_geometry_policy_action_projection_bridge",
        "blockers": ["manipulation_pov_geometry_missing"],
        "policy_ranking_claim_safe": False,
    }
    if bridge_readiness.get("status") != "ready":
        blockers.append("kitchen_task_action_projection_bridge_not_ready")
    if blockers:
        export_manifest = {
            "schema_version": POLICY_OBSERVATION_EXPORT_SCHEMA_VERSION,
            "status": "blocked",
            "generated_at": generated,
            "preflight_manifest_path": str(manifest_path),
            "task_id": task_id,
            "blockers": sorted(set(blockers)),
            "failed_local_gates": failed_gates,
            "task_status": task_report.get("status"),
            "task_dir": task_report.get("task_dir"),
            "artifacts": {key: str(value) for key, value in required_paths.items()},
            "wam_seed_eligibility": wam_seed_eligibility,
            "action_projection_bridge_readiness": bridge_readiness,
            "claim_boundary": (
                "Blocked exports intentionally do not create an initial policy observation. "
                "A WAM run must start from a task that passed local placement, semantic target, "
                "and straight-arms seed framing gates."
            ),
        }
        _write_json(export_dir / "kitchen_task_policy_observation_export_manifest.json", export_manifest)
        return export_manifest

    target_object_id = _target_object_id_for_export(task_report=task_report, stance_plan=stance_plan)
    task_prompt = str(
        task_report.get("description")
        or summary.get("task")
        or f"Perform the kitchen task {task_id}."
    )
    visual = {
        "available": True,
        "camera_id": str(geometry.get("camera") or "robot_pov"),
        "camera_frame_path": str(preview_path.resolve()),
        "source_kind": "local_kitchen_task_scaling_dry_render_seed",
        "dry_render_preview_path": str(preview_path.resolve()),
        "manipulation_pov_geometry_path": str(geometry_path.resolve()),
        "isaac_manipulation_pov_geometry_path": str(geometry_path.resolve()),
        "placement_validation_path": str(placement_path.resolve()),
        "isaac_scene_manifest_path": str(placement_path.resolve()),
        "task_stance_plan_path": str(stance_path.resolve()),
        "target_projection": geometry.get("target_projection"),
        "target_affordance_xyz": geometry.get("target_affordance_xyz"),
        "camera_meta": geometry.get("camera_meta"),
        "claim_boundary": {
            "local_dry_render_seed": True,
            "capture_truth": False,
            "physical_robot_sensor_proof": False,
            "generated_media_success_proven": False,
        },
    }
    target_bbox = _target_bbox_for_export(stance_plan=stance_plan)
    affordance_points = []
    if isinstance(geometry.get("target_affordance_xyz"), Sequence) and not isinstance(
        geometry.get("target_affordance_xyz"), (str, bytes)
    ):
        affordance_points.append(
            {
                "xyz": list(geometry.get("target_affordance_xyz") or []),
                "source": "manipulation_pov_geometry_target_affordance_xyz",
            }
        )
    observation = {
        "schema_version": "initial_policy_observation.v1",
        "status": "ready",
        "generated_at": generated,
        "task_id": task_id,
        "task_prompt": task_prompt,
        "task_instruction": task_prompt,
        "task_description": task_prompt,
        "target_object_id": target_object_id,
        "robot_profile_id": "unitree_g1",
        "policy_source": "unitree_groot_n17_sonic_policy",
        "source_kind": "local_kitchen_task_scaling_dry_render_seed",
        "camera_frame_path": str(preview_path.resolve()),
        "visual_observation": visual,
        "task_stance_plan_path": str(stance_path.resolve()),
        "placement_validation_path": str(placement_path.resolve()),
        "isaac_scene_manifest_path": str(placement_path.resolve()),
        "manipulation_pov_geometry_path": str(geometry_path.resolve()),
        "isaac_manipulation_pov_geometry_path": str(geometry_path.resolve()),
        "task_stance": {
            "accepted_pose": stance_plan.get("accepted_pose"),
            "accepted_yaw": stance_plan.get("accepted_yaw"),
            "selected_candidate_index": stance_plan.get("selected_candidate_index"),
            "reachability_selection_enabled": bool(
                stance_plan.get("reachability_selection_enabled")
            ),
        },
        "target": {
            "object_id": target_object_id,
            "bbox": target_bbox,
            "affordance_points": affordance_points,
            "target_resolution": stance_plan.get("target_resolution"),
            "affordance_resolution": stance_plan.get("affordance_resolution"),
        },
        "target_bbox": target_bbox,
        "affordance_points": affordance_points,
        "unitree_g1_sonic_state": _neutral_unitree_g1_sonic_state(),
        "unitree_g1_sonic_state_source": "neutral_unitree_g1_sonic_contract_state",
        "unitree_g1_sonic_state_metadata": {
            "complete": True,
            "robot_profile_id": "unitree_g1",
            "state_vector_dims": dict(UNITREE_G1_SONIC_STATE_DIMS),
            "neutral_state_for_initial_sim_observation": True,
            "scene_or_task_specific_coordinates_hardcoded": False,
        },
        "local_preflight": {
            "manifest_path": str(manifest_path),
            "task_report_status": task_report.get("status"),
            "all_local_gates_passed": True,
            "artifacts": {key: str(value.resolve()) for key, value in required_paths.items()},
            "action_projection_bridge_readiness": bridge_readiness,
            "wam_seed_eligibility": wam_seed_eligibility,
        },
        "claim_boundary": {
            "simulator_generated_world_observation_only": True,
            "local_dry_render_seed": True,
            "source_frame_is_not_raw_capture_truth": True,
            "capture_truth": False,
            "physical_robot_sensor_proof": False,
            "static_reach_gate_passed": True,
            "manipulation_success_proven": False,
            "deployment_readiness_proven": False,
            "wam_visual_success_proven": False,
            "policy_ranking_claim_safe": False,
            "action_projection_bridge_ready_for_conditioning": bridge_readiness.get("status")
            == "ready",
        },
    }
    auxiliary = build_wam_auxiliary_observation_manifest(
        output_dir=export_dir / "wam_auxiliary_observation",
        source_image_path=preview_path.resolve(),
        policy_observation=observation,
        generated_at=generated,
        source_kind="local_kitchen_task_scaling_dry_render_seed",
        camera_id=str(geometry.get("camera") or "robot_pov"),
        robot_profile_id="unitree_g1",
        task_id=task_id,
        target_object_id=target_object_id,
        target_bbox=target_bbox,
        affordance_points=affordance_points,
        truth_overrides={
            "capture_truth": False,
            "geometry_truth": False,
            "camera_pose_truth": False,
            "proprioception_truth": False,
        },
    )
    aux_path = str(auxiliary.get("manifest_path") or "")
    observation["wam_auxiliary_observation_manifest_path"] = aux_path
    observation["wam_auxiliary_observation"] = {
        "manifest_path": aux_path,
        "modalities_available": auxiliary.get("modalities_available"),
        "truth": auxiliary.get("truth"),
    }
    visual["wam_auxiliary_observation_manifest_path"] = aux_path
    observation["visual_observation"] = visual
    observation_path = export_dir / "initial_policy_observation.json"
    _write_json(observation_path, {"observation": observation})
    export_manifest = {
        "schema_version": POLICY_OBSERVATION_EXPORT_SCHEMA_VERSION,
        "status": "completed",
        "generated_at": generated,
        "preflight_manifest_path": str(manifest_path),
        "task_id": task_id,
        "task_status": task_report.get("status"),
        "policy_observation_path": str(observation_path),
        "wam_auxiliary_observation_manifest_path": aux_path,
        "source_policy_observation_frame_path": str(preview_path.resolve()),
        "artifacts": {key: str(value.resolve()) for key, value in required_paths.items()},
        "target_object_id": target_object_id,
        "wam_seed_eligibility": wam_seed_eligibility,
        "action_projection_bridge_readiness": bridge_readiness,
        "next_step": (
            "render_review_quality_isaac_rgb_policy_observation_before_wam"
            if wam_seed_eligibility.get("status") != "eligible"
            else "run_source_policy_observation_visual_qa_before_bounded_wam_short_visual_sanity"
        ),
        "claim_boundary": (
            "This export adapts a passed local kitchen preflight into a WAM seed input. "
            "It does not claim raw capture truth, generated-video success, manipulation "
            "success, policy ranking safety, deployment approval, or physical robot readiness."
        ),
    }
    _write_json(export_dir / "kitchen_task_policy_observation_export_manifest.json", export_manifest)
    return export_manifest


def export_all_policy_observations_from_preflight(
    *,
    preflight_manifest_path: str | Path,
    out_dir: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Export every task in a preflight manifest and write a compact task index."""
    generated = generated_at or utc_now_iso()
    manifest_path = Path(preflight_manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_path)
    export_base = (
        Path(out_dir).expanduser().resolve()
        if out_dir
        else (manifest_path.parent / "wam_seed").resolve()
    )
    tasks: list[dict[str, Any]] = []
    for task_report in manifest.get("tasks") or []:
        if not isinstance(task_report, Mapping):
            continue
        task_id = str(task_report.get("task_id") or "").strip()
        if not task_id:
            continue
        task_export = export_policy_observation_from_preflight(
            preflight_manifest_path=manifest_path,
            task_id=task_id,
            out_dir=export_base / task_id,
            generated_at=generated,
        )
        tasks.append(
            {
                "task_id": task_id,
                "status": task_export.get("status"),
                "task_status": task_export.get("task_status"),
                "target_object_id": task_export.get("target_object_id"),
                "policy_observation_path": task_export.get("policy_observation_path"),
                "export_manifest_path": str(
                    export_base / task_id / "kitchen_task_policy_observation_export_manifest.json"
                ),
                "wam_seed_eligibility": task_export.get("wam_seed_eligibility"),
                "action_projection_bridge_readiness": task_export.get(
                    "action_projection_bridge_readiness"
                ),
                "next_step": task_export.get("next_step"),
                "blockers": task_export.get("blockers", []),
            }
        )
    blockers: list[str] = []
    if not tasks:
        blockers.append("no_tasks_available_for_policy_observation_export")
    if any(task.get("status") != "completed" for task in tasks):
        blockers.append("one_or_more_policy_observation_exports_blocked")
    index = {
        "schema_version": POLICY_OBSERVATION_EXPORT_INDEX_SCHEMA_VERSION,
        "status": "completed" if not blockers else "blocked",
        "generated_at": generated,
        "preflight_manifest_path": str(manifest_path),
        "export_base_dir": str(export_base),
        "tasks": tasks,
        "blockers": sorted(set(blockers)),
        "all_action_projection_bridges_ready": bool(tasks)
        and all(
            (
                (task.get("action_projection_bridge_readiness") or {}).get("status")
                == "ready"
            )
            for task in tasks
        ),
        "all_wam_seed_frames_review_quality_eligible": bool(tasks)
        and all(
            ((task.get("wam_seed_eligibility") or {}).get("status") == "eligible")
            for task in tasks
        ),
        "next_step": (
            "render_review_quality_isaac_rgb_policy_observations_before_wam"
            if tasks
            and any(
                ((task.get("wam_seed_eligibility") or {}).get("status") != "eligible")
                for task in tasks
            )
            else "run_source_policy_observation_visual_qa_before_bounded_wam_short_visual_sanity"
            if tasks
            else "fix_preflight_manifest_before_export"
        ),
        "claim_boundary": (
            "This index summarizes policy-observation seed exports from a local "
            "preflight manifest. It does not prove review-quality RGB, WAM success, "
            "SAM3/DA3 completion, manipulation success, ranking safety, deployment "
            "approval, or physical robot readiness."
        ),
    }
    _write_json(export_base / "kitchen_task_policy_observation_export_index.json", index)
    return index


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local no-spend kitchen task-scaling preflight")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--kitchen-usd", default=None)
    parser.add_argument("--g1-usd", default=None)
    parser.add_argument("--source-zip", default=None)
    parser.add_argument("--source-repo-root", default=None)
    parser.add_argument("--robot-review-material-override", action="store_true")
    parser.add_argument("--task", action="append", default=[], choices=[s["task_id"] for s in default_task_specs()])
    parser.add_argument("--min-scene-objects", type=int, default=MIN_FULL_KITCHEN_OBJECT_COUNT)
    parser.add_argument(
        "--export-policy-observation-from-manifest",
        default=None,
        help="export a passed task from an existing preflight manifest as a WAM policy observation",
    )
    parser.add_argument(
        "--export-task",
        default=None,
        choices=[s["task_id"] for s in default_task_specs()],
        help="task id to export when --export-policy-observation-from-manifest is provided",
    )
    parser.add_argument(
        "--export-all-tasks",
        action="store_true",
        help="export every task in --export-policy-observation-from-manifest",
    )
    parser.add_argument(
        "--export-out-dir",
        default=None,
        help="optional output directory for exported policy-observation artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.export_policy_observation_from_manifest:
        if args.export_all_tasks:
            export_index = export_all_policy_observations_from_preflight(
                preflight_manifest_path=args.export_policy_observation_from_manifest,
                out_dir=args.export_out_dir,
            )
            print(
                json.dumps(
                    {
                        "status": export_index.get("status"),
                        "export_index_path": str(
                            (
                                Path(args.export_out_dir).expanduser().resolve()
                                if args.export_out_dir
                                else Path(args.export_policy_observation_from_manifest)
                                .expanduser()
                                .resolve()
                                .parent
                                / "wam_seed"
                            )
                            / "kitchen_task_policy_observation_export_index.json"
                        ),
                        "task_count": len(export_index.get("tasks") or []),
                    }
                )
            )
            return 0 if export_index.get("status") == "completed" else 1
        if not args.export_task:
            raise SystemExit("--export-task is required with --export-policy-observation-from-manifest")
        export_manifest = export_policy_observation_from_preflight(
            preflight_manifest_path=args.export_policy_observation_from_manifest,
            task_id=args.export_task,
            out_dir=args.export_out_dir,
        )
        print(
            json.dumps(
                {
                    "status": export_manifest.get("status"),
                    "policy_observation_path": export_manifest.get("policy_observation_path"),
                    "export_manifest_path": str(
                        (
                            Path(args.export_out_dir).expanduser().resolve()
                            if args.export_out_dir
                            else Path(args.export_policy_observation_from_manifest)
                            .expanduser()
                            .resolve()
                            .parent
                            / "wam_seed"
                            / str(args.export_task)
                        )
                        / "kitchen_task_policy_observation_export_manifest.json"
                    ),
                }
            )
        )
        return 0 if export_manifest.get("status") == "completed" else 1
    if not args.out_dir:
        raise SystemExit("--out-dir is required unless exporting from an existing manifest")
    manifest = run_preflight(
        out_dir=Path(args.out_dir),
        kitchen_usd=resolve_kitchen_usd(args.kitchen_usd),
        g1_usd=args.g1_usd,
        source_zip=args.source_zip,
        source_repo_root=args.source_repo_root,
        robot_review_material_override=args.robot_review_material_override,
        task_ids=args.task,
        min_scene_objects=args.min_scene_objects,
    )
    print(json.dumps({"status": manifest.get("status"), "out_dir": str(Path(args.out_dir).resolve())}))
    return 0 if str(manifest.get("local_preflight_status")) == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
