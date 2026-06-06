"""Episode spec compiler for CPU-only pre-GPU robot-eval setup."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

from .common import PipelineError, ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context
from .scene_asset_preflight import build_scene_asset_preflight


EPISODE_SPEC_SCHEMA_VERSION = "episode_spec.v1"
EPISODE_SPEC_MANIFEST_SCHEMA_VERSION = "episode_spec_manifest.v1"
AGENT_EPISODE_PROPOSAL_SCHEMA_VERSION = "episode_spec_agent_proposals.v1"
TASK_ANCHOR_PROPOSAL_SCHEMA_VERSION = "task_anchor_proposal_manifest.v1"

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "episode_setup_specification_only",
    "deterministic_code_owns_proof_booleans": True,
    "agents_advisory_only": True,
    "live_provider_calls_performed": False,
    "remote_asset_downloads_performed": False,
    "gpu_required": False,
    "simulators_run": False,
    "simulator_execution_proven": False,
    "robot_readiness_proven": False,
    "robot_policy_execution_proven": False,
    "physics_contact_validated": False,
    "safety_validated": False,
    "public_claim_upgrade_allowed": False,
}

DEFAULT_ROBOT_PROFILES: List[Dict[str, Any]] = [
    {
        "robot_profile_id": "mobile_manipulator_rgbd_fixture",
        "label": "Mobile manipulator RGB-D fixture",
        "embodiment": "mobile_manipulator",
        "base_type": "wheeled",
        "sensors": ["rgb", "depth"],
        "source": "deterministic_default_profile",
    },
    {
        "robot_profile_id": "differential_drive_rgbd_fixture",
        "label": "Differential-drive RGB-D fixture",
        "embodiment": "differential_drive",
        "base_type": "wheeled",
        "sensors": ["rgb", "depth"],
        "source": "deterministic_default_profile",
    },
    {
        "robot_profile_id": "humanoid_rgbd_fixture",
        "label": "Humanoid RGB-D fixture",
        "embodiment": "humanoid",
        "base_type": "biped",
        "sensors": ["rgb", "depth"],
        "source": "deterministic_default_profile",
    },
]


class EpisodeSpecAgentAdapter(Protocol):
    def build_proposals(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class FakeEpisodeSpecAgentAdapter:
    """Network-free advisory adapter used in tests."""

    adapter_name: str = "fake"

    def build_proposals(self, *, plan_context: Mapping[str, Any]) -> Dict[str, Any]:
        tasks = [item for item in plan_context.get("tasks", []) if isinstance(item, Mapping)]
        proposals: List[Dict[str, Any]] = []
        for task in tasks:
            task_id = _string(task.get("task_id"))
            proposals.append(
                {
                    "proposal_id": f"proposal_spawn_{_stable_slug(task_id, fallback='task')}",
                    "task_id": task_id,
                    "field": "robot_spawn_pose",
                    "candidate_value": {"xyz": [0.0, 0.0, 0.05], "rpy": [0.0, 0.0, 0.0]},
                    "confidence": "low",
                    "provenance": {
                        "source": self.adapter_name,
                        "basis": "default floor-frame candidate; human/operator review required",
                    },
                    "proof_booleans_modified": False,
                }
            )
        return {
            "schema_version": AGENT_EPISODE_PROPOSAL_SCHEMA_VERSION,
            "adapter": self.adapter_name,
            "status": "completed",
            "agent_authority": "advisory_only",
            "proof_booleans_mutable_by_agent": False,
            "proposal_count": len(proposals),
            "proposals": proposals,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> List[str]:
    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    out: List[str] = []
    seen: set[str] = set()
    for item in values:
        text = _string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _stable_slug(value: Any, *, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", _string(value)).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"n_{text}"
    return text[:80]


def _sha_payload(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _float_list(value: Any, *, fallback: Sequence[float]) -> List[float]:
    out: List[float] = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value[:3]:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                out.append(float(fallback[len(out)]))
    while len(out) < 3:
        out.append(float(fallback[len(out)]))
    return out[:3]


def _source_artifacts(*, automation_dir: Path, pipeline_dir: Path) -> Dict[str, str]:
    candidates = {
        "scene_asset_inspection": automation_dir / "scene_asset_inspection.json",
        "scene_frame_estimate": automation_dir / "scene_frame_estimate.json",
        "cpu_preflight_scorecard": automation_dir / "cpu_preflight_scorecard.json",
        "task_anchor_proposal_manifest": automation_dir / "task_anchor_proposal_manifest.json",
        "episode_specs": automation_dir / "episode_specs.json",
        "scene_asset_inventory": automation_dir / "scene_asset_inventory.json",
        "scene_asset_dependency_audit": automation_dir / "scene_asset_dependency_audit.json",
        "collider_proxy_plan": automation_dir / "collider_proxy_plan.json",
        "cpu_scene_proxy_manifest": automation_dir / "cpu_scene_proxy_manifest.json",
        "task_anchor_manifest": pipeline_dir / "evaluation_prep" / "task_anchor_manifest.json",
        "site_world_spec": pipeline_dir / "evaluation_prep" / "site_world_spec.json",
        "hosted_session_runtime_manifest": (
            pipeline_dir / "evaluation_prep" / "hosted_session_runtime_manifest.json"
        ),
        "task_cards": pipeline_dir / "robot_eval_dataset" / "task_cards.json",
        "scenario_cards": pipeline_dir / "robot_eval_dataset" / "scenario_cards.json",
    }
    return {
        key: _relative_to(automation_dir, path)
        for key, path in candidates.items()
        if path.is_file()
    }


def _site_type_text(capture_root: Path) -> str:
    descriptor = _read_optional_mapping(capture_root / "capture_descriptor.json")
    raw_manifest = _read_optional_mapping(capture_root / "raw" / "manifest.json")
    metadata = _mapping(descriptor.get("metadata"))
    values = [
        descriptor.get("site_type"),
        raw_manifest.get("site_type"),
        metadata.get("site_type"),
        _mapping(metadata.get("site_identity")).get("site_type"),
        _mapping(raw_manifest.get("site_identity")).get("site_type"),
    ]
    return " ".join(_string(value) for value in values if _string(value)).lower()


def _scene_class_task_hints(site_text: str) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []
    mapping = [
        ("stockroom", "stockroom_bin_inspection", "Inspect labeled bins and staging shelves", "inspection_route"),
        ("warehouse", "warehouse_tote_transfer", "Move a tote between staging and shelf zones", "tote_transfer"),
        ("grocery", "grocery_backroom_shelf_check", "Check shelf/bin state in the backroom route", "shelf_bin_check"),
        ("kitchen", "kitchen_counter_navigation", "Navigate around counter, sink, and appliance zones", "navigation"),
        ("factory", "factory_line_side_delivery", "Deliver an item to a line-side fixture", "line_side_delivery"),
        ("lab", "lab_bench_inspection", "Inspect a bench-side target zone", "inspection_route"),
        ("hospital", "hospital_supply_delivery", "Deliver supplies through a constrained service route", "line_side_delivery"),
    ]
    for token, task_id, task_text, category in mapping:
        if token in site_text:
            hints.append(
                {
                    "task_id": task_id,
                    "task_text": task_text,
                    "task_category": category,
                    "source": "capture_site_type_hint",
                }
            )
    return hints


def _object_label_task_hints(pipeline_dir: Path) -> List[Dict[str, Any]]:
    manifest = _read_optional_mapping(pipeline_dir / "evaluation_prep" / "object_geometry_manifest.json")
    objects = manifest.get("objects")
    if not isinstance(objects, list):
        return []
    hints: List[Dict[str, Any]] = []
    for item in objects[:12]:
        if not isinstance(item, Mapping):
            continue
        object_id = _string(item.get("object_id") or item.get("id") or item.get("label"))
        label = _string(item.get("label") or object_id)
        if not object_id and not label:
            continue
        slug = _stable_slug(object_id or label, fallback="object")
        hints.append(
            {
                "task_id": f"inspect_{slug}",
                "task_text": f"Inspect or approach {label}",
                "task_category": "object_inspection",
                "target_object_ids": [object_id] if object_id else [],
                "source": "evaluation_prep/object_geometry_manifest.json",
            }
        )
    return hints


def _scene_asset_task_hints(automation_dir: Path) -> List[Dict[str, Any]]:
    inspection = _read_optional_mapping(automation_dir / "scene_asset_inspection.json")
    assets = inspection.get("assets")
    if not isinstance(assets, list):
        return []
    hints: List[Dict[str, Any]] = []
    for asset in assets:
        if not isinstance(asset, Mapping):
            continue
        for hint in asset.get("semantic_hints") or []:
            if not isinstance(hint, Mapping):
                continue
            label = _string(hint.get("label"))
            if not label:
                continue
            lower = label.lower()
            if any(token in lower for token in ("floor", "wall", "ceiling", "material")):
                continue
            slug = _stable_slug(label, fallback="scene_anchor")
            category = "pick_place" if any(token in lower for token in ("bin", "shelf", "counter", "table")) else "navigation"
            hints.append(
                {
                    "task_id": f"scene_anchor_{slug}",
                    "task_text": f"Review a robot task anchored near {label}",
                    "task_category": category,
                    "target_object_ids": [label],
                    "source": "scene_asset_semantic_hint",
                }
            )
    return hints[:12]


def _task_hypothesis_hints(capture_root: Path) -> List[Dict[str, Any]]:
    hypothesis = _read_optional_mapping(capture_root / "raw" / "task_hypothesis.json")
    candidates = hypothesis.get("tasks") or hypothesis.get("task_candidates")
    if not isinstance(candidates, list):
        return []
    hints: List[Dict[str, Any]] = []
    for index, item in enumerate(candidates[:12]):
        if isinstance(item, Mapping):
            task_text = _string(item.get("task_text") or item.get("task") or item.get("name"))
            task_id = _string(item.get("task_id") or item.get("id")) or f"task_hypothesis_{index + 1}"
            category = _string(item.get("task_category") or item.get("category") or "capture_hypothesis")
        else:
            task_text = _string(item)
            task_id = f"task_hypothesis_{index + 1}"
            category = "capture_hypothesis"
        if task_text:
            hints.append(
                {
                    "task_id": task_id,
                    "task_text": task_text,
                    "task_category": category,
                    "source": "raw/task_hypothesis.json",
                }
            )
    return hints


def _proposal_from_hint(hint: Mapping[str, Any], *, index: int) -> Dict[str, Any]:
    task_id = _stable_slug(hint.get("task_id"), fallback=f"task_{index + 1}")
    return {
        "proposal_id": f"task_anchor_proposal_{task_id}",
        "task_id": task_id,
        "task_text": _string(hint.get("task_text")) or "Review a scene-grounded robot task",
        "task_category": _string(hint.get("task_category")) or "scene_review",
        "start_zone": hint.get("start_zone"),
        "goal_zone": hint.get("goal_zone"),
        "target_object_ids": _string_list(hint.get("target_object_ids")),
        "source": _string(hint.get("source")) or "deterministic_hint",
        "review_required": True,
        "accepted": False,
        "confidence": "medium" if hint.get("source") != "deterministic_default" else "low",
        "proof_gaps": [
            "human_or_owner_acceptance_required",
            "metric_scale_proof_required",
            "collision_proof_required",
            "owner_system_simulator_execution_not_run",
        ],
        "claim_boundary": "task_anchor_proposal_is_advisory_not_robot_eval_proof",
    }


def build_task_anchor_proposals(
    *,
    capture_root: Path,
    pipeline_dir: Path,
    automation_dir: Path,
    generated_at: str,
) -> Dict[str, Any]:
    hints: List[Dict[str, Any]] = []
    hints.extend(_task_hypothesis_hints(capture_root))
    hints.extend(_object_label_task_hints(pipeline_dir))
    hints.extend(_scene_asset_task_hints(automation_dir))
    hints.extend(_scene_class_task_hints(_site_type_text(capture_root)))
    if not hints:
        hints.append(
            {
                "task_id": "default_site_navigation_review",
                "task_text": "Review a default site navigation episode candidate",
                "task_category": "navigation",
                "source": "deterministic_default",
            }
        )
    proposals: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for hint in hints:
        proposal = _proposal_from_hint(hint, index=len(proposals))
        if proposal["task_id"] in seen:
            continue
        seen.add(proposal["task_id"])
        proposals.append(proposal)
    manifest = {
        "schema_version": TASK_ANCHOR_PROPOSAL_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "compiled_review_required",
        "proposal_count": len(proposals),
        "proposals": proposals,
        "advisory_only": True,
        "review_required": True,
        "proof_booleans_mutable_by_proposals": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    manifest["deterministic_fingerprint"] = _sha_payload(
        {"proposals": proposals, "status": manifest["status"]}
    )
    write_json(automation_dir / "task_anchor_proposal_manifest.json", manifest)
    return manifest


def _tasks_from_proposals(automation_dir: Path) -> List[Dict[str, Any]]:
    manifest = _read_optional_mapping(automation_dir / "task_anchor_proposal_manifest.json")
    proposals = manifest.get("proposals")
    if not isinstance(proposals, list):
        return []
    tasks: List[Dict[str, Any]] = []
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, Mapping):
            continue
        task_id = _string(proposal.get("task_id") or f"proposal_task_{index}")
        tasks.append(
            {
                "task_id": task_id,
                "task_text": _string(proposal.get("task_text") or task_id),
                "task_category": _string(proposal.get("task_category") or "scene_review"),
                "start_zone": proposal.get("start_zone"),
                "goal_zone": proposal.get("goal_zone"),
                "target_object_ids": _string_list(proposal.get("target_object_ids")),
                "anchor_source": "simulation_automation/task_anchor_proposal_manifest.json",
                "anchor_accepted": False,
                "proposal_id": proposal.get("proposal_id"),
            }
        )
    return tasks


def _load_tasks(pipeline_dir: Path, automation_dir: Path) -> List[Dict[str, Any]]:
    task_anchor = _read_optional_mapping(pipeline_dir / "evaluation_prep" / "task_anchor_manifest.json")
    raw_tasks = task_anchor.get("tasks")
    if isinstance(raw_tasks, list) and raw_tasks:
        tasks: List[Dict[str, Any]] = []
        for index, task in enumerate(raw_tasks):
            if not isinstance(task, Mapping):
                continue
            task_id = _string(task.get("task_id") or task.get("id") or f"task_{index}")
            tasks.append(
                {
                    "task_id": task_id,
                    "task_text": _string(task.get("task_text") or task.get("name") or task_id),
                    "task_category": _string(task.get("task_category") or "generic"),
                    "start_zone": task.get("start_zone"),
                    "goal_zone": task.get("goal_zone"),
                    "target_object_ids": _string_list(task.get("target_object_ids")),
                    "anchor_source": "evaluation_prep/task_anchor_manifest.json",
                    "anchor_accepted": bool(task.get("anchor_accepted") or task.get("accepted")),
                }
            )
        if tasks:
            return tasks
    task_cards = _read_optional_mapping(pipeline_dir / "robot_eval_dataset" / "task_cards.json")
    cards = task_cards.get("cards")
    if isinstance(cards, list) and cards:
        return [
            {
                "task_id": _string(card.get("task_id") or f"task_{index}"),
                "task_text": _string(card.get("task_statement") or card.get("task_id")),
                "task_category": _string(card.get("task_category") or "generic"),
                "start_zone": _mapping(card.get("start_state")).get("start_zone"),
                "goal_zone": None,
                "target_object_ids": [],
                "anchor_source": "robot_eval_dataset/task_cards.json",
                "anchor_accepted": False,
            }
            for index, card in enumerate(cards)
            if isinstance(card, Mapping)
        ]
    proposed_tasks = _tasks_from_proposals(automation_dir)
    if proposed_tasks:
        return proposed_tasks
    return [
        {
            "task_id": "default_site_navigation_review",
            "task_text": "Review a default site navigation episode candidate",
            "task_category": "navigation",
            "start_zone": None,
            "goal_zone": None,
            "target_object_ids": [],
            "anchor_source": "deterministic_default",
            "anchor_accepted": False,
        }
    ]


def _load_scenarios(pipeline_dir: Path, tasks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    scenario_cards = _read_optional_mapping(pipeline_dir / "robot_eval_dataset" / "scenario_cards.json")
    cards = scenario_cards.get("cards")
    if isinstance(cards, list) and cards:
        return [
            {
                "scenario_id": _string(card.get("scenario_id") or f"scenario_{index}"),
                "task_id": _string(card.get("task_id")),
                "robot_profile_id": _string(card.get("robot_profile_id")),
                "scenario_source": "robot_eval_dataset/scenario_cards.json",
            }
            for index, card in enumerate(cards)
            if isinstance(card, Mapping)
        ]
    return [
        {
            "scenario_id": f"scenario_{_stable_slug(task.get('task_id'), fallback='task')}_capture_observed",
            "task_id": _string(task.get("task_id")),
            "robot_profile_id": "",
            "scenario_source": "deterministic_default_capture_observed_layout",
        }
        for task in tasks
    ]


def _robot_profiles(pipeline_dir: Path) -> tuple[List[Dict[str, Any]], bool]:
    site_world = _read_optional_mapping(pipeline_dir / "evaluation_prep" / "site_world_spec.json")
    hosted = _read_optional_mapping(
        pipeline_dir / "evaluation_prep" / "hosted_session_runtime_manifest.json"
    )
    raw = site_world.get("robot_profiles")
    if not isinstance(raw, list) or not raw:
        raw = hosted.get("robot_profiles")
    profiles: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                continue
            profile_id = _string(
                item.get("robot_profile_id")
                or item.get("id")
                or item.get("profile_id")
                or f"robot_profile_{index}"
            )
            profiles.append(
                {
                    **dict(item),
                    "robot_profile_id": profile_id,
                    "source": "site_world_or_hosted_session_runtime_manifest",
                }
            )
    return (profiles, True) if profiles else ([dict(item) for item in DEFAULT_ROBOT_PROFILES], False)


def _frame_payload(automation_dir: Path) -> Dict[str, Any]:
    frame_manifest = _read_optional_mapping(automation_dir / "scene_frame_estimate.json")
    frame = _mapping(frame_manifest.get("frame"))
    return {
        "manifest": frame_manifest,
        "frame": frame,
        "bounds": _mapping(frame.get("bounds")),
        "floor_z": frame.get("floor_z_estimate"),
        "confidence": _string(frame.get("confidence") or "low"),
        "scale_proven": False,
    }


def _region_from_bounds(bounds: Mapping[str, Any]) -> Dict[str, Any]:
    low = _float_list(bounds.get("min"), fallback=(-1.0, -1.0, 0.0))
    high = _float_list(bounds.get("max"), fallback=(1.0, 1.0, 1.0))
    return {
        "type": "axis_aligned_box",
        "min_xyz": low,
        "max_xyz": high,
        "source": "scene_frame_estimate",
        "confidence": "estimated",
    }


def _pose_from_zone(value: Any, *, fallback: Sequence[float]) -> Dict[str, Any]:
    xyz = _float_list(value, fallback=fallback)
    return {"xyz": xyz, "rpy": [0.0, 0.0, 0.0], "source": "task_anchor_or_frame_default"}


def _missing_proof_labels(
    *,
    task: Mapping[str, Any],
    robot_profile_from_request: bool,
    frame: Mapping[str, Any],
    scorecard: Mapping[str, Any],
) -> List[str]:
    missing = ["simulator_execution_not_run"]
    if not task.get("anchor_accepted"):
        missing.append("accepted_task_anchor_required")
    if not robot_profile_from_request:
        missing.append("robot_team_profile_required")
    if not _mapping(frame).get("scale_proven"):
        missing.append("metric_scale_proof_required")
    if not bool(scorecard.get("isaac_usd_collision_verified")):
        missing.append("collision_proof_required")
    if bool(scorecard.get("portable_collider_glb_missing")):
        missing.append("portable_collider_glb_missing")
    return list(dict.fromkeys(missing))


def _default_agent_proposals(*, generated_at: str) -> Dict[str, Any]:
    return {
        "schema_version": AGENT_EPISODE_PROPOSAL_SCHEMA_VERSION,
        "generated_at": generated_at,
        "adapter": "none",
        "status": "not_requested",
        "agent_authority": "advisory_only",
        "proof_booleans_mutable_by_agent": False,
        "proposal_count": 0,
        "proposals": [],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def build_episode_specs(
    *,
    capture_root: str | Path,
    agent_adapter: EpisodeSpecAgentAdapter | None = None,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    automation_dir = pipeline_dir / "simulation_automation"
    ensure_dir(automation_dir)
    if not (automation_dir / "scene_frame_estimate.json").is_file():
        build_scene_asset_preflight(capture_root=context.capture_root)
    generated_at = utc_now_iso()
    frame = _frame_payload(automation_dir)
    scorecard = _read_optional_mapping(automation_dir / "cpu_preflight_scorecard.json")
    task_anchor_proposals = build_task_anchor_proposals(
        capture_root=context.capture_root,
        pipeline_dir=pipeline_dir,
        automation_dir=automation_dir,
        generated_at=generated_at,
    )
    tasks = _load_tasks(pipeline_dir, automation_dir)
    scenarios = _load_scenarios(pipeline_dir, tasks)
    profiles, profile_from_request = _robot_profiles(pipeline_dir)
    proposals = (
        agent_adapter.build_proposals(
            plan_context={
                "capture_root": str(context.capture_root),
                "tasks": tasks,
                "scenarios": scenarios,
                "robot_profiles": profiles,
                "frame": frame,
            }
        )
        if agent_adapter is not None
        else _default_agent_proposals(generated_at=generated_at)
    )
    proposals.setdefault("generated_at", generated_at)
    proposals.setdefault("claim_boundary", dict(CLAIM_BOUNDARY))
    proposals["proof_booleans_mutable_by_agent"] = False

    bounds = _mapping(frame.get("bounds"))
    motion_region = _region_from_bounds(bounds)
    floor_z = frame.get("floor_z")
    try:
        floor_z_float = float(floor_z)
    except (TypeError, ValueError):
        floor_z_float = 0.0
    episodes: List[Dict[str, Any]] = []
    for task in tasks:
        scenario_matches = [
            scenario
            for scenario in scenarios
            if not scenario.get("task_id") or scenario.get("task_id") == task.get("task_id")
        ] or scenarios[:1]
        for scenario in scenario_matches:
            target_profiles = [
                profile
                for profile in profiles
                if not scenario.get("robot_profile_id")
                or scenario.get("robot_profile_id") == profile.get("robot_profile_id")
            ] or profiles
            for profile in target_profiles:
                task_id = _string(task.get("task_id"))
                scenario_id = _string(scenario.get("scenario_id"))
                profile_id = _string(profile.get("robot_profile_id"))
                spawn = _pose_from_zone(
                    task.get("start_zone"),
                    fallback=[0.0, 0.0, floor_z_float + 0.05],
                )
                goal = _pose_from_zone(
                    task.get("goal_zone"),
                    fallback=list(_mapping(motion_region).get("max_xyz") or [1.0, 1.0, floor_z_float]),
                )
                missing = _missing_proof_labels(
                    task=task,
                    robot_profile_from_request=profile_from_request,
                    frame=frame,
                    scorecard=scorecard,
                )
                confidence = (
                    "review_ready_with_accepted_inputs"
                    if not missing
                    else "estimated_review_required"
                )
                episodes.append(
                    {
                        "episode_id": (
                            f"episode_{_stable_slug(task_id, fallback='task')}_"
                            f"{_stable_slug(scenario_id, fallback='scenario')}_"
                            f"{_stable_slug(profile_id, fallback='robot')}"
                        ),
                        "task_id": task_id,
                        "scenario_id": scenario_id,
                        "robot_profile_id": profile_id,
                        "robot_profile": profile,
                        "robot_spawn_pose": spawn,
                        "robot_initial_heading": {
                            "yaw_radians": 0.0,
                            "source": "deterministic_default_until_anchor_review",
                        },
                        "camera_pose": {
                            "xyz": [spawn["xyz"][0], spawn["xyz"][1], spawn["xyz"][2] + 1.2],
                            "rpy": [0.0, 0.0, 0.0],
                            "source": "robot_spawn_pose_rgbd_fixture_default",
                        },
                        "target_region": {
                            **_region_from_bounds(
                                {"min": goal["xyz"], "max": goal["xyz"]}
                            ),
                            "source": "task_goal_zone_or_frame_default",
                        },
                        "reset_conditions": [
                            "reset_robot_to_spawn_pose",
                            "reset_dynamic_objects_to_capture_observed_or_review_state",
                            "do_not_claim_contact_or_safety_validation_without_owner_logs",
                        ],
                        "allowed_motion_region": motion_region,
                        "collision_check_required": True,
                        "simulator_backend_preferences": [
                            {
                                "backend": "pybullet",
                                "mode": "DIRECT_cpu_proxy_preflight",
                                "proof_status": "optional_local_smoke_only",
                            },
                            {
                                "backend": "mujoco",
                                "mode": "CPU_compile_step_preflight",
                                "proof_status": "optional_local_smoke_only",
                            },
                            {
                                "backend": "isaac_sim",
                                "mode": "future_gpu_or_owner_system_review",
                                "proof_status": "not_run",
                            },
                        ],
                        "confidence": confidence,
                        "review_required": bool(missing),
                        "provenance": {
                            "task_source": task.get("anchor_source"),
                            "scenario_source": scenario.get("scenario_source"),
                            "frame_source": _mapping(frame.get("frame")).get("source_asset"),
                            "agent_proposals_path": "agent_episode_spec_proposals.json",
                        },
                        "missing_proof_labels": missing,
                        "proof_booleans": {
                            "simulator_execution_proven": False,
                            "robot_readiness_proven": False,
                            "physics_contact_validated": False,
                            "safety_validated": False,
                        },
                    }
                )
    episode_spec = {
        "schema_version": EPISODE_SPEC_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "compiled_review_required"
        if any(item.get("review_required") for item in episodes)
        else "compiled",
        "episode_count": len(episodes),
        "episodes": sorted(episodes, key=lambda item: item["episode_id"]),
        "default_robot_profiles_used": not profile_from_request,
        "agent_proposals_path": "agent_episode_spec_proposals.json",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    manifest = {
        "schema_version": EPISODE_SPEC_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": episode_spec["status"],
        "episode_spec_path": "episode_spec.v1.json",
        "episode_specs_path": "episode_specs.json",
        "task_anchor_proposal_manifest_path": "task_anchor_proposal_manifest.json",
        "agent_episode_spec_proposals_path": "agent_episode_spec_proposals.json",
        "episode_count": len(episodes),
        "task_anchor_proposal_count": task_anchor_proposals.get("proposal_count"),
        "default_robot_profiles_used": not profile_from_request,
        "source_artifacts": _source_artifacts(automation_dir=automation_dir, pipeline_dir=pipeline_dir),
        "missing_proof_labels": sorted(
            {
                label
                for episode in episodes
                for label in _string_list(episode.get("missing_proof_labels"))
            }
        ),
        "webapp_display": {
            "display_as": "episode_setup_review_required",
            "must_not_display_as": [
                "robot_ready",
                "deployment_ready",
                "simulator_completed",
                "safety_validated",
            ],
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    episode_spec["deterministic_fingerprint"] = _sha_payload(
        {"episodes": episodes, "default_robot_profiles_used": not profile_from_request}
    )
    manifest["deterministic_fingerprint"] = _sha_payload(
        {"episode_spec": episode_spec, "source_artifacts": manifest["source_artifacts"]}
    )
    write_json(automation_dir / "agent_episode_spec_proposals.json", proposals)
    write_json(automation_dir / "episode_spec.v1.json", episode_spec)
    write_json(automation_dir / "episode_specs.json", episode_spec)
    write_json(automation_dir / "episode_spec_manifest.json", manifest)
    return {
        "schema_version": "episode_spec_result.v1",
        "capture_root": str(context.capture_root),
        "automation_dir": str(automation_dir),
        "status": manifest["status"],
        "episode_spec_path": str((automation_dir / "episode_spec.v1.json").resolve()),
        "episode_specs_path": str((automation_dir / "episode_specs.json").resolve()),
        "episode_spec_manifest_path": str((automation_dir / "episode_spec_manifest.json").resolve()),
        "task_anchor_proposal_manifest_path": str(
            (automation_dir / "task_anchor_proposal_manifest.json").resolve()
        ),
        "agent_episode_spec_proposals_path": str(
            (automation_dir / "agent_episode_spec_proposals.json").resolve()
        ),
        "episode_count": len(episodes),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _agent_adapter_from_mode(mode: str) -> EpisodeSpecAgentAdapter | None:
    if mode == "fake":
        return FakeEpisodeSpecAgentAdapter()
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile episode_spec.v1 manifests from scene/task/scenario/robot profile inputs"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument(
        "--agent-mode",
        choices=("none", "fake"),
        default="none",
        help="Optional advisory proposal adapter; deterministic code owns proof booleans",
    )
    args = parser.parse_args(argv)
    try:
        result = build_episode_specs(
            capture_root=args.capture_root,
            agent_adapter=_agent_adapter_from_mode(args.agent_mode),
        )
    except (OSError, ValueError, PipelineError) as exc:
        print(f"[episode-spec] FAILED: {exc}")
        return 1
    print(f"[episode-spec] spec={result['episode_spec_path']}")
    print(f"[episode-spec] status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
