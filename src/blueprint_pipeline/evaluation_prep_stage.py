"""Scene-aware downstream evaluation prep artifacts for qualified captures."""

from __future__ import annotations

import argparse
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .common import PipelineError, ensure_dir, optional_read_json, read_json_any, relative_scene_path, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context
from .object_geometry_stage import resolve_object_geometry_manifest
from .runtime_layer_grounding import (
    build_canonical_render_policy,
    build_presentation_variance_policy,
    build_protected_regions_manifest,
    compute_canonical_package_version,
    task_critical_object_ids,
    with_grounding_fields,
)
from .site_world_runtime_service_client import SiteWorldRuntimeServiceClient, SiteWorldRuntimeServiceConfig
from .world_model_policy import (
    WorldModelPolicy,
    build_output_linkage,
    build_presentation_derivation_policy,
    build_provenance_record,
)


def _read_optional_json_any(path: Path) -> Any:
    if not path.is_file():
        return None
    return read_json_any(path)


def _string_list(*values: Any) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, (list, tuple, set)):
            items = [str(item) for item in value]
        elif value is None:
            items = []
        else:
            items = [str(value)]
        for item in items:
            text = item.strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def _stable_id(prefix: str, value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    token = normalized or fallback
    return f"{prefix}_{token}"


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _copy_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def _resolve_object_geometry_manifest(*, context, provider_name: str) -> Dict[str, Any]:
    return resolve_object_geometry_manifest(
        capture_root=context.capture_root,
        provider_name=provider_name,
    )


def _adapter_manifest_details(scene_memory_bundle_manifest: Mapping[str, Any], *, eval_dir: Path) -> Dict[str, Dict[str, Any]]:
    key_map = {
        "neoverse": "neoverse_adapter_manifest_path",
        "gen3c": "gen3c_adapter_manifest_path",
        "cosmos_transfer": "cosmos_transfer_adapter_manifest_path",
    }
    details: Dict[str, Dict[str, Any]] = {}
    for backend, key in key_map.items():
        rel_path = str(scene_memory_bundle_manifest.get(key) or "").strip()
        if not rel_path:
            continue
        payload = optional_read_json(eval_dir / rel_path)
        details[backend] = dict(payload) if isinstance(payload, Mapping) else {}
    return details


def _task_category(task_text: str) -> str:
    lowered = task_text.strip().lower()
    if "open and close" in lowered or lowered.startswith("open "):
        return "open_close"
    if "navigate to" in lowered or lowered.startswith("navigate "):
        return "navigate"
    if "pick up" in lowered or "place" in lowered:
        return "pick"
    return "generic"


def _default_task_id(scope_record: Mapping[str, Any], handoff: Mapping[str, Any], capture_id: str) -> str:
    scoped = handoff.get("scoped_task_definition") if isinstance(handoff.get("scoped_task_definition"), Mapping) else {}
    task_id = str(scoped.get("task_id") or "").strip()
    if task_id:
        return task_id
    tasks = scope_record.get("tasks") if isinstance(scope_record.get("tasks"), list) else []
    if tasks and isinstance(tasks[0], Mapping):
        task_id = str(tasks[0].get("task_id") or "").strip()
        if task_id:
            return task_id
    return capture_id


def _default_task_text(scope_record: Mapping[str, Any], handoff: Mapping[str, Any], capture_id: str) -> str:
    scoped = handoff.get("scoped_task_definition") if isinstance(handoff.get("scoped_task_definition"), Mapping) else {}
    task_text = str(scoped.get("scoped_task_statement") or "").strip()
    if task_text:
        return task_text
    task_text = str(scope_record.get("task_statement") or "").strip()
    if task_text:
        return task_text
    return capture_id


def _default_robot_profiles() -> List[Dict[str, Any]]:
    return [
        {
            "id": "mobile_manipulator_rgb_v1",
            "display_name": "Mobile manipulator",
            "embodiment_type": "mobile_manipulator",
            "action_space": {
                "name": "ee_delta_pose_gripper",
                "dim": 7,
                "labels": [
                    "base_x",
                    "base_y",
                    "base_yaw",
                    "ee_x",
                    "ee_y",
                    "ee_z",
                    "gripper",
                ],
            },
            "observation_cameras": [
                {"id": "head_rgb", "role": "head", "required": True, "default_enabled": True},
                {"id": "wrist_rgb", "role": "wrist", "required": False, "default_enabled": True},
                {"id": "site_context_rgb", "role": "context", "required": False, "default_enabled": True},
            ],
            "base_semantics": "holonomic_mobile_base",
            "gripper_semantics": "parallel_jaw_gripper",
            "urdf_uri": None,
            "usd_uri": None,
            "allowed_policy_adapters": ["openvla_oft", "pi05", "dreamzero"],
            "default_policy_adapter": "openvla_oft",
        },
        {
            "id": "humanoid_dual_camera_v1",
            "display_name": "Humanoid",
            "embodiment_type": "humanoid",
            "action_space": {
                "name": "whole_body_delta_pose_gripper",
                "dim": 7,
                "labels": [
                    "body_x",
                    "body_y",
                    "body_yaw",
                    "hand_x",
                    "hand_y",
                    "hand_z",
                    "gripper",
                ],
            },
            "observation_cameras": [
                {"id": "head_rgb", "role": "head", "required": True, "default_enabled": True},
                {"id": "left_wrist_rgb", "role": "wrist_left", "required": False, "default_enabled": True},
                {"id": "right_wrist_rgb", "role": "wrist_right", "required": False, "default_enabled": True},
                {"id": "site_context_rgb", "role": "context", "required": False, "default_enabled": True},
            ],
            "base_semantics": "bipedal_base",
            "gripper_semantics": "multi_finger_gripper",
            "urdf_uri": None,
            "usd_uri": None,
            "allowed_policy_adapters": ["openvla_oft", "dreamzero"],
            "default_policy_adapter": "openvla_oft",
        },
        {
            "id": "fixed_arm_cell_v1",
            "display_name": "Fixed arm cell",
            "embodiment_type": "fixed_arm",
            "action_space": {
                "name": "joint_delta_gripper",
                "dim": 7,
                "labels": [
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "gripper",
                ],
            },
            "observation_cameras": [
                {"id": "cell_rgb", "role": "head", "required": True, "default_enabled": True},
                {"id": "wrist_rgb", "role": "wrist", "required": False, "default_enabled": True},
                {"id": "site_context_rgb", "role": "context", "required": False, "default_enabled": False},
            ],
            "base_semantics": "fixed_base",
            "gripper_semantics": "parallel_jaw_gripper",
            "urdf_uri": None,
            "usd_uri": None,
            "allowed_policy_adapters": ["openvla_oft", "pi05"],
            "default_policy_adapter": "pi05",
        },
    ]


def _load_task_run_entries(capture_root: Path) -> List[Dict[str, Any]]:
    pipeline_dir = capture_root / "pipeline"
    manifest = _read_optional_json_any(pipeline_dir / "task_run_manifest.json")
    if not isinstance(manifest, Mapping):
        return []
    groups = manifest.get("groups") if isinstance(manifest.get("groups"), Mapping) else {}
    entries: List[Dict[str, Any]] = []
    for category, items in groups.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            task_capture_root = Path(str(item.get("capture_root") or "")).resolve()
            task_scope = _read_optional_json_any(task_capture_root / "pipeline" / "task_scope_record.json")
            task_scope = dict(task_scope) if isinstance(task_scope, Mapping) else {}
            entries.append(
                {
                    "task_id": _default_task_id(task_scope, {}, str(item.get("capture_id") or "")),
                    "task_text": str(item.get("task_text") or ""),
                    "task_category": str(category),
                    "capture_root": str(task_capture_root),
                    "capture_id": str(item.get("capture_id") or ""),
                    "target_object_ids": _string_list(task_scope.get("target_object_ids")),
                    "articulation_required_ids": _string_list(task_scope.get("articulation_required_ids")),
                }
            )
    return entries


def _build_default_task_run_manifest(
    *,
    capture_root: Path,
    handoff: Mapping[str, Any],
    scope_record: Mapping[str, Any],
) -> Dict[str, Any]:
    task_text = _default_task_text(scope_record, handoff, capture_root.name)
    return {
        "schema_version": "v1",
        "scene_id": capture_root.parts[-3],
        "base_capture_id": capture_root.name,
        "generated_at": utc_now_iso(),
        "source_dir": str(capture_root),
        "groups": {
            _task_category(task_text): [
                {
                    "task_text": task_text,
                    "capture_root": str(capture_root),
                    "capture_id": capture_root.name,
                    "final_bundle_path": str(capture_root / "pipeline" / "agent_review_bundle.json"),
                    "final_memo_path": str(capture_root / "pipeline" / "agent_readiness_memo.md"),
                }
            ]
        },
    }


def _build_task_anchor_manifest(
    *,
    capture_root: Path,
    handoff: Mapping[str, Any],
    scope_record: Mapping[str, Any],
    task_run_entries: Sequence[Mapping[str, Any]],
    object_geometry_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    geometry_objects = object_geometry_manifest.get("objects") if isinstance(object_geometry_manifest.get("objects"), list) else []
    geometry_by_id = {
        str(item.get("object_id") or ""): item
        for item in geometry_objects
        if isinstance(item, Mapping) and str(item.get("object_id") or "")
    }

    def _zone_center(ids: Sequence[str]) -> List[float]:
        centers: List[List[float]] = []
        for object_id in ids:
            geometry = geometry_by_id.get(str(object_id))
            bbox = geometry.get("placement_bbox") if isinstance(geometry, Mapping) and isinstance(geometry.get("placement_bbox"), Mapping) else {}
            center = bbox.get("center") if isinstance(bbox.get("center"), list) else None
            if isinstance(center, list) and len(center) >= 3:
                centers.append([float(center[0]), float(center[1]), float(center[2])])
        if not centers:
            task_zone = scope_record.get("task_zone") if isinstance(scope_record.get("task_zone"), Mapping) else {}
            center = task_zone.get("center") if isinstance(task_zone.get("center"), list) else None
            if isinstance(center, list) and len(center) >= 3:
                return [float(center[0]), float(center[1]), float(center[2])]
            return [0.0, 0.0, 0.0]
        return [
            round(sum(center[idx] for center in centers) / float(len(centers)), 6)
            for idx in range(3)
        ]

    tasks: List[Dict[str, Any]] = []
    if task_run_entries:
        for item in task_run_entries:
            target_ids = _string_list(item.get("target_object_ids"))
            articulation_ids = _string_list(item.get("articulation_required_ids"))
            goal = _zone_center(target_ids or articulation_ids)
            provenance = build_provenance_record(
                grounding_level="reconstructed" if target_ids else "inferred",
                evidence_sources=[str(item.get("capture_root") or ""), str(capture_root / "pipeline" / "task_scope_record.json")],
                observation_coverage={
                    "target_object_count": len(target_ids),
                    "articulation_required_count": len(articulation_ids),
                },
                confidence=1.0 if target_ids else 0.5,
                canonical_truth=True,
                presentation_only=False,
            )
            tasks.append(
                with_grounding_fields(
                    {
                    "task_id": str(item.get("task_id") or item.get("capture_id") or ""),
                    "task_text": str(item.get("task_text") or ""),
                    "task_category": str(item.get("task_category") or "generic"),
                    "capture_root": str(item.get("capture_root") or ""),
                    "capture_id": str(item.get("capture_id") or ""),
                    "target_object_ids": target_ids,
                    "articulation_required_ids": articulation_ids,
                    "task_critical": bool(target_ids or articulation_ids),
                    "scene_relative_transforms": {
                        object_id: (
                            dict(geometry_by_id[object_id].get("placement_bbox") or {})
                            if object_id in geometry_by_id
                            else {}
                        )
                        for object_id in list(target_ids) + list(articulation_ids)
                    },
                    "task_zone": {"center": goal},
                    "start_zone": [round(goal[0] - 1.0, 6), round(goal[1], 6), round(goal[2], 6)],
                    "goal_zone": goal,
                    "provenance": provenance,
                    },
                    provenance=provenance,
                )
            )
    else:
        target_ids = _string_list(scope_record.get("target_object_ids"))
        articulation_ids = _string_list(scope_record.get("articulation_required_ids"))
        task_text = _default_task_text(scope_record, handoff, capture_root.name)
        goal = _zone_center(target_ids or articulation_ids)
        provenance = build_provenance_record(
            grounding_level="reconstructed" if target_ids else "inferred",
            evidence_sources=[str(capture_root / "pipeline" / "task_scope_record.json")],
            observation_coverage={
                "target_object_count": len(target_ids),
                "articulation_required_count": len(articulation_ids),
            },
            confidence=1.0 if target_ids else 0.5,
            canonical_truth=True,
            presentation_only=False,
        )
        tasks.append(
            with_grounding_fields(
                {
                "task_id": _default_task_id(scope_record, handoff, capture_root.name),
                "task_text": task_text,
                "task_category": _task_category(task_text),
                "capture_root": str(capture_root),
                "capture_id": capture_root.name,
                "target_object_ids": target_ids,
                "articulation_required_ids": articulation_ids,
                "task_critical": bool(target_ids or articulation_ids),
                "scene_relative_transforms": {
                    object_id: (
                        dict(geometry_by_id[object_id].get("placement_bbox") or {})
                        if object_id in geometry_by_id
                        else {}
                    )
                    for object_id in list(target_ids) + list(articulation_ids)
                },
                "task_zone": dict(scope_record.get("task_zone") or {}) if isinstance(scope_record.get("task_zone"), Mapping) else {"center": goal},
                "start_zone": [round(goal[0] - 1.0, 6), round(goal[1], 6), round(goal[2], 6)],
                "goal_zone": goal,
                "provenance": provenance,
                },
                provenance=provenance,
            )
        )

    manifest_provenance = build_provenance_record(
        grounding_level="reconstructed" if tasks else "inferred",
        evidence_sources=[str(capture_root / "pipeline" / "task_scope_record.json")],
        observation_coverage={"task_count": len(tasks)},
        confidence=1.0 if tasks else 0.0,
        canonical_truth=True,
        presentation_only=False,
    )
    return with_grounding_fields({
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "scene_id": capture_root.parts[-3],
        "capture_id": capture_root.name,
        "tasks": tasks,
        "provenance": manifest_provenance,
    }, provenance=manifest_provenance)


def _build_geometry_bundle_manifest(*, pipeline_dir: Path, eval_dir: Path) -> Dict[str, Any]:
    advanced_dir = pipeline_dir / "advanced_geometry"
    files = {
        "bundle_path": advanced_dir,
        "ply_path": advanced_dir / "3dgs_compressed.ply",
        "labels_path": advanced_dir / "labels.json",
        "structure_path": advanced_dir / "structure.json",
        "task_hints_path": advanced_dir / "task_targets.synthetic.json",
        "holi_spatial_grounding_path": advanced_dir / "holi_spatial_grounding.json",
    }
    entries: Dict[str, str] = {}
    available = 0
    for key, path in files.items():
        if key == "bundle_path":
            if path.is_dir():
                entries[key] = _relative_to(eval_dir, path)
                available += 1
            continue
        if path.is_file():
            entries[key] = _relative_to(eval_dir, path)
            available += 1
    status = "complete" if {"ply_path", "labels_path", "structure_path", "task_hints_path"}.issubset(entries) else "partial" if available > 0 else "missing"
    missing = [key for key in ("ply_path", "labels_path", "structure_path", "task_hints_path") if key not in entries]
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": status,
        "missing_required_fields": missing,
        "available_fields": sorted(entries.keys()),
        **entries,
    }


def _build_scene_memory_bundle_manifest(*, pipeline_dir: Path, eval_dir: Path) -> Dict[str, Any]:
    scene_memory_dir = pipeline_dir / "scene_memory"
    adapter_dir = scene_memory_dir / "adapter_manifests"
    preview_dir = pipeline_dir / "preview_simulation"
    presentation_dir = pipeline_dir / "presentation_world"
    geometry_dir = pipeline_dir / "geometry"
    files = {
        "bundle_path": scene_memory_dir,
        "scene_memory_manifest_path": scene_memory_dir / "scene_memory_manifest.json",
        "scene_memory_readiness_path": scene_memory_dir / "scene_memory_readiness.json",
        "conditioning_bundle_path": scene_memory_dir / "conditioning_bundle.json",
        "preview_simulation_manifest_path": preview_dir / "preview_simulation_manifest.json",
        "gen3c_adapter_manifest_path": adapter_dir / "gen3c.json",
        "neoverse_adapter_manifest_path": adapter_dir / "neoverse.json",
        "cosmos_transfer_adapter_manifest_path": adapter_dir / "cosmos_transfer.json",
        "presentation_bundle_path": presentation_dir / "presentation_bundle.json",
        "presentation_world_manifest_path": presentation_dir / "presentation_world_manifest.json",
        "runtime_demo_manifest_path": presentation_dir / "runtime_demo_manifest.json",
        "protected_regions_manifest_path": eval_dir / "protected_regions_manifest.json",
        "canonical_render_policy_path": eval_dir / "canonical_render_policy.json",
        "presentation_variance_policy_path": eval_dir / "presentation_variance_policy.json",
        "site_world_spec_path": eval_dir / "site_world_spec.json",
        "geometry_manifest_path": geometry_dir / "geometry_manifest.json",
        "geometry_summary_path": geometry_dir / "geometry_summary.json",
        "geometry_poses_path": geometry_dir / "camera" / "poses.jsonl",
        "geometry_intrinsics_path": geometry_dir / "camera" / "intrinsics.json",
        "geometry_depth_manifest_path": geometry_dir / "depth" / "depth_manifest.json",
        "geometry_confidence_manifest_path": geometry_dir / "confidence" / "confidence_manifest.json",
    }
    entries: Dict[str, str] = {}
    available = 0
    for key, path in files.items():
        if key == "bundle_path":
            if path.is_dir():
                entries[key] = _relative_to(eval_dir, path)
                available += 1
            continue
        if path.is_file():
            entries[key] = _relative_to(eval_dir, path)
            available += 1
    required = {
        "scene_memory_manifest_path",
        "scene_memory_readiness_path",
        "conditioning_bundle_path",
        "preview_simulation_manifest_path",
        "gen3c_adapter_manifest_path",
        "neoverse_adapter_manifest_path",
        "cosmos_transfer_adapter_manifest_path",
        "presentation_bundle_path",
        "presentation_world_manifest_path",
        "runtime_demo_manifest_path",
        "protected_regions_manifest_path",
        "canonical_render_policy_path",
        "presentation_variance_policy_path",
        "site_world_spec_path",
    }
    status = "complete" if required.issubset(entries) else "partial" if available > 0 else "missing"
    missing = [key for key in sorted(required) if key not in entries]
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": status,
        "missing_required_fields": missing,
        "available_fields": sorted(entries.keys()),
        **entries,
    }


def _normalize_rich_handoff(
    *,
    handoff: Mapping[str, Any],
    scope_record: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
    capture_root: Path,
    geometry_bundle_manifest: Mapping[str, Any],
    scene_memory_bundle_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = dict(handoff)
    qualification_state = str(payload.get("qualification_state") or payload.get("readiness_state") or qualification_record.get("readiness_state") or "not_ready_yet")
    eligibility = bool(
        payload.get("downstream_evaluation_eligibility")
        if payload.get("downstream_evaluation_eligibility") is not None
        else payload.get("match_ready")
    )
    scoped = payload.get("scoped_task_definition") if isinstance(payload.get("scoped_task_definition"), Mapping) else {}
    if not scoped:
        task_text = _default_task_text(scope_record, payload, capture_root.name)
        scoped = {
            "task_id": _default_task_id(scope_record, payload, capture_root.name),
            "scoped_task_statement": task_text,
            "success_criteria": _string_list(scope_record.get("success_criteria")) or ["Complete the scoped task safely"],
            "in_scope_zone": dict(scope_record.get("task_zone") or {}) if isinstance(scope_record.get("task_zone"), Mapping) else capture_root.parts[-3],
        }
    site_constraints = payload.get("site_constraints") if isinstance(payload.get("site_constraints"), Mapping) else {}
    if not site_constraints:
        site_constraints = {
            "operating_constraints": ["Not provided in intake metadata"],
            "privacy_security_constraints": ["Not provided in intake metadata"],
            "known_blockers": _string_list(scope_record.get("blockers")) or ["No known blockers supplied"],
        }
    payload.update(
        {
            "schema_version": "v1",
            "site_submission_id": str(payload.get("site_submission_id") or capture_root.name),
            "opportunity_id": str(payload.get("opportunity_id") or capture_root.parts[-3]),
            "qualification_state": qualification_state,
            "downstream_evaluation_eligibility": eligibility,
            "operator_approved_summary": str(payload.get("operator_approved_summary") or payload.get("summary") or f"Qualified opportunity for {capture_root.parts[-3]}").strip(),
            "scoped_task_definition": dict(scoped),
            "site_constraints": dict(site_constraints),
        }
    )
    normalized_geometry: Dict[str, Any] = {}
    for key in ("bundle_path", "ply_path", "labels_path", "structure_path", "task_hints_path", "holi_spatial_grounding_path"):
        text = str(geometry_bundle_manifest.get(key) or "").strip()
        if text:
            normalized_geometry[key] = text
    if normalized_geometry:
        payload["geometry_package"] = normalized_geometry
    normalized_scene_memory: Dict[str, Any] = {}
    for key in (
        "bundle_path",
        "scene_memory_manifest_path",
        "scene_memory_readiness_path",
        "conditioning_bundle_path",
        "preview_simulation_manifest_path",
        "gen3c_adapter_manifest_path",
        "neoverse_adapter_manifest_path",
        "cosmos_transfer_adapter_manifest_path",
        "presentation_bundle_path",
        "presentation_world_manifest_path",
        "runtime_demo_manifest_path",
        "protected_regions_manifest_path",
        "canonical_render_policy_path",
        "presentation_variance_policy_path",
        "site_world_spec_path",
        "geometry_manifest_path",
        "geometry_summary_path",
        "geometry_poses_path",
        "geometry_intrinsics_path",
        "geometry_depth_manifest_path",
        "geometry_confidence_manifest_path",
    ):
        text = str(scene_memory_bundle_manifest.get(key) or "").strip()
        if text:
            normalized_scene_memory[key] = text
    if normalized_scene_memory:
        payload["scene_memory_package"] = normalized_scene_memory
    return payload


def _build_review_queue(
    *,
    object_geometry_manifest: Mapping[str, Any],
    task_anchor_manifest: Mapping[str, Any],
    simready_validation: Optional[Mapping[str, Any]],
    geometry_bundle_manifest: Mapping[str, Any],
    scene_memory_bundle_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    geometry_objects = object_geometry_manifest.get("objects") if isinstance(object_geometry_manifest.get("objects"), list) else []
    primary_ids: set[str] = set()
    for task in task_anchor_manifest.get("tasks", []):
        if isinstance(task, Mapping):
            primary_ids.update(_string_list(task.get("target_object_ids")))
    for obj in geometry_objects:
        if not isinstance(obj, Mapping):
            continue
        object_id = str(obj.get("object_id") or "")
        if not object_id:
            continue
        selected_views = obj.get("selected_views") if isinstance(obj.get("selected_views"), list) else []
        collision_hulls = obj.get("collision_hulls") if isinstance(obj.get("collision_hulls"), list) else []
        support_surfaces = obj.get("support_surfaces") if isinstance(obj.get("support_surfaces"), list) else []
        if object_id in primary_ids and not selected_views:
            items.append({"severity": "high", "subject_id": object_id, "kind": "missing_selected_views", "detail": "Primary target has no selected views for downstream rendering or review."})
        if object_id in primary_ids and not support_surfaces:
            items.append({"severity": "medium", "subject_id": object_id, "kind": "missing_support_surfaces", "detail": "Primary target has no support surfaces; scene builder may need manual support metadata."})
        if not collision_hulls:
            items.append({"severity": "medium" if object_id in primary_ids else "low", "subject_id": object_id, "kind": "missing_collision_hulls", "detail": "Object geometry has no collision hulls; downstream will rely on coarse collision fallback."})
    if geometry_bundle_manifest.get("status") != "complete":
        items.append(
            {
                "severity": "low" if scene_memory_bundle_manifest.get("status") == "complete" else "medium",
                "subject_id": "geometry_bundle",
                "kind": "incomplete_geometry_bundle",
                "detail": (
                    "Geometry bundle is partial; downstream can continue on canonical scene-memory artifacts."
                    if scene_memory_bundle_manifest.get("status") == "complete"
                    else "Geometry bundle is partial; downstream bootstrap may rely on degraded metadata."
                ),
            }
        )
    if scene_memory_bundle_manifest.get("status") != "complete":
        items.append({"severity": "medium", "subject_id": "scene_memory_bundle", "kind": "incomplete_scene_memory_bundle", "detail": "Canonical scene-memory handoff is partial; downstream adapters may require manual repair."})
    if isinstance(simready_validation, Mapping) and str(simready_validation.get("overall_status") or "") == "degraded":
        items.append({"severity": "low", "subject_id": "simready", "kind": "degraded_simready_prep", "detail": "Best-effort simready prep is degraded; use as advisory only."})
    return {"schema_version": "v1", "generated_at": utc_now_iso(), "items": items}


def _gs_uri(context, relative_path: str) -> str:
    return f"gs://{context.bucket}/{context.capture_prefix}/pipeline/{relative_path}"


def _real_path_from_eval_dir(eval_dir: Path, relative_path: str) -> Optional[Path]:
    text = str(relative_path or "").strip()
    if not text:
        return None
    candidate = (eval_dir / text).resolve()
    return candidate if candidate.exists() else None


def _conditioning_local_paths(*, context, conditioning_bundle: Mapping[str, Any]) -> Dict[str, str]:
    raw_video_path = context.raw_root / "walkthrough.mov"
    world_model_video_path = context.capture_root / "privacy" / "final_walkthrough.mov"
    keyframe_candidates = [
        context.capture_root / "frames" / "keyframe.jpg",
        context.capture_root / "frames" / "keyframe.jpeg",
        context.capture_root / "frames" / "keyframe.png",
        context.raw_root / "keyframe.jpg",
        context.raw_root / "keyframe.jpeg",
        context.raw_root / "keyframe.png",
    ]
    keyframe_path = next((path for path in keyframe_candidates if path.is_file()), None)
    local_paths: Dict[str, str] = {
        "raw_video_path": str(raw_video_path) if raw_video_path.is_file() else "",
        "world_model_video_path": str(world_model_video_path) if world_model_video_path.is_file() else "",
        "keyframe_path": str(keyframe_path) if keyframe_path is not None else "",
        "arkit_poses_path": str(context.raw_root / "arkit" / "poses.jsonl"),
        "arkit_intrinsics_path": str(context.raw_root / "arkit" / "intrinsics.json"),
        "arkit_depth_path": str(context.raw_root / "arkit" / "depth"),
        "geometry_manifest_path": str(context.pipeline_root / "geometry" / "geometry_manifest.json"),
        "geometry_summary_path": str(context.pipeline_root / "geometry" / "geometry_summary.json"),
        "geometry_poses_path": str(context.pipeline_root / "geometry" / "camera" / "poses.jsonl"),
        "geometry_intrinsics_path": str(context.pipeline_root / "geometry" / "camera" / "intrinsics.json"),
        "geometry_depth_manifest_path": str(context.pipeline_root / "geometry" / "depth" / "depth_manifest.json"),
        "geometry_confidence_manifest_path": str(context.pipeline_root / "geometry" / "confidence" / "confidence_manifest.json"),
        "object_index_path": str(context.raw_root / "object_index.json"),
        "scene_memory_manifest_path": str(context.pipeline_root / "scene_memory" / "scene_memory_manifest.json"),
        "conditioning_bundle_path": str(context.pipeline_root / "scene_memory" / "conditioning_bundle.json"),
        "object_geometry_manifest_path": str(context.pipeline_root / "evaluation_prep" / "object_geometry_manifest.json"),
    }
    return local_paths


def _canonical_world_model_payload(
    *,
    context,
    capture_orientation: Mapping[str, Any],
) -> Dict[str, Any]:
    pipeline_root = context.pipeline_root
    raw_root = context.raw_root
    authoritative_runtime_render_manifest_path = pipeline_root / "presentation_world" / "authoritative_runtime_render_manifest.json"
    if authoritative_runtime_render_manifest_path.is_file():
        manifest = _read_json_object(authoritative_runtime_render_manifest_path)
        primary_asset_path = str(manifest.get("primary_asset_path") or "").strip()
        primary_asset_uri = str(manifest.get("primary_asset_uri") or "").strip()
        supporting_assets = [
            dict(item)
            for item in manifest.get("supporting_assets", [])
            if isinstance(item, Mapping)
        ]
        if primary_asset_path or primary_asset_uri:
            return {
                "world_model_backend": "neoverse",
                "primary_runtime_backend": "neoverse",
                "scene_representation": str(manifest.get("scene_representation") or "neoverse_video_world_model_v1"),
                "render_source": str(manifest.get("render_source") or "neoverse_full_capture"),
                "fallback_mode": str(manifest.get("fallback_mode") or "none"),
                "evidence_mode": "full_capture_persistent_scene",
                "primary_render_asset_role": "authoritative_runtime_render_asset",
                "renderer_backend": str(manifest.get("renderer_backend") or "neoverse"),
                "bundle_type": str(manifest.get("bundle_type") or manifest.get("scene_representation") or "neoverse_video_world_model_v1"),
                "status": "ready",
                "primary_asset_path": primary_asset_path,
                "primary_asset_uri": primary_asset_uri,
                "primary_asset_source": str(manifest.get("primary_asset_source") or "authoritative_runtime_render"),
                "orientation": dict(capture_orientation),
                "supporting_assets": supporting_assets,
            }
    supporting_assets: List[Dict[str, Any]] = []
    advanced_bundle = pipeline_root / "advanced_geometry" / "advanced_geometry_bundle.json"
    if advanced_bundle.is_file():
        relative = relative_scene_path(advanced_bundle, context.storage_root)
        suffix = relative.split("/pipeline/", 1)[1] if "/pipeline/" in relative else advanced_bundle.name
        supporting_assets.append(
            {
                "name": advanced_bundle.name,
                "path": str(advanced_bundle.resolve()),
                "uri": _gs_uri(context, suffix),
            }
        )
    return {
        "world_model_backend": "neoverse",
        "primary_runtime_backend": "neoverse",
        "scene_representation": "pending_world_model_service",
        "render_source": "pending_world_model_service",
        "fallback_mode": "none",
        "evidence_mode": "full_capture_persistent_scene",
        "primary_render_asset_role": "authoritative_runtime_render_asset",
        "renderer_backend": None,
        "bundle_type": None,
        "status": "missing",
        "primary_asset_path": "",
        "primary_asset_uri": "",
        "primary_asset_source": "",
        "orientation": dict(capture_orientation),
        "supporting_assets": supporting_assets,
    }


def _primary_runtime_render_descriptor(
    *,
    conditioning_bundle: Mapping[str, Any],
    local_paths: Mapping[str, Any],
    canonical_world_model: Mapping[str, Any],
) -> Dict[str, str]:
    canonical_status = str(canonical_world_model.get("status") or "").strip().lower()
    if canonical_status == "ready":
        return {
            "world_model_backend": str(canonical_world_model.get("world_model_backend") or "neoverse"),
            "scene_representation": str(canonical_world_model.get("scene_representation") or "neoverse_video_world_model_v1"),
            "runtime_render_source": str(canonical_world_model.get("render_source") or "canonical_world_model"),
            "fallback_mode": str(canonical_world_model.get("fallback_mode") or "none"),
        }

    raw_video_ref = str(
        conditioning_bundle.get("world_model_video_uri")
        or conditioning_bundle.get("privacy_processed_video_uri")
        or conditioning_bundle.get("raw_video_uri")
        or local_paths.get("world_model_video_path")
        or local_paths.get("raw_video_path")
        or ""
    ).strip()
    arkit = dict(conditioning_bundle.get("arkit") or {}) if isinstance(conditioning_bundle.get("arkit"), Mapping) else {}
    geometry = dict(conditioning_bundle.get("geometry") or {}) if isinstance(conditioning_bundle.get("geometry"), Mapping) else {}
    poses_ref = str(arkit.get("poses_uri") or local_paths.get("arkit_poses_path") or "").strip()
    intrinsics_ref = str(arkit.get("intrinsics_uri") or local_paths.get("arkit_intrinsics_path") or "").strip()
    depth_ref = str(arkit.get("depth_prefix_uri") or local_paths.get("arkit_depth_path") or "").strip()
    if raw_video_ref and poses_ref and intrinsics_ref:
        return {
            "world_model_backend": "neoverse",
            "scene_representation": "neoverse_video_world_model_v1",
            "runtime_render_source": "neoverse_full_capture",
            "fallback_mode": "arkit_rgbd_last_resort",
        }
    geometry_poses_ref = str(geometry.get("poses_uri") or local_paths.get("geometry_poses_path") or "").strip()
    geometry_intrinsics_ref = str(
        geometry.get("intrinsics_uri") or local_paths.get("geometry_intrinsics_path") or ""
    ).strip()
    geometry_depth_ref = str(
        geometry.get("depth_manifest_uri") or local_paths.get("geometry_depth_manifest_path") or ""
    ).strip()
    if raw_video_ref and geometry_poses_ref and geometry_intrinsics_ref and geometry_depth_ref:
        return {
            "world_model_backend": "neoverse",
            "scene_representation": "geometry_conditioned_capture_v1",
            "runtime_render_source": "geometry_conditioned_capture",
            "fallback_mode": "geometry_lane_conditioning",
        }
    if raw_video_ref and poses_ref and intrinsics_ref and depth_ref:
        return {
            "world_model_backend": "neoverse",
            "scene_representation": "neoverse_video_world_model_v1",
            "runtime_render_source": "neoverse_full_capture",
            "fallback_mode": "arkit_rgbd_last_resort",
        }

    return {
        "world_model_backend": str(canonical_world_model.get("world_model_backend") or "neoverse"),
        "scene_representation": str(canonical_world_model.get("scene_representation") or "unavailable"),
        "runtime_render_source": str(canonical_world_model.get("render_source") or "unavailable"),
        "fallback_mode": str(canonical_world_model.get("fallback_mode") or "arkit_rgbd_last_resort"),
    }


def _runtime_capabilities_payload(
    *,
    launchable: bool,
    base: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    keys = [
        "supports_step_rollout",
        "supports_batch_rollout",
        "supports_camera_views",
        "supports_stream",
        "supports_rlds_export",
        "supports_preview_render",
        "protected_region_locking",
        "runtime_layer_compositing",
        "debug_render_outputs",
    ]
    payload: Dict[str, Any] = {}
    raw = dict(base) if isinstance(base, Mapping) else {}
    for key in keys:
        payload[key] = bool(raw.get(key)) if launchable else False
    return payload


def _runtime_readiness_state(*, launchable: bool, blockers: Sequence[str]) -> str:
    if launchable:
        return "launchable"
    if blockers:
        return "blocked"
    return "incomplete"


def _runtime_eligibility_payload(runtime_status: Mapping[str, Any]) -> Dict[str, Any]:
    blockers = list(runtime_status.get("blockers") or [])
    launchable = bool(runtime_status.get("launchable"))
    return {
        "launchable": launchable,
        "readiness_state": _runtime_readiness_state(
            launchable=launchable,
            blockers=blockers,
        ),
        "blockers": blockers,
        "warnings": list(runtime_status.get("warnings") or []),
        "runtime_base_url": runtime_status.get("runtime_base_url"),
        "websocket_base_url": runtime_status.get("websocket_base_url"),
        "grounding_status": runtime_status.get("grounding_status"),
        "ungrounded_reason": runtime_status.get("ungrounded_reason"),
        "empty_index_cause": runtime_status.get("empty_index_cause"),
        "object_index_backend_blockers": list(runtime_status.get("object_index_backend_blockers") or []),
    }


def _canonical_site_world_runtime_status(
    *,
    qualification_state: object,
    downstream_evaluation_eligibility: bool,
    scene_memory_bundle_manifest: Mapping[str, Any],
    object_geometry_manifest: Mapping[str, Any],
    protected_regions_manifest: Mapping[str, Any],
    required_runtime_artifact_paths: Sequence[Path],
    runtime_service_url: str,
) -> Dict[str, Any]:
    blockers: List[str] = []
    warnings: List[str] = []
    normalized_qualification_state = str(qualification_state or "").strip().lower()
    grounding_status = str(protected_regions_manifest.get("grounding_status") or "grounded").strip().lower() or "grounded"
    ungrounded_reason = str(protected_regions_manifest.get("ungrounded_reason") or "").strip() or None
    empty_index_cause = str(protected_regions_manifest.get("empty_index_cause") or "").strip() or None
    object_index_backend_blockers = _string_list(object_geometry_manifest.get("object_index_backend_blockers"))

    if normalized_qualification_state and normalized_qualification_state != "ready":
        warnings.append(f"qualification_state:{normalized_qualification_state}")
    if not downstream_evaluation_eligibility:
        warnings.append("downstream_evaluation_eligibility:false")
    if str(scene_memory_bundle_manifest.get("status") or "").strip() != "complete":
        warnings.append(f"scene_memory_bundle:{scene_memory_bundle_manifest.get('status')}")
    if grounding_status != "grounded":
        warnings.append(f"grounding_status:{grounding_status}")
        if ungrounded_reason:
            warnings.append(f"ungrounded_reason:{ungrounded_reason}")
    if not runtime_service_url:
        blockers.append("missing_runtime_service_url")
    for artifact_path in required_runtime_artifact_paths:
        if not artifact_path.is_file():
            blockers.append(f"missing_runtime_artifact:{artifact_path.name}")
    for blocker in object_index_backend_blockers:
        if blocker not in warnings:
            warnings.append(blocker)
    if empty_index_cause:
        warnings.append(f"empty_index_cause:{empty_index_cause}")

    launchable = len(blockers) == 0
    return {
        "status": "ready" if launchable else "blocked",
        "launchable": launchable,
        "readiness_state": _runtime_readiness_state(
            launchable=launchable,
            blockers=blockers,
        ),
        "blockers": blockers,
        "warnings": warnings,
        "runtime_base_url": runtime_service_url or None,
        "websocket_base_url": runtime_service_url.replace("http://", "ws://").replace("https://", "wss://") if runtime_service_url else None,
        "grounding_status": grounding_status,
        "ungrounded_reason": ungrounded_reason,
        "empty_index_cause": empty_index_cause,
        "object_index_backend_blockers": object_index_backend_blockers,
        "scene_memory_bundle_status": str(scene_memory_bundle_manifest.get("status") or "missing"),
    }


def _site_world_id(scene_id: str, capture_id: str) -> str:
    return f"siteworld-{sha256(f'{scene_id}::{capture_id}'.encode('utf-8')).hexdigest()[:12]}"


def _policy() -> WorldModelPolicy:
    return WorldModelPolicy.from_env()


def _read_json_object(path: Path) -> Dict[str, Any]:
    payload = _read_optional_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


_CANONICAL_HASH_VOLATILE_KEYS = {
    "canonical_package_version",
    "generated_at",
    "last_heartbeat_at",
    "runtime_registration_attempt",
    "runtime_registration_attempted",
    "runtime_registration_status",
}


def _canonical_hash_payload(payload: Any) -> Any:
    # Pipeline strips transport/runtime timestamps before delegating hashing to
    # BlueprintContracts so repeated package builds remain deterministic.
    if isinstance(payload, Mapping):
        return {
            str(key): _canonical_hash_payload(value)
            for key, value in payload.items()
            if str(key) not in _CANONICAL_HASH_VOLATILE_KEYS
        }
    if isinstance(payload, list):
        return [_canonical_hash_payload(item) for item in payload]
    return payload


def _runtime_registration_attempt(
    *,
    attempted: bool,
    status: str,
    reason: Optional[str],
    spec: Mapping[str, Any],
    registration: Mapping[str, Any],
    health: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "mode": "optional_downstream_package_registration",
        "attempted": attempted,
        "status": status,
        "reason": reason,
        "submitted_package": {
            "site_world_id": registration.get("site_world_id"),
            "canonical_package_version": spec.get("canonical_package_version"),
            "spec_uri": spec.get("canonical_package_uri"),
            "registration_uri": registration.get("canonical_artifact_uri"),
            "health_uri": health.get("canonical_artifact_uri"),
        },
    }


def _task_critical_ids_from_manifest(task_anchor_manifest: Mapping[str, Any]) -> set[str]:
    tasks = (
        task_anchor_manifest.get("tasks")
        if isinstance(task_anchor_manifest.get("tasks"), list)
        else []
    )
    return task_critical_object_ids([task for task in tasks if isinstance(task, Mapping)])


def _descriptor_scene_memory_capture(descriptor: Mapping[str, Any]) -> Dict[str, Any]:
    scene_memory_capture = descriptor.get("scene_memory_capture")
    if isinstance(scene_memory_capture, Mapping):
        return dict(scene_memory_capture)
    metadata = descriptor.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("scene_memory_capture"), Mapping):
        return dict(metadata.get("scene_memory_capture"))
    return {}


def _descriptor_capture_orientation(
    descriptor: Mapping[str, Any],
    conditioning_bundle: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(descriptor.get("capture_orientation"), Mapping):
        return dict(descriptor.get("capture_orientation") or {})
    metadata = descriptor.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("capture_orientation"), Mapping):
        return dict(metadata.get("capture_orientation") or {})
    if isinstance(conditioning_bundle, Mapping) and isinstance(
        conditioning_bundle.get("capture_orientation"), Mapping
    ):
        return dict(conditioning_bundle.get("capture_orientation") or {})
    return {}


def _presentation_demo_readiness(runtime_demo_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    interactive_demo = (
        runtime_demo_manifest.get("interactive_demo")
        if isinstance(runtime_demo_manifest.get("interactive_demo"), Mapping)
        else {}
    )
    blockers = _string_list(interactive_demo.get("blockers"))
    bundle_status = str(
        runtime_demo_manifest.get("bundle_status")
        or ((runtime_demo_manifest.get("readiness") or {}).get("bundle_status") if isinstance(runtime_demo_manifest.get("readiness"), Mapping) else "")
        or runtime_demo_manifest.get("status")
        or ""
    ).strip().lower()
    if bundle_status and bundle_status not in {"bundle_ready", "ready", "demo_ready"}:
        blockers = list(dict.fromkeys([*blockers, f"presentation_bundle_status:{bundle_status}"]))
    readiness_state = str(interactive_demo.get("readiness_state") or "").strip().lower()
    if readiness_state:
        return {
            "readiness_state": readiness_state if not blockers else "blocked",
            "blockers": blockers,
        }
    if str(runtime_demo_manifest.get("ui_base_url") or "").strip() or str(
        runtime_demo_manifest.get("public_ui_base_url") or ""
    ).strip():
        return {"readiness_state": "ready", "blockers": blockers}
    blockers = list(dict.fromkeys([*blockers, "missing_demo_ui_base_url"]))
    return {"readiness_state": "blocked", "blockers": blockers}


def _refresh_presentation_contract_payload(
    *,
    payload: Mapping[str, Any],
    context,
    canonical_package_version: str,
    derivation_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    updated = dict(payload)
    canonical_package_uri = _gs_uri(context, "evaluation_prep/site_world_spec.json")
    updated["canonical_package_version"] = canonical_package_version
    updated["canonical_package_uri"] = canonical_package_uri
    if "derivation_policy" not in updated:
        updated["derivation_policy"] = dict(derivation_policy)

    if isinstance(updated.get("canonical_source"), Mapping):
        canonical_source = dict(updated.get("canonical_source") or {})
        canonical_source.update(
            {
                "canonical_package_uri": canonical_package_uri,
                "canonical_package_version": canonical_package_version,
                "protected_regions_manifest_uri": _gs_uri(
                    context, "evaluation_prep/protected_regions_manifest.json"
                ),
                "object_geometry_manifest_uri": _gs_uri(
                    context, "evaluation_prep/object_geometry_manifest.json"
                ),
                "site_world_spec_uri": canonical_package_uri,
            }
        )
        updated["canonical_source"] = canonical_source

    if isinstance(updated.get("render_inputs"), Mapping):
        render_inputs = dict(updated.get("render_inputs") or {})
        render_inputs.update(
            {
                "protected_regions_manifest_uri": _gs_uri(
                    context, "evaluation_prep/protected_regions_manifest.json"
                ),
                "object_geometry_manifest_uri": _gs_uri(
                    context, "evaluation_prep/object_geometry_manifest.json"
                ),
                "site_world_spec_uri": canonical_package_uri,
            }
        )
        updated["render_inputs"] = render_inputs

    if isinstance(updated.get("runtime_contract"), Mapping):
        runtime_contract = dict(updated.get("runtime_contract") or {})
        runtime_contract.update(
            {
                "canonical_package_uri": canonical_package_uri,
                "canonical_package_version": canonical_package_version,
            }
        )
        updated["runtime_contract"] = runtime_contract

    if isinstance(updated.get("interactive_demo"), Mapping):
        interactive_demo = dict(updated.get("interactive_demo") or {})
        if isinstance(interactive_demo.get("render_inputs"), Mapping):
            render_inputs = dict(interactive_demo.get("render_inputs") or {})
            render_inputs.update(
                {
                    "protected_regions_manifest_uri": _gs_uri(
                        context, "evaluation_prep/protected_regions_manifest.json"
                    ),
                    "object_geometry_manifest_uri": _gs_uri(
                        context, "evaluation_prep/object_geometry_manifest.json"
                    ),
                    "site_world_spec_uri": canonical_package_uri,
                }
            )
            interactive_demo["render_inputs"] = render_inputs
        updated["interactive_demo"] = interactive_demo

    if isinstance(updated.get("readiness"), Mapping):
        readiness = dict(updated.get("readiness") or {})
        readiness.setdefault("bundle_status", str(updated.get("status") or "unknown"))
        updated["readiness"] = readiness

    return updated


def _gate(prefixes: Sequence[str], items: Sequence[str]) -> bool:
    normalized = [str(item or "").strip() for item in items if str(item or "").strip()]
    return not any(any(item.startswith(prefix) for prefix in prefixes) for item in normalized)


def _object_geometry_has_provenance(object_geometry_manifest: Mapping[str, Any]) -> bool:
    objects = (
        object_geometry_manifest.get("objects")
        if isinstance(object_geometry_manifest.get("objects"), list)
        else []
    )
    if not objects:
        return False
    for item in objects:
        if not isinstance(item, Mapping):
            continue
        provenance = item.get("provenance")
        if (
            isinstance(provenance, Mapping)
            and str(provenance.get("grounding_level") or "").strip()
            and provenance.get("canonical_truth") is True
        ):
            return True
    return False


def _build_site_world_spec(
    *,
    context,
    eval_dir: Path,
    normalized_handoff: Mapping[str, Any],
    scene_memory_bundle_manifest: Mapping[str, Any],
    object_geometry_manifest: Mapping[str, Any],
    task_anchor_manifest: Mapping[str, Any],
    task_run_manifest: Mapping[str, Any],
    protected_regions_manifest: Mapping[str, Any],
    canonical_render_policy: Mapping[str, Any],
    presentation_variance_policy: Mapping[str, Any],
    canonical_runtime_status: Mapping[str, Any],
    canonical_package_version: Optional[str] = None,
) -> Dict[str, Any]:
    policy = _policy()
    descriptor = _read_optional_json_any(context.descriptor_path)
    descriptor_map = dict(descriptor) if isinstance(descriptor, Mapping) else {}
    conditioning_bundle_path = _real_path_from_eval_dir(
        eval_dir, str(scene_memory_bundle_manifest.get("conditioning_bundle_path") or "")
    )
    conditioning_bundle = _read_optional_json_any(conditioning_bundle_path) if conditioning_bundle_path else {}
    conditioning_map = dict(conditioning_bundle) if isinstance(conditioning_bundle, Mapping) else {}
    local_paths = _conditioning_local_paths(context=context, conditioning_bundle=conditioning_map)
    capture_orientation = _descriptor_capture_orientation(descriptor_map, conditioning_map)
    canonical_world_model = _canonical_world_model_payload(
        context=context,
        capture_orientation=capture_orientation,
    )
    runtime_render_descriptor = _primary_runtime_render_descriptor(
        conditioning_bundle=conditioning_map,
        local_paths=local_paths,
        canonical_world_model=canonical_world_model,
    )
    critical_ids = _task_critical_ids_from_manifest(task_anchor_manifest)
    normalized_tasks = []
    for task in task_anchor_manifest.get("tasks", []) if isinstance(task_anchor_manifest.get("tasks"), list) else []:
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("task_id") or task.get("id") or "").strip()
        task_text = str(task.get("task_text") or task.get("name") or task_id or "task").strip()
        task_provenance = build_provenance_record(
            grounding_level="reconstructed" if _string_list(task.get("target_object_ids")) else "inferred",
            evidence_sources=[_gs_uri(context, "evaluation_prep/task_anchor_manifest.json")],
            observation_coverage={"target_object_count": len(_string_list(task.get("target_object_ids")))},
            confidence=1.0 if _string_list(task.get("target_object_ids")) else 0.5,
            canonical_truth=True,
            presentation_only=False,
        )
        normalized_tasks.append(
            with_grounding_fields(
                {
                "id": task_id or _stable_id("task", task_text, fallback="task_default"),
                "task_id": task_id,
                "task_text": task_text,
                "task_category": str(task.get("task_category") or "generic"),
                "target_object_ids": _string_list(task.get("target_object_ids")),
                "articulation_required_ids": _string_list(task.get("articulation_required_ids")),
                "task_critical": bool(_string_list(task.get("target_object_ids"), task.get("articulation_required_ids"))),
                "provenance": task_provenance,
                },
                provenance=task_provenance,
            )
        )
    geometry_bundle = scene_memory_bundle_manifest.get("bundle_path")
    object_geometry_path = eval_dir / "object_geometry_manifest.json"
    scene_memory_capture = _descriptor_scene_memory_capture(descriptor_map)
    presentation_world_manifest_path = context.pipeline_root / "presentation_world" / "presentation_world_manifest.json"
    runtime_demo_manifest_path = context.pipeline_root / "presentation_world" / "runtime_demo_manifest.json"
    presentation_world_manifest = _read_json_object(presentation_world_manifest_path)
    runtime_demo_manifest = _read_json_object(runtime_demo_manifest_path)
    spec_provenance = build_provenance_record(
        grounding_level="reconstructed" if bool(canonical_runtime_status.get("launchable")) else "observed",
        evidence_sources=[
            _gs_uri(context, "scene_memory/scene_memory_manifest.json"),
            _gs_uri(context, "scene_memory/conditioning_bundle.json"),
            _gs_uri(context, "evaluation_prep/object_geometry_manifest.json"),
        ],
        observation_coverage={
            "task_count": len(normalized_tasks),
            "runtime_launchable": bool(canonical_runtime_status.get("launchable")),
            "task_critical_object_count": len(critical_ids),
        },
        confidence=1.0 if bool(canonical_runtime_status.get("launchable")) else 0.75,
        canonical_truth=True,
        presentation_only=False,
    )
    spec = with_grounding_fields({
        "schema_version": "v1",
        "site_world_id": _site_world_id(context.scene_id, context.capture_id),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "site_submission_id": str(normalized_handoff.get("site_submission_id") or context.capture_id),
        "canonical_package_uri": _gs_uri(context, "evaluation_prep/site_world_spec.json"),
        "canonical_package_version": canonical_package_version,
        "qualification_state": normalized_handoff.get("qualification_state"),
        "downstream_evaluation_eligibility": bool(normalized_handoff.get("downstream_evaluation_eligibility")),
        "capture_source": descriptor_map.get("capture_source") or descriptor_map.get("capture_modality"),
        "capture_orientation": capture_orientation,
        "processing_profile": descriptor_map.get("processing_profile"),
        "runtime_render_source": runtime_render_descriptor["runtime_render_source"],
        "fallback_mode": runtime_render_descriptor["fallback_mode"],
        "world_model_backend": runtime_render_descriptor["world_model_backend"],
        "scene_representation": runtime_render_descriptor["scene_representation"],
        "conditioning": {
            "scene_memory_manifest_uri": _gs_uri(context, "scene_memory/scene_memory_manifest.json"),
            "conditioning_bundle_uri": _gs_uri(context, "scene_memory/conditioning_bundle.json"),
            "scene_memory_manifest_path": str((context.pipeline_root / "scene_memory" / "scene_memory_manifest.json").resolve()),
            "conditioning_bundle_path": str((context.pipeline_root / "scene_memory" / "conditioning_bundle.json").resolve()),
            "capture_orientation": capture_orientation,
            "raw_video_uri": conditioning_map.get("raw_video_uri"),
            "keyframe_uri": conditioning_map.get("keyframe_uri"),
            "arkit_poses_uri": ((conditioning_map.get("arkit") or {}) if isinstance(conditioning_map.get("arkit"), Mapping) else {}).get("poses_uri"),
            "arkit_intrinsics_uri": ((conditioning_map.get("arkit") or {}) if isinstance(conditioning_map.get("arkit"), Mapping) else {}).get("intrinsics_uri"),
            "arkit_depth_uri": ((conditioning_map.get("arkit") or {}) if isinstance(conditioning_map.get("arkit"), Mapping) else {}).get("depth_prefix_uri"),
            "geometry_manifest_uri": ((conditioning_map.get("geometry") or {}) if isinstance(conditioning_map.get("geometry"), Mapping) else {}).get("manifest_uri"),
            "geometry_summary_uri": ((conditioning_map.get("geometry") or {}) if isinstance(conditioning_map.get("geometry"), Mapping) else {}).get("summary_uri"),
            "geometry_poses_uri": ((conditioning_map.get("geometry") or {}) if isinstance(conditioning_map.get("geometry"), Mapping) else {}).get("poses_uri"),
            "geometry_intrinsics_uri": ((conditioning_map.get("geometry") or {}) if isinstance(conditioning_map.get("geometry"), Mapping) else {}).get("intrinsics_uri"),
            "geometry_depth_manifest_uri": ((conditioning_map.get("geometry") or {}) if isinstance(conditioning_map.get("geometry"), Mapping) else {}).get("depth_manifest_uri"),
            "geometry_confidence_manifest_uri": ((conditioning_map.get("geometry") or {}) if isinstance(conditioning_map.get("geometry"), Mapping) else {}).get("confidence_manifest_uri"),
            "sensor_availability": scene_memory_capture.get("sensor_availability", {}),
            "local_paths": local_paths,
        },
        "primary_runtime_backend": "neoverse",
        "canonical_world_model": canonical_world_model,
        "geometry": {
            "scene_memory_bundle_path": str(_real_path_from_eval_dir(eval_dir, str(geometry_bundle or "")) or ""),
            "object_geometry_manifest_path": str(object_geometry_path.resolve()),
            "object_index_path": local_paths.get("object_index_path"),
            "advanced_geometry_bundle_path": str((context.pipeline_root / "advanced_geometry" / "advanced_geometry_bundle.json").resolve()),
            "geometry_manifest_path": local_paths.get("geometry_manifest_path"),
            "geometry_summary_path": local_paths.get("geometry_summary_path"),
            "geometry_poses_path": local_paths.get("geometry_poses_path"),
            "geometry_intrinsics_path": local_paths.get("geometry_intrinsics_path"),
            "geometry_depth_manifest_path": local_paths.get("geometry_depth_manifest_path"),
            "geometry_confidence_manifest_path": local_paths.get("geometry_confidence_manifest_path"),
            "geometry_summary": dict(
                ((conditioning_map.get("geometry") or {}) if isinstance(conditioning_map.get("geometry"), Mapping) else {}).get("summary")
                or {}
            ),
        },
        "presentation": {
            "presentation_world_manifest_uri": _gs_uri(context, "presentation_world/presentation_world_manifest.json"),
            "presentation_world_manifest_path": str(presentation_world_manifest_path.resolve()),
            "runtime_demo_manifest_uri": _gs_uri(context, "presentation_world/runtime_demo_manifest.json"),
            "runtime_demo_manifest_path": str(runtime_demo_manifest_path.resolve()),
            "bundle_type": str(
                presentation_world_manifest.get("bundle_type")
                or runtime_demo_manifest.get("bundle_type")
                or ""
            ),
            "renderer_backend": str(
                presentation_world_manifest.get("renderer_backend")
                or runtime_demo_manifest.get("renderer_backend")
                or "neoverse"
            ),
            "bundle_status": str(
                runtime_demo_manifest.get("bundle_status")
                or ((presentation_world_manifest.get("readiness") or {}).get("bundle_status") if isinstance(presentation_world_manifest.get("readiness"), Mapping) else "")
                or presentation_world_manifest.get("status")
                or "missing"
            ),
            "primary_asset_path": str(
                presentation_world_manifest.get("primary_asset_path")
                or runtime_demo_manifest.get("primary_asset_path")
                or ""
            ),
            "orientation": dict(
                presentation_world_manifest.get("orientation")
                if isinstance(presentation_world_manifest.get("orientation"), Mapping)
                else runtime_demo_manifest.get("orientation")
                if isinstance(runtime_demo_manifest.get("orientation"), Mapping)
                else capture_orientation
            ),
            "fallback_policy": str(
                runtime_demo_manifest.get("fallback_policy")
                or presentation_world_manifest.get("fallback_policy")
                or "canonical_only"
            ),
        },
        "grounding_status": str(protected_regions_manifest.get("grounding_status") or "grounded"),
        "ungrounded_reason": protected_regions_manifest.get("ungrounded_reason"),
        "empty_index_cause": protected_regions_manifest.get("empty_index_cause"),
        "object_index_backend_blockers": _string_list(object_geometry_manifest.get("object_index_backend_blockers")),
        "runtime_layer_policy": {
            "protected_regions_manifest_uri": _gs_uri(context, "evaluation_prep/protected_regions_manifest.json"),
            "canonical_render_policy_uri": _gs_uri(context, "evaluation_prep/canonical_render_policy.json"),
            "presentation_variance_policy_uri": _gs_uri(context, "evaluation_prep/presentation_variance_policy.json"),
            "protected_regions_manifest_path": str((eval_dir / "protected_regions_manifest.json").resolve()),
            "canonical_render_policy_path": str((eval_dir / "canonical_render_policy.json").resolve()),
            "presentation_variance_policy_path": str((eval_dir / "presentation_variance_policy.json").resolve()),
            "grounding_status": str(protected_regions_manifest.get("grounding_status") or "grounded"),
            "ungrounded_reason": protected_regions_manifest.get("ungrounded_reason"),
            "empty_index_cause": protected_regions_manifest.get("empty_index_cause"),
            "object_index_backend_blockers": _string_list(object_geometry_manifest.get("object_index_backend_blockers")),
            "region_count": int(protected_regions_manifest.get("region_count") or 0),
            "protected_region_locking": True,
            "runtime_layer_compositing": True,
            "debug_render_outputs": [
                "canonical_only.png",
                "locked_mask.png",
                "editable_mask.png",
                "final_composite.png",
            ],
            "canonical_render_policy": dict(canonical_render_policy),
            "presentation_variance_policy": dict(presentation_variance_policy),
        },
        "task_anchor_manifest_path": str((eval_dir / "task_anchor_manifest.json").resolve()),
        "task_catalog": normalized_tasks,
        "scenario_catalog": [
            {
                "id": _stable_id("scenario", text, fallback=f"scenario_{index}"),
                "name": text,
                "source": "site_world_runtime",
            }
            for index, text in enumerate(_string_list("default", "counterfactual_lighting", "counterfactual_clutter"))
        ],
        "start_state_catalog": list(
            task_run_manifest.get("start_state_catalog")
            or [
                {
                    "id": _stable_id("start", text, fallback=f"state_{index}"),
                    "name": text,
                    "task_id": None,
                    "source": "task_run_manifest",
                }
                for index, text in enumerate(_string_list(task_run_manifest.get("start_states")) or ["default_start_state"])
            ]
        ),
        "robot_profiles": _default_robot_profiles(),
        "qualification_references": {
            "qualified_opportunity_handoff_uri": _gs_uri(context, "evaluation_prep/qualified_opportunity_handoff.json"),
            "qualification_record_uri": _gs_uri(context, "qualification_record.json"),
            "task_scope_record_uri": _gs_uri(context, "task_scope_record.json"),
        },
        "runtime_eligibility": _runtime_eligibility_payload(canonical_runtime_status),
        "world_model_policy": policy.to_dict(),
        "canonical_output": build_output_linkage(
            policy=policy,
            canonical_artifact_uri=_gs_uri(context, "evaluation_prep/site_world_spec.json"),
            presentation_artifact_uri=_gs_uri(context, "presentation_world/presentation_world_manifest.json") if policy.emit_presentation else None,
            authoritative_record=True,
        ),
        "presentation_output": build_output_linkage(
            policy=policy,
            canonical_artifact_uri=_gs_uri(context, "evaluation_prep/site_world_spec.json"),
            presentation_artifact_uri=_gs_uri(context, "presentation_world/presentation_bundle.json") if policy.emit_presentation else None,
            authoritative_record=False,
            derivation_mode=policy.allow_generative_completion,
        ),
        "provenance": spec_provenance,
        "generated_at": utc_now_iso(),
    }, provenance=spec_provenance)
    return spec


def _build_site_world_runtime_records(
    *,
    context,
    spec: Mapping[str, Any],
    canonical_runtime_status: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    policy = _policy()
    site_world_id = str(spec.get("site_world_id") or _site_world_id(context.scene_id, context.capture_id))
    service_url = str(canonical_runtime_status.get("runtime_base_url") or "").strip()
    task_catalog = list(spec.get("task_catalog") or [])
    scenario_catalog = list(spec.get("scenario_catalog") or [])
    start_state_catalog = list(spec.get("start_state_catalog") or [])
    robot_profiles = list(spec.get("robot_profiles") or [])
    blocked_runtime_capabilities = _runtime_capabilities_payload(launchable=False)
    if not bool(canonical_runtime_status.get("launchable")):
        registration = {
            "schema_version": "v1",
            "site_world_id": site_world_id,
            "build_id": None,
            "scene_id": context.scene_id,
            "capture_id": context.capture_id,
            "site_submission_id": spec.get("site_submission_id"),
            "status": "blocked",
            "runtime_base_url": service_url or None,
            "websocket_base_url": canonical_runtime_status.get("websocket_base_url"),
            "vm_instance_id": os.getenv("VASTAI_INSTANCE_ID") or os.getenv("HOSTNAME") or None,
            "supported_cameras": [],
            "scenario_catalog": scenario_catalog,
            "start_state_catalog": start_state_catalog,
            "task_catalog": task_catalog,
            "robot_profiles": robot_profiles,
            "runtime_capabilities": blocked_runtime_capabilities,
            "blockers": list(canonical_runtime_status.get("blockers") or []),
            "warnings": list(canonical_runtime_status.get("warnings") or []),
            "world_model_policy": policy.to_dict(),
            "grounding_status": canonical_runtime_status.get("grounding_status"),
            "ungrounded_reason": canonical_runtime_status.get("ungrounded_reason"),
            "empty_index_cause": canonical_runtime_status.get("empty_index_cause"),
            "object_index_backend_blockers": list(canonical_runtime_status.get("object_index_backend_blockers") or []),
            "canonical_package_version": spec.get("canonical_package_version"),
            "canonical_artifact_uri": _gs_uri(context, "evaluation_prep/site_world_registration.json"),
            "presentation_artifact_uri": _gs_uri(context, "presentation_world/runtime_demo_manifest.json") if policy.emit_presentation else None,
            "derivation_mode": policy.output_policy,
            "authoritative_record": True,
            "generated_at": utc_now_iso(),
        }
        health = {
            "schema_version": "v1",
            "site_world_id": site_world_id,
            "build_id": None,
            "scene_id": context.scene_id,
            "capture_id": context.capture_id,
            "site_submission_id": spec.get("site_submission_id"),
            "healthy": False,
            "launchable": False,
            "status": "blocked",
            "runtime_base_url": service_url or None,
            "websocket_base_url": canonical_runtime_status.get("websocket_base_url"),
            "vm_instance_id": os.getenv("VASTAI_INSTANCE_ID") or os.getenv("HOSTNAME") or None,
            "supported_cameras": [],
            "scenario_catalog": scenario_catalog,
            "start_state_catalog": start_state_catalog,
            "task_catalog": task_catalog,
            "robot_profiles": robot_profiles,
            "runtime_capabilities": blocked_runtime_capabilities,
            "blockers": list(canonical_runtime_status.get("blockers") or []),
            "warnings": list(canonical_runtime_status.get("warnings") or []),
            "world_model_policy": policy.to_dict(),
            "grounding_status": canonical_runtime_status.get("grounding_status"),
            "ungrounded_reason": canonical_runtime_status.get("ungrounded_reason"),
            "empty_index_cause": canonical_runtime_status.get("empty_index_cause"),
            "object_index_backend_blockers": list(canonical_runtime_status.get("object_index_backend_blockers") or []),
            "canonical_package_version": spec.get("canonical_package_version"),
            "canonical_artifact_uri": _gs_uri(context, "evaluation_prep/site_world_health.json"),
            "presentation_artifact_uri": _gs_uri(context, "presentation_world/runtime_demo_manifest.json") if policy.emit_presentation else None,
            "derivation_mode": policy.output_policy,
            "authoritative_record": True,
            "last_heartbeat_at": utc_now_iso(),
        }
        registration_attempt = _runtime_registration_attempt(
            attempted=False,
            status="skipped",
            reason="runtime_registration_blocked",
            spec=spec,
            registration=registration,
            health=health,
        )
        registration["runtime_registration_attempted"] = False
        registration["runtime_registration_status"] = "skipped"
        registration["runtime_registration_attempt"] = registration_attempt
        health["runtime_registration_attempted"] = False
        health["runtime_registration_status"] = "skipped"
        health["runtime_registration_attempt"] = dict(registration_attempt)
        return registration, health

    registration = {
        "schema_version": "v1",
        "site_world_id": site_world_id,
        "build_id": None,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "site_submission_id": spec.get("site_submission_id"),
        "status": "ready",
        "runtime_base_url": service_url or None,
        "websocket_base_url": canonical_runtime_status.get("websocket_base_url"),
        "vm_instance_id": os.getenv("VASTAI_INSTANCE_ID") or os.getenv("HOSTNAME") or None,
        "supported_cameras": [],
        "primary_runtime_backend": "neoverse",
        "canonical_world_model": dict(spec.get("canonical_world_model") or {}),
        "world_model_backend": spec.get("world_model_backend"),
        "scene_representation": spec.get("scene_representation"),
        "render_source": spec.get("runtime_render_source"),
        "fallback_mode": spec.get("fallback_mode"),
        "scenario_catalog": scenario_catalog,
        "start_state_catalog": start_state_catalog,
        "task_catalog": task_catalog,
        "robot_profiles": robot_profiles,
        "runtime_capabilities": _runtime_capabilities_payload(launchable=True),
        "blockers": list(canonical_runtime_status.get("blockers") or []),
        "warnings": list(canonical_runtime_status.get("warnings") or []),
        "world_model_policy": policy.to_dict(),
        "grounding_status": canonical_runtime_status.get("grounding_status"),
        "ungrounded_reason": canonical_runtime_status.get("ungrounded_reason"),
        "empty_index_cause": canonical_runtime_status.get("empty_index_cause"),
        "object_index_backend_blockers": list(canonical_runtime_status.get("object_index_backend_blockers") or []),
        "canonical_package_version": spec.get("canonical_package_version"),
        "canonical_artifact_uri": _gs_uri(context, "evaluation_prep/site_world_registration.json"),
        "presentation_artifact_uri": _gs_uri(context, "presentation_world/runtime_demo_manifest.json") if policy.emit_presentation else None,
        "derivation_mode": policy.output_policy,
        "authoritative_record": True,
        "generated_at": utc_now_iso(),
    }
    health = {
        "schema_version": "v1",
        "site_world_id": site_world_id,
        "build_id": None,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "site_submission_id": spec.get("site_submission_id"),
        "healthy": True,
        "launchable": True,
        "status": "healthy",
        "runtime_base_url": service_url or None,
        "websocket_base_url": canonical_runtime_status.get("websocket_base_url"),
        "vm_instance_id": os.getenv("VASTAI_INSTANCE_ID") or os.getenv("HOSTNAME") or None,
        "supported_cameras": [],
        "primary_runtime_backend": "neoverse",
        "canonical_world_model": dict(spec.get("canonical_world_model") or {}),
        "world_model_backend": spec.get("world_model_backend"),
        "scene_representation": spec.get("scene_representation"),
        "render_source": spec.get("runtime_render_source"),
        "fallback_mode": spec.get("fallback_mode"),
        "scenario_catalog": scenario_catalog,
        "start_state_catalog": start_state_catalog,
        "task_catalog": task_catalog,
        "robot_profiles": robot_profiles,
        "runtime_capabilities": _runtime_capabilities_payload(launchable=True),
        "blockers": list(canonical_runtime_status.get("blockers") or []),
        "warnings": list(canonical_runtime_status.get("warnings") or []),
        "world_model_policy": policy.to_dict(),
        "grounding_status": canonical_runtime_status.get("grounding_status"),
        "ungrounded_reason": canonical_runtime_status.get("ungrounded_reason"),
        "empty_index_cause": canonical_runtime_status.get("empty_index_cause"),
        "object_index_backend_blockers": list(canonical_runtime_status.get("object_index_backend_blockers") or []),
        "canonical_package_version": spec.get("canonical_package_version"),
        "canonical_artifact_uri": _gs_uri(context, "evaluation_prep/site_world_health.json"),
        "presentation_artifact_uri": _gs_uri(context, "presentation_world/runtime_demo_manifest.json") if policy.emit_presentation else None,
        "derivation_mode": policy.output_policy,
        "authoritative_record": True,
        "last_heartbeat_at": utc_now_iso(),
    }
    registration_attempt = _runtime_registration_attempt(
        attempted=True,
        status="submitted",
        reason=None,
        spec=spec,
        registration=registration,
        health=health,
    )
    registration["runtime_registration_attempted"] = True
    registration["runtime_registration_status"] = "submitted"
    registration["runtime_registration_attempt"] = registration_attempt
    health["runtime_registration_attempted"] = True
    health["runtime_registration_status"] = "submitted"
    health["runtime_registration_attempt"] = dict(registration_attempt)
    prebuilt_package_payload = {
        "spec": dict(spec),
        "registration": dict(registration),
        "health": dict(health),
    }

    client = SiteWorldRuntimeServiceClient(SiteWorldRuntimeServiceConfig.from_env())
    try:
        registration_response_payload = dict(
            client.register_site_world_package(
                spec=prebuilt_package_payload["spec"],
                registration=prebuilt_package_payload["registration"],
                health=prebuilt_package_payload["health"],
            )
        )
    except Exception as exc:
        failure_blocker = f"runtime_registration_failed:{exc}"
        registration["status"] = "blocked"
        registration["runtime_capabilities"] = _runtime_capabilities_payload(launchable=False)
        registration["blockers"] = list(
            dict.fromkeys([*list(canonical_runtime_status.get("blockers") or []), failure_blocker])
        )
        registration["warnings"] = list(canonical_runtime_status.get("warnings") or [])
        registration["runtime_registration_attempted"] = True
        registration["runtime_registration_status"] = "failed"
        registration["runtime_registration_attempt"] = _runtime_registration_attempt(
            attempted=True,
            status="failed",
            reason=str(exc),
            spec=spec,
            registration=registration,
            health=health,
        )
        health = {
            **health,
            "healthy": False,
            "launchable": False,
            "status": "degraded",
            "blockers": list(dict.fromkeys([*list(canonical_runtime_status.get("blockers") or []), failure_blocker])),
            "warnings": list(canonical_runtime_status.get("warnings") or []),
            "last_heartbeat_at": utc_now_iso(),
            "runtime_registration_attempted": True,
            "runtime_registration_status": "failed",
            "runtime_registration_attempt": dict(registration["runtime_registration_attempt"]),
        }
        health["runtime_capabilities"] = _runtime_capabilities_payload(
            launchable=False,
            base=health.get("runtime_capabilities"),
        )
        return registration, health

    runtime_registration = {
        key: registration_response_payload.get(key)
        for key in (
            "schema_version",
            "site_world_id",
            "build_id",
            "scene_id",
            "capture_id",
            "site_submission_id",
            "status",
            "runtime_base_url",
            "websocket_base_url",
            "vm_instance_id",
            "cache_path",
            "conditioning_source_path",
            "seed_frame_path",
            "supported_cameras",
            "scenario_catalog",
            "start_state_catalog",
            "task_catalog",
            "robot_profiles",
            "runtime_capabilities",
            "health_uri",
            "generated_at",
            "blockers",
            "warnings",
            "grounding_status",
            "ungrounded_reason",
            "empty_index_cause",
            "object_index_backend_blockers",
            "canonical_package_version",
            "canonical_artifact_uri",
            "presentation_artifact_uri",
            "derivation_mode",
            "authoritative_record",
        )
        if key in registration_response_payload and registration_response_payload.get(key) is not None
    }
    registration.update(runtime_registration)
    health = dict(
        registration_response_payload.get("health")
        or client.get_site_world_health(str(registration.get("site_world_id") or site_world_id))
    )
    registration["blockers"] = list(
        dict.fromkeys(
            [
                *list(canonical_runtime_status.get("blockers") or []),
                *list(registration.get("blockers") or []),
            ]
        )
    )
    registration["warnings"] = list(
        dict.fromkeys(
            [
                *list(canonical_runtime_status.get("warnings") or []),
                *list(registration.get("warnings") or []),
            ]
        )
    )
    registration["runtime_capabilities"] = _runtime_capabilities_payload(
        launchable=True,
        base=registration.get("runtime_capabilities") if isinstance(registration.get("runtime_capabilities"), Mapping) else {},
    )
    if registration.get("status") == "ready":
        if task_catalog and scenario_catalog and start_state_catalog and robot_profiles:
            try:
                session = client.create_session(
                    str(registration["site_world_id"]),
                    robot_profile_id=str((robot_profiles[0] or {}).get("id") or "mobile_manipulator_rgb_v1"),
                    task_id=str((task_catalog[0] or {}).get("id") or (task_catalog[0] or {}).get("task_id") or ""),
                    scenario_id=str((scenario_catalog[0] or {}).get("id") or ""),
                    start_state_id=str((start_state_catalog[0] or {}).get("id") or ""),
                    notes="pipeline_runtime_smoke",
                )
                client.reset_session(str(session.get("session_id") or ""))
                health = dict(client.get_site_world_health(str(registration["site_world_id"])))
            except Exception as exc:
                health = {
                    **health,
                    "healthy": False,
                    "launchable": False,
                    "status": "degraded",
                    "blockers": list(health.get("blockers") or []) + [f"runtime_smoke_failed:{exc}"],
                    "last_heartbeat_at": utc_now_iso(),
                }
                health["runtime_capabilities"] = _runtime_capabilities_payload(launchable=False, base=health.get("runtime_capabilities"))
    registration.setdefault("blockers", list(canonical_runtime_status.get("blockers") or []))
    registration.setdefault("warnings", list(canonical_runtime_status.get("warnings") or []))
    registration.setdefault("task_catalog", task_catalog)
    registration.setdefault("scenario_catalog", scenario_catalog)
    registration.setdefault("start_state_catalog", start_state_catalog)
    registration.setdefault("robot_profiles", robot_profiles)
    registration.setdefault("supported_cameras", [])
    registration["primary_runtime_backend"] = spec.get("primary_runtime_backend")
    registration["canonical_world_model"] = dict(spec.get("canonical_world_model") or {})
    registration["world_model_backend"] = spec.get("world_model_backend")
    registration["scene_representation"] = spec.get("scene_representation")
    registration["render_source"] = spec.get("runtime_render_source")
    registration["fallback_mode"] = spec.get("fallback_mode")
    registration["grounding_status"] = canonical_runtime_status.get("grounding_status")
    registration["ungrounded_reason"] = canonical_runtime_status.get("ungrounded_reason")
    registration["empty_index_cause"] = canonical_runtime_status.get("empty_index_cause")
    registration["object_index_backend_blockers"] = list(canonical_runtime_status.get("object_index_backend_blockers") or [])
    registration.setdefault("runtime_registration_attempted", True)
    registration.setdefault("runtime_registration_status", "submitted")
    registration.setdefault(
        "runtime_registration_attempt",
        _runtime_registration_attempt(
            attempted=True,
            status=str(registration.get("runtime_registration_status") or "submitted"),
            reason=None,
            spec=spec,
            registration=registration,
            health=health,
        ),
    )
    health.setdefault("scene_id", context.scene_id)
    health.setdefault("capture_id", context.capture_id)
    health.setdefault("site_submission_id", spec.get("site_submission_id"))
    health.setdefault("blockers", list(canonical_runtime_status.get("blockers") or []))
    health.setdefault("warnings", list(canonical_runtime_status.get("warnings") or []))
    health.setdefault("task_catalog", task_catalog)
    health.setdefault("scenario_catalog", scenario_catalog)
    health.setdefault("start_state_catalog", start_state_catalog)
    health.setdefault("robot_profiles", robot_profiles)
    health.setdefault("supported_cameras", registration.get("supported_cameras") or [])
    health["primary_runtime_backend"] = spec.get("primary_runtime_backend")
    health["canonical_world_model"] = dict(spec.get("canonical_world_model") or {})
    health["world_model_backend"] = spec.get("world_model_backend")
    health["scene_representation"] = spec.get("scene_representation")
    health["render_source"] = spec.get("runtime_render_source")
    health["fallback_mode"] = spec.get("fallback_mode")
    health.setdefault("runtime_base_url", registration.get("runtime_base_url"))
    health.setdefault("websocket_base_url", registration.get("websocket_base_url"))
    health.setdefault("vm_instance_id", registration.get("vm_instance_id"))
    health.setdefault("world_model_policy", policy.to_dict())
    health.setdefault("canonical_package_version", spec.get("canonical_package_version"))
    health["grounding_status"] = canonical_runtime_status.get("grounding_status")
    health["ungrounded_reason"] = canonical_runtime_status.get("ungrounded_reason")
    health["empty_index_cause"] = canonical_runtime_status.get("empty_index_cause")
    health["object_index_backend_blockers"] = list(canonical_runtime_status.get("object_index_backend_blockers") or [])
    health.setdefault("canonical_artifact_uri", _gs_uri(context, "evaluation_prep/site_world_health.json"))
    health.setdefault("presentation_artifact_uri", _gs_uri(context, "presentation_world/runtime_demo_manifest.json") if policy.emit_presentation else None)
    health.setdefault("derivation_mode", policy.output_policy)
    health.setdefault("authoritative_record", True)
    health.setdefault("runtime_registration_attempted", registration.get("runtime_registration_attempted", True))
    health.setdefault(
        "runtime_registration_status",
        str(registration.get("runtime_registration_status") or "submitted"),
    )
    health.setdefault(
        "runtime_registration_attempt",
        dict(registration.get("runtime_registration_attempt") or {}),
    )
    health["runtime_capabilities"] = _runtime_capabilities_payload(
        launchable=bool(health.get("launchable", False)),
        base=health.get("runtime_capabilities")
        if isinstance(health.get("runtime_capabilities"), Mapping)
        else registration.get("runtime_capabilities")
        if isinstance(registration.get("runtime_capabilities"), Mapping)
        else {},
    )
    return registration, health


def _build_hosted_session_runtime_manifest(
    *,
    context,
    normalized_handoff: Mapping[str, Any],
    scene_memory_bundle_manifest: Mapping[str, Any],
    task_anchor_manifest: Mapping[str, Any],
    task_run_manifest: Mapping[str, Any],
    canonical_runtime_status: Mapping[str, Any],
    canonical_package_version: Optional[str] = None,
) -> Dict[str, Any]:
    policy = _policy()
    eval_dir = context.capture_root / "pipeline" / "evaluation_prep"
    descriptor_map = _read_json_object(context.descriptor_path)
    conditioning_bundle_path = _real_path_from_eval_dir(
        eval_dir, str(scene_memory_bundle_manifest.get("conditioning_bundle_path") or "")
    )
    conditioning_bundle = _read_optional_json_any(conditioning_bundle_path) if conditioning_bundle_path else {}
    conditioning_map = dict(conditioning_bundle) if isinstance(conditioning_bundle, Mapping) else {}
    capture_orientation = _descriptor_capture_orientation(descriptor_map, conditioning_map)
    canonical_world_model = _canonical_world_model_payload(
        context=context,
        capture_orientation=capture_orientation,
    )
    runtime_render_descriptor = _primary_runtime_render_descriptor(
        conditioning_bundle=conditioning_map,
        local_paths=_conditioning_local_paths(context=context, conditioning_bundle=conditioning_map),
        canonical_world_model=canonical_world_model,
    )
    adapter_key_map = {
        "neoverse": "neoverse_adapter_manifest_path",
        "gen3c": "gen3c_adapter_manifest_path",
        "cosmos_transfer": "cosmos_transfer_adapter_manifest_path",
    }
    adapter_details = _adapter_manifest_details(scene_memory_bundle_manifest, eval_dir=eval_dir)
    available_backends = [
        backend
        for backend, key in adapter_key_map.items()
        if str(scene_memory_bundle_manifest.get(key) or "").strip()
    ]
    launchable_backends = [
        backend
        for backend in available_backends
        if str(adapter_details.get(backend, {}).get("status") or "").strip().startswith("available_stage1_")
    ]
    preferred_order = ["neoverse", "gen3c", "cosmos_transfer"]
    default_backend = next((backend for backend in preferred_order if backend in launchable_backends), None)

    tasks = (
        task_anchor_manifest.get("tasks")
        if isinstance(task_anchor_manifest.get("tasks"), list)
        else []
    )
    task_ids = [
        str(task.get("task_id") or "")
        for task in tasks
        if isinstance(task, Mapping) and str(task.get("task_id") or "").strip()
    ]
    task_texts = [
        str(task.get("task_text") or "")
        for task in tasks
        if isinstance(task, Mapping) and str(task.get("task_text") or "").strip()
    ]
    task_catalog: List[Dict[str, Any]] = []
    start_state_catalog: List[Dict[str, Any]] = []
    start_states = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("task_id") or "").strip()
        task_text = str(task.get("task_text") or task.get("task_id") or "").strip()
        task_catalog.append(
            {
                "id": task_id or _stable_id("task", task_text, fallback="default"),
                "task_id": task_id,
                "task_text": task_text,
                "task_category": str(task.get("task_category") or "generic"),
                "target_object_ids": _string_list(task.get("target_object_ids")),
                "articulation_required_ids": _string_list(task.get("articulation_required_ids")),
                "task_critical": bool(
                    _string_list(task.get("target_object_ids"), task.get("articulation_required_ids"))
                ),
            }
        )
        if task_text and task_text not in start_states:
            start_states.append(task_text)
            start_state_catalog.append(
                {
                    "id": _stable_id("start", task_text, fallback=f"task_{len(start_state_catalog)}"),
                    "name": task_text,
                    "task_id": task_id or None,
                    "source": "task_anchor_manifest",
                }
            )
    if not start_states:
        for text in (
            task_run_manifest.get("start_states")
            if isinstance(task_run_manifest, Mapping)
            else []
        ) or []:
            name = str(text).strip()
            if not name or name in start_states:
                continue
            start_states.append(name)
            start_state_catalog.append(
                {
                    "id": _stable_id("start", name, fallback=f"state_{len(start_state_catalog)}"),
                    "name": name,
                    "task_id": None,
                    "source": "task_run_manifest",
                }
            )
    if not start_states:
        start_states = ["default_start_state"]
        start_state_catalog = [
            {
                "id": "start_default",
                "name": "default_start_state",
                "task_id": None,
                "source": "runtime_default",
            }
        ]

    scenario_variants = ["default", "counterfactual_lighting", "counterfactual_clutter"]
    if str(scene_memory_bundle_manifest.get("preview_simulation_manifest_path") or "").strip():
        scenario_variants.insert(0, "preview_simulation_default")
    scenario_catalog = [
        {
            "id": _stable_id("scenario", variant, fallback=f"scenario_{index}"),
            "name": variant,
            "source": "preview_simulation" if variant == "preview_simulation_default" else "hosted_runtime",
        }
        for index, variant in enumerate(scenario_variants)
    ]
    robot_profiles = _default_robot_profiles()
    export_defaults = [
        "observation_frames",
        "action_trace",
        "reward",
        "summary_metrics",
        "rollout_video",
        "rlds_dataset",
    ]
    runtime_capabilities = _runtime_capabilities_payload(
        launchable=True,
        base={
            "supports_step_rollout": True,
            "supports_batch_rollout": True,
            "supports_camera_views": True,
            "supports_rlds_export": True,
            "supports_preview_render": "gen3c" in available_backends,
            "protected_region_locking": True,
            "runtime_layer_compositing": True,
            "debug_render_outputs": True,
        },
    )

    blockers: List[str] = list(canonical_runtime_status.get("blockers") or [])
    if not task_ids:
        blockers.append("missing_task_anchor_manifest")
    if not available_backends:
        blockers.append("runtime_manifest_only")
    if available_backends and not launchable_backends:
        blockers.append("no_launchable_stage1_backend")

    return {
        "schema_version": "v1",
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "site_submission_id": str(
            normalized_handoff.get("site_submission_id") or context.capture_id
        ),
        "pipeline_prefix": f"{context.capture_prefix}/pipeline",
        "scene_memory_manifest_uri": _gs_uri(
            context, "scene_memory/scene_memory_manifest.json"
        ),
        "conditioning_bundle_uri": _gs_uri(
            context, "scene_memory/conditioning_bundle.json"
        ),
        "preview_simulation_manifest_uri": (
            _gs_uri(context, "preview_simulation/preview_simulation_manifest.json")
            if str(scene_memory_bundle_manifest.get("preview_simulation_manifest_path") or "").strip()
            else None
        ),
        "presentation_bundle_uri": (
            _gs_uri(context, "presentation_world/presentation_bundle.json")
            if str(scene_memory_bundle_manifest.get("presentation_bundle_path") or "").strip()
            else None
        ),
        "canonical_package_uri": _gs_uri(context, "evaluation_prep/site_world_spec.json"),
        "canonical_package_version": canonical_package_version,
        "capture_orientation": capture_orientation,
        "task_anchor_manifest_uri": _gs_uri(
            context, "evaluation_prep/task_anchor_manifest.json"
        ),
        "task_run_manifest_uri": _gs_uri(
            context, "evaluation_prep/task_run_manifest.json"
        ),
        "protected_regions_manifest_uri": _gs_uri(
            context, "evaluation_prep/protected_regions_manifest.json"
        ),
        "canonical_render_policy_uri": _gs_uri(
            context, "evaluation_prep/canonical_render_policy.json"
        ),
        "presentation_variance_policy_uri": _gs_uri(
            context, "evaluation_prep/presentation_variance_policy.json"
        ),
        "primary_runtime_backend": "neoverse",
        "canonical_world_model": canonical_world_model,
        "world_model_backend": runtime_render_descriptor["world_model_backend"],
        "scene_representation": runtime_render_descriptor["scene_representation"],
        "render_source": runtime_render_descriptor["runtime_render_source"],
        "fallback_mode": runtime_render_descriptor["fallback_mode"],
        "available_backends": available_backends,
        "launchable_backends": launchable_backends,
        "default_backend": default_backend,
        "customer_facing_runtime": (
            "Hosted site runtime"
            if default_backend and not blockers
            else "Qualified site package"
        ),
        "task_ids": task_ids,
        "task_texts": task_texts,
        "task_catalog": task_catalog,
        "start_states": start_states,
        "start_state_catalog": start_state_catalog,
        "scenario_variants": scenario_variants,
        "scenario_catalog": scenario_catalog,
        "robot_profiles": robot_profiles,
        "default_robot_profile_id": robot_profiles[0]["id"],
        "export_defaults": export_defaults,
        "runtime_capabilities": runtime_capabilities,
        "supports_step_rollout": runtime_capabilities["supports_step_rollout"],
        "supports_batch_rollout": runtime_capabilities["supports_batch_rollout"],
        "supports_camera_views": runtime_capabilities["supports_camera_views"],
        "grounding_status": canonical_runtime_status.get("grounding_status"),
        "ungrounded_reason": canonical_runtime_status.get("ungrounded_reason"),
        "empty_index_cause": canonical_runtime_status.get("empty_index_cause"),
        "object_index_backend_blockers": list(canonical_runtime_status.get("object_index_backend_blockers") or []),
        "launchable": len(blockers) == 0,
        "blockers": blockers,
        "launch_blockers": list(blockers),
        "adapter_manifest_uris": {
            backend: _gs_uri(
                context, f"scene_memory/adapter_manifests/{backend}.json"
            )
            for backend in available_backends
        },
        "backend_launch_requirements": {
            backend: {
                "status": adapter_details.get(backend, {}).get("status"),
                "execution_mode": adapter_details.get(backend, {}).get("execution_mode"),
                "required_conditioning": adapter_details.get(backend, {}).get("required_conditioning", []),
                "service_contract_version": adapter_details.get(backend, {}).get("service_contract_version"),
            }
            for backend in available_backends
        },
        "world_model_policy": policy.to_dict(),
        "canonical_artifact_uri": _gs_uri(context, "evaluation_prep/site_world_spec.json"),
        "presentation_artifact_uri": _gs_uri(context, "presentation_world/runtime_demo_manifest.json") if policy.emit_presentation else None,
        "derivation_mode": policy.allow_generative_completion,
        "authoritative_record": False,
        "generated_at": utc_now_iso(),
    }


def _build_site_normalization_package(
    *,
    context,
    normalized_handoff: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
    scope_record: Mapping[str, Any],
    scene_memory_bundle_manifest: Mapping[str, Any],
    geometry_bundle_manifest: Mapping[str, Any],
    object_geometry_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    geometry_objects = (
        object_geometry_manifest.get("objects")
        if isinstance(object_geometry_manifest.get("objects"), list)
        else []
    )
    measurements = (
        qualification_record.get("measurements")
        if isinstance(qualification_record.get("measurements"), Mapping)
        else {}
    )
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "site_submission_id": str(normalized_handoff.get("site_submission_id") or context.capture_id),
        "opportunity_id": str(normalized_handoff.get("opportunity_id") or context.scene_id),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "qualification_state": str(normalized_handoff.get("qualification_state") or "not_ready_yet"),
        "downstream_evaluation_eligibility": bool(
            normalized_handoff.get("downstream_evaluation_eligibility")
        ),
        "summary": str(
            normalized_handoff.get("operator_approved_summary")
            or qualification_record.get("summary")
            or f"Qualified site package for {context.scene_id}"
        ).strip(),
        "scoped_task_definition": dict(
            normalized_handoff.get("scoped_task_definition")
            if isinstance(normalized_handoff.get("scoped_task_definition"), Mapping)
            else {}
        ),
        "site_constraints": dict(
            normalized_handoff.get("site_constraints")
            if isinstance(normalized_handoff.get("site_constraints"), Mapping)
            else {}
        ),
        "measurements": dict(measurements) if isinstance(measurements, Mapping) else {},
        "scene_memory_bundle_status": str(scene_memory_bundle_manifest.get("status") or "missing"),
        "geometry_bundle_status": str(geometry_bundle_manifest.get("status") or "missing"),
        "object_count": len([item for item in geometry_objects if isinstance(item, Mapping)]),
        "rights_and_compliance": {
            "consent_scope": _string_list(
                normalized_handoff.get("capture_rights_scope"),
                normalized_handoff.get("site_constraints", {}).get("privacy_security_constraints")
                if isinstance(normalized_handoff.get("site_constraints"), Mapping)
                else [],
            ),
            "export_entitlements": _string_list(
                normalized_handoff.get("allowed_exports"),
                "scene_memory" if scene_memory_bundle_manifest.get("status") == "complete" else "",
                "geometry_bundle" if geometry_bundle_manifest.get("status") != "missing" else "",
            ),
            "customer_specific_sharing": _string_list(
                normalized_handoff.get("customer_specific_sharing")
            ),
            "audit_trail_uri": None,
            "retention_policy": str(normalized_handoff.get("retention_policy") or "").strip() or None,
        },
        "authoritative_sources": {
            "qualification_record_path": "qualification_record.json",
            "task_scope_record_path": "task_scope_record.json",
            "scene_memory_manifest_path": scene_memory_bundle_manifest.get("scene_memory_manifest_path"),
            "geometry_bundle_path": geometry_bundle_manifest.get("bundle_path"),
        },
    }


def _build_benchmark_suite_manifest(
    *,
    normalized_handoff: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
    task_anchor_manifest: Mapping[str, Any],
    task_run_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    tasks = (
        task_anchor_manifest.get("tasks")
        if isinstance(task_anchor_manifest.get("tasks"), list)
        else []
    )
    success_criteria = []
    scoped = (
        normalized_handoff.get("scoped_task_definition")
        if isinstance(normalized_handoff.get("scoped_task_definition"), Mapping)
        else {}
    )
    if isinstance(scoped, Mapping):
        success_criteria = _string_list(scoped.get("success_criteria"))
    risks = (
        qualification_record.get("risks")
        if isinstance(qualification_record.get("risks"), list)
        else []
    )
    edge_case_hints = [
        str(item.get("detail") or "").strip()
        for item in risks
        if isinstance(item, Mapping) and str(item.get("detail") or "").strip()
    ]
    benchmark_tasks: List[Dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        benchmark_tasks.append(
            {
                "task_id": str(task.get("task_id") or ""),
                "task_text": str(task.get("task_text") or ""),
                "task_category": str(task.get("task_category") or "generic"),
                "start_state_candidates": _string_list(
                    task.get("task_text"),
                    task.get("task_id"),
                ),
                "pass_criteria": success_criteria or ["Complete the scoped task safely."],
                "edge_case_hints": edge_case_hints[:5],
                "target_object_ids": _string_list(task.get("target_object_ids")),
                "articulation_required_ids": _string_list(task.get("articulation_required_ids")),
            }
        )
    task_categories = sorted(
        {
            str(task.get("task_category") or "generic")
            for task in benchmark_tasks
            if isinstance(task, Mapping)
        }
    )
    default_start_states = _string_list(
        task_run_manifest.get("start_states") if isinstance(task_run_manifest, Mapping) else [],
        [
            item.get("task_text")
            for item in benchmark_tasks
            if isinstance(item, Mapping) and str(item.get("task_text") or "").strip()
        ],
    )
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": "ready" if benchmark_tasks else "missing",
        "task_count": len(benchmark_tasks),
        "task_categories": task_categories,
        "default_start_states": default_start_states,
        "tasks": benchmark_tasks,
    }


def _build_compatibility_matrix(
    *,
    qualification_record: Mapping[str, Any],
    task_anchor_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    measurements = (
        qualification_record.get("measurements")
        if isinstance(qualification_record.get("measurements"), Mapping)
        else {}
    )
    minimum_route_width = float(measurements.get("minimum_route_width_m") or 0.0)
    maximum_reach = float(measurements.get("maximum_target_reach_m") or 0.0)
    reference_capability_envelope = {
        "embodiment_type": "vendor_neutral",
        "minimum_path_width_m": minimum_route_width or 0.95,
        "maximum_reach_m": maximum_reach or 1.1,
        "maximum_payload_kg": None,
        "sensor_requirements": ["rgb"],
        "controller_interface_assumptions": ["checkpoint + bounded rollout interface"],
        "safety_envelope": ["respect restricted zones", "bounded task scope"],
        "facility_constraints": _string_list(
            qualification_record.get("blockers"),
        ),
    }
    robot_classes = [
        {
            "robot_class": "humanoid",
            "fit": "fit" if minimum_route_width >= 1.05 or minimum_route_width == 0.0 else "conditional",
            "reason": "Humanoid class benefits from wider route clearance and bounded handoff zones.",
        },
        {
            "robot_class": "mobile_manipulator",
            "fit": "fit" if minimum_route_width >= 0.95 or minimum_route_width == 0.0 else "conditional",
            "reason": "Mobile manipulator class is the default neutral target for this site package.",
        },
        {
            "robot_class": "fixed_arm",
            "fit": "fit" if maximum_reach <= 1.1 or maximum_reach == 0.0 else "conditional",
            "reason": "Fixed-arm evaluation depends on reach to target objects and workcell access.",
        },
        {
            "robot_class": "cart_tug",
            "fit": "fit" if minimum_route_width >= 1.15 or minimum_route_width == 0.0 else "not_recommended",
            "reason": "Cart tug class benefits from wider path widths and simpler route geometry.",
        },
    ]
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": "ready",
        "reference_capability_envelope": reference_capability_envelope,
        "task_count": len(
            task_anchor_manifest.get("tasks")
            if isinstance(task_anchor_manifest.get("tasks"), list)
            else []
        ),
        "robot_classes": robot_classes,
    }


def _find_previous_site_normalization_path(*, capture_root: Path, current_capture_id: str) -> Optional[Path]:
    captures_root = capture_root.parent
    candidates: List[Path] = []
    for sibling in captures_root.glob("*/pipeline/evaluation_prep/site_normalization_package.json"):
        if sibling.parts[-4] == current_capture_id:
            continue
        candidates.append(sibling)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _build_recapture_diff(
    *,
    capture_root: Path,
    current_capture_id: str,
    site_normalization_package: Mapping[str, Any],
    benchmark_suite_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    previous_path = _find_previous_site_normalization_path(
        capture_root=capture_root,
        current_capture_id=current_capture_id,
    )
    if previous_path is None:
        return {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "status": "no_prior_baseline",
            "recapture_required": False,
            "changed_fields": [],
            "previous_capture_id": None,
        }

    previous = read_json_any(previous_path)
    scoped = (
        site_normalization_package.get("scoped_task_definition")
        if isinstance(site_normalization_package.get("scoped_task_definition"), Mapping)
        else {}
    )
    previous_scoped = (
        previous.get("scoped_task_definition")
        if isinstance(previous, Mapping) and isinstance(previous.get("scoped_task_definition"), Mapping)
        else {}
    )
    changed_fields: List[str] = []
    comparisons = {
        "qualification_state": (
            site_normalization_package.get("qualification_state"),
            previous.get("qualification_state") if isinstance(previous, Mapping) else None,
        ),
        "task_statement": (
            scoped.get("scoped_task_statement") if isinstance(scoped, Mapping) else None,
            previous_scoped.get("scoped_task_statement") if isinstance(previous_scoped, Mapping) else None,
        ),
        "known_blockers": (
            site_normalization_package.get("site_constraints"),
            previous.get("site_constraints") if isinstance(previous, Mapping) else None,
        ),
        "minimum_route_width_m": (
            site_normalization_package.get("measurements"),
            previous.get("measurements") if isinstance(previous, Mapping) else None,
        ),
    }
    for field_name, (current_value, previous_value) in comparisons.items():
        if current_value != previous_value:
            changed_fields.append(field_name)

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": "changed" if changed_fields else "unchanged",
        "recapture_required": bool(changed_fields),
        "changed_fields": changed_fields,
        "benchmark_task_count": benchmark_suite_manifest.get("task_count"),
        "previous_capture_id": (
            str(previous.get("capture_id") or "").strip()
            if isinstance(previous, Mapping)
            else None
        ),
    }


def _build_launchable_export_bundle(
    *,
    scene_memory_bundle_manifest: Mapping[str, Any],
    geometry_bundle_manifest: Mapping[str, Any],
    site_world_registration: Mapping[str, Any],
    site_world_health: Mapping[str, Any],
    runtime_demo_manifest: Mapping[str, Any],
    simready_prep_manifest_path: Optional[Path],
) -> Dict[str, Any]:
    runtime_capabilities = (
        site_world_registration.get("runtime_capabilities")
        if isinstance(site_world_registration.get("runtime_capabilities"), Mapping)
        else {}
    )
    demo_readiness = _presentation_demo_readiness(runtime_demo_manifest)
    bundles = {
        "world_model_runtime": {
            "launchable": bool(site_world_health.get("launchable")),
            "required_artifacts": [
                "scene_memory_manifest",
                "conditioning_bundle",
                "site_world_spec",
                "site_world_registration",
                "site_world_health",
            ],
            "backend": "neoverse",
        },
        "geometry_conditioning": {
            "launchable": bool(scene_memory_bundle_manifest.get("geometry_summary_path")),
            "required_artifacts": [
                "geometry_manifest",
                "geometry_summary",
                "geometry_poses",
                "geometry_intrinsics",
                "geometry_depth_manifest",
            ],
            "backend": "geometry_lane",
        },
        "isaac_sim": {
            "launchable": simready_prep_manifest_path is not None,
            "required_artifacts": ["simready_scene_manifest", "validation_manifest"],
            "backend": "isaac_sim",
        },
        "mujoco_robosuite": {
            "launchable": str(geometry_bundle_manifest.get("status") or "") in {"complete", "partial"},
            "required_artifacts": ["geometry_bundle", "task_hints"],
            "backend": "mujoco_robosuite",
        },
        "presentation_demo_ui": {
            "launchable": demo_readiness["readiness_state"] == "ready",
            "required_artifacts": ["presentation_bundle", "runtime_demo_manifest", "interactive_demo.readiness_state"],
            "backend": "neoverse_gradio",
            "blockers": demo_readiness["blockers"],
        },
    }
    full_export_ready = any(
        item["launchable"]
        for name, item in bundles.items()
        if name != "geometry_conditioning"
    )
    partial_ready = full_export_ready or bool(bundles["geometry_conditioning"]["launchable"])
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": "ready" if full_export_ready else "partial" if partial_ready else "partial",
        "public_runtime_label": "NeoVerse site world runtime",
        "default_backend": "neoverse",
        "scenario_variants": [
            str(item.get("name") or "")
            for item in site_world_registration.get("scenario_catalog", [])
            if isinstance(item, Mapping)
        ],
        "runtime_capabilities": dict(runtime_capabilities) if isinstance(runtime_capabilities, Mapping) else {},
        "site_world_status": site_world_health.get("status"),
        "bundles": bundles,
        "scene_memory_bundle_status": scene_memory_bundle_manifest.get("status"),
        "geometry_bundle_status": geometry_bundle_manifest.get("status"),
    }


def _world_model_validation_summary(
    *,
    policy: WorldModelPolicy,
    site_world_health: Mapping[str, Any],
    launchable_export_bundle: Mapping[str, Any],
    runtime_demo_manifest: Mapping[str, Any],
    object_geometry_manifest: Mapping[str, Any],
    geometry_bundle_manifest: Mapping[str, Any],
    scene_memory_bundle_manifest: Mapping[str, Any],
    task_anchor_manifest: Mapping[str, Any],
    review_queue: Mapping[str, Any],
) -> Dict[str, Any]:
    objects = (
        object_geometry_manifest.get("objects")
        if isinstance(object_geometry_manifest.get("objects"), list)
        else []
    )
    object_count = len([item for item in objects if isinstance(item, Mapping)])
    object_index_nonempty = object_count > 0
    conditioning_blockers = [
        str(item)
        for item in site_world_health.get("blockers", [])
        if str(item).startswith("missing_spatial_conditioning:")
        or str(item).startswith("missing_local_conditioning:")
    ]
    runtime_demo_ready = bool(site_world_health.get("launchable")) or bool(
        ((launchable_export_bundle.get("bundles") or {}).get("world_model_runtime") or {}).get("launchable")
    )
    demo_readiness = _presentation_demo_readiness(runtime_demo_manifest)
    presentation_demo_ui_ready = demo_readiness["readiness_state"] == "ready"
    grounding_quality_ready = (
        object_index_nonempty
        and not conditioning_blockers
        and _object_geometry_has_provenance(object_geometry_manifest)
        and str(scene_memory_bundle_manifest.get("status") or "") == "complete"
    )
    geometry_quality_ready = (
        object_index_nonempty
        and str(geometry_bundle_manifest.get("status") or "") in {"complete", "partial"}
    )
    geometry_conditioning_ready = bool(scene_memory_bundle_manifest.get("geometry_summary_path"))
    task_ids = {
        target_id
        for task in task_anchor_manifest.get("tasks", [])
        if isinstance(task, Mapping)
        for target_id in _string_list(task.get("target_object_ids"))
    }
    geometry_ids = {
        str(item.get("object_id") or "")
        for item in objects
        if isinstance(item, Mapping) and str(item.get("object_id") or "").strip()
    }
    task_representation_ready = not task_ids or task_ids.issubset(geometry_ids)
    unresolved_high_risks = [
        item
        for item in review_queue.get("items", [])
        if isinstance(item, Mapping) and str(item.get("severity") or "").lower() == "high"
    ]

    validation_gates = {
        "runtime_demo_ready": {
            "passed": runtime_demo_ready,
            "detail": "Runnable/demo runtime or runtime-adjacent package is available.",
        },
        "presentation_demo_ui_ready": {
            "passed": presentation_demo_ui_ready,
            "detail": (
                "Presentation demo contract is bundle-backed and includes a truthful interactive demo endpoint."
                if presentation_demo_ui_ready
                else f"Presentation demo contract is blocked: {', '.join(demo_readiness['blockers']) or 'unknown'}."
            ),
        },
        "grounding_quality_ready": {
            "passed": grounding_quality_ready,
            "detail": "Canonical package has non-empty grounded object geometry and no unresolved conditioning blockers.",
        },
        "geometry_quality_ready": {
            "passed": geometry_quality_ready,
            "detail": "Geometry package is available at partial-or-better quality.",
        },
        "geometry_conditioning_ready": {
            "passed": geometry_conditioning_ready,
            "detail": "Geometry-lane conditioning summary and camera/depth artifacts are available."
            if geometry_conditioning_ready
            else "Geometry-lane conditioning is missing or incomplete.",
        },
        "task_representation_ready": {
            "passed": task_representation_ready,
            "detail": "Task anchors resolve against canonical geometry objects.",
        },
    }

    if (
        validation_gates["runtime_demo_ready"]["passed"]
        and validation_gates["presentation_demo_ui_ready"]["passed"]
        and validation_gates["grounding_quality_ready"]["passed"]
        and validation_gates["geometry_quality_ready"]["passed"]
        and validation_gates["task_representation_ready"]["passed"]
        and not unresolved_high_risks
        and bool(site_world_health.get("launchable"))
    ):
        classification = "validated_site_world"
    elif (
        validation_gates["runtime_demo_ready"]["passed"]
        and validation_gates["grounding_quality_ready"]["passed"]
    ):
        classification = "grounded_world_model"
    else:
        classification = "prototype_demo"

    return {
        "world_model_classification": classification,
        "validation_gates": validation_gates,
        "unresolved_high_risk_count": len(unresolved_high_risks),
        "policy": policy.to_dict(),
    }


def run_evaluation_prep_stage(
    *,
    capture_root: str | Path,
    provider_name: str = "manual",
) -> Dict[str, Any]:
    policy = _policy()
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    eval_dir = pipeline_dir / "evaluation_prep"
    ensure_dir(eval_dir)

    handoff = optional_read_json(pipeline_dir / "opportunity_handoff.json")
    if handoff is None:
        raise PipelineError(f"Missing opportunity_handoff.json at {pipeline_dir}")
    qualification_record = optional_read_json(pipeline_dir / "qualification_record.json") or {}
    scope_record = optional_read_json(pipeline_dir / "task_scope_record.json") or {}

    object_geometry_manifest = _resolve_object_geometry_manifest(
        context=context,
        provider_name=provider_name,
    )
    object_geometry_target_path = eval_dir / "object_geometry_manifest.json"
    _copy_json(
        object_geometry_target_path,
        object_geometry_manifest if isinstance(object_geometry_manifest, Mapping) else {},
    )

    geometry_bundle_manifest = _build_geometry_bundle_manifest(pipeline_dir=pipeline_dir, eval_dir=eval_dir)
    geometry_bundle_manifest_path = eval_dir / "geometry_bundle_manifest.json"
    _copy_json(geometry_bundle_manifest_path, geometry_bundle_manifest)
    scene_memory_bundle_manifest = _build_scene_memory_bundle_manifest(
        pipeline_dir=pipeline_dir,
        eval_dir=eval_dir,
    )
    scene_memory_bundle_manifest_path = eval_dir / "scene_memory_bundle_manifest.json"
    _copy_json(scene_memory_bundle_manifest_path, scene_memory_bundle_manifest)

    normalized_handoff = _normalize_rich_handoff(
        handoff=handoff,
        scope_record=scope_record,
        qualification_record=qualification_record,
        capture_root=context.capture_root,
        geometry_bundle_manifest=geometry_bundle_manifest,
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
    )
    rich_handoff_path = eval_dir / "qualified_opportunity_handoff.json"
    _copy_json(rich_handoff_path, normalized_handoff)

    existing_task_run_manifest = _read_optional_json_any(pipeline_dir / "task_run_manifest.json")
    if isinstance(existing_task_run_manifest, Mapping):
        task_run_manifest = dict(existing_task_run_manifest)
        task_run_entries = _load_task_run_entries(context.capture_root)
    else:
        task_run_manifest = _build_default_task_run_manifest(
            capture_root=context.capture_root,
            handoff=normalized_handoff,
            scope_record=scope_record,
        )
        task_run_entries = []
    task_run_manifest_path = eval_dir / "task_run_manifest.json"
    _copy_json(task_run_manifest_path, task_run_manifest)

    task_anchor_manifest = _build_task_anchor_manifest(
        capture_root=context.capture_root,
        handoff=normalized_handoff,
        scope_record=scope_record,
        task_run_entries=task_run_entries,
        object_geometry_manifest=object_geometry_manifest if isinstance(object_geometry_manifest, Mapping) else {},
    )
    task_anchor_manifest_path = eval_dir / "task_anchor_manifest.json"
    _copy_json(task_anchor_manifest_path, task_anchor_manifest)

    protected_regions_manifest = build_protected_regions_manifest(
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        object_geometry_manifest=object_geometry_manifest if isinstance(object_geometry_manifest, Mapping) else {},
        task_anchor_manifest=task_anchor_manifest,
    )
    protected_regions_manifest_path = eval_dir / "protected_regions_manifest.json"
    _copy_json(protected_regions_manifest_path, protected_regions_manifest)

    canonical_render_policy = build_canonical_render_policy()
    canonical_render_policy_path = eval_dir / "canonical_render_policy.json"
    _copy_json(canonical_render_policy_path, canonical_render_policy)

    presentation_variance_policy = build_presentation_variance_policy()
    presentation_derivation_policy = build_presentation_derivation_policy(
        policy=policy,
        variance_policy=presentation_variance_policy,
    )
    presentation_variance_policy_path = eval_dir / "presentation_variance_policy.json"
    _copy_json(presentation_variance_policy_path, presentation_variance_policy)

    runtime_service_url = (os.getenv("NEOVERSE_RUNTIME_SERVICE_URL") or "").strip().rstrip("/")
    canonical_runtime_status = _canonical_site_world_runtime_status(
        qualification_state=normalized_handoff.get("qualification_state"),
        downstream_evaluation_eligibility=bool(normalized_handoff.get("downstream_evaluation_eligibility")),
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
        object_geometry_manifest=object_geometry_manifest if isinstance(object_geometry_manifest, Mapping) else {},
        protected_regions_manifest=protected_regions_manifest,
        required_runtime_artifact_paths=[
            protected_regions_manifest_path,
            canonical_render_policy_path,
            presentation_variance_policy_path,
        ],
        runtime_service_url=runtime_service_url,
    )

    site_world_spec = _build_site_world_spec(
        context=context,
        eval_dir=eval_dir,
        normalized_handoff=normalized_handoff,
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
        object_geometry_manifest=object_geometry_manifest if isinstance(object_geometry_manifest, Mapping) else {},
        task_anchor_manifest=task_anchor_manifest,
        task_run_manifest=task_run_manifest,
        protected_regions_manifest=protected_regions_manifest,
        canonical_render_policy=canonical_render_policy,
        presentation_variance_policy=presentation_variance_policy,
        canonical_runtime_status=canonical_runtime_status,
        canonical_package_version=None,
    )
    scene_memory_manifest_path = pipeline_dir / "scene_memory" / "scene_memory_manifest.json"
    conditioning_bundle_path = pipeline_dir / "scene_memory" / "conditioning_bundle.json"
    scene_memory_manifest = _read_json_object(scene_memory_manifest_path)
    conditioning_bundle = _read_json_object(conditioning_bundle_path)
    canonical_package_version = compute_canonical_package_version(
        scene_memory_manifest=_canonical_hash_payload(scene_memory_manifest),
        conditioning_bundle=_canonical_hash_payload(conditioning_bundle),
        object_geometry_manifest=_canonical_hash_payload(
            object_geometry_manifest if isinstance(object_geometry_manifest, Mapping) else {}
        ),
        task_anchor_manifest=_canonical_hash_payload(task_anchor_manifest),
        site_world_spec=_canonical_hash_payload(site_world_spec),
        protected_regions_manifest=_canonical_hash_payload(protected_regions_manifest),
        canonical_render_policy=_canonical_hash_payload(canonical_render_policy),
        presentation_variance_policy=_canonical_hash_payload(presentation_variance_policy),
    )
    site_world_spec["canonical_package_version"] = canonical_package_version
    site_world_spec_path = eval_dir / "site_world_spec.json"
    _copy_json(site_world_spec_path, site_world_spec)

    presentation_bundle_path = pipeline_dir / "presentation_world" / "presentation_bundle.json"
    presentation_world_manifest_path = pipeline_dir / "presentation_world" / "presentation_world_manifest.json"
    runtime_demo_manifest_path = pipeline_dir / "presentation_world" / "runtime_demo_manifest.json"
    for path in (
        scene_memory_manifest_path,
        conditioning_bundle_path,
        presentation_bundle_path,
        presentation_world_manifest_path,
        runtime_demo_manifest_path,
    ):
        payload = _read_json_object(path)
        if payload:
            write_json(
                path,
                _refresh_presentation_contract_payload(
                    payload=payload,
                    context=context,
                    canonical_package_version=canonical_package_version,
                    derivation_policy=presentation_derivation_policy,
                )
                if path in {
                    presentation_bundle_path,
                    presentation_world_manifest_path,
                    runtime_demo_manifest_path,
                }
                else {**payload, "canonical_package_version": canonical_package_version},
            )

    scene_memory_bundle_manifest = _build_scene_memory_bundle_manifest(
        pipeline_dir=pipeline_dir,
        eval_dir=eval_dir,
    )
    normalized_handoff = _normalize_rich_handoff(
        handoff=handoff,
        scope_record=scope_record,
        qualification_record=qualification_record,
        capture_root=context.capture_root,
        geometry_bundle_manifest=geometry_bundle_manifest,
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
    )
    canonical_runtime_status = _canonical_site_world_runtime_status(
        qualification_state=normalized_handoff.get("qualification_state"),
        downstream_evaluation_eligibility=bool(normalized_handoff.get("downstream_evaluation_eligibility")),
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
        object_geometry_manifest=object_geometry_manifest if isinstance(object_geometry_manifest, Mapping) else {},
        protected_regions_manifest=protected_regions_manifest,
        required_runtime_artifact_paths=[
            protected_regions_manifest_path,
            canonical_render_policy_path,
            presentation_variance_policy_path,
        ],
        runtime_service_url=runtime_service_url,
    )
    site_world_spec = _build_site_world_spec(
        context=context,
        eval_dir=eval_dir,
        normalized_handoff=normalized_handoff,
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
        object_geometry_manifest=object_geometry_manifest if isinstance(object_geometry_manifest, Mapping) else {},
        task_anchor_manifest=task_anchor_manifest,
        task_run_manifest=task_run_manifest,
        protected_regions_manifest=protected_regions_manifest,
        canonical_render_policy=canonical_render_policy,
        presentation_variance_policy=presentation_variance_policy,
        canonical_runtime_status=canonical_runtime_status,
        canonical_package_version=None,
    )
    canonical_package_version = compute_canonical_package_version(
        scene_memory_manifest=_canonical_hash_payload(_read_json_object(scene_memory_manifest_path)),
        conditioning_bundle=_canonical_hash_payload(_read_json_object(conditioning_bundle_path)),
        object_geometry_manifest=_canonical_hash_payload(
            object_geometry_manifest if isinstance(object_geometry_manifest, Mapping) else {}
        ),
        task_anchor_manifest=_canonical_hash_payload(task_anchor_manifest),
        site_world_spec=_canonical_hash_payload(site_world_spec),
        protected_regions_manifest=_canonical_hash_payload(protected_regions_manifest),
        canonical_render_policy=_canonical_hash_payload(canonical_render_policy),
        presentation_variance_policy=_canonical_hash_payload(presentation_variance_policy),
    )
    site_world_spec["canonical_package_version"] = canonical_package_version
    _copy_json(site_world_spec_path, site_world_spec)
    scene_memory_bundle_manifest["canonical_package_version"] = canonical_package_version
    _copy_json(scene_memory_bundle_manifest_path, scene_memory_bundle_manifest)
    for path in (
        scene_memory_manifest_path,
        conditioning_bundle_path,
        presentation_bundle_path,
        presentation_world_manifest_path,
        runtime_demo_manifest_path,
    ):
        payload = _read_json_object(path)
        if payload:
            write_json(
                path,
                _refresh_presentation_contract_payload(
                    payload=payload,
                    context=context,
                    canonical_package_version=canonical_package_version,
                    derivation_policy=presentation_derivation_policy,
                )
                if path in {
                    presentation_bundle_path,
                    presentation_world_manifest_path,
                    runtime_demo_manifest_path,
                }
                else {**payload, "canonical_package_version": canonical_package_version},
            )
    normalized_handoff = _normalize_rich_handoff(
        handoff=handoff,
        scope_record=scope_record,
        qualification_record=qualification_record,
        capture_root=context.capture_root,
        geometry_bundle_manifest=geometry_bundle_manifest,
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
    )
    _copy_json(rich_handoff_path, normalized_handoff)

    hosted_session_runtime_manifest = _build_hosted_session_runtime_manifest(
        context=context,
        normalized_handoff=normalized_handoff,
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
        task_anchor_manifest=task_anchor_manifest,
        task_run_manifest=task_run_manifest,
        canonical_runtime_status=canonical_runtime_status,
        canonical_package_version=canonical_package_version,
    )
    hosted_session_runtime_manifest_path = eval_dir / "hosted_session_runtime_manifest.json"
    _copy_json(hosted_session_runtime_manifest_path, hosted_session_runtime_manifest)

    site_world_registration, site_world_health = _build_site_world_runtime_records(
        context=context,
        spec=site_world_spec,
        canonical_runtime_status=canonical_runtime_status,
    )
    site_world_registration.setdefault("canonical_package_version", canonical_package_version)
    site_world_health.setdefault("canonical_package_version", canonical_package_version)
    site_world_registration_path = eval_dir / "site_world_registration.json"
    _copy_json(site_world_registration_path, site_world_registration)
    site_world_health_path = eval_dir / "site_world_health.json"
    _copy_json(site_world_health_path, site_world_health)

    site_normalization_package = _build_site_normalization_package(
        context=context,
        normalized_handoff=normalized_handoff,
        qualification_record=qualification_record,
        scope_record=scope_record,
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
        geometry_bundle_manifest=geometry_bundle_manifest,
        object_geometry_manifest=object_geometry_manifest if isinstance(object_geometry_manifest, Mapping) else {},
    )
    site_normalization_package_path = eval_dir / "site_normalization_package.json"
    _copy_json(site_normalization_package_path, site_normalization_package)

    benchmark_suite_manifest = _build_benchmark_suite_manifest(
        normalized_handoff=normalized_handoff,
        qualification_record=qualification_record,
        task_anchor_manifest=task_anchor_manifest,
        task_run_manifest=task_run_manifest,
    )
    benchmark_suite_manifest_path = eval_dir / "benchmark_suite_manifest.json"
    _copy_json(benchmark_suite_manifest_path, benchmark_suite_manifest)

    compatibility_matrix = _build_compatibility_matrix(
        qualification_record=qualification_record,
        task_anchor_manifest=task_anchor_manifest,
    )
    compatibility_matrix_path = eval_dir / "compatibility_matrix.json"
    _copy_json(compatibility_matrix_path, compatibility_matrix)

    simready_prep_manifest_path = None
    simready_scene_manifest = _read_optional_json_any(pipeline_dir / "simready" / "simready_scene_manifest.json")
    simready_validation = _read_optional_json_any(pipeline_dir / "simready" / "simready_validation.json")
    if isinstance(simready_scene_manifest, Mapping):
        simready_prep_manifest = {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "scene_manifest_path": _relative_to(eval_dir, pipeline_dir / "simready" / "simready_scene_manifest.json"),
            "validation_path": _relative_to(eval_dir, pipeline_dir / "simready" / "simready_validation.json") if (pipeline_dir / "simready" / "simready_validation.json").is_file() else "",
            "status": str((simready_validation or {}).get("overall_status") or "unknown"),
        }
        simready_prep_manifest_path = eval_dir / "simready_prep_manifest.json"
        _copy_json(simready_prep_manifest_path, simready_prep_manifest)

    recapture_diff = _build_recapture_diff(
        capture_root=context.capture_root,
        current_capture_id=context.capture_id,
        site_normalization_package=site_normalization_package,
        benchmark_suite_manifest=benchmark_suite_manifest,
    )
    recapture_diff_path = eval_dir / "recapture_diff.json"
    _copy_json(recapture_diff_path, recapture_diff)

    launchable_export_bundle = _build_launchable_export_bundle(
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
        geometry_bundle_manifest=geometry_bundle_manifest,
        site_world_registration=site_world_registration,
        site_world_health=site_world_health,
        runtime_demo_manifest=_read_json_object(runtime_demo_manifest_path),
        simready_prep_manifest_path=simready_prep_manifest_path,
    )
    launchable_export_bundle_path = eval_dir / "launchable_export_bundle.json"
    _copy_json(launchable_export_bundle_path, launchable_export_bundle)

    review_queue = _build_review_queue(
        object_geometry_manifest=object_geometry_manifest if isinstance(object_geometry_manifest, Mapping) else {},
        task_anchor_manifest=task_anchor_manifest,
        simready_validation=simready_validation if isinstance(simready_validation, Mapping) else None,
        geometry_bundle_manifest=geometry_bundle_manifest,
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
    )
    review_queue_path = eval_dir / "review_queue.json"
    _copy_json(review_queue_path, review_queue)

    validation_summary = _world_model_validation_summary(
        policy=policy,
        site_world_health=site_world_health,
        launchable_export_bundle=launchable_export_bundle,
        runtime_demo_manifest=_read_json_object(runtime_demo_manifest_path),
        object_geometry_manifest=object_geometry_manifest if isinstance(object_geometry_manifest, Mapping) else {},
        geometry_bundle_manifest=geometry_bundle_manifest,
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
        task_anchor_manifest=task_anchor_manifest,
        review_queue=review_queue,
    )
    site_world_registration["world_model_classification"] = validation_summary["world_model_classification"]
    site_world_registration["validation_gates"] = validation_summary["validation_gates"]
    site_world_health["world_model_classification"] = validation_summary["world_model_classification"]
    site_world_health["validation_gates"] = validation_summary["validation_gates"]
    _copy_json(site_world_registration_path, site_world_registration)
    _copy_json(site_world_health_path, site_world_health)
    launchable_export_bundle["world_model_classification"] = validation_summary["world_model_classification"]
    launchable_export_bundle["validation_gates"] = validation_summary["validation_gates"]
    _copy_json(launchable_export_bundle_path, launchable_export_bundle)

    geometry_objects = object_geometry_manifest.get("objects") if isinstance(object_geometry_manifest, Mapping) and isinstance(object_geometry_manifest.get("objects"), list) else []
    object_count = len([item for item in geometry_objects if isinstance(item, Mapping)])
    mesh_count = sum(1 for item in geometry_objects if isinstance(item, Mapping) and Path(str(item.get("mesh_glb_path") or "")).is_file())
    mask_count = sum(1 for item in geometry_objects if isinstance(item, Mapping) and any(isinstance(mask, Mapping) and str(mask.get("mask_path") or "") for mask in item.get("visual_replacement_masks", [])))
    articulated_count = sum(1 for item in geometry_objects if isinstance(item, Mapping) and str(item.get("task_role") or "") == "required_fixture")
    downstream_risks = [str(item.get("kind") or "") for item in review_queue.get("items", []) if isinstance(item, Mapping)]
    summary = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "task_count": len(task_anchor_manifest.get("tasks", [])) if isinstance(task_anchor_manifest.get("tasks"), list) else 0,
        "object_count": object_count,
        "geometry_coverage_ratio": round(mesh_count / float(object_count or 1), 4),
        "view_mask_coverage_ratio": round(mask_count / float(object_count or 1), 4),
        "articulation_count": articulated_count,
        "known_downstream_risks": downstream_risks,
        "benchmark_suite_status": benchmark_suite_manifest.get("status"),
        "compatibility_matrix_status": compatibility_matrix.get("status"),
        "recapture_diff_status": recapture_diff.get("status"),
        "export_bundle_status": launchable_export_bundle.get("status"),
        "site_world_status": site_world_health.get("status"),
        "geometry_conditioning_status": "available"
        if scene_memory_bundle_manifest.get("geometry_summary_path")
        else "missing",
        "world_model_classification": validation_summary["world_model_classification"],
        "validation_gates": validation_summary["validation_gates"],
        "canonical_package_version": canonical_package_version,
        "capture_orientation": site_world_spec.get("capture_orientation"),
    }
    summary_path = eval_dir / "evaluation_prep_summary.json"
    _copy_json(summary_path, summary)

    qualification_state = str(normalized_handoff.get("qualification_state") or "not_ready_yet")
    eligibility = bool(normalized_handoff.get("downstream_evaluation_eligibility"))
    degradation_reasons: List[str] = []
    degradation_seen: set[str] = set()

    def _append_degradation(reason: Any) -> None:
        text = str(reason or "").strip()
        if text and text not in degradation_seen:
            degradation_seen.add(text)
            degradation_reasons.append(text)

    if qualification_state != "ready":
        _append_degradation(f"qualification_state:{qualification_state}")
    if not eligibility:
        _append_degradation("downstream_evaluation_eligibility:false")
    if scene_memory_bundle_manifest.get("status") != "complete":
        _append_degradation(f"scene_memory_bundle:{scene_memory_bundle_manifest.get('status')}")
    if (
        scene_memory_bundle_manifest.get("status") != "complete"
        and geometry_bundle_manifest.get("status") != "complete"
    ):
        _append_degradation(f"geometry_bundle:{geometry_bundle_manifest.get('status')}")
    if not bool(site_world_health.get("launchable")):
        for item in site_world_health.get("blockers", []):
            _append_degradation(item)
    if object_count == 0:
        _append_degradation("object_geometry:missing")
    empty_index_cause = str(object_geometry_manifest.get("empty_index_cause") or "").strip() if isinstance(object_geometry_manifest, Mapping) else ""
    if empty_index_cause:
        _append_degradation(f"empty_index_cause:{empty_index_cause}")
    if isinstance(object_geometry_manifest, Mapping):
        for blocker in _string_list(object_geometry_manifest.get("object_index_backend_blockers")):
            _append_degradation(blocker)
    # Legacy status values are kept for compatibility with existing consumers.
    legacy_status = "ready_for_validation"
    if qualification_state != "ready" or not eligibility:
        legacy_status = "not_ready_for_validation"
    elif degradation_reasons:
        legacy_status = "degraded_but_usable"
    canonical_package_status = (
        "registration_blocked"
        if not bool(canonical_runtime_status.get("launchable"))
        else "degraded_but_usable"
        if degradation_reasons
        else "ready_for_runtime_registration"
    )

    task_ids = [str(task.get("task_id") or "") for task in task_anchor_manifest.get("tasks", []) if isinstance(task, Mapping)]
    task_categories = sorted({str(task.get("task_category") or "generic") for task in task_anchor_manifest.get("tasks", []) if isinstance(task, Mapping)})
    runtime_registration_attempt = dict(site_world_registration.get("runtime_registration_attempt") or {})
    manifest = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "site_submission_id": str(normalized_handoff.get("site_submission_id") or ""),
        "opportunity_id": str(normalized_handoff.get("opportunity_id") or ""),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "qualification_state": qualification_state,
        "downstream_evaluation_eligibility": eligibility,
        "readiness_state": str(normalized_handoff.get("readiness_state") or qualification_state),
        "canonical_package_status": canonical_package_status,
        "world_model_classification": validation_summary["world_model_classification"],
        "validation_gates": validation_summary["validation_gates"],
        "canonical_package_version": canonical_package_version,
        "capture_orientation": site_world_spec.get("capture_orientation"),
        "geometry_conditioning_status": "available"
        if scene_memory_bundle_manifest.get("geometry_summary_path")
        else "missing",
        "grounding_status": protected_regions_manifest.get("grounding_status"),
        "ungrounded_reason": protected_regions_manifest.get("ungrounded_reason"),
        "empty_index_cause": object_geometry_manifest.get("empty_index_cause")
        if isinstance(object_geometry_manifest, Mapping)
        else None,
        "object_index_backend_blockers": (
            _string_list(object_geometry_manifest.get("object_index_backend_blockers"))
            if isinstance(object_geometry_manifest, Mapping)
            else []
        ),
        "world_model_policy": policy.to_dict(),
        "canonical_output": build_output_linkage(
            policy=policy,
            canonical_artifact_uri=_gs_uri(context, "evaluation_prep/evaluation_prep_manifest.json"),
            presentation_artifact_uri=_gs_uri(context, "presentation_world/presentation_world_manifest.json") if policy.emit_presentation else None,
            authoritative_record=True,
        ),
        "presentation_output": build_output_linkage(
            policy=policy,
            canonical_artifact_uri=_gs_uri(context, "evaluation_prep/evaluation_prep_manifest.json"),
            presentation_artifact_uri=_gs_uri(context, "presentation_world/presentation_bundle.json") if policy.emit_presentation else None,
            authoritative_record=False,
            derivation_mode=policy.allow_generative_completion,
        ),
        "task_ids": task_ids,
        "task_categories": task_categories,
        "source_handoff_path": _relative_to(eval_dir, pipeline_dir / "opportunity_handoff.json"),
        "status": legacy_status,
        "degradation_reasons": degradation_reasons,
        "runtime_registration_attempted": bool(site_world_registration.get("runtime_registration_attempted")),
        "runtime_registration_status": str(site_world_registration.get("runtime_registration_status") or "unknown"),
        "runtime_registration_attempt": runtime_registration_attempt,
        "artifacts": {
            "qualified_opportunity_handoff": _relative_to(eval_dir, rich_handoff_path),
            "scene_memory_bundle_manifest": _relative_to(eval_dir, scene_memory_bundle_manifest_path),
            "geometry_bundle_manifest": _relative_to(eval_dir, geometry_bundle_manifest_path),
            "task_run_manifest": _relative_to(eval_dir, task_run_manifest_path),
            "task_anchor_manifest": _relative_to(eval_dir, task_anchor_manifest_path),
            "protected_regions_manifest": _relative_to(eval_dir, protected_regions_manifest_path),
            "canonical_render_policy": _relative_to(eval_dir, canonical_render_policy_path),
            "presentation_variance_policy": _relative_to(eval_dir, presentation_variance_policy_path),
            "site_world_spec": _relative_to(eval_dir, site_world_spec_path),
            "site_world_registration": _relative_to(eval_dir, site_world_registration_path),
            "site_world_health": _relative_to(eval_dir, site_world_health_path),
            "hosted_session_runtime_manifest": _relative_to(eval_dir, hosted_session_runtime_manifest_path),
            "site_normalization_package": _relative_to(eval_dir, site_normalization_package_path),
            "benchmark_suite_manifest": _relative_to(eval_dir, benchmark_suite_manifest_path),
            "compatibility_matrix": _relative_to(eval_dir, compatibility_matrix_path),
            "recapture_diff": _relative_to(eval_dir, recapture_diff_path),
            "launchable_export_bundle": _relative_to(eval_dir, launchable_export_bundle_path),
            "object_geometry_manifest": _relative_to(eval_dir, object_geometry_target_path),
            "evaluation_prep_summary": _relative_to(eval_dir, summary_path),
            "review_queue": _relative_to(eval_dir, review_queue_path),
            **(
                {"geometry_manifest": str(scene_memory_bundle_manifest.get("geometry_manifest_path"))}
                if scene_memory_bundle_manifest.get("geometry_manifest_path")
                else {}
            ),
            **(
                {"geometry_summary": str(scene_memory_bundle_manifest.get("geometry_summary_path"))}
                if scene_memory_bundle_manifest.get("geometry_summary_path")
                else {}
            ),
            **(
                {"presentation_bundle": _relative_to(eval_dir, pipeline_dir / "presentation_world" / "presentation_bundle.json")}
                if (pipeline_dir / "presentation_world" / "presentation_bundle.json").is_file()
                else {}
            ),
            **(
                {"presentation_world_manifest": _relative_to(eval_dir, pipeline_dir / "presentation_world" / "presentation_world_manifest.json")}
                if (pipeline_dir / "presentation_world" / "presentation_world_manifest.json").is_file()
                else {}
            ),
            **(
                {"runtime_demo_manifest": _relative_to(eval_dir, pipeline_dir / "presentation_world" / "runtime_demo_manifest.json")}
                if (pipeline_dir / "presentation_world" / "runtime_demo_manifest.json").is_file()
                else {}
            ),
            **({"simready_prep_manifest": _relative_to(eval_dir, simready_prep_manifest_path)} if simready_prep_manifest_path is not None else {}),
        },
    }
    manifest_path = eval_dir / "evaluation_prep_manifest.json"
    _copy_json(manifest_path, manifest)
    return {
        "schema_version": "v1",
        "capture_root": str(context.capture_root),
        "manifest_path": str(manifest_path),
        "status": legacy_status,
        "canonical_package_status": canonical_package_status,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build downstream evaluation prep artifacts for a qualified capture")
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument("--provider", default="manual", help="Provider adapter name for object geometry stage")
    args = parser.parse_args(argv)

    try:
        result = run_evaluation_prep_stage(capture_root=args.capture_root, provider_name=args.provider)
    except Exception as exc:
        print(f"[evaluation-prep] FAILED: {exc}")
        return 1

    print(f"[evaluation-prep] manifest={result['manifest_path']}")
    print(f"[evaluation-prep] status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
