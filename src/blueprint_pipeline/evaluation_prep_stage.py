"""Scene-aware downstream evaluation prep artifacts for qualified captures."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .common import PipelineError, ensure_dir, optional_read_json, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context
from .object_geometry_stage import run_object_geometry_stage


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


def _object_index_candidates(capture_root: Path) -> List[Path]:
    return [
        capture_root / "raw" / "object_index.json",
        capture_root / "raw" / "arkit" / "objects" / "index.json",
    ]


def _has_object_index_entries(capture_root: Path) -> bool:
    for path in _object_index_candidates(capture_root):
        if not path.is_file():
            continue
        payload = _read_optional_json_any(path)
        if isinstance(payload, list) and payload:
            return True
        if isinstance(payload, Mapping):
            for key in ("objects", "items", "summaries"):
                value = payload.get(key)
                if isinstance(value, list) and value:
                    return True
    return False


def _build_missing_object_geometry_manifest(*, context, provider_name: str) -> Dict[str, Any]:
    missing_inputs = [str(path) for path in _object_index_candidates(context.capture_root) if not path.is_file()]
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "provider_name": provider_name,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "missing_object_index",
        "provenance": "evaluation_prep_fallback",
        "objects": [],
        "notes": [
            "Object geometry was skipped because no object index was found in the staged capture bundle."
        ],
        "missing_inputs": missing_inputs,
    }


def _resolve_object_geometry_manifest(*, context, provider_name: str) -> Dict[str, Any]:
    if _has_object_index_entries(context.capture_root):
        object_geometry_result = run_object_geometry_stage(
            capture_root=context.capture_root,
            provider_name=provider_name,
        )
        object_geometry_source_path = Path(str(object_geometry_result.get("manifest_path") or ""))
        loaded = read_json_any(object_geometry_source_path)
        if isinstance(loaded, Mapping):
            return dict(loaded)
        raise PipelineError(f"Object geometry manifest is not a JSON object: {object_geometry_source_path}")
    return _build_missing_object_geometry_manifest(context=context, provider_name=provider_name)


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
            tasks.append(
                {
                    "task_id": str(item.get("task_id") or item.get("capture_id") or ""),
                    "task_text": str(item.get("task_text") or ""),
                    "task_category": str(item.get("task_category") or "generic"),
                    "capture_root": str(item.get("capture_root") or ""),
                    "capture_id": str(item.get("capture_id") or ""),
                    "target_object_ids": target_ids,
                    "articulation_required_ids": articulation_ids,
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
                }
            )
    else:
        target_ids = _string_list(scope_record.get("target_object_ids"))
        articulation_ids = _string_list(scope_record.get("articulation_required_ids"))
        task_text = _default_task_text(scope_record, handoff, capture_root.name)
        goal = _zone_center(target_ids or articulation_ids)
        tasks.append(
            {
                "task_id": _default_task_id(scope_record, handoff, capture_root.name),
                "task_text": task_text,
                "task_category": _task_category(task_text),
                "capture_root": str(capture_root),
                "capture_id": capture_root.name,
                "target_object_ids": target_ids,
                "articulation_required_ids": articulation_ids,
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
            }
        )

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "scene_id": capture_root.parts[-3],
        "capture_id": capture_root.name,
        "tasks": tasks,
    }


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
    files = {
        "bundle_path": scene_memory_dir,
        "scene_memory_manifest_path": scene_memory_dir / "scene_memory_manifest.json",
        "scene_memory_readiness_path": scene_memory_dir / "scene_memory_readiness.json",
        "conditioning_bundle_path": scene_memory_dir / "conditioning_bundle.json",
        "preview_simulation_manifest_path": preview_dir / "preview_simulation_manifest.json",
        "gen3c_adapter_manifest_path": adapter_dir / "gen3c.json",
        "neoverse_adapter_manifest_path": adapter_dir / "neoverse.json",
        "cosmos_transfer_adapter_manifest_path": adapter_dir / "cosmos_transfer.json",
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


def _build_hosted_session_runtime_manifest(
    *,
    context,
    normalized_handoff: Mapping[str, Any],
    scene_memory_bundle_manifest: Mapping[str, Any],
    task_anchor_manifest: Mapping[str, Any],
    task_run_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    eval_dir = context.capture_root / "pipeline" / "evaluation_prep"
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
    runtime_capabilities = {
        "supports_step_rollout": True,
        "supports_batch_rollout": True,
        "supports_camera_views": True,
        "supports_rlds_export": True,
        "supports_preview_render": "gen3c" in available_backends,
    }

    launch_blockers: List[str] = []
    if str(normalized_handoff.get("qualification_state") or "").strip().lower() != "ready":
        launch_blockers.append(
            f"qualification_state:{normalized_handoff.get('qualification_state')}"
        )
    if not bool(normalized_handoff.get("downstream_evaluation_eligibility")):
        launch_blockers.append("downstream_evaluation_eligibility:false")
    if scene_memory_bundle_manifest.get("status") != "complete":
        launch_blockers.append(
            f"scene_memory_bundle:{scene_memory_bundle_manifest.get('status')}"
        )
    if not task_ids:
        launch_blockers.append("missing_task_anchor_manifest")
    if not available_backends:
        launch_blockers.append("runtime_manifest_only")
    if available_backends and not launchable_backends:
        launch_blockers.append("no_launchable_stage1_backend")

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
        "task_anchor_manifest_uri": _gs_uri(
            context, "evaluation_prep/task_anchor_manifest.json"
        ),
        "task_run_manifest_uri": _gs_uri(
            context, "evaluation_prep/task_run_manifest.json"
        ),
        "available_backends": available_backends,
        "launchable_backends": launchable_backends,
        "default_backend": default_backend,
        "customer_facing_runtime": (
            "Hosted site runtime"
            if default_backend
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
        "launchable": len(launch_blockers) == 0,
        "launch_blockers": launch_blockers,
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
    hosted_session_runtime_manifest: Mapping[str, Any],
    simready_prep_manifest_path: Optional[Path],
) -> Dict[str, Any]:
    default_backend = str(hosted_session_runtime_manifest.get("default_backend") or "").strip()
    public_runtime_label = str(
        hosted_session_runtime_manifest.get("customer_facing_runtime") or "Hosted site runtime"
    ).strip()
    bundles = {
        "world_model_runtime": {
            "launchable": bool(hosted_session_runtime_manifest.get("launchable")),
            "required_artifacts": [
                "scene_memory_manifest",
                "conditioning_bundle",
                "task_anchor_manifest",
                "task_run_manifest",
            ],
            "backend": default_backend or None,
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
    }
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": "ready" if any(item["launchable"] for item in bundles.values()) else "partial",
        "public_runtime_label": public_runtime_label,
        "default_backend": default_backend or None,
        "scenario_variants": hosted_session_runtime_manifest.get("scenario_variants", []),
        "bundles": bundles,
        "scene_memory_bundle_status": scene_memory_bundle_manifest.get("status"),
        "geometry_bundle_status": geometry_bundle_manifest.get("status"),
    }


def run_evaluation_prep_stage(
    *,
    capture_root: str | Path,
    provider_name: str = "manual",
) -> Dict[str, Any]:
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

    hosted_session_runtime_manifest = _build_hosted_session_runtime_manifest(
        context=context,
        normalized_handoff=normalized_handoff,
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
        task_anchor_manifest=task_anchor_manifest,
        task_run_manifest=task_run_manifest,
    )
    hosted_session_runtime_manifest_path = eval_dir / "hosted_session_runtime_manifest.json"
    _copy_json(hosted_session_runtime_manifest_path, hosted_session_runtime_manifest)

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
        hosted_session_runtime_manifest=hosted_session_runtime_manifest,
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
    }
    summary_path = eval_dir / "evaluation_prep_summary.json"
    _copy_json(summary_path, summary)

    qualification_state = str(normalized_handoff.get("qualification_state") or "not_ready_yet")
    eligibility = bool(normalized_handoff.get("downstream_evaluation_eligibility"))
    degradation_reasons: List[str] = []
    if qualification_state != "ready":
        degradation_reasons.append(f"qualification_state:{qualification_state}")
    if not eligibility:
        degradation_reasons.append("downstream_evaluation_eligibility:false")
    if scene_memory_bundle_manifest.get("status") != "complete":
        degradation_reasons.append(f"scene_memory_bundle:{scene_memory_bundle_manifest.get('status')}")
    if (
        scene_memory_bundle_manifest.get("status") != "complete"
        and geometry_bundle_manifest.get("status") != "complete"
    ):
        degradation_reasons.append(f"geometry_bundle:{geometry_bundle_manifest.get('status')}")
    if object_count == 0:
        degradation_reasons.append("object_geometry:missing")
    status = "ready_for_validation"
    if qualification_state != "ready" or not eligibility:
        status = "not_ready_for_validation"
    elif degradation_reasons:
        status = "degraded_but_usable"

    task_ids = [str(task.get("task_id") or "") for task in task_anchor_manifest.get("tasks", []) if isinstance(task, Mapping)]
    task_categories = sorted({str(task.get("task_category") or "generic") for task in task_anchor_manifest.get("tasks", []) if isinstance(task, Mapping)})
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
        "task_ids": task_ids,
        "task_categories": task_categories,
        "source_handoff_path": _relative_to(eval_dir, pipeline_dir / "opportunity_handoff.json"),
        "status": status,
        "degradation_reasons": degradation_reasons,
        "artifacts": {
            "qualified_opportunity_handoff": _relative_to(eval_dir, rich_handoff_path),
            "scene_memory_bundle_manifest": _relative_to(eval_dir, scene_memory_bundle_manifest_path),
            "geometry_bundle_manifest": _relative_to(eval_dir, geometry_bundle_manifest_path),
            "task_run_manifest": _relative_to(eval_dir, task_run_manifest_path),
            "task_anchor_manifest": _relative_to(eval_dir, task_anchor_manifest_path),
            "hosted_session_runtime_manifest": _relative_to(
                eval_dir, hosted_session_runtime_manifest_path
            ),
            "site_normalization_package": _relative_to(eval_dir, site_normalization_package_path),
            "benchmark_suite_manifest": _relative_to(eval_dir, benchmark_suite_manifest_path),
            "compatibility_matrix": _relative_to(eval_dir, compatibility_matrix_path),
            "recapture_diff": _relative_to(eval_dir, recapture_diff_path),
            "launchable_export_bundle": _relative_to(eval_dir, launchable_export_bundle_path),
            "object_geometry_manifest": _relative_to(eval_dir, object_geometry_target_path),
            "evaluation_prep_summary": _relative_to(eval_dir, summary_path),
            "review_queue": _relative_to(eval_dir, review_queue_path),
            **({"simready_prep_manifest": _relative_to(eval_dir, simready_prep_manifest_path)} if simready_prep_manifest_path is not None else {}),
        },
    }
    manifest_path = eval_dir / "evaluation_prep_manifest.json"
    _copy_json(manifest_path, manifest)
    return {
        "schema_version": "v1",
        "capture_root": str(context.capture_root),
        "manifest_path": str(manifest_path),
        "status": status,
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
