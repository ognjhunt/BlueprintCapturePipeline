"""Scene-aware downstream evaluation prep artifacts for qualified captures."""

from __future__ import annotations

import argparse
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .alpha_readiness import sync_webapp_evaluation_prep, write_alpha_readiness_summary
from .backend_support_artifacts import resolve_backend_support_artifacts
from .common import PipelineError, ensure_dir, optional_read_json, read_json_any, relative_scene_path, utc_now_iso, write_json
from .launch_proof_policy import runtime_required
from .local_capture import resolve_local_capture_context
from .object_geometry_stage import resolve_object_geometry_manifest
from .proof_contracts import (
    build_hosted_review_readiness,
    build_proof_pack_manifest,
    build_proof_path_status,
    build_site_package_manifest,
)
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


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


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
        "site_world_runtime": "site_world_runtime_adapter_manifest_path",
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


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    payload = _read_optional_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _strict_proven_flag(
    primary: Mapping[str, Any], fallback: Mapping[str, Any], key: str
) -> bool:
    """Fail-closed proof flag: a strict boolean in ``primary`` (the proof boundary /
    validation manifest) is authoritative either way; otherwise only a strict True in
    ``fallback`` counts. Truthy non-booleans ("true", 1.0, {...}) never read as proof."""
    value = primary.get(key)
    if isinstance(value, bool):
        return value
    return fallback.get(key) is True


def _cosmos_runtime_backend_variants(
    *,
    context,
    pipeline_dir: Path,
    native_semantics: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    support = resolve_backend_support_artifacts(
        pipeline_dir, backend_id="cosmos_predict2_5"
    )
    benchmark_path, benchmark, benchmark_relative = support["cosmos_zero_shot_benchmark"]
    export_path, export_manifest, export_relative = support["cosmos_training_export"]
    training_run_path, training_run, training_relative = support["cosmos_lora_training"]
    native_primary_ready = bool(native_semantics.get("native_world_model_primary"))

    variants: Dict[str, Dict[str, Any]] = {}

    zero_shot_blockers: List[str] = []
    benchmark_status = str(benchmark.get("status") or "").strip()
    if benchmark_status != "completed":
        zero_shot_blockers.append(
            f"cosmos_zero_shot_benchmark:{benchmark_status or 'missing'}"
        )
        if str(benchmark.get("reason") or "").strip():
            zero_shot_blockers.append(str(benchmark.get("reason") or "").strip())
    zero_shot_blockers.append("native_serving_contract_unimplemented")
    variants["cosmos_zero_shot_i2w"] = {
        "backend_id": "cosmos_zero_shot_i2w",
        "bundle_manifest_uri": _gs_uri(context, benchmark_relative)
        if benchmark_path.is_file()
        else None,
        "adapter_manifest_uri": None,
        "launchable": False,
        "readiness_state": _runtime_readiness_state(launchable=False, blockers=zero_shot_blockers),
        "blockers": zero_shot_blockers,
        "warnings": [
            f"native_world_model_primary:{native_primary_ready}",
        ],
        "runtime_mode": "local_gpu_runtime",
        "grounding_status": str(native_semantics.get("native_world_model_status") or "not_ready"),
        "quality_flags": {
            "benchmark_status": benchmark_status or "missing",
            "benchmark_reason": benchmark.get("reason"),
        },
        "conversion": {
            "benchmark_manifest_uri": _gs_uri(context, benchmark_relative)
            if benchmark_path.is_file()
            else None,
        },
        "canonical_write_allowed": False,
    }

    training_blockers: List[str] = []
    export_status = str(export_manifest.get("status") or "").strip()
    training_status = str(training_run.get("status") or "").strip()
    if export_status != "ready":
        training_blockers.append(f"cosmos_training_export:{export_status or 'missing'}")
    if benchmark_status != "completed":
        training_blockers.append(f"cosmos_zero_shot_benchmark:{benchmark_status or 'missing'}")
    if training_status != "completed":
        training_blockers.append(f"cosmos_lora_training:{training_status or 'missing'}")
        if str(training_run.get("reason") or "").strip():
            training_blockers.append(str(training_run.get("reason") or "").strip())
    if not native_primary_ready:
        training_blockers.append("native_world_model_not_primary")
    training_launchable = not training_blockers
    variants["cosmos_predict_lora_adapter"] = {
        "backend_id": "cosmos_predict_lora_adapter",
        "bundle_manifest_uri": _gs_uri(context, export_relative)
        if export_path.is_file()
        else None,
        "adapter_manifest_uri": _gs_uri(context, training_relative)
        if training_run_path.is_file()
        else None,
        "launchable": training_launchable,
        "readiness_state": _runtime_readiness_state(
            launchable=training_launchable,
            blockers=training_blockers,
        ),
        "blockers": training_blockers,
        "warnings": [
            f"source_mode:{str(export_manifest.get('source_mode') or 'unknown')}",
        ]
        if export_manifest
        else [],
        "runtime_mode": "trained_adapter",
        "grounding_status": str(native_semantics.get("native_world_model_status") or "not_ready"),
        "quality_flags": {
            "benchmark_status": benchmark_status or "missing",
            "training_status": training_status or "missing",
            "source_mode": export_manifest.get("source_mode"),
        },
        "conversion": {
            "training_export_manifest_uri": _gs_uri(context, export_relative)
            if export_path.is_file()
            else None,
            "benchmark_manifest_uri": _gs_uri(context, benchmark_relative)
            if benchmark_path.is_file()
            else None,
            "training_run_manifest_uri": _gs_uri(context, training_relative)
            if training_run_path.is_file()
            else None,
            "checkpoint_path": training_run.get("checkpoint_path"),
        },
        "canonical_write_allowed": False,
    }
    return variants


def _build_runtime_backend_variants(
    *,
    context,
    eval_dir: Path,
    pipeline_dir: Path,
    scene_memory_bundle_manifest: Mapping[str, Any],
    native_semantics: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    adapter_key_map = {
        "site_world_runtime": "site_world_runtime_adapter_manifest_path",
        "gen3c": "gen3c_adapter_manifest_path",
        "cosmos_transfer": "cosmos_transfer_adapter_manifest_path",
    }
    adapter_details = _adapter_manifest_details(scene_memory_bundle_manifest, eval_dir=eval_dir)
    variants: Dict[str, Dict[str, Any]] = {}
    for backend, key in adapter_key_map.items():
        adapter_rel_path = str(scene_memory_bundle_manifest.get(key) or "").strip()
        if not adapter_rel_path:
            continue
        detail = adapter_details.get(backend, {})
        status = str(detail.get("status") or "").strip()
        blockers: List[str] = []
        if not status.startswith("available_stage1_"):
            blockers.append(f"adapter_status:{status or 'missing'}")
        variants[backend] = {
            "backend_id": backend,
            "bundle_manifest_uri": _gs_uri(context, "scene_memory/scene_memory_manifest.json"),
            "adapter_manifest_uri": _gs_uri(context, f"scene_memory/adapter_manifests/{backend}.json"),
            "launchable": not blockers,
            "readiness_state": _runtime_readiness_state(launchable=not blockers, blockers=blockers),
            "blockers": blockers,
            "warnings": [],
            "runtime_mode": str(detail.get("execution_mode") or "unknown"),
            "grounding_status": str(native_semantics.get("native_world_model_status") or "not_ready"),
            "quality_flags": {
                "adapter_status": status or "missing",
                "native_world_model_primary": bool(native_semantics.get("native_world_model_primary")),
            },
            "canonical_write_allowed": False,
        }
    variants.update(
        _cosmos_runtime_backend_variants(
            context=context,
            pipeline_dir=pipeline_dir,
            native_semantics=native_semantics,
        )
    )
    return variants


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


def _hosted_session_runtime_claim_boundary() -> Dict[str, Any]:
    return {
        "schema_version": "hosted_session_runtime_claim_boundary.v1",
        "artifact_purpose": "hosted_session_runtime_contract_only",
        "hosted_session_artifacts_only": True,
        "live_provider_jobs_called": False,
        "simulators_run": False,
        "deployments_performed": False,
        "simulator_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "deployment_approval_proven": False,
        "safety_validation_proven": False,
        "physical_readiness_proven": False,
        "physical_robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "allowed_display": [
            "hosted review runtime contract",
            "available runtime backend catalog",
            "task and scenario review catalogs",
            "export defaults for reviewer sessions",
        ],
        "disallowed_claims": [
            "simulator_execution_completed",
            "rank_fidelity_result",
            "deployment_approval",
            "safety_validation",
            "physical_readiness",
            "robot_ready",
        ],
        "blocked_claim_upgrades": [
            "simulator_execution_completed_claim",
            "rank_fidelity_result_claim",
            "deployment_approval_claim",
            "safety_validation_claim",
            "physical_readiness_claim",
            "robot_ready_claim",
        ],
        "proof_upgrade_requires": [
            "owner-system simulator execution trace",
            "accepted policy-ranking methodology with heldout or real anchors",
            "buyer or operator deployment approval record",
            "safety validation evidence for the exact robot, site, and task",
            "physical robot trial or owner-attested readiness evidence",
        ],
    }


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
        "site_world_runtime_adapter_manifest_path": adapter_dir / "site_world_runtime.json",
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
        "site_world_runtime_adapter_manifest_path",
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
    payload = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": status,
        "missing_required_fields": missing,
        "available_fields": sorted(entries.keys()),
        **entries,
    }
    geometry_summary = _read_json_object(geometry_dir / "geometry_summary.json")
    if geometry_summary:
        payload["geometry_summary"] = geometry_summary
        payload["geometry_truth"] = _geometry_conditioning_truth(payload)
    return payload


def _geometry_conditioning_truth(scene_memory_bundle_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    geometry_summary = (
        scene_memory_bundle_manifest.get("geometry_summary")
        if isinstance(scene_memory_bundle_manifest.get("geometry_summary"), Mapping)
        else {}
    )
    geometry_summary_path = str(scene_memory_bundle_manifest.get("geometry_summary_path") or "").strip()
    geometry_source = str(geometry_summary.get("geometry_source") or "missing").strip()
    fallback_used = bool(geometry_summary.get("fallback_used"))
    ready_for_world_model = bool(geometry_summary.get("ready_for_world_model"))
    provider_native_result = bool(geometry_summary.get("provider_native_result"))
    site_frame_available = bool(geometry_summary.get("site_frame_available"))
    scale_resolved = bool(geometry_summary.get("scale_resolved"))
    contract_ready_for_world_model = bool(geometry_summary.get("contract_ready_for_world_model"))
    geometry_live_ready = bool(
        geometry_summary.get("geometry_live_ready")
        if geometry_summary.get("geometry_live_ready") is not None
        else (
            ready_for_world_model
            and geometry_source == "video_to_world"
            and not fallback_used
            and provider_native_result
            and site_frame_available
            and scale_resolved
        )
    )
    # Fallback/synthetic geometry (including local_sfm) is never treated as
    # live/reference-ready; contract_ready_for_world_model already requires
    # not fallback_used, so no fallback kind gets a carve-out here.
    blockers = list(geometry_summary.get("launch_blockers") or [])
    if geometry_summary_path and fallback_used:
        blockers.append("fallback_geometry_not_live_video_to_world")
    if geometry_summary_path and geometry_source != "video_to_world":
        blockers.append(f"geometry_source_not_video_to_world:{geometry_source or 'missing'}")
    if geometry_summary_path and not provider_native_result:
        blockers.append("provider_native_geometry_missing")
    if geometry_summary_path and not site_frame_available:
        blockers.append("site_frame_not_proven")
    if geometry_summary_path and not scale_resolved:
        blockers.append("scale_not_proven")
    if geometry_summary_path and not geometry_live_ready:
        blockers.append("geometry_not_live_video_to_world")
    local_reference_ready = bool(
        geometry_summary_path
        and geometry_source == "local_sfm"
        and not fallback_used
        and contract_ready_for_world_model
    )
    non_arkit_geometry_state = (
        "ready"
        if geometry_live_ready and provider_native_result and geometry_source == "video_to_world"
        else "degraded"
        if local_reference_ready
        else "blocked"
    )
    return {
        "geometry_summary_path": geometry_summary_path,
        "geometry_source": geometry_source,
        "fallback_used": fallback_used,
        "fallback_kind": geometry_summary.get("fallback_kind"),
        "ready_for_world_model": ready_for_world_model,
        "contract_ready_for_world_model": contract_ready_for_world_model,
        "internal_fallback_ready": bool(geometry_summary.get("internal_fallback_ready")),
        "geometry_live_ready": geometry_live_ready,
        "site_faithful_market_ready": bool(geometry_summary.get("site_faithful_market_ready")),
        "provider_native_result": provider_native_result,
        "site_frame_available": site_frame_available,
        "scale_resolved": scale_resolved,
        "pose_match_rate": geometry_summary.get("pose_match_rate"),
        "p95_pose_delta_sec": geometry_summary.get("p95_pose_delta_sec"),
        "local_reference_ready": local_reference_ready,
        "provider_native_geometry_ready": bool(geometry_live_ready and provider_native_result),
        "non_arkit_geometry_state": non_arkit_geometry_state,
        "launchable": bool(geometry_summary_path and geometry_live_ready and provider_native_result),
        "blockers": list(dict.fromkeys(blockers)),
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
    site_submission_id = str(payload.get("site_submission_id") or "").strip()
    buyer_request_id = str(payload.get("buyer_request_id") or "").strip()
    capture_job_id = str(payload.get("capture_job_id") or "").strip()
    upstream_link_blockers = _string_list(payload.get("upstream_link_blockers"))
    for blocker, value in (
        ("missing_site_submission_id", site_submission_id),
        ("missing_buyer_request_id", buyer_request_id),
        ("missing_capture_job_id", capture_job_id),
    ):
        if not value and blocker not in upstream_link_blockers:
            upstream_link_blockers.append(blocker)
    payload.update(
        {
            "schema_version": "v1",
            "site_submission_id": site_submission_id,
            "buyer_request_id": buyer_request_id,
            "capture_job_id": capture_job_id,
            "upstream_link_truth_state": "verified" if not upstream_link_blockers else "blocked_missing_upstream_ids",
            "upstream_link_blockers": upstream_link_blockers,
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
        "site_world_runtime_adapter_manifest_path",
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


def simulation_automation_evaluation_prep_surface(
    *,
    capture_root: str | Path,
    eval_dir: Path,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    automation_dir = context.pipeline_root / "simulation_automation"
    artifact_paths = {
        "simulation_automation_plan": automation_dir / "simulation_automation_plan.json",
        "simulation_automation_run_manifest": (
            automation_dir / "simulation_automation_run_manifest.json"
        ),
        "robot_eval_scene_asset_inventory": automation_dir / "scene_asset_inventory.json",
        "robot_eval_scene_asset_dependency_audit": (
            automation_dir / "scene_asset_dependency_audit.json"
        ),
        "robot_eval_scene_asset_preflight": automation_dir / "scene_asset_preflight.json",
        "robot_eval_scene_asset_inspection": automation_dir / "scene_asset_inspection.json",
        "robot_eval_scene_frame_estimate": automation_dir / "scene_frame_estimate.json",
        "robot_eval_collider_proxy_plan": automation_dir / "collider_proxy_plan.json",
        "robot_eval_cpu_scene_proxy_manifest": automation_dir / "cpu_scene_proxy_manifest.json",
        "robot_eval_cpu_preflight_scorecard": automation_dir / "cpu_preflight_scorecard.json",
        "robot_eval_task_anchor_proposal_manifest": (
            automation_dir / "task_anchor_proposal_manifest.json"
        ),
        "robot_eval_episode_spec_manifest": automation_dir / "episode_spec_manifest.json",
        "robot_eval_episode_specs": automation_dir / "episode_specs.json",
        "robot_eval_spawn_pose_validation_manifest": (
            automation_dir / "spawn_pose_validation_manifest.json"
        ),
        "robot_eval_cpu_preflight_manifest": automation_dir / "cpu_preflight_manifest.json",
        "robot_eval_pre_gpu_readiness_summary": (
            automation_dir / "pre_gpu_readiness_summary.json"
        ),
        "robot_eval_cpu_simulator_preflight_manifest": (
            automation_dir / "cpu_simulator_preflight_manifest.json"
        ),
        "robot_eval_scenario_variation_instances": (
            automation_dir / "scenario_variation_instances.json"
        ),
        "robot_eval_arena_environment_packet": automation_dir / "arena_environment_packet.json",
        "robot_eval_simulator_engine_plugin_registry": (
            automation_dir / "simulator_engine_plugin_registry.json"
        ),
        "robot_eval_gpu_handoff_packet": automation_dir / "gpu_handoff_packet.json",
        "robot_eval_gpu_owner_system_proof_schema": (
            automation_dir / "gpu_owner_system_proof_schema.json"
        ),
        "robot_eval_gpu_run_checklist": automation_dir / "gpu_run_checklist.md",
        "robot_eval_owner_gpu_simulator_execution_blocked_manifest": (
            automation_dir / "owner_gpu_simulator_execution_blocked_manifest.json"
        ),
        "asset_conversion_plan": automation_dir / "asset_conversion_plan.json",
        "simulator_execution_manifest": automation_dir / "simulator_execution_manifest.json",
        "training_orchestration_manifest": (
            automation_dir / "training_orchestration_manifest.json"
        ),
        "simulation_automation_proof_boundary": automation_dir / "proof_boundary.json",
        "simulation_automation_agent_decision_ledger": (
            automation_dir / "agent_decision_ledger.json"
        ),
    }
    artifacts = {
        key: _relative_to(eval_dir, path)
        for key, path in artifact_paths.items()
        if path.is_file()
    }
    artifact_uris = {
        f"{key}_uri": _gs_uri(context, f"simulation_automation/{path.name}")
        for key, path in artifact_paths.items()
        if path.is_file()
    }
    run_manifest = _read_optional_mapping(automation_dir / "simulation_automation_run_manifest.json")
    proof_boundary = _read_optional_mapping(automation_dir / "proof_boundary.json")

    def _proven_flag(key: str) -> bool:
        # Fail closed: the proof boundary is authoritative when it states the flag.
        # A run manifest can only claim proof when the boundary doesn't contradict it,
        # and only a strict boolean True counts as a claim.
        boundary_value = proof_boundary.get(key)
        if isinstance(boundary_value, bool):
            return boundary_value
        return run_manifest.get(key) is True

    return {
        "schema_version": "simulation_automation_evaluation_prep_surface.v1",
        "status": str(run_manifest.get("status") or "missing"),
        "artifacts": artifacts,
        "artifact_uris": artifact_uris,
        "simulator_execution_proven": _proven_flag("simulator_execution_proven"),
        "rank_fidelity_result_proven": _proven_flag("rank_fidelity_result_proven"),
        "public_claim_upgrade_allowed": _proven_flag("public_claim_upgrade_allowed"),
    }


def palatial_physready_evaluation_prep_surface(
    *,
    capture_root: str | Path,
    eval_dir: Path,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    palatial_dir = context.pipeline_root / "palatial_physready"
    artifact_paths = {
        "palatial_physready_twin_candidates": (
            palatial_dir / "twin_candidate_manifest.json"
        ),
        "palatial_physready_request_manifest": (
            palatial_dir / "palatial_request_manifest.json"
        ),
        "palatial_physready_run_manifest": (
            palatial_dir / "palatial_physready_run_manifest.json"
        ),
        "palatial_physready_materialization_manifest": (
            palatial_dir / "materialization_manifest.json"
        ),
        "palatial_physready_validation_manifest": (
            palatial_dir / "validation_manifest.json"
        ),
    }
    artifacts = {
        key: _relative_to(eval_dir, path)
        for key, path in artifact_paths.items()
        if path.is_file()
    }
    artifact_uris = {
        f"{key}_uri": _gs_uri(context, f"palatial_physready/{path.name}")
        for key, path in artifact_paths.items()
        if path.is_file()
    }
    run_manifest = _read_optional_mapping(
        palatial_dir / "palatial_physready_run_manifest.json"
    )
    validation = _read_optional_mapping(palatial_dir / "validation_manifest.json")
    return {
        "schema_version": "palatial_physready_evaluation_prep_surface.v1",
        "status": str(run_manifest.get("status") or "missing"),
        "artifacts": artifacts,
        "artifact_uris": artifact_uris,
        "model_derived_support_assets_present": bool(
            validation.get("inspection_count")
            or run_manifest.get("local_exports_materialized")
        ),
        "live_provider_calls_performed": bool(
            run_manifest.get("live_provider_calls_performed")
        ),
        "remote_asset_downloads_performed": bool(
            run_manifest.get("remote_asset_downloads_performed")
        ),
        "simulator_execution_proven": _strict_proven_flag(
            validation, run_manifest, "simulator_execution_proven"
        ),
        "rank_fidelity_result_proven": _strict_proven_flag(
            validation, run_manifest, "rank_fidelity_result_proven"
        ),
        "public_claim_upgrade_allowed": _strict_proven_flag(
            validation, run_manifest, "public_claim_upgrade_allowed"
        ),
    }


def site_eval_director_evaluation_prep_surface(
    *,
    capture_root: str | Path,
    eval_dir: Path,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    automation_dir = context.pipeline_root / "simulation_automation"
    artifact_paths = {
        "scenario_execution_plan": automation_dir / "scenario_execution_plan.json",
        "task_simulation_requests": automation_dir / "task_simulation_requests.json",
        "scenario_simulator_matrix": automation_dir / "scenario_simulator_matrix.json",
        "agent_review_queue": automation_dir / "agent_review_queue.json",
        "site_eval_director_run_manifest": (
            automation_dir / "site_eval_director_run_manifest.json"
        ),
        "site_eval_director_proof_boundary": (
            automation_dir / "site_eval_director_proof_boundary.json"
        ),
        "site_eval_director_blocked_manifest": (
            automation_dir / "site_eval_director_blocked_manifest.json"
        ),
        "agents_sdk_site_eval_director_request": (
            automation_dir / "agents_sdk_site_eval_director_request.json"
        ),
        "codex_sdk_code_maintainer_request": (
            automation_dir / "codex_sdk_code_maintainer_request.json"
        ),
        "normalized_simulator_attempt_trace": (
            automation_dir / "normalized_simulator_attempt_trace.json"
        ),
        "failure_labels": automation_dir / "failure_labels.json",
        "site_eval_prediction_outcome_ledger": (
            automation_dir / "site_eval_prediction_outcome_ledger.json"
        ),
        "site_eval_calibration_report": (
            automation_dir / "site_eval_calibration_report.json"
        ),
        "learned_facility_breakage_library": (
            automation_dir / "learned_facility_breakage_library.json"
        ),
        "updated_eval_cards": automation_dir / "updated_eval_cards.json",
        "cosmos_orchestration_exports": (
            automation_dir / "cosmos_orchestration_exports.json"
        ),
        "site_eval_real_evidence_blocked_manifest": (
            automation_dir / "site_eval_real_evidence_blocked_manifest.json"
        ),
        "site_eval_fixture_runner_blocked_manifest": (
            automation_dir / "site_eval_fixture_runner_blocked_manifest.json"
        ),
    }
    artifacts = {
        key: _relative_to(eval_dir, path)
        for key, path in artifact_paths.items()
        if path.is_file()
    }
    artifact_uris = {
        f"{key}_uri": _gs_uri(context, f"simulation_automation/{path.name}")
        for key, path in artifact_paths.items()
        if path.is_file()
    }
    run_manifest = _read_optional_mapping(
        automation_dir / "site_eval_director_run_manifest.json"
    )
    proof_boundary = _read_optional_mapping(
        automation_dir / "site_eval_director_proof_boundary.json"
    )
    blocked_manifest = _read_optional_mapping(
        automation_dir / "site_eval_director_blocked_manifest.json"
    )
    return {
        "schema_version": "site_eval_director_evaluation_prep_surface.v1",
        "status": str(
            run_manifest.get("status")
            or blocked_manifest.get("status")
            or "missing"
        ),
        "artifacts": artifacts,
        "artifact_uris": artifact_uris,
        "simulator_execution_proven": _strict_proven_flag(
            proof_boundary, run_manifest, "simulator_execution_proven"
        ),
        "rank_fidelity_result_proven": _strict_proven_flag(
            proof_boundary, run_manifest, "rank_fidelity_result_proven"
        ),
        "public_claim_upgrade_allowed": _strict_proven_flag(
            proof_boundary, run_manifest, "public_claim_upgrade_allowed"
        ),
    }


def robot_eval_job_evaluation_prep_surface(
    *,
    capture_root: str | Path,
    eval_dir: Path,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    jobs_dir = context.pipeline_root / "robot_eval_jobs"
    artifact_names = {
        "job_request": "job_request.json",
        "run_manifest": "job_run_manifest.json",
        "proof_boundary": "proof_boundary.json",
        "blocked_manifest": "blocked_manifest.json",
        "job_validation": "job_validation.json",
        "scheduler_decision": "scheduler_decision.json",
        "worker_launch_plan": "worker_launch_plan.json",
        "gpu_startup_pipeline_plan": "gpu_startup_pipeline_plan.json",
        "worker_manifest": "worker_manifest.json",
        "gpu_provisioning_request": "gpu_provisioning_request.json",
        "gpu_provider_launch_request": "gpu_provider_launch_request.json",
        "gpu_provider_launcher_result": "gpu_provider_launcher_result.json",
        "gpu_provider_launcher_stdout_log": "gpu_provider_launcher.stdout.log",
        "gpu_provider_launcher_stderr_log": "gpu_provider_launcher.stderr.log",
        "runpod_provider_adapter_result": "runpod_provider_adapter_result.json",
        "gpu_cost_control_ledger": "gpu_cost_control_ledger.json",
        "remote_cloud_execution_closure_manifest": (
            "remote_cloud_execution_closure_manifest.json"
        ),
        "robot_team_grade_eval_closure_manifest": (
            "robot_team_grade_eval_closure_manifest.json"
        ),
        "gpu_provisioning_result": "gpu_provisioning_result.json",
        "startup_architecture_audit": "startup_architecture_audit.json",
        "worker_runtime_manifest": "worker_runtime_manifest.json",
        "worker_runtime_preflight": "worker_runtime_preflight.json",
        "worker_runtime_preflight_stdout_log": "worker_runtime_preflight.stdout.log",
        "worker_runtime_preflight_stderr_log": "worker_runtime_preflight.stderr.log",
        "policy_package_manifest": "policy_package_manifest.json",
        "scenario_eval_matrix": "scenario_eval_matrix.json",
        "robot_pov_observation_manifest": "robot_pov_observation_manifest.json",
        "robot_pov_observation_candidate_set": "robot_pov_observation_candidate_set.json",
        "selected_initial_policy_observation": "selected_initial_policy_observation.json",
        "robot_pov_observations": "robot_pov_observations.jsonl",
        "robot_pov_frame_sequence_manifest": "robot_pov_frame_sequence_manifest.json",
        "robot_pov_render_storyboard": "robot_pov_render_storyboard.json",
        "policy_execution_manifest": "policy_execution_manifest.json",
        "policy_execution_trace": "policy_execution_trace.json",
        "policy_execution_trace_jsonl": "policy_execution_trace.jsonl",
        "simulator_service_result": "simulator_service_result.json",
        "normalized_attempt_trace": "normalized_attempt_trace.json",
        "failure_labels": "failure_labels.json",
        "prediction_outcome_ledger": "prediction_outcome_ledger.json",
        "calibration_report": "calibration_report.json",
        "deployment_outcome_intake_manifest": "deployment_outcome_intake_manifest.json",
        "deployment_outcome_ledger": "deployment_outcome_ledger.json",
        "sim_vs_real_calibration_report": "sim_vs_real_calibration_report.json",
        "prediction_vs_actual_deployment_summary": (
            "prediction_vs_actual_deployment_summary.json"
        ),
        "real_world_validation_followup_plan": "real_world_validation_followup_plan.json",
        "real_world_validation_followup_request_queue": (
            "real_world_validation_followup_request_queue.json"
        ),
        "live_eval_closure_manifest": "live_eval_closure_manifest.json",
        "training_result": "training_result.json",
        "evaluation_result": "evaluation_result.json",
        "robot_eval_report": "robot_eval_report.json",
        "robot_eval_report_markdown": "robot_eval_report.md",
        "post_training_data_package_export_manifest": (
            "post_training_data_package_export_manifest.json"
        ),
        "webapp_robot_eval_status_projection": "webapp_robot_eval_status_projection.json",
        "customer_handoff_report": "customer_handoff_report.json",
        "delivery_manifest": "delivery_manifest.json",
        "arena_rerun_plan": "arena_rerun_plan.json",
    }
    artifacts: Dict[str, str] = {}
    artifact_uris: Dict[str, str] = {}
    jobs: List[Dict[str, Any]] = []
    if jobs_dir.is_dir():
        for job_dir in sorted(path for path in jobs_dir.iterdir() if path.is_dir()):
            job_id = job_dir.name
            run_manifest = _read_optional_mapping(job_dir / "job_run_manifest.json")
            proof_boundary = _read_optional_mapping(job_dir / "proof_boundary.json")
            job_artifacts: Dict[str, str] = {}
            job_artifact_uris: Dict[str, str] = {}
            for key, filename in artifact_names.items():
                path = job_dir / filename
                if not path.is_file():
                    continue
                artifact_key = f"robot_eval_job_{job_id}_{key}"
                relative_path = _relative_to(eval_dir, path)
                artifacts[artifact_key] = relative_path
                artifact_uris[f"{artifact_key}_uri"] = _gs_uri(
                    context,
                    f"robot_eval_jobs/{job_id}/{filename}",
                )
                job_artifacts[key] = relative_path
                job_artifact_uris[f"{key}_uri"] = _gs_uri(
                    context,
                    f"robot_eval_jobs/{job_id}/{filename}",
                )
            jobs.append(
                {
                    "job_id": job_id,
                    "status": str(run_manifest.get("status") or "missing"),
                    "state": str(run_manifest.get("state") or "missing"),
                    "artifacts": job_artifacts,
                    "artifact_uris": job_artifact_uris,
                    "simulator_execution_proven": _strict_proven_flag(
                        proof_boundary, run_manifest, "simulator_execution_proven"
                    ),
                    "rank_fidelity_result_proven": _strict_proven_flag(
                        proof_boundary, run_manifest, "rank_fidelity_result_proven"
                    ),
                    "live_end_to_end_verified": _strict_proven_flag(
                        proof_boundary, run_manifest, "live_end_to_end_verified"
                    ),
                    "live_eval_closure_status": run_manifest.get("live_eval_closure_status")
                    or proof_boundary.get("live_eval_closure_status"),
                    "public_claim_upgrade_allowed": _strict_proven_flag(
                        proof_boundary, run_manifest, "public_claim_upgrade_allowed"
                    ),
                }
            )
    status = "missing"
    if jobs:
        status = next(
            (
                str(job.get("status"))
                for job in jobs
                if str(job.get("status")) not in {"blocked", "missing", ""}
            ),
            str(jobs[0].get("status") or "blocked"),
        )
        latest_job = jobs[-1]
        latest_artifact_uris = (
            latest_job.get("artifact_uris")
            if isinstance(latest_job.get("artifact_uris"), Mapping)
            else {}
        )
        alias_map = {
            "job_request_uri": "robot_eval_job_request_uri",
            "run_manifest_uri": "robot_eval_job_run_manifest_uri",
            "proof_boundary_uri": "robot_eval_job_proof_boundary_uri",
            "blocked_manifest_uri": "robot_eval_job_blocked_manifest_uri",
            "worker_manifest_uri": "robot_eval_job_worker_manifest_uri",
            "gpu_startup_pipeline_plan_uri": (
                "robot_eval_job_gpu_startup_pipeline_plan_uri"
            ),
            "gpu_provider_launcher_result_uri": (
                "robot_eval_job_gpu_provider_launcher_result_uri"
            ),
            "gpu_provider_launcher_stdout_log_uri": (
                "robot_eval_job_gpu_provider_launcher_stdout_log_uri"
            ),
            "gpu_provider_launcher_stderr_log_uri": (
                "robot_eval_job_gpu_provider_launcher_stderr_log_uri"
            ),
            "runpod_provider_adapter_result_uri": (
                "robot_eval_job_runpod_provider_adapter_result_uri"
            ),
            "startup_architecture_audit_uri": (
                "robot_eval_job_startup_architecture_audit_uri"
            ),
            "worker_runtime_manifest_uri": "robot_eval_job_worker_runtime_manifest_uri",
            "worker_runtime_preflight_uri": "robot_eval_job_worker_runtime_preflight_uri",
            "worker_runtime_preflight_stdout_log_uri": (
                "robot_eval_job_worker_runtime_preflight_stdout_log_uri"
            ),
            "worker_runtime_preflight_stderr_log_uri": (
                "robot_eval_job_worker_runtime_preflight_stderr_log_uri"
            ),
            "evaluation_result_uri": "robot_eval_job_evaluation_result_uri",
            "scenario_eval_matrix_uri": "robot_eval_job_scenario_eval_matrix_uri",
            "deployment_outcome_ledger_uri": "robot_eval_job_deployment_outcome_ledger_uri",
            "sim_vs_real_calibration_report_uri": (
                "robot_eval_job_sim_vs_real_calibration_report_uri"
            ),
            "real_world_validation_followup_plan_uri": (
                "robot_eval_job_real_world_validation_followup_plan_uri"
            ),
            "real_world_validation_followup_request_queue_uri": (
                "robot_eval_job_real_world_validation_followup_request_queue_uri"
            ),
            "live_eval_closure_manifest_uri": (
                "robot_eval_job_live_eval_closure_manifest_uri"
            ),
            "robot_eval_report_uri": "robot_eval_job_robot_eval_report_uri",
            "robot_eval_report_markdown_uri": "robot_eval_job_robot_eval_report_markdown_uri",
        "post_training_data_package_export_manifest_uri": (
            "robot_eval_job_post_training_data_package_export_manifest_uri"
        ),
        "webapp_robot_eval_status_projection_uri": (
            "robot_eval_job_webapp_robot_eval_status_projection_uri"
        ),
        "remote_cloud_execution_closure_manifest_uri": (
            "robot_eval_job_remote_cloud_execution_closure_manifest_uri"
        ),
        "robot_team_grade_eval_closure_manifest_uri": (
            "robot_eval_job_robot_team_grade_eval_closure_manifest_uri"
        ),
    }
        for source_key, alias_key in alias_map.items():
            value = latest_artifact_uris.get(source_key)
            if isinstance(value, str) and value:
                artifact_uris[alias_key] = value
        startup_alias_map = {
            "scheduler_decision_uri": "robot_eval_scheduler_decision_uri",
            "worker_launch_plan_uri": "robot_eval_worker_launch_plan_uri",
            "gpu_startup_pipeline_plan_uri": "robot_eval_gpu_startup_pipeline_plan_uri",
            "worker_manifest_uri": "robot_eval_worker_manifest_uri",
            "gpu_provider_launch_request_uri": "robot_eval_gpu_provider_launch_request_uri",
            "gpu_provider_launcher_result_uri": "robot_eval_gpu_provider_launcher_result_uri",
            "runpod_provider_adapter_result_uri": (
                "robot_eval_runpod_provider_adapter_result_uri"
            ),
            "gpu_cost_control_ledger_uri": "robot_eval_gpu_cost_control_ledger_uri",
            "startup_architecture_audit_uri": "robot_eval_startup_architecture_audit_uri",
            "worker_runtime_manifest_uri": "robot_eval_worker_runtime_manifest_uri",
            "worker_runtime_preflight_uri": "robot_eval_worker_runtime_preflight_uri",
        }
        for source_key, alias_key in startup_alias_map.items():
            value = latest_artifact_uris.get(source_key)
            if isinstance(value, str) and value:
                artifact_uris[alias_key] = value
    return {
        "schema_version": "robot_eval_job_evaluation_prep_surface.v1",
        "status": status,
        "job_count": len(jobs),
        "jobs": jobs,
        "artifacts": artifacts,
        "artifact_uris": artifact_uris,
        "simulator_execution_proven": any(
            bool(job.get("simulator_execution_proven")) for job in jobs
        ),
        "rank_fidelity_result_proven": any(
            bool(job.get("rank_fidelity_result_proven")) for job in jobs
        ),
        "public_claim_upgrade_allowed": any(
            bool(job.get("public_claim_upgrade_allowed")) for job in jobs
        ),
        "claim_boundary": (
            "robot_eval_job_artifacts_are_advisory_unless_live_eval_closure_manifest_is_verified"
        ),
    }


def _real_path_from_eval_dir(eval_dir: Path, relative_path: str) -> Optional[Path]:
    text = str(relative_path or "").strip()
    if not text:
        return None
    candidate = (eval_dir / text).resolve()
    return candidate if candidate.exists() else None


def _conditioning_local_paths(*, context, conditioning_bundle: Mapping[str, Any]) -> Dict[str, str]:
    # PIPE-01: the un-redacted raw walkthrough (raw_root/walkthrough.mov) is NEVER
    # exported into the buyer-facing site_world_spec conditioning. Only the
    # privacy-processed world-model video path is surfaced. Do not add raw_video_path
    # back — it embeds the raw capture into the launchable artifact.
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
        "world_model_video_path": str(world_model_video_path) if world_model_video_path.is_file() else "",
        "keyframe_path": str(keyframe_path) if keyframe_path is not None else "",
        "arkit_poses_path": str(context.raw_root / "arkit" / "poses.jsonl"),
        "arkit_intrinsics_path": str(context.raw_root / "arkit" / "intrinsics.json"),
        "arkit_depth_path": str(context.raw_root / "arkit" / "depth"),
        "privacy_depth_manifest_path": str(context.pipeline_root / "privacy_depth" / "depth_manifest.json"),
        "privacy_confidence_manifest_path": str(context.pipeline_root / "privacy_depth" / "confidence_manifest.json"),
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
                "world_model_backend": "site_world_runtime",
                "primary_runtime_backend": "site_world_runtime",
                "scene_representation": str(manifest.get("scene_representation") or "site_world_runtime_video_world_model_v1"),
                "render_source": str(manifest.get("render_source") or "site_world_runtime_full_capture"),
                "fallback_mode": str(manifest.get("fallback_mode") or "none"),
                "evidence_mode": "full_capture_persistent_scene",
                "primary_render_asset_role": "authoritative_runtime_render_asset",
                "renderer_backend": str(manifest.get("renderer_backend") or "site_world_runtime"),
                "bundle_type": str(manifest.get("bundle_type") or manifest.get("scene_representation") or "site_world_runtime_video_world_model_v1"),
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
        "world_model_backend": "site_world_runtime",
        "primary_runtime_backend": "site_world_runtime",
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
            "world_model_backend": str(canonical_world_model.get("world_model_backend") or "site_world_runtime"),
            "scene_representation": str(canonical_world_model.get("scene_representation") or "site_world_runtime_video_world_model_v1"),
            "runtime_render_source": str(canonical_world_model.get("render_source") or "canonical_world_model"),
            "fallback_mode": str(canonical_world_model.get("fallback_mode") or "none"),
        }

    # Never retain or fall back to the raw, unredacted walkthrough in derived runtime
    # artifacts. The scene-memory writer emits only privacy-processed references.
    privacy_video_ref = str(
        conditioning_bundle.get("world_model_video_uri")
        or conditioning_bundle.get("privacy_processed_video_uri")
        or local_paths.get("world_model_video_path")
        or ""
    ).strip()
    arkit = dict(conditioning_bundle.get("arkit") or {}) if isinstance(conditioning_bundle.get("arkit"), Mapping) else {}
    geometry = dict(conditioning_bundle.get("geometry") or {}) if isinstance(conditioning_bundle.get("geometry"), Mapping) else {}
    poses_ref = str(arkit.get("poses_uri") or local_paths.get("arkit_poses_path") or "").strip()
    intrinsics_ref = str(arkit.get("intrinsics_uri") or local_paths.get("arkit_intrinsics_path") or "").strip()
    if privacy_video_ref and poses_ref and intrinsics_ref:
        return {
            "world_model_backend": "site_world_runtime",
            "scene_representation": "site_world_runtime_video_world_model_v1",
            "runtime_render_source": "site_world_runtime_full_capture",
            "fallback_mode": "arkit_rgbd_last_resort",
        }
    geometry_poses_ref = str(geometry.get("poses_uri") or local_paths.get("geometry_poses_path") or "").strip()
    geometry_intrinsics_ref = str(
        geometry.get("intrinsics_uri") or local_paths.get("geometry_intrinsics_path") or ""
    ).strip()
    geometry_depth_ref = str(
        geometry.get("depth_manifest_uri") or local_paths.get("geometry_depth_manifest_path") or ""
    ).strip()
    if privacy_video_ref and geometry_poses_ref and geometry_intrinsics_ref and geometry_depth_ref:
        return {
            "world_model_backend": "site_world_runtime",
            "scene_representation": "geometry_conditioned_capture_v1",
            "runtime_render_source": "geometry_conditioned_capture",
            "fallback_mode": "geometry_lane_conditioning",
        }
    return {
        "world_model_backend": str(canonical_world_model.get("world_model_backend") or "site_world_runtime"),
        "scene_representation": str(canonical_world_model.get("scene_representation") or "unavailable"),
        "runtime_render_source": str(canonical_world_model.get("render_source") or "unavailable"),
        "fallback_mode": str(canonical_world_model.get("fallback_mode") or "arkit_rgbd_last_resort"),
    }


def _native_world_model_semantics(
    *,
    context,
    canonical_world_model: Mapping[str, Any],
    runtime_render_descriptor: Mapping[str, Any],
    scene_memory_bundle_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    canonical_status = str(canonical_world_model.get("status") or "").strip().lower()
    runtime_render_source = str(runtime_render_descriptor.get("runtime_render_source") or "").strip().lower()
    preview_manifest_available = bool(
        str(scene_memory_bundle_manifest.get("preview_simulation_manifest_path") or "").strip()
    )

    if canonical_status == "ready":
        native_world_model_path = "authoritative_native_render"
    elif runtime_render_source == "geometry_conditioned_capture":
        native_world_model_path = "geometry_conditioned_native_path"
    elif runtime_render_source == "site_world_runtime_full_capture":
        native_world_model_path = "full_capture_native_path"
    else:
        native_world_model_path = None

    native_world_model_primary = native_world_model_path is not None
    provider_fallback_preview_status = "fallback_available" if preview_manifest_available else "not_requested"
    provider_fallback_only = (
        not native_world_model_primary and provider_fallback_preview_status == "fallback_available"
    )

    authoritative_manifest_path = (
        context.pipeline_root / "presentation_world" / "authoritative_runtime_render_manifest.json"
    )
    authoritative_runtime_render_manifest_uri = (
        _gs_uri(context, "presentation_world/authoritative_runtime_render_manifest.json")
        if authoritative_manifest_path.is_file()
        else None
    )

    return {
        "native_world_model_status": "primary_ready" if native_world_model_primary else "not_ready",
        "native_world_model_primary": native_world_model_primary,
        "native_world_model_path": native_world_model_path,
        "provider_fallback_preview_status": provider_fallback_preview_status,
        "provider_fallback_only": provider_fallback_only,
        "authoritative_runtime_render_manifest_uri": authoritative_runtime_render_manifest_uri,
        "fallback_order": [
            "authoritative_native_render",
            "geometry_conditioned_native_path",
            "provider_fallback_preview",
        ],
    }


def _artifact_family_payload(
    *,
    context,
    native_semantics: Mapping[str, Any],
    scene_memory_bundle_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    presentation_bundle_available = bool(
        str(scene_memory_bundle_manifest.get("presentation_bundle_path") or "").strip()
    )
    presentation_manifest_available = bool(
        str(scene_memory_bundle_manifest.get("presentation_world_manifest_path") or "").strip()
    )
    runtime_demo_available = bool(
        str(scene_memory_bundle_manifest.get("runtime_demo_manifest_path") or "").strip()
    )
    preview_available = bool(
        str(scene_memory_bundle_manifest.get("preview_simulation_manifest_path") or "").strip()
    )
    return {
        "canonical_native_artifacts": {
            "status": str(native_semantics.get("native_world_model_status") or "not_ready"),
            "primary": bool(native_semantics.get("native_world_model_primary")),
            "path": native_semantics.get("native_world_model_path"),
            "artifacts": [
                _gs_uri(context, "evaluation_prep/site_world_spec.json"),
                _gs_uri(context, "evaluation_prep/site_world_registration.json"),
                _gs_uri(context, "evaluation_prep/site_world_health.json"),
                *(
                    [str(native_semantics.get("authoritative_runtime_render_manifest_uri"))]
                    if str(native_semantics.get("authoritative_runtime_render_manifest_uri") or "").strip()
                    else []
                ),
            ],
        },
        "derived_presentation_demo_artifacts": {
            "status": (
                "ready"
                if presentation_bundle_available and presentation_manifest_available and runtime_demo_available
                else "partial"
                if presentation_bundle_available or presentation_manifest_available or runtime_demo_available
                else "missing"
            ),
            "artifacts": [
                *(
                    [_gs_uri(context, "presentation_world/presentation_bundle.json")]
                    if presentation_bundle_available
                    else []
                ),
                *(
                    [_gs_uri(context, "presentation_world/presentation_world_manifest.json")]
                    if presentation_manifest_available
                    else []
                ),
                *(
                    [_gs_uri(context, "presentation_world/runtime_demo_manifest.json")]
                    if runtime_demo_available
                    else []
                ),
            ],
        },
        "provider_fallback_artifacts": {
            "status": str(native_semantics.get("provider_fallback_preview_status") or "not_requested"),
            "provider_fallback_only": bool(native_semantics.get("provider_fallback_only")),
            "artifacts": (
                [_gs_uri(context, "preview_simulation/preview_simulation_manifest.json")]
                if preview_available
                else []
            ),
        },
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


# Privacy pipeline statuses that mean people are provably not present or fully
# removed. Fallback redaction statuses intentionally do not appear here because
# build_rights_provenance_review marks them needs_review, not cleared. A capture
# whose raw privacy pipeline status is not in this set has NOT been privacy-cleared
# and must never reach a buyer-facing "launchable" artifact.
_PRIVACY_CLEARED_STATUSES = frozenset(
    {
        "no_people_detected",
        "person_removed",
    }
)


def _privacy_processing_cleared(
    *,
    rights_review: Mapping[str, Any] | None,
    privacy_processing: Mapping[str, Any] | None,
) -> bool:
    """Return True only when privacy post-processing is provably complete/redacted.

    Reads the normalized rights-provenance review privacy verdict first (cleared /
    needs_review / blocked) and falls back to the raw privacy pipeline status
    cleared set. Fails closed when neither source proves clearance.
    """
    review = rights_review if isinstance(rights_review, Mapping) else {}
    review_privacy = review.get("privacy") if isinstance(review.get("privacy"), Mapping) else {}
    review_privacy_status = str(review_privacy.get("status") or "").strip().lower()
    if review_privacy_status:
        return review_privacy_status == "cleared"
    raw_status = str(
        (privacy_processing or {}).get("status") if isinstance(privacy_processing, Mapping) else ""
    ).strip().lower()
    return raw_status in _PRIVACY_CLEARED_STATUSES


def _rights_review_cleared(rights_review: Mapping[str, Any] | None) -> bool:
    """Return True only when the rights-provenance review clears derived rights."""
    review = rights_review if isinstance(rights_review, Mapping) else {}
    nested = review.get("rights") if isinstance(review.get("rights"), Mapping) else {}
    nested_status = str(nested.get("status") or "").strip().lower()
    if nested_status:
        return nested_status == "cleared"
    # Fall back to the overall review verdict when the nested rights block is absent.
    return str(review.get("status") or "").strip().lower() == "cleared"


def _canonical_site_world_runtime_status(
    *,
    qualification_state: object,
    downstream_evaluation_eligibility: bool,
    scene_memory_bundle_manifest: Mapping[str, Any],
    object_geometry_manifest: Mapping[str, Any],
    protected_regions_manifest: Mapping[str, Any],
    required_runtime_artifact_paths: Sequence[Path],
    runtime_service_url: str,
    rights_review: Mapping[str, Any] | None = None,
    privacy_processing: Mapping[str, Any] | None = None,
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

    # PIPE-01 (beta launch audit): privacy + rights are authoritative. Raw,
    # un-redacted, or rights-unverified captures must NOT be marked launchable, since
    # `launchable` flows into site_world_health -> launchable_export_bundle "ready"
    # and satisfies the buyer_fulfillment_bundle_ready launch gate. These are HARD
    # blockers, not warnings.
    if not _privacy_processing_cleared(
        rights_review=rights_review,
        privacy_processing=privacy_processing,
    ):
        privacy_review = (
            rights_review.get("privacy") if isinstance(rights_review, Mapping) else None
        )
        privacy_status_label = str(
            (privacy_review or {}).get("status")
            if isinstance(privacy_review, Mapping)
            else ((privacy_processing or {}).get("status") if isinstance(privacy_processing, Mapping) else "")
        ).strip() or "unavailable"
        blockers.append(f"privacy_processing_not_cleared:{privacy_status_label}")
    if not _rights_review_cleared(rights_review):
        rights_status_label = ""
        if isinstance(rights_review, Mapping):
            nested = rights_review.get("rights") if isinstance(rights_review.get("rights"), Mapping) else {}
            rights_status_label = str(
                nested.get("status") or rights_review.get("status") or ""
            ).strip()
        blockers.append(f"rights_review_not_cleared:{rights_status_label or 'unavailable'}")

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
    native_semantics = _native_world_model_semantics(
        context=context,
        canonical_world_model=canonical_world_model,
        runtime_render_descriptor=runtime_render_descriptor,
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
    )
    artifact_families = _artifact_family_payload(
        context=context,
        native_semantics=native_semantics,
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
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
        "site_submission_id": str(normalized_handoff.get("site_submission_id") or ""),
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
            "privacy_processed_video_uri": (
                conditioning_map.get("world_model_video_uri")
                or conditioning_map.get("privacy_processed_video_uri")
            ),
            "keyframe_uri": conditioning_map.get("keyframe_uri"),
            "arkit_poses_uri": ((conditioning_map.get("arkit") or {}) if isinstance(conditioning_map.get("arkit"), Mapping) else {}).get("poses_uri"),
            "arkit_intrinsics_uri": ((conditioning_map.get("arkit") or {}) if isinstance(conditioning_map.get("arkit"), Mapping) else {}).get("intrinsics_uri"),
            "arkit_depth_uri": ((conditioning_map.get("arkit") or {}) if isinstance(conditioning_map.get("arkit"), Mapping) else {}).get("depth_prefix_uri"),
            "depth_conditioning": (
                dict(conditioning_map.get("depth_conditioning"))
                if isinstance(conditioning_map.get("depth_conditioning"), Mapping)
                else {}
            ),
            "privacy_depth_manifest_uri": (
                ((conditioning_map.get("depth_conditioning") or {}) if isinstance(conditioning_map.get("depth_conditioning"), Mapping) else {}).get("depth_manifest_uri")
            ),
            "privacy_confidence_manifest_uri": (
                ((conditioning_map.get("depth_conditioning") or {}) if isinstance(conditioning_map.get("depth_conditioning"), Mapping) else {}).get("confidence_manifest_uri")
            ),
            "geometry_manifest_uri": ((conditioning_map.get("geometry") or {}) if isinstance(conditioning_map.get("geometry"), Mapping) else {}).get("manifest_uri"),
            "geometry_summary_uri": ((conditioning_map.get("geometry") or {}) if isinstance(conditioning_map.get("geometry"), Mapping) else {}).get("summary_uri"),
            "geometry_poses_uri": ((conditioning_map.get("geometry") or {}) if isinstance(conditioning_map.get("geometry"), Mapping) else {}).get("poses_uri"),
            "geometry_intrinsics_uri": ((conditioning_map.get("geometry") or {}) if isinstance(conditioning_map.get("geometry"), Mapping) else {}).get("intrinsics_uri"),
            "geometry_depth_manifest_uri": ((conditioning_map.get("geometry") or {}) if isinstance(conditioning_map.get("geometry"), Mapping) else {}).get("depth_manifest_uri"),
            "geometry_confidence_manifest_uri": ((conditioning_map.get("geometry") or {}) if isinstance(conditioning_map.get("geometry"), Mapping) else {}).get("confidence_manifest_uri"),
            "sensor_availability": scene_memory_capture.get("sensor_availability", {}),
            "local_paths": local_paths,
        },
        "primary_runtime_backend": "site_world_runtime",
        "canonical_world_model": canonical_world_model,
        "native_world_model_status": native_semantics["native_world_model_status"],
        "native_world_model_primary": native_semantics["native_world_model_primary"],
        "native_world_model_path": native_semantics["native_world_model_path"],
        "provider_fallback_preview_status": native_semantics["provider_fallback_preview_status"],
        "provider_fallback_only": native_semantics["provider_fallback_only"],
        "fallback_order": native_semantics["fallback_order"],
        "artifact_families": artifact_families,
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
                or "site_world_runtime"
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
        "primary_runtime_backend": "site_world_runtime",
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
        "primary_runtime_backend": "site_world_runtime",
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
    try:
        remote_site_world = dict(
            client.get_site_world(str(registration.get("site_world_id") or site_world_id))
        )
    except Exception:
        remote_site_world = {}
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
    verification_blockers: List[str] = []
    remote_build_id = str(remote_site_world.get("build_id") or "").strip()
    if remote_build_id and str(registration.get("build_id") or "").strip() and remote_build_id != str(registration.get("build_id") or "").strip():
        verification_blockers.append("runtime_registered_build_id_mismatch")
    remote_package_version = str(remote_site_world.get("canonical_package_version") or "").strip()
    local_package_version = str(spec.get("canonical_package_version") or "").strip()
    if remote_package_version and local_package_version and remote_package_version != local_package_version:
        verification_blockers.append("runtime_registered_package_version_mismatch")
    if not str(remote_site_world.get("runtime_base_url") or registration.get("runtime_base_url") or "").strip():
        verification_blockers.append("runtime_base_url_missing_after_registration")
    if not bool(health.get("healthy")) or not bool(health.get("launchable")):
        verification_blockers.append("runtime_health_not_launchable")
    if verification_blockers:
        registration["status"] = "blocked"
        registration["blockers"] = list(
            dict.fromkeys([*list(registration.get("blockers") or []), *verification_blockers])
        )
        registration["runtime_capabilities"] = _runtime_capabilities_payload(
            launchable=False,
            base=registration.get("runtime_capabilities") if isinstance(registration.get("runtime_capabilities"), Mapping) else {},
        )
        health = {
            **health,
            "healthy": False,
            "launchable": False,
            "status": "degraded",
            "blockers": list(dict.fromkeys([*list(health.get("blockers") or []), *verification_blockers])),
        }
    runtime_smoke = {
        "schema_version": "v1",
        "attempted": False,
        "status": "not_run",
        "session_id": None,
        "session_created": False,
        "session_reset": False,
        "runtime_base_url": registration.get("runtime_base_url") or health.get("runtime_base_url"),
        "blockers": [],
    }
    if registration.get("status") == "ready":
        if task_catalog and scenario_catalog and start_state_catalog and robot_profiles:
            try:
                runtime_smoke["attempted"] = True
                session = client.create_session(
                    str(registration["site_world_id"]),
                    robot_profile_id=str((robot_profiles[0] or {}).get("id") or "mobile_manipulator_rgb_v1"),
                    task_id=str((task_catalog[0] or {}).get("id") or (task_catalog[0] or {}).get("task_id") or ""),
                    scenario_id=str((scenario_catalog[0] or {}).get("id") or ""),
                    start_state_id=str((start_state_catalog[0] or {}).get("id") or ""),
                    notes="pipeline_runtime_smoke",
                )
                session_id = str(session.get("session_id") or "").strip()
                runtime_smoke["session_id"] = session_id or None
                runtime_smoke["session_created"] = bool(session_id)
                client.reset_session(session_id)
                runtime_smoke["session_reset"] = True
                runtime_smoke["status"] = "succeeded"
                health = dict(client.get_site_world_health(str(registration["site_world_id"])))
            except Exception as exc:
                runtime_smoke["status"] = "failed"
                runtime_smoke["blockers"] = [f"runtime_smoke_failed:{exc}"]
                health = {
                    **health,
                    "healthy": False,
                    "launchable": False,
                    "status": "degraded",
                    "blockers": list(health.get("blockers") or []) + [f"runtime_smoke_failed:{exc}"],
                    "last_heartbeat_at": utc_now_iso(),
                }
                health["runtime_capabilities"] = _runtime_capabilities_payload(launchable=False, base=health.get("runtime_capabilities"))
        else:
            runtime_smoke["status"] = "blocked"
            runtime_smoke["blockers"] = ["runtime_smoke_catalogs_missing"]
    registration.setdefault("blockers", list(canonical_runtime_status.get("blockers") or []))
    registration.setdefault("warnings", list(canonical_runtime_status.get("warnings") or []))
    registration.setdefault("task_catalog", task_catalog)
    registration.setdefault("scenario_catalog", scenario_catalog)
    registration.setdefault("start_state_catalog", start_state_catalog)
    registration.setdefault("robot_profiles", robot_profiles)
    registration.setdefault("supported_cameras", [])
    registration["primary_runtime_backend"] = spec.get("primary_runtime_backend")
    registration["canonical_world_model"] = dict(spec.get("canonical_world_model") or {})
    registration["native_world_model_status"] = spec.get("native_world_model_status")
    registration["native_world_model_primary"] = spec.get("native_world_model_primary")
    registration["provider_fallback_preview_status"] = spec.get("provider_fallback_preview_status")
    registration["provider_fallback_only"] = spec.get("provider_fallback_only")
    registration["artifact_families"] = dict(spec.get("artifact_families") or {})
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
    health["native_world_model_status"] = spec.get("native_world_model_status")
    health["native_world_model_primary"] = spec.get("native_world_model_primary")
    health["provider_fallback_preview_status"] = spec.get("provider_fallback_preview_status")
    health["provider_fallback_only"] = spec.get("provider_fallback_only")
    health["artifact_families"] = dict(spec.get("artifact_families") or {})
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
    if runtime_required() and runtime_smoke.get("status") != "succeeded":
        blocker = "runtime_session_smoke_required"
        registration["status"] = "blocked"
        registration["blockers"] = list(dict.fromkeys([*list(registration.get("blockers") or []), blocker]))
        registration["runtime_capabilities"] = _runtime_capabilities_payload(
            launchable=False,
            base=registration.get("runtime_capabilities")
            if isinstance(registration.get("runtime_capabilities"), Mapping)
            else {},
        )
        health = {
            **health,
            "healthy": False,
            "launchable": False,
            "status": "degraded",
            "blockers": list(dict.fromkeys([*list(health.get("blockers") or []), blocker])),
        }
        health["runtime_capabilities"] = _runtime_capabilities_payload(launchable=False, base=health.get("runtime_capabilities"))
    registration["runtime_smoke"] = dict(runtime_smoke)
    health["runtime_smoke"] = dict(runtime_smoke)
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
    native_semantics = _native_world_model_semantics(
        context=context,
        canonical_world_model=canonical_world_model,
        runtime_render_descriptor=runtime_render_descriptor,
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
    )
    adapter_key_map = {
        "site_world_runtime": "site_world_runtime_adapter_manifest_path",
        "gen3c": "gen3c_adapter_manifest_path",
        "cosmos_transfer": "cosmos_transfer_adapter_manifest_path",
    }
    adapter_details = _adapter_manifest_details(scene_memory_bundle_manifest, eval_dir=eval_dir)
    backend_variants = _build_runtime_backend_variants(
        context=context,
        eval_dir=eval_dir,
        pipeline_dir=context.capture_root / "pipeline",
        scene_memory_bundle_manifest=scene_memory_bundle_manifest,
        native_semantics=native_semantics,
    )
    available_backends = list(backend_variants.keys())
    launchable_backends = [
        backend
        for backend, detail in backend_variants.items()
        if bool(detail.get("launchable"))
    ]
    preferred_order = ["site_world_runtime", "cosmos_predict_lora_adapter", "gen3c", "cosmos_transfer"]
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

    claim_boundary = _hosted_session_runtime_claim_boundary()
    proof_boundary = {
        **claim_boundary,
        "artifact_purpose": "hosted_session_runtime_proof_boundary",
    }

    return {
        "schema_version": "v1",
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "site_submission_id": str(normalized_handoff.get("site_submission_id") or ""),
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
        "primary_runtime_backend": "site_world_runtime",
        "canonical_world_model": canonical_world_model,
        "native_world_model_status": native_semantics["native_world_model_status"],
        "native_world_model_primary": native_semantics["native_world_model_primary"],
        "provider_fallback_preview_status": native_semantics["provider_fallback_preview_status"],
        "provider_fallback_only": native_semantics["provider_fallback_only"],
        "world_model_backend": runtime_render_descriptor["world_model_backend"],
        "scene_representation": runtime_render_descriptor["scene_representation"],
        "render_source": runtime_render_descriptor["runtime_render_source"],
        "fallback_mode": runtime_render_descriptor["fallback_mode"],
        "available_backends": available_backends,
        "launchable_backends": launchable_backends,
        "default_backend": default_backend,
        "backend_variants": backend_variants,
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
        "simulator_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "deployment_approval_proven": False,
        "safety_validation_proven": False,
        "physical_readiness_proven": False,
        "physical_robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": claim_boundary,
        "proof_boundary": proof_boundary,
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
                "status": (
                    adapter_details.get(backend, {}).get("status")
                    if backend in adapter_key_map
                    else backend_variants.get(backend, {}).get("readiness_state")
                ),
                "execution_mode": (
                    adapter_details.get(backend, {}).get("execution_mode")
                    if backend in adapter_key_map
                    else backend_variants.get(backend, {}).get("runtime_mode")
                ),
                "required_conditioning": (
                    adapter_details.get(backend, {}).get("required_conditioning", [])
                    if backend in adapter_key_map
                    else []
                ),
                "service_contract_version": (
                    adapter_details.get(backend, {}).get("service_contract_version")
                    if backend in adapter_key_map
                    else None
                ),
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
        "site_submission_id": str(normalized_handoff.get("site_submission_id") or ""),
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
    geometry_truth = _geometry_conditioning_truth(scene_memory_bundle_manifest)
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
            "backend": "site_world_runtime",
        },
        "geometry_conditioning": {
            "launchable": bool(geometry_truth["launchable"]),
            "required_artifacts": [
                "geometry_manifest",
                "geometry_summary",
                "geometry_poses",
                "geometry_intrinsics",
                "geometry_depth_manifest",
            ],
            "backend": "geometry_lane",
            "geometry_source": geometry_truth["geometry_source"],
            "fallback_used": geometry_truth["fallback_used"],
            "fallback_kind": geometry_truth["fallback_kind"],
            "ready_for_world_model": geometry_truth["ready_for_world_model"],
            "contract_ready_for_world_model": geometry_truth["contract_ready_for_world_model"],
            "internal_fallback_ready": geometry_truth["internal_fallback_ready"],
            "geometry_live_ready": geometry_truth["geometry_live_ready"],
            "site_faithful_market_ready": geometry_truth["site_faithful_market_ready"],
            "provider_native_result": geometry_truth["provider_native_result"],
            "site_frame_available": geometry_truth["site_frame_available"],
            "scale_resolved": geometry_truth["scale_resolved"],
            "local_reference_ready": geometry_truth["local_reference_ready"],
            "provider_native_geometry_ready": geometry_truth["provider_native_geometry_ready"],
            "non_arkit_geometry_state": geometry_truth["non_arkit_geometry_state"],
            "blockers": geometry_truth["blockers"],
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
            "backend": "site_world_runtime_ui",
            "blockers": demo_readiness["blockers"],
        },
    }
    runtime_export_required = runtime_required()
    runtime_blockers = list(site_world_health.get("blockers") or [])
    if runtime_export_required and not bool(site_world_health.get("launchable")):
        runtime_blockers = list(dict.fromkeys([*runtime_blockers, "runtime_required_for_buyer_launch"]))
        bundles["world_model_runtime"]["blockers"] = runtime_blockers
    full_export_ready = any(
        item["launchable"]
        for name, item in bundles.items()
        if name != "geometry_conditioning"
    )
    partial_ready = full_export_ready or bool(bundles["geometry_conditioning"]["launchable"])
    status = "ready" if full_export_ready else "partial" if partial_ready else "partial"
    if runtime_export_required and not bool(site_world_health.get("launchable")):
        status = "blocked"
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": status,
        "runtime_required": runtime_export_required,
        "launch_blockers": runtime_blockers if status == "blocked" else [],
        "public_runtime_label": "Site world runtime",
        "default_backend": "site_world_runtime",
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
    geometry_truth = _geometry_conditioning_truth(scene_memory_bundle_manifest)
    geometry_conditioning_ready = bool(geometry_truth["launchable"])
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
            "detail": "Geometry-lane conditioning is backed by live video_to_world output."
            if geometry_conditioning_ready
            else (
                "Geometry-lane conditioning is missing, fallback-only, or not live video_to_world: "
                + ", ".join(geometry_truth["blockers"] or ["unknown"])
            ),
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
    # PIPE-01: privacy/rights verdicts feed the canonical runtime "launchable" gate.
    # Load the authoritative rights-provenance review + privacy processing manifest so a
    # not-cleared capture is never marked launchable / embedded into the buyer-facing spec.
    rights_provenance_review_gate = (
        optional_read_json(pipeline_dir / "rights_provenance_review.json") or {}
    )
    privacy_processing_gate = (
        optional_read_json(pipeline_dir / "privacy_processing_manifest.json") or {}
    )

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

    runtime_service_url = (os.getenv("SITE_WORLD_RUNTIME_SERVICE_URL") or "").strip().rstrip("/")
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
        rights_review=rights_provenance_review_gate,
        privacy_processing=privacy_processing_gate,
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
        rights_review=rights_provenance_review_gate,
        privacy_processing=privacy_processing_gate,
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
    for target in (site_world_spec,):
        target["default_backend"] = hosted_session_runtime_manifest.get("default_backend")
        target["launchable_backends"] = list(hosted_session_runtime_manifest.get("launchable_backends") or [])
        target["backend_variants"] = dict(hosted_session_runtime_manifest.get("backend_variants") or {})
    _copy_json(site_world_spec_path, site_world_spec)

    site_world_registration, site_world_health = _build_site_world_runtime_records(
        context=context,
        spec=site_world_spec,
        canonical_runtime_status=canonical_runtime_status,
    )
    site_world_registration.setdefault("canonical_package_version", canonical_package_version)
    site_world_health.setdefault("canonical_package_version", canonical_package_version)
    for target in (site_world_registration, site_world_health):
        target["default_backend"] = hosted_session_runtime_manifest.get("default_backend")
        target["launchable_backends"] = list(hosted_session_runtime_manifest.get("launchable_backends") or [])
        target["backend_variants"] = dict(hosted_session_runtime_manifest.get("backend_variants") or {})
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

    simready_assets: Dict[str, Any] = {
        "schema_version": "simready_assets_result.v1",
        "status": "not_requested",
        "reason": "legacy_simready_eval_prep_auto_build_disabled",
    }
    if _env_truthy("BLUEPRINT_ALLOW_LEGACY_SIMREADY_EVAL_PREP"):
        from .simready_assets import build_simready_assets

        simready_assets = build_simready_assets(
            capture_root=context.capture_root,
            object_geometry_manifest=object_geometry_manifest
            if isinstance(object_geometry_manifest, Mapping)
            else {},
            task_anchor_manifest=task_anchor_manifest,
            site_world_spec=site_world_spec,
            hosted_session_runtime_manifest=hosted_session_runtime_manifest,
        )
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

    marble_sim_assets: Dict[str, Any] = {}
    marble_dir = pipeline_dir / "marble_sim_assets"
    marble_world_manifest_path = pipeline_dir / "worldlabs_world_manifest.json"
    if marble_world_manifest_path.is_file() and _env_truthy("BLUEPRINT_ALLOW_LEGACY_MARBLE_EVAL_PREP"):
        from .marble_sim_assets import build_marble_sim_assets

        marble_sim_assets = build_marble_sim_assets(capture_root=context.capture_root)
    marble_simready_bridge_path = marble_dir / "marble_simready_bridge.json"
    marble_asset_validation_path = marble_dir / "marble_asset_validation.json"
    marble_simready_bridge = _read_optional_json_any(marble_simready_bridge_path)
    marble_asset_validation = _read_optional_json_any(marble_asset_validation_path)
    marble_asset_validation_status = (
        str(marble_asset_validation.get("overall_status") or "")
        if isinstance(marble_asset_validation, Mapping)
        else ""
    )
    if not marble_sim_assets and isinstance(marble_simready_bridge, Mapping):
        marble_sim_assets = {
            "schema_version": "marble_sim_assets_result.v1",
            "status": str(marble_simready_bridge.get("status") or "unknown"),
            "bridge_path": str(marble_simready_bridge_path.resolve()),
            "validation_path": str(marble_asset_validation_path.resolve())
            if marble_asset_validation_path.is_file()
            else "",
        }
    if not marble_sim_assets:
        marble_sim_assets = {
            "schema_version": "marble_sim_assets_result.v1",
            "status": "not_requested",
            "reason": "legacy_marble_eval_prep_auto_build_disabled",
        }

    from .robot_eval_dataset import build_real_site_robot_eval_dataset

    robot_eval_dataset = build_real_site_robot_eval_dataset(
        capture_root=context.capture_root,
        object_geometry_manifest=object_geometry_manifest
        if isinstance(object_geometry_manifest, Mapping)
        else {},
        task_anchor_manifest=task_anchor_manifest,
        site_world_spec=site_world_spec,
        hosted_session_runtime_manifest=hosted_session_runtime_manifest,
    )
    robot_eval_dir = pipeline_dir / "robot_eval_dataset"
    robot_eval_dataset_manifest_path = robot_eval_dir / "robot_eval_dataset_manifest.json"
    robot_eval_legacy_manifest_path = (
        robot_eval_dir / "real_site_robot_eval_dataset_manifest.json"
    )
    robot_site_card_path = robot_eval_dir / "site_card.json"
    robot_task_cards_path = robot_eval_dir / "task_cards.json"
    robot_scenario_cards_path = robot_eval_dir / "scenario_cards.json"
    robot_eval_cards_path = robot_eval_dir / "eval_cards.json"
    robot_annotation_backlog_path = robot_eval_dir / "annotation_backlog.json"
    robot_proof_boundaries_path = robot_eval_dir / "proof_boundaries.json"
    robot_task_library_path = robot_eval_dir / "robot_task_library.json"
    robot_task_ontology_path = robot_eval_dir / "task_ontology_v1.json"
    robot_scenario_library_path = robot_eval_dir / "scenario_library.json"
    robot_scenario_family_library_path = robot_eval_dir / "scenario_family_library.json"
    robot_pov_requirements_path = robot_eval_dir / "robot_pov_evidence_requirements.json"
    human_demo_requirements_path = robot_eval_dir / "human_demo_evidence_requirements.json"
    robot_eval_inputs_evidence_contract_path = (
        robot_eval_dir / "robot_eval_inputs_evidence_contract.json"
    )
    robot_team_test_submission_modalities_path = (
        robot_eval_dir / "robot_team_test_submission_modalities.json"
    )
    robot_failure_taxonomy_path = robot_eval_dir / "failure_taxonomy.json"
    prediction_outcome_ledger_path = robot_eval_dir / "prediction_outcome_ledger.json"
    prediction_vs_actual_summary_path = robot_eval_dir / "prediction_vs_actual_summary.json"
    robot_scoring_methodology_path = robot_eval_dir / "scoring_methodology.json"
    robot_task_thresholds_path = robot_eval_dir / "task_thresholds.json"
    robot_publication_readiness_path = robot_eval_dir / "publication_readiness.json"
    recorded_trace_eval_report_path = robot_eval_dir / "recorded_trace_eval_report.json"
    policy_eval_report_path = robot_eval_dir / "policy_eval_report.json"
    robot_rights_packet_path = robot_eval_dir / "rights_packet.json"
    robot_rights_ledger_path = robot_eval_dir / "rights_ledger.json"
    eval_methodology_summary_path = robot_eval_dir / "eval_methodology_summary.md"

    recapture_diff = _build_recapture_diff(
        capture_root=context.capture_root,
        current_capture_id=context.capture_id,
        site_normalization_package=site_normalization_package,
        benchmark_suite_manifest=benchmark_suite_manifest,
    )
    recapture_diff_path = eval_dir / "recapture_diff.json"
    _copy_json(recapture_diff_path, recapture_diff)

    backend_support = resolve_backend_support_artifacts(
        pipeline_dir, backend_id="cosmos_predict2_5"
    )
    cosmos_training_export_path, cosmos_training_export, _ = backend_support["cosmos_training_export"]
    cosmos_training_run_path, cosmos_training_run, _ = backend_support["cosmos_lora_training"]
    cosmos_zero_shot_benchmark_path, cosmos_zero_shot_benchmark, _ = backend_support["cosmos_zero_shot_benchmark"]
    simulation_automation_surface = simulation_automation_evaluation_prep_surface(
        capture_root=context.capture_root,
        eval_dir=eval_dir,
    )
    palatial_physready_surface = palatial_physready_evaluation_prep_surface(
        capture_root=context.capture_root,
        eval_dir=eval_dir,
    )
    site_eval_director_surface = site_eval_director_evaluation_prep_surface(
        capture_root=context.capture_root,
        eval_dir=eval_dir,
    )
    robot_eval_job_surface = robot_eval_job_evaluation_prep_surface(
        capture_root=context.capture_root,
        eval_dir=eval_dir,
    )

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
    geometry_truth = _geometry_conditioning_truth(scene_memory_bundle_manifest)
    geometry_conditioning_status = (
        "live_video_to_world"
        if geometry_truth["launchable"]
        else "fallback_not_launchable"
        if geometry_truth["fallback_used"]
        else "missing_or_not_ready"
    )
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
        "simready_asset_lane_status": simready_assets.get("status"),
        "marble_sim_asset_lane_status": marble_sim_assets.get("status"),
        "marble_asset_validation_status": marble_asset_validation_status or None,
        "palatial_physready_status": palatial_physready_surface.get("status"),
        "palatial_physready_model_derived_support_assets_present": (
            palatial_physready_surface.get("model_derived_support_assets_present")
        ),
        "palatial_physready_live_provider_calls_performed": (
            palatial_physready_surface.get("live_provider_calls_performed")
        ),
        "robot_eval_dataset_status": robot_eval_dataset.get("status"),
        "robot_eval_dataset_statuses": robot_eval_dataset.get("dataset_statuses"),
        "robot_eval_recorded_trace_eval_status": robot_eval_dataset.get(
            "recorded_trace_eval_status"
        ),
        "robot_eval_prediction_vs_actual_status": robot_eval_dataset.get(
            "prediction_vs_actual_status"
        ),
        "robot_eval_rights_packet_status": robot_eval_dataset.get("rights_packet_status"),
        "recapture_diff_status": recapture_diff.get("status"),
        "cosmos_zero_shot_benchmark_status": cosmos_zero_shot_benchmark.get("status"),
        "cosmos_training_export_status": cosmos_training_export.get("status"),
        "cosmos_lora_training_status": cosmos_training_run.get("status"),
        "simulation_automation_status": simulation_automation_surface.get("status"),
        "simulation_automation_simulator_execution_proven": simulation_automation_surface.get(
            "simulator_execution_proven"
        ),
        "simulation_automation_rank_fidelity_result_proven": simulation_automation_surface.get(
            "rank_fidelity_result_proven"
        ),
        "site_eval_director_status": site_eval_director_surface.get("status"),
        "site_eval_director_simulator_execution_proven": site_eval_director_surface.get(
            "simulator_execution_proven"
        ),
        "site_eval_director_rank_fidelity_result_proven": site_eval_director_surface.get(
            "rank_fidelity_result_proven"
        ),
        "robot_eval_job_status": robot_eval_job_surface.get("status"),
        "robot_eval_job_count": robot_eval_job_surface.get("job_count"),
        "robot_eval_job_simulator_execution_proven": robot_eval_job_surface.get(
            "simulator_execution_proven"
        ),
        "robot_eval_job_rank_fidelity_result_proven": robot_eval_job_surface.get(
            "rank_fidelity_result_proven"
        ),
        "export_bundle_status": launchable_export_bundle.get("status"),
        "site_world_status": site_world_health.get("status"),
        "geometry_conditioning_status": geometry_conditioning_status,
        "geometry_truth": geometry_truth,
        "world_model_classification": validation_summary["world_model_classification"],
        "validation_gates": validation_summary["validation_gates"],
        "canonical_package_version": canonical_package_version,
        "capture_orientation": site_world_spec.get("capture_orientation"),
        "native_world_model_status": site_world_spec.get("native_world_model_status"),
        "native_world_model_primary": site_world_spec.get("native_world_model_primary"),
        "provider_fallback_preview_status": site_world_spec.get("provider_fallback_preview_status"),
        "provider_fallback_only": site_world_spec.get("provider_fallback_only"),
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
    if not geometry_truth["launchable"] and geometry_truth["geometry_summary_path"]:
        for item in geometry_truth["blockers"]:
            _append_degradation(f"geometry_conditioning:{item}")
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
    native_world_model_status = str(site_world_spec.get("native_world_model_status") or "not_ready")
    native_world_model_path = str(site_world_spec.get("native_world_model_path") or "")
    provider_fallback_only = bool(site_world_spec.get("provider_fallback_only"))
    canonical_package_status = (
        "registration_blocked"
        if not bool(site_world_health.get("launchable"))
        else "native_authoritative_ready"
        if native_world_model_status == "primary_ready" and native_world_model_path == "authoritative_native_render"
        else "geometry_conditioned_native_ready"
        if native_world_model_status == "primary_ready" and native_world_model_path == "geometry_conditioned_native_path"
        else "native_primary_ready"
        if native_world_model_status == "primary_ready"
        else "provider_fallback_only"
        if provider_fallback_only
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
        "native_world_model_status": site_world_spec.get("native_world_model_status"),
        "native_world_model_primary": site_world_spec.get("native_world_model_primary"),
        "provider_fallback_preview_status": site_world_spec.get("provider_fallback_preview_status"),
        "provider_fallback_only": site_world_spec.get("provider_fallback_only"),
        "marble_sim_asset_lane_status": marble_sim_assets.get("status"),
        "simready_asset_lane_status": simready_assets.get("status"),
        "artifact_families": site_world_spec.get("artifact_families"),
        "geometry_conditioning_status": geometry_conditioning_status,
        "geometry_truth": geometry_truth,
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
            **(
                {"cosmos_zero_shot_benchmark": _relative_to(eval_dir, cosmos_zero_shot_benchmark_path)}
                if cosmos_zero_shot_benchmark_path.is_file()
                else {}
            ),
            **(
                {"cosmos_training_export": _relative_to(eval_dir, cosmos_training_export_path)}
                if cosmos_training_export_path.is_file()
                else {}
            ),
            **(
                {"cosmos_lora_training": _relative_to(eval_dir, cosmos_training_run_path)}
                if cosmos_training_run_path.is_file()
                else {}
            ),
            "launchable_export_bundle": _relative_to(eval_dir, launchable_export_bundle_path),
            "object_geometry_manifest": _relative_to(eval_dir, object_geometry_target_path),
            "evaluation_prep_summary": _relative_to(eval_dir, summary_path),
            "review_queue": _relative_to(eval_dir, review_queue_path),
            **(
                {"marble_simready_bridge": _relative_to(eval_dir, marble_simready_bridge_path)}
                if marble_simready_bridge_path.is_file()
                else {}
            ),
            **(
                {"marble_asset_validation": _relative_to(eval_dir, marble_asset_validation_path)}
                if marble_asset_validation_path.is_file()
                else {}
            ),
            **(
                {
                    "robot_eval_dataset_manifest": _relative_to(
                        eval_dir,
                        robot_eval_dataset_manifest_path,
                    )
                }
                if robot_eval_dataset_manifest_path.is_file()
                else {}
            ),
            **(
                {
                    "robot_eval_legacy_manifest": _relative_to(
                        eval_dir,
                        robot_eval_legacy_manifest_path,
                    )
                }
                if robot_eval_legacy_manifest_path.is_file()
                else {}
            ),
            **(
                {"robot_eval_site_card": _relative_to(eval_dir, robot_site_card_path)}
                if robot_site_card_path.is_file()
                else {}
            ),
            **(
                {"robot_eval_task_cards": _relative_to(eval_dir, robot_task_cards_path)}
                if robot_task_cards_path.is_file()
                else {}
            ),
            **(
                {"robot_eval_scenario_cards": _relative_to(eval_dir, robot_scenario_cards_path)}
                if robot_scenario_cards_path.is_file()
                else {}
            ),
            **(
                {"robot_eval_cards": _relative_to(eval_dir, robot_eval_cards_path)}
                if robot_eval_cards_path.is_file()
                else {}
            ),
            **(
                {
                    "robot_eval_annotation_backlog": _relative_to(
                        eval_dir,
                        robot_annotation_backlog_path,
                    )
                }
                if robot_annotation_backlog_path.is_file()
                else {}
            ),
            **(
                {
                    "robot_eval_proof_boundaries": _relative_to(
                        eval_dir,
                        robot_proof_boundaries_path,
                    )
                }
                if robot_proof_boundaries_path.is_file()
                else {}
            ),
            **(
                {"robot_task_library": _relative_to(eval_dir, robot_task_library_path)}
                if robot_task_library_path.is_file()
                else {}
            ),
            **(
                {"robot_task_ontology_v1": _relative_to(eval_dir, robot_task_ontology_path)}
                if robot_task_ontology_path.is_file()
                else {}
            ),
            **(
                {"robot_scenario_library": _relative_to(eval_dir, robot_scenario_library_path)}
                if robot_scenario_library_path.is_file()
                else {}
            ),
            **(
                {
                    "robot_scenario_family_library": _relative_to(
                        eval_dir,
                        robot_scenario_family_library_path,
                    )
                }
                if robot_scenario_family_library_path.is_file()
                else {}
            ),
            **(
                {
                    "robot_pov_evidence_requirements": _relative_to(
                        eval_dir,
                        robot_pov_requirements_path,
                    )
                }
                if robot_pov_requirements_path.is_file()
                else {}
            ),
            **(
                {
                    "human_demo_evidence_requirements": _relative_to(
                        eval_dir,
                        human_demo_requirements_path,
                    )
                }
                if human_demo_requirements_path.is_file()
                else {}
            ),
            **(
                {
                    "robot_eval_inputs_evidence_contract": _relative_to(
                        eval_dir,
                        robot_eval_inputs_evidence_contract_path,
                    )
                }
                if robot_eval_inputs_evidence_contract_path.is_file()
                else {}
            ),
            **(
                {
                    "robot_team_test_submission_modalities": _relative_to(
                        eval_dir,
                        robot_team_test_submission_modalities_path,
                    )
                }
                if robot_team_test_submission_modalities_path.is_file()
                else {}
            ),
            **(
                {"robot_failure_taxonomy": _relative_to(eval_dir, robot_failure_taxonomy_path)}
                if robot_failure_taxonomy_path.is_file()
                else {}
            ),
            **(
                {
                    "prediction_outcome_ledger": _relative_to(
                        eval_dir,
                        prediction_outcome_ledger_path,
                    )
                }
                if prediction_outcome_ledger_path.is_file()
                else {}
            ),
            **(
                {
                    "prediction_vs_actual_summary": _relative_to(
                        eval_dir,
                        prediction_vs_actual_summary_path,
                    )
                }
                if prediction_vs_actual_summary_path.is_file()
                else {}
            ),
            **(
                {"robot_scoring_methodology": _relative_to(eval_dir, robot_scoring_methodology_path)}
                if robot_scoring_methodology_path.is_file()
                else {}
            ),
            **(
                {"robot_eval_task_thresholds": _relative_to(eval_dir, robot_task_thresholds_path)}
                if robot_task_thresholds_path.is_file()
                else {}
            ),
            **(
                {
                    "robot_eval_publication_readiness": _relative_to(
                        eval_dir,
                        robot_publication_readiness_path,
                    )
                }
                if robot_publication_readiness_path.is_file()
                else {}
            ),
            **(
                {
                    "recorded_trace_eval_report": _relative_to(
                        eval_dir,
                        recorded_trace_eval_report_path,
                    )
                }
                if recorded_trace_eval_report_path.is_file()
                else {}
            ),
            **(
                {"policy_eval_report": _relative_to(eval_dir, policy_eval_report_path)}
                if policy_eval_report_path.is_file()
                else {}
            ),
            **(
                {"robot_rights_packet": _relative_to(eval_dir, robot_rights_packet_path)}
                if robot_rights_packet_path.is_file()
                else {}
            ),
            **(
                {"robot_rights_ledger": _relative_to(eval_dir, robot_rights_ledger_path)}
                if robot_rights_ledger_path.is_file()
                else {}
            ),
            **(
                {
                    "robot_eval_methodology_summary": _relative_to(
                        eval_dir,
                        eval_methodology_summary_path,
                    )
                }
                if eval_methodology_summary_path.is_file()
                else {}
            ),
            **(
                {"authoritative_runtime_render_manifest": _relative_to(eval_dir, pipeline_dir / "presentation_world" / "authoritative_runtime_render_manifest.json")}
                if (pipeline_dir / "presentation_world" / "authoritative_runtime_render_manifest.json").is_file()
                else {}
            ),
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
            **dict(palatial_physready_surface.get("artifacts") or {}),
            **dict(simulation_automation_surface.get("artifacts") or {}),
            **dict(site_eval_director_surface.get("artifacts") or {}),
            **dict(robot_eval_job_surface.get("artifacts") or {}),
        },
    }
    descriptor_payload = _read_optional_json_any(context.descriptor_path)
    descriptor_metadata = (
        descriptor_payload.get("metadata")
        if isinstance(descriptor_payload, Mapping) and isinstance(descriptor_payload.get("metadata"), Mapping)
        else {}
    )
    site_identity = (
        descriptor_metadata.get("site_identity")
        if isinstance(descriptor_metadata.get("site_identity"), Mapping)
        else {}
    )
    adjacent_systems = _string_list(descriptor_metadata.get("adjacent_systems"))
    rights_provenance_review = optional_read_json(pipeline_dir / "rights_provenance_review.json") or {}
    preview_manifest = optional_read_json(pipeline_dir / "preview_manifest.json") or {}
    provider_run_manifest = optional_read_json(pipeline_dir / "provider_run_manifest.json") or {}
    worldlabs_launch_url = str(
        provider_run_manifest.get("worldlabs_launch_url")
        or provider_run_manifest.get("preview_launch_url")
        or provider_run_manifest.get("launch_url")
        or preview_manifest.get("worldlabs_launch_url")
        or preview_manifest.get("preview_launch_url")
        or preview_manifest.get("launch_url")
        or ""
    ).strip() or None
    runtime_demo_manifest = _read_json_object(runtime_demo_manifest_path)
    demo_readiness = _presentation_demo_readiness(runtime_demo_manifest)
    shared_artifact_uris = {
        "qualified_opportunity_handoff_uri": _gs_uri(context, "evaluation_prep/qualified_opportunity_handoff.json"),
        "evaluation_prep_manifest_uri": _gs_uri(context, "evaluation_prep/evaluation_prep_manifest.json"),
        "site_world_spec_uri": _gs_uri(context, "evaluation_prep/site_world_spec.json"),
        "site_world_registration_uri": _gs_uri(context, "evaluation_prep/site_world_registration.json"),
        "site_world_health_uri": _gs_uri(context, "evaluation_prep/site_world_health.json"),
        "hosted_session_runtime_manifest_uri": _gs_uri(context, "evaluation_prep/hosted_session_runtime_manifest.json"),
        "launchable_export_bundle_uri": _gs_uri(context, "evaluation_prep/launchable_export_bundle.json"),
        "runtime_demo_manifest_uri": _gs_uri(context, "presentation_world/runtime_demo_manifest.json"),
        "authoritative_runtime_render_manifest_uri": _gs_uri(
            context, "presentation_world/authoritative_runtime_render_manifest.json"
        ),
        "preview_manifest_uri": _gs_uri(context, "preview_manifest.json")
        if (pipeline_dir / "preview_manifest.json").is_file()
        else None,
        "worldlabs_launch_url": worldlabs_launch_url,
        "marble_simready_bridge_uri": _gs_uri(context, "marble_sim_assets/marble_simready_bridge.json")
        if marble_simready_bridge_path.is_file()
        else None,
        "marble_asset_validation_uri": _gs_uri(context, "marble_sim_assets/marble_asset_validation.json")
        if marble_asset_validation_path.is_file()
        else None,
        "robot_eval_dataset_manifest_uri": _gs_uri(
            context,
            "robot_eval_dataset/robot_eval_dataset_manifest.json",
        )
        if robot_eval_dataset_manifest_path.is_file()
        else None,
        "robot_eval_legacy_manifest_uri": _gs_uri(
            context,
            "robot_eval_dataset/real_site_robot_eval_dataset_manifest.json",
        )
        if robot_eval_legacy_manifest_path.is_file()
        else None,
        "robot_eval_site_card_uri": _gs_uri(context, "robot_eval_dataset/site_card.json")
        if robot_site_card_path.is_file()
        else None,
        "robot_eval_task_cards_uri": _gs_uri(context, "robot_eval_dataset/task_cards.json")
        if robot_task_cards_path.is_file()
        else None,
        "robot_eval_scenario_cards_uri": _gs_uri(
            context,
            "robot_eval_dataset/scenario_cards.json",
        )
        if robot_scenario_cards_path.is_file()
        else None,
        "robot_eval_cards_uri": _gs_uri(context, "robot_eval_dataset/eval_cards.json")
        if robot_eval_cards_path.is_file()
        else None,
        "robot_eval_annotation_backlog_uri": _gs_uri(
            context,
            "robot_eval_dataset/annotation_backlog.json",
        )
        if robot_annotation_backlog_path.is_file()
        else None,
        "robot_eval_proof_boundaries_uri": _gs_uri(
            context,
            "robot_eval_dataset/proof_boundaries.json",
        )
        if robot_proof_boundaries_path.is_file()
        else None,
        "robot_task_library_uri": _gs_uri(context, "robot_eval_dataset/robot_task_library.json")
        if robot_task_library_path.is_file()
        else None,
        "robot_task_ontology_v1_uri": _gs_uri(
            context,
            "robot_eval_dataset/task_ontology_v1.json",
        )
        if robot_task_ontology_path.is_file()
        else None,
        "robot_scenario_library_uri": _gs_uri(
            context,
            "robot_eval_dataset/scenario_library.json",
        )
        if robot_scenario_library_path.is_file()
        else None,
        "robot_scenario_family_library_uri": _gs_uri(
            context,
            "robot_eval_dataset/scenario_family_library.json",
        )
        if robot_scenario_family_library_path.is_file()
        else None,
        "robot_pov_evidence_requirements_uri": _gs_uri(
            context,
            "robot_eval_dataset/robot_pov_evidence_requirements.json",
        )
        if robot_pov_requirements_path.is_file()
        else None,
        "human_demo_evidence_requirements_uri": _gs_uri(
            context,
            "robot_eval_dataset/human_demo_evidence_requirements.json",
        )
        if human_demo_requirements_path.is_file()
        else None,
        "robot_eval_inputs_evidence_contract_uri": _gs_uri(
            context,
            "robot_eval_dataset/robot_eval_inputs_evidence_contract.json",
        )
        if robot_eval_inputs_evidence_contract_path.is_file()
        else None,
        "robot_team_test_submission_modalities_uri": _gs_uri(
            context,
            "robot_eval_dataset/robot_team_test_submission_modalities.json",
        )
        if robot_team_test_submission_modalities_path.is_file()
        else None,
        "robot_failure_taxonomy_uri": _gs_uri(
            context,
            "robot_eval_dataset/failure_taxonomy.json",
        )
        if robot_failure_taxonomy_path.is_file()
        else None,
        "prediction_outcome_ledger_uri": _gs_uri(
            context,
            "robot_eval_dataset/prediction_outcome_ledger.json",
        )
        if prediction_outcome_ledger_path.is_file()
        else None,
        "prediction_vs_actual_summary_uri": _gs_uri(
            context,
            "robot_eval_dataset/prediction_vs_actual_summary.json",
        )
        if prediction_vs_actual_summary_path.is_file()
        else None,
        "robot_scoring_methodology_uri": _gs_uri(
            context,
            "robot_eval_dataset/scoring_methodology.json",
        )
        if robot_scoring_methodology_path.is_file()
        else None,
        "robot_eval_task_thresholds_uri": _gs_uri(
            context,
            "robot_eval_dataset/task_thresholds.json",
        )
        if robot_task_thresholds_path.is_file()
        else None,
        "robot_eval_publication_readiness_uri": _gs_uri(
            context,
            "robot_eval_dataset/publication_readiness.json",
        )
        if robot_publication_readiness_path.is_file()
        else None,
        "recorded_trace_eval_report_uri": _gs_uri(
            context,
            "robot_eval_dataset/recorded_trace_eval_report.json",
        )
        if recorded_trace_eval_report_path.is_file()
        else None,
        "policy_eval_report_uri": _gs_uri(
            context,
            "robot_eval_dataset/policy_eval_report.json",
        )
        if policy_eval_report_path.is_file()
        else None,
        "robot_rights_packet_uri": _gs_uri(
            context,
            "robot_eval_dataset/rights_packet.json",
        )
        if robot_rights_packet_path.is_file()
        else None,
        "robot_rights_ledger_uri": _gs_uri(
            context,
            "robot_eval_dataset/rights_ledger.json",
        )
        if robot_rights_ledger_path.is_file()
        else None,
        "robot_eval_methodology_summary_uri": _gs_uri(
            context,
            "robot_eval_dataset/eval_methodology_summary.md",
        )
        if eval_methodology_summary_path.is_file()
        else None,
        **dict(palatial_physready_surface.get("artifact_uris") or {}),
        **dict(simulation_automation_surface.get("artifact_uris") or {}),
        **dict(site_eval_director_surface.get("artifact_uris") or {}),
        **dict(robot_eval_job_surface.get("artifact_uris") or {}),
    }
    site_package_manifest = build_site_package_manifest(
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        site_submission_id=str(normalized_handoff.get("site_submission_id") or ""),
        opportunity_id=str(normalized_handoff.get("opportunity_id") or ""),
        evaluation_prep_manifest=manifest,
        site_world_spec=site_world_spec,
        site_world_registration=site_world_registration,
        site_world_health=site_world_health,
        launchable_export_bundle=launchable_export_bundle,
        site_identity=site_identity,
        adjacent_systems=adjacent_systems,
        rights_review=rights_provenance_review,
        artifact_uris=shared_artifact_uris,
    )
    hosted_review_readiness = build_hosted_review_readiness(
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        site_submission_id=str(normalized_handoff.get("site_submission_id") or ""),
        opportunity_id=str(normalized_handoff.get("opportunity_id") or ""),
        site_identity=site_identity,
        adjacent_systems=adjacent_systems,
        preview_manifest_uri=(
            shared_artifact_uris.get("preview_manifest_uri")
            if isinstance(shared_artifact_uris.get("preview_manifest_uri"), str)
            else None
        ),
        worldlabs_launch_url=worldlabs_launch_url,
        runtime_demo_manifest_uri=_gs_uri(context, "presentation_world/runtime_demo_manifest.json"),
        demo_readiness_state=str(demo_readiness.get("readiness_state") or "blocked"),
        demo_blockers=_string_list(demo_readiness.get("blockers")),
        site_world_health=site_world_health,
        launchable_export_bundle=launchable_export_bundle,
        # Fail closed on rights/privacy so hosted-review readiness never projects
        # "ready" for an unverified-consent capture (beta-launch audit PIPE-02).
        rights_review=rights_provenance_review,
        artifact_uris=shared_artifact_uris,
    )
    proof_pack_manifest = build_proof_pack_manifest(
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        site_submission_id=str(normalized_handoff.get("site_submission_id") or ""),
        opportunity_id=str(normalized_handoff.get("opportunity_id") or ""),
        site_package_manifest=site_package_manifest,
        rights_review=rights_provenance_review,
        hosted_review_readiness=hosted_review_readiness,
        artifact_uris={
            **shared_artifact_uris,
            "site_package_manifest_uri": _gs_uri(context, "evaluation_prep/site_package_manifest.json"),
            "rights_provenance_review_uri": _gs_uri(context, "rights_provenance_review.json"),
            "hosted_review_readiness_uri": _gs_uri(context, "evaluation_prep/hosted_review_readiness.json"),
        },
    )
    proof_path_status = build_proof_path_status(
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        site_submission_id=str(normalized_handoff.get("site_submission_id") or ""),
        opportunity_id=str(normalized_handoff.get("opportunity_id") or ""),
        rights_review=rights_provenance_review,
        site_package_manifest=site_package_manifest,
        proof_pack_manifest=proof_pack_manifest,
        hosted_review_readiness=hosted_review_readiness,
    )
    _copy_json(eval_dir / "site_package_manifest.json", site_package_manifest)
    _copy_json(eval_dir / "hosted_review_readiness.json", hosted_review_readiness)
    _copy_json(eval_dir / "proof_pack_manifest.json", proof_pack_manifest)
    _copy_json(eval_dir / "proof_path_status.json", proof_path_status)
    manifest["artifacts"].update(
        {
            "site_package_manifest": _relative_to(eval_dir, eval_dir / "site_package_manifest.json"),
            "hosted_review_readiness": _relative_to(eval_dir, eval_dir / "hosted_review_readiness.json"),
            "proof_pack_manifest": _relative_to(eval_dir, eval_dir / "proof_pack_manifest.json"),
            "proof_path_status": _relative_to(eval_dir, eval_dir / "proof_path_status.json"),
        }
    )
    manifest_path = eval_dir / "evaluation_prep_manifest.json"
    _copy_json(manifest_path, manifest)
    webapp_sync_result = sync_webapp_evaluation_prep(capture_root=context.capture_root)
    alpha_summary = write_alpha_readiness_summary(capture_root=context.capture_root)
    return {
        "schema_version": "v1",
        "capture_root": str(context.capture_root),
        "manifest_path": str(manifest_path),
        "status": legacy_status,
        "canonical_package_status": canonical_package_status,
        "site_package_manifest": site_package_manifest,
        "hosted_review_readiness": hosted_review_readiness,
        "proof_pack_manifest": proof_pack_manifest,
        "proof_path_status": proof_path_status,
        "simready_assets": simready_assets,
        "marble_sim_assets": marble_sim_assets,
        "palatial_physready": palatial_physready_surface,
        "robot_eval_dataset": robot_eval_dataset,
        "webapp_sync_result": webapp_sync_result,
        "alpha_readiness_summary": alpha_summary,
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
