"""Build explicit auxiliary observation manifests for WAM/OSCAR inputs.

The manifest records optional robot-realistic side channels next to the RGB
policy frame. It is an input/conditioning contract, not proof that a generated
rollout is physically valid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json


SCHEMA_VERSION = "wam_auxiliary_observation_manifest.v1"
CLAIM_BOUNDARY_SCHEMA_VERSION = "wam_auxiliary_observation_claim_boundary.v1"

SYNTHETIC_SUPPORT_SOURCE_KINDS = {
    "synthetic_gpt_image_2_seed",
    "image_model_enhanced_3d_render_seed",
    "synthetic_fallback",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except TypeError:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_jsonable(item) for item in value]
        return str(value)


def _local_path_exists(path_text: str) -> bool | None:
    if not path_text:
        return None
    if "://" in path_text:
        return None
    return Path(path_text).expanduser().is_file()


def _first_path(*values: Any) -> str | None:
    for value in values:
        text = _string(value)
        if text:
            return text
    return None


def _nested_first_path(*containers_and_keys: tuple[Mapping[str, Any], Sequence[str]]) -> str | None:
    for container, keys in containers_and_keys:
        for key in keys:
            text = _string(container.get(key))
            if text:
                return text
    return None


def _truth_value(
    *,
    truth_overrides: Mapping[str, Any],
    observation_boundary: Mapping[str, Any],
    key: str,
    default: bool = False,
) -> bool:
    if key in truth_overrides:
        return bool(truth_overrides.get(key))
    if key in observation_boundary:
        return bool(observation_boundary.get(key))
    return default


def _extract_camera_intrinsics(
    *,
    explicit: Mapping[str, Any] | None,
    camera: Mapping[str, Any],
    visual: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any] | None:
    for value in (
        explicit,
        camera.get("intrinsics"),
        camera.get("camera_intrinsics"),
        visual.get("camera_intrinsics"),
        observation.get("camera_intrinsics"),
    ):
        mapping = _mapping(value)
        if mapping:
            return mapping
    return None


def _extract_camera_extrinsics(
    *,
    explicit: Mapping[str, Any] | None,
    camera: Mapping[str, Any],
    visual: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any] | None:
    for value in (
        explicit,
        camera.get("extrinsics"),
        camera.get("camera_extrinsics"),
        visual.get("camera_extrinsics"),
        observation.get("camera_extrinsics"),
    ):
        mapping = _mapping(value)
        if mapping:
            return mapping
    return None


def _extract_head_pose(
    *,
    explicit: Mapping[str, Any] | None,
    camera: Mapping[str, Any],
    visual: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any] | None:
    for value in (
        explicit,
        camera.get("head_pose"),
        visual.get("head_pose"),
        observation.get("head_pose"),
        observation.get("base_pose"),
    ):
        mapping = _mapping(value)
        if mapping:
            return mapping
    return None


def _extract_proprioception(
    *,
    explicit: Mapping[str, Any] | None,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    explicit_mapping = _mapping(explicit)
    if explicit_mapping:
        return {
            "available": True,
            "source": _string(explicit_mapping.get("source")) or "explicit_auxiliary_observation",
            "values": _jsonable(explicit_mapping),
        }
    proprioception = _mapping(observation.get("proprioception"))
    if proprioception:
        return {
            "available": True,
            "source": _string(proprioception.get("source")) or "policy_observation_proprioception",
            "values": _jsonable(proprioception),
        }
    unitree_state = _mapping(observation.get("unitree_g1_sonic_state"))
    if unitree_state:
        return {
            "available": True,
            "source": _string(observation.get("unitree_g1_sonic_state_source"))
            or "policy_observation_unitree_g1_sonic_state",
            "values": {
                "unitree_g1_sonic_state": _jsonable(unitree_state),
                "metadata": _jsonable(_mapping(observation.get("unitree_g1_sonic_state_metadata"))),
            },
        }
    return {"available": False, "source": None, "values": {}}


def _action_values(action: Mapping[str, Any]) -> list[float]:
    values: Any = action.get("action_chunk")
    if not _sequence(values):
        values = action.get("values")
    if not _sequence(values):
        values = action.get("sonic_latent_action")
        if _sequence(values) and values and _sequence(values[0]):
            values = values[0]
    result: list[float] = []
    for value in _sequence(values):
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def _extract_action_conditioning(
    *,
    explicit: Mapping[str, Any] | None,
    source_policy_action: Mapping[str, Any] | None,
    projected_skeleton_trace_path: str | Path | None,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    explicit_mapping = _mapping(explicit)
    action = _mapping(source_policy_action)
    values = _action_values(action)
    projected_path = _first_path(
        projected_skeleton_trace_path,
        explicit_mapping.get("projected_skeleton_trace_path"),
        explicit_mapping.get("projected_hand_keypoint_trace_path"),
        observation.get("g1_projected_skeleton_trace_jsonl"),
        _mapping(observation.get("visual_observation")).get("g1_projected_skeleton_trace_jsonl"),
    )
    return {
        "available": bool(values or explicit_mapping or projected_path),
        "action_type": action.get("action_type") or explicit_mapping.get("action_type"),
        "action_chunk_value_count": len(values),
        "action_chunk_l1_mean": round(sum(abs(item) for item in values) / max(len(values), 1), 6),
        "projected_skeleton_trace_path": projected_path,
        "projected_hand_keypoint_trace_path": _first_path(
            explicit_mapping.get("projected_hand_keypoint_trace_path"), projected_path
        ),
        "projected_trace_path_exists": _local_path_exists(projected_path or ""),
        "values_source": "source_policy_action" if values else None,
        "raw_action_values_recorded": bool(values),
    }


def _extract_target_bbox(
    *,
    explicit: Any,
    visual: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> Any:
    candidates = [
        explicit,
        visual.get("target_bbox"),
        observation.get("target_bbox"),
        _mapping(visual.get("target")).get("bbox"),
        _mapping(observation.get("target")).get("bbox"),
        _mapping(visual.get("target_object")).get("bbox"),
    ]
    for candidate in candidates:
        if _mapping(candidate) or _sequence(candidate):
            return _jsonable(candidate)
    return None


def _extract_target_keypoints(
    *,
    explicit: Any,
    visual: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> Any:
    candidates = [
        explicit,
        visual.get("target_keypoints"),
        observation.get("target_keypoints"),
        _mapping(visual.get("target")).get("keypoints"),
        _mapping(observation.get("target")).get("keypoints"),
    ]
    for candidate in candidates:
        if _mapping(candidate) or _sequence(candidate):
            return _jsonable(candidate)
    return []


def _extract_affordance_points(
    *,
    explicit: Any,
    visual: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> Any:
    candidates = [
        explicit,
        visual.get("affordance_points"),
        observation.get("affordance_points"),
        visual.get("affordance_point"),
        observation.get("affordance_point"),
        _mapping(visual.get("target")).get("affordance_points"),
        _mapping(observation.get("target")).get("affordance_points"),
    ]
    for candidate in candidates:
        if _mapping(candidate) or _sequence(candidate):
            return _jsonable(candidate)
    return []


def build_wam_auxiliary_observation_manifest(
    *,
    output_dir: str | Path,
    source_image_path: str | Path,
    policy_observation: Mapping[str, Any] | None = None,
    source_policy_action: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
    source_kind: str | None = None,
    camera_id: str | None = None,
    robot_profile_id: str | None = None,
    task_id: str | None = None,
    target_object_id: str | None = None,
    projected_skeleton_trace_path: str | Path | None = None,
    depth_map_path: str | Path | None = None,
    depth_confidence_path: str | Path | None = None,
    camera_intrinsics: Mapping[str, Any] | None = None,
    camera_extrinsics: Mapping[str, Any] | None = None,
    head_pose: Mapping[str, Any] | None = None,
    target_segmentation_mask_path: str | Path | None = None,
    robot_mask_path: str | Path | None = None,
    target_bbox: Any = None,
    target_keypoints: Any = None,
    affordance_points: Any = None,
    timestamp_ns: int | None = None,
    sync_metadata: Mapping[str, Any] | None = None,
    proprioception: Mapping[str, Any] | None = None,
    action_conditioning: Mapping[str, Any] | None = None,
    truth_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write and return a WAM auxiliary observation manifest."""

    generated = generated_at or utc_now_iso()
    output = Path(output_dir).expanduser().resolve()
    ensure_dir(output)
    observation = _mapping(policy_observation)
    visual = _mapping(observation.get("visual_observation"))
    camera = _mapping(observation.get("camera")) or _mapping(visual.get("camera"))
    observation_boundary = {
        **_mapping(visual.get("claim_boundary")),
        **_mapping(observation.get("claim_boundary")),
    }
    overrides = _mapping(truth_overrides)
    resolved_source_kind = (
        _string(source_kind)
        or _string(observation.get("source_kind"))
        or _string(visual.get("source_kind"))
        or "unspecified"
    )
    synthetic_support = resolved_source_kind in SYNTHETIC_SUPPORT_SOURCE_KINDS
    capture_truth = (
        False
        if synthetic_support
        else _truth_value(
            truth_overrides=overrides,
            observation_boundary=observation_boundary,
            key="capture_truth",
        )
    )
    geometry_truth = (
        False
        if synthetic_support
        else _truth_value(
            truth_overrides=overrides,
            observation_boundary=observation_boundary,
            key="geometry_truth",
        )
    )
    depth_truth = (
        False
        if synthetic_support
        else _truth_value(
            truth_overrides=overrides,
            observation_boundary=observation_boundary,
            key="depth_truth",
        )
    )
    segmentation_truth = (
        False
        if synthetic_support
        else _truth_value(
            truth_overrides=overrides,
            observation_boundary=observation_boundary,
            key="segmentation_truth",
        )
    )
    camera_pose_truth = (
        False
        if synthetic_support
        else _truth_value(
            truth_overrides=overrides,
            observation_boundary=observation_boundary,
            key="camera_pose_truth",
            default=geometry_truth,
        )
    )
    proprioception_truth = (
        False
        if synthetic_support
        else _truth_value(
            truth_overrides=overrides,
            observation_boundary=observation_boundary,
            key="proprioception_truth",
        )
    )
    resolved_depth_map = _first_path(
        depth_map_path,
        visual.get("depth_map_path"),
        visual.get("depth_path"),
        visual.get("depth_uri"),
        observation.get("depth_map_path"),
        observation.get("depth_path"),
    )
    resolved_depth_confidence = _first_path(
        depth_confidence_path,
        visual.get("depth_confidence_path"),
        visual.get("depth_confidence_uri"),
        observation.get("depth_confidence_path"),
    )
    resolved_target_mask = _first_path(
        target_segmentation_mask_path,
        visual.get("target_segmentation_mask_path"),
        observation.get("target_segmentation_mask_path"),
        _mapping(visual.get("target")).get("segmentation_mask_path"),
    )
    resolved_robot_mask = _first_path(
        robot_mask_path,
        visual.get("robot_mask_path"),
        observation.get("robot_mask_path"),
    )
    intrinsics = _extract_camera_intrinsics(
        explicit=camera_intrinsics,
        camera=camera,
        visual=visual,
        observation=observation,
    )
    extrinsics = _extract_camera_extrinsics(
        explicit=camera_extrinsics,
        camera=camera,
        visual=visual,
        observation=observation,
    )
    resolved_head_pose = _extract_head_pose(
        explicit=head_pose,
        camera=camera,
        visual=visual,
        observation=observation,
    )
    proprioception_block = _extract_proprioception(
        explicit=proprioception,
        observation=observation,
    )
    target_bbox_value = _extract_target_bbox(
        explicit=target_bbox,
        visual=visual,
        observation=observation,
    )
    target_keypoints_value = _extract_target_keypoints(
        explicit=target_keypoints,
        visual=visual,
        observation=observation,
    )
    affordance_points_value = _extract_affordance_points(
        explicit=affordance_points,
        visual=visual,
        observation=observation,
    )
    action_conditioning_block = _extract_action_conditioning(
        explicit=action_conditioning,
        source_policy_action=source_policy_action,
        projected_skeleton_trace_path=projected_skeleton_trace_path,
        observation=observation,
    )
    resolved_timestamp = (
        int(timestamp_ns)
        if timestamp_ns is not None
        else observation.get("timestamp_ns") or visual.get("timestamp_ns")
    )
    sync_block = {
        "timestamp_ns": resolved_timestamp,
        "modalities_aligned": bool(resolved_timestamp or _mapping(sync_metadata)),
        "metadata": _jsonable(
            _mapping(sync_metadata) or _mapping(observation.get("sync_metadata"))
        ),
    }
    camera_block = {
        "camera_id": _string(camera_id)
        or _string(observation.get("camera_id"))
        or _string(visual.get("camera_id"))
        or "head_pov",
        "robot_profile_id": _string(robot_profile_id)
        or _string(observation.get("robot_profile_id")),
        "intrinsics": _jsonable(intrinsics) if intrinsics else None,
        "extrinsics": _jsonable(extrinsics) if extrinsics else None,
        "head_pose": _jsonable(resolved_head_pose) if resolved_head_pose else None,
        "intrinsics_available": bool(intrinsics),
        "extrinsics_or_head_pose_available": bool(extrinsics or resolved_head_pose),
        "camera_pose_truth": camera_pose_truth,
    }
    depth_block = {
        "available": bool(resolved_depth_map),
        "depth_map_path": resolved_depth_map,
        "depth_confidence_path": resolved_depth_confidence,
        "depth_map_path_exists": _local_path_exists(resolved_depth_map or ""),
        "depth_confidence_path_exists": _local_path_exists(resolved_depth_confidence or ""),
        "depth_truth": depth_truth,
        "estimated_depth": bool(resolved_depth_map and not depth_truth),
        "source_kind": resolved_source_kind,
    }
    segmentation_block = {
        "available": bool(
            resolved_target_mask
            or resolved_robot_mask
            or target_bbox_value
            or target_keypoints_value
            or affordance_points_value
        ),
        "target_segmentation_mask_path": resolved_target_mask,
        "robot_mask_path": resolved_robot_mask,
        "target_segmentation_mask_path_exists": _local_path_exists(resolved_target_mask or ""),
        "robot_mask_path_exists": _local_path_exists(resolved_robot_mask or ""),
        "target_bbox": target_bbox_value,
        "target_keypoints": target_keypoints_value,
        "affordance_points": affordance_points_value,
        "segmentation_truth": segmentation_truth,
        "estimated_segmentation": bool(
            (resolved_target_mask or resolved_robot_mask) and not segmentation_truth
        ),
    }
    modalities_available = {
        "rgb": bool(_string(source_image_path)),
        "depth": bool(depth_block["available"]),
        "depth_confidence": bool(resolved_depth_confidence),
        "camera_intrinsics": bool(camera_block["intrinsics_available"]),
        "camera_extrinsics_or_head_pose": bool(camera_block["extrinsics_or_head_pose_available"]),
        "target_segmentation_mask": bool(resolved_target_mask),
        "robot_mask": bool(resolved_robot_mask),
        "target_bbox": bool(target_bbox_value),
        "target_keypoints": bool(target_keypoints_value),
        "affordance_points": bool(affordance_points_value),
        "proprioception": bool(proprioception_block["available"]),
        "action_conditioning": bool(action_conditioning_block["available"]),
    }
    claim_boundary = {
        "schema_version": CLAIM_BOUNDARY_SCHEMA_VERSION,
        "source_kind": resolved_source_kind,
        "auxiliary_observation_is_wam_conditioning_support": True,
        "auxiliary_observation_is_not_wam_output": True,
        "capture_truth": capture_truth,
        "geometry_truth": geometry_truth,
        "collision_truth": False,
        "depth_truth": depth_truth,
        "segmentation_truth": segmentation_truth,
        "camera_pose_truth": camera_pose_truth,
        "proprioception_truth": proprioception_truth,
        "synthetic_2d_sidecars_are_estimated_support_only": synthetic_support,
        "visual_seed_for_wam_experiment": bool(synthetic_support),
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "non_ranking_operational_claim_proven": False,
        "accepted_anchor_manipulation_success_proven": False,
    }
    manifest_path = output / "wam_auxiliary_observation_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed",
        "manifest_path": str(manifest_path),
        "source_image_path": str(Path(source_image_path).expanduser()),
        "source_image_path_exists": _local_path_exists(str(Path(source_image_path).expanduser())),
        "source_kind": resolved_source_kind,
        "task_id": _string(task_id) or observation.get("task_id"),
        "target_object_id": _string(target_object_id) or observation.get("target_object_id"),
        "camera": camera_block,
        "depth": depth_block,
        "segmentation": segmentation_block,
        "proprioception": {
            **proprioception_block,
            "proprioception_truth": proprioception_truth,
        },
        "sync": sync_block,
        "action_conditioning": action_conditioning_block,
        "modalities_available": modalities_available,
        "oscar_conditioning_support": {
            "raw_aux_modalities_consumed_by_public_oscar_entrypoint": False,
            "overlay_conditioning_recommended": True,
            "overlay_supported_modalities": [
                "target_bbox",
                "target_keypoints",
                "affordance_points",
                "projected_skeleton_trace",
                "policy_action_chunk",
            ],
            "depth_and_mask_sidecars_packaged_as_support_metadata": True,
        },
        "blockers": [],
        "claim_boundary": claim_boundary,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(manifest_path, manifest)
    write_json(output / "wam_auxiliary_observation_claim_boundary.json", claim_boundary)
    return manifest


def summarize_wam_auxiliary_observation_manifest(
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a compact, path-safe summary suitable for embedding."""

    value = _mapping(manifest)
    if not value:
        return {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "modalities_available": {},
        }
    return {
        "schema_version": _string(value.get("schema_version")) or SCHEMA_VERSION,
        "available": True,
        "manifest_path": value.get("manifest_path"),
        "source_kind": value.get("source_kind"),
        "source_image_path": value.get("source_image_path"),
        "task_id": value.get("task_id"),
        "target_object_id": value.get("target_object_id"),
        "camera_id": _mapping(value.get("camera")).get("camera_id"),
        "modalities_available": _mapping(value.get("modalities_available")),
        "oscar_conditioning_support": _mapping(value.get("oscar_conditioning_support")),
        "claim_boundary": _mapping(value.get("claim_boundary")),
        "blockers": list(value.get("blockers") or []),
    }
