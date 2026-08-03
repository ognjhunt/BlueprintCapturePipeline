"""Compose movable object splats with SimReady physics on one pose channel.

The static background comes from an exhaustive Gaussian partition in which the
object primitives are absent.  Exactly one object-local SplatMesh is then driven
by the same simulator body pose as the object's collider.  The renderer receipt
proves scene-graph composition and rendered bytes, not semantic segmentation
completeness, physical accuracy, or robot-task success.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .gaussian_object_partition import (
    validate_gaussian_object_partition,
    verify_gaussian_object_partition_files,
)
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest
from .simready_asset_lane import validate_simready_asset_manifest


DYNAMIC_SPLAT_SCENE_SCHEMA_VERSION = "dynamic_splat_scene.v1"
DYNAMIC_SPLAT_RENDER_REQUEST_SCHEMA_VERSION = "dynamic_splat_render_request.v1"
DYNAMIC_SPLAT_RENDER_MANIFEST_SCHEMA_VERSION = "dynamic_splat_render_manifest.v1"
RENDER_HARNESS_REL = "tools/splat_render/render_splat.mjs"
RENDER_ENTRY_REL = "tools/splat_render/src/render_entry.mjs"


class DynamicSplatSceneError(ValueError):
    def __init__(self, *codes: str) -> None:
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise DynamicSplatSceneError("dynamic_splat_artifact_not_json") from exc
    if not isinstance(result, dict):
        raise DynamicSplatSceneError("dynamic_splat_artifact_not_object")
    return result


def _digest(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_digest({key: item for key, item in value.items() if key != field})


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _valid_transform(value: Any, *, code: str) -> list[list[float]]:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DynamicSplatSceneError(f"{code}_invalid") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise DynamicSplatSceneError(f"{code}_invalid")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0, atol=1e-9):
        raise DynamicSplatSceneError(f"{code}_bottom_row_invalid")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6):
        raise DynamicSplatSceneError(f"{code}_rotation_not_orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0, abs_tol=1e-6):
        raise DynamicSplatSceneError(f"{code}_rotation_not_proper")
    return [[float(component) for component in row] for row in matrix]


def _identity() -> list[list[float]]:
    return np.eye(4, dtype=np.float64).tolist()


def validate_dynamic_splat_scene(value: Mapping[str, Any]) -> dict[str, Any]:
    scene = _clone(value)
    errors: list[str] = []
    if scene.get("schema_version") != DYNAMIC_SPLAT_SCENE_SCHEMA_VERSION:
        errors.append("dynamic_splat_scene_schema_invalid")
    if not str(scene.get("scene_id") or "").strip():
        errors.append("dynamic_splat_scene_id_missing")
    background = scene.get("background")
    if not isinstance(background, Mapping):
        errors.append("dynamic_splat_background_missing")
    elif background.get("static_instance_count") != 1:
        errors.append("dynamic_splat_background_instance_count_not_one")
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects:
        errors.append("dynamic_splat_objects_missing")
        objects = []
    object_ids: list[str] = []
    pose_channels: list[str] = []
    for row in objects:
        if not isinstance(row, Mapping):
            errors.append("dynamic_splat_object_row_invalid")
            continue
        object_id = str(row.get("object_id") or "").strip()
        if not object_id:
            errors.append("dynamic_splat_object_id_missing")
        object_ids.append(object_id)
        appearance = row.get("appearance")
        physics = row.get("physics")
        binding = row.get("pose_binding")
        if not isinstance(appearance, Mapping):
            errors.append(f"dynamic_splat_appearance_missing:{object_id}")
        elif appearance.get("visual_instance_count") != 1:
            errors.append(f"dynamic_splat_visual_instance_count_not_one:{object_id}")
        if not isinstance(physics, Mapping):
            errors.append(f"dynamic_splat_physics_missing:{object_id}")
        elif physics.get("collider_instance_count") != 1:
            errors.append(f"dynamic_splat_collider_instance_count_not_one:{object_id}")
        if not isinstance(binding, Mapping):
            errors.append(f"dynamic_splat_pose_binding_missing:{object_id}")
            continue
        pose_channel = str(binding.get("pose_channel_id") or "").strip()
        pose_channels.append(pose_channel)
        if not pose_channel:
            errors.append(f"dynamic_splat_pose_channel_missing:{object_id}")
        if binding.get("appearance_and_collider_share_body_pose") is not True:
            errors.append(f"dynamic_splat_shared_pose_false:{object_id}")
        try:
            _valid_transform(
                binding.get("T_body_appearance"),
                code=f"dynamic_splat_body_appearance:{object_id}",
            )
        except DynamicSplatSceneError as exc:
            errors.extend(exc.codes)
    if len(object_ids) != len(set(object_ids)):
        errors.append("dynamic_splat_duplicate_object_id")
    if len(pose_channels) != len(set(pose_channels)):
        errors.append("dynamic_splat_duplicate_pose_channel")
    frames = scene.get("frames")
    if not isinstance(frames, list) or not frames:
        errors.append("dynamic_splat_frames_missing")
        frames = []
    frame_ids: list[str] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            errors.append("dynamic_splat_frame_row_invalid")
            continue
        frame_id = str(frame.get("frame_id") or "").strip()
        frame_ids.append(frame_id)
        poses = frame.get("body_poses")
        if not isinstance(poses, Mapping) or set(poses) != set(pose_channels):
            errors.append(f"dynamic_splat_frame_pose_channels_incomplete:{frame_id}")
            continue
        for channel, transform in poses.items():
            try:
                _valid_transform(
                    transform,
                    code=f"dynamic_splat_frame_transform:{frame_id}:{channel}",
                )
            except DynamicSplatSceneError as exc:
                errors.extend(exc.codes)
    if len(frame_ids) != len(set(frame_ids)) or any(not item for item in frame_ids):
        errors.append("dynamic_splat_frame_ids_missing_or_duplicate")
    invariants = scene.get("render_invariants")
    if not isinstance(invariants, Mapping):
        errors.append("dynamic_splat_render_invariants_missing")
    else:
        for key in (
            "background_excludes_object_gaussians",
            "one_visual_instance_per_object",
            "one_collider_instance_per_object",
            "appearance_and_collider_share_pose_channel",
            "object_gaussians_absent_from_static_background",
        ):
            if invariants.get(key) is not True:
                errors.append(f"dynamic_splat_render_invariant_false:{key}")
    if scene.get("semantic_completeness_validated") is not False:
        errors.append("dynamic_splat_semantic_completeness_must_be_false")
    if scene.get("physical_accuracy_established") is not False:
        errors.append("dynamic_splat_physical_accuracy_must_be_false")
    expected = _digest(scene, "dynamic_splat_scene_digest")
    supplied = scene.get("dynamic_splat_scene_digest")
    if supplied is not None and supplied != expected:
        errors.append("dynamic_splat_scene_digest_mismatch")
    if errors:
        raise DynamicSplatSceneError(*errors)
    scene["dynamic_splat_scene_digest"] = expected
    return scene


def build_dynamic_splat_scene(
    partition_value: Mapping[str, Any],
    simready_manifest_value: Mapping[str, Any],
    *,
    frames: Sequence[Mapping[str, Any]],
    T_body_appearance: Sequence[Sequence[float]] | None = None,
    pose_channel_id: str | None = None,
) -> dict[str, Any]:
    """Bind one movable Gaussian object and collider to the same body pose."""

    partition = validate_gaussian_object_partition(partition_value)
    verification = verify_gaussian_object_partition_files(partition)
    if verification["status"] != "passed":
        raise DynamicSplatSceneError(*verification["errors"])
    simready = validate_simready_asset_manifest(simready_manifest_value)
    if simready["object_id"] != partition["object_id"]:
        raise DynamicSplatSceneError("dynamic_splat_simready_object_id_mismatch")
    channel = str(pose_channel_id or f"body_pose:{partition['object_id']}").strip()
    if not channel:
        raise DynamicSplatSceneError("dynamic_splat_pose_channel_missing")
    body_from_appearance = _valid_transform(
        T_body_appearance if T_body_appearance is not None else _identity(),
        code="dynamic_splat_body_appearance",
    )
    normalized_frames: list[dict[str, Any]] = []
    for raw_frame in frames:
        if not isinstance(raw_frame, Mapping):
            raise DynamicSplatSceneError("dynamic_splat_frame_row_invalid")
        frame_id = str(raw_frame.get("frame_id") or "").strip()
        transform = _valid_transform(
            raw_frame.get("T_world_body"),
            code=f"dynamic_splat_frame_transform:{frame_id or 'missing'}",
        )
        normalized_frames.append(
            {
                "frame_id": frame_id,
                "body_poses": {channel: transform},
                "pose_source": str(
                    raw_frame.get("pose_source") or "simulator_body_transform"
                ),
            }
        )
    object_row = {
        "object_id": partition["object_id"],
        "appearance": {
            "path": partition["artifacts"]["object"]["path"],
            "digest": partition["artifacts"]["object"]["digest"],
            "gaussian_count": partition["counts"]["object"],
            "partition_digest": partition["gaussian_object_partition_digest"],
            "visual_instance_count": 1,
        },
        "physics": {
            "asset_id": simready["asset_id"],
            "simready_asset_digest": simready["simready_asset_digest"],
            "collider_count": len(simready.get("colliders") or []),
            "mjcf_content_digest": (
                simready.get("exports", {}).get("mjcf", {}).get("content_digest")
            ),
            "usd_content_digest": (
                simready.get("exports", {}).get("usd", {}).get("content_digest")
            ),
            "collider_instance_count": 1,
            "validated": False,
        },
        "pose_binding": {
            "pose_channel_id": channel,
            "T_body_appearance": body_from_appearance,
            "appearance_and_collider_share_body_pose": True,
        },
    }
    scene = {
        "schema_version": DYNAMIC_SPLAT_SCENE_SCHEMA_VERSION,
        "scene_id": (
            f"dynamic-splat-{partition['object_id']}-"
            f"{partition['gaussian_object_partition_digest'][-12:]}"
        ),
        "background": {
            "path": partition["artifacts"]["background"]["path"],
            "digest": partition["artifacts"]["background"]["digest"],
            "gaussian_count": partition["counts"]["background"],
            "static_instance_count": 1,
        },
        "objects": [object_row],
        "frames": normalized_frames,
        "render_invariants": {
            "background_excludes_object_gaussians": True,
            "one_visual_instance_per_object": True,
            "one_collider_instance_per_object": True,
            "appearance_and_collider_share_pose_channel": True,
            "object_gaussians_absent_from_static_background": True,
        },
        "claim_ceiling": "dynamic_visual_composition_development_evidence",
        "semantic_completeness_validated": False,
        "physical_accuracy_established": False,
        "robot_camera_parity_established": False,
    }
    scene["dynamic_splat_scene_digest"] = _digest(scene, "dynamic_splat_scene_digest")
    return validate_dynamic_splat_scene(scene)


def build_dynamic_splat_render_request(
    scene_value: Mapping[str, Any], *, frame_id: str
) -> dict[str, Any]:
    scene = validate_dynamic_splat_scene(scene_value)
    matches = [frame for frame in scene["frames"] if frame["frame_id"] == frame_id]
    if len(matches) != 1:
        raise DynamicSplatSceneError("dynamic_splat_render_frame_not_exactly_one")
    frame = matches[0]
    object_rows: list[dict[str, Any]] = []
    for row in scene["objects"]:
        binding = row["pose_binding"]
        body_pose = np.asarray(
            frame["body_poses"][binding["pose_channel_id"]], dtype=np.float64
        )
        body_from_appearance = np.asarray(binding["T_body_appearance"], dtype=np.float64)
        world_from_appearance = body_pose @ body_from_appearance
        object_rows.append(
            {
                "object_id": row["object_id"],
                "path": row["appearance"]["path"],
                "digest": row["appearance"]["digest"],
                "gaussian_count": row["appearance"]["gaussian_count"],
                "T_world_object": world_from_appearance.tolist(),
                "pose_channel_id": binding["pose_channel_id"],
            }
        )
    request = {
        "schema_version": DYNAMIC_SPLAT_RENDER_REQUEST_SCHEMA_VERSION,
        "dynamic_splat_scene_digest": scene["dynamic_splat_scene_digest"],
        "frame_id": frame_id,
        "background": dict(scene["background"]),
        "objects": object_rows,
        "expected_object_ids": sorted(row["object_id"] for row in object_rows),
        "require_exactly_one_visual_instance_per_object": True,
    }
    request["dynamic_splat_render_request_digest"] = _digest(
        request, "dynamic_splat_render_request_digest"
    )
    return request


def render_dynamic_splat_frame(
    scene_value: Mapping[str, Any],
    *,
    frame_id: str,
    cameras: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    repo_root: str | Path | None = None,
    node: str = "node",
    graphics_backend: str = "swiftshader",
    width: int = 640,
    height: int = 480,
    timeout_seconds: int = 240,
) -> dict[str, Any]:
    """Render a dynamic frame through the real local Spark.js composition path."""

    scene = validate_dynamic_splat_scene(scene_value)
    request = build_dynamic_splat_render_request(scene, frame_id=frame_id)
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    harness = root / RENDER_HARNESS_REL
    entry = root / RENDER_ENTRY_REL
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    if not harness.is_file() or not entry.is_file():
        blockers.append("dynamic_splat_render_harness_missing")
    if not isinstance(cameras, Sequence) or isinstance(cameras, (str, bytes)) or not cameras:
        blockers.append("dynamic_splat_render_cameras_missing")
    if graphics_backend not in {"swiftshader", "metal"}:
        blockers.append("dynamic_splat_render_graphics_backend_invalid")
    for artifact in [request["background"], *request["objects"]]:
        path = Path(artifact["path"])
        if path.is_symlink() or not path.is_file():
            blockers.append("dynamic_splat_render_asset_missing_or_symlink")
        elif _sha256_file(path) != artifact["digest"]:
            blockers.append("dynamic_splat_render_asset_digest_mismatch")
    base = {
        "schema_version": DYNAMIC_SPLAT_RENDER_MANIFEST_SCHEMA_VERSION,
        "dynamic_splat_scene_digest": scene["dynamic_splat_scene_digest"],
        "dynamic_splat_render_request_digest": request[
            "dynamic_splat_render_request_digest"
        ],
        "frame_id": frame_id,
        "status": "blocked",
        "blockers": sorted(set(blockers)),
        "renderer": {
            "renderer_id": "sparkjs-multi-splat-object-compositor",
            "harness_digest": _sha256_file(harness) if harness.is_file() else None,
            "entry_digest": _sha256_file(entry) if entry.is_file() else None,
            "graphics_backend": graphics_backend,
        },
        "render_result": None,
        "proof_boundary": {
            "exactly_one_declared_visual_instance_rendered": False,
            "object_absent_from_static_background_by_partition": True,
            "semantic_object_completeness_established": False,
            "physical_accuracy_established": False,
            "robot_camera_parity_established": False,
        },
    }
    if blockers:
        base["dynamic_splat_render_manifest_digest"] = _digest(
            base, "dynamic_splat_render_manifest_digest"
        )
        return base
    request_path = destination / "dynamic_splat_render_request.json"
    cameras_path = destination / "cameras.json"
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
    cameras_path.write_text(json.dumps(list(cameras), indent=2, sort_keys=True), encoding="utf-8")
    command = [
        node,
        str(harness),
        "--composition",
        str(request_path),
        "--out",
        str(destination),
        "--cameras",
        str(cameras_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--graphics-backend",
        graphics_backend,
    ]
    try:
        process = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        base["blockers"] = ["dynamic_splat_render_node_unavailable"]
    except subprocess.TimeoutExpired:
        base["blockers"] = ["dynamic_splat_render_timeout"]
    else:
        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError:
            result = None
            base["blockers"] = ["dynamic_splat_render_output_not_json"]
        if isinstance(result, Mapping):
            base["render_result"] = dict(result)
            composition = result.get("composition")
            exact = (
                process.returncode == 0
                and result.get("status") == "completed"
                and isinstance(composition, Mapping)
                and composition.get("exactly_one_visual_instance_per_object") is True
                and sorted(composition.get("object_ids") or [])
                == request["expected_object_ids"]
            )
            if exact:
                base["status"] = "completed"
                base["blockers"] = []
                base["proof_boundary"][
                    "exactly_one_declared_visual_instance_rendered"
                ] = True
            else:
                base["blockers"] = sorted(
                    set(
                        list(result.get("blockers") or [])
                        + ["dynamic_splat_render_exactly_once_not_proven"]
                    )
                )
        if process.returncode != 0 and not base["blockers"]:
            base["blockers"] = ["dynamic_splat_render_process_failed"]
        if process.stderr:
            base["stderr_tail"] = process.stderr[-2000:]
    base["dynamic_splat_render_manifest_digest"] = _digest(
        base, "dynamic_splat_render_manifest_digest"
    )
    return base


__all__ = [
    "DYNAMIC_SPLAT_RENDER_MANIFEST_SCHEMA_VERSION",
    "DYNAMIC_SPLAT_RENDER_REQUEST_SCHEMA_VERSION",
    "DYNAMIC_SPLAT_SCENE_SCHEMA_VERSION",
    "DynamicSplatSceneError",
    "build_dynamic_splat_render_request",
    "build_dynamic_splat_scene",
    "render_dynamic_splat_frame",
    "validate_dynamic_splat_scene",
]
