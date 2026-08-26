"""Materialize provider-safe ArtiFixer method inputs after Website intake.

Raw InteriorGS bytes remain on the production control plane.  The worker
derives an exact, digest-bound target camera ring and invokes the qualified
reference renderer locally.  Its output packet contains only derived PNGs and
calibration/renderer receipts; that packet is the maximum disclosure allowed
to the external scene-configuration provider.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_disclosure import (
    MATERIALIZED_STATUS,
    PENDING_PROVIDER_RENDER_STATUS,
    renders_on_provider,
    resolve_scene_configuration_disclosure,
)
from .gaussian_splat_decode import (
    convert_to_standard_ply,
    read_standard_3dgs_ply,
    verify_standard_3dgs_ply_subset_exact,
    write_standard_3dgs_ply_subset_exact,
)
from .sealed_camera_render import render_splat_at_exact_cameras
from .task_evaluation_splat_render_runtime import runtime_from_environment


RESULT_SCHEMA_VERSION = "task_evaluation_scene_configuration_render_inputs.v1"
Renderer = Callable[..., Mapping[str, Any]]
RuntimeResolver = Callable[..., Mapping[str, Any]]
SplatDecoder = Callable[..., Mapping[str, Any]]


class TaskEvaluationSceneConfigurationRenderInputsError(ValueError):
    """The source render could not be prepared without disclosure or drift."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationRenderInputsError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationRenderInputsError(code)
    return dict(value)


def _materialized(
    envelope: Mapping[str, Any], *, contract_path: str
) -> tuple[Mapping[str, Any], Path]:
    rows = [
        row
        for row in envelope.get("materialized_references") or []
        if isinstance(row, Mapping) and row.get("contract_path") == contract_path
    ]
    if len(rows) != 1:
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            f"scene_configuration_render_reference_missing:{contract_path}"
        )
    row = rows[0]
    path = Path(str(row.get("materialized_path") or "")).resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != row.get("size_bytes")
        or _sha256(path) != row.get("digest")
        or row.get("full_byte_service_account_readback_passed") is not True
    ):
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            f"scene_configuration_render_reference_invalid:{contract_path}"
        )
    return row, path


def _look_at_opencv(eye: Sequence[float], target: Sequence[float]) -> list[list[float]]:
    position = np.asarray(eye, dtype=np.float64)
    look = np.asarray(target, dtype=np.float64)
    forward = look - position
    norm = float(np.linalg.norm(forward))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_camera_degenerate"
        )
    forward /= norm
    down_seed = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    right = np.cross(forward, down_seed)
    if float(np.linalg.norm(right)) <= 1e-9:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = down
    matrix[:3, 2] = forward
    matrix[:3, 3] = position
    return matrix.tolist()


def _stage_disclosure_intent(disclosure: Mapping[str, Any]) -> Any:
    """The stage's explicit intent about uploading source appearance bytes."""

    for key in ("source_appearance_bytes", "raw_interiorgs_bytes"):
        if key in disclosure:
            return disclosure[key]
    return None


def _target_camera_ring(
    *, minimum_xyz: Sequence[float], maximum_xyz: Sequence[float]
) -> list[dict[str, Any]]:
    low = np.asarray(minimum_xyz, dtype=np.float64)
    high = np.asarray(maximum_xyz, dtype=np.float64)
    if (
        low.shape != (3,)
        or high.shape != (3,)
        or not np.isfinite([*low, *high]).all()
        or np.any(high <= low)
    ):
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_target_bounds_invalid"
        )
    center = (low + high) / 2.0
    radius = max(float(np.linalg.norm(high - low)) * 2.5, 0.4)
    width = height = 1024
    vfov = math.radians(55.0)
    focal = height / (2.0 * math.tan(vfov / 2.0))
    rows: list[dict[str, Any]] = []
    for elevation_index, elevation_deg in enumerate((25.0, 55.0)):
        elevation = math.radians(elevation_deg)
        for azimuth_index in range(4):
            azimuth = 2.0 * math.pi * azimuth_index / 4.0
            eye = center + radius * np.asarray(
                [
                    math.cos(elevation) * math.cos(azimuth),
                    math.cos(elevation) * math.sin(azimuth),
                    math.sin(elevation),
                ]
            )
            rows.append(
                {
                    "camera_id": (f"target-e{elevation_index}-a{azimuth_index}"),
                    "T_world_camera_provider_frame": _look_at_opencv(eye.tolist(), center.tolist()),
                    "intrinsics": {
                        "fx": focal,
                        "fy": focal,
                        "cx": width / 2.0,
                        "cy": height / 2.0,
                        "width": width,
                        "height": height,
                        "near": 0.01,
                        "far": 100.0,
                    },
                }
            )
    return rows


def _project_registered_bounds_mask(
    *,
    minimum_xyz: Sequence[float],
    maximum_xyz: Sequence[float],
    camera: Mapping[str, Any],
    frame_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Project the preregistered source-object AABB into one exact camera.

    The mask is a conservative method input, not observed segmentation truth.
    Its provenance remains explicit so ArtiFixer cannot silently turn an
    inferred box projection into capture evidence.
    """

    low = np.asarray(minimum_xyz, dtype=np.float64)
    high = np.asarray(maximum_xyz, dtype=np.float64)
    pose = np.asarray(camera["T_world_camera_provider_frame"], dtype=np.float64)
    intrinsics = camera["intrinsics"]
    if (
        low.shape != (3,)
        or high.shape != (3,)
        or pose.shape != (4, 4)
        or not np.isfinite([*low, *high]).all()
        or not np.isfinite(pose).all()
    ):
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_mask_projection_invalid"
        )
    try:
        with Image.open(frame_path) as frame:
            width, height = frame.size
    except (OSError, ValueError) as exc:
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_frame_invalid"
        ) from exc
    if (width, height) != (
        int(intrinsics["width"]),
        int(intrinsics["height"]),
    ):
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_frame_dimensions_invalid"
        )
    corners = np.asarray(
        [
            [x, y, z, 1.0]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=np.float64,
    )
    try:
        camera_from_world = np.linalg.inv(pose)
    except np.linalg.LinAlgError as exc:
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_mask_projection_invalid"
        ) from exc
    camera_points = (camera_from_world @ corners.T).T[:, :3]
    near = float(intrinsics["near"])
    if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= near):
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_target_not_visible"
        )
    projected_u = float(intrinsics["fx"]) * camera_points[:, 0] / camera_points[:, 2] + float(
        intrinsics["cx"]
    )
    projected_v = float(intrinsics["fy"]) * camera_points[:, 1] / camera_points[:, 2] + float(
        intrinsics["cy"]
    )
    left = max(0, int(math.floor(float(projected_u.min()))))
    top = max(0, int(math.floor(float(projected_v.min()))))
    right = min(width, int(math.ceil(float(projected_u.max()))))
    bottom = min(height, int(math.ceil(float(projected_v.max()))))
    if right <= left or bottom <= top:
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_target_not_visible"
        )
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path, format="PNG", optimize=False)
    return {
        "path": str(output_path),
        "digest": _sha256(output_path),
        "size_bytes": output_path.stat().st_size,
        "projection_kind": "registered_world_aabb_conservative_projection",
        "observed_segmentation_truth": False,
        "pixel_bounds_xyxy": [left, top, right, bottom],
        "foreground_pixel_count": int((right - left) * (bottom - top)),
    }


def materialize_scene_configuration_render_inputs(
    *,
    envelope: Mapping[str, Any],
    stage_one_configuration: Mapping[str, Any],
    output_root: str | Path,
    renderer: Renderer = render_splat_at_exact_cameras,
    runtime_resolver: RuntimeResolver = runtime_from_environment,
    splat_decoder: SplatDecoder = convert_to_standard_ply,
) -> dict[str, Any]:
    """Render exact derived method inputs without exposing the raw source."""

    source_object = stage_one_configuration.get("source_object")
    gaussian_cutout = stage_one_configuration.get("gaussian_cutout")
    required_views = stage_one_configuration.get("required_views")
    disclosure = stage_one_configuration.get("provider_disclosure")
    human_authority = stage_one_configuration.get("human_authority")
    if (
        stage_one_configuration.get("schema_version")
        != "observed_appearance_object_removal_configuration.v1"
        or stage_one_configuration.get("production_render_required") is not True
        or not isinstance(source_object, Mapping)
        or not isinstance(gaussian_cutout, Mapping)
        or gaussian_cutout.get("selection_rule")
        != "gaussian_center_inside_registered_source_object_aabb"
        or not isinstance(gaussian_cutout.get("aabb_padding_m"), (int, float))
        or isinstance(gaussian_cutout.get("aabb_padding_m"), bool)
        or not 0.0 <= float(gaussian_cutout["aabb_padding_m"]) <= 0.10
        or gaussian_cutout.get("retained_rows_must_remain_byte_exact") is not True
        or not isinstance(required_views, Mapping)
        or required_views.get("minimum", 0) > 8
        or required_views.get("lossless_inputs") is not True
        or required_views.get("mask_source") != "registered_source_object_bounds_projection"
        or not str(source_object.get("publisher_instance_id") or "").strip()
        or not isinstance(disclosure, Mapping)
        # Whether source appearance bytes may reach the provider is decided
        # against the scene's rights admission, not asserted here. The stage
        # must still state an explicit boolean intent rather than stay silent.
        or not isinstance(
            _stage_disclosure_intent(disclosure), bool
        )
        or disclosure.get("derived_rendered_views") is not True
        or not isinstance(human_authority, Mapping)
        or not str(human_authority.get("accepted_by") or "").strip()
        or not str(human_authority.get("accepted_on") or "").strip()
        or not str(human_authority.get("authority_reference") or "").strip()
        or human_authority.get("private_derived_frame_disclosure_authorized") is not True
        or human_authority.get("provider_retention_terms_accepted") is not True
        or human_authority.get("provider_training_terms_accepted") is not True
        or human_authority.get("provider_training_authorized") is not False
    ):
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_stage_configuration_invalid"
        )
    appearance_row, appearance_path = _materialized(
        envelope, contract_path="scene.appearance.representation"
    )
    _manifest_row, manifest_path = _materialized(envelope, contract_path="scene.source_manifest")
    _plan_row, plan_path = _materialized(
        envelope, contract_path="scene.appearance.renderer_qualification"
    )
    manifest = _read(manifest_path, code="scene_configuration_render_source_manifest_invalid")
    plan = _read(plan_path, code="scene_configuration_render_qualification_plan_invalid")
    # An absent or unreadable admission is not an error here -- it simply
    # cannot grant an upload, so the render stays on the control plane.
    try:
        _rights_row, rights_path = _materialized(
            envelope, contract_path="scene.rights.admission"
        )
        rights_admission: Mapping[str, Any] = _read(
            rights_path, code="scene_configuration_render_rights_admission_invalid"
        )
    except TaskEvaluationSceneConfigurationRenderInputsError:
        rights_admission = {}
    disclosure_decision = resolve_scene_configuration_disclosure(
        stage_one_configuration=stage_one_configuration,
        rights_admission=rights_admission,
    )
    provider_render = renders_on_provider(disclosure_decision)
    source_matches = [
        row
        for row in manifest.get("artifacts") or []
        if isinstance(row, Mapping)
        and row.get("role") == "interiorgs_source_splat"
        and row.get("sha256") == appearance_row["digest"]
        and row.get("size_bytes") == appearance_row["size_bytes"]
    ]
    if (
        len(source_matches) != 1
        or source_matches[0].get("provider_upload_allowed") is not False
        or plan.get("schema_version") != "task_evaluation_renderer_qualification_plan.v1"
        or plan.get("status") != "execute_during_scene_configuration_run"
        or plan.get("appearance_source") != "InteriorGS"
        or plan.get("browser_preview_qualifies") is not False
        or plan.get("debug_sage_render_qualifies_as_appearance") is not False
    ):
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_source_or_plan_invalid"
        )
    source = source_matches[0]
    cameras = _target_camera_ring(
        minimum_xyz=source_object["aabb_min_xyz_m"],
        maximum_xyz=source_object["aabb_max_xyz_m"],
    )
    repository_root = Path(__file__).resolve().parents[2]
    runtime = dict(runtime_resolver(repo_root=repository_root))
    root = Path(output_root).resolve()
    if root.is_symlink() or (root.exists() and any(root.iterdir())):
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_output_not_empty"
        )
    root.mkdir(parents=True, exist_ok=True)
    decoded_path = root / "source_standard_decoded_for_local_cutout.ply"
    decoded = dict(
        splat_decoder(
            appearance_path,
            decoded_path,
            repo_root=runtime["renderer_root"],
            node=str(runtime["node"]),
        )
    )
    if decoded.get("status") != "completed" or not decoded_path.is_file():
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_splat_decode_failed"
        )
    splat = read_standard_3dgs_ply(decoded_path)
    padding = float(gaussian_cutout["aabb_padding_m"])
    cutout_low = np.asarray(source_object["aabb_min_xyz_m"], dtype=np.float64) - padding
    cutout_high = np.asarray(source_object["aabb_max_xyz_m"], dtype=np.float64) + padding
    selected_mask = np.all(
        (splat.xyz.astype(np.float64) >= cutout_low)
        & (splat.xyz.astype(np.float64) <= cutout_high),
        axis=1,
    )
    removed_indices = np.flatnonzero(selected_mask).astype(np.int64)
    retained_indices = np.flatnonzero(~selected_mask).astype(np.int64)
    if not removed_indices.size or not retained_indices.size:
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_gaussian_cutout_invalid"
        )
    removed_path = root / "source_object_candidate_gaussians.ply"
    retained_path = root / "retained_scene_gaussians_without_source_object.ply"
    write_standard_3dgs_ply_subset_exact(decoded_path, removed_path, removed_indices)
    write_standard_3dgs_ply_subset_exact(decoded_path, retained_path, retained_indices)
    preservation = verify_standard_3dgs_ply_subset_exact(
        decoded_path, retained_path, retained_indices
    )
    if preservation.get("retained_rows_byte_exact") is not True:
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_gaussian_cutout_preservation_failed"
        )
    calibration_path = root / "artifixer_method_input_cameras.v1.json"
    calibration_path.write_text(
        canonical_json(
            [
                {
                    "id": row["camera_id"],
                    "spec": {
                        "pose": {"T_world_camera_opencv": row["T_world_camera_provider_frame"]},
                        "intrinsics": row["intrinsics"],
                    },
                }
                for row in cameras
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rendered: dict[str, Any] = {}
    render_manifest_path = root / "rendered" / "sealed_camera_render_manifest.v1.json"
    derived_frames: list[dict[str, Any]] = []
    # When the scene's rights admit it, the already-rented configuration GPU
    # renders these exact cameras instead. Nothing else about the packet
    # changes: the same cutout, calibration and camera ring are produced here,
    # and the provider reproduces the frames and masks from them.
    if not provider_render:
        rendered = dict(
            renderer(
                splat_path=appearance_path,
                cameras=cameras,
                output_dir=root / "rendered",
                provider_splat_import_receipt_digest=appearance_row["digest"],
                alignment_digest=envelope["request"]["scene"]["registration"]["metric_registration"][
                    "digest"
                ],
                camera_set_label="artifixer-source-object-method-inputs",
                calibrated_camera_file=calibration_path,
                retained_gaussian_count=int(source["splat_count"]),
                source_splat_digest=appearance_row["digest"],
                purpose="artifixer_source_object_removal_method_inputs",
                authorization_class="method_input",
                repo_root=repository_root,
                node=str(runtime["node"]),
                renderer_runtime_root=str(runtime["renderer_root"]),
                browser_executable=str(runtime["browser_executable"]),
                renderer_runtime_identity=dict(runtime["identity"]),
            )
        )
        if (
            rendered.get("status") != "rendered_exact_cameras"
            or rendered.get("authorization_class") != "method_input"
            or rendered.get("render_count") != len(cameras)
            or rendered.get("splat_digest") != appearance_row["digest"]
            or rendered.get("sealed_camera_render_manifest_digest")
            != canonical_digest(rendered, digest_field="sealed_camera_render_manifest_digest")
        ):
            raise TaskEvaluationSceneConfigurationRenderInputsError(
                "scene_configuration_render_result_invalid"
            )
        render_manifest_path = root / "rendered" / "sealed_camera_render_manifest.v1.json"
        if not render_manifest_path.is_file():
            render_manifest_path.write_text(canonical_json(rendered) + "\n", encoding="utf-8")
        cameras_by_id = {row["camera_id"]: row for row in cameras}
        derived_frames = []
        for row in rendered["renders"]:
            frame = root / "rendered" / row["relative_path"]
            if frame.is_symlink() or not frame.is_file() or _sha256(frame) != row["digest"]:
                raise TaskEvaluationSceneConfigurationRenderInputsError(
                    "scene_configuration_render_frame_invalid"
                )
            camera_id = str(row["camera_id"])
            camera = cameras_by_id.get(camera_id)
            if camera is None:
                raise TaskEvaluationSceneConfigurationRenderInputsError(
                    "scene_configuration_render_camera_result_mismatch"
                )
            mask = _project_registered_bounds_mask(
                minimum_xyz=source_object["aabb_min_xyz_m"],
                maximum_xyz=source_object["aabb_max_xyz_m"],
                camera=camera,
                frame_path=frame,
                output_path=root / "masks" / f"{camera_id}.png",
            )
            derived_frames.append(
                {
                    "camera_id": camera_id,
                    "path": str(frame),
                    "digest": row["digest"],
                    "size_bytes": frame.stat().st_size,
                    "source_object_mask": mask,
                }
            )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": (
            "derived_method_inputs_pending_provider_render"
            if provider_render
            else "derived_method_inputs_materialized"
        ),
        "run_id": envelope["run_id"],
        "publisher_instance_id": source_object["publisher_instance_id"],
        "source_splat_digest": appearance_row["digest"],
        "source_splat_bytes_retained_on_control_plane": not provider_render,
        "raw_interiorgs_bytes_in_provider_packet": provider_render,
        "provider_disclosure_scope": (
            "source_appearance_bytes_and_derived_views"
            if provider_render
            else "derived_rendered_views_only"
        ),
        "disclosure_decision": disclosure_decision,
        "render_execution_site": disclosure_decision["render_execution_site"],
        "source_appearance": {
            "path": str(appearance_path),
            "digest": appearance_row["digest"],
            "size_bytes": appearance_row["size_bytes"],
        },
        "camera_calibration": {
            "path": str(calibration_path),
            "digest": _sha256(calibration_path),
            "size_bytes": calibration_path.stat().st_size,
        },
        "render_manifest": (
            None
            if provider_render
            else {
                "path": str(render_manifest_path),
                "digest": _sha256(render_manifest_path),
                "size_bytes": render_manifest_path.stat().st_size,
                "manifest_digest": rendered["sealed_camera_render_manifest_digest"],
            }
        ),
        "derived_frames": derived_frames,
        "derived_frame_count": len(derived_frames),
        "source_object_masks": {
            "count": len(derived_frames),
            "source": required_views["mask_source"],
            "source_object_identity": {
                "publisher_instance_id": source_object["publisher_instance_id"],
            },
            "observed_segmentation_truth": False,
            "all_masks_digest_bound": True,
        },
        "derived_gaussian_cutout": {
            "selection_rule": gaussian_cutout["selection_rule"],
            "aabb_padding_m": padding,
            "source_count": splat.count,
            "removed_count": int(removed_indices.size),
            "retained_count": int(retained_indices.size),
            "source_object_candidate": {
                "path": str(removed_path),
                "digest": _sha256(removed_path),
                "size_bytes": removed_path.stat().st_size,
            },
            "retained_scene_without_source_object": {
                "path": str(retained_path),
                "digest": _sha256(retained_path),
                "size_bytes": retained_path.stat().st_size,
            },
            "retained_rows_byte_exact": True,
            "selection_is_candidate_not_observed_object_ownership_truth": True,
            "raw_source_bytes_in_provider_packet": provider_render,
        },
        "browser_preview_used_as_method_input": False,
        "sage_render_used_as_appearance": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "renderer_runtime": dict(runtime["identity"]),
        "provider_render_required": provider_render,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (root / f"{RESULT_SCHEMA_VERSION}.json").write_text(
        canonical_json(result) + "\n", encoding="utf-8"
    )
    return result



def complete_provider_render_inputs(
    *,
    render_inputs: Mapping[str, Any],
    appearance_path: str | Path,
    source_object: Mapping[str, Any],
    output_root: str | Path,
    input_root: str | Path | None = None,
    renderer: Renderer = render_splat_at_exact_cameras,
    runtime_resolver: RuntimeResolver = runtime_from_environment,
    graphics_backend: str = "egl",
) -> dict[str, Any]:
    """Render the owed views on the provider that already holds the scene.

    The control plane produced everything that *binds* this render -- the exact
    camera ring, the calibration file, the cutout layers -- and deferred only
    the rasterisation.  This reproduces the frames and masks from those exact
    inputs using the same renderer and the same projection, so the completed
    result is the packet the rest of the chain already knows how to consume.

    ``graphics_backend`` defaults to a real GPU: the renderer refuses to fall
    back to software rasterisation, so a host without acceleration fails closed
    here rather than silently spending an hour.
    """

    if str(render_inputs.get("status") or "") != PENDING_PROVIDER_RENDER_STATUS:
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_completion_not_pending"
        )
    if not renders_on_provider(render_inputs.get("disclosure_decision") or {}):
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_completion_not_authorized"
        )
    # Inside a provider bundle these paths are bundle-relative; resolve them
    # against the unpacked root rather than whatever the working directory is.
    base = Path(input_root).resolve() if input_root is not None else None

    def _resolve(value: str | Path) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute() and base is not None:
            candidate = base / candidate
        return candidate.resolve()

    splat = _resolve(appearance_path)
    if splat.is_symlink() or not splat.is_file():
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_completion_source_missing"
        )
    declared = str((render_inputs.get("source_appearance") or {}).get("digest") or "")
    if declared and _sha256(splat) != declared:
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_completion_source_digest_mismatch"
        )
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    calibration_path = _resolve(
        str((render_inputs.get("camera_calibration") or {}).get("path") or "")
    )
    try:
        calibration_rows = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_completion_calibration_invalid"
        ) from exc
    cameras = [
        {
            "camera_id": str(row["id"]),
            "T_world_camera_provider_frame": row["spec"]["pose"]["T_world_camera_opencv"],
            "intrinsics": row["spec"]["intrinsics"],
        }
        for row in calibration_rows
    ]
    if not cameras:
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_completion_calibration_invalid"
        )
    runtime = dict(runtime_resolver(repo_root=Path(__file__).resolve().parents[2]))
    rendered = dict(
        renderer(
            splat_path=splat,
            cameras=cameras,
            output_dir=root / "rendered",
            provider_splat_import_receipt_digest=render_inputs["source_splat_digest"],
            alignment_digest=render_inputs["source_splat_digest"],
            camera_set_label="artifixer-source-object-method-inputs",
            calibrated_camera_file=calibration_path,
            source_splat_digest=render_inputs["source_splat_digest"],
            purpose="artifixer_source_object_removal_method_inputs",
            authorization_class="method_input",
            repo_root=Path(__file__).resolve().parents[2],
            node=str(runtime["node"]),
            renderer_runtime_root=str(runtime["renderer_root"]),
            browser_executable=str(runtime["browser_executable"]),
            renderer_runtime_identity=dict(runtime["identity"]),
            graphics_backend=graphics_backend,
        )
    )
    if (
        rendered.get("status") != "rendered_exact_cameras"
        or rendered.get("authorization_class") != "method_input"
        or int(rendered.get("render_count") or 0) != len(cameras)
    ):
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_completion_failed"
        )
    render_manifest_path = root / "rendered" / "sealed_camera_render_manifest.v1.json"
    if not render_manifest_path.is_file():
        render_manifest_path.write_text(canonical_json(rendered) + "\n", encoding="utf-8")
    cameras_by_id = {row["camera_id"]: row for row in cameras}
    derived_frames: list[dict[str, Any]] = []
    for row in rendered["renders"]:
        frame = (root / "rendered" / str(row["relative_path"])).resolve()
        camera_id = str(row["camera_id"])
        camera = cameras_by_id.get(camera_id)
        if camera is None or not frame.is_file() or _sha256(frame) != row["digest"]:
            raise TaskEvaluationSceneConfigurationRenderInputsError(
                "scene_configuration_render_camera_result_mismatch"
            )
        mask = _project_registered_bounds_mask(
            minimum_xyz=source_object["aabb_min_xyz_m"],
            maximum_xyz=source_object["aabb_max_xyz_m"],
            camera=camera,
            frame_path=frame,
            output_path=root / "masks" / f"{camera_id}.png",
        )
        derived_frames.append(
            {
                "camera_id": camera_id,
                "path": str(frame),
                "digest": row["digest"],
                "size_bytes": frame.stat().st_size,
                "source_object_mask": mask,
            }
        )
    completed = json.loads(json.dumps(dict(render_inputs)))
    completed["status"] = MATERIALIZED_STATUS
    completed["render_manifest"] = {
        "path": str(render_manifest_path),
        "digest": _sha256(render_manifest_path),
        "size_bytes": render_manifest_path.stat().st_size,
        "manifest_digest": rendered["sealed_camera_render_manifest_digest"],
    }
    completed["derived_frames"] = derived_frames
    completed["derived_frame_count"] = len(derived_frames)
    completed["source_object_masks"] = {
        **dict(render_inputs.get("source_object_masks") or {}),
        "count": len(derived_frames),
    }
    completed["renderer_runtime"] = dict(runtime["identity"])
    completed["camera_calibration"] = {
        **dict(render_inputs.get("camera_calibration") or {}),
        "path": str(calibration_path),
    }
    completed["render_completed_on_provider"] = True
    completed["control_plane_result_digest"] = render_inputs.get("result_digest")
    completed["result_digest"] = ""
    completed["result_digest"] = canonical_digest(
        completed, digest_field="result_digest"
    )
    (root / f"{RESULT_SCHEMA_VERSION}.json").write_text(
        canonical_json(completed) + "\n", encoding="utf-8"
    )
    return completed


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationRenderInputsError",
    "complete_provider_render_inputs",
    "materialize_scene_configuration_render_inputs",
]
