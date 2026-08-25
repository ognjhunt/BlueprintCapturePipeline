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

from .decision_evidence_contracts import canonical_digest, canonical_json
from .sealed_camera_render import render_splat_at_exact_cameras
from .task_evaluation_splat_render_runtime import runtime_from_environment


RESULT_SCHEMA_VERSION = "task_evaluation_scene_configuration_render_inputs.v1"
Renderer = Callable[..., Mapping[str, Any]]
RuntimeResolver = Callable[..., Mapping[str, Any]]


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
                    "camera_id": (
                        f"target-e{elevation_index}-a{azimuth_index}"
                    ),
                    "T_world_camera_provider_frame": _look_at_opencv(
                        eye.tolist(), center.tolist()
                    ),
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


def materialize_scene_configuration_render_inputs(
    *,
    envelope: Mapping[str, Any],
    stage_one_configuration: Mapping[str, Any],
    output_root: str | Path,
    renderer: Renderer = render_splat_at_exact_cameras,
    runtime_resolver: RuntimeResolver = runtime_from_environment,
) -> dict[str, Any]:
    """Render exact derived method inputs without exposing the raw source."""

    source_object = stage_one_configuration.get("source_object")
    required_views = stage_one_configuration.get("required_views")
    disclosure = stage_one_configuration.get("provider_disclosure")
    if (
        stage_one_configuration.get("schema_version")
        != "observed_appearance_object_removal_configuration.v1"
        or stage_one_configuration.get("production_render_required") is not True
        or not isinstance(source_object, Mapping)
        or not isinstance(required_views, Mapping)
        or required_views.get("minimum", 0) > 8
        or required_views.get("lossless_inputs") is not True
        or not isinstance(disclosure, Mapping)
        or disclosure.get("raw_interiorgs_bytes") is not False
        or disclosure.get("derived_rendered_views") is not True
    ):
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_stage_configuration_invalid"
        )
    appearance_row, appearance_path = _materialized(
        envelope, contract_path="scene.appearance.representation"
    )
    _manifest_row, manifest_path = _materialized(
        envelope, contract_path="scene.source_manifest"
    )
    _plan_row, plan_path = _materialized(
        envelope, contract_path="scene.appearance.renderer_qualification"
    )
    manifest = _read(
        manifest_path, code="scene_configuration_render_source_manifest_invalid"
    )
    plan = _read(
        plan_path, code="scene_configuration_render_qualification_plan_invalid"
    )
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
        or plan.get("schema_version")
        != "task_evaluation_renderer_qualification_plan.v1"
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
    root = Path(output_root).resolve()
    if root.is_symlink() or (root.exists() and any(root.iterdir())):
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_output_not_empty"
        )
    root.mkdir(parents=True, exist_ok=True)
    calibration_path = root / "artifixer_method_input_cameras.v1.json"
    calibration_path.write_text(
        canonical_json(
            [
                {
                    "id": row["camera_id"],
                    "spec": {
                        "pose": {
                            "T_world_camera_opencv": row[
                                "T_world_camera_provider_frame"
                            ]
                        },
                        "intrinsics": row["intrinsics"],
                    },
                }
                for row in cameras
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repository_root = Path(__file__).resolve().parents[2]
    runtime = dict(runtime_resolver(repo_root=repository_root))
    rendered = dict(
        renderer(
            splat_path=appearance_path,
            cameras=cameras,
            output_dir=root / "rendered",
            provider_splat_import_receipt_digest=appearance_row["digest"],
            alignment_digest=envelope["request"]["scene"]["registration"][
                "metric_registration"
            ]["digest"],
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
        != canonical_digest(
            rendered, digest_field="sealed_camera_render_manifest_digest"
        )
    ):
        raise TaskEvaluationSceneConfigurationRenderInputsError(
            "scene_configuration_render_result_invalid"
        )
    render_manifest_path = root / "rendered" / "sealed_camera_render_manifest.v1.json"
    if not render_manifest_path.is_file():
        render_manifest_path.write_text(
            canonical_json(rendered) + "\n", encoding="utf-8"
        )
    derived_frames = []
    for row in rendered["renders"]:
        frame = root / "rendered" / row["relative_path"]
        if (
            frame.is_symlink()
            or not frame.is_file()
            or _sha256(frame) != row["digest"]
        ):
            raise TaskEvaluationSceneConfigurationRenderInputsError(
                "scene_configuration_render_frame_invalid"
            )
        derived_frames.append(
            {
                "camera_id": row["camera_id"],
                "path": str(frame),
                "digest": row["digest"],
                "size_bytes": frame.stat().st_size,
            }
        )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "derived_method_inputs_materialized",
        "run_id": envelope["run_id"],
        "publisher_instance_id": source_object["publisher_instance_id"],
        "source_splat_digest": appearance_row["digest"],
        "source_splat_bytes_retained_on_control_plane": True,
        "raw_interiorgs_bytes_in_provider_packet": False,
        "provider_disclosure_scope": "derived_rendered_views_only",
        "camera_calibration": {
            "path": str(calibration_path),
            "digest": _sha256(calibration_path),
            "size_bytes": calibration_path.stat().st_size,
        },
        "render_manifest": {
            "path": str(render_manifest_path),
            "digest": _sha256(render_manifest_path),
            "size_bytes": render_manifest_path.stat().st_size,
            "manifest_digest": rendered[
                "sealed_camera_render_manifest_digest"
            ],
        },
        "derived_frames": derived_frames,
        "derived_frame_count": len(derived_frames),
        "browser_preview_used_as_method_input": False,
        "sage_render_used_as_appearance": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "renderer_runtime": dict(runtime["identity"]),
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(
        result, digest_field="result_digest"
    )
    (root / f"{RESULT_SCHEMA_VERSION}.json").write_text(
        canonical_json(result) + "\n", encoding="utf-8"
    )
    return result


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationRenderInputsError",
    "materialize_scene_configuration_render_inputs",
]
