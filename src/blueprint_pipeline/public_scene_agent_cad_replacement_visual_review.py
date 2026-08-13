"""Render agent-authored replacement CAD over protected ArtiFixer3D views.

This module is the deterministic visual-review seam between an object-free,
exact-support-protected ArtiFixer3D representation and an already-authored
replacement USD.  It never authors candidate geometry: each generated USD
contains only a reference to the digest-bound candidate plus one calibrated
review camera.  The renderer emits an RGBA replacement layer, which is
composited over the corresponding final-composite frame.  Every zero-alpha
background pixel must remain byte-identical.

The result is review media, not native-simulator, collision, policy, capture, or
physical evidence.  One call accepts one to five co-present replacement tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from PIL import Image

from .agent_cad_graph_visual_composition import (
    COMPOSITION_SCHEMA_VERSION,
    validate_agent_cad_visual_composition,
)
from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS


SCHEMA_VERSION = "public_scene_agent_cad_replacement_visual_review.v1"
DUAL_INPUT_SCHEMA = "public_scene_artifixer3d_dual_target_inputs.v1"
FINAL_COMPOSITE_SCHEMA = "public_scene_artifixer3d_final_composite.v1"
DEFAULT_RENDERER = "/usr/bin/usdrecord"
DEFAULT_RENDERER_PLUGIN = "Metal"
HORIZONTAL_APERTURE_MM = 20.955
MAX_CAMERAS_PER_TASK = 64
_DUAL_STATUS = "paired_target_inputs_prepared_no_model_no_execution"


class AgentCadReplacementVisualReviewError(ValueError):
    """Stable fail-closed error codes for replacement visual review."""


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
        raise AgentCadReplacementVisualReviewError(code) from exc
    if not isinstance(value, dict):
        raise AgentCadReplacementVisualReviewError(code)
    return value


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if root is None:
        record["path"] = str(path.resolve())
    else:
        record["relative_path"] = path.relative_to(root).as_posix()
    return record


def _bound_file(value: Any, *, root: Path | None, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise AgentCadReplacementVisualReviewError(code)
    raw = Path(str(value.get("path") or value.get("relative_path") or "")).expanduser()
    if not raw.is_absolute():
        if root is None:
            raise AgentCadReplacementVisualReviewError(code)
        raw = root / raw
    if raw.is_symlink():
        raise AgentCadReplacementVisualReviewError(code)
    try:
        path = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AgentCadReplacementVisualReviewError(code) from exc
    if (
        not path.is_file()
        or isinstance(value.get("size_bytes"), bool)
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise AgentCadReplacementVisualReviewError(code)
    return path


def _load_png(path: Path, *, mode: str, code: str) -> np.ndarray:
    try:
        with Image.open(path) as opened:
            if opened.format != "PNG":
                raise AgentCadReplacementVisualReviewError(code)
            opened.load()
            return np.asarray(opened.convert(mode), dtype=np.uint8)
    except (OSError, ValueError, SyntaxError) as exc:
        raise AgentCadReplacementVisualReviewError(code) from exc


def _validate_camera(
    frame: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> tuple[int, int, float, np.ndarray]:
    def value(name: str) -> Any:
        return frame.get(name, trajectory.get(name))

    try:
        width = int(value("w"))
        height = int(value("h"))
        fx = float(value("fl_x"))
        fy = float(value("fl_y"))
        cx = float(value("cx"))
        cy = float(value("cy"))
        transform = np.asarray(frame["transform_matrix"], dtype=np.float64)
        distortion = [float(value(name)) for name in ("k1", "k2", "p1", "p2")]
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentCadReplacementVisualReviewError(
            "replacement_visual_camera_calibration_invalid"
        ) from exc
    if (
        frame.get("camera_model", trajectory.get("camera_model")) != "OPENCV"
        or not 1 <= width <= 16384
        or not 1 <= height <= 16384
        or not np.isfinite([fx, fy, cx, cy, *distortion, *transform.ravel()]).all()
        or fx <= 0
        or fy <= 0
        or transform.shape != (4, 4)
        or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
        or not np.isclose(fx, fy, rtol=0.0, atol=1e-6)
        or not np.isclose(cx, width / 2.0, rtol=0.0, atol=1e-6)
        or not np.isclose(cy, height / 2.0, rtol=0.0, atol=1e-6)
        or any(abs(item) > 1e-12 for item in distortion)
    ):
        # The current camera adapter is deliberately narrow: centered, square-pixel,
        # already-undistorted OPENCV cameras.  Other intrinsics need another proven
        # adapter, not an unreviewed approximation at this seam.
        raise AgentCadReplacementVisualReviewError(
            "replacement_visual_camera_adapter_unsupported"
        )
    return width, height, fx, transform


def _write_camera_stage(
    *, asset: Path, destination: Path, width: int, height: int, fx: float,
    camera_to_world: np.ndarray
) -> None:
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - environment guard
        raise AgentCadReplacementVisualReviewError(
            "replacement_visual_openusd_runtime_missing"
        ) from exc
    stage = Usd.Stage.CreateNew(str(destination))
    if stage is None:
        raise AgentCadReplacementVisualReviewError(
            "replacement_visual_camera_stage_failed"
        )
    asset_prim = stage.DefinePrim("/Asset", "Xform")
    asset_prim.GetReferences().AddReference(asset.as_posix(), "/Asset")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    camera = UsdGeom.Camera.Define(stage, "/ReviewCamera")
    matrix = Gf.Matrix4d(*camera_to_world.T.reshape(-1).tolist())
    UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)
    camera.GetHorizontalApertureAttr().Set(HORIZONTAL_APERTURE_MM)
    camera.GetVerticalApertureAttr().Set(HORIZONTAL_APERTURE_MM * height / width)
    camera.GetFocalLengthAttr().Set(fx * HORIZONTAL_APERTURE_MM / width)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
    stage.GetRootLayer().documentation = (
        "Deterministic review wrapper only: references immutable agent-authored "
        "candidate geometry and adds one calibrated camera; authors no geometry."
    )
    stage.GetRootLayer().Save()


def _renderer_identity(renderer: Path) -> dict[str, Any]:
    if renderer.is_symlink():
        raise AgentCadReplacementVisualReviewError("replacement_visual_renderer_invalid")
    try:
        path = renderer.resolve(strict=True)
        completed = subprocess.run(
            [str(path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise AgentCadReplacementVisualReviewError(
            "replacement_visual_renderer_invalid"
        ) from exc
    version = (completed.stdout + completed.stderr).strip()
    if not path.is_file() or not version:
        raise AgentCadReplacementVisualReviewError("replacement_visual_renderer_invalid")
    return {**_file_record(path), "version": version}


def _render_layer(
    *, renderer: Path, plugin: str, stage: Path, output: Path, width: int
) -> None:
    command = [
        str(renderer),
        "--renderer",
        plugin,
        "--camera",
        "/ReviewCamera",
        "--imageWidth",
        str(width),
        str(stage),
        str(output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        raise AgentCadReplacementVisualReviewError(
            "replacement_visual_renderer_failed"
        ) from exc
    if not output.is_file() or output.is_symlink() or output.stat().st_size <= 0:
        raise AgentCadReplacementVisualReviewError("replacement_visual_layer_invalid")


def _composition_claims_valid(receipt: Mapping[str, Any]) -> bool:
    claim = receipt.get("claim_boundary")
    return bool(
        isinstance(claim, Mapping)
        and claim.get("agent_authored_step_visual_geometry") is True
        and claim.get("deterministic_geometry_generator_used") is False
        and claim.get("appearance_materially_qualified") is False
        and claim.get("native_simulator_import_qualified") is False
        and claim.get("joint_physics_behavior_qualified") is False
        and claim.get("physical_equivalence_proven") is False
    )


def materialize_agent_cad_replacement_visual_review(
    *,
    dual_input_receipt_paths: Sequence[str | Path],
    final_composite_receipt_paths: Sequence[str | Path],
    visual_composition_receipt_paths: Sequence[str | Path],
    output_root: str | Path,
    renderer_executable: str | Path = DEFAULT_RENDERER,
    renderer_plugin: str = DEFAULT_RENDERER_PLUGIN,
) -> dict[str, Any]:
    """Render, composite, verify, and seal one to five replacement reviews."""

    if (
        not dual_input_receipt_paths
        or len(dual_input_receipt_paths) != len(final_composite_receipt_paths)
        or len(final_composite_receipt_paths)
        != len(visual_composition_receipt_paths)
        or not 1 <= len(final_composite_receipt_paths) <= MAX_REPLACEMENT_OBJECTS
    ):
        raise AgentCadReplacementVisualReviewError(
            "replacement_visual_input_pairing_invalid"
        )
    output = Path(output_root).expanduser().resolve()
    if output.exists() and (
        output.is_symlink() or not output.is_dir() or any(output.iterdir())
    ):
        raise AgentCadReplacementVisualReviewError("replacement_visual_output_not_empty")
    output.mkdir(parents=True, exist_ok=True)
    renderer = Path(renderer_executable).expanduser()
    renderer_record = _renderer_identity(renderer)
    renderer = Path(renderer_record["path"])
    if not renderer_plugin or len(renderer_plugin) > 128:
        raise AgentCadReplacementVisualReviewError("replacement_visual_renderer_invalid")

    scene_id: str | None = None
    task_results: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for dual_value, final_value, composition_value in zip(
        dual_input_receipt_paths,
        final_composite_receipt_paths,
        visual_composition_receipt_paths,
        strict=True,
    ):
        dual_path = Path(dual_value).expanduser().resolve()
        final_path = Path(final_value).expanduser().resolve()
        composition_path = Path(composition_value).expanduser().resolve()
        dual = _read(dual_path, code="replacement_visual_dual_input_unreadable")
        final = _read(
            final_path, code="replacement_visual_final_composite_unreadable"
        )
        composition = _read(
            composition_path, code="replacement_visual_composition_unreadable"
        )
        if (
            dual.get("schema_version") != DUAL_INPUT_SCHEMA
            or dual.get("receipt_digest")
            != canonical_digest(dual, digest_field="receipt_digest")
            or dual.get("status") != _DUAL_STATUS
            or dual.get("pipeline_mode") != "dual_target_artifixer3d_only"
            or final.get("schema_version") != FINAL_COMPOSITE_SCHEMA
            or final.get("receipt_digest")
            != canonical_digest(final, digest_field="receipt_digest")
            or final.get("status")
            != "final_composite_materialized_pending_human_multiview_review"
            or final.get("appearance_repair_qualified") is not False
            or final.get("outside_support_invariance_proven") is not True
            or final.get("outside_support_changed_pixels_total") != 0
            or final.get("generated_output_is_capture_or_physical_evidence") is not False
            or composition.get("schema_version") != COMPOSITION_SCHEMA_VERSION
            or composition.get("receipt_digest")
            != canonical_digest(composition, digest_field="receipt_digest")
            or not _composition_claims_valid(composition)
        ):
            raise AgentCadReplacementVisualReviewError(
                "replacement_visual_input_receipt_invalid"
            )
        try:
            composition = validate_agent_cad_visual_composition(
                composition, verify_files=True
            )
        except ValueError as exc:
            raise AgentCadReplacementVisualReviewError(
                "replacement_visual_composition_invalid"
            ) from exc
        candidate_usd = _bound_file(
            composition.get("output_usd"),
            root=None,
            code="replacement_visual_candidate_usd_invalid",
        )
        dual_tasks = dual.get("tasks")
        final_tasks = final.get("tasks")
        if (
            not isinstance(dual_tasks, list)
            or len(dual_tasks) != 1
            or not isinstance(final_tasks, list)
            or not 1 <= len(final_tasks) <= MAX_REPLACEMENT_OBJECTS
            or not isinstance(dual_tasks[0], Mapping)
            or any(not isinstance(row, Mapping) for row in final_tasks)
        ):
            raise AgentCadReplacementVisualReviewError(
                "replacement_visual_task_inventory_invalid"
            )
        dual_task = dual_tasks[0]
        task_id = str(dual_task.get("task_id") or "")
        matching_final_tasks = [
            row for row in final_tasks if row.get("task_id") == task_id
        ]
        if len(matching_final_tasks) != 1:
            raise AgentCadReplacementVisualReviewError(
                "replacement_visual_task_inventory_invalid"
            )
        final_task = matching_final_tasks[0]
        this_scene = str(dual.get("publisher_scene_id") or "")
        if (
            not task_id
            or task_id in seen_tasks
            or final_task.get("task_id") != task_id
            or composition.get("task_id") != task_id
            or composition.get("scene_id") != this_scene
            or final.get("publisher_scene_id") != this_scene
            or (scene_id is not None and scene_id != this_scene)
        ):
            raise AgentCadReplacementVisualReviewError(
                "replacement_visual_task_binding_invalid"
            )
        scene_id = this_scene
        scene_directory = Path(str(dual_task.get("scene_directory") or "")).resolve()
        trajectory_path = _bound_file(
            dual_task.get("review_trajectory"),
            root=scene_directory,
            code="replacement_visual_trajectory_invalid",
        )
        trajectory = _read(
            trajectory_path, code="replacement_visual_trajectory_invalid"
        )
        trajectory_frames = trajectory.get("frames")
        dual_frames = dual_task.get("frames")
        final_frames = final_task.get("frames")
        camera_count = dual_task.get("physical_camera_count")
        if (
            isinstance(camera_count, bool)
            or not isinstance(camera_count, int)
            or not 1 <= camera_count <= MAX_CAMERAS_PER_TASK
            or not isinstance(trajectory_frames, list)
            or not isinstance(dual_frames, list)
            or not isinstance(final_frames, list)
            or len(trajectory_frames) != camera_count
            or len(dual_frames) != camera_count
            or len(final_frames) != camera_count
            or final_task.get("outside_support_invariance_proven") is not True
            or final_task.get("outside_support_changed_pixels_total") != 0
        ):
            raise AgentCadReplacementVisualReviewError(
                "replacement_visual_frame_inventory_invalid"
            )
        task_root = output / task_id
        task_root.mkdir()
        review_frames: list[dict[str, Any]] = []
        camera_ids: set[str] = set()
        for index, (camera_row, dual_frame, final_frame) in enumerate(
            zip(trajectory_frames, dual_frames, final_frames, strict=True)
        ):
            if not all(
                isinstance(row, Mapping)
                for row in (camera_row, dual_frame, final_frame)
            ):
                raise AgentCadReplacementVisualReviewError(
                    "replacement_visual_frame_inventory_invalid"
                )
            camera_id = str(camera_row.get("camera_id") or "")
            if (
                not camera_id
                or camera_id in camera_ids
                or camera_row.get("physical_camera_index") != index
                or dual_frame.get("physical_camera_index") != index
                or final_frame.get("frame_index") != index
                or dual_frame.get("camera_id") != camera_id
                or final_frame.get("camera_id") != camera_id
                or final_frame.get("outside_support_changed_pixels") != 0
            ):
                raise AgentCadReplacementVisualReviewError(
                    "replacement_visual_camera_binding_invalid"
                )
            width, height, fx, transform = _validate_camera(camera_row, trajectory)
            background_path = _bound_file(
                final_frame,
                root=None,
                code="replacement_visual_background_invalid",
            )
            exact_mask_path = _bound_file(
                dual_frame.get("source_exact_repair_mask"),
                root=None,
                code="replacement_visual_exact_mask_invalid",
            )
            background = _load_png(
                background_path, mode="RGB", code="replacement_visual_background_invalid"
            )
            exact_mask = _load_png(
                exact_mask_path, mode="L", code="replacement_visual_exact_mask_invalid"
            )
            if background.shape != (height, width, 3) or exact_mask.shape != (
                height,
                width,
            ):
                raise AgentCadReplacementVisualReviewError(
                    "replacement_visual_frame_geometry_invalid"
                )
            frame_root = task_root / f"{index:05d}_{camera_id}"
            frame_root.mkdir()
            camera_stage = frame_root / "camera_reference.usda"
            layer_path = frame_root / "replacement_layer.png"
            combined_path = task_root / f"{index:05d}.png"
            _write_camera_stage(
                asset=candidate_usd,
                destination=camera_stage,
                width=width,
                height=height,
                fx=fx,
                camera_to_world=transform,
            )
            _render_layer(
                renderer=renderer,
                plugin=renderer_plugin,
                stage=camera_stage,
                output=layer_path,
                width=width,
            )
            layer = _load_png(
                layer_path, mode="RGBA", code="replacement_visual_layer_invalid"
            )
            if layer.shape != (height, width, 4):
                raise AgentCadReplacementVisualReviewError(
                    "replacement_visual_layer_geometry_invalid"
                )
            alpha = layer[:, :, 3]
            alpha_support = alpha > 0
            if not np.any(alpha_support) or np.all(alpha_support):
                raise AgentCadReplacementVisualReviewError(
                    "replacement_visual_layer_alpha_invalid"
                )
            alpha16 = alpha.astype(np.uint16)[:, :, None]
            combined = (
                layer[:, :, :3].astype(np.uint16) * alpha16
                + background.astype(np.uint16) * (255 - alpha16)
                + 127
            ) // 255
            combined = combined.astype(np.uint8)
            outside = ~alpha_support
            outside_changes = int(
                np.count_nonzero(np.any(combined[outside] != background[outside], axis=1))
            )
            if outside_changes != 0:
                raise AgentCadReplacementVisualReviewError(
                    "replacement_visual_outside_alpha_changed"
                )
            Image.fromarray(combined, mode="RGB").save(combined_path)
            exact_core = exact_mask > 0
            review_frames.append(
                {
                    "frame_index": index,
                    "camera_id": camera_id,
                    "path": str(combined_path.resolve()),
                    "size_bytes": combined_path.stat().st_size,
                    "sha256": _sha256(combined_path),
                    "background": _file_record(background_path),
                    "exact_repair_mask": _file_record(exact_mask_path),
                    "camera_reference_stage": _file_record(
                        camera_stage, root=task_root
                    ),
                    "replacement_layer": _file_record(layer_path, root=task_root),
                    "replacement_alpha_pixel_count": int(np.count_nonzero(alpha_support)),
                    "exact_repair_pixel_count": int(np.count_nonzero(exact_core)),
                    "exact_repair_pixels_occluded_by_replacement": int(
                        np.count_nonzero(exact_core & alpha_support)
                    ),
                    "outside_replacement_alpha_pixel_count": int(
                        np.count_nonzero(outside)
                    ),
                    "outside_replacement_alpha_changed_pixels": outside_changes,
                }
            )
            camera_ids.add(camera_id)
        task_results.append(
            {
                "task_id": task_id,
                "asset_id": composition["asset_id"],
                "physical_camera_count": camera_count,
                "frames": review_frames,
                "all_camera_bindings_exact": True,
                "outside_replacement_alpha_changed_pixels_total": 0,
                "outside_replacement_alpha_invariance_proven": True,
                "replacement_pose_and_occlusion_human_review": "pending",
                "native_simulator_import_qualified": False,
                "contact_support_and_joint_physics_qualified": False,
            }
        )
        input_records.append(
            {
                "task_id": task_id,
                "dual_target_inputs": {
                    **_file_record(dual_path),
                    "receipt_digest": dual["receipt_digest"],
                },
                "protected_artifixer3d_final_composite": {
                    **_file_record(final_path),
                    "receipt_digest": final["receipt_digest"],
                },
                "agent_cad_visual_composition": {
                    **_file_record(composition_path),
                    "receipt_digest": composition["receipt_digest"],
                },
                "agent_cad_visual_usd": _file_record(candidate_usd),
                "review_trajectory": _file_record(trajectory_path),
            }
        )
        seen_tasks.add(task_id)

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "replacement_visual_review_materialized_pending_human_and_native_review",
        "publisher_scene_id": scene_id,
        "pipeline_mode": (
            "protected_paired_target_artifixer3d_plus_agent_authored_cad_visual_review"
        ),
        "replacement_object_count": len(task_results),
        "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
        "inputs": input_records,
        "renderer": {
            **renderer_record,
            "plugin": renderer_plugin,
            "camera_adapter": "centered_square_pixel_undistorted_opencv_to_usd_v1",
            "horizontal_aperture_mm": HORIZONTAL_APERTURE_MM,
        },
        "implementation": {
            "module_source": _file_record(Path(__file__).resolve()),
            "geometry_operations": "reference_existing_candidate_only",
            "image_operation": (
                "integer_alpha_composite_over_exact_support_protected_background"
            ),
        },
        "tasks": task_results,
        "all_camera_bindings_exact": True,
        "outside_replacement_alpha_changed_pixels_total": 0,
        "outside_replacement_alpha_invariance_proven": True,
        "agent_authored_candidate_geometry_referenced": True,
        "deterministic_geometry_generator_used": False,
        "replacement_pose_and_occlusion_human_review": "pending",
        "appearance_repair_qualified": False,
        "native_simulator_import_qualified": False,
        "contact_support_and_joint_physics_qualified": False,
        "simready_or_policy_gate_unlocked": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "claim_boundary": (
            "deterministic_calibrated_visual_composition_of_agent_authored_candidate_"
            "over_exact_support_protected_reconstructed_background_"
            "pending_human_pose_occlusion_and_native_simulator_review_not_capture_"
            "collision_policy_or_physical_evidence"
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output / f"{SCHEMA_VERSION}.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return {**receipt, "receipt_path": str(receipt_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-input", action="append", required=True)
    parser.add_argument("--final-composite", action="append", required=True)
    parser.add_argument("--visual-composition", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--renderer-executable", default=DEFAULT_RENDERER)
    parser.add_argument("--renderer-plugin", default=DEFAULT_RENDERER_PLUGIN)
    args = parser.parse_args(argv)
    result = materialize_agent_cad_replacement_visual_review(
        dual_input_receipt_paths=args.dual_input,
        final_composite_receipt_paths=args.final_composite,
        visual_composition_receipt_paths=args.visual_composition,
        output_root=args.output_root,
        renderer_executable=args.renderer_executable,
        renderer_plugin=args.renderer_plugin,
    )
    print(
        canonical_json(
            {
                "status": result["status"],
                "receipt_path": result["receipt_path"],
                "receipt_digest": result["receipt_digest"],
                "display_paths": [
                    frame["path"]
                    for task in result["tasks"]
                    for frame in task["frames"]
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AgentCadReplacementVisualReviewError",
    "SCHEMA_VERSION",
    "materialize_agent_cad_replacement_visual_review",
]
