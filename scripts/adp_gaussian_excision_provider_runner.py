#!/usr/bin/env python3
"""Run pinned FlashSplat contribution accumulation on frozen calibration views."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image


RESULT_SCHEMA = "adp009b_gaussian_excision_result.v1"
CONTRIBUTION_SCHEMA = "adp009b_gaussian_excision_contribution_evidence.v1"
CLASS_ORDER = ("protected", "target_core", "uncertain")
SOURCE_REPOSITORY = "https://github.com/florinshen/FlashSplat"
SOURCE_COMMIT = "3e3b14786333bf0163ba1b8541e86a3765112d7d"
RASTERIZER_REPOSITORY = "https://github.com/florinshen/flashsplat-rasterization"
RASTERIZER_COMMIT = "189c483ffa33dd6d5661343ce496df0c6eb80a0c"
RUNTIME_IMPORT_PREFLIGHT_SCHEMA = (
    "adp009b_gaussian_excision_runtime_import_preflight.v1"
)
RUNTIME_IMPORT_MODULES = (
    "numpy",
    "PIL",
    "plyfile",
    "cv2",
    "torch",
    "diff_gaussian_rasterization",
    "flashsplat_rasterization",
    "simple_knn._C",
    "gaussian_renderer",
    "scene.cameras",
    "scene.gaussian_model",
)
SAFE_FAILURE_CODE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_digest(value: Mapping[str, Any], *, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("gaussian_excision_json_not_object")
    return value


def runtime_import_preflight(
    *,
    source_dir: Path,
    importer: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    """Probe the complete pinned execution import set without short-circuiting."""

    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    rows = []
    for module_name in RUNTIME_IMPORT_MODULES:
        try:
            importer(module_name)
            rows.append(
                {
                    "module": module_name,
                    "status": "imported",
                    "error_type": None,
                    "missing_module_name": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - retain every import failure
            rows.append(
                {
                    "module": module_name,
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "missing_module_name": (
                        str(exc.name)
                        if isinstance(exc, ModuleNotFoundError) and exc.name
                        else None
                    ),
                }
            )
    blocked = [row for row in rows if row["status"] != "imported"]
    result = {
        "schema_version": RUNTIME_IMPORT_PREFLIGHT_SCHEMA,
        "status": "passed" if not blocked else "blocked",
        "required_modules": list(RUNTIME_IMPORT_MODULES),
        "imports": rows,
        "failed_import_count": len(blocked),
        "failed_modules": [row["module"] for row in blocked],
        "missing_module_names": sorted(
            {
                str(row["missing_module_name"])
                for row in blocked
                if row["missing_module_name"]
            }
        ),
        "all_imports_attempted": len(rows) == len(RUNTIME_IMPORT_MODULES),
        "blockers": (
            []
            if not blocked
            else ["gaussian_excision_runtime_import_closure_incomplete"]
        ),
    }
    result["preflight_digest"] = _canonical_digest(
        result, field="preflight_digest"
    )
    return result


def camera_parameters(camera: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a frozen camera-to-world pose to canonical FlashSplat inputs.

    Method-input packets historically named the already calibrated pose
    ``T_world_camera_provider_frame`` while newer freezes normalize the same
    pose to ``T_world_camera_opencv``.  Accept either contract spelling, but
    fail closed if a caller supplies conflicting aliases.
    """

    opencv_transform = camera.get("T_world_camera_opencv")
    provider_transform = camera.get("T_world_camera_provider_frame")
    if opencv_transform is None and provider_transform is None:
        raise ValueError("gaussian_excision_camera_transform_missing")
    if opencv_transform is not None and provider_transform is not None:
        opencv_array = np.asarray(opencv_transform, dtype=np.float64)
        provider_array = np.asarray(provider_transform, dtype=np.float64)
        if (
            opencv_array.shape != provider_array.shape
            or not np.array_equal(opencv_array, provider_array)
        ):
            raise ValueError("gaussian_excision_camera_transform_alias_conflict")
    transform = np.asarray(
        opencv_transform if opencv_transform is not None else provider_transform,
        dtype=np.float64,
    )
    intrinsics = camera.get("intrinsics")
    if transform.shape != (4, 4) or not isinstance(intrinsics, Mapping):
        raise ValueError("gaussian_excision_camera_invalid")
    width = int(intrinsics.get("width") or 0)
    height = int(intrinsics.get("height") or 0)
    fx = float(intrinsics.get("fx") or 0.0)
    fy = float(intrinsics.get("fy") or 0.0)
    if (
        width <= 0
        or height <= 0
        or fx <= 0.0
        or fy <= 0.0
        or float(intrinsics.get("cx")) != width / 2.0
        or float(intrinsics.get("cy")) != height / 2.0
    ):
        raise ValueError("gaussian_excision_camera_intrinsics_unsupported")
    world_to_camera = np.linalg.inv(transform)
    return {
        "R": world_to_camera[:3, :3].T,
        "T": world_to_camera[:3, 3],
        "FoVx": 2.0 * math.atan(width / (2.0 * fx)),
        "FoVy": 2.0 * math.atan(height / (2.0 * fy)),
        "width": width,
        "height": height,
    }


def _failure_diagnostics(exc: Exception) -> dict[str, Any]:
    """Retain deterministic, non-secret diagnostics for a paid worker failure."""

    message = str(exc)
    return {
        "failure_type": type(exc).__name__,
        "failure_code": message if SAFE_FAILURE_CODE.fullmatch(message) else None,
        "failure_message_sha256": "sha256:"
        + hashlib.sha256(message.encode("utf-8")).hexdigest(),
    }


def load_class_labels(mask_root: Path, camera_id: str) -> np.ndarray:
    """Load the frozen, exhaustive three-zone pixel assignment."""

    masks = []
    for name in CLASS_ORDER:
        path = mask_root / f"{camera_id}.{name}.png"
        with Image.open(path) as image:
            masks.append(np.asarray(image.convert("L"), dtype=np.uint8) >= 128)
    stack = np.stack(masks, axis=0)
    if not np.all(stack.sum(axis=0) == 1):
        raise ValueError("gaussian_excision_mask_zones_not_exhaustive")
    return np.argmax(stack, axis=0).astype(np.float32)


def validated_camera_split(
    camera_split: Mapping[str, Any], cameras_by_id: Mapping[str, Any]
) -> list[str]:
    """Validate a freeze camera partition, including legacy count-less freezes.

    Early production freezes bound the complete calibration and held-out ID
    lists but predated the redundant ``*_count`` fields.  Reopen those exact
    immutable lists while still rejecting any present count that disagrees
    with the ID partition.
    """

    calibration_raw = camera_split.get("calibration_camera_ids")
    heldout_raw = camera_split.get("heldout_camera_ids")
    if (
        not isinstance(calibration_raw, list)
        or not isinstance(heldout_raw, list)
        or any(not isinstance(value, str) or not value for value in calibration_raw)
        or any(not isinstance(value, str) or not value for value in heldout_raw)
    ):
        raise ValueError("gaussian_excision_camera_split_invalid")
    calibration = list(calibration_raw)
    heldout_list = list(heldout_raw)
    heldout = set(heldout_list)
    expected_counts = {
        "camera_count": len(calibration) + len(heldout_list),
        "calibration_camera_count": len(calibration),
        "heldout_camera_count": len(heldout_list),
    }
    for field, expected in expected_counts.items():
        observed = camera_split.get(field)
        if observed is not None and (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed != expected
        ):
            raise ValueError("gaussian_excision_camera_split_invalid")
    if (
        len(calibration) < 2
        or len(set(calibration)) != len(calibration)
        or len(heldout) != len(heldout_list)
        or heldout.intersection(calibration)
        or set(calibration).union(heldout) != set(cameras_by_id)
    ):
        raise ValueError("gaussian_excision_camera_split_invalid")
    return calibration


def _save_render(path: Path, tensor: Any) -> None:
    values = tensor.detach().float().clamp(0.0, 1.0).cpu().numpy()
    image = np.rint(np.moveaxis(values[:3], 0, 2) * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path, compress_level=9)


def execute(*, runtime_dir: Path, source_dir: Path, output_dir: Path) -> dict[str, Any]:
    import torch

    sys.path.insert(0, str(source_dir))
    from gaussian_renderer import flashsplat_render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel

    freeze_path = runtime_dir / "freeze/adp009b_gaussian_excision_audit_freeze.v1.json"
    freeze = _read_json(freeze_path)
    if freeze.get("freeze_digest") != _canonical_digest(freeze, field="freeze_digest"):
        raise ValueError("gaussian_excision_freeze_digest_invalid")
    scene_path = runtime_dir / "input/scene_standard.ply"
    camera_path = runtime_dir / "input/cameras.v1.json"
    if (
        _sha256(scene_path) != freeze["source_standard_splat"]["sha256"]
        or _sha256(camera_path) != freeze["camera_contract"]["sha256"]
    ):
        raise ValueError("gaussian_excision_bound_input_changed")
    cameras_value = json.loads(camera_path.read_text(encoding="utf-8"))
    cameras_by_id = {
        str(row["camera_id"]): row for row in cameras_value if isinstance(row, Mapping)
    }
    camera_split = freeze["camera_split"]
    calibration = validated_camera_split(camera_split, cameras_by_id)
    model = GaussianModel(3)
    model.load_ply(str(scene_path))
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(
        debug=False, compute_cov3D_python=False, convert_SHs_python=False
    )
    mask_root = runtime_dir / "freeze/masks"
    camera_objects = []
    labels = []
    for index, camera_id in enumerate(calibration):
        row = cameras_by_id[camera_id]
        parameters = camera_parameters(row)
        image = torch.zeros(
            (3, parameters["height"], parameters["width"]), dtype=torch.float32
        )
        camera_objects.append(
            Camera(
                colmap_id=index,
                R=parameters["R"],
                T=parameters["T"],
                FoVx=parameters["FoVx"],
                FoVy=parameters["FoVy"],
                image=image,
                gt_alpha_mask=None,
                image_name=camera_id,
                uid=index,
                gt_depth=None,
                data_device="cuda",
            )
        )
        labels.append(load_class_labels(mask_root, camera_id))

    repetition_count = int(freeze["policy"]["deterministic_repetitions"])
    repetition_rows = []
    render_rows = []
    for repetition in range(repetition_count):
        per_view = []
        for camera_id, camera, class_labels in zip(
            calibration, camera_objects, labels, strict=True
        ):
            torch.cuda.synchronize()
            package = flashsplat_render(
                camera,
                model,
                pipeline,
                background,
                gt_mask=torch.from_numpy(class_labels).to("cuda").contiguous(),
                obj_num=len(CLASS_ORDER),
            )
            torch.cuda.synchronize()
            used = package["used_count"][: len(CLASS_ORDER)].detach().cpu().numpy()
            per_view.append(used.astype(np.float32, copy=False))
            if repetition == 0:
                render_path = output_dir / f"calibration_{camera_id}.png"
                _save_render(render_path, package["render"])
                render_rows.append({"camera_id": camera_id, **_record(render_path, output_dir)})
        array = np.stack(per_view, axis=0)
        path = output_dir / f"contribution_repetition_{repetition}.npz"
        np.savez_compressed(path, per_view_class_contribution=array)
        repetition_rows.append(_record(path, output_dir))

    method = {
        **freeze["contribution_method"],
        "released_code_executed": True,
    }
    manifest = {
        "schema_version": CONTRIBUTION_SCHEMA,
        "freeze_digest": freeze["freeze_digest"],
        "class_order": list(CLASS_ORDER),
        "camera_ids": calibration,
        "method": method,
        "repetitions": repetition_rows,
        "calibration_renders": render_rows,
        "heldout_cameras_accessed_for_classification": False,
    }
    manifest["manifest_digest"] = _canonical_digest(manifest, field="manifest_digest")
    manifest_path = output_dir / f"{CONTRIBUTION_SCHEMA}.json"
    manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed",
        "freeze_digest": freeze["freeze_digest"],
        "contribution_manifest": _record(manifest_path, output_dir),
        "contribution_manifest_digest": manifest["manifest_digest"],
        "released_code": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "rasterizer_repository": RASTERIZER_REPOSITORY,
            "rasterizer_commit": RASTERIZER_COMMIT,
            "source_modified": False,
        },
        "released_code_executed": True,
        "used_count_semantics": "front_to_back_transmittance_times_alpha",
        "heldout_cameras_accessed_for_classification": False,
        "provider_zero_required_after_return": True,
        "depth_anything_3_used": False,
        "retry_cap": 0,
        "blockers": [],
        "raw_secret_values_recorded": False,
    }
    result["result_digest"] = _canonical_digest(result, field="result_digest")
    result_path = output_dir / "adp009b_gaussian_excision_result.json"
    result_path.write_text(_canonical_json(result) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args(argv)
    output = Path(arguments.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_dir = Path(arguments.source_dir).resolve()
    import_preflight = runtime_import_preflight(source_dir=source_dir)
    import_preflight_path = (
        output / "adp009b_gaussian_excision_runtime_import_preflight.json"
    )
    import_preflight_path.write_text(
        _canonical_json(import_preflight) + "\n", encoding="utf-8"
    )
    import_preflight_record = _record(import_preflight_path, output)
    if import_preflight["status"] != "passed":
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "blocked",
            "blockers": list(import_preflight["blockers"]),
            "runtime_import_preflight": import_preflight_record,
            "failed_import_count": import_preflight["failed_import_count"],
            "failed_modules": import_preflight["failed_modules"],
            "missing_module_names": import_preflight["missing_module_names"],
            "released_code_executed": False,
            "heldout_cameras_accessed_for_classification": False,
            "provider_zero_required_after_return": True,
            "depth_anything_3_used": False,
            "retry_cap": 0,
            "raw_secret_values_recorded": False,
        }
        result["result_digest"] = _canonical_digest(result, field="result_digest")
        (output / "adp009b_gaussian_excision_result.json").write_text(
            _canonical_json(result) + "\n", encoding="utf-8"
        )
        return 2
    try:
        result = execute(
            runtime_dir=Path(arguments.runtime_dir).resolve(),
            source_dir=source_dir,
            output_dir=output,
        )
        result["runtime_import_preflight"] = import_preflight_record
        result["result_digest"] = _canonical_digest(result, field="result_digest")
        (output / "adp009b_gaussian_excision_result.json").write_text(
            _canonical_json(result) + "\n", encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001 - paid worker must retain typed failure
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "blocked",
            "blockers": [f"gaussian_excision_runtime_failed:{type(exc).__name__}"],
            "runtime_import_preflight": import_preflight_record,
            "missing_module_name": (
                str(exc.name)
                if isinstance(exc, ModuleNotFoundError) and exc.name
                else None
            ),
            "released_code_executed": False,
            "heldout_cameras_accessed_for_classification": False,
            "provider_zero_required_after_return": True,
            "depth_anything_3_used": False,
            "retry_cap": 0,
            "raw_secret_values_recorded": False,
            **_failure_diagnostics(exc),
        }
        result["result_digest"] = _canonical_digest(result, field="result_digest")
        (output / "adp009b_gaussian_excision_result.json").write_text(
            _canonical_json(result) + "\n", encoding="utf-8"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
