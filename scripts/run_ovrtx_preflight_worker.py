#!/usr/bin/env python3
"""Headless ovrtx 0.4 preflight worker for an isolated Linux/RTX environment.

The core pipeline launches this script as a subprocess. It deliberately has no
Blueprint imports so prerelease NVIDIA dependencies remain outside the core
environment. Network access is neither needed nor used: every USD dependency
must already resolve locally.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from typing import Any


AOV_BY_KIND = {
    "rgb": "LdrColor",
    "depth": "DepthSD",
    "normal": "Normal",
    "semantic_segmentation": "SemanticSegmentation",
    "semantic_id_map": "SemanticIdMap",
}

RTX_RENDER_PRODUCT_API_SCHEMAS = (
    "OmniRtxSettingsCommonAdvancedAPI_1",
    "OmniRtxSettingsRtAdvancedAPI_1",
    "OmniRtxSettingsPtAdvancedAPI_1",
)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _usd_asset_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("@", "\\@").replace("#", "\\#")


def _camera_layer(scene: Path, config: dict[str, Any]) -> str:
    width = int(config.get("width", 640))
    height = int(config.get("height", 480))
    camera_path = str(config.get("camera_prim_path") or "/BlueprintPreflight/Camera")
    if not camera_path.startswith("/") or "/" not in camera_path[1:]:
        raise ValueError("camera_prim_path must be an absolute child prim path")
    camera_parent, camera_name = camera_path.rsplit("/", 1)
    parent_components = [part for part in camera_parent.split("/") if part]
    if len(parent_components) != 1:
        raise ValueError("worker-authored camera currently requires a one-level parent path")
    position = config.get("camera_position", [2.5, -2.5, 1.8])
    rotation = config.get("camera_rotate_xyz_degrees", [68.0, 0.0, 45.0])
    transform = config.get("camera_transform_matrix_usd")
    if transform is not None:
        if (
            not isinstance(transform, list)
            or len(transform) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in transform)
        ):
            raise ValueError("camera_transform_matrix_usd must be a 4x4 row-major matrix")
        transform_rows = ",\n            ".join(
            "(" + ", ".join(str(float(value)) for value in row) + ")"
            for row in transform
        )
        transform_usda = (
            "        matrix4d xformOp:transform = (\n"
            f"            {transform_rows}\n"
            "        )\n"
            '        uniform token[] xformOpOrder = ["xformOp:transform"]'
        )
    else:
        transform_usda = (
            "        double3 xformOp:translate = "
            f"({float(position[0])}, {float(position[1])}, {float(position[2])})\n"
            "        float3 xformOp:rotateXYZ = "
            f"({float(rotation[0])}, {float(rotation[1])}, {float(rotation[2])})\n"
            '        uniform token[] xformOpOrder = '
            '["xformOp:translate", "xformOp:rotateXYZ"]'
        )
    focal = float(config.get("focal_length_mm", 24.0))
    aperture = float(config.get("horizontal_aperture_mm", 20.955))
    vertical_aperture = float(
        config.get("vertical_aperture_mm", aperture * height / width)
    )
    horizontal_offset = float(config.get("horizontal_aperture_offset_mm", 0.0))
    vertical_offset = float(config.get("vertical_aperture_offset_mm", 0.0))
    clipping = config.get("clipping_range", [0.01, 10000.0])
    render_mode = str(config.get("render_mode", "RealTimePathTracing"))
    if render_mode not in {"RaytracedLighting", "RealTimePathTracing", "PathTracing"}:
        raise ValueError("render_mode is not supported")
    ordered = [AOV_BY_KIND[kind] for kind in AOV_BY_KIND]
    ordered_vars_usda = ", ".join(f"<{name}>" for name in ordered)
    vars_usda = "\n".join(
        f'''        def RenderVar "{name}" {{ string sourceName = "{name}" }}''' for name in ordered
    )
    return f'''#usda 1.0
(
    subLayers = [@{_usd_asset_path(scene)}@]
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{parent_components[0]}"
{{
    def Camera "{camera_name}"
    {{
        float focalLength = {focal}
        float horizontalAperture = {aperture}
        float verticalAperture = {vertical_aperture}
        float horizontalApertureOffset = {horizontal_offset}
        float verticalApertureOffset = {vertical_offset}
        float2 clippingRange = ({float(clipping[0])}, {float(clipping[1])})
{transform_usda}
    }}
}}

def "Render"
{{
    def RenderProduct "BlueprintCamera" (
        prepend apiSchemas = ["{RTX_RENDER_PRODUCT_API_SCHEMAS[0]}", "{RTX_RENDER_PRODUCT_API_SCHEMAS[1]}", "{RTX_RENDER_PRODUCT_API_SCHEMAS[2]}"]
    )
    {{
        rel camera = <{camera_path}>
        int2 resolution = ({width}, {height})
        token omni:rtx:rendermode = "{render_mode}"
        rel orderedVars = [{ordered_vars_usda}]
{vars_usda}
    }}
}}
'''


def _runtime_identity() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in ("ovrtx", "ovstage", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    gpu: dict[str, Any] = {}
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu = {
            "name": str(pynvml.nvmlDeviceGetName(handle)),
            "uuid": str(pynvml.nvmlDeviceGetUUID(handle)),
            "memory_total_bytes": int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
            "memory_used_bytes_at_identity_query": int(pynvml.nvmlDeviceGetMemoryInfo(handle).used),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        gpu = {"query_error": type(exc).__name__}
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_version": os.environ.get("CUDA_VERSION"),
        "driver_version": os.environ.get("NVIDIA_DRIVER_VERSION"),
        "gpu_identity": gpu,
        "library_versions": versions,
    }


def _gpu_memory_used_bytes() -> int | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
    except Exception:  # noqa: BLE001 - optional diagnostic sample
        return None


def _map_array(frame: Any, name: str, ovrtx: Any, np: Any) -> Any:
    mapped = frame.render_vars[name].map(device=ovrtx.Device.CPU)
    try:
        return np.from_dlpack(mapped).copy()
    finally:
        try:
            mapped.unmap()
        except AttributeError:
            pass


def _semantic_labels(raw: Any, np: Any) -> list[str]:
    data = np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
    if data.size < 4:
        return []
    count = int.from_bytes(data[-4:].tobytes(), "little")
    entry_size = 24
    if count < 0 or count * entry_size > data.size - 4:
        return []
    labels: list[str] = []
    for index in range(count):
        base = index * entry_size
        length = int.from_bytes(data[base + 16 : base + 20].tobytes(), "little")
        offset = int.from_bytes(data[base + 20 : base + 24].tobytes(), "little")
        if length > 0 and offset + length <= data.size:
            labels.append(
                data[offset : offset + length].tobytes().decode("utf-8", "replace").rstrip("\0")
            )
    return sorted(set(filter(None, labels)))


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": "passed" if passed else "failed", "details": details}


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    required_checks = set(config.get("_blueprint_required_checks", []))
    import numpy as np
    import ovrtx
    import ovstage

    installed_version = importlib.metadata.version("ovrtx")
    checks: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    gpu_memory_samples = [_gpu_memory_used_bytes()]
    renderer = None
    stage = None
    try:
        renderer = ovrtx.Renderer(
            ovrtx.RendererConfig(log_file_path=str(args.output_dir / "ovrtx-worker.log"))
        )
        gpu_memory_samples.append(_gpu_memory_used_bytes())
        stage = ovstage.Stage("blueprint.ovrtx.preflight")
        renderer.attach_ovstage(stage)
        ordinal = 1
        camera_modalities = [kind for kind in AOV_BY_KIND if kind in args.modalities]
        if any(kind in args.modalities for kind in ("lidar", "radar")):
            scene_payload: str | Path = args.input
            render_products = set(config.get("render_product_paths", []))
            if not render_products:
                raise ValueError(
                    "lidar/radar requires render_product_paths authored in the input USD"
                )
            ovstage.population.open_usd(stage, str(scene_payload), ordinal=ordinal)
        else:
            scene_payload = _camera_layer(args.input, config)
            render_products = {"/Render/BlueprintCamera"}
            ovstage.population.open_usd_from_string(stage, scene_payload, ordinal=ordinal)
        stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
        checks.append(_check("usd_scene_load", True, ovstage_attached=True))

        warmup = max(0, int(config.get("warmup_frames", 3)))
        quality_steps = max(1, int(config.get("quality_steps", 1)))
        delta_time = float(config.get("delta_time_seconds", 1.0 / 60.0))
        for _ in range(warmup):
            renderer.step(render_products=render_products, delta_time=delta_time, ordinal=ordinal)
        gpu_memory_samples.append(_gpu_memory_used_bytes())

        if "dynamic_transform_update" in required_checks:
            ordinal += 1
            usd_time = float(config.get("episode_sample_time_seconds", delta_time))
            ovstage.population.apply_usd_time(stage, ordinal, usd_time)
            stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
            checks.append(
                _check("dynamic_transform_update", True, usd_time_seconds=usd_time, ordinal=ordinal)
            )

        products = {}
        for _ in range(quality_steps):
            products = renderer.step(
                render_products=render_products,
                delta_time=0.0 if quality_steps > 1 else delta_time,
                ordinal=ordinal,
            )
        gpu_memory_samples.append(_gpu_memory_used_bytes())
        nonempty = True
        labels: list[str] = []
        for product_name, product in products.items():
            if not product.frames:
                nonempty = False
                continue
            frame = product.frames[0]
            if "/Render/BlueprintCamera" in product_name or camera_modalities:
                for kind in camera_modalities:
                    name = AOV_BY_KIND[kind]
                    if name not in frame.render_vars:
                        nonempty = False
                        continue
                    array = _map_array(frame, name, ovrtx, np)
                    path = args.output_dir / f"{kind}.npy"
                    np.save(path, array, allow_pickle=False)
                    metadata = {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "width": int(config.get("width", 640)),
                        "height": int(config.get("height", 480)),
                        "camera_prim_path": str(
                            config.get("camera_prim_path") or "/BlueprintPreflight/Camera"
                        ),
                        "focal_length_mm": float(config.get("focal_length_mm", 24.0)),
                        "horizontal_aperture_mm": float(
                            config.get("horizontal_aperture_mm", 20.955)
                        ),
                        "vertical_aperture_mm": float(
                            config.get(
                                "vertical_aperture_mm",
                                float(config.get("horizontal_aperture_mm", 20.955))
                                * int(config.get("height", 480))
                                / int(config.get("width", 640)),
                            )
                        ),
                        "render_mode": str(
                            config.get("render_mode", "RealTimePathTracing")
                        ),
                        "quality_steps": quality_steps,
                    }
                    outputs.append({"kind": kind, "path": path.name, "metadata": metadata})
                    nonempty = nonempty and array.size > 0
                    if kind == "semantic_id_map":
                        labels = _semantic_labels(array, np)
            if "lidar" in args.modalities and "PointCloud" in frame.render_vars:
                with frame.render_vars["PointCloud"].map(device=ovrtx.Device.CPU) as pointcloud:
                    counts = np.from_dlpack(pointcloud["Counts"])
                    count = int(counts[0])
                    coordinates = np.from_dlpack(pointcloud["Coordinates"])[:, :count].T.copy()
                path = args.output_dir / "lidar.npy"
                np.save(path, coordinates, allow_pickle=False)
                outputs.append(
                    {"kind": "lidar", "path": path.name, "metadata": {"point_count": count}}
                )
                checks.append(_check("lidar_structured_output", count > 0, point_count=count))
                nonempty = nonempty and count > 0
            if "radar" in args.modalities and "PointCloud" in frame.render_vars:
                with frame.render_vars["PointCloud"].map(device=ovrtx.Device.CPU) as pointcloud:
                    counts = np.from_dlpack(pointcloud["Counts"])
                    count = int(counts[0])
                    coordinates = np.from_dlpack(pointcloud["Coordinates"])[:, :count].T.copy()
                path = args.output_dir / "radar.npy"
                np.save(path, coordinates, allow_pickle=False)
                outputs.append(
                    {"kind": "radar", "path": path.name, "metadata": {"detection_count": count}}
                )
                checks.append(_check("radar_structured_output", count > 0, detection_count=count))
                nonempty = nonempty and count > 0

        checks.append(
            _check("requested_sensor_outputs_nonempty", nonempty, output_count=len(outputs))
        )
        metadata_complete = all(bool(item["metadata"]) for item in outputs)
        checks.append(_check("sensor_metadata_complete", metadata_complete))
        if "semantic_segmentation" in args.modalities:
            id_map_present = any(item["kind"] == "semantic_id_map" for item in outputs)
            checks.append(
                _check("semantic_id_map", id_map_present and bool(labels), labels=labels[:100])
            )
        source_text = ""
        if args.input.suffix.lower() in {".usd", ".usda"}:
            source_text = args.input.read_text(encoding="utf-8", errors="ignore")
        if (
            "particlefield_gaussian_splat_render" in required_checks
            or "ParticleField3DGaussianSplat" in source_text
        ):
            rgb_path = args.output_dir / "rgb.npy"
            splat_pass = False
            if rgb_path.is_file():
                rgb = np.load(rgb_path, allow_pickle=False)
                splat_pass = bool(rgb.size and float(np.std(rgb)) > 0.0)
            checks.append(_check("particlefield_gaussian_splat_render", splat_pass))
        if "robot_and_target_visibility" in required_checks:
            expected = [str(value) for value in config.get("expected_visible_semantic_labels", [])]
            visible = bool(expected) and all(
                any(value in label for label in labels) for value in expected
            )
            checks.append(
                _check(
                    "robot_and_target_visibility", visible, expected=expected, observed=labels[:100]
                )
            )
    finally:
        if renderer is not None and stage is not None:
            renderer.detach_ovstage()
        if stage is not None:
            stage.destroy()
        if renderer is not None:
            renderer.destroy()

    passed = bool(checks) and all(item["status"] == "passed" for item in checks)
    report = {
        "component_name": "ovrtx",
        "component_version": installed_version,
        "source_revision": args.source_revision,
        "configuration_sha256": _sha256_json(config),
        "runtime": _runtime_identity(),
        "checks": checks,
        "outputs": outputs,
        "metrics": {
            "worker_wall_seconds": time.monotonic() - started,
            "mode": args.mode,
            "gpu_memory_baseline_bytes": next(
                (value for value in gpu_memory_samples if value is not None), None
            ),
            "gpu_memory_peak_observed_bytes": max(
                (value for value in gpu_memory_samples if value is not None),
                default=None,
            ),
            "gpu_memory_samples_bytes": [
                value for value in gpu_memory_samples if value is not None
            ],
        },
        "failure_classes_checked": [
            "usd_scene_load",
            "empty_sensor_output",
            "sensor_metadata_loss",
            "semantic_id_map_loss",
        ],
        "required_sensor_metadata_preserved": passed,
    }
    return report, 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("cold", "warm"), required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--modality", dest="modalities", action="append", default=[])
    args = parser.parse_args()
    args.modalities = args.modalities or [
        "rgb",
        "depth",
        "semantic_segmentation",
        "semantic_id_map",
    ]
    try:
        report, status = _run(args)
    except Exception as exc:  # noqa: BLE001 - worker must preserve failure evidence
        report = {
            "component_name": "ovrtx",
            "component_version": "unavailable",
            "source_revision": args.source_revision,
            "configuration_sha256": _sha256_json(
                json.loads(args.config.read_text(encoding="utf-8"))
            ),
            "runtime": _runtime_identity(),
            "checks": [
                {
                    "name": "worker_execution",
                    "status": "failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ],
            "outputs": [],
            "failure_classes_checked": [],
            "required_sensor_metadata_preserved": False,
        }
        status = 2
    _write_json(args.output, report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
