"""Produce retained rigid-site simulator readings for the existing qualifier.

Beta: prepared scene -> paired site-task evidence (AUT-06). This is a local,
read-only extraction from a retained native-engine snapshot and calibrated
rendered RGB-D. It does not create independent site measurements, review
thresholds, qualify a scene, or authorize execution. The snapshot and render
must be produced by the admitted runtime; merely constructing these documents
does not prove that runtime ran.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .site_task_fidelity_profile_templates import measurement_coverage, template
from .site_task_simulation_qualification import BINDING_FIELDS, DIMENSIONS

RECIPE_SCHEMA = "blueprint.site_task_simulator_measurement_recipe.v1"
SNAPSHOT_SCHEMA = "blueprint.site_task_runtime_metrology_snapshot.v1"
SAMPLES_SCHEMA = "blueprint.site_task_measurement_samples.v1"
METHOD = "retained_rigid_engine_and_rectified_rgbd.v1"
MAX_SOURCE_BYTES = 128 * 1024 * 1024
_SHA = re.compile(r"sha256:[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_OPERATIONS = {
    "scale": {"point_distance", "horizontal_distance"},
    "registration": {"floor_height"},
    "collision": {"collision_ray_clearance"},
    "task_state": {"collision_extent"},
    "observation": {"rgbd_length"},
    "task_dynamics": {"rigid_body_mass"},
}


class SimulatorMeasurementError(ValueError):
    """Sanitized extraction refusal; source/provider content is never printed."""


def _require(ok: Any, code: str) -> None:
    if not ok:
        raise SimulatorMeasurementError("site_task_simulator_measurement_" + code)


def _vector(value: Any, size: int = 3) -> np.ndarray:
    _require(isinstance(value, list) and len(value) == size
             and all(type(x) in (float, int) and math.isfinite(x) for x in value), "numeric_source_invalid")
    return np.asarray(value, dtype=float)


def _source(root: Path, record: Any) -> tuple[Path, bytes]:
    _require(isinstance(record, Mapping) and set(record) == {"path", "sha256", "size_bytes"}, "source_record_invalid")
    name = record["path"]
    _require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name
             and not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
             and str(PurePosixPath(name)) == name and _SHA.fullmatch(str(record["sha256"]))
             and type(record["size_bytes"]) is int and 0 < record["size_bytes"] <= MAX_SOURCE_BYTES,
             "source_record_invalid")
    path = root / name
    _require(path.resolve().is_relative_to(root) and path.is_file()
             and not any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)),
             "source_path_invalid")
    from .team_robot_model_delivery import _anchored_directory_fd, _walk_directory_fd
    root_fd = parent_fd = None
    try:
        root_fd = _anchored_directory_fd(root, create=False)
        parts = PurePosixPath(name).parts
        parent_fd = _walk_directory_fd(root_fd, parts[:-1], create=False)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        with os.fdopen(fd, "rb") as stream:
            status = os.fstat(stream.fileno())
            _require(stat.S_ISREG(status.st_mode) and status.st_size == record["size_bytes"], "source_bytes_changed")
            data = stream.read(record["size_bytes"] + 1)
    except OSError:
        raise SimulatorMeasurementError("site_task_simulator_measurement_source_unavailable") from None
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
        if root_fd is not None:
            os.close(root_fd)
    _require(len(data) == record["size_bytes"]
             and "sha256:" + hashlib.sha256(data).hexdigest() == record["sha256"], "source_bytes_changed")
    return path, data


def _json(data: bytes) -> dict[str, Any]:
    try:
        result = json.loads(data)
    except (ValueError, UnicodeError):
        raise SimulatorMeasurementError("site_task_simulator_measurement_source_json_invalid") from None
    _require(isinstance(result, dict), "source_json_invalid")
    return result


def _rgbd_length(data: bytes, reading: Mapping[str, Any], camera: Mapping[str, Any]) -> float:
    import io
    import zipfile

    _require(camera.get("model") == "rectified_pinhole" and camera.get("depth_quantity") == "optical_axis_m"
             and all(_SHA.fullmatch(str(camera.get(k))) for k in ("renderer_digest", "camera_pose_digest")),
             "render_calibration_invalid")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            _require({row.filename for row in archive.infolist()} == {"rgb.npy", "depth_m.npy", "intrinsics.npy"}
                     and len(archive.infolist()) == 3 and
                     sum(row.file_size for row in archive.infolist()) <= MAX_SOURCE_BYTES,
                     "render_arrays_invalid")
            for row in archive.infolist():
                with archive.open(row) as stream:
                    version = np.lib.format.read_magic(stream)
                    _require(version in {(1, 0), (2, 0)}, "render_arrays_invalid")
                    header_reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                     else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = header_reader(stream, max_header_size=4096)
                    _require(1 <= len(shape) <= 3 and not dtype.hasobject
                             and 0 < math.prod(shape) * dtype.itemsize <= row.file_size,
                             "render_arrays_invalid")
        with np.load(io.BytesIO(data), allow_pickle=False) as frame:
            _require(set(frame.files) == {"rgb", "depth_m", "intrinsics"}, "render_arrays_invalid")
            rgb, depth, k = frame["rgb"], frame["depth_m"], frame["intrinsics"]
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        raise SimulatorMeasurementError("site_task_simulator_measurement_render_arrays_invalid") from None
    _require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
             and depth.ndim == 2 and depth.shape == rgb.shape[:2]
             and np.issubdtype(depth.dtype, np.floating) and k.shape == (3, 3)
             and np.isfinite(k).all() and np.isfinite(depth).all()
             and k[0, 0] > 0 and k[1, 1] > 0
             and np.allclose(k[2], [0, 0, 1]) and k[0, 1] == 0 and k[1, 0] == 0,
             "render_arrays_invalid")
    pixels = reading.get("pixels")
    _require(isinstance(pixels, list) and len(pixels) == 2, "render_endpoints_invalid")
    points = []
    for pixel in pixels:
        _require(isinstance(pixel, list) and len(pixel) == 2 and all(type(x) is int for x in pixel),
                 "render_endpoints_invalid")
        u, v = pixel
        _require(0 <= v < depth.shape[0] and 0 <= u < depth.shape[1] and depth[v, u] > 0,
                 "render_endpoint_depth_missing")
        z = float(depth[v, u])
        points.append(np.array([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z]))
    return float(np.linalg.norm(points[0] - points[1]))


def _extract(reading: Mapping[str, Any], snapshot: Mapping[str, Any], *, renders: Mapping[str, bytes]) -> tuple[float, str]:
    op, name = reading.get("operation"), reading.get("name")
    if op in {"point_distance", "horizontal_distance"}:
        points = snapshot.get("geometry_points_m", {})
        _require(isinstance(reading.get("points"), list) and len(reading["points"]) == 2
                 and all(p in points for p in reading["points"]), "geometry_anchor_missing")
        a, b = (_vector(points[p]) for p in reading["points"])
        delta = a - b
        if op == "horizontal_distance":
            normal = _vector(snapshot.get("registered_floor_plane"), 4)[:3]
            _require(abs(float(np.linalg.norm(normal)) - 1) <= 1e-6, "floor_plane_invalid")
            delta -= float(np.dot(delta, normal)) * normal
        return float(np.linalg.norm(delta)), "authored_scene_geometry"
    if op == "floor_height":
        point = snapshot.get("geometry_points_m", {}).get(name)
        plane = _vector(snapshot.get("registered_floor_plane"), 4)
        _require(abs(float(np.linalg.norm(plane[:3])) - 1) <= 1e-6, "floor_plane_invalid")
        return float(np.dot(_vector(point), plane[:3]) + plane[3]), "registered_scene_geometry"
    if op == "collision_extent":
        bounds = snapshot.get("collision_bounds_m", {}).get(name)
        _require(isinstance(bounds, Mapping) and type(reading.get("axis")) is int
                 and reading["axis"] in (0, 1, 2), "collision_bounds_missing")
        low, high = _vector(bounds.get("min")), _vector(bounds.get("max"))
        _require(bool(np.all(high > low)), "collision_bounds_invalid")
        return float((high - low)[reading["axis"]]), "engine_collision_geometry"
    if op == "collision_ray_clearance":
        ray = snapshot.get("collision_rays", {}).get(name)
        _require(isinstance(ray, Mapping) and ray.get("query") == "nearest_scene_collision"
                 and ray.get("hit") is True, "collision_raycast_missing")
        return float(np.linalg.norm(_vector(ray.get("hit_point_m")) - _vector(ray.get("origin_m")))), "engine_collision_query"
    if op == "rigid_body_mass":
        mass = snapshot.get("rigid_body_masses_kg", {}).get(name)
        _require(type(mass) in (int, float) and math.isfinite(mass) and mass > 0, "rigid_body_mass_missing")
        return float(mass), "engine_rigid_body_attribute"
    _require(op == "rgbd_length" and name in renders, "reading_operation_invalid")
    camera = snapshot.get("rendered_rgbd", {}).get(name)
    _require(isinstance(camera, Mapping), "render_calibration_missing")
    return _rgbd_length(renders[name], reading, camera), "calibrated_rendered_rgbd"


def produce_measurements(recipe: Mapping[str, Any], *, evidence_root: Path) -> dict[str, Any]:
    """Calculate frozen samples; never accept independent site readings as inputs.

    ``snapshot`` retains native scene/engine attributes and exact runtime/model
    identities. ``rendered_rgbd`` records name lossless NPZs containing RGB,
    optical-axis depth in metres, and the intrinsics used for that render.
    The operator/reviewer must still qualify the provenance and methods.
    """
    root = Path(evidence_root).resolve()
    _require(recipe.get("schema_version") == RECIPE_SCHEMA and recipe.get("family") == "rigid_pick_place",
             "recipe_or_family_invalid")
    binding = recipe.get("binding")
    _require(isinstance(binding, Mapping) and set(binding) == BINDING_FIELDS
             and all(isinstance(v, str) and v for v in binding.values())
             and all(_SHA.fullmatch(binding[k]) for k in BINDING_FIELDS if "digest" in k), "binding_invalid")
    producer = recipe.get("producer_id")
    _require(isinstance(producer, str) and _ID.fullmatch(producer), "producer_invalid")
    _, raw = _source(root, recipe.get("snapshot"))
    snapshot = _json(raw)
    _require(snapshot.get("schema_version") == SNAPSHOT_SCHEMA and snapshot.get("binding") == binding
             and snapshot.get("source_kind") == "native_engine_readback"
             and isinstance(snapshot.get("engine_build"), str) and snapshot["engine_build"]
             and _SHA.fullmatch(str(snapshot.get("runtime_digest")))
             and _SHA.fullmatch(str(snapshot.get("snapshot_digest")))
             and snapshot["snapshot_digest"] == digest(snapshot, digest_field="snapshot_digest"), "snapshot_binding_invalid")
    from .site_task_simulation_qualification import _time
    _time(snapshot.get("observed_at_iso"))
    _, scene_raw = _source(root, snapshot.get("scene_artifact"))
    scene = _json(scene_raw)
    _require(scene.get("scene_revision_digest") == binding["scene_revision_digest"], "scene_binding_invalid")
    _source(root, snapshot.get("robot_model_artifact"))
    _require(snapshot["robot_model_artifact"]["sha256"] == binding["robot_model_digest"], "robot_model_binding_invalid")
    renders, sources = {}, [dict(recipe["snapshot"]), dict(snapshot["scene_artifact"]),
                           dict(snapshot["robot_model_artifact"])]
    for name, camera in snapshot.get("rendered_rgbd", {}).items():
        _require(isinstance(name, str) and _ID.fullmatch(name) and isinstance(camera, Mapping), "render_record_invalid")
        _, data = _source(root, camera.get("artifact"))
        renders[name] = data
        sources.append(dict(camera["artifact"]))
    calibration_ids = snapshot.get("calibration_sample_ids", [])
    _require(isinstance(calibration_ids, list) and all(isinstance(x, str) for x in calibration_ids), "calibration_ids_invalid")
    readings = recipe.get("readings")
    _require(isinstance(readings, list) and 1 <= len(readings) <= 1000, "readings_invalid")
    frozen = template("rigid_pick_place")
    checks = {row["id"]: {k: row[k] for k in ("id", "dimension", "unit")} | {"samples": []}
              for row in frozen["checks"]}
    allowed = {row["id"]: set(row["sample_ids"]) for row in frozen["checks"]}
    extractions, seen = [], set()
    for reading in readings:
        _require(isinstance(reading, Mapping), "reading_invalid")
        check_id, sample_id = reading.get("check_id"), reading.get("sample_id")
        _require(check_id in checks and sample_id in allowed[check_id]
                 and (check_id, sample_id) not in seen, "reading_binding_invalid")
        _require(sample_id not in calibration_ids, "calibration_reading_reused")
        check = checks[check_id]
        _require(reading.get("operation") in _OPERATIONS[check["dimension"]], "reading_provenance_invalid")
        value, provenance = _extract(reading, snapshot, renders=renders)
        _require(math.isfinite(value) and value >= 0, "reading_value_invalid")
        seen.add((check_id, sample_id))
        check["samples"].append({"sample_id": sample_id, "value": value})
        extractions.append({"check_id": check_id, "sample_id": sample_id, "provenance": provenance,
                            "selector": dict(reading), "snapshot_digest": snapshot["snapshot_digest"]})
    document = {"schema_version": SAMPLES_SCHEMA, "binding": dict(binding), "producer_id": producer,
                "observed_at_iso": snapshot["observed_at_iso"], "checks": list(checks.values()),
                "extraction": {"method_id": METHOD, "implementation_sha256": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "recipe_digest": digest(recipe), "runtime_digest": snapshot["runtime_digest"],
                               "engine_build": snapshot["engine_build"], "source_artifacts": sources,
                               "samples": extractions, "calibration_sample_ids": calibration_ids},
                "physical_performance_claimed": False, "adp_qualified_evidence_eligible": False}
    coverage = measurement_coverage(frozen, document)
    _require(coverage["complete"] and {c["dimension"] for c in checks.values()} == DIMENSIONS, "coverage_incomplete")
    return document


def retain_measurements(path: Path, document: Mapping[str, Any]) -> None:
    """Exclusive, fsynced publication with byte-identical restart adoption."""
    raw = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        _require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw, "output_conflict")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".metrology-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw, "output_conflict")
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def produce_from_operator_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Reopen the operator-pinned recipe and authenticate its original sources."""
    settings = profile.get("simulator_measurement_producer")
    _require(isinstance(settings, Mapping) and set(settings) == {
        "recipe_path", "recipe_digest", "evidence_root", "producer_id"}, "operator_configuration_invalid")
    recipe_path = Path(str(settings["recipe_path"]))
    root = Path(str(settings["evidence_root"]))
    _require(recipe_path.is_absolute() and root.is_absolute() and recipe_path.is_file()
             and not recipe_path.is_symlink() and recipe_path.stat().st_size <= MAX_SOURCE_BYTES,
             "operator_configuration_invalid")
    recipe = _json(recipe_path.read_bytes())
    _require(digest(recipe) == settings["recipe_digest"] and recipe.get("producer_id") == settings["producer_id"],
             "operator_recipe_changed")
    return produce_measurements(recipe, evidence_root=root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = produce_measurements(_json(args.recipe.read_bytes()), evidence_root=args.evidence_root)
    retain_measurements(args.out, result)
    print(json.dumps({"path": str(args.out.resolve()), "measurement_digest": digest(result),
                      "evidence_scope": "simulator_readings_not_site_qualification"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
