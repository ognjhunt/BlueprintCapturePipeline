"""Compile measured source records into the existing family layout contract.

The supported input is an admitted native metric capture plus explicit survey
and robot-placement records. Estimated Atlas geometry, an AABB, or a provider
success cannot supply calibration, target orientation or base registration.
This CPU adapter grants no source, runtime, policy or spending authority.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path

import numpy as np

from .capture_intake import build_capture_admission, validate_capture_intake_envelope
from .camera_geometry_validation import validate_se3_matrix
from .decision_evidence_contracts import canonical_digest
from .task_family_authoring_evidence import _read
from .task_family_native_contracts import _pose

SCHEMA = "website_metric_layout_registration.v1"
_KINDS = ("capture_intake", "calibration", "frame_registration", "target_poses", "robot_registration")
_QUANTITIES = {"capture_intake": "metric_capture_admission", "calibration": "metric_capture_calibration",
               "frame_registration": "source_runtime_registration", "target_poses": "target_pose_measurements",
               "robot_registration": "robot_base_registration"}


def _require(condition, reason):
    if not condition:
        raise ValueError("website_metric_layout_" + reason)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _frame(record, scene):
    """Check the retained registration against measured point pairs, without fitting it to pass."""
    for key in ("source_points_m", "runtime_points"):
        rows = record.get(key)
        _require(isinstance(rows, list) and all(isinstance(row, list) and len(row) == 3
            and all(_number(value) for value in row) for row in rows), "correspondences_invalid")
    points = np.asarray(record["source_points_m"], dtype=float)
    targets = np.asarray(record["runtime_points"], dtype=float)
    unit = scene.get("meters_per_unit")
    tolerance = record.get("maximum_registration_residual_m")
    _require(points.ndim == 2 and points.shape[1:] == (3,) and 3 <= len(points) <= 256
        and targets.shape == points.shape and np.isfinite(points).all() and np.isfinite(targets).all()
        and _number(unit) and unit > 0 and scene.get("up_axis") in {"Y", "Z", "-Y"}
        and _number(tolerance) and tolerance > 0, "correspondences_invalid")
    _require(np.linalg.matrix_rank(points - points.mean(axis=0)) >= 2
        and np.linalg.matrix_rank(targets - targets.mean(axis=0)) >= 2,
             "correspondences_degenerate")
    transform = record.get("source_to_runtime")
    _require(isinstance(transform, list) and len(transform) == 4
        and all(isinstance(row, list) and len(row) == 4 and all(_number(value) for value in row)
                for row in transform), "registration_matrix_invalid")
    matrix = np.asarray(transform, dtype=float)
    _require(matrix.shape == (4, 4) and np.isfinite(matrix).all() and np.array_equal(matrix[3], [0, 0, 0, 1]),
             "registration_matrix_invalid")
    rotation = matrix[:3, :3] * unit
    _require(np.linalg.det(rotation) > 0 and np.allclose(rotation.T @ rotation, np.eye(3), rtol=1e-10, atol=1e-10),
             "registration_scale_or_orientation_invalid")
    projected = points @ matrix[:3, :3].T + matrix[:3, 3]
    residual = float(np.max(np.linalg.norm(projected - targets, axis=1))) * unit
    _require(residual <= tolerance, "registration_residual_exceeded")
    value = {"source_geometry_digest": record["source_geometry_digest"],
        "collision_mesh_digest": scene["collision_mesh_digest"], "runtime_up_axis": scene["up_axis"],
        "runtime_meters_per_unit": unit, "source_to_runtime": matrix.tolist()}
    value["registration_digest"] = canonical_digest(value, digest_field="registration_digest")
    return value


def _records(references):
    _require(isinstance(references, Mapping) and set(references) == set(_KINDS), "source_records_invalid")
    return {kind: _read(references[kind], "metric_layout_" + kind, as_json=True) for kind in _KINDS}


def _explicit_pose(value):
    _pose(value)
    _require(all(_number(number) for key in ("position_world_m", "orientation_xyzw") for number in value[key]),
             "pose_numbers_invalid")


def _camera_pose(value):
    _require(isinstance(value, list) and len(value) == 4
        and all(isinstance(row, list) and len(row) == 4 and all(_number(number) for number in row)
                for row in value)
        and validate_se3_matrix(value, field="world_from_camera")["valid"], "native_pose_invalid")
    return value


def _camera_records(calibration, coordinates, raw_originals):
    """Normalize only declared native camera axes; never fit a camera to the scene."""
    record_format = calibration.get("camera_record_format", "normalized_opencv_v1")
    _require(isinstance(record_format, str)
        and record_format in {"normalized_opencv_v1", "blueprint_raw_v3_2"}, "camera_record_format_invalid")
    native_arkit = record_format == "blueprint_raw_v3_2"
    convention = "arkit_x_right_y_up_z_backward" if native_arkit else "opencv"
    _require(coordinates.get("camera_convention") == convention, "native_coordinate_normalization_required")
    intrinsics_document = json.loads(raw_originals["camera_intrinsics"])
    rows = [json.loads(line) for line in raw_originals["camera_poses"].splitlines() if line.strip()]
    _require(isinstance(intrinsics_document, Mapping) and len(rows) <= 4096
        and all(isinstance(row, Mapping) and isinstance(row.get("frame_id"), str)
                and bool(row["frame_id"]) for row in rows), "native_calibration_invalid")
    intrinsics = intrinsics_document.get("intrinsics") if native_arkit else intrinsics_document
    _require(isinstance(intrinsics, Mapping)
        and all(_number(intrinsics.get(key)) for key in ("fx", "fy", "cx", "cy"))
        and intrinsics["fx"] > 0 and intrinsics["fy"] > 0
        and all(type(intrinsics.get(key)) is int and intrinsics[key] > 0 for key in ("width", "height")),
        "native_intrinsics_invalid")
    if native_arkit:
        session_id = coordinates.get("coordinate_frame_session_id")
        _require(intrinsics_document.get("coordinate_frame_session_id") == session_id
            and all(row.get("coordinate_frame_session_id") == session_id for row in rows),
            "native_camera_session_changed")
        normalized = []
        for row in rows:
            pose = _camera_pose(row.get("T_world_camera"))
            normalized.append({"frame_id": row["frame_id"], "world_from_camera": (
                np.asarray(pose, dtype=float) @ np.diag([1, -1, -1, 1])).tolist()})
        rows = normalized
    by_native_frame = {row["frame_id"]: row for row in rows}
    _require(len(by_native_frame) == len(rows), "native_frame_identity_duplicate")
    return intrinsics, by_native_frame


def _derive(*, records, binding, scene, frame_rows, target_ids):
    envelope = validate_capture_intake_envelope(records["capture_intake"])
    admission = build_capture_admission(envelope)
    _require(admission["status"] == "accepted" and admission["claim_ceiling"]["metric_geometry"] is True
        and envelope["capture_authority_profile"] == "iphone_arkit_lidar", "metric_capture_required")
    _require(envelope["scene_id"] == binding["scene_id"], "capture_scene_changed")
    coordinates = envelope["coordinate_frame_declaration"]
    _require(coordinates.get("unit") == "meters"
        and coordinates.get("handedness") == "right", "native_coordinate_normalization_required")
    for kind in _KINDS[1:]:
        record = records[kind]
        _require(isinstance(record, Mapping) and all(record.get(key) == binding[key] for key in
            ("capture_id", "source_video_digest", "source_geometry_digest"))
            and record.get("coordinate_frame_session_id") == records["calibration"].get("coordinate_frame_session_id")
            and isinstance(record.get("coordinate_frame_session_id"), str)
            and bool(record["coordinate_frame_session_id"]), "source_binding_changed:" + kind)
    calibration = records["calibration"]
    _require(calibration.get("coordinate_frame_session_id") == coordinates.get("coordinate_frame_session_id"),
             "coordinate_frame_changed")
    _require(calibration.get("schema_version") == "website_metric_capture_calibration.v1"
        and calibration.get("intake_envelope_digest") == envelope["envelope_digest"]
        and calibration.get("measurement_source") == "sensor_depth"
        and calibration.get("metric_depth_truth") is True and calibration.get("metric_pose_truth") is True
        and calibration.get("intrinsics_capture_truth") is True, "calibration_not_measured")
    originals = calibration.get("original_streams")
    _require(isinstance(originals, Mapping) and set(originals) == {
        "camera_intrinsics", "camera_poses", "depth", "depth_confidence"}, "calibration_originals_required")
    files = {row["relative_path"]: row for row in envelope["original_files"]}
    streams = {row["stream_type"]: row.get("source_relative_path") for row in envelope["available_sensor_streams"]
               if row.get("status") == "available"}
    _require(files.get(streams.get("retained_video"), {}).get("sha256") == binding["source_video_digest"],
             "source_video_changed")
    raw_originals = {}
    for kind, reference in originals.items():
        original = files.get(streams.get(kind), {})
        _require(isinstance(reference, Mapping) and reference.get("sha256") == original.get("sha256")
            and reference.get("size_bytes") == original.get("size_bytes"), "calibration_original_changed")
        raw_originals[kind] = _read(reference, "metric_layout_original_" + kind,
                                    as_bytes=kind in {"camera_intrinsics", "camera_poses"})
    intrinsics, by_native_frame = _camera_records(calibration, coordinates, raw_originals)
    measured = calibration.get("frames")
    _require(isinstance(measured, list) and 0 < len(measured) <= 4096
        and all(isinstance(row, Mapping) and isinstance(row.get("frame_id"), str)
                and bool(row["frame_id"]) for row in measured)
        and len({row["frame_id"] for row in measured}) == len(measured), "camera_frames_invalid")
    calibrated = {row["frame_id"]: row for row in measured}
    for frame in frame_rows:
        actual = calibrated.get(frame["frame_id"])
        _require(actual is not None and all(actual.get(key) == frame.get(key) for key in (
            "intrinsics", "world_from_camera", "image_digest", "geometry_digest", "width", "height")),
            "camera_frame_changed")
        native = by_native_frame.get(actual.get("native_frame_id"))
        matrix = [[intrinsics.get("fx"), 0, intrinsics.get("cx")],
                  [0, intrinsics.get("fy"), intrinsics.get("cy")], [0, 0, 1]]
        _require(native is not None and native.get("world_from_camera") == frame["world_from_camera"]
            and matrix == frame["intrinsics"] and intrinsics.get("width") == frame["width"]
            and intrinsics.get("height") == frame["height"], "native_camera_normalization_required")
        _camera_pose(native["world_from_camera"])
    registration = records["frame_registration"]
    _require(registration.get("schema_version") == "website_metric_frame_correspondences.v1"
        and registration.get("measurement_source") == "independent_metric_survey"
        and registration.get("collision_mesh_digest") == scene["collision_mesh_digest"], "registration_source_invalid")
    _read(registration["measurement_evidence"], "metric_layout_registration_measurement")
    frame = _frame(registration, scene)
    poses = records["target_poses"]
    _require(poses.get("schema_version") == "website_metric_target_poses.v1"
        and poses.get("measurement_source") == "independent_metric_survey"
        and isinstance(poses.get("targets"), list) and len(poses["targets"]) <= 32, "target_measurements_invalid")
    _read(poses["measurement_evidence"], "metric_layout_target_measurement")
    _require(all(isinstance(row, Mapping) and isinstance(row.get("target_id"), str) for row in poses["targets"]),
             "target_measurements_invalid")
    by_target = {row["target_id"]: row for row in poses["targets"]}
    _require(len(by_target) == len(poses["targets"]) and set(target_ids) <= set(by_target), "target_measurements_missing")
    targets = {}
    available_frames = {row["frame_id"] for row in frame_rows}
    for target_id in target_ids:
        row = by_target[target_id]
        _explicit_pose(row.get("pose_source"))
        _require(isinstance(row.get("asset_id"), str) and bool(row["asset_id"])
            and isinstance(row.get("source_frame_ids"), list) and bool(row["source_frame_ids"])
            and set(row["source_frame_ids"]) <= available_frames, "target_frame_or_asset_invalid")
        targets[target_id] = {key: row[key] for key in ("asset_id", "pose_source", "source_frame_ids")}
    robot = records["robot_registration"]
    _require(robot.get("schema_version") == "website_robot_base_registration.v1"
        and robot.get("placement_basis") in {"measured_source_registration", "engineered_placement_in_measured_frame"},
        "robot_registration_required")
    _read(robot["placement_evidence"], "metric_layout_robot_placement")
    _explicit_pose(robot.get("base_pose_source"))
    configuration, reset = robot.get("configuration"), robot.get("joint_reset_positions_rad")
    _require(isinstance(configuration, Mapping) and all(isinstance(configuration.get(key), str) and configuration[key]
        for key in ("embodiment_id", "controller", "observation_schema_id", "action_schema_id"))
        and isinstance(reset, Mapping) and bool(reset) and all(isinstance(key, str) and _number(value) for key, value in reset.items()),
        "robot_interface_or_reset_required")
    cameras = robot.get("camera_frame_ids")
    _require(isinstance(cameras, list) and bool(cameras) and len(set(cameras)) == len(cameras)
        and set(cameras) <= available_frames, "robot_camera_registration_required")
    return {"schema_version": "website_registered_task_layout.v1",
        "source_video_digest": binding["source_video_digest"], "source_geometry_digest": binding["source_geometry_digest"],
        "collision_mesh_digest": scene["collision_mesh_digest"], "metric_registration_verified": True,
        "frame_registration": frame, "targets": targets,
        "robot": {key: robot[key] for key in ("configuration", "base_pose_source", "joint_reset_positions_rad")},
        "camera_frame_ids": cameras, "robot_placement_basis": robot["placement_basis"]}


def verify_metric_layout_registration(value):
    """Reopen original measurements again at native packet readback."""
    _require(value.get("schema_version") == SCHEMA
        and value.get("registration_digest") == canonical_digest(value, digest_field="registration_digest"), "receipt_changed")
    expected = _derive(records=_records(value["source_records"]), binding=value["binding"], scene=value["scene"],
        frame_rows=value["source_frames"], target_ids=value["target_ids"])
    _require(expected == value["layout_projection"], "layout_projection_changed")


def emit_registered_task_layout(*, task_context, task_masks, source_geometry, base_scene, output_root):
    """Use existing family authoring stages; missing measurements stay required."""
    if task_masks.get("registered_layout") is not None:
        return task_masks
    from .task_family_button_authoring import _retain
    references = task_masks.get("layout_source_records")
    binding = {"capture_id": task_context["capture_id"], "scene_id": task_context["scene_id"],
        "source_video_digest": (source_geometry.get("binding") or {}).get("source_video_digest"),
        "source_geometry_digest": source_geometry["digest"]}
    missing = [kind for kind in _KINDS if not isinstance(references, Mapping) or references.get(kind) is None]
    if (source_geometry.get("metric_measurement_proven") is not True or source_geometry.get("unit") not in {"m", "meters"}
            or source_geometry.get("geometry_backend") not in {"arkit", "native_arkit_rgbd.v1"}):
        missing = list(_KINDS)
    requirements = [{"requirement_id": "job:" + _QUANTITIES[kind], "stage_id": "job", "quantity": _QUANTITIES[kind],
        "category": "source_measurement", "owner": "scene_preparation", "status": "required",
        "question": "Retain source-bound " + _QUANTITIES[kind].replace("_", " ") + "; estimated geometry cannot supply it.",
        "unit": None, "authority_created": False} for kind in missing]
    receipt = {"binding": binding, "source_records": dict(references) if isinstance(references, Mapping) else {},
               "requirements": requirements, "status": "needs_measurement" if missing else "layout_emitted"}
    result = dict(task_masks)
    if not missing:
        frame_rows = [{key: row.get(key) for key in ("frame_id", "intrinsics", "world_from_camera",
            "image_digest", "geometry_digest", "width", "height")} for row in source_geometry["frames"]]
        scene = {key: base_scene[key] for key in ("collision_mesh_digest", "meters_per_unit", "up_axis")}
        target_ids = [row["target_id"] for row in task_masks["targets"] if row.get("task_effect") in {"manipulated", "static_contact"}]
        layout = _derive(records=_records(references), binding=binding, scene=scene, frame_rows=frame_rows, target_ids=target_ids)
        registration = {"schema_version": SCHEMA, "source_records": dict(references), "binding": binding,
            "scene": scene, "source_frames": frame_rows, "target_ids": target_ids, "layout_projection": layout,
            "source_video_digest": binding["source_video_digest"], "frame_registration": layout["frame_registration"],
            "claim_ceiling": "development_only", "automatic_paid_execution_authorized": False}
        registration["registration_digest"] = canonical_digest(registration, digest_field="registration_digest")
        stage = _retain(Path(output_root), "metric-layout-registration", registration)
        # _layout consumes a flat, immutable registration record. Reuse the
        # same exclusive authoring writer rather than rewriting a current file.
        from .launch_immutable_input_writer import write_exclusive_private_bytes
        from .task_object_authoring_contracts import file_record
        path = Path(output_root) / "family_authoring" / "registrations" / registration["registration_digest"][7:] / "registration.json"
        write_exclusive_private_bytes(path, (json.dumps(registration, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
        layout = {**layout, "registration_evidence": file_record(path)}
        layout["layout_digest"] = canonical_digest(layout, digest_field="layout_digest")
        result["registered_layout"] = layout
        receipt.update(layout_digest=layout["layout_digest"], registration_stage=stage)
    result["layout_authoring"] = {"reference": _retain(Path(output_root), "registered-layout", receipt), "requirements": requirements}
    result["digest"] = canonical_digest(result, digest_field="digest")
    return result
