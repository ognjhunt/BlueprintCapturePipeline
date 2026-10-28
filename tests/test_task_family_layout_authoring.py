# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_family_layout_authoring.py
#   src/blueprint_pipeline/task_family_preparation.py
#   src/blueprint_pipeline/task_family_button_authoring.py
#   src/blueprint_pipeline/task_family_authoring_evidence.py
"""Native-record development fixtures; no real measurement or runtime admission."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.capture_intake import _REQUIRED_STREAMS, validate_capture_intake_envelope
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_family_layout_authoring import emit_registered_task_layout
from blueprint_pipeline.task_object_authoring_contracts import file_record
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from tests.test_capture_intake import _envelope
from tests.test_task_family_button_authoring import _fixture, _reseal_facts


def _native_records(tmp_path, monkeypatch):
    args, _row = _fixture(tmp_path, monkeypatch)
    context, geometry, masks, scene = [args[key] for key in ("task_context", "source_geometry", "task_masks", "base_scene")]
    previous_layout = masks.pop("registered_layout")
    root = tmp_path / "native-measurement-fixture"
    root.mkdir()
    video = root / "capture.mp4"
    payload = b"development-only source video identity; not a real measurement"
    video.write_bytes(payload)
    video_digest = file_record(video)["sha256"]
    geometry["binding"]["source_video_digest"] = video_digest
    geometry["geometry_backend"] = "arkit"
    geometry["unit"] = "meters"
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    masks["source_geometry_digest"] = geometry["digest"]
    context["operator_answers"]["family_task_facts"]["source_video_digest"] = video_digest
    _reseal_facts(args)
    first = geometry["frames"][0]
    intrinsics = root / "intrinsics.json"
    write_json(intrinsics, {"fx": first["intrinsics"][0][0], "fy": first["intrinsics"][1][1],
        "cx": first["intrinsics"][0][2], "cy": first["intrinsics"][1][2],
        "width": first["width"], "height": first["height"]})
    poses = root / "poses.jsonl"
    poses.write_text("".join(json.dumps({"frame_id": frame["frame_id"],
        "world_from_camera": frame["world_from_camera"]}) + "\n" for frame in geometry["frames"]))
    depth, confidence, metadata = root / "depth.bin", root / "confidence.bin", root / "stream-metadata.json"
    depth.write_bytes(b"development fixture depth stream")
    confidence.write_bytes(b"development fixture confidence stream")
    write_json(metadata, {"claim_ceiling": "development_only", "actual_measurement": False})
    stream_files = {"retained_video": video, "camera_intrinsics": intrinsics, "camera_poses": poses,
                    "depth": depth, "depth_confidence": confidence}
    native = _envelope(payload, profile="iphone_arkit_lidar", streams=sorted(_REQUIRED_STREAMS["iphone_arkit_lidar"]))
    native["scene_id"] = context["scene_id"]
    native["coordinate_frame_declaration"] = {"unit": "meters", "camera_convention": "opencv",
        "handedness": "right", "coordinate_frame_session_id": "fixture-native-frame"}
    native["original_files"] = [{"original_filename": path.name, "relative_path": path.name,
        "sha256": file_record(path)["sha256"], "size_bytes": path.stat().st_size,
        "media_type": "application/octet-stream"} for path in [video, intrinsics, poses, depth, confidence, metadata]]
    for row in native["available_sensor_streams"]:
        row["source_relative_path"] = stream_files.get(row["stream_type"], metadata).name
    native = validate_capture_intake_envelope(native)
    intake_path = root / "intake.json"
    write_json(intake_path, native)
    binding = {"capture_id": context["capture_id"], "source_video_digest": video_digest,
        "source_geometry_digest": geometry["digest"], "coordinate_frame_session_id": "fixture-native-frame"}
    calibration = {"schema_version": "website_metric_capture_calibration.v1", **binding,
        "intake_envelope_digest": native["envelope_digest"], "measurement_source": "sensor_depth",
        "metric_depth_truth": True, "metric_pose_truth": True, "intrinsics_capture_truth": True,
        "original_streams": {key: file_record(path) for key, path in stream_files.items() if key != "retained_video"},
        "frames": [{**{key: frame[key] for key in ("frame_id", "intrinsics", "world_from_camera", "image_digest",
            "geometry_digest", "width", "height")}, "native_frame_id": frame["frame_id"]} for frame in geometry["frames"]]}
    survey = root / "survey.json"
    write_json(survey, {"development_fixture": True, "actual_measurement": False})
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] /= scene["meters_per_unit"]
    matrix[:3, 3] = [1, 2, 3]
    registration = {"schema_version": "website_metric_frame_correspondences.v1", **binding,
        "measurement_source": "independent_metric_survey", "measurement_evidence": file_record(survey),
        "collision_mesh_digest": scene["collision_mesh_digest"], "source_points_m": points.tolist(),
        "runtime_points": (points @ matrix[:3, :3].T + matrix[:3, 3]).tolist(),
        "source_to_runtime": matrix.tolist(), "maximum_registration_residual_m": 1e-6}
    targets = {"schema_version": "website_metric_target_poses.v1", **binding,
        "measurement_source": "independent_metric_survey", "measurement_evidence": file_record(survey),
        "targets": [{"target_id": key, **value} for key, value in previous_layout["targets"].items()]}
    robot = {"schema_version": "website_robot_base_registration.v1", **binding,
        "placement_basis": "engineered_placement_in_measured_frame", "placement_evidence": file_record(survey),
        **previous_layout["robot"], "camera_frame_ids": previous_layout["camera_frame_ids"]}
    sources = {"capture_intake": file_record(intake_path)}
    for kind, value in (("calibration", calibration), ("frame_registration", registration),
                         ("target_poses", targets), ("robot_registration", robot)):
        path = root / (kind + ".json")
        write_json(path, value)
        sources[kind] = file_record(path)
    masks["layout_source_records"] = sources
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    return args, sources


def _change(args, references, kind, edit):
    path = Path(references[kind]["path"])
    value = json.loads(path.read_text())
    edit(value)
    write_json(path, value)
    references[kind] = file_record(path)
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")


def _raw_v32_records(args, references):
    """Exercise documented Raw V3.2 field/axis names with development-only bytes."""
    envelope_path = Path(references["capture_intake"]["path"])
    envelope = json.loads(envelope_path.read_text())
    envelope["coordinate_frame_declaration"]["camera_convention"] = "arkit_x_right_y_up_z_backward"
    calibration_path = Path(references["calibration"]["path"])
    calibration = json.loads(calibration_path.read_text())
    calibration["camera_record_format"] = "blueprint_raw_v3_2"
    session_id = calibration["coordinate_frame_session_id"]
    intrinsics_path = Path(calibration["original_streams"]["camera_intrinsics"]["path"])
    intrinsics = json.loads(intrinsics_path.read_text())
    write_json(intrinsics_path, {"coordinate_frame_session_id": session_id, "intrinsics": intrinsics})
    poses_path = Path(calibration["original_streams"]["camera_poses"]["path"])
    poses = [{"frame_id": row["frame_id"], "coordinate_frame_session_id": session_id,
              "T_world_camera": (np.asarray(row["world_from_camera"]) @ np.diag([1, -1, -1, 1])).tolist()}
             for row in calibration["frames"]]
    poses_path.write_text("".join(json.dumps(row) + "\n" for row in poses))
    for kind, path in (("camera_intrinsics", intrinsics_path), ("camera_poses", poses_path)):
        reference = file_record(path)
        calibration["original_streams"][kind] = reference
        next(row for row in envelope["original_files"] if row["relative_path"] == path.name).update(
            sha256=reference["sha256"], size_bytes=reference["size_bytes"])
    envelope.pop("envelope_digest")
    envelope = validate_capture_intake_envelope(envelope)
    write_json(envelope_path, envelope)
    references["capture_intake"] = file_record(envelope_path)
    calibration["intake_envelope_digest"] = envelope["envelope_digest"]
    write_json(calibration_path, calibration)
    references["calibration"] = file_record(calibration_path)
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")


def test_ordinary_native_records_emit_layout_and_native_inputs_without_injected_packet(tmp_path, monkeypatch):
    args, sources = _native_records(tmp_path, monkeypatch)
    assert "registered_layout" not in args["task_masks"] and "evaluation_inputs" not in args
    result = compile_website_scene_preparation(**args)
    assert result["status"] == "intake_ready", result
    inputs = result["task_evaluation_inputs"]
    assert inputs["authoring"]["request"]["native_inputs_authored"] is True
    native = inputs["native_runtime_contract"]
    position = native["assets"][0]["reset_state"]["pose_world"]["position_world_m"]
    assert position == pytest.approx([2.2, 4.5, 7.5])
    assert native["cameras"][0]["role"] == "source_reference"
    assert inputs["automatic_paid_execution_authorized"] is False
    assert all(Path(ref["path"]).is_file() for ref in sources.values())


def test_raw_v32_camera_axes_normalize_without_fitting_source_or_placement(tmp_path, monkeypatch):
    args, sources = _native_records(tmp_path, monkeypatch)
    expected_frames = json.loads(json.dumps(args["source_geometry"]["frames"]))
    _raw_v32_records(args, sources)
    value = emit_registered_task_layout(**{key: args[key] for key in (
        "task_context", "task_masks", "source_geometry", "base_scene", "output_root")})
    assert args["source_geometry"]["frames"] == expected_frames
    assert value["registered_layout"]["robot_placement_basis"] == "engineered_placement_in_measured_frame"
    result = compile_website_scene_preparation(**args)
    assert result["status"] == "intake_ready"
    assert result["task_evaluation_inputs"]["automatic_paid_execution_authorized"] is False


def test_raw_camera_boolean_does_not_normalize_to_a_numeric_pose(tmp_path, monkeypatch):
    args, sources = _native_records(tmp_path, monkeypatch)
    _raw_v32_records(args, sources)
    calibration = json.loads(Path(sources["calibration"]["path"]).read_text())
    poses_path = Path(calibration["original_streams"]["camera_poses"]["path"])
    rows = [json.loads(row) for row in poses_path.read_text().splitlines()]
    rows[0]["T_world_camera"][0][0] = True
    poses_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    changed = file_record(poses_path)
    envelope_path = Path(sources["capture_intake"]["path"])
    envelope = json.loads(envelope_path.read_text())
    next(row for row in envelope["original_files"] if row["relative_path"] == poses_path.name).update(
        sha256=changed["sha256"], size_bytes=changed["size_bytes"])
    envelope.pop("envelope_digest")
    envelope = validate_capture_intake_envelope(envelope)
    write_json(envelope_path, envelope)
    sources["capture_intake"] = file_record(envelope_path)
    calibration["intake_envelope_digest"] = envelope["envelope_digest"]
    calibration["original_streams"]["camera_poses"] = changed
    write_json(Path(sources["calibration"]["path"]), calibration)
    sources["calibration"] = file_record(Path(sources["calibration"]["path"]))
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    with pytest.raises(ValueError, match="native_pose_invalid"):
        compile_website_scene_preparation(**args)


def test_layout_restart_adopts_identical_immutable_bytes(tmp_path, monkeypatch):
    args, _sources = _native_records(tmp_path, monkeypatch)
    first = compile_website_scene_preparation(**args)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in (args["output_root"] / "family_authoring").rglob("*.json")}
    assert compile_website_scene_preparation(**args) == first
    assert before and all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items())


@pytest.mark.parametrize("kind", ["calibration", "frame_registration", "target_poses", "robot_registration"])
def test_metric_record_cannot_select_different_capture_coordinate_session(tmp_path, monkeypatch, kind):
    args, sources = _native_records(tmp_path, monkeypatch)
    _change(args, sources, kind, lambda value: value.update(coordinate_frame_session_id="different-session"))
    with pytest.raises(ValueError, match="source_binding_changed|coordinate_frame_changed"):
        compile_website_scene_preparation(**args)


@pytest.mark.parametrize("camera_format", [None, "estimated_camera", True])
def test_unknown_camera_format_is_not_a_native_normalization(tmp_path, monkeypatch, camera_format):
    args, sources = _native_records(tmp_path, monkeypatch)
    _change(args, sources, "calibration", lambda value: value.update(camera_record_format=camera_format))
    with pytest.raises(ValueError, match="camera_record_format_invalid"):
        compile_website_scene_preparation(**args)


def test_true_literal_cannot_supply_registration_scale(tmp_path, monkeypatch):
    args, sources = _native_records(tmp_path, monkeypatch)
    _change(args, sources, "frame_registration", lambda value: value["source_to_runtime"][3].__setitem__(3, True))
    with pytest.raises(ValueError, match="registration_matrix_invalid"):
        compile_website_scene_preparation(**args)


def test_retained_video_stream_cannot_bind_an_unrelated_original(tmp_path, monkeypatch):
    args, sources = _native_records(tmp_path, monkeypatch)
    path = Path(sources["capture_intake"]["path"])
    envelope = json.loads(path.read_text())
    next(row for row in envelope["available_sensor_streams"] if row["stream_type"] == "retained_video")[
        "source_relative_path"] = "intrinsics.json"
    envelope.pop("envelope_digest")
    envelope = validate_capture_intake_envelope(envelope)
    write_json(path, envelope)
    sources["capture_intake"] = file_record(path)
    _change(args, sources, "calibration", lambda value: value.update(intake_envelope_digest=envelope["envelope_digest"]))
    with pytest.raises(ValueError, match="source_video_changed"):
        compile_website_scene_preparation(**args)


def test_camera_bytes_reader_keeps_metadata_limit_and_hash_identity(tmp_path):
    from blueprint_pipeline.task_family_authoring_evidence import MAX_METADATA_BYTES, _read

    path = tmp_path / "camera.jsonl"
    path.write_bytes(b"pinned-camera-data\n")
    reference = file_record(path)
    assert _read(reference, "camera", as_bytes=True) == path.read_bytes()
    assert _read(reference, "depth") is None  # Large streams remain hash-only.
    with pytest.raises(ValueError, match="metadata_limit:camera"):
        _read({**reference, "size_bytes": MAX_METADATA_BYTES + 1}, "camera", as_bytes=True)
    path.write_bytes(b"changed-camera-data\n")
    with pytest.raises(ValueError, match="bytes_changed:camera"):
        _read(reference, "camera", as_bytes=True)


@pytest.mark.parametrize("fault,reason", [
    ("source_capture", "source_binding_changed"), ("camera_pose", "camera_frame_changed"),
    ("raw_camera", "native_camera_normalization_required"), ("degenerate", "correspondences_degenerate"),
    ("wrong_scale", "registration_scale_or_orientation_invalid"), ("bad_fit", "registration_residual_exceeded"),
    ("estimated", "registration_source_invalid"), ("missing_target", "target_measurements_missing"),
    ("robot_camera", "robot_camera_registration_required"),
])
def test_incoherent_records_refuse_before_native_or_provider_work(tmp_path, monkeypatch, fault, reason):
    args, sources = _native_records(tmp_path, monkeypatch)
    if fault == "source_capture":
        _change(args, sources, "target_poses", lambda value: value.update(capture_id="another-capture"))
    elif fault == "camera_pose":
        _change(args, sources, "calibration", lambda value: value["frames"][0]["world_from_camera"][0].__setitem__(3, 99))
    elif fault == "raw_camera":
        _change(args, sources, "calibration", lambda value: value["frames"][0].update(native_frame_id="another-frame"))
    elif fault == "degenerate":
        _change(args, sources, "frame_registration", lambda value: value.update(source_points_m=[[0, 0, 0]] * 4))
    elif fault == "wrong_scale":
        _change(args, sources, "frame_registration", lambda value: value["source_to_runtime"][0].__setitem__(0, 99))
    elif fault == "bad_fit":
        _change(args, sources, "frame_registration", lambda value: value["runtime_points"][0].__setitem__(0, 99))
    elif fault == "estimated":
        _change(args, sources, "frame_registration", lambda value: value.update(measurement_source="model_estimated"))
    elif fault == "missing_target":
        _change(args, sources, "target_poses", lambda value: value.update(targets=[]))
    else:
        _change(args, sources, "robot_registration", lambda value: value.update(camera_frame_ids=["unobserved-camera"]))
    with pytest.raises(ValueError, match=reason):
        compile_website_scene_preparation(**args)


def test_estimated_geometry_cannot_gain_measurement_authority_from_layout_records(tmp_path, monkeypatch):
    args, _sources = _native_records(tmp_path, monkeypatch)
    args["source_geometry"]["geometry_backend"] = "world_labs_atlas"
    args["source_geometry"]["metric_measurement_proven"] = True  # A relabel is insufficient.
    args["source_geometry"]["digest"] = canonical_digest(args["source_geometry"], digest_field="digest")
    args["task_masks"]["source_geometry_digest"] = args["source_geometry"]["digest"]
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    result = compile_website_scene_preparation(**args)
    assert result["status"] == "needs_input" and result["intake_request"] is None
    requirements = result["task_evaluation_inputs"]["authoring"]["request"]["requirements"]
    assert any(row["quantity"] == "metric_capture_calibration" and row["owner"] == "scene_preparation" for row in requirements)
    assert args["source_geometry"]["geometry_backend"] == "world_labs_atlas"


def test_missing_records_persist_precise_measurement_requirements_without_losing_job(tmp_path, monkeypatch):
    args, _sources = _native_records(tmp_path, monkeypatch)
    args["task_masks"].pop("layout_source_records")
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    result = compile_website_scene_preparation(**args)
    assert result["job_disposition"] == "accepted" and result["status"] == "needs_input"
    request = result["task_evaluation_inputs"]["authoring"]["request"]
    assert any(row["quantity"] == "robot_base_registration" for row in request["requirements"])
    assert all(row["owner"] != "task_owner" for row in request["requirements"] if row["quantity"] in {
        "metric_capture_calibration", "source_runtime_registration", "target_pose_measurements", "robot_base_registration"})
    assert list((args["output_root"] / "family_authoring" / "stages").glob("*/registered-layout.json"))


def test_packet_read_reopens_original_metric_records(tmp_path, monkeypatch):
    from blueprint_pipeline.native_task_arena_bundle import verify_native_task_arena_packet
    from tests.test_task_family_authoring_evidence import _packet

    args, sources = _native_records(tmp_path, monkeypatch)
    prepared = compile_website_scene_preparation(**args)
    native_path = tmp_path / "native.json"
    write_json(native_path, prepared["task_evaluation_inputs"]["native_runtime_contract"])
    packet = _packet(tmp_path, args, prepared, native_path)
    verify_native_task_arena_packet(packet["packet_root"])
    path = Path(sources["robot_registration"]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="authoring_evidence_bytes_changed:metric_layout_robot_registration"):
        verify_native_task_arena_packet(packet["packet_root"])
