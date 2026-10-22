"""Atlas posing as website source geometry: hermetic, against a scene with known truth.

Atlas returns cameras and z-depth on its own cover-cropped grid. Here a plane is
rendered for known cameras straight onto that grid, so every lifted pixel has an
exact expected world point. No provider call; the receipt is a fixture.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import atlas_source_geometry as geometry
from blueprint_pipeline import website_atlas_pose as pose
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_scene_geometry import INPUT_SCHEMA
from blueprint_pipeline.website_task_masks import masked_source_points

# A 20x14 model input covers a 60x36 grid at exactly 3x with 3 rows cropped top
# and bottom, so each model pixel centre is an Atlas pixel centre.
MODEL, GRID = (20, 14), (60, 36)
K = {"width": GRID[0], "height": GRID[1], "fx": 50.0, "fy": 52.0, "cx": 30.0, "cy": 18.0}
PLANE_NORMAL, PLANE_OFFSET = np.array([0.1, -0.2, 1.0]) / np.linalg.norm([0.1, -0.2, 1.0]), 4.0
SHA = "sha256:" + "a" * 64


def write_exr(path, depth):
    """Uncompressed single-channel FLOAT scanline EXR (read_exr_depth's accepted layout)."""
    data = np.ascontiguousarray(depth, dtype="<f4")
    height, width = data.shape

    def attribute(name, kind, value):
        return name.encode() + b"\0" + kind.encode() + b"\0" + struct.pack("<I", len(value)) + value

    window = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    header = b"".join([b"\x76\x2f\x31\x01", struct.pack("<I", 2),
                       attribute("channels", "chlist", b"Y\0" + struct.pack("<iB3xii", 2, 0, 1, 1) + b"\0"),
                       attribute("compression", "compression", b"\0"),
                       attribute("dataWindow", "box2i", window), attribute("displayWindow", "box2i", window),
                       attribute("lineOrder", "lineOrder", b"\0"), b"\0"])
    chunks = [struct.pack("<ii", y, width * 4) + data[y].tobytes() for y in range(height)]
    offsets, offset = [], len(header) + 8 * len(chunks)
    for chunk in chunks:
        offsets.append(offset)
        offset += len(chunk)
    path.write_bytes(header + struct.pack(f"<{len(chunks)}Q", *offsets) + b"".join(chunks))


def rotation(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])


def quaternion_xyzw(matrix):
    w = np.sqrt(max(0.0, 1 + np.trace(matrix))) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) / 2, matrix[2, 1] - matrix[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2])) / 2, matrix[0, 2] - matrix[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2])) / 2, matrix[1, 0] - matrix[0, 1])
    return [float(x), float(y), float(z), float(w)]


# OpenCV camera-to-world truth for three views of the plane.
CAMERAS = [(rotation(0.0, 0.0), np.array([0.0, 0.0, 0.0])),
           (rotation(0.15, -0.05), np.array([0.4, 0.1, 0.2])),
           (rotation(-0.1, 0.08), np.array([-0.3, -0.2, 0.1]))]


def atlas_pixel_rays(rotation_cv):
    """World directions through every Atlas pixel centre, z-component 1 in camera space."""
    u = (np.arange(GRID[0]) + 0.5 - K["cx"]) / K["fx"]
    v = (np.arange(GRID[1]) + 0.5 - K["cy"]) / K["fy"]
    camera = np.stack(np.broadcast_arrays(u[None, :], v[:, None], np.ones((GRID[1], GRID[0]))), axis=-1)
    return camera @ rotation_cv.T


def plane_depth(rotation_cv, position):
    rays = atlas_pixel_rays(rotation_cv)
    return ((PLANE_OFFSET - PLANE_NORMAL @ position) / (rays @ PLANE_NORMAL)).astype(np.float32)


def camera_payload(rotation_cv, position, axes):
    flip = np.diag([1.0, -1.0, -1.0])
    matrix = rotation_cv @ flip if axes == "rub" else rotation_cv
    return {"intrinsics": dict(K), "extrinsics": {"coordinateSystem": axes, "position": position.tolist(),
                                                   "quaternion": quaternion_xyzw(matrix)}}


def build(root, *, axes="rub", corrupt_rgb=False, binding_digest=None):
    """A geometry root as scene preparation and an admitted pose run would leave it."""
    rng = np.random.default_rng(7)
    inputs_root = root / "worker_inputs"
    (inputs_root / "model_inputs").mkdir(parents=True)
    (inputs_root / "observations").mkdir()
    frames, outputs, sent, returned = [], [], [], []
    for index, (rotation_cv, position) in enumerate(CAMERAS):
        model = (rng.random((MODEL[1], MODEL[0], 3)) * 255).astype(np.uint8)
        model_path = inputs_root / "model_inputs" / f"{index:06d}.png"
        source_path = inputs_root / "observations" / f"{index:06d}.png"
        Image.fromarray(model).save(model_path)
        Image.fromarray(model).save(source_path)
        frames.append({"frame_id": f"decoded-{index:09d}", "timestamp_seconds": float(index),
                       "source_image_path": f"observations/{index:06d}.png", "source_image_digest": _sha256_file(source_path),
                       "image_path": f"model_inputs/{index:06d}.png", "image_digest": _sha256_file(model_path),
                       "source_width": MODEL[0], "source_height": MODEL[1], "width": MODEL[0], "height": MODEL[1],
                       "display_rotation_degrees": 0.0,
                       "source_to_geometry_pixels": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
        # Atlas's reframed RGB: the input at 3x, centre-cropped to the grid.
        atlas_rgb = np.repeat(np.repeat(model, 3, axis=0), 3, axis=1)[3:39]
        if corrupt_rgb:
            atlas_rgb = (rng.random(atlas_rgb.shape) * 255).astype(np.uint8)
        out = root / "atlas_pose" / "outputs" / f"{index:06d}"
        out.mkdir(parents=True)
        cv2.imwrite(str(out / "image.png"), atlas_rgb[..., ::-1])
        write_exr(out / "depth.exr", plane_depth(rotation_cv, position))
        confidence = np.full((GRID[1], GRID[0]), 200, np.uint8)
        confidence[10, 10] = 0
        cv2.imwrite(str(out / "confidence.png"), confidence)
        outputs.append([{"role": role, "asset_id": f"asset_{role}_{index}", "path": f"atlas_pose/outputs/{index:06d}/{name}",
                         "sha256": _sha256_file(out / name), "bytes": (out / name).stat().st_size}
                        for role, name in (("image", "image.png"), ("depth", "depth.exr"), ("confidence", "confidence.png"))])
        sent.append({"imageAsset": {"assetId": f"asset_in_{index}"}})
        returned.append({"imageAsset": {"assetId": f"asset_image_{index}"}, "camera": camera_payload(rotation_cv, position, axes),
                         "depth": {"depthAsset": {"assetId": f"asset_depth_{index}"},
                                   "confidenceAsset": {"assetId": f"asset_confidence_{index}"}}})
    inputs = {"schema_version": INPUT_SCHEMA, "binding": {"source_video_digest": SHA, "maximum_frames": 16,
                                                          "adapter_version": 2},
              "frames": frames, "heldout_pixels_included": False, "claim_ceiling": "development_only"}
    inputs["digest"] = canonical_digest(inputs, digest_field="digest")
    (inputs_root / "geometry_inputs.json").write_text(json.dumps(inputs))
    source_binding = {"source_video_digest": SHA, "geometry_input_digest": binding_digest or inputs["digest"]}
    identities = [{"frame_id": row["frame_id"], "image_digest": row["image_digest"]} for row in frames]
    receipt_binding = {"source_binding": source_binding, "frames": identities, "request_options": pose.request_options(),
                       "target_resolution": list(GRID), "provider": pose.PROVIDER}
    request = {"frames": sent, "targetResolution": list(GRID), **pose.request_options()}
    receipt = {"schema_version": pose.RECEIPT_SCHEMA, "operation_id": "op_geometry",
               "request_digest": canonical_digest(receipt_binding), "binding": receipt_binding, "request": request,
               "input_usage": {"submitted_frame_count": len(frames), "frames": [
                   {**identity, "asset_id": item["imageAsset"]["assetId"], "content_type": "image/png", "bytes": 1,
                    "outputs": rows} for identity, item, rows in zip(identities, sent, outputs)]}}
    result = {"schema_version": "website_atlas_pose_result.v1", "endpoint": pose.ATLAS_POSE_ENDPOINT,
              "binding": source_binding, "operation_receipt": receipt, "claim_ceiling": "development_only",
              "operation": {"id": "op_geometry", "done": True, "error": None,
                            "response": {"poseRefinementAccepted": True, "frames": returned}}}
    result["digest"] = canonical_digest(result, digest_field="digest")
    (root / "atlas_posed_rgbd_result.json").write_text(json.dumps(result))
    (root / "atlas_pose" / "controller_admission.json").write_text(json.dumps(
        {"status": "admitted", "provider": pose.PROVIDER, "allocation_binding_digest": receipt["request_digest"]}))
    (root / "atlas_pose" / "submission.json").write_text(json.dumps(
        {"status": "completed", "request_digest": receipt["request_digest"], "operation_id": "op_geometry",
         "result_digest": result["digest"]}))
    return inputs


def expected_world(index, mask):
    """The plane point seen through each selected model pixel, i.e. its Atlas pixel centre."""
    rotation_cv, position = CAMERAS[index]
    rows, columns = np.nonzero(mask)
    rays = atlas_pixel_rays(rotation_cv)[3 * rows - 2, 3 * columns + 1]
    return position + rays * ((PLANE_OFFSET - PLANE_NORMAL @ position) / (rays @ PLANE_NORMAL))[:, None]


def lifted(result, index, mask):
    frame = result["frames"][index]
    runs = np.flatnonzero(np.diff(np.pad(mask.reshape(-1).astype(np.int8), (1, 1))))
    observation = {"source_frame_id": frame["frame_id"], "width": MODEL[0], "height": MODEL[1],
                   "runs": [{"start": int(a), "length": int(b - a)} for a, b in zip(runs[0::2], runs[1::2])]}
    return masked_source_points({"observations": [observation]}, result["frames"])


SCALE = {"meters_per_unit": 1.25, "basis": "reference_estimate", "evidence_digest": SHA,
         "evidence_reference": "automation/scene_measurements.json"}


@pytest.mark.parametrize("axes", ["rub", "rdf"])
def test_masks_lift_onto_atlas_depth_exactly_as_onto_mapanything_depth(tmp_path, axes):
    build(tmp_path, axes=axes)
    result = geometry.atlas_source_geometry(geometry_root=tmp_path)
    assert result["unit"] == "provider_world_units" and result["scale_status"] == "unscaled_provider_world_units"
    assert result["metric_measurement_proven"] is False and result["coordinate_system"] == "opencv_camera_to_world"
    for index, frame in enumerate(result["frames"]):
        with np.load(frame["geometry_path"]) as data:
            valid = data["valid_mask"]
        # The crop removed the top and bottom model rows; the zero-confidence pixel is invalid.
        assert not valid[0].any() and not valid[-1].any() and not valid[4, 3]
        assert frame["atlas_frame"]["pixel_binding_ncc"] > 0.999
        mask = valid.copy()
        np.testing.assert_allclose(lifted(result, index, mask), expected_world(index, mask), rtol=1e-5, atol=1e-5)


def test_a_scale_applies_to_depth_and_cameras_and_only_scaled_geometry_loads(tmp_path):
    inputs = build(tmp_path / "unscaled")
    unscaled = geometry.atlas_source_geometry(geometry_root=tmp_path / "unscaled")
    with pytest.raises(ValueError, match="atlas_source_geometry_unscaled"):
        geometry.load_atlas_source_geometry(manifest_path=Path(unscaled["manifest_path"]),
                                            inputs=inputs)
    inputs = build(tmp_path / "scaled")
    scaled = geometry.atlas_source_geometry(geometry_root=tmp_path / "scaled", scale=SCALE)
    assert scaled["unit"] == "estimated_meters" and scaled["scale_status"] == "reference_scaled"
    assert scaled["binding"]["scale"] == SCALE and scaled["metric_measurement_proven"] is False
    with np.load(scaled["frames"][1]["geometry_path"]) as data:
        mask = data["valid_mask"]
    np.testing.assert_allclose(lifted(scaled, 1, mask), 1.25 * expected_world(1, mask), rtol=1e-5, atol=1e-5)
    loaded = geometry.load_atlas_source_geometry(
        manifest_path=Path(scaled["manifest_path"]), inputs=inputs)
    assert loaded["binding"]["input_digest"] == inputs["digest"] and len(loaded["frames"]) == 3


def test_rerun_reuses_the_result_and_a_changed_scale_conflicts(tmp_path):
    build(tmp_path)
    first = geometry.atlas_source_geometry(geometry_root=tmp_path)
    assert geometry.atlas_source_geometry(geometry_root=tmp_path)["digest"] == first["digest"]
    with pytest.raises(ValueError, match="atlas_source_geometry_identity_conflict"):
        geometry.atlas_source_geometry(geometry_root=tmp_path, scale=SCALE)


@pytest.mark.parametrize("scale", [{**SCALE, "meters_per_unit": 0}, {**SCALE, "basis": "guess"},
                                   {**SCALE, "evidence_digest": "sha256:short"},
                                   {key: SCALE[key] for key in ("meters_per_unit", "basis", "evidence_digest")}])
def test_a_scale_names_a_positive_value_its_basis_and_evidence(tmp_path, scale):
    build(tmp_path)
    with pytest.raises(ValueError, match="atlas_source_geometry_scale_invalid"):
        geometry.atlas_source_geometry(geometry_root=tmp_path, scale=scale)


def test_refuses_a_run_over_other_inputs_changed_outputs_or_a_wrong_crop(tmp_path):
    build(tmp_path / "other", binding_digest=SHA)
    with pytest.raises(ValueError, match="atlas_source_geometry_pose_binding_mismatch"):
        geometry.atlas_source_geometry(geometry_root=tmp_path / "other")
    build(tmp_path / "changed")
    (tmp_path / "changed" / "atlas_pose" / "outputs" / "000001" / "depth.exr").write_bytes(b"changed")
    with pytest.raises(ValueError, match="website_atlas_retained_output_changed"):
        geometry.atlas_source_geometry(geometry_root=tmp_path / "changed")
    build(tmp_path / "unadmitted")
    (tmp_path / "unadmitted" / "atlas_pose" / "controller_admission.json").write_text("{}")
    with pytest.raises(ValueError, match="website_atlas_pose_receipt_invalid"):
        geometry.atlas_source_geometry(geometry_root=tmp_path / "unadmitted")
    build(tmp_path / "crop", corrupt_rgb=True)
    with pytest.raises(ValueError, match="atlas_source_geometry_pixel_binding_failed"):
        geometry.atlas_source_geometry(geometry_root=tmp_path / "crop")


def test_crop_map_and_intrinsics_follow_cover_and_centre_crop():
    mapping = geometry.crop_map(518, 294, (1280, 720))
    assert mapping["scale"] == pytest.approx(1280 / 518)
    assert mapping["offset"][0] == pytest.approx(0) and mapping["offset"][1] > 0
    k = geometry.model_intrinsics({"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0}, mapping)
    # The optical centre of a centred crop is the model grid's centre, in integer-pixel convention.
    assert k[0][2] == pytest.approx((518 - 1) / 2) and k[1][2] == pytest.approx((294 - 1) / 2)
    assert k[0][0] == pytest.approx(1000 * 518 / 1280)
