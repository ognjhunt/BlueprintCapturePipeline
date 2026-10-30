"""A capture's Atlas job as the website scene source, against a scene with known truth.

The job's original views see one plane through three known cameras, filmed in
portrait (90 degree display rotation) at a size Atlas covers and crops. The
background was posed separately, so its Atlas world differs from the source
pose's by a known similarity. No provider call; receipts are fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import website_atlas_scene_source as source
from blueprint_pipeline import website_atlas_pose as pose
from blueprint_pipeline.atlas_source_geometry import atlas_posed_view_geometry
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_scene_geometry import INPUT_SCHEMA
from blueprint_pipeline.website_task_masks import bind_task_masks_to_geometry, masked_source_points
from scipy.spatial.transform import Rotation
from tests.atlas_level_room_fixture import room_parts, tilt, write_room
from tests.test_atlas_source_geometry import CAMERAS, GRID, K, PLANE_NORMAL, PLANE_OFFSET, SHA, plane_depth, write_exr

UPRIGHT = (40, 28)  # Atlas covers 60x36 at 1.5x: a 60x42 view with three rows cropped top and bottom.
VIEW = (60, 42)
ROTATION = 90.0
MPU = 1.25
BACKGROUND_SCALE, BACKGROUND_TRANSLATION = 2.0, np.array([1.0, 2.0, 3.0])
FRAME_IDS = [f"decoded-{index:09d}" for index in (10, 40, 70)]
TIMES = [1 / 3, 4 / 3, 7 / 3]


def camera_payload(rotation_cv, position, axes):
    """Atlas camera JSON for an OpenCV camera-to-world pose (any rotation, including w = 0)."""
    matrix = rotation_cv @ np.diag([1.0, -1.0, -1.0]) if axes == "rub" else rotation_cv
    return {"intrinsics": dict(K), "extrinsics": {"coordinateSystem": axes, "position": list(map(float, position)),
                                                   "quaternion": Rotation.from_matrix(matrix).as_quat().tolist()}}


def yaw(degrees):
    angle = np.radians(degrees)
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])


BACKGROUND_ROTATION = yaw(20.0)


def digest(value, field="digest"):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


def pose_root_fixture(pose_root: Path, survey_digest: str, *, padded=False, cameras=CAMERAS):
    """The source pose as the preview leaves it: native upright views and an admitted run bound to them."""
    rng = np.random.default_rng(11)
    frames, outputs, sent, returned, sources = [], [], [], [], []
    for index, ((rotation_cv, position), frame_id) in enumerate(zip(cameras, FRAME_IDS)):
        upright = Image.fromarray((rng.random((UPRIGHT[1], UPRIGHT[0], 3)) * 255).astype(np.uint8))
        upright_path = pose_root / "atlas_source_views" / f"{frame_id}.png"
        observation_path = pose_root / "atlas_observations" / f"{frame_id}.png"
        for path in (upright_path, observation_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        upright.save(upright_path)
        upright.rotate(-ROTATION, expand=True).save(observation_path)
        frames.append({"frame_id": frame_id, "timestamp_seconds": TIMES[index],
                       "source_image_path": str(observation_path.resolve()),
                       "source_image_digest": _sha256_file(observation_path), "width": UPRIGHT[0],
                       "height": UPRIGHT[1], "image_path": str(upright_path.resolve()),
                       "image_digest": _sha256_file(upright_path)})
        if padded:
            from blueprint_pipeline.atlas_context_fov import square_pad
            pixels, transform = square_pad(np.asarray(upright))
            padded_path = pose_root / "padded" / f"{frame_id}.png"
            padded_path.parent.mkdir(exist_ok=True)
            Image.fromarray(pixels).save(padded_path)
            sources.append({"frame_id": frame_id, "image_path": str(upright_path.resolve()),
                            "image_digest": _sha256_file(upright_path), "padding": transform})
            frames[-1].update(image_path=str(padded_path.resolve()), image_digest=_sha256_file(padded_path))
        # Atlas's reframed RGB: the upright view covered at 1.5x and centre-cropped.
        atlas_rgb = np.asarray(upright.resize(VIEW, Image.Resampling.LANCZOS))[3:39]
        out = pose_root / "atlas_pose" / "outputs" / f"{index:06d}"
        out.mkdir(parents=True)
        cv2.imwrite(str(out / "image.png"), atlas_rgb[..., ::-1])
        write_exr(out / "depth.exr", plane_depth(rotation_cv, position))
        cv2.imwrite(str(out / "confidence.png"), np.full((GRID[1], GRID[0]), 200, np.uint8))
        outputs.append([{"role": role, "asset_id": f"asset_{role}_{index}", "sha256": _sha256_file(out / name),
                         "path": f"atlas_pose/outputs/{index:06d}/{name}", "bytes": (out / name).stat().st_size}
                        for role, name in (("image", "image.png"), ("depth", "depth.exr"),
                                           ("confidence", "confidence.png"))])
        sent.append({"imageAsset": {"assetId": f"asset_in_{index}"}})
        returned.append({"imageAsset": {"assetId": f"asset_image_{index}"},
                         "camera": camera_payload(rotation_cv, position, "rub"),
                         "depth": {"depthAsset": {"assetId": f"asset_depth_{index}"},
                                   "confidenceAsset": {"assetId": f"asset_confidence_{index}"}}})
    inputs = digest({"schema_version": INPUT_SCHEMA, "binding": {
        "source_video_digest": SHA, "maximum_frames": 3, "adapter_version": 3, "survey_digest": survey_digest},
        "frames": frames, "source_sample_count": 9, "heldout_frame_count": 1, "heldout_pixels_included": False})
    if padded:
        inputs["full_view_sources"] = sources
        digest(inputs)
    write(pose_root / "atlas_geometry_inputs.json", inputs)
    binding = {"source_video_digest": SHA, "geometry_input_digest": inputs["digest"]}
    identities = [{"frame_id": row["frame_id"], "image_digest": row["image_digest"]} for row in frames]
    receipt_binding = {"source_binding": binding, "frames": identities, "request_options": pose.request_options(),
                       "target_resolution": list(GRID), "provider": pose.PROVIDER}
    receipt = {"schema_version": pose.RECEIPT_SCHEMA, "operation_id": "op_source",
               "request_digest": canonical_digest(receipt_binding), "binding": receipt_binding,
               "request": {"frames": sent, "targetResolution": list(GRID), **pose.request_options()},
               "input_usage": {"submitted_frame_count": 3, "frames": [
                   {**identity, "asset_id": item["imageAsset"]["assetId"], "content_type": "image/png", "bytes": 1,
                    "outputs": rows} for identity, item, rows in zip(identities, sent, outputs)]}}
    result = write(pose_root / "atlas_posed_rgbd_result.json", digest({
        "schema_version": "website_atlas_pose_result.v1", "endpoint": pose.ATLAS_POSE_ENDPOINT, "binding": binding,
        "operation_receipt": receipt, "claim_ceiling": "development_only",
        "operation": {"id": "op_source", "done": True, "error": None,
                      "response": {"poseRefinementAccepted": True, "frames": returned}}}))
    write(pose_root / "atlas_pose" / "controller_admission.json",
          {"status": "admitted", "provider": pose.PROVIDER, "allocation_binding_digest": receipt["request_digest"]})
    write(pose_root / "atlas_pose" / "submission.json", {"status": "completed", "operation_id": "op_source",
          "request_digest": receipt["request_digest"], "result_digest": result["digest"]})
    return inputs, result


def background_views(job_dir: Path, cameras):
    """The edited views' returned RGBD in the background world: same pixels, depth in its units."""
    outputs = []
    for index, (rotation_cv, position) in enumerate(cameras):
        source_out = job_dir / "pose" / "atlas_pose" / "outputs" / f"{index:06d}"
        out = job_dir / "automation" / "fallback_pose-v2" / "atlas_pose" / "outputs" / f"{index:06d}"
        out.mkdir(parents=True)
        for name in ("image.png", "confidence.png"):
            (out / name).write_bytes((source_out / name).read_bytes())
        write_exr(out / "depth.exr", BACKGROUND_SCALE * plane_depth(rotation_cv, position))
        outputs.append([{"role": role, "path": f"atlas_pose/outputs/{index:06d}/{name}",
                         "sha256": _sha256_file(out / name)}
                        for role, name in (("image", "image.png"), ("depth", "depth.exr"),
                                           ("confidence", "confidence.png"))])
    return outputs


def job_fixture(capture_root: Path, *, same_pose=False, padded=False, cameras=CAMERAS, views=False,
                room_rotation=np.eye(3), room=None):
    job_dir = capture_root / "pipeline" / "atlas"
    survey = write(job_dir / "survey" / "atlas_video_survey.json", digest({
        "schema_version": "atlas_video_survey.v1", "display_rotation_degrees": ROTATION, "frames": []}))
    write(job_dir / "selection_binding.json", {"survey_relative_path": "survey/atlas_video_survey.json"})
    _, source_pose = pose_root_fixture(job_dir / "pose", survey["digest"], padded=padded, cameras=cameras)
    measurements = write(job_dir / "pose" / "scene_measurements.json", digest({
        "schema_version": "atlas_scene_measurements.v1", "status": "estimated", "meters_per_unit_estimate": MPU,
        "scale_status": "estimated_from_named_assumptions", "metric_measurement_proven": False}))
    if same_pose:
        background_pose = source_pose
    else:
        # The edited views posed again: the same cameras in another Atlas world.
        moved = [{"camera": camera_payload(BACKGROUND_ROTATION @ rotation_cv,
                                           BACKGROUND_SCALE * BACKGROUND_ROTATION @ position + BACKGROUND_TRANSLATION,
                                           "rub")} for rotation_cv, position in cameras]
        outputs = background_views(job_dir, cameras) if views else [None] * len(FRAME_IDS)
        background_pose = write(job_dir / "automation" / "fallback_pose-v2" / "atlas_posed_rgbd_result.json", digest({
            "schema_version": "website_atlas_pose_result.v1", "operation": {"id": "op_background",
            "response": {"frames": moved}}, "operation_receipt": {"input_usage": {"frames": [
                {"frame_id": frame_id, **({"outputs": rows} if rows else {})}
                for frame_id, rows in zip(FRAME_IDS, outputs)]}}}))
    splat_root = job_dir / "automation" / "background_splat"
    mesh_root = job_dir / "automation" / "background_mesh" / "registered"
    # A room in the background world, its floor through the origin below the cameras.
    write_room(splat_root / "atlas_depth_splat.ply", mesh_root / "atlas_mesh_supported.glb",
               rotation=room_rotation, meters_per_unit=MPU if same_pose else MPU / BACKGROUND_SCALE,
               noise_m=0.0, **(room or {}))
    splat = write(splat_root / "atlas_depth_splat_manifest.json", digest({
        "pose_result_digest": background_pose["digest"], "world_frame": "atlas_world_rub_y_up",
        "outputs": [{"role": "splat_ply", "path": "atlas_depth_splat.ply",
                     "sha256": _sha256_file(splat_root / "atlas_depth_splat.ply")}]}))
    write(mesh_root / "atlas_mesh_registration.json", digest({
        "splat_manifest_digest": splat["digest"], "outputs": [{"role": "supported_mesh_glb",
        "path": "atlas_mesh_supported.glb", "sha256": _sha256_file(mesh_root / "atlas_mesh_supported.glb")}]}))
    result = write(job_dir / "result.json", digest({
        "schema_version": "website_atlas_pipeline_result.v1", "status": "needs_input",
        "automation": {"status": "completed", "background": {
            "status": "completed", "splat_manifest_digest": splat["digest"], "mesh_available": True}},
        "source_pose": {"status": "completed", "result_digest": source_pose["digest"],
                        "scene_measurements_digest": measurements["digest"]}}))
    write(job_dir / "preview_job.json", {"state": "ready", "result_digest": result["digest"]})
    return job_dir


def ready(tmp_path, **options):
    job_fixture(tmp_path, **options)
    value = source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert value["status"] == "ready"
    return value


def test_the_background_is_the_scene_aligned_to_the_scaled_source_pose_by_shared_cameras(tmp_path):
    value = ready(tmp_path)
    base = value["base_scene"]
    assert base["provider"] == "world_labs_atlas" and base["up_axis"] == "Y"
    # The background is already level: its copy keeps the frame, and the floor is declared.
    assert base["scale_authority"] == "atlas_reference_estimate" and base["ground_plane_offset_m"] == 0.0
    assert base["levelling"]["up_basis"] == "observed_floor" and base["levelling"]["tilt_degrees"] < 1e-6
    assert base["splat_digest"] == _sha256_file(Path(base["splat_path"]))
    assert base["splat_binding_id"] == "website-splat-" + base["splat_digest"][7:39]
    assert base["collision_mesh_path"].endswith("atlas_mesh_levelled.glb")
    retained = tmp_path / "pipeline/atlas/automation"
    assert base["levelling"]["provider_splat_digest"] == _sha256_file(retained / "background_splat/atlas_depth_splat.ply")
    assert base["levelling"]["provider_collision_mesh_digest"] == _sha256_file(
        retained / "background_mesh/registered/atlas_mesh_supported.glb")
    # Metres per background unit: the source scale divided by the worlds' relative scale.
    assert base["meters_per_unit"] == pytest.approx(MPU / BACKGROUND_SCALE)
    anchor = base["anchor"]
    assert anchor["kind"] == "source_world_alignment" and anchor["basis"] == "shared_atlas_cameras"
    assert anchor["shared_frame_count"] == 3 and anchor["max_rotation_residual_degrees"] < 1e-4
    transform = np.asarray(anchor["source_world_to_provider_m"])
    assert np.allclose(transform[:3, :3], BACKGROUND_ROTATION, atol=1e-6)
    assert np.allclose(transform[:3, 3], BACKGROUND_TRANSLATION * base["meters_per_unit"], atol=1e-6)
    # A source point in metres reaches the background world exactly.
    point_units = np.array([0.2, -0.1, 4.0])
    runtime = (transform[:3, :3] @ (point_units * MPU) + transform[:3, 3]) / base["meters_per_unit"]
    assert np.allclose(runtime, BACKGROUND_SCALE * BACKGROUND_ROTATION @ point_units + BACKGROUND_TRANSLATION)
    assert value["alignment"] == "anchored" and base["operation_id"] == "op_background"


def test_a_background_fused_on_the_source_pose_shares_its_world(tmp_path):
    anchor = ready(tmp_path, same_pose=True)["base_scene"]["anchor"]
    assert anchor["basis"] == "same_atlas_pose"
    assert np.allclose(anchor["source_world_to_provider_m"], np.eye(4), atol=1e-9)


def test_a_tilted_background_is_levelled_and_the_source_still_lands_in_it(tmp_path):
    base = ready(tmp_path, room_rotation=tilt(15.0))["base_scene"]
    assert base["levelling"]["tilt_degrees"] == pytest.approx(15.0, abs=1e-3)
    assert base["levelling"]["up_basis"] == "observed_floor" and base["ground_plane_offset_m"] == 0.0
    # A source point in metres lands where the levelled copy put the background point.
    lift = np.asarray(base["levelling"]["provider_to_levelled"])
    transform = np.asarray(base["anchor"]["source_world_to_provider_m"])
    point_units = np.array([0.2, -0.1, 4.0])
    runtime = (transform[:3, :3] @ (point_units * MPU) + transform[:3, 3]) / base["meters_per_unit"]
    background = BACKGROUND_SCALE * BACKGROUND_ROTATION @ point_units + BACKGROUND_TRANSLATION
    assert np.allclose(runtime, lift[:3, :3] @ background + lift[:3, 3])
    import trimesh
    vertices = trimesh.load(base["collision_mesh_path"], force="mesh", process=False).vertices
    floor_count = len(room_parts()["floor"][0])
    assert np.abs(vertices[:floor_count, 1]).max() < 1e-4


def test_a_background_with_no_level_surface_waits_for_input(tmp_path):
    job_fixture(tmp_path, room={"floor": False, "counter": False})
    value = source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert value == {"status": "needs_input", "reason": "website_atlas_scene_level_unobserved"}


def completed_bundle(job_dir: Path, *, retained_views=False):
    """The final portable download; a retained handoff also moves the background pose's views into it."""
    import shutil
    final = job_dir / "automation/downloads/completed-scene"
    shutil.copytree(job_dir / "automation/background_splat", final / "splat")
    shutil.copytree(job_dir / "automation/background_mesh/registered", final / "mesh/registered")
    ply, glb = final / "splat/atlas_depth_splat.ply", final / "mesh/registered/atlas_mesh_supported.glb"
    # Completed generated geometry: the room with a table the baseline lacked.
    write_room(ply, glb, meters_per_unit=MPU / BACKGROUND_SCALE, noise_m=0.0, table=True)
    splat_path = final / "splat/atlas_depth_splat_manifest.json"
    splat = json.loads(splat_path.read_text())
    splat["outputs"][0]["sha256"] = _sha256_file(ply)
    write(splat_path, digest(splat))
    reg_path = final / "mesh/registered/atlas_mesh_registration.json"
    registration = json.loads(reg_path.read_text())
    registration["splat_manifest_digest"] = splat["digest"]
    registration["outputs"][0]["sha256"] = _sha256_file(glb)
    write(reg_path, digest(registration))
    assets = []
    if retained_views:
        # Only the pose result stays beside the pose; its outputs travel in the bundle.
        for path in sorted((job_dir / "automation/fallback_pose-v2/atlas_pose/outputs").rglob("*.*")):
            target = final / "views" / f"{path.parent.name}-{path.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
            assets.append({"path": str(target.relative_to(final)), "sha256": _sha256_file(target)})
    bundle = digest({"schema_version": "atlas_download_bundle.v1", "splat_available": True,
        "mesh_available": True, "splat_manifest_digest": splat["digest"],
        "mesh_registration_digest": registration["digest"], "assets": assets})
    write(final / "manifest.json", bundle)
    result = json.loads((job_dir / "result.json").read_text())
    result["automation"]["background"].update(splat_manifest_digest=splat["digest"],
        download={"directory": str(final), "manifest_digest": bundle["digest"]})
    write(job_dir / "result.json", digest(result))
    write(job_dir / "preview_job.json", {"state": "ready", "result_digest": result["digest"]})
    return final, bundle, ply, glb


@pytest.mark.parametrize("tamper_bundle", [False, True])
def test_scene_source_uses_the_final_completed_bundle_with_a_stale_baseline_present(tmp_path, tamper_bundle):
    job_dir = job_fixture(tmp_path)
    final, bundle, ply, glb = completed_bundle(job_dir)
    if tamper_bundle:
        bundle["mesh_available"] = False
        write(final / "manifest.json", bundle)
        with pytest.raises(ValueError, match="website_atlas_scene_download_changed"):
            source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    else:
        value = source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
        assert value["status"] == "ready"
        levelling = value["base_scene"]["levelling"]
        assert Path(levelling["provider_splat_path"]) == ply and levelling["provider_splat_digest"] == _sha256_file(ply)
        assert Path(levelling["provider_collision_mesh_path"]) == glb
        assert levelling["provider_collision_mesh_digest"] == _sha256_file(glb)


def test_scene_source_accepts_native_splat_names_in_the_same_bound_bundle(tmp_path):
    job_dir = job_fixture(tmp_path)
    final, bundle, ply, _ = completed_bundle(job_dir)
    prior_path = final / "splat/atlas_depth_splat_manifest.json"
    prior = json.loads(prior_path.read_text())
    native_ply = ply.with_name("atlas_refined_splat.ply")
    ply.rename(native_ply)
    prior["outputs"][0]["path"] = native_ply.name
    prior.update(schema_version="atlas_refined_splat.v1", task="posedImages2Splats",
                 splat_kind="atlas_refined", source_geometry_preservation_proven=False)
    native_manifest = digest(prior)
    write(final / "splat/atlas_refined_splat_manifest.json", native_manifest)
    # Leave the obsolete manifest present, as it can be in a retained revision.
    reg_path = final / "mesh/registered/atlas_mesh_registration.json"
    registration = json.loads(reg_path.read_text())
    registration["splat_manifest_digest"] = native_manifest["digest"]
    write(reg_path, digest(registration))
    bundle.update(splat_manifest_digest=native_manifest["digest"], mesh_registration_digest=registration["digest"])
    write(final / "manifest.json", digest(bundle))
    result = json.loads((job_dir / "result.json").read_text())
    result["automation"]["background"].update(splat_manifest_digest=native_manifest["digest"],
        download={"directory": str(final), "manifest_digest": bundle["digest"]})
    write(job_dir / "result.json", digest(result))
    write(job_dir / "preview_job.json", {"state": "ready", "result_digest": result["digest"]})
    value = source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert value["status"] == "ready"
    assert value["base_scene"]["levelling"]["provider_splat_path"] == str(native_ply)


@pytest.mark.parametrize("change", [None, "video", "context", "source_pose", "digest", "outside",
                                    "missing_path", "missing_digest", "tamper"])
def test_retained_revision_reuses_only_the_same_capture_observations(tmp_path, change):
    job_dir = job_fixture(tmp_path)
    original = json.loads((job_dir / "result.json").read_text())
    original.update(source_video_digest=SHA, task_context_digest="sha256:" + "b" * 64)
    write(job_dir / "result.json", digest(original))
    write(job_dir / "preview_job.json", {"state": "ready", "result_digest": original["digest"]})
    original_job = (job_dir / "preview_job.json").read_bytes()
    original_result = (job_dir / "result.json").read_bytes()
    _, _, ply, glb = completed_bundle(job_dir)
    retained = json.loads((job_dir / "result.json").read_text())
    (job_dir / "result.json").write_bytes(original_result)
    (job_dir / "preview_job.json").write_bytes(original_job)
    if change == "video":
        retained["source_video_digest"] = "sha256:" + "c" * 64
    elif change == "context":
        retained["task_context_digest"] = "sha256:" + "c" * 64
    elif change == "source_pose":
        retained["source_pose"]["scene_measurements_digest"] = "sha256:" + "c" * 64
    digest(retained)
    path = (tmp_path if change == "outside" else job_dir / "retained-results") / "revision.json"
    write(path, retained)
    expected = "sha256:" + "0" * 64 if change == "digest" else retained["digest"]
    args = dict(capture_root=tmp_path, output_root=tmp_path / "replay",
                retained_result_path=path, retained_result_digest=expected)
    if change == "missing_path":
        args["retained_result_path"] = None
    elif change == "missing_digest":
        args["retained_result_digest"] = None
    elif change == "tamper":
        retained["status"] = "changed_after_sealing"
        write(path, retained)
    if change is not None:
        with pytest.raises(ValueError, match="website_atlas_scene_(retained_result_changed|retained_selection_incomplete|artifact_missing)"):
            source.atlas_scene_source(**args)
    else:
        value = source.atlas_scene_source(**args)
        assert value["status"] == "ready"
        assert value["base_scene"]["levelling"]["provider_splat_path"] == str(ply)
        assert value["base_scene"]["levelling"]["provider_collision_mesh_path"] == str(glb)
        assert value["base_scene"]["atlas_result_digest"] == retained["digest"]
    assert (job_dir / "preview_job.json").read_bytes() == original_job
    assert (job_dir / "result.json").read_bytes() == original_result


# Turning in place leaves the shared cameras no baseline to fix a similarity.
TURNING = [(rotation_cv, np.array([0.1, -0.05, 0.0])) for rotation_cv, _ in CAMERAS]


@pytest.mark.parametrize("retained", [False, True])
def test_cameras_that_fix_no_similarity_defer_to_the_depth_of_the_shared_views(tmp_path, retained):
    completed_bundle(job_fixture(tmp_path, cameras=TURNING, views=True), retained_views=retained)
    value = source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    base = value["base_scene"]
    anchor = base["anchor"]
    assert value["alignment"] == "anchored" and anchor["basis"] == "shared_view_depth"
    assert anchor["registered_frame_count"] == anchor["agreeing_frame_count"] == 3
    assert base["meters_per_unit"] == pytest.approx(MPU / BACKGROUND_SCALE)
    transform = np.asarray(anchor["source_world_to_provider_m"])
    assert np.allclose(transform[:3, :3], BACKGROUND_ROTATION, atol=1e-6)
    assert np.allclose(transform[:3, 3], BACKGROUND_TRANSLATION * base["meters_per_unit"], atol=1e-6)


def test_a_background_without_retained_views_leaves_registration_to_scene_preparation(tmp_path):
    completed_bundle(job_fixture(tmp_path, cameras=TURNING))
    value = source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert value["alignment"] == "registration_search" and value["base_scene"]["anchor"] is None


@pytest.mark.parametrize("invalid_prior", [None, "digest", "video"])
def test_expanded_source_bridges_retained_same_video_depth_without_direct_overlap(tmp_path, invalid_prior):
    import copy
    import shutil

    job = job_fixture(tmp_path, cameras=TURNING, views=True)
    old_path = job / "pose/atlas_posed_rgbd_result.json"
    prior = json.loads(old_path.read_text())
    new_root = job / "pose/expanded"
    shutil.copytree(job / "pose/atlas_pose", new_root / "atlas_pose")
    expanded = copy.deepcopy(prior)
    new_scale, new_rotation, new_translation = 1.4, yaw(-31.0), np.array([0.5, -0.2, 0.8])
    expanded["operation"]["response"]["frames"] = expanded["operation"]["response"]["frames"][:2]
    expanded["operation_receipt"]["input_usage"]["frames"] = expanded["operation_receipt"]["input_usage"]["frames"][:2]
    for index, (frame, used) in enumerate(zip(expanded["operation"]["response"]["frames"],
                                            expanded["operation_receipt"]["input_usage"]["frames"])):
        rotation, position = TURNING[index]
        frame["camera"] = camera_payload(new_rotation @ rotation,
            new_scale * new_rotation @ position + new_translation, "rub")
        row = next(row for row in used["outputs"] if row["role"] == "depth")
        write_exr(new_root / row["path"], new_scale * plane_depth(rotation, position))
        row["sha256"] = _sha256_file(new_root / row["path"])
    digest(expanded)
    bg_root = job / "automation/fallback_pose-v2"
    background = json.loads((bg_root / "atlas_posed_rgbd_result.json").read_text())
    background["operation"]["response"]["frames"] = background["operation"]["response"]["frames"][1:]
    background["operation_receipt"]["input_usage"]["frames"] = background["operation_receipt"]["input_usage"]["frames"][1:]
    digest(background)
    options = dict(job_dir=job, source_root=new_root, source=expanded,
                   background_root=bg_root, background=background, bundle_root=None, bundle=None)
    assert source._shared_view_alignment(**options) is None  # Only one view overlaps directly.
    if invalid_prior:
        prior["binding"]["source_video_digest"] = "sha256:" + "e" * 64
        if invalid_prior == "video":
            digest(prior)
        write(old_path, prior)
    alignment = source._retained_pose_alignment(**options)
    if invalid_prior:
        assert alignment is None
    else:
        assert alignment["intermediate_pose_digest"] == prior["digest"]
        assert [leg["agreeing_frame_count"] for leg in alignment["alignment_legs"]] == [2, 2]
        assert alignment["scale"] == pytest.approx(BACKGROUND_SCALE / new_scale)
        rotation = BACKGROUND_ROTATION @ new_rotation.T
        np.testing.assert_allclose(alignment["rotation"], rotation, atol=1e-6)
        np.testing.assert_allclose(alignment["translation"],
            BACKGROUND_TRANSLATION - (BACKGROUND_SCALE / new_scale) * rotation @ new_translation, atol=1e-6)


def test_the_source_views_become_scaled_geometry_on_the_atlas_grid(tmp_path):
    geometry = ready(tmp_path)["source_geometry"]
    assert geometry["unit"] == "estimated_meters" and geometry["scale_status"] == "reference_scaled"
    assert geometry["binding"]["scale"]["basis"] == "reference_estimate"
    assert geometry["binding"]["scale"]["evidence_reference"] == "pipeline/atlas/pose/scene_measurements.json"
    for index, frame in enumerate(geometry["frames"]):
        assert (frame["width"], frame["height"]) == VIEW and frame["frame_id"] == FRAME_IDS[index]
        assert (frame["source_width"], frame["source_height"]) == UPRIGHT[::-1]
        assert frame["atlas_frame"]["pixel_binding_ncc"] > 0.999
        with np.load(frame["geometry_path"]) as data:
            valid, depth = data["valid_mask"], data["depth_m"]
        assert not valid[:3].any() and not valid[-3:].any() and valid[3:-3].all()
        # Each valid geometry pixel is the plane, in metres.
        rows, columns = np.nonzero(valid)
        rays = np.stack([columns, rows, np.ones_like(rows)], axis=1) @ np.linalg.inv(frame["intrinsics"]).T
        pose_matrix = np.asarray(frame["world_from_camera"])
        world = (rays * depth[rows, columns, None]) @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]
        np.testing.assert_allclose(world @ PLANE_NORMAL, PLANE_OFFSET * MPU, rtol=1e-4)
        # The source pixel transform follows the portrait rotation onto the view grid.
        mapped = np.asarray(frame["source_to_geometry_pixels"]) @ np.array([0.0, 0.0, 1.0])
        assert mapped[:2] == pytest.approx([1.5 * 0 + 0.25, 1.5 * (UPRIGHT[1] - 1) + 0.25])


def continuous_masks(rows=None, *, width=UPRIGHT[0], height=UPRIGHT[1]):
    registry = rows or [{"source_frame_id": frame_id, "decoded_pts_seconds": time, "width": width, "height": height}
                        for frame_id, time in zip(FRAME_IDS, TIMES)]
    mask = np.zeros((height, width), dtype=np.int8)
    mask[10:18, 15:25] = 1
    edges = np.flatnonzero(np.diff(np.pad(mask.reshape(-1), (1, 1))))
    observation = {"width": width, "height": height,
                   "runs": [{"start": int(a), "length": int(b - a)} for a, b in zip(edges[0::2], edges[1::2])]}
    return digest({"binding": {"mask_input_pixels": "continuous_source_video_v1",
                               "geometry_input_digest": "sha256:" + "9" * 64},
                   "source_video_digest": SHA, "source_frame_registry": registry,
                   "targets": [{"target_id": "object", "source_track": {"observations": [
                       {**observation, "source_frame_id": frame_id} for frame_id in FRAME_IDS[:2]]}}]})


@pytest.mark.parametrize("mode", ["continuous_source_video_v1", "selected_original_views_v1"])
def test_tracks_on_the_source_timeline_lift_onto_atlas_geometry_by_frame_identity(tmp_path, mode):
    geometry = ready(tmp_path)["source_geometry"]
    masks = continuous_masks()
    masks["binding"]["mask_input_pixels"] = mode
    bound = bind_task_masks_to_geometry(task_masks=digest(masks), source_geometry=geometry)
    target = bound["targets"][0]
    assert {row["source_frame_id"] for row in target["track"]["observations"]} == set(FRAME_IDS[:2])
    points = masked_source_points(target["track"], geometry["frames"])
    np.testing.assert_allclose(points @ PLANE_NORMAL, PLANE_OFFSET * MPU, rtol=1e-4)
    assert target["estimated_visible_bounds"]["unit"] == "estimated_meters"


def off_geometry(target, **fields):
    """The target seen only on a source frame the geometry views do not include."""
    observation = {**target["source_track"]["observations"][0], "source_frame_id": "decoded-000000099"}
    return {**target, **fields, "source_track": {"observations": [observation]}}


def test_a_removed_person_needs_no_geometry_but_a_task_object_does(tmp_path):
    geometry = ready(tmp_path)["source_geometry"]
    masks = continuous_masks()
    # Geometry views are chosen without people, so the operator is never on them.
    masks["targets"].append(off_geometry(masks["targets"][0], target_id="person_operator", target_class="person",
                                         target_role="person", task_effect="privacy", disposition="remove"))
    bound = bind_task_masks_to_geometry(task_masks=digest(masks), source_geometry=geometry)
    by_id = {target["target_id"]: target for target in bound["targets"]}
    assert by_id["person_operator"]["estimated_visible_bounds"] is None
    assert not by_id["person_operator"]["track"]["observations"]
    assert by_id["object"]["estimated_visible_bounds"]["unit"] == "estimated_meters"
    masks = continuous_masks()
    masks["targets"][0] = off_geometry(masks["targets"][0])
    with pytest.raises(ValueError, match="task_target_has_no_estimated_geometry"):
        bind_task_masks_to_geometry(task_masks=digest(masks), source_geometry=geometry)


@pytest.mark.parametrize("change", ["time", "size", "sampled"])
def test_tracks_that_cannot_name_the_geometry_frames_are_refused(tmp_path, change):
    geometry = ready(tmp_path)["source_geometry"]
    if change == "time":
        masks = continuous_masks([{"source_frame_id": frame_id, "decoded_pts_seconds": time + 0.01,
                                   "width": UPRIGHT[0], "height": UPRIGHT[1]}
                                  for frame_id, time in zip(FRAME_IDS, TIMES)])
    elif change == "size":
        masks = continuous_masks(width=UPRIGHT[1], height=UPRIGHT[0])
    else:
        masks = continuous_masks()
        masks["binding"]["mask_input_pixels"] = "geometry_v1"
        masks = digest(masks)
    with pytest.raises(ValueError, match="website_task_geometry_binding_mismatch"):
        bind_task_masks_to_geometry(task_masks=masks, source_geometry=geometry)


@pytest.mark.parametrize("case, reason", [
    ("missing", "website_atlas_job_required"),
    ("processing", "website_atlas_job_processing"),
    ("failed", "website_atlas_job_failed"),
    ("background", "website_atlas_background_unavailable"),
    ("mesh", "website_atlas_background_mesh_unavailable"),
    ("source_pose", "website_atlas_source_pose_unavailable"),
    ("scale", "website_atlas_scale_estimate_unavailable"),
    ("references_disagree", "website_atlas_scale_references_disagree"),
])
def test_an_unready_job_says_what_scene_preparation_waits_for(tmp_path, case, reason):
    job_dir = job_fixture(tmp_path)
    job = json.loads((job_dir / "preview_job.json").read_text())
    result = json.loads((job_dir / "result.json").read_text())
    if case == "missing":
        (job_dir / "preview_job.json").unlink()
    elif case in {"processing", "failed"}:
        write(job_dir / "preview_job.json", {**job, "state": case})
    elif case in {"scale", "references_disagree"}:
        agreement = {"reference_agreement": {"basis": "no_majority"}} if case == "references_disagree" else {}
        write(job_dir / "pose" / "scene_measurements.json", digest({"status": "relative_sizes_only", **agreement}))
        result["source_pose"]["scene_measurements_digest"] = json.loads(
            (job_dir / "pose" / "scene_measurements.json").read_text())["digest"]
    else:
        if case == "background":
            result["automation"]["background"]["status"] = "needs_input"
        elif case == "mesh":
            result["automation"]["background"]["mesh_available"] = False
        else:
            result["source_pose"] = {"status": "needs_input", "reason": "atlas_runtime_price_review_expired"}
    if case in {"scale", "references_disagree", "background", "mesh", "source_pose"}:
        write(job_dir / "result.json", digest(result))
        write(job_dir / "preview_job.json", {**job, "result_digest": result["digest"]})
    value = source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert value["status"] == ("pending" if case == "processing" else "needs_input") and value["reason"] == reason


@pytest.mark.parametrize("case, error", [
    ("result", "website_atlas_scene_result_changed"),
    ("splat", "website_atlas_scene_artifact_changed"),
    ("measurements", "website_atlas_scene_measurements_changed"),
    ("survey", "website_atlas_scene_survey_changed"),
])
def test_changed_job_outputs_are_refused(tmp_path, case, error):
    job_dir = job_fixture(tmp_path)
    if case == "result":
        result = json.loads((job_dir / "result.json").read_text())
        write(job_dir / "result.json", {**result, "status": "completed"})
    elif case == "splat":
        (job_dir / "automation/background_splat/atlas_depth_splat.ply").write_bytes(b"changed")
    elif case == "measurements":
        write(job_dir / "pose" / "scene_measurements.json", digest({"status": "estimated",
                                                                    "meters_per_unit_estimate": 9.0}))
    else:
        write(job_dir / "survey" / "atlas_video_survey.json", digest({"display_rotation_degrees": 0.0}))
    with pytest.raises(ValueError, match=error):
        source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")


def test_world_alignment_needs_agreeing_shared_cameras():
    cameras = {frame_id: np.eye(4) for frame_id in FRAME_IDS}
    for index, frame_id in enumerate(FRAME_IDS):
        cameras[frame_id][:3, 3] = [index, 0.5 * index, 0.0]
    assert source.world_alignment(cameras, {FRAME_IDS[0]: cameras[FRAME_IDS[0]]}) is None
    turned = {key: value.copy() for key, value in cameras.items()}
    turned[FRAME_IDS[1]][:3, :3] = yaw(30.0)
    assert source.world_alignment(cameras, turned) is None
    found = source.world_alignment(cameras, cameras)
    assert found["scale"] == pytest.approx(1.0) and np.allclose(found["rotation"], np.eye(3))


def measured_by(job_dir: Path, **fields):
    """Rewrite the job's source measurements with ``fields`` and bind the job's result to them."""
    measurements = write(job_dir / "pose" / "scene_measurements.json", digest({
        "schema_version": "atlas_scene_measurements.v1", "status": "estimated", "meters_per_unit_estimate": MPU,
        "scale_status": "estimated_from_named_assumptions", "metric_measurement_proven": False, **fields}))
    result = json.loads((job_dir / "result.json").read_text())
    result["source_pose"]["scene_measurements_digest"] = measurements["digest"]
    write(job_dir / "result.json", digest(result))
    job = json.loads((job_dir / "preview_job.json").read_text())
    write(job_dir / "preview_job.json", {**job, "result_digest": result["digest"]})


CAMERA_ONLY = {"meters_per_unit_range": [MPU * 1.2 / 1.4, MPU * 1.6 / 1.4],
               "reference_agreement": {"basis": "camera_height_prior_only", "object_count": 1,
                                       "agreeing_object_count": 1}}


def prepared(tmp_path):
    value = source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert value["status"] == "ready"
    return value


def test_the_scale_record_carries_its_range_and_how_the_cues_agreed(tmp_path):
    measured_by(job_fixture(tmp_path), **CAMERA_ONLY)
    value = prepared(tmp_path)
    scale = value["source_geometry"]["binding"]["scale"]
    assert scale["meters_per_unit_range"] == pytest.approx(CAMERA_ONLY["meters_per_unit_range"])
    assert scale["agreement_basis"] == "camera_height_prior_only"
    assert scale["meters_per_unit"] == MPU and scale["basis"] == "reference_estimate"
    # Nothing else moves: the same metres per unit reach the scene.
    assert value["base_scene"]["meters_per_unit"] == pytest.approx(MPU / BACKGROUND_SCALE)


def test_geometry_made_before_its_scale_carried_a_range_is_kept(tmp_path):
    measured_by(job_fixture(tmp_path), **CAMERA_ONLY)
    first = prepared(tmp_path)["source_geometry"]
    # The geometry as retained before the record carried its range and basis.
    manifest = Path(first["manifest_path"])
    retained = json.loads(manifest.read_text())
    retained["binding"]["scale"] = {key: value for key, value in retained["binding"]["scale"].items()
                                    if key not in ("meters_per_unit_range", "agreement_basis")}
    manifest.write_text(json.dumps(digest(retained)))
    before = manifest.read_bytes()
    kept = prepared(tmp_path)["source_geometry"]
    assert manifest.read_bytes() == before and kept["binding"] == retained["binding"]
    assert set(kept["binding"]["scale"]) == {"meters_per_unit", "basis", "evidence_digest", "evidence_reference"}


@pytest.mark.parametrize("change", [{"meters_per_unit_range": [MPU * 1.1, MPU * 1.2]},
                                    {"meters_per_unit_range": [MPU, float("nan")]},
                                    {"agreement_basis": " "}, {"meters_per_unit_unknown": 1.0}])
def test_a_scale_record_with_an_inconsistent_range_or_unknown_field_is_refused(tmp_path, change):
    job_fixture(tmp_path)
    scale = {"meters_per_unit": MPU, "basis": "reference_estimate", "evidence_digest": "sha256:" + "0" * 64,
             "evidence_reference": "pipeline/atlas/pose/scene_measurements.json", **change}
    with pytest.raises(ValueError, match="atlas_source_geometry_scale_invalid"):
        atlas_posed_view_geometry(pose_root=tmp_path / "pipeline/atlas/pose", rotation=ROTATION, scale=scale,
                                  output_root=tmp_path / "out")


def test_posed_view_geometry_requires_a_scale(tmp_path):
    job_fixture(tmp_path)
    with pytest.raises(ValueError, match="atlas_source_geometry_scale_invalid"):
        atlas_posed_view_geometry(pose_root=tmp_path / "pipeline/atlas/pose", rotation=ROTATION, scale=None,
                                  output_root=tmp_path / "out")
