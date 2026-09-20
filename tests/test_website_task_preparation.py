from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

NOW = 1_800_000_000.0
WIDTH, HEIGHT, FOCAL = 40, 30, 30.0
RUNTIME_ROTATION = np.diag([1.0, -1.0, -1.0])  # source Y-down/Z-forward -> Y-up runtime
RUNTIME_SCALE, RUNTIME_TRANSLATION = 0.5, np.array([1.0, 2.0, 3.0])


def _depth():
    depth = np.full((HEIGHT, WIDTH), 3.0)
    rows = np.arange(HEIGHT)[:, None]
    table = rows > HEIGHT / 2
    depth = np.where(table, 0.5 * FOCAL / np.maximum(rows - HEIGHT / 2, 1e-6), depth)
    return depth


def _frame(root: Path, index: int):
    depth = _depth()
    geometry = root / f"frame-{index}.npz"
    np.savez(geometry, depth_m=depth, valid_mask=np.ones_like(depth, dtype=bool))
    image = root / f"frame-{index}.png"
    Image.fromarray(np.full((HEIGHT, WIDTH, 3), 120, dtype=np.uint8)).save(image)
    pose = np.eye(4)
    pose[0, 3] = 0.05 * index
    return {"frame_id": f"frame-{index}", "timestamp_seconds": index * 0.5, "image_path": str(image),
            "image_digest": _sha256_file(image), "geometry_path": str(geometry), "geometry_digest": _sha256_file(geometry),
            "intrinsics": [[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]],
            "world_from_camera": pose.tolist(), "width": WIDTH, "height": HEIGHT}


def _source_geometry(root: Path):
    frames = [_frame(root, index) for index in range(2)]
    value = {"schema_version": "website_source_geometry.v1", "frames": frames, "unit": "estimated_meters",
             "scale_status": "model_estimated", "metric_measurement_proven": False}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


def _grid(source_geometry):
    # A real connected triangle grid: arbitrary point triplets are not a
    # support surface even when their enclosing box resembles one.
    points, faces = [], []
    for frame in source_geometry["frames"]:
        yy, xx = np.indices((HEIGHT, WIDTH))
        pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3)
        camera = (pixels @ np.linalg.inv(frame["intrinsics"]).T) * _depth().reshape(-1, 1)
        pose = np.array(frame["world_from_camera"])
        offset = len(points) * WIDTH * HEIGHT
        points.append(camera @ pose[:3, :3].T + pose[:3, 3])
        for y in range(HEIGHT - 1):
            for x in range(WIDTH - 1):
                a = offset + y * WIDTH + x
                faces.extend([[a, a + 1, a + WIDTH], [a + 1, a + WIDTH + 1, a + WIDTH]])
    return np.concatenate(points), faces


def _base_scene(root: Path, source_geometry, *, symmetric=False):
    points, faces = _grid(source_geometry)
    if symmetric:
        points = np.random.default_rng(3).uniform(-1, 1, points.shape)
    vertices = (points @ RUNTIME_ROTATION.T) * RUNTIME_SCALE + RUNTIME_TRANSLATION
    mesh_path = root / "collider.glb"
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh_path)
    splat = root / "world.ply"
    splat.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
    return {"splat_path": str(splat), "splat_digest": _sha256_file(splat), "splat_binding_id": "marble-splat-1",
            "collision_mesh_path": str(mesh_path), "collision_mesh_digest": _sha256_file(mesh_path),
            "collision_binding_id": "marble-mesh-1", "up_axis": "Y", "meters_per_unit": 2.0,
            "provider": "world_labs", "operation_id": "op-1"}


def _bounds(minimum, maximum):
    return {"minimum": list(minimum), "maximum": list(maximum), "unit": "estimated_meters",
            "metric_measurement_proven": False, "complete_object_dimensions": False}


def _masks(source_geometry, *, destination=True):
    runs = [{"start": row * WIDTH + 21, "length": 4, "probability": 0.9} for row in range(20, 25)]
    track = {"track_id": "t1", "label": "cup", "observations": [
        {"source_frame_id": "frame-0", "height": HEIGHT, "width": WIDTH, "runs": runs}]}
    targets = [{"target_id": "cup-1", "task_effect": "manipulated", "disposition": "remove", "track": track,
                "estimated_visible_bounds": _bounds([0.05, 0.4, 1.45], [0.10, 0.5, 1.55])}]
    if destination:
        targets.append({"target_id": "tray-1", "task_effect": "static_contact", "disposition": "keep",
                        "target_role": "destination", "placement_relation": "on", "track": track,
                        "estimated_visible_bounds": _bounds([0.4, 0.48, 1.4], [0.6, 0.5, 1.6])})
    value = {"schema_version": "website_task_masks.v1", "status": "completed", "targets": targets,
             "source_geometry_digest": source_geometry["digest"]}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


def _task_context(confirmed_at=NOW - 100):
    value = {"schema_version": "website_site_task_context.v1", "request_id": "req1", "scene_id": "site-req1",
             "capture_id": "walkthrough-req1", "description": "Move the cup onto the tray.", "confirmed": True,
             "confirmed_at": datetime.fromtimestamp(confirmed_at, timezone.utc).isoformat(),
             "operator_answers": {}, "unresolved": [],
             "capture_rights": {"derived_scene_generation_allowed": True}}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


REMOVAL = {"schema_version": "clean_plate_removal_manifest.v1", "entries": [
    {"target_id": "cup-1", "semantic_label": "cup", "task_effect": "manipulated", "disposition": "remove",
     "compose_back": {"replacement_asset_id": None, "pose_world": None, "replacement_asset_frame_registration_uri": None}}]}
SPEND = {"max_total_spend_usd": 40, "max_paid_attempts": 2, "expires_at_epoch": NOW + 3600,
         "owner": {"user_id": "owner-1", "organization_id": "org-1"},
         "consent": {"accepted_by": "owner-1", "accepted_at_epoch": NOW - 100,
                     "rights_reference": "recorded-owner-rights", "provider_terms_reference": "accepted-provider-terms",
                     "private_processing_authorized": True, "provider_training_authorized": False,
                     "task_confirmed": True, "spend_authorized": True}}


def _arguments(tmp_path, overrides=None, **keyword_overrides):
    geometry = _source_geometry(tmp_path)
    arguments = {"task_context": _task_context(), "task_masks": _masks(geometry), "removal_manifest": REMOVAL,
                 "source_geometry": geometry, "base_scene": _base_scene(tmp_path, geometry),
                 "output_root": tmp_path / "out", "spend": SPEND, "now": NOW}
    arguments.update(overrides(geometry) if callable(overrides) else (overrides or {}))
    arguments.update(keyword_overrides)
    return arguments


def _compile(tmp_path, overrides=None, **keyword_overrides):
    return preparation.compile_website_scene_preparation(**_arguments(tmp_path, overrides, **keyword_overrides))


def test_registered_estimates_compile_into_an_intake_ready_request(tmp_path):
    value = _compile(tmp_path)
    assert value["status"] == "intake_ready", value["blockers"]
    registration = value["registration"]
    assert registration["scale"] == pytest.approx(RUNTIME_SCALE, rel=1e-6)
    assert np.allclose(registration["rotation"], RUNTIME_ROTATION)
    assert registration["physical_scale_measured"] is False
    subject = value["subject"]
    # Source bounds land on the registered table top (runtime y = 2 - 0.5 * 0.5).
    assert subject["aabb_min_xyz"][2] == pytest.approx(3.5, abs=1e-6)
    assert subject["aabb_max_xyz"][2] == pytest.approx(3.6, abs=1e-6)
    assert value["support"]["top_runtime_units"] == pytest.approx(1.75, abs=1e-6)
    assert value["destination"]["position_world_m"][2] == pytest.approx(3.5, abs=1e-6)
    assert value["physics"]["basis"] == "estimated"
    assert value["physics"]["dimensions_m"] == pytest.approx([0.05, 0.1, 0.1], abs=1e-6)
    assert value["physics"]["sensitivity"] == "robust_within_range"
    request = value["intake_request"]
    assert request["source"]["kind"] == "gaussian_splat"
    assert [row["id"] for row in request["execution"]["policy_candidates"]] == ["pi05_droid", "groot_n17_droid"]
    assert value["authoring_inputs"]["dimension_authority"] == "estimated"
    assert value["authoring_inputs"]["source_frames"][0]["role"] == "observed_source"
    composed = json.loads((tmp_path / "out" / "removal_manifest.composed.json").read_text())
    assert composed["entries"][0]["compose_back"]["replacement_asset_id"].startswith("website-subject-")
    assert REMOVAL["entries"][0]["compose_back"]["replacement_asset_id"] is None
    thumbnail = value["thumbnail"]
    with Image.open(thumbnail["path"]) as image:
        assert image.size == (thumbnail["width"], thumbnail["height"]) and image.width <= 480 and image.height <= 300
    assert _sha256_file(Path(thumbnail["path"])) == thumbnail["digest"]
    assert preparation.compile_website_scene_preparation(**_arguments(tmp_path))["digest"] == value["digest"]


def test_missing_destination_is_a_typed_blocker_not_a_default(tmp_path):
    value = _compile(tmp_path, lambda geometry: {"task_masks": _masks(geometry, destination=False)})
    assert value["status"] == "needs_input"
    assert value["blockers"] == ["task_destination_pose_required"]
    assert value["destination"] is None


def test_stale_execution_authority_holds_intake(tmp_path):
    value = _compile(tmp_path, spend={**SPEND, "consent": {**SPEND["consent"], "accepted_at_epoch": NOW - 2 * 86400}})
    assert value["status"] == "needs_input"
    assert value["blockers"] == ["scene_intake_consent_actor_or_time_invalid"]


def test_symmetric_geometry_cannot_register(tmp_path):
    geometry = _source_geometry(tmp_path)
    base = _base_scene(tmp_path, geometry, symmetric=True)
    with pytest.raises(ValueError, match="website_registration_"):
        preparation.register_source_to_runtime(source_geometry=geometry, collision_mesh_path=Path(base["collision_mesh_path"]))


@pytest.mark.parametrize("noise", [0.0, 0.001])
def test_registration_recovers_arbitrary_camera_orientation(tmp_path, noise):
    geometry = _source_geometry(tmp_path)
    base = _base_scene(tmp_path, geometry)
    path = Path(base["collision_mesh_path"])
    mesh = trimesh.load(path, force="mesh", process=False)
    turn = trimesh.transformations.euler_matrix(0.31, -0.47, 0.63)
    mesh.apply_transform(turn)
    if noise:
        mesh.vertices += np.random.default_rng(701).normal(0, noise, mesh.vertices.shape)
    mesh.export(path)
    registration = preparation.register_source_to_runtime(source_geometry=geometry, collision_mesh_path=path)
    tolerance = 0.002 if noise else 1e-5
    assert registration["scale"] == pytest.approx(RUNTIME_SCALE, rel=tolerance)
    assert np.allclose(registration["rotation"], turn[:3, :3] @ RUNTIME_ROTATION, atol=tolerance)
    assert np.allclose(registration["translation"], turn[:3, :3] @ RUNTIME_TRANSLATION, atol=tolerance)
    assert registration["physical_registration_proven"] is False


def test_changed_base_scene_bytes_are_refused(tmp_path):
    def swap(geometry):
        base = _base_scene(tmp_path, geometry)
        Path(base["splat_path"]).write_bytes(b"different")
        return {"base_scene": base}
    with pytest.raises(ValueError, match="website_base_scene_changed"):
        _compile(tmp_path, swap)


def test_physics_screen_escalates_the_smallest_missing_measurement():
    heavy = preparation.screen_physics([0.3, 0.3, 0.3])
    assert heavy["sensitivity"] == "blocked_by_estimate"
    assert heavy["measurement_escalation"]["property"] == "smallest_dimension_m"
    medium = preparation.screen_physics([0.06, 0.06, 0.6])
    assert medium["sensitivity"] == "outcome_depends_on_estimate"
    assert medium["measurement_escalation"]["property"] == "mass_kg"
    assert all(row["basis"] == "estimated" for row in [medium])


def test_capture_consent_does_not_invent_paid_execution_permission(tmp_path):
    value = _compile(tmp_path, spend={key: val for key, val in SPEND.items() if key not in {"owner", "consent"}})
    assert value["status"] == "needs_input"
    assert "website_scene_execution_authority_required" in value["blockers"]
    assert value["intake_request"]["consent"] == {}


def test_registration_inputs_cannot_change_without_rebinding(tmp_path):
    arguments = _arguments(tmp_path)
    arguments["task_masks"]["targets"][0]["estimated_visible_bounds"]["minimum"][0] = 999
    with pytest.raises(ValueError, match="input_digest_mismatch"):
        preparation.compile_website_scene_preparation(**arguments)


def test_pose_and_authoring_bounds_share_normalized_simulator_frame(tmp_path):
    value = _compile(tmp_path)
    lower = value["subject"]["aabb_min_xyz"]
    assert lower == value["authoring_inputs"]["metric_envelope"]["minimum_xyz_m"]
    assert value["coordinate_frame"]["task_coordinates"] == "Z_up_estimated_meters"
    # Y-up (x,y,z) -> Z-up (2*x,-2*z,2*y), including translation.
    assert lower[1] < 0
    assert value["intake_request"]["consent"] == SPEND["consent"]


def test_collected_reconstruction_reaches_real_website_preparation(tmp_path):
    from blueprint_pipeline.common import write_json
    from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    args = _arguments(pipeline)
    base = args["base_scene"]
    # A World Labs export uses OpenCV Y-down, unlike the generic fixture.
    collider_path = Path(base["collision_mesh_path"])
    collider = trimesh.load(collider_path, force="mesh", process=False)
    collider.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))
    collider.export(collider_path)
    base["collision_mesh_digest"] = _sha256_file(collider_path)
    rows = [{"kind": kind, "local_path": base[path], "sha256": base[digest][7:]}
            for kind, path, digest in (("splat_ply", "splat_path", "splat_digest"),
                                      ("collider_mesh_glb", "collision_mesh_path", "collision_mesh_digest"))]
    assets_path = pipeline / "assets.json"
    removal_path = pipeline / "removal.json"
    write_json(assets_path, {"world_id": "world-1", "downloads": rows})
    write_json(removal_path, args["removal_manifest"])
    context = args["task_context"]
    result = prepare_website_scene_handoff(
        descriptor={"capture_id": context["capture_id"], "scene_id": context["scene_id"], "metadata": {
            "site_task_context": context, "website_scene_execution_authority": SPEND}},
        clean_plate={"status": "objects_removed", "privacy_verified": True,
                     "task_masks": args["task_masks"], "source_geometry": args["source_geometry"],
                     "removal_manifest_path": str(removal_path)},
        provider_run={"status": "ready", "world_id": "world-1", "provider_run_id": "op-1",
                      "worldlabs_asset_materialization": {"manifest_path": str(assets_path)}},
        capture_root=tmp_path, now=NOW,
    )
    assert result["status"] == "intake_ready", result["blockers"]
    compiled = json.loads(Path(result["preparation_path"]).read_text())
    assert compiled["coordinate_frame"]["declared_meters_per_unit"] == pytest.approx(1 / RUNTIME_SCALE)
    assert Path(result["thumbnail"]["path"]).is_file()
    assert result["simulator_ready"] is False
    assert result["provider_mutation_performed"] is False
    assert result["runtime_inputs"]["status"] == "background_collision_prepared"
    assert Path(result["runtime_inputs"]["path"]).is_file()


def test_container_destination_is_not_silently_changed_to_top_surface(tmp_path):
    args = _arguments(tmp_path)
    args["task_masks"]["targets"][1]["placement_relation"] = "inside"
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    result = preparation.compile_website_scene_preparation(**args)
    assert result["destination"]["relation"] == "inside"
    assert "mode" not in result["destination"]
    assert result["status"] == "needs_input"
    assert "task_destination_interior_geometry_required" in result["blockers"]


def test_destination_outside_reconstructed_surface_cannot_reach_intake(tmp_path):
    args = _arguments(tmp_path)
    args["task_masks"]["targets"][1]["estimated_visible_bounds"] = _bounds([100, .48, 1.4], [101, .5, 1.6])
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    result = preparation.compile_website_scene_preparation(**args)
    assert result["status"] == "needs_input"
    assert "task_destination_surface_contact_required" in result["blockers"]
    assert result["destination"]["basis"] == "registered_estimated_visible_bounds"


def test_source_far_above_support_is_not_snapped_down_to_floor(tmp_path):
    args = _arguments(tmp_path)
    args["task_masks"]["targets"][0]["estimated_visible_bounds"] = _bounds([.05, -.6, 1.45], [.1, -.5, 1.55])
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    result = preparation.compile_website_scene_preparation(**args)
    assert result["status"] == "needs_input"
    assert "support_surface_not_found_under_subject" in result["blockers"]
    assert result["compose_back"]["pose_world"]["support_snap_runtime_units"] == 0


def test_website_task_and_original_frames_reach_existing_astra_request(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_configuration_astra_driver import build_authoring_request
    args = _arguments(tmp_path)
    prepared = preparation.compile_website_scene_preparation(**args)
    source = tmp_path / "out/preparation.json"
    request = build_authoring_request(
        {"configuration": prepared["authoring_inputs"]["configuration"], "construction_envelope": {},
         "run_id": "development-website-test", "source_commit": "a" * 40},
        {"path": str(source), "digest": _sha256_file(source)},
        [Path(row["path"]) for row in prepared["authoring_inputs"]["source_frames"]],
        {"status": "admitted_for_internal_development", "private_provider_processing_allowed": True,
         "provider_training_allowed": False, "public_redistribution_allowed": False},
    )
    assert args["task_context"]["description"] in request.construction_constraints
    assert "rebuild_only_this_subject" in request.construction_constraints
    assert request.dimension_authority == "estimated"
    assert request.physical_review_input.measured.mass_kg is None
    assert request.source_frames[0].description.startswith("Original website capture frame")
    assert request.dimensions_m == pytest.approx(prepared["physics"]["dimensions_m"])


def _anchored_base_scene(root: Path, source_geometry, *, rotation=np.eye(3), translation=(0.1, 0.05, 0.02),
                         ground_plane_offset_m=0.6):
    points, faces = _grid(source_geometry)
    vertices = (points @ np.asarray(rotation).T) * RUNTIME_SCALE + np.asarray(translation)
    mesh_path = root / "anchored-collider.glb"
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh_path)
    splat = root / "anchored-world.ply"
    splat.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
    return {"splat_path": str(splat), "splat_digest": _sha256_file(splat), "splat_binding_id": "marble-splat-2",
            "collision_mesh_path": str(mesh_path), "collision_mesh_digest": _sha256_file(mesh_path),
            "collision_binding_id": "marble-mesh-2", "up_axis": "-Y", "meters_per_unit": 1 / RUNTIME_SCALE,
            "ground_plane_offset_m": ground_plane_offset_m, "provider": "world_labs", "operation_id": "op-2",
            "anchor": {"kind": "first_input_view_camera", "frame_id": "frame-0"}}


def _anchor(base, **overrides):
    return {**base["anchor"], "meters_per_unit": base["meters_per_unit"], "up_axis": base["up_axis"],
            "ground_plane_offset_m": base["ground_plane_offset_m"], **overrides}


def test_provider_anchor_registers_from_the_first_view_and_checks_the_declared_ground(tmp_path):
    geometry = _source_geometry(tmp_path)
    base = _anchored_base_scene(tmp_path, geometry)
    value = preparation.register_source_to_runtime(
        source_geometry=geometry, collision_mesh_path=Path(base["collision_mesh_path"]), anchor=_anchor(base),
        focus_bounds={"minimum": [0.05, 0.4, 1.45], "maximum": [0.10, 0.5, 1.55]})
    assert value["scale_status"] == "provider_declared_anchor"
    assert value["scale"] == pytest.approx(RUNTIME_SCALE, rel=1e-3)
    assert value["translation"] == pytest.approx([0.1, 0.05, 0.02], abs=1e-3)
    anchor = value["anchor"]
    assert anchor["roll_degrees"] == 0 and anchor["rotation_deviation_degrees"] < 0.5
    assert anchor["translation_deviation_m"] == pytest.approx(2 * np.linalg.norm([0.1, 0.05, 0.02]), abs=0.01)
    assert anchor["scale_ratio_to_declared"] == pytest.approx(1.0, abs=1e-3)
    # The fixture's table sits 0.5 m below the anchor camera: 0.6 m declared matches.
    assert value["ground_plane"]["checked"] is True
    assert value["ground_plane"]["residual_m"] < 0.02
    assert value["task_region"]["point_count"] >= 50 and value["task_region"]["trimmed_rmse_m"] < 0.01
    assert value["physical_registration_proven"] is False


def test_declared_ground_plane_that_contradicts_the_footage_is_refused(tmp_path):
    geometry = _source_geometry(tmp_path)
    base = _anchored_base_scene(tmp_path, geometry, ground_plane_offset_m=1.5)
    with pytest.raises(ValueError, match="website_registration_ground_plane_inconsistent"):
        preparation.register_source_to_runtime(source_geometry=geometry,
            collision_mesh_path=Path(base["collision_mesh_path"]), anchor=_anchor(base))


def test_anchor_view_absent_from_the_estimate_is_refused(tmp_path):
    geometry = _source_geometry(tmp_path)
    base = _anchored_base_scene(tmp_path, geometry)
    with pytest.raises(ValueError, match="website_registration_anchor_frame_missing"):
        preparation.register_source_to_runtime(source_geometry=geometry,
            collision_mesh_path=Path(base["collision_mesh_path"]), anchor=_anchor(base, frame_id="frame-9"))


def test_world_that_does_not_follow_its_declared_anchor_is_a_typed_refusal(tmp_path):
    geometry = _source_geometry(tmp_path)
    base = _anchored_base_scene(tmp_path, geometry, rotation=RUNTIME_ROTATION, translation=(0.0, 0.0, 0.0),
                                ground_plane_offset_m=None)
    with pytest.raises(ValueError, match="website_registration_(anchor_deviation|conflicts_provider_anchor|ambiguous)"):
        preparation.register_source_to_runtime(source_geometry=geometry,
            collision_mesh_path=Path(base["collision_mesh_path"]), anchor=_anchor(base))


def test_anchored_world_compiles_with_the_declared_scale(tmp_path):
    value = _compile(tmp_path, lambda geometry: {"base_scene": _anchored_base_scene(tmp_path, geometry)})
    assert value["status"] == "intake_ready", value["blockers"]
    assert value["registration"]["scale_status"] == "provider_declared_anchor"
    assert value["coordinate_frame"]["declared_meters_per_unit"] == pytest.approx(1 / RUNTIME_SCALE)
    assert value["physics"]["dimensions_m"] == pytest.approx([0.05, 0.1, 0.1], abs=1e-6)


def test_declared_scale_without_an_anchor_must_agree_with_the_registration(tmp_path):
    def base(geometry):
        scene = _base_scene(tmp_path, geometry)
        return {"base_scene": {**scene, "meters_per_unit": 8.0}}
    value = _compile(tmp_path, base)
    assert "website_registration_scale_conflicts_declared" in value["blockers"]
