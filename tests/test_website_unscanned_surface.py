"""A destination surface the scan does not show (glass) is estimated from its visible edges in the frames.

Hermetic: a small room is ray-cast here (floor, back wall, a table with a cup) for six cameras. The glass shelf on
the wall is in each frame's mask but not in its depth, so the collider made from that depth has no shelf, as test
job 4's glass wall shelf had none. Compile runs for real on it; no provider is reached and nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_unscanned_surface.py
# Covers: src/blueprint_pipeline/website_task_preparation.py
# Covers: src/blueprint_pipeline/atlas_scene_estimates.py
from __future__ import annotations

import functools
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from blueprint_pipeline import website_gemini_receipts
from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline import website_unscanned_surface as unscanned
from blueprint_pipeline.atlas_scene_estimates import (DESTINATION_SURFACE_ESTIMATED, base_scene_flags,
                                                      preparation_flags)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_task_masks import estimate_target_bounds

NOW = 1_800_000_000.0
WIDTH, HEIGHT, FOCAL = 320, 240, 160.0
#: The room in upright metres (y up). Source frames are OpenCV (y down, z forward): ``SOURCE @ upright``.
SOURCE = np.diag([1.0, -1.0, -1.0])
#: The collider's frame: half the source's units, upright again, moved (``meters_per_unit`` 2).
RUNTIME_SCALE, RUNTIME_TRANSLATION = 0.5, np.array([1.0, 2.0, 3.0])
SHELF_TOP, SHELF_THICKNESS = 1.2, 0.01
SHELF = ((0.55, SHELF_TOP - SHELF_THICKNESS, 2.8), (1.15, SHELF_TOP, 3.0))  # 0.6 m long, 0.2 m deep, on the wall
SHELF_TOP_Z = SHELF_TOP + RUNTIME_TRANSLATION[1] / RUNTIME_SCALE
#: The room turned over about x: a collider whose up is -Y (a proper rotation), and each collider's turn by path.
FLIP, TURNS = np.diag([1.0, -1.0, -1.0]), {}
TABLE, CUP = ((-0.9, 0.0, 2.2), (-0.3, 0.75, 2.8)), ((-0.66, 0.75, 2.44), (-0.54, 0.87, 2.56))
#: Six frames filmed walking past, looking down at the wall between the table and the shelf.
CAMERAS = [((-0.3, 1.9, 1.5), (0.2, 1.0, 2.8)), ((0.1, 2.0, 1.4), (0.3, 1.0, 2.8)), ((0.5, 1.85, 1.6), (0.2, 1.0, 2.8)),
           ((0.9, 1.95, 1.5), (0.2, 1.0, 2.8)), ((1.3, 1.9, 1.7), (0.3, 1.0, 2.8)), ((0.3, 1.8, 1.9), (0.1, 1.0, 2.8))]


def _room(*, opaque_shelf: bool, cup: bool = True, shelf: tuple = SHELF, cup_box: tuple = CUP) -> list:
    """The boxes depth sees: the room's six sides, the table, the cup unless taken out (the clean plate the scan
    is made from), and the shelf only when it is not glass."""
    return [((-1.5, -0.02, 0.3), (2.5, 0.0, 3.0)), ((-1.5, 0.0, 3.0), (2.5, 2.5, 3.05)),
            ((-1.5, 2.5, 0.3), (2.5, 2.52, 3.0)), ((-1.52, 0.0, 0.3), (-1.5, 2.5, 3.0)),
            ((2.5, 0.0, 0.3), (2.52, 2.5, 3.0)), ((-1.5, 0.0, 0.28), (2.5, 2.5, 0.3)), TABLE,
            *([cup_box] if cup else []), *([shelf] if opaque_shelf else [])]


def _camera(position, target) -> np.ndarray:
    """The camera's pose in upright metres: OpenCV axes (x right, y down, z forward)."""
    forward = np.subtract(target, position) / np.linalg.norm(np.subtract(target, position))
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = np.c_[right, np.cross(forward, right), forward], position
    return pose


def _cast(boxes: list, pose: np.ndarray, focal: float = FOCAL) -> np.ndarray:
    """Each pixel's depth (camera z) to the nearest of ``boxes``, inf where its ray meets none (slab method: a ray
    whose camera z is 1 reaches depth t at parameter t)."""
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    rays = np.stack([(columns.ravel() - WIDTH / 2) / focal, (rows.ravel() - HEIGHT / 2) / focal,
                     np.ones(rows.size)], axis=1) @ pose[:3, :3].T
    depth = np.full(rows.size, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for low, high in boxes:
            near, far = (np.asarray(low) - pose[:3, 3]) / rays, (np.asarray(high) - pose[:3, 3]) / rays
            enter = np.nanmax(np.minimum(near, far), axis=1)
            leave = np.nanmin(np.maximum(near, far), axis=1)
            hit = (enter <= leave) & (enter > 0)
            depth[hit] = np.minimum(depth[hit], enter[hit])
    return depth.reshape(HEIGHT, WIDTH)


@functools.lru_cache(maxsize=None)
def _views(opaque_shelf: bool, cameras: tuple | None = None, focal: float = FOCAL, shelf: tuple = SHELF,
           cup_box: tuple = CUP) -> tuple:
    """Per camera: the frame's depth, the depth of its clean plate (the cup taken out) the scan is made from, and
    the shelf's, the cup's and the table's masks. Glass: the shelf is in its mask but in neither depth."""
    views = []
    for position, target in cameras or CAMERAS:
        pose = _camera(position, target)
        depth = _cast(_room(opaque_shelf=opaque_shelf, shelf=shelf, cup_box=cup_box), pose, focal)
        shelf_depth, cup_depth, table_depth = (_cast([box], pose, focal) for box in (shelf, cup_box, TABLE))
        views.append((pose, depth, _cast(_room(opaque_shelf=opaque_shelf, cup=False, shelf=shelf), pose, focal),
                      *(np.isfinite(seen) & (seen <= depth + 1e-6) for seen in (shelf_depth, cup_depth, table_depth))))
    return tuple(views)


def _runs(mask: np.ndarray) -> list[dict]:
    edges = np.diff(np.pad(mask.reshape(-1).astype(np.int8), (1, 1)))
    return [{"start": int(start), "length": int(end - start)}
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def _track(name: str, masks: list[np.ndarray]) -> dict:
    return {"track_id": name, "label": name, "observations": [
        {"source_frame_id": f"frame-{index}", "height": HEIGHT, "width": WIDTH, "runs": _runs(mask)}
        for index, mask in enumerate(masks) if mask.any()]}


def _arguments(root: Path, *, opaque_shelf: bool = False, shelf_masks=None, cup_masks=None,
               cameras: tuple | None = None, focal: float = FOCAL, onto_table: bool = False, shelf: tuple = SHELF,
               flipped: bool = False, cup_box: tuple = CUP) -> dict:
    """Compile's inputs for the room: its source geometry, a collider made from that depth alone, and masks
    (``cameras`` and ``focal``: another set of views of the same room; ``onto_table``: the destination is the
    table the cup stands on, a surface the scan shows; ``shelf``, ``cup_box``: another shelf or cup; ``flipped``:
    the collider's up is -Y, the room turned over about x)."""
    root.mkdir(parents=True, exist_ok=True)
    frames, points, faces = [], [], []
    turn = FLIP if flipped else np.eye(3)
    for index, (pose, depth, clean, *_masks) in enumerate(_views(opaque_shelf, cameras, focal, shelf, cup_box)):
        valid = np.isfinite(depth)
        geometry = root / f"frame-{index}.npz"
        np.savez(geometry, depth_m=np.where(valid, depth, 0.0), valid_mask=valid)
        image = root / f"frame-{index}.png"
        Image.fromarray(np.full((HEIGHT, WIDTH, 3), 120, dtype=np.uint8)).save(image)
        source_pose = np.eye(4)
        source_pose[:3, :3], source_pose[:3, 3] = SOURCE @ pose[:3, :3], SOURCE @ pose[:3, 3]
        frames.append({"frame_id": f"frame-{index}", "timestamp_seconds": index * 0.5, "image_path": str(image),
                       "image_digest": _sha256_file(image), "geometry_path": str(geometry),
                       "geometry_digest": _sha256_file(geometry),
                       "intrinsics": [[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1]],
                       "world_from_camera": source_pose.tolist(), "width": WIDTH, "height": HEIGHT})
        # The collider is the depth's own surface, a pixel grid like a splat-derived mesh.
        rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
        camera = np.stack([(columns - WIDTH / 2) / focal, (rows - HEIGHT / 2) / focal, np.ones_like(rows, float)],
                          axis=-1) * clean[..., None]
        upright = camera.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
        offset = sum(len(block) for block in points)
        points.append(upright @ turn.T * RUNTIME_SCALE + RUNTIME_TRANSLATION)
        a = offset + rows[:-1, :-1] * WIDTH + columns[:-1, :-1]
        keep = (valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]).ravel()
        a = a.ravel()[keep]
        faces += [np.c_[a, a + 1, a + WIDTH], np.c_[a + 1, a + WIDTH + 1, a + WIDTH]]
    collider = root / "collider.glb"
    TURNS[str(collider)] = turn
    trimesh.Trimesh(vertices=np.concatenate(points), faces=np.concatenate(faces), process=False).export(collider)
    splat = root / "world.ply"
    splat.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
    source_geometry = {"schema_version": "website_source_geometry.v1", "frames": frames, "unit": "estimated_meters",
                       "scale_status": "model_estimated", "metric_measurement_proven": False}
    source_geometry["digest"] = canonical_digest(source_geometry, digest_field="digest")
    views = _views(opaque_shelf, cameras, focal, shelf, cup_box)
    cup = _track("cup", cup_masks or [view[4] for view in views])
    shelf = _track("glass_shelf", shelf_masks or [view[5 if onto_table else 3] for view in views])
    targets = [{"target_id": "cup-1", "task_effect": "manipulated", "disposition": "remove", "semantic_label": "cup",
                "track": cup, "estimated_visible_bounds": estimate_target_bounds(cup, frames)},
               {"target_id": "glass_shelf", "task_effect": "static_contact", "disposition": "keep",
                "target_role": "destination", "placement_relation": "on", "semantic_label": "glass wall shelf",
                "track": shelf, "estimated_visible_bounds": estimate_target_bounds(shelf, frames)}]
    masks = {"schema_version": "website_task_masks.v1", "status": "completed", "targets": targets,
             "source_geometry_digest": source_geometry["digest"]}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    context = {"schema_version": "website_site_task_context.v1", "request_id": "req1", "scene_id": "site-req1",
               "capture_id": "walkthrough-req1", "description": "Move the cup onto the glass shelf on the wall.",
               "confirmed": True, "confirmed_at": datetime.fromtimestamp(NOW - 100, timezone.utc).isoformat(),
               "operator_answers": {}, "unresolved": [], "capture_rights": {"derived_scene_generation_allowed": True}}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    removal = {"schema_version": "clean_plate_removal_manifest.v1", "entries": [
        {"target_id": "cup-1", "semantic_label": "cup", "task_effect": "manipulated", "disposition": "remove",
         "compose_back": {"replacement_asset_id": None, "pose_world": None,
                          "replacement_asset_frame_registration_uri": None}}]}
    spend = {"max_total_spend_usd": 40, "max_paid_attempts": 2, "expires_at_epoch": NOW + 3600,
             "owner": {"user_id": "owner-1", "organization_id": "org-1"},
             "consent": {"accepted_by": "owner-1", "accepted_at_epoch": NOW - 100,
                         "rights_reference": "recorded-owner-rights",
                         "provider_terms_reference": "accepted-provider-terms", "private_processing_authorized": True,
                         "provider_training_authorized": False, "task_confirmed": True, "spend_authorized": True}}
    base = {"splat_path": str(splat), "splat_digest": _sha256_file(splat), "splat_binding_id": "splat-1",
            "collision_mesh_path": str(collider), "collision_mesh_digest": _sha256_file(collider),
            "collision_binding_id": "collider-1", "up_axis": "-Y" if flipped else "Y", "meters_per_unit": 1 / RUNTIME_SCALE,
            "provider": "world_labs", "operation_id": "op-1"}
    return {"task_context": context, "task_masks": masks, "removal_manifest": removal,
            "source_geometry": source_geometry, "base_scene": base, "output_root": root / "out", "spend": spend,
            "now": NOW}


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """No step here may ask a model: every Gemini call of website preparation goes through this one function."""
    def refuse(**_kwargs):
        raise AssertionError("a model was called")
    monkeypatch.setattr(website_gemini_receipts, "retained_gemini_call", refuse)


@pytest.fixture(autouse=True)
def exact_registration(monkeypatch):
    """The room's own source-to-collider similarity, as registration records it: its unconstrained pose search
    (tested in ``test_website_task_preparation``) is not what these tests are about."""
    def register(*, source_geometry, collision_mesh_path, **_options):
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = RUNTIME_SCALE * TURNS.get(str(collision_mesh_path), np.eye(3)) @ SOURCE, \
            RUNTIME_TRANSLATION
        return {"schema_version": "website_source_registration.v1", "source_to_runtime": matrix.tolist(),
                "scale": RUNTIME_SCALE, "rotation": SOURCE.tolist(), "translation": RUNTIME_TRANSLATION.tolist(),
                "trimmed_rmse_runtime_units": 0.005, "trimmed_rmse_m": 0.01, "runner_up_ratio": 3.0, "anchor": None,
                "ground_plane": None, "task_region": {"radius_m": 1.0, "point_count": 5000, "trimmed_rmse_m": 0.01},
                "source_geometry_digest": source_geometry["digest"],
                "collision_mesh_digest": _sha256_file(Path(collision_mesh_path)),
                "scale_status": "estimated_registration", "physical_scale_measured": False,
                "physical_registration_proven": False}
    monkeypatch.setattr(preparation, "register_source_to_runtime", register)


def test_a_see_through_shelf_with_visible_edges_is_estimated_and_the_destination_rests_on_it(tmp_path):
    args = _arguments(tmp_path)
    raw = trimesh.load(args["base_scene"]["collision_mesh_path"], force="mesh", process=False)
    value = preparation.compile_website_scene_preparation(**args)
    record = value["destination_surface_estimate"]
    assert record["status"] == "estimated", record.get("reasons")
    assert record["reconstructed_share"] < unscanned.RECONSTRUCTED_MIN_SHARE
    # The plane and size the frames show: 1.2 m up (the collider's frame is 4 m lower), 0.2 x 0.6 m on the wall.
    assert record["top_z_world_m"] == pytest.approx(SHELF_TOP_Z, abs=0.02)
    assert record["sides_m"][0] == pytest.approx(0.2, abs=0.05) and record["sides_m"][1] == pytest.approx(0.6, abs=0.06)
    assert record["frames_outline"]["agreeing_frames"] == len(CAMERAS)
    assert record["frames_outline"]["median_boundary_px"] <= unscanned.REPROJECTION_MAX_SHARE * np.hypot(WIDTH, HEIGHT)
    # Glass: its masks' depth reads the wall behind, never a level band.
    assert record["depth_anchor"]["status"] in {"no_level_band", "no_depth"}
    assert record["flag"] == DESTINATION_SURFACE_ESTIMATED and record["look"] == "unchanged"
    assert record["subject_footprint"]["supported_at_centre"] is True
    # The destination check rests the cup on the estimated surface, bound by name.
    assert "task_destination_surface_contact_required" not in value["blockers"], value["blockers"]
    assert value["destination"]["basis"] == "registered_estimated_visible_bounds_and_collider_contact"
    assert value["destination"]["position_world_m"][2] == pytest.approx(SHELF_TOP_Z, abs=0.02)
    assert "destination_support" in value
    # The collision shape holds the estimate after every face it had; the look (the splat) is the same file.
    base = preparation.json.loads((tmp_path / "out" / "base_scene.json").read_text())
    sealed = trimesh.load(base["collision_mesh_path"], force="mesh", process=False)
    assert len(sealed.faces) > len(raw.faces) and np.array_equal(np.asarray(sealed.faces)[:len(raw.faces)], raw.faces)
    assert value["binding"]["collision_mesh_digest"] == base["collision_mesh_digest"] == _sha256_file(
        Path(base["collision_mesh_path"]))
    assert base["splat_digest"] == args["base_scene"]["splat_digest"]
    assert base["estimated_surface"]["source_collision_mesh_digest"] == args["base_scene"]["collision_mesh_digest"]
    assert DESTINATION_SURFACE_ESTIMATED in base_scene_flags(base)
    assert DESTINATION_SURFACE_ESTIMATED in preparation_flags(tmp_path / "out" / "preparation.json")
    preparation.json.dumps(value, allow_nan=False)
    # A retry writes nothing new: the same input reads the same sealed collider.
    again = preparation.compile_website_scene_preparation(**{**args, "output_root": tmp_path / "out"})
    assert again["binding"]["collision_mesh_digest"] == value["binding"]["collision_mesh_digest"]


def test_frames_that_disagree_are_refused_and_the_same_code_stops_as_before(tmp_path):
    views = _views(False)
    # Every other frame's outline is moved 40 pixels: the frames no longer place the shelf in one place.
    masks = [np.roll(view[3], 40 * (index % 2), axis=1) for index, view in enumerate(views)]
    args = _arguments(tmp_path, shelf_masks=masks)
    value = preparation.compile_website_scene_preparation(**args)
    record = value["destination_surface_estimate"]
    assert record["status"] == "not_estimated" and record["reasons"], record
    assert "flag" not in record and "collider" not in record
    assert "task_destination_surface_contact_required" in value["blockers"]
    assert value["binding"]["collision_mesh_digest"] == args["base_scene"]["collision_mesh_digest"]
    assert DESTINATION_SURFACE_ESTIMATED not in preparation_flags(tmp_path / "out" / "preparation.json")
    assert not (tmp_path / "out" / "collider").exists() or not any((tmp_path / "out" / "collider").glob(
        "estimated-surface-*"))


def test_too_few_masked_frames_are_refused(tmp_path):
    views = _views(False)
    masks = [view[3] if index < unscanned.MIN_FRAMES - 1 else np.zeros_like(view[3])
             for index, view in enumerate(views)]
    value = preparation.compile_website_scene_preparation(**_arguments(tmp_path, shelf_masks=masks))
    assert value["destination_surface_estimate"]["reasons"] == ["too_few_masked_frames"]
    assert "task_destination_surface_contact_required" in value["blockers"]


def test_a_reconstructed_shelf_takes_the_old_path_unchanged_and_asks_no_model(tmp_path, monkeypatch):
    args = _arguments(tmp_path, opaque_shelf=True)
    monkeypatch.setattr(unscanned, "estimate_surface", lambda **_: pytest.fail("a scanned shelf is never estimated"))
    value = preparation.compile_website_scene_preparation(**args)
    assert "destination_surface_estimate" not in value
    assert "task_destination_surface_contact_required" not in value["blockers"], value["blockers"]
    assert DESTINATION_SURFACE_ESTIMATED not in preparation_flags(tmp_path / "out" / "preparation.json")
    assert not any((tmp_path / "out").rglob("estimated-surface-*"))
    # The same preparation as with the estimate left out of compile entirely.
    monkeypatch.setattr(unscanned, "estimate_unscanned_destination",
                        lambda **kwargs: (kwargs["collider"], kwargs["base_scene"], kwargs["rows"], {}))
    without = preparation.compile_website_scene_preparation(**{**args, "output_root": tmp_path / "without"})
    for key in ("status", "blockers", "binding", "subject", "destination", "destination_support"):
        assert value.get(key) == without.get(key), key


def test_the_shelf_is_estimated_and_used_the_same_in_a_collider_whose_up_is_minus_y(tmp_path):
    value = preparation.compile_website_scene_preparation(**_arguments(tmp_path, flipped=True))
    record = value["destination_surface_estimate"]
    assert record["status"] == "estimated", record.get("reasons")
    # Turned over, the collider's frame is 4 m above the room's floor: the simulator's z is 4 m less.
    expected = SHELF_TOP - RUNTIME_TRANSLATION[1] / RUNTIME_SCALE
    assert record["top_z_world_m"] == pytest.approx(expected, abs=0.02)
    assert record["sides_m"][0] == pytest.approx(0.2, abs=0.05) and record["sides_m"][1] == pytest.approx(0.6, abs=0.06)
    assert "task_destination_surface_contact_required" not in value["blockers"], value["blockers"]
    assert value["destination"]["position_world_m"][2] == pytest.approx(expected, abs=0.02)


def test_a_shelf_masked_in_one_frame_is_estimated_in_the_frames_of_the_video_track_that_mask_matches(tmp_path):
    """Job move-the-lamp-2 (24 Nov): the glass shelf was kept as one frame's mask, so the estimate refused it on
    ``too_few_masked_frames`` and the room package shipped without the lamp. The retained video track that one-frame
    mask matches (``website_task_masks.surface_views``) shows the shelf in every frame; the estimate reads those, under
    every check it has, and records where they came from."""
    views = _views(False)
    seen = max(range(len(views)), key=lambda index: int(views[index][3].sum()))
    one = [view[3] if index == seen else np.zeros_like(view[3]) for index, view in enumerate(views)]
    args = _arguments(tmp_path, shelf_masks=one)
    masks = args["task_masks"]
    shelf = masks["targets"][1]
    assert len(shelf["track"]["observations"]) == 1
    shelf["surface_views"] = {"basis": "retained_video_track_matching_the_one_frame_mask", "track_id": "video-shelf",
                              "matched_frame_id": f"frame-{seen}", "matched_iou": 0.91,
                              "track": _track("glass_shelf", [view[3] for view in views])}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    value = preparation.compile_website_scene_preparation(**args)
    record = value["destination_surface_estimate"]
    assert record["status"] == "estimated", record.get("reasons")
    assert record["masked_frames"] == len(CAMERAS) and record["frames_outline"]["agreeing_frames"] == len(CAMERAS)
    assert record["masks"] == {"basis": "retained_video_track_matching_the_one_frame_mask", "track_id": "video-shelf",
                               "matched_frame_id": f"frame-{seen}", "matched_iou": 0.91, "own_masked_frames": 1}
    assert record["top_z_world_m"] == pytest.approx(SHELF_TOP_Z, abs=0.02)
    assert record["flag"] == DESTINATION_SURFACE_ESTIMATED
    assert "task_destination_surface_contact_required" not in value["blockers"], value["blockers"]
    assert value["destination"]["position_world_m"][2] == pytest.approx(SHELF_TOP_Z, abs=0.02)
    base = preparation.json.loads((tmp_path / "out" / "base_scene.json").read_text())
    sealed = next(row for row in base["scene_estimates"] if row["code"] == DESTINATION_SURFACE_ESTIMATED)
    assert sealed["evidence"]["masks_basis"] == "retained_video_track_matching_the_one_frame_mask"
    assert DESTINATION_SURFACE_ESTIMATED in preparation_flags(tmp_path / "out" / "preparation.json")
    # Without those frames the one frame is refused as before, and the same code stops the destination.
    del shelf["surface_views"]
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    without = preparation.compile_website_scene_preparation(**{**args, "output_root": tmp_path / "without"})
    assert without["destination_surface_estimate"]["reasons"] == ["too_few_masked_frames"]
    assert "masks" not in without["destination_surface_estimate"]
    assert "task_destination_surface_contact_required" in without["blockers"]
