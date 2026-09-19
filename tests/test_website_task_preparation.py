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


def _base_scene(root: Path, source_geometry, *, symmetric=False):
    points = np.random.default_rng(3).uniform(-1, 1, (3000, 3)) if symmetric else preparation._source_points(source_geometry)
    vertices = (points @ RUNTIME_ROTATION.T) * RUNTIME_SCALE + RUNTIME_TRANSLATION
    faces = np.arange(len(vertices) - len(vertices) % 3).reshape(-1, 3)
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
                        "target_role": "destination", "track": track,
                        "estimated_visible_bounds": _bounds([0.4, 0.48, 1.4], [0.6, 0.5, 1.6])})
    value = {"schema_version": "website_task_masks.v1", "status": "completed", "targets": targets,
             "source_geometry_digest": source_geometry["digest"]}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


def _task_context(confirmed_at=NOW - 100):
    value = {"schema_version": "website_site_task_context.v1", "request_id": "req1", "scene_id": "site-req1",
             "capture_id": "walkthrough-req1", "description": "Move the cup onto the tray.", "confirmed": True,
             "confirmed_at": datetime.fromtimestamp(confirmed_at, timezone.utc).isoformat(),
             "operator_answers": {}, "unresolved": []}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


REMOVAL = {"schema_version": "clean_plate_removal_manifest.v1", "entries": [
    {"target_id": "cup-1", "semantic_label": "cup", "task_effect": "manipulated", "disposition": "remove",
     "compose_back": {"replacement_asset_id": None, "pose_world": None, "replacement_asset_frame_registration_uri": None}}]}
SPEND = {"max_total_spend_usd": 40, "max_paid_attempts": 2, "expires_at_epoch": NOW + 3600}


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
    assert subject["aabb_min_xyz"][1] == pytest.approx(1.75, abs=1e-6)
    assert subject["aabb_max_xyz"][1] == pytest.approx(1.80, abs=1e-6)
    assert value["support"]["top_runtime_units"] == pytest.approx(1.75, abs=1e-6)
    assert value["destination"]["position_world_m"][1] == pytest.approx(1.76, abs=1e-6)
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


def test_stale_confirmation_holds_intake(tmp_path):
    value = _compile(tmp_path, task_context=_task_context(NOW - 2 * 86400))
    assert value["status"] == "needs_input"
    assert value["blockers"] == ["scene_intake_consent_actor_or_time_invalid"]


def test_symmetric_geometry_cannot_register(tmp_path):
    geometry = _source_geometry(tmp_path)
    base = _base_scene(tmp_path, geometry, symmetric=True)
    with pytest.raises(ValueError, match="website_registration_"):
        preparation.register_source_to_runtime(source_geometry=geometry, collision_mesh_path=Path(base["collision_mesh_path"]))


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
