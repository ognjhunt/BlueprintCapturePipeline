from __future__ import annotations

import json
import math
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
    # A non-planar observed object patch above the table; masks must describe
    # actual points, not an unrelated hand-written bounding box.
    yy, xx = np.indices((5, 4))
    depth[20:25, 21:25] = 1.57 + 0.005 * (xx + yy)
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


def _masks(source_geometry, *, destination=True, articulated=False):
    runs = [{"start": row * WIDTH + 21, "length": 4, "probability": 0.9} for row in range(20, 25)]
    track = {"track_id": "t1", "label": "cup", "observations": [
        {"source_frame_id": "frame-0", "height": HEIGHT, "width": WIDTH, "runs": runs}]}
    targets = [{"target_id": "cup-1", "task_effect": "manipulated", "disposition": "remove", "track": track,
                "estimated_visible_bounds": _bounds([0.05, 0.4, 1.45], [0.10, 0.5, 1.55]),
                **({"semantic_label": "three-drawer cabinet", "articulated_part": "middle drawer",
                    "articulation_kind": "prismatic"} if articulated else {})}]
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


@pytest.mark.parametrize("family", [None, "rigid_pick_place", "rigid_pick_and_place", "rigid_object_pick_place",
                                     "rigid_relocation", "articulated_open_close"])
def test_existing_native_profiles_and_legacy_contexts_do_not_gain_a_new_hold(family):
    context = {} if family is None else {"task_family": family}
    assert preparation.native_task_preparation_gap(context) is None


def test_a_novel_task_is_retained_without_compiling_it_as_pick_place(tmp_path, monkeypatch):
    arguments = _arguments(tmp_path)
    context = arguments["task_context"]
    context.update(task_family="unimplemented_task_family", description="Fold the towels.")
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    masks = arguments["task_masks"]
    masks["deferred_target_ids"] = ["towels"]
    masks["digest"] = canonical_digest(masks, digest_field="digest")

    def unexpected(**_):
        pytest.fail("A different task must not enter rigid-object placement or authoring")

    monkeypatch.setattr(preparation, "register_source_to_runtime", unexpected)
    result = preparation.compile_website_scene_preparation(**arguments)
    assert result["status"] == "needs_input"
    assert result["blockers"] == ["website_native_task_profile_preparation_required"]
    assert result["intake_request"] is None and result["provider_mutation_performed"] is False
    assert result["native_task_preparation_gap"]["description"] == "Fold the towels."
    assert result["native_task_preparation_gap"]["job_disposition"] == "held_for_preparation"
    assert result["binding"]["task_context_digest"] == context["context_digest"]
    assert result["digest"] == canonical_digest(result, digest_field="digest")
    assert not (tmp_path / "out/removal_manifest.composed.json").exists()
    assert json.loads((tmp_path / "out/preparation.json").read_text()) == result


def test_a_novel_task_still_requires_unchanged_source_bound_inputs(tmp_path):
    arguments = _arguments(tmp_path)
    arguments["task_context"]["task_family"] = "unimplemented_task_family"
    with pytest.raises(ValueError, match="input_digest_mismatch"):
        preparation.compile_website_scene_preparation(**arguments)


@pytest.mark.parametrize("family,mechanism,inferred", [
    ("articulated_open_close", "", "rigid_pick_place"),
    ("rigid_pick_and_place", "prismatic", "articulated_open_close"),
])
def test_explicit_task_semantics_cannot_be_changed_by_a_mechanism_hint(
        tmp_path, monkeypatch, family, mechanism, inferred):
    arguments = _arguments(tmp_path)
    context, masks = arguments["task_context"], arguments["task_masks"]
    context["task_family"] = family
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    next(row for row in masks["targets"] if row["target_id"] == "cup-1")["articulation_kind"] = mechanism
    masks["digest"] = canonical_digest(masks, digest_field="digest")

    def unexpected(**_):
        pytest.fail("A semantic mismatch must hold before registration or authoring")

    monkeypatch.setattr(preparation, "register_source_to_runtime", unexpected)
    result = preparation.compile_website_scene_preparation(**arguments)
    assert result["blockers"] == ["website_task_profile_mechanism_mismatch"]
    assert result["intake_request"] is None
    assert result["native_task_preparation_gap"]["task_family"] == family
    assert result["native_task_preparation_gap"]["inferred_profile_id"] == inferred


def test_owner_targets_and_weight_statements_reach_compiler_without_becoming_scorer_claims(tmp_path):
    context = _task_context()
    context["success_criteria"] = {"successDefinition": "Arrives intact", "successRate": 95, "cycleTimeSeconds": None, "unknown": False}
    context["operator_task_details"] = {"item_weight": "about 5 lb, not weighed", "item_make_model": "ACME X2"}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    result = _compile(tmp_path, task_context=context)
    assert result["status"] == "intake_ready"
    assert result["intake_request"]["execution"]["purpose"] == "scene_preparation"
    assert result["intake_request"]["execution"]["policy_candidates"] == []
    assert result["owner_success_criteria"]["targets"] == context["success_criteria"]
    assert result["owner_success_criteria"]["scorer_translation_verified"] is False
    assert "about 5 lb, not weighed" in json.dumps(result)


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
    assert subject["aabb_max_xyz"][2] == pytest.approx(3.71939, abs=1e-6)
    assert value["support"]["top_runtime_units"] == pytest.approx(1.75, abs=1e-6)
    assert value["destination"]["position_world_m"][2] == pytest.approx(3.5, abs=1e-6)
    assert value["physics"]["basis"] == "estimated"
    assert value["physics"]["dimensions_m"] == pytest.approx([0.16150833333333348, 0.0331, 0.21939], abs=1e-6)
    assert value["physics"]["sensitivity"] == "awaiting_robot_team_selection"
    request = value["intake_request"]
    assert request["source"]["kind"] == "gaussian_splat"
    assert request["execution"]["purpose"] == "scene_preparation"
    assert request["execution"]["policy_candidates"] == []
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


ARTICULATED_REMOVAL = {"schema_version": "clean_plate_removal_manifest.v1", "entries": [
    {"target_id": "cup-1", "semantic_label": "three-drawer cabinet", "task_effect": "manipulated", "disposition": "remove",
     "articulated_part": "middle drawer", "articulation_kind": "prismatic",
     "compose_back": {"replacement_asset_id": None, "pose_world": None, "replacement_asset_frame_registration_uri": None}}]}


# What the classifier would answer for the drawer footage: closed and open
# fronts in the two geometry frames, and oblique views between them.
DRAWER_VIEWS = [
    ("frame-0", 0.0, "closed", "front", ["body_front", "bottom_drawer_front", "handle", "middle_drawer_front",
                                         "top_drawer_front"]),
    ("frame-1", 0.5, "open", "front", ["body_front", "drawer_interior", "handle", "middle_drawer_front"]),
    ("decoded-000000030", 1.0, "partially_open", "left_oblique", ["left_side", "middle_drawer_front"]),
    ("decoded-000000045", 1.5, "closed", "right_oblique", ["right_side", "top_surface"]),
]


def _covered(masks, source_geometry, root: Path, *, views=DRAWER_VIEWS):
    """Coverage built by the real selection, sizing and record functions; only the classifier answer is given."""
    from blueprint_pipeline import website_assembly_coverage as coverage
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, (frame_id, timestamp, _, _, _) in enumerate(views):
        path = root / f"{frame_id}.png"
        Image.fromarray(np.full((HEIGHT, WIDTH, 3), 90 + index, dtype=np.uint8)).save(path)
        frames.append({"frame_id": frame_id, "timestamp_seconds": timestamp, "mask_area_fraction": 0.02,
                       "geometry_frame": frame_id.startswith("frame-"), "path": str(path), "sha256": _sha256_file(path)})
    answer = {"hinge_edge": None, "task_part_components": sorted({"drawer_interior", "handle", "middle_drawer_front"}
                                                                 & {part for row in views for part in row[4]}),
              "frames": [{"frame_id": row[0], "visible_parts": row[4], "task_part_state": row[2], "view": row[3],
                          "label_text": []}
                         for row in views]}
    labels = coverage.validate_classification(answer, frame_ids=[row[0] for row in views], articulation_kind="prismatic")
    classified = {"frames": [{**frame, **label} for frame, label in zip(frames, labels["frames"])],
                  "task_part_components": labels["task_part_components"], "hinge_edges": [],
                  "receipts": [{"binding_digest": "sha256:" + "1" * 64, "result_digest": "sha256:" + "2" * 64}]}
    targets = []
    for row in masks["targets"]:
        if row.get("task_effect") == "manipulated":
            body, blockers = coverage.estimate_body_bounds(track=row["track"], frames=source_geometry["frames"],
                states={label["frame_id"]: label["part_state"] for label in classified["frames"]})
            row = {**row, "authoring_coverage": coverage.coverage_record(target_id=row["target_id"],
                binding=coverage.coverage_binding(target=row, source_geometry=source_geometry,
                                                  task_context=_task_context()),
                articulation_kind="prismatic", classified=classified,
                selected=coverage.select_reference_frames(classified["frames"]), body_bounds=body,
                body_blockers=blockers, candidate_count=len(frames))}
        targets.append(row)
    value = {**masks, "targets": targets}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


INTERIOR_DEPTH_M = 0.4
# Source-frame pixel boxes (rows, cols) and depths of the drawer cabinet.
CLOSED_FRONT = ((14, 28), (14, 27), 1.2)  # Its lowest row rests on the table.
OPEN_INTERIOR = ((12, 22), (15, 25), 1.2 + INTERIOR_DEPTH_M)
PULLED_DRAWER = ((22, 25), (15, 25), 0.9)


def _assembly_inputs(root: Path, *, covered=True):
    """A cabinet filmed closed (frame-0) and with its drawer pulled out (frame-1).

    The open view sees the interior 0.4 m behind the closed front and the
    drawer itself in front of it; only the interior is body depth.
    """
    root = root / "assembly"
    root.mkdir(parents=True, exist_ok=True)
    frames, observations = [], []
    for index, boxes in enumerate(((CLOSED_FRONT,), (OPEN_INTERIOR, PULLED_DRAWER))):
        frame = _frame(root, index)
        depth, mask = _depth(), np.zeros((HEIGHT, WIDTH), dtype=bool)
        for (top, bottom), (left, right), value in boxes:
            depth[top:bottom, left:right] = value
            mask[top:bottom, left:right] = True
        np.savez(frame["geometry_path"], depth_m=depth, valid_mask=np.ones_like(depth, dtype=bool))
        frames.append({**frame, "geometry_digest": _sha256_file(Path(frame["geometry_path"]))})
        flat = np.flatnonzero(np.diff(np.pad(mask.reshape(-1).astype(np.int8), (1, 1))))
        observations.append({"source_frame_id": frame["frame_id"], "height": HEIGHT, "width": WIDTH,
                             "runs": [{"start": int(a), "length": int(b - a)} for a, b in zip(flat[::2], flat[1::2])]})
    geometry = {"schema_version": "website_source_geometry.v1", "frames": frames, "unit": "estimated_meters",
                "scale_status": "model_estimated", "metric_measurement_proven": False}
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    masks = _masks(geometry, destination=False, articulated=True)
    masks["targets"][0]["track"] = {**masks["targets"][0]["track"], "observations": observations}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    return {"source_geometry": geometry, "removal_manifest": ARTICULATED_REMOVAL,
            "task_masks": _covered(masks, geometry, root / "views") if covered else masks}


def test_articulated_assembly_without_whole_object_coverage_is_not_bought(tmp_path):
    # Surfaces from the depth-sampling frames are not the whole assembly: a thin
    # front would pass a dimension check against them, so the build is held.
    value = _compile(tmp_path, _assembly_inputs(tmp_path, covered=False))
    assert value["status"] == "needs_input"
    # Those surfaces mix the closed front, the open interior and the pulled
    # drawer; the box they enclose does not even rest on the counter.
    assert value["blockers"] == ["support_surface_not_found_under_subject",
                                 "website_assembly_whole_object_coverage_required"]
    assert value["subject"]["complete_object_dimensions"] is False
    assert "required_parts" not in value["authoring_inputs"]["configuration"]


@pytest.mark.parametrize('mode', ['gravity_loaded', 'unknown'])
def test_recorded_gravity_response_reaches_the_actual_website_authoring_configuration(tmp_path, mode):
    inputs = _assembly_inputs(tmp_path)
    response = {'mode': mode, 'reason': 'Mechanism estimate retained with the source review.'}
    masks = inputs['task_masks']
    record = masks['targets'][0]['authoring_coverage']
    record['gravity_response'] = response
    record['digest'] = canonical_digest(record, digest_field='digest')
    masks['digest'] = canonical_digest(masks, digest_field='digest')
    value = _compile(tmp_path, inputs)
    assert value['status'] == 'intake_ready', value['blockers']
    assert value['intake_request']['task']['articulation']['gravity_response'] == response
    assert value['authoring_inputs']['configuration']['mechanism']['passive_dynamics']['gravity_response'] == response


def test_articulated_assembly_compiles_into_an_open_close_intake(tmp_path):
    inputs = _assembly_inputs(tmp_path)
    value = _compile(tmp_path, inputs)
    assert value["status"] == "intake_ready", value["blockers"]
    task = value["intake_request"]["task"]
    assert task["strategy"] == "articulated_open_close" and "destination" not in task
    mechanism = task["articulation"]
    assert mechanism["joint_type"] == "prismatic" and mechanism["part_label"] == "middle drawer"
    assert mechanism["lock_status"] == "unknown" and mechanism["part_observed_open_in_footage"] is True
    assert mechanism["travel_authority"] == "object_prior_estimate_from_observed_body_depth"
    assert mechanism["physical_measurement_proven"] is False
    normal = np.asarray(mechanism["estimated_front_normal_world"])
    assert np.linalg.norm(normal) == pytest.approx(1.0) and normal[2] == pytest.approx(0.0)
    dims = value["physics"]["dimensions_m"]
    depth = abs(normal[0]) * dims[0] + abs(normal[1]) * dims[1]
    assert mechanism["estimated_usable_stroke_m"] == pytest.approx(0.75 * depth, abs=1e-4)
    assert task["success"] == preparation.ARTICULATED_SUCCESS
    configuration = value["authoring_inputs"]["configuration"]
    assert configuration["schema_version"] == "articulated_replacement_authoring_configuration.v1"
    assert configuration["mechanism"]["joint_limits"] == [0.0, mechanism["estimated_usable_stroke_m"]]
    assert configuration["mechanism"]["passive_dynamics"]["task_joint_drive"] == "none"
    assert configuration["required_output"]["task_joint_count"] == 1
    assert value["authoring_inputs"]["adapter"] == "astra_articulated_replacement"
    assert value["recipe_plan"]["articulated_replacement_authoring"] == "astra_articulated_replacement"
    assert value["physics"]["sensitivity"] == "awaiting_robot_team_selection"
    assert value["physics"]["measurement_escalation"]["property"] == "joint_friction"
    assert value["destination"] is None
    # The whole assembly still rests on the support beneath its footprint.
    assert value["support"] is not None
    # The builder sees the chosen upright views, not the depth-sampling frames.
    coverage = inputs["task_masks"]["targets"][0]["authoring_coverage"]
    frames = value["authoring_inputs"]["source_frames"]
    assert [row["frame_id"] for row in frames] == [row["frame_id"] for row in coverage["selected_frames"]]
    assert all(row["role"] == "observed_source" and row["reason"] for row in frames)
    # Each frame goes out as a provider-sized derivative; the retained original stays bound.
    references = configuration["reference_frames"]
    assert [{key: row[key] for key in ("frame_id", "timestamp_seconds", "visible_parts", "part_state", "view",
                                       "reason", "selection_rank")} for row in references] == [
        {key: row[key] for key in ("frame_id", "timestamp_seconds", "visible_parts", "part_state", "view", "reason",
                                   "selection_rank")} for row in coverage["selected_frames"]]
    for row, original in zip(references, coverage["selected_frames"]):
        assert row["transmission"]["source_sha256"] == original["sha256"] == _sha256_file(Path(original["path"]))
        assert row["sha256"] == _sha256_file(Path(row["path"])) and row["transmission"]["format"] == "png"
    assert configuration["reference_frame_budget"]["provider"] == "openai"
    assert configuration["reference_frame_budget"]["dropped_frame_ids"] == []
    assert [row["sha256"] for row in frames] == [row["sha256"] for row in references]
    assert configuration["assembly_family"] == "stacked_drawer_cabinet" and configuration["hinge_edge"] is None
    roles = {row["part_id"]: row["role"] for row in configuration["required_parts"]}
    # Non-task drawers are fixed bays, never carcass panels named by their position word.
    assert roles == {"body_front": "body_feature", "bottom_drawer_front": "fixed_interior", "drawer_interior": "task_part",
                     "handle": "door_feature", "left_side": "body_feature", "middle_drawer_front": "task_part",
                     "right_side": "body_feature", "top_drawer_front": "fixed_interior", "top_surface": "body_feature"}
    assert all(row["observed_frame_ids"] for row in configuration["required_parts"])
    # Whole body: the interior 0.4 m behind the closed front, scaled 0.5 x 2 into simulator metres.
    assert configuration["body_depth"]["basis"] == "interior_observed_open_state"
    assert configuration["body_depth"]["frame_ids"] == ["frame-1"]
    assert configuration["body_depth"]["value_m"] == pytest.approx(INTERIOR_DEPTH_M, abs=0.03)
    assert depth == pytest.approx(configuration["body_depth"]["value_m"], rel=0.05)
    constraints = configuration["construction_constraints"]
    assert constraints["complete_object_dimensions_observed"] is True and value["subject"]["complete_object_dimensions"]
    assert constraints["task_part"] == "middle drawer" and constraints["hinge_edge"] is None
    assert set(constraints["fixed_parts"]) == set(constraints["required_parts"]) - {"drawer_interior", "handle",
                                                                                   "middle_drawer_front"}
    assert configuration["metric_envelope"]["minimum_xyz_m"] == value["subject"]["aabb_min_xyz"]


def test_articulated_assembly_with_stale_or_incomplete_coverage_is_not_bought(tmp_path):
    inputs = _assembly_inputs(tmp_path)
    masks = inputs["task_masks"]
    record = masks["targets"][0]["authoring_coverage"]
    stale = {**record, "binding": {**record["binding"], "source_geometry_digest": "sha256:" + "0" * 64}}
    stale["digest"] = canonical_digest(stale, digest_field="digest")
    changed = {**masks, "targets": [{**masks["targets"][0], "authoring_coverage": stale}]}
    changed["digest"] = canonical_digest(changed, digest_field="digest")
    value = _compile(tmp_path, {**inputs, "task_masks": changed})
    assert value["blockers"][1:] == ["website_assembly_coverage_binding_mismatch",
                                     "website_assembly_whole_object_coverage_required"]
    unopened = [row for row in DRAWER_VIEWS if row[2] == "closed"]
    inputs = _assembly_inputs(tmp_path / "held", covered=False)
    held = _covered(inputs["task_masks"], inputs["source_geometry"], tmp_path / "held" / "views", views=unopened)
    value = _compile(tmp_path, {**inputs, "task_masks": held})
    assert value["status"] == "needs_input"
    assert value["blockers"][1:] == ["website_assembly_whole_object_coverage_required",
                                     "website_assembly_body_depth_unobserved"]
    assert value["intake_request"]["task"]["articulation"]["part_observed_open_in_footage"] is False


def test_articulated_task_refuses_a_destination_row(tmp_path):
    value = _compile(tmp_path, lambda geometry: {"task_masks": _masks(geometry, destination=True, articulated=True),
                                                 "removal_manifest": ARTICULATED_REMOVAL})
    assert value["status"] == "needs_input"
    assert "website_articulated_task_has_no_destination" in value["blockers"]


def test_missing_destination_is_a_typed_blocker_not_a_default(tmp_path):
    value = _compile(tmp_path, lambda geometry: {"task_masks": _masks(geometry, destination=False)})
    assert value["status"] == "needs_input"
    assert value["blockers"] == ["task_destination_pose_required"]
    assert value["destination"] is None


def test_stale_execution_authority_holds_intake(tmp_path):
    value = _compile(tmp_path, spend={**SPEND, "consent": {**SPEND["consent"], "accepted_at_epoch": NOW - 2 * 86400}})
    assert value["status"] == "needs_input"
    assert value["blockers"] == ["scene_intake_consent_actor_or_time_invalid"]


def test_anthropic_authoring_requires_explicit_matching_terms_in_new_authority(tmp_path):
    missing = _compile(tmp_path, spend={**SPEND, "replacement_authoring_backend": "astra_cad_blender_v1", "authoring_provider": "anthropic"})
    assert missing["status"] == "needs_input"
    assert "website_anthropic_provider_terms_authority_required" in missing["blockers"]
    terms = "sha256:" + "a" * 64
    approved = _compile(tmp_path, spend={**SPEND, "replacement_authoring_backend": "astra_cad_blender_v1", "authoring_provider": "anthropic",
        "anthropic_provider_terms_reference": terms})
    assert approved["status"] == "intake_ready", approved["blockers"]
    assert "anthropic" in approved["intake_request"]["execution"]["allowed_providers"]
    assert approved["authoring_inputs"]["configuration"]["authoring_model_provider"] == "anthropic"
    assert approved["authoring_provider_terms_reference"] == terms
    assert approved["intake_request"]["consent"]["provider_terms_reference"] == SPEND["consent"]["provider_terms_reference"]


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
    heavy = preparation.screen_physics([0.3, 0.3, 0.3], gripper=preparation.GRIPPER)
    assert heavy["sensitivity"] == "blocked_by_estimate"
    assert heavy["measurement_escalation"]["property"] == "smallest_dimension_m"
    medium = preparation.screen_physics([0.06, 0.06, 0.6], gripper=preparation.GRIPPER)
    assert medium["sensitivity"] == "outcome_depends_on_estimate"
    assert medium["measurement_escalation"]["property"] == "mass_kg"
    assert all(row["basis"] == "estimated" for row in [medium])


def test_a_registered_bodys_gripper_is_screened_on_its_own_opening_and_its_hold_stays_unassessed(tmp_path):
    from blueprint_pipeline.team_embodiment import validate_embodiment_manifest
    from tests.test_team_embodiment import manifest, write_robot_usd

    body = validate_embodiment_manifest(manifest(write_robot_usd(tmp_path / "robot.usda")))
    [tool] = preparation.embodiment_grippers(body)
    # The manifest's own 80 mm opening; it declares no grip force, so no reference force stands in for it.
    assert tool == {"model": "acme-arm-v2/gripper", "full_stroke_m": 0.08, "grip_force_n": None,
                    "source": "registered_embodiment_manifest"}
    narrow = preparation.screen_physics([0.06, 0.06, 0.6], gripper=tool)
    assert narrow["sensitivity"] == "hold_unassessed_grip_force_not_declared"
    assert narrow["measurement_escalation"] is None and narrow["gripper"] == tool
    wide = preparation.screen_physics([0.1, 0.1, 0.1], gripper=tool)
    assert wide["sensitivity"] == "blocked_by_estimate"
    assert wide["measurement_escalation"]["property"] == "smallest_dimension_m"
    # The Robotiq reference's verdict on the narrow object is unchanged.
    assert preparation.screen_physics([0.06, 0.06, 0.6], gripper=preparation.GRIPPER)["sensitivity"] == (
        "outcome_depends_on_estimate")
    assert preparation.embodiment_grippers({**body, "gripper": None}) == []


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


def test_a_ready_atlas_scene_reaches_real_website_preparation(tmp_path, monkeypatch):
    from blueprint_pipeline.common import write_json
    from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    args = _arguments(pipeline)
    base = {**args["base_scene"], "provider": "world_labs_atlas", "scale_authority": "atlas_reference_estimate"}
    geometry = {**args["source_geometry"], "geometry_backend": "world_labs_atlas"}
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    masks = {**args["task_masks"], "source_geometry_digest": geometry["digest"]}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    removal_path = pipeline / "removal.json"
    write_json(removal_path, args["removal_manifest"])
    # The capture's Atlas job supplies the scene and the posed source geometry.
    monkeypatch.setattr("blueprint_pipeline.website_atlas_scene_source.atlas_scene_source", lambda **_: {
        "status": "ready", "base_scene": base, "source_geometry": geometry, "alignment": "anchored", "source_pose_digest": "sha256:" + "1" * 64,
        "background_pose_digest": "sha256:" + "2" * 64})
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **_: masks)
    context = args["task_context"]
    result = prepare_website_scene_handoff(
        descriptor={"capture_id": context["capture_id"], "scene_id": context["scene_id"], "metadata": {
            "site_task_context": context, "website_scene_execution_authority": SPEND}},
        clean_plate={"status": "objects_removed", "privacy_verified": True, "source_geometry": None,
                     "task_masks": args["task_masks"], "removal_manifest_path": str(removal_path)},
        capture_root=tmp_path, now=NOW,
    )
    assert result["status"] == "intake_ready", result["blockers"]
    compiled = json.loads(Path(result["preparation_path"]).read_text())
    assert compiled["coordinate_frame"]["declared_meters_per_unit"] == pytest.approx(1 / RUNTIME_SCALE)
    assert Path(result["thumbnail"]["path"]).is_file()
    assert result["simulator_ready"] is False
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


def container_arguments(args):
    """A captured open bin on the existing table in the same registered frame."""
    walls = []
    for size, center in [([.02, .4, .44], [.28, .3, 1.5]),
                         ([.02, .4, .44], [.72, .3, 1.5]),
                         ([.46, .4, .02], [.5, .3, 1.28]),
                         ([.46, .4, .02], [.5, .3, 1.72])]:
        wall = trimesh.creation.box(size)
        wall.apply_translation(center)
        wall.vertices = np.asarray(wall.vertices) @ RUNTIME_ROTATION.T * RUNTIME_SCALE + RUNTIME_TRANSLATION
        walls.append(wall)
    path = Path(args["base_scene"]["collision_mesh_path"])
    mesh = trimesh.load(path, force="mesh", process=False)
    trimesh.util.concatenate([mesh, *walls]).export(path)
    args["base_scene"]["collision_mesh_digest"] = _sha256_file(path)
    row = args["task_masks"]["targets"][1]
    row.update(placement_relation="inside", estimated_visible_bounds=_bounds([.27, .1, 1.27], [.73, .5, 1.73]))
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    return args


def test_captured_open_container_reaches_intake_with_real_floor_and_inside_bounds(tmp_path):
    result = preparation.compile_website_scene_preparation(**container_arguments(_arguments(tmp_path)))
    assert result["status"] == "intake_ready", result["blockers"]
    destination = result["destination"]
    assert destination["relation"] == "inside"
    assert destination["mode"] == "existing_support_surface"
    assert destination["interior_wall_face_indices"]
    assert destination["interior_bounds_world_m"]["minimum"][2] == pytest.approx(destination["position_world_m"][2])
    assert result["destination_support"]["face_indices"]


def test_destination_outside_reconstructed_surface_cannot_reach_intake(tmp_path):
    args = _arguments(tmp_path)
    args["task_masks"]["targets"][1]["estimated_visible_bounds"] = _bounds([100, .48, 1.4], [101, .5, 1.6])
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    result = preparation.compile_website_scene_preparation(**args)
    assert result["status"] == "needs_input"
    assert "task_destination_surface_contact_required" in result["blockers"]
    assert result["destination"]["basis"] == "registered_estimated_visible_bounds"


@pytest.mark.parametrize("shift, distinct", [(5, False), (50, True)])
def test_a_destination_on_the_same_or_another_fitted_surface_is_bound(tmp_path, monkeypatch, shift, distinct):
    # Two fits of one bumpy counter converge on nearly the same faces. Another surface
    # (floor to desk) is bound by name with its own bounds, never refused by kind.
    args = _arguments(tmp_path)
    calls = []

    def fitted(mesh, lower, upper, *, up, meters_per_unit, up_sign=1, placement_residual_m=None):
        start = 0 if not calls else shift
        calls.append(start)
        return {"top_runtime_units": float(lower[up]), "aabb_min": list(lower), "aabb_max": list(upper),
                "face_indices": list(range(start, start + 100)), "basis": "fitted_surface_under_footprint",
                "physical_measurement": False}
    monkeypatch.setattr(preparation, "support_under", fitted)
    result = preparation.compile_website_scene_preparation(**args)
    assert len(calls) == 2
    assert "task_distinct_destination_surface_binding_required" not in result["blockers"]
    assert result["destination"]["basis"] == "registered_estimated_visible_bounds_and_collider_contact"
    assert ("support" in result["destination"]) is distinct and ("destination_support" in result) is distinct
    if distinct:
        support = result["destination"]["support"]
        assert set(support) == {"description", "aabb_min_xyz", "aabb_max_xyz"}
        assert all(a <= b for a, b in zip(support["aabb_min_xyz"], support["aabb_max_xyz"]))
        assert result["destination_support"]["face_indices"] == list(range(50, 150))
        assert result["intake_request"]["task"]["destination"]["support"] == support


def test_source_far_above_support_is_not_snapped_down_to_floor(tmp_path):
    args = _arguments(tmp_path)
    frame = args["source_geometry"]["frames"][0]
    with np.load(frame["geometry_path"]) as data:
        depth, valid = data["depth_m"], data["valid_mask"]
    depth[20:25, 21:25] *= 0.2
    np.savez(frame["geometry_path"], depth_m=depth, valid_mask=valid)
    frame["geometry_digest"] = _sha256_file(Path(frame["geometry_path"]))
    args["source_geometry"]["digest"] = canonical_digest(args["source_geometry"], digest_field="digest")
    args["task_masks"]["source_geometry_digest"] = args["source_geometry"]["digest"]
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    result = preparation.compile_website_scene_preparation(**args)
    assert result["status"] == "needs_input"
    assert "support_surface_not_found_under_subject" in result["blockers"]
    assert result["compose_back"]["pose_world"]["support_snap_runtime_units"] == 0


def test_website_task_and_original_frames_reach_artisan_request(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_TASK_OBJECT_AUTHORING_BACKEND", "artisan_v1")  # switched on (go-live)
    from blueprint_pipeline.website_native_inputs import preflight_website_authoring_request
    from tests.test_website_native_inputs import packet
    # Exercise the website's complete consent and source bindings, rather than
    # substituting a legacy rights record and an empty construction envelope.
    envelope, configs = packet(tmp_path)
    source = next(row for row in envelope["materialized_references"]
                  if row["contract_path"] == "scene.source_manifest")
    prepared = json.loads(Path(source["materialized_path"]).read_text())
    request = preflight_website_authoring_request(envelope=envelope, configurations=configs)
    assert request["construction_constraints"] == configs["stage-3"]["construction_constraints"]
    assert request["construction_constraints"]["rebuild_only_this_subject"] is True
    assert request["dimension_authority"] == "estimated"
    assert request["source_frames"] and request["view_selection"]["omitted"] == []
    assert request["dimensions_m"] == pytest.approx(prepared["physics"]["dimensions_m"])


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


def test_missing_anchor_refuses_before_decoding_depth_maps(tmp_path, monkeypatch):
    geometry = _source_geometry(tmp_path)
    base = _anchored_base_scene(tmp_path, geometry)
    def expensive_depth_decode(*_args, **_kwargs):
        pytest.fail("missing declared anchor must be checked first")
    monkeypatch.setattr(preparation, "_source_points", expensive_depth_decode)
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


def test_a_registration_refusal_keeps_the_scores_it_was_decided_on(tmp_path, monkeypatch):
    """2026-09-27 website dishwasher: the conflict refusal recorded only its code."""
    evidence = {"anchored": {"score_m": 0.2}, "conflicting_unconstrained": [{"score_m": 0.05}]}

    def refuse(**_kwargs):
        raise preparation.RegistrationRefusal("website_registration_conflicts_provider_anchor", evidence)
    monkeypatch.setattr(preparation, "register_source_to_runtime", refuse)
    base = {}
    with pytest.raises(ValueError, match="website_registration_conflicts_provider_anchor"):
        _compile(tmp_path, lambda geometry: {"base_scene": base.setdefault("scene", _anchored_base_scene(tmp_path, geometry))})
    written = [json.loads(path.read_text()) for path in tmp_path.rglob("registration_refusal.json")]
    assert len(written) == 1
    assert written[0]["blocker"] == "website_registration_conflicts_provider_anchor"
    assert written[0]["anchored"] == evidence["anchored"]
    assert written[0]["collision_mesh_digest"] == base["scene"]["collision_mesh_digest"]
    assert written[0]["anchor"]["frame_id"] == "frame-0"


def test_anchored_world_compiles_with_the_declared_scale(tmp_path):
    value = _compile(tmp_path, lambda geometry: {"base_scene": _anchored_base_scene(tmp_path, geometry)})
    assert value["status"] == "intake_ready", value["blockers"]
    assert value["registration"]["scale_status"] == "provider_declared_anchor"
    assert value["coordinate_frame"]["declared_meters_per_unit"] == pytest.approx(1 / RUNTIME_SCALE)
    assert value["physics"]["dimensions_m"] == pytest.approx([0.16150833333333348, 0.0331, 0.21939], abs=1e-6)


def test_declared_scale_without_an_anchor_must_agree_with_the_registration(tmp_path):
    def base(geometry):
        scene = _base_scene(tmp_path, geometry)
        return {"base_scene": {**scene, "meters_per_unit": 8.0}}
    value = _compile(tmp_path, base)
    assert "website_registration_scale_conflicts_declared" in value["blockers"]


def _levelled_room_seen_from_a_pitched_camera(tmp_path, monkeypatch, *, pitch_degrees):
    """A gravity-levelled generated room (Y down, floor 1.5 m below the first
    camera) observed by a first camera pitched down, as phone walkthroughs are."""
    rng = np.random.default_rng(0)
    half_x, half_z, count = 2.0, 1.5, 1500
    floor = np.c_[rng.uniform(-half_x, half_x, 2 * count), np.full(2 * count, 1.5), rng.uniform(-half_z, half_z, 2 * count)]
    ceiling = np.c_[rng.uniform(-half_x, half_x, count), np.full(count, -1.2), rng.uniform(-half_z, half_z, count)]
    walls = []
    for side in (-1, 1):
        walls.append(np.c_[np.full(count // 2, side * half_x), rng.uniform(-1.2, 1.5, count // 2),
                           rng.uniform(-half_z, half_z, count // 2)])
        walls.append(np.c_[rng.uniform(-half_x, half_x, count // 2), rng.uniform(-1.2, 1.5, count // 2),
                           np.full(count // 2, side * half_z)])
    counter = np.c_[rng.uniform(0.5, half_x, count // 2), np.full(count // 2, 0.6),
                    rng.uniform(half_z - 0.6, half_z, count // 2)]
    world = np.concatenate([floor, ceiling, *walls, counter])
    c, s = math.cos(math.radians(pitch_degrees)), math.sin(math.radians(pitch_degrees))
    camera_from_world = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    # The estimate's world is the pitched first camera; the generated world is levelled.
    monkeypatch.setattr(preparation, "_source_points", lambda _geometry: world @ camera_from_world.T)
    mpu = 1.25
    mesh_path = tmp_path / "levelled-collider.glb"
    vertices = world / mpu
    faces = np.c_[np.arange(len(vertices) - 2), np.arange(1, len(vertices) - 1), np.arange(2, len(vertices))]
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh_path)
    geometry = {"frames": [{"frame_id": "frame-0", "world_from_camera": np.eye(4).tolist()}], "digest": "sha256:" + "0" * 64}
    anchor = {"kind": "first_input_view_camera", "frame_id": "frame-0", "meters_per_unit": mpu,
              "up_axis": "-Y", "ground_plane_offset_m": 1.5}
    return geometry, mesh_path, anchor


def test_pitched_first_camera_is_levelled_by_the_observed_declared_floor(tmp_path, monkeypatch):
    geometry, mesh_path, anchor = _levelled_room_seen_from_a_pitched_camera(tmp_path, monkeypatch, pitch_degrees=30)
    value = preparation.register_source_to_runtime(source_geometry=geometry, collision_mesh_path=mesh_path,
                                                   anchor=anchor)
    report = value["anchor"]
    assert report["hypothesis_axis"] == "declared_up"
    assert report["levelled_tilt_degrees"] == pytest.approx(30, abs=1)
    assert report["roll_degrees"] == 0 and report["rotation_deviation_degrees"] < 1
    assert report["scale_ratio_to_declared"] == pytest.approx(1.0, abs=0.01)
    assert value["ground_plane"]["checked"] is True and value["ground_plane"]["residual_m"] < 0.05
    assert value["runner_up_ratio"] >= 1.2


def test_unlevelled_prior_from_a_pitched_camera_stays_a_typed_refusal(tmp_path, monkeypatch):
    # Without an observed floor at the declared height, the camera's own axes are
    # the only prior; the pitched view must refuse rather than guess a pose.
    geometry, mesh_path, anchor = _levelled_room_seen_from_a_pitched_camera(tmp_path, monkeypatch, pitch_degrees=30)
    monkeypatch.setattr(preparation, "_observed_declared_ground", lambda *_args: None)
    with pytest.raises(ValueError, match="website_registration_"):
        preparation.register_source_to_runtime(source_geometry=geometry, collision_mesh_path=mesh_path,
                                               anchor=anchor)


def _yaw(degrees):
    angle = np.radians(degrees)
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])


def test_a_shared_camera_alignment_anchors_registration_in_any_orientation(tmp_path):
    geometry = _source_geometry(tmp_path)
    rotation, translation = _yaw(35.0), np.array([0.4, -0.2, 0.3])
    base = _anchored_base_scene(tmp_path, geometry, rotation=rotation, translation=translation,
                                ground_plane_offset_m=None)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation / RUNTIME_SCALE
    anchor = {"kind": "source_world_alignment", "source_world_to_provider_m": transform.tolist(),
              "meters_per_unit": 1 / RUNTIME_SCALE, "up_axis": "Y", "ground_plane_offset_m": None}
    value = preparation.register_source_to_runtime(
        source_geometry=geometry, collision_mesh_path=Path(base["collision_mesh_path"]), anchor=anchor)
    assert value["scale_status"] == "provider_declared_anchor"
    assert value["scale"] == pytest.approx(RUNTIME_SCALE, rel=1e-3)
    assert np.allclose(value["rotation"], rotation, atol=1e-3)
    assert value["translation"] == pytest.approx(translation.tolist(), abs=1e-3)
    assert value["anchor"]["kind"] == "source_world_alignment" and value["anchor"]["frame_id"] is None
    assert value["anchor"]["roll_degrees"] == 0 and value["anchor"]["rotation_deviation_degrees"] < 0.5
    assert value["ground_plane"] is None


def test_a_levelled_atlas_world_checks_its_floor_through_the_alignment_alone(tmp_path, monkeypatch):
    # atlas_world_levelling leaves an Atlas world Y-up with its floor at y = 0;
    # the source reaches it through the shared-view alignment, so nothing
    # here may level it again as if it were anchored at a first camera.
    rng = np.random.default_rng(1)
    floor = np.c_[rng.uniform(-2.0, 2.0, 3000), np.zeros(3000), rng.uniform(-1.5, 1.5, 3000)]
    walls = [np.c_[np.full(750, side * 2.0), rng.uniform(0.0, 2.5, 750), rng.uniform(-1.5, 1.5, 750)]
             for side in (-1, 1)]
    walls += [np.c_[rng.uniform(-2.0, 2.0, 750), rng.uniform(0.0, 2.5, 750), np.full(750, side * 1.5)]
              for side in (-1, 1)]
    counter = np.c_[rng.uniform(0.5, 2.0, 750), np.full(750, 0.9), rng.uniform(0.9, 1.5, 750)]
    world = np.concatenate([floor, *walls, counter])
    rotation, translation = _yaw(35.0), np.array([0.4, -1.2, 0.3])
    monkeypatch.setattr(preparation, "_source_points", lambda _geometry: (world - translation) @ rotation)

    def first_camera_levelling(*_args):
        raise AssertionError("an Atlas world is levelled by its scene source")
    monkeypatch.setattr(preparation, "_observed_declared_ground", first_camera_levelling)
    mpu = 0.8
    mesh_path = tmp_path / "levelled-atlas.glb"
    vertices = world / mpu
    faces = np.c_[np.arange(len(vertices) - 2), np.arange(1, len(vertices) - 1), np.arange(2, len(vertices))]
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh_path)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    anchor = {"kind": "source_world_alignment", "source_world_to_provider_m": transform.tolist(),
              "meters_per_unit": mpu, "up_axis": "Y", "ground_plane_offset_m": 0.0}
    value = preparation.register_source_to_runtime(source_geometry={"frames": [], "digest": "sha256:" + "0" * 64},
                                                   collision_mesh_path=mesh_path, anchor=anchor)
    assert value["anchor"]["rotation_deviation_degrees"] < 1 and value["trimmed_rmse_m"] < 0.01
    ground = value["ground_plane"]
    assert ground["checked"] is True and ground["declared_offset_m"] == 0.0 and ground["residual_m"] < 0.02


@pytest.mark.parametrize("transform", [np.diag([1.0, 1.0, -1.0, 1.0]), np.diag([2.0, 2.0, 2.0, 1.0]),
                                       np.full((4, 4), np.nan)])
def test_an_alignment_that_is_not_a_proper_rigid_motion_is_refused(tmp_path, transform):
    geometry = _source_geometry(tmp_path)
    base = _anchored_base_scene(tmp_path, geometry, ground_plane_offset_m=None)
    anchor = {"kind": "source_world_alignment", "source_world_to_provider_m": transform.tolist(),
              "meters_per_unit": 1 / RUNTIME_SCALE, "up_axis": "Y", "ground_plane_offset_m": None}
    with pytest.raises(ValueError, match="website_registration_anchor_frame_invalid"):
        preparation.register_source_to_runtime(source_geometry=geometry,
            collision_mesh_path=Path(base["collision_mesh_path"]), anchor=anchor)


SYNTHETIC_BALL = {"semantic_label": "ball", "shape": "sphere", "dimensions_m": [0.065, 0.065, 0.065], "mass_kg": 0.06,
                  "task_basis_quote": "a synthetic ball", "start_support_target_id": "table-1",
                  "decision_reason": "The task adds a ball the video does not show.",
                  "design_basis": "synthetic_design_choice", "physical_measurement": False}


def synthetic_arguments(args):
    """The fixture task, with its cup replaced by a designed ball starting on the observed table."""
    masks = args["task_masks"]
    masks["targets"][0] = {**masks["targets"][0], "target_id": "table-1", "semantic_label": "table",
                           "task_effect": "static_contact", "disposition": "keep", "target_role": "support",
                           "estimated_visible_bounds": _bounds([-0.25, 0.48, 1.3], [0.25, 0.5, 1.7])}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    args["removal_manifest"] = {"schema_version": REMOVAL["schema_version"], "entries": [],
                                "synthetic_subject": dict(SYNTHETIC_BALL)}
    context = {**args["task_context"], "description": "Place a synthetic ball onto the tray."}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    args["task_context"] = context
    return args


@pytest.mark.parametrize("family,support_mechanism", [(None, ""), ("rigid_pick_and_place", "prismatic")])
def test_a_designed_ball_starts_on_the_named_support_and_is_authored_as_supplied_geometry(
        tmp_path, family, support_mechanism):
    from blueprint_pipeline.website_synthetic_subject import AUTHORING_METHOD

    arguments = synthetic_arguments(_arguments(tmp_path))
    arguments["task_masks"]["targets"][0]["articulation_kind"] = support_mechanism
    arguments["task_masks"]["digest"] = canonical_digest(arguments["task_masks"], digest_field="digest")
    if family:
        arguments["task_context"]["task_family"] = family
        arguments["task_context"]["context_digest"] = canonical_digest(arguments["task_context"], digest_field="context_digest")
    result = preparation.compile_website_scene_preparation(**arguments)
    assert result["status"] == "intake_ready", result["blockers"]
    subject = result["subject"]
    assert subject["geometry_origin"] == "synthetic_design" and subject["complete_object_dimensions"] is True
    dims = [b - a for a, b in zip(subject["aabb_min_xyz"], subject["aabb_max_xyz"])]
    assert dims == pytest.approx([0.065] * 3, abs=1e-6)
    authoring = result["authoring_inputs"]
    assert authoring["adapter"] == "provided_mesh_rigid_authoring" and authoring["source_frames"] == []
    assert authoring["configuration"]["authoring_method"] == AUTHORING_METHOD
    assert "authoring_backend" not in authoring["configuration"]
    assert authoring["configuration"]["required_output"]["mass_kg_bounds"] == pytest.approx([0.054, 0.066])
    assert result["recipe_plan"]["rigid_replacement_authoring"] == "provided_mesh_rigid_authoring"
    start = result["synthetic_subject"]["start"]
    assert start["rule"] == "closest_clear_spot_on_named_support_to_destination" and start["support_lookups"] >= 1
    # The ball rests on the table it was given, not wherever its region box ends.
    assert result["support"]["face_indices"]


def test_a_designed_rigid_subject_cannot_be_compiled_as_an_articulated_task(tmp_path, monkeypatch):
    arguments = synthetic_arguments(_arguments(tmp_path))
    arguments["task_context"]["task_family"] = "articulated_open_close"
    arguments["task_context"]["context_digest"] = canonical_digest(arguments["task_context"], digest_field="context_digest")
    # A mechanism on the support does not articulate the designed ball.
    arguments["task_masks"]["targets"][0]["articulation_kind"] = "prismatic"
    arguments["task_masks"]["digest"] = canonical_digest(arguments["task_masks"], digest_field="digest")
    monkeypatch.setattr(preparation, "register_source_to_runtime", lambda **_: pytest.fail("Hold before registration"))
    result = preparation.compile_website_scene_preparation(**arguments)
    assert result["blockers"] == ["website_task_profile_mechanism_mismatch"]
    assert result["native_task_preparation_gap"]["inferred_profile_id"] == "rigid_pick_place"
    assert result["intake_request"] is None


def test_a_designed_subject_cannot_stand_beside_a_removed_captured_one(tmp_path):
    args = synthetic_arguments(_arguments(tmp_path))
    args["removal_manifest"]["entries"] = list(REMOVAL["entries"])
    with pytest.raises(ValueError, match="website_synthetic_subject_binding_invalid"):
        preparation.compile_website_scene_preparation(**args)


def test_generated_world_past_the_footage_keeps_the_anchored_pose(tmp_path, monkeypatch):
    # 2026-10-30 capture-447ae4f4: the Atlas collider extended far past the observed footage.
    # Rotated unconstrained poses won the two-way score only by covering more unobserved
    # surface, and fit the observed source worse; the exact anchor was refused.
    def grid(a0, a1, b0, b1, step):
        a, b = np.meshgrid(np.arange(a0, a1 + 1e-9, step), np.arange(b0, b1 + 1e-9, step))
        return a.ravel(), b.ravel()

    def floor(extent):
        x, y = grid(0.0, extent, 0.0, extent, 0.1)
        return np.stack([x, y, np.zeros_like(x)], 1)

    def box(centre, size):
        x, y = grid(centre - size / 2, centre + size / 2, centre - size / 2, centre + size / 2, 0.05)
        a, z = grid(centre - size / 2, centre + size / 2, 0.0, size, 0.05)
        low, high = np.full_like(a, centre - size / 2), np.full_like(a, centre + size / 2)
        return np.concatenate([np.stack([x, y, np.full_like(x, size)], 1), np.stack([low, a, z], 1),
                               np.stack([high, a, z], 1), np.stack([a, low, z], 1), np.stack([a, high, z], 1)])

    observed = np.concatenate([floor(2.0), box(1.0, 0.5)])
    generated = np.concatenate([floor(4.0), box(1.0, 0.5)])
    turn = preparation._roll(90.0)

    def refine(_source, _target, scale, rotation, translation):
        if math.isclose(scale, 1.0):
            return scale, rotation, translation
        # Every unconstrained seed converges to the observed patch turned and moved to the middle.
        return 1.0, turn, np.array([2.0, 2.0, 0.0]) - turn @ np.array([1.0, 1.0, 0.0])

    monkeypatch.setattr(preparation, "_source_points", lambda _geometry: observed)
    monkeypatch.setattr(preparation, "_mesh_vertices", lambda _path: generated)
    monkeypatch.setattr(preparation, "_refine_registration", refine)
    mesh = tmp_path / "collider.glb"
    mesh.write_bytes(b"collider")
    value = preparation.register_source_to_runtime(
        source_geometry={"frames": [], "digest": "sha256:" + "0" * 64}, collision_mesh_path=mesh,
        anchor={"kind": "source_world_alignment", "source_world_to_provider_m": np.eye(4).tolist(),
                "meters_per_unit": 1.0, "up_axis": "Z", "ground_plane_offset_m": None})
    report = value["anchor"]
    assert np.allclose(value["source_to_runtime"], np.eye(4))
    assert report["forward_score_m"] == pytest.approx(0.0, abs=1e-9)
    rotated = report["two_way_only_unconstrained"]
    assert rotated and all(row["score_m"] * 1.2 < report["score_m"] for row in rotated)
    assert all(row["forward_score_m"] > 0.1 for row in rotated)
    assert {round(row["rotation_from_anchored_degrees"]) for row in rotated} == {90}


@pytest.mark.parametrize("residual_m,found", [(0.07, True), (0.01, False)])
def test_subject_estimated_into_its_support_is_snapped_within_the_measured_residual(tmp_path, monkeypatch,
                                                                                   residual_m, found):
    # capture-447ae4f4: the estimated tray bottom sat 4 cm into the counter, and the
    # registration's residual near the task was 7.2 cm.
    arguments = _arguments(tmp_path)
    assert arguments["base_scene"]["up_axis"] == "Y"
    mpu = arguments["base_scene"]["meters_per_unit"]
    real_bounds, real_registration = preparation.estimate_target_bounds, preparation.register_source_to_runtime

    def into_support(*args, **kwargs):
        value = real_bounds(*args, **kwargs)
        # The table top is runtime y = 1.75 (test_registered_estimates_compile_into_an_intake_ready_request).
        shift = (1.75 - 0.04 / mpu) - value["minimum"][1]
        value["minimum"][1] += shift
        value["maximum"][1] += shift
        return value

    def measured(**kwargs):
        value = real_registration(**kwargs)
        return {**value, "task_region": {**(value.get("task_region") or {}), "trimmed_rmse_m": residual_m}}

    monkeypatch.setattr(preparation, "estimate_target_bounds", into_support)
    monkeypatch.setattr(preparation, "register_source_to_runtime", measured)
    value = preparation.compile_website_scene_preparation(**arguments)
    if not found:
        assert "support_surface_not_found_under_subject" in value["blockers"]
        return
    assert "support_surface_not_found_under_subject" not in value["blockers"]
    assert value["support"]["search_window_m"] == pytest.approx(0.07)
    assert value["support"]["search_window_basis"] == "registration_task_region_residual"
    assert abs(value["support"]["subject_snap_m"]) == pytest.approx(0.04, abs=1e-6)
    assert value["support"]["subject_snap_basis"] == "estimate"
    # Snapped back onto the same table top as the unshifted estimate.
    assert value["subject"]["aabb_min_xyz"][2] == pytest.approx(3.5, abs=1e-6)


def test_the_destination_support_is_searched_within_the_same_measured_residual(tmp_path, monkeypatch):
    # capture-447ae4f4: the destination on the same counter was refused over a 0.4 cm mesh gap
    # at the default window, while the subject's support used the 7.2 cm residual.
    real, calls = preparation.support_under, []

    def recording(*args, **kwargs):
        calls.append(kwargs.get("placement_residual_m"))
        return real(*args, **kwargs)

    monkeypatch.setattr(preparation, "support_under", recording)
    value = _compile(tmp_path)
    assert value["status"] == "intake_ready", value["blockers"]
    assert calls == [value["registration"]["task_region"]["trimmed_rmse_m"]] * 2
