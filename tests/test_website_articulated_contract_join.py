"""ADP-009/ADP-030: website coverage -> compile -> articulated builder, joined end to end.

Real coverage selection, sizing, compile, observation handoff, assembly
planning and per-part brief construction; only the classifier answer is given.
No provider is called.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import authoring_frame_budget as budget
from blueprint_pipeline import task_evaluation_scene_configuration_astra_driver as driver
from blueprint_pipeline import website_assembly_coverage as coverage
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.task_object_articulated_packaging import plan_articulated_assembly
from blueprint_pipeline.website_articulated_mass import articulated_mass_bounds
from blueprint_pipeline.website_object_observations import (
    materialize_object_observations, validate_observation_handoff,
)
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from tests.test_website_task_preparation import (
    CLOSED_FRONT, HEIGHT, WIDTH, _arguments, _assembly_inputs, _depth, _frame, _masks,
)

DISHWASHER_REMOVAL = {"schema_version": "clean_plate_removal_manifest.v1", "entries": [
    {"target_id": "cup-1", "semantic_label": "dishwasher", "task_effect": "manipulated", "disposition": "remove",
     "articulated_part": "dishwasher door", "articulation_kind": "revolute",
     "compose_back": {"replacement_asset_id": None, "pose_world": None, "replacement_asset_frame_registration_uri": None}}]}
TUB_DEPTH_M = 0.6
# The door folds down: the open view sees the tub 0.6 m behind the closed
# front and the lowered door in front of it; only the tub is body depth.
OPEN_TUB = ((14, 24), (15, 26), 1.2 + TUB_DEPTH_M)
LOWERED_DOOR = ((24, 28), (14, 27), 0.95)
DISHWASHER_VIEWS = [
    ("frame-0", 0.0, "closed", "front", ["body", "control_panel", "door", "handle"]),
    ("frame-1", 0.5, "open", "front", ["body", "door", "lower_rack", "tub", "upper_rack"]),
    ("decoded-000000030", 1.0, "partially_open", "left_oblique", ["door", "handle", "tub"]),
    ("decoded-000000045", 1.5, "open", "top_down", ["lower_rack", "tub", "upper_rack"]),
    ("decoded-000000060", 2.0, "closed", "right_oblique", ["body", "control_panel"]),
]
REQUIRED = {"body", "tub", "door", "control_panel", "handle", "upper_rack", "lower_rack"}


def _classified(views, root: Path, *, kind: str, hinge: str | None, components):
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, (frame_id, timestamp, _, _, _) in enumerate(views):
        path = root / f"{frame_id}.png"
        Image.fromarray(np.full((HEIGHT, WIDTH, 3), 60 + 7 * index, dtype=np.uint8)).save(path)
        frames.append({"frame_id": frame_id, "timestamp_seconds": timestamp, "mask_area_fraction": 0.02,
                       "geometry_frame": frame_id.startswith("frame-"), "path": str(path), "sha256": _sha256_file(path)})
    answer = {"hinge_edge": hinge, "task_part_components": components,
              "frames": [{"frame_id": row[0], "visible_parts": row[4], "task_part_state": row[2], "view": row[3],
                         "label_text": []}
                         for row in views]}
    labels = coverage.validate_classification(answer, frame_ids=[row[0] for row in views], articulation_kind=kind)
    return {"frames": [{**frame, **label} for frame, label in zip(frames, labels["frames"])],
            "task_part_components": labels["task_part_components"],
            "hinge_edges": [labels["hinge_edge"]] if labels["hinge_edge"] else [],
            "receipts": [{"binding_digest": "sha256:" + "1" * 64, "result_digest": "sha256:" + "2" * 64}]}


def _dishwasher_inputs(root: Path, *, object_spec=None, views=DISHWASHER_VIEWS):
    root = root / "dishwasher"
    root.mkdir(parents=True, exist_ok=True)
    frames, observations = [], []
    for index, boxes in enumerate(((CLOSED_FRONT,), (OPEN_TUB, LOWERED_DOOR))):
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
    target = {**masks["targets"][0], "semantic_label": "dishwasher", "articulated_part": "dishwasher door",
              "articulation_kind": "revolute", "track": {**masks["targets"][0]["track"], "observations": observations}}
    classified = _classified(views, root / "views", kind="revolute", hinge="bottom",
                             components=["control_panel", "door", "handle"])
    states = {row["frame_id"]: row["part_state"] for row in classified["frames"]}
    body, blockers = coverage.estimate_body_bounds(track=target["track"], frames=frames, states=states)
    target["authoring_coverage"] = coverage.coverage_record(
        target_id=target["target_id"], binding=coverage.coverage_binding(target=target, source_geometry=geometry),
        articulation_kind="revolute", classified=classified,
        selected=coverage.select_reference_frames(classified["frames"],
                                                  seed_frame_ids=coverage.depth_seed(classified["frames"], body)),
        body_bounds=body, body_blockers=blockers, candidate_count=len(views))
    if object_spec is not None:
        target["object_spec"] = object_spec
    masks = {**masks, "targets": [target]}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    return {"source_geometry": geometry, "removal_manifest": DISHWASHER_REMOVAL, "task_masks": masks}


def _briefs(tmp_path, monkeypatch, args, preparation):
    """The builder's own request translation, from the materialized observation handoff."""
    from blueprint_pipeline import website_native_inputs

    handoff = materialize_object_observations(preparation=preparation, source_geometry=args["source_geometry"],
                                              task_masks=args["task_masks"], output_root=tmp_path / "observations")
    configuration = handoff["configuration"]
    _, references = validate_observation_handoff(Path(handoff["manifest"]["path"]), configuration=configuration)
    # The envelope and rights re-binding is pinned in test_website_native_inputs.
    monkeypatch.setattr(website_native_inputs, "validate_website_authoring_disclosure", lambda **_: None)
    envelope = {"run_id": "join-run", "envelope_digest": ""}
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    stage_input = {"run_id": "join-run", "source_commit": "a" * 40, "construction_envelope": envelope,
                   "configuration": configuration, "configuration_sha256": canonical_digest(configuration)}
    source = driver._file_record(Path(handoff["candidate"]["path"]))
    plan, requests = driver.build_articulated_authoring_requests(stage_input, source, references, {})
    return configuration, references, plan, requests


def test_dishwasher_coverage_compiles_plans_and_briefs_end_to_end(tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    args.update(_dishwasher_inputs(tmp_path))
    args["removal_manifest"] = DISHWASHER_REMOVAL
    preparation = compile_website_scene_preparation(**args)
    assert preparation["status"] == "intake_ready", preparation["blockers"]
    configuration = preparation["authoring_inputs"]["configuration"]
    assert configuration["assembly_family"] == "hinged_door_appliance" and configuration["hinge_edge"] == "bottom"
    roles = {row["part_id"]: row["role"] for row in configuration["required_parts"]}
    assert roles == {"body": "body_feature", "tub": "body", "door": "task_part", "control_panel": "door_feature",
                     "handle": "door_feature", "upper_rack": "fixed_interior", "lower_rack": "fixed_interior"}
    # Simulator metres: the tub seen 0.6 m behind the closed front (registration scale 0.5 x 2).
    depth = configuration["body_depth"]
    assert depth["value_m"] == pytest.approx(TUB_DEPTH_M, abs=0.03) and depth["frame_ids"] == ["frame-1"]
    assert configuration["body_extent_m"]["depth"] == depth["value_m"]
    assert configuration["mechanism"]["estimated_usable_swing_rad"] == pytest.approx(math.pi / 2)
    output = configuration["required_output"]
    assert output["fixed_part_mass_kg_bounds"][0] > 0 and configuration["mass_authority"] == "estimated"

    plan = plan_articulated_assembly(configuration)
    assert {row["link_id"] for row in plan["links"]} == {"body", "door", "upper_rack", "lower_rack"}
    joint = plan["task_joint"]
    assert joint["joint_type"] == "revolute" and joint["axis_asset_frame"] == [0.0, 1.0, 0.0]
    assert joint["anchor_asset_frame_m"][2] == 0.0 and joint["limits_rad"] == [0.0, math.pi / 2]
    assert plan["assembly_dimensions_m"]["depth_x"] == round(depth["value_m"], 5)
    assert {row["part_id"] for row in plan["required_parts"]} == REQUIRED

    configuration, references, plan, requests = _briefs(tmp_path, monkeypatch, args, preparation)
    assert set(requests) == {"body", "door", "upper_rack", "lower_rack"}
    rows = {row["sha256"]: row for row in configuration["reference_frames"]}
    frames = requests["door"].source_frames
    assert [frame.sha256 for frame in frames] == [_sha256_file(path) for path in references]
    assert len(frames) == len(configuration["reference_frames"]) == len(DISHWASHER_VIEWS)
    for frame in frames:
        row = rows[frame.sha256]
        assert f"Original capture frame {row['frame_id']}" in frame.description
        assert row["reason"] in frame.description and row["transmission"]["source_sha256"] != frame.sha256
    assert "the dishwasher door is open" in frames[1].description
    brief = json.dumps([request.model_dump(mode="json") for request in requests.values()]).lower()
    assert "wood-grain" not in brief and "silver bar handles" not in brief
    bounds = driver._articulated_physics_bounds(configuration, plan)
    assert bounds["upper_rack"]["mass_kg"] == output["fixed_part_mass_kg_bounds"]
    assert bounds["door"]["mass_kg"] == output["task_part_mass_kg_bounds"]


def test_drawer_coverage_compiles_plans_and_briefs_end_to_end(tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    args.update(_assembly_inputs(tmp_path))
    preparation = compile_website_scene_preparation(**args)
    assert preparation["status"] == "intake_ready", preparation["blockers"]
    configuration, references, plan, requests = _briefs(tmp_path, monkeypatch, args, preparation)
    assert plan["family"] == "stacked_drawer_cabinet" and plan["task_joint"]["joint_type"] == "prismatic"
    assert set(requests) == {"carcass", "drawer"}
    frames = requests["drawer"].source_frames
    assert len(frames) == len(configuration["reference_frames"]) == len(references)
    assert all("Original capture frame" in frame.description for frame in frames)
    brief = json.dumps([request.model_dump(mode="json") for request in requests.values()]).lower()
    assert "wood-grain" not in brief and "fixed_part_mass_kg_bounds" not in configuration["required_output"]


def test_yawed_body_keeps_its_own_extent_and_the_builder_checks_it(tmp_path):
    args = _arguments(tmp_path)
    args.update(_dishwasher_inputs(tmp_path))
    args["removal_manifest"] = DISHWASHER_REMOVAL
    configuration = compile_website_scene_preparation(**args)["authoring_inputs"]["configuration"]
    yaw = math.radians(35)
    extent = configuration["body_extent_m"]
    half = [(extent["depth"] * math.cos(yaw) + extent["width"] * math.sin(yaw)) / 2,
            (extent["depth"] * math.sin(yaw) + extent["width"] * math.cos(yaw)) / 2, extent["height"] / 2]
    yawed = {**configuration, "mechanism": {**configuration["mechanism"],
                                            "estimated_front_normal_world": [math.cos(yaw), math.sin(yaw), 0.0]},
             "metric_envelope": {**configuration["metric_envelope"], "minimum_xyz_m": [-half[0], -half[1], 0.0],
                                 "maximum_xyz_m": [half[0], half[1], 2 * half[2]]}}
    plan = plan_articulated_assembly(yawed)
    # The yawed world box is ~1.4x deeper than the body; the plan keeps the body.
    assert plan["assembly_dimensions_m"]["depth_x"] == round(extent["depth"], 5)
    assert plan["assembly_dimensions_m"]["width_y"] == round(extent["width"], 5)
    wrong = {**yawed, "body_extent_m": {**extent, "depth": 2 * extent["depth"]}}
    with pytest.raises(Exception, match="articulated_body_extent_disagrees_with_envelope"):
        plan_articulated_assembly(wrong)


def test_builder_refusal_holds_compile_before_anything_is_bought(tmp_path):
    views = [row if row[0] != "decoded-000000045" else (*row[:4], ["lower_rack", "spray_arm", "tub", "upper_rack"])
             for row in DISHWASHER_VIEWS]
    args = _arguments(tmp_path)
    args.update(_dishwasher_inputs(tmp_path, views=views))
    args["removal_manifest"] = DISHWASHER_REMOVAL
    preparation = compile_website_scene_preparation(**args)
    assert preparation["status"] == "needs_input"
    assert "website_assembly_builder_refused:articulated_required_part_unplanned:spray_arm" in preparation["blockers"]


def _published_spec(**specs):
    value = {"schema_version": "website_object_spec.v1",
             "identity": {"brand": "ExampleBrand", "model": "EX-24", "basis": "label_read"}, "specs": specs}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


def _weight(value, match, unit="kg"):
    return {"value": value, "unit": unit, "source_urls": ["https://example.com/ex-24/specs"], "match": match}


def test_mass_bounds_use_published_weights_when_identified_and_estimates_otherwise():
    target = {"target_id": "dw-1"}
    extent = [0.6, 0.6, 0.85]
    estimated = articulated_mass_bounds(target, joint_type="revolute", body_extent_m=extent,
                                        fixed_part_ids=["upper_rack", "lower_rack"])
    assert estimated["mass_authority"] == "estimated" and estimated["source_urls"] == []
    assert estimated["task_part_mass_kg_bounds"] == [2.04, 12.75]
    exact = articulated_mass_bounds(target, joint_type="revolute", body_extent_m=extent,
        fixed_part_ids=["upper_rack", "lower_rack"],
        object_spec=_published_spec(weight=_weight(40.0, "exact_model"), door_weight=_weight(17.6, "exact_model", "lb"),
                                    rack_weight=_weight(2.0, "exact_model")))
    assert exact["mass_authority"] == "manufacturer_published"
    assert exact["task_part_mass_kg_bounds"] == pytest.approx([7.184, 8.781], abs=1e-3)
    assert exact["fixed_part_mass_kg_bounds"] == [1.8, 2.2]
    # The body is what the published whole leaves after the door and racks.
    assert exact["mass_kg_bounds"] == pytest.approx([36 - 8.781 - 4.4, 44 - 7.184 - 3.6], abs=1e-2)
    assert exact["provenance"]["body"]["basis"] == "published_weight_minus_other_link_bounds"
    assert exact["source_urls"] == ["https://example.com/ex-24/specs"]
    family = articulated_mass_bounds(target, joint_type="revolute", body_extent_m=extent,
                                     object_spec=_published_spec(door_weight=_weight(8.0, "model_family")))
    assert family["task_part_mass_kg_bounds"] == [6.0, 10.0] and family["mass_authority"] == "mixed"
    unknown = _published_spec(door_weight=_weight(8.0, "exact_model"))
    unknown["identity"]["basis"] = "unknown"
    unknown["digest"] = canonical_digest(unknown, digest_field="digest")
    assert articulated_mass_bounds(target, joint_type="revolute", body_extent_m=extent,
                                   object_spec=unknown)["mass_authority"] == "estimated"
    tampered = {**_published_spec(door_weight=_weight(8.0, "exact_model")), "digest": "sha256:" + "0" * 64}
    with pytest.raises(ValueError, match="website_object_spec_invalid"):
        articulated_mass_bounds(target, joint_type="revolute", body_extent_m=extent, object_spec=tampered)
    unsourced = _published_spec(door_weight={**_weight(8.0, "exact_model"), "source_urls": []})
    with pytest.raises(ValueError, match="website_object_spec_invalid:door_weight"):
        articulated_mass_bounds(target, joint_type="revolute", body_extent_m=extent, object_spec=unsourced)


def test_compile_records_published_mass_authority_from_the_target_spec(tmp_path):
    spec = _published_spec(door_weight=_weight(8.0, "exact_model"), rack_weight=_weight(2.0, "exact_model"))
    args = _arguments(tmp_path)
    args.update(_dishwasher_inputs(tmp_path, object_spec=spec))
    args["removal_manifest"] = DISHWASHER_REMOVAL
    preparation = compile_website_scene_preparation(**args)
    assert preparation["status"] == "intake_ready", preparation["blockers"]
    configuration = preparation["authoring_inputs"]["configuration"]
    assert configuration["required_output"]["task_part_mass_kg_bounds"] == [7.2, 8.8]
    assert configuration["required_output"]["fixed_part_mass_kg_bounds"] == [1.8, 2.2]
    assert configuration["mass_authority"] == "mixed"
    assert configuration["mass_bounds_provenance"]["task_part"]["mass_authority"] == "manufacturer_published"
    assert configuration["mass_source_urls"] == ["https://example.com/ex-24/specs"]


def _rows(count, parts_by_frame):
    rows = []
    for index in range(count):
        rows.append({"frame_id": f"f{index}", "timestamp_seconds": float(index), "selection_rank": index,
                     "part_state": "closed" if index % 2 else "open", "visible_parts": parts_by_frame.get(index, ["body"])})
    return rows


def test_frame_cap_keeps_every_part_state_and_depth_view_by_priority():
    assert budget.frame_cap("openai") == 12 and budget.frame_cap("anthropic") == 8
    rows = _rows(12, {11: ["handle"]})
    parts = [{"part_id": "body", "observed_frame_ids": [f"f{i}" for i in range(11)]},
             {"part_id": "handle", "observed_frame_ids": ["f11"]}]
    kept = budget.choose_frames(rows, required_parts=parts, depth_frame_ids=["f10"], cap=8)
    ids = [row["frame_id"] for row in kept]
    # The lowest-priority frame is kept: it is the only one showing the handle.
    assert len(ids) == 8 and "f11" in ids and "f10" in ids and ids[:6] == ["f0", "f1", "f2", "f3", "f4", "f5"]
    with pytest.raises(budget.FrameBudgetError, match="required_coverage_exceeds_provider_cap"):
        budget.choose_frames(rows, required_parts=parts, depth_frame_ids=["f10"], cap=1)
    with pytest.raises(budget.FrameBudgetError, match="priority_missing"):
        budget.choose_frames([{**row, "selection_rank": None} for row in rows], required_parts=parts,
                             depth_frame_ids=["f10"], cap=8)


def _contract(tmp_path, count, *, noise=False):
    rng = np.random.default_rng(5)
    rows = []
    for index in range(count):
        path = tmp_path / f"orig-{index}.png"
        if noise:  # Incompressible and already at the smallest ladder side.
            pixels = rng.integers(0, 256, (576, 768, 3), dtype=np.uint8)
        else:
            pixels = np.broadcast_to(np.linspace(0, 255, 2000, dtype=np.uint8)[None, :, None], (1500, 2000, 3)).copy()
            pixels[..., 1] = index
        Image.fromarray(pixels).save(path, compress_level=1)
        rows.append({"frame_id": f"f{index}", "timestamp_seconds": float(index), "selection_rank": index,
                     "part_state": "open", "visible_parts": ["body"], "path": str(path), "sha256": _sha256_file(path)})
    return {"reference_frames": rows,
            "required_parts": [{"part_id": "body", "observed_frame_ids": [row["frame_id"] for row in rows]}],
            "body_depth": {"frame_ids": ["f0"]}}


def test_full_resolution_frames_are_sent_as_bounded_derivatives_with_both_digests(tmp_path):
    fitted = budget.fit_reference_frames(_contract(tmp_path, 9), provider="anthropic", output_root=tmp_path / "out")
    frames = fitted["reference_frames"]
    assert len(frames) == 8 and fitted["reference_frame_budget"]["dropped_frame_ids"] == ["f8"]
    for row in frames:
        record = row["transmission"]
        assert (record["source_width"], record["source_height"]) == (2000, 1500)
        assert record["long_side_px"] <= 1568 and record["source_sha256"] != row["sha256"]
        assert _sha256_file(Path(row["path"])) == row["sha256"] and record["bytes"] <= budget.MAX_FRAME_BYTES
    assert fitted["reference_frame_budget"]["tokens_reserved"] == 8 * 4_784
    receipt = budget.check_transmitted_frames([Path(row["path"]) for row in frames], provider="anthropic")
    assert receipt["tokens_reserved"] <= budget.SOURCE_FRAME_TOKEN_BUDGET
    # Deterministic: a rerun reproduces the same transmitted bytes.
    again = budget.fit_reference_frames(_contract(tmp_path, 9), provider="anthropic", output_root=tmp_path / "out")
    assert [row["sha256"] for row in again["reference_frames"]] == [row["sha256"] for row in frames]
    openai = budget.fit_reference_frames(_contract(tmp_path, 9), provider="openai", output_root=tmp_path / "o")
    assert len(openai["reference_frames"]) == 9
    assert openai["reference_frame_budget"]["tokens_reserved"] <= budget.SOURCE_FRAME_TOKEN_BUDGET


def test_frames_that_cannot_fit_fail_closed(tmp_path):
    with pytest.raises(budget.FrameBudgetError, match="authoring_frame_bytes_exceed_budget"):
        budget.fit_reference_frames(_contract(tmp_path, 12, noise=True), provider="openai",
                                    output_root=tmp_path / "out")
    large = tmp_path / "large.png"
    Image.new("RGB", (4032, 3024)).save(large)
    with pytest.raises(budget.FrameBudgetError, match="not_a_bounded_png"):
        budget.check_transmitted_frames([large], provider="openai")
    small = tmp_path / "small.png"
    Image.new("RGB", (64, 48)).save(small)
    with pytest.raises(budget.FrameBudgetError, match="count_exceeds_provider_cap"):
        budget.check_transmitted_frames([small] * 9, provider="anthropic")
