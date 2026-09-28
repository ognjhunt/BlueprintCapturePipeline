"""ADP-030/day 28: whole-assembly views and body size, with no live model calls."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import website_assembly_coverage as coverage
from blueprint_pipeline import website_gemini_receipts as receipts
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

WIDTH, HEIGHT, FOCAL = 60, 40, 50.0
FPS = 30.0
FRONT_Z, INTERIOR_Z, DOOR_Z = 1.0, 1.6, 0.7
FRONT = ((5, 36), (15, 46))  # rows, cols of the closed dishwasher front at FRONT_Z.
CLOSED, OPEN = 0, 60  # decoded indices of the two geometry frames.
FRONT_PARTS = ["body_front", "brand_label", "control_panel", "door_outer", "handle", "kickplate"]
OPEN_PARTS = ["cutlery_basket", "door_inner", "lower_rack", "tub_interior", "upper_rack"]
MOVING = {"brand_label", "control_panel", "door_inner", "door_outer", "handle"}


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.pad(mask.reshape(-1).astype(np.int8), (1, 1))))
    return [{"start": int(a), "length": int(b - a)} for a, b in zip(edges[::2], edges[1::2])]


def _box(rows, cols):
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[rows[0]:rows[1], cols[0]:cols[1]] = True
    return mask


def _geometry_frame(root, index, regions):
    """Regions are (rows, cols, depth); the mask is their union."""
    depth, mask = np.full((HEIGHT, WIDTH), 4.0), np.zeros((HEIGHT, WIDTH), dtype=bool)
    for rows, cols, value in regions:
        depth[rows[0]:rows[1], cols[0]:cols[1]] = value
        mask |= _box(rows, cols)
    path = root / f"decoded-{index:09d}.npz"
    np.savez(path, depth_m=depth, valid_mask=np.ones_like(depth, dtype=bool))
    frame = {"frame_id": f"decoded-{index:09d}", "timestamp_seconds": index / FPS, "geometry_path": str(path),
             "geometry_digest": _sha256_file(path), "intrinsics": [[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]],
             "world_from_camera": np.eye(4).tolist(), "width": WIDTH, "height": HEIGHT, "display_rotation_degrees": 0}
    return frame, {"source_frame_id": frame["frame_id"], "width": WIDTH, "height": HEIGHT, "runs": _runs(mask)}


# Closed: the front. Open: the door folded down in front of the face and the
# tub's back wall 0.6 m behind it. Only the latter is body depth.
CLOSED_REGIONS = [(FRONT[0], FRONT[1], FRONT_Z)]
OPEN_REGIONS = [((3, 28), (17, 44), INTERIOR_Z), ((28, 38), (15, 46), DOOR_Z)]


def _source_geometry(root, *, closed=True, opened=True):
    root.mkdir(parents=True, exist_ok=True)
    rows = ([_geometry_frame(root, CLOSED, CLOSED_REGIONS)] if closed else []) + (
        [_geometry_frame(root, OPEN, OPEN_REGIONS)] if opened else [])
    geometry = {"schema_version": "website_source_geometry.v1", "frames": [row[0] for row in rows],
                "binding": {"source_video_digest": "sha256:" + "a" * 64}, "unit": "estimated_meters"}
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    return geometry, {"track_id": "t1", "label": "dishwasher-1", "observations": [row[1] for row in rows]}


def _target(geometry_track, *, target_id="dishwasher-1", frames=120, tiny=()):
    observations = []
    for index in range(frames):
        mask = _box((12, 30), (20, 40)) if index not in tiny else _box((0, 1), (0, 3))
        observations.append({"source_frame_id": f"decoded-{index:09d}", "width": WIDTH, "height": HEIGHT,
                             "runs": _runs(mask)})
    return {"target_id": target_id, "task_effect": "manipulated", "disposition": "remove",
            "semantic_label": "dishwasher", "articulated_part": "dishwasher door", "articulation_kind": "revolute",
            "track": {**geometry_track, "label": target_id},
            "source_track": {"track_id": "full", "label": target_id, "observations": observations}}


def _registry(frames=120):
    return [{"source_frame_id": f"decoded-{i:09d}", "model_frame_index": i, "decoded_pts_seconds": i / FPS,
             "width": WIDTH, "height": HEIGHT} for i in range(frames)]


def _label(index):
    """What the dishwasher footage shows: front closed, door open, then each side."""
    if index < 40:
        return ["closed", "front", FRONT_PARTS]
    if index < 80:
        return ["open", "interior", OPEN_PARTS]
    if index < 100:
        return ["closed", "left_oblique", ["body_front", "door_outer", "left_side"]]
    return ["partially_open", "right_oblique", ["door_outer", "handle", "right_side"]]


def _answer(binding, *, hinge="bottom"):
    ids = [row["frame_id"] for row in binding["images"]]
    frames = [dict(zip(("task_part_state", "view", "visible_parts"), _label(int(i[8:]))), frame_id=i,
                   label_text=["BOSCH", "800 Series"] if int(i[8:]) < 40 else []) for i in ids]
    visible = {part for row in frames for part in row["visible_parts"]}
    return {"hinge_edge": hinge, "task_part_components": sorted(MOVING & visible), "frames": frames}


def _no_brand(binding):
    return {"frames": [{"frame_id": row["frame_id"], "label_text": []} for row in binding["images"]]}


def _context():
    value = {"schema_version": "website_site_task_context.v1", "request_id": "req", "scene_id": "scene",
             "capture_id": "capture", "confirmed": True,
             "confirmed_at": "2026-09-19T00:00:00Z", "description": "Open and close the dishwasher"}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


@pytest.fixture
def model(monkeypatch):
    """The real retained receipt path with a fake reservation and model answer."""
    calls = []
    real = receipts.retained_gemini_call
    monkeypatch.setattr(receipts, "reserve_website_preparation_spend", lambda **_: ({"status": "admitted"}, object()))
    # The focused brand read is its own retained call, counted apart from the classifier batches.
    state = {"answer": _answer, "brand": _no_brand, "brand_calls": []}

    def retained(**kwargs):
        def invoke():
            if kwargs["binding"]["kind"] == "website_assembly_brand_read":
                state["brand_calls"].append(kwargs["binding"])
                return {"brand_read": state["brand"](kwargs["binding"])}
            calls.append(kwargs["binding"])
            return {"classification": state["answer"](kwargs["binding"])}
        return real(**{**kwargs, "preflight": lambda: None, "invoke": invoke})
    monkeypatch.setattr(receipts, "retained_gemini_call", retained)

    def decode(*, video, video_digest, rotation, frames, output_root):
        output_root.mkdir(parents=True, exist_ok=True)
        result = {}
        for row in frames:
            path = output_root / f"{row['frame_id']}.png"
            Image.new("RGB", (WIDTH, HEIGHT), (row["decoded_index"] % 256, 0, 0)).save(path)
            result[row["frame_id"]] = {"path": str(path), "sha256": _sha256_file(path), "width": WIDTH, "height": HEIGHT}
        return result
    monkeypatch.setattr(coverage, "decode_upright_frames", decode)
    return calls, state


def _run(tmp_path, target, geometry, **overrides):
    arguments = dict(target=target, assembly_label="dishwasher", task_part="dishwasher door",
                     articulation_kind="revolute", registry=_registry(), source_geometry=geometry,
                     task_context=_context(), source_video=tmp_path / "clip.mov", output_root=tmp_path / "coverage")
    return coverage.assembly_coverage(**{**arguments, **overrides})


def test_views_cover_every_part_and_state_within_the_cap_and_restart_is_free(tmp_path, model):
    calls, _ = model
    geometry, track = _source_geometry(tmp_path / "geometry")
    # A 20 s clip: more useful frames than one request may carry.
    target = _target(track, frames=600, tiny={7, 8})
    record = _run(tmp_path, target, geometry, registry=_registry(600))
    assert record["status"] == "complete", record["blockers"]
    assert record["digest"] == canonical_digest(record, digest_field="digest")
    selected = record["selected_frames"]
    assert len(selected) == coverage.MAX_REFERENCE_FRAMES == 12
    shown = {part for row in selected for part in row["visible_parts"]}
    assert shown == set(record["observed_parts"]) == set(FRONT_PARTS + OPEN_PARTS + ["left_side", "right_side"])
    assert record["missing_parts"] == []
    assert {row["part_state"] for row in selected} == {"closed", "open", "partially_open"}
    assert {row["view"] for row in selected} == {"front", "interior", "left_oblique", "right_oblique"}
    assert all(row["reason"] and Path(row["path"]).is_file() and _sha256_file(Path(row["path"])) == row["sha256"]
               for row in selected)
    assert record["part_observed_open"] is True and record["hinge_edge"] == "bottom"
    assert set(record["task_part_components"]) == MOVING
    # Candidates: both geometry frames, never a tiny-mask frame, at least 0.4 s apart otherwise.
    stamps = sorted(int(image["frame_id"][8:]) / FPS for binding in calls for image in binding["images"])
    ids = {image["frame_id"] for binding in calls for image in binding["images"]}
    assert {f"decoded-{CLOSED:09d}", f"decoded-{OPEN:09d}"} <= ids and not ids & {"decoded-000000007", "decoded-000000008"}
    assert len(ids) == record["candidate_frame_count"] <= coverage.MAX_CANDIDATE_FRAMES
    assert all(b - a >= coverage.MIN_CANDIDATE_SPACING_SECONDS - 1e-9 for a, b in zip(stamps, stamps[1:]))
    assert len(calls) == math.ceil(len(ids) / coverage.CLASSIFY_BATCH) == len(record["classifier"]["receipts"])
    assert all(len(binding["images"]) <= coverage.CLASSIFY_BATCH and binding["model"] == "gemini-3.8-flash"
               and binding["target_id"] == "dishwasher-1" for binding in calls)
    # Body: the tub's back wall 0.6 m behind the closed front, not the door's sweep in front of it.
    body = record["body_bounds"]
    assert body["depth_m"] == pytest.approx(INTERIOR_Z - FRONT_Z, abs=1e-6)
    assert body["minimum"][2] == pytest.approx(FRONT_Z, abs=1e-6)
    assert body["maximum"][2] == pytest.approx(INTERIOR_Z, abs=1e-6)
    assert body["width_m"] == pytest.approx(0.6, abs=0.03) and body["height_m"] == pytest.approx(0.6, abs=0.03)
    assert body["closed_frame_ids"] == [f"decoded-{CLOSED:09d}"] and body["open_frame_ids"] == [f"decoded-{OPEN:09d}"]
    # Label text is kept verbatim, per frame, only where some was read.
    readings = record["label_readings"]
    assert readings and all(row["label_text"] == ["BOSCH", "800 Series"] and int(row["frame_id"][8:]) < 40
                            and Path(row["path"]).is_file() for row in readings)
    assert "label_text" in calls[0]["prompt"] and record["classifier"]["revision"] == 3
    assert "part_boxes" in calls[0]["prompt"] and "box_xywh_normalized" in calls[0]["prompt"]
    # Retained receipts: a restart buys nothing and reproduces the record.
    assert _run(tmp_path, target, geometry, registry=_registry(600)) == record
    assert len(calls) == math.ceil(len(ids) / coverage.CLASSIFY_BATCH)


def _unread(binding):
    """The 2026-09-27 incident: the batch classifier reads no label text anywhere."""
    return {**_answer(binding), "frames": [{**row, "label_text": []} for row in _answer(binding)["frames"]]}


def test_focused_brand_read_recovers_a_wordmark_the_batch_classifier_missed(tmp_path, model):
    from blueprint_pipeline.website_object_spec_research import identify_object
    calls, state = model
    state["answer"] = _unread
    state["brand"] = lambda binding: {"frames": [
        {"frame_id": row["frame_id"], "label_text": ["Whirlpool", "FAN DRYING TECHNOLOGY"]}
        for row in binding["images"]]}
    geometry, track = _source_geometry(tmp_path / "geometry")
    target = _target(track)
    record = _run(tmp_path, target, geometry)
    assert record["status"] == "complete", record["blockers"]
    # One bounded request on closed-state views of the face: every front view (the control strip
    # shows) first, then the closed oblique; never an open or interior view.
    (brand,) = state["brand_calls"]
    ids = [row["frame_id"] for row in brand["images"]]
    assert len(ids) == coverage.BRAND_READ_FRAMES and ids[:3] == [f"decoded-{i:09d}" for i in (0, 15, 30)]
    assert 80 <= int(ids[3][8:]) < 100  # closed left_oblique
    assert brand["model"] == coverage.CLASSIFIER_MODEL and brand["revision"] == coverage.BRAND_READ_REVISION
    assert "serial number" in brand["prompt"] and "stylized" in brand["prompt"]
    for row in brand["images"]:
        # Full resolution crop of the subject box (cols 20-40, rows 12-30) with its margin.
        left, top, right, bottom = row["crop_box_xyxy"]
        assert left <= 20 and top <= 12 and right >= 40 and bottom >= 30 and right - left < WIDTH
    readings = {row["frame_id"]: row for row in record["label_readings"]}
    assert set(readings) == set(ids)
    assert all(row["label_text"] == ["Whirlpool", "FAN DRYING TECHNOLOGY"]
               and row["label_text_by_pass"] == {"focused_brand_read": ["Whirlpool", "FAN DRYING TECHNOLOGY"]}
               and Path(row["path"]).is_file() for row in readings.values())
    assert record["brand_read"]["status"] == "read" and record["brand_read"]["receipt"]["binding_digest"]
    identity = identify_object(task_context=_context(), coverage=record, category="dishwasher")
    assert identity["basis"] == "label_read" and identity["specificity"] == "brand_only"
    assert {row["text"] for row in identity["label_reads"]} == {"Whirlpool", "FAN DRYING TECHNOLOGY"}
    # Retained: a restart buys neither pass again.
    before = len(calls)
    assert _run(tmp_path, target, geometry) == record
    assert len(state["brand_calls"]) == 1 and len(calls) == before


def test_both_passes_merge_verbatim_with_their_provenance(tmp_path, model):
    _, state = model
    state["brand"] = lambda binding: {"frames": [{"frame_id": row["frame_id"], "label_text": ["Bosch", "BOSCH"]}
                                                 for row in binding["images"]]}
    geometry, track = _source_geometry(tmp_path / "geometry")
    record = _run(tmp_path, _target(track), geometry)
    focused = {row["frame_id"] for row in state["brand_calls"][0]["images"]}
    rows = {row["frame_id"]: row for row in record["label_readings"]}
    for frame_id, row in rows.items():
        read = int(frame_id[8:]) < 40  # _answer reads the classifier text on front frames only.
        expected = {**({"coverage_classification": ["BOSCH", "800 Series"]} if read else {}),
                    **({"focused_brand_read": ["Bosch", "BOSCH"]} if frame_id in focused else {})}
        assert row["label_text_by_pass"] == expected
        # Verbatim union, classifier first: "BOSCH" is not repeated, "Bosch" is kept as printed.
        assert row["label_text"] == list(dict.fromkeys(sum(expected.values(), [])))
    assert focused <= set(rows) and rows["decoded-000000000"]["label_text"] == ["BOSCH", "800 Series", "Bosch"]


def test_malformed_brand_read_never_holds_coverage_and_is_rebought_only_on_its_revision(tmp_path, model,
                                                                                      monkeypatch):
    calls, state = model
    state["brand"] = lambda binding: {"frames": [{"frame_id": row["frame_id"], "label_text": ["SN 4F2A1-0093"],
                                                  "serial": True} for row in binding["images"]]}
    geometry, track = _source_geometry(tmp_path / "geometry")
    target = _target(track)
    record = _run(tmp_path, target, geometry)
    assert record["status"] == "complete" and record["brand_read"]["status"] == "invalid"
    assert all(set(row["label_text_by_pass"]) == {"coverage_classification"} for row in record["label_readings"])
    assert _run(tmp_path, target, geometry) == record and len(state["brand_calls"]) == 1
    # A revision bump re-buys the brand read alone; classifier receipts replay for free.
    before = len(calls)
    state["brand"] = _no_brand
    monkeypatch.setattr(coverage, "BRAND_READ_REVISION", coverage.BRAND_READ_REVISION + 1)
    assert not coverage.coverage_matches(record, target=target, source_geometry=geometry)
    bumped = _run(tmp_path, target, geometry)
    assert bumped["brand_read"]["status"] == "read" and len(state["brand_calls"]) == 2 and len(calls) == before


def _boxed(binding):
    """The same answer, each listed part boxed inside the subject box (cols 20-40, rows 12-30), one twice."""
    answer = _answer(binding)
    for row in answer["frames"]:
        row["part_boxes"] = [{"part": part, "box_xywh_normalized": [0.35, 0.32 + 0.01 * index, 0.25, 0.1]}
                             for index, part in enumerate(row["visible_parts"])]
        row["part_boxes"].append({"part": row["visible_parts"][0], "box_xywh_normalized": [0.36, 0.4, 0.2, 0.1]})
    return answer


def test_classifier_revision_bump_buys_part_boxes_once_and_replays_the_brand_read(tmp_path, model, monkeypatch):
    calls, state = model
    geometry, track = _source_geometry(tmp_path / "geometry")
    target = _target(track)
    # Receipts retained before part boxes were asked for.
    monkeypatch.setattr(coverage, "CLASSIFIER_REVISION", 2)
    before = _run(tmp_path, target, geometry)
    assert before["status"] == "complete" and len(state["brand_calls"]) == 1
    assert before["part_localization"] and all(
        row["part_boxes"] == {} and row["part_box_rejections"] == [{"part": None, "reason": "part_boxes_missing"}]
        for row in before["part_localization"])
    bought = len(calls)
    monkeypatch.setattr(coverage, "CLASSIFIER_REVISION", 3)
    state["answer"] = _boxed
    assert not coverage.coverage_matches(before, target=target, source_geometry=geometry)
    after = _run(tmp_path, target, geometry)
    # Every classifier batch is bought once more; the focused brand read replays for free.
    assert len(calls) == 2 * bought and len(state["brand_calls"]) == 1
    assert after["brand_read"] == before["brand_read"] and after["label_readings"] == before["label_readings"]
    assert all(binding["revision"] == 3 and "part_boxes" in binding["prompt"] for binding in calls[bought:])
    assert {row["frame_id"] for row in after["part_localization"]} == {
        image["frame_id"] for binding in calls[bought:] for image in binding["images"]}
    for row in after["part_localization"]:
        parts = sorted(_label(int(row["frame_id"][8:]))[2])
        # The part given two boxes keeps neither; every other listed part keeps its own.
        assert sorted(row["part_boxes"]) == parts[1:]
        assert row["part_box_rejections"] == [{"part": parts[0], "reason": "part_boxed_twice"}]
    # One geometry frame shows the interior: too thin to measure, so each fixed part stays a
    # prior with its reason, and never holds the record.
    assert after["status"] == "complete" and after["part_extents"]["estimates"] == []
    assert {row["reason"] for row in after["part_extents"]["diagnostics"]} <= {
        "part_boxed_in_fewer_than_two_depth_frames", "part_never_boxed"}
    fixed = coverage.fixed_interior_parts({"frames": [{"visible_parts": parts} for parts in (FRONT_PARTS, OPEN_PARTS)],
                                           "task_part_components": sorted(MOVING)}, articulation_kind="revolute")
    assert fixed and {row["part_id"] for row in after["part_extents"]["diagnostics"]} == set(fixed)
    assert _run(tmp_path, target, geometry) == after and len(calls) == 2 * bought


def test_brand_read_uses_only_closed_front_and_oblique_views():
    frames = [{"frame_id": f"f{i}", "timestamp_seconds": float(i), "mask_area_fraction": 0.1 + 0.01 * i,
               "visible_parts": parts, "part_state": state, "view": view}
              for i, (state, view, parts) in enumerate([
                  ("open", "front", ["control_panel"]), ("closed", "interior", ["tub_interior"]),
                  ("closed", "left_oblique", ["left_side"]), ("closed", "front", ["body_front"]),
                  ("not_visible", "right_oblique", ["brand_label"]), ("partially_open", "front", ["door_outer"])])]
    boxes = {row["frame_id"]: [0.1, 0.1, 0.5, 0.5] for row in frames}
    chosen = coverage.brand_read_frames(frames, boxes=boxes)
    assert [row["frame_id"] for row in chosen] == ["f3", "f4", "f2"]
    assert chosen[0]["subject_box_xywh_normalized"] == [0.1, 0.1, 0.5, 0.5]
    assert coverage.brand_read_frames(frames[:2], boxes=boxes) == []


def test_selection_that_cannot_show_every_part_is_incomplete():
    frames = [{"frame_id": f"f{i}", "timestamp_seconds": float(i), "mask_area_fraction": 0.1,
               "visible_parts": [f"part_{i}"], "part_state": "closed", "view": "front", "path": "p", "sha256": "s"}
              for i in range(5)]
    selected = coverage.select_reference_frames(frames, cap=3)
    assert len(selected) == 3 and all(row["reason"].startswith("adds parts: ") for row in selected)
    record = coverage.coverage_record(target_id="t", binding={}, articulation_kind="prismatic",
        classified={"frames": frames, "task_part_components": ["part_0"], "hinge_edges": [], "receipts": []},
        selected=selected, body_bounds={"depth_m": 0.5}, body_blockers=[], candidate_count=5)
    assert record["status"] == "incomplete"
    assert record["missing_parts"] == ["part_3", "part_4"]
    assert record["blockers"] == ["website_assembly_reference_parts_uncovered"]
    assert coverage.coverage_blockers(record, target_id="t", several=True) == [
        "website_assembly_reference_parts_uncovered:t"]


def test_depth_view_is_chosen_first_and_the_contract_cites_only_shown_frames():
    frames = [{"frame_id": f"f{i}", "timestamp_seconds": float(i), "mask_area_fraction": 0.1 + 0.01 * i,
               "visible_parts": ["body_front", "door"] if i < 3 else ["tub"], "view": "front",
               "part_state": "closed" if i < 3 else "open", "path": "p", "sha256": "s"} for i in range(5)]
    body = {"open_frame_ids": ["f3", "f4"], "depth_m": 0.5, "width_m": 0.6, "height_m": 0.8,
            "depth_basis": "interior_observed_open_state", "basis": "b"}
    assert coverage.depth_seed(frames, body) == ["f4"]
    selected = coverage.select_reference_frames(frames, cap=2, seed_frame_ids=["f4"])
    assert [(row["frame_id"], row["selection_rank"]) for row in selected] == [("f2", 1), ("f4", 0)]
    record = {"status": "complete", "body_bounds": body, "hinge_edge": "bottom", "task_part_components": ["door"],
              "observed_parts": ["body_front", "door", "tub"], "selected_frames": selected}
    contract = coverage.assembly_contract(record, articulation_kind="revolute", source_to_simulator_scale=2.0)
    assert contract["body_depth"] == {"value_m": 1.0, "basis": "interior_observed_open_state", "frame_ids": ["f4"]}
    assert (contract["body_extent_m"]["depth"], contract["body_extent_m"]["width"]) == (1.0, 1.2)
    with pytest.raises(ValueError, match="website_assembly_depth_frame_not_referenced"):
        coverage.assembly_contract({**record, "selected_frames": selected[:1]}, articulation_kind="revolute",
                                   source_to_simulator_scale=2.0)


def test_drawer_cabinet_parts_plan_each_drawer_as_its_bay_and_carcass_words_as_panels():
    from blueprint_pipeline.task_object_articulated_packaging import plan_articulated_assembly
    from tests.test_task_object_articulated_packaging import configuration

    parts = ["bottom_drawer", "cabinet_top", "left_side", "middle_drawer", "top_drawer"]
    frames = [{"frame_id": "f0", "timestamp_seconds": 0.0, "visible_parts": parts, "part_state": "closed",
               "view": "front", "path": "p0", "sha256": "s0", "reason": "front", "selection_rank": 1},
              {"frame_id": "f1", "timestamp_seconds": 1.0, "visible_parts": ["middle_drawer"], "part_state": "open",
               "view": "front", "path": "p1", "sha256": "s1", "reason": "depth", "selection_rank": 0}]
    body = {"open_frame_ids": ["f1"], "depth_m": 0.55, "width_m": 0.42, "height_m": 0.62,
            "depth_basis": "interior_observed_open_state", "basis": "b"}
    record = {"status": "complete", "body_bounds": body, "hinge_edge": None, "task_part_components": ["middle_drawer"],
              "observed_parts": parts, "selected_frames": frames}
    contract = coverage.assembly_contract(record, articulation_kind="prismatic", source_to_simulator_scale=1.0)
    assert {row["part_id"]: row["role"] for row in contract["required_parts"]} == {
        "top_drawer": "fixed_interior", "middle_drawer": "task_part", "bottom_drawer": "fixed_interior",
        "cabinet_top": "body_feature", "left_side": "body_feature"}
    value = configuration()  # three-drawer cabinet, task part "middle drawer"
    value.update(contract, source_observation_kind="website_capture_frames",
                 reference_frames=[{**row, "sha256": "sha256:" + str(index) * 64}
                                   for index, row in enumerate(contract["reference_frames"])])
    plan = plan_articulated_assembly(value)
    assert {row["part_id"]: (row["link_id"], row["feature"]) for row in plan["required_parts"]} == {
        "top_drawer": ("drawer_0", "link"), "middle_drawer": ("drawer_1", "link"),
        "bottom_drawer": ("drawer_2", "link"), "cabinet_top": ("carcass", "top_panel"),
        "left_side": ("carcass", "left_side_panel")}


def test_selection_fills_remaining_slots_with_distinct_views():
    frames = [{"frame_id": f"f{i}", "timestamp_seconds": float(i), "mask_area_fraction": 0.1,
               "visible_parts": ["body_front"], "part_state": "closed", "view": view}
              for i, view in enumerate(["front", "front", "front", "left_oblique"])]
    selected = coverage.select_reference_frames(frames, cap=3)
    assert [row["view"] for row in selected].count("left_oblique") == 1
    assert sum(row["reason"].startswith("adds diversity") for row in selected) == 1


@pytest.mark.parametrize("closed,opened,blocker", [
    (True, False, "website_assembly_body_depth_unobserved"),
    (False, True, "website_assembly_closed_front_unobserved"),
])
def test_body_size_is_never_invented(tmp_path, closed, opened, blocker):
    geometry, track = _source_geometry(tmp_path, closed=closed, opened=opened)
    states = {f"decoded-{CLOSED:09d}": "closed", f"decoded-{OPEN:09d}": "open"}
    body, blockers = coverage.estimate_body_bounds(track=track, frames=geometry["frames"], states=states)
    assert body is None and blockers == [blocker]


def test_open_door_sweep_alone_is_not_body_depth(tmp_path):
    geometry, track = _source_geometry(tmp_path)
    # Only the door, folded down in front of the face, is seen while open.
    frame = geometry["frames"][1]
    np.savez(frame["geometry_path"], depth_m=np.where(_box((28, 38), (15, 46)), DOOR_Z, 4.0),
             valid_mask=np.ones((HEIGHT, WIDTH), dtype=bool))
    frame["geometry_digest"] = _sha256_file(Path(frame["geometry_path"]))
    track["observations"][1]["runs"] = _runs(_box((28, 38), (15, 46)))
    body, blockers = coverage.estimate_body_bounds(track=track, frames=geometry["frames"],
        states={f"decoded-{CLOSED:09d}": "closed", f"decoded-{OPEN:09d}": "open"})
    assert body is None and blockers == ["website_assembly_body_depth_unobserved"]


@pytest.mark.parametrize("change", [
    lambda a: {**a, "hinge_edge": "diagonal"},
    lambda a: {**a, "frames": a["frames"][1:]},
    lambda a: {**a, "frames": [{**a["frames"][0], "task_part_state": "ajar"}, *a["frames"][1:]]},
    lambda a: {**a, "frames": [{**a["frames"][0], "visible_parts": ["Door Outer"]}, *a["frames"][1:]]},
    lambda a: {**a, "task_part_components": ["never_seen"]},
    lambda a: {**a, "extra": True},
    lambda a: {**a, "frames": [{k: v for k, v in a["frames"][0].items() if k != "label_text"}, *a["frames"][1:]]},
    lambda a: {**a, "frames": [{**a["frames"][0], "label_text": "BOSCH"}, *a["frames"][1:]]},
    lambda a: {**a, "frames": [{**a["frames"][0], "label_text": [" BOSCH"]}, *a["frames"][1:]]},
    lambda a: {**a, "frames": [{**a["frames"][0], "label_text": ["BOSCH", "BOSCH"]}, *a["frames"][1:]]},
    lambda a: {**a, "frames": [{**a["frames"][0], "label_text": [""]}, *a["frames"][1:]]},
    lambda a: {**a, "frames": [{**a["frames"][0], "label_text": ["BO\nSCH"]}, *a["frames"][1:]]},
    lambda a: {**a, "frames": [{**a["frames"][0], "label_text": ["x" * 121]}, *a["frames"][1:]]},
    lambda a: {**a, "frames": [{**a["frames"][0], "label_text": [str(i) for i in range(9)]}, *a["frames"][1:]]},
])
def test_malformed_classification_is_a_typed_incomplete_record_never_rebought(tmp_path, model, change):
    calls, state = model
    state["answer"] = lambda binding: change(_answer(binding))
    geometry, track = _source_geometry(tmp_path / "geometry")
    target = _target(track)
    record = _run(tmp_path, target, geometry)
    assert record["status"] == "incomplete" and record["body_bounds"] is None
    assert record["blockers"] == ["website_assembly_coverage_classification_invalid"]
    assert len(record["classifier"]["receipts"]) == 1 and record["candidate_frame_count"] > 0
    assert _run(tmp_path, target, geometry) == record
    assert len(calls) == 1


def test_malformed_classification_is_held_every_tick_until_a_revision_bump(tmp_path, model, monkeypatch):
    calls, state = model
    state["answer"] = lambda binding: {**_answer(binding), "extra": True}
    geometry, track = _source_geometry(tmp_path / "geometry")
    masks = {"schema_version": "website_task_masks.v1", "targets": [_target(track)],
             "source_frame_registry": _registry()}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    arguments = dict(task_masks=masks, source_geometry=geometry, removal_manifest={"entries": []},
                     task_context=_context(), source_video=tmp_path / "clip.mov", output_root=tmp_path / "coverage")
    value = coverage.attach_assembly_coverage(**arguments)
    record = value["targets"][0]["authoring_coverage"]
    assert coverage.coverage_blockers(record, target_id="dishwasher-1", several=False) == [
        "website_assembly_coverage_classification_invalid"]
    assert coverage.attach_assembly_coverage(**{**arguments, "task_masks": value}) == value and len(calls) == 1
    state["answer"] = _answer
    monkeypatch.setattr(coverage, "CLASSIFIER_REVISION", coverage.CLASSIFIER_REVISION + 1)
    bumped = coverage.attach_assembly_coverage(**{**arguments, "task_masks": value})
    assert bumped["targets"][0]["authoring_coverage"]["status"] == "complete" and len(calls) > 1


def test_inconsistent_hinge_is_a_blocker(tmp_path, model):
    _, state = model
    state["answer"] = lambda binding: _answer(binding, hinge="bottom" if binding["images"][0]["frame_id"] < "decoded-000000300" else "left")
    geometry, track = _source_geometry(tmp_path / "geometry")
    record = _run(tmp_path, _target(track, frames=600), geometry, registry=_registry(600))
    assert len(record["classifier"]["receipts"]) > 1
    assert record["status"] == "incomplete"
    assert record["blockers"] == ["website_assembly_hinge_edge_inconsistent"]


def test_each_articulated_target_gets_its_own_record_and_uncaptured_ones_are_not_complete(tmp_path, model):
    calls, _ = model
    geometry, track = _source_geometry(tmp_path / "geometry")
    dishwasher = _target(track)
    oven = {**_target(track, target_id="oven-1"), "semantic_label": "oven", "articulated_part": "oven door"}
    unseen = {**_target(track, target_id="cabinet-1", frames=0), "semantic_label": "cabinet",
              "articulated_part": "cabinet door"}
    cup = {**_target(track, target_id="cup-1"), "articulation_kind": ""}
    masks = {"schema_version": "website_task_masks.v1", "targets": [dishwasher, oven, unseen, cup],
             "source_frame_registry": _registry()}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    arguments = dict(task_masks=masks, source_geometry=geometry, removal_manifest={"entries": []},
                     task_context=_context(), source_video=tmp_path / "clip.mov", output_root=tmp_path / "coverage")
    value = coverage.attach_assembly_coverage(**arguments)
    assert value["digest"] == canonical_digest(value, digest_field="digest") != masks["digest"]
    records = {row["target_id"]: row.get("authoring_coverage") for row in value["targets"]}
    assert records["cup-1"] is None
    assert records["dishwasher-1"]["status"] == records["oven-1"]["status"] == "complete"
    assert records["dishwasher-1"]["target_id"] == "dishwasher-1" and records["oven-1"]["target_id"] == "oven-1"
    assert {binding["target_id"] for binding in calls} == {"dishwasher-1", "oven-1"}
    assert (tmp_path / "coverage" / "dishwasher-1" / "receipts").is_dir()
    assert (tmp_path / "coverage" / "oven-1" / "receipts").is_dir()
    assert records["cabinet-1"]["status"] == "not_captured"
    assert records["cabinet-1"]["observed_parts"] == records["cabinet-1"]["selected_frames"] == []
    assert records["cabinet-1"]["body_bounds"] is None
    assert records["cabinet-1"]["blockers"] == ["website_assembly_not_captured"]
    # Matching records are kept; nothing is recomputed or bought again.
    before = len(calls)
    assert coverage.attach_assembly_coverage(**{**arguments, "task_masks": value}) == value
    assert len(calls) == before


def test_uncaptured_target_is_held_by_the_preparation_gate(tmp_path):
    from tests.test_website_task_preparation import _assembly_inputs, _compile
    inputs = _assembly_inputs(tmp_path, covered=False)
    masks = inputs["task_masks"]
    target = masks["targets"][0]
    record = coverage.empty_record(target_id=target["target_id"],
        binding=coverage.coverage_binding(target=target, source_geometry=inputs["source_geometry"]),
        articulation_kind="prismatic", status="not_captured", blocker="website_assembly_not_captured")
    masks["targets"][0] = {**target, "authoring_coverage": record}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    value = _compile(tmp_path, inputs)
    assert value["status"] == "needs_input"
    assert value["blockers"][-2:] == ["website_assembly_whole_object_coverage_required", "website_assembly_not_captured"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decoder_returns_exact_indices_rotated_upright(tmp_path):
    colors = [(40 * i, 255 - 40 * i, 7 * i) for i in range(6)]
    for i, color in enumerate(colors):
        image = Image.new("RGB", (32, 16), color)
        image.putpixel((0, 0), (255, 255, 255))
        image.save(tmp_path / f"in-{i:02d}.png")
    video = tmp_path / "clip.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "10", "-i", str(tmp_path / "in-%02d.png"),
                    "-c:v", "png", str(video)], check=True, capture_output=True, timeout=60)
    frames = [{"frame_id": f"decoded-{i:09d}", "decoded_index": i} for i in (4, 1)]
    result = coverage.decode_upright_frames(video=video, video_digest=_sha256_file(video), rotation=-90,
                                            frames=frames, output_root=tmp_path / "out")
    for i in (1, 4):
        row = result[f"decoded-{i:09d}"]
        with Image.open(row["path"]) as image:
            assert image.size == (16, 32) == (row["width"], row["height"])
            assert image.getpixel((8, 16)) == colors[i]
            assert image.getpixel((15, 0)) == (255, 255, 255)  # Stored top-left, turned clockwise.
    assert json.loads((next((tmp_path / "out").glob("*/upright_frames.json"))).read_text())["frames"] == result
    assert coverage.decode_upright_frames(video=video, video_digest=_sha256_file(video), rotation=-90,
                                          frames=frames, output_root=tmp_path / "out") == result
