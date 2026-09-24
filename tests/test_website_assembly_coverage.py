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
    frames = [dict(zip(("task_part_state", "view", "visible_parts"), _label(int(i[8:]))), frame_id=i) for i in ids]
    visible = {part for row in frames for part in row["visible_parts"]}
    return {"hinge_edge": hinge, "task_part_components": sorted(MOVING & visible), "frames": frames}


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
    state = {"answer": _answer}

    def retained(**kwargs):
        def invoke():
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
    # Retained receipts: a restart buys nothing and reproduces the record.
    assert _run(tmp_path, target, geometry, registry=_registry(600)) == record
    assert len(calls) == math.ceil(len(ids) / coverage.CLASSIFY_BATCH)


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
])
def test_malformed_classification_fails_closed_without_rebuying(tmp_path, model, change):
    calls, state = model
    state["answer"] = lambda binding: change(_answer(binding))
    geometry, track = _source_geometry(tmp_path / "geometry")
    target = _target(track)
    with pytest.raises(ValueError, match="website_assembly_coverage_classification_invalid"):
        _run(tmp_path, target, geometry)
    with pytest.raises(ValueError, match="website_assembly_coverage_classification_invalid"):
        _run(tmp_path, target, geometry)
    assert len(calls) == 1


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
