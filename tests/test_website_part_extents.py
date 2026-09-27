"""ADP-009/ADP-030: each fixed assembly part boxed per frame and sized from depth inside its boxes.

2026-09-27 website capture: coverage named the parts each frame shows but
located none inside the whole-object mask, so every fixed interior part was a
template prior. Family- and name-agnostic: a side-hinged storage cabinet with
two fixed shelves is ray-cast into synthetic depth frames; only the classifier
answer is given. No provider is called.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import website_assembly_coverage as coverage
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.task_object_articulated_packaging import OBSERVED_ESTIMATE_BASIS, plan_articulated_assembly
from blueprint_pipeline.website_task_preparation import published_body_bounds

W, H, F = 160, 120, 120.0  # The field of view of the 40 x 30 preparation fixture, four times finer.
# World: OpenCV-like, +Y down, cameras look along +Z. The closed door's outer
# face is the observed front; the shell's inner back wall is 0.42 m behind it.
DOOR_FACE_Z, SHELL_FRONT_Z, INNER_BACK_Z, OUTER_BACK_Z = 1.08, 1.1, 1.5, 1.52
HALF_WIDTH, TOP_Y, BOTTOM_Y, WALL = 0.25, -0.06, 0.5, 0.02
BODY_HEIGHT = BOTTOM_Y - TOP_Y
# Fixed shelves (above the bottom, in metres): top face and underside heights.
SHELVES = {"upper_shelf": (0.36, 0.38), "lower_shelf": (0.18, 0.20)}
SHELF_Z = (1.12, 1.46)  # 0.04 m behind the door face to 0.38 m.
SHELF_HALF_WIDTH = HALF_WIDTH - WALL


def _shell():
    return {
        "left_wall": ([-HALF_WIDTH, TOP_Y, SHELL_FRONT_Z], [-HALF_WIDTH + WALL, BOTTOM_Y, OUTER_BACK_Z]),
        "right_wall": ([HALF_WIDTH - WALL, TOP_Y, SHELL_FRONT_Z], [HALF_WIDTH, BOTTOM_Y, OUTER_BACK_Z]),
        "top_panel": ([-HALF_WIDTH, TOP_Y, SHELL_FRONT_Z], [HALF_WIDTH, TOP_Y + WALL, OUTER_BACK_Z]),
        "bottom_panel": ([-HALF_WIDTH, BOTTOM_Y - WALL, SHELL_FRONT_Z], [HALF_WIDTH, BOTTOM_Y, OUTER_BACK_Z]),
        "back_wall": ([-HALF_WIDTH, TOP_Y, INNER_BACK_Z], [HALF_WIDTH, BOTTOM_Y, OUTER_BACK_Z]),
        **{name: ([-SHELF_HALF_WIDTH, BOTTOM_Y - high, SHELF_Z[0]], [SHELF_HALF_WIDTH, BOTTOM_Y - low, SHELF_Z[1]])
           for name, (low, high) in SHELVES.items()},
    }


CLOSED_SCENE = {**_shell(), "door": ([-HALF_WIDTH, TOP_Y, DOOR_FACE_Z], [HALF_WIDTH, BOTTOM_Y, SHELL_FRONT_Z])}
OPEN_SCENE = _shell()  # The door is swung fully aside, out of view.


def look_at(position, target=(0.0, 0.2, 1.3)):
    z = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    pose[:3, 3] = position
    return pose


def render(scene, pose, background=None):
    """Depth along the camera axis and the name of the nearest box, per pixel (pixel-corner rays)."""
    ys, xs = np.indices((H, W))
    rays = np.stack([(xs - W / 2) / F, (ys - H / 2) / F, np.ones((H, W))], axis=-1).reshape(-1, 3)
    directions, origin = rays @ pose[:3, :3].T, pose[:3, 3]
    best = np.full(len(rays), np.inf)
    label = np.full(len(rays), "", dtype=object)
    for name, (low, high) in scene.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (np.asarray(low) - origin) / directions
            b = (np.asarray(high) - origin) / directions
        near = np.nanmax(np.minimum(a, b), axis=1)
        far = np.nanmin(np.maximum(a, b), axis=1)
        hit = (far >= near) & (near > 0) & (near < best)
        best[hit], label[hit] = near[hit], name
    best, label = best.reshape(H, W), label.reshape(H, W)
    subject = label != ""
    depth = np.where(subject, best, background if background is not None else 0.0)
    return depth, label, subject


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.pad(mask.reshape(-1).astype(np.int8), (1, 1))))
    return [{"start": int(a), "length": int(b - a)} for a, b in zip(edges[::2], edges[1::2])]


def _normalized_box(mask, grow=0.0):
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    dx, dy = grow * (x1 - x0), grow * (y1 - y0)
    x0, x1, y0, y1 = max(0, x0 - dx), min(W, x1 + dx), max(0, y0 - dy), min(H, y1 + dy)
    return [round(x0 / W, 6), round(y0 / H, 6), round((x1 - x0) / W, 6), round((y1 - y0) / H, 6)]


def capture(root: Path, views, *, background=None):
    """Geometry frames, the subject track, per-frame labels and each shelf's true image region."""
    root.mkdir(parents=True, exist_ok=True)
    frames, observations, regions = [], [], {}
    for frame_id, pose, state in views:
        depth, label, subject = render(CLOSED_SCENE if state == "closed" else OPEN_SCENE, pose, background)
        valid = subject if background is None else np.ones_like(subject)
        path = root / f"{frame_id}.npz"
        np.savez(path, depth_m=depth, valid_mask=valid)
        image = root / f"{frame_id}.png"
        Image.fromarray(np.full((H, W, 3), 110, dtype=np.uint8)).save(image)
        frames.append({"frame_id": frame_id, "timestamp_seconds": len(frames) * 0.5, "geometry_path": str(path),
                       "geometry_digest": _sha256_file(path), "image_path": str(image),
                       "image_digest": _sha256_file(image), "intrinsics": [[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]],
                       "world_from_camera": pose.tolist(), "width": W, "height": H, "display_rotation_degrees": 0})
        observations.append({"source_frame_id": frame_id, "width": W, "height": H, "runs": _runs(subject)})
        regions[frame_id] = {"subject": subject, **{name: label == name for name in SHELVES}}
    return frames, {"track_id": "t1", "label": "cabinet-1", "observations": observations}, regions


OPEN_VIEWS = [("frame-1", look_at([0.0, -0.3, 0.1]), "open"), ("frame-2", look_at([-0.3, -0.3, 0.15]), "open"),
              ("frame-3", look_at([0.3, -0.3, 0.15]), "open")]
VIEWS = [("frame-0", look_at([0.0, -0.3, 0.1]), "closed"), *OPEN_VIEWS]


def classified_rows(regions, boxed, *, grow=0.0, states=None):
    """What the classifier would say: which shelves each frame boxes (``boxed``: frame id -> part -> region)."""
    rows = []
    for frame_id, region in regions.items():
        state = (states or {}).get(frame_id, "closed" if frame_id == "frame-0" else "open")
        parts = sorted(boxed.get(frame_id, {}))
        answer = {"frame_id": frame_id, "visible_parts": ["body_front", "door_outer", "handle"] if state == "closed"
                  else ["cabinet_interior", *parts], "task_part_state": state,
                  "view": "front" if frame_id in {"frame-0", "frame-1"} else "interior", "label_text": [],
                  "part_boxes": [{"part": part, "box_xywh_normalized": _normalized_box(region[source], grow)}
                                 for part, source in sorted(boxed.get(frame_id, {}).items())]}
        rows.append(answer)
    subjects = {frame_id: _normalized_box(region["subject"]) for frame_id, region in regions.items()}
    value = coverage.validate_classification(
        {"hinge_edge": "left", "task_part_components": ["door_outer", "handle"], "frames": rows},
        frame_ids=list(regions), articulation_kind="revolute", subject_boxes=subjects)
    return [{**row, "timestamp_seconds": index * 0.5, "mask_area_fraction": 0.2, "geometry_frame": True,
             "path": "p", "sha256": "s"} for index, row in enumerate(value["frames"])]


def _body(frames, track, rows):
    body, blockers = coverage.estimate_body_bounds(track=track, frames=frames,
                                                   states={row["frame_id"]: row["part_state"] for row in rows})
    assert blockers == [] and body is not None
    return body


ALL_SHELVES = {frame_id: {name: name for name in SHELVES} for frame_id, _, _ in OPEN_VIEWS}


@pytest.fixture(scope="module")
def cabinet(tmp_path_factory):
    frames, track, regions = capture(tmp_path_factory.mktemp("cabinet"), VIEWS)
    return frames, track, regions


def _extents(cabinet, boxed=ALL_SHELVES, *, grow=0.0, parts=tuple(SHELVES)):
    frames, track, regions = cabinet
    rows = classified_rows(regions, boxed, grow=grow)
    body = _body(frames, track, rows)
    return body, rows, coverage.estimate_part_extents(track=track, frames=frames, classified=rows, body=body,
                                                      parts=parts)


# ---- per-part boxes in the classifier answer ----------------------------------------------------------------

SUBJECT = [0.2, 0.1, 0.6, 0.8]


@pytest.mark.parametrize("entry,reason", [
    ({"part": "shelf", "box_xywh_normalized": [0.3, 0.3, 0.2, 0.1]}, None),
    ({"part": "drawer", "box_xywh_normalized": [0.3, 0.3, 0.2, 0.1]}, "part_not_listed_visible"),
    ({"part": "shelf", "box_xywh_normalized": "0.3,0.3,0.2,0.1"}, "box_malformed"),
    ({"part": "shelf", "box_xywh_normalized": [0.3, 0.3, 0.2]}, "box_malformed"),
    ({"part": "shelf", "box_xywh_normalized": [0.3, float("nan"), 0.2, 0.1]}, "box_malformed"),
    ({"part": "shelf", "box_xywh_normalized": [0.3, True, 0.2, 0.1]}, "box_malformed"),
    ({"part": "shelf", "box_xywh_normalized": [0.3, 0.3, -0.2, 0.1]}, "box_outside_frame_or_empty"),
    ({"part": "shelf", "box_xywh_normalized": [0.3, 0.3, 0.0, 0.1]}, "box_outside_frame_or_empty"),
    ({"part": "shelf", "box_xywh_normalized": [0.9, 0.3, 0.2, 0.1]}, "box_outside_frame_or_empty"),
    ({"part": "shelf", "box_xywh_normalized": [-0.1, 0.3, 0.2, 0.1]}, "box_outside_frame_or_empty"),
    ({"part": "shelf", "box_xywh_normalized": [0.3, 0.3, 0.005, 0.005]}, "box_area_too_small"),
    ({"part": "shelf", "box_xywh_normalized": [0.0, 0.0, 0.15, 0.1]}, "box_outside_subject"),
    ({"part": "shelf", "box_xywh_normalized": [0.3, 0.3, 0.2, 0.1], "score": 0.9}, "entry_malformed"),
    ({"box_xywh_normalized": [0.3, 0.3, 0.2, 0.1]}, "entry_malformed"),
    ("shelf", "entry_malformed"),
])
def test_each_part_box_is_validated_and_a_malformed_one_is_dropped_never_repaired(entry, reason):
    boxes, rejected = coverage.validate_part_boxes([entry], visible_parts=["door", "shelf"], subject_box=SUBJECT)
    if reason is None:
        assert boxes == {"shelf": [0.3, 0.3, 0.2, 0.1]} and rejected == []
    else:
        assert boxes == {} and [row["reason"] for row in rejected] == [reason]


def test_box_edges_within_tolerance_are_clipped_to_the_frame_and_a_twice_boxed_part_keeps_neither():
    edge = {"part": "door", "box_xywh_normalized": [0.5, 0.2, 0.5005, 0.8]}
    twice = [{"part": "shelf", "box_xywh_normalized": [0.3, 0.3, 0.2, 0.1]},
             {"part": "shelf", "box_xywh_normalized": [0.3, 0.5, 0.2, 0.1]}]
    boxes, rejected = coverage.validate_part_boxes([edge, *twice], visible_parts=["door", "shelf"],
                                                   subject_box=[0.2, 0.1, 0.8, 0.9])
    assert boxes == {"door": [0.5, 0.2, 0.5, 0.8]}
    assert rejected == [{"part": "shelf", "reason": "part_boxed_twice"}]
    assert coverage.validate_part_boxes([twice[0]], visible_parts=["shelf"], subject_box=None) == (
        {}, [{"part": "shelf", "reason": "subject_box_unavailable"}])
    assert coverage.validate_part_boxes(None, visible_parts=["shelf"], subject_box=SUBJECT) == (
        {}, [{"part": None, "reason": "part_boxes_missing"}])
    assert coverage.validate_part_boxes({"shelf": [0.3, 0.3, 0.2, 0.1]}, visible_parts=["shelf"],
                                        subject_box=SUBJECT)[1] == [{"part": None, "reason": "part_boxes_not_a_bounded_list"}]


def test_a_malformed_box_keeps_its_part_listed_and_never_refuses_the_batch():
    answer = {"hinge_edge": "left", "task_part_components": ["door"], "frames": [
        {"frame_id": "f0", "visible_parts": ["door", "shelf"], "task_part_state": "open", "view": "interior",
         "label_text": [], "part_boxes": [{"part": "shelf", "box_xywh_normalized": [0.3, 0.3, 0.2, 9.0]},
                                          {"part": "door", "box_xywh_normalized": [0.25, 0.2, 0.3, 0.6]}]},
        {"frame_id": "f1", "visible_parts": ["door"], "task_part_state": "closed", "view": "front",
         "label_text": []}]}
    value = coverage.validate_classification(answer, frame_ids=["f0", "f1"], articulation_kind="revolute",
                                             subject_boxes={"f0": SUBJECT, "f1": SUBJECT})
    first, second = value["frames"]
    assert first["visible_parts"] == ["door", "shelf"] and first["part_boxes"] == {"door": [0.25, 0.2, 0.3, 0.6]}
    assert first["part_box_rejections"] == [{"part": "shelf", "reason": "box_outside_frame_or_empty"}]
    assert second["part_boxes"] == {} and second["part_box_rejections"] == [{"part": None, "reason": "part_boxes_missing"}]
    # The rest of the answer stays strict.
    with pytest.raises(ValueError, match="classification_invalid"):
        coverage.validate_classification({**answer, "frames": [{**answer["frames"][0], "boxes": []}, answer["frames"][1]]},
                                         frame_ids=["f0", "f1"], articulation_kind="revolute")


# ---- depth-derived extents -----------------------------------------------------------------------------------

def _truth(name):
    low, high = SHELVES[name]
    return {"behind_front_m": [SHELF_Z[0] - DOOR_FACE_Z, SHELF_Z[1] - DOOR_FACE_Z],
            "across_from_centre_m": [-SHELF_HALF_WIDTH, SHELF_HALF_WIDTH], "above_bottom_m": [low, high]}


def _assert_near_truth(estimate, name):
    truth = _truth(name)
    assert estimate["above_bottom_m"] == pytest.approx(truth["above_bottom_m"], abs=0.015)
    # Wall rejection trims the sides by the enclosure margin; the front edge and the far
    # end are seen, less a trimmed percentile of points.
    assert estimate["across_from_centre_m"] == pytest.approx(truth["across_from_centre_m"], abs=0.035)
    assert estimate["behind_front_m"] == pytest.approx(truth["behind_front_m"], abs=0.035)


# Loose boxes take in both side walls, the back wall, the floor or top, and the edge of the other shelf.
@pytest.mark.parametrize("grow", [0.0, 0.3])
def test_two_shelves_are_measured_at_their_heights_and_the_enclosure_is_rejected(cabinet, grow):
    body, _, extents = _extents(cabinet, grow=grow)
    assert body["depth_m"] == pytest.approx(INNER_BACK_Z - DOOR_FACE_Z, abs=0.01)
    assert body["height_m"] == pytest.approx(BODY_HEIGHT, abs=0.02)
    assert extents["frame"] == coverage.PART_EXTENT_FRAME and extents["diagnostics"] == []
    assert extents["metric_measurement_proven"] is False
    rows = {row["part_id"]: row for row in extents["estimates"]}
    assert set(rows) == set(SHELVES)
    for name, row in rows.items():
        _assert_near_truth(row, name)
        assert row["frame_ids"] == ["frame-1", "frame-2", "frame-3"] and row["set_aside_frame_ids"] == []
        assert row["views"] == ["front", "interior"] and row["point_count"] >= 3 * coverage.MIN_PART_POINTS
        assert all(coverage.PART_EXTENT_UNCERTAINTY_FLOOR_M <= v <= coverage.PART_VIEW_AGREEMENT_M
                   for v in row["uncertainty_m"])
    # A loose box also takes in the side walls, the back wall and the floor: all rejected,
    # so it measures what a tight box does.
    if grow:
        _, _, tight = _extents(cabinet)
        for row, reference in zip(extents["estimates"], tight["estimates"]):
            for axis in ("behind_front_m", "across_from_centre_m", "above_bottom_m"):
                assert row[axis] == pytest.approx(reference[axis], abs=0.01)


def test_a_surface_only_one_view_boxes_is_not_the_part(cabinet):
    frames, track, regions = cabinet
    # frame-1's upper-shelf box reaches down over the lower shelf; the other views box the upper shelf alone.
    loose = {**ALL_SHELVES, "frame-1": {"upper_shelf": "upper_shelf", "lower_shelf": "lower_shelf"}}
    rows = classified_rows(regions, loose)
    both = regions["frame-1"]["upper_shelf"] | regions["frame-1"]["lower_shelf"]
    rows[1]["part_boxes"]["upper_shelf"] = _normalized_box(both)
    extents = coverage.estimate_part_extents(track=track, frames=frames, classified=rows,
                                             body=_body(frames, track, rows), parts=["upper_shelf"])
    (row,) = extents["estimates"]
    _assert_near_truth(row, "upper_shelf")


def test_one_depth_frame_is_too_thin_and_the_part_stays_a_prior_with_the_reason(cabinet):
    body, _, extents = _extents(cabinet, {"frame-1": {"upper_shelf": "upper_shelf"}},
                                parts=["upper_shelf", "lower_shelf"])
    assert extents["estimates"] == []
    assert extents["diagnostics"] == [
        {"part_id": "lower_shelf", "reason": "part_never_boxed", "frame_ids": [], "boxed_frame_ids": []},
        {"part_id": "upper_shelf", "reason": "part_boxed_in_fewer_than_two_depth_frames",
         "frame_ids": ["frame-1"], "boxed_frame_ids": ["frame-1"]}]


def test_views_that_box_different_surfaces_under_one_name_give_no_estimate(cabinet):
    # Two views call different shelves "upper_shelf": no surface is seen inside both boxes.
    swapped = {"frame-1": {"upper_shelf": "upper_shelf"}, "frame-2": {"upper_shelf": "lower_shelf"}}
    _, _, extents = _extents(cabinet, swapped, parts=["upper_shelf"])
    assert extents["estimates"] == []
    (diagnostic,) = extents["diagnostics"]
    assert diagnostic["part_id"] == "upper_shelf" and diagnostic["frame_ids"] == ["frame-1", "frame-2"]
    assert diagnostic["reason"] == "part_points_not_seen_by_another_view"


def test_a_frame_that_disagrees_is_set_aside_and_two_that_disagree_give_nothing(cabinet):
    frames, track, regions = cabinet
    rows = classified_rows(regions, ALL_SHELVES)
    body = _body(frames, track, rows)
    # Depth of frame-3 is shifted 0.15 m deeper, as a badly posed view would be.
    shifted = dict(frames[3])
    with np.load(shifted["geometry_path"]) as geometry:
        depth, valid = geometry["depth_m"] * 1.12, geometry["valid_mask"]
    path = Path(shifted["geometry_path"]).with_name("frame-3-shifted.npz")
    np.savez(path, depth_m=depth, valid_mask=valid)
    shifted.update(geometry_path=str(path), geometry_digest=_sha256_file(path))
    extents = coverage.estimate_part_extents(track=track, frames=[*frames[:3], shifted], classified=rows,
                                             body=body, parts=["upper_shelf"])
    (row,) = extents["estimates"]
    assert "frame-3" not in row["frame_ids"] and len(row["frame_ids"]) == 2
    _assert_near_truth(row, "upper_shelf")
    pair = [row for row in rows if row["frame_id"] in {"frame-1", "frame-3"}] + [
        {**row, "part_boxes": {}} for row in rows if row["frame_id"] not in {"frame-1", "frame-3"}]
    extents = coverage.estimate_part_extents(track=track, frames=[*frames[:3], shifted], classified=pair,
                                             body=body, parts=["upper_shelf"])
    assert extents["estimates"] == []
    assert extents["diagnostics"][0]["reason"] in {"part_views_disagree", "part_points_not_seen_by_another_view"}


def test_without_a_body_no_part_is_measured(cabinet):
    frames, track, regions = cabinet
    extents = coverage.estimate_part_extents(track=track, frames=frames, classified=classified_rows(regions, ALL_SHELVES),
                                             body=None, parts=["upper_shelf"])
    assert extents["estimates"] == [] and extents["diagnostics"] == [
        {"part_id": "upper_shelf", "reason": "body_bounds_unavailable", "frame_ids": []}]


# ---- into the builder's assembly frame ---------------------------------------------------------------------

def _record(body, rows, extents, *, selected=None):
    frames = selected or coverage.select_reference_frames(
        rows, **dict(zip(("seed_frame_ids", "seed_reasons"), coverage.measurement_seeds(rows, body, extents))))
    return {"status": "complete", "body_bounds": body, "hinge_edge": "left",
            "task_part_components": ["door_outer", "handle"],
            "observed_parts": sorted({part for row in rows for part in row["visible_parts"]}),
            "selected_frames": frames, "part_extents": extents}


def test_the_frames_extents_were_measured_from_are_shown_to_the_builder(cabinet):
    body, rows, extents = _extents(cabinet)
    seeds, reasons = coverage.measurement_seeds(rows, body, extents)
    assert seeds[0] in body["open_frame_ids"] and reasons[seeds[0]].startswith(coverage.DEPTH_SEED_REASON)
    # One frame shows both measured shelves; it is the depth view here, so it carries both reasons.
    assert "lower_shelf, upper_shelf where the extent was measured from depth" in reasons[seeds[0]]
    selected = coverage.select_reference_frames(rows, seed_frame_ids=seeds, seed_reasons=reasons)
    assert {row["frame_id"]: row["reason"] for row in selected}[seeds[0]] == reasons[seeds[0]]


def test_contract_carries_measured_boxes_in_the_assembly_frame(cabinet):
    body, rows, extents = _extents(cabinet)
    contract = coverage.assembly_contract(_record(body, rows, extents), articulation_kind="revolute",
                                          source_to_simulator_scale=2.0)
    depth = contract["body_depth"]["value_m"]
    assert depth == pytest.approx(2 * body["depth_m"])
    rows_by_part = {row["part_id"]: row for row in contract["part_extent_estimates"]}
    assert set(rows_by_part) == set(SHELVES) and contract["part_extent_diagnostics"] == []
    shown = {row["part_id"]: row["observed_frame_ids"] for row in contract["required_parts"]}
    for name, row in rows_by_part.items():
        measured = next(item for item in extents["estimates"] if item["part_id"] == name)
        assert row["basis"] == OBSERVED_ESTIMATE_BASIS and row["physical_measurement_proven"] is False
        assert set(row["frame_ids"]) <= set(shown[name]) and row["measurement_frame_ids"] == measured["frame_ids"]
        (b0, b1), (u0, u1), (v0, v1) = (measured[key] for key in ("behind_front_m", "across_from_centre_m",
                                                                    "above_bottom_m"))
        # +X out of the front (front plane at depth / 2), Z up from the bottom, both times the scale.
        assert row["box_assembly_m"] == {"minimum": pytest.approx([depth / 2 - 2 * b1, 2 * u0, 2 * v0], abs=1e-4),
                                         "maximum": pytest.approx([depth / 2 - 2 * b0, 2 * u1, 2 * v1], abs=1e-4)}
        assert row["uncertainty_m"] == pytest.approx([2 * v for v in measured["uncertainty_m"]])
        assert row["unit"] == "estimated_simulator_meters"
        assert row["body_scaling"] == {"source_to_simulator_scale": 2.0, "published_body_axis_ratios": None,
                                       "anchor": "observed_closed_front_bottom_centre"}
    # The builder's viewer-right is +Y: the observed right wall is at +width / 2.
    right = contract["body_extent_m"]["width"] / 2
    assert 0 < rows_by_part["upper_shelf"]["box_assembly_m"]["maximum"][1] < right


def test_a_published_resize_scales_each_measured_part_about_the_same_anchor(cabinet):
    body, rows, extents = _extents(cabinet)
    published = {"depth_m": 0.5, "width_m": 0.6, "height_m": 0.84,
                 "dimension_authority": "published_product_specification"}
    resized = published_body_bounds(body, published, source_to_simulator_scale=1.0)
    ratios = [published[key] / body[key] for key in ("depth_m", "width_m", "height_m")]
    plain = coverage.assembly_contract(_record(body, rows, extents), articulation_kind="revolute",
                                       source_to_simulator_scale=1.0)
    scaled = coverage.assembly_contract(_record(resized, rows, extents), articulation_kind="revolute",
                                        source_to_simulator_scale=1.0)
    assert scaled["body_extent_m"]["height"] == pytest.approx(0.84)
    for before, after in zip(plain["part_extent_estimates"], scaled["part_extent_estimates"]):
        for key in ("minimum", "maximum"):
            x, y, z = before["box_assembly_m"][key]
            # Distance behind the front, offset from the centre line and height above the bottom
            # each grow by that axis's published ratio.
            assert after["box_assembly_m"][key] == pytest.approx(
                [0.25 - (plain["body_depth"]["value_m"] / 2 - x) * ratios[0], y * ratios[1], z * ratios[2]], abs=1e-4)
        assert after["uncertainty_m"] == pytest.approx([v * r for v, r in zip(before["uncertainty_m"], ratios)],
                                                       abs=2e-5)
        assert after["unit"] == "estimated_simulator_meters_scaled_to_published_body"
        assert after["body_scaling"]["published_body_axis_ratios"] == pytest.approx(
            dict(zip(("depth", "width", "height"), ratios)), abs=1e-5)


def test_measured_frames_the_builder_is_not_shown_are_not_cited(cabinet):
    body, rows, extents = _extents(cabinet)
    record = _record(body, rows, extents)
    record["part_extents"] = {**extents, "estimates": [{**row, "frame_ids": ["decoded-000000999"]}
                                                       for row in extents["estimates"]]}
    contract = coverage.assembly_contract(record, articulation_kind="revolute", source_to_simulator_scale=1.0)
    assert contract["part_extent_estimates"] == []
    assert {row["reason"] for row in contract["part_extent_diagnostics"]} == {"measured_frames_not_shown_to_builder"}


# ---- end to end: coverage -> compile -> plan -----------------------------------------------------------------

def _translated(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def _fixture_background():
    """The preparation fixture's table (0.5 m below the camera) and far wall, at this resolution."""
    rows = np.arange(H)[:, None] * np.ones((1, W))
    return np.where(rows > H / 2, 0.5 * F / np.maximum(rows - H / 2, 1e-6), 3.0)


def _cabinet_inputs(root: Path):
    from tests.test_website_task_preparation import _masks

    views = [("frame-0", _translated(0.0), "closed"), ("frame-1", _translated(0.0), "open"),
             ("frame-2", _translated(0.1), "open")]
    # The fixture scene: camera 0.5 m above the table, the cabinet standing on it 1 m away.
    frames, track, regions = capture(root / "geometry", views, background=_fixture_background())
    geometry = {"schema_version": "website_source_geometry.v1", "frames": frames, "unit": "estimated_meters",
                "scale_status": "model_estimated", "metric_measurement_proven": False}
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    masks = _masks(geometry, destination=False, articulated=True)
    target = {**masks["targets"][0], "semantic_label": "storage cabinet", "articulated_part": "cabinet door",
              "articulation_kind": "revolute", "track": {**masks["targets"][0]["track"], **track}}
    rows = classified_rows(regions, {"frame-1": {name: name for name in SHELVES},
                                     "frame-2": {name: name for name in SHELVES}}, grow=0.2)
    for index, row in enumerate(rows):
        path = root / f"{row['frame_id']}-upright.png"
        Image.fromarray(np.full((H, W, 3), 60 + 7 * index, dtype=np.uint8)).save(path)
        row.update(path=str(path), sha256=_sha256_file(path))
    classified = {"frames": rows, "task_part_components": ["door_outer", "handle"], "hinge_edges": ["left"],
                  "receipts": [{"binding_digest": "sha256:" + "1" * 64, "result_digest": "sha256:" + "2" * 64}]}
    states = {row["frame_id"]: row["part_state"] for row in rows}
    body, blockers = coverage.estimate_body_bounds(track=track, frames=frames, states=states)
    extents = coverage.estimate_part_extents(track=track, frames=frames, classified=rows, body=body,
                                             parts=coverage.fixed_interior_parts(classified, articulation_kind="revolute"))
    seeds, reasons = coverage.measurement_seeds(rows, body, extents)
    target["authoring_coverage"] = coverage.coverage_record(
        target_id=target["target_id"], binding=coverage.coverage_binding(target=target, source_geometry=geometry),
        articulation_kind="revolute", classified=classified,
        selected=coverage.select_reference_frames(rows, seed_frame_ids=seeds, seed_reasons=reasons),
        body_bounds=body, body_blockers=blockers, candidate_count=len(rows), part_extents=extents)
    masks = {**masks, "targets": [target]}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    removal = {"schema_version": "clean_plate_removal_manifest.v1", "entries": [
        {"target_id": target["target_id"], "semantic_label": "storage cabinet", "task_effect": "manipulated",
         "disposition": "remove", "articulated_part": "cabinet door", "articulation_kind": "revolute",
         "compose_back": {"replacement_asset_id": None, "pose_world": None,
                          "replacement_asset_frame_registration_uri": None}}]}
    return {"source_geometry": geometry, "removal_manifest": removal, "task_masks": masks}


def _fixture_registration(*, source_geometry, collision_mesh_path, anchor=None, focus_bounds=None, **_):
    """The preparation fixture's exact source-to-runtime similarity (registration has its own tests)."""
    from tests.test_website_task_preparation import RUNTIME_ROTATION, RUNTIME_SCALE, RUNTIME_TRANSLATION

    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = RUNTIME_SCALE * RUNTIME_ROTATION, RUNTIME_TRANSLATION
    return {"schema_version": "website_source_registration.v1", "source_to_runtime": matrix.tolist(),
            "scale": RUNTIME_SCALE, "rotation": RUNTIME_ROTATION.tolist(), "translation": RUNTIME_TRANSLATION.tolist(),
            "trimmed_rmse_runtime_units": 0.0, "trimmed_rmse_m": 0.0, "runner_up_ratio": None, "anchor": None,
            "ground_plane": None, "task_region": None, "source_geometry_digest": source_geometry["digest"],
            "collision_mesh_digest": _sha256_file(collision_mesh_path), "scale_status": "estimated_registration",
            "physical_scale_measured": False, "physical_registration_proven": False}


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    from blueprint_pipeline import website_task_preparation as preparation
    from tests.test_website_task_preparation import _arguments

    root = tmp_path_factory.mktemp("e2e")
    args = _arguments(root)
    args.update(_cabinet_inputs(root))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(preparation, "register_source_to_runtime", _fixture_registration)
        yield args, preparation.compile_website_scene_preparation(**args)


@pytest.fixture
def fixture_registration(monkeypatch):
    from blueprint_pipeline import website_task_preparation as preparation
    monkeypatch.setattr(preparation, "register_source_to_runtime", _fixture_registration)


def test_measured_shelves_flow_into_the_plan_as_observed_estimates(compiled):
    args, preparation = compiled
    assert preparation["status"] == "intake_ready", preparation["blockers"]
    configuration = preparation["authoring_inputs"]["configuration"]
    assert configuration["assembly_family"] == "hinged_door_appliance" and configuration["hinge_edge"] == "left"
    record = args["task_masks"]["targets"][0]["authoring_coverage"]
    assert {row["part_id"] for row in record["part_extents"]["estimates"]} == set(SHELVES)
    assert {row["frame_id"] for row in record["part_localization"]} == {"frame-1", "frame-2"}
    estimates = {row["part_id"]: row for row in configuration["part_extent_estimates"]}
    assert set(estimates) == set(SHELVES) and configuration["part_extent_diagnostics"] == []
    references = {row["frame_id"] for row in configuration["reference_frames"]}
    for row in estimates.values():
        assert row["basis"] == OBSERVED_ESTIMATE_BASIS and set(row["frame_ids"]) <= references
        assert row["measurement_frame_ids"] == ["frame-1", "frame-2"]
    # Registration scale 0.5 x 2 metres per unit: simulator metres equal the source estimate.
    upper, lower = estimates["upper_shelf"]["box_assembly_m"], estimates["lower_shelf"]["box_assembly_m"]
    assert upper["maximum"][2] == pytest.approx(SHELVES["upper_shelf"][1], abs=0.015)
    assert lower["maximum"][2] == pytest.approx(SHELVES["lower_shelf"][1], abs=0.015)

    plan = plan_articulated_assembly(configuration)
    bases = plan["part_dimension_bases"]
    for name in SHELVES:
        basis = bases[name]
        assert basis["basis"] == OBSERVED_ESTIMATE_BASIS and basis["frame_ids"] == estimates[name]["frame_ids"]
        assert basis["cavity_id"] == "tub" and basis["physical_measurement_proven"] is False
        link = next(row for row in plan["links"] if row["link_id"] == name)
        size = plan["parts"][name]["dimensions_m"]
        # The shelf sits where it was seen: its top at the measured height (inside the cavity).
        assert link["rest_translation_m"][2] + size[2] == pytest.approx(
            estimates[name]["box_assembly_m"]["maximum"][2], abs=1e-4)
        assert "estimated from frames" in plan["parts"][name]["description"]
    # Every other template-placed part is still labelled a prior.
    assert {part for part, basis in bases.items() if basis["basis"] != OBSERVED_ESTIMATE_BASIS} == {
        "body_front", "handle"}
    assert math.isclose(plan["assembly_dimensions_m"]["depth_x"], configuration["body_depth"]["value_m"],
                        abs_tol=1e-5)


def test_an_estimate_the_builder_cannot_place_is_withdrawn_with_its_reason(compiled, fixture_registration,
                                                                          monkeypatch):
    """Drawer-family fixed bays share one drawer solid: a measured box for one cannot replace it."""
    from blueprint_pipeline import website_task_preparation as preparation
    from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError

    args, _ = compiled
    refused = []

    def plan(configuration):
        if any(row["part_id"] == "lower_shelf" for row in configuration.get("part_extent_estimates") or []):
            refused.append(True)
            raise AssetAuthoringError("articulated_part_extent_estimate_unplaceable:lower_shelf")
        return {}
    monkeypatch.setattr("blueprint_pipeline.task_object_articulated_packaging.plan_articulated_assembly", plan)
    value = preparation.compile_website_scene_preparation(**{**args, "output_root": args["output_root"].parent / "w"})
    assert value["status"] == "intake_ready", value["blockers"]
    assert refused == [True]
    configuration = value["authoring_inputs"]["configuration"]
    assert [row["part_id"] for row in configuration["part_extent_estimates"]] == ["upper_shelf"]
    (withdrawn,) = configuration["part_extent_diagnostics"]
    assert withdrawn["part_id"] == "lower_shelf" and withdrawn["reason"] == "builder_cannot_place_measured_extent"
    assert withdrawn["builder_code"] == "articulated_part_extent_estimate_unplaceable:lower_shelf"
    # Any other refusal still holds the build.
    def malformed(configuration):
        raise AssetAuthoringError("articulated_part_extent_estimates_invalid")
    monkeypatch.setattr("blueprint_pipeline.task_object_articulated_packaging.plan_articulated_assembly", malformed)
    held = preparation.compile_website_scene_preparation(**{**args, "output_root": args["output_root"].parent / "h"})
    assert "website_assembly_builder_refused:articulated_part_extent_estimates_invalid" in held["blockers"]


def test_a_body_grounded_on_the_floor_raises_each_measured_part_by_the_same_gap(compiled, monkeypatch):
    """The unseen lowest band: the body's bottom moves down to the floor; the parts stay where they were seen."""
    from blueprint_pipeline import website_task_preparation as preparation

    args, before = compiled
    gap = 0.05  # Runtime units: 0.1 m at 2 m per unit.

    def register(**kwargs):
        return {**_fixture_registration(**kwargs), "ground_plane": {"checked": True, "observed_floor_offset_m": 0.0}}

    def ground(mesh, lower, upper, *, up, meters_per_unit, floor_height, up_sign=1):
        return {"top_runtime_units": float(lower[up]) - gap, "aabb_min": list(lower), "aabb_max": list(upper),
                "face_indices": [], "extended_runtime_units": gap, "basis": "registered_observed_floor_plane",
                "physical_measurement": False}
    monkeypatch.setattr(preparation, "register_source_to_runtime", register)
    monkeypatch.setattr(preparation, "support_under", lambda *args, **kwargs: None)
    monkeypatch.setattr(preparation, "ground_on_observed_floor", ground)
    grounded = preparation.compile_website_scene_preparation(**{
        **args, "output_root": args["output_root"].parent / "grounded",
        "base_scene": {**args["base_scene"], "anchor": {"frame_id": "frame-0", "kind": "test"}}})
    assert grounded["status"] == "intake_ready", grounded["blockers"]
    configuration = grounded["authoring_inputs"]["configuration"]
    assert configuration["body_extent_m"]["grounding"]["extended_to_observed_floor_m"] == pytest.approx(0.1)
    ungrounded = {row["part_id"]: row for row in before["authoring_inputs"]["configuration"]["part_extent_estimates"]}
    assert {row["part_id"] for row in configuration["part_extent_estimates"]} == set(SHELVES)
    for row in configuration["part_extent_estimates"]:
        reference = ungrounded[row["part_id"]]["box_assembly_m"]
        for key in ("minimum", "maximum"):
            assert row["box_assembly_m"][key] == pytest.approx(
                [reference[key][0], reference[key][1], reference[key][2] + 0.1], abs=1e-5)
        assert row["body_scaling"]["raised_by_body_grounding_m"] == pytest.approx(0.1)
    plan = plan_articulated_assembly(configuration)
    assert {plan["part_dimension_bases"][name]["basis"] for name in SHELVES} == {OBSERVED_ESTIMATE_BASIS}


def test_a_measured_extent_cites_only_frames_the_provider_budget_keeps(tmp_path, monkeypatch):
    from blueprint_pipeline import authoring_frame_budget as budget

    rows = []
    for index, frame_id in enumerate(("frame-1", "frame-2", "frame-3")):
        path = tmp_path / f"{frame_id}.png"
        Image.fromarray(np.full((12, 16, 3), 40 * index, dtype=np.uint8)).save(path)
        rows.append({"frame_id": frame_id, "path": str(path), "sha256": _sha256_file(path), "selection_rank": index,
                     "visible_parts": ["shelf"], "part_state": "open"})
    box = {"minimum": [0.0, -0.1, 0.2], "maximum": [0.2, 0.1, 0.22]}
    contract = {"reference_frames": rows, "body_depth": {"frame_ids": ["frame-1"]},
                "required_parts": [{"part_id": "shelf", "observed_frame_ids": ["frame-1", "frame-2", "frame-3"]}],
                "part_extent_estimates": [
                    {"part_id": "shelf", "basis": OBSERVED_ESTIMATE_BASIS, "frame_ids": ["frame-2", "frame-3"],
                     "box_assembly_m": box, "uncertainty_m": [0.02] * 3},
                    {"part_id": "divider", "basis": OBSERVED_ESTIMATE_BASIS, "frame_ids": ["frame-3"],
                     "box_assembly_m": box, "uncertainty_m": [0.02] * 3}],
                "part_extent_diagnostics": [{"part_id": "rail", "reason": "part_never_boxed", "frame_ids": []}]}
    monkeypatch.setattr(budget, "choose_frames", lambda rows, **_: [dict(row) for row in rows[:2]])
    fitted = budget.fit_reference_frames(contract, provider="openai", output_root=tmp_path / "out")
    assert fitted["part_extent_estimates"] == [{**contract["part_extent_estimates"][0], "frame_ids": ["frame-2"]}]
    assert fitted["part_extent_diagnostics"] == [
        {"part_id": "rail", "reason": "part_never_boxed", "frame_ids": []},
        {"part_id": "divider", "reason": "measured_frames_dropped_by_frame_budget", "frame_ids": ["frame-3"]}]
    # A contract without measured extents is returned without those keys, exactly as before.
    plain = {key: value for key, value in contract.items() if not key.startswith("part_extent")}
    assert "part_extent_estimates" not in budget.fit_reference_frames(plain, provider="openai",
                                                                      output_root=tmp_path / "out")
