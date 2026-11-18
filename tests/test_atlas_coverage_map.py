"""The coverage map before the refined splat: weak where clutter was filmed from one direction, never on flat floor.

The scene is ray-cast: a flat floor (y = 0, up is +y) with a cluster of nine
4 cm spheres at the origin, filmed from 1.2 m by twelve cameras either on a
3 degree arc (itest-c0cbbf20's table-top items had 2.9-3.2 degrees) or on a
70 degree one.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from blueprint_pipeline import atlas_coverage_map as coverage
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

W, H = 320, 240
K = {"fx": 500.0, "fy": 500.0, "cx": W / 2, "cy": H / 2, "width": W, "height": H}
SPHERES = [(x, .04, z) for x in (-.09, 0, .09) for z in (-.09, 0, .09)]


def _look_at(position, target):
    forward = np.asarray(target, float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(right, forward), -forward])


def _quaternion(m):
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return [(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w), w]


def _frame(index, position, target=(0, .03, 0)):
    rotation = _look_at(np.asarray(position, float), target)
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u + .5 - K["cx"]) / K["fx"], -(v + .5 - K["cy"]) / K["fy"], -np.ones(u.shape)], -1) @ rotation.T
    depth = np.full((H, W), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        floor = -position[1] / rays[..., 1]
    depth = np.where(floor > 0, floor, depth)
    for centre in SPHERES:
        oc = position - np.asarray(centre)
        b = np.einsum("ijk,k->ij", rays, oc)
        a = np.einsum("ijk,ijk->ij", rays, rays)
        disc = b * b - a * (oc @ oc - .04 ** 2)
        t = (-b - np.sqrt(np.maximum(disc, 0))) / a
        depth = np.where((disc > 0) & (t > 0) & (t < depth), t, depth)
    # The ray parameter is the camera-frame depth: the local ray has z = -1.
    depth = np.where(np.isfinite(depth) & (depth < 4), depth, 0).astype(np.float32)
    rgb = np.full((H, W, 3), 128, np.uint8)
    rgb[::4] = 30
    return {"frame_id": f"decoded-{index:09d}", "intrinsics": dict(K), "rotation": rotation,
            "position": np.asarray(position, float),
            "extrinsics": {"coordinateSystem": "rub", "position": list(map(float, position)),
                           "quaternion": _quaternion(rotation)},
            "depth": depth, "confidence": (depth > 0).astype(np.float32), "rgb": rgb}


def _arc(half_degrees, count=12):
    frames = []
    for index, angle in enumerate(np.linspace(-half_degrees, half_degrees, count)):
        a = math.radians(angle)
        frames.append(_frame(index, np.array([1.0 * math.sin(a), .65, 1.0 * math.cos(a)])))
    return frames


def _cluster_voxels(mapped):
    arrays, centroid = mapped["arrays"], mapped["arrays"]["centroid"]
    near = np.linalg.norm(centroid - [0, .04, 0], axis=1) < .14
    on_cluster = near & (centroid[:, 1] > .02)
    floor = (np.abs(centroid[:, 1]) < .02) & (np.linalg.norm(centroid[:, [0, 2]], axis=1) > .3)
    return arrays, on_cluster, floor


def test_clutter_filmed_over_a_narrow_arc_is_a_weak_region_and_the_flat_floor_is_not():
    frames = _arc(1.5)
    mapped = coverage.coverage_map(frames, meters_per_unit=1.0, priority_frame_ids=[frames[6]["frame_id"]])
    arrays, on_cluster, floor = _cluster_voxels(mapped)
    assert on_cluster.sum() > 20 and floor.sum() > 200
    assert np.median(arrays["angle_spread_deg"][on_cluster]) == pytest.approx(3.0, abs=.5)
    assert np.nanmedian(arrays["mm_per_px"][on_cluster]) < coverage.COARSE_MM_PER_PIXEL
    assert arrays["weak"][on_cluster].mean() > .6
    assert arrays["flat"][floor].mean() > .95 and arrays["weak"][floor].mean() < .05
    region = mapped["regions"][0]
    assert np.linalg.norm(np.asarray(region["centroid"]) - [0, .04, 0]) < .1
    assert region["id"] == "wr-000" and region["reasons"] == ["narrow_angle_spread"]
    # The widest pair seen from the region: the 3 degree arc, foreshortened by the cameras' height.
    ends = [frames[0]["position"] - region["centroid"], frames[-1]["position"] - region["centroid"]]
    widest = math.degrees(math.acos(ends[0] @ ends[1] / np.linalg.norm(ends[0]) / np.linalg.norm(ends[1])))
    assert region["angle_spread_deg"] == pytest.approx(widest, abs=.05) and region["view_count"] == 12
    assert set(region["seen_by"]) == {frame["frame_id"] for frame in frames}
    assert region["orbit_deg"] == [15, 30] and region["seen_by_job_route_view"] is True
    orbit = region["suggested_orbit"]
    assert [camera["orbit_degrees"] for camera in orbit["cameras"]] == [-30, -15, 15, 30]
    assert orbit["axis"] == pytest.approx([0, 1, 0], abs=1e-6)
    # Each orbit camera stands at the filmed distance, turned about the up axis, looking at the region.
    from blueprint_pipeline.atlas_depth_splat import _rotation_matrix
    for camera in orbit["cameras"]:
        position = np.asarray(camera["extrinsics"]["position"])
        toward = position - region["centroid"]
        assert np.linalg.norm(toward) == pytest.approx(orbit["radius_units"], rel=1e-6)
        filmed = np.asarray(orbit["filmed_direction"])
        horizontal = [toward[0], toward[2]], [filmed[0], filmed[2]]
        turn = math.degrees(math.atan2(np.cross(*horizontal), np.dot(*horizontal)))
        assert abs(turn) == pytest.approx(abs(camera["orbit_degrees"]), abs=.01)
        forward = -_rotation_matrix(camera["extrinsics"]["quaternion"])[:, 2]
        assert forward @ (-toward / np.linalg.norm(toward)) == pytest.approx(1, abs=1e-6)


def test_the_same_clutter_filmed_over_a_wide_arc_is_not_weak():
    mapped = coverage.coverage_map(_arc(35), meters_per_unit=1.0)
    arrays, on_cluster, floor = _cluster_voxels(mapped)
    assert np.median(arrays["angle_spread_deg"][on_cluster]) > 30
    assert arrays["weak"][on_cluster].mean() < .1 and arrays["weak"][floor].mean() < .05
    # Only floor in the spheres' shadow, seen by a view or two, may still be weak there.
    assert (arrays["region"][on_cluster] >= 0).mean() < .1


def test_clutter_seen_only_at_coarse_resolution_is_weak_even_from_a_wide_arc():
    mapped = coverage.coverage_map(_arc(35), meters_per_unit=4.0)  # every unit is 4 m: 9 mm a pixel
    arrays, on_cluster, floor = _cluster_voxels(mapped)
    assert np.nanmedian(arrays["mm_per_px"][on_cluster]) > coverage.COARSE_MM_PER_PIXEL
    assert arrays["weak"][on_cluster].mean() > .6 and arrays["weak"][floor].mean() < .05
    assert mapped["regions"][0]["reasons"] == ["coarse_mm_per_pixel"]


def test_the_record_is_sealed_in_the_atlas_root_read_back_once_per_pose_and_loaded_by_capture(tmp_path):
    atlas_root = tmp_path / "capture" / "pipeline" / "atlas"
    frames = _arc(1.5)
    record = coverage.write_weak_regions(frames=frames, output_root=atlas_root, pose_digest="sha256:" + "1" * 64,
                                         meters_per_unit=1.1)
    saved = json.loads((atlas_root / coverage.WEAK_REGIONS_FILE).read_text())
    assert saved == record and record["digest"] == canonical_digest(record, digest_field="digest")
    assert record["schema_version"] == "atlas_weak_regions.v1" and record["status"] == "mapped"
    assert record["voxel_size_m"] == pytest.approx(record["grid"]["voxel_size_units"] * 1.1, rel=1e-5)
    assert record["inputs_digest"].startswith("sha256:") and record["purchase_performed"] is False
    for key in ("id", "centroid", "extent", "angle_spread_deg", "mm_per_px", "flatness", "view_count", "seen_by",
                "orbit_deg"):
        assert key in record["regions"][0]
    assert coverage.load_weak_regions(tmp_path / "capture") == record["regions"]
    # The same pose reads the record back; a changed record refuses rather than being trusted.
    assert coverage.write_weak_regions(frames=frames, output_root=atlas_root, pose_digest="sha256:" + "1" * 64,
                                       meters_per_unit=1.1)["seconds"] == record["seconds"]
    (atlas_root / coverage.WEAK_REGIONS_FILE).write_text(json.dumps({**record, "regions": []}))
    with pytest.raises(ValueError, match="atlas_weak_regions_changed"):
        coverage.load_weak_regions(tmp_path / "capture")


def test_a_job_without_the_inputs_skips_cleanly(tmp_path):
    assert coverage.load_weak_regions(tmp_path) is None
    frames = [{**frame, "depth": np.zeros_like(frame["depth"]), "confidence": np.zeros_like(frame["confidence"])}
              for frame in _arc(1.5, count=3)]
    record = coverage.write_weak_regions(frames=frames, output_root=tmp_path / "pipeline/atlas", pose_digest=None)
    assert record["status"] == "not_mapped" and record["regions"] == [] and record["reason"] == "atlas_coverage_no_ranged_surface"
    assert coverage.load_weak_regions(tmp_path) is None
    # The pipeline's call never raises, whatever it is given.
    assert coverage.coverage_before_splat(frames=[{"frame_id": "a"}], pose={}, output_root=tmp_path / "x") is None
