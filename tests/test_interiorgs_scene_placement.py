"""Hermetic tests for the InteriorGS labeled-splat placement backend.

Covers: labels.json -> SceneObject catalog, structure.json -> walls/rooms,
the structure-aware floor probe (walk-over + same-room rules), instance-token
target resolution, stance cameras, and the compressed-PLY chunk-bounds reader.
No GPU, no network, no real scene assets — everything is synthetic.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from blueprint_pipeline.gaussian_splat_decode import read_compressed_ply_chunk_bounds
from blueprint_pipeline.scene_placement import (
    InteriorGSSceneSpatialIndex,
    StandPose,
    build_interiorgs_object_index,
    build_interiorgs_probe,
    build_scene_index,
    compute_stand_pose,
    load_interiorgs_labels,
    load_interiorgs_structure,
    inventory_articulated_open_close_candidates,
    link_mounted_camera_spec,
    point_in_polygon,
    resolve_target_by_instance,
    stance_task_cameras,
    supporting_fixtures_for,
    to_splat_render_specs,
    validate_stand_pose,
)
from blueprint_pipeline.scene_placement.interiorgs_index import (
    INTERIORGS_LABELS_SOURCE,
    INTERIORGS_STRUCTURE_SOURCE,
)


# ----------------------------- synthetic fixtures -----------------------------

def _box_corners(x0, y0, z0, x1, y1, z1):
    return [
        {"x": x, "y": y, "z": z}
        for z in (z0, z1)
        for x, y in ((x0, y0), (x0, y1), (x1, y1), (x1, y0))
    ]


def _labels_payload():
    return [
        {"ins_id": "92", "label": "sideboards",
         "bounding_box": _box_corners(2.0, 5.5, 0.0, 3.0, 6.0, 0.85)},
        {"ins_id": "88", "label": "pot",
         "bounding_box": _box_corners(2.4, 5.6, 0.85, 2.6, 5.8, 0.95)},
        {"ins_id": "80", "label": "carpet",
         "bounding_box": _box_corners(1.0, 1.0, -0.0000001, 5.0, 5.0, 0.01)},
        {"ins_id": "50", "label": "wall cabinet",
         "bounding_box": _box_corners(2.0, 5.0, 1.5, 3.0, 6.0, 2.2)},
        {"ins_id": "79", "label": "bath heater",
         "bounding_box": _box_corners(2.8, 2.8, 2.55, 3.2, 3.2, 2.6)},
        {"ins_id": "61", "label": "door",
         "bounding_box": _box_corners(6.0, 2.6, 0.0, 6.24, 3.5, 2.1)},
        # degenerate entries that must be skipped
        {"ins_id": "999", "label": "broken", "bounding_box": []},
        {"ins_id": "", "label": "unnamed",
         "bounding_box": _box_corners(0, 0, 0, 1, 1, 1)},
    ]


def _structure_payload():
    return {
        "rooms": [
            {"profile": [[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]]},
            {"profile": [[6.24, 0.0], [9.0, 0.0], [9.0, 6.0], [6.24, 6.0]]},
        ],
        "walls": [
            {"thickness": 0.24, "height": 2.6,
             "location": [[6.12, 0.0], [6.12, 6.0]]},
            {"thickness": 0.2, "height": 2.6,
             "location": [[0.0, 1.0], [1.0, 0.0]]},  # skew segment
        ],
        "holes": [
            {"type": "DOOR", "thickness": 0.24,
             "profile": [[6.12, 2.6, 2.1], [6.12, 3.5, 2.1], [6.12, 3.5, 0.0], [6.12, 2.6, 0.0]]},
        ],
    }


@pytest.fixture()
def scene_dir(tmp_path: Path) -> Path:
    (tmp_path / "labels.json").write_text(json.dumps(_labels_payload()))
    (tmp_path / "structure.json").write_text(json.dumps(_structure_payload()))
    return tmp_path


def write_synthetic_compressed_ply(path: Path, chunk_bounds: list[tuple], vertex_count: int = 4) -> Path:
    """Minimal PlayCanvas-compressed-layout PLY: real chunk floats, dummy vertex/sh."""
    chunk_props = [
        "min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
        "min_scale_x", "min_scale_y", "min_scale_z",
        "max_scale_x", "max_scale_y", "max_scale_z",
        "min_r", "min_g", "min_b", "max_r", "max_g", "max_b",
    ]
    header = ["ply", "format binary_little_endian 1.0", f"element chunk {len(chunk_bounds)}"]
    header += [f"property float {name}" for name in chunk_props]
    header += [f"element vertex {vertex_count}"]
    header += [f"property uint packed_{name}" for name in ("position", "rotation", "scale", "color")]
    header += [f"element sh {vertex_count}", "property uchar f_rest_0", "end_header"]
    body = b""
    for bounds in chunk_bounds:
        row = list(bounds) + [0.0] * (len(chunk_props) - len(bounds))
        body += struct.pack(f"<{len(chunk_props)}f", *row)
    body += b"\x00" * (vertex_count * 16 + vertex_count)
    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + body)
    return path


# ----------------------------- labels -----------------------------

class TestLoadLabels:
    def test_catalog_shape_and_normalization(self, scene_dir):
        objects = load_interiorgs_labels(scene_dir / "labels.json")
        by_id = {o.id: o for o in objects}
        assert set(by_id) == {"92", "88", "80", "50", "79", "61"}
        pot = by_id["88"]
        assert pot.label == "pot"
        assert pot.bbox_min == (2.4, 5.6, 0.85)
        assert pot.bbox_max == (2.6, 5.8, 0.95)
        assert pot.centroid == pytest.approx((2.5, 5.7, 0.9))
        assert pot.source == INTERIORGS_LABELS_SOURCE
        assert pot.extra["instance_name"] == "pot_88"
        assert by_id["92"].label == "sideboards"
        assert by_id["50"].label == "wall_cabinet"  # spaces normalized to underscores

    def test_degenerate_entries_skipped(self, scene_dir):
        objects = load_interiorgs_labels(scene_dir / "labels.json")
        ids = {o.id for o in objects}
        assert "999" not in ids and "" not in ids

    def test_rotated_box_retains_exact_corners_and_conservative_aabb(self, tmp_path):
        rotated_corners = [
            {"x": x, "y": y, "z": z}
            for z in (0.0, 1.0)
            for x, y in ((0.0, 1.0), (1.0, 2.0), (2.0, 1.0), (1.0, 0.0))
        ]
        labels = tmp_path / "labels.json"
        labels.write_text(
            json.dumps(
                [{"ins_id": "rotated-1", "label": "work table", "bounding_box": rotated_corners}]
            )
        )

        [table] = load_interiorgs_labels(labels)

        assert table.bbox_min == (0.0, 0.0, 0.0)
        assert table.bbox_max == (2.0, 2.0, 1.0)
        assert table.extra["placement_bounds_kind"] == "conservative_world_aabb"
        assert table.extra["oriented_bounding_box"]["corners_world_m"] == [
            [corner["x"], corner["y"], corner["z"]] for corner in rotated_corners
        ]


class TestObjectIndex:
    def test_deterministic_hash_bound_index_preserves_authority(self, scene_dir):
        splat = scene_dir / "3dgs_compressed.ply"
        splat.write_bytes(b"synthetic-splat-fixture")

        first = build_interiorgs_object_index(
            scene_dir / "labels.json",
            splat_path=splat,
            structure_path=scene_dir / "structure.json",
        )
        replay = build_interiorgs_object_index(
            scene_dir / "labels.json",
            splat_path=splat,
            structure_path=scene_dir / "structure.json",
        )

        assert first == replay
        assert first["schema_version"] == "object_index.v2"
        assert [item["label"] for item in first["objects"]] == sorted(
            item["label"] for item in first["objects"]
        )
        pot = next(item for item in first["objects"] if item["id"] == "88")
        assert len(pot["orientedBoundingBox"]["corners_world_m"]) == 8
        assert pot["boundingBox"]["kind"] == "conservative_world_aabb"
        assert first["scene_structure"] == {
            "room_count": 2,
            "wall_count": 2,
            "hole_count": 1,
            "source_digest": first["provenance"]["source_files"]["structure"]["sha256"],
        }
        for source in first["provenance"]["source_files"].values():
            assert source["sha256"].startswith("sha256:")
            assert len(source["sha256"]) == 71
            assert source["size_bytes"] > 0
        assert first["claim_boundary"]["raw_capture_authority"] is False
        assert first["claim_boundary"]["collision_or_physics_authority"] is False
        assert first["claim_boundary"]["comparative_policy_ranking_verdict"] == (
            "thesis_not_supported"
        )


class TestArticulatedOpenCloseInventory:
    def test_original_rigid_fixture_does_not_become_articulated(self, tmp_path):
        labels = tmp_path / "labels.json"
        labels.write_text(
            json.dumps(
                [
                    {
                        "ins_id": "160",
                        "label": "canned beverage",
                        "bounding_box": _box_corners(0, 0, 0.8, 0.07, 0.07, 0.97),
                    },
                    {
                        "ins_id": "299",
                        "label": "TV cabinet",
                        "bounding_box": _box_corners(-1, -1, 0, 1, 1, 0.8),
                    },
                ]
            )
        )

        observed = inventory_articulated_open_close_candidates(
            load_interiorgs_labels(labels)
        )

        assert observed["candidate_count"] == 0
        assert observed["aggregate_only_count"] == 1
        assert observed["aggregate_only"][0]["ins_id"] == "299"
        assert observed["claim_boundary"]["task_selected"] is False

    def test_new_fixture_prioritizes_oven_but_does_not_infer_joint(self, tmp_path):
        labels = tmp_path / "labels.json"
        labels.write_text(
            json.dumps(
                [
                    {
                        "ins_id": "172",
                        "label": "oven",
                        "bounding_box": _box_corners(4.5, -1.2, 0.08, 5.2, -0.5, 0.69),
                    },
                    {
                        "ins_id": "83",
                        "label": "door",
                        "bounding_box": _box_corners(0, 0, 0, 0.2, 0.9, 2.1),
                    },
                    {
                        "ins_id": "84",
                        "label": "cabinet",
                        "bounding_box": _box_corners(1, 1, 0, 2, 2, 1),
                    },
                ]
            )
        )

        observed = inventory_articulated_open_close_candidates(
            load_interiorgs_labels(labels)
        )

        assert [row["ins_id"] for row in observed["candidates"]] == ["172", "83"]
        assert observed["candidates"][0]["candidate_kind"] == "appliance_assembly"
        assert observed["candidates"][0]["articulation_qualified"] is False
        assert observed["aggregate_only"][0]["ins_id"] == "84"
        assert observed["claim_boundary"]["joint_or_articulation_inferred"] is False


# ----------------------------- structure -----------------------------

class TestLoadStructure:
    def test_rooms_and_walls(self, scene_dir):
        structure = load_interiorgs_structure(scene_dir / "structure.json")
        assert len(structure.rooms) == 2
        assert len(structure.wall_boxes) == 2
        wall = structure.wall_boxes[0]
        assert wall.source == INTERIORGS_STRUCTURE_SOURCE
        assert wall.label == "wall"
        # centerline x=6.12, thickness 0.24 -> box x in [6.0, 6.24]; ends sealed by half_t
        assert wall.bbox_min[0] == pytest.approx(6.0)
        assert wall.bbox_max[0] == pytest.approx(6.24)
        assert wall.bbox_min[1] == pytest.approx(-0.12)
        assert wall.bbox_max[1] == pytest.approx(6.12)
        assert wall.bbox_max[2] == pytest.approx(2.6)
        assert wall.extra["axis_aligned"] is True

    def test_skew_wall_conservative_box(self, scene_dir):
        structure = load_interiorgs_structure(scene_dir / "structure.json")
        skew = structure.wall_boxes[1]
        assert skew.extra["axis_aligned"] is False
        assert skew.bbox_min[0] == pytest.approx(-0.1)
        assert skew.bbox_max[0] == pytest.approx(1.1)

    def test_room_index_of_point(self, scene_dir):
        structure = load_interiorgs_structure(scene_dir / "structure.json")
        assert structure.room_index_of_point((3.0, 3.0)) == 0
        assert structure.room_index_of_point((7.0, 3.0)) == 1
        assert structure.room_index_of_point((6.12, 3.0)) is None  # wall band
        assert structure.room_index_of_point((50.0, 50.0)) is None


class TestPointInPolygon:
    def test_square(self):
        poly = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
        assert point_in_polygon((2.0, 2.0), poly)
        assert not point_in_polygon((5.0, 2.0), poly)

    def test_concave_L(self):
        poly = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        assert point_in_polygon((1.0, 3.0), poly)
        assert not point_in_polygon((3.0, 3.0), poly)  # inside the notch

    def test_degenerate(self):
        assert not point_in_polygon((0.0, 0.0), [(0, 0), (1, 1)])


# ----------------------------- the index -----------------------------

class TestIndex:
    def test_objects_and_obstacles(self, scene_dir):
        index = InteriorGSSceneSpatialIndex(
            scene_dir / "labels.json", scene_dir / "structure.json"
        )
        assert len(index.objects()) == 6
        assert len(index.obstacle_boxes()) == 8  # + 2 wall boxes
        assert index.floor_z == 0.0  # sub-mm export noise snapped to the floor plane

    def test_factory_backend(self, scene_dir):
        index = build_scene_index(
            "interiorgs",
            labels_path=scene_dir / "labels.json",
            structure_path=scene_dir / "structure.json",
        )
        assert isinstance(index, InteriorGSSceneSpatialIndex)

    def test_object_by_instance(self, scene_dir):
        index = InteriorGSSceneSpatialIndex(scene_dir / "labels.json")
        assert index.object_by_instance("88").label == "pot"
        assert index.object_by_instance("nope") is None

    def test_floor_ignores_outlier_bottom(self, tmp_path):
        # One label dipping 8cm under the floor must NOT drag floor_z down
        # (that shrinks the ankle band and turns carpets into phantom blockers).
        payload = _labels_payload()
        payload.append(
            {"ins_id": "300", "label": "curtain",
             "bounding_box": _box_corners(0.1, 0.1, -0.082, 0.3, 0.3, 2.0)}
        )
        (tmp_path / "labels.json").write_text(json.dumps(payload))
        index = InteriorGSSceneSpatialIndex(tmp_path / "labels.json")
        assert index.floor_z == 0.0

    def test_scene_bounds(self, scene_dir):
        index = InteriorGSSceneSpatialIndex(
            scene_dir / "labels.json", scene_dir / "structure.json"
        )
        mins, maxs = index.scene_bounds()
        assert mins[0] <= -0.1 and maxs[0] >= 6.24


# ----------------------------- probe -----------------------------

class TestProbe:
    @pytest.fixture()
    def index(self, scene_dir):
        return InteriorGSSceneSpatialIndex(
            scene_dir / "labels.json", scene_dir / "structure.json"
        )

    def test_furniture_blocks(self, index):
        probe = build_interiorgs_probe(index)
        assert probe((2.5, 5.7, 0.79), math.pi / 2) > 0  # inside the sideboard

    def test_carpet_is_walk_over(self, index):
        probe = build_interiorgs_probe(index)
        assert probe((3.0, 3.9, 0.79), 0.0) == 0  # on the carpet, clear of the heater column

    def test_overhead_cabinet_does_not_block(self, index):
        # (2.5, 5.2): under the wall cabinet's y-span edge but clear of the sideboard.
        probe = build_interiorgs_probe(index)
        assert probe((2.5, 5.05, 0.79), math.pi / 2) == 0

    def test_wall_band_blocks(self, index):
        probe = build_interiorgs_probe(index)
        assert probe((6.12, 1.0, 0.79), 0.0) > 0

    def test_other_room_blocks_when_target_given(self, index):
        pot = index.object_by_instance("88")
        probe = build_interiorgs_probe(index, target=pot)
        assert probe((7.5, 1.0, 0.79), 0.0) > 0  # room B, target in room A
        assert probe((3.0, 3.9, 0.79), 0.0) == 0  # room A stays clear

    def test_non_finite_pose_blocked(self, index):
        probe = build_interiorgs_probe(index)
        assert probe((float("nan"), 1.0, 0.79), 0.0) == 1


# ----------------------------- instance resolution -----------------------------

class TestInstanceResolution:
    @pytest.fixture()
    def objects(self, scene_dir):
        return load_interiorgs_labels(scene_dir / "labels.json")

    def test_instance_token_resolves(self, objects):
        obj = resolve_target_by_instance(
            "Pick up pot_88 and place it in the target zone", objects
        )
        assert obj is not None and obj.id == "88"

    def test_multiword_instance_token(self, objects):
        obj = resolve_target_by_instance("Turn on bath_heater_79 and then turn it off", objects)
        assert obj is not None and obj.id == "79"

    def test_no_token_returns_none(self, objects):
        assert resolve_target_by_instance("open the door", objects) is None

    def test_unknown_id_returns_none(self, objects):
        assert resolve_target_by_instance("Pick up ghost_4242", objects) is None

    def test_id_match_with_label_drift(self, objects):
        # Wrong label prefix but a real id -> still resolves by id (tier 3).
        obj = resolve_target_by_instance("Pick up kettle_88", objects)
        assert obj is not None and obj.id == "88"


# ----------------------------- end-to-end stance + validation -----------------------------

class TestEndToEndPlacement:
    def test_pot_task_stance_is_clear_and_valid(self, scene_dir):
        index = InteriorGSSceneSpatialIndex(
            scene_dir / "labels.json", scene_dir / "structure.json"
        )
        pot = index.object_by_instance("88")
        walkable_top = index.floor_z + 0.06
        obstacles = [o for o in index.obstacle_boxes() if o.max_z() > walkable_top]
        fixtures = supporting_fixtures_for(pot, obstacles)
        assert [f.id for f in fixtures] == ["92"]
        probe = build_interiorgs_probe(
            index, target=pot, standoff_obstacles=fixtures
        )
        pose = compute_stand_pose(
            pot, probe=probe, floor_z=index.floor_z, standing_distance=0.81
        )
        assert pose.clear
        # The winning approach is -y, in FRONT of the sideboard: the lateral
        # (+x/-x) spots hug the fixture below the standoff floor and are rejected.
        assert pose.position[1] < 5.5
        assert 2.0 <= pose.position[0] <= 3.0
        verdict = validate_stand_pose(
            pose.position, pose.yaw, pot, obstacles, index.floor_z,
            standoff_obstacles=fixtures,
        )
        assert verdict.ok, verdict.failures


class TestRingScanStandPose:
    """A wall-pinned target defeats ray probing but not the annulus scan."""

    @staticmethod
    def _probe(pose, yaw):
        # Walls modeled solid to -inf so a ray cannot "step through" them (the
        # real backend's same-room rule provides this; the stub bakes it in).
        boxes = [
            (-9.0, -9.0, 0.12, 9.0),    # west wall  (x0, y0, x1, y1)
            (-9.0, -9.0, 9.0, 0.12),    # south wall
            (-1.0, 1.0, 9.0, 4.0),      # deep counter block north of the target
        ]
        hx = hy = 0.36
        px, py = pose[0], pose[1]
        return sum(
            1
            for (x0, y0, x1, y1) in boxes
            if px - hx < x1 and px + hx > x0 and py - hy < y1 and py + hy > y0
        )

    @pytest.fixture()
    def pinned_target(self):
        # Hugs the west wall, close to the south wall: every cardinal/diagonal
        # ray through its center clips a wall or the counter.
        return next(
            o for o in load_interiorgs_labels_from_payload([
                {"ins_id": "1", "label": "rice cooker",
                 "bounding_box": _box_corners(0.15, 0.15, 0.8, 0.45, 0.45, 1.1)},
            ])
        )

    def test_ray_probe_fails_ring_scan_succeeds(self, pinned_target):
        from blueprint_pipeline.scene_placement import ring_scan_stand_pose

        ray = compute_stand_pose(
            pinned_target, probe=self._probe, floor_z=0.0,
            standing_distance=0.7, include_diagonals=True,
        )
        assert not ray.clear
        ring = ring_scan_stand_pose(
            pinned_target, probe=self._probe, floor_z=0.0,
            standing_distance=0.7, max_standing_distance=1.4,
        )
        assert ring.clear
        assert self._probe(ring.position, ring.yaw) == 0
        assert ring.standoff_m >= 0.7 - 1e-9
        # Faces back at the target centroid.
        expected = math.atan2(0.3 - ring.position[1], 0.3 - ring.position[0])
        assert abs(ring.yaw - expected) < 1e-6

    def test_fully_boxed_in_reports_not_clear(self, pinned_target):
        from blueprint_pipeline.scene_placement import ring_scan_stand_pose

        ring = ring_scan_stand_pose(
            pinned_target, probe=lambda pose, yaw: 1, floor_z=0.0,
            standing_distance=0.7, max_standing_distance=1.2,
        )
        assert not ring.clear
        assert "no clear stance" in ring.notes


def load_interiorgs_labels_from_payload(payload):
    import json as _json
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        _json.dump(payload, fh)
        path = fh.name
    return load_interiorgs_labels(path)


# ----------------------------- cameras -----------------------------

class TestStanceCameras:
    @pytest.fixture()
    def pose_and_target(self, scene_dir):
        index = InteriorGSSceneSpatialIndex(scene_dir / "labels.json")
        pot = index.object_by_instance("88")
        pose = StandPose(
            position=(2.5, 4.8, 0.79), yaw=math.pi / 2, target_id="88",
            clear=True, standoff_m=0.5,
        )
        return pose, pot

    def test_camera_schema(self, pose_and_target):
        pose, pot = pose_and_target
        cams = stance_task_cameras(pose, pot, floor_z=0.0)
        assert set(cams) == {"head_pov", "third_person", "overhead", "task_focus"}
        head = cams["head_pov"]
        assert head["eye"][0] == pytest.approx(2.5)
        assert head["eye"][1] == pytest.approx(4.8)
        assert head["eye"][2] == pytest.approx(1.23)  # G1-scale default eye height
        assert head["target"] == pytest.approx((2.5, 5.7, 0.9))
        assert 0.0 < float(head["vfov"]) < math.pi  # radians
        for cam in cams.values():
            assert all(math.isfinite(float(v)) for v in (*cam["eye"], *cam["target"]))

    def test_ceiling_clamps_overhead(self, pose_and_target):
        pose, pot = pose_and_target
        cams = stance_task_cameras(pose, pot, floor_z=0.0, ceiling_z=2.6)
        assert cams["overhead"]["eye"][2] <= 2.6 - 0.15 + 1e-9

    def test_splat_render_specs_use_degrees(self, pose_and_target):
        pose, pot = pose_and_target
        cams = stance_task_cameras(pose, pot, floor_z=0.0, vfov_deg=60.0)
        specs = to_splat_render_specs(cams)
        assert len(specs) == 4
        by_id = {s["id"]: s["spec"] for s in specs}
        assert by_id["head_pov"]["fov"] == pytest.approx(60.0)
        assert by_id["head_pov"]["pos"] == pytest.approx([2.5, 4.8, 1.23])
        assert by_id["head_pov"]["up"] == [0.0, 0.0, 1.0]

    def test_link_mounted_camera_tracks_parent_translation(self):
        kwargs = {
            "parent_rotation_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "mount_translation": [0.0, -0.09, 0.1],
            "mount_forward": [-0.04, 0.83, -0.56],
        }
        first = link_mounted_camera_spec(parent_translation=[1.9, 1.15, 0.84], **kwargs)
        moved = link_mounted_camera_spec(parent_translation=[2.0, 1.35, 0.94], **kwargs)
        assert first["pos"] == pytest.approx([1.9, 1.06, 0.94])
        assert moved["pos"] == pytest.approx([2.0, 1.26, 1.04])
        assert [moved["target"][i] - first["target"][i] for i in range(3)] == pytest.approx(
            [0.1, 0.2, 0.1]
        )

    def test_link_mounted_camera_rejects_collinear_up(self):
        with pytest.raises(ValueError, match="collinear"):
            link_mounted_camera_spec(
                parent_translation=[0.0, 0.0, 0.0],
                parent_rotation_row_major=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                mount_translation=[0.0, 0.0, 0.0],
                mount_forward=[0.0, 0.0, 1.0],
                mount_up=[0.0, 0.0, 2.0],
            )


# ----------------------------- compressed PLY chunk bounds -----------------------------

class TestChunkBounds:
    def test_reads_bounds_and_floor(self, tmp_path):
        ply = write_synthetic_compressed_ply(
            tmp_path / "scene.compressed.ply",
            [
                (-1.0, -2.0, 0.0, 1.0, 2.0, 2.5),
                (0.5, 0.5, 0.02, 3.0, 3.0, 2.6),
            ],
        )
        bounds = read_compressed_ply_chunk_bounds(ply)
        assert bounds.chunk_count == 2
        assert bounds.vertex_count == 4
        aabb_min, aabb_max = bounds.aabb()
        assert list(aabb_min) == pytest.approx([-1.0, -2.0, 0.0])
        assert list(aabb_max) == pytest.approx([3.0, 3.0, 2.6])
        assert abs(bounds.floor_z_estimate()) <= 0.05

    def test_floor_mode_skips_underfloor_fuzz(self, tmp_path):
        # 3 sparse under-floor chunks, a dense pile at z=0, walls higher up:
        # the estimate must land on the pile, not the fuzz minimum.
        chunks = [(-0.9, 0, -0.6, 1, 1, 2), (-0.9, 0, -0.5, 1, 1, 2), (-0.9, 0, -0.45, 1, 1, 2)]
        chunks += [(-1, -1, 0.0, 1, 1, 2)] * 12
        chunks += [(-1, -1, 0.9, 1, 1, 2)] * 10
        ply = write_synthetic_compressed_ply(tmp_path / "scene.compressed.ply", chunks)
        est = read_compressed_ply_chunk_bounds(ply).floor_z_estimate()
        assert abs(est) <= 0.06, est

    def test_rejects_standard_ply(self, tmp_path):
        from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
        import numpy as np

        splat = SplatData(
            count=1,
            xyz=np.zeros((1, 3), dtype=np.float32),
            opacity=np.zeros(1, dtype=np.float32),
            f_dc=np.zeros((1, 3), dtype=np.float32),
            scales=np.zeros((1, 3), dtype=np.float32),
            quats=np.zeros((1, 4), dtype=np.float32),
            properties=(),
        )
        path = write_standard_3dgs_ply(splat, tmp_path / "standard.ply")
        with pytest.raises(ValueError, match="not_a_compressed_splat_ply"):
            read_compressed_ply_chunk_bounds(path)

    def test_truncated_chunk_data_raises(self, tmp_path):
        ply = write_synthetic_compressed_ply(
            tmp_path / "scene.compressed.ply", [(-1, -1, 0, 1, 1, 2)]
        )
        data = ply.read_bytes()
        ply.write_bytes(data[:-80])  # chop into the chunk floats
        with pytest.raises(ValueError, match="truncated"):
            read_compressed_ply_chunk_bounds(ply)
