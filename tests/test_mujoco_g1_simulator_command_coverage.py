from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.slow
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline import mujoco_g1_simulator_command as mg


def _write_glb(path: Path, payload: dict[str, object] | None) -> None:
    chunks = []
    if payload is not None:
        raw = json.dumps(payload).encode("utf-8")
        raw += b" " * ((4 - len(raw) % 4) % 4)
        chunks.append(struct.pack("<II", len(raw), 0x4E4F534A) + raw)
    body = b"".join(chunks)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def test_mujoco_low_level_ids_bounds_glb_and_scene_discovery(tmp_path: Path) -> None:
    assert mg._string(None) == ""
    assert mg._mapping({"a": 1}) == {"a": 1}
    assert mg._mapping(["bad"]) == {}
    assert mg._safe_id(" A/B  c! ") == "a_b_c"
    assert mg._safe_id("!!!", fallback="fallback") == "fallback"
    assert mg._float_triplet([1, "2", 3, 4]) == [1.0, 2.0, 3.0]
    assert mg._float_triplet([1, "bad", 3]) is None
    assert mg._float_triplet("1,2,3") is None
    assert mg._bounds_payload("bad") is None
    assert mg._bounds_payload([[0, 0], [1, 2, 3]]) is None
    assert mg._bounds_payload([[0, 0, 0], [0, 1, 1]]) is None
    assert mg._bounds_payload([[0, 0, 0], [1, 2, 3]]) == {
        "bounds": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        "extents": [1.0, 2.0, 3.0],
        "volume_m3_estimate": 6.0,
    }
    assert mg._xml_escape('a&b"c') == "a&amp;b&quot;c"

    scene_root = tmp_path / "capture"
    fallback_glb = scene_root / "pipeline" / "nested" / "scene.glb"
    fallback_glb.parent.mkdir(parents=True)
    fallback_glb.write_bytes(b"fallback")
    assert mg._find_scene_glb(scene_root) == fallback_glb
    preferred_glb = scene_root / "pipeline" / "worldlabs_assets" / "worldlabs_collider.glb"
    preferred_glb.parent.mkdir(parents=True, exist_ok=True)
    preferred_glb.write_bytes(b"preferred")
    assert mg._find_scene_glb(scene_root) == preferred_glb
    with pytest.raises(FileNotFoundError):
        mg._find_scene_glb(tmp_path / "empty")

    short_glb = tmp_path / "short.glb"
    short_glb.write_bytes(b"short")
    assert mg._glb_visual_summary(short_glb) == {
        "status": "unreadable",
        "reason": "file_too_short",
    }
    bad_magic = tmp_path / "bad.glb"
    bad_magic.write_bytes(b"BAD!" + b"\0" * 20)
    assert mg._glb_visual_summary(bad_magic)["reason"] == "not_binary_gltf"
    missing_json = tmp_path / "missing-json.glb"
    missing_json.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 20) + struct.pack("<II", 0, 0x004E4942)
    )
    assert mg._glb_visual_summary(missing_json)["reason"] == "missing_gltf_json_chunk"

    valid_glb = tmp_path / "valid.glb"
    _write_glb(
        valid_glb,
        {
            "materials": [{"name": "Paint"}, "skip"],
            "textures": [{"source": 0}],
            "images": [{"uri": "texture.png"}],
            "meshes": [
                "skip",
                {
                    "name": "ShelfMesh",
                    "primitives": [
                        {"material": 0, "attributes": {"POSITION": 0, "COLOR_0": 1}},
                        "skip",
                    ],
                },
            ],
            "nodes": ["skip", {"name": "ShelfNode", "mesh": 1}, {"name": ""}],
        },
    )
    summary = mg._glb_visual_summary(valid_glb)
    assert summary["status"] == "inspected"
    assert summary["materials_count"] == 2
    assert summary["textures_count"] == 1
    assert summary["images_count"] == 1
    assert summary["primitive_count"] == 1
    assert summary["has_vertex_colors"] is True
    assert summary["named_mesh_count"] == 1
    assert summary["named_node_count"] == 1


def test_mujoco_geometry_semantics_conversion_and_collision_proxies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadLen:
        def __len__(self):
            raise RuntimeError("bad len")

    assert mg._safe_len(BadLen()) == 0
    assert mg._is_generated_semantic_name("") is True
    assert mg._is_generated_semantic_name("mesh") is True
    assert mg._is_generated_semantic_name("mesh_001") is True
    assert mg._is_generated_semantic_name("Forklift") is False

    geom = SimpleNamespace(
        bounds=[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        metadata={"name": "Named Shelf"},
        visual=SimpleNamespace(material=SimpleNamespace(name="Steel")),
        vertices=[1, 2, 3],
        faces=[1],
    )
    visible = mg._geometry_visible_object(
        index=0,
        geometry_name="mesh_000",
        geometry=geom,
        gltf_mesh={"mesh_index": 4, "name": "ShelfMesh", "material_indexes": [2, "bad"]},
    )
    assert visible is not None
    assert visible["semantic_label_available"] is True
    assert visible["material_name"] == "Steel"
    assert visible["gltf_material_indexes"] == [2]
    assert mg._geometry_visible_object(
        index=1,
        geometry_name="bad",
        geometry=SimpleNamespace(bounds="bad"),
        gltf_mesh=None,
    ) is None

    scene = SimpleNamespace(geometry={"mesh_000": geom})
    semantics = mg._visual_object_semantics_summary(
        scene,
        fallback_mesh=geom,
        visual_summary={"meshes": [{"mesh_index": 0, "name": "ShelfMesh"}], "nodes": [{"name": "ShelfNode"}]},
    )
    assert semantics["status"] == "available"
    assert semantics["visible_object_count"] == 1
    missing_semantics = mg._visual_object_semantics_summary(
        SimpleNamespace(geometry={}),
        fallback_mesh=SimpleNamespace(bounds="bad", metadata={"name": "mesh_000"}),
        visual_summary={"meshes": []},
    )
    assert missing_semantics["status"] == "missing"
    assert "no_visible_geometry_objects_detected" in missing_semantics["blockers"]
    generated_only = mg._visual_object_semantics_summary(
        SimpleNamespace(geometry={}),
        fallback_mesh=SimpleNamespace(
            bounds=[[0, 0, 0], [1, 1, 1]],
            metadata={"name": "mesh_000"},
            visual=SimpleNamespace(material=None),
            vertices=[],
            faces=[],
        ),
        visual_summary={"meshes": []},
    )
    assert "visible_geometry_object_names_missing" in generated_only["blockers"]
    invalid_rgb_obj = tmp_path / "invalid-rgb.obj"
    invalid_rgb_obj.write_text("v 0 0 0 red green blue\nf 1 1 1\n", encoding="utf-8")
    invalid_rgb = mg._obj_vertex_color_summary(invalid_rgb_obj)
    assert invalid_rgb["vertex_count"] == 1
    assert invalid_rgb["vertex_rgb_count"] == 0

    class Component:
        def __init__(self, bounds):
            self.bounds = np.asarray(bounds, dtype=float)

    class BadBounds:
        @property
        def bounds(self):
            raise RuntimeError("no bounds")

    class Mesh:
        is_empty = False
        vertices = [1, 2, 3]
        faces = [1]
        bounds = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        extents = np.array([1, 1, 1], dtype=float)
        centroid = np.array([0.5, 0.5, 0.5], dtype=float)

        def __init__(self, components=None):
            self._components = components or [Component([[0, 0, 0], [1, 1, 1]])]

        def split(self, only_watertight=False):
            return self._components

        def export(self, path):
            path.write_text("v 0 0 0 0.1 0.2 0.3\nf 1 1 1\n", encoding="utf-8")

    components = [
        BadBounds(),
        Component([[0, 0, 0], [0, 1, 1]]),
        Component([[0, 0, 0], [2, 2, 0.1]]),
        Component([[0, 0, 2.5], [1, 1, 3.0]]),
        Component([[0, 0, 0], [9, 9, 3]]),
        Component([[0, 0, 0], [0.05, 0.05, 0.05]]),
        Component([[0, 0, 0], [1, 1, 1]]),
        Component([[0, 0, 0], [0.8, 0.8, 0.8]]),
    ]
    proxies, proxy_summary = mg._collision_proxy_geoms_from_mesh(Mesh(components), max_proxies=1)
    assert len(proxies) == 1
    assert proxy_summary["skipped"]["degenerate"] == 3
    assert proxy_summary["skipped"]["floor_like"] == 1
    assert proxy_summary["skipped"]["overhead"] == 1
    assert proxy_summary["skipped"]["scene_shell"] == 1
    assert proxy_summary["component_coverage"]["truncated_source_component_count"] == 1

    class RaisingSplitMesh(Mesh):
        def split(self, only_watertight=False):
            raise RuntimeError("split failed")

    fallback_proxies, fallback_summary = mg._collision_proxy_geoms_from_mesh(RaisingSplitMesh(), max_proxies=3)
    assert fallback_proxies
    assert fallback_summary["source_component_count"] == 1
    wrapper = tmp_path / "wrapper.xml"
    mg._write_mjcf_wrapper(
        tmp_path / "scene.obj",
        tmp_path / "g1.xml",
        wrapper,
        collision_proxies=[{"pos": "bad", "size": []}],
    )
    assert "blueprint_scene_collision" in wrapper.read_text(encoding="utf-8")

    class EmptyMesh(Mesh):
        is_empty = True

    class FakeScene:
        def __init__(self, mesh):
            self.geometry = {"Named Shelf": geom}
            self._mesh = mesh

        def to_geometry(self):
            return self._mesh

    valid_glb = tmp_path / "scene.glb"
    _write_glb(valid_glb, {"meshes": [{"name": "Named Shelf", "primitives": [{"attributes": {"POSITION": 0}}]}]})
    fake_trimesh = SimpleNamespace(Scene=FakeScene, load=lambda path, force=None: FakeScene(Mesh()))
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)
    converted = mg._convert_glb_to_obj(valid_glb, tmp_path / "scene.obj", collision_proxy_limit=2)
    assert converted["vertices"] == 3
    assert converted["visual_object_semantics_summary"]["status"] == "available"

    monkeypatch.setitem(sys.modules, "trimesh", SimpleNamespace(Scene=FakeScene, load=lambda path, force=None: EmptyMesh()))
    with pytest.raises(RuntimeError, match="scene mesh is empty"):
        mg._convert_glb_to_obj(valid_glb, tmp_path / "empty.obj")


def test_mujoco_asset_resolution_xml_matrix_and_route_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g1_root = tmp_path / "unitree_g1"
    (g1_root / "assets").mkdir(parents=True)
    (g1_root / "assets" / "mesh.stl").write_text("mesh", encoding="utf-8")
    (g1_root / "g1.xml").write_text(
        '<mujoco><compiler/><asset><mesh file="mesh.stl"/></asset></mujoco>',
        encoding="utf-8",
    )
    (g1_root / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
    (g1_root / "LICENSE").write_text("license", encoding="utf-8")

    monkeypatch.setattr(mg, "_git_commit", lambda path: "commit")
    source = mg._asset_source_manifest(g1_root)
    assert source["menagerie_git_commit"] == "commit"
    assert source["asset_file_count"] == 1
    assert source["checksums"]["scene.xml"]
    assert source["fixture_asset"] is False

    (g1_root / "BLUEPRINT_FIXTURE_ASSET.txt").write_text(
        "fixture only", encoding="utf-8"
    )
    fixture_source = mg._asset_source_manifest(g1_root)
    assert fixture_source["source"] == "blueprint_committed_fixture_mjcf"
    assert fixture_source["source_url"] is None
    assert fixture_source["menagerie_git_commit"] is None
    assert fixture_source["fixture_asset"] is True

    output_xml = tmp_path / "generated" / "g1.xml"
    mg._write_g1_xml_with_absolute_meshes(g1_root / "g1.xml", output_xml)
    xml = output_xml.read_text(encoding="utf-8")
    assert f'meshdir="{g1_root / "assets"}"' in xml
    assert f'file="{g1_root / "assets" / "mesh.stl"}"' in xml

    assert mg._fetch_g1_assets(menagerie_root=tmp_path, menagerie_ref="ref") == g1_root
    clone_root = tmp_path / "clone"
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        (clone_root / ".git").mkdir(parents=True, exist_ok=True)
        (clone_root / "unitree_g1").mkdir(parents=True, exist_ok=True)
        (clone_root / "unitree_g1" / "g1.xml").write_text("<mujoco/>", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(mg.subprocess, "run", fake_run)
    assert mg._fetch_g1_assets(menagerie_root=clone_root, menagerie_ref="ref") == clone_root / "unitree_g1"
    assert calls[0][:3] == ["git", "clone", "--filter=blob:none"]

    broken_root = tmp_path / "broken-clone"
    monkeypatch.setattr(
        mg.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )
    with pytest.raises(FileNotFoundError, match="missing"):
        mg._fetch_g1_assets(menagerie_root=broken_root, menagerie_ref="ref")

    monkeypatch.setattr(mg, "_repo_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    relative_root = tmp_path / "relative_g1"
    relative_root.mkdir()
    (relative_root / "g1.xml").write_text("<mujoco/>", encoding="utf-8")
    assert mg._resolve_g1_model_root(
        explicit_root="relative_g1",
        capture_root=tmp_path / "capture",
        allow_fetch=False,
        menagerie_ref="ref",
    ) == relative_root.resolve()
    monkeypatch.setenv("BLUEPRINT_MUJOCO_G1_MODEL_ROOT", str(g1_root))
    assert mg._resolve_g1_model_root(
        explicit_root=None,
        capture_root=tmp_path / "capture",
        allow_fetch=False,
        menagerie_ref="ref",
    ) == g1_root
    monkeypatch.delenv("BLUEPRINT_MUJOCO_G1_MODEL_ROOT")
    monkeypatch.setattr(mg, "_fetch_g1_assets", lambda **kwargs: g1_root)
    assert mg._resolve_g1_model_root(
        explicit_root=None,
        capture_root=tmp_path / "capture",
        allow_fetch=True,
        menagerie_ref="ref",
    ) == g1_root
    with pytest.raises(FileNotFoundError, match="missing MuJoCo Menagerie"):
        mg._resolve_g1_model_root(
            explicit_root=None,
            capture_root=tmp_path / "capture",
            allow_fetch=False,
            menagerie_ref="ref",
        )

    default_runs, default_summary = mg._matrix_runs(None)
    assert default_runs[0]["scenario_eval_run_id"] == "mujoco_g1_default_eval_run_0001"
    assert default_summary["status"] == "synthesized_default_run"
    invalid_matrix = tmp_path / "invalid.json"
    invalid_matrix.write_text("[1, 2]", encoding="utf-8")
    runs, summary = mg._matrix_runs(invalid_matrix)
    assert runs == []
    assert summary["reason"] == "matrix_payload_not_mapping"
    empty_runs_matrix = tmp_path / "empty-runs.json"
    empty_runs_matrix.write_text('{"runs": ["skip"]}', encoding="utf-8")
    assert mg._matrix_runs(empty_runs_matrix)[0] == []
    assert mg._first_matrix_run(empty_runs_matrix) == {}
    assert mg._first_matrix_run(None) == {}

    assert mg._number(True) is None
    assert mg._number("bad") is None
    assert mg._pose_triplet({"position": ["1", "2"]}) == (1.0, 2.0, 0.793)
    assert mg._pose_triplet([1, "bad"]) is None
    assert mg._nested_pose({"navigation": {"goal_pose": [3, 4, 5]}}, ("goal_pose",)) == (3.0, 4.0, 5.0)
    assert mg._scene_route_frame({"centroid": [1, 2]}) == (1.0, 2.0, 0.8, 0.8)
    assert mg._scene_route_frame({}) == (0.0, 0.0, 0.8, 0.8)
    start_only = mg._episode_navigation_spec(
        run={"spawn_pose": [1, 0, 0.8]},
        mesh_info={"bounds": [[-1, -1, 0], [1, 1, 1]]},
        index=1,
    )
    assert start_only["route_source"] == "matrix_explicit_spawn_deterministic_target"
    target_only = mg._episode_navigation_spec(
        run={"target_pose": [1, 0, 0.8]},
        mesh_info={"bounds": [[-1, -1, 0], [1, 1, 1]]},
        index=1,
    )
    assert target_only["route_source"] == "deterministic_spawn_matrix_explicit_target"
    route, source_name = mg._route_waypoints_from_run(
        run={"waypoints": [[0.5, 0, 0.8], "bad"]},
        start=[0, 0, 0.8],
        target=[1, 0, 0.8],
    )
    assert source_name == "matrix_waypoints"
    assert route == [(0.0, 0.0, 0.8), (0.5, 0.0, 0.8), (1.0, 0.0, 0.8)]
    assert mg._route_waypoints_from_run(run={}, start=[1, 9, 0.8], target=[2, -9, 0.8])[1].startswith(
        "deterministic_warehouse"
    )
    assert mg._interpolate_route([[0, 0, 0], [1, 0, 0]], 2.0) == ((1.0, 0.0, 0.0), 0.0, 0)
    endpoint, _yaw, segment_index = mg._interpolate_route(
        [[0, 0, 0], [float("nan"), 0, 0], [2, 1, 0]],
        0.5,
    )
    assert endpoint == (2.0, 1.0, 0.0)
    assert segment_index == 1
    assert mg._render_episode_indexes(5, 3) == {0, 2, 4}
    assert mg._render_capture_steps(0) == {0}


def test_mujoco_task_preview_contact_video_and_manifest_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert mg._action_pose({"pose": "bad"}, "pose") is None
    assert mg._action_pose({"pose": [1, "bad", 3]}, "pose") is None
    success = mg._attempt_task_outcome(
        actions=[
            {"root_position": [0, 0, 0.8], "desired_root_position": [0, 0, 0.8]},
            {"root_position": [1, 0, 0.8], "desired_root_position": [1, 0, 0.8]},
        ],
        start=[0, 0, 0.8],
        target=[1, 0, 0.8],
        route_distance_m=1.0,
        collision_summary={},
        bounded_steps=2,
        model_timestep_s=0.01,
    )
    assert success["task_success"] is True
    failed = mg._attempt_task_outcome(
        actions=[
            {"root_position": [0, 0, 0.3], "desired_root_position": [0, 0, 0.8], "policy_action": "stopped_by_collision_probe"},
            {"root_position": [0, 0, 0.3], "desired_root_position": [1, 0, 0.8], "policy_action": "stopped_by_collision_probe"},
        ],
        start=[0, 0, 0.8],
        target=[1, 0, 0.8],
        route_distance_m=0.0,
        collision_summary={
            "robot_scene_contact_event_count": 1,
            "rejected_scene_collision_probe_count": 1,
            "collision_response_event_count": 1,
            "min_clearance_m": 0.01,
            "clearance_threshold_m": 0.15,
        },
        bounded_steps=2,
        model_timestep_s=0.0,
    )
    assert "failure_scene_collision_contact" in failed["failure_mode_ids"]
    assert "failure_robot_fall_detected" in failed["failure_mode_ids"]
    assert failed["path_efficiency_ratio"] is None

    assert mg._interpolate_route([], 0.5) == ((0.0, 0.0, 0.793), 0.0, 0)
    assert mg._interpolate_route([[1, 2, 3]], 0.5) == ((1.0, 2.0, 3.0), 0.0, 0)
    assert mg._interpolate_route([[0, 0, 0], [0, 0, 0]], 0.5) == ((0.0, 0.0, 0.0), 0.0, 1)
    assert mg._interpolate_route([[0, 0, 0], [0, 0, 0], [1, 0, 0]], 0.5)[2] == 1
    assert mg._yaw_quaternion(0.0) == [1.0, 0.0, 0.0, 0.0]

    class FakeMujoco:
        class mjtObj:
            mjOBJ_JOINT = "joint"
            mjOBJ_GEOM = "geom"
            mjOBJ_BODY = "body"

        @staticmethod
        def mj_name2id(model, obj_type, name):
            return model.joints.get(name, -1)

        @staticmethod
        def mj_forward(model, data):
            data.forwarded = True

        @staticmethod
        def mj_id2name(model, obj_type, identifier):
            if obj_type == "geom":
                if identifier == 9:
                    raise RuntimeError("bad geom")
                return model.geom_names.get(identifier)
            if obj_type == "body":
                if identifier == 9:
                    raise RuntimeError("bad body")
                return model.body_names.get(identifier)
            return None

        @staticmethod
        def mj_contactForce(model, data, index, force):
            if index == 1:
                raise RuntimeError("force failed")
            force[:] = [1, 2, 3, 4, 5, 6]

    model = SimpleNamespace(
        joints={"left_hip_pitch_joint": 0, "left_knee_joint": 1},
        jnt_qposadr=[7, 8],
        geom_bodyid=[0, 1, 9],
        geom_names={0: "blueprint_scene_collision", 1: "floor"},
        body_names={0: "world", 1: "pelvis"},
    )
    addresses = mg._g1_preview_joint_addresses(model, FakeMujoco)
    assert addresses == {"left_hip_pitch_joint": 7, "left_knee_joint": 8}
    qpos = np.zeros(10)
    base_qpos = np.ones(10)
    mg._apply_preview_gait_pose(qpos=qpos, base_qpos=base_qpos, joint_addresses=addresses, phase=1.0, moving=False)
    assert np.all(qpos == 0)
    mg._apply_preview_gait_pose(qpos=qpos, base_qpos=base_qpos, joint_addresses={**addresses, "right_elbow_joint": 999}, phase=1.0, moving=True)
    assert qpos[7] != 0
    data = SimpleNamespace(qpos=np.zeros(10), qvel=np.ones(10))
    mg._set_preview_pose(
        data=data,
        base_qpos=base_qpos,
        root_qpos=0,
        pose=[1, 2, 3],
        yaw=0,
        joint_addresses=addresses,
        phase=1.0,
        moving=True,
    )
    assert data.qpos[:3].tolist() == [1.0, 2.0, 3.0]
    assert np.all(data.qvel == 0)
    assert len(mg._candidate_pose_specs(desired_pose=[0, 0, 0.8], previous_pose=None, yaw=0)) == 47
    assert mg._candidate_pose_specs(desired_pose=[0, 0, 0.8], previous_pose=[1, 1, 0.8], yaw=0)[-1][
        "candidate_kind"
    ] == "stop"

    monkeypatch.setattr(mg, "_contact_records", lambda *args: [{"scene_collision_contact": False}])
    candidate = mg._evaluate_preview_candidate(
        model=model,
        data=data,
        mujoco_module=FakeMujoco,
        base_qpos=base_qpos,
        root_qpos=0,
        joint_addresses=addresses,
        candidate={"candidate_kind": "direct", "pose": [0, 0, 0.8]},
        yaw=0.5,
        phase=1.0,
        moving=True,
    )
    assert candidate["accepted"] is True

    qpos = np.zeros(40)
    base_qpos = np.ones(40)
    arm_addresses = {
        "left_shoulder_pitch_joint": 22,
        "left_elbow_joint": 25,
        "left_wrist_pitch_joint": 27,
        "right_shoulder_pitch_joint": 29,
        "right_elbow_joint": 32,
        "right_wrist_pitch_joint": 34,
    }
    data_with_arms = SimpleNamespace(qpos=qpos, qvel=np.ones(40))
    mg._set_preview_pose(
        data=data_with_arms,
        base_qpos=base_qpos,
        root_qpos=0,
        pose=[1, 2, 0.79],
        yaw=0.0,
        joint_addresses=arm_addresses,
        phase=0.0,
        moving=False,
        manipulation_ready_arms=True,
        manipulation_reach_arm="both",
    )
    assert data_with_arms.qpos[22] == pytest.approx(base_qpos[22] - 0.85)
    assert data_with_arms.qpos[25] == pytest.approx(base_qpos[25] - 0.23)
    assert data_with_arms.qpos[29] == pytest.approx(base_qpos[29] - 0.85)
    assert data_with_arms.qpos[32] == pytest.approx(base_qpos[32] - 0.23)
    assert np.all(data_with_arms.qvel == 0)

    right_only = np.zeros(40)
    applied = mg._apply_manipulation_ready_arm_pose(
        qpos=right_only,
        base_qpos=base_qpos,
        joint_addresses=arm_addresses,
        arm="right",
    )
    assert "right_shoulder_pitch_joint" in applied
    assert "left_shoulder_pitch_joint" not in applied
    assert right_only[29] == pytest.approx(base_qpos[29] - 0.85)
    assert right_only[22] == 0.0

    camera = SimpleNamespace(lookat=[0.0, 0.0, 0.0], distance=0.0, azimuth=0.0, elevation=0.0)
    selected = mg._configure_robot_pov_camera(
        camera,
        pose=[1, 2, 0.79],
        yaw=0.0,
        manipulation_ready_arms=True,
    )
    assert selected["camera_mode"] == "virtual_manipulation_pov_near_head_aimed_at_workspace"
    assert selected["azimuth"] == pytest.approx(0.0)
    assert camera.lookat[0] > 1.0
    assert camera.distance < 1.0
    assert mg._is_robot_pov_self_occluding_body_name("torso_link") is True
    assert mg._is_robot_pov_self_occluding_body_name("left_wrist_yaw_link") is False
    fake_model = SimpleNamespace(geom_rgba=np.ones((3, 4)))
    previous_alpha = mg._set_geom_alpha(fake_model, [0, 2], 0.0)
    assert fake_model.geom_rgba[0, 3] == 0.0
    assert fake_model.geom_rgba[1, 3] == 1.0
    mg._restore_geom_alpha(fake_model, previous_alpha)
    assert np.all(fake_model.geom_rgba[:, 3] == 1.0)

    with pytest.raises(ValueError):
        mg._manipulation_ready_arm_joint_deltas("center")

    monkeypatch.undo()
    data = SimpleNamespace(
        ncon=3,
        contact=[
            SimpleNamespace(geom1=0, geom2=1, dist=-0.1, pos=[1, 2, 3]),
            SimpleNamespace(geom1=9, geom2=2, dist=0.2, pos="bad"),
        ],
    )
    contacts = mg._contact_records(model, data, FakeMujoco)
    assert contacts[0]["scene_collision_contact"] is True
    assert contacts[0]["contact_force_6d"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert contacts[1]["contact_force_6d"] is None
    assert contacts[1]["position_xyz"] == []
    assert mg._is_scene_collision_contact({"geom_names": "bad"}) is False

    assert mg._render_episode_indexes(0, 1) == set()
    assert mg._render_episode_indexes(5, 1) == {0}
    assert mg._camera_record_with_time("overview", tmp_path / "f.png", 1, "mode", sim_time_s=None)["sim_time_s"] is None
    frames = [
        {"camera": "overview", "path": "overview.png", "sim_time_s": "0.1"},
        {"camera": "bad", "path": "bad.png", "sim_time_s": "bad"},
        {"camera": "side", "path": "", "sim_time_s": None},
    ]
    assert mg._frame_groups(frames)["overview"] == ["overview.png"]
    assert mg._frame_time_groups(frames)["overview"] == [0.1]
    assert mg._rendered_array_scene_score(np.zeros((2, 2), dtype=np.uint8)) == 0.0
    assert mg._rendered_array_scene_score(np.full((20, 20, 3), 255, dtype=np.uint8)) == 0.0
    colorful = np.zeros((20, 20, 3), dtype=np.uint8)
    colorful[:, :, 0] = np.arange(20, dtype=np.uint8)[:, None] * 10
    colorful[:, :, 1] = np.arange(20, dtype=np.uint8)[None, :] * 10
    assert mg._rendered_array_nonblank(colorful) is True

    assert mg._write_frame_video(
        camera="overview",
        frame_paths=["one.png"],
        frame_times_s=[],
        output_root=tmp_path,
        fallback_frame_duration_s=0.1,
    )["reason"] == "requires_at_least_two_frames"
    monkeypatch.setattr(mg.shutil, "which", lambda name: None)
    assert mg._write_frame_video(
        camera="overview",
        frame_paths=["one.png", "two.png"],
        frame_times_s=[],
        output_root=tmp_path,
        fallback_frame_duration_s=0.1,
    )["reason"] == "ffmpeg_unavailable"
    monkeypatch.setattr(mg.shutil, "which", lambda name: "/fake/ffmpeg")
    monkeypatch.setattr(
        mg.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, stdout="", stderr="failure"),
    )
    assert mg._write_frame_video(
        camera="overview",
        frame_paths=["one.png", "two.png"],
        frame_times_s=[0.0, None],
        output_root=tmp_path,
        fallback_frame_duration_s=0.0,
    )["reason"] == "ffmpeg_failed"

    def fake_ffmpeg(command, **kwargs):
        Path(command[-1]).write_bytes(b"video")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(mg.subprocess, "run", fake_ffmpeg)
    video = mg._write_frame_video(
        camera="overview",
        frame_paths=["one'quote.png", "two.png", "three.png"],
        frame_times_s=[0.0, 0.2],
        output_root=tmp_path,
        fallback_frame_duration_s=0.1,
    )
    assert video["status"] == "complete"
    assert video["frame_durations_s"] == [0.2, 0.2, 0.2]
    white = tmp_path / "white.png"
    Image.new("RGB", (32, 32), "white").save(white)
    monkeypatch.setattr(mg.shutil, "which", lambda name: None)
    visual = mg._visual_artifact_summary(
        frames=[{"camera": "overview", "path": str(white), "sim_time_s": 0.0}],
        output_root=tmp_path,
        mesh_info={"obj_vertex_color_summary": {"has_vertex_rgb": False}, "visual_asset_summary": {}},
        model_timestep_s=0.01,
    )
    assert "texture_material_evidence:vertex_rgb_not_detected" in visual["limitations"]
    assert "blank_scene_checks:one_or_more_frames_blank" in visual["limitations"]


def test_mujoco_package_coverage_closure_and_cli_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    present = tmp_path / "present.txt"
    present.write_text("present", encoding="utf-8")
    assert mg._file_artifact(present, base_dir=tmp_path)["present"] is True
    assert mg._file_artifact(tmp_path / "missing.txt", base_dir=tmp_path, required=False)["required"] is False
    monkeypatch.setattr(mg.os.path, "relpath", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad relpath")))
    assert mg._relative_path(tmp_path, present) == str(present)
    monkeypatch.setattr(mg.os.path, "relpath", __import__("os").path.relpath)
    rows_path = tmp_path / "rows.jsonl"
    mg._write_jsonl(rows_path, [{"b": 2, "a": 1}])
    assert rows_path.read_text(encoding="utf-8").strip() == '{"a": 1, "b": 2}'

    visual = mg._write_visual_media_coverage_manifest(
        output_root=tmp_path,
        generated_at="2026-06-20T00:00:00Z",
        required_scenario_eval_run_ids=["run-a", "run-b"],
        visual_artifacts={
            "frames": [
                {"scenario_eval_run_id": "run-a", "camera": "overview", "path": "overview.png"},
                {"scenario_eval_run_id": "run-a", "camera": "sim_robot_follow_pov", "path": "pov.png"},
                {"scenario_eval_run_id": "extra", "camera": "side", "path": "side.png"},
                {"camera": "", "path": "skip.png"},
            ],
            "overview_video": {"status": "complete", "path": "overview.mp4"},
            "robot_pov_video": {"status": "not_generated"},
            "side_video": {"status": "complete", "path": "side.mp4"},
            "limitations": ["limited"],
        },
    )
    assert visual["status"] == "incomplete"
    assert visual["extra_rendered_scenario_eval_run_ids"] == ["extra"]
    assert visual["rows"][0]["robot_pov_frames_present"] is True

    complete_metrics = {key: 1 for key in mg.REQUIRED_TASK_METRIC_KEYS}
    coverage = mg._metric_coverage(
        [
            {"attempt_id": "a", "metrics": complete_metrics, "success": True, "task_success": True},
            {"attempt_id": "b", "metrics": {}, "failure_mode_ids": ["failure"]},
        ]
    )
    assert coverage["missing_metric_row_count"] == 1
    assert mg._sequence3([1, "bad", 3]) is None
    assert mg._mesh_bounds_summary({"bounds": [[0, 0, 0], [1, 2, 3]]})["volume_m3_estimate"] == 6.0
    assert mg._mesh_bounds_summary({"extents": [1, 0, 1]})["positive_extents"] is False
    assert mg._bounds_payload(np.array([[0, 0, 0], [1, 2, 3]], dtype=float)) == {
        "bounds": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        "extents": [1.0, 2.0, 3.0],
        "volume_m3_estimate": 6.0,
    }
    assert mg._int_set(["1", "bad", 2]) == {1, 2}
    assert mg._visual_object_physics_coverage(
        object_semantics_summary={},
        collision_summary={},
        proxy_summary={},
        collider_loaded=False,
    )["reason"] == "visible_object_semantics_missing"
    full_mesh = mg._visual_object_physics_coverage(
        object_semantics_summary={"visible_objects": [{"object_id": "obj", "source_component_index": "bad"}]},
        collision_summary={"scene_collision_mesh_geom_enabled": True},
        proxy_summary={},
        collider_loaded=True,
    )
    assert full_mesh["coverage_method"] == "full_scene_collision_mesh_geom"
    proxy_missing = mg._visual_object_physics_coverage(
        object_semantics_summary={"visible_objects": [{"object_id": "obj", "source_component_index": "bad"}]},
        collision_summary={"scene_collision_proxy_geoms_enabled": True},
        proxy_summary={"component_coverage": {"covered_source_component_indexes": []}},
        collider_loaded=True,
    )
    assert proxy_missing["unmapped_visible_object_ids"] == ["obj"]
    qa_missing = mg._build_digital_twin_fidelity_qa(
        generated_at="2026-06-20T00:00:00Z",
        mesh_info={},
        collision_summary={},
        visual_artifacts={"blank_scene_checks": {"status": "checked", "all_frames_nonblank": False}},
    )
    assert "digital_twin_scale_bounds_missing" in qa_missing["blockers"]
    assert "digital_twin_texture_material_truth_missing" in qa_missing["blockers"]
    assert "digital_twin_collider_coverage_missing" in qa_missing["blockers"]

    linux_matrix = tmp_path / "linux-empty-matrix.json"
    linux_matrix.write_text('{"runs": []}', encoding="utf-8")
    monkeypatch.setattr(mg.platform, "system", lambda: "Linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    with pytest.raises(RuntimeError, match="contains no executable runs"):
        mg.run_mujoco_g1_simulator_command(
            capture_root=tmp_path,
            scenario_eval_matrix_path=linux_matrix,
            render_frames=False,
        )
    # No frames rendered -> GL must be disabled so `import mujoco` works on GL-less hosts.
    assert mg.os.environ["MUJOCO_GL"] == "disable"
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    with pytest.raises(RuntimeError, match="contains no executable runs"):
        mg.run_mujoco_g1_simulator_command(
            capture_root=tmp_path,
            scenario_eval_matrix_path=linux_matrix,
            render_frames=True,
        )
    # Rendering on headless Linux workers keeps the EGL default.
    assert mg.os.environ["MUJOCO_GL"] == "egl"
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    with pytest.raises(RuntimeError, match="contains no executable runs"):
        mg.run_mujoco_g1_simulator_command(
            capture_root=tmp_path,
            scenario_eval_matrix_path=linux_matrix,
            render_frames=False,
        )
    # An explicit operator choice is never overridden.
    assert mg.os.environ["MUJOCO_GL"] == "osmesa"


def test_mujoco_runtime_render_and_error_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeOpt:
        timestep = 0.01

    class FakeModel:
        opt = FakeOpt()
        jnt_qposadr = np.array([0])
        key_qpos = np.array([[0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0]])
        qpos0 = np.array([0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0])

        @classmethod
        def from_xml_path(cls, _path: str) -> "FakeModel":
            return cls()

    class FakeData:
        def __init__(self, _model: FakeModel) -> None:
            self.qpos = np.zeros(7)
            self.qvel = np.zeros(7)
            self.time = 0.0

    class FakeCamera:
        def __init__(self) -> None:
            self.type = None
            self.lookat = np.zeros(3)
            self.distance = 0.0
            self.azimuth = 0.0
            self.elevation = 0.0

    def patterned_image(height: int = 360, width: int = 640) -> np.ndarray:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(height, dtype=np.uint16)[:, None] % 255
        image[:, :, 1] = np.arange(width, dtype=np.uint16) % 255
        image[:, :, 2] = 120
        return image

    class FakeRenderer:
        def __init__(self, _model: FakeModel, *, height: int, width: int) -> None:
            self.height = height
            self.width = width
            self.camera = None
            self.side_option_counts: dict[tuple[float, float, float], int] = {}

        def update_scene(self, _data: FakeData, camera: object) -> None:
            self.camera = camera

        def render(self) -> np.ndarray:
            if self.camera == "overview":
                return patterned_image(self.height, self.width)
            if isinstance(self.camera, FakeCamera):
                key = (
                    round(float(self.camera.azimuth), 3),
                    round(float(self.camera.distance), 3),
                    round(float(self.camera.elevation), 3),
                )
                self.side_option_counts[key] = self.side_option_counts.get(key, 0) + 1
                if round(float(self.camera.distance), 2) in {2.15, 2.35}:
                    return np.zeros((self.height, self.width, 3), dtype=np.uint8)
                if key == (45.0, 2.8, -12.0) and self.side_option_counts[key] == 2:
                    return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return patterned_image(self.height, self.width)

        def close(self) -> None:
            return None

    def fake_name_to_id(_model: FakeModel, object_type: object, name: str) -> int:
        if name in {"floating_base_joint", "stand"}:
            return 0
        return -1

    def fake_step(model: FakeModel, data: FakeData) -> None:
        data.time += model.opt.timestep

    fake_mujoco = SimpleNamespace(
        __version__="fake-render",
        MjModel=FakeModel,
        MjData=FakeData,
        MjvCamera=FakeCamera,
        Renderer=FakeRenderer,
        mjtObj=SimpleNamespace(mjOBJ_JOINT=1, mjOBJ_KEY=2),
        mjtCamera=SimpleNamespace(mjCAMERA_FREE=1),
        mj_name2id=fake_name_to_id,
        mj_forward=lambda _model, _data: None,
        mj_step=fake_step,
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    capture_root = tmp_path / "capture"
    scene_glb = capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_collider.glb"
    scene_glb.parent.mkdir(parents=True)
    scene_glb.write_bytes(b"fake glb")
    g1_root = tmp_path / "unitree_g1"
    (g1_root / "assets").mkdir(parents=True)
    (g1_root / "g1.xml").write_text("<mujoco><worldbody/></mujoco>", encoding="utf-8")

    converted_modes: list[str] = []

    def fake_convert(
        _glb_path: Path,
        obj_path: Path,
        *,
        collision_proxy_mode: str = "aabb",
    ) -> dict[str, object]:
        converted_modes.append(collision_proxy_mode)
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        obj_path.write_text("v 0 0 0 0.1 0.2 0.3\n", encoding="utf-8")
        return {
            "source_glb": str(scene_glb),
            "converted_obj": str(obj_path),
            "vertices": 1,
            "faces": 0,
            "bounds": [[-2.0, -1.0, 0.0], [2.0, 1.0, 1.0]],
            "extents": [4.0, 2.0, 1.0],
            "centroid": [0.0, 0.0, 0.5],
            "visual_asset_summary": {
                "materials_count": 1,
                "textures_count": 0,
                "images_count": 0,
                "has_vertex_colors": True,
            },
            "obj_vertex_color_summary": {
                "has_vertex_rgb": True,
                "vertex_rgb_fraction": 1.0,
            },
            "visual_object_semantics_summary": {
                "status": "available",
                "visible_object_count": 1,
                "named_visible_object_count": 1,
                "visible_objects": [
                    {
                        "object_id": "visible_object_0000_proxy",
                        "source_component_index": 0,
                        "name": "proxy",
                    }
                ],
            },
            "collision_proxy_geoms": [
                {"name": "proxy", "pos": [10.0, 10.0, 0.5], "size": [0.1, 0.1, 0.5]}
            ],
            "collision_proxy_summary": {
                "source_component_count": 1,
                "proxy_count": 1,
                "max_proxy_count": 160,
                "component_coverage": {
                    "covered_source_component_indexes": [0],
                    "reference_floor_covered_source_component_indexes": [],
                    "truncated_source_component_indexes": [],
                    "truncated_source_component_count": 0,
                    "uncovered_source_component_indexes": [],
                    "uncovered_source_component_count": 0,
                    "component_proxy_coverage_complete": True,
                },
            },
            "mujoco_visual_fidelity_boundary": "test boundary",
        }

    monkeypatch.setattr(mg, "_convert_glb_to_obj", fake_convert)
    monkeypatch.setattr(
        mg,
        "_write_frame_video",
        lambda **kwargs: {
            "status": "complete",
            "path": str(kwargs["output_root"] / f"{kwargs['camera']}.mp4"),
            "frame_count": len(kwargs["frame_paths"]),
            "realtime_timing_from_sim_time": True,
        },
    )

    matrix_path = tmp_path / "matrix.json"
    matrix_rows = [
        {
            "scenario_eval_run_id": f"run-{index}",
            "episode_id": f"episode-{index}",
            "task_id": "walk_to_target",
            "scenario_id": f"scenario-{index}",
            "concrete_mutation": {
                "spawn_pose": [-1.0, 0.0, 0.793],
                "target_pose": [1.0, 0.0, 0.793],
            },
        }
        for index in (1, 2)
    ]
    matrix_path.write_text(
        json.dumps({"scenario_eval_run_count": 2, "runs": matrix_rows}),
        encoding="utf-8",
    )

    payload = mg.run_mujoco_g1_simulator_command(
        capture_root=capture_root,
        g1_model_root=g1_root,
        output_dir=tmp_path / "rendered-multi",
        scenario_eval_matrix_path=matrix_path,
        duration_seconds=0.02,
        render_frames=True,
        render_every_step=True,
        max_rendered_episodes=2,
        max_rendered_steps=2,
    )
    assert payload["step_count_source"] == "duration_seconds_and_model_timestep"
    assert converted_modes == ["aabb"]
    assert payload["rendered_episode_count"] == 2
    assert payload["scene_collision_proxy_geoms_enabled"] is True
    assert "collision_dynamics_not_validated" in payload["robot_team_handoff_blockers"]
    assert "visible_scene_collision_alignment_not_validated" in payload["robot_team_handoff_blockers"]
    robot_frames = [
        frame for frame in payload["visual_artifacts"]["frames"] if frame["camera"] == "sim_robot_follow_pov"
    ]
    side_frames = [frame for frame in payload["visual_artifacts"]["frames"] if frame["camera"] == "side"]
    assert any(frame["robot_camera_selected"]["fallback_used"] for frame in robot_frames)
    assert any(frame["side_camera_selected"].get("fallback_reason") == "reused_side_camera_frame_blank" for frame in side_frames)

    def floor_contacts(_model, _data, _mujoco):
        return [
            {
                "geom_names": ["blueprint_reference_floor", "foot"],
                "reference_floor_contact": True,
                "scene_collision_contact": False,
            }
        ]

    monkeypatch.setattr(mg, "_contact_records", floor_contacts)
    single_matrix = tmp_path / "single-matrix.json"
    single_matrix.write_text(
        json.dumps({"scenario_eval_run_count": 1, "runs": [matrix_rows[0]]}),
        encoding="utf-8",
    )
    single = mg.run_mujoco_g1_simulator_command(
        capture_root=capture_root,
        g1_model_root=g1_root,
        output_dir=tmp_path / "rendered-single",
        scenario_eval_matrix_path=single_matrix,
        steps=2,
        render_frames=True,
        render_every_step=True,
        max_rendered_episodes=1,
        max_rendered_steps=2,
    )
    assert single["committed_contact_sample_count"] > 0
    assert any("overview_" in Path(frame["path"]).name for frame in single["visual_artifacts"]["frames"])

    monkeypatch.setattr(
        mg,
        "_evaluate_preview_candidate",
        lambda **kwargs: {
            "candidate_kind": "direct",
            "pose": (0.0, 0.0, 0.793),
            "yaw": 0.0,
            "phase": 0.0,
            "moving": False,
            "contacts": [],
            "contact_count": 0,
            "scene_collision_contact_count": 0,
            "accepted": True,
        },
    )
    monkeypatch.setattr(
        mg,
        "_contact_records",
        lambda *_args: [{"scene_collision_contact": True, "geom_names": ["blueprint_scene_collision"]}],
    )
    with pytest.raises(RuntimeError, match="committed a scene-colliding pose"):
        mg.run_mujoco_g1_simulator_command(
            capture_root=capture_root,
            g1_model_root=g1_root,
            output_dir=tmp_path / "committed-collision",
            scenario_eval_matrix_path=single_matrix,
            steps=1,
            render_frames=False,
        )

    fake_mujoco_missing_root = SimpleNamespace(
        **{
            **fake_mujoco.__dict__,
            "mj_name2id": lambda _model, _object_type, _name: -1,
        }
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco_missing_root)
    with pytest.raises(RuntimeError, match="floating_base_joint"):
        mg.run_mujoco_g1_simulator_command(
            capture_root=capture_root,
            g1_model_root=g1_root,
            output_dir=tmp_path / "missing-root",
            scenario_eval_matrix_path=single_matrix,
            steps=1,
            render_frames=False,
        )

    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setattr(mg, "_contact_records", lambda *_args: [])
    monkeypatch.setattr(
        mg,
        "_attempt_task_outcome",
        lambda **kwargs: {
            "task_success": False,
            "task_status": "failed_task_criteria",
            "failure_mode_ids": "not-a-list",
            "failure_reason": "bad",
            "goal_reached": False,
            "endpoint_clean": False,
            "spawn_clean": True,
            "timeout": True,
            "fall_detected": False,
            "stuck_detected": False,
            "policy_instability_detected": False,
            "final_pose": [0, 0, 0.793],
            "final_target_error_m": 1.0,
            "goal_tolerance_m": 0.25,
            "min_clearance_m": 0.15,
            "clearance_threshold_m": 0.15,
            "clearance_threshold_violation": False,
            "actual_path_distance_m": 0.0,
            "path_efficiency_ratio": None,
            "progress_to_goal_m": 0.0,
            "progress_to_goal_ratio": 0.0,
            "max_path_deviation_m": 0.0,
            "mean_path_deviation_m": 0.0,
            "min_root_height_m": 0.793,
            "near_miss_event_count": 0,
            "collision_response_event_count": 0,
            "robot_scene_contact_event_count": 0,
            "success_criteria": {},
        },
    )
    weird_failure = mg.run_mujoco_g1_simulator_command(
        capture_root=capture_root,
        g1_model_root=g1_root,
        output_dir=tmp_path / "weird-failure",
        scenario_eval_matrix_path=single_matrix,
        steps=1,
        render_frames=False,
    )
    assert weird_failure["task_success_summary"]["failure_mode_counts"] == {}

    monkeypatch.setattr(
        mg,
        "_contact_records",
        lambda *_args: [{"scene_collision_contact": True, "geom_names": ["blueprint_scene_collision"]}],
    )
    monkeypatch.setattr(mg, "_scene_collision_contact_count", lambda _records: 0)
    collision_avoidance_blocked = mg.run_mujoco_g1_simulator_command(
        capture_root=capture_root,
        g1_model_root=g1_root,
        output_dir=tmp_path / "collision-avoidance-blocked",
        scenario_eval_matrix_path=single_matrix,
        steps=1,
        render_frames=False,
    )
    assert "collision_avoidance_not_validated" in collision_avoidance_blocked["robot_team_handoff_blockers"]

    complete_metrics = {key: 1 for key in mg.REQUIRED_TASK_METRIC_KEYS}
    present = tmp_path / "present-runtime.txt"
    present.write_text("present", encoding="utf-8")
    attempt = {
        "attempt_id": "attempt-fail",
        "episode_id": "episode-fail",
        "scenario_eval_run_id": "run-a",
        "scenario_variation_instance_id": "variation-a",
        "variation_name": "variation",
        "task_id": "walk_to_target",
        "scenario_id": "scenario",
        "status": "failed_task_criteria",
        "success": False,
        "task_success": False,
        "failure_mode_ids": ["failure_target_not_reached"],
        "task_outcome": {
            "task_success": False,
            "task_status": "failed_task_criteria",
            "failure_mode_ids": ["failure_target_not_reached"],
            "failure_reason": "failure_target_not_reached",
            "goal_reached": False,
            "endpoint_clean": False,
            "timeout": True,
            "fall_detected": False,
            "stuck_detected": True,
            "policy_instability_detected": False,
            "clearance_threshold_violation": False,
            "final_target_error_m": 1.0,
            "goal_tolerance_m": 0.25,
            "min_clearance_m": 0.15,
            "clearance_threshold_m": 0.15,
            "near_miss_event_count": 0,
            "collision_response_event_count": 0,
            "robot_scene_contact_event_count": 0,
            "progress_to_goal_ratio": 0.0,
            "path_efficiency_ratio": None,
            "cycle_time_seconds": 0.1,
        },
        "metrics": complete_metrics,
        "actions": [{"policy_action": "stopped_by_collision_probe"}],
        "route_waypoints": [{"x": 0}],
        "artifact_paths": {
            "scene_trace": "scene.json",
            "frames": ["frame1.png", "frame2.png"],
            "overview_video": "overview.mp4",
        },
    }
    trace_package = mg._write_mujoco_batch_trace_package(
        output_root=tmp_path,
        generated_at="2026-06-20T00:00:00Z",
        attempts=[attempt],
        full_contact_trace=[{"geom_names": ["floor"]}],
        full_collision_probe_trace=[{"geom_names": ["blueprint_collision_proxy_000"]}],
        full_collision_response_events=[{"event_type": "motion_stopped_by_collision_probe"}],
        required_scenario_eval_run_ids=["run-a"],
        covered_scenario_eval_run_ids=["run-a"],
        missing_scenario_eval_run_ids=[],
        duplicate_scenario_eval_run_ids=[],
        scenario_eval_run_coverage_complete=True,
        visual_artifacts={
            "frames": [
                {"scenario_eval_run_id": "run-a", "camera": "overview", "path": "overview.png"},
                {"scenario_eval_run_id": "run-a", "camera": "sim_robot_follow_pov", "path": "pov.png"},
                {"scenario_eval_run_id": "run-a", "camera": "side", "path": "side.png"},
            ],
            "overview_video": {"status": "complete", "path": "overview.mp4"},
            "robot_pov_video": {"status": "complete", "path": "pov.mp4"},
            "side_video": {"status": "complete", "path": "side.mp4"},
            "limitations": [],
        },
    )
    assert trace_package["manifest"]["status"] == "completed"
    assert trace_package["failure_labels"]["label_count"] == 1
    assert trace_package["visual_review_ledger"]["records"][0]["media_evidence_present"] is True

    closure_blocked = mg._build_mujoco_batch_closure_manifest(
        output_root=tmp_path,
        generated_at="2026-06-20T00:00:00Z",
        attempts=[attempt],
        required_scenario_eval_run_ids=["run-a", "run-b"],
        covered_scenario_eval_run_ids=["run-a"],
        missing_scenario_eval_run_ids=["run-b"],
        duplicate_scenario_eval_run_ids=["dupe"],
        attempt_count_matches_matrix_count=False,
        scenario_eval_run_id_coverage_exact=False,
        scenario_eval_run_coverage_complete=False,
        batch_trace_package={"manifest": {}, "metrics": {}, "failure_labels": {}, "visual_review_ledger": {}},
        support_artifacts={"present": present, "missing": tmp_path / "missing.txt"},
        visual_artifacts={"frames": [], "overview_video": {}, "robot_pov_video": {}, "side_video": {}},
        collision_summary={"collision_dynamics_validated": False},
        digital_twin_fidelity_qa={"robot_team_grade_fidelity_passed": False, "blockers": ["qa"]},
        robot_team_handoff_blockers=["handoff"],
        claim_boundary={"sim": True},
    )
    assert closure_blocked["status"] == "blocked"
    assert "scenario_eval_run_coverage_incomplete" in closure_blocked["blockers"]
    assert "qa" in closure_blocked["robot_team_grade_blockers"]

    closure_complete = mg._build_mujoco_batch_closure_manifest(
        output_root=tmp_path,
        generated_at="2026-06-20T00:00:00Z",
        attempts=[attempt],
        required_scenario_eval_run_ids=["run-a"],
        covered_scenario_eval_run_ids=["run-a"],
        missing_scenario_eval_run_ids=[],
        duplicate_scenario_eval_run_ids=[],
        attempt_count_matches_matrix_count=True,
        scenario_eval_run_id_coverage_exact=True,
        scenario_eval_run_coverage_complete=True,
        batch_trace_package=trace_package,
        support_artifacts={"present": present},
        visual_artifacts={
            "frames": [
                {"scenario_eval_run_id": "run-a", "camera": "overview"},
                {"scenario_eval_run_id": "run-a", "camera": "side"},
                {"scenario_eval_run_id": "run-a", "camera": "sim_robot_follow_pov"},
            ],
            "overview_video": {"status": "complete"},
            "robot_pov_video": {"status": "complete"},
            "side_video": {"status": "complete"},
        },
        collision_summary={"collision_dynamics_validated": True},
        digital_twin_fidelity_qa={"robot_team_grade_fidelity_passed": True},
        robot_team_handoff_blockers=[],
        claim_boundary={"sim": True},
    )
    assert closure_complete["status"] == "completed"

    monkeypatch.setattr(
        mg,
        "run_mujoco_g1_simulator_command",
        lambda **kwargs: {
            "status": "completed",
            "simulator_backend": "mujoco",
            "unitree_g1_asset_spawned": True,
            "simulator_execution_proven": True,
            "attempt_count": 1,
            "scenario_eval_run_count": 1,
            "missing_scenario_eval_run_count": 0,
            "collision_geometry_loaded": True,
            "collision_dynamics_validated": True,
            "robot_team_handoff_ready": False,
            "output_dir": str(tmp_path),
        },
    )
    assert mg.main(
        [
            "--capture-root",
            str(tmp_path),
            "--g1-model-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--simulator-output",
            str(tmp_path / "sim.json"),
            "--steps",
            "1",
            "--duration-seconds",
            "0.1",
            "--skip-render-frames",
            "--render-every-step",
            "--max-rendered-episodes",
            "1",
            "--max-rendered-steps",
            "1",
            "--allow-fetch-g1-assets",
            "--no-fetch-g1-assets",
            "--menagerie-ref",
            "ref",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    monkeypatch.delenv("BLUEPRINT_CAPTURE_ROOT", raising=False)
    with pytest.raises(SystemExit):
        mg.main([])
