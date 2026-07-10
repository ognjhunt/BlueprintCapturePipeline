from __future__ import annotations

import inspect
import json
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.slow
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline.robot_eval_execution import build_simulator_command_artifacts
from blueprint_pipeline.mujoco_g1_simulator_command import (
    _build_digital_twin_fidelity_qa,
    _collision_summary,
    _convert_glb_to_obj,
    _episode_navigation_spec,
    _matrix_runs,
    _obj_texture_material_summary,
    _obj_vertex_color_summary,
    _render_capture_steps,
    _scene_collision_contact_count,
    _visual_artifact_summary,
    _write_mjcf_wrapper,
    run_mujoco_g1_simulator_command,
)


def test_render_capture_steps_samples_continuous_motion() -> None:
    steps = _render_capture_steps(240)

    assert len(steps) == 24
    assert min(steps) == 0
    assert max(steps) == 239
    assert len({b - a for a, b in zip(sorted(steps), sorted(steps)[1:])}) <= 2


def test_matrix_runs_loads_rows_and_reports_missing_required_ids(tmp_path: Path) -> None:
    matrix_path = tmp_path / "scenario_eval_matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "scenario_eval_run_count": 2,
                "runs": [
                    {"scenario_eval_run_id": "run-explicit", "task_id": "walk_to_target"},
                    {"task_id": "walk_to_target", "scenario_id": "scenario-b"},
                ],
            }
        ),
        encoding="utf-8",
    )

    runs, summary = _matrix_runs(matrix_path)

    assert len(runs) == 2
    assert summary["scenario_eval_run_count"] == 2
    assert runs[0]["scenario_eval_run_id"] == "run-explicit"
    assert "scenario_eval_run_id" not in runs[1]
    assert summary["missing_scenario_eval_run_id_indexes"] == [2]
    assert summary["scenario_eval_run_ids_unique"] is True


def test_mujoco_g1_command_rejects_malformed_supplied_matrix_before_scene_work(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "scenario_eval_matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "scenario_eval_run_count": 2,
                "runs": [
                    {"scenario_eval_run_id": "duplicate-run", "task_id": "walk_to_target"},
                    {"scenario_eval_run_id": "duplicate-run", "task_id": "walk_to_target"},
                    {"task_id": "walk_to_target"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_mujoco_g1_simulator_command(
            capture_root=tmp_path / "capture-without-scene",
            scenario_eval_matrix_path=matrix_path,
            render_frames=False,
        )

    message = str(exc_info.value)
    assert "scenario_eval_matrix_missing_scenario_eval_run_id" in message
    assert "scenario_eval_matrix_duplicate_scenario_eval_run_id" in message
    assert "scenario_eval_matrix_declared_count_mismatch" in message


def test_episode_navigation_spec_uses_explicit_route_and_stable_seed() -> None:
    run = {
        "scenario_eval_run_id": "run-a",
        "task_id": "walk_to_target",
        "scenario_id": "scenario-a",
        "concrete_mutation": {
            "spawn_pose": [1.0, 2.0, 0.81],
            "target_pose": {"x": 3.0, "y": 4.0, "z": 0.82},
        },
    }
    mesh_info = {"bounds": [[-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]]}

    first = _episode_navigation_spec(run=run, mesh_info=mesh_info, index=1)
    second = _episode_navigation_spec(run=run, mesh_info=mesh_info, index=1)

    assert first == second
    assert first["route_source"] == "matrix_explicit_spawn_and_target"
    assert first["start"] == (1.0, 2.0, 0.81)
    assert first["target"] == (3.0, 4.0, 0.82)
    assert first["route_distance_m"] > 0


def test_obj_vertex_color_summary_counts_rgb_vertices(tmp_path: Path) -> None:
    obj_path = tmp_path / "scene.obj"
    obj_path.write_text(
        "\n".join(
            [
                "v 0 0 0 0.1 0.2 0.3",
                "v 1 0 0 0.4 0.5 0.6",
                "v 0 1 0",
                "f 1 2 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _obj_vertex_color_summary(obj_path)

    assert summary["status"] == "inspected"
    assert summary["vertex_count"] == 3
    assert summary["face_count"] == 1
    assert summary["vertex_rgb_count"] == 2
    assert summary["has_vertex_rgb"] is True
    assert summary["vertex_rgb_fraction"] == 0.666667


def test_obj_texture_material_summary_detects_map_kd_texture(tmp_path: Path) -> None:
    obj_path = tmp_path / "scene.obj"
    texture_path = tmp_path / "tex.png"
    Image.new("RGB", (4, 4), color=(180, 90, 40)).save(texture_path)
    (tmp_path / "material.mtl").write_text(
        "newmtl scene\nmap_Kd tex.png\n",
        encoding="utf-8",
    )
    obj_path.write_text(
        "mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )

    summary = _obj_texture_material_summary(obj_path)

    assert summary["status"] == "inspected"
    assert summary["mtl_file"] == "material.mtl"
    assert summary["map_kd_texture_file"] == "tex.png"
    assert summary["map_kd_texture_path"] == str(texture_path.resolve())
    assert summary["texture_exists"] is True

    only_obj_dir = tmp_path / "only_obj"
    only_obj_dir.mkdir()
    only_obj = only_obj_dir / "only_scene.obj"
    only_obj.write_text("v 0 0 0\n", encoding="utf-8")

    missing = _obj_texture_material_summary(only_obj)

    assert missing["status"] == "no_mtl"
    assert missing["texture_exists"] is False
    assert missing["map_kd_texture_file"] is None


def test_convert_glb_to_obj_includes_texture_material_summary_and_keeps_signature(
    tmp_path: Path,
    monkeypatch,
) -> None:
    signature = inspect.signature(_convert_glb_to_obj)
    assert list(signature.parameters) == [
        "glb_path",
        "obj_path",
        "collision_proxy_limit",
        "collision_proxy_mode",
    ]
    assert signature.parameters["collision_proxy_limit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["collision_proxy_limit"].default == 160
    assert signature.parameters["collision_proxy_mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["collision_proxy_mode"].default == "aabb"

    class FakeMesh:
        is_empty = False
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        extents = np.array([1.0, 1.0, 0.0])
        centroid = np.array([0.33, 0.33, 0.0])

        def export(self, obj_path: Path) -> None:
            Image.new("RGB", (4, 4), color=(20, 140, 220)).save(obj_path.parent / "tex.png")
            (obj_path.parent / "material.mtl").write_text(
                "newmtl scene\nmap_Kd tex.png\n",
                encoding="utf-8",
            )
            obj_path.write_text(
                "mtllib material.mtl\n"
                "v 0 0 0 0.1 0.2 0.3\n"
                "v 1 0 0 0.4 0.5 0.6\n"
                "v 0 1 0 0.7 0.8 0.9\n"
                "f 1 2 3\n",
                encoding="utf-8",
            )

    fake_trimesh = types.SimpleNamespace(
        Scene=type("FakeScene", (), {}),
        load=lambda _path, force=None: FakeMesh(),
    )
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)
    monkeypatch.setattr(
        "blueprint_pipeline.mujoco_g1_simulator_command._glb_visual_summary",
        lambda _path: {"materials_count": 1},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.mujoco_g1_simulator_command._visual_object_semantics_summary",
        lambda **_kwargs: {"status": "available"},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.mujoco_g1_simulator_command._collision_proxy_geoms_from_mesh",
        lambda _mesh, max_proxies, mode="aabb": (
            [],
            {"max_proxy_count": max_proxies, "collision_proxy_mode": mode},
        ),
    )

    result = _convert_glb_to_obj(tmp_path / "scene.glb", tmp_path / "scene.obj")

    assert "obj_texture_material_summary" in result
    assert result["obj_texture_material_summary"]["texture_exists"] is True
    assert result["obj_texture_material_summary"]["map_kd_texture_file"] == "tex.png"
    assert result["obj_vertex_color_summary"]["has_vertex_rgb"] is True


def test_mjcf_wrapper_has_separate_scene_collision_geom(tmp_path: Path) -> None:
    scene_obj = tmp_path / "scene.obj"
    g1_xml = tmp_path / "g1.xml"
    wrapper = tmp_path / "wrapper.xml"
    scene_obj.write_text("v 0 0 0\n", encoding="utf-8")
    g1_xml.write_text("<mujoco/>", encoding="utf-8")

    _write_mjcf_wrapper(scene_obj, g1_xml, wrapper)
    xml = wrapper.read_text(encoding="utf-8")

    assert 'name="blueprint_scene_visual"' in xml
    assert 'name="blueprint_scene_collision"' in xml
    assert 'name="blueprint_reference_floor"' in xml
    assert 'material="blueprint_scene_mat" contype="0" conaffinity="0"' in xml
    assert (
        'material="blueprint_scene_collision_mat" contype="1" conaffinity="1"'
        in xml
    )


def test_mjcf_wrapper_uses_scene_derived_lights_and_shadow_quality(
    tmp_path: Path,
) -> None:
    scene_obj = tmp_path / "scene.obj"
    g1_xml = tmp_path / "g1.xml"
    wrapper = tmp_path / "wrapper.xml"
    scene_obj.write_text("v 0 0 0\n", encoding="utf-8")
    g1_xml.write_text("<mujoco/>", encoding="utf-8")

    _write_mjcf_wrapper(
        scene_obj,
        g1_xml,
        wrapper,
        scene_bounds=[[-3.0, -2.0, 0.0], [3.0, 2.0, 2.5]],
        scene_centroid=[0.0, 0.0, 1.25],
    )
    xml = wrapper.read_text(encoding="utf-8")
    root = ET.fromstring(xml)
    quality = root.find("./visual/quality")
    lights = root.findall("./worldbody/light")
    light_names = {light.attrib.get("name") for light in lights}
    key_light = next(light for light in lights if light.attrib.get("name") == "blueprint_key")

    assert quality is not None
    assert int(quality.attrib["shadowsize"]) > 0
    assert {"blueprint_key", "blueprint_fill"}.issubset(light_names)
    assert key_light.attrib["castshadow"] == "true"
    assert key_light.attrib["pos"] != "0 -4 8"
    assert '<material name="blueprint_scene_mat" rgba="0.45 0.50 0.55 1"/>' in xml


def test_mjcf_wrapper_without_scene_bounds_keeps_parseable_directional_light(
    tmp_path: Path,
) -> None:
    scene_obj = tmp_path / "scene.obj"
    g1_xml = tmp_path / "g1.xml"
    wrapper = tmp_path / "wrapper.xml"
    scene_obj.write_text("v 0 0 0\n", encoding="utf-8")
    g1_xml.write_text("<mujoco/>", encoding="utf-8")

    _write_mjcf_wrapper(scene_obj, g1_xml, wrapper)
    xml = wrapper.read_text(encoding="utf-8")
    root = ET.fromstring(xml)
    lights = root.findall("./worldbody/light")

    assert any(light.attrib.get("directional") == "true" for light in lights)
    assert any(light.attrib.get("pos") == "0 -4 8" for light in lights)


def test_mjcf_wrapper_binds_scene_texture_when_present(tmp_path: Path) -> None:
    scene_obj = tmp_path / "scene.obj"
    g1_xml = tmp_path / "g1.xml"
    wrapper = tmp_path / "wrapper.xml"
    texture = tmp_path / "tex.png"
    scene_obj.write_text("v 0 0 0\n", encoding="utf-8")
    g1_xml.write_text("<mujoco/>", encoding="utf-8")
    Image.new("RGB", (4, 4), color=(50, 180, 100)).save(texture)

    _write_mjcf_wrapper(scene_obj, g1_xml, wrapper, scene_texture_file=texture)
    xml = wrapper.read_text(encoding="utf-8")

    assert '<texture name="blueprint_scene_tex" type="2d"' in xml
    assert 'texture="blueprint_scene_tex"' in xml
    assert '<material name="blueprint_scene_mat" rgba="0.45 0.50 0.55 1"/>' not in xml


def test_mjcf_wrapper_uses_exact_grey_scene_material_without_texture(
    tmp_path: Path,
) -> None:
    scene_obj = tmp_path / "scene.obj"
    g1_xml = tmp_path / "g1.xml"
    wrapper = tmp_path / "wrapper.xml"
    scene_obj.write_text("v 0 0 0\n", encoding="utf-8")
    g1_xml.write_text("<mujoco/>", encoding="utf-8")

    _write_mjcf_wrapper(scene_obj, g1_xml, wrapper)
    xml = wrapper.read_text(encoding="utf-8")

    assert '<material name="blueprint_scene_mat" rgba="0.45 0.50 0.55 1"/>' in xml
    assert 'texture="blueprint_scene_tex"' not in xml


def test_mujoco_loads_textured_scene_wrapper_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform == "darwin":
        monkeypatch.delenv("MUJOCO_GL", raising=False)
    mujoco = pytest.importorskip("mujoco")
    scene_obj = tmp_path / "scene.obj"
    g1_xml = tmp_path / "g1.xml"
    wrapper = tmp_path / "wrapper.xml"
    texture = tmp_path / "tex.png"
    Image.new("RGB", (4, 4), color=(120, 40, 200)).save(texture)
    scene_obj.write_text(
        "\n".join(
            [
                "mtllib material.mtl",
                "usemtl scene",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "vt 0 0",
                "vt 1 0",
                "vt 0 1",
                "vt 1 1",
                "f 1/1 2/2 3/3",
                "f 1/1 2/2 4/4",
                "f 1/1 3/3 4/4",
                "f 2/2 3/3 4/4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "material.mtl").write_text(
        "newmtl scene\nmap_Kd tex.png\n",
        encoding="utf-8",
    )
    g1_xml.write_text("<mujoco><worldbody/></mujoco>", encoding="utf-8")

    _write_mjcf_wrapper(scene_obj, g1_xml, wrapper, scene_texture_file=texture)

    model = mujoco.MjModel.from_xml_path(str(wrapper))

    assert model.ntex >= 1
    assert model.nmat >= 1


def test_mjcf_wrapper_prefers_collision_proxy_boxes(tmp_path: Path) -> None:
    scene_obj = tmp_path / "scene.obj"
    g1_xml = tmp_path / "g1.xml"
    wrapper = tmp_path / "wrapper.xml"
    scene_obj.write_text("v 0 0 0\n", encoding="utf-8")
    g1_xml.write_text("<mujoco/>", encoding="utf-8")

    _write_mjcf_wrapper(
        scene_obj,
        g1_xml,
        wrapper,
        collision_proxies=[
            {
                "name": "rack_a",
                "pos": [1.0, 2.0, 0.5],
                "size": [0.4, 0.2, 0.5],
            }
        ],
    )
    xml = wrapper.read_text(encoding="utf-8")

    assert 'name="blueprint_scene_collision"' not in xml
    assert 'name="blueprint_collision_proxy_000_rack_a"' in xml
    assert 'type="box"' in xml
    assert 'contype="1" conaffinity="1"' in xml


def test_scene_collision_contact_count_includes_proxy_geoms() -> None:
    records = [
        {"geom_names": ["blueprint_reference_floor", "geom_1"]},
        {"geom_names": ["blueprint_collision_proxy_001_box", "geom_2"]},
        {"scene_collision_contact": True},
    ]

    assert _scene_collision_contact_count(records) == 2


def test_visual_artifact_summary_classifies_frames_and_records_material_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("blueprint_pipeline.mujoco_g1_simulator_command.shutil.which", lambda _: None)
    overview = tmp_path / "overview_0000.png"
    pov = tmp_path / "sim_robot_follow_pov_0000.png"
    side = tmp_path / "side_0000.png"
    for path, base in ((overview, 20), (pov, 80), (side, 140)):
        image = Image.new("RGB", (24, 16))
        for x in range(image.width):
            for y in range(image.height):
                image.putpixel(
                    (x, y),
                    (
                        (base + x * 5) % 255,
                        (base + y * 9) % 255,
                        (base + x * 3 + y * 4) % 255,
                    ),
                )
        image.save(path)
    frames = [
        {"camera": "overview", "path": str(overview), "step": 0},
        {"camera": "sim_robot_follow_pov", "path": str(pov), "step": 0},
        {"camera": "side", "path": str(side), "step": 0},
    ]
    mesh_info = {
        "source_glb": str(tmp_path / "scene.glb"),
        "converted_obj": str(tmp_path / "scene.obj"),
        "visual_asset_summary": {
            "materials_count": 1,
            "textures_count": 0,
            "images_count": 0,
            "has_vertex_colors": False,
        },
        "obj_vertex_color_summary": {
            "has_vertex_rgb": True,
            "vertex_rgb_fraction": 1.0,
        },
        "obj_texture_material_summary": {
            "map_kd_texture_file": "tex.png",
            "texture_exists": True,
        },
        "mujoco_visual_fidelity_boundary": "test boundary",
    }

    summary = _visual_artifact_summary(
        frames=frames,
        output_root=tmp_path,
        mesh_info=mesh_info,
        model_timestep_s=0.01,
    )

    assert summary["status"] == "complete"
    assert summary["overview_frames"] == [str(overview)]
    assert summary["robot_pov_frames"] == [str(pov)]
    assert summary["side_frames"] == [str(side)]
    assert summary["overview_video"]["status"] == "not_generated"
    assert summary["robot_pov_video"]["status"] == "not_generated"
    assert summary["side_video"]["status"] == "not_generated"
    assert summary["texture_material_evidence"]["status"] == (
        "materialized_vertex_color_scene_evidence_present"
    )
    assert summary["texture_material_evidence"]["obj_map_kd_texture_present"] is True
    assert summary["texture_material_evidence"]["obj_map_kd_texture_file"] == "tex.png"
    assert (
        summary["texture_material_evidence"]["mujoco_scene_material_mode"]
        == "pbr_texture_bound"
    )
    assert summary["texture_material_evidence"]["white_scene_success_allowed"] is False
    assert summary["blank_scene_checks"]["status"] == "checked"
    assert summary["blank_scene_checks"]["all_frames_nonblank"] is True


def _install_fake_mujoco_backend(monkeypatch) -> None:
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

    class FakeRenderer:
        def __init__(self, _model: FakeModel, *, height: int, width: int) -> None:
            self.height = height
            self.width = width
            self.render_count = 0

        def update_scene(self, _data: FakeData, camera: object) -> None:
            self.camera = camera

        def render(self) -> np.ndarray:
            self.render_count += 1
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            image[:, :, 0] = (self.render_count * 31) % 255
            image[:, :, 1] = np.arange(self.width, dtype=np.uint16) % 255
            image[:, :, 2] = np.arange(self.height, dtype=np.uint16)[:, None] % 255
            return image

        def close(self) -> None:
            return None

    def fake_name_to_id(_model: FakeModel, _object_type: object, name: str) -> int:
        return 0 if name in {"floating_base_joint", "stand"} else -1

    def fake_step(model: FakeModel, data: FakeData) -> None:
        data.time += model.opt.timestep

    fake_mujoco = types.SimpleNamespace(
        __version__="fake-3.0",
        MjModel=FakeModel,
        MjData=FakeData,
        MjvCamera=FakeCamera,
        Renderer=FakeRenderer,
        mjtObj=types.SimpleNamespace(mjOBJ_JOINT=1, mjOBJ_KEY=2),
        mjtCamera=types.SimpleNamespace(mjCAMERA_FREE=1),
        mj_name2id=fake_name_to_id,
        mj_forward=lambda _model, _data: None,
        mj_step=fake_step,
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)


def _seed_fake_capture_and_g1(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    capture_root = tmp_path / "capture"
    scene_glb = capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_collider.glb"
    scene_glb.parent.mkdir(parents=True)
    scene_glb.write_bytes(b"fake glb")
    g1_root = tmp_path / "unitree_g1"
    (g1_root / "assets").mkdir(parents=True)
    (g1_root / "g1.xml").write_text("<mujoco><worldbody/></mujoco>", encoding="utf-8")

    def fake_convert(
        _glb_path: Path,
        obj_path: Path,
        *,
        collision_proxy_mode: str = "aabb",
    ) -> dict[str, object]:
        assert collision_proxy_mode == "aabb"
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
            "mujoco_visual_fidelity_boundary": "test boundary",
        }

    monkeypatch.setattr(
        "blueprint_pipeline.mujoco_g1_simulator_command._convert_glb_to_obj",
        fake_convert,
    )
    return capture_root, g1_root


def test_mujoco_g1_command_runs_every_matrix_row_with_fake_backend(
    tmp_path: Path,
    monkeypatch,
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

    def fake_name_to_id(_model: FakeModel, _object_type: object, name: str) -> int:
        return 0 if name in {"floating_base_joint", "stand"} else -1

    def fake_step(model: FakeModel, data: FakeData) -> None:
        data.time += model.opt.timestep

    fake_mujoco = types.SimpleNamespace(
        __version__="fake-3.0",
        MjModel=FakeModel,
        MjData=FakeData,
        MjvCamera=FakeCamera,
        mjtObj=types.SimpleNamespace(mjOBJ_JOINT=1, mjOBJ_KEY=2),
        mjtCamera=types.SimpleNamespace(mjCAMERA_FREE=1),
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
    matrix_path = tmp_path / "scenario_eval_matrix.json"
    matrix_rows = [
        {
            "scenario_eval_run_id": "run-a",
            "episode_id": "episode-a",
            "task_id": "walk_to_target",
            "scenario_id": "scenario-a",
            "scenario_variation_instance_id": "variation-a",
            "variation_name": "lighting_variation",
            "concrete_mutation": {
                "spawn_pose": [-1.0, 0.0, 0.793],
                "target_pose": [1.0, 0.0, 0.793],
            },
        },
        {
            "scenario_eval_run_id": "run-b",
            "task_id": "walk_to_target",
            "scenario_id": "scenario-b",
            "scenario_variation_instance_id": "variation-b",
            "variation_name": "blocked_path",
        },
        {
            "scenario_eval_run_id": "run-c",
            "task_id": "walk_to_target",
            "scenario_id": "scenario-c",
            "scenario_variation_instance_id": "variation-c",
            "variation_name": "narrow_approach_angle",
        },
    ]
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scenario_eval_matrix.v1",
                "status": "completed",
                "scenario_eval_run_count": len(matrix_rows),
                "runs": matrix_rows,
            }
        ),
        encoding="utf-8",
    )

    def fake_convert(
        _glb_path: Path,
        obj_path: Path,
        *,
        collision_proxy_mode: str = "aabb",
    ) -> dict[str, object]:
        assert collision_proxy_mode == "aabb"
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
            "mujoco_visual_fidelity_boundary": "test boundary",
        }

    monkeypatch.setattr(
        "blueprint_pipeline.mujoco_g1_simulator_command._convert_glb_to_obj",
        fake_convert,
    )

    simulator_output = tmp_path / "mujoco_g1_simulator_output.json"
    payload = run_mujoco_g1_simulator_command(
        capture_root=capture_root,
        g1_model_root=g1_root,
        output_dir=tmp_path / "mujoco-output",
        simulator_output_path=simulator_output,
        scenario_eval_matrix_path=matrix_path,
        steps=4,
        render_frames=False,
        max_rendered_episodes=1,
    )

    assert payload["status"] == "completed"
    assert payload["attempt_count"] == 3
    assert payload["scenario_eval_run_count"] == 3
    assert payload["covered_scenario_eval_run_ids"] == ["run-a", "run-b", "run-c"]
    assert payload["missing_scenario_eval_run_ids"] == []
    assert payload["scenario_eval_run_coverage_complete"] is True
    assert payload["rendered_episode_count"] == 0
    assert payload["deterministic_per_episode_spawn_target_seed_handling"] is True
    assert payload["ai_route_selection_used_at_runtime"] is False
    assert payload["collision_geometry_loaded"] is True
    assert payload["scene_collision_mesh_geom_enabled"] is True
    assert payload["scene_visual_mesh_collision_twin_enabled"] is True
    assert payload["scene_visual_mesh_collisions_enabled"] is False
    assert payload["collision_dynamics_validated"] is True
    assert payload["collision_avoidance_validated"] is True
    assert payload["physics_controlled_preview_proven"] is True
    assert payload["robot_team_handoff_ready"] is False
    assert "balanced_walking_controller_not_integrated_in_default_mujoco_preview" in payload[
        "robot_team_handoff_blockers"
    ]
    assert payload["official_policy_handoff"]["entrypoint"] == (
        "python -m blueprint_pipeline.official_g1_policy_handoff"
    )
    assert [attempt["scenario_eval_run_id"] for attempt in payload["attempts"]] == [
        "run-a",
        "run-b",
        "run-c",
    ]
    assert payload["successful_task_attempt_count"] == 3
    assert payload["failed_task_attempt_count"] == 0
    assert payload["task_success_rate"] == 1.0
    assert payload["task_success_summary"]["goal_reached_attempt_count"] == 3
    assert payload["attempts"][0]["spawn_pose"] == [-1.0, 0.0, 0.793]
    assert payload["attempts"][0]["target_pose"] == [1.0, 0.0, 0.793]
    assert payload["attempts"][0]["task_success"] is True
    assert payload["attempts"][0]["metrics"]["endpoint_clean"] is True
    assert payload["attempts"][0]["metrics"]["final_target_error_m"] == 0.0
    assert payload["machine_trace_package_complete"] is True
    assert payload["robot_team_grade_package_complete"] is False
    closure_path = Path(payload["artifact_paths"]["batch_closure_manifest"])
    trace_package_path = Path(payload["artifact_paths"]["batch_trace_package_manifest"])
    metrics_path = Path(payload["artifact_paths"]["batch_metrics"])
    failure_labels_path = Path(payload["artifact_paths"]["batch_failure_labels"])
    visual_review_ledger_path = Path(payload["artifact_paths"]["batch_visual_review_ledger"])
    visual_media_coverage_path = Path(
        payload["artifact_paths"]["batch_visual_media_coverage"]
    )
    fidelity_qa_path = Path(payload["artifact_paths"]["digital_twin_fidelity_qa"])
    planner_state_path = Path(payload["artifact_paths"]["batch_planner_state_jsonl"])
    control_stream_path = Path(payload["artifact_paths"]["batch_control_stream_jsonl"])
    assert closure_path.is_file()
    assert trace_package_path.is_file()
    assert metrics_path.is_file()
    assert failure_labels_path.is_file()
    assert visual_review_ledger_path.is_file()
    assert visual_media_coverage_path.is_file()
    assert fidelity_qa_path.is_file()
    assert planner_state_path.is_file()
    assert control_stream_path.is_file()
    control_rows = [
        json.loads(line)
        for line in control_stream_path.read_text(encoding="utf-8").splitlines()
    ]
    first_control_row = control_rows[0]
    first_action = first_control_row["action"]
    assert first_control_row["sim_time_s"] == 0.0
    assert first_action["sim_time_s"] == 0.0
    assert len(first_control_row["base_pose_7d"]) == 7
    assert first_control_row["base_pose_7d"] == first_action["base_pose_7d"]
    assert first_control_row["base_pose_7d"][0] == pytest.approx(-1.0)
    assert first_control_row["base_pose_7d"][2] == pytest.approx(0.793)
    assert first_control_row["robot_state_source"] == (
        "mujoco_qpos_root_pose_after_mj_forward"
    )
    assert first_control_row["timestamp_source"] == "mujoco_data_time_s"
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    trace_package = json.loads(trace_package_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    visual_review = json.loads(visual_review_ledger_path.read_text(encoding="utf-8"))
    visual_media_coverage = json.loads(
        visual_media_coverage_path.read_text(encoding="utf-8")
    )
    fidelity_qa = json.loads(fidelity_qa_path.read_text(encoding="utf-8"))
    assert closure["batch_execution_status"] == "completed"
    assert closure["machine_trace_package_complete"] is True
    assert closure["robot_team_grade_package_complete"] is False
    assert closure["scenario_eval_run_coverage_complete"] is True
    assert closure["metric_coverage_complete"] is True
    assert closure["failure_label_coverage_complete"] is True
    assert closure["visual_review_coverage_complete"] is True
    assert closure["visual_review"]["review_count"] == 3
    assert visual_review["status"] == "accepted"
    assert visual_review["review_count"] == 3
    assert visual_review["visual_review_coverage_complete"] is True
    assert visual_review["records"][0]["review_status"] == (
        "accepted_deterministic_simulator_visual_review"
    )
    assert "visual_video_coverage_not_complete_for_all_runs" in closure[
        "robot_team_grade_blockers"
    ]
    assert closure["digital_twin_fidelity_qa"]["machine_fidelity_audit_complete"] is False
    assert closure["digital_twin_fidelity_qa"]["robot_team_grade_fidelity_passed"] is False
    assert "digital_twin_visual_frames_blank_or_missing" in closure["robot_team_grade_blockers"]
    assert trace_package["attempt_count"] == 3
    assert trace_package["metric_coverage_complete"] is True
    assert trace_package["planner_state_record_count"] == 3
    assert trace_package["control_stream_record_count"] > 0
    assert trace_package["planner_state_coverage_complete"] is True
    assert trace_package["control_stream_coverage_complete"] is True
    assert metrics["attempt_metric_row_count"] == 3
    assert metrics["metric_coverage_complete"] is True
    assert "min_clearance_m" in metrics["required_metric_keys"]
    assert "clearance_threshold_m" in metrics["required_metric_keys"]
    assert visual_media_coverage["status"] == "incomplete"
    assert visual_media_coverage["required_scenario_eval_run_count"] == 3
    assert visual_media_coverage["missing_visual_media_run_count"] == 3
    assert visual_media_coverage["missing_visual_media_scenario_eval_run_ids"] == [
        "run-a",
        "run-b",
        "run-c",
    ]
    assert visual_media_coverage["all_required_runs_have_visual_recording"] is False
    assert visual_media_coverage["all_required_runs_have_robot_pov_video"] is False
    assert visual_media_coverage["all_required_runs_have_third_person_video"] is False
    assert fidelity_qa["machine_fidelity_audit_complete"] is False
    assert fidelity_qa["robot_team_grade_fidelity_passed"] is False
    assert fidelity_qa["gates"]["nonblank_visual_evidence"]["passed"] is False
    assert simulator_output.is_file()

    artifacts = build_simulator_command_artifacts(
        job_dir=tmp_path / "job",
        simulator="mujoco",
        simulator_output=payload,
        generated_at="2026-06-14T00:00:00Z",
    )
    trace = artifacts["normalized_attempt_trace"]
    manifest = artifacts["manifest"]
    job_dir = tmp_path / "job"
    assert trace["attempt_count"] == 3
    assert trace["required_scenario_eval_run_count"] == 3
    assert trace["covered_scenario_eval_run_count"] == 3
    assert trace["missing_scenario_eval_run_count"] == 0
    assert trace["scenario_eval_run_coverage_complete"] is True
    assert trace["task_success_summary"]["successful_attempt_count"] == 3
    assert trace["task_success_rate"] == 1.0
    assert [attempt["scenario_eval_run_id"] for attempt in trace["attempts"]] == [
        "run-a",
        "run-b",
        "run-c",
    ]
    assert manifest["artifact_paths"]["simulator_command_batch_closure_manifest"] == (
        "simulator_command_batch_closure_manifest.json"
    )
    assert manifest["artifact_paths"]["simulator_command_batch_planner_state_jsonl"] == (
        "simulator_command_batch_planner_state.jsonl"
    )
    assert manifest["artifact_paths"]["simulator_command_batch_control_stream_jsonl"] == (
        "simulator_command_batch_control_stream.jsonl"
    )
    assert manifest["artifact_paths"]["simulator_command_batch_visual_media_coverage"] == (
        "simulator_command_batch_visual_media_coverage.json"
    )
    assert manifest["artifact_paths"]["simulator_command_batch_visual_review_ledger"] == (
        "simulator_command_batch_visual_review_ledger.json"
    )
    assert manifest["artifact_paths"]["visual_review_ledger"] == "visual_review_ledger.json"
    assert manifest["visual_review_coverage_complete"] is True
    assert manifest["artifact_paths"]["simulator_command_digital_twin_fidelity_qa"] == (
        "simulator_command_digital_twin_fidelity_qa.json"
    )
    assert manifest["command_batch_trace_job_artifacts_copied"] is True
    assert (job_dir / "simulator_command_batch_closure_manifest.json").is_file()
    assert (job_dir / "simulator_command_batch_trace_package_manifest.json").is_file()
    assert (job_dir / "simulator_command_batch_attempt_trace.jsonl").is_file()
    assert (job_dir / "simulator_command_batch_contact_stream.jsonl").is_file()
    assert (job_dir / "simulator_command_batch_planner_state.jsonl").is_file()
    assert (job_dir / "simulator_command_batch_control_stream.jsonl").is_file()
    assert (job_dir / "simulator_command_batch_visual_media_coverage.json").is_file()
    assert (job_dir / "simulator_command_batch_visual_review_ledger.json").is_file()
    assert (job_dir / "visual_review_ledger.json").is_file()
    assert (job_dir / "simulator_command_digital_twin_fidelity_qa.json").is_file()
    job_trace_package = json.loads(
        (job_dir / "simulator_command_batch_trace_package_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert job_trace_package["artifact_paths"]["planner_state_jsonl"] == (
        "simulator_command_batch_planner_state.jsonl"
    )
    assert job_trace_package["artifact_paths"]["control_stream_jsonl"] == (
        "simulator_command_batch_control_stream.jsonl"
    )
    assert job_trace_package["source_artifact_paths"]["planner_state_jsonl"] == str(
        planner_state_path
    )


def test_mujoco_g1_command_records_diagnostic_visuals_for_blocked_episode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_mujoco_backend(monkeypatch)
    capture_root, g1_root = _seed_fake_capture_and_g1(tmp_path, monkeypatch)
    matrix_path = tmp_path / "scenario_eval_matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scenario_eval_matrix.v1",
                "status": "completed",
                "scenario_eval_run_count": 1,
                "runs": [
                    {
                        "scenario_eval_run_id": "run-blocked",
                        "task_id": "walk_to_target",
                        "scenario_id": "scenario-blocked",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "blueprint_pipeline.mujoco_g1_simulator_command._evaluate_preview_candidate",
        lambda **_kwargs: {
            "candidate_kind": "direct",
            "pose": (0.0, 0.0, 0.793),
            "yaw": 0.0,
            "phase": 0.0,
            "moving": False,
            "contacts": [{"scene_collision_contact": True}],
            "contact_count": 1,
            "scene_collision_contact_count": 1,
            "accepted": False,
        },
    )

    def fake_video_writer(
        *,
        camera: str,
        frame_paths: list[str],
        frame_times_s: list[float | None],
        output_root: Path,
        fallback_frame_duration_s: float,
    ) -> dict[str, object]:
        del frame_times_s, fallback_frame_duration_s
        video_path = output_root / f"{camera}.mp4"
        video_path.write_bytes(b"fake mp4 bytes")
        return {
            "status": "complete",
            "path": str(video_path),
            "frame_count": len(frame_paths),
            "realtime_timing_from_sim_time": True,
            "source_frames": list(frame_paths),
        }

    monkeypatch.setattr(
        "blueprint_pipeline.mujoco_g1_simulator_command._write_frame_video",
        fake_video_writer,
    )

    payload = run_mujoco_g1_simulator_command(
        capture_root=capture_root,
        g1_model_root=g1_root,
        output_dir=tmp_path / "mujoco-output-blocked",
        scenario_eval_matrix_path=matrix_path,
        steps=2,
        render_frames=True,
        max_rendered_episodes=1,
    )

    visual_media_coverage = json.loads(
        Path(payload["artifact_paths"]["batch_visual_media_coverage"]).read_text(
            encoding="utf-8"
        )
    )
    row = visual_media_coverage["rows"][0]

    assert payload["status"] == "completed"
    assert payload["rendered_episode_count"] == 1
    assert row["status"] == "complete"
    assert row["frame_counts"] == {
        "overview": 2,
        "sim_robot_follow_pov": 2,
        "side": 2,
    }
    assert row["robot_pov_video_present"] is True
    assert row["third_person_video_present"] is True
    assert visual_media_coverage["all_required_runs_have_visual_recording"] is True
    assert visual_media_coverage["missing_visual_media_run_count"] == 0


def test_mujoco_g1_command_stops_before_fake_scene_collision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_mujoco_backend(monkeypatch)
    capture_root, g1_root = _seed_fake_capture_and_g1(tmp_path, monkeypatch)
    matrix_path = tmp_path / "scenario_eval_matrix_collision.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scenario_eval_matrix.v1",
                "status": "completed",
                "scenario_eval_run_count": 1,
                "runs": [
                    {
                        "scenario_eval_run_id": "run-collision-wall",
                        "task_id": "walk_to_target",
                        "scenario_id": "scenario-collision-wall",
                        "concrete_mutation": {
                            "spawn_pose": [-1.0, 0.0, 0.793],
                            "target_pose": [1.0, 0.0, 0.793],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_contact_records(_model, data, _mujoco):
        x_position = float(data.qpos[0])
        if x_position > 0.05:
            return [
                {
                    "contact_index": 0,
                    "geom_ids": [1, 2],
                    "geom_names": ["blueprint_collision_proxy_000_wall", "pelvis"],
                    "body_names": ["world", "pelvis"],
                    "distance": -0.1,
                    "position_xyz": [x_position, 0.0, 0.7],
                    "contact_force_6d": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "scene_collision_contact": True,
                    "reference_floor_contact": False,
                }
            ]
        return []

    monkeypatch.setattr(
        "blueprint_pipeline.mujoco_g1_simulator_command._contact_records",
        fake_contact_records,
    )

    payload = run_mujoco_g1_simulator_command(
        capture_root=capture_root,
        g1_model_root=g1_root,
        output_dir=tmp_path / "mujoco-output-collision",
        simulator_output_path=tmp_path / "mujoco_g1_simulator_output_collision.json",
        scenario_eval_matrix_path=matrix_path,
        steps=8,
        render_frames=False,
        max_rendered_episodes=0,
    )
    attempt = payload["attempts"][0]
    actions = attempt["actions"]

    assert payload["status"] == "completed"
    assert payload["physics_controlled_preview_proven"] is True
    assert payload["collision_dynamics_validated"] is True
    assert payload["collision_avoidance_validated"] is True
    assert payload["robot_scene_contact_event_count"] == 0
    assert payload["collision_response_event_count"] > 0
    assert payload["collision_summary"]["rejected_scene_collision_probe_count"] > 0
    assert attempt["status"] == "failed_task_criteria"
    assert attempt["success"] is False
    assert attempt["task_success"] is False
    assert "failure_target_not_reached" in attempt["failure_mode_ids"]
    assert "failure_endpoint_not_clean" in attempt["failure_mode_ids"]
    assert "failure_clearance_near_miss" in attempt["failure_mode_ids"]
    assert attempt["metrics"]["robot_scene_contact_event_count"] == 0
    assert attempt["metrics"]["collision_response_event_count"] > 0
    assert attempt["metrics"]["near_miss_event_count"] > 0
    assert attempt["metrics"]["min_clearance_m"] == 0.0
    assert attempt["metrics"]["clearance_threshold_m"] == 0.15
    assert attempt["metrics"]["clearance_threshold_violation"] is True
    assert attempt["metrics"]["endpoint_clean"] is False
    assert attempt["metrics"]["goal_reached"] is False
    assert attempt["metrics"]["final_target_error_m"] > 0.25
    assert payload["task_success_summary"]["failed_attempt_count"] == 1
    assert payload["task_success_summary"]["near_miss_attempt_count"] == 1
    assert payload["task_success_summary"]["near_miss_event_count"] > 0
    assert payload["task_success_summary"]["min_clearance_m"] == 0.0
    assert payload["task_success_summary"]["failure_mode_counts"][
        "failure_target_not_reached"
    ] == 1
    assert payload["task_success_summary"]["failure_mode_counts"][
        "failure_clearance_near_miss"
    ] == 1
    failure_labels = json.loads(
        Path(payload["artifact_paths"]["batch_failure_labels"]).read_text(
            encoding="utf-8"
        )
    )
    failure_label = failure_labels["labels"][0]
    assert failure_label["status"] == "deterministically_labeled_failure"
    assert failure_label["label"] == "failure"
    assert failure_label["primary_failure_mode"] == "failure_target_not_reached"
    assert failure_label["criteria_results"]["success_criteria"][
        "goal_reached_within_tolerance"
    ] is False
    assert failure_label["criteria_results"]["metrics"]["min_clearance_m"] == 0.0
    assert failure_label["evidence_refs"]
    assert any(
        action["policy_action"] in {"stopped_by_collision_probe", "redirected_by_collision_probe"}
        for action in actions
    )
    assert max(action["root_position"][0] for action in actions) <= 0.05


def test_proxy_only_collision_summary_does_not_validate_visible_scene_collision() -> None:
    summary = _collision_summary([], collision_proxy_count=3)

    assert summary["scene_collision_proxy_geoms_enabled"] is True
    assert summary["proxy_collision_model_used"] is True
    assert summary["collision_avoidance_validated"] is True
    assert summary["proxy_collision_governed_preview_proven"] is True
    assert summary["visible_scene_collision_alignment_validated"] is False
    assert summary["collision_dynamics_validated"] is False
    assert summary["physics_controlled_preview_proven"] is False


def test_digital_twin_fidelity_qa_blocks_proxy_only_visual_physics_parity() -> None:
    collision_summary = _collision_summary([], collision_proxy_count=3)
    mesh_info = {
        "source_glb": "scene.glb",
        "converted_obj": "scene.obj",
        "vertices": 12,
        "faces": 8,
        "bounds": [[-1.0, -1.0, 0.0], [1.0, 1.0, 1.5]],
        "extents": [2.0, 2.0, 1.5],
        "visual_asset_summary": {
            "materials_count": 1,
            "textures_count": 0,
            "images_count": 0,
            "has_embedded_or_referenced_image_textures": False,
        },
        "visual_object_semantics_summary": {
            "status": "missing",
            "visible_object_count": 1,
            "named_visible_object_count": 0,
            "semantic_labeled_visible_object_count": 0,
            "visible_objects": [
                {
                    "object_id": "visible_object_0000_geometry_0",
                    "source_component_index": 0,
                    "name": "geometry_0",
                    "semantic_label_available": False,
                    "geometry_name": "geometry_0",
                    "gltf_mesh_name": "geometry_0",
                }
            ],
            "blockers": ["visible_geometry_object_names_missing"],
        },
        "obj_vertex_color_summary": {
            "has_vertex_rgb": True,
            "vertex_rgb_fraction": 1.0,
        },
        "collision_proxy_summary": {
            "source_component_count": 3,
            "proxy_count": 3,
            "max_proxy_count": 160,
            "skipped": {},
        },
    }
    visual_artifacts = {
        "texture_material_evidence": {
            "status": "materialized_vertex_color_scene_evidence_present"
        },
        "blank_scene_checks": {"status": "not_applicable"},
    }

    qa = _build_digital_twin_fidelity_qa(
        generated_at="2026-06-14T00:00:00Z",
        mesh_info=mesh_info,
        collision_summary=collision_summary,
        visual_artifacts=visual_artifacts,
        artifact_refs={
            "scene_load_trace": "scene_load_trace.json",
            "source_scene_glb": "scene.glb",
            "converted_scene_obj": "scene.obj",
        },
    )

    assert qa["machine_fidelity_audit_complete"] is False
    assert qa["robot_team_grade_fidelity_passed"] is False
    assert qa["gates"]["object_semantics_available"]["passed"] is False
    assert qa["gates"]["visible_objects_have_physics_coverage"]["passed"] is False
    assert qa["gates"]["visual_object_has_matching_physics"]["passed"] is False
    assert "digital_twin_object_semantics_missing" in qa["blockers"]
    assert "visible_objects_without_physics_coverage" in qa["blockers"]
    assert "visual_collision_alignment_not_validated" in qa["blockers"]
    gaps = qa["object_level_fidelity_gaps"]
    missing_semantics = gaps["missing_semantic_objects"]
    missing_physics = gaps["visible_objects_without_physics_coverage"]
    assert missing_semantics[0]["object_id"] == "visible_object_0000_geometry_0"
    assert missing_semantics[0]["reason"] == (
        "visible_glb_object_has_only_generated_or_missing_semantic_name"
    )
    assert missing_semantics[0]["evidence_refs"][0]["path"] == "scene_load_trace.json"
    assert missing_physics[0]["object_id"] == "visible_object_0000_geometry_0"
    assert missing_physics[0]["coverage_reason"] == (
        "visible_object_proxy_component_mapping_not_one_to_one"
    )
    assert (
        missing_physics[0]["component_coverage_summary"]["component_mapping_status"]
        == "not_one_to_one"
    )


def test_digital_twin_fidelity_qa_passes_component_mapped_proxy_physics_coverage() -> None:
    collision_summary = _collision_summary([], collision_proxy_count=1)
    mesh_info = {
        "source_glb": "scene.glb",
        "converted_obj": "scene.obj",
        "vertices": 12,
        "faces": 8,
        "bounds": [[-1.0, -1.0, 0.0], [1.0, 1.0, 1.5]],
        "extents": [2.0, 2.0, 1.5],
        "visual_asset_summary": {
            "materials_count": 1,
            "textures_count": 0,
            "images_count": 0,
            "has_embedded_or_referenced_image_textures": False,
        },
        "visual_object_semantics_summary": {
            "status": "available",
            "visible_object_count": 1,
            "named_visible_object_count": 1,
            "visible_objects": [
                {
                    "object_id": "visible_object_0000_test_counter",
                    "source_component_index": 0,
                    "name": "test_counter",
                    "bounds": [[-0.3, -0.2, 0.0], [0.3, 0.2, 0.8]],
                    "extents": [0.6, 0.4, 0.8],
                }
            ],
        },
        "obj_vertex_color_summary": {
            "has_vertex_rgb": True,
            "vertex_rgb_fraction": 1.0,
        },
        "collision_proxy_summary": {
            "source_component_count": 1,
            "proxy_count": 1,
            "max_proxy_count": 160,
            "skipped": {},
            "component_coverage": {
                "covered_source_component_indexes": [0],
                "covered_source_component_count": 1,
                "reference_floor_covered_source_component_indexes": [],
                "intentionally_excluded_source_component_indexes": [],
                "truncated_source_component_indexes": [],
                "truncated_source_component_count": 0,
                "uncovered_source_component_indexes": [],
                "uncovered_source_component_count": 0,
                "component_proxy_coverage_complete": True,
            },
        },
    }
    visual_artifacts = {
        "texture_material_evidence": {
            "status": "materialized_vertex_color_scene_evidence_present"
        },
        "blank_scene_checks": {"status": "not_applicable"},
    }

    qa = _build_digital_twin_fidelity_qa(
        generated_at="2026-06-14T00:00:00Z",
        mesh_info=mesh_info,
        collision_summary=collision_summary,
        visual_artifacts=visual_artifacts,
    )

    assert qa["machine_fidelity_audit_complete"] is True
    assert qa["robot_team_grade_fidelity_passed"] is True
    assert qa["gates"]["object_semantics_available"]["passed"] is True
    assert qa["gates"]["visible_objects_have_physics_coverage"]["passed"] is True
    assert qa["gates"]["visual_object_has_matching_physics"]["passed"] is True
    assert qa["visual_collision_parity"]["visible_scene_collision_alignment_validated"] is False
    assert qa["object_level_fidelity_gaps"]["missing_semantic_objects"] == []
    assert qa["object_level_fidelity_gaps"]["visible_objects_without_physics_coverage"] == []
    assert qa["blockers"] == []


def test_digital_twin_fidelity_qa_blocks_uncovered_visible_object_component() -> None:
    collision_summary = _collision_summary([], collision_proxy_count=1)
    mesh_info = {
        "source_glb": "scene.glb",
        "converted_obj": "scene.obj",
        "vertices": 12,
        "faces": 8,
        "bounds": [[-1.0, -1.0, 0.0], [1.0, 1.0, 1.5]],
        "extents": [2.0, 2.0, 1.5],
        "visual_asset_summary": {"materials_count": 1},
        "visual_object_semantics_summary": {
            "status": "available",
            "visible_object_count": 2,
            "named_visible_object_count": 2,
            "visible_objects": [
                {
                    "object_id": "visible_object_0000_test_counter",
                    "source_component_index": 0,
                    "name": "test_counter",
                },
                {
                    "object_id": "visible_object_0001_test_shelf",
                    "source_component_index": 1,
                    "name": "test_shelf",
                },
            ],
        },
        "obj_vertex_color_summary": {
            "has_vertex_rgb": True,
            "vertex_rgb_fraction": 1.0,
        },
        "collision_proxy_summary": {
            "source_component_count": 2,
            "proxy_count": 1,
            "max_proxy_count": 160,
            "skipped": {},
            "component_coverage": {
                "covered_source_component_indexes": [0],
                "covered_source_component_count": 1,
                "reference_floor_covered_source_component_indexes": [],
                "intentionally_excluded_source_component_indexes": [],
                "truncated_source_component_indexes": [],
                "truncated_source_component_count": 0,
                "uncovered_source_component_indexes": [1],
                "uncovered_source_component_count": 1,
                "component_proxy_coverage_complete": False,
            },
        },
    }
    visual_artifacts = {
        "texture_material_evidence": {
            "status": "materialized_vertex_color_scene_evidence_present"
        },
        "blank_scene_checks": {"status": "not_applicable"},
    }

    qa = _build_digital_twin_fidelity_qa(
        generated_at="2026-06-14T00:00:00Z",
        mesh_info=mesh_info,
        collision_summary=collision_summary,
        visual_artifacts=visual_artifacts,
    )

    coverage = qa["gates"]["visible_objects_have_physics_coverage"]["evidence"]
    assert qa["machine_fidelity_audit_complete"] is False
    assert qa["robot_team_grade_fidelity_passed"] is False
    assert qa["gates"]["object_semantics_available"]["passed"] is True
    assert qa["gates"]["visible_objects_have_physics_coverage"]["passed"] is False
    assert "visible_object_0001_test_shelf" in coverage["missing_physics_object_ids"]
    assert coverage["missing_physics_objects"][0]["object_id"] == (
        "visible_object_0001_test_shelf"
    )
    assert coverage["missing_physics_objects"][0]["coverage_reason"] == (
        "source_component_uncovered_by_proxy_generation"
    )
    assert coverage["component_coverage_summary"]["component_mapping_status"] == "one_to_one"
    assert "visible_objects_without_physics_coverage" in qa["blockers"]


def test_mujoco_g1_command_covers_500_matrix_rows_with_fake_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_mujoco_backend(monkeypatch)
    capture_root, g1_root = _seed_fake_capture_and_g1(tmp_path, monkeypatch)
    matrix_rows = [
        {
            "scenario_eval_run_id": f"run-{index:04d}",
            "episode_id": f"episode-{index:04d}",
            "task_id": "walk_to_target",
            "scenario_id": f"scenario-{index % 10:02d}",
            "scenario_variation_instance_id": f"variation-{index % 11:02d}",
            "variation_name": f"variation_{index % 11:02d}",
        }
        for index in range(1, 501)
    ]
    matrix_path = tmp_path / "scenario_eval_matrix_500.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scenario_eval_matrix.v1",
                "status": "completed",
                "scenario_eval_run_count": len(matrix_rows),
                "runs": matrix_rows,
            }
        ),
        encoding="utf-8",
    )

    payload = run_mujoco_g1_simulator_command(
        capture_root=capture_root,
        g1_model_root=g1_root,
        output_dir=tmp_path / "mujoco-output-500",
        simulator_output_path=tmp_path / "mujoco_g1_simulator_output_500.json",
        scenario_eval_matrix_path=matrix_path,
        steps=1,
        render_frames=False,
        max_rendered_episodes=0,
    )

    assert payload["status"] == "completed"
    assert payload["scenario_eval_run_count"] == 500
    assert payload["attempt_count"] == 500
    assert payload["required_scenario_eval_run_count"] == 500
    assert payload["covered_scenario_eval_run_count"] == 500
    assert payload["missing_scenario_eval_run_count"] == 0
    assert payload["scenario_eval_run_coverage_complete"] is True
    assert payload["covered_scenario_eval_run_ids"][0] == "run-0001"
    assert payload["covered_scenario_eval_run_ids"][-1] == "run-0500"
    assert payload["rendered_episode_count"] == 0
    assert payload["ai_route_selection_used_at_runtime"] is False
    assert payload["deterministic_per_episode_spawn_target_seed_handling"] is True
    assert payload["task_success_summary"]["attempt_count"] == 500
    assert payload["task_success_summary"]["failed_attempt_count"] == 500
    assert payload["task_success_summary"]["failure_mode_counts"][
        "failure_target_not_reached"
    ] == 500
    assert payload["machine_trace_package_complete"] is True
    assert payload["robot_team_grade_package_complete"] is False
    closure = json.loads(
        Path(payload["artifact_paths"]["batch_closure_manifest"]).read_text(encoding="utf-8")
    )
    assert closure["attempt_count"] == 500
    assert closure["required_scenario_eval_run_count"] == 500
    assert closure["covered_scenario_eval_run_count"] == 500
    assert closure["missing_scenario_eval_run_count"] == 0
    assert closure["machine_trace_package_complete"] is True
    assert closure["robot_team_grade_package_complete"] is False
    assert closure["visual_coverage"]["all_required_runs_have_visual_recording"] is False
    visual_media_coverage = json.loads(
        Path(payload["artifact_paths"]["batch_visual_media_coverage"]).read_text(
            encoding="utf-8"
        )
    )
    assert visual_media_coverage["status"] == "incomplete"
    assert visual_media_coverage["required_scenario_eval_run_count"] == 500
    assert visual_media_coverage["missing_visual_media_run_count"] == 500
    assert visual_media_coverage["missing_visual_media_scenario_eval_run_ids"][0] == (
        "run-0001"
    )
    assert visual_media_coverage["missing_visual_media_scenario_eval_run_ids"][-1] == (
        "run-0500"
    )
    assert visual_media_coverage["all_required_runs_have_visual_recording"] is False
    assert visual_media_coverage["all_required_runs_have_robot_pov_video"] is False
    assert visual_media_coverage["all_required_runs_have_third_person_video"] is False


def test_simulator_command_artifacts_block_incomplete_required_run_coverage(
    tmp_path: Path,
) -> None:
    payload = {
        "required_scenario_eval_run_ids": ["run-a", "run-b"],
        "attempts": [
            {
                "attempt_id": "attempt-run-a",
                "scenario_eval_run_id": "run-a",
                "scenario_id": "scenario-a",
                "task_id": "walk_to_target",
                "status": "completed",
                "success": True,
            }
        ],
    }

    artifacts = build_simulator_command_artifacts(
        job_dir=tmp_path / "job",
        simulator="mujoco",
        simulator_output=payload,
        generated_at="2026-06-14T00:00:00Z",
    )

    trace = artifacts["normalized_attempt_trace"]
    manifest = artifacts["manifest"]
    assert trace["status"] == "blocked_incomplete_scenario_eval_run_coverage"
    assert trace["attempt_count_matches_matrix_count"] is False
    assert trace["scenario_eval_run_id_coverage_exact"] is False
    assert trace["missing_scenario_eval_run_ids"] == ["run-b"]
    assert trace["scenario_eval_run_coverage_complete"] is False
    assert manifest["status"] == "blocked_incomplete_scenario_eval_run_coverage"


def test_simulator_command_artifacts_normalize_task_metrics_and_failure_labels(
    tmp_path: Path,
) -> None:
    payload = {
        "required_scenario_eval_run_ids": ["run-a", "run-b"],
        "attempts": [
            {
                "attempt_id": "attempt-run-a",
                "scenario_eval_run_id": "run-a",
                "scenario_id": "scenario-a",
                "task_id": "walk_to_target",
                "status": "passed_task_criteria",
                "success": True,
                "task_success": True,
                "deterministic_seed": 101,
                "spawn_pose": [0.0, 0.0, 0.793],
                "target_pose": [1.0, 0.0, 0.793],
                "final_pose": [1.0, 0.0, 0.793],
                "metrics": {
                    "cycle_time_seconds": 1.0,
                    "endpoint_clean": True,
                    "goal_reached": True,
                },
                "task_outcome": {
                    "task_success": True,
                    "task_status": "passed",
                    "endpoint_clean": True,
                    "goal_reached": True,
                    "final_target_error_m": 0.0,
                    "max_path_deviation_m": 0.0,
                },
            },
            {
                "attempt_id": "attempt-run-b",
                "scenario_eval_run_id": "run-b",
                "scenario_id": "scenario-b",
                "task_id": "walk_to_target",
                "status": "failed_task_criteria",
                "success": False,
                "task_success": False,
                "failure_mode_ids": [
                    "failure_target_not_reached",
                    "failure_endpoint_not_clean",
                ],
                "failure_reason": "failure_target_not_reached,failure_endpoint_not_clean",
                "deterministic_seed": 202,
                "spawn_pose": [0.0, 0.0, 0.793],
                "target_pose": [1.0, 0.0, 0.793],
                "final_pose": [0.2, 0.0, 0.793],
                "metrics": {
                    "cycle_time_seconds": 1.0,
                    "endpoint_clean": False,
                    "goal_reached": False,
                },
                "task_outcome": {
                    "task_success": False,
                    "task_status": "failed_task_criteria",
                    "endpoint_clean": False,
                    "goal_reached": False,
                    "final_target_error_m": 0.8,
                    "max_path_deviation_m": 0.4,
                },
            },
        ],
    }

    artifacts = build_simulator_command_artifacts(
        job_dir=tmp_path / "job",
        simulator="mujoco",
        simulator_output=payload,
        generated_at="2026-06-14T00:00:00Z",
    )

    trace = artifacts["normalized_attempt_trace"]
    labels = artifacts["failure_labels"]
    visual_review = artifacts["visual_review_ledger"]
    prediction = artifacts["prediction_outcome_ledger"]
    manifest = artifacts["manifest"]

    assert trace["scenario_eval_run_coverage_complete"] is True
    assert trace["task_success_summary"]["successful_attempt_count"] == 1
    assert trace["task_success_summary"]["failed_attempt_count"] == 1
    assert trace["task_success_summary"]["failed_scenario_eval_run_ids"] == ["run-b"]
    assert trace["task_success_summary"]["failure_mode_counts"] == {
        "failure_endpoint_not_clean": 1,
        "failure_target_not_reached": 1,
    }
    assert trace["attempts"][1]["task_status"] == "failed_task_criteria"
    assert trace["attempts"][1]["task_outcome"]["final_target_error_m"] == 0.8
    assert labels["label_count"] == 1
    assert labels["labels"][0]["status"] == "deterministically_labeled_failure"
    assert labels["labels"][0]["label"] == "failure"
    assert labels["labels"][0]["primary_failure_mode"] == "failure_target_not_reached"
    assert labels["labels"][0]["criteria_results"]["success_criteria"][
        "goal_reached_within_tolerance"
    ] is False
    assert visual_review["review_count"] == 2
    assert visual_review["success_count"] == 1
    assert visual_review["failure_count"] == 1
    assert visual_review["records"][1]["decision"] == "failure"
    assert manifest["visual_review_coverage_complete"] is True
    assert labels["labels"][0]["task_outcome"]["endpoint_clean"] is False
    assert prediction["records"][1]["predicted_task_success"] is False
    assert prediction["records"][1]["predicted_final_target_error_m"] == 0.8
    assert manifest["task_success_rate"] == 0.5
