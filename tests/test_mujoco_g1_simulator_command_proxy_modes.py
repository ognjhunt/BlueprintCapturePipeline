from __future__ import annotations

import math
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import mujoco_g1_simulator_command as mg


class Component:
    def __init__(self, bounds, *, vertices=None) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        self.vertices = np.asarray(vertices or [[0.0, 0.0, 0.0]], dtype=float)


class Mesh:
    def __init__(self, components) -> None:
        self._components = list(components)

    def split(self, only_watertight=False):
        return list(self._components)


def _component(bounds=None) -> Component:
    return Component(bounds or [[0.0, 0.0, 0.2], [2.0, 1.0, 1.4]])


def test_collision_proxy_mode_aabb_preserves_unoriented_default() -> None:
    proxies, summary = mg._collision_proxy_geoms_from_mesh(Mesh([_component()]), max_proxies=10)

    assert len(proxies) == 1
    assert "quat" not in proxies[0]
    assert summary["collision_proxy_mode"] == "aabb"
    assert (
        summary["generation_method"]
        == "component_aabb_obstacle_proxies_excluding_floor_overhead_and_scene_shell"
    )


def test_collision_proxy_mode_obb_uses_oriented_bounds(monkeypatch) -> None:
    angle = math.radians(45.0)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0, 1.25],
            [math.sin(angle), math.cos(angle), 0.0, -0.5],
            [0.0, 0.0, 1.0, 0.8],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    to_origin = np.linalg.inv(rotation)

    def oriented_bounds(_component):
        return to_origin, np.array([1.0, 0.4, 0.6], dtype=float)

    fake_trimesh = types.SimpleNamespace(bounds=types.SimpleNamespace(oriented_bounds=oriented_bounds))
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)

    aabb_proxies, _ = mg._collision_proxy_geoms_from_mesh(
        Mesh([_component([[0.0, 0.0, 0.2], [4.0, 2.0, 1.4]])]),
        max_proxies=10,
        mode="aabb",
    )
    proxies, summary = mg._collision_proxy_geoms_from_mesh(
        Mesh([_component([[0.0, 0.0, 0.2], [4.0, 2.0, 1.4]])]),
        max_proxies=10,
        mode="obb",
    )

    assert len(proxies) == 1
    assert len(proxies[0]["quat"]) == 4
    assert np.prod(proxies[0]["size"]) < np.prod(aabb_proxies[0]["size"])
    assert summary["collision_proxy_mode"] == "obb"
    assert "obb" in summary["generation_method"]
    assert summary["obb_fallback_component_indexes"] == []


def test_collision_proxy_mode_obb_falls_back_per_component(monkeypatch) -> None:
    def oriented_bounds(_component):
        raise RuntimeError("bad component")

    fake_trimesh = types.SimpleNamespace(bounds=types.SimpleNamespace(oriented_bounds=oriented_bounds))
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)

    proxies, summary = mg._collision_proxy_geoms_from_mesh(
        Mesh([_component()]),
        max_proxies=10,
        mode="obb",
    )

    assert len(proxies) == 1
    assert "quat" not in proxies[0]
    assert summary["collision_proxy_mode"] == "obb"
    assert summary["obb_fallback_component_indexes"] == [0]


def test_collision_proxy_mode_convex_unavailable_falls_back_to_aabb(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "coacd", None)
    fake_trimesh = types.SimpleNamespace(interfaces=None, decomposition=None)
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)

    proxies, summary = mg._collision_proxy_geoms_from_mesh(
        Mesh([_component()]),
        max_proxies=10,
        mode="convex",
    )

    assert len(proxies) == 1
    assert "quat" not in proxies[0]
    assert "convex_hull_vertices" not in proxies[0]
    assert summary["collision_proxy_mode"] == "convex"
    assert summary["convex_decomposition_status"] == "unavailable_fell_back_to_aabb"


def test_collision_proxy_mode_convex_available_records_decomposition_vertices(monkeypatch) -> None:
    class FakeCoacdMesh:
        def __init__(self, vertices, faces) -> None:
            self.vertices = vertices
            self.faces = faces

    fake_coacd = types.SimpleNamespace(
        Mesh=FakeCoacdMesh,
        run_coacd=lambda mesh: [(mesh.vertices[:3], mesh.faces)],
    )
    monkeypatch.setitem(sys.modules, "coacd", fake_coacd)
    monkeypatch.setitem(sys.modules, "trimesh", types.SimpleNamespace(bounds=types.SimpleNamespace()))
    component = Component(
        [[0.0, 0.0, 0.2], [2.0, 1.0, 1.4]],
        vertices=[
            [0.0, 0.0, 0.2],
            [2.0, 0.0, 0.2],
            [0.0, 1.0, 1.4],
            [2.0, 1.0, 1.4],
        ],
    )
    component.faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)

    proxies, summary = mg._collision_proxy_geoms_from_mesh(
        Mesh([component]),
        max_proxies=10,
        mode="convex",
    )

    assert len(proxies) == 1
    assert proxies[0]["convex_hull_vertices"] == [
        [0.0, 0.0, 0.2],
        [2.0, 0.0, 0.2],
        [0.0, 1.0, 1.4],
    ]
    assert summary["collision_proxy_mode"] == "convex"
    assert summary["convex_decomposition_backend"] == "coacd"
    assert summary["convex_decomposition_status"] == "generated"
    assert summary["convex_decomposition_generated_proxy_count"] == 1


def test_write_mjcf_wrapper_emits_quat_only_for_oriented_proxy(tmp_path: Path) -> None:
    scene_obj = tmp_path / "scene.obj"
    scene_obj.write_text("v 0 0 0\nf 1 1 1\n", encoding="utf-8")
    g1_xml = tmp_path / "g1.xml"
    g1_xml.write_text("<mujoco><worldbody/></mujoco>", encoding="utf-8")
    wrapper = tmp_path / "wrapper.xml"

    mg._write_mjcf_wrapper(
        scene_obj,
        g1_xml,
        wrapper,
        collision_proxies=[
            {
                "name": "oriented",
                "pos": [0.0, 0.0, 0.5],
                "size": [0.2, 0.3, 0.4],
                "quat": [0.9238795, 0.0, 0.0, 0.3826834],
            },
            {"name": "axis_aligned", "pos": [1.0, 0.0, 0.5], "size": [0.2, 0.3, 0.4]},
        ],
    )

    root = ET.fromstring(wrapper.read_text(encoding="utf-8"))
    geoms = {
        geom.attrib["name"]: geom.attrib
        for geom in root.findall(".//geom")
        if geom.attrib.get("name", "").startswith("blueprint_collision_proxy")
    }
    oriented = next(attrs for name, attrs in geoms.items() if "oriented" in name)
    axis_aligned = next(attrs for name, attrs in geoms.items() if "axis_aligned" in name)
    assert "quat" in oriented
    assert "quat" not in axis_aligned
    assert len(geoms) == 2


def test_collision_proxy_mode_argparse_choices() -> None:
    parser = mg.build_arg_parser()

    parsed = parser.parse_args([
        "--capture-root",
        "capture",
        "--collision-proxy-mode",
        "obb",
    ])
    assert parsed.collision_proxy_mode == "obb"
    with pytest.raises(SystemExit):
        parser.parse_args(["--capture-root", "capture", "--collision-proxy-mode", "invalid"])


def test_collision_proxy_mode_main_forwards_mode(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "status": "completed",
            "simulator_backend": "mujoco",
            "unitree_g1_asset_spawned": True,
            "simulator_execution_proven": False,
            "attempt_count": 0,
            "scenario_eval_run_count": 0,
            "missing_scenario_eval_run_count": 0,
            "collision_geometry_loaded": False,
            "collision_dynamics_validated": False,
            "robot_team_handoff_ready": False,
            "output_dir": str(tmp_path),
        }

    monkeypatch.setattr(mg, "run_mujoco_g1_simulator_command", fake_run)

    assert mg.main(["--capture-root", str(tmp_path), "--collision-proxy-mode", "obb"]) == 0
    assert captured["collision_proxy_mode"] == "obb"
