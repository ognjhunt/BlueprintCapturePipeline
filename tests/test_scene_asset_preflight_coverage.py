from __future__ import annotations

import builtins
import json
import math
import runpy
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from blueprint_pipeline import scene_asset_preflight as sap
from blueprint_pipeline.common import PipelineError


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {"site_identity": {"site_id": "site-1"}},
        },
    )
    _write_json(capture_root / "raw" / "manifest.json", {"scene_id": "scene-1", "capture_id": "capture-1"})
    return capture_root


def _write_ascii_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0.2",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_scene_asset_preflight_scalar_path_and_ascii_ply_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert sap._string_list(None) == []
    assert sap._string_list("one") == ["one"]
    assert sap._string_list(["one", "", "one", "two"]) == ["one", "two"]
    assert sap._string_list(5) == ["5"]

    existing = tmp_path / "asset.gltf"
    existing.write_text("{}", encoding="utf-8")
    assert sap._relative_to(tmp_path, existing) == "asset.gltf"
    assert sap._relative_if_file(tmp_path, existing) == "asset.gltf"
    assert sap._relative_if_file(tmp_path, tmp_path / "missing.obj") is None
    assert sap._sha_file(tmp_path / "missing.obj") is None
    assert sap._sha_file(existing)
    assert sap._asset_type_for_path(existing) == "gltf"
    assert sap._asset_type_for_path(tmp_path / "world.mjcf") == "mjcf"

    robot_xml = tmp_path / "robot.xml"
    robot_xml.write_text("<robot />", encoding="utf-8")
    assert sap._asset_type_for_path(robot_xml) == "urdf"
    mujoco_xml = tmp_path / "scene.xml"
    mujoco_xml.write_text("<mujoco />", encoding="utf-8")
    assert sap._asset_type_for_path(mujoco_xml) == "mjcf"
    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_text("<robot", encoding="utf-8")
    assert sap._asset_type_for_path(bad_xml) == "xml"
    assert sap._resolve_local_ref(existing, str(existing.resolve())) == existing.resolve()
    assert sap._walk_payload_strings(["a", {"nested": ["b"]}, b"bytes"]) == ["a", "b"]
    assert sap._finite_float_list(["1", object(), "3"]) is None
    assert sap._finite_float_list([1, math.inf, 3]) is None
    assert sap._bounds_from_points([["bad", 0, 0]]) is None
    assert sap._semantic_hints_from_names(["", "table", "table"], source="unit") == [
        {"label": "table", "source": "unit"}
    ]
    assert sap._percentile([], 0.5) is None
    assert sap._percentile([4.0], 0.5) == 4.0

    crlf_ply = tmp_path / "crlf.ply"
    crlf_ply.write_bytes(b"ply\r\nformat ascii 1.0\r\nend_header\r\n")
    assert sap._ply_header(crlf_ply)[1] > 0
    missing_header = tmp_path / "missing_header.ply"
    missing_header.write_text("ply\nformat ascii 1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="end_header"):
        sap._ply_header(missing_header)
    parsed = sap._parse_ply_header(
        [
            "",
            "format ascii 1.0",
            "element face 1",
            "property list uchar int vertex_indices",
            "property float confidence",
        ]
    )
    assert parsed["elements"][0]["properties"][0]["kind"] == "list"
    assert sap._ply_scalar_size("unknown") == 0
    assert sap._ply_record_size([{"kind": "scalar", "type": "float"}]) == 4
    assert sap._ply_record_size([{"kind": "list", "type": "float"}]) is None
    monkeypatch.setitem(sap._PLY_SCALAR_TYPES, "broken", ("", 0))
    with pytest.raises(ValueError, match="Unsupported"):
        sap._unpack_scalar(b"\x00", 0, "broken", "<")

    no_xyz = tmp_path / "no_xyz.ply"
    no_xyz.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float red\nend_header\n1\n",
        encoding="utf-8",
    )
    assert sap.inspect_ply_asset(no_xyz)["estimate_method"] == "ascii_header_only_no_xyz_properties"

    bad_points = tmp_path / "bad_points.ply"
    bad_points.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2\nbad 0 0\n1 2 3\n",
        encoding="utf-8",
    )
    lines, header_end = sap._ply_header(bad_points)
    assert sap._inspect_ascii_ply(bad_points, sap._parse_ply_header(lines), header_end)["sampled_point_count"] == 1

    no_points = tmp_path / "no_points.ply"
    no_points.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\nbad 0 0\n",
        encoding="utf-8",
    )
    assert sap.inspect_ply_asset(no_points)["estimate_method"] == "ascii_no_points_read"

    capped = tmp_path / "capped.ply"
    capped.write_text(
        "ply\nformat ascii 1.0\nelement vertex 200001\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        + ("0 0 0\n" * 200001),
        encoding="utf-8",
    )
    lines, header_end = sap._ply_header(capped)
    assert sap._inspect_ascii_ply(capped, sap._parse_ply_header(lines), header_end)["sampled_point_count"] == 200000


def test_scene_asset_preflight_binary_ply_and_usd_dependency_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert sap._inspect_binary_chunk_bounds(tmp_path / "missing.ply", {"elements": []}, 0) is None
    assert sap._inspect_binary_chunk_bounds(
        tmp_path / "missing.ply",
        {"elements": [{"name": "vertex", "count": 1, "properties": []}]},
        0,
    ) is None
    assert sap._inspect_binary_chunk_bounds(
        tmp_path / "missing.ply",
        {"elements": [{"name": "chunk", "count": 1, "properties": [{"kind": "scalar", "type": "float", "name": "min_x"}]}]},
        0,
    ) is None
    props = [
        {"kind": "scalar", "type": "float", "name": name}
        for name in ("min_x", "min_y", "min_z", "max_x", "max_y", "max_z")
    ]
    list_props = [*props[:-1], {"kind": "list", "type": "float", "name": "max_z"}]
    assert sap._inspect_binary_chunk_bounds(
        tmp_path / "missing.ply",
        {"elements": [{"name": "chunk", "count": 1, "properties": list_props}]},
        0,
    ) is None

    partial = tmp_path / "partial_binary.ply"
    partial.write_bytes(b"header")
    parsed = {"format": "binary_little_endian", "elements": [{"name": "chunk", "count": 1, "properties": props}]}
    assert sap._inspect_binary_chunk_bounds(partial, parsed, len(b"header")) is None
    success = tmp_path / "chunk_binary.ply"
    success.write_bytes(b"header" + struct.pack("<ffffff", 0.0, 0.0, -0.1, 1.0, 2.0, 3.0))
    assert sap._inspect_binary_chunk_bounds(success, parsed, len(b"header"))["sampled_chunk_count"] == 1

    binary_fallback = tmp_path / "binary_fallback.ply"
    binary_fallback.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n\x00")
    assert sap.inspect_ply_asset(binary_fallback)["estimate_method"] == "binary_header_only_no_decoded_xyz"
    unsupported = tmp_path / "unsupported_format.ply"
    unsupported.write_text("ply\nformat odd 1.0\nend_header\n", encoding="utf-8")
    assert sap.inspect_ply_asset(unsupported)["estimate_method"] == "unsupported_ply_format"

    usd = tmp_path / "scene.usda"
    tex = tmp_path / "texture.png"
    tex.write_bytes(b"png")
    usd.write_text(
        "\n".join(
            [
                "#usda 1.0",
                "payload = @payload.usda@",
                "asset material = @OmniPBR.mdl@",
                "asset diffuseTexture = @texture.png@",
                "asset diffuseTexture = @texture.png@",
                "asset generic = @thing.bin@",
            ]
        ),
        encoding="utf-8",
    )
    deps = sap._extract_usd_dependencies(usd, usd.read_text(encoding="utf-8"))
    assert {dep["relationship"] for dep in deps} >= {
        "payload",
        "owner_system_material_library",
        "texture_or_material_asset",
        "asset_reference",
    }
    assert len([dep for dep in deps if dep["ref"] == "texture.png"]) == 1
    assert len(sap._dedupe_dependencies([deps[0], deps[0]])) == 1
    assert sap._dependency_relationship_for_path("a.png", "fallback") == "texture_or_material_asset"
    assert sap._dependency_relationship_for_path("OmniPBR.mdl", "fallback") == "owner_system_material_library"
    assert sap._dependency_relationship_for_path("layer.usd", "fallback") == "usd_layer_or_reference"
    assert sap._dependency_relationship_for_path("asset.bin", "fallback") == "fallback"

    original_import = builtins.__import__

    def fail_pxr_import(name, *args, **kwargs):
        if name == "pxr":
            raise ImportError("missing fake pxr")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_pxr_import)
    assert sap._extract_openusd_dependencies(usd) == []
    monkeypatch.setattr(builtins, "__import__", original_import)

    failing_pxr = ModuleType("pxr")
    failing_pxr.UsdUtils = SimpleNamespace(ComputeAllDependencies=lambda _path: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setitem(sys.modules, "pxr", failing_pxr)
    assert sap._extract_openusd_dependencies(usd) == []

    layer = tmp_path / "layer.usda"
    layer.write_text("#usda 1.0", encoding="utf-8")

    class FakeUsdUtils:
        @staticmethod
        def ComputeAllDependencies(_path: str):
            return (
                [SimpleNamespace(realPath=str(usd.resolve())), SimpleNamespace(realPath=str(layer.resolve())), SimpleNamespace(identifier="")],
                ["texture.png", "", "OmniPBR.mdl", "asset.bin"],
                ["missing.usda", "", "lost.bin", "https://assets.example/remote.usd"],
            )

    pxr = ModuleType("pxr")
    pxr.UsdUtils = FakeUsdUtils
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    openusd_deps = sap._extract_openusd_dependencies(usd)
    assert {dep["relationship"] for dep in openusd_deps} >= {
        "usd_layer_or_reference",
        "openusd_asset_dependency",
        "owner_system_material_library",
        "unresolved_openusd_dependency",
    }
    assert any(dep.get("warning") == "unresolved_openusd_dependency" for dep in openusd_deps)


def test_scene_asset_preflight_fake_openusd_and_gltf_trimesh_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_find_spec = sap.importlib.util.find_spec
    usd = tmp_path / "scene.usda"
    usd.write_text("#usda 1.0", encoding="utf-8")

    monkeypatch.setattr(sap.importlib.util, "find_spec", lambda name: None if name == "pxr" else original_find_spec(name))
    assert sap._inspect_usd_with_pxr(usd) is None

    monkeypatch.setattr(sap.importlib.util, "find_spec", lambda name: object() if name == "pxr" else original_find_spec(name))
    original_import = builtins.__import__

    def fail_pxr_import(name, *args, **kwargs):
        if name == "pxr":
            raise ImportError("missing fake pxr")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_pxr_import)
    assert sap._inspect_usd_with_pxr(usd) is None

    def fake_pxr_import(name, *args, **kwargs):
        if name == "pxr":
            return sys.modules["pxr"]
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_pxr_import)

    class RaisingStage:
        @staticmethod
        def Open(_path: str):
            raise RuntimeError("open failed")

    pxr_raising = ModuleType("pxr")
    pxr_raising.Usd = SimpleNamespace(Stage=RaisingStage)
    pxr_raising.UsdGeom = SimpleNamespace()
    pxr_raising.UsdPhysics = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pxr", pxr_raising)
    assert sap._inspect_usd_with_pxr(usd) is None

    class NoneStage:
        @staticmethod
        def Open(_path: str):
            return None

    pxr_none = ModuleType("pxr")
    pxr_none.Usd = SimpleNamespace(Stage=NoneStage)
    pxr_none.UsdGeom = SimpleNamespace()
    pxr_none.UsdPhysics = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pxr", pxr_none)
    assert sap._inspect_usd_with_pxr(usd)["status"] == "blocked_openusd_stage_open_failed"

    class FakeMesh:
        pass

    class FakeCollisionAPI:
        pass

    class FakeRigidBodyAPI:
        pass

    class FakePrim:
        def __init__(self, *, mesh: bool = False, type_name: str = "", apis: tuple[type, ...] = (), refs: bool = False) -> None:
            self.mesh = mesh
            self.type_name = type_name
            self.apis = apis
            self.refs = refs

        def IsA(self, cls: type) -> bool:
            return self.mesh and cls is FakeMesh

        def GetTypeName(self) -> str:
            return self.type_name

        def HasAPI(self, cls: type) -> bool:
            return cls in self.apis

        def HasAuthoredReferences(self) -> bool:
            return self.refs

    class FakeStage:
        def Traverse(self):
            return [
                FakePrim(mesh=True, type_name="Material", apis=(FakeCollisionAPI, FakeRigidBodyAPI), refs=True),
                FakePrim(),
            ]

        def GetPseudoRoot(self):
            return object()

    class StageFactory:
        @staticmethod
        def Open(_path: str):
            return FakeStage()

    class FakeRange:
        def IsEmpty(self) -> bool:
            return False

        def GetMin(self):
            return [-1.0, -2.0, 0.0]

        def GetMax(self):
            return [1.0, 2.0, 3.0]

    class FakeBound:
        def ComputeAlignedRange(self):
            return FakeRange()

    class FakeBBoxCache:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def ComputeWorldBound(self, _root):
            return FakeBound()

    class FakeTimeCode:
        @staticmethod
        def Default():
            return "default"

    fake_usd = SimpleNamespace(Stage=StageFactory, TimeCode=FakeTimeCode)
    fake_usd_geom = SimpleNamespace(
        Mesh=FakeMesh,
        BBoxCache=FakeBBoxCache,
        Tokens=SimpleNamespace(default_="default", render="render", proxy="proxy"),
        GetStageMetersPerUnit=lambda _stage: 1.0,
        GetStageUpAxis=lambda _stage: "Z",
    )
    fake_usd_physics = SimpleNamespace(CollisionAPI=FakeCollisionAPI, RigidBodyAPI=FakeRigidBodyAPI)
    pxr_success = ModuleType("pxr")
    pxr_success.Usd = fake_usd
    pxr_success.UsdGeom = fake_usd_geom
    pxr_success.UsdPhysics = fake_usd_physics
    monkeypatch.setitem(sys.modules, "pxr", pxr_success)
    result = sap._inspect_usd_with_pxr(usd)
    assert result["bounds"]["max"] == [1.0, 2.0, 3.0]
    assert result["prim_counts"]["physics_collision_api"] == 1
    assert result["isaac_usd_collision_verified"] is True

    class RaisingBBoxCache(FakeBBoxCache):
        def ComputeWorldBound(self, _root):
            raise RuntimeError("bounds failed")

    pxr_success.UsdGeom = SimpleNamespace(
        Mesh=FakeMesh,
        BBoxCache=RaisingBBoxCache,
        Tokens=fake_usd_geom.Tokens,
        GetStageMetersPerUnit=lambda _stage: 1.0,
        GetStageUpAxis=lambda _stage: "Z",
    )
    assert sap._inspect_usd_with_pxr(usd)["bounds"] is None

    text_usd = tmp_path / "pxr_text.usda"
    text_usd.write_text('def Mesh "Collider" { prepend references = @texture.png@ }', encoding="utf-8")
    monkeypatch.setattr(
        sap,
        "_inspect_usd_with_pxr",
        lambda _path: {
            "asset_type": "usd",
            "path": str(text_usd.resolve()),
            "status": "inspected_with_openusd",
            "isaac_usd_collision_verified": True,
            "collision_verification_status": "verified_by_openusd_api",
        },
    )
    assert sap.inspect_usd_asset(text_usd)["collision_evidence"]["real_collider_proven"] is True

    bad_glb = tmp_path / "bad.glb"
    bad_glb.write_bytes(b"bad")
    with pytest.raises(ValueError, match="magic"):
        sap._gltf_from_glb(bad_glb)
    missing_chunk = tmp_path / "missing_chunk.glb"
    missing_chunk.write_bytes(b"glTF" + struct.pack("<II", 2, 12))
    with pytest.raises(ValueError, match="JSON chunk"):
        sap._gltf_from_glb(missing_chunk)
    wrong_chunk = tmp_path / "wrong_chunk.glb"
    wrong_chunk.write_bytes(b"glTF" + struct.pack("<II", 2, 20) + struct.pack("<II", 0, 0))
    with pytest.raises(ValueError, match="first chunk"):
        sap._gltf_from_glb(wrong_chunk)

    payload = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "mesh.bin"}],
        "images": [{"uri": "texture.png"}],
        "nodes": [{"name": "visual_node"}],
        "meshes": [{"name": "visual_mesh", "primitives": [{"attributes": {"POSITION": "bad"}}]}, "not-a-mesh"],
        "accessors": [{"type": "VEC3", "min": [-1, -1, 0], "max": [1, 1, 2]}],
    }
    gltf = tmp_path / "visual.gltf"
    gltf.write_text(json.dumps(payload), encoding="utf-8")
    assert sap.inspect_gltf_asset(gltf)["asset_type"] == "gltf"
    assert len(sap._gltf_dependencies(gltf, payload)) == 2
    assert sap._bounds_from_min_max(["bad", 0, 0], [1, 1, 1]) is None
    assert sap._bounds_from_min_max([2, 0, 0], [1, 1, 1]) is None
    assert sap._gltf_position_accessor_indexes({"meshes": ["bad", {"primitives": ["bad", {"attributes": {"POSITION": "bad"}}]}]}) == []
    assert sap._gltf_accessor_bounds({}) is None
    assert sap._gltf_accessor_bounds({"meshes": [{"primitives": [{"attributes": {"POSITION": 3}}]}], "accessors": [{"type": "VEC3"}]}) is None
    assert sap._gltf_accessor_bounds({"accessors": [{"type": "VEC3", "min": ["bad", 0, 0], "max": [1, 1, 1]}]}) is None
    monkeypatch.setattr(sap, "_bounds_from_min_max", lambda *_args: {"bounds": {"min": [0, 0], "max": [1, 1]}})
    assert sap._gltf_accessor_bounds({"accessors": [{"type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 1]}]}) is None

    monkeypatch.setattr(sap.importlib.util, "find_spec", lambda name: None if name == "trimesh" else original_find_spec(name))
    assert sap._trimesh_gltf_bounds(gltf) is None
    monkeypatch.setattr(sap.importlib.util, "find_spec", lambda name: object() if name == "trimesh" else original_find_spec(name))

    def fail_trimesh_import(name, *args, **kwargs):
        if name == "trimesh":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_trimesh_import)
    assert sap._trimesh_gltf_bounds(gltf) is None
    monkeypatch.setattr(builtins, "__import__", original_import)

    fake_trimesh = ModuleType("trimesh")
    fake_trimesh.load = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("load failed"))
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)
    assert sap._trimesh_gltf_bounds(gltf) is None
    fake_trimesh.load = lambda *_args, **_kwargs: SimpleNamespace(bounds=None)
    assert sap._trimesh_gltf_bounds(gltf) is None
    fake_trimesh.load = lambda *_args, **_kwargs: SimpleNamespace(bounds=[["bad"], [1, 2, 3]])
    assert sap._trimesh_gltf_bounds(gltf) is None
    fake_trimesh.load = lambda *_args, **_kwargs: SimpleNamespace(bounds=[[0, 0], [1, 1, 1]])
    assert sap._trimesh_gltf_bounds(gltf) is None
    fake_trimesh.load = lambda *_args, **_kwargs: SimpleNamespace(bounds=[[0, 0, math.inf], [1, 1, 1]])
    assert sap._trimesh_gltf_bounds(gltf) is None
    fake_trimesh.load = lambda *_args, **_kwargs: SimpleNamespace(bounds=[[0, 0, 0], [1, 1, 1]])
    assert sap._trimesh_gltf_bounds(gltf)["estimate_method"] == "trimesh_scene_bounds"


def test_scene_asset_preflight_obj_xml_discovery_and_queue_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj = tmp_path / "collider.obj"
    obj.write_text(
        "\n".join(["o collider_part", "v bad 0 0", "v 0 0 0", "v 1 1 1", "mtllib missing.mtl", ""]),
        encoding="utf-8",
    )
    obj_result = sap.inspect_obj_asset(obj)
    assert obj_result["vertex_count_sampled"] == 2
    assert obj_result["dependencies"][0]["relationship"] == "material_library"

    broken_xml = tmp_path / "broken.urdf"
    broken_xml.write_text("<robot", encoding="utf-8")
    assert sap._inspect_xml_asset(broken_xml)["status"] == "blocked_xml_parse_failed"
    mjcf = tmp_path / "scene.xml"
    mjcf.write_text('<mujoco><worldbody><geom name="floor" type="plane" /></worldbody></mujoco>', encoding="utf-8")
    assert sap._inspect_xml_asset(mjcf)["collision_evidence"]["real_collider_proven"] is True

    capture_root = _build_capture_root(tmp_path)
    asset = capture_root / "raw" / "scene.obj"
    asset.write_text("v 0 0 0\n", encoding="utf-8")
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "obj_scene_path": "raw/scene.obj",
            "assets": {"splats": {"ply_urls": {"main": "raw/scene.obj"}, "usd_urls": {"usd": "raw/scene.obj"}}},
            "nested": {"candidate": "raw/scene.obj"},
        },
    )
    candidates = sap._candidate_paths_from_payload(capture_root, json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8")))
    assert candidates
    assert sap.discover_scene_assets(capture_root, explicit_assets=[asset, asset]) == [asset.resolve()]
    assert sap.inspect_scene_asset(tmp_path / "unsupported.txt")["status"] == "unsupported_asset_type"

    dep_scene = tmp_path / "dep_scene.usd"
    dep_scene.write_text("#usda", encoding="utf-8")
    dep_supported = tmp_path / "dep.obj"
    dep_supported.write_text("v 0 0 0\n", encoding="utf-8")
    dep_unsupported = tmp_path / "dep.txt"
    dep_unsupported.write_text("note", encoding="utf-8")

    def fake_inspect(path: Path) -> dict[str, object]:
        if path == dep_scene.resolve():
            return {
                "asset_type": "usd",
                "path": str(path),
                "dependencies": [
                    "not-a-mapping",
                    {"local_path": str(dep_unsupported.resolve()), "exists_local": True},
                    {"local_path": str(dep_supported.resolve()), "exists_local": True},
                ],
            }
        return {"asset_type": path.suffix.lstrip("."), "path": str(path), "dependencies": []}

    monkeypatch.setattr(sap, "inspect_scene_asset", fake_inspect)
    inspected = sap._inspect_scene_assets_with_local_dependencies([dep_scene, dep_scene])
    assert [Path(item["path"]).name for item in inspected] == ["dep_scene.usd", "dep.obj"]

    _write_json(
        capture_root / "pipeline" / "evaluation_prep" / "object_geometry_manifest.json",
        {"objects": ["bad", {"object_id": "bad-bounds", "placement_bbox": {"center": ["bad"], "extents": [1, 1, 1]}}]},
    )
    assert sap._object_geometry_proxy_obstacles(capture_root / "pipeline") == []
    primitive = sap._proxy_primitive_from_frame(
        {"bounds": {"min": [0, 0, -0.2], "max": [1, 1, 1]}, "floor_z_estimate": "bad"}
    )
    assert primitive["floor_z"] == -0.2


def test_scene_asset_preflight_blocked_outputs_and_main_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture_root = _build_capture_root(tmp_path)
    result = sap.build_scene_asset_preflight(capture_root=capture_root)
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    preflight = json.loads((automation_dir / "scene_asset_preflight.json").read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert {"missing_local_scene_asset", "missing_scene_frame_estimate"} <= set(preflight["blockers"])

    ply = tmp_path / "scene.ply"
    _write_ascii_ply(ply)
    assert sap.main(["--capture-root", str(capture_root), "--scene-asset", str(ply)]) == 0
    assert "scorecard=" in capsys.readouterr().out

    monkeypatch.setattr(sap, "build_scene_asset_preflight", lambda **_kwargs: (_ for _ in ()).throw(PipelineError("nope")))
    assert sap.main(["--capture-root", str(capture_root)]) == 1
    assert "FAILED: nope" in capsys.readouterr().out

    module_capture_root = _build_capture_root(tmp_path / "module")
    module_ply = tmp_path / "module_scene.ply"
    _write_ascii_ply(module_ply)
    monkeypatch.setattr(sys, "argv", ["scene_asset_preflight", "--capture-root", str(module_capture_root), "--scene-asset", str(module_ply)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("blueprint_pipeline.scene_asset_preflight", run_name="__main__")
    assert exc.value.code == 0
