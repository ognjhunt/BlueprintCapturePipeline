from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.common import PipelineError
from blueprint_pipeline import object_geometry_stage as ogs


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "pipeline").mkdir(parents=True, exist_ok=True)
    return root


def _object_entry(object_id: str = "obj-1") -> dict[str, object]:
    return {
        "id": object_id,
        "label": "box",
        "boundingBox": {
            "center": [0.0, 0.0, 0.5],
            "extents": [0.5, 0.4, 1.0],
            "orientationQuaternion": [1.0, 0.0, 0.0, 0.0],
        },
    }


def test_object_geometry_small_helper_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert ogs._safe_float(object(), default=1.25) == 1.25
    assert ogs._string_list("x", ["x", " y "], {"z"}, 4, None) == ["x", "y", "z", "4"]

    gray_path = tmp_path / "gray.png"
    ogs._write_simple_png(gray_path, width=4, height=3, grayscale=300)
    assert ogs._png_dimensions(gray_path) == (4, 3)
    assert ogs._png_dimensions(tmp_path / "missing.png") == (256, 256)
    invalid_png = tmp_path / "invalid.png"
    invalid_png.write_text("not-png", encoding="utf-8")
    assert ogs._png_dimensions(invalid_png) == (256, 256)

    bbox = ogs._normalize_bbox({"boundingBox": "bad"})
    assert bbox["center"] == [0.0, 0.0, 0.0]
    assert bbox["axes"][2] == [0.0, 0.0, 1.0]
    bbox = ogs._normalize_bbox({"boundingBox": {"axes": [[1], [0, 1], [0, 0, 1]]}})
    assert bbox["axes"][0] == [1.0, 0.0, 0.0]
    payload_path = tmp_path / "list.json"
    _write_json(payload_path, [])
    assert ogs._optional_json(payload_path) == {}

    root = _capture_root(tmp_path)
    _write_json(root / "raw" / "object_index.json", [_object_entry()])
    _write_json(
        root / "raw" / "object_index_build_report.json",
        {
            "empty_index_cause": "no detections",
            "runtime_preflight": {"backends": {"optional_backend": {"support_level": "optional"}}},
            "backend_summary": {
                "providers": [
                    "skip",
                    {"backend": "optional_backend", "reason": "missing_weights"},
                    {"backend": "", "reason": "missing"},
                    {"backend": "required_backend", "reason": ""},
                    {"backend": "required_backend", "reason": "weights_missing"},
                ]
            },
        },
    )
    summary = ogs._object_index_summary(root)
    assert summary["present"]
    assert summary["has_entries"]
    assert summary["empty_index_cause"] == "no detections"
    assert summary["runtime_blockers"] == [
        "object_index_backend:required_backend:weights_missing"
    ]

    empty_index_root = _capture_root(tmp_path / "empty-index")
    _write_json(empty_index_root / "raw" / "object_index.json", [])
    empty_manifest = ogs.build_missing_object_geometry_manifest(
        capture_root=empty_index_root,
        provider_name="manual",
    )
    assert empty_manifest["status"] == "empty_object_index"

    missing_root = _capture_root(tmp_path / "missing")
    missing_manifest = ogs.build_missing_object_geometry_manifest(
        capture_root=missing_root,
        provider_name="manual",
    )
    assert missing_manifest["status"] == "missing_object_index"
    assert missing_manifest["objects"] == []

    monkeypatch.setattr(ogs, "_load_object_entries", lambda capture_root: ([_object_entry()], root / "raw" / "object_index.json"))
    bad_manifest = tmp_path / "bad-manifest.json"
    _write_json(bad_manifest, [])
    monkeypatch.setattr(ogs, "run_object_geometry_stage", lambda **kwargs: {"manifest_path": str(bad_manifest)})
    with pytest.raises(PipelineError, match="not a JSON object"):
        ogs.resolve_object_geometry_manifest(capture_root=root)
    monkeypatch.setattr(ogs, "_load_object_entries", lambda capture_root: ([], root / "raw" / "object_index.json"))
    assert ogs.resolve_object_geometry_manifest(capture_root=root)["status"] in {
        "missing_object_index",
        "empty_object_index",
    }
    good_manifest = tmp_path / "good-manifest.json"
    _write_json(good_manifest, {"objects": []})
    monkeypatch.setattr(ogs, "_load_object_entries", lambda capture_root: ([_object_entry()], root / "raw" / "object_index.json"))
    monkeypatch.setattr(ogs, "run_object_geometry_stage", lambda **kwargs: {"manifest_path": str(good_manifest)})
    assert ogs.resolve_object_geometry_manifest(capture_root=root) == {"objects": []}


def test_object_entry_file_crop_and_mesh_helper_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _capture_root(tmp_path)
    index_path = root / "raw" / "object_index.json"
    _write_json(
        index_path,
        [
            "bad",
            {"label": "missing-id"},
            _object_entry("obj-list"),
        ],
    )
    entries, resolved_index = ogs._load_object_entries(root)
    assert resolved_index == index_path
    assert [entry["object_id"] for entry in entries] == ["obj-list"]
    _write_json(root / "raw" / "arkit" / "objects" / "index.json", {"objects": "bad"})
    (root / "raw" / "object_index.json").unlink()
    assert ogs._load_object_entries(root)[0] == []
    _write_json(root / "raw" / "object_index.json", "bad")
    assert ogs._load_object_entries(root)[0] == []

    mesh_file = root / "raw" / "mesh.ply"
    mesh_file.write_text("ply\n", encoding="utf-8")
    assert ogs._resolve_object_file(None, index_path=index_path, capture_root=root) is None
    assert ogs._resolve_object_file(str(mesh_file), index_path=index_path, capture_root=root) == mesh_file
    assert ogs._resolve_object_file("mesh.ply", index_path=index_path, capture_root=root) == mesh_file
    assert ogs._resolve_object_file("missing.ply", index_path=index_path, capture_root=root) is None

    crop = root / "crop.png"
    ogs._write_simple_png(crop, width=5, height=6)
    ignored = root / "crop.txt"
    ignored.write_text("x", encoding="utf-8")
    crops = ogs._resolve_real_crop_paths(
        {"reference_crop": "crop.png", "all_crops": [str(crop), str(ignored)]},
        capture_root=root,
    )
    assert crops == [crop.resolve()]

    assert ogs._quaternion_matrix([0.0, 0.0, 0.0, 0.0]) == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    real_np = ogs.np
    monkeypatch.setattr(ogs, "np", None)
    with pytest.raises(PipelineError, match="numpy is required"):
        ogs._rotate_points([[1.0, 0.0, 0.0]], [1.0, 0.0, 0.0, 0.0])
    assert ogs._normalize_mesh_to_local("mesh", {"center": [0, 0, 0]}) == "mesh"
    assert ogs._mesh_bounds(None)["extents"] == [0.25, 0.25, 0.25]
    monkeypatch.setattr(ogs, "np", real_np)

    real_trimesh = ogs.trimesh
    monkeypatch.setattr(ogs, "trimesh", None)
    assert ogs._load_mesh_or_points(entry=entries[0], index_path=index_path, capture_root=root) == (
        None,
        "geometry_lib_unavailable",
    )
    assert ogs._box_mesh_from_bbox(entries[0]["boundingBox"]) is None
    assert ogs._mesh_components(None) == []
    monkeypatch.setattr(ogs, "trimesh", real_trimesh)

    output = tmp_path / "mesh.glb"
    assert ogs._export_mesh(None, output) == str(output)
    assert output.read_bytes() == b"glb"


def test_object_geometry_fake_trimesh_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _capture_root(tmp_path)
    index_path = root / "raw" / "object_index.json"
    mesh_path = root / "raw" / "mesh.obj"
    mesh_path.write_text("mesh", encoding="utf-8")
    entry = _object_entry()
    entry["pointCloudFile"] = "mesh.obj"

    class FakeMesh:
        def __init__(self, vertex_count: int = 8, fail_hull: bool = False) -> None:
            self.vertices = ogs.np.asarray(
                [[float(idx), float(idx % 2), float(idx % 3)] for idx in range(vertex_count)],
                dtype=float,
            )
            self.bounds = ogs.np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
            self.faces = ogs.np.asarray([[0, 1, 2]], dtype=int)
            self.face_normals = ogs.np.asarray([[0.0, 0.0, 1.0]], dtype=float)
            self.triangles_center = ogs.np.asarray([[0.5, 0.5, 0.5]], dtype=float)
            self.area_faces = ogs.np.asarray([0.25], dtype=float)
            self.fail_hull = fail_hull
            self.bounding_box = SimpleNamespace(to_mesh=lambda: FakeMesh())

        def copy(self) -> "FakeMesh":
            return FakeMesh()

        @property
        def convex_hull(self) -> "FakeMesh":
            if self.fail_hull:
                raise RuntimeError("no hull")
            return FakeMesh()

        def split(self, only_watertight: bool = False) -> list["FakeMesh"]:
            return [FakeMesh(), FakeMesh(fail_hull=True)]

        def export(self, path: Path) -> None:
            path.write_bytes(b"mesh")

    class FakeScene:
        def __init__(self, geometries: dict[str, object]) -> None:
            self.geometry = geometries

    class FakePointCloud:
        def __init__(self, points: object) -> None:
            self.points = points

        @property
        def convex_hull(self) -> FakeMesh:
            return FakeMesh()

    load_results = [
        FakeScene({"a": FakeMesh()}),
        FakeScene({}),
        FakeMesh(),
        SimpleNamespace(vertices=[]),
        SimpleNamespace(vertices=[[0.0, 0.0, 0.0]]),
        SimpleNamespace(vertices=[]),
        SimpleNamespace(vertices=[[float(idx), float(idx % 5), 0.0] for idx in range(20)]),
    ]

    fake_trimesh = SimpleNamespace(
        Scene=FakeScene,
        Trimesh=FakeMesh,
        util=SimpleNamespace(concatenate=lambda geometries: "concatenated"),
        load=lambda *args, **kwargs: load_results.pop(0),
        voxel=SimpleNamespace(
            ops=SimpleNamespace(
                points_to_marching_cubes=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("mc failed"))
            )
        ),
        points=SimpleNamespace(PointCloud=FakePointCloud),
    )
    monkeypatch.setattr(ogs, "trimesh", fake_trimesh)

    assert ogs._load_mesh_or_points(entry=entry, index_path=index_path, capture_root=root) == (
        "concatenated",
        "source_scene_mesh",
    )
    assert ogs._load_mesh_or_points(entry=entry, index_path=index_path, capture_root=root) == (
        None,
        "empty_scene_mesh",
    )
    assert ogs._load_mesh_or_points(entry=entry, index_path=index_path, capture_root=root)[1] == "source_mesh"
    assert ogs._load_mesh_or_points(entry=entry, index_path=index_path, capture_root=root) == (
        None,
        "insufficient_point_cloud",
    )
    assert ogs._load_mesh_or_points(entry=entry, index_path=index_path, capture_root=root)[1] == "point_cloud_convex_hull"
    load_results.extend(
        [
            SimpleNamespace(vertices=[]),
            SimpleNamespace(vertices=[[float(idx), float(idx % 5), 0.0] for idx in range(20)]),
        ]
    )

    class FailingPointCloud:
        def __init__(self, points: object) -> None:
            self.points = points

        @property
        def convex_hull(self) -> object:
            raise RuntimeError("no hull")

    fake_trimesh.points = SimpleNamespace(PointCloud=FailingPointCloud)
    assert ogs._load_mesh_or_points(entry=entry, index_path=index_path, capture_root=root) == (
        None,
        "point_cloud_reconstruction_failed",
    )
    load_results.extend(
        [
            SimpleNamespace(vertices=[]),
            SimpleNamespace(vertices=[[float(idx), float(idx % 5), 0.0] for idx in range(20)]),
        ]
    )
    fake_trimesh.voxel = SimpleNamespace(
        ops=SimpleNamespace(points_to_marching_cubes=lambda *args, **kwargs: FakeMesh())
    )
    assert ogs._load_mesh_or_points(entry=entry, index_path=index_path, capture_root=root)[1] == "point_cloud_marching_cubes"

    assert ogs._mesh_components(SimpleNamespace(split=lambda only_watertight=False: (_ for _ in ()).throw(RuntimeError("split")))) == []
    assert ogs._kmeans2([[0.0, 0.0]] * 7) is None
    assert ogs._kmeans2([[0.0, 0.0]] * 8) is None
    clustered = ogs._kmeans2([[float(idx), 0.0] for idx in range(12)] + [[float(idx), 5.0] for idx in range(12)])
    assert clustered is not None
    monkeypatch.setattr(ogs.np, "argmin", lambda *args, **kwargs: ogs.np.zeros(8, dtype=int))
    assert ogs._kmeans2([[float(idx), 0.0] for idx in range(8)]) is None

    real_trimesh = ogs.trimesh
    monkeypatch.setattr(ogs, "trimesh", None)
    assert ogs._collision_hull_meshes(FakeMesh()) == ([], "geometry_unavailable")
    monkeypatch.setattr(ogs, "trimesh", fake_trimesh)

    hulls, method = ogs._collision_hull_meshes(FakeMesh())
    assert method == "component_convex_hulls"
    assert len(hulls) == 2
    fallback_hulls, fallback_method = ogs._collision_hull_meshes(
        SimpleNamespace(
            vertices=ogs.np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]),
            split=lambda only_watertight=False: [],
            convex_hull=SimpleNamespace(),
            bounding_box=SimpleNamespace(to_mesh=lambda: FakeMesh()),
        )
    )
    assert fallback_method == "single_convex_hull"
    assert fallback_hulls

    labels = ogs.np.asarray([0] * 10 + [1] * 10)
    monkeypatch.setattr(
        ogs,
        "_kmeans2",
        lambda points_xy: (labels, ogs.np.asarray([[0.0, 0.0], [10.0, 10.0]])),
    )
    fake_trimesh.points = SimpleNamespace(PointCloud=FakePointCloud)
    kmeans_hulls, kmeans_method = ogs._collision_hull_meshes(
        SimpleNamespace(
            vertices=ogs.np.asarray([[float(idx), float(idx % 4), 0.0] for idx in range(20)]),
            split=lambda only_watertight=False: [],
            convex_hull=SimpleNamespace(),
            bounding_box=SimpleNamespace(to_mesh=lambda: FakeMesh()),
        )
    )
    assert kmeans_method == "kmeans_convex_hulls"
    assert len(kmeans_hulls) == 2
    small_cluster_labels = ogs.np.asarray([0] * 4 + [1] * 16)
    monkeypatch.setattr(
        ogs,
        "_kmeans2",
        lambda points_xy: (small_cluster_labels, ogs.np.asarray([[0.0, 0.0], [10.0, 10.0]])),
    )
    fake_trimesh.points = SimpleNamespace(PointCloud=FakePointCloud)
    partial_hulls, partial_method = ogs._collision_hull_meshes(
        SimpleNamespace(
            vertices=ogs.np.asarray([[float(idx), float(idx % 4), 0.0] for idx in range(20)]),
            split=lambda only_watertight=False: [],
            convex_hull=SimpleNamespace(),
            bounding_box=SimpleNamespace(to_mesh=lambda: FakeMesh()),
        )
    )
    assert partial_method == "single_convex_hull"
    assert partial_hulls
    fake_trimesh.points = SimpleNamespace(PointCloud=FailingPointCloud)

    class BadSingleMesh:
        vertices = ogs.np.asarray([[float(idx), float(idx % 4), 0.0] for idx in range(20)])
        bounding_box = SimpleNamespace(to_mesh=lambda: FakeMesh())

        def split(self, only_watertight: bool = False) -> list[object]:
            return []

        @property
        def convex_hull(self) -> object:
            raise RuntimeError("bad hull")

    fallback_hulls, fallback_method = ogs._collision_hull_meshes(
        BadSingleMesh()
    )
    assert fallback_method == "bounding_box_hull"
    assert fallback_hulls
    monkeypatch.setattr(ogs, "trimesh", real_trimesh)


def test_2d_only_object_index_entries_use_bbox_proxy_meshes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _capture_root(tmp_path)
    _write_json(
        root / "raw" / "object_index.json",
        [
            {
                "id": "fridge-door",
                "label": "fridge door",
                "boundingBox": {"center": [0.0, 0.0, 1.0], "extents": [0.8, 0.1, 2.0]},
            },
            {
                "id": "fridge-handle",
                "label": "fridge handle",
                "boundingBox": {"center": [0.35, -0.08, 1.1], "extents": [0.05, 0.05, 0.8]},
            },
        ],
    )

    class FakeMesh:
        def __init__(self, extents: object = (0.2, 0.2, 0.2)) -> None:
            ex = ogs.np.asarray(extents, dtype=float)
            half = ex / 2.0
            self.vertices = ogs.np.asarray(
                [
                    [-half[0], -half[1], -half[2]],
                    [half[0], -half[1], -half[2]],
                    [half[0], half[1], -half[2]],
                    [-half[0], half[1], -half[2]],
                    [-half[0], -half[1], half[2]],
                    [half[0], -half[1], half[2]],
                    [half[0], half[1], half[2]],
                    [-half[0], half[1], half[2]],
                ],
                dtype=float,
            )
            self.bounds = ogs.np.asarray([[-half[0], -half[1], -half[2]], [half[0], half[1], half[2]]])
            self.faces = ogs.np.asarray([[4, 5, 6]], dtype=int)
            self.face_normals = ogs.np.asarray([[0.0, 0.0, 1.0]], dtype=float)
            self.triangles_center = ogs.np.asarray([[0.0, 0.0, half[2]]], dtype=float)
            self.area_faces = ogs.np.asarray([max(0.01, float(ex[0] * ex[1]))], dtype=float)
            self.bounding_box = SimpleNamespace(to_mesh=lambda: FakeMesh(extents))

        def copy(self) -> "FakeMesh":
            copied = FakeMesh((1.0, 1.0, 1.0))
            copied.vertices = self.vertices.copy()
            copied.bounds = self.bounds.copy()
            copied.faces = self.faces.copy()
            copied.face_normals = self.face_normals.copy()
            copied.triangles_center = self.triangles_center.copy()
            copied.area_faces = self.area_faces.copy()
            copied.bounding_box = SimpleNamespace(to_mesh=lambda: FakeMesh((1.0, 1.0, 1.0)))
            return copied

        @property
        def convex_hull(self) -> "FakeMesh":
            return self

        def split(self, only_watertight: bool = False) -> list["FakeMesh"]:
            return [self]

        def export(self, path: Path) -> None:
            path.write_bytes(b"mesh")

    fake_trimesh = SimpleNamespace(
        creation=SimpleNamespace(box=lambda extents: FakeMesh(extents)),
    )
    monkeypatch.setattr(ogs, "trimesh", fake_trimesh)

    result = ogs.run_object_geometry_stage(capture_root=root, provider_name="manual")
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    assert result["object_count"] == 2
    assert {item["object_id"] for item in manifest["objects"]} == {"fridge-door", "fridge-handle"}
    for item in manifest["objects"]:
        assert item["mesh_source"] == "bbox_proxy_mesh"
        assert item["grounding_level"] == "inferred"
        assert item["provenance"]["grounding_level"] == "inferred"


def test_object_geometry_surfaces_views_support_and_run_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bbox = {"center": [0.0, 0.0, 0.5], "extents": [1.0, 1.0, 1.0], "orientationQuaternion": [1.0, 0.0, 0.0, 0.0]}
    surfaces, method = ogs._support_surfaces(None, bbox)
    assert method == "bbox_fallback"
    assert surfaces[0]["method"] == "bbox_fallback"
    real_np = ogs.np
    monkeypatch.setattr(ogs, "np", None)
    assert ogs._surface_polygon([[0, 0, 0]])["bounds_xy"] == [0.0, 0.0, 0.0, 0.0]
    monkeypatch.setattr(ogs, "np", real_np)

    no_top_mesh = SimpleNamespace(
        faces=ogs.np.asarray([[0, 1, 2]], dtype=int),
        face_normals=ogs.np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        triangles_center=ogs.np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        area_faces=ogs.np.asarray([0.1], dtype=float),
    )
    assert ogs._support_surfaces(no_top_mesh, bbox)[1] == "bbox_fallback"

    views_dir = tmp_path / "views"
    crop = tmp_path / "crop.png"
    ogs._write_simple_png(crop, width=7, height=8)
    real_views = ogs._build_real_views(object_id="obj-1", crop_paths=[crop], views_dir=views_dir)
    assert real_views["candidates"][0]["prompt_box"] == [0, 0, 7, 8]

    target = {"object_id": "target", "placement_bbox": {"center": [0.0, 0.0, 1.0], "extents": [0.2, 0.2, 0.2]}}
    assert ogs._support_link_for_target(
        target=target,
        other_objects=[
            {"object_id": "target", "support_surfaces": []},
            {"object_id": "bad", "support_surfaces": ["skip", {"center_world": [0.0]}]},
            {"object_id": "far", "support_surfaces": [{"center_world": [5.0, 5.0, 0.9]}]},
        ],
    ) is None
    assert ogs._support_link_for_target(
        target=target,
        other_objects=[
            {"object_id": "shelf", "support_surfaces": [{"center_world": [0.05, 0.05, 0.9]}]},
        ],
    ) == "shelf"

    root = _capture_root(tmp_path / "run")
    _write_json(root / "raw" / "object_index.json", {"objects": [_object_entry("target"), _object_entry("fixture")]})
    _write_json(root / "pipeline" / "task_scope_record.json", {"target_object_ids": ["target"], "articulation_required_ids": ["fixture"]})
    result = ogs.run_object_geometry_stage(
        capture_root=root,
        ai_hint_runner=lambda payload: {"custom_hint": payload["object_id"]},
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["object_count"] if "object_count" in manifest else len(manifest["objects"]) == 2
    assert manifest["objects"][0]["ai_hints"]["source"] == "ai_runner"

    result = ogs.run_object_geometry_stage(
        capture_root=root,
        ai_hint_runner=lambda payload: (_ for _ in ()).throw(RuntimeError("hint failed")),
    )
    assert result["object_count"] == 2

    real_np = ogs.np
    monkeypatch.setattr(ogs, "np", None)
    with pytest.raises(PipelineError, match="numpy is required"):
        ogs.run_object_geometry_stage(capture_root=root)
    monkeypatch.setattr(ogs, "np", real_np)

    empty_root = _capture_root(tmp_path / "empty")
    with pytest.raises(PipelineError, match="requires an object index"):
        ogs.run_object_geometry_stage(capture_root=empty_root)

    monkeypatch.setattr(ogs, "run_object_geometry_stage", lambda **kwargs: {"manifest_path": "manifest.json", "object_count": 2})
    assert ogs.main(["--capture-root", str(root), "--provider", "manual"]) == 0
    assert "object_count=2" in capsys.readouterr().out
    monkeypatch.setattr(ogs, "run_object_geometry_stage", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert ogs.main(["--capture-root", str(root)]) == 1
    assert "FAILED" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["object_geometry_stage", "--capture-root", str(root)])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("blueprint_pipeline.object_geometry_stage", run_name="__main__")
    assert exc_info.value.code == 0
