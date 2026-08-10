from __future__ import annotations

from pathlib import Path

from pxr import Usd, UsdPhysics

from blueprint_pipeline.native_task_usd_import_normalization import (
    inspect_environment_import_usd,
    normalize_environment_import_usd,
)


def _asset(path: Path, *, with_scene: bool) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    stage.DefinePrim("/Asset/body", "Xform")
    if with_scene:
        UsdPhysics.Scene.Define(stage, "/Asset/PhysicsScene")
    stage.GetRootLayer().Save()
    return path


def test_embedded_standalone_scene_is_removed_from_derived_import_copy(
    tmp_path: Path,
) -> None:
    source = _asset(tmp_path / "source.usda", with_scene=True)
    original = source.read_bytes()
    destination = tmp_path / "packet" / "task.usda"

    receipt = normalize_environment_import_usd(
        source, destination, semantic_role="task_object"
    )

    assert source.read_bytes() == original
    assert receipt["source_bytes_mutated"] is False
    assert receipt["staged_bytes_derived"] is True
    assert receipt["removed_physics_scene_paths"] == ["/Asset/PhysicsScene"]
    assert receipt["active_embedded_physics_scene_paths_after"] == []
    assert inspect_environment_import_usd(
        destination, semantic_role="task_object"
    )["environment_import_scene_free"] is True


def test_scene_free_asset_remains_byte_exact(tmp_path: Path) -> None:
    source = _asset(tmp_path / "source.usda", with_scene=False)
    destination = tmp_path / "packet" / "task.usda"

    receipt = normalize_environment_import_usd(
        source, destination, semantic_role="task_object"
    )

    assert destination.read_bytes() == source.read_bytes()
    assert receipt["staged_bytes_derived"] is False
    assert receipt["source_and_staged_bytes_identical"] is True
    assert receipt["removed_physics_scene_paths"] == []
    assert receipt["deactivated_physics_scene_paths"] == []


def test_referenced_physics_scene_is_deactivated_without_mutating_dependency(
    tmp_path: Path,
) -> None:
    dependency = _asset(tmp_path / "dependency.usda", with_scene=True)
    dependency_bytes = dependency.read_bytes()
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(str(dependency), "/Asset")
    stage.GetRootLayer().Save()
    destination = tmp_path / "packet" / "task.usda"

    receipt = normalize_environment_import_usd(
        source, destination, semantic_role="task_object"
    )

    assert dependency.read_bytes() == dependency_bytes
    assert receipt["removed_physics_scene_paths"] == []
    assert receipt["deactivated_physics_scene_paths"] == [
        "/Asset/PhysicsScene"
    ]
    assert inspect_environment_import_usd(
        destination, semantic_role="task_object"
    )["environment_import_scene_free"] is True
