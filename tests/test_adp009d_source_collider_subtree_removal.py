from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pxr import Usd, UsdGeom

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.source_collider_subtree_removal import (
    SourceColliderSubtreeRemovalError,
    materialize_source_collider_batch_removal,
    remove_source_collider_subtree,
)


def _fixture(path: Path, scene_id: str) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    target = UsdGeom.Cube.Define(stage, f"/Root/target_{scene_id}")
    target.CreateSizeAttr(1.0)
    child = UsdGeom.Cube.Define(stage, f"/Root/target_{scene_id}/child")
    child.CreateSizeAttr(0.5)
    neighbor = UsdGeom.Cube.Define(stage, "/Root/protected_neighbor")
    neighbor.CreateSizeAttr(2.0)
    neighbor.AddTranslateOp().Set((1.0, 2.0, 3.0))
    stage.GetRootLayer().Save()
    return path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("scene_id", ["840313", "840796"])
def test_exact_collider_subtree_removal_preserves_unrelated_prims(
    tmp_path: Path, scene_id: str
) -> None:
    source = _fixture(tmp_path / f"{scene_id}.usda", scene_id)
    output = tmp_path / f"{scene_id}.removed.usda"

    receipt = remove_source_collider_subtree(
        source_usd_path=source,
        target_prim_path=f"/Root/target_{scene_id}",
        output_usda_path=output,
        expected_source_sha256=_sha256(source),
    )

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert not stage.GetPrimAtPath(f"/Root/target_{scene_id}").IsValid()
    neighbor = UsdGeom.Cube(stage.GetPrimAtPath("/Root/protected_neighbor"))
    assert neighbor.GetSizeAttr().Get() == 2.0
    assert receipt["removed_prim_count"] == 2
    assert receipt["remaining_target_collision_prim_count"] == 0
    assert receipt["unrelated_prim_inventory_unchanged"] is True
    assert receipt["source_bytes_unchanged"] is True
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_collider_removal_fails_closed_on_wrong_digest_or_missing_target(
    tmp_path: Path,
) -> None:
    source = _fixture(tmp_path / "scene.usda", "fixture")
    with pytest.raises(
        SourceColliderSubtreeRemovalError,
        match="source_collider_usd_digest_mismatch",
    ):
        remove_source_collider_subtree(
            source_usd_path=source,
            target_prim_path="/Root/target_fixture",
            output_usda_path=tmp_path / "wrong.usda",
            expected_source_sha256="sha256:" + "0" * 64,
        )
    with pytest.raises(
        SourceColliderSubtreeRemovalError,
        match="source_collider_target_prim_missing",
    ):
        remove_source_collider_subtree(
            source_usd_path=source,
            target_prim_path="/Root/missing",
            output_usda_path=tmp_path / "missing.usda",
        )


def test_batch_removal_materializes_independent_receipts_and_shared_scene(
    tmp_path: Path,
) -> None:
    source = _fixture(tmp_path / "dual_task.usda", "task_a")
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    task_b = UsdGeom.Cube.Define(stage, "/Root/target_task_b")
    task_b.CreateSizeAttr(0.25)
    stage.GetRootLayer().Save()

    receipt = materialize_source_collider_batch_removal(
        source_usd_path=source,
        targets=[
            {"removal_id": "task_a", "target_prim_path": "/Root/target_task_a"},
            {"removal_id": "task_b", "target_prim_path": "/Root/target_task_b"},
        ],
        output_root=tmp_path / "batch",
        expected_source_sha256=_sha256(source),
    )

    assert receipt["target_count"] == 2
    assert receipt["unrelated_prim_inventory_unchanged"] is True
    assert receipt["source_bytes_unchanged"] is True
    assert receipt["independent_receipts_share_exact_source_digest"] is True
    assert receipt["independent_removed_scenes_are_distinct"] is True
    assert len({row["receipt_digest"] for row in receipt["target_removals"]}) == 2
    shared = Usd.Stage.Open(
        str(tmp_path / "batch" / "scene_without_source_colliders.usda")
    )
    assert shared is not None
    assert not shared.GetPrimAtPath("/Root/target_task_a").IsValid()
    assert not shared.GetPrimAtPath("/Root/target_task_b").IsValid()
    assert shared.GetPrimAtPath("/Root/protected_neighbor").IsValid()
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_batch_removal_rejects_duplicate_or_nested_targets(tmp_path: Path) -> None:
    source = _fixture(tmp_path / "scene.usda", "task")
    with pytest.raises(
        SourceColliderSubtreeRemovalError,
        match="source_collider_batch_target_paths_duplicate",
    ):
        materialize_source_collider_batch_removal(
            source_usd_path=source,
            targets=[
                {"removal_id": "a", "target_prim_path": "/Root/target_task"},
                {"removal_id": "b", "target_prim_path": "/Root/target_task"},
            ],
            output_root=tmp_path / "duplicate",
        )
